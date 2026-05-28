from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from uuid import uuid4

from app.core.config import OLLAMA_HOST, OLLAMA_LLM_MODEL, OLLAMA_EMBED_MODEL
from app.models.schemas import (
    UploadResponse,
    PaperMeta,
    PaperDetail,
    ChatRequest,
    ChatAnswer,
    CompareRequest,
    CompareAnswer,
    NotesRequest,
    IdeaRequest,
)
from app.services.pdf_service import PDFService
from app.storage.vector_store import LocalFAISSStore
from app.utils.ollama_client import OllamaClient, OllamaClientError
from app.services.embedding_service import EmbeddingService
from app.services.retrieval_service import RetrievalService
from app.services.agent_service import AgentService

router = APIRouter()

pdf_service = PDFService()
vector_store = LocalFAISSStore()
ollama_client = OllamaClient()
embedding_service = EmbeddingService(ollama_client)
retrieval_service = RetrievalService(vector_store, embedding_service)
agent_service = AgentService(retrieval_service, pdf_service, ollama_client)

ALLOWED_EXTENSIONS = {".pdf"}


def validate_pdf(filename: str) -> None:
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid file name.")

    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if f".{suffix}" not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only PDF uploads are allowed."
        )


def process_pdf_upload(upload: UploadFile, data: bytes):
    validate_pdf(upload.filename)

    if not data:
        raise HTTPException(
            status_code=400,
            detail=f"{upload.filename} is empty."
        )

    paper_id = upload.filename.replace(" ", "_")
    paper_id = f"{paper_id}-{uuid4().hex[:8]}"

    saved_path = pdf_service.save_pdf(upload.filename, data)
    pages = pdf_service.parse_pdf(saved_path)

    if not pages:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded PDF {upload.filename} has no extractable text."
        )

    chunks = pdf_service.chunk_text(paper_id, upload.filename, pages)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail=f"No chunks were created for {upload.filename}."
        )

    embeddings = embedding_service.embed_chunks(
        [chunk["text"] for chunk in chunks]
    )

    vector_store.add_embeddings(embeddings, chunks)

    return pdf_service.build_upload_response(
        paper_id,
        upload.filename,
        len(pages),
        len(chunks),
    )


@router.get("/health")
def health_check():
    ollama_running = ollama_client.check_health()
    
    # Try to test the embedding endpoint
    embedding_test = False
    embedding_error = None
    try:
        test_embeddings = ollama_client.embed_texts(["test"])
        embedding_test = len(test_embeddings) > 0
    except Exception as e:
        embedding_error = str(e)

    return {
        "status": "ok" if ollama_running and embedding_test else "warning",
        "details": {
            "ollama_host": OLLAMA_HOST,
            "llm_model": OLLAMA_LLM_MODEL,
            "embed_model": OLLAMA_EMBED_MODEL,
            "ollama_running": ollama_running,
            "embedding_working": embedding_test,
            "embedding_error": embedding_error,
            "faiss_index_exists": vector_store.index.ntotal > 0,
        },
    }


@router.post("/upload", response_model=UploadResponse)
async def upload_paper(file: UploadFile = File(...)):
    try:
        data = await file.read()
        return process_pdf_upload(file, data)

    except OllamaClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload paper: {str(exc)}"
        )


@router.post("/upload-multiple", response_model=List[UploadResponse])
async def upload_multiple_papers(files: List[UploadFile] = File(...)):
    responses = []

    try:
        for file in files:
            data = await file.read()
            response = process_pdf_upload(file, data)
            responses.append(response)

        return responses

    except OllamaClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload papers: {str(exc)}"
        )


@router.get("/papers", response_model=List[PaperMeta])
def get_papers():
    return vector_store.get_all_papers()


@router.get("/papers/{paper_id}", response_model=PaperDetail)
def get_paper(paper_id: str):
    papers = vector_store.get_all_papers()
    paper = next((item for item in papers if item["paper_id"] == paper_id), None)

    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")

    try:
        breakdown = agent_service.generate_breakdown(
            paper_id,
            paper["file_name"]
        )
        return PaperDetail(**paper, breakdown=breakdown)

    except OllamaClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.post("/chat", response_model=ChatAnswer)
def chat(request: ChatRequest):
    if vector_store.index.ntotal == 0:
        raise HTTPException(
            status_code=400,
            detail="No papers uploaded yet. Please upload a PDF first."
        )

    try:
        return agent_service.answer_question(
            request.question,
            request.explain_like_im_five
        )

    except OllamaClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.post("/compare", response_model=CompareAnswer)
def compare(request: CompareRequest):
    papers = vector_store.get_all_papers()

    selected = [
        paper for paper in papers
        if paper["paper_id"] in request.paper_ids
    ]

    if len(selected) < 2:
        raise HTTPException(
            status_code=400,
            detail="Select at least two uploaded papers to compare."
        )

    try:
        paper_map = {
            paper["paper_id"]: paper["file_name"]
            for paper in selected
        }

        return agent_service.compare_papers(request.paper_ids, paper_map)

    except OllamaClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.post("/notes")
def notes(request: NotesRequest):
    if not request.paper_ids:
        raise HTTPException(status_code=400, detail="No paper IDs provided.")

    try:
        content = agent_service.generate_notes(request.paper_ids, request.mode)
        return {"notes": content}

    except OllamaClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.post("/ideas")
def ideas(request: IdeaRequest):
    all_papers = vector_store.get_all_papers()

    paper_ids = request.paper_ids or [
        paper["paper_id"] for paper in all_papers
    ]

    if not paper_ids:
        raise HTTPException(
            status_code=400,
            detail="No papers are uploaded yet."
        )

    try:
        content = agent_service.generate_ideas(paper_ids)
        return {"ideas": content}

    except OllamaClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc))