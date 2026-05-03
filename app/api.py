from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
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
    suffix = filename.lower().rsplit('.', 1)[-1] if '.' in filename else ''
    if f".{suffix}" not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF uploads are allowed.")


@router.get("/health")
def health_check():
    ollama_running = ollama_client.check_health()
    status = {
        "status": "ok" if ollama_running else "warning",
        "details": {
            "ollama_host": OLLAMA_HOST,
            "llm_model": OLLAMA_LLM_MODEL,
            "embed_model": OLLAMA_EMBED_MODEL,
            "faiss_index_exists": vector_store.index.ntotal > 0,
        },
    }
    return status


@router.post("/upload", response_model=List[UploadResponse])
async def upload_papers(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No PDF uploaded.")
    responses = []
    for upload in files:
        validate_pdf(upload.filename)
        data = await upload.read()
        paper_id = upload.filename.replace(" ", "_")
        paper_id = f"{paper_id}-{uuid4().hex[:8]}"
        saved_path = pdf_service.save_pdf(upload.filename, data)
        pages = pdf_service.parse_pdf(saved_path)
        if not pages:
            raise HTTPException(status_code=400, detail=f"Uploaded PDF {upload.filename} has no extractable text.")
        chunks = pdf_service.chunk_text(paper_id, upload.filename, pages)
        embeddings = embedding_service.embed_chunks([chunk["text"] for chunk in chunks])
        vector_store.add_embeddings(embeddings, chunks)
        responses.append(pdf_service.build_upload_response(paper_id, upload.filename, len(pages), len(chunks)))
    return responses


@router.get("/papers", response_model=List[PaperMeta])
def get_papers():
    return vector_store.get_all_papers()


@router.get("/papers/{paper_id}", response_model=PaperDetail)
def get_paper(paper_id: str):
    papers = vector_store.get_all_papers()
    paper = next((item for item in papers if item["paper_id"] == paper_id), None)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")
    breakdown = agent_service.generate_breakdown(paper_id, paper["file_name"])
    return PaperDetail(**paper, breakdown=breakdown)


@router.post("/chat", response_model=ChatAnswer)
def chat(request: ChatRequest):
    try:
        return agent_service.answer_question(request.question, request.explain_like_im_five)
    except OllamaClientError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.post("/compare", response_model=CompareAnswer)
def compare(request: CompareRequest):
    papers = vector_store.get_all_papers()
    selected = [paper for paper in papers if paper["paper_id"] in request.paper_ids]
    if len(selected) < 2:
        raise HTTPException(status_code=400, detail="Select at least two papers to compare.")
    paper_map = {paper["paper_id"]: paper["file_name"] for paper in selected}
    return agent_service.compare_papers(request.paper_ids, paper_map)


@router.post("/notes")
def notes(request: NotesRequest):
    if not request.paper_ids:
        raise HTTPException(status_code=400, detail="No paper IDs provided.")
    content = agent_service.generate_notes(request.paper_ids, request.mode)
    return {"notes": content}


@router.post("/ideas")
def ideas(request: IdeaRequest):
    all_papers = vector_store.get_all_papers()
    paper_ids = request.paper_ids or [paper["paper_id"] for paper in all_papers]
    if not paper_ids:
        raise HTTPException(status_code=400, detail="No papers are uploaded yet.")
    content = agent_service.generate_ideas(paper_ids)
    return {"ideas": content}
