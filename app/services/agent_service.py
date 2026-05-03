from typing import List, Dict
from app.services.retrieval_service import RetrievalService
from app.services.pdf_service import PDFService
from app.utils.ollama_client import OllamaClient
from app.models.schemas import ChatAnswer, Citation, CompareAnswer


class AgentService:
    def __init__(self, retrieval_service: RetrievalService, pdf_service: PDFService, ollama_client: OllamaClient):
        self.retrieval = retrieval_service
        self.pdf_service = pdf_service
        self.ollama = ollama_client

    def _build_context(self, citations: List[Citation]) -> str:
        context_sections = []
        for citation in citations:
            context_sections.append(
                f"[{citation.file_name} - page {citation.page_number}] {citation.snippet}"
            )
        return "\n\n".join(context_sections)

    def _format_citation_list(self, citations: List[Citation]) -> str:
        lines = []
        for citation in citations:
            lines.append(f"- {citation.file_name} (page {citation.page_number}): {citation.snippet}")
        return "\n".join(lines)

    def answer_question(self, question: str, explain_like_im_five: bool = False) -> ChatAnswer:
        citations = self.retrieval.retrieve_citations(question)
        if not citations:
            return ChatAnswer(
                answer="I could not find enough evidence in the uploaded documents to answer that question.",
                citations=[],
                source_papers=[]
            )
        context = self._build_context(citations)
        prompt = (
            "You are a local research assistant. Answer the user question using only the provided document context. "
            "If the answer cannot be found in the context, say so clearly and do not hallucinate.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
        )
        if explain_like_im_five:
            prompt += "Explain the answer as if the reader is five years old.\n"
        prompt += (
            "Provide a concise answer, then list sources by paper name and page number. "
            "Use citations and make it clear which snippets were used."
        )
        answer_text = self.ollama.generate(prompt, temperature=0.1, max_tokens=400)
        source_papers = sorted({c.file_name for c in citations})
        return ChatAnswer(answer=answer_text.strip(), citations=citations, source_papers=source_papers)

    def generate_breakdown(self, paper_id: str, paper_name: str) -> Dict:
        chunks = self.retrieval.vector_store.get_paper_chunks(paper_id)
        if not chunks:
            return {}
        prompt = (
            "You are an academic summarizer. Use the text fragments below to generate a detailed paper breakdown. "
            "Organize your response into: TL;DR summary, Key contributions, Method explained simply, Dataset or experiment details, Limitations, Future work, Important keywords.\n\n"
            "Fragments:\n"
        )
        prompt += "\n\n".join([f"[{chunk['file_name']} page {chunk['page_number']}]: {chunk['text']}" for chunk in chunks[:8]])
        prompt += f"\n\nPaper title: {paper_name}\n"
        answer = self.ollama.generate(prompt, temperature=0.2, max_tokens=600)
        return {"breakdown": answer.strip()}

    def compare_papers(self, paper_ids: List[str], paper_map: Dict[str, str]) -> CompareAnswer:
        fragments = []
        for pid in paper_ids:
            paper_name = paper_map.get(pid, pid)
            chunks = self.retrieval.vector_store.get_paper_chunks(pid)[:6]
            for chunk in chunks:
                fragments.append(f"Paper: {paper_name} page {chunk['page_number']} - {chunk['text']}")
        prompt = (
            "You are a comparison analyst. Compare the selected research papers across: research problem, methodology, dataset, strengths, weaknesses, limitations, future scope, and best use case. "
            "Use only the provided text fragments and cite paper names.\n\n"
            "Fragments:\n"
            + "\n\n".join(fragments)
        )
        result = self.ollama.generate(prompt, temperature=0.2, max_tokens=600)
        citations = self.retrieval.retrieve_citations('compare papers ' + ' '.join(paper_ids))
        return CompareAnswer(comparison=result.strip(), citations=citations)

    def generate_notes(self, paper_ids: List[str], mode: str = "bullet") -> str:
        paper_titles = ", ".join(paper_ids)
        prompt = (
            f"You are a study assistant. Generate {mode} notes for the uploaded papers: {paper_titles}. "
            "Focus on key definitions, core ideas, and exam-style questions if requested. "
            "Use the text fragments below. Do not invent unsupported information.\n\n"
        )
        fragments = []
        for pid in paper_ids:
            chunks = self.retrieval.vector_store.get_paper_chunks(pid)[:5]
            for chunk in chunks:
                fragments.append(f"Paper {pid} page {chunk['page_number']}: {chunk['text']}")
        prompt += "\n\n".join(fragments)
        return self.ollama.generate(prompt, temperature=0.2, max_tokens=500).strip()

    def generate_ideas(self, paper_ids: List[str]) -> str:
        prompt = (
            "You are a research idea generator. Analyze the uploaded papers and identify research gaps, possible improvements, new project ideas, and future paper topics. "
            "Use only available document evidence and clearly label each idea category.\n\n"
        )
        fragments = []
        for pid in paper_ids:
            chunks = self.retrieval.vector_store.get_paper_chunks(pid)[:5]
            for chunk in chunks:
                fragments.append(f"Paper {pid} page {chunk['page_number']}: {chunk['text']}")
        prompt += "\n\n".join(fragments)
        return self.ollama.generate(prompt, temperature=0.2, max_tokens=500).strip()
