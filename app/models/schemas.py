from typing import List, Optional
from pydantic import BaseModel, Field

class UploadResponse(BaseModel):
    paper_id: str
    file_name: str
    pages: int
    chunks: int
    message: str

class Citation(BaseModel):
    paper_id: str
    file_name: str
    page_number: int
    chunk_index: int
    snippet: str

class ChatRequest(BaseModel):
    question: str
    explain_like_im_five: Optional[bool] = False

class ChatAnswer(BaseModel):
    answer: str
    citations: List[Citation]
    source_papers: List[str]

class PaperMeta(BaseModel):
    paper_id: str
    file_name: str
    pages: int
    chunks: int

class PaperDetail(PaperMeta):
    breakdown: Optional[dict] = None

class CompareRequest(BaseModel):
    paper_ids: List[str]

class CompareAnswer(BaseModel):
    comparison: str
    citations: List[Citation]

class NotesRequest(BaseModel):
    paper_ids: List[str]
    mode: Optional[str] = Field("bullet", description="bullet, flashcards, questions, definitions")

class IdeaRequest(BaseModel):
    paper_ids: Optional[List[str]] = None

class ToolResponse(BaseModel):
    summary: str
    citations: List[Citation]
