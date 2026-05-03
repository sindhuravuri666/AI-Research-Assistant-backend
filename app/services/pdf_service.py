import uuid
from pathlib import Path
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.utils.pdf_utils import extract_pdf_pages
from app.core.config import UPLOAD_DIR
from app.models.schemas import UploadResponse


class PDFService:
    def __init__(self):
        self.upload_dir = Path(UPLOAD_DIR)

    def save_pdf(self, file_name: str, contents: bytes) -> str:
        file_name = Path(file_name).name
        target = self.upload_dir / file_name
        with open(target, "wb") as handle:
            handle.write(contents)
        return str(target)

    def parse_pdf(self, file_path: str) -> List[Dict]:
        pages = extract_pdf_pages(file_path)
        return pages

    def chunk_text(self, paper_id: str, file_name: str, pages: List[Dict]) -> List[Dict]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=140,
            separators=["\n\n", "\n", ". ", " "]
        )
        output = []
        chunk_index = 0
        for page in pages:
            raw_text = page["text"]
            fragments = splitter.split_text(raw_text)
            for fragment in fragments:
                output.append({
                    "paper_id": paper_id,
                    "file_name": file_name,
                    "page_number": page["page_number"],
                    "chunk_index": chunk_index,
                    "text": fragment,
                    "pages": len(pages),
                })
                chunk_index += 1
        return output

    def build_upload_response(self, paper_id: str, file_name: str, pages: int, chunks: int) -> UploadResponse:
        return UploadResponse(
            paper_id=paper_id,
            file_name=file_name,
            pages=pages,
            chunks=chunks,
            message="Paper uploaded and processed successfully."
        )
