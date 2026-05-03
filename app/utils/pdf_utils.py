import fitz
from typing import List, Dict


def normalize_text(text: str) -> str:
    cleaned = " ".join(text.replace("\n", " ").split())
    return cleaned.strip()


def extract_pdf_pages(file_path: str) -> List[Dict]:
    document = fitz.open(file_path)
    pages = []
    for page_number in range(len(document)):
        page = document.load_page(page_number)
        raw_text = page.get_text("text")
        cleaned_text = normalize_text(raw_text)
        if cleaned_text:
            pages.append({
                "page_number": page_number + 1,
                "text": cleaned_text,
            })
    document.close()
    return pages
