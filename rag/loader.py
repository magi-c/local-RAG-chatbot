from dataclasses import dataclass
from pathlib import Path
import re
from pypdf import PdfReader

@dataclass
class Page:
    source: str  # file name
    page: int  # 1-based, as a human sees it
    text: str

def clean(text: str) -> str:
    text = text.replace("\u00ad", "")  # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # "process-\ning" -> "processing"
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def load_pdf(path: str | Path) -> list[Page]:
    path = Path(path)
    reader = PdfReader(path)
    pages = []
    for i, p in enumerate(reader.pages, start=1):
        text = clean(p.extract_text() or "")
        if text:
            # skip empty pages (e.g. scans without OCR)
            pages.append(Page(source=path.name, page=i, text=text))
    return pages