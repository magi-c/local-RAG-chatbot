from dataclasses import dataclass
import hashlib
import re
from .loader import Page

@dataclass
class Chunk: 
    id: str
    text: str
    source: str
    page: int
    chunk_index: int

def split_sentences(text: str) -> list[str]: 
    # Paragraphs first, then sentences. Simple, but good enough to start.
    parts = [] 
    for para in text.split("\n\n"): 
        para = para.replace("\n", " ").strip() 
        if para: 
            parts.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", para) 
                         if s.strip()) 
            return parts

# TODO compare with RecursiveCharacterTextSplitter from langchain-text-splitters
def chunk_page(page: Page, chunk_size: int = 1000, overlap: int = 200) -> list[Chunk]:
    sentences = split_sentences(page.text) 
    chunks, current = [], [] 
    def flush(): 
        text = " ".join(current).strip() 
        if text: 
            idx = len(chunks) 
            uid = hashlib.md5(f"{page.source}|{page.page}|{idx}|{text}".encode()).hexdigest()[:16] 
            chunks.append(Chunk(uid, text, page.source, page.page, idx)) 
    for sent in sentences: 
        # a very long "sentence" (e.g. a table) is hard-split 
        while len(sent) > chunk_size: 
            current.append(sent[:chunk_size]) 
            flush() 
            current = [] 
            sent = sent[chunk_size - overlap:] 
        if sum(len(s) + 1 for s in current) + len(sent) > chunk_size and current: 
            flush() 
            # overlap: carry the last sentences (up to `overlap` chars) into the next chunk 
            tail, size = [], 0 
            for s in reversed(current): 
                if size + len(s) > overlap: 
                    break 
                tail.insert(0, s) 
                size += len(s) + 1 
            current = tail 
        current.append(sent) 
    flush()
    return chunks

def chunk_pages(pages: list[Page], chunk_size: int = 1000, overlap: int = 200) -> list[Chunk]: 
    out = [] 
    for p in pages: 
        out.extend(chunk_page(p, chunk_size, overlap)) 
    return out