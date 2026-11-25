import sys
from pathlib import Path

from rag.loader import load_pdf
from rag.chunker import chunk_pages
from rag.store import add_chunks

for arg in sys.argv[1:]:
    for pdf in sorted(Path(arg).glob("*.pdf")) if Path(arg).is_dir() else [Path(arg)]:
        pages = load_pdf(pdf)
        chunks = chunk_pages(pages)
        add_chunks(chunks)
        print(f"{pdf.name}: {len(pages)} pages, {len(chunks)} chunks")