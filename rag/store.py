import chromadb
from .chunker import Chunk
from .config import EMBED_MODEL

from .embedder import embed_passages, embed_query

_client = chromadb.PersistentClient(path="chroma_db")


def get_collection():
    # one collection per embedding model: vectors from different models must never mix
    name = "docs_" + EMBED_MODEL.replace(":", "_").replace("/", "_")
    return _client.get_or_create_collection(
        name=name,
        configuration={"hnsw": {"space": "cosine"}},   # distance = 1 - cosine similarity
    )


def add_chunks(chunks: list[Chunk], batch_size: int = 64) -> None:
    col = get_collection()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        col.upsert(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            embeddings=embed_passages([c.text for c in batch]),
            metadatas=[{"source": c.source, "page": c.page, "chunk_index": c.chunk_index} for c in batch],
        )


def search(question: str, k: int = 5, source: str | None = None) -> list[dict]:
    res = get_collection().query(
        query_embeddings=[embed_query(question)],
        n_results=k,
        where={"source": source} if source else None,     # optional: search one file only
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": doc, "metadata": meta, "score": 1 - dist}
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]