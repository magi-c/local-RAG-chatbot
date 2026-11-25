import numpy as np
from ollama import Client
from .config import EMBED_MODEL, OLLAMA_HOST

client = Client(host=OLLAMA_HOST)

# nomic-embed-text expects task prefixes:
# # "search_document: " for chunks, "search_query: " for questions.
DOC_PREFIX, QUERY_PREFIX = "search_document: ", "search_query: "

def _embed(texts: list[str]) -> list[list[float]]: 
    vecs = np.array(client.embed(model=EMBED_MODEL, input=texts).embeddings, dtype="float32") 
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) # unit length -> dot product = cosine
    return vecs.tolist()

def embed_passages(texts: list[str], batch_size: int = 32) -> list[list[float]]: 
    out = [] 
    for i in range(0, len(texts), batch_size): 
        out.extend(_embed([DOC_PREFIX + t for t in texts[i:i + batch_size]])) 
    return out

def embed_query(text: str) -> list[float]: 
    return _embed([QUERY_PREFIX + text])[0]
