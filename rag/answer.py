import re

from ollama import Client

from .config import CHAT_MODEL, OLLAMA_HOST

client = Client(host=OLLAMA_HOST)

SYSTEM = """You answer questions using ONLY the numbered document fragments provided.
Rules:
- End every sentence that states a fact from the document with its fragment number in square brackets, e.g. [2] or [1][3].
- Only use fragment numbers you were given.
- If the fragments do not contain the answer, reply exactly: "I couldn't find that in the document." Do not guess.
- Answer in the language of the question, concisely."""


def format_context(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        m = h["metadata"]
        blocks.append(f'<fragment id="{i}" file="{m["source"]}" page="{m["page"]}">\n{h["text"]}\n</fragment>')
    return "\n\n".join(blocks)


def cited_sources(answer: str, hits: list[dict]) -> list[dict]:
    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    return [
        {"n": n, "source": hits[n - 1]["metadata"]["source"], "page": hits[n - 1]["metadata"]["page"],
         "text": hits[n - 1]["text"]}
        for n in used if 1 <= n <= len(hits)          # drop numbers the model could not have seen
    ]


def build_messages(question: str, hits: list[dict]) -> list[dict]:
    prompt = f"<document>\n{format_context(hits)}\n</document>\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


def generate_answer(question: str, hits: list[dict], model: str = CHAT_MODEL) -> dict:
    if not hits:
        return {"answer": "I couldn't find that in the document.", "sources": []}
    response = client.chat(
        model=model,
        messages=build_messages(question, hits),
        options={"temperature": 0, "num_ctx": 8192},  # deterministic; room for 5+ chunks
    )
    text = response.message.content
    return {"answer": text, "sources": cited_sources(text, hits)}