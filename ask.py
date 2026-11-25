import sys

from rag.store import search
from rag.answer import generate_answer

MIN_SCORE = 0.5   # starting point: calibrate on your own questions (step 7)

question = " ".join(sys.argv[1:])
hits = [h for h in search(question, k=5) if h["score"] >= MIN_SCORE]
result = generate_answer(question, hits)

print(result["answer"], "\n\nSources:")
for s in result["sources"]:
    print(f'  [{s["n"]}] {s["source"]}, p. {s["page"]}: "{s["text"][:120]}..."')