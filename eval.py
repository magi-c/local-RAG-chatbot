import json

from rag.store import search

cases = json.load(open("eval_set.json", encoding="utf-8"))
answerable = [c for c in cases if c.get("page") is not None]
off_topic = [c for c in cases if c.get("page") is None]

# 1. Retrieval quality: is the right page among the top-k chunks?
print("== Retrieval ==")
for k in (1, 3, 5):
    found = sum(
        any(h["metadata"]["page"] == c["page"] for h in search(c["question"], k=k))
        for c in answerable
    )
    print(f"hit rate@{k}: {found / len(answerable):.0%}")


# 2. Threshold calibration: which MIN_SCORE best separates answerable from off-topic questions?
def top_score(question: str) -> float:
    hits = search(question, k=1)
    return hits[0]["score"] if hits else 0.0


pos = [top_score(c["question"]) for c in answerable]
neg = [top_score(c["question"]) for c in off_topic]

print("\n== Top-1 scores ==")
for score, c in sorted(zip(pos + neg, answerable + off_topic), key=lambda x: -x[0]):
    label = "answerable" if c.get("page") is not None else "OFF-TOPIC "
    print(f"{score:.3f}  {label}  {c['question']}")

if neg:
    def accuracy(t: float) -> float:
        kept = sum(s >= t for s in pos)        # answerable questions that still get chunks
        rejected = sum(s < t for s in neg)     # off-topic questions correctly cut off
        return (kept + rejected) / (len(pos) + len(neg))

    # candidate thresholds: midpoints between neighbouring scores
    scores = sorted(set(pos + neg))
    candidates = [scores[0]] + [(a + b) / 2 for a, b in zip(scores, scores[1:])]
    best = max(candidates, key=lambda t: (accuracy(t), -t))   # tie -> lower threshold, fewer missed answers

    print("\n== Threshold ==")
    print(f"answerable: min {min(pos):.3f}, max {max(pos):.3f}")
    print(f"off-topic:  min {min(neg):.3f}, max {max(neg):.3f}")
    print(f"suggested MIN_SCORE = {best:.2f}  (correct decision for {accuracy(best):.0%} of questions)")
    if min(pos) <= max(neg):
        print("Warning: the groups overlap, so no threshold separates them perfectly. "
              "Prefer a slightly lower value and let the prompt handle the rest.")
else:
    print("\nAdd off-topic questions (\"page\": null) to eval_set.json to calibrate MIN_SCORE.")