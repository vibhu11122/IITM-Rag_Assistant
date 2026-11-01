"""
evaluate.py - is retrieval actually finding the right pages?

1. Open test_questions.json and write real questions + the page(s)
   in the handbook where the answer is (check the PDF yourself).
2. Run:  python evaluate.py

It reports "hit rate": how often a correct page appears in the top-k.
Change CHUNK_SIZE / TOP_K / EMBED_MODEL in config.py, re-run ingest.py,
and compare - now you're tuning with evidence instead of guessing.
"""
import json

import config
from rag import HandbookRAG

with open(config.BASE_DIR / "test_questions.json", encoding="utf-8") as f:
    tests = [t for t in json.load(f) if t.get("pages")]

if not tests:
    raise SystemExit("Add questions with their correct pages to test_questions.json")

rag = HandbookRAG()
hits = 0
for t in tests:
    results = rag.retrieve(t["question"], config.TOP_K)
    found_pages = [r["page"] for r in results]
    ok = any(p in found_pages for p in t["pages"])
    hits += ok
    print(f"{'✅' if ok else '❌'} {t['question']}\n"
          f"   expected {t['pages']} | got {found_pages} | "
          f"best score {results[0]['score']:.3f}")

print(f"\nHit rate @ top-{config.TOP_K}: {hits}/{len(tests)} "
      f"= {hits / len(tests):.0%}")
