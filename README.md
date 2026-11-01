# 📘 IIT Madras Handbook Chatbot (vector RAG)

Ask questions about the student handbook and get answers **with page citations**,
built from scratch with embeddings + a vector database + an LLM.

```
                 ingest.py (run once)                          rag.py (every question)
PDF ──► pages ──► chunks ──► embeddings ──► ChromaDB ◄── embed question ── "Can I skip classes?"
                                              │
                                     top-5 similar chunks
                                              ▼
                               LLM (Groq / Llama 3.1) + strict prompt ──► answer (p. 12)
```

## ▶️ Run it

```bash
cd ~/Code/handbook-rag
source venv/bin/activate
pip install -r requirements.txt

# 1. Put the handbook PDF here:   data/handbook.pdf
# 2. Free LLM key: https://console.groq.com/keys
cp .env.example .env              # then paste your key into .env

python ingest.py                  # builds the vector DB (run again if the PDF changes)
python chat.py                    # chat in the terminal
streamlit run app.py              # chat in the browser
```

No API key yet? Everything still runs. The bot shows the most relevant handbook
passages instead of writing an answer.

## 🗂️ What each file does → which lesson it uses

| File | Job | Lesson |
|---|---|---|
| `config.py` | Every setting (chunk size, top-k, model names, threshold) | all |
| `ingest.py` | PDF → clean text → chunks → embeddings → ChromaDB | 1, 2, 3, 4 |
| `rag.py` | Retrieve similar chunks + ask the LLM (the "brain") | 2, 5 |
| `chat.py` | Terminal chatbot (shows retrieved chunks + scores) | 6 |
| `app.py` | Streamlit web chatbot with settings sliders | 6 |
| `evaluate.py` | Measures whether retrieval finds the right pages | 6 |
| `chunks.json` | Created by ingest: every chunk, readable. **Open it!** | 3 |
| `db/` | Created by ingest: the saved vector database | 4 |

## 🧠 Key ideas, where they live in the code

- **Embeddings** (`ingest.py → build_index`, `rag.py → retrieve`): text → 384 numbers;
  similar meaning = similar direction.
- **Normalize + cosine** (`normalize_embeddings=True`, `"hnsw:space": "cosine"`):
  the same maths as your `fast_search` in Lesson 2. Chroma returns
  `distance = 1 − similarity`, so `rag.py` converts it back to a score.
- **Query prefix** (`config.QUERY_PREFIX`): bge models want questions marked, which fixes the
  "question vs statement" style gap you saw in Lesson 1.
- **Chunking with overlap** (`ingest.py → chunk_text`): ~800 chars, cut at sentence ends,
  150-char overlap, tiny leftovers merged. This is the improved chunker from Lesson 3 homework Q4.
- **Two hallucination guards** (`rag.py → answer`):
  1. `MIN_SCORE`: if even the best chunk is a weak match, say "not found" *without* calling the LLM.
  2. `SYSTEM_PROMPT`: the LLM may only use the excerpts and must cite pages.
- **Load once, answer many** (`HandbookRAG.__init__`, `@st.cache_resource`): the model loads once,
  not once per question.
- **Conversation memory** (`HISTORY_TURNS`): the last 2 Q&A pairs go to the LLM so follow-ups
  like "and what about fees?" work.

## 🔬 Experiments to learn from (do them in order)

1. **Read `chunks.json`.** Are any chunks garbled (tables)? Do rules get cut in half?
2. **Watch the scores** in `chat.py`. Ask a good question and a nonsense one ("Is there a
   swimming pool?"). Where should `MIN_SCORE` sit to separate them?
3. **Fill `test_questions.json`** with real questions + correct pages, then run `python evaluate.py`.
4. **Tune with evidence:** change `CHUNK_SIZE` (300 / 800 / 1500) → `python ingest.py` →
   `python evaluate.py`. Which wins?
5. **Swap the model:** set `EMBED_MODEL = "all-MiniLM-L6-v2"` and `QUERY_PREFIX = ""`,
   re-ingest, and compare the hit rate.
6. **Break the prompt:** delete the "ONLY the handbook excerpts" rule in `rag.py` and ask
   something not in the handbook. Watch it make things up.
