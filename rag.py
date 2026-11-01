"""
rag.py - STEP 2: the brain. Used by both chat.py and app.py.

    question -> embed -> find similar chunks (RETRIEVAL)
             -> send chunks + question to the LLM (GENERATION) -> answer
"""
import chromadb
from sentence_transformers import SentenceTransformer

import config

SYSTEM_PROMPT = """You are a helpful assistant for students of the IIT Madras BS degree.
Answer the question using ONLY the handbook excerpts provided.
Rules:
- Cite the page number for every fact, like (p. 12).
- If the excerpts do not contain the answer, reply exactly:
  "I couldn't find this in the handbook." and nothing else.
- Never guess rules, numbers, dates or fees.
- Be concise. Use bullet points for multi-part answers."""

NOT_FOUND = "I couldn't find this in the handbook."


class HandbookRAG:
    def __init__(self):
        # Load everything ONCE (slow), then answer many questions (fast).
        self.model = SentenceTransformer(config.EMBED_MODEL)
        client = chromadb.PersistentClient(path=str(config.DB_DIR))
        try:
            self.collection = client.get_collection(config.COLLECTION)
        except Exception:
            raise SystemExit("No vector database found. Run:  python ingest.py")

        self.llm = None
        if config.GROQ_API_KEY:
            from groq import Groq
            self.llm = Groq(api_key=config.GROQ_API_KEY)

    # ---------------- RETRIEVAL ----------------
    def retrieve(self, question, k=config.TOP_K):
        q_vec = self.model.encode(config.QUERY_PREFIX + question,
                                  normalize_embeddings=True).tolist()
        res = self.collection.query(query_embeddings=[q_vec], n_results=k,
                                    include=["documents", "metadatas", "distances"])
        hits = []
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0],
                                    res["distances"][0]):
            hits.append({"text": text, "page": meta["page"],
                         "score": 1 - dist})        # distance -> similarity
        return hits                                  # already sorted best-first

    # ---------------- GENERATION ----------------
    @staticmethod
    def build_messages(question, hits, history=None):
        context = "\n\n".join(f"[Page {h['page']}]\n{h['text']}" for h in hits)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:                                  # earlier Q&A for follow-ups
            messages += history[-2 * config.HISTORY_TURNS:]
        messages.append({"role": "user", "content":
                         f"Handbook excerpts:\n{context}\n\nQuestion: {question}"})
        return messages

    def answer(self, question, history=None, k=config.TOP_K,
               min_score=config.MIN_SCORE):
        hits = self.retrieve(question, k)

        # Guard 1: nothing relevant enough -> don't let the LLM make things up
        if not hits or hits[0]["score"] < min_score:
            return {"answer": NOT_FOUND, "sources": hits, "used_llm": False}

        # No API key -> still useful: show the best passages
        if self.llm is None:
            text = ("*(No GROQ_API_KEY set - showing the most relevant passages "
                    "instead of a written answer.)*\n\n" +
                    "\n\n".join(f"**Page {h['page']}** (score {h['score']:.2f}): "
                                f"{h['text']}" for h in hits[:3]))
            return {"answer": text, "sources": hits, "used_llm": False}

        # Guard 2: the prompt tells the LLM to answer only from the excerpts
        resp = self.llm.chat.completions.create(
            model=config.LLM_MODEL,
            messages=self.build_messages(question, hits, history),
            temperature=0,                           # factual, not creative
        )
        return {"answer": resp.choices[0].message.content, "sources": hits,
                "used_llm": True}
