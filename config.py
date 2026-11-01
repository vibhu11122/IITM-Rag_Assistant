"""
config.py - every setting for the project lives here.
Change values here instead of hunting through the code.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")            # reads GROQ_API_KEY from the .env file

# ---- Files & folders -------------------------------------------------
PDF_PATH = BASE_DIR / "data" / "handbook.pdf"   # put the handbook here
DB_DIR = BASE_DIR / "db"                        # ChromaDB saves vectors here
CHUNKS_FILE = BASE_DIR / "chunks.json"          # human-readable copy of chunks
COLLECTION = "handbook"                         # name of the table in ChromaDB

# ---- Chunking (Lesson 3) ---------------------------------------------
CHUNK_SIZE = 800        # max characters per chunk
CHUNK_OVERLAP = 150     # characters repeated between neighbouring chunks
MIN_CHUNK = 100         # tiny leftover pieces get merged into the previous chunk

# ---- Embeddings (Lessons 1-2) ----------------------------------------
# bge-small: free, runs on CPU, better at retrieval than all-MiniLM-L6-v2.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge models expect this prefix on QUESTIONS (not on documents). It bridges
# the "question vs statement" gap you noticed in Lesson 1.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ---- Retrieval -------------------------------------------------------
TOP_K = 5               # how many chunks to send to the LLM
MIN_SCORE = 0.35        # if the BEST chunk scores below this -> "not found"
                        # (tune it: look at scores for good vs. bad questions)

# ---- Generation (LLM) ------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
HISTORY_TURNS = 2       # how many previous Q&A pairs the LLM sees (for follow-ups)
