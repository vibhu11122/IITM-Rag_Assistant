"""
ingest.py - STEP 1: turn the handbook PDF into a searchable vector database.

    PDF  ->  page text  ->  chunks  ->  embeddings  ->  ChromaDB (saved in ./db)

Run once (and again whenever the handbook changes):
    python ingest.py
"""
import json
import re
import sys

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import config


# ---------------------------------------------------------------------
# 1. READ: extract and clean text, one page at a time
# ---------------------------------------------------------------------
def load_pages(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""            # None for image-only pages
        text = re.sub(r"-\s*\n\s*", "", text)        # join words hyphen-split at line end
        text = re.sub(r"\s+", " ", text).strip()     # collapse all whitespace
        pages.append({"page": i + 1, "text": text})
    return pages


# ---------------------------------------------------------------------
# 2. CHUNK: overlapping windows that try to end on a full sentence
# ---------------------------------------------------------------------
def chunk_text(text, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP,
               min_chunk=config.MIN_CHUNK):
    if overlap >= size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))

        # Not at the end of the text? Try to cut at the last ". " in the
        # second half of the window, so we don't slice a rule in half.
        if end < len(text):
            cut = text.rfind(". ", start + size // 2, end)
            if cut != -1:
                end = cut + 1                        # keep the full stop

        piece = text[start:end].strip()
        if chunks and len(piece) < min_chunk:
            chunks[-1] += " " + piece                # tiny leftover -> merge back
        elif piece:
            chunks.append(piece)

        if end >= len(text):                         # reached the end: stop
            break
        start = max(end - overlap, start + 1)        # step back for overlap

        # Start the next chunk cleanly: at a sentence start inside the
        # overlap if there is one, otherwise at least at a word start.
        sentence = text.find(". ", start, end)
        if sentence != -1 and sentence + 2 < end:
            start = sentence + 2
        elif text[start - 1] != " ":
            space = text.find(" ", start, end)
            if space != -1:
                start = space + 1
    return chunks


def make_chunks(pages):
    all_chunks = []
    for p in pages:
        for j, piece in enumerate(chunk_text(p["text"])):
            all_chunks.append({"id": f"p{p['page']}_c{j}",   # unique id
                               "page": p["page"],             # for citations
                               "text": piece})
    return all_chunks


# ---------------------------------------------------------------------
# 3 + 4. EMBED and STORE in ChromaDB
# ---------------------------------------------------------------------
def build_index(chunks, model):
    texts = [c["text"] for c in chunks]
    # Documents get NO prefix; only questions do (see config.QUERY_PREFIX).
    vectors = model.encode(texts, normalize_embeddings=True,
                           show_progress_bar=True, batch_size=32).tolist()

    client = chromadb.PersistentClient(path=str(config.DB_DIR))
    try:
        client.delete_collection(config.COLLECTION)  # start fresh every run
    except Exception:
        pass
    # "cosine" -> Chroma returns distance = 1 - cosine_similarity
    collection = client.create_collection(config.COLLECTION,
                                          metadata={"hnsw:space": "cosine"})

    batch = 1000
    for i in range(0, len(chunks), batch):
        collection.add(
            ids=[c["id"] for c in chunks[i:i + batch]],
            documents=texts[i:i + batch],
            embeddings=vectors[i:i + batch],
            metadatas=[{"page": c["page"]} for c in chunks[i:i + batch]],
        )
    return collection


def main():
    pdf_path = config.PDF_PATH
    if not pdf_path.exists():
        sys.exit(f"Handbook not found at {pdf_path}\n"
                 f"Download the PDF and save it there as 'handbook.pdf'.")

    print(f"1/4 Reading {pdf_path.name} ...")
    pages = load_pages(pdf_path)
    empty = [p["page"] for p in pages if len(p["text"]) < 50]
    print(f"    {len(pages)} pages | nearly empty pages: {empty or 'none'}")
    if len(empty) > len(pages) / 2:
        print("    WARNING: most pages have no text - the PDF may be scanned "
              "images and need OCR.")

    print("2/4 Chunking ...")
    chunks = make_chunks(pages)
    sizes = [len(c["text"]) for c in chunks]
    print(f"    {len(chunks)} chunks | size min {min(sizes)}, "
          f"avg {sum(sizes) // len(sizes)}, max {max(sizes)} chars")
    with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"    saved a readable copy to {config.CHUNKS_FILE.name}")

    print(f"3/4 Loading embedding model {config.EMBED_MODEL} ...")
    model = SentenceTransformer(config.EMBED_MODEL)

    print("4/4 Embedding and storing in ChromaDB ...")
    collection = build_index(chunks, model)
    print(f"\nDone! {collection.count()} vectors saved in ./{config.DB_DIR.name}/")
    print("Next:  python chat.py   or   streamlit run app.py")


if __name__ == "__main__":
    main()
