"""
Ingest the knowledge corpus into the vector store.

Loads policy/FAQ text and past ticket transcripts, chunks them, embeds each
chunk, and upserts into the `documents` table. Idempotent: it clears existing
documents first, so re-running re-embeds the whole corpus (needed after
switching embedders — e.g. dev hash -> Voyage).

Run once after `python -m src.db`:
    python -m src.ingest
"""
import hashlib

from src.db import db
from src.embeddings import get_embedder
from src.prompt.knowledge import FAQS, POLICY_DOCS, TRANSCRIPTS


def chunk_text(text, max_chars=600, overlap=100):
    """Split text into overlapping character windows (small, demo-grade).

    Production should chunk by the embedder's token limit and on semantic
    boundaries; this keeps the pipeline simple and dependency-free.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start : start + max_chars].strip()
        if chunk:
            chunks.append(chunk)
        start += max_chars - overlap
    return chunks


def _rows_for(source, doc_id, text, customer_id=None):
    rows = []
    for i, chunk in enumerate(chunk_text(text)):
        rows.append(
            {
                "source": source,
                "doc_id": doc_id,
                "chunk_index": i,
                "content": chunk,
                "content_hash": hashlib.sha256(chunk.encode()).hexdigest(),
                "customer_id": customer_id,  # None for org-wide Knowledge Base
                "meta": {},
            }
        )
    return rows


def ingest():
    db.init_db()
    embedder = get_embedder()

    rows = []
    for doc_id, text in POLICY_DOCS.items():
        rows += _rows_for("policy", doc_id, text)
    for doc_id, text in FAQS.items():
        rows += _rows_for("faq", doc_id, text)
    for t in TRANSCRIPTS:
        rows += _rows_for(
            "transcript", t["doc_id"], t["text"], customer_id=t["customer_id"]
        )

    # Embed all chunks in one batch, then attach the vectors.
    vectors = embedder.embed([r["content"] for r in rows], input_type="document")
    for row, vector in zip(rows, vectors):
        row["embedding"] = vector

    db.reset_documents()  # idempotent re-ingest
    db.add_documents(rows)
    return len(rows)


if __name__ == "__main__":
    count = ingest()
    print(f"Ingested {count} chunks using embedder '{get_embedder().name}'")
