"""
Embedding providers for RAG.

Production uses Voyage AI (set VOYAGE_API_KEY). When that's unavailable the
factory falls back to a dependency-free hashing embedder so the pipeline runs
in dev without an API key or network.

Vectors from different embedders are NOT comparable — re-run ingestion whenever
you switch providers (e.g. dev hash -> Voyage).
"""
import hashlib
import os
import re

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Deterministic hashing bag-of-words embedder for dev/testing.

    No API, no model download. Tokens are hashed into fixed buckets and the
    vector is L2-normalized, so texts sharing words land near each other —
    enough lexical signal to exercise retrieval locally.
    """

    name = "hash-dev"

    def __init__(self, dim=256):
        self.dim = dim

    def _embed_one(self, text):
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in _TOKEN_RE.findall(text.lower()):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        norm = np.linalg.norm(vec)
        return (vec / norm if norm else vec).tolist()

    def embed(self, texts, input_type="document"):
        # input_type is ignored here; it exists to match the Voyage interface.
        return [self._embed_one(t) for t in texts]


class VoyageEmbedder:
    """Voyage AI embeddings (production)."""

    def __init__(self, model="voyage-3.5", api_key=None):
        import voyageai  # lazy: only needed when actually using Voyage

        self.model = model
        self.name = model
        self._client = voyageai.Client(api_key=api_key or os.getenv("VOYAGE_API_KEY"))

    def embed(self, texts, input_type="document"):
        # input_type "document" for stored chunks, "query" for searches — Voyage
        # embeds them slightly differently for better retrieval.
        result = self._client.embed(texts, model=self.model, input_type=input_type)
        return result.embeddings


def get_embedder():
    """Voyage when installed + configured, else the dev hashing embedder."""
    if os.getenv("VOYAGE_API_KEY"):
        try:
            return VoyageEmbedder()
        except ImportError:
            print("[embeddings] voyageai not installed; falling back to hash embedder")
    return HashEmbedder()
