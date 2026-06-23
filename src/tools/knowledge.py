"""RAG retrieval tools (semantic search over the vector store)."""
from src.db import db
from src.embeddings import get_embedder
from src.tools.registry import tool

# Embedder used to vectorize search queries (matches what ingestion used).
_embedder = get_embedder()


def _format_hits(hits):
    """Turn (Document, score) pairs into JSON-serializable result dicts."""
    return [
        {
            "content": doc.content,
            "source": doc.source,
            "doc_id": doc.doc_id,
            "score": round(score, 4),
        }
        for doc, score in hits
    ]


def _semantic_search(query, sources):
    """Embed the query and return the top knowledge chunks for the given sources."""
    query_vec = _embedder.embed([query], input_type="query")[0]
    hits = db.search_documents(query_vec, top_k=4, sources=sources)
    return {"results": _format_hits(hits)}


# A single text-query schema reused by the knowledge-search tools.
_QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Natural-language question to search for",
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}


@tool(
    name="search_policies",
    description=(
        "Search company policy documents (refunds, shipping, returns) for "
        "relevant text. Call this whenever the customer asks how a policy works "
        "or what the rules are, instead of answering from memory. Returns the "
        "most relevant policy passages."
    ),
    input_schema=_QUERY_SCHEMA,
    strict=True,
)
def search_policies(query):
    return _semantic_search(query, ["policy"])


@tool(
    name="search_faqs",
    description=(
        "Search the FAQ knowledge base for relevant text. Call this for general "
        "'how do I...' questions (tracking an order, handling a damaged item, "
        "etc.) instead of answering from memory. Returns the most relevant FAQ "
        "passages."
    ),
    input_schema=_QUERY_SCHEMA,
    strict=True,
)
def search_faqs(query):
    return _semantic_search(query, ["faq"])


@tool(
    name="search_past_tickets",
    description=(
        "Search a verified customer's past support tickets for similar prior "
        "issues. Requires the verified customer_id; only that customer's "
        "transcripts are searched. Use it to check 'have we seen this before'."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "customer_id": {
                "type": "string",
                "description": "Verified customer id (from get_customer)",
            },
            "query": {
                "type": "string",
                "description": "Natural-language description of the issue",
            },
        },
        "required": ["customer_id", "query"],
        "additionalProperties": False,
    },
    strict=True,
)
def search_past_tickets(customer_id, query):
    query_vec = _embedder.embed([query], input_type="query")[0]
    hits = db.search_documents(
        query_vec, top_k=4, sources=["transcript"], customer_id=customer_id
    )
    return {"results": _format_hits(hits)}
