# Single Postgres + pgvector store for records and RAG vectors

## Context

The agent needs a scalable RAG store (target: ~1M Document chunks, ~100s of Customers, 10–50 retrieval QPS, p95 < 300ms, continuous updates). It already runs on Postgres for Customer/Order records.

## Decision

Keep **one Postgres database** holding both relational records and RAG vectors, using the **pgvector** extension with an **HNSW** index. We do not run a dedicated vector database (Qdrant/Weaviate/Milvus).

Per-Customer privacy is **single-tenant, per-Customer row scoping** — not organization-level multi-tenancy. `customer_id` is promoted to a first-class **nullable indexed column** on the `documents` table (`NULL` = organization-wide Knowledge Base). Transcript reads are filtered `WHERE source = 'transcript' AND customer_id = :id`; Knowledge Base reads filter `source IN ('policy','faq')`. The scope filter is enforced in a single query choke point (`Database.search_documents`), never in Python after the scan.

Schema: a single `documents` table (no partitioning) with an HNSW index on the embedding and a B-tree on `(source, customer_id)`, relying on pgvector ≥ 0.8 **iterative index scans** for filtered-query recall.

## Considered Options

- **Dedicated vector DB** — rejected: at 1M vectors / 50 QPS pgvector is well within range, and a second system means extra ops plus keeping it in sync with the Postgres records. Revisit only if the corpus passes ~10M or filtered recall can't be met in pgvector.
- **Partition `documents` by `source`** — deferred: the first lever to pull if filtered transcript recall/latency misses the p95 budget; not built upfront.
- **Per-Customer partial indexes** — rejected: best filtered recall but hundreds of indexes that don't scale as Customers grow.

## Consequences

- The HNSW-plus-`WHERE`-filter recall risk (a Customer owns a tiny slice of 1M vectors) must be validated against a golden set before relying on Option A; partitioning by `source` is the known fallback.
- pgvector's `Vector` column is Postgres-only; dev still runs on SQLite, so the embedding column must be dialect-aware (pgvector `Vector` on Postgres, JSON + numpy cosine on SQLite).
