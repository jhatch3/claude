# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ The README is stale

`readme.md` documents a FastAPI + Redis "Chatbot API" (`app.py`, `helpers.py`, `streamlit_app.py`) that **no longer exists** — those files were in `foo/`, which was deleted. Ignore the README for architecture; the live code is the `src/` package described below.

## What this is

A terminal customer-support agent built on the Anthropic Messages API with tool use. A single agent drives a manual tool-use loop over customer-support tools (lookup, refunds, escalation) backed by a SQL database, plus RAG retrieval over a policy/FAQ + ticket-transcript knowledge base. A **multi-agent harness** (hub-and-spoke) is also provided as a demonstration — see `docs/adr/0002`.

## Commands

Everything runs as a **module from the repo root** (imports use the `src.` prefix; run from the root, never `cd` into `src/`).

```bash
pip install -r requirements.txt
python -m src.db        # create + seed the relational tables (idempotent)
python -m src.ingest    # chunk + embed + load the RAG knowledge base (idempotent)
python -m src.llm       # single-agent chat REPL  (type 'exit'/'quit')
python -m src.harness   # multi-agent hub-and-spoke harness
```

Run `src.db` before `src.ingest` before `src.llm`/`src.harness` on a fresh checkout. The chat alone (no DB) fails the moment a record/RAG tool is called.

**Postgres (prod):** `python -m src.db` (`create_all`) is dev-only; the schema source of truth on Postgres is Alembic — `DATABASE_URL=postgresql+psycopg://... alembic upgrade head` (creates the `vector` extension + HNSW index).

There is **no test suite and no linter configured.** Verification in this repo has been ad-hoc, pointing the app at a throwaway SQLite file and exercising tools directly, e.g.:

```bash
DATABASE_URL="sqlite:///./_tmp.db" python -c "
from src.db import seed; seed()
from src.ingest import ingest; ingest()
from src.tools import run_tool
print(run_tool('get_customer', {'email':'ada@example.com','phone':None,'customer_id':None}))
"
```

## Environment

Config is read from `.env` (via `load_dotenv()`):
- `ANTHROPIC_API_KEY` — required.
- `DATABASE_URL` — optional; **defaults to `sqlite:///support.db`** (dev). Production target is `postgresql+psycopg://...`.
- `VOYAGE_API_KEY` — optional; enables real Voyage embeddings (see RAG below).

## Architecture

Dependency flow: `harness → llm → tools → {db, embeddings}`. `db` and `embeddings` know nothing about Claude. The code is organized into packages:

- **`src/llm/` — the Claude loop.** `conversation.py` holds the `Conversation` class: it owns the message history, builds request params, and runs the **manual** agentic loop (not the SDK tool runner) — one `tool_use` per turn (`disable_parallel_tool_use`), append `tool_result`, re-call until the model stops. `run_task()` is the non-interactive variant the harness uses (returns a tool trace, accepts a custom `dispatch`). `client.py` holds the `client` (60s timeout) and `_json_default` (encodes `Decimal` money as a JSON number). Entry point: `python -m src.llm`.
- **`src/tools/` — tool registry + business rules.** `registry.py` has the `@tool` / `@hook` decorators and `run_tool` (hooks run first; a non-`None` hook result **blocks** the tool). Domain modules — `customer.py`, `refunds.py` (+ threshold hooks), `escalation.py`, `knowledge.py` (RAG) — register their tools on import via the package `__init__`. Record tools query the `db` singleton; RAG tools embed the query and call `db.search_documents`.
- **`src/db/` — data layer.** `models.py` (ORM: `Customer`, `Order`, `Document` + the dialect-aware embedding column), `database.py` (the `Database` class + `db` singleton; all SQL lives here), `seed.py` (mock data + `seed()`). `python -m src.db` seeds.
- **`src/embeddings.py`** — `get_embedder()` returns `VoyageEmbedder` when `VOYAGE_API_KEY` is set, else a dependency-free `HashEmbedder` dev fallback.
- **`src/ingest.py`** — RAG ingestion (corpus → chunk → embed → upsert). The corpus text lives in `src/prompts/corpus.py`.
- **`src/prompts/`** — `system.py` (`SYSTEM_MESSAGE`) and `corpus.py` (the RAG source text).
- **`src/harness/` — the multi-agent demo.** `agents.py` (spoke roster, prompts, the gate set, orchestrator delegation tool defs), `orchestrator.py` (the `Harness`: orchestrator loop, spokes, deterministic gate, findings buffer, Synth). See ADR-0002.
- **`alembic/`** — Postgres migrations (the `vector` extension + HNSW index live in `0001_initial`).

## Conventions that span files (read before editing)

- **Errors vs. business outcomes in tool results.** A tool returning a dict with an `"error"` key is a genuine failure — the loop (`src/llm/conversation.py`) flags the `tool_result` with `is_error: true` so Claude recovers. A guardrail block (e.g. a refund over the limit) returns `{"status": "blocked", ...}` **without** an `"error"` key — it's a valid outcome the model acts on (by escalating), not an error. Preserve this distinction.
- **Money is `Decimal` end-to-end.** `Order.total` is `Numeric`; all refund logic compares/returns `Decimal` (parse the model's number via `Decimal(str(amount))`). It is never converted to `float` in logic. The only float conversion is at the JSON wire boundary, in `src/llm/client.py`'s `_json_default`, which encodes `Decimal` as a JSON number. Don't reintroduce `float()` in the `tools/` or `db/` money paths.
- **Guardrails live in hooks, not tool bodies.** The refund threshold (`REFUND_THRESHOLD`) is enforced by `@hook` functions that run before the tool. Add new gating as a hook so it can short-circuit.
- **Strict tools require fully-specified schemas.** Tools use `strict=True`, which demands `additionalProperties: false` and **every** property listed in `required`. Genuinely optional params are modeled as nullable-and-required (`"type": ["string", "null"]` + in `required`), so the model always sends the key but may pass `null`.
- **SQLite dev / Postgres prod is a deliberate dual-target.** Code must run on both. `db.search_documents` branches by dialect: pgvector `<=>` + HNSW on Postgres, numpy cosine full-scan on SQLite. The Postgres path (the `vector(1024)` column, HNSW index in Alembic `0001`, and the `<=>` query) is **written but unverified** — there's no Postgres/pgvector in dev, so smoke-test it on a real Postgres before trusting it.
- **Harness gate is deterministic.** In `src/harness/`, money/history/escalation spokes only run for a `customer_id` the Customer spoke verified this conversation (the `verified` set in `orchestrator.py`), enforced in `_dispatch` — not by prompt.
- **Embedder relevance caveat.** The dev `HashEmbedder` is lexical bag-of-words with poor semantic relevance — fine for exercising the pipeline, weak for real ranking. Switching to Voyage requires setting `VOYAGE_API_KEY` **and re-running `python -m src.ingest`** (vectors from different embedders aren't comparable).
- **Model choice.** Defaults to `claude-sonnet-4-6` with a `temperature`. Sonnet accepts sampling params; switching to an Opus 4.7/4.8 or Fable model would 400 on `temperature` — drop it (and prefer `thinking: {"type": "adaptive"}`) if you change the model.
