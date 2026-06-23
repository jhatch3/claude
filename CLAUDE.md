# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ The README is stale

`readme.md` documents a FastAPI + Redis "Chatbot API" (`app.py`, `helpers.py`, `streamlit_app.py`) that **no longer exists** — those files were in `foo/`, which was deleted. Ignore the README for architecture; the live code is the `src/` package described below.

## What this is

A terminal customer-support agent built on the Anthropic Messages API with tool use. Claude drives a manual tool-use loop over a set of customer-support tools (lookup, refunds, escalation) backed by a SQL database, plus RAG retrieval over a policy/FAQ + ticket-transcript knowledge base.

## Commands

Everything runs as a **package from the repo root** (imports use the `src.` prefix; there is no `__init__.py` — it's a namespace package, so `cd`-ing into `src/` breaks imports).

```bash
pip install -r requirements.txt
python -m src.db        # create + seed the relational tables (idempotent)
python -m src.ingest    # chunk + embed + load the RAG knowledge base (idempotent)
python -m src.ai        # interactive chat REPL  (type 'exit'/'quit' to stop)
```

Run `src.db` before `src.ingest` before `src.ai` on a fresh checkout. `src.ai` alone (no DB) will fail the moment a record/RAG tool is called.

There is **no test suite and no linter configured.** Verification in this repo has been ad-hoc, pointing the app at a throwaway SQLite file and exercising tools directly, e.g.:

```bash
DATABASE_URL="sqlite:///./_tmp.db" python -c "
from src.db import db; db.seed()
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

Dependency flow: `ai → tools → {db, embeddings}`. `db` and `embeddings` know nothing about Claude.

- **`src/ai.py` — the Claude loop.** The `Conversation` class owns the message history and builds request params. `_run_tools()` is a **manual** agentic loop (not the SDK tool runner): it resolves one `tool_use` per turn (`disable_parallel_tool_use`) by calling `run_tool`, appending a `tool_result`, and re-calling the API until the model stops. A module-level `client` is created with a 60s timeout. `add_message` accepts a string, a list of content blocks, or a raw API `Message` (it extracts `.content`).
- **`src/tools.py` — tool registry + business rules.** Two decorators: `@tool(...)` registers a function's API schema + callable in `tool_dic`; `@hook("tool_name")` registers a **pre-call guard** in `hook_dic`. `run_tool` runs all hooks first — if any returns non-`None`, the tool is **blocked** and that value becomes the result without the tool ever executing. Customer-support tools query the `db` singleton; RAG tools embed the query and call `db.search_documents`.
- **`src/db.py` — data layer (OOP).** The `Database` class owns the SQLAlchemy engine + session factory and exposes all queries (the tools layer contains no SQL). ORM models: `Customer`, `Order`, `Document` (RAG chunks). A module-level `db = Database()` singleton is what `tools.py` imports.
- **`src/embeddings.py`** — `get_embedder()` returns `VoyageEmbedder` when `VOYAGE_API_KEY` is set, else a dependency-free `HashEmbedder` dev fallback.
- **`src/ingest.py`** — the RAG ingestion pipeline (corpus → chunk → embed → upsert). The sample corpus is inline here.
- **`src/prompt/sys.py`** — the `system_message` string passed into `Conversation`.

## Conventions that span files (read before editing)

- **Errors vs. business outcomes in tool results.** A tool returning a dict with an `"error"` key is a genuine failure — `ai.py` flags the `tool_result` with `is_error: true` so Claude recovers. A guardrail block (e.g. a refund over the limit) returns `{"status": "blocked", ...}` **without** an `"error"` key — it's a valid outcome the model acts on (by escalating), not an error. Preserve this distinction.
- **Money is `Decimal` end-to-end.** `Order.total` is `Numeric`; all refund logic compares/returns `Decimal` (parse the model's number via `Decimal(str(amount))`). It is never converted to `float` in logic. The only float conversion is at the JSON wire boundary, in `ai.py`'s `_json_default`, which encodes `Decimal` as a JSON number. Don't reintroduce `float()` in `tools.py`/`db.py` money paths.
- **Guardrails live in hooks, not tool bodies.** The refund threshold (`REFUND_THRESHOLD`) is enforced by `@hook` functions that run before the tool. Add new gating as a hook so it can short-circuit.
- **Strict tools require fully-specified schemas.** Tools use `strict=True`, which demands `additionalProperties: false` and **every** property listed in `required`. Genuinely optional params are modeled as nullable-and-required (`"type": ["string", "null"]` + in `required`), so the model always sends the key but may pass `null`.
- **SQLite dev / Postgres prod is a deliberate dual-target.** Code must run on both. Vector search in `db.search_documents` is portable numpy cosine (full scan) — correct on SQLite and at dev scale. The production swap to a pgvector `Vector` column + ANN index + `<=>` operator is localized to that method (`pgvector` is in requirements but not yet wired).
- **Embedder relevance caveat.** The dev `HashEmbedder` is lexical bag-of-words with poor semantic relevance — fine for exercising the pipeline, weak for real ranking. Switching to Voyage requires setting `VOYAGE_API_KEY` **and re-running `python -m src.ingest`** (vectors from different embedders aren't comparable).
- **Model choice.** Defaults to `claude-sonnet-4-6` with a `temperature`. Sonnet accepts sampling params; switching to an Opus 4.7/4.8 or Fable model would 400 on `temperature` — drop it (and prefer `thinking: {"type": "adaptive"}`) if you change the model.
