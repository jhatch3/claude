# Chatbot API

A small FastAPI service that wraps Claude with **server-side, per-user conversation
memory**. History lives in Redis (Upstash REST), so the API stays stateless and
horizontally scalable. Ships with prompt caching, per-user rate limiting, and a
Streamlit UI for poking at the routes.

## Architecture

The code is split by concern; dependencies flow one way (`app → ai → helpers`):

| File | Responsibility |
|------|----------------|
| `app.py` | HTTP layer only — routes, request/response schemas, the `RateLimitExceeded → 429` handler, and the lifespan. |
| `ai.py` | Claude logic — Anthropic clients (sync for the CLI, async for the API), the system prompt, `get_response()` (async, with prompt caching), and the interactive CLI. |
| `helpers.py` | Infrastructure — typed `Settings`, logging, the Redis client, the conversation store (history CRUD), and the rate limiter. No web/model SDK code. |
| `streamlit_app.py` | A "route explorer" UI to exercise the endpoints. |

`helpers.py` knows nothing about FastAPI or the Anthropic SDK, so the core logic is reusable and testable on its own.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in the values below
uvicorn app:app --reload    # API at http://localhost:8000, docs at /docs
```

Other entry points:
- `python -m ai` — interactive terminal chat (in-memory history, not Redis).
- `streamlit run streamlit_app.py` — the route explorer UI.

## Configuration

All config is read from the environment / `.env` via a typed `Settings` object (`helpers.py`). `.env` is gitignored.

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `ANTHROPIC_API_KEY` | yes | — | Claude API key. |
| `UPSTASH_REDIS_REST_URL` | yes | — | Upstash REST endpoint (`https://<name>.upstash.io`). `REDIS_URL` is accepted as a fallback. |
| `UPSTASH_REDIS_REST_TOKEN` | yes | — | Upstash REST token (≠ the native Redis password). |
| `CHAT_MODEL` | no | `claude-sonnet-4-6` | Model used for replies. |
| `CHAT_HISTORY_TTL` | no | `604800` | Idle conversation lifetime (seconds). |
| `CHAT_MAX_MESSAGES` | no | `40` | Sliding-window cap on stored messages per user. |
| `CHAT_RATE_LIMIT_MAX` | no | `20` | Max `/chat` requests per window, per user. |
| `CHAT_RATE_LIMIT_WINDOW` | no | `60` | Rate-limit window (seconds). |

## API

Interactive docs are auto-generated at `/docs`.

| Method | Path | Body | Notes |
|--------|------|------|-------|
| `GET` | `/` | — | Health: `{status, redis, agent}` (Redis + Anthropic reachability). |
| `POST` | `/chat` | `{user_id, message}` | Appends to the user's history and returns `{response}`. `429` + `Retry-After` when rate-limited. |
| `DELETE` | `/chat/{user_id}` | — | Clears one user's stored history. |

```bash
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"user_id":"alice","message":"My name is Alice."}'
# follow-up in the same conversation:
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"user_id":"alice","message":"What is my name?"}'
```

## How it works

- **History** — stored as a Redis list per `chat:{user_id}`, trimmed to `CHAT_MAX_MESSAGES` with a refreshed TTL on every write.
- **Rate limiting** — fixed-window counter per user (`ratelimit:{user_id}:{window}`); raises the domain error `RateLimitExceeded`, which `app.py` maps to a `429`.
- **Prompt caching** — the system prompt and conversation prefix are marked `cache_control: ephemeral`, so repeat turns read the prefix from cache (~0.1× cost). Caching only engages once the prefix exceeds the model's minimum (~2048 tokens on Sonnet 4.6); short chats won't cache, which is expected.
- **Async I/O** — the API path uses `AsyncAnthropic`, so the event loop isn't blocked on model calls.

## Operational notes

- **Upstash REST** means one HTTPS round-trip per Redis command (no pipelining). Fine at this scale; switch to a native `redis://` client + pipelines if latency matters.
- **Shutdown flushes all conversations.** On graceful shutdown the lifespan clears every `chat:*` key — convenient for dev, destructive in prod (every deploy wipes history). Gate it behind a setting before shipping. It only runs on graceful shutdown, not a hard kill.
- **No auth yet.** `user_id` is client-supplied and trusted. Derive it from an authenticated token before exposing this publicly.
