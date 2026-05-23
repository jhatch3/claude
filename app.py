"""
File app.py

Thin FastAPI layer over the core logic. This file only does HTTP: request/response
schemas, routing, status codes, the lifespan, and translating domain errors into
HTTP responses. The real work lives in two modules:
  - helpers.py  (Redis history, rate limiting, settings, logging)
  - ai.py       (Anthropic clients, system prompt, the model call)

run by:
>>> uvicorn app:app --reload

Conversation history is stored in Redis (Upstash), keyed per user_id, so the app
server stays stateless and can run with multiple workers / behind a load balancer.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from helpers import (
    load_history,
    append_user_message,
    append_assistant_message,
    reset_history,
    clear_all_history,
    check_rate_limit,
    redis_healthcheck,
    close_redis,
    RateLimitExceeded,
)
from ai import prompt, get_response, agent_healthcheck


logger = logging.getLogger("chatbot")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Probe Redis on startup; close the client's HTTP session on shutdown."""
    if not await redis_healthcheck():
        logger.warning("Redis (Upstash) not reachable at startup")
    yield
    # Shutdown: flush all stored conversations, then close the client.
    removed = await clear_all_history()
    logger.info("flushed %s stored conversation(s) on shutdown", removed)
    await close_redis()


app = FastAPI(title="Chatbot API", version="1.0.0", lifespan=lifespan)


# --- Schemas ---
class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    response: str


class HealthResponse(BaseModel):
    status: str = "ok"
    redis: bool
    agent: bool


class ResetResponse(BaseModel):
    status: str = "ok"
    message: str


# --- Domain-error -> HTTP mapping ---
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Translate the core RateLimitExceeded into a 429, using its message + retry_after."""
    # Only emit Retry-After when the exception carries one (it's optional now).
    headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after is not None else None
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": exc.message},
        headers=headers,
    )


# --- Routes ---
@app.get("/", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Liveness check; also reports whether Redis is reachable."""
    return HealthResponse(redis=await redis_healthcheck(), agent=await agent_healthcheck())


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat_endpoint(body: ChatRequest) -> ChatResponse:
    """
    Send a message and get the model's reply. History is stored in Redis per
    user_id, so each call appends to that user's running (windowed) conversation.
    """
    # Raises RateLimitExceeded -> handled by rate_limit_handler above (429).
    await check_rate_limit(body.user_id)

    try:
        await append_user_message(body.user_id, body.message)
        messages = await load_history(body.user_id)

        
        reply = await get_response(messages, system_message=prompt)
        await append_assistant_message(body.user_id, reply)

        return ChatResponse(response=reply)
    
    except Exception as e:
        logger.exception("chat failed for user_id=%s", body.user_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.delete("/chat/{user_id}", response_model=ResetResponse, tags=["chat"])
async def reset_conversation(user_id: str) -> ResetResponse:
    """Delete a user's stored history so the next /chat call starts fresh."""
    await reset_history(user_id)
    return ResetResponse(message=f"history cleared for {user_id}")
