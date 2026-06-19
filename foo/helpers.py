"""
File helpers.py

Support / infrastructure layer shared by the API (app.py) and the AI logic (ai.py):
  - typed Settings loaded from environment / .env,
  - logging setup,
  - the Redis (Upstash REST) client,
  - the per-user conversation store (history CRUD) and the rate limiter.

No web-framework or model-SDK code lives here — just plumbing the rest builds on.
"""

import json
import time
import logging
from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from upstash_redis.asyncio import Redis

load_dotenv()

# Dedicated logger with its own handler so our lines show under uvicorn (which
# only configures its own loggers) without double-handling its output.
logger = logging.getLogger("chatbot")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


# ---------------------------------------------------------------------------
# Settings (typed, env-driven). Field names map to env vars case-insensitively,
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Redis Upstash REST
    upstash_redis_rest_url: str | None = None
    upstash_redis_rest_token: str | None = None
    redis_url: str | None = None

    chat_history_ttl: int = 604800       # idle conversation lifetime, seconds (7 days)
    chat_max_messages: int = 40          # sliding-window cap on stored messages per user
    chat_rate_limit_max: int = 20        # max /chat requests per window, per user
    chat_rate_limit_window: int = 60     # rate-limit window, seconds

    chat_model: str = "claude-sonnet-4-6"

    @property
    def redis_rest_url(self) -> str | None:
        """Native Upstash REST URL, falling back to REDIS_URL."""
        return self.upstash_redis_rest_url or self.redis_url


@lru_cache
def get_settings() -> Settings:
    """Cached so the env/.env is read once; override in tests via this function."""
    return Settings()


settings = get_settings()

# Upstash REST client: talks to Redis over HTTPS (one request per command)
redis_client = Redis(url=settings.redis_rest_url, token=settings.upstash_redis_rest_token)


# --- Domain errors ---
class RateLimitExceeded(Exception):
    """
    Exception raised when the API rate limit is reached.
    
    Attributes:
        message (str): Error message.
        retry_after (int): Time in seconds to wait before retrying.
        
    """
    def __init__(self, message="API rate limit exceeded. Please wait before retrying.", retry_after=None):
        self.message = message
        self.retry_after = retry_after
        super().__init__(self.message)

# ---------------------------------------------------------------------------
# Redis-backed conversation history (Upstash)
# ---------------------------------------------------------------------------
def _key(user_id: str) -> str:
    """Redis key holding one user's message history."""
    return f"chat:{user_id}"


async def load_history(user_id: str) -> list[dict]:
    """Return the stored message history for a user (oldest -> newest)."""
    raw = await redis_client.lrange(_key(user_id), 0, -1)

    try:
        return [json.loads(item) for item in raw]
    except json.JSONDecodeError:
        logger.warning("Failed to decode message history for user_id=%s", user_id)
        return []


async def append_message(user_id: str, role: str, content: str) -> None:
    """
    Append one message, trim to the sliding window, and refresh the TTL.

    RPUSH is atomic, so concurrent requests append cleanly instead of clobbering a
    shared blob. (Each command is its own HTTPS round-trip via the REST client.)
    """
    try:
        key = _key(user_id)
        await redis_client.rpush(key, json.dumps({"role": role, "content": content}))
        await redis_client.ltrim(key, -settings.chat_max_messages, -1)
        await redis_client.expire(key, settings.chat_history_ttl)
    except Exception as e:
        logger.error("Error occurred while appending message for user_id=%s: %s", user_id, e)

async def append_user_message(user_id: str, content: str) -> None:
    """Append a user message to the user's stored history."""
    await append_message(user_id, "user", content)


async def append_assistant_message(user_id: str, content: str) -> None:
    """Append an assistant message to the user's stored history."""
    await append_message(user_id, "assistant", content)


async def reset_history(user_id: str) -> None:
    """Delete a user's stored history so the next turn starts fresh."""
    await redis_client.delete(_key(user_id))


async def clear_all_history() -> int:
    """
    Delete EVERY stored conversation (all chat:* keys); returns the number removed.

    Used on shutdown to flush the store. Uses SCAN (not KEYS) so it won't block Redis
    on a large keyspace. Destructive across all users — not a per-user reset.
    """
    deleted = 0
    cursor = 0
    while True:
        cursor, keys = await redis_client.scan(cursor, match="chat:*", count=100)
        if keys:
            await redis_client.delete(*keys)
            deleted += len(keys)
        if cursor == 0:
            break
    return deleted


async def check_rate_limit(user_id: str) -> None:
    """
    Fixed-window rate limit, per user. One counter key per (user, time window);
    INCR is atomic and the key auto-expires when the window passes. Raises
    RateLimitExceeded once a user goes over the configured budget for the window.
    """
    
    # Gives number of seconds, divided by window size, floored to an int.
    # 0 to 59 seconds in a minute all map to the same window value, then 60-119 to the next, etc.
    window = int(time.time()) // settings.chat_rate_limit_window
    key = f"ratelimit:{user_id}:{window}"
    count = await redis_client.incr(key)

    if count > settings.chat_rate_limit_max:
        raise RateLimitExceeded(retry_after=settings.chat_rate_limit_window)
    
    await redis_client.expire(key, settings.chat_rate_limit_window)


async def redis_healthcheck() -> bool:
    """Return True if Redis is reachable, False otherwise."""
    try:
        await redis_client.get("__healthcheck__")
        return True
    except Exception:
        return False


async def close_redis() -> None:
    """Close the underlying HTTP session if this client version exposes one."""
    try:
        await redis_client.close()
    except Exception:
        pass
