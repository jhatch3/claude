"""
File ai.py

AI logic for the chatbot:
  - the Anthropic clients (sync for the CLI stream, async for the API),
  - the system prompt,
  - get_response(): the async single-turn model call, with prompt caching,
  - the interactive terminal chat() loop.

Shared infrastructure (settings, logging) is imported from helpers.py.

Run the CLI directly (history kept in memory, not Redis):
>>> python -m ai
Type 'exit' or 'quit' to leave.
"""

from pprint import pprint

import anthropic

from helpers import settings, logger

# --- Clients ---
# Sync client for the CLI's streaming loop; async client for the API's get_response.
client = anthropic.Anthropic()
aclient = anthropic.AsyncAnthropic()

# --- System prompt ---
sys = (
    "Your are a agent the is tasked with anwering questions to the best of your ability. "
    "You are not allowed to ask for more information, you must answer the question with the information "
    "given. If you don't know the answer, say 'I don't know'. You will be given more information on this task by  "
    "delimiters <rules> </rules> <structured_output></structured_output> telling all the rules to follow and how to "
    "format your output. You will be given a prompt to answer after the delimiters."
    )

rules = (
    "You are not allowed to ask for more information, you must answer the question with the information given. If you "
    "don't know the answer, say 'I don't know'."
    )

structured_output = (
    "every response should be in pure text with the following format: 'your answer here' no new lines, no markdown, no code "
    "blocks, no quotes, no formatting, just pure text. If you don't know the answer, say 'I don't know'."
    )

prompt = sys + "<rules>" + rules + "</rules>" + "<structured_output>" + structured_output + "</structured_output>"


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------
async def get_response(messages: list[dict], system_message: str | None = None, temperature: float = 0.5, max_tokens: int = 1000, model: str | None = "claude-sonnet-4-6") -> str:
    """
    Async, non-interactive single-turn model call for use by the API.

    Uses the async Anthropic client so the network wait is real async I/O — callers
    can `await` it directly without offloading to a thread. Takes a list of messages
    (plus optional model params), sends them once, and returns the reply as text.

    Prompt caching: the system prompt and the whole conversation prefix are marked
    with cache_control. Caching is a prefix match, so each turn reads the prior
    prefix from cache (~0.1x cost) and only writes the new tail. Note Sonnet 4.6's
    minimum cacheable prefix is 2048 tokens — short early conversations won't cache
    until the history grows past that, which is expected.
    """
    model = model or settings.chat_model
    # Mark the last message so everything before it (system + prior turns) is the
    # cacheable prefix. Build a copy — we must NOT persist cache_control into the
    # stored history, and we must not mutate the caller's list.
    cached_messages = list(messages)
    if cached_messages:
        last = cached_messages[-1]
        cached_messages[-1] = {
            "role": last["role"],
            "content": [
                {
                    "type": "text",
                    "text": last["content"],
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }

    params = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": cached_messages,
    }

    if system_message:
        # System rendered as a content block so we can attach cache_control.
        params["system"] = [
            {
                "type": "text",
                "text": system_message,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    message = await aclient.messages.create(**params)

    # Surface cache activity so you can confirm caching is working in the logs.
    usage = message.usage
    logger.info(
        "cache read=%s write=%s uncached_input=%s",
        usage.cache_read_input_tokens,
        usage.cache_creation_input_tokens,
        usage.input_tokens,
    )

    return message.content[0].text


# ---------------------------------------------------------------------------
# CLI (interactive terminal) — keeps history in memory, not Redis
# ---------------------------------------------------------------------------
def add_user_message(content: str, messages: list[dict]) -> list[dict]:
    """Append a user message to an in-memory list and return it (CLI helper)."""
    messages.append({"role": "user", "content": content})
    return messages


def add_assistant_message(content: str, messages: list[dict]) -> list[dict]:
    """Append an assistant message to an in-memory list and return it (CLI helper)."""
    messages.append({"role": "assistant", "content": content})
    return messages


def chat(messages: list[dict], system_message: str | None = None, temperature: float = 0.5, max_tokens: int = 1000, model: str | None = None) -> None:
    """
    Interactive chatbot CLI loop.

    Streams the model's reply token-by-token to stdout and keeps the running
    conversation in the passed-in `messages` list.

    Quit: type 'exit' or 'quit'.
    """
    params = {
        "model": model or settings.chat_model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    if system_message:
        params["system"] = system_message

    while True:
        user_input = input("You: ")
        print("\n")

        if user_input.lower() in ["exit", "quit"]:
            break

        add_user_message(user_input, messages)

        try:
            # Live streaming of the response.
            with client.messages.stream(**params) as stream:
                for event in stream.text_stream:
                    print(event, end="", flush=True)

            # Grab the completed message and store it.
            message = stream.get_final_message()
            add_assistant_message(message.content[0].text, messages)
            print("\n")

        except Exception as e:
            add_assistant_message(f"Error occurred while processing the request. {e}", messages)
            break

async def agent_healthcheck() -> bool:
    """
    Return True if the Anthropic API is reachable with a valid key, else False.

    Uses models.retrieve — a cheap GET that generates no tokens — rather than a full
    message. A health endpoint can be polled often, so it must not cost output tokens
    or burn rate limit on every call. Any failure (bad key, network, unknown model)
    raises, which we treat as unhealthy.
    """
    try:
        await aclient.models.retrieve(settings.chat_model)
        return True
    except Exception as e:
        logger.warning("agent healthcheck failed: %s", e)
        return False



def main():
    """Run the interactive CLI, then pretty-print the final transcript."""
    messages = []
    chat(messages, system_message=prompt, temperature=0.0, max_tokens=1000)
    pprint(messages)


if __name__ == "__main__":
    main()
