"""
Main file for the AI module

This module contains the core AI logic for the application.

"""
import json
from typing import Literal

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.prompt.sys import system_message
from src.tools import get_tool_definitions, run_tool

load_dotenv()

# Globals
# ==================================
# 60s timeout (vs the 10-min default) so a stuck network surfaces quickly
# instead of looking like a freeze.
client = anthropic.Anthropic(timeout=60)
# ==================================


class SupportReply(BaseModel):
    """Structured shape for the assistant's final answer each turn.

    Constrained via output_config.format so every turn yields a machine-readable
    result the app can route on, not just free prose.
    """

    reply: str = Field(description="The plain-text message to show the user.")
    resolution: Literal[
        "resolved", "refunded", "info_provided", "escalated", "needs_more_info"
    ] = Field(description="How this turn was resolved.")
    escalated: bool = Field(description="True if the case was handed to a human.")


def _strict_schema(model_cls):
    """Build a strict json_schema (additionalProperties: false) from a model."""
    schema = model_cls.model_json_schema()
    schema["additionalProperties"] = False
    return schema


class Conversation:
    """A single chat conversation that owns its own message history.

    Each instance keeps its own `messages` list, so multiple conversations can
    run independently without sharing global state.
    """

    def __init__(
        self,
        model="claude-sonnet-4-6",
        temperature=0.7,
        max_tokens=1024,
        system_message=None,
        tools=None,
        response_schema=SupportReply,
    ):
        self.messages = []
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_message = system_message
        self.tools = tools if tools is not None else get_tool_definitions()
        # Pydantic model the final reply is constrained to; None = free text.
        self.response_schema = response_schema

    def add_message(self, role, content):
        """Append a message after validating role and content.

        Centralizes the error logic so callers just hand over content; an empty
        or wrong-typed payload raises here instead of failing later at the API.
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")
        if content is None:
            raise ValueError("content must not be None")

        if isinstance(content, (str, list)):
            # A string or an explicit list of content blocks goes in as-is.
            self.messages.append({"role": role, "content": content})
        else:
            # Otherwise it's an API Message — store its content blocks.
            self.messages.append({"role": role, "content": content.content})

    def add_user_message(self, content):
        self.add_message("user", content)

    def add_assistant_message(self, content):
        self.add_message("assistant", content)

    def _build_params(self):
        params = {
            "messages": self.messages,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if self.tools:
            params["tools"] = self.tools
            params["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}

        if self.response_schema is not None:
            # Constrains the final text turn to the schema; tool turns are
            # unaffected (the model still emits tool_use blocks mid-loop).
            params["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": _strict_schema(self.response_schema),
                }
            }

        if self.system_message:
            # System rendered as a content block so we can attach cache_control.
            params["system"] = [
                {
                    "type": "text",
                    "text": self.system_message,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        return params

    def _run_tools(self, message):
        """Resolve tool-use turns until the model returns a final answer.

        disable_parallel_tool_use guarantees at most one tool_use block per turn.
        Returns the final (non-tool-use) message.
        """
        while message.stop_reason == "tool_use":
            tool_use = next(b for b in message.content if b.type == "tool_use")
            result = run_tool(tool_use.name, tool_use.input)

            # An "error" key marks a genuine failure (unknown tool, exception,
            # or validation) — flag it so Claude can recover. Business outcomes
            # like a blocked refund carry no "error" key and stay normal.
            tool_result = {
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": json.dumps(result),
            }
            if isinstance(result, dict) and "error" in result:
                tool_result["is_error"] = True

            self.add_assistant_message(message)
            self.add_user_message([tool_result])
            message = client.messages.create(**self._build_params())

        return message

    def chat(self):
        while True:
            user_input = input("User: ")
            if user_input.strip().lower() in {"exit", "quit"}:
                break
            if not user_input.strip():
                continue  # skip empty input — the API rejects empty messages

            self.add_user_message(user_input)
            print("...", end="", flush=True)  # show we're waiting on the API
            try:
                message = client.messages.create(**self._build_params())
                message = self._run_tools(message)
            except anthropic.APITimeoutError:
                print("\r[timed out reaching the API — check your connection]")
                self.messages.pop()  # drop the unanswered user turn
                continue
            except anthropic.APIError as exc:
                print(f"\r[API error: {exc}]")
                self.messages.pop()
                continue
            print("\r", end="")  # clear the "..." indicator

            self.add_assistant_message(message)

            # output_config.format guarantees the final turn's first block is
            # text containing JSON valid against the schema.
            text = next((b.text for b in message.content if b.type == "text"), "")
            if self.response_schema is None:
                print(f"Assistant: {text}")
                continue

            reply = self.response_schema.model_validate_json(text)
            print(f"Assistant: {reply.reply}")
            tag = f"{reply.resolution}{' · escalated' if reply.escalated else ''}"


def main():
    print("Starting chat...")

    Conversation(system_message=system_message).chat()


if __name__ == "__main__":
    main()
