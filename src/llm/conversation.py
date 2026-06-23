"""
The Conversation class — single-agent message history, the tool-use loop, the
interactive REPL, and the programmatic run_task() used by the harness.
"""
import json

import anthropic

from src.llm.client import _json_default, client
from src.tools import get_tool_definitions, run_tool


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
    ):
        self.messages = []
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_message = system_message
        self.tools = tools if tools is not None else get_tool_definitions()

    def add_message(self, role, content):
        """Append a message after validating role and content."""
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
                "content": json.dumps(result, default=_json_default),
            }
            if isinstance(result, dict) and "error" in result:
                tool_result["is_error"] = True

            self.add_assistant_message(message)
            self.add_user_message([tool_result])
            message = client.messages.create(**self._build_params())

        return message

    def run_task(self, task, dispatch=run_tool, max_iters=6):
        """Run a single task to completion (non-interactive).

        Used by the multi-agent harness for sub-agents. Returns
        (final_text, trace, capped) where `trace` is a list of
        {tool, input, result} for each tool the agent executed, and `capped`
        is True if it hit max_iters while still wanting to call tools.
        `dispatch` lets the orchestrator route to delegation tools instead of
        the global run_tool. max_iters=None disables the cap.
        """
        self.add_user_message(task)
        trace = []
        message = client.messages.create(**self._build_params())
        iters = 0
        while message.stop_reason == "tool_use" and (
            max_iters is None or iters < max_iters
        ):
            tool_use = next(b for b in message.content if b.type == "tool_use")
            result = dispatch(tool_use.name, dict(tool_use.input))
            trace.append(
                {"tool": tool_use.name, "input": dict(tool_use.input), "result": result}
            )
            tool_result = {
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": json.dumps(result, default=_json_default),
            }
            if isinstance(result, dict) and "error" in result:
                tool_result["is_error"] = True
            self.add_assistant_message(message)
            self.add_user_message([tool_result])
            message = client.messages.create(**self._build_params())
            iters += 1

        self.add_assistant_message(message)
        final_text = next((b.text for b in message.content if b.type == "text"), "")
        capped = (
            max_iters is not None
            and iters >= max_iters
            and message.stop_reason == "tool_use"
        )
        return final_text, trace, capped

    def chat(self):
        while True:
            user_input = input("User: ")
            if user_input.strip().lower() in {"exit", "quit"}:
                break
            if not user_input.strip():
                continue  # skip empty input — the API rejects empty messages

            self.add_user_message(user_input)
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

            self.add_assistant_message(message)

            text = next((b.text for b in message.content if b.type == "text"), "")
            print(f"Assistant: {text}")
