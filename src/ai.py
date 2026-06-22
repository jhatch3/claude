"""
Main file for the AI module

This module contains the core AI logic for the application.

"""
import anthropic
from dotenv import load_dotenv

load_dotenv()

# Globals
# ==================================
client = anthropic.Anthropic()
# ==================================


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
    ):
        self.messages = []
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_message = system_message

    def add_message(self, role, content):
        """Append a message after validating role and content.

        Centralizes the error logic so callers just hand over content; an empty
        or wrong-typed payload raises here instead of failing later at the API.
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")
        if content is None:
            raise ValueError("content must not be None")

        if isinstance(content, str):
            self.messages.append({"role": role, "content": content})
        else:
            # Not a string, so it's an API Message — store its content blocks.
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

    def chat(self):
        while True:
            user_input = input("User: ")
            if user_input.strip().lower() in {"exit", "quit"}:
                break

            self.add_user_message(user_input)
            message = client.messages.create(**self._build_params())
            self.add_assistant_message(message)
            print(f"Assistant: {message.content[0].text}")


def main():
    print("Starting chat...")
    Conversation().chat()


if __name__ == "__main__":
    main()
