"""
Main file for the AI module

This module contains the core AI logic for the application.

"""
import anthropic 
import os 
import json

from dotenv import load_dotenv
from domain_one.prompt.sys import system_prompt
from domain_one.tools import get_tool_definitions, run_tool

load_dotenv()

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
assert anthropic_api_key is not None, "ANTHROPIC_API_KEY environment variable is not set"

# Globals
# ==================================
client = anthropic.Anthropic()

messages = []
# ==================================


# Models in the Opus 4.7+ / Fable 5 family reject sampling params (temperature, etc.).
def _omits_sampling(model):
    return model.startswith(("claude-opus-4-7", "claude-opus-4-8", "claude-fable-5"))

def add_message(role, content):
    messages.append({"role": role, "content": content})

def add_user_message(content, messages):
    messages.append({"role": "user", "content": content})

def add_assistant_message(content, messages):
    messages.append({"role": "assistant", "content": content})

def chat(messages=messages, model="claude-sonnet-4-6", temperature=0.7, max_tokens=1024, system_message=None, tools=None):
    
    
    

    params = {
        "messages": messages,
        "model": model,
        "max_tokens": max_tokens,
    }

    if not _omits_sampling(model):
        params["temperature"] = temperature

    if tools:
        params["tools"] = tools
        params["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}

    if system_message:
        # System rendered as a content block so we can attach cache_control.
        params["system"] = [
            {
                "type": "text",
                "text": system_message,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    while True:
        user_input = input("User: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break

        add_user_message(user_input, messages)
        response = client.messages.create(**params)

        # Resolve any tool calls, then persist and print the final assistant turn.
        response = use_tools(messages, response, params)
        messages.append({"role": "assistant", "content": response.content})

        final_text = next(
            (block.text for block in response.content if block.type == "text"), ""
        )
        print(f"Assistant: {final_text}")


def use_tools(messages, response, params):
    """Resolve tool-use turns until the model returns a final answer.

    Reuses the caller's `params` (model, tools, system, etc.) so follow-up
    requests stay consistent with the original call. Returns the final response;
    the assistant turn for that response is appended by the caller.
    """
    while response.stop_reason == "tool_use":
        # disable_parallel_tool_use guarantees at most one tool_use block.
        tool_use = next(block for block in response.content if block.type == "tool_use")
        result = run_tool(tool_use.name, tool_use.input)

        messages.append({"role": "assistant", "content": response.content})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": json.dumps(result),
                    }
                ],
            }
        )

        response = client.messages.create(**params)

    return response


def main():
    
    tools = get_tool_definitions() 
      
    print("Starting chat...")
    chat(messages, system_message=system_prompt, tools=tools)


if __name__ == "__main__":
    main()