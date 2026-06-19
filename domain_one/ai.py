"""
Main file for the AI module

This module contains the core AI logic for the application.

"""
import anthropic 



# Globals
# ==================================
client = anthropic.Anthropic()

tool_dic = {}
messages = []
# ==================================

def add_message(role, content):
    messages.append({"role": role, "content": content})

def add_user_message(content, messages):
    messages.append({"role": "user", "content": content})

def add_assistant_message(content, messages):
    messages.append({"role": "assistant", "content": content})

def chat(messages=messages, model="claude-sonnet-4-7", temperature=0.7, max_tokens=1024, system_message=None):
    
    
    params = {
        "messages": messages,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
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

    

    while True:
        user_input = input("User: ")
        
        add_user_message(user_input, messages)
        message = client.messages.create(**params)
        add_assistant_message(message.content[0].text, messages)

def use_tool(tool_name, tool_input):
    # Placeholder for the tool usage functionality
    pass

def use_tools(messages):
    # Placeholder for the tool usage functionality
    pass



def main():
    print("Starting chat...")
    chat(messages)


if __name__ == "__main__":
    main()