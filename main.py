"""
File main.py

This is the main file for the command line interface. It takes in user input and sends it to the model, then prints the response. It also handles errors and exits gracefully.

run main.py to start the command line interface. Type 'exit' or 'quit' to exit the program.

from project root directory, run the following command to start the command line interface:
>>> python3 -m main 
"""



messages = []

import anthropic
import json 
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic() 


sys = "Your are a agent the is tasked with anwering questions to the best of your ability. You are not allowed to ask for more information, you must answer the question with the information given. If you don't know the answer, say 'I don't know'. You will be given more information on this task by  delimiters <rules> </rules> <structured_output></structured_output> telling all the rules to follow and how to format your output. You will be given a prompt to answer after the delimiters."
rules = "You are not allowed to ask for more information, you must answer the question with the information given. If you don't know the answer, say 'I don't know'."
structured_output = "every response should be in pure text with the following format: 'your answer here' no new lines, no markdown, no code blocks, no quotes, no formatting, just pure text. If you don't know the answer, say 'I don't know'."

prompt = sys + "<rules>" + rules + "</rules>" + "<structured_output>" + structured_output + "</structured_output>"




def add_user_message(content: str, messages: list[dict]) -> list[dict]:
    """
    Takes in a message content and a list of messages, and adds the user message to the list then returns it.

    """
    
    messages.append({
        "role": "user",
        "content": content
    })
    return messages

def add_assistant_message(content: str, messages: list[dict]) -> list[dict]:
    """
    Takes in message content and a list of messages, and adds the assistant message to the list then returns it.

    """
    messages.append({
        "role": "assistant",
        "content": content
    })
    return messages





def main():
    """ Driver function for the command line interface. It takes in user input and sends it to the model, then prints the response. It also handles errors and exits gracefully. """
    
    while True:

        user_input = input("You: ")
        print("\n")
        
        if user_input.lower() in ["exit", "quit"]:
            break

        add_user_message(user_input, messages)
        
        try:
            with client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=300,
                temperature=0.01,
                messages=messages,
                system=prompt
            ) as stream:
                # Live Streaming of the response
                for event in stream.text_stream:
                        print(event, end="", flush=True)

            # get the final message from the stream and add it to the messages list
            message = stream.get_final_message()
            add_assistant_message(message.content[0].text, messages)
            print("\n")
        
        except Exception as e:
            add_assistant_message(f"Error occurred while processing the request. {e}", messages)
            break   
        
    print(json.dumps(messages, indent=4))



if __name__ == "__main__":
    main()