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




def add_user_message(content, messages):
    messages.append({
        "role": "user",
        "content": content
    })
    return messages

def add_assistant_message(content, messages):
    messages.append({
        "role": "assistant",
        "content": content
    })
    return messages





def main():
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
                for event in stream.text_stream:
                        print(event, end="", flush=True)

            
            message = stream.get_final_message()
            print("\n")
            add_assistant_message(message.content[0].text, messages)
        except Exception as e:
            add_assistant_message(f"Error occurred while processing the request. {e}", messages)
            continue

        
    print(json.dumps(messages, indent=4))



if __name__ == "__main__":
    main()