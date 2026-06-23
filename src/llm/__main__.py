"""`python -m src.llm` — the single-agent interactive chat."""
from src.llm.conversation import Conversation
from src.prompts.system import SYSTEM_MESSAGE

print("Starting chat...")
Conversation(system_message=SYSTEM_MESSAGE).chat()
