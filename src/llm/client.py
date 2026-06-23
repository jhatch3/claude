"""Anthropic client and the JSON encoding helper for Decimal money."""
from decimal import Decimal

import anthropic
from dotenv import load_dotenv

load_dotenv()

# 60s timeout (vs the 10-min default) so a stuck network surfaces quickly
# instead of looking like a freeze.
client = anthropic.Anthropic(timeout=60)


def _json_default(obj):
    """Let json.dumps emit Decimal money as a JSON number."""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
