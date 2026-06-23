"""
Tool package. Import surface: `from src.tools import run_tool, get_tool_definitions`.

Importing this package imports every domain module so their @tool / @hook
decorators run and register into the shared registry.
"""
from src.tools.registry import (
    get_tool_definitions,
    hook,
    hook_dic,
    run_tool,
    tool,
    tool_dic,
)

# Side-effect imports: registering the tools. Keep after the registry import.
from src.tools import customer, escalation, knowledge, refunds  # noqa: E402,F401

__all__ = [
    "tool",
    "hook",
    "run_tool",
    "get_tool_definitions",
    "tool_dic",
    "hook_dic",
]
