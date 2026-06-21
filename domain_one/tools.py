"""
Tool registry and implementations for the AI module.

Register a tool with the @tool decorator; ai.py pulls the definitions via
get_tool_definitions() and dispatches calls via run_tool().
"""

# Maps tool name -> {"definition": <API schema>, "fn": <callable>}
tool_dic = {}


def tool(name, description, input_schema):
    """Register a function as a Claude tool.

    Stores the API-facing definition alongside the callable so run_tool can
    dispatch by name. Use as a decorator on the implementing function.
    """
    def decorator(fn):
        tool_dic[name] = {
            "definition": {
                "name": name,
                "description": description,
                "input_schema": input_schema,
            },
            "fn": fn,
        }
        return fn
    return decorator


def get_tool_definitions():
    """Return the list of tool definitions to pass to the API."""
    return [entry["definition"] for entry in tool_dic.values()]


def run_tool(tool_name, tool_input):
    """Dispatch a tool call to its registered implementation."""
    entry = tool_dic.get(tool_name)
    if entry is None:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        return entry["fn"](**tool_input)
    except Exception as exc:  # surface failures to the model rather than crashing
        return {"error": str(exc)}


# Tools
# ==================================
@tool(
    name="get_weather",
    description="Get the current weather for a given city.",
    input_schema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name, e.g. 'Paris'"},
        },
        "required": ["city"],
    },
)
def get_weather(city):
    # Stub implementation — replace with a real lookup.
    return {"city": city, "temperature_c": 21, "conditions": "sunny"}
