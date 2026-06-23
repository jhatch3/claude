"""
Tool + hook registry and dispatch.

Register a tool with the @tool decorator and a pre-call guard with @hook; the
harness pulls definitions via get_tool_definitions() and dispatches via run_tool().
"""

# Maps tool name -> {"definition": <API schema>, "fn": <callable>}
tool_dic = {}

# Maps tool name -> list of pre-call hooks.
hook_dic = {}


def hook(tool_name):
    """Register a pre-call hook for a tool.

    The hook receives the tool's input dict and returns either None to allow
    the call, or a dict to short-circuit it — the returned dict becomes the
    tool result, so the model sees the block reason instead of the tool ever
    running. Use as a decorator on the guard function.
    """
    def decorator(fn):
        hook_dic.setdefault(tool_name, []).append(fn)
        return fn
    return decorator


def tool(name, description, input_schema, strict=False):
    """Register a function as a Claude tool.

    Stores the API-facing definition alongside the callable so run_tool can
    dispatch by name. Use as a decorator on the implementing function.

    Pass strict=True to have the API guarantee tool_use.input validates exactly
    against input_schema (requires additionalProperties: false + required).
    """
    def decorator(fn):
        definition = {
            "name": name,
            "description": description,
            "input_schema": input_schema,
        }
        if strict:
            definition["strict"] = True
        tool_dic[name] = {"definition": definition, "fn": fn}
        return fn
    return decorator


def get_tool_definitions():
    """Return the list of tool definitions to pass to the API."""
    return [entry["definition"] for entry in tool_dic.values()]


def run_tool(tool_name, tool_input):
    """Dispatch a tool call to its registered implementation.

    Pre-call hooks run first; if any returns a non-None value, the tool is
    blocked and that value is returned as the result.
    """
    entry = tool_dic.get(tool_name)
    if entry is None:
        return {"error": f"Unknown tool: {tool_name}"}

    for guard in hook_dic.get(tool_name, []):
        decision = guard(tool_input)
        if decision is not None:
            return decision

    try:
        return entry["fn"](**tool_input)
    except Exception as exc:  # surface failures to the model rather than crashing
        return {"error": str(exc)}
