"""
Tool registry and implementations for the AI module.

Register a tool with the @tool decorator; ai.py pulls the definitions via
get_tool_definitions() and dispatches calls via run_tool().
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
        "additionalProperties": False,
    },
    strict=True,
)
def get_weather(city):
    # Stub implementation — replace with a real lookup.
    return {"city": city, "temperature_c": 21, "conditions": "sunny"}


# Scenario-1: Customer Support
# ==================================
# In-memory mock data — stand-ins for real back-end systems.
_CUSTOMERS = [
    {
        "customer_id": "cust_001",
        "name": "Ada Lovelace",
        "status": "active",
        "email": "ada@example.com",
        "phone": "+1-555-0001",
    },
    {
        "customer_id": "cust_002",
        "name": "Alan Turing",
        "status": "active",
        "email": "alan@example.com",
        "phone": "+1-555-0002",
    },
]

_ORDERS = [
    {
        "order_id": "ord_1001",
        "customer_id": "cust_001",
        "items": ["Mechanical keyboard", "USB-C cable"],
        "status": "delivered",
        "total": 149.99,
        "order_date": "2026-05-01",
        "delivery_date": "2026-05-04",
    },
    {
        "order_id": "ord_1002",
        "customer_id": "cust_001",
        "items": ["4K monitor"],
        "status": "shipped",
        "total": 612.00,
        "order_date": "2026-06-10",
        "delivery_date": None,
    },
]

# Refunds at or below this amount are auto-approved; above it must escalate.
REFUND_THRESHOLD = 500.0


def _find_order(customer_id, order_id):
    """Return the order if it exists and belongs to the customer, else None."""
    return next(
        (
            o
            for o in _ORDERS
            if o["order_id"] == order_id and o["customer_id"] == customer_id
        ),
        None,
    )


@tool(
    name="get_customer",
    description=(
        "Look up and verify a customer by one or more identifiers. Provide at "
        "least one of email, phone, or customer_id. Returns the verified "
        "customer record on a single match, a list of matches when ambiguous "
        "(ask the user for another identifier), or an empty result on no match."
    ),
    input_schema={
        "type": "object",
        # Each identifier is optional, so under strict it's nullable + required:
        # the model must send all three keys but may pass null for unknowns.
        "properties": {
            "email": {
                "type": ["string", "null"],
                "description": "Customer email address, or null if unknown",
            },
            "phone": {
                "type": ["string", "null"],
                "description": "Customer phone number, or null if unknown",
            },
            "customer_id": {
                "type": ["string", "null"],
                "description": "Known customer id, or null if unknown",
            },
        },
        "required": ["email", "phone", "customer_id"],
        "additionalProperties": False,
    },
    strict=True,
)
def get_customer(email=None, phone=None, customer_id=None):
    if not any([email, phone, customer_id]):
        return {"error": "Provide at least one of: email, phone, customer_id."}

    matches = [
        c
        for c in _CUSTOMERS
        if (customer_id and c["customer_id"] == customer_id)
        or (email and c["email"].lower() == email.lower())
        or (phone and c["phone"] == phone)
    ]

    record_fields = ("customer_id", "name", "status")
    if len(matches) == 1:
        c = matches[0]
        return {"customer": {k: c[k] for k in record_fields}}
    if len(matches) > 1:
        return {
            "matches": [{k: c[k] for k in record_fields} for c in matches],
            "message": "Multiple customers matched; ask for another identifier.",
        }
    return {"customer": None, "message": "No matching customer found."}


@tool(
    name="lookup_order",
    description=(
        "List orders for a verified customer. Pass a verified customer_id; "
        "optionally narrow to a single order_id. An empty list is a valid "
        "result meaning the customer has no matching orders."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "customer_id": {
                "type": "string",
                "description": "Verified customer id (from get_customer)",
            },
            "order_id": {
                "type": ["string", "null"],
                "description": "Specific order id to fetch, or null for all orders",
            },
        },
        "required": ["customer_id", "order_id"],
        "additionalProperties": False,
    },
    strict=True,
)
def lookup_order(customer_id, order_id=None):
    orders = [o for o in _ORDERS if o["customer_id"] == customer_id]
    if order_id:
        orders = [o for o in orders if o["order_id"] == order_id]
    return {"orders": orders}


@tool(
    name="process_refund",
    description=(
        "Issue a refund against an order for a verified customer. Requires a "
        "verified customer_id and the order_id. Refunds over the threshold are "
        "blocked and must be escalated to a human instead."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "customer_id": {
                "type": "string",
                "description": "Verified customer id (from get_customer)",
            },
            "order_id": {"type": "string", "description": "Order to refund"},
            "amount": {
                "type": "number",
                "description": "Refund amount in dollars",
            },
        },
        "required": ["customer_id", "order_id", "amount"],
        "additionalProperties": False,
    },
    strict=True,  # money-critical: guarantee the input validates exactly
)
def process_refund(customer_id, order_id, amount):
    # Gate: the order must exist and belong to the verified customer.
    order = _find_order(customer_id, order_id)
    if order is None:
        return {"error": "No such order for this customer."}
    if amount <= 0:
        return {"error": "Refund amount must be positive."}
    if amount > order["total"]:
        return {"error": "Refund amount exceeds the order total."}

    # Note: the > REFUND_THRESHOLD guard lives in a pre-call hook
    # (_block_large_refunds), so by the time we get here the amount is approved.
    return {"refund_id": f"ref_{order_id}", "status": "approved", "amount": amount}


@tool(
    name="refund_order",
    description=(
        "Refund an order in full — the default refund. Use this when the "
        "customer wants their whole order refunded; only use process_refund "
        "when refunding a partial amount. Requires a verified customer_id and "
        "the order_id. Full refunds over the threshold are blocked and must be "
        "escalated to a human instead."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "customer_id": {
                "type": "string",
                "description": "Verified customer id (from get_customer)",
            },
            "order_id": {"type": "string", "description": "Order to refund in full"},
        },
        "required": ["customer_id", "order_id"],
        "additionalProperties": False,
    },
    strict=True,  # money-critical: guarantee the input validates exactly
)
def refund_order(customer_id, order_id):
    order = _find_order(customer_id, order_id)
    if order is None:
        return {"error": "No such order for this customer."}
    # Refund the whole order total; threshold guard lives in the pre-call hook.
    return {
        "refund_id": f"ref_{order_id}",
        "status": "approved",
        "amount": order["total"],
    }


def _refund_exceeds_threshold(amount):
    """Shared block decision for a refund amount over the threshold."""
    if amount > REFUND_THRESHOLD:
        return {
            "status": "blocked",
            "reason": (
                f"Amount ${amount:.2f} exceeds ${REFUND_THRESHOLD:.2f}; "
                "escalate to a human."
            ),
        }
    return None


@hook("process_refund")
def _block_large_refunds(tool_input):
    """Block partial refunds over the threshold before the tool runs."""
    return _refund_exceeds_threshold(tool_input.get("amount", 0))


@hook("refund_order")
def _block_large_full_refunds(tool_input):
    """Block full refunds over the threshold by resolving the order total."""
    order = _find_order(tool_input.get("customer_id"), tool_input.get("order_id"))
    if order is None:
        return None  # let the tool return the not-found error
    return _refund_exceeds_threshold(order["total"])


@tool(
    name="escalate_to_human",
    description=(
        "Hand the case to a human agent. Pass a structured summary — never a "
        "raw transcript — so the agent has the customer, root cause, amount, "
        "and recommended action at a glance."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "summary": {
                "type": "object",
                "description": "Structured case summary for the human agent",
                "properties": {
                    "customer_id": {"type": "string"},
                    "root_cause": {"type": "string"},
                    "amount": {
                        "type": ["number", "null"],
                        "description": "Dollar amount in question, or null if N/A",
                    },
                    "recommended_action": {"type": "string"},
                },
                "required": [
                    "customer_id",
                    "root_cause",
                    "amount",
                    "recommended_action",
                ],
                "additionalProperties": False,
            },
        },
        "required": ["summary"],
        "additionalProperties": False,
    },
    strict=True,
)
def escalate_to_human(summary):
    customer_id = summary.get("customer_id", "unknown")
    return {"ticket_id": f"tkt_{customer_id}", "status": "open"}
