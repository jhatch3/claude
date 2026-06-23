"""
Tool registry and implementations for the AI module.

Register a tool with the @tool decorator; ai.py pulls the definitions via
get_tool_definitions() and dispatches calls via run_tool().
"""
from decimal import Decimal

from src.db import db
from src.embeddings import get_embedder

# Embedder used to vectorize search queries (matches what ingestion used).
_embedder = get_embedder()

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
# Records live in the relational store (src/db.py). Run `python -m src.db`
# once to create and seed the tables. Queries go through the `db` object.

# Refunds at or below this amount are auto-approved; above it must escalate.
REFUND_THRESHOLD = Decimal("500.00")


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

    matches = db.find_customers(email=email, phone=phone, customer_id=customer_id)

    if len(matches) == 1:
        return {"customer": matches[0].to_record()}
    if len(matches) > 1:
        return {
            "matches": [c.to_record() for c in matches],
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
    orders = db.list_orders(customer_id, order_id)
    return {"orders": [o.to_dict() for o in orders]}


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
    # Convert the model's JSON number to Decimal via str so no float artifact
    # leaks into the money comparison (Decimal(str(149.99)) == 149.99 exactly).
    amount = Decimal(str(amount))
    # Gate: the order must exist and belong to the verified customer.
    order = db.find_order(customer_id, order_id)
    if order is None:
        return {"error": "No such order for this customer."}
    if amount <= 0:
        return {"error": "Refund amount must be positive."}
    if amount > order.total:
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
    order = db.find_order(customer_id, order_id)
    if order is None:
        return {"error": "No such order for this customer."}
    # Refund the whole order total; threshold guard lives in the pre-call hook.
    return {
        "refund_id": f"ref_{order_id}",
        "status": "approved",
        "amount": order.total,
    }


def _refund_exceeds_threshold(amount):
    """Shared block decision for a refund amount (Decimal) over the threshold."""
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
    return _refund_exceeds_threshold(Decimal(str(tool_input.get("amount", 0))))


@hook("refund_order")
def _block_large_full_refunds(tool_input):
    """Block full refunds over the threshold by resolving the order total."""
    order = db.find_order(tool_input.get("customer_id"), tool_input.get("order_id"))
    if order is None:
        return None  # let the tool return the not-found error
    return _refund_exceeds_threshold(order.total)


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


# RAG retrieval tools
# ==================================
def _format_hits(hits):
    """Turn (Document, score) pairs into JSON-serializable result dicts."""
    return [
        {
            "content": doc.content,
            "source": doc.source,
            "doc_id": doc.doc_id,
            "score": round(score, 4),
        }
        for doc, score in hits
    ]


def _semantic_search(query, sources):
    """Embed the query and return the top knowledge chunks for the given sources."""
    query_vec = _embedder.embed([query], input_type="query")[0]
    hits = db.search_documents(query_vec, top_k=4, sources=sources)
    return {"results": _format_hits(hits)}


# A single text-query schema reused by the knowledge-search tools.
_QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Natural-language question to search for",
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}


@tool(
    name="search_policies",
    description=(
        "Search company policy documents (refunds, shipping, returns) for "
        "relevant text. Call this whenever the customer asks how a policy works "
        "or what the rules are, instead of answering from memory. Returns the "
        "most relevant policy passages."
    ),
    input_schema=_QUERY_SCHEMA,
    strict=True,
)
def search_policies(query):
    return _semantic_search(query, ["policy"])


@tool(
    name="search_faqs",
    description=(
        "Search the FAQ knowledge base for relevant text. Call this for general "
        "'how do I...' questions (tracking an order, handling a damaged item, "
        "etc.) instead of answering from memory. Returns the most relevant FAQ "
        "passages."
    ),
    input_schema=_QUERY_SCHEMA,
    strict=True,
)
def search_faqs(query):
    return _semantic_search(query, ["faq"])


@tool(
    name="search_past_tickets",
    description=(
        "Search a verified customer's past support tickets for similar prior "
        "issues. Requires the verified customer_id; only that customer's "
        "transcripts are searched. Use it to check 'have we seen this before'."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "customer_id": {
                "type": "string",
                "description": "Verified customer id (from get_customer)",
            },
            "query": {
                "type": "string",
                "description": "Natural-language description of the issue",
            },
        },
        "required": ["customer_id", "query"],
        "additionalProperties": False,
    },
    strict=True,
)
def search_past_tickets(customer_id, query):
    query_vec = _embedder.embed([query], input_type="query")[0]
    hits = db.search_documents(
        query_vec, top_k=4, sources=["transcript"], customer_id=customer_id
    )
    return {"results": _format_hits(hits)}
