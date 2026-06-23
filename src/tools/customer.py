"""Customer identity + order lookup tools (exact SQL, not RAG)."""
from src.db import db
from src.tools.registry import tool


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
