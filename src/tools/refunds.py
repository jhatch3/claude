"""Refund tools + the over-threshold guard hooks (money path, Decimal end-to-end)."""
from decimal import Decimal

from src.db import db
from src.tools.registry import hook, tool

# Refunds at or below this amount are auto-approved; above it must escalate.
REFUND_THRESHOLD = Decimal("500.00")


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
