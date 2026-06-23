"""Human escalation tool."""
from src.tools.registry import tool


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
