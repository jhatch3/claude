"""
Harness configuration: the spoke roster, prompts, the gate set, and the
orchestrator's delegation tool definitions.
"""
from src.tools import get_tool_definitions

MODEL = "claude-sonnet-4-6"
MAX_SPOKE_ITERS = 6  # per-spoke loop cap; the orchestrator turn is intentionally uncapped

# Spoke roster: name -> (system prompt, the tool names it may use).
# Tools are shared across spokes by design; get_customer is the gate's source.
SPOKES = {
    "customer": (
        "You verify customer identity and fetch account info. Use get_customer "
        "to find the customer by email/phone/id and lookup_order for their "
        "orders. Report the verified customer_id and a short summary. If several "
        "customers match, say so and ask for another identifier.",
        ["get_customer", "lookup_order"],
    ),
    "refund": (
        "You handle refunds for an ALREADY-VERIFIED customer. Use lookup_order "
        "to check the order, then process_refund (partial) or refund_order "
        "(full). If a refund is blocked for being over the limit, use "
        "escalate_to_human. State the outcome (refund id, amount, status) plainly.",
        ["lookup_order", "process_refund", "refund_order", "escalate_to_human"],
    ),
    "knowledge": (
        "You answer policy and FAQ questions. Use search_policies for rules "
        "(refunds, shipping, returns) and search_faqs for how-to questions. "
        "Quote the relevant passage. If nothing relevant is found, say so.",
        ["search_policies", "search_faqs", "escalate_to_human"],
    ),
    "history": (
        "You look up a verified customer's past tickets. Use search_past_tickets "
        "(and lookup_order / get_customer for context). Summarize relevant prior "
        "issues; if none, say so.",
        ["search_past_tickets", "lookup_order", "get_customer"],
    ),
    "escalation": (
        "You hand a verified customer's case to a human via escalate_to_human "
        "with a structured summary (customer_id, root cause, amount, recommended "
        "action). Confirm the ticket id.",
        ["escalate_to_human", "lookup_order"],
    ),
}

# Spokes that touch money/private history require a verified customer_id.
GATED = {"refund", "history", "escalation"}

# Map delegation tool name -> spoke name.
DELEGATION_TO_SPOKE = {
    "ask_customer_agent": "customer",
    "ask_refund_agent": "refund",
    "ask_knowledge_agent": "knowledge",
    "ask_history_agent": "history",
    "ask_escalation_agent": "escalation",
}

ORCHESTRATOR_PROMPT = (
    "You are the orchestrator of a customer-support system. You never call data "
    "tools directly — you delegate to specialist agents, then call "
    "draft_response to produce the final reply.\n\n"
    "Rules:\n"
    "1. For ANY request about a specific customer's account, orders, refunds, or "
    "past tickets, you MUST first call ask_customer_agent to verify them. It "
    "returns a verified customer_id.\n"
    "2. ask_refund_agent, ask_history_agent and ask_escalation_agent REQUIRE a "
    "verified customer_id — pass the one from step 1. They are blocked otherwise.\n"
    "3. For policy or FAQ questions, call ask_knowledge_agent (no verification "
    "needed).\n"
    "4. Put all relevant details (emails, order ids, the question) into the "
    "`task` string you pass to each agent.\n"
    "5. When you have gathered what you need, call draft_response. Do NOT write "
    "the final answer yourself."
)

SYNTH_PROMPT = (
    "You draft the final reply to a customer from a JSON list of findings, each "
    "produced by a specialist agent. Rules:\n"
    "- Reproduce every money amount, refund id, refund decision, and quoted "
    "policy text EXACTLY as it appears in a finding's `data` — never alter, "
    "round, or invent figures.\n"
    "- Use each finding's `conclusion` for narrative.\n"
    "- If a finding's status is 'blocked' or 'error', do not pretend the action "
    "succeeded.\n"
    "- Write one warm, concise, plain-text message to the customer. Output only "
    "that message."
)


def tool_subset(names):
    """The global tool definitions filtered to the given names."""
    by_name = {d["name"]: d for d in get_tool_definitions()}
    return [by_name[n] for n in names]


def _delegation_tool(name, description, needs_customer_id=False):
    properties = {"task": {"type": "string", "description": "What to ask the agent to do"}}
    required = ["task"]
    if needs_customer_id:
        properties["customer_id"] = {
            "type": "string",
            "description": "Verified customer id from ask_customer_agent",
        }
        required.append("customer_id")
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        "strict": True,
    }


ORCHESTRATOR_TOOLS = [
    _delegation_tool(
        "ask_customer_agent",
        "Verify a customer's identity and fetch their account/orders. Call this "
        "FIRST for any account-specific request; it returns a verified customer_id.",
    ),
    _delegation_tool(
        "ask_refund_agent",
        "Issue a refund for a verified customer. Requires a verified customer_id.",
        needs_customer_id=True,
    ),
    _delegation_tool(
        "ask_knowledge_agent",
        "Answer a policy or FAQ question. No verification needed.",
    ),
    _delegation_tool(
        "ask_history_agent",
        "Search a verified customer's past tickets. Requires a verified customer_id.",
        needs_customer_id=True,
    ),
    _delegation_tool(
        "ask_escalation_agent",
        "Escalate a verified customer's case to a human. Requires a verified customer_id.",
        needs_customer_id=True,
    ),
    {
        "name": "draft_response",
        "description": "Write the final user-facing reply from everything gathered "
        "so far. Call this LAST, once per turn.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
]
