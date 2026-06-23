"""
Multi-agent harness (hub-and-spoke, sub-agent-as-tool).

An orchestrator delegates to five specialist spokes (Customer / Refund /
Knowledge / History / Escalation) and a tool-less Synth agent that drafts the
final reply. See docs/adr/0002-multi-agent-harness.md for the design and the
honest "single agent is the production recommendation" caveat.

Run interactively:  python -m src.harness   (needs ANTHROPIC_API_KEY)
"""
import json

import anthropic

from src.ai import Conversation, _json_default, client
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


def _tool_subset(names):
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

# Map delegation tool name -> spoke name.
_DELEGATION_TO_SPOKE = {
    "ask_customer_agent": "customer",
    "ask_refund_agent": "refund",
    "ask_knowledge_agent": "knowledge",
    "ask_history_agent": "history",
    "ask_escalation_agent": "escalation",
}


class Harness:
    """Runs the orchestrator loop, the spokes, the gate, and the Synth."""

    def __init__(self, model=MODEL):
        self.model = model
        self.verified = set()  # verified customer_ids (persists across turns)
        self.findings = []  # structured findings buffer (reset each user turn)
        self.orchestrator = Conversation(
            model=model, system_message=ORCHESTRATOR_PROMPT, tools=ORCHESTRATOR_TOOLS
        )

    # Spokes
    # ------------------------------------------------------------------
    def _run_spoke(self, name, task, customer_id=None):
        """Run a spoke to completion and build its structured findings record."""
        prompt, tool_names = SPOKES[name]
        full_task = task
        if customer_id is not None:
            full_task = f"{task}\n\nVerified customer_id: {customer_id}"
        spoke = Conversation(
            model=self.model, system_message=prompt, tools=_tool_subset(tool_names)
        )
        conclusion, trace, capped = spoke.run_task(full_task, max_iters=MAX_SPOKE_ITERS)

        status = "ok"
        if capped:
            status = "error"
        elif any(
            isinstance(t["result"], dict) and "error" in t["result"] for t in trace
        ):
            status = "error"

        return {
            "agent": name,
            "request": task,
            "status": status,
            "data": [t["result"] for t in trace],  # verbatim structured tool outputs
            "conclusion": conclusion,
            "tools_used": [t["tool"] for t in trace],
        }

    def _record_and_ack(self, record):
        """Append to the findings buffer; return a lean ack for the orchestrator."""
        self.findings.append(record)
        return {
            "agent": record["agent"],
            "status": record["status"],
            "conclusion": record["conclusion"],
        }

    # Delegation dispatch (the orchestrator's "tools")
    # ------------------------------------------------------------------
    def _dispatch(self, name, tool_input):
        spoke = _DELEGATION_TO_SPOKE.get(name)
        if spoke is None:
            return {"error": f"Unknown delegation tool: {name}"}

        # The gate: money/history/escalation require a verified customer_id.
        if spoke in GATED:
            customer_id = tool_input.get("customer_id")
            if customer_id not in self.verified:
                return self._record_and_ack(
                    {
                        "agent": spoke,
                        "request": tool_input.get("task", ""),
                        "status": "blocked",
                        "data": [],
                        "conclusion": (
                            f"Blocked: customer_id {customer_id!r} is not verified. "
                            "Call ask_customer_agent first."
                        ),
                        "tools_used": [],
                    }
                )

        record = self._run_spoke(
            spoke, tool_input["task"], customer_id=tool_input.get("customer_id")
        )

        # The Customer spoke is the gate's source: a single-match verification
        # marks that customer_id verified for the rest of the conversation.
        if spoke == "customer":
            for result in record["data"]:
                if isinstance(result, dict) and isinstance(result.get("customer"), dict):
                    self.verified.add(result["customer"]["customer_id"])

        return self._record_and_ack(record)

    # Synth
    # ------------------------------------------------------------------
    def _draft_response(self):
        synth = Conversation(model=self.model, system_message=SYNTH_PROMPT, tools=[])
        payload = json.dumps(self.findings, default=_json_default, indent=2)
        draft, _, _ = synth.run_task(
            "Draft the customer-facing reply from these findings:\n\n" + payload,
            max_iters=0,
        )
        return draft

    # Orchestrator loop
    # ------------------------------------------------------------------
    def handle(self, user_message):
        """Process one user turn; return the final reply."""
        self.findings = []  # per-turn buffer
        orch = self.orchestrator
        orch.add_user_message(user_message)

        while True:
            message = client.messages.create(**orch._build_params())

            if message.stop_reason != "tool_use":
                # Orchestrator answered directly (e.g. a greeting) — relay it.
                orch.add_assistant_message(message)
                return next((b.text for b in message.content if b.type == "text"), "")

            tool_use = next(b for b in message.content if b.type == "tool_use")
            orch.add_assistant_message(message)

            if tool_use.name == "draft_response":
                # Terminal: the Synth drafts; we relay it verbatim (no orchestrator
                # post-processing) so money/policy figures can't be paraphrased.
                draft = self._draft_response()
                orch.add_user_message(
                    [{"type": "tool_result", "tool_use_id": tool_use.id, "content": draft}]
                )
                return draft

            result = self._dispatch(tool_use.name, dict(tool_use.input))
            orch.add_user_message(
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": json.dumps(result, default=_json_default),
                    }
                ]
            )

    def chat(self):
        print("Support harness ready (type 'exit' to quit).")
        while True:
            user_input = input("User: ")
            if user_input.strip().lower() in {"exit", "quit"}:
                break
            if not user_input.strip():
                continue
            print("...", end="", flush=True)
            try:
                reply = self.handle(user_input)
            except anthropic.APIError as exc:
                print(f"\r[API error: {exc}]")
                continue
            print(f"\rAssistant: {reply}")


def main():
    Harness().chat()


if __name__ == "__main__":
    main()
