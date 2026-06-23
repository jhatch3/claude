"""
The Harness: orchestrator loop, spokes, the deterministic gate, the findings
buffer, and the Synth agent. See docs/adr/0002-multi-agent-harness.md.
"""
import json

import anthropic

from src.harness.agents import (
    DELEGATION_TO_SPOKE,
    GATED,
    MAX_SPOKE_ITERS,
    MODEL,
    ORCHESTRATOR_PROMPT,
    ORCHESTRATOR_TOOLS,
    SPOKES,
    SYNTH_PROMPT,
    tool_subset,
)
from src.llm.client import _json_default, client
from src.llm.conversation import Conversation


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
            model=self.model, system_message=prompt, tools=tool_subset(tool_names)
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
        spoke = DELEGATION_TO_SPOKE.get(name)
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
