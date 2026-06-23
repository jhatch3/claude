# Multi-agent hub-and-spoke harness (chosen for demonstration)

## Context

The support agent works as a single Conversation with ~9 tools. We wanted a multi-agent harness: an orchestrator delegating to specialist sub-agents.

## Decision

Build `src/harness.py` as a **hub-and-spoke, sub-agent-as-tool** system: an **orchestrator** whose tools are five domain **spokes** (Customer / Refund / Knowledge / History / Escalation) plus a tool-less **Synth** agent that drafts the final reply. Each spoke is a `Conversation(system_message, tools=<subset>)` run via `Conversation.run_task()` (6-loop cap). Customer verification is a deterministic **gate**: a per-conversation verified-customer set, and the money/history/escalation delegation tools are blocked unless their `customer_id` is verified. Spokes report **structured findings records** `{agent, request, status, data (verbatim), conclusion, tools_used}` into a per-turn buffer; the Synth reads the buffer and drafts, quoting money/policy from `data` exactly; the orchestrator relays the draft verbatim.

## Status

Accepted as a **demonstration / portfolio** build — explicitly *not* the production recommendation.

## Considered Options

- **Single agent (the production-correct choice).** Anthropic's first agent principle is "start simple"; one agent with 9 well-described tools handles this sequential support flow fine, at a fraction of the tokens. This remains the recommended way to *ship* this product.
- **Deterministic router** (classify → hand to one specialist) — rejected: can't compose across the verification gate (a "refund my monitor" turn needs verify *then* refund).
- **Managed Agents coordinator** — rejected: different infra (sessions/environments), abandons the existing custom loop.

## Consequences

- This deliberately violates "start simple." The task is **sequential, not parallel**, so we take on multi-agent's coordination cost and ~order-of-magnitude higher token usage **without** its main benefit (parallel exploration). Acceptable only because the goal is learning, not production economics.
- No evaluation harness yet — the biggest missing best practice for an agent system.
- The gate, isolated spoke context, structured/verbatim handoffs, bounded sub-agent loops, and failure→escalation are all sound; the *internal mechanics* follow best practice even though the *decision to go multi-agent* does not.
- The single-agent `ai.py` `chat()` is kept intact as the recommended production path.
