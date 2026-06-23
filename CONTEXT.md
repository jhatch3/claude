# Support Agent

Domain language for a single-organization customer-support agent: structured Customer/Order records plus a RAG knowledge base of policies, FAQs, and past ticket transcripts.

## Language

**Customer**:
A person the support organization serves; the unit of data scoping for private content.
_Avoid_: tenant, account, user.

**Order**:
A purchase belonging to a Customer.
_Avoid_: transaction, purchase.

**Knowledge Base**:
The organization-wide corpus of policy and FAQ text, visible for every Customer.
_Avoid_: docs, corpus.

**Document** (a.k.a. **Chunk**):
A single embedded passage of retrievable text — one chunked piece of a larger source document.
_Avoid_: vector, record, embedding.

**Source**:
The kind of knowledge a Document holds — one of `policy`, `faq`, `transcript`.

**Transcript**:
A past support-ticket conversation, scoped privately to one Customer.

### Harness (multi-agent)

**Orchestrator**:
The hub agent. It never touches data directly — it delegates to Spokes and then asks the Synth agent for the final reply.

**Spoke**:
A specialist sub-agent the Orchestrator delegates to (Customer, Refund, Knowledge, History, Escalation), each with a focused toolset.
_Avoid_: worker, child agent.

**Synth agent**:
A tool-less agent that drafts the final customer-facing reply from the findings, quoting money/policy facts verbatim.

**Customer gate**:
The rule that money/history/escalation Spokes run only for a Customer the Customer Spoke has verified this conversation.

**Findings record**:
A Spoke's structured output `{agent, request, status, data, conclusion, tools_used}` collected for the Synth agent.

## Relationships

- A **Customer** has many **Orders** and many **Transcripts**
- A **Transcript** belongs to exactly one **Customer**; **Knowledge Base** content (policy/faq) belongs to no Customer (organization-wide)
- A **Document** has exactly one **Source**
- Retrieval over the **Knowledge Base** is unscoped; retrieval over **Transcripts** is always scoped to a single **Customer**

## Example dialogue

> **Dev:** "When we search past tickets, do we search across all Customers?"
> **Expert:** "Never — Transcript search is always scoped to the one verified Customer. Only Knowledge Base content is shared."

## Flagged ambiguities

- "tenant" was used for the 100s of scoped entities — resolved: these are **Customers** within a single organization (a single-tenant application), not separate orgs. Isolation is per-Customer row scoping, not organization-level multi-tenancy.
