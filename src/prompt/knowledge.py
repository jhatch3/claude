"""
RAG corpus: the source text ingested into the vector store.

POLICY_DOCS / FAQS are organization-wide Knowledge Base content (no Customer
scoping). TRANSCRIPTS are past tickets, each scoped to one Customer. Consumed by
src/ingest.py.
"""

# Knowledge: policies + FAQs (shared, no Customer scoping).
# ==================================
POLICY_DOCS = {
    "refund-policy": (
        "Refund policy. Customers may request a refund within 30 days of "
        "delivery. Orders refunded in full are auto-approved up to $500; "
        "refunds above $500 require a human agent to approve. Items must be "
        "returned in original condition unless the item arrived damaged."
    ),
    "shipping-policy": (
        "Shipping policy. Standard shipping takes 3-5 business days. Express "
        "shipping arrives in 1-2 business days. We ship to the US and Canada. "
        "Lost packages are reshipped at no cost after a 7 day carrier window."
    ),
}

FAQS = {
    "faq-track-order": (
        "How do I track my order? Once an order ships you receive a tracking "
        "number by email. Delivered orders show a delivery date in your account."
    ),
    "faq-damaged-item": (
        "My item arrived damaged. Contact support with your order id; damaged "
        "items qualify for a full refund or replacement regardless of the "
        "30 day window."
    ),
}

# Past ticket transcripts (scoped to a Customer via the customer_id column).
# ==================================
TRANSCRIPTS = [
    {
        "doc_id": "ticket-9001",
        "customer_id": "cust_001",
        "text": (
            "Customer cust_001 reported the 4K monitor (ord_1002) had a dead "
            "pixel. Agent offered a replacement; customer asked for a refund "
            "instead. Escalated because the amount exceeded the refund limit."
        ),
    },
    {
        "doc_id": "ticket-9002",
        "customer_id": "cust_002",
        "text": (
            "Customer cust_002 asked about express shipping options for a "
            "future order. Agent explained 1-2 business day delivery."
        ),
    },
]
