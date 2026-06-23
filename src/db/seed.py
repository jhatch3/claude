"""Seed data and the seeding routine (run via `python -m src.db`)."""
from sqlalchemy import select

from src.db.database import db
from src.db.models import Customer, Order

_SEED_CUSTOMERS = [
    {
        "customer_id": "cust_001",
        "name": "Ada Lovelace",
        "status": "active",
        "email": "ada@example.com",
        "phone": "+1-555-0001",
    },
    {
        "customer_id": "cust_002",
        "name": "Alan Turing",
        "status": "active",
        "email": "alan@example.com",
        "phone": "+1-555-0002",
    },
]

_SEED_ORDERS = [
    {
        "order_id": "ord_1001",
        "customer_id": "cust_001",
        "items": ["Mechanical keyboard", "USB-C cable"],
        "status": "delivered",
        "total": 149.99,
        "order_date": "2026-05-01",
        "delivery_date": "2026-05-04",
    },
    {
        "order_id": "ord_1002",
        "customer_id": "cust_001",
        "items": ["4K monitor"],
        "status": "shipped",
        "total": 612.00,
        "order_date": "2026-06-10",
        "delivery_date": None,
    },
]


def seed():
    """Create tables and insert the seed data if the store is empty."""
    db.init_db()
    with db.session() as session:
        if session.scalar(select(Customer).limit(1)) is not None:
            return  # already seeded
        session.add_all(Customer(**c) for c in _SEED_CUSTOMERS)
        session.add_all(Order(**o) for o in _SEED_ORDERS)
