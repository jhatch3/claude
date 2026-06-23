"""
Database layer — SQLAlchemy models and a Database access object.

Phase 0: the relational store for customer/order records. Targets Postgres in
production (set DATABASE_URL=postgresql+psycopg://user:pass@host/db); defaults
to a local SQLite file so the app runs without a Postgres server during
development. pgvector tables for RAG arrive in a later phase.

Run `python -m src.db` once to create the tables and seed the mock data.
"""
import os
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal

import numpy as np
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    cast,
    create_engine,
    func,
    literal,
    or_,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)
from sqlalchemy.types import JSON

# Embedding dimensionality is pinned to Voyage voyage-3.5 (1024). Changing the
# model means an ALTER on the vector column + a full re-embed.
EMBEDDING_DIM = 1024


def _embedding_column_type():
    """vector(N) on Postgres (pgvector), JSON elsewhere (SQLite dev/test)."""
    try:
        from pgvector.sqlalchemy import Vector

        return JSON().with_variant(Vector(EMBEDDING_DIM), "postgresql")
    except ImportError:
        return JSON()


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    """Audit columns set DB-side: when a row was created and last updated."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Customer(Base, TimestampMixin):
    __tablename__ = "customers"

    customer_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)
    email: Mapped[str] = mapped_column(String, index=True)
    phone: Mapped[str] = mapped_column(String, index=True)

    orders: Mapped[list["Order"]] = relationship(back_populates="customer")

    def to_record(self):
        """The verified-customer shape the tools expose to the model."""
        return {
            "customer_id": self.customer_id,
            "name": self.name,
            "status": self.status,
        }


class Order(Base, TimestampMixin):
    __tablename__ = "orders"

    order_id: Mapped[str] = mapped_column(String, primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        ForeignKey("customers.customer_id"), index=True
    )
    items: Mapped[list] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String)
    # Money as exact Decimal end-to-end. The JSON boundary (src/ai.py) encodes
    # Decimal as a number, so no float ever touches the amount in logic.
    total: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    order_date: Mapped[str] = mapped_column(String)
    delivery_date: Mapped[str | None] = mapped_column(String, nullable=True)

    customer: Mapped["Customer"] = relationship(back_populates="orders")

    def to_dict(self):
        """Plain dict the tools return; total stays Decimal (encoded at the wire)."""
        return {
            "order_id": self.order_id,
            "customer_id": self.customer_id,
            "items": self.items,
            "status": self.status,
            "total": self.total,
            "order_date": self.order_date,
            "delivery_date": self.delivery_date,
        }


class Document(Base, TimestampMixin):
    """A chunk of knowledge-base text plus its embedding (the RAG store).

    `embedding` is a pgvector `vector(EMBEDDING_DIM)` on Postgres (HNSW-indexed)
    and portable JSON on SQLite. `customer_id` is the scoping column: NULL means
    organization-wide Knowledge Base; a value scopes a private Transcript chunk.
    """

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String, index=True)  # policy | faq | transcript
    doc_id: Mapped[str] = mapped_column(String, index=True)  # logical document id
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    # Promoted to a first-class indexed column (NULL = org-wide). Scoping filters
    # MUST use this column, never JSON meta.
    customer_id: Mapped[str | None] = mapped_column(
        ForeignKey("customers.customer_id"), nullable=True, index=True
    )
    embedding = mapped_column(_embedding_column_type())
    meta: Mapped[dict] = mapped_column(JSON, default=dict)

    __table_args__ = (
        # Idempotent upserts target (doc_id, chunk_index).
        UniqueConstraint("doc_id", "chunk_index", name="uq_documents_doc_chunk"),
        # Composite B-tree for the scoping filter; the HNSW vector index is
        # created in the Alembic migration (Postgres-only DDL).
        Index("ix_documents_source_customer", "source", "customer_id"),
    )


class Database:
    """Owns the engine and session factory, and exposes the record queries.

    Data access lives here so the tools layer stays free of SQL. Construct with
    an explicit url for tests; otherwise it reads DATABASE_URL (SQLite default).
    """

    def __init__(self, url=None):
        self.url = url or os.getenv("DATABASE_URL", "sqlite:///support.db")
        # pool_pre_ping validates a connection before use (avoids "server closed
        # the connection" after a DB restart/idle timeout). The size knobs are
        # QueuePool-only, so skip them for SQLite.
        engine_kwargs = {"future": True, "pool_pre_ping": True}
        if not self.url.startswith("sqlite"):
            engine_kwargs.update(pool_size=5, max_overflow=10, pool_recycle=1800)
        self.engine = create_engine(self.url, **engine_kwargs)
        # expire_on_commit=False lets callers read column values after the
        # session closes (we hand detached ORM rows back to the tools).
        self._session_factory = sessionmaker(
            bind=self.engine, future=True, expire_on_commit=False
        )

    @contextmanager
    def session(self):
        """Short-lived session: commit on success, roll back on error."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # Schema / seeding
    # ------------------------------------------------------------------
    def init_db(self):
        """Create all tables (idempotent)."""
        Base.metadata.create_all(self.engine)

    def seed(self):
        """Create tables and insert the seed data if the store is empty."""
        self.init_db()
        with self.session() as session:
            if session.scalar(select(Customer).limit(1)) is not None:
                return  # already seeded
            session.add_all(Customer(**c) for c in _SEED_CUSTOMERS)
            session.add_all(Order(**o) for o in _SEED_ORDERS)

    # Queries
    # ------------------------------------------------------------------
    def find_customers(self, email=None, phone=None, customer_id=None):
        """Customers matching any supplied identifier (case-insensitive email)."""
        conditions = []
        if customer_id:
            conditions.append(Customer.customer_id == customer_id)
        if email:
            conditions.append(func.lower(Customer.email) == email.lower())
        if phone:
            conditions.append(Customer.phone == phone)
        if not conditions:
            return []
        with self.session() as session:
            return list(session.scalars(select(Customer).where(or_(*conditions))).all())

    def find_order(self, customer_id, order_id):
        """The order if it exists and belongs to the customer, else None."""
        with self.session() as session:
            return session.scalar(
                select(Order).where(
                    Order.order_id == order_id, Order.customer_id == customer_id
                )
            )

    def list_orders(self, customer_id, order_id=None):
        """All of a customer's orders, optionally narrowed to one order_id."""
        stmt = select(Order).where(Order.customer_id == customer_id)
        if order_id:
            stmt = stmt.where(Order.order_id == order_id)
        with self.session() as session:
            return list(session.scalars(stmt).all())

    # Vector store (RAG)
    # ------------------------------------------------------------------
    def reset_documents(self):
        """Delete all documents — lets ingestion re-run idempotently."""
        with self.session() as session:
            session.query(Document).delete()

    def add_documents(self, rows):
        """Bulk-insert document chunks (list of dicts matching Document columns)."""
        with self.session() as session:
            session.add_all(Document(**r) for r in rows)

    def search_documents(self, query_embedding, top_k=4, sources=None, customer_id=None):
        """Top-k chunks by cosine similarity, with source / Customer filters.

        This is the single scoping choke point: filters are applied in SQL, never
        in Python. On Postgres the ranking uses the pgvector cosine operator
        (HNSW-indexed); on SQLite it falls back to a numpy cosine scan. Returns a
        list of (Document, score).
        """
        filters = []
        if sources:
            filters.append(Document.source.in_(sources))
        if customer_id is not None:
            filters.append(Document.customer_id == customer_id)

        if self.engine.dialect.name == "postgresql":
            return self._search_pgvector(query_embedding, top_k, filters)
        return self._search_numpy(query_embedding, top_k, filters)

    def _search_pgvector(self, query_embedding, top_k, filters):
        from pgvector.sqlalchemy import Vector

        # Cosine distance via the `<=>` operator; ascending = most similar first.
        vector_literal = "[" + ",".join(repr(float(x)) for x in query_embedding) + "]"
        query_vec = cast(literal(vector_literal), Vector(EMBEDDING_DIM))
        distance = Document.embedding.op("<=>")(query_vec)
        stmt = (
            select(Document, (1 - distance).label("score"))
            .where(*filters)
            .order_by(distance)
            .limit(top_k)
        )
        with self.session() as session:
            return [(row[0], float(row[1])) for row in session.execute(stmt).all()]

    def _search_numpy(self, query_embedding, top_k, filters):
        stmt = select(Document)
        if filters:
            stmt = stmt.where(*filters)
        with self.session() as session:
            docs = list(session.scalars(stmt).all())

        query = np.asarray(query_embedding, dtype=np.float32)
        query_norm = float(np.linalg.norm(query)) or 1.0
        scored = []
        for doc in docs:
            vec = np.asarray(doc.embedding, dtype=np.float32)
            vec_norm = float(np.linalg.norm(vec)) or 1.0
            score = float(np.dot(query, vec) / (query_norm * vec_norm))
            scored.append((doc, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


# Default instance used by the tools layer (reads DATABASE_URL).
db = Database()


# Seed data — the mock records that used to live in src/tools.py.
# ==================================
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


if __name__ == "__main__":
    db.seed()
    print(f"Database ready at {db.url}")
