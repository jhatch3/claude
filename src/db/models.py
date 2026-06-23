"""
ORM models and the dialect-aware embedding column.

`embedding` is a pgvector `vector(EMBEDDING_DIM)` on Postgres and portable JSON
on SQLite (dev/test). Money is exact `Decimal`; audit timestamps are DB-side.
"""
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
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
    # Money as exact Decimal end-to-end. The JSON boundary (src/llm/client.py)
    # encodes Decimal as a number, so no float ever touches the amount in logic.
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

    `customer_id` is the scoping column: NULL means organization-wide Knowledge
    Base; a value scopes a private Transcript chunk.
    """

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String, index=True)  # policy | faq | transcript
    doc_id: Mapped[str] = mapped_column(String, index=True)  # logical document id
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    customer_id: Mapped[str | None] = mapped_column(
        ForeignKey("customers.customer_id"), nullable=True, index=True
    )
    embedding = mapped_column(_embedding_column_type())
    meta: Mapped[dict] = mapped_column(JSON, default=dict)

    __table_args__ = (
        UniqueConstraint("doc_id", "chunk_index", name="uq_documents_doc_chunk"),
        Index("ix_documents_source_customer", "source", "customer_id"),
    )
