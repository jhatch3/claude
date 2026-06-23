"""
The Database access object — engine, sessions, and all record/vector queries.

Targets Postgres in production (DATABASE_URL=postgresql+psycopg://...); defaults
to a local SQLite file so the app runs without a Postgres server in development.
Data access lives here so the tools layer stays free of SQL.
"""
import os
from contextlib import contextmanager

import numpy as np
from sqlalchemy import cast, create_engine, func, literal, or_, select
from sqlalchemy.orm import sessionmaker

from src.db.models import EMBEDDING_DIM, Base, Customer, Document, Order


class Database:
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

    def init_db(self):
        """Create all tables (idempotent)."""
        Base.metadata.create_all(self.engine)

    # Record queries
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
