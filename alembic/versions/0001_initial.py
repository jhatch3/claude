"""initial schema: customers, orders, documents (+ pgvector HNSW on Postgres)

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-22
"""
from alembic import op

from src.db import Base

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"

    # The documents.embedding column is vector(N) on Postgres, so the extension
    # must exist before the tables are created.
    if is_postgres:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create all tables from the model metadata (customers, orders, documents)
    # including their declared indexes and constraints.
    Base.metadata.create_all(bind)

    # The HNSW vector index is Postgres-only DDL (kept out of the model so
    # SQLite create_all doesn't choke on `USING hnsw`).
    if is_postgres:
        op.create_index(
            "ix_documents_embedding_hnsw",
            "documents",
            ["embedding"],
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.drop_index("ix_documents_embedding_hnsw", table_name="documents")
    Base.metadata.drop_all(bind)
