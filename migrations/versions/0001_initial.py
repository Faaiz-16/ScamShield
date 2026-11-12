"""Create analyses and feedback tables

Revision ID: 0001
Revises:
Create Date: 2026-11-12
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

MYSQL_OPTIONS = {
    "mysql_engine": "InnoDB",
    "mysql_charset": "utf8mb4",
    "mysql_collate": "utf8mb4_unicode_ci",
}


def upgrade() -> None:
    op.create_table(
        "analyses",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("public_id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("channel", sa.String(length=20), nullable=False),
        sa.Column("client_hash", sa.String(length=64), nullable=False),
        sa.Column("anonymized_text", sa.Text(), nullable=False),
        sa.Column("language", sa.String(length=10), nullable=False),
        sa.Column("response_language", sa.String(length=5), nullable=False),
        sa.Column("verdict", sa.String(length=20), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("scam_type", sa.String(length=40), nullable=True),
        sa.Column("rule_ids", sa.JSON(), nullable=False),
        sa.Column("link_domains", sa.JSON(), nullable=False),
        sa.Column("model_name", sa.String(length=40), nullable=True),
        sa.Column("scam_probability", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_analyses")),
        **MYSQL_OPTIONS,
    )
    op.create_index(op.f("ix_analyses_public_id"), "analyses", ["public_id"], unique=True)
    op.create_index(op.f("ix_analyses_created_at"), "analyses", ["created_at"])
    op.create_index(op.f("ix_analyses_client_hash"), "analyses", ["client_hash"])
    op.create_index(op.f("ix_analyses_verdict"), "analyses", ["verdict"])
    op.create_index(op.f("ix_analyses_scam_type"), "analyses", ["scam_type"])

    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("analysis_id", sa.Integer(), nullable=False),
        sa.Column("client_hash", sa.String(length=64), nullable=False),
        sa.Column("is_correct", sa.Boolean(), nullable=False),
        sa.Column("correct_label", sa.String(length=20), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["analysis_id"],
            ["analyses.id"],
            name=op.f("fk_feedback_analysis_id_analyses"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_feedback")),
        sa.UniqueConstraint("analysis_id", "client_hash", name=op.f("uq_feedback_analysis_id")),
        **MYSQL_OPTIONS,
    )
    op.create_index(op.f("ix_feedback_analysis_id"), "feedback", ["analysis_id"])


def downgrade() -> None:
    op.drop_index(op.f("ix_feedback_analysis_id"), table_name="feedback")
    op.drop_table("feedback")
    op.drop_index(op.f("ix_analyses_scam_type"), table_name="analyses")
    op.drop_index(op.f("ix_analyses_verdict"), table_name="analyses")
    op.drop_index(op.f("ix_analyses_client_hash"), table_name="analyses")
    op.drop_index(op.f("ix_analyses_created_at"), table_name="analyses")
    op.drop_index(op.f("ix_analyses_public_id"), table_name="analyses")
    op.drop_table("analyses")
