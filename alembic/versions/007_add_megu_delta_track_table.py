"""Add megu_delta_track table for date x venue x surface track correction storage."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "007_megu_delta_track"
down_revision: Union[str, None] = "006_megu_par_class_bucket"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "megu_delta_track",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("venue", sa.String(10), nullable=False),
        sa.Column("surface", sa.String(10), nullable=False),
        sa.Column(
            "delta_track_sec",
            sa.Numeric(6, 3),
            nullable=True,
            comment="馬場補正値(秒)。正=重馬場(タイム遅化)、負=軽馬場(タイム速化)。n_races<3のときNULL",
        ),
        sa.Column(
            "n_races",
            sa.Integer(),
            nullable=False,
            comment="delta_track_sec 算出に使用したレース数",
        ),
        sa.Column(
            "is_fallback",
            sa.Boolean(),
            nullable=False,
            server_default="false",
            comment="n_races < 3 のため delta_track_sec を 0 で代替したとき true",
        ),
        sa.Column("model_version", sa.String(20), nullable=False, server_default="stg-v1"),
        sa.Column("computed_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("date", "venue", "surface", "model_version", name="uq_megu_delta_track"),
    )
    op.create_index("idx_megu_delta_track_date_venue", "megu_delta_track", ["date", "venue"])


def downgrade() -> None:
    op.drop_index("idx_megu_delta_track_date_venue", table_name="megu_delta_track")
    op.drop_table("megu_delta_track")
