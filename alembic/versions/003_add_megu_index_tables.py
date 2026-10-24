"""Add megu_index, megu_par_time, megu_regression_params tables (AREA-11)."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "003_megu_index"
down_revision: Union[str, None] = "002_user_favorites_notifications"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "megu_par_time",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("distance", sa.Integer(), nullable=False),
        sa.Column("course", sa.String(20), nullable=False),
        sa.Column("surface", sa.String(10), nullable=False),
        sa.Column("track_condition", sa.String(10), nullable=False),
        sa.Column("par_time_sec", sa.Numeric(6, 2), nullable=False),
        sa.Column("par_front_split_sec", sa.Numeric(5, 2), nullable=True),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("model_version", sa.String(20), nullable=False, server_default="stg-v1"),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.UniqueConstraint("distance", "course", "surface", "track_condition", "model_version",
                            name="uq_megu_par_time"),
    )
    op.create_table(
        "megu_regression_params",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("param_name", sa.String(50), nullable=False),
        sa.Column("param_value", sa.Numeric(10, 6), nullable=False),
        sa.Column("std_error", sa.Numeric(10, 6), nullable=True),
        sa.Column("sample_count", sa.Integer(), nullable=True),
        sa.Column("model_version", sa.String(20), nullable=False, server_default="stg-v1"),
        sa.Column("fitted_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.UniqueConstraint("param_name", "model_version", name="uq_megu_reg_params"),
    )
    op.create_table(
        "megu_index",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("race_id", sa.String(20), nullable=False),
        sa.Column("horse_id", sa.String(20), nullable=False),
        sa.Column("finish_time_sec", sa.Numeric(6, 2), nullable=False),
        sa.Column("par_time_sec", sa.Numeric(6, 2), nullable=True),
        sa.Column("delta_pace_sec", sa.Numeric(5, 3), nullable=False, server_default="0"),
        sa.Column("delta_track_sec", sa.Numeric(5, 3), nullable=False, server_default="0"),
        sa.Column("delta_weight_sec", sa.Numeric(5, 3), nullable=False, server_default="0"),
        sa.Column("delta_level_sec", sa.Numeric(5, 3), nullable=False, server_default="0"),
        sa.Column("adjusted_time_sec", sa.Numeric(6, 2), nullable=False),
        sa.Column("megu_index", sa.Numeric(6, 1), nullable=False),
        sa.Column("field_quality", sa.Numeric(14, 0), nullable=True),
        sa.Column("model_version", sa.String(20), nullable=False, server_default="stg-v1"),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.UniqueConstraint("race_id", "horse_id", "model_version", name="uq_megu_index"),
    )
    op.create_index("idx_megu_index_horse", "megu_index", ["horse_id", "computed_at"])
    op.create_index("idx_megu_index_race", "megu_index", ["race_id"])


def downgrade() -> None:
    op.drop_index("idx_megu_index_race", table_name="megu_index")
    op.drop_index("idx_megu_index_horse", table_name="megu_index")
    op.drop_table("megu_index")
    op.drop_table("megu_regression_params")
    op.drop_table("megu_par_time")
