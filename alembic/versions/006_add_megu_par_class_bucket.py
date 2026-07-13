"""Add class_bucket to megu_par_time for v2 par_time cells."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "006_megu_par_class_bucket"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "megu_par_time",
        sa.Column("class_bucket", sa.String(10), nullable=False, server_default=""),
    )
    op.drop_constraint("uq_megu_par_time", "megu_par_time", type_="unique")
    op.create_unique_constraint(
        "uq_megu_par_time",
        "megu_par_time",
        ["distance", "course", "surface", "track_condition", "class_bucket", "model_version"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_megu_par_time", "megu_par_time", type_="unique")
    op.drop_column("megu_par_time", "class_bucket")
    op.create_unique_constraint(
        "uq_megu_par_time",
        "megu_par_time",
        ["distance", "course", "surface", "track_condition", "model_version"],
    )
