"""add front_split_sec and split_point_m to megu_index

Revision ID: 004
Revises: 003
Create Date: 2026-07-10
"""
from alembic import op
import sqlalchemy as sa

revision = '004'
down_revision = '003_megu_index'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('megu_index', sa.Column('front_split_sec', sa.Numeric(5, 2), nullable=True))
    op.add_column('megu_index', sa.Column('split_point_m', sa.SmallInteger(), nullable=True))
    op.add_column('megu_index', sa.Column('tsi_raw', sa.Numeric(6, 2), nullable=True))


def downgrade() -> None:
    op.drop_column('megu_index', 'tsi_raw')
    op.drop_column('megu_index', 'split_point_m')
    op.drop_column('megu_index', 'front_split_sec')
