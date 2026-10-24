"""add computation_status to megu_index

Revision ID: 005
Revises: 004
Create Date: 2026-07-10
"""
from alembic import op
import sqlalchemy as sa

revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.add_column(
        'megu_index',
        sa.Column('computation_status', sa.String(20), nullable=True, server_default='valid')
    )
    # 既存データを 'valid' で埋める
    op.execute("UPDATE megu_index SET computation_status = 'valid' WHERE computation_status IS NULL")
    # out_of_range（着差2秒超）は megu_index=NULL になるため NOT NULL 制約を削除
    for col in ['megu_index', 'finish_time_sec', 'adjusted_time_sec',
                'delta_pace_sec', 'delta_track_sec', 'delta_weight_sec', 'delta_level_sec']:
        op.alter_column('megu_index', col, nullable=True)

def downgrade() -> None:
    op.drop_column('megu_index', 'computation_status')

# NOTE: NOT NULL 制約の緩和も同 migration で管理（DBには手動適用済み）
# 以下は upgrade 関数に追加を推奨（既存環境との互換のため別ファイルも可）
