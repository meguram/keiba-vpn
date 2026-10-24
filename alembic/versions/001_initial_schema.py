"""Initial schema — Layer 1〜5 + predictions + scrape_runs."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')

    op.create_table(
        "courses",
        sa.Column("course_id", sa.String(20), primary_key=True),
        sa.Column("course_name", sa.String(50), nullable=False),
        sa.Column("region", sa.String(20)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_table(
        "sires",
        sa.Column("sire_id", sa.String(20), primary_key=True),
        sa.Column("sire_name", sa.String(100), nullable=False),
        sa.Column("sire_line", sa.String(50)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_table(
        "jockeys",
        sa.Column("jockey_id", sa.String(20), primary_key=True),
        sa.Column("jockey_name", sa.String(100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_table(
        "trainers",
        sa.Column("trainer_id", sa.String(20), primary_key=True),
        sa.Column("trainer_name", sa.String(100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_table(
        "horses",
        sa.Column("horse_id", sa.String(20), primary_key=True),
        sa.Column("horse_name", sa.String(100), nullable=False),
        sa.Column("sex", sa.String(5)),
        sa.Column("birth_year", sa.SmallInteger()),
        sa.Column("sire_id", sa.String(20), sa.ForeignKey("sires.sire_id")),
        sa.Column("dam_sire", sa.String(100)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("idx_horses_sire_id", "horses", ["sire_id"])

    op.create_table(
        "races",
        sa.Column("race_id", sa.String(20), primary_key=True),
        sa.Column("race_name", sa.String(200)),
        sa.Column("course", sa.String(20)),
        sa.Column("venue", sa.String(20)),
        sa.Column("surface", sa.String(10)),
        sa.Column("distance", sa.Integer()),
        sa.Column("direction", sa.String(10)),
        sa.Column("weather", sa.String(20)),
        sa.Column("track_condition", sa.String(10)),
        sa.Column("start_time", sa.Time()),
        sa.Column("race_date", sa.Date()),
        sa.Column("field_size", sa.SmallInteger()),
        sa.Column("grade", sa.String(20)),
        sa.Column("race_class", sa.String(100)),
        sa.Column("weight_rule", sa.String(50)),
        sa.Column("is_excluded", sa.Boolean(), server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index(
        "idx_races_filter_axes",
        "races",
        ["course", "surface", "track_condition", "race_class", "distance", "race_date"],
    )

    op.create_table(
        "entries",
        sa.Column("entry_id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("race_id", sa.String(20), sa.ForeignKey("races.race_id"), nullable=False),
        sa.Column("horse_id", sa.String(20), sa.ForeignKey("horses.horse_id"), nullable=False),
        sa.Column("post_no", sa.SmallInteger()),
        sa.Column("bracket_number", sa.SmallInteger()),
        sa.Column("jockey_id", sa.String(20), sa.ForeignKey("jockeys.jockey_id")),
        sa.Column("trainer_id", sa.String(20), sa.ForeignKey("trainers.trainer_id")),
        sa.Column("jockey_weight", sa.Numeric(4, 1)),
        sa.Column("weight", sa.SmallInteger()),
        sa.Column("weight_change", sa.SmallInteger()),
        sa.Column("sex_age", sa.String(10)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.UniqueConstraint("race_id", "horse_id", name="uq_entries_race_horse"),
    )
    op.create_index("idx_entries_horse_race", "entries", ["horse_id", "race_id"])

    op.create_table(
        "race_results",
        sa.Column("result_id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("race_id", sa.String(20), sa.ForeignKey("races.race_id"), nullable=False),
        sa.Column("horse_id", sa.String(20), sa.ForeignKey("horses.horse_id"), nullable=False),
        sa.Column("finish_pos", sa.SmallInteger()),
        sa.Column("finish_time_sec", sa.Numeric(7, 2)),
        sa.Column("margin", sa.String(20)),
        sa.Column("last_3f_sec", sa.Numeric(5, 2)),
        sa.Column("weight", sa.SmallInteger()),
        sa.Column("jockey_id", sa.String(20), sa.ForeignKey("jockeys.jockey_id")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.UniqueConstraint("race_id", "horse_id", name="uq_race_results_race_horse"),
    )
    op.create_index("idx_results_race_finish", "race_results", ["race_id", "finish_pos"])

    for name, entity_col in (
        ("horse_stats_snapshot", "horse_id"),
        ("jockey_stats_snapshot", "jockey_id"),
        ("trainer_stats_snapshot", "trainer_id"),
    ):
        cols = [
            sa.Column("snapshot_id", sa.BigInteger(), primary_key=True, autoincrement=True),
            sa.Column(entity_col, sa.String(20), nullable=False),
            sa.Column("as_of_race_id", sa.String(20), nullable=False),
            sa.Column("as_of_date", sa.Date(), nullable=False),
            sa.Column("win_rate_all", sa.Numeric(5, 4)),
            sa.Column("place_rate_all", sa.Numeric(5, 4)),
            sa.Column("show_rate_all", sa.Numeric(5, 4)),
            sa.Column("sample_count", sa.SmallInteger()),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
            sa.UniqueConstraint(entity_col, "as_of_race_id", name=f"uq_{name}_as_of"),
        ]
        if name == "horse_stats_snapshot":
            cols.extend([
                sa.Column("win_rate_turf", sa.Numeric(5, 4)),
                sa.Column("win_rate_dirt", sa.Numeric(5, 4)),
                sa.Column("win_rate_distance", sa.Numeric(5, 4)),
                sa.Column("win_rate_course", sa.Numeric(5, 4)),
                sa.Column("win_rate_going", sa.Numeric(5, 4)),
                sa.Column("avg_last_3f", sa.Numeric(5, 2)),
                sa.Column("speed_index_avg", sa.Numeric(6, 2)),
                sa.Column("speed_index_max", sa.Numeric(6, 2)),
                sa.Column("running_style_score", sa.Numeric(5, 2)),
            ])
        op.create_table(name, *cols)

    op.create_table(
        "race_lap_times",
        sa.Column("race_id", sa.String(20), primary_key=True),
        sa.Column("furlong_index", sa.SmallInteger(), primary_key=True),
        sa.Column("lap_time_sec", sa.Numeric(4, 2), nullable=False),
        sa.Column("cumulative_sec", sa.Numeric(6, 2)),
    )
    op.create_table(
        "race_corner_positions",
        sa.Column("race_id", sa.String(20), primary_key=True),
        sa.Column("horse_id", sa.String(20), primary_key=True),
        sa.Column("corner_1", sa.SmallInteger()),
        sa.Column("corner_2", sa.SmallInteger()),
        sa.Column("corner_3", sa.SmallInteger()),
        sa.Column("corner_4", sa.SmallInteger()),
    )
    op.create_table(
        "race_pace_summary",
        sa.Column("race_id", sa.String(20), primary_key=True),
        sa.Column("first_3f_sec", sa.Numeric(5, 2)),
        sa.Column("last_3f_sec", sa.Numeric(5, 2)),
        sa.Column("pace_category", sa.String(10)),
        sa.Column("front_runner_count", sa.SmallInteger()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.CheckConstraint("pace_category IN ('HIGH','MIDDLE','SLOW')", name="ck_pace_category"),
    )

    op.create_table(
        "race_odds_snapshot",
        sa.Column("snapshot_id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("race_id", sa.String(20), nullable=False),
        sa.Column("horse_id", sa.String(20), nullable=False),
        sa.Column("snapshot_type", sa.String(20), nullable=False),
        sa.Column("odds_value", sa.Numeric(7, 1), nullable=False),
        sa.Column("odds_place_low", sa.Numeric(7, 1)),
        sa.Column("odds_place_high", sa.Numeric(7, 1)),
        sa.Column("snapshot_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("race_id", "horse_id", "snapshot_type", "snapshot_at", name="uq_odds_snapshot"),
        sa.CheckConstraint(
            "snapshot_type IN ('WIN','PLACE','EXACTA','QUINELLA','WIDE')",
            name="ck_snapshot_type",
        ),
    )
    op.create_index("idx_odds_race_horse_time", "race_odds_snapshot", ["race_id", "horse_id", "snapshot_at"])

    op.create_table(
        "prediction_results",
        sa.Column("prediction_id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("race_id", sa.String(20), nullable=False),
        sa.Column("horse_id", sa.String(20), nullable=False),
        sa.Column("model_version", sa.String(50), nullable=False),
        sa.Column("predicted_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("win_prob", sa.Numeric(5, 4)),
        sa.Column("place_prob", sa.Numeric(5, 4)),
        sa.Column("show_prob", sa.Numeric(5, 4)),
        sa.Column("predicted_win_odds", sa.Numeric(7, 1)),
        sa.Column("predicted_place_odds", sa.Numeric(7, 1)),
        sa.Column("expected_win_roi", sa.Numeric(7, 2)),
        sa.Column("expected_show_roi", sa.Numeric(7, 2)),
        sa.Column("predicted_position", sa.SmallInteger()),
        sa.Column("predicted_running_style", sa.String(10)),
        sa.UniqueConstraint("race_id", "horse_id", "model_version", name="uq_prediction_race_horse_model"),
    )
    op.create_table(
        "prediction_lap_times",
        sa.Column("race_id", sa.String(20), primary_key=True),
        sa.Column("model_version", sa.String(50), primary_key=True),
        sa.Column("furlong_index", sa.SmallInteger(), primary_key=True),
        sa.Column("predicted_lap_sec", sa.Numeric(4, 2)),
        sa.Column("predicted_pace_cat", sa.String(10)),
        sa.CheckConstraint(
            "predicted_pace_cat IN ('HIGH','MIDDLE','SLOW')",
            name="ck_predicted_pace_cat",
        ),
    )

    op.create_table(
        "scrape_runs",
        sa.Column("run_id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("target_type", sa.String(30), nullable=False),
        sa.Column("target_id", sa.String(20), nullable=False),
        sa.Column("status", sa.String(10), nullable=False),
        sa.Column("retry_count", sa.SmallInteger(), server_default=sa.text("0")),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("error_message", sa.Text()),
        sa.Column("gcs_path", sa.Text()),
        sa.CheckConstraint("status IN ('SUCCESS','FAILED','RETRY')", name="ck_scrape_status"),
    )

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_table(
        "saved_analyses",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE")),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("analysis_type", sa.String(20), nullable=False),
        sa.Column("filter_conditions", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("last_run_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "analysis_type IN ('sire','course','jockey','trainer')",
            name="ck_analysis_type",
        ),
    )
    op.create_index(
        "idx_saved_analyses_params",
        "saved_analyses",
        ["filter_conditions"],
        postgresql_using="gin",
    )

    op.create_table(
        "course_stats_cache",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("track", sa.String(20), nullable=False),
        sa.Column("distance", sa.Integer(), nullable=False),
        sa.Column("surface", sa.String(10), nullable=False),
        sa.Column("track_condition", sa.String(10), nullable=False),
        sa.Column("stat_type", sa.String(30), nullable=False),
        sa.Column("stat_key", sa.String(50), nullable=False),
        sa.Column("n_runs", sa.Integer()),
        sa.Column("win_rate", sa.Numeric(5, 4)),
        sa.Column("place_rate", sa.Numeric(5, 4)),
        sa.Column("roi_win", sa.Numeric(7, 4)),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
        sa.UniqueConstraint(
            "track", "distance", "surface", "track_condition", "stat_type", "stat_key",
            name="uq_course_stats_cache",
        ),
    )


def downgrade() -> None:
    for table in (
        "course_stats_cache",
        "saved_analyses",
        "users",
        "scrape_runs",
        "prediction_lap_times",
        "prediction_results",
        "race_odds_snapshot",
        "race_pace_summary",
        "race_corner_positions",
        "race_lap_times",
        "trainer_stats_snapshot",
        "jockey_stats_snapshot",
        "horse_stats_snapshot",
        "race_results",
        "entries",
        "races",
        "horses",
        "trainers",
        "jockeys",
        "sires",
        "courses",
    ):
        op.drop_table(table)
