"""PostgreSQL ORM — Layer 1〜5 + 推論結果 + scrape_runs（AREA-03 / AREA-01）。"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Optional
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    Time,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ── Layer 1: 静的マスター ───────────────────────────────────────────────────


class Course(Base):
    __tablename__ = "courses"

    course_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    course_name: Mapped[str] = mapped_column(String(50), nullable=False)
    region: Mapped[Optional[str]] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Sire(Base):
    __tablename__ = "sires"

    sire_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    sire_name: Mapped[str] = mapped_column(String(100), nullable=False)
    sire_line: Mapped[Optional[str]] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Jockey(Base):
    __tablename__ = "jockeys"

    jockey_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    jockey_name: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Trainer(Base):
    __tablename__ = "trainers"

    trainer_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    trainer_name: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Horse(Base):
    __tablename__ = "horses"

    horse_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    horse_name: Mapped[str] = mapped_column(String(100), nullable=False)
    sex: Mapped[Optional[str]] = mapped_column(String(5))
    birth_year: Mapped[Optional[int]] = mapped_column(SmallInteger)
    sire_id: Mapped[Optional[str]] = mapped_column(String(20), ForeignKey("sires.sire_id"))
    dam_sire: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    sire: Mapped[Optional[Sire]] = relationship()


class Race(Base):
    __tablename__ = "races"
    __table_args__ = (
        Index("idx_races_filter_axes", "course", "surface", "track_condition", "race_class", "distance", "race_date"),
    )

    race_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    race_name: Mapped[Optional[str]] = mapped_column(String(200))
    course: Mapped[Optional[str]] = mapped_column(String(20))
    venue: Mapped[Optional[str]] = mapped_column(String(20))
    surface: Mapped[Optional[str]] = mapped_column(String(10))
    distance: Mapped[Optional[int]] = mapped_column(Integer)
    direction: Mapped[Optional[str]] = mapped_column(String(10))
    weather: Mapped[Optional[str]] = mapped_column(String(20))
    track_condition: Mapped[Optional[str]] = mapped_column(String(10))
    start_time: Mapped[Optional[time]] = mapped_column(Time)
    race_date: Mapped[Optional[date]] = mapped_column(Date)
    field_size: Mapped[Optional[int]] = mapped_column(SmallInteger)
    grade: Mapped[Optional[str]] = mapped_column(String(20))
    race_class: Mapped[Optional[str]] = mapped_column(String(100))
    weight_rule: Mapped[Optional[str]] = mapped_column(String(50))
    is_excluded: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    entries: Mapped[list["Entry"]] = relationship(back_populates="race")


class Entry(Base):
    __tablename__ = "entries"
    __table_args__ = (
        UniqueConstraint("race_id", "horse_id", name="uq_entries_race_horse"),
        Index("idx_entries_horse_race", "horse_id", "race_id"),
    )

    entry_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    race_id: Mapped[str] = mapped_column(String(20), ForeignKey("races.race_id"), nullable=False)
    horse_id: Mapped[str] = mapped_column(String(20), ForeignKey("horses.horse_id"), nullable=False)
    post_no: Mapped[Optional[int]] = mapped_column(SmallInteger)
    bracket_number: Mapped[Optional[int]] = mapped_column(SmallInteger)
    jockey_id: Mapped[Optional[str]] = mapped_column(String(20), ForeignKey("jockeys.jockey_id"))
    trainer_id: Mapped[Optional[str]] = mapped_column(String(20), ForeignKey("trainers.trainer_id"))
    jockey_weight: Mapped[Optional[float]] = mapped_column(Numeric(4, 1))
    weight: Mapped[Optional[int]] = mapped_column(SmallInteger)
    weight_change: Mapped[Optional[int]] = mapped_column(SmallInteger)
    sex_age: Mapped[Optional[str]] = mapped_column(String(10))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    race: Mapped[Race] = relationship(back_populates="entries")
    horse: Mapped[Horse] = relationship()


# ── Layer 2: 確定結果 ───────────────────────────────────────────────────────


class RaceResult(Base):
    __tablename__ = "race_results"
    __table_args__ = (
        UniqueConstraint("race_id", "horse_id", name="uq_race_results_race_horse"),
        Index("idx_results_race_finish", "race_id", "finish_pos"),
    )

    result_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    race_id: Mapped[str] = mapped_column(String(20), ForeignKey("races.race_id"), nullable=False)
    horse_id: Mapped[str] = mapped_column(String(20), ForeignKey("horses.horse_id"), nullable=False)
    finish_pos: Mapped[Optional[int]] = mapped_column(SmallInteger)
    finish_time_sec: Mapped[Optional[float]] = mapped_column(Numeric(7, 2))
    margin: Mapped[Optional[str]] = mapped_column(String(20))
    last_3f_sec: Mapped[Optional[float]] = mapped_column(Numeric(5, 2))
    weight: Mapped[Optional[int]] = mapped_column(SmallInteger)
    jockey_id: Mapped[Optional[str]] = mapped_column(String(20), ForeignKey("jockeys.jockey_id"))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


# ── Layer 3: 集計スナップショット ───────────────────────────────────────────


class HorseStatsSnapshot(Base):
    __tablename__ = "horse_stats_snapshot"

    snapshot_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    horse_id: Mapped[str] = mapped_column(String(20), nullable=False)
    as_of_race_id: Mapped[str] = mapped_column(String(20), nullable=False)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    win_rate_all: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    win_rate_turf: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    win_rate_dirt: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    place_rate_all: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    show_rate_all: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    win_rate_distance: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    win_rate_course: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    win_rate_going: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    avg_last_3f: Mapped[Optional[float]] = mapped_column(Numeric(5, 2))
    speed_index_avg: Mapped[Optional[float]] = mapped_column(Numeric(6, 2))
    speed_index_max: Mapped[Optional[float]] = mapped_column(Numeric(6, 2))
    running_style_score: Mapped[Optional[float]] = mapped_column(Numeric(5, 2))
    sample_count: Mapped[Optional[int]] = mapped_column(SmallInteger)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (UniqueConstraint("horse_id", "as_of_race_id", name="uq_horse_stats_as_of"),)


class JockeyStatsSnapshot(Base):
    __tablename__ = "jockey_stats_snapshot"

    snapshot_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    jockey_id: Mapped[str] = mapped_column(String(20), nullable=False)
    as_of_race_id: Mapped[str] = mapped_column(String(20), nullable=False)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    win_rate_all: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    place_rate_all: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    show_rate_all: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    sample_count: Mapped[Optional[int]] = mapped_column(SmallInteger)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (UniqueConstraint("jockey_id", "as_of_race_id", name="uq_jockey_stats_as_of"),)


class TrainerStatsSnapshot(Base):
    __tablename__ = "trainer_stats_snapshot"

    snapshot_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    trainer_id: Mapped[str] = mapped_column(String(20), nullable=False)
    as_of_race_id: Mapped[str] = mapped_column(String(20), nullable=False)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    win_rate_all: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    place_rate_all: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    show_rate_all: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    sample_count: Mapped[Optional[int]] = mapped_column(SmallInteger)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (UniqueConstraint("trainer_id", "as_of_race_id", name="uq_trainer_stats_as_of"),)


# ── Layer 4: ラップ・ペース・コーナー ───────────────────────────────────────


class RaceLapTime(Base):
    __tablename__ = "race_lap_times"

    race_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    furlong_index: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    lap_time_sec: Mapped[float] = mapped_column(Numeric(4, 2), nullable=False)
    cumulative_sec: Mapped[Optional[float]] = mapped_column(Numeric(6, 2))


class RaceCornerPosition(Base):
    __tablename__ = "race_corner_positions"

    race_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    horse_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    corner_1: Mapped[Optional[int]] = mapped_column(SmallInteger)
    corner_2: Mapped[Optional[int]] = mapped_column(SmallInteger)
    corner_3: Mapped[Optional[int]] = mapped_column(SmallInteger)
    corner_4: Mapped[Optional[int]] = mapped_column(SmallInteger)


class RacePaceSummary(Base):
    __tablename__ = "race_pace_summary"
    __table_args__ = (
        CheckConstraint("pace_category IN ('HIGH','MIDDLE','SLOW')", name="ck_pace_category"),
    )

    race_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    first_3f_sec: Mapped[Optional[float]] = mapped_column(Numeric(5, 2))
    last_3f_sec: Mapped[Optional[float]] = mapped_column(Numeric(5, 2))
    pace_category: Mapped[Optional[str]] = mapped_column(String(10))
    front_runner_count: Mapped[Optional[int]] = mapped_column(SmallInteger)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


# ── Layer 5: オッズ時系列 ───────────────────────────────────────────────────


class RaceOddsSnapshot(Base):
    __tablename__ = "race_odds_snapshot"
    __table_args__ = (
        UniqueConstraint(
            "race_id", "horse_id", "snapshot_type", "snapshot_at",
            name="uq_odds_snapshot",
        ),
        CheckConstraint(
            "snapshot_type IN ('WIN','PLACE','EXACTA','QUINELLA','WIDE')",
            name="ck_snapshot_type",
        ),
        Index("idx_odds_race_horse_time", "race_id", "horse_id", "snapshot_at"),
    )

    snapshot_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    race_id: Mapped[str] = mapped_column(String(20), nullable=False)
    horse_id: Mapped[str] = mapped_column(String(20), nullable=False)
    snapshot_type: Mapped[str] = mapped_column(String(20), nullable=False)
    odds_value: Mapped[float] = mapped_column(Numeric(7, 1), nullable=False)
    odds_place_low: Mapped[Optional[float]] = mapped_column(Numeric(7, 1))
    odds_place_high: Mapped[Optional[float]] = mapped_column(Numeric(7, 1))
    snapshot_at: Mapped[datetime] = mapped_column(nullable=False)


# ── 推論結果 ─────────────────────────────────────────────────────────────────


class PredictionResult(Base):
    __tablename__ = "prediction_results"

    prediction_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    race_id: Mapped[str] = mapped_column(String(20), nullable=False)
    horse_id: Mapped[str] = mapped_column(String(20), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(server_default=func.now())
    win_prob: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    place_prob: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    show_prob: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    predicted_win_odds: Mapped[Optional[float]] = mapped_column(Numeric(7, 1))
    predicted_place_odds: Mapped[Optional[float]] = mapped_column(Numeric(7, 1))
    expected_win_roi: Mapped[Optional[float]] = mapped_column(Numeric(7, 2))
    expected_show_roi: Mapped[Optional[float]] = mapped_column(Numeric(7, 2))
    predicted_position: Mapped[Optional[int]] = mapped_column(SmallInteger)
    predicted_running_style: Mapped[Optional[str]] = mapped_column(String(10))

    __table_args__ = (
        UniqueConstraint("race_id", "horse_id", "model_version", name="uq_prediction_race_horse_model"),
    )


class PredictionLapTime(Base):
    __tablename__ = "prediction_lap_times"
    __table_args__ = (
        CheckConstraint(
            "predicted_pace_cat IN ('HIGH','MIDDLE','SLOW')",
            name="ck_predicted_pace_cat",
        ),
    )

    race_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    model_version: Mapped[str] = mapped_column(String(50), primary_key=True)
    furlong_index: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    predicted_lap_sec: Mapped[Optional[float]] = mapped_column(Numeric(4, 2))
    predicted_pace_cat: Mapped[Optional[str]] = mapped_column(String(10))


# ── スクレイプ実行ログ ───────────────────────────────────────────────────────


class ScrapeRun(Base):
    __tablename__ = "scrape_runs"
    __table_args__ = (
        CheckConstraint("status IN ('SUCCESS','FAILED','RETRY')", name="ck_scrape_status"),
    )

    run_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    target_type: Mapped[str] = mapped_column(String(30), nullable=False)
    target_id: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(10), nullable=False)
    retry_count: Mapped[int] = mapped_column(SmallInteger, default=0, server_default="0")
    started_at: Mapped[datetime] = mapped_column(nullable=False)
    finished_at: Mapped[Optional[datetime]] = mapped_column()
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    gcs_path: Mapped[Optional[str]] = mapped_column(Text)


# ── 分析機能（AREA-01 §3-3）──────────────────────────────────────────────────


class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    is_member: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    subscription_status: Mapped[str] = mapped_column(String(20), default="none", server_default="'none'")
    subscription_expires_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    payjp_customer_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)


class SavedAnalysis(Base):
    __tablename__ = "saved_analyses"
    __table_args__ = (
        CheckConstraint(
            "analysis_type IN ('sire','course','jockey','trainer')",
            name="ck_analysis_type",
        ),
        Index("idx_saved_analyses_params", "filter_conditions", postgresql_using="gin"),
    )

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    user_id: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    analysis_type: Mapped[str] = mapped_column(String(20), nullable=False)
    filter_conditions: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    last_run_at: Mapped[Optional[datetime]] = mapped_column()


class SireAptitudeCache(Base):
    """父・母父の舞台適性統計キャッシュ（毎週月曜集計）。"""
    __tablename__ = "sire_aptitude_cache"
    __table_args__ = (
        UniqueConstraint(
            "sire_name", "sire_type", "surface", "distance_band", "track_condition", "week_label",
            name="uq_sire_aptitude_cache",
        ),
        Index("idx_sire_aptitude_lookup", "sire_name", "sire_type", "surface", "distance_band"),
        CheckConstraint("sire_type IN ('sire', 'dam_sire')", name="ck_sire_type"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    sire_name: Mapped[str] = mapped_column(String(100), nullable=False)
    sire_type: Mapped[str] = mapped_column(String(20), nullable=False)
    surface: Mapped[str] = mapped_column(String(10), nullable=False)
    distance_band: Mapped[str] = mapped_column(String(10), nullable=False)
    track_condition: Mapped[str] = mapped_column(String(10), nullable=False)
    n_runs: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    n_wins: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    n_place: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    win_rate: Mapped[Optional[float]] = mapped_column(Numeric(6, 4))
    place_rate: Mapped[Optional[float]] = mapped_column(Numeric(6, 4))
    roi_win: Mapped[Optional[float]] = mapped_column(Numeric(7, 4))
    roi_place: Mapped[Optional[float]] = mapped_column(Numeric(7, 4))
    week_label: Mapped[str] = mapped_column(String(10), nullable=False)
    computed_at: Mapped[datetime] = mapped_column(server_default=func.now())


class CourseStatsCache(Base):
    __tablename__ = "course_stats_cache"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    track: Mapped[str] = mapped_column(String(20), nullable=False)
    distance: Mapped[int] = mapped_column(Integer, nullable=False)
    surface: Mapped[str] = mapped_column(String(10), nullable=False)
    track_condition: Mapped[str] = mapped_column(String(10), nullable=False)
    stat_type: Mapped[str] = mapped_column(String(30), nullable=False)
    stat_key: Mapped[str] = mapped_column(String(50), nullable=False)
    n_runs: Mapped[Optional[int]] = mapped_column(Integer)
    win_rate: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    place_rate: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    roi_win: Mapped[Optional[float]] = mapped_column(Numeric(7, 4))
    computed_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        UniqueConstraint(
            "track", "distance", "surface", "track_condition", "stat_type", "stat_key",
            name="uq_course_stats_cache",
        ),
    )


# ── ユーザー拡張機能（F-12 / F-09）──────────────────────────────────────────


class UserFavorite(Base):
    """ユーザーのお気に入り馬（F-12）。"""
    __tablename__ = "user_favorites"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    horse_id: Mapped[str] = mapped_column(String(20), nullable=False)
    horse_name: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    __table_args__ = (UniqueConstraint("user_id", "horse_id", name="uq_user_horse"),)


class NotificationSetting(Base):
    """ユーザーの通知設定（F-09）。"""
    __tablename__ = "notification_settings"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True)
    email: Mapped[Optional[str]] = mapped_column(String(255))
    notify_favorite_race: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, onupdate=datetime.utcnow)


class NotificationLog(Base):
    """送信済み通知ログ（F-09）。"""
    __tablename__ = "notification_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    race_id: Mapped[str] = mapped_column(String(20), nullable=False)
    horse_id: Mapped[str] = mapped_column(String(20), nullable=False)
    sent_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    status: Mapped[str] = mapped_column(String(20), default="sent")


# ── めぐ指数（AREA-11）───────────────────────────────────────────────────────


class MeguParTime(Base):
    """基準タイムマスター（セル別）。"""
    __tablename__ = "megu_par_time"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    distance: Mapped[int] = mapped_column(Integer, nullable=False)
    course: Mapped[str] = mapped_column(String(20), nullable=False)
    surface: Mapped[str] = mapped_column(String(10), nullable=False)
    track_condition: Mapped[str] = mapped_column(String(10), nullable=False)
    par_time_sec: Mapped[float] = mapped_column(Numeric(6, 2), nullable=False)
    par_front_split_sec: Mapped[Optional[float]] = mapped_column(Numeric(5, 2))
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    model_version: Mapped[str] = mapped_column(String(20), nullable=False, default="stg-v1")
    computed_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("distance", "course", "surface", "track_condition", "model_version",
                         name="uq_megu_par_time"),
    )


class MeguRegressionParams(Base):
    """OLS 回帰係数（β1〜β4）。"""
    __tablename__ = "megu_regression_params"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    param_name: Mapped[str] = mapped_column(String(50), nullable=False)
    param_value: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False)
    std_error: Mapped[Optional[float]] = mapped_column(Numeric(10, 6))
    sample_count: Mapped[Optional[int]] = mapped_column(Integer)
    model_version: Mapped[str] = mapped_column(String(20), nullable=False, default="stg-v1")
    fitted_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("param_name", "model_version", name="uq_megu_reg_params"),
    )


class MeguIndex(Base):
    """馬×レース単位のめぐ指数。"""
    __tablename__ = "megu_index"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    race_id: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    horse_id: Mapped[str] = mapped_column(String(20), nullable=False)
    finish_time_sec: Mapped[float] = mapped_column(Numeric(6, 2), nullable=False)
    par_time_sec: Mapped[Optional[float]] = mapped_column(Numeric(6, 2))
    delta_pace_sec: Mapped[float] = mapped_column(Numeric(5, 3), nullable=False, default=0)
    delta_track_sec: Mapped[float] = mapped_column(Numeric(5, 3), nullable=False, default=0)
    delta_weight_sec: Mapped[float] = mapped_column(Numeric(5, 3), nullable=False, default=0)
    delta_level_sec: Mapped[float] = mapped_column(Numeric(5, 3), nullable=False, default=0)
    adjusted_time_sec: Mapped[float] = mapped_column(Numeric(6, 2), nullable=False)
    megu_index: Mapped[float] = mapped_column(Numeric(6, 1), nullable=False)
    field_quality: Mapped[Optional[float]] = mapped_column(Numeric(14, 0))
    model_version: Mapped[str] = mapped_column(String(20), nullable=False, default="stg-v1")
    computed_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("race_id", "horse_id", "model_version", name="uq_megu_index"),
        Index("idx_megu_index_horse", "horse_id", "computed_at"),
    )
