"""スクレイピング各エンドポイントの SLA 仕様（いつから取得でき、誰がいつ取りに行くか）。

正本は docs/operations/scraping_sla_audit_20261013.md の表。このモジュールは「機械で検査できる形」で持ち、
tests/scraper/test_sla_spec.py が TASK_CATALOG・スケジュール定義・各タスクリストとの食い違いを検出する。

- ``available_from`` は 2026-10-13 の実測（docs の probe 表）に基づく。確度 low/medium は
  ``availability_watch`` の first_seen_at 観測で更新する。
- ``fetches`` の ``job`` は ``auto_scrape.TASKS`` のキー、または次の固定名:
    pipeline.race_day / backfill.fast / backfill.full / backfill.horse / script.* / manual
  ``via`` に ``module.LIST`` を書くと、その list に ``task`` が入っていることをテストが確認する。
- 定期取得経路が無いものは ``KNOWN_GAPS`` に理由を書く（解消したら両方から消す。テストが食い違いを検出する）。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Fetch:
    job: str
    when: str
    task: str | None = None  # queue タスク id（via と組み）
    via: str | None = None  # 例 "src.scraper.auto_scrape_queue.T15_RACE_TASKS"


@dataclass(frozen=True)
class Endpoint:
    id: str
    label: str
    available_from: str
    confidence: str  # high | medium | low
    required: bool  # dev UI（予測→期待値→推奨馬券）または予測パイプラインが使う
    task_ids: tuple[str, ...] = ()  # queue_tasks.TASK_CATALOG の id
    fetches: tuple[Fetch, ...] = field(default_factory=tuple)


_Q = "src.scraper.auto_scrape_queue."
_EVE = _Q + "EVE_TASKS"

ENDPOINTS: tuple[Endpoint, ...] = (
    Endpoint("race_list", "レース一覧", "開催週の早い段階から部分的（火曜時点で 14R）。全レースは週後半", "medium", True, ("race_list", "date_all"),
             (Fetch("daily-race-lists", "毎日 07:00（設計は 17:00 も）"),)),
    Endpoint("race_card", "出馬表", "頭数・馬名・騎手は火曜から。枠番/馬番は枠順確定後（木〜金）。馬体重は当日", "medium", True, ("race_shutuba", "date_cards"),
             (Fetch("raceday-eve", "前日 18:00", "race_shutuba", _EVE),
              Fetch("raceday-runner", "各R T-15", "race_shutuba", _Q + "T15_RACE_TASKS"),
              Fetch("raceday-weight-runner", "各R T-65〜T-30（馬体重の検知。cron 未登録・手動起動）", "race_shutuba", _Q + "WEIGHT_WATCH_TASKS"))),
    Endpoint("odds", "単複オッズ", "前売開始（通常は前日夕方）以降。確定値は発走後も残る", "medium", True, ("race_odds", "race_odds_t60", "race_odds_t15", "race_odds_t10", "race_odds_t05", "race_odds_t02", "race_odds_final"),
             (Fetch("raceday-runner", "各R T-60", "race_odds_t60", _Q + "ODDS_T60_TASKS"),
              Fetch("raceday-runner", "各R T-15", "race_odds_t15", _Q + "T15_RACE_TASKS"),
              Fetch("raceday-runner", "各R T-10（KEIBA_ODDS_LATE_SNAPSHOTS=1 のとき・単複のみ）", "race_odds_t10", _Q + "ODDS_LATE_TASKS"),
              Fetch("raceday-runner", "各R T-5（同上）", "race_odds_t05", _Q + "ODDS_LATE_TASKS"),
              Fetch("raceday-runner", "各R T-2（同上）", "race_odds_t02", _Q + "ODDS_LATE_TASKS"),
              Fetch("raceday-evening", "開催日 17:30（確定）", "race_odds_final", _Q + "EVENING_TASKS"))),
    Endpoint("pair_odds", "馬連・ワイド・馬単・3連複オッズ", "odds と同じ", "high", True, ("race_pair_odds", "race_pair_odds_t60", "race_pair_odds_t15", "race_pair_odds_final"),
             (Fetch("raceday-runner", "各R T-60", "race_pair_odds_t60", _Q + "ODDS_T60_TASKS"),
              Fetch("raceday-runner", "各R T-15", "race_pair_odds_t15", _Q + "T15_RACE_TASKS"),
              Fetch("raceday-evening", "開催日 17:30（確定）", "race_pair_odds_final", _Q + "EVENING_TASKS"))),
    Endpoint("trifecta_odds", "3連単オッズ（全組合せ）", "odds と同じ", "high", False, ("race_trifecta_odds", "race_trifecta_odds_t60", "race_trifecta_odds_t15", "race_trifecta_odds_final"),
             (Fetch("raceday-runner", "各R T-60（KEIBA_ODDS_TRIFECTA=1 のとき）", "race_trifecta_odds_t60", _Q + "ODDS_T60_TASKS"),
              Fetch("raceday-runner", "各R T-15（同上）", "race_trifecta_odds_t15", _Q + "T15_RACE_TASKS"),
              Fetch("raceday-evening", "開催日 17:30 確定（同上）", "race_trifecta_odds_final", _Q + "EVENING_TASKS"),
              Fetch("backfill.trifecta", "過去分の一括取得（src.scripts.data.backfill_trifecta_odds）"))),
    Endpoint("shutuba_past", "馬柱", "開催週の前半（火曜時点で取得可）", "high", True, ("race_shutuba_past",),
             (Fetch("raceday-eve", "前日 18:00", "race_shutuba_past", _EVE),
              Fetch("raceday-runner", "各R T-15", "race_shutuba_past", _Q + "T15_RACE_TASKS"))),
    Endpoint("oikiri", "追い切り", "週中（水〜木）。火曜時点は 0 頭", "medium", True, ("race_oikiri",),
             (Fetch("raceday-eve", "前日 18:00", "race_oikiri", _EVE),
              Fetch("raceday-runner", "各R T-15", "race_oikiri", _Q + "T15_RACE_TASKS"))),
    Endpoint("paddock", "パドック評価", "当日（取得できるのは上位 5 頭のみ）", "medium", False, ("race_paddock",),
             (Fetch("pipeline.race_day", "予測パイプライン実行時（T-45）"), Fetch("backfill.full", "深夜バックフィル"))),
    Endpoint("trainer_comment", "厩舎コメント", "前日 16 時頃（ユーザー指摘。未観測）〜当日", "medium", False, ("race_trainer_comment",),
             (Fetch("raceday-comment", "前日 20:00 と開催日 07:40（コメントが出ているレースだけ保存）"),
              Fetch("recent-retry", "毎日 2 回（未取得のみ・間隔 3 時間・最大 30 回）"))),
    Endpoint("race_index", "タイム指数", "レース翌日までに取得可（既存文書の「週次まで待つ」より早い）", "medium", False, ("race_index",),
             (Fetch("weekly-update", "金曜 18:00（直近 10 日）"), Fetch("backfill.full", "深夜バックフィル"), Fetch("recent-retry", "毎日 2 回（未取得のみ・間隔 3 時間・最大 30 回）"))),
    Endpoint("barometer", "バロメーター（走行データ）", "レース翌週の金曜 18 時頃。障害は対象外", "high", False, ("race_barometer",),
             (Fetch("weekly-update", "金曜 18:00"), Fetch("barometer-retry", "毎日（公開済み・未取得のみ）"))),
    Endpoint("race_result_db", "レース結果（db.netkeiba）", "翌日までに取得可（金曜 16–17 時に反映とする既存仕様）", "medium", True, ("race_result", "date_results"),
             (Fetch("weekly-update", "金曜 18:00"), Fetch("backfill.fast", "深夜バックフィル"), Fetch("recent-retry", "毎日 2 回（未取得のみ・間隔 3 時間・最大 30 回）"))),
    Endpoint("race_result_on_time", "速報結果（race.netkeiba）", "発走後数分（T+15 で取得）", "medium", True, ("race_result_on_time",),
             (Fetch("raceday-result-runner", "各R 発走 +15 分"), Fetch("raceday-evening", "開催日 17:30", "race_result_on_time", _Q + "EVENING_TASKS"))),
    Endpoint("race_result_lap", "結果ラップ", "翌日に取得可（要ログイン）", "medium", False, ("race_result_lap",),
             (Fetch("recent-retry", "毎日 2 回（未取得のみ・間隔 3 時間・最大 30 回）"),)),
    Endpoint("race_detail", "レース詳細", "上流データ取得後（合成）", "medium", False, ()),
    Endpoint("smartrc", "SmartRC", "発走前〜（要観測）", "low", False, ("smartrc",),
             (Fetch("raceday-eve", "前日 18:00", "smartrc", _EVE),
              Fetch("raceday-runner", "各R T-15", "smartrc", _Q + "T15_RACE_TASKS"),
              Fetch("raceday-evening", "開催日 17:30", "smartrc", _Q + "EVENING_TASKS"))),
    Endpoint("horse_profile", "馬ページ（戦績）", "レース翌日までに反映", "medium", False, ("horse_profile",),
             (Fetch("weekly-update", "金曜 18:00（対象開催日の出走馬）"), Fetch("backfill.horse", "深夜バックフィル"))),
    Endpoint("horse_pedigree_5gen", "5世代血統", "馬が登録されたら", "medium", False, ("horse_pedigree_5gen",),
             (Fetch("new-horse-pedigree", "毎日 21:00（今後 1 週間の出走予定馬で ped_tbl が無い馬だけ）"),
              Fetch("script.run_after_scrape_missing_5gen_10gen_chain", "手動（IDE 外で実行。過去分の一括）"))),
    Endpoint("horse_mating_date", "種付け日", "母馬ページ", "low", False, ("horse_mating_date",)),
    Endpoint("horse_training", "調教タイム全ページ", "要追加観測", "low", False, ("horse_training",)),
    Endpoint("jra_baba", "JRA 馬場（クッション値・含水率）", "開催日の朝〜（非開催日は直近値が残る）", "medium", False, (),
             (Fetch("jra-baba-morning", "5:00–9:00 の 10 分毎"),)),
    Endpoint("jra_calendar", "JRA 開催カレンダー", "年初に年間日程", "high", True, (),
             (Fetch("daily-race-lists", "各タスクがカレンダーを参照（取得は都度）"),)),
)

# 集約・内部用で単独のエンドポイントを持たない task id
AGGREGATE_TASK_IDS: frozenset[str] = frozenset({"race_all"})

# 定期取得経路が無いエンドポイント → 理由（解消したら ENDPOINTS に Fetch を足してここから消す）
KNOWN_GAPS: dict[str, str] = {
    "race_detail": "どのスケジュールにも入っていない（手動・capture のみ）",
    "horse_mating_date": "手動投入のみ",
    "horse_training": "手動投入のみ",
}

NON_TASK_JOBS_PREFIXES = ("pipeline.", "backfill.", "script.", "manual")


def endpoints_without_schedule() -> set[str]:
    return {e.id for e in ENDPOINTS if not e.fetches}
