"""
スクレイピングキュー用: タスク定義と ScraperRunner へのディスパッチ。

job 辞書（job_kind / target_id / tasks）を実行する。
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

# API・UI 向けメタ（id は job.tasks に入れる文字列と一致）
TASK_CATALOG: list[dict[str, Any]] = [
    {"id": "race_all", "entity": "race", "label": "レース一式（従来どおり）", "hint": "出馬表・指数・オッズ・SmartRC・馬情報など scrape_race_all"},
    {"id": "race_result", "entity": "race", "label": "レース結果", "hint": "race_result"},
    {"id": "race_shutuba", "entity": "race", "label": "出馬表", "hint": "race_shutuba"},
    {"id": "race_odds", "entity": "race", "label": "単複オッズ", "hint": "race_odds"},
    {"id": "race_pair_odds", "entity": "race", "label": "2連系オッズ", "hint": "race_pair_odds"},
    {"id": "race_index", "entity": "race", "label": "タイム指数", "hint": "race_index"},
    {"id": "race_shutuba_past", "entity": "race", "label": "馬柱・過去成績", "hint": "race_shutuba_past"},
    {"id": "race_paddock", "entity": "race", "label": "パドック", "hint": "race_paddock"},
    {"id": "race_barometer", "entity": "race", "label": "偏差値", "hint": "race_barometer"},
    {"id": "race_oikiri", "entity": "race", "label": "追い切り", "hint": "race_oikiri"},
    {"id": "race_trainer_comment", "entity": "race", "label": "厩舎コメント", "hint": "race_trainer_comment"},
    {"id": "smartrc", "entity": "race", "label": "SmartRC 一式", "hint": "smartrc_race（開催日はジョブの date 推奨）"},
    {"id": "horse_profile", "entity": "horse", "label": "馬ページ（成績・血統HTML含む）", "hint": "horse_result + アーカイブ"},
    {"id": "horse_pedigree", "entity": "horse", "label": "血統（馬ページ経由・血統HTML）", "hint": "horse_profile と同じ（出走馬集合は期間投入APIから）"},
    {"id": "horse_pedigree_5gen", "entity": "horse", "label": "5世代血統JSON（db.netkeiba /horse/ped/）", "hint": "horse_pedigree_5gen カテゴリへ保存"},
    {"id": "horse_training", "entity": "horse", "label": "調教タイム全ページ", "hint": "horse_training（要ログイン）"},
    {"id": "race_list", "entity": "date", "label": "その日のレース一覧", "hint": "race_lists"},
    {"id": "date_results", "entity": "date", "label": "その日の結果一括", "hint": "scrape_date_results"},
    {"id": "date_cards", "entity": "date", "label": "その日の出馬表一括", "hint": "scrape_date_cards"},
    {"id": "date_all", "entity": "date", "label": "その日の一式（smart_skip 可）", "hint": "scrape_date_all"},
]


def catalog_for_api() -> list[dict[str, Any]]:
    return list(TASK_CATALOG)


def normalize_tasks(task_list: list[str] | None) -> list[str]:
    raw = [str(x).strip() for x in (task_list or []) if str(x).strip()]
    if "race_all" in raw:
        return ["race_all"]
    return raw


def normalize_job_for_run(job: dict) -> dict[str, Any]:
    """ディスク上の旧ジョブも実行可能にする。"""
    if job.get("job_kind") and job.get("target_id") is not None:
        j = dict(job)
        j["tasks"] = normalize_tasks(job.get("tasks"))
        if not j["tasks"]:
            j["tasks"] = ["race_all"] if j.get("job_kind") == "race" else []
        return j
    rid = job.get("race_id")
    if rid:
        j = dict(job)
        j["job_kind"] = "race"
        j["target_id"] = rid
        j["tasks"] = ["race_all"]
        return j
    raise ValueError("ジョブに target_id / race_id がありません")


def _pause():
    time.sleep(random.uniform(0.4, 1.2))


def execute_job(runner: Any, job: dict) -> None:
    """1ジョブ分のタスクを順に実行。例外は上位へ。"""
    j = normalize_job_for_run(job)
    kind = j["job_kind"]
    tid = str(j["target_id"]).strip()
    if not tid:
        raise ValueError("target_id が空です")
    tasks = j["tasks"]
    if not tasks:
        raise ValueError("tasks が空です")
    meta_date = str(j.get("date") or "").replace("-", "").replace("/", "")[:8]

    smart_skip = bool(j.get("smart_skip", True))

    for task in tasks:
        logger.info("キュータスク実行: kind=%s target=%s task=%s", kind, tid, task)
        _dispatch(runner, kind, tid, task, meta_date, smart_skip)
        _pause()


def _dispatch(
    runner: Any,
    kind: str,
    tid: str,
    task: str,
    meta_date: str,
    smart_skip: bool = True,
) -> None:
    if kind == "race":
        _race_task(runner, tid, task, meta_date)
    elif kind == "horse":
        _horse_task(runner, tid, task, smart_skip=smart_skip)
    elif kind == "date":
        _date_task(runner, tid, task, smart_skip=smart_skip)
    else:
        raise ValueError(f"不明な job_kind: {kind}")


def _race_task(runner: Any, race_id: str, task: str, meta_date: str) -> None:
    if task == "race_all":
        runner.scrape_race_all(race_id, smart_skip=False)
        return
    if task == "smartrc":
        runner.scrape_smartrc(race_id, date=meta_date)
        return
    fn_map: dict[str, Callable[..., Any]] = {
        "race_result": runner.scrape_race_result,
        "race_shutuba": runner.scrape_race_card,
        "race_odds": runner.scrape_odds,
        "race_pair_odds": runner.scrape_pair_odds,
        "race_index": runner.scrape_speed_index,
        "race_shutuba_past": runner.scrape_shutuba_past,
        "race_paddock": runner.scrape_paddock,
        "race_barometer": runner.scrape_barometer,
        "race_oikiri": runner.scrape_oikiri,
        "race_trainer_comment": runner.scrape_trainer_comment,
    }
    fn = fn_map.get(task)
    if not fn:
        raise ValueError(f"レース向け未対応タスク: {task}")
    fn(race_id, skip_existing=False)


def _horse_task(
    runner: Any, horse_id: str, task: str, *, smart_skip: bool = True
) -> None:
    if task == "horse_pedigree_5gen":
        runner.scrape_horse_pedigree_5gen(horse_id, skip_existing=smart_skip)
        return
    if task in ("horse_profile", "horse_pedigree"):
        runner.scrape_horse(horse_id, skip_existing=False, with_history=True)
        return
    if task == "horse_training":
        runner.scrape_horse_training(horse_id, skip_existing=False)
        return
    raise ValueError(f"馬向け未対応タスク: {task}")


def _date_task(
    runner: Any, date_key: str, task: str, *, smart_skip: bool = True
) -> None:
    if task == "race_list":
        runner.scrape_race_list(date_key)
        return
    if task == "date_results":
        runner.scrape_date_results(date_key)
        return
    if task == "date_cards":
        runner.scrape_date_cards(date_key)
        return
    if task == "date_all":
        runner.scrape_date_all(date_key, smart_skip=smart_skip)
        return
    raise ValueError(f"開催日向け未対応タスク: {task}")


def build_job_label(job_kind: str, target_id: str, tasks: list[str]) -> str:
    tstr = ",".join(tasks) if tasks else "?"
    return f"{job_kind}:{target_id} [{tstr}]"


def validate_tasks_for_kind(job_kind: str, tasks: list[str]) -> str | None:
    """エラー文 or None"""
    allowed = {t["id"] for t in TASK_CATALOG if t["entity"] == job_kind}
    for x in tasks:
        if x not in allowed:
            return f"タスク {x!r} は entity={job_kind!r} では使えません"
    return None
