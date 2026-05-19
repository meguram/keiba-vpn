"""
スクレイピングキュー用: タスク定義と ScraperRunner へのディスパッチ。

job 辞書（job_kind / target_id / tasks）を実行する。
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

logger = logging.getLogger(__name__)

# API・UI 向けメタ（id は job.tasks に入れる文字列と一致）
TASK_CATALOG: list[dict[str, Any]] = [
    {"id": "race_all", "entity": "race", "label": "レース一式（従来どおり）", "hint": "出馬表・指数・オッズ・SmartRC・馬情報など scrape_race_all"},
    {"id": "race_result", "entity": "race", "label": "レース結果 (DB版)", "hint": "race_result (db.netkeiba.com)"},
    {"id": "race_result_on_time", "entity": "race", "label": "速報結果 (当日)", "hint": "race_result_on_time (race.netkeiba.com 当日17:30)"},
    {
        "id": "race_result_lap",
        "entity": "race",
        "label": "結果ラップ",
        "hint": "race_result_lap（成績ページのラップ・通過。期間一括のレース系で選択可。要ログイン）",
    },
    {"id": "race_shutuba", "entity": "race", "label": "出馬表", "hint": "race_shutuba"},
    {"id": "race_odds", "entity": "race", "label": "単複オッズ", "hint": "race_odds"},
    {"id": "race_pair_odds", "entity": "race", "label": "2連系オッズ", "hint": "race_pair_odds"},
    {"id": "race_index", "entity": "race", "label": "タイム指数", "hint": "race_index"},
    {"id": "race_paddock", "entity": "race", "label": "パドック", "hint": "race_paddock"},
    {"id": "race_barometer", "entity": "race", "label": "偏差値", "hint": "race_barometer"},
    {"id": "race_trainer_comment", "entity": "race", "label": "厩舎コメント", "hint": "race_trainer_comment"},
    {"id": "smartrc", "entity": "race", "label": "SmartRC 一式", "hint": "smartrc_race（開催日はジョブの date 推奨）"},
    {"id": "horse_profile", "entity": "horse", "label": "馬ページ（成績・血統HTML含む）", "hint": "horse_result + アーカイブ"},
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
    time.sleep(random.uniform(0.3, 0.8))


def execute_job(runner: Any, job: dict) -> None:
    """1ジョブ分のタスクを順に実行。例外は上位へ。"""
    from src.scraper.persist_context import reset_skip_local_mirror, set_skip_local_mirror
    from src.scraper.queue_job_progress import (
        clear_current_job_id,
        clear_job_progress,
        set_current_job_id,
        update_job_progress,
    )

    j = normalize_job_for_run(job)
    jid = str(j.get("job_id") or "").strip() or None
    kind = j["job_kind"]
    tid = str(j["target_id"]).strip()
    if not tid:
        raise ValueError("target_id が空です")
    tasks = j["tasks"]
    if not tasks:
        raise ValueError("tasks が空です")
    meta_date = str(j.get("date") or "").replace("-", "").replace("/", "")[:8]

    from src.scraper.scrape_policy import effective_smart_skip_for_queue_job

    smart_skip = effective_smart_skip_for_queue_job(j)
    n_tasks = len(tasks)

    tok = set_skip_local_mirror(
        bool(job.get("skip_local_mirror") or j.get("skip_local_mirror"))
    )
    set_current_job_id(jid)
    try:
        for ti, task in enumerate(tasks):
            if jid:
                update_job_progress(
                    jid,
                    job_kind=kind,
                    target_id=tid,
                    task_index=ti + 1,
                    task_total=n_tasks,
                    current_task=task,
                    phase="task",
                    message=f"タスク {ti + 1}/{n_tasks}: {task}",
                    step_kind="",
                    step_current=0,
                    step_total=0,
                    step_id="",
                    step_name="",
                )
            logger.info("キュータスク実行: kind=%s target=%s task=%s", kind, tid, task)
            skip_pause = _dispatch(runner, kind, tid, task, meta_date, smart_skip)
            if not skip_pause:
                _pause()
    finally:
        try:
            reset_skip_local_mirror(tok)
        except Exception:
            pass
        clear_current_job_id()
        if jid:
            clear_job_progress(jid)


def _dispatch(
    runner: Any,
    kind: str,
    tid: str,
    task: str,
    meta_date: str,
    smart_skip: bool = True,
) -> bool:
    """
    Returns:
        True ならタスク間の _pause を省略（既存 5gen スキップなど短時間のとき）。
    """
    if kind == "race":
        return _race_task(runner, tid, task, meta_date, smart_skip=smart_skip)
    if kind == "horse":
        return _horse_task(runner, tid, task, smart_skip=smart_skip)
    if kind == "date":
        return _date_task(runner, tid, task, smart_skip=smart_skip)
    raise ValueError(f"不明な job_kind: {kind}")


def _race_task(
    runner: Any,
    race_id: str,
    task: str,
    meta_date: str,
    *,
    smart_skip: bool = True,
) -> bool:
    from src.scraper.queue_job_progress import get_current_job_id, update_job_progress

    jid = get_current_job_id()
    if jid and task == "race_all":
        update_job_progress(
            jid,
            phase="race_all",
            step_kind="race",
            step_id=race_id,
            step_name="",
            message=f"レース一式取得 {race_id}",
        )
    if task == "race_all":
        # ジョブの smart_skip（既定 True）を反映。False 固定だと Phase1 の is_fresh
        # スキップが常に無効になり、キュー経由でも全カテゴリ再取得になっていた。
        runner.scrape_race_all(race_id, smart_skip=smart_skip)
        return False
    if task == "smartrc":
        runner.scrape_smartrc(race_id, date=meta_date)
        return False
    # horse タスクが race ジョブに含まれる場合: race_shutuba から horse_id を解決して委譲
    _horse_from_race: dict[str, str] = {
        "horse_result":       "horse_profile",
        "horse_pedigree_5gen": "horse_pedigree_5gen",
        "horse_training":     "horse_training",
    }
    if task in _horse_from_race:
        horse_task = _horse_from_race[task]
        card = runner.storage.load("race_shutuba", race_id)
        horse_ids: list[str] = [
            e["horse_id"] for e in ((card.get("entries", []) if card else []))
            if e.get("horse_id")
        ]
        if jid:
            update_job_progress(
                jid, phase=task, step_kind="race", step_id=race_id,
                message=f"{task} · {race_id} ({len(horse_ids)}頭)",
            )
        for hid in horse_ids:
            _horse_task(runner, hid, horse_task, smart_skip=smart_skip)
            _pause()
        return False

    # getattr で遅延解決: fn_map リテラルで存在しないメソッドを列挙すると
    # 他タスク実行時も AttributeError になるためメソッド名だけ列挙する。
    _race_fn: dict[str, str] = {
        "race_result": "scrape_race_result",
        "race_result_on_time": "scrape_race_result_on_time",
        "race_result_lap": "scrape_race_result_lap",
        "race_shutuba": "scrape_race_card",
        "race_odds": "scrape_odds",
        "race_pair_odds": "scrape_pair_odds",
        "race_index": "scrape_speed_index",
        "race_paddock": "scrape_paddock",
        "race_barometer": "scrape_barometer",
        "race_trainer_comment": "scrape_trainer_comment",
    }
    mname = _race_fn.get(task)
    if not mname:
        raise ValueError(f"レース向け未対応タスク: {task}")
    fn = getattr(runner, mname, None)
    if not callable(fn):
        raise ValueError(
            f"ScraperRunner に {mname} がありません（タスク: {task}）"
        )
    if jid:
        update_job_progress(
            jid, phase=task, step_kind="race", step_id=race_id, message=f"{task} · {race_id}",
        )
    fn(race_id, skip_existing=smart_skip)
    return False


def _horse_task(
    runner: Any, horse_id: str, task: str, *, smart_skip: bool = True
) -> bool:
    from src.scraper.queue_job_progress import get_current_job_id, update_job_progress

    jid = get_current_job_id()
    if jid:
        update_job_progress(
            jid,
            phase=task,
            step_kind="horse",
            step_id=horse_id,
            message=f"{task} · {horse_id}",
        )
    if task == "horse_pedigree_5gen":
        runner.scrape_horse_pedigree_5gen(horse_id, skip_existing=smart_skip)
        if smart_skip and getattr(runner, "_last_pedigree_5gen_skip", False):
            return True
        return False
    if task in ("horse_profile", "horse_pedigree"):
        runner.scrape_horse(horse_id, skip_existing=smart_skip, with_history=True)
        return False
    if task == "horse_training":
        runner.scrape_horse_training(horse_id, smart_skip=smart_skip)
        if getattr(runner, "_queue_mute_scrape_throttle", None) is True:
            return True
        return False
    raise ValueError(f"馬向け未対応タスク: {task}")


def _date_task(
    runner: Any, date_key: str, task: str, *, smart_skip: bool = True
) -> bool:
    from src.scraper.queue_job_progress import get_current_job_id, update_job_progress

    jid = get_current_job_id()
    if jid and task != "date_all":
        update_job_progress(
            jid,
            phase=task,
            step_kind="date",
            message=f"{task} · {date_key}",
        )
    if task == "race_list":
        runner.scrape_race_list(date_key)
        return False
    if task == "date_results":
        runner.scrape_date_results(date_key)
        return False
    if task == "date_cards":
        runner.scrape_date_cards(date_key)
        return False
    if task == "date_all":
        if smart_skip:
            try:
                from src.scraper.date_complete import DateCompleteRegistry
                dc = DateCompleteRegistry(runner.storage._base_dir)
                sv = runner.storage._load_structure_versions()
                if dc.is_complete(date_key, sv):
                    logger.info("日付完了スキップ (キュー): %s", date_key)
                    if jid:
                        update_job_progress(
                            jid, phase="date_all",
                            message=f"完了済みスキップ · {date_key}",
                        )
                    return True
            except Exception:
                pass
        runner.scrape_date_all(date_key, smart_skip=smart_skip)
        return False
    raise ValueError(f"開催日向け未対応タスク: {task}")


def build_job_label(job_kind: str, target_id: str, tasks: list[str]) -> str:
    tstr = ",".join(tasks) if tasks else "?"
    return f"{job_kind}:{target_id} [{tstr}]"


_HORSE_TASKS_IN_RACE = frozenset({"horse_result", "horse_pedigree_5gen", "horse_training"})


def validate_tasks_for_kind(job_kind: str, tasks: list[str]) -> str | None:
    """エラー文 or None"""
    allowed = {t["id"] for t in TASK_CATALOG if t["entity"] == job_kind}
    if job_kind == "race":
        allowed |= _HORSE_TASKS_IN_RACE  # race_shutuba 経由で horse_id 解決
    for x in tasks:
        if x not in allowed:
            return f"タスク {x!r} は entity={job_kind!r} では使えません"
    return None
