"""
最終めぐ指数（megu_index v2）計算に必要なデータギャップを監査し、タスクリストを管理する。

スキップしてよいのは「データがそもそも存在しない」場合のみ:
  - 障害レース（芝/ダート対象外）
  - 非開催日
  - GCS/flat/DB いずれにも結果が無い（未開催・恒久的ページ欠損）

それ以外（メタ欠損・未 scrape・未 compute）は backfill タスクとして登録する。

Usage:
    KEIBA_ENV=stg python -m src.scripts.data.megu_index_gap_audit
    KEIBA_ENV=stg python -m src.scripts.data.megu_index_gap_audit --refresh
    KEIBA_ENV=stg python -m src.scripts.data.megu_index_gap_audit --run repair_flat_metadata
    KEIBA_ENV=stg python -m src.scripts.data.megu_index_gap_audit --run compute_missing
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

from src.pipeline.megu_index.flat_metadata import audit_flat_metadata, repair_race_result_flat_metadata
from src.utils.keiba_logging import script_basic_config

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TASKS_PATH = PROJECT_ROOT / "data/local/megu_index/gap_tasks.json"
REPORT_PATH = PROJECT_ROOT / "data/local/megu_index/gap_report.json"

# compute_for_date の skipped reason → 対応アクション（legitimate_skip=True ならデータ不存在）
SKIP_POLICY: dict[str, dict[str, Any]] = {
    "no_params": {
        "title": "DB 回帰パラメータ (megu_regression_params v2) 未登録",
        "action": "register_regression_params_v2",
        "legitimate_skip": False,
        "command": "KEIBA_ENV=stg python -m src.pipeline.megu_index.build_par_time --model-version v2",
    },
    "no_flat_data": {
        "title": "race_result_flat.parquet 不存在",
        "action": "export_race_result_flat",
        "legitimate_skip": False,
        "command": "python -m src.scraper.export_tables --years {year}",
    },
    "no_data_on_date": {
        "title": "対象日のレース結果なし",
        "action": "verify_meeting_or_scrape",
        "legitimate_skip": True,
        "command": "python -m src.scraper.auto_scrape --task raceday-evening --date {date}",
    },
    "no_results_after_compute": {
        "title": "flat メタ欠損等で compute 後に保存行ゼロ",
        "action": "repair_flat_metadata",
        "legitimate_skip": False,
        "command": "KEIBA_ENV=stg python -m src.scripts.data.megu_index_gap_audit --run repair_flat_metadata",
    },
}


def _task_id(action: str, scope: str) -> str:
    return f"{action}:{scope}"


def _load_tasks() -> dict:
    if TASKS_PATH.exists():
        return json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    return {"version": 1, "updated_at": None, "tasks": []}


def _save_tasks(data: dict) -> None:
    TASKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    TASKS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _upsert_task(tasks: list[dict], task: dict) -> None:
    tid = task["id"]
    for i, t in enumerate(tasks):
        if t.get("id") == tid:
            if t.get("status") == "done":
                task["status"] = "done"
            tasks[i] = {**t, **task}
            return
    tasks.append(task)


def audit_db_gaps(session) -> dict:
    """DB 上の megu_index 欠損・メタ欠損を集計。"""
    by_year = session.execute(
        text("""
            SELECT SUBSTRING(r.race_id, 1, 4) AS yr,
                   COUNT(DISTINCT r.race_id) AS races_fin,
                   COUNT(DISTINCT mi.race_id) AS races_megu_valid
            FROM race_results rr
            JOIN races r ON r.race_id = rr.race_id
            LEFT JOIN megu_index mi ON mi.race_id = r.race_id AND mi.horse_id = rr.horse_id
                AND mi.model_version = 'v2' AND mi.computation_status = 'valid'
            WHERE rr.finish_pos > 0 AND rr.finish_time_sec > 0
              AND r.surface IN ('芝', 'ダート')
              AND r.race_date >= '2020-01-01'
            GROUP BY 1 ORDER BY 1
        """)
    ).fetchall()

    missing_races = session.execute(
        text("""
            SELECT COUNT(DISTINCT r.race_id)
            FROM races r
            WHERE r.race_date >= '2020-01-01' AND r.race_date < CURRENT_DATE
              AND r.surface IN ('芝', 'ダート')
              AND EXISTS (
                SELECT 1 FROM race_results rr
                WHERE rr.race_id = r.race_id AND rr.finish_pos > 0 AND rr.finish_time_sec > 0
              )
              AND NOT EXISTS (
                SELECT 1 FROM megu_index mi
                WHERE mi.race_id = r.race_id AND mi.model_version = 'v2'
                  AND mi.computation_status = 'valid'
              )
        """)
    ).scalar()

    params_v2 = session.execute(
        text("SELECT COUNT(*) FROM megu_regression_params WHERE model_version = 'v2'")
    ).scalar()
    params_v1 = session.execute(
        text("SELECT COUNT(*) FROM megu_regression_params WHERE model_version = 'v1'")
    ).scalar()
    par_v2 = session.execute(
        text("SELECT COUNT(*) FROM megu_par_time WHERE model_version = 'v2'")
    ).scalar()

    bad_meta = session.execute(
        text("""
            SELECT COUNT(DISTINCT r.race_id)
            FROM races r
            WHERE r.race_date >= '2020-01-01'
              AND r.surface IN ('芝', 'ダート')
              AND EXISTS (
                SELECT 1 FROM race_results rr
                WHERE rr.race_id = r.race_id AND rr.finish_pos > 0
              )
              AND (r.distance IS NULL OR r.distance <= 0 OR r.surface IS NULL OR r.surface = '')
        """)
    ).scalar()

    return {
        "by_year": [dict(r._mapping) for r in by_year],
        "missing_megu_races": int(missing_races or 0),
        "races_bad_metadata_db": int(bad_meta or 0),
        "regression_params_v2": int(params_v2 or 0),
        "regression_params_v1": int(params_v1 or 0),
        "par_time_v2": int(par_v2 or 0),
    }


def build_tasks_from_audit(report: dict) -> list[dict]:
    """監査結果からタスクリストを生成。"""
    tasks: list[dict] = []

    if report["db"]["regression_params_v2"] == 0 and report["db"]["regression_params_v1"] > 0:
        pol = SKIP_POLICY["no_params"]
        tasks.append({
            "id": _task_id("register_regression_params_v2", "global"),
            "status": "pending",
            "priority": 1,
            "category": "db_params",
            "title": pol["title"],
            "detail": "compute.py は v1 fallback するが v2 正本化を推奨",
            "legitimate_skip": False,
            "affected_count": 1,
            "command": (
                "NB-02 出力を DB 投入、または v1→v2 コピー。"
                " 例: notebooks/megu_index/output/nb02/regression_params.json 参照"
            ),
        })

    for year_str, flat in report.get("flat_quality", {}).items():
        if flat.get("missing_file"):
            pol = SKIP_POLICY["no_flat_data"]
            tasks.append({
                "id": _task_id("export_race_result_flat", year_str),
                "status": "pending",
                "priority": 2,
                "category": "flat_export",
                "title": f"{year_str}年 race_result_flat 不存在",
                "legitimate_skip": False,
                "affected_count": 1,
                "command": pol["command"].format(year=year_str),
            })
            continue
        bad_races = int(flat.get("bad_races") or 0)
        backfill_races = (
            int(flat["backfill_races"])
            if flat.get("backfill_races") is not None
            else bad_races
        )
        obstacle_races = (
            int(flat["obstacle_races"])
            if flat.get("obstacle_races") is not None
            else 0
        )
        backfill_ids = flat.get("backfill_race_ids") or []

        if bad_races > 0 and backfill_races > 0:
            tasks.append({
                "id": _task_id("repair_flat_metadata", year_str),
                "status": "pending",
                "priority": 2,
                "category": "flat_metadata",
                "title": f"{year_str}年 flat surface/distance 欠損（芝/ダート要修復）",
                "detail": (
                    f"backfill={backfill_races} obstacle_skip={obstacle_races} "
                    f"/ total_bad={bad_races}"
                ),
                "legitimate_skip": False,
                "affected_count": backfill_races,
                "race_ids": backfill_ids[:20],
                "command": (
                    f"KEIBA_ENV=stg python -m src.scripts.data.megu_index_gap_audit "
                    f"--run repair_flat_metadata --year {year_str}"
                ),
            })
        elif bad_races > 0 and backfill_races == 0:
            tasks.append({
                "id": _task_id("repair_flat_metadata", year_str),
                "status": "done",
                "priority": 2,
                "category": "flat_metadata",
                "title": f"{year_str}年 flat メタ欠損は障害レースのみ",
                "detail": f"obstacle_skip={obstacle_races}（正当スキップ）",
                "legitimate_skip": True,
                "affected_count": obstacle_races,
                "command": "",
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            })

        if backfill_races > 0:
            ids_hint = ", ".join(backfill_ids[:5])
            tasks.append({
                "id": _task_id("scrape_race_result_gcs", year_str),
                "status": "pending",
                "priority": 3,
                "category": "gcs_scrape",
                "title": f"{year_str}年 GCS race_result 欠損（芝/ダート {backfill_races}R）",
                "detail": f"例: {ids_hint}" if ids_hint else "lap 推定で救えないレース",
                "legitimate_skip": False,
                "affected_count": backfill_races,
                "race_ids": backfill_ids[:20],
                "command": (
                    f"python -m src.scraper.auto_scrape --task weekly-update "
                    f"（対象年 {year_str}、race_id 指定可）"
                ),
            })
        elif bad_races > 0:
            tasks.append({
                "id": _task_id("scrape_race_result_gcs", year_str),
                "status": "done",
                "priority": 3,
                "category": "gcs_scrape",
                "title": f"{year_str}年 GCS 欠損は障害のみ",
                "detail": f"obstacle_skip={obstacle_races}",
                "legitimate_skip": True,
                "affected_count": obstacle_races,
                "command": "",
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            })

    missing = int(report["db"].get("missing_megu_races") or 0)
    if missing > 0:
        tasks.append({
            "id": _task_id("compute_missing_megu", "2020-2026"),
            "status": "pending",
            "priority": 2,
            "category": "compute",
            "title": "DB megu_index v2 valid 欠損レース",
            "legitimate_skip": False,
            "affected_count": missing,
            "command": "KEIBA_ENV=stg python -m src.scripts.data.megu_index_gap_audit --run compute_missing",
        })
    else:
        tasks.append({
            "id": _task_id("compute_missing_megu", "2020-2026"),
            "status": "done",
            "priority": 2,
            "category": "compute",
            "title": "DB megu_index v2 valid 欠損なし",
            "legitimate_skip": True,
            "affected_count": 0,
            "command": "",
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        })

    nb_outputs = report.get("notebook_outputs", {})
    for name, ok in nb_outputs.items():
        if not ok:
            tasks.append({
                "id": _task_id("run_notebook", name),
                "status": "pending",
                "priority": 4,
                "category": "notebook",
                "title": f"ノートブック出力不足: {name}",
                "detail": "研究・import 用。本番 UI は compute.py + DB が正本",
                "legitimate_skip": True,
                "affected_count": 1,
                "command": f"notebooks/megu_index/{name} を実行",
            })

    return tasks


def refresh_audit(years: list[int] | None = None) -> dict:
    """監査実行 + タスクリスト更新。"""
    from src.db.session import get_session, init_engine

    years = years or list(range(2020, date.today().year + 1))
    flat_quality = {str(y): audit_flat_metadata(y) for y in years}

    nb_dir = PROJECT_ROOT / "notebooks/megu_index/output"
    nb_outputs = {
        "nb01/megu_dataset.parquet": (nb_dir / "nb01/megu_dataset.parquet").exists(),
        "nb02/coeff_pace.parquet": (nb_dir / "nb02/coeff_pace.parquet").exists(),
        "nb03/delta_track.parquet": (nb_dir / "nb03/delta_track.parquet").exists(),
        "nb04/megu_index.parquet": (nb_dir / "nb04/megu_index.parquet").exists(),
        "nb05/megu_final.parquet": (nb_dir / "nb05/megu_final.parquet").exists(),
    }

    init_engine()
    with get_session() as session:
        db_report = audit_db_gaps(session)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "db": db_report,
        "flat_quality": flat_quality,
        "notebook_outputs": nb_outputs,
        "skip_policy": SKIP_POLICY,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    store = _load_tasks()
    new_tasks = build_tasks_from_audit(report)
    merged = store.get("tasks") or []
    for t in new_tasks:
        _upsert_task(merged, t)
    store["tasks"] = sorted(merged, key=lambda x: (x.get("priority", 99), x.get("id", "")))
    _save_tasks(store)

    logger.info(
        "audit done: missing_megu_races=%d tasks=%d",
        db_report["missing_megu_races"],
        len(store["tasks"]),
    )
    return report


def run_repair_flat_metadata(year: int, dry_run: bool = False) -> dict:
    result = repair_race_result_flat_metadata(year, dry_run=dry_run)
    if not dry_run:
        flat = audit_flat_metadata(year)
        bf = flat.get("backfill_races")
        if bf is not None and int(bf) == 0:
            _mark_task_done(_task_id("repair_flat_metadata", str(year)))
            if int(flat.get("obstacle_races") or 0) > 0:
                _mark_task_done(_task_id("scrape_race_result_gcs", str(year)))
    return result


def run_compute_missing(years: list[int] | None = None) -> list[dict]:
    from src.pipeline.megu_index.compute import compute_for_date
    from src.db.session import get_session, init_engine

    years = years or list(range(2020, date.today().year + 1))
    init_engine()
    results = []
    with get_session() as session:
        for yr in years:
            rows = session.execute(
                text("""
                    SELECT DISTINCT to_char(r.race_date, 'YYYY-MM-DD') AS d
                    FROM races r
                    WHERE SUBSTRING(r.race_id, 1, 4) = :yr
                      AND r.surface IN ('芝', 'ダート')
                      AND r.race_date < CURRENT_DATE
                      AND EXISTS (
                        SELECT 1 FROM race_results rr
                        WHERE rr.race_id = r.race_id AND rr.finish_pos > 0 AND rr.finish_time_sec > 0
                      )
                      AND NOT EXISTS (
                        SELECT 1 FROM megu_index mi
                        WHERE mi.race_id = r.race_id AND mi.model_version = 'v2'
                          AND mi.computation_status = 'valid'
                      )
                    ORDER BY d
                """),
                {"yr": str(yr)},
            ).fetchall()
            for row in rows:
                d = row.d
                r = compute_for_date(d)
                results.append(r)
                if r.get("status") == "skipped" and not SKIP_POLICY.get(r.get("reason", ""), {}).get(
                    "legitimate_skip"
                ):
                    logger.warning("compute skipped (needs backfill): %s %s", d, r)

    if results and all(x.get("status") == "ok" for x in results):
        _mark_task_done(_task_id("compute_missing_megu", "2020-2026"))
    return results


def _mark_task_done(task_id: str) -> None:
    store = _load_tasks()
    for t in store.get("tasks", []):
        if t.get("id") == task_id:
            t["status"] = "done"
            t["completed_at"] = datetime.now().isoformat(timespec="seconds")
    _save_tasks(store)


def print_task_summary() -> None:
    store = _load_tasks()
    tasks = store.get("tasks") or []
    pending = [t for t in tasks if t.get("status") != "done"]
    done = [t for t in tasks if t.get("status") == "done"]
    print(f"\n=== megu_index gap tasks ({TASKS_PATH}) ===")
    print(f"pending: {len(pending)}  done: {len(done)}  updated: {store.get('updated_at')}\n")
    for t in pending:
        skip = " [skip可]" if t.get("legitimate_skip") else ""
        print(f"  [P{t.get('priority')}] {t['id']}{skip}")
        print(f"       {t.get('title')} (n={t.get('affected_count')})")
        if t.get("command"):
            print(f"       → {t['command']}")
    if not pending:
        print("  （未完了タスクなし）")


def run_register_regression_params_v2(*, dry_run: bool = False) -> dict:
    """v1 回帰係数を v2 として DB に複製（NB-02 正本化前の暫定）。"""
    from src.db.session import get_session, init_engine

    init_engine()
    with get_session() as session:
        existing = session.execute(
            text("SELECT COUNT(*) FROM megu_regression_params WHERE model_version = 'v2'")
        ).scalar()
        if existing and int(existing) > 0:
            _mark_task_done(_task_id("register_regression_params_v2", "global"))
            return {"status": "already_registered", "v2_rows": int(existing)}

        rows = session.execute(
            text("""
                SELECT param_name, param_value, std_error, sample_count, fitted_at
                FROM megu_regression_params
                WHERE model_version = 'v1'
                ORDER BY param_name
            """)
        ).fetchall()
        if not rows:
            raise RuntimeError("megu_regression_params v1 が見つかりません")

        if dry_run:
            return {"status": "dry_run", "would_insert": len(rows)}

        for row in rows:
            session.execute(
                text("""
                    INSERT INTO megu_regression_params
                        (param_name, param_value, std_error, sample_count, model_version, fitted_at)
                    VALUES (:name, :val, :se, :n, 'v2', :fitted)
                    ON CONFLICT (param_name, model_version) DO UPDATE
                    SET param_value = EXCLUDED.param_value,
                        std_error = EXCLUDED.std_error,
                        sample_count = EXCLUDED.sample_count,
                        fitted_at = EXCLUDED.fitted_at
                """),
                {
                    "name": row.param_name,
                    "val": row.param_value,
                    "se": row.std_error,
                    "n": row.sample_count,
                    "fitted": row.fitted_at,
                },
            )

    _mark_task_done(_task_id("register_regression_params_v2", "global"))
    return {"status": "ok", "inserted": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="megu_index データギャップ監査・タスク管理")
    parser.add_argument("--refresh", action="store_true", help="監査実行して gap_tasks.json を更新")
    parser.add_argument("--summary", action="store_true", help="タスク一覧表示")
    parser.add_argument(
        "--run",
        choices=["repair_flat_metadata", "compute_missing", "register_regression_params_v2"],
        help="pending タスクのアクションを実行",
    )
    parser.add_argument("--year", type=int, help="--run repair_flat_metadata 対象年")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    script_basic_config()

    if args.refresh or (not args.summary and not args.run):
        refresh_audit()
        print_task_summary()
        return

    if args.summary:
        print_task_summary()
        return

    if args.run == "repair_flat_metadata":
        year = args.year or date.today().year
        r = run_repair_flat_metadata(year, dry_run=args.dry_run)
        print(json.dumps(r, indent=2, ensure_ascii=False))
        refresh_audit()
        return

    if args.run == "compute_missing":
        rs = run_compute_missing()
        print(json.dumps(rs, indent=2, ensure_ascii=False, default=str))
        refresh_audit()
        return

    if args.run == "register_regression_params_v2":
        r = run_register_regression_params_v2(dry_run=args.dry_run)
        print(json.dumps(r, indent=2, ensure_ascii=False))
        refresh_audit()
        return


if __name__ == "__main__":
    main()
