"""
ストレージ上の既存 JSON を走査し、型・必須キー・既知の品質問題を検査する。

  python -m scraper.validate_storage
  python -m scraper.validate_storage --category race_shutuba_past --max-files 2000
  python -m scraper.validate_storage --json-out data/meta/validate_report.json

GCS が無効な環境では race_lists（ローカル）のみ列挙・検査される。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from scraper.storage import HybridStorage
from scraper.schemas import validate as validate_schema, SCHEMAS
from utils.keiba_logging import script_basic_config

logger = logging.getLogger(__name__)

# entries を主に持つレース系（空は警告）
_RACE_ENTRY_CATS = frozenset(
    {
        "race_shutuba",
        "race_result",
        "race_result_on_time",
        "race_index",
        "race_info",
        "race_lap",
        "race_lap_on_time",
        "race_oikiri",
        "race_paddock",
        "race_barometer",
        "race_return",
        "race_trainer_comment",
        "race_odds",
        "race_shutuba_past",
        "race_detail",
        "smartrc_race",
        "race_predictions",
    }
)

_HORSE_RESULT_LIKE = frozenset({"horse_result", "horse_training"})


def _issues_for(
    category: str, key: str, data: dict[str, Any] | None,
) -> tuple[list[str], list[str]]:
    """(errors, warnings) を返す。"""
    err: list[str] = []
    warn: list[str] = []

    if data is None:
        return (["load_returned_none"], [])
    if not isinstance(data, dict):
        return (["not_json_object"], [])

    meta = data.get("_meta")
    if not isinstance(meta, dict):
        warn.append("meta_missing_or_invalid")
    else:
        if not meta.get("scraped_at"):
            warn.append("no_scraped_at")
        dq = meta.get("data_quality")
        if category in ("race_shutuba", "race_result", "race_shutuba_past"):
            if not isinstance(dq, dict):
                warn.append("no_data_quality_meta")
            elif dq.get("level") in ("partial", "empty"):
                warn.append(f"data_quality_{dq.get('level')}")

    if category in _RACE_ENTRY_CATS:
        ent = data.get("entries")
        if ent is None:
            err.append("entries_key_missing")
        elif not isinstance(ent, list):
            err.append("entries_not_list")
        elif len(ent) == 0:
            warn.append("entries_empty")
        else:
            if category == "race_shutuba":
                rid = data.get("race_id")
                if rid is not None and str(rid) != str(key):
                    err.append("race_id_mismatch")
                if len(ent) > 0 and len(ent) < 4:
                    warn.append("low_entry_count_lt4")
                n_waku = sum(
                    1 for e in ent if int((e or {}).get("bracket_number") or 0) > 0
                )
                n_uma = sum(
                    1 for e in ent if int((e or {}).get("horse_number") or 0) > 0
                )
                if len(ent) >= 4 and n_waku < len(ent) // 2:
                    warn.append("many_missing_waku")
                if len(ent) >= 4 and n_uma < len(ent) // 2:
                    warn.append("many_missing_umaban")

            if category == "race_shutuba_past":
                n_past = sum(len((e or {}).get("past_races") or []) for e in ent)
                n_train = len(data.get("training") or [])
                if n_past == 0 and n_train == 0:
                    warn.append("no_past_and_no_training")
                truncated = 0
                long_raw = 0
                for e in ent:
                    for pr in (e or {}).get("past_races") or []:
                        if not isinstance(pr, dict):
                            continue
                        raw = pr.get("raw") or ""
                        if isinstance(raw, str):
                            if len(raw) == 80:
                                truncated += 1
                            elif len(raw) > 80:
                                long_raw += 1
                if truncated > 0 and long_raw == 0 and n_past > 0:
                    warn.append("past_raw_all_len80_legacy_truncation")
                elif truncated > 0:
                    warn.append("past_raw_some_len80_possible_legacy")

    if category in _HORSE_RESULT_LIKE:
        if not (data.get("horse_name") or "").strip():
            warn.append("horse_name_empty")

    if category == "horse_pedigree_5gen":
        if not data.get("pedigree") and not data.get("generations"):
            warn.append("pedigree_structure_sparse")

    # ── スキーマバリデーション（既に _meta に記録されていれば参照、なければ実行） ──
    sv = (meta or {}).get("schema_validation") if isinstance(meta, dict) else None
    if sv is None and category in SCHEMAS:
        sv = validate_schema(category, data)
    if isinstance(sv, dict) and not sv.get("passed", True):
        if sv.get("top_missing"):
            err.append(f"schema_top_missing:{','.join(sv['top_missing'])}")
        if sv.get("top_type_errors"):
            for te in sv["top_type_errors"]:
                err.append(f"schema_type_error:{te.get('field','?')}")
        if sv.get("top_constraint_errors"):
            for ce in sv["top_constraint_errors"]:
                warn.append(f"schema_constraint:{ce.get('field','?')}={ce.get('detail','')}")
        ei = sv.get("entry_issues") or {}
        for field, cnt in (ei.get("missing_field_counts") or {}).items():
            warn.append(f"schema_entry_missing:{field}({cnt})")
        for field, cnt in (ei.get("type_error_counts") or {}).items():
            warn.append(f"schema_entry_type:{field}({cnt})")
    elif isinstance(sv, dict) and sv.get("skipped"):
        pass  # no schema defined for this category
    elif sv is None and category in SCHEMAS:
        warn.append("no_schema_validation_meta")

    return (err, warn)


def _collect_tasks(
    storage: HybridStorage,
    categories: list[str],
    key_prefix: str = "",
) -> list[tuple[str, str]]:
    """key_prefix が 4 桁年なら list_keys(..., year) で GCS 列挙を絞る。"""
    tasks: list[tuple[str, str]] = []
    list_year: str | None = None
    if len(key_prefix) >= 4 and key_prefix[:4].isdigit():
        list_year = key_prefix[:4]

    for cat in categories:
        try:
            keys = (
                storage.list_keys(cat, list_year)
                if list_year
                else storage.list_keys(cat)
            )
        except Exception as e:
            logger.warning("list_keys 失敗 %s: %s", cat, e)
            continue
        for k in keys:
            if key_prefix and not str(k).startswith(key_prefix):
                continue
            tasks.append((cat, k))
    return tasks


def run_validate(
    storage: HybridStorage,
    *,
    categories: list[str] | None = None,
    max_files: int | None = None,
    key_prefix: str = "",
    workers: int = 6,
    bypass_cache: bool = False,
) -> dict[str, Any]:
    if categories is None:
        categories = [
            c
            for c, t in storage.CATEGORY_MAP.items()
            if t not in ("local_only", "other")
        ]
        categories = sorted(set(categories))

    tasks = _collect_tasks(storage, categories, key_prefix)
    if max_files is not None:
        tasks = tasks[:max_files]

    err_counts: dict[str, int] = defaultdict(int)
    warn_counts: dict[str, int] = defaultdict(int)
    err_samples: dict[str, list[tuple[str, str]]] = defaultdict(list)
    warn_samples: dict[str, list[tuple[str, str]]] = defaultdict(list)
    by_cat_files: dict[str, int] = defaultdict(int)

    def _one(item: tuple[str, str]) -> tuple[str, str, list[str], list[str]]:
        cat, key = item
        data = storage.load(cat, key, bypass_cache=bypass_cache)
        e, w = _issues_for(cat, key, data)
        return (cat, key, e, w)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, t) for t in tasks]
        for fut in as_completed(futs):
            cat, key, e, w = fut.result()
            by_cat_files[cat] += 1
            for code in e:
                err_counts[code] += 1
                err_samples[code].append((cat, key))
            for code in w:
                warn_counts[code] += 1
                warn_samples[code].append((cat, key))

    # race_lists ローカル
    rl_tasks: list[tuple[str, str]] = []
    try:
        for k in storage.list_keys("race_lists"):
            rl_tasks.append(("race_lists", k))
    except Exception as e:
        logger.warning("race_lists list_keys: %s", e)

    for cat, key in rl_tasks:
        data = storage.load(cat, key)
        e, w = _issues_for(cat, key, data)
        by_cat_files[cat] += 1
        for code in e:
            err_counts[code] += 1
            err_samples[code].append((cat, key))
        for code in w:
            warn_counts[code] += 1
            warn_samples[code].append((cat, key))

    return {
        "gcs_enabled": storage.gcs_enabled,
        "gcs_healthy": getattr(storage, "_gcs_healthy", False),
        "bypass_cache": bypass_cache,
        "key_prefix": key_prefix or None,
        "files_checked": len(tasks) + len(rl_tasks),
        "tasks_gcs": len(tasks),
        "tasks_race_lists": len(rl_tasks),
        "by_category": dict(sorted(by_cat_files.items())),
        "error_counts": dict(sorted(err_counts.items(), key=lambda x: -x[1])),
        "warning_counts": dict(sorted(warn_counts.items(), key=lambda x: -x[1])),
        "error_samples": {k: v for k, v in sorted(err_samples.items())},
        "warning_samples": {k: v for k, v in sorted(warn_samples.items())},
    }


_STRICT_MATCH_CATS = ["race_odds", "race_oikiri", "race_shutuba_past"]
_LOOSE_MATCH_CATS = ["race_result"]
_EXISTENCE_ONLY_CATS = [
    "race_index", "race_barometer", "race_paddock",
    "race_trainer_comment", "race_detail",
]
_ALL_CHECK_CATS = (
    ["race_shutuba"] + _STRICT_MATCH_CATS + _LOOSE_MATCH_CATS + _EXISTENCE_ONLY_CATS
)


def run_validate_recent(
    storage: HybridStorage,
    *,
    days: int = 35,
    workers: int = 4,
) -> dict[str, Any]:
    """直近 N 日分のデータ品質を検証する軽量バージョン。

    race_lists から日付 → race_id を逆引きして対象レースを特定する。

    チェック内容:
      1. 頭数整合 — race_lists の entries_count (または race_shutuba) を基準に
                    各カテゴリの entries 数が一致するか
      2. スキーマ整合 — SCHEMAS に定義があるカテゴリのスキーマチェック
    """
    from datetime import datetime, timedelta

    today = datetime.now()
    target_dates: set[str] = set()
    for delta in range(days):
        d = today - timedelta(days=delta)
        target_dates.add(d.strftime("%Y%m%d"))

    all_rl_keys = storage.list_keys("race_lists")
    race_info: dict[str, tuple[int, str]] = {}  # race_id → (entries_count, date_str)

    for date_key in sorted(target_dates):
        if date_key not in all_rl_keys:
            continue
        rl = storage.load("race_lists", date_key)
        if not rl or not isinstance(rl.get("races"), list):
            continue
        for r in rl["races"]:
            rid = r.get("race_id")
            if rid:
                race_info[str(rid)] = (
                    int(r.get("entries_count") or 0),
                    date_key,
                )

    race_ids = sorted(race_info.keys())
    today_str = today.strftime("%Y%m%d")

    err_counts: dict[str, int] = defaultdict(int)
    warn_counts: dict[str, int] = defaultdict(int)
    err_samples: dict[str, list[tuple[str, str]]] = defaultdict(list)
    warn_samples: dict[str, list[tuple[str, str]]] = defaultdict(list)
    by_cat_files: dict[str, int] = defaultdict(int)
    files_checked = 0

    def _check_race(
        rid: str, expected_count: int, race_date: str,
    ) -> list[tuple[str, int, list[str], list[str]]]:
        """1 レースの全カテゴリを検査し (cat, n_entries, errors, warnings) を返す。"""
        results = []
        is_past = race_date <= today_str

        ref_count = expected_count

        for cat in _ALL_CHECK_CATS:
            data = storage.load(cat, rid)
            errs: list[str] = []
            warns: list[str] = []

            if data is None:
                if cat == "race_shutuba":
                    errs.append("shutuba_missing")
                elif is_past and cat in _STRICT_MATCH_CATS:
                    warns.append("data_missing")
                results.append((cat, 0, errs, warns))
                continue

            ent = data.get("entries")
            n = len(ent) if isinstance(ent, list) else 0

            # 頭数整合 — 過去レースのみ厳密比較
            if is_past and cat in _STRICT_MATCH_CATS and ref_count > 0:
                if isinstance(ent, list) and n > 0 and n != ref_count:
                    errs.append(f"entry_count_mismatch:{n}vs{ref_count}")

            # race_result: 取消/除外で少なくなりうる（差4超で警告）
            if is_past and cat in _LOOSE_MATCH_CATS and ref_count > 0:
                if isinstance(ent, list) and n > 0 and abs(n - ref_count) > 4:
                    warns.append(f"result_count_diff:{n}vs{ref_count}")

            # race_shutuba: race_lists の頭数と大きくずれていれば警告（再取得推奨）
            if cat == "race_shutuba" and ref_count > 0:
                if isinstance(ent, list) and n > 0 and n != ref_count:
                    warns.append(f"shutuba_stale:{n}vs{ref_count}")

            # スキーマ整合
            if cat in SCHEMAS:
                sv = validate_schema(cat, data)
                if isinstance(sv, dict) and not sv.get("passed", True):
                    if sv.get("top_missing"):
                        errs.append(f"schema_top_missing:{','.join(sv['top_missing'])}")
                    if sv.get("top_type_errors"):
                        for te in sv["top_type_errors"]:
                            errs.append(f"schema_type:{te.get('field','?')}")
                    ei = sv.get("entry_issues") or {}
                    for field, cnt in (ei.get("missing_field_counts") or {}).items():
                        warns.append(f"schema_entry_missing:{field}({cnt})")

            results.append((cat, n, errs, warns))
        return results

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(_check_race, rid, race_info[rid][0], race_info[rid][1]): rid
            for rid in race_ids
        }
        for fut in as_completed(futs):
            rid = futs[fut]
            try:
                results = fut.result()
            except Exception:
                err_counts["check_exception"] += 1
                err_samples["check_exception"].append(("race_shutuba", rid))
                continue
            for cat, _n, errs, warns in results:
                by_cat_files[cat] += 1
                files_checked += 1
                for code in errs:
                    err_counts[code] += 1
                    if len(err_samples[code]) < 200:
                        err_samples[code].append((cat, rid))
                for code in warns:
                    warn_counts[code] += 1
                    if len(warn_samples[code]) < 200:
                        warn_samples[code].append((cat, rid))

    date_strs = sorted(target_dates)
    return {
        "scope": "recent",
        "days": days,
        "date_range": f"{date_strs[0]}–{date_strs[-1]}" if date_strs else "",
        "races_checked": len(race_ids),
        "files_checked": files_checked,
        "categories_checked": len(_ALL_CHECK_CATS),
        "by_category": dict(sorted(by_cat_files.items())),
        "error_counts": dict(sorted(err_counts.items(), key=lambda x: -x[1])),
        "warning_counts": dict(sorted(warn_counts.items(), key=lambda x: -x[1])),
        "error_samples": {k: v for k, v in sorted(err_samples.items())},
        "warning_samples": {k: v for k, v in sorted(warn_samples.items())},
    }


def main() -> int:
    script_basic_config()
    ap = argparse.ArgumentParser(description="ストレージ JSON 整合性チェック")
    ap.add_argument("--category", action="append", help="対象カテゴリ（複数可）")
    ap.add_argument("--max-files", type=int, default=None, help="先頭 N 件だけ（デバッグ用）")
    ap.add_argument(
        "--key-prefix",
        default="",
        help="race_id / horse_id の先頭一致で絞る（例: 2026 で当年分のみ）",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=6,
        help="並列数（大きすぎると GCS 5s タイムアウトやバックオフの原因になる）",
    )
    ap.add_argument(
        "--bypass-cache",
        action="store_true",
        help="load 時にキャッシュを無視し GCS 直読み（重いがキャッシュ汚染を避ける）",
    )
    ap.add_argument("--json-out", type=str, default="", help="レポート JSON の保存先")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    storage = HybridStorage(base_dir=str(root))

    cats = args.category if args.category else None
    report = run_validate(
        storage,
        categories=cats,
        max_files=args.max_files,
        key_prefix=(args.key_prefix or "").strip(),
        workers=max(1, args.workers),
        bypass_cache=args.bypass_cache,
    )

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Wrote {out}", file=sys.stderr)

    print("=== ストレージ検証サマリ ===")
    print(f"GCS: enabled={report['gcs_enabled']} healthy={report['gcs_healthy']}")
    print(f"検査ファイル数: {report['files_checked']} (GCS系 {report['tasks_gcs']} + race_lists {report['tasks_race_lists']})")
    print("\nカテゴリ別 件数:")
    for c, n in report["by_category"].items():
        print(f"  {c}: {n}")

    if report["error_counts"]:
        print("\n【エラー】")
        for code, n in report["error_counts"].items():
            print(f"  {code}: {n}")
            for pair in report["error_samples"].get(code, []):
                print(f"    - {pair[0]}/{pair[1]}")
    else:
        print("\n【エラー】なし")

    if report["warning_counts"]:
        print("\n【警告】")
        for code, n in report["warning_counts"].items():
            print(f"  {code}: {n}")
            for pair in report["warning_samples"].get(code, []):
                print(f"    - {pair[0]}/{pair[1]}")
    else:
        print("\n【警告】なし")

    return 1 if report["error_counts"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
