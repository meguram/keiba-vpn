"""
2020/1/1 - 2026/5/20 の出走レース・出走馬について GCS データ存在チェック。

blob list キャッシュ (data/cache/_blob_list/{category}_{year}.json) を使用して
ネットワークアクセスなしに網羅性を確認する。

対象カテゴリ (cron-jobs ページの定期スクレイピングで収集するもの):
  Race-level  : race_result, race_shutuba, race_odds, race_pair_odds,
                race_index, race_paddock, race_trainer_comment, race_barometer,
                race_shutuba_past, race_oikiri, smartrc_race
  Horse-level : horse_pedigree_5gen

Usage:
    python3 scripts/gcs_coverage_check_2020_2026.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR  = ROOT / "data/local/tables"
BLOB_DIR    = ROOT / "data/cache/_blob_list"
REPORT_PATH = ROOT / "data/local/research/gcs_coverage_check_2020_2026.json"
CUTOFF_DATE = "2026-05-20"

RACE_CATEGORIES = [
    "race_result",
    "race_shutuba",
    "race_odds",
    "race_pair_odds",
    "race_index",
    "race_paddock",
    "race_trainer_comment",
    "race_barometer",
    "race_shutuba_past",
    "race_oikiri",
    "smartrc_race",
    "race_result_on_time",
]

HORSE_CATEGORIES = [
    "horse_pedigree_5gen",
    "horse_result",
    "horse_training",
]

# race_shutuba_past / race_oikiri / smartrc_race は特性に注意:
# - race_shutuba_past: 前日取得。過去分は weekly-update 対象外のため 2020-2023 は未取得の可能性大
# - race_oikiri     : 同上
# - smartrc_race    : 2026 から本格運用開始のため 2020-2025 は空が正常
# - race_result_on_time: レース当日速報。過去分は不要（race_result が確定版）

NOTE_CATEGORIES = {
    "race_shutuba_past": "前日取得。過去分(2020-2023)は定期スクレイプ対象外のため空が正常",
    "race_oikiri":       "前日取得。過去分(2020-2023)は定期スクレイプ対象外のため空が正常",
    "smartrc_race":      "2026年から本格運用開始。2020-2025は空が正常",
    "race_result_on_time": "当日速報。過去分は race_result（確定版）で代替されるため対象外",
    "horse_result":      "blob list キャッシュなし。GCS APIで直接確認が必要",
    "horse_training":    "blob list キャッシュなし。GCS APIで直接確認が必要",
    "race_barometer":    "タイム指数偏差値。2024年から本格収集開始（2020-2022は空が正常）",
}


def load_blob_cache(category: str, year: int) -> set[str]:
    p = BLOB_DIR / f"{category}_{year}.json"
    if not p.exists():
        return set()
    try:
        d = json.loads(p.read_text())
        return set(d.keys())
    except Exception:
        return set()


def load_all_blob_cache(category: str, years: list[int]) -> set[str]:
    result: set[str] = set()
    for y in years:
        result |= load_blob_cache(category, y)
    return result


def collect_race_ids_by_year() -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    for y in range(2020, 2027):
        p = TABLES_DIR / str(y) / "race_result_flat.parquet"
        if not p.exists():
            continue
        if y < 2026:
            t = pq.read_table(p, columns=["race_id"])
            rset = {str(r) for r in t.column("race_id").to_pylist() if r}
        else:
            t = pq.read_table(p, columns=["race_id", "date"])
            rids  = t.column("race_id").to_pylist()
            dates = t.column("date").to_pylist()
            rset = {str(r) for r, d in zip(rids, dates)
                    if r and d and str(d) <= CUTOFF_DATE}
        out[y] = rset
    return out


def collect_horse_ids(race_ids_by_year: dict[int, set[str]]) -> set[str]:
    out: set[str] = set()
    for y in range(2020, 2027):
        p = TABLES_DIR / str(y) / "race_result_flat.parquet"
        if not p.exists():
            continue
        if y < 2026:
            t = pq.read_table(p, columns=["horse_id"])
            for h in t.column("horse_id").to_pylist():
                if h:
                    out.add(str(h))
        else:
            t = pq.read_table(p, columns=["horse_id", "date"])
            hids  = t.column("horse_id").to_pylist()
            dates = t.column("date").to_pylist()
            for h, d in zip(hids, dates):
                if h and d and str(d) <= CUTOFF_DATE:
                    out.add(str(h))
    return out


def horse_prefix_year(horse_id: str) -> str:
    """horse_id の先頭4桁 (= 生年) を返す。blob list cache のキーに使う。"""
    return horse_id[:4] if len(horse_id) >= 4 else "0000"


def check_race_category(
    cat: str,
    race_ids_by_year: dict[int, set[str]],
) -> dict:
    years = sorted(race_ids_by_year.keys())
    has_cache = any((BLOB_DIR / f"{cat}_{y}.json").exists() for y in years)

    if not has_cache:
        return {
            "status": "no_cache",
            "note": NOTE_CATEGORIES.get(cat, "blob list キャッシュファイルが存在しません"),
        }

    rows = {}
    total_expected = 0
    total_found = 0
    total_missing = 0

    for y in years:
        expected = race_ids_by_year.get(y, set())
        if not expected:
            continue
        gcs_keys = load_blob_cache(cat, y)
        found   = expected & gcs_keys
        missing = expected - gcs_keys
        total_expected += len(expected)
        total_found    += len(found)
        total_missing  += len(missing)
        rows[y] = {
            "expected": len(expected),
            "found": len(found),
            "missing": len(missing),
            "coverage_pct": round(100 * len(found) / len(expected), 1) if expected else 100.0,
            "missing_ids": sorted(missing)[:20],  # 先頭20件のみ
        }

    coverage_pct = round(100 * total_found / total_expected, 2) if total_expected else 0.0
    status = (
        "ok"       if total_missing == 0 else
        "partial"  if total_missing < total_expected * 0.1 else
        "warning"  if total_missing < total_expected * 0.5 else
        "missing"
    )
    return {
        "status": status,
        "total_expected": total_expected,
        "total_found": total_found,
        "total_missing": total_missing,
        "coverage_pct": coverage_pct,
        "by_year": rows,
        **({"note": NOTE_CATEGORIES[cat]} if cat in NOTE_CATEGORIES else {}),
    }


def check_horse_category(
    cat: str,
    horse_ids: set[str],
) -> dict:
    # horse categories の blob list は birth_year ベースのキー
    birth_years = sorted({horse_prefix_year(h) for h in horse_ids if horse_prefix_year(h).isdigit()})
    has_cache = any((BLOB_DIR / f"{cat}_{y}.json").exists() for y in birth_years)

    if not has_cache:
        return {
            "status": "no_cache",
            "note": NOTE_CATEGORIES.get(cat, "blob list キャッシュファイルが存在しません"),
        }

    gcs_keys: set[str] = set()
    for y in birth_years:
        gcs_keys |= load_blob_cache(cat, y)

    found   = horse_ids & gcs_keys
    missing = horse_ids - gcs_keys
    coverage_pct = round(100 * len(found) / len(horse_ids), 2) if horse_ids else 0.0
    status = (
        "ok"      if not missing else
        "partial" if len(missing) < len(horse_ids) * 0.1 else
        "warning" if len(missing) < len(horse_ids) * 0.5 else
        "missing"
    )
    return {
        "status": status,
        "total_expected": len(horse_ids),
        "total_found": len(found),
        "total_missing": len(missing),
        "coverage_pct": coverage_pct,
        "missing_ids_sample": sorted(missing)[:30],
        **({"note": NOTE_CATEGORIES[cat]} if cat in NOTE_CATEGORIES else {}),
    }


def fmt_row(cat: str, r: dict) -> str:
    if r["status"] == "no_cache":
        return f"  {'[NO CACHE]':12s} {cat:30s}  {r.get('note','')}"
    if "total_expected" not in r:
        return f"  {'[?]':12s} {cat}"
    icon = {"ok": "✓", "partial": "△", "warning": "▲", "missing": "✗"}.get(r["status"], "?")
    return (f"  {icon} {r['coverage_pct']:5.1f}%  {cat:30s}"
            f"  found={r['total_found']:5d}/{r['total_expected']:5d}"
            f"  missing={r['total_missing']}")


def main() -> None:
    t0 = time.time()
    print("=" * 65)
    print(f"GCS カバレッジチェック: 2020/1/1 - {CUTOFF_DATE}")
    print("=" * 65)

    print("\n[0] Parquet から対象レース・馬 ID 収集中...", flush=True)
    race_ids_by_year = collect_race_ids_by_year()
    all_race_ids = set().union(*race_ids_by_year.values())
    horse_ids = collect_horse_ids(race_ids_by_year)
    print(f"    ユニークレース: {len(all_race_ids):,}  出走馬: {len(horse_ids):,}", flush=True)

    # ── Race categories ──
    print("\n[1] レースデータ カテゴリ チェック", flush=True)
    print("-" * 65)
    race_results: dict[str, dict] = {}
    for cat in RACE_CATEGORIES:
        r = check_race_category(cat, race_ids_by_year)
        race_results[cat] = r
        print(fmt_row(cat, r), flush=True)

    # ── Horse categories ──
    print("\n[2] 馬データ カテゴリ チェック", flush=True)
    print("-" * 65)
    horse_results: dict[str, dict] = {}
    for cat in HORSE_CATEGORIES:
        r = check_horse_category(cat, horse_ids)
        horse_results[cat] = r
        print(fmt_row(cat, r), flush=True)

    # ── Summary ──
    print("\n" + "=" * 65)
    print("【サマリー】")
    issues: list[str] = []
    for cat, r in {**race_results, **horse_results}.items():
        if r["status"] in ("warning", "missing"):
            issues.append(f"  ▲ {cat}: missing={r.get('total_missing','?')} / {r.get('total_expected','?')}")
        elif r["status"] == "partial":
            issues.append(f"  △ {cat}: missing={r.get('total_missing','?')} / {r.get('total_expected','?')} ({r.get('coverage_pct')}%)")
        elif r["status"] == "no_cache":
            issues.append(f"  [?] {cat}: キャッシュ未生成 - 要 GCS API 確認")
    if issues:
        print("要確認 / 不足:")
        for i in issues:
            print(i)
    else:
        print("  ✓ すべてのキャッシュ済みカテゴリでデータが揃っています")

    print(f"\n処理時間: {time.time()-t0:.1f}s")

    # ── レポート保存 ──
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cutoff_date": CUTOFF_DATE,
        "scope": {
            "total_races": len(all_race_ids),
            "total_horses": len(horse_ids),
            "years": sorted(race_ids_by_year.keys()),
            "races_by_year": {y: len(s) for y, s in sorted(race_ids_by_year.items())},
        },
        "race_categories": race_results,
        "horse_categories": horse_results,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"レポート: {REPORT_PATH}")


if __name__ == "__main__":
    main()
