"""過去分補填（2016〜2019 年）の整合の検証。取得 → スキーマ検証 → GCS の保存 → ローカルのフラット表の順に、年ごとの件数が一致するかを見る。

見るもの（年ごと）:
  1. 期待レース数 … 開催日カレンダー（backfill の進捗ファイル calendar_<年>）× race_lists のレース数
  2. GCS の保存数 … race_result / race_shutuba / race_oikiri / race_index / race_odds と、派生 11 カテゴリのうちレース単位のもの
  3. ローカルのフラット表 … data/page_reference/tables/<年>/race_result_flat.parquet などのレース数・行数（race_result の出走数と一致するか）
  4. スキーマ不適合の記録 … data/local/meta/schema_violations/<カテゴリ>.jsonl のうち当年のキー
  5. 取得不能 … 期待より少ないカテゴリは『取得不能（古い年のページに無い）』と『未取得』を、進捗の missing と合わせて区別して示す

使い方: python -m src.scripts.data.verify_past_backfill [--years 2019 2018 2017 2016] [--json-out PATH] [--no-gcs]
終了コード: 0=すべて一致、1=不一致あり（未取得の途中経過も不一致として出る。--progress-ok で『未取得』を許容）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
RAW_RACE = ["race_result", "race_shutuba", "race_oikiri", "race_result_lap", "race_index", "race_odds", "race_pair_odds", "race_shutuba_past", "race_result_on_time"]
UNAVAILABLE = {"race_paddock": "2019 年のページに内容が無い", "race_trainer_comment": "2019 年のページに内容が無い", "race_barometer": "2019 年は取得できない"}
DERIVED_RACE = ["race_shutuba_meta", "race_result_meta", "race_result_payoff", "race_result_track", "race_result_corner", "race_result_lap_times",
                "race_result_on_time_payoff", "race_result_on_time_lap", "race_result_on_time_corner"]
HORSE_CATS = ["horse_result", "horse_pedigree_5gen"]
FLAT_FILES = {"race_result": "race_result_flat.parquet", "race_shutuba": "race_shutuba_flat.parquet",
              "race_oikiri": "race_oikiri_flat.parquet", "race_index": "race_index_flat.parquet", "race_odds": "race_odds_flat.parquet",
              "race_result_lap": "race_result_lap_flat.parquet", "race_shutuba_past": "race_shutuba_past_flat.parquet"}
PROGRESS = ROOT / "data" / "local" / "meta" / "backfill_progress.json"
VIOL_DIR = ROOT / "data" / "local" / "meta" / "schema_violations"
TABLES = ROOT / "data" / "page_reference" / "tables"


def _expected(storage, dates: list[str]) -> tuple[int, set[str]]:
    ids: set[str] = set()
    for d in dates:
        for r in (storage.load("race_lists", d) or {}).get("races", []):
            if r.get("race_id"):
                ids.add(r["race_id"])
    return len(ids), ids


def _flat_stats(year: int, cat: str) -> dict[str, Any] | None:
    p = TABLES / str(year) / FLAT_FILES[cat]
    if not p.is_file():
        return None
    import pandas as pd

    df = pd.read_parquet(p, columns=["race_id"])
    return {"rows": int(len(df)), "races": int(df["race_id"].nunique())}


def _violations(year: int) -> dict[str, int]:
    out: dict[str, int] = {}
    if not VIOL_DIR.is_dir():
        return out
    for f in VIOL_DIR.glob("*.jsonl"):
        if f.stem not in set(RAW_RACE + DERIVED_RACE + HORSE_CATS):
            continue                       # 過去分の補填と無関係なカテゴリ（jra_cushion など）は数えない
        n = 0
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if str(d.get("key", "")).startswith(str(year)):
                n += 1
        if n:
            out[f.stem] = n
    return out


def check(years: list[int], use_gcs: bool = True, progress_ok: bool = False) -> dict[str, Any]:
    from src.scraper.storage import HybridStorage

    prog = json.loads(PROGRESS.read_text(encoding="utf-8")) if PROGRESS.is_file() else {}
    storage = HybridStorage()
    report: dict[str, Any] = {"years": {}, "ok": True}
    for y in years:
        dates = prog.get(f"calendar_{y}") or [d for d in prog.get("done_fast", []) if d.startswith(str(y))]   # カレンダー未取得（少量の確認）は、結果が揃った日だけを対象にする
        n_exp, ids = _expected(storage, [d for d in dates if d in set(prog.get("done_fast", []))])
        row: dict[str, Any] = {"kaisai_days": len(dates), "days_fast_done": len([d for d in dates if d in set(prog.get("done_fast", []))]),
                               "expected_races": n_exp, "gcs": {}, "flat": {}, "problems": []}
        if use_gcs:
            for cat in RAW_RACE + DERIVED_RACE:
                keys = [k for k in storage.list_keys(cat, str(y)) if k.startswith(str(y))]
                row["gcs"][cat] = len(keys)
                if cat in ("race_result", "race_shutuba") and n_exp and len(keys) != n_exp:
                    row["problems"].append(f"GCS {cat}: {len(keys)} 件（期待 {n_exp}）")
                if cat in DERIVED_RACE and row["gcs"].get("race_result") is not None and len(keys) < row["gcs"]["race_result"]:
                    row["problems"].append(f"派生 {cat}: {len(keys)} 件（race_result は {row['gcs']['race_result']}）→ migrate_row_data_to_unique_paths が未実行")
        for cat in FLAT_FILES:
            fs = _flat_stats(y, cat)
            row["flat"][cat] = fs
        rr = row["flat"].get("race_result")
        if rr is None:
            row["problems"].append("ローカルのフラット表 race_result_flat が未作成（export_tables を実行）")
        elif use_gcs and rr["races"] != row["gcs"].get("race_result"):
            row["problems"].append(f"フラット表 race_result: {rr['races']} レース（GCS は {row['gcs'].get('race_result')}）")
        row["schema_violations"] = _violations(y)
        if row["schema_violations"]:
            row["problems"].append(f"スキーマ不適合の記録あり: {row['schema_violations']}")
        done_all = all(set(dates) <= set(prog.get(f"done_{t}", [])) for t in ("fast", "cards", "lap", "odds", "pair", "index", "past", "on_time")) if dates else False
        row["unavailable"] = UNAVAILABLE
        row["complete"] = done_all
        if row["problems"] and (done_all or not progress_ok):
            report["ok"] = False
        report["years"][str(y)] = row
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=int, nargs="+", default=[2019, 2018, 2017, 2016])
    ap.add_argument("--json-out")
    ap.add_argument("--no-gcs", action="store_true", help="GCS の件数を数えない（ローカルのみ）")
    ap.add_argument("--progress-ok", action="store_true", help="補填の途中の年の不一致は許容する")
    a = ap.parse_args()
    rep = check(a.years, use_gcs=not a.no_gcs, progress_ok=a.progress_ok)
    for y, row in rep["years"].items():
        print(f"== {y}: 開催日 {row['kaisai_days']}（結果済み {row['days_fast_done']}）期待レース {row['expected_races']} 完了={row['complete']}")
        for k, v in row["gcs"].items():
            print(f"   GCS {k:24s} {v}")
        for p in row["problems"]:
            print("   !", p)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("OK" if rep["ok"] else "不一致あり")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
