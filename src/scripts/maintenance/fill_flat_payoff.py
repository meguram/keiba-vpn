"""race_result_flat / race_result_race の払戻が空のレースを、保存済み race_result の払戻で埋める。

python -m src.scripts.maintenance.fill_flat_payoff --years 2019-2026            # dry-run（件数だけ）
python -m src.scripts.maintenance.fill_flat_payoff --years 2019-2026 --apply    # 退避してから parquet を書き換える

原因: export_tables が、JSON 列の長さが 1,024 バイトを超えると列ごと落としていた。同着や頭数の多いレースは払戻の JSON が 1,024 を超え、
保存本体には払戻があるのに平坦表だけ空になっていた（export_tables は払戻だけ上限を 8,192 に広げて修正済み。この道具は既に出力済みの表の穴埋め用）。
退避先は data/local/backup/flat_payoff_fill_<日時>/、結果は data/local/meta/flat_payoff_fill_report.json。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.scraper.storage import HybridStorage

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "data/page_reference/tables"
FILES = ("race_result_flat.parquet", "race_result_race.parquet")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2019-2026")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    lo, _, hi = a.years.partition("-")
    years = list(range(int(lo), int(hi or lo) + 1))
    s = HybridStorage()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    bdir = ROOT / f"data/local/backup/flat_payoff_fill_{stamp}"
    rep: dict = {"mode": "apply" if a.apply else "dry-run", "years": {}}
    for y in years:
        yr: dict = {}
        fill: dict[str, str] = {}
        for fname in FILES:
            f = TABLES / str(y) / fname
            if not f.exists():
                continue
            df = pd.read_parquet(f)
            if "payoff" not in df.columns:
                continue
            empty = df["payoff"].isna() | (df["payoff"].astype(str).str.strip().isin(["", "None", "{}", "null"]))
            rids = sorted(df.loc[empty, "race_id"].astype(str).unique())
            for rid in rids:
                if rid not in fill:
                    d = s.load("race_result", rid, bypass_cache=True)
                    p = d.get("payoff") if isinstance(d, dict) else None
                    fill[rid] = json.dumps(p, ensure_ascii=False) if p else ""
            fillable = [r for r in rids if fill.get(r)]
            yr[fname] = {"empty_races": len(rids), "fillable": len(fillable), "unfillable": [r for r in rids if not fill.get(r)]}
            if a.apply and fillable:
                bdir.mkdir(parents=True, exist_ok=True)
                (bdir / str(y)).mkdir(exist_ok=True)
                df.to_parquet(bdir / str(y) / fname, index=False)
                m = empty & df["race_id"].astype(str).isin(fillable)
                df.loc[m, "payoff"] = df.loc[m, "race_id"].astype(str).map(fill)
                tmp = f.with_suffix(".parquet.new")
                df.to_parquet(tmp, index=False)
                tmp.replace(f)
                yr[fname]["rows_filled"] = int(m.sum())
        rep["years"][str(y)] = yr
        print(y, json.dumps(yr, ensure_ascii=False), flush=True)
    rep["backup_dir"] = str(bdir) if a.apply else None
    out = ROOT / "data/local/meta/flat_payoff_fill_report.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
