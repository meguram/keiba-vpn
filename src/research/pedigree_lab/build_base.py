"""段 0: 出走の土台の表と、馬ごとの 5 代血統の表を作る（手元のファイルだけ。GCS を読まない）。

出力（OUT=/home/jovyan/work/pedigree_lab_out）:
  runs.parquet  平地の全出走（2016〜2026）。着順・オッズは「目的変数・評価の基準線」にだけ使う。
  ped.parquet   馬ごとの血統（父・母・母父・父父・母母・母母父・5 代の祖先の id の並び）。

5 代の位置: 代 g の位置 p は 0..2^g-1。偶数が牡、p を 2 で割ると子。gen1: 0=父 1=母。gen2: 0=父父 1=父母 2=母父 3=母母。
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

MAIN = Path("/home/jovyan/work/keiba-vpn")
TABLES = MAIN / "data/page_reference/tables"
PED_DIRS = [MAIN / "data/local/horse_pedigree_5gen", MAIN / "data/local/horse_pedigree_10gen"]
OUT = Path(os.environ.get("PEDLAB_OUT", "/home/jovyan/work/pedigree_lab_out"))
COLS = ["race_id", "date", "venue", "venue_code", "surface", "distance", "direction", "grade", "race_class",
        "track_condition", "field_size", "course_type", "race_name", "finish_position", "finish_status",
        "horse_number", "bracket_number", "horse_id", "sex_age", "odds", "popularity", "payoff"]


def load_runs() -> pd.DataFrame:
    fr = []
    for y in range(2016, 2027):
        x = pd.read_parquet(TABLES / str(y) / "race_result_flat.parquet")      # 年で列が少し違う（course_type・finish_status）
        fr.append(x.reindex(columns=COLS))
    d = pd.concat(fr, ignore_index=True)
    d = d[d["surface"].isin(["芝", "ダート"])].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["meeting"] = d["race_id"].str[:8]
    d["sex"] = d["sex_age"].str[0]
    d["age"] = pd.to_numeric(d["sex_age"].str[1:], errors="coerce")
    return d


def _one(h: str):
    for base in PED_DIRS:
        f = base / h[:4] / f"{h}.json"
        if not f.exists():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        anc = data.get("ancestors") or []
        if not anc:
            continue
        g = {}
        for a in anc:
            gen, pos, i = a.get("generation"), a.get("position"), str(a.get("horse_id") or "")
            if gen and gen <= 5:
                g[(int(gen), int(pos))] = i
        return h, g
    return h, None


def load_ped(horse_ids) -> pd.DataFrame:
    ids = sorted({str(x) for x in horse_ids})
    with ProcessPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(_one, ids, chunksize=500))
    rows = []
    for h, g in res:
        if g is None:
            rows.append({"horse_id": h, "has_ped": False})
            continue
        r = {"horse_id": h, "has_ped": True,
             "sire": g.get((1, 0), ""), "dam": g.get((1, 1), ""),
             "ss": g.get((2, 0), ""), "sd": g.get((2, 1), ""), "ds": g.get((2, 2), ""), "dd": g.get((2, 3), ""),
             "sss": g.get((3, 0), ""), "dsd": g.get((3, 5), ""), "dds": g.get((3, 6), ""), "ddd": g.get((3, 7), ""),
             "ssss": g.get((4, 0), ""), "sssss": g.get((5, 0), ""), "ddds": g.get((4, 14), "")}
        r["anc5"] = ";".join(f"{k[0]}:{k[1]}:{v}" for k, v in sorted(g.items()) if v)
        r["n_anc"] = sum(1 for v in g.values() if v)
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    runs.to_parquet(OUT / "runs.parquet", index=False)
    print("runs", runs.shape, runs["date"].min(), runs["date"].max())
    ped = load_ped(runs["horse_id"].unique())
    ped.to_parquet(OUT / "ped.parquet", index=False)
    print("ped", ped.shape, "has_ped", ped["has_ped"].mean())


if __name__ == "__main__":
    main()
