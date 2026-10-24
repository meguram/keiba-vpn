"""v4 の推論用の列（serving_v4）を、学習時の同じ行の値と群ごとに比べる。

python -m src.scripts.data.verify_serving_v4 --days 20250302 20250614 --out /tmp/serve_v4_parity.json [--groups owner self8 field7 ped]
  履歴は「その日より前」だけ。当日の行は出馬表（race_shutuba_flat）から作り、結果は与えない。
  field7 の当日の入力（V0 の列）は、学習表（fixed）の同じ行を使う（V0 の再現は verify_v3_serving が見る）。
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from src.pipeline.features import serving_v4 as sv

ROOT = Path(__file__).resolve().parents[3]
FIXED = ROOT / "data/local/modeling/training_pipeline_fixed"
KEY = ["race_id", "horse_id"]


def _train_keys() -> pd.DataFrame:
    from src.pipeline.models import ablation as ab

    t = ab.load_table(FIXED / "features/train_table.parquet")
    t["race_id"], t["horse_id"] = t["race_id"].astype(str), t["horse_id"].astype(str)
    return t


def _ref(kind: str, name: str, keys: pd.DataFrame) -> pd.DataFrame:
    if kind == "conn":
        G = pd.read_parquet(sv.FAC / "conn/groups" / f"{name}.parquet")
        G[KEY[0]], G[KEY[1]] = keys["race_id"].to_numpy(), keys["horse_id"].to_numpy()
    elif kind == "self":
        G = pd.read_parquet(sv.FAC / "self" / f"{name}.parquet")
    elif kind == "field":
        g, pat = {"course_draw_post": ("course_draw", r"^fdr_(course|br|hn|third|cond_third|fs_third)_")}.get(name, (name, None))
        G = pd.read_parquet(sv.FAC / "field" / f"{g}.parquet")
        if pat:
            G = G[KEY + [c for c in G.columns if c not in KEY and re.search(pat, c)]]
    else:
        G = pd.read_parquet(sv.FAC / "src" / f"cand_{name}.parquet")
    G["race_id"], G["horse_id"] = G["race_id"].astype(str), G["horse_id"].astype(str)
    return G


def _compare(got: pd.DataFrame, ref: pd.DataFrame) -> dict:
    m = got.merge(ref, on=KEY, suffixes=("", "__r"))
    cols = [c for c in ref.columns if c not in KEY and c in got.columns]
    bad = {}
    for c in cols:
        a, b = m[c].to_numpy(dtype=float), m[c + "__r"].to_numpy(dtype=float)
        ok = (np.isnan(a) & np.isnan(b)) | np.isclose(a, b, rtol=1e-4, atol=1e-5)
        if not ok.all():
            d = np.abs(a - b)
            bad[c] = {"mismatch": float(1 - ok.mean()), "max_abs_diff": float(np.nanmax(d)) if np.isfinite(d).any() else None}
    return {"rows": int(len(m)), "ref_cols": len(ref.columns) - 2, "compared_cols": len(cols), "missing_cols": [c for c in ref.columns if c not in KEY and c not in got.columns][:10],
            "bad_cols": len(bad), "worst": dict(sorted(bad.items(), key=lambda kv: -kv[1]["mismatch"])[:5])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--groups", nargs="+", default=["owner", "self8", "field7", "ped"])
    ap.add_argument("--max-races", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    keys = _train_keys()
    hist = sv.V4History(train_table=FIXED / "features/train_table.parquet")
    info = hist.warm()
    G = sv.V4Groups(hist)
    report: dict = {"warm": info, "days": {}}
    for day in a.days:
        d = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        card = pl.read_parquet(sv.TABLES / day[:4] / "race_shutuba_flat.parquet").filter(pl.col("date").cast(pl.Utf8) == d).to_pandas()
        if a.max_races:
            card = card[card["race_id"].isin(sorted(card["race_id"].unique())[: a.max_races])]
        card = card[card["horse_id"].notna()].copy()
        res: dict = {"rows": len(card)}
        tgt = keys[keys["date"].astype(str).str[:10] == d] if "date" in keys.columns else keys
        if "owner" in a.groups:
            t1 = time.time()
            F = G.owner(card)
            res["owner_sec"] = round(time.time() - t1, 1)
            res["owner"] = {n: _compare(F, _ref("conn", n, keys)) for n in ("ent_o", "ent_ob", "win_o", "streak_o")}
        if "self8" in a.groups:
            t1 = time.time()
            F = G.self8(card)
            res["self8_sec"] = round(time.time() - t1, 1)
            res["self8"] = {n: _compare(F, _ref("self", n, keys)) for n in sv.SELF8}
        if "field7" in a.groups:
            t1 = time.time()
            T = tgt[tgt["race_id"].isin(card["race_id"].astype(str))].copy()
            T["date"] = pd.to_datetime(T["date"])
            F = G.field7(T)
            res["field7_sec"] = round(time.time() - t1, 1)
            res["field7"] = {n: _compare(F, _ref("field", n, keys)) for n in ("mem_strength", "mem_demo", "mem_conn", "course_draw_post", "calendar_env", "race_type")}
        if "ped" in a.groups:
            t1 = time.time()
            F = G.ped_cond(card)
            res["ped_sec"] = round(time.time() - t1, 1)
            res["ped"] = _compare(F, _ref("src", "PED_sire_cond", keys))
        report["days"][day] = res
        print(day, json.dumps({k: v for k, v in res.items() if k.endswith("_sec")}), flush=True)
    report["sec"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({d: {g: (v if isinstance(v, (int, float)) else {n: (x["bad_cols"], x["compared_cols"]) for n, x in v.items()} if isinstance(v, dict) and "bad_cols" not in v else (v["bad_cols"], v["compared_cols"]))
                      for g, v in r.items() if not g.endswith("_sec") and g != "rows"} for d, r in report["days"].items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
