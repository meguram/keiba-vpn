"""記事（@umatter_sub）の主張の検証 V1〜V3 が、親モデルの予測の残差に残っているか。

    python -m src.scripts.data.article_check_eval build          # 特徴量表を作る → <OUT>/v1.parquet, v2.parquet, v3.parquet
    python -m src.scripts.data.article_check_eval score          # 第 1 段: スコア検定 + 残差の木 + 層別（記述のみ） → <OUT>/score.json, boost.json, strata.json
    python -m src.scripts.data.article_check_eval tier2 V1,V2    # 第 2 段: 第 1 段を通った群だけ、test 2025 に 1 回だけ

候補・合格線は docs/operations/retrain_trials/README.md「記事検証 V1〜V3」に測る前に固定した（共通部品 residual_harness の定数）。
2026 年・市場オッズは使わない。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from src.pipeline.eval import residual_harness as H
from src.pipeline.features.article_check import ped14, prev_rescore, wkbias
from src.pipeline.features.article_check.common import load_runs

R = Path("/home/jovyan/work/retrain_out")
BASE = R / "base_2016"
OUT = BASE / "article_check"
OOF = BASE / "tp27_v6e/ens/v6.parquet"
T25 = BASE / "final_eval_v6e16/test2025_predictions.parquet"
TABLE = BASE / "features/train_table.parquet"
YEARS = range(2016, 2026)

V1 = prev_rescore.OUT_COLS
V2 = wkbias.OUT_COLS
V3R = ped14.raw_cols()
V3G = ped14.gated_cols()
SRCS = {"V1": V1, "V2": V2, "V3RAW": V3R, "V3GATE": V3G}
CANDS = [(c, c, None) for c in V1 + V2 + V3R + V3G]   # K = 10 + 9 + 12 + 12 = 43
STRATA_OF = {"V1": V1, "V2": V2, "V3RAW": V3R, "V3GATE": V3G}


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs(YEARS)
    prev_rescore.build(runs).to_parquet(OUT / "v1.parquet")
    print("v1 done", flush=True)
    t = pd.read_parquet(TABLE, columns=["race_id", "horse_id", "a_style"])
    t["race_id"], t["horse_id"] = t["race_id"].astype(str), t["horse_id"].astype(str)
    assert not t.duplicated(["race_id", "horse_id"]).any(), "train_table のキーが重複"
    d = wkbias.prepare(runs.merge(t, on=["race_id", "horse_id"], how="left"))
    wkbias.build(d).to_parquet(OUT / "v2.parquet")
    print("v2 done", flush=True)
    pf = OUT / "parents.parquet"
    if pf.exists():
        parents = pd.read_parquet(pf)
    else:
        parents = ped14.parent_ids(runs["horse_id"].unique())
        parents.to_parquet(pf)
    print("parents", len(parents), (parents["sire"] != "").mean(), (parents["dam"] != "").mean(), flush=True)
    ped14.build(runs, parents).to_parquet(OUT / "v3.parquet")
    print("v3 done", flush=True)


def _load(pwin_path: Path) -> pd.DataFrame:
    p = pd.read_parquet(pwin_path)
    p["race_id"], p["horse_id"] = p["race_id"].astype(str), p["horse_id"].astype(str)
    cols = ["race_id", "horse_id", "date", "venue_code", "surface", "distance", "track_condition", "grade", "field_size", "horse_number", "finish_position", "a_style", "a_l3fz_ewm2"]
    t = pd.read_parquet(TABLE, columns=cols)
    t["race_id"], t["horse_id"] = t["race_id"].astype(str), t["horse_id"].astype(str)
    t["venue_code"], t["surface"] = t["venue_code"].astype(str), t["surface"].astype(str)
    t["year"] = pd.to_datetime(t["date"].astype(str).str[:10]).dt.year
    t["gate"] = (t["horse_number"] - 1) / (t["field_size"] - 1).clip(lower=1)
    keep = [c for c in p.columns if c in ("race_id", "horse_id", "p_win", "finish_position")]
    if "finish_position" in keep:
        t = t.drop(columns="finish_position")
    d = p[keep].merge(t.drop(columns=["date"]), on=["race_id", "horse_id"], how="left")
    for n in ("v1", "v2", "v3"):
        f = pd.read_parquet(OUT / f"{n}.parquet")
        f["race_id"], f["horse_id"] = f["race_id"].astype(str), f["horse_id"].astype(str)
        assert not f.duplicated(["race_id", "horse_id"]).any(), f"{n} のキーが重複"
        d = d.merge(f, on=["race_id", "horse_id"], how="left")
    return d


def _strata(d: pd.DataFrame) -> dict[str, pd.Series]:
    n = d["ped14_nruns"]
    return {"career<=3": n <= 3, "career4-9": (n >= 4) & (n <= 9), "career10+": n >= 10, "turf": d["surface"] == "芝", "dirt": d["surface"] == "ダート", "field17+": d["field_size"] >= 17}


def score() -> None:
    d = H.prepare(_load(OOF))
    d = d[d["surface"].isin(H.SURFACES)].reset_index(drop=True)
    rows = H.score_test(d, CANDS)
    (OUT / "score.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in rows:
        print(r, flush=True)
    st = {}
    for sname, mask in _strata(d).items():
        sub = d[mask].reset_index(drop=True)
        st[sname] = {r["name"]: {"n": r["n"], "z": r["z"], "beta": r["beta"], "z_by_year": r["z_by_year"]} for r in H.score_test(sub, CANDS)}
    (OUT / "strata.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    b = H.boost_test(d, SRCS)
    (OUT / "boost.json").write_text(json.dumps(b, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(b, ensure_ascii=False), flush=True)


def _passed_stage1() -> set[str]:
    """第 1 段を通った群（残差の木が合格、またはスコア検定の合格列を含む）。ALL は 1 群でも通っていれば可。"""
    boost = json.loads((OUT / "boost.json").read_text(encoding="utf-8"))
    score_pass = {r["name"] for r in json.loads((OUT / "score.json").read_text(encoding="utf-8")) if r["pass"]}
    ok = {g for g in SRCS if boost.get(g, {}).get("pass") or score_pass & set(SRCS[g])}
    return ok | ({"ALL"} if ok else set())


def tier2(names: list[str]) -> None:
    bad = [n for n in names if n not in _passed_stage1()]
    if bad:
        sys.exit(f"第 1 段を通っていない群は test 2025 に当てない: {bad}")
    tr = H.prepare(_load(OOF))
    te = H.prepare(_load(T25))
    srcs = {**SRCS, "ALL": sum(SRCS.values(), [])}
    res = H.tier2(tr, te, {n: srcs[n] for n in names})
    (OUT / "tier2_TEST2025_EVALUATED.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        build()
    elif cmd == "score":
        score()
    elif cmd == "tier2" and len(sys.argv) > 2:
        if (OUT / "tier2_TEST2025_EVALUATED.json").exists():
            sys.exit("tier2 は test 2025 を 1 回だけ使う。出力が既にあるので止める（再実行するなら出力を退避してから）")
        tier2(sys.argv[2].split(","))
    else:
        sys.exit("使い方: build | score | tier2 V1,V2,V3RAW,V3GATE,ALL")
