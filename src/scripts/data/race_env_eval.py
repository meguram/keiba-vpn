"""レース環境 ENV（層 A）が、親モデルの予測の残差に残っているか。再学習の候補評価（s10b）。

    python -m src.scripts.data.race_env_eval score   # 第 1 段: スコア検定 + 残差の木 → <OUT>/score.json, boost.json
    python -m src.scripts.data.race_env_eval tier2 GL,BIAS   # 第 2 段: 第 1 段を通った群だけ、test 2025 に 1 回だけ適用

候補・合格線は docs/operations/retrain_trials/README.md「レース環境 ENV（層 A）」に測る前に固定した（共通部品 residual_harness の定数）。
2026 年・市場オッズは使わない。表は build_race_env.py が作る env_horse.parquet。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from src.pipeline.eval import residual_harness as H

R = Path("/home/jovyan/work/retrain_out")
BASE = R / "base_2016"
OUT = BASE / "race_env"
OOF = BASE / "tp27_v6e/ens/v6.parquet"
T25 = BASE / "final_eval_v6e16/test2025_predictions.parquet"
TABLE = BASE / "features/train_table.parquet"

GL = [f"env_gl_dev_{m}_{w}" for m in ("time", "front", "l3f") for w in ("prior", "day")]
BIAS = ["env_bias_bias_front_prior", "env_bias_bias_front_day"]
SRCS = {"GL": GL + ["env_gl_prior_gap", "env_gl_day_n"], "BIAS": BIAS}
CANDS = [(c, c, None) for c in GL + BIAS] + [(f"{c}_x_s", c, "s") for c in BIAS]   # K = 10


def _load(pwin_path: Path) -> pd.DataFrame:
    p = pd.read_parquet(pwin_path)
    p["race_id"] = p["race_id"].astype(str)
    p["horse_id"] = p["horse_id"].astype(str)
    cols = ["race_id", "horse_id", "date", "venue_code", "surface", "distance", "track_condition", "grade", "field_size", "horse_number", "finish_position", "a_style", "a_l3fz_ewm2"]
    t = pd.read_parquet(TABLE, columns=cols)
    t["race_id"] = t["race_id"].astype(str)
    t["horse_id"] = t["horse_id"].astype(str)
    t["venue_code"] = t["venue_code"].astype(str)
    t["surface"] = t["surface"].astype(str)
    t["year"] = pd.to_datetime(t["date"].astype(str).str[:10]).dt.year
    t["gate"] = (t["horse_number"] - 1) / (t["field_size"] - 1).clip(lower=1)
    keep = [c for c in p.columns if c in ("race_id", "horse_id", "p_win", "finish_position")]
    if "finish_position" in keep:
        t = t.drop(columns="finish_position")
    d = p[keep].merge(t.drop(columns=["date"]), on=["race_id", "horse_id"], how="left")
    env = pd.read_parquet(OUT / "env_horse.parquet")
    return d.merge(env, on=["race_id", "horse_id"], how="left")


def score() -> None:
    d = H.prepare(_load(OOF))
    d = d[d["surface"].isin(H.SURFACES)].reset_index(drop=True)
    rows = H.score_test(d, CANDS)
    (OUT / "score.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in rows:
        print(r, flush=True)
    b = H.boost_test(d, SRCS)
    (OUT / "boost.json").write_text(json.dumps(b, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(b, ensure_ascii=False), flush=True)


def tier2(names: list[str]) -> None:
    tr = H.prepare(_load(OOF))
    te = H.prepare(_load(T25))
    srcs = {**SRCS, "ALL": sum(SRCS.values(), [])}
    res = H.tier2(tr, te, {n: srcs[n] for n in names})
    (OUT / "tier2_TEST2025_EVALUATED.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "score":
        score()
    elif cmd == "tier2" and len(sys.argv) > 2:
        if (OUT / "tier2_TEST2025_EVALUATED.json").exists():
            sys.exit("tier2 は test 2025 を 1 回だけ使う。出力が既にあるので止める（再実行するなら出力を退避してから）")
        tier2(sys.argv[2].split(","))
    else:
        sys.exit("使い方: score | tier2 GL,BIAS,ALL")
