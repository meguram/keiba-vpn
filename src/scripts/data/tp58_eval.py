"""tp-58 LightGBM での確認: V5（1,142 列）に掛け合わせ列を足して同じ設定（検証 2022/23/24・seed 0,1）で学習し、V5 の OOF と比べる。

python -m src.scripts.data.tp58_eval run --name pairs_top20 --pairs out/pairs_top20.json [--hand out/hand_cols.json] [--seeds 0,1] [--base-oof PATH]
  pairs: [{"op": "prod", "a": "col1", "b": "rel(col2)"}, ...]（基の列は V5 の列名）
  hand : ["x58_jk_tr_win_dev", ...]（tp58_hand build の出力列名）
結果: $TP58_DIR/out/lgb_<name>/{oof.parquet,result.json}。採用基準は feature_bakeoff.compare_oof（勝率 CI 上限<0・効果≥2e-4・連対/複勝が悪化しない）。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features import tp58_pairs as pp
from src.scripts.data import tp27_eval as t
from src.scripts.data import v5_select_eval as v5

ROOT = Path(__file__).resolve().parents[3]
WORKDIR = Path(os.environ.get("TP58_DIR", ROOT / "data/local/modeling/shared_cache/tp58"))
OUT = WORKDIR / "out"
BASE_OOF = ROOT / "data/local/modeling/training_pipeline_v4/v5_select/oof_d_CY_BEST.parquet"
KEY = ["race_id", "horse_id"]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_extra(X: pd.DataFrame, df: pd.DataFrame, pairs: list, hand: list) -> pd.DataFrame:
    cols = {}
    if pairs:
        names = sorted({n for p in pairs for n in (p["a"], p["b"])})
        src = X.copy()
        src["race_id"] = df["race_id"].to_numpy()
        consts = pp.fit_constants(src, names, (df["year"] <= 2021).to_numpy())
        for p in pairs:
            cols[pp.column_name(p["op"], p["a"], p["b"])] = pp.pair_column(src, p["op"], p["a"], p["b"], consts).astype("float32")
    if hand:
        H = pd.read_parquet(OUT / "hand_crosses.parquet")
        H["race_id"], H["horse_id"] = H["race_id"].astype(str), H["horse_id"].astype(str)
        M = df[KEY].merge(H, on=KEY, how="left")
        for c in hand:
            cols[c] = M[c].to_numpy(dtype="float32")
    return pd.DataFrame(cols, index=X.index)


def cmd_run(a) -> None:
    t.SEEDS = [int(s) for s in a.seeds.split(",")]
    df, base, meta, folds, lp = t.setup()
    X = v5.build_frame(v5.CONFIGS["d_CY_BEST"], df)
    pairs = json.loads(Path(a.pairs).read_text(encoding="utf-8")) if a.pairs else []
    hand = json.loads(Path(a.hand).read_text(encoding="utf-8")) if a.hand else []
    E = build_extra(X, df, pairs, hand)
    _log(f"追加 {E.shape[1]} 列（pairs={len(pairs)} hand={len(hand)}）")
    X2 = pd.concat([X, E], axis=1)
    assert not X2.columns.duplicated().any()
    od = OUT / f"lgb_{a.name}"
    od.mkdir(parents=True, exist_ok=True)
    lv, by, oof = t._run(a.name, X2, meta, folds, lp)
    oof.to_parquet(od / "oof.parquet", index=False)
    prev = pd.read_parquet(a.base_oof or BASE_OOF)
    for d in (prev, oof):
        d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    fin_map = meta.set_index(KEY)["finish_position"]
    cmp = t._cmp(prev, oof, fin_map, a.name)
    per_year = {}
    for y in (2022, 2023, 2024):
        py, oy = prev[prev["year"] == y], oof[oof["year"] == y]
        per_year[str(y)] = t._cmp(py, oy, fin_map, f"{a.name}_{y}")
    res = {"name": a.name, "n_added": int(E.shape[1]), "levels": lv, "by_fold": by, "compare": cmp, "per_year": per_year, "pairs": pairs, "hand": hand, "seeds": t.SEEDS}
    (od / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    _log(f"{a.name}: Δwin={cmp['Δwin']:.5f} 上限={cmp['win_hi']:.5f} Δtop2={cmp['Δtop2']:.5f} Δtop3={cmp['Δtop3']:.5f} 採用={cmp['採用']}")
    for y, c in per_year.items():
        _log(f"  {y}: Δwin={c['Δwin']:.5f} Δtop2={c['Δtop2']:.5f} Δtop3={c['Δtop3']:.5f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--pairs")
    r.add_argument("--hand")
    r.add_argument("--seeds", default="0,1")
    r.add_argument("--base-oof")
    a = ap.parse_args()
    {"run": cmd_run}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
