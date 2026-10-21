"""tp-64: 展開利の特徴量（pb_*）を V5 の構成 d（1,142 列）に足したときの増分を評価する（v6_eval と同じ手順・同じ基準）。

python -m src.scripts.data.tp64_eval run NAME [--tag s2 --years 2024 --seeds 2,3]
python -m src.scripts.data.tp64_eval compare [--tag s2]          # 基準 d_CY_BEST（v6/eval[_tag]）との対の比較

構成（すべて d_CY_BEST に足す）:
  PB_HIST  過去走の展開利・展開補正後の着順・恵まれ/不利フラグ・位置の安定度（今走の事前を使わない）
  PB_GAP   今走の期待展開利と、前走（直近3走）の展開利とのずれ（pb_exp_today / pb_gap_1 / pb_gap_m3 / レース内相対）
  PB_ALL   上の両方
基準 OOF は v6_eval run d_CY_BEST（valid 2022〜24・seed 0,1）と同じ設定。test 2025・2026 年は使わない。市場オッズ・人気は使わない。
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TP40_THREADS", os.environ.get("V6_THREADS", "4"))
os.environ.setdefault("TP40_LOAD_LIMIT", os.environ.get("V6_LOAD_LIMIT", "14"))

from src.pipeline.features import tp64_pace_benefit as pb  # noqa: E402
from src.scripts.data import tp27_eval as t  # noqa: E402
from src.scripts.data import tp40_eval as e40  # noqa: E402
from src.scripts.data import v5_select_eval as sel  # noqa: E402
from src.scripts.data import v6_eval as v6  # noqa: E402

KEY = ["race_id", "horse_id"]
FEATS = Path(os.environ.get("TP64_OUT") or "/home/jovyan/work/tp64_out") / "pb_features.parquet"
OUT = Path(os.environ.get("TP64_EVAL") or e40.V / "tp64_eval")
HIST = ["pb_ben_1", "pb_ben_m3", "pb_ben_m5", "pb_ben_max3", "pb_adj_1", "pb_adj_m3", "pb_adj_best3",
        "pb_lucky_good", "pb_hard_good", "pb_lucky_bad", "pb_unlucky_bad", "pb_pos_m3", "pb_pos_sd5", "pb_nruns"]
GAP = ["pb_exp_today", "pb_gap_1", "pb_gap_m3", "pb_exp_rel", "pb_exp_rank"]
CONFIGS = {"PB_HIST": HIST, "PB_GAP": GAP, "PB_ALL": HIST + GAP}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _paths(tag: str) -> tuple[Path, Path]:
    d = OUT / ("eval" if not tag else f"eval_{tag}")
    d.mkdir(parents=True, exist_ok=True)
    return d, d / "levels.json"


def build_frame(cols: list[str], df: pd.DataFrame) -> pd.DataFrame:
    X = sel.build_frame(v6.BASE_CFG | {}, df)
    G = pd.read_parquet(FEATS)
    G["race_id"], G["horse_id"] = G["race_id"].astype(str), G["horse_id"].astype(str)
    m = df[KEY].astype(str).merge(G[KEY + cols], on=KEY, how="left")
    assert len(m) == len(df)
    m.index = X.index
    dup = set(cols) & set(X.columns)
    assert not dup, f"列名が重なる: {sorted(dup)}"
    return pd.concat([X, m[cols].astype("float32")], axis=1)


def cmd_run(a) -> None:
    d, resp = _paths(a.tag)
    years = [int(x) for x in a.years.split(",")] if a.years else v6.YEARS
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else v6.SEEDS
    res = e40._jload(resp) or {}
    if a.name in res and (d / f"oof_{a.name}.parquet").exists():
        _log(f"{a.name}: 既に完了")
        return
    df, base, meta, folds, lp = t.setup()
    X = build_frame(CONFIGS[a.name], df)
    _log(f"{a.name}: 列 {X.shape[1]}（基準 d は 1,142 列）")
    lv, oof = e40._run_subset(a.name, X, meta, folds, lp, years, seeds)
    oof.to_parquet(d / f"oof_{a.name}.parquet", index=False)
    res = e40._jload(resp) or {}
    res[a.name] = {"levels": lv, "cols": int(X.shape[1]), "new": CONFIGS[a.name], "years": years, "seeds": seeds}
    e40._jsave(resp, res)


def cmd_compare(a) -> None:
    d, resp = _paths(a.tag)
    base_path = v6.OUT / ("eval" if not a.tag else f"eval_{a.tag}") / "oof_d_CY_BEST.parquet"
    df, base, meta, folds, lp = t.setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    rm = pd.read_parquet(e40.OUT / "run_meta.parquet")
    rm["race_id"], rm["horse_id"] = rm["race_id"].astype(str), rm["horse_id"].astype(str)
    o0 = pd.read_parquet(base_path)
    out: dict = {"pairs": {}, "baseline": str(base_path)}
    for n in CONFIGS:
        p = d / f"oof_{n}.parquet"
        if not p.exists():
            continue
        o = pd.read_parquet(p)
        c = t._cmp(o0, o, fin_map, f"{n} vs d")
        by_year = {}
        for y in v6.YEARS:
            ids = set(meta.loc[meta["year"] == y, "race_id"].astype(str))
            ma, mb = o0["race_id"].astype(str).isin(ids), o["race_id"].astype(str).isin(ids)
            if ma.sum() > 1000 and mb.sum() > 1000:
                cy = t._cmp(o0[ma], o[mb], fin_map, f"{n} vs d {y}")
                by_year[str(y)] = {k: cy[k] for k in ("Δwin", "win_hi", "Δtop2", "Δtop3")}
        sub = e40._subset_cmp(o0, o, fin_map, rm, f"{n} vs d")
        out["pairs"][n] = {"overall": c, "by_year": by_year, "subsets": sub}
        _log(f"{n} vs d: Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f} 採用={c['採用']}")
    out["levels"] = e40._jload(resp)
    e40._jsave(d / "results.json", out)


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("name", choices=list(CONFIGS))
    r.add_argument("--tag", default="")
    r.add_argument("--years", default="")
    r.add_argument("--seeds", default="")
    c = sp.add_parser("compare")
    c.add_argument("--tag", default="")
    a = ap.parse_args()
    {"run": cmd_run, "compare": cmd_compare}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
