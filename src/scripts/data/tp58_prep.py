"""tp-58 の下準備: V5 の列セット（構成 d、1,142 列）の表を共有キャッシュに保存し、LightGBM の gain 重要度を出す。

python -m src.scripts.data.tp58_prep frame       # 表（float32）・メタ・生の ID 列を data/local/modeling/shared_cache/tp58/ に保存
python -m src.scripts.data.tp58_prep importance  # 学習は 2021 年まで（2022 年以降は見ない）の LightGBM で gain 重要度

test 2025・2026 年は使わない。市場オッズ・人気は使わない。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import tp27_eval as t
from src.scripts.data import v5_select_eval as v5

ROOT = Path(__file__).resolve().parents[3]
WORKDIR = Path(os.environ.get("TP58_DIR", ROOT / "data/local/modeling/shared_cache/tp58"))
CACHE = WORKDIR / "cache"
KEY = ["race_id", "horse_id"]
THREADS = int(os.environ.get("TP58_THREADS", 4))


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cmd_frame(a) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    df, base, meta, folds, lp = t.setup()
    X = v5.build_frame(v5.CONFIGS["d_CY_BEST"], df)
    _log(f"V5 表: {X.shape}")
    cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    Xn = X[num_cols].astype("float32")
    out = pd.concat([meta[KEY + ["finish_position", "year", "date"]].reset_index(drop=True), Xn.reset_index(drop=True)], axis=1)
    out.to_parquet(CACHE / "v5_frame_num.parquet", index=False)
    if cat_cols:
        C = X[cat_cols].astype(str).reset_index(drop=True)
        pd.concat([meta[KEY].reset_index(drop=True), C], axis=1).to_parquet(CACHE / "v5_frame_cat.parquet", index=False)
    # 交差用の生の列（ID・条件）。X に無いものを df から
    raw_names = [c for c in df.columns if c in KEY or any(k in c for k in (
        "jockey", "trainer", "venue", "surface", "distance", "track_cond", "going", "class", "grade", "sire", "owner", "breeder", "sex", "age",
        "weight", "bracket", "horse_number", "field_size", "round", "weather", "course", "direction", "prev_"))]
    raw = df[raw_names].copy()
    raw.to_parquet(CACHE / "raw_cols.parquet", index=False)
    (CACHE / "columns.json").write_text(json.dumps({"num": num_cols, "cat": cat_cols, "raw": raw_names}, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"保存: num={len(num_cols)} cat={len(cat_cols)} raw={len(raw_names)} -> {CACHE}")


def cmd_importance(a) -> None:
    import lightgbm as lgb

    F = pd.read_parquet(CACHE / "v5_frame_num.parquet")
    cols = json.loads((CACHE / "columns.json").read_text(encoding="utf-8"))["num"]
    tr = F[F["year"] <= 2021]
    y = (pd.to_numeric(tr["finish_position"], errors="coerce") == 1).astype(int).to_numpy()
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.4, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=5.0, num_threads=THREADS, verbose=-1, seed=0)
    ds = lgb.Dataset(tr[cols].to_numpy(dtype="float32"), y, feature_name=cols, free_raw_data=True)
    _log(f"importance 学習: rows={len(tr)} cols={len(cols)}")
    m = lgb.train(params, ds, num_boost_round=400)
    gain = m.feature_importance("gain")
    imp = sorted(zip(cols, gain.tolist()), key=lambda x: -x[1])
    (CACHE / "importance_gain.json").write_text(json.dumps(imp, ensure_ascii=False, indent=0), encoding="utf-8")
    _log("上位 15: " + ", ".join(f"{c}:{g:.0f}" for c, g in imp[:15]))
    pairs = tree_path_pairs(m.trees_to_dataframe())
    (CACHE / "tree_pairs.json").write_text(json.dumps(pairs[:2000], ensure_ascii=False), encoding="utf-8")
    _log("同じ経路で続けて分割される組 上位 10: " + ", ".join(f"{a}×{b}:{g:.0f}" for a, b, g, _ in pairs[:10]))


def tree_path_pairs(td: pd.DataFrame) -> list:
    """木の経路上で祖先の分割列 × 子の分割列を、子の分割 gain で重み付けして集計（[a, b, gain 合計, 回数] を gain 降順）。"""
    agg: dict = {}
    for _, tr in td.groupby("tree_index"):
        info = {r.node_index: (r.split_feature, r.split_gain, r.parent_index) for r in tr.itertuples() if isinstance(r.split_feature, str)}
        for nid, (feat, gain, parent) in info.items():
            anc = parent
            while isinstance(anc, str) and anc in info:
                af, _, anc_next = info[anc]
                if af != feat:
                    k = tuple(sorted((af, feat)))
                    s = agg.setdefault(k, [0.0, 0])
                    s[0] += float(gain)
                    s[1] += 1
                anc = anc_next
    return sorted(([a, b, g, n] for (a, b), (g, n) in agg.items()), key=lambda x: -x[2])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["frame", "importance"])
    a = ap.parse_args()
    {"frame": cmd_frame, "importance": cmd_importance}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
