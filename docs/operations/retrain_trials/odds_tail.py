"""想定単勝（L2 の本）の学習で、高オッズの馬が低オッズ側の当てはめを引っ張っていないかの比較。ホストで実行（重い処理は /tmp/heavy_run.sh 経由）。

    REBUILD_2016=1 REBUILD_MEGU_ALL=1 ODDS_TRAIN_FROM=2016 PYTHONPATH=. python docs/operations/retrain_trials/odds_tail.py

条件（学習の目標値の作り方だけが違う。予測は丸めない）:
  base   log 占有率そのまま（試行 6）
  clip50 市場の単勝が 50 倍を超える馬は 50 倍の占有率に下限（学習の目標値だけ。予測は丸めない）
  clip100 同、100 倍
  huber  Huber 損失（alpha=0.5）
分け方は試行 6 と同じ（各検証年の前年を早期終了用、それ以前を学習）。検証年は 2024・2025。
予測は検証用の年の前年で較正の直線を当ててから、レース内で正規化 → 0.797 / 占有率（運用と同じ）。

合格基準（結果を見る前に固定。見てから動かさない）— base に対して、2024・2025 の両年で
  P1: 実際の単勝が 20 倍未満の馬の log 誤差（MAE）が 0.003 以上下がる
  P2: 実際の単勝が 20〜50 倍の馬の log 誤差が 0.010 を超えて悪化しない
  P3: レース内の順位相関（想定単勝 と 実際の単勝）が 0.002 を超えて下がらない
全部満たした条件だけを採用候補にする。満たすものがなければ base のまま。50 倍以上の馬の誤差は参考（採否に使わない）。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jovyan/work/keiba-vpn")
OUT = Path(os.environ.get("ODDS_TAIL_OUT", "/home/jovyan/work/retrain_out/odds_tail"))
OUT.mkdir(parents=True, exist_ok=True)
os.environ["ODDS_PRED_OUT"] = str(OUT)
from src.pipeline.inference.odds_t6 import normalize_log_share  # noqa: E402
from src.scripts.data import odds_pred_eval as oe  # noqa: E402

THREADS = int(os.environ.get("ODDS_THREADS", "6"))
TRAIN_FROM = int(os.environ.get("ODDS_TRAIN_FROM", "2016"))
TEST_YEARS = (2024, 2025)
PARAMS = dict(learning_rate=0.06, num_leaves=63, min_data_in_leaf=60, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.4,
              lambda_l2=5.0, verbose=-1, num_threads=THREADS, seed=0)
BANDS = [("<3", 0, 3), ("3-10", 3, 10), ("10-20", 10, 20), ("20-50", 20, 50), (">=50", 50, 1e9)]


def log(m: str) -> None:
    print(f"[{time.strftime('%T')}] {m}", flush=True)


def target(name: str, p: np.ndarray, odds: np.ndarray) -> tuple[np.ndarray, dict]:
    if name.startswith("clip"):
        f = float(name[4:])
        return np.log(p * odds / np.minimum(odds, f)), {"objective": "l2"}
    if name == "huber":
        return np.log(p), {"objective": "huber", "alpha": 0.5}
    return np.log(p), {"objective": "l2"}


def to_odds(ls: np.ndarray, rid: np.ndarray, line: tuple[float, float]) -> np.ndarray:
    share = np.exp(normalize_log_share(line[0] + line[1] * ls, rid))
    return np.clip(oe.PAYBACK_WIN / np.clip(share, 1e-6, None), 1.0, None)


def main() -> None:
    if os.environ.get("REBUILD_2016") == "1":
        from src.scripts.data import rebuild_groups_2016 as rg

        rg.redirect()
        oe.BET = rg.BASE / "betting"
    df, X = oe._frame_with_targets()
    keys = df[oe.KEY].astype(str).reset_index(drop=True)
    year = pd.to_datetime(df["date"]).dt.year.to_numpy()
    wp = oe.load_wp()
    T = keys.assign(_i=np.arange(len(keys))).merge(wp[oe.KEY + ["win_odds", "p_mkt"]], on=oe.KEY, how="inner")
    T = T.sort_values(["race_id", "horse_id"]).reset_index(drop=True)
    idx = T["_i"].to_numpy()
    Xs = X.iloc[idx].to_numpy(np.float32)
    yr = year[idx]
    rid = T["race_id"].to_numpy()
    p = T["p_mkt"].to_numpy()
    odds = T["win_odds"].to_numpy()
    log(f"行 {len(T):,}、列 {Xs.shape[1]}")
    rows = []
    for name in ("base", "clip50", "clip100", "huber"):
        y, extra = target(name, p, odds)
        for Y in TEST_YEARS:
            te = yr == Y
            va = yr == Y - 1
            tr = (yr < Y - 1) & (yr >= TRAIN_FROM)
            dtr = lgb.Dataset(Xs[tr], y[tr], free_raw_data=False)
            m = lgb.train(dict(PARAMS, **extra), dtr, 3000, valid_sets=[lgb.Dataset(Xs[va], y[va], reference=dtr)],
                          callbacks=[lgb.early_stopping(60, verbose=False)])
            it = int(m.best_iteration or m.current_iteration())
            ls_va = normalize_log_share(m.predict(Xs[va]), rid[va])
            line = oe._fit_line(ls_va, np.log(p[va]))
            pr = to_odds(normalize_log_share(m.predict(Xs[te]), rid[te]), rid[te], line)
            act = odds[te]
            err = np.abs(np.log(pr) - np.log(act))
            r = {"cond": name, "year": Y, "iters": it, "line": [float(line[0]), float(line[1])], "mae_all": float(err.mean())}
            for lab, lo, hi in BANDS:
                mk = (act >= lo) & (act < hi)
                r[f"mae_{lab}"] = float(err[mk].mean())
            r["mae_<20"] = float(err[act < 20].mean())
            sp = pd.DataFrame({"r": rid[te], "a": pr, "b": act})
            r["spearman_race"] = float(sp.groupby("r").apply(lambda g: g["a"].corr(g["b"], method="spearman")).mean())
            rows.append(r)
            log(json.dumps(r, ensure_ascii=False))
            pd.DataFrame(rows).to_csv(OUT / "odds_tail.csv", index=False)
    d = pd.DataFrame(rows)
    verdict = {}
    for name in ("clip50", "clip100", "huber"):
        ok = True
        detail = {}
        for Y in TEST_YEARS:
            b = d[(d.cond == "base") & (d.year == Y)].iloc[0]
            c = d[(d.cond == name) & (d.year == Y)].iloc[0]
            lowb, lowc = b["mae_<20"], c["mae_<20"]
            detail[Y] = {"P1_low_mae_diff(base-cond)": float(lowb - lowc), "P2_mid_mae_worse": float(c["mae_20-50"] - b["mae_20-50"]),
                         "P3_spearman_drop": float(b["spearman_race"] - c["spearman_race"])}
            ok &= (lowb - lowc >= 0.003) and (c["mae_20-50"] - b["mae_20-50"] <= 0.010) and (b["spearman_race"] - c["spearman_race"] <= 0.002)
        verdict[name] = {"pass": bool(ok), "detail": detail}
    (OUT / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=1), encoding="utf-8")
    log("判定 " + json.dumps({k: v["pass"] for k, v in verdict.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
