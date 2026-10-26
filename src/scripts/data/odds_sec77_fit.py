"""第77節の想定単勝・想定複勝モデルを、本番の推論で使える形に保存する。

    python -m src.scripts.data.odds_sec77_fit check   # サービングの特徴量列（v6 バンドル）と学習列の重なりを確認
    python -m src.scripts.data.odds_sec77_fit fit     # 2020〜2025 年で最終モデルを学習し、列・較正係数と一緒に保存

保存先: data/local/modeling/odds_pred/serving/（ODDS_SEC77_DIR で変更）
  share_model.txt / place_model.txt / meta.json（列・較正の直線・払戻率・学習行数・木の本数）
2026 年は学習に使わない。市場オッズ・人気は特徴量にしない（odds_pred_eval と同じ除外）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.scripts.data import odds_pred_eval as E  # noqa: E402

OUT = Path(os.environ.get("ODDS_SEC77_DIR", E.OUT / "serving"))
KEY = E.KEY


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bundle_columns() -> list[str]:
    from src.pipeline.inference import v3_serving as vs
    from src.pipeline.models.ensemble_bundle import EnsembleBundle

    b = EnsembleBundle.load(vs.v6_paths()["bundle"])
    return list(dict.fromkeys(c for m in b.members for c in m.feature_cols))


def cmd_check(a) -> None:
    df, X = E._frame_with_targets()
    bc = set(bundle_columns())
    cols = list(X.columns)
    inb = [c for c in cols if c in bc]
    _log(f"学習列 {len(cols)}、サービング列（v6 バンドル）{len(bc)}、学習列のうちサービングで作れる列 {len(inb)}")
    miss = [c for c in cols if c not in bc]
    _log(f"サービングで作れない学習列 {len(miss)}: {miss[:20]}")


def cmd_fit(a) -> None:
    import lightgbm as lgb

    OUT.mkdir(parents=True, exist_ok=True)
    threads = int(os.environ.get("ODDS_THREADS", "4"))
    df, X = E._frame_with_targets()
    bc = set(bundle_columns())
    cols = [c for c in X.columns if c in bc]
    dropped = [c for c in X.columns if c not in bc]
    _log(f"列 {len(cols)}（サービングで作れない {len(dropped)} 列は除く: {dropped[:10]}）")
    keys = df[KEY].astype(str).reset_index(drop=True)
    year = pd.to_datetime(df["date"]).dt.year.to_numpy()
    wp = E.load_wp()
    T = keys.assign(_i=np.arange(len(keys))).merge(wp[KEY + ["win_odds", "place_mid", "p_mkt"]], on=KEY, how="inner")
    idx = T["_i"].to_numpy()
    Xs = X[cols].iloc[idx].reset_index(drop=True)
    yr = year[idx]
    ok = (yr >= 2020) & (yr <= 2025)
    y_share = np.log(T["p_mkt"].to_numpy())
    y_place = np.log(T["place_mid"].to_numpy())
    info = json.loads((E.OUT / "wp_fit_info.json").read_text(encoding="utf-8"))
    models = {}
    for name, y in (("share", y_share), ("place", y_place)):
        it = max(int(info[f"2025_{name}"]["iters"] * 1.1), 100)
        m = lgb.LGBMRegressor(n_estimators=it, learning_rate=0.06, num_leaves=63, min_child_samples=60, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.4, reg_lambda=5.0, n_jobs=threads, verbose=-1)
        m.fit(Xs[ok], y[ok])
        m.booster_.save_model(str(OUT / f"{name}_model.txt"))
        models[name] = {"iters": it, "n_train": int(ok.sum())}
        _log(f"{name}: 木 {it} 本、学習 {int(ok.sum())} 行（2020〜2025）")
    # 較正の直線: OOF（2022〜2025）の生出力 → 市場の対数シェア / 対数複勝（odds_pred_eval.build_predictions と同じ当てはめ。全 OOF 年を使う）
    raw = pd.read_parquet(E.OUT / "wp_raw_oof.parquet")
    d = raw[raw["year"] >= 2022].merge(wp[KEY + ["p_mkt", "place_mid"]], on=KEY, how="inner")
    a1, b1 = E._fit_line(d["share_raw"].to_numpy(), np.log(d["p_mkt"]).to_numpy())
    a2, b2 = E._fit_line(d["place_raw"].to_numpy(), np.log(d["place_mid"]).to_numpy())
    meta = {"cols": cols, "dropped_cols": dropped, "win_line": [a1, b1], "place_line": [a2, b2], "payback_win": E.PAYBACK_WIN,
            "place_floor": 1.05, "models": models, "trained_years": [2020, 2025], "fitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": "第77節の win_B / place_B と同じ変換。2026 年は学習に使っていない。"}
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"保存: {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("check")
    sp.add_parser("fit")
    a = ap.parse_args()
    {"check": cmd_check, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    main()
