"""想定オッズ（試行 6）の推論用の最終モデル（2016〜2025 の全行で 1 回学習）。ホストで実行（重い処理は /tmp/heavy_run.sh 経由）。

    REBUILD_2016=1 REBUILD_MEGU_ALL=1 ODDS_TRAIN_FROM=2016 python -m src.scripts.data.fit_final_odds_t6 \\
        --out retrain_out/base_2016/final_odds_t6 --bundle retrain_out/base_2016/final_v6e16/bundle_v6_2025 \\
        --oof retrain_out/odds_try16a_6 --info-l2 retrain_out/odds_try16a_2 --info-softmax retrain_out/odds_try16a_3

作り方は ``docs/operations/retrain_trials/odds_try.py``・``odds_place2.py``・``odds_blend.py``（試行 6）と同じ:
- share_l2 / share_softmax / place_direct: 特徴量は v6 の枠（市場のオッズ・人気は除く）のうち、サービングの束で作れる列だけ。木の本数は
  2025 年の早期終了の本数 × 1.1（OOF の学習し直しと同じ）。
- place_struct: 実際の市場の占有率から作った量 → log 複勝（中央）。800 本。予測では予測した占有率に当てて、直接予測と半々（place_struct_weight）。
- 較正の直線: 試行 6 の OOF（2022〜2025）の share_raw / place_raw → 市場の log 占有率 / log 複勝（odds_sec77_fit と同じ当てはめ）。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.pipeline.inference.odds_t6 import normalize_log_share, place_struct_features
from src.scripts.data import odds_pred_eval as oe

KEY = oe.KEY
PARAMS = dict(learning_rate=0.06, num_leaves=63, min_data_in_leaf=60, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.4,
              lambda_l2=5.0, verbose=-1)
STRUCT_PARAMS = dict(objective="l2", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, verbose=-1)
STRUCT_ROUNDS = 800
PLACE_STRUCT_WEIGHT = 0.5      # 試行 6 の年別の重みは 0.45〜0.50（odds_try16a_5/place2_weights.json）


def log(m: str) -> None:
    print(f"[{time.strftime('%T')}] {m}", flush=True)


def softmax_obj(gidx: np.ndarray, target: np.ndarray):
    def f(preds, _ds):
        m = pd.Series(preds).groupby(gidx).transform("max").to_numpy()
        e = np.exp(preds - m)
        p = e / np.bincount(gidx, e)[gidx]
        return p - target, np.maximum(p * (1 - p), 1e-6)
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bundle", required=True, help="v6 の束（サービングで作れる列の集合）")
    ap.add_argument("--oof", required=True, help="試行 6 の OOF（較正の直線用）")
    ap.add_argument("--info-l2", required=True, help="L2 の wp_fit_info.json があるディレクトリ")
    ap.add_argument("--info-softmax", required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--check", action="store_true", help="列の重なりだけ確かめて終わる")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from src.pipeline.models.ensemble_bundle import EnsembleBundle

    if __import__("os").environ.get("REBUILD_2016") == "1":
        from src.scripts.data import rebuild_groups_2016 as rg

        rg.redirect()
        oe.BET = rg.BASE / "betting"
    b = EnsembleBundle.load(Path(a.bundle))
    bc = set(c for m in b.members for c in m.feature_cols)
    df, X = oe._frame_with_targets()
    cols = [c for c in X.columns if c in bc]
    dropped = [c for c in X.columns if c not in bc]
    log(f"列 {len(cols)}（サービングで作れない {len(dropped)} 列は除く: {dropped[:10]}）")
    if a.check:
        return
    keys = df[KEY].astype(str).reset_index(drop=True)
    year = pd.to_datetime(df["date"]).dt.year.to_numpy()
    wp = oe.load_wp()
    T = keys.assign(_i=np.arange(len(keys))).merge(wp[KEY + ["win_odds", "place_mid", "p_mkt"]], on=KEY, how="inner")
    T = T.sort_values(["race_id", "horse_id"]).reset_index(drop=True)
    idx = T["_i"].to_numpy()
    yr = year[idx]
    ok = (yr >= 2016) & (yr <= 2025)
    assert yr[ok].min() <= 2016 and yr[ok].max() >= 2025, "2016〜2025 の土台の表ではない（REBUILD_2016=1 が必要）"
    Xs = X[cols].iloc[idx].to_numpy(np.float32)[ok]
    T = T[ok].reset_index(drop=True)
    rid = T["race_id"].to_numpy()
    gidx = pd.factorize(rid)[0]
    y_share = np.log(T["p_mkt"].to_numpy())
    y_place = np.log(T["place_mid"].to_numpy())
    it_l2 = json.loads((Path(a.info_l2) / "wp_fit_info.json").read_text(encoding="utf-8"))
    it_sm = json.loads((Path(a.info_softmax) / "wp_fit_info.json").read_text(encoding="utf-8"))
    n_share_l2 = max(int(it_l2["2025_share"]["iters"] * 1.1), 100)
    n_place = max(int(it_l2["2025_place"]["iters"] * 1.1), 100)
    n_share_sm = max(int(it_sm["2025_share"]["iters"] * 1.1), 100)
    log(f"学習 {len(T):,} 行（{yr[ok].min()}〜{yr[ok].max()}）、木 share_l2 {n_share_l2}・softmax {n_share_sm}・place {n_place}")

    P = dict(PARAMS, num_threads=a.threads, seed=0)
    m = lgb.train(dict(P, objective="l2"), lgb.Dataset(Xs, y_share), n_share_l2)
    m.save_model(str(out / "share_l2.txt")); log("share_l2 done")
    t = np.exp(y_share)
    m = lgb.train(dict(P, objective=softmax_obj(gidx, t), metric="None"), lgb.Dataset(Xs, t), n_share_sm)
    m.save_model(str(out / "share_softmax.txt")); log("share_softmax done")
    m = lgb.train(dict(P, objective="l2"), lgb.Dataset(Xs, y_place), n_place)
    m.save_model(str(out / "place_direct.txt")); log("place_direct done")

    # 複勝の 2 段目: 実際の市場の占有率の並び → log 複勝（odds_place2 と同じ。全年・n≥4 のレース）
    allwp = wp[(wp["year"] >= 2016) & (wp["year"] <= 2025)].sort_values(["race_id", "horse_id"]).reset_index(drop=True)
    FA = place_struct_features(np.log(allwp["p_mkt"].to_numpy()), allwp["race_id"].to_numpy())
    ms = lgb.train(dict(STRUCT_PARAMS, num_threads=a.threads), lgb.Dataset(FA, np.log(allwp["place_mid"].to_numpy())), STRUCT_ROUNDS)
    ms.save_model(str(out / "place_struct.txt")); log("place_struct done")

    # 較正の直線（OOF 2022〜2025）
    raw = pd.read_parquet(Path(a.oof) / "wp_raw_oof.parquet")
    d = raw[raw["year"] >= 2022].merge(wp[KEY + ["p_mkt", "place_mid"]], on=KEY, how="inner")
    a1, b1 = oe._fit_line(d["share_raw"].to_numpy(), np.log(d["p_mkt"]).to_numpy())
    a2, b2 = oe._fit_line(d["place_raw"].to_numpy(), np.log(d["place_mid"]).to_numpy())
    wy = wp[(wp["year"] >= 2016) & (wp["year"] <= 2025)]
    half = float(np.nanmedian((wy["place_odds_max"] - wy["place_odds_min"]) / (wy["place_odds_max"] + wy["place_odds_min"])))
    meta = {"cols": cols, "dropped_cols": dropped, "win_line": [a1, b1], "place_line": [a2, b2], "payback_win": oe.PAYBACK_WIN,
            "place_floor": 1.05, "place_half_range": half, "place_struct_weight": PLACE_STRUCT_WEIGHT,
            "trees": {"share_l2": n_share_l2, "share_softmax": n_share_sm, "place_direct": n_place, "place_struct": STRUCT_ROUNDS},
            "n_train": int(len(T)), "trained_years": [2016, 2025], "fitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": "試行 6（odds_try / odds_place2 / odds_blend）と同じ作り方。2026 年は学習に使っていない。"
            "較正の直線は全列モデルの OOF で当てたもの（最終モデルは束で作れる列のみ）。place_struct_weight は年別 0.45〜0.50 の代表値。"}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    # 学習内の確認（過学習した値なので参考）: 想定の占有率と市場の占有率の相関
    A = pd.DataFrame(Xs[:20000], columns=cols)
    r20 = rid[:20000]
    ls = (normalize_log_share(lgb.Booster(model_file=str(out / "share_l2.txt")).predict(A), r20)
          + normalize_log_share(lgb.Booster(model_file=str(out / "share_softmax.txt")).predict(A, raw_score=True), r20)) / 2
    (out / "sanity.json").write_text(json.dumps({"corr_in_sample_logshare": float(np.corrcoef(ls, y_share[:20000])[0, 1]), "win_line": [a1, b1],
                                                 "place_line": [a2, b2]}, indent=1), encoding="utf-8")
    log(f"保存 {out}")


if __name__ == "__main__":
    main()
