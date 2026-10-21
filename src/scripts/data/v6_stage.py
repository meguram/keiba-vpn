"""v6: 前段 S3（s_time_pred / s_time_gap / s_rank_pred）を、新しい入力で作り直す（年ごとの expanding OOF。tp07_base.stage と同じ設定）。

python -m src.scripts.data.v6_stage build a    # a: 入力 = base 175 列 + tl_（再取得版）+ CY（追い切りの新しい部分だけを入れ替える）
python -m src.scripts.data.v6_stage build b    # b: a + 追走難度 v2（EASE2）+ ラップ予想（LAP）+ 位置取り予想（pf_early など）
python -m src.scripts.data.v6_stage build d    # d: b の入力で、ペース補正後の走破力（時計・上がりからレース全体のペースの効果を引いた目的変数）を予測（s_adj_time / s_adj_l3f、追加列のみ）
python -m src.scripts.data.v6_stage build c    # c: b + 『レース全体のペース（結果由来）』を入力にして学習し、予測時はスロー/ミドル/ハイに差し替えて
                                              #    ラップ予想のシナリオ確率で混合（シナリオ条件付き。s_time_exp 等の追加列も出す）
出力: $V6_OUT（既定 data/local/modeling/training_pipeline_v4/v6）/s3x_{a,b,c}.parquet

ペースの結果由来の列（r_pace_bal）は学習の入力にだけ使い、予測では必ず差し替える（その年の結果は予測に入らない）。
EASE2・LAP は年ごとの leave-one-year-out の予測、S3 本体は expanding なので、評価年のラベルが評価年の予測に入ることはない。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import tp07_base as tb

V4 = tb.V4
OUT = Path(os.environ.get("V6_OUT") or V4 / "v6")
KEY = ["race_id", "horse_id"]
EASE2 = Path(os.environ.get("V6_EASE2") or "/home/jovyan/work/tp54_out/groups/EASE2.parquet")
HORSE_EASE = Path(os.environ.get("V6_HORSE_EASE") or "/home/jovyan/work/tp54_out/horse_ease.parquet")
LAP = Path(os.environ.get("V6_LAP") or "/home/jovyan/work/tp55_out3/groups/LAP.parquet")
PF_COLS = ["pf_early", "pf_rank", "z_hat", "td_ease", "td_l3f_ability", "td_stamina"]
THREADS = int(os.environ.get("V6_THREADS", "4"))


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _merge(df: pd.DataFrame, T: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    T = T.drop_duplicates(KEY).copy()
    T["race_id"], T["horse_id"] = T["race_id"].astype(str), T["horse_id"].astype(str)
    if cols is not None:
        T = T[KEY + cols]
    m = df[KEY].astype(str).merge(T, on=KEY, how="left")
    assert len(m) == len(df)
    m = m[[c for c in m.columns if c not in KEY]]
    m.index = df.index
    return m


def new_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """a: tl_（再取得版）+ CY。"""
    from src.scripts.data import tp40_eval as e40

    os.environ["TP40_BASE"] = "fresh"
    F = e40._frame(df)
    tl = F[[c for c in F.columns if c.startswith("tl_")]]
    return pd.concat([tl, e40._load_group("CY", df)], axis=1)


def forecast_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """b の追加分: 追走難度 v2 / ラップ予想 / 位置取り予想。"""
    e = _merge(df, pd.read_parquet(EASE2))
    lap = _merge(df, pd.read_parquet(LAP))
    he = _merge(df, pd.read_parquet(HORSE_EASE, columns=KEY + PF_COLS))
    return pd.concat([e, lap, he], axis=1)


def scenario_values(T: pd.DataFrame) -> dict[str, float]:
    """学習データの r_pace_bal の区分（pace_bal_cls: 0=後傾=スロー, 1=ミドル, 2=前傾=ハイ）ごとの平均。"""
    s = T["r_pace_bal"].astype(float)
    c = T["r_pace_bal_cls"].astype(float)
    return {"slow": float(s[c == 0].mean()), "mid": float(s[c == 1].mean()), "high": float(s[c == 2].mean())}


def make_inputs(variant: str, df: pd.DataFrame, T: pd.DataFrame, base: list[str]) -> pd.DataFrame:
    """前段の入力行列。a: base + tl_ 再取得 + CY。b/c/d: a + 追走難度 v2・ラップ予想・位置取り予想。c は学習用にペースの結果由来の列を足す。"""
    X = pd.concat([df[base].copy(), new_inputs(df)], axis=1)
    if variant in ("b", "c", "d"):
        X = pd.concat([X, forecast_inputs(df)], axis=1)
    if variant == "c":
        X["v6_pace_score"] = T["r_pace_bal"].to_numpy(dtype=float)
    for c in X.columns:
        if str(X[c].dtype) in ("object", "string"):
            X[c] = X[c].astype("category")
    return X


def build(variant: str) -> None:
    from src.pipeline.models import stage_models as sm

    OUT.mkdir(parents=True, exist_ok=True)
    base = tb._base_cols()
    df = tb._load(V4)
    labels = pd.read_parquet(tb.OLD / "oof/stage_labels.parquet")
    fit_params = json.loads((tb.OLD / "features/fit_params.json").read_text(encoding="utf-8"))
    race_cls = sm.pace_class_labels(labels, fit_params["pace_q"], fit_params["rpci_q"])
    T = sm.build_targets(df, labels, race_cls)
    X = make_inputs(variant, df, T, base)
    sv = None
    if variant == "c":
        sv = scenario_values(T)
        _log(f"シナリオ値（r_pace_bal の区分平均）: {sv}")
    years = sorted(int(y) for y in df["year"].unique() if int(y) <= 2025)
    out = pd.DataFrame({"race_id": df["race_id"].to_numpy(), "horse_id": df["horse_id"].to_numpy()})
    t0 = time.time()
    n = len(df)
    if variant == "c":
        P = df[["race_id"]].merge(pd.read_parquet(LAP, columns=["race_id", "lp_p_slow", "lp_p_mid", "lp_p_high"]).drop_duplicates("race_id").astype({"race_id": str}),
                                  on="race_id", how="left")[["lp_p_slow", "lp_p_mid", "lp_p_high"]].to_numpy(dtype=float)
        P = np.where(np.isfinite(P).all(axis=1, keepdims=True), P, np.array([[0.25, 0.5, 0.25]]))
        P = P / P.sum(axis=1, keepdims=True)
        variants = {k: {"v6_pace_score": np.full(n, v)} for k, v in sv.items()}
    else:
        variants = None
    res = {}
    if variant == "d":
        z = T["r_pace_bal"].to_numpy(dtype=float)
        outs = {}
        for col, tname in (("s_adj_time", "t_time_dev"), ("s_adj_l3f", "t_l3f_dev")):
            y = T[tname].to_numpy(dtype=float)
            ok = np.isfinite(y) & np.isfinite(z)
            slope = float(np.polyfit(z[ok], y[ok], 1)[0])                       # ペース 1 単位あたりの効果（全期間の 1 係数。ラベル側の変換だけ）
            y_adj = y - slope * np.where(np.isfinite(z), z, 0.0)
            pred = sm.oof_regression(df, X, y_adj, years, "expanding", THREADS)["base"]
            out[col] = pred.astype("float32")
            res[col] = {"slope": slope, "nonnull": float(np.isfinite(pred).mean())}
            _log(f"[{time.time() - t0:.0f}s] {col}: slope {slope:+.4f} 非欠損 {res[col]['nonnull']:.3f}")
        out.to_parquet(OUT / "s3x_d.parquet", index=False)
        (OUT / "s3x_d.json").write_text(json.dumps({"cols": int(X.shape[1]), "variant": "d", "nonnull": res}, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    for col, tname in (("s_time_pred", "t_time_dev"), ("s_rank_pred", "t_rank_norm")):
        pred = sm.oof_regression(df, X, T[tname].to_numpy(dtype=float), years, "expanding", THREADS, variants=variants)
        if variant == "c":
            A = np.stack([pred["slow"], pred["mid"], pred["high"]], axis=1)
            mix = (P * A).sum(axis=1)
            sd = np.sqrt((P * (A - mix[:, None]) ** 2).sum(axis=1))
            out[col] = mix.astype("float32")
            short = "time" if "time" in col else "rank"
            out[f"s_{short}_sd"] = sd.astype("float32")
            out[f"s_{short}_sens"] = (A[:, 2] - A[:, 0]).astype("float32")
            res[col] = {"nonnull": float(np.isfinite(mix).mean())}
        else:
            out[col] = pred["base"].astype("float32")
            res[col] = {"nonnull": float(np.isfinite(pred["base"]).mean())}
        _log(f"[{time.time() - t0:.0f}s] {col}: 非欠損 {res[col]['nonnull']:.3f}")
    out["s_time_gap"] = (out["s_time_pred"] - out.groupby("race_id")["s_time_pred"].transform("min")).astype("float32")
    out.to_parquet(OUT / f"s3x_{variant}.parquet", index=False)
    old = pd.read_parquet(V4 / "oof/stage_features_expanding.parquet")
    old["race_id"], old["horse_id"] = old["race_id"].astype(str), old["horse_id"].astype(str)
    m = out.merge(old, on=KEY, suffixes=("", "_old"))
    rep = {}
    for c in ("s_time_pred", "s_time_gap", "s_rank_pred"):
        ok = m[c].notna() & m[c + "_old"].notna()
        rep[c] = {"rows_both": int(ok.sum()), "corr_vs_v4": float(np.corrcoef(m.loc[ok, c], m.loc[ok, c + "_old"])[0, 1])}
    (OUT / f"s3x_{variant}.json").write_text(json.dumps({"cols": int(X.shape[1]), "variant": variant, "scenario_values": sv, "report": rep, "nonnull": res},
                                                         ensure_ascii=False, indent=1), encoding="utf-8")
    _log(json.dumps(rep, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("variant", choices=["a", "b", "c", "d"])
    a = ap.parse_args()
    build(a.variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
