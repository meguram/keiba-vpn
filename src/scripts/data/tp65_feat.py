"""tp-65 (4): めぐ指数・netkeiba タイム指数のロバストな履歴特徴量を V5（1,142 列）に累積して評価する。

python -m src.scripts.data.tp65_feat build                 # 全出走行（2020〜2026）の as-of 特徴量 → <TP65_OUT>/rob.parquet
python -m src.scripts.data.tp65_feat audit                 # 先読み検査（切断日以降の値を消して作り直しても、切断日より前の行は完全一致）
python -m src.scripts.data.tp65_feat run --name NAME [--only PREFIX,...] [--seeds 0,1]   # V5 と同じ設定の LightGBM。V5 の OOF と対で比較

特徴量: 各走の『直近 5 走（その走より前で、指数がある走）』から、ewm / max / median / trimmed / huber を作る。
  rb_mg_*  めぐ指数 v3a（相対）をクリップ（学習期間 2020〜2021 の 0.5〜99.5 パーセンタイル）した値の履歴
  rb_nk_*  netkeiba タイム指数を同様にクリップした値の履歴（V5 の speed_* は平均・最大・直近の素の値）
予測時点で見える値だけ: めぐ指数は前走までに確定した値、netkeiba は次の出走前ページに載った最初の値。
採用基準: feature_bakeoff.compare_oof（勝率の 95% 区間の上限 < 0、効果 ≥ 0.0002、連対・複勝の悪化 ≤ 0.0002）。test 2025・2026 は使わない。市場オッズ・人気は使わない。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TP40_THREADS", "4")

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("TP65_OUT", "/home/jovyan/work/tp65_out"))
BASE_OOF = ROOT / "data/local/modeling/training_pipeline_v4/v5_select/oof_d_CY_BEST.parquet"
KEY = ["race_id", "horse_id"]
SERIES = {"rb_mg": "megu_index_v3a", "rb_nk": "nk"}
CLIP_YEARS = (2020, 2021)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _variants(M: np.ndarray, scale: float) -> dict[str, np.ndarray]:
    from src.scripts.data.tp65_var import feature_variants

    v = feature_variants(M, scale)
    return {k: v[k] for k in ("ewm5", "max5", "median5", "trimmed5", "huber5")}


FLOOR = 40.0  # 大差負けの下限: 同じレースの勝ち馬の指数 − 40 点（約 4 秒）より下は引き上げる


def asof_features(runs: pd.DataFrame, cut: str | None = None) -> pd.DataFrame:
    """runs: horse_id, race_id, date と各指数列。全出走行（指数が無い行も）に、その日より前の直近 5 走（指数がある走）から作った特徴量を付ける。"""
    r = runs[["race_id", "horse_id", "date", "megu_index_v3a", "nk", "finish_position"]].copy()
    r["fp"] = pd.to_numeric(r["finish_position"], errors="coerce")
    r["date"] = pd.to_datetime(r["date"])
    r["year"] = r["date"].dt.year
    if cut:
        r = r[r["date"] < pd.Timestamp(cut)].copy()
    out = r[KEY + ["date"]].copy()
    out = out.sort_values(["date", "race_id", "horse_id"]).reset_index(drop=True)
    for pref, col in SERIES.items():
        v = r[r[col].notna()].sort_values(["horse_id", "date", "race_id"]).copy()
        lo, hi = v.loc[v["year"].isin(CLIP_YEARS), col].quantile([0.005, 0.995])
        ref = v.loc[v["year"].isin(CLIP_YEARS), col]  # 尺度も 2020〜2021 年だけから決める（将来の値で変わらない）
        scale = float(1.4826 * np.median(np.abs(ref - ref.median())))
        v["x"] = v[col].clip(lo, hi)
        wx = v[v["fp"] == 1].groupby("race_id")["x"].max().rename("wx")
        v = v.join(wx, on="race_id")
        v["xf"] = np.where(v["wx"].notna(), np.maximum(v["x"], v["wx"] - FLOOR), v["x"])
        g = v["horse_id"]
        # 各有効走を『最後の走』とした直近 5 走（自分を含む）
        M = np.column_stack([v["x"].groupby(g).shift(i).to_numpy() for i in range(0, 5)])
        feats = _variants(M, scale)
        Mf = np.column_stack([v["xf"].groupby(g).shift(i).to_numpy() for i in range(0, 5)])
        from src.scripts.data.tp65_var import ewm_decay, weighted_huber

        feats["fl_ewm"] = ewm_decay(Mf, 0.65)
        feats["fl_whuber"] = weighted_huber(Mf, 0.65, scale)
        F = pd.DataFrame({f"{pref}_{k}": a for k, a in feats.items()})
        F["horse_id"], F["vdate"] = v["horse_id"].to_numpy(), v["date"].to_numpy()
        F[f"{pref}_n"] = np.sum(~np.isnan(M), axis=1)
        F[f"{pref}_last_gap"] = np.nan
        F = F.sort_values("vdate").reset_index(drop=True)
        tgt = out[["horse_id", "date"]].copy().sort_values("date")
        m = pd.merge_asof(tgt, F, left_on="date", right_on="vdate", by="horse_id", allow_exact_matches=False)
        m = m.drop(columns=["vdate", f"{pref}_last_gap"])
        # merge_asof は左（tgt）の並びを保つので、tgt の元の行ラベルから race_id を付けてキーで結合する
        m["race_id"] = out.loc[tgt.index, "race_id"].to_numpy()
        out = out.merge(m.drop(columns=["date"]), on=["race_id", "horse_id"], how="left")
    return out.drop(columns=["date"])


def cmd_build(a) -> None:
    runs = pd.read_parquet(OUT / "runs.parquet")
    runs = runs.drop_duplicates(KEY)
    F = asof_features(runs)
    F = F.drop_duplicates(KEY)
    for c in F.columns:
        if c not in KEY:
            F[c] = F[c].astype("float32")
    F.to_parquet(OUT / "rob.parquet", index=False)
    cov = {c: float(F[c].notna().mean()) for c in F.columns if c.endswith("ewm5") or c.endswith("_n")}
    (OUT / "rob_build.json").write_text(json.dumps({"rows": int(len(F)), "coverage": cov}, ensure_ascii=False, indent=1))
    _log(f"saved {F.shape} {cov}")


def cmd_audit(a) -> None:
    """先読みの検査: 2023-07-01 以降の指数をすべて乱数に置き換えて（結果ごと反転）作り直しても、その日より前の行の特徴量は完全に一致する。
    さらに、元の表で『当日の行の指数を使っていない』ことを、当日の指数を NaN にしても同日の特徴量が変わらないことで確かめる。"""
    runs = pd.read_parquet(OUT / "runs.parquet").drop_duplicates(KEY)
    cut = pd.Timestamp("2023-07-01")
    base = asof_features(runs[pd.to_datetime(runs["date"]) < pd.Timestamp("2024-01-01")])
    alt_runs = runs[pd.to_datetime(runs["date"]) < pd.Timestamp("2024-01-01")].copy()
    m = pd.to_datetime(alt_runs["date"]) >= cut
    rng = np.random.default_rng(0)
    for c in ("megu_index_v3a", "nk"):
        alt_runs.loc[m, c] = rng.normal(60, 20, int(m.sum()))
    alt = asof_features(alt_runs)
    bd = base.merge(alt, on=KEY, suffixes=("", "_alt")).merge(runs[KEY + ["date"]], on=KEY, how="left")
    bd["date"] = pd.to_datetime(bd["date"])
    before = bd["date"] < cut
    cols = [c for c in base.columns if c not in KEY]
    diff = {c: float(np.nanmax(np.abs(bd.loc[before, c].to_numpy(float) - bd.loc[before, c + "_alt"].to_numpy(float)))) for c in cols}
    rep = {"cut": str(cut.date()), "rows_before_cut": int(before.sum()), "max_abs_diff": diff}
    # 同日: 切断日当日の行（その日の指数を乱数にしたもの）も、その日の特徴量は変わらない
    same = bd["date"] == cut
    if same.any():
        rep["max_abs_diff_cut_day"] = {c: float(np.nanmax(np.abs(bd.loc[same, c].to_numpy(float) - bd.loc[same, c + "_alt"].to_numpy(float)))) for c in cols}
    (OUT / "rob_audit.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    _log(f"先読み検査: 切断日より前の最大差 {max(diff.values()):.2e}（{int(before.sum())} 行）")


def cmd_run(a) -> None:
    from src.scripts.data import tp27_eval as t
    from src.scripts.data import v5_select_eval as v5

    t.SEEDS = [int(s) for s in a.seeds.split(",")]
    df, base, meta, folds, lp = t.setup()
    X = v5.build_frame(v5.CONFIGS["d_CY_BEST"], df)
    U = pd.read_parquet(OUT / "rob.parquet")
    M = df[KEY].merge(U, on=KEY, how="left")
    cols = [c for c in U.columns if c not in KEY]
    if a.only:
        cols = [c for c in cols if any(c.startswith(p) for p in a.only.split(","))]
    E = M[cols].astype("float32")
    E.index = X.index
    X = pd.concat([X, E], axis=1)
    _log(f"{len(cols)} 列を足す → {X.shape[1]} 列")
    assert not X.columns.duplicated().any()
    od = OUT / f"lgb_{a.name}"
    od.mkdir(parents=True, exist_ok=True)
    lv, by, oof = t._run(a.name, X, meta, folds, lp)
    oof.to_parquet(od / "oof.parquet", index=False)
    prev = pd.read_parquet(BASE_OOF)
    for d in (prev, oof):
        d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    fin_map = meta.set_index(KEY)["finish_position"]
    cmp = t._cmp(prev, oof, fin_map, a.name)
    per_year = {str(y): t._cmp(prev[prev["year"] == y], oof[oof["year"] == y], fin_map, f"{a.name}_{y}") for y in (2022, 2023, 2024)}
    res = {"name": a.name, "n_cols": int(X.shape[1]), "added": cols, "levels": lv, "by_fold": by, "compare": cmp, "per_year": per_year, "seeds": t.SEEDS}
    (od / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    _log(f"{a.name}: Δwin={cmp['Δwin']:.5f} 上限={cmp['win_hi']:.5f} Δtop2={cmp['Δtop2']:.5f} Δtop3={cmp['Δtop3']:.5f} 採用={cmp['採用']}")
    for y, c in per_year.items():
        _log(f"  {y}: Δwin={c['Δwin']:.5f} Δtop2={c['Δtop2']:.5f} Δtop3={c['Δtop3']:.5f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("audit")
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--only", default="")
    r.add_argument("--seeds", default="0,1")
    a = ap.parse_args()
    {"build": cmd_build, "audit": cmd_audit, "run": cmd_run}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
