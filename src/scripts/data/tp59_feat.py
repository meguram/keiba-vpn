"""tp-59 (5): 騎手・調教師の特徴量としての寄与。V5（1,142 列）に対する（a）関係者の既存列を外した増分、（b）『上積み』群（tp59_uplift）を足した増分。

python -m src.scripts.data.tp59_feat build                    # 上積み群を作る（2020〜2025 年の全出走）→ <TP59_OUT>/uplift.parquet
python -m src.scripts.data.tp59_feat audit                    # 先読みの検査（直前 1 行をずらした値との一致、同日の結果を使っていないこと）と被覆
python -m src.scripts.data.tp59_feat run --name NAME [--drop-rider] [--add-uplift] [--seeds 0,1]
                                                              # V5 と同じ設定（検証 2022/23/24・seed）で学習し、V5 の OOF と対で比べる
採用基準は feature_bakeoff.compare_oof（勝率の 95% 区間の上限 < 0、効果 ≥ 0.0002、連対・複勝の悪化 ≤ 0.0002）。test 2025・2026 は使わない。市場オッズ・人気は特徴量に使わない。
環境変数: TP40_THREADS（既定 4）, TP59_OUT。
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
OUT = Path(os.environ.get("TP59_OUT", "/home/jovyan/work/tp59_out"))
TABLES = ROOT / "data/page_reference/tables"
BASE_OOF = ROOT / "data/local/modeling/training_pipeline_v4/v5_select/oof_d_CY_BEST.parquet"
KEY = ["race_id", "horse_id"]
RIDER_COLS = ["jockey_prior_races", "jockey_prior_wins", "jockey_prior_top3", "jockey_prior_win_rate", "jockey_prior_top3_rate",
              "trainer_prior_races", "trainer_prior_wins", "trainer_prior_top3", "trainer_prior_win_rate", "trainer_prior_top3_rate"]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_history(years=range(2020, 2026)) -> pd.DataFrame:
    import polars as pl

    from src.pipeline.features.tp59_uplift import HIST_COLS

    parts = []
    for y in years:
        f = TABLES / str(y) / "race_result_flat.parquet"
        if f.exists():
            sch = pl.read_parquet_schema(f)
            parts.append(pl.read_parquet(f, columns=[c for c in HIST_COLS if c in sch]))
    h = pl.concat(parts, how="diagonal_relaxed").to_pandas()
    h["race_id"], h["horse_id"] = h["race_id"].astype(str), h["horse_id"].astype(str)
    return h.drop_duplicates(["race_id", "horse_number"]).reset_index(drop=True)


def cmd_build(a) -> None:
    from src.pipeline.features import tp59_uplift as U

    h = U.prepare_history(load_history())
    F = U.build_features(h)
    res = pd.concat([h[KEY].reset_index(drop=True), F.reset_index(drop=True)], axis=1).drop_duplicates(KEY)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT / "uplift.parquet", index=False)
    _log(f"saved {res.shape}")


def cmd_audit(a) -> None:
    """(1) 被覆、(2) 先読み: 日付を 1 日ずらして（全出走の日付 +1 日）作り直しても、元の日付の値と『同日の結果を含まない』ことの確認、(3) 期待値の分布。"""
    from src.pipeline.features import tp59_uplift as U

    F = pd.read_parquet(OUT / "uplift.parquet")
    cols = [c for c in F.columns if c.startswith("up_")]
    cov = {c: float(F[c].notna().mean()) for c in cols}
    h = U.prepare_history(load_history(range(2020, 2023)))
    base = U.build_features(h)
    # 先読みの検査: 2022 年の全レースの結果を『反転』（着順を逆順）しても、その日以前の行の特徴量は変わらない
    h2 = h.copy()
    cut = pd.Timestamp("2022-06-01")
    m = h2["date"] >= cut
    h2.loc[m, "finish_position"] = pd.to_numeric(h2.loc[m, "finish_position"], errors="coerce").sample(frac=1.0, random_state=0).to_numpy()
    h2 = U.prepare_history(h2)
    alt = U.build_features(h2)
    before = (h["date"] < cut).to_numpy()
    diff = {c: float(np.nanmax(np.abs(base.loc[before, c].to_numpy(dtype=float) - alt.loc[before, c].to_numpy(dtype=float)))) for c in base.columns}
    # 当日（cut の日）の行も、同日の結果を使わないので変わらないはず
    same = (h["date"] == cut).to_numpy() if (h["date"] == cut).any() else (h["date"] == h.loc[~before, "date"].min()).to_numpy()
    diff_same = {c: float(np.nanmax(np.abs(base.loc[same, c].to_numpy(dtype=float) - alt.loc[same, c].to_numpy(dtype=float)))) for c in base.columns}
    rep = {"coverage": cov, "max_abs_diff_before_cut": diff, "max_abs_diff_cut_day": diff_same, "cut": str(cut.date()), "rows": int(len(h))}
    (OUT / "uplift_audit.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    _log(f"先読み検査: 切断日より前の最大差 {max(diff.values()):.2e} / 切断日当日 {max(diff_same.values()):.2e}")


UP_GROUPS = {
    "UP_j": ["up_j_res", "up_j_top3", "up_j_n"], "UP_t": ["up_t_res", "up_t_top3", "up_t_n"], "UP_jt": ["up_jt_res", "up_jt_top3", "up_jt_n"],
    "UP_cond": ["up_jsd_res", "up_jsd_top3", "up_jsd_n", "up_tsd_res", "up_tsd_top3", "up_tsd_n", "up_jv_res", "up_jv_top3", "up_jv_n"],
    "UP_w180": ["up_j_w180_res", "up_j_w180_top3", "up_t_w180_res", "up_t_w180_top3"],
    "UP_transfer": ["up_t_transfer_res", "up_t_transfer_top3", "up_t_transfer_n"], "UP_change": ["up_j_change"],
}
UP_GROUPS["UP_ALL"] = sorted({c for v in UP_GROUPS.values() for c in v})


def add_oof_residual_features(d: pd.DataFrame, m: pd.DataFrame, k: float = 100.0) -> pd.DataFrame:
    """（一次選別だけ）関係者の『V5 の OOF 確率に対する勝ち・複勝の残差』の as-of 縮約平均。オッズは使わない。
    OOF は他の年で学習したモデルの予測なので、特徴量として本番に使うなら expanding の OOF で作り直す必要がある（ここは増分の有無を見るだけ）。"""
    ids = pd.read_parquet(OUT / "master.parquet", columns=["race_id", "horse_id", "jockey_id", "trainer_id"])
    ids["race_id"], ids["horse_id"] = ids["race_id"].astype(str), ids["horse_id"].astype(str)
    d = d.merge(ids.drop_duplicates(KEY), on=KEY, how="left")
    d["r_win"] = d["y_win"] - d["p_win"]
    d["r_top3"] = d["y_top3"] - d["p_top3"]
    for nm, key in (("j", "jockey_id"), ("t", "trainer_id")):
        daily = d.groupby([key, "date"], as_index=False).agg(sw=("r_win", "sum"), s3=("r_top3", "sum"), n=("r_win", "size")).sort_values([key, "date"])
        for c in ("sw", "s3", "n"):
            daily[f"c_{c}"] = daily.groupby(key)[c].cumsum() - daily[c]
        d = d.merge(daily[[key, "date", "c_sw", "c_s3", "c_n"]], on=[key, "date"], how="left")
        d[f"rs_{nm}_win"] = d["c_sw"].fillna(0.0) / (d["c_n"].fillna(0.0) + k)
        d[f"rs_{nm}_top3"] = d["c_s3"].fillna(0.0) / (d["c_n"].fillna(0.0) + k)
        d = d.drop(columns=["c_sw", "c_s3", "c_n"])
    return d


def cmd_screen(a) -> None:
    """V5 の OOF のロジットに上積み列の線形項を足す一次選別（tp56_eval.loyo。LOYO、レース単位ブートストラップ）。LightGBM での本評価（run）の前に見込みを見る。"""
    from src.scripts.data import tp56_eval as e56

    oof = pd.read_parquet(BASE_OOF)
    for c in KEY:
        oof[c] = oof[c].astype(str)
    m = pd.read_parquet(OUT / "master.parquet", columns=["race_id", "horse_id", "date", "finish_position"])
    m["race_id"], m["horse_id"] = m["race_id"].astype(str), m["horse_id"].astype(str)
    U = pd.read_parquet(OUT / "uplift.parquet")
    d = oof.merge(m.drop_duplicates(KEY), on=KEY, how="inner").merge(U, on=KEY, how="left")
    d = d.sort_values(["date", "race_id", "horse_id"], kind="stable").reset_index(drop=True)
    d["y_win"] = (d["finish_position"] == 1).astype(float)
    d["y_top2"] = (d["finish_position"] <= 2).astype(float)
    d["y_top3"] = (d["finish_position"] <= 3).astype(float)
    d = d[d.groupby("race_id")["y_win"].transform("sum") == 1].reset_index(drop=True)
    d["ridx"] = pd.factorize(d["race_id"])[0]
    d = add_oof_residual_features(d, m)
    UP_GROUPS["RS_j"] = ["rs_j_win", "rs_j_top3"]
    UP_GROUPS["RS_t"] = ["rs_t_win", "rs_t_top3"]
    UP_GROUPS["RS_all"] = ["rs_j_win", "rs_j_top3", "rs_t_win", "rs_t_top3"]
    res = {"rows": int(len(d)), "groups": {}}
    for name, cols in UP_GROUPS.items():
        r = e56.loyo(d, cols)
        w, t2, t3 = r["win"], r["top2"], r["top3"]
        ok = (w["hi"] < 0 and w["delta"] <= -0.0002 and t2["delta"] <= 0.0002 and t3["delta"] <= 0.0002)
        res["groups"][name] = {"cols": cols, **{k: v for k, v in r.items() if k != "theta"}, "adopt": bool(ok)}
        _log(f"{name:12s} win {w['delta']:+.5f} [{w['lo']:+.5f},{w['hi']:+.5f}] top2 {t2['delta']:+.5f} top3 {t3['delta']:+.5f} 採用={ok}")
    (OUT / "uplift_screen.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def rs_features(df: pd.DataFrame, k: float = 100.0) -> pd.DataFrame:
    """関係者の『V5 の OOF（expanding。評価年より前の年だけで学習したモデル）に対する勝ち・複勝の残差』の as-of 縮約平均。
    OOF のある 2022〜2024 年の行にだけ値が付き（履歴は 2022 年以降）、それ以外は欠損。同日の結果は使わない。市場オッズは使わない。"""
    oof = pd.read_parquet(BASE_OOF)[KEY + ["p_win", "p_top3"]]
    for c in KEY:
        oof[c] = oof[c].astype(str)
    base = df[KEY + ["date", "jockey_id", "trainer_id", "finish_position"]].copy()
    base = base.merge(oof, on=KEY, how="left")
    base["date"] = pd.to_datetime(base["date"])
    has = base["p_win"].notna()
    base["r_win"] = ((base["finish_position"] == 1).astype(float) - base["p_win"]).where(has)
    base["r_top3"] = ((base["finish_position"] <= 3).astype(float) - base["p_top3"]).where(has)
    base["one"] = has.astype(float)
    out = pd.DataFrame(index=df.index)
    in_oof_years = (pd.to_datetime(df["date"]) >= pd.Timestamp("2022-01-01")).to_numpy()
    for nm, key in (("j", "jockey_id"), ("t", "trainer_id")):
        daily = base[has].groupby([key, "date"], as_index=False).agg(sw=("r_win", "sum"), s3=("r_top3", "sum"), n=("one", "sum")).sort_values([key, "date"])
        for c in ("sw", "s3", "n"):
            daily[f"c_{c}"] = daily.groupby(key)[c].cumsum() - daily[c]
        m = base[[key, "date"]].merge(daily[[key, "date", "c_sw", "c_s3", "c_n"]], on=[key, "date"], how="left")
        m.index = df.index
        out[f"rs_{nm}_win"] = (m["c_sw"].fillna(0.0) / (m["c_n"].fillna(0.0) + k)).where(in_oof_years)
        out[f"rs_{nm}_top3"] = (m["c_s3"].fillna(0.0) / (m["c_n"].fillna(0.0) + k)).where(in_oof_years)
    return out.astype("float32")


def cmd_run(a) -> None:
    from src.scripts.data import tp27_eval as t
    from src.scripts.data import v5_select_eval as v5

    t.SEEDS = [int(s) for s in a.seeds.split(",")]
    df, base, meta, folds, lp = t.setup()
    X = v5.build_frame(v5.CONFIGS["d_CY_BEST"], df)
    if a.drop_rider:
        miss = [c for c in RIDER_COLS if c not in X.columns]
        assert not miss, miss
        X = X.drop(columns=RIDER_COLS)
        _log(f"関係者の既存列 {len(RIDER_COLS)} 本を外す → {X.shape[1]} 列")
    n_add = 0
    if a.add_uplift:
        U = pd.read_parquet(OUT / "uplift.parquet")
        M = df[KEY].merge(U, on=KEY, how="left")
        cols = [c for c in U.columns if c.startswith("up_")]
        if a.only:
            cols = [c for c in cols if any(c.startswith(p) for p in a.only.split(","))]
        E = M[cols].astype("float32")
        E.index = X.index
        X = pd.concat([X, E], axis=1)
        n_add = len(cols)
        _log(f"上積み {n_add} 列を足す → {X.shape[1]} 列")
    if a.add_rs:
        E = rs_features(df)
        E.index = X.index
        X = pd.concat([X, E], axis=1)
        n_add += E.shape[1]
        _log(f"OOF 残差の群 {E.shape[1]} 列を足す → {X.shape[1]} 列")
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
    res = {"name": a.name, "n_cols": int(X.shape[1]), "n_added": n_add, "drop_rider": bool(a.drop_rider), "levels": lv, "by_fold": by, "compare": cmp, "per_year": per_year, "seeds": t.SEEDS}
    (od / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    _log(f"{a.name}: Δwin={cmp['Δwin']:.5f} 上限={cmp['win_hi']:.5f} Δtop2={cmp['Δtop2']:.5f} Δtop3={cmp['Δtop3']:.5f} 採用={cmp['採用']}")
    for y, c in per_year.items():
        _log(f"  {y}: Δwin={c['Δwin']:.5f} Δtop2={c['Δtop2']:.5f} Δtop3={c['Δtop3']:.5f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("audit")
    sub.add_parser("screen")
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--drop-rider", action="store_true")
    r.add_argument("--add-uplift", action="store_true")
    r.add_argument("--add-rs", action="store_true")
    r.add_argument("--only", default="")
    r.add_argument("--seeds", default="0,1")
    a = ap.parse_args()
    {"build": cmd_build, "audit": cmd_audit, "screen": cmd_screen, "run": cmd_run}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
