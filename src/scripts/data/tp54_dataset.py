"""tp-54 (2/3): 追走難度 v2 の学習表（1 行 = 1 頭 1 レース）。

python -m src.scripts.data.tp54_dataset      # ホストで実行（KEIBA_ROOT / TP54_OUT で場所を指定可）
出力: $TP54_OUT/horse_ease.parquet

定義（ユーザー指示 2026-10-20）: 追走難度 ＝ 『想定位置取りを取るのに必要な通過ラップが、その馬にとって楽か大変か』。
- 要求（demand）: 想定位置取り p̂（position_forecast の出力）と、今回の想定ペース z_hat（先行勢の数・圧力から推定）。
- 供給（capacity）: その馬の過去の走から測った、位置取りの効果を除いた上がりの力（as-of）と、前で運んだときの終いの落ち方。
- 実測のラベル Y: 実際の上がり偏差（レース内の中央値との差、秒）から、その馬の『位置取りの効果を除いた過去の上がり偏差（as-of）』を引いたもの。
  Y が大きい＝終いが落ちた（余裕が無かった）、小さい＝余裕があった。位置取り・ペースの効果は Y に残る（＝要求が重いほど Y は大きい）。
- すべて as-of: 馬ごとの過去の集計は shift(1)（対象のレースを含まない）。同じ日の別レースも使わない（日付昇順で並べ、同日は除く）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("KEIBA_ROOT") or Path(__file__).resolve().parents[3])
TP53 = ROOT / "data/local/modeling/tp53"
TABLES = ROOT / "data/page_reference/tables"
OUT = Path(os.environ.get("TP54_OUT") or ROOT / "data/local/modeling/tp54")
KEY = ["venue", "surface", "distance", "going_group"]
FRONT_E = 0.35      # 『前に行った』の境界（初角の相対位置）
REAR_E = 0.65


def prior_stats(df: pd.DataFrame, value_cols: dict[str, pd.Series], group: str = "horse_id", date: str = "date", window: int = 5) -> pd.DataFrame:
    """馬ごとの過去（対象の行より前）の集計。df は horse_id, date で整列済みであること。同日の行は互いに見ない。

    返り値は df と同じ index。value_cols の各列について 直近 window 走の平均（prior5_<name>）と通算の平均（prior_<name>）を返す。
    同日に複数走る馬（実際には無いが、同日 2 レースの抜けを防ぐ）は、同日分を shift の前に日付単位でまとめず、
    単純に『行順で 1 つ前まで』を使う。日付昇順で整列していれば未来の行は含まれない。
    """
    out = {}
    for name, s in value_cols.items():
        s = s.astype(float)
        sh = s.groupby(df[group], sort=False).shift(1)
        out[f"prior5_{name}"] = sh.groupby(df[group], sort=False).transform(lambda x: x.rolling(window, min_periods=1).mean())
        csum = s.fillna(0).groupby(df[group], sort=False).cumsum() - s.fillna(0)
        ccnt = s.notna().astype(float).groupby(df[group], sort=False).cumsum() - s.notna().astype(float)
        out[f"prior_{name}"] = (csum / ccnt.replace(0, np.nan))
    return pd.DataFrame(out, index=df.index)


def conditional_prior_mean(df: pd.DataFrame, value: pd.Series, mask: pd.Series, group: str = "horse_id") -> pd.Series:
    """mask が真の過去の走だけの value の平均（対象の行より前）。"""
    v = (value.astype(float) * mask.astype(float)).fillna(0)
    m = (mask & value.notna()).astype(float)
    cs = v.groupby(df[group], sort=False).cumsum() - v
    cc = m.groupby(df[group], sort=False).cumsum() - m
    return cs / cc.replace(0, np.nan)


def conditional_prior_count(df: pd.DataFrame, mask: pd.Series, group: str = "horse_id") -> pd.Series:
    m = mask.astype(float)
    return m.groupby(df[group], sort=False).cumsum() - m


def position_effect_model(h: pd.DataFrame) -> dict:
    """上がり偏差 ~ 位置取り e（0.5 からの差）× (1, 実測ペース z, 距離, 面) の最小二乗。位置取りの効果を除くために使う。"""
    d = h.dropna(subset=["act_l3f_dev", "act_early_norm", "z_act"]).copy()
    e = d["act_early_norm"] - 0.5
    X = np.column_stack([np.ones(len(d)), e, e * d["z_act"].clip(-2, 2), e * (d["distance"].astype(float) - 1600) / 1000, e * (d["surface"] == "芝").astype(float)])
    y = d["act_l3f_dev"].to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"features": ["const", "e", "e_z", "e_dist", "e_turf"], "coef": [float(c) for c in coef]}


def position_effect(m: dict, early: pd.Series, z: pd.Series, distance: pd.Series, surface: pd.Series) -> pd.Series:
    c = dict(zip(m["features"], m["coef"]))
    e = early - 0.5
    return c["e"] * e + c["e_z"] * e * z.clip(-2, 2) + c["e_dist"] * e * (distance.astype(float) - 1600) / 1000 + c["e_turf"] * e * (surface == "芝").astype(float)


def build() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    h = pd.read_parquet(TP53 / "horse.parquet")
    r = pd.read_parquet(TP53 / "race.parquet")
    r["act_t3f"] = pd.to_numeric(r["act_t3f"], errors="coerce")
    r["going_group"] = r["track_condition"].map(lambda t: "良" if str(t) == "良" else "道悪")
    ok = r["act_t3f"].between(28, 45)
    g = r[ok].groupby(KEY)["act_t3f"]
    r.loc[ok, "n_c"], r.loc[ok, "med"], r.loc[ok, "sd"] = g.transform("size"), g.transform("median"), g.transform("std")
    r["z_act"] = np.where(ok & (r["n_c"] >= 30) & (r["sd"] > 0), (r["act_t3f"] - r["med"]) / r["sd"], np.nan)

    # 想定ペース z_hat: 年別の expanding 当てはめ（年 Y の係数は Y より前の年で当てはめ。2020 年だけ自年）
    try:
        from src.scripts.data.tp54_pace import design, fit_ridge, predict_ridge
    except ImportError:      # ホスト（repo に未配置）では同じ場所のファイルを使う
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from tp54_pace import design, fit_ridge, predict_ridge  # type: ignore

    rr = r.copy()
    X = design(rr)
    z_hat = pd.Series(np.nan, index=rr.index)
    lab = rr["z_act"].notna()
    for y in sorted(rr["year"].unique()):
        tr = lab & (rr["year"] < y) if y > rr["year"].min() else lab & (rr["year"] == y)
        m = fit_ridge(X[tr], rr.loc[tr, "z_act"].to_numpy())
        te = rr["year"] == y
        z_hat[te] = predict_ridge(m, X[te])
    rr["z_hat"] = z_hat
    rcols = ["race_id", "z_act", "z_hat", "pf_level", "pf_front", "pf_stalker", "pf_pressure", "pf_known", "act_t3f", "act_race_l3f"]
    h = h.merge(rr[rcols], on="race_id", how="left")

    # 追加の列（走破タイム・馬体重・斤量・性齢）
    extra = []
    for y in sorted(h["year"].unique()):
        e = pd.read_parquet(TABLES / str(int(y)) / "race_result_flat.parquet", columns=["race_id", "horse_number", "time_sec", "bracket_number", "weight", "weight_change", "jockey_weight", "sex_age", "jockey_id", "trainer_id"])
        extra.append(e)
    ex = pd.concat(extra, ignore_index=True)
    h = h.merge(ex, on=["race_id", "horse_number"], how="left")
    h["act_tpre"] = pd.to_numeric(h["time_sec"], errors="coerce") - h["act_l3f"]      # 上がり 3F の手前までの走破タイム（秒）
    h["tpre_dev"] = h["act_tpre"] - h.groupby("race_id")["act_tpre"].transform("median")
    h["age"] = pd.to_numeric(h["sex_age"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    h["date"] = pd.to_datetime(h["date"].astype(str).str.replace(r"[./]", "-", regex=True), errors="coerce")
    h = h.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)

    # 位置取りの効果を除いた上がり偏差（その馬の『位置取りに依らない上がりの力』）
    pe = position_effect_model(h)
    h["pos_eff_act"] = position_effect(pe, h["act_early_norm"], h["z_act"].fillna(0), h["distance"], h["surface"])
    h["l3f_adj"] = h["act_l3f_dev"] - h["pos_eff_act"]
    h["dpos"] = h["act_late_norm"] - h["act_early_norm"]     # 負＝道中・終盤で押し上げた
    front = h["act_early_norm"] < FRONT_E
    rear = h["act_early_norm"] > REAR_E
    ps = prior_stats(h, {"l3f_adj": h["l3f_adj"], "l3f_dev": h["act_l3f_dev"], "early": h["act_early_norm"], "tpre_dev": h["tpre_dev"], "zact": h["z_act"], "finq": (h["finish_position"] / h["field_size"])})
    h = pd.concat([h, ps], axis=1)
    h["n_prior"] = conditional_prior_count(h, h["act_early_norm"].notna())
    h["n_prior_front"] = conditional_prior_count(h, front)
    h["n_prior_rear"] = conditional_prior_count(h, rear)
    h["prior_l3f_adj_front"] = conditional_prior_mean(h, h["l3f_adj"], front)
    h["prior_l3f_dev_front"] = conditional_prior_mean(h, h["act_l3f_dev"], front)
    h["prior_l3f_dev_rear"] = conditional_prior_mean(h, h["act_l3f_dev"], rear)
    fastrace = h["z_act"] < -0.5
    h["prior_l3f_dev_fast"] = conditional_prior_mean(h, h["act_l3f_dev"], fastrace)
    h["n_prior_fast"] = conditional_prior_count(h, fastrace)
    h["prior_l3f_adj_fast"] = conditional_prior_mean(h, h["l3f_adj"], fastrace)

    # ラベル Y: 実測の上がり偏差 − その馬の『位置取りの効果を除いた過去の上がり偏差』（as-of）
    base = h["prior5_l3f_adj"].fillna(h["prior_l3f_adj"])
    h["ability_adj"] = base
    h["Y"] = h["act_l3f_dev"] - h["ability_adj"]
    h["Y_adj"] = h["l3f_adj"] - h["ability_adj"]          # 今回の位置取りの効果も除いた『力のぶれ』（参考）

    # 前走（実測）
    grp = h.groupby("horse_id", sort=False)
    for c in ("act_early_norm", "act_l3f_dev", "z_act", "distance", "field_size", "date", "finish_position"):
        h[f"prev_{c}"] = grp[c].shift(1)
    h["prev_surface"] = grp["surface"].shift(1)
    h["rest_days"] = (h["date"] - h["prev_date"]).dt.days
    h["gate_norm"] = h["horse_number"] / h["field_size"]
    h["pf_early_filled"] = h["pf_early"]
    h.to_parquet(OUT / "horse_ease.parquet", index=False)
    (OUT / "position_effect.json").write_text(json.dumps(pe, ensure_ascii=False), encoding="utf-8")
    print({"rows": len(h), "Y_nonnull": int(h["Y"].notna().sum()), "years": sorted(map(int, h["year"].unique()))})
    return h


if __name__ == "__main__":
    build()
