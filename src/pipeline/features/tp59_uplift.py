"""tp-59: 騎手・調教師の『上積み』（その馬の過去の平均よりどれだけ良い結果を出させたか）の as-of 特徴量。

どの列も『そのレースの日より厳密に前』の結果だけで作る。市場オッズ・人気は使わない。関係者は ID（jockey_id / trainer_id）。
定義: 馬ごとに『それまでの出走の平均（正規化着順 res、複勝圏 top3）』を期待値とし、今回の結果との差 d = 結果 − 期待値 を各出走に付ける（期待値には 2 走以上必要）。
関係者の上積み = d の as-of の縮約平均（(Σd) / (n + k)、平均 0 へ縮約）。騎手 / 調教師 / 騎手×調教師 / 騎手×条件 / 直近 180 日 / 前走騎手との差 / 転厩直後の上積み。
既存の G2（factory_conn）は『結果そのものの率』で、馬の力を引いた上積みは持っていない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from src.pipeline.features.factory_conn import _asof_sums, _full

K_SHR = 30.0
HIST_COLS = ["race_id", "date", "venue_code", "surface", "distance", "finish_position", "horse_number", "horse_id", "jockey_id", "trainer_id"]


def prepare_history(h: pd.DataFrame) -> pd.DataFrame:
    """h: race_result_flat 由来（全馬・全レース）。馬ごとの期待値との差 d_res / d_top3 と、条件の区分を足す。行の並び・index は変えない。"""
    h = h.copy()
    h["_i"] = np.arange(len(h))
    h["date"] = pd.to_datetime(h["date"])
    fin = pd.to_numeric(h["finish_position"], errors="coerce")
    h["ok"] = fin > 0
    fin = fin.where(h["ok"])
    n_fin = fin.groupby(h["race_id"]).transform("count")
    h["res"] = (0.5 - (fin - 1.0) / np.maximum(n_fin - 1.0, 1.0)).where(h["ok"])
    h["top3"] = (fin <= 3).astype(float).where(h["ok"])
    s = h.sort_values(["horse_id", "date", "race_id"], kind="stable")
    g = s.groupby("horse_id", sort=False)
    cnt = g["res"].transform(lambda x: x.notna().cumsum().shift(1))
    for v in ("res", "top3"):
        csum = g[v].transform(lambda x: x.fillna(0.0).cumsum().shift(1))
        mean_prior = (csum / cnt).where(cnt >= 2)
        s[f"d_{v}"] = s[v] - mean_prior
    s["prev_jockey_id"] = g["jockey_id"].shift(1)
    s["prev_trainer_id"] = g["trainer_id"].shift(1)
    s["chg_trainer"] = np.where(s["prev_trainer_id"].notna(), (s["trainer_id"] != s["prev_trainer_id"]).astype(float), np.nan)
    s["d_res_tr"] = s["d_res"].where(s["chg_trainer"] == 1)
    s["d_top3_tr"] = s["d_top3"].where(s["chg_trainer"] == 1)
    h = s.sort_values("_i")
    d = pd.to_numeric(h["distance"], errors="coerce")
    db = pd.cut(d, [0, 1300, 1500, 1900, 2400, 99999], labels=False).astype("float")
    h["c_sd"] = h["surface"].astype(str) + db.astype("Int64").astype(str)
    return h


def _key(h: pd.DataFrame, parts: list[str]) -> pd.Series:
    s = h[parts[0]].astype("string")
    for p in parts[1:]:
        s = s + "|" + h[p].astype("string")
    return s.where(h[parts].notna().all(axis=1))


def _frame(h: pd.DataFrame, keys: dict[str, pd.Series], vcols: list[str]) -> pl.DataFrame:
    df = pd.DataFrame({"_i": h["_i"].to_numpy(), "date": h["date"].dt.date, "ok": h["ok"].to_numpy()})
    for c in vcols:
        df[c] = h[c].to_numpy()
    out = pl.from_pandas(df)
    for k, s in keys.items():
        out = out.with_columns(pl.Series(k, s.astype("object").where(s.notna(), None).tolist(), dtype=pl.Utf8))
    return out


def _shr(sums: dict[str, np.ndarray], v: str, k: float = K_SHR) -> np.ndarray:
    return (sums["S_" + v] / (sums["n_" + v] + k)).astype("float32")


def build_features(h: pd.DataFrame) -> pd.DataFrame:
    """h は prepare_history を通したもの。戻り値は h と同じ並びの特徴量表（列名は up_ 始まり）。"""
    keys = {
        "k_j": _key(h, ["jockey_id"]), "k_t": _key(h, ["trainer_id"]), "k_jt": _key(h, ["jockey_id", "trainer_id"]),
        "k_jsd": _key(h, ["jockey_id", "c_sd"]), "k_tsd": _key(h, ["trainer_id", "c_sd"]), "k_jv": _key(h, ["jockey_id", "venue_code"]),
    }
    vcols = ["d_res", "d_top3", "d_res_tr", "d_top3_tr"]
    P = _frame(h, keys, vcols)
    n = len(h)
    out: dict[str, np.ndarray] = {}
    for ent in ("j", "t", "jt", "jsd", "tsd", "jv"):
        s = _full(_asof_sums(P, f"k_{ent}", ["d_res", "d_top3"]), n, ["d_res", "d_top3"])
        out[f"up_{ent}_res"] = _shr(s, "d_res")
        out[f"up_{ent}_top3"] = _shr(s, "d_top3")
        out[f"up_{ent}_n"] = np.log1p(s["n_d_res"]).astype("float32")
    for ent in ("j", "t"):
        s = _full(_asof_sums(P, f"k_{ent}", ["d_res", "d_top3"], window=180), n, ["d_res", "d_top3"])
        out[f"up_{ent}_w180_res"] = _shr(s, "d_res", 10.0)
        out[f"up_{ent}_w180_top3"] = _shr(s, "d_top3", 10.0)
    s = _full(_asof_sums(P, "k_t", ["d_res_tr", "d_top3_tr"]), n, ["d_res_tr", "d_top3_tr"])
    out["up_t_transfer_res"] = _shr(s, "d_res_tr")
    out["up_t_transfer_top3"] = _shr(s, "d_top3_tr")
    out["up_t_transfer_n"] = np.log1p(s["n_d_res_tr"]).astype("float32")
    # 前走の騎手の上積みとの差（乗り替わりの上げ下げ。同じ騎手・初出走は 0）
    prevj = _key(h, ["prev_jockey_id"])
    src = P.filter(pl.col("k_j").is_not_null() & pl.col("ok")).select(["k_j", "date", "d_res"])
    daily = src.group_by(["k_j", "date"]).agg([pl.col("d_res").sum().alias("S"), pl.col("d_res").count().alias("n")]).sort(["k_j", "date"])
    cum = daily.with_columns([pl.col("S").cast(pl.Float64).cum_sum().over("k_j"), pl.col("n").cast(pl.Float64).cum_sum().over("k_j")]).sort("date")
    q = pl.DataFrame({"_i": h["_i"].to_numpy(), "k_j": prevj.astype("object").where(prevj.notna(), None).tolist(), "date": h["date"].dt.date}, schema_overrides={"k_j": pl.Utf8}).filter(pl.col("k_j").is_not_null()).sort("date")
    look = q.join_asof(cum, on="date", by="k_j", strategy="backward", allow_exact_matches=False).sort("_i")
    prev_up = np.full(n, np.nan)
    prev_up[look["_i"].to_numpy()] = (look["S"].fill_null(0.0) / (look["n"].fill_null(0.0) + K_SHR)).to_numpy()
    chg = (h["jockey_id"].astype("string") != h["prev_jockey_id"].astype("string")) & h["prev_jockey_id"].notna()
    out["up_j_change"] = np.where(chg.to_numpy(), out["up_j_res"] - prev_up, 0.0).astype("float32")
    return pd.DataFrame(out, index=h.index)
