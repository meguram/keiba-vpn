"""tp-31: コース特注の特徴量（厳密コース × 属性の as-of 経験ベイズ lift）。

各レースの各馬について、「そのコースで、この属性の馬が、全コースでの同じ属性の馬より、どれだけ着順残差が良かったか」を、
そのレース日より厳密に前の結果だけで作る。階層縮約: 全体 → 属性（全コース）→ 面×距離帯×属性 → 場×面×距離帯×属性 → 厳密コース×属性。
出力は属性ごとに lift（細かい階層の推定 − 属性の全コース推定）と、厳密コースのセルの件数。

先読み防止: 集計は (キー, 日付) 単位の累積和から「同日を含まない」前日までの値を引く（同日・同レースの結果は使わない）。
キーは ID（種牡馬・母父・騎手・調教師）と、結果に依存しないビン（脚質・枠・年齢など）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

K_GLOBAL = 30.0                     # 属性の全コース推定を全体平均へ縮約する強さ
K_LEVEL = (40.0, 30.0, 20.0)         # 面×距離帯 → 場×面×距離帯 → 厳密コース の縮約の強さ


# ───────────────────────── キー
def dist_band(distance: pd.Series) -> pd.Series:
    d = pd.to_numeric(distance, errors="coerce")
    return pd.cut(d, [0, 1300, 1500, 1900, 2400, 99999], labels=["S", "SM", "M", "ML", "L"]).astype("string").fillna("NA")


def course_keys(df: pd.DataFrame) -> pd.DataFrame:
    """c2 = 面|距離帯、c1 = 場|面|距離帯、c0 = 場|面|距離|内外（内外は course_type。空は '-'）。"""
    v = df["venue_code"].astype("string").fillna("NA")
    s = df["surface"].astype("string").fillna("NA")
    d = pd.to_numeric(df["distance"], errors="coerce").fillna(0).astype(int).astype(str)
    ct = df["course_type"].astype("string").fillna("").replace("", "-") if "course_type" in df.columns else pd.Series("-", index=df.index)
    band = dist_band(df["distance"])
    return pd.DataFrame({"c2": s + "|" + band, "c1": v + "|" + s + "|" + band, "c0": v + "|" + s + "|" + d + "|" + ct}, index=df.index)


# ───────────────────────── 属性（結果に依存しないビンと ID）
def _bins(x: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(pd.to_numeric(x, errors="coerce"), edges, labels=labels).astype("string").fillna("NA")


def feature_values(df: pd.DataFrame) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    out["sire"] = df["sire"].astype("string").fillna("").replace("", "NA")
    out["dam_sire"] = df["dam_sire"].astype("string").fillna("").replace("", "NA")
    out["jockey"] = df["jockey_id"].astype("string").fillna("NA")
    out["trainer"] = df["trainer_id"].astype("string").fillna("NA")
    st = pd.to_numeric(df["a_style"], errors="coerce")
    qs = st.quantile([1 / 3, 2 / 3]).to_numpy()          # 結果に依存しない分位で 3 分割する
    out["style"] = pd.cut(st, [-1e9, qs[0], qs[1], 1e9], labels=["s1", "s2", "s3"], duplicates="drop").astype("string").fillna("NA")
    fs = pd.to_numeric(df["field_size"], errors="coerce").clip(lower=1)
    hn = pd.to_numeric(df["horse_number"], errors="coerce")
    pos = (hn - 0.5) / fs
    out["post"] = pd.cut(pos, [-1, 0.34, 0.67, 2], labels=["in", "mid", "out"]).astype("string").fillna("NA")
    out["style_post"] = out["style"] + "|" + out["post"]
    sa = df["sex_age"].astype("string").fillna("")
    out["sex"] = sa.str[:1].replace("", "NA")
    out["age"] = _bins(sa.str[1:], [0, 3, 4, 5, 99], ["a3", "a4", "a5", "a6p"])
    jw = pd.to_numeric(df["jockey_weight"], errors="coerce")
    rel = jw - jw.groupby(df["race_id"]).transform("mean")
    out["carried"] = _bins(rel, [-99, -1.0, 1.0, 99], ["light", "avg", "heavy"])
    out["bodywt"] = _bins(df["weight"], [0, 440, 480, 520, 999], ["w1", "w2", "w3", "w4"])
    out["rest"] = _bins(df["d_rest"], [-1, 14, 35, 70, 9999], ["r1", "r2", "r3", "r4"])
    cs = pd.to_numeric(df["c_cls_step"], errors="coerce")
    out["clsstep"] = pd.cut(cs, [-1e9, -0.01, 0.01, 1e9], labels=["down", "same", "up"]).astype("string").fillna("NA")
    return out


# ───────────────────────── as-of 集計
def date_ordinal(date: pd.Series) -> np.ndarray:
    return pd.to_datetime(date).to_numpy("datetime64[D]").astype(np.int64)


def asof_sn(key: pd.Series | np.ndarray, d: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """行ごとに、同じキーの『日付が厳密に前』の y の合計と件数。"""
    codes = pd.factorize(pd.Series(key).astype("string").fillna("NA"))[0]
    t = pd.DataFrame({"k": codes, "d": d, "y": y})
    g = t.groupby(["k", "d"], sort=True)["y"].agg(["sum", "count"]).reset_index()
    g["cs"] = g.groupby("k")["sum"].cumsum() - g["sum"]
    g["cn"] = g.groupby("k")["count"].cumsum() - g["count"]
    m = t[["k", "d"]].merge(g[["k", "d", "cs", "cn"]], on=["k", "d"], how="left")
    return m["cs"].to_numpy(dtype="float64"), m["cn"].to_numpy(dtype="float64")


def _shrink(S: np.ndarray, N: np.ndarray, parent: np.ndarray, k: float) -> np.ndarray:
    return (S + k * parent) / (N + k)


def rank_residual(df: pd.DataFrame) -> np.ndarray:
    fin = pd.to_numeric(df["finish_position"], errors="coerce").to_numpy(dtype="float64")
    n = pd.to_numeric(df["field_size"], errors="coerce").to_numpy(dtype="float64")
    y = 0.5 - (fin - 1.0) / np.maximum(n - 1.0, 1.0)
    y[~np.isfinite(y)] = 0.0
    return y


def course_lift(df: pd.DataFrame, fvals: dict[str, pd.Series], names: list[str], y: np.ndarray | None = None, prefix: str = "tp31_") -> pd.DataFrame:
    """names の各属性について lift_c0（厳密コース）・lift_c1（場×面×距離帯）と、厳密コースのセルの件数 n_c0 を返す。"""
    y = rank_residual(df) if y is None else y
    d = date_ordinal(df["date"])
    ck = course_keys(df)
    S0, N0 = asof_sn(np.zeros(len(df), dtype=int), d, y)
    mu = np.where(N0 > 0, S0 / np.maximum(N0, 1), 0.0)
    out: dict[str, np.ndarray] = {}
    for nm in names:
        f = fvals[nm]
        Sf, Nf = asof_sn(f, d, y)
        est_f = _shrink(Sf, Nf, mu, K_GLOBAL)
        est = est_f
        ests = {}
        for lvl, k in zip(("c2", "c1", "c0"), K_LEVEL):
            S, N = asof_sn(ck[lvl] + "#" + f, d, y)
            est = _shrink(S, N, est, k)
            ests[lvl] = est
            if lvl == "c0":
                n_c0 = N
        out[f"{prefix}{nm}_lift0"] = (ests["c0"] - est_f).astype("float32")
        out[f"{prefix}{nm}_lift1"] = (ests["c1"] - est_f).astype("float32")
        out[f"{prefix}{nm}_n0"] = np.log1p(n_c0).astype("float32")
    return pd.DataFrame(out, index=df.index)


# ───────────────────────── 群
GROUPS: dict[str, list[str]] = {
    "T1_sire": ["sire", "dam_sire"],
    "T2_conn": ["jockey", "trainer"],
    "T3_style_post": ["style", "post", "style_post"],
    "T4_body": ["sex", "age", "carried", "bodywt"],
    "T5_rest_cls": ["rest", "clsstep"],
}


def build_group(df: pd.DataFrame, group: str, fvals: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    fvals = fvals or feature_values(df)
    return course_lift(df, fvals, GROUPS[group])


def build_aggregate(df: pd.DataFrame, parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """T6: 属性の lift0 の合計（ID を除く）と、信頼度で重みづけした平均。"""
    cols0 = [c for g in ("T3_style_post", "T4_body", "T5_rest_cls") for c in parts[g].columns if c.endswith("_lift0")]
    L = pd.concat([parts[g][[c for c in parts[g].columns if c.endswith("_lift0")]] for g in ("T3_style_post", "T4_body", "T5_rest_cls")], axis=1)
    Nn = pd.concat([parts[g][[c for c in parts[g].columns if c.endswith("_n0")]] for g in ("T3_style_post", "T4_body", "T5_rest_cls")], axis=1)
    w = np.expm1(Nn.to_numpy())
    wl = (L.to_numpy() * w).sum(axis=1) / np.maximum(w.sum(axis=1), 1.0)
    return pd.DataFrame({"tp31_lift_sum": L.sum(axis=1).astype("float32"), "tp31_lift_wmean": wl.astype("float32")}, index=df.index)


def build_all(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    fv = feature_values(df)
    parts = {g: build_group(df, g, fv) for g in GROUPS}
    parts["T6_agg"] = build_aggregate(df, parts)
    return parts
