"""tp-25a: スタート(テン1F)・位置取りの前段モデルの入力作り込み（as-of）と、前段の予測から win モデルへ足す列。

先読み防止:
- SmartRC の fullresults（career）は rdate < 当該レース日 の行だけを特徴量に使う。当該レース自身のテン/コーナーは目的変数にだけ使う。
- 騎手・調教師・コースの傾向は、それ以前の日のレースの目的変数だけを集計する（同日は使わない。``asof_group_mean``）。
キーは race_id / horse_id / jockey_id / trainer_id。市場オッズ・人気系の列は使わない。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.models.stage_models import asof_group_mean

ROOT = Path(__file__).resolve().parents[3]
ROUTES = ROOT / "data/meta/course_master/geo/routes.json"
PREFIX = "t25_"
FRONT, MID_END = 0.25, 0.60          # 最初のコーナー位置（0=先頭〜1=最後尾）の 3 区分の境目


# ───────────────────────── コース形状（スタートから最初のコーナーまで）
def course_start_table(path: Path = ROUTES) -> pd.DataFrame:
    """(venue_code, surface, distance) ごとの、最初のコーナーまでの距離・発走位置の種類。内回り/外回りが両方ある距離は平均（n_variants で区別）。"""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for vc, v in d["venues"].items():
        for key, c in v["courses"].items():
            face, _inout, dist = key.split("|")
            fc = c.get("first_curve_master") or c.get("first_curve_calc")
            rows.append({"venue_code": vc, "surface": face, "distance": int(dist), "first_curve_m": fc,
                         "start_chute": float(c.get("start") == "chute"), "start_pocket": float(c.get("start") == "pocket"),
                         "stub_len": float(c.get("stub_len") or 0.0), "laps_before": float(c.get("laps_before") or 0)})
    t = pd.DataFrame(rows)
    g = t.groupby(["venue_code", "surface", "distance"], as_index=False).agg(
        first_curve_m=("first_curve_m", "mean"), first_curve_min=("first_curve_m", "min"), start_chute=("start_chute", "mean"),
        start_pocket=("start_pocket", "mean"), stub_len=("stub_len", "mean"), laps_before=("laps_before", "mean"), n_variants=("first_curve_m", "size"))
    return g


def add_course_start(df: pd.DataFrame) -> pd.DataFrame:
    t = course_start_table().rename(columns={c: PREFIX + c for c in ("first_curve_m", "first_curve_min", "start_chute", "start_pocket", "stub_len", "laps_before", "n_variants")})
    k = pd.DataFrame({"venue_code": df["venue_code"].astype(str).str.zfill(2).to_numpy(), "surface": df["surface"].astype(str).to_numpy(),
                      "distance": pd.to_numeric(df["distance"], errors="coerce").astype("Int64").to_numpy()})
    m = k.merge(t, on=["venue_code", "surface", "distance"], how="left")
    return pd.DataFrame({c: m[c].to_numpy() for c in t.columns if c.startswith(PREFIX)}, index=df.index)


# ───────────────────────── 馬の過去のテン・位置取り（as-of）
def _past_rel(career: pd.DataFrame, min_n: int = 6) -> pd.DataFrame:
    """過去走のテンを、その走のレース内平均との差にする（fullresults に載る全馬から rcode 単位で平均。min_n 頭未満は欠損）。"""
    c = career.copy()
    g = c.groupby("rcode")["ten"]
    mean, cnt = g.transform("mean"), g.transform("count")
    c["ten_rel"] = (c["ten"] - mean).where(cnt >= min_n)
    return c


def horse_history_features(entries: pd.DataFrame, career: pd.DataFrame, decay: float = 0.6) -> pd.DataFrame:
    """entries: race_id, horse_id, date, distance, surface。rdate < date の戦績だけを使う。"""
    e = entries[["race_id", "horse_id", "date", "distance", "surface"]].copy()
    e["_d"] = pd.to_datetime(e["date"]).dt.strftime("%Y%m%d")
    e["_i"] = np.arange(len(e))
    c = _past_rel(career)[["horse_id", "rdate", "ten", "ten_rel", "c1_rel", "range", "trackkind"]].dropna(subset=["rdate"])
    m = e[["_i", "horse_id", "_d", "distance", "surface"]].merge(c, on="horse_id", how="inner")
    m = m[m["rdate"] < m["_d"]].sort_values(["_i", "rdate"], ascending=[True, False])
    out = pd.DataFrame(index=np.arange(len(e)))

    def ewm(col: str, n_max: int, name: str, with_stats: bool = True):
        v = m[m[col].notna()].copy()
        v["_k"] = v.groupby("_i").cumcount()
        v = v[v["_k"] < n_max]
        v["_w"] = decay ** v["_k"]
        v["_wx"] = v["_w"] * v[col]
        s = v.groupby("_i")[["_w", "_wx"]].sum()
        out[f"{PREFIX}{name}_ewm"] = s["_wx"] / s["_w"]
        out[f"{PREFIX}{name}_n"] = v.groupby("_i").size()
        if with_stats:
            out[f"{PREFIX}{name}_last"] = v[v["_k"] == 0].set_index("_i")[col]
            out[f"{PREFIX}{name}_best"] = v.groupby("_i")[col].min()
            out[f"{PREFIX}{name}_sd"] = v.groupby("_i")[col].std()
        return v

    ewm("ten_rel", 8, "trel")
    ewm("c1_rel", 8, "c1")
    v = m[m["c1_rel"].notna()].copy()
    v["_k"] = v.groupby("_i").cumcount()
    v = v[v["_k"] < 8]
    out[f"{PREFIX}front_rate"] = (v["c1_rel"] <= FRONT).groupby(v["_i"]).mean()
    out[f"{PREFIX}back_rate"] = (v["c1_rel"] >= MID_END).groupby(v["_i"]).mean()
    # 今回と同じ面・距離 ±200m の過去走
    sk = m["surface"].map(lambda s: 0.0 if str(s).startswith("芝") else (1.0 if str(s).startswith("ダ") else np.nan))
    sim = m[(m["trackkind"] == sk) & ((m["range"] - m["distance"]).abs() <= 200)]
    out[f"{PREFIX}trel_sim"] = sim[sim["ten_rel"].notna()].groupby("_i")["ten_rel"].mean()
    out[f"{PREFIX}c1_sim"] = sim[sim["c1_rel"].notna()].groupby("_i")["c1_rel"].mean()
    # 距離が延びたか縮んだか（前走との差）。テンは短い距離ほど速い
    last = m.groupby("_i").first()
    out[f"{PREFIX}dist_delta"] = pd.Series(e["distance"].to_numpy()[last.index.to_numpy()] - last["range"].to_numpy(), index=last.index)
    out = out.reindex(np.arange(len(e)))
    out.index = entries.index
    return out


# ───────────────────────── 騎手・調教師・コース・ゲートの傾向（as-of）
def tendency_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: RangeIndex、date, jockey_id, trainer_id, venue_code, surface, distance, horse_number, field_size, ten_rel_actual, c1_rel_actual。
    目的変数（実際のテン/位置）の過去の平均を、経験ベイズで縮約して付ける。同日の行は使わない。"""
    assert isinstance(df.index, pd.RangeIndex)
    d = df[["date", "jockey_id", "trainer_id", "venue_code", "surface", "distance", "horse_number", "field_size", "ten_rel_actual", "c1_rel_actual"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    course = d["venue_code"].astype(str) + "|" + d["surface"].astype(str) + "|" + d["distance"].astype(str)
    gate = (pd.to_numeric(d["horse_number"], errors="coerce") - 1) / (pd.to_numeric(d["field_size"], errors="coerce") - 1).clip(lower=1)
    gb = pd.cut(gate, [-0.01, 0.2, 0.5, 0.8, 1.01], labels=False)
    d["_course"] = course
    d["_draw"] = course + "|" + gb.astype("string").fillna("x")
    d["_jk_course"] = d["jockey_id"].astype(str) + "|" + d["surface"].astype(str)
    out = {}
    for key, nm, k in (("jockey_id", "jk", 30.0), ("trainer_id", "tr", 30.0), ("_course", "crs", 50.0), ("_draw", "draw", 50.0), ("_jk_course", "jks", 30.0)):
        for tgt, tn in (("ten_rel_actual", "ten"), ("c1_rel_actual", "c1")):
            r = asof_group_mean(d.assign(_key=d[key].astype(str)), "_key", tgt, k=k, name="x")
            out[f"{PREFIX}{nm}_{tn}"] = r["x_mean"].to_numpy()
            if tn == "ten" and nm in ("jk", "tr", "draw"):
                out[f"{PREFIX}{nm}_n"] = r["x_n"].to_numpy()
    o = pd.DataFrame(out, index=df.index)
    o[f"{PREFIX}gate_norm"] = gate.to_numpy()
    o[f"{PREFIX}gate_inside"] = (gate <= 0.2).astype("float32").to_numpy()
    return o


# ───────────────────────── 目的変数
def three_class(c1_rel: np.ndarray) -> np.ndarray:
    y = np.where(np.isnan(c1_rel), -1, np.where(c1_rel <= FRONT, 0, np.where(c1_rel < MID_END, 1, 2)))
    return y.astype(int)


def rank_label(race_id: np.ndarray, ten_rel: np.ndarray, levels: int = 5) -> np.ndarray:
    """レース内でテンが速い順に 4(最速グループ)〜0。テンの無い馬は -1。lambdarank 用。"""
    s = pd.Series(ten_rel)
    pct = s.groupby(race_id).rank(method="average", pct=True, ascending=True)        # 小さい=速い=0 に近い
    lab = (levels - 1 - np.floor(pct.to_numpy() * levels - 1e-9)).clip(0, levels - 1)
    return np.where(np.isnan(ten_rel), -1, lab).astype(int)


# ───────────────────────── win モデルへ足す列（前段の OOF から）
def win_columns(df: pd.DataFrame, oof: pd.DataFrame, prefix: str = "ts_") -> dict[str, pd.DataFrame]:
    """戻り値: {'raw': 前段予測そのもの, 'rel': レース内の相対化(_rz/_rrank/_rgap), 'agg': レース集約}。df と同じ index。"""
    from src.pipeline.features.serving_stage_rel import race_relative

    key = df[["race_id", "horse_id"]].astype(str).reset_index(drop=True)
    o = key.merge(oof.assign(race_id=oof["race_id"].astype(str), horse_id=oof["horse_id"].astype(str)), on=["race_id", "horse_id"], how="left")
    cols = [c for c in oof.columns if c.startswith(prefix)]
    raw = o[cols].astype("float32")
    raw.index = df.index
    work = raw.copy()
    work["race_id"] = df["race_id"].astype(str).to_numpy()
    rel_src = [c for c in (f"{prefix}ten_pred", f"{prefix}c1_pred", f"{prefix}p_front", f"{prefix}ten_rank") if c in cols]
    rel = race_relative(work, rel_src)
    g = work.groupby("race_id")
    agg = pd.DataFrame(index=df.index)
    if f"{prefix}ten_pred" in cols:
        t = work[f"{prefix}ten_pred"]
        agg[f"{prefix}ag_ten_min"] = g[f"{prefix}ten_pred"].transform("min")
        agg[f"{prefix}ag_ten_sd"] = g[f"{prefix}ten_pred"].transform("std")
        agg[f"{prefix}ag_ten_gap"] = t - agg[f"{prefix}ag_ten_min"]
        # 最速のテンの馬が 2 番手を引き離しているか（単騎逃げの代理）
        second = g[f"{prefix}ten_pred"].transform(lambda s: s.nsmallest(2).iloc[-1] if s.notna().sum() >= 2 else np.nan)
        agg[f"{prefix}ag_ten_lead"] = second - agg[f"{prefix}ag_ten_min"]
        agg[f"{prefix}ag_n_fast"] = (t - agg[f"{prefix}ag_ten_min"] <= 0.3).groupby(work["race_id"]).transform("sum").astype("float32")
    if f"{prefix}p_front" in cols:
        p = work[f"{prefix}p_front"]
        agg[f"{prefix}ag_front_sum"] = g[f"{prefix}p_front"].transform("sum")
        agg[f"{prefix}ag_front_others"] = agg[f"{prefix}ag_front_sum"] - p                   # 自分以外の先行馬の見込み数（ペース圧）
        agg[f"{prefix}ag_front_share"] = p / agg[f"{prefix}ag_front_sum"].replace(0, np.nan)
        top1 = g[f"{prefix}p_front"].transform("max")
        top2 = g[f"{prefix}p_front"].transform(lambda s: s.nlargest(2).iloc[-1] if s.notna().sum() >= 2 else np.nan)
        agg[f"{prefix}ag_lone"] = p - np.where(p >= top1, top2, top1)            # 自分が先行馬の筆頭でどれだけ抜けているか（負なら他に強い先行馬がいる）
    if f"{prefix}c1_pred" in cols:
        agg[f"{prefix}ag_c1_mean"] = g[f"{prefix}c1_pred"].transform("mean")
    return {"raw": raw, "rel": rel.astype("float32"), "agg": agg.astype("float32")}
