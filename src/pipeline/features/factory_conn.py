"""特徴量ファクトリー G2: 関係者（騎手・調教師・馬主・生産者・産地）と、その組み合わせの as-of 特徴量。

どの列も「そのレースの日より厳密に前」の結果だけで作る（同日のレースは使わない。例外は same_day_* で、発走が 90 分以上前の同日レースのみ）。
経験ベイズ縮約: 条件別の率 = (S + k × 親の率) / (n + k)。親は関係者全体の率（それ自体も全体平均へ縮約）。
入力 h は ``load_history`` の形（race_result_flat 由来 + 馬のプロファイル）。戻り値は h と同じ並びの特徴量表。
市場オッズ・人気は一切使わない。騎手・調教師・馬は ID、馬主・生産者・産地は名前（ID が無いため。正規化して使う）。
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

TABLES = Path("data/page_reference/tables")
PROFILE = Path("data/local/modeling/training_pipeline/factory/conn/horse_profile.parquet")
HIST_COLS = ["race_id", "date", "venue_code", "surface", "distance", "grade", "race_class", "track_condition", "start_time", "field_size", "round",
             "finish_position", "bracket_number", "horse_number", "horse_id", "jockey_id", "trainer_id", "jockey_weight", "last_3f", "passing_order",
             "weight", "sex_age"]
K_ENT, K_COND, K_WIN = 20.0, 20.0, 8.0
SAME_DAY_GAP_MIN = 90


# ───────────────────────────── 入力

def _norm_name(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace(r"[\s　]+", "", regex=True).str.strip().replace("", pd.NA)


def load_history(tables_dir: str | Path | None = None, years: list[int] | None = None, profile: pd.DataFrame | str | Path | None = None) -> pd.DataFrame:
    base = Path(tables_dir) if tables_dir else TABLES
    ys = years or sorted(int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit())
    parts = []
    for y in ys:
        f = base / str(y) / "race_result_flat.parquet"
        if f.exists():
            sch = pl.read_parquet_schema(f)
            parts.append(pl.read_parquet(f, columns=[c for c in HIST_COLS if c in sch]))
    h = pl.concat(parts, how="diagonal_relaxed").to_pandas()
    h["race_id"] = h["race_id"].astype(str)
    h["horse_id"] = h["horse_id"].astype(str)
    if profile is None:
        profile = PROFILE if PROFILE.exists() else None
    if profile is not None:
        p = pd.read_parquet(profile) if isinstance(profile, (str, Path)) else profile
        p = p.drop_duplicates("horse_id").assign(horse_id=lambda d: d["horse_id"].astype(str))
        for c in ("owner", "breeder", "birthplace"):
            if c in p.columns:
                p[c] = _norm_name(p[c])
        h = h.merge(p[[c for c in ("horse_id", "owner", "breeder", "birthplace") if c in p.columns]], on="horse_id", how="left")
    return h


def prepare(h: pd.DataFrame) -> pd.DataFrame:
    """レース内の標準化した結果値と、条件の区分を足す。行の並びと index は変えない。"""
    h = h.copy()
    h["_i"] = np.arange(len(h))
    h["date"] = pd.to_datetime(h["date"])
    fin = pd.to_numeric(h["finish_position"], errors="coerce")
    h["ok"] = fin > 0
    fin = fin.where(h["ok"])
    n_fin = fin.groupby(h["race_id"]).transform("count")
    h["win"] = (fin == 1).astype(float).where(h["ok"])
    h["top3"] = (fin <= 3).astype(float).where(h["ok"])
    h["res"] = (0.5 - (fin - 1.0) / np.maximum(n_fin - 1.0, 1.0)).where(h["ok"])
    l3 = pd.to_numeric(h["last_3f"], errors="coerce")
    l3 = l3.where((l3 >= 30) & (l3 <= 50))
    h["l3dev"] = (l3 - l3.groupby(h["race_id"]).transform("mean")).where(h["ok"])
    first = pd.to_numeric(h["passing_order"].astype(str).str.extract(r"^(\d+)")[0], errors="coerce")
    h["pos"] = (first / np.maximum(n_fin, 1.0)).where(h["ok"] & (first > 0) & (first <= n_fin))
    d = pd.to_numeric(h["distance"], errors="coerce")
    db = pd.cut(d, [0, 1300, 1500, 1900, 2400, 99999], labels=False).astype("float")
    sf = h["surface"].astype(str)
    h["c_venue"] = h["venue_code"].astype(str)
    h["c_surf"] = sf
    h["c_sdb"] = sf + db.astype("Int64").astype(str)
    h["c_go"] = sf + np.where(h["track_condition"].astype(str).isin(["良"]), "g", "h")
    cls = h["race_class"].astype(str) + "|" + h["grade"].astype(str)
    h["c_cls"] = np.select([cls.str.contains("新馬"), cls.str.contains("未勝利"), cls.str.contains("1勝|１勝"), cls.str.contains("2勝|２勝"), cls.str.contains("3勝|３勝"),
                            cls.str.contains("G1|GI|Ｇ１", regex=True), cls.str.contains("G2|GII|Ｇ２|G3|GIII|Ｇ３", regex=True)],
                           ["c0", "c1", "c2", "c3", "c4", "g1", "g23"], "op")
    fs = pd.to_numeric(h["field_size"], errors="coerce")
    h["c_fld"] = pd.cut(fs, [0, 9, 13, 16, 99], labels=False).astype("Int64").astype(str)
    br = pd.to_numeric(h["bracket_number"], errors="coerce")
    h["c_brk"] = np.select([br <= 2, br <= 6, br >= 7], ["in", "mid", "out"], "na")
    h["c_vsd"] = h["c_venue"] + h["c_sdb"]
    h = h.sort_values(["horse_id", "date", "race_id"], kind="stable")
    prev_date = h.groupby("horse_id")["date"].shift(1)
    lay = (h["date"] - prev_date).dt.days
    h["lay_days"] = lay
    h["run_no"] = h.groupby("horse_id").cumcount()
    h["c_lay"] = np.select([lay.isna(), lay <= 14, lay <= 35, lay <= 90], ["first", "s", "m", "l"], "xl")
    h["c_runno"] = np.select([h["run_no"] == 0, h["run_no"] == 1, h["run_no"] <= 4], ["r0", "r1", "r24"], "r5")
    sa = h["sex_age"].astype(str)
    age = pd.to_numeric(sa.str[1:], errors="coerce")
    h["c_sexage"] = sa.str[:1] + np.where(age <= 2, "2", np.where(age == 3, "3", "4"))
    h["jw"] = pd.to_numeric(h["jockey_weight"], errors="coerce")
    h = h.sort_values("_i")
    return h


# ───────────────────────────── as-of の集計の核

def _asof_sums(h: pl.DataFrame, kcol: str, vcols: list[str], window: int | None = None, strict_days: int = 0) -> pl.DataFrame:
    """kcol ごとに、行の日付より厳密に前（window 日以内）の vcols の合計 S_v と件数 n_v（欠損を除く）を返す。_i で結合する。"""
    src = h.filter(pl.col(kcol).is_not_null() & pl.col("ok")).select([kcol, "date"] + vcols)
    aggs = []
    for v in vcols:
        aggs += [pl.col(v).sum().alias(f"S_{v}"), pl.col(v).count().alias(f"n_{v}")]
    daily = src.group_by([kcol, "date"]).agg(aggs).sort([kcol, "date"])
    cum = daily.with_columns([pl.col(c).cast(pl.Float64).cum_sum().over(kcol) for c in daily.columns if c[:2] in ("S_", "n_")]).sort("date")
    left = h.filter(pl.col(kcol).is_not_null()).select(["_i", kcol, "date"]).sort("date")
    cols = [c for c in cum.columns if c[:2] in ("S_", "n_")]
    cur = left.join_asof(cum, on="date", by=kcol, strategy="backward", allow_exact_matches=False)
    if window is None:
        return cur.select(["_i"] + cols).sort("_i")
    left2 = left.with_columns((pl.col("date") - pl.duration(days=window)).alias("date"))
    prev = left2.join_asof(cum, on="date", by=kcol, strategy="backward", allow_exact_matches=False)
    cur = cur.sort("_i")
    prev = prev.sort("_i")
    out = cur.select("_i").with_columns([(cur[c].fill_null(0.0) - prev[c].fill_null(0.0)).alias(c) for c in cols])
    return out


def _full(sums: pl.DataFrame, n_rows: int, vcols: list[str]) -> dict[str, np.ndarray]:
    """_i で並べ直して全行の配列にする（キーが無い行は 0）。"""
    out = {}
    idx = sums["_i"].to_numpy()
    for v in vcols:
        for p in ("S_", "n_"):
            a = np.zeros(n_rows)
            a[idx] = sums[p + v].fill_null(0.0).to_numpy().astype(float)
            out[p + v] = a
    return out


def _global_prior(h: pl.DataFrame, vcols: list[str]) -> dict[str, np.ndarray]:
    g = h.with_columns(pl.lit("ALL").alias("_all"))
    s = _full(_asof_sums(g, "_all", vcols), g.height, vcols)
    return {v: np.where(s["n_" + v] > 0, s["S_" + v] / np.maximum(s["n_" + v], 1), 0.0) for v in vcols}


def _key(h: pd.DataFrame, parts: list[str]) -> pd.Series:
    if len(parts) == 1:
        return h[parts[0]].astype("string")
    s = h[parts[0]].astype("string")
    for p in parts[1:]:
        s = s + "|" + h[p].astype("string")
    return s


ENTITIES = {
    "j": ["jockey_id"], "t": ["trainer_id"], "o": ["owner"], "b": ["breeder"], "bp": ["birthplace"],
    "jt": ["jockey_id", "trainer_id"], "jh": ["jockey_id", "horse_id"], "to": ["trainer_id", "owner"], "tb": ["trainer_id", "breeder"],
    "ob": ["owner", "breeder"], "jo": ["jockey_id", "owner"], "jb": ["jockey_id", "breeder"],
}
CONDS = ["venue", "surf", "sdb", "go", "cls", "fld", "brk", "vsd", "lay", "runno", "sexage"]
VALS_BASE = ["win", "top3", "res", "l3dev", "pos"]
VALS_COND = ["win", "top3", "res"]


class Engine:
    """polars の表を 1 つ持ち、キー列の追加と集計を行う。"""

    def __init__(self, h: pd.DataFrame):
        self.h = h
        keep = ["_i", "date", "ok", "win", "top3", "res", "l3dev", "pos"]
        self.pl = pl.from_pandas(h[keep].assign(date=h["date"].dt.date))
        self.n = len(h)
        self.G = _global_prior(self.pl, VALS_BASE)
        self._cols: set[str] = set()

    def add_key(self, name: str, series: pd.Series) -> None:
        if name in self._cols:
            return
        self.pl = self.pl.with_columns(pl.Series(name, series.astype("object").where(series.notna(), None).tolist(), dtype=pl.Utf8))
        self._cols.add(name)

    def rates(self, kcol: str, vcols: list[str], prior: dict[str, np.ndarray], k: float, window: int | None = None) -> tuple[dict[str, np.ndarray], np.ndarray]:
        s = _full(_asof_sums(self.pl, kcol, vcols, window), self.n, vcols)
        out = {}
        for v in vcols:
            out[v] = (s["S_" + v] + k * prior[v]) / (s["n_" + v] + k)
        return out, s["n_" + vcols[0]]


def entity_features(eng: Engine, ent: str, conds: list[str], vals_base=VALS_BASE, vals_cond=VALS_COND) -> dict[str, np.ndarray]:
    h = eng.h
    parts = ENTITIES[ent]
    if any(p not in h.columns for p in parts):
        return {}
    ek = f"k_{ent}"
    eng.add_key(ek, _key(h, parts).where(h[parts].notna().all(axis=1)))
    base, n0 = eng.rates(ek, vals_base, eng.G, K_ENT)
    out = {f"{ent}_{v}": base[v].astype("float32") for v in vals_base}
    out[f"{ent}_n"] = np.log1p(n0).astype("float32")
    for c in conds:
        ck = f"k_{ent}_{c}"
        s = _key(h, parts + ["c_" + c]).where(h[parts].notna().all(axis=1))
        eng.add_key(ck, s)
        r, n = eng.rates(ck, vals_cond, {v: base[v] for v in vals_cond}, K_COND)
        for v in vals_cond:
            out[f"{ent}_{c}_{v}"] = r[v].astype("float32")
        out[f"{ent}_{c}_n"] = np.log1p(n).astype("float32")
    return out


def window_features(eng: Engine, ent: str, windows=(7, 14, 30, 90, 365)) -> dict[str, np.ndarray]:
    ek = f"k_{ent}"
    if ek not in eng._cols:
        eng.add_key(ek, _key(eng.h, ENTITIES[ent]).where(eng.h[ENTITIES[ent]].notna().all(axis=1)))
    base, _ = eng.rates(ek, VALS_COND, eng.G, K_ENT)
    out = {}
    for w in windows:
        r, n = eng.rates(ek, VALS_COND, base, K_WIN, window=w)
        for v in VALS_COND:
            out[f"{ent}_w{w}_{v}"] = r[v].astype("float32")
        out[f"{ent}_w{w}_n"] = n.astype("float32")
        if w in (14, 90):
            out[f"{ent}_w{w}_form_gap_res"] = (r["res"] - base["res"]).astype("float32")
    return out


def streak_features(eng: Engine, ent: str) -> dict[str, np.ndarray]:
    """最後の勝利・最後の出走からの日数と、最後の勝利からの出走数（as-of）。"""
    h = eng.h
    ek = f"k_{ent}"
    if ek not in eng._cols:
        eng.add_key(ek, _key(h, ENTITIES[ent]).where(h[ENTITIES[ent]].notna().all(axis=1)))
    p = eng.pl.filter(pl.col(ek).is_not_null() & pl.col("ok")).select([ek, "date", "win"])
    daily = p.group_by([ek, "date"]).agg(pl.col("win").sum().alias("w"), pl.len().alias("n")).sort([ek, "date"])
    daily = daily.with_columns(pl.col("n").cum_sum().over(ek).alias("cn"))
    daily = daily.with_columns(
        pl.when(pl.col("w") > 0).then(pl.col("date")).otherwise(None).alias("wd"),
        pl.when(pl.col("w") > 0).then(pl.col("cn")).otherwise(None).alias("cn_w"),
    ).with_columns(pl.col("wd").cast(pl.Date).forward_fill().over(ek).alias("last_win_date"), pl.col("cn_w").forward_fill().over(ek).alias("cn_at_win"))
    daily = daily.with_columns(pl.col("date").alias("last_ride_date")).sort("date")
    left = eng.pl.filter(pl.col(ek).is_not_null()).select(["_i", ek, "date"]).sort("date")
    j = left.join_asof(daily.select([ek, "date", "last_win_date", "cn_at_win", "cn", "last_ride_date"]), on="date", by=ek, strategy="backward", allow_exact_matches=False).sort("_i")
    idx = j["_i"].to_numpy()
    dts = j["date"].cast(pl.Date)
    a_win = np.full(eng.n, np.nan)
    a_ride = np.full(eng.n, np.nan)
    a_since = np.full(eng.n, np.nan)
    a_win[idx] = (dts - j["last_win_date"]).dt.total_days().cast(pl.Float64).to_numpy()
    a_ride[idx] = (dts - j["last_ride_date"]).dt.total_days().cast(pl.Float64).to_numpy()
    a_since[idx] = (j["cn"].cast(pl.Float64) - j["cn_at_win"].cast(pl.Float64)).to_numpy()
    return {f"{ent}_days_since_win": np.minimum(a_win, 730).astype("float32"), f"{ent}_days_since_ride": np.minimum(a_ride, 365).astype("float32"),
            f"{ent}_rides_since_win": a_since.astype("float32")}


def scale_features(eng: Engine, ent: str) -> dict[str, np.ndarray]:
    """関係者が抱える頭数（これまでに走らせた馬の数。as-of）。"""
    h = eng.h
    parts = ENTITIES[ent]
    ek = f"k_{ent}"
    eng.add_key(ek, _key(h, parts).where(h[parts].notna().all(axis=1)))
    srt = h.assign(_k=_key(h, parts)).sort_values(["date", "_i"], kind="stable")
    first = ~srt.duplicated(["_k", "horse_id"])
    u = pd.Series(first.astype(float).to_numpy(), index=srt["_i"].to_numpy()).reindex(h["_i"].to_numpy()).to_numpy()
    e = eng.pl.with_columns(pl.Series("uniq", u), pl.lit(True).alias("ok"))
    s = _full(_asof_sums(e, ek, ["uniq"]), eng.n, ["uniq"])
    return {f"{ent}_n_horses": np.log1p(s["S_uniq"]).astype("float32")}


# ───────────────────────────── 同日（発走が 90 分以上前のレースの結果だけ）

def _minutes(st: pd.Series) -> np.ndarray:
    m = st.astype(str).str.extract(r"(\d{1,2}):(\d{2})")
    return (pd.to_numeric(m[0], errors="coerce") * 60 + pd.to_numeric(m[1], errors="coerce")).to_numpy(dtype=float)


def same_day_features(h: pd.DataFrame) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    t = _minutes(h["start_time"])
    d = h["date"].dt.strftime("%Y%m%d")
    for ent, col in (("j", "jockey_id"), ("t", "trainer_id")):
        key = h[col].astype("string")
        g = pd.DataFrame({"_i": h["_i"].to_numpy(), "k": key.to_numpy(), "d": d.to_numpy(), "t": t, "race": h["race_id"].to_numpy(),
                          "win": h["win"].to_numpy(), "res": h["res"].to_numpy(), "ok": h["ok"].to_numpy()})
        g = g[g["k"].notna()]
        cnt = g.groupby(["k", "d"])["race"].transform("size")
        out[f"{ent}_rides_today"] = pd.Series(cnt.to_numpy(), index=g["_i"].to_numpy()).reindex(h["_i"].to_numpy()).to_numpy().astype("float32")
        if ent == "t":
            nr = g.groupby(["k", "race"])["race"].transform("size")
            out["t_runners_in_race"] = pd.Series(nr.to_numpy(), index=g["_i"].to_numpy()).reindex(h["_i"].to_numpy()).to_numpy().astype("float32")
        m = g.merge(g, on=["k", "d"], suffixes=("", "_o"))
        m = m[(m["t_o"] <= m["t"] - SAME_DAY_GAP_MIN) & m["ok_o"].astype(bool)]
        a = m.groupby("_i").agg(n=("res_o", "size"), w=("win_o", "sum"), r=("res_o", "mean"))
        idx = h["_i"].to_numpy()
        n = a["n"].reindex(idx).fillna(0).to_numpy()
        out[f"{ent}_sd_rides_before"] = n.astype("float32")
        out[f"{ent}_sd_wins_before"] = a["w"].reindex(idx).fillna(0).to_numpy().astype("float32")
        out[f"{ent}_sd_res_before"] = a["r"].reindex(idx).to_numpy().astype("float32")
    return out


# ───────────────────────────── 乗り替わり・転厩・遠征・斤量

def change_features(eng: Engine, base_rates: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    h = eng.h
    s = h.sort_values(["horse_id", "date", "race_id"], kind="stable")
    g = s.groupby("horse_id")
    out: dict[str, np.ndarray] = {}
    pj, pt, pv, pw = g["jockey_id"].shift(1), g["trainer_id"].shift(1), g["venue_code"].shift(1), g["jw"].shift(1)
    chg_j = (s["jockey_id"] != pj) & pj.notna()
    chg_t = (s["trainer_id"] != pt) & pt.notna()
    ser = lambda x: pd.Series(np.asarray(x, dtype=float), index=s["_i"].to_numpy()).reindex(h["_i"].to_numpy()).to_numpy().astype("float32")
    out["chg_jockey"] = ser(chg_j.astype(float).where(pj.notna()))
    out["chg_trainer"] = ser(chg_t.astype(float).where(pt.notna()))
    out["chg_venue"] = ser((s["venue_code"] != pv).astype(float).where(pv.notna()))
    out["jw_diff_prev"] = ser(s["jw"] - pw)
    out["jw_rel_field"] = (h["jw"] - h.groupby("race_id")["jw"].transform("mean")).to_numpy().astype("float32")
    # 前走の騎手と今回の騎手の、as-of の総合力の差（乗り替わりの上げ下げ）
    for v in ("res", "win"):
        cur = base_rates["j"][v]
        prev_key = pd.Series(pj.to_numpy(), index=s["_i"].to_numpy()).reindex(h["_i"].to_numpy())
        eng.add_key("k_prevj", prev_key.astype("string"))
        r, _ = eng.rates("k_prevj", [v], eng.G, K_ENT)
        out[f"jockey_upgrade_{v}"] = np.where(out["chg_jockey"] == 1, cur - r[v], 0.0).astype("float32")
    # 調教師の遠征: 厩舎が今回の場所で走らせた割合（as-of）
    ek_tv = "k_tv"
    eng.add_key(ek_tv, _key(h, ["trainer_id", "venue_code"]).where(h[["trainer_id", "venue_code"]].notna().all(axis=1)))
    eng.add_key("k_tr", h["trainer_id"].astype("string"))
    s1 = eng.pl.with_columns(pl.lit(1.0).alias("one"))
    a = _full(_asof_sums(s1, ek_tv, ["one"]), eng.n, ["one"])["S_one"]
    b = _full(_asof_sums(s1, "k_tr", ["one"]), eng.n, ["one"])["S_one"]
    share = np.where(b > 0, a / np.maximum(b, 1), np.nan)
    out["t_venue_share"] = share.astype("float32")
    out["t_away"] = np.where(b >= 20, (share < 0.1).astype(float), np.nan).astype("float32")
    out["lay_days"] = h["lay_days"].to_numpy().astype("float32")
    out["run_no"] = h["run_no"].to_numpy().astype("float32")
    return out


# ───────────────────────────── 群の組み立て

def build_all(h: pd.DataFrame, entities=None, conds=None) -> dict[str, dict[str, np.ndarray]]:
    """群名 → {列名: 配列}。h は ``prepare`` を通したもの。"""
    eng = Engine(h)
    groups: dict[str, dict[str, np.ndarray]] = {}
    base_rates: dict[str, dict[str, np.ndarray]] = {}
    for ent in (entities or list(ENTITIES)):
        f = entity_features(eng, ent, conds or CONDS)
        if not f:
            continue
        groups[f"ent_{ent}"] = f
        base_rates[ent] = {v: f[f"{ent}_{v}"] for v in VALS_BASE}
    for ent in ("j", "t", "o", "b"):
        if ent in base_rates:
            groups[f"win_{ent}"] = window_features(eng, ent)
    for ent in ("j", "t", "o"):
        if ent in base_rates:
            groups[f"streak_{ent}"] = streak_features(eng, ent)
    for ent in ("o", "b", "t"):
        if ent in base_rates:
            groups[f"scale_{ent}"] = scale_features(eng, ent)
    groups["same_day"] = same_day_features(h)
    if "j" in base_rates:
        groups["change"] = change_features(eng, base_rates)
    return groups
