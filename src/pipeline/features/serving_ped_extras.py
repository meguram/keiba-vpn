"""推論時の ped_* 24 列（血統×条件）と extras 7 列（追い切り評価・厩舎コメント）。

学習側（``pedigree_context.build`` / ``prerace_extras`` + ``tp05_data.load_extras_ped``）と同じ値を、履歴の集計表（状態）と当日の出馬表から作る。

ped_*: 学習側は「同じキーの、日付が厳密に前の行」の平均を全体平均へ経験ベイズ縮約（k=30）。ここでは履歴を (キー, 日付) ごとの累積和・件数にして
持ち、当日の行は「その日より前」の累積を引くだけにする（連結して学習側関数を呼ぶのと同じ値）。履歴 28 万行の集計は数秒、当日の適用はミリ秒。

注意（2026-10-17 時点の事実）: 学習表の sire / dam_sire は全行が空文字（出馬表フラットの血統列が空）で、ped_* は実質「条件別の全体平均」になっている。
学習時と同じ値にするため、既定では履歴・当日ともキーの血統名を空にする（``blank_pedigree=True``）。学習側が血統名入りに直ったら False にして状態を作り直す。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import polars as pl

from src.pipeline.features import pedigree_context as pc
from src.pipeline.features import prerace_extras as pe

K_SHRINK = 30.0
EXTRAS_COLUMNS = ["oikiri_eval", "has_comment", "comment_len", "comment_pos", "comment_neg", "oikiri_eval_rank", "oikiri_eval_gap"]
HIST_COLS = ["date", "race_id", "horse_id", "surface", "distance", "track_condition", "venue_code", "finish_position", "pos_early", "l3f"]
_GOOD = "良"


def ped_specs() -> list[tuple[str, str, str, str]]:
    """(列名, role, 条件名, 値名)。学習側 ``pedigree_context.build`` の列の並びと同じ。"""
    out = []
    for role in ("sire", "dam_sire"):
        for cname in pc.CONDS:
            for vname in pc.VALUES:
                if role == "dam_sire" and vname in ("pe", "l3"):
                    continue
                out.append((f"ped_{role}_{cname}_{vname}", role, cname, vname))
                if vname == "win":
                    out.append((f"ped_{role}_{cname}_n", role, cname, "n"))
    return out


PED_COLUMNS = [s[0] for s in ped_specs()]


def _prep(df: pd.DataFrame, *, blank_pedigree: bool, going_unknown_as_good: bool) -> pd.DataFrame:
    d = pd.DataFrame({
        "date": pd.to_datetime(df["date"]).to_numpy(),
        "race_id": df["race_id"].astype(str).to_numpy(),
        "surface": df["surface"].astype(str).to_numpy(),
        "venue_code": df["venue_code"].astype(str).to_numpy(),
    }, index=df.index)
    tc = df["track_condition"].astype(object)
    if going_unknown_as_good:
        tc = tc.where(tc.notna() & (tc.astype(str) != ""), _GOOD)
    d["track_condition"] = tc.astype(str).to_numpy()
    for role in ("sire", "dam_sire"):
        if blank_pedigree or role not in df.columns:
            d[role] = ""
        else:
            d[role] = df[role].astype(object).where(df[role].notna(), "").astype(str).to_numpy()
    d["dist_band"] = pc.dist_band(df["distance"]).to_numpy()
    return d


def _keys(d: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    return {(role, cname): (d[role].astype(str) + "|" + fn(d)).to_numpy() for role in ("sire", "dam_sire") for cname, fn in pc.CONDS.items()}


def history_values(hist: pd.DataFrame) -> pd.DataFrame:
    """履歴（HIST_COLS）から値の列 is_win / is_top3 / pos_early / l3f_dev を作る。l3f_dev はレース内平均との差（学習側と同じ）。"""
    h = pd.DataFrame({"race_id": hist["race_id"].astype(str).to_numpy()}, index=hist.index)
    fin = hist["finish_position"].astype(float)
    h["is_win"] = (fin == 1).astype(float)
    h["is_top3"] = (fin <= 3).astype(float)
    h["pos_early"] = hist["pos_early"].astype(float)
    l3f = hist["l3f"].astype(float)
    h["l3f_dev"] = l3f - l3f.groupby(h["race_id"]).transform("mean")
    return h


class PedState:
    """履歴の (キー, 日付) ごとの累積和・件数。``features`` で当日の行の ped_* を作る。"""

    def __init__(self, daily: pl.DataFrame, glob: pl.DataFrame, blank_pedigree: bool = True, last_date: str | None = None):
        self.daily = daily            # name, k, date, S, N
        self.glob = glob              # v, date, GS, GN
        self.blank_pedigree = blank_pedigree
        self.last_date = last_date
        self._cache: dict[str, tuple[pl.DataFrame, pl.DataFrame]] = {}

    @classmethod
    def from_history(cls, hist: pd.DataFrame, *, blank_pedigree: bool = True, going_unknown_as_good: bool = False) -> "PedState":
        """hist: HIST_COLS を持つ履歴（結果と pos_early / l3f が入っているもの）。sire / dam_sire 列があれば blank_pedigree=False で使う。"""
        hist = hist.reset_index(drop=True)
        d = _prep(hist, blank_pedigree=blank_pedigree, going_unknown_as_good=going_unknown_as_good)
        vals = history_values(hist)
        keys = _keys(d)
        dates = pl.Series(d["date"].to_numpy()).cast(pl.Date)
        daily_parts, glob_parts = [], []
        vmap = {"win": "is_win", "t3": "is_top3", "pe": "pos_early", "l3": "l3f_dev"}
        for vname, vcol in vmap.items():
            v = pl.Series(vals[vcol].to_numpy().astype("float64"), nan_to_null=True)
            g = (pl.DataFrame({"date": dates, "v": v}).group_by("date").agg(pl.col("v").sum().alias("gs"), pl.col("v").is_not_null().sum().alias("gn"))
                 .sort("date").with_columns(pl.col("gs").cum_sum().alias("GS"), pl.col("gn").cum_sum().alias("GN")).select("date", "GS", "GN"))
            glob_parts.append(g.with_columns(pl.lit(vname).alias("v")))
            for (role, cname), key in keys.items():
                if role == "dam_sire" and vname in ("pe", "l3"):
                    continue
                base = pl.DataFrame({"k": key, "date": dates, "v": v})
                dd = (base.group_by(["k", "date"]).agg(pl.col("v").sum().alias("s"), pl.col("v").is_not_null().sum().alias("n"))
                      .sort("date").with_columns(pl.col("s").cum_sum().over("k").alias("S"), pl.col("n").cum_sum().over("k").alias("N"))
                      .select("k", "date", "S", "N"))
                daily_parts.append(dd.with_columns(pl.lit(f"{role}|{cname}|{vname}").alias("name")))
        daily = pl.concat(daily_parts).select("name", "k", "date", "S", "N")
        glob = pl.concat(glob_parts).select("v", "date", "GS", "GN")
        return cls(daily, glob, blank_pedigree, last_date=str(d["date"].max().date()) if len(d) else None)

    def _tables(self, name: str, vname: str) -> tuple[pl.DataFrame, pl.DataFrame]:
        if name not in self._cache:
            self._cache[name] = (self.daily.filter(pl.col("name") == name).select("k", "date", "S", "N").sort("date"),
                                 self.glob.filter(pl.col("v") == vname).select("date", "GS", "GN").sort("date"))
        return self._cache[name]

    def features(self, target: pd.DataFrame, *, going_unknown_as_good: bool = True) -> pd.DataFrame:
        """target: race_id, date, surface, distance, track_condition, venue_code（+ blank_pedigree=False なら sire, dam_sire）。
        同じ日付以降の履歴は使わない（累積は「その日より前」）。戻り値は PED_COLUMNS の float32、target と同じ index。"""
        d = _prep(target, blank_pedigree=self.blank_pedigree, going_unknown_as_good=going_unknown_as_good)
        keys = _keys(d)
        dates = pl.Series(d["date"].to_numpy()).cast(pl.Date)
        out: dict[str, np.ndarray] = {}
        shr: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
        for col, role, cname, vname in ped_specs():
            real = "win" if vname == "n" else vname
            ck = (role, cname, real)
            if ck not in shr:
                dd, gl = self._tables(f"{role}|{cname}|{real}", real)
                left = pl.DataFrame({"k": keys[(role, cname)], "date": dates, "_i": np.arange(len(d))}).sort("date")
                j = left.join_asof(dd, on="date", by="k", strategy="backward", allow_exact_matches=False).sort("_i")
                g = left.join_asof(gl, on="date", strategy="backward", allow_exact_matches=False).sort("_i")
                gm = (g["GS"] / g["GN"]).to_numpy().astype(float)
                shr[ck] = (j["N"].fill_null(0).to_numpy().astype(float), j["S"].fill_null(0.0).to_numpy().astype(float), gm)
            n, s, gm = shr[ck]
            out[col] = n.astype("float32") if vname == "n" else ((s + K_SHRINK * gm) / (n + K_SHRINK)).astype("float32")
        return pd.DataFrame(out, index=target.index)[PED_COLUMNS]

    def save(self, directory: str | Path) -> None:
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        self.daily.write_parquet(p / "ped_daily.parquet")
        self.glob.write_parquet(p / "ped_global.parquet")
        (p / "ped_meta.txt").write_text(f"blank_pedigree={int(self.blank_pedigree)}\nlast_date={self.last_date}\n", encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "PedState":
        p = Path(directory)
        meta = dict(x.split("=", 1) for x in (p / "ped_meta.txt").read_text(encoding="utf-8").split() if "=" in x)
        return cls(pl.read_parquet(p / "ped_daily.parquet"), pl.read_parquet(p / "ped_global.parquet"), bool(int(meta.get("blank_pedigree", "1"))), meta.get("last_date"))


def history_from_runs(runs: pl.DataFrame) -> pd.DataFrame:
    """``common.load_runs`` → ``apply_repair`` を通した走行データ（pos_early / l3f 入り）から履歴（HIST_COLS）を作る。"""
    cols = [c for c in HIST_COLS if c != "date"] + ["date"]
    h = runs.filter(pl.col("finish_position") > 0).select(cols).to_pandas()
    h["date"] = pd.to_datetime(h["date"])
    return h[HIST_COLS]


# ---------------------------------------------------------------- extras

def fetch_extras_rows(race_ids: Iterable[str], storage: Any, workers: int = 4) -> pd.DataFrame:
    """race_oikiri / race_trainer_comment を読んで馬ごとの行（oikiri_eval・comment_*）にする。無いレース・馬は行が出ない。"""
    return pe.build(list(race_ids), storage, workers=workers)


def extras_features(keys: pd.DataFrame, rows: pd.DataFrame | None) -> pd.DataFrame:
    """keys: race_id, horse_id（当日の出走馬）。rows: ``fetch_extras_rows`` の結果。学習側 ``load_extras_ped`` の後処理と同じ。
    追い切り・コメントが取れていない馬は oikiri_eval 等が NaN、has_comment は 0。"""
    k = keys[["race_id", "horse_id"]].astype(str).reset_index(drop=True)
    if rows is None or len(rows) == 0:
        rows = pd.DataFrame(columns=["race_id", "horse_id"])
    rows = rows.drop(columns=[c for c in ("comment_result_words",) if c in rows.columns]).copy()
    rows["race_id"] = rows["race_id"].astype(str)
    rows["horse_id"] = rows["horse_id"].astype(str)
    rows = rows.drop_duplicates(["race_id", "horse_id"], keep="last")
    e = k.merge(rows, on=["race_id", "horse_id"], how="left")
    for c in ("oikiri_eval", "has_comment", "comment_len", "comment_pos", "comment_neg"):
        if c not in e.columns:
            e[c] = np.nan
    e["has_comment"] = e["has_comment"].astype(float).fillna(0.0)
    e["oikiri_eval_rank"] = e.groupby("race_id")["oikiri_eval"].rank(ascending=False, method="average")
    e["oikiri_eval_gap"] = e["oikiri_eval"] - e.groupby("race_id")["oikiri_eval"].transform("max")
    out = e[EXTRAS_COLUMNS].astype(float)
    out.index = keys.index
    return out
