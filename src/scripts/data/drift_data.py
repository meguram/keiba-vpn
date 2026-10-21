"""ドリフト診断・監視用の評価表（結果 + 市場確率 + モデル予測）を作る読み込み関数。

1 行 = 1 頭。列: race_id, horse_id, date, year, month, venue, surface, distance, grade, race_class, field_size, track_condition,
age, n_prior, p_win/p_top2/p_top3（モデル）, p_mkt（確定単勝オッズの逆数を正規化）, y_win/y_top2/y_top3, src（oof/test1/forward）。
中止・取消（着順なし）は除く。市場確率は Σ(1/odds) が 1.0〜1.6 の外のレースを除く（壊れたオッズ）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import readiness_eval as R

CFG_PATH = Path(__file__).resolve().parents[3] / "config" / "drift_periods.json"
RES_COLS = ["race_id", "date", "venue", "surface", "distance", "grade", "race_class", "field_size", "track_condition", "horse_id",
            "horse_number", "finish_position", "sex_age", "last_3f", "popularity"]
PCOLS = ["p_win", "p_top2", "p_top3"]


def load_cfg(path: Path | None = None) -> dict:
    return json.loads((path or CFG_PATH).read_text(encoding="utf-8"))


def load_results_year(tables: Path, year: int) -> pd.DataFrame:
    df = pd.read_parquet(tables / str(year) / "race_result_flat.parquet", columns=[c for c in RES_COLS])
    df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce")
    for c in ("race_id", "horse_id"):
        df[c] = df[c].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["race_id"].str[4:6].isin([f"{i:02d}" for i in range(1, 11)])]
    return df[df["finish_position"] >= 1].copy()


def market_probs(tables: Path, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(馬ごとの市場確率, レースごとの Σ1/odds)。確定単勝オッズ。"""
    import pyarrow.parquet as pq

    fo = tables / str(year) / "race_odds_flat.parquet"
    names = pq.read_schema(fo).names
    cols = [c for c in ["race_id", "horse_number", "win_odds", "snapshot_at"] if c in names]
    o = pd.read_parquet(fo, columns=cols)
    o["race_id"] = o["race_id"].astype(str)
    if "snapshot_at" in o.columns:
        o = o.sort_values("snapshot_at")
    o = o.drop_duplicates(["race_id", "horse_number"], keep="last")
    o["win_odds"] = pd.to_numeric(o["win_odds"], errors="coerce")
    o = o[o["win_odds"] >= 1.0].copy()
    o["inv"] = 1.0 / o["win_odds"]
    tot = o.groupby("race_id")["inv"].sum().rename("overround").reset_index()
    ok = tot[(tot["overround"] > 1.0) & (tot["overround"] < 1.6)]
    o = o.merge(ok, on="race_id")
    o["p_mkt"] = o["inv"] / o["overround"]
    return o[["race_id", "horse_number", "p_mkt", "win_odds"]], ok


def load_predictions(cfg: dict, root: Path, years: list[int]) -> pd.DataFrame:
    ps = cfg["prediction_sources"]
    parts = []
    if any(y <= 2024 for y in years) and (root / ps["oof"]).exists():
        x = pd.read_parquet(root / ps["oof"])[["race_id", "horse_id"] + PCOLS]
        x["src"] = "oof"
        parts.append(x)
    if any(y == 2025 for y in years) and (root / ps["test1"]).exists():
        x = pd.read_parquet(root / ps["test1"])[["race_id", "horse_id"] + PCOLS]
        x["src"] = "test1"
        parts.append(x)
    if any(y >= 2026 for y in years):
        f = R.load_preds([root / d for d in ps["forward_dirs"]])
        if len(f):
            f = f[["race_id", "horse_id"] + PCOLS].copy()
            f["src"] = "forward"
            parts.append(f)
    if not parts:
        return pd.DataFrame(columns=["race_id", "horse_id"] + PCOLS + ["src"])
    p = pd.concat(parts, ignore_index=True)
    for c in ("race_id", "horse_id"):
        p[c] = p[c].astype(str)
    return p.drop_duplicates(["race_id", "horse_id"], keep="last")


def load_eval_frame(root: Path, years: list[int], cfg: dict | None = None, with_prior: bool = True) -> pd.DataFrame:
    cfg = cfg or load_cfg()
    tables = root / "data/page_reference/tables"
    res, mkt = [], []
    for y in years:
        r = load_results_year(tables, y)
        r["year"] = y
        res.append(r)
        m, _ = market_probs(tables, y)
        mkt.append(m)
    res = pd.concat(res, ignore_index=True)
    if with_prior:
        res["n_prior"] = R.prior_runs(tables, res)
    mk = pd.concat(mkt, ignore_index=True)
    df = res.merge(mk, on=["race_id", "horse_number"], how="inner")
    pred = load_predictions(cfg, root, years)
    df = df.merge(pred, on=["race_id", "horse_id"], how="inner")
    df["month"] = df["date"].dt.month
    for k, n in R.TARGETS.items():
        df[f"y_{k}"] = (df["finish_position"] <= n).astype(float)
    df["age"] = pd.to_numeric(df["sex_age"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    return df


def add_strata(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "n_prior" in d:
        d["hist"] = pd.cut(d["n_prior"], [-1, 0, 2, 5, 999], labels=["0走", "1-2走", "3-5走", "6走~"]).astype(str)
    d["age_c"] = d["age"].clip(2, 6).fillna(0).astype(int).astype(str) + "歳"
    d["dist_band"] = pd.cut(d["distance"].astype(float), [0, 1300, 1700, 2100, 9999], labels=["~1200", "1400-1600", "1800-2000", "2200~"]).astype(str)
    d["f_size"] = pd.cut(d["field_size"].astype(float), [0, 9, 13, 16, 99], labels=["~9頭", "10-13頭", "14-16頭", "17頭~"]).astype(str)
    d["cls"] = d["race_class"].astype(str)
    d["is_newhorse_class"] = np.where(d["grade"].astype(str) == "新馬", "新馬", "その他")
    return d
