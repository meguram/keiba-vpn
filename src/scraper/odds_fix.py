"""race_result の単勝オッズ(odds)・人気(popularity)が指数列の値になっているレースの補正。

2025-05 以降の結果ページは会員表示で「タイム指数・スタート指数・追走指数・上がり指数」の列が入り 25 列になった。
2026-10-14 の列名ベースへの修正（d34c416）より前の固定位置フォールバック（odds=12, 人気=13 …）で保存した分は、
odds が追走指数、人気が上がり指数、上り3F がスタート指数…を読んでいた。`repair_race_result_from_on_time` は
上り・通過・馬体重・人気を on_time で直したが、odds は「0 より大きければ正常」と見なして直さなかった。
結果として 2025-05〜12 のおよそ 9 割のレース（約 1,900R）の odds が単勝オッズではない（Σ(1/odds)≈0.2、正常は約 1.26）。

race_odds（確定の単勝オッズ。正常な年は race_result.odds と 99.5% 一致）から対応表を作り、読む側で当てる。
既存の保存物・フラット表は書き換えない。

python -m src.scraper.odds_fix            # 対応表を data/local/meta/odds_fix/win_odds_fix.parquet に書く
KEIBA_ODDS_FIX=0 で読む側の補正を止める。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "data/page_reference/tables"
OUT_DIR = ROOT / "data/local/meta/odds_fix"
OVERLAY = OUT_DIR / "win_odds_fix.parquet"
RESIDUAL = OUT_DIR / "unfixable_races.txt"

# JRA の単勝は払戻率 80% なので Σ(1/odds) は約 1.25。20 年分の正常レースでこの範囲に入る（build の出力で確認）
SUM_INV_LO, SUM_INV_HI = 1.0, 1.6
_TOL_ABS, _TOL_REL = 0.05, 0.01


def sum_inv_odds(odds: Iterable[float]) -> float:
    v = np.asarray([float(x) for x in odds if x is not None and float(x) > 0], dtype=float)
    return float((1.0 / np.clip(v, 1.0, None)).sum()) if v.size else 0.0


def is_plausible_win_odds(odds: Iterable[float]) -> bool:
    """レース内の単勝オッズとして妥当か（Σ(1/odds) が払戻率から決まる範囲）。取消馬(0)は数えない。"""
    odds = list(odds)
    if sum(1 for x in odds if x and float(x) > 0) < 4:
        return True   # 頭数が少なすぎて判定できない（障害の少頭数など）。誤って捨てない
    return SUM_INV_LO <= sum_inv_odds(odds) <= SUM_INV_HI


def _match(odds: pd.Series, win: pd.Series) -> pd.Series:
    return (odds - win).abs() <= (_TOL_ABS + _TOL_REL * win)


def build_overlay(years: range = range(2020, 2027)) -> tuple[pd.DataFrame, list[str], dict]:
    """(対応表, 直せないレース ID, 集計) を返す。対応表は race_odds で置き換える馬単位の行。"""
    parts, unfix, stat = [], [], {}
    for y in years:
        rr_f, ro_f = TABLES / str(y) / "race_result_flat.parquet", TABLES / str(y) / "race_odds_flat.parquet"
        if not rr_f.exists():
            continue
        rr = pd.read_parquet(rr_f, columns=["race_id", "date", "horse_number", "odds", "popularity"])
        rr["race_id"] = rr["race_id"].astype(str)
        rr["odds"] = pd.to_numeric(rr["odds"], errors="coerce")
        s = rr.groupby("race_id")["odds"].apply(lambda x: is_plausible_win_odds(x.tolist()))
        implausible = set(s[~s].index)
        if ro_f.exists():
            ro = pd.read_parquet(ro_f, columns=["race_id", "horse_number", "win_odds", "popularity"]).rename(columns={"popularity": "pop_odds"})
            ro["race_id"] = ro["race_id"].astype(str)
            j = rr.merge(ro, on=["race_id", "horse_number"], how="inner")
            j = j[j["win_odds"] > 0]
            j["m"] = _match(j["odds"], j["win_odds"])
            share = j.groupby("race_id")["m"].mean()
            bad = set(share[share < 0.5].index)
        else:
            ro, bad = pd.DataFrame(columns=["race_id", "horse_number", "win_odds", "pop_odds"]), set()
        flagged = bad | implausible
        ro_ids = set(ro["race_id"])
        fixable = flagged & ro_ids
        unfix += sorted(flagged - ro_ids)
        fx = ro[ro["race_id"].isin(fixable) & (ro["win_odds"] > 0)].copy()
        # 人気は race_odds の値。無ければオッズの昇順（同率は最小順位）
        fx["rank"] = fx.groupby("race_id")["win_odds"].rank(method="min").astype(int)
        fx["popularity_fixed"] = fx["pop_odds"].where(fx["pop_odds"] > 0, fx["rank"]).astype(int)
        fx = fx.rename(columns={"win_odds": "odds_fixed"})[["race_id", "horse_number", "odds_fixed", "popularity_fixed"]]
        fx["reason"] = np.where(fx["race_id"].isin(bad), "odds_mismatch_with_race_odds", "implausible_sum_inv_odds")
        parts.append(fx)
        stat[y] = {"races": int(rr["race_id"].nunique()), "flagged": len(flagged), "fixed": len(fixable), "no_race_odds": len(flagged - ro_ids)}
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["race_id", "horse_number", "odds_fixed", "popularity_fixed", "reason"])
    return out, unfix, stat


def write_overlay(path: Path | None = None) -> dict:
    """対応表を作って書く。既存の対応表の行は残す（補正済みで再書き出ししたフラット表から作り直しても失われないように）。"""
    path = path or OVERLAY
    ov, unfix, stat = build_overlay()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_parquet(path)
        ov = pd.concat([old, ov], ignore_index=True).drop_duplicates(["race_id", "horse_number"], keep="last")
        global _CACHE
        _CACHE = None
    ov.to_parquet(path, index=False)
    RESIDUAL.write_text("\n".join(unfix) + ("\n" if unfix else ""), encoding="utf-8")
    return {"rows": len(ov), "races": int(ov["race_id"].nunique()), "unfixable": len(unfix), "by_year": stat}


_CACHE: pd.DataFrame | None = None


def load_overlay(path: Path | None = None) -> pd.DataFrame:
    """対応表。無い／KEIBA_ODDS_FIX=0 のときは空。"""
    global _CACHE
    path = path or OVERLAY
    if os.environ.get("KEIBA_ODDS_FIX", "1") in ("0", "false") or not path.exists():
        return pd.DataFrame(columns=["race_id", "horse_number", "odds_fixed", "popularity_fixed", "reason"])
    if _CACHE is None:
        _CACHE = pd.read_parquet(path)
    return _CACHE


def load_lookup() -> dict[tuple[str, int], tuple[float, int]]:
    d = load_overlay()
    return {(r, int(h)): (float(o), int(p)) for r, h, o, p in zip(d["race_id"], d["horse_number"], d["odds_fixed"], d["popularity_fixed"])}


def fixed_race_ids() -> set[str]:
    return set(load_overlay()["race_id"])


_LOOKUP: dict | None = None


def fix_result(result: dict | None) -> dict | None:
    """保存済み race_result（entries 形式）に補正を当てたコピーを返す。補正が無いレースは同じ辞書をそのまま返す。"""
    global _LOOKUP
    if not result or not result.get("entries"):
        return result
    if _LOOKUP is None:
        _LOOKUP = load_lookup()
    rid = str(result.get("race_id") or "")
    if not _LOOKUP or not any((rid, int(e.get("horse_number") or 0)) in _LOOKUP for e in result["entries"] if isinstance(e, dict)):
        return result
    entries = []
    for e in result["entries"]:
        fx = _LOOKUP.get((rid, int(e.get("horse_number") or 0))) if isinstance(e, dict) else None
        entries.append({**e, "odds": fx[0], "popularity": fx[1]} if fx else e)
    return {**result, "entries": entries}


def apply_pandas(df: pd.DataFrame, odds_col: str = "odds", pop_col: str = "popularity") -> pd.DataFrame:
    """race_result_flat 形式の DataFrame に補正を当てたコピーを返す（元の値は odds_raw / popularity_raw に残す）。"""
    ov = load_overlay()
    if ov.empty or "race_id" not in df.columns or "horse_number" not in df.columns:
        return df
    out = df.copy()
    key = pd.MultiIndex.from_arrays([out["race_id"].astype(str), out["horse_number"].astype("Int64").fillna(-1).astype(int)])
    o = ov.set_index(["race_id", "horse_number"])
    o = o[~o.index.duplicated(keep="last")]
    hit = o.reindex(key)
    m = hit["odds_fixed"].notna().to_numpy()
    if m.any():
        out[f"{odds_col}_raw"] = out[odds_col]
        out.loc[m, odds_col] = hit["odds_fixed"].to_numpy()[m]
        if pop_col in out.columns:
            out[f"{pop_col}_raw"] = out[pop_col]
            out.loc[m, pop_col] = hit["popularity_fixed"].to_numpy()[m]
    return out


def load_result_flat_fixed(years: Iterable[int], columns: list[str] | None = None) -> pd.DataFrame:
    """race_result_flat を年ごとに読み、odds / popularity を補正して連結する（オッズを使う読み手向けの入口）。"""
    parts = []
    for y in years:
        f = TABLES / str(y) / "race_result_flat.parquet"
        if f.exists():
            cols = None if columns is None else sorted(set(columns) | {"race_id", "horse_number"})
            parts.append(pd.read_parquet(f, columns=cols))
    d = pd.concat(parts, ignore_index=True)
    return apply_pandas(d)


if __name__ == "__main__":
    import json

    print(json.dumps(write_overlay(), ensure_ascii=False, indent=1))
