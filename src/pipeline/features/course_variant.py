"""コースマスタ（jra_courses.json）から、内回り/外回り（course_type）を (場, 面, 距離) で導出する。

背景: 出馬表の course_type は 2020〜2026 の全行が空だった（パーサーが実書式「芝2400m (右 外 B)」に合っていなかった）。
パーサーは 2026-10-17 に直したが、過去の保存データは再取得していないので、マスタから補う。

規則:
- (場, 面, 距離) に内外の候補が 1 つだけ → その値（'' は内外の区別が無いコース）
- 候補が複数（新潟芝 1400/2000、京都芝 1400/1600/2000、阪神芝 1400）→ 過去分は不明（NaN）
- パーサーが直った後の行（parsed_available=True）では、netkeiba は外回りだけに「外」を付け、内回りは無印なので、
  内外の両方がある距離で無印なら '内' とする
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

MASTER = Path(__file__).resolve().parents[3] / "data/meta/course_master/jra_courses.json"


@lru_cache(maxsize=1)
def key_variants(path: str = str(MASTER)) -> dict[tuple[str, str, int], frozenset]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[tuple[str, str, int], set] = {}
    for r in rows:
        out.setdefault((str(r["venue_code"]).zfill(2), str(r["surface"]), int(r["distance"])), set()).add(str(r.get("variant") or ""))
    return {k: frozenset(v) for k, v in out.items()}


def _norm_surface(s) -> str:
    s = str(s)
    return "ダート" if s.startswith("ダ") else s


def resolve_course_type(venue_code, surface, distance, parsed: str = "", parsed_available: bool = False) -> str | None:
    """内回り/外回りの値（'内' '外' ''=区別なし）。決まらなければ None。parsed は出馬表/結果ページから読んだ値。"""
    if parsed:
        return parsed
    try:
        v = key_variants().get((str(venue_code).zfill(2), _norm_surface(surface), int(distance)))
    except (TypeError, ValueError):
        return None
    if not v:
        return None
    if len(v) == 1:
        return next(iter(v))
    if "" in v:
        return "" if parsed_available else None
    if parsed_available and "内" in v and "外" in v:
        return "内"
    return None


def add_course_variant(df: pd.DataFrame, parsed_col: str | None = "course_type", parsed_available: bool = False) -> pd.Series:
    """df の各行の course_type をマスタで補う。決まらない行は NaN。venue_code・surface・distance の列が要る。"""
    parsed = df[parsed_col].fillna("").astype(str) if parsed_col and parsed_col in df.columns else pd.Series("", index=df.index)
    keys = pd.DataFrame({"v": df["venue_code"].astype(str).str.zfill(2), "s": df["surface"].map(_norm_surface), "d": pd.to_numeric(df["distance"], errors="coerce"), "p": parsed})
    cache: dict[tuple, str | None] = {}
    out = []
    for v, s, d, p in zip(keys["v"], keys["s"], keys["d"], keys["p"]):
        k = (v, s, d, p)
        if k not in cache:
            cache[k] = resolve_course_type(v, s, d, p, parsed_available) if d == d else None
        out.append(cache[k])
    return pd.Series(out, index=df.index, dtype="object").replace({None: np.nan})


def ambiguous_keys() -> list[tuple[str, str, int, list[str]]]:
    return sorted((k[0], k[1], k[2], sorted(v)) for k, v in key_variants().items() if len(v) > 1)


def stats(tables_dir: str | Path = "data/page_reference/tables", years=range(2020, 2027)) -> pd.DataFrame:
    """年別に、導出できた割合（一意に決まる）と、決まらない行数の内訳を返す。"""
    import polars as pl

    rows = []
    for y in years:
        f = Path(tables_dir) / str(y) / "race_shutuba_flat.parquet"
        if not f.exists():
            continue
        d = pl.read_parquet(f, columns=["race_id", "venue_code", "surface", "distance"]).unique("race_id").to_pandas()
        got = add_course_variant(d, parsed_col=None)
        turf = d["surface"].eq("芝")
        rows.append({"year": y, "races": len(d), "resolved_share": float(got.notna().mean()), "turf_races": int(turf.sum()),
                     "turf_resolved_share": float(got[turf].notna().mean()) if turf.any() else np.nan,
                     "ambiguous_races": int(got.isna().sum()), "outer_or_inner_races": int(got.isin(["内", "外"]).sum())})
    return pd.DataFrame(rows)


def ambiguous_race_ids(tables_dir: str | Path = "data/page_reference/tables", years=range(2020, 2027)) -> list[str]:
    """マスタだけでは内外が決まらないレース（結果ページから読み直すと決まる）。"""
    import polars as pl

    out: list[str] = []
    for y in years:
        f = Path(tables_dir) / str(y) / "race_shutuba_flat.parquet"
        if f.exists():
            d = pl.read_parquet(f, columns=["race_id", "venue_code", "surface", "distance"]).unique("race_id").to_pandas()
            out += d.loc[add_course_variant(d, parsed_col=None).isna(), "race_id"].astype(str).tolist()
    return sorted(out)


if __name__ == "__main__":
    print(stats().to_string(index=False))
    print("ambiguous keys:", ambiguous_keys())
