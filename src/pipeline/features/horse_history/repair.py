"""race_result_flat の last_3f・通過順・馬体重が壊れたレースを、GCS の race_result_on_time から補うための取得処理。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

from .common import _valid_passing


def suspicious_race_ids(runs: pl.DataFrame, min_bad_share: float = 0.5) -> list[str]:
    """last_3f または通過順が、半数以上の馬で不正なレース（障害レースは別扱いのため除く）。"""
    bad = runs.filter(pl.col("surface") != "障").with_columns(
        (~((pl.col("last_3f") >= 30) & (pl.col("last_3f") <= 50))).alias("bad_l3f"),
        (~_valid_passing("n_fin", "passing_order")).alias("bad_po"),
    )
    g = bad.group_by("race_id").agg(pl.col("bad_l3f").mean().alias("a"), pl.col("bad_po").mean().alias("b"))
    return g.filter((pl.col("a") >= min_bad_share) | (pl.col("b") >= min_bad_share))["race_id"].sort().to_list()


def _rows(race_id: str, doc: dict | None) -> list[dict]:
    if not doc:
        return []
    entries = next((v for v in doc.values() if isinstance(v, list) and v and isinstance(v[0], dict) and "horse_number" in v[0]), [])
    return [{"race_id": race_id, "horse_number": int(e["horse_number"]), "ot_l3f": e.get("last_3f"),
             "ot_passing_order": e.get("passing_order"), "ot_weight": e.get("weight")}
            for e in entries if e.get("horse_number") is not None]


def fetch_on_time(race_ids: list[str], cache_path: str | Path, workers: int = 8) -> pl.DataFrame:
    """GCS から race_result_on_time を読む（読み取りのみ）。取得済みの race_id は cache_path から再利用する。"""
    cache = Path(cache_path)
    have = pl.read_parquet(cache) if cache.exists() else None
    done = set(have["race_id"].unique().to_list()) if have is not None else set()
    todo = [r for r in race_ids if r not in done]
    if todo:
        from src.scraper.storage import HybridStorage

        st = HybridStorage()
        with ThreadPoolExecutor(workers) as ex:
            rows = [r for part in ex.map(lambda rid: _rows(rid, st.load("race_result_on_time", rid)), todo) for r in part]
        new = pl.DataFrame(rows, schema={"race_id": pl.Utf8, "horse_number": pl.Int64, "ot_l3f": pl.Float64,
                                         "ot_passing_order": pl.Utf8, "ot_weight": pl.Float64})
        have = new if have is None else pl.concat([have, new])
        cache.parent.mkdir(parents=True, exist_ok=True)
        have.write_parquet(cache)
    return have.filter(pl.col("race_id").is_in(race_ids)) if have is not None else pl.DataFrame()
