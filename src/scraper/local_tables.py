"""
ローカル Parquet テーブルへの高速アクセスユーティリティ。

GCS を経由せずにローカルの Parquet テーブルから直接データを読み込むことで、
GCS オペレーションコストとネットワーク転送を完全に回避する。

エクスポート済みの data/local/tables/{year}/{category}_flat.parquet を
メモリにキャッシュして提供する。
"""

from __future__ import annotations

import json as _json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _try_json_parse(val: Any) -> Any:
    """JSON 文字列ならパースして返す。失敗時はそのまま返す。"""
    if isinstance(val, str):
        try:
            return _json.loads(val)
        except (ValueError, TypeError):
            pass
    return val


def _numpy_to_python(val: Any) -> Any:
    """numpy スカラーを Python ネイティブ型に変換する。"""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        if math.isnan(val):
            return None
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, (np.ndarray,)):
        return val.tolist()
    return val

_cache_lock = threading.Lock()
_df_cache: dict[str, tuple[float, Any]] = {}
_DF_CACHE_TTL = 3600.0

TABLES_DIR = Path("data/page_reference/tables")


def _get_tables_dir(base_dir: str | Path = ".") -> Path:
    return Path(base_dir) / "data" / "page_reference" / "tables"


def available_years(base_dir: str | Path = ".") -> list[str]:
    d = _get_tables_dir(base_dir)
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and p.name.isdigit())


def has_local_table(category: str, year: str, base_dir: str | Path = ".") -> bool:
    return (_get_tables_dir(base_dir) / year / f"{category}_flat.parquet").exists()


def load_flat_df(
    category: str,
    years: list[str] | None = None,
    columns: list[str] | None = None,
    base_dir: str | Path = ".",
):
    """
    ローカル Parquet テーブルを DataFrame として読み込む。

    キャッシュ付き (1時間)。GCS コスト完全ゼロ。
    columns を指定すると必要な列のみ読み込んでメモリと速度を最適化。
    """
    import pandas as pd

    try:
        import pyarrow.parquet as pq
    except ImportError:
        return pd.DataFrame()

    tables_dir = _get_tables_dir(base_dir)
    if not tables_dir.exists():
        return pd.DataFrame()

    if years is None:
        years = available_years(base_dir)

    frames = []
    for y in years:
        cache_key = f"{category}/{y}/{','.join(columns) if columns else '*'}"

        with _cache_lock:
            cached = _df_cache.get(cache_key)
        if cached and (time.time() - cached[0]) < _DF_CACHE_TTL:
            frames.append(cached[1])
            continue

        flat_path = tables_dir / y / f"{category}_flat.parquet"
        if not flat_path.exists():
            continue

        try:
            read_cols = None
            if columns:
                schema = pq.read_schema(flat_path)
                read_cols = [c for c in columns if c in schema.names]
                if not read_cols:
                    continue

            df = pq.read_table(flat_path, columns=read_cols).to_pandas()
            with _cache_lock:
                _df_cache[cache_key] = (time.time(), df)
            frames.append(df)
        except Exception as e:
            logger.warning("Parquet 読込失敗 %s/%s: %s", category, y, e)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_race_json(
    category: str,
    race_id: str,
    base_dir: str | Path = ".",
) -> dict[str, Any] | None:
    """
    ローカル Parquet から 1 レース分のデータを JSON 互換 dict として返す。

    race_id をキーにフィルタし、entries/runners 形式に再構成する。
    Parquet にデータがなければ None を返す (GCS フォールバックのシグナル)。
    """
    import pandas as pd

    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None

    year = race_id[:4] if len(race_id) >= 4 else ""
    if not year:
        return None

    tables_dir = _get_tables_dir(base_dir)
    flat_path = tables_dir / year / f"{category}_flat.parquet"
    if not flat_path.exists():
        return None

    cache_key = f"json/{category}/{race_id}"
    with _cache_lock:
        cached = _df_cache.get(cache_key)
    if cached and (time.time() - cached[0]) < _DF_CACHE_TTL:
        return cached[1]

    try:
        df = pq.read_table(flat_path).to_pandas()
        if "race_id" not in df.columns:
            return None

        race_rows = df[df["race_id"] == race_id]
        if race_rows.empty:
            return None

        entry_key = "entries"
        if category == "smartrc_race":
            entry_key = "runners"
        elif category == "race_result_lap":
            entry_key = "entries_lap"

        meta_cols = [
            "race_id", "date", "venue", "surface", "distance", "direction",
            "grade", "race_class", "weather", "track_condition", "start_time",
            "field_size", "race_name", "round",
        ]
        result: dict[str, Any] = {}
        for col in meta_cols:
            if col in race_rows.columns:
                val = race_rows[col].iloc[0]
                if pd.notna(val):
                    val = _numpy_to_python(val)
                    result[col] = val if not isinstance(val, float) or val != int(val) else int(val)

        # race_result_lap: lap_times/pace/corner_passing はレース単位メタ
        _lap_meta_cols: set[str] = set()
        if category == "race_result_lap":
            _lap_meta_cols = {"lap_times", "pace", "corner_passing"}
            first = race_rows.iloc[0]
            for col in _lap_meta_cols:
                if col in race_rows.columns:
                    raw = first[col]
                    if pd.notna(raw):
                        result[col] = _try_json_parse(raw)
                    else:
                        result[col] = [] if col != "pace" else {}

        entries = race_rows.to_dict("records")
        for entry in entries:
            for k in meta_cols:
                entry.pop(k, None)
            for k in _lap_meta_cols:
                entry.pop(k, None)
            for k, v in list(entry.items()):
                if pd.isna(v) if not isinstance(v, (list, dict)) else False:
                    entry[k] = None
                else:
                    v = _numpy_to_python(v)
                    if isinstance(v, str) and v and v[0] in ("{", "["):
                        v = _try_json_parse(v)
                    entry[k] = v

        result[entry_key] = entries

        # smartrc_race: _race.parquet から horses/fullresults を補完
        if category == "smartrc_race":
            race_cache_key = f"__df__smartrc_race_race_{year}"
            df_race = None
            with _cache_lock:
                cached = _df_cache.get(race_cache_key)
                if cached and (time.time() - cached[0]) < 3600:
                    df_race = cached[1]
            if df_race is None:
                race_path = tables_dir / year / f"{category}_race.parquet"
                if race_path.exists():
                    try:
                        df_race = pq.read_table(race_path).to_pandas()
                        with _cache_lock:
                            _df_cache[race_cache_key] = (time.time(), df_race)
                    except Exception:
                        pass
            if df_race is not None:
                race_row = df_race[df_race["race_id"] == race_id]
                if not race_row.empty:
                    for col in ("horses", "fullresults"):
                        if col in race_row.columns:
                            raw = race_row[col].iloc[0]
                            if pd.notna(raw):
                                result[col] = _try_json_parse(raw)

        with _cache_lock:
            _df_cache[cache_key] = (time.time(), result)
        return result

    except Exception as e:
        logger.debug("Parquet race load 失敗 %s/%s: %s", category, race_id, e)
        return None


def load_all_races_grouped(
    category: str,
    years: list[str] | None = None,
    base_dir: str | Path = ".",
) -> dict[str, dict[str, Any]]:
    """
    カテゴリの全レースデータを {race_id: {meta + entries}} の dict で返す。

    研究モジュールのバルクアクセス用。DataFrame を1回だけ読み込み、
    race_id ごとにグループ化して返す。storage.load() のN回呼出を回避。
    """
    import pandas as pd

    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {}

    tables_dir = _get_tables_dir(base_dir)
    if not tables_dir.exists():
        return {}

    if years is None:
        years = available_years(base_dir)

    cache_key = f"grouped/{category}/{','.join(years)}"
    with _cache_lock:
        cached = _df_cache.get(cache_key)
    if cached and (time.time() - cached[0]) < _DF_CACHE_TTL:
        return cached[1]

    entry_key = "entries"
    if category == "smartrc_race":
        entry_key = "runners"
    elif category == "race_result_lap":
        entry_key = "entries_lap"

    meta_cols = [
        "race_id", "date", "venue", "surface", "distance", "direction",
        "grade", "race_class", "weather", "track_condition", "start_time",
        "field_size", "race_name", "round",
    ]

    result: dict[str, dict[str, Any]] = {}
    for y in years:
        flat_path = tables_dir / y / f"{category}_flat.parquet"
        if not flat_path.exists():
            continue

        try:
            df = pq.read_table(flat_path).to_pandas()
            if "race_id" not in df.columns:
                continue

            for rid, group in df.groupby("race_id"):
                race_dict: dict[str, Any] = {}
                first_row = group.iloc[0]
                for col in meta_cols:
                    if col in group.columns and col != "race_id":
                        val = first_row[col]
                        if pd.notna(val):
                            val = _numpy_to_python(val)
                            race_dict[col] = (
                                val if not isinstance(val, float) or val != int(val)
                                else int(val)
                            )

                entries = group.to_dict("records")
                for entry in entries:
                    for k in meta_cols:
                        entry.pop(k, None)
                    for k, v in list(entry.items()):
                        if pd.isna(v) if not isinstance(v, (list, dict)) else False:
                            entry[k] = None
                        else:
                            v = _numpy_to_python(v)
                            if isinstance(v, str) and v and v[0] in ("{", "["):
                                v = _try_json_parse(v)
                            entry[k] = v
                race_dict[entry_key] = entries
                result[str(rid)] = race_dict
        except Exception as e:
            logger.warning("grouped load 失敗 %s/%s: %s", category, y, e)

    with _cache_lock:
        _df_cache[cache_key] = (time.time(), result)
    return result


def list_race_ids(
    category: str,
    year: str | None = None,
    base_dir: str | Path = ".",
) -> list[str]:
    """
    ローカル Parquet テーブルから race_id の一覧を返す (GCS コスト ¥0)。

    storage.list_keys() の GCS list API 呼出を完全に回避する。
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return []

    tables_dir = _get_tables_dir(base_dir)
    if not tables_dir.exists():
        return []

    years = [year] if year else available_years(base_dir)
    all_ids: list[str] = []
    for y in years:
        cache_key = f"race_ids/{category}/{y}"
        with _cache_lock:
            cached = _df_cache.get(cache_key)
        if cached and (time.time() - cached[0]) < _DF_CACHE_TTL:
            all_ids.extend(cached[1])
            continue

        flat_path = tables_dir / y / f"{category}_flat.parquet"
        if not flat_path.exists():
            continue

        try:
            schema = pq.read_schema(flat_path)
            if "race_id" not in schema.names:
                continue
            tbl = pq.read_table(flat_path, columns=["race_id"])
            ids = sorted(tbl.column("race_id").to_pandas().unique().tolist())
            with _cache_lock:
                _df_cache[cache_key] = (time.time(), ids)
            all_ids.extend(ids)
        except Exception as e:
            logger.warning("Parquet race_id 一覧取得失敗 %s/%s: %s", category, y, e)

    return sorted(set(all_ids))


def clear_cache():
    """キャッシュを全クリアする。"""
    with _cache_lock:
        _df_cache.clear()
