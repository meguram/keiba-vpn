"""
`data/local/tables` の flat テーブルから、特徴量ストア向けに選別した列をブロック別に登録する。

  python -m src.pipeline.register_raw_table_features
  python -m src.pipeline.register_raw_table_features --years 2024 2025 --overwrite

選別定義は `pipeline/raw_table_features.py`。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.pipeline.features.feature_layout import (
    BLOCK_BASE_TBL,
    BLOCK_HORSE_TBL,
    BLOCK_RACE_HORSE_TBL,
    BLOCK_RACE_JOCKEY_TBL,
    BLOCK_RACE_TBL,
    BLOCK_RACE_TRAINER_TBL,
    MERGE_KEYS_BASE_TBL,
    MERGE_KEYS_HORSE_TBL,
    MERGE_KEYS_RACE_HORSE_TBL,
    MERGE_KEYS_RACE_JOCKEY_TBL,
    MERGE_KEYS_RACE_TBL,
    MERGE_KEYS_RACE_TRAINER_TBL,
    SOURCE_ROW_KEYS,
)
from src.pipeline.features.feature_store import FeatureStore
from src.pipeline.features.id_value_policy import NETKEIBA_STRING_ID_COLUMNS, netkeiba_string_id_present, sanitize_netkeiba_string_id
from src.pipeline.features.race_index_speed_recent import expand_speed_recent_column
from src.pipeline.features.raw_table_features import (
    ALL_RAW_SPECS,
    INDEX_SPECS,
    PADDOCK_SPECS,
    SHUTUBA_ID_PRESENCE_STORE_NAMES,
    SHUTUBA_RACE_LEVEL_STORE_NAMES,
    SHUTUBA_PAST_SPECS,
    SHUTUBA_SPECS,
    TRAINER_COMMENT_SPECS,
    RawTableColumnSpec,
    _SHUTUBA_ENTRY_LEVEL_COLS,
    _SHUTUBA_HORSE_PROFILE_SOURCE_COLS,
    _SHUTUBA_RACE_HORSE_SOURCE_COLS,
    all_raw_feature_store_stems,
)
from src.utils.keiba_logging import script_basic_config

logger = logging.getLogger(__name__)

MANIFEST_PATH = Path("data/local/features/_raw_table_feature_selection.json")

BASE_KEYS = list(MERGE_KEYS_BASE_TBL)


def _normalize_race_horse_source(df: pd.DataFrame) -> pd.DataFrame:
    """race_id + horse_id をキーにする前処理（馬番は削除）。"""
    out = df.dropna(subset=["race_id"]).copy()
    out["horse_id"] = sanitize_netkeiba_string_id(out["horse_id"])
    out = out.dropna(subset=["horse_id"])
    return out.drop(columns=["horse_number"], errors="ignore")


def _prepare_shutuba_base(df: pd.DataFrame) -> pd.DataFrame:
    """出馬表行に base_tbl 用の ID 列を付与（馬番は除去前に使用可）。"""
    out = df.dropna(subset=["race_id"]).copy()
    for c in ("horse_id", "jockey_id", "trainer_id"):
        if c in out.columns:
            out[c] = sanitize_netkeiba_string_id(out[c])
    return out.drop(columns=["horse_number"], errors="ignore")


def _purge_legacy_shutuba_fragment_columns(store: FeatureStore) -> None:
    """旧レイアウト（base_tbl に出馬表が stem 別）のレジストリ・ファイルを除去。"""
    stems = [f"shutuba_{c}" for c in _SHUTUBA_ENTRY_LEVEL_COLS] + list(SHUTUBA_ID_PRESENCE_STORE_NAMES)
    for stem in stems:
        if stem in store._registry:
            store.delete_column(stem)


def _purge_shutuba_bundle_files(store: FeatureStore) -> None:
    """一体型出馬表 stem を再生成前に削除（上書きで置換）。"""
    for stem in ("shutuba", "shutuba_race_horse", "shutuba_horse", "shutuba_race_jockey", "shutuba_race_trainer"):
        if stem in store._registry:
            store.delete_column(stem)


def _register_shutuba_bundle(
    store: FeatureStore,
    years: list[str] | None,
    overwrite: bool,
    partition_partial: bool,
) -> list[dict]:
    """出馬表: race_tbl + base_tbl（4キーのみ）+ race_horse + horse_tbl（血統・馬プロファイル）+ race_jockey / race_trainer。"""
    cols_needed = sorted({s.column for s in SHUTUBA_SPECS} | {"horse_id", "jockey_id", "trainer_id"})
    df = store.load_source("race_shutuba", years=years, columns=cols_needed)
    if df.empty:
        return [{"store_name": "shutuba", "ok": False, "error": "race_shutuba 空"}]

    _purge_legacy_shutuba_fragment_columns(store)
    _purge_shutuba_bundle_files(store)

    df = _prepare_shutuba_base(df)
    df = df.dropna(subset=["horse_id"])
    results: list[dict] = []

    for spec in SHUTUBA_SPECS:
        if spec.store_name not in SHUTUBA_RACE_LEVEL_STORE_NAMES:
            continue
        try:
            transform = sanitize_netkeiba_string_id if spec.column in NETKEIBA_STRING_ID_COLUMNS else None
            reg_extra = (
                {
                    "value_policy": "nullable_string_id",
                    "id_semantics": "empty_or_zero_string_to_null_not_numeric_zero_fill",
                }
                if transform is not None
                else None
            )
            col_series = df[spec.column]
            if transform is not None:
                col_series = transform(col_series)
            tmp = pd.DataFrame({"race_id": df["race_id"], spec.store_name: col_series})
            rdf = tmp.groupby("race_id", as_index=False).first()
            path = store.save_feature_column(
                spec.store_name,
                rdf,
                table_block=BLOCK_RACE_TBL,
                merge_keys=list(MERGE_KEYS_RACE_TBL),
                overwrite=overwrite,
                partition_partial=partition_partial,
                registry_extra={
                    "source": spec.source,
                    "source_column": spec.column,
                    **(reg_extra or {}),
                },
            )
            results.append({"store_name": spec.store_name, "ok": True, "path": str(path), "note": spec.note})
        except Exception as e:
            logger.warning("登録失敗 %s: %s", spec.store_name, e)
            results.append({"store_name": spec.store_name, "ok": False, "error": str(e), "note": spec.note})

    try:
        base_df = df[BASE_KEYS].copy()
        p_base = store.save_feature_column(
            "shutuba",
            base_df,
            table_block=BLOCK_BASE_TBL,
            merge_keys=BASE_KEYS,
            overwrite=overwrite,
            partition_partial=partition_partial,
            registry_extra={
                "source": "race_shutuba_flat",
                "value_columns": [],
                "note": "予測ベースのスパイン（race_id, horse_id, jockey_id, trainer_id のみ）。",
            },
        )
        results.append({"store_name": "shutuba", "ok": True, "path": str(p_base), "rows": len(base_df)})
    except Exception as e:
        logger.warning("shutuba 一体登録失敗: %s", e)
        results.append({"store_name": "shutuba", "ok": False, "error": str(e)})

    try:
        rh = df[["race_id", "horse_id"]].copy()
        for c in _SHUTUBA_RACE_HORSE_SOURCE_COLS:
            if c not in df.columns:
                continue
            dest = "shutuba_horse_id" if c == "horse_id" else f"shutuba_{c}"
            ser = df[c]
            if c in NETKEIBA_STRING_ID_COLUMNS:
                ser = sanitize_netkeiba_string_id(ser)
            rh[dest] = ser
        for col in sorted(NETKEIBA_STRING_ID_COLUMNS):
            if col not in df.columns:
                continue
            rh[f"shutuba_{col}_present"] = netkeiba_string_id_present(df[col])
        p_rh = store.save_feature_column(
            "shutuba_race_horse",
            rh,
            table_block=BLOCK_RACE_HORSE_TBL,
            merge_keys=list(MERGE_KEYS_RACE_HORSE_TBL),
            overwrite=overwrite,
            partition_partial=partition_partial,
            registry_extra={
                "source": "race_shutuba_flat",
                "value_columns": [c for c in rh.columns if c not in ("race_id", "horse_id")],
                "note": "race_id+horse_id の当該レース馬行（枠・体重・斤量・ID 有効フラグ等）。",
            },
        )
        results.append({"store_name": "shutuba_race_horse", "ok": True, "path": str(p_rh), "rows": len(rh)})
    except Exception as e:
        logger.warning("shutuba_race_horse 登録失敗: %s", e)
        results.append({"store_name": "shutuba_race_horse", "ok": False, "error": str(e)})

    try:
        hp_cols = [c for c in _SHUTUBA_HORSE_PROFILE_SOURCE_COLS if c in df.columns]
        if "horse_id" not in hp_cols:
            raise ValueError("horse profile に horse_id 列が必要")
        hp = df[["race_id", *hp_cols]].copy()
        hp["horse_id"] = sanitize_netkeiba_string_id(hp["horse_id"])
        hp = hp.dropna(subset=["horse_id"])
        hp = hp.sort_values(["horse_id", "race_id"]).drop_duplicates(subset=["horse_id"], keep="last")
        hp = hp.drop(columns=["race_id"])
        _hp_ren = {
            "horse_name": "shutuba_horse_name",
            "sex_age": "shutuba_sex_age",
            "sire": "shutuba_sire",
            "dam_sire": "shutuba_dam_sire",
        }
        hp = hp.rename(columns={k: v for k, v in _hp_ren.items() if k in hp.columns})
        if partition_partial:
            try:
                old_hp = store.load_column("shutuba_horse")
                hp = pd.concat([old_hp, hp], ignore_index=True)
                hp = hp.sort_values("horse_id").drop_duplicates(subset=["horse_id"], keep="last")
            except FileNotFoundError:
                pass
        p_hp = store.save_feature_column(
            "shutuba_horse",
            hp,
            table_block=BLOCK_HORSE_TBL,
            merge_keys=list(MERGE_KEYS_HORSE_TBL),
            overwrite=overwrite or partition_partial,
            partition_partial=False,
            registry_extra={
                "source": "race_shutuba_flat",
                "value_columns": [c for c in hp.columns if c != "horse_id"],
                "note": "馬単位（血統・馬名・性齢）。部分年登録時は既存行とマージ。",
            },
        )
        results.append({"store_name": "shutuba_horse", "ok": True, "path": str(p_hp), "rows": len(hp)})
    except Exception as e:
        logger.warning("shutuba_horse 登録失敗: %s", e)
        results.append({"store_name": "shutuba_horse", "ok": False, "error": str(e)})

    try:
        jdf = df.sort_values(BASE_KEYS).groupby(["race_id", "jockey_id"], sort=False).head(1)
        jdf = jdf[["race_id", "jockey_id", "jockey_name"]].copy()
        jdf["jockey_id"] = sanitize_netkeiba_string_id(jdf["jockey_id"])
        jdf = jdf.dropna(subset=["jockey_id"])
        jdf = jdf.rename(columns={"jockey_name": "shutuba_jockey_name"})
        p_rj = store.save_feature_column(
            "shutuba_race_jockey",
            jdf,
            table_block=BLOCK_RACE_JOCKEY_TBL,
            merge_keys=list(MERGE_KEYS_RACE_JOCKEY_TBL),
            overwrite=overwrite,
            partition_partial=partition_partial,
            registry_extra={
                "source": "race_shutuba_flat",
                "note": "race_id+jockey_id 一意化（レース内騎手参照用）",
            },
        )
        results.append({"store_name": "shutuba_race_jockey", "ok": True, "path": str(p_rj), "rows": len(jdf)})
    except Exception as e:
        logger.warning("shutuba_race_jockey 登録失敗: %s", e)
        results.append({"store_name": "shutuba_race_jockey", "ok": False, "error": str(e)})

    try:
        tdf = df.sort_values(BASE_KEYS).groupby(["race_id", "trainer_id"], sort=False).head(1)
        tdf = tdf[["race_id", "trainer_id", "trainer_name"]].copy()
        tdf["trainer_id"] = sanitize_netkeiba_string_id(tdf["trainer_id"])
        tdf = tdf.dropna(subset=["trainer_id"])
        tdf = tdf.rename(columns={"trainer_name": "shutuba_trainer_name"})
        p_rt = store.save_feature_column(
            "shutuba_race_trainer",
            tdf,
            table_block=BLOCK_RACE_TRAINER_TBL,
            merge_keys=list(MERGE_KEYS_RACE_TRAINER_TBL),
            overwrite=overwrite,
            partition_partial=partition_partial,
            registry_extra={
                "source": "race_shutuba_flat",
                "note": "race_id+trainer_id 一意化（レース内調教師参照用）",
            },
        )
        results.append({"store_name": "shutuba_race_trainer", "ok": True, "path": str(p_rt), "rows": len(tdf)})
    except Exception as e:
        logger.warning("shutuba_race_trainer 登録失敗: %s", e)
        results.append({"store_name": "shutuba_race_trainer", "ok": False, "error": str(e)})

    return results


def _prune_obsolete_feature_columns(store: FeatureStore) -> list[str]:
    """旧列（time_index_m・リスト版 speed_recent）をストアから削除。"""
    names = ("idx_time_index_m", "idx_speed_recent")
    for name in names:
        store.delete_column(name)
    return list(names)


def _register_race_index_speed_recent_expanded(
    store: FeatureStore,
    years: list[str] | None,
    overwrite: bool,
    partition_partial: bool,
) -> list[dict]:
    """``speed_recent`` リストを ``idx_speed_recent_{1,2,3}`` の float 列に分割して保存。"""
    df = store.load_source("race_index", years=years, columns=["speed_recent", "horse_id"])
    if df.empty:
        return [{"store_name": f"idx_speed_recent_{i}", "ok": False, "error": "race_index 空"} for i in (1, 2, 3)]

    df = _normalize_race_horse_source(df)
    expanded = expand_speed_recent_column(df["speed_recent"], dest_prefix="idx_speed_recent", n_slots=3)
    results: list[dict] = []
    for name, ser in expanded.items():
        try:
            sub = df[["race_id", "horse_id"]].copy()
            sub[name] = ser
            store.save_feature_column(
                name,
                sub,
                table_block=BLOCK_RACE_HORSE_TBL,
                merge_keys=list(MERGE_KEYS_RACE_HORSE_TBL),
                overwrite=overwrite,
                partition_partial=partition_partial,
                registry_extra={
                    "source": "race_index_flat",
                    "source_column": "speed_recent",
                    "derived_from_list": True,
                    "slot": name.rsplit("_", 1)[-1],
                    "value_policy": "parser_zero_as_null",
                },
            )
            results.append({"store_name": name, "ok": True, "rows": len(sub)})
        except Exception as e:
            logger.warning("speed_recent 展開登録失敗 %s: %s", name, e)
            results.append({"store_name": name, "ok": False, "error": str(e)})
    return results


def _load_flat_per_year(
    base_dir: Path,
    source: str,
    years: list[str],
    columns: list[str],
) -> pd.DataFrame:
    """年別 Parquet を読み、スキーマに無い列がある年はスキップして縦結合。"""
    want = list(dict.fromkeys(list(SOURCE_ROW_KEYS) + ["horse_id"] + columns))
    frames: list[pd.DataFrame] = []
    for y in years:
        path = base_dir / Path("data/page_reference/tables") / y / f"{source}_flat.parquet"
        if not path.exists():
            continue
        schema = pq.read_schema(path)
        missing = [c for c in want if c not in schema.names]
        if missing:
            logger.warning("%s %s: 列不足のためスキップ missing=%s", y, source, missing)
            continue
        frames.append(pq.read_table(path, columns=want).to_pandas())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _register_paddock_deduped(
    store: FeatureStore,
    base_dir: Path,
    years: list[str] | None,
    overwrite: bool,
    partition_partial: bool,
) -> list[dict]:
    cols_needed = list({s.column for s in PADDOCK_SPECS})
    ys = years or [str(y) for y in store.available_years()]
    df = _load_flat_per_year(base_dir, "race_paddock", ys, cols_needed)
    if df.empty:
        return [{"store_name": s.store_name, "ok": False, "error": "race_paddock 空"} for s in PADDOCK_SPECS]

    df = _normalize_race_horse_source(df)
    df = df.sort_values(["race_id", "horse_id"])
    df = df.drop_duplicates(subset=["race_id", "horse_id"], keep="last")

    results: list[dict] = []
    for spec in PADDOCK_SPECS:
        try:
            sub = df[["race_id", "horse_id", spec.column]].rename(columns={spec.column: spec.store_name})
            store.save_feature_column(
                spec.store_name,
                sub,
                table_block=BLOCK_RACE_HORSE_TBL,
                merge_keys=list(MERGE_KEYS_RACE_HORSE_TBL),
                overwrite=overwrite,
                partition_partial=partition_partial,
                registry_extra={
                    "source": "race_paddock_flat",
                    "source_column": spec.column,
                    "preprocess": "drop_duplicates(race_id,horse_id,keep=last)",
                },
            )
            results.append({"store_name": spec.store_name, "ok": True, "rows": len(sub), "note": spec.note})
        except Exception as e:
            logger.warning("paddock 登録失敗 %s: %s", spec.store_name, e)
            results.append({"store_name": spec.store_name, "ok": False, "error": str(e)})
    return results


def _register_trainer_comment(
    store: FeatureStore,
    years: list[str] | None,
    overwrite: bool,
    partition_partial: bool,
) -> list[dict]:
    cols_needed = list({s.column for s in TRAINER_COMMENT_SPECS})
    df = store.load_source("race_trainer_comment", years=years, columns=cols_needed + ["horse_id"])
    if df.empty:
        return [{"store_name": s.store_name, "ok": False, "error": "race_trainer_comment 空"} for s in TRAINER_COMMENT_SPECS]

    df = _normalize_race_horse_source(df)

    results: list[dict] = []
    for spec in TRAINER_COMMENT_SPECS:
        try:
            sub = df[["race_id", "horse_id", spec.column]].rename(columns={spec.column: spec.store_name})
            store.save_feature_column(
                spec.store_name,
                sub,
                table_block=BLOCK_RACE_HORSE_TBL,
                merge_keys=list(MERGE_KEYS_RACE_HORSE_TBL),
                overwrite=overwrite,
                partition_partial=partition_partial,
                registry_extra={
                    "source": "race_trainer_comment_flat",
                    "source_column": spec.column,
                    "preprocess": "horse_id sanitize; rows with null horse_id dropped",
                },
            )
            results.append({"store_name": spec.store_name, "ok": True, "rows": len(sub), "note": spec.note})
        except Exception as e:
            logger.warning("trainer_comment 登録失敗 %s: %s", spec.store_name, e)
            results.append({"store_name": spec.store_name, "ok": False, "error": str(e)})
    return results


def register_all_raw_table_features(
    *,
    base_dir: str | Path = ".",
    years: list[str] | None = None,
    overwrite: bool = False,
) -> dict:
    base_path = Path(base_dir)
    store = FeatureStore(base_dir=base_dir)
    partition_partial = years is not None
    if years is None:
        years = [str(y) for y in store.available_years()]

    obsolete_removed = _prune_obsolete_feature_columns(store)

    def _simple_index_and_past() -> list[dict]:
        results: list[dict] = []
        for spec in INDEX_SPECS + SHUTUBA_PAST_SPECS:
            try:
                transform = sanitize_netkeiba_string_id if spec.column in NETKEIBA_STRING_ID_COLUMNS else None
                reg_extra = (
                    {
                        "value_policy": "nullable_string_id",
                        "id_semantics": "empty_or_zero_string_to_null_not_numeric_zero_fill",
                    }
                    if transform is not None
                    else None
                )
                path = store.register_column(
                    spec.store_name,
                    source=spec.source,
                    column=spec.column,
                    years=years,
                    transform=transform,
                    overwrite=overwrite,
                    registry_extra=reg_extra,
                    table_block=BLOCK_RACE_HORSE_TBL,
                    merge_keys=list(MERGE_KEYS_RACE_HORSE_TBL),
                    partition_partial=partition_partial,
                )
                results.append({"store_name": spec.store_name, "ok": True, "path": str(path), "note": spec.note})
            except Exception as e:
                logger.warning("登録失敗 %s: %s", spec.store_name, e)
                results.append({"store_name": spec.store_name, "ok": False, "error": str(e), "note": spec.note})
        return results

    results: dict[str, list] = {
        "obsolete_removed": [{"store_name": n, "ok": True} for n in obsolete_removed],
        "shutuba_bundle": _register_shutuba_bundle(store, years, overwrite, partition_partial),
        "race_index": _simple_index_and_past(),
        "race_index_speed_recent_expanded": _register_race_index_speed_recent_expanded(
            store, years, overwrite, partition_partial
        ),
        "paddock": _register_paddock_deduped(store, base_path, years, overwrite, partition_partial),
        "trainer_comment": _register_trainer_comment(store, years, overwrite, partition_partial),
    }

    manifest = {
        "years": years,
        "overwrite": overwrite,
        "spec_count": len(ALL_RAW_SPECS),
        "race_index_note": "time_index_m は特徴量ストアから除外。speed_recent は idx_speed_recent_1..3 に展開。",
        "policy": "pre_race_feature_whitelist.md LayerA-oriented; no odds/popularity from shutuba/index",
        "id_value_policy": "pipeline/id_value_policy.py — string id null not zero; shutuba_*_present flags",
        "excluded_tables_note": "race_oikiri_flat はキー重複のため別途集約してから登録すること",
        "layout": "base_tbl=shutuba（4キーのみ）/ horse_tbl=shutuba_horse / race_horse_tbl / race_jockey_tbl / race_trainer_tbl / race_tbl（馬番キー廃止）",
        "results": results,
        "store_names": list(all_raw_feature_store_stems()),
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("マニフェスト保存: %s", MANIFEST_PATH)
    return manifest


def main() -> None:
    script_basic_config()
    p = argparse.ArgumentParser(description="raw flat テーブル由来の列を特徴量ストアに登録")
    p.add_argument("--base-dir", default=".", help="リポジトリルート")
    p.add_argument("--years", nargs="*", help="対象年 (省略時は tables にある全年)")
    p.add_argument("--overwrite", action="store_true", help="既存 Parquet を上書き")
    args = p.parse_args()

    years = [str(y) for y in args.years] if args.years else None
    register_all_raw_table_features(base_dir=args.base_dir, years=years, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
