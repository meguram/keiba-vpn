"""
特徴量ストア

ローカル Parquet テーブル (data/local/tables/) をソースとし、
特徴量列をブロック別ディレクトリの列指向 Parquet で保存・取得する。

=== ディレクトリ構成 ===

  data/features/
  ├── _registry.json
  ├── base_tbl/          # 4キーのみ（race_id 年別 shutuba.parquet）
  ├── race_tbl/          # race_id（同上）
  ├── race_horse_tbl/    # race_id + horse_id（同上）
  ├── horse_tbl/         # horse_id（血統・馬プロファイル等。race_id なしの単一 or 運用で更新）
  ├── race_jockey_tbl/   # race_id + jockey_id
  ├── race_trainer_tbl/  # race_id + trainer_id
  ├── jockey_tbl/        # jockey_id
  ├── trainer_tbl/       # trainer_id
  ├── target/rank_tbl/   # ラベル（レジストリ外・別モジュールで生成）
  ├── horse/             # 馬単位エンティティ（レジストリ外・build_horse_entity_store）
  └── snapshots/

=== 使い方 ===

  store = FeatureStore()
  store.save_feature_column("idx_speed_max", df, table_block="race_horse_tbl",
                            merge_keys=["race_id", "horse_id"])

詳細は ``pipeline/feature_layout.py``。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.pipeline.features.feature_layout import (
    BLOCK_BASE_TBL,
    BLOCK_RACE_HORSE_TBL,
    FEATURES_DIR,
    JT_STATS_META_SUBDIR,
    MERGE_KEYS_RACE_HORSE_TBL,
    SOURCE_ROW_KEYS,
)

logger = logging.getLogger(__name__)

TABLES_DIR = Path("data/page_reference/tables")
SNAPSHOTS_DIR = FEATURES_DIR / "snapshots"
REGISTRY_PATH = FEATURES_DIR / "_registry.json"

# 後方互換: 旧コード・旧データが参照する定数（race_id + horse_id に移行済み）
KEY_COLS = list(MERGE_KEYS_RACE_HORSE_TBL)

# 特徴量ブロック直下を走査するとき除外（年ディレクトリではないもの）
_BLOCK_DIR_SKIP = frozenset({"snapshots", JT_STATS_META_SUBDIR, "race_performance", "target", "horse"})


def _is_race_year_dir(dirname: str) -> bool:
    return len(dirname) == 4 and dirname.isdigit()


class FeatureStore:
    """列指向の特徴量ストア。ローカル Parquet テーブルから特徴量を抽出・管理する。"""

    def __init__(
        self,
        base_dir: str | Path = ".",
        tables_dir: str | Path | None = None,
        features_dir: str | Path | None = None,
    ):
        base = Path(base_dir)
        self._tables_dir = Path(tables_dir) if tables_dir else base / TABLES_DIR
        self._features_dir = Path(features_dir) if features_dir else base / FEATURES_DIR
        self._snapshots_dir = self._features_dir / "snapshots"
        self._registry_path = self._features_dir / "_registry.json"
        self._registry: dict[str, dict] = self._load_registry()

    def block_dir(self, table_block: str) -> Path:
        return self._features_dir / table_block

    def _partition_years_from_df(self, df: pd.DataFrame) -> list[str] | None:
        """race_id 先頭4桁で年分割。race_id が無い・有効年が無いときは None（単一ファイル）。"""
        if "race_id" not in df.columns:
            return None
        s = df["race_id"].astype(str).str.slice(0, 4)
        years = sorted({y for y in s.dropna().unique() if len(y) == 4 and y.isdigit()})
        return years if years else None

    def _cleanup_column_files_before_write(
        self,
        name: str,
        table_block: str,
        *,
        active_partition_years: set[str] | None,
        partition_partial: bool,
    ) -> None:
        """上書き前の整合。

        - 年分割で **部分年のみ** 書き換えるとき（``partition_partial=True``）は、
          他年のシャードは削除しない（単一ファイル ``<name>.parquet`` のみ除去）。
        - **全年** を書き直すときは、データに無い年のシャードも削除する。
        - **単一ファイル** へ書くときは、同名の年シャードをすべて削除する。
        """
        d = self.block_dir(table_block)
        if not d.exists():
            return
        flat = d / f"{name}.parquet"
        if active_partition_years is not None:
            if flat.is_file():
                flat.unlink()
            if not partition_partial:
                for ydir in d.iterdir():
                    if not ydir.is_dir() or not _is_race_year_dir(ydir.name):
                        continue
                    p = ydir / f"{name}.parquet"
                    if p.is_file() and ydir.name not in active_partition_years:
                        p.unlink()
        else:
            for ydir in d.iterdir():
                if not ydir.is_dir() or not _is_race_year_dir(ydir.name):
                    continue
                p = ydir / f"{name}.parquet"
                if p.is_file():
                    p.unlink()

    def _registry_primary_path(self, name: str) -> Path | None:
        meta = self._registry.get(name)
        if not meta:
            return None
        if meta.get("year_partitioned") and meta.get("year_paths"):
            yp = meta["year_paths"]
            if not isinstance(yp, dict) or not yp:
                return None
            first_y = sorted(yp.keys())[0]
            rel = yp[first_y]
            p = self._features_dir / rel
            return p if p.is_file() else None
        rel = meta.get("rel_path")
        if rel:
            p = self._features_dir / rel
            return p if p.is_file() else None
        return None

    def _discover_paths_for_column(
        self, name: str, years: list[str | int] | None = None
    ) -> list[Path]:
        """レジストリが無い場合のフォールバック探索（単一 + 年別）。"""
        yset = {str(y) for y in years} if years else None
        paths: list[Path] = []
        for subdir in sorted(self._features_dir.iterdir()):
            if not subdir.is_dir() or subdir.name in _BLOCK_DIR_SKIP:
                continue
            flat = subdir / f"{name}.parquet"
            if flat.is_file():
                paths.append(flat)
            for ydir in sorted(subdir.iterdir()):
                if not ydir.is_dir() or not _is_race_year_dir(ydir.name):
                    continue
                if yset is not None and ydir.name not in yset:
                    continue
                p = ydir / f"{name}.parquet"
                if p.is_file():
                    paths.append(p)
        return paths

    # ── ソーステーブル操作 ─────────────────────────────

    def available_years(self) -> list[str]:
        """エクスポート済みの年を返す。"""
        if not self._tables_dir.exists():
            return []
        return sorted(
            d.name for d in self._tables_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        )

    def list_sources(self, year: str | int) -> list[str]:
        """指定年のエクスポート済みカテゴリ (flat テーブル) 一覧。"""
        year_dir = self._tables_dir / str(year)
        if not year_dir.exists():
            return []
        return sorted(
            f.stem.replace("_flat", "")
            for f in year_dir.glob("*_flat.parquet")
        )

    def load_source(
        self,
        category: str,
        years: list[str | int] | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """ソーステーブル (flat) を年横断で読み込む。

        columns を指定すると SOURCE_ROW_KEYS + 指定列のみ読み込むため高速。
        """
        if years is None:
            years = self.available_years()

        frames = []
        read_cols = None
        if columns:
            read_cols = list(dict.fromkeys(list(SOURCE_ROW_KEYS) + columns))

        for y in years:
            path = self._tables_dir / str(y) / f"{category}_flat.parquet"
            if not path.exists():
                continue
            try:
                tbl = pq.read_table(path, columns=read_cols)
                frames.append(tbl.to_pandas())
            except Exception as e:
                logger.warning("読み込み失敗 %s: %s", path, e)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def source_schema(self, category: str, year: str | int | None = None) -> dict[str, str]:
        """ソーステーブルのスキーマ (列名→型) を返す。"""
        if year is None:
            years = self.available_years()
            if not years:
                return {}
            year = years[0]
        path = self._tables_dir / str(year) / f"{category}_flat.parquet"
        if not path.exists():
            return {}
        schema = pq.read_schema(path)
        return {f.name: str(f.type) for f in schema}

    # ── 列の登録・保存 ──────────────────────────────

    def save_feature_column(
        self,
        name: str,
        df: pd.DataFrame,
        *,
        table_block: str,
        merge_keys: list[str],
        overwrite: bool = False,
        registry_extra: dict[str, Any] | None = None,
        partition_partial: bool = False,
    ) -> Path:
        """ブロック配下に1特徴量列（または同キーの複数値列）を保存する。

        ``race_id`` 列があるときは ``<block>/<YYYY>/<name>.parquet`` に年分割する。
        無いとき（騎手・調教師の参照表など）は従来どおり ``<block>/<name>.parquet`` 1本。

        ``partition_partial``:
            True のときは **渡した DataFrame に含まれる年のファイルだけ** を更新し、
            他年の Parquet は残す（``register_column(..., years=...)`` 等の部分更新用）。
        """
        if not overwrite and name in self._registry:
            p = self._registry_primary_path(name)
            if p is not None:
                logger.info("既存スキップ(レジストリ): %s", name)
                return p

        for kc in merge_keys:
            if kc not in df.columns:
                raise ValueError(f"キー列 '{kc}' が DataFrame に必要です (feature={name})")

        value_cols = [c for c in df.columns if c not in merge_keys]
        if len(value_cols) == 1 and value_cols[0] != name:
            df = df.rename(columns={value_cols[0]: name})
            value_cols = [name]

        write_df = df[list(merge_keys) + value_cols]
        part_years = self._partition_years_from_df(write_df)
        if overwrite:
            self._cleanup_column_files_before_write(
                name,
                table_block,
                active_partition_years=set(part_years) if part_years else None,
                partition_partial=partition_partial,
            )

        primary_path, meta_paths = self._save_column_file(name, write_df, table_block, part_years)

        meta: dict[str, Any] = {
            "source": "custom",
            "table_block": table_block,
            "merge_keys": list(merge_keys),
            "rel_path": str(primary_path.relative_to(self._features_dir)),
            "columns": value_cols,
            "rows": len(write_df),
            "dtype": (
                str(write_df[value_cols[0]].dtype)
                if len(value_cols) == 1
                else ("multi" if len(value_cols) > 1 else "keys_only")
            ),
        }
        if meta_paths.get("year_partitioned"):
            meta["year_partitioned"] = True
            new_yp = meta_paths["year_paths"]
            if partition_partial and name in self._registry:
                old = self._registry.get(name) or {}
                old_yp = old.get("year_paths") if isinstance(old.get("year_paths"), dict) else {}
                merged_yp = {**old_yp, **new_yp}
                meta["year_paths"] = merged_yp
                first_y = sorted(merged_yp.keys())[0]
                meta["rel_path"] = merged_yp[first_y]
                total_r = 0
                for rel in merged_yp.values():
                    mp = self._features_dir / rel
                    if mp.is_file():
                        total_r += int(pq.read_metadata(mp).num_rows)
                meta["rows"] = total_r
            else:
                meta["year_paths"] = new_yp
        if registry_extra:
            meta.update(registry_extra)
        self._update_registry(name, meta)
        return primary_path

    def register_column(
        self,
        name: str,
        *,
        source: str,
        column: str,
        years: list[str | int] | None = None,
        transform: Any = None,
        overwrite: bool = False,
        registry_extra: dict[str, Any] | None = None,
        table_block: str = BLOCK_RACE_HORSE_TBL,
        merge_keys: list[str] | None = None,
        partition_partial: bool = False,
    ) -> Path:
        """ソーステーブルから特定列を抽出して列ストアに保存する。

        horse_id がソースに無い場合はエラー（馬番キーでの保存は廃止）。
        """
        mk = list(merge_keys or MERGE_KEYS_RACE_HORSE_TBL)
        if not overwrite and name in self._registry:
            p = self._registry_primary_path(name)
            if p is not None:
                logger.info("既存スキップ: %s", name)
                return p

        need = ["horse_id", column]
        df = self.load_source(source, years=years, columns=need)
        if df.empty:
            raise ValueError(f"ソースデータなし: {source}/{column}")

        if "horse_id" not in df.columns:
            raise ValueError(f"ソース {source} に horse_id 列がありません（特徴量ストアは ID キーのみ）")

        df = df[list(SOURCE_ROW_KEYS) + ["horse_id", column]].copy()
        from src.pipeline.features.id_value_policy import sanitize_netkeiba_string_id

        df["horse_id"] = sanitize_netkeiba_string_id(df["horse_id"])
        df = df.dropna(subset=["horse_id"])
        df = df.drop(columns=["horse_number"], errors="ignore")

        if transform is not None:
            df[column] = transform(df[column])
        if column != name:
            df = df.rename(columns={column: name})

        extra: dict[str, Any] = {
            "source": source,
            "source_column": column,
            "has_transform": transform is not None,
        }
        if registry_extra:
            extra.update(registry_extra)
        return self.save_feature_column(
            name,
            df,
            table_block=table_block,
            merge_keys=mk,
            overwrite=overwrite,
            registry_extra=extra,
            partition_partial=partition_partial,
        )

    def register_dataframe(
        self,
        name: str,
        df: pd.DataFrame,
        *,
        overwrite: bool = False,
        registry_extra: dict[str, Any] | None = None,
        table_block: str = BLOCK_RACE_HORSE_TBL,
        merge_keys: list[str] | None = None,
    ) -> Path:
        """任意の DataFrame を特徴量列として保存する。"""
        mk = list(merge_keys or MERGE_KEYS_RACE_HORSE_TBL)
        return self.save_feature_column(
            name,
            df,
            table_block=table_block,
            merge_keys=mk,
            overwrite=overwrite,
            registry_extra=registry_extra,
        )

    # ── 列の読み込み ──────────────────────────────

    def _resolve_column_path(self, name: str) -> Path | None:
        """代表ファイル1つ（互換・削除・キー推定用）。年分割時は最小年。"""
        p = self._registry_primary_path(name)
        if p is not None:
            return p
        paths = self._discover_paths_for_column(name)
        return paths[0] if paths else None

    def list_columns(self) -> list[str]:
        """登録済み特徴量列の一覧（レジストリ優先、無ければブロック走査）。"""
        if self._registry:
            return sorted(self._registry.keys())
        names: set[str] = set()
        for subdir in self._features_dir.iterdir():
            if not subdir.is_dir() or subdir.name in _BLOCK_DIR_SKIP:
                continue
            for f in subdir.glob("*.parquet"):
                names.add(f.stem)
            for ydir in subdir.iterdir():
                if not ydir.is_dir() or not _is_race_year_dir(ydir.name):
                    continue
                for f in ydir.glob("*.parquet"):
                    names.add(f.stem)
        return sorted(names)

    def column_info(self, name: str) -> dict | None:
        """特徴量のメタ情報。"""
        return self._registry.get(name)

    def load_column(self, name: str, years: list[str | int] | None = None) -> pd.DataFrame:
        """単一の特徴量列を読み込む。年分割ストアでは ``years`` 指定で該当年のみ読む。"""
        meta = self._registry.get(name)
        paths: list[Path] = []
        if meta and meta.get("year_partitioned") and meta.get("year_paths"):
            yp = meta["year_paths"]
            use_years = [str(y) for y in years] if years else sorted(yp.keys())
            for y in use_years:
                rel = yp.get(y)
                if not rel:
                    continue
                p = self._features_dir / rel
                if p.is_file():
                    paths.append(p)
        elif meta and meta.get("rel_path"):
            p = self._features_dir / meta["rel_path"]
            if p.is_file():
                paths.append(p)
        else:
            paths = self._discover_paths_for_column(name, years=years)

        if not paths:
            raise FileNotFoundError(f"特徴量なし: {name}")

        dfs = [pd.read_parquet(p) for p in paths]
        out = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        if years and "race_id" in out.columns:
            yset = tuple(str(y) for y in years)
            out = out[out["race_id"].astype(str).str[:4].isin(yset)].reset_index(drop=True)
        return out

    def _merge_keys_for_column(self, name: str) -> list[str]:
        meta = self._registry.get(name)
        if meta and meta.get("merge_keys"):
            return list(meta["merge_keys"])
        p = self._resolve_column_path(name)
        if p is None:
            raise FileNotFoundError(f"特徴量なし: {name}")
        # 旧データ: ファイル位置から推定
        parts = p.parts
        if BLOCK_BASE_TBL in parts:
            from src.pipeline.features.feature_layout import MERGE_KEYS_BASE_TBL

            return list(MERGE_KEYS_BASE_TBL)
        if BLOCK_RACE_TBL in parts:
            from src.pipeline.features.feature_layout import MERGE_KEYS_RACE_TBL

            return list(MERGE_KEYS_RACE_TBL)
        if "race_jockey_tbl" in parts:
            from src.pipeline.features.feature_layout import MERGE_KEYS_RACE_JOCKEY_TBL

            return list(MERGE_KEYS_RACE_JOCKEY_TBL)
        if "race_trainer_tbl" in parts:
            from src.pipeline.features.feature_layout import MERGE_KEYS_RACE_TRAINER_TBL

            return list(MERGE_KEYS_RACE_TRAINER_TBL)
        if BLOCK_RACE_HORSE_TBL in parts:
            return list(MERGE_KEYS_RACE_HORSE_TBL)
        if "jockey_tbl" in parts:
            from src.pipeline.features.feature_layout import MERGE_KEYS_JOCKEY_TBL

            return list(MERGE_KEYS_JOCKEY_TBL)
        if "trainer_tbl" in parts:
            from src.pipeline.features.feature_layout import MERGE_KEYS_TRAINER_TBL

            return list(MERGE_KEYS_TRAINER_TBL)
        if "horse_tbl" in parts:
            from src.pipeline.features.feature_layout import MERGE_KEYS_HORSE_TBL

            return list(MERGE_KEYS_HORSE_TBL)
        return list(MERGE_KEYS_RACE_HORSE_TBL)

    def load_columns(
        self,
        names: list[str],
        years: list[str | int] | None = None,
    ) -> pd.DataFrame:
        """複数の特徴量列を結合キーで結合して返す。

        同一 ``merge_keys`` の列のみサポート。
        """
        if not names:
            return pd.DataFrame()

        keys0 = self._merge_keys_for_column(names[0])
        base_df: pd.DataFrame | None = None
        for name in names:
            keys = self._merge_keys_for_column(name)
            if keys != keys0:
                raise ValueError(
                    f"load_columns は同一マージキーのみ: {names[0]!r} は {keys0}, {name!r} は {keys}"
                )
            col_df = self.load_column(name, years=years)
            if base_df is None:
                base_df = col_df
            else:
                base_df = base_df.merge(col_df, on=keys0, how="outer")

        if base_df is None:
            return pd.DataFrame()

        return base_df

    # ── 学習用マトリクス構築 ──────────────────────────

    def build_training_matrix(
        self,
        feature_cols: list[str],
        target_col: str = "finish_position",
        years: list[str | int] | None = None,
    ) -> pd.DataFrame:
        """学習用の特徴量マトリクスを構築する。"""
        all_cols = list(dict.fromkeys(feature_cols + [target_col]))
        return self.load_columns(all_cols, years=years)

    # ── スナップショット ──────────────────────────────

    def save_snapshot(
        self,
        name: str,
        df: pd.DataFrame,
    ) -> Path:
        """学習用に結合済みの DataFrame をスナップショットとして保存。"""
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)
        path = self._snapshots_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.info("スナップショット保存: %s (%d行 x %d列)", path, len(df), len(df.columns))
        return path

    def load_snapshot(self, name: str) -> pd.DataFrame:
        """スナップショットを読み込む。"""
        path = self._snapshots_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"スナップショットなし: {name}")
        return pd.read_parquet(path)

    def list_snapshots(self) -> list[str]:
        if not self._snapshots_dir.exists():
            return []
        return sorted(f.stem for f in self._snapshots_dir.glob("*.parquet"))

    # ── 内部ヘルパー ──────────────────────────────

    def _save_column_file(
        self,
        name: str,
        df: pd.DataFrame,
        table_block: str,
        part_years: list[str] | None,
    ) -> tuple[Path, dict[str, Any]]:
        """ディスクへ書き込み。戻り値は (代表パス, year_partitioned/year_paths 用の side dict)。"""
        d = self.block_dir(table_block)
        d.mkdir(parents=True, exist_ok=True)
        if not part_years:
            path = d / f"{name}.parquet"
            df.to_parquet(path, index=False)
            logger.info("列保存: %s / %s (%d rows, single)", table_block, name, len(df))
            return path, {}

        s_years = df["race_id"].astype(str).str.slice(0, 4)
        year_paths: dict[str, str] = {}
        primary: Path | None = None
        for y in part_years:
            sub_df = df.loc[s_years == y].copy()
            if sub_df.empty:
                continue
            ydir = d / y
            ydir.mkdir(parents=True, exist_ok=True)
            path = ydir / f"{name}.parquet"
            sub_df.to_parquet(path, index=False)
            year_paths[y] = str(path.relative_to(self._features_dir))
            if primary is None:
                primary = path
        if not year_paths:
            path = d / f"{name}.parquet"
            df.to_parquet(path, index=False)
            logger.warning("年分割対象だが行無し — 単一保存: %s / %s", table_block, name)
            return path, {}

        logger.info(
            "列保存: %s / %s (%d rows, %d year-files)",
            table_block,
            name,
            len(df),
            len(year_paths),
        )
        assert primary is not None
        return primary, {"year_partitioned": True, "year_paths": year_paths}

    def _load_registry(self) -> dict[str, dict]:
        if self._registry_path.exists():
            try:
                return json.loads(self._registry_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _update_registry(self, name: str, meta: dict) -> None:
        meta["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._registry[name] = meta
        self._features_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path.write_text(
            json.dumps(self._registry, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    def delete_column(self, name: str) -> None:
        """特徴量列を削除する。"""
        meta = self._registry.get(name)
        to_unlink: list[Path] = []
        if meta and meta.get("year_paths"):
            for rel in meta["year_paths"].values():
                to_unlink.append(self._features_dir / rel)
        elif meta and meta.get("rel_path"):
            to_unlink.append(self._features_dir / meta["rel_path"])
        for p in self._discover_paths_for_column(name):
            if p not in to_unlink:
                to_unlink.append(p)
        for path in dict.fromkeys(to_unlink):
            if path.is_file():
                path.unlink()
        self._registry.pop(name, None)
        self._features_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path.write_text(
            json.dumps(self._registry, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    def summary(self) -> dict:
        """ストアのサマリーを返す。"""
        return {
            "years": self.available_years(),
            "registered_columns": len(self.list_columns()),
            "snapshots": len(self.list_snapshots()),
            "columns": {
                name: {
                    "source": meta.get("source", "?"),
                    "rows": meta.get("rows", 0),
                    "dtype": meta.get("dtype", "?"),
                    "table_block": meta.get("table_block", "?"),
                }
                for name, meta in self._registry.items()
            },
        }
