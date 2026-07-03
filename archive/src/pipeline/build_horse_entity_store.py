"""
プロジェクト内に現れる全 ``horse_id`` を母集団に、馬単位 Parquet を ``data/features/horse/`` に構築する。

- ``ped_tbl``: ``data/local/horse_pedigree_5gen/{horse_id 先頭4文字}/{horse_id}.json`` があればロング表化。
  **牡（種牡馬）スロットの行のみ**（牝スロットは行として出さない）。ツリー位置は ``path_fm`` と
  ``generation`` / ``position`` / ``merged_global_depth`` で保持し、``ancestor_horse_id`` は当該牡スロットの
  種牡馬 ID のみ。現行スクレイパは **5世代**（62祖先）; JSON が伸びれば **最大10世代** まで同方針で載る。
  **クロス列**（父系/母系/全体の重複・濃度・両系統出現・主馬単位集計）は ``pipeline.features.horse_pedigree_cross`` を参照。
- ``result_tbl``: ``data/local/tables/*/race_result_flat.parquet`` を縦結合し、馬ごとに全レース行を分割保存。
  ネスト列 ``payoff`` / ``lap_times`` / ``pace`` はレース単位で別管理できるため **出力しない**。
- ``training_tbl``: ``--training-json-dir`` を指定し ``{horse_id}.json`` が存在する馬のみ書き出し
  （未指定時はスキップし manifest に記録）。

  python -m src.pipeline.build_horse_entity_store --overwrite
  python -m src.pipeline.build_horse_entity_store --max-horses 5000 --overwrite
  python -m src.pipeline.build_horse_entity_store --overwrite --merge-gen5-sires

``--overwrite`` 時のみ既存 ``data/features/horse`` 配下の対象 stem を置換（部分更新は未対応）。
``--merge-gen5-sires`` 時は ``ped_tbl`` に 5 世代目牡の枝を接木した行を載せ、``path_fm`` 等を付与する（``docs/html/modeling/horse_pedigree_10gen_merge_design.html``）。
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from src.pipeline.features.feature_store import TABLES_DIR
from src.pipeline.features.horse_entity_layout import (
    HORSE_ENTITY_ROOT,
    PED_TBL,
    RESULT_TBL,
    TRAINING_TBL,
    horse_entity_root,
    horse_shard4,
    is_male_pedigree_slot_upto,
    ped_parquet_path,
    result_parquet_path,
    training_parquet_path,
)
from src.pipeline.features.horse_pedigree_cross import PED_CROSS_COLUMN_NAMES, add_pedigree_cross_columns
from src.pipeline.features.horse_pedigree_expand import (
    fm_path_from_gp,
    iter_gen5_male_anchor_horse_ids,
    merge_primary_and_branches,
)
from src.pipeline.features.id_value_policy import sanitize_netkeiba_string_id
from src.utils.keiba_logging import script_basic_config

logger = logging.getLogger(__name__)

MANIFEST_NAME = "_manifest.json"

# race_result_flat にあっても result_tbl から除く（レースキー別ストア向けのネストデータ）
RESULT_TBL_DROP_NESTED_COLS: frozenset[str] = frozenset({"payoff", "lap_times", "pace"})

# ped_tbl: 牡スロットのみ・位置は path_fm / generation / position で表す（merge 有無で共通）
PED_TBL_LONG_COLUMNS: list[str] = [
    "subject_horse_id",
    "subject_horse_name",
    "generation",
    "position",
    "path_fm",
    "merged_global_depth",
    "source",
    "anchor_horse_id",
    "ancestor_horse_id",
    "ancestor_name",
    "is_male_pedigree_slot",
    "smartrc_code",
    "country",
    "pedigree_max_generation_observed",
    *PED_CROSS_COLUMN_NAMES,
]


def _iter_tables_years(tables_dir: Path) -> list[str]:
    if not tables_dir.is_dir():
        return []
    return sorted(d.name for d in tables_dir.iterdir() if d.is_dir() and d.name.isdigit())


def collect_horse_ids_from_tables(tables_dir: Path, years: list[str] | None) -> set[str]:
    ys = years if years is not None else _iter_tables_years(tables_dir)
    out: set[str] = set()
    for y in ys:
        for stem, cols in (
            ("race_result", ["horse_id"]),
            ("race_shutuba", ["horse_id"]),
        ):
            p = tables_dir / y / f"{stem}_flat.parquet"
            if not p.is_file():
                continue
            try:
                sch = pq.read_schema(p)
                if "horse_id" not in sch.names:
                    continue
                hid = pq.read_table(p, columns=["horse_id"]).to_pandas()["horse_id"]
                hid = sanitize_netkeiba_string_id(hid)
                out.update(hid.dropna().astype(str).unique().tolist())
            except Exception as e:
                logger.warning("horse_id 収集スキップ %s: %s", p, e)
    return out


def _flatten_pedigree_json(rec: dict[str, Any], subject_horse_id: str) -> pd.DataFrame:
    """5〜10世代想定の ancestors を、牡スロットのみのロング形式に（牝行は出さない）。"""
    sub_name = str(rec.get("horse_name") or rec.get("name") or "")
    rows: list[dict[str, Any]] = []
    for anc in rec.get("ancestors") or []:
        try:
            g = int(anc.get("generation", 0))
            pos = int(anc.get("position", 0))
        except (TypeError, ValueError):
            continue
        if not is_male_pedigree_slot_upto(g, pos):
            continue
        path = fm_path_from_gp(g, pos)
        rows.append(
            {
                "subject_horse_id": str(subject_horse_id),
                "subject_horse_name": sub_name or pd.NA,
                "generation": g,
                "position": pos,
                "path_fm": path,
                "merged_global_depth": len(path),
                "source": "primary",
                "anchor_horse_id": pd.NA,
                "ancestor_horse_id": anc.get("horse_id") or pd.NA,
                "ancestor_name": anc.get("name") or pd.NA,
                "is_male_pedigree_slot": 1,
                "smartrc_code": anc.get("smartrc_code", "") or pd.NA,
                "country": anc.get("country", "") or pd.NA,
            }
        )
    if not rows:
        return pd.DataFrame(columns=PED_TBL_LONG_COLUMNS)
    df = pd.DataFrame(rows)
    df["pedigree_max_generation_observed"] = int(df["generation"].max())
    df["ancestor_horse_id"] = sanitize_netkeiba_string_id(df["ancestor_horse_id"].astype("string"))
    df["anchor_horse_id"] = sanitize_netkeiba_string_id(df["anchor_horse_id"].astype("string"))
    if not df.empty:
        df = add_pedigree_cross_columns(df)
    return df[PED_TBL_LONG_COLUMNS]


def _flatten_pedigree_json_with_gen5_merge(
    rec: dict[str, Any],
    subject_horse_id: str,
    pedigree_dir: Path,
) -> pd.DataFrame:
    """主表 + 5 世代目牡のアンカーごとの 5 世代枝を接木（最大論理深さ 10）。設計: docs/html/modeling/horse_pedigree_10gen_merge_design.html"""
    sub_name = str(rec.get("horse_name") or rec.get("name") or "")
    ancestors: list[dict[str, Any]] = list(rec.get("ancestors") or [])
    branch_by_anchor: dict[str, list[dict[str, Any]]] = {}
    for anchor_id, _ in iter_gen5_male_anchor_horse_ids(ancestors):
        if str(anchor_id) == str(subject_horse_id):
            continue
        br = _load_pedigree_json(pedigree_dir, anchor_id)
        if br:
            branch_by_anchor[str(anchor_id)] = list(br.get("ancestors") or [])

    merged_rows = merge_primary_and_branches(
        str(subject_horse_id),
        ancestors,
        branch_by_anchor,
        subject_horse_name=sub_name,
    )
    if not merged_rows:
        return pd.DataFrame(columns=PED_TBL_LONG_COLUMNS)

    rows_out: list[dict[str, Any]] = []
    for r in merged_rows:
        try:
            g = int(r.get("generation", 0))
            p = int(r.get("position", 0))
        except (TypeError, ValueError):
            continue
        rows_out.append(
            {
                "subject_horse_id": str(subject_horse_id),
                "subject_horse_name": (r.get("subject_horse_name") or sub_name or pd.NA),
                "generation": g,
                "position": p,
                "path_fm": r.get("path_fm"),
                "merged_global_depth": r.get("merged_global_depth"),
                "source": r.get("source"),
                "anchor_horse_id": r.get("anchor_horse_id"),
                "ancestor_horse_id": r.get("horse_id"),
                "ancestor_name": r.get("name"),
                "is_male_pedigree_slot": 1,
                "smartrc_code": r.get("smartrc_code", "") or pd.NA,
                "country": r.get("country", "") or pd.NA,
            }
        )
    df = pd.DataFrame(rows_out)
    if df.empty:
        return pd.DataFrame(columns=PED_TBL_LONG_COLUMNS)
    max_gen = int(df["generation"].max())
    df["pedigree_max_generation_observed"] = max_gen
    df["ancestor_horse_id"] = sanitize_netkeiba_string_id(df["ancestor_horse_id"].astype("string"))
    df["anchor_horse_id"] = sanitize_netkeiba_string_id(df["anchor_horse_id"].astype("string"))
    if not df.empty:
        df = add_pedigree_cross_columns(df)
    return df[PED_TBL_LONG_COLUMNS]


def _load_pedigree_json(pedigree_dir: Path, horse_id: str) -> dict[str, Any] | None:
    sh = horse_shard4(horse_id)
    p = pedigree_dir / sh / f"{horse_id}.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("血統 JSON 読み込み失敗 %s: %s", p, e)
        return None


def _flatten_training_json(rec: dict[str, Any], horse_id: str) -> pd.DataFrame:
    """horse_training ストア相当の dict を行に展開（entries リスト）。"""
    entries = rec.get("entries") or rec.get("race_history") or []
    if not isinstance(entries, list):
        return pd.DataFrame()
    rows = []
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            continue
        row = {"horse_id": str(horse_id), "entry_index": i}
        for k, v in e.items():
            row[str(k)] = v
        rows.append(row)
    if not rows:
        return pd.DataFrame({"horse_id": [str(horse_id)]})
    return pd.DataFrame(rows)


def _load_training_json(training_dir: Path, horse_id: str) -> dict[str, Any] | None:
    """既定: ``training_dir/{shard4}/{horse_id}.json`` または ``training_dir/{horse_id}.json``。"""
    sh = horse_shard4(horse_id)
    for p in (training_dir / sh / f"{horse_id}.json", training_dir / f"{horse_id}.json"):
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("調教 JSON 読み込み失敗 %s: %s", p, e)
                return None
    return None


def build_horse_entity_store(
    *,
    base_dir: str | Path = ".",
    years: list[str] | None = None,
    pedigree_json_dir: Path | None = None,
    training_json_dir: Path | None = None,
    max_horses: int | None = None,
    overwrite: bool = False,
    merge_gen5_sires: bool = False,
) -> dict[str, Any]:
    base = Path(base_dir)
    tables_dir = base / TABLES_DIR
    root = horse_entity_root(base)

    horse_ids = sorted(collect_horse_ids_from_tables(tables_dir, years))
    if max_horses is not None:
        horse_ids = horse_ids[: int(max_horses)]

    if overwrite and root.is_dir():
        shutil.rmtree(root)
    (root / PED_TBL).mkdir(parents=True, exist_ok=True)
    (root / RESULT_TBL).mkdir(parents=True, exist_ok=True)
    (root / TRAINING_TBL).mkdir(parents=True, exist_ok=True)

    # --- result: 全 race_result を一度に読んで groupby ---
    ys = years if years is not None else _iter_tables_years(tables_dir)
    rr_frames: list[pd.DataFrame] = []
    for y in ys:
        p = tables_dir / y / "race_result_flat.parquet"
        if not p.is_file():
            continue
        try:
            rr_frames.append(pq.read_table(p).to_pandas())
        except Exception as e:
            logger.warning("race_result 読み込みスキップ %s: %s", p, e)
    result_written = 0
    rr_all = pd.concat(rr_frames, ignore_index=True) if rr_frames else pd.DataFrame()
    if not rr_all.empty and "horse_id" in rr_all.columns:
        rr_all["horse_id"] = sanitize_netkeiba_string_id(rr_all["horse_id"])
        rr_all = rr_all.dropna(subset=["horse_id"])
        rr_all["horse_id"] = rr_all["horse_id"].astype(str)
        drop_nest = [c for c in RESULT_TBL_DROP_NESTED_COLS if c in rr_all.columns]
        if drop_nest:
            rr_all = rr_all.drop(columns=drop_nest)
    cols_rr = (
        list(rr_all.columns)
        if not rr_all.empty
        else ["race_id", "horse_id", "finish_position"]
    )
    by_horse: dict[str, pd.DataFrame] = (
        {str(k): v for k, v in rr_all.groupby("horse_id", sort=False)} if not rr_all.empty else {}
    )
    for hid in horse_ids:
        sub = by_horse.get(str(hid), pd.DataFrame(columns=cols_rr))
        outp = result_parquet_path(hid, base)
        outp.parent.mkdir(parents=True, exist_ok=True)
        sub.to_parquet(outp, index=False)
        result_written += 1

    ped_dir = Path(pedigree_json_dir) if pedigree_json_dir else base / "data" / "local" / "horse_pedigree_5gen"
    ped_written = 0
    ped_missing_json = 0
    for hid in horse_ids:
        rec = _load_pedigree_json(ped_dir, hid) if ped_dir.is_dir() else None
        if not rec:
            ped_missing_json += 1
            continue
        if merge_gen5_sires:
            pdf = _flatten_pedigree_json_with_gen5_merge(rec, hid, ped_dir)
        else:
            pdf = _flatten_pedigree_json(rec, hid)
        outp = ped_parquet_path(hid, base)
        outp.parent.mkdir(parents=True, exist_ok=True)
        pdf.to_parquet(outp, index=False)
        ped_written += 1

    if training_json_dir is not None:
        train_dir = Path(training_json_dir)
    else:
        _td = base / "data" / "local" / "horse_training"
        train_dir = _td if _td.is_dir() else None
    training_written = 0
    training_missing = 0
    training_skipped_no_dir = train_dir is None or not train_dir.is_dir()
    if train_dir and train_dir.is_dir():
        for hid in horse_ids:
            tr = _load_training_json(train_dir, hid)
            if not tr:
                training_missing += 1
                continue
            tdf = _flatten_training_json(tr, hid)
            outp = training_parquet_path(hid, base)
            outp.parent.mkdir(parents=True, exist_ok=True)
            tdf.to_parquet(outp, index=False)
            training_written += 1

    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "layout_version": 1,
        "root": str(HORSE_ENTITY_ROOT),
        "sharding": "ped_tbl|result_tbl|training_tbl/{horse_id[:4]}/{horse_id}.parquet",
        "pedigree_note": (
            "ped_tbl は牡（種牡馬）スロットのみを1行1ノードのロング表。牝スロットは行として持たず、"
            "ツリー位置は path_fm・generation・position・merged_global_depth で表す。ancestor_horse_id は当該牡スロットの種牡馬 ID。"
            "現行 netkeiba 血統は 5 世代（62祖先）; JSON が伸びれば最大10世代まで同方針。--merge-gen5-sires で論理深さ最大10の枝接木。"
            "クロス列: ped_root_side, ancestor_occurrence_*, ancestor_cross_both_roots, ancestor_pct_*, subject_* 。"
        ),
        "horse_id_universe": {
            "count": len(horse_ids),
            "sources": ["race_result_flat", "race_shutuba_flat"],
            "years": ys,
        },
        "result_tbl": {
            "written": result_written,
            "note": "race_result に一度も出ていない馬は空の DataFrame（列のみ）。payoff/lap_times/pace は含めない。",
        },
        "ped_tbl": {
            "json_source": str(ped_dir),
            "written": ped_written,
            "missing_json": ped_missing_json,
            "merge_gen5_sires": bool(merge_gen5_sires),
        },
        "training_tbl": {
            "json_source": str(train_dir) if train_dir else None,
            "written": training_written,
            "missing_json_per_horse": training_missing,
            "skipped_no_input_dir": training_skipped_no_dir,
        },
    }
    mp = root / MANIFEST_NAME
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "horse entity store: horses=%d result=%d ped=%d training=%d → %s",
        len(horse_ids),
        result_written,
        ped_written,
        training_written,
        root,
    )
    return manifest


def main() -> int:
    script_basic_config()
    ap = argparse.ArgumentParser(description="data/local/features/horse/ に馬単位 Parquet を構築")
    ap.add_argument("--base-dir", type=Path, default=Path("."))
    ap.add_argument("--years", nargs="*", help="tables の対象年（省略で全年）")
    ap.add_argument("--pedigree-json-dir", type=Path, default=None, help="既定: data/local/horse_pedigree_5gen")
    ap.add_argument(
        "--training-json-dir",
        type=Path,
        default=None,
        help="調教 JSON ルート（省略時、data/local/horse_training が存在すれば自動使用）",
    )
    ap.add_argument("--max-horses", type=int, default=None, help="デバッグ用の頭数上限")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--merge-gen5-sires",
        action="store_true",
        help="5世代目の種牡馬ごとの5世代枝を接木し path_fm 等を付与（論理深さ最大10）",
    )
    args = ap.parse_args()
    try:
        build_horse_entity_store(
            base_dir=args.base_dir,
            years=list(args.years) if args.years else None,
            pedigree_json_dir=args.pedigree_json_dir,
            training_json_dir=args.training_json_dir,
            max_horses=args.max_horses,
            overwrite=args.overwrite,
            merge_gen5_sires=args.merge_gen5_sires,
        )
    except Exception as e:
        logger.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
