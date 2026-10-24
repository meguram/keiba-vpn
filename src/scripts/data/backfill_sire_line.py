"""
既存の horse_pedigree_5gen JSON から sires テーブルを一括バックフィルする。

処理フロー
----------
1. data/local/horse_pedigree_5gen/ 以下の全 JSON を走査
2. gen=1 (父・母) の祖先を抽出し、牡馬 (sex='牡') のみを種牡馬として収集
3. sire_line を以下の優先順位で解決:
   a. data/local/research/pedigree_race_index/stallion_lineage.parquet
      → anchor_name + "系" を sire_line として採用
   b. src/research/pedigree/bloodline_vector.py の SIRE_LINES ハードコード辞書
   c. None（次回スクレイピング時に HTML から補完される）
4. sires テーブルへ UPSERT（sire_line が既に入っている行は上書きしない）

Usage
-----
  # 全ローカル JSON を対象
  python -m src.scripts.data.backfill_sire_line

  # dry-run（DB 書き込みなし）
  python -m src.scripts.data.backfill_sire_line --dry-run

  # 進捗確認のみ（DB の充足率レポート）
  python -m src.scripts.data.backfill_sire_line --report-only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.keiba_logging import script_basic_config

logger = logging.getLogger("backfill_sire_line")

LOCAL_PED_DIR   = _ROOT / "data" / "local" / "horse_pedigree_5gen"
STALLION_LIN    = _ROOT / "data" / "local" / "research" / "pedigree_race_index" / "stallion_lineage.parquet"
BATCH_COMMIT    = 500


# ── 系統ルックアップ構築 ─────────────────────────────────────────────


def _build_lineage_lookup() -> tuple[dict[str, str], dict[str, str]]:
    """stallion_id → 系統名 の辞書を構築する。

    1. stallion_lineage.parquet の anchor_name を優先
    2. SIRE_LINES ハードコード辞書をフォールバック
    """
    lookup: dict[str, str] = {}

    # 1. stallion_lineage.parquet
    if STALLION_LIN.exists():
        try:
            import pandas as pd
            lin = pd.read_parquet(STALLION_LIN)
            for _, row in lin.iterrows():
                sid = str(row.get("stallion_id", "")).strip()
                anchor = str(row.get("anchor_name", "")).strip()
                if sid and anchor:
                    line_name = anchor if anchor.endswith("系") else anchor + "系"
                    lookup[sid] = line_name
            logger.info("stallion_lineage.parquet から %d 頭のルックアップを構築", len(lookup))
        except Exception as exc:
            logger.warning("stallion_lineage.parquet 読み込み失敗: %s", exc)

    # 2. SIRE_LINES ハードコード辞書（名前ベース補完用）
    _name_to_line: dict[str, str] = {}
    try:
        from src.research.pedigree.bloodline_vector import SIRE_LINES
        for line_name, names in SIRE_LINES.items():
            for name in names:
                _name_to_line[name.strip()] = line_name
        logger.info("SIRE_LINES から %d 頭の名前マッピングを取得", len(_name_to_line))
    except Exception as exc:
        logger.warning("SIRE_LINES 読み込み失敗: %s", exc)

    return lookup, _name_to_line


# ── JSON 走査 ───────────────────────────────────────────────────────


def _iter_local_jsons():
    """LOCAL_PED_DIR 以下の .json ファイルをすべて yield する。"""
    if not LOCAL_PED_DIR.exists():
        logger.warning("LOCAL_PED_DIR が見つかりません: %s", LOCAL_PED_DIR)
        return
    yield from LOCAL_PED_DIR.rglob("*.json")


def _extract_sires_from_json(data: dict[str, Any]) -> list[dict[str, Any]]:
    """horse_pedigree_5gen JSON から gen=1 の祖先を返す。"""
    results = []
    for anc in data.get("ancestors", []):
        if anc.get("generation") != 1:
            continue
        horse_id = (anc.get("horse_id") or "").strip()
        name = (anc.get("name") or "").strip()
        sex = anc.get("sex", "")
        if horse_id and name:
            results.append({"horse_id": horse_id, "name": name, "sex": sex})
    return results


# ── メイン処理 ────────────────────────────────────────────────────


def run(dry_run: bool = False, report_only: bool = False) -> None:
    from src.db.etl.upsert_sires import upsert_sire
    from src.db.session import get_session, init_engine

    init_engine()

    # ── レポートのみ ──
    if report_only:
        with get_session() as sess:
            from sqlalchemy import text
            total_r   = sess.execute(text("SELECT COUNT(*) FROM sires")).scalar()
            has_line  = sess.execute(
                text("SELECT COUNT(*) FROM sires WHERE sire_line IS NOT NULL AND sire_line != ''")
            ).scalar()
        pct = has_line / total_r * 100 if total_r else 0
        print(f"sires テーブル: {total_r:,} 件 / sire_line 有り: {has_line:,} 件 ({pct:.1f}%)")
        return

    id_lookup, name_lookup = _build_lineage_lookup()

    json_files = sorted(_iter_local_jsons())
    logger.info("対象 JSON: %d ファイル", len(json_files))
    if not json_files:
        logger.info("対象 JSON がありません。")
        return

    collected: dict[str, dict] = {}
    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        for anc in _extract_sires_from_json(data):
            hid = anc["horse_id"]
            if hid not in collected:
                collected[hid] = anc

    logger.info("ユニーク種牡馬候補: %d 頭", len(collected))

    if not collected:
        logger.info("種牡馬候補が見つかりませんでした。")
        return

    n_insert = 0
    n_skip   = 0
    t0 = time.time()

    items = list(collected.items())

    if dry_run:
        for hid, anc in items[:20]:
            line = id_lookup.get(hid) or name_lookup.get(anc["name"])
            print(f"  {anc['name']:<30} ({hid})  sire_line={line or '(なし)'}")
        print(f"... dry-run: {len(items)} 頭分をスキップ")
        return

    with get_session() as sess:
        for i, (hid, anc) in enumerate(items):
            sire_line = id_lookup.get(hid) or name_lookup.get(anc["name"])
            upsert_sire(sess, hid, anc["name"], sire_line)
            n_insert += 1

            if (i + 1) % BATCH_COMMIT == 0:
                sess.commit()
                elapsed = time.time() - t0
                logger.info(
                    "進捗: %d/%d (%.0f件/秒)",
                    i + 1, len(items), (i + 1) / max(elapsed, 0.1),
                )

        sess.commit()

    elapsed = time.time() - t0
    logger.info(
        "完了: INSERT/UPDATE=%d  スキップ=%d  (%.1f秒)",
        n_insert, n_skip, elapsed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="sires テーブル バックフィル")
    parser.add_argument("--dry-run", action="store_true", help="DB に書き込まずサンプル表示")
    parser.add_argument("--report-only", action="store_true", help="充足率レポートのみ表示")
    args = parser.parse_args()

    script_basic_config()
    run(dry_run=args.dry_run, report_only=args.report_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
