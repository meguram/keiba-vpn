"""
horse_training を GCS とは別に ``data/local/horse_training/{shard4}/{horse_id}.json`` に保存する。

シャードは ``pipeline.horse_entity_layout.horse_shard4``（馬ID先頭4文字）で、
``horse_pedigree_5gen`` / ``ped_tbl`` と同じ規則。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def horse_training_local_root(base_dir: str | Path) -> Path:
    return Path(base_dir) / "data" / "local" / "horse_training"


def horse_training_json_path(base_dir: str | Path, horse_id: str) -> Path:
    """``{root}/{shard4}/{horse_id}.json`` の絶対パス。"""
    from pipeline.horse_entity_layout import horse_shard4

    hid = str(horse_id).strip()
    if not hid:
        raise ValueError("horse_id が空です")
    return horse_training_local_root(base_dir) / horse_shard4(hid) / f"{hid}.json"


def write_horse_training_json(
    base_dir: str | Path,
    horse_id: str,
    data: dict[str, Any],
) -> Path:
    """調教 dict を UTF-8 JSON で書き出す。戻り値は出力パス。"""
    from pipeline.horse_entity_layout import horse_shard4

    hid = str(horse_id).strip()
    if not hid:
        raise ValueError("horse_id が空です")
    sh = horse_shard4(hid)
    outp = horse_training_local_root(base_dir) / sh / f"{hid}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_suffix(".json.tmp")
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(outp)
    logger.debug("horse_training ローカル保存: %s", outp)
    return outp
