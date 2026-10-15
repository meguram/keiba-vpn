#!/usr/bin/env python3
"""取り直せない（取り直すと情報が減る）文字化けデータの、馬名・騎手名・調教師名・性齢・レース名を、正常なデータから ID で復元する。

文字化けは UTF-8 を EUC-JP として読んだ際の置換文字（U+FFFD）で、元の文字には戻せない。ただし馬名などは horse_id / jockey_id /
trainer_id が正しく残っているため、db.netkeiba 由来の正常な結果表（race_result_flat）の対応表から復元できる。それでも残る自由記述
（コメント・印象・パドック評価の文など）は復元できないので、空文字にして ``_meta.mojibake_blanked`` に項目と件数を記録する
（文字化けの文字列が後で特徴量に混ざらないようにするため。元の JSON は data/local/backup/mojibake_docs_offline/ に退避する）。

  python -m src.scripts.maintenance.restore_mojibake_fields --category race_oikiri            # 件数の確認（既定はドライラン）
  python -m src.scripts.maintenance.restore_mojibake_fields --category race_oikiri --apply
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.scripts.maintenance.repair_mojibake_docs import BAD, has_mojibake

logger = logging.getLogger("maintenance.restore_mojibake_fields")
ENTRY_ID_FIELDS = {"horse_name": "horse_id", "jockey_name": "jockey_id", "trainer_name": "trainer_id"}


def build_maps(flat: "Any") -> dict:
    """race_result_flat（pandas）から対応表を作る。"""
    def m(k, v):
        d = flat[[k, v]].dropna()
        d = d[(d[v].astype(str) != "") & ~d[v].astype(str).str.contains(BAD)]
        return dict(zip(d[k].astype(str), d[v].astype(str)))

    maps = {"horse_id": m("horse_id", "horse_name"), "jockey_id": m("jockey_id", "jockey_name"),
            "trainer_id": m("trainer_id", "trainer_name"), "race_name": m("race_id", "race_name")}
    sa = flat[["race_id", "horse_id", "sex_age"]].dropna()
    sa = sa[~sa["sex_age"].astype(str).str.contains(BAD)]
    maps["sex_age"] = {(str(r), str(h)): str(s) for r, h, s in zip(sa["race_id"], sa["horse_id"], sa["sex_age"])}
    return maps


def restore_doc(doc: dict, maps: dict) -> tuple[dict, dict]:
    """(復元後の文書, {'restored': 件数, 'blanked': {項目: 件数}})。元の文書は変更しない。"""
    out = json.loads(json.dumps(doc))
    restored = 0
    blanked: dict[str, int] = {}
    rid = str(out.get("race_id", ""))
    if isinstance(out.get("race_name"), str) and BAD in out["race_name"] and maps["race_name"].get(rid):
        out["race_name"] = maps["race_name"][rid]
        restored += 1
    for e in out.get("entries") or []:
        if not isinstance(e, dict):
            continue
        for name_f, id_f in ENTRY_ID_FIELDS.items():
            if isinstance(e.get(name_f), str) and BAD in e[name_f]:
                v = maps[id_f].get(str(e.get(id_f, "")))
                if v:
                    e[name_f] = v
                    restored += 1
        if isinstance(e.get("sex_age"), str) and BAD in e["sex_age"]:
            v = maps["sex_age"].get((rid, str(e.get("horse_id", ""))))
            if v:
                e["sex_age"] = v
                restored += 1

    def blank(obj: Any, key: str = "") -> Any:
        if isinstance(obj, str):
            if BAD in obj:
                blanked[key] = blanked.get(key, 0) + 1
                return ""
            return obj
        if isinstance(obj, dict):
            return {k: (v if k == "_meta" else blank(v, k)) for k, v in obj.items()}
        if isinstance(obj, list):
            return [blank(v, key) for v in obj]
        return obj

    out = blank(out)
    if blanked:
        meta = dict(out.get("_meta") or {})
        meta["mojibake_blanked"] = blanked
        meta["mojibake_fields_restored_at"] = datetime.now().isoformat(timespec="seconds")
        out["_meta"] = meta
    elif restored:
        meta = dict(out.get("_meta") or {})
        meta["mojibake_fields_restored_at"] = datetime.now().isoformat(timespec="seconds")
        out["_meta"] = meta
    return out, {"restored": restored, "blanked": blanked}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--year", default="2026")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    import pandas as pd

    from src.scraper.storage import HybridStorage

    st = HybridStorage()
    if st._dev_store is not None or not st.gcs_enabled:
        raise SystemExit("GCS に接続できない環境（dev または未設定）です。KEIBA_ENV=stg で実行すること")
    frames = [pd.read_parquet(f"data/page_reference/tables/{y}/race_result_flat.parquet",
                              columns=["race_id", "race_name", "horse_id", "horse_name", "jockey_id", "jockey_name", "trainer_id", "trainer_name", "sex_age"])
              for y in range(2020, 2027)]
    maps = build_maps(pd.concat(frames, ignore_index=True))
    keys = sorted(st.batch_list_blobs(args.category, args.year))
    bak = Path("data/local/backup/mojibake_docs_offline") / args.category
    stats = {"docs": 0, "fully_restored": 0, "with_blanked": 0, "saved": 0, "failed": 0}
    fields: dict[str, int] = {}
    for k in keys:
        doc = st.load(args.category, k, bypass_cache=True)
        if not doc or not has_mojibake(doc):
            continue
        if args.limit and stats["docs"] >= args.limit:
            break
        stats["docs"] += 1
        new, info = restore_doc(doc, maps)
        if info["blanked"]:
            stats["with_blanked"] += 1
            for f, n in info["blanked"].items():
                fields[f] = fields.get(f, 0) + n
        else:
            stats["fully_restored"] += 1
        if args.apply:
            bak.mkdir(parents=True, exist_ok=True)
            (bak / f"{k}.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            if st.save(args.category, k, new):
                stats["saved"] += 1
            else:
                stats["failed"] += 1
                logger.warning("保存できませんでした: %s", k)
    logger.info("[%s] %s 空にした項目 %s", args.category, stats, dict(sorted(fields.items(), key=lambda x: -x[1])[:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
