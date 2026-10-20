"""過去の開催日の追走難度 v2 を、保存済みの出馬表・馬柱から作って保存する（前夜の保存の穴埋め・過去分の評価用）。

python -m src.scripts.data.tracking_v2_backfill --from 20260101 --to 20261018            # 保存済みのレースは飛ばす
python -m src.scripts.data.tracking_v2_backfill --date 20260906 --overwrite              # 作り直す
python -m src.scripts.data.tracking_v2_backfill --from A --to B --dry-run                # 対象の数だけ数える

この CLI は明示的な手動実行なので、KEIBA_TRACKING_V2 が無くても有効にして動く（API や eve の挙動は変えない）。
開催日は race_result_flat（確定結果の表）にある日。出力先は KEIBA_TRACKING_V2_DIR（既定 data/local/inference/tracking_v2）。
ホストでは `nice -n 19` か `/tmp/heavy_run.sh <ラベル> …` 経由で直列に実行する（馬柱の読み込みが主で、計算は 1 レース 0.3 秒ほど）。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import pandas as pd

logger = logging.getLogger("tracking_v2_backfill")


def race_days(frm: str, to: str) -> dict[str, list[str]]:
    """確定結果の表にある開催日 → race_id の一覧。"""
    from src.pipeline.inference.tracking_v2_eval import _iso, tables_dir

    out: dict[str, list[str]] = {}
    years = range(int(frm[:4]), int(to[:4]) + 1)
    for y in years:
        f = tables_dir() / str(y) / "race_result_flat.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f, columns=["race_id", "date"]).drop_duplicates("race_id")
        d["date"] = d["date"].astype(str).str.replace("/", "-").str[:10]
        d = d[(d["date"] >= _iso(frm)) & (d["date"] <= _iso(to))]
        for day, g in d.groupby("date"):
            out[day] = sorted(g["race_id"].astype(str))
    return dict(sorted(out.items()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date")
    ap.add_argument("--from", dest="frm")
    ap.add_argument("--to")
    ap.add_argument("--overwrite", action="store_true", help="保存済みのレースも作り直す")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="作るレース数の上限（試し用）")
    a = ap.parse_args(argv)
    frm, to = (a.date, a.date) if a.date else (a.frm, a.to)
    if not frm or not to:
        ap.error("--date か --from/--to を指定してください")
    frm, to = frm.replace("-", ""), to.replace("-", "")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("KEIBA_TRACKING_V2", "1")       # 明示的な手動実行なので有効にする
    from src.pipeline.inference.tracking_v2_service import base_dir, precompute_tracking_v2_for_race_ids

    days = race_days(frm, to)
    total = sum(len(v) for v in days.values())
    logger.info("対象: %d 開催日 / %d レース → %s", len(days), total, base_dir())
    if a.dry_run:
        print(json.dumps({"days": len(days), "races": total, "out": str(base_dir())}, ensure_ascii=False))
        return 0
    from src.scraper.storage import HybridStorage

    storage = HybridStorage()
    agg = {"ok": 0, "skip": 0, "fail": 0, "total": 0}
    left = a.limit
    for day, rids in days.items():
        if a.limit:
            if left <= 0:
                break
            rids = rids[:left]
        st = precompute_tracking_v2_for_race_ids(rids, storage, source="backfill", overwrite=a.overwrite)
        for k in agg:
            agg[k] += int(st.get(k, 0))
        left -= st.get("ok", 0)
        logger.info("%s: %s（累計 %s）", day, {k: st.get(k) for k in ("ok", "skip", "fail", "total")}, agg)
    print(json.dumps({"days": len(days), **agg}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
