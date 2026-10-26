"""開催日の朝の下準備（Jobs ``day-prep``）: その日の全レースの「日単位のキャッシュ」だけを作り、予測は作らない。

    python -m src.ops.jobs.day_prep --date YYYYMMDD

v6e の特徴量には、馬体重に依存しない「日単位」の部分（v4 の馬主・馬自身の群 = ``state/<日>/v4_day.parquet``、v5 の PX = ``state/<日>/px_day.parquet``）がある。
その日の最初の 1 レースがこれを作ると、同じプロセスの残りのレースは使い回せる。ところが Jobs はレースごとにコンテナが新しくなるので、そのたびに作り直すことになり、
1 レース目と同じ重さ（ホスト実測: ピーク RSS 約 12GB・約 90 秒）が毎回かかって 8GiB で OOM になる（2026-10-26）。
そこで、朝に 1 回だけ大きな Run Job（16GiB）でこの 2 つを作って ``state/`` へ上げ、``predict-race`` は ``state/`` から読むだけにする
（実測: 2 レース目以降の相当で、ピーク RSS 約 4.1GB・約 31 秒）。予測（predictions_v6）は作らない＝馬体重の前に予測を出さない方針を守る。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def run(date: str, storage: Any | None = None, ctx: Any | None = None) -> dict[str, Any]:
    import polars as pl

    if storage is None:
        from src.scraper.storage import HybridStorage

        storage = HybridStorage(".")
    from src.pipeline.inference import race_day_workflow as wf
    from src.pipeline.inference import weight_triggered_predict as wtp

    rows = wf.load_day_schedule(storage, date)
    if not rows:
        return {"status": "skipped", "reason": "no-races", "date": date}
    if ctx is None:
        ctx = wtp.InferenceContext()
        info = ctx.warm()
        if (info.get("v6") or {}).get("error"):
            raise RuntimeError(f"v6 の準備に失敗: {info['v6']['error']}")
    if getattr(ctx, "v6", None) is None:
        raise RuntimeError("v6 が有効でない（KEIBA_PRED_MODEL / KEIBA_V6_SHADOW を確認）")
    ids = [r["race_id"] for r in rows]
    card = wtp.fill_card_meta(wtp.card_frame(storage, ids), rows)
    if card.height == 0:
        return {"status": "error", "reason": "出馬表の行がありません", "date": date}
    # 面・距離が空のレース、馬番が全頭未確定のレースは除く（run_for_races と同じ扱い）
    bad = set(card.filter(~(pl.col("distance").cast(pl.Float64, strict=False) > 0) | pl.col("surface").cast(pl.Utf8).fill_null("").eq(""))["race_id"].unique().to_list())
    bad |= set(card.group_by("race_id").agg((pl.col("horse_number").fill_null(0) > 0).sum().alias("k")).filter(pl.col("k") == 0)["race_id"].to_list())
    card = card.filter(~pl.col("race_id").is_in(list(bad)))
    if card.height == 0:
        return {"status": "error", "reason": "使える出馬表がありません（面・距離または馬番が未確定）", "date": date}
    F, _info = ctx.build(card)
    ctx.v6.features(card, F, storage)                       # v4_day / px_day が state/<日>/ に保存される（v3_serving._px_for_day・serving_v4.day_groups）
    day = str(card["date"].min()).replace("/", "-")
    sd = Path(ctx.v6.v4.state_dir) / day
    made = {n: (sd / n).exists() for n in ("v4_day.parquet", "px_day.parquet")}
    return {"status": "ok" if all(made.values()) else "error", "date": date, "races": card["race_id"].n_unique(), "rows": card.height, "excluded": sorted(bad), "files": made}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ns = ap.parse_args(argv)
    if len(ns.date) != 8 or not ns.date.isdigit():
        print("date は YYYYMMDD", file=sys.stderr)
        return 2
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    res = run(ns.date)
    print(json.dumps(res, ensure_ascii=False, default=str))
    return 0 if res["status"] in ("ok", "skipped") else 1


if __name__ == "__main__":
    sys.exit(main())
