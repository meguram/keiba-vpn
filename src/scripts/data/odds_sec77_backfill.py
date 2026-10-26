"""過去の開催日（2026 年）の想定オッズを §77 のモデル（sec77）で算出し、最終オッズ予測のキャッシュへ保存する（バックフィル／差分確認）。

  KEIBA_INFERENCE_ENGINE=full python -m src.scripts.data.odds_sec77_backfill --from 20261017 --to 20261024 --dry-run   # 保存せず比較だけ
  KEIBA_INFERENCE_ENGINE=full nice -n 19 python -m src.scripts.data.odds_sec77_backfill --from 20261017 --to 20261024 # 保存する

特徴量は backfill_predictions と同じ手順（その日より前の履歴 + 同日に先に発走したレースの結果）で v6 のサービング経路から作る。
比較: sec77 / 従来の想定（キャッシュがあれば）/ 確定オッズ（race_odds がある日）の対数誤差を日別に出す（data/local/meta/odds_sec77_compare.json）。
モデルは 2020〜2025 年で学習済みなので、2026 年は未学習の期間（検証として使える）。
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger("scripts.odds_sec77_backfill")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="since", default="20260101")
    ap.add_argument("--to", dest="until", default=datetime.date.today().strftime("%Y%m%d"))
    ap.add_argument("--dry-run", action="store_true", help="保存せず、比較だけ出す")
    ap.add_argument("--legacy-compare", action="store_true", help="従来の最終オッズ予測も（キャッシュが無ければその場で推論して）比較に加える。保存はしない")
    ap.add_argument("--max-days", type=int, default=0)
    ap.add_argument("--max-load", type=float, default=18.0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    from src.pipeline.inference import odds_sec77 as os77
    from src.pipeline.inference import weight_triggered_predict as wtp
    from src.pipeline.inference.final_odds_service import load_cached_response
    from src.pipeline.inference.v3_serving import V3Serving, v6_paths
    from src.scraper.storage import HybridStorage
    from src.scripts.data.backfill_predictions import race_days, wait_for_load

    if not os77.available():
        raise SystemExit("sec77 のモデルがありません（odds_sec77_fit fit を先に実行）")
    pth = v6_paths()
    storage = HybridStorage()
    tables = Path(os.environ.get("KEIBA_TABLES_DIR") or "data/page_reference/tables")
    years = sorted({int(a.since[:4]) + i for i in range(int(a.until[:4]) - int(a.since[:4]) + 1)})
    days = race_days(years, tables, a.since, a.until)
    ctx = wtp.InferenceContext()
    ctx.warm()
    assert ctx.raw_hist is not None, "KEIBA_INFERENCE_ENGINE=full で実行する"
    raw_full = ctx.raw_hist
    v3 = V3Serving(pth["bundle"], pth["stage"], tl_reference=pth["tl_reference"])
    v3.rb_state = pth["rb_state"]
    v3.warm()
    hist_full = v3.history.copy()
    report: dict = {}
    done = 0
    for day, ids in days.items():
        if a.max_days and done >= a.max_days:
            break
        wait_for_load(a.max_load)
        d = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        t0 = time.time()
        try:
            card = wtp.fill_card_meta(wtp.card_frame(storage, ids), [])
            ok_ids = (card.filter(pl.col("horse_number").fill_null(0) > 0).filter(pl.col("distance").cast(pl.Float64, strict=False) > 0)
                      .filter(pl.col("surface").cast(pl.Utf8).fill_null("") != "")["race_id"].unique().to_list())
            card = card.filter(pl.col("race_id").is_in(ok_ids))
            if card.height == 0:
                continue
            cut = datetime.date.fromisoformat(d) if raw_full.schema["date"] == pl.Date else d
            ctx.raw_hist = raw_full.filter(pl.col("date") < cut)
            F, _ = ctx.build(card)
            errs = {"sec77": [], "legacy": []}
            saved = 0
            for rid in sorted(ok_ids):
                cr = card.filter(pl.col("race_id") == rid)
                st = str(cr["start_time"][0])
                h = hist_full[(hist_full["date"].astype(str) < d)
                              | ((hist_full["date"].astype(str) == d) & (hist_full["start_time"].astype(str) < st) & (hist_full["race_id"].astype(str) != rid))]
                X = v3.features(cr, F.filter(pl.col("race_id") == rid), storage, history=h)
                R = os77.predict_frame(X)
                cp = cr.to_pandas()
                hn = {str(hh): int(n) for hh, n in zip(cp["horse_id"], cp["horse_number"].fillna(0)) if n and int(n) > 0}
                name = str(cp["race_name"].iloc[0]) if "race_name" in cp.columns else ""
                payload = os77.payload_for_race(R, hn, name)
                if payload is None:
                    continue
                act = {int(e["horse_number"]): float(e["win_odds"]) for e in ((storage.load("race_odds", rid) or {}).get("entries") or []) if e.get("win_odds") and float(e["win_odds"]) > 0}
                leg = load_cached_response(storage, rid)
                if leg is None and a.legacy_compare:
                    try:
                        from src.pipeline.inference.final_odds_service import build_final_odds_response

                        leg = build_final_odds_response(rid, storage)
                    except Exception:  # noqa: BLE001
                        leg = None
                leg_map = {int(e["horse_number"]): float(e["predicted_win_odds"]) for e in (leg or {}).get("entries") or []
                           if e.get("source") != "sec77" and e.get("predicted_win_odds")}
                for e in payload["entries"]:
                    n = e["horse_number"]
                    if n in act:
                        errs["sec77"].append(np.log(e["predicted_win_odds"]) - np.log(act[n]))
                        if n in leg_map:
                            errs["legacy"].append(np.log(leg_map[n]) - np.log(act[n]))
                if not a.dry_run and os77.save_for_race(storage, payload):
                    saved += 1
            rep = {k: ({"n": len(v), "rmse": float(np.sqrt(np.mean(np.square(v)))), "bias": float(np.mean(v))} if v else None) for k, v in errs.items()}
            rep.update({"races": len(ok_ids), "saved": saved, "sec": round(time.time() - t0)})
            report[day] = rep
            logger.info("%s: %s", day, json.dumps(rep, ensure_ascii=False))
            done += 1
        except Exception:  # noqa: BLE001
            logger.exception("%s: 算出に失敗（次の日へ）", day)
    out = Path("data/local/meta/odds_sec77_compare.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
