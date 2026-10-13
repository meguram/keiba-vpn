"""特定の開催日を実データで取得し、dev ストアに取り込む（モックの一部を実データに置き換える）。

dev 環境（KEIBA_ENV=dev）専用。HybridStorage は dev では GCS ではなく ``data/dev_mock``
（local_only カテゴリは ``data/page_reference``）へ保存するので、既存のモックはそのまま残り、
指定日のレース分だけが実データになる。実データ以外（馬ごとの統計・血統・AI 予測など）はモックのまま。

  - 取得するのは UI が使うレース系のデータ（出馬表・オッズ・2連系/3連複・結果・指数・追い切り・馬柱・厩舎コメント・
    パドック・レース詳細）。馬ごとのデータ（成績・血統・調教）は 1 頭 20 秒超かかるため既定では取らない（--with-horses）。
  - AI 予測（race_predictions）は実モデルが無いため、実際の出走馬に対してモック予測を生成する
    （``_meta.mock_prediction=true``。実データの取り込み日の一部なので make dev-mock --clean の削除対象 ``dev_mock`` にはしない）。
  - 既に保存済みのデータはスキップするので、中断しても同じコマンドで再開できる（--force で取り直し）。

使い方（実アクセスあり。1 日 36 レースで約 30 分）:
  python3 -m src.scripts.scraping.capture_dev_day --date 20260906 --yes
  python3 -m src.scripts.scraping.capture_dev_day --date 20260906 --max-races 2 --yes   # お試し
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np
from typing import Any, Callable

logger = logging.getLogger("capture_dev_day")

# (表示名, 保存カテゴリ, runner 呼び出し)。失敗しても次へ進む。
RACE_STEPS: list[tuple[str, str, Callable[[Any, str, str, bool], Any]]] = [
    ("出馬表", "race_shutuba", lambda r, rid, d, sk: r.scrape_race_card(rid, skip_existing=sk)),
    ("単複オッズ", "race_odds", lambda r, rid, d, sk: r.scrape_odds(rid, skip_existing=sk)),
    ("2連系・3連複", "race_pair_odds", lambda r, rid, d, sk: r.scrape_pair_odds(rid, skip_existing=sk)),
    ("結果", "race_result", lambda r, rid, d, sk: r.scrape_race_result(rid, skip_existing=sk)),
    ("結果(速報)", "race_result_on_time",
     lambda r, rid, d, sk: r.scrape_race_result_on_time(rid, skip_existing=sk, opening_date=d)),
    ("結果ラップ", "race_result_lap", lambda r, rid, d, sk: r.scrape_race_result_lap(rid, skip_existing=sk)),
    ("速度指数", "race_index", lambda r, rid, d, sk: r.scrape_speed_index(rid, skip_existing=sk)),
    ("追い切り", "race_oikiri", lambda r, rid, d, sk: r.scrape_oikiri(rid, skip_existing=sk)),
    ("馬柱", "race_shutuba_past", lambda r, rid, d, sk: r.scrape_shutuba_past(rid, skip_existing=sk)),
    ("厩舎コメント", "race_trainer_comment", lambda r, rid, d, sk: r.scrape_trainer_comment(rid, skip_existing=sk)),
    ("パドック", "race_paddock", lambda r, rid, d, sk: r.scrape_paddock(rid, skip_existing=sk)),
    ("レース詳細", "race_detail", lambda r, rid, d, sk: r.scrape_race_detail(rid)),
]

HORSE_STEPS: list[tuple[str, str, Callable[[Any, str, bool], Any]]] = [
    ("馬 過去成績", "horse_result", lambda r, hid, sk: r.scrape_horse(hid, skip_existing=sk)),
    ("馬 5世代血統", "horse_pedigree_5gen", lambda r, hid, sk: r.scrape_horse_pedigree_5gen(hid, skip_existing=sk)),
    ("馬 調教", "horse_training", lambda r, hid, sk: r.scrape_horse_training(hid, skip_existing=sk)),
]


MOCK_ALPHA2, MOCK_ALPHA3 = 0.75, 0.70


def build_mock_prediction(rid: str, card: dict, odds: dict | None, result_meta: dict | None = None) -> dict | None:
    """実際の出走馬・実オッズから、UI/馬券推奨が読む形式のモック予測を作る（決定論的）。

    モック予測モデル = 市場の暗黙確率（1/単勝オッズ）に正規ノイズを乗せたスコア。実モデルではない。
    確率はスコアの softmax + Harville、期待値 = 確率 × 実オッズ（複勝は下限オッズ）。
    """
    import math
    import random

    from src.pipeline.inference.exotic_probabilities import RankDistribution

    entries = [e for e in (card.get("entries") or []) if e.get("horse_number") is not None]
    omap = {int(o["horse_number"]): o for o in (odds or {}).get("entries", []) if o.get("horse_number") is not None}
    if len(entries) < 4 or not omap:
        return None
    rng = random.Random(int(rid))
    rows = []
    for e in entries:
        hn = int(e["horse_number"])
        o = omap.get(hn, {})
        wo = float(o.get("win_odds") or 0)
        if wo <= 0:
            continue
        rows.append((e, hn, wo, float(o.get("place_odds_min") or 0), float(o.get("place_odds_max") or 0)))
    if len(rows) < 4:
        return None
    implied = [1.0 / r[2] for r in rows]
    z = sum(implied)
    scores = [math.log(p / z) + rng.gauss(0, 0.35) for p in implied]
    ex = np.exp(np.array(scores) - max(scores))
    win = ex / ex.sum()
    # 連対率・複勝率は割引 Harville（2・3 着ほど荒れる）で作る。市場の実測（2026-09-06, 36R）では α≈0.72 前後。
    # Harville そのもの（α=1）より穴馬の連対・複勝が高く出る、実際の傾向に近いモック
    rd = RankDistribution(win, MOCK_ALPHA2, MOCK_ALPHA3)
    t2, t3 = rd.top2(), rd.top3()
    probs = [{"win_prob": float(win[i]), "top2_prob": float(t2[i]), "top3_prob": float(t3[i])} for i in range(len(scores))]
    mean = sum(scores) / len(scores)
    sd = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5 or 1.0

    order = sorted(range(len(rows)), key=lambda i: -probs[i]["win_prob"])
    rank = {i: k + 1 for k, i in enumerate(order)}
    ev_win = [probs[i]["win_prob"] * rows[i][2] for i in range(len(rows))]
    ev_pl = [probs[i]["top3_prob"] * rows[i][3] if rows[i][3] else None for i in range(len(rows))]
    show_val = max((i for i in range(len(rows)) if rank[i] > 3 and ev_pl[i]), key=lambda i: ev_pl[i], default=None)
    star = max((i for i in range(len(rows)) if rank[i] > 3 and rows[i][2] >= 10 and i != show_val),
               key=lambda i: ev_win[i], default=None)
    marks = {1: "honmei", 2: "pair", 3: "anchor"}

    preds = []
    for i, (e, hn, wo, pmin, pmax) in enumerate(rows):
        evw, evp = ev_win[i], ev_pl[i]
        mark = marks.get(rank[i]) or ("show_val" if i == show_val else "star" if i == star else "none")
        best = max(evw, evp or 0.0)
        preds.append({
            "horse_number": hn, "horse_name": e.get("horse_name", f"#{hn}"), "horse_id": e.get("horse_id", ""),
            "pred_score": round(scores[i], 4), "normalized_score": round(50 + 10 * (scores[i] - mean) / sd, 2),
            "pred_rank": rank[i], "composite_rank": rank[i], "composite_score": round(scores[i], 4),
            "win_prob": round(probs[i]["win_prob"], 4), "top2_prob": round(probs[i]["top2_prob"], 4),
            "top3_prob": round(probs[i]["top3_prob"], 4), "win_odds": wo,
            "place_odds_min": pmin or None, "place_odds_max": pmax or None,
            "ev_win": round(evw, 3), "ev_place": round(evp, 3) if evp is not None else None,
            "mark_type": mark, "buy_tier": "A" if best >= 1.3 else "B" if best >= 1.0 else "C",
        })
    preds.sort(key=lambda x: x["pred_rank"])
    meta = {**(card or {}), **{k: v for k, v in (result_meta or {}).items() if v not in (None, "", 0)}}
    return {
        "race_id": rid, "status": "success", "has_prediction": True,
        "race_name": meta.get("race_name", ""), "round": int(meta.get("round") or int(rid[-2:])),
        "venue": meta.get("venue", ""), "surface": meta.get("surface", ""), "distance": int(meta.get("distance") or 0),
        "track_condition": meta.get("track_condition", ""), "total_horses": len(preds), "elapsed_sec": 0.0,
        "model_type": "dev-mock", "model_version": "dev-mock-v1",
        "model_description": "開発用モック予測（市場オッズ+ノイズ。実モデルではありません）",
        "predictions": preds, "_meta": {"mock_prediction": True},
    }


def _save_mock_predictions(storage, rid: str, force: bool) -> bool:
    if not force and storage.exists("race_predictions", rid):
        return False
    card = storage.load("race_shutuba", rid) or {}
    payload = build_mock_prediction(rid, card, storage.load("race_odds", rid), storage.load("race_result", rid))
    if not payload:
        return False
    # 本番と同じ推論キャッシュ形式（_cache_meta 付き）で保存する。source=dev_mock は dev で期限切れにならない
    from src.pipeline.inference.race_prediction_service import save_cached

    return bool(save_cached(storage, rid, payload, source="dev_mock"))


def build_mock_final_odds(rid: str, race_name: str, odds: dict | None) -> dict | None:
    """予測オッズ（最終オッズ予測のキャッシュ）のモック。実オッズに ±のずれを乗せた値（モデルではない）。"""
    import random

    if not odds or not odds.get("entries"):
        return None
    rng = random.Random(int(rid) + 1)
    entries = []
    for o in odds["entries"]:
        wo = float(o.get("win_odds") or 0)
        if wo <= 0:
            continue
        f = rng.uniform(0.8, 1.25)
        pmin, pmax = float(o.get("place_odds_min") or 0), float(o.get("place_odds_max") or 0)
        entries.append({
            "horse_number": int(o["horse_number"]), "predicted_win_odds": round(wo * f, 1),
            "predicted_place_odds_min": round(pmin * f ** 0.5, 1) if pmin else 0.0,
            "predicted_place_odds_max": round(pmax * f ** 0.5, 1) if pmax else 0.0, "source": "dev_mock",
        })
    return {"race_id": rid, "race_name": race_name, "entries": entries, "elapsed_sec": 0.0,
            "_meta": {"mock_prediction": True}} if entries else None


def _save_mock_final_odds(storage, rid: str, force: bool) -> bool:
    if not force and storage.exists("final_odds_prediction", rid):
        return False
    payload = build_mock_final_odds(rid, (storage.load("race_shutuba", rid) or {}).get("race_name", ""), storage.load("race_odds", rid))
    if not payload:
        return False
    from src.pipeline.inference.final_odds_service import FinalOddsCache

    return bool(FinalOddsCache.save_cached(storage, rid, payload, source="dev_mock"))


def _save_derived(storage, rid: str) -> int:
    """本番と同じ抽出ロジックで派生カテゴリ（race_shutuba_meta など）を作る。"""
    from src.scraper.row_data_extractor import DERIVED_CATEGORY_MAP

    n = 0
    for derived, (parent, fn) in DERIVED_CATEGORY_MAP.items():
        if not parent.startswith("race_"):
            continue
        try:
            data = storage.load(parent, rid)
            if data and storage.save(derived, rid, fn(data)):
                n += 1
        except Exception as e:  # noqa: BLE001
            logger.warning("派生 %s/%s を作れません: %s", derived, rid, e)
    return n


def capture(date: str, *, max_races: int | None, force: bool, with_horses: bool, gap: float) -> dict[str, Any]:
    from src.scraper.run import ScraperRunner

    runner = ScraperRunner()
    storage = runner.storage
    skip = not force

    races = runner.scrape_race_list(date) or []
    if max_races:
        races = races[:max_races]
    ids = [r["race_id"] for r in races if r.get("race_id")]
    print(f"{date}: {len(ids)} レースを取り込みます", flush=True)

    stats: dict[str, Any] = {"races": len(ids), "failed": {}, "derived": 0, "predictions": 0}
    t_all = time.time()
    for i, rid in enumerate(ids, 1):
        t0 = time.time()
        failed: list[str] = []
        for label, cat, fn in RACE_STEPS:
            try:
                ok = bool(fn(runner, rid, date, skip))
            except Exception as e:  # noqa: BLE001
                ok = False
                logger.warning("%s %s: %s", rid, label, e)
            # 速度指数・パドック等は元データ自体が無いレースがある。保存済みなら成功扱い
            if not ok and not storage.exists(cat, rid):
                failed.append(label)
        card = storage.load("race_shutuba", rid) or storage.load("race_result", rid) or {}
        entries = card.get("entries") or card.get("results") or []
        horse_ids = [e["horse_id"] for e in entries if e.get("horse_id")]
        if _save_mock_predictions(storage, rid, force):
            stats["predictions"] += 1
        _save_mock_final_odds(storage, rid, force)
        stats["derived"] += _save_derived(storage, rid)
        if with_horses:
            for hid in horse_ids:
                for label, cat, fn in HORSE_STEPS:
                    try:
                        fn(runner, hid, skip)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("%s %s: %s", hid, label, e)
        if failed:
            stats["failed"][rid] = failed
        print(f"[{i}/{len(ids)}] {rid} {time.time() - t0:5.1f}s 馬{len(horse_ids)}頭"
              + (f" 未取得: {','.join(failed)}" if failed else ""), flush=True)
        time.sleep(gap)

    # 発走時刻表（UI の週間予測・監視が参照）。出馬表の発走時刻から合成する
    try:
        from src.scraper.race_day_schedule import synthesize_race_day_schedule_payload

        payload = synthesize_race_day_schedule_payload(storage, date)
        storage.save("race_day_schedule", date, payload)
    except Exception as e:  # noqa: BLE001
        print(f"race_day_schedule を作れませんでした: {e}", file=sys.stderr)

    stats["elapsed_sec"] = round(time.time() - t_all)
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True, help="開催日 YYYYMMDD")
    ap.add_argument("--max-races", type=int, help="先頭 N レースだけ（お試し）")
    ap.add_argument("--force", action="store_true", help="保存済みも取り直す")
    ap.add_argument("--with-horses", action="store_true", help="馬ごとのデータ（成績・血統・調教）も取る（非常に遅い）")
    ap.add_argument("--gap", type=float, default=1.0, help="レース間の追加待機（秒）")
    ap.add_argument("--yes", action="store_true", help="実アクセスを許可")
    args = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    logging.basicConfig(level=logging.ERROR)

    if (os.environ.get("KEIBA_ENV") or "").strip().lower() != "dev":
        print("KEIBA_ENV=dev 以外では実行しません（stg/prod だと GCS に書き込まれます）", file=sys.stderr)
        return 2
    if not args.yes:
        print("実アクセスを伴います。--yes を指定してください。", file=sys.stderr)
        return 2

    stats = capture(args.date, max_races=args.max_races, force=args.force, with_horses=args.with_horses, gap=args.gap)
    print(f"完了: {stats['races']} レース / モック予測 {stats['predictions']} / 派生 {stats['derived']} / {stats['elapsed_sec']}秒")
    if stats["failed"]:
        print(f"一部未取得 {len(stats['failed'])} レース（元データが無い可能性。詳細は上記行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
