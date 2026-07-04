#!/usr/bin/env python3
"""
セグメント別に NNLS の残差二乗和を最小化する log 補正（8次元）を推定し、
data/research/race_quality_priors.json に書き出す。

教師ラベルは使わず、「着順との線形整合」を同一セグメント内レースで平均的に良くする
という弱い目的関数。手設計の _hand_segment_archetype_prior に exp(log) を乗算する形。

使用例（リポジトリルートで）:
  python research/tune_race_quality_priors.py --max-races 800 --min-per-segment 30
  python research/tune_race_quality_priors.py --dry-run --max-races 200
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.race.race_quality_model import (  # noqa: E402
    PRIORS_JSON_PATH,
    _column_minmax_nonneg,
    _fit_mixture,
    _hand_segment_archetype_prior,
    _load_priors_json,
    collect_race_xy_tensors,
)

JRA_PLACE = {"01", "02", "03", "04", "05", "06", "07", "08", "09", "10"}


def _iter_race_ids(storage: object, max_dates: int) -> list[str]:
    try:
        dates = sorted(storage.list_keys("race_lists"), reverse=True)[:max_dates]
    except Exception:
        return []
    out: list[str] = []
    for d in dates:
        rl = storage.load("race_lists", d)
        for r in (rl or {}).get("races") or []:
            rid = (r.get("race_id") or "").strip()
            if len(rid) >= 10 and rid[4:6] in JRA_PLACE:
                out.append(rid)
    return out


def _sample_loss(
    log_extra: np.ndarray,
    samples: list[dict],
    hand_w: np.ndarray,
    reg: float,
) -> float:
    mult = np.exp(np.clip(log_extra, -1.0, 1.0))
    seg_w = hand_w * mult
    total = 0.0
    for s in samples:
        base = s["base_rows"]
        y = s["y"]
        pw = s["pace_w"]
        gw = s["going_w"]
        rs = seg_w * pw * gw
        x_raw = base * rs
        x = _column_minmax_nonneg(x_raw)
        _, _, res = _fit_mixture(x, y)
        total += float(res) ** 2
    return total + reg * float(np.sum(log_extra ** 2))


def main() -> int:
    ap = argparse.ArgumentParser(description="レース質セグメント事前のデータ駆動チューニング")
    ap.add_argument("--max-races", type=int, default=600, help="走査する最大レース数")
    ap.add_argument("--max-dates", type=int, default=500, help="race_lists を遡る最大日数")
    ap.add_argument("--min-per-segment", type=int, default=25, help="最適化に必要なセグメント内最小レース数")
    ap.add_argument("--reg", type=float, default=0.025, help="log 補正の L2 正則化")
    ap.add_argument("--dry-run", action="store_true", help="JSON を書かず損失のみ表示")
    ap.add_argument("--output", type=str, default="", help="出力 JSON（既定: data/research/race_quality_priors.json）")
    args = ap.parse_args()

    try:
        from src.scraper.storage import HybridStorage
    except ImportError:
        print("scraper.storage を読めません。リポジトリルートで実行してください。", file=sys.stderr)
        return 1

    storage = HybridStorage(base_dir=str(ROOT))
    stats_data = None
    try:
        from src.research.pedigree.sire_factor_stats import load_sire_factor_stats

        stats_data = load_sire_factor_stats(storage=storage)
    except Exception as e:
        print(f"警告: sire stats 読込失敗（血統0埋めで続行）: {e}", file=sys.stderr)

    race_ids = _iter_race_ids(storage, args.max_dates)[: args.max_races * 2]
    by_seg: dict[str, list[dict]] = defaultdict(list)
    n_ok = 0
    for rid in race_ids:
        if n_ok >= args.max_races:
            break
        t = collect_race_xy_tensors(storage, rid, stats_data=stats_data)
        if not t:
            continue
        by_seg[t["segment_key"]].append(
            {
                "race_id": rid,
                "base_rows": t["base_rows"],
                "y": t["y"],
                "pace_w": t["pace_w"],
                "going_w": t["going_w"],
            }
        )
        n_ok += 1

    print(f"収集: 有効レース {n_ok} / 走査上限 {args.max_races} · セグメント数 {len(by_seg)}")

    existing = _load_priors_json()
    prev_mult = dict(existing.get("segment_extra_log_mult") or {})
    new_mult: dict[str, list[float]] = dict(prev_mult)

    bounds = [(-1.0, 1.0)] * 8
    x0_base = np.zeros(8, dtype=np.float64)

    for seg, samples in sorted(by_seg.items(), key=lambda x: -len(x[1])):
        if len(samples) < args.min_per_segment:
            continue
        hand = _hand_segment_archetype_prior(seg)
        x0 = x0_base.copy()
        if seg in prev_mult and len(prev_mult[seg]) == 8:
            try:
                x0 = np.array([float(x) for x in prev_mult[seg]], dtype=np.float64)
            except (TypeError, ValueError):
                pass

        def obj(x: np.ndarray) -> float:
            return _sample_loss(x, samples, hand, args.reg)

        res = minimize(
            obj,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 80, "ftol": 1e-7},
        )
        if not res.success:
            print(f"  [{seg}] 最適化警告: {res.message} (n={len(samples)})")
        loss_before = obj(x0)
        loss_after = float(res.fun)
        new_mult[seg] = [round(float(x), 6) for x in res.x]
        print(
            f"  [{seg}] n={len(samples)} loss {loss_before:.6f} -> {loss_after:.6f} "
            f"log={new_mult[seg]}"
        )

    out_path = Path(args.output) if args.output else PRIORS_JSON_PATH
    payload = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "segment_extra_log_mult": new_mult,
        "meta": {
            "max_races_cap": args.max_races,
            "max_dates_scanned": args.max_dates,
            "min_per_segment": args.min_per_segment,
            "regularization": args.reg,
            "collected_races": n_ok,
        },
    }

    if args.dry_run:
        print("dry-run: ファイルは書きません")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"書き込み: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
