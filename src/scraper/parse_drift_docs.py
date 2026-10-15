"""GCS の保存データ（JSON）をカテゴリごとに抽出し、項目ごとの空・文字化けの割合を基準年と直近で比べる。

flat テーブルにしていないカテゴリ（オッズ・追い切り・パドック・指数・バロメーター・厩舎コメントなど）の、
パース失敗（ページ表記の変更・文字コードの誤判定）を見つけるための検査。スキーマ検証は「項目があるか・型が合うか」しか見ないため、
空文字や 0 のまま保存された値は通ってしまう。
  python -m src.scraper.parse_drift_docs [--categories race_odds,race_oikiri] [--sample 60]
"""

from __future__ import annotations

import argparse
import random
import sys
from typing import Any

BAD = "�"
RACE_CATEGORIES = [
    "race_result", "race_result_lap", "race_result_on_time", "race_shutuba", "race_shutuba_past", "race_odds", "race_pair_odds",
    "race_oikiri", "race_paddock", "race_barometer", "race_index", "race_trainer_comment",
]
SKIP_FIELDS = {"_meta"}


def _empty(v: Any) -> bool:
    return v is None or v == "" or (isinstance(v, float) and v != v) or v == [] or v == {}


def _mojibake(v: Any) -> bool:
    return isinstance(v, str) and BAD in v


def field_rates(docs: list[dict]) -> dict[str, tuple[float, float]]:
    """項目名 -> (空の割合, 文字化けの割合)。トップの項目は文書単位、entries の項目は文書ごとの割合の平均。"""
    acc: dict[str, list[tuple[float, float]]] = {}
    for d in docs:
        for k, v in d.items():
            if k in SKIP_FIELDS:
                continue
            if k == "entries" and isinstance(v, list):
                rows = [e for e in v if isinstance(e, dict)]
                if not rows:
                    acc.setdefault("entries[]", []).append((1.0, 0.0))
                    continue
                keys = {kk for e in rows for kk in e}
                for kk in keys:
                    acc.setdefault(f"entries.{kk}", []).append(
                        (sum(_empty(e.get(kk)) for e in rows) / len(rows), sum(_mojibake(e.get(kk)) for e in rows) / len(rows)))
                continue
            if isinstance(v, (list, dict)) and not _empty(v):
                continue   # 構造化された中身（ラップ・払戻など）は存在だけ見る
            acc.setdefault(k, []).append((float(_empty(v)), float(_mojibake(v))))
    return {k: (sum(a for a, _ in xs) / len(xs), sum(b for _, b in xs) / len(xs)) for k, xs in acc.items() if len(xs) >= max(5, len(docs) // 4)}


def compare(base: dict[str, tuple[float, float]], recent: dict[str, tuple[float, float]], threshold: float = 0.15) -> list[dict]:
    out = []
    for k, (e, m) in recent.items():
        be, bm = base.get(k, (None, None))
        if be is None:
            continue
        if e - be > threshold:
            out.append({"field": k, "kind": "空", "baseline": round(be, 3), "recent": round(e, 3)})
        if m - bm > threshold:
            out.append({"field": k, "kind": "文字化け", "baseline": round(bm, 3), "recent": round(m, 3)})
    return out


def sample_docs(storage: Any, category: str, year: str, n: int, newest: bool, rng: random.Random) -> list[dict]:
    stamps = storage.batch_list_blobs(category, year)
    if not stamps:
        return []
    # 直近は「最近保存された文書」（今の取得処理の品質を見る）。基準年は無作為
    keys = sorted(stamps, key=lambda k: stamps[k]) if newest else sorted(stamps)
    picked = keys[-n * 3:] if newest else rng.sample(keys, min(len(keys), n * 2))
    if newest:
        picked = picked[::3][:n]   # 直近の開催日にまたがるよう間引く
    docs = []
    for k in picked:
        d = storage.load(category, k, bypass_cache=True)
        if isinstance(d, dict):
            docs.append(d)
    return docs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default=",".join(RACE_CATEGORIES))
    ap.add_argument("--baseline-year", default="2025")
    ap.add_argument("--recent-year", default="2026")
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--threshold", type=float, default=0.15)
    args = ap.parse_args()

    from src.utils.project_env import load_project_dotenv
    from src.scraper.storage import HybridStorage

    load_project_dotenv()
    st = HybridStorage()
    rng = random.Random(0)
    total = 0
    for cat in [c.strip() for c in args.categories.split(",") if c.strip()]:
        base = sample_docs(st, cat, args.baseline_year, args.sample, False, rng)
        rec = sample_docs(st, cat, args.recent_year, args.sample, True, rng)
        if not base or not rec:
            print(f"[{cat}] データなし（基準 {len(base)} / 直近 {len(rec)}）")
            continue
        f = compare(field_rates(base), field_rates(rec), args.threshold)
        total += len(f)
        print(f"[{cat}] 基準 {len(base)} 件 / 直近 {len(rec)} 件 → " + ("異常なし" if not f else f"{len(f)} 項目"))
        for x in f:
            print(f"    {x['field']:34s} {x['kind']}  基準 {x['baseline']:.3f} → 直近 {x['recent']:.3f}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
