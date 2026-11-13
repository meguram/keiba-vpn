"""GCS の保存データ（JSON）をカテゴリごとに抽出し、項目ごとの空・文字化けの割合を基準年と直近で比べる。

flat テーブルにしていないカテゴリ（オッズ・追い切り・パドック・指数・バロメーター・厩舎コメントなど）の、
パース失敗（ページ表記の変更・文字コードの誤判定）を見つけるための検査。スキーマ検証は「項目があるか・型が合うか」しか見ないため、
空文字や 0 のまま保存された値は通ってしまう。
  python -m src.scraper.parse_drift_docs [--categories race_odds,race_oikiri] [--sample 60]

標本の選び方（2026-11-13 から。GCS の一覧は取らない＝CLAUDE.md の費用の決まり）:
  - 直近の年: ローカルの表（そのカテゴリの _race.parquet、無ければ race_result_flat）の race_id を、レースの日付の新しい順に
    sample*3 件取り、3 件おきに間引いて名前で読む。以前は GCS の一覧の「最後に保存した時刻」の新しい順だったので、
    古いレースを保存し直した分（取り直しの品質）は標本に入らなくなった。
  - 基準の年: 同じ表から無作為（種はカテゴリと年で固定）に sample*2 件を名前で読み、項目ごとの割合をローカルに保存して 30 日使い回す
    （BASELINE_VERSION を上げるか 30 日たつと作り直す。過去年の文書の取り直し・数え方の変更を拾うため）。
  - 毎日読むのは直近の sample 件 × カテゴリ数だけ（12 カテゴリ・40 件で 480 件、約 3MB）。
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


def _tables_dir():
    from src.scraper.export_tables import OUTPUT_DIR

    return OUTPUT_DIR


def local_sample_ids(category: str, year: str, n: int, newest: bool, rng: random.Random, tables_dir=None) -> list[str]:
    """ローカルの表から標本の race_id を選ぶ（GCS の一覧は取らない）。"""
    from src.scraper.export_tables import _flat_dates, _race_table_ids_or_empty

    base = (tables_dir or _tables_dir()) / str(year)
    dates = _flat_dates(base / "race_result_flat.parquet")
    for c in ("race_shutuba", "race_result_on_time"):
        for r, d in _flat_dates(base / f"{c}_flat.parquet").items():
            dates.setdefault(r, d)
    ids = _race_table_ids_or_empty(base / f"{category}_race.parquet") or set(dates)
    if not ids:
        return []
    if newest:
        keys = sorted(ids, key=lambda r: (dates.get(r, ""), r))
        return keys[-n * 3:][::3][:n]   # 直近の開催日にまたがるよう間引く
    return rng.sample(sorted(ids), min(len(ids), n * 2))


def sample_docs(storage: Any, category: str, year: str, n: int, newest: bool, rng: random.Random, tables_dir=None) -> list[dict]:
    docs = []
    for k in local_sample_ids(category, year, n, newest, rng, tables_dir):
        d = storage.load(category, k, bypass_cache=True)
        if isinstance(d, dict):
            docs.append(d)
    return docs


BASELINE_VERSION = 1           # field_rates の数え方を変えたら上げる（保存した基準を作り直す）
BASELINE_MAX_AGE_DAYS = 30     # 過去年の文書の取り直し・パーサの変更を拾うため、保存した基準もこの日数で作り直す


def _baseline_path(category: str, year: str, n: int):
    from src.scraper.date_coverage import NOT_AVAILABLE_DIR

    return NOT_AVAILABLE_DIR.parent / "parse_drift_docs_baseline" / f"{category}_{year}_{n}_v{BASELINE_VERSION}.json"


def baseline_rates(storage: Any, category: str, year: str, n: int, rng: random.Random, tables_dir=None) -> tuple[dict[str, tuple[float, float]], int]:
    """基準の年の項目ごとの割合。保存があれば読み直さない（基準の年の文書は変わらない）。(割合, 文書数) を返す。"""
    import json
    import os

    import time

    path = _baseline_path(category, year, n)
    try:
        if time.time() - path.stat().st_mtime > BASELINE_MAX_AGE_DAYS * 86400:
            raise OSError("古い")
        saved = json.loads(path.read_text(encoding="utf-8"))
        return {k: (float(v[0]), float(v[1])) for k, v in saved["rates"].items()}, int(saved["docs"])
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        pass
    docs = sample_docs(storage, category, year, n, False, rng, tables_dir)
    rates = field_rates(docs) if docs else {}
    if len(docs) >= n:          # 少なすぎる標本（取り損ね・GCS の不調）は保存しない
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"docs": len(docs), "rates": rates}, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            pass
    return rates, len(docs)


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
    total = 0
    for cat in [c.strip() for c in args.categories.split(",") if c.strip()]:
        rng = random.Random(f"{cat}:{args.baseline_year}")    # カテゴリごとに固定（前のカテゴリの保存の有無で標本が変わらない）
        base, n_base = baseline_rates(st, cat, args.baseline_year, args.sample, rng)
        rec = sample_docs(st, cat, args.recent_year, args.sample, True, rng)
        if not n_base or not rec:
            print(f"[{cat}] データなし（基準 {n_base} / 直近 {len(rec)}）")
            continue
        f = compare(base, field_rates(rec), args.threshold)
        total += len(f)
        print(f"[{cat}] 基準 {n_base} 件 / 直近 {len(rec)} 件 → " + ("異常なし" if not f else f"{len(f)} 項目"))
        for x in f:
            print(f"    {x['field']:34s} {x['kind']}  基準 {x['baseline']:.3f} → 直近 {x['recent']:.3f}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
