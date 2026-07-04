"""
2020/1/1 - 2026/5/20 の出走馬に対して:
  Phase 1: 5gen ローカル JSON の存在チェック (欠損リスト出力)
  Phase 2: 10gen JSON 生成 (既存スキップ)

Usage:
    python3 scripts/check_and_build_pedigree_2020_2026.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.pedigree.build_horse_pedigree_10gen import (  # noqa: E402
    build_10gen_record,
    build_all_index,
)

TABLES_DIR = ROOT / "data/local/tables"
PED5_DIR   = ROOT / "data/local/horse_pedigree_5gen"
PED10_DIR  = ROOT / "data/local/horse_pedigree_10gen"
REPORT_PATH = ROOT / "data/local/research/pedigree_5gen_10gen_check_2020_2026.json"
CUTOFF_DATE = "2026-05-20"
YEARS = list(range(2020, 2027))


def prefix_of(horse_id: str) -> str:
    return horse_id[:4] if len(horse_id) >= 4 else "0000"


def path_5gen(horse_id: str) -> Path:
    return PED5_DIR / prefix_of(horse_id) / f"{horse_id}.json"


def path_10gen(horse_id: str) -> Path:
    return PED10_DIR / prefix_of(horse_id) / f"{horse_id}.json"


# ──────────────────────────────────────────────
#  Phase 0: 対象 horse_id 収集
# ──────────────────────────────────────────────

def collect_target_horse_ids() -> set[str]:
    ids: set[str] = set()
    for y in YEARS:
        p = TABLES_DIR / str(y) / "race_result_flat.parquet"
        if not p.exists():
            print(f"  [skip] {p} が存在しません", flush=True)
            continue
        if y < 2026:
            tbl = pq.read_table(p, columns=["horse_id"])
        else:
            tbl = pq.read_table(p, columns=["horse_id", "date"])
            # 2026 は cutoff 以降を除外
            dates = tbl.column("date").to_pylist()
            horse_ids = tbl.column("horse_id").to_pylist()
            for hid, d in zip(horse_ids, dates):
                if d and d <= CUTOFF_DATE and hid:
                    ids.add(str(hid).strip())
            print(f"  {y}: {sum(1 for d in dates if d and d <= CUTOFF_DATE):,} 行 → ユニーク追加", flush=True)
            continue
        for x in tbl.column("horse_id").to_pylist():
            s = str(x or "").strip()
            if s:
                ids.add(s)
        print(f"  {y}: {len(tbl):,} 行", flush=True)
    return ids


# ──────────────────────────────────────────────
#  Phase 1: 5gen チェック
# ──────────────────────────────────────────────

def check_5gen(horse_ids: set[str]) -> tuple[list[str], list[str], list[str]]:
    """
    Returns:
        (found_ok, missing, weak)
        found_ok: 5gen あり ancestors>=5
        missing:  ファイル自体が存在しない
        weak:     ファイルはあるが ancestors<5
    """
    found_ok: list[str] = []
    missing:  list[str] = []
    weak:     list[str] = []

    for hid in sorted(horse_ids):
        p = path_5gen(hid)
        if not p.exists():
            missing.append(hid)
            continue
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            missing.append(hid)
            continue
        if len(rec.get("ancestors") or []) < 5:
            weak.append(hid)
        else:
            found_ok.append(hid)
    return found_ok, missing, weak


# ──────────────────────────────────────────────
#  Phase 2: 10gen 生成
# ──────────────────────────────────────────────

def is_valid_10gen(p: Path) -> bool:
    if not p.exists():
        return False
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
        return rec.get("source") == "10gen_merge" and "generated_at" in rec
    except Exception:
        return False


def build_10gen_for_horses(
    target_ids: list[str],
    idx: dict[str, dict],
) -> dict:
    PED10_DIR.mkdir(parents=True, exist_ok=True)
    counts = {"success": 0, "skipped_existing": 0, "no_5gen": 0, "failed": 0}
    t0 = time.time()
    n = len(target_ids)

    for i, hid in enumerate(target_ids, 1):
        p10 = path_10gen(hid)
        if is_valid_10gen(p10):
            counts["skipped_existing"] += 1
            continue
        if hid not in idx:
            counts["no_5gen"] += 1
            continue
        rec = build_10gen_record(hid, idx)
        if not rec:
            counts["failed"] += 1
            continue
        p10.parent.mkdir(parents=True, exist_ok=True)
        p10.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        counts["success"] += 1
        if i % 2000 == 0:
            el = time.time() - t0
            print(f"  [phase2] {i:,}/{n:,}  success={counts['success']:,}  "
                  f"skipped={counts['skipped_existing']:,}  "
                  f"({el:.0f}s)", flush=True)

    return counts


# ──────────────────────────────────────────────
#  main
# ──────────────────────────────────────────────

def main() -> None:
    t_total = time.time()
    print("=" * 60, flush=True)
    print("Phase 0: 対象 horse_id 収集 (2020-2026/5/20)", flush=True)
    print("=" * 60, flush=True)
    all_ids = collect_target_horse_ids()
    print(f"  → ユニーク出走馬: {len(all_ids):,} 頭\n", flush=True)

    print("=" * 60, flush=True)
    print("Phase 1: 5gen ローカル JSON チェック", flush=True)
    print("=" * 60, flush=True)
    t1 = time.time()
    found_ok, missing, weak = check_5gen(all_ids)
    print(f"  完了 ({time.time()-t1:.1f}s)", flush=True)
    print(f"  5gen あり (ancestors>=5) : {len(found_ok):,}", flush=True)
    print(f"  5gen なし (ファイル欠損)  : {len(missing):,}", flush=True)
    print(f"  5gen 弱 (ancestors<5)    : {len(weak):,}", flush=True)
    if missing:
        print(f"\n  ★ 5gen 未取得 先頭20件:", flush=True)
        for hid in missing[:20]:
            print(f"    {hid}", flush=True)
        if len(missing) > 20:
            print(f"    ... 他 {len(missing)-20:,} 件", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("Phase 2: 10gen JSON 生成 (5gen あり馬のみ、既存スキップ)", flush=True)
    print("=" * 60, flush=True)

    # 5gen があり ancestors>=5 の馬だけを対象
    target_for_10gen = sorted(set(found_ok) | set(weak))
    print(f"  対象: {len(target_for_10gen):,} 頭 (弱含む)", flush=True)
    print(f"  5gen インデックス構築中...", flush=True)
    t2 = time.time()
    idx = build_all_index()
    print(f"  インデックス: {len(idx):,} 馬 ({time.time()-t2:.1f}s)", flush=True)

    t3 = time.time()
    counts = build_10gen_for_horses(target_for_10gen, idx)
    elapsed_10gen = time.time() - t3
    print(f"\n  10gen 生成完了 ({elapsed_10gen:.1f}s)", flush=True)
    print(f"    新規生成    : {counts['success']:,}", flush=True)
    print(f"    既存スキップ: {counts['skipped_existing']:,}", flush=True)
    print(f"    5gen なし   : {counts['no_5gen']:,}", flush=True)
    print(f"    失敗        : {counts['failed']:,}", flush=True)

    # レポート保存
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cutoff_date": CUTOFF_DATE,
        "years": YEARS,
        "phase1": {
            "total_race_horses": len(all_ids),
            "has_5gen_ok": len(found_ok),
            "missing_5gen": len(missing),
            "weak_5gen": len(weak),
            "missing_ids": missing,
            "weak_ids": weak,
        },
        "phase2": {
            "target": len(target_for_10gen),
            **counts,
            "elapsed_sec": round(elapsed_10gen, 1),
        },
        "total_elapsed_sec": round(time.time() - t_total, 1),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nレポート保存: {REPORT_PATH}", flush=True)
    print(f"合計所要時間: {time.time()-t_total:.1f}s", flush=True)


if __name__ == "__main__":
    main()
