"""保存済み horse_result（race_history）の通過順・上り3F を、race_result_flat 由来の補正（scraper.horse_result_fix のオーバーレイ）で恒久的に直す。

python -m src.scripts.maintenance.bake_horse_result_fix --dry-run            # 影響件数と先頭の前後比較だけ
python -m src.scripts.maintenance.bake_horse_result_fix --apply --workers 3  # 退避してから保存（HybridStorage.save）

- 補正は冪等（オーバーレイは race_result_flat の値。同じ値なら変更なし）。overlay に無い走（2019 年以前など）は触らない。
- 変更のある馬だけ、元の文書を data/local/backup/horse_result_fix_<日時>/ へ退避してから保存する。結果は data/local/meta/horse_result_bake_report.json。
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from src.scraper import horse_result_fix as hf
from src.scraper.storage import HybridStorage

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "data/local/meta/horse_result_bake_report.json"
CAT = "horse_result"


def _changed_rows(old: dict, new: dict) -> list[tuple[str, tuple, tuple]]:
    out = []
    for a, b in zip(old.get("race_history") or [], new.get("race_history") or []):
        if (a.get("passing_order"), a.get("last_3f")) != (b.get("passing_order"), b.get("last_3f")):
            out.append((str(a.get("race_id")), (a.get("passing_order"), a.get("last_3f")), (b.get("passing_order"), b.get("last_3f"))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="先頭 N 頭だけ（試験用）")
    a = ap.parse_args()
    if a.apply == a.dry_run:
        ap.error("--apply か --dry-run のどちらか一方を指定")
    s = HybridStorage()
    lookup = hf.load_lookup()
    if not lookup:
        print("overlay が空/無効（KEIBA_HORSE_RESULT_FIX=0 か overlay.parquet 無し）")
        return 2
    keys = s.list_keys(CAT)
    if a.limit:
        keys = keys[: a.limit]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    bdir = ROOT / f"data/local/backup/horse_result_fix_{stamp}"
    stats = {"horses": len(keys), "changed_horses": 0, "changed_rows": 0, "saved": 0, "errors": 0, "no_doc": 0, "rows_in_overlay": 0}
    samples: list[dict] = []

    def work(hid: str) -> None:
        try:
            d = s.load(CAT, hid)
            if not isinstance(d, dict) or not d.get("race_history"):
                stats["no_doc"] += 1
                return
            stats["rows_in_overlay"] += sum(1 for e in d["race_history"] if (hid, str(e.get("race_id") or "")) in lookup)
            new = hf.apply_overlay(d, lookup)
            ch = _changed_rows(d, new)
            if not ch:
                return
            stats["changed_horses"] += 1
            stats["changed_rows"] += len(ch)
            if len(samples) < 20:
                samples.append({"horse_id": hid, "rows": [{"race_id": r, "before": b, "after": af} for r, b, af in ch[:2]]})
            if a.apply:
                bdir.mkdir(parents=True, exist_ok=True)
                (bdir / f"{hid}.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
                latest = s.load(CAT, hid)  # 並走する取得で更新されていないか、保存直前に読み直して再適用（冪等）
                new2 = hf.apply_overlay(latest, lookup) if isinstance(latest, dict) else new
                s.save(CAT, hid, new2)
                stats["saved"] += 1
        except Exception as exc:  # noqa: BLE001
            stats["errors"] += 1
            if stats["errors"] <= 5:
                print("ERR", hid, repr(exc)[:200])

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        for i, _ in enumerate(ex.map(work, keys), 1):
            if i % 2000 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(keys)} 変更馬={stats['changed_horses']} 保存={stats['saved']} err={stats['errors']}", flush=True)
    stats["seconds"] = round(time.time() - t0)
    rep = {"mode": "apply" if a.apply else "dry-run", "backup_dir": str(bdir) if a.apply else None, **stats, "samples": samples}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "samples"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
