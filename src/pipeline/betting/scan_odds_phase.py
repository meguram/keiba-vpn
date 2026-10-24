"""GCS のオッズ文書（race_odds・race_pair_odds）に、取得の時点（phase）とスナップショット（snapshots）がどれだけあるかを調べる（読み取りのみ）。

馬券シミュで「購入判断の時点のオッズ」が使えるかを判断するための調査。2020〜2025 年は年ごとの無作為標本、2026 年は全件を見る。
  python -m src.pipeline.betting.scan_odds_phase --sample 150
結果は ``data/local/modeling/betting/phase_scan.json``。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def classify(doc: dict | None) -> dict:
    if not doc:
        return {"missing": True}
    meta = doc.get("_meta") or {}
    snaps = doc.get("snapshots") or []
    return {"phase": doc.get("phase") or "なし", "snapshots": sorted({str(s.get("phase")) for s in snaps if isinstance(s, dict)}),
            "scraped_month": str(meta.get("scraped_at_jst") or "")[:7], "snapshot_at": bool(doc.get("snapshot_at"))}


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("missing")]
    return {"docs": len(rows), "missing": len(rows) - len(ok), "phase": dict(Counter(r["phase"] for r in ok)),
            "with_snapshots": sum(1 for r in ok if r["snapshots"]), "snapshot_phases": dict(Counter(p for r in ok for p in r["snapshots"])),
            "scraped_month": dict(sorted(Counter(r["scraped_month"] for r in ok).items()))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=150)
    ap.add_argument("--out", type=Path, default=Path("data/local/modeling/betting/phase_scan.json"))
    a = ap.parse_args()
    from src.utils.project_env import load_project_dotenv
    from src.scraper.storage import HybridStorage

    load_project_dotenv()
    st = HybridStorage()
    rng = random.Random(0)
    out: dict = {}
    for cat in ("race_odds", "race_pair_odds"):
        out[cat] = {}
        for y in range(2020, 2027):
            keys = sorted(st.batch_list_blobs(cat, str(y)))
            pick = keys if y == 2026 else rng.sample(keys, min(a.sample, len(keys)))
            with ThreadPoolExecutor(max_workers=2) as ex:
                rows = list(ex.map(lambda k: classify(st.load(cat, k, bypass_cache=True)), pick))
            out[cat][str(y)] = summarize(rows)
            print(cat, y, out[cat][str(y)]["docs"], out[cat][str(y)]["phase"], "snapshots付き", out[cat][str(y)]["with_snapshots"], flush=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
