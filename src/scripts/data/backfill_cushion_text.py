"""JRA 馬場情報 PDF（2018〜）を再読して、保管済みの含水率・クッション値の欠けを埋める。

python -m src.scripts.data.backfill_cushion_text --years 2018 2019 2020 2021 2022 2023 2024 2025 2026          # 既定は dry-run（書かない）
python -m src.scripts.data.backfill_cushion_text --years 2018 ... --apply                                   # 書き込む（先に data/local/backup/ へ退避）

方針: 既存の値は上書きしない（空の項目だけ埋め、無い日は足す）。PDF と既存値の食い違いは報告に残す。画像だけの PDF は data/meta/cushion_manual/ の書き起こしを使う。
書き込み先: data/page_reference/cushion/（正）と、同じ内容を持つ data/jra_baba/・data/calculated_data/cushion/・data/local/jra_baba/ のうち存在するもの。GCS へは書かない。
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

from src.scraper import jra_cushion_text as jt
from src.scraper.jra_cushion import JRACushionScraper

ROOT = Path(__file__).resolve().parents[3]
CANON = ROOT / "data" / "page_reference" / "cushion"
COPIES = [ROOT / "data" / "jra_baba", ROOT / "data" / "calculated_data" / "cushion", ROOT / "data" / "local" / "jra_baba"]


def _load(d: Path) -> list[dict]:
    p = d / "cushion_values.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def _write(d: Path, rows: list[dict]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "cushion_values.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    by_year: dict[int, list[dict]] = {}
    for r in rows:
        by_year.setdefault(int(r.get("year") or str(r.get("date", "0000"))[:4]), []).append(r)
    for y, rs in by_year.items():
        (d / f"cushion_{y}.json").write_text(json.dumps(rs, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=list(range(2018, 2025)))     # 2025 年以降は別形式（日別の表）なので既存パーサに任せる
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--report", default=str(ROOT / "data" / "local" / "meta" / "cushion_backfill_report.json"))
    a = ap.parse_args()

    existing = _load(CANON)
    sc = JRACushionScraper(output_dir="/tmp/cushion_backfill_unused")
    incoming: list[dict] = []
    warnings: list[str] = []
    no_text: list[str] = []
    for y in a.years:
        for link in sc.discover_pdf_links(y):
            time.sleep(0.6)
            p = sc.download_pdf(link)
            if not p:
                warnings.append(f"ダウンロード失敗: {link['url']}")
                continue
            try:
                recs = jt.parse_pdf_text(p, link, warnings=warnings)
            finally:
                p.unlink(missing_ok=True)
            if not recs:
                no_text.append(f"{y} {link['venue_name']} 第{link['kai']}回")
            incoming += recs
    incoming += jt.load_manual(ROOT / "data" / "meta" / "cushion_manual")
    rows, st = jt.merge_missing(existing, incoming)
    rep = {"evaluated_at": datetime.now().isoformat(timespec="seconds"), "years": a.years, "existing_rows": len(existing), "incoming_rows": len(incoming), "result_rows": len(rows),
           "added_rows": st["added"], "filled_fields": st["filled_fields"], "agree_fields": st["agree_fields"], "mismatch_count": len(st["mismatches"]), "mismatches": st["mismatches"][:200],
           "warnings": warnings, "pdf_without_text_records": no_text, "applied": bool(a.apply)}
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("mismatches", "warnings", "pdf_without_text_records")}, ensure_ascii=False))
    print("警告", len(warnings), "| 食い違い", len(st["mismatches"]), "| 読めなかった PDF", len(no_text))
    if a.apply:
        bk = ROOT / "data" / "local" / "backup" / f"jra_cushion_{datetime.now():%Y%m%d_%H%M%S}"
        for d in [CANON] + COPIES:
            if d.exists():
                shutil.copytree(d, bk / d.name, dirs_exist_ok=True)
        for d in [CANON] + [c for c in COPIES if c.exists()]:
            _write(d, rows)
        print("書き込み済み:", [str(d) for d in [CANON] + [c for c in COPIES if c.exists()]], "| 退避:", bk)


if __name__ == "__main__":
    main()
