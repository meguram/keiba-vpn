"""保存済みの race_result / race_result_meta で、レース名に明記されたクラスと grade が食い違っているものを訂正する。

背景: 結果ページのパーサは grade を『ページ全体の文字列』から拾っていたため、周辺の「オープン」「未勝利」「(3勝)」に引かれて誤る（2020 年以降の既存データにも約 0.8%）。
2019 年 6 月の改称前の旧クラス名（500万下・1000万下・1600万下）は読めず『その他』になっていた。規則は RaceResultParser.class_from_name（旧名称・未勝利・新馬・N勝クラス・(N勝)）。
オープン/リステッド/重賞はレース名から決められないので対象外。訂正するのは grade だけ。出馬表系の grade は語彙が違いうるので対象外。
安全策: 既定はドライラン（件数と例だけ）。--apply で、元の JSON を退避（既定 data/local/meta/grade_fix_backup/<日時>/）してから HybridStorage.save で保存する（スキーマ検証・キャッシュ世代の更新つき）。
使い方: python -m src.scripts.maintenance.fix_old_class_grade --years 2019 2020 [--apply] [--backup-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CATEGORIES = ["race_result", "race_result_meta"]      # 同じ抽出（結果ページのパーサ）の語彙


def corrected_grade(d: dict) -> str | None:
    """訂正後の grade（訂正が要らなければ None）。レース名にクラスが明記され、grade がそれと違うときだけ。"""
    from src.scraper.parsers import RaceResultParser

    want = RaceResultParser.class_from_name(str(d.get("race_name") or ""))
    return want if want and d.get("grade") != want else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--years", nargs="+", default=["2019"])
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--backup-dir")
    args = ap.parse_args()

    from src.scraper.storage import HybridStorage

    s = HybridStorage()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    bdir = Path(args.backup_dir) if args.backup_dir else ROOT / "data" / "local" / "meta" / "grade_fix_backup" / stamp
    total = {"checked": 0, "targets": 0, "saved": 0, "failed": 0}
    for y in args.years:
        found: dict[str, int] = {}
        examples: list[str] = []
        for rid in sorted(s.list_keys("race_result", y)):
            for cat in CATEGORIES:
                d = s.load(cat, rid, bypass_cache=True)
                if not d:
                    continue
                total["checked"] += 1
                g = corrected_grade(d)
                if not g:
                    continue
                total["targets"] += 1
                key = f"{cat}:{d.get('grade')}->{g}"
                found[key] = found.get(key, 0) + 1
                if len(examples) < 5:
                    examples.append(f"{cat}/{rid} {d.get('race_name')} {d.get('grade')}->{g}")
                if not args.apply:
                    continue
                out = bdir / cat
                out.mkdir(parents=True, exist_ok=True)
                (out / f"{rid}.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")      # 元のデータを先に退避
                old = d.get("grade")
                d["grade"] = g
                d.setdefault("_meta", {})["grade_fix"] = {"from": old, "to": g, "at": stamp, "reason": "レース名のクラス表記と不一致（パーサがページ全体の文字列から拾っていた）"}
                try:
                    s.save(cat, rid, d)
                    total["saved"] += 1
                except Exception as e:  # noqa: BLE001
                    total["failed"] += 1
                    print(f"失敗 {cat}/{rid}: {e}", file=sys.stderr)
        print(json.dumps({"year": y, "mode": "apply" if args.apply else "dry-run", "by": found, "examples": examples}, ensure_ascii=False), flush=True)
    print(json.dumps({"total": total, "backup_dir": str(bdir)}, ensure_ascii=False))
    return 1 if total["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
