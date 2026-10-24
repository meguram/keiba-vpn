"""出馬表ページ（netkeiba shutuba.html）の“いま取れるもの”を調べる読み取り専用の確認ツール（保存しない）。

JRA の公示の段階（特別登録 → 出走馬決定 → 前日の確定）で、出馬表に何が入っているか（頭数・馬名・騎手・斤量・枠番/馬番・馬体重）を
数だけ出す。生の HTML・テキストは出力しない（件数と有無のみ）。取得は DEC-023 / NetkeibaClient のスロットに従い、間隔を空けて数レースだけ。

使い方（ホストでは sudo -n が要る: スロットのディレクトリが root 所有）:
  sudo -n env KEIBA_ENV=stg PYTHONPATH=. python3 src/scripts/data/entries_probe.py 202608040801 202608040911
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time


def summarize(card: dict | None) -> dict:
    if not card:
        return {"ok": False}
    ents = card.get("entries") or []

    def has(key: str) -> int:
        return sum(1 for e in ents if e.get(key) not in (None, "", 0, "--", "-"))

    key_sets = sorted({k for e in ents for k in e})
    norm = json.dumps(
        [[e.get("horse_number"), e.get("horse_name"), e.get("jockey_id"), e.get("jockey_weight")] for e in ents],
        ensure_ascii=False,
    )
    return {
        "ok": True,
        "race_name": bool(card.get("race_name")),
        "start_time": card.get("start_time"),
        "surface": card.get("surface"),
        "distance": card.get("distance"),
        "grade": card.get("grade"),
        "field_size": card.get("field_size"),
        "n_entries": len(ents),
        "with_gate": has("bracket_number"),
        "with_number": has("horse_number"),
        "with_horse_name": has("horse_name"),
        "with_jockey": has("jockey_id"),
        "with_weight_carried": has("jockey_weight"),
        "with_body_weight": has("weight"),
        "horse_number_values": sorted({repr(e.get("horse_number")) for e in ents})[:5],
        "bracket_number_values": sorted({repr(e.get("bracket_number")) for e in ents})[:5],
        "entry_keys": key_sets,
        "content_hash": hashlib.sha256(norm.encode()).hexdigest()[:12],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("race_ids", nargs="+")
    ap.add_argument("--sleep", type=float, default=6.0, help="レース間の待ち（秒）。4.5〜7 秒の範囲")
    a = ap.parse_args(argv)
    if len(a.race_ids) > 8:
        print("確認は 8 レースまで", file=sys.stderr)
        return 2
    from src.scraper.run import ScraperRunner

    runner = ScraperRunner()
    out = {}
    for i, rid in enumerate(a.race_ids):
        if i:
            time.sleep(a.sleep)
        url = runner.RACE_CARD_URL.format(race_id=rid)
        try:
            html = runner.client.fetch(url, use_cache=False)
            card = runner.card_parser.parse(html, race_id=rid)
            s = summarize(card)
            s["html_len"] = len(html)
            try:                                              # 保存前のスキーマ検証に通るか（馬番が未確定でも保存できるか）
                from src.scraper import schemas

                s["schema_passed"] = bool(schemas.validate("race_shutuba", card).get("passed"))
            except Exception as exc:  # noqa: BLE001
                s["schema_passed"] = f"error: {str(exc)[:60]}"
            # ページ上の段階を示す語（有無だけ）
            for label in ("特別登録", "出走馬名表", "出馬投票", "枠順", "馬体重", "発走時刻"):
                s["has_word_" + label] = label in html
        except Exception as exc:  # noqa: BLE001
            s = {"ok": False, "error": str(exc)[:120]}
        out[rid] = s
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
