"""祖先種牡馬の生年月日を複数の出典から統合し、生年が確定しなかった馬の一覧を出す。

出典と優先順位（年は常に netkeiba を正とし、月日だけを補う）:
  1. netkeiba のプロフィール（年月日 or 年のみ）
  2. Wikipedia 自動照会（wiki_month.py の結果。年が一致したものだけ採用、不一致は wiki_conflict に記録）
  3. 手入力・Web 検索で見つけた値 data/meta/dormello/birth_manual.csv（最優先。年も上書きできる）

    python -m src.research.pedigree.ancestor_birthdays merge \
        --need need_birth.csv --nk nk_birth_results.jsonl --wiki wiki_month_results.jsonl \
        --manual data/meta/dormello/birth_manual.csv \
        --out-table ancestor_birthdays.csv --out-unknown data/meta/dormello/birth_unknown.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

MANUAL_COLUMNS = ["horse_id", "name", "birth_year", "birth_month", "birth_day", "source_url", "note"]
UNKNOWN_COLUMNS = ["horse_id", "name", "min_gen", "n_pairs", "reason", "wiki_hint"]
TABLE_COLUMNS = ["horse_id", "name", "birthday", "precision", "source", "source_url", "wiki_conflict"]
_FULL = re.compile(r"(\d{4})年(?:(\d{1,2})月(?:(\d{1,2})日)?)?")


def parse_raw(s: str) -> tuple[int, int | None, int | None] | None:
    m = _FULL.fullmatch((s or "").strip())
    if not m:
        return None
    return int(m[1]), int(m[2]) if m[2] else None, int(m[3]) if m[3] else None


def fmt(y: int, mo: int | None, d: int | None) -> tuple[str, str]:
    if mo and d:
        return f"{y:04d}-{mo:02d}-{d:02d}", "day"
    if mo:
        return f"{y:04d}-{mo:02d}", "month"
    return f"{y:04d}", "year"


def _read_jsonl(path: Path | None) -> list[dict]:
    if not path or not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_manual_births(path: Path | None) -> dict[str, dict]:
    if not path or not path.exists():
        return {}
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            y = (r.get("birth_year") or "").strip()
            if not y:
                continue
            mo, d = (r.get("birth_month") or "").strip(), (r.get("birth_day") or "").strip()
            if d and not mo:
                raise ValueError(f"birth_day だけで birth_month が空: {r}")
            out[r["horse_id"]] = {
                "y": int(y), "mo": int(mo) if mo else None, "d": int(d) if d else None,
                "url": r.get("source_url") or "", "name": r.get("name") or "",
            }
    return out


def merge(need_rows: list[dict], nk: list[dict], wiki: list[dict], manual: dict[str, dict]):
    """(統合テーブル, 生年不明リスト) を返す。need_rows は horse_id,name,min_gen,n_pairs の行。"""
    nk_by = {r["horse_id"]: r for r in nk}
    wiki_by = {r["horse_id"]: r for r in wiki}
    table, unknown = [], []
    for n in need_rows:
        hid = n["horse_id"]
        rec = nk_by.get(hid)
        y = mo = d = None
        source, url, conflict = "", "", ""
        if rec and rec.get("status") == "ok":
            p = parse_raw(rec.get("birthday_raw", ""))
            if p:
                y, mo, d = p
                source, url = "netkeiba", f"https://db.netkeiba.com/horse/{hid}/"
        w = wiki_by.get(hid)
        if w and w.get("status") in ("day", "month", "year_only"):
            if y is not None and w["wiki_year"] == y:
                if (mo is None and w.get("month")) :
                    mo, d = w["month"], w.get("day")
                    source, url = "netkeiba+wikipedia", w["url"]
                elif mo and d is None and w.get("month") == mo and w.get("day"):
                    d = w["day"]
                    source, url = "netkeiba+wikipedia", w["url"]
        elif w and w.get("status") == "year_mismatch":
            conflict = f"wikipedia={w.get('wiki_year')}-{w.get('month')}-{w.get('day')} {w.get('url', '')}"
        m = manual.get(hid)
        if m:
            y, mo, d, source, url = m["y"], m["mo"], m["d"], "manual", m["url"]
        if y is None:
            if rec is None:
                reason = "not_fetched"
            elif rec.get("status") == "error":
                reason = "fetch_error"
            elif rec.get("status") == "no_birthday":
                reason = "netkeiba_no_birthday"
            else:
                reason = "unparseable"
            unknown.append({"horse_id": hid, "name": n["name"], "min_gen": n["min_gen"], "n_pairs": n["n_pairs"],
                            "reason": reason, "wiki_hint": conflict})
            continue
        bd, prec = fmt(y, mo, d)
        table.append({"horse_id": hid, "name": n["name"], "birthday": bd, "precision": prec,
                      "source": source, "source_url": url, "wiki_conflict": conflict})
    return table, unknown


def _write(path: Path, cols: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--need", required=True, type=Path)
    m.add_argument("--nk", required=True, type=Path)
    m.add_argument("--wiki", type=Path)
    m.add_argument("--manual", type=Path)
    m.add_argument("--out-table", required=True, type=Path)
    m.add_argument("--out-unknown", required=True, type=Path)
    a = ap.parse_args(argv)
    with open(a.need, encoding="utf-8", newline="") as f:
        need = list(csv.DictReader(f))
    table, unknown = merge(need, _read_jsonl(a.nk), _read_jsonl(a.wiki), load_manual_births(a.manual))
    _write(a.out_table, TABLE_COLUMNS, table)
    _write(a.out_unknown, UNKNOWN_COLUMNS, unknown)
    prec = {}
    for r in table:
        prec[r["precision"]] = prec.get(r["precision"], 0) + 1
    reasons = {}
    for r in unknown:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    print(json.dumps({"resolved": len(table), "by_precision": prec, "unknown": len(unknown), "unknown_reasons": reasons},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
