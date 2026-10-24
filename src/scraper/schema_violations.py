"""スキーマ違反の記録: 「どのカテゴリの・どの key の・どの項目が・どの値で」引っかかったかを残す。

保存時（``HybridStorage.save``）に不合格だったとき、次の 3 か所に記録する（git 管理外の ``data/local/``）。
  1. ``data/local/meta/schema_violations/<category>.jsonl`` … 1 行 1 回の不合格。違反の中身（項目・規則・期待・実際の値・要素番号・馬番など）
  2. ``data/local/quarantine/<category>/<key>.json``        … 拒否して保存しなかったデータ本体（原因調査・スキーマ再構成に使う）。
                                                              同じ key が後で合格して保存されたら削除する
  3. 保存された JSON の ``_meta.schema_validation.violations`` … advisory／lenient で保存した場合は、データ自体にも残る

判断（decision）: rejected（保存せず拒否）/ saved_advisory（advisory スキーマ。保存した）/ saved_lenient（KEIBA_SCHEMA_STRICT=0。保存した）

集計・確認:
  python -m src.scraper.schema_violations summary [--category C] [--days 30]   # どの項目がどの値で何回引っかかったか
  python -m src.scraper.schema_violations show CATEGORY KEY                    # 1 件の違反と、拒否したデータ
  python -m src.scraper.schema_violations tail [-n 20]

無効化: ``KEIBA_SCHEMA_VIOLATION_LOG=0``（記録）/ ``KEIBA_SCHEMA_QUARANTINE=0``（隔離）。記録の失敗で保存処理を止めることはない。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from src.scraper import schemas

logger = logging.getLogger("scraper.schema_violations")
JST = timezone(timedelta(hours=9))
LOG_MAX_BYTES = 5_000_000          # 超えたら .1 に退避（1 世代）
LOGGED_VIOLATIONS = 12             # 1 行に残す違反の最大数


def _on(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() not in ("0", "false", "off", "no")


def log_dir(base_dir: str | Path = ".") -> Path:
    return Path(base_dir) / "data" / "local" / "meta" / "schema_violations"


def quarantine_dir(base_dir: str | Path = ".") -> Path:
    return Path(base_dir) / "data" / "local" / "quarantine"


def _env_name() -> str:
    try:
        from src.config.deployment import keiba_env

        return keiba_env()
    except Exception:
        return ""


def record(base_dir: str | Path, category: str, key: str, report: dict[str, Any], decision: str, *,
           payload: dict[str, Any] | None = None) -> None:
    """不合格を記録する。例外は握りつぶす（記録の失敗で保存処理を止めない）。"""
    try:
        now = datetime.now(JST).isoformat(timespec="seconds")
        if _on("KEIBA_SCHEMA_VIOLATION_LOG"):
            d = log_dir(base_dir)
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{category}.jsonl"
            if path.exists() and path.stat().st_size > LOG_MAX_BYTES:
                path.replace(path.with_suffix(".jsonl.1"))
            vs = report.get("violations") or []
            line = {"at": now, "category": category, "key": key, "decision": decision, "env": _env_name(),
                    "schema_version": report.get("schema_version"), "schema": schemas.category_fingerprint(category),
                    "violations_total": len(vs) + (1 if report.get("violations_truncated") else 0),
                    "violations": vs[:LOGGED_VIOLATIONS]}
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        if decision == "rejected" and payload is not None and _on("KEIBA_SCHEMA_QUARANTINE"):
            q = quarantine_dir(base_dir) / category
            q.mkdir(parents=True, exist_ok=True)
            (q / f"{key}.json").write_text(json.dumps(
                {"category": category, "key": key, "rejected_at": now, "schema": schemas.category_fingerprint(category),
                 "violations": report.get("violations") or [], "data": payload}, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        logger.warning("スキーマ違反の記録に失敗（保存処理は継続）: %s/%s: %s", category, key, e)


def resolve(base_dir: str | Path, category: str, key: str) -> None:
    """同じ key が後で合格して保存されたら、隔離していたデータを消す。"""
    try:
        p = quarantine_dir(base_dir) / category / f"{key}.json"
        if p.exists():
            p.unlink()
    except OSError:
        pass


def read_log(base_dir: str | Path = ".", category: str | None = None, *, days: int | None = None,
             since: str | None = None) -> Iterator[dict[str, Any]]:
    d = log_dir(base_dir)
    files = [d / f"{category}.jsonl"] if category else sorted(d.glob("*.jsonl")) if d.is_dir() else []
    cutoff = since or ((datetime.now(JST) - timedelta(days=days)).isoformat() if days else None)
    for f in files:
        for name in (f.with_suffix(".jsonl.1"), f):
            if not name.exists():
                continue
            for line in name.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if cutoff is None or rec.get("at", "") >= cutoff:
                    yield rec


def quarantined(base_dir: str | Path = ".") -> dict[str, list[str]]:
    d = quarantine_dir(base_dir)
    return {c.name: sorted(p.stem for p in c.glob("*.json")) for c in sorted(d.iterdir()) if c.is_dir()} if d.is_dir() else {}


def summarize(base_dir: str | Path = ".", category: str | None = None, *, days: int | None = 30, top: int = 20,
              examples: int = 5, since: str | None = None) -> dict[str, Any]:
    """(カテゴリ, 項目, 規則) ごとに、何回・何 key で・どの値で引っかかったか。"""
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    decisions: Counter[str] = Counter()
    records = 0
    for rec in read_log(base_dir, category, days=days, since=since):
        records += 1
        decisions[rec["decision"]] += 1
        for v in rec.get("violations", []):
            g = groups.setdefault((rec["category"], v["field"], v["rule"]), {
                "category": rec["category"], "field": v["field"], "rule": v["rule"], "expected": v.get("expected"),
                "count": 0, "keys": set(), "values": Counter(), "first": rec["at"], "last": rec["at"], "decisions": Counter()})
            g["count"] += 1
            g["keys"].add(rec["key"])
            g["values"][f"{v.get('actual_type')} {v.get('actual')}"] += 1
            g["last"] = max(g["last"], rec["at"])
            g["first"] = min(g["first"], rec["at"])
            g["decisions"][rec["decision"]] += 1
    rows = sorted(groups.values(), key=lambda g: -g["count"])[:top]
    return {"records": records, "decisions": dict(decisions), "quarantined": {c: len(k) for c, k in quarantined(base_dir).items()},
            "groups": [{"category": g["category"], "field": g["field"], "rule": g["rule"], "expected": g["expected"],
                        "count": g["count"], "keys": len(g["keys"]), "first": g["first"], "last": g["last"],
                        "decisions": dict(g["decisions"]), "values": g["values"].most_common(examples)} for g in rows]}


# ── CLI ─────────────────────────────────────────────────────────────────────

def _print_summary(s: dict[str, Any]) -> None:
    print(f"不合格の記録 {s['records']} 回 / 判断 {s['decisions']} / 隔離中 {s['quarantined'] or 'なし'}")
    for g in s["groups"]:
        print(f"\n■ {g['category']}  {g['field']}  [{g['rule']}] 期待: {g['expected']}")
        print(f"   {g['count']} 回 / {g['keys']} key / {g['first']} 〜 {g['last']} / {g['decisions']}")
        for val, n in g["values"]:
            print(f"     {n:>5} × {val}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="スキーマ違反の記録を確認する")
    ap.add_argument("--base-dir", default=".")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("summary", help="項目×規則ごとに、どの値で何回引っかかったか")
    a.add_argument("--category")
    a.add_argument("--days", type=int, default=30)
    a.add_argument("--top", type=int, default=20)
    b = sub.add_parser("show", help="1 件の違反と、隔離したデータ")
    b.add_argument("category")
    b.add_argument("key")
    c = sub.add_parser("tail", help="最近の記録")
    c.add_argument("-n", type=int, default=20)
    args = ap.parse_args(argv)
    if args.cmd == "summary":
        _print_summary(summarize(args.base_dir, args.category, days=args.days, top=args.top))
    elif args.cmd == "show":
        recs = [r for r in read_log(args.base_dir, args.category) if r["key"] == args.key]
        for r in recs[-3:]:
            print(f"{r['at']}  {r['decision']}  schema={r['schema']}  env={r['env']}")
            for v in r["violations"]:
                print("   -", schemas.describe_violation(v))
        q = quarantine_dir(args.base_dir) / args.category / f"{args.key}.json"
        print(f"\n隔離データ: {q if q.exists() else 'なし（保存済み、または解決済み）'}")
        if not recs and not q.exists():
            return 1
    else:
        rows = list(read_log(args.base_dir))[-args.n:]
        for r in rows:
            first = schemas.describe_violation(r["violations"][0]) if r["violations"] else ""
            print(f"{r['at']} {r['decision']:<15} {r['category']}/{r['key']}  {first}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
