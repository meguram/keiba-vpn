"""FastAPI の全ルートについて、未ログインで取れるもの（現行の認証設定）と、auth.mode=enforce にしたときの変化を一覧にする。

Usage: python -m src.scripts.maintenance.anti_scraping_audit [--out PATH]
標準出力に要約、--out に Markdown の全一覧を書く。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ID_PARAMS = re.compile(r"\{(race_id|horse_id|jockey_id|trainer_id|sire_id|person_id|date|year)\b[^}]*\}")


def collect() -> list[dict]:
    os.environ.setdefault("KEIBA_ENV", "dev")
    sys.path.insert(0, str(REPO_ROOT))
    from src.api import app as A
    from src.api.auth import requires_auth
    from src.utils.anti_scraping import get_engine

    eng = get_engine()
    rows = []
    for r in A.app.routes:
        path = getattr(r, "path", None)
        if not path:
            continue
        for meth in sorted(getattr(r, "methods", None) or ["GET"]):
            if meth in ("HEAD", "OPTIONS"):
                continue
            sample = re.sub(r"\{[^}]+\}", "X", path)
            kind = eng.path_kind(sample)
            legacy_auth = requires_auth(sample, meth)
            enforce_auth = legacy_auth or (kind == "data" and eng._needs_auth(sample))
            rows.append({"path": path, "method": meth, "kind": kind, "open_now": not legacy_auth, "open_after_enforce": not enforce_auth,
                         "enumerable": bool(ID_PARAMS.search(path))})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rows = collect()
    get = [r for r in rows if r["method"] == "GET"]
    api_get = [r for r in get if r["kind"] in ("data", "honeypot", "internal")]
    open_now = [r for r in api_get if r["open_now"]]
    open_after = [r for r in api_get if r["open_after_enforce"]]
    enum_open = [r for r in open_now if r["enumerable"]]
    print(f"ルート総数 {len(rows)}（GET {len(get)}、そのうち API {len(api_get)}）")
    print(f"現行設定で未ログインでも取れる API(GET): {len(open_now)} 本（ID・日付で列挙できる形: {len(enum_open)} 本）")
    print(f"auth=enforce にした後に未ログインで取れる API(GET): {len(open_after)} 本（健康確認・認証状態などの例外のみ）")
    if a.out:
        lines = ["# スクレイピング対策: ルート一覧", "", "| メソッド | パス | 種別 | 現行で未ログインOK | enforce 後に未ログインOK | ID/日付で列挙可 |", "|---|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda x: (x["path"], x["method"])):
            lines.append(f"| {r['method']} | `{r['path']}` | {r['kind']} | {'○' if r['open_now'] else ''} | {'○' if r['open_after_enforce'] else ''} | {'○' if r['enumerable'] else ''} |")
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"全一覧: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
