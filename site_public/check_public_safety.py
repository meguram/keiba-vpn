#!/usr/bin/env python3
"""公開側サイトに、公開してはいけないものが混ざっていないかを検査する（標準ライブラリのみ）。

禁止するもの: 中核データ（予測・期待値・推奨・印・独自指数）の痕跡、外部サイト由来のデータ元の名称、レース ID・馬 ID、
秘密情報（トークン・API キー）、内部ホスト名・IP、実在のメールアドレス・電話番号・郵便番号、許可リスト外の URL。
注意: 馬名・騎手名などの固有名詞は機械的には判定できない（公開前に目視で確認する）。

  python site_public/check_public_safety.py              # content/ config.json static/ を検査
  python site_public/check_public_safety.py --dist DIR   # ビルド後の出力も検査
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

RULES: list[tuple[str, re.Pattern]] = [
    ("データ元の名称", re.compile(r"netkeiba|smartrc|jra-?van|nar\.keiba", re.I)),
    ("レースID", re.compile(r"(?<!\d)\d{12}(?!\d)")),
    ("馬ID", re.compile(r"(?<!\d)(?:19|20)\d{8}(?!\d)")),
    ("秘密情報(JWT/トークン)", re.compile(r"eyJ[A-Za-z0-9_\-]{20,}")),
    ("秘密情報(Slack)", re.compile(r"hooks\.slack\.com", re.I)),
    ("秘密情報(AWS)", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("秘密情報(sk-)", re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    ("秘密情報(秘密鍵)", re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY")),
    ("秘密情報(代入)", re.compile(r"(?i)\b(?:api[_-]?key|secret|token|passwd|password)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{8,}")),
    ("内部ホスト・IP", re.compile(r"app\.megu-keiba\.com|monitor\.megu-keiba\.com|localhost|127\.0\.0\.1|192\.168\.|\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|tcpexposer|cloudflareaccess|trycloudflare|160\.251\.", re.I)),
    ("中核データの痕跡(語)", re.compile(r"めぐ指数|win_prob|top[23]_prob|ev_win|ev_place|megu_mark|race_id|horse_id|predictions?\.json")),
    ("中核データの痕跡(印)", re.compile(r"[◎▲★△]")),
    ("電話番号", re.compile(r"(?<!\d)0\d{1,4}-\d{1,4}-\d{3,4}(?!\d)")),
    ("郵便番号", re.compile(r"〒\s?\d{3}-?\d{4}")),
]
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@([A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+)")
URL_RE = re.compile(r"https?://([A-Za-z0-9.\-]+)", re.I)
ALLOWED_EMAIL_DOMAINS = {"example.com", "example.org", "example.net"}
TEXT_SUFFIXES = {".html", ".json", ".css", ".svg", ".txt", ".xml", ".md", ""}


def load_allowed_domains(root: Path = ROOT) -> set[str]:
    try:
        cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
        return set(cfg.get("allowed_domains", []))
    except OSError:
        return set()


def scan_text(text: str, allowed_domains: set[str], where: str = "") -> list[str]:
    hits: list[str] = []
    for name, rx in RULES:
        for m in rx.finditer(text):
            hits.append(f"{where}: {name}: {m.group(0)[:40]!r}")
    for m in EMAIL_RE.finditer(text):
        if m.group(1).lower() not in ALLOWED_EMAIL_DOMAINS:
            hits.append(f"{where}: 実在の可能性があるメールアドレス: {m.group(0)[:40]!r}")
    for m in URL_RE.finditer(text):
        if m.group(1).lower() not in allowed_domains:
            hits.append(f"{where}: 許可リスト外の URL: {m.group(0)[:60]!r}")
    return hits


def iter_files(paths: list[Path]):
    for base in paths:
        if base.is_file():
            yield base
        elif base.is_dir():
            for p in sorted(base.rglob("*")):
                if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES and "__pycache__" not in p.parts:
                    yield p


def scan_paths(paths: list[Path], allowed_domains: set[str], root: Path = ROOT) -> list[str]:
    hits: list[str] = []
    for f in iter_files(paths):
        try:
            text = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            where = str(f.relative_to(root))
        except ValueError:
            where = str(f)
        hits += scan_text(text, allowed_domains, where)
    return hits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dist", help="ビルド後の出力ディレクトリも検査する")
    args = ap.parse_args(argv)
    allowed = load_allowed_domains()
    targets = [ROOT / "config.json", ROOT / "content", ROOT / "static"]
    if args.dist:
        targets.append(Path(args.dist))
    hits = scan_paths(targets, allowed)
    for h in hits:
        print("NG:", h)
    print("公開前の安全検査:", "OK（禁止パターンなし）" if not hits else f"{len(hits)} 件の指摘")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
