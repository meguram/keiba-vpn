#!/usr/bin/env python3
"""ドキュメント・設定ファイルに書かれたファイル/ディレクトリパスが実在するかを検証する。

チェック対象:
  - AGENTS.md            （レイアウト表内のバックティックパス）
  - docs/operations/service-endpoints.md （バックティックパス）
  - config/settings.yaml （`paths:` 等、リポジトリ相対パスに見える文字列値）

Usage:
    python3 .claude/skills/evaluate-keiba-architecture/scripts/check_paths.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]

TOP_DIRS = r"src|data|docs|scripts|config|tests|mlflow|notebooks|templates|static|models|pipelines"
BACKTICK_PATH_RE = re.compile(rf"`((?:{TOP_DIRS})/[^`]*)`")
YAML_PATH_RE = re.compile(rf"^(?:{TOP_DIRS})/")


def _clean(raw: str) -> str:
    # 動的セグメント（<race_id> や {race_id}、glob の * など）より前で切る
    p = raw.split("<")[0].split("{")[0].split("*")[0]
    return p.rstrip("/")


def find_backtick_paths(text: str) -> set[str]:
    found = set()
    for m in BACKTICK_PATH_RE.finditer(text):
        p = _clean(m.group(1))
        if p:
            found.add(p)
    return found


def find_yaml_paths(data) -> set[str]:
    found: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, str) and YAML_PATH_RE.match(node):
            found.add(node.rstrip("/"))

    walk(data)
    return found


def check(paths: set[str]) -> list[dict]:
    out = []
    for p in sorted(paths):
        out.append({"path": p, "exists": (ROOT / p).exists()})
    return out


def main() -> None:
    result: dict[str, list[dict]] = {}

    agents_md = ROOT / "AGENTS.md"
    if agents_md.exists():
        result["AGENTS.md"] = check(find_backtick_paths(agents_md.read_text(encoding="utf-8")))

    ep_doc = ROOT / "docs/operations/service-endpoints.md"
    if ep_doc.exists():
        result["docs/operations/service-endpoints.md"] = check(
            find_backtick_paths(ep_doc.read_text(encoding="utf-8"))
        )

    settings_yaml = ROOT / "config/settings.yaml"
    if settings_yaml.exists():
        data = yaml.safe_load(settings_yaml.read_text(encoding="utf-8"))
        result["config/settings.yaml"] = check(find_yaml_paths(data))

    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
