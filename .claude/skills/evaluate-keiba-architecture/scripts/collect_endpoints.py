#!/usr/bin/env python3
"""keiba-vpn の3つのルーティング層（FastAPI legacy / Flask v1 / monitor）から
HTTPエンドポイントを静的解析（ast）で列挙し、JSONで出力する。

対象ファイルは巨大（src/api/app.py は1万行超）なため、import して実行するのではなく
AST解析のみで安全に一覧化する。

Usage:
    python3 .claude/skills/evaluate-keiba-architecture/scripts/collect_endpoints.py
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

LAYERS = [
    {
        "layer": "fastapi_legacy",
        "file": "src/api/app.py",
        "port": 8000,
        "framework": "fastapi",
        "note": "DEC-013により段階廃止予定。新規ルートを追加しないこと。",
    },
    {
        "layer": "flask_v1",
        "file": "src/api/flask_app.py",
        "port": 5000,
        "framework": "flask",
        "note": "DEC-013により仕様上の正。新規APIはここに追加する。",
        "blueprint_glob": "src/api/v1/routes/*.py",
        "blueprint_prefix": "/api/v1",
    },
    {
        "layer": "monitor",
        "file": "src/monitor/app.py",
        "port": 9090,
        "framework": "flask",
        "note": "開発者専用監視ポータル。エンドユーザー非公開。",
    },
]

HTTP_METHOD_ATTRS = {"get", "post", "put", "delete", "patch", "options", "head"}


def _literal(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def extract_routes(path: Path) -> list[dict]:
    routes: list[dict] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call) or not isinstance(dec.func, ast.Attribute):
                continue
            attr = dec.func.attr
            args = [_literal(a) for a in dec.args]
            path_arg = args[0] if args and isinstance(args[0], str) else None
            if not path_arg:
                continue
            if attr in HTTP_METHOD_ATTRS:
                methods = [attr.upper()]
            elif attr in ("route", "api_route"):
                methods = ["GET"]
                for kw in dec.keywords:
                    if kw.arg == "methods":
                        m = _literal(kw.value)
                        if m:
                            methods = [str(x).upper() for x in m]
            else:
                continue
            routes.append(
                {
                    "method": methods,
                    "path": path_arg,
                    "handler": node.name,
                    "line": node.lineno,
                }
            )
    return routes


def main() -> None:
    result = []
    for layer in LAYERS:
        fp = ROOT / layer["file"]
        entry = dict(layer)
        if not fp.exists():
            entry["error"] = "file_not_found"
            entry["routes"] = []
        else:
            routes = extract_routes(fp)
            # ブループリント配下のルート（flask_app.py の register_blueprint(url_prefix=...) 経由）
            if layer.get("blueprint_glob"):
                for bp_file in sorted(ROOT.glob(layer["blueprint_glob"])):
                    if bp_file.name.startswith("__"):
                        continue
                    for r in extract_routes(bp_file):
                        r["path"] = layer["blueprint_prefix"] + r["path"]
                        r["source_file"] = str(bp_file.relative_to(ROOT))
                        routes.append(r)
            entry["routes"] = routes
            entry["route_count"] = len(routes)
        result.append(entry)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
