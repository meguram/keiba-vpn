#!/usr/bin/env python3
"""keiba-vpn の現在状態（git / コード / 設計 / データ / TODO 痕跡）を静的に収集して JSON 出力する。

要件定義書 docs/html/index.html を更新するための「事実の下調べ」専用。重いデータ本体
（data/ 配下の Parquet/JSON）は読まず、ディレクトリ存在・直下件数のみを見る。

Usage:
    python3 .claude/skills/update-requirements/scripts/collect_state.py > /tmp/keiba_state.json
    python3 .../collect_state.py --doc PATH   # 前回版を docs/html/index.html 以外から読む（検証用）
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REQ_DOC = ROOT / "docs/html/index.html"
ENDPOINT_SCRIPT = ROOT / ".claude/skills/evaluate-keiba-architecture/scripts/collect_endpoints.py"

SKIP_DIRS = {"node_modules", ".next", "__pycache__", ".git", "catboost_info", ".venv", "venv"}
MARKER_RE = re.compile(r"\b(TODO|FIXME|XXX|HACK|NotImplementedError|mock|dummy|stub|placeholder)\b", re.I)


def _run(args: list[str], timeout: int = 60) -> str:
    try:
        return subprocess.run(
            args, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False
        ).stdout.strip()
    except Exception:
        return ""


def _walk(base: Path, suffixes: tuple[str, ...]):
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(suffixes):
                yield Path(dirpath) / fn


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def _title(path: Path) -> str:
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:4000]
    except Exception:
        return ""
    m = re.search(r"<title>(.*?)</title>", head, re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def previous_doc(doc: Path = REQ_DOC) -> dict:
    """既存の要件書から前回コミット・前回 TODO を取り出す（ID 継続とインクリメンタル更新用）。"""
    out = {"exists": doc.exists(), "commit": None, "updated": None, "todos": [], "is_requirements_doc": False}
    if not doc.exists():
        return out
    text = re.sub(r"<!--.*?-->", "", doc.read_text(encoding="utf-8", errors="ignore"), flags=re.S)
    m = re.search(r'<meta name="requirements-commit" content="([^"]*)"', text)
    out["commit"] = m.group(1) if m else None
    m = re.search(r'<meta name="requirements-updated" content="([^"]*)"', text)
    out["updated"] = m.group(1) if m else None
    out["is_requirements_doc"] = bool(out["commit"] is not None or 'id="scheduling"' in text)
    for tr in re.finditer(r"<tr\b([^>]*\bdata-todo-id=[^>]*)>(.*?)</tr>", text, re.S):
        attrs = dict(re.findall(r'data-([\w-]+)="([^"]*)"', tr.group(1)))
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", tr.group(2), re.S)]
        out["todos"].append({**attrs, "cells": cells})
    return out


def git_state(prev_commit: str | None) -> dict:
    g = {
        "commit": _run(["git", "rev-parse", "--short", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty_files": [l for l in _run(["git", "status", "--porcelain"]).splitlines() if l],
        "recent_commits": _run(["git", "log", "-25", "--pretty=%h %ad %s", "--date=short"]).splitlines(),
        "total_commits": _run(["git", "rev-list", "--count", "HEAD"]),
    }
    g["prev_commit"] = prev_commit
    g["prev_commit_valid"] = bool(prev_commit) and _run(["git", "cat-file", "-t", prev_commit]) == "commit"
    if g["prev_commit_valid"]:
        g["since_prev"] = {
            "commits": _run(["git", "log", f"{prev_commit}..HEAD", "--pretty=%h %ad %s", "--date=short"]).splitlines(),
            "changed_files": _run(["git", "diff", "--name-only", f"{prev_commit}..HEAD"]).splitlines(),
        }
        tops: dict[str, int] = {}
        for f in g["since_prev"]["changed_files"]:
            key = "/".join(f.split("/")[:2]) if "/" in f else f
            tops[key] = tops.get(key, 0) + 1
        g["since_prev"]["changed_by_area"] = dict(sorted(tops.items(), key=lambda kv: -kv[1]))
    return g


def decisions() -> list[dict]:
    rows = []
    for p in sorted((ROOT / "docs/decisions").glob("*.html")):
        text = p.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"ステータス</td><td>(.*?)</td>", text, re.S) or re.search(
            r'<span class="badge[^"]*">(.*?)</span>', text, re.S
        )
        status = re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else ""
        rows.append({"file": _rel(p), "title": _title(p).replace(" — keiba-vpn docs", ""), "status": status})
    return rows


def python_packages() -> list[dict]:
    rows = []
    src = ROOT / "src"
    for sub in sorted(d for d in src.iterdir() if d.is_dir() and d.name not in SKIP_DIRS):
        files = list(_walk(sub, (".py",)))
        lines = 0
        for f in files:
            try:
                lines += sum(1 for _ in f.open(encoding="utf-8", errors="ignore"))
            except Exception:
                pass
        rows.append({"package": f"src/{sub.name}", "py_files": len(files), "lines": lines})
    return rows


def endpoints() -> list | dict:
    if not ENDPOINT_SCRIPT.exists():
        return {"error": "collect_endpoints.py not found"}
    out = _run([sys.executable, str(ENDPOINT_SCRIPT)], timeout=120)
    try:
        return json.loads(out)
    except Exception:
        return {"error": "collect_endpoints.py output not JSON"}


def frontend_pages() -> list[dict]:
    base = ROOT / "frontend/app"
    rows = []
    if base.exists():
        for p in sorted(base.rglob("page.tsx")):
            route = "/" + "/".join(p.relative_to(base).parts[:-1])
            src = p.read_text(encoding="utf-8", errors="ignore")
            rows.append({
                "route": route if route != "/" else "/",
                "file": _rel(p),
                "api_paths": sorted(set(re.findall(r"[\"'`](/api/[\w/\-\[\]${}.]*)", src)))[:8],
                "uses_mock_flag": bool(re.search(r"USE_MOCK|NEXT_PUBLIC_MOCK", src)),
                "imports_hooks_or_lib": sorted(set(re.findall(r"from [\"']@/lib/([\w/\-]+)[\"']", src)))[:6],
            })
    return rows


def mlflow_models() -> dict:
    out: dict = {"catalog": [], "settings_models": []}
    cat = ROOT / "src/pipeline/mlflow/catalog.py"
    if cat.exists():
        try:
            tree = ast.parse(cat.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "ModelSpec":
                    kw = {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg and isinstance(k.value, ast.Constant)}
                    out["catalog"].append(kw)
        except Exception as e:  # 構造が変わっても収集は止めない
            out["catalog_error"] = str(e)
    settings = ROOT / "config/settings.yaml"
    if settings.exists():
        txt = settings.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"^mlflow:.*?^\s{2}models:\n(.*?)(?=^\S|\Z)", txt, re.S | re.M)
        if m:
            out["settings_models"] = re.findall(r"^\s{4}(\w+):\s*$", m.group(1), re.M)
    return out


def data_layout() -> list[dict]:
    rows = []
    for name in ["calculated_data", "jra_baba", "knowledge", "local", "page_reference", "queue", "research", "skill_logs",
                 "features", "raw", "meta", "mock", "processed"]:
        p = ROOT / "data" / name
        if p.exists():
            try:
                n = sum(1 for _ in os.scandir(p))
            except Exception:
                n = -1
            rows.append({"path": f"data/{name}", "exists": True, "direct_entries": n})
        else:
            rows.append({"path": f"data/{name}", "exists": False, "direct_entries": 0})
    feats = ROOT / "data/features"
    if feats.exists():
        rows.append({"path": "data/features(blocks)", "exists": True,
                     "blocks": sorted(d.name for d in feats.iterdir() if d.is_dir())})
    return rows


def ops_assets() -> dict:
    def names(rel: str, pat: str = "*") -> list[str]:
        d = ROOT / rel
        return sorted(_rel(p) for p in d.glob(pat) if p.is_file()) if d.exists() else []

    return {
        "cron": names("scripts/cron"),
        "server": names("scripts/server"),
        "gcp": names("scripts/gcp"),
        "docker": [_rel(p) for p in ROOT.glob("Dockerfile*") if not p.name.endswith("dockerignore")]
                  + [_rel(p) for p in ROOT.glob("docker-compose*.yml")],
        "ci": names(".github/workflows"),
        "alembic_versions": len(list((ROOT / "alembic/versions").glob("*.py"))) if (ROOT / "alembic/versions").exists() else 0,
        "env_templates": sorted(p.name for p in ROOT.glob(".env*") if "example" in p.name or "template" in p.name or p.name in (".env.stg", ".env.prod")),
    }


def test_layout() -> list[dict]:
    rows = []
    base = ROOT / "tests"
    if base.exists():
        for d in sorted(x for x in base.iterdir() if x.is_dir() and x.name not in SKIP_DIRS):
            files = list(_walk(d, (".py",)))
            rows.append({"dir": f"tests/{d.name}",
                         "test_files": sum(1 for f in files if (f.name.startswith("test_") or f.name.endswith("_test.py"))
                                           and "manual" not in f.relative_to(ROOT).parts),  # CI は */manual を除外
                         "py_files_incl_helpers": len(files)})
    return rows


def todo_markers(limit_per_file: int = 3, limit_total: int = 120) -> dict:
    """TODO/モック/スタブの痕跡。『設計未了』の候補であり、最終判定は人間（AI）が実装を読んで行う。"""
    hits, per_file = [], {}
    roots = [ROOT / "src", ROOT / "frontend/app", ROOT / "frontend/components", ROOT / "frontend/lib"]
    for base in roots:
        if not base.exists():
            continue
        for f in _walk(base, (".py", ".ts", ".tsx")):
            try:
                for i, line in enumerate(f.open(encoding="utf-8", errors="ignore"), 1):
                    if MARKER_RE.search(line) and not line.lstrip().startswith(("import ", "from ")):
                        rel = _rel(f)
                        per_file[rel] = per_file.get(rel, 0) + 1
                        if per_file[rel] <= limit_per_file and len(hits) < limit_total:
                            hits.append({"file": rel, "line": i, "text": line.strip()[:160]})
            except Exception:
                pass
    top = sorted(per_file.items(), key=lambda kv: -kv[1])[:25]
    return {"hits": hits, "files_with_markers": len(per_file), "top_files": [{"file": k, "count": v} for k, v in top]}


def docs_index() -> list[dict]:
    rows = []
    for p in sorted((ROOT / "docs/html").rglob("*.html")):
        if p.name == "index.html":
            continue
        rows.append({"file": _rel(p), "title": _title(p)})
    for p in sorted((ROOT / "docs/decisions").glob("*.html")):
        rows.append({"file": _rel(p), "title": _title(p)})
    for p in sorted((ROOT / "docs").glob("*.md")) + sorted((ROOT / "docs/operations").glob("*.md")):
        rows.append({"file": _rel(p), "title": p.stem})
    for p in sorted((ROOT / "docs/git_management").rglob("*.md")):
        rows.append({"file": _rel(p), "title": p.stem})
    return rows


def task_docs() -> list[dict]:
    """TODO の実態が書かれているタスク管理ドキュメント（ユーザー作業・環境タスク含む）。"""
    rows = []
    for p in sorted((ROOT / "docs/git_management/todo").glob("*.md")):
        text = p.read_text(encoding="utf-8", errors="ignore")
        rows.append({
            "file": _rel(p),
            "open_checkboxes": len(re.findall(r"^\s*[-*] \[ \]", text, re.M)),
            "done_checkboxes": len(re.findall(r"^\s*[-*] \[[xX]\]", text, re.M)),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="keiba-vpn の現状を JSON で stdout に出力する（要件書更新の下調べ）")
    ap.add_argument("--doc", type=Path, default=REQ_DOC, help="前回版の要件書パス（既定: docs/html/index.html）")
    doc = ap.parse_args().doc.resolve()
    prev = previous_doc(doc)
    state = {
        "generated_for_root": str(ROOT),
        "previous_doc": prev,
        "git": git_state(prev["commit"]),
        "decisions": decisions(),
        "python_packages": python_packages(),
        "endpoints": endpoints(),
        "frontend_pages": frontend_pages(),
        "mlflow_models": mlflow_models(),
        "data_layout": data_layout(),
        "ops_assets": ops_assets(),
        "tests": test_layout(),
        "todo_markers": todo_markers(),
        "task_docs": task_docs(),
        "docs_index": docs_index(),
    }
    json.dump(state, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
