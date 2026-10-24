"""運用経路のコードが、ホストPC固有のもの（パス・ポート・別名・ローカルデータ・DB）に依存していないかを機械的に調べる。

最終形（docs/operations/decoupling_plan.md）は『ホストPC(stg)と無関係に公開』。ホストは研究専用になるので、運用の経路
（API・取得・予測の配信・監視・VPS/GCP の道具）にホスト固有の参照が残っていないことを、このツールで確かめる。

方式は許可リスト（ベースライン）: 今ある依存は ``config/host_dependency_baseline.json`` に件数で記録してあり、
**ベースラインを超えて増えたとき**だけ失敗する（新しい依存を作らせない）。依存を減らしたら ``--update-baseline`` で下げる。
``--strict`` は、ベースライン内のものも『返すべき負債』として全部出す。

  python -m src.scripts.ops.host_dependency_scan                    # 増えていないか検査（終了コード 0/1）
  python -m src.scripts.ops.host_dependency_scan --strict --summary  # 負債の一覧と件数
  python -m src.scripts.ops.host_dependency_scan --update-baseline   # 依存を減らしたあと、ベースラインを書き直す
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
BASELINE_PATH = ROOT / "config" / "host_dependency_baseline.json"

# 運用経路（公開が動くために実行時または運用で使うもの）。研究用（tp-*, research, notebooks, モデル学習）は含めない
OPERATIONAL_GLOBS = (
    "src/api/**/*.py",
    "src/utils/**/*.py",
    "src/ops/**/*.py",
    "src/monitor/**/*.py",
    "src/scraper/**/*.py",
    "src/config/**/*.py",
    "src/pipeline/inference/**/*.py",
    "src/scripts/ops/**/*.py",
    "src/scripts/cloud_jobs/**/*.py",
    "scripts/ops/**/*",
    "scripts/gcp/**/*",
    "scripts/server/**/*",
    "frontend/next.config.js",
    "frontend/middleware.ts",
    "frontend/lib/**/*.ts",
    "frontend/app/**/*.ts",
    "frontend/app/**/*.tsx",
    "frontend/components/**/*.tsx",
)
EXCLUDE_GLOBS = (
    "**/__pycache__/**",
    "**/node_modules/**",
    "src/scripts/ops/host_dependency_scan.py",  # このツール自身（パターンを文字列で持つ）
)
SUFFIXES = {".py", ".sh", ".ts", ".tsx", ".js", ".conf", ".service", ".timer", ".env", ".json", ".example"}

# 区分 → (正規表現, 重大度, 説明)
PATTERNS: dict[str, tuple[re.Pattern[str], str, str]] = {
    "host_path": (re.compile(r"/home/jovyan"), "high", "ホストの絶対パス"),
    "host_alias": (re.compile(r"keiba-remote"), "high", "ホストへの ssh 別名"),
    "host_port": (re.compile(r"(?<![\d.])(?:127\.0\.0\.1|localhost):232\d\d"), "high", "stg(ホスト)のポートへの直接参照"),
    "host_port_literal": (re.compile(r"(?<!\d)232\d\d(?!\d)"), "medium", "stg(ホスト)のポート番号の直書き"),
    "api_internal": (re.compile(r"api-internal\.megu-keiba\.com"), "high", "ホストの API を指す内部ホスト名"),
    "local_data": (re.compile(r"data/local/|\"data\"\s*/\s*\"local\"|'data'\s*/\s*'local'"), "medium", "ホストのローカルデータ(data/local)の参照"),
    "page_reference": (re.compile(r"data/page_reference|\"page_reference\""), "medium", "UI バンドル(data/page_reference)の参照"),
    "postgres": (re.compile(r"\bpsycopg2?\b|\bsqlalchemy\b|ensure_postgres|sync_pg_from_gcs"), "medium", "ホストの PostgreSQL"),
}
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _is_text_candidate(p: Path) -> bool:
    return p.suffix in SUFFIXES or p.name in {"Dockerfile", "Makefile"} or p.suffix == ""


def _tracked(root: Path) -> set[str] | None:
    """git で追跡中のファイル（相対パス）。git が使えなければ None。"""
    import subprocess

    try:
        r = subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return set(r.stdout.splitlines())


def iter_files(root: Path, tracked_only: bool = False) -> list[Path]:
    tracked = _tracked(root) if tracked_only else None
    seen: set[Path] = set()
    for pat in OPERATIONAL_GLOBS:
        for p in root.glob(pat):
            if p.is_file() and _is_text_candidate(p):
                seen.add(p)
    out = []
    for p in sorted(seen):
        rel = p.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel, g) for g in EXCLUDE_GLOBS):
            continue
        if tracked is not None and rel not in tracked:
            continue
        out.append(p)
    return out


def scan(root: Path = ROOT, tracked_only: bool = False) -> list[dict[str, Any]]:
    """各行を走査して findings を返す: file / line / kind / severity / text。tracked_only=True は git 追跡中のファイルだけ。"""
    findings: list[dict[str, Any]] = []
    for p in iter_files(root, tracked_only):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = p.relative_to(root).as_posix()
        for i, line in enumerate(text.splitlines(), 1):
            for kind, (rx, sev, _doc) in PATTERNS.items():
                if rx.search(line):
                    findings.append({"file": rel, "line": i, "kind": kind, "severity": sev, "text": line.strip()[:160]})
    return findings


def counts(findings: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    c: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for f in findings:
        c[f["file"]][f["kind"]] += 1
    return {k: dict(v) for k, v in sorted(c.items())}


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, dict[str, int]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("counts", {})


def compare(current: dict[str, dict[str, int]], baseline: dict[str, dict[str, int]]) -> list[str]:
    """ベースラインを超えて増えた（または新しいファイルに出た）依存を文章で返す。"""
    problems: list[str] = []
    for f, kinds in current.items():
        for kind, n in kinds.items():
            allowed = baseline.get(f, {}).get(kind, 0)
            if n > allowed:
                problems.append(f"{f}: {kind} が {allowed} → {n} 件に増えた（{PATTERNS[kind][2]}）")
    return problems


def summary(findings: list[dict[str, Any]]) -> list[str]:
    by_kind = Counter(f["kind"] for f in findings)
    files = {k: len({f["file"] for f in findings if f["kind"] == k}) for k in by_kind}
    rows = sorted(by_kind, key=lambda k: (SEVERITY_ORDER[PATTERNS[k][1]], -by_kind[k]))
    out = ["区分                 重大度  件数  ファイル数  内容"]
    for k in rows:
        out.append(f"{k:<20} {PATTERNS[k][1]:<7} {by_kind[k]:>4}  {files[k]:>9}  {PATTERNS[k][2]}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--baseline", default=str(BASELINE_PATH))
    ap.add_argument("--tracked-only", action="store_true", help="git で追跡中のファイルだけ調べる（作業中の未コミットを除く。CI と同じ見え方）")
    ap.add_argument("--strict", action="store_true", help="ベースライン内のものも含めて全部出す（返すべき負債の一覧）")
    ap.add_argument("--summary", action="store_true", help="区分ごとの件数表を出す")
    ap.add_argument("--update-baseline", action="store_true", help="現在の件数をベースラインに書く（依存を減らしたあと）")
    a = ap.parse_args(argv)

    root = Path(a.root)
    findings = scan(root, a.tracked_only)
    cur = counts(findings)
    base_path = Path(a.baseline)

    if a.update_baseline:
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text(json.dumps({"_doc": "ホスト固有の参照の許容件数(ファイル×区分)。増えたら host_dependency_scan が失敗する。"
                                         " 依存を減らしたら --update-baseline で下げる。", "counts": cur}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"ベースラインを更新: {base_path}（{sum(sum(v.values()) for v in cur.values())} 件 / {len(cur)} ファイル）")
        return 0

    if a.summary:
        print("\n".join(summary(findings)))
    if a.strict:
        for f in sorted(findings, key=lambda x: (SEVERITY_ORDER[x["severity"]], x["file"], x["line"])):
            print(f"[{f['severity']}] {f['file']}:{f['line']} {f['kind']}: {f['text']}")

    problems = compare(cur, load_baseline(base_path))
    if problems:
        print("ホスト固有の依存が増えています:", file=sys.stderr)
        for p in problems:
            print("  - " + p, file=sys.stderr)
        return 1
    if not a.strict and not a.summary:
        print(f"OK: ホスト依存はベースライン以内（{len(findings)} 件 / {len(cur)} ファイル。--strict --summary で一覧）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
