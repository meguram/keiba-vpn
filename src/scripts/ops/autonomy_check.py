"""『ホスト(stg)なしで定期実行が動くか』を、クラウドで動かす前に機械的に確かめる 2 つの検査。

  coverage --crontab <ファイル>   ホストの crontab に載っているジョブが、すべて移行先の表（config/gcp_job_map.json）に載っているか。
                                  載っていないジョブは、ホストを止めた瞬間に黙って消える。
  deps [--image jobs|cloudrun]     移行先が GCP のジョブの実行モジュール（steps の ``-m``）が読み込む外部ライブラリが、
                                  イメージの要件（Dockerfile.jobs の requirements）に入っているか。入っていなければ初回の実行で落ちる。

どちらも読み取りだけ。実際に動くことの証明（並走・合否・停止試験）は cron_cutover_status と decoupling_plan.md の 6 章が担う。
終了コード: 0=問題なし、1=準備済み(ready)のジョブに問題あり／載っていないジョブあり。
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# Dockerfile.jobs が requirements.txt から除く接頭辞（イメージに入らない）
JOBS_EXCLUDE = re.compile(r"^(ipykernel|ipywidgets|pytest|playwright|mlflow|optuna|cloud-sql)", re.I)
_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_names(path: Path, exclude: re.Pattern | None = None) -> set[str]:
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0]
        if not line.strip() or line.lstrip().startswith("-") or (exclude and exclude.match(line.strip())):
            continue
        m = _NAME.match(line)
        if m:
            out.add(norm(m.group(1)))
    return out


def expand_with_dependencies(names: set[str]) -> set[str]:
    """要件の依存先（requests→urllib3 など）も入っているものとして数える。この環境にインストールされているものだけ辿れる。"""
    seen: set[str] = set()
    todo = list(names)
    while todo:
        n = todo.pop()
        if n in seen:
            continue
        seen.add(n)
        try:
            reqs = metadata.requires(n) or []
        except metadata.PackageNotFoundError:
            continue
        for r in reqs:
            if "extra ==" in r:
                continue
            m = _NAME.match(r)
            if m:
                todo.append(norm(m.group(1)))
    return seen


def image_requirements(image: str, root: Path = ROOT) -> set[str]:
    if image == "cloudrun":
        return expand_with_dependencies(requirement_names(root / "deploy/docker/requirements-cloudrun.txt"))
    return expand_with_dependencies(requirement_names(root / "requirements.txt", JOBS_EXCLUDE))


def _module_file(mod: str, root: Path) -> Path | None:
    p = root / Path(*mod.split("."))
    if (p.with_suffix(".py")).is_file():
        return p.with_suffix(".py")
    if (p / "__init__.py").is_file():
        return p / "__init__.py"
    return None


def imports_of(path: Path) -> list[tuple[str, bool]]:
    """(インポート名, 常に読み込まれるか)。関数の中・try の中のインポートは『常に』ではない。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[str, bool]] = []

    def walk(node: ast.AST, hard: bool) -> None:
        for ch in ast.iter_child_nodes(node):
            h = hard and not isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Try))
            if isinstance(ch, ast.Import):
                found.extend((a.name, h) for a in ch.names)
            elif isinstance(ch, ast.ImportFrom) and ch.module and ch.level == 0:
                found.append((ch.module, h))
                found.extend((f"{ch.module}.{a.name}", h) for a in ch.names)    # from pkg import submodule
            walk(ch, h)

    walk(tree, True)
    return found


def closure(module: str, root: Path = ROOT) -> tuple[dict[str, bool], list[str]]:
    """モジュールから辿れる自前コード(src.*)の全部と、外部ライブラリ(トップレベル名 → 常に読み込まれるか)。"""
    seen: set[str] = set()
    external: dict[str, bool] = {}
    missing: list[str] = []
    stack = [(module, True)]
    while stack:
        mod, hard = stack.pop()
        if (mod, hard) in seen:
            continue
        seen.add((mod, hard))
        f = _module_file(mod, root)
        if f is None:
            continue
        # 親パッケージの __init__ も読み込まれる
        parts = mod.split(".")
        for i in range(1, len(parts)):
            stack.append((".".join(parts[:i]), hard))
        for name, h in imports_of(f):
            top = name.split(".")[0]
            if top == "src":
                if _module_file(name, root) is not None:
                    stack.append((name, hard and h))
            elif top not in sys.stdlib_module_names and top != "__future__":
                external[top] = external.get(top, False) or (hard and h)
    return external, missing


def dist_of(top: str, mapping: dict[str, list[str]]) -> str | None:
    d = mapping.get(top)
    return norm(d[0]) if d else None


def check_deps(jobs: list[dict], reqs: set[str], root: Path = ROOT, mapping: dict[str, list[str]] | None = None) -> list[dict]:
    mapping = mapping if mapping is not None else metadata.packages_distributions()
    rows = []
    for j in jobs:
        if j.get("destination") != "gcp":
            continue
        miss_hard: set[str] = set()
        miss_soft: set[str] = set()
        unknown: set[str] = set()
        for st in j.get("steps") or []:
            argv = st.get("argv") or []
            if len(argv) < 2 or argv[0] != "-m":
                continue
            ext, _ = closure(argv[1], root)
            for top, hard in ext.items():
                dist = dist_of(top, mapping)
                if dist is None:
                    unknown.add(top)
                elif dist not in reqs:
                    (miss_hard if hard else miss_soft).add(dist)
        rows.append({"id": j["id"], "ready": bool(j.get("ready")), "missing_hard": sorted(miss_hard), "missing_soft": sorted(miss_soft - miss_hard),
                     "unknown_import": sorted(unknown)})
    return rows


def cron_names(crontab_text: str) -> set[str]:
    return set(re.findall(r"cron_run\.sh\s+([A-Za-z0-9_-]+)\s+--", "\n".join(l for l in crontab_text.splitlines() if not l.lstrip().startswith("#"))))


def check_coverage(names: set[str], jobs: list[dict]) -> list[str]:
    known: set[str] = set()
    for j in jobs:
        known.add(j["id"])
        cid = (j.get("origin") or {}).get("cron_job_id")
        if cid:
            known.add(cid)
    return sorted(n for n in names if n not in known)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coverage")
    c.add_argument("--crontab", required=True, help="ホストで `crontab -l` した出力のファイル（- で標準入力）")
    d = sub.add_parser("deps")
    d.add_argument("--image", choices=["jobs", "cloudrun"], default="jobs")
    d.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    jobs = json.loads((ROOT / "config/gcp_job_map.json").read_text(encoding="utf-8"))["jobs"]
    if a.cmd == "coverage":
        text = sys.stdin.read() if a.crontab == "-" else Path(a.crontab).read_text(encoding="utf-8")
        names = cron_names(text)
        miss = check_coverage(names, jobs)
        print(f"ホストの cron {len(names)} 本のうち、移行先の表に無いもの: {len(miss)} 本")
        for n in miss:
            print(f"  - {n}")
        return 1 if miss else 0
    rows = check_deps(jobs, image_requirements(a.image))
    bad = [r for r in rows if r["missing_hard"] or r["missing_soft"]]
    if a.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
    else:
        print(f"イメージ({a.image})の要件に無い外部ライブラリを読むジョブ: {len(bad)} / {len(rows)}（常に読む=必ず落ちる、条件付き=その経路で落ちる）")
        for r in bad:
            print(f"  {'[準備済み]' if r['ready'] else '[未着手  ]'} {r['id']}: 必ず={r['missing_hard'] or '-'} 条件付き={r['missing_soft'] or '-'}")
    return 1 if any(r["ready"] and r["missing_hard"] for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
