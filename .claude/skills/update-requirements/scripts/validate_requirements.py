#!/usr/bin/env python3
"""生成した docs/html/index.html（要件定義書）の機械検証。

検査項目:
  - 必須セクション (id) が揃い、順序どおりで、scheduling が最後の <section> であること
  - 未置換プレースホルダ ({{...}}) / GUIDE コメントが残っていないこと
  - requirements-commit / requirements-updated メタが埋まっていること
  - TODO 行: ID 一意・必須属性・依存先の実在・循環なし・依存先が上に並ぶ・優先度順
  - §0 の件数カード(data-card)が §11 内の状況タグ数と一致すること（dep は対象外）
  - 関連ドキュメントの相対リンク先が実在すること

Usage:
    python3 .claude/skills/update-requirements/scripts/validate_requirements.py [path/to/index.html]
終了コード: 0=OK, 1=エラーあり
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEFAULT = ROOT / "docs/html/index.html"
SECTIONS = ["summary", "overview", "requirements", "kpi", "architecture", "data-flow", "ml-flow",
            "delivery", "infra", "quality", "decisions", "status-matrix", "risks", "related-docs", "scheduling"]
PRIO = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    raw = path.read_text(encoding="utf-8")
    errors: list[str] = []
    warns: list[str] = []

    if "GUIDE:" in raw:
        errors.append("テンプレートの GUIDE コメントが残っています")
    text = re.sub(r"<!--.*?-->", "", raw, flags=re.S)

    for m in re.finditer(r"\{\{[^}]*\}\}", text):
        errors.append(f"未置換プレースホルダ: {m.group(0)}")
    for name in ("requirements-commit", "requirements-updated"):
        m = re.search(rf'<meta name="{name}" content="([^"]*)"', text)
        if not m or not m.group(1).strip():
            errors.append(f"meta {name} が空です")

    ids = re.findall(r'<section id="([\w-]+)"', text)
    if ids != SECTIONS:
        errors.append(f"セクション構成/順序が不正。期待={SECTIONS} 実際={ids}")
    elif ids[-1] != "scheduling":
        errors.append("scheduling が最後のセクションではありません")

    rows = []
    for tr in re.finditer(r"<tr\b([^>]*\bdata-todo-id=[^>]*)>", text):
        rows.append(dict(re.findall(r'data-([\w-]+)="([^"]*)"', tr.group(1))))
    seen: dict[str, int] = {}
    for idx, r in enumerate(rows):
        tid = r.get("todo-id", "")
        if tid in seen:
            errors.append(f"TODO ID 重複: {tid}")
        seen[tid] = idx
        for attr in ("priority", "kind", "depends", "order", "status"):
            if attr not in r:
                errors.append(f"{tid}: data-{attr} がありません")
        if not re.fullmatch(r"T-\d{3,}", tid):
            errors.append(f"{tid}: TODO-ID は T- + 3桁以上の数字（例: T-001）。T-45 など業務用語と区別するため")
        if r.get("priority") not in PRIO:
            errors.append(f"{tid}: priority は P0-P3 (実際: {r.get('priority')})")
        if r.get("kind") not in ("TODO", "MOCK"):
            errors.append(f"{tid}: kind は TODO|MOCK (実際: {r.get('kind')})")
        if r.get("status") not in ("open", "doing", "done"):
            errors.append(f"{tid}: status は open|doing|done (実際: {r.get('status')})")
    if not rows:
        errors.append("TODO 行が 1 件もありません（§14 を確認）")

    deps = {r["todo-id"]: [d for d in r.get("depends", "").split(",") if d.strip()] for r in rows if "todo-id" in r}
    for tid, ds in deps.items():
        for d in ds:
            if d not in deps:
                errors.append(f"{tid}: 依存先 {d} が存在しません")
            elif seen[d] > seen[tid]:
                errors.append(f"{tid}: 依存先 {d} が自分より下に並んでいます（依存先を上に）")
    # 循環検出
    state: dict[str, int] = {}

    def dfs(n: str, stack: list[str]) -> None:
        state[n] = 1
        for d in deps.get(n, []):
            if d not in deps:
                continue
            if state.get(d) == 1:
                errors.append("依存の循環: " + " -> ".join(stack + [n, d]))
            elif d not in state:
                dfs(d, stack + [n])
        state[n] = 2

    for n in deps:
        if n not in state:
            dfs(n, [])
    # フェーズ(phase-head)内の優先度順（警告）
    cur_phase, last = None, -1
    for m in re.finditer(r'(<tr class="phase-head">)|<tr\b[^>]*data-todo-id="([^"]+)"[^>]*data-priority="(P\d)"', text):
        if m.group(1):
            cur_phase, last = m.start(), -1
        elif m.group(3):
            p = PRIO.get(m.group(3), 9)
            if p < last:
                warns.append(f"{m.group(2)}: 同フェーズ内で優先度が逆転（依存による例外なら可）")
            last = max(last, p)
    orders = [int(r["order"]) for r in rows if r.get("order", "").isdigit()]
    if orders != sorted(orders) or len(set(orders)) != len(orders):
        errors.append("data-order が昇順・一意になっていません")

    for href in sorted(set(re.findall(r'href="([^"#:]+\.(?:html|md))(?:#[^"]*)?"', text))):
        if not (path.parent / href).resolve().exists():
            errors.append(f"リンク切れ: {href}")

    sec = re.search(r'<section id="status-matrix".*?(?=<section id=|</main>|$)', text, re.S)
    sec_rows = re.findall(r"<tr\b.*?</tr>", sec.group(0), re.S) if sec else []
    matrix = {k: sum(len(re.findall(rf'class="tag {k}"', r)) for r in sec_rows) for k in ("done", "partial", "mock", "todo", "dep")}
    if sec and len(re.findall(r'class="tag ', sec.group(0))) != sum(matrix.values()):
        errors.append("§11 の表の行の外（本文・凡例）に状況タグがあります。タグは行の状況セルにだけ置く")
    for r in sec_rows:
        if re.search(r'class="tag (?:mock|todo)"', r):
            refs = re.findall(r"(?<![\w-])T-\d{3,}(?!\d)", re.sub(r"<[^>]+>", " ", r))
            if not refs:
                warns.append("§11 の MOCK/TODO 行に §14 の TODO-ID が無い: " + re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", r))[:60])
            for ref in refs:
                if ref not in deps:
                    errors.append(f"§11 が存在しない TODO-ID {ref} を参照しています")
    for k in ("done", "partial", "mock", "todo"):
        m = re.search(rf'data-card="{k}"[^>]*>\s*(\d+)\s*<', text)
        if not m:
            errors.append(f"§0 のカード data-card=\"{k}\" に数値がありません")
        elif int(m.group(1)) != matrix[k]:
            errors.append(f"§0 カード {k}={m.group(1)} が §11 のタグ数 {matrix[k]} と不一致")
    linked = {h.split("#")[0] for h in re.findall(r'href="([^"]+)"', text)}
    on_disk = [f for f in path.parent.rglob("*.html") if f != path and not f.is_symlink()] + \
              list((path.parent.parent / "decisions").glob("*.html")) if path.parent.name == "html" else []
    unlinked = sorted(str(f.relative_to(path.parent.parent)) for f in on_disk
                      if not any((path.parent / h).resolve() == f.resolve() for h in linked if h and not h.startswith(("http", "#"))))
    print(f"[info] §13 載せ漏れ検査の走査対象 html = {len(on_disk)} 件（0 の場合は検査が効いていない）")
    if unlinked:
        warns.append(f"§13 に載っていない html ドキュメントが {len(unlinked)} 件: {unlinked[:4]}")
    tags = {k: len(re.findall(rf'class="tag {k}"', text)) for k in ("done", "partial", "mock", "todo", "dep")}
    print(f"[info] sections={len(ids)} todos={len(rows)} matrix(§11)={matrix} whole_doc_tags(参考)={tags}")
    for w in warns:
        print(f"[warn] {w}")
    for e in errors:
        print(f"[error] {e}")
    print("OK" if not errors else f"NG ({len(errors)} errors)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
