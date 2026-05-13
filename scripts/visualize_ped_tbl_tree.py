#!/usr/bin/env python3
"""
ped_tbl の Parquet（path_fm 付き）を HTML で可視化する。

- **netkeiba 風**: 1〜5 世代を「世代ごと 2^g 列」のブロックグリッド（血統表のピラミッド）で表示。
- **接木分**: 5 世代目の種牡馬（アンカー）ごとにブロックを分け、各ブロック内で 6〜10 代目を
  **主表と同じ 2^lg 列グリッド**（枝の 1〜5 代目＝全体の 6〜10 代目相当）で表示。
- **参考**: 折りたたみで path_fm 分岐ツリー・生 JSON。

例::

  python3 scripts/visualize_ped_tbl_tree.py \\
    data/features/horse/ped_tbl/2009/2009100502.parquet \\
    -o docs/html/ped_tree_2009100502.html
"""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _fmt_cell(v: Any) -> str:
    if v is None:
        return ""
    s = str(v)
    if s in ("<NA>", "nan", "None"):
        return ""
    return s


PRIMARY_MAX_GEN = 5


def _anchor_name_from_primary(
    primary: dict[tuple[int, int], dict[str, Any]], anchor_id: str
) -> str:
    for r in primary.values():
        if _fmt_cell(r.get("ancestor_horse_id")) == anchor_id:
            return _fmt_cell(r.get("ancestor_name"))
    return ""


def _index_primary_by_gen_pos(rows: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    m: dict[tuple[int, int], dict[str, Any]] = {}
    for r in rows:
        if _fmt_cell(r.get("source")) != "primary":
            continue
        try:
            g = int(r.get("generation") or 0)
            p = int(r.get("position") or 0)
        except (TypeError, ValueError):
            continue
        m[(g, p)] = r
    return m


def _cell_inner(r: dict[str, Any] | None, *, is_sire_slot: bool) -> str:
    if r:
        nm = _fmt_cell(r.get("ancestor_name"))
        hid = _fmt_cell(r.get("ancestor_horse_id"))
        src = _fmt_cell(r.get("source"))
        cls = "src-primary" if src == "primary" else "src-merged" if "merged" in src else "src-other"
        return (
            f"<div class='cell-name'>{html.escape(nm)}</div>"
            f"<div class='cell-id'><code>{html.escape(hid)}</code></div>"
            f"<div><span class='src {cls}'>{html.escape(src or '—')}</span></div>"
        )
    if is_sire_slot:
        return "<div class='cell-empty'>牡スロット<br/><span class='muted'>（データなし）</span></div>"
    return "<div class='cell-mare'>牝</div>"


def render_netkeiba_blocks(
    rows: list[dict[str, Any]],
    *,
    subject_id: str,
    subject_name: str,
) -> str:
    primary = _index_primary_by_gen_pos(rows)
    max_g = max((g for (g, _) in primary), default=0)
    max_primary = min(PRIMARY_MAX_GEN, max(max_g, PRIMARY_MAX_GEN)) if primary else PRIMARY_MAX_GEN

    lines: list[str] = []
    lines.append("<section class='ped-netkeiba'>")
    lines.append("<h2>主表（1〜5世代）・netkeiba 風ブロック</h2>")
    lines.append(
        "<p class='hint'>各段は<strong>左から position 0,1,…</strong>。"
        "偶数＝牡（種牡馬）スロット、奇数＝牝スロット（本データでは牝 ID は持たないため空ブロック）。</p>"
    )

    for g in range(1, max_primary + 1):
        ncols = 2**g
        lines.append(f"<div class='gen-wrap'><div class='gen-label'>{g} 代目（{ncols} 頭分）</div>")
        lines.append(f"<div class='ped-row' style='--cols:{ncols}'>")
        for p in range(ncols):
            r = primary.get((g, p))
            is_sire = p % 2 == 0
            cls = "ped-cell"
            if r:
                cls += " has-data"
                if "merged" in _fmt_cell(r.get("source")):
                    cls += " merged"
            elif is_sire:
                cls += " sire-slot empty"
            else:
                cls += " mare-slot"
            inner = _cell_inner(r, is_sire_slot=is_sire)
            lines.append(f"<div class='{cls}' data-pos='{p}'>{inner}</div>")
        lines.append("</div></div>")

    lines.append("</section>")

    merged = [r for r in rows if "merged" in _fmt_cell(r.get("source"))]
    if merged:
        by_anchor: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in merged:
            aid = _fmt_cell(r.get("anchor_horse_id"))
            if not aid:
                continue
            by_anchor[aid].append(r)

        lines.append("<section class='ped-extended'>")
        lines.append("<h2>接木（6〜10 世代相当）・netkeiba 風ブロック</h2>")
        lines.append(
            "<p class='hint'>5 世代目の<strong>種牡馬（アンカー）ごと</strong>に枠を分け、"
            "各枠の中は<strong>枝血統表と同じ並び</strong>（枝の <em>lg</em> 代目＝ <em>5+lg</em> 代目相当、"
            "列数 2<sup>lg</sup>、左から <code>position</code> 0,1,…）。牡スロットのみデータあり、牝はグレーブロック。</p>"
        )

        for anchor_id in sorted(by_anchor):
            lst = by_anchor[anchor_id]
            idx: dict[tuple[int, int], dict[str, Any]] = {}
            for r in lst:
                try:
                    gg = int(r.get("generation") or 0)
                    pp = int(r.get("position") or 0)
                except (TypeError, ValueError):
                    continue
                idx[(gg, pp)] = r

            anm = _anchor_name_from_primary(primary, anchor_id)
            title = f"アンカー <code>{html.escape(anchor_id)}</code>"
            if anm:
                title += f" <span class='anchor-nm'>{html.escape(anm)}</span>"
            lines.append(f"<div class='anchor-block'><div class='anchor-title'>{title}</div>")

            for lg in range(1, PRIMARY_MAX_GEN + 1):
                global_g = PRIMARY_MAX_GEN + lg
                ncols = 2**lg
                lines.append(
                    f"<div class='gen-wrap'><div class='gen-label'>"
                    f"{global_g} 代目相当（枝の {lg} 代目・{ncols} 列）</div>"
                )
                lines.append(f"<div class='ped-row' style='--cols:{ncols}'>")
                for p in range(ncols):
                    r = idx.get((global_g, p))
                    is_sire = p % 2 == 0
                    cls = "ped-cell"
                    if r:
                        cls += " has-data merged"
                    elif is_sire:
                        cls += " sire-slot empty"
                    else:
                        cls += " mare-slot"
                    inner = _cell_inner(r, is_sire_slot=is_sire)
                    lines.append(f"<div class='{cls}' data-pos='{p}'>{inner}</div>")
                lines.append("</div></div>")

            lines.append("</div>")

        lines.append("</section>")

    return "".join(lines)


def build_tree(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """path_fm の trie（参考ツリー用）。"""
    root: dict[str, Any] = {"children": {}, "rows": []}
    for r in rows:
        path = str(r.get("path_fm") or "")
        meta = {k: _fmt_cell(r.get(k)) for k in r}
        cur = root
        for ch in path:
            if ch not in ("F", "M"):
                continue
            if ch not in cur["children"]:
                cur["children"][ch] = {"step": ch, "depth": 0, "children": {}, "rows": []}
            cur = cur["children"][ch]
        cur.setdefault("rows", []).append(meta)
    return root


def _render_subtree(node: dict[str, Any], *, depth: int) -> str:
    ch = node.get("step")
    rows_html = ""
    if node.get("rows"):
        bits = []
        for r in node["rows"]:
            nm = r.get("ancestor_name") or ""
            hid = r.get("ancestor_horse_id") or ""
            src = r.get("source") or ""
            cls = "src-primary" if src == "primary" else "src-merged" if "merged" in src else "src-other"
            bits.append(
                "<div class='sire-slot'>"
                f"<span class='name'>{html.escape(nm)}</span> "
                f"<code>{html.escape(hid)}</code> "
                f"<span class='src {cls}'>{html.escape(src)}</span>"
                "</div>"
            )
        side = "父側 (F)" if ch == "F" else "母側 (M)"
        rows_html = (
            f"<div class='node depth-{depth}'>"
            f"<span class='edge'>{html.escape(side)}</span><div class='slots'>{''.join(bits)}</div></div>"
        )
    children = node.get("children") or {}
    if not children:
        return rows_html
    keys = sorted(children.keys(), key=lambda k: (0 if k == "F" else 1, k))
    lis = [f"<li>{_render_subtree(children[k], depth=depth + 1)}</li>" for k in keys]
    return rows_html + "<ul class='tree'>" + "".join(lis) + "</ul>"


def _render_tree_reference(rows: list[dict[str, Any]], subject_id: str, subject_name: str) -> str:
    tree = build_tree(rows)
    subj = (
        f"<div class='node root'><strong>対象馬</strong> <code>{html.escape(subject_id)}</code>"
        f" <span class='name'>{html.escape(subject_name)}</span></div>"
    )
    keys = sorted(tree["children"].keys(), key=lambda k: (0 if k == "F" else 1, k))
    branches = [f"<li>{_render_subtree(tree['children'][k], depth=1)}</li>" for k in keys]
    return subj + "<ul class='tree'>" + "".join(branches) + "</ul>"


def render_html(rows: list[dict[str, Any]], *, title: str, subject_id: str, subject_name: str) -> str:
    if not rows:
        body = "<p>データがありません。</p>"
        tree_ref = ""
    else:
        body = render_netkeiba_blocks(rows, subject_id=subject_id, subject_name=subject_name)
        tree_ref = _render_tree_reference(rows, subject_id, subject_name)

    raw = html.escape(json.dumps(rows[:200], ensure_ascii=False, indent=2, default=str))
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: "Hiragino Sans", "Yu Gothic UI", Meiryo, system-ui, sans-serif;
      margin: 1rem; background: #f0f2f5; color: #222; line-height: 1.45; }}
    h1 {{ font-size: 1.2rem; margin-bottom: 0.25rem; }}
    h2 {{ font-size: 1rem; margin: 1rem 0 0.5rem; color: #333; border-bottom: 2px solid #c62828; padding-bottom: 0.2rem; }}
    .subject-bar {{
      background: linear-gradient(180deg, #fff 0%, #fafafa 100%);
      border: 1px solid #ddd; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1rem;
      box-shadow: 0 1px 3px rgba(0,0,0,.06);
    }}
    .subject-bar code {{ font-size: 1rem; }}
    .note {{ color: #555; font-size: 0.88rem; margin-bottom: 0.5rem; max-width: 56rem; }}
    .hint {{ font-size: 0.8rem; color: #666; margin: 0 0 0.6rem; max-width: 56rem; }}

    .ped-netkeiba, .ped-extended {{ background: #fff; border: 1px solid #ddd; border-radius: 8px;
      padding: 0.75rem 1rem 1rem; margin-bottom: 1rem; box-shadow: 0 1px 2px rgba(0,0,0,.05); }}

    .gen-wrap {{ margin-bottom: 1rem; }}
    .gen-label {{
      font-size: 0.75rem; font-weight: 700; color: #555; margin-bottom: 0.35rem;
      letter-spacing: 0.02em;
    }}

    .ped-row {{
      display: grid; grid-template-columns: repeat(var(--cols), 1fr); gap: 3px;
      width: 100%;
    }}
    .anchor-block {{
      margin: 1rem 0; padding: 0.65rem 0.75rem 0.85rem;
      border: 1px solid #ffb74d; border-radius: 8px; background: linear-gradient(180deg, #fff8e7 0%, #fff 0.5rem);
    }}
    .anchor-title {{ font-size: 0.82rem; font-weight: 700; color: #5d4037; margin-bottom: 0.45rem; }}
    .anchor-title .anchor-nm {{ font-weight: 600; color: #333; margin-left: 0.35rem; }}
    .ped-cell {{
      border: 1px solid #bbb; border-radius: 4px; padding: 5px 4px;
      font-size: 0.68rem; min-height: 3.4rem;
      display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;
      background: #fff; overflow: hidden; word-break: break-all;
    }}
    .ped-cell.has-data {{ background: #fffef7; border-color: #999; }}
    .ped-cell.has-data.merged {{ background: #fff8e1; border-color: #e65100; }}
    .ped-cell.sire-slot.empty {{ background: #fafafa; border-style: dashed; color: #888; }}
    .ped-cell.mare-slot {{ background: #eceff1; color: #78909c; border-color: #cfd8dc; }}
    .cell-name {{ font-weight: 700; font-size: 0.7rem; line-height: 1.25; max-height: 2.6em; overflow: hidden; }}
    .cell-id code {{ font-size: 0.62rem; background: #eee; padding: 0 0.2rem; border-radius: 2px; }}
    .cell-empty, .cell-mare {{ font-size: 0.68rem; }}
    .cell-mare {{ font-weight: 600; }}
    .muted {{ font-size: 0.62rem; color: #999; }}
    .cell-meta {{ font-size: 0.62rem; margin-top: 2px; }}
    .path-fm {{ margin-top: 3px; }}
    .path-fm code {{ font-size: 0.58rem; color: #444; }}

    .src {{ font-size: 0.62rem; padding: 0.05rem 0.25rem; border-radius: 2px; }}
    .src-primary {{ background: #e3f2fd; color: #1565c0; }}
    .src-merged {{ background: #ffe0b2; color: #e65100; }}

    details {{ margin-top: 1rem; font-size: 0.85rem; background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 0.5rem 1rem; }}
    .tree {{ list-style: none; padding-left: 0.75rem; margin: 0.25rem 0; border-left: 1px solid #ccc; }}
    .tree li {{ margin: 0.3rem 0; }}
    .node.root {{ margin-bottom: 0.4rem; }}
    .edge {{ font-weight: 600; color: #444; margin-right: 0.35rem; }}
    pre {{ background: #1e1e1e; color: #d4d4d4; padding: 0.75rem; overflow: auto; max-height: 380px; font-size: 0.72rem; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="subject-bar">
    <strong>対象馬</strong> <code>{html.escape(subject_id)}</code>
    <span class='name'>{html.escape(subject_name)}</span>
  </div>
  <p class="note">
    上段から netkeiba の血統表と同様に、<strong>世代ごとに左から position 順</strong>のブロックです。
    接木がある馬は下のセクションで、<strong>アンカー種牡馬ごと</strong>に 6 代目以降を netkeiba と同様のグリッドで表示します。
  </p>
  {body}
  <details>
    <summary>path_fm 分岐ツリー（参考）</summary>
    {tree_ref if rows else "<p>なし</p>"}
  </details>
  <details>
    <summary>生データ（先頭200行・JSON）</summary>
    <pre id="raw">{raw}</pre>
  </details>
</body>
</html>
"""


def main() -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(description="ped_tbl Parquet を HTML（netkeiba風ブロック）に出力")
    ap.add_argument("parquet", type=Path, help="ped_tbl の .parquet パス")
    ap.add_argument("-o", "--out", type=Path, required=True, help="出力 .html")
    args = ap.parse_args()
    pq = args.parquet.resolve()
    if not pq.is_file():
        raise SystemExit(f"ファイルがありません: {pq}")
    df = pd.read_parquet(pq)
    rows = []
    for r in df.to_dict("records"):
        rows.append({k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()})
    hid = str(df.iloc[0]["subject_horse_id"]) if not df.empty else pq.stem
    sn = _fmt_cell(df.iloc[0].get("subject_horse_name")) if not df.empty else ""
    title = f"血統表 {hid}"
    html_out = render_html(rows, title=title, subject_id=hid, subject_name=sn)
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_out, encoding="utf-8")
    print(f"Wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
