"""種牡馬系統の分析を 1 枚にまとめた HTML を作る（タブで切り替え）。

使い方: python src/scripts/docs/build_sire_line_hub_html.py [OUT.html]
タブ:
  1. まとめ … docs/ideas/sire_dam_influence.md（父と母の影響の言語化・結論）
  2. 父と母の影響 … 形質ごとの寄与（父・母父・母母父・母母母父）、遺伝の割合との比、種牡馬そのもの・4 つとも分かる馬の版、
     LightGBM の重要度、因子分析とクラスタ、向き（上位・下位の小系統）
  3. 小系統の距離配置（走法軸） … MDS の散布図・Ward の樹形図・プロファイルのヒートマップ・近い系統（父として／母父として）
  4. 成績マトリクス … data/meta/sire_lines/sire_line_course_matrix.html（build_sire_line_course_matrix_html.py の出力）を埋め込む
  5. 系統一覧 … docs/html/pedigree/sire_lines_all.html（build_sire_lines_html.py の出力）を埋め込む
  6. 欧州瞬発力の検証 … docs/ideas/euro_kick_sire_lines.md
  7. マーキング … data/meta/sire_lines/euro_kick_marking.html（build_euro_marking_html.py の出力）を埋め込む
埋め込むページは、元の HTML をそのまま iframe（srcdoc）で開く（CSS・JS がぶつからず、並べ替え・マーキングの保存もそのまま動く）。
元の HTML は <textarea hidden> に入れておき、タブを初めて開いたときに iframe を作る（最初の表示を軽くする）。
元のページを更新したときは、各スクリプトで作り直してから、このスクリプトを流す。
入力: data/local/pedigree/sire_dam_influence.json（python -m src.research.pedigree.sire_dam_influence で作る。無ければ 2・3 のタブは「未計算」）
      data/meta/sire_lines/smartrc_line_master.json（大系統の色）
出力: data/meta/sire_lines/sire_line_analysis.html
本（build_docs_book.py）には載せない（本は元ページの id・相対リンクを章ごとに書き換えるので、埋め込んだページが動かなくなる。単独のページとして開く）。
"""

from __future__ import annotations

import html
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "data/local/pedigree/sire_dam_influence.json"
MASTER = REPO / "data/meta/sire_lines/smartrc_line_master.json"
EMBEDS = {
    "matrix": REPO / "data/meta/sire_lines/sire_line_course_matrix.html",
    "lines": REPO / "docs/html/pedigree/sire_lines_all.html",
    "marking": REPO / "data/meta/sire_lines/euro_kick_marking.html",
}
DOCS = {"summary": REPO / "docs/ideas/sire_dam_influence.md", "euro": REPO / "docs/ideas/euro_kick_sire_lines.md"}
OUT = REPO / "data/meta/sire_lines/sire_line_analysis.html"
JST = timezone(timedelta(hours=9))
POS = ("父", "母父", "母母父", "母母母父")
POS_COLOR = {"父": "#2f6fdf", "母父": "#d9480f", "母母父": "#e8a33d", "母母母父": "#9c6ade"}
CSS_COLOR = {"Aqua": "#00ffff", "Moccasin": "#ffe4b5", "LimeGreen": "#32cd32", "DeepPink": "#ff1493", "Sienna": "#a0522d", "Orange": "#ffa500"}
E = html.escape


# ---------------------------------------------------------------- 小さな部品
def md_to_html(md: str) -> str:
    """docs の Markdown を簡単に HTML にする（見出し・箇条書き（入れ子 1 段）・太字・コード・リンク・段落）。表などは段落のまま"""
    out, in_ul, in_sub = [], False, False

    def inline(s: str) -> str:
        s = E(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        return re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', s)

    def close(level: int = 0):
        nonlocal in_ul, in_sub
        if in_sub and level <= 1:
            out.append("</ul>")
            in_sub = False
        if in_ul and level == 0:
            out.append("</ul>")
            in_ul = False

    for line in md.splitlines():
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            close()
            n = min(len(m.group(1)) + 1, 5)
            out.append(f"<h{n}>{inline(m.group(2))}</h{n}>")
            continue
        m = re.match(r"^(\s*)[-*]\s+(.*)", line)
        if m:
            sub = len(m.group(1)) >= 2
            if sub and in_ul:
                if not in_sub:
                    out.append("<ul>")
                    in_sub = True
            else:
                close(1)
                if not in_ul:
                    out.append("<ul>")
                    in_ul = True
            out.append(f"<li>{inline(m.group(2))}</li>")
            continue
        close()
        if line.strip().startswith(">"):
            out.append(f"<p class=note>{inline(line.strip()[1:].strip())}</p>")
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    close()
    return "\n".join(out)


def large_colors() -> dict[str, str]:
    """小系統・大系統の名前 → 大系統の色"""
    try:
        m = json.loads(MASTER.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    large = {c: v for c, v in m.get("large", {}).items()}
    out = {}
    for v in large.values():
        out[v["name"]] = CSS_COLOR.get(v.get("color", ""), v.get("color", "#888"))
    for v in m.get("small", {}).values():
        lg = large.get(v.get("large"))
        if lg:
            out[v["name"]] = CSS_COLOR.get(lg.get("color", ""), lg.get("color", "#888"))
    return {k: (c if re.fullmatch(r"#?[A-Za-z0-9]+", c or "") else "#888") for k, c in out.items()}


def pct(x: float | None, d: int = 1) -> str:
    return "—" if x is None else f"{x * 100:.{d}f}%"


def bar(v: float, scale: float, color: str, null95: float | None = None, ci: list | None = None) -> str:
    """横棒（幅 = v / scale）。帰無の 95% 点は細い縦線、90% 区間は薄い帯"""
    w = max(0.0, min(1.0, v / scale)) * 100 if scale > 0 else 0
    parts = [f'<span class="bar" style="width:{w:.1f}%;background:{color}"></span>']
    if ci:
        lo, hi = (max(0.0, min(1.0, c / scale)) * 100 for c in ci)
        parts.insert(0, f'<span class="ci" style="left:{lo:.1f}%;width:{max(hi - lo, 0.5):.1f}%"></span>')
    if null95 is not None and null95 > 0:
        parts.append(f'<span class="nul" style="left:{min(1.0, null95 / scale) * 100:.1f}%" title="帰無の 95% 点 {pct(null95, 2)}"></span>')
    return f'<span class="track">{"".join(parts)}</span>'


# ---------------------------------------------------------------- 2. 父と母の影響
def influence_section(r: dict) -> str:
    rel = r.get("reliability", {})
    groups = [("同じ馬の中の適性の差・脚質", r["traits"]["broad"]), ("走法軸（2023〜）", r["traits"]["fine"]), ("出走の配分（参考: 調教師の選択を含む）", r["traits"]["place"])]
    gene = r.get("gene_share", {})
    exp_ratio = (gene.get("父", .5) / gene.get("母父", .25)) ** 2
    h = [f"""<p>対象: 2 勝クラス以上を 1 回でも走った馬 {r['n_horses']:,} 頭（{r['n_runs']:,} 走）。系統の判明率:
{"・".join(f"{p} {pct(r['coverage'][p], 0)}" for p in POS)}（表に無い古い種牡馬は、父系をさかのぼるか、息子の系統の多数決で補った）。
数値は「その系統で、馬ごとの形質のばらつきを何 % 説明できるか」（4 つの系統を同時に入れたリッジの交差検証 R² を Shapley 値で分け、
y を並べ替えた帰無の平均を引いた正味の値）。棒の薄い帯は父単位のブートストラップの 90% 区間（上への偏りを打ち消す基本区間、0 で切る。点が区間の外に出ることがあり、目安）。
「帰無内」は、点の値が帰無の 95% 点を超えないもの（帰無は 4 つの系統をまとめて並べ替えたもので、系統どうしの相関があると、効果の無い位置にも分け前が出うる。
多重比較の補正もしていないので、目安として読む）。
信頼性は折半法（同じ馬の走を半分に分けて作った形質どうしの一致。系統で説明できる割合の上限の目安）。
遺伝子の期待の割合は 父 1/2・母父 1/4 なので、分散で見た 父:母父 の期待の比は {exp_ratio:.0f}:1。</p>
<div class=legend>{"".join(f'<span><i style="background:{POS_COLOR[p]}"></i>{p}</span>' for p in POS)}</div>"""]
    for title, traits in groups:
        traits = [t for t in traits if t in r["A_line"]]
        scale = max(0.01, max(max(max(net(r["A_line"][t]).values()), max((c[1] for c in r["A_line"][t].get("ci90", {}).values()), default=0)) for t in traits) * 1.05)
        rows = []
        for t in traits:
            a = r["A_line"][t]
            nt = net(a)
            cells = "".join(f'<td class=b>{bar(nt[p], scale, POS_COLOR[p], None, a["ci90"].get(p))}<small>{pct(nt[p], 2)}{"" if a["shapley"][p] > a["null95"].get(p, 0) else "（帰無内）"}</small></td>' for p in POS)
            dam = sum(max(0.0, nt[p]) for p in POS[1:])
            fa, ma = max(0.0, nt["父"]), max(0.0, nt["母父"])
            ratio = "—" if ma <= 0 else f"{fa / ma:.1f}"
            st = net(r.get("A_stallion", {}).get(t, {}))
            cp = r.get("A_line_complete", {}).get(t, {})
            lg = r.get("C_lgb", {}).get(t, {})
            rows.append(f"<tr><th>{E(t)}<small>n={a['n']:,}・信頼性 {rel.get(t, {}).get('reliability', '—')}</small></th>{cells}"
                        f"<td>{pct(a['full_r2'], 2)}</td><td>{pct(fa, 2)} / {pct(dam, 2)}</td><td>{ratio}</td>"
                        f"<td>{pct(st.get('父(種牡馬)'), 2)} / {pct(st.get('母父(種牡馬)'), 2)}</td>"
                        f"<td>{pct(cp.get('full_r2'), 2)}（{' / '.join(pct(net(cp).get(p), 2) for p in POS)}）</td>"
                        f"<td>{' / '.join(pct(lg.get(p), 1) for p in POS)}</td></tr>")
        h.append(f"""<h3>{E(title)}</h3><div class=scroll><table class=inf><thead><tr><th>形質</th>{"".join(f"<th>{p}</th>" for p in POS)}
<th>合計 R²<br>（点）</th><th>父 / 母側の合計</th><th>父÷母父（期待 {exp_ratio:.0f}）</th><th>種牡馬そのもの<br>父 / 母父</th><th>4 つとも分かる馬<br>（{r.get('n_complete', 0):,} 頭）</th><th>LightGBM の重要度<br>父 / 母父 / 母母父 / 母母母父</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>""")
    rb = r.get("robust_table_only")
    if rb:
        rows = "".join(f"<tr><th>{E(t)}<small>n={v['n']:,}</small></th><td>{pct(v['net']['父'], 2)}</td><td>{pct(v['net']['母父'], 2)}</td>"
                       f"<td>{'父' if v['net']['父'] > v['net']['母父'] else '母父'}寄り</td></tr>" for t, v in rb["traits"].items())
        h.append(f"""<h3>補完の影響の確認: 母父の系統が表から直接分かる馬だけ（{rb['n']:,} 頭、父と母父の 2 つで計算）</h3>
<p>母母父・母母母父の多くは息子からの推定なので、母父が系統の表（またはその父系）から分かる馬だけで、父と母父の正味の寄与を比べた。</p>
<div class=scroll><table class=small><thead><tr><th>形質</th><th>父</th><th>母父</th><th>向き</th></tr></thead><tbody>{rows}</tbody></table></div>""")
    for key, title in (("B_all", "教師なし: 適性・脚質・走法軸をまとめた行列（走法軸のある馬）"), ("B_broad", "教師なし: 適性・脚質だけの行列（全年）")):
        b = r.get(key)
        if not b:
            continue
        fl = []
        for f, load in b["fa_load"].items():
            top = sorted(load.items(), key=lambda x: -abs(x[1]))[:4]
            inf = net(b["factor_influence"].get(f, {}))
            fl.append(f"<tr><th>{f}</th><td>{'、'.join(f'{E(k)} {v:+.2f}' for k, v in top)}</td>{''.join(f'<td>{pct(inf.get(p), 2)}</td>' for p in POS)}</tr>")
        prof_cols = b["cols"]
        cl = "".join(f"<tr><th>群 {k}（{b['cluster_size'].get(str(k), b['cluster_size'].get(k, ''))} 頭）</th>" + "".join(
            f'<td style="background:{heat(v.get(c, 0))}">{v.get(c, 0):+.2f}</td>' for c in prof_cols) + "</tr>" for k, v in b["cluster_profile"].items())
        ami = "".join(f"<tr><th>{E(m)}</th>" + "".join(f"<td>{d[p][0]:.4f}{'' if d[p][1] is None else f'（帰無 {d[p][1]:.4f}）'}</td>" for p in POS) + "</tr>" for m, d in b["ami"].items())
        h.append(f"""<h3>{E(title)}</h3><p>{b['n']:,} 頭。主成分の寄与率 {', '.join(pct(x) for x in b['pca_var'])}。
因子分析（varimax）の因子ごとに、負荷の大きい形質と、その因子の得点を各系統が説明する割合。KMeans の群の数はシルエットで {b['kmeans_k']}、GMM は BIC で {b['gmm_k']}。</p>
<div class=scroll><table class=small><thead><tr><th>因子</th><th>負荷の大きい形質</th>{"".join(f"<th>{p}</th>" for p in POS)}</tr></thead><tbody>{"".join(fl)}</tbody></table></div>
<div class=scroll><table class="small heat"><thead><tr><th>KMeans の群</th>{"".join(f"<th>{E(c)}</th>" for c in prof_cols)}</tr></thead><tbody>{cl}</tbody></table></div>
<p>群と系統の調整済み相互情報量（AMI。0 = 偶然と同じ、1 = 系統で群が決まる。括弧は系統を並べ替えた帰無の 95% 点）</p>
<div class=scroll><table class=small><thead><tr><th>群の作り方</th>{"".join(f"<th>{p}</th>" for p in POS)}</tr></thead><tbody>{ami}</tbody></table></div>""")
    d = r.get("D", {})
    if d:
        rows = []
        for t, v in d.items():
            cells = []
            for p in ("父", "母父"):
                items = list(v[p].items())
                lo = "、".join(f"{E(k)} {s[0]:+.2f}" for k, s in items[:3])
                hi = "、".join(f"{E(k)} {s[0]:+.2f}" for k, s in items[::-1][:3])
                cells.append(f"<td>{hi}</td><td>{lo}</td>")
            rows.append(f"<tr><th>{E(t)}</th>{''.join(cells)}</tr>")
        h.append(f"""<h3>向き: 形質ごとに高い・低い小系統</h3><p>系統の平均（形質を標準化、頭数で 0 に縮めた値）。高い = 形質の名前の向きに強い。</p>
<div class=scroll><table class=small><thead><tr><th>形質</th><th>父として 高い</th><th>父として 低い</th><th>母父として 高い</th><th>母父として 低い</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>""")
    return "\n".join(h)


def net(a: dict) -> dict:
    """正味の寄与（Shapley の点 − y を並べ替えた帰無の平均）。古い結果で net が無ければ点の値"""
    return a.get("net") or a.get("shapley", {})


def heat(v: float) -> str:
    """-1〜+1 を 青〜赤 の薄い色に"""
    v = max(-1.0, min(1.0, float(v)))
    return f"rgba(220,60,40,{v * 0.55:.2f})" if v >= 0 else f"rgba(40,100,220,{-v * 0.55:.2f})"


# ---------------------------------------------------------------- 3. 小系統の距離配置
def scatter_svg(e: dict, colors: dict[str, str]) -> str:
    xy = e["xy"]
    if not xy:
        return "<p>系統が少なく、配置を作れません。</p>"
    xs, ys = [v[0] for v in xy.values()], [v[1] for v in xy.values()]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    W, H, P = 640, 460, 70
    sx = lambda x: P + (x - x0) / ((x1 - x0) or 1) * (W - 2 * P)   # noqa: E731
    sy = lambda y: P + (y - y0) / ((y1 - y0) or 1) * (H - 2 * P)   # noqa: E731
    nmax = max(e["n"].values())
    pts = []
    for a, (x, y) in xy.items():
        r = 5 + 12 * (e["n"][a] / nmax) ** 0.5
        pts.append(f'<g class=pt><title>{E(a)}（{e["n"][a]} 頭）／近い: {E("、".join(n for n, _ in e["near"][a]))}</title>'
                   f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{r:.1f}" fill="{colors.get(a, "#999")}" fill-opacity=".75" stroke="#333" stroke-width=".6"/>'
                   f'<text x="{sx(x):.1f}" y="{sy(y) - r - 3:.1f}" text-anchor="middle">{E(a)}</text></g>')
    return f'<svg class=map viewBox="0 0 {W} {H}" role="img" aria-label="小系統の MDS 配置">{"".join(pts)}</svg>'


def dendro_svg(e: dict) -> str:
    L, lines = e.get("linkage"), e.get("lines", [])
    if not L:
        return ""
    n = len(lines)
    order = e["dendro_order"]
    pos = {lines.index(a): i for i, a in enumerate(order)}   # 葉の番号 → 並びの位置
    hmax = max(row[2] for row in L) or 1
    W, rowh, left, right = 640, 22, 150, 20
    H = n * rowh + 20
    sx = lambda d: left + d / hmax * (W - left - right)   # noqa: E731
    node_y, node_x = {i: 10 + pos[i] * rowh + rowh / 2 for i in range(n)}, {i: 0.0 for i in range(n)}
    seg = []
    for k, (a, b, d, _) in enumerate(L):
        a, b = int(a), int(b)
        x = sx(d)
        ya, yb = node_y[a], node_y[b]
        seg.append(f'<path d="M{sx(node_x[a]):.1f},{ya:.1f}H{x:.1f}V{yb:.1f}H{sx(node_x[b]):.1f}"/>')
        node_y[n + k], node_x[n + k] = (ya + yb) / 2, d
    labels = "".join(f'<text x="{left - 6}" y="{10 + i * rowh + rowh / 2 + 4:.1f}" text-anchor="end">{E(a)}</text>' for i, a in enumerate(order))
    return f'<svg class=dendro viewBox="0 0 {W} {H}" role="img" aria-label="小系統の樹形図">{labels}<g class=lk>{"".join(seg)}</g></svg>'


def linemap_section(r: dict, colors: dict[str, str]) -> str:
    E_ = r.get("E", {})
    blocks, btns = [], []
    for i, p in enumerate(("父", "母父")):
        e = E_.get(p)
        if not e or not e.get("lines"):
            continue
        cols = e["cols"]
        prof = "".join(f"<tr><th>{E(a)}<small>{e['n'][a]}</small></th>" + "".join(
            f'<td style="background:{heat(e["profile"][a][c] * 2)}">{e["profile"][a][c]:+.2f}</td>' for c in cols) + "</tr>" for a in e["dendro_order"])
        near = "".join(f"<tr><th>{E(a)}</th><td>{'、'.join(f'{E(b)}（{d:.2f}）' for b, d in e['near'][a])}</td>"
                       f"<td>{'、'.join(f'{E(b)} {v:.0%}' for b, v in sorted(e['nn_stability'].get(a, {}).items(), key=lambda x: -x[1])[:2])}</td></tr>" for a in e["dendro_order"])
        btns.append(f'<button type=button class="sub{" on" if i == 0 else ""}" data-map="{i}">{p}の系統として</button>')
        blocks.append(f"""<div class=mapblk data-map="{i}"{"" if i == 0 else " hidden"}>
<p>{p}が各小系統の馬（2 勝クラス以上、走法軸が 3 走以上ある馬。{sum(e['n'].values()):,} 頭・{len(e['lines'])} 系統、30 頭以上の系統）の走法軸の平均（標準化、頭数で 0 に縮めた値）で、系統どうしの距離を測った。
近い点ほど、産駒の走り方（走法軸 8 つ: TS・TS持続力・ギア・Hペース耐性・先行力・出脚・Cr・まくり）が似ている。点の色は大系統、大きさは頭数。</p>
<div class=two><figure>{scatter_svg(e, colors)}<figcaption>MDS（距離を 2 次元に写したもの。軸そのものに意味はない）</figcaption></figure>
<figure>{dendro_svg(e)}<figcaption>Ward の樹形図（横の位置 = まとまるときの距離）</figcaption></figure></div>
<h4>プロファイル（樹形図の順）</h4><div class=scroll><table class="small heat"><thead><tr><th>小系統</th>{"".join(f"<th>{E(c)}</th>" for c in cols)}</tr></thead><tbody>{prof}</tbody></table></div>
<h4>近い系統</h4><p>距離が近い順に 3 つ。右は、父（種牡馬）単位で引き直したとき（ブートストラップ）に最も近い系統になった割合（安定度）。</p>
<div class=scroll><table class=small><thead><tr><th>小系統</th><th>近い系統（距離）</th><th>最も近い系統の安定度</th></tr></thead><tbody>{near}</tbody></table></div></div>""")
    if not blocks:
        return "<p>未計算です。</p>"
    return f'<div class=subtabs>{"".join(btns)}</div>{"".join(blocks)}'


# ---------------------------------------------------------------- 全体
CSS = """
:root{--bg:#fff;--fg:#1f2328;--mut:#59636e;--line:#d0d7de;--card:#f6f8fa;--acc:#2f6fdf}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0d1117;--fg:#e6edf3;--mut:#9198a1;--line:#30363d;--card:#161b22;--acc:#6ea8fe}}
:root[data-theme=dark]{--bg:#0d1117;--fg:#e6edf3;--mut:#9198a1;--line:#30363d;--card:#161b22;--acc:#6ea8fe}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.6 system-ui,-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif}
main{max-width:1200px;margin:0 auto;padding:12px 16px 40px}h1{font-size:20px;margin:4px 0}h2{font-size:17px}h3{font-size:15px;margin-top:22px}h4{font-size:14px}
.meta,small{color:var(--mut)}p,li,h2,h3,h4{overflow-wrap:anywhere}.note{color:var(--mut);border-left:3px solid var(--line);padding-left:8px}
nav.tabs{display:flex;flex-wrap:wrap;gap:4px;border-bottom:1px solid var(--line);margin:8px 0 12px;position:sticky;top:0;background:var(--bg);z-index:2;padding:6px 0}
nav.tabs button,.subtabs button{border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;padding:4px 10px;font:inherit;cursor:pointer}
nav.tabs button.on,.subtabs button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
section[hidden],.mapblk[hidden]{display:none}.subtabs{display:flex;gap:6px;margin:6px 0}
.scroll{overflow-x:auto;max-width:100%}table{border-collapse:collapse;margin:6px 0}th,td{border:1px solid var(--line);padding:3px 6px;vertical-align:top;text-align:left}
th small,td small{display:block;font-weight:400}table.small{font-size:12px}table.heat td{text-align:right;white-space:nowrap}
table.inf td.b{min-width:120px}.track{position:relative;display:block;height:12px;background:var(--card);border-radius:3px;overflow:hidden}
.track .bar{position:absolute;left:0;top:2px;height:8px;border-radius:2px}.track .ci{position:absolute;top:0;height:12px;background:rgba(128,128,128,.25)}
.track .nul{position:absolute;top:0;width:2px;height:12px;background:var(--fg);opacity:.6}
.legend{display:flex;gap:12px;flex-wrap:wrap}.legend i{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media (max-width:800px){.two{grid-template-columns:1fr}}
figure{margin:0;overflow-x:auto}figcaption{font-size:12px;color:var(--mut)}svg{width:100%;min-width:560px;height:auto}svg text{font-size:11px;fill:var(--fg)}
svg.dendro .lk path{fill:none;stroke:var(--mut);stroke-width:1.2}
iframe.emb{width:100%;height:85vh;border:1px solid var(--line);border-radius:6px;background:#fff}
textarea.src{display:none}code{background:var(--card);padding:0 3px;border-radius:3px}
"""

JS = """
(function(){
  var root=document.getElementById('slh'); if(!root) return;
  var tabs=root.querySelectorAll('nav.tabs button'), secs=root.querySelectorAll('section.slh-tab');
  function open(i){
    tabs.forEach(function(b){b.classList.toggle('on', b.dataset.tab===i);});
    secs.forEach(function(s){s.hidden = s.dataset.tab!==i;});
    var s=root.querySelector('section.slh-tab[data-tab="'+i+'"]'), src=s&&s.querySelector('textarea.src');
    if(src && !s.querySelector('iframe')){ var f=document.createElement('iframe'); f.className='emb'; f.srcdoc=src.value; s.appendChild(f); }
    try{ history.replaceState(null,'','#'+i); }catch(e){}
  }
  tabs.forEach(function(b){ b.addEventListener('click', function(){ open(b.dataset.tab); }); });
  root.querySelectorAll('.subtabs button').forEach(function(b){ b.addEventListener('click', function(){
    var p=b.closest('section'); p.querySelectorAll('.subtabs button').forEach(function(x){x.classList.toggle('on', x===b);});
    p.querySelectorAll('.mapblk').forEach(function(m){ m.hidden = m.dataset.map!==b.dataset.map; });
  }); });
  var h=(location.hash||'').slice(1); open(root.querySelector('nav.tabs button[data-tab="'+h+'"]') ? h : 'summary');
})();
"""

TABS = [("summary", "まとめ"), ("influence", "父と母の影響"), ("linemap", "小系統の距離配置"), ("matrix", "成績マトリクス"),
        ("lines", "系統一覧"), ("euro", "欧州瞬発力の検証"), ("marking", "マーキング")]


def embed(path: Path) -> str:
    """元の HTML をそのまま <textarea hidden> に入れる（RCDATA なので &・< をエスケープすれば中身は崩れない）。タブを開くと iframe の srcdoc になる"""
    if not path.exists():
        return f"<p>元のページ（{E(str(path.relative_to(REPO)))}）がまだありません。</p>"
    return f'<p class=meta>元のページ: <code>{E(str(path.relative_to(REPO)))}</code></p><textarea class=src hidden aria-hidden=true>{E(path.read_text(encoding="utf-8"))}</textarea>'


def build(results: dict | None, docs: dict[str, str], embeds: dict[str, Path], colors: dict[str, str]) -> str:
    missing = "<p>未計算です（<code>python -m src.research.pedigree.sire_dam_influence</code> で作ってから、このページを作り直してください）。</p>"
    body = {
        "summary": md_to_html(docs.get("summary", "")) or "<p>まとめの文書がありません。</p>",
        "influence": influence_section(results) if results else missing,
        "linemap": linemap_section(results, colors) if results else missing,
        "matrix": embed(embeds["matrix"]), "lines": embed(embeds["lines"]),
        "euro": md_to_html(docs.get("euro", "")) or "<p>文書がありません。</p>", "marking": embed(embeds["marking"]),
    }
    built = results.get("built", "—") if results else "—"
    nav = "".join(f'<button type=button data-tab="{k}">{E(t)}</button>' for k, t in TABS)
    secs = "".join(f'<section class=slh-tab data-tab="{k}" hidden><h2>{E(t)}</h2>{body[k]}</section>' for k, t in TABS)
    return f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>種牡馬系統の分析</title><style>{CSS}</style></head><body><main id=slh>
<h1>種牡馬系統の分析</h1><p class=meta>作成 {datetime.now(JST).strftime("%Y-%m-%d %H:%M")}（分析の計算 {E(built)}）。父・母側の系統の影響、小系統の近さ、系統 × 舞台の成績、系統一覧、欧州瞬発力の検証を 1 枚にまとめたページ。</p>
<nav class=tabs>{nav}</nav>{secs}</main><script>{JS}</script></body></html>
"""


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    results = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else None
    docs = {k: p.read_text(encoding="utf-8") for k, p in DOCS.items() if p.exists()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(results, docs, EMBEDS, large_colors()), encoding="utf-8")
    print(json.dumps({"results": bool(results), "out": str(out), "bytes": out.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
