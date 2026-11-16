"""重賞のルールベース予想家の分析サイクル（ゴール: ◎ の勝率 30%・単勝の回収率 120%、サブゴール: 消しの 95%）の推移を 1 枚の HTML にする。
外部の JS・CSS なし、ライト・ダーク両対応、スマホ幅。推移のグラフは素の SVG。

使い方: python -m src.scripts.docs.build_graded_cycles_html [入力 JSON] [出力 HTML]
入力: <GRADED_OUT>/cycles.json（python -m src.research.graded.run cycles の出力）
出力: data/meta/graded/graded_cycles.html
"""

from __future__ import annotations

import json
import re
import os
import sys
from html import escape
from pathlib import Path

from src.scripts.docs.build_graded_handicapper_html import CSS

REPO = Path(__file__).resolve().parents[3]
SRC = Path(os.environ.get("GRADED_OUT", "/home/jovyan/work/graded_out_v2")) / "cycles.json"
OUT = REPO / "data/meta/graded/graded_cycles.html"

# A（2025・2026 年の結果を見て選んだ版）は廃止（2026-11-16、ユーザーの指示）。B（2018〜2024 年で選んで当てはめた版）だけを出す
SERIES = [("B・重賞ごと", "B", "honmei_by_rkey", "#2f6fde"), ("B・全体で 1 つ", "B", "honmei_global", "#e08a2c")]
KSERIES = [("B・重賞ごと", "B", "kill_by_rkey", "#2f6fde"), ("B・全体で 1 つ", "B", "kill_global", "#e08a2c")]


def _h(rep: dict, m: str, kind: str) -> dict:
    """◎ の値（全重賞）: win・roi_t・place・roi_f。"""
    x = rep[m][kind]
    return x["all"] if kind == "honmei_by_rkey" else x["val"]


def _k(rep: dict, m: str, kind: str) -> dict:
    x = rep[m][kind]
    return x["all"] if kind == "kill_by_rkey" else x["val"]


def _dist(v: dict, goal: dict) -> float:
    return min(v["win"] / goal["win"], v["roi_t"] / goal["roi_t"])


def svg_lines(series: list[tuple[str, list[float], str]], goal: float, ylab: str, fmt) -> str:
    """推移の折れ線（x = サイクル、y = 値）。goal の横線つき。"""
    W, H, L, R, T, B = 640, 260, 52, 12, 14, 34
    ys = [v for _, vals, _ in series for v in vals if v is not None] + [goal]
    span = max(ys) - min(ys) or 1.0
    lo, hi = min(ys) - 0.08 * span, max(ys) + 0.08 * span
    n = max(len(vals) for _, vals, _ in series)
    X = lambda i: L + (W - L - R) * (i / max(n - 1, 1))   # noqa: E731
    Y = lambda v: T + (H - T - B) * (1 - (v - lo) / (hi - lo or 1))   # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{escape(ylab)}" style="width:100%;height:auto;max-width:{W}px">']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var(--border)"/><text x="{L - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" font-size="11" fill="var(--muted)">{fmt(v)}</text>')
    out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(goal):.1f}" y2="{Y(goal):.1f}" stroke="var(--good)" stroke-dasharray="5 4"/>'
               f'<text x="{W - R}" y="{Y(goal) - 5:.1f}" text-anchor="end" font-size="11" fill="var(--good)">目標 {fmt(goal)}</text>')
    for i in range(n):
        out.append(f'<text x="{X(i):.1f}" y="{H - 14}" text-anchor="middle" font-size="11" fill="var(--muted)">{i + 1}</text>')
    out.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 1}" text-anchor="middle" font-size="11" fill="var(--muted)">サイクル</text>')
    for name, vals, color in series:
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals) if v is not None)
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        out += [f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3" fill="{color}"><title>{escape(name)} サイクル {i + 1}: {fmt(v)}</title></circle>'
                for i, v in enumerate(vals) if v is not None]
    out.append("</svg>")
    leg = "".join(f'<span style="display:inline-flex;align-items:center;gap:4px;margin-right:12px"><span style="width:14px;height:3px;background:{c};display:inline-block"></span>{escape(nm)}</span>'
                  for nm, _, c in series)
    return "".join(out) + f'<div class="muted">{leg}</div>'


def _note_b_only(note: str) -> str:
    """記録（cycles.json）の説明から、廃止した A の文を外す（記録そのものは履歴として残す）。"""
    parts = [x for x in re.split(r"(?<=。)", note) if x and not x.startswith("A（")]
    return "".join(parts).replace("B（正直な版）は", "B は")


def pct(v) -> str:
    return "—" if v is None else f"{100 * v:.1f}%"


def build(src: Path = SRC, out: Path = OUT) -> Path:
    L = json.loads(src.read_text(encoding="utf-8"))
    C = L["cycles"]
    goal = L["goal"]
    sg = L["sub_goal"]
    last = C[-1]["report"]
    # 推移の表
    head = ["サイクル", "時刻", "仮説（足した決め方）", "試した数（累計）"] + [f"{nm} ◎勝率／単回収" for nm, *_ in SERIES] + \
           ["B 達成した重賞"] + [f"{nm} 消しの的中率（頭/レース）" for nm, *_ in KSERIES] + ["前のサイクルからの変化（B・重賞ごとの近さ）"]
    rows = []
    prev = None
    for c in C:
        r = c["report"]
        cells = [str(c["cycle"]), escape(c["at"][11:16]), f'<span class="wrap">{escape(c["hypothesis"])}</span>', f"{r['tried']:,}"]
        for _, m, kind, _c in SERIES:
            v = _h(r, m, kind)
            ok = v["win"] >= goal["win"] and v["roi_t"] >= goal["roi_t"]
            cells.append(f'<span class="{"good" if ok else ""}">{pct(v["win"])}／{pct(v["roi_t"])}</span>')
        cells.append(f"{r['B']['honmei_by_rkey']['achieved']}/{r['n_rkeys']}")
        for _, m, kind, _c in KSERIES:
            v = _k(r, m, kind)
            ok = v["precision"] >= sg["precision"] and v["killed_per_race"] >= sg["killed_per_race"]
            cells.append(f'<span class="{"good" if ok else ""}">{pct(v["precision"])}（{v["killed_per_race"]:.1f}）</span>')
        d = _dist(_h(r, "B", "honmei_by_rkey"), goal)
        cells.append("—" if prev is None else f"{d - prev:+.3f}")
        prev = d
        rows.append("<tr>" + "".join(f'<td class="n">{x}</td>' if i not in (2,) else f"<td>{x}</td>" for i, x in enumerate(cells)) + "</tr>")
    tbl = f'<div class="scroll"><table><thead><tr>{"".join(f"<th>{escape(h)}</th>" for h in head)}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    g1 = svg_lines([(nm, [_dist(_h(c["report"], m, kind), goal) for c in C], col) for nm, m, kind, col in SERIES], 1.0,
                   "ゴールまでの近さ", lambda v: f"{v:.2f}")
    g2 = svg_lines([(nm, [_k(c["report"], m, kind)["precision"] for c in C], col) for nm, m, kind, col in KSERIES], sg["precision"],
                   "消しの的中率", lambda v: f"{100 * v:.0f}%")
    # 最後のサイクルの G1〜G3 別
    def grade_tbl(rep):
        hh = ["版", "全重賞 ◎勝率", "単回収", "複勝率", "複回収", "G1 ◎勝率／単回収", "G2", "G3", "達成した重賞", "消しの的中率", "消した頭数/レース"]
        rr = []
        for m, nm in (("B", "B・重賞ごと"),):
            h, k = rep[m]["honmei_by_rkey"], rep[m]["kill_by_rkey"]
            rr.append(f"<tr><td>{nm}</td><td class='n'>{pct(h['all']['win'])}</td><td class='n'>{pct(h['all']['roi_t'])}</td><td class='n'>{pct(h['all']['place'])}</td>"
                      f"<td class='n'>{pct(h['all']['roi_f'])}</td>" + "".join(f"<td class='n'>{pct(h[g]['win'])}／{pct(h[g]['roi_t'])}（{h[g]['achieved']}/{h[g]['rkeys']}）</td>" for g in ("G1", "G2", "G3"))
                      + f"<td class='n'>{h['achieved']}/{rep['n_rkeys']}</td><td class='n'>{pct(k['all']['precision'])}</td><td class='n'>{k['all']['killed_per_race']:.1f}</td></tr>")
        for m, nm in (("B", "B・全体で 1 つの決め方"),):
            v, kv = rep[m]["honmei_global"]["val"], rep[m]["kill_global"]["val"]
            rr.append(f"<tr><td>{nm}<div class='muted wrap'>◎: {escape(rep[m]['honmei_global']['sel'])}<br>消し: {escape(rep[m]['kill_global']['sel'])}</div></td>"
                      f"<td class='n'>{pct(v['win'])}</td><td class='n'>{pct(v['roi_t'])}</td><td class='n'>{pct(v['place'])}</td><td class='n'>{pct(v['roi_f'])}</td>"
                      f"<td colspan='3' class='muted'>開発（2018〜2024 年）の値: ◎勝率 {pct(v['dev_win'])}・単回収 {pct(v['dev_roi_t'])}</td><td>—</td>"
                      f"<td class='n'>{pct(kv['precision'])}</td><td class='n'>{kv['killed_per_race']:.1f}</td></tr>")
        return f'<div class="scroll"><table><thead><tr>{"".join(f"<th>{escape(h)}</th>" for h in hh)}</tr></thead><tbody>{"".join(rr)}</tbody></table></div>'
    def rkey_tbl(rep):
        hh = ["重賞", "格", "評価のレース", "達成", "◎勝率／単回収", "決め方", "消しの的中率"]
        rr = []
        for b, kb in zip(rep["B"]["honmei_by_rkey"]["per_rkey"], rep["B"]["kill_by_rkey"]["per_rkey"]):
            rr.append(f"<tr><td>{escape(b['rkey'])}</td><td>{escape(b['grade'])}</td><td class='n'>{b['races']}</td>"
                      f"<td class='{'good' if b['achieved'] else 'bad'}'>{'達成' if b['achieved'] else '未達'}</td><td class='n'>{pct(b['win'])}／{pct(b['roi_t'])}</td>"
                      f"<td class='wrap muted'>{'既定（開発のレースが無い）: ' if b.get('default') else ''}{escape(b['rule'] or '')}</td><td class='n'>{pct(kb['precision'])}</td></tr>")
        return f'<div class="scroll"><table><thead><tr>{"".join(f"<th>{escape(h)}</th>" for h in hh)}</tr></thead><tbody>{"".join(rr)}</tbody></table></div>'
    stop = L.get("stopped", {})
    html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>重賞の分析サイクル</title><style>{CSS}</style></head><body><main>
<h1>重賞の分析サイクル（ゴール: ◎ 勝率 {pct(goal['win'])}・単回収 {pct(goal['roi_t'])}、サブゴール: 消しの的中 {pct(sg['precision'])}）</h1>
<div class="panel wrap"><p>{escape(_note_b_only(L['note']))}</p>
<ul class="muted"><li>表: 2018〜2026 年の重賞 {L['races']:,} レース・{L['rows']:,} 頭。特徴 {L['n_features']} 列（as-of の型ごとの点 x・前日版と当日版の点・前日版の上位の印）。オッズ・人気は使わない。</li>
<li>◎ の決め方 = 特徴の重み付きの和の最大の馬（同点は前日版の点 → 馬番）。消しの決め方 = 同じ点の下位 k 頭（k = 3〜6。頭数 − 3 頭まで）。消しの的中率 = 消した馬のうち 4 着以下の割合。</li>
<li>B: 2018〜2024 年で一番良い決め方を選んで、2025・2026 年に当てはめた値。2025・2026 年の結果を見て選んだ版（A）は、評価の答えを見て選んだ版なので廃止した（2026-11-16、ユーザーの指示）。
ただし、サイクルごとの仮説（候補の作り方）は A の値も見ながら決めたので、B の 2025・2026 年の値もまっさらな評価ではない。</li>
<li>ゴールまでの近さ = min(◎勝率 ÷ 30%, 単回収 ÷ 120%)。1 以上で達成。止めた理由: {escape(stop.get('reason', '途中'))}（サイクル {stop.get('cycle', len(C))}）。</li></ul></div>
<h2>ゴールまでの近さの推移（全重賞の ◎）</h2>{g1}
<h2>消しの的中率の推移</h2>{g2}
<h2>サイクルごとの記録</h2>{tbl}
<h2>最後のサイクルの値（2025・2026 年、全重賞と G1〜G3 別）</h2>{grade_tbl(last)}
<h2>重賞ごと（最後のサイクル）</h2><p>重賞ごとの決め方を日本語の文と 2018〜2024 年の成績つきで読むには: <a href="graded_race_playbook.html">重賞の選び方の早見表</a></p>{rkey_tbl(last)}
<p class="muted">戻る: <a href="graded_handicapper.html">重賞のルールベース予想家</a>・<a href="graded_rules_table.html">重賞のルール一覧</a>・重賞ごとのまとめ: <a href="graded_race_playbook.html">重賞の選び方の早見表</a>。作り: src/research/graded/cycles.py → src/scripts/docs/build_graded_cycles_html.py</p>
</main></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(out, f"{out.stat().st_size / 1e6:.2f} MB", len(C), "cycles")
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    build(Path(a[0]) if a else SRC, Path(a[1]) if len(a) > 1 else OUT)
