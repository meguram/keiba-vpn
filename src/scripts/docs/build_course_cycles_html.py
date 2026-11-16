"""全レース版（コース × クラスの群 × 良馬場）の 20 サイクルの記録を 1 枚の HTML にする（途中の版でも作れる）。

使い方: python -m src.scripts.docs.build_course_cycles_html [入力 JSON] [出力 HTML]
入力: <COURSE_OUT>/course_cycles.json（python -m src.research.course.run cycles の出力）
出力: data/meta/course/course_cycles.html
値はすべて 2016〜2024 年の当てはめ（過学習してよい。ユーザーの指示）。2025・2026 年は見ていない。
"""

from __future__ import annotations

import json
import os
import sys
from html import escape
from pathlib import Path

from src.scripts.docs.build_graded_handicapper_html import CSS

REPO = Path(__file__).resolve().parents[3]
SRC = Path(os.environ.get("COURSE_OUT") or "/home/jovyan/work/course_out") / "course_cycles.json"
OUT = REPO / "data/meta/course/course_cycles.html"


def pct(v) -> str:
    return "—" if v is None else f"{100 * v:.1f}%"


def build(src: Path = SRC, out: Path = OUT) -> Path:
    L = json.loads(src.read_text(encoding="utf-8"))
    C = L.get("cycles", [])
    g = L.get("goal", {})
    rows = []
    for c in C:
        a, cf = c.get("all") or {}, c.get("conf_2022_2024") or {}
        cls = "good" if c.get("adopted") else ""
        rows.append(
            f"<tr><td class='n'>{c['cycle']}</td><td class='wrap'>{escape(c['hyp'])}<div class='muted'>{escape((c.get('params') or {}).get('name', ''))}</div></td>"
            f"<td class='{cls}'>{'採用' if c.get('adopted') else '—'}</td><td class='n'>{c['achieved_units']}/{c['units']}</td>"
            f"<td class='n'>{pct(a.get('win'))}</td><td class='n'>{pct(a.get('roi_t'))}</td><td class='n'>{pct(a.get('roi_t_capped'))}</td>"
            f"<td class='n'>{pct(a.get('coverage'))}</td><td class='n'>{pct(a.get('kill_coverage'))}</td>"
            f"<td class='n'>{pct(cf.get('win'))}／{pct(cf.get('roi_t'))}</td><td class='n'>{(c.get('closeness') or 0):.2f}</td><td class='n'>{(c.get('seconds') or 0):.0f}</td></tr>")
    head = ["サイクル", "仮説（それまでに採った版から 1 つ変える）", "採否", "達成した単位", "◎ 勝率", "◎ 単回収", "単回収（50 倍で打ち切り）",
            "買ったレース", "消しを付けたレース", "参考: 2022〜2024 年の ◎ 勝率／単回収", "ゴールまでの近さ", "秒"]
    from src.research.course.run import CYCLES
    done = "途中" if not L.get("finished_at") else f"終了 {escape(L['finished_at'])}"
    best = f"採った最良の版: サイクル {L['best_cycle']}" if L.get("best_cycle") else ""
    html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>全レース版の分析サイクル</title><style>{CSS}</style></head><body><main>
<h1>全レース版の分析サイクル（コース × クラスの群 × 良馬場）</h1>
<div class="panel wrap"><p>{escape(L.get('note', ''))}。{done}（{len(C)} / {len(CYCLES)} サイクル）。{best}対象: 良馬場の平地 {L.get('races', 0):,} レース・{L.get('rows', 0):,} 走（2016〜2024 年）。</p>
<ul class="muted">
<li>値はすべて 2016〜2024 年の当てはめ（条件を探した同じ走で数えた値）。過学習してよい（ユーザーの指示）。2025・2026 年は、条件を凍結するまで見ない。</li>
<li>ゴール: ◎ の勝率 {pct(g.get('win'))}・単勝の回収率 {pct(g.get('roi_t'))}。単位の達成には、買ったレースの割合 30% 以上・5 回以上も要る。</li>
<li>買い = 切り口 × 水準の AND で、複勝率 70% 以上・単複どちらかの回収率 100% 超。◎ = 当てはまった買いの条件の数が最大の馬（いなければ買わない）。</li>
<li>消し = 3 着以内が 0 の条件。当てはめでは定義上すべて 4 着以下になるので、的中ではなく、消しを付けたレースの割合を出す。</li>
<li>参考の 2022〜2024 年の値も、条件を探した期間の中（当てはめ）。採否には使わない。</li></ul></div>
<div class="scroll"><table><thead><tr>{''.join(f'<th>{escape(h)}</th>' for h in head)}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="muted">作り: src/research/course/run.py → src/scripts/docs/build_course_cycles_html.py。計画書: docs/ideas/course_class_handicapper.md</p>
</main></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(out, len(C), "cycles")
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    build(Path(a[0]) if a else SRC, Path(a[1]) if len(a) > 1 else OUT)
