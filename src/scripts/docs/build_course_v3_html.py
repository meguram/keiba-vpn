"""全レース版の第 3 版（2025 年を見ながら試行錯誤する版）のサイクルを 1 枚の HTML にする。

使い方: python -m src.scripts.docs.build_course_v3_html [出力 HTML]
入力: <COURSE_OUT>/v3/course_v3_cycles.json（・kept_<n>.json があれば、採った版の条件の一覧）
出力: data/meta/course/course_v3_cycles.html
- サイクルごとに、仮説・訓練 2016〜2024 と 2025 年の値（本番の再現・楽観）・回顧・判断を並べる。
- 値: 買ったレースの数と割合、◎ の勝率・連対率・複勝率、単複の回収率、上位 5 件を除いた回収率、消しの的中・消しを付けたレース。
- 条件の一覧（採った版）: 仮説の型・方向・年ごとの安定性・広さ・偶然の見込みの列つき。
"""

from __future__ import annotations

import json
import os
import sys
from html import escape
from pathlib import Path

from src.scripts.docs.build_graded_handicapper_html import CSS

REPO = Path(__file__).resolve().parents[3]
BASE = Path(os.environ.get("COURSE_OUT") or "/home/jovyan/work/course_out")
OUT = REPO / "data/meta/course/course_v3_cycles.html"
COLS = [("bet_races", "買ったレース", "n"), ("coverage", "買った割合", "p"), ("win", "勝率", "p"), ("rentai", "連対率", "p"), ("place", "複勝率", "p"),
        ("roi_t", "単回収", "p"), ("roi_f", "複回収", "p"), ("roi_t_trim5", "単回収（上位 5 件を除く）", "p"), ("roi_f_trim5", "複回収（上位 5 件を除く）", "p"),
        ("pay_per_win", "勝ったときの平均の単勝払戻", "y"), ("kill_precision", "消しの的中", "p"), ("kill_coverage", "消しを付けたレース", "p")]


def fmt(v, kind: str) -> str:
    if v is None:
        return "—"
    if kind == "y":
        return f"{v:,.0f} 円"
    return f"{int(round(v)):,}" if kind == "n" else f"{100 * v:.1f}%"


def cells(a: dict | None) -> str:
    a = dict(a or {})
    if a.get("win") and a.get("roi_t") is not None:
        a["pay_per_win"] = 100 * a["roi_t"] / a["win"]       # 回収率 = 的中率 × 平均の払戻（的中の分と払戻の分に分ける）
    return "".join(f"<td class='n'>{fmt(a.get(k), t)}</td>" for k, _, t in COLS)


def cond_table(kept: dict, limit: int = 400) -> str:
    rows = []
    for u, cu in sorted(kept.items()):
        for b in sorted(cu["buy"], key=lambda b: -b.get("n_t", 0))[:5]:
            e = b.get("ev") or {}
            rows.append(f"<tr><td>{escape(u)}</td><td class='wrap'>{escape(' × '.join(f'{t}={lv}' for t, lv in zip(b['types'], b['levels'])))}</td>"
                        f"<td>{escape(str(e.get('hyp', '')))}</td><td>{escape(str(e.get('dir', '')))}</td><td>{escape(str(e.get('stab', '')))}</td>"
                        f"<td>{escape(str(e.get('breadth', '')))}</td><td class='n'>{e.get('p_null', 1):.3f}</td><td class='n'>{b.get('k_t')}/{b.get('n_t')}</td>"
                        f"<td class='n'>{fmt(b.get('rt_t'), 'p')}</td></tr>")
            if len(rows) >= limit:
                break
    head = "".join(f"<th>{h}</th>" for h in ["単位", "条件", "仮説の型", "方向", "年ごとの安定性", "広さ", "偶然の見込み", "訓練の 3 着内/走", "訓練の単回収"])
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def build(out: Path = OUT) -> Path:
    L = json.loads((BASE / "v3/course_v3_cycles.json").read_text(encoding="utf-8"))
    rows = []
    for c in L["cycles"]:
        per = [("2025 年（本番の再現）", c["eval2025"]["reproduction"])]
        if c["eval2025"]["optimistic"] != c["eval2025"]["reproduction"]:
            per.append(("2025 年（楽観）", c["eval2025"]["optimistic"]))
        per.append(("訓練 2016〜2024（本番の再現）", c["train"]["reproduction"]))
        for i, (lab, a) in enumerate(per):
            lead = ""
            if i == 0:
                es = c.get("ev_summary") or {}
                lead = (f"<td class='n' rowspan='{len(per)}'>{c['cycle']}</td><td class='wrap' rowspan='{len(per)}'>{escape(c['hyp'])}"
                        f"<div class='muted'>{escape(c['params'].get('name', ''))}・条件 {c.get('n_buy', 0):,} 個"
                        f"（検査を通った条件のある単位 {es.get('units_with_any', '—')}/{es.get('units', '—')}）</div></td>"
                        f"<td class='wrap' rowspan='{len(per)}'>{escape(c.get('review', ''))}"
                        f"{'<div><b>事象</b>: ' + escape(c['event']) + '</div>' if c.get('event') else ''}"
                        f"{'<div><b>根拠</b>: ' + escape(c['basis']) + '</div>' if c.get('basis') else ''}</td>"
                        f"<td rowspan='{len(per)}'>{escape(c.get('decision', ''))}{'・目標に届く' if c.get('goal_met') else ''}</td>")
            rows.append(f"<tr class='{'good' if c.get('decision') == '採用' and i == 0 else ''}'>{lead}<td>{lab}</td>{cells(a)}</tr>")
    head = "".join(f"<th>{escape(h)}</th>" for h in ["サイクル", "仮説", "回顧・事象と根拠", "判断", "期間"] + [h for _, h, _ in COLS])
    ep = BASE / "v3/eval_course_v3.json"
    te = ""
    if ep.exists():
        E = json.loads(ep.read_text(encoding="utf-8"))
        a = E["all"]
        er = [("2025 年（試行錯誤に使った期間・本番の再現）", E["eval2025"]["reproduction"]), ("2026 年（封印・本番の再現）", a["reproduction"]),
              ("2026 年（楽観＝確定オッズ）", a["optimistic"]), ("2026 年（絞り込みなし）", a["no_filter"])]
        if a.get("real_t60"):
            er.append(("2026 年の発走前オッズのあるレース（本物の T-60）", a["real_t60"]))
        te = (f"<h2>凍結した版（サイクル {escape(str(E['params'].get('name', '')))}）の 2026 年の確かめ（1 回だけ・3 回目）</h2>"
              f"<p class='muted'>{escape(E.get('note', ''))}</p><div class='scroll'><table><thead><tr><th>期間</th>"
              + "".join(f"<th>{escape(h)}</th>" for _, h, _ in COLS) + "</tr></thead><tbody>"
              + "".join(f"<tr><td>{escape(lab)}</td>{cells(x)}</tr>" for lab, x in er) + "</tbody></table></div>")
    adopted = L.get("adopted_cycle")
    kp = BASE / "v3" / f"kept_{adopted}.json" if adopted else None
    ct = cond_table(json.loads(kp.read_text(encoding="utf-8"))) if kp and kp.exists() else "<p class='muted'>採った版はまだ無い。</p>"
    html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>全レース版 第 3 版</title><style>{CSS}</style></head><body><main>
<h1>全レース版 第 3 版（2025 年を見ながら試行錯誤する版）</h1>
<div class="panel wrap"><p><b>注意: {escape(L.get('note_looks', ''))}。</b></p>
<ul class="muted"><li>目標: {escape(L.get('goal', ''))}。</li>
<li>訓練 2016〜2024 年の走で条件を探し、根拠の検査（仮説の型・方向・年ごとの安定性・広さ・偶然の見込み）を通った条件だけを使う（計画書 §10）。</li>
<li>2026 年は封印（サイクルの間は読まない）。凍結の後に 1 回だけ見る。</li>
<li>{escape(L.get('shift_note', ''))}。最後の確かめの期間（2026 年）のオッズを使っている。</li></ul></div>
{te}<h2>サイクル</h2><div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<h2>採った版の条件（単位ごとに走数の多い 5 個まで）</h2>{ct}
<p class="muted">作り: src/research/course/v3.py → src/scripts/docs/build_course_v3_html.py。計画書: docs/ideas/course_class_handicapper.md §10。
第 2 版: <a href="course_v2.html">第 1 版・第 2 版</a>・単位ごとの規則と成績: <a href="../graded/rules_overview.html">ルールの一覧（重賞・通常レース）</a></p>
</main></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(out, len(L["cycles"]), "cycles")
    return out


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
