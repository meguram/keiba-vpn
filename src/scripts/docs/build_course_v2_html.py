"""全レース版の第 2 版（時間で分けて採る版）のサイクルと評価を 1 枚の HTML にする。第 1 版の値も並べる。

使い方: python -m src.scripts.docs.build_course_v2_html [出力 HTML]
入力: <COURSE_OUT>/v2/course_v2_cycles.json・eval_course_v2.json（あれば）、第 1 版の <COURSE_OUT>/course_cycles.json・eval_course.json
出力: data/meta/course/course_v2.html
- 各期間（発見 2016〜2021・確認 2022〜2024・評価 2025〜2026）で、◎ の勝率・連対率・複勝率・単複の回収率、消しの的中・消しを付けたレースの割合、
  買ったレースの数と割合を並べる（ユーザーの依頼、2026-11-17）。
- オッズで絞る版は、楽観（確定オッズで判定）と本番の再現（発走前のオッズを 20 通り作って判定）を並べる。採否は本番の再現。
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
OUT = REPO / "data/meta/course/course_v2.html"
COLS = [("bet_races", "買ったレース", "n"), ("coverage", "買った割合", "p"), ("win", "◎ 勝率", "p"), ("rentai", "◎ 連対率", "p"),
        ("place", "◎ 複勝率", "p"), ("roi_t", "単回収", "p"), ("roi_f", "複回収", "p"), ("kill_precision", "消しの的中", "p"),
        ("kill_coverage", "消しを付けたレース", "p")]


def fmt(v, kind: str) -> str:
    if v is None:
        return "—"
    return f"{int(round(v)):,}" if kind == "n" else f"{100 * v:.1f}%"


def cells(a: dict | None) -> str:
    a = a or {}
    return "".join(f"<td class='n'>{fmt(a.get(k), t)}</td>" for k, _, t in COLS)


def head(extra: list[str]) -> str:
    return "<tr>" + "".join(f"<th>{escape(h)}</th>" for h in extra + [h for _, h, _ in COLS]) + "</tr>"


def build(out: Path = OUT) -> Path:
    L2 = json.loads((BASE / "v2/course_v2_cycles.json").read_text(encoding="utf-8"))
    E2p = BASE / "v2/eval_course_v2.json"
    E2 = json.loads(E2p.read_text(encoding="utf-8")) if E2p.exists() else None
    L1p, E1p = BASE / "course_cycles.json", BASE / "eval_course.json"
    L1 = json.loads(L1p.read_text(encoding="utf-8")) if L1p.exists() else None
    E1 = json.loads(E1p.read_text(encoding="utf-8")) if E1p.exists() else None
    rows = []
    for c in L2["cycles"]:
        ad = "採用" if c.get("adopted") else ("—" if c.get("qualified", True) else f"—（{L2.get('min_bet_conf', 500)} 未満）")
        name = escape(c["params"].get("name", ""))
        opt = c["conf"]["optimistic"] != c["conf"]["reproduction"]          # オッズで絞る版だけ、楽観の行を足す
        per = [("確認 2022〜2024（本番の再現）", c["conf"]["reproduction"])] + ([("確認 2022〜2024（楽観）", c["conf"]["optimistic"])] if opt else []) \
            + [("発見 2016〜2021", c["disc"]["reproduction"])]
        for i, (lab, src) in enumerate(per):
            lead = (f"<td class='n' rowspan='{len(per)}'>{c['cycle']}</td><td class='wrap' rowspan='{len(per)}'>{escape(c['hyp'])}"
                    f"<div class='muted'>{name}</div></td><td rowspan='{len(per)}'>{ad}</td>") if i == 0 else ""
            rows.append(f"<tr class='{'good' if c.get('adopted') and i == 0 else ''}'>{lead}<td>{lab}</td>{cells(src)}</tr>")
    t2 = f"<table><thead>{head(['サイクル', '仮説', '採否', '期間'])}</thead><tbody>{''.join(rows)}</tbody></table>"
    # 第 1 版と第 2 版の最後の版（評価があれば評価も）
    cmp = []
    if L1:
        b1 = next((c for c in L1["cycles"] if c["cycle"] == L1.get("best_cycle")), None)
        if b1:
            cmp.append(f"<tr><td>第 1 版</td><td>当てはめ 2016〜2024（探した走で数えた値。消しの的中は作りの上で 100%）</td>{cells(b1['all'])}</tr>")
            if b1.get("conf_2022_2024"):
                cmp.append(f"<tr><td>第 1 版</td><td>当てはめのうち 2022〜2024</td>{cells(b1['conf_2022_2024'])}</tr>")
        if E1:
            cmp.append(f"<tr><td>第 1 版</td><td>評価 2025〜2026（1 回目）</td>{cells(E1['all'])}</tr>")
    b2 = next((c for c in L2["cycles"] if c["cycle"] == L2.get("best_cycle")), None)
    if b2:
        cmp.append(f"<tr class='good'><td>第 2 版</td><td>発見 2016〜2021</td>{cells(b2['disc']['reproduction'])}</tr>")
        cmp.append(f"<tr class='good'><td>第 2 版</td><td>確認 2022〜2024（本番の再現）</td>{cells(b2['conf']['reproduction'])}</tr>")
        if b2["conf"]["optimistic"] != b2["conf"]["reproduction"]:
            cmp.append(f"<tr class='good'><td>第 2 版</td><td>確認 2022〜2024（楽観）</td>{cells(b2['conf']['optimistic'])}</tr>")
    if E2:
        a = E2["all"]
        cmp.append(f"<tr class='good'><td>第 2 版</td><td>評価 2025〜2026（2 回目・本番の再現）</td>{cells(a['reproduction'])}</tr>")
        if a["optimistic"] != a["reproduction"]:
            cmp.append(f"<tr class='good'><td>第 2 版</td><td>評価 2025〜2026（楽観）</td>{cells(a['optimistic'])}</tr>")
        if a.get("real_t60"):
            cmp.append(f"<tr class='good'><td>第 2 版</td><td>2026 年の発走前オッズのあるレース（本物の T-60）</td>{cells(a['real_t60'])}</tr>")
    tc = f"<table><thead>{head(['版', '期間'])}</thead><tbody>{''.join(cmp)}</tbody></table>"
    tu = ""
    if E2 and E2.get("per_unit"):
        us = [u for u in E2["per_unit"] if (u["reproduction"].get("bet_races") or 0) >= 20]
        us.sort(key=lambda u: -(u["reproduction"].get("roi_t") or 0))
        ur = "".join(f"<tr><td>{escape(u['unit'])}</td>{cells(u['reproduction'])}</tr>" for u in us)
        tu = (f"<h2>評価 2025〜2026 の単位ごとの値（本番の再現。◎ を 20 レース以上買った {len(us)} 単位、単回収の高い順）</h2>"
              f"<p class='muted'>回数が少ないので、上の方の単位も偶然の幅が大きい。単位を選んで使うなら、その選び方も次の版の確認の期間で確かめる。</p>"
              f"<div class='scroll'><table><thead>{head(['単位'])}</thead><tbody>{ur}</tbody></table></div>")
    kp = next((c["kill_pick"] for c in L2["cycles"] if c.get("kill_pick")), None)
    tk = ""
    if kp:
        kr = "".join(f"<tr class='{'good' if all(g[k] == v for k, v in kp['selected'].items()) else ''}'><td>{escape(str({k: g[k] for k in kp['selected']}))}</td>"
                     f"<td class='n'>{fmt(g.get('kill_precision'), 'p')}</td><td class='n'>{fmt(g.get('kill_coverage'), 'p')}</td>"
                     f"<td class='n'>{fmt(g.get('killed'), 'n')}</td></tr>" for g in kp["grid"])
        tk = (f"<h2>消しの選び直し（サイクル {next(c['cycle'] for c in L2['cycles'] if c.get('kill_pick'))}）</h2><p class='muted'>{escape(kp['rule'])}。"
              f"{escape(kp.get('note', ''))}</p><div class='scroll'><table><thead><tr><th>消しの決め方</th><th>確認の消しの的中</th>"
              f"<th>消しを付けたレース</th><th>消した頭数</th></tr></thead><tbody>{kr}</tbody></table></div>")
    done = f"終了 {escape(L2['finished_at'])}" if L2.get("finished_at") else "途中"
    html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>全レース版 第 2 版</title><style>{CSS}</style></head><body><main>
<h1>全レース版 第 2 版（時間で分けて採る版）</h1>
<div class="panel wrap"><p><b>注意: {escape(L2.get('note_conf', ''))}。</b></p>
<ul class="muted"><li>発見 2016〜2021 年で条件を探し、確認 2022〜2024 年でも買い（複勝率 70% 以上・回収率 100% 超）・消し（4 着以下の割合）を保った条件だけを採る。
採否は確認の期間の ◎ の単回収（本番の再現。買ったレース {L2.get('min_bet_conf', 500)} 以上）。発見の期間の値は並べるだけ。{done}（{len(L2['cycles'])} 本）。</li>
<li>オッズは条件の切り口に使わず、最後の買う／買わないの絞り込みだけに使う。本番の再現は、2026 年の発走前（T-60）と確定のオッズの組（着順・払戻は見ない）から作った
ずれを確定オッズに逆に当てて、20 通りの平均で判定した値。楽観は確定オッズで判定した値。</li>
<li>2025・2026 年の評価は、凍結の後に 1 回だけ（第 1 版で一度見たので 2 回目）。</li></ul></div>
<h2>第 1 版と第 2 版</h2><div class="scroll">{tc}</div>
{tu}<h2>第 2 版のサイクル</h2><div class="scroll">{t2}</div>{tk}
<p class="muted">作り: src/research/course/v2.py → src/scripts/docs/build_course_v2_html.py。計画書: docs/ideas/course_class_handicapper.md §8。
第 1 版: <a href="course_cycles.html">サイクル</a>・<a href="course_playbook.html">早見表</a></p>
</main></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(out, len(L2["cycles"]), "cycles")
    return out


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
