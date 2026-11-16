"""全レース版の単位ごとの早見表（コース × クラスの群 × 良馬場）を 1 枚の HTML にする。

使い方: python -m src.scripts.docs.build_course_playbook_html [course_best.json] [出力 HTML] [eval_course.json（あれば）]
- 単位ごとのカード: コース・クラスの群・層・回数、◎ と消しの条件（日本語の文。走数の多い順に上位）、2016〜2024 年の当てはめの値。
- 評価（2025・2026 年）が済んでいれば、同じ単位の評価の値を並べ、差を過学習の大きさとして読めるようにする。
- 会場 → コースのプルダウン（graded_course_filter）とクラスの群の絞り込み。外部の JS・CSS なし。
"""

from __future__ import annotations

import json
import os
import sys
from html import escape
from pathlib import Path

from src.research.graded import level_text as LT
from src.scripts.docs import graded_course_filter as CF
from src.scripts.docs.build_graded_handicapper_html import CSS

REPO = Path(__file__).resolve().parents[3]
BASE = Path(os.environ.get("COURSE_OUT") or "/home/jovyan/work/course_out")
OUT = REPO / "data/meta/course/course_playbook.html"
TOP = 5

EXTRA_CSS = """.cards{display:grid;grid-template-columns:1fr;gap:10px}@media (min-width:900px){.cards{grid-template-columns:1fr 1fr}}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:10px;min-width:0}.card h3{margin:0 0 4px;font-size:15px}
.card ul{margin:4px 0;padding-left:18px;font-size:12.5px}.sm{font-size:12px}.hide{display:none}.ok{color:var(--good);font-weight:700}"""

JS = r"""
const cards=[...document.querySelectorAll('.card')];const $=id=>document.getElementById(id);
function apply(){const v=$('f-v').value,c=$('f-c').value,g=$('f-g').value,a=$('f-a').value;let n=0;
 cards.forEach(e=>{const d=e.dataset;const ok=(!v||d.venue===v)&&(!c||d.course===c)&&(!g||d.group===g)&&(!a||d.ach==='1');e.classList.toggle('hide',!ok);if(ok)n++});
 $('f-n').textContent=n+' / '+cards.length+' 単位'}
setupCourse($('f-v'),$('f-c'),COPTS,apply,true,'単位');['f-g','f-a'].forEach(id=>$(id).addEventListener('input',apply));apply();
"""


def pct(v) -> str:
    return "—" if v is None else f"{100 * v:.1f}%"


def _cond(c: dict) -> str:
    s = LT.describe_combo(c["types"], c["levels"])
    tag = "（極小: 1〜2 走）" if c.get("tiny") else ""
    stat = f"［{c.get('k', '—')}/{c.get('n', '—')} 走で 3 着内・単回収 {pct(c.get('roi_t'))}］" if "k" in c else f"［{c.get('n', '—')} 走で 3 着内 0］"
    return f"<li>{escape(s)}{escape(tag)} <span class='muted'>{escape(stat)}</span></li>"


def card(u: dict, cells: dict | None, ev: dict | None) -> str:
    course, group = u["unit"].split("|", 1)
    buy = sorted((cells or {}).get("buy", []), key=lambda c: (-c.get("n", 0), len(c["types"])))[:TOP]
    kill = sorted((cells or {}).get("kill", []), key=lambda c: (-c.get("n", 0), len(c["types"])))[:3]
    evl = (f"<div class='sm'><b>評価 2025・2026 年</b>: ◎ 勝率 {pct(ev.get('win'))}・単回収 {pct(ev.get('roi_t'))}（{ev.get('bet_races', 0)}/{ev.get('races', 0)} レースで ◎）"
           f"・消しのレース {pct(ev.get('kill_coverage'))}・消しの的中 {pct(ev.get('kill_precision'))}</div>") if ev else ""
    return (f'<article class="card" data-venue="{escape(CF.split_course(course)[0])}" data-course="{escape(course)}" data-group="{escape(group)}" data-ach="{int(bool(u.get("achieved")))}">'
            f"<h3>{escape(course)} <span class='muted'>{escape(group)}・{escape(u.get('layer', ''))}の層・{u.get('races', 0)} 回</span></h3>"
            f"<div class='sm'>当てはめ 2016〜2024 年: ◎ 勝率 {pct(u.get('win'))}・単回収 {pct(u.get('roi_t'))}（{u.get('bet_races', 0)}/{u.get('races', 0)} レース）"
            f"{' <span class=ok>達成</span>' if u.get('achieved') else ''}</div>{evl}"
            f"<div class='sm'><b>◎ の条件</b>（買いの条件 {u.get('n_buy', 0)} 個のうち走数の多い {len(buy)} 個。当てはまる数が最も多い馬を ◎）</div>"
            f"<ul>{''.join(_cond(c) for c in buy) or '<li class=muted>なし</li>'}</ul>"
            f"<div class='sm'><b>消しの条件</b>（{u.get('n_kill', 0)} 個のうち上位。3 頭以上当たったレースだけ消す）</div>"
            f"<ul>{''.join(_cond(c) for c in kill) or '<li class=muted>なし</li>'}</ul></article>")


def build(src: Path = BASE / "course_best.json", out: Path = OUT, ev_src: Path | None = BASE / "eval_course.json") -> Path:
    B = json.loads(src.read_text(encoding="utf-8"))
    E = json.loads(ev_src.read_text(encoding="utf-8")) if ev_src and ev_src.exists() else None
    # 凍結が済んでいれば、凍結した条件の表（sha を照らす）を出す（凍結の後にサイクルを流し直しても、評価と並ぶのは凍結した条件）
    from src.research.course import run as CR
    if CR.PREREG.exists():
        P = json.loads(CR.PREREG.read_text(encoding="utf-8"))
        cp = src.parent / P["cells_file"]
        if not cp.exists() or CR._sha_file(cp) != P["sha256_cells"]:
            raise SystemExit("凍結した条件の表が無いか、sha が違う")
        if P["params"] != B["params"]:
            raise SystemExit("course_best.json の版が凍結の版と違う")
        B = B | {"cells": json.loads(cp.read_text(encoding="utf-8"))}
    evu = {x["unit"]: x for x in (E or {}).get("per_unit", [])}
    units = sorted(B["per_unit"], key=lambda u: CF.sort_key(u["unit"].split("|")[0]))
    groups = sorted({u["unit"].split("|", 1)[1] for u in units})
    copts = CF.options_json({u["unit"]: u["unit"].split("|")[0] for u in units})
    a = B["all"]
    evh = (f"<p><b>評価（2025・2026 年、1 回だけ）</b>: ◎ 勝率 {pct(E['all'].get('win'))}・単回収 {pct(E['all'].get('roi_t'))}。"
           f"当てはめ（2016〜2024 年）の {pct(a.get('win'))}・{pct(a.get('roi_t'))} との差が、過学習の大きさ。</p>") if E else \
        "<p class='muted'>評価（2025・2026 年）はまだ。条件を凍結・commit してから 1 回だけ行う。</p>"
    html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>全レース版の早見表</title><style>{CSS}{EXTRA_CSS}</style></head><body><main>
<h1>全レース版の早見表（コース × クラスの群 × 良馬場）</h1>
<div class="panel wrap"><p>版: {escape(B['params'].get('name', ''))}。単位 {B['units']}・達成 {B['achieved_units']}。
当てはめ（2016〜2024 年、条件を探した同じ走で数えた値。過学習してよい）: ◎ 勝率 {pct(a.get('win'))}・単回収 {pct(a.get('roi_t'))}（50 倍で打ち切り {pct(a.get('roi_t_capped'))}）・◎ を出したレース {pct(a.get('coverage'))}。</p>
{evh}<ul class="muted"><li>買いの条件: 切り口 × 水準の AND で、複勝率 70% 以上・単勝か複勝の回収率 100% 超。◎ = 当てはまる条件の数が最も多い馬。</li>
<li>消しの条件: 3 着以内が 0。当てはめでは定義上すべて 4 着以下になるので、本当の値は評価で見る。オッズ・人気は条件に使わない。</li></ul></div>
<div class="ctl"><select id="f-v" aria-label="会場"></select><select id="f-c" aria-label="コース"></select>
<select id="f-g"><option value="">クラスの群: すべて</option>{''.join(f'<option>{escape(g)}</option>' for g in groups)}</select>
<select id="f-a"><option value="">達成: すべて</option><option value="1">達成した単位だけ</option></select><span id="f-n" class="muted"></span></div>
<div class="cards">{''.join(card(u, B.get('cells', {}).get(u['unit']), evu.get(u['unit'])) for u in units)}</div>
<p class="muted">作り: src/research/course/run.py → src/scripts/docs/build_course_playbook_html.py。サイクルの記録: <a href="course_cycles.html">全レース版の分析サイクル</a></p>
</main><script>const COPTS={copts};{CF.COURSE_JS}{JS}</script></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(out, f"{out.stat().st_size / 1e6:.2f} MB", len(units), "units")
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    build(Path(a[0]) if a else BASE / "course_best.json", Path(a[1]) if len(a) > 1 else OUT, Path(a[2]) if len(a) > 2 else BASE / "eval_course.json")
