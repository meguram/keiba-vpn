"""重賞ごとの「◎ の選び方・消しの選び方」の早見表を 1 枚の HTML にする（1 重賞 = 1 枚のカード）。
外部の JS・CSS なし、ライト・ダーク両対応、スマホ幅。絞り込み（格・会場・コース・検索・達成）と並べ替え（格・日付・達成）は素の JS。

使い方: python -m src.scripts.docs.build_graded_race_playbook_html [入力 JSON] [出力 HTML]
入力: <GRADED_OUT>/playbook.json（python -m src.research.graded.run playbook の出力。分析サイクル 20 回の最後の規則を当て直した値）
出力: data/meta/graded/graded_race_playbook.html
"""

from __future__ import annotations

import json
import os
import sys
from html import escape
from pathlib import Path

from src.research.graded import level_text as LT
from src.research.graded import playbook as PB
from src.scripts.docs import graded_course_filter as CF
from src.scripts.docs.build_graded_handicapper_html import CSS

REPO = Path(__file__).resolve().parents[3]
SRC = Path(os.environ.get("GRADED_OUT", "/home/jovyan/work/graded_out_v2")) / "playbook.json"
OUT = REPO / "data/meta/graded/graded_race_playbook.html"

EXTRA_CSS = """
.cards{display:grid;grid-template-columns:1fr;gap:12px}.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:12px;min-width:0}
.card h3{margin:0 0 2px;font-size:16px}.badges{display:flex;flex-wrap:wrap;gap:5px;margin:4px 0 6px}
.bd{font-size:11.5px;padding:1px 8px;border-radius:10px;border:1px solid var(--border);white-space:nowrap}
.bd.b{background:var(--good);color:var(--panel);border-color:var(--good);font-weight:700}
.bd.x{color:var(--muted)}.bd.w{color:var(--bad);border-color:var(--bad)}
.two{display:grid;grid-template-columns:1fr;gap:10px}@media (min-width:860px){.two{grid-template-columns:1fr 1fr}}
.box{border:1px solid var(--border);border-radius:8px;padding:8px;min-width:0}.box.ok{border-color:var(--good);box-shadow:inset 3px 0 0 var(--good)}
.box h4{margin:0 0 4px;font-size:13.5px}.rule{margin:2px 0 6px;padding-left:18px}.rule li{margin:1px 0}.sm{font-size:12px}
.note{font-size:12px;color:var(--bad);margin-top:6px}.sumgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin:8px 0}
.tile{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:8px}.tile b{font-size:20px;display:block}
.hide{display:none}.rc{margin:6px 0 10px}.rc h5{margin:6px 0 3px;font-size:13px}tr.hit td{background:color-mix(in srgb,var(--bad) 16%,transparent)}
tr.hit td.fin{color:var(--bad);font-weight:700}.mk{white-space:nowrap}
"""

JS = r"""
const cards=[...document.querySelectorAll('.card')];const box=document.getElementById('cards');
const $=id=>document.getElementById(id);
function apply(){const g=$('f-grade').value,v=$('f-venue').value,c=$('f-course').value,q=$('f-q').value.trim(),a=$('f-ach').value;let n=0;
 cards.forEach(e=>{const d=e.dataset;const ok=(!g||d.grade===g)&&(!v||d.venue===v)&&(!c||d.course===c)&&(!q||d.text.includes(q))&&(!a||d[a]==='1');
  e.classList.toggle('hide',!ok);if(ok)n++});$('f-n').textContent=n+' / '+cards.length+' 重賞';
 const s=$('f-sort').value;const key={grade:e=>e.dataset.grade+e.dataset.md,date:e=>e.dataset.md+e.dataset.grade,
  ach:e=>String(9-Number(e.dataset.score))+e.dataset.grade+e.dataset.md}[s];
 cards.slice().sort((x,y)=>key(x)<key(y)?-1:key(x)>key(y)?1:0).forEach(e=>box.appendChild(e))}
setupCourse($('f-venue'),$('f-course'),COPTS,apply,true);
['f-grade','f-q','f-ach','f-sort'].forEach(id=>$(id).addEventListener('input',apply));apply();
"""

GRADE_ORDER = {"G1": "1", "G2": "2", "G3": "3"}
CRIT = {"roi_t_over": PB.ROI_OK, "kill_top3_at_most": PB.KILL_OK}   # build で playbook.json の criteria に置き換える


def pct(v) -> str:
    return "—" if v is None else f"{100 * v:.1f}%"


def rec(a) -> str:
    return "—" if not a else f"[{'-'.join(str(int(x)) for x in a)}]"


def rule_html(r: dict, k: int | None = None) -> str:
    """決め方の文（重みの大きい順）。消しは「次の点の低い順に k 頭を消す」。"""
    head = ""
    if k is not None:
        head = f"<div class='sm'>次の点で順位を付け、<b>消す側から {k} 頭</b>を消す（頭数が少ないレースは 3 頭を残す）。</div>"
    else:
        head = "<div class='sm'>次の点を足し合わせ、<b>一番高い馬を ◎</b> にする。</div>"
    if r.get("filter"):
        head += f"<div class='sm'><b>{escape(r['filter'])}</b></div>"
    small = '<span class="muted">・ほぼ効かない</span>'
    head += "<div class='muted'>効きの大きい順（効き = |重み| × レースの中での点のばらつき。レースで全馬同じ点は効かない）</div>"
    lis = "".join(f"<li>{escape(t['text'])}（重み {t['w']:+g}）{small if t['small'] else ''}</li>" for t in r["terms"])
    return f"{head}<ol class='rule sm'>{lis}</ol><div class='muted'>{escape(r['tie'])}</div>"


def h_table(e: dict, d: dict, m: str = "B") -> str:
    rows = []
    for nm, s, note in (("2025・2026 年", e, "決めた後に当てはめた値"), ("2018〜2024 年", d, "選んだ期間（当てはめ）")):
        if not s or not s.get("races"):
            rows.append(f"<tr><td>{nm}</td><td colspan='5' class='muted'>レースなし</td></tr>")
            continue
        cls = "good" if (s.get("roi_t") or 0) > CRIT["roi_t_over"] else ""
        if "same_as_base" in s:
            note += f"・前日版の ◎ と同じ馬 {s['same_as_base']}/{s['races']} 回"
        rows.append(f"<tr><td>{nm}<div class='muted'>{note}</div></td><td class='n'>{s['races']}</td><td class='n'>{rec(s['rec'])}</td><td class='n'>{pct(s['win'])}</td>"
                    f"<td class='n {cls}'>{pct(s['roi_t'])}</td><td class='n'>{pct(s['roi_f'])}</td></tr>")
    return ("<div class='scroll'><table><thead><tr><th>期間</th><th>レース</th><th>◎ の着</th><th>勝率</th><th>単回収</th><th>複回収</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def k_table(e: dict, d: dict, m: str = "B") -> str:
    rows = []
    for nm, s, note in (("2025・2026 年", e, "決めた後に当てはめた値"), ("2018〜2024 年", d, "選んだ期間（当てはめ）")):
        if not s or not s.get("races"):
            rows.append(f"<tr><td>{nm}</td><td colspan='4' class='muted'>レースなし</td></tr>")
            continue
        cls = "good" if s.get("top3_rate") is not None and s["top3_rate"] <= CRIT["kill_top3_at_most"] else "bad"
        rows.append(f"<tr><td>{nm}<div class='muted'>{note}</div></td><td class='n'>{s['races']}</td><td class='n'>{rec(s['rec'])}</td>"
                    f"<td class='n {cls}'>{pct(s['top3_rate'])}</td><td class='n'>{s['per_race']:.1f}</td></tr>")
    return ("<div class='scroll'><table><thead><tr><th>期間</th><th>レース</th><th>消した馬の着</th><th>3 着以内の率</th><th>頭/レース</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def examples_html(ex: list[dict]) -> str:
    if not ex:
        return ""
    li = []
    for x in ex:
        lv = "・".join(f"{escape(LT.describe(v['type'], v['level'])[0])}（{escape(v['type'])} {escape(v['name'])}、{escape(v['layer'])}の層）" for v in x["levels"])
        fin = int(x["fin"]) if x["fin"] == x["fin"] else "—"
        lvh = f'<br><span class="muted">{lv}</span>' if lv else ""
        pay = f"単勝の払戻 {round(x['tan']):,} 円" if x.get("tan") else ("払戻なし" if x.get("pay_ok", True) else "払戻のデータなし")
        nm = f" {escape(x['name'])}" if x.get("name") else ""
        li.append(f"<li>{escape(x['date'])}: {x['hn']} 番{nm} → {fin} 着（{pay}）{lvh}</li>")
    return f"<details><summary class='sm'>2025・2026 年の ◎（重みの大きい型で当たった水準）</summary><ul class='sm'>{''.join(li)}</ul></details>"


def _odds(h: dict) -> str:
    return "—" if h.get("odds") is None else f"{h['odds']:.1f}"


def _pop(h: dict) -> str:
    return "—" if h.get("pop") is None else f"{h['pop']}"


def _fin(h: dict) -> str:
    return f"{int(h['fin'])}" if h["fin"] == h["fin"] else "—"


def _yen(v) -> str:
    return "—" if v is None else f"{v:,}"


def _name(h: dict) -> str:
    return escape(h.get("name") or "（馬名なし）")


def race_html(x: dict) -> str:
    """1 回分: 見出し（日付・会場・頭数・勝ち馬）、◎ の表（前日版・B）、消しの表（B で消した馬。3 着以内は目立たせる）。"""
    win = "・".join(f"{_name(w)}（{w['hn']} 番・単勝 {_odds(w)} 倍・{_pop(w)} 番人気）" for w in x["winners"]) or "—"
    pay = "" if x["pay_ok"] else "<span class='bd w'>払戻のデータなし（この回は回収率の分母に入れていない）</span>"
    rows = []
    for lab, h in (("前日版の点の ◎（参考）", x["base"]), ("B の ◎", x["B"]["honmei"])):
        if h is None:
            rows.append(f"<tr><td class='mk'>{lab}</td><td colspan='7' class='muted'>規則なし</td></tr>")
            continue
        g = " good" if h["fin"] <= 3 else ""   # ◎ は 3 着以内を緑
        rows.append(f"<tr><td class='mk'>{lab}</td><td class='n'>{h['hn']}</td><td>{_name(h)}</td><td class='n'>{_odds(h)}</td><td class='n'>{_pop(h)}</td>"
                    f"<td class='n{g}'>{_fin(h)}</td><td class='n'>{_yen(h['tan'])}</td><td class='n'>{_yen(h['fuku'])}</td></tr>")
    ht = ("<div class='scroll'><table><thead><tr><th>印</th><th>馬番</th><th>馬名</th><th>単勝オッズ</th><th>人気</th><th>着順</th><th>単勝の払戻</th><th>複勝の払戻</th></tr></thead>"
          f"<tbody>{''.join(rows)}</tbody></table></div>")
    kill = sorted(x["B"]["kill"] or [], key=lambda q: q["hn"])
    if kill:
        kr = []
        for h in kill:
            hit = " class='hit'" if h["fin"] <= 3 else ""
            kr.append(f"<tr{hit}><td class='n'>{h['hn']}</td><td>{_name(h)}</td><td class='n'>{_odds(h)}</td><td class='n'>{_pop(h)}</td><td class='n fin'>{_fin(h)}</td></tr>")
        bad = sum(h["fin"] <= 3 for h in kill)
        kt = (f"<div class='sm'>B で消した馬（{len(kill)} 頭のうち 3 着以内 {bad}。3 着以内に来た馬は赤）</div>"
              "<div class='scroll'><table><thead><tr><th>馬番</th><th>馬名</th><th>単勝オッズ</th><th>人気</th><th>着順</th></tr></thead>"
              f"<tbody>{''.join(kr)}</tbody></table></div>")
    elif x["B"]["kill"] is None:
        kt = "<div class='muted sm'>消しの規則なし</div>"
    else:
        kt = "<div class='muted sm'>この回は消した馬なし（頭数が少なく 3 頭を残した）</div>"
    return (f"<div class='rc'><h5>{escape(x['date'])} {escape(x['venue'])}・{x['field']} 頭立て・勝ち馬 {win} {pay}</h5>{ht}{kt}</div>")


def races_html(c: dict) -> str:
    if not c.get("races"):
        return ""
    return ("<details><summary><b>2025・2026 年に実際に選ばれた馬（◎・消しの馬名・確定の単勝オッズ・人気・着順）</b></summary>"
            "<div class='muted sm'>単勝オッズ・人気は確定の値（結果として見せるためだけで、◎ や消しの決め方には使っていない）。払戻は 100 円あたりの円。</div>"
            f"{''.join(race_html(x) for x in c['races'])}</details>")


def side(card: dict, m: str, kind: str) -> str:
    x = card[m][kind]
    lab = "B（2018〜2024 年で決めて当てはめた）"
    if x is None:
        return f"<div class='box'><h4>{lab}</h4><div class='muted'>規則なし</div></div>"
    f = card["flags"]
    ok = f[f"{m}_{kind}"]
    tag = ""
    if x["default"]:
        tag = "<span class='bd x'>既定（2018〜2024 年に回が無く、最初の候補が残っただけ）</span>"
    elif ok:
        tag = "<span class='bd b'>達成</span>"
    if kind == "honmei":
        body = rule_html(x["rule"]) + h_table(x["eval"], x["dev"], m) + examples_html(x.get("examples", []))
    else:
        body = rule_html(x["rule"], x["k"]) + k_table(x["eval"], x["dev"], m)
    return f"<div class='box{' ok' if ok else ''}'><h4>{lab} {tag}</h4>{body}</div>"


def card_html(c: dict, tried: str = "多数の") -> str:
    f = c["flags"]
    md = (c["eval_dates"][-1][5:] if c["eval_dates"] else "99-99")
    badges = []
    if f["B_both"]:
        badges.append("<span class='bd b'>◎・消しの両方を達成</span>")
    else:
        if f["B_honmei"]:
            badges.append("<span class='bd b'>◎ を達成</span>")
        if f["B_kill"]:
            badges.append("<span class='bd b'>消しを達成</span>")
    if f["base_honmei"]:
        badges.append("<span class='bd x'>前日版の点だけでも ◎ の単回収 100% 超</span>")
    if f["B_default"]:
        badges.append("<span class='bd x'>B は既定</span>")
    if c["eval_races"] == 1:
        badges.append("<span class='bd w'>2025・2026 年は 1 回だけ</span>")
    b = c["base"]
    base = (f"<div class='muted sm'>比べる基準（前日版の点が最大の馬を ◎）: 2025・2026 年 {rec(b['eval'].get('rec'))}・単回収 {pct(b['eval'].get('roi_t'))}"
            f"／2018〜2024 年 {rec(b['dev'].get('rec'))}・単回収 {pct(b['dev'].get('roi_t'))}</div>" if b["eval"].get("races") else "")
    score = 4 * f["B_both"] + 2 * f["B_honmei"] + f["B_kill"]
    text = " ".join([c["rkey"], c["grade"], c["venue"], c["course"]])
    attrs = (f'data-grade="{GRADE_ORDER.get(c["grade"], "9")}" data-venue="{escape(c["venue"])}" data-course="{escape(c["course"])}" '
             f'data-md="{escape(md)}" data-text="{escape(text)}" data-score="{score}" '
             + " ".join(f'data-{k.lower().replace("_", "")}="{int(f[k])}"' for k in ("B_honmei", "B_kill", "B_both")))
    return f"""<article class="card" {attrs}>
<h3>{escape(c['rkey'])} <span class="muted">{escape(c['grade'])}・{escape(c['course'])}</span></h3>
<div class="muted sm">2025・2026 年の回: {escape('・'.join(c['eval_dates']))}（{c['eval_races']} 回）・2018〜2024 年の回: {c['dev_races']} 回</div>
<div class="badges">{''.join(badges)}</div>
<h4>◎ の選び方（単勝の回収率 100% 超を達成とする）</h4>{base}
{side(c, 'B', 'honmei')}
<h4 style="margin-top:10px">消しの選び方（消した馬の 3 着以内の率 5% 以下を達成とする）</h4>
{side(c, 'B', 'kill')}
{races_html(c)}
<div class="note">注意: 2025・2026 年の値は {c['eval_races']} 回だけの成績です。1 回の高配当で回収率 100% を超えることがあります。</div>
</article>"""


def build(src: Path = SRC, out: Path = OUT) -> Path:
    P = PB.drop_a(json.loads(src.read_text(encoding="utf-8")))   # A（廃止した版）は、古い値に残っていても出さない
    C, S = P["cards"], P["summary"]
    copts = CF.options_json({c["rkey"]: c["course"] for c in C})
    both = "・".join(escape(x) for x in S["B_both_rkeys"]) or "なし"
    tried = f" {P['tried']:,} " if P.get("tried") else "多数の"
    CRIT.update(P.get("criteria") or {})
    glob = ""
    G = P.get("global")
    if G:
        rows = []
        for m in PB.MODES:
            if m not in G:
                continue
            h, k = G[m]["honmei"], G[m]["kill"]
            lab = "B（2018〜2024 年で選んだ）"
            rows.append(f"<div class='box'><h4>{lab}</h4><div class='sm'><b>◎</b>: {rec(h['eval'].get('rec'))}・勝率 {pct(h['eval'].get('win'))}・単回収 {pct(h['eval'].get('roi_t'))}・複回収 {pct(h['eval'].get('roi_f'))}</div>"
                        f"{rule_html(h['rule'])}<div class='sm'><b>消し</b>: {rec(k['eval'].get('rec'))}・3 着以内の率 {pct(k['eval'].get('top3_rate'))}・{k['eval'].get('per_race', 0):.1f} 頭/レース</div>"
                        f"{rule_html(k['rule'], k['k'])}</div>")
        glob = ("<details><summary><b>参考: 全重賞に同じ決め方を使う版（2025・2026 年の全重賞）</b></summary>"
                f"{''.join(rows)}</details>")
    html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>重賞の選び方の早見表</title><style>{CSS}{EXTRA_CSS}</style></head><body><main>
<h1>重賞ごとの ◎ と消しの選び方（早見表）</h1>
<div class="panel wrap">
<p>分析サイクル 20 回（<a href="graded_cycles.html">重賞の分析サイクル</a>）の最後の決め方を、重賞ごとに 1 枚のカードにまとめた。評価はやり直していない。
2018〜2024 年の値は、同じ決め方を手元の表に当て直して数えた（2025・2026 年の値は、サイクルの記録と一致することを確かめた）。</p>
<div class="sumgrid">
<div class="tile"><span class="muted">◎ の単回収 100% 超</span><b class="good">{S['B_honmei']} / {S['n']}</b><span class="muted">2025・2026 年（2018〜2024 年で決めた B）</span></div>
<div class="tile"><span class="muted">消しの 3 着以内 5% 以下</span><b class="good">{S['B_kill']} / {S['n']}</b></div>
<div class="tile"><span class="muted">比べる基準: 前日版の点だけで ◎ の単回収 100% 超</span><b>{S['base_honmei']} / {S['n']}</b></div>
</div>
<p><b>◎・消しの両方を達成した重賞（{S['B_both']}）</b>: {both}</p>
<p class="muted">B で ◎ を達成した {S['B_honmei']} 重賞のうち {S.get('B_honmei_same_as_base', '—')} 重賞は、2025・2026 年の ◎ がすべて前日版の点の ◎ と同じ馬だった（決め方を変えた効き目ではない）。</p>
{glob}
<p class="note"><b>全カード共通の注意</b>: 2025・2026 年の回は 1 重賞あたり 1〜2 回（1 回だけの重賞が {S['eval_races_1']}）しかない。
単勝の回収率 100% 超は 1 回の高配当で起きる。前日版の点だけでも {S['base_honmei']} 重賞で 100% を超えており、B の達成の数はこれと比べて読む。</p>
<ul class="muted">
<li>◎ の決め方: 型の点を重み付きで足し、一番高い馬を ◎ にする。型の点は、その馬が当たった水準の過去の成績（3 着内率）を縮めた値で、高いほど過去の成績が良い水準。
条件そのもの（「前走 G1 で 5 着以内」など）ではなく、どの型をどちら向きに効かせるかの決め方。どの水準に当たったかは、カードの「2025・2026 年の ◎」に書いた。</li>
<li>層: 型の点は馬ごとに、同じレース名 × 同じコースの過去の回（主）→ 近い条件の重賞（副）→ 全重賞（全体）の順に、件数が足りる層で自動で決まる。決め方ごとに層を選ぶものではない。</li>
<li>消しの決め方: 同じように点を足し、低い側から k 頭（3〜6 頭、3 頭は残す）を消す。オッズ・人気は使わない。</li>
<li>2018〜2024 年の値は、決め方を選んだ期間そのもの（当てはめ）なので良く見える。比べて読めるのは 2025・2026 年の値。ただしそれも、2025・2026 年の結果を何度も見ながら（20 サイクルの毎回と、それより前の 4 回の評価）候補の作り方を決めたので、まっさらな評価ではない。
実際に ◎ の達成（{S['B_honmei']}）は、前日版の点だけの {S['base_honmei']} より少ない。</li>
<li>2025・2026 年の結果を見て重賞ごとに選んだ版（A）は、評価の答えを見て選んだ版なので廃止した（2026-11-16）。</li>
</ul></div>
<div class="ctl">
<select id="f-grade"><option value="">格: すべて</option><option value="1">G1</option><option value="2">G2</option><option value="3">G3</option></select>
<select id="f-venue" aria-label="会場"></select>
<select id="f-course" aria-label="コース"></select>
<input id="f-q" type="search" placeholder="重賞名で検索">
<select id="f-ach"><option value="">達成: すべて</option><option value="bboth">◎・消しの両方を達成</option><option value="bhonmei">◎ を達成</option>
<option value="bkill">消しを達成</option></select>
<select id="f-sort"><option value="ach">並べ替え: 達成の多い順</option><option value="grade">格の順</option><option value="date">日付（月日）の順</option></select>
<span id="f-n" class="muted"></span></div>
<div id="cards" class="cards">{''.join(card_html(c, tried) for c in C)}</div>
<p class="muted">作り: {escape(P.get('made_at', ''))}（サイクルの終了 {escape(P.get('from_cycles_finished_at', ''))}）。src/research/graded/playbook.py → src/scripts/docs/build_graded_race_playbook_html.py。
戻る: <a href="graded_handicapper.html">重賞のルールベース予想家</a>・<a href="graded_cycles.html">重賞の分析サイクル</a>・<a href="graded_rules_table.html">重賞のルール一覧</a></p>
</main><script>const COPTS={copts};{CF.COURSE_JS}{JS}</script></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(out, f"{out.stat().st_size / 1e6:.2f} MB", len(C), "cards")
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    build(Path(a[0]) if a else SRC, Path(a[1]) if len(a) > 1 else OUT)
