"""重賞のルールベース予想家の結果を 1 枚の HTML にまとめる（外部の JS・CSS なし、ライト・ダーク両対応、スマホ幅ではみ出さない）。

使い方: python -m src.scripts.docs.build_graded_handicapper_html [入力 JSON] [出力 HTML]
入力: <GRADED_OUT>/graded_report.json（python -m src.research.graded.run export の出力）
出力: data/meta/graded/graded_handicapper.html
タブ: 検証の結果（2025・2026 年、v6e との比較）／重賞ごと（主・副の採用パターン）／検証のレース（各馬の印と該当パターン）／切り口の採否
計算はしない（JSON を埋め込み、表は開いたタブだけ素の JS で描く）。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = Path(os.environ.get("GRADED_OUT", "/home/jovyan/work/graded_out")) / "graded_report.json"
OUT = REPO / "data/meta/graded/graded_handicapper.html"

CSS = """
:root{--bg:#f7f8fa;--panel:#fff;--text:#1d2330;--muted:#5d6676;--border:#d9dee7;--accent:#2f6fde;--good:#137a3d;--bad:#b3261e;--head:#eef2f8}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f1420;--panel:#161d2b;--text:#e4e9f2;--muted:#9aa6ba;--border:#2a3446;
--accent:#7aa7ff;--good:#5fd38d;--bad:#ff8a80;--head:#1d2638}}
:root[data-theme="dark"]{--bg:#0f1420;--panel:#161d2b;--text:#e4e9f2;--muted:#9aa6ba;--border:#2a3446;--accent:#7aa7ff;--good:#5fd38d;--bad:#ff8a80;--head:#1d2638}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:"Noto Sans JP",system-ui,sans-serif;font-size:14px;line-height:1.6}
main{max-width:1180px;margin:0 auto;padding:16px}h1{font-size:20px;margin:4px 0 4px}h2{font-size:16px;margin:18px 0 8px}
.muted{color:var(--muted);font-size:12.5px}.tabs{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0}
.tabs button{border:1px solid var(--border);background:var(--panel);color:var(--text);padding:6px 12px;border-radius:16px;cursor:pointer;font-size:13px}
.tabs button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.panel{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:12px;margin:10px 0}
.scroll{overflow-x:auto;max-width:100%}table{border-collapse:collapse;font-size:12.5px;min-width:100%}
th,td{border-bottom:1px solid var(--border);padding:4px 7px;text-align:left;white-space:nowrap;vertical-align:top}
th{background:var(--head);position:sticky;top:0;cursor:pointer;user-select:none}td.n{text-align:right;font-variant-numeric:tabular-nums}
.good{color:var(--good);font-weight:600}.bad{color:var(--bad);font-weight:600}.ctl{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0}
.ctl input,.ctl select{background:var(--panel);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:4px 6px;font-size:13px;max-width:100%}
details{margin:2px 0}summary{cursor:pointer}.mk{font-weight:700;color:var(--accent)}.wrap{white-space:normal}
"""

JS = r"""
const D=JSON.parse(document.getElementById('data').textContent);
const TN={};D.types.forEach(t=>TN[t.id]=t.name);
const $=s=>document.querySelector(s);const pct=v=>v==null?'—':(100*v).toFixed(1)+'%';
const ci=o=>o?`${pct(o.est)} <span class="muted">[${pct(o.lo)}, ${pct(o.hi)}]</span>`:'—';
const num=o=>o?`${o.est.toFixed(2)} <span class="muted">[${o.lo.toFixed(2)}, ${o.hi.toFixed(2)}]</span>`:'—';
const rec=a=>a?`[${a.join('-')}]`:'—';
function sortable(tbl){tbl.querySelectorAll('th').forEach((th,i)=>th.onclick=()=>{const tb=tbl.tBodies[0];const rows=[...tb.rows];
 const asc=th.dataset.s!=='a';th.dataset.s=asc?'a':'d';const v=r=>{const t=r.cells[i].dataset.v??r.cells[i].textContent;const x=parseFloat(t);return isNaN(x)?t:x};
 rows.sort((a,b)=>{const x=v(a),y=v(b);return (x>y?1:x<y?-1:0)*(asc?1:-1)});rows.forEach(r=>tb.appendChild(r))})}
function table(head,rows){return `<div class="scroll"><table><thead><tr>${head.map(h=>`<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table></div>`}
const kcls=k=>k==='好成績'?'good':k==='凡走'?'bad':'';
// 1) 検証の結果
function renderEval(){const E=D.eval.versions;const BK=Object.keys(D.eval.baselines||{});let h='<p class="muted">v6e16 = 2016 年からの表で学習した v6e（2025 年だけ）。v6e本番 = 2020 年からの表の本番 v6e（2025 年は test の予測、2026 年は推論の経路で流した予測で重賞 95 レース）。v6e16 の 2026 年の予測は、2026 年の学習側の表が無く作れなかった。</p>';const P=['2025','2026','2025-2026'];
 const head=['版','期間','レース','◎勝率','◎複勝率','◎○▲の3着内','◎単勝','◎複勝','馬連流し','ワイド流し'];
 const row=(n,p,s)=>`<tr><td>${n}</td><td>${p}</td><td class="n">${s.races}</td><td class="n">${ci(s.win)}</td><td class="n">${ci(s.place)}</td><td class="n">${num(s.top3_in3)}</td><td class="n">${ci(s.roi_tansho)}</td><td class="n">${ci(s.roi_fukusho)}</td><td class="n">${ci(s.roi_umaren)}</td><td class="n">${ci(s.roi_wide)}</td></tr>`;
 const rows=[];for(const [v,e] of Object.entries(E)){for(const p of P){if(!e[p])continue;rows.push(row(v,p,e[p].all));if(v==='前日版')for(const bk of BK){if(e[p]['base_'+bk]){rows.push(row(`<span class="muted">前日版（${bk} と同じレース）</span>`,p,e[p]['self_same_races_'+bk]));rows.push(row(`<b>${bk}（同じレース）</b>`,p,e[p]['base_'+bk]))}}}}
 h+='<h2>全体（重賞・平地）</h2>'+table(head,rows);
 const dr=[];for(const bk of BK)for(const p of P){const e=E['前日版'][p];if(!e||!e['diff_vs_'+bk])continue;const x=e['diff_vs_'+bk];
  dr.push(`<tr><td>${bk}</td><td>${p}</td><td class="n">${x.races}</td>${['win','place'].map(k=>`<td class="n">${ci(x[k])}</td>`).join('')}<td class="n">${num(x.top3_in3)}</td>${['roi_tansho','roi_fukusho','roi_umaren','roi_wide'].map(k=>`<td class="n">${ci(x[k])}</td>`).join('')}</tr>`)}
 h+='<h2>前日版 − v6e（同じレースでの差、レース単位のブートストラップ 95%）</h2>'+table(['相手','期間','レース','◎勝率','◎複勝率','◎○▲の3着内','◎単勝','◎複勝','馬連流し','ワイド流し'],dr);
 const gr=[];for(const p of P){const e=E['前日版'][p];if(!e)continue;for(const g of ['G1','G2','G3'])if(e[g])gr.push(row(g,p,e[g]))}
 h+='<h2>前日版のグレード別</h2>'+table(head,gr);
 const pr=D.prereg;h+=`<h2>凍結した規則（${pr.frozen_at}）</h2><div class="panel wrap">縮める強さ m（全体・副・主）= ${pr.shrink.m_g}・${pr.shrink.m_s}・${pr.shrink.m_m}、主の重み λ = ${pr.score.lam}、点の足し方 = ${pr.score.weight==='z'?'発見期間の z で重み付け':'等しい重み'}。前日版で使う型 ${pr.sets.preday.used.length}（採用 ${pr.sets.preday.adopted.length} を束ねたもの）、当日版 ${pr.sets.dayof.used.length}。<br>前日版で使う型: ${pr.sets.preday.used.map(t=>t+' '+TN[t]).join('、')}</div>`;
 $('#t-eval').innerHTML=h;document.querySelectorAll('#t-eval table').forEach(sortable)}
// 2) 重賞ごと
function renderRaces(){const R=D.races;const grades=[...new Set(R.map(r=>r.grade))].sort();
 $('#t-race').innerHTML=`<div class="ctl"><input id="rq" placeholder="重賞名・コースで絞り込み"><select id="rg"><option value="">全グレード</option>${grades.map(g=>`<option>${g}</option>`).join('')}</select><select id="rs"></select><select id="rk"><option value="">全パターン</option><option>好成績</option><option>凡走</option><option>参考</option></select></div><div id="rb"></div>`;
 const fill=()=>{const q=$('#rq').value.trim(),g=$('#rg').value;const L=R.filter(r=>(!g||r.grade===g)&&(!q||(r.rkey+r.course).includes(q)));
  $('#rs').innerHTML=L.map((r,i)=>`<option value="${R.indexOf(r)}">${r.rkey}（${r.grade}・${r.course}）</option>`).join('');show()};
 const show=()=>{const r=R[+$('#rs').value];if(!r){$('#rb').innerHTML='<p class="muted">該当なし</p>';return}const k=$('#rk').value;
  const ex=r.excluded.length?r.excluded.map(e=>`${e.year} 年（${e.course}）`).join('、'):'なし';
  const rows=r.patterns.filter(p=>!k||p.k===k).sort((a,b)=>b.x-a.x).map(p=>`<tr><td>${p.t} ${TN[p.t]}</td><td class="wrap">${p.lv}</td><td class="${kcls(p.k)}">${p.k}</td><td class="n" data-v="${p.m[0]+p.m[1]+p.m[2]+p.m[3]}">${rec(p.m)}</td><td class="n">${pct(p.mr[0])}</td><td class="n">${pct(p.mr[1])}</td><td class="n">${pct(p.mr[2])}</td><td class="n">${pct(p.ps)}</td><td class="n" data-v="${p.med[0]}">${p.med[0]}/${p.med[1]}</td><td class="n">${rec(p.s)}</td><td class="n">${pct(p.sr[2])}</td><td class="n">${p.x.toFixed(3)}</td><td>${p.ly}</td></tr>`);
  $('#rb').innerHTML=`<div class="panel">${r.rkey}（${r.grade}・${r.course}、最新の回 ${r.latest}）。主の集計: 2016〜2024 年の ${r.n_main_ed} 回・${r.n_main_runs} 走。副の集計（同じコースの OP 以上の別レース）: ${r.n_sub_runs} 走。<br>主から外した回（別のコース）: ${ex}</div>`+
   table(['切り口','水準','種別','主 [1-2-3-着外]','勝率','連対率','複勝率','縮小後の複勝率','3着内の回/回','副 [1-2-3-着外]','副の複勝率','点 x','使った層'],rows)+'<p class="muted">点 x = log（縮小した 実際の3着内 ÷ 頭数から見た期待）。好成績 = 主の集計で上振れし、3着内が出た回が 3 回以上。凡走 = 主で下振れし、該当馬がいた回が 4 回以上で 3着内の回が 1 回以下。</p>';
  document.querySelectorAll('#rb table').forEach(sortable)};
 $('#rq').oninput=fill;$('#rg').onchange=fill;$('#rs').onchange=show;$('#rk').onchange=show;fill()}
// 3) 検証のレース
function renderEvalRaces(){const R=D.eval_races;$('#t-er').innerHTML=`<div class="ctl"><input id="eq" placeholder="レース名で絞り込み"><select id="ey"><option value="">2025・2026</option><option>2025</option><option>2026</option></select><select id="eg"><option value="">全グレード</option><option>G1</option><option>G2</option><option>G3</option></select><select id="es"></select></div><div id="eb"></div>`;
 const fill=()=>{const q=$('#eq').value.trim(),y=$('#ey').value,g=$('#eg').value;const L=R.filter(r=>(!y||r.date.startsWith(y))&&(!g||r.grade===g)&&(!q||(r.name+r.rkey).includes(q)));
  $('#es').innerHTML=L.map(r=>`<option value="${R.indexOf(r)}">${r.date} ${r.rkey}（${r.grade}）</option>`).join('');show()};
 const show=()=>{const r=R[+$('#es').value];if(!r){$('#eb').innerHTML='<p class="muted">該当なし</p>';return}
  const rows=r.horses.map(h=>`<tr><td class="mk">${h.mk}</td><td class="n">${h.hn}</td><td class="n">${h.sc.toFixed(2)}</td><td class="n">${h.n_good}</td><td class="n">${h.n_bad}</td><td class="n">${h.fin??'—'}</td><td class="n">${h.v6??'—'}</td><td>${h.jk??''}</td><td class="wrap"><details><summary>${h.p.length} パターン</summary>${h.p.map(p=>`<div class="${kcls(p.k)}">${TN[p.t]}: ${p.lv} — ${p.ly} 主${rec(p.m)}（${p.med[0]}/${p.med[1]} 回）副${rec(p.s)} 全体${rec(p.g)} 点 ${p.x.toFixed(3)}</div>`).join('')}</details></td></tr>`);
  $('#eb').innerHTML=`<div class="panel">${r.date} ${r.name}（${r.grade}・${r.course}）。◎の払戻: 単勝 ${r.pay.tansho} 円・複勝 ${r.pay.fukusho} 円${r.has_pay?'':'（払戻のデータなし。評価から除外）'}</div>`+
   table(['印','馬番','点','好成績','凡走','着順','v6e本番の順位','騎手','該当パターン（|点| の大きい順）'],rows);document.querySelectorAll('#eb table').forEach(sortable)};
 $('#eq').oninput=fill;$('#ey').onchange=fill;$('#eg').onchange=fill;$('#es').onchange=show;fill()}
// 4) 切り口の採否
function renderTypes(){const rows=D.types.map(t=>`<tr><td>${t.id}</td><td class="wrap">${t.name}</td><td>${t.family}</td><td>${t.status??'比較だけ'}</td><td class="n">${t.z_disc==null?'—':t.z_disc.toFixed(2)}</td><td class="n">${t.z_conf==null?'—':t.z_conf.toFixed(2)}</td><td>${t.bh10_disc?'○':''}</td><td>${t.used_preday?'○':''}</td><td>${t.used_dayof?'○':''}</td><td>${t.day_of?'当日':''}</td><td>${t.start_year}〜</td></tr>`);
 const nb=D.not_built.map(x=>`<tr><td>${x.id}</td><td>${x.name}</td><td class="wrap">${x.reason}</td></tr>`);
 $('#t-ty').innerHTML='<h2>型ごとの採否（発見 2018〜2021・確認 2022〜2024）</h2>'+table(['ID','切り口','族','状態','z（発見）','z（確認）','BH 10%','前日版','当日版','値','集計の開始'],rows)+
  '<p class="muted">採用 = 発見で z ≥ 1.645 かつ 確認で z ≥ 1.2816。前日版・当日版の○は、束ねた後に点に使う型（相関 0.7 以上の型は代表 1 つにまとめる）。</p><h2>作れなかった・入れなかった切り口</h2>'+table(['ID','切り口','理由'],nb);
 document.querySelectorAll('#t-ty table').forEach(sortable)}
const R={eval:renderEval,race:renderRaces,er:renderEvalRaces,ty:renderTypes},done={};
function tab(k){document.querySelectorAll('.tabs button').forEach(b=>b.classList.toggle('on',b.dataset.k===k));document.querySelectorAll('section').forEach(s=>s.hidden=s.id!=='t-'+k);if(!done[k]){R[k]();done[k]=1}}
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>tab(b.dataset.k));tab('eval');
"""


def build(src: Path = SRC, out: Path = OUT) -> Path:
    rep = json.loads(src.read_text(encoding="utf-8"))
    data = json.dumps(rep, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    pre = rep["prereg"]
    html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>重賞のルールベース予想家</title><style>{CSS}</style></head><body><main>
<h1>重賞のルールベース予想家（研究）</h1>
<p class="muted">好成績のパターン（前走・条件の変化・脚質・経歴・騎手・厩舎・血統・調教など約 120 型）を、主（同じレース名 × 同じコースの過去の回）・副（同じコースの OP 以上の別レース）・
全重賞の 3 層で as-of に数え、縮めた上振れの和で ◎○▲△△ を付ける。オッズ・人気は条件に使わず、回収率の評価にだけ使う。規則は {pre['frozen_at']} に 2016〜2024 年の中だけで凍結し、
2025・2026 年の重賞で 1 回だけ評価した。計画: docs/ideas/graded_rule_based_handicapper.md・切り口の目録: docs/ideas/graded_handicapper_cuts.md・コード: src/research/graded/</p>
<div class="tabs"><button data-k="eval">検証の結果</button><button data-k="race">重賞ごと</button><button data-k="er">検証のレース</button><button data-k="ty">切り口の採否</button></div>
<section id="t-eval"></section><section id="t-race" hidden></section><section id="t-er" hidden></section><section id="t-ty" hidden></section>
</main><script type="application/json" id="data">{data}</script><script>{JS}</script></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(out, f"{out.stat().st_size / 1e6:.1f} MB")
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    build(Path(a[0]) if a else SRC, Path(a[1]) if len(a) > 1 else OUT)
