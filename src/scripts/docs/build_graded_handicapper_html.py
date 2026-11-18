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
const LVT=(t,lv)=>String((D.lvtext&&D.lvtext[t+'|'+lv])||(TN[t]+' '+lv)).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));   // 切り口 × 水準の日本語の文（level_text.describe。エスケープ済み）
const $=s=>document.querySelector(s);const pct=v=>v==null?'—':(100*v).toFixed(1)+'%';
const ci=o=>o?`${pct(o.est)} <span class="muted">[${pct(o.lo)}, ${pct(o.hi)}]</span>`:'—';
const num=o=>o?`${o.est.toFixed(2)} <span class="muted">[${o.lo.toFixed(2)}, ${o.hi.toFixed(2)}]</span>`:'—';
const rec=a=>a?`[${a.join('-')}]`:'—';
function sortable(tbl){tbl.querySelectorAll('th').forEach((th,i)=>th.onclick=()=>{const tb=tbl.tBodies[0];const rows=[...tb.rows];
 const asc=th.dataset.s!=='a';th.dataset.s=asc?'a':'d';const v=r=>{const t=r.cells[i].dataset.v??r.cells[i].textContent;const x=parseFloat(t);return isNaN(x)?t:x};
 rows.sort((a,b)=>{const x=v(a),y=v(b);return (x>y?1:x<y?-1:0)*(asc?1:-1)});rows.forEach(r=>tb.appendChild(r))})}
function table(head,rows){return `<div class="scroll"><table><thead><tr>${head.map(h=>`<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table></div>`}
const kcls=k=>k==='好成績'?'good':k==='凡走'?'bad':'';
const roi3=a=>a&&a[0]!=null?`${a[0].toFixed(2)} <span class="muted">[${a[1].toFixed(2)}, ${a[2].toFixed(2)}]</span>`:'—';
const rcls=k=>k==='回収率で残った'?'good':k?'muted':'';
const isRoi=n=>n.startsWith('回収率版');
// 1) 検証の結果
function renderEval(){const E=D.eval.versions;const BK=Object.keys(D.eval.baselines||{});let h='<p class="muted">v6e16 = 2016 年からの表で学習した v6e（2025 年だけ）。v6e本番 = 2020 年からの表の本番 v6e（2025 年は test の予測、2026 年は推論の経路で流した予測で重賞 95 レース）。v6e16 の 2026 年の予測は、2026 年の学習側の表が無く作れなかった。</p>';const P=['2025','2026','2025-2026'];
 const head=['版','期間','レース','◎勝率','◎複勝率','◎○▲の3着内','◎単勝','◎複勝','馬連流し','ワイド流し'];
 const row=(n,p,s)=>`<tr><td>${n}</td><td>${p}</td><td class="n">${s.races}</td><td class="n">${ci(s.win)}</td><td class="n">${ci(s.place)}</td><td class="n">${num(s.top3_in3)}</td><td class="n">${ci(s.roi_tansho)}</td><td class="n">${ci(s.roi_fukusho)}</td><td class="n">${ci(s.roi_umaren)}</td><td class="n">${ci(s.roi_wide)}</td></tr>`;
 const PRIM=D.roi?D.roi.primary:null;const rows=[];for(const [v,e] of Object.entries(E)){if(isRoi(v)&&v!==PRIM)continue;for(const p of P){if(!e[p])continue;rows.push(row(v===PRIM?`<b>${v}</b>（主に見る版）`:v,p,e[p].all));if(v==='前日版'||v===PRIM)for(const bk of BK){if(e[p]['base_'+bk]){rows.push(row(`<span class="muted">${v===PRIM?'主に見る回収率版':'前日版'}（${bk} と同じレース）</span>`,p,e[p]['self_same_races_'+bk]));rows.push(row(`<b>${bk}（同じレース）</b>`,p,e[p]['base_'+bk]))}}}}
 if(D.eval.evaluation_round===2)h='<div class="panel wrap"><b>2 回目の評価（2025・2026 年の結果を一度見た後）</b>。ユーザーの依頼で、各パターンに回収率を持たせ、回収率まで裏付けのあるパターンだけで印を付ける版を足した。データは中止の馬の扱いを直して作り直した（第 1 回の数字は下の「第 1 回の評価」）。前日版・当日版・単純な基準は、第 1 回の規則（prereg.json）のまま直したデータで流した成績版。回収率で選ぶ版は '+D.roi.n_versions+' 版を並べた。版が多いと、どれかが偶然よく見えるので、一番良かった版の値は割り引いて読む。主に見る版は結果を見る前に決めた。</div>'+h;
 h+='<h2>全体（重賞・平地）</h2>'+table(head,rows);
 const dr=[];for(const vn of ['前日版',PRIM].filter(Boolean))for(const bk of BK)for(const p of P){const e=(E[vn]||{})[p];if(!e||!e['diff_vs_'+bk])continue;const x=e['diff_vs_'+bk];
  dr.push(`<tr><td>${vn===PRIM?'主に見る回収率版':'前日版'} − ${bk}</td><td>${p}</td><td class="n">${x.races}</td>${['win','place'].map(k=>`<td class="n">${ci(x[k])}</td>`).join('')}<td class="n">${num(x.top3_in3)}</td>${['roi_tansho','roi_fukusho','roi_umaren','roi_wide'].map(k=>`<td class="n">${ci(x[k])}</td>`).join('')}</tr>`)}
 h+='<h2>v6e との差（同じレースでの差、レース単位のブートストラップ 95%）</h2>'+(D.eval.posthoc_v6e_note?'<p class="muted">'+D.eval.posthoc_v6e_note+'。v6e本番 との比較（2026 年を含む）は評価の後に足した。</p>':'')+table(['比べたもの','期間','レース','◎勝率','◎複勝率','◎○▲の3着内','◎単勝','◎複勝','馬連流し','ワイド流し'],dr);
 const SL=k=>k.replace(/^surf:/,'').replace(/^main:/,'');const gr=[];for(const p of P){const e=E['前日版'][p];if(!e)continue;for(const k of Object.keys(e))if(/^(G[123]|surf:|main:)/.test(k))gr.push(row(SL(k),p,e[k]))}
 h+='<h2>前日版の内訳（グレード・芝ダ・主の集計が使えたか）</h2>'+table(['区分',...head.slice(1)],gr)+'<p class="muted">主あり = 同じレース名 × 同じコースの過去の回が 4 回以上あったレース。副・全体だけ = それ未満（新設・コースの変更など）で、同じコースの別レースと全重賞の集計だけで点を付けた。5 レース未満の区分は載せない。</p>';
 const pr=D.prereg;h+=`<h2>凍結した規則（${pr.frozen_at}）</h2><div class="panel wrap">縮める強さ m（全体・副・主）= ${pr.shrink.m_g}・${pr.shrink.m_s}・${pr.shrink.m_m}、主の重み λ = ${pr.score.lam}、点の足し方 = ${pr.score.weight==='z'?'発見期間の z で重み付け':'等しい重み'}。前日版で使う型 ${pr.sets.preday.used.length}（採用 ${pr.sets.preday.adopted.length} を束ねたもの）、当日版 ${pr.sets.dayof.used.length}。<br>前日版で使う型: ${pr.sets.preday.used.map(t=>t+' '+TN[t]).join('、')}</div>`;
 if(D.roi){const X=D.roi;const vr=X.versions.map(v=>{const e=(E[v.name]||{})['2025-2026'];const a=e&&e.all;const df=e&&e['diff_vs_前日版'];const t=(D.eval.ties||{})[v.name];const q=v.params;
   return `<tr><td>${v.primary?'<b>主</b>':''}</td><td>${q.bet}</td><td>${q.crit}</td><td>${q.cap}</td><td class="n">${q.nmin}</td><td>${q.layer}</td><td>${q.base}</td><td>${q.tie}</td><td class="n">${v.n_good}</td><td class="n">${v.n_bad}</td><td class="n">${v.used.length}</td>`+
   (a?`<td class="n">${a.races}</td><td class="n" data-v="${a.place.est}">${ci(a.place)}</td><td class="n" data-v="${a.roi_tansho.est}">${ci(a.roi_tansho)}</td><td class="n" data-v="${a.roi_fukusho.est}">${ci(a.roi_fukusho)}</td><td class="n" data-v="${a.roi_wide.est}">${ci(a.roi_wide)}</td><td class="n" data-v="${df?df.place.est:''}">${df?ci(df.place):'—'}</td><td class="n" data-v="${df?df.roi_tansho.est:''}">${df?ci(df.roi_tansho):'—'}</td><td class="n" data-v="${df?df.roi_fukusho.est:''}">${df?ci(df.roi_fukusho):'—'}</td><td class="n">${t?pct(t.tie_top):'—'}</td><td class="n">${t?pct(t.zero_top):'—'}</td>`:'<td class="muted wrap">該当なし（残ったパターンが無いか、評価期間に当たる馬がいない）</td>'+'<td class="n">—</td>'.repeat(9))+'</tr>'});
  const VH=['','券種','基準','頭打ち','件数の下限','層','元の型','同点の並べ方','好の数','凡の数','使う型','レース','◎複勝率','◎単勝の回収率','◎複勝の回収率','ワイド流し','複勝率の差（−前日版）','単勝の回収率の差','複勝の回収率の差','◎が同点','◎の点が 0'];
  const vrB=vr.filter((_,i)=>X.versions[i].family!=='件数が少なく過学習のおそれがある版'),vrS=vr.filter((_,i)=>X.versions[i].family==='件数が少なく過学習のおそれがある版');
  h+=`<h2>回収率で選ぶ版（全 ${X.n_versions} 版、2025-2026）</h2><p class="muted wrap">パターン（型 × 水準 × 向き）を、発見 2018〜2021・確認 2022〜2024 の重賞で実際に全頭 100 円ずつ買った回収率で選んだ。同じ期間の重賞の全頭の回収率: 発見 単勝 ${X.overall_roi.disc.ret_tan.toFixed(3)}・複勝 ${X.overall_roi.disc.ret_fuku.toFixed(3)}、確認 単勝 ${X.overall_roi.conf.ret_tan.toFixed(3)}・複勝 ${X.overall_roi.conf.ret_fuku.toFixed(3)}。区間の下限 = 発見で片側 95%・確認で片側 90% の下限（凡は上限）が全体を上回る（下回る）。点＋0.10 = 点の推定が全体 ±0.10。頭打ち = 単勝 50 倍・複勝 10 倍で打ち切っても同じ向き。成績の向き（3 着内が期待を上回る／下回る）も両方の期間で満たすものだけ。凍結 ${X.frozen_at}。<b>${X.n_versions} 版を比べたので、一番良い版の値は偶然で上振れしている。</b></p>`+
   '<h2>件数が多い版（各期間 20 走・50 走以上）</h2>'+table(VH,vrB)+'<h2>件数が少なく過学習のおそれがある版（各期間 3・5・8 走以上）</h2><p class="muted">ユーザーの追記「件数が少ないやや過学習気味の抽出条件でも構わない」で足した版。件数が少ないので、点（縮めた点）が全体 ±0.10 を越えるかで判定した。偶然で残ったパターンが多く混じるおそれがある。縮めた点 = (払戻の和 + m × 全体の回収率) / (走数 + m)。</p>'+table(VH,vrS)+'<p class="muted">差は同じレースでの「回収率版 − 前日版（成績版）」。◎が同点 = ◎ と ○ の点が同じで、同点の並べ方（成績版の点、または 主の好成績パターンの数 → 馬番）で ◎ が決まったレースの割合。</p>'}
 if(D.eval_r1){const E1=D.eval_r1.versions;const r1=[];for(const [v,e] of Object.entries(E1))for(const p of P){if(e[p])r1.push(row(v,p,e[p].all))}
  h+='<h2>第 1 回の評価（中止の馬を直す前のデータ。'+(D.eval_r1.evaluated_at||'')+'）</h2>'+table(head,r1)}
 if(D.kill)h+=renderKill();
 $('#t-eval').innerHTML=h;document.querySelectorAll('#t-eval table').forEach(sortable)}
// 2) 重賞ごと
function renderRaces(){const R=D.races;const grades=[...new Set(R.map(r=>r.grade))].sort();
 $('#t-race').innerHTML=`<div class="ctl"><select id="rv" aria-label="会場"></select><select id="rc" aria-label="コース"></select><input id="rq" placeholder="重賞名・コースで絞り込み"><select id="rg"><option value="">全グレード</option>${grades.map(g=>`<option>${g}</option>`).join('')}</select><select id="rs"></select><select id="rk"><option value="">全パターン</option><option>好成績</option><option>凡走</option><option>参考</option><option>回収率で残った</option><option>成績だけ（回収率は足りない）</option></select></div><div id="rb"></div>`;
 const fill=()=>{const q=$('#rq').value.trim(),g=$('#rg').value,cv=$('#rv').value,cc=$('#rc').value;
  const L=R.filter(r=>(!g||r.grade===g)&&(!q||(r.rkey+r.course).includes(q))&&(!cv||(r.course||'').startsWith(cv))&&(!cc||r.course===cc));
  $('#rs').innerHTML=L.map((r,i)=>`<option value="${R.indexOf(r)}">${r.rkey}（${r.grade}・${r.course}）</option>`).join('');show()};
 const show=()=>{const r=R[+$('#rs').value];if(!r){$('#rb').innerHTML='<p class="muted">該当なし</p>';return}const k=$('#rk').value;
  const ex=r.excluded.length?r.excluded.map(e=>`${e.year} 年（${e.course}）`).join('、'):'なし';
  const rows=r.patterns.filter(p=>!k||p.k===k||p.rk===k).sort((a,b)=>b.x-a.x).map(p=>`<tr><td class="wrap">${LVT(p.t,p.lv)}<div class="muted" style="font-size:11px">${p.t} ${TN[p.t]}</div></td><td class="wrap">${p.lv}</td><td class="${kcls(p.k)}">${p.k}</td><td class="n" data-v="${p.m[0]+p.m[1]+p.m[2]+p.m[3]}">${rec(p.m)}</td><td class="n">${pct(p.mr[0])}</td><td class="n">${pct(p.mr[1])}</td><td class="n">${pct(p.mr[2])}</td><td class="n">${pct(p.ps)}</td><td class="n" data-v="${p.med[0]}">${p.med[0]}/${p.med[1]}</td><td class="n">${rec(p.s)}</td><td class="n">${pct(p.sr[2])}</td><td class="n">${p.x.toFixed(3)}</td><td>${p.ly}</td>${D.roi?`<td class="n" data-v="${p.rm?p.rm.np:''}">${p.rm?p.rm.np:'—'}</td><td class="n" data-v="${p.rm&&p.rm.t[0]!=null?p.rm.t[0]:''}">${roi3(p.rm&&p.rm.t)}</td><td class="n" data-v="${p.rm&&p.rm.f[0]!=null?p.rm.f[0]:''}">${roi3(p.rm&&p.rm.f)}</td><td class="n">${roi3(p.rs&&p.rs.f)}</td><td class="${rcls(p.rk)} wrap">${p.rk||''}</td>`:''}</tr>`);
  $('#rb').innerHTML=`<div class="panel">${r.rkey}（${r.grade}・${r.course}、最新の回 ${r.latest}）。主の集計: 2016〜2024 年の ${r.n_main_ed} 回・${r.n_main_runs} 走。副の集計（同じコースの OP 以上の別レース）: ${r.n_sub_runs} 走。<br>主から外した回（別のコース）: ${ex}</div>`+
   table(['切り口','水準','種別','主 [1-2-3-着外]','勝率','連対率','複勝率','縮小後の複勝率','3着内の回/回','副 [1-2-3-着外]','副の複勝率','点 x','使った層'].concat(D.roi?['主の走数（払戻あり）','主の単勝の回収率','主の複勝の回収率','副の複勝の回収率','回収率の判定']:[]),rows)+(D.roi?'<p class="muted">回収率 = その条件の馬を全頭 100 円ずつ買ったときの払戻 ÷ 購入額（2016〜2024 年、[ ] はレース単位のブートストラップ 95% 区間）。回収率の判定は主に見る版（'+D.roi.primary+'）で、パターン（型 × 水準 × 点の向き）が発見・確認の両方の重賞で回収率の基準を満たしたか。「成績だけ」= 好成績・凡走だが回収率の基準は満たさない。</p>':'')+`<p class="muted">${(!k||k==='参考')?`「参考」（好成績・凡走の条件に届かないもの）は |点| の大きい ${r.n_ref_shown??'—'} 件だけ載せた（全 ${r.n_ref_all??'—'} 件）。`:''}`+'点 x = log（縮小した 実際の3着内 ÷ 頭数から見た期待）。好成績 = 主の集計で上振れし、3着内が出た回が 3 回以上。凡走 = 主で下振れし、該当馬がいた回が 4 回以上で 3着内の回が 1 回以下。</p>';
  document.querySelectorAll('#rb table').forEach(sortable)};
 setupCourse($('#rv'),$('#rc'),D.course_opts||[],fill,true);$('#rq').oninput=fill;$('#rg').onchange=fill;$('#rs').onchange=show;$('#rk').onchange=show;fill()}
// 3) 検証のレース
function renderEvalRaces(){const R=D.eval_races;$('#t-er').innerHTML=`<div class="ctl"><input id="eq" placeholder="レース名で絞り込み"><select id="ey"><option value="">2025・2026</option><option>2025</option><option>2026</option></select><select id="eg"><option value="">全グレード</option><option>G1</option><option>G2</option><option>G3</option></select><select id="es"></select></div><div id="eb"></div>`;
 const fill=()=>{const q=$('#eq').value.trim(),y=$('#ey').value,g=$('#eg').value;const L=R.filter(r=>(!y||r.date.startsWith(y))&&(!g||r.grade===g)&&(!q||(r.name+r.rkey).includes(q)));
  $('#es').innerHTML=L.map(r=>`<option value="${R.indexOf(r)}">${r.date} ${r.rkey}（${r.grade}）</option>`).join('');show()};
 const show=()=>{const r=R[+$('#es').value];if(!r){$('#eb').innerHTML='<p class="muted">該当なし</p>';return}
  const rows=r.horses.map(h=>`<tr><td class="mk">${h.mk}</td><td class="n">${h.hn}</td><td class="n">${h.sc.toFixed(2)}</td><td class="n">${h.n_good}</td><td class="n">${h.n_bad}</td><td class="n">${h.fin??'—'}</td><td class="n">${h.v6??'—'}</td>${D.roi?`<td class="n">${h.rv??'—'}</td>`:''}<td>${h.jk??''}</td><td class="wrap"><details><summary>上位 ${h.p.length} パターン</summary>${h.p.map(p=>`<div class="${kcls(p.k)}">${LVT(p.t,p.lv)}（${p.t}）— ${p.ly} 主${rec(p.m)}（${p.med[0]}/${p.med[1]} 回）副${rec(p.s)} 全体${rec(p.g)} 点 ${p.x.toFixed(3)}${p.rm||p.rs||p.rg?` ／ 回収率（単・複）${['rm','rs','rg'].map((k,i)=>p[k]?`${['主','副','全重賞'][i]} ${p[k][1].toFixed(2)}・${p[k][2].toFixed(2)}（${p[k][0]} 走）`:'').filter(Boolean).join(' ')}`:''}${p.rk?` <span class="${rcls(p.rk)}">${p.rk}</span>`:''}</div>`).join('')}</details></td></tr>`);
  $('#eb').innerHTML=`<div class="panel">${r.date} ${r.name}（${r.grade}・${r.course}）。◎の払戻: 単勝 ${r.pay.tansho} 円・複勝 ${r.pay.fukusho} 円${r.has_pay?'':'（払戻のデータなし。評価から除外）'}</div>`+
   table(['印','馬番','点','好成績','凡走','着順','v6e本番の順位'].concat(D.roi?['回収率版の順位']:[]).concat(['騎手','該当パターン（|点| の大きい順）']),rows);document.querySelectorAll('#eb table').forEach(sortable)};
 $('#eq').oninput=fill;$('#ey').onchange=fill;$('#eg').onchange=fill;$('#es').onchange=show;fill()}
// 4) 切り口の採否
function renderTypes(){const rows=D.types.map(t=>`<tr><td>${t.id}</td><td class="wrap">${t.name}</td><td>${t.family}</td><td>${t.status??'比較だけ'}</td><td class="n">${t.z_disc==null?'—':t.z_disc.toFixed(2)}</td><td class="n">${t.z_conf==null?'—':t.z_conf.toFixed(2)}</td><td>${t.bh10_disc?'○':''}</td><td>${t.used_preday?'○':''}</td><td>${t.used_dayof?'○':''}</td><td>${t.day_of?'当日':''}</td><td>${t.start_year}〜</td></tr>`);
 const nb=D.not_built.map(x=>`<tr><td>${x.id}</td><td>${x.name}</td><td class="wrap">${x.reason}</td></tr>`);
 $('#t-ty').innerHTML='<h2>型ごとの採否（発見 2018〜2021・確認 2022〜2024）</h2>'+table(['ID','切り口','族','状態','z（発見）','z（確認）','BH 10%','前日版','当日版','値','集計の開始'],rows)+
  '<p class="muted">採用 = 発見で z ≥ 1.645 かつ 確認で z ≥ 1.2816。前日版・当日版の○は、束ねた後に点に使う型（相関 0.7 以上の型は代表 1 つにまとめる）。</p><h2>作れなかった・入れなかった切り口</h2>'+table(['ID','切り口','理由'],nb);
 document.querySelectorAll('#t-ty table').forEach(sortable)}
function renderRoi(){const X=D.roi;if(!X){$('#t-roi').innerHTML='<p class="muted">2 回目の評価の材料が無い</p>';return}
 const c=o=>o?`${o.n}/${o.races}`:'—';const rows=X.primary_patterns.map(p=>`<tr><td class="wrap">${LVT(p.t,p.lv)}<div class="muted" style="font-size:11px">${p.t} ${TN[p.t]}</div></td><td class="wrap">${p.lv}</td><td class="${p.dir==='好'?'good':'bad'}">${p.dir}</td>`+['disc','conf'].map(k=>{const o=p[k];return o?`<td class="n">${c(o)}</td><td class="n">${pct(o.y3)} <span class="muted">(${pct(o.e3)})</span></td><td class="n" data-v="${o.t[0]}">${roi3(o.t)}</td><td class="n" data-v="${o.f[0]}">${roi3(o.f)}</td>`:'<td colspan="4">—</td>'}).join('')+'</tr>');
 $('#t-roi').innerHTML=`<h2>主に見る版で残ったパターン（${X.primary}）</h2><p class="muted wrap">好 = そのレースの as-of の点が正のときに当てはまった馬、凡 = 負のとき。発見 2018〜2021・確認 2022〜2024 の重賞での実際の成績と回収率。[ ] はレース単位のブートストラップの、採否に使った片側の区間（発見は 5%〜95%、確認は 10%〜90% の点。好は下限、凡は上限が全体の回収率を越えたものが残る）。複勝率の（ ）は頭数から見た期待。</p>`+
  table(['切り口','水準','向き','発見 走/レース','発見 複勝率','発見 単勝の回収率','発見 複勝の回収率','確認 走/レース','確認 複勝率','確認 単勝の回収率','確認 複勝の回収率'],rows);document.querySelectorAll('#t-roi table').forEach(sortable)}
const R={eval:renderEval,race:renderRaces,er:renderEvalRaces,ty:renderTypes,roi:renderRoi},done={};
// 消しの版（3 回目の評価）
function renderKill(){const K=D.kill;const E=x=>String(x).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));const L=`${K.span[0]}-${K.span[1]}`;const B=K.base[L],Pe=K.primary_eval[L];if(!B||!Pe)return '';
 const pc=o=>o&&o.est!=null?`${pct(o.est)} <span class="muted">[${pct(o.lo)}, ${pct(o.hi)}]</span>`:'—';
 const dp=o=>o&&o.est!=null?`${o.est>=0?'+':''}${(100*o.est).toFixed(1)}pt <span class="muted">[${(100*o.lo).toFixed(1)}, ${(100*o.hi).toFixed(1)}]</span>`:'—';
 const BL={win:'◎ 勝率',place:'◎ 複勝率',top3_in3:'◎○▲ の 3 着内（頭）',roi_tansho:'◎ 単勝',roi_fukusho:'◎ 複勝',roi_umaren:'◎→○▲ 馬連（2 点）',roi_wide:'◎→○▲ ワイド（2 点）',
  roi_umaren_all:'◎→消しでない全頭 馬連',roi_wide_all:'◎→消しでない全頭 ワイド',roi_umaren_3:'◎→相手 3 頭 馬連',roi_wide_3:'◎→相手 3 頭 ワイド',roi_umaren_4:'◎→相手 4 頭 馬連',
  roi_wide_4:'◎→相手 4 頭 ワイド',roi_umaren_5:'◎→相手 5 頭 馬連',roi_wide_5:'◎→相手 5 頭 ワイド'};
 const num1=o=>o&&o.est!=null?`${o.est.toFixed(2)} <span class="muted">[${o.lo.toFixed(2)}, ${o.hi.toFixed(2)}]</span>`:'—';
 const r1=Object.keys(BL).map(k=>{const isN=k==='top3_in3';const f=isN?num1:pc;const d=Pe.diff_vs_前日版[k];
  const dd=isN?(d?`${d.est>=0?'+':''}${d.est.toFixed(2)} <span class="muted">[${d.lo.toFixed(2)}, ${d.hi.toFixed(2)}]</span>`:'—'):dp(d);
  const c=x=>x.all[k]&&x.all[k].cost!=null?Math.round(x.all[k].cost/100).toLocaleString()+' 点':'';
  return `<tr><td>${BL[k]}</td><td class="n">${f(B.all[k])}</td><td class="n">${f(Pe.all[k])}</td><td class="n">${dd}</td><td class="n">${c(B)}</td><td class="n">${c(Pe)}</td></tr>`});
 const ks=o=>`<td class="n">${o.races}</td><td class="n">${o.killed_per_race.toFixed(1)}</td><td class="n">${o.left_per_race.toFixed(1)}</td><td class="n">${pct(o.killed_top3_rate)}</td><td class="n">${pct(o.killed_e3_rate)}</td><td class="n">${pct(o.killed_win_rate)}</td><td class="n">${pct(o.race_miss_rate)}</td><td class="n">${pct(o.honmei_killed_rate)}</td>`;
 const r2=Object.keys(K.primary_eval).map(p=>`<tr><td>${E(p)}</td>${ks(K.primary_eval[p].kill)}</tr>`);
 const r3=K.table.map(v=>{const q=v.params;const nm=q.only_roi?'回収率の避けるだけ':`${q.crit}・${q.nmin}走・${q.layer}・${q.hits}つ以上・避ける${q.roi7==='入れる'?'あり':'なし'}`;
  const dv=(a)=>a&&a[0]!=null?`${pct(a[0])} <span class="muted">(${pct(a[1])})</span>`:'—';const k=v.kill;
  const dd=x=>x?`<span class="${x[1]>0?'good':x[2]<0?'bad':''}">${x[0]>=0?'+':''}${(100*x[0]).toFixed(1)}</span>`:'—';
  return `<tr><td class="wrap">${v.primary?'<b>'+E(nm)+'</b>（主に見る版）':E(nm)}</td><td>${E(q.honmei)}</td><td>${v.passes?'通過':''}</td><td class="n">${dv(v.dev.disc)}</td><td class="n">${dv(v.dev.conf)}</td>`+
   `<td class="n">${k.killed_per_race.toFixed(1)}</td><td class="n">${pct(k.killed_top3_rate)} <span class="muted">(${pct(k.killed_e3_rate)})</span></td><td class="n">${pct(k.race_miss_rate)}</td><td class="n">${pct(v.all.place)}</td>`+
   ['roi_wide','roi_umaren_all','roi_wide_all','roi_umaren_3','roi_wide_3'].map(c=>`<td class="n" data-v="${v.all[c]??''}">${pct(v.all[c])} ${dd(v.diff[c])}</td>`).join('')+'</tr>'});
 return `<h2>消しの版（3 回目の評価: 2025・2026 年の結果を二度見た後。${K.n_versions} 版）</h2><div class="panel wrap">好成績の点（前日版）で ◎○ を選び、極端に成績の悪いルールに当てはまる馬を「消し」にして、相手を消去法で絞る（ユーザーの依頼、統括役経由）。
 版と主に見る版は 2016〜2024 年の中だけで、払戻を使わない消しの精度で決め、${E(K.frozen_at)} に凍結してから ${E(K.evaluated_at)} に 1 回だけ評価した。主に見る版: <b>${E(K.primary)}</b>。
 消しの条件: ${E(K.rules.stat)}。基準: ${Object.entries(K.rules.crit).map(([a,b])=>E(a+' = '+b)).join('／')}。並び: ${E(K.rules.marks)}。買い方: ${E(K.rules.bets)}。
 「そのまま」の版は ◎ が前日版と同じなので、◎ の単勝・複勝は前日版と同じ値（差が出るのは相手の券種だけ）。区間はレース単位のポアソン(1) の重みのブートストラップ 95%。
 ${K.n_versions} 版 × 券種の数だけ比べているので、どれかの版が偶然よく見える。主に見る版以外の値は割り引いて読む。</div>
 <h2>主に見る版と前日版（消しなし、同じ ${B.all.races} レース）</h2>`+table(['買い方','前日版（消しなし）','主に見る消しの版','差（消し − 前日版）','前日版の点数','消しの版の点数'],r1)+
 '<h2>消しの誤り（主に見る版）</h2>'+table(['期間','レース','消した頭数','残った頭数','消した馬の 3 着内率','その期待（Σ3/頭数）','消した馬の 1 着率','消した馬が 3 着内に来たレース','◎ が消しに当たったレース'],r2)+
 '<h2>全版の一覧（2025・2026 年）</h2>'+table(['版','◎○','開発の基準','発見 2018〜2021 の消した馬の 3着内率（期待）','確認 2022〜2024','消した頭数','消した馬の 3着内率（期待）','消した馬が 3着内のレース','◎ 複勝率','◎→○▲ ワイド（差pt）','消しでない全頭 馬連（差pt）','消しでない全頭 ワイド（差pt）','相手 3 頭 馬連（差pt）','相手 3 頭 ワイド（差pt）'],r3)+
 '<p class="muted">差 pt の色: 緑 = 前日版との差の 95% 区間が 0 より上、赤 = 0 より下。差 pt は前日版（消しなし、同じレース）との差。区間が 0 より上の組は 2025・2026 年で 54 組あるが、偶然でも約 40 組は出る数で、しかも点数の多い「消しでない全頭の馬連」に偏るので、裏付けとしては弱い。</p>'}

function tab(k){document.querySelectorAll('.tabs button').forEach(b=>b.classList.toggle('on',b.dataset.k===k));document.querySelectorAll('section').forEach(s=>s.hidden=s.id!=='t-'+k);if(!done[k]){R[k]();done[k]=1}}
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>tab(b.dataset.k));tab('eval');
"""


MAX_REF_PER_RACE = 15     # 重賞ごとの表に載せる「参考」のパターンの数（|点| の大きい順。好成績・凡走はすべて載せる）
MAX_PAT_PER_HORSE = 6    # 検証のレースで 1 頭あたりに載せるパターンの数（全件は data/local/graded/races/<race_id>.json）


def slim(rep: dict) -> dict:
    """HTML を 10MB 程度に収めるため、載せるパターンを絞る（元の JSON は全件のまま。2 回目の評価で回収率の列が増えたので 30→15・10→6 に減らした）。"""
    for r in rep["races"]:
        keep = [p for p in r["patterns"] if p["k"] != "参考" or p.get("rk") == "回収率で残った"]
        ref = sorted((p for p in r["patterns"] if p["k"] == "参考" and p.get("rk") != "回収率で残った"), key=lambda p: -abs(p["x"]))
        r["n_ref_all"] = len(ref)
        r["n_ref_shown"] = min(len(ref), MAX_REF_PER_RACE)
        r["patterns"] = keep + ref[:MAX_REF_PER_RACE]
    for r in rep["eval_races"]:
        for h in r["horses"]:
            h["p"] = sorted(h["p"], key=lambda p: -abs(p["x"]))[:MAX_PAT_PER_HORSE]
    return rep


from src.scripts.docs.graded_course_filter import COURSE_JS  # noqa: E402


def add_texts(rep: dict) -> dict:
    """切り口 × 水準の日本語の文（lvtext: "型|水準" → 文）と、コースのプルダウンの候補（course_opts）を足す。"""
    from src.research.graded import level_text as LT
    from src.scripts.docs import graded_course_filter as CF
    pairs = {(p["t"], p["lv"]) for r in rep.get("races", []) for p in r.get("patterns", [])}
    pairs |= {(p["t"], p["lv"]) for r in rep.get("eval_races", []) for h in r.get("horses", []) for p in h.get("p", [])}
    pairs |= {(p["t"], p["lv"]) for p in (rep.get("roi") or {}).get("primary_patterns", []) if "t" in p and "lv" in p}
    rep["lvtext"] = {f"{t}|{lv}": LT.describe(t, lv)[0] for t, lv in sorted(pairs)}
    rep["course_opts"] = CF.options({r["rkey"]: r.get("course", "") for r in rep.get("races", [])})
    return rep


def build(src: Path = SRC, out: Path = OUT) -> Path:
    rep = add_texts(slim(json.loads(src.read_text(encoding="utf-8"))))
    data = json.dumps(rep, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    pre = rep["prereg"]
    html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>重賞のルールベース予想家</title><style>{CSS}</style></head><body><main>
<h1>重賞のルールベース予想家（研究）</h1>
<p class="muted">好成績のパターン（前走・条件の変化・脚質・経歴・騎手・厩舎・血統・調教など約 120 型）を、主（同じレース名 × 同じコースの過去の回）・副（同じコースの OP 以上の別レース）・
全重賞の 3 層で as-of に数え、縮めた上振れの和で ◎○▲△△ を付ける。オッズ・人気は条件に使わず、回収率の評価にだけ使う。規則は {pre['frozen_at']} に 2016〜2024 年の中だけで凍結し、
2025・2026 年の重賞で 1 回だけ評価した。{'そのあとユーザーの依頼で、回収率まで裏付けのあるパターンだけで印を付ける版を足し、中止の馬の扱いを直したデータで 2 回目の評価（データを一度見た後）を行った（' + rep['roi']['frozen_at'] + ' に版の一覧を凍結）。' if rep.get('roi') else ''}各重賞で点に効いているルールの全件の表: <a href="graded_rules_table.html">重賞のルール一覧</a>・重賞と通常レースの単位ごとの規則と訓練・2025・2026 年の成績: <a href="rules_overview.html">ルールの一覧（重賞・通常レース）</a>・ゴール（◎ 勝率 30%・単回収 120%、消しの 95%）に向けた分析サイクル: <a href="graded_cycles.html">重賞の分析サイクル</a>（重賞ごとのまとめ: <a href="graded_race_playbook.html">重賞の選び方の早見表</a>）・うまうまー式の穴馬の分析: <a href="graded_longshot.html">重賞の穴馬分析</a>・計画: docs/ideas/graded_rule_based_handicapper.md・切り口の目録: docs/ideas/graded_handicapper_cuts.md・コード: src/research/graded/</p>
<div class="tabs"><button data-k="eval">検証の結果</button><button data-k="race">重賞ごと</button><button data-k="er">検証のレース</button><button data-k="ty">切り口の採否</button>{'<button data-k="roi">回収率で残ったパターン</button>' if rep.get('roi') else ''}</div>
<section id="t-eval"></section><section id="t-race" hidden></section><section id="t-er" hidden></section><section id="t-ty" hidden></section><section id="t-roi" hidden></section>
</main><script type="application/json" id="data">{data}</script><script>{COURSE_JS}{JS}</script></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(out, f"{out.stat().st_size / 1e6:.1f} MB")
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    build(Path(a[0]) if a else SRC, Path(a[1]) if len(a) > 1 else OUT)
