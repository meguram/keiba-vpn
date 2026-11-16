"""重賞のルールベース予想家: 各重賞で点に効いているルールを、1 行 = 重賞 × ルールの 1 枚の表にする（外部の JS・CSS なし、ライト・ダーク両対応、スマホ幅）。

使い方: python -m src.scripts.docs.build_graded_rules_table_html [入力 JSON] [出力 HTML]
入力: <GRADED_OUT>/graded_rules.json（python -m src.research.graded.run export_rules の出力。列ごとの配列と辞書）
出力: data/meta/graded/graded_rules_table.html
計算はしない（JSON を埋め込み、絞り込み・並べ替え・ページ送りは素の JS で、見えているページの行だけ描く）。
"""

from __future__ import annotations

import json
import os
from html import escape
import sys
from pathlib import Path

from src.research.graded import level_text as LT
from src.scripts.docs import graded_course_filter as CF
from src.scripts.docs.build_graded_handicapper_html import CSS

REPO = Path(__file__).resolve().parents[3]
SRC = Path(os.environ.get("GRADED_OUT", "/home/jovyan/work/graded_out_v2")) / "graded_rules.json"
OUT = REPO / "data/meta/graded/graded_rules_table.html"

HEAD = ["重賞", "格", "コース", "条件（切り口 × 水準）", "水準", "好／凡", "区分", "集計の階層", "[1-2-3-着外]", "件数", "勝率", "連対率", "複勝率",
        "払戻の解けた走数", "単勝の回収率", "複勝の回収率", "縮めた点 x", "縮めた比", "型の重み", "主の [1-2-3-着外]", "採否", "回収率の判定", "消しに使う"]

EXTRA_CSS = """
.pager{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:8px 0}.pager button{border:1px solid var(--border);background:var(--panel);color:var(--text);
padding:4px 10px;border-radius:6px;cursor:pointer}.pager button:disabled{opacity:.4;cursor:default}th[data-s="a"]::after{content:" ▲"}th[data-s="d"]::after{content:" ▼"}
.ctl label{display:flex;flex-direction:column;font-size:12px;color:var(--muted);gap:2px}.ctl input[type=search]{min-width:12em}
.lg{display:inline-block;width:1.4em;height:.9em;border:1px solid var(--border);border-radius:3px;vertical-align:-1px;margin:0 4px 0 10px}
tr.ad-roi td,.lg.ad-roi{background:var(--row-roi)}tr.ad-kill td,.lg.ad-kill{background:var(--row-kill)}
td .sub{display:block;font-size:11px;color:var(--muted)}
:root{--row-roi:rgba(47,111,222,.24);--row-perf:rgba(47,111,222,.09);--row-kill:rgba(179,38,30,.15)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--row-roi:rgba(122,167,255,.30);--row-perf:rgba(122,167,255,.11);--row-kill:rgba(255,138,128,.20)}}
:root[data-theme="dark"]{--row-roi:rgba(122,167,255,.30);--row-perf:rgba(122,167,255,.11);--row-kill:rgba(255,138,128,.20)}
.theme{float:right;border:1px solid var(--border);background:var(--panel);color:var(--text);border-radius:14px;padding:2px 10px;cursor:pointer;font-size:12px}
"""

JS = r"""
const D=JSON.parse(document.getElementById('data').textContent);const C=D.cols;const N=C.r.length;
const $=s=>document.querySelector(s);const pct=v=>v==null?'—':(100*v).toFixed(1)+'%';const f2=v=>v==null?'—':v.toFixed(2);
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const RN=D.races.map(r=>`${r.rkey}`);const n_=i=>C.rec[i][0]+C.rec[i][1]+C.rec[i][2]+C.rec[i][3];
const rate=(i,k)=>{const n=n_(i);if(!n)return null;const a=C.rec[i];return (k===0?a[0]:k===1?a[0]+a[1]:a[0]+a[1]+a[2])/n};
function adopt(f){const s=(f&3)===3?'成績版（前日・当日）':(f&1)?'成績版（前日のみ）':(f&2)?'成績版（当日のみ）':'';return f&4?(s?s+'＋回収率版':'回収率版のみ'):s}
const judge=i=>(C.flags[i]&4)?'回収率でも残った':(C.flags[i]&8)?'回収率の基準は満たすが、全重賞の層なので回収率版では使わない':'成績だけ（回収率は足りない）';
// 並べ替えのキー（見出しの順）
const KEY=[i=>RN[C.r[i]],i=>D.races[C.r[i]].grade,i=>D.races[C.r[i]].course,i=>D.types[C.t[i]].id,i=>D.levels[C.lv[i]],i=>C.dir[i],i=>C.k[i],i=>C.ly[i],
 i=>C.rec[i][0]*1e6+C.rec[i][1]*1e3+C.rec[i][2],i=>n_(i),i=>rate(i,0),i=>rate(i,1),i=>rate(i,2),i=>C.np[i],i=>C.roi_t[i],i=>C.roi_f[i],i=>C.x[i],i=>C.x[i],
 i=>D.types[C.t[i]].w,i=>C.main[i][0],i=>C.flags[i]&7,i=>C.flags[i]&12,i=>C.flags[i]&16];
KEY[5]=i=>C.dir[i]*2+(C.flags[i]&32?0:1);
let view=[],page=0,sortK=-1,asc=true;const PS=200;
// 行の背景: 消しに使う（赤み）＞ 回収率版で採用（青）。成績版だけで採用の行（表のほぼ全行。成績版は点の付く水準をすべて使う）は色なし。
// 採否の列の文字はそのまま
const rowCls=i=>{const f=C.flags[i];return f&16?'ad-kill':f&4?'ad-roi':''};
const lvText=i=>D.lvtext[D.types[C.t[i]].id+'|'+D.levels[C.lv[i]]]||'';
function cell(i){const r=D.races[C.r[i]],t=D.types[C.t[i]];const dir=D.dict.dir[C.dir[i]],k=D.dict.k[C.k[i]];const ly=D.dict.ly[C.ly[i]];const sk=D.dict.sub_kind[C.sub_kind[i]];
 const m=C.main[i];const kp=C.flags[i]&4;
 return `<tr class="${rowCls(i)}"><td>${esc(r.rkey)}</td><td>${esc(r.grade)}</td><td>${esc(r.venue)}${esc(r.surf)}${r.dist}</td><td class="wrap">${esc(lvText(i))}<span class="sub">${esc(t.id)} ${esc(t.name)}</span></td><td class="wrap">${esc(D.levels[C.lv[i]])}</td>`+
 `<td class="${dir==='好'?(C.flags[i]&32?'good':''):'bad'}">${dir==='好'?(C.flags[i]&32?'好（買い）':'好傾向（弱い）'):'凡（避ける）'}</td><td>${esc(k)}</td><td>${esc(ly)}${sk?`<span class="muted">（${esc(sk)}）</span>`:''}</td>`+
 `<td class="n">[${C.rec[i].join('-')}]</td><td class="n">${n_(i)}</td><td class="n">${pct(rate(i,0))}</td><td class="n">${pct(rate(i,1))}</td><td class="n">${pct(rate(i,2))}</td>`+
 `<td class="n">${C.np[i]}</td><td class="n">${pct(C.roi_t[i])}</td><td class="n">${pct(C.roi_f[i])}</td><td class="n">${C.x[i].toFixed(4)}</td><td class="n">${f2(Math.exp(C.x[i]))}</td><td class="n">${t.w==null?'—':f2(t.w)}</td>`+
 `<td class="n">${m[0]?`[${m[1]}-${m[2]}-${m[3]}-${m[0]-m[1]-m[2]-m[3]}]`:'—'}</td><td class="wrap">${adopt(C.flags[i])}</td><td class="${kp?'good':'muted'} wrap">${judge(i)}</td><td class="${C.flags[i]&16?'bad':''}">${C.flags[i]&16?'消し':''}</td></tr>`}
function apply(){const cv=$('#cv').value,cc=$('#cc').value;const q=$('#q').value.trim().toLowerCase(),g=$('#g').value,dr=$('#dir').value,ly=$('#ly').value,ad=$('#ad').value,jd=$('#jd').value,kd=$('#kd').value,kl=$('#kl').value;
 const mn=parseInt($('#mn').value)||0;const qt=$('#qt').value.trim().toLowerCase();
 const rok=D.races.map(r=>(!q||(r.rkey+' '+r.course).toLowerCase().includes(q))&&(!g||r.grade===g)&&(!cv||r.course.startsWith(cv))&&(!cc||r.course===cc));
 const tok=D.types.map(t=>!qt||(t.id+' '+t.name).toLowerCase().includes(qt));
 view=[];for(let i=0;i<N;i++){if(!rok[C.r[i]])continue;if(qt&&!tok[C.t[i]]&&!D.levels[C.lv[i]].toLowerCase().includes(qt)&&!lvText(i).toLowerCase().includes(qt))continue;
  if(dr==='buy'&&!(C.dir[i]===0&&(C.flags[i]&32)))continue;if(dr==='weak'&&!(C.dir[i]===0&&!(C.flags[i]&32)))continue;if(dr==='1'&&C.dir[i]!==1)continue;if(ly!==''&&C.ly[i]!=+ly)continue;if(kd!==''&&C.k[i]!=+kd)continue;const f=C.flags[i];
  if(ad==='seiseki'&&!(f&3))continue;if(ad==='roi'&&!(f&4))continue;if(ad==='both'&&!((f&3)&&(f&4)))continue;if(ad==='seiseki_only'&&(f&4))continue;if(ad==='roi_only'&&(f&3))continue;
  if(ad==='preday'&&!(f&1))continue;if(ad==='dayof'&&!(f&2))continue;
  if(jd==='kept'&&!(f&4))continue;if(jd==='perf'&&(f&12))continue;if(jd==='ly'&&!(f&8))continue;if(kl==='kill'&&!(f&16))continue;if(kl==='keep'&&(f&16))continue;if(n_(i)<mn)continue;view.push(i)}
 if(sortK>=0)sortView();page=0;draw()}
function sortView(){const k=KEY[sortK];const sg=asc?1:-1;view.sort((a,b)=>{const x=k(a),y=k(b);if(x==null&&y==null)return a-b;if(x==null)return 1;if(y==null)return -1;return (x>y?1:x<y?-1:a-b)*sg})}
function draw(){const np=Math.max(1,Math.ceil(view.length/PS));page=Math.min(page,np-1);const s=page*PS;
 $('#tb').innerHTML=view.slice(s,s+PS).map(cell).join('');
 const nr=new Set(view.map(i=>C.r[i])).size;
 $('#cnt').textContent=`${view.length.toLocaleString()} 行（全 ${N.toLocaleString()} 行）・重賞 ${nr}`;
 document.querySelectorAll('.pg').forEach(e=>e.textContent=`${page+1} / ${np} ページ`);
 document.querySelectorAll('.prev').forEach(b=>b.disabled=page===0);document.querySelectorAll('.next').forEach(b=>b.disabled=page>=np-1)}
document.querySelectorAll('#th th').forEach((th,j)=>th.onclick=()=>{asc=sortK===j?!asc:true;sortK=j;document.querySelectorAll('#th th').forEach(h=>delete h.dataset.s);th.dataset.s=asc?'a':'d';sortView();page=0;draw()});
document.querySelectorAll('.prev').forEach(b=>b.onclick=()=>{page--;draw()});document.querySelectorAll('.next').forEach(b=>b.onclick=()=>{page++;draw()});
['#q','#qt','#mn'].forEach(s=>$(s).oninput=apply);['#g','#dir','#ly','#ad','#jd','#kd','#kl'].forEach(s=>$(s).onchange=apply);
setupCourse($('#cv'),$('#cc'),D.course_opts,apply,false);if($('#bcv'))setupCourse($('#bcv'),$('#bcc'),D.course_opts,()=>bapply(),false);
$('#theme').onclick=()=>{const r=document.documentElement;const dk=r.dataset.theme?r.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;r.dataset.theme=dk?'light':'dark'};
apply();

// 買いの組み合わせ（主に見る買いの版、2016〜2024 年で探した組）
const BY=D.buy;let bview=[],bpage=0,bsk=-1,basc=true;
const BN=i=>BY.cols.rec[i].reduce((a,b)=>a+b,0);
const BKEY=[i=>RN[BY.cols.r[i]],i=>D.races[BY.cols.r[i]].grade,i=>BY.cols.layer[i],i=>BY.cols.types[i],i=>BY.cols.levels[i],i=>BY.cols.rec[i][0]*1e6+BY.cols.rec[i][1]*1e3+BY.cols.rec[i][2],BN,
 i=>(BY.cols.rec[i][0]+BY.cols.rec[i][1]+BY.cols.rec[i][2])/BN(i),i=>BY.cols.roi_t[i],i=>BY.cols.roi_f[i],i=>BY.cols.ed[i],i=>BY.cols.nh[i],i=>BY.cols.p0[i],i=>BY.cols.pbin[i],i=>BY.cols.n_equiv[i]];
function bcell(i){const c=BY.cols,r=D.races[c.r[i]];const a=c.rec[i];const n=BN(i);
 return `<tr><td>${esc(r.rkey)}</td><td>${esc(r.grade)}</td><td>${esc(c.layer[i])}</td><td class="wrap">${esc(c.text[i])}<span class="sub">${esc(c.types[i]).replace(/\|/g,' × ')}</span></td><td class="wrap">${esc(c.levels[i])}</td>`+
 `<td class="n">[${a.join('-')}]</td><td class="n">${n}</td><td class="n">${pct((a[0]+a[1]+a[2])/n)}</td><td class="n">${pct(c.roi_t[i])}</td><td class="n">${pct(c.roi_f[i])}</td>`+
 `<td class="n">${c.ed[i]}</td><td class="n">${c.nh[i]}</td><td class="n">${pct(c.p0[i])}</td><td class="n">${c.pbin[i].toExponential(1)}</td><td class="n">${c.n_equiv[i]}</td></tr>`}
function bapply(){const q=$('#bq').value.trim().toLowerCase(),ly=$('#bly').value,mn=parseInt($('#bmn').value)||0,qt=$('#bqt').value.trim().toLowerCase();
 const cv=$('#bcv').value,cc=$('#bcc').value;
 bview=[];const c=BY.cols;for(let i=0;i<c.r.length;i++){const r=D.races[c.r[i]];if(q&&!(r.rkey+' '+r.course).toLowerCase().includes(q))continue;if(ly&&c.layer[i]!==ly)continue;
  if(cv&&!r.course.startsWith(cv))continue;if(cc&&r.course!==cc)continue;
  if(BN(i)<mn)continue;if(qt&&!(c.types[i]+' '+c.levels[i]+' '+c.text[i]).toLowerCase().includes(qt))continue;bview.push(i)}
 if(bsk>=0)bsort();bpage=0;bdraw()}
function bsort(){const k=BKEY[bsk],sg=basc?1:-1;bview.sort((a,b)=>{const x=k(a),y=k(b);if(x==null&&y==null)return a-b;if(x==null)return 1;if(y==null)return -1;return (x>y?1:x<y?-1:a-b)*sg})}
function bdraw(){const np=Math.max(1,Math.ceil(bview.length/PS));bpage=Math.min(bpage,np-1);const s=bpage*PS;$('#btb').innerHTML=bview.slice(s,s+PS).map(bcell).join('');
 $('#bcnt').textContent=`${bview.length.toLocaleString()} 行（載せた ${BY.cols.r.length.toLocaleString()} 行）`;document.querySelectorAll('.bpg').forEach(e=>e.textContent=`${bpage+1} / ${np} ページ`);
 document.querySelectorAll('.bprev').forEach(b=>b.disabled=bpage===0);document.querySelectorAll('.bnext').forEach(b=>b.disabled=bpage>=np-1)}
if(BY){document.querySelectorAll('#bth th').forEach((th,j)=>th.onclick=()=>{basc=bsk===j?!basc:true;bsk=j;document.querySelectorAll('#bth th').forEach(h=>delete h.dataset.s);th.dataset.s=basc?'a':'d';bsort();bpage=0;bdraw()});
 document.querySelectorAll('.bprev').forEach(b=>b.onclick=()=>{bpage--;bdraw()});document.querySelectorAll('.bnext').forEach(b=>b.onclick=()=>{bpage++;bdraw()});
 ['#bq','#bqt','#bmn'].forEach(s=>$(s).oninput=bapply);$('#bly').onchange=bapply;bapply()}
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tabs button').forEach(x=>x.classList.toggle('on',x===b));
 document.querySelectorAll('section.tab').forEach(s=>s.hidden=s.id!==b.dataset.k)});
"""


def _opt(pairs) -> str:
    return "".join(f'<option value="{escape(v)}">{escape(t)}</option>' for v, t in pairs)


BUY_HEAD = ["重賞", "格", "層", "条件（切り口の AND）", "水準", "[1-2-3-着外]", "件数", "3着内率", "単勝の回収率", "複勝の回収率", "異なる回", "異なる馬",
            "期待の3着内率", "二項の確率（単独）", "同じ馬の集まりになる条件の数"]


def buy_section(B: dict | None, R: dict) -> tuple[str, str]:
    """「買いの組み合わせ」の節（主に見る買いの版）と、上のタブ。B が無ければ空。"""
    if not B:
        return "", ""
    cnt = B["counts"]
    chance = [v for k, v in cnt.items() if k.startswith("perm")]
    ch = sum(chance) / len(chance) if chance else 0
    dv = B["dev"].get("real", {})
    dp = [v for k, v in B["dev"].items() if k.startswith("perm")]
    def rec(e):
        b = e.get("buy", {})
        return f"{b.get('n', 0)} 頭 [{'-'.join(map(str, b.get('rec', [])))}]・3着内率 {100 * (b.get('top3_rate') or 0):.1f}%・単勝 {100 * (b.get('roi_t') or 0):.1f}%・複勝 {100 * (b.get('roi_f') or 0):.1f}%"
    E = B.get("eval")
    ev_li = ""
    if E:
        ci = E["ci"]
        pc = lambda o: f"{100 * o['est']:.1f}%［{100 * o['lo']:.1f}, {100 * o['hi']:.1f}］"   # noqa: E731
        dpt = lambda o: f"{100 * o['est']:+.1f}pt［{100 * o['lo']:+.1f}, {100 * o['hi']:+.1f}］"   # noqa: E731
        hb, hp = E["real"]["honmei_buy"], E["real"]["honmei_preday_same_races"]
        ev_li = (f"<li><b>{E['evaluation_round']} 回目の評価（2025・2026 年の結果を {E['evaluation_round'] - 1} 度見た後。{escape(E['evaluated_at'])} に 1 回だけ）</b>: 最新の回と同じコースの重賞 {E['races']} レース"
                 f"（別のコースの {E['races_other_course']} レースは外した）。買いの馬 {escape(rec(E['real']))}（{E['real']['races_with_buy']} レース、出走馬の "
                 f"{100 * E['real']['buy_share']:.1f}%）。レース単位の 95% 区間: 3着内率 {pc(ci['top3_rate'])}・単勝 {pc(ci['roi_t'])}・複勝 {pc(ci['roi_f'])}。"
                 f"◎ を買いの馬に限ると 3着内率 {100 * (hb['top3_rate'] or 0):.1f}%・単勝 {100 * (hb['roi_t'] or 0):.1f}%、同じレースの前日版の ◎ は "
                 f"{100 * (hp['top3_rate'] or 0):.1f}%・{100 * (hp['roi_t'] or 0):.1f}%（差 3着内率 {dpt(ci['honmei_place_diff'])}）。"
                 f"偶然の組の買いの馬: {escape(' ／ '.join(rec(x) for x in E['perm']))}。全 {B['n_versions']} 版のうち 3 着内 7 割以上は {E['n_versions_70']} 版（買いの馬が 20 頭以上の版では最高 {100 * (E['max_top3_20'] or 0):.1f}%）。</li>")
    pager = '<div class="pager"><button class="bprev">前へ</button><span class="bpg"></span><button class="bnext">次へ</button></div>'
    sec = f"""<section class="tab" id="t-buy" hidden><div class="panel wrap"><p><b>買いの組み合わせ</b>（主に見る買いの版: {escape(B['primary'])}、{escape(B['frozen_at'])} に凍結、全 {B['n_versions']} 版）。
ユーザーの依頼（統括役経由）: 「買い」は 3 着内が 7〜8 割を占め、単勝か複勝の回収率が 100% を超える抽出条件に限る（過学習は許す）。</p>
<ul class="muted"><li>条件: {escape(B['rules']['cell'])}。{escape(B['rules']['version'])}。</li>
<li>探した型の組み合わせの数（1 重賞あたり）: 主 {B['n_type_combos']['主']:,}・副 {B['n_type_combos']['副']:,}。{escape(B['rules']['prune'])}。</li>
<li>2016〜2024 年で見つかった組（主に見る版の条件）: 本物 {cnt.get('real', 0):,} 組。結果を入れ替えた偶然のデータでも 平均 {ch:,.0f} 組 見つかる。{escape(B['rules']['chance'])}。</li>
<li>開発の検証（2016〜2021 年で探して 2022〜2024 年の重賞に当てはめた買いの馬）: 本物の組 {escape(rec(dv))}。偶然の組 {escape(' ／ '.join(rec(e) for e in dp))}。</li>
{ev_li}<li>{escape(B['rules']['roi_note'])}。{escape(B['rules']['dedupe'])}。</li>
<li>表は重賞 × 層ごとに、二項の確率の小さい順に {B['max_per_race']} 組まで（載せた {B['n_rows']:,} 組）。</li></ul></div>
<div class="ctl"><label>会場<select id="bcv"></select></label><label>コース<select id="bcc"></select></label><label>重賞名・コース<input id="bq" type="search" placeholder="例: 京都牝馬S"></label><label>層<select id="bly">{_opt([("", "すべて"), ("主", "主"), ("副", "副")])}</select></label>
<label>切り口・水準<input id="bqt" type="search" placeholder="例: A04・1着"></label><label>件数の下限<input id="bmn" type="number" min="0" value="0" style="width:6em"></label></div>
<div class="muted" id="bcnt"></div>{pager}<div class="scroll"><table><thead><tr id="bth">{''.join(f'<th>{h}</th>' for h in BUY_HEAD)}</tr></thead><tbody id="btb"></tbody></table></div>{pager}</section>"""
    tabs = '<div class="tabs"><button class="on" data-k="t-rules">ルールの一覧</button><button data-k="t-buy">買いの組み合わせ</button></div>'
    return sec, tabs


def add_texts(R: dict) -> dict:
    """切り口 × 水準の日本語の文（lvtext: "型|水準" → 文）・買いの組の文（buy.cols.text）・コースのプルダウンの候補（course_opts）を足す。"""
    C = R["cols"]
    pairs = {(R["types"][t]["id"], R["levels"][lv]) for t, lv in zip(C["t"], C["lv"])}
    R["lvtext"] = {f"{t}|{lv}": LT.describe(t, lv)[0] for t, lv in sorted(pairs)}
    B = R.get("buy")
    if B:
        B["cols"]["text"] = [LT.describe_combo(t.split("|"), lv.split("|")) for t, lv in zip(B["cols"]["types"], B["cols"]["levels"])]
    R["course_opts"] = CF.options({r["rkey"]: r["course"] for r in R["races"]})
    return R


def build(src: Path = SRC, out: Path = OUT) -> Path:
    R = add_texts(json.loads(src.read_text(encoding="utf-8")))
    data = json.dumps(R, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    o = R["overall_roi"]
    base = "・".join(f"{lab} 単勝 {100 * o[k]['ret_tan']:.1f}%・複勝 {100 * o[k]['ret_fuku']:.1f}%" for k, lab in (("disc", "2018〜2021 年"), ("conf", "2022〜2024 年")) if k in o)
    pats = escape("、".join(f"{t} {lv}（{dr}）" for t, lv, dr in R["primary_patterns"]))
    y0, y1 = R["years"]
    K = R.get("kill")
    kill_li = (f'<li><b>消しに使う</b>: 主に見る消しの版（{escape(K["primary"])}、{escape(K["frozen_at"])} に凍結）の基準を、'
               f'この表の {y0}〜{y1} 年の凍結の値に当てはめた場合に当たる行。基準: {escape(K["rules"]["stat"])}。'
               '評価（2025・2026 年）では、各レースの日より前の件数（as-of）で判定するので、同じ行でも当たり方が変わる。'
               '「2 つ以上で消す」版では、馬が 2 つ以上の行に当たったときだけ消す（この列は行ごとの当たり）。</li>') if K else ""
    ctl = (f'<div class="ctl">'
           f'<label>会場<select id="cv"></select></label><label>コース<select id="cc"></select></label>'
           f'<label>重賞名・コース<input id="q" type="search" placeholder="例: 皐月賞・中山芝2000"></label>'
           f'<label>格<select id="g">{_opt([("", "すべて"), ("G1", "G1"), ("G2", "G2"), ("G3", "G3")])}</select></label>'
           f'<label>条件（文・切り口・水準）<input id="qt" type="search" placeholder="例: 2 走前・A01・4角"></label>'
           f'<label>好／凡<select id="dir">{_opt([("", "すべて"), ("buy", "好（買い）"), ("weak", "好傾向（弱い）"), ("1", "凡（避ける）")])}</select></label>'
           f'<label>区分<select id="kd">{_opt([("", "すべて"), ("0", "好成績"), ("1", "凡走"), ("2", "参考")])}</select></label>'
           f'<label>集計の階層<select id="ly">{_opt([("", "すべて"), ("0", "主"), ("1", "副"), ("2", "全重賞")])}</select></label>'
           f'<label>採否<select id="ad">{_opt([("", "すべて"), ("seiseki", "成績版で採用"), ("preday", "前日版で採用"), ("dayof", "当日版で採用"), ("roi", "回収率版で採用"), ("both", "両方で採用"), ("seiseki_only", "成績版だけ"), ("roi_only", "回収率版だけ")])}</select></label>'
           f'<label>回収率の判定<select id="jd">{_opt([("", "すべて"), ("kept", "回収率でも残った"), ("ly", "基準は満たすが全重賞の層で使わない"), ("perf", "成績だけ")])}</select></label>'
           f'<label>消しに使う<select id="kl">{_opt([("", "すべて"), ("kill", "消しに使う"), ("keep", "消しに使わない")])}</select></label>'
           f'<label>件数の下限<input id="mn" type="number" min="0" step="1" value="0" inputmode="numeric" style="width:6em"></label>'
           f'</div>')
    buy_sec, tabs = buy_section(R.get("buy"), R)
    Bk = R.get("buy") or {}
    n32 = sum(1 for f in R["cols"]["flags"] if f & 32)
    buy_note = (f"今の凍結では、切り口 1 つで条件を満たす組は {Bk['n_depth1']} 組（走数 {Bk['depth1_main_n'][0]}〜{Bk['depth1_main_n'][1]}）で、"
                f"重賞・層まで一致して「好（買い）」になった行は {n32} 行。" if Bk.get("depth1_main_n") else "")
    pager = '<div class="pager"><button class="prev">前へ</button><span class="pg"></span><button class="next">次へ</button></div>'
    html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>重賞のルール一覧</title><style>{CSS}{EXTRA_CSS}</style></head><body><main>
<button id="theme" class="theme" type="button">明暗</button>
<h1>重賞のルール一覧（ルールベース予想家・研究）</h1>
<div class="panel wrap"><p>1 行 = 重賞 × その重賞で点に効いているルール（切り口 × 水準）。全 {R['n_rows']:,} 行・重賞 {len(R['races'])}。
集計の期間は <b>{y0}〜{y1} 年（凍結した値）</b>で、各重賞の最新の回のコースで数えた（規則の凍結 {R['frozen_at']}、回収率版の凍結 {R['frozen_at_roi']}）。
評価（2025・2026 年）はやり直していない。</p>
<ul class="muted">
<li><b>集計の階層</b>: 点 x を作った層。主 = 同じレース名 × 同じコースの過去の回（主の走数 15 以上かつ 4 回以上のとき）、副 = 同じコースの OP 以上の別レース（足りないコースは 3 勝クラス以上）、全重賞 = 全コースの重賞。
件数・率・回収率は、その層の母集団の値（主の件数が無い水準も、副・全重賞で点が付けば載せた。全重賞の行は、同じ切り口・水準なら重賞が違っても同じ値になる）。参考に主の [1-2-3-着外] も載せた。</li>
<li><b>回収率</b> = その母集団の馬を全頭 100 円ずつ買ったときの払戻 ÷ 購入額。分母は払戻の解けた走数（件数とは少し違う）。区間は載せていない。
比べる相手の、全重賞の全頭の回収率: {base}。</li>
<li><b>縮めた点 x</b> = log（縮めた 実際の3着内 ÷ 頭数から見た期待）。正なら好、負なら凡（避ける）。好のうち「好（買い）」は、「買いの組み合わせ」のタブの主に見る版の組の条件（3 着内 70% 以上・回収率 100% 超・走数の下限など）を満たす切り口 1 つの組と、重賞・層・切り口・水準が同じ行だけ。ほかは「好傾向（弱い）」（3 着内の割合が低く、着外が多い）。馬が買いになるのは、版の下限の数以上の組に当たったとき。{buy_note}縮めた比 = exp(x)。成績版の点は、馬ごとの x × 型の重み（発見の z）の和。回収率版は重みを付けない（回収率版だけで使う型の重みは —）。</li>
<li><b>区分</b>: 好成績 = 主で上振れし、3着内が出た回が 3 回以上。凡走 = 主で下振れし、該当馬がいた回が 4 回以上で 3着内の回が 1 回以下。参考 = それ以外（点には効く）。</li>
<li><b>採否</b>: 成績版（前日版・当日版）は、点に使う型（束ねの代表）の x≠0 の水準すべてで採用。前日版が評価の比較の基準。
回収率版（主に見る版: {escape(R['primary'])}）は、発見 2018〜2021・確認 2022〜2024 の両方で回収率の基準を満たしたパターン（{pats}）が主・副の層で当たるときだけ点が付き、ほかの水準は点 0。</li>
<li><b>回収率の判定</b>: 回収率でも残った = 上の回収率版で採用されたパターン。基準は満たすが全重賞の層で使わない = パターンは回収率の基準を満たしたが、この重賞では点が全重賞の層で付くので、回収率版（主・副の層だけ）では点 0。成績だけ = 成績では点が付くが、回収率の基準は満たさない。</li>
{kill_li}</ul></div>
{tabs}<section class="tab" id="t-rules">
{ctl}<div class="muted" style="font-size:12px">行の色: <span class="lg ad-roi"></span>回収率版で採用<span class="lg ad-kill"></span>消しに使う。色なし = 成績版（前日・当日）だけで採用（表のほぼ全行。成績版は点の付く水準をすべて使うため色で分けない）。採否は「採否」「消しに使う」の列にも文字で出す</div>
<div class="muted" id="cnt"></div>{pager}
<div class="scroll"><table><thead><tr id="th">{''.join(f'<th>{h}</th>' for h in HEAD)}</tr></thead><tbody id="tb"></tbody></table></div>{pager}
</section>{buy_sec}
<p class="muted">結果の全体（検証・各重賞の表・検証のレース）: <a href="graded_handicapper.html">重賞のルールベース予想家</a>。
作り: src/research/graded/rules_table.py → src/scripts/docs/build_graded_rules_table_html.py</p>
</main><script type="application/json" id="data">{data}</script><script>{CF.COURSE_JS}{JS}</script></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(out, f"{out.stat().st_size / 1e6:.1f} MB", R["n_rows"], "rows")
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    build(Path(a[0]) if a else SRC, Path(a[1]) if len(a) > 1 else OUT)
