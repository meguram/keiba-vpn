"""race_time_explorer.html（基準の走破時計の関数 v2 を、コースごとに見る 1 枚の HTML）を作る。

  python -m src.research.race_time.explorer          # docs/html/operations/final/race_time_explorer.html

- 係数（I-spline の節点と係数・コースの水準・残差の分位）を JSON で埋め込み、計算はページの JS で行う（外部の CDN は使わない）。
- 実際のレースの点は、コースごとに 250 件まで間引く（ファイルは 5MB 以下）。図は選んだコースだけを描く。
"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.race_time.features import AGE_LABEL, CLASS_LABEL
from src.pipeline.race_time.model import RaceTimeModel

REPO = Path(__file__).resolve().parents[3]
MODEL = REPO / "config/race_time_model/v2.json"
REPORT_DIR = REPO / "docs/operations/race_time"
CACHE = REPO / "data/local/research/race_time/races.parquet"
OUT = REPO / "docs/html/operations/final/race_time_explorer.html"
MAX_POINTS = 250
REF = {"class_ord": 2, "age_ord": 2, "newcomer": 0, "field_size": 14}     # コースの一覧の「基準の時計」の条件（1勝・古馬・14 頭・平均的な前半）


def _course_table(m: RaceTimeModel, races: pd.DataFrame) -> list[dict]:
    rows = []
    cnt = races["course"].value_counts()
    for c in m.courses:
        ref = pd.DataFrame([{**REF, "course": c, "front600": m.pace_ref[c]}])
        base = float(m.predict(ref)[0])
        cls = m.term_curve(c, "cls", np.array([1.0, 8.0]))
        pace = m.term_curve(c, "pace", np.array([-0.5, 0.5]))
        rows.append({"course": c, "n": int(cnt.get(c, 0)), "base": round(base, 2), "per_class": round(float(cls[1] - cls[0]) / 7, 3),
                     "per_front_sec": round(float(pace[1] - pace[0]), 3)})
    return rows


def _points(m: RaceTimeModel, races: pd.DataFrame) -> dict[str, list]:
    rng = np.random.default_rng(20261126)
    out: dict[str, list] = {}
    d = races.dropna(subset=["front600"])
    for c, g in d.groupby("course"):
        if c not in m.pace_ref:
            continue
        if len(g) > MAX_POINTS:
            g = g.iloc[np.sort(rng.choice(len(g), MAX_POINTS, replace=False))]
        out[c] = [[int(r.class_ord), int(r.age_ord), int(r.newcomer), int(r.field_size), round(float(r.front600), 2), round(float(r.y_top5), 2),
                   str(r.track_condition or ""), str(r.date)] for r in g.itertuples()]
    return out


def build() -> Path:
    m = RaceTimeModel.load(MODEL)
    races = pd.read_parquet(CACHE)
    races = races[races["date"] < "20250101"]
    rep = {}
    for name in ("compare_v2_report.json", "compare_report.json", "fit_report.json", "levels_report.json"):
        p = REPORT_DIR / name
        if p.exists():
            rep[name] = json.loads(p.read_text(encoding="utf-8"))
    boot_p = REPORT_DIR / "bootstrap_v2.json"
    boot = json.loads(boot_p.read_text(encoding="utf-8")) if boot_p.exists() else None
    viol = m.monotone_violations(n_grid=200)
    data = {"model": m.to_json(), "courses": _course_table(m, races), "points": _points(m, races), "class_label": CLASS_LABEL,
            "age_label": AGE_LABEL, "viol": viol, "boot": boot}
    c2 = rep.get("compare_v2_report.json", {})
    mt = c2.get("metrics", {})
    row = lambda k, lab: (f"<tr><td>{lab}</td><td>{mt[k]['loo_mad']:.3f}</td><td>{mt[k]['icc']:.3f}</td><td>{mt[k]['next_day']:.3f}</td></tr>"  # noqa: E731
                          if k in mt else "")
    viol_total = sum(sum(r.values()) for r in viol.values())
    lv = rep.get("levels_report.json", {}).get("member_check", {})
    tert = "".join(f"<tr><td>{html.escape(str(t['tert']))}</td><td>{int(t['n']):,}</td><td>{t['top3']*100:.1f}%</td><td>{t['pos_dev']:.3f}</td></tr>"
                   for t in lv.get("by_tertile", []))
    meta = m.meta
    page = TEMPLATE.replace("%DATA%", json.dumps(data, ensure_ascii=False, separators=(",", ":"))).replace("%VERSION%", html.escape(str(meta.get("version")))) \
        .replace("%BUILT%", html.escape(str(meta.get("built")))).replace("%TRAIN%", html.escape(str(meta.get("train")))) \
        .replace("%STATUS%", html.escape(str(meta.get("status")))).replace("%LAM%", str(meta.get("lam"))).replace("%FSIGN%", "減（多いほど速い）" if meta.get("field_sign") == -1 else "増（多いほど遅い）") \
        .replace("%CMP%", row("new_v2", "第 2 版（採用）") + row("eng_nocond", "従来の馬場の速さ（同じ窓・馬場状態の群なし）") + row("new_v2_2nd", "第 2 版・2 着の時計にそろえた版（参考）")) \
        .replace("%NCOMMON%", f"{c2.get('n_races_common', 0):,}").replace("%VIOL%", str(viol_total)).replace("%TERT%", tert) \
        .replace("%CORR%", f"{lv.get('corr_top5_win', float('nan')):.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    return OUT


TEMPLATE = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>走破時計の分布（コース別）</title>
<style>
:root{--bg:#0f172a;--surface:#111827;--border:#334155;--text:#e5e7eb;--dim:#94a3b8;--accent:#60a5fa;--band95:rgba(96,165,250,.15);--band80:rgba(96,165,250,.28);--band50:rgba(96,165,250,.45)}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.6 system-ui,"Noto Sans JP",sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:4px 0 2px}h2{font-size:16px;margin:22px 0 6px}
.dim{color:var(--dim)}.box{border:1px solid var(--border);border-radius:8px;padding:10px 12px;background:var(--surface);margin:8px 0}
table{border-collapse:collapse;width:100%;font-size:12.5px}th,td{border-bottom:1px solid var(--border);padding:4px 6px;text-align:right}th:first-child,td:first-child{text-align:left}
select,input{background:#0b1220;color:var(--text);border:1px solid var(--border);border-radius:6px;padding:3px 6px}
.ctl{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:8px 14px;align-items:center}
.small{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:10px}
svg{width:100%;height:auto;background:var(--surface);border:1px solid var(--border);border-radius:8px}
a{color:var(--accent)}tr.click{cursor:pointer}tr.click:hover{background:#1e293b}
.scroll{max-height:360px;overflow:auto}
</style></head><body><main>
<h1>走破時計の分布（コース別）</h1>
<div class="dim">式の版 %VERSION% ・作った日 %BUILT% ・学習 %TRAIN%（as-of: この期間の後のレースには固定した式を当てる）・状態: %STATUS%</div>
<div class="box">
<b>基準の時計 T̂</b> ＝ コースの水準 ＋ クラス（単調）＋ 年齢の条件（単調・コースごと）＋ 新馬 ＋ 頭数（単調）＋ 前半 600m の速さ（単調）。馬場状態は入れない（馬場の速さを測るため）。
目的の値は 1〜5 着の走破時計の中央値。各項は 3 次の I-spline（係数 ≥ 0）で、<b>コースごとの合計の曲線まで単調を係数の制約で保証</b>（下の表で破れ 0）。
外れ値は Huber の損失、コースごとの曲線は共通の形へ縮める（λ=%LAM%）。頭数の向きは学習の中の検証で %FSIGN%。前半 600m はラップから補間（結果表の前半 3F は 1700・1900・2100m で 500m 分のため使わない）。
</div>

<h2>コースを選ぶ</h2>
<div class="ctl">
<label>競馬場 <select id="venue"></select></label><label>芝ダ <select id="surf"></select></label><label>距離 <select id="dist"></select></label>
<span id="cinfo" class="dim"></span>
</div>
<div class="ctl" style="margin-top:8px">
<label>クラス <select id="cls"></select></label>
<label>年齢の条件 <select id="age"></select></label>
<label><input type="checkbox" id="newc"> 新馬</label>
<label>頭数 <input type="range" id="fld" min="5" max="18" step="1"> <b id="fldv"></b></label>
<label>前半 600m <input type="range" id="front" step="0.1"> <b id="frontv"></b></label>
<label>（参考）馬場状態の点 <select id="cond"><option value="">すべて</option><option>良</option><option>稍重</option><option>重</option><option>不良</option></select></label>
</div>
<h2>走破時計の分布</h2>
<div id="dist_txt" class="dim"></div>
<svg id="distsvg" viewBox="0 0 900 220"></svg>
<h2>1 つずつ動かしたときの基準の時計（選んだ条件の点が中心）</h2>
<div class="small"><svg id="c_cls" viewBox="0 0 320 200"></svg><svg id="c_fld" viewBox="0 0 320 200"></svg><svg id="c_front" viewBox="0 0 320 200"></svg></div>
<div class="dim" id="bootnote"></div>

<h2>コースの一覧</h2>
<div class="dim">基準の時計は 1勝・古馬・14 頭・平均的な前半の条件。クラス 1 段あたり＝未勝利→G1 の差 ÷ 7。前半 1 秒あたり＝前半 600m が 1 秒遅いときの差。行を押すとそのコースへ。</div>
<div class="scroll"><table id="ctab"><thead><tr><th>コース</th><th>件数</th><th>基準の時計（秒）</th><th>クラス 1 段あたり（秒）</th><th>前半 1 秒あたり（秒）</th></tr></thead><tbody></tbody></table></div>

<h2>確かめ</h2>
<div class="box">
<b>採否の規則（数字を見る前に固定）</b>: 確かめの期間 2023〜24 で、従来の馬場の速さ（同じ窓 2016〜22 で作り直し・馬場状態の群なし）と、一つ抜きの残差の MAD（小さいほど良い）・同じ日 × 場 × 芝ダの中の一貫性 ICC（大きいほど良い）・土日の開催日レベルの相関（大きいほど良い）で比べ、2 つ以上で良ければ採用。
物差しは途中で替えた: 最初は残差の MAD だったが、測りたい馬場の分と日の分が残る新しい関数に不利で、馬場状態で条件を付けた基準ほど有利になるため（統括役の判断）。第 1 版は 0/3 で不採用、ユーザーの判断で形を細かくした第 2 版を、比べの前に作りを固定して比べ直した。
<table><thead><tr><th>作り（共通の %NCOMMON% レース）</th><th>一つ抜きの MAD（秒）</th><th>ICC</th><th>土日の相関</th></tr></thead><tbody>%CMP%</tbody></table>
<p><b>勝ち方は小さい</b>: 実質の勝ちは一つ抜きの MAD の 0.02 秒だけで、土日の相関の差は偶然の範囲。同じ日の一貫性（ICC）は従来の方が高い。2 着の時計にそろえても ICC の差は縮まない。</p>
<p>単調の破れ（全コース × クラス・ペース・頭数・年齢、各 200 点）: <b>%VIOL%</b> 件。</p>
<p>メンバーレベル（レース − 自分を除いた開催日）の確かめ（2023〜24、次走が同じクラスの出走馬）:</p>
<table><thead><tr><th>メンバーレベルの組</th><th>頭数</th><th>次走の 3 着内率</th><th>着順の偏差（着順÷頭数）</th></tr></thead><tbody>%TERT%</tbody></table>
<p>上位 5 頭の中央値の版と勝ち時計の版の相関: %CORR%（基準 0.7 以上）。</p>
</div>
<script>
const D=%DATA%;const M=D.model;const P=M.params;const T=M.terms;const K=M.degree+1;
function bsplineRow(x,t,k){const n=t.length-k-1;const B=new Array(n).fill(0);
 for(let i=0;i<t.length-1;i++){} // Cox–de Boor
 let N=[];for(let i=0;i<t.length-1;i++)N.push((x>=t[i]&&x<t[i+1])?1:0);
 for(let d=1;d<=k;d++){const Nn=[];for(let i=0;i<t.length-1-d;i++){let a=0,b=0;const d1=t[i+d]-t[i],d2=t[i+d+1]-t[i+1];
  if(d1>0)a=(x-t[i])/d1*N[i];if(d2>0)b=(t[i+d+1]-x)/d2*N[i+1];Nn.push(a+b);}N=Nn;}
 for(let i=0;i<n;i++)B[i]=N[i]||0;return B;}
function ibasis(term,x){const lo=term.lo,hi=term.hi;x=Math.min(Math.max(x,lo),hi-1e-9);
 const t=[];for(let i=0;i<K+1;i++)t.push(lo);term.knots.forEach(v=>t.push(v));for(let i=0;i<K+1;i++)t.push(hi);
 const B=bsplineRow(x,t,K);const I=[];let s=0;for(let i=B.length-1;i>=0;i--){s+=B[i];I[i]=s;}return I.slice(1);}
const CI={};M.courses.forEach((c,i)=>CI[c]=i);
function termEff(c,name,x){const term=T[name];const ci=CI[c];const coef=ci!==undefined?P["B_"+name][ci]:P["th_"+name];const I=ibasis(term,x);
 let s=0;for(let j=0;j<I.length;j++)s+=coef[j]*I[j];return term.sign*s;}
function predict(c,o){const ci=CI[c];const band=M.course_band[c];const bi=M.bands.indexOf(band);
 let v=(bi>=0?P.lvl_band[bi]:0)+(ci!==undefined?P.lvl_dev[ci]:0);
 const ag=(P.B_age&&ci!==undefined)?P.B_age[ci]:P.age;v-=(o.age>=1?ag[0]:0)+(o.age>=2?ag[1]:0);v+=o.newc?P.new[0]:0;
 v+=termEff(c,"cls",o.cls)+termEff(c,"fld",o.fld)+termEff(c,"pace",o.front-M.pace_ref[c]);return v;}
function rq(c){const band=M.course_band[c];return M.resid_q[c]||M.resid_q[band]||M.resid_q._all;}
const $=id=>document.getElementById(id);const fmt=s=>{const m=Math.floor(s/60);return (m?m+":":"")+(s-60*m).toFixed(1).padStart(m?4:0,"0");};
const courses=M.courses.slice();const parse=c=>{const [v,s,d]=c.split("|");return {v,s,d:+d};};
function fill(sel,arr,val){sel.innerHTML=arr.map(a=>`<option value="${a[0]}">${a[1]}</option>`).join("");if(val!==undefined)sel.value=val;}
const cnt={};D.courses.forEach(r=>cnt[r.course]=r.n);
function venues(){return [...new Set(courses.map(c=>parse(c).v))];}
function setCourse(c){const p=parse(c);fill($("venue"),venues().map(v=>[v,v]),p.v);onVenue(p.s,p.d);}
function onVenue(s,d){const v=$("venue").value;const ss=[...new Set(courses.filter(c=>parse(c).v===v).map(c=>parse(c).s))];fill($("surf"),ss.map(x=>[x,x==="ダ"?"ダート":x]),s&&ss.includes(s)?s:ss[0]);onSurf(d);}
function onSurf(d){const v=$("venue").value,s=$("surf").value;const ds=courses.filter(c=>{const p=parse(c);return p.v===v&&p.s===s;}).map(c=>parse(c).d).sort((a,b)=>a-b);
 fill($("dist"),ds.map(x=>[x,`${x}m（${(cnt[`${v}|${s}|${x}`]||0).toLocaleString()} 件）`]),d&&ds.includes(d)?d:ds[0]);onCourse();}
function cur(){return `${$("venue").value}|${$("surf").value}|${$("dist").value}`;}
function onCourse(){const c=cur();const pts=D.points[c]||[];const fr=pts.map(p=>p[4]).sort((a,b)=>a-b);
 const lo=fr.length?fr[Math.floor(fr.length*0.02)]:M.pace_ref[c]-2,hi=fr.length?fr[Math.floor(fr.length*0.98)]:M.pace_ref[c]+2;
 $("front").min=lo.toFixed(1);$("front").max=hi.toFixed(1);$("front").value=M.pace_ref[c].toFixed(1);
 $("cinfo").textContent=`${(cnt[c]||0).toLocaleString()} 件・前半 600m の中央値 ${M.pace_ref[c].toFixed(1)} 秒`;
 try{history.replaceState(null,"","?course="+encodeURIComponent(c));}catch(e){}draw();}
function opts(){return {cls:+$("cls").value,age:+$("age").value,newc:$("newc").checked,fld:+$("fld").value,front:+$("front").value};}
function axis(svg,x0,x1,w,h,pad,ticks){}
function draw(){const c=cur(),o=opts();$("fldv").textContent=o.fld+" 頭";$("frontv").textContent=o.front.toFixed(1)+" 秒";
 const mu=predict(c,o),q=rq(c);const pts=(D.points[c]||[]).filter(p=>Math.abs(p[0]-o.cls)<0.5&&Math.abs(p[3]-o.fld)<=2&&Math.abs(p[4]-o.front)<=0.6&&p[1]===o.age&&(!$("cond").value||p[6]===$("cond").value));
 $("dist_txt").innerHTML=`基準の時計 <b>${fmt(mu)}</b>（${mu.toFixed(2)} 秒）・50% ${fmt(mu+q[2])}〜${fmt(mu+q[4])}・80% ${fmt(mu+q[1])}〜${fmt(mu+q[5])}・95% ${fmt(mu+q[0])}〜${fmt(mu+q[6])}・近い条件の実際のレース ${pts.length} 件（クラス同じ・頭数 ±2・前半 ±0.6 秒・年齢の条件同じ）`;
 const lo=mu+q[0]-1.5,hi=mu+q[6]+1.5,W=900,H=220,L=40,R=20;const X=v=>L+(v-lo)/(hi-lo)*(W-L-R);
 let s="";[[0,6,"--band95"],[1,5,"--band80"],[2,4,"--band50"]].forEach(([a,b,col])=>{s+=`<rect x="${X(mu+q[a])}" y="40" width="${X(mu+q[b])-X(mu+q[a])}" height="110" fill="var(${col})"/>`;});
 s+=`<line x1="${X(mu)}" x2="${X(mu)}" y1="30" y2="160" stroke="var(--accent)" stroke-width="2"/>`;
 pts.forEach((p,i)=>{s+=`<circle cx="${X(p[5])}" cy="${60+((i*37)%80)}" r="3" fill="#fbbf24" opacity=".8"><title>${p[7]} ${p[6]} ${fmt(p[5])}</title></circle>`;});
 for(let v=Math.ceil(lo);v<=hi;v+=1){if((v-Math.ceil(lo))%Math.max(1,Math.round((hi-lo)/10)))continue;s+=`<line x1="${X(v)}" x2="${X(v)}" y1="160" y2="166" stroke="var(--dim)"/><text x="${X(v)}" y="182" fill="var(--dim)" font-size="11" text-anchor="middle">${fmt(v)}</text>`;}
 s+=`<text x="${L}" y="20" fill="var(--dim)" font-size="12">横軸＝走破時計（左が速い）。帯＝95%・80%・50% の区間（残差の分位）、線＝基準の時計、点＝近い条件の実際のレース</text>`;$("distsvg").innerHTML=s;
 small("c_cls","クラス",[1,2,3,4,5,6,7,8],v=>predict(c,{...o,cls:v}),o.cls,v=>D.class_label[v],"cls");
 const fl=[];for(let v=T.fld.lo;v<=T.fld.hi+1e-9;v+=1)fl.push(v);small("c_fld","頭数",fl,v=>predict(c,{...o,fld:v}),o.fld,v=>v+"頭","fld");
 const fr=[];const a=+$("front").min,b=+$("front").max;for(let i=0;i<=30;i++)fr.push(a+(b-a)*i/30);small("c_front","前半 600m（秒）",fr,v=>predict(c,{...o,front:v}),o.front,v=>v.toFixed(1),"pace");
 const B=D.boot;$("bootnote").textContent=B&&B.courses.includes(c)?`灰の帯は 95% 区間（開催日を単位にしたブートストラップ ${B.reps} 回。項の効きの形の幅で、水準は含まない）`:"95% 区間はブートストラップした主な 8 コースだけ";}
function small(id,lab,xs,f,x0,tl,term){const W=320,H=200,L=44,R=10,Tp=22,Bt=30;const ys=xs.map(f);const lo=Math.min(...ys)-0.3,hi=Math.max(...ys)+0.3;
 const X=v=>L+(v-xs[0])/(xs[xs.length-1]-xs[0])*(W-L-R),Y=v=>Tp+(hi-v)/(hi-lo)*(H-Tp-Bt);let s=`<text x="6" y="14" fill="var(--dim)" font-size="11">${lab} と基準の時計（秒）</text>`;
 const B=D.boot,c=cur();if(B&&B.courses.includes(c)&&B.curves[c]&&B.curves[c][term]){const g=B.grids[term],cv=B.curves[c][term];const off=f(x0)-interp(g,cv.mid,term==="pace"?x0-M.pace_ref[c]:x0);
  let up="",dn="";g.forEach((gx,i)=>{const xx=term==="pace"?gx+M.pace_ref[c]:gx;if(xx<xs[0]-1e-9||xx>xs[xs.length-1]+1e-9)return;up+=`${X(xx)},${Y(cv.hi[i]+off)} `;dn=`${X(xx)},${Y(cv.lo[i]+off)} `+dn;});
  if(up)s+=`<polygon points="${up}${dn}" fill="rgba(148,163,184,.25)"/>`;}
 s+=`<polyline fill="none" stroke="var(--accent)" stroke-width="2" points="${xs.map((v,i)=>`${X(v)},${Y(ys[i])}`).join(" ")}"/>`;
 s+=`<circle cx="${X(x0)}" cy="${Y(f(x0))}" r="4" fill="#fbbf24"/>`;
 [xs[0],xs[Math.floor(xs.length/2)],xs[xs.length-1]].forEach(v=>{s+=`<text x="${X(v)}" y="${H-10}" fill="var(--dim)" font-size="10" text-anchor="middle">${tl(v)}</text>`;});
 [lo+0.3,hi-0.3].forEach(v=>{s+=`<text x="${L-4}" y="${Y(v)+4}" fill="var(--dim)" font-size="10" text-anchor="end">${v.toFixed(1)}</text>`;});$(id).innerHTML=s;}
function interp(g,y,x){if(x<=g[0])return y[0];for(let i=1;i<g.length;i++)if(x<=g[i]){const t=(x-g[i-1])/(g[i]-g[i-1]);return y[i-1]+t*(y[i]-y[i-1]);}return y[y.length-1];}
fill($("cls"),Object.entries(D.class_label).map(([k,v])=>[k,v]),"2");fill($("age"),Object.entries(D.age_label).map(([k,v])=>[k,v]),"2");$("fld").min=T.fld.lo;$("fld").max=T.fld.hi;$("fld").value=14;
$("venue").onchange=()=>onVenue();$("surf").onchange=()=>onSurf();$("dist").onchange=onCourse;["cls","age","newc","fld","front","cond"].forEach(k=>$(k).oninput=draw);
const tb=$("ctab").querySelector("tbody");tb.innerHTML=D.courses.slice().sort((a,b)=>b.n-a.n).map(r=>`<tr class="click" data-c="${r.course}"><td>${r.course.replaceAll("|"," ").replace(" ダ "," ダート ")}m</td><td>${r.n.toLocaleString()}</td><td>${fmt(r.base)}</td><td>${r.per_class.toFixed(3)}</td><td>${r.per_front_sec.toFixed(3)}</td></tr>`).join("");
tb.onclick=e=>{const tr=e.target.closest("tr");if(tr){setCourse(tr.dataset.c);window.scrollTo({top:0,behavior:"smooth"});}};
const q=new URLSearchParams(location.search).get("course");setCourse(q&&CI[q]!==undefined?q:D.courses.slice().sort((a,b)=>b.n-a.n)[0].course);
</script>
</main></body></html>
"""

if __name__ == "__main__":
    print(build())
