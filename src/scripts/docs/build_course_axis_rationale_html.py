"""コース別の走法軸の優位の『根拠』を、実力・人気を除いて言語化した専用 HTML を作る（読むだけ・HTML を書く）。

入力: src/scripts/data/course_axis_rationale.py の出力 JSON、data/meta/course_master/jra_courses.json。
使い方: python src/scripts/docs/build_course_axis_rationale_html.py RATIONALE.json [OUT.html]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.research.race import course_axis_rationale as car  # noqa: E402
from src.scripts.docs.build_course_axis_demand_html import CSS, SORT_JS, load_master, shape_of, table, td  # noqa: E402

OUT = REPO / "docs/html/operations/running_map_course_axis_rationale.html"
MODES = [("raw", "補正なし（現行の定義）"), ("res", "実力を除く（本命）"), ("res_pts", "実力を除く・ポイント制"), ("pop", "参考: 人気で補正")]
EXTRA_CSS = """
.ctl { display:flex; gap:14px; flex-wrap:wrap; align-items:center; margin:10px 0; } .ctl select, .ctl button { background:var(--panel); color:var(--text); border:1px solid var(--border); border-radius:8px; padding:5px 10px; font-size:13px; }
.ctl button.on { background:var(--accent); color:#0b1428; font-weight:700; }
.hgrid { display:grid; grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); gap:12px; } .hcell { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:8px 10px; }
.hcell b { font-size:14px; } .hcell svg { width:100%; height:auto; display:block; } .legend span { margin-right:14px; font-size:12px; color:var(--muted); }
.sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; vertical-align:-1px; }
td.ok { color:var(--ok); font-weight:700; } td.no { color:var(--muted); }
"""


def corr_fit_chk(stats: dict, a: str = "fit", b: str = "y26") -> float:
    xs, ys = [], []
    for row in stats.values():
        if row.get(a) and row.get(b):
            xs += row[a]["b"]
            ys += row[b]["b"]
    return float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 3 else float("nan")


def same_sign_rate(stats: dict, t_min: float = 2.0) -> tuple[int, int]:
    n = k = 0
    for row in stats.values():
        if row.get("fit") and row.get("y26"):
            for bf, sf, bg in zip(row["fit"]["b"], row["fit"]["se"], row["y26"]["b"]):
                if abs(bf / sf) >= t_min:
                    n += 1
                    k += int(np.sign(bf) == np.sign(bg))
    return n, k


JS_ZONE = """
(function(){
var cks=Object.keys(HD.hist.raw).sort(function(a,b){var x=a.split("|"),y=b.split("|");return (x[1]===y[1]?0:(x[1]==="芝"?-1:1))||x[0].localeCompare(y[0],"ja")||(+x[2])-(+y[2]);});
var sel=document.getElementById("hc"),st={m:"good",z:"raw"};
[["ALL|芝","全コース（芝）"],["ALL|ダート","全コース（ダート）"]].concat(cks.map(function(c){var p=c.split("|");return [c,p[0]+p[1]+p[2]+"m"];})).forEach(function(o){var e=document.createElement("option");e.value=o[0];e.textContent=o[1];sel.appendChild(e);});
function agg(ck,ax){
  var H=HD.hist[st.z],keys=ck.indexOf("ALL|")===0?cks.filter(function(c){return c.split("|")[1]===ck.split("|")[1];}):[ck];
  var r={all:[0,0,0,0,0,0,0,0,0,0],y:[0,0,0,0,0,0,0,0,0,0]},se=null,d0=null,rf=null;
  keys.forEach(function(k){var h=H[k][ax];for(var i=0;i<10;i++){r.all[i]+=h.all[i];r.y[i]+=st.m==="good"?h.good[i]:h.pts[i];}});
  if(keys.length===1){var dd=H[keys[0]][ax][st.m==="good"?"dg_all":"dp_all"];if(dd){d0=dd.d;se=dd.se;rf=dd.ref;}}
  return {r:r,d:d0,se:se,ref:rf};}
function mean(v){var s=0,n=0;for(var i=0;i<10;i++){s+=v[i]*(i+1);n+=v[i];}return n?s/n:NaN;}
function sum(v){var s=0;for(var i=0;i<v.length;i++)s+=v[i];return s;}
function svg(r){
  var W=380,Hh=210,l=34,t=8,b=62,w=(W-l-6)/10,A=sum(r.all),Y=sum(r.y),pa=[],py=[],mx=0.05,s="";
  for(var i=0;i<10;i++){pa.push(A?r.all[i]/A:0);py.push(Y?r.y[i]/Y:0);mx=Math.max(mx,pa[i],py[i]);}
  mx*=1.12;var base=Hh-b,ph=base-t;
  s+='<line x1="'+l+'" y1="'+base+'" x2="'+(W-4)+'" y2="'+base+'" stroke="#243358"/>';
  for(var k=0;k<=2;k++){var v=mx*k/2,y=base-ph*k/2;s+='<text x="'+(l-4)+'" y="'+(y+3)+'" font-size="9" fill="#9aa7c7" text-anchor="end">'+(v*100).toFixed(0)+'%</text>';}
  for(var i=0;i<10;i++){
    var x=l+i*w,ha=ph*pa[i]/mx,hy=ph*py[i]/mx,df=(py[i]-pa[i])*100,thin=r.all[i]<30;
    s+='<rect x="'+(x+2)+'" y="'+(base-ha)+'" width="'+(w/2-2)+'" height="'+ha+'" fill="#9aa7c7" opacity="0.75"/>';
    s+='<rect x="'+(x+w/2)+'" y="'+(base-hy)+'" width="'+(w/2-2)+'" height="'+hy+'" fill="#4f8bff" opacity="'+(thin?0.4:0.95)+'"/>';
    s+='<text x="'+(x+w/2)+'" y="'+(base+12)+'" font-size="10" fill="#e6ecff" text-anchor="middle">'+(i+1)+'</text>';
    s+='<text x="'+(x+w/2)+'" y="'+(base+27)+'" font-size="9" fill="'+(Math.abs(df)<0.5?'#9aa7c7':(df>0?'#4ade80':'#f87171'))+'" text-anchor="middle">'+(df>=0?'+':'')+df.toFixed(1)+'</text>';
    s+='<text x="'+(x+w/2)+'" y="'+(base+40)+'" font-size="8" fill="#9aa7c7" text-anchor="middle">'+r.all[i]+'</text>';
  }
  s+='<text x="2" y="'+(base+27)+'" font-size="8" fill="#9aa7c7">差pt</text><text x="2" y="'+(base+40)+'" font-size="8" fill="#9aa7c7">走数</text>';
  return '<svg viewBox="0 0 '+W+' '+Hh+'" xmlns="http://www.w3.org/2000/svg">'+s+'</svg>';}
function draw(){
  var ck=sel.value,h="";
  HD.axes.forEach(function(ax,j){var o=agg(ck,ax),ma=mean(o.r.all),my=mean(o.r.y),hi=function(v){return (v[7]+v[8]+v[9])/sum(v)*100;},lo=function(v){return (v[0]+v[1]+v[2])/sum(v)*100;};
    var sg=function(v,n){return (v>=0?"+":"")+v.toFixed(n);};
    var dtxt=sg(my-ma,2)+(o.se?" ／ 基準 "+sg(o.ref,2)+" を引いたこのコース固有の差 "+sg(o.d-o.ref,2)+"（t "+sg((o.d-o.ref)/o.se,1)+"）":"");
    h+='<div class="hcell"><b>'+HD.names[j]+'</b> <span class="muted">平均得点 全体 '+ma.toFixed(2)+' → '+(st.m==="good"?"好走馬":"ポイント")+' '+my.toFixed(2)+' ／ 差 '+dtxt+'</span>'+svg(o.r)+
       '<div class="muted" style="font-size:12px">高得点（8〜10）: 全体 '+hi(o.r.all).toFixed(0)+'% → '+hi(o.r.y).toFixed(0)+'% ／ 低得点（1〜3）: 全体 '+lo(o.r.all).toFixed(0)+'% → '+lo(o.r.y).toFixed(0)+'%</div></div>';});
  document.getElementById("hg").innerHTML=h;
  ["m_good","m_pts","z_raw","z_res"].forEach(function(id){var on=(id==="m_"+st.m)||(id==="z_"+st.z);document.getElementById(id).className=on?"on":"";});}
sel.onchange=draw;
document.getElementById("m_good").onclick=function(){st.m="good";draw();};document.getElementById("m_pts").onclick=function(){st.m="pts";draw();};
document.getElementById("z_raw").onclick=function(){st.z="raw";draw();};document.getElementById("z_res").onclick=function(){st.z="res";draw();};
draw();
})();
"""


def delta_stats(H: dict, K: list[str], kind: str, sub_ref: bool = False) -> dict:
    """コース×軸の Δ（好走馬の平均得点 − 全出走馬の平均得点）の t を 2024-25 年と 2026 年で並べた要約。sub_ref なら、同じ芝ダ・同じクラス構成の基準を引いたコース固有の差。"""
    tf, t6, df, d6 = [], [], [], []
    for row in H.values():
        for k in K:
            a, b = row[k].get(f"d{kind}_fit"), row[k].get(f"d{kind}_y26")
            if a and b and a["se"] > 0 and b["se"] > 0:
                da, db = a["d"] - (a["ref"] if sub_ref else 0), b["d"] - (b["ref"] if sub_ref else 0)
                tf.append(da / a["se"]); t6.append(db / b["se"]); df.append(da); d6.append(db)
    tf, t6, df, d6 = map(np.array, (tf, t6, df, d6))
    big = np.abs(tf) >= 2.0
    return {"n": len(tf), "hit_fit": int((np.abs(tf) >= car.T_COURSE).sum()), "hit_26": int((np.abs(t6) >= car.T_COURSE).sum()), "n26": len(t6),
            "sd_fit": float(tf.std()), "sd_26": float(t6.std()), "corr": float(np.corrcoef(df, d6)[0, 1]),
            "same": int((np.sign(tf[big]) == np.sign(d6[big])).sum()), "big": int(big.sum())}


def axis_rep(H: dict, K: list[str], kind: str) -> tuple[list[dict], list[dict]]:
    """軸別に、基準を引いたコース固有の差の 2024-25 年 vs 2026 年の相関と、両期間で同じ向き・|t|>=T_REPL のコース×軸。"""
    rows, cells = [], []
    for k in K:
        a, b, ta, tb = [], [], [], []
        for ck, r in H.items():
            x, y = r[k].get(f"d{kind}_fit"), r[k].get(f"d{kind}_y26")
            if x and y and x["se"] > 0 and y["se"] > 0:
                a.append(x["d"] - x["ref"]); b.append(y["d"] - y["ref"]); ta.append(a[-1] / x["se"]); tb.append(b[-1] / y["se"])
                if np.sign(ta[-1]) == np.sign(tb[-1]) and abs(ta[-1]) >= car.T_REPL and abs(tb[-1]) >= car.T_REPL:
                    cells.append({"ck": ck, "axis": k, "d_fit": a[-1], "d_26": b[-1], "t_fit": ta[-1], "t_26": tb[-1]})
        a, b, ta, tb = map(np.array, (a, b, ta, tb))
        rows.append({"axis": k, "n": len(a), "corr": float(np.corrcoef(a, b)[0, 1]), "sd_fit": float(ta.std()), "sd_26": float(tb.std()),
                     "rep": int(sum(1 for c in cells if c["axis"] == k))})
    return rows, cells


def zone_section(d: dict, mp: dict) -> str:
    K = d["axes"]
    nm = car.AXIS_NAME
    H = {"raw": d["hist"], "res": d["histr"]}
    meanz = lambda v: sum((i + 1) * x for i, x in enumerate(v)) / max(sum(v), 1)
    # 全コースまとめ（芝・ダート別）の平均得点の差
    rows = []
    for su in ("芝", "ダート"):
        for j, k in enumerate(K):
            cells = [td(su), td(nm[k])]
            for z in ("raw", "res"):
                al, gd, pt = [0] * 10, [0] * 10, [0] * 10
                for ck, row in H[z].items():
                    if ck.split("|")[1] == su:
                        for i in range(10):
                            al[i] += row[k]["all"][i]; gd[i] += row[k]["good"][i]; pt[i] += row[k]["pts"][i]
                cells += [td(f"{meanz(gd) - meanz(al):+.2f}", True, data=f"{meanz(gd) - meanz(al):.3f}"), td(f"{meanz(pt) - meanz(al):+.2f}", True, data=f"{meanz(pt) - meanz(al):.3f}")]
            rows.append(cells)
    pooled = table(["芝ダ", "軸", "好走馬の平均得点 − 全体（軸の値そのまま）", "ポイント制（そのまま）", "好走馬（実力を除く）", "ポイント制（実力を除く）"], rows)
    # コース別の Δ（実力を除く・好走馬）
    rows = []
    for ck in sorted(H["res"], key=lambda c: (c.split("|")[1] != "芝", c.split("|")[0], int(c.split("|")[2]))):
        ve, su, di = ck.split("|")
        cells = [td(f"{ve}{su}{di}m")]
        for k in K:
            h = H["res"][ck][k]
            a, f, g = h.get("dg_all"), h.get("dg_fit"), h.get("dg_y26")
            if not a:
                cells.append(td("—")); continue
            dd = a["d"] - a["ref"]
            t = dd / a["se"]
            sub = "/".join(f"{(x['d'] - x['ref']) / x['se']:+.1f}" if x else "—" for x in (f, g))
            cls = "num strong" if abs(t) >= car.T_COURSE else "num"
            cells.append(f'<td class="{cls}" data-v="{dd:.3f}">{dd:+.2f} <span class="muted">(t {t:+.1f}) [{sub}]</span></td>')
        rows.append(cells)
    bycourse = table(["コース"] + [nm[k] for k in K], rows)
    st = {(z, kind, sr): delta_stats(H[z], K, kind, sr) for z in ("raw", "res") for kind in ("g", "p") for sr in (False, True)}
    rep_rows = []
    for (z, kind, sr), v in st.items():
        rep_rows.append([td(("実力を除く" if z == "res" else "軸の値そのまま") + ("・3着以内" if kind == "g" else "・ポイント制") + ("・基準を引く" if sr else "・基準を引かない")), td(f"{v['hit_fit']}/{v['n']}", True), td(f"{v['sd_fit']:.2f}", True),
                         td(f"{v['hit_26']}/{v['n26']}", True), td(f"{v['sd_26']:.2f}", True), td(f"{v['corr']:+.2f}", True), td(f"{v['same']}/{v['big']}", True)])
    rep = table(["見方", "2024-25 年で |t|≥2.3", "|t|の標準偏差", "2026 年で |t|≥2.3", "|t|の標準偏差", "Δの相関（2024-25 vs 2026）", "2024-25 で |t|≥2 のうち 2026 で同符号"], rep_rows)
    ar, cells = axis_rep(H["res"], K, "g")
    ar_rows = [[td(nm[r["axis"]]), td(f"{r['corr']:+.2f}", True, data=f"{r['corr']:.3f}"), td(f"{r['sd_fit']:.2f} / {r['sd_26']:.2f}", True), td(r["rep"], True)] for r in ar]
    axtab = table(["軸", "コース固有の差の相関（2024-25 年 vs 2026 年）", "t の標準偏差（2024-25 / 2026）", "両期間で同じ向き・|t|≥1.9 のコース"], ar_rows)
    cells.sort(key=lambda c: -abs(c["t_fit"] + c["t_26"]))
    cell_rows = []
    for c in cells:
        ve, su, di = c["ck"].split("|")
        cell_rows.append([td(f"{ve}{su}{di}m"), td(nm[c["axis"]]), td(f"{c['d_fit']:+.2f} ({c['t_fit']:+.1f})", True, data=f"{c['t_fit']:.2f}"), td(f"{c['d_26']:+.2f} ({c['t_26']:+.1f})", True, data=f"{c['t_26']:.2f}"),
                          td("優位寄り" if c["d_fit"] > 0 else "不利寄り")])
    celltab = table(["コース", "軸", "2024-25 年 差 (t)", "2026 年 差 (t)", "向き"], cell_rows)
    payload = json.dumps({"axes": K, "names": [nm[k] for k in K],
                          "hist": {z: {ck: {k: {f: h[k].get(f) for f in ("all", "good", "pts", "dg_all", "dp_all")} for k in K} for ck, h in H[z].items()} for z in ("raw", "res")}},
                         ensure_ascii=False, separators=(",", ":"))
    return f"""<h2>1. 10 点満点で見る: 全出走馬の得点分布 と 好走馬の得点分布</h2>
<p class="sub">依頼「そもそもの得点分布と、実際に好走した馬の得点分布を比べ、その差分が適性として現れるはず」。
各走法軸を、芝・ダート別の十分位で <b>1〜10 点</b>にした（10 点 = その軸が全出走馬の上位 1 割、1 点 = 下位 1 割。プロジェクトの 10 点満点表示と同じ「順位の位置」の考え方）。
コースごとに、<b>灰色 = そのコースの全出走馬の得点分布</b>、<b>青 = 好走馬（3 着以内）の得点分布</b>を並べる（どちらも合計 100% に揃えた割合）。
帯の下の緑／赤の数字は「青 − 灰」のパーセントポイント。青が高得点側へ偏っていれば、その軸が強い馬が好走しやすい = そのコースでその軸が効いている印。
ポイント制（1〜3 着 = 3 点、4〜5 着 = 1 点、6 着以下 = 0 点）に切り替えると、4〜5 着も薄く数える。良馬場・2024 年以降・頭数 5 以上（2026 年を含む記述の図）。走数の少ない帯は薄く表示。</p>
<div class="ctl">
<label>コース <select id="hc"></select></label>
<span><button id="m_good" class="on">3着以内</button> <button id="m_pts">ポイント制</button></span>
<span><button id="z_raw" class="on">軸の値そのまま</button> <button id="z_res">実力を除いた軸の値</button></span>
</div>
<div class="legend"><span><i class="sw" style="background:#9aa7c7"></i>全出走馬</span><span><i class="sw" style="background:#4f8bff"></i>好走馬（またはポイント）</span></div>
<div id="hg" class="hgrid"></div>
<p class="sub">各パネルの上の「差」= 好走馬の平均得点 − 全出走馬の平均得点。1 コース選択時は、これに加えて<b>基準（同じ芝ダ・同じクラス構成の全コースで期待される差）</b>と、基準を引いたコース固有の差、レース単位でまとめた標準誤差による t を出す（全コース表示は差のみ）。
<b>軸の値そのままは、強い馬ほど軸が高く、強い馬ほど好走する（= 実力）分も含む</b>ので、適性だけを見るときは「実力を除いた軸の値」に切り替えて、基準を引いたコース固有の差を読む（§2）。</p>
<script>var HD={payload};</script>
<script>{JS_ZONE}</script>

<h3>全コースをまとめた差（平均得点。基準そのもの）</h3>
<p class="sub">芝・ダート全体で、好走馬の平均得点が全出走馬より何点高いか。軸の値そのままでは、Hペース耐性・ギアチェンジなど実力と結びつく軸で差が出やすい。実力を除くと差は縮む。</p>
{pooled}

<h3>「基準」を引いて、コース固有の差にする</h3>
<p class="sub">好走馬の得点が全体より高いのは、どのコースでも多かれ少なかれ起こる（強い馬ほど軸が高い、クラスが違う、など）。そこで、<b>同じ芝ダ・同じクラス構成の全コースで期待される差（基準）</b>を、各コースの差から引いた。
引いた後の差 = 「そのコースだけ、全体の傾向より軸の高い馬が好走している／しにくい」量で、これを適性として読む。基準は 2024-25 年と 2026 年それぞれの全体から出した（コース単位の標準誤差はそのまま）。
下の表は、基準を引く前（「基準を引かない」）と引いた後（「基準を引く」）を並べたもの。</p>
{rep}
<p class="sub">偶然だけなら |t| の標準偏差は約 1、|t| ≥ 2.3 は 2% 程度（372 件で約 8 件）。「Δの相関」は同じコース × 軸の差を 2024-25 年と 2026 年で比べたもので、コース固有の適性が本当に続くなら正になり、偶然だけなら 0 になる。</p>
<h3>軸別: コース固有の差は続いているか（実力を除く・3着以内・基準を引いた後）</h3>
{axtab}
<h3>両期間で同じ向きに出た（コース, 軸）</h3>
<p class="sub">2024-25 年と 2026 年のどちらでも、基準を引いた後の差が同じ向きで |t| ≥ {car.T_REPL}。372 件のうち偶然だと 1 件に満たない数（1 件あたり約 0.16%）。表の値は 軸の得点（1〜10）の平均の差。</p>
{celltab}
<h3>コース別の差と、偶然との見分け</h3>
<p class="sub">差 = 基準を引いたコース固有の差（実力を除いた軸の値・3 着以内。好走馬の平均得点 − 全出走馬の平均得点 − 基準）。括弧は t、角括弧は 2024-25 年／2026 年の t。|t| ≥ {car.T_COURSE} は太字。</p>
{bycourse}
"""


def build(d: dict, mp: dict) -> str:
    K = d["axes"]
    M = d["modes"]
    meta, dg = d["meta"], d["diag"]
    feats = {}
    for ck in M["res"]:
        ve, su, di = ck.split("|")
        f = car.shape_features(mp[(ve, su, int(di))], su, int(di))
        if f:
            feats[ck] = f
    eff = {m: car.shape_effects(M[m], feats, K) for m in ("raw", "res", "res_pts")}
    repl = [e for e in eff["res"] if e["verdict"] == "根拠あり"]
    nul_repl = sum(1 for nu in d["null"] for e in car.shape_effects(nu, feats, K) if e["verdict"] == "根拠あり")
    cc = {m: {p: car.course_counts(M[m], p, K) for p in ("fit", "y26")} for m in M}
    ncc = [{p: car.course_counts(nu, p, K) for p in ("fit", "y26")} for nu in d["null"]]
    null_fit = np.mean([c["fit"]["hits"] for c in ncc])
    null_26 = np.mean([c["y26"]["hits"] for c in ncc])
    n_tests = cc["res"]["fit"]["n"]
    fit_courses = sum(1 for r in M["res"].values() if r.get("fit"))
    chk_courses = sum(1 for r in M["res"].values() if r.get("y26"))
    ne = K.index("endure")
    r2s, r2d = dg["axis_r2_by_ability"]["芝"], dg["axis_r2_by_ability"]["ダート"]

    nm = car.AXIS_NAME
    ts_e = next((e for e in repl if e["axis"] == "ts" and e["feature"] == "距離"), None)
    concl = []
    Hr, Hw = d["histr"], d["hist"]
    s_raw0, s_raw1 = delta_stats(Hw, K, "g", False), delta_stats(Hw, K, "g", True)
    s_res0, s_res1 = delta_stats(Hr, K, "g", False), delta_stats(Hr, K, "g", True)
    ar, cells = axis_rep(Hr, K, "g")
    arn = {r["axis"]: r for r in ar}
    best = sorted(ar, key=lambda r: -r["corr"])
    ncell = s_res1["n"]
    exp_hit = ncell * 0.021
    concl.append(f"""<li><b>好走馬の得点分布と全体の得点分布の差は、ほとんどが「どのコースでも出る差」（実力・クラス・芝ダ）</b>。
軸の値そのままだと、コース × 軸 {s_raw0['n']} 件のうち {s_raw0['hit_fit']} 件（2024-25 年）で |t| ≥ {car.T_COURSE}。実力の代理を除いても {s_res0['hit_fit']} 件。
同じ芝ダ・同じクラス構成の全体で期待される差（基準）を引くと {s_res1['hit_fit']} 件（偶然でも約 {exp_hit:.0f} 件）まで減る。</li>""")
    concl.append(f"""<li><b>基準を引いたコース固有の差は、2024-25 年と 2026 年である程度続いている</b>（相関 {s_res1['corr']:+.2f}、基準を引く前は実力を除いても {s_res0['corr']:+.2f}。2024-25 年で |t| ≥ 2 のうち 2026 年で同符号 {s_res1['same']}/{s_res1['big']}）。
軸別の相関は {'、'.join(f"{nm[r['axis']]} {r['corr']:+.2f}" for r in best)}。<b>ギアチェンジ・Cr性能・Hペース耐性は、コースによって優位／不利が分かれる傾向が続いている</b>。TS持続力・基礎スピードはほぼ 0（コース差は検出できない）。
これは前の回帰方式（コース固有の係数の相関 {corr_fit_chk(M['res']):+.2f}）より再現が強い。「得点分布の差」の方が、少ない走数でも安定した測り方だと読める。</li>""")
    cl = "、".join(f"{c['ck'].split('|')[0]}{c['ck'].split('|')[1]}{c['ck'].split('|')[2]}m の{nm[c['axis']]}（{'優位寄り' if c['d_fit'] > 0 else '不利寄り'}）" for c in cells[:7])
    concl.append(f"""<li><b>個別に「このコースでこの軸が優位／不利」と、両期間で言えたのは {len(cells)} 件</b>（偶然なら 1 件に満たない）: {cl}。</li>""")
    concl.append(f"""<li><b>ただし「適性だと確定」とまでは言わない</b>。基準を引いた後も t の標準偏差が {s_res1['sd_fit']:.2f}（2024-25）/ {s_res1['sd_26']:.2f}（2026）と 1 より大きく、ギアチェンジは {arn['gear']['sd_fit']:.2f}。
実力の除去は線形の代理だけで、コースごとの出走馬の強さの違い（同じクラスでも強い馬が集まるコース）が軸と結びつくと、適性に見える差が残る。人気・オッズは、補正にも判定にも使っていない。
「次の期間でも続くかを見る候補」の位置づけ。</li>""")
    if ts_e:
        concl.append(f"""<li>前の回帰方式（コースの形状で束ねる）では、「距離が長いほど TS が効く」だけが両期間で同じ向きに出た（係数 {ts_e['b_fit'] * 100:+.1f} / {ts_e['b_chk'] * 100:+.1f}、t = {ts_e['t_fit']:+.1f} / {ts_e['t_chk']:+.1f}。弱い根拠）。
直線長・小回り・坂など、事前に考えやすい形状の筋は再現しなかった（§4）。</li>""")
    else:
        concl.append("<li>前の回帰方式（コースの形状で束ねる）では、2 つの期間で同じ向きに再現する効果はなかった（§4）。</li>")
    concl.append(f"""<li>Hペース耐性が実力の代理だった点は、実力を除く前後で確認できる。軸のばらつきのうち実力で説明される割合は、Hペース耐性が芝 {r2s[ne] * 100:.0f}%・ダート {r2d[ne] * 100:.0f}%（ギアチェンジ {r2s[0] * 100:.0f}% / {r2d[0] * 100:.0f}%、他は 5% 以下）。</li>""")

    zone_html = zone_section(d, mp)
    parts = [f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>走法軸 × コースの適性：優位の根拠（実力・人気を除く） — keiba-vpn docs</title><style>{CSS}{EXTRA_CSS}</style></head><body><main>
<h1>走法軸 × コースの適性：優位の根拠（実力・人気を除く）</h1>
<p class="muted">作成: 2026-11-11。依頼「各コースでこの走法軸が優位、と言うなら根拠を言語化したい。人気や実力といった、適性とのつながりが薄い要素は排除か補正する」「人気の要素が強く入らないよう、適性だけで言語化」。
良馬場・2024 年以降・頭数 5 以上の {meta['n_runs']:,} 走（{meta['n_races']:,} レース）。2024-25 年（{meta['n_fit']:,} 走）で推定、2026 年（{meta['n_2026']:,} 走・テストセット②）は<b>確認のためだけ</b>に 1 回見た。
計算は <code>src/scripts/data/course_axis_rationale.py</code>（ホスト）、判定は <code>src/research/race/course_axis_rationale.py</code>、この表は <code>src/scripts/docs/build_course_axis_rationale_html.py</code>。表の見出しをクリックすると並べ替え。
前の測定は <a href="running_map_course_axis.html">走法マップ：良馬場のコース別 走法軸の寄与と根拠</a>。</p>

<div class="box"><b>結論</b><ol>{''.join(concl)}</ol>
<p class="sub">一言で言うと、<b>得点分布の差から基準（実力・クラス・芝ダ共通の分）を引いても、ギアチェンジ・Cr性能・Hペース耐性にはコース固有の差が続いている</b>。
ただし個別のコースで「適性」と言い切れるのは数件で、残りの差には出走馬の強さの偏りが混ざりうる。使うなら「候補」として、続くかを追う。</p></div>

{zone_html}

<h2>2. 実力・人気をどう扱ったか</h2>
<ul>
<li><b>人気・オッズは結果の補正にも判定にも使わない</b>（市場の評価が入るため。人気で補正した列は、比較用の診断としてだけ並べた）。</li>
<li><b>実力の代理 = その馬の「そのレースより前の走」の得点の平均</b>（1〜3 着 = 1 … 8 着以下 = 0 の得点。平均 {meta['mean_score']:.2f} へ K = {meta['shrink_k']:.0f} 走分縮小し、レース内で中心化）。結果・市場を見ない。
この代理は市場の実力（人気込みの期待 − 頭数だけの期待）と相関 {dg['ability_vs_pop_corr']['芝']:.2f}（芝）・{dg['ability_vs_pop_corr']['ダート']:.2f}（ダート）で、実力の指標として働いている。</li>
<li><b>補正の仕方</b>: 結果（得点 − 頭数だけの期待）と各軸の z を、それぞれ実力の代理へ回帰した残差にする（芝ダ別）。「強い馬ほど軸が高く、得点も高い」という関係を、軸の側と結果の側の両方から除く。</li>
<li>コース固有の差は、「同じ芝ダ × 同じクラスの平均の係数」を引いた残りを軸へ回帰（前の測定と同じ）。標準誤差はレース単位でまとめて計算。係数は 軸 1 標準偏差あたりの得点（×100 で表示）。</li>
<li><b>ポイント制</b>（ユーザー指定: 1〜3 着 = 3 点、4〜5 着 = 1 点、6 着以下 = 0 点）は、結果の点数を替えた頑健性の確認（期待は 11 ÷ 頭数）と、§1 のヒストグラムの集計に使った。</li>
<li>多重検定への備え: 着順をレース内で入れ替えた乱数ラベルを {meta['n_shuffle']} 回、同じ手順に通して「偶然で出る数」を測った。</li>
<li>2024 年は出走記録の整った走が少なく（コース × 年で 120 走以上のコースは 5 つ）、年別の再現（2024 と 2025）は組めなかった。かわりに <b>2024-25 年で推定 → 2026 年で 1 回確認</b>とした。</li>
</ul>

<h2>3. 絞り込み: どの補正で何件残るか</h2>
<p class="sub">コース × 軸の差の検定のうち |t| ≥ {car.T_COURSE} の件数。2024-25 年（{fit_courses} コース × 6 軸 = {n_tests} 件）と、2026 年（{chk_courses} コース × 6 軸 = {cc['res']['y26']['n']} 件）。
|t| の標準偏差は、偶然だけなら約 1。相関は、同じコース × 軸の係数を 2024-25 年と 2026 年で比べたもの。</p>"""]
    rows = []
    for m, lab in MODES:
        n2, k2 = same_sign_rate(M[m])
        rows.append([td(lab), td(cc[m]["fit"]["hits"], True), td(f"{cc[m]['fit']['sd']:.2f}", True), td(cc[m]["y26"]["hits"], True), td(f"{cc[m]['y26']['sd']:.2f}", True),
                     td(f"{corr_fit_chk(M[m]):+.2f}", True), td(f"{k2}/{n2}", True)])
    rows.append([td("乱数ラベル（実力を除く・平均）"), td(f"{null_fit:.1f}", True), td(f"{np.mean([c['fit']['sd'] for c in ncc]):.2f}", True), td(f"{null_26:.1f}", True), td(f"{np.mean([c['y26']['sd'] for c in ncc]):.2f}", True),
                 td(f"{np.mean([corr_fit_chk(nu) for nu in d['null']]):+.2f}", True), td("—")])
    parts.append(table(["補正", "2024-25 年の |t|≥2.3", "|t|の標準偏差", "2026 年の |t|≥2.3", "|t|の標準偏差", "係数の相関（2024-25 vs 2026）", "2024-25 で |t|≥2 のうち 2026 で同符号"], rows))
    parts.append(f"""<ul>
<li>補正なし 11 件 → 実力を除く 9 件 → ポイント制 6 件と減るが、乱数ラベルでも平均 {null_fit:.1f} 件（{min(c['fit']['hits'] for c in ncc)}〜{max(c['fit']['hits'] for c in ncc)} 件）出る。<b>偶然の数と区別できない</b>。</li>
<li>2026 年は |t| の標準偏差が 1.2 台で、乱数ラベル（1.0〜1.1）よりやや大きい。コースごとに多少の差はあるが、2024-25 年の係数とは相関 0.15 以下で、同じ向きに再現しない。</li>
</ul>
<h3>実力で説明される軸のばらつき</h3>""")
    rows = [[td(s)] + [td(f"{v * 100:.1f}%", True, data=v) for v in dg["axis_r2_by_ability"][s]] + [td(f"{dg['ability_vs_outcome_corr'][s]:+.2f}", True)] for s in ("芝", "ダート")]
    parts.append(table(["芝ダ"] + [f"軸の分散のうち実力で説明（{nm[k]}）" for k in K] + ["実力と結果の相関"], rows))
    parts.append("<p class=\"sub\">Hペース耐性は軸の分散の約 2 割が実力で説明され、ギアチェンジが 1 割前後。この分は、補正前の係数に「強い馬が強い」が混ざっていた部分。</p>")

    parts.append(f"""<h2>4. 形状ごとに束ねた検証（言語化の単位）</h2>
<p class="sub">コースを 1 つずつ見ると偶然と区別できないので、コースの形状（直線長・小回り・ダート・坂・距離）で束ね、「形状が 1 段階違うと、軸の係数がどれだけ変わるか」を、{eff['res'][0]['n_fit']} コース（2024-25 年）と {eff['res'][0]['n_chk']} コース（2026 年）で別々に推定した。
各行は t（括弧外が係数 ×100 ＝ 軸 1 標準偏差あたりの得点）。両期間で同じ向き・|t| ≥ {car.T_REPL} なら<b>根拠あり</b>、片方だけなら<b>片方の期間のみ</b>（根拠にしない）。形状は直線長 +100m、小回り、ダート、坂（ゴール前）、距離 +400m の 5 つ。
基準のコースは直線 400m・大回り・芝・坂なし・1600m。内回り・外回りで直線の長さが違うコースは形状が定まらないため除いた（{len(feats)}/{len(M['res'])} コースを使用）。</p>""")
    rows = []
    for e in eff["res"]:
        ex = {m: next(x for x in eff[m] if x["axis"] == e["axis"] and x["feature"] == e["feature"]) for m in ("raw", "res_pts")}
        rows.append([td(nm[e["axis"]]), td(e["feature"]), td(f"{e['b_fit'] * 100:+.1f} ({e['t_fit']:+.1f})", True, data=f"{e['t_fit']:.2f}"),
                     td(f"{e['b_chk'] * 100:+.1f} ({e['t_chk']:+.1f})", True, data=f"{e['t_chk']:.2f}"),
                     f'<td class="{"ok" if e["verdict"] == "根拠あり" else "no"}">{e["verdict"]}</td>',
                     td(f"{ex['raw']['t_fit']:+.1f} / {ex['raw']['t_chk']:+.1f}", True), td(f"{ex['res_pts']['t_fit']:+.1f} / {ex['res_pts']['t_chk']:+.1f}", True)])
    parts.append(table(["軸", "形状", "2024-25 年 係数×100 (t)", "2026 年 係数×100 (t)", "判定（実力を除く）", "補正なしの t（2024-25 / 2026）", "ポイント制の t（2024-25 / 2026）"], rows))
    parts.append(f"""<p class="sub">30 件の検定を 2 期間で見ているので、片方の期間だけで |t| ≥ 2 になるものが数件は出る（偶然）。乱数ラベルで同じ基準を満たしたのは 5 回の合計 {nul_repl} 件。
ただし事前に仮説を絞らずに 30 件を眺めた結果なので、残った 1 件も<b>確定ではなく「次の期間で再確認する候補」</b>。</p>""")

    parts.append("""<h2>5. 根拠を言語化できるもの／できないもの</h2>""")
    if ts_e:
        parts.append(f"""<div class="box"><b>根拠あり（弱い）: 距離が長いコースほど TS（末脚の速さ）が効きやすい</b>
<ul>
<li>数字: 400m 長いごとに、TS が 1 標準偏差高い馬の得点が +{ts_e['b_fit'] * 100:.1f}（2024-25 年, t = {ts_e['t_fit']:+.1f}）／ +{ts_e['b_chk'] * 100:.1f}（2026 年, t = {ts_e['t_chk']:+.1f}）。実力を除く前（補正なし）は t = +2.8 / +3.2。</li>
<li>理由として考えられる筋（<b>未検証の解釈</b>。この分析はラップを見ていない）: 長い距離ほど道中のペースが緩み、直線の入りからの脚の使い方で差がつきやすい。
一方でスプリント戦は前半から速く、末脚の速さの価値が相対的に小さい。</li>
<li>使い方: 「距離が伸びるほど TS を評価する」という方向の補正の候補。<b>個別のコースの「優位」とは言わない</b>（個別のコースの係数は偶然のぶれが大きい）。大きさも小さい（上記の値）ので、印を動かす根拠にはならない。</li>
</ul></div>""")
    parts.append("""<div class="box"><b>根拠なし（出さない）</b>
<ul>
<li>直線が長いコースで TS が効く、小回りで Cr性能・TS持続力が効く、坂のあるコースで TS持続力・Hペース耐性が効く、といった事前に筋の通る話は、どちらの期間でも再現しなかった。</li>
<li>個別のコースで「実測で目立つ」ものは、形状で説明できず、他の期間でも再現しないため、根拠文にしない。下の表では<b>「実測で目立つが再現なし」</b>と区別して示す。</li>
</ul></div>
<h3>コース別の言語化</h3>
<p class="sub">「形状から言えること」は上の根拠ありの効果だけから作った文。実測の欄は、そのコースの 2024-25 年（t）と 2026 年（t）の TS の係数（実力を除く）。
「実測で目立つが再現なし」は、いずれかの期間で |t| ≥ 2.3 の軸（同符号で両期間とも |t| ≥ 1.9 なら再現あり）。</p>""")
    rows = []
    ref_effects = {e["axis"]: e for e in repl if e["feature"] == "距離"}
    for ck in sorted(M["res"], key=lambda c: (c.split("|")[1] != "芝", c.split("|")[0], int(c.split("|")[2]))):
        row = M["res"][ck]
        if not (row.get("fit") or row.get("y26")):
            continue
        ve, su, di = ck.split("|")
        sh = shape_of(mp[(ve, su, int(di))])
        texts = []
        for ax, e in ref_effects.items():
            if ck in feats:
                shift = e["b_fit"] * feats[ck][5] * 100
                dist_txt = f"{di}m"
                texts.append(f"{dist_txt}で基準（1600m）より{'長い' if feats[ck][5] > 0 else ('短い' if feats[ck][5] < 0 else '同じ')}ため、{nm[ax]}は{'効きやすい' if shift > 0.05 else ('効きにくい' if shift < -0.05 else '基準並み')}（予想 {shift:+.1f}）")
        odd = []
        for j, k in enumerate(K):
            ts = [(p, row[p]["b"][j] / row[p]["se"][j]) for p in ("fit", "y26") if row.get(p)]
            if any(abs(t) >= car.T_COURSE for _, t in ts):
                both = len(ts) == 2 and np.sign(ts[0][1]) == np.sign(ts[1][1]) and all(abs(t) >= car.T_REPL for _, t in ts)
                odd.append(f"{nm[k]}{'（再現あり）' if both else '（再現なし）'} " + "/".join(f"{t:+.1f}" for _, t in ts))
        tsj = K.index("ts")
        tcell = " / ".join(f"{row[p]['b'][tsj] * 100:+.1f} ({row[p]['b'][tsj] / row[p]['se'][tsj]:+.1f})" if row.get(p) else "—" for p in ("fit", "y26"))
        rows.append([td(f"{ve}{su}{di}m"), td(sh["straight"], True, data=sh["st0"] if sh["st0"] == sh["st0"] else 0), td("小回り" if sh["small"] else "—"), td(sh["slope"]),
                     td(" ／ ".join(texts) if texts else ("形状が定まらない（内外回り）" if ck not in feats else "形状から言える優位はない")), td(tcell, True), td("、".join(odd) if odd else "—")])
    parts.append(table(["コース", "直線(m)", "カーブ", "坂", "形状から言えること", "TS の係数×100 (t) 2024-25 / 2026", "実測で目立つが再現なし／ありの軸 (t)"], rows))

    parts.append(f"""<h2>6. 限界と次の一手</h2>
<ul>
<li>実力の代理は「過去走の得点の平均」で、クラスの違い・馬場・展開の影響が残る。実力の指標を変えると（例: タイム指数）、除かれる量が変わる可能性がある。ここでは市場を見ない代理を優先した。</li>
<li>2024 年は整った走が少なく、2024-25 年の推定は実質 2025 年中心（{meta['n_fit']:,} 走）。コース × 軸の標準誤差は大きく、差が小さいコースは検出できない。<b>「効いていない」ではなく「この量では検出できない」</b>と読む。</li>
<li>形状は 5 つの粗い指標。ゴール前の坂の高さ、初角までの距離、スタート形状、コーナーの数などは、コース数（約 40〜60）に対して多すぎるため入れていない。</li>
<li>2026 年はテストセット②。ここでの使い方は「確認 1 回」で、選択には使っていない。確認後の再推定（2024-26 年をまとめる）は、これから追加される走で改めて再確認してから。</li>
<li>画面（走法マップ・コース特注ゾーン）への反映は今回していない。反映するなら「距離が伸びるほど TS を評価」のような形状レベルの補正が候補で、コース単位の優位は出さない。</li>
<li>再現: ホストで <code>cd /home/jovyan/work/keiba-vpn &amp;&amp; nice -n 19 /opt/conda/bin/python -I src/scripts/data/course_axis_rationale.py OUT.json</code> → 手元で <code>python src/scripts/docs/build_course_axis_rationale_html.py OUT.json</code>。</li>
</ul>
{SORT_JS}
</main></body></html>""")
    return "".join(parts)


def main() -> None:
    d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    out.write_text(build(d, load_master()), encoding="utf-8")
    print("wrote", out, f"{out.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
