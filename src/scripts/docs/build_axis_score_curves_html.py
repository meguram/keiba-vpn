"""走法軸の点数（1〜10）ごとの勝率・複勝率を、対象（クラス・馬場・距離・コース）× 走法軸のマトリクスで見る専用 HTML を作る（読むだけ・HTML を書く）。

入力: src/scripts/data/axis_score_curves.py の出力 JSON。
使い方: python src/scripts/docs/build_axis_score_curves_html.py CURVES.json [OUT.html]
"""
from __future__ import annotations

import html
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.scripts.docs.build_course_axis_demand_html import CSS  # noqa: E402

OUT = REPO / "docs/html/operations/running_axis_curves.html"
MIN_N = 30            # 線を出す最低の走数（JS の MINN と同じ）
MIN_SIDE = 100        # 高い側・低い側の走数がこれ未満なら差を出さない
Z_MARK = 2.6          # 印を付ける z（軸 × 対象で数百の組を見るので 1.96 より厳しめ）
EXTRA_CSS = """
.ctl { display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin:10px 0; } .ctl label { font-size:13px; color:var(--muted); }
.ctl button { background:var(--panel); color:var(--text); border:1px solid var(--border); border-radius:8px; padding:5px 12px; font-size:13px; cursor:pointer; }
.ctl button.on { background:var(--accent); color:#0b1428; font-weight:700; }
.mxwrap { overflow:auto; max-height:82vh; border:1px solid var(--border); border-radius:10px; }
table.mx { border-collapse:separate; border-spacing:0; margin:0; width:auto; font-size:12px; }
table.mx th, table.mx td { border:0; border-right:1px solid var(--border); border-bottom:1px solid var(--border); padding:4px 5px; vertical-align:top; background:var(--bg); }
table.mx thead th { position:sticky; top:0; z-index:3; background:var(--panel); white-space:normal; min-width:178px; text-align:center; }
table.mx th.rh { position:sticky; left:0; z-index:2; background:var(--panel); min-width:132px; max-width:150px; white-space:normal; text-align:left; font-weight:600; color:var(--text); }
table.mx thead th.rh { z-index:4; }
table.mx td { cursor:pointer; } table.mx td:hover { outline:2px solid var(--accent); outline-offset:-2px; }
table.mx td.sel { outline:2px solid var(--warn); outline-offset:-2px; }
.rh small, th small { display:block; color:var(--muted); font-weight:400; font-size:11px; line-height:1.35; }
.cap { font-size:10.5px; color:var(--muted); line-height:1.35; text-align:center; } .cap b { color:var(--text); }
.big { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:10px 14px; margin:12px 0; }
.big svg { width:100%; max-width:880px; height:auto; display:block; } .big table { width:auto; }
.lg span { margin-right:16px; font-size:12px; color:var(--muted); } .lg i { display:inline-block; width:18px; height:0; border-top:3px solid; vertical-align:middle; margin-right:5px; } .lg i.d { border-top-style:dashed; }
td.hm { text-align:center; font-variant-numeric:tabular-nums; min-width:62px; } td.hm.k { font-weight:700; outline:1px solid var(--text); outline-offset:-1px; }
"""


def side_stats(arr: list, lo: tuple[int, int] = (0, 4), hi: tuple[int, int] = (5, 10)) -> dict | None:
    """arr = [n, w, p, ew, ep]（各 10 区間）。低（4.5 点以下）と高（6.0 点以上）の複勝率・人気補正後の差。"""
    def agg(a: tuple[int, int]) -> tuple[float, float, float]:
        n = sum(arr[0][a[0]:a[1]])
        return n, sum(arr[2][a[0]:a[1]]), sum(arr[4][a[0]:a[1]])
    nl, pl, el = agg(lo)
    nh, ph, eh = agg(hi)
    if nl < MIN_SIDE or nh < MIN_SIDE:
        return None
    rl, rh = pl / nl, ph / nh
    se = math.sqrt(rl * (1 - rl) / nl + rh * (1 - rh) / nh)
    d_adj = (ph - eh) / nh - (pl - el) / nl
    return {"nl": nl, "nh": nh, "raw": rh - rl, "adj": d_adj, "z": d_adj / se if se > 0 else 0.0}


def fmt_pt(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:+.1f}"


def build(d: dict) -> str:
    meta = d["meta"]
    g_all = d["dims"]["all"]["groups"][0]["data"]
    tr = []
    for ax in d["axes"]:
        c = g_all.get(ax["key"])
        if not c:
            continue
        a, b = side_stats(c["A"]), side_stats(c["B"])
        mark = ' class="strong"' if a and abs(a["z"]) >= Z_MARK else ""
        tr.append(
            f"<tr><td>{html.escape(ax['label'])}</td><td class=\"num\">{sum(c['A'][0]):,}</td><td class=\"num\">{sum(c['B'][0]):,}</td>"
            f"<td class=\"num\">{fmt_pt(a['raw'] if a else None)}</td><td class=\"num\"{mark}>{fmt_pt(a['adj'] if a else None)}</td><td class=\"num\">{('%+.1f' % a['z']) if a else '—'}</td>"
            f"<td class=\"num\">{fmt_pt(b['raw'] if b else None)}</td><td class=\"num\">{fmt_pt(b['adj'] if b else None)}</td></tr>")
    axis_rows = "".join(
        f"<tr><td>{html.escape(a['label'])}</td><td>{html.escape(a['high'])}</td><td>{html.escape(a['low'])}</td>"
        f"<td class=\"num\">{meta['axis_runs'][a['key']]['A']:,}</td><td class=\"num\">{meta['axis_runs'][a['key']]['B']:,}</td></tr>"
        for a in d["axes"])
    n_marked = 0
    n_all = 0
    for dim in d["dims"].values():
        for g in dim["groups"]:
            for ax in d["axes"]:
                c = g["data"].get(ax["key"])
                s = side_stats(c["A"]) if c else None
                if s:
                    n_all += 1
                    n_marked += abs(s["z"]) >= Z_MARK
    chance = n_all * math.erfc(Z_MARK / math.sqrt(2))      # |z|>=Z_MARK の両側確率 × 組数
    bonf = NormalDist().inv_cdf(1 - 0.025 / max(n_all, 1))   # 全組で 5% に抑える線（Bonferroni）
    data_json = json.dumps(d, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>走法軸の点数別 勝率・複勝率（軸 × 対象のマトリクス） — keiba-vpn docs</title><style>{CSS}{EXTRA_CSS}</style></head><body><main>
<h1>走法軸の点数別 勝率・複勝率（軸 × 対象のマトリクス）</h1>
<p class="muted">作成: {meta['built']}。走法マップの 10 点満点の走法軸（出走前の点）ごとに、横軸 = 点数、縦軸 = その点数の馬の<b>勝率（左軸）</b>と<b>複勝率（右軸）</b>を、
対象（クラス・馬場・距離・コース）ごとに並べたチャート集。集計は <code>src/scripts/data/axis_score_curves.py</code>（ホストのローカルの表だけ。GCS・API は使わない）、この HTML は <code>src/scripts/docs/build_axis_score_curves_html.py</code> が作る。
費用: 集計は一度きりのホスト内の計算で、周期処理・GCS 読み出しは増えない（GCP・Cloudflare の課金に影響なし）。</p>

<h2>1. 読み方</h2>
<ul>
<li><b>点数</b>: 各走の<b>直前までの走</b>で作った走法軸の点（1〜10、0.5 刻み。画面の走法マップと同じ）。点の整数部で 10 区間に分けた（「5」の区間 = 5.0〜5.5 点）。測れない軸は欠測で、その走は数えない。</li>
<li><b>勝率</b> = 1 着の割合。<b>複勝率</b> = 3 着以内（8 頭以上）／2 着以内（5〜7 頭）の割合。5 頭未満・障害・取消除外などは除いた。</li>
<li>チャート: 青の実線・丸 = 勝率（左軸）、橙の実線・四角 = 複勝率（右軸）。<b>右軸は左軸の 3 倍</b>に固定してあるので、全体の平均（点線）が近い高さに並ぶ。灰色の棒 = 走数。走数 {MIN_N} 未満の区間は線を切って出さない。</li>
<li><b>薄い破線 = 人気補正の目安</b>: 同じ頭数・同じ人気順の馬が、{meta['exp_years'][0]}〜{meta['exp_years'][1]} 年に平均して取っていた勝率・複勝率の、その区間での平均。
走法軸の点が高い馬ほど人気も高い（Hペース耐性は実力の代理でもある）ので、<b>実線が破線を上回る分</b>が「人気（市場が見ている実力）では説明できない上乗せ」の目安。実線だけを見ると実力の差が混ざる。</li>
<li>期間: 実線 = {meta['periods']['A'][0]}〜{meta['periods']['A'][1]}（走数 {meta['runs']['A']:,}）。「2026 年を重ねる」を押すと、{meta['periods']['B'][0]}〜（走数 {meta['runs']['B']:,}。テストセット②。<b>確認だけで、選択には使わない</b>）を白抜きの点で重ねる。</li>
<li>各マスの下の数字 = <b>高 − 低</b>（6.0 点以上の複勝率 − 4.5 点以下の複勝率、ポイント。5.0・5.5 点は除く）と、<b>補正後</b>（同じ差を、人気から期待される複勝率を引いてから取ったもの）。マスをクリックすると、下の拡大図に信頼区間（Wilson 95%）と数表が出る。</li>
</ul>

<h2>2. 走法軸の一覧と走数</h2>
<div class="scroll"><table><thead><tr><th>走法軸</th><th>点が高い</th><th>点が低い</th><th>走数 2024-25</th><th>走数 2026</th></tr></thead><tbody>{axis_rows}</tbody></table></div>
<p class="sub">走数 = その軸の点が測れた走の数。位置取り由来の軸（Hペース耐性・先行力・Cr性能・まくり特性）は 2024 年の位置データが薄いので、2025 年中心。
道悪適性（芝）は芝の走だけ、道悪適性（ダート）はダートの走だけ。急坂適性は、中山・阪神での過去走から作った点（どのコースの走にも付く）。<b>基礎スピード・道悪適性は、走法マップの良馬場／道悪の地図の軸で、馬場状態が点の計算に入る</b>。</p>

<h2>3. マトリクス</h2>
<div class="ctl" id="ctl"></div>
<div class="big" id="big"><span class="muted">下のマスをクリックすると、ここに拡大図と数表が出る。</span></div>
<div class="mxwrap"><table class="mx" id="mx"></table></div>
<p class="sub">左軸（勝率）の目盛は、この対象の中で走数 200 以上の区間の最大値（複勝率の 1/3 も含む）で共通（マス同士の高さを比べられる）。上限を超える区間は上端で切る（走数の少ない区間で起きる）。</p>

<h2>4. 軸ごとの要約（全体・複勝率の高 − 低）</h2>
<div class="scroll"><table><thead><tr><th>走法軸</th><th>走数 2024-25</th><th>走数 2026</th><th>高−低 (pt) 2024-25</th><th>補正後 (pt) 2024-25</th><th>z</th><th>高−低 (pt) 2026</th><th>補正後 (pt) 2026</th></tr></thead><tbody>{''.join(tr)}</tbody></table></div>
<p class="sub">「高−低」= 6.0 点以上の複勝率 − 4.5 点以下の複勝率（5.0・5.5 点は除く。区間は整数部なので、低い側は 1〜4 の区間、高い側は 6〜10 の区間）。「補正後」は人気から期待される複勝率を引いた差（人気では説明できない分）。z = 補正後の差 ÷ 標準誤差（期待値は固定として扱う）。</p>
<p class="sub">軸 × 対象の全 {n_all} 組（2024-25 年）のうち |z| ≥ {Z_MARK} は {n_marked} 組。偶然だけでも約 {chance:.0f} 組は出る（全 {n_all} 組で偶然を 5% に抑える Bonferroni の線は |z| ≥ {bonf:.1f}）。対象（全体・クラス・馬場・距離・コース）は同じ走を重ねて切っているので、印は独立な発見の数ではない。z は、同じレースの馬どうしの相関と同じ馬の重複を無視した標準誤差なので、実際より大きめに出る。<b>印は「見る候補」で、個別のコース・軸の優位とは言い切れない</b>。再現するかは 2026 年の欄（同符号か）で確認する。</p>

<h2>5. 一覧（補正後の差・対象 × 軸）</h2>
<p class="sub">複勝率の「高 − 低」から人気分を引いた差（pt）。色 = 向き（緑 = 高い点ほど人気以上に来る、赤 = 逆）、濃さ = |z|。太枠 = |z| ≥ {Z_MARK}。上の対象のボタンで切り替わる。</p>
<div class="scroll"><table class="mx2" id="hm"></table></div>

<h2>6. 限界</h2>
<ul>
<li><b>実力が混ざる</b>: 実線（補正なし）は、点の高い馬ほど強い馬（人気の馬）が多いことを含む。人気で補正した破線との差を見ること。人気は市場が見ている実力のほぼ全てで、補正後の差は小さい（多くは 0 近辺）のが普通。</li>
<li><b>点は「過去走」の値だが、基準は全期間</b>: 各走の点はその走より前の走から作るが、走歴が浅い馬の点を寄せる先（父の平均・全馬の平均）と、10 点への換算の基準は全期間（2024〜25 年・2026 年を含む）の値で作ってある（<code>build_axis_scores10.py</code>）。走歴の浅い馬では、点と着順の関係が実際より強く出うる。ここでは走歴で絞っていない。点の換算は全馬に同じ単調な変換なので、点と着順の関係そのものはほぼ変えない。</li>
<li><b>頭数と中止の扱い</b>: 複勝の頭数の区切り（8 頭以上は 3 着以内）には、結果表の頭数（取消・除外を含みうる）を使った。着順のない走（競走中止など）は除いたので、勝率・複勝率はわずかに高めに出る。人気から期待される率の表には障害のレースも混ざる。いずれも影響は小さい見込みだが、数値は確かめていない。</li>
<li><b>軸どうしは独立でない</b>: Hペース耐性は TS・TS持続力・前後半比と、先行力は出脚と相関がある。軸ごとの曲線は互いの影響を含む（1 軸ずつの周辺の図）。</li>
<li>期間は 2024 年以降だけ（位置・ラップ由来の軸が使えるのが 2024 年以降のため）。2020〜2023 年は含めていない。2026 年は確認用で、決定には使わない。</li>
<li>コース別は走数が少なく（{meta['course_min_runs']} 走以上のコースだけ）、区間ごとの走数は数十〜数百。線が切れるのは走数不足で、「効かない」ではなく「見えない」。</li>
<li>点が高い・低いに寄る走は少なく、1〜2 点・9〜10 点の区間は走数が少ない（棒を見ること）。</li>
</ul>
<p class="muted">再現: ホストで <code>cd /home/jovyan/work/keiba-vpn &amp;&amp; nice -n 19 /opt/conda/bin/python -I src/scripts/data/axis_score_curves.py OUT.json</code> → 手元で <code>python src/scripts/docs/build_axis_score_curves_html.py OUT.json</code>。
元の点の表: <code>data/local/running_profile/axis_scores10.parquet</code>（{html.escape(meta['history_mtime'][:10])} 時点）。</p>

<script>
var D={data_json};
{JS.replace("__MINN__", str(MIN_N)).replace("__SIDE__", str(MIN_SIDE)).replace("__ZM__", str(Z_MARK))}
</script>
</main></body></html>"""


JS = r"""
(function(){
var MINN=__MINN__,MINB=20,SIDE=__SIDE__,ZM=__ZM__,NS="http://www.w3.org/2000/svg";
var DIMS=Object.keys(D.dims),st={dim:"class",ov:false,exp:true,sel:null};
var AX=D.axes,axIdx={};AX.forEach(function(a){axIdx[a.key]=a;});
function wilson(k,n){if(n<=0)return [0,0];var z=1.96,p=k/n,d=1+z*z/n,c=p+z*z/(2*n),h=z*Math.sqrt(p*(1-p)/n+z*z/(4*n*n));return [(c-h)/d,(c+h)/d];}
function sum(a,i,j){var s=0;for(var k=i;k<j;k++)s+=a[k];return s;}
function side(c){ // c = [n,w,p,ew,ep]
  var nl=sum(c[0],0,4),pl=sum(c[2],0,4),el=sum(c[4],0,4),nh=sum(c[0],5,10),ph=sum(c[2],5,10),eh=sum(c[4],5,10);
  if(nl<SIDE||nh<SIDE)return null;
  var rl=pl/nl,rh=ph/nh,se=Math.sqrt(rl*(1-rl)/nl+rh*(1-rh)/nh),adj=(ph-eh)/nh-(pl-el)/nl;
  return {raw:rh-rl,adj:adj,z:se>0?adj/se:0};
}
function f1(x){return (x>=0?"+":"")+(x*100).toFixed(1);}
function groups(){return D.dims[st.dim].groups;}
function wmaxOf(gs){
  var m=0;gs.forEach(function(g){Object.keys(g.data).forEach(function(k){var c=g.data[k].A;for(var i=0;i<10;i++){if(c[0][i]>=200){var r=c[1][i]/c[0][i];if(r>m)m=r;var q=c[2][i]/c[0][i]/3;if(q>m)m=q;}}});});
  m=Math.min(Math.max(m*1.08,0.10),0.35);return Math.ceil(m*100/5)/20; // 0.05 刻みに切り上げ
}
function chart(c,W,H,wmax,big){
  var pl=big?46:26,pr=big?50:26,pt=big?14:8,pb=big?34:18,iw=W-pl-pr,ih=H-pt-pb;
  var A=c.A,B=c.B,s=[];
  function X(i){return pl+(i+0.5)*iw/10;}
  function YW(v){return pt+ih*(1-Math.min(v/wmax,1));}
  function YP(v){return pt+ih*(1-Math.min(v/(3*wmax),1));}
  s.push('<svg xmlns="'+NS+'" viewBox="0 0 '+W+' '+H+'" role="img" aria-label="走法軸の点数別 勝率・複勝率">');
  s.push('<rect x="'+pl+'" y="'+pt+'" width="'+iw+'" height="'+ih+'" fill="#0b1428" stroke="#243358"/>');
  var nmax=Math.max.apply(null,A[0].concat([1]));
  for(var i=0;i<10;i++){var bh=ih*0.18*A[0][i]/nmax;s.push('<rect x="'+(X(i)-iw/26)+'" y="'+(pt+ih-bh)+'" width="'+(iw/13)+'" height="'+bh+'" fill="#3b4a73" opacity="0.8"/>');}
  var nt=sum(A[0],0,10),wm=sum(A[1],0,10)/Math.max(nt,1),pm=sum(A[2],0,10)/Math.max(nt,1);
  if(nt>0){s.push('<line x1="'+pl+'" x2="'+(pl+iw)+'" y1="'+YW(wm)+'" y2="'+YW(wm)+'" stroke="#7aa9ff" stroke-dasharray="1 3" opacity="0.7"/>');
    s.push('<line x1="'+pl+'" x2="'+(pl+iw)+'" y1="'+YP(pm)+'" y2="'+YP(pm)+'" stroke="#fbbf24" stroke-dasharray="1 3" opacity="0.7"/>');}
  function band(idx,col,Y){ if(!big)return; // Wilson の帯
    var up=[],lo=[];for(var i=0;i<10;i++){if(A[0][i]>=MINN){var w=wilson(A[idx][i],A[0][i]);up.push([X(i),Y(w[1])]);lo.push([X(i),Y(w[0])]);}else{flush();}}
    flush();
    function flush(){if(up.length>1){var d="M"+up.map(function(p){return p[0]+" "+p[1];}).join("L")+"L"+lo.slice().reverse().map(function(p){return p[0]+" "+p[1];}).join("L")+"Z";s.push('<path d="'+d+'" fill="'+col+'" opacity="0.16"/>');}up=[];lo=[];}
  }
  band(1,"#7aa9ff",YW);band(2,"#fbbf24",YP);
  function line(val,Y,col,dash,w,marker,minn,arr){
    var seg=[];function flush(){if(seg.length>1)s.push('<polyline points="'+seg.map(function(p){return p[0]+","+p[1];}).join(" ")+'" fill="none" stroke="'+col+'" stroke-width="'+w+'"'+(dash?' stroke-dasharray="4 3"':'')+'/>');seg=[];}
    for(var i=0;i<10;i++){if(arr[0][i]>=minn)seg.push([X(i),Y(val(i))]);else flush();}flush();
    if(marker){var r=big?4:2.4;for(var i=0;i<10;i++){if(arr[0][i]>=minn){var x=X(i),y=Y(val(i));
      s.push(marker==="c"?'<circle cx="'+x+'" cy="'+y+'" r="'+r+'" fill="'+col+'"/>':'<rect x="'+(x-r)+'" y="'+(y-r)+'" width="'+2*r+'" height="'+2*r+'" fill="'+col+'"/>');}}}
  }
  if(st.exp){line(function(i){return A[3][i]/A[0][i];},YW,"#7aa9ff",true,big?1.6:1,null,MINN,A);line(function(i){return A[4][i]/A[0][i];},YP,"#fbbf24",true,big?1.6:1,null,MINN,A);}
  line(function(i){return A[1][i]/A[0][i];},YW,"#7aa9ff",false,big?2.4:1.6,"c",MINN,A);line(function(i){return A[2][i]/A[0][i];},YP,"#fbbf24",false,big?2.4:1.6,"s",MINN,A);
  if(st.ov){for(var i=0;i<10;i++){if(B[0][i]>=MINB){var r=big?4.5:2.8;
    s.push('<circle cx="'+X(i)+'" cy="'+YW(B[1][i]/B[0][i])+'" r="'+r+'" fill="none" stroke="#e6ecff" stroke-width="1.3"/>');
    s.push('<rect x="'+(X(i)-r)+'" y="'+(YP(B[2][i]/B[0][i])-r)+'" width="'+2*r+'" height="'+2*r+'" fill="none" stroke="#e6ecff" stroke-width="1.3"/>');}}}
  // 軸の目盛
  var tf=big?11:8.5;
  s.push('<text x="'+(pl-4)+'" y="'+(pt+4)+'" fill="#7aa9ff" font-size="'+tf+'" text-anchor="end">'+(wmax*100).toFixed(0)+'%</text><text x="'+(pl-4)+'" y="'+(pt+ih)+'" fill="#7aa9ff" font-size="'+tf+'" text-anchor="end">0</text>');
  s.push('<text x="'+(pl+iw+4)+'" y="'+(pt+4)+'" fill="#fbbf24" font-size="'+tf+'">'+(3*wmax*100).toFixed(0)+'%</text><text x="'+(pl+iw+4)+'" y="'+(pt+ih)+'" fill="#fbbf24" font-size="'+tf+'">0</text>');
  for(var i=0;i<10;i++){if(big||i===0||i===4||i===9)s.push('<text x="'+X(i)+'" y="'+(pt+ih+(big?16:11))+'" fill="#9aa7c7" font-size="'+tf+'" text-anchor="middle">'+(i+1)+'</text>');}
  if(big){s.push('<text x="'+(pl+iw/2)+'" y="'+(H-4)+'" fill="#9aa7c7" font-size="12" text-anchor="middle">走法軸の点数（整数部。10 = 10 点）</text>');
    s.push('<text transform="translate(12,'+(pt+ih/2)+') rotate(-90)" fill="#7aa9ff" font-size="12" text-anchor="middle">勝率</text><text transform="translate('+(W-8)+','+(pt+ih/2)+') rotate(90)" fill="#fbbf24" font-size="12" text-anchor="middle">複勝率</text>');}
  s.push('</svg>');return s.join("");
}
function ctl(){
  var h=[],el=document.getElementById("ctl");
  h.push('<label>対象</label>');
  DIMS.forEach(function(k){h.push('<button data-d="'+k+'" class="'+(k===st.dim?'on':'')+'">'+D.dims[k].label+'</button>');});
  h.push('<button data-t="ov" class="'+(st.ov?'on':'')+'">2026 年を重ねる</button><button data-t="exp" class="'+(st.exp?'on':'')+'">人気補正の破線</button>');
  el.innerHTML=h.join("");
  Array.prototype.forEach.call(el.querySelectorAll("button"),function(b){b.onclick=function(){
    if(b.dataset.d){st.dim=b.dataset.d;st.sel=null;}else if(b.dataset.t==="ov")st.ov=!st.ov;else st.exp=!st.exp;
    ctl();render();big();};});
}
function render(){
  var gs=groups(),wmax=wmaxOf(gs),h=[];
  h.push('<thead><tr><th class="rh">走法軸</th>'+gs.map(function(g){var n=0;Object.keys(g.data).forEach(function(k){n=Math.max(n,sum(g.data[k].A[0],0,10));});return '<th>'+g.label+'<small>最大走数 '+n.toLocaleString()+'</small></th>';}).join("")+'</tr></thead><tbody>');
  AX.forEach(function(a){
    h.push('<tr><th class="rh">'+a.label+'<small>高: '+a.high+'<br>低: '+a.low+'</small></th>');
    gs.forEach(function(g,gi){
      var c=g.data[a.key];
      if(!c||sum(c.A[0],0,10)<MINN){h.push('<td><div class="cap">—</div></td>');return;}
      var s=side(c.A),cap=s?'高−低 <b>'+f1(s.raw)+'</b><br>補正後 <b'+(Math.abs(s.z)>=ZM?' style="color:var(--warn)"':'')+'>'+f1(s.adj)+'</b>':'走数不足';
      h.push('<td data-a="'+a.key+'" data-g="'+gi+'" class="'+(st.sel&&st.sel[0]===a.key&&st.sel[1]===gi?'sel':'')+'">'+chart(c,178,104,wmax,false)+'<div class="cap">'+cap+'</div></td>');
    });
    h.push('</tr>');
  });
  h.push('</tbody>');
  var mx=document.getElementById("mx");mx.innerHTML=h.join("");
  Array.prototype.forEach.call(mx.querySelectorAll("td[data-a]"),function(td){td.onclick=function(){st.sel=[td.dataset.a,+td.dataset.g];var o=mx.querySelector("td.sel");if(o)o.classList.remove("sel");td.classList.add("sel");big();document.getElementById("big").scrollIntoView({block:"nearest"});};});
  heat();
}
function big(){
  var el=document.getElementById("big");
  if(!st.sel){el.innerHTML='<span class="muted">下のマスをクリックすると、ここに拡大図と数表が出る。</span>';return;}
  var g=groups()[st.sel[1]],a=axIdx[st.sel[0]],c=g.data[a.key];
  var wmax=wmaxOf([g]),h=[];
  var s=side(c.A),sb=side(c.B);
  h.push('<b>'+a.label+' × '+g.label+'</b> <span class="muted">（高: '+a.high+' ／ 低: '+a.low+'）</span>');
  h.push('<div class="lg"><span><i style="border-color:#7aa9ff"></i>勝率（左軸）</span><span><i style="border-color:#fbbf24"></i>複勝率（右軸 = 左の 3 倍）</span><span><i class="d" style="border-color:#7aa9ff"></i><i class="d" style="border-color:#fbbf24"></i>人気から期待される率（同じ色の破線）</span><span>帯 = Wilson 95%</span><span>白抜き = 2026 年</span></div>');
  h.push(chart(c,880,380,wmax,true));
  h.push('<p class="sub">高−低（6.0 点以上 − 4.5 点以下の複勝率）: 2024-25 '+(s?f1(s.raw)+' pt（補正後 '+f1(s.adj)+'、z '+s.z.toFixed(1)+'）':'走数不足')+' ／ 2026 '+(sb?f1(sb.raw)+' pt（補正後 '+f1(sb.adj)+'、z '+sb.z.toFixed(1)+'）':'走数不足')+'</p>');
  h.push('<div class="scroll"><table><thead><tr><th>点</th><th>走数</th><th>勝率</th><th>勝率の期待</th><th>複勝率</th><th>複勝率の期待</th><th>2026 走数</th><th>2026 勝率</th><th>2026 複勝率</th></tr></thead><tbody>');
  function r(x,n){return n>0?(x/n*100).toFixed(1)+"%":"—";}
  for(var i=0;i<10;i++){var n=c.A[0][i],m=c.B[0][i];
    h.push('<tr><td>'+(i+1)+'</td><td class="num">'+n+'</td><td class="num">'+r(c.A[1][i],n)+'</td><td class="num">'+r(c.A[3][i],n)+'</td><td class="num">'+r(c.A[2][i],n)+'</td><td class="num">'+r(c.A[4][i],n)+'</td><td class="num">'+m+'</td><td class="num">'+r(c.B[1][i],m)+'</td><td class="num">'+r(c.B[2][i],m)+'</td></tr>');}
  h.push('</tbody></table></div>');
  el.innerHTML=h.join("");
}
function heat(){
  var gs=groups(),h=['<thead><tr><th>走法軸</th>'+gs.map(function(g){return '<th>'+g.label+'</th>';}).join("")+'</tr></thead><tbody>'];
  AX.forEach(function(a){
    h.push('<tr><td>'+a.label+'</td>');
    gs.forEach(function(g){var c=g.data[a.key],s=c?side(c.A):null;
      if(!s){h.push('<td class="hm">—</td>');return;}
      var al=Math.min(Math.abs(s.z)/4,1)*0.75,col=s.adj>=0?'74,222,128':'248,113,113';
      h.push('<td class="hm'+(Math.abs(s.z)>=ZM?' k':'')+'" style="background:rgba('+col+','+al.toFixed(2)+')" title="z '+s.z.toFixed(1)+'">'+f1(s.adj)+'</td>');});
    h.push('</tr>');});
  h.push('</tbody>');document.getElementById("hm").innerHTML=h.join("");
}
ctl();render();
})();
"""


def main() -> None:
    d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    out.write_text(build(d), encoding="utf-8")
    print("wrote", out, f"{out.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
