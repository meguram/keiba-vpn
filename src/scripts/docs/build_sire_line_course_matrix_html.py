"""父の系統（smartrc の大系統 → 小系統）× 会場・コースの、勝率・複勝率と着別カウント（1-2-3-4着以下）のマトリクス HTML を作る。

使い方: python src/scripts/docs/build_sire_line_course_matrix_html.py [OUT.html]
列 = 父の系統。1 階層目が大系統、2 階層目が小系統（○ に印を付けた種牡馬が半数以上の小系統は緑の名前）。
行 = 先頭に馬場状態別（芝・ダ × 良・稍重・重/不良。全会場）、続けて会場（1 階層。＋で開く）→ コース（芝・ダートと距離。2 階層）。
画面の操作（すべて画面側の JS。データは JSON で埋め込む）:
  - 色: 同じ行の全系統の率との差（出走数が少ないセルは率を行の基準へ縮める）。明るいほど好成績（緑）、暗いほど不振。水準以上のセルは枠で強調。
  - 並べ替え: 列見出し（小系統）・大系統の見出し・「全系統」を押すと、その列の率で行を並べ替える（降順 → 昇順 → 解除）。
    行ラベル（コース・会場・馬場状態）を押すと、その行の率で列を並べ替える（大系統の枠の中で小系統を、大系統どうしも）。
入力:
  data/page_reference/tables/<年>/race_result_flat.parquet （2016〜2026 の平地。障害を除く）
  data/page_reference/pedigree_race_index/horse_sire_map.parquet （馬 → 父）
  data/meta/sire_lines/sire_smartrc_colors.csv （父 → smartrc の大系統・小系統。系統が空欄の父は集計から外す）
  data/meta/sire_lines/euro_kick_marks.json （○ の種牡馬）
出力: data/meta/sire_lines/sire_line_course_matrix.html（docs/ 配下ではないので本には載せない）
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
FLAT = REPO / "data/page_reference/tables"
SIRE_MAP = REPO / "data/page_reference/pedigree_race_index/horse_sire_map.parquet"
COLORS = REPO / "data/meta/sire_lines/sire_smartrc_colors.csv"
MARKS = REPO / "data/meta/sire_lines/euro_kick_marks.json"
OUT = REPO / "data/meta/sire_lines/sire_line_course_matrix.html"
JST = timezone(timedelta(hours=9))
YEARS = range(2016, 2027)
VENUES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
MIN_LINE_RUNS = 300          # 列に出す小系統の最少出走数
_JUMP = r"障害|ジャンプ|JS\(|JG|J・G"
COND_ROWS = [("芝", "芝", "良"), ("芝", "芝", "稍重"), ("芝", "芝", "重・不良"), ("ダート", "ダ", "良"), ("ダート", "ダ", "稍重"), ("ダート", "ダ", "重・不良")]


def load_runs() -> tuple[pd.DataFrame, dict[str, dict]]:
    """走ごとの表（venue・surface・distance・line・pos・cond）と、小系統ごとの情報（大系統・種牡馬数・○ の種牡馬数）。"""
    parts = []
    for y in YEARS:
        p = FLAT / str(y) / "race_result_flat.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p, columns=["venue", "surface", "distance", "race_name", "finish_position", "horse_id", "track_condition"]))
    r = pd.concat(parts, ignore_index=True)
    r = r[r["surface"].isin(["芝", "ダート"]) & ~r["race_name"].fillna("").str.contains(_JUMP)].copy()
    r["pos"] = pd.to_numeric(r["finish_position"], errors="coerce")
    r["distance"] = pd.to_numeric(r["distance"], errors="coerce")
    r = r[(r["pos"] > 0) & r["distance"].notna()].copy()       # 着順が付いた走だけ（競走中止・取消・除外は -1 で外れる）
    r["horse_id"] = r["horse_id"].astype(str)
    m = pd.read_parquet(SIRE_MAP, columns=["horse_id", "sire_id"]).astype(str).drop_duplicates("horse_id")
    c = pd.read_csv(COLORS, dtype=str)
    ok = c[c["status"] == "ok"].set_index("sire_id")["small_name"]
    r = r.merge(m, on="horse_id", how="left")
    r["line"] = r["sire_id"].map(ok)
    r = r[r["line"].notna()].copy()
    marks = json.loads(MARKS.read_text(encoding="utf-8"))["marks"] if MARKS.exists() else {}
    o = {k for k, v in marks.items() if v.get("mark") == "o"}
    used = set(r["sire_id"])                                   # 母数は、この期間に産駒が走った父だけ（表の集計範囲に合わせる）
    c = c[(c["status"] == "ok") & c["sire_id"].isin(used)].assign(o=lambda d: d["sire_id"].isin(o))
    meta = {s: {"sires": int(len(g)), "o": int(g["o"].sum()), "large": str(g["large_name"].mode().iat[0])} for s, g in c.groupby("small_name")}
    r["cond"] = r["track_condition"].map({"良": "良", "稍重": "稍重", "重": "重・不良", "不良": "重・不良"})
    return r[["venue", "surface", "distance", "line", "pos", "cond"]], meta


def _with_counts(d: pd.DataFrame) -> pd.DataFrame:
    return d.assign(c1=(d["pos"] == 1).astype(int), c2=(d["pos"] == 2).astype(int), c3=(d["pos"] == 3).astype(int), c4=(d["pos"] >= 4).astype(int))


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    """(venue, surface, distance, line) ごとの 1・2・3・4着以下の数。"""
    return _with_counts(runs).groupby(["venue", "surface", "distance", "line"], as_index=False)[["c1", "c2", "c3", "c4"]].sum()


def aggregate_cond(runs: pd.DataFrame) -> pd.DataFrame:
    """(surface, cond, line) ごとの 1・2・3・4着以下の数（全会場。cond = 良・稍重・重/不良）。"""
    return _with_counts(runs.dropna(subset=["cond"])).groupby(["surface", "cond", "line"], as_index=False)[["c1", "c2", "c3", "c4"]].sum()


def _row(rid: str, kind: str, label: str, sub: pd.DataFrame, cols: list[str], extra: dict) -> dict:
    by = {s: g for s, g in sub.groupby("line")}

    def cnt(s: str) -> list[int]:
        g = by.get(s)
        return [0, 0, 0, 0] if g is None else [int(x) for x in g[["c1", "c2", "c3", "c4"]].sum()]

    return {"id": rid, "kind": kind, "label": label, "t": [int(x) for x in sub[["c1", "c2", "c3", "c4"]].sum()], "c": [cnt(s) for s in cols], **extra}


def model(counts: pd.DataFrame, meta: dict[str, dict], cond_counts: pd.DataFrame | None) -> dict:
    """画面に埋め込むデータ。cols = 小系統（大系統ごとに、大系統の出走数の多い順 → 小系統の出走数の多い順）、rows = 馬場状態・会場・コース。"""
    tot = counts.groupby("line")[["c1", "c2", "c3", "c4"]].sum().sum(axis=1)
    lines = [s for s in tot.index if tot[s] >= MIN_LINE_RUNS]
    large = {s: meta.get(s, {}).get("large", "（不明）") for s in lines}
    gtot: dict[str, int] = {}
    for s in lines:
        gtot[large[s]] = gtot.get(large[s], 0) + int(tot[s])
    lines.sort(key=lambda s: (-gtot[large[s]], large[s], -int(tot[s])))
    cols = [{"name": s, "large": large[s], "sires": meta.get(s, {}).get("sires", 0), "o": meta.get(s, {}).get("o", 0), "runs": int(tot[s])} for s in lines]
    rows: list[dict] = []
    if cond_counts is not None and len(cond_counts):
        for sf, sfl, cond in COND_ROWS:
            sub = cond_counts[(cond_counts["surface"] == sf) & (cond_counts["cond"] == cond)]
            if len(sub):
                rows.append(_row(f"cond:{sf}:{cond}", "cond", f"{sfl} {cond}", sub, lines, {"sf": sf}))
    for vi, v in enumerate(VENUES):
        vc = counts[counts["venue"] == v]
        if vc.empty:
            continue
        for vsf, vlab, sub in (("all", "全コース", vc), ("芝", "芝全体", vc[vc["surface"] == "芝"]), ("ダート", "ダート全体", vc[vc["surface"] == "ダート"])):
            if len(sub):
                rows.append(_row(f"venue:{v}:{vsf}", "venue", f"{v} {vlab}", sub, lines, {"v": vi, "vname": v, "vsf": vsf}))
        for (sf, dist), g in vc.groupby(["surface", "distance"]):
            rows.append(_row(f"course:{v}:{sf}:{int(dist)}", "course", f"{sf}{int(dist)}m", g, lines, {"v": vi, "vname": v, "sf": sf, "dist": int(dist)}))
    return {"cols": cols, "rows": rows, "min_line_runs": MIN_LINE_RUNS}


# ── 画面の JS ─────────────────────────────────────────────────────────────
# 純粋な計算（率・縮め・色・並べ替え）。DOM に触れないので、単体でテストできる（tests/scripts/…）。
JS_PURE = r"""
var PURE=(function(){
  var K=30;                                                    // 出走数が少ないセルの率を、行の基準へ縮める強さ（疑似標本数）
  function cnt(c){return c[0]+c[1]+c[2]+c[3];}
  function pl(c){return c[0]+c[1]+c[2];}
  function xv(c,m){return m==='p'?pl(c):c[0];}
  function rate(c,m){var n=cnt(c);return n?xv(c,m)/n:NaN;}
  function shrunk(c,b,m){var n=cnt(c);return n?(xv(c,m)+K*b)/(n+K):NaN;}
  function level(c,b,m,minN){var n=cnt(c);if(!n||n<minN)return null;var d=shrunk(c,b,m)-b,s=(m==='p'?0.10:0.05);return Math.max(-1,Math.min(1,d/s));}
  function sum(cs){var t=[0,0,0,0];cs.forEach(function(c){for(var i=0;i<4;i++)t[i]+=c[i];});return t;}
  function val(c,m,minN){return cnt(c)>=minN?rate(c,m):NaN;}      // 並べ替えに使う値（出走数が最少に満たなければ NaN＝末尾）
  function sortIdx(vals,dir){                                    // dir=1 降順、-1 昇順。NaN は常に末尾。同値は元の並び
    var ix=vals.map(function(v,i){return i;});
    ix.sort(function(a,b){var x1=vals[a],y1=vals[b],nx=!(x1===x1),ny=!(y1===y1);
      if(nx&&ny)return a-b; if(nx)return 1; if(ny)return -1; return (dir*(y1-x1))||(a-b);});
    return ix;
  }
  var NEU=[36,45,66], BAD=[58,10,20], GOOD=[80,240,150];          // 中立(暗い青灰) → 不振(さらに暗い赤) / 好成績(明るい緑)。RGB の線形補間なので 0 で途切れない
  function mix(a,b,t){return [0,1,2].map(function(i){return Math.round(a[i]+(b[i]-a[i])*t);});}
  function color(s){                                             // s: -1(不振)〜+1(好成績)。明るいほど好成績、暗いほど不振
    var c=s>=0?mix(NEU,GOOD,s):mix(NEU,BAD,-s), lum=0.299*c[0]+0.587*c[1]+0.114*c[2];
    return {bg:'rgb('+c.join(',')+')',fg:lum>=120?'#0b1b12':'#e5e9f5',hi:s>=0.5,lum:lum};
  }
  // 列の並び: 大系統の枠を保つ。rowCounts = 基準の行の各列の [c1..c4]（無ければ元の並び）、cols = [{large}]
  function orderCols(cols,rowCounts,m,minN,dir){
    var gs=[],gi={};cols.forEach(function(c,i){if(!(c.large in gi)){gi[c.large]=gs.length;gs.push({large:c.large,idx:[]});}gs[gi[c.large]].idx.push(i);});
    if(!rowCounts)return gs;
    gs.forEach(function(g){var v=g.idx.map(function(i){return val(rowCounts[i],m,minN);});var o=sortIdx(v,dir);g.idx=o.map(function(k){return g.idx[k];});});
    var gv=gs.map(function(g){return val(sum(g.idx.map(function(i){return rowCounts[i];})),m,minN);});
    return sortIdx(gv,dir).map(function(k){return gs[k];});
  }
  return {K:K,cnt:cnt,rate:rate,shrunk:shrunk,level:level,sum:sum,val:val,sortIdx:sortIdx,color:color,orderCols:orderCols};
})();
"""

JS_UI = r"""
(function(){
  var D=JSON.parse(document.getElementById('data').textContent), P=PURE;
  var st={m:'p',minN:30,sf:'',onlyO:false,open:{},rs:null,cs:null,showW:true,showP:true,showK:true};
  var tbl=document.getElementById('m'), VN=[]; D.rows.forEach(function(r){ if(r.vname&&VN.indexOf(r.vname)<0) VN.push(r.vname); });
  function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
  function isO(c){return c.sires>0&&c.o*2>=c.sires;}
  function visCols(){ var vis={}; D.cols.forEach(function(c,i){ if(!st.onlyO||isO(c)) vis[i]=1; }); return vis; }
  function rowById(id){ for(var i=0;i<D.rows.length;i++) if(D.rows[i].id===id) return D.rows[i]; return null; }
  // 列で行を並べ替えるときの値（key: 小系統の添字 / 'g:大系統' / 'tot'）
  function rowVal(r,key){
    var m=st.m,mn=st.minN,vis=visCols();
    if(key==='tot') return P.val(r.t,m,mn);
    if(typeof key==='string'&&key.indexOf('g:')===0){ var L=key.slice(2),cs=[]; D.cols.forEach(function(c,i){ if(c.large===L&&vis[i]) cs.push(r.c[i]); }); return P.val(P.sum(cs),m,mn); }   // 見えている列だけの合計
    return P.val(r.c[key],m,mn);
  }
  // 状態の整合: 並べ替えの基準が見えなくなったら付け替える／解除する
  function fixState(){
    var vis=visCols(), want=st.sf||'all';
    if(st.rs&&typeof st.rs.key==='number'&&!vis[st.rs.key]) st.rs=null;                      // 基準の列が「○だけ」で隠れた
    if(st.rs&&typeof st.rs.key==='string'&&st.rs.key.indexOf('g:')===0&&!D.cols.some(function(c,i){return vis[i]&&'g:'+c.large===st.rs.key;})) st.rs=null;
    if(st.cs){
      var id=st.cs.id, p=id.split(':');
      if(p[0]==='venue'&&p[2]!==want){ id='venue:'+p[1]+':'+want; if(!rowById(id)) st.cs=null; else st.cs.id=id; }   // 会場の行は、絞り込みに合う行へ付け替える
      else if((p[0]==='cond'||p[0]==='course')&&want!=='all'&&p[1]!==want&&p[0]==='cond') st.cs=null;
      else if(p[0]==='course'&&want!=='all'&&p[2]!==want) st.cs=null;
    }
  }
  function orderRows(rows){
    if(!st.rs) return rows.slice();
    var v=rows.map(function(r){return rowVal(r,st.rs.key);}), o=P.sortIdx(v,st.rs.dir); return o.map(function(i){return rows[i];});
  }
  function layout(){
    var want=st.sf||'all', out=[];
    var cond=D.rows.filter(function(r){return r.kind==='cond'&&(want==='all'||r.sf===want);});
    if(cond.length){ out.push({title:'馬場状態別（全会場。良・稍重・重／不良）'}); orderRows(cond).forEach(function(r){out.push({row:r});}); }
    out.push({title:'会場別（＋で開くとコース別）'});
    var blocks=VN.map(function(v){ return {v:v,head:D.rows.filter(function(r){return r.kind==='venue'&&r.vname===v&&r.vsf===want;})[0],
        courses:D.rows.filter(function(r){return r.kind==='course'&&r.vname===v&&(want==='all'||r.sf===want);})}; }).filter(function(b){return b.head;});
    if(st.rs){ var bv=blocks.map(function(b){return rowVal(b.head,st.rs.key);}); blocks=P.sortIdx(bv,st.rs.dir).map(function(i){return blocks[i];}); }
    blocks.forEach(function(b){ out.push({row:b.head,toggle:true,v:b.v}); if(st.open[b.v]) orderRows(b.courses).forEach(function(r){out.push({row:r});}); });
    return out;
  }
  function arrow(on,dir){ return on?(dir===1?' ▼':' ▲'):''; }
  function render(){
    fixState();
    var rowC=st.cs?rowById(st.cs.id):null, vis=visCols();
    var vi=D.cols.map(function(c,i){return i;}).filter(function(i){return vis[i];});          // 見えている列だけで並べる（隠れた列の値を合計に入れない）
    var gs=P.orderCols(vi.map(function(i){return D.cols[i];}),rowC?vi.map(function(i){return rowC.c[i];}):null,st.m,st.minN,st.cs?st.cs.dir:1)
      .map(function(g){return {large:g.large,idx:g.idx.map(function(k){return vi[k];})};});
    var h1='<tr><th class="sticky" rowspan="2">会場／コース<div class="hint">行名を押す＝列を並べ替え</div></th><th class="tot" rowspan="2" data-key="tot">全系統'+arrow(st.rs&&st.rs.key==='tot',st.rs&&st.rs.dir)+'</th>';
    var h2='<tr>', order=[];
    gs.forEach(function(g){
      var ix=g.idx.filter(function(i){return vis[i];}); if(!ix.length) return;
      var runs=ix.reduce(function(a,i){return a+D.cols[i].runs;},0), gk='g:'+g.large;
      h1+='<th class="grp" colspan="'+ix.length+'" data-key="'+esc(gk)+'">'+esc(g.large)+'<span class="sub"> 全体 '+runs.toLocaleString()+'走</span>'+arrow(st.rs&&st.rs.key===gk,st.rs&&st.rs.dir)+'</th>';
      ix.forEach(function(i){ var c=D.cols[i]; order.push(i);
        h2+='<th class="line'+(isO(c)?' o':'')+'" data-key="'+i+'"><div class="nm">'+esc(c.name)+arrow(st.rs&&st.rs.key===i,st.rs&&st.rs.dir)+'</div><div class="sub">○ '+c.o+'／'+c.sires+'頭<br>全体 '+c.runs.toLocaleString()+'走</div></th>'; });
    });
    h1+='</tr>';h2+='</tr>';
    var b='', lay=layout(), n=2+order.length;
    function cell(c,base,tot){
      var nn=P.cnt(c); if(!nn) return '<td class="c e"></td>';
      var s=tot?null:P.level(c,base,st.m,st.minN), col=s===null?null:P.color(s), w=c[0]/nn, p=P.rate(c,'p');
      var style=col?' style="background:'+col.bg+';color:'+col.fg+'"':'';
      var cls='c'+(tot?' t':'')+((!tot&&s===null)?' low':'')+((col&&col.hi)?' hi':'');
      return '<td class="'+cls+'"'+style+' title="'+nn+'走"><div class="r">'+(st.showW?'<span>勝 '+(100*w).toFixed(1)+'%</span>':'')+(st.showP?'<span>複 '+(100*p).toFixed(1)+'%</span>':'')+'</div>'
        +(st.showK?'<div class="k">'+c.join('-')+'<i>（'+nn+'）</i></div>':'')+'</td>';
    }
    lay.forEach(function(e){
      if(e.title){ b+='<tr class="gt"><th class="sticky" colspan="'+n+'">'+esc(e.title)+'</th></tr>'; return; }
      var r=e.row, base=P.rate(r.t,st.m), sorted=st.cs&&st.cs.id===r.id;
      var lab=(e.toggle?'<button type="button" class="tg" data-v="'+esc(e.v)+'">'+(st.open[e.v]?'－':'＋')+'</button> ':'')+esc(r.label)+arrow(sorted,st.cs&&st.cs.dir);
      b+='<tr class="'+r.kind+'"><th class="sticky rl" data-row="'+esc(r.id)+'">'+lab+'</th>'+cell(r.t,base,true);
      order.forEach(function(i){ b+=cell(r.c[i],base,false); });
      b+='</tr>';
    });
    tbl.innerHTML='<thead>'+h1+h2+'</thead><tbody>'+b+'</tbody>';
    var info=[]; if(st.rs){ var kk=st.rs.key; info.push('行の並べ替え: '+(kk==='tot'?'全系統':typeof kk==='number'?D.cols[kk].name:String(kk).slice(2))+(st.rs.dir===1?' ▼降順':' ▲昇順')); }
    if(st.cs){ var rr=rowById(st.cs.id); info.push('列の並べ替え: '+(rr?rr.label:'')+(st.cs.dir===1?' ▼降順':' ▲昇順')); }
    document.getElementById('sortinfo').textContent=info.join(' ／ ');
    var th1=tbl.querySelector('thead tr'); tbl.style.setProperty('--h1', th1.offsetHeight+'px');
  }
  function cycle(cur,key,idKey){ // 降順 → 昇順 → 解除
    var o={}; if(cur&&cur[idKey]===key){ if(cur.dir===1){ o.dir=-1; o[idKey]=key; return o; } return null; }
    o.dir=1; o[idKey]=key; return o;
  }
  tbl.addEventListener('click',function(e){
    var tg=e.target.closest('.tg'); if(tg){ var v=tg.dataset.v; st.open[v]=!st.open[v]; render(); return; }
    var th=e.target.closest('th[data-key]'); if(th){ var k=th.dataset.key; if(/^\d+$/.test(k)) k=+k; st.rs=cycle(st.rs,k,'key'); render(); return; }
    var rl=e.target.closest('th[data-row]'); if(rl){ st.cs=cycle(st.cs,rl.dataset.row,'id'); render(); return; }
  });
  function bind(id,fn,ev){ document.getElementById(id).addEventListener(ev||'change',function(){fn(this);render();}); }
  bind('metric',function(x){st.m=x.value;}); bind('minn',function(x){st.minN=Math.max(1,+x.value||1);},'input');
  bind('sw',function(x){st.showW=x.checked;}); bind('sp',function(x){st.showP=x.checked;}); bind('sk',function(x){st.showK=x.checked;});
  bind('onlyo',function(x){st.onlyO=x.checked;}); bind('sf',function(x){st.sf=x.value;});
  document.getElementById('open').addEventListener('click',function(){ VN.forEach(function(v){st.open[v]=true;}); render(); });
  document.getElementById('close').addEventListener('click',function(){ st.open={}; render(); });
  document.getElementById('reset').addEventListener('click',function(){ st.rs=null; st.cs=null; render(); });
  render();
})();
"""

CSS = """
:root { --bg:#0f172a; --panel:#15213c; --border:#243358; --text:#e5e9f5; --muted:#9aa7c7; --accent:#7aa9ff; --o:#34d399; }
body { margin:0; background:var(--bg); color:var(--text); font:13px/1.45 "Noto Sans JP",system-ui,sans-serif; }
main { padding:16px 14px 60px; } h1 { font-size:20px; margin:0 0 4px; } .muted { color:var(--muted); } p.lead { max-width:1180px; margin:4px 0 8px; color:var(--muted); }
code { background:#1c2a4d; padding:1px 6px; border-radius:5px; font-size:12px; }
.bar { display:flex; gap:12px; flex-wrap:wrap; align-items:center; margin:8px 0; position:sticky; top:0; z-index:20; background:var(--bg); padding:6px 0; border-bottom:1px solid var(--border); }
.bar select,.bar input[type=number],.bar button { background:var(--panel); color:var(--text); border:1px solid var(--border); border-radius:6px; padding:4px 8px; font-size:13px; }
.bar button { cursor:pointer; } .bar button:hover { border-color:var(--accent); }
.legend { display:inline-flex; align-items:center; gap:6px; font-size:12px; color:var(--muted); } .legend i { display:inline-block; width:150px; height:12px; border-radius:6px; background:linear-gradient(90deg,rgb(58,10,20),rgb(36,45,66),rgb(80,240,150)); } .sortinfo { color:var(--accent); font-size:12px; }
.wrap { overflow:auto; max-height:calc(100vh - 150px); border:1px solid var(--border); border-radius:8px; }
table { border-collapse:separate; border-spacing:0; }
th,td { border-bottom:1px solid var(--border); border-right:1px solid var(--border); padding:3px 6px; }
thead th { position:sticky; background:var(--panel); vertical-align:bottom; z-index:3; cursor:pointer; user-select:none; }
thead tr:first-child th { top:0; } thead tr:nth-child(2) th { top:var(--h1,30px); }
thead th.sticky { left:0; z-index:6; width:210px; min-width:210px; max-width:210px; cursor:default; } thead th.tot { left:210px; z-index:5; min-width:96px; }
thead th.grp { text-align:center; font-size:13px; } thead th:hover { background:#1d2d52; }
th.line { min-width:96px; } th.line .nm { font-weight:700; } th.line .sub,.grp .sub { font-weight:400; color:var(--muted); font-size:11px; } th.line.o .nm { color:var(--o); } th.line.o { box-shadow:inset 0 -3px 0 var(--o); }
.hint { font-weight:400; color:var(--muted); font-size:11px; }
tbody th.sticky { position:sticky; left:0; background:var(--panel); z-index:2; text-align:left; white-space:nowrap; cursor:pointer; width:210px; min-width:210px; max-width:210px; overflow:hidden; text-overflow:ellipsis; }
tbody th.sticky:hover { background:#1d2d52; } tr.gt th { background:var(--bg) !important; color:var(--accent); font-size:12px; padding:8px 6px 3px; cursor:default; }
td.c { text-align:center; min-width:96px; font-variant-numeric:tabular-nums; } td.c .k { font-size:11px; opacity:.85; white-space:nowrap; } td.c .k i { font-style:normal; opacity:.8; }
td.c .r span { margin:0 3px; } td.c.low { opacity:.45; } td.c.e { background:transparent; }
td.c.t { position:sticky; left:210px; z-index:1; background:#17264a; border-right:2px solid var(--border); }
td.c.hi { box-shadow:inset 0 0 0 2px rgba(190,255,215,.9); font-weight:700; }
tr.venue th.sticky,tr.venue td { font-weight:600; } tr.course th.sticky { padding-left:28px; font-weight:400; } tr.cond th.sticky { padding-left:14px; font-weight:600; }
.tg { width:22px; height:20px; padding:0; line-height:1; background:var(--panel); color:var(--text); border:1px solid var(--border); border-radius:5px; cursor:pointer; }
"""

TEMPLATE = """<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>父の系統 × 会場・コース 成績マトリクス</title><style>%CSS%</style></head><body><main>
<h1>父の系統 × 会場・コース 成績マトリクス</h1>
<p class="lead">作成: %NOW%。列は父の smartrc の系統（1 階層目が大系統、2 階層目が小系統。○ が付いた種牡馬が半数以上の小系統は緑の名前）、行は馬場状態 → 会場 → コース。2016〜2026 年の平地 %NRUNS% 走（父の系統が空欄の馬は除外）。
セルは「勝率・複勝率」と「1着-2着-3着-4着以下（出走数）」。<b>色</b>は同じ行の全系統の率との差で、<b>明るい緑ほど好成績、暗いほど不振</b>（複勝率は ±10pt、勝率は ±5pt で最大。出走数が少ないセルは率を行の基準に寄せて色を付け、最少出走数に満たないセルは薄く表示）。水準を明確に上回るセル（目盛りの半分以上）は白い枠で強調。<br>
<b>並べ替え</b>: 列の見出し（小系統・大系統・「全系統」）を押すと、その列の率で<b>行</b>を並べ替え（降順 → 昇順 → 解除）。行名（コース・会場・馬場状態）を押すと、その行の率で<b>列</b>を並べ替え（大系統の枠の中で小系統を、大系統どうしも）。並べ替えの基準は「率の基準」（複勝率／勝率）。<br>
<b>集計の範囲</b>: 競走中止・取消・除外は数えない（着順が付いた走だけ。全体の約 0.4%）。コースは「芝・ダートと距離」で、同じ会場の内回り・外回りは区別できず 1 行にまとめている（京都 芝1400・1600、新潟 芝2000 など）。
「全系統」の列と色の基準は、列に出していない小系統（出走数 %MINRUNS% 走未満）を含み、父の系統が空欄の馬を含まない。馬場状態の行は全会場の合計（重・不良は重と不良の合計）。生成: <code>src/scripts/docs/build_sire_line_course_matrix_html.py</code>。</p>
<div class="bar">
<label>率の基準（色・並べ替え） <select id="metric"><option value="p">複勝率</option><option value="w">勝率</option></select></label>
<label>最少出走数 <input id="minn" type="number" min="1" value="30" style="width:64px"></label>
<label><input type="checkbox" id="sw" checked> 勝率</label><label><input type="checkbox" id="sp" checked> 複勝率</label><label><input type="checkbox" id="sk" checked> 着別カウント</label>
<label><input type="checkbox" id="onlyo"> ○の小系統だけ</label>
<label>コース <select id="sf"><option value="">芝・ダート</option><option value="芝">芝のみ</option><option value="ダート">ダートのみ</option></select></label>
<button type="button" id="open">すべて開く</button><button type="button" id="close">すべて閉じる</button><button type="button" id="reset">並べ替えを解除</button>
<span class="legend">不振<i></i>好成績</span><span id="sortinfo" class="sortinfo"></span>
</div>
<div class="wrap"><table id="m"></table></div>
<script type="application/json" id="data">%DATA%</script>
<script>%PURE%</script>
<script>%UI%</script>
</main></body></html>
"""


def build(counts: pd.DataFrame, meta: dict[str, dict], cond_counts: pd.DataFrame | None = None) -> str:
    """集計表から HTML を作る。"""
    d = model(counts, meta, cond_counts)
    data = json.dumps(d, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")      # <script> や <!-- の途中で HTML の解析が切れないように（JSON としては同じ値）
    n = int(counts[["c1", "c2", "c3", "c4"]].sum().sum())
    return (TEMPLATE.replace("%CSS%", CSS).replace("%NOW%", datetime.now(JST).strftime("%Y-%m-%d %H:%M")).replace("%NRUNS%", f"{n:,}")
            .replace("%MINRUNS%", str(MIN_LINE_RUNS)).replace("%PURE%", JS_PURE).replace("%UI%", JS_UI).replace("%DATA%", data))     # データは最後に差し込む（データ内の %…% を置換しない）


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    runs, meta = load_runs()
    counts = aggregate(runs)
    cond = aggregate_cond(runs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(counts, meta, cond), encoding="utf-8")
    print(json.dumps({"runs": len(runs), "cells": len(counts), "lines": int(counts["line"].nunique()), "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
