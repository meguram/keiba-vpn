#!/usr/bin/env python3
"""NB-01 データ探索・前処理のインタラクティブ HTML を生成する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "notebooks/megu_index/output/nb01"
SAMPLE_PER_CELL = 500


def _cell_key(*parts) -> str:
    return "|".join(str(p) for p in parts)


def build_weight_viz_payload(
    df_fit: pd.DataFrame,
    coef_by_cell: pd.DataFrame,
    sample_per_cell: int = SAMPLE_PER_CELL,
) -> dict:
    points: dict[str, list[list[float]]] = {}
    if not df_fit.empty:
        for (surf, sex_g, dist), grp in df_fit.groupby(
            ["surface", "sex_group", "distance_num"], sort=False
        ):
            key = _cell_key(surf, sex_g, int(dist))
            sub = grp
            if len(sub) > sample_per_cell:
                sub = sub.sample(sample_per_cell, random_state=42)
            points[key] = (
                sub[["weight_dev_dm", "time_dm"]]
                .astype(float)
                .round(4)
                .values.tolist()
            )

    cells = []
    for row in coef_by_cell.drop_duplicates(["surface", "sex_group", "distance_num"]).itertuples(index=False):
        cells.append(
            {
                "surface": str(row.surface),
                "sex_group": str(row.sex_group),
                "distance": int(row.distance_num),
                "sec_per_kg": float(row.sec_per_kg_final),
                "source": str(row.weight_coef_source),
                "n_fit": int(row.n_fit),
                "key": _cell_key(row.surface, row.sex_group, int(row.distance_num)),
            }
        )

    return {
        "meta": {"n_cells": len(cells), "n_fit_rows": int(len(df_fit))},
        "cells": cells,
        "points": points,
    }


def build_track_viz_payload(day_course_tbl: pd.DataFrame) -> dict:
    series: dict[str, list[dict]] = {}
    if day_course_tbl.empty:
        return {"meta": {"n_rows": 0}, "series": series}

    work = day_course_tbl.copy()
    work["date_str"] = pd.to_datetime(work["date_str"], errors="coerce").dt.strftime("%Y-%m-%d")
    for (venue, surface), grp in work.groupby(["venue", "surface"], sort=False):
        key = _cell_key(venue, surface)
        sub = grp.sort_values("date_str")
        series[key] = [
            {
                "date": str(r.date_str),
                "track_dev_sec": float(r.track_dev_sec) if pd.notna(r.track_dev_sec) else None,
                "tsi_raw": float(r.tsi_raw) if pd.notna(r.tsi_raw) else None,
                "n_races": int(r.n_races_track) if pd.notna(r.n_races_track) else 0,
            }
            for r in sub.itertuples(index=False)
        ]

    venues = sorted(work["venue"].dropna().unique().tolist())
    return {"meta": {"n_rows": int(len(work)), "venues": venues}, "series": series}


def build_par_split_viz_payload(
    df_2nd: pd.DataFrame,
    par_split_full: pd.DataFrame,
    sample_per_cell: int = SAMPLE_PER_CELL,
) -> dict:
    points: dict[str, list[list[float]]] = {}
    if not df_2nd.empty:
        for (dist, surf), grp in df_2nd.groupby(["distance", "surface"], sort=False):
            key = _cell_key(int(dist), surf)
            sub = grp.dropna(subset=["race_t2nd_sec", "front_split_sec"])
            if len(sub) > sample_per_cell:
                sub = sub.sample(sample_per_cell, random_state=42)
            points[key] = (
                sub[["race_t2nd_sec", "front_split_sec"]]
                .astype(float)
                .round(4)
                .values.tolist()
            )

    cells = []
    for row in par_split_full.drop_duplicates(["distance", "surface"]).itertuples(index=False):
        cells.append(
            {
                "distance": int(row.distance),
                "surface": str(row.surface),
                "intercept": float(row.par_intercept),
                "slope": float(row.par_slope),
                "t2nd_ref": float(row.t2nd_ref),
                "n_fit": int(row.n_fit),
                "model": str(row.model),
                "key": _cell_key(int(row.distance), row.surface),
            }
        )

    return {
        "meta": {"n_cells": len(cells), "n_2nd_rows": int(len(df_2nd))},
        "cells": cells,
        "points": points,
    }


def build_overview_viz_payload(df_save: pd.DataFrame, bins: int = 50) -> dict:
    out: dict = {"meta": {"n_rows": int(len(df_save))}, "histograms": {}}
    if df_save.empty:
        return out

    for surf, col, title in [
        ("芝", "finish_time_sec", "斤量補正後タイム（芝）"),
        ("ダート", "finish_time_sec", "斤量補正後タイム（ダート）"),
    ]:
        sub = df_save[df_save["surface"] == surf][col].dropna()
        if len(sub):
            counts, edges = np.histogram(sub.to_numpy(), bins=bins)
            out["histograms"][surf] = {
                "title": title,
                "counts": counts.astype(int).tolist(),
                "edges": edges.round(3).tolist(),
            }

    track = df_save["track_dev_sec"].dropna()
    if len(track):
        counts, edges = np.histogram(track.to_numpy(), bins=bins)
        out["histograms"]["track_dev"] = {
            "title": "馬場乖離 track_dev_sec（日×コース）",
            "counts": counts.astype(int).tolist(),
            "edges": edges.round(3).tolist(),
        }
    return out


def _write_html(template: str, payload: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = template.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    out_path.write_text(html, encoding="utf-8")
    return out_path


WEIGHT_HTML = """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <title>斤量補正エクスプローラ — NB-01</title>
  <script src="../../static/plotly-2.35.2.min.js"></script>
  <style>
    body{margin:0;font-family:"Noto Sans JP",sans-serif;background:#f6f8fb;color:#1f2937}
    header{background:linear-gradient(135deg,#1e3a8a,#2563eb);color:#fff;padding:1.1rem 1.4rem}
    header h1{margin:0 0 .2rem;font-size:1.3rem} header p{margin:0;opacity:.92;font-size:.9rem}
    main{max-width:1200px;margin:0 auto;padding:1rem 1.2rem 2rem}
    .card{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:1rem;margin-bottom:1rem}
    .controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.8rem;align-items:end}
    label{font-size:.8rem;color:#6b7280;display:block;margin-bottom:.2rem}
    select{width:100%;padding:.5rem;border:1px solid #cbd5e1;border-radius:8px}
    .metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.6rem;margin-top:.6rem}
    .metric{background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:.55rem .65rem}
    .metric .k{font-size:.75rem;color:#6b7280}.metric .v{font-size:1.05rem;font-weight:700}
    #chart{width:100%;height:520px}
    .note{color:#6b7280;font-size:.84rem;margin-top:.5rem}
  </style>
</head>
<body>
<header><h1>斤量補正エクスプローラ</h1>
<p>レース内偏差: weight_dev_dm → time_dm（surface×distance×性別）</p></header>
<main>
  <section class="card controls">
    <div><label>馬場</label><select id="surface"></select></div>
    <div><label>性別</label><select id="sex"></select></div>
    <div><label>距離</label><select id="distance"></select></div>
  </section>
  <section class="card">
    <div class="metrics">
      <div class="metric"><div class="k">sec/kg</div><div class="v" id="mSec">—</div></div>
      <div class="metric"><div class="k">source</div><div class="v" id="mSource">—</div></div>
      <div class="metric"><div class="k">n_fit</div><div class="v" id="mN">—</div></div>
    </div>
    <p class="note">5着以内・レース内平均からの偏差で OLS。理論値は 2000m で 0.2 秒/kg。</p>
  </section>
  <section class="card"><div id="chart"></div></section>
</main>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
const DATA=JSON.parse(document.getElementById('data').textContent);
const cells=DATA.cells, points=DATA.points;
const surfaceEl=document.getElementById('surface'), sexEl=document.getElementById('sex'), distEl=document.getElementById('distance');
function uniq(a){return[...new Set(a)].sort()}
function numSort(a){return[...new Set(a)].sort((x,y)=>x-y)}
function fill(el,vals,fmt=v=>v){el.innerHTML='';vals.forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=fmt(v);el.appendChild(o)})}
function current(){return cells.find(c=>c.surface===surfaceEl.value&&c.sex_group===sexEl.value&&c.distance===Number(distEl.value))}
function refreshSex(){const s=uniq(cells.filter(c=>c.surface===surfaceEl.value).map(c=>c.sex_group));const p=sexEl.value;fill(sexEl,s);if(s.includes(p))sexEl.value=p;refreshDist()}
function refreshDist(){const d=numSort(cells.filter(c=>c.surface===surfaceEl.value&&c.sex_group===sexEl.value).map(c=>c.distance));const p=Number(distEl.value);fill(distEl,d,v=>`${v} m`);if(d.includes(p))distEl.value=String(p)}
function render(){
  const cell=current(); if(!cell){Plotly.purge('chart');return}
  document.getElementById('mSec').textContent=cell.sec_per_kg.toFixed(4);
  document.getElementById('mSource').textContent=cell.source;
  document.getElementById('mN').textContent=String(cell.n_fit);
  const pts=points[cell.key]||[];
  const xs=pts.map(p=>p[0]), ys=pts.map(p=>p[1]);
  let lineX=[], lineY=[];
  if(xs.length>1){
    const mn=Math.min(...xs), mx=Math.max(...xs);
  const beta=cell.sec_per_kg*(cell.distance/2000);
    lineX=[mn,mx]; lineY=[mn*beta,mx*beta];
  }
  Plotly.react('chart',[
    {x:xs,y:ys,mode:'markers',type:'scatter',name:`sample (n=${pts.length})`,marker:{size:6,color:'#2563eb',opacity:0.4}},
    {x:lineX,y:lineY,mode:'lines',type:'scatter',name:'推定傾き',line:{color:'#dc2626',width:2}}
  ],{title:`${cell.surface} ${cell.distance}m ${cell.sex_group}`,xaxis:{title:'weight_dev_dm (kg)'},yaxis:{title:'time_dm (秒)'},legend:{orientation:'h',y:-0.12},margin:{t:50,b:60,l:55,r:20}},{responsive:true});
}
function init(){
  fill(surfaceEl,uniq(cells.map(c=>c.surface)));
  const pref=cells.find(c=>c.surface==='芝'&&c.sex_group==='牡'&&c.distance===1800)||cells[0];
  surfaceEl.value=pref.surface; refreshSex(); sexEl.value=pref.sex_group; refreshDist(); distEl.value=String(pref.distance);
  surfaceEl.onchange=()=>{refreshSex();render()}; sexEl.onchange=()=>{refreshDist();render()}; distEl.onchange=render; render();
}
init();
</script></body></html>"""


TRACK_HTML = """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <title>馬場速度指数エクスプローラ — NB-01</title>
  <script src="../../static/plotly-2.35.2.min.js"></script>
  <style>
    body{margin:0;font-family:"Noto Sans JP",sans-serif;background:#f6f8fb;color:#1f2937}
    header{background:linear-gradient(135deg,#14532d,#16a34a);color:#fff;padding:1.1rem 1.4rem}
    header h1{margin:0 0 .2rem;font-size:1.3rem} header p{margin:0;opacity:.92;font-size:.9rem}
    main{max-width:1200px;margin:0 auto;padding:1rem 1.2rem 2rem}
    .card{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:1rem;margin-bottom:1rem}
    .controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:.8rem;align-items:end}
    label{font-size:.8rem;color:#6b7280;display:block;margin-bottom:.2rem}
    select{width:100%;padding:.5rem;border:1px solid #cbd5e1;border-radius:8px}
    #chart{width:100%;height:520px}
    .note{color:#6b7280;font-size:.84rem;margin-top:.5rem}
  </style>
</head>
<body>
<header><h1>馬場速度指数エクスプローラ</h1>
<p>日×会場×馬場の track_dev_sec / tsi_raw（正=遅い馬場）</p></header>
<main>
  <section class="card controls">
    <div><label>会場</label><select id="venue"></select></div>
    <div><label>馬場</label><select id="surface"></select></div>
    <div><label>指標</label><select id="metric"><option value="track_dev_sec">track_dev_sec</option><option value="tsi_raw">tsi_raw</option></select></div>
  </section>
  <section class="card"><div id="chart"></div>
  <p class="note">tsi_raw = −track_dev_sec。学習年（2020–2024）の基準2着タイムとの乖離を日次集計。</p></section>
</main>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
const DATA=JSON.parse(document.getElementById('data').textContent);
const series=DATA.series;
const venueEl=document.getElementById('venue'), surfaceEl=document.getElementById('surface'), metricEl=document.getElementById('metric');
function keysForVenue(v){return Object.keys(series).filter(k=>k.startsWith(v+'|'))}
function uniq(a){return[...new Set(a)].sort()}
function fill(el,vals){el.innerHTML='';vals.forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;el.appendChild(o)})}
function currentKey(){return venueEl.value+'|'+surfaceEl.value}
function refreshSurface(){const surfs=keysForVenue(venueEl.value).map(k=>k.split('|')[1]);const p=surfaceEl.value;fill(surfaceEl,surfs);if(surfs.includes(p))surfaceEl.value=p}
function render(){
  const rows=series[currentKey()]||[];
  const m=metricEl.value;
  const ys=rows.map(r=>r[m]).filter(v=>v!=null);
  const xs=rows.filter(r=>r[m]!=null).map(r=>r.date);
  const bar={x:xs,y:ys,mode:'lines+markers',type:'scatter',name:m,line:{color:m==='tsi_raw'?'#16a34a':'#dc2626',width:2},marker:{size:5}};
  Plotly.react('chart',[bar],{title:`${venueEl.value} ${surfaceEl.value}`,xaxis:{title:'日付'},yaxis:{title:m==='track_dev_sec'?'秒（正=遅い）':'tsi_raw'},margin:{t:50,b:60,l:55,r:20}},{responsive:true});
}
function init(){
  const venues=uniq(Object.keys(series).map(k=>k.split('|')[0]));
  fill(venueEl,venues);
  venueEl.value=venues.includes('東京')?'東京':venues[0];
  refreshSurface(); surfaceEl.value=surfaceEl.options[0]?.value||'';
  venueEl.onchange=()=>{refreshSurface();render()}; surfaceEl.onchange=render; metricEl.onchange=render; render();
}
init();
</script></body></html>"""


PAR_SPLIT_HTML = """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <title>基準前半スプリットエクスプローラ — NB-01</title>
  <script src="../../static/plotly-2.35.2.min.js"></script>
  <style>
    body{margin:0;font-family:"Noto Sans JP",sans-serif;background:#f6f8fb;color:#1f2937}
    header{background:linear-gradient(135deg,#7c2d12,#ea580c);color:#fff;padding:1.1rem 1.4rem}
    header h1{margin:0 0 .2rem;font-size:1.3rem} header p{margin:0;opacity:.92;font-size:.9rem}
    main{max-width:1200px;margin:0 auto;padding:1rem 1.2rem 2rem}
    .card{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:1rem;margin-bottom:1rem}
    .controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:.8rem;align-items:end}
    label{font-size:.8rem;color:#6b7280;display:block;margin-bottom:.2rem}
    select{width:100%;padding:.5rem;border:1px solid #cbd5e1;border-radius:8px}
    .metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.6rem;margin-top:.6rem}
    .metric{background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:.55rem .65rem}
    .metric .k{font-size:.75rem;color:#6b7280}.metric .v{font-size:1.05rem;font-weight:700}
    #chart{width:100%;height:520px}
    .note{color:#6b7280;font-size:.84rem;margin-top:.5rem}
  </style>
</head>
<body>
<header><h1>基準前半スプリット（par_front_split）エクスプローラ</h1>
<p>2着馬: race_t2nd_sec → front_split_sec の OLS（2020–2024 学習）</p></header>
<main>
  <section class="card controls">
    <div><label>馬場</label><select id="surface"></select></div>
    <div><label>距離</label><select id="distance"></select></div>
  </section>
  <section class="card">
    <div class="metrics">
      <div class="metric"><div class="k">intercept</div><div class="v" id="mI">—</div></div>
      <div class="metric"><div class="k">slope</div><div class="v" id="mS">—</div></div>
      <div class="metric"><div class="k">t2nd_ref</div><div class="v" id="mR">—</div></div>
      <div class="metric"><div class="k">n_fit</div><div class="v" id="mN">—</div></div>
    </div>
    <p class="note">par_front_split = intercept + slope × (race_t2nd_sec − t2nd_ref)</p>
  </section>
  <section class="card"><div id="chart"></div></section>
</main>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
const DATA=JSON.parse(document.getElementById('data').textContent);
const cells=DATA.cells, points=DATA.points;
const surfaceEl=document.getElementById('surface'), distEl=document.getElementById('distance');
function uniq(a){return[...new Set(a)].sort()}
function numSort(a){return[...new Set(a)].sort((x,y)=>x-y)}
function fill(el,vals,fmt=v=>v){el.innerHTML='';vals.forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=fmt(v);el.appendChild(o)})}
function current(){return cells.find(c=>c.surface===surfaceEl.value&&c.distance===Number(distEl.value))}
function refreshDist(){const d=numSort(cells.filter(c=>c.surface===surfaceEl.value).map(c=>c.distance));const p=Number(distEl.value);fill(distEl,d,v=>`${v} m`);if(d.includes(p))distEl.value=String(p)}
function render(){
  const cell=current(); if(!cell){Plotly.purge('chart');return}
  document.getElementById('mI').textContent=cell.intercept.toFixed(3);
  document.getElementById('mS').textContent=cell.slope.toFixed(4);
  document.getElementById('mR').textContent=cell.t2nd_ref.toFixed(2)+'s';
  document.getElementById('mN').textContent=String(cell.n_fit);
  const pts=points[cell.key]||[];
  const xs=pts.map(p=>p[0]);
  const x0=Math.min(...xs,...[cell.t2nd_ref-5,cell.t2nd_ref+5]);
  const x1=Math.max(...xs,...[cell.t2nd_ref-5,cell.t2nd_ref+5]);
  const lineX=[x0,x1];
  const lineY=lineX.map(x=>cell.intercept+cell.slope*(x-cell.t2nd_ref));
  Plotly.react('chart',[
    {x:pts.map(p=>p[0]),y:pts.map(p=>p[1]),mode:'markers',type:'scatter',name:`2着馬 (n=${pts.length})`,marker:{size:6,color:'#2563eb',opacity:0.4}},
    {x:lineX,y:lineY,mode:'lines',type:'scatter',name:'OLS 推定',line:{color:'#dc2626',width:2}}
  ],{title:`${cell.surface} ${cell.distance}m`,xaxis:{title:'race_t2nd_sec (秒)'},yaxis:{title:'front_split_sec (秒)'},legend:{orientation:'h',y:-0.12},margin:{t:50,b:60,l:55,r:20}},{responsive:true});
}
function init(){
  fill(surfaceEl,uniq(cells.map(c=>c.surface)));
  const pref=cells.find(c=>c.surface==='芝'&&c.distance===1600)||cells[0];
  surfaceEl.value=pref.surface; refreshDist(); distEl.value=String(pref.distance);
  surfaceEl.onchange=()=>{refreshDist();render()}; distEl.onchange=render; render();
}
init();
</script></body></html>"""


OVERVIEW_HTML = """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <title>データ分布概要 — NB-01</title>
  <script src="../../static/plotly-2.35.2.min.js"></script>
  <style>
    body{margin:0;font-family:"Noto Sans JP",sans-serif;background:#f6f8fb;color:#1f2937}
    header{background:linear-gradient(135deg,#4c1d95,#7c3aed);color:#fff;padding:1.1rem 1.4rem}
    header h1{margin:0;font-size:1.3rem}
    main{max-width:1200px;margin:0 auto;padding:1rem 1.2rem 2rem}
    .card{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:1rem;margin-bottom:1rem}
    #chart{width:100%;height:480px}
  </style>
</head>
<body>
<header><h1>megu_dataset 分布概要</h1></header>
<main>
  <section class="card">
    <label>パネル</label>
    <select id="panel"></select>
    <div id="chart"></div>
  </section>
</main>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
const DATA=JSON.parse(document.getElementById('data').textContent);
const hists=DATA.histograms;
const panelEl=document.getElementById('panel');
const keys=Object.keys(hists);
function fill(el,vals){el.innerHTML='';vals.forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=hists[v].title;el.appendChild(o)})}
function render(){
  const h=hists[panelEl.value]; if(!h)return;
  const centers=h.edges.slice(0,-1).map((e,i)=>(e+h.edges[i+1])/2);
  Plotly.react('chart',[{x:centers,y:h.counts,type:'bar',marker:{color:'#6366f1'}}],{title:h.title,xaxis:{title:'秒'},yaxis:{title:'件数'},margin:{t:50,b:50,l:50,r:20}},{responsive:true});
}
fill(panelEl,keys); panelEl.onchange=render; render();
</script></body></html>"""


def write_weight_explorer_html(df_fit: pd.DataFrame, coef_by_cell: pd.DataFrame, out_path: Path) -> Path:
    payload = build_weight_viz_payload(df_fit, coef_by_cell)
    return _write_html(WEIGHT_HTML, payload, out_path)


def write_track_explorer_html(day_course_tbl: pd.DataFrame, out_path: Path) -> Path:
    payload = build_track_viz_payload(day_course_tbl)
    return _write_html(TRACK_HTML, payload, out_path)


def write_par_split_explorer_html(
    df_2nd: pd.DataFrame, par_split_full: pd.DataFrame, out_path: Path
) -> Path:
    payload = build_par_split_viz_payload(df_2nd, par_split_full)
    return _write_html(PAR_SPLIT_HTML, payload, out_path)


def write_dataset_overview_html(df_save: pd.DataFrame, out_path: Path) -> Path:
    payload = build_overview_viz_payload(df_save)
    return _write_html(OVERVIEW_HTML, payload, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="NB-01 可視化 HTML を生成（保存済み parquet から）")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_OUT_DIR / "megu_dataset.parquet")
    parser.add_argument("--par-splits", type=Path, default=DEFAULT_OUT_DIR / "par_splits.parquet")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if args.dataset.exists():
        df = pd.read_parquet(args.dataset)
        write_dataset_overview_html(df, args.out_dir / "dataset_overview.html")
        print(f"wrote {args.out_dir / 'dataset_overview.html'}")

    if args.par_splits.exists():
        par = pd.read_parquet(args.par_splits)
        df2 = df[df["finish_pos"] == 2] if args.dataset.exists() else pd.DataFrame()
        if not df2.empty:
            write_par_split_explorer_html(df2, par, args.out_dir / "par_split_explorer.html")
            print(f"wrote {args.out_dir / 'par_split_explorer.html'}")


if __name__ == "__main__":
    main()
