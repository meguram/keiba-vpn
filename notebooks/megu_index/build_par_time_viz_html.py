#!/usr/bin/env python3
"""NB-02 par_time_class 推定結果のインタラクティブ HTML を生成する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAR_TIME = ROOT / "notebooks/megu_index/output/nb02/par_time_class.parquet"
DEFAULT_OUT = ROOT / "notebooks/megu_index/output/nb02/par_time_explorer.html"
DEFAULT_OVERVIEW_OUT = ROOT / "notebooks/megu_index/output/nb02/par_time_overview.html"
SAMPLE_PER_CELL = 400


def _cell_key(venue: str, surface: str, distance: int) -> str:
    return f"{venue}|{surface}|{int(distance)}"


def build_par_time_viz_payload(
    df_par_base: pd.DataFrame,
    par_time_class: pd.DataFrame,
    sample_per_cell: int = SAMPLE_PER_CELL,
) -> dict:
    points: dict[str, list[list[float]]] = {}
    if not df_par_base.empty:
        for (venue, surface, distance), grp in df_par_base.groupby(
            ["venue", "surface", "distance"], sort=False
        ):
            key = _cell_key(venue, surface, distance)
            sub = grp
            if len(sub) > sample_per_cell:
                sub = sub.sample(sample_per_cell, random_state=42)
            points[key] = (
                sub[["class_rank", "adjusted_time_sec"]]
                .astype(float)
                .round(4)
                .values.tolist()
            )

    cells = []
    for row in par_time_class.drop_duplicates(["venue", "surface", "distance"]).itertuples(index=False):
        par_rows = par_time_class[
            (par_time_class["venue"] == row.venue)
            & (par_time_class["surface"] == row.surface)
            & (par_time_class["distance"] == row.distance)
        ].sort_values("class_rank")
        par_curve = {
            int(r.class_rank): float(r.par_time_sec)
            for r in par_rows.itertuples(index=False)
        }
        pt2 = par_curve.get(2)
        pt7 = par_curve.get(7)
        diff_27 = (pt2 - pt7) if pt2 is not None and pt7 is not None else None
        cells.append(
            {
                "venue": str(row.venue),
                "surface": str(row.surface),
                "distance": int(row.distance),
                "alpha": float(row.alpha),
                "beta": float(row.beta),
                "n_fit": int(row.n_fit),
                "source": str(row.source),
                "diff_rank2_rank7": diff_27,
                "par_curve": par_curve,
                "key": _cell_key(row.venue, row.surface, row.distance),
            }
        )

    return {
        "meta": {"n_cells": len(cells), "n_rows": int(len(df_par_base))},
        "cells": cells,
        "points": points,
    }


PAR_TIME_HTML = """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <title>par_time エクスプローラ — NB-02</title>
  <script src="../../static/plotly-2.35.2.min.js"></script>
  <style>
    body { margin:0; font-family:"Noto Sans JP",sans-serif; background:#f6f8fb; color:#1f2937; }
    header { background:linear-gradient(135deg,#14532d,#16a34a); color:#fff; padding:1.1rem 1.4rem; }
    header h1 { margin:0 0 .2rem; font-size:1.3rem; }
    header p { margin:0; opacity:.92; font-size:.9rem; }
    main { max-width:1200px; margin:0 auto; padding:1rem 1.2rem 2rem; }
    .card { background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:1rem; margin-bottom:1rem; }
    .controls { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:.8rem; align-items:end; }
    label { font-size:.8rem; color:#6b7280; display:block; margin-bottom:.2rem; }
    select { width:100%; padding:.5rem; border:1px solid #cbd5e1; border-radius:8px; }
    .metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:.6rem; margin-top:.6rem; }
    .metric { background:#f8fafc; border:1px solid #e5e7eb; border-radius:8px; padding:.55rem .65rem; }
    .metric .k { font-size:.75rem; color:#6b7280; }
    .metric .v { font-size:1.05rem; font-weight:700; }
    table { width:100%; border-collapse:collapse; font-size:.88rem; margin-top:.6rem; }
    th,td { border:1px solid #e5e7eb; padding:.4rem .5rem; text-align:right; }
    th { background:#f1f5f9; text-align:center; }
    td:first-child,th:first-child { text-align:left; }
    #chart { width:100%; height:560px; }
    .note { color:#6b7280; font-size:.84rem; margin-top:.5rem; }
  </style>
</head>
<body>
<header><h1>par_time クラス回帰エクスプローラ</h1>
<p>2着馬の class_rank → adjusted_time_sec と推定 par_time 直線（venue×surface×distance）</p></header>
<main>
  <section class="card controls">
    <div><label>会場</label><select id="venue"></select></div>
    <div><label>コース</label><select id="surface"></select></div>
    <div><label>距離</label><select id="distance"></select></div>
  </section>
  <section class="card">
    <div class="metrics">
      <div class="metric"><div class="k">alpha</div><div class="v" id="mAlpha">—</div></div>
      <div class="metric"><div class="k">beta</div><div class="v" id="mBeta">—</div></div>
      <div class="metric"><div class="k">rank2−rank7</div><div class="v" id="mDiff">—</div></div>
      <div class="metric"><div class="k">source</div><div class="v" id="mSource">—</div></div>
      <div class="metric"><div class="k">n_fit</div><div class="v" id="mNFit">—</div></div>
    </div>
    <table><thead><tr><th>class_rank</th><th>クラス</th><th>par_time_sec</th></tr></thead><tbody id="parTable"></tbody></table>
    <p class="note">モデル: <code>par_time = alpha + beta × class_rank</code>。beta は負が正常。rank=2 が指数50の基準。</p>
  </section>
  <section class="card"><div id="chart"></div></section>
</main>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const cells = DATA.cells, points = DATA.points;
const RANK_LABEL = {1:'未勝利',2:'1勝',3:'2勝',4:'3勝',5:'OP',6:'重賞',7:'G1'};
const venueEl = document.getElementById('venue');
const surfaceEl = document.getElementById('surface');
const distanceEl = document.getElementById('distance');

function uniq(a){ return [...new Set(a)].sort(); }
function numSort(a){ return [...new Set(a)].sort((x,y)=>x-y); }
function fill(el, vals, fmt=v=>v){ el.innerHTML=''; vals.forEach(v=>{ const o=document.createElement('option'); o.value=v; o.textContent=fmt(v); el.appendChild(o); }); }
function current(){ return cells.find(c=>c.venue===venueEl.value && c.surface===surfaceEl.value && c.distance===Number(distanceEl.value)); }
function refreshSurface(){ const s=uniq(cells.filter(c=>c.venue===venueEl.value).map(c=>c.surface)); const p=surfaceEl.value; fill(surfaceEl,s); if(s.includes(p)) surfaceEl.value=p; refreshDistance(); }
function refreshDistance(){ const d=numSort(cells.filter(c=>c.venue===venueEl.value && c.surface===surfaceEl.value).map(c=>c.distance)); const p=Number(distanceEl.value); fill(distanceEl,d,v=>`${v} m`); if(d.includes(p)) distanceEl.value=String(p); }
function render(){
  const cell = current();
  if(!cell){ Plotly.purge('chart'); return; }
  document.getElementById('mAlpha').textContent = cell.alpha.toFixed(3);
  document.getElementById('mBeta').textContent = cell.beta.toFixed(4);
  document.getElementById('mDiff').textContent = cell.diff_rank2_rank7==null?'—':cell.diff_rank2_rank7.toFixed(2)+'s';
  document.getElementById('mSource').textContent = cell.source;
  document.getElementById('mNFit').textContent = String(cell.n_fit);
  const tbody = document.getElementById('parTable'); tbody.innerHTML='';
  Object.keys(cell.par_curve).map(Number).sort((a,b)=>a-b).forEach(r=>{
    const tr=document.createElement('tr');
    tr.innerHTML = `<td>${r}</td><td>${RANK_LABEL[r]||''}</td><td>${cell.par_curve[r].toFixed(3)}</td>`;
    tbody.appendChild(tr);
  });
  const pts = points[cell.key] || [];
  const ranks = Object.keys(cell.par_curve).map(Number).sort((a,b)=>a-b);
  const lineX = ranks, lineY = ranks.map(r=>cell.par_curve[r]);
  const traces = [
    { x: pts.map(p=>p[0]), y: pts.map(p=>p[1]), mode:'markers', type:'scatter', name:`2着馬 sample (n=${pts.length})`,
      marker:{size:7,color:'#2563eb',opacity:0.45}, hovertemplate:'rank=%{x}<br>time=%{y:.2f}s<extra></extra>' },
    { x: lineX, y: lineY, mode:'lines+markers', type:'scatter', name:'par_time 推定',
      line:{color:'#dc2626',width:2.5}, marker:{size:9,color:'#dc2626'}, hovertemplate:'rank=%{x}<br>par=%{y:.2f}s<extra></extra>' }
  ];
  Plotly.react('chart', traces, {
    title: `${cell.venue} ${cell.surface} ${cell.distance}m`,
    xaxis:{ title:'class_rank', dtick:1, range:[0.5,7.5] },
    yaxis:{ title:'タイム (秒)' },
    legend:{orientation:'h', y:-0.15},
    margin:{t:50,b:70,l:55,r:20}
  }, {responsive:true});
}
function init(){
  fill(venueEl, uniq(cells.map(c=>c.venue)));
  const pref = cells.find(c=>c.venue==='東京'&&c.surface==='芝'&&c.distance===1600) || cells[0];
  venueEl.value = pref.venue; refreshSurface(); surfaceEl.value = pref.surface; refreshDistance(); distanceEl.value = String(pref.distance);
  venueEl.onchange = ()=>{ refreshSurface(); render(); };
  surfaceEl.onchange = ()=>{ refreshDistance(); render(); };
  distanceEl.onchange = render;
  render();
}
init();
</script>
</body></html>
"""


OVERVIEW_HTML = """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <title>par_time 品質概要 — NB-02</title>
  <script src="../../static/plotly-2.35.2.min.js"></script>
  <style>
    body{margin:0;font-family:"Noto Sans JP",sans-serif;background:#f6f8fb;color:#1f2937}
    header{background:linear-gradient(135deg,#7c2d12,#ea580c);color:#fff;padding:1.1rem 1.4rem}
    header h1{margin:0 0 .2rem;font-size:1.3rem}
    main{max-width:1200px;margin:0 auto;padding:1rem 1.2rem 2rem}
    .card{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:1rem;margin-bottom:1rem}
    .metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.6rem}
    .metric{background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:.55rem .65rem}
    .metric .k{font-size:.75rem;color:#6b7280}.metric .v{font-size:1.05rem;font-weight:700}
    #chart1,#chart2{height:420px}
  </style>
</head>
<body>
<header><h1>par_time 品質概要</h1><p>全セルの rank2−rank7 差と beta 分布</p></header>
<main>
  <section class="card metrics" id="metrics"></section>
  <section class="card"><div id="chart1"></div></section>
  <section class="card"><div id="chart2"></div></section>
</main>
<script id="data" type="application/json">__PAYLOAD__</script>
<script>
const cells = JSON.parse(document.getElementById('data').textContent).cells;
const diffs = cells.map(c=>c.diff_rank2_rank7).filter(v=>v!=null);
const betas = cells.map(c=>c.beta);
function stat(arr){ const s=[...arr].sort((a,b)=>a-b); const q=p=>s[Math.floor((s.length-1)*p)]; return {min:s[0],p25:q(0.25),med:q(0.5),p75:q(0.75),max:s[s.length-1],mean:arr.reduce((a,b)=>a+b,0)/arr.length}; }
const ds = stat(diffs);
document.getElementById('metrics').innerHTML = [
  ['セル数', cells.length], ['beta>0', cells.filter(c=>c.beta>0).length],
  ['diff min', ds.min.toFixed(2)+'s'], ['diff mean', ds.mean.toFixed(2)+'s'], ['diff max', ds.max.toFixed(2)+'s']
].map(([k,v])=>`<div class="metric"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
const sorted = [...cells].sort((a,b)=>(b.diff_rank2_rank7||0)-(a.diff_rank2_rank7||0));
Plotly.newPlot('chart1', [{
  x: sorted.map(c=>`${c.venue} ${c.surface} ${c.distance}`),
  y: sorted.map(c=>c.diff_rank2_rank7),
  type:'bar', marker:{color:'#2563eb'}
}], {title:'rank2 − rank7 par_time差 (秒) セル別', xaxis:{tickangle:-45}, yaxis:{title:'秒'}}, {responsive:true});
Plotly.newPlot('chart2', [{
  x: betas, type:'histogram', nbinsx:25, marker:{color:'#16a34a'}
}], {title:'beta 分布（全セル）', xaxis:{title:'beta (秒/rank)'}, yaxis:{title:'セル数'}}, {responsive:true});
</script>
</body></html>
"""


def render_par_time_viz_html(payload: dict) -> str:
    return PAR_TIME_HTML.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def render_par_time_overview_html(payload: dict) -> str:
    return OVERVIEW_HTML.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def build_par_time_viz_html(
    df_par_base: pd.DataFrame,
    par_time_class: pd.DataFrame,
    out_path: Path = DEFAULT_OUT,
) -> Path:
    payload = build_par_time_viz_payload(df_par_base, par_time_class)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_par_time_viz_html(payload), encoding="utf-8")
    return out_path


def build_par_time_overview_html(
    par_time_class: pd.DataFrame,
    out_path: Path = DEFAULT_OVERVIEW_OUT,
) -> Path:
    empty = pd.DataFrame(columns=["venue", "surface", "distance", "class_rank", "adjusted_time_sec"])
    payload = build_par_time_viz_payload(empty, par_time_class)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_par_time_overview_html(payload), encoding="utf-8")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--par-base", type=Path, required=True)
    ap.add_argument("--par-time", type=Path, default=DEFAULT_PAR_TIME)
    ap.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--overview-out", type=Path, default=DEFAULT_OVERVIEW_OUT)
    args = ap.parse_args()
    base = pd.read_parquet(args.par_base)
    par_time = pd.read_parquet(args.par_time)
    p1 = build_par_time_viz_html(base, par_time, args.out)
    p2 = build_par_time_overview_html(par_time, args.overview_out)
    print(f"OK: {p1}\nOK: {p2}")


if __name__ == "__main__":
    main()
