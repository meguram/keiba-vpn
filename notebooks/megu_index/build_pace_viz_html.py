#!/usr/bin/env python3
"""NB-02 Δpace 推定結果のインタラクティブ HTML を生成する。

会場・コース（芝/ダート）・距離を選び、ペース偏差と走破タイムの関係を可視化する。

例::

  python3 notebooks/megu_index/build_pace_viz_html.py
  python3 notebooks/megu_index/build_pace_viz_html.py \\
    --dataset notebooks/megu_index/output/nb01/megu_dataset.parquet \\
    --coeff notebooks/megu_index/output/nb02/coeff_pace.parquet \\
    -o notebooks/megu_index/output/nb02/pace_coeff_explorer.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "notebooks/megu_index/output/nb01/megu_dataset.parquet"
DEFAULT_COEFF = ROOT / "notebooks/megu_index/output/nb02/coeff_pace.parquet"
DEFAULT_OUT = ROOT / "notebooks/megu_index/output/nb02/pace_coeff_explorer.html"
TRAIN_YEAR_MAX = 2024
HORSE_SAMPLE_PER_CELL = 800


def _cell_key(venue: str, surface: str, distance: int) -> str:
    return f"{venue}|{surface}|{int(distance)}"


def _prepare_front_split_dev(df: pd.DataFrame) -> pd.Series:
    if "front_split_dev" in df.columns and df["front_split_dev"].notna().any():
        return df["front_split_dev"]
    if "par_front_split_sec" in df.columns:
        return (df["front_split_sec"] - df["par_front_split_sec"]).fillna(0.0)
    return pd.Series(0.0, index=df.index)


def build_pace_viz_payload(
    dataset_path: Path | None = None,
    coeff_path: Path | None = None,
    df_train: pd.DataFrame | None = None,
    coeff_df: pd.DataFrame | None = None,
    train_year_max: int = TRAIN_YEAR_MAX,
    horse_sample_per_cell: int = HORSE_SAMPLE_PER_CELL,
) -> dict:
    if df_train is None:
        if dataset_path is None:
            raise ValueError("dataset_path or df_train required")
        df = pd.read_parquet(dataset_path)
    else:
        df = df_train.copy()
    if coeff_df is None:
        if coeff_path is None:
            raise ValueError("coeff_path or coeff_df required")
        coeff = pd.read_parquet(coeff_path)
    else:
        coeff = coeff_df.copy()

    df = df[df["year"] <= train_year_max].copy()
    df["front_split_dev"] = _prepare_front_split_dev(df)
    pace_fit = df[
        df["front_split_sec"].notna() & df["adjusted_time_sec"].notna()
    ].copy()

    race_df = (
        pace_fit.groupby(["venue", "surface", "distance", "race_id"], as_index=False)
        .agg(
            front_split_dev=("front_split_dev", "mean"),
            adjusted_time_sec=("adjusted_time_sec", "mean"),
        )
    )

    races: dict[str, list[list[float]]] = {}
    horses: dict[str, list[list[float]]] = {}
    for (venue, surface, distance), grp in race_df.groupby(
        ["venue", "surface", "distance"], sort=False
    ):
        key = _cell_key(venue, surface, distance)
        races[key] = (
            grp[["front_split_dev", "adjusted_time_sec"]]
            .astype(float)
            .round(4)
            .values.tolist()
        )

        horse_grp = pace_fit[
            (pace_fit["venue"] == venue)
            & (pace_fit["surface"] == surface)
            & (pace_fit["distance"] == distance)
        ]
        if len(horse_grp) > horse_sample_per_cell:
            horse_grp = horse_grp.sample(horse_sample_per_cell, random_state=42)
        horses[key] = (
            horse_grp[["front_split_dev", "adjusted_time_sec"]]
            .astype(float)
            .round(4)
            .values.tolist()
        )

    cells = []
    for row in coeff.itertuples(index=False):
        cells.append(
            {
                "venue": str(row.venue),
                "surface": str(row.surface),
                "distance": int(row.distance),
                "coeff_pace": float(row.coeff_pace),
                "n_fit": int(row.n_fit),
                "source": str(row.source),
                "key": _cell_key(row.venue, row.surface, row.distance),
            }
        )

    return {
        "meta": {
            "train_year_max": train_year_max,
            "n_cells": len(cells),
            "n_races": int(len(race_df)),
        },
        "cells": cells,
        "races": races,
        "horses": horses,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Δpace 係数エクスプローラ — NB-02</title>
  <script src="../../static/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      --bg: #f6f8fb;
      --card: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --accent: #2563eb;
      --line: #e5e7eb;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Noto Sans JP", "Hiragino Sans", "Yu Gothic UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }
    header {
      background: linear-gradient(135deg, #1e3a5f, #2563eb);
      color: #fff;
      padding: 1.25rem 1.5rem;
    }
    header h1 { margin: 0 0 0.25rem; font-size: 1.35rem; }
    header p { margin: 0; opacity: 0.9; font-size: 0.92rem; }
    main { max-width: 1200px; margin: 0 auto; padding: 1rem 1.25rem 2rem; }
    .controls, .info, .chart-card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 1rem 1.1rem;
      margin-bottom: 1rem;
      box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .controls {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 0.85rem 1rem;
      align-items: end;
    }
    label { display: block; font-size: 0.82rem; color: var(--muted); margin-bottom: 0.25rem; }
    select, button {
      width: 100%;
      padding: 0.55rem 0.65rem;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      font-size: 0.95rem;
      background: #fff;
    }
    button {
      background: var(--accent);
      color: #fff;
      border: none;
      cursor: pointer;
      font-weight: 600;
    }
    button:hover { filter: brightness(1.05); }
    .info-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 0.75rem;
    }
    .metric {
      background: #f8fafc;
      border-radius: 8px;
      padding: 0.65rem 0.75rem;
      border: 1px solid var(--line);
    }
    .metric .k { font-size: 0.78rem; color: var(--muted); }
    .metric .v { font-size: 1.1rem; font-weight: 700; margin-top: 0.15rem; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
      margin-top: 0.75rem;
    }
    th, td {
      border: 1px solid var(--line);
      padding: 0.45rem 0.55rem;
      text-align: right;
    }
    th { background: #f1f5f9; text-align: center; }
    td:first-child, th:first-child { text-align: left; }
    #chart { width: 100%; height: 560px; }
    .note { color: var(--muted); font-size: 0.85rem; margin-top: 0.5rem; }
    .toggle-row { display: flex; align-items: center; gap: 0.5rem; margin-top: 0.35rem; }
    .toggle-row input { width: auto; }
  </style>
</head>
<body>
  <header>
    <h1>Δpace 係数エクスプローラ</h1>
    <p>会場・コース・距離を選び、ペース偏差（front_split_dev）と走破タイムの関係を確認します。</p>
  </header>
  <main>
    <section class="controls">
      <div>
        <label for="venue">会場</label>
        <select id="venue"></select>
      </div>
      <div>
        <label for="surface">コース</label>
        <select id="surface"></select>
      </div>
      <div>
        <label for="distance">距離 (m)</label>
        <select id="distance"></select>
      </div>
      <div>
        <label>&nbsp;</label>
        <button id="plotBtn" type="button">再描画</button>
      </div>
    </section>

    <section class="info">
      <div class="info-grid">
        <div class="metric"><div class="k">coeff_pace</div><div class="v" id="mCoeff">—</div></div>
        <div class="metric"><div class="k">推定ソース</div><div class="v" id="mSource">—</div></div>
        <div class="metric"><div class="k">学習レース数 n_fit</div><div class="v" id="mNFit">—</div></div>
        <div class="metric"><div class="k">散布図レース数</div><div class="v" id="mNRace">—</div></div>
      </div>
      <div class="toggle-row">
        <input type="checkbox" id="showHorses" />
        <label for="showHorses" style="margin:0;color:var(--text);">馬単位を薄く重ねる（セル内最大サンプル）</label>
      </div>
      <table id="corrTable">
        <thead>
          <tr><th>ペース状況</th><th>dev (秒)</th><th>Δpace (秒)</th><th>補正の意味</th></tr>
        </thead>
        <tbody></tbody>
      </table>
      <p class="note">
        適用式: <code>Δpace = coeff_pace × front_split_dev</code> /
        <code>corrected_time = adjusted_time_sec − Δpace</code>。
        dev &lt; 0 は標準より速い前半、dev &gt; 0 は遅い前半。
        学習データ: year ≤ __TRAIN_YEAR__。
      </p>
    </section>

    <section class="chart-card">
      <div id="chart"></div>
    </section>
  </main>

  <script id="pace-data" type="application/json">__PAYLOAD_JSON__</script>
  <script>
    const DATA = JSON.parse(document.getElementById('pace-data').textContent);
    const cells = DATA.cells;
    const races = DATA.races;
    const horses = DATA.horses;

    const venueEl = document.getElementById('venue');
    const surfaceEl = document.getElementById('surface');
    const distanceEl = document.getElementById('distance');
    const showHorsesEl = document.getElementById('showHorses');

    function uniqueSorted(arr) {
      return [...new Set(arr)].sort((a, b) => (a > b ? 1 : a < b ? -1 : 0));
    }

    function numericSort(arr) {
      return [...new Set(arr)].sort((a, b) => a - b);
    }

    function currentCell() {
      const venue = venueEl.value;
      const surface = surfaceEl.value;
      const distance = Number(distanceEl.value);
      return cells.find(c => c.venue === venue && c.surface === surface && c.distance === distance) || null;
    }

    function fillSelect(el, values, formatter = v => v) {
      el.innerHTML = '';
      values.forEach(v => {
        const opt = document.createElement('option');
        opt.value = v;
        opt.textContent = formatter(v);
        el.appendChild(opt);
      });
    }

    function refreshSurfaceOptions() {
      const venue = venueEl.value;
      const surfaces = uniqueSorted(cells.filter(c => c.venue === venue).map(c => c.surface));
      const prev = surfaceEl.value;
      fillSelect(surfaceEl, surfaces);
      if (surfaces.includes(prev)) surfaceEl.value = prev;
      refreshDistanceOptions();
    }

    function refreshDistanceOptions() {
      const venue = venueEl.value;
      const surface = surfaceEl.value;
      const distances = numericSort(
        cells.filter(c => c.venue === venue && c.surface === surface).map(c => c.distance)
      );
      const prev = Number(distanceEl.value);
      fillSelect(distanceEl, distances, d => `${d} m`);
      if (distances.includes(prev)) distanceEl.value = String(prev);
    }

    function regressionLine(points, slope) {
      if (!points.length) return { x: [], y: [] };
      const xs = points.map(p => p[0]);
      const ys = points.map(p => p[1]);
      const xMean = xs.reduce((a, b) => a + b, 0) / xs.length;
      const yMean = ys.reduce((a, b) => a + b, 0) / ys.length;
      const intercept = yMean - slope * xMean;
      const xMin = Math.min(...xs);
      const xMax = Math.max(...xs);
      const pad = (xMax - xMin) * 0.05 || 0.2;
      const xLine = [xMin - pad, xMax + pad];
      return {
        x: xLine,
        y: xLine.map(x => intercept + slope * x),
        intercept,
      };
    }

    function updateInfo(cell, racePts) {
      document.getElementById('mCoeff').textContent = cell ? cell.coeff_pace.toFixed(3) : '—';
      document.getElementById('mSource').textContent = cell ? cell.source : '—';
      document.getElementById('mNFit').textContent = cell ? String(cell.n_fit) : '—';
      document.getElementById('mNRace').textContent = String(racePts.length);

      const tbody = document.querySelector('#corrTable tbody');
      tbody.innerHTML = '';
      if (!cell) return;
      const coeff = cell.coeff_pace;
      const rows = [
        [-2, '標準より2秒速い'],
        [-1, '標準より1秒速い'],
        [0, '標準ペース'],
        [1, '標準より1秒遅い'],
        [2, '標準より2秒遅い'],
      ];
      rows.forEach(([dev, label]) => {
        const dPace = coeff * dev;
        let meaning = '補正なし';
        if (dPace < 0) meaning = `corrected は adjusted より ${Math.abs(dPace).toFixed(2)}秒 速く`;
        if (dPace > 0) meaning = `corrected は adjusted より ${dPace.toFixed(2)}秒 遅く`;
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${label}</td><td>${dev > 0 ? '+' : ''}${dev}</td><td>${dPace >= 0 ? '+' : ''}${dPace.toFixed(2)}</td><td>${meaning}</td>`;
        tbody.appendChild(tr);
      });
    }

    function renderPlot() {
      const cell = currentCell();
      if (!cell) {
        Plotly.purge('chart');
        updateInfo(null, []);
        return;
      }
      const key = cell.key;
      const racePts = races[key] || [];
      const horsePts = horses[key] || [];
      const traces = [];

      if (showHorsesEl.checked && horsePts.length) {
        traces.push({
          x: horsePts.map(p => p[0]),
          y: horsePts.map(p => p[1]),
          mode: 'markers',
          type: 'scatter',
          name: `馬単位 (sample ${horsePts.length})`,
          marker: { size: 4, color: 'rgba(120,120,120,0.18)' },
          hovertemplate: 'dev=%{x:.2f}s<br>time=%{y:.2f}s<extra>馬</extra>',
        });
      }

      traces.push({
        x: racePts.map(p => p[0]),
        y: racePts.map(p => p[1]),
        mode: 'markers',
        type: 'scatter',
        name: `レース平均 (n=${racePts.length})`,
        marker: { size: 8, color: '#2563eb', opacity: 0.55, line: { width: 0.5, color: '#fff' } },
        hovertemplate: 'dev=%{x:.2f}s<br>time=%{y:.2f}s<extra>レース</extra>',
      });

      const line = regressionLine(racePts, cell.coeff_pace);
      if (line.x.length) {
        traces.push({
          x: line.x,
          y: line.y,
          mode: 'lines',
          type: 'scatter',
          name: `推定直線 (slope=${cell.coeff_pace.toFixed(3)})`,
          line: { color: '#dc2626', width: 2.5 },
          hoverinfo: 'skip',
        });
      }

      const title = `${cell.venue} ${cell.surface} ${cell.distance}m — coeff=${cell.coeff_pace.toFixed(3)} (${cell.source})`;
      const layout = {
        title,
        xaxis: {
          title: 'front_split_dev (秒)　負=標準より速い / 正=標準より遅い',
          zeroline: true,
          zerolinecolor: '#9ca3af',
          zerolinewidth: 1,
        },
        yaxis: { title: 'adjusted_time_sec (秒)' },
        legend: { orientation: 'h', y: -0.18 },
        margin: { t: 60, r: 20, b: 80, l: 60 },
        hovermode: 'closest',
      };

      Plotly.react('chart', traces, layout, { responsive: true, displayModeBar: true });
      updateInfo(cell, racePts);
    }

    function init() {
      const venues = uniqueSorted(cells.map(c => c.venue));
      fillSelect(venueEl, venues);
      venueEl.addEventListener('change', () => { refreshSurfaceOptions(); renderPlot(); });
      surfaceEl.addEventListener('change', () => { refreshDistanceOptions(); renderPlot(); });
      distanceEl.addEventListener('change', renderPlot);
      showHorsesEl.addEventListener('change', renderPlot);
      document.getElementById('plotBtn').addEventListener('click', renderPlot);

      // 初期値: 東京芝1600 があればそれ、なければ先頭セル
      const preferred = cells.find(c => c.venue === '東京' && c.surface === '芝' && c.distance === 1600) || cells[0];
      venueEl.value = preferred.venue;
      refreshSurfaceOptions();
      surfaceEl.value = preferred.surface;
      refreshDistanceOptions();
      distanceEl.value = String(preferred.distance);
      renderPlot();
    }

    init();
  </script>
</body>
</html>
"""


def render_pace_viz_html(payload: dict) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__PAYLOAD_JSON__", payload_json)
    return html.replace("__TRAIN_YEAR__", str(payload["meta"]["train_year_max"]))


def build_pace_viz_html(
    dataset_path: Path | None = None,
    coeff_path: Path | None = None,
    out_path: Path = DEFAULT_OUT,
    train_year_max: int = TRAIN_YEAR_MAX,
    df_train: pd.DataFrame | None = None,
    coeff_df: pd.DataFrame | None = None,
) -> Path:
    payload = build_pace_viz_payload(
        dataset_path=dataset_path,
        coeff_path=coeff_path,
        df_train=df_train,
        coeff_df=coeff_df,
        train_year_max=train_year_max,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_pace_viz_html(payload), encoding="utf-8")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="NB-02 Δpace インタラクティブ HTML を生成")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--coeff", type=Path, default=DEFAULT_COEFF)
    ap.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--train-year-max", type=int, default=TRAIN_YEAR_MAX)
    args = ap.parse_args()

    out = build_pace_viz_html(
        dataset_path=args.dataset,
        coeff_path=args.coeff,
        out_path=args.out,
        train_year_max=args.train_year_max,
    )
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"OK: {out} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
