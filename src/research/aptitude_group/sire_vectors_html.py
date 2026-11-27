"""種牡馬の血統ベクトルの 1 枚の HTML（docs/html/operations/final/sire_vectors.html）。外部の CDN を使わない。

    python -m src.research.aptitude_group.sire_vectors_html   # 先に sire_vectors（と detailed_lines）を流す

- 地図は主の 6 本（距離・芝ダ・道悪・前の位置・TS・ギア。版の尺度の z）で作る。PCA（バイプロットの矢印つき）と t-SNE（Python で計算して埋め込む）。
- 非共通の地図: 6 本 − 同じ大系統（netkeiba の「○○系」・lines.py）の走数の重み付き平均。
- 地図のゆがみ: 6 本の元の距離と、地図の上の距離の相関（Spearman）を 1 行で出す。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.manifold import TSNE

from src.research.aptitude_group import common
from src.research.aptitude_group.lines import line_labels, short

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/html/operations/final/sire_vectors.html"
MAIN6 = ["dist", "surf", "wet", "pos", "ts", "gear"]
LABEL = {"dist": "距離（長い方へ）", "surf": "ダート寄り", "wet_T": "芝の道悪", "wet_D": "ダートの道悪", "pos": "前の位置",
         "ts": "トップスピード", "gear": "ギアチェンジ", "steep": "急坂", "small": "小回り", "straight": "長い直線",
         "left": "左回り", "yoshiba": "洋芝", "kyoto": "京都", "wet": "道悪", "course": "コースの偏り"}


def maps(X: np.ndarray) -> dict:
    """PCA（2 次元・矢印）と t-SNE の座標と、それぞれの地図のゆがみ（元の距離との Spearman の相関）。"""
    Xc = X - X.mean(0)
    U, S, Wt = np.linalg.svd(Xc, full_matrices=False)
    pc = Xc @ Wt[:2].T
    ts = TSNE(n_components=2, perplexity=min(15, max(5, len(X) // 6)), random_state=20261127, init="pca", learning_rate="auto").fit_transform(X)
    iu = np.triu_indices(len(X), 1)
    D0 = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))[iu]

    def rho(Y):
        return round(float(spearmanr(D0, np.sqrt(((Y[:, None] - Y[None]) ** 2).sum(-1))[iu]).statistic), 3)
    return {"pc": pc.round(3).tolist(), "ts": ts.round(3).tolist(), "load": Wt[:2].round(3).tolist(),
            "var": (S[:2] ** 2 / (S ** 2).sum()).round(3).tolist(), "rho_pc": rho(pc), "rho_ts": rho(ts)}


PED5 = Path("/home/jovyan/work/keiba-vpn/data/local/horse_pedigree_5gen")


def line_of(h: str, lab: dict[str, str]) -> str | None:
    """札の表（sire_line_origin）に無い種牡馬は、5 代の血統の父系の祖先を上へたどり、最初に札のある祖先の札を使う（critic 3 回目の 2）。"""
    if h in lab:
        return lab[h]
    p = PED5 / h[:4] / f"{h}.json"
    if not p.exists():
        return None
    try:
        anc = json.loads(p.read_text(encoding="utf-8")).get("ancestors", [])
    except Exception:
        return None
    for g in range(1, 6):
        for a in anc:
            if a.get("generation") == g and a.get("position") == 0 and a.get("horse_id") in lab:
                return lab[a["horse_id"]]
    return None


def side_payload(rows: list[dict], dims: list[str], lab: dict[str, str], dl: dict[str, tuple[str, str]]) -> dict:
    ix = {d: i for i, d in enumerate(dims)}
    for r in rows:
        v = r["v"]
        r["m6"] = [round((v[ix["wet_T"]] + v[ix["wet_D"]]) / 2, 3) if d == "wet" else v[ix[d]] for d in MAIN6]
        lo = line_of(r["id"], lab)
        r["line"] = short(lo) if lo else "札なし"
        if r["id"] in dl:
            r["dl"], r["dl_name"] = dl[r["id"]]
    X = np.array([r["m6"] for r in rows], float)
    w = np.array([r["runs"] if r.get("core", True) else 0.0 for r in rows], float) + 1e-9   # 系統の平均は中核（300 走以上）で作る
    L = pd.Series([r["line"] for r in rows])
    mean = pd.DataFrame(X * w[:, None]).groupby(L).sum().div(pd.Series(w).groupby(L).sum(), axis=0)
    gm = (X * w[:, None]).sum(0) / w.sum()
    mean.loc["札なし"] = gm                          # 札の無い種牡馬は、ひとまとめの平均ではなく全体の平均を引く（critic 3 回目の 2）
    Xn = X - mean.loc[L].to_numpy()
    for r, xn in zip(rows, Xn):
        r["n6"] = [round(float(x), 3) for x in xn]
    return {"rows": rows, "common": maps(X), "noncommon": maps(Xn)}


def add_boot(payload_side: dict, ids: np.ndarray, B: np.ndarray) -> None:
    """ブートストラップの各回のベクトルを、同じ PCA の軸に写して 2 次元の共分散（1σ の楕円）にし、各回の 6 本も埋め込む（距離の区間用）。"""
    rows = payload_side["rows"]
    pos = {h: i for i, h in enumerate(ids)}
    X = np.array([r["m6"] for r in rows], float)
    for space, key in (("common", "m6"), ("noncommon", "n6")):
        M = payload_side[space]
        W = np.array(M["load"])
        Xs = np.array([r[key] for r in rows], float)
        mu = Xs.mean(0)
        for i, r in enumerate(rows):
            if r["id"] not in pos:
                continue
            b = B[:, pos[r["id"]], :]
            b = b[np.isfinite(b).all(1)]
            if len(b) < 10:
                continue
            bs = b - (X[i] - Xs[i])                 # 非共通は、同じ系統の平均（全データの値）を引く
            P = (bs - mu) @ W.T
            C = np.cov(P.T)
            ev, evec = np.linalg.eigh(C)
            ev = np.maximum(ev, 0)
            r.setdefault("ell", {})[space] = [round(float(np.sqrt(ev[1])), 3), round(float(np.sqrt(ev[0])), 3),
                                              round(float(np.arctan2(evec[1, 1], evec[0, 1])), 3)]
    for r in rows:
        if r["id"] in pos:
            b = B[:, pos[r["id"]], :]
            r["boot"] = [[round(float(x), 2) for x in v] if np.isfinite(v).all() else None for v in b]   # 欠けた回は null で残す（回をそろえる・critic 3 回目の 3）


def detailed() -> tuple[dict[str, tuple[str, str]], dict]:
    """詳細系統の一番新しい版（v3 → v2）の群の札と、記録の要約。"""
    info: dict = {"versions": []}
    dl: dict[str, tuple[str, str]] = {}
    for v in (1, 2, 3):
        p = ROOT / f"config/aptitude_group/detailed_lines_v{v}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        ks = {g: x["k"] for g, x in d["groups"].items()}
        best = {}
        for g, x in d["groups"].items():
            for k, tr in x.get("tried", {}).items():
                if isinstance(tr, dict) and tr.get("boot_ari") is not None:
                    best[g] = max(best.get(g, -1), tr["boot_ari"])
        info["versions"].append({"v": v, "k": ks, "max_boot_ari": best, "status": d.get("status", "")})
        if any(k > 1 for k in ks.values()):
            dl = {}
            for g, x in d["groups"].items():
                if x["k"] < 2:                       # 群が 1 つの大系統には群の札を付けない
                    continue
                for c in x["clusters"]:
                    for h in c["members"]:
                        dl[h] = (c["cid"], c["name"])
            info["used"] = v
            tl = common.OUT / f"detailed_timeline_v{v}.json"
            if tl.exists():                         # 50〜299 走の種牡馬は、2025-01-01 の切り口の割り当て（群の作りには入らない）
                names = {c["cid"]: c["name"] for x in d["groups"].values() if x["k"] > 1 for c in x["clusters"]}
                for r in json.loads(tl.read_text()):
                    if r["as_of"] == "2025-01-01" and r["label"] in names and r["sire"] not in dl:
                        dl[r["sire"]] = (r["label"], names[r["label"]] + "（割り当て）")
    return dl, info


CSS = """
:root { color-scheme: light; --bg:#f7f8fb; --panel:#ffffff; --panel2:#eef1f7; --border:#d5dbe7; --text:#191919; --muted:#5b6578;
  --accent:#257de0; --s1:#257de0; --s2:#f47230; --s3:#00b894; --s4:#f2a600; --other:#9aa3b2; --pos:#257de0; --neg:#f47230; --grid:#e3e7ef; --hl:#191919; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) { color-scheme: dark; --bg:#0f172a; --panel:#15213c; --panel2:#1b2a4a;
  --border:#2a3a60; --text:#efefef; --muted:#9c9c9c; --accent:#338ced; --s1:#338ced; --s2:#e76624; --s3:#00a888; --s4:#d28e05; --other:#6b7486;
  --pos:#338ced; --neg:#e76624; --grid:#24345a; --hl:#efefef; } }
:root[data-theme="dark"] { color-scheme: dark; --bg:#0f172a; --panel:#15213c; --panel2:#1b2a4a; --border:#2a3a60; --text:#efefef; --muted:#9c9c9c;
  --accent:#338ced; --s1:#338ced; --s2:#e76624; --s3:#00a888; --s4:#d28e05; --other:#6b7486; --pos:#338ced; --neg:#e76624; --grid:#24345a; --hl:#efefef; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.7 "Noto Sans JP",system-ui,sans-serif; }
main { max-width:1100px; margin:0 auto; padding:24px 16px 80px; }
h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:19px; margin:32px 0 8px; padding-top:12px; border-top:1px solid var(--border); }
.sub { color:var(--muted); font-size:13.5px; } .box { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:12px 16px; margin:12px 0; }
.big { background:var(--panel); border:2px solid var(--accent); border-radius:14px; padding:12px 16px; margin:12px 0; }
.row { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin:8px 0; }
select, input, button { font:inherit; padding:4px 8px; border:1px solid var(--border); border-radius:8px; background:var(--panel); color:var(--text); }
button.on { background:var(--accent); color:#fff; border-color:var(--accent); }
.grid2 { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:14px; } @media (max-width:820px) { .grid2 { grid-template-columns:1fr; } }
svg { display:block; width:100%; height:auto; } svg text { fill:var(--text); font-size:12px; } .mut { fill:var(--muted); }
.scroll { overflow-x:auto; border:1px solid var(--border); border-radius:8px; }
table { border-collapse:collapse; width:100%; font-size:13.5px; background:var(--panel); }
th, td { border-bottom:1px solid var(--border); padding:4px 8px; text-align:left; white-space:nowrap; } th { background:var(--panel2); color:var(--muted); font-weight:600; }
td.n { text-align:right; font-variant-numeric:tabular-nums; } tr.click { cursor:pointer; } tr.click:hover { background:var(--panel2); }
.legend { display:flex; gap:14px; flex-wrap:wrap; font-size:13px; color:var(--muted); }
#tip { position:fixed; pointer-events:none; background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:6px 9px; font-size:12.5px; display:none; z-index:9; box-shadow:0 2px 8px rgba(0,0,0,.15); }
"""

JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const LAB = JSON.parse(document.getElementById('lab').textContent);
const M6 = ['dist', 'surf', 'wet', 'pos', 'ts', 'gear'];
let side = 'sire', space = 'common', view = 'pc', colorBy = 'line', line = '', sel = null, cmp = null, ellMode = 'sel';
const $ = s => document.querySelector(s); const tip = $('#tip');
const S = () => D[side]; const rows = () => S().rows;
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const vec = r => space === 'common' ? r.m6 : r.n6;
const dist = (a, b) => Math.sqrt(a.reduce((s, x, i) => s + (x - b[i]) ** 2, 0));
const cosv = (a, b) => { let p = 0, x = 0, y = 0; for (let i = 0; i < a.length; i++) { p += a[i] * b[i]; x += a[i] ** 2; y += b[i] ** 2; } return p / Math.sqrt(x * y || 1); };
const SHAPE = ['●', '■', '▲', '◆'];
function key(r) { return colorBy === 'dl' && r.dl ? r.dl : r.line; }
function keyName(k) { const r = rows().find(x => key(x) === k); return r ? (colorBy === 'dl' && r.dl ? r.dl_name : r.line) : k; }
function topKeys(P) { const m = new Map(); for (const r of P) m.set(key(r), (m.get(key(r)) || 0) + r.runs); return [...m.entries()].sort((a, b) => b[1] - a[1]).slice(0, 4).map(e => e[0]); }
function lines() { const m = new Map(); for (const r of rows()) { const o = m.get(r.line) || {k: r.line, n: 0, runs: 0}; o.n++; o.runs += r.runs; m.set(r.line, o); } return [...m.values()].sort((a, b) => b.runs - a.runs); }
function fillLines() { $('#line').innerHTML = '<option value="">全部（' + rows().length + ' 頭）</option>' + lines().map(o => `<option value="${esc(o.k)}">${esc(o.k)}（${o.n} 頭）</option>`).join(''); $('#line').value = line; }
function mark(slot, x, y, r, fill, on, id) {
  const rr = rows().find(x => x.id === id); const op = rr ? (0.9 - 0.65 * uncRank(rr)).toFixed(2) : 0.8;
  const st = `fill="${fill}" fill-opacity="${op}" stroke="${on ? 'var(--hl)' : 'var(--panel)'}" stroke-width="${on ? 3 : 1.5}" data-id="${esc(id)}" style="cursor:pointer"`;
  if (slot === 1) return `<rect x="${(x - r * 0.9).toFixed(1)}" y="${(y - r * 0.9).toFixed(1)}" width="${(r * 1.8).toFixed(1)}" height="${(r * 1.8).toFixed(1)}" rx="2" ${st}/>`;
  if (slot === 2) return `<polygon points="${x.toFixed(1)},${(y - r * 1.15).toFixed(1)} ${(x + r).toFixed(1)},${(y + r * 0.75).toFixed(1)} ${(x - r).toFixed(1)},${(y + r * 0.75).toFixed(1)}" ${st}/>`;
  if (slot === 3) return `<polygon points="${x.toFixed(1)},${(y - r * 1.2).toFixed(1)} ${(x + r * 1.1).toFixed(1)},${y.toFixed(1)} ${x.toFixed(1)},${(y + r * 1.2).toFixed(1)} ${(x - r * 1.1).toFixed(1)},${y.toFixed(1)}" ${st}/>`;
  return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.toFixed(1)}" ${st}/>`;
}
function bootv(r, b) { const v = r.boot && r.boot[b]; if (!v) return null; return space === 'common' ? v : v.map((x, i) => x - (r.m6[i] - r.n6[i])); }
function distCI(a, b) { const n = Math.min((a.boot || []).length, (b.boot || []).length); const ds = [];
  for (let k = 0; k < n; k++) { const x = bootv(a, k), y = bootv(b, k); if (x && y) ds.push(dist(x, y)); } if (ds.length < 10) return null; ds.sort((p, q) => p - q);
  return [ds[Math.floor(0.025 * (ds.length - 1))], ds[Math.ceil(0.975 * (ds.length - 1))]]; }
const fmtCI = c => c ? `${c[0].toFixed(2)}〜${c[1].toFixed(2)}` : '—';
let uncSorted = null, uncSide = null;
function uncRank(r) { if (uncSide !== side) { uncSorted = rows().map(x => x.unc).sort((a, b) => a - b); uncSide = side; }   // 不確かさの順位（0 = 確か・1 = 不確か）
  let lo = 0, hi = uncSorted.length; while (lo < hi) { const m = (lo + hi) >> 1; if (uncSorted[m] < r.unc) lo = m + 1; else hi = m; } return lo / Math.max(uncSorted.length - 1, 1); }
function near(r, pool, n = 10) { const a = vec(r); return pool.filter(x => x.id !== r.id).map(x => [x, dist(a, vec(x)), cosv(a, vec(x))]).sort((p, q) => p[1] - q[1]).slice(0, n); }
function drawMap() {
  const M = S()[space], C = view === 'pc' ? M.pc : M.ts, R = rows();
  const show = R.map(r => !line || r.line === line);
  const keys = topKeys(R.filter((r, i) => show[i])); const col = k => { const i = keys.indexOf(k); return i < 0 ? 'var(--other)' : `var(--s${i + 1})`; };
  const xs = C.map(c => c[0]), ys = C.map(c => c[1]); const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const W = 720, H = 520, pad = 34; const sx = x => pad + (x - x0) / ((x1 - x0) || 1) * (W - 2 * pad), sy = y => H - pad - (y - y0) / ((y1 - y0) || 1) * (H - 2 * pad);
  const maxr = Math.max(...R.map(r => r.runs));
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="種牡馬の 2 次元の地図"><defs><marker id="ar" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="var(--muted)"/></marker></defs><rect width="${W}" height="${H}" fill="var(--panel)" rx="8"/>`;
  if (view === 'pc') {                     // バイプロット: 6 本の向き
    const cx = sx(0), cy = sy(0), mxl = Math.max(...M.load[0].map((a, i) => Math.hypot(a, M.load[1][i]))), L = 0.32 * Math.min(W, H) / (mxl || 1);
    M6.forEach((d, i) => { const ex = cx + M.load[0][i] * L, ey = cy - M.load[1][i] * L;
      s += `<line x1="${cx.toFixed(1)}" y1="${cy.toFixed(1)}" x2="${ex.toFixed(1)}" y2="${ey.toFixed(1)}" stroke="var(--muted)" stroke-width="1.5" marker-end="url(#ar)"/><text x="${(ex + (ex >= cx ? 4 : -4)).toFixed(1)}" y="${(ey - 4).toFixed(1)}" text-anchor="${ex >= cx ? 'start' : 'end'}" class="mut">${esc(LAB[d])}</text>`; });
  }
  if (view === 'pc' && ellMode !== 'none') {         // 誤差の楕円（ブートストラップ・68% と 95%）
    for (let i = 0; i < R.length; i++) { const r = R[i]; const e = r.ell && r.ell[space]; if (!e || !show[i]) continue;
      if (ellMode === 'sel' && !(sel && sel.id === r.id) && !(cmp && cmp.id === r.id) && !(line && r.line === line)) continue;
      for (const [q, op] of [[1.52, 0.55], [2.45, 0.3]]) { let d = '';
        for (let k = 0; k <= 36; k++) { const t = 2 * Math.PI * k / 36, ux = e[0] * q * Math.cos(t), uy = e[1] * q * Math.sin(t);
          const px = C[i][0] + ux * Math.cos(e[2]) - uy * Math.sin(e[2]), py = C[i][1] + ux * Math.sin(e[2]) + uy * Math.cos(e[2]);
          d += (k ? 'L' : 'M') + sx(px).toFixed(1) + ',' + sy(py).toFixed(1); }
        s += `<path d="${d}Z" fill="none" stroke="${col(key(r))}" stroke-opacity="${op}" stroke-width="1.3" ${q > 2 ? 'stroke-dasharray="4 3"' : ''}/>`; } } }
  if (sel) { const i = R.indexOf(sel); for (const [x] of near(sel, R)) { const j = R.indexOf(x); s += `<line x1="${sx(C[i][0]).toFixed(1)}" y1="${sy(C[i][1]).toFixed(1)}" x2="${sx(C[j][0]).toFixed(1)}" y2="${sy(C[j][1]).toFixed(1)}" stroke="var(--hl)" stroke-opacity="0.45" stroke-width="1.2"/>`; } }
  const order = R.map((r, i) => i).sort((a, b) => R[b].runs - R[a].runs);
  for (const i of order) { const r = R[i]; const rad = 3.5 + 9 * Math.sqrt(r.runs / maxr);
    if (!show[i]) { s += `<circle cx="${sx(C[i][0]).toFixed(1)}" cy="${sy(C[i][1]).toFixed(1)}" r="${rad.toFixed(1)}" fill="var(--other)" fill-opacity="0.18" data-id="${esc(r.id)}"/>`; continue; }
    s += mark(keys.indexOf(key(r)), sx(C[i][0]), sy(C[i][1]), rad, col(key(r)), (sel && sel.id === r.id) || (cmp && cmp.id === r.id), r.id); }
  const labs = new Set(order.filter(i => show[i]).slice(0, 8)); if (sel) labs.add(R.indexOf(sel)); if (cmp) labs.add(R.indexOf(cmp));
  for (const i of labs) s += `<text x="${(sx(C[i][0]) + 8).toFixed(1)}" y="${(sy(C[i][1]) - 6).toFixed(1)}">${esc(R[i].name)}</text>`;
  $('#map').innerHTML = s + '</svg>';
  $('#legend').innerHTML = keys.map((k, i) => `<span><span style="color:var(--s${i + 1})">${SHAPE[i]}</span> ${esc(keyName(k))}</span>`).join('') + '<span><span style="color:var(--other)">●</span> その他</span>';
  $('#distort').textContent = `この地図のゆがみ: 6 本の元の距離と、地図の上の距離の順位相関 ${(view === 'pc' ? M.rho_pc : M.rho_ts).toFixed(2)}（1 に近いほど、地図の遠近が元の近さに忠実）。` + (view === 'pc' ? `2 本の軸で 6 本の分散の ${((M.var[0] + M.var[1]) * 100).toFixed(0)}% を表す。` : 't-SNE は近い同士の並びを保ち、遠い同士の距離は当てにならない。');
  $('#map').querySelectorAll('[data-id]').forEach(c => {
    c.addEventListener('mousemove', e => { const r = R.find(x => x.id === c.dataset.id); tip.style.display = 'block'; tip.style.left = (e.clientX + 12) + 'px'; tip.style.top = (e.clientY + 12) + 'px';
      tip.innerHTML = `<b>${esc(r.name)}</b>（${esc(r.line)}${r.dl_name ? '・' + esc(r.dl_name) : ''}）<br>走数 ${r.runs.toLocaleString()}${r.core ? '' : '（300 未満・値はぶれやすい）'}・不確かさ ${r.unc.toFixed(2)}・得意な距離帯 ${esc(r.best)}<br>` + M6.map((d, k) => `${LAB[d]} ${vec(r)[k].toFixed(2)}`).join('・'); });
    c.addEventListener('mouseleave', () => tip.style.display = 'none');
    c.addEventListener('click', e => { if (e.shiftKey && sel) { cmp = R.find(x => x.id === c.dataset.id); compare(); drawMap(); } else pick(c.dataset.id); });
  });
}
function radar(r) {
  const n = 6, W = 340, H = 300, cx = 170, cy = 150, R = 105, lim = 2.5;
  const mates = rows().filter(x => r.line === '札なし' ? true : x.line === r.line);   // 札なしは全体の平均（critic 4 回目） const wt = x => x.core ? x.runs : 0; const mean = M6.map((_, k) => mates.reduce((a, x) => a + x.m6[k] * wt(x), 0) / (mates.reduce((a, x) => a + wt(x), 0) || 1));
  const pt = (v, k) => { const t = -Math.PI / 2 + 2 * Math.PI * k / n, rr = R * (Math.max(-lim, Math.min(lim, v)) + lim) / (2 * lim); return [cx + rr * Math.cos(t), cy + rr * Math.sin(t)]; };
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="レーダー図">`;
  for (const g of [-2, -1, 0, 1, 2]) s += `<polygon points="${M6.map((_, k) => pt(g, k).join(',')).join(' ')}" fill="none" stroke="var(--grid)" stroke-width="${g === 0 ? 1.5 : 1}"/>`;
  M6.forEach((a, k) => { const [x, y] = pt(lim + 0.95, k); s += `<text x="${x.toFixed(0)}" y="${(y + 4).toFixed(0)}" text-anchor="middle" class="mut">${esc(LAB[a])}</text>`; });
  s += `<polygon points="${mean.map((v, k) => pt(v, k).join(',')).join(' ')}" fill="var(--s2)" fill-opacity="0.1" stroke="var(--s2)" stroke-width="2" stroke-dasharray="5 4"/>`;
  s += `<polygon points="${r.m6.map((v, k) => pt(v, k).join(',')).join(' ')}" fill="var(--s1)" fill-opacity="0.18" stroke="var(--s1)" stroke-width="2"/>`;
  r.m6.forEach((v, k) => { const [x, y] = pt(v, k); s += `<circle cx="${x}" cy="${y}" r="4" fill="var(--s1)" stroke="var(--panel)" stroke-width="2"><title>${esc(LAB[M6[k]])} ${v.toFixed(2)}</title></circle>`; });
  return s + '</svg>';
}
function hbars(names, vals, W = 340) {
  const H = 16 + names.length * 30, x0 = 150, sc = 50; let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="棒の図"><line x1="${x0}" y1="6" x2="${x0}" y2="${H - 6}" stroke="var(--grid)"/>`;
  names.forEach((nm, k) => { const raw = vals[k], v = Math.max(-2.8, Math.min(2.8, raw)), y = 20 + k * 30, w = Math.abs(v) * sc;
    s += `<text x="6" y="${y + 5}" class="mut">${esc(nm)}</text><rect x="${v >= 0 ? x0 : x0 - w}" y="${y - 9}" width="${w.toFixed(1)}" height="18" rx="4" fill="${v >= 0 ? 'var(--pos)' : 'var(--neg)'}"><title>${raw.toFixed(2)}</title></rect>`;
    s += `<text x="${v >= 0 ? x0 + w + 4 : x0 - w - 4}" y="${y + 5}" text-anchor="${v >= 0 ? 'start' : 'end'}">${raw.toFixed(2)}</text>`; });
  return s + '</svg>';
}
function pick(id) {
  sel = rows().find(x => x.id === id); if (!sel) return; const r = sel;
  if (cmp && cmp.id === r.id) cmp = null;
  $('#selname').textContent = `${r.name}（大系統 ${r.line}・父の父 ${r.mid_name}${r.dl_name ? '・群 ' + r.dl_name : ''}・走数 ${r.runs.toLocaleString()}・得意な距離帯 ${r.best}）`;
  $('#radar').innerHTML = radar(r);
  const ci = ['steep', 'small', 'straight', 'left', 'yoshiba', 'kyoto'].map(k => D.dims.indexOf(k));
  $('#bars').innerHTML = hbars(ci.map(i => LAB[D.dims[i]]), ci.map(i => r.v[i]));
  const up = side === 'sire' ? ['父の父の系統との差', '4 代上の系統との差'] : ['母父の父の系統との差', '母父の 3 代上の系統との差'];   // critic 3 回目の 7
  $('#dimtab').innerHTML = `<tr><th>次元</th><th>値（z）</th><th>${up[0]}</th><th>${up[1]}</th></tr>` + D.dims.map((d, i) => `<tr><td>${esc(LAB[d])}</td><td class="n">${r.v[i].toFixed(2)}</td><td class="n">${r.nm[i].toFixed(2)}</td><td class="n">${r.nt[i].toFixed(2)}</td></tr>`).join('');
  const tb = L => '<tr><th>順</th><th>種牡馬</th><th>大系統</th><th>距離</th><th>引き直しの幅（95%）</th><th>コサイン</th></tr>' + L.map(([x, d, c], k) => `<tr class="click" data-id="${esc(x.id)}"><td class="n">${k + 1}</td><td>${esc(x.name)}</td><td>${esc(x.line)}</td><td class="n">${d.toFixed(2)}</td><td class="n">${fmtCI(distCI(r, x))}</td><td class="n">${c.toFixed(2)}</td></tr>`).join('');
  $('#nearall').innerHTML = tb(near(r, rows())); $('#nearline').innerHTML = r.line === '札なし' ? '<tr><td>大系統の札が無い種牡馬（全体の表を見る）</td></tr>' : tb(near(r, rows().filter(x => x.line === r.line)));
  document.querySelectorAll('#nearall tr.click, #nearline tr.click').forEach(t => t.addEventListener('click', e => { if (e.shiftKey) { cmp = rows().find(x => x.id === t.dataset.id); compare(); drawMap(); } else pick(t.dataset.id); }));
  compare(); heat(); drawMap();
}
function compare() {
  if (!sel || !cmp) { $('#cmp').innerHTML = '<p class="sub">比べる種牡馬を選ぶと（上の欄に名前を入れるか、地図の点を Shift を押しながら押す）、2 頭の近さを数で出す。</p>'; return; }
  const a = vec(sel), b = vec(cmp); const d = dist(a, b), c = cosv(a, b);
  const all = rows(); let lo = 0, n = 0; for (let i = 0; i < all.length; i++) for (let j = i + 1; j < all.length; j++) { n++; if (dist(vec(all[i]), vec(all[j])) < d) lo++; }
  $('#cmp').innerHTML = `<p><b>${esc(sel.name)}</b> と <b>${esc(cmp.name)}</b>: 標準化したユークリッド距離 <b>${d.toFixed(2)}</b>（引き直しの幅 ${fmtCI(distCI(sel, cmp))}・全部の組の近い方から ${(100 * lo / n).toFixed(0)}%）・コサインの類似度 <b>${c.toFixed(2)}</b></p><p class="sub">棒 = ${esc(sel.name)} − ${esc(cmp.name)}（右 = ${esc(sel.name)} の方が強い）</p>` + hbars(M6.map(x => LAB[x]), a.map((x, i) => x - b[i]), 420);
}
function order(P) {                          // 平均連結の階層クラスタの葉の並び（近い順の並べ替え）
  let cl = P.map((_, i) => [i]); const d = (A, B) => { let s = 0; for (const i of A) for (const j of B) s += dist(vec(P[i]), vec(P[j])); return s / (A.length * B.length); };
  while (cl.length > 1) { let bi = 0, bj = 1, bd = Infinity; for (let i = 0; i < cl.length; i++) for (let j = i + 1; j < cl.length; j++) { const x = d(cl[i], cl[j]); if (x < bd) { bd = x; bi = i; bj = j; } }
    cl[bi] = cl[bi].concat(cl[bj]); cl.splice(bj, 1); }
  return cl[0].map(i => P[i]);
}
function heat() {
  let P = (line ? rows().filter(r => r.line === line).sort((a, b) => b.runs - a.runs) : (sel ? [sel, ...near(sel, rows(), 24).map(x => x[0])] : rows().slice().sort((a, b) => b.runs - a.runs).slice(0, 25))).slice(0, 30);
  if (P.length < 2) { $('#heat').innerHTML = ''; return; } P = order(P);
  const n = P.length, c = 18, L = 130, W = L + n * c + 10, H = L + n * c + 10; const M = P.map(a => P.map(b => dist(vec(a), vec(b)))); const mx = Math.max(...M.flat());
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="距離の行列">`;
  P.forEach((a, i) => { s += `<text x="${L - 4}" y="${L + i * c + 13}" text-anchor="end" class="mut">${esc(a.name.slice(0, 10))}</text><text transform="translate(${L + i * c + 13},${L - 4}) rotate(-60)" class="mut">${esc(a.name.slice(0, 10))}</text>`;
    P.forEach((b, j) => { s += `<rect x="${L + j * c}" y="${L + i * c}" width="${c - 2}" height="${c - 2}" rx="2" fill="var(--s1)" fill-opacity="${(1 - M[i][j] / (mx || 1)).toFixed(2)}"><title>${esc(a.name)} × ${esc(b.name)}: 距離 ${M[i][j].toFixed(2)}</title></rect>`; }); });
  $('#heat').innerHTML = s + '</svg>';
  $('#heatnote').textContent = line ? `大系統 ${line} の種牡馬（走数の多い 30 頭まで）。` : (sel ? `${sel.name} と、近い 24 頭。` : '走数の多い 25 頭。');
}
function setBtn(group, v) { document.querySelectorAll(group + ' button').forEach(b => b.classList.toggle('on', b.dataset.v === v)); }
document.querySelectorAll('#views button').forEach(b => b.addEventListener('click', () => { view = b.dataset.v; setBtn('#views', view); drawMap(); }));
document.querySelectorAll('#spaces button').forEach(b => b.addEventListener('click', () => { space = b.dataset.v; setBtn('#spaces', space); if (sel) pick(sel.id); else drawMap(); }));
document.querySelectorAll('#ells button').forEach(b => b.addEventListener('click', () => { ellMode = b.dataset.v; setBtn('#ells', ellMode); drawMap(); }));
document.querySelectorAll('#colors button').forEach(b => b.addEventListener('click', () => { colorBy = b.dataset.v; setBtn('#colors', colorBy); drawMap(); }));
$('#side').addEventListener('change', e => { side = e.target.value; line = ''; sel = null; cmp = null; fillLines(); start(); });
$('#line').addEventListener('change', e => { line = e.target.value; heat(); drawMap(); });
function find(q) { q = q.trim(); return q ? rows().find(x => x.name.includes(q)) : null; }
$('#q').addEventListener('change', e => { const r = find(e.target.value); if (r) { line = ''; fillLines(); pick(r.id); } });
$('#q2').addEventListener('change', e => { const r = find(e.target.value); if (r && sel) { cmp = r; compare(); drawMap(); } });
function start() { const f = rows().slice().sort((a, b) => b.runs - a.runs)[0]; if (f) pick(f.id); }
if (!D.has_dl) document.querySelectorAll('#colors button[data-v="dl"]').forEach(b => b.style.display = 'none');
if (!D.n_boot) document.querySelectorAll('#ells').forEach(b => b.style.display = 'none');
fillLines(); setBtn('#ells', ellMode); setBtn('#views', view); setBtn('#spaces', space); setBtn('#colors', colorBy); start();
"""


def summary_html(info: dict) -> str:
    vs = {v["v"]: v for v in info["versions"]}

    def rng(b):
        vals = [x for x in b.values() if x is not None]
        return f"{min(vals):.2f}〜{max(vals):.2f}" if vals else "—"
    li = []
    if 1 in vs:
        li.append(f"<li>v1（13 本・大系統の中・k = 2〜4）: 引き直すと群の境が大きく動き、安定の線（ARI 0.6）に届かなかった（大系統ごとの最良の ARI {rng(vs[1]['max_boot_ari'])}）。</li>")
    if 2 in vs:
        ok = [g for g, k in vs[2]["k"].items() if k > 1]
        li.append("<li>v2（主の 6 本・大系統の中）: " + (f"{len(ok)} の大系統で群ができた。" if ok else f"どの大系統も届かなかった（最良の ARI {rng(vs[2]['max_boot_ari'])}）。") + "</li>")
    if 3 in vs:
        k = vs[3]["k"].get("全体", 1)
        li.append("<li>v3（主の 6 本・大系統の枠を外して全種牡馬・k = 3〜12）: " + (f"k = {k} の群ができた（色の「群」で見られる）。" if k > 1 else f"届かなかった（最良の ARI {rng(vs[3]['max_boot_ari'])}）。") + "</li>")
    used = info.get("used")
    if not used:
        head = ("<p><b>種牡馬のベクトルは、地図の上で連続に散らばり、はっきり離れた群（詳細系統）には分かれなかった。</b>"
                "同じ系統の中でも近い・遠いはあるが、境目を引くと、データを少し入れ替えただけで境が動く。</p>")
        tail = ("<p>そのため、群に割り当てる代わりに「近い種牡馬の上位 10 頭」で表す形にした。新しい種牡馬も、その時点までの産駒の走りで同じように近い種牡馬が付く"
                "（産駒が少ないうちは、父の父の系統の値に寄る）。出走馬どうしの近さは、この近傍から作れる。</p>")
    else:
        n_ok = sum(1 for k in vs[used]["k"].values() if k > 1)
        head = (f"<p><b>種牡馬のベクトルは、ほとんどの系統で連続に散らばり、はっきり離れた群（詳細系統）には分かれなかった。</b>"
                f"v{used} の規則で安定した群ができたのは、{len(vs[used]['k'])} の大系統のうち {n_ok} つだけ。</p>")
        tail = ("<p>群に分かれない系統は、群の代わりに「近い種牡馬の上位 10 頭」で表す（下の地図と表）。新しい種牡馬も、その時点までの産駒の走りで同じように近い種牡馬が付く"
                "（産駒が少ないうちは、父の父の系統の値に寄る）。出走馬どうしの近さは、この近傍から作れる。</p>")
    return f'<div class="big">{head}<ul>{"".join(li)}</ul>{tail}</div>'


def build() -> None:
    d = json.loads((common.OUT / "sire_vectors.json").read_text())
    dl, info = detailed()
    qualified = {r["id"] for r in d["sire"]["rows"] if r.get("core", True)}   # 札は中核（300 走以上）で数える（lines・detailed と同じ・critic 3 回目の 5）
    lab = line_labels(qualified)
    payload = {"dims": d["dims"], "sire": side_payload(d["sire"]["rows"], d["dims"], lab, dl),
               "bms": side_payload(d["bms"]["rows"], d["dims"], lab, {}), "has_dl": bool(dl)}
    bp = common.OUT / "sire_vectors_boot.npz"
    if bp.exists():
        z = np.load(bp, allow_pickle=False)
        add_boot(payload["sire"], z["sire_ids"].astype(str), z["sire_B"])
        add_boot(payload["bms"], z["bms_ids"].astype(str), z["bms_B"])
        payload["n_boot"] = int(z["sire_B"].shape[0])
    js_data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")      # < をすべて逃がす（critic 2 回目の 16）
    lab_js = json.dumps(LABEL, ensure_ascii=False).replace("<", "\\u003c")
    ps, pb = payload["sire"], payload["bms"]
    body = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>種牡馬の血統ベクトル</title><style>{CSS}</style></head><body><main>
<h1>種牡馬の血統ベクトル</h1>
<p class="sub">作成 {d['made']}（JST）。担当 aptitude-group。データは {d['period']} 年の平地の走り（手元の表だけ）。記述と可視化で、馬券の検定ではない。記録は <code>docs/operations/aptitude_group/log.md</code>。</p>
{summary_html(info)}
<div class="box">
<p><b>何を見る図か</b>: 種牡馬ごとに、産駒（または母父としての孫）の走りから適性を作り、ベクトルとして並べた。地図は主の 6 本（距離・ダート寄り・道悪・前の位置・トップスピード・ギアチェンジ）で作る。</p>
<ul>
<li>対象は産駒の走数 {d.get('show_runs', 50)} 以上の種牡馬。尺度（0 = 平均・±1 = 1 標準偏差）と系統の平均は、走数 {d['min_runs']} 以上の種牡馬（中核）で作った。走数が少ないほど、4 代上 → 父の父 → 種牡馬の順に上の段へ寄せた（擬似件数 {d['k']:.0f}）。</li>
<li>非共通の地図 = 6 本 − 同じ大系統（netkeiba の「○○系」を父から上へたどった札）の平均。系統の中での個性。</li>
<li>ギアチェンジは、終いのラップが速いレースで、4 角の位置を除いて入線までに上げた度合い（レースのラップで近似）。</li>
<li>限界: 産駒の走りには、母の質・厩舎・使われ方が混ざる。値は「産駒がその条件でふだんよりどれだけ走ったか」で、遺伝だけの値ではない。</li>
</ul></div>
<h2>全種牡馬の 2 次元の地図</h2>
<div class="row"><label>見る側 <select id="side"><option value="sire">父として（{len(ps['rows'])} 頭）</option><option value="bms">母父として（{len(pb['rows'])} 頭）</option></select></label>
<label>大系統で絞る <select id="line"></select></label><label>名前で探す <input id="q" placeholder="例: ディープ"></label></div>
<div class="row"><span class="sub">地図</span><span id="views"><button data-v="pc">PCA（軸の意味の矢印つき）</button> <button data-v="ts">t-SNE</button></span>
<span class="sub">ベクトル</span><span id="spaces"><button data-v="common">そのまま（共通の部分）</button> <button data-v="noncommon">系統の平均を引いた個性</button></span>
<span id="ells"><span class="sub">誤差の楕円</span> <button data-v="sel">選んだ馬と系統</button> <button data-v="all">全頭</button> <button data-v="none">なし</button></span>
<span class="sub">色</span><span id="colors"><button data-v="line">大系統</button> <button data-v="dl">群</button></span></div>
<p class="sub" id="distort"></p>
<div class="legend" id="legend"></div>
<div id="map"></div>
<p class="sub">誤差の楕円（PCA の地図だけ）: 産駒の走りを開催の単位で引き直したブートストラップの各回のベクトルを、同じ軸に写した広がり。実線 = 68%、破線 = 95%。走数の少ない種牡馬ほど楕円が大きい。t-SNE は写しが回ごとに変わるので、楕円は出さず、点の濃さで見せる。</p>
<p class="sub">距離の「引き直しの幅」: ブートストラップの各回の 2 頭の距離の 2.5〜97.5%。測りの雑音で距離は上ぶれするので、幅は本当の距離より上に寄る（近い組ほど強い）。</p>
<p class="sub">点の大きさ = 走数。点の濃さ = 値の確かさの順位（同じ側の中で、走数が少なく上の段の値に寄った種牡馬ほど薄い）。色と形 = 走数の多い上位 4 つ（ほかは「その他」）。点を押すと下に詳しく出て、近い 10 頭と線で結ぶ。Shift を押しながら押すと、2 頭の比べ。</p>
<h2 id="selname"></h2>
<div class="grid2"><div class="box"><b>主の 6 本</b>（実線 = この種牡馬・破線 = 同じ大系統〔札が無ければ全体〕の中核の走数の重み付き平均）<div id="radar"></div></div>
<div class="box"><b>コースの 6 本</b>（右 = 得意・左 = 苦手）<div id="bars"></div></div></div>
<h2>2 頭の近さ</h2><div class="row"><label>比べる種牡馬 <input id="q2" placeholder="名前"></label></div><div class="box" id="cmp"></div>
<h2>近い種牡馬（主の 6 本の距離）</h2>
<div class="grid2"><div><b>全体</b><div class="scroll"><table id="nearall"></table></div></div><div><b>同じ大系統</b><div class="scroll"><table id="nearline"></table></div></div></div>
<h2>距離の行列</h2><p class="sub" id="heatnote"></p><p class="sub">近い順に並べ替えた（平均連結の階層クラスタの並び）。色が濃いほど近い。</p><div class="scroll" id="heat"></div>
<h2>13 本の値</h2><div class="scroll"><table id="dimtab"></table></div>
<div id="tip"></div>
<script type="application/json" id="data">{js_data}</script>
<script type="application/json" id="lab">{lab_js}</script>
<script>{JS}</script></main></body></html>"""
    OUT.write_text(body, encoding="utf-8")
    print(OUT, OUT.stat().st_size, "rho", ps["common"]["rho_pc"], ps["common"]["rho_ts"])


if __name__ == "__main__":
    build()
