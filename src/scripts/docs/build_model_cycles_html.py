"""親モデルの改善サイクルのログ（各サイクルの仮説・変更・結果・結論）を 1 枚の HTML にする。
外部の JS・CSS なし、ライト・ダーク両対応、スマホ幅。推移のグラフは素の SVG。

使い方: python -m src.scripts.docs.build_model_cycles_html [入力 JSON] [出力 HTML]
入力: docs/operations/retrain_trials/model_cycles_log.json（python -m src.scripts.data.model_cycles record で 1 サイクルずつ足す）
出力: docs/html/operations/model_cycles.html（作り直したら python -m src.scripts.docs.build_docs_book で本も作り直す）
"""

from __future__ import annotations

import json
import math
import sys
import time
from html import escape
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "docs/operations/retrain_trials/model_cycles_log.json"
OUT = REPO / "docs/html/operations/model_cycles.html"

PHASES = {"before_fix": "データを直す前・参考（中断前の篩い。学習の表と fold を直す前の値）",
          "fix1": "直した表・参考（追い切りの生き残りの漏れを除く前。2016〜2019 年の wc_*・tl_* に漏れが残っていた）",
          "fix2": "直した表・漏れの除去あり・前段 S3 は旧（参考。前段を作り直す前の値）",
          "main": "本番のサイクル（直した表・漏れの除去・前段 S3 も作り直した後）"}
VERDICT_CLASS = {"採用": "good", "不採用": "bad", "保留": "hold", "基準": "base"}
METRICS = [("ll", "勝ち馬の対数損失", 5, False), ("top1_win", "◎ の勝率", 1, True), ("top1_place", "◎ の複勝率", 1, True),
           ("roi_tansho", "単勝の回収率", 1, True), ("roi_fukusho", "複勝の回収率", 1, True),
           ("roi_umaren", "馬連の流しの回収率", 1, True), ("roi_wide", "ワイドの流しの回収率", 1, True)]
BY = [("ll_2022", "2022 年"), ("ll_2023", "2023 年"), ("ll_2024", "2024 年"), ("ll_newcomer", "新馬戦"), ("ll_2yo", "2 歳戦")]

CSS = """
:root{--bg:#f7f8fa;--panel:#fff;--text:#1d2330;--muted:#5b6578;--border:#d8dde6;--accent:#2f6fde;--good:#1d8a4e;--bad:#b3261e;
--hold:#a86400;--base:#5b6578;--head:#eef1f6;--grid:#e3e7ee}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f1420;--panel:#161d2b;--text:#e4e9f2;--muted:#9aa6ba;
--border:#2a3446;--accent:#7aa7ff;--good:#5fd38d;--bad:#ff8a80;--hold:#f0b65a;--base:#9aa6ba;--head:#1d2638;--grid:#263043}}
:root[data-theme="dark"]{--bg:#0f1420;--panel:#161d2b;--text:#e4e9f2;--muted:#9aa6ba;--border:#2a3446;--accent:#7aa7ff;--good:#5fd38d;
--bad:#ff8a80;--hold:#f0b65a;--base:#9aa6ba;--head:#1d2638;--grid:#263043}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.65 system-ui,-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif}
main{max-width:980px;margin:0 auto;padding:16px}
h1{font-size:1.4rem;margin:.2em 0}h2{font-size:1.15rem;margin:1.6em 0 .5em;border-bottom:1px solid var(--border);padding-bottom:.2em}
h3{font-size:1.02rem;margin:0 0 .4em}
.muted{color:var(--muted);font-size:.88rem}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px;margin:12px 0}
.card{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:10px 12px}
.card .k{color:var(--muted);font-size:.82rem}.card .v{font-size:1rem;font-weight:600}
.cycle{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px;margin:12px 0}
.badge{display:inline-block;border-radius:4px;padding:0 .5em;font-size:.82rem;font-weight:600;border:1px solid currentColor}
.good{color:var(--good)}.bad{color:var(--bad)}.hold{color:var(--hold)}.base{color:var(--base)}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:.88rem;margin:.4em 0}
th,td{border:1px solid var(--border);padding:4px 6px;text-align:right;white-space:nowrap}
th{background:var(--head);font-weight:600}td.l,th.l{text-align:left;white-space:normal}
code{font-size:.82rem;background:var(--head);padding:0 .3em;border-radius:3px;word-break:break-all}
dl{margin:.3em 0}dt{color:var(--muted);font-size:.84rem;margin-top:.4em}dd{margin:0 0 0 .2em}
svg{display:block;width:100%;height:auto}
svg text{fill:var(--muted);font-size:11px}
ul{padding-left:1.2em}
"""


def _ok(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def _pct(v: float | None, d: int = 1) -> str:
    return f"{v * 100:.{d}f}%" if _ok(v) else "－"


def _num(v: float | None, d: int = 5) -> str:
    return f"{v:.{d}f}" if _ok(v) else "－"


def _sign(s: str) -> str:
    return s.replace("-", "−")


def _ci(x: list | None, pct: bool = False, d: int = 4) -> str:
    """[差, 下限, 上限] を「+0.0012（−0.0020〜+0.0043）」に。率は %pt。"""
    if not x or not all(_ok(v) for v in x):
        return "－"
    if pct:
        m, lo, hi = (v * 100 for v in x)
        f = lambda v: f"{v:+.1f}"  # noqa: E731
        return _sign(f"{f(m)}pt（{f(lo)}〜{f(hi)}）")
    f = lambda v: f"{v:+.{d}f}"  # noqa: E731
    return _sign(f"{f(x[0])}（{f(x[1])}〜{f(x[2])}）")


def _metric_val(v: float | None, is_rate: bool) -> str:
    return _pct(v) if is_rate else _num(v)


def _diff_class(x: list | None, higher_is_better: bool) -> str:
    """区間が 0 をまたがない差に色を付ける（良い向きなら good、悪い向きなら bad）。"""
    if not x or not all(_ok(v) for v in x) or x[1] <= 0 <= x[2]:
        return ""
    return "good" if (x[0] > 0) == higher_is_better else "bad"


def chart(cycles: list[dict]) -> str:
    """各サイクルの「変更 − 相手」の勝ち馬の対数損失（点）と 95% 区間（縦線）。0 より下が良い。"""
    pts = [(c["id"], c["result"]["diff"].get("ll"), c["verdict"], c.get("phase", "main")) for c in cycles if c.get("result")]
    pts = [p for p in pts if p[1] and all(_ok(v) for v in p[1])]
    if not pts:
        return '<p class="muted">差の付いたサイクルはまだ無い。</p>'
    W, H, L, R, T, B = 640, 240, 56, 12, 14, 34
    lo = min(min(p[1][1] for p in pts), 0.0)
    hi = max(max(p[1][2] for p in pts), 0.0)
    pad = (hi - lo) * 0.1 or 0.001
    lo, hi = lo - pad, hi + pad
    n = len(pts)
    xs = [L + (W - L - R) * (i + 0.5) / n for i in range(n)]
    y = lambda v: T + (H - T - B) * (hi - v) / (hi - lo)  # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="サイクルごとの勝ち馬の対数損失の差と 95% 区間">']
    raw = (hi - lo) / 4
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    dec = max(0, -math.floor(math.log10(step)) + (1 if round(step / 10 ** math.floor(math.log10(step)), 6) == 2.5 else 0))
    for k in range(math.ceil(lo / step), math.floor(hi / step) + 1):     # 0 を通るきりの良い刻み
        v = k * step
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="var(--grid)"/>'
                   f'<text x="{L - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{_sign(f"{v:+.{dec}f}")}</text>')
    phases = [p[3] for p in pts]
    if "main" in phases and phases.index("main") > 0:     # 直す前（参考）と直した後の境目
        xb = (xs[phases.index("main") - 1] + xs[phases.index("main")]) / 2
        out.append(f'<line x1="{xb:.1f}" x2="{xb:.1f}" y1="{T}" y2="{H - B}" stroke="var(--muted)"/>'
                   f'<text x="{xb + 4:.1f}" y="{T + 10}">直した表 →</text>')
    out.append(f'<line x1="{L}" x2="{W - R}" y1="{y(0):.1f}" y2="{y(0):.1f}" stroke="var(--muted)" stroke-dasharray="4 3"/>')
    for x, (cid, d, verdict, ph) in zip(xs, pts):
        col = f"var(--{VERDICT_CLASS.get(verdict, 'base')})"
        fill = "var(--panel)" if ph != "main" else col                     # 参考の段（直す前・漏れを除く前）は白抜き
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y(d[1]):.1f}" y2="{y(d[2]):.1f}" stroke="{col}" stroke-width="2"/>'
                   f'<circle cx="{x:.1f}" cy="{y(d[0]):.1f}" r="4.5" fill="{fill}" stroke="{col}" stroke-width="2"><title>{escape(cid)}: {_ci(d)}（{escape(verdict)}）</title></circle>'
                   f'<text x="{x:.1f}" y="{H - B + 16}" text-anchor="middle">{escape(cid)}</text>')
    out.append(f'<text x="{L}" y="{H - 4}">差 = 変更 − 相手（負なら変更が良い）。縦線は 95% 区間。白抜きは参考の段（直す前・漏れを除く前）</text></svg>')
    return "".join(out)


def trend_table(cycles: list[dict]) -> str:
    rows = []
    for c in cycles:
        r = c.get("result") or {}
        m = r.get("metrics") or {}
        d = (r.get("diff") or {}).get("ll")
        cls = VERDICT_CLASS.get(c["verdict"], "base")
        rows.append(f'<tr><td class="l"><a href="#{escape(c["id"])}">{escape(c["id"])}</a></td><td class="l">{escape(c["change"])}</td>'
                    f'<td>{_num(m.get("ll"))}</td><td>{_ci(d)}<br><span class="muted">相手: {escape(r.get("ref") or "－")}</span></td><td>{_pct(m.get("top1_win"))}</td><td>{_pct(m.get("top1_place"))}</td>'
                    f'<td>{_pct(m.get("roi_tansho"))}</td><td>{_pct(m.get("roi_fukusho"))}</td><td>{_pct(m.get("roi_umaren"))}</td><td>{_pct(m.get("roi_wide"))}</td>'
                    f'<td><span class="badge {cls}">{escape(c["verdict"])}</span></td></tr>')
    return ('<div class="tw"><table><thead><tr><th class="l">回</th><th class="l">変更</th><th>対数損失</th><th>相手との差（区間。正なら悪い）</th>'
            '<th>◎勝率</th><th>◎複勝率</th><th>単勝</th><th>複勝</th><th>馬連流し</th><th>ワイド流し</th><th>判定</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>")


def cycle_section(c: dict) -> str:
    r = c.get("result")
    cls = VERDICT_CLASS.get(c["verdict"], "base")
    h = [f'<section class="cycle" id="{escape(c["id"])}"><h3>{escape(c["id"])} <span class="badge {cls}">{escape(c["verdict"])}</span> '
         f'<span class="muted">{escape(c.get("date", ""))}</span></h3><dl>'
         f'<dt>仮説</dt><dd>{escape(c["hypothesis"])}</dd><dt>変えた中身</dt><dd>{escape(c["change"])}'
         + (f'<br><code>{escape(json.dumps(c["spec"], ensure_ascii=False))}</code>' if c.get("spec") else "") + "</dd></dl>"]
    if r:
        ref = r.get("ref")
        h.append(f'<div class="muted">開発の検証 2022〜2024 年（{r.get("races", 0):,} レース）。相手: {escape(ref or "なし（基準そのもの）")}</div>')
        rows = []
        for k, label, _d, rate in METRICS:
            me = (r.get("metrics") or {}).get(k)
            rf = (r.get("ref_metrics") or {}).get(k) if r.get("ref_metrics") else None
            dx = (r.get("diff") or {}).get(k)
            rows.append(f'<tr><td class="l">{label}</td><td class="l">{"大きいほど良い" if rate else "小さいほど良い"}</td>'
                        f'<td>{_metric_val(me, rate)}</td><td>{_metric_val(rf, rate)}</td>'
                        f'<td class="{_diff_class(dx, rate)}">{_ci(dx, pct=rate)}</td></tr>')
        h.append('<div class="tw"><table><thead><tr><th class="l">指標</th><th class="l">良い向き</th><th>この回</th><th>相手</th>'
                 '<th>差 = この回 − 相手（95% 区間）</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>"
                 '<p class="muted">差の色: 区間が 0 をまたがず、良い向きなら緑、悪い向きなら赤。</p>')
        by = r.get("by") or {}
        if by:
            h.append('<div class="tw"><table><thead><tr><th class="l">勝ち馬の対数損失の差の内訳</th><th>差 = この回 − 相手（95% 区間。正なら悪い）</th></tr></thead><tbody>'
                     + "".join(f'<tr><td class="l">{lab}</td><td class="{_diff_class(by.get(k), False)}">{_ci(by.get(k))}</td></tr>' for k, lab in BY)
                     + "</tbody></table></div>")
    else:
        h.append('<p class="muted">結果なし（学習していない、または途中で止めた）。</p>')
    h.append(f'<dl><dt>入力の欠けの検査（学習と推論）</dt><dd>{escape(c.get("parity") or "－")}</dd>'
             f'<dt>結論</dt><dd><b class="{cls}">{escape(c["verdict"])}</b>: {escape(c.get("reason") or "")}</dd>'
             f'<dt>次への示唆</dt><dd>{escape(c.get("next") or "－")}</dd></dl></section>')
    return "".join(h)


def render(log: dict, updated: str | None = None) -> str:
    updated = updated or time.strftime("%Y-%m-%d")
    cyc = log.get("cycles", [])
    base_rows = "".join(
        f'<tr><td class="l">{escape(b["name"])}</td><td>{_num(b["ll"])}</td><td>{_pct(b["top1_win"])}</td><td>{_pct(b["top1_place"])}</td>'
        f'<td>{_pct(b["roi_tansho"])}</td><td>{_pct(b["roi_fukusho"])}</td><td>{_pct(b["roi_umaren"])}</td><td>{_pct(b["roi_wide"])}</td></tr>'
        for b in log.get("baselines", []))
    parts = [f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>親モデルの改善サイクル</title><style>{CSS}</style></head><body><main>
<h1>{escape(log.get("title", "親モデルの改善サイクル"))}</h1>
<p class="muted">最終更新: {escape(updated)}。元の記録: <code>docs/operations/retrain_trials/model_cycles_log.json</code>
（<code>src/scripts/docs/build_model_cycles_html.py</code> で作り直す）。詳しい手順と計画: <code>docs/operations/retrain_trials/model_cycles_20261116.md</code></p>
<p>{escape(log.get("request", ""))}</p>
<div class="cards">
<div class="card"><div class="k">状況</div><div class="v">{escape(log.get("status", ""))}</div></div>
<div class="card"><div class="k">今いちばん良い版</div><div class="v">{escape((log.get("best") or {}).get("name", "－"))}</div><div class="muted">{escape((log.get("best") or {}).get("note", ""))}</div></div>
<div class="card"><div class="k">本番 v6e との差</div><div class="v">{escape(log.get("vs_prod", "－"))}</div></div>
<div class="card"><div class="k">全体の結論</div><div class="v">{escape(log.get("conclusion", "－"))}</div></div>
</div>
<h2>推移</h2>
{chart(cyc)}
{trend_table(cyc)}
<h2>基準</h2>
<div class="tw"><table><thead><tr><th class="l">版</th><th>勝ち馬の対数損失</th><th>◎勝率</th><th>◎複勝率</th><th>単勝</th><th>複勝</th><th>馬連流し</th><th>ワイド流し</th></tr></thead>
<tbody>{base_rows}</tbody></table></div>
<p class="muted">{escape(log.get("baseline_note", ""))}</p>
{"".join(f'<h2>{escape(n.get("title", ""))}</h2>' + "".join(f"<p>{escape(x)}</p>" for x in n.get("body", [])) for n in log.get("notes", []))}
<h2>決まり（サイクルを回す前に固定）</h2><ul>{"".join(f"<li>{escape(x)}</li>" for x in log.get("rules", []))}</ul>
"""]
    for ph, title in PHASES.items():
        cs = [c for c in cyc if c.get("phase", "main") == ph]
        if cs:
            parts.append(f"<h2>{escape(title)}</h2>" + "".join(cycle_section(c) for c in cs))
    parts.append("</main></body></html>\n")
    return "".join(parts)


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    log = json.loads(src.read_text(encoding="utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(log), encoding="utf-8")
    print(f"書き出し: {out}（サイクル {len(log.get('cycles', []))} 件）")


if __name__ == "__main__":
    main()
