"""v5 の構成選択・LOYO アンサンブル・最終学習・推論の一致検証の結果から、章 docs/html/modeling/chapters/ensemble_v5.html を作る。

python -m src.scripts.data.v5_report --select <v5_select/results.json> [--ens <tp27_v5/ens/v5.json>] [--final <final_eval_v5/report.json>] [--serving <verify_v5_cy.json>] [--out <html>]
結果のファイルがあるところだけ節を出す（途中経過でも作り直せる）。文章の解釈はここに書いてある（数値は JSON から）。
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/html/modeling/chapters/ensemble_v5.html"

HEAD = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>アンサンブル v5 の構成選択と最終学習 — keiba-vpn</title>
<style>:root { --bg:#0f172a; --panel:#15213c; --border:#243358; --text:#e6ecff; --muted:#9aa7c7; --accent:#4f8bff; --accent-2:#7aa9ff; --warn:#fbbf24; }
* { box-sizing: border-box; } body { font-family: "Noto Sans JP", system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.8; font-size: 15px; padding: 40px 56px 96px; max-width: 1100px; margin: 0 auto; }
a { color: var(--accent-2); } h1 { font-size: 26px; } h2 { font-size: 20px; margin: 40px 0 12px; border-left: 4px solid var(--accent); padding-left: 12px; } h3 { font-size: 16px; margin: 22px 0 8px; }
table { border-collapse: collapse; margin: 12px 0 18px; font-size: 13px; background: var(--panel); } th, td { border: 1px solid var(--border); padding: 5px 10px; text-align: left; vertical-align: top; } th { background: #1a2848; }
td.num { text-align: right; font-family: monospace; } .muted { color: var(--muted); font-size: 13px; } .note { background: var(--panel); border: 1px solid var(--border); border-left: 4px solid var(--warn); border-radius: 8px; padding: 12px 18px; margin: 14px 0; }</style></head>
<body><a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>アンサンブル v5 の構成選択と最終学習</h1>
"""

CONFIG_JP = {
    "a_V4": "V4（基準。tl_ は再取得前）", "b_tlfresh": "V4 + tl_ を再取得版に", "c_CY": "b + CY（tp-40、37 列）", "d_CY_BEST": "c + ALL_BEST（tp-38、43 列）",
    "e_CY_RLC": "c + RL_C（tp-42、11 列＋差分型）", "f_CY_BEST_RLC": "c + ALL_BEST + RL_C",
}


def _f(x, d=5) -> str:
    return "-" if x is None or x != x else f"{x:+.{d}f}"


def _j(p) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8")) if p and Path(p).exists() else {}


def sel_section(r: dict) -> str:
    lv = r.get("levels", {})
    rows = ["<tr><th>構成</th><th>列数</th><th>勝率 LogLoss</th><th>連対</th><th>複勝</th></tr>"]
    for n, c in CONFIG_JP.items():
        if n in lv:
            x = lv[n]
            rows.append(f"<tr><td>{n}: {html.escape(c)}</td><td class=num>{x['cols']}</td><td class=num>{x['levels']['logloss_win']:.5f}</td><td class=num>{x['levels']['logloss_top2']:.5f}</td><td class=num>{x['levels']['logloss_top3']:.5f}</td></tr>")
    t1 = f"<table>{''.join(rows)}</table>"
    prow = ["<tr><th>比較（新 vs 基準）</th><th>Δwin</th><th>95% CI 上限</th><th>Δtop2</th><th>Δtop3</th><th>採否</th><th>年別 Δwin（2022 / 2023 / 2024）</th><th>初出走 Δwin</th></tr>"]
    for k, v in r.get("pairs", {}).items():
        o = v["overall"]
        by = " / ".join(_f(v["by_year"][y]["Δwin"], 4) for y in ("2022", "2023", "2024") if y in v["by_year"])
        s0 = next((x for n, x in v["subsets"].items() if n.startswith("履歴0走")), None)
        prow.append(f"<tr><td>{html.escape(k)}</td><td class=num>{_f(o['Δwin'])}</td><td class=num>{_f(o['win_hi'])}</td><td class=num>{_f(o['Δtop2'])}</td><td class=num>{_f(o['Δtop3'])}</td><td>{'採用' if o['採用'] else '不採用'}</td><td class=num>{by}</td><td class=num>{_f(s0['Δwin']) if s0 else '-'}</td></tr>")
    t2 = f"<table>{''.join(prow)}</table>"
    return f"""<h2>1. 構成の選択（V4 全体への累積、LightGBM、valid 2022・2023・2024、2 seed）</h2>
<p class="muted">採否の基準は既定（95% paired-bootstrap・レース単位の CI 上限 &lt; 0、効果 ≥ 0.0002、連対・複勝の悪化 ≤ 0.0002）。試した構成は 6、対の比較は {len(r.get('pairs', {}))}。test 2025 と 2026 年は選択に使っていない。</p>
{t1}{t2}"""


def ens_section(e: dict) -> str:
    if not e:
        return ""
    rows = []
    for k, v in e.items():
        if k.startswith("vs_") and isinstance(v, dict):
            rows.append(f"<tr><td>{html.escape(k)}</td><td class=num>{_f(v.get('Δwin'))}</td><td class=num>{_f(v.get('win_hi'))}</td><td class=num>{_f(v.get('Δtop2'))}</td><td class=num>{_f(v.get('Δtop3'))}</td></tr>")
    return f"""<h2>2. LOYO アンサンブル（7 メンバー、valid 2022〜24）</h2>
<table><tr><th>比較</th><th>Δwin</th><th>95% CI 上限</th><th>Δtop2</th><th>Δtop3</th></tr>{''.join(rows)}</table>"""


def final_section(f: dict) -> str:
    if not f:
        return ""
    ll, ref, vs = f.get("logloss", {}), f.get("reference", {}), f.get("vs_v4", {})
    return f"""<h2>3. 最終学習（test 2025 の 1 回評価。選択には使っていない）</h2>
<table><tr><th>指標</th><th>v5</th><th>v4</th></tr><tr><td>勝率 LogLoss</td><td class=num>{ll.get('win', float('nan')):.5f}</td><td class=num>{ref.get('v4_win', float('nan')):.5f}</td></tr>
<tr><td>連対</td><td class=num>{ll.get('top2', float('nan')):.5f}</td><td></td></tr><tr><td>複勝</td><td class=num>{ll.get('top3', float('nan')):.5f}</td><td></td></tr></table>
<p>v5 − v4（同じ行）: Δwin {_f(vs.get('Δwin'))}（95% CI 上限 {_f(vs.get('win_hi'))}）、Δtop2 {_f(vs.get('Δtop2'))}、Δtop3 {_f(vs.get('Δtop3'))}。</p>"""


def serving_section(s: dict) -> str:
    if not s:
        return ""
    rows = "".join(f"<tr><td>{d}</td><td class=num>{v['rows']}</td><td class=num>{v['cols']}</td><td class=num>{v['bad']}</td><td>{html.escape(json.dumps(v.get('worst', {}), ensure_ascii=False)[:200])}</td></tr>" for d, v in s.items())
    return f"""<h2>4. 推論側（CY）の一致検証</h2>
<p class="muted">推論の経路（serving_v5.cy_columns_for_races）が作る CY と、学習側の CY（tp40/groups/CY.parquet）を、過去の日で比べる。</p>
<table><tr><th>日</th><th>行</th><th>列</th><th>不一致の列数</th><th>最も不一致の多い列（行の割合）</th></tr>{rows}</table>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", required=True)
    ap.add_argument("--ens")
    ap.add_argument("--final")
    ap.add_argument("--serving")
    ap.add_argument("--note", default="", help="要点（HTML の段落）。省略時は既定の文")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    r = _j(a.select)
    note = a.note or "<p><b>要点.</b> 選択の結果は下の表。採用した構成と理由は決定ログ（tp-51）に書いてある。</p>"
    body = HEAD + f'<div class="note">{note}</div>' + sel_section(r) + ens_section(_j(a.ens)) + final_section(_j(a.final)) + serving_section(_j(a.serving))
    body += '\n<p class="muted">コード: src/scripts/data/v5_select_eval.py、v5_members.py、fit_final_v5.py、register_final_v5.py、verify_v5_serving_cy.py、src/pipeline/features/serving_v5.py。</p></body></html>\n'
    Path(a.out).write_text(body, encoding="utf-8")
    print("保存:", a.out)


if __name__ == "__main__":
    main()
