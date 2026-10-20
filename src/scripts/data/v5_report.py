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


def ens_section(e: dict, v: dict) -> str:
    if not e and not v:
        return ""
    rows = []
    for k, x in e.items():
        if k.startswith("vs_") and isinstance(x, dict):
            rows.append(f"<tr><td>v5（LOYO） {html.escape(k)}</td><td class=num>{_f(x.get('Δwin'))}</td><td class=num>{_f(x.get('win_hi'))}</td><td class=num>{_f(x.get('Δtop2'))}</td><td class=num>{_f(x.get('Δtop3'))}</td></tr>")
    if v.get("overall"):
        o = v["overall"]
        rows.append(f"<tr><td><b>v5 vs v4（同じ {v.get('rows', 0):,} 行）</b></td><td class=num>{_f(o.get('Δwin'))}</td><td class=num>{_f(o.get('win_hi'))}</td><td class=num>{_f(o.get('Δtop2'))}</td><td class=num>{_f(o.get('Δtop3'))}</td></tr>")
        for y in ("2022", "2023", "2024"):
            if y in v:
                rows.append(f"<tr><td>　{y} 年</td><td class=num>{_f(v[y].get('Δwin'))}</td><td class=num>{_f(v[y].get('win_hi'))}</td><td class=num>{_f(v[y].get('Δtop2'))}</td><td class=num>{_f(v[y].get('Δtop3'))}</td></tr>")
    return f"""<h2>2. LOYO アンサンブル（7 メンバー、valid 2022〜24。メンバーの OOF は V5_GROUPS=CY,BEST の特徴量で作り直し）</h2>
<table><tr><th>比較</th><th>Δwin</th><th>95% CI 上限</th><th>Δtop2</th><th>Δtop3</th></tr>{''.join(rows)}</table>"""


def final_section(f: dict) -> str:
    if not f:
        return ""
    ll, ref, vs = f.get("logloss", {}), f.get("reference", {}), f.get("vs_v4", {})
    return f"""<h2>3. 最終学習（test 2025 の 1 回評価。選択には使っていない）</h2>
<table><tr><th>指標</th><th>v5</th><th>v4</th></tr><tr><td>勝率 LogLoss</td><td class=num>{ll.get('win', float('nan')):.5f}</td><td class=num>{ref.get('v4_win', float('nan')):.5f}</td></tr>
<tr><td>連対</td><td class=num>{ll.get('top2', float('nan')):.5f}</td><td></td></tr><tr><td>複勝</td><td class=num>{ll.get('top3', float('nan')):.5f}</td><td></td></tr></table>
<p>v5 − v4（同じ行）: Δwin {_f(vs.get('Δwin'))}（95% CI 上限 {_f(vs.get('win_hi'))}）、Δtop2 {_f(vs.get('Δtop2'))}、Δtop3 {_f(vs.get('Δtop3'))}。</p>"""


def serving_section(s: dict, px: dict, e2e: dict | None = None) -> str:
    if not s and not px and not e2e:
        return ""
    rows = ""
    for d, v in (s or {}).items():
        rows += f"<tr><td>CY（tp-40）</td><td>{d}</td><td class=num>{v['rows']}</td><td class=num>{v['cols']}</td><td class=num>{v['bad']}</td><td>{html.escape(json.dumps(v.get('worst', {}), ensure_ascii=False)[:160])}</td></tr>"
    for d, v in (px or {}).items():
        for g in ("PX_cell_sd", "PX_emb", "PX_ridge_als"):
            if g in v:
                x = v[g]
                rows += f"<tr><td>{g}（tp-38）</td><td>{d}</td><td class=num>{x['rows']}</td><td class=num>{x['cols']}</td><td class=num>{x['bad']}</td><td>{html.escape(json.dumps(x.get('worst', {}), ensure_ascii=False)[:160])}</td></tr>"
    e2e_rows = ""
    for d, v in (e2e or {}).items():
        e2e_rows += f"<tr><td>{d}</td><td class=num>{v['rows']}</td>" + "".join(f"<td class=num>{v[t]['corr']:.4f}</td><td class=num>{v[t]['diff']:+.4f}</td>" for t in ("win", "top2", "top3")) + f"<td class=num>{v['sec']}</td></tr>"
    e2e_tbl = ("<h3>通しの経路（V3Serving + bundle_v5_2024）の確率と、最終学習の学習側の予測（test 2025）との対比</h3>"
               "<table><tr><th>日</th><th>行</th><th>勝 相関</th><th>勝 LogLoss 差</th><th>連対 相関</th><th>連対 LogLoss 差</th><th>複勝 相関</th><th>複勝 LogLoss 差</th><th>秒</th></tr>" + e2e_rows + "</table>"
               "<p class=\"muted\">V0 の列（LayerA など）は推論の経路で作るので、学習表の値と完全一致はしない（v4 の同じ検証は相関 0.9987）。LogLoss 差は 1 日 80〜100 行で、符号がばらつく雑音の水準。</p>") if e2e_rows else ""
    return f"""<h2>4. 推論側の一致検証（過去の日）</h2>
<p class="muted">推論の経路（serving_v5 の CY、serving_v5_px の ALL_BEST）が、履歴 + 当日の行から作る列を、学習側の群の値と比べた（相対 1e-3・絶対 1e-4 以内を一致とする）。CY・セル統計・SVD 射影は一致し、リッジは再学習時の lsqr の収束差で 0.3〜6% の行が僅かにずれる。</p>
<table><tr><th>群</th><th>日</th><th>行</th><th>列</th><th>不一致の列数</th><th>最も不一致の多い列（行の割合）</th></tr>{rows}</table>
<p class="muted">PX の 1 日あたりの計算時間は約 15〜18 秒（セル統計の履歴計算が主）。状態は data/local/inference/state/px_v5/（祖先表・SVD の成分・半年ごとのリッジ）。</p>{e2e_tbl}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", required=True)
    ap.add_argument("--ens")
    ap.add_argument("--ens-vs", dest="ens_vs", help="v5_ens_compare の出力（v5 vs v4 の対比較）")
    ap.add_argument("--final")
    ap.add_argument("--serving")
    ap.add_argument("--serving-px", dest="serving_px")
    ap.add_argument("--e2e")
    ap.add_argument("--note", default="", help="要点（HTML の段落）。省略時は既定の文")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    r = _j(a.select)
    note = a.note or "<p><b>要点.</b> 選択の結果は下の表。採用した構成と理由は決定ログ（tp-51）に書いてある。</p>"
    body = HEAD + f'<div class="note">{note}</div>' + sel_section(r) + ens_section(_j(a.ens), _j(a.ens_vs)) + final_section(_j(a.final)) + serving_section(_j(a.serving), _j(a.serving_px), _j(a.e2e))
    body += '\n<p class="muted">コード: src/scripts/data/v5_select_eval.py、v5_members.py、fit_final_v5.py、register_final_v5.py、verify_v5_serving_cy.py、src/pipeline/features/serving_v5.py。</p></body></html>\n'
    Path(a.out).write_text(body, encoding="utf-8")
    print("保存:", a.out)


if __name__ == "__main__":
    main()
