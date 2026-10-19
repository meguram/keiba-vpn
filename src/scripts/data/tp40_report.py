"""tp-40 の結果（tp40/*.json）から章 docs/html/modeling/chapters/training_condition_growth.html の表の部分を作る。

python -m src.scripts.data.tp40_report <tp40 の結果ディレクトリ> <出力 html>
文章の解釈（要点・結論）は章の本文側に手で書く。ここでは数値の表だけを JSON から生成して `<!-- tp40:tables:begin/end -->` の間に差し込む。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BEGIN, END = "<!-- tp40:tables:begin -->", "<!-- tp40:tables:end -->"


def _j(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _f(x, d=5, sign=True) -> str:
    return "-" if x is None or x != x else (f"{x:+.{d}f}" if sign else f"{x:.{d}f}")


def stage_table(r: dict, title: str) -> str:
    gs = r.get("groups", {})
    if not gs:
        return ""
    base = r.get("base", {}).get("levels", {})
    rows = ["<tr><th>群</th><th>列数</th><th>勝率 LogLoss</th><th>Δwin</th><th>95% CI 上限</th><th>Δtop2</th><th>Δtop3</th><th>採否</th></tr>"]
    for g, s in gs.items():
        c = s["vs_V4"]
        rows.append(f"<tr><td>{g}</td><td class=num>{s['cols']}</td><td class=num>{s['levels']['logloss_win']:.5f}</td><td class=num>{_f(c['Δwin'])}</td><td class=num>{_f(c['win_hi'])}</td><td class=num>{_f(c['Δtop2'])}</td><td class=num>{_f(c['Δtop3'])}</td><td>{'採用' if c['採用'] else '不採用'}</td></tr>")
    return f"<h3>{title}（基準の勝率 LogLoss {base.get('logloss_win', float('nan')):.5f}）</h3><table>{''.join(rows)}</table>"


def subset_table(r: dict, title: str) -> str:
    gs = r.get("groups", {})
    names = ["履歴0走(初出走)", "履歴0〜2走", "履歴3走以上", "新馬戦", "未勝利", "条件戦", "OP以上他"]
    if not gs:
        return ""
    head = "<tr><th>群</th>" + "".join(f"<th>{n}</th>" for n in names) + "</tr>"
    rows = [head]
    for g, s in gs.items():
        cells = []
        for n in names:
            x = s.get("subsets", {}).get(n)
            cells.append(f"<td class=num>{_f(x['Δwin'])}<br><span class=muted>({_f(x['win_hi'])}) {x['rows']:,}行</span></td>" if x else "<td>-</td>")
        rows.append(f"<tr><td>{g}</td>{''.join(cells)}</tr>")
    return f"<h3>{title}（勝率 LogLoss の差 Δwin、括弧は 95% CI 上限）</h3><table>{''.join(rows)}</table>"


def describe_tables(d: dict) -> str:
    out = []
    for i, (f, v) in enumerate(d.get("features", {}).items()):
        keep = ("全体", "履歴0〜2走") + (("最終=slope", "最終=wood", "最終=poly") if i == 0 else ())
        rows = ["<tr><th>集団</th><th>帯</th><th>頭数</th><th>レース数</th><th>1R平均頭数</th><th>馬単位勝率(実/V4)</th><th>レース勝率 実測</th><th>V4 予測</th><th>差</th><th>馬単位複勝率(実/V4)</th></tr>"]
        for r in v["rows"]:
            if r["subset"] not in keep:
                continue
            rows.append(f"<tr><td>{r['subset']}</td><td class=num>{r['bin']}</td><td class=num>{r['runs']:,}</td><td class=num>{r['races']:,}</td><td class=num>{r['mean_horses_per_race']:.2f}</td>"
                        f"<td class=num>{r['horse_win']:.4f} / {r['horse_pred']:.4f}</td><td class=num>{r['race_win']:.4f}</td><td class=num>{r['race_win_pred']:.4f}</td><td class=num>{r['race_win'] - r['race_win_pred']:+.4f}</td>"
                        f"<td class=num>{r['horse_top3']:.4f} / {r['horse_top3_pred']:.4f}</td></tr>")
        out.append(f"<h3>{v['label']}（帯の境界 {', '.join(f'{e:.2f}' for e in v['edges'])}）</h3><table>{''.join(rows)}</table>")
    return "".join(out)


def repro_table(r: dict) -> str:
    rows = ["<tr><th>指標</th><th>馬の数</th><th>前半×後半の相関</th><th>95% 区間</th><th>帰無（着順を馬の中で並べ替え）</th><th>3 分類の一致率</th><th>帰無の一致率</th></tr>"]
    for tag in ("最終の良さ x1", "サイクルの上昇 x2"):
        v = r.get(tag)
        if v:
            rows.append(f"<tr><td>傾き（{tag}）</td><td class=num>{v['horses']:,}</td><td class=num>{v['corr_slope']:+.3f}</td><td class=num>{v['corr_slope_ci'][0]:+.3f}〜{v['corr_slope_ci'][1]:+.3f}</td><td class=num>{v['null_corr']:+.3f}</td>"
                        f"<td class=num>{v['classes']['agree_share']:.3f}</td><td class=num>{v['classes_null']['agree_share']:.3f}</td></tr>")
    for tag in ("サイクルの上昇の馬ごとの平均", "最終の良さの馬ごとの平均"):
        v = r.get(tag)
        if v:
            rows.append(f"<tr><td>{tag}</td><td class=num>{v['horses']:,}</td><td class=num>{v['corr']:+.3f}</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>")
    return f"<table>{''.join(rows)}</table>"


def classes_table(r: dict, tag: str) -> str:
    v = r.get(tag)
    if not v:
        return ""
    sh = v["classes"]["share_a"]
    return "<table><tr><th>前半の傾きの t 値による分類</th><th>割合</th></tr>" + "".join(f"<tr><td>{k}</td><td class=num>{sh.get(k, 0):.3f}</td></tr>" for k in ("増えると上がる", "無関係", "減ると上がる")) + "</table>"


def audit_summary(a: dict) -> str:
    if not a:
        return ""
    out = []
    for mode, title in (("asof", "その日までに存在したデータだけで作った版（追い切りは調教日で切る）"), ("race", "追い切りを『紐づくレースの日付』で切った版（参考: 標準化の基準が後のレースに紐づく追い切りに依存する分）")):
        v = a.get(mode)
        if not v:
            continue
        rows = ["<tr><th>群</th><th>列数</th><th>不一致のある列数</th><th>最大の差</th></tr>"]
        for g, cols in v["groups"].items():
            bad = {c: x for c, x in cols.items() if x["nan_diff_rows"] or x["value_diff_rows"]}
            mx = max([x["max_abs_diff"] for x in cols.values()] or [0.0])
            rows.append(f"<tr><td>{g}</td><td class=num>{len(cols)}</td><td class=num>{len(bad)}</td><td class=num>{mx:.2e}</td></tr>")
        out.append(f"<h4>{title}</h4><p>{v['cut']} で切った版（{v['races']:,} レース）と全期間版を、切った日以前のレースで列ごとに完全一致で比べた。不一致のセル合計: <b>{v['total_mismatch_cells']}</b>。</p><table>{''.join(rows)}</table>")
    return "".join(out)


def _replace(html: str, name: str, body: str) -> str:
    b, e = f"<!-- tp40:{name}:begin -->", f"<!-- tp40:{name}:end -->"
    if b not in html:
        raise SystemExit(f"章に差し込み位置 {name} が無い")
    i, j = html.index(b), html.index(e) + len(e)
    return html[:i] + b + "\n" + body + "\n" + e + html[j:]


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    html = dst.read_text(encoding="utf-8")
    s1, s2 = _j(src / "stage1/results.json"), _j(src / "stage2/results.json")
    blocks = {
        "audit": audit_summary(_j(src / "audit.json")),
        "repro": repro_table(_j(src / "repro.json")) + classes_table(_j(src / "repro.json"), "最終の良さ x1"),
        "stage1": stage_table(s1, "第 1 段（valid 2022+2023・seed 1 本）") + subset_table(s1, "第 1 段: 履歴の浅い馬・クラス別"),
        "stage2": stage_table(s2, "第 2 段（valid 2024・新 seed 2 本）") + subset_table(s2, "第 2 段: 履歴の浅い馬・クラス別"),
        "describe": describe_tables(_j(src / "describe.json")),
    }
    for k, v in blocks.items():
        html = _replace(html, k, v)
    dst.write_text(html, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
