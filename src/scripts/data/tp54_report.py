"""tp-54 の結果（results.json / pace_eval.json / v5_select_tp54/results.json）を章 tracking_difficulty_v2.html にする。

python -m src.scripts.data.tp54_report --results <results.json> --pace <pace_eval.json> [--v5 <v5_select_tp54/results.json>]
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CHAPTER = ROOT / "docs/html/modeling/chapters/tracking_difficulty_v2.html"


def _f(x, nd=3) -> str:
    return "—" if x is None or x != x else f"{x:+.{nd}f}"


def _table(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{th}</tr>{body}</table>"


def build(results: dict, pace: dict, v5: dict | None, intro: str = "") -> str:
    parts = ["<h1>追走難度 v2（tp-54）</h1>", intro]
    by = results.get("by_year", {})
    rows = []
    for y, r in by.items():
        rows.append([y, str(r["n_test"]), _f(r["v2_full"]["corr"]), _f(r["v2_demand_only"]["corr"]), _f(r["baseline_current_ease"]["corr_neg_ease"]),
                     _f(r["baseline_current_stamina"]["corr_stamina"]), _f(r["baseline_position"]["corr_pf_early"]), f"{r['current_ease_share_ge60']*100:.1f}% / {r['current_ease_share_lt30']*100:.1f}%"])
    parts.append("<h2>実測の『終いの落ち』Y を予測できるか（walk-forward）</h2>")
    parts.append(_table(["評価年", "件数", "v2（要求＋供給）の相関", "v2（要求のみ）", "現行 ease の相関（符号を反転）", "現行 消耗 の相関", "想定位置取り単独", "現行 ease ≥60 / <30 の割合"], rows))
    for y, r in by.items():
        parts.append(f"<h3>{y} 年: 予測の十分位ごとの実測 Y（秒）</h3>")
        d1 = {d['q']: d['mean_Y'] for d in r["deciles_by_pred"]}
        d2 = {d['q']: d['mean_Y'] for d in r["deciles_by_current_ease"]}
        parts.append(_table(["十分位（0=楽 … 9=大変）", "v2 の実測 Y", "現行 ease の実測 Y"], [[str(q), _f(d1.get(q)), _f(d2.get(q))] for q in range(10)]))
    parts.append("<h2>位置取り×ペースでの整合（予測と実測の平均 Y）</h2>")
    for y, r in by.items():
        g = r["position_pace_grid"]
        parts.append(f"<h3>{y} 年</h3>" + _table(["位置取り", "ペース", "予測 Y", "実測 Y", "件数"], [[x["pos"], x["pace"], _f(x["pred"]), _f(x["actual"]), str(x["n"])] for x in g]))
    parts.append("<h2>前走比（前走と同じ位置を取るなら、今回の条件は前走より楽か）</h2>")
    parts.append(_table(["評価年", "対の数", "同じ位置の対", "相関（全）", "相関（同位置）", "順位相関（同位置）"],
                        [[y, str(r["vs_prev"]["n"]), str(r["vs_prev"]["n_same_pos"]), _f(r["vs_prev"]["corr_all"]), _f(r["vs_prev"]["corr_same_pos"]), _f(r["vs_prev"]["spearman_same_pos"])] for y, r in by.items()]))
    pp = pace.get("by_year", {})
    parts.append("<h2>ペース想定の較正し直し</h2>")
    parts.append(_table(["評価年", "現行 相関", "新 相関", "GBM 相関", "現行 3分類一致", "新 3分類一致"],
                        [[y, _f(r["current"]["corr"]), _f(r["ridge_v2"]["corr"]), _f(r.get("gbm", {}).get("corr")), _f(r["current"]["acc3"]), _f(r["ridge_v2"]["acc3"])] for y, r in pp.items()]))
    dis = pace.get("disagreement", {})
    f0 = pace.get("front0", {})
    parts.append(f"<p>位置取り側ラベルと整合層（現行）の向きが真逆のレース: {dis.get('n_opposite_pf_vs_current')} / {dis.get('n_known')} ({dis.get('opposite_rate', 0)*100:.1f}%)。"
                 f"真逆のとき、実測の3分類に当たる割合は 位置取り側 {dis.get('when_opposite_acc', {}).get('pf_level')}、整合層（現行）{dis.get('when_opposite_acc', {}).get('current')}、新 {dis.get('when_opposite_acc', {}).get('ridge_v2')}。"
                 f"逃げ0頭のレース {f0.get('n')} 件の実測の z 平均 {f0.get('mean_z')}（現行の予測 {f0.get('mean_pred_current')}、新 {f0.get('mean_pred_v2')}）。</p>")
    if v5:
        parts.append("<h2>勝率予測への増分（v5 の列セットに足す）</h2>")
        rows = []
        for k, v in (v5.get("pairs") or {}).items():
            o = v["overall"]
            rows.append([html.escape(k), _f(o["Δwin"], 5), _f(o["win_hi"], 5), _f(o["Δtop2"], 5), _f(o["Δtop3"], 5), "採用" if o["採用"] else "不採用"])
        parts.append(_table(["比較", "Δ勝率", "上限", "Δ連対", "Δ複勝", "判定"], rows))
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--pace", required=True)
    ap.add_argument("--v5")
    ap.add_argument("--intro", default="")
    a = ap.parse_args()
    res = json.loads(Path(a.results).read_text(encoding="utf-8"))
    pace = json.loads(Path(a.pace).read_text(encoding="utf-8"))
    v5 = json.loads(Path(a.v5).read_text(encoding="utf-8")) if a.v5 and Path(a.v5).exists() else None
    body = build(res, pace, v5, a.intro)
    CHAPTER.write_text(f"<!doctype html><html lang=\"ja\"><head><meta charset=\"utf-8\"><title>追走難度 v2（tp-54）</title></head><body>{body}</body></html>", encoding="utf-8")
    print("wrote", CHAPTER)


if __name__ == "__main__":
    main()
