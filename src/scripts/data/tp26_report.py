"""tp-26: 類似ペアの集計（有意 / 概形 / 両方）、食い違いの一覧、コース類似マップ（HTML の章）を作る。

python -m src.scripts.data.tp26_report            # ホストで実行（tp26_eval prep のあと。eval の結果があれば章に載せる）
出力: /tmp/tp26_report.json|md、data/local/modeling/training_pipeline/tp26/pairs_<metric>.csv、docs/html/modeling/chapters/course_performance_similarity.html
"""

from __future__ import annotations

import html
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features import tp26_course_transfer as t

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/local/modeling/training_pipeline/tp26"
CHAPTER = ROOT / "docs/html/modeling/chapters/course_performance_similarity.html"
METRIC_JA = {"fin": "着順の相対位置", "l3f": "上がり3F（速いほど良い）", "pos": "序盤の位置取り（前にいるほど良い）"}


def name(c: str) -> str:
    v, s, d, var = c.split("|")
    venue = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京", "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}.get(v, v)
    return f"{venue}{s}{d}{('・' + var) if var and var != '?' else ('(内外不明)' if var == '?' else '')}"


def reason(geo: pd.DataFrame, i: int, j: int) -> str:
    a, b = geo.iloc[i], geo.iloc[j]
    facts = []
    if a.distance == b.distance:
        facts.append("同じ距離")
    elif abs(a.distance - b.distance) <= 200:
        facts.append(f"距離差{abs(int(a.distance - b.distance))}m")
    if a.venue_code == b.venue_code:
        facts.append("同じ競馬場")
    if a.direction == b.direction:
        facts.append(f"同じ回り({a.direction})")
    else:
        facts.append("回りが違う")
    for k, lab in (("final_straight_m", "直線長"), ("first_curve_m", "最初のコーナーまで")):
        if pd.notna(a[k]) and pd.notna(b[k]):
            facts.append(f"{lab}差{abs(a[k] - b[k]):.0f}m")
    if a.start_geometry != b.start_geometry:
        facts.append("スタート形状が違う")
    return "、".join(facts)


def geo_distance_no_dist(geo: pd.DataFrame) -> np.ndarray:
    return t.geo_distance(geo)


def pair_table(snap: dict, courses: list[str], geo: pd.DataFrame, d: np.ndarray, close: np.ndarray) -> pd.DataFrame:
    iu = np.triu_indices(len(courses), 1)
    rows = []
    for i, j in zip(*iu):
        if not snap["valid"][i, j]:
            continue
        sig = bool(snap["q"][i, j] < t.Q_MAX and snap["r_exc"][i, j] >= t.R_MIN)
        rows.append({"a": courses[i], "b": courses[j], "ia": i, "ib": j, "N": int(snap["N"][i, j]), "r": float(snap["r"][i, j]), "r_exc": float(snap["r_exc"][i, j]),
                     "z": float(snap["z"][i, j]), "q": float(snap["q"][i, j]), "sig": sig, "geo_d": float(d[i, j]), "geo_close": bool(close[i, j]),
                     "same_surface": geo.surface.iloc[i] == geo.surface.iloc[j]})
    return pd.DataFrame(rows)


def enrichment(pt: pd.DataFrame, geo: pd.DataFrame | None = None) -> dict:
    from scipy.stats import fisher_exact, spearmanr

    same = pt[pt["same_surface"]]
    tab = [[int((same.sig & same.geo_close).sum()), int((same.sig & ~same.geo_close).sum())], [int((~same.sig & same.geo_close).sum()), int((~same.sig & ~same.geo_close).sum())]]
    orr, p = fisher_exact(tab)
    rho = spearmanr(same["geo_d"], same["r_exc"]).correlation if len(same) > 3 else float("nan")
    # 概形の近さで「有意」を当てる AUC
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(same["sig"], -same["geo_d"]) if same["sig"].nunique() == 2 else float("nan")
    extra: dict = {}
    if geo is not None and same["sig"].nunique() == 2:
        dd = np.abs(geo["distance"].to_numpy()[same["ia"].to_numpy()] - geo["distance"].to_numpy()[same["ib"].to_numpy()])
        g2 = geo.copy()
        g2["log_dist"] = 0.0
        d2 = geo_distance_no_dist(g2)
        nod = d2[same["ia"].to_numpy(), same["ib"].to_numpy()]
        extra = {"auc_abs_distance_gap": float(roc_auc_score(same["sig"], -dd)), "auc_geo_without_distance": float(roc_auc_score(same["sig"], -nod)),
                 "share_sig_same_or_adjacent_distance": float((dd[same["sig"].to_numpy()] <= 200).mean()), "share_all_same_or_adjacent_distance": float((dd <= 200).mean()),
                 "spearman_abs_distance_gap_vs_r_exc": float(spearmanr(dd, same["r_exc"]).correlation)}
    return {**extra, "valid_pairs": int(len(pt)), "valid_same_surface": int(len(same)), "sig": int(pt.sig.sum()), "geo_close": int(pt.geo_close.sum()),
            "sig_and_close": int((pt.sig & pt.geo_close).sum()), "fisher_table": tab, "odds_ratio": float(orr), "fisher_p": float(p),
            "spearman_geo_dist_vs_r_exc": float(rho), "auc_geo_for_sig": float(auc)}


def heatmap_html(snap: dict, courses: list[str], close: np.ndarray, title: str) -> str:
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    r = snap["r_exc"].copy()
    valid = snap["valid"]
    dist = np.where(valid, 1.0 - np.clip(r / 0.3, -1, 1), 1.0)
    np.fill_diagonal(dist, 0.0)
    order = leaves_list(linkage(squareform((dist + dist.T) / 2, checks=False), "average"))
    sig = valid & (snap["q"] < t.Q_MAX) & (r >= t.R_MIN)
    head = "".join(f"<th class='v'><div>{html.escape(name(courses[j]))}</div></th>" for j in order)
    rows = []
    for i in order:
        cells = []
        for j in order:
            if i == j:
                cells.append("<td class='d'></td>")
                continue
            v = r[i, j] if valid[i, j] else np.nan
            if np.isnan(v):
                cells.append("<td class='n'></td>")
                continue
            a = min(abs(v) / 0.25, 1.0)
            col = f"rgba(248,113,113,{a:.2f})" if v > 0 else f"rgba(96,165,250,{a:.2f})"
            mark = ("●" if close[i, j] else "○") if sig[i, j] else ("△" if close[i, j] else "")
            cells.append(f"<td style='background:{col}' title='{html.escape(name(courses[i]))} × {html.escape(name(courses[j]))}  r_exc={v:+.3f}  q={snap['q'][i, j]:.3g}  N={int(snap['N'][i, j])}'>{mark}</td>")
        rows.append(f"<tr><th class='h'>{html.escape(name(courses[i]))}</th>{''.join(cells)}</tr>")
    return (f"<h3>{html.escape(title)}</h3><div class='hm'><table class='hm'><tr><th></th>{head}</tr>{''.join(rows)}</table></div>"
            "<p class='muted'>赤=超過相関が正（成績が連動）、青=負。● 統計的に有意かつ概形が近い／○ 有意だが概形は近くない／△ 概形は近いが有意でない。</p>")


def main() -> None:
    snaps = pickle.loads((OUT / "snapshots.pkl").read_bytes())
    geo = pd.read_parquet(OUT / "geo.parquet")
    g = np.load(OUT / "geo.npz")
    d, close = g["d"], g["close"]
    courses = geo["course"].tolist()
    report: dict = {"courses": len(courses), "snapshot": "2020〜2025 年の走（date < 2026-01-01）", "metrics": {}}
    sections, tables = [], []
    for m in t.METRICS:
        snap = snaps[(m, 2026)]
        pt = pair_table(snap, courses, geo, d, close)
        pt["a_name"], pt["b_name"] = pt["a"].map(name), pt["b"].map(name)
        pt.drop(columns=["ia", "ib"]).to_csv(OUT / f"pairs_{m}.csv", index=False)
        e = enrichment(pt, geo)
        report["metrics"][m] = e
        far = pt[pt.sig & ~pt.geo_close].sort_values("z", ascending=False).head(8)
        near = pt[pt.geo_close & ~pt.sig & (pt.N >= 300)].sort_values("geo_d").head(8)
        report["metrics"][m]["perf_linked_not_geo_close"] = [{"pair": f"{name(r.a)} × {name(r.b)}", "z": round(r.z, 1), "r_exc": round(r.r_exc, 3), "N": r.N, "note": reason(geo, int(r.ia), int(r.ib))} for r in far.itertuples()]
        report["metrics"][m]["geo_close_not_linked"] = [{"pair": f"{name(r.a)} × {name(r.b)}", "z": round(r.z, 1), "r_exc": round(r.r_exc, 3), "N": r.N, "geo_d": round(r.geo_d, 2), "note": reason(geo, int(r.ia), int(r.ib))} for r in near.itertuples()]
        top = pt[pt.sig & pt.geo_close].sort_values("z", ascending=False).head(10)
        report["metrics"][m]["supported_pairs_top"] = [{"pair": f"{name(r.a)} × {name(r.b)}", "z": round(r.z, 1), "r_exc": round(r.r_exc, 3), "N": r.N} for r in top.itertuples()]
        if m in ("pos", "l3f"):
            sections.append(heatmap_html(snap, courses, close, f"コース類似マップ — {METRIC_JA[m]}"))
    (Path("/tmp") / "tp26_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    eval_path = OUT / "impact/report.json"
    ev = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else None
    (OUT / "report_summary.json").write_text(json.dumps({"pairs": report, "eval": ev}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    md = ["# tp-26 コース転移の結果", ""]
    for m, e in report["metrics"].items():
        ex = (f"\n- 距離差だけで有意を当てる AUC {e['auc_abs_distance_gap']:.3f}、距離を除いた概形の AUC {e['auc_geo_without_distance']:.3f}（概形全体 {e['auc_geo_for_sig']:.3f}）。有意なペアの {e['share_sig_same_or_adjacent_distance']:.0%} が同距離か 200m 差以内（全ペアでは {e['share_all_same_or_adjacent_distance']:.0%}）" if "auc_abs_distance_gap" in e else "")
        md.append(f"## {METRIC_JA[m]}\n- 有効な対 {e['valid_pairs']}（同じ面 {e['valid_same_surface']}）／有意 {e['sig']}／概形が近い {e['geo_close']}／両方 {e['sig_and_close']}\n"
                  f"- 概形の近さと有意の関係: オッズ比 {e['odds_ratio']:.2f}（Fisher p={e['fisher_p']:.3g}）、概形距離と超過相関の Spearman {e['spearman_geo_dist_vs_r_exc']:+.3f}、AUC {e['auc_geo_for_sig']:.3f}" + ex)
    if ev:
        md.append("\n## 勝率への増分\n" + "\n".join(f"- {k}: Δwin {v['Δwin']:+.5f}（上限 {v['win_hi']:+.5f}）Δtop2 {v['Δtop2']:+.5f} Δtop3 {v['Δtop3']:+.5f} {'採用' if v.get('採用') else '不採用'}" for k, v in ev["compare"].items()))
    Path("/tmp/tp26_report.md").write_text("\n".join(md), encoding="utf-8")
    CHAPTER.write_text(build_chapter(report, ev, sections), encoding="utf-8")
    print("保存: /tmp/tp26_report.md /tmp/tp26_report.json", CHAPTER)


def build_chapter(report: dict, ev: dict | None, sections: list[str]) -> str:
    def li(items):
        return "".join(f"<tr><td>{html.escape(x['pair'])}</td><td class='num'>{x.get('r_exc', '')}</td><td class='num'>{x.get('z', '')}</td><td class='num'>{x.get('N', '')}</td><td>{html.escape(x.get('note', ''))}</td></tr>" for x in items)

    mt = report["metrics"]
    concl = ""
    if ev:
        cp = ev["compare"]
        stat, stat3, exact, andv = cp.get("V0+STAT − V0", {}), cp.get("V0+S3+STAT − V0+S3", {}), cp.get("V0+EXACT − V0", {}), cp.get("V0+AND − V0", {})
        concl = (
            "<h2>結論</h2><ul>"
            f"<li><b>数値的に連動するコースは存在する。</b>両方を走った馬 {t.N_MIN} 頭以上の対（約 {mt['fin']['valid_pairs']}）のうち、FDR 5% で有意かつ超過相関 0.05 以上は、着順 {mt['fin']['sig']}、上がり3F {mt['l3f']['sig']}、序盤の位置取り {mt['pos']['sig']} 対。</li>"
            f"<li><b>概形は『裏付け』にはなるが主因ではない。</b>概形が近いペアは有意になりやすい（オッズ比 着順 {mt['fin']['odds_ratio']:.1f}・上がり {mt['l3f']['odds_ratio']:.1f}・位置取り {mt['pos']['odds_ratio']:.1f}）が、"
            f"有意なペアの {mt['pos']['share_sig_same_or_adjacent_distance']:.0%}（位置取り）・{mt['l3f']['share_sig_same_or_adjacent_distance']:.0%}（上がり）は同距離か 200m 差以内で、距離差だけの AUC（{mt['pos']['auc_abs_distance_gap']:.2f}・{mt['l3f']['auc_abs_distance_gap']:.2f}）が概形全体（{mt['pos']['auc_geo_for_sig']:.2f}・{mt['l3f']['auc_geo_for_sig']:.2f}）を上回る。"
            f"距離を除いた概形が効くのは序盤の位置取りだけ（AUC {mt['pos']['auc_geo_without_distance']:.2f}、概形距離と超過相関の Spearman {mt['pos']['spearman_geo_dist_vs_r_exc']:+.2f}）で、スタートから最初のコーナーまでの形が位置取りの転移を支えている。</li>"
            f"<li><b>特徴量にしても勝率は上がらない（全変種が不採用）。</b>最良の STAT（統計で絞る）は V0 比 {stat.get('Δwin', 0):+.5f}（上限 {stat.get('win_hi', 0):+.5f}）で効果が基準 0.0002 に届かず、現行の前段 3 列に重ねると {stat3.get('Δwin', 0):+.5f}（上限 {stat3.get('win_hi', 0):+.5f}）。"
            f"概形で絞る GEO は 0、統計と概形の積 AND は {andv.get('Δwin', 0):+.5f}。対照の EXACT（自分のコースの再中心化した実績）が {exact.get('Δwin', 0):+.5f} で STAT と同程度なので、<b>転移で借りた情報は、既存のコース別実績（b_aff_*）に新しいものを足していない</b>。</li>"
            "<li>使い道: 特徴量としては入れない（tp-15 と同じ結論）。コース類似マップは、コースの説明・位置取り前段の入力の設計材料として残す。</li></ul>")
    blocks = []
    for m, e in report["metrics"].items():
        auc_note = (f"<p class='muted'>有意を当てる AUC: 概形全体 {e['auc_geo_for_sig']:.3f} ／ 距離差だけ {e['auc_abs_distance_gap']:.3f} ／ 距離を除いた概形 {e['auc_geo_without_distance']:.3f}。"
                    f"有意なペアの {e['share_sig_same_or_adjacent_distance']:.0%} が同距離か 200m 差以内（全ペアでは {e['share_all_same_or_adjacent_distance']:.0%}）。</p>") if "auc_abs_distance_gap" in e else ""
        blocks.append(
            f"<h2>{METRIC_JA[m]}</h2><table><tr><th>有効な対</th><th>有意</th><th>概形が近い</th><th>両方</th><th>概形が近いほど連動するか（Spearman）</th><th>オッズ比（Fisher p）</th></tr>"
            f"<tr><td class='num'>{e['valid_pairs']}</td><td class='num'>{e['sig']}</td><td class='num'>{e['geo_close']}</td><td class='num'>{e['sig_and_close']}</td>"
            f"<td class='num'>{e['spearman_geo_dist_vs_r_exc']:+.3f}</td><td class='num'>{e['odds_ratio']:.2f}（{e['fisher_p']:.2g}）</td></tr></table>" + auc_note
            + f"<h3>概形でも統計でも裏付けられたペア（上位）</h3><table><tr><th>ペア</th><th>超過相関</th><th>z</th><th>両方を走った馬</th><th></th></tr>{li(e['supported_pairs_top'])}</table>"
            f"<h3>成績は連動するが、概形では近くないペア</h3><table><tr><th>ペア</th><th>超過相関</th><th>z</th><th>馬</th><th>概形の差（仮説の材料）</th></tr>{li(e['perf_linked_not_geo_close'])}</table>"
            f"<h3>概形は近いが、成績が連動しないペア</h3><table><tr><th>ペア</th><th>超過相関</th><th>z</th><th>馬</th><th>概形の差</th></tr>{li(e['geo_close_not_linked'])}</table>")
    ev_html = ""
    if ev:
        rows = "".join(f"<tr><td>{html.escape(k)}</td><td class='num'>{v['Δwin']:+.5f}</td><td class='num'>{v['win_hi']:+.5f}</td><td class='num'>{v['Δtop2']:+.5f}</td><td class='num'>{v['Δtop3']:+.5f}</td><td>{'採用' if v.get('採用') else '不採用'}</td></tr>" for k, v in ev["compare"].items())
        lv = "".join(f"<tr><td>{html.escape(k)}</td><td class='num'>{v['logloss_win']:.5f}</td><td class='num'>{v['logloss_top2']:.5f}</td><td class='num'>{v['logloss_top3']:.5f}</td></tr>" for k, v in ev["levels"].items())
        ev_html = (f"<h2>勝率予測への増分（LightGBM 固定、valid 2022〜2024 の expanding、2 seed。test 2025 は未使用）</h2><table><tr><th>比較</th><th>Δ勝率</th><th>95%上限</th><th>Δ連対</th><th>Δ複勝</th><th>採否</th></tr>{rows}</table>"
                   f"<table><tr><th>変種</th><th>勝率</th><th>連対</th><th>複勝</th></tr>{lv}</table>"
                   "<p class='muted'>採否の基準: 勝率の差の 95% 上限 &lt; 0、効果 ≥ 0.0002、連対・複勝の悪化が 0.0002 以内。変種は ALL（全ペア）／STAT（統計）／GEO（概形）／AND／OR、EXACT（自分のコース）、ARCH（会場を問わない原型）。</p>")
    css = """:root { --bg:#0f172a; --panel:#15213c; --border:#243358; --text:#e6ecff; --muted:#9aa7c7; --accent:#4f8bff; --accent-2:#7aa9ff; --warn:#fbbf24; }
* { box-sizing: border-box; } body { font-family: "Noto Sans JP", system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.8; font-size: 15px; padding: 40px 56px 96px; max-width: 1100px; margin: 0 auto; }
a { color: var(--accent-2); } h1 { font-size: 26px; } h2 { font-size: 20px; margin: 40px 0 12px; border-left: 4px solid var(--accent); padding-left: 12px; } h3 { font-size: 16px; margin: 22px 0 8px; }
table { border-collapse: collapse; margin: 12px 0 18px; font-size: 13px; background: var(--panel); } th, td { border: 1px solid var(--border); padding: 5px 10px; text-align: left; vertical-align: top; } th { background: #1b2a4d; }
td.num { text-align: right; font-family: monospace; } .muted { color: var(--muted); font-size: 13px; } .note { background: var(--panel); border: 1px solid var(--border); border-left: 4px solid var(--warn); border-radius: 8px; padding: 12px 16px; margin: 14px 0; }
.hm { overflow: auto; max-height: 760px; border: 1px solid var(--border); } table.hm { font-size: 9px; margin: 0; } table.hm td { width: 14px; height: 14px; min-width: 14px; padding: 0; text-align: center; font-size: 8px; line-height: 14px; border: 1px solid #1b2a4d; }
table.hm th.v { height: 150px; padding: 0; vertical-align: bottom; } table.hm th.v div { writing-mode: vertical-rl; white-space: nowrap; font-size: 9px; padding: 2px 0; } table.hm th.h { white-space: nowrap; font-size: 9px; padding: 0 4px; text-align: right; position: sticky; left: 0; }
td.d { background: #243358; } td.n { background: #0b1428; }"""
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>成績が連動するコースの対応づけ（tp-26）— keiba-vpn</title><style>{css}</style></head>
<body><a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>成績が連動するコースの対応づけ（tp-26）</h1>
<p class="muted">馬の過去の成績から「あるコースで好走する馬は別のコースでも好走するか」を数値化し、コース概形（コースマスタ）または統計的検定で裏付けられるペアに絞って特徴量にできるかを調べた記録。データは 2020〜2025 年の中央競馬（{report['courses']} コース。300 走以上）。</p>
<div class="note"><b>方法.</b> コース A, B の両方を走った馬について、馬ごとの『コース別の平均成績 − その馬の全走平均』の相関を取る。走数の少ない馬では全走平均に自分の走が入るため負の偏りが出るので、同じ馬の走のコースの割り当てを入れ替えた帰無分布（40 回）から期待値と標準偏差を求め、
 超過相関 r_exc = r − 帰無の平均、z = r_exc / 帰無の標準偏差、片側 p、Benjamini–Hochberg の q を出す。絞り込みは「統計（q&lt;0.05 かつ r_exc≥0.05）」「概形（同じ面の中で形状距離の近い 8 コース）」「その積／和」「全ペア」。成績は着順の相対位置・上がり3F・序盤の位置取り（いずれもレース内で相対化）。
 特徴量は、行の年の 1 月 1 日より前の走だけで作った類似度と、その行の日付より前の馬の成績だけを使う（先読みなし）。</div>
{concl}{''.join(blocks)}
<h2>コース類似マップ</h2>
{''.join(sections)}
{ev_html}
<p class="muted">コード: <code>src/pipeline/features/tp26_course_transfer.py</code>、<code>src/scripts/data/tp26_eval.py</code>、<code>tp26_report.py</code>。結果の生データ: <code>data/local/modeling/training_pipeline/tp26/</code>。</p>
</body></html>"""


if __name__ == "__main__":
    main()
