"""ドリフト週次監視のレポート HTML（スコアカード・管理図・較正曲線・分布比較・層別・バックテスト表）。自己完結の HTML（外部ライブラリなし、SVG はインライン）。"""

from __future__ import annotations

import html

import numpy as np

COLORS = {"green": "#2e7d32", "yellow": "#f9a825", "orange": "#ef6c00", "red": "#c62828", "gray": "#757575"}
CSS = ("body{font-family:sans-serif;margin:20px;max-width:1100px}table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #ccc;padding:3px 8px;font-size:13px}"
       "h2{margin-top:28px;border-left:4px solid #1976d2;padding-left:8px}.chip{display:inline-block;padding:4px 10px;border-radius:4px;color:#fff;font-weight:bold}"
       ".muted{color:#666;font-size:12px}.card{display:inline-block;margin:4px;padding:8px 12px;border-radius:6px;color:#fff;min-width:170px;vertical-align:top}")


def _f(x, n: int = 4, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:+.{n}f}" if sign else f"{x:.{n}f}"


def svg_line(xs: list[str], series: dict[str, list[float]], hlines: dict[str, float] | None = None, w: int = 720, h: int = 190, vband: tuple[int, int] | None = None) -> str:
    vals = [v for s in series.values() for v in s if v is not None and not np.isnan(v)] + list((hlines or {}).values())
    if not vals or not xs:
        return ""
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.08 or 1e-3
    lo, hi = lo - pad, hi + pad
    sx = lambda i: 44 + (w - 56) * i / max(len(xs) - 1, 1)  # noqa: E731
    sy = lambda v: h - 22 - (h - 38) * (v - lo) / (hi - lo)  # noqa: E731
    cols = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    p = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" style="background:#fafafa;border:1px solid #ddd">']
    if vband:
        p.append(f'<rect x="{sx(vband[0]):.1f}" y="0" width="{max(sx(vband[1]) - sx(vband[0]), 1):.1f}" height="{h - 22}" fill="#fff3e0"/>')
    for name, v in (hlines or {}).items():
        p.append(f'<line x1="44" x2="{w - 8}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="#c62828" stroke-dasharray="4 3"/><text x="{w - 10}" y="{sy(v) - 3:.1f}" font-size="10" text-anchor="end" fill="#c62828">{html.escape(name)} {v:.4f}</text>')
    for j, (name, s) in enumerate(series.items()):
        pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(s) if v is not None and not np.isnan(v))
        p.append(f'<polyline fill="none" stroke="{cols[j % 4]}" stroke-width="1.5" points="{pts}"/><text x="48" y="{12 + 11 * j}" font-size="10" fill="{cols[j % 4]}">{html.escape(name)}</text>')
    p.append(f'<text x="44" y="{h - 5}" font-size="10">{html.escape(xs[0])}</text><text x="{w - 8}" y="{h - 5}" font-size="10" text-anchor="end">{html.escape(xs[-1])}</text>')
    p.append(f'<text x="2" y="{sy(hi - pad):.0f}" font-size="10">{hi - pad:.3f}</text><text x="2" y="{sy(lo + pad):.0f}" font-size="10">{lo + pad:.3f}</text></svg>')
    return "".join(p)


def svg_calibration(curves: dict[str, list[dict]], w: int = 340, h: int = 340) -> str:
    allv = [r[k] for c in curves.values() for r in c for k in ("p", "y")]
    if not allv:
        return ""
    m = max(allv) * 1.05
    sx = lambda v: 36 + (w - 46) * v / m  # noqa: E731
    sy = lambda v: h - 26 - (h - 36) * v / m  # noqa: E731
    p = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" style="background:#fafafa;border:1px solid #ddd">',
         f'<line x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(m):.1f}" y2="{sy(m):.1f}" stroke="#999" stroke-dasharray="3 3"/>']
    for j, (name, c) in enumerate(curves.items()):
        col = ["#d62728", "#1f77b4", "#2ca02c"][j % 3]
        pts = " ".join("%.1f,%.1f" % (sx(r["p"]), sy(r["y"])) for r in c)
        p.append(f'<polyline fill="none" stroke="{col}" points="{pts}"/><text x="40" y="{14 + 11 * j}" font-size="10" fill="{col}">{html.escape(name)}</text>')
    p.append(f'<text x="{w / 2:.0f}" y="{h - 6}" font-size="10" text-anchor="middle">予測確率（十分位の平均）→ 実現率 ↑</text></svg>')
    return "".join(p)


def _table(head: list[str], rows: list[list[str]]) -> str:
    return "<table><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in head) + "</tr>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>"


def render_html(rep: dict) -> str:
    st = rep.get("stage", "保留")
    stage_color = {"ok": "green", "注意": "yellow", "警告": "orange", "要対応": "red"}.get(st, "gray")
    perf = rep["performance"]
    seq = perf.get("sequence") or []
    xs = [r["week"] for r in seq]
    lim = perf.get("limits", {})
    P = [f"<html><head><meta charset='utf-8'><title>ドリフト監視</title><style>{CSS}</style></head><body>",
         f"<h1>ドリフト監視 <span class='chip' style='background:{COLORS[stage_color]}'>{html.escape(st)}</span></h1>",
         f"<p class='muted'>作成: {html.escape(rep['asof'])} / 結果が揃っている最新の週: {html.escape(str(rep.get('latest_result_week')))} / 窓: 直近 {rep['window_weeks']} 週 / 基準期間: "
         f"{html.escape(rep['periods']['baseline']['from'])}〜{html.escape(rep['periods']['baseline']['to'])}。"
         "<b>この指標は報告・警報のみ。モデルや閾値の選択・学習には使わない（2026 年はテストセット②）。</b>対処手順は docs/operations/drift_monitor.md。</p>"]
    P.append("<h2>スコアカード</h2>" + "".join(f"<div class='card' style='background:{COLORS.get(c['color'], '#757575')}'><b>{html.escape(c['name'])}</b><br>{html.escape(str(c['detail']))}</div>" for c in rep.get("scorecard", [])))
    lw = perf.get("latest_window", {})
    cm = perf.get("cumulative_since_monitor_start", {})
    if lw:
        ci = lw.get("gap_ci", {})
        rows = []
        for lab, w in (("直近窓", lw), ("監視開始からの累積", cm)):
            if not w:
                continue
            rows.append([lab, str(w["races"]), f"<b>{_f(w['gap'], 4, True)}</b> [{_f(w.get('gap_ci', {}).get('lo'), 4, True)}, {_f(w.get('gap_ci', {}).get('hi'), 4, True)}]",
                         f"{_f(w['ll_win'])} / {_f(w['ll_win_mkt'])}", f"{_f(w['ll_top2'])} ({_f(w['gap_top2'], 4, True)})", f"{_f(w['ll_top3'])} ({_f(w['gap_top3'], 4, True)})",
                         f"{_f(w['brier_win'], 5)} / {_f(w['brier_win_mkt'], 5)}", f"{_f(w['ece'])} / {_f(w['slope'], 3)}", f"{_f(w.get('rank_corr'), 3)} / {_f(w.get('rank_corr_mkt'), 3)}"])
        P.append("<h2>性能（ラベルあり）: モデル−市場の差が主指標</h2><p class='muted'>差＝モデル LogLoss − 市場(確定単勝オッズ)の LogLoss（正が悪い）。連対/複勝の市場値は市場の勝率からの Harville（参考）。"
                 f"窓内のレース数が {perf['min_races_window']} 未満は判定保留。</p>"
                 + _table(["", "レース", "勝率 差 [95%区間]", "勝率 LL（モデル/市場）", "連対 LL（差）", "複勝 LL（差）", "Brier 勝率（モデル/市場）", "ECE / 傾き（勝率）", "順位相関（モデル/市場）"], rows))
    if seq and lim:
        band = None
        P.append("<h2>管理図: 窓の差（8 週）と、週次の EWMA</h2>" + svg_line(xs, {"8週窓の差": [r["gap"] for r in seq], "EWMA(週)": [r["ewma_gap"] for r in seq], "週の差": [r["week_gap"] for r in seq]},
                                                                  {"平均": lim["gap"]["mean"], "+2σ": lim["gap"]["k2_upper"], "+3σ": lim["gap"]["ucl"]}, vband=band))
        P.append("<h2>管理図: CUSUM（週次の差、k=0.5・h=5）</h2>" + svg_line(xs, {"CUSUM": [r["cusum_gap"] for r in seq]}, {"h": 5.0}))
        P.append("<h2>予測の尖り（モデルの対数オッズ SD ÷ 市場）</h2>" + svg_line(xs, {"sd_ratio": [r["sd_ratio"] for r in seq]}, {"-3σ": lim["sd_ratio"]["lcl"], "平均": lim["sd_ratio"]["mean"]}))
        P.append("<h2>較正: ECE（勝率）</h2>" + svg_line(xs, {"ECE": [r["ece"] for r in seq]}, {"+3σ": lim["ece"]["ucl"]}))
        P.append("<h2>警報の段階の推移</h2>" + _table(["週", "レース", "差", "区分", "尖り", "ECE", "傾き", "段階"], [
            [r["week"], str(r["races"]), _f(r["gap"], 4, True), r["gap_class"], _f(r["sd_ratio"], 3), _f(r["ece"]), _f(r["slope"], 3),
             f"<span class='chip' style='background:{COLORS.get({'ok': 'green', '注意': 'yellow', '警告': 'orange', '要対応': 'red'}.get(r['stage'], 'gray'))}'>{html.escape(r['stage'])}</span>"]
            for r in seq[-26:]]))
        P.append(f"<p class='muted'>基準（窓の差）: 平均 {_f(lim['gap']['mean'], 4)}、σ {_f(lim['gap']['sd'], 4)}、+2σ {_f(lim['gap']['k2_upper'], 4)}、+3σ {_f(lim['gap']['ucl'], 4)}。"
                 f"週次の基準: 平均 {_f(perf['weekly_baseline']['gap']['mean'], 4)}、σ {_f(perf['weekly_baseline']['gap']['sd'], 4)}。</p>")
    cal = perf.get("calibration", {})
    if cal.get("window"):
        P.append("<h2>較正曲線（勝率、十分位）</h2>" + svg_calibration({"基準期間": cal["baseline"], "直近窓": cal["window"]}) + svg_calibration({"直近窓 複勝": cal["window_top3"]}))
    lf = rep.get("labelfree", {})
    P.append(f"<h2>先行指標（ラベルなし）: 予測分布 {html.escape(str(lf.get('status')))}</h2>")
    if lf.get("shift"):
        sh = lf["shift"]
        rows = []
        for k in ("p_win", "p_top3"):
            if k in sh:
                rows.append([k, _f(sh[k]["psi"], 3), _f(sh[k]["ks"], 3), _f(sh[k]["mean_base"], 4), _f(sh[k]["mean_cur"], 4)] + [f"{sh[k]['q_base'][q]:.4f} → {sh[k]['q_cur'][q]:.4f}" for q in ("q05", "q50", "q95")])
        rows.append(["最大勝率(レース)", _f(sh["max_p_win"]["psi"], 3), _f(sh["max_p_win"]["ks"], 3), _f(sh["max_p_win"]["mean_base"], 4), _f(sh["max_p_win"]["mean_cur"], 4), "", "", ""])
        P.append(f"<p class='muted'>最新の予測週 {html.escape(str(lf.get('latest_prediction_week')))}。基準＝学習の OOF と test①。結果が出る前にも出せる。</p>" + _table(["", "PSI", "KS", "平均(基準)", "平均(今回)", "5%点", "中央値", "95%点"], rows))
        cm_, bm_ = lf["cur_metrics"], lf["base_metrics"]
        P.append(_table(["", "最大勝率の平均", "エントロピー/log頭数", "対数オッズ SD", "p<1% の割合"], [
            ["基準", _f(bm_["mean_top_p"]), _f(bm_["entropy_rel"]), _f(bm_["logit_sd"]), _f(bm_["share_p_lt_001"])],
            ["今回", _f(cm_["mean_top_p"]), _f(cm_["entropy_rel"]), _f(cm_["logit_sd"]), _f(cm_["share_p_lt_001"])]]))
        if lf.get("new_ids"):
            n = lf["new_ids"]
            P.append(f"<p>新しい ID の出現率（最新週）: 初出走の馬 {_f(n.get('new_horse'), 3)} / 学習期間に出ていない騎手 {_f(n.get('new_jockey'), 3)} / 調教師 {_f(n.get('new_trainer'), 3)}</p>")
    bm = rep.get("benchmark", {})
    if bm.get("status") == "ok":
        P.append("<h2>市場側か自前か: 固定の簡素な基準モデルとの比較</h2>" + _table(["", "基準モデル−市場の差", "SE"], [
            ["直近窓", _f(bm["gap_benchmark"], 4, True), _f(bm["se"], 4)], ["2025 年の同じ月", _f(bm["gap_benchmark_2025_same_months"], 4, True), _f(bm["se_2025"], 4)],
            ["変化", _f(bm["change"], 4, True), _f(bm["change_se"], 4)]]) + f"<p class='muted'>{html.escape(bm.get('note', ''))}</p>")
    cv = rep.get("coverage", {})
    P.append(f"<h2>入力データの供給（調教ページの被覆）: {html.escape(str(cv.get('status')))}</h2>")
    if cv.get("table"):
        P.append("<p class='muted'>学習側(2020-2025)は約 100%。低いと tl_・wc_ が欠損のまま予測され、予測が平らになる。</p>"
                 + _table(["月", "行数", "調教の行がある率", "記録がある率"], [[r["ym"], str(r["rows"]), f"{r['block']:.3f}", f"{r['rec']:.3f}"] for r in cv["table"]]))
    f = rep.get("features", {})
    P.append(f"<h2>特徴量（PSI 上位）: {html.escape(str(f.get('status')))}</h2>")
    if f.get("top_psi"):
        P.append(_table(["列", "PSI", "KS", "欠損(基準)", "欠損(今回)"], [[html.escape(str(r["feature"])), f"{r['psi']:.3f}", f"{r['ks']:.3f}", f"{r['miss_ref']:.3f}", f"{r['miss_cur']:.3f}"] for r in f["top_psi"]]))
    stt = rep.get("strata", [])
    if stt:
        P.append("<h2>層別（参考。FDR 補正 q<0.10 を強調）</h2><p class='muted'>直近窓の差を、基準期間の同じ層の差と比べた z 検定（片側）を BH で補正。多重比較のため、層別は原因の手がかりとして読む。</p>"
                 + _table(["層", "値", "行", "差(直近)", "差(基準)", "z", "q"], [[r["strata"], html.escape(r["value"]), str(r["rows"]), _f(r["gap_cur"], 4, True), _f(r["gap_base"], 4, True), _f(r["z"], 2),
                                                                         ("<b>" if r["flag"] else "") + _f(r["q"], 3) + ("</b>" if r["flag"] else "")] for r in stt[:18]]))
    sh5 = rep.get("shadow_v5", {})
    P.append(f"<h2>v4 と v5 の shadow: {html.escape(str(sh5.get('status')))}</h2>")
    if sh5.get("status") == "ok":
        P.append(_table(["指標", "差(v5−v4)", "95%区間", "レース数"], [[k, _f(v.get("delta"), 5, True), f"[{_f(v.get('lo'), 5, True)}, {_f(v.get('hi'), 5, True)}]", str(v.get("races"))]
                                                              for k, v in sh5["delta_v5_minus_v4"].items()]) + f"<p class='muted'>{html.escape(sh5.get('note', ''))}</p>")
    P.append("<h2>警報の段階</h2><p class='muted'>注意＝1 項目が基準期間の 2σ を超えた／警告＝独立な 2 群以上が連続 2 週で超えた／要対応＝モデル−市場の差が 3σ 超えを 4 週連続、または較正のずれが回収率に効く水準"
             "（ECE 勝率>0.012 か logit 傾きが 0.85〜1.15 の外）を 2 週連続。再学習は単発の悪化では動かさない。各段階の対処は docs/operations/drift_monitor.md。</p></body></html>")
    return "".join(P)
