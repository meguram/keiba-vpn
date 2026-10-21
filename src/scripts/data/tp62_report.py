"""tp-62 の結果から章 docs/html/modeling/chapters/monthly_roi_search.html の表を作る。

python -m src.scripts.data.tp62_report --in <ホストの tp62_out をコピーしたディレクトリ> [--write]
入力: wf_ov.json（EV 族+騎手調教師の条件）, r500_wf.json / r500_sens.json / r500_census.json（的中率下限+500% 族）,
      null_ov_seed*.json / r500_null_seed*.json（帰無）, power*.json（検出力）。無いものは飛ばす。
"""

from __future__ import annotations

import argparse
import glob
import html
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CHAPTER = ROOT / "docs/html/modeling/chapters/monthly_roi_search.html"
MODES = ("A", "B", "C")
MODE_NAME = {"A": "A 下側信頼限界の最大", "B": "B 全訓練年で100%超＋月の勝ち越しを条件に下限最大", "C": "C 訓練回収率の最大（過学習の見本）"}
YEARS = ("2023", "2024", "2025", "2026")


def jl(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else None


def esc(s) -> str:
    return html.escape(str(s))


def pct(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def wf_table(wf: dict, r500: bool = False) -> str:
    head = ["検証年", "選び方", "選ばれた戦略（訓練年だけで選択）", "訓練 ROI", "検証 ROI", "購入R", "勝ち月/月数(購入月)", "A", "B", "B(購入月のみ)"]
    if r500:
        head += ["レース的中 実績/較正後予測"]
    rows = []
    for y in YEARS:
        if y not in wf:
            continue
        for m in MODES:
            r = wf[y].get(m)
            if not r:
                rows.append(f"<tr><td>{y}</td><td>{m}</td><td class='muted'>選べる戦略なし</td>" + "<td></td>" * (len(head) - 3) + "</tr>")
                continue
            t = r["test"]
            ok = lambda b: "<span class='ok'>○</span>" if b else "<span class='ng'>×</span>"
            cells = [y, m, esc(r["strategy"]), pct(r["train_roi"], 0), pct(t["roi"], 0), f"{t['bet_races']:.0f}", f"{t['win_months']}/{t['months']}({t['bought_months']})", ok(t["A"]), ok(t["B"]), ok(t["B_bought"])]
            if r500:
                dt = r.get("detail", {})
                cells.append(f"{pct(dt.get('race_hit_rate'), 0)} / {pct(dt.get('pred_race_hit'), 0)}" if dt.get("bet_races") else "—")
            rows.append("<tr>" + "".join(f"<td class='n'>{c}</td>" if i in (3, 4, 5, 6) else f"<td>{c}</td>" for i, c in enumerate(cells)) + "</tr>")
    return "<table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>" + "".join(rows) + "</table>"


def null_summary(files: list[str]) -> dict | None:
    reps = []
    for f in files:
        j = jl(f)
        if j:
            reps += j["reps"]
    if not reps:
        return None
    out = {"n": len(reps), "cells": {}, "joint": {}, "insample": {}}
    for m in MODES:
        passes = {y: [] for y in YEARS}
        for r in reps:
            for y in YEARS:
                s = r.get(y, {}).get(m) if y in r else None
                passes[y].append(s["test"] if s else None)
        for y in YEARS:
            ts = passes[y]
            n = len(ts)
            out["cells"][(m, y)] = {
                "A": sum(1 for t in ts if t and t["A"]) / n,
                "B": sum(1 for t in ts if t and t["B"]) / n,
                "AB": sum(1 for t in ts if t and t["AB"]) / n,
                "selected": sum(1 for t in ts if t) / n,
                "roi_p50": float(np.nanmedian([t["roi"] for t in ts if t and t["roi"] == t["roi"]] or [np.nan])),
                "roi_p95": float(np.nanpercentile([t["roi"] for t in ts if t and t["roi"] == t["roi"]] or [np.nan], 95)),
            }
        cnt = []
        for i in range(len(reps)):
            cnt.append([1 if (passes[y][i] and passes[y][i]["AB"]) else 0 for y in ("2023", "2024", "2025")])
        cnt = np.array(cnt)
        out["joint"][m] = {"ge1": float((cnt.sum(1) >= 1).mean()), "ge2": float((cnt.sum(1) >= 2).mean()), "all3": float((cnt.sum(1) >= 3).mean()),
                           "2025_2026": float(np.mean([1 if (passes["2025"][i] and passes["2025"][i]["AB"] and passes["2026"][i] and passes["2026"][i]["AB"]) else 0 for i in range(len(reps))]))}
    for y in YEARS:
        v = [r[y]["insample"]["AB"] for r in reps if y in r]
        e = [r[y]["insample"]["eligible"] for r in reps if y in r]
        out["insample"][y] = {"mean": float(np.mean(v)), "p95": float(np.percentile(v, 95)), "eligible": float(np.mean(e))}
    return out


def null_table(ns: dict, wf: dict) -> str:
    head = ["検証年", "選び方", "実データ: A", "B", "AB", "帰無で A を満たす確率", "B", "AB（運だけで通る確率）", "帰無の検証 ROI 中央値 / 95%点"]
    rows = []
    for y in YEARS:
        for m in MODES:
            c = ns["cells"].get((m, y))
            r = (wf.get(y) or {}).get(m)
            if c is None:
                continue
            t = r["test"] if r else None
            ob = lambda k: ("○" if t and t[k] else "×") if t else "—"
            rows.append(f"<tr><td>{y}</td><td>{m}</td><td>{ob('A')}</td><td>{ob('B')}</td><td>{ob('AB')}</td><td class='n'>{pct(c['A'], 0)}</td><td class='n'>{pct(c['B'], 0)}</td>"
                        f"<td class='n'><b>{pct(c['AB'], 1)}</b></td><td class='n'>{pct(c['roi_p50'], 0)} / {pct(c['roi_p95'], 0)}</td></tr>")
    j = "".join(f"<tr><td>{m}</td><td class='n'>{pct(v['ge1'], 1)}</td><td class='n'>{pct(v['ge2'], 1)}</td><td class='n'>{pct(v['all3'], 1)}</td><td class='n'>{pct(v['2025_2026'], 1)}</td></tr>" for m, v in ns["joint"].items())
    ins = "".join(f"<tr><td>{y}</td><td class='n'>{v['eligible']:.0f}</td><td class='n'>{v['mean']:.0f}</td><td class='n'>{v['p95']:.0f}</td><td class='n'>{((wf.get(y) or {}).get('insample') or {}).get('AB', '—')}</td></tr>" for y, v in ns["insample"].items())
    return (f"<p class='muted'>帰無の反復: {ns['n']} 回（市場が正しい世界で同じ選び方を最初からやり直した）。</p>"
            "<table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>" + "".join(rows) + "</table>"
            "<h3>複数年の同時達成（検証年 2023〜2025 で基準 A・B をともに満たした年の数）</h3>"
            "<table><tr><th>選び方</th><th>1 年以上</th><th>2 年以上</th><th>3 年とも</th><th>2025 と 2026 の両方</th></tr>" + j + "</table>"
            "<h3>事後に（その年の中で）基準を満たす戦略がいくつ見つかるか</h3>"
            "<table><tr><th>年</th><th>最低購入数を満たす戦略数</th><th>帰無の平均（A・B を満たす数）</th><th>帰無の 95% 点</th><th>実データ</th></tr>" + ins + "</table>")


def sens_table(sens: dict) -> str:
    rows = []
    for k, v in sens.items():
        cells = []
        for y in YEARS:
            r = (v.get(y) or {}).get("A")
            if not r:
                cells.append("—")
                continue
            t = r["test"]
            cells.append(f"{pct(t['roi'], 0)}（{t['bet_races']:.0f}R、{t['win_months']}/{t['months']}）")
        rows.append(f"<tr><td>{k.replace('_', ' ')}</td>" + "".join(f"<td class='n'>{c}</td>" for c in cells) + "</tr>")
    return "<table><tr><th>水準（レース的中率の下限 H・払戻倍率 M）</th>" + "".join(f"<th>{y}</th>" for y in YEARS) + "</tr>" + "".join(rows) + "</table>"


def census_table(c: dict) -> str:
    rows = []
    for y, lst in c.items():
        for r in lst:
            if not r.get("bet_races"):
                rows.append(f"<tr><td>{y}</td><td>{r['bt']}</td><td class='n'>EV{r.get('theta', 1.0):g}</td><td class='n'>0</td>" + "<td></td>" * 6 + "</tr>")
                continue
            hm = r["hit_multiple_median"]
            rows.append(f"<tr><td>{y}</td><td>{r['bt']}</td><td class='n'>EV{r.get('theta', 1.0):g}</td><td class='n'>{r['bet_races']}</td><td class='n'>{r['mean_points']:.2f}</td>"
                        f"<td class='n'>{pct(r['race_hit_rate'], 0)}</td><td class='n'>{pct(r['pred_race_hit'], 0)}</td><td class='n'>{'—' if hm is None else round(hm, 1)}</td>"
                        f"<td class='n'><b>{pct(r.get('roi'), 0)}</b></td><td class='n'>{r.get('win_months')}/{r.get('months')}({r.get('bought_months')})</td></tr>")
    return ("<table><tr><th>年</th><th>券種</th><th>EV 閾値(較正後)</th><th>満たすレース数（年間）</th><th>平均点数</th><th>レース的中率（実績）</th><th>較正後の予測</th>"
            "<th>的中時の払戻倍率の中央値（対 総賭け金）</th><th>回収率</th><th>勝ち月/月数(購入月)</th></tr>" + "".join(rows) + "</table>")


def power_table(pw: dict) -> str:
    rows = []
    for k, v in pw.items():
        p = v["power"]
        rows.append(f"<tr><td>{k}</td><td>{esc(v['strategy'])}</td>" + "".join(f"<td class='n'>{pct(p[t]['AB'], 0)}</td>" for t in p) + "</tr>")
    ts = list(next(iter(pw.values()))["power"].keys()) if pw else []
    return "<table><tr><th>検証年|選び方</th><th>戦略</th>" + "".join(f"<th>真の回収率 {float(t) * 100:.0f}%</th>" for t in ts) + "</tr>" + "".join(rows) + "</table>"


def build(indir: Path) -> dict:
    parts = {}
    wf = jl(indir / "wf_ov.json") or jl(indir / "wf.json")
    r5 = jl(indir / "r500_wf.json")
    parts["ev_wf"] = wf_table(wf) if wf else ""
    parts["r5_wf"] = wf_table(r5, True) if r5 else ""
    nsa = null_summary(sorted(glob.glob(str(indir / "null_ov_seed*.json"))) or sorted(glob.glob(str(indir / "null_seed*.json"))))
    nsb = null_summary(sorted(glob.glob(str(indir / "r500_null_seed*.json"))))
    parts["ev_null"] = null_table(nsa, wf) if nsa and wf else ""
    parts["r5_null"] = null_table(nsb, r5) if nsb and r5 else ""
    sens = jl(indir / "r500_sens.json")
    parts["sens"] = sens_table(sens) if sens else ""
    cen = jl(indir / "r500_census.json")
    parts["census"] = census_table(cen) if cen else ""
    pw = jl(indir / "power_ov.json") or jl(indir / "power.json")
    parts["power"] = power_table(pw) if pw else ""
    parts["_ns"] = {"ev": nsa, "r5": nsb}
    parts["_wf"] = {"ev": wf, "r5": r5}
    return parts


def monthly_table(wf: dict, title: str) -> str:
    rows = []
    for y in YEARS:
        r = (wf.get(y) or {}).get("A")
        if not r:
            continue
        t = r["test"]
        cells = "".join(f"<td class='n'>{'—' if v is None else f'{100 * v:.0f}'}</td>" for v in t["monthly_roi"])
        rows.append(f"<tr><td>{y}</td><td>{esc(r['strategy'])}</td>{cells}</tr>")
    n = max((len(((wf.get(y) or {}).get('A') or {}).get('test', {}).get('monthly_roi', [])) for y in YEARS), default=12)
    return (f"<h3>{title}</h3><table><tr><th>検証年</th><th>戦略</th>" + "".join(f"<th>{i + 1}月</th>" for i in range(n)) + "</tr>" + "".join(rows) + "</table>"
            "<p class='muted'>月別の回収率（%）。— は購入なし（基準 B では不成立の月として数える）。</p>")


def count_pass(wf: dict, mode: str, key: str) -> tuple[int, int]:
    n = sum(1 for y in YEARS if (wf.get(y) or {}).get(mode))
    k = sum(1 for y in YEARS if (wf.get(y) or {}).get(mode) and wf[y][mode]["test"][key])
    return k, n


def level_null_block(indir: Path, sens: dict | None) -> str:
    rows = []
    for tag, hm in (("H0.25_M3", "H0.25_M3"), ("H0.3_M3", "H0.3_M3")):
        files = sorted(glob.glob(str(indir / f"r500_null_{tag}_seed*.json")))
        ns = null_summary(files) if files else None
        if not ns or not sens or hm not in sens:
            continue
        ab = [(y, (sens[hm].get(y) or {}).get("A")) for y in ("2023", "2024", "2025")]
        passed = sum(1 for y, r in ab if r and r["test"]["AB"])
        rows.append(f"<tr><td>{tag.replace('_', ' ')}</td><td class='n'>{passed}/3</td><td class='n'>{pct(ns['joint']['A']['ge1'], 1)}</td><td class='n'>{pct(ns['joint']['A']['ge2'], 1)}</td><td class='n'>{ns['n']}</td></tr>")
    if not rows:
        return ""
    return ("<h3>感度の中で目についた 2 水準の運の確率（同じ手順で帰無を作り直した）</h3><table><tr><th>水準</th><th>実データで A・B を満たした年（2023〜2025）</th><th>帰無: 1 年以上</th><th>帰無: 2 年以上</th><th>反復数</th></tr>"
            + "".join(rows) + "</table><p class='muted'>水準を 12 通り見た中から事後に選んだ 2 つなので、表の確率に 12 を掛けた程度（最大 12 倍）が妥当な多重比較の見積もり。反復が 60 回なので、0% は『概ね 5% 未満』の意味（上限の見積もり。多重比較後は有意と言えない）。H=25%・M=300% は 2 年（2023・2025）で A・B を満たした唯一の水準で、反復を増やした確認（<code>tp62_r500 null --H 0.25 --M 3 --reps 150</code>）は、ホストの負荷が下がるまで未実施。</p>")


def power_block(pw: dict | None) -> str:
    if not pw:
        return ""
    ts = list(next(iter(pw.values()))["power_2025"].keys())
    rows = []
    for k, v in pw.items():
        pc = v["power_2025"]
        rows.append(f"<tr><td>{k}</td><td>{esc(v['rule'])}</td><td class='n'>{v['bet_races_per_year']:.0f}</td><td class='n'>{pct(v['per_race_mean'] - 1, 1)}</td><td class='n'>{v['per_race_sd']:.2f}</td>"
                    + "".join(f"<td class='n'>{pct(pc[t]['AB'], 0)}</td>" for t in ts)
                    + f"<td class='n'>{v['need']['1.05']['races']:.0f}R（{v['need']['1.05']['years']:.0f} 年）</td><td class='n'>{v['need']['1.1']['races']:.0f}R（{v['need']['1.1']['years']:.0f} 年）</td><td class='n'>{v['need']['1.2']['races']:.0f}R（{v['need']['1.2']['years']:.1f} 年）</td></tr>")
    return ("<table><tr><th>券種</th><th>ルール（水準だけ、絞り込みなし、2025 年の較正）</th><th>年間の購入R</th><th>実績の平均収益率</th><th>1R の回収倍率の標準偏差</th>"
            + "".join(f"<th>真の回収率 {float(t) * 100:.0f}% のとき、その年に A・B を満たす確率</th>" for t in ts)
            + "<th>真の回収率 105% と言い切る（90%）のに要る購入数（年数）</th><th>110%</th><th>120%</th></tr>" + "".join(rows) + "</table>")


def render(indir: Path) -> str:
    parts = build(indir)
    wf_ev, wf_r5 = parts["_wf"]["ev"] or {}, parts["_wf"]["r5"] or {}
    ns_ev, ns_r5 = parts["_ns"]["ev"], parts["_ns"]["r5"]
    pw = jl(indir / "r500_power.json")
    kA_ev = count_pass(wf_ev, "A", "AB")
    kA_r5 = count_pass(wf_r5, "A", "AB")
    kA_r5a = count_pass(wf_r5, "A", "A")
    kA_eva = count_pass(wf_ev, "A", "A")
    def nullp(ns, key):
        return {y: ns["cells"][("A", y)][key] for y in YEARS} if ns else {}
    pr5 = nullp(ns_r5, "AB")
    pev = nullp(ns_ev, "AB")
    jr5 = ns_r5["joint"]["A"] if ns_r5 else {}
    jev = ns_ev["joint"]["A"] if ns_ev else {}
    best23 = (wf_r5.get("2023") or {}).get("A")
    t23 = best23["test"] if best23 else {}
    cen = jl(indir / "r500_census.json") or {}
    um = {y: next((r for r in lst if r["bt"] == "umaren" and r.get("theta") == 1.0), None) for y, lst in cen.items()}
    um_s = " / ".join(f"{y} 年 {pct(r['roi'], 0)}（{r['bet_races']}R、勝ち月 {r['win_months']}/{r['months']}）" for y, r in um.items() if r and r.get("roi") is not None)
    def pz(v, ns):
        n = ns["n"] if ns else 0
        return f"0%（{n} 回中 0 回）" if v == 0 else pct(v, 1)

    hl = lambda d, ns: "、".join(f"{y} 年 {pz(v, ns)}" for y, v in d.items())

    def ins_ev(y, k):
        if k == "real":
            return f"{(wf_ev.get(y) or {}).get('insample', {}).get('AB', '—'):,}"
        v = (ns_ev or {}).get("insample", {}).get(y)
        return "—" if not v else f"{v['mean']:,.0f}" if k == "null" else f"{v['p95']:,.0f}"

    def ins_r5(y, k):
        if k == "real":
            return f"{(wf_r5.get(y) or {}).get('insample', {}).get('AB', '—'):,}"
        v = (ns_r5 or {}).get("insample", {}).get(y)
        return "—" if not v else f"{v['mean']:,.0f}" if k == "null" else f"{v['p95']:,.0f}"

    def hit_pair(y):
        r = (wf_r5.get(y) or {}).get("A") or {}
        dt = r.get("detail") or {}
        return f"{y} 年 {pct(dt.get('race_hit_rate'), 0)}（予測 {pct(dt.get('pred_race_hit'), 0)}、{dt.get('bet_races', 0)}R）" if dt.get("bet_races") else f"{y} 年 —"
    hits_s = "、".join(hit_pair(y) for y in YEARS)
    n_ev_elig = (ns_ev or {}).get("insample", {}).get("2025", {}).get("eligible")
    wd = (pw or {}).get("wide", {}).get("power_2025", {})
    um_p = (pw or {}).get("umaren", {}).get("power_2025", {})
    css = ("<style>:root { --bg:#0f172a; --panel:#15213c; --border:#243358; --text:#e6ecff; --muted:#9aa7c7; --accent:#4f8bff; --accent-2:#7aa9ff; --warn:#fbbf24; --good:#34d399; --bad:#f87171; }\n"
           "* { box-sizing: border-box; } body { font-family: \"Noto Sans JP\", system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.8; font-size: 15px; padding: 40px 56px 96px; max-width: 1250px; margin: 0 auto; }\n"
           "a { color: var(--accent-2); } h1 { font-size: 26px; } h2 { font-size: 20px; margin: 40px 0 12px; border-left: 4px solid var(--accent); padding-left: 12px; } h3 { font-size: 16px; margin: 22px 0 8px; }\n"
           "table { border-collapse: collapse; margin: 12px 0 18px; font-size: 12.5px; background: var(--panel); } th, td { border: 1px solid var(--border); padding: 4px 8px; text-align: left; vertical-align: top; } th { background: #1b2a4d; }\n"
           "td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }\n"
           ".muted { color: var(--muted); font-size: 13px; } .note { background: var(--panel); border: 1px solid var(--border); border-left: 4px solid var(--warn); border-radius: 8px; padding: 12px 16px; margin: 14px 0; }\n"
           ".ok { color: var(--good); } .ng { color: var(--bad); } code { background: #0b1428; padding: 1px 5px; border-radius: 4px; }</style>")
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>月次過半・年間 100% 超の馬券戦略の探索（tp-62）— keiba-vpn</title>
{css}</head>
<body><a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>月次過半・年間 100% 超の馬券戦略の探索（tp-62）</h1>
<p class="muted">2026-10-21、ユーザー依頼『購入機会が減ってもよいので、回収率が 100% を超える月が半数以上、かつ 1 年の回収率が 100% を超えるシミュレーションをもっと探してほしい。回収率分析ベースの戦略を組み込むなど、やれることをやって』、続けて『期待値だけでなく、券種ごとの的中率の下限を用意するべき。レース回収率 500% 以上・的中率 20% 以上が健全』。
資料: <a href="betting_edge_search.html">tp-36</a>・<a href="jockey_trainer_roi.html">tp-59（騎手・調教師）</a>・<a href="proven_methods_roi.html">tp-44</a>・<a href="ev_calibration_adopt.html">tp-49（期待値の較正）</a>・<a href="training_pipeline_decisions.html">決定ログ</a>。
入力は tp-50 のチケット表（レース × 買い目、確定オッズ、V4 の確率と単勝の Benter 混合、2022〜2026 年）。2026 年はテストセット②（選択には使わない）。市場オッズは買い目の評価にだけ使い、特徴量には使わない。</p>

<div class="note"><b>要点（先に答え）.</b>
<ul>
<li><b>基準 A（年間 100% 超）かつ B（100% 超の月が過半）を、『その年より前のデータだけで選んで翌年に使う』形（ウォークフォワード）で、複数年続けて満たす方策は見つからなかった</b>。期待値の族（騎手・調教師の条件つき）では検証年 2023〜2026 の A・B 同時達成が {kA_ev[0]}/{kA_ev[1]} 年、的中率の下限と 500% を使う族（固定水準 H=20%・M=500%）では {kA_r5[0]}/{kA_r5[1]} 年（2023 年のみ。2024・2025 年は 49% 前後に落ち、2026 年は A のみ）。</li>
<li><b>運だけで通る確率（帰無: 市場が正しい世界で、同じ選び方を最初からやり直す）</b>は、検証年 1 年あたり A・B 同時で {hl(pr5, ns_r5)}（的中率下限の族）、{hl(pev, ns_ev)}（期待値の族）。ただし<b>複数年のうち 1 年以上</b>では {pct(jr5.get('ge1'), 1)}（的中率下限の族）で、2023 年だけ通った実績はこの確率に相当する。2 年以上は 0%。<b>探索した族が複数あり（期待値の族・的中率下限の族・水準 12 通り・選び方 3 通り）、2024 年以降で再現しないので、見つかったとは言えない</b>。</li>
<li><b>事後に（その年の中で）基準 A・B を満たす戦略は、購入が少ないほど何百〜何千とある</b>。最低購入数を満たす戦略は 5〜7 万通りで、そのうち年内に A・B を満たす数は、期待値の族の 2025 年で実データ {ins_ev('2025','real')}、帰無の平均 {ins_ev('2025', 'null')}（95% 点 {ins_ev('2025', 'p95')}）、的中率下限の族の 2023 年で実データ {ins_r5('2023', 'real')}、帰無の平均 {ins_r5('2023', 'null')}（95% 点 {ins_r5('2023', 'p95')}）。<b>実データのほうが帰無より多い年がある＝年内には『他より当たった組』が確かにあるが、翌年に持ち越せていない</b>（例: 期待値の族で 2023 年に選ばれたワイド『EV≥2、オッズ 30〜100 倍、先頭 1 点、9 頭立て以下』は訓練年 2022 年で回収率 347%・7 的中だが、上位 5 的中を除くと 56%、検証年は 0%）。『達成した戦略がある』だけでは何も言えず、ウォークフォワードでの再現が要る。</li>
<li><b>『的中率 20% 以上・払戻 500% 以上』を選ぶ前の素の形で見ると、馬連が最も近い</b>（EV 較正後 ≥ 1、5 点まで、絞り込みなし）: {um_s}。4 年とも約 100〜122% だが、100% 超の月は過半に届かない（4〜6 か月）。<b>この水準の統計的な確かさは低い</b>: 1 レースの回収倍率の標準偏差が約 2.9 倍で、真の回収率が 105% だと言い切るには約 5,300 レース（約 100 年分）が要る。</li>
<li><b>基準 B は、真に儲かる戦略でも満たしにくい</b>: 馬連の上記ルールは、真の回収率が 120% あっても A・B を同時に満たす確率が約 7%（年間 50R 弱で、月ごとの購入が数点のため）。ワイド（年間約 240R）は真の回収率 120% で約 50%。月次過半は『買う回数が多く、的中率が高い券種』に偏って通りやすい基準。</li>
<li>馬連のルールは事後に見つけた候補なので採用せず、<code>config/tp62_frozen_rule.json</code> に較正の折れ線ごと凍結して、2026-10-22 以降の新しいレースで前向きに追跡する（確証には年単位でかかる見込み）。</li>
</ul></div>

<h2>1. 何を、どう検証したか</h2>
<p><b>成功基準（先に固定）</b>: A 年間回収率 &gt; 100%。B 100% 超の月がその年の月数の半分超（購入のない月は不成立として数える。購入のある月だけで数えた版も併記）。事後に最良の戦略を選んで『達成』とは数えない。</p>
<p><b>ウォークフォワード</b>: 検証年 Y（2023・2024・2025、最後に 2026）について、Y より前の年だけで戦略とパラメータを選び、Y に適用した月次・年次の回収率を出す。選び方は 3 通り: A＝訓練年の下側信頼限界（90%）の最大、B＝訓練の全年で 100% 超かつ月の勝ち越し 5 割以上の中から下限最大、C＝訓練回収率の最大（過学習の見本）。券種が存在する訓練年すべてで最低購入レース数（期待値の族 40、的中率下限の族 20）を満たすことを条件にする（3連単は 2024 年からなので、2025 年の検証は 2024 年だけで選ぶ）。</p>
<p><b>運の確率（帰無）</b>: 『すべての買い目の期待回収率が控除率相当（的中確率 = 控除率 / オッズ）』という市場が正しい世界で、<b>同じ選び方（全戦略を訓練年で評価して 1 つ選び、検証年に適用）を最初からやり直して</b>、基準 A・B を満たす頻度を数える。単勝・馬連・馬単・3連単はレース内で 1 点だけ的中する（排他）。選び方まで含めて繰り返すので、探索した戦略の数の多重比較が自動的に入る。
<span class="muted">（初版の帰無は、着順を単勝の市場確率から乱数で作る方式で、組合せ券の確定オッズと整合せず、選んだ戦略の検証 ROI の中央値が 150% を超えた＝帰無になっていなかったので捨てて、買い目ごとに公正な期待値とする方式に作り直した。）</span></p>

<h2>2. 戦略の族</h2>
<h3>2-1. 期待値の族（騎手・調教師の条件つき）</h3>
<p>券種（単勝・複勝・馬連・ワイド・馬単・3連単）× スコア（V4 の確率 × オッズ、単勝の Benter 混合 × オッズ）× 閾値 × オッズ帯 × レース内上位 K 点（1・2・3・5）× 賭け方（定額／レース均等）× レースの絞り込み（頭数・本命の強さ・モデルと市場の乖離・混戦度・荒れ型など 16 通り）。加えて tp-59 の as-of 条件（騎手・調教師がともに『当年より前の勝ち数が市場の見込み超』『超過払戻が正』を、先頭の馬だけ／全馬に課す）を重ねた族。全体で、最低購入数を満たす戦略が約 {(f'{int(n_ev_elig):,}' if n_ev_elig else '—')} 通り（検証年ごとに異なる）。</p>
<h3>2-2. 的中率の下限と 500% を使う族（ユーザー指定）</h3>
<p>各レース・各券種で、較正後の的中確率 p<sub>cal</sub>（検証年より前の年で、p<sub>mix</sub> → 的中 を単調回帰。訓練年の行は自分の入らない半分のレースで当てはめる交差適合）が券種別の下限（0・3・5・8・12・20%）以上で、期待値（p<sub>cal</sub> × オッズ）が 1.0 または 1.2 以上の買い目を、EV 降順（または確率降順）に最大 K（1・2・3・5）点まで等額で買う。K 点買うときは次の 2 つを満たすものだけ: <b>的中した買い目の払戻が、そのレースの総賭け金の 5 倍以上</b>（すべての買い目のオッズ ≥ 5K）、<b>いずれかが的中する確率（較正後）が 20% 以上</b>。満たす K が無いレースは買わない。5 倍 × 20% = 100% なので、これは『期待値 1 以上かつ的中率 20% 以上』と同じ向き。水準 20%・500% はユーザー指定のまま固定し、他の水準は参考の別表にする。</p>

<h2>3. 期待値の族の結果</h2>
{parts['ev_wf']}
{monthly_table(wf_ev, '選び方 A で選ばれた戦略の月別回収率（検証年）') if wf_ev else ''}
{parts['ev_null']}

<h2>4. 的中率の下限と 500% の族の結果（固定水準 H=20%・M=500%）</h2>
{parts['r5_wf']}
{monthly_table(wf_r5, '選び方 A で選ばれた戦略の月別回収率（検証年）') if wf_r5 else ''}
{parts['r5_null']}
<p>2023 年に A・B を満たした戦略は『{esc(best23['strategy']) if best23 else ''}』（検証 ROI {pct(t23.get('roi'), 0)}、購入 {t23.get('bet_races', 0):.0f}R、勝ち月 {t23.get('win_months')}/{t23.get('months')}）で、その年の帰無の確率は {pz(pr5.get('2023'), ns_r5)}、複数年のうち 1 年以上通る確率は {pct(jr5.get('ge1'), 1)}。ただし訓練年が 2022 年の 1 年だけで、翌 2024・2025 年は 49% 前後に落ちた。選ばれた戦略のレース的中率（実績 / 較正後の予測）: {hits_s}。<b>2024・2025 年は予測を大きく下回った</b>＝買い目を『較正後の確率や期待値が高い順に選ぶ』選抜が、較正の過大（選択による楽観）を拾っている。</p>

<h3>4-1. 絞り込みなしで水準だけを課した場合（選択の無い形）</h3>
<p>券種ごとに『較正後の的中確率の下限なし、EV ≥ 1.0 または 1.2、EV 降順、5 点まで』で、レース的中率 20% 以上・払戻 500% 以上を満たすレースだけを買う。選ぶ操作がないので、運の確率は 12 通り（券種 6 × 閾値 2）の多重比較だけ。</p>
{parts['census']}
<p class="muted">『較正後の予測』は、各レースの選んだ買い目の的中確率（排他の券種は和、複勝・ワイドは独立の併合）の平均。複勝は、確定オッズの下限寄りの目安を使っていて、実際の払戻（対 総賭け金）の中央値が 3〜4.5 倍にとどまり、500% の条件を満たしていない（目安のオッズが実際より高い）。</p>

<h3>4-2. 水準を動かしたときの感度（参考。選び方 A、水準の選択バイアスあり）</h3>
{parts['sens']}
<p>H=25%・M=300% では 2023 年 112%（勝ち月 7/12）・2025 年 106%（7/12）で A・B を同時に満たし、H=30%・M=300% では 2025 年 148%（8/12）が目を引く。ただし 12 通りの水準を見た中の最良で、2024・2026 年は 100% を下回る。水準は事前指定の 20%・500% を主結果とし、この表は水準探索の参考として扱う。</p>
{level_null_block(indir, jl(indir / 'r500_sens.json'))}

<h2>5. 検出力: 真に儲かる戦略でも、基準 B は満たしにくい</h2>
{power_block(pw)}
<p class="muted">真の回収率 t を作るため、帰無の払戻（期待値 = 控除率）を t / 控除率 倍している。『要る購入数』は、実データの 1 レースの回収倍率の分散から、真の回収率 105%・110%・120% を 90% の確からしさで 100% 超と言うのに要るレース数。</p>
<p>結論: 真の回収率 110% でも、A・B を同時に満たす年は、馬連のルールで {pct(um_p.get('1.1', {}).get('AB'), 0)}、ワイドで {pct(wd.get('1.1', {}).get('AB'), 0)}。120% でも馬連 {pct(um_p.get('1.2', {}).get('AB'), 0)}、ワイド {pct(wd.get('1.2', {}).get('AB'), 0)}。月次の過半は、買う回数が多い券種（ワイド）が有利になる基準で、<b>100% 超を『見つけた』と言える水準の証拠を作るには、5 年分でも足りない</b>（馬連のルールで真の回収率 120% でも約 6.5 年）。</p>

<h2>6. 使えなかったもの・限界</h2>
<ul>
<li>評価は<b>確定オッズ</b>。購入時点（T-15〜T-2）のオッズは確定より動くので、直前オッズ（10/24 から蓄積）で再検証が要る。実運用では、買える量（オッズ変動、馬連・馬単の点数）に上限がある。</li>
<li>tp-60（血統の回収率）と tp-61（展開シナリオ別の有利度）は、このチケット表に過去分の馬ごとの値が無い（tp-61 は事前の確率でも精度の増分が小さい）ので組み込まなかった。tp-59 の騎手・調教師の as-of 条件は期待値の族に組み込み済み（選ばれた戦略に毎回入ったわけではなく、検証でも上乗せはなかった）。</li>
<li>帰無は券種間の相関を入れていない（分散がやや大きめ = 運で通る確率を多めに見る向き）。排他は単勝・馬連・馬単・3連単のみ。</li>
<li>較正後の的中確率の下限は、訓練年の結果で作った単調回帰。選抜で拾った買い目ほど過大になる（上記）ため、レース的中率 20% の制約は、実績では 10〜20% 程度しか満たされない年が多い。</li>
</ul>

<h2>7. 凍結した候補と、次の一手</h2>
<p>最も近い形の『馬連・EV 較正後 ≥ 1.0・レース的中率 20% 以上・払戻 500% 以上・5 点まで・絞り込みなし』（上の 4-1）は、4 年とも約 100〜122% だが、事後に見つけた候補で、月次過半は未達、運の区間も広い。採用せず、較正の折れ線（2022〜2025 年で当てはめ）ごと <code>config/tp62_frozen_rule.json</code> に凍結した。2026-10-22 以降のレースで、確定オッズの時点と直前オッズの時点の両方で前向きに追跡する（確証には、真の回収率が 110% でも約 26 年、120% で約 6.5 年かかる見込みなので、<b>この規則は『確かめる』ためではなく『悪化しないことを確認する』ための監視</b>と位置づける）。</p>
<ul>
<li>月次過半・年間 100% 超を本気で狙うなら、回収率が 100% を超えうる情報の追加が先（直前オッズの動き、血統・展開の履歴を持つチケット表）。この枠での再探索は、直前オッズが約 1 年分（約 3,500R）貯まってから。</li>
<li>画面の推奨には、的中率の下限を足す価値はある（資金の変動を抑える）が、回収率を上げる根拠にはならない。『期待値だけを見ている』点の改善は、リスク管理として別に扱う。</li>
</ul>
</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    out = render(Path(a.indir))
    if a.write:
        CHAPTER.write_text(out, encoding="utf-8")
        print("wrote", CHAPTER, len(out))
    else:
        print(len(out))


if __name__ == "__main__":
    main()
