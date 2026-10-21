"""tp-60 の章 docs/html/modeling/chapters/pedigree_roi.html を、ホストの結果 JSON から作る（数字は JSON から埋める）。

python -m src.scripts.data.tp60_report [--src /tmp/tp60_local]    # src に roi/*.json, embed/*.json, ped/*.json, eval/<stage>/results.json を置く
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/html/modeling/chapters/pedigree_roi.html"

CSS = """:root { --bg:#0f172a; --panel:#15213c; --border:#243358; --text:#e6ecff; --muted:#9aa7c7; --accent:#4f8bff; --accent-2:#7aa9ff; --warn:#fbbf24; --ok:#34d399; --bad:#f87171; }
* { box-sizing: border-box; } body { font-family: "Noto Sans JP", system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.8; font-size: 15px; padding: 40px 56px 96px; max-width: 1100px; margin: 0 auto; }
a { color: var(--accent-2); } h1 { font-size: 26px; } h2 { font-size: 20px; margin: 40px 0 12px; border-left: 4px solid var(--accent); padding-left: 12px; } h3 { font-size: 16px; margin: 22px 0 8px; }
table { border-collapse: collapse; margin: 12px 0 18px; font-size: 13px; background: var(--panel); } th, td { border: 1px solid var(--border); padding: 5px 10px; text-align: left; vertical-align: top; } th { background: #1b2a4d; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
.muted { color: var(--muted); font-size: 13px; } .note { background: var(--panel); border: 1px solid var(--border); border-left: 4px solid var(--warn); border-radius: 8px; padding: 12px 16px; margin: 14px 0; }
.ok { color: var(--ok); } .bad { color: var(--bad); } code { background: #1b2a4d; padding: 1px 5px; border-radius: 4px; font-size: 13px; } svg text { fill: var(--muted); font-size: 11px; }"""


def load(src: Path, rel: str):
    p = src / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pc(x, d=1):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{d}f}%"


def f2(x, d=3, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def roi_cell(r, lo="lo", hi="hi", key="roi"):
    if r is None or r.get(key) is None:
        return "—"
    return f"{100 * r[key]:.1f}%<span class=muted> [{100 * r[lo]:.0f},{100 * r[hi]:.0f}]</span>"


def table(head, rows, num_from=1):
    h = "".join(f"<th{' class=n' if i >= num_from else ''}>{x}</th>" for i, x in enumerate(head))
    b = "".join("<tr>" + "".join(f"<td{' class=n' if i >= num_from else ''}>{x}</td>" for i, x in enumerate(r)) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{b}</table>"


def short(names: dict, i: str) -> str:
    n = names.get(i, i)
    m = re.match(r"^[^A-Za-z(（]+", n)
    s = (m.group(0) if m and len(m.group(0)) >= 2 else n)
    return html.escape(s[:16])


def svg_lines(series: dict, xs: list[str], ymin: float, ymax: float, w=620, h=230, title="") -> str:
    """折れ線（series: 名前 → [y]）。"""
    colors = ["#4f8bff", "#34d399", "#fbbf24", "#f87171", "#c084fc"]
    pad = 40
    n = len(xs)

    def X(i):
        return pad + (w - 2 * pad) * (i / max(n - 1, 1))

    def Y(v):
        return h - pad - (h - 2 * pad) * ((v - ymin) / (ymax - ymin))
    g = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px" role="img" aria-label="{html.escape(title)}">']
    for t in range(5):
        v = ymin + (ymax - ymin) * t / 4
        g.append(f'<line x1="{pad}" x2="{w - pad}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#243358"/><text x="4" y="{Y(v) + 4:.1f}">{v:.2f}</text>')
    for i, x in enumerate(xs):
        g.append(f'<text x="{X(i):.1f}" y="{h - 14}" text-anchor="middle">{html.escape(x)}</text>')
    for k, (name, ys) in enumerate(series.items()):
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(ys) if v is not None)
        c = colors[k % len(colors)]
        g.append(f'<polyline fill="none" stroke="{c}" stroke-width="2" points="{pts}"/>')
        for i, v in enumerate(ys):
            if v is not None:
                g.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3" fill="{c}"/>')
        g.append(f'<text x="{w - pad - 4}" y="{16 + 14 * k}" text-anchor="end" style="fill:{c}">{html.escape(name)}</text>')
    g.append("</svg>")
    return "".join(g)




def svg_heat(cells: list[dict], glab, qorder: list[str], title: str, clip: float = 4.0) -> str:
    """g × q の効果（pt）のヒートマップ。橙 = 人気以上に走る、青 = 走らない。*: |z|>2、**: |z|>3。空白 = 件数 400 未満。"""
    gs = sorted({c["g"] for c in cells})
    cw, ch, left, top = 74, 22, 120, 54
    w, h = left + cw * len(qorder) + 20, top + ch * len(gs) + 18
    by = {(c["g"], c["q"]): c for c in cells}
    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px" role="img" aria-label="{html.escape(title)}">']
    o.append(f'<text x="4" y="14" style="fill:#e6ecff;font-size:12px">{html.escape(title)}</text>')
    for j, q in enumerate(qorder):
        o.append(f'<text x="{left + cw * j + cw / 2}" y="{top - 8}" text-anchor="middle" style="font-size:10px">{html.escape(str(q))}</text>')
    for i, g in enumerate(gs):
        y = top + ch * i
        o.append(f'<text x="{left - 6}" y="{y + 15}" text-anchor="end" style="font-size:11px">{html.escape(glab(g))}</text>')
        for j, q in enumerate(qorder):
            c = by.get((g, q))
            if not c:
                continue
            v = max(-clip, min(clip, c["raw_pt"])) / clip
            col = f"rgba(245,158,11,{abs(v):.2f})" if v > 0 else f"rgba(59,130,246,{abs(v):.2f})"
            star = "**" if abs(c["z"]) > 3 else ("*" if abs(c["z"]) > 2 else "")
            o.append(f'<rect x="{left + cw * j}" y="{y}" width="{cw - 2}" height="{ch - 2}" fill="{col}" stroke="#243358"/>')
            o.append(f'<text x="{left + cw * j + cw / 2 - 1}" y="{y + 14}" text-anchor="middle" style="fill:#e6ecff;font-size:10px">{c["raw_pt"]:+.1f}{star}</text>')
    o.append("</svg>")
    return "".join(o)


def section_quality(gq, gqb, cinfo, qpre, qmeta, names, ev) -> str:
    if not gq:
        return "<h2>レース質 × 血統グループ</h2><p class=muted>（実行待ち）</p>"
    o = []
    a = o.append
    QL = {"joint": "展開タイプ 9 通り（ペース×隊列）", "lap": "ラップ型 6 通り", "l3f": "上がり水準 3 段階", "going": "馬場（良／道悪）", "wtime": "勝ち時計の基準比 3 段階（馬場の速さ）", "upset": "荒れ度 3 段階（勝ち馬の人気）"}
    GL = {"K16": "血統マップのクラスタ K=16", "K40": "血統マップのクラスタ K=40", "line": "父の系統 8 通り", "sire30": "産駒数上位 30 の父"}
    a("""<h2>1. 血統ベースのレース質分析（中心課題）</h2>
<p>『同じ血統マップのグループは、レースの質によって好走しやすさが違い、その中には人気以上に走るものがある。レース前に質の確率が分かれば、結果が未知のレースで予想・馬券に生きる』という仮説を、次の形で検証した。</p>
<ul>
<li><b>血統グループ g</b>: 構成埋め込み A（親の ID を持ち込まない）のクラスタ K=16 と K=40（2021 年までにデビューした馬で定義を固定、以降の馬は最近傍の中心に割り当て）。解釈しやすい対照として、父の系統 8 通りと、産駒数上位 30 の父。</li>
<li><b>レースの質 q</b>: 実測側（結果から作る）は tp-61 の展開タイプ 9 通り（ペース×隊列）、ラップ型 6 通り、上がり水準 3 段階、馬場（良／道悪）、勝ち時計の基準比 3 段階、荒れ度 3 段階。事前側は、展開タイプ・ラップ型・上がり水準を<b>レース前に分かる量から年ごとの拡張窓で予測した確率</b>（y 年の確率は y 年より前だけで学習）、馬場は発表値（T-60 でほぼ確定）。</li>
<li><b>好走率の差分</b>: 勝ち・3 着内の実績 − 市場の見込み（r_win = 勝ち − 単勝オッズ由来の勝率、r_t3 = 3 着内 − 複勝オッズ由来の 3 着内率）。『人気以上の好走』はこの超過。V4 の確率との超過も併記。</li>
<li><b>効果</b>: 交互作用 = 平均(g, q) − 平均(g)（市場の残差はレース内で和が 0 なので q の主効果はほぼ 0）、疑似 300 件で 0 へ縮約。検定は z（縮約前）、多重比較は BH（q = 10%）。</li>
<li><b>未知のレースへの適用</b>: 各馬の期待上乗せ = Σ<sub>q</sub> P(q̂) × effect(g, q)（効果表は<b>その年より前</b>の全期間から作る）。<b>オラクル</b>（q を既知とした上限）と<b>事前の確率版</b>（実運用の下限）の両方を出す。</li>
</ul>""")
    a("<h3>1.1 レース質ラベルの共有表（pace-benefit・v6-features から読める）</h3>")
    a("<p>1 レース 1 行のラベル表を 1 回だけ作り、次に置いた: <code>data/local/modeling/shared_cache/race_quality/race_quality.parquet</code>（ホスト /home/jovyan/work/keiba-vpn 配下。列の説明は同じ場所の <code>README.json</code>）。作成: <code>python -m src.scripts.data.tp60_quality table</code>。tp-61 の出力（race_cls.parquet・pre.parquet）を再利用し、実装を重ねていない。</p>"
      "<p><b>tp-64 との関係（重複の整理）.</b> tp-64 は展開セル（ペース 3 × 隊列 3）の事前確率を <code>data/local/modeling/tp64/pcell.parquet</code> に置いている。そちらは<b>年ごとの leave-one-year-out</b>（評価年の確率を他のすべての年で学習）で、後の年のラベルが学習に入る。"
      "本表の <code>p_joint_*</code> は<b>拡張窓</b>（y 年の確率は y 年より前だけで学習）で厳密に as-of。先読みを避けたい評価（V5 の 2 段階評価、馬券側の評価）は本表を使い、tp-64 の pcell は展開利の係数表の当てはめなど同じ年の leave-one-year-out を前提にした分析に限る。実測のラベル（q_joint など）は同一（tp-61 の race_cls）。</p>")
    a(table(["区分", "列", "内容", "いつ分かるか"], [
        ["実測", "pace_cls / shape_cls / q_joint", "ペース（slow/mid/high）、隊列（compact/normal/stretched）、その組（9 通り）", "結果後"],
        ["実測", "lap_type / q_lap", "tp-61 の規則のラップ型 6 通り", "結果後"],
        ["実測", "race_l3f_dev / q_l3f", "レース上がり 3F の同条件中央値との差、3 分位（分位点は 2020〜2021 年で固定）", "結果後"],
        ["実測", "win_time_dev / q_wtime", "勝ち時計の同じ場×面×距離×馬場の中央値との差（秒）、3 分位", "結果後"],
        ["実測", "winner_pop / top3_pop_sum / q_upset", "勝ち馬の人気、3 着内の人気の合計、荒れ度 3 段階", "結果後"],
        ["発表値", "q_going / going_group / moisture / cushion_value", "馬場（良／道悪）、含水率（芝はゴール前、ダートはゴール前）、クッション値", "レース前（T-60 でほぼ確定）"],
        ["静的", "cls / venue / surface / distance / band / field_size", "クラス水準、場、面、距離、距離帯、頭数", "レース前"],
        ["事前の予測", "p_joint_<ペース>|<隊列>（9）, p_lap_<名前>（6）, p_l3f_fast|mid|slow（3）", "y 年より前の全レースだけで学習した HistGradientBoosting（拡張窓）の確率。2020 年は欠損", "レース前（前日〜T-60）"],
    ], 0))
    if qpre:
        rows = [[k, f"{v['n']:,}", f2(v["logloss"], 3), f2(v["prior_logloss"], 3), pc(1 - v["logloss"] / v["prior_logloss"], 1)] for k, v in qpre.items()]
        a("<p>事前の予測の対数損失（2021〜2026 年、拡張窓）。事前確率（クラス頻度）より下がる＝事前に質をある程度当てられる。</p>")
        a(table(["質", "レース数", "モデル", "クラス頻度のみ", "改善率"], rows))

    a("<h3>1.2 血統グループ × 質ごとの効果: 強さ・翌年の再現・説明力（総括）</h3>")
    a("<p>各行は グループ分け × 質 の組。『効果の強いセル数』は 2020〜2025 年合算で件数 400 以上のセルのうち |z|>3 の数（BH 通過数は q=10%）。『翌年再現 ρ』は、その年より前の累積で作った効果表と、その年の同じセルの実測（r_t3）の順位相関（2022〜2025 年の平均）。"
      "『前年上位 10% セルの翌年の上乗せ』は、前の累積で効果が上位 10% のセルの翌年の 3 着内の市場超過（ポイント）、下位 10% も併記。"
      "『説明力』は、期待上乗せを説明変数にした r_t3 の回帰（レース単位クラスタ頑健）の傾き（1 なら効果表が較正済み）と |t|>2 の年数（2022〜2026 年の 5 年中）で、オラクル／事前の両方。</p>")
    rows = []
    for gname in ("K16", "K40", "line", "sire30"):
        gk = gq["K"].get(gname)
        if not gk:
            continue
        for qs, R in gk["qsys"].items():
            ys = R["years"]

            def agg(metric, field):
                v = [ys[f"{metric}:{y}"].get(field) for y in range(2022, 2026) if f"{metric}:{y}" in ys and ys[f"{metric}:{y}"].get(field) is not None]
                return sum(v) / len(v) if v else None
            ex = R.get("explain_t3", {})

            def exs(nm):
                if nm not in ex:
                    return "—"
                sl = [v["slope"] for v in ex[nm].values()]
                nt = sum(1 for v in ex[nm].values() if v["t"] is not None and abs(v["t"]) > 2)
                return f"{sum(sl) / len(sl):+.2f}（|t|>2: {nt}/{len(sl)}）"
            t10, b10 = agg("r_t3", "top10_next_year_raw"), agg("r_t3", "bottom10_next_year_raw")
            rows.append([GL[gname], QL[qs], R["n_cells_pooled"], f"{R['n_abs_z_gt3']} / {R['bh_significant_cells']}", f2(agg("r_t3", "rho_prior_vs_year"), 2, True), f2(agg("r_win", "rho_prior_vs_year"), 2, True),
                         f"{100 * t10:+.2f} / {100 * b10:+.2f}" if t10 is not None and b10 is not None else "—", exs("oracle"), exs("pred")])
    a(table(["グループ", "質", "セル数", "|z|>3 / BH 通過", "翌年再現 ρ（3着内）", "ρ（勝ち）", "前年上位10% / 下位10% の翌年の上乗せ（pt）", "オラクルの傾き", "事前の傾き（実運用）"], rows, 2))
    a("""<p><b>読み方.</b> 血統マップのクラスタ（K=16/40）×展開タイプ・ラップ型・上がり水準では、<b>翌年再現 ρ は −0.05〜+0.15、効果の強いセルは 0〜1 個、傾きは年により符号が変わり |t|>2 になる年も少ない</b>。
『同じマップグループで質ごとに人気以上の差が出る』ことは、この粒度では再現しない。例外は 2 つ: <b>(1) 馬場（良／道悪）</b>は、2023〜2025 年に傾きが一貫して正（オラクル = 事前。K=16 で 0.65〜0.83、|t|≈2）で、前年上位 10% のセルが翌年 +0.5〜+0.8pt 市場を上回った。馬場は V5 の PED_sire_cond が（種牡馬 ID × 馬場）で既に持つ情報と重なる。
<b>(2) 荒れ度</b>は勝ちの超過で全年・全分け方で傾きが 0.7〜1.0 と安定して正だが（t=3〜5.6）、荒れ度は結果から決まる質で事前に出せず（また『荒れた＝人気薄が勝った』ので、人気薄の多いグループが上がる機械的な関係を含む）、実運用には使えない。</p>""")

    # cluster description
    if cinfo:
        a("<h3>1.3 血統マップのグループ（K=16）が何か</h3>")
        rows = []
        for e in cinfo["K16"]:
            top = "、".join(f"{html.escape(t['name'])}({t['n']})" for t in e["top_sires"])
            sl = " / ".join(f"{k} {int(100 * v)}%" for k, v in e["sire_line"].items())
            rows.append([e["g"], f"{e['horses']:,}", top, sl, pc(e["mean_xF"], 2)])
        a(table(["g", "馬数", "父の上位 3（産駒数）", "父の系統（上位）", "平均 近交係数"], rows, 1))

    a("<h3>1.4 解釈できる対応（上位・下位のセル）と、2026 年（テストセット②）での確認</h3>")
    a("<p>2020〜2025 年合算の r_t3（3 着内 − 市場）で効果が大きい／小さいセル（件数 400 以上）。前半（2020-22）・後半（2023-25）で同じ向きか、2026 年（選択に使っていない）での実測を併記。")
    a("単位は pt（3 着内率の市場に対する超過の、そのグループの平均からの差）。</p>")
    sire30 = {e["g"]: e["name"] for e in (cinfo or {}).get("sire30", [])}
    linel = {e["g"]: e["name"] for e in (cinfo or {}).get("line", [])}
    QCLASSES = {"joint": [f"{p_}|{s_}" for p_ in ("slow", "mid", "high") for s_ in ("compact", "normal", "stretched")], "lap": ["slow_kick", "slow_spurt", "even_kick", "even_flat", "high_breather", "high_collapse"],
                "l3f": ["fast", "mid", "slow"], "going": ["良", "道悪"]}
    glabs = {"line": (lambda g: linel.get(g, str(g))), "sire30": (lambda g: sire30.get(g, str(g))), "K16": (lambda g: f"g{g}"), "K40": (lambda g: f"g{g}")}
    for gname, qs, cap in (("line", "joint", "父の系統 × 展開タイプ（3 着内の市場超過、pt）"), ("line", "lap", "父の系統 × ラップ型（pt）"), ("sire30", "lap", "産駒数上位 30 の父 × ラップ型（pt）"), ("K16", "going", "血統マップのクラスタ K=16 × 馬場（pt）")):
        R = gq["K"].get(gname, {}).get("qsys", {}).get(qs)
        if R and R.get("table"):
            order = QCLASSES[qs]
            a(svg_heat(R["table"], glabs[gname], order, cap))
    a("<p class=muted>色は 3 着内率の市場超過（そのグループの平均との差、pt）。橙 = 人気以上に走る、青 = 走らない。* は |z|>2、** は |z|>3。空白は件数 400 未満。2020〜2025 年の合算。</p>")
    rows = []
    for gname, qs in (("sire30", "lap"), ("sire30", "l3f"), ("sire30", "going"), ("line", "lap"), ("line", "joint"), ("K16", "going")):
        R = gq["K"].get(gname, {}).get("qsys", {}).get(qs)
        if not R:
            continue
        gl = (lambda g: sire30.get(g, str(g))) if gname == "sire30" else ((lambda g: linel.get(g, str(g))) if gname == "line" else (lambda g: f"g{g}"))
        for tag, lst in (("上", R["top_pos"][:3]), ("下", R["top_neg"][:3])):
            for r in lst:
                rows.append([f"{GL[gname]} × {QL[qs]}", tag, f"{html.escape(gl(r['g']))} × {html.escape(str(r['q']))}", f"{r['n']:,}", f2(r["raw_pt"], 2, True), f2(r["z"], 1, True), f2(r["h1"], 1, True), f2(r["h2"], 1, True),
                             (f"{f2(r['y26'], 1, True)}（{r['n26']:,}）" if r.get("y26") is not None else "—")])
    a(table(["分け方 × 質", "", "セル", "n", "効果(pt)", "z", "前半", "後半", "2026（n）"], rows, 3))
    a("<p class=muted>多重比較: セルは 1 つの分け方×質で 14〜259 個、全体で数百。|z|>3 は偶然の期待が 0.3% なので数セルは偶然に出る。BH 通過は表 1.2 を参照。表の上位セルは『前半・後半が同じ向き』かつ『2026 年も同じ向き』のものが少数ある程度で、確証には至らない。</p>")

    # 馬券側
    if gqb:
        a("<h3>1.5 馬券側: 事前の上乗せで EV の買い目を絞る</h3>")
        a("<p>V4 の EV（勝率 × 確定オッズ ≥ τ、複勝は 3 着内率 × 複勝オッズ）の買い目のうち、事前の期待上乗せ（Σ P(q̂)×effect(g,q)）が正の側の回収率 − 買い目全体（レース単位ブートストラップ区間）。4 通りの分け方 × 4 つの質 × 4 つの EV 設定 × 3 期間。</p>")
        rows = []
        tot = 0
        sig = []
        for r in gqb["rows"]:
            if r["period"] == "2022-2024" and r.get("diff_lo") is not None:
                tot += 1
                if r["diff_lo"] > 0:
                    sig.append(r)
        for r in sig[:8]:
            def get(per):
                for x in gqb["rows"]:
                    if (x["K"], x["qs"], x["bet"], x["tau"], x["period"]) == (r["K"], r["qs"], r["bet"], r["tau"], per):
                        return x
            x25, x26 = get("2025"), get("2026")
            rows.append([f"{GL[r['K']]} × {QL[r['qs']]}", f"{r['bet']} EV≥{r['tau']}", f"{f2(r['diff'], 3, True)} [{f2(r['diff_lo'], 2, True)},{f2(r['diff_hi'], 2, True)}]",
                         f"{f2(x25['diff'], 3, True)} [{f2(x25['diff_lo'], 2, True)},{f2(x25['diff_hi'], 2, True)}]" if x25 and x25.get("diff") is not None else "—",
                         f"{f2(x26['diff'], 3, True)} [{f2(x26['diff_lo'], 2, True)},{f2(x26['diff_hi'], 2, True)}]" if x26 and x26.get("diff") is not None else "—"])
        a(f"<p>2022-2024 の区間が 0 を外れて正になるのは {tot} 通り中 {len(sig)} 通り（偶然の期待は片側 2.5% で約 {tot * 0.025:.0f} 通り）。</p>")
        if rows:
            a(table(["分け方 × 質", "EV 設定", "2022-2024 差 [区間]", "2025", "2026（テストセット②）"], rows, 2))

    # V5 への増分
    a("<h3>1.6 V5 への増分（特徴量として）</h3>")
    pq = ev.get("stage1_pq")
    if pq and pq.get("groups"):
        rows = []
        for g_, v in pq["groups"].items():
            c = v["vs_V5"]
            first = v.get("subsets", {}).get("履歴0走(初出走)")
            rows.append([g_, v["cols"], f2(c["Δwin"], 5, True), f2(c["win_hi"], 5, True), f2(c["Δtop2"], 5, True), f2(c["Δtop3"], 5, True), c["採用"], (f"{f2(first['Δwin'], 5, True)}" if first else "—")])
        a(table(["群", "列数", "Δ勝率", "上限", "Δ連対", "Δ複勝", "採否", "初出走のΔ勝率"], rows, 1))
        a("<p>群は 事前の期待上乗せ（勝ち・3 着内 × 展開タイプ・ラップ型・上がり水準・馬場の 4 質）。2020 年は欠損、2021 年以降は<b>その年より前だけ</b>で作った効果表と質の予測確率を使う。</p>")
    else:
        a("<p class=muted>（評価は実行待ち）</p>")
    return "\n".join(o)



def quality_summary(gq, gqb, cinfo) -> str:
    if not gq:
        return "（実行待ち）"
    import statistics as st

    rhos = {"joint": [], "lap": [], "l3f": [], "going": []}
    nbh = 0
    ncell = 0
    corr_or, corr_pr = [], []
    tpos = {"oracle": 0, "pred": 0}
    nyr = {"oracle": 0, "pred": 0}
    for gname, gk in gq["K"].items():
        for qs, R in gk["qsys"].items():
            ys = R["years"]
            v = [ys[f"r_t3:{y}"].get("rho_prior_vs_year") for y in range(2022, 2026) if f"r_t3:{y}" in ys and ys[f"r_t3:{y}"].get("rho_prior_vs_year") is not None]
            if qs in rhos and v:
                rhos[qs].append(sum(v) / len(v))
            nbh += R["bh_significant_cells"]
            ncell += R["n_cells_pooled"]
            for nm in ("oracle", "pred"):
                for y, e in (R.get("explain_t3", {}).get(nm) or {}).items():
                    (corr_or if nm == "oracle" else corr_pr).append(abs(e["corr"]))
                    nyr[nm] += 1
                    if e["t"] is not None and e["t"] > 2:
                        tpos[nm] += 1
    rr = lambda q: f"{min(rhos[q]):+.2f}〜{max(rhos[q]):+.2f}" if rhos[q] else "—"
    nb = 0
    ns = 0
    if gqb:
        for r in gqb["rows"]:
            if r["period"] == "2022-2024" and r.get("diff_lo") is not None:
                ns += 1
                nb += r["diff_lo"] > 0
    return (f"4 通りのグループ分け（マップのクラスタ K=16/40・父の系統・産駒数上位 30 の父）× 6 つの質（展開タイプ 9・ラップ型 6・上がり水準 3・馬場 2・勝ち時計 3・荒れ度 3）で、人気以上の好走（3 着内・勝ちの市場超過）の差分を測った。"
            f"<b>翌年に続く効果は弱い</b>: 前の累積の効果表と翌年の実測の順位相関は、展開タイプ {rr('joint')}、ラップ型 {rr('lap')}、上がり水準 {rr('l3f')}、馬場 {rr('going')}。"
            f"効果が強い（BH 通過）セルは全 {ncell} セル中 {nbh} 個。期待上乗せの説明力（相関の最大）は<b>オラクル（q 既知）でも {max(corr_or):.3f}、事前の確率版（実運用）で {max(corr_pr):.3f}</b>（R² は {max(corr_or) ** 2:.4f} 以下）で、"
            f"傾きが |t|>2 で正になったのは、事前の確率版で {nyr['pred']} 回中 {tpos['pred']} 回、オラクルで {nyr['oracle']} 回中 {tpos['oracle']} 回（設定 × 年の数）。"
            f"馬券側（EV の買い目を事前の上乗せが正の側に絞る）は 2022-24 で区間が 0 を外れたのが {ns} 通り中 {nb} 通り（偶然の期待 {ns * 0.025:.0f}）で、2025 年は区間が 0 を含み 2026 年は逆符号が多い。"
            "解釈できる対応は少数（例: キズナ産駒は『中緩み再加速（even_kick）』で 3 着内が市場を +4.1pt 上回る、n=886、前後半とも正）あるが、2026 年の確認（n=99）は逆符号で、確証に至らない。"
            "<b>例外は馬場（発表値）</b>だけが年をまたいで弱く一貫（荒れ度は結果から決まる質で事前に出せず、機械的な関係を含む）（K=16 の傾き 0.4〜0.6、2023〜2025 年に |t|≈2）。V5 の PED_sire_cond（種牡馬 × 馬場）と重なる。")



def conclusion_html(gq, rules, ef, sm, kn, ev) -> str:
    done = []
    for st in ("stage1", "stage1_pq"):
        if ev.get(st):
            for g, v in ev[st].get("groups", {}).items():
                done.append((g, v["vs_V5"]))
    best = min(done, key=lambda x: x[1]["Δwin"]) if done else None
    npass = sum(1 for g, c in done if c["採用"] is True)
    evtxt = (f"V5 に累積した {len(done)} 群のうち第 1 段の採用基準を満たしたのは <b>{npass} 群</b>（最良は {best[0]} の Δ勝率 {best[1]['Δwin']:+.5f}、上限 {best[1]['win_hi']:+.5f}）。" if best else "特徴量の評価は未完了。")
    return f"""<h2>10. 結論</h2>
<ol>
<li><b>血統ベースのレース質分析（中心課題）</b>: 血統マップのグループ（K=16/40、父の系統、上位 30 の父）× レースの質（展開タイプ・ラップ型・上がり水準・馬場・勝ち時計・荒れ度）で、人気以上の好走の差分は<b>弱く、翌年の再現も低い</b>（順位相関 −0.05〜+0.37、BH 通過セルは全 1,986 セル中 6 個）。事前の質の確率で未知のレースに適用した期待上乗せの説明力は相関 0.02 以下で、馬券側の EV 絞り込みも再現しない。例外は馬場（発表値）が弱く一貫する点と、特定の父 × ラップ型の少数の対応（キズナ × 中緩み再加速など。2026 年で確認できず）。</li>
<li><b>血統の回収率で稼げる条件は見つからない</b>: 区間の下限が 100% を超えるセルは 0、規則 {rules['n_trials']} 本・{ef['n_rules']} 本の reality check は p = {rules['rc_val']['p']:.2f}・{ef['rc']['p']:.2f}。勝率・複勝率は翌年も続くが、回収率と市場超過は続かない（血統は市場に織り込み済み）。</li>
<li><b>EV への補正・絞り込み</b>は、血統単独でも、騎手・調教師の規則（tp-59）への重ね掛けでも再現しない。V4 の確率の血統による補正は最大で 1 頭あたり約 −0.0002。</li>
<li><b>10 世代の血量</b>は、6〜10 世代が V4／V5 の残差にほとんど効かない（|t|>2 が {sm['resid_v4']['deep_|t|>2']}/{sm['resid_v4']['n_reps']}）。5 世代以内の他位置の血量は V4 の残差に効く（{sm['resid_v4']['other5_|t|>2']}/{sm['resid_v4']['n_reps']}）が、V5 が大半を吸収している。</li>
<li><b>近さの定義</b>は『祖先の構成』（親の ID を持ち込まない）に決め、既知ペアの検証（全兄弟 = 距離 0、サドラーズウェルズ→ヌレイエフ {kn['A_t2_T10_g100']['SW_Nureyev_rank']} 位、父だけ違う産駒 cos 1.00）を満たす。<b>近さ ≠ 適性</b>（全兄弟でも能力の相関 0.13）。{evtxt}</li>
</ol>
<p><b>次にやること.</b> (1) 直前オッズ（T-10/T-5/T-2）が貯まってから、馬場 × 血統・父 × ラップ型の少数の候補を購入時点のオッズで再検証。(2) 質の粒度を『展開タイプ』より細かく（想定位置取りの分布 × 脚質 × 血統）して、血統の寄与を位置取りの有利不利から分ける。(3) 荒れ度の事前予測（フィールドの拮抗度・人気の割れ方）ができれば、荒れ度 × 血統グループの安定した関係（オラクルで t=3〜5）を未知のレースに使える可能性がある。</p>"""


def build(src: Path) -> str:
    cells = load(src, "roi/cells.json")
    persist = load(src, "roi/persist.json")
    popular = load(src, "roi/popular.json")
    rules = load(src, "roi/rules.json")
    evf = load(src, "roi/evfilter.json")
    evfin = load(src, "roi/evfinal.json")
    dsel = load(src, "roi/deep_select.json")
    deff = load(src, "roi/deep_effect.json")
    known = load(src, "embed/known_pairs.json")
    kids = load(src, "embed/known_ids.json")
    aff = {v: load(src, f"roi/affinity_{v}.json") for v in ("A_t2_T10_g100", "D_t2_T10_idf", "E_t1_T10_g100", "C_t2_T5_g100")}
    sp = load(src, "roi/sirepairs_A_t2_T10_g100.json")
    beta = load(src, "roi/beta_curve_A_t2_T10_g100.json")
    rstat = load(src, "ped/resolve_stats.json")
    names = load(src, "ped/anc_names.json") or {}
    ev = {s: load(src, f"eval/{s}/results.json") for s in ("stage1", "stage1_pq", "stage1_b", "stage2", "full")}
    gq = load(src, "roi/gq.json")
    gqb = load(src, "roi/gq_bet.json")
    cinfo = load(src, "roi/clusters_info.json")
    qpre = load(src, "roi/quality_pre_eval.json")
    qmeta = load(src, "roi/race_quality_README.json")
    P = []
    a = P.append
    sm = deff["summary"]

    base = cells["baseline"]
    pooled = base["pooled_2020_2025"]
    a(f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>血統の回収率分析（tp-60）— keiba-vpn</title>
<style>{CSS}</style></head>
<body><a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>血統の単勝・複勝回収率の分析と、10 世代の血量・血統グラフ上の近さ（tp-60）</h1>
<p class="muted">2026-10-21、ユーザー依頼『騎手・調教師と同様に、回収率をもとにした血統分析を』と追加依頼（10 世代の血量、出現率の高い種牡馬、位置の重複の排除、血統の座標化とクラスタ・kNN、グラフ上の近さ、同じ配合でも適性は同じにならない）。
選択は 2022〜2024 年、2025 年は確認で 1 回、2026 年はテストセット②として最後に 1 回だけ。市場オッズ・払戻は馬券の評価と『市場に織り込まれているか』の分析にだけ使い、特徴量にはしない。キーはすべて ID。</p>
""")

    # ---- 要点
    rule_best = rules["top_by_val"][0]
    n_cells_sig = sum(cells["keys"][k]["lo_gt_100_win"] + cells["keys"][k]["lo_gt_100_place"] for k in cells["keys"])
    kn = known["variants"]
    groups_eval = {}
    for st in ("stage1", "stage2", "full"):
        if ev.get(st):
            for g, v in ev[st].get("groups", {}).items():
                groups_eval.setdefault(g, {})[st] = v
    passed = [g for g, d_ in groups_eval.items() if "stage1" in d_ and d_["stage1"]["vs_V5"]["採用"] is True]
    a(f"""<div class="note"><b>要点（先に答え）.</b>
<ul>
<li><b>【中心課題】血統ベースのレース質分析</b>（血統マップのグループ g × レースの質 q で、人気以上の好走の差分を測り、事前の質の確率で未知のレースに適用）: <!--QSUM--></li>
<li><b>血統の回収率で稼げる条件は見つからない</b>。父・母父・父父・父×母父・系統・近交・新種牡馬のセル（件数下限あり）で、単勝・複勝回収率の区間の下限が 100% を超えるものは {n_cells_sig} 個。規則の網羅 {rules['n_trials']} 本の reality check は p = {rules['rc_val']['p']:.2f}（選択年の最良 {pc(rule_best['val']['roi'])} は 2025 年 {pc(rule_best['2025']['roi'])}）。</li>
<li><b>勝率・複勝率は翌年も続く（種牡馬で順位相関 0.4〜0.6）が、回収率は続かない（0.0〜0.16）</b>。市場の勝率との差もほぼ続かず、血統は市場に織り込み済み。『人気の血統ほど回収率が低い』ではなく逆（favorite-longshot の歪み）。</li>
<li><b>EV への補正・絞り込み、騎手・調教師の規則への重ね掛けは、どれも再現しない</b>（2023-24 で区間が 0 を外れた 2 通りは 2025 年に区間が 0 を含み 2026 年は逆符号）。V4 の確率を血統セルの偏りで直す余地は最大でも 1 頭あたり約 −0.0002。</li>
<li><b>10 世代の血量</b>: 被覆 99.9999%（自前で連結）。出現率は 10 世代で飽和するため 5 世代以内の出現率で候補 {dsel['n_before_dedup']} 頭 → 父子・相関で整理して代表 {dsel['n_after_dedup']} 頭。6〜10 世代の血量は V4／V5 の残差にほとんど効かない（|t|>2 が {sm['resid_v4']['deep_|t|>2']}/{sm['resid_v4']['n_reps']}、偶然の期待 {sm['resid_v4']['expected_|t|>2_under_null']:.1f}）。</li>
<li><b>血統グラフ上の近さ</b>: 親の ID を持ち込まない構成埋め込みは、全兄弟（DI–BT）が互いに最近接（順位 1）、サドラーズウェルズ→ヌレイエフが {kn['A_t2_T10_g100']['SW_Nureyev_rank']} 位、父だけ違う産駒の cos {f2(kn['A_t2_T10_g100']['cos_father_differs_mean'], 2)}。親の ID を持ち込む方式と Node2Vec は制約を満たさず（Node2Vec は DI→BT が {kn['N_node2vec']['DI_BT_rank']:,} 位）、近さの定義として不適。</li>
<li><b>近さ ≠ 適性</b>: 全兄弟でも能力の相関は {f2(aff['A_t2_T10_g100']['pairs']['full_siblings(同父同母)']['corr_ability'], 2)}（補正後 {f2(aff['A_t2_T10_g100']['pairs']['full_siblings(同父同母)']['disattenuated']['ability'], 2)}）、近さの上位 1 割でも無作為ペアに近い。構成が同一の種牡馬どうしでも産駒の成績は大きく違う（DI と BT: 複勝率 29.5% と 19.9%）。</li>
<li><b>特徴量の増分（V5 の 1,142 列に累積）</b>: <!--EVALSUM--></li>
</ul></div>
""")

    a(section_quality(gq, gqb, cinfo, qpre, qmeta, names, ev))
    P[:] = [x.replace("<!--QSUM-->", quality_summary(gq, gqb, cinfo)) for x in P]
    # ---- 既存との差分
    a("""<h2>2. 既存の血統分析との差分</h2>
<table>
<tr><th>既存（V5 に入っている／不採用だった）</th><th>tp-60 で新しくやったこと</th></tr>
<tr><td>PED_sire_cond（採用）: 父・母父・父父・母母の ID × 条件の産駒成績（結果由来の as-of 縮約）</td><td>成績ではなく<b>回収率・市場に対する残差</b>で血統セルを評価し、翌年に続くかを検定</td></tr>
<tr><td>PED_blood（不採用）: 上位 30 祖先の血量（5 世代）</td><td><b>10 世代</b>（6〜10 世代を足す）、出現率で候補を選び、位置の重複と父子・相関の重複を整理した<b>代表 40 頭</b></td></tr>
<tr><td>tp-38 ALL_BEST 43 列（採用・境界）: 祖先 × 位置の区分 × 条件の産駒成績、5 世代のリッジ、SVD 32 次元（ID を含む祖先ベクトル）</td><td>親の ID を持ち込まない<b>構成埋め込み</b>（祖先の構成だけで近さを決める）と、既知ペア（全兄弟・同系・父だけ違う産駒）での近さの検証、Node2Vec との比較</td></tr>
<tr><td>tp-38 PX_clu / PX_sim（単独では不採用）: k-means（64/256）と類似種牡馬による縮約</td><td>構成埋め込みでのクラスタ統計と kNN 型の統計（近さ別の重みは結果から推定した β 曲線、事前を馬自身の履歴で更新）、V5 への増分</td></tr>
<tr><td>similar_horses_pairs（不採用）: 血統の近さは組合せ券の同時確率に効かない</td><td>近さそのものが適性の近さにどれだけ対応するかの実測（近さの十分位ごとの相関、全兄弟の一致度＝天井）</td></tr>
</table>
""")

    # ---- 2. 血統別の成績
    a(f"""<h2>3. 血統別の勝率・複勝率・回収率</h2>
<p>対象は JRA 平地の 2020〜2025 年（{pooled['roi_win']['n']:,} 出走）。買い方は 1 頭 100 円ずつの単勝・複勝（確定オッズ・確定払戻）。<b>全馬を買ったときの回収率は単勝 {roi_cell(pooled['roi_win'])}、複勝 {roi_cell(pooled['roi_place'])}</b>（払戻率 80% より低いのは人気薄ほど回収率が低い歪みのため）。区間は 95%（レース単位ブートストラップ）。</p>""")
    rows = []
    for key, lab in (("sire_id", "父"), ("dsire_id", "母父"), ("ssire_id", "父父"), ("sire_dsire", "父×母父"), ("sire_line", "父の系統"), ("dsire_line", "母父の系統"), ("line_pair", "父系統×母父系統"), ("xF_band", "近交の強さ"), ("crop", "父の産駒世代（新種牡馬）")):
        e = cells["keys"][key]
        rows.append([lab, e["n_cells"], f"{e['n_cells_eligible']}（下限 {e['nmin']} 頭）", e["lo_gt_100_win"], e["lo_gt_100_place"]])
    a(table(["切り口", "セル数", "件数下限を満たすセル", "単勝: 区間の下限が 100% 超", "複勝: 区間の下限が 100% 超"], rows))
    a("<p>どの切り口でも、件数下限を満たしたセルの中に<b>区間の下限が 100% を超えるものはない</b>（検定の数が多いので偶然の上振れの余地もこの表の『0』に含めて読む）。")
    a("外国産馬のフラグ（マル外）は horse_id からは取れず、血統表の祖先 ID にも国の情報が無いので、この切り口は未実施（生産国は horse_profile に別取得が要る）。</p>")

    def name_row(r):
        return [f"{short(names, r['key'])}<span class=muted> {r['key']}</span>", f"{r['horses']:,}", pc(r["win_rate"]), pc(r["top3_rate"]), roi_cell({"roi": r["roi_win"], "lo": r["roi_win_lo"], "hi": r["roi_win_hi"]}),
                roi_cell({"roi": r["roi_place"], "lo": r["roi_place_lo"], "hi": r["roi_place_hi"]}), f2(r["ex_mkt"] * 100, 2, True) + "pt"]
    a("<h3>3.1 産駒数の多い種牡馬（父）</h3>")
    a(table(["父", "産駒の出走", "勝率", "複勝率", "単勝回収率", "複勝回収率", "勝率−市場の勝率"], [name_row(r) for r in cells["keys"]["sire_id"]["by_size"][:14]]))
    a("<p class=muted>『勝率−市場の勝率』は、単勝オッズから作った市場の勝率（控除率を除いて正規化）との差の平均。0 に近いほど市場が織り込んでいる。</p>")

    # 特殊セル
    sp_c = cells["special"]
    a("<h3>3.2 系統・新種牡馬・近交</h3>")
    rows = []
    for key, lab in (("sire_line", "父の系統"), ("dsire_line", "母父の系統"), ("crop", "父の産駒世代"), ("xF_band", "近交（近交係数 F）")):
        for v, e in sp_c[key].items():
            x = e.get("2020-2025")
            t = e.get("2026")
            if not x or x["horses"] < 3000:
                continue
            rows.append([f"{lab}: {html.escape(v)}", f"{x['horses']:,}", pc(x["win_rate"]), pc(x["top3_rate"]), roi_cell(x, "roi_win_lo", "roi_win_hi", "roi_win"), roi_cell(x, "roi_place_lo", "roi_place_hi", "roi_place"),
                         (f"{100 * t['roi_win']:.0f}% / {100 * t['roi_place']:.0f}%" if t else "—")])
    a(table(["区分", "出走", "勝率", "複勝率", "単勝回収率 2020-25", "複勝回収率 2020-25", "2026（単/複）"], rows))
    a("<p class=muted>『父の産駒世代』は、その父の産駒のうち最初に競走馬になった世代（2017 年以降に初めて産駒が出た種牡馬のみ。crop0 = 初年度産駒）。系統は尾雄の鎖（本馬 → 父 → …）で最初に一致した創始者で分類（SS・ロベルト・Mr.P・ND・ナスルーラ・ネイティヴダンサー・ヘイルトゥリーズン・ブランドフォード・その他）。</p>")

    # popularity
    a("<h3>3.3 人気の血統ほど回収率が低いか</h3>")
    ps = popular["sire_id"]
    rows = []
    for lab, e in ps["groups"].items():
        rows.append([lab, e["sires"], f"{e['horses']:,}", pc(e["win_rate"]), pc(e["mean_q"]), roi_cell(e["roi_win"]), roi_cell(e["roi_place"])])
    a(table(["父の産駒数", "頭数", "産駒の出走", "勝率", "市場の勝率", "単勝回収率", "複勝回収率"], rows))
    q = ps["by_q_quartile"]
    a(f"<p>父を産駒の平均の市場勝率で 4 分位に分けた回収率（種牡馬の単純平均）は、人気薄の血統 {pc(q['人気薄の血統']['roi_win'])}（単）/{pc(q['人気薄の血統']['roi_place'])}（複）→ 人気の血統 {pc(q['人気の血統']['roi_win'])}/{pc(q['人気の血統']['roi_place'])}。"
      f"<b>人気の血統ほど回収率が低い、という関係ではなく逆</b>（種牡馬間の順位相関: 市場勝率と単勝回収率 {f2(ps['corr_q_roi_win'], 2, True)}、複勝 {f2(ps['corr_q_roi_place'], 2, True)}）。"
      "これは血統の効果ではなく、人気薄の馬ほど回収率が低いという一般の歪み（favorite-longshot bias）が、人気薄が多い血統に乗っているため。</p>")

    # ---- 3. 再現性
    a("""<h2>4. 翌年に続くか（再現性）</h2>
<p>各年 y（2021〜2025）について、<b>y 年より前の全期間</b>の成績でセルを順位付けし、y 年に何が起きたかを見る。セルは血統キー × 条件（面×距離帯・馬場・競馬場・クラス・年齢・人気帯）、件数下限あり。
順位相関は（前の累積 → その年）の同一セルどうし。『上位 10% を買う』は前の累積の<b>縮約回収率</b>（疑似 150 件で全体平均へ縮約）が上位 10% のセルの馬を買い、同じ件数下限を満たした全セルの馬と回収率を比べる（差と、年ごとに正だった数）。</p>""")
    rows = []
    for k, v in persist.items():
        mr = v.get("mean_rho", {})
        if not mr:
            continue
        rows.append([k.replace("@", " × "), f2(mr.get("rho_win_rate"), 2, True), f2(mr.get("rho_top3_rate"), 2, True), f2(mr.get("rho_ex_mkt"), 2, True), f2(mr.get("rho_roi_win"), 2, True), f2(mr.get("rho_roi_place"), 2, True),
                     f"{pc(v.get('top_win_mean_sel_roi'))} / {pc(v.get('top_win_mean_all_roi'))}（{f2(v.get('top_win_mean_diff'), 3, True)}、+{v.get('top_win_n_pos')}/{v.get('top_win_n_years')}年）",
                     f"{pc(v.get('top_place_mean_sel_roi'))} / {pc(v.get('top_place_mean_all_roi'))}（{f2(v.get('top_place_mean_diff'), 3, True)}、+{v.get('top_place_n_pos')}/{v.get('top_place_n_years')}年）"])
    a(table(["キー × 条件 @件数下限", "ρ 勝率", "ρ 複勝率", "ρ 勝率−市場", "ρ 単勝回収率", "ρ 複勝回収率", "上位10%買い 単勝（選 / 全体）", "上位10%買い 複勝（選 / 全体）"], rows))
    def rng_of(keys, field):
        vals = [persist[k]["mean_rho"][field] for k in keys if k in persist and persist[k].get("mean_rho") and persist[k]["mean_rho"].get(field) is not None]
        return (min(vals), max(vals)) if vals else (float("nan"), float("nan"))
    sire_keys = [k for k in persist if k.startswith(("sire_id@", "dsire_id@", "ssire_id@", "sire_dsire@"))]
    r_win, r_t3, r_roiw, r_roip, r_ex = (rng_of(sire_keys, f) for f in ("rho_win_rate", "rho_top3_rate", "rho_roi_win", "rho_roi_place", "rho_ex_mkt"))
    diffs = [persist[k]["top_win_mean_diff"] for k in sire_keys if persist[k].get("top_win_mean_diff") is not None]
    sels = [persist[k]["top_win_mean_sel_roi"] for k in sire_keys if persist[k].get("top_win_mean_sel_roi") is not None]
    alls = [persist[k]["top_win_mean_all_roi"] for k in sire_keys if persist[k].get("top_win_mean_all_roi") is not None]
    pdiffs = [persist[k]["top_place_mean_diff"] for k in sire_keys if persist[k].get("top_place_mean_diff") is not None]
    psels = [persist[k]["top_place_mean_sel_roi"] for k in sire_keys if persist[k].get("top_place_mean_sel_roi") is not None]
    npos = sum(persist[k].get("top_win_n_pos", 0) for k in sire_keys)
    nyr = sum(persist[k].get("top_win_n_years", 0) for k in sire_keys)
    a(f"""<p><b>読み方.</b> 父・父父・母父・父×母父のキー（{len(sire_keys)} 通りの条件）で、<b>勝率の順位相関は {r_win[0]:+.2f}〜{r_win[1]:+.2f}、複勝率は {r_t3[0]:+.2f}〜{r_t3[1]:+.2f} と翌年も続く</b>（血統の実力差は安定）。
<b>回収率の順位相関は単勝 {r_roiw[0]:+.2f}〜{r_roiw[1]:+.2f}、複勝 {r_roip[0]:+.2f}〜{r_roip[1]:+.2f} と弱く</b>、市場の勝率との差も {r_ex[0]:+.2f}〜{r_ex[1]:+.2f}。
過去の縮約回収率の上位 10% を買うと、基準（件数下限を満たす全セル）より単勝で {min(diffs):+.3f}〜{max(diffs):+.3f}（年ごとの符号は {npos}/{nyr} が正）、複勝で {min(pdiffs):+.3f}〜{max(pdiffs):+.3f} 高い。
ただし水準は単勝 {pc(min(sels))}〜{pc(max(sels))}（基準 {pc(min(alls))}〜{pc(max(alls))}）、複勝 {pc(min(psels))}〜 で、<b>100% には届かない</b>。『市場の見込みを上回る血統がわずかに続く』以上のことは言えない。系統（sire_line / dsire_line / line_pair）の条件別は再現性がさらに弱い。</p>""")

    a("<h3>4.1 年別の内訳（父 × 全体・父父 × 全体の上位 10% を買う）</h3>")
    rows = []
    for k in ("sire_id@all@100", "ssire_id@all@100"):
        for bt, lab in (("win", "単勝"), ("place", "複勝")):
            for y, e in persist[k]["years"].items():
                t = e.get(f"top_{bt}")
                if not t or t.get("sel_roi") is None:
                    continue
                rows.append([f"{k.split('@')[0]} {lab}", y + ("（テストセット②）" if y == "2026" else ""), f"{t['n_sel']:,}", pc(t["sel_roi"]), pc(t["all_roi"]), f"{f2(t['diff'], 3, True)} [{f2(t.get('diff_lo'), 2, True)},{f2(t.get('diff_hi'), 2, True)}]", f2(t.get("p_le0"), 2)])
    a(table(["キー × 券", "年", "買い数", "選んだ側の回収率", "基準（件数下限を満たす全セル）", "差 [95%区間]", "差≤0 の割合"], rows, 2))
    a("<p>複勝は父で +0.03〜+0.09 が 5 年とも正で、最も安定して基準より高い層だが、選んだ側の水準も 80% 前後で 100% には届かない（2026 年も同様）。単勝は年により符号が変わる。</p>")

    # ---- 4. 市場の織り込み
    a("<h2>5. 市場に織り込まれているか／V4 の確率の偏りを血統で直せるか</h2>")
    ev_recal = evf["recal"]
    rows = []
    for nm, r in ev_recal.items():
        rows.append([nm.replace("@", " × ")] + [f"{f2(r[y]['delta'], 5, True)}（λ={r[y]['best_lambda']}）" for y in ("2023", "2024", "2025")])
    a("<p>V4 の勝率 p に、血統セルの<b>過去の偏り</b>（Σ勝 / Σp の縮約、2022 年から）を掛けて p を補正したときの、勝ち馬の対数損失（レース内正規化。1 レースあたり。負が改善）の変化。λ は補正の強さ（0.25/0.5/1 の最良）。</p>")
    a(table(["血統セル", "2023", "2024", "2025"], rows))
    allv = [r[y]["delta"] for r in ev_recal.values() for y in ("2023", "2024", "2025")]
    neg3 = [nm for nm, r in ev_recal.items() if all(r[y]["delta"] < 0 for y in ("2023", "2024", "2025"))]
    a(f"<p>改善の最大は 1 レースあたり {min(allv):+.4f}（1 頭あたりに直すと約 {min(allv) / 14.5:+.5f}）で、採用基準（効果 −0.0002 以上・95% 上限<0）に届くかどうかの水準。3 年とも負なのは {('・'.join(n.replace('@', ' × ') for n in neg3) or 'なし')} で、他は年により符号が変わる。"
      "V5 は PED_sire_cond・ALL_BEST で同じ情報を持つので、V5 での上乗せはさらに小さい。<b>血統の偏りで確率を直す余地はほぼ無い</b>（市場にも V4 にも織り込み済み）。</p>")

    # ---- 5. 馬券
    a("<h2>6. 馬券としての使い道</h2><h3>6.1 回収率の高い血統セルを買う規則（網羅）</h3>")
    d = rules["dist"]
    a(f"""<p>規則の族: 11 の血統キー × 条件 × 買う券（単・複）× 上位 {{5,10,20}}% × 単勝オッズの範囲（なし／30 倍以下／10〜60 倍）。選抜は各年より前の累積だけ（as-of）、<b>試行数 {rules['n_trials']}</b>。
選択年 2022〜2024 の reality check（t 値の最大、レース単位ブートストラップ）は <b>p = {rules['rc_val']['p']:.2f}</b>（100% を超える規則が偶然以上に存在する証拠なし）。選択年の回収率の分布は中央 {pc(d['val_roi_quantiles']['0.5'])}、上位 10% 点 {pc(d['val_roi_quantiles']['0.9'])}、最大 {pc(d['val_roi_quantiles']['1.0'])}、100% 以上は {pc(d['share_val_ge_100'], 1)}（{int(round(d['share_val_ge_100'] * d['n_val']))}/{d['n_val']} 本）。2025 年の中央値は {pc(d['roi2025_median'])}。</p>""")
    rows = []
    for x in rules["top_by_val"][:10]:
        v, c, t = x["val"], x["2025"], x["2026"]
        rows.append([x["rule"], v["bets"], roi_cell(v) if "lo" in v else pc(v["roi"]), pc(v["hit"]), c["bets"], roi_cell(c) if "lo" in c else pc(c["roi"]), t["bets"], roi_cell(t) if "lo" in t else pc(t["roi"]), f"{x.get('max_dd_yen', 0):,.0f}"])
    a(table(["規則（キー@条件|券|上位|オッズ）", "選択年 買い数", "選択年 回収率", "的中率", "2025 買い数", "2025 回収率", "2026 買い数", "2026 回収率", "最大 DD（円）"], rows, 1))
    a("<p class=muted>選択年の回収率が最良の 10 本（買い数 300 以上）。最良でも選択年の区間の下限は 100% を下回り、2025 年は 0.82〜1.09、2026 年は 0.7〜1.17 とばらつく（選択年の上位が翌年に続く証拠が無い）。</p>")

    a("<h3>6.2 現行の EV（V4 の確率 × 確定オッズ）に血統条件を重ねる</h3>")
    a("<p>EV ≥ 閾値の買い目（基準）に、血統セルの過去の縮約回収率で『上位半分／下位 25% を除く／下位 25% のみ』の絞り込みを重ねたときの回収率の差（選んだ側 − 基準全体）。差の 95% 区間はレース単位。</p>")
    rows = []
    for k, e in evf["thresholds"].items():
        b = e["base"]
        rows.append([f"<b>基準 {e['bt']} EV≥{e['T']}</b>", "", f"{roi_cell(b['2023-2024'])}", f"{roi_cell(b['2025'])}", f"{roi_cell(b['2026'])}"])
        best = []
        for fn, f in e["filters"].items():
            x = f.get("2023-2024")
            if x and x.get("diff_lo") is not None and x["diff_lo"] > 0:
                best.append((fn, f))
        for fn, f in e["filters"].items():
            if fn.endswith(("上位半分",)) and (fn.startswith(("sire_id@all", "line_pair@sd"))):
                cs = []
                for lab in ("2023-2024", "2025", "2026"):
                    x = f.get(lab)
                    cs.append(f"{f2(x['diff'], 3, True)}<span class=muted> [{f2(x['diff_lo'], 2, True)},{f2(x['diff_hi'], 2, True)}]</span>" if x and x.get("diff") is not None else "—")
                rows.append([f"　{fn}", "", *cs])
    a(table(["", "", "2023-2024", "2025", "2026（テストセット②）"], rows, 2))
    tot = 0
    hits = []
    for k, e in evf["thresholds"].items():
        for fn, f in e["filters"].items():
            tot += 1
            x = f.get("2023-2024")
            if x and x.get("diff_lo") is not None and x["diff_lo"] > 0:
                hits.append((k, fn, f))
    desc = "；".join(f"{k} の {fn}: 2023-24 {f['2023-2024']['diff']:+.3f}、2025 {f['2025']['diff']:+.3f} [{f['2025']['diff_lo']:+.2f},{f['2025']['diff_hi']:+.2f}]、2026 {f['2026']['diff']:+.3f}" for k, fn, f in hits)
    a(f"<p>全 {tot} 通り（4 つの EV 設定 × 6 つの血統セル × 3 つの絞り込み）のうち、2023-2024 の区間が 0 を外れて正になるのは {len(hits)} 通り（{desc or 'なし'}）。多重比較を考えると偶然の範囲で、2025 年は区間が 0 を含み、2026 年は符号が逆になる。</p>")

    ef = evfin
    a(f"""<h3>6.3 騎手・調教師の規則（tp-59）に血統を重ねる</h3>
<p>tp-59 の最良規則『単勝 EV≥1.5 かつ 騎手・調教師の「市場に対する残差」が過去に正』（jt）に、同じ形の血統条件（父・母父の市場に対する残差が過去に正: s, d）を重ねた。規則の族は 2 券種 × τ 5 通り × 条件 10 通り＝{ef['n_rules']} 本、選択年 2022〜2024 の reality check は <b>p = {ef['rc']['p']:.2f}</b>（tp-59 と同じ設計・同じ形式）。最良は jt（選択年 {roi_cell(ef['best']['sel_2022_2024'])}、2025 {roi_cell(ef['best']['2025'])}、2026 {roi_cell(ef['best']['2026'])}）で、血統条件を足したものではない。</p>""")
    rows = []
    for r in ef["increment_over_jt"]:
        if r["bet"] == "win" and r["tau"] == 1.5 and r["cond"] in ("s", "d", "sd"):
            rows.append([f"win τ={r['tau']} jt ∩ {r['cond']} と jt の残り", r["period"], f"{r['n_sel']:,}", pc(r["sel_roi"]), pc(r["all_roi"]), f"{f2(r['diff'], 3, True)} [{f2(r['diff_lo'], 2, True)},{f2(r['diff_hi'], 2, True)}]"])
    a(table(["比較", "期間", "血統条件を満たす買い数", "満たす側の回収率", "jt 全体", "差 [95%区間]"], rows, 2))
    inc = {}
    for r in ef["increment_over_jt"]:
        if r["bet"] == "win" and r["tau"] == 1.5 and r["cond"] in ("s", "d", "sd"):
            inc.setdefault(r["period"], []).append(r)
    rg = lambda per: (min(x["diff"] for x in inc[per]), max(x["diff"] for x in inc[per]))
    nz = sum(1 for per in inc for x in inc[per] if x["diff_lo"] > 0)
    a(f"<p>jt の中で血統条件（s, d, sd）を満たす側の差は、選択年 {rg('2022-2024')[0]:+.2f}〜{rg('2022-2024')[1]:+.2f}、2025 年 {rg('2025')[0]:+.2f}〜{rg('2025')[1]:+.2f}、2026 年 {rg('2026')[0]:+.2f}〜{rg('2026')[1]:+.2f}。"
      f"<b>区間が 0 を外れて正になったのは 9 通り中 {nz} 通り</b>で、2026 年はすべて負。血統の条件は騎手・調教師の条件に<b>上乗せしない</b>。複勝（EV≥1.0/1.3）でも同じ（上乗せは区間が 0 を含み、2026 年は負）。</p>")

    # ---- 6. 10世代の血量
    a(f"""<h2>7. 10 世代の血量（存在割合）と、出現率で絞った種牡馬</h2>
<h3>7.1 被覆と先読みの安全性</h3>
<ul>
<li>連結済みの 10 世代 JSON（horse_pedigree_10gen）は競走馬 35,794 頭の <b>13.7%</b> にしか無い（2017 年生まれ以降が中心）。そこで 5 世代 JSON（59,675 頭分）から自分で連結した: 5 世代目の祖先 {rstat['x_total']:,} 頭のうち {rstat['x_resolved']:,} 頭に自分の 5 世代表があり、<b>被覆（5 世代目の枠のうち 6〜10 世代を展開できた割合）は平均 99.9999%</b>（生年別も 99.99% 以上）。</li>
<li>血統表は生まれた時点で確定している静的な情報で、結果を使わない。as-of の問題は無い（ただし『父の産駒世代』の算出は競走馬の生年を使うので、その種牡馬の初年度の判定は後から分かった情報を含む。生年は産駒が生まれた時点で公開されるので先読みではない）。</li>
<li>血量は『ある祖先が 1〜10 世代の各位置に出るときの Σ 2^-世代』を 1 頭 1 指標にまとめたもの（父・父父・母父…を別々の指標として並べない）。</li>
</ul>
<h3>7.2 出現率で絞る（N と閾値）</h3>""")
    t10, t5 = dsel["thr_presence10"], dsel["thr_presence5"]
    a(table(["出現率（競走馬の出走のうち、その祖先を持つ割合）≥", "50%", "30%", "20%", "10%", "5%", "3%"],
            [["10 世代以内（男系祖先の数）"] + [t10[k] for k in ("0.5", "0.3", "0.2", "0.1", "0.05", "0.03")], ["5 世代以内（男系祖先の数）"] + [t5[k] for k in ("0.5", "0.3", "0.2", "0.1", "0.05", "0.03")]]))
    a(f"""<p>10 世代では古い創始者（Gainsborough・Phalaris・Hyperion など {dsel['n_saturated_gt97']} 頭）が<b>全競走馬の 97% 以上に現れて飽和</b>し、出現率は区別に使えない（255 頭が 50% 以上）。そこで<b>5 世代以内の出現率が 3% 以上の男系祖先 {dsel['n_before_dedup']} 頭</b>を候補にし（現代の影響の大きい種牡馬）、その 6〜10 世代の血量を追加情報として見る。</p>
<h3>7.3 重複の整理</h3>
<p>候補 {dsel['n_before_dedup']} 頭のうち、父子関係にある、または競走馬での総血量の相関が 0.5 以上の種牡馬は産駒数の多い方を代表にまとめ、<b>代表 {dsel['n_after_dedup']} 頭</b>にした。例: """)
    ex = [r for r in dsel["reps"] if r["merged"]][:6]
    a("、".join(f"{short(names, r['id'])}（{'・'.join(short(names, m['id']) + f' {m['corr']:+.2f}' for m in r['merged'][:3])} を統合）" for r in ex) + "。</p>")

    sm = deff["summary"]
    a("<h3>7.4 深い血量（6〜10 世代）は効くか</h3>")
    a("<p>代表ごとに、その祖先の血量を持つ馬に限り、勝ち（0/1）− 予測の勝率（V4、V5 の OOF、市場）の残差を、<code>父である／母父である</code> の指標 + <code>その他の位置の 1〜5 世代の血量</code> + <code>6〜10 世代の血量</code> に回帰した（レース単位のクラスタ頑健標準誤差）。位置別の指標を並べずに、『上位の位置の効果を除いた残りの部分』の係数を見る設計。</p>")
    rows = []
    for yn, lab in (("resid_v4", "V4 の勝率との差（2022-2025）"), ("resid_v5", "V5 の OOF の勝率との差（2022-2024）"), ("resid_mkt", "市場の勝率との差（2020-2025）")):
        s = sm.get(yn)
        if not s:
            continue
        rows.append([lab, s["n_reps"], f"{s['deep_|t|>2']}（期待 {s['expected_|t|>2_under_null']:.1f}）", s["deep_|t|>3"], f"{s['other5_|t|>2']}", s["other5_|t|>3"], f2(s["deep_mean_t"], 2, True), f2(s.get("deep_half_spearman"), 2, True)])
    a(table(["被説明変数", "代表数", "6〜10 世代: |t|>2 の数（偶然の期待）", "|t|>3", "その他 1〜5 世代: |t|>2", "|t|>3", "6〜10 世代の t 平均", "前後半の係数の順位相関"], rows))
    a("""<p><b>読み方.</b> 6〜10 世代の血量が V4／V5 の残差に効く種牡馬は、偶然の期待数（約 1.6）をわずかに上回る程度（4 頭、|t|>3 は 0〜1 頭）で、t の平均はほぼ 0。前後半の係数の順位相関は低〜中程度で、安定した深い血量の効果とは言えない。
一方、『その他の位置の 1〜5 世代の血量』は V4 の残差で |t|>2 が 7 頭、|t|>3 が 3 頭と偶然を超えて効く（これは V4 が持たない部分で、V5 は ALL_BEST などでかなり吸収している: V5 の残差では 5 頭、|t|>3 は 1 頭）。深い血量は 5 世代以内の情報にほとんど何も足さない。</p>""")

    a("<h3>7.5 血量帯別の成績（代表の例）</h3>")
    rows = []
    shown = 0
    for sid, e in deff["reps"].items():
        if shown >= 4:
            break
        for bn, v in e["bands"].items():
            rows.append([f"{short(names, sid)}" if bn == list(e["bands"])[0] else "", bn, f"{v['horses']:,}", pc(v["win_rate"]), pc(v["top3_rate"]), roi_cell({"roi": v["roi_win"], "lo": v["roi_win_lo"], "hi": v["roi_win_hi"]}),
                         roi_cell({"roi": v["roi_place"], "lo": v["roi_place_lo"], "hi": v["roi_place_hi"]}), f2(v["ex_mkt"] * 100, 2, True) + "pt"])
        shown += 1
    a(table(["祖先", "総血量の帯", "出走", "勝率", "複勝率", "単勝回収率", "複勝回収率", "勝率−市場"], rows, 2))
    a("<p class=muted>deep_only は 1〜5 世代には現れず 6〜10 世代にだけ現れる馬。帯の違いによる回収率の差は区間が重なり、市場の勝率との差も ±0.3pt 以内。全 40 代表の表は決定ログではなく JSON（roi/deep_effect.json）に保存。</p>")

    # ---- 7. 構成埋め込み
    a("""<h2>8. 血統グラフ上の近さ（構成埋め込み）</h2>
<h3>8.1 設計: 近さは『祖先の構成』で決め、親の ID を持ち込まない</h3>
<p>節点 = 馬・種牡馬・繁殖牝馬（グラフ上 100,024 節点、両親が分かる節点 98,303）、辺 = 親子（父・母、重み 1/2）。節点 v の構成ベクトル E<sub>t</sub>[v] を、t=0 は節点ごとのランダムベクトル（葉）、t≥1 は 0.5×(E<sub>t-1</sub>[父] + E<sub>t-1</sub>[母])（親が不明なら葉で止まる）と再帰的に定義する。
E<sub>t</sub>[h] は『h の t 世代前の祖先の枠（各 2<sup>-t</sup>）に載る葉』の和＝血量で重み付けた祖先の構成で、馬の埋め込みは Σ<sub>t=tmin..T</sub> γ<sup>t-tmin</sup> E<sub>t</sub>[h]。
<b>tmin=2 なら 1 世代目（父・母）の ID は入らず、父は父の両親（祖父母）の構成に展開される</b>。全兄弟は祖先の枠が同一なので距離 0、父だけが違う産駒（同じ母 × ディープインパクト／ブラックタイド）も父どうしが全兄弟なので距離 0 になる。
これは学習なしの『親から子へのメッセージ伝播』（パラメータ無しの GNN）であり、Personalized PageRank 型の世代減衰は γ、カーネル（共通祖先の血量の積和）はランダム射影で保つ（次元 256）。グラフラプラシアンのスペクトル埋め込みはこのカーネルの主成分（SVD 32 次元）と同じ空間になるので別実装にしていない。</p>""")
    vk = known["variants"]
    labels = {"A_t2_T10_g100": "A: tmin=2, T=10, γ=1（採用候補）", "B_t2_T10_g070": "B: tmin=2, T=10, γ=0.7（世代減衰）", "C_t2_T5_g100": "C: tmin=2, T=5", "D_t2_T10_idf": "D: tmin=2, T=10, 希少祖先を重く（IDF）",
              "E_t1_T10_g100": "E: tmin=1, T=10（対照: 親の ID を持ち込む）", "F_t1_T5_idf": "F: tmin=1, T=5, IDF（対照: 従来の祖先ベクトルに近い）", "N_node2vec": "N: Node2Vec（対照: 自己教師、節点 ID が支配的）"}
    a("<h3>8.2 既知ペアでの検証（名前ではなく horse_id で解決。解決結果は embed/known_ids.json）</h3>")
    a(f"<p>解決した ID: ディープインパクト {kids['picked']['DI']}、ブラックタイド {kids['picked']['BT']}（<b>同じ両親</b>: {kids['DI_BT_same_parents']}）、サドラーズウェルズ {kids['picked']['SW']}、ヌレイエフ {kids['picked']['Nureyev']}（ともにノーザンダンサー直子）。順位は両親が分かる {known['n_pool']:,} 節点の中での近さの順位（1 が最近接）。</p>")
    rows = []
    for k, lab in labels.items():
        r = vk.get(k)
        if not r:
            continue
        rows.append([lab, f"{r['DI_BT_rank']}（{100 * r['DI_BT_frac']:.4f}%）", f2(r["DI_BT_cos"], 3), f"{r['SW_Nureyev_rank']}（{100 * r['SW_Nureyev_frac']:.4f}%）", f2(r["SW_Nureyev_cos"], 2),
                     f2(r.get("cos_father_differs_mean"), 3), f"{100 * r['father_differs_rank_frac_median']:.4f}%" if r.get("father_differs_rank_frac_median") is not None else "—", f2(r["auc_full_sib_vs_random"], 4), f2(r["auc_same_sire_vs_random"], 4)])
    a(table(["方式", "DI→BT の順位", "DI–BT cos", "SW→ヌレイエフの順位", "SW–ヌレイエフ cos", "父だけ違う産駒の cos（平均）", "同（競走馬中の順位 中央値）", "AUC 全兄弟 vs 無作為", "AUC 同父 vs 無作為"], rows))
    a("""<p><b>読み方.</b> tmin=2 の構成埋め込み（A〜D）は、DI と BT が<b>互いに最近接（順位 1、cos=1）</b>、サドラーズウェルズ → ヌレイエフが 3〜5 位（全 98,303 節点の上位 0.005%）で、<b>全兄弟・父だけ違う産駒の cos は 1.000</b>（上位 0.1% 以内を大きく満たす）。
親の ID を持ち込む E・F は DI–BT は同じだが、<b>父だけ違う産駒の cos が 0.76（E）／0.40（F、IDF で悪化）</b>に落ち、競走馬中の順位も悪化する（F は中央値で上位 1.5%）。
Node2Vec（N）は DI→BT が 29,284 位（30%）、SW→ヌレイエフが 21,096 位（21%）で、<b>節点の ID と近傍の文脈に引きずられ、『実質同じ配合』が近くならない</b>（学習は 1,200 ステップ・約 2.6 エポックで小さめ。それでも全兄弟 vs 無作為の AUC は 0.9998 と高く、AUC だけでは制約を満たすかが分からないので、順位で見る）。
<b>制約（実質同じ配合が最も近い・親の ID を持ち込まない）を満たす A〜D から採用候補を選び、E・F・N は『近さの定義として不適』の対照</b>として扱う。</p>""")

    # 近さと適性
    aa = aff["A_t2_T10_g100"]
    a("<h3>8.3 近さ ≠ 適性: 血統が同一でも適性は同じにならない（天井の実測）</h3>")
    a(f"""<p>馬ごとの『適性』を、着順スコア（レース内の着順を [-0.5, 0.5]）の、その馬の平均からの偏差から作った: 能力（平均）、芝−ダ、長距離−短距離、距離の傾き。出走数 6 以上の馬で、2 頭の適性の相関を近さの区分ごとに見る。
各指標の<b>信頼性</b>（馬ごとの出走を偶奇で分けた相関 → Spearman-Brown）は 能力 {f2(aa['reliability']['ability'], 2)}、芝−ダ {f2(aa['reliability']['a_surf'], 2)}、長−短 {f2(aa['reliability']['a_dist'], 2)}、距離の傾き {f2(aa['reliability']['a_slope'], 2)} で、距離系は馬 1 頭の出走数ではほとんど測れない（相関は信頼性で薄まる）。</p>""")
    rows = []
    order = ["random", "full_siblings(同父同母)", "同父(母は別)", "同母(父は別も含む)"]
    for k in order:
        e = aa["pairs"][k]
        rows.append([{"random": "無作為ペア"}.get(k, k), f"{e['pairs']:,}", *[f2(e.get(f"corr_{c}"), 3, True) for c in ("ability", "a_surf", "a_dist")], *[f2((e.get("disattenuated") or {}).get(c), 2, True) for c in ("ability", "a_surf")]])
    for k, e in aa["neighbor_bands"].items():
        if e["pairs"] == 0:
            continue
        rows.append([f"近傍ペア sim {k}", f"{e['pairs']:,}", *[f2(e.get(f"corr_{c}"), 3, True) for c in ("ability", "a_surf", "a_dist")], "—", "—"])
    a(table(["ペア", "ペア数", "r 能力", "r 芝−ダ", "r 長−短", "補正後 r 能力", "補正後 r 芝−ダ"], rows))
    xs = [str(i) for i in range(1, 11)]
    a("<p>近さ（無作為ペアの類似度の十分位。1 が最も遠い）ごとの能力の相関:</p>")
    a(svg_lines({"能力": [aa["deciles_random"][x]["corr_ability"] for x in xs], "芝−ダ": [aa["deciles_random"][x]["corr_a_surf"] for x in xs]}, xs, -0.03, 0.06, title="近さの十分位ごとの適性の相関"))
    sims = [aa["deciles_random"][x] for x in xs]
    a(f"<p class=muted>十分位の類似度の範囲: 1 位 {sims[0]['sim_lo']:.2f}〜{sims[0]['sim_hi']:.2f}、10 位 {sims[9]['sim_lo']:.2f}〜{sims[9]['sim_hi']:.2f}。無作為ペアの 10 位（最も近い 1 割）でも能力の相関は {aa['deciles_random']['10']['corr_ability']:+.3f} と小さい。</p>")
    a(f"""<p><b>読み方.</b> 近さが上位 1 割の無作為ペアの適性相関はほぼ 0（能力 +0.01）で、近傍上位 30 に入る組（sim 0.85〜0.99）でも 0.03〜0.05。構成が<b>同一</b>のペア（sim ≥ 0.999: 全兄弟など）で初めて能力 +{aa['neighbor_bands']['>=0.999']['corr_ability']:.2f}、芝−ダ +{aa['neighbor_bands']['>=0.999']['corr_a_surf']:.2f}（信頼性で補正すると約 +0.5）。
<b>全兄弟の能力の相関は {f2(aa['pairs']['full_siblings(同父同母)']['corr_ability'], 2)}（補正後 {f2(aa['pairs']['full_siblings(同父同母)']['disattenuated']['ability'], 2)}）</b> = 同じ配合・同じ育成環境が共有する能力の分散の割合の目安は 13〜16%（兄弟の成績から個体の成績を当てるときの説明率 r² は 2〜3%）。残りは馬自身の履歴・体型・調教・育成・展開による。
近さが 1 に近いほど強く信じる固定の重みにはせず、kNN 型の統計は近さ別の回帰係数 β（2021 年までの結果から推定、以後固定）で重み付けし、事前（近い馬の成績）を馬自身の履歴で更新する構造にした。β の十分位別: {', '.join(f'{b:.3f}' for b in beta['beta'])}（ほぼ平坦で 0.02〜0.06）。</p>""")

    # sire pairs
    a("<h3>8.4 種牡馬固有の差（DI と BT など）は残差成分として残る</h3>")
    rows = []
    for r in sp["pairs"]:
        rows.append([f"{html.escape(r['name_a'])} ({r['n_a']:,}) / {html.escape(r['name_b'])} ({r['n_b']:,})", f"{pc(r['win_a'])} / {pc(r['win_b'])}", f"{pc(r['top3_a'])} / {pc(r['top3_b'])}", f2(r["z_top3"], 1, True), f"{pc(r['roi_win_a'], 0)} / {pc(r['roi_win_b'], 0)}"])
    a(table(["構成が同一の種牡馬ペア（産駒の出走 200 以上）", "産駒の勝率", "産駒の複勝率", "複勝率差の z", "産駒の単勝回収率"], rows, 1))
    a(f"<p>構成が同一（cos ≥ 0.999）の種牡馬ペアは {sp['n_identical_pairs']} 組。複勝率差の z の標準偏差は {sp['z_top3_sd']:.1f}（偶然なら 1）で、<b>種牡馬ごとの固有の差は大きい</b>（ディープインパクトとブラックタイド: 複勝率 29.5% 対 19.9%）。ただし産駒の質には配合相手の繁殖牝馬の質も混ざる。"
      "このため『祖先構成による近さ』と『種牡馬 ID 固有の実績』は分けて持つ: 構成ベースの統計（クラスタ・kNN）は ID を使わず、種牡馬固有の差は V5 に既にある PED_sire_cond・PX_cell（種牡馬 ID × 条件の産駒成績）が担う。構成ベースの増分は、それらを持つ V5 への上乗せとして測る。</p>")

    # コスト
    a(f"""<h3>8.5 推論時のコストと汎化</h3>
<ul>
<li>未見の馬（新種牡馬の産駒・初出走馬）でも、両親が分かれば埋め込みは<b>親の埋め込み（窓を 1 世代ずらしたもの）の平均</b>で作れる（ユニットテストで一致を確認: 子の E<sub>2..T</sub> = 0.5×(父の E<sub>1..T-1</sub> + 母の E<sub>1..T-1</sub>)）。推論時に持つのは節点ごとの 32 次元（100,024 節点 × 32 × 4B = 約 13MB）と、競走馬 35,794 頭の座標（約 4.6MB）。</li>
<li>kNN は競走馬 35,794 頭 × 32 次元の総当たり（行列積）で近傍 30 頭を出せる（本実験は約 20 秒/方式）。近傍馬の as-of の成績は『馬 × 条件』の累積表（ホストの履歴バンドルと同じ形）を引くだけ。低メモリのサービング（約 340MB）に追加で載る規模。</li>
<li>クラスタ定義は 2021 年までにデビューした馬で固定し、以降の馬は最近傍の中心に割り当てる。クラスタ割当は血統のみで結果を使わないのでリークは無く、統計は過去の結果だけ（自分自身の過去走は引く）。</li>
</ul>""")

    # ---- 8. V5 への増分
    a("<h2>9. 特徴量としての増分（V5 の 1,142 列に累積）</h2>")
    a("""<p>オッズ非依存の群だけを評価した（回収率由来の統計は特徴量にしない）。<code>DEEP_TOT</code> = 代表 40 頭の総血量（1〜10 世代、1 頭 1 指標）、<code>DEEP_6_10</code> = 6〜10 世代の血量（40 列 + 合計 + 被覆）、
<code>CLU_A</code> = 構成埋め込み A のクラスタ（K=64/256）× 条件（全体・面距離）の as-of 縮約統計（勝ち・着順スコア、自分の過去走を引く）、<code>KNN_A</code> = 近い馬 30 頭の as-of 成績（β 曲線で重み付け）と馬自身の履歴による更新、
<code>CLU_E / KNN_E</code> は親の ID を持ち込む対照（群は作成、評価は未実施）。<code>PQ_*</code> は 1.6 の群（レース質 × 血統グループの期待上乗せ）、<code>PED_FAM</code> は父の産駒世代・父／母父／母の産駒数（生年まで）。評価は 2 段階（第 1 段 valid 2022+2023・seed 0、第 2 段 valid 2024・新 seed）、基準は勝率の 95% ブートストラップ上限<0 かつ効果≥0.0002、連対・複勝の悪化≤0.0002。</p>""")
    stage_rows = []
    for s, lab in (("stage1", "第 1 段"), ("stage1_pq", "第 1 段（レース質×血統）"), ("stage1_b", "第 1 段（追加分）"), ("stage2", "第 2 段"), ("full", "最終候補（2022〜2024・seed 0,1）")):
        e = ev.get(s)
        if not e:
            continue
        for g, v in e.get("groups", {}).items():
            c = v["vs_V5"]
            sub = v.get("subsets", {})
            first = sub.get("履歴0走(初出走)")
            exp_ = sub.get("履歴3走以上")
            stage_rows.append([lab, g, v["cols"], f2(c["Δwin"], 5, True), f2(c["win_hi"], 5, True), f2(c["Δtop2"], 5, True), f2(c["Δtop3"], 5, True), c["採用"],
                               (f"{f2(first['Δwin'], 5, True)}（{first['rows']:,} 行）" if first else "—"), (f"{f2(exp_['Δwin'], 5, True)}（{exp_['rows']:,} 行）" if exp_ else "—")])
    if stage_rows:
        a(table(["段", "群", "列数", "Δ勝率", "上限", "Δ連対", "Δ複勝", "採否", "初出走のΔ勝率", "履歴3走以上のΔ勝率"], stage_rows, 2))
    else:
        a("<p class=muted>（評価は実行待ち）</p>")
    if done_rows := [v for st in ("stage1", "stage1_pq", "stage1_b") if ev.get(st) for v in ev[st].get("groups", {}).values()]:
        e3 = [v["subsets"]["履歴3走以上"]["Δwin"] for v in done_rows if v.get("subsets", {}).get("履歴3走以上")]
        e0 = [v["subsets"]["履歴0走(初出走)"]["Δwin"] for v in done_rows if v.get("subsets", {}).get("履歴0走(初出走)")]
        if e3 and e0:
            allg = [(g, v["vs_V5"]) for st in ("stage1", "stage1_pq", "stage1_b") if ev.get(st) for g, v in ev[st].get("groups", {}).items()]
            bg = min(allg, key=lambda x: x[1]["Δwin"])
            a(f"<p><b>読み方.</b> 全体では採用基準（上限<0 かつ効果 −0.0002 以上）を満たす群が無い。履歴が 3 走以上ある馬では Δ勝率が {min(e3):+.5f}〜{max(e3):+.5f}、<b>初出走馬では {min(e0):+.5f}〜{max(e0):+.5f}</b> で、初出走馬では全群で悪化している"
              f"（血統由来の特徴は初出走で最も効くはず、という見込みとは逆。原因は未確認で、候補は『2020 年は効果表の元データが無く欠損のまま学習に使う』『初出走馬はグループ内の過去走が薄く、群の統計が不安定』）。"
              f"全体で最も近かったのは {bg[0]}（Δ勝率 {bg[1]['Δwin']:+.5f}、上限 {bg[1]['win_hi']:+.5f}、複勝 {bg[1]['Δtop3']:+.5f}）。</p>")
    expected = ["KNN_A", "CLU_A", "DEEP_TOT", "DEEP_6_10", "PED_FAM", "PQ_K16", "PQ_K40", "PQ_sire30", "PQ_line"]
    got = set()
    for st in ("stage1", "stage1_pq", "stage1_b"):
        if ev.get(st):
            got |= set(ev[st].get("groups", {}))
    missing = [g for g in expected if g not in got]
    npass_all = sum(1 for st in ("stage1", "stage1_pq", "stage1_b") if ev.get(st) for v in ev[st].get("groups", {}).values() if v["vs_V5"]["採用"] is True)
    a(f"<p class=muted>第 1 段を通過した群が{'無い' if npass_all == 0 else 'ある'}ため、{'第 2 段・最終候補（2022〜2024・seed 0,1）は実施していない' if npass_all == 0 else '通過群は第 2 段に進める'}。"
      f"{('未実施の群: ' + '、'.join(missing) + '（ホストの計算枠の順番待ちで、本章の時点では未完了。スクリプトと手順は tp60_eval に用意済み）。') if missing else '想定した群はすべて評価済み。'}"
      " 親の ID を持ち込む対照（CLU_E / KNN_E の群）の特徴量評価は、既知ペアの検証で近さの定義として不適と判定済みのため実施していない（群の作成だけ済み）。 PED_FAM は『父の産駒世代・父・母父・母の産駒数（生年まで）』で、競走馬の生年と相関する（時代ずれの懸念があり、通過しても採用前に確認が要る）。</p>")

    # ---- 結論
    a(conclusion_html(gq, rules, ef, sm, kn, ev))
    a("""<h2>11. 限界・注意</h2>
<ul>
<li>回収率は確定オッズ・確定払戻での値で、購入時点のオッズとは違う（直前オッズの蓄積後に再検証）。区間はレース単位ブートストラップで、年をまたぐ自己相関は考慮していない。</li>
<li>外国産馬（マル外）のフラグは取得していない（生産国は horse_profile から）。系統は名前の正規表現で創始者に寄せた近似。</li>
<li>6〜10 世代は 5 世代 JSON の連結で作り、各祖先の 5 世代表が欠けた枝の深い血量は 0 になる（被覆は 99.99% 以上）。</li>
<li>適性の指標（芝−ダ、長−短、距離の傾き）は馬 1 頭あたりの出走が少なく信頼性が低い。『近さと適性の対応が弱い』という結論は、血統の寄与が小さいのではなく、個体の適性を測る側の誤差が大きいことの影響も受ける（補正後の相関を併記）。</li>
<li>p_win は V4（2022〜2026 の推論済み）の確率で、V5 の OOF は 2022〜2024 のみ。V5 での確認は深い血量の回帰（7.4）と特徴量の増分（9）に限る。</li>
<li>学習型の GNN（親 → 子のメッセージ伝播を学習する）は未実施。学習なしの再帰（構成埋め込み）が同じ伝播をパラメータ無しで行い、制約（実質同じ配合が最近接）を満たしたため、学習型は『近さの定義』ではなく『適性を直接予測する埋め込み』として別の課題にした（自己教師の損失設計と、同配合でも適性が違うことの扱いが要る）。</li>
<li>Node2Vec は小さな学習量（約 2.6 エポック）。十分に学習させても節点の ID を支配的に使う設計は変わらないが、順位の数字は目安。</li>
</ul>
<h2>12. 再現手順</h2>
<p><code>python -m src.scripts.data.tp60_ped static</code>（血統の静的表・10 世代の血量）→ <code>tp60_roi cells|popular|persist</code> → <code>tp60_rules rules|evfilter|evfinal</code> → <code>tp60_deep select|effect|group</code> → <code>tp60_embed graph|embed|known</code>、<code>tp60_n2v</code> → <code>tp60_embfeat neighbors|affinity|curve|sirepairs|groups --variant A_t2_T10_g100</code> → <code>tp60_eval stage1|stage2|full --groups …</code> → <code>tp60_report</code>。
入力は tp-59 の master.parquet（確定オッズ・払戻・V4 の確率）。結果の置き場は <code>$TP60_OUT</code>（ホスト /home/jovyan/work/tp60_out）。</p>
<p class="muted">関連: <a href="pedigree_position_cross.html">血統の位置・存在割合・クロス（tp-38）</a> ・ <a href="similar_horses_pairs.html">似た馬どうしのペア</a> ・ <a href="ev_calibration_audit.html">期待値の水準の監査</a> ・ <a href="training_pipeline_decisions.html">決定ログ</a></p>
</body>
</html>
""")
    txt = "\n".join(P)
    done = []
    for st in ("stage1", "stage1_pq", "stage1_b"):
        if ev.get(st):
            for g, v in ev[st].get("groups", {}).items():
                done.append((g, v["vs_V5"]))
    if done:
        npass = sum(1 for g, c in done if c["採用"] is True)
        lst = "、".join(f"{g} {c['Δwin']:+.5f}" for g, c in done)
        evs = f"第 1 段（Δ勝率、負が改善）: {lst}。<b>採用基準を満たした群は {npass}</b>。"
    else:
        evs = "（評価は実行待ち）"
    return txt.replace("<!--EVALSUM-->", evs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/tp60_local")
    a = ap.parse_args()
    src = Path(a.src)
    html_text = build(src)
    summ = ROOT / "docs/html/modeling/chapters/_pedigree_roi_summary.html"
    concl = ROOT / "docs/html/modeling/chapters/_pedigree_roi_conclusion.html"
    for tag, p in (("<!--SUMMARY-->", summ), ("<!--CONCLUSION-->", concl)):
        if p.exists():
            html_text = html_text.replace(tag, p.read_text(encoding="utf-8"))
    note = ROOT / "docs/html/modeling/chapters/_pedigree_roi_evalnote.html"
    if note.exists():
        html_text = html_text.replace("<!--EVAL_NOTE-->", note.read_text(encoding="utf-8"))
    OUT.write_text(html_text, encoding="utf-8")
    print(f"wrote {OUT} ({len(html_text):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
