"""体重の増減と指数の結びつき・成長時期の血統・成長指数（tp-57）の章 HTML を、結果ファイルから作る。dev PC で実行する。

  python -m src.research.pedigree.growth_weight_report --res <結果を置いたディレクトリ> --out docs/html/modeling/chapters/weight_growth_performance.html

入力（--res）: summary.json、growth_index_eval.json、growth_index_explore.json、growth_index_explore_residual.json、sire_profile.csv、dam_sire_profile.csv、
               （あれば）v4_stage1.json / v4_stage2.json。数値はすべてこれらから埋める（手書きの数値は置かない）。
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import pandas as pd

CSS = """:root { --bg:#0f172a; --panel:#15213c; --border:#243358; --text:#e6ecff; --muted:#9aa7c7; --accent:#4f8bff; --accent-2:#7aa9ff; --warn:#fbbf24; --ok:#4ade80; --bad:#f87171; }
* { box-sizing: border-box; } body { font-family: "Noto Sans JP", system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.8; font-size: 15px; padding: 40px 56px 96px; max-width: 1100px; margin: 0 auto; }
a { color: var(--accent-2); } h1 { font-size: 26px; } h2 { font-size: 20px; margin: 40px 0 12px; border-left: 4px solid var(--accent); padding-left: 12px; } h3 { font-size: 16px; margin: 22px 0 8px; }
table { border-collapse: collapse; margin: 12px 0 18px; font-size: 13px; background: var(--panel); } th, td { border: 1px solid var(--border); padding: 5px 10px; text-align: left; vertical-align: top; } th { background: #1b2a4d; }
td.num { text-align: right; font-family: monospace; } .muted { color: var(--muted); font-size: 13px; } .note { background: var(--panel); border: 1px solid var(--border); border-left: 4px solid var(--warn); border-radius: 8px; padding: 12px 16px; margin: 14px 0; }
.key { background: var(--panel); border: 1px solid var(--border); border-left: 4px solid var(--ok); border-radius: 8px; padding: 12px 16px; margin: 14px 0; } svg { max-width: 100%; height: auto; background: var(--panel); border: 1px solid var(--border); border-radius: 8px; margin: 8px 0 14px; }
.pos { color: var(--ok); } .neg { color: var(--bad); } code { font-family: ui-monospace, Menlo, Consolas, monospace; background: #0b1428; padding: 1px 5px; border-radius: 4px; }"""


def f(x, nd=2, sign=False):
    if x is None:
        return "—"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "−")


def ci(c, nd=2, sign=True):
    v = next(c[k] for k in ("coef", "mean", "diff", "slope") if k in c)
    return f"{f(v, nd, sign)}（{f(c['lo'], nd, sign)}〜{f(c['hi'], nd, sign)}）"


def short_name(n: str) -> str:
    s = re.sub(r"[A-Za-z].*$", "", str(n or "")).strip()
    return s or str(n)


def ci_svg(items, ylabel: str, w=760, h=270, color="#4f8bff") -> str:
    """items: [(label, n, coef, lo, hi)]。縦に誤差棒つきの点。0 に線。"""
    pl, pr, pt, pb = 56, 16, 14, 56
    lo = min(min(i[3] for i in items), 0) - 0.15
    hi = max(max(i[4] for i in items), 0) + 0.15
    X = lambda k: pl + (k + 0.5) * (w - pl - pr) / len(items)
    Y = lambda v: pt + (hi - v) * (h - pt - pb) / (hi - lo)
    out = [f"<svg viewBox='0 0 {w} {h}' role='img' aria-label='{html.escape(ylabel)}'>"]
    for t in range(int(lo) - 1, int(hi) + 2):
        if lo <= t <= hi:
            out.append(f"<line x1='{pl}' x2='{w - pr}' y1='{Y(t):.1f}' y2='{Y(t):.1f}' stroke='{'#7aa9ff' if t == 0 else '#243358'}' stroke-width='{1.4 if t == 0 else 1}'/>")
            out.append(f"<text x='{pl - 6}' y='{Y(t) + 4:.1f}' font-size='11' fill='#9aa7c7' text-anchor='end'>{t}</text>")
    for k, (lab, n, c, l, u) in enumerate(items):
        col = "#f87171" if u < 0 else ("#4ade80" if l > 0 else color)
        out.append(f"<line x1='{X(k):.1f}' x2='{X(k):.1f}' y1='{Y(u):.1f}' y2='{Y(l):.1f}' stroke='{col}' stroke-width='2'/>")
        out.append(f"<circle cx='{X(k):.1f}' cy='{Y(c):.1f}' r='4.5' fill='{col}'/>")
        out.append(f"<text x='{X(k):.1f}' y='{h - 34}' font-size='11' fill='#e6ecff' text-anchor='middle'>{html.escape(str(lab))}</text>")
        out.append(f"<text x='{X(k):.1f}' y='{h - 18}' font-size='10' fill='#9aa7c7' text-anchor='middle'>n={n:,}</text>")
    out.append(f"<text x='12' y='{(pt + h - pb) / 2:.0f}' font-size='11' fill='#9aa7c7' transform='rotate(-90 12 {(pt + h - pb) / 2:.0f})' text-anchor='middle'>{html.escape(ylabel)}</text></svg>")
    return "".join(out)


def bars_svg(items, unit: str = "", w=760, row=26, maxv=None, color="#4f8bff") -> str:
    """items: [(label, value, note)]。横棒。"""
    pl, pr = 190, 70
    mv = maxv or max(abs(i[1]) for i in items) or 1
    h = row * len(items) + 12
    out = [f"<svg viewBox='0 0 {w} {h}'>"]
    for k, (lab, v, note) in enumerate(items):
        y = 6 + k * row
        wd = (w - pl - pr) * abs(v) / mv
        out.append(f"<text x='{pl - 8}' y='{y + 16}' font-size='12' fill='#e6ecff' text-anchor='end'>{html.escape(lab)}</text>")
        out.append(f"<rect x='{pl}' y='{y + 3}' width='{max(wd, 1):.1f}' height='{row - 9}' fill='{color if v >= 0 else '#f87171'}' rx='3'/>")
        out.append(f"<text x='{pl + wd + 6:.1f}' y='{y + 16}' font-size='12' fill='#9aa7c7'>{html.escape(f(v, 3) + unit)} {html.escape(note)}</text>")
    out.append("</svg>")
    return "".join(out)


def table(head, rows, num_cols=()):
    t = "<table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>"
    for r in rows:
        t += "<tr>" + "".join(f"<td class='{'num' if i in num_cols else ''}'>{c}</td>" for i, c in enumerate(r)) + "</tr>"
    return t + "</table>"


def build(res: Path) -> str:
    S = json.loads((res / "summary.json").read_text(encoding="utf-8"))
    E = json.loads((res / "growth_index_eval.json").read_text(encoding="utf-8"))
    X1 = json.loads((res / "growth_index_explore.json").read_text(encoding="utf-8")) if (res / "growth_index_explore.json").exists() else {}
    X2 = json.loads((res / "growth_index_explore_residual.json").read_text(encoding="utf-8")) if (res / "growth_index_explore_residual.json").exists() else {}
    V1 = json.loads((res / "v4_stage1.json").read_text(encoding="utf-8")) if (res / "v4_stage1.json").exists() else None
    V2 = json.loads((res / "v4_stage2.json").read_text(encoding="utf-8")) if (res / "v4_stage2.json").exists() else None
    sire = pd.read_csv(res / "sire_profile.csv")
    A, B, C = S["A"], S["B"], S["C"]
    lin, quad = A["linear"], A["quad"]
    bins = A["bins"]["bins"]
    om = A["own_mean"]
    big_up = next(b for b in bins if b["bin"] == "≥10")
    big_dn = next(b for b in bins if b["bin"] == "≤-11")
    d_rel, l_rel = B["delta"]["reliability"], B["level"]["reliability"]
    ev = C["eval"]
    e4, e1, e2, nz = E["cand4_adopted"], E["cand1_lgb"], E["cand2_additive"], E["noise"]
    lv = E["levels_test"]
    comp, logo, only = E["components"], E["leave_one_group_out"], E["only_group"]
    s_banse = ev["sire:late_minus_early_idx->late_minus_early_idx:min10"]
    s_wt = ev["sire:late_minus_early_wt->late_minus_early_wt:min10"]
    d_wt = ev["dam_sire:late_minus_early_wt->late_minus_early_wt:min10"]
    d_banse = ev["dam_sire:late_minus_early_idx->late_minus_early_idx:min10"]
    s_x = ev["sire:late_minus_early_wt->late_minus_early_idx:min10"]

    o = []
    o.append("<h1>体重の増減と指数の結びつき・成長時期の血統・成長指数（tp-57）</h1>")
    o.append(f"<p class='muted'>ユーザーの問い（2026-10-21）: ①馬体重の増加とパフォーマンス（めぐ指数）の上昇が、各馬ごとにどのくらい結びついているかを可視化できないか。②それを血統分析に使えないか（体重が増えやすい時期・晩成など）。"
             f"③馬体重の発表（T-60）の時点で、追い切り・血統・成長期の要素から『成長指数』を作れないか（UI 用と、モデルの特徴量用）。"
             f"対象: 2020〜2025 年の JRA {S['rows']:,} 出走・{S['horses']:,} 頭。指数は絶対値のめぐ指数 <code>megu_v3a_abs_robust</code>（斤量 55kg 換算済み、1 点 = 0.1 秒）。低信頼・障害は除く。2026 年は使わない。</p>")

    o.append("<h2>1. 要点</h2><div class='key'><ul>")
    o.append(f"<li><b>A 母集団: 『体重が増えたから指数が上がる』という関係は、前走比では見えません。</b>統制（前走の指数水準・月齢段階・休養・距離・面・性・前走体重）のうえで、前走比 +10kg あたりの指数の変化は {ci(lin)} 点で、0 と区別できません（{A['n_pairs']:,} 組・{A['n_races']:,} レース、レース単位のクラスタ頑健）。"
             f"むしろ <b>大きく動くほど悪化</b>します（前走比 ≥+10kg は {ci(big_up)} 点、≤−11kg は {ci(big_dn)} 点）。一方 <b>自分の過去平均より重い</b>状態は、指数の高さと結びつきます（+10kg あたり {ci(om['linear'])} 点）。</li>")
    o.append(f"<li><b>B 馬ごと: 個体差は『ある』が、先の走を予測するほど安定していません。</b>馬ごとの傾き（指数の変化 ÷ 体重の変化）の真の個体差は τ={f(B['delta']['tau'])} 点/10kg ですが、馬ごとの推定誤差は平均 {f(B['delta']['mean_se'])}、縮約後に個体の値が残る割合は平均 {f(B['delta']['mean_weight'], 2)}。"
             f"前半の走で出した傾きが後半の走を予測するか（時系列で前半・後半に分割）は順位相関 {f(d_rel['firstsecond']['spearman'], 3)}（帰無の 95% 点 {f(d_rel['firstsecond']['null_sd'] * 1.96, 3)} 程度、z={f(d_rel['firstsecond']['z'], 1)}）で、<b>検出できません</b>。"
             f"奇数走と偶数走の分割では {f(d_rel['oddeven']['spearman'], 3)}（z={f(d_rel['oddeven']['z'], 1)}）と弱く検出されますが、隣り合う走の共通の好不調を拾っているだけです。→ <b>『馬ごとの結びつき指数』は UI に載せません。</b>代わりに、成長曲線に体重を重ねて見せます。</li>")
    o.append(f"<li><b>C 血統: 体重の伸びは血統で再現します。晩成度は父だけ、弱く再現します。</b>2017〜2019 年生まれ（〜2023 年のデータ）で作った種牡馬の値が、別の馬たち（2020〜2021 年生まれ、〜2025 年）の産駒で再現するかを見ました。"
             f"体重の伸び（4 歳以上 − 2〜3 歳の平均体重）: 父 ρ={f(s_wt['spearman_shrunk'], 2)}・上位 20% − 下位 20% = {ci(s_wt['top_vs_bottom20'], 1)} kg、母父 ρ={f(d_wt['spearman_shrunk'], 2)}・{ci(d_wt['top_vs_bottom20'], 1)} kg。"
             f"晩成度（4 歳以上 − 2〜3 歳の平均指数）: 父 ρ={f(s_banse['spearman_shrunk'], 2)}、馬単位の回帰の傾き {ci(s_banse['horse_regression'])}、上位 − 下位 {ci(s_banse['top_vs_bottom20'], 1)} 点（区間が 0 をまたぐ）。母父の晩成度は個体差ゼロ（τ²={f(d_banse['tau2_train'], 2)}）。"
             f"<b>ただし体重の伸びと指数の伸びは、馬のレベルでほぼ無関係</b>（相関 {f(C['horse_level_corr_train']['pearson'], 2)}、n={C['horse_level_corr_train']['n']:,}）で、種牡馬の体重の伸びは産駒の指数の伸びを予測しません（父 ρ={f(s_x['spearman_shrunk'], 2)}、傾き {ci(s_x['horse_regression'])}）。</li>")
    o.append(f"<li><b>E 成長指数: 作れます。ただし中身の主役は『追い切り・厩舎コメント』で、体重と血統の上乗せはほぼありません。</b>目的変数は『今走の指数 − 直近水準』から統制で説明できる分を引いた残差（1 走の指数のノイズ σ={f(nz['sigma_e'], 1)} 点が大きく、検出できる相関の上限は {f(nz['max_corr_now'], 2)}）。"
             f"採用した小さな木のモデルの相関は {f(e4['pearson_now'], 3)}（上限の約 {f(e4['pearson_now'] / nz['max_corr_now'] * 100, 0)}%）、五分位の上位 − 下位は {ci(e4['top_minus_bottom_now'], 1)} 点（= {f(e4['top_minus_bottom_now']['diff'] / 10, 2)} 秒）。"
             f"要素別: 追い切り単独 {f(only['oikiri'], 3)}、成長期 {f(only['stage'], 3)}、体重 {f(only['weight'], 3)}、血統 {f(only['ped'], 3)}。抜いたときの落ち: 追い切り {f(logo['oikiri'], 3)}、成長期 {f(logo['stage'], 3)}、体重 {f(logo['weight'], 3)}、血統 {f(logo['ped'], 3)}。</li>")
    feat = "第 5.6 節（実行中または未実施）"
    if V1 and V2 and V1.get("groups") and V2.get("groups"):
        g2 = list(V2["groups"])[0]
        c1, c2 = V1["groups"][g2]["vs_V4"], V2["groups"][g2]["vs_V4"]
        feat = f"<b>見送り</b>（V4 に足した勝率 LogLoss は、valid 2022+2023 で {f(c1['Δwin'], 5, True)}、valid 2024 で {f(c2['Δwin'], 5, True)} と一貫せず、最小効果量 0.0002 にも届かない。第 5.6 節）"
    o.append(f"<li><b>UI</b>: 5 段階（下降・やや停滞・横ばい・成長・大きく成長）と理由の短文（第 5.4・5.5 節）。<b>特徴量化</b>: {feat}。</li></ul></div>")

    # ── 2. A
    o.append("<h2>2. A 母集団: 前走比の体重変化と、指数の変化</h2>")
    o.append(f"<p>連続する 2 走（休養 7〜180 日、低信頼・障害を除く）の指数の差 Δ指数 を、前走比の体重 Δ体重 のビンごとに、統制をそろえて比べました。基準は −1〜+1kg。統制は前走の指数（2 次）、月齢段階（6 区分）、休養日数（対数）、距離、面、性、前走体重。標準誤差は<b>レース単位のクラスタ</b>（同じレースの馬を独立と数えない）。</p>")
    o.append(f"<p>粗い平均（統制なし）では、体重が大きく減った組ほど指数が上がって見えます（≤−11kg で +{f(A['raw_by_bin'][0]['mean_didx'])} 点）が、これは前走が悪かった馬の平均への回帰です。統制すると次のようになります。</p>")
    o.append(ci_svg([(b["bin"], b["n"], b["coef"], b["lo"], b["hi"]) for b in bins], "基準（−1〜+1kg）との差（指数ポイント）"))
    o.append(table(["前走比の体重(kg)", "組数", "基準との差（点）", "95% 区間"], [[b["bin"] + ("（基準）" if b.get("ref") else ""), f"{b['n']:,}", f(b["coef"], 2, True), f"{f(b['lo'], 2, True)}〜{f(b['hi'], 2, True)}"] for b in bins], num_cols=(1, 2, 3)))
    o.append(f"<p>線形の傾き（+10kg あたり）は {ci(lin)} 点。2 次の項を入れると、1 次 {ci(quad)}・2 次 {ci({'coef': quad['coef_sq'], 'lo': quad['lo_sq'], 'hi': quad['hi_sq']})}（体重変化/10kg の 2 乗あたり）と<b>上に凸（逆 U 字）</b>で、頂点は前走比 ほぼ 0kg 付近です。<b>体重が安定しているときが最良で、増えても減っても大きく動くと悪化</b>します。</p>")
    o.append("<h3>2.1 自分の過去平均からのずれ（適正体重付近が良いか）</h3>")
    o.append(f"<p>その走より前の自分の平均体重・平均指数（3 走以上）からのずれで比べます（{om['n']:,} 走）。体重が自分の平均より重いほど、指数は自分の平均より高い傾向です。成長の積み重ね（体が大きくなり、同時に指数も上がる）を含みます。</p>")
    o.append(ci_svg([(b["bin"], b["n"], b["coef"], b["lo"], b["hi"]) for b in om["bins"]["bins"]], "基準（−2〜+2kg）との指数の差（点）", color="#7aa9ff"))
    o.append(f"<p>線形は +10kg あたり {ci(om['linear'])} 点、2 次では 1 次 {ci(om['quad'])}・2 次の係数 {f(om['quad']['coef_sq'], 3)}（上に凸）。頂点は約 +{f(om['quad']['coef'] / (2 * -om['quad']['coef_sq']) * 10, 0)}kg で、それ以上はほぼ頭打ちです。</p>")
    o.append("<h3>2.2 どんな馬で効くか（前走比 +10kg あたりの指数の変化）</h3>")
    rows = [[k, f"{v['n']:,}", f(v["coef"], 2, True), f"{f(v['lo'], 2, True)}〜{f(v['hi'], 2, True)}"] for k, v in A["subgroups"].items()]
    o.append(table(["部分集団", "組数", "傾き（点/10kg）", "95% 区間"], rows, num_cols=(1, 2, 3)))
    o.append("<p>区間が 0 を含まないのは、<b>休養 36〜180 日は増加が良化と結びつく</b>（休み明けの馬体増は回復・成長を表すのかもしれません。これも推測です）、<b>休養 35 日以内と 5 歳以上は増加が悪化と結びつく</b>（連戦・古馬の増加は、調整や余裕の持ちすぎを表すのかもしれません。これは推測で、データでは確かめていません）という点です。3 歳前半も正の向き（区間の下限はほぼ 0）。</p>")
    o.append(f"<p class='muted'>感度: 休養 7〜60 日に限ると {ci(A['sensitivity']['休養7〜60日'])}、7〜365 日で {ci(A['sensitivity']['休養7〜365日'])}、低信頼を含めて {ci(A['sensitivity']['低信頼を含む'])}。前走比の線形の結びつきは、条件の取り方で符号が入れ替わる程度の小ささです。</p>")

    # ── 3. B
    o.append("<h2>3. B 馬ごと: 結びつきの強さに個体差はあるか</h2>")
    o.append(f"<p>各馬の連続 2 走の組（{B['delta']['min_pairs']} 組以上 = {B['delta']['n_horses']:,} 頭）で、統制の残差どうしの傾き（指数の変化 ÷ 体重の変化、点/10kg）を出しました。走数ごとの頭数: " + "、".join(f"{k} 組 {v:,} 頭" for k, v in B["pairs_per_horse"].items()) + "。</p>")
    o.append(table(["指標", "値", "読み"], [
        ["傾きの母集団平均 μ", f(B["delta"]["mu"], 3), "全体では 0（A と整合）"],
        ["真の個体差 τ（経験ベイズ）", f(B["delta"]["tau"], 2), "点/10kg"],
        ["馬ごとの推定誤差（平均 SE）", f(B["delta"]["mean_se"], 2), "τ より大きい＝1 頭ごとの値はほぼ誤差"],
        ["縮約後に個体の値が残る割合（平均）", f(B["delta"]["mean_weight"], 2), "τ²/(τ²+SE²)。0 に近いほど母集団平均へ寄る"],
        ["再現性: 前半の走 → 後半の走（順位相関）", f"{f(d_rel['firstsecond']['spearman'], 3)}（z={f(d_rel['firstsecond']['z'], 1)}）", f"{d_rel['firstsecond']['n_horses']:,} 頭。帰無（体重の順序を入れ替え）の標準偏差 {f(d_rel['firstsecond']['null_sd'], 3)}"],
        ["再現性: 奇数番目の走 ↔ 偶数番目の走", f"{f(d_rel['oddeven']['spearman'], 3)}（z={f(d_rel['oddeven']['z'], 1)}）", "弱く検出。隣り合う走の共通の好不調"],
    ], num_cols=(1,)))
    o.append(f"<p>水準どうしの版（母集団の段階×性の平均を引いた体重と指数の、馬ごとの傾き。{B['level']['min_runs']} 走以上 {B['level']['n_horses']:,} 頭）でも同じで、奇数/偶数 {f(l_rel['oddeven']['spearman'], 3)}（z={f(l_rel['oddeven']['z'], 1)}）、前半/後半 {f(l_rel['firstsecond']['spearman'], 3)}（z={f(l_rel['firstsecond']['z'], 1)}）です。</p>")
    o.append("<div class='note'><b>結論（B）</b>: 馬ごとの『体重と指数の結びつき』は、キャリアの前半で測っても後半の予測には使えません。<b>『この馬は体重が増えると良くなるタイプ』という個体の指標は、現在の走数では出せません。</b>走数を絞って（前半・後半に 4 組ずつ以上 = 8 組以上の馬で）測っても再現しないので、可視化するならば、馬の体重と指数の推移を同じグラフに重ねて見せるだけにして、数値化した『相性』は出さないのが誠実です（成長曲線に体重を重ねる UI は別途実装）。</div>")

    # ── 4. C
    o.append("<h2>4. C 血統: 成長時期の特徴（晩成度・体重の伸びやすさ）</h2>")
    o.append(f"<p>馬ごとに『2〜3 歳の平均 → 4 歳以上の平均』の指数の伸び（<b>晩成度</b>、点）と体重の伸び（kg）を出し、性別の平均を引き、種牡馬（父）・母父ごとに経験ベイズで縮約しました（産駒 2 走以上ずつ）。"
             f"<b>学習と評価は別の馬</b>: 学習は 2017〜2019 年生まれ（{C['n_train_horses']['idx_metric']:,} 頭・{C['n_train_horses']['wt_metric']:,} 頭、〜2023 年の走だけ）、評価は 2020〜2021 年生まれ（{C['n_test_horses']['idx_metric']:,} 頭・{C['n_test_horses']['wt_metric']:,} 頭、〜2025 年）。産駒 10 頭以上の種牡馬だけを使います。</p>")
    rows = []
    for name, k in (("父: 晩成度 → 産駒の晩成度", "sire:late_minus_early_idx->late_minus_early_idx:min10"), ("父: 体重の伸び → 産駒の体重の伸び", "sire:late_minus_early_wt->late_minus_early_wt:min10"),
                    ("父: 体重の伸び → 産駒の晩成度", "sire:late_minus_early_wt->late_minus_early_idx:min10"), ("父: 晩成度 → 産駒の体重の伸び", "sire:late_minus_early_idx->late_minus_early_wt:min10"),
                    ("母父: 晩成度 → 産駒の晩成度", "dam_sire:late_minus_early_idx->late_minus_early_idx:min10"), ("母父: 体重の伸び → 産駒の体重の伸び", "dam_sire:late_minus_early_wt->late_minus_early_wt:min10"),
                    ("母父: 体重の伸び → 産駒の晩成度", "dam_sire:late_minus_early_wt->late_minus_early_idx:min10")):
        v = ev[k]
        hr, tb = v.get("horse_regression"), v.get("top_vs_bottom20")
        rows.append([name, f"{v['n_sires_eval']}", f"{v['n_test_horses']:,}", f"{f(v.get('spearman_shrunk'), 2) if v.get('spearman_shrunk') is not None else '—（個体差0）'}", f(v.get("spearman_raw"), 2),
                     ci(hr) if hr and v.get("tau2_train") else "—", ci(tb, 1) if tb and v.get("tau2_train") else "—"])
    o.append(table(["組み合わせ（学習 → 評価）", "評価の種牡馬数", "評価の馬数", "順位相関（縮約後）", "順位相関（縮約前）", "馬単位の回帰の傾き", "上位20% − 下位20%"], rows, num_cols=(1, 2, 3, 4)))
    o.append("<p class='muted'>順位相関は、種牡馬ごとの『学習側の値』と『評価側の産駒の平均』（評価の産駒 5 頭以上の種牡馬）の間。回帰の傾きは、評価側の馬の値を、その父の学習側の縮約値に回帰（種牡馬クラスタの頑健 SE）。上位/下位は学習側の値で分けた産駒の平均の差（種牡馬単位のブートストラップ 300 回の 95% 区間）。個体差ゼロの組は『—』。</p>")
    o.append("<h3>4.1 種牡馬の成長プロファイル（2017〜2021 年生まれ・〜2025 年の全データ、産駒の指数あり 20 頭以上）</h3>")
    sp = sire[sire["n_idx"].fillna(0) >= 20].copy()
    top, bot = sp.sort_values("banse_shrunk", ascending=False).head(6), sp.sort_values("banse_shrunk").head(6)
    cols = lambda d: [[short_name(r["name"]), f"{int(r['n_idx'])}", f(r["banse_raw"], 2, True), f(r["banse_shrunk"], 2, True), f(r["wt_gain_shrunk"], 2, True)] for _, r in d.iterrows()]
    o.append("<h3>晩成型（4 歳以上の指数の伸びが大きい）</h3>" + table(["種牡馬", "産駒数", "晩成度 生（点）", "晩成度 縮約後（点）", "体重の伸び 縮約後（kg）"], cols(top), num_cols=(1, 2, 3, 4)))
    o.append("<h3>早熟型</h3>" + table(["種牡馬", "産駒数", "晩成度 生（点）", "晩成度 縮約後（点）", "体重の伸び 縮約後（kg）"], cols(bot), num_cols=(1, 2, 3, 4)))
    o.append(f"<div class='note'><b>結論（C）</b>: 種牡馬の『体重の伸び』は産駒で再現し（父・母父とも上位 − 下位が有意）、血統分析の指標として出せます。『晩成度』は父では弱く再現しますが、上位 − 下位の差は有意とは言えません。"
             f"体重の伸びと指数の伸びが馬のレベルで結びついていない（相関 {f(C['horse_level_corr_train']['pearson'], 2)}）ため、『体重が増えやすい血統＝成長して強くなる血統』とは言えません。種牡馬ごとの成長プロファイル（段階間の指数・体重の伸びを含む）は <code>sire_profile.csv</code> / <code>dam_sire_profile.csv</code> と、サーバ用の <code>config/growth_index/ped_profile.json</code> にあります。</div>")

    # ── 5. E
    o.append("<h2>5. E 成長指数（馬体重発表 T-60 時点）</h2>")
    o.append(f"<h3>5.1 目的変数と、検出できる限界</h3><p>目的変数は『今走の指数 − 直近水準』（直近水準＝過去 5 走の指数を新しい順に 0.7<sup>k</sup> で重み付けした平均。前走が 7 日以内なら前走を除く＝指数の確定が遅れるため）。"
             f"ここから、統制（直近水準・走数・休養・距離・面・性）で説明できる分を引いた残差を『期待より上がったか』として評価します。低信頼・障害は水準・目的変数から除外。"
             f"1 走の指数のノイズは、短い休養・同じ面・同じ距離帯の連続 2 走の差から σ={f(nz['sigma_e'], 2)} 点（約 {f(nz['sigma_e'] / 10, 2)} 秒、{nz['pairs']:,} 組）と見積もりました。残差の分散 {f(nz['var_target_now'], 1)} のうちノイズが {f(nz['noise_var_now'], 1)}（{f(nz['noise_var_now'] / nz['var_target_now'] * 100, 0)}%）を占め、"
             f"<b>どんなに良い入力でも相関は {f(nz['max_corr_now'], 2)} を超えられません</b>（次の 3 走の平均にすると {f(nz['max_corr_fut'], 2)}）。この上限に対する割合で評価します。検出できる最小の相関は {f(nz['min_detectable_corr_now'], 3)}（n={nz['n_now']:,}）。</p>")
    o.append("<h3>5.2 候補の比較（学習 2020〜2023 年 → 評価 2024〜2025 年）</h3>")
    rows = []
    for k, c in (("cand4_adopted", e4), ("cand1_lgb", e1), ("cand2_additive", e2)):
        rows.append([c["name"], f(c["pearson_now"], 3), f(c["spearman_now"], 3), f(c["pearson_fut"], 3), ci(c["top_minus_bottom_now"], 1), ci(c["top_minus_bottom_fut"], 1), f"{f(c['pearson_now'] / nz['max_corr_now'] * 100, 0)}%"])
    o.append(table(["定義", "相関（今走）", "順位相関", "相関（次の3走平均）", "上位 − 下位（今走・点）", "上位 − 下位（次の3走・点）", "上限に対する割合"], rows, num_cols=(1, 2, 3, 6)))
    o.append(f"<p>年別（今走の相関）: " + "、".join(f"{y}年 採用 {f(v['cand4_pearson_now'], 3)}・候補1 {f(v['cand1_pearson_now'], 3)}・候補2 {f(v['cand2_pearson_now'], 3)}（{v['n']:,} 走）" for y, v in E['by_year'].items()) + "。前走の水準の三分位ごとの相関（採用）: "
             + "、".join(f"{t['tercile']}: {f(t['pearson_now'], 3)}" for t in e4["by_level_tercile"]) + "（水準が低い馬ほど効く＝平均への回帰を除いても残る）。</p>")
    o.append(f"<p><b>候補1</b>＝『統制 + 全入力』を入れた大きな LightGBM の予測から、統制だけの予測を引いたもの（精度の目安）。<b>候補2</b>＝4 要素（成長期・体重・追い切り・血統）の寄与をリッジ回帰で足し合わせた簡易版（解釈は容易だが、追い切りの効き方が非線形のため精度は約半分）。"
             f"<b>採用（候補4）</b>＝統制を馬単位の交差当てはめで引いた残差を、統制 + 要素の入力を入れた<b>小さな木（300 本 × 葉 7）</b>で予測。精度は候補1にほぼ並び、サーバでは LightGBM の model.txt 1 つで動き、要素ごとの寄与は SHAP を要素ごとに足して出せます。</p>")
    o.append("<h3>5.3 要素の寄与</h3>")
    o.append(bars_svg([("追い切り・厩舎コメント", only["oikiri"], "（単独）"), ("成長期（月齢段階）", only["stage"], "（単独）"), ("体重", only["weight"], "（単独）"), ("血統", only["ped"], "（単独）")], maxv=0.25))
    o.append(table(["要素", "単独の相関", "全体から抜いたときの相関", "寄与の標準偏差（点）", "入力"], [
        ["追い切り・厩舎コメント", f(only["oikiri"], 3), f(logo["oikiri"], 3), f(comp["oikiri"]["sd"], 2), "TL_A の自己比（f_t4z_vs_self など）、追い切り評価、厩舎コメントの肯定/否定語数"],
        ["成長期（月齢段階）", f(only["stage"], 3), f(logo["stage"], 3), f(comp["stage"]["sd"], 2), "2歳前半〜5歳以上"],
        ["体重", f(only["weight"], 3), f(logo["weight"], 3), f(comp["weight"]["sd"], 2), "前走比、自己平均からのずれ（若馬・休養明けとの交互作用）"],
        ["血統", f(only["ped"], 3), f(logo["ped"], 3), f(comp["ped"]["sd"], 2), "父・母父の晩成度、体重の伸び、成長段階間の指数/体重の伸び（その年より前のデータで縮約）"],
    ], num_cols=(1, 2, 3)))
    o.append(f"<div class='note'><b>正直な読み</b>: 今回の『成長指数』の中身は、成長というよりも<b>仕上がり（追い切りと厩舎コメント）</b>と成長期です。体重と血統は、単独では弱く（{f(only['weight'], 3)}・{f(only['ped'], 3)}）、追い切りと成長期があると<b>上乗せはほぼゼロ</b>（抜いても {f(logo['weight'], 3)}・{f(logo['ped'], 3)}）でした。"
             f"ユーザーが想定した『体重が成長期の月齢で増え、過去の推移から外れず、追い切りが良化し、血統的に増えやすい時期』が揃うとき、という組み立ては、体重・血統の要素が次走の指数の上がりにほとんど効かないため、現状のデータでは支持されません。UI では『成長度』の名前を使うなら、理由の主役が追い切りになる旨を正直に出すか、名称を『上昇度（次走で期待より上がりそうか）』にするのが妥当です。"
             f"データの充足: 追い切りあり {f(E['coverage_test']['has_training'] * 100, 0)}%、追い切り評価 {f(E['coverage_test']['oikiri_eval'] * 100, 0)}%、厩舎コメント {f(E['coverage_test']['comment'] * 100, 0)}%、父の血統プロファイル {f(E['coverage_test']['ped_sire'] * 100, 0)}%。</div>")
    o.append("<h3>5.4 5 段階と、その実績（評価期間 2024〜2025 年）</h3>")
    o.append(f"<p>境目は評価期間の指数の分布の 10/30/70/90 パーセンタイル（{', '.join(f(t, 2, True) for t in E['thresholds'])} 点）で、割合は 10/20/40/20/10%。『期待より上がった割合』は統制だけで予想した値より指数が高かった走の割合（50% が中立）。『実際に上がった割合』は直近水準より指数が高かった走の割合（平均への回帰を含む素の値）。</p>")
    o.append(table(["段階", "割合", "走数", "期待より上がった割合", "期待との差（点）", "実際に上がった割合", "平均の上がり幅（点）", "次の3走平均の上がり幅（点）"], [
        [r["label"], f"{r['share'] * 100:.0f}%", f"{r['n']:,}", f"{r['share_resid_up'] * 100:.1f}%（{r['share_resid_up_lo'] * 100:.1f}〜{r['share_resid_up_hi'] * 100:.1f}）", ci({'mean': r['mean_resid'], 'lo': r['mean_resid_lo'], 'hi': r['mean_resid_hi']}, 1),
         f"{r['share_up'] * 100:.1f}%", f(r["mean_y_now"], 1, True), f(r["mean_y_fut"], 1, True)] for r in lv], num_cols=(1, 2, 3, 4, 5, 6, 7)))
    o.append(ci_svg([(r["label"], r["n"], r["mean_resid"], r["mean_resid_lo"], r["mean_resid_hi"]) for r in lv], "期待との差（指数ポイント）"))
    o.append(f"<p>段階が上がるほど、実際に指数が上がった割合も平均の上がり幅も単調に増えます（{lv[0]['label']} {lv[0]['share_up'] * 100:.1f}% → {lv[-1]['label']} {lv[-1]['share_up'] * 100:.1f}%）。ただし 1 走の指数のノイズが大きいため、<b>『大きく成長』でも 4 頭に 1 頭は下がります</b>。UI では『期待より +x 点（= x/10 秒）』の目安と、この実績の表を併記してください。</p>")
    o.append("<h3>5.5 UI・サーバ用の出力</h3>")
    o.append("<ul><li>純粋関数: <code>src/pipeline/inference/growth_index.py</code> の <code>growth_index_model(x, params, booster)</code>（採用版）と <code>growth_index(x, params)</code>（lightgbm なしの簡易版）。返り値: <code>index</code>（点）、<code>level</code>（0〜4）、<code>label</code>（下降〜大きく成長）、<code>contributions</code>（stage/weight/oikiri/ped/ctrl の点）、<code>reasons</code>（寄与の大きい順に最大 3 件の短文）、<code>coverage</code>（体重・追い切り・血統の入力があったか）。</li>"
             "<li>必要ファイル: <code>config/growth_index/params.json</code>（列順・閾値・統制の平均・5 段階の実績表）、<code>model.txt</code>（LightGBM、約 250KB）、<code>ped_profile.json</code>（父・母父の成長プロファイル。2025 年末まで）。</li>"
             "<li>入力 <code>x</code>: <code>stage</code>（age=レース年−馬 ID 先頭 4 桁、月 → 段階）、<code>sex</code>、<code>surface</code>、<code>distance</code>、<code>rest_days</code>、<code>weight_change</code>（発表の前走比）、<code>weight_dev</code>（今回体重 − その馬の過去平均体重。過去体重なしは None）、"
             "<code>level</code>（過去 5 走の指数の 0.7<sup>k</sup> 加重平均。有効な過去指数が 2 つ未満の馬は指数を出さない）、<code>n_valid_prior</code>、<code>stale_prev</code>（前走が 7 日以内、または前走の指数が未確定なら 1。このとき level は前走を除いて作る）、"
             "追い切り系（<code>f_t4z_vs_self</code> ほか TL_A、<code>oikiri_eval</code>、<code>comment_pos/neg</code>、<code>has_training</code>）、<code>ped</code>（<code>ped_inputs</code> で ped_profile.json から作る）。</li>"
             "<li>欠損の扱い: 追い切り・コメント・血統・体重の過去平均は欠けて構いません（欠損のまま木に渡す。学習時に欠損を含めて学習済み。追い切りが無い馬は寄与が小さくなるだけ）。<code>level</code> が作れない馬（出走が少ない）は指数を出さず『判定なし』にしてください。</li>"
             "<li>先読み: 入力はすべて T-60 までに分かるもの。血統プロファイルは『その年より前』のデータで作ったものを評価に使い、本番用の ped_profile.json は 2025 年末までのデータ。</li></ul>")
    o.append("<h3>5.6 モデルの特徴量にしたときの影響（V4 に足した勝率 LogLoss）</h3>")
    if V1 and V1.get("groups"):
        base = V1["base"]["levels"]["logloss_win"]
        rows = []
        for g, s in V1["groups"].items():
            c = s["vs_V4"]
            rows.append([g, f"{s['cols']}", f(s["levels"]["logloss_win"], 5), f(c["Δwin"], 5, True), f(c["win_hi"], 5, True), f(c["Δtop2"], 5, True), f(c["Δtop3"], 5, True), str(c.get("採用"))])
        o.append(f"<p>既存の特徴量検証と同じ手順（tp-35 と同じ第 1 段: valid 2022+2023・seed 0、V4 全体に対する増分。V4 の勝率 LogLoss {f(base, 5)}）。指数は各年『その年より前』で学習（2020〜2021 年は欠損）。</p>")
        o.append(table(["群", "列数", "勝率 LogLoss", "Δ勝率", "Δ勝率の上限", "Δ連対", "Δ複勝", "採用"], rows, num_cols=(1, 2, 3, 4, 5, 6)))
    else:
        o.append("<p class='muted'>V4 への追加評価は実行中または未実施です（結果ファイルなし）。</p>")
    if V2 and V2.get("groups"):
        rows = []
        for g, s in V2["groups"].items():
            c = s["vs_V4"]
            rows.append([g, f(s["levels"]["logloss_win"], 5), f(c["Δwin"], 5, True), f(c["win_hi"], 5, True), str(c.get("採用"))])
        o.append("<p>第 2 段（valid 2024・seed 2/3）:</p>" + table(["群", "勝率 LogLoss", "Δ勝率", "Δ勝率の上限", "採用"], rows, num_cols=(1, 2, 3)))
        if V1 and V1.get("groups"):
            p1 = {g: s["vs_V4"] for g, s in V1["groups"].items()}
            g2 = list(V2["groups"])[0]
            c2 = V2["groups"][g2]["vs_V4"]
            c1 = p1.get(g2)
            o.append(f"<div class='note'><b>結論（特徴量化）: 見送り。</b>{g2}（4 要素の寄与）は第 1 段（valid 2022+2023）で Δ勝率 {f(c1['Δwin'], 5, True)}（上限 {f(c1['win_hi'], 5, True)}）と基準（Δ&lt;−0.00012 かつ上限&lt;0.0001）を通りましたが、"
                     f"第 2 段（valid 2024）では {f(c2['Δwin'], 5, True)}（上限 {f(c2['win_hi'], 5, True)}）と<b>逆向き</b>で、3 分割で一貫せず、最小効果量 0.0002 にも届きません。"
                     f"成長指数の合計（GI_idx）は第 1 段で {f(p1['GI_idx']['Δwin'], 5, True)}（上限 {f(p1['GI_idx']['win_hi'], 5, True)}）で通過しませんでした。"
                     "既存の V4 は追い切り・体重・月齢段階などの入力をすでに持っており、成長指数の中身（主役は追い切りと成長期）は新しい情報をほとんど足しません。指数の価値は特徴量よりも、UI の説明用の要約にあります。</div>")
    o.append("<h2>6. 限界と未解決</h2><ul>"
             "<li>月齢は馬 ID 先頭 4 桁の生年と開催月から作った段階（生まれ月は不明）。成長期の区切りは粗い。</li>"
             f"<li>1 走のめぐ指数のノイズが大きく（σ≈{f(nz['sigma_e'], 1)} 点）、個体ごと・走ごとの『成長』は検出の限界にあります。集計（段階・五分位）では有意でも、1 頭 1 走では 4 頭に 1 頭は逆に動きます。</li>"
             "<li>追い切りの要素は『仕上がり』を測っており、成長そのものではありません。厩舎コメントの充足は約 3 割です。</li>"
             "<li>血統プロファイルの学習側は 2017〜2019 年生まれの産駒に偏り、4 歳以上まで走った馬だけ（生存者バイアス）。</li>"
             "<li>V4 への追加評価の指数は、各年『その年より前』のデータで学習した値（expanding）で、2020〜2021 年は欠損です。</li></ul>")
    o.append("<h2>7. 成果物</h2><ul>"
             "<li>スクリプト: <code>src/research/pedigree/growth_weight_{stats,dataset,analysis}.py</code>（A〜C）、<code>growth_index_{dataset,fit,explore}.py</code>（E）、<code>growth_weight_report.py</code>（この章）、<code>src/scripts/data/growth_index_v4eval.py</code>（V4 追加評価）、<code>src/pipeline/inference/growth_index.py</code>（純粋関数）。テスト: <code>tests/research/test_growth_weight_stats.py</code>、<code>tests/pipeline/test_growth_index.py</code>。</li>"
             "<li>結果（ホスト <code>data/analysis/pedigree/growth_weight/</code>）: <code>runs.parquet</code>、<code>summary.json</code>、<code>horse_link.csv</code>、<code>sire_profile.csv</code>、<code>dam_sire_profile.csv</code>、<code>growth_index_eval.json</code>、<code>gi_rows.parquet</code>、<code>gi_group.parquet</code>、<code>params.json</code>、<code>model.txt</code>、<code>ped_profile.json</code>。</li></ul>")
    return "<!doctype html><html lang='ja'><head><meta charset='utf-8'><title>体重の増減と指数の結びつき・成長時期の血統・成長指数（tp-57）</title><style>" + CSS + "</style></head><body>" + "\n".join(o) + "</body></html>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).write_text(build(Path(a.res)), encoding="utf-8")


if __name__ == "__main__":
    main()
