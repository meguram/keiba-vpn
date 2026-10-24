"""tp-74 の結果（results.json / cells.json / aux.json / prereg.json / fig*.png）から章 HTML を作る。

python -m src.scripts.data.tp74_report --in DIR --out docs/html/modeling/chapters/position_pace_course.html
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

BAND = ["逃げ", "先行", "差し", "追込"]
PACE3 = ["スロー", "ミドル", "ハイ"]
PP3 = ["スロー寄り", "中間", "ハイ寄り"]
G3 = ["良", "稍重", "重・不良"]
CHG = ["普段より前へ（−0.10 以上）", "普段どおり", "普段より後ろへ（+0.10 以上）"]


def esc(s) -> str:
    return html.escape(str(s))


def table(head: list[str], rows: list[list[str]], num_from: int = 1) -> str:
    h = "".join("<th" + (' class="n"' if i >= num_from else '') + f">{esc(c)}</th>" for i, c in enumerate(head))
    b = "".join("<tr>" + "".join(f"<td{' class=\"n\"' if i >= num_from else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{b}</table>"


def png_b64(path: Path) -> str:
    from PIL import Image

    im = Image.open(path).convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=128)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def pt(x, nd=2, sign=True) -> str:
    return "—" if x is None else (f"{x * 100:+.{nd}f}" if sign else f"{x * 100:.{nd}f}")


def e4(x, nd=2) -> str:
    return "—" if x is None else f"{x * 1e4:+.{nd}f}"


def cellmap(cells: list[dict], spec: str, metric: str, period: str = "disc") -> dict:
    out = {}
    for c in cells:
        if c["spec"] == spec and f"{period}_{metric}" in c:
            out[tuple(c["key"])] = c[f"{period}_{metric}"]
    return out


def cross_table(cells, spec, metric, rlabels, clabels, scale=100.0, base=0.0, nd=1, period="disc") -> str:
    d = cellmap(cells, spec, metric, period)
    rows = []
    for i, rl in enumerate(rlabels):
        row = [esc(rl)]
        for j in range(len(clabels)):
            v = d.get((float(i), float(j)))
            row.append("—" if v is None else f"{(v[1] + base) * scale:.{nd}f}<span class='muted'>（{v[0]:,}）</span>")
        rows.append(row)
    return table(["位置の帯＼" + "／".join([""])] + clabels, rows)


def build(src: Path) -> str:
    R = json.loads((src / "results.json").read_text(encoding="utf-8"))
    C = json.loads((src / "cells.json").read_text(encoding="utf-8"))
    A = json.loads((src / "aux.json").read_text(encoding="utf-8"))
    P = json.loads((src / "prereg.json").read_text(encoding="utf-8"))
    img = {k: png_b64(src / f"fig{k}.png") for k in range(1, 6)}
    fam = {}
    for c in C:
        fam.setdefault(c["family"], []).append(c)

    def fam_stats(f):
        xs = [x for x in fam[f] if "disc_r3" in x]
        n = len(xs)
        z196 = sum(1 for x in xs if abs(x["disc_r3"][1] / x["disc_r3"][2]) > 1.96)
        return {"n": n, "bh_r3": sum(1 for x in xs if x.get("bh_r3")), "loyo": sum(1 for x in xs if x.get("bh_r3") and x.get("loyo_same_sign")),
                "z196": z196, "chance": 0.05 * n, "bh_roi": sum(1 for x in xs if x.get("bh_roi"))}

    FS = {f: fam_stats(f) for f in fam}
    pcells = [x for x in fam["P"] if "disc_rp" in x]
    best_roi = sorted(pcells, key=lambda x: -(x["disc_rp"][1] - 1.96 * x["disc_rp"][2]))[:6]
    best_z = sorted([x for x in fam["P"] if "disc_r3" in x], key=lambda x: -abs(x["disc_r3"][1] / x["disc_r3"][2]))[:6]
    lad = R["ladder"]

    def ladder_rows(which, key):
        rows = []
        for x in lad[which]:
            row = [esc(x["label"])]
            for y in ("y2024", "y2025", "y2026"):
                a = x[key].get(y)
                row.append("—" if not a else f"{a['d_logloss'] * 1e4:+.2f}<span class='muted'>（±{1.96 * a['se'] * 1e4:.2f}）</span>")
            rows.append(row)
        return rows

    def het_rows(which):
        rows = []
        for h in R[which]:
            g = h["gaps"]
            t = [v["t3_gap"] for v in g.values()]
            q = [v["r3_gap"] for v in g.values()]
            wt, wr = h.get("wald_t3", {}), h.get("wald_r3", {})
            rows.append([esc(h["label"]), str(h["levels"]), f"{min(t) * 100:.1f}〜{max(t) * 100:.1f}" if t else "—", f"{wt.get('p', float('nan')):.3g}" if "p" in wt else "—",
                         f"{min(q) * 100:+.1f}〜{max(q) * 100:+.1f}" if q else "—", f"{wr.get('p', float('nan')):.3g}" if "p" in wr else "—"])
        return rows

    # 位置の帯の全体表
    def band_rows(key):
        rows = []
        for x in R["bands"][key]:
            b = int(next(v for k, v in x.items() if k in ("bp", "ba1", "ba4")))
            rows.append([BAND[b], f"{x['n']:,}", f"{x['t3'] * 100:.1f}", f"{x['p'] * 100:.1f}", f"{(x['t3'] - x['p']) * 100:+.2f}", f"{x['rp']:.1f}", f"{x['rwin']:.1f}"])
        return rows

    bh = ["位置の帯", "出走数", "3着以内率(%)", "現行モデルの予測(%)", "差(pt)", "複勝回収率(%)", "単勝回収率(%)"]
    cov = R["coverage"]

    # 型の検定の一覧
    frows = []
    names = {"P": "P：事前に使える形（位置取り予想の帯 × 条件）", "O1": "O1：実際の初角の帯 × 条件（オラクル）", "O4": "O4：実際の4角の帯 × 条件（オラクル）", "M": "M：今走に前へ行けたか（結果由来）"}
    for f in ("P", "O1", "O4", "M"):
        s = FS[f]
        frows.append([esc(names[f]), f"{s['n']}", f"{s['z196']}（偶然の期待 {s['chance']:.0f}）", f"{s['bh_r3']}", f"{s['loyo']}", f"{s['bh_roi']}"])

    def cellname(x):
        return esc(x["spec"] + " " + str(x["key"]))

    roi_rows = []
    for x in best_roi:
        c25, c26 = x.get("conf2025_rp"), x.get("d2026_rp")
        roi_rows.append([cellname(x), f"{x['disc_rp'][0]:,}", f"{100 + x['disc_rp'][1]:.1f}", f"{100 + x['disc_rp'][1] - 1.96 * x['disc_rp'][2]:.1f}", f"{100 + c25[1]:.1f}" if c25 else "—", f"{100 + c26[1]:.1f}" if c26 else "—"])
    z_rows = []
    for x in best_z:
        c25 = x.get("conf2025_r3")
        z_rows.append([cellname(x), f"{x['disc_r3'][0]:,}", pt(x["disc_r3"][1]), f"{x['disc_r3'][1] / x['disc_r3'][2]:+.2f}", pt(c25[1]) if c25 else "—", "✔" if x.get("loyo_same_sign") else "×"])
    # M の結果由来
    m_rows = []
    for a in A["chg3_resid"]:
        m_rows.append([CHG[int(a["chg3"])], f"{a['n']:,.0f}", f"{a['t3'] * 100:.1f}", f"{a['p'] * 100:.1f}", pt(a["r3"])])
    auc_rows = [[esc(k), f"{v['auc_forward']:.3f}", f"{v['auc_back']:.3f}", f"{v['n']:,}"] for k, v in A["move_auc_2025"].items()]
    mroi = sorted([x for x in fam["M"] if x.get("bh_roi")], key=lambda x: -(x["disc_rp"][1] - 1.96 * x["disc_rp"][2]))[:4]
    mroi_rows = [[cellname(x), f"{x['disc_rp'][0]:,}", f"{100 + x['disc_rp'][1]:.1f}", f"{100 + x['disc_rp'][1] - 1.96 * x['disc_rp'][2]:.1f}", f"{100 + x['conf2025_rp'][1]:.1f}" if "conf2025_rp" in x else "—"] for x in mroi]

    pre_rows = [[esc(k), str(v)] for k, v in P["n_cells_by_family"].items()]

    existing = table(
        ["章", "位置取り", "コース", "馬場", "ペース", "見たもの", "4 要素の同時"],
        [
            ["<a href='pace_type.html'>展開タイプ（tp-61）</a>", "脚質（逃げ〜追込）", "—（全コース合算）", "—", "実測のペース 3 × 隊列 3（事前確率も作成）", "3着以内率の比、馬単位の LogLoss", "位置×ペース×隊列の 3 軸。コース・馬場なし"],
            ["<a href='pace_benefit.html'>展開利（tp-64）</a>", "脚質 × ペース × 隊列の 9 セル", "—", "—", "実測", "過去走の展開利と V5 の残差", "なし（過去走側の指標）"],
            ["<a href='pace_consistency.html'>位置取り・追走難度・上がりの連動（tp-53）</a>", "想定位置取り", "—", "—", "ペースの定義の食い違い", "連動の有無", "好走率は見ていない"],
            ["位置取り予想（<code>position_forecast</code>、tp-25 系）", "馬柱から作る予想（枠補正）", "枠補正の係数だけコース別", "—", "逃げ・先行の頭数", "予想そのもの", "好走率との関係は見ていない"],
            ["<a href='tracking_difficulty_v2.html'>追走難度 v2（tp-54）</a>", "想定位置取り", "コース・距離", "馬場", "想定ペース", "<b>終いの落ち（上がり）</b>を予測", "好走率の条件別の変化は見ていない"],
            ["<a href='edge_factor_mining.html'>買う要素の網羅探索（tp-72）</a>", "脚質 × 枠", "一部（コース別の枠）", "クッション値・含水率・馬場", "隊列の頭数", "残差・回収率（73 型ほか）", "位置×馬場×ペースの 3 要素以上は未実施"],
            ["<a href='lap_forecast.html'>ラップ予想（tp-55）</a>", "—", "コース", "馬場", "ペース予想", "ラップの確率と時計", "好走率は見ていない"],
            ["<a href='pedigree_position_cross.html'>血統の位置・クロス（tp-38）</a>", "（血統表の中の位置。走る位置ではない）", "—", "—", "—", "別の概念", "対象外"],
            ["<b>本章（tp-74）</b>", "<b>事前の予想の帯 / 実際の初角・4 角の帯 / 今走に前へ行けたか</b>", "<b>面×距離帯、競馬場×面</b>", "<b>状態 3 段階＋クッション値・含水率の硬さ</b>", "<b>事前のペース予想（三分位）/ 実測</b>", "<b>3着以内率、現行モデルとの差、複勝・単勝回収率、標本外の LogLoss</b>", "<b>あり（1〜4 要素の段階で比較）</b>"],
        ], num_from=99)

    z25 = R["ladder"]["P"][5]["inc"]["y2025"]
    body = f"""
<h2>0. 先に答え</h2>
<ul>
<li><b>『条件によって好走率の条件が変わる』のは事実。ただし、事前に分かる範囲の変わり方は、現行モデルがすでに織り込んでいる。</b>事前の位置取り予想の帯（逃げ・先行 vs 差し・追込）の3着以内率の差は、コースで {R['het_P'][0]['gaps'] and min(v['t3_gap'] for v in R['het_P'][0]['gaps'].values()) * 100:.1f}〜{R['het_P'][0]['gaps'] and max(v['t3_gap'] for v in R['het_P'][0]['gaps'].values()) * 100:.1f}pt、硬さ・面×馬場×ペース予想でも有意に変わる（ワルド検定 p&lt;0.01）。しかし<b>現行モデル（V4 の OOF）との差（残差）はコース・馬場・ペース予想の水準でほとんど変わらない</b>（p=0.10〜0.71。馬場状態は p=0.71）。</li>
<li><b>型の検定: 事前に使える形（P）は {FS['P']['n']} セルで BH 通過 0、回収率&gt;100% の通過も 0。</b>z&gt;1.96 は {FS['P']['z196']} 本で偶然の期待（{FS['P']['chance']:.0f} 本）と同じ。複勝回収率の区間下限が最も高いセルでも {100 + best_roi[0]['disc_rp'][1] - 1.96 * best_roi[0]['disc_rp'][2]:.1f}%。凍結する規則はなし。</li>
<li><b>標本外（2025）で、条件を重ねて現行モデルに補正を足すと、逆に悪化する</b>（要素を増やすほど悪化: 位置の帯のみ {e4(lad['P'][0]['inc']['y2025']['d_logloss'])}→コース・馬場・ペース予想・硬さを重ねると {e4(z25['d_logloss'])} ×10⁻⁴）。条件による変化を取りに行っても、モデルへの上乗せはない。</li>
<li><b>オラクル（実際の位置・実測のペース）では、条件の効果は非常に大きい</b>（実際の初角の帯だけで 3着以内率 {R['bands']['O1_disc'][0]['t3'] * 100:.0f}% → {R['bands']['O1_disc'][3]['t3'] * 100:.0f}%、実測のペースで帯の差が 9〜25pt に変わる）が、これはレース後にしか分からない。<b>『今走に前へ行けたか』の影響は ±{abs(A['chg3_resid'][0]['r3']) * 100:.1f}pt</b>だが、事前の情報でそれを当てる力（AUC 0.62〜0.64）が弱く、予想には使えない。</li>
<li>結論: <b>『条件による変化はモデルと市場が織り込み済み』</b>。位置取り予想 × コース × 馬場 × ペース予想の組み合わせに、買い目や特徴量の上乗せは見つからなかった。</li>
</ul>

<h2>1. 既存の分析は、4 要素をどこまで同時に扱ったか</h2>
{existing}
<p class="muted">位置取り・コース・馬場・ペースの 4 つを同時に条件として、<b>好走率（3着以内率）と回収率</b>の変化を測った章はなかった（ペースと脚質の組は tp-61、馬場と脚質と枠の組は tp-72 まで）。本章がこの空白を埋める。</p>

<h2>2. 設計（結果を見る前に登録: コミット 0b7767b）</h2>
<ul>
<li><b>データ</b>: 2022〜2026 年の JRA 平地 {cov['rows']:,} 出走（現行モデルの確率が OOF である年）。発見 2022〜2024、確認 2025 を 1 回、2026 年は記述のみ（選択に使わない）。市場オッズ・人気は特徴量にしない（回収率の計算にだけ払戻を使う）。モデルは V4 の OOF（V5/v6e の馬ごとの予測は 2025 年以降に無い）。</li>
<li><b>位置取りの 2 つの形</b>: <b>P（事前に使える形）</b>=位置取り予想 <code>pf_early</code>（馬柱から作る、枠補正込み。逃げ ≤0.15 / 先行 ≤0.35 / 差し ≤0.65 / 追込）。取得できた割合は {cov['pf_cov'] * 100:.0f}%（2026 年は枠補正なしの過去 5 走平均で代用し、記述のみ）。<b>O（オラクル）</b>=今走の実際の初角・4 角の帯。レース後にしか分からない。</li>
<li><b>コース</b>: 面×距離帯（10）、競馬場×面（20）。<b>馬場</b>: 状態 3 段階（良／稍重／重・不良）、硬さ 3 段階（芝=クッション値、ダート=含水率。2020〜2021 年の三分位）。<b>ペース</b>: 事前=レース質の事前確率の P(ハイ)−P(スロー) の三分位（2021 年のレース単位の分位で固定。カバー {cov['pp3_cov'] * 100:.0f}%）、実測=ペース 3 × 隊列 3。</li>
<li><b>型</b>: 位置の帯 × 条件の水準の組で、発見期間の出走が 600 以上のもの。<b>{P['n_cells_total']} セル（検定 {P['n_tests_primary']} 本: 3着以内率の残差の両側検定と、複勝回収率&gt;100% の片側検定）</b>。BH（FDR 5%）は family ごと。内訳: {'、'.join(f'{k} {v}' for k, v in P['n_cells_by_family'].items())}。</li>
<li><b>交互作用</b>: 『逃げ・先行 − 差し・追込』の差が条件の水準で違うか（生の3着以内率と、現行モデルの残差の両方。レース単位のクラスタ頑健な線形モデルのワルド検定）。</li>
<li><b>標本外の評価</b>: セルの平均を階層縮約（位置の帯→+条件を 1 つずつ。k=200 出走）して、(a) 現行モデルの確率への補正、(b) 全体平均に対するセルだけの説明力、の 3着以内の LogLoss の変化を測る（2024=2022–23 で学習、2025=2022–24 で学習。負=改善）。</li>
</ul>

<h2>3. 条件によって好走率は変わるか（生の3着以内率）</h2>
<p class="muted">事前の位置取り予想の帯（P）。括弧内は出走数。発見期間（2022〜2024）。</p>
{table(bh, band_rows('P_disc'))}
<h3>3.1 位置の帯 × 事前のペース予想（3着以内率 %）</h3>
{cross_table(C, 'bp×pp3', 't3', BAND, PP3)}
<h3>3.2 位置の帯 × 馬場状態（3着以内率 %）</h3>
{cross_table(C, 'bp×g3', 't3', BAND, G3)}
<h3>3.3 帯の効きの違いはどの条件で変わるか（交互作用）</h3>
<p class="muted">『逃げ・先行 − 差し・追込』の3着以内率の差（pt）の範囲と、水準間で差が等しいというワルド検定の p 値。右 2 列は同じ差を<b>現行モデルの残差</b>で見たもの（0 に近く p が大きいほど、モデルが条件ごとの変化を織り込んでいる）。事前に使える形（P）。</p>
{table(['条件', '水準数', '3着以内率の差の範囲(pt)', 'p(生)', 'モデルとの差の範囲(pt)', 'p(残差)'], het_rows('het_P'))}
<img alt="コース別・実測ペース別" style="max-width:100%" src="data:image/png;base64,{img[2]}" />
<p class="muted">図 1: 事前の位置取り予想の帯別の3着以内率（コース別、実測のペース別の色）。白抜きは標本 300 出走未満。ペースが遅いほど前が有利になる。コースによって前後の差の大きさが違う（中山ダートで大きい）。</p>
<img alt="コース別・馬場別" style="max-width:100%" src="data:image/png;base64,{img[3]}" />
<p class="muted">図 2: 同じく馬場状態別。状態の違いによる帯の差の変化は小さい（馬場状態の交互作用 p=0.21）。</p>

<h2>4. 変わる分は織り込み済みか</h2>
<ul>
<li>交互作用を<b>残差</b>で見ると、コース（p=0.10）、競馬場×面（0.13）、馬場（0.71）、ペース予想（0.28）、硬さ（0.32）、面×馬場×ペース予想（0.48）のいずれも有意でない。<b>生の3着以内率が条件で変わる分は、現行モデルがほぼ織り込んでいる。</b></li>
<li>セル別の検定（事前に使える形 P）: {FS['P']['n']} セルで、BH 通過 {FS['P']['bh_r3']}、z&gt;1.96 は {FS['P']['z196']} 本（偶然の期待 {FS['P']['chance']:.0f} 本）。</li>
</ul>
{table(['family', 'セル数', 'z>1.96 の数', 'BH 通過（残差）', 'うち LOYO 同符号', 'BH 通過（回収率>100%）'], frows)}
<p class="muted">O1・O4・M は結果由来（実際の位置）なので、通過が多いのは当然。買い目の根拠にはならない（第 6 節）。</p>
<h3>4.1 事前に使える形（P）で残差の z が大きかったセル</h3>
{table(['セル（位置の帯×条件。水準は事前登録の順）', '出走数', '残差(pt)', 'z', '2025 の残差(pt)', 'LOYO 同符号'], z_rows)}
<p class="muted">上位でも z は 2.4〜2.7 で、2025 年に再現しない（符号が逆転するものもある）。</p>
<img alt="残差" style="max-width:100%" src="data:image/png;base64,{img[4]}" />
<p class="muted">図 3: 事前の位置取り予想の帯別の『実績 − 現行モデルの3着以内確率』（事前のペース予想別）。信頼区間は常に 0 を含み、ペース予想の違いで系統的にずれない。</p>
<img alt="年別" style="max-width:100%" src="data:image/png;base64,{img[5]}" />
<p class="muted">図 4: 前（逃げ・先行）と後ろ（差し・追込）の残差の年別推移。年ごとの符号は揃わない（2026 年は白抜き＝記述のみ）。</p>

<h2>5. 要素を増やしたときの標本外の効き（事前に使える形）</h2>
<p class="muted">3着以内の LogLoss の変化（×10⁻⁴、括弧は 95% 区間の半幅）。<b>負=改善</b>。『現行モデルへの補正』は、V4 の確率に条件別の補正を足した場合。『セルだけ』は、全体平均に対して条件別のセルだけで予測した場合（モデルは使わない）。</p>
<h3>5.1 現行モデルへの補正（P）</h3>
{table(['要素の数', '2024', '2025', '2026（記述）'], ladder_rows('P', 'inc'))}
<h3>5.2 セルだけ（P。全体平均に対する改善）</h3>
{table(['要素の数', '2024', '2025', '2026（記述）'], ladder_rows('P', 'raw'))}
<p class="muted">セルだけで見れば、位置の帯だけで {e4(lad['P'][0]['raw']['y2025']['d_logloss'])}、コースを足すと {e4(lad['P'][1]['raw']['y2025']['d_logloss'])}、コースとペース予想を足すと {e4(lad['P'][3]['raw']['y2025']['d_logloss'])}（×10⁻⁴）と、<b>条件を増やすほど好走率をよく説明する</b>（馬場の追加は変わらない）。しかし現行モデルの確率に足すと悪化する（5.1）。モデルはすでにコース・ペース予想・位置取りを入力にしており、ここで足した補正は雑音になる。</p>

<h2>6. オラクル（実際の位置・実測のペース）と『今走に前へ行けたか』</h2>
<p class="warn">この節の数字は<b>レース後に分かる情報</b>（今走の実際の通過順、実測のペース）を使う。買い目や予想には使えない。仕組みの確認用。</p>
<h3>6.1 実際の初角の帯と 3着以内率</h3>
{table(bh, band_rows('O1_disc'))}
{table(bh, band_rows('O4_disc'))}
<p class="muted">実際の4角の位置が前（逃げ・先行）だと3着以内率が 43%（複勝回収率 134%）、後ろ（追込）だと 6%（28%）。当然だが、これは予想できない。</p>
<img alt="実際の初角の帯" style="max-width:100%" src="data:image/png;base64,{img[1]}" />
<p class="muted">図 5: 実際の初角の帯別の3着以内率（コース別、実測のペース別の色）。事前の予想（図 1）よりも、帯の差とペースによる変化がはるかに大きい。</p>
<h3>6.2 条件の効き（オラクル）</h3>
{table(['条件', '水準数', '3着以内率の差の範囲(pt)', 'p(生)', 'モデルとの差の範囲(pt)', 'p(残差)'], het_rows('het_O'))}
<p class="muted">実測のペース・隊列では、帯の効きが大きく変わる（スローなら前が有利、ハイなら差し・追込が追い上げる）。コース別でも 10〜38pt と広い。</p>
<h3>6.3 今走に普段より前へ行けたか（結果由来）</h3>
{table(['今走の位置取り − 事前の予想', '出走数', '3着以内率(%)', '現行モデルの予測(%)', '差(pt)'], m_rows)}
<p class="muted">事前の予想より 0.10 以上前へ行けた馬は、現行モデルより +5.6pt、後ろへ下がった馬は −5.6pt。<b>『前へ行けるか』が成績を大きく左右する</b>が、事前の情報でそれを当てる力は弱い。</p>
{table(['事前に使う情報（2022–24 で作り 2025 で評価）', 'AUC（前へ行く）', 'AUC（後ろへ下がる）', '出走数'], auc_rows, num_from=1)}
<p class="muted">AUC 0.5=当たらない。位置の帯だけで 0.62、コース・ペース予想を足しても 0.64 までで、『誰が前へ行けるか』の事前予測は弱い。</p>
<h3>6.4 参考: M・O の回収率が 100% を超えるセル</h3>
{table(['セル（M family）', '出走数', '複勝回収率(%)', '区間下限(%)', '2025(%)'], mroi_rows)}
<p class="warn">これらは<b>今走の実際の位置取り（結果）で条件づけたもの</b>で、レース前に買う根拠にはならない（先読み）。BH を通るのはこのためで、事前に使える形（P）では 0 本。</p>

<h2>7. 回収率（事前に使える形）</h2>
<p class="muted">複勝回収率が高かったセル上位（発見期間の区間下限の順）。</p>
{table(['セル', '出走数', '回収率(%)', '区間下限(%)', '2025(%)', '2026(%)'], roi_rows)}
<ul>
<li>区間下限が 100% を超えるセルは 0。最も高い下限でも {100 + best_roi[0]['disc_rp'][1] - 1.96 * best_roi[0]['disc_rp'][2]:.1f}%。BH の通過は 0。</li>
<li><b>凍結する候補はなし</b>（tp-59 の手順で凍結する基準を満たすセルがない）。</li>
<li>位置の帯別の複勝回収率は 65〜76% で、全体（約 73%）と同水準（市場は前も後ろも同じように価格づけている）。</li>
</ul>

<h2>8. 特徴量化の見込み</h2>
<ul>
<li>現行モデルへの補正としては、条件を 1 つ足すごとに標本外で悪化するか変わらない（5.1）。<b>上乗せの見込みはない</b>（V5/V6 の入力にすでに位置取り予想・ペース・コース・馬場が含まれる）。</li>
<li>見込みがあるとすれば、『前へ行けるか』を当てる別の情報（枠、隊列の相手関係、騎手の積極性など）を作れた場合。AUC が 0.64 から上がらない限り、効果は出ない。今回は新しい情報を作っていない。</li>
</ul>

<h2>9. 結論と限界</h2>
<ul>
<li><b>結論</b>: 追走位置の効き方は、コース・ペース・硬さで確かに変わるが、<b>事前に分かる形（位置取り予想・ペース予想）に直すと、現行モデルがほぼ織り込んでおり、残差にも回収率にも優位は残らない</b>。実際の位置取りが分かればほぼ決まるが、それはレース後。『条件による変化はモデルと市場が織り込み済み』。</li>
<li><b>限界</b>: モデルは V4 の OOF（V5/v6e は 2025 年以降の馬ごとの予測が無い）。2026 年の位置取りの予想は枠補正なしの過去 5 走平均で代用（記述のみ）。内外回りの区別は距離で近似。事前のペース予想は tp-60/61 の事前確率で、ラップ予想（tp-55）の確率そのものではない。標本の少ないセルは検定から除外（発見期間 600 出走未満）。</li>
<li><b>事後の修正</b>: なし（定義と型の数は実行前に登録したまま。図の日本語フォントの設定だけ直した）。</li>
</ul>

<h2>10. 再現</h2>
<pre>ホストで（コード: tp74_pos.py、出力: /home/jovyan/work/tp74pos_out）
python tp74_pos.py prereg     # 型の定義と数（実行前に登録）
python tp74_pos.py eval       # 検定・交互作用・標本外
python tp74_pos.py figs       # 図
python -m src.scripts.data.tp74_report --in DIR --out docs/html/modeling/chapters/position_pace_course.html</pre>
"""
    return body


def page(body: str) -> str:
    head = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>追走位置 × コース × 馬場 × ペースと好走率（tp-74）— keiba-vpn</title>
<style>:root { --bg:#0f172a; --panel:#15213c; --border:#243358; --text:#e6ecff; --muted:#9aa7c7; --accent:#4f8bff; --accent-2:#7aa9ff; --warn:#fbbf24; --good:#34d399; --bad:#f87171; }
* { box-sizing: border-box; } body { font-family: "Noto Sans JP", system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.8; font-size: 15px; padding: 40px 56px 96px; max-width: 1250px; margin: 0 auto; }
a { color: var(--accent-2); } h1 { font-size: 26px; } h2 { font-size: 20px; margin: 40px 0 12px; border-left: 4px solid var(--accent); padding-left: 12px; } h3 { font-size: 16px; margin: 22px 0 8px; }
table { border-collapse: collapse; margin: 12px 0 18px; font-size: 12.5px; background: var(--panel); } th, td { border: 1px solid var(--border); padding: 4px 9px; text-align: left; vertical-align: top; } th { background: #1b2a4d; color: var(--muted); }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
.muted { color: var(--muted); font-size: 13px; } .warn { background: var(--panel); border: 1px solid var(--border); border-left: 4px solid var(--warn); border-radius: 8px; padding: 12px 16px; margin: 14px 0; }
code, pre { background: #0b1428; padding: 1px 5px; border-radius: 4px; } pre { padding: 10px 14px; overflow-x: auto; }</style></head>
<body><a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>追走位置 × コース × 馬場 × ペースと好走率（tp-74）</h1>
<p class="muted">2026-10-22、ユーザー依頼『追走位置とコース、馬場やペースによって好走率の条件が変わる気がする。分析はすでにされているか』。
資料: <a href="training_pipeline_decisions.html">決定ログ</a>（§65）・<a href="pace_type.html">展開タイプ（tp-61）</a>・<a href="pace_benefit.html">展開利（tp-64）</a>・<a href="edge_factor_mining.html">買う要素の網羅探索（tp-72）</a>・<a href="tracking_difficulty_v2.html">追走難度 v2（tp-54）</a>。
回収率は『1 出走（1 点）に 100 円』、区間はレース単位のクラスタ頑健（95%）。選択は 2022〜2024、2025 は確認 1 回、2026 年はテストセット②（選択に使わない。記述のみ）。</p>
"""
    return head + "{body}" + "\n</body></html>\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    body = build(Path(a.src))
    Path(a.out).write_text(page(body).replace("{body}", body), encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
