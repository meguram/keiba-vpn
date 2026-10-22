"""距離帯ごとの騎手依存度の章 docs/html/modeling/chapters/jockey_distance_dependence.html を、result.json から作る。

python -m src.scripts.data.jockey_distance_dep report   # JDD_OUT/result.json を読む（ホストで作った JSON を JDD_OUT に置いて実行）
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/html/modeling/chapters/jockey_distance_dependence.html"

CSS = """:root { --bg:#0f172a; --panel:#15213c; --border:#243358; --text:#e6ecff; --muted:#9aa7c7; --accent:#4f8bff; --accent-2:#7aa9ff; --warn:#fbbf24; --good:#34d399; --bad:#f87171; }
* { box-sizing: border-box; } body { font-family: "Noto Sans JP", system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.8; font-size: 15px; padding: 40px 56px 96px; max-width: 1150px; margin: 0 auto; }
a { color: var(--accent-2); } h1 { font-size: 26px; } h2 { font-size: 20px; margin: 40px 0 12px; border-left: 4px solid var(--accent); padding-left: 12px; } h3 { font-size: 16px; margin: 22px 0 8px; }
table { border-collapse: collapse; margin: 12px 0 18px; font-size: 13px; background: var(--panel); } th, td { border: 1px solid var(--border); padding: 5px 10px; text-align: left; vertical-align: top; } th { background: #1b2a4d; color: var(--muted); }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
.muted { color: var(--muted); font-size: 13px; } .note { background: var(--panel); border: 1px solid var(--border); border-left: 4px solid var(--warn); border-radius: 8px; padding: 12px 16px; margin: 14px 0; }
.ok { color: var(--good); } .ng { color: var(--bad); } code { background: #0b1428; padding: 1px 5px; border-radius: 4px; } img { max-width: 100%; background: #fff; border-radius: 6px; }"""

LBL = {"b1": "1000-1400", "b2": "1500-1800", "b3": "1900-2200", "b4": "2300-2800", "b5": "3000+", "b4p": "2300+"}
LBL_D = {"b4p": "1900+"}
BANDS = ["b1", "b2", "b3", "b4p"]


def e(x) -> str:
    return html.escape(str(x))


def lbl(sf: str, b: str) -> str:
    return (LBL_D.get(b) if sf == "ダ" else None) or LBL[b]


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%"


def pt(x: float, nd: int = 2) -> str:
    return f"{100 * x:+.{nd}f}"


def table(head: list[str], rows: list[list], num_from: int = 1) -> str:
    th = "".join(f'<th class="n">{e(h)}</th>' if i >= num_from else f"<th>{e(h)}</th>" for i, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f'<td class="n">{c}</td>' if i >= num_from else f"<td>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<table><tr>{th}</tr>{tr}</table>"


def b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()


def figs(res: dict) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out: dict[str, str] = {}
    col = {"disc": "#1f77b4", "2025": "#d62728", "2026": "#7f7f7f"}
    nm = {"disc": "2020-24 (discovery)", "2025": "2025 (confirm)", "2026": "2026 (descriptive)"}

    def band_plot(ax, getter, sf, title, ylab, periods=("disc", "2025", "2026")):
        xs = list(range(len(BANDS)))
        for k, per in enumerate(periods):
            ys, lo, hi, xx = [], [], [], []
            for i, b in enumerate(BANDS):
                v = getter(per, sf, b)
                if v is None:
                    continue
                xx.append(i + (k - 1) * 0.12); ys.append(v[0]); lo.append(v[0] - v[1][0]); hi.append(v[1][1] - v[0])
            if xx:
                ax.errorbar(xx, ys, yerr=[lo, hi], fmt="o-" if per == "disc" else "s", color=col[per], capsize=3, label=nm[per], alpha=0.9 if per == "disc" else 0.75, lw=1.4)
        ax.set_xticks(xs); ax.set_xticklabels([LBL_D.get(b, LBL[b]) if sf == "ダ" else LBL[b] for b in BANDS], fontsize=8)
        ax.set_title(f"{'Turf' if sf == '芝' else 'Dirt'}: {title}", fontsize=10); ax.set_ylabel(ylab, fontsize=9); ax.grid(alpha=0.3)

    def share_get(blk):
        def g(per, sf, b):
            c = res[blk][per]["cells"].get(f"{sf}:{b}")
            if not c or not c.get("reliable"):
                return None
            return 100 * c["share"], [100 * c["ci"][0], 100 * c["ci"][1]]
        return g

    for key, blk, title in (("share_z", "share_z", "jockey share of result variance (%)"), ("share_pos", "share_pos", "jockey share of first-corner position change (%)")):
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
        for ax, sf in zip(axs, ("芝", "ダ")):
            band_plot(ax, share_get(blk), sf, title, "%")
        axs[0].legend(fontsize=7)
        fig.tight_layout(); out[key] = b64(fig); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, sf in zip(axs, ("芝", "ダ")):
        def g(per, sf_, b, _sf=sf):
            c = res["switch"]["cells"].get(f"{per}|{_sf}:{b}")
            return None if not c else (c["beta"], c["ci"])
        band_plot(ax, g, sf, "z gain per 1SD better jockey after a switch", "beta")
    axs[0].legend(fontsize=7); fig.tight_layout(); out["switch"] = b64(fig); plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(10, 6.4))
    for j, sf in enumerate(("芝", "ダ")):
        def g(per, sf_, b, _sf=sf):
            c = res["elite"]["cells"].get(f"{per}|{_sf}:{b}")
            return None if not c else (100 * c["r_top3"]["diff"], [100 * c["r_top3"]["ci"][0], 100 * c["r_top3"]["ci"][1]])
        band_plot(axs[0, j], g, sf, "elite - others, top3 minus model (pt)", "pt")
        axs[0, j].axhline(0, color="k", lw=0.8)

        def g2(per, sf_, b, _sf=sf):
            c = res["elite"]["cells"].get(f"{per}|{_sf}:{b}")
            return None if not c else (c["roi_win"]["roi"], [c["roi_win"]["lo"], c["roi_win"]["hi"]])
        band_plot(axs[1, j], g2, sf, "elite jockeys: win ROI (%)", "%")
        axs[1, j].axhline(100, color="k", lw=0.8)
    axs[0, 0].legend(fontsize=7); fig.tight_layout(); out["elite"] = b64(fig); plt.close(fig)
    return out


def main(res: dict) -> None:
    im = figs(res)
    sz, sp, sw, cr, el, mk, fp, bh = res["share_z"], res["share_pos"], res["switch"], res["carry"], res["elite"], res["market_ctrl"], res["feature_potential"], res["bh_family"]
    d = sz["disc"]

    def cellrow(blk, sf):
        rows = []
        for b in BANDS:
            c = blk["disc"]["cells"].get(f"{sf}:{b}")
            if not c:
                continue
            r25, r26 = blk["2025"]["cells"].get(f"{sf}:{b}"), blk["2026"]["cells"].get(f"{sf}:{b}")
            f = lambda x: (f"{pct(x['share'])} [{pct(x['ci'][0])}, {pct(x['ci'][1])}]" if x and x.get("reliable") else "（標本不足）")
            rows.append([lbl(sf, b), f"{c['n']:,}", f"{c['n_jockeys']:.0f}", f(c), f(r25), f(r26)])
        return rows

    def trow(blk, sf):
        t = blk["disc"]["tests"]
        return t.get(f"{sf}:trend"), t.get(f"{sf}:b1-b4p")

    tz_t, tz_c = trow(sz, "芝"), None
    tzs = {sf: trow(sz, sf) for sf in ("芝", "ダ")}
    tps = {sf: trow(sp, sf) for sf in ("芝", "ダ")}
    loyo = res["share_z_loyo"]
    loyo_b1b3 = [v["trend"].get("芝:b1-b3") for v in loyo.values()]
    n_loyo_neg = sum(1 for x in loyo_b1b3 if x is not None and x < 0)
    holm_min = min(el["roi_holm"].values()) if el["roi_holm"] else float("nan")
    disc_el = [c for k, c in el["cells"].items() if k.startswith("disc|")]
    roi_all = [c[m]["roi"] for c in disc_el for m in ("roi_win", "roi_place")]
    roi_lo, roi_hi = min(roi_all), max(roi_all)
    n_roi_cells = len(disc_el)
    betas = [c["beta"] for k, c in sw["cells"].items() if k.startswith("disc|")]
    mk_t = {sf: [c["coef"] for k, c in mk["cells"].items() if k.startswith("disc|") and f"|{sf}:" in k and k.endswith("|top3")] for sf in ("芝", "ダ")}
    z_r = [c["r"] for k, c in cr["cells"].items() if k.startswith("z|disc|")]
    rs_r = [c["r"] for k, c in cr["cells"].items() if k.startswith("resid|disc|")]
    q_min = min(bh["q"].values()) if bh["q"] else float("nan")
    body: list[str] = []
    body.append(f"""<a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>距離の長さによって、騎手への依存度は変わるか</h1>
<p class="muted">2026-10-22、ユーザー質問『距離の長さによって、騎手の依存度が変わるかどうかも検証したい』への回答。
資料: <a href="jockey_switch_ev.html">騎手の乗り替わり（有力↔非有力）</a>・<a href="jockey_trainer_roi.html">騎手・調教師の分析と回収率（tp-59）</a>・<a href="equipment_and_change.html">馬具・条件替わり（tp-32、JOCKEY 群）</a>・<a href="edge_factor_mining.html">買う要素の網羅探索（tp-72）</a>。
対象は平地（障害を除く）。発見は 2020〜2024（モデル予測を使うものは 2022〜2024、有力騎手の順位が有効なのは 2021 年以降）、確認は 2025 を 1 回、2026 年はテストセット②（選択に使わず記述のみ）。モデルは学習していない（集計・回帰・ブートストラップ）。</p>

<div class="note"><b>要点（先に答え）.</b>
<ul>
<li><b>距離によって騎手への依存度が変わる、という証拠は見つからなかった。</b>馬の力を除いたうえで、騎手が着順のばらつきに占める割合は、芝で {pct(d['cells']['芝:b1']['share'])}（1000〜1400）→ {pct(d['cells']['芝:b2']['share'])}（1500〜1800）→ {pct(d['cells']['芝:b3']['share'])}（1900〜2200）→ {pct(d['cells']['芝:b4p']['share'])}（2300 以上）、ダートで {pct(d['cells']['ダ:b1']['share'])} → {pct(d['cells']['ダ:b2']['share'])} → {pct(d['cells']['ダ:b4p']['share'])}（1900 以上）。距離に対する傾きは芝 {pt(tzs['芝'][0]['stat'])}pt/帯（95% 区間 {pt(tzs['芝'][0]['ci'][0])}〜{pt(tzs['芝'][0]['ci'][1])}）、ダート {pt(tzs['ダ'][0]['stat'])}pt/帯（{pt(tzs['ダ'][0]['ci'][0])}〜{pt(tzs['ダ'][0]['ci'][1])}）で、どちらも 0 を含む。</li>
<li>騎手が着順を動かす大きさそのものは小さく、<b>馬の力を除いて着順のばらつきの 3〜5%</b>。騎手が馬の初角位置を普段から動かす割合は <b>1〜2%</b> で、距離帯による違いはない（傾き 芝 {pt(tps['芝'][0]['stat'])}pt・ダート {pt(tps['ダ'][0]['stat'])}pt、どちらも区間が 0 を含む）。</li>
<li>同じ馬が別の騎手に替わったときの成績の変化も、騎手の格の差 1 標準偏差あたり z で {min(betas):+.2f}〜{max(betas):+.2f} と一定で、短距離と長距離の差は有意でない（芝 p={sw['tests']['disc|芝:long-minus-short']['p']:.2f}、ダート p={sw['tests']['disc|ダ:long-minus-short']['p']:.2f}）。</li>
<li>有力騎手（直近 365 日の平地勝利数の上位 10 人）は、V4 の予測に対し<b>芝の短距離で 3 着内が +{100 * el['cells']['disc|芝:b1']['r_top3']['diff']:.1f}pt 上振れ、芝の 2300 以上で {100 * el['cells']['disc|芝:b4p']['r_top3']['diff']:+.1f}pt 下振れ</b>した（短距離 − 長距離の差 {pt(el['tests']['disc|芝:r_top3:b1-b4p']['stat'], 1)}pt、区間 {pt(el['tests']['disc|芝:r_top3:b1-b4p']['ci'][0], 1)}〜{pt(el['tests']['disc|芝:r_top3:b1-b4p']['ci'][1], 1)}、未補正 p={el['tests']['disc|芝:r_top3:b1-b4p']['p']:.2f}）。ただし<b>多重比較（BH）の補正後 q={q_min:.2f} で有意ではなく</b>、2025 年は同じ向きに再現しない（芝の短距離 {pt(el['cells']['2025|芝:b1']['r_top3']['diff'], 1)}pt）。市場の単勝確率で統制すると有力騎手の効果は 0 に近い。</li>
<li>有力騎手の単勝・複勝の回収率は、全 {n_roi_cells} セル（発見期間）の点推定が {roi_lo:.0f}〜{roi_hi:.0f}% で、回収率 &gt; 100% の検定 16 本は Holm 補正後の最小 p が {holm_min:.2f}。有力騎手の割引は市場が距離を問わず織り込んでいる。</li>
<li>『有力 × 距離帯』を V4 に足した場合の 2025 年の対数損失は、3 着内で {fp['top3']['delta_logloss']:+.5f}（区間 {fp['top3']['ci'][0]:+.5f}〜{fp['top3']['ci'][1]:+.5f}）、勝ちで {fp['win']['delta_logloss']:+.5f}（{fp['win']['ci'][0]:+.5f}〜{fp['win']['ci'][1]:+.5f}）と<b>わずかに悪化</b>（正が悪化）。特徴量化は見送り。</li>
</ul></div>

<h2>1. 事前に固定した設計</h2>
<ul>
<li><b>距離帯.</b> 1000〜1400／1500〜1800／1900〜2200／2300〜2800／3000 以上。芝・ダート別。芝の b4p は 2300 以上（b4 + b5）。</li>
<li><b>実行後の変更（標本数の理由）.</b> 最初の実行の結果を見たうえで、標本が足りず検定が成り立たない区分を直した。効果の方向で決めたものではないが、事前登録どおりではない（変更後に全部を再実行した）。(1) ダートの 2300 以上は約 4,000 走のため、ダートの『長い側』は 1900 以上（b4p）にした。(2) 両方の標本に 20 走以上ある騎手が 25 人未満の区分（1 年分の 2300 以上など）は、割合が 0 に退化するため『標本不足』とし検定から外した。(3) 持ち越しの最低騎乗数を 50 から 30 に下げた。(4) 1 年分（2025・2026・LOYO）の最小行数を 1,500 にした。</li>
<li><b>定義 (a) 分散分解.</b> 応答 z = レース内の着順の正規スコア。馬の効果＋騎手の効果をバックフィッティング（弱いリッジ）で推定し、<b>馬で二分した 2 つの独立な標本で推定した同じ騎手の効果の共分散</b>を『騎手の真の効果の分散』とする（ノイズが独立なので不偏）。割合 = その共分散 / z の分散。乱数分割 5 通りの平均。区間は騎手単位のブートストラップ 400 回（距離帯をまたいで同じ抽出）。位置取りへの影響は、応答を『今走の初角位置 − その馬の過去 3 走の平均（as-of、2 走以上）、レース内で中心化』に替えて同じ手順。</li>
<li><b>定義 (b) 騎手交代の対比.</b> 同じ馬が前走と違う騎手になった走だけ。z ~ 前走の z + それ以前の z の平均 + log 休養日数 + Δq（今走の騎手の as-of 勝率（直近 365 日、縮約）− 前走の騎手の同じ値、標準偏差 {res['switch_sd']:.3f} で割る）。レース単位のクラスタ頑健 SE。</li>
<li><b>定義 (c) 持ち越し.</b> 騎手 × 距離帯 × 芝ダ × 年（30 騎乗以上）の z の平均、およびモデルに対する 3 着内の残差の平均を、翌年と比べた相関（騎手単位のブートストラップ）。</li>
<li><b>定義 (d) 有力騎手と市場の織り込み.</b> 有力 = その日より前の直近 365 日の平地勝利数の上位 10 人（<a href="jockey_switch_ev.html">乗り替わりの章</a>と同じ）。モデル予測に対する残差、単勝・複勝の回収率（1 出走 100 円、レース単位）、市場の単勝確率（q_win）で統制したロジスティック回帰の係数。</li>
<li><b>検定の族.</b> 距離への依存の変化 13 本（BH）= 分散割合の傾き（芝・ダ）、位置取りの傾き（芝・ダ）、騎手交代（芝・ダ）、持ち越し（芝のみ。ダートの長距離は騎手 × 年の標本不足）、有力騎手の残差（3 着内・勝ち × 芝・ダ）、市場統制後の係数（芝・ダ）。回収率 &gt; 100% の検定 16 本（Holm）= 芝 5 帯 + ダ 3 帯 × 単勝・複勝。事前登録は 32 本、標本数の理由で 29 本に減った。</li>
<li>期間: 発見 2020〜2024、確認 2025 を 1 回、2026 は記述のみ。年ごとの再現は分散割合の傾きの符号で見る。モデルは tp-59 の master の p_*（<b>V4 の予測</b>、2022 年以降）。市場オッズは分析（統制・回収率）にのみ使い、特徴量にしない。キーは jockey_id・horse_id。</li>
</ul>

<h2>2. 騎手が着順を動かす割合（分散分解）</h2>
<p>馬の力を除いたあとの、騎手の真の効果が着順のばらつき（z の分散）に占める割合。発見期間（2020〜2024）、2025、2026 の順。2025・2026 は 1 年分で標本が少なく区間が広い。両方の標本に 20 走以上ある騎手が 25 人未満の区分（2300 以上など）は割合を推定できないので『標本不足』とした。</p>""")
    for sf, name in (("芝", "芝"), ("ダ", "ダート")):
        body.append(f"<h3>{name}</h3>" + table(["距離帯", "走数（発見）", "騎手数", "2020〜24 割合 [95%]", "2025", "2026（記述）"], cellrow(sz, sf), 1))
    body.append(f"""<img alt="騎手の分散割合" src="data:image/png;base64,{im['share_z']}" />
<p class="muted">横軸の距離帯は短い順。青=発見期間、赤=2025、灰=2026。</p>
<ul>
<li>距離への傾き（発見期間、1 帯上がるごとの変化）: 芝 {pt(tzs['芝'][0]['stat'], 2)}pt [{pt(tzs['芝'][0]['ci'][0])}, {pt(tzs['芝'][0]['ci'][1])}] p={tzs['芝'][0]['p']:.2f}、ダ {pt(tzs['ダ'][0]['stat'], 2)}pt [{pt(tzs['ダ'][0]['ci'][0])}, {pt(tzs['ダ'][0]['ci'][1])}] p={tzs['ダ'][0]['p']:.2f}。短距離 − 長距離（b1 − b4p）は芝 {pt(tzs['芝'][1]['stat'], 2)}pt [{pt(tzs['芝'][1]['ci'][0])}, {pt(tzs['芝'][1]['ci'][1])}]、ダ {pt(tzs['ダ'][1]['stat'], 2)}pt [{pt(tzs['ダ'][1]['ci'][0])}, {pt(tzs['ダ'][1]['ci'][1])}]。</li>
<li>2025 の芝（1000〜2200 の 3 帯）の傾きは {pt(sz['2025']['tests']['芝:trend']['stat'], 2)}pt [{pt(sz['2025']['tests']['芝:trend']['ci'][0])}, {pt(sz['2025']['tests']['芝:trend']['ci'][1])}]、2026 は {pt(sz['2026']['tests']['芝:trend']['stat'], 2)}pt [{pt(sz['2026']['tests']['芝:trend']['ci'][0])}, {pt(sz['2026']['tests']['芝:trend']['ci'][1])}]。いずれも 0 を含む。</li>
<li><b>年ごと（LOYO 相当）.</b> 芝の短距離（1000〜1400）の割合が 1900〜2200 より小さい向き（b1 − b3 &lt; 0）は {n_loyo_neg}/5 年で一致した（{', '.join(f'{k}: {100 * v["trend"]["芝:b1-b3"]:+.1f}pt' for k, v in loyo.items() if v['trend'].get('芝:b1-b3') is not None)}）。ただし発見期間をまとめた事前の検定（b1 − b4p）は有意でなく、1 年ごとの値は区間が広い。<b>『短距離のほうが騎手の影響がやや小さい』という向きの弱い傾向がありうる</b>が、このデータでは確かめられない。</li>
<li>解釈の注意: この推定は、馬が複数の騎手に乗られている（つながっている）部分から騎手の効果を識別するため、騎手と馬が固定的に組む場合は過小になる（下限に近い値）。馬の効果は期間を通じて一定と置いたので、馬の成長による変化は騎手の効果に混ざりうる。</li>
</ul>

<h2>3. 騎手が馬の位置取りを動かす割合</h2>
<p>応答を『今走の初角位置 − その馬の普段の初角位置（過去 3 走の平均）』に替えた同じ手順。騎手が馬の普段の位置取りをどれだけ動かすか。</p>""")
    for sf, name in (("芝", "芝"), ("ダ", "ダート")):
        body.append(f"<h3>{name}</h3>" + table(["距離帯", "走数（発見）", "騎手数", "2021〜24 割合 [95%]", "2025", "2026（記述）"], cellrow(sp, sf), 1))
    body.append(f"""<img alt="位置取りへの影響" src="data:image/png;base64,{im['share_pos']}" />
<ul>
<li>割合は 1〜2% で、距離を問わずほぼ一定（傾き 芝 {pt(tps['芝'][0]['stat'], 2)}pt p={tps['芝'][0]['p']:.2f}、ダ {pt(tps['ダ'][0]['stat'], 2)}pt p={tps['ダ'][0]['p']:.2f}）。『短距離ほど出遅れ・位置取りで騎手の影響が大きい』『長距離ほどペース判断の影響が大きい』のどちらも、このデータでは見えない。</li>
</ul>

<h2>4. 同じ馬が騎手を替えたときの成績の変化（騎手交代の対比）</h2>
<p>騎手の格（as-of 勝率）が 1 標準偏差（勝率で約 {res['switch_sd'] * 100:.1f}pt）良くなる騎手への交代で、今走の z（前走の成績を調整後）がどれだけ上がるか。</p>""")
    rows = []
    for sf in ("芝", "ダ"):
        for b in BANDS:
            r = []
            for per in ("disc", "2025", "2026"):
                c = sw["cells"].get(f"{per}|{sf}:{b}")
                r.append(f"{c['beta']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]" if c else "—")
            c0 = sw["cells"].get(f"disc|{sf}:{b}")
            rows.append([f"{'芝' if sf == '芝' else 'ダ'} {lbl(sf, b)}", f"{c0['n']:,}" if c0 else "—"] + r)
    body.append(table(["区分", "走数（発見）", "2021〜24", "2025", "2026（記述）"], rows, 1))
    body.append(f"""<img alt="騎手交代の対比" src="data:image/png;base64,{im['switch']}" />
<ul>
<li>傾きは距離を問わず {min(betas):+.2f}〜{max(betas):+.2f} で一定。短距離との差（長い側 − 短い側）は 芝 {sw['tests']['disc|芝:long-minus-short']['stat']:+.3f} [{sw['tests']['disc|芝:long-minus-short']['ci'][0]:+.3f}, {sw['tests']['disc|芝:long-minus-short']['ci'][1]:+.3f}] p={sw['tests']['disc|芝:long-minus-short']['p']:.2f}、ダ {sw['tests']['disc|ダ:long-minus-short']['stat']:+.3f} [{sw['tests']['disc|ダ:long-minus-short']['ci'][0]:+.3f}, {sw['tests']['disc|ダ:long-minus-short']['ci'][1]:+.3f}] p={sw['tests']['disc|ダ:long-minus-short']['p']:.2f}。2025・2026 も有意でない。</li>
<li>注意: 騎手の交代は偶然ではなく、陣営の判断（調子が悪い、勝負がかり、騎手の事情）で起きる。この傾きは因果効果ではなく、交代後の成績との関連。</li>
</ul>

<h2>5. 騎手の成績の持ち越し（再現性）</h2>
<p>騎手 × 距離帯 × 年の指標を翌年と比べた相関。z の平均は<b>馬の質（良い馬に乗る）を含む</b>ので高く出る。モデルに対する 3 着内の残差は、モデルが騎手の情報をすでに含むため、騎手の追加効果の持ち越しを見る。</p>""")
    rows = []
    for kind, kn in (("z", "z の平均"), ("resid", "モデルに対する 3 着内の残差")):
        for sf in ("芝", "ダ"):
            for b in BANDS:
                r = []
                for per in ("disc", "2025", "2026"):
                    c = cr["cells"].get(f"{kind}|{per}|{sf}:{b}")
                    r.append(f"{c['r']:+.2f} [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}]（{c['pairs']}）" if c else "—")
                if any(x != "—" for x in r):
                    rows.append([kn, f"{sf} {lbl(sf, b)}"] + r)
    body.append(table(["指標", "区分", "2020〜24（組数）", "2025", "2026（記述）"], rows, 2))
    t_cr = cr["tests"].get("z|disc|芝:b1-b4p")
    body.append(f"""<ul>
<li>z の平均の持ち越しは {min(z_r):.1f}〜{max(z_r):.1f} と高いが、これは主に乗る馬の質の持ち越し。モデルに対する残差の持ち越しは {min(rs_r):.2f}〜{max(rs_r):.2f} と弱く、<b>距離帯による系統的な違いはない</b>。</li>
<li>短距離 − 長距離の差の検定ができたのは芝の z だけ（{('差 ' + f"{t_cr['stat']:+.2f} [{t_cr['ci'][0]:+.2f}, {t_cr['ci'][1]:+.2f}] p={t_cr['p']:.2f}") if t_cr else '—'}）。ダートの長距離（1900 以上）は 30 騎乗以上の騎手 × 年が少なく、検定にならなかった。</li>
</ul>

<h2>6. 有力騎手の残差・回収率と、市場の織り込み</h2>
<p>有力騎手（上位 10 人）の馬 − それ以外の馬の、モデル（V4）の予測に対する実績の差（3 着内、勝ち。ポイント）と、有力騎手の単勝・複勝の回収率。発見期間は 2022〜2024。</p>""")
    rows = []
    for sf in ("芝", "ダ"):
        cells = ["b1", "b2", "b3", "b4", "b5", "b4p"] if sf == "芝" else ["b1", "b2", "b3", "b4", "b4p"]
        for b in cells:
            c = el["cells"].get(f"disc|{sf}:{b}")
            if not c:
                continue
            c25, c26 = el["cells"].get(f"2025|{sf}:{b}"), el["cells"].get(f"2026|{sf}:{b}")
            rows.append([f"{sf} {lbl(sf, b) if b != 'b3' or sf == '芝' else '1900-2200'}", f"{c['n_elite']:,}",
                         f"{pt(c['r_top3']['diff'], 1)} [{pt(c['r_top3']['ci'][0], 1)}, {pt(c['r_top3']['ci'][1], 1)}]",
                         f"{pt(c25['r_top3']['diff'], 1) if c25 else '—'} / {pt(c26['r_top3']['diff'], 1) if c26 else '—'}",
                         f"{c['roi_win']['roi']:.0f}% [{c['roi_win']['lo']:.0f}, {c['roi_win']['hi']:.0f}]",
                         f"{c['roi_place']['roi']:.0f}% [{c['roi_place']['lo']:.0f}, {c['roi_place']['hi']:.0f}]",
                         f"{c['roi_win_non']['roi']:.0f}% / {c['roi_place_non']['roi']:.0f}%"])
    body.append(table(["区分", "有力騎手の走数", "3 着内: 有力 − 他（pt、発見）", "2025 / 2026", "有力の単勝回収率", "有力の複勝回収率", "他の騎手 単/複"], rows, 1))
    body.append(f"""<img alt="有力騎手の残差と回収率" src="data:image/png;base64,{im['elite']}" />
<ul>
<li>モデル（V4）に対する有力騎手の上振れは、<b>芝の短距離（1000〜1400）で 3 着内 {pt(el['cells']['disc|芝:b1']['r_top3']['diff'], 1)}pt（区間 {pt(el['cells']['disc|芝:b1']['r_top3']['ci'][0], 1)}〜{pt(el['cells']['disc|芝:b1']['r_top3']['ci'][1], 1)}）</b>と、長距離側（2300 以上 {pt(el['cells']['disc|芝:b4p']['r_top3']['diff'], 1)}pt）より大きい（差 {pt(el['tests']['disc|芝:r_top3:b1-b4p']['stat'], 1)}pt、未補正 p={el['tests']['disc|芝:r_top3:b1-b4p']['p']:.2f}、BH 補正後 q={bh['q'].get('elite|disc|芝:r_top3:b1-b4p', float('nan')):.2f}）。<b>補正後は有意でなく、2025 年は再現しなかった</b>（芝の短距離 {pt(el['cells']['2025|芝:b1']['r_top3']['diff'], 1)}pt、2026 は {pt(el['cells']['2026|芝:b1']['r_top3']['diff'], 1)}pt の記述）。ダートでは差がなく、2025 はむしろ長い側のほうが大きかった。</li>
<li>市場の単勝確率で統制した有力騎手の係数（ロジスティック、3 着内）は、発見期間で 芝 {min(mk_t['芝']):+.2f}〜{max(mk_t['芝']):+.2f}、ダ {min(mk_t['ダ']):+.2f}〜{max(mk_t['ダ']):+.2f} の範囲に収まり、距離による差は有意でない（芝 p={mk['tests']['disc|芝:top3:long-minus-short']['p']:.2f}、ダ p={mk['tests']['disc|ダ:top3:long-minus-short']['p']:.2f}）。<b>人気が同じなら有力騎手かどうかで3 着内率は変わらない</b>（市場は騎手の価値を距離を問わず織り込んでいる）。</li>
<li>有力騎手の単勝・複勝の回収率は、全セルの点推定が 100% を下回る（Holm 補正後の最小 p={holm_min:.2f}）。短距離芝の単勝は 2026 年だけ 100% 台（101%）だが、記述のみで標本も小さい。</li>
</ul>

<h2>7. 特徴量化の見込み</h2>
<p>V4 の確率（ロジットをオフセット）に『有力 × 距離帯』の項を足して、発見期間（2022〜2024）で当てはめ、2025 の対数損失の差を 1 回だけ見た（レース単位ブートストラップ、モデルの学習ではなくロジスティック回帰）。</p>""")
    body.append(table(["目的", "2025 の走数", "対数損失の差（正が悪化）", "95% 区間"], [
        ["3 着内", f"{fp['top3']['n_test']:,}", f"{fp['top3']['delta_logloss']:+.5f}", f"{fp['top3']['ci'][0]:+.5f} 〜 {fp['top3']['ci'][1]:+.5f}"],
        ["勝ち", f"{fp['win']['n_test']:,}", f"{fp['win']['delta_logloss']:+.5f}", f"{fp['win']['ci'][0]:+.5f} 〜 {fp['win']['ci'][1]:+.5f}"]], 1))
    body.append(f"""<ul>
<li>差は正（悪化）で区間は 0 を外れる。発見期間で当てはめた『距離帯ごとの有力騎手の補正』は 2025 年に通用せず、特徴量にはならない。現行モデルは騎手の格を（距離を問わず）すでに織り込んでいる。</li>
</ul>

<h2>8. 結論</h2>
<ul>
<li><b>距離によって騎手の依存度が変わるか → 変わるという証拠はない。</b>騎手の効果は、馬の力を除いた着順のばらつきの 3〜5%、位置取りの変化の 1〜2% と小さく、距離帯で系統的に変わらない（傾きの区間は 0 を含み、2025・2026 も同様）。</li>
<li>弱い傾向として、芝の短距離（1000〜1400）は 1500〜2200 より騎手の割合がやや小さい向きが 5 年中 {n_loyo_neg} 年で出たが、事前に決めた検定（短距離 − 長距離）では有意でなく、確認できない。</li>
<li>有力騎手が V4 の予測を上回るのは芝の短距離だが、補正後は有意でなく 2025 年に再現せず、市場が織り込み済み。回収率の優位も、特徴量としての増分もない。</li>
<li>馬券・表示への示唆: 距離帯で騎手の重みを変える設計（短距離で騎手を重視する、長距離で割り引く）は、このデータでは根拠がない。</li>
</ul>

<h2>9. 限界</h2>
<ul>
<li>分散分解は『馬の効果＋騎手の効果』の加法モデルで、馬と騎手が固定的に組む場合の騎手の効果は識別できない（下限に近い）。馬の効果は期間を通じて一定と置いた。</li>
<li>2300 以上（特にダート）は標本が少なく、割合・持ち越し・2025 の確認は『標本不足』または広い区間になった。長距離側の結論は弱い。</li>
<li>モデルとの比較は V4（tp-59 の master の p_*）。V5/v6e の馬ごとの予測を持つ表ができれば、同じ手順で取り直せる（結論が動く可能性は小さい）。</li>
<li>『有力』を勝利数の上位 10 人にしたので、騎手の格の定義が違うと数値は動く（5・20 人は感度として未集計）。</li>
<li>位置取りは初角の相対位置（tp-61）。出遅れの有無やスタート直後の動きは別に測っていない。</li>
</ul>

<h2>10. 再現</h2>
<pre>python -m src.scripts.data.jockey_distance_dep build     # master.parquet ＋ tp-61 の位置取りから as-of の特徴を作る（JDD_OUT/frame.parquet）
python -m src.scripts.data.jockey_distance_dep analyze   # 分散分解・騎手交代・持ち越し・有力騎手の残差と回収率（JDD_OUT/result.json）
python -m src.scripts.data.jockey_distance_dep report    # この章を作る</pre>
<p class="muted">事前登録: コミット 7974254（評価の実行前）。関連: <a href="training_pipeline_decisions.html">決定ログ（§67）</a>・<a href="jockey_switch_ev.html">騎手の乗り替わり</a>・<a href="jockey_trainer_roi.html">騎手・調教師の分析と回収率（tp-59）</a>。</p>
</body></html>""")
    head = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>距離と騎手の依存度 — keiba-vpn</title>
<style>{CSS}</style></head>
<body>"""
    OUT.write_text(head + "".join(body), encoding="utf-8")
    print("wrote", OUT)
