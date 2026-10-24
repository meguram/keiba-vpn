"""tp-65 の章 docs/html/modeling/chapters/index_variance.html を、結果の JSON から作る。

python -m src.scripts.data.tp65_report --in /tmp/tp65_local     # ホストの <TP65_OUT> から持ってきた JSON のディレクトリ
入力: dist.json / pred.json / pred2.json / combo.json / rob_audit.json / rob_build.json、任意で lgb_*/result.json（LightGBM での増分。無ければ『実行待ち』と書く）。
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/html/modeling/chapters/index_variance.html"

CSS = """:root { --bg:#0f172a; --panel:#15213c; --border:#243358; --text:#e6ecff; --muted:#9aa7c7; --accent:#4f8bff; --accent-2:#7aa9ff; --warn:#fbbf24; --good:#34d399; --bad:#f87171; }
* { box-sizing: border-box; } body { font-family: "Noto Sans JP", system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.8; font-size: 15px; max-width: 1100px; margin: 0 auto; padding: 24px 28px 80px; }
a { color: var(--accent-2); } h1 { font-size: 26px; } h2 { font-size: 20px; margin: 40px 0 12px; border-left: 4px solid var(--accent); padding-left: 12px; } h3 { font-size: 16px; margin: 24px 0 8px; color: var(--accent-2); }
table { border-collapse: collapse; margin: 12px 0 18px; font-size: 13px; background: var(--panel); } th, td { border: 1px solid var(--border); padding: 5px 10px; } th { background: #1a2748; color: var(--muted); text-align: left; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
.muted { color: var(--muted); font-size: 13px; } .note { background: var(--panel); border: 1px solid var(--border); border-left: 4px solid var(--warn); border-radius: 8px; padding: 12px 16px; margin: 14px 0; }
.ok { color: var(--good); } .ng { color: var(--bad); } code { background: #0b1428; padding: 1px 5px; border-radius: 4px; }"""


def e(x) -> str:
    return html.escape(str(x))


def table(head: list[str], rows: list[list], num_from: int = 1) -> str:
    h = "".join(f"<th{' class=\"n\"' if i >= num_from else ''}>{e(c)}</th>" for i, c in enumerate(head))
    b = "".join("<tr>" + "".join(f"<td{' class=\"n\"' if i >= num_from else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{b}</table>"


def f(x, nd=2) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def load(d: Path, name: str):
    p = d / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def build(d: Path) -> str:
    dist, pred, pred2, combo = load(d, "dist.json"), load(d, "pred.json"), load(d, "pred2.json"), load(d, "combo.json")
    audit, rb = load(d, "rob_audit.json"), load(d, "rob_build.json")
    lgb = {p.parent.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("lgb_*/result.json"))}
    A = dist["A_dist"]
    out: list[str] = []
    P = out.append
    P(f'<!doctype html>\n<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>めぐ指数と netkeiba タイム指数の分散・外れ値の扱い（tp-65）— keiba-vpn</title><style>{CSS}</style></head><body>')
    P('<a href="../../index.html" class="muted">← ドキュメント一覧</a>')
    P("<h1>めぐ指数と netkeiba タイム指数の、分散と外れ値の扱いの比較（tp-65）</h1>")
    P('<p class="muted">2026-10-21、ユーザー依頼『めぐ指数の分散と netkeiba のタイム指数の分散が同様なのかを調べてほしい。netkeiba 側は外れ値に何らかの対応をしていて、確度の高い算出をしている気がする』への回答。'
      '資料: <a href="megu_index_v3a.html">めぐ指数 v3a</a>・<a href="race_performance_rating_system.html">レーティング指数の設計</a>・<a href="training_pipeline_decisions.html">決定ログ</a>。'
      f'対象は 2020〜2025 年の中央の平地（障害を除く）で、両方の指数がある {dist["n_both_2020_25"]:,} 走（2026 年は選択・当てはめに使わない）。'
      'netkeiba のタイム指数は各レースの出走前ページに載る直近 3 走分の値を、出馬表の過去走の日付と突き合わせて走ごとに復元した（指数 0 は欠損）。公式の算出方法は公開情報から確認していない（データからの推定のみ）。</p>')
    # ── 要点 ────────────────────────────────────────────────────────────
    ic, rt = dist["C_icc"], dist["C_retest_fair"]
    lc = {x["loss_bin"]: x for x in dist["F_loss_compression"]}
    key = pred["features"]
    def g3(k):
        return sum(key[k][y]["ll_gain"] for y in ("2022", "2023", "2024")) / 3
    P('<div class="note"><b>要点（先に答え）.</b><ul>')
    P(f'<li><b>分散は「単位は同じ、尺度は条件で入れ替わる」</b>。どちらも 1 点 ≒ 0.1 秒（着差 0.2〜8 秒の範囲で 約 10 点／秒）。全体の標準偏差は めぐ指数（相対）{f(A["megu"]["sd"],1)}・めぐ指数（絶対）{f(A["megu_abs"]["sd"],1)}・netkeiba {f(A["nk"]["sd"],1)} で、<b>めぐ指数の「相対」は同クラスの 2 着馬が 50 になるよう基準を揃えるため、クラスの差を持つ netkeiba（絶対）より小さく見える</b>だけ。絶対どうし（{f(A["megu_abs"]["sd"],1)} 対 {f(A["nk"]["sd"],1)}）ではほぼ同程度。距離で差が出て、短距離は netkeiba の方がばらつきが大きく（〜1200m: {f(dist["B_by_group"]["dband"]["~1200"]["sd_nk"],1)} 対 {f(dist["B_by_group"]["dband"]["~1200"]["sd_megu"],1)}）、長距離は めぐ指数の方が大きい（2500m〜: {f(dist["B_by_group"]["dband"]["2500~"]["sd_megu"],1)} 対 {f(dist["B_by_group"]["dband"]["2500~"]["sd_nk"],1)}）。</li>')
    P(f'<li><b>1 走のノイズ（馬内の標準偏差）はほぼ同じ</b>：絶対どうしで めぐ {f(ic["megu_abs(絶対)"]["within_sd"],1)} 点・netkeiba {f(ic["nk(絶対)"]["within_sd"],1)} 点（約 1.1〜1.2 秒）。連続する走の相関は {f(rt["megu_abs(絶対)"]["pearson"],3)} 対 {f(rt["nk(絶対)"]["pearson"],3)}、ICC(1) は {f(ic["megu_abs(絶対)"]["icc1"],3)} 対 {f(ic["nk(絶対)"]["icc1"],3)}。<b>netkeiba の方が 0.03〜0.04 だけ信頼性が高い</b>が、これは 1 走のノイズが小さいからではなく<b>馬どうしの差（馬間の標準偏差 {f(ic["nk(絶対)"]["between_sd"],1)} 対 {f(ic["megu_abs(絶対)"]["between_sd"],1)}）が大きい</b>から。『netkeiba の方が 1 走ごとの確度が高い』は、この比較では支持されない。</li>')
    P(f'<li><b>外れ値への対応は「弱い」</b>。netkeiba にも下限は無い（最小 {f(dist["F_edges"]["nk"]["min"],0)}、−50 未満が {dist["F_edges"]["nk"]["share_le_m50"]*100:.2f}%）。上は 130 付近が最大で 120 以上は {dist["F_edges"]["nk"]["share_ge120"]*100:.2f}%。大差負けの圧縮は穏やか：着差 8 秒までは めぐ指数と同じ 約 10 点／秒、8〜12 秒で 9.1、30 秒超で 6.8 点／秒（めぐ指数は 10.0 のまま）。極端な低値の件数は めぐ指数の 1/3〜1/5（−50 未満: netkeiba 91 件・めぐ指数 486 件、−100 未満: 32 件・110 件）で、z 値でも下側 0.1% 点は {f(dist["A_z_quantiles"]["nk"]["0.001"],1)} 対 {f(dist["A_z_quantiles"]["megu"]["0.001"],1)}。<b>確かに外れ値は抑えられているが、強い丸めではなく、めぐ指数の極端値の多くは基準タイムの薄いセル（低信頼 約 9%）に集中</b>している（2 指標の食い違いが 2 SD を超える割合は 低信頼で {dist["F_zdiff_groups"]["low_conf"]["True"]["share_gt2"]*100:.1f}% ／それ以外 {dist["F_zdiff_groups"]["low_conf"]["False"]["share_gt2"]*100:.1f}%）。</li>')
    P(f'<li><b>netkeiba の値は後から改訂されない</b>（同じ走を次走・次々走・3 走先のページで見ると 100% 一致）。基準の更新は過去にさかのぼらない。一方 めぐ指数は基準を年・月で固定した v3a で、勝ち馬の水準は 年ごとに 52.3〜52.7（netkeiba は 87〜90、月ごとの幅 {dist["E_month_range"]["nk"]:.0f} 点 対 {dist["E_month_range"]["megu"]:.1f} 点）。</li>')
    P(f'<li><b>次走の予測力は 2 指標で同程度、組み合わせると明確に上がる</b>。指数加重平均（ewm5）の勝ち馬の対数尤度の改善は、めぐ指数 {f(g3("megu__ewm5"),3)}・netkeiba {f(g3("nk__ewm5"),3)}（2022〜24 年平均）、2025 年は {f(key["megu__ewm5"]["2025"]["ll_gain"],3)} 対 {f(key["nk__ewm5"]["2025"]["ll_gain"],3)}。両方を入れると 2025 年で {f(combo["2025"]["gain_both"],3)}。')
    P('めぐ指数の係数は 2022〜24 年で選んでおり（基準タイムは 2020〜2025 年固定）、2022〜24 年側はやや楽観になる。</li>')
    P('</ul></div>')
    # ── 1. データと定義 ─────────────────────────────────────────────────
    pr = load(d, "prep_report.json")
    P("<h2>1. データと定義</h2>")
    P("<ul>")
    P("<li><b>めぐ指数</b>: v3a の相対（<code>megu_index_v3a</code>。同世代・同クラスの 2 着馬が 50）、大敗を −30 で丸めた版（<code>_robust</code>）、絶対（<code>megu_index_v3a_abs_robust</code>。1 勝クラス・3 歳以上を基準にした単一尺度）。1 点 = 0.1 秒の固定。</li>")
    P("<li><b>netkeiba のタイム指数</b>: 出走前ページ（<code>race_index</code>）の『直近 3 走の指数』を、出馬表の過去走の日付と突き合わせて走ごとに復元した。実データで、ページの 1 つ目の値が『前走』、2 つ目が『前々走』と対応することを確認（めぐ指数（絶対）との相関 0.87／0.86／本走 0.61）。同じ走は次走・次々走・3 走先のページに最大 3 回現れ、最初に現れた値（次走の出走前ページ＝予測時点で見える値）を使う。指数 0 は欠損。</li>")
    P(f'<li>被覆: 2020〜2025 年の中央の平地 {dist["n_runs_all"]:,} 走のうち、めぐ指数 {dist["cover"]["megu"]*100:.1f}%・netkeiba {dist["cover"]["nk"]*100:.1f}%、両方 {dist["n_both_2020_25"]:,} 走。netkeiba の欠けは、次走が無い走（引退・長期休養）と、2025 年の走の次走が 2026 年のページ（<code>page_reference</code> に <code>race_index</code> が無い）に載る分（2025 年の欠け 24.8% 対 他年 約 10%）。着順の低い馬ほど欠ける（3 着以内 4.3%・10 着以下 23.2%）ので、比較は『両方ある走』に揃えた。</li>')
    P("<li>2026 年は選択・当てはめに使っていない（netkeiba のページが無く、そもそも比較できない）。</li>")
    P("</ul>")

    # ── 2. 分散 ─────────────────────────────────────────────────────────
    P("<h2>2. 分散の比較（両方ある走）</h2>")
    names = [("megu", "めぐ指数（相対）"), ("megu_clip", "めぐ指数（相対・−30 丸め）"), ("megu_abs", "めぐ指数（絶対・丸め）"), ("nk", "netkeiba")]
    rows = []
    for k, nm in names:
        a_ = A[k]
        rows.append([nm, f"{a_['mean']:.1f}", f"{a_['sd']:.1f}", f"{a_['mad_sd']:.1f}", f"{a_['sd_over_mad']:.2f}", f"{a_['skew']:.1f}", f"{a_['ex_kurt']:.0f}", f"{a_['q']['0.001']:.0f}", f"{a_['q']['0.01']:.0f}", f"{a_['q']['0.5']:.0f}", f"{a_['q']['0.99']:.0f}", f"{a_['q']['0.999']:.0f}", f"{a_['min']:.0f}", f"{a_['max']:.0f}"])
    P(table(["指数", "平均", "SD", "MAD 由来の SD", "SD/MAD-SD", "歪度", "超過尖度", "0.1%点", "1%点", "中央値", "99%点", "99.9%点", "最小", "最大"], rows))
    P(f'<p class="muted">超過尖度が 100 を超えるのは、どちらも数件の巨大な外れ値（最小 {A["megu"]["min"]:.0f}／{A["nk"]["min"]:.0f}）のため。SD と MAD 由来の SD の比（1 に近いほど裾が軽い）は めぐ指数（絶対）1.08・netkeiba 1.15・めぐ指数（相対）1.28。2 指標の相関は 相対×netkeiba {f(dist["A_corr"]["pearson"],3)}（クラスの違いが netkeiba にだけ入る）、<b>絶対×netkeiba {f(dist["A_corr"]["abs_pearson"],3)}（順位相関 {f(dist["A_corr"]["abs_spearman"],3)}、傾き {f(dist["A_corr"]["abs_slope_nk_on_megu"],2)}）</b>。</p>')
    P("<h3>2-1. 条件別の標準偏差（点）</h3>")
    rows = []
    for gcol, lab in (("surface", "面"), ("dband", "距離"), ("cls", "クラス"), ("tc", "馬場")):
        for lv, x in dist["B_by_group"][gcol].items():
            rows.append([f"{lab}: {lv}", f"{x['n']:,}", f"{x['sd_megu']:.1f}", f"{x['sd_nk']:.1f}", f"{x['mad_megu']:.1f}", f"{x['mad_nk']:.1f}", f"{x['corr']:.2f}"])
    P(table(["条件", "走数", "SD めぐ(相対)", "SD netkeiba", "MAD-SD めぐ", "MAD-SD netkeiba", "相関"], rows))
    P('<p class="muted">クラス別では、新馬・未勝利で ばらつきと相関（0.84〜0.89）が最も大きい。ダートは芝よりばらつきが大きく（めぐ 14.9 対 12.3、netkeiba 19.7 対 17.0）、不良馬場でも広がる。距離が延びると netkeiba は縮み（20.2→12.7）、めぐ指数は広がる（12.0→19.2）ので、<b>2 指標の尺度は距離帯で逆に動く</b>。</p>')

    # ── 3. 走ごとのばらつき ───────────────────────────────────────────
    P("<h2>3. 走ごとのばらつき（信頼性）</h2>")
    rows = []
    for k in ("megu(相対)", "megu_abs(絶対)", "nk(絶対)", "nk_rel(クラス内相対)"):
        r_ = rt[k]
        rows.append([k, f"{r_['n']:,}", f"{r_['pearson']:.3f}", f"{r_['diff_sd']:.1f}", f"{r_['diff_mad']:.1f}", f"{ic[k]['within_sd']:.1f}", f"{ic[k]['between_sd']:.1f}", f"{ic[k]['icc1']:.3f}"])
    rows.append(["megu(相対, 0.5〜99.5% クリップ)", "—", "—", "—", "—", f"{ic['megu(相対,0.5-99.5%クリップ)']['within_sd']:.1f}", f"{ic['megu(相対,0.5-99.5%クリップ)']['between_sd']:.1f}", f"{ic['megu(相対,0.5-99.5%クリップ)']['icc1']:.3f}"])
    P(table(["指数", "連続走ペア", "前走との相関", "差の SD", "差の MAD-SD", "馬内 SD", "馬間 SD", "ICC(1)"], rows))
    P('<p class="muted">連続走は間隔 100 日以内。ICC(1) は 4 走以上の馬（19,996 頭）での一元配置の分散成分。『相対』どうしと『絶対』どうしを比べるのが公平（めぐ指数の相対は同クラスで 50 に揃うので、クラスの上下による馬間の差が消える。netkeiba もクラス内の中央値で引くと同じ水準になる: 相関 0.352・ICC 0.210）。</p>')
    dw, dr = dist["D_winner_cell"], dist["D_race_median_sd"]
    P(f'<p>レース水準の揺れ: 同じ面・距離帯・クラス・年のセルで、レースごとの勝ち馬の指数の標準偏差は めぐ {dw["mean_sd_megu"]:.1f}・netkeiba {dw["mean_sd_nk"]:.1f}（{dw["cells"]} セルの平均。同程度。勝ち馬どうしの能力差を含む）。レースごとの中央値の標準偏差は めぐ(相対) {dr["sd_megu"]:.1f}・netkeiba {dr["sd_nk"]:.1f}（相関 {dr["corr"]:.2f}）で、netkeiba の方が大きいのはクラスの差を持つため。勝ち馬の水準の年変動は めぐ指数の方が小さい（年ごと 52.3〜52.7、月ごとの幅 {dist["E_month_range"]["megu"]:.1f} 点 対 netkeiba {dist["E_month_range"]["nk"]:.0f} 点）。</p>')

    # ── 4. 外れ値 ───────────────────────────────────────────────────────
    P("<h2>4. 外れ値の扱い（データからの推定）</h2>")
    P("<h3>4-1. 大差負けでの圧縮（同レースの勝ち馬との指数差 ÷ 着差の秒）</h3>")
    rows = []
    for x in dist["F_loss_compression"]:
        if x["pts_per_sec_megu"] is None:
            continue
        rows.append([x["loss_bin"], f"{x['n']:,}", f"{x['loss_sec_med']:.1f}", f"{-x['d_megu_med']:.0f}", f"{-x['d_nk_med']:.0f}", f"{x['pts_per_sec_megu']:.2f}", f"{x['pts_per_sec_nk']:.2f}", f"{x['sd_megu']:.1f}", f"{x['sd_nk']:.1f}"])
    P(table(["着差（秒）", "走数", "中央値(秒)", "差 めぐ", "差 netkeiba", "点/秒 めぐ", "点/秒 netkeiba", "SD めぐ", "SD netkeiba"], rows))
    P('<p class="muted">8 秒を超える大差は少数（12 秒超は 130 走余り）。さらに 12〜16 秒で 8.97、16〜20 秒で 9.09、30〜60 秒で 6.83 点／秒（netkeiba）、めぐ指数は一貫して 10.0。同じ着差での差の SD は netkeiba の方が 1.4〜2 倍強大きい（斤量・馬場・ペースなど走ごとの補正の入り方が大きいため。ノイズかどうかは次節の予測力で見る）。</p>')
    P("<h3>4-2. 端と値の集中</h3>")
    e_ = dist["F_edges"]
    P(table(["", "最小", "最大", "0 以下", "100 以上", "120 以上", "−50 以下"], [[k, f"{e_[k]['min']:.0f}", f"{e_[k]['max']:.0f}", f"{e_[k]['share_le0']*100:.2f}%", f"{e_[k]['share_ge100']*100:.2f}%", f"{e_[k]['share_ge120']*100:.3f}%", f"{e_[k]['share_le_m50']*100:.3f}%"] for k in ("nk", "megu")]))
    P('<ul><li>netkeiba に<b>下限も固定の上限も見えない</b>（最小 −1243）。上側は 125 を超えるのが 10 走、130 が 1 走で、緩やかな頭打ち（99.9% 点 115）に見える。整数に丸められ、値の集中（同じ値が突出して多い）は無い（最頻でも 2.5%）。</li>')
    P('<li>深い下側の外れ値は netkeiba の方がはるかに少ない（−50 以下 0.039% 対 0.176%、−100 以下 32 件対 110 件）。例: 2600m・207 秒の大敗は netkeiba −160 に対し めぐ −365。着差が極端なとき netkeiba は差を縮めて表している。</li>')
    P('<li>めぐ指数の相対版の巨大な負値は、基準タイムのサンプルが薄いコース・非標準距離に偏る。低信頼フラグ（約 9%）の走は、2 指標の食い違い（z 差の絶対値が 2 超）が 6.5% で、それ以外の 2.6% の 2.5 倍。新馬は netkeiba との食い違いが特に多い（6.7%）。</li></ul>')
    P("<h3>4-3. 改訂（基準の更新）</h3>")
    rv = dist["G_revision"]
    P(f'<p>同じ走の値を、次走のページ（lag0）と次々走のページ（lag1）で比べると、{rv["nk_lag0->nk_lag1"]["n"]:,} 走で一致率 {rv["nk_lag0->nk_lag1"]["share_equal"]*100:.1f}%（差の SD {rv["nk_lag0->nk_lag1"]["sd_diff"]:.2f}）、lag1 と lag2 は {rv["nk_lag1->nk_lag2"]["n"]:,} 走で完全一致。<b>netkeiba の指数は一度出ると改訂されない</b>（基準タイムの更新は将来の走だけに効く）。</p>')

    # ── 5. 予測力 ───────────────────────────────────────────────────────
    P("<h2>5. 次走の予測力と、ロバスト化の効果</h2>")
    P('<p>『前走までの指数から、次走の勝ち馬を当てる』対数尤度の改善（レースごとの softmax(β×指数)。一様な予測との差。大きいほど良い）。<b>両方の指数がある走だけで履歴を作り、同じ履歴で比較</b>。β は評価年を除く 2022〜24 年で当てはめ、区間はレース単位ブートストラップ（約 ±0.025）。2025 年は参考として併記（めぐ指数の基準・係数は 2022〜24 年で選んだため、2022〜24 年側はめぐ指数に楽観的）。</p>')
    def row(k, nm):
        v = key[k]
        return [nm] + [f"{v[y]['ll_gain']:.4f}" for y in ("2022", "2023", "2024", "2025")] + [f"{v['2025']['rank_corr']:.3f}"]
    sel = [("megu__last1", "めぐ 直近"), ("megu__mean3", "めぐ 3 走平均"), ("megu__max5", "めぐ 5 走の最高"), ("megu__median5", "めぐ 5 走の中央値"), ("megu__huber5", "めぐ 5 走の Huber"), ("megu__ewm5", "めぐ 指数加重平均（0.65）"),
           ("nk__last1", "netkeiba 直近"), ("nk__mean3", "netkeiba 3 走平均"), ("nk__max5", "netkeiba 5 走の最高"), ("nk__median5", "netkeiba 5 走の中央値"), ("nk__huber5", "netkeiba 5 走の Huber"), ("nk__ewm5", "netkeiba 指数加重平均（0.65）")]
    P(table(["特徴量（履歴 5 走）", "2022", "2023", "2024", "2025", "着順との順位相関(2025)"], [row(k, n) for k, n in sel]))
    P('<p class="muted">直近 1 走や最高値は弱く、<b>新しい走ほど重い加重平均が最良</b>（中央値・Huber・トリム平均のような位置推定は新しさを捨てるため劣る）。netkeiba の最高値・直近の弱さは『1 走ごとの揺れ』の大きさを示唆する。</p>')
    P("<h3>5-1. 外れ値処理を後付けしたときの効果</h3>")
    ps = pred2["features"]
    def g3b(k):
        return sum(ps[k][y]["ll_gain"] for y in ("2022", "2023", "2024")) / 3
    rows = []
    for nm, pref in (("めぐ指数（相対）", "megu"), ("netkeiba", "nk")):
        rows.append([f"{nm} そのまま ewm(0.65)", f"{g3b(pref+'_raw__ewm0.65'):.4f}", f"{ps[pref+'_raw__ewm0.65']['2025']['ll_gain']:.4f}", "—"])
        for m in (20, 30, 40, 60):
            rows.append([f"{nm} 下限（勝ち馬−{m}点）ewm(0.65)", f"{g3b(pref+f'_floor{m}__ewm0.65'):.4f}", f"{ps[pref+f'_floor{m}__ewm0.65']['2025']['ll_gain']:.4f}", f"{g3b(pref+f'_floor{m}__ewm0.65')-g3b(pref+'_raw__ewm0.65'):+.4f}"])
        rows.append([f"{nm} 下限40 + 減衰 0.5", f"{g3b(pref+'_floor40__ewm0.5'):.4f}", f"{ps[pref+'_floor40__ewm0.5']['2025']['ll_gain']:.4f}", f"{g3b(pref+'_floor40__ewm0.5')-g3b(pref+'_raw__ewm0.65'):+.4f}"])
        rows.append([f"{nm} 下限40 + 加重 Huber(0.65)", f"{g3b(pref+'_floor40__whuber0.65'):.4f}", f"{ps[pref+'_floor40__whuber0.65']['2025']['ll_gain']:.4f}", f"{g3b(pref+'_floor40__whuber0.65')-g3b(pref+'_raw__ewm0.65'):+.4f}"])
    P(table(["処理", "2022〜24 平均", "2025", "そのまま ewm との差(22〜24)"], rows))
    P('<ul><li><b>大差負けの下限（同じレースの勝ち馬の指数 − 約 4 秒）を入れると、両方の指数で次走予測が改善</b>する。めぐ指数 +0.006、netkeiba +0.006（減衰 0.5 で +0.013、2025 年で +0.015）。下限が厳しすぎる（−20 点）と情報を捨てて悪化し、30〜60 点は同程度で、40 点付近が良い。</li>')
    P('<li><b>netkeiba 自身も、同じ下限を足すとまだ改善する</b>。つまり netkeiba の外れ値処理は予測の面では十分ではない（極端な着差の縮小は穏やか）。</li>')
    P('<li>減衰は めぐ指数が 0.65 付近、netkeiba は 0.5 のほうが良い（推測: netkeiba は走ごとの補正の入り方が大きく、直近を重く見て離れた走を軽く扱う方が合う）。重み付き Huber は ewm より僅かに良い。</li></ul>')
    P("<h3>5-2. 2 指標を合わせる</h3>")
    rows = []
    for y in ("2022", "2023", "2024", "2025"):
        v = combo[y]
        bm = v["both_minus_single"]
        rows.append([y, f"{v['races']:,}", f"{v['gain_single']['f_megu__ewm5']:.4f}", f"{v['gain_single']['f_nk__ewm5']:.4f}", f"{v['gain_both']:.4f}", f"{bm['f_megu__ewm5']['diff']:+.4f}（{bm['f_megu__ewm5']['lo']:+.3f}〜{bm['f_megu__ewm5']['hi']:+.3f}）", f"{bm['f_nk__ewm5']['diff']:+.4f}（{bm['f_nk__ewm5']['lo']:+.3f}〜{bm['f_nk__ewm5']['hi']:+.3f}）"])
    P(table(["年", "レース数", "めぐ ewm", "netkeiba ewm", "両方", "両方−めぐ(区間)", "両方−netkeiba(区間)"], rows))
    P('<p>2 指標は<b>補完的</b>で、両方を使うとどの年でも片方だけより 0.03〜0.05 上がる（区間は 0 を含まない）。係数はほぼ 0.7 対 0.6 で、同じくらい重く効く。</p>')
    # ── 6. V5 への増分 ──────────────────────────────────────────────────
    P("<h2>6. 特徴量としての増分（V5 の 1,142 列に累積）</h2>")
    P('<p>追加するのは、各走の履歴（その走より前で、指数がある直近 5 走）から作る列。めぐ指数・netkeiba のそれぞれについて、0.5〜99.5% でクリップした（2020〜2021 年の値で固定）値の ewm（0.65）・最高・中央値・トリム平均・Huber、<b>大差負けの下限（勝ち馬−40 点）をかけた ewm と加重 Huber</b>、使った走の数。V5 には めぐ指数も netkeiba のタイム指数も入っていない（V4 では netkeiba の速度指数を履歴特徴量が代わりに担うとして不採用）。先読み検査: 2023-07-01 以降の指数をすべて乱数に置き換えて作り直しても、それより前の '
      f'{audit["rows_before_cut"]:,} 行の特徴量は最大差 {max(audit["max_abs_diff"].values()):.1e}（完全一致）。値は予測時点で見えるものだけ（めぐ指数は前走までに確定した値、netkeiba は次の出走前ページに載った最初の値）。</p>')
    if lgb:
        rows = []
        for nm, r_ in lgb.items():
            c = r_["compare"]
            rows.append([nm, str(r_["n_cols"]), f"{c['Δwin']:+.5f}", f"{c['win_hi']:+.5f}", f"{c['Δtop2']:+.5f}", f"{c['Δtop3']:+.5f}", "採用" if c.get("採用") else "不採用"])
        P(table(["構成", "列数", "Δ勝率", "勝率の上限", "Δ連対", "Δ複勝", "判定"], rows))
        for nm, r_ in lgb.items():
            P(f'<p class="muted">{e(nm)}: 年別 ' + " ／ ".join(f"{y}: Δ勝率 {c['Δwin']:+.5f}・連対 {c['Δtop2']:+.5f}・複勝 {c['Δtop3']:+.5f}" for y, c in r_["per_year"].items()) + "</p>")
    else:
        P('<div class="note"><b>実行待ち</b>: LightGBM での評価（<code>python -m src.scripts.data.tp65_feat run --name rb_both</code>）はホストの共有スロット待ち。結果は <code>/home/jovyan/work/tp65_out/lgb_rb_both/result.json</code>。取り込み後に <code>tp65_report</code> を再実行して、この節に反映する。</div>')

    # ── 7. 結論・改良案・未確認 ─────────────────────────────────────────
    P("<h2>7. 結論と、めぐ指数の改良案</h2>")
    P("<ol>")
    P("<li><b>分散は同様か</b>: 同じ尺度（1 点 = 0.1 秒）で、絶対どうしの全体の SD は ほぼ同程度（17.3 対 18.7）。1 走のノイズ（馬内 SD）も 11.7 対 11.9 点でほぼ同じ。差が出るのは距離帯（短距離は netkeiba、長距離はめぐ指数の方が大きい）と、深い外れ値（めぐ指数に多い）。</li>")
    P("<li><b>netkeiba の外れ値対応</b>: 下限・上限の固定は見えず、大差負けを緩やかに縮める程度（8 秒超で 約 9 点／秒、30 秒超で 6.8）。値は一度出たら改訂されない。『確度が高い』は、1 走ごとの精度というより 馬間の差の大きさ（ICC）と極端値の少なさに表れている。</li>")
    P("<li><b>改善の余地は めぐ指数側に具体的にある</b>: (a) 大差負けの下限（勝ち馬−40 点）を指数の定義に入れる（予測 +0.006）。(b) 低信頼（基準タイムの薄いセル、約 9%）の走を履歴の平均で軽く扱う、または基準が薄いセルの極端値を縮約する。(c) 履歴の集約は 5 走の中央値などの位置推定より、新しさを重くした加重 Huber が良い（+0.003）。(d) 1 走のノイズ（約 11 点）は定義の違いでは減らず、複数走の集約と 2 指標の併用で減らす。</li>")
    P("<li><b>2 指標の併用</b>が最も効く（片方だけより次走予測で +0.03〜0.05）。ただし netkeiba のタイム指数は V5 の入力に無く、運用では次走の出走前ページを取得できる必要がある（下記）。</li>")
    P("</ol>")
    P("<h3>未確認・注意</h3><ul>")
    P("<li><b>netkeiba のタイム指数の運用での取得</b>: 2026 年の <code>race_index</code> は <code>page_reference/tables/2026/</code> に無く、2025 年の走の次走が 2026 年にある分は欠けている。予測時点（T-45）に『前走までの指数』を出走前ページから取れるかは、この調査では実機で確認していない（スクレイピング SLA の監査 G10・G11 の続き）。取れないなら、netkeiba 側の列は特徴量として使えず、めぐ指数の改良（上記）だけが残る。</li>")
    P("<li>めぐ指数の係数・基準は 2022〜24 年で選んだため、2022〜24 年の比較はめぐ指数にやや楽観的。2025 年だけの比較では netkeiba が上（ewm で 0.2525 対 0.2294）。区間は ±0.025 と広く、1 年分の差は有意ではない。</li>")
    P("<li>netkeiba の算出方法は公開情報で確認していない。外れ値処理の推定は上記の観察に基づくもの。</li>")
    P("<li>V5 への増分の採否は、上の LightGBM の結果と、取得可否の確認を合わせて決める（市場オッズ・人気は使っていない。test 2025・2026 は選択に使っていない）。</li></ul>")
    P(f'<p class="muted">コード: <code>src/scripts/data/tp65_prep.py</code>（走の表）・<code>tp65_var.py</code>（分散・予測力）・<code>tp65_feat.py</code>（特徴量・先読み検査・LightGBM）・<code>tp65_report.py</code>（この章）、テスト <code>tests/pipeline/test_tp65.py</code>。出力はホスト <code>/home/jovyan/work/tp65_out/</code>。</p>')
    P("</body></html>")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    a = ap.parse_args()
    OUT.write_text(build(Path(a.inp)), encoding="utf-8")


if __name__ == "__main__":
    main()
