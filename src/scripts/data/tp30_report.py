"""tp-30 の結果（tp30/*.json）から docs/html/modeling/chapters/race_volatility.html を作る。

python -m src.scripts.data.tp30_report [--src data/local/modeling/training_pipeline/tp30]
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/html/modeling/chapters/race_volatility.html"
STYLE_FROM = ROOT / "docs/html/modeling/chapters/course_performance_similarity.html"
VENUE = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京", "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}


def esc(x) -> str:
    return html.escape(str(x))


def fmt(x, d=4, plus=False) -> str:
    if x is None:
        return "-"
    try:
        v = float(x)
    except Exception:
        return esc(x)
    return f"{v:+.{d}f}" if plus else f"{v:.{d}f}"


def table(head: list[str], rows: list[list], num_cols: set[int] | None = None) -> str:
    num_cols = num_cols or set()
    h = "".join(f"<th>{esc(c)}</th>" for c in head)
    body = "".join("<tr>" + "".join(f"<td{' class=num' if i in num_cols else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{body}</table>"


def course_label(c: str) -> str:
    v, s, d, var = (c.split("|") + ["", "", "", ""])[:4]
    return f"{VENUE.get(v, v)}{s}{d}{var}"


def load(src: Path, name: str):
    p = src / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def build(src: Path) -> str:
    d, pr, tm, ft, bt = (load(src, n) for n in ("describe", "predict", "temper", "feature", "bet"))
    style = re.search(r"<style>(.*?)</style>", STYLE_FROM.read_text(encoding="utf-8"), re.S).group(1)
    parts = [f'<!doctype html>\n<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>荒れやすいレースの分析（tp-30）— keiba-vpn</title><style>{style}</style></head>\n<body><a href="../../index.html" class="muted">← ドキュメント一覧</a>\n<h1>荒れやすいレースの分析（tp-30）</h1>']
    parts.append('<p class="muted">レースの「荒れ」を結果から定義し、(1) どんな条件で荒れるか、(2) 事前に予測できるか、(3) 予想の特徴量に寄与するか、(4) 馬券の買い目の幅の決め方に寄与するかを分析した。対象は JRA の 2020〜2024 年（17,272 レース）。馬券の評価は valid 2022〜2024 の v3 OOF（10,061 レース）。test 2025・2026 年は使っていない。</p>')
    parts.append('<div class="note"><b>市場情報の扱い.</b> 予想モデル（勝率予測）の特徴量にはオッズ・人気を入れない（「モデル用」）。馬券の幅の決定にだけ、購入前のオッズ構造を使う（「馬券用」。履歴は確定オッズで代用しており、購入時点の T-60/T-15 ではない）。荒れの正解（事後指標）は結果から作る。</div>')
    parts.append("<h2>結論</h2><ul>")
    if pr:
        m, k = pr["model"]["targets"]["vol"], pr["market"]["targets"]["vol"]
        parts.append(f"<li><b>荒れやすさは事前にある程度予測できる。</b>モデル用（オッズなし）でも荒れ度の R² {m['r2']:.2f}、荒れ上位 20% の AUC {m['auc_top20']:.2f}、予測上位 20% の中の実際の荒れ割合は基準の {m['lift']:.1f} 倍。馬券用（確定オッズの構造を足す）は R² {k['r2']:.2f}、AUC {k['auc_top20']:.2f}、{k['lift']:.1f} 倍。</li>")
    parts.append("<li><b>荒れやすい条件:</b> 出走頭数が多い（15 頭以上）、未勝利・条件戦より上位クラス（オープン・3 勝クラスで荒れやすい）、短距離、小倉・新潟・福島（ローカル場）、開催後半の番組（10〜12R）。逆に 11 頭以下、未勝利戦、東京の芝中距離、2000m 以上は堅い。</li>")
    if tm:
        parts.append("<li><b>確率の温度調整（荒れやすいレースほど平らに）は効かない。</b>v3 の確率はすでに、荒れ度に応じた散らばりをほぼ織り込んでいる。</li>")
    if ft:
        parts.append("<li><b>予想の特徴量（文脈列）への寄与は下の表のとおり。</b></li>")
    parts.append("<li><b>馬券の幅:</b> 荒れ度の帯で 3連複の点数やフォーメーションの広さを変えても、同じ平均点数の固定戦略に対して回収率・最大ドローダウンの改善は確認できなかった（reality check で有意でない）。一方、<b>買い目を『3連複の予測確率の上位 n 点』で選ぶと、同じ点数の箱・フォーメーションより回収率が高い</b>（これは荒れ度とは別の効果）。</li>")
    parts.append("</ul>")
    # 事後指標
    if d:
        cs = d["corr_spearman"]["vol"]
        parts.append("<h2>1. 荒れの指標（事後）とその関係</h2><p>荒れ度 <code>vol</code> は、標準化した log 3連複配当・log 勝ち馬の人気・log 3着以内の平均人気・−(人気と着順の順位相関) の平均（標準化は 2020〜21 年で固定）。</p>")
        names = {"log_pay_sanrenpuku": "log 3連複配当", "log_pay_sanrentan": "log 3連単配当", "log_pay_umaren": "log 馬連配当", "log_pay_tansho": "log 単勝配当", "log_win_pop": "log 勝ち馬の人気", "log_mean_pop_top3": "log 3着以内の平均人気",
                 "rho": "人気と着順の順位相関", "fav3_all_top3": "上位人気 3 頭が全て 3 着以内", "n_long_top3": "6 番人気以下の 3 着以内の頭数", "manba": "3連複 1 万円以上"}
        parts.append(table(["指標", "vol との順位相関"], [[esc(names[k]), fmt(cs[k], 3, True)] for k in names], {1}))
        parts.append(f"<p class='muted'>全体: 3連複が 1 万円以上のレースは {d['overall']['manba_rate']:.1%}、上位人気 3 頭が全て 3 着以内は {d['overall']['fav3_all_top3']:.1%}、勝ち馬の人気の中央値は {d['overall']['win_pop_median']:.0f} 番人気。</p>")
        parts.append("<h2>2. どんな条件で荒れるか</h2><p>各条件の荒れ度 <code>vol</code> の平均（0 が全体の目安）と、3連複 1 万円以上の割合。Kruskal–Wallis の p 値はいずれも条件と荒れ度に関係があることを示す（日目を除く）。</p>")
        lv = d["levels"]
        for f, title, order in (("fs_band", "出走頭数", ["~11", "12-14", "15-16", "17-18"]), ("dist_band", "距離", ["~1300", "1400-1500", "1600-1900", "2000-2400", "2500~"]), ("surface", "面", None), ("venue_code", "競馬場", None),
                                ("grade", "クラス（格）", None), ("round", "レース番号", None), ("track_condition", "馬場状態", None), ("month", "月", None)):
            rows = lv[f]["table"]
            if order:
                rows = sorted(rows, key=lambda x: order.index(x[f]) if x[f] in order else 99)
            parts.append(f"<h3>{esc(title)}（Kruskal–Wallis p = {lv[f]['kruskal_p']:.1e}）</h3>")
            parts.append(table([title, "レース数", "vol 平均", "上位 20% の割合", "3連複 1 万円以上"],
                               [[esc(VENUE.get(str(x[f]), x[f]) if f == "venue_code" else x[f]), x["count"], fmt(x["mean"], 3, True), f"{x['share_top20']:.1%}", f"{x['manba']:.1%}"] for x in rows], {1, 2, 3, 4}))
        cd = d["courses"]
        top = sorted(cd, key=lambda c: -c["diff"])[:8]
        bot = sorted(cd, key=lambda c: c["diff"])[:8]
        parts.append(f"<h3>コース別（走 120 レース以上の {d['n_course_tests']} コース。他のレースとの Welch t 検定、BH で FDR 補正）</h3>")
        for title, lst in (("荒れやすいコース", top), ("堅いコース", bot)):
            parts.append(f"<p><b>{title}</b></p>" + table(["コース", "レース数", "vol の差", "3連複 1 万円以上", "上位人気 3 頭が全て 3 着以内", "q（BH）"],
                                                           [[esc(course_label(c['course'])), c["n"], fmt(c["diff"], 3, True), f"{c['manba']:.1%}", f"{c['fav3_all_top3']:.1%}", f"{c['q_bh']:.4f}"] for c in lst], {1, 2, 3, 4, 5}))
        parts.append(f"<p class='muted'>コースの荒れ度は年で安定している: 2020〜21 年と 2022〜24 年のコース別平均の Spearman 相関 {d['course_vol_year_stability_spearman']:.2f}。</p>")
    # 事前予測
    if pr:
        parts.append("<h2>3. 事前に荒れやすさを予測できるか</h2><p>レース単位の LightGBM（年ごとの expanding OOF。valid 2022〜2024）。モデル用 = レース条件と出走メンバーの能力の散らばり（オッズ・人気なし）、馬券用 = それに確定オッズの構造（1 番人気のオッズ、オッズのエントロピー、上位 3 頭の差 など）を足したもの。</p>")
        rows = []
        for t, label in (("vol", "荒れ度（合成）"), ("log_pay_sanrenpuku", "log 3連複配当"), ("log_win_pop", "log 勝ち馬の人気"), ("neg_rho", "−（人気と着順の順位相関）")):
            m, k = pr["model"]["targets"][t], pr["market"]["targets"][t]
            rows.append([esc(label), fmt(m["r2"], 3), fmt(m["spearman"], 3), fmt(k["r2"], 3), fmt(k["spearman"], 3)])
        parts.append(table(["目的変数", "モデル用 R²", "モデル用 Spearman", "馬券用 R²", "馬券用 Spearman"], rows, {1, 2, 3, 4}))
        m, k = pr["model"]["targets"]["vol"], pr["market"]["targets"]["vol"]
        parts.append(table(["荒れ上位 20% の拾いやすさ", "AUC", f"予測上位 20% の中の荒れの割合（基準 {100 * m['base_rate']:.1f}%）", "リフト"],
                           [["モデル用", fmt(m["auc_top20"], 3), f"{m['precision_at20']:.1%}", f"{m['lift']:.2f}"], ["馬券用", fmt(k["auc_top20"], 3), f"{k['precision_at20']:.1%}", f"{k['lift']:.2f}"]], {1, 2, 3}))
        imp = pr["model"].get("importance_top", {})
        parts.append("<p class='muted'>モデル用の重要度（分割回数）の上位: " + "、".join(f"{esc(a)}（{b}）" for a, b in list(imp.items())[:8]) + "。出走頭数・クラス・競馬場・メンバーの能力の散らばりが効く。</p>")
    # 予想への寄与
    parts.append("<h2>4. 予想の特徴量への寄与（モデル用の荒れ度のみ）</h2>")
    if ft:
        base = {k: v for k, v in ft["levels"].items()}
        rows = [[esc(k), fmt(v["logloss_win"], 5), fmt(v["logloss_top2"], 5), fmt(v["logloss_top3"], 5)] for k, v in base.items()]
        parts.append("<h3>LogLoss（LightGBM 固定、valid 2022〜2024、2 seed）</h3>" + table(["構成", "勝率", "連対", "複勝"], rows, {1, 2, 3}))
        rows = [[esc(k), fmt(c["Δwin"], 5, True), fmt(c["win_hi"], 5, True), fmt(c["Δtop2"], 5, True), fmt(c["Δtop3"], 5, True), "採用" if c["採用"] else "不採用"] for k, c in ft["compare"].items()]
        parts.append("<h3>増分（95% 信頼区間の上限つき）</h3>" + table(["比較", "Δ勝率", "勝率の上限", "Δ連対", "Δ複勝", "採否"], rows, {1, 2, 3, 4}))
    else:
        parts.append("<p class='muted'>（結果待ち）</p>")
    if tm:
        rows = []
        for kind, label in (("model", "モデル用の荒れ度"), ("market", "馬券用の荒れ度")):
            for pc, nm in (("p_win", "勝率"), ("p_top2", "連対率"), ("p_top3", "複勝率")):
                x = tm[kind]["by_metric"][pc]
                rows.append([esc(label), nm, fmt(x["base"], 5), fmt(x["tempered"], 5), fmt(x["diff"], 5, True), fmt(x["hi"], 5, True)])
        parts.append("<h3>荒れ度による確率の温度調整（v3 の OOF。調整の係数は評価年より前の年だけで当てはめ、評価は 2023・2024 年）</h3>" + table(["荒れ度", "確率", "調整前", "調整後", "差", "95% 上限"], rows, {2, 3, 4, 5}))
        parts.append("<p class='muted'>logit(p)/τ + c（c でレース内の合計をそろえる）、τ = exp(a + b·荒れ度の z)。差が負なら改善。勝率で効果 0.0002 未満・上限が 0 をまたぐので不採用。</p>")
    # 馬券
    if bt:
        parts.append("<h2>5. 馬券の幅への寄与</h2>")
        parts.append(f"<p>1 点 100 円、3連複中心（{bt['n_races']:,} レース、valid 2022〜2024）。買い目の選択は v3 の予測確率だけで決め、オッズは使わない。荒れ度の帯の閾値は評価年より前の年の予測荒れ度の三分位。試した戦略は固定 {sum(1 for t in bt['table'] if t['name'].startswith('F_'))} 通りと荒れ度で幅を変える {bt['reality_check']['same_family']['n_tested']} 通り。</p>")
        t = bt["table"]
        parts.append("<h3>固定戦略</h3>" + table(["戦略", "平均点数", "回収率", "的中率", "最大ドローダウン（総購入額比）", "1 レースの損益の標準偏差（円）"],
                                                 [[esc(x["name"]), f"{x['points_per_bought_race']:.1f}", f"{x['roi']:.3f}", f"{x['hit_rate']:.1%}", f"{x['max_dd_per_stake']:.3f}", f"{x['pl_sd_per_race']:.0f}"] for x in t if x["name"].startswith("F_")], {1, 2, 3, 4, 5}))
        comps = bt["compare_adaptive_vs_matched_fixed"]
        parts.append("<h3>荒れ度で幅を変える戦略（堅い / 中 / 荒れの 3 帯）と、点数が近い固定戦略との比較</h3>")
        parts.append(table(["戦略（堅い|中|荒れ）@荒れ度の種類", "平均点数", "回収率", "同系統の固定", "固定の回収率", "差（95%CI）", "点数が近い最良の固定", "その回収率", "差（95%CI）", "最大DD（戦略 / 同系統）"],
                           [[esc(c["adaptive"]), f"{c['pts_a']:.1f}", f"{c['roi_a']:.3f}", esc(c["bench"]), f"{c['roi_b']:.3f}", f"{c['diff']:+.3f}（{c['ci_lo']:+.3f}〜{c['ci_hi']:+.3f}）", esc(c["strong_bench"]), f"{c['strong_roi']:.3f}",
                             f"{c['diff_strong']:+.3f}（{c['ci_lo_strong']:+.3f}〜{c['ci_hi_strong']:+.3f}）", f"{c['maxdd_a']:.3f} / {c['maxdd_b']:.3f}"] for c in comps], {1, 2, 4, 7}))
        rc = bt["reality_check"]
        parts.append(f"<p>reality check（日単位のブートストラップ、{rc['same_family']['n_tested']} 戦略の最大の差）: 同系統の固定に対して p = {rc['same_family']['p']:.2f}、点数が近い最良の固定に対して p = {rc['strongest_matched']['p']:.2f}。荒れ度で幅を変える効果は有意でない。逆向き（堅いときに広く、荒れのときに狭く）は明確に悪い（R1：差の CI が 0 を含まず負）ので、荒れのときに広げる向きは正しいが、固定に勝つほどではない。</p>")
        bnames = {0: "堅い", 1: "中", 2: "荒れ"}
        rows = []
        for x in bt["roi_by_band_fixed"]:
            if x["strategy"] in ("F_top6", "F_top10", "F_top20", "F_top40", "F_form_2_5_7", "F_form_3_7_11"):
                rows.append([esc(x["vol"]), esc(x["strategy"]), bnames[x["band"]], x["n_races"], f"{x['roi']:.3f}", f"{x['ci'][0]:.3f}〜{x['ci'][1]:.3f}", f"{x['hit']:.1%}"])
        parts.append("<h3>荒れ度の帯ごとの回収率（固定戦略。日単位ブートストラップの 95% 区間）</h3>" + table(["荒れ度", "戦略", "帯", "レース数", "回収率", "95% 区間", "的中率"], rows, {3, 4, 5, 6}))
        parts.append("<p class='muted'>top40（広く買う）は荒れの帯で回収率が上がる傾向（モデル用で堅い 0.77 → 荒れ 0.84）があるが、top10 では帯による差が無い。区間は重なる。</p>")
        cal = bt["trio_calibration"]
        parts.append("<h3>3連複の予測確率の較正（上位 n 点の累積確率と実際の的中率）</h3>" + table(["上位 n 点", "予測の累積確率の平均", "実際の的中率"], [[c["top_n"], f"{c['mean_pred_cum_prob']:.3f}", f"{c['actual_hit']:.3f}"] for c in cal], {0, 1, 2}))
        tk = bt["roi_by_ticket_band"]
        rows = []
        tickets = ["tansho_top1", "fukusho_top1", "umaren_box3", "wide_box3", "sanrenpuku_box5"]
        tn = {"tansho_top1": "単勝（本命 1 点）", "fukusho_top1": "複勝（本命 1 点）", "umaren_box3": "馬連（3 頭ボックス 3 点）", "wide_box3": "ワイド（3 頭ボックス 3 点）", "sanrenpuku_box5": "3連複（5 頭ボックス 10 点）"}
        for vol in ("model", "market"):
            for tkt in tickets:
                v = {x["band"]: x for x in tk if x["vol"] == vol and x["ticket"] == tkt}
                if len(v) == 3:
                    rows.append([vol, tn[tkt]] + [f"{v[b]['roi']:.3f}" for b in (0, 1, 2)])
        parts.append("<h3>券種ごとの回収率（荒れ度の帯別。複勝・ワイドは払戻のセルが連結されて分解できないレースを除く）</h3>" + table(["荒れ度", "券種", "堅い", "中", "荒れ"], rows, {2, 3, 4}))
        parts.append("<p class='muted'>荒れの帯で複勝が下がる（0.73 → 0.68 前後）以外は、券種ごとの一貫した傾向は無い（馬連はモデル用では荒れで上がり、馬券用では下がる）。どの券種・帯でも回収率は 1 を大きく下回り、期待値の向上は見られない。</p>")
    parts.append('<h2>6. 注意・限界</h2><ul><li>2024 年の一部の払戻データは、複勝・ワイドのセルが連結されて分解できない（例: numbers=<code>3168</code>、payout=<code>330120540</code>）。単勝・馬連・3連複・3連単は問題なし。複勝・ワイドは該当レースを除いて集計した。パーサー側の確認が必要。</li>'
                 '<li>馬券用の荒れ度は確定オッズで代用している。購入前（T-60/T-15）のオッズの蓄積後に再評価する。</li>'
                 '<li>試した戦略数は 16（荒れ度で幅を変える）＋固定 20 弱。多重検定は reality check で見ている。</li>'
                 '<li>3連複の予測確率は上位 n 点の累積確率が実際の的中率よりやや低い（過小評価）が、較正は概ね取れている。</li></ul>')
    parts.append('<p class="muted">コード: <code>src/pipeline/features/tp30_volatility.py</code>、<code>src/scripts/data/tp30_eval.py</code>、<code>tp30_bet_eval.py</code>、<code>tp30_report.py</code>。結果の生データ: <code>data/local/modeling/training_pipeline/tp30/*.json</code>。</p></body></html>')
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data/local/modeling/training_pipeline/tp30"))
    a = ap.parse_args()
    OUT.write_text(build(Path(a.src)), encoding="utf-8")
    print("保存:", OUT)


if __name__ == "__main__":
    main()
