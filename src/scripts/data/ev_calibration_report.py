"""期待値（EV）の水準の監査（audit_ev_calibration.py の結果）を docs の 1 章（ev_calibration_audit.html）にする。

python -m src.scripts.data.ev_calibration_report --dir <ev_audit の出力ディレクトリ>
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

from src.scripts.data import tp29_report as r29

ROOT = Path(__file__).resolve().parents[3]
CHAPTER = ROOT / "docs/html/modeling/chapters/ev_calibration_audit.html"
DEFAULT_DIR = ROOT / "data/local/modeling/betting/ev_audit"
BT_JP = {"tansho": "単勝", "fukusho": "複勝", "umaren": "馬連", "wide": "ワイド", "umatan": "馬単", "sanrenpuku": "3連複", "sanrentan": "3連単", "trio": "3連複", "tierce": "3連単"}
RET = {"tansho": 0.80, "fukusho": 0.80, "umaren": 0.775, "wide": 0.775, "umatan": 0.75, "sanrenpuku": 0.75, "sanrentan": 0.725}


def _f(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and x != x) else f"{x:.{nd}f}"


def _p(x, nd=1):
    return "-" if x is None or (isinstance(x, float) and x != x) else f"{100 * x:.{nd}f}%"


def _load(d: Path, name: str) -> dict:
    p = d / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _cal_tbl(rows: list[dict]) -> str:
    out = []
    for r in rows:
        if not r.get("n"):
            continue
        ci = f" [{_f(r.get('roi_lo'), 2)}, {_f(r.get('roi_hi'), 2)}]" if "roi_lo" in r else ""
        out.append([html.escape(r["bin"]), f"{r['n']:,}", _f(r["mean_ev"], 2), f"<b>{_f(r['roi'], 2)}</b>{ci}"])
    return r29._tbl(["EV の帯", "賭け数", "平均 EV", "実現回収率 [95%区間]"], out, ["", "n", "n", "n"])


def build(d: Path) -> None:
    idt, hs, cb, ex, sg, pf, sgf = (_load(d, n) for n in ("identities", "horse", "calib", "exotic", "suggest", "portfolio", "suggest_fixed"))
    o = idt["odds_side"]
    # 1) オッズ側の恒等式
    rows = []
    for key, label, exp in (("win", "単勝 Σ(1/odds)", 1 / 0.8), ("umaren", "馬連 Σ(1/odds)", 1 / 0.775), ("umatan", "馬単 Σ(1/odds)", 1 / 0.75),
                            ("place_min", "複勝 Σ(1/下限)", 3 / 0.8), ("place_mid", "複勝 Σ(1/中央)", 3 / 0.8), ("place_max", "複勝 Σ(1/上限)", 3 / 0.8),
                            ("wide_odds_min", "ワイド Σ(1/下限)", 3 / 0.775), ("wide_odds_max", "ワイド Σ(1/上限)", 3 / 0.775)):
        v = o[key]
        rows.append([label, _f(exp, 3), _f(v["mean"], 3), _f(v["ratio_to_expected"], 3), f"{_f(v['p5'], 2)}〜{_f(v['p95'], 2)}", _p(v["within_tol"], 1)])
    odds_tbl = r29._tbl(["指標", "期待値（払戻率から）", "実測の平均", "実測÷期待", "5〜95%点", "±5% 以内のレース"], rows, ["", "n", "n", "n", "n", "n"])
    # 2) 全買い
    br = idt["blanket_roi"]
    blk_tbl = r29._tbl(["券種", "払戻率", "全組合せを 1 点ずつ買った回収率"] + sorted(next(iter(br.values()))["by_year"]),
                       [[BT_JP[k], _p(v["expected"], 1), f"<b>{_p(v['all'], 1)}</b>"] + [_p(v["by_year"][y], 1) for y in sorted(v["by_year"])] for k, v in br.items()],
                       ["", "n", "n"] + ["n"] * len(next(iter(br.values()))["by_year"]))
    # 3) 水準: 単勝・複勝・連対
    lv = hs["levels"]
    lv_rows = []
    for key, label, note in (("ev_win", "単勝 EV（勝率×単勝オッズ）", "本番と同じ"), ("ev_place_mid", "複勝 EV（中央値）", "修正前（本番）"), ("ev_place_eff", "複勝 EV（実効オッズ）", "修正後"),
                             ("ev_top2_heur", "連対 EV（単勝オッズ×1.65）", "修正前（本番）"), ("ev_top2_new", "連対 EV（馬連プールと整合）", "修正後")):
        v = lv[key]
        lv_rows.append([label, note, _f(v["mean"], 3), _f(v["median"], 3), _p(v["share_ge_1.0"], 1), _p(v["share_ge_1.2"], 1), _p(v["share_ge_1.5"], 1)])
    lv_tbl = r29._tbl(["EV の種類", "状態", "全馬の平均", "中央値", "EV≥1.0 の馬", "EV≥1.2", "EV≥1.5"], lv_rows, ["", "", "n", "n", "n", "n", "n"])
    t2 = hs["top2_heuristic_vs_market"]
    band_tbl = r29._tbl(["単勝オッズ帯", "馬数", "『×1.65』÷市場の公正な連対オッズ"], [[k, f"{v['n']:,}", f"<b>{_f(v['ratio_mean'], 2)} 倍</b>"] for k, v in t2["by_win_odds_band"].items()], ["", "n", "n"])
    fit = cb["top2_market_harville_fit"]
    fit_tbl = r29._tbl(["λ（市場確率 q^λ）", "log 誤差の RMSE", "馬連プールの連対確率 ÷ Harville の平均"], [[k, _f(v["rmse_log"], 4), _f(v["mean_ratio_pool_over_harville"], 3)] for k, v in fit.items()], ["n", "n", "n"])
    pp, pw = cb["place_payout_position"], cb["place_w_sum_identity"]
    # 4) 組合せ券
    ex_rows = []
    for mode, ml in (("model", "本番の確率（V4 + 割引 Harville）"), ("harville_market", "市場の単勝確率（α=1）")):
        for bt in ("umaren", "wide", "umatan"):
            v = ex["modes"][mode][bt]
            ex_rows.append([ml, BT_JP[bt], f"{v['n_bets']:,}", _f(v["mean_ev_equal_weight"], 3), _f(v["median_ev"], 3), f"<b>{_f(v['realized_blanket_roi'], 3)}</b>",
                            _p(v["share_ev_ge_1.2"], 1), _p(v["share_ev_ge_1.5"], 1)])
    ex_tbl = r29._tbl(["確率の入力", "券種", "組合せ数", "平均 EV", "中央値", "全買いの実現回収率", "EV≥1.2", "EV≥1.5"], ex_rows, ["", "", "n", "n", "n", "n", "n", "n"])
    cal_blocks = "".join(f"<h3>{BT_JP[bt]}（本番の確率 × 確定オッズ、払戻率 {_p(RET[bt], 1)}）</h3>" + _cal_tbl(ex["modes"]["model"][bt]["calibration"]) for bt in ("umaren", "wide", "umatan"))
    # 5) 画面の指標
    sg_rows = [[BT_JP.get(k, k), f"{v['n']:,}", _f(v["mean_claimed_ev"], 2), f"<b>{_f(v['realized_roi'], 2)}</b>"] for k, v in sg["by_bet_type"].items()]
    sg_tbl = r29._tbl(["買い目提案の券種", "提案数", "提案の平均 EV", "実現回収率"], sg_rows, ["", "n", "n", "n"])
    sgc_tbl = r29._tbl(["推奨度", "提案数", "提案の平均 EV", "実現回収率"], [[k, f"{v['n']:,}", _f(v["mean_claimed_ev"], 2), _f(v["realized_roi"], 2)] for k, v in sg["by_confidence"].items()], ["", "n", "n", "n"])
    sgf_tbl = r29._tbl(["買い目提案の券種", "提案数", "提案の平均 EV", "実現回収率"], [[BT_JP.get(k, k), f"{v['n']:,}", _f(v["mean_claimed_ev"], 2), _f(v["realized_roi"], 2)] for k, v in sgf["by_bet_type"].items()], ["", "n", "n", "n"])
    sgfc_tbl = r29._tbl(["推奨度", "提案数", "提案の平均 EV", "実現回収率"], [[k, f"{v['n']:,}", _f(v["mean_claimed_ev"], 2), _f(v["realized_roi"], 2)] for k, v in sgf["by_confidence"].items()], ["", "n", "n", "n"])
    lvl_jp = {"hot": "オレンジ（150% 以上）", "good": "緑（125% 以上）", "ok": "青（105% 以上）"}
    pf_rows = [[lvl_jp.get(k, k), f"{v['n_races']:,}", _p(v["mean_claimed_roi"], 0), f"<b>{_p(v['realized_roi'], 1)}</b>"] for k, v in pf["by_level"].items()]
    pf_tbl = r29._tbl(["レース価値の色（絶対水準）", "レース数", "推奨ポートフォリオの期待回収率（平均）", "実現回収率"], pf_rows, ["", "n", "n", "n"])
    pfc_tbl = _cal_tbl(pf["candidates_calibration"])
    # 6) 縮約
    k_h, k_ex, k_pf = hs["shrink_k_all_years"], {bt: ex["modes"]["model"][bt]["shrink_k_all_years"] for bt in ("umaren", "wide", "umatan")}, pf.get("shrink_k_all_years", {})
    k_port = pf.get("portfolio_race_level", {}).get("shrink_k_all_years")
    k_rows = [["単勝（馬単位）", _f(k_h["win"], 3), "0.800"], ["複勝（馬単位）", _f(k_h["place"], 3), "0.800"]] + \
             [[f"{BT_JP[b]}（全組合せ）", _f(k, 3), _f(RET[b], 3)] for b, k in k_ex.items()] + [[f"推奨ポートフォリオ内の{BT_JP[b]}", _f(k, 3), _f(RET[b], 3)] for b, k in k_pf.items()] + \
             [["推奨ポートフォリオ（レース単位の期待回収率）", _f(k_port, 3), "0.800"]]
    k_tbl = r29._tbl(["対象", "縮約係数 k（実現−払戻率 = k×(EV−払戻率)）", "払戻率"], k_rows, ["", "n", "n"])

    def need(k, base, target):
        return "-" if not k else f"{base + (target - base) / k:.2f}"

    thr_rows = []
    for lab, k, base in (("単勝", k_h["win"], 0.8), ("複勝", k_h["place"], 0.8), ("馬連", k_ex["umaren"], 0.775), ("ワイド", k_ex["wide"], 0.775), ("推奨ポートフォリオ", k_port, 0.8)):
        thr_rows.append([lab, need(k, base, 1.05), need(k, base, 1.2), need(k, base, 1.5)])
    thr_tbl = r29._tbl(["対象", "縮約後 1.05 に相当する元の EV", "縮約後 1.2 に相当", "縮約後 1.5 に相当"], thr_rows, ["", "n", "n", "n"])
    leg = idt["legacy_softmax_of_logit"]
    style = re.search(r"<style>.*?</style>", r29.STYLE_SRC.read_text(encoding="utf-8"), re.S).group(0)
    body = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>期待値の水準の監査（控除率との整合）— keiba-vpn</title>
{style}</head>
<body><a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>期待値の水準の監査（控除率との整合）</h1>
<p class="muted">ユーザーの疑問（2026-10-20）: 複合馬券の期待値が本当に控除率を加味した水準（約 0.8）になっているか、高く設定されていないか。2022〜2025 年の {idt['n_races']:,} レース（V4 の leave-one-year-out と bundle_v4_2024 の予測、確定オッズ、確定払戻）で調べた。オッズは締切後の確定オッズで、購入時点の値ではない（tp-36 と同じ限界）。</p>
<div class="note"><b>要点.</b>
 <ul>
 <li><b>オッズ側は控除率どおりだった.</b>確定の単勝は Σ(1/オッズ) = {_f(o['win']['mean'], 3)}（期待 1.25）、馬連 {_f(o['umaren']['mean'], 3)}（1.290）、馬単 {_f(o['umatan']['mean'], 3)}（1.333）。確率側も勝率・連対率・複勝率・各組合せの総和が理論値に一致した（誤差 0）。<b>『期待値の合計』を計算する式そのものに誤りは無い</b>。</li>
 <li><b>『全部の期待値の平均が 0.8』にならないのは正常（市場とモデルが違うから）.</b>モデルの確率が市場の確率と一致するときだけ EV は払戻率になる。モデルが市場より尖っている（人気薄の確率が市場より高いことが多い）と、全組合せを均等に見た平均 EV は払戻率より高く出る。実際に全組合せを 1 点ずつ買った回収率は、単勝 {_p(br['tansho']['all'])}・馬連 {_p(br['umaren']['all'])}・ワイド {_p(br['wide']['all'])}・3連複 {_p(br['sanrenpuku']['all'])} で、払戻率（80/77.5/75%）より低い（人気薄の過剰な人気＝favorite–longshot bias）。市場の確率を入れたときの馬連 EV の平均 {_f(ex['modes']['harville_market']['umaren']['mean_ev_equal_weight'], 2)} は、この実現回収率 {_f(ex['modes']['harville_market']['umaren']['realized_blanket_roi'], 2)} とほぼ一致する（計算は正しい）。</li>
 <li><b>水準が高く出ていた箇所（明らかな誤り 2 つ）.</b>(1) 連対 EV = 連対率 ×（単勝オッズ × 1.65）は、市場の馬連プールと整合する連対の公正オッズの平均 {_f(t2['overall_ratio_mean'], 2)} 倍で、全馬の平均が {_f(lv['ev_top2_heur']['mean'], 2)}、<b>{_p(lv['ev_top2_heur']['share_ge_1.0'], 0)} の馬が EV≥1</b> だった。これが買い推奨度（最大の EV が 1.2 以上で『強推奨』）と買い目提案（馬連・馬単）の EV に入り、<b>ほぼ全馬が強推奨、提案の {_p(sg['by_confidence']['high']['n'] / sg['n_picks'], 0)} が『高』</b>になっていた。(2) 複勝 EV の払戻に使う (下限+上限)/2 は、実際の払戻より平均で約 9% 高い（複勝の Σ(1/中央) = {_f(o['place_mid']['mean'], 2)} ＜ 3.75）。</li>
 <li><b>モデルの EV が実現回収率より高いのは、直したのとは別の問題（モデルが市場に勝てていない）.</b>本番の確率 × 確定オッズの EV は、馬連で平均 {_f(ex['modes']['model']['umaren']['mean_ev_equal_weight'], 2)}、実現は {_f(ex['modes']['model']['umaren']['realized_blanket_roi'], 2)}。EV が 1.05〜2 の帯の実現回収率は {_p(ex['modes']['model']['umaren']['calibration'][3]['roi'], 0)}〜{_p(ex['modes']['model']['umaren']['calibration'][5]['roi'], 0)} で、EV に比例して上がらない（縮約係数 k は単勝 {_f(k_h['win'], 2)}・複勝 {_f(k_h['place'], 2)}・馬連 {_f(k_ex['umaren'], 2)}・ワイド {_f(k_ex['wide'], 2)}）。</li>
 <li><b>画面の指標は絶対水準が高い.</b>レース価値の色分けで『オレンジ』（期待回収率 150% 以上）が {_p(pf['by_level']['hot']['n_races'] / pf['n_races'], 0)} のレースに付き、その平均の期待回収率 {_p(pf['by_level']['hot']['mean_claimed_roi'], 0)} に対し実現は {_p(pf['by_level']['hot']['realized_roi'], 0)}。順序（オレンジ＞緑＞青）は保たれるが、水準は信じられない。</li>
 <li><b>対応.</b>(1)(2) は修正した（連対は『馬連プールと整合する公正オッズ』、複勝は実効オッズ = 下限 + 0.3×(上限−下限)）。縮約（EV_adj = 払戻率 + k×(EV − 払戻率)）は設定ファイル config/ev_calibration.json で用意したが、<b>既定は無効</b>（有効にすると買い推奨度とレース価値の判定が縮約後の値になる）。</li>
 </ul></div>
<h2>1. オッズ側の恒等式（確定オッズが控除率どおりか）</h2>
<p class="muted">プールの全組合せについて、払戻率 R のとき Σ(1/オッズ) = 1/R（ワイドと複勝は的中が 3 通りなので 3/R）。</p>
{odds_tbl}
<p>単勝は年が新しいほど 1.265 に上がる（2022・23 年 1.255）が、いずれも 1.25 の ±1.2% 以内。複勝は『下限』で見ると 4.46 と高く、『上限』で 2.87 と低い。確定の払戻は下限と上限の間にある（的中時の払戻の位置は下限から {_f(pp['mean_w'], 2)}、Σ(1/オッズ) = 3.75 を満たす位置は {_f(pw['w'], 2)}）。</p>
<h2>2. 確率側の恒等式</h2>
<p>V4 の勝率・連対率・複勝率の合計と、本番の割引 Harville（連対率・複勝率へ α を当てはめたもの）の馬連・ワイド・3連複・馬単の総和は、すべて理論値（1・2・3・1・3・1・1）に一致した（{idt['prob_identities']['wide']['max_abs_dev']} の誤差）。一方、旧経路（app.py の <code>_compute_composite_scores</code>）は勝率を softmax し直すため、<code>pred_score</code> に logit(勝率) を入れる v3/v4 の保存形式を通すと、本命の勝率が平均 {_f(leg['favorite_prob_ratio_mean'], 2)} 倍に膨らむ（平均の絶対誤差 {_f(leg['mean_abs_diff_win_prob'], 3)}）。この経路は影モード（shadow）の表示には使われておらず（local_predictions は勝率をそのまま使う）、v3/v4 の予測を『predict』で出すときに通さないよう注意が要る。また v3/v4 の保存形式（<code>v3_payload</code>）は連対率・複勝率を <code>place_prob</code> / <code>show_prob</code> に入れるが、推奨ポートフォリオ（BettingOptimizer）は <code>top2_prob</code> / <code>top3_prob</code> を探していたため、predict に切り替えると α が連対率・複勝率に当てはまらず既定値になる状態だった。<b>両方のキー名を読むように直した</b>（本章の検証は連対率・複勝率へ当てはめる本来の形）。</p>
<h2>3. 全組合せを買った回収率（払戻率との差）</h2>
<p class="muted">均等に 1 点ずつ全買いした回収率。市場の確率で賭け金を比例配分すれば払戻率になるが、均等だと人気薄に厚くなるので払戻率より低い。EV の『平均』を見るときの基準はこちら。</p>
{blk_tbl}
<h2>4. 単勝・複勝・連対の EV の水準</h2>
<p class="muted">V4 の予測確率 × 確定オッズ。複勝は下限・中央・実効の 3 通り。『実現』は的中時の実際の払戻で精算。</p>
{lv_tbl}
<h3>連対 EV の『単勝オッズ × 1.65』は市場の連対オッズの何倍か</h3>
<p class="muted">馬連の確定オッズから作る、市場の連対確率（0.775×Σ(1/オッズ)）に整合する公正な連対オッズとの比。</p>
{band_tbl}
<p>単勝オッズの市場確率から連対確率を作る Harville の指数 λ は、λ = 1（素の Harville）だと馬連プールの連対確率との比が {_f(fit['1.0']['mean_ratio_pool_over_harville'], 2)}（人気薄ほど過小）で、λ = {cb['top2_best_lambda']} のとき誤差が最小。</p>
{fit_tbl}
<h3>単勝 EV の較正曲線</h3>
{_cal_tbl(hs['calibration']['ev_win'])}
<h3>複勝 EV の較正曲線（実効オッズ）</h3>
{_cal_tbl(hs['calibration']['ev_place_eff'])}
<h2>5. 組合せ券（馬連・ワイド・馬単）: 本番の確率 × 確定オッズ</h2>
<p class="muted">全組合せを対象にした EV の水準。『市場の単勝確率（α=1）』の行は検算（市場どうしなら EV は全買いの回収率に近いはず）。ワイドは下限オッズで EV を計算（本番と同じ）。3連複は過去の確定オッズが無いので対象外。</p>
{ex_tbl}
{cal_blocks}
<h2>6. 画面の指標への影響</h2>
<h3>買い目提案（suggest_race_bets、修正前の本番の定義）</h3>
<p class="muted">連対 EV の誤りが入った状態の結果。馬連の『EV』は◎と○の連対 EV の平均で、実在する馬連の期待値ではない。</p>
{sg_tbl}
{sgc_tbl}
<p><b>修正後</b>（連対 EV・複勝 EV の修正と、印の新ルール＝◎○▲★各 1 頭＋△複数。提案 {sgf['n_picks']:,} 件）。馬連の平均 EV は {_f(sg['by_bet_type']['umaren']['mean_claimed_ev'], 2)} から {_f(sgf['by_bet_type']['umaren']['mean_claimed_ev'], 2)} に下がり、実現回収率に近づいた。『高』の割合は {_p(sg['by_confidence']['high']['n'] / sg['n_picks'], 0)} から {_p(sgf['by_confidence']['high']['n'] / sgf['n_picks'], 0)} に下がったが、まだ多い（閾値が元の EV の尺度で置かれているため。縮約を有効にしたときの閾値は 7 章）。3連複・3連単の提案は、実在しない EV（構成馬の複勝 EV の平均、◎の勝 EV）で出ている近似で、実現回収率が高く見えるのは件数が少なく重い裾が効いているため（参考値）。</p>
{sgf_tbl}
{sgfc_tbl}
<h3>レース価値の色分け（推奨ポートフォリオの期待回収率、上限 2,000 円/R）</h3>
{pf_tbl}
<h3>推奨ポートフォリオの買い目の較正曲線</h3>
{pfc_tbl}
<h2>7. 縮約（較正）</h2>
<p>実現回収率 − 払戻率 = k × (EV − 払戻率) の k を、原点を通る最小二乗で当てはめた（2022〜2025 年の全体。評価年より前の年だけで当てはめた walk-forward の k も <code>portfolio.json</code> に残してある）。k が 1 なら EV は信頼でき、0 なら EV に情報が無い。</p>
{k_tbl}
<p>縮約後に買い推奨の閾値を越えるために、元の EV がいくつ要るか:</p>
{thr_tbl}
<p>例えば単勝で縮約後に『推奨』（1.05）になるには、元の EV が {need(k_h['win'], 0.8, 1.05)} 要る。現在の閾値（買い推奨度 1.05 / 1.2、買い目 0.95〜1.1）は、元の EV がそのまま実現回収率に近い前提で置かれている。</p>
<h2>8. 直したことと、直していないこと</h2>
<ul>
 <li><b>直した（既定で有効）</b>: 連対 EV を『馬連プールと整合する公正オッズ』（<code>market_top2_fair_odds</code>、λ = {cb['top2_best_lambda']}、払戻率 0.775）に。複勝 EV の払戻を実効オッズ（<code>place_effective_odds</code>、下限 + 0.3×(上限−下限)）に。画面の API には <code>place_odds_eff</code> を追加し、<code>place_odds_avg</code>（表示用の平均）は残した。</li>
 <li><b>直した（キー名）</b>: <code>_RankContext</code> が <code>place_prob</code> / <code>show_prob</code> も連対率・複勝率として読む（v3/v4 の保存形式で α を当てはめるため）。</li>
 <li><b>用意した（既定で無効）</b>: <code>src/utils/ev_calibration.py</code> と <code>config/ev_calibration.json</code>。有効にすると <code>buy_recommendation_tier</code>（買い推奨度）と <code>value_level</code>（レース価値の色）が縮約後の EV で判定する。</li>
 <li><b>直していない</b>: 買い目提案（<code>race_bet_suggestion</code>）の組合せ券の『EV』は、構成馬の単独 EV の平均という近似のまま（実オッズが無い時点で出すため）。推奨ポートフォリオ（BettingOptimizer）は実オッズ × 割引 Harville の正しい EV。EV の高い順に上位を選ぶ以上、選ばれた買い目の EV は選択バイアスで高めに出る（winner's curse）。</li>
</ul>
<h2>9. 限界</h2>
<ul>
 <li>オッズは締切後の確定オッズ。購入時点のオッズ（T-60・T-15）で出る EV は、確定オッズに向かって動く分だけ異なる。</li>
 <li>3連複・3連単は過去の確定オッズが無く、較正できていない（3連単のバックフィル後に再検証）。</li>
 <li>k は 2022〜2025 年の当てはめで、モデルが更新（v5 など）されれば変わる。有効にする前に、更新後の予測で当てはめ直すこと。</li>
</ul>
<p class="muted">関連: <a href="exotic_bet_probabilities.html">組合せ券の確率の作り方</a> / <a href="betting_edge_search.html">馬券で回収率 100% を超える余地の探索（tp-36）</a> / <a href="mark_rules.html">印の付け方</a> / 実行: <code>python -m src.scripts.data.audit_ev_calibration identities|horse|calib|exotic|suggest|portfolio</code>、章は <code>python -m src.scripts.data.ev_calibration_report</code>。</p>
</body></html>
"""
    CHAPTER.write_text(body, encoding="utf-8")
    print("保存:", CHAPTER)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    build(Path(ap.parse_args().dir))


if __name__ == "__main__":
    main()
