"""期待値の較正の検証（calibrate_ev.py の結果）を docs の 1 章（ev_calibration_adopt.html）にする。

python -m src.scripts.data.calibrate_ev_report --dir <calibrate_ev の出力ディレクトリ（final.json / fit.json / eval2026.json / production.json）>
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

from src.scripts.data import tp29_report as r29

ROOT = Path(__file__).resolve().parents[3]
CHAPTER = ROOT / "docs/html/modeling/chapters/ev_calibration_adopt.html"
CONFIG = ROOT / "config/ev_calibration.json"
JP = {"win": "単勝（馬ごとの EV）", "place": "複勝（馬ごとの EV）", "top2": "連対（馬ごとの EV）", "tansho": "単勝（推奨の買い目）", "fukusho": "複勝（推奨の買い目）",
      "umaren": "馬連（推奨の買い目）", "wide": "ワイド（推奨の買い目）"}
METHOD_JP = {"raw": "較正なし", "shrink": "縮約 base+k(EV−base)", "linear": "線形 a+b·EV", "isotonic": "単調回帰", "const": "定数（実績の平均）", "market": "市場並み（払戻率）"}


def _f(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and x != x) else f"{x:.{nd}f}"


def _p(x, nd=0):
    return "-" if x is None or (isinstance(x, float) and x != x) else f"{100 * x:.{nd}f}%"


def _j(d: Path, name: str) -> dict:
    p = d / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _method_tbl(fit_kind: dict) -> str:
    rows = []
    for m in ("raw", "shrink", "linear", "isotonic", "const", "market"):
        r = (fit_kind.get(m) or {}).get("pooled")
        if not r or "gap" not in r:
            continue
        rows.append([METHOD_JP[m], f"{r['n']:,}", _f(r.get("mse"), 3), f"{r['gap']:+.3f}", _f(r["ece"], 3), r["cover"],
                     " / ".join(f"{y}:{fit_kind[m][y]['gap']:+.2f}" for y in ("2023", "2024", "2025") if y in fit_kind[m] and "gap" in fit_kind[m][y])])
    return r29._tbl(["方式", "n", "MSE", "ずれ（表示−実現）", "較正誤差 ece", "帯のうち区間が表示を含む", "年別のずれ 2023/24/25"], rows, ["", "n", "n", "n", "n", "n", "n"])


def _bins_tbl(bins: list[dict]) -> str:
    return r29._tbl(["表示した値（帯の平均）", "実現回収率", "95% 区間", "n"], [[_f(b["pred"], 3), f"<b>{_f(b['real'], 3)}</b>", f"{_f(b['lo'], 2)}〜{_f(b['hi'], 2)}", f"{b['n']:,}"] for b in bins], ["n", "n", "n", "n"])


def _lv_tbl(rows: list[dict], label: str) -> str:
    out = []
    for r in rows:
        if r.get("share") is None:
            out.append([f"≥ {r['min']:.2f}", f"{r['n']:,}", "-", "-", "-", "-"])
        else:
            out.append([f"≥ {r['min']:.2f}", f"{r['n']:,}（{_p(r['share'], 1)}）", _f(r["pred"], 3), f"<b>{_f(r['real'], 3)}</b>", f"{_f(r['lo'], 2)}〜{_f(r['hi'], 2)}", ""])
    return r29._tbl([label, "対象数", "表示の平均", "実現回収率", "95% 区間", ""], out, ["", "n", "n", "n", "n", ""])


def build(d: Path) -> None:
    fit, fin, ev26, prod = _j(d, "fit.json"), _j(d, "final.json"), _j(d, "eval2026.json"), _j(d, "production.json")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    sel = fin["selection"]
    # 1) 馬ごと
    horse_tbls = "".join(f"<h3>{JP[k]}</h3>" + _method_tbl(fit["horse"][k]) +
                         f"<p class='muted'>選んだ方式: <b>{METHOD_JP[sel['horse'][k][0]]}</b>（基準 {'を満たす' if sel['horse'][k][1].get('ok') else 'を満たさない'}）。2026 年: 表示−実現 {ev26['horse'][k]['calibrated']['gap']:+.3f}"
                         f"（較正なしは {ev26['horse'][k]['raw']['gap']:+.3f}）、ece {_f(ev26['horse'][k]['calibrated']['ece'], 3)}。</p>" for k in ("win", "place", "top2"))
    hb = fin["horse_best_calibration"]
    hb26 = ev26["horse_best"]
    # 2) 買い目
    cand_tbls = "".join(f"<h3>{JP[k]}</h3>" + _method_tbl(fit["cand"][k]) +
                        f"<p class='muted'>選んだ方式: <b>{METHOD_JP[sel['cand'][k][0]]}</b>（基準 {'を満たす' if sel['cand'][k][1].get('ok') else 'を満たさない'}）。2026 年: 表示−実現 {ev26['cand'][k]['calibrated']['gap']:+.3f}"
                        f"（較正なしは {ev26['cand'][k]['raw']['gap']:+.3f}）。</p>" for k in ("tansho", "fukusho", "umaren", "wide"))
    rc = fin["race_composed"]
    race_rows = [[y, f"{rc[y]['n']:,}", f"{rc[y]['gap']:+.3f}", _f(rc[y]["ece"], 3), rc[y]["cover"]] for y in ("2023", "2024", "2025", "pooled")]
    race_rows.append(["2026（未使用）", f"{ev26['race']['n']:,}", f"{ev26['race']['calibrated']['gap']:+.3f}", _f(ev26["race"]["calibrated"]["ece"], 3), ev26["race"]["calibrated"]["cover"]])
    race_tbl = r29._tbl(["年", "レース数", "ずれ（表示−実現）", "ece", "帯のうち区間が表示を含む"], race_rows, ["", "n", "n", "n", "n"])
    yearly = prod["yearly_realized"]
    yearly_tbl = r29._tbl(["年", "推奨ポートフォリオの実現回収率（購入額加重）"], [[y, f"<b>{_p(v, 1)}</b>"] for y, v in yearly.items()], ["", "n"])
    raw = fin["race_raw_claimed"]
    # 3) 水準
    tier_rows = []
    for t, name in ((cfg["tiers"]["watch"], "様子見（≥ 0.90）"), (cfg["tiers"]["recommend"], "推奨（≥ 1.00）"), (cfg["tiers"]["strong"], "強推奨（≥ 1.10）")):
        r = next((x for x in fin["horse_levels_pooled"] if abs(x["min"] - t) < 1e-9), None)
        r26 = next((x for x in ev26["horse_tiers"] if abs(x["min"] - t) < 1e-9), None)
        if r and r.get("share") is not None:
            tier_rows.append([name, f"{_p(r['share'], 1)}", _f(r["pred"], 2), f"<b>{_f(r['real'], 2)}</b> [{_f(r['lo'], 2)}, {_f(r['hi'], 2)}]",
                              "-" if not r26 or r26.get("share") is None else f"<b>{_f(r26['real'], 2)}</b> [{_f(r26['lo'], 2)}, {_f(r26['hi'], 2)}]（{_p(r26['share'], 1)}）"])
    tier_tbl = r29._tbl(["買い推奨度（較正後の EV の最大）", "該当の馬", "表示の平均", "実現（2023〜25 年 walk-forward）", "実現（2026 年）"], tier_rows, ["", "n", "n", "n", "n"])
    lvl_rows = []
    for name, t in (("オレンジ hot（≥ %.2f、かつその日の上位 20%%）" % cfg["levels"]["hot"], cfg["levels"]["hot"]), ("緑 good（≥ %.2f）" % cfg["levels"]["good"], cfg["levels"]["good"]),
                    ("青 ok（≥ %.2f）" % cfg["levels"]["ok"], cfg["levels"]["ok"])):
        row = [name]
        for y in ("2022", "2023", "2024", "2025", "2026"):
            r = next((x for x in prod["levels_by_year"].get(y, []) if abs(x["min"] - t) < 1e-9), None)
            row.append("-" if not r or r.get("real") is None else f"{_p(r['real'], 0)}（{r['n']:,}）")
        r = next((x for x in prod["levels_pooled"] if abs(x["min"] - t) < 1e-9), None)
        row.append("-" if not r or r.get("real") is None else f"<b>{_p(r['real'], 0)}</b>（{r['n']:,}）")
        lvl_rows.append(row)
    lvl_tbl = r29._tbl(["レース価値（較正後の期待回収率）", "2022", "2023", "2024", "2025", "2026", "全期間"], lvl_rows, ["", "n", "n", "n", "n", "n", "n"])
    pq = prod["pred_quantiles"]
    real_all = fin["race_realized_overall"][0]["real"]
    raw_gaps = [fit["horse"][k]["raw"]["pooled"]["gap"] for k in ("win", "place", "top2")]
    raw_h_lo, raw_h_hi = min(raw_gaps), max(raw_gaps)
    style = re.search(r"<style>.*?</style>", r29.STYLE_SRC.read_text(encoding="utf-8"), re.S).group(0)
    body = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>期待値の較正（表示する期待値を実現回収率に収束させる）— keiba-vpn</title>
{style}</head>
<body><a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>期待値の較正（表示する期待値を実現回収率に収束させる）</h1>
<p class="muted">ユーザー要望（2026-10-20）: 期待値の縮約を有効にするかを考えたい。ただし、実際にその期待値で買い続けて、表示した期待値に収束しない（実現回収率がずれる）のは問題。tp-47 の続き。2022〜2025 年は V4 の予測 × 確定オッズ・確定払戻（13,041 レース）、2026 年は未使用の評価（2,512 レース）。</p>
<div class="note"><b>要点.</b>
 <ul>
 <li><b>馬ごとの期待値（勝期待値・複期待値・買い推奨度）は、較正すると実現回収率に収束した.</b>較正なしの表示は実現より平均 {raw_h_lo:+.2f}〜{raw_h_hi:+.2f} 高かったが、較正後の『最大の期待値』は、2023〜2025 年の walk-forward でずれ {hb['gap']:+.3f}・ece {_f(hb['ece'], 3)}、未使用の 2026 年でもずれ {hb26['gap']:+.3f}・ece {_f(hb26['ece'], 3)}。単勝・連対・複勝のいずれも、2026 年のずれが基準（±0.12 以内）に収まった。<b>この範囲は有効にする。</b></li>
 <li><b>推奨ポートフォリオの期待回収率は、収束するとは言えない.</b>較正前は {_p(real_all + raw['gap'], 0)} 前後を表示していたが、実現は年によって {_p(min(yearly.values()), 0)}〜{_p(max(yearly.values()), 0)} と動く。2023〜2025 年の walk-forward ではずれ {rc['pooled']['gap']:+.3f} に収まったが、<b>2026 年は実現 {_p(ev26['race']['overall'][0]['real'], 0)}（較正後の表示 {_p(ev26['race']['overall'][0]['pred'], 0)}）で、ずれ {ev26['race']['calibrated']['gap']:+.2f}</b>。基準（0.12）を満たさない。年ごとの揺れが大きく、1 年の区間（±10 ポイント程度）に収まる保証はない。</li>
 <li><b>そこで、ポートフォリオ側は『目安』として出す.</b>2022〜2026 年を合わせて当てはめ直し（平均で 95〜96% 程度）、画面には『期待回収率（較正後）は目安。過去の年別の実現回収率は {_p(min(yearly.values()), 0)}〜{_p(max(yearly.values()), 0)}』を併記する。較正前の 200% 超の表示は廃止する。</li>
 <li><b>『オレンジ』などの色分けは、相対的な順位の目安に変えた.</b>閾値は較正後の期待回収率で hot ≥ {cfg['levels']['hot']:.2f}、good ≥ {cfg['levels']['good']:.2f}、ok ≥ {cfg['levels']['ok']:.2f}（従来は 1.5 / 1.25 / 1.05）。2022〜2025 年では上位の段ほど実現が高い傾向があるが、2026 年は段による差がなかった。『プラス期待』『特に高い』の語は『平均並み』『相対的に高い』に改めた。</li>
 <li><b>モデルに優位があるとは言えない.</b>較正後の表示は、ほとんどが 90〜100% の間に収まる（レースの期待回収率の分位: 5% {pq['0.05']:.2f}、中央 {pq['0.5']:.2f}、95% {pq['0.95']:.2f}）。これは正直な水準で、買い目の選び方（期待値 1.05 以上）が実現回収率をほぼ損益分岐に近づける程度の効果しか示していない（tp-36・tp-43・tp-44 の結論と整合）。</li>
 </ul></div>
<h2>1. 方法</h2>
<ul>
 <li><b>較正の対象</b>は『画面に出る値』そのもの: (a) 馬ごとの EV（勝期待値 = 勝率×単勝オッズ、複勝 = 複勝率×実効オッズ、連対 = 連対率×馬連プールと整合する公正オッズ）、(b) 推奨ポートフォリオの買い目（単勝・複勝・馬連・ワイド）の EV と、購入額加重の期待回収率。買い目は『EV の高いものを選んだ後』の値なので（winner's curse）、全馬ではなく実際に選ばれる買い目で当てはめる。</li>
 <li><b>比較した方式</b>: 縮約 base+k(EV−base)（現行）、線形 a+b·EV、単調回帰（EV の分位 20 帯 → PAVA、端は外挿しない）、定数（EV の高低を使わず実績の平均）、市場並み（払戻率）。</li>
 <li><b>評価は walk-forward</b>: 年 Y の較正は Y より前の年だけで当てはめ、Y で評価（2023・2024・2025 年）。2022 年は当てはめ専用。2026 年は方式と基準を決めた後に 1 回だけ。</li>
 <li><b>『収束』の指標</b>: ずれ（表示の平均 − 実現の平均）、ece（表示の分位帯ごとの |平均表示 − 平均実現| の加重平均）、各帯の実現の 95% 区間（レース単位ブートストラップ）が表示を含む割合、MSE（較正なしからの改善）。</li>
 <li><b>事前に決めた基準</b>（{html.escape(json.dumps(fin['criteria'], ensure_ascii=False))}）: プールのずれ ±0.08 以内、ece 0.10 以下、区間が表示を含む帯が 75% 以上、2024・2025 年のずれが ±0.12 以内。shrink / linear / isotonic のうち基準を満たすものから ece 最小を選び、どれも満たさなければ定数にする（定数は ece が自明に小さいので比較に入れない）。</li>
</ul>
<h2>2. 馬ごとの EV</h2>
<p class="muted">MSE は『表示した値』と『1 点買ったときの払戻（100 円あたり）』の二乗誤差の平均。較正なしは MSE が 1〜2% 大きく、ずれが +0.17〜+0.25 ある。</p>
{horse_tbls}
<p><b>買い推奨度の元になる『最大の期待値』</b>（単勝・連対・複勝の較正後の最大。実現はその券種のもの）: 2023〜2025 年でずれ {hb['gap']:+.4f}、ece {_f(hb['ece'], 3)}（帯の区間が表示を含む割合 {hb['cover']}）。2026 年はずれ {hb26['gap']:+.3f}、ece {_f(hb26['ece'], 3)}（{hb26['cover']}）。</p>
{_bins_tbl(hb['bins'])}
<h2>3. 買い目と推奨ポートフォリオ</h2>
{cand_tbls}
<h3>レースの期待回収率（買い目ごとの較正後の EV の購入額加重平均）</h3>
<p class="muted">較正なしの表示は実現より平均 {raw['gap']:+.2f} も高かった（2023〜2025 年）。較正後（上限 {cfg['cap']['roi']:.2f} で打ち切り）:</p>
{race_tbl}
{_bins_tbl(rc['pooled']['bins'])}
<h3>実現回収率の年別（購入額加重）</h3>
<p class="muted">推奨ポートフォリオ（上限 2,000 円/R）を確定オッズで買い続けたときの実現。年によって大きく動く。</p>
{yearly_tbl}
<p>2026 年は、ワイド・馬連の買い目が当てはめ期間より大きく負けた（実現 {_p(ev26['race']['overall'][0]['real'], 0)}）。2026 年のモデルは市場に対して弱かった（tp-36 の 2026 年の評価と同じ）。そのため、ポートフォリオ側は『収束』を約束できず、<b>年ごとの範囲を併記する目安</b>にとどめる。</p>
<h2>4. 画面の指標の閾値</h2>
<h3>買い推奨度（馬ごと）</h3>
{tier_tbl}
<p>閾値を超える馬は少数（推奨で約 5%）。2023〜2025 年は実現が表示以上だったが、2026 年は表示をやや下回る段もあり（区間は広い）、強推奨の実現が高いのは件数が少ないため。</p>
<h3>レース価値（推奨ポートフォリオ）</h3>
{lvl_tbl}
<p>2022〜2025 年は上位の段ほど実現が高いが、2026 年は段による差がなく、全体が約 75% だった。色分けは『相対的な順位の目安』で、損益分岐を保証しない。hot はその日の期待回収率の上位 20% に限る（従来どおり）。</p>
<h2>5. 反映した仕様</h2>
<ul>
 <li><code>config/ev_calibration.json</code>（版 2、<code>enabled: true</code>）: 券種ごとの方式とパラメータ、買い推奨度の閾値、レース価値の閾値、打ち切り（EV 1.30、期待回収率 1.15）、年別の範囲。</li>
 <li><code>src/utils/ev_calibration.py</code>: <code>calibrated_ev</code> / <code>calibrate_portfolio</code> / <code>tier_thresholds</code> / <code>value_levels</code>。検証していない券種（馬単・3連複）は払戻率（市場並み）に置く。</li>
 <li>API: 予測の応答の各馬に <code>ev_win_cal</code> / <code>ev_top2_cal</code> / <code>ev_place_cal</code>（保存済みの予測にも読み出し時に付与し、買い推奨度を付け直す）。推奨買い目の応答に <code>ev_calibrated</code>・<code>expected_roi_calibrated</code>・<code>roi_range_calibrated</code>。レース価値は較正後の期待回収率で判定。</li>
 <li>画面: 勝期待値・複期待値・推奨・期待回収率に『（較正後）』を明記。買い目の期待値は較正後で、マウスを載せると較正前の値と、較正が目安の券種であることを表示。期待回収率の下に年別の実現回収率の範囲を表示。</li>
 <li>買い目の選定は従来どおり較正前の EV 1.05 以上（較正はこの選定を前提に当てはめた。選定の条件を変えたら当てはめ直すこと）。</li>
</ul>
<h2>6. 限界</h2>
<ul>
 <li>オッズは締切後の確定オッズ。購入時点（T-60・T-15）のオッズで出る EV は、確定に向かって動く分だけ異なる。T-15 の履歴が 1 シーズン分たまったら、購入時点のオッズで当てはめ直す。</li>
 <li>推奨ポートフォリオの較正は、年によるモデルの強さの変動（2026 年の悪化）を予測できない。『目安』と年別の範囲の併記は、そのための措置。</li>
 <li>ワイドは年のずれが大きく、買い目の較正は基準を満たさない（目安）。3連複・馬単は過去の確定オッズが無く、払戻率（市場並み）に置いた。</li>
 <li>2026 年の単勝だけは、方式を選ぶ作業の途中で 1 度見ている（最終の方式と基準は 2022〜2025 年の walk-forward だけで決めた）。</li>
 <li>2022〜2024 年の予測は leave-one-year-out（後の年も学習に使う）で、2025 年と 2026 年の予測は前の年までの学習。前者はやや楽観的な可能性がある。</li>
 <li>モデルを更新（v5 など）したら、<code>python -m src.scripts.data.calibrate_ev</code>（build_horse → build_portfolio → fit → final → eval2026 → config）で当てはめ直すこと。</li>
</ul>
<p class="muted">関連: <a href="ev_calibration_audit.html">期待値の水準の監査（tp-47）</a> / <a href="betting_edge_search.html">馬券で回収率 100% を超える余地の探索（tp-36）</a> / <a href="proven_methods_roi.html">実績者の手法と検証（tp-44）</a> / <a href="training_pipeline_decisions.html">決定ログ</a></p>
</body></html>
"""
    CHAPTER.write_text(body, encoding="utf-8")
    print("保存:", CHAPTER)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    build(Path(ap.parse_args().dir))


if __name__ == "__main__":
    main()
