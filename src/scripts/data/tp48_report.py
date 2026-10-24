"""tp-48 の章 docs/html/modeling/chapters/trifecta_true_odds.html を eval.json / qa.json から作る。

python -m src.scripts.data.tp48_report --json-dir <eval.json などのあるディレクトリ>
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CH = ROOT / "docs/html/modeling/chapters"
OUT = CH / "trifecta_true_odds.html"


def head_style() -> str:
    t = (CH / "race_type_marks.html").read_text(encoding="utf-8")
    m = re.search(r"<style>.*?</style>", t, re.S)
    return m.group(0) if m else "<style></style>"


def pc(x) -> str:
    return "—" if x is None or x != x else f"{100 * x:.0f}%"


def cell(v) -> str:
    if not v or v[0] is None:
        return "<td class='num'>—</td>"
    return f"<td class='num'>{pc(v[0])}<br><span class='muted'>{v[1]:,}点</span></td>"


def design_cell(d) -> str:
    if not d:
        return "<td class='num'>—</td>"
    return f"<td class='num'>{pc(d[0])}<br><span class='muted'>[{pc(d[1])}–{pc(d[2])}] {d[3]:,}点</span></td>"


def table(rows: list[dict], pick, years=("y2023", "y2024", "y2025", "y2026"), labels=("2023", "2024", "2025", "2026")) -> str:
    out = ["<table><tr><th>方策</th>" + "".join(f"<th>{lb}</th>" for lb in labels) + "<th>2024〜25 [95% 区間]</th></tr>"]
    for r in rows:
        name = pick(r)
        if not name:
            continue
        out.append(f"<tr><td>{html.escape(name)}</td>" + "".join(cell(r.get(y)) for y in years) + design_cell(r.get("design")) + "</tr>")
    out.append("</table>")
    return "".join(out)


def build(ev: dict, qa: dict, ex: dict) -> str:
    T = ev["tables"]
    rc = ev["reality_check_2024_25"]
    det = ev["detail"]

    def sel(tn: str, keep: set[str]) -> list[dict]:
        rows = [r for r in T[tn] if r["policy"].split("|", 2)[-1] in keep]
        return rows

    def lab(r: dict) -> str:
        parts = r["policy"].split("|")
        return " ".join(parts[1:]) if len(parts) > 2 else r["policy"]

    qa_rows = "".join(
        f"<tr><td>{y}</td><td class='num'>{v['races']:,}</td><td class='num'>{v['rows']:,}</td><td class='num'>{v['inv_sum_median']:.4f}</td><td class='num'>{100 * v['hit_exact_rate']:.1f}%</td>"
        f"<td class='num'>{100 * v['any_missing_rate']:.0f}%（平均 {100 * v['miss_rate_mean']:.2f}%）</td><td class='num'>{len(ex[y]['missing'])} + {len(ex[y]['inconsistent'])}</td></tr>"
        for y, v in qa.items())
    keep_s = {"top1", "top2", "top5", "top10", "top50", "ev>=1.2|top5", "ev>=2.0|top5", "ev>=3.0|top5", "shrunk1.05(raw>=3.98)|top50"}
    s_model = table(sel("S_sanrentan|model", keep_s), lab)
    s_mixed = table(sel("S_sanrentan|mixed", keep_s), lab)
    inc = table([r for r in T["S_incons|model"] if r["policy"].endswith("|ev>=1.2") or r["policy"].endswith("|ev>=0.0")], lambda r: r["policy"])
    trio = table(sel("T_sanrenpuku_implied|model", {"top1", "top5", "top50", "ev>=2.0|top5"}), lab)
    keep_e = {"top1", "top3", "ev>=1.0|top1", "ev>=1.2|top1", "ev>=1.5|top1", "ev>=2.0|top1", "ev>=3.0|top1", "shrunk1.0(raw>=1.63)|top50", "shrunk1.05(raw>=1.84)|top50", "ev>=2.0|top5", "ev>=1.5|top5"}
    e_tabs = ""
    for bt, nm in (("fukusho", "複勝"), ("wide", "ワイド"), ("umaren", "馬連")):
        for sch, sn in (("model", "市場なし（V4 単体）"), ("mixed", "混合（Benter 型）")):
            rows = [r for r in T[f"E_{bt}|{sch}"] if r["policy"].split("|", 2)[-1] in keep_e or "shrunk" in r["policy"] and sch == "model"]
            e_tabs += f"<h4>{nm} × {sn}</h4>" + table(rows, lab)
    m_tab = table(T["M_marks"], lambda r: r["policy"].replace("marks|", ""))
    fin = "".join(f"<tr><td>{html.escape(r['policy'])}</td><td class='num'>{pc(r['roi'])}</td><td class='num'>{pc(r['lo'])}–{pc(r['hi'])}</td><td class='num'>{r['bets']:,}</td></tr>" for r in ev["final_2026"])
    top_lo = "".join(f"<tr><td>{html.escape(r['policy'])}</td><td class='num'>{pc(r['roi'])}</td><td class='num'>{pc(r['lo'])}–{pc(r['hi'])}</td><td class='num'>{r['bets']:,}</td></tr>" for r in ev["top_by_lower_bound_2024_25"][:10])
    det_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td class='num'>{pc(v['roi'])}</td><td class='num'>{pc(v['roi_without_top3_hits'])}</td><td class='num'>{100 * v['hit_race_rate']:.1f}%</td>"
        f"<td class='num'>{v['top3_hit_payouts'][0] / 1e4:.0f} 万円</td><td class='num'>{pc(v.get('ruin_prob_100u'))}</td></tr>" for k, v in det.items())
    ns = ev["noise_sensitivity"]
    ns_rows = ""
    for sch in ("model", "mixed"):
        for k in (1, 5):
            ns_rows += f"<tr><td>{sch} top{k}</td>" + "".join(f"<td class='num'>{pc(ns[f'{sch}|top{k}|sigma{s}'][0])}</td>" for s in (0.0, 0.05, 0.1, 0.2, 0.3)) + "</tr>"
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>3連単の実オッズで再評価（tp-48）</title>
{head_style()}</head>
<body><a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>3連単の実オッズで再評価（tp-48）</h1>
<p class="muted">tp-36 で『確定オッズが無いと検証できない』として保留した買い方を、3連単の実オッズ（2024〜2026 年、{sum(v['races'] for v in qa.values()):,} レース）で再評価した。ユーザーの質問『的中率が高い上位3券種に、期待値が高いものだけを買う検証はしたか』にも答える（§4.5）。オッズは確定後に取得した値で、購入時点の値ではない。</p>
<div class="note"><b>要点.</b>
 <ul>
 <li><b>区間下限が 100% を超える方策は、無かった.</b>{ev['n_policies']} 方策を試し、2024〜25 年の最良の下限は 96%（複勝、モデルの EV を縮約して選ぶ）。White の Reality Check は p={rc['p_value']:.2f}（最大 t={rc['t_max']:.2f}）。</li>
 <li><b>3連単で 2024〜25 年に 170% 近い回収率が出る方策（モデルの EV 上位 5 点）は、的中 3 レースの払戻に支えられた偶然</b>。上位 3 的中を除くと 52%、年別は 270%（2024）、69%（2025）、12%（2026）と揺れる。破産確率（100 単位）は 95%。</li>
 <li><b>市場を混ぜた確率（mixed）で選ぶ3連単は 18〜71% で、100% に届かない</b>。モデル単体（model）は年で逆転する（2026 年は上位 k 点が 0〜31%、上位 50 点が 87%、縮約しきい値が 106〜118%。2025 年は 59〜65%）。</li>
 <li><b>的中率が高い券種の『期待値が高いものだけ』</b>: 複勝はモデルの EV≥2.0 で 113%（区間 93〜132%）、上位 3 的中を除いても 106%。年別も 103%・110%・99% と安定して 100% 前後に位置するが、下限は 100% を超えない。ワイド・馬連の 1 点買いは 117%・156% だが、的中 3 レースを除くと 101%・89%。</li>
 <li>印の組合せ（◎○▲★△）は 89〜96%。2026 年は 66〜85%。</li>
 </ul></div>
<h2>1. データの品質確認</h2>
<table><tr><th>年</th><th>レース</th><th>行（組合せ）</th><th>Σ(1/オッズ) の中央値<br>（払戻率 72.5% なら 1.379）</th><th>的中組合せの odds×100 と払戻が一致</th><th>組合せが欠けるレース</th><th>除外（取得なし + 不整合）</th></tr>{qa_rows}</table>
<ul>
 <li>Σ(1/オッズ) は 1.3798 で、払戻率 72.5% と一致する。的中した組合せの odds×100 は、確定払戻と 98.4〜99.7% のレースで完全に一致する（残りは同着などで半額になる払戻や、払戻の連結）。</li>
 <li>応答には、人気薄の組合せが一部欠ける日がある（平均 0.3〜0.4%）。欠けた組合せは買えないものとして扱う。欠けが 3% 以上、Σ(1/オッズ) が 1.37〜1.39 の外、または的中組合せの払戻が合わないレースは、3連単の評価から除外する。</li>
 <li>オッズは確定後（締切後）に取得した値で、購入時点のオッズではない。</li>
</ul>
<h2>2. 評価の設計</h2>
<ul>
 <li>確率は割引 Harville（レースごとに連対率・複勝率へ α を当てはめる）。『model』は V4 の勝率・連対率・複勝率（市場なし）、『mixed』は単勝の Benter 混合を勝率とし連対率・複勝率を拡縮。</li>
 <li>EV = 確率 × 実オッズ。複勝は下限＋0.3×（上限−下限）（tp-47 の修正）。『縮約後』は tp-47 の縮約係数 k（複勝 0.24、馬連 0.11、ワイド 0.10、3連系は仮に 0.10）で払戻率の側へ寄せた EV のしきい値。</li>
 <li>各方策は 100 円ずつの均等買い、払戻は確定払戻。レース単位の母数、ブートストラップで 95% 区間。設計は 2024〜25 年、2026 年は最後に 1 回（実オッズのある券種のみ）。</li>
 <li>試した方策は {ev['n_policies']} 通り（3連単、プール間の不整合、3連複、複勝・ワイド・馬連、印の組合せ）。</li>
</ul>
<h2>3. 結果 1: 3連単（実オッズ）</h2>
<h3>モデル単体（市場なし）の EV 上位 k 点</h3>
{s_model}
<p class="muted">EV のしきい値を上げても、上位 5 点の回収率はほとんど変わらない（上位 5 点の 93% は EV が 3 以上）。年別の揺れが非常に大きい（2024 年 270%、2025 年 69%、2026 年 12%）。</p>
<h3>混合（Benter 型）の EV 上位 k 点</h3>
{s_mixed}
<h3>プール間の不整合（実オッズ ÷ 単勝オッズから作った推定価格）</h3>
{inc}
<p class="muted">実オッズが推定価格より高い組合せを買っても、回収率は上がらない（tp-44 と同じ結論）。</p>
<h2>4. 結果 2: 3連複・印・的中率が高い券種</h2>
<h3>4.1 3連複（3連単プールから作った価格）</h3>
<p class="muted">3連複の過去の確定オッズは無いので、3連単の逆数の 6 通りの和から作った『3連単プール由来の価格』で選び、確定払戻で精算した（取引できる価格ではない、探索）。</p>
{trio}
<h3>4.2 印（◎○▲★△、新ルール）の組合せ</h3>
{m_tab}
<p class="muted">印の組合せは、どれも 100% 未満（2024〜25 年で 87〜96%、2026 年は 66〜85%）。</p>
<h3>4.5 的中率が高い券種（複勝・ワイド・馬連）の『期待値が高いものだけ』</h3>
<p class="muted">ユーザーの質問への答え。実オッズは 2023〜2025 年（複勝は 2026 年のオッズが無い）。しきい値は raw の EV と、縮約後 EV 相当のしきい値の両方。</p>
{e_tabs}
<h2>5. 主な方策の詳細（2024〜25 年）</h2>
<table><tr><th>方策</th><th>回収率</th><th>上位 3 的中を除く</th><th>的中レース率</th><th>最大払戻</th><th>破産確率（100 単位）</th></tr>{det_rows}</table>
<div class="note">100% を超える方策は、的中が稀な券種（3連単・馬連の 1 点買い）では、上位 3 レースの払戻が全体を押し上げている。上位 3 的中を除くと 3連単 52%、馬連 89%。複勝（EV≥2）とワイドは除いても 100% 前後に残る。</div>
<h2>6. 多重比較と 2026 年</h2>
<ul>
 <li>Reality Check（{rc['n_strategies']} 方策、2024〜25 年のレース単位ブートストラップ）: p = {rc['p_value']:.2f}、最大 t = {rc['t_max']:.2f}（{html.escape(rc['best'])}）。</li>
 <li>2024〜25 年の下限が高い方策（点数 300 以上）:</li>
</ul>
<table><tr><th>方策</th><th>回収率</th><th>95% 区間</th><th>点数</th></tr>{top_lo}</table>
<h3>2026 年の最終評価（事前に決めた規則: 下限上位 5 + 回収率上位 5 を 1 回だけ）</h3>
<table><tr><th>方策</th><th>回収率</th><th>95% 区間</th><th>点数</th></tr>{fin}</table>
<p class="muted">複勝・ワイド・馬連は、2026 年の確定オッズ（複勝の幅と組合せ券）が履歴に無く、評価できない（点数 0）。3連単の方策は、2026 年に 0〜13% だった（ただし縮約しきい値の別の方策は 2026 年に 106〜118% で、2025 年の 59〜65% と逆転しており、どちらも偶然の範囲）。</p>
<h2>7. 購入時点のオッズが確定と違う場合（対数正規の誤差）</h2>
<p class="muted">判断用オッズに σ の対数正規の誤差を加えて EV 上位を選び直し、払戻は確定オッズで精算した（3連単、2024〜25 年）。</p>
<table><tr><th>方策</th><th>σ=0</th><th>0.05</th><th>0.1</th><th>0.2</th><th>0.3</th></tr>{ns_rows}</table>
<p class="muted">モデル単体の 3連単は、誤差を少し入れるだけで 84〜217% の間を行き来する。的中 3 レースの有無で決まる偶然の水準で、感度を見る意味が小さい。</p>
<h2>8. 結論と次の一手</h2>
<ul>
 <li>実オッズの3連単でも、区間下限が 100% を超える方策は無い。3連単・馬連の 100% 超は、まれな高配当に依存する。</li>
 <li>最も近いのは複勝の『モデルの EV≥2.0』（113%、除外後 106%、年別 103・110・99%）。ただし区間下限は 93% で、2026 年の複勝オッズが無く未確認。<b>2026 年の複勝・馬連・ワイドの確定オッズ（過去分の取得）と、T-60/T-15 の実オッズの蓄積後に再検証</b>する。</li>
 <li>3連複の確定オッズの履歴は無い。取得を始めた 10/17 以降のスナップショットが貯まってから評価する。</li>
</ul>
</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", required=True)
    a = ap.parse_args()
    d = Path(a.json_dir)
    OUT.write_text(build(json.loads((d / "eval.json").read_text(encoding="utf-8")), json.loads((d / "qa.json").read_text(encoding="utf-8")),
                         json.loads((d / "exclude.json").read_text(encoding="utf-8"))), encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    main()
