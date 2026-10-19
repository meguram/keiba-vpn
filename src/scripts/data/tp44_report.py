"""tp-44 の章 docs/html/modeling/chapters/proven_methods_roi.html を pools.json / rules.json から作る。

python -m src.scripts.data.tp44_report --pools pools.json --rules rules.json
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CH = ROOT / "docs/html/modeling/chapters"
OUT = CH / "proven_methods_roi.html"
SRC = "~/research/keiba-roi-over-100/sources/"

# (手法, 種類の印, 検証可能性, 原文の場所)
SOURCES = [
    ("市場オッズ（公開）をモデルに混ぜる Benter 型。モデルの確率と公衆の確率を対数で混ぜ、混合確率で期待値を見る", "〔論文〕", "手法は公開。Benter 自身の損益は本人の報告で、第三者検証は無い（生存者バイアスは不明）", "benter-1994-computer-based-handicapping.txt:19,466-470"),
    ("同上の実績（1990-91 シーズンに 300 万ドル、通算約 10 億ドルとされる）", "〔まとめ〕", "報道・百科事典の記述。期間・点数・購入ルールは非公開で検証できない", "guinness-2025-benter-billion.md:68 / wiki-en-bill-benter.md:105"),
    ("持ち越しで大きくなった重勝に大きく賭ける（香港 Triple Trio、HK$160 万で約 5.1 万通り。全組合せは 1,000 万通り超のごく一部）", "〔まとめ〕", "1 回の事例。賭式の仕組み（持ち越し）自体は検証できる", "guinness-2025-benter-billion.md:27,74"),
    ("favorite–longshot bias（人気薄が過大評価）を使う。長期の回収率が人気帯で単調", "〔論文〕", "多数の市場で再現。JRA では tp-36 の全数調査と本章の 17 番人気以下で再確認", "snowberg-wolfers-2010-flb.txt:57 / wiki-en-favourite-longshot-bias.md"),
    ("最後の 5 分のオッズ低下は、同じ最終オッズの馬より高い実現回収率（JRA 2004-2023、5 分刻みのオッズ）", "〔論文〕", "JRA-VAN の学術利用データ。再現には 5 分刻みの履歴が要る（自前は 10/18 の 1 開催日だけ）", "arxiv-2509.14645-last-minute-parimutuel.txt:16,86-91,121"),
    ("最終変動の約 4 割は発走 5 分前を過ぎてから到着する（13 日・実測）", "〔本人〕", "個人の観測。期間が短い", "rindokulab-2026-odds-arrival.md:72"),
    ("「函館の3番人気の単勝」: 発見期間 105.2%（432 点）→検証期間 56.9%", "〔本人〕", "著者自身が期間を分けて否定した記録。ルールは明確で再現可能", "koheikameyama-2026-roi105-split.md:38,44,53,396"),
    ("単勝・複勝で 5 年 17,000 レースを検証し、市場に勝てなかった", "〔本人〕", "負の結果。モデル・ブレンド・ケリーの設計は本リポジトリと同じ", "egu777-2026-keiba-ev-tansho-fukusho.md"),
    ("オーバーレイ探索: 条件を探しても walk-forward で劣化、人気馬に絞るほど改善（FLB）", "〔本人〕", "負の結果。条件は不明確", "teruyotasaki-2026-dev8-overlay.md:94"),
    ("オフライン検証で回収率 180% が出ても、本番（前向き）で 3 回崩れた", "〔本人〕", "前向き記録を先に用意する運用の記録", "a1yama-2026-validation.md:32,161"),
    ("ソフトによる機械的な馬券購入（2007-2009 年）が課税上 雑所得と認められた大阪地裁判決（2013-05-23）", "〔まとめ〕", "継続的な黒字が前提の訴訟。購入ルール・回収率の数字は本ページに無く、第三者は検証できない", "wiki-ja-toupyouken-kouei.md:287"),
    ("WIN5 は控除率 30%（払戻率 70%）で最も高い。的中が無い・上限超過でキャリーオーバー", "〔公式〕", "仕組みは公開。持ち越しの発生は稀（2011-09-04、2018-12-23 など）", "wiki-ja-win5.md:90,100,192"),
]


def head_style() -> str:
    t = (CH / "race_type_marks.html").read_text(encoding="utf-8")
    m = re.search(r"<style>.*?</style>", t, re.S)
    return m.group(0) if m else "<style></style>"


def pct(x) -> str:
    return "—" if x is None or x != x else f"{100 * x:.1f}%"


def strat_table(rows: list[dict]) -> str:
    kinds = [("fukusho", "odds_lo", "複勝（下限）"), ("fukusho", "odds_mid", "複勝（中央）"), ("umaren", "odds_lo", "馬連"),
             ("wide", "odds_lo", "ワイド（下限）"), ("wide", "odds_mid", "ワイド（中央）"), ("umatan", "odds_lo", "馬単")]
    ths = (1.0, 1.1, 1.2, 1.3, 1.5, 2.0)
    out = ["<table><tr><th>券種</th>" + "".join(f"<th>EV≥{t:g}</th>" for t in ths) + "</tr>"]
    for k, o, name in kinds:
        cells = []
        for t in ths:
            r = next((r for r in rows if r["kind"] == k and r["odds"] == o and abs(r["threshold"] - t) < 1e-9 and r["pmin"] == 0.0), None)
            if not r or r["n_bets"] == 0:
                cells.append("<td class='num'>—</td>")
            else:
                cells.append(f"<td class='num'>{pct(r['roi'])}<br><span class='muted'>[{pct(r['lo'])}–{pct(r['hi'])}] {r['n_bets']:,}点</span></td>")
        out.append(f"<tr><td>{name}</td>{''.join(cells)}</tr>")
    out.append("</table>")
    return "".join(out)


def bins_table(bins: dict) -> str:
    names = {"fukusho:odds_lo": "複勝（下限）", "umaren:odds_lo": "馬連", "wide:odds_lo": "ワイド（下限）", "umatan:odds_lo": "馬単"}
    edges = None
    out = ["<table><tr><th>推定 EV の帯</th>" + "".join(f"<th>{n}</th>" for n in names.values()) + "</tr>"]
    labels = [b["ev_bin"] for b in bins["umaren:odds_lo"]]
    for lab in labels:
        cells = []
        for k in names:
            b = next((x for x in bins[k] if x["ev_bin"] == lab), None)
            cells.append(f"<td class='num'>{pct(b['roi'])}<br><span class='muted'>{b['n']:,}点</span></td>" if b else "<td class='num'>—</td>")
        out.append(f"<tr><td>{html.escape(lab)}</td>{''.join(cells)}</tr>")
    out.append("</table>")
    return "".join(out)


def rules_table(rules: list[dict]) -> str:
    out = ["<table><tr><th>ルール（単勝 1 点 100 円）</th><th>2020-25 回収率 [95% 区間]</th><th>点数</th>" + "".join(f"<th>{y}</th>" for y in range(2020, 2026)) + "</tr>"]
    for r in rules:
        out.append(f"<tr><td>{html.escape(r['rule'])}</td><td class='num'>{pct(r['roi'])} [{pct(r['lo'])}–{pct(r['hi'])}]</td><td class='num'>{r['n_bets']:,}</td>" + "".join(f"<td class='num'>{pct(r.get(str(y)))}</td>" for y in range(2020, 2026)) + "</tr>")
    out.append("</table>")
    return "".join(out)


def win5_table() -> str:
    ev = [("2011-09-04（的中なし）", 1155567400, 852808741), ("2018-12-23（的中なし）", 856860300, 599802210), ("2011-06-26（上限 2 億円超過）", 1446 * 10000 * 100, 467188320)]
    out = ["<table><tr><th>持ち越しが出た日</th><th>当日の売上</th><th>次回への持ち越し額</th><th>持ち越し/売上</th><th>次回の売上が同じ額なら、市場比例の買いの期待回収率 (0.7 + C/S)</th></tr>"]
    for name, s, c in ev:
        out.append(f"<tr><td>{name}</td><td class='num'>{s / 1e8:.2f} 億円</td><td class='num'>{c / 1e8:.2f} 億円</td><td class='num'>{c / s:.2f}</td><td class='num'>{100 * (0.7 + c / s):.0f}%</td></tr>")
    out.append("</table>")
    return "".join(out)


def build(pools: dict, rules: list[dict]) -> str:
    rc = pools["reality_check"]
    par = pools["params_by_year"]
    par_rows = "".join(f"<tr><td>{y}</td><td class='num'>{p[0]:.3f}</td><td class='num'>{p[1]:.3f}</td><td class='num'>{p[2]:.3f}</td></tr>" for y, p in par.items())
    src_rows = "".join(f"<tr><td>{html.escape(a)}</td><td>{b}</td><td>{html.escape(c)}</td><td><code>{html.escape(d)}</code></td></tr>" for a, b, c, d in SOURCES)
    best = max((r for r in pools["strategies"] if r["n_bets"] >= 150 and r["roi"] == r["roi"]), key=lambda r: r["roi"])
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>回収率 100% 超の実績者の手法と検証（tp-44）— keiba-vpn</title>{head_style()}</head>
<body><a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>回収率 100% 超の実績者の手法と検証（tp-44）</h1>
<p class="muted">ユーザー依頼: 誰でもよいので回収率 100% 超の実績を出している人の予想方法を取り入れられるか。公開情報から手法を集め（A）、検証できる形にして自前の 2020〜2025 年の確定オッズ・払戻で評価した（B）。原文は <code>{SRC}</code> に保存（本リポジトリには入れない）。</p>
<div class="note"><b>要点.</b>
 <ul>
 <li><b>検証できる形で『100% 超』を示した公開記録は、見つからなかった.</b>実績の主張は Benter らの香港のシンジケート（本人の報告・報道のみ）と、ソフトによる機械購入を課税上認めた判決だけで、購入ルールも期間別の回収率も公開されていない。個人の記事（Zenn）は、むしろ『期間を分けると崩れた』という負の結果が中心だった。</li>
 <li><b>プール間の不整合（単勝オッズで作る順位分布と、複勝・馬連・ワイド・馬単のオッズの差）は、買っても 100% に届かない.</b>67 戦略（2022〜25 年、{pools['n_races']:,} レース）の最良でも {pct(best['roi'])}（区間 {pct(best['lo'])}〜{pct(best['hi'])}、{best['n_bets']:,} 点）。White の reality check は p = {rc['p']:.2f}。推定 EV が高い帯ほど回収率が下がる（EV 1.2〜1.5 の複勝で 20.9%）。</li>
 <li><b>公開ルールの再現: 『函館の3番人気の単勝』は 2020〜25 年で 70.1%（区間 58.1〜82.8%）.</b>記事の 105.2% は再現せず、56.9% の側に近い。</li>
 <li><b>最も取り入れる価値があるのは『発走直前のオッズ低下』（JRA 2004-2023 の査読前論文）と、持ち越しのある重勝（WIN5）の例外日.</b>どちらも自前の履歴データが足りない（直前のオッズ履歴は 1 開催日分のみ）ので、データを貯めてからの再検証になる。</li>
 </ul></div>
<h2>1. 集めた実績者・手法（種類の印・検証可能性・原文の場所）</h2>
<p class="muted">要約は付けず、原文の場所（ファイル:行）を指す。種類の印は 〔論文〕〔本人〕〔観察〕〔まとめ〕〔公式〕。検証可能性は、期間・点数・購入ルールが公開され、第三者が再現できるかで見ている。</p>
<table><tr><th>手法・主張</th><th>種類</th><th>検証可能性</th><th>原文の場所（sources/）</th></tr>{src_rows}</table>
<div class="note">検索エンジンが使えない環境（検索サイトのボット確認は回避しない）で、既知の論文・百科事典・個人記事の RSS 新着から掘ったため、網羅ではない。日本の回収率ランキング（ウマニティ等）や書籍は、本文を取得できず未収集（有料・会員制・著作権のため）。</div>
<h2>2. 検証 1: プール間の不整合（Hausch–Ziemba・Dr. Z 型）</h2>
<ul>
 <li><b>考え方</b>: 単勝の確定オッズから市場の勝率 q を作り、1 着 ∝ q^γ、2 着|1 着 ∝ q^λ2、3 着|1・2 着 ∝ q^λ3 の順位分布で、複勝（3 着以内）・馬連・ワイド・馬単の確率を出す（モデル不使用）。確率 × 確定オッズ（複勝・ワイドは下限と中央）が EV 推定。しきい値以上の組合せを全点均等買い。</li>
 <li><b>期間</b>: 検証年 2022〜2025 年。γ・λ は検証年より前の年だけで当てはめ（walk-forward）。精算は確定払戻（曖昧なレースは除外）。</li>
 <li><b>戦略数</b>: 6 種のオッズの取り方 × しきい値 6 × 確率の下限 2 = 72（点数 150 未満の 5 つを除いた {pools['n_strategies'] - 5} 本で検定）。区間はレース単位ブートストラップ、reality check は White 型（スチューデント化）。</li>
</ul>
<table><tr><th>検証年</th><th>γ（1 着）</th><th>λ2（2 着）</th><th>λ3（3 着）</th></tr>{par_rows}</table>
<p class="muted">γ≈1.03 は、1 着の確率がほぼ市場の勝率に比例することを示す。λ2≈0.83・λ3≈0.71 は、2・3 着ほど市場の勝率の差が縮む（割引 Harville と同じ向き）ことを示す。</p>
<h3>しきい値ごとの回収率（確率の下限なし、2022〜25 年のプール）</h3>
{strat_table(pools['strategies'])}
<h3>推定 EV の帯ごとの実現回収率</h3>
<p class="muted">推定 EV が実現回収率の目安になっているかの確認。EV が 1 を超えるほど回収率が下がるのは、確率の誤差が大きい組合せ（人気薄）ほど『EV が高く見える』選択バイアスと、組合せ券のオッズの方が単勝より情報を含むことによると考えられる（本データだけでは切り分けていない）。</p>
{bins_table(pools['ev_bins'])}
<div class="note"><b>結論.</b>単勝のオッズだけで作る確率と、他のプールのオッズの差は『割安』のシグナルにならない。推定 EV が 1.0〜1.2 の組合せでも、回収率は 35〜74%（複勝 1.1〜1.2 帯は 34.8%）で、EV 0.5〜0.9 の帯（66〜76%）より低い。reality check は p = {rc['p']:.2f}（最大の t = {rc['t_max']:.2f} で負）。複勝（Dr. Z）の歪みも、JRA の確定オッズでは見えない（Dr. Z は賭け金の割合で見る手法で、本データには各プールの賭け金が無く、オッズ経由の代用になっている点は限界）。</div>
<p class="muted">限界: 確定オッズは締切後の値で、購入時点の値ではない。賭け金のプール量（Dr. Z の本来の指標）は未取得。2026 年は組合せ券・複勝の確定オッズが無く、評価していない。</p>
<h2>3. 検証 2: 公開されているルールの再現</h2>
{rules_table(rules)}
<div class="note">記事の『函館の3番人気』は、2020〜25 年の全体で 70.1% で、全場の3番人気（77.4%）より低い。控除率 20% の単勝の平均（約 80%）に近づくか、下回る。17 番人気以下は 16.2%（記事の 17 番人気 24.3% と同じ向き）で、favorite–longshot bias を裏づける。</div>
<h2>4. キャリーオーバー（WIN5）の期待値</h2>
<p>WIN5 は控除率 30%（払戻率 70%）で最も高く、的中が無い、または払戻が上限（6 億円/100 円）を超えると、差額が次回に持ち越される。持ち越し額 C があるとき、次回の売上 S に対して『票数が的中確率に比例する買い』の期待回収率は 0.7 + C/S になり、C/S ≥ 0.3 で 100% を超える。</p>
{win5_table()}
<div class="note">過去の持ち越しは数回で、約 15 年の開催日（約 700 回）の 1% 未満。次回の売上が当日と同じとして、期待回収率は約 140〜144%（市場比例の買い）。ただし (1) 的中には 5 レースの勝ち馬をすべて当てる必要があり、全組合せ（約 50〜100 万通り）を 100 円ずつ買うには 5,000 万〜1 億円が要る、(2) 人気薄を狙うと favorite–longshot bias が控除を上乗せする、(3) 持ち越しの発生は予測できず頻度が低い。Benter が香港の持ち越しで約 5.1 万通りに HK$160 万を賭けた事例（表 1）は、この性質を使ったもの。日本では持ち越し発生を JRA の発表で確認してから、モデルの確率で上位の組合せだけを買う運用（EV = モデル確率 × 推定配当）が現実的で、過去の WIN5 払戻・持ち越し履歴を自前で取得する必要がある（現状は未保存）。</div>
<h2>5. 発走直前のオッズ変化（最も有望だが、データ不足）</h2>
<ul>
 <li>JRA の 2004〜2023 年の全レースを 5 分刻みのオッズで見た論文（表 1）は、<b>最後の 5 分でオッズが下がった馬は、最終オッズが同じ馬より実現回収率が高い</b>と報告している（最終 5 分のオッズ 10% の変化が示す回収率の差は、最終オッズの 10% の差の約 14 倍：最終オッズ 25.5 の場合）。この効果は購入の向き（締切直前に資金が集まる馬を、その流れの途中で買う）に使える。</li>
 <li>本リポジトリのスナップショット（T-60・T-15・確定）は 2026-10-18 の 1 開催日（36 レース）だけで、10/17 は確定のみ。検証には 1 年分（約 3,456 レース）が目安で、現行のペースで約 1 年かかる。</li>
 <li><b>実装案</b>: 開催日ランナーに T-10・T-5・T-2 のスナップショットを足し（オッズ API の負荷に注意、1 レース 3 回追加）、貯まるまでは JRA-VAN の時系列オッズ（5 分刻み）の導入可否を判断する。13 日の実測では最終変動の約 4 割が発走 5 分前以降に到着する（表 1）ため、T-5 で買い目を確定するなら残りの変動は織り込めない点に注意。</li>
</ul>
<h2>6. 取り入れる価値のある手法と実装案</h2>
<table><tr><th>手法</th><th>判断</th><th>実装案（買い目の出し方）</th></tr>
<tr><td>Benter 型の混合確率</td><td>すでに採用済み（tp-36）。2023〜25 年は高 EV 側で良い区間があるが 2026 年は再現せず</td><td>混合 EV≥1.2 の単勝を参考値として表示（購入根拠にはしない）</td></tr>
<tr><td>プール間の不整合</td><td>不採用（回収率 83% が最良）</td><td>—</td></tr>
<tr><td>Dr. Z（複勝）</td><td>不採用（オッズ経由では見えない）。賭け金のプール量が取れれば再評価</td><td>—</td></tr>
<tr><td>公開ルール（函館の3番人気など）</td><td>不採用（70.1%）</td><td>—</td></tr>
<tr><td>発走直前のオッズ低下</td><td>有望。データ待ち</td><td>T-10・T-5 のスナップショットを貯め、V4 の EV に『直近のオッズ変化率』を馬券判断の条件として足す（特徴量にはしない）</td></tr>
<tr><td>WIN5 の持ち越し日</td><td>稀な例外日。履歴を貯めてから</td><td>持ち越しを検知したら、モデル確率の上位組合せを予算内で買う</td></tr></table>
<h2>7. 試した数</h2>
<p>A: 手法 12 件を収集。B: プール間 72 戦略（検定 {pools['n_strategies'] - 5} 本）、公開ルール 5 件、WIN5 は理論計算（持ち越し 3 日）。<a href="betting_edge_search.html">tp-36</a> と <a href="race_type_marks.html">tp-43</a> の戦略（3,080 本など）とは別集合なので、全体の多重比較は両者を足した数を割り引いて読む。</p>
<p class="muted">再現: <code>python -m src.scripts.data.tp44_pools</code> / <code>tp44_rules</code> / <code>tp44_report</code>（pools.json・rules.json は <code>data/local/modeling/betting/tp44/</code>）。</p>
</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", type=Path, required=True)
    ap.add_argument("--rules", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    text = build(json.loads(a.pools.read_text(encoding="utf-8")), json.loads(a.rules.read_text(encoding="utf-8")))
    if a.out.exists() and "<!-- tp45:start -->" in a.out.read_text(encoding="utf-8"):      # tp-45 が足した節（5b）を作り直しで消さない
        old = a.out.read_text(encoding="utf-8")
        block = old[old.index("<!-- tp45:start -->"): old.index("<!-- tp45:end -->") + len("<!-- tp45:end -->")]
        text = text.replace("<h2>6. 取り入れる価値のある手法と実装案</h2>", block + "\n<h2>6. 取り入れる価値のある手法と実装案</h2>", 1)
    a.out.write_text(text, encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
