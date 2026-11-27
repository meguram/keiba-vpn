"""馬連・ワイドの買い方の研究（docs/html/operations/final/pair_bets.html）を、段 1 の JSON から作る（数字を手で写さない）。

入力（src/research/pair_bets/ の道具が書く）:
  docs/operations/pair_bets/arith.json（上乗せの算数）・market.json（市場どうし）・sstar.json（組の s*）
使い方: python -m src.scripts.docs.build_pair_bets
費用 0 円（手元のファイルだけ）。
"""

from __future__ import annotations

import html
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.scripts.docs.build_latest_strategy_tables import CSS

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "docs/operations/pair_bets"
DST = ROOT / "docs/html/operations/final/pair_bets.html"
TITLE = "馬連・ワイドの買い方の研究"
JST = timezone(timedelta(hours=9))
NEED = 1 / 0.775
# c13 の選ぶ期間（2017〜19）の市場比（select_a・すでに見た値）
WIN_REAL = json.loads((ROOT / "config/megu_cycles/candidates/c13_flat.json").read_text(encoding="utf-8"))["select_a"]["market_ratio"]


def esc(x) -> str:
    return html.escape(str(x))


def f3(x) -> str:
    return "－" if x is None else f"{x:.3f}"


def table(head: list[str], rows: list[list]) -> str:
    h = "".join(f"<th>{esc(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f'<td class="n">{c}</td>' if i else f"<td>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def mark(x: float) -> str:
    return f"<b>{f3(x)}</b>" if x >= NEED else f3(x)


def build() -> str:
    A = json.loads((SRC / "arith.json").read_text(encoding="utf-8"))
    M = json.loads((SRC / "market.json").read_text(encoding="utf-8"))
    S = json.loads((SRC / "sstar.json").read_text(encoding="utf-8"))
    T = A["table"]
    g = lambda kind, m, s, r: T[f"{kind}|{m}|{s}|{r}"]["全体"]
    rs = sorted({float(k.split("|")[3]) for k in T})
    rmax = rs[-1]
    axis_max = max(g(k, m, "axis", rmax) for k in ("umaren", "wide") for m in ("harville", "lbs"))
    both_min = min(g(k, m, "both", rmax) for k in ("umaren", "wide") for m in ("harville", "lbs"))
    both_lo = max(g(k, m, "both", rs[0]) for k in ("umaren", "wide") for m in ("harville", "lbs"))
    WIN_CLAIM = M["win"]["1.3"]["p_over_q"]          # 単勝の期待値 1.3 以上の見かけの p/q
    keep = math.log(WIN_REAL) / math.log(WIN_CLAIM)
    pair_claim = {ty: M[ty]["ev_lbs_o"]["1.3"]["ai_over_m"] for ty in ("umaren", "wide")}       # 払戻を決めるのは組の市場 m（critic）
    m_hq = [M[ty]["ev_lbs_o"]["1.3"]["m_over_Hq"] for ty in ("umaren", "wide")]
    pair_real = {ty: math.exp(keep * math.log(v)) for ty, v in pair_claim.items()}
    bands_pop = [f"{lo}〜{hi}" for lo, hi in M["pair_pop_bands"]]
    lbs_u = M["umaren"]["m_over_H_sum_lbs"]
    lbs_w = M["wide"]["m_over_H_sum_lbs"]
    made = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    sb = S["bands"]

    o = [f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>{TITLE}</h1>
<p class="sub">作成 {made}（JST）。数字は段 1 の道具（<code>src/research/pair_bets/</code>）の JSON から生成（<code>src/scripts/docs/build_pair_bets.py</code>）。
段 1 は着順も払戻も使っていない。記録は docs/operations/pair_bets/README.md。判定は rules-council の 1 号（12/4 の A の門）。</p>
<div class="box">
<b>結論</b>
<ul>
<li>単勝で AI が持つ上乗せは、「軸の 1 頭だけ」に上乗せがある形では、組で薄まる。軸の上乗せ ×{rmax:.2f} でも組は最大 {f3(axis_max)} 倍で、控除の線 {f3(NEED)} に届かない。</li>
<li>届くのは、2 頭ともが互いに独立して上乗せを持つ形だけ（2 頭とも ×{rmax:.2f} で {f3(both_min)} 倍以上）。2 頭とも ×{rs[0]:.2f}（確かめの実績の大きさ）では最大 {f3(both_lo)} 倍で届かない。</li>
<li>組の市場（確定）と、単勝の市場から Lo・Bacon-Shone（LBS）の型で作った組の確率は、組の人気の帯の和で比べると 80 番人気まで {f3(min(min(list(lbs_u.values())[:4]), min(list(lbs_w.values())[:4])))}〜{f3(max(max(list(lbs_u.values())[:4]), max(list(lbs_w.values())[:4])))} で近い。
ただし組ごとにはばらつきがあり、AI の期待値が 1.3 以上の組では、組の市場の値は単勝からの値より {100 * (1 - max(m_hq)):.0f}〜{100 * (1 - min(m_hq)):.0f}% 低い。どちらが正しいかは着順を見ないと決められない。</li>
<li>AI の組の期待値は、見かけの上で大きく出る。期待値 1.3 以上の組は 1 レースに馬連 {M['umaren']['ev_lbs_o']['1.3']['per_race']:.1f}・ワイド {M['wide']['ev_lbs_o']['1.3']['per_race']:.1f} 組（単勝は {M['win']['1.3']['per_race']:.2f} 頭）。
AI と市場の食い違いが 2 頭分で掛け算になるためで、上乗せが増えたからではない。
参考: 単勝の見かけの ×{WIN_CLAIM:.2f}（2022〜24・確定）と、選ぶ期間の実際の ×{WIN_REAL:.2f}（2017〜19・T-60 の再現）の比（母集団が違うので目安）を、組の見かけ（AI ÷ 組の市場）に当てると、馬連 {f3(pair_real['umaren'])}・ワイド {f3(pair_real['wide'])} 倍で、{f3(NEED)} に届かない。</li>
<li>A の門の読み方（1 号が事前に固定）に当てると、札は「上限だけで届く」。100% 超とは言えない。会員の画面・本番の推奨は替えない。</li>
</ul>
</div>
"""]

    o.append("<h2>なぜ組では単勝の上乗せが増えないか</h2><ol>"
             "<li>組が当たるには、軸と相手の 2 頭が両方来る必要がある。AI の強みが軸の 1 頭だけなら、相手の分は市場どおりで、強みは薄まる。</li>"
             "<li>人気の軸ほど薄まる。軸の勝ちを増やすと、ほかの馬の取り分が減り、相手の 2・3 着の見込みが下がるため。ワイドは 3 着内なので、さらに薄まる。</li>"
             "<li>強みが掛け算で増えるのは、2 頭ともに独立した強みがあるときだけ。AI が 2 頭を同時に買いかぶると、誤差も同じ向きに出やすい。</li>"
             "<li>期待値の高い組がたくさん見えるのは、AI と市場の食い違いが 2 頭分で掛け算になるから。しかも、その組の多くは、組の市場が単勝の市場より低く見ている組である。</li></ol>")

    rows = []
    for s, nm in (("axis", "主: 軸だけ"), ("both", "上限: 2 頭とも")):
        for r in rs:
            rows.append([nm, f"{r:.2f}"] + [mark(g(k, m, s, r)) for k in ("umaren", "wide") for m in ("harville", "lbs")])
    o.append(f"<h2>段 1 (a) 上乗せの算数</h2><p>単勝の上乗せ r を、和 1 に直した後の値で植え、同じ型で組の確率の比を出した（{A['races']:,} レース・2022〜24・確定の q だけ）。"
             f"太字は控除の線 {f3(NEED)} 以上。単勝の s* 0.86 を掛けた線は {f3(NEED / 0.86)}。</p>")
    o.append(table(["型", "r", "馬連 Harville", "馬連 LBS", "ワイド Harville", "ワイド LBS"], rows))
    ab = A["bands"]
    ax_rows = []
    for kind, nm in (("umaren", "馬連"), ("wide", "ワイド")):
        B = T[f"{kind}|harville|axis|{rmax}"]["帯"]
        ax_rows.append([nm] + [f3(max(B[i])) if max(B[i]) > 0 else "－" for i in range(len(ab))])
    o.append(f"<p>軸の人気ごと（主の型・r {rmax:.2f}・Harville・相手の帯の中で最大）:</p>")
    o.append(table(["券種"] + [f"軸 {b} 番人気" for b in ab], ax_rows))

    m_rows = [[b, f3(M["umaren"]["m_over_H_sum_harville"][b]), f3(lbs_u[b]), f3(M["wide"]["m_over_H_sum_harville"][b]), f3(lbs_w[b])] for b in bands_pop]
    o.append(f"<h2>段 1 (b) 組の市場と単勝の市場</h2><p>確定の組のオッズの m ÷ 単勝の q からの組の確率（帯の和の比）。c13 と同じ平場の {M['umaren']['races']:,} レース・{M['umaren']['pairs']:,} 組（2022〜24）。"
             "Harville との差は、人気馬の 2・3 着を見すぎる型の偏りでも生じる。LBS で直すと帯の和ではほぼ 1 になる（組ごとにはばらつきがある）。</p>")
    o.append(table(["組の人気", "馬連 Harville", "馬連 LBS", "ワイド Harville", "ワイド LBS"], m_rows))
    e_rows = [[t, f"{M['win'][t]['per_race']:.2f}", f"{M['umaren']['ev_lbs_o'][t]['per_race']:.1f}", f"{M['wide']['ev_lbs_o'][t]['per_race']:.1f}",
               f3(M["umaren"]["ev_lbs_o"][t]["ai_over_m"]), f3(M["wide"]["ev_lbs_o"][t]["ai_over_m"]),
               f3(M["umaren"]["ev_lbs_o"][t]["m_over_Hq"]), f3(M["wide"]["ev_lbs_o"][t]["m_over_Hq"])] for t in map(str, M["ev_lines"])]
    o.append("<p>AI（c13 の凍結の組み合わせ・確定のオッズ・LBS。ワイドは下限）の期待値の線ごとの数と、その組の AI ÷ 組の市場・組の市場 ÷ 単勝からの組:</p>")
    o.append(table(["線", "単勝（頭/レース）", "馬連（組/レース）", "ワイド（組/レース）", "馬連 AI ÷ 組の市場", "ワイド AI ÷ 組の市場", "馬連 組の市場 ÷ 単勝から", "ワイド 組の市場 ÷ 単勝から"], e_rows))

    s_rows = [[b, f3(S["umaren"]["o"][b]["median"]), f3(S["wide"]["o"][b]["median"]), f3(S["wide"]["o_mid"][b]["median"])] for b in sb]
    o.append(f"<h2>組の s*（確定 ÷ T-60 のオッズ）</h2><p>2026-10-24 以降の {S['umaren']['races']} レース（手元の写し・着順なし）。選ぶ前の全部の組の中央値で、選んだ組ほど縮む分は入らない。門 0 の模擬の払戻の縮みにだけ使う。</p>")
    o.append(table(["T-60 の組の人気", "馬連", "ワイド（下限）", "ワイド（中点）"], s_rows))

    o.append("""<h2>この先</h2><ul>
<li>着順を使う方策の探索（軸の選び方・相手・点数・見送り）は、A の門と R0-12 の門 0 の両方を通ったときだけ、R0-12 の段 B の 10 回目の中で 1 回流す。型は深さ 2 以下の木か格子に限る（1 号の条件）。</li>
<li>出せるのは凍結した方策・検出力の模擬・前向きの事前登録の案だけ。100% 超と言える道は、凍結の後の発走前オッズでの前向きの記録だけ。</li>
</ul>
<h2>作り</h2><ul>
<li>組の確率: <code>src/research/rules_council/place_w.py</code> の <code>ordered_probs</code>（Harville (1, 1, 1)・LBS (1, 0.81, 0.65)）。</li>
<li>AI: めぐ AI v5bag（年ごとに前へ進める学習）と c13 の凍結の組み合わせ。q は確定で置いた（楽観の上限）。</li>
<li>限界: 確定のオッズの値。型の中の算数で、AI の誤差の 2 頭の間の相関は入れていない（上限の型は 2 頭の上乗せが独立の場合）。単勝の縮みの目安は母集団の違う 2 つの値の比で、1.16 はすでに見た値。</li>
<li>費用 0 円（手元のファイルだけ。GCS を読まない）。</li>
</ul>
</main>
</body>
</html>
""")
    return "\n".join(o)


def main() -> None:
    DST.write_text(build(), encoding="utf-8")
    print("書いた", DST)


if __name__ == "__main__":
    main()
