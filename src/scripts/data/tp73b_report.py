"""tp-73b の章 docs/html/modeling/chapters/similar_pairs_multi.html を、ホストで作った JSON（<TP73B_REPORT_DIR>）から作る。

python -m src.scripts.data.tp73b_report      # TP73B_REPORT_DIR（既定 /tmp/tp73b）に describe.json, joint.json, bet.json を置いて実行
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import norm

from src.scripts.data.jockey_switch_ev_report import CSS, table
from src.scripts.data.tp73_report import bh

ROOT = Path(__file__).resolve().parents[3]
SRC = Path(os.environ.get("TP73B_REPORT_DIR") or "/tmp/tp73b")
OUT = ROOT / "docs/html/modeling/chapters/similar_pairs_multi.html"
AXN = {"str": "力", "sty": "位置取り", "gate": "馬番", "apt": "適性", "front": "ともに先行", "type": "母系タイプ同じ"}
TGT = {"w3": "同時3着以内（ワイド）", "w2": "同時2着以内（馬連）"}
PER = {"disc": "発見 2022〜24", "test2025": "test 2025", "desc2026": "2026（記述）"}


def oe(x: dict | None) -> str:
    return "—" if not x else f"{x['oe']:.3f}（{x['lo']:.3f}〜{x['hi']:.3f}）"


def jl(n: str):
    return json.load(open(SRC / n, encoding="utf-8")) if (SRC / n).exists() else None


def main() -> None:
    d, jt, bt = jl("describe.json"), jl("joint.json"), jl("bet.json")
    body: list[str] = []
    body.append("""<h1>複数の軸が同時に近いペアは、さらに一緒に来やすいか（tp-73b。tp-41 の拡張）</h1>
<p class="muted">ユーザーの問い（2026-10-22）: tp-41（似た馬は着順が近いか）で、単一の軸（適性・馬番・位置取り）が近いペアは割引 Harville より少し一緒に上位に来やすかった。複数の軸が同時に近いと、効果はさらに伸びるか。軸の数の用量反応、2 軸の全組合せ、組合せ券の確率の改善、買い目としての評価を調べた。モデルの学習はしていない（集計・ロジスティック回帰・softmax の係数のみ）。市場オッズ・払戻は分析（統制・回収率）にだけ使った。</p>""")
    if d:
        body.append(f"""<div class="note"><b>結論（先に）.</b><ul>
<li><b>近い軸の数が多いほど、一緒に来やすい（用量反応は単調で、3 期間とも再現する）。</b>ワイド（同時3着以内）の観測/期待は、近い軸が 0 → 1 → 2 → 3 以上で {d['w3']['disc']['k']['0']['oe']:.3f} → {d['w3']['disc']['k']['1']['oe']:.3f} → {d['w3']['disc']['k']['2']['oe']:.3f} → {d['w3']['disc']['k']['3+']['oe']:.3f}（発見 2022〜24）、{d['w3']['test2025']['k']['0']['oe']:.3f} → {d['w3']['test2025']['k']['3+']['oe']:.3f}（2025）、{d['w3']['desc2026']['k']['0']['oe']:.3f} → {d['w3']['desc2026']['k']['3+']['oe']:.3f}（2026）。『ともに先行』が逆向きなので、それを除いた 5 軸の数ではさらに 0.96 → 1.08〜1.12。</li>
<li><b>ただし相乗ではなく加算（積）だった。</b>単一軸の効果から予想した値と 2 軸が同時に近いペアの観測は、ほぼ一致する。ロジット・オフセット GLM に『2 軸』『3 軸以上』の追加項を入れても係数は 0 と区別できない（|z|&lt;0.5）。2 軸の全 15 組合せの交互作用 30 本のうち BH を通るのは『力×適性』の負の 2 本（ワイド・馬連。発見 z=−2.8/−3.2、q=0.08/0.04）だけで、2025 年には再現しない（z=+0.1/+1.1）。</li>
<li><b>組合せ券の確率の改善は、並列条件（旗を重ねた項）を足しても、連続の単一軸の補正を超えない</b>（下表）。3 連複で連続軸＋d_str4 の補正（2025 年 −0.0173）が最も良く、並列条件を足すと −0.0159 とわずかに改善が減る（過学習の方向）。馬単の補正は、類似度が対称なので馬連と同じ改善になる（向きの情報がない）。</li>
<li><b>買い目としては、補正で回収率が −1〜+4pt 動く程度で、区間の下限は 0.75 以下。</b>確率×確定オッズ ≥ θ で選ぶ回収率は 0.75〜0.90 で、補正（連続軸＋d_str4）の差は区間に収まる。区間の上限が 100% を超えるのは馬連・馬単の 2025 年の一部（θ≥1.1）だけで、確率の改善は買い目の優位にならなかった。</li>
</ul></div>""")
    body.append("""<h2>1. 設計（事前に固定したもの）</h2><ul>
<li><b>軸（6 本）.</b> 力（d_str: as-of の強さの差）・位置取り（d_sty: 最初のコーナーの相対位置の差）・馬番（d_gate: 馬番の差/頭数）・適性（d_apt: 距離×芝割合）・ともに先行（both_front）・母系タイプが同じ（tp-73 の母系の高頻度種牡馬ベースのタイプ K=6、none は含めない）。力の差の V4 版（d_str4: V4 予測の複勝 logit の差）は、すべてのモデルで統制／補正に入れる。</li>
<li><b>外した軸.</b> 季節は同じレースの中では全馬が同じなので、ペアの『近さ』にならない。馬場適性（pref_hard）は被覆が 16〜19% と低く外した。</li>
<li><b>『近い』の閾値.</b> 連続の 4 軸は、発見期間（2022〜24）の下位 20%（d_str ≤ 0.234、d_sty ≤ 0.080、d_gate ≤ 0.133、d_apt ≤ 0.479）。ともに先行と母系タイプは二値。</li>
<li><b>期待.</b> V4 の p_win・p_top2・p_top3（master）に整合する割引 Harville（exotic_probabilities）の、ペアの『ともに3着以内』（ワイド）・『ともに2着以内』（馬連）の確率。観測/期待の区間はレース単位ブートストラップ。</li>
<li><b>検定.</b> 近い軸の数 k = 0・1・2・3+ の 4 区分、2 軸の全組合せ 15、交互作用 15×2 目標 = 30 本。ロジット・オフセット GLM（期待の logit をオフセット、レース単位のクラスタ頑健 SE）で、主効果（6 旗 + d_str4）に対し (a) k=2・k≥3 の追加項、(b) 2 軸の積を 1 つずつ追加。BH で補正。</li>
<li><b>期間.</b> 発見 2022〜24、確認 2025 を 1 回、2026 は記述のみ（選択に使わない）。ペアは 15,584 レース・1,412,561 件。</li>
<li><b>V5・v6e の OOF.</b> 馬ごとの予測を使える形で見つけられなかったため、V4（master の p_win 等。2022〜24 は OOF、2025・2026 は 2024 までの学習のテスト予測）で行った。V5・v6e での確認は未実施。</li></ul>""")
    if d:
        body.append("<h2>2. 近い軸の数と 観測/期待（用量反応）</h2><p>k は 6 軸のうち近い軸の数（事前の定義）。k5 は『ともに先行』を除く 5 軸の数（補足。both_front は tp-41 で逆向きと分かっていたので、事後に追加した）。値は 観測/期待（95% 区間）。</p>")
        for t in ("w3", "w2"):
            rows = []
            for lab in ("0", "1", "2", "3+"):
                rows.append([f"k={lab}"] + [oe(d[t][p]["k"][lab]) for p in PER] + [f"k5={lab}"] + [oe(d[t][p]["k5"][lab]) for p in PER])
            body.append(f"<h3>{TGT[t]}</h3>" + table(["近い軸の数 (6)"] + list(PER.values()) + ["5 軸"] + [v for v in PER.values()], rows, 1))
        body.append("<h3>2.1 単一軸（旗が立つペア）の 観測/期待</h3>")
        rows = [[AXN[a]] + [f"{d[t][p]['single'][a]['oe']:.3f}" for t in ("w3", "w2") for p in PER] for a in AXN]
        body.append(table(["軸"] + [f"{TGT[t][:6]} {PER[p]}" for t in ("w3", "w2") for p in PER], rows, 1))
        body.append("<h3>2.2 ロジット・オフセット GLM の主効果（発見 2022〜24。係数 / z）</h3>")
        rows = []
        for t in ("w3", "w2"):
            g = d[t]["glm"]
            rows.append([TGT[t]] + [f"{g['disc']['main'][k][0]:+.3f} / {g['disc']['main'][k][0] / g['disc']['main'][k][1]:+.1f}" for k in list(AXN) + ["d_str4"]])
        body.append(table(["目標"] + list(AXN.values()) + ["d_str4"], rows, 1))
        body.append("<p>馬番が近い（+）・ともに先行（−、ワイドのみ）・d_str4（力の差が大きいほど一緒に来にくい、−）が主な効果。適性・位置取りは弱い正。」</p>".replace("」", ""))
        body.append("<h2>3. 相乗か、加算か（k=2・k≥3 の追加項）</h2>")
        rows = []
        for t in ("w3", "w2"):
            for p in PER:
                c = d[t]["glm"][p]["count_terms"]
                rows.append([TGT[t], PER[p], f"{c['k2'][0]:+.3f}（z {c['k2'][0] / c['k2'][1]:+.1f}）", f"{c['k3p'][0]:+.3f}（z {c['k3p'][0] / c['k3p'][1]:+.1f}）", f"{c['llf_count'] - c['llf_main']:+.2f}"])
        body.append(table(["目標", "期間", "k=2 の追加項", "k≥3 の追加項", "対数尤度の増分"], rows, 2)
                    + "<p>主効果（各軸の効果の積）に『k が 2・3 以上』を足しても係数は 0 付近で、対数尤度もほとんど増えない。<b>効果は軸ごとの積で説明でき、相乗（掛け合わせて大きくなる）は見られない。</b>k が増えるほど O/E が上がるのは、軸ごとの小さな効果が重なるため。</p>")
        body.append("<h2>4. 2 軸の全組合せ</h2><p>2 軸が同時に近いペアの 観測/期待（発見 2022〜24）と、単一軸から予想した値（積／和）。交互作用 z は、主効果に 2 軸の積を足した GLM の係数の z（発見／2025）。BH は 30 本（15 組合せ×2 目標）で補正した q（発見）。</p>")
        rows = []
        allz = []
        for t in ("w3", "w2"):
            for k, v in d[t]["glm"]["disc"]["inter"].items():
                allz.append((t, k, v["z"]))
        qmap = {}
        pv = [2 * (1 - norm.cdf(abs(z))) for _, _, z in allz]
        for (t, k, z), q in zip(allz, bh(pv)):
            qmap[(t, k)] = q
        for k, v in d["w3"]["disc"]["two"].items():
            if not v:
                continue
            r = [k.replace("×", " × ").replace("str", AXN["str"]).replace("sty", AXN["sty"]).replace("gate", AXN["gate"]).replace("apt", AXN["apt"]).replace("front", AXN["front"]).replace("type", AXN["type"]),
                 f"{v['n']:,}"]
            for t in ("w3", "w2"):
                v2 = d[t]["disc"]["two"][k]
                gi, g25 = d[t]["glm"]["disc"]["inter"].get(k), d[t]["glm"]["test2025"]["inter"].get(k)
                r += [f"{v2['oe']:.3f} / {v2['mult']:.3f} / {v2['add']:.3f}", f"{gi['z']:+.1f}" if gi else "—", f"{g25['z']:+.1f}" if g25 else "—", f"{qmap.get((t, k), float('nan')):.2f}"]
            rows.append(r)
        body.append(table(["組合せ", "ペア数"] + [f"{TGT[t][:6]} {x}" for t in ("w3", "w2") for x in ("観測 / 積 / 和", "z 発見", "z 2025", "q")], rows, 1))
        body.append("<p>展開上の意味: 『ともに先行』を含む組合せは、ワイドではどれも 1 を下回る（先行どうしは展開で競合し、一緒に来にくい）。馬連（1-2 着）では、馬番・適性が近い先行馬の組は 1 を少し超え、先行どうしが 1-2 着を分け合う形は出る。そのうえで位置取り・馬番・適性が近い先行馬どうしは、さらに来にくくなる傾向（位置取り×先行 0.93、馬番×先行 0.94）。『力×適性』の負（観測 1.035 に対し積 1.094）は、力が近くて適性も近いペアでは、力の近さの効果が適性の近さで一部吸収されるという解釈ができるが、2025 年には再現しなかった。母系タイプが同じことは、他の軸に上乗せする効果が小さい（|z|&lt;2.5）。</p>")
    if jt:
        names = {"base": "補正なし", "d_str4": "力の差（d_str4）のみ", "single": "単一軸（連続 5 本）", "single+d4": "単一軸＋d_str4", "parallel": "並列条件（旗 6 + 積 15 + k 項）", "parallel+d4": "並列条件＋d_str4",
                 "single+par": "単一軸＋並列条件", "single+par+d4": "単一軸＋並列条件＋d_str4"}
        body.append("<h2>5. 組合せ券の確率の改善（対数損失。補正なしとの差、負が改善）</h2><p>各年の前の年で係数を当てはめる（2024: 2022〜23、2025: 2022〜24 を 1 回、2026: 2022〜25 の記述）。馬連・3 連複・馬単は『1 つの組合せが起きる』softmax の 1 レースあたり負の対数尤度、ワイドは 1 ペアあたりの二値の対数損失。区間はレース単位ブートストラップ。補正は割引 Harville の log 確率に、ペアの特徴の和を足す形（tp-41 と同じ）。</p>")
        for name in ("馬連", "ワイド", "3連複", "馬単"):
            rows = []
            for v in names:
                f = jt["res"][name].get(v)
                if not f:
                    continue
                rows.append([names[v]] + [f"{f[fo]['diff']:+.5f}（{f[fo]['ci'][0]:+.5f}〜{f[fo]['ci'][1]:+.5f}）" for fo in ("2024", "2025", "2026")])
            base = jt["res"][name]["base"]
            body.append(f"<h3>{name}（補正なしの対数損失: 2024 {base['2024']['nll']:.4f}／2025 {base['2025']['nll']:.4f}／2026 {base['2026']['nll']:.4f}）</h3>" + table(["補正", "2024", "2025（1 回）", "2026（記述）"], rows, 1))
        body.append("<p>読み方: 並列条件だけ（旗を重ねた粗い項）は連続の単一軸より弱く、馬連では改善がほぼ 0。旗は『下位 20% かどうか』に情報を落とすため、連続の距離を使う単一軸のほうが良い。単一軸に並列条件を足しても上乗せは無く、むしろ改善が減る（2025 年の馬連 −0.0094 → −0.0070、3 連複 −0.0173 → −0.0159。係数が増えて過学習する方向）。最も効くのは力の差（d_str4）で、3 連複で −0.012〜−0.014、類似度の軸を足すと −0.017〜−0.019（tp-41 の 2024 年 −0.0166 と整合）。馬単は馬連と同じ値になる（ペアの向きの情報を持たない）。</p>")
    if bt:
        body.append("<h2>6. 買い目としての評価（確率 × 確定オッズ ≥ θ）</h2><p>補正した確率（上と同じ当てはめ）に確定オッズ（odds_pair。ワイドは下限側）を掛けた期待値が θ 以上の組を、1 点 1 単位ずつ買う。確定の払戻で精算。レース単位ブートストラップの 95% 区間。確定オッズは分析にだけ使い、予測の入力にはしていない。3 連複は確定オッズの表が無く評価できない。</p>")
        for name in ("馬連", "ワイド", "馬単"):
            if name not in bt:
                continue
            rows = []
            for v, f in bt[name].items():
                for fo, tt in f.items():
                    r = [names.get(v, v), fo]
                    for th, x in tt.items():
                        r.append("—" if x["roi"] is None else f"{x['roi'] * 100:.1f}%（{x['ci'][0] * 100:.0f}〜{x['ci'][1] * 100:.0f}）n={x['bets']:,}")
                    rows.append(r)
            body.append(f"<h3>{name}</h3>" + table(["補正", "年", "θ=1.0", "θ=1.1", "θ=1.2"], rows, 2))
    body.append("""<h2>7. 結論と限界</h2><ul>
<li>複数の軸が同時に近いほど一緒に上位に来やすいのは確かで、近い軸が 0 のペアと 3 つ以上のペアの間で観測/期待が 0.96 → 1.05〜1.09（5 軸では 1.08〜1.12）と 10〜15% 開く。これは 3 期間で再現する。ただし<b>軸ごとの小さな効果の積で説明でき、相乗ではない</b>。</li>
<li>組合せ券の確率は、連続の単一軸と力の差（d_str4）の補正で十分で、並列条件（旗の重ね）を足しても有意に良くならない。</li>
<li>買い目としては、補正で回収率が −1〜+4pt 動く程度（馬連 2025 年 θ=1.0 で 78.8% → 82.7% など）で、区間の下限が 100% に届く設定は無く、確率の改善にとどまる（tp-41 と同じ結論）。</li>
<li>限界: V4 のみ（V5・v6e は未確認）。季節・馬場適性は軸にできなかった。『近い』の閾値は 1 通り（下位 20%）。3 連複の買い目評価は確定オッズが無く未実施。</li>
<li>次の案: 連続の軸どうしの積（旗にしない）、V5・v6e の確率での再確認、展開予想（脚質の組合せ）を軸に足す。</li></ul>""")
    body.append('<p class="muted">コード: <code>src/scripts/data/tp73b_multi.py</code>（pairs / describe / joint / bet）、この章の生成: <code>tp73b_report.py</code>。ホストの出力: <code>/home/jovyan/work/tp73b_out/</code>。tp-41 の章: <a href="similar_horses_pairs.html">similar_horses_pairs.html</a>。</p>')
    html = f'<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>複数の軸が同時に近いペア（tp-73b）— keiba-vpn</title><style>{CSS}</style></head><body><a href="../../index.html">← ドキュメント一覧</a>{"".join(body)}</body></html>'
    OUT.write_text(html, encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
