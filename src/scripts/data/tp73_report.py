"""tp-73 の章 docs/html/modeling/chapters/pedigree_side_cohit.html を、ホストで作った JSON（<TP73_REPORT_DIR>）から作る。

python -m src.scripts.data.tp73_report      # TP73_REPORT_DIR（既定 /tmp/tp73）に thin*.json, profile*.json, dose*.json, pair*.json, rlp*.json, pop_*.parquet を置いて実行
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.scripts.data.jockey_switch_ev_report import CSS, e, table

ROOT = Path(__file__).resolve().parents[3]
SRC = Path(os.environ.get("TP73_REPORT_DIR") or "/tmp/tp73")
OUT = ROOT / "docs/html/modeling/chapters/pedigree_side_cohit.html"


def J(name: str):
    f = SRC / name
    return json.load(open(f, encoding="utf-8")) if f.exists() else None


def bh(p) -> np.ndarray:
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty(len(p))
    r = p[o] * len(p) / (np.arange(len(p)) + 1)
    q[o] = np.minimum.accumulate(r[::-1])[::-1].clip(max=1)
    return q


def bp(x) -> str:
    return "—" if x is None else f"{x * 1e4:+.1f}"


def fold_mean(v: dict, k: str) -> float:
    return float(np.mean([f[k] for f in v["folds"].values()]))


def thin_table(thin: dict, resp: list[str], S: str = "S50", c: str = "c0.2") -> str:
    rows = []
    for r in resp:
        for var in ("decay", "presence", "weak"):
            row = [r, {"decay": "減衰（Σ2^-世代）", "presence": "ありなし", "weak": "弱い減衰（√）"}[var]]
            for side in ("mat", "pat"):
                v = thin.get(f"{side}|{S}|{var}|{r}|{c}")
                if v is None:
                    row += ["—"] * 5
                    continue
                t25 = v.get("test2025", {})
                row += [bp(fold_mean(v, "cond")), bp(fold_mean(v, "flat")), bp(fold_mean(v, "agg")) if "agg" in next(iter(v["folds"].values())) else "—",
                        f"{v['t_df']:.1f}", bp(t25.get("flat"))]
            rows.append(row)
    head = ["応答", "深い世代の重み"] + [f"{s}{x}" for s in ("母系 ", "父系 ") for x in ("条件別", "条件なし", "合計×条件", "t(なし)", "2025 なし")]
    return table(head, rows, 2)


def main() -> None:
    thin2, thin1, strong2 = J("thin_s2.json"), J("thin.json"), J("thin_strong_s2.json")
    prof2, prof1 = J("profile_s2.json"), J("profile.json")
    dose2, dose1 = J("dose_s2.json"), J("dose.json")
    pair2, pair1 = J("pair_s2.json"), J("pair.json")
    rlp2 = J("rlp_s2.json")
    pops = {s: pd.read_parquet(SRC / f"pop_{s}.parquet") for s in ("mat", "pat")}
    body: list[str] = []

    # --- 高頻度の祖先
    def top_list(side: str, lo: float, hi: float, n: int = 12) -> pd.DataFrame:
        p = pops[side]
        el = p[(p.share_pop >= lo) & (p.share_pop <= hi)].sort_values("n_pop", ascending=False).reset_index(drop=True)
        return el

    names_check = {"Ribot": "Ribot", "ノーザンテースト": "ノーザンテースト", "サンデーサイレンス": "サンデーサイレンス", "Halo": "Halo"}
    rank_rows = []
    for key in names_check:
        row = [key]
        for side in ("mat", "pat"):
            for lo, hi in ((0.02, 0.80), (0.05, 0.60)):
                el = top_list(side, lo, hi)
                m = el[el.name.str.contains(key, na=False)]
                if len(m):
                    i = int(m.index[0])
                    row.append(f"{i + 1} 位（母集団の {m.iloc[0].share_pop * 100:.0f}%）" if i < 300 else f"{i + 1} 位")
                else:
                    row.append("範囲外")
        rank_rows.append(row)
    top2 = {s: top_list(s, 0.05, 0.60).head(12) for s in ("mat", "pat")}
    topn_rows = [[i + 1] + [f"{top2[s].iloc[i]['name'][:22]}（{top2[s].iloc[i].share_pop * 100:.0f}%）" for s in ("mat", "pat")] for i in range(12)]

    body.append(f"""
<h1>血統の高頻度の種牡馬（母系・父系）の特性と、血統クラスタ対の同時好走（tp-73）</h1>
<p class="muted">ユーザーの問い（2026-10-22）: ①母系の深い位置（6〜10 世代）に、世の中に残っている数の多い種牡馬（Ribot、ノーザンテーストなど）がいる馬は、血が薄くても競馬場・距離・馬場・レース質・ペースの傾向が出るか。母系が細かい適性に関与するのか（対照として父系も同じ手順）。②その種牡馬の特性をクラスタリングして、任意の馬の血統表から「どのタイプか」を出せるか。③血統クラスタの対で一緒に好走（馬連・ワイド・3 連複で買える組合せ）しやすいものはあるか。④登場回数が平均以上のとき特性がより出るか（用量反応）。応答は、成績の母集団平均ではなく<b>馬自身の平均に対する条件ごとの相対成績</b>（馬内の対比）を主にした。モデルの学習はしていない（集計・リッジ回帰・クラスタリング）。</p>
<div class="note"><b>結論（先に）.</b>
<ul>
<li><b>「母系は血が薄くても効く」は成り立たなかった。差が出たのは父系の側だった。</b>深い世代（6〜10 世代）の高頻度の種牡馬が、馬内の相対適性（着順の正規スコア）を保留した年で説明する大きさは、強い縮約（c=20）で母系 +1.1bp（t=3.1）、父系 +3.8bp（t=8.0、2025 年 +5.1bp）。縮約を弱めると父系 +7bp（2025 年 +14bp）まで伸びるが、母系は過学習で負。いずれも二乗誤差の 0.01〜0.14% で、実用の水準ではない。<b>現行モデル（V4）の残差には、どちらも上乗せがない</b>（y_v4 で 0 付近）ので、モデルはすでに織り込んでいる。</li>
<li>祖先ごとの特性プロファイルを 2020〜2021 年で固定して後の年に当てると、母系はどの年も予測力がなく（2025 年 t=+1.1）、父系は 2022 年 t=+3.9 → 2025 年 −0.2 と薄れる。同じ年を含めて当てはめると父系は効くので、父系のプロファイルは年とともに古くなる（定期的な更新が要る）。「母集団で単純に集計」した版は、能力（クラス）の差を拾って<b>大きく見える</b>（条件なし +7〜+13bp、相対適性の 5〜10 倍）が、能力と適性を取り違えている。</li>
<li>登場回数 n の用量反応は、母系ではどの軸も再現しない。父系は距離帯・芝ダ・ペースの一部が発見期間で有意（BH）で、芝ダ軸の『深い世代 1 回』とペース軸の『平均以上』は 3 期間で同符号だが、効きは小さい。n≥2 と n=1 の差の祖先別検定は父系で過剰に有意（40/250、期待 12）で、原因は未解明（採用しない）。</li>
<li>血統クラスタ対の同時好走は、市場の織り込みを除くと lift が 1.0〜1.13 にとどまり、事前登録の検定（約 1,900 本）は偶然の想定数程度しか有意にならない。上位の対を馬連・ワイドで買った回収率は 2025 年で馬連 36〜66%、ワイド 54〜63%。<b>買える対は見つからなかった</b>。</li>
</ul></div>
""")

    body.append("<h2>1. 設計（事前に固定したもの）</h2>")
    body.append(f"""<ul>
<li><b>側.</b> 母系 = 馬の母から始まる木（母、母父、母母、…）の 10 世代以内。父系 = 馬の父から始まる木（父、父父、父母父、…）。男の祖先（誰かの父の位置）だけを数える。同じ種牡馬が複数回出る（クロス）は別の登場として数える。母系の分析で父系を、父系の分析で母系を、血量の統制変数に入れた。</li>
<li><b>残存数（高頻度）.</b> 母集団（35,794 頭）のうち、その側の 10 世代以内にその祖先を持つ馬の頭数 n_pop（延べ登場数・出走馬の頭数・生年で切った as-of の頭数も併記して保存）。上位 N（20・<b>50</b>・100）を使う。母集団の割合が 80% を超える創始者（Tetratema・Sunstar など約 79% の古い祖先）は区別に使えないので外す。<b>範囲の定義は 2 通り</b>: 定義 1（2〜80%、最初に決めた）と定義 2（5〜60%、Ribot・Halo を含み、ほぼ全馬が持つ古い創始者を外す）。定義 1 は上位が 79% の古い祖先で埋まり、ユーザーの例（Ribot・ノーザンテースト）を含まなかったため、<b>祖先の一覧を見たあと（成績の結果は見ずに）に定義 2 を足した</b>。結果は定義 2 を主、定義 1 を補助とする。</li>
<li><b>応答.</b> 主は R_pro = 着順の正規スコア（Φ⁻¹((頭数−着順+0.5)/頭数)）の、馬ごとの as-of 平均（疑似走数 3 で 0 へ縮約、最低 3 走）に対する残差。ほかに R_tim（タイム偏差）、R_meg（めぐ指数 v3a のレース内相対）。比較として、従来の母集団集計の y_old = 3 着内 − 3/頭数、V4 の OOF 残差 y_v4 = 3 着内 − p_top3（2022 年〜）。</li>
<li><b>条件軸.</b> 競馬場（10）・距離帯（4）・馬場（4）・芝ダ（2）・ペース（3。実測のペース分類で、事前には分からないオラクル）の計 23 水準。</li>
<li><b>深い世代の重みの比較.</b> 減衰（Σ 2^-世代）／ありなし／弱い減衰（√）。近い世代（母系は 2〜5 世代、父系は 1〜5）の血量と、もう一方の側の血量は統制。</li>
<li><b>評価.</b> 保留した年の二乗誤差の減少（対「統制だけ」）。発見 2020〜2024 は LOYO、2025 は 1 回、2026 は記述のみ。リッジは標準化した列で λ=c×行数（c=0.2 主、0.02 補助。過学習だったので、結果を見たあとに強い縮約 c=2・20 を追加した）。条件別（祖先×23 水準）／条件なし（祖先ごと）／合計×条件（祖先を区別せず深い世代の合計×23 水準）の 3 モデル。</li>
<li><b>プロファイル.</b> 祖先ごと・水準ごとの「祖先固有の偏り」= 縮約した水準の平均 − その祖先の全体の平均 − 全馬の水準の偏り（共通の条件の効果を引く。縮約は k=200 で『祖先の効果なし』の期待値へ）。2020〜2021 年のデータで固定（as-of）。上位 100 祖先を K=6 でクラスタリングし、馬のタイプは血量の合計が最大のクラスタ（合計 0.02 未満は none）。系統→全体の階層縮約のうち「系統」は、祖先の系統を引く表がないため<b>未実装</b>で、祖先→全体のみ。</li>
<li><b>ペア.</b> 割り当て 6 種（A 埋め込み K=16・B 埋め込み K=40・C 母系タイプ（相対／母集団の 2 版）・D 父系タイプ（同））。同一レースの 3 着以内（連対）の馬の対のクラスタ対の観測数を、市場の勝率（Harville）・モデル（V4）・一様の期待数と比べた lift。検定の統計量は、レースごとの（観測−期待）の和／√（二乗和）（レース単位のクラスタ頑健）。期待数 5 未満は除外。事前登録の検定は、割り当て×(3 着内・連対) の約 1,900 本。BH は全体と割り当て別。</li>
</ul>""")

    body.append("<h2>2. 高頻度の祖先（残存数）</h2>")
    body.append("<p>定義 2（母集団の 5〜60%）の残存数の上位。ユーザーの例の位置（定義 1 / 定義 2 それぞれの範囲内の順位）も示す。</p>")
    body.append(table(["順位", "母系（母から）", "父系（父から）"], topn_rows, 1))
    body.append(table(["祖先", "母系 定義1", "母系 定義2", "父系 定義1", "父系 定義2"], rank_rows, 1))
    body.append('<p class="muted">残存数は血統表の構造だけで決まり、成績を使わない（リークなし）。生年で切った as-of のカウント（2019・2021・2023 年生まれまで）も <code>pop_*.parquet</code> に保存した。</p>')

    # --- 血が薄くても効くか
    body.append("<h2>3. 『血が薄くても効く』は成り立つか（深い世代 6〜10）</h2>")
    body.append("<p>各セルは、保留した年の二乗誤差の減少（単位 bp = 0.01%。正が良い）。LOYO は発見 5 年（y_v4 のみ 2022〜24 の 3 年）の平均。『条件別』は祖先×23 水準の積（過学習すると負）、『条件なし』は祖先ごとの 1 列、『合計×条件』は祖先を区別せず深い世代の合計×23 水準。定義 2・上位 50・c=0.2。</p>")
    if thin2:
        body.append(thin_table(thin2, ["R_pro", "R_tim", "R_meg", "y_old", "y_v4"]))
        body.append("<h3>3.1 上位 N と縮約の強さ（R_pro）</h3>")
        rows = []
        for S in ("S20", "S50", "S100"):
            for side in ("mat", "pat"):
                v = thin2.get(f"{side}|{S}|decay|R_pro|c0.2")
                if v:
                    rows.append([S, "母系" if side == "mat" else "父系", bp(fold_mean(v, "cond")), bp(fold_mean(v, "flat")), f"{v['t_df']:.1f}", bp(v['test2025'].get('flat')), bp(v['test2025'].get('cond'))])
        body.append(table(["上位 N", "側", "条件別 LOYO", "条件なし LOYO", "t(なし)", "条件なし 2025", "条件別 2025"], rows, 2))
        if strong2:
            body.append("<h3>3.2 強い縮約（c=2・20。結果を見たあとに追加）</h3>")
            rows = []
            for side in ("mat", "pat"):
                for resp in ("R_pro", "y_v4"):
                    for c in ("c2.0", "c20.0"):
                        v = strong2.get(f"{side}|S50|decay|{resp}|{c}")
                        if v:
                            rows.append(["母系" if side == "mat" else "父系", resp, c, bp(fold_mean(v, "cond")), bp(fold_mean(v, "flat")), bp(v["test2025"].get("cond")), bp(v["test2025"].get("flat"))])
            body.append(table(["側", "応答", "縮約", "条件別 LOYO", "条件なし LOYO", "条件別 2025", "条件なし 2025"], rows, 3))
    body.append("""<p><b>読み方.</b></p><ul>
<li>縮約が弱い（c=0.2・0.02）と、条件別モデルは過学習で、どの応答でも「統制だけ」より悪い。縮約を強める（c=2・20。結果を見たあとに追加）と、父系は R_pro で +7bp（c=2）・+3.8bp（c=20）と正になり、2025 年でも +14bp・+5bp と再現する。母系は c=20 で +1.1bp とほぼ 0。<b>祖先を区別せず合計×条件にしたモデル（agg）は負</b>なので、効きは祖先ごとの違いから来ている。</li>
<li>条件なし（祖先ごとの 1 列）は、母系が −2〜+1bp、父系が −2〜+3bp（R_pro・R_tim は +1〜+3bp、t=2〜3）。深い世代にいること自体がわずかに押し上げる効果は父系のみ。</li>
<li>V4 の残差（y_v4）に対しては、どちらの側も条件別・条件なしとも 0 付近（|t|≲2）。<b>現行モデルは、この血統情報をすでに持っている</b>。</li>
<li>例外は父系×R_meg（めぐ指数の相対）の条件別で +37〜+41bp（t≈5〜6、2025 年は +47〜+73bp）。同じ成績を着順（R_pro）やタイム（R_tim）で測ると出ない。めぐ指数は馬場・コースの補正を内部に持つため、<b>指数の作り方に依存した見かけの可能性が高い</b>（確認は未了。採用しない）。</li>
<li>母集団集計（y_old）は、条件なしで +7〜+13bp と<b>相対適性（R_pro）の 5〜10 倍大きく見える</b>。これは能力・クラスの差（その祖先を持つ馬が強い／弱い）を拾うためで、「平たくなる」のではなく「能力を適性と取り違える」方向のずれ。馬内の対比は、この能力成分を引くので小さくなる。</li>
<li>『深い世代だけの馬』の部分集合は、高頻度の祖先 50 頭のどれかが 6〜10 世代にいる馬が全体の 100% で、制約にならなかった（母集団の 5〜60% を占める祖先は、ほぼ全馬のどこかの深い位置にいる）。そこで、重みの比較（減衰・ありなし・弱い減衰）と、祖先ごとの条件別の傾きで見た。</li>
<li>pedigree_roi §7（V4/V5 の残差、種牡馬を絞らず血量の合算）の「6〜10 世代はほとんど効かない」と、今回の結果は矛盾しない。違いは、(1) 母系／父系を分けた、(2) 残存数で絞った祖先に限った、(3) 応答を馬内の相対適性にした、(4) 条件別の傾きも見た点で、(2)〜(4) で父系にだけ小さな効きが見えたが、V4 の残差には現れない。</li>
</ul>""")

    # --- プロファイルとクラスタ
    body.append("<h2>4. 特性プロファイルとクラスタ（上位 100 祖先・K=6、2020〜2021 年で固定）</h2>")
    if prof2:
        for side, ttl in (("mat", "母系"), ("pat", "父系")):
            rows = []
            for c in prof2[side]["R_pro"]["clusters"]:
                up = "、".join(f"{k}（{v:+.3f}）" for k, v in c["up"])
                dn = "、".join(f"{k}（{v:+.3f}）" for k, v in c["down"])
                rows.append([c["k"], c["n_anc"], "、".join(c["members"][:5]), up, dn])
            body.append(f"<h3>{ttl}</h3>" + table(["クラスタ", "祖先数", "代表", "出やすい条件（偏り）", "出にくい条件"], rows, 1))
        body.append("<p class=\"muted\">水準の記号: venue 01 札幌・02 函館・03 福島・04 新潟・05 東京・06 中山・07 中京・08 京都・09 阪神・10 小倉。偏りの単位は着順の正規スコア（馬の平均に対する残差）で、最大でも 0.03 程度（約 0.03 標準偏差）。</p>")
        rows = []
        for side in ("mat", "pat"):
            for basis, nm in (("R_pro", "相対適性（馬内の対比）"), ("y_old", "母集団集計")):
                v = prof2[side][basis]["validation"]
                rows.append(["母系" if side == "mat" else "父系", nm] + [f"{v[y]['cluster']['t']:+.1f}" for y in ("2022", "2023", "2024", "2025", "2026")])
        body.append("<h3>4.1 プロファイルは、後の年の馬内の相対適性を予測するか（クラスタ割り当てによる予測の傾きの t。条件の固定効果を統制）</h3>" + table(["側", "プロファイルの基準", "2022", "2023", "2024", "2025", "2026（記述）"], rows, 2))
        body.append("<p>母系は 2023〜2025 年で |t|≲3 と、年によって符号が変わり、予測力は無い。父系は 2022 年に t=+3.9 で、年とともに薄れて 2025 年は −0.2。<b>母系だけが細かい適性に関与する、という見立ては、この指標では支持されなかった</b>。なお、全馬の条件の効果を引く前の最初の版では t=+10〜+15 と見えたが、全馬に共通の条件の偏り（その条件で走る馬の平均成績の偏り）を拾っただけだった（本章は共通成分を引いた版）。</p>")
    if rlp2:
        rows = [[k.split("|")[0].replace("mat", "母系").replace("pat", "父系"), "相対適性" if "R_pro" in k else "母集団集計", k.split("|")[2], v["n_anc"], f"{v['pooled_r']:+.2f}", f"{v['median_r']:+.2f}"] for k, v in rlp2.items()]
        body.append("<h3>4.2 既存の role_lift_profiles（maternal_deep）との関係</h3><p>競馬場ごとの偏り（lift_win_v_＊ − その祖先の平均）と、今回のプロファイルの競馬場の偏りの相関（共通の祖先のみ）。</p>" + table(["側", "今回の基準", "既存の役割", "共通の祖先数", "相関（全点）", "相関の中央値（祖先別）"], rows, 3)
                    + "<p>相対適性の版は既存の lift とほぼ無相関（+0.0〜+0.1）、母集団集計の版は弱く相関（+0.25〜+0.31）。既存の lift は勝率の母集団集計で、能力の差を含む。</p>")

    # --- 用量反応
    body.append("<h2>5. 登場回数 n の用量反応（第 2 段）</h2>")
    body.append("<p>各祖先の『その条件でのプロファイルの偏り』を、その馬が持つ登場回数のビンで集めた特徴量（近い世代 1 回／2 回以上、深い世代 1／2／3 回以上、n が平均以上／未満）で、馬内の相対適性 R_pro を回帰した（条件の固定効果・近い世代の血量・もう一方の側の血量を統制、レース単位のクラスタ頑健）。型は 2 側×5 軸×7 係数 = 70 本（発見 2022〜2024 で BH、2025 を 1 回、2026 は記述）。祖先ごとの検定（50 祖先×5 軸 = 250 本。n≥2 と n=1 の一致度の差）も行った。</p>")
    for dose, ttl in ((dose2, "定義 2（主）"), (dose1, "定義 1")):
        if not dose:
            continue
        rows = []
        for side in ("mat", "pat"):
            c = pd.DataFrame(dose[f"{side}_coef"])
            d = c[(c.resp == "R_pro") & (c.period == "disc") & c.grp.isin(["bins", "ab"])].copy()
            d["p"] = 2 * (1 - norm.cdf(d.t.abs()))
            d["q"] = bh(d.p)
            for _, x in d[d.q < 0.1].iterrows():
                cc = c[(c.resp == "R_pro") & (c.fam == x.fam) & (c.coef == x.coef) & c.grp.isin(["bins", "ab"])]
                rows.append(["母系" if side == "mat" else "父系", x.fam, x.coef, f"{x.t:+.1f}", f"{float(cc[cc.period == 'test2025'].t.iloc[0]):+.1f}", f"{float(cc[cc.period == 'desc2026'].t.iloc[0]):+.1f}"])
        body.append(f"<h3>{ttl}: 発見期間で BH(q&lt;0.1) を通った係数と、その後の t</h3>" + (table(["側", "条件軸", "係数", "発見 t", "2025 t", "2026 t"], rows, 3) if rows else "<p>なし。</p>"))
    anc_rows = []
    for dose, ttl in ((dose2, "定義 2"), (dose1, "定義 1")):
        if not dose:
            continue
        for side in ("mat", "pat"):
            a = pd.DataFrame(dose[f"{side}_anc"])
            a["pd"] = 2 * (1 - norm.cdf(a.t_dose.abs()))
            a["pm"] = 2 * (1 - norm.cdf(a.t_match.abs()))
            anc_rows.append([ttl, "母系" if side == "mat" else "父系", len(a), int((a.t_match.abs() > 2).sum()), int((bh(a.pm.fillna(1)) < 0.1).sum()), int((a.t_dose.abs() > 2).sum()), int((bh(a.pd.fillna(1)) < 0.1).sum())])
    body.append("<h3>祖先ごとの検定（発見 2022〜2024）</h3>" + table(["定義", "側", "検定数", "一致度 |t|&gt;2", "同 BH 通過", "用量差 |t|&gt;2", "同 BH 通過"], anc_rows, 2))
    body.append("<p>読み方: 母系は、70 本のうち BH を通るのは定義 2 で 2 本（芝ダ軸の『3 回以上』『平均以上』）で、2025 年は t=+2.0・+0.1、2026 年は負で再現しない。父系は 10 本が通り、芝ダ軸の『深い世代 1 回』（発見・2025・2026 の t=+4.1/+3.8/+2.4）とペース軸の『平均以上』（+2.3/+2.7/+1.6）は 3 期間で同符号。距離帯・芝ダ軸の多くは 2025 年に大きく縮む。祖先ごとの検定では、プロファイルとの一致度は母系が 250 本中 |t|&gt;2 が 0 本、父系が 9 本（BH 通過 0）で、個別の祖先の傾きは検出できない。n≥2 と n=1 の差は、父系で |t|&gt;2 が 40 本（偶然の期待 約 12）・BH 通過 25 本と過剰だが、一致度そのものは有意でなく、原因は未解明（n≥2 の馬が特定の血統・クラスに偏る可能性）。母系は 14 本（BH 0）。<b>母系だけ n が多いほど特性が出る、という用量反応は見えなかった。</b></p>")

    # --- ペア
    body.append("<h2>6. 血統クラスタ対の同時好走</h2>")
    for pair, ttl in ((pair2, "定義 2（主）"), (pair1, "定義 1")):
        if not pair:
            continue
        rows = []
        for k, v in pair["z_summary"].items():
            a, t = k.split("|")
            rows.append([a, "3 着内の対" if t == "3" else "連対（1-2 着）", v["n"], f"{v['mean_z']:+.2f}", f"{v['sd_z']:.2f}", f"{v['max_z']:.2f}", pair["sig_assign_q10"][k], pair["sig_assign_q10_lift_gt1"][k]])
        body.append(f"<h3>{ttl}: 検定 {pair['n_tests']} 本（{pair['n_races']:,} レース）、全体 BH(q&lt;0.1) = {pair['sig_global_q10']} 本</h3>"
                    + table(["割り当て", "対象", "検定数", "z の平均", "z の sd", "z の最大", "q&lt;0.1（割り当て内）", "うち lift&gt;1"], rows, 2))
    if pair2:
        rows = []
        for k, v in pair2["confirm_top5"].items():
            d, t, g = v.get("disc", {}), v.get("test2025", {}), v.get("desc2026", {})
            ro = v.get("roi_test2025")
            rows.append([k.replace("|3", "（3 着内）").replace("|2", "（連対）"), f"{d.get('lift', float('nan')):.3f}", f"{t.get('lift', float('nan')):.3f}（z {t.get('z', float('nan')):+.1f}）", f"{g.get('lift', float('nan')):.3f}",
                         (f"馬連 {ro['umaren']['roi'] * 100:.0f}%（{ro['umaren']['lo'] * 100:.0f}〜{ro['umaren']['hi'] * 100:.0f}）／ワイド {ro['wide']['roi'] * 100:.0f}%（{ro['wide']['lo'] * 100:.0f}〜{ro['wide']['hi'] * 100:.0f}）" if ro else "—")])
        body.append("<h3>発見期間の z 上位 5 対（期待数 10 以上・lift&gt;1）を、2025 年で 1 回評価（定義 2）</h3>" + table(["割り当て", "発見の lift", "2025 の lift", "2026 の lift（記述）", "2025 の回収率（そのクラスタ対の全組合せを 100 円ずつ）"], rows, 1))
        body.append("<p>市場の勝率（Harville）を除いた lift。lift の期待値は 1。発見期間で上位に選んだ対は、2025 年に lift が縮む（B の連対 1.39→0.96 など。典型的な平均への回帰）。回収率は馬連 36〜66%・ワイド 54〜63% で、払戻率（約 77%）を下回る。</p>")
        rows = []
        for k, v in pair2["feature_prospect_2025"].items():
            a, r = k.split("|")
            rows.append([a, r, f"{v['slope']:+.2f}", f"{v['t']:+.1f}", f"{v['slope_ctrl_own']:+.2f}", f"{v['t_ctrl_own']:+.1f}", f"{v['sd_f']:.3f}"])
        body.append("<h3>特徴量化の簡易評価（2025 年のみ、発見期間の対の lift から作る『同レースの他馬との平均 lift−1』の傾き）</h3>"
                    + table(["割り当て", "応答", "傾き", "t", "傾き（自分のクラスタの平均成績を統制）", "t", "特徴量の sd"], rows, 2)
                    + "<p>36 本（12 割り当て×3 応答）のうち、V4 の残差 y_v4 で t が 2 を超えるのは A（K=16）2.7、B（K=40）3.0、C（母系・母集団基準）2.4。自分のクラスタの平均成績を統制しても残るのは B（2.6）と C（2.4）。傾きは 0.1〜0.3 で、特徴量の 1 sd あたりの残差の変化は小さい（下表の sd_f を参照）。2025 年 1 回・36 本の多重性を考えると弱い手がかりにとどまり、LOYO での再現（二段目の基準）は未確認。<b>特徴量としては保留</b>（採用しない）。本格的な再学習はしていない。</p>")

    body.append("<h2>7. 限界と、やらなかったこと</h2><ul>"
                "<li>モデルの比較相手は V4 の OOF（p_win・p_top3。2022 年〜）。V5・v6e の馬ごとの予測はここには無く、V5 以降でも結論が変わらないかは未確認（pedigree_roi §7 では V4/V5 で同じ結論）。</li>"
                "<li>展開面の補完（逃げ×差しなど）の仮説は検定していない。市場の織り込みの仮説は、市場の勝率で期待数を作ることで除いた（lift_mkt）。同じ適性の仮説は、競馬場・距離・馬場などの層別に分けた記述のみ。</li>"
                "<li>3 連複（三つ組）は評価していない。ペアの規則は馬連・ワイドの 2 頭組に限る。</li>"
                "<li>祖先の系統による階層縮約（種牡馬→系統→全体）の『系統』は未実装。クラスタ数は K=6 の 1 通りで、決定的でない（シード固定）。</li>"
                "<li>血統クラスタ（A・B）の k-means は全馬（2026 年生まれを含む）の血統表だけで作る。成績を使わないため結果のリークは無い。</li>"
                "<li>事後に追加したもの: 定義 2（5〜60%）、強い縮約 c=2・20、『合計×条件』のモデル、共通成分を引いたプロファイル（最初の版は共通の条件の効果を拾って見かけの有意になっていた）、祖先ごとの検定での共通成分の除去。いずれも成績の良し悪しではなく、設計の欠陥（飽和した祖先、共線性、共通成分）を直すためで、修正の前後を本文に記した。</li>"
                "<li>2026 年は記述のみで、選択に使っていない。市場オッズ・払戻は期待数と回収率の評価にだけ使い、特徴量にしていない。</li></ul>")
    body.append("<h2>8. 結論</h2><ul>"
                "<li>『母系の深い位置にいる高頻度の種牡馬は、血が薄くても競馬場・距離・馬場・ペースの傾向を出す』は、馬内の相対適性で検証して<b>支持されなかった</b>（母系 +1.1bp）。ユーザーの見立て（母系が細かい適性に関与し、父系と差が出る）のうち、差が出ること自体は確認できたが、<b>向きは逆で、効きが（弱いながら）見えたのは父系</b>（+3.8〜+7bp、2025 年でも再現）。ただし二乗誤差の 0.04〜0.14% で、現行モデルの残差には上乗せがない。</li>"
                "<li>祖先のタイプ（K=6）に分けて血統表から確率を出す仕組み（<code>tp73_side.py profile</code> が <code>type_*.parquet</code> を出す）は作れた。ただし固定したプロファイルは年とともに古くなり（父系は 2022 年 t=+3.9 → 2025 年 −0.2）、予測力が安定しないので、特徴量・表示には使わない。使うなら、プロファイルを年次で更新する運用（cron）が要る。</li>"
                "<li>血統クラスタ対の同時好走は、市場を除くと lift が 1.0〜1.13 の小さな偏りで、回収率は払戻率を下回る。買える対は見つからなかった。</li>"
                "<li>『母集団で単純に集計』した成績率は、能力の差を拾って大きく見えるだけで、適性の指標としては向かない（馬内の対比で 1/5 に縮む）。適性を測るときは、馬内の対比を使う方針を採る。</li>"
                "<li>次の案: (1) 父系プロファイルの年次更新（ローリング）での再検証、(2) V5・v6e の残差での再確認、(3) めぐ指数の相対（R_meg）で父系だけ条件別が効く理由の調査、(4) 用量差（n≥2）の過剰有意の原因調査。</li></ul>")
    body.append('<p class="muted">コード: <code>src/scripts/data/tp73_side.py</code>（anc / prep / thin / thin_strong / profile / dose / pair / rlp）、この章の生成: <code>tp73_report.py</code>。ホストの出力: <code>/home/jovyan/work/tp73_out/</code>（prereg.json に事前登録した定義）。</p>')
    html = f'<!doctype html><html lang="ja"><head><meta charset="utf-8"><title>血統の高頻度の種牡馬の特性と血統クラスタ対の同時好走（tp-73）— keiba-vpn</title><style>{CSS}</style></head><body><a href="../../index.html">← ドキュメント一覧</a>{"".join(body)}</body></html>'
    OUT.write_text(html, encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
