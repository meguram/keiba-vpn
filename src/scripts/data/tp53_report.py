"""tp-53 の章 docs/html/modeling/chapters/pace_consistency.html を analysis.json から作る。

python -m src.scripts.data.tp53_report [--json data/local/modeling/tp53/analysis.json]
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CH = ROOT / "docs/html/modeling/chapters"
OUT = CH / "pace_consistency.html"
DEFAULT_JSON = ROOT / "data/local/modeling/tp53/analysis.json"


def head_style() -> str:
    t = (CH / "race_type_marks.html").read_text(encoding="utf-8")
    m = re.search(r"<style>.*?</style>", t, re.S)
    return m.group(0) if m else "<style></style>"


def f3(x) -> str:
    return "—" if x is None or x != x else f"{x:+.3f}"


def f2(x) -> str:
    return "—" if x is None or x != x else f"{x:.3f}"


def pc(x) -> str:
    return "—" if x is None or x != x else f"{100 * x:.1f}%"


def table(head: list[str], rows: list[list[str]]) -> str:
    return "<table><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in head) + "</tr>" + "".join(
        "<tr>" + "".join(f"<td class='num'>{c}</td>" if i else f"<td>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows) + "</table>"


NEW = "位置×ペース＋自己の過去上がり（傾きの上限つき・採用）"


def build(a: dict) -> str:
    A, BC, D = a["A"], a["BC"], a["D"]
    wf = D["walk_forward"]
    labs = ["ハイ", "ミドル", "スロー", "不明"]
    cross = A["pf_x_td"]
    tds = ["ハイ", "ミドル", "スロー"]
    t1 = table(["位置取り側のペース ＼ 追走側のペース"] + tds, [[lb] + [f"{cross.get(lb, {}).get(td, 0):,}" for td in tds] for lb in labs if lb in cross])
    acc = BC["accuracy"]
    link = BC["linkage"]
    t2 = table(["組合せ（レース内で平均を引いた相関）", "相関", "頭数"],
               [[k, f3(v[0]), f"{v[1]:,}"] for k, v in acc.items() if k.startswith(("td_early", "pf_early", "td_l3f", "td_stamina", "td_ease"))])
    t3 = table(["予測値どうしの連動 / 実測", "相関"], [[k, f3(v[0])] for k, v in link.items()])
    sl = BC["slopes"]
    w25 = wf["2025"]["slopes_by_actual_pace"]
    t4 = table(["ペース（実測）", "実測 2020〜25（位置が後ろ1.0あたりの上がり偏差・秒）", "追走側の想定上がり 2020〜25", "実測 2025", "追走側 2025", "整合層（新）2025"],
               [[p, f3(sl.get(f"実測(ペース={p})")), f3(sl.get(f"TD予測(実測ペース={p})")), f3(w25.get(p, {}).get("実測")), f3(w25.get(p, {}).get("TD")), f3(w25.get(p, {}).get("新"))] for p in ("ハイ", "ミドル", "スロー")]
               + [["全体", f3(sl.get("実測 全体")), f3(sl.get("TD予測 全体")), f3(wf["2025"]["slope_all"]["実測"]), f3(wf["2025"]["slope_all"]["TD"]), f3(wf["2025"]["slope_all"]["新"])]])
    corr = A["corr_actual_z"]
    t5 = table(["ペースの指標", "実測の前半3F（同条件で標準化・小さいほど速い）との相関"], [[k, f3(v)] for k, v in corr.items()])
    t6 = table(["年（学習は前年まで）", "ペース 3 分類の一致率: 新", "位置取り側", "追走側", "全部ミドルと答える", "ペース相関: 新", "上がり偏差の相関: 新 / 既存", "MAE(秒): 新 / 既存 / 全員0"],
               [[y, pc(w["pace_acc_new"]), pc(w["pace_acc_PF"]), pc(w["pace_acc_TD"]), pc(w["pace_acc_always_mid"]), f3(w["pace_corr_new"]),
                 f"{f3(w['ablation'][NEW]['corr'])} / {f3(w['ablation']['TD既存（そのまま）']['corr'])}",
                 f"{w['ablation'][NEW]['mae']:.3f} / {w['ablation']['TD既存（そのまま）']['mae']:.3f} / {w['ablation']['全員 0（偏差なし）']['mae']:.3f}"] for y, w in wf.items()])
    abl_rows = []
    for k in (NEW, "位置×ペース＋自己の過去上がり", "位置×ペースのみ", "自己の過去上がりのみ", "TD既存（そのまま）", "TD既存（最良の線形再較正）", "全員 0（偏差なし）"):
        abl_rows.append([k] + [f"{f3(wf[y]['ablation'][k]['corr'])} / {wf[y]['ablation'][k]['mae']:.3f}" for y in wf])
    t7 = table(["上がり偏差の予測（相関 / MAE 秒）"] + list(wf), abl_rows)
    bl = D["baseline"]
    t8 = table(["年", "追走側の基準上がり: MAE(秒) / 偏り / 相関", "同条件の中央値（前年まで）: MAE / 偏り / 相関"],
               [[y, f"{b['MAE_td']:.2f} / {b['bias_td']:+.2f} / {b['corr_td']:.2f}", f"{b['MAE_cohort']:.2f} / {b['bias_cohort']:+.2f} / {b['corr_cohort']:.2f}"] for y, b in bl.items()])
    fc = D["final_coef"]
    t9 = table(["項目", "係数"], [[f"ペース z: {k}", f"{v:+.3f}"] for k, v in fc["pace"].items()] + [[f"上がり偏差: {k}", f"{v:+.3f}"] for k, v in fc["l3f"].items()])
    fb = A["pf_front_vs_actual_z"]
    t10 = table(["位置取り側の『逃げ』の頭数", "レース数", "実測の前半3F z の平均（大きいほど遅い）", "『スロー』判定の割合"],
                [[("3頭以上" if k == "3" else f"{k}頭"), f"{v['races']:,}", f3(v["mean_act_z"]), pc(v["slow_share"])] for k, v in fb.items() if k != "-1"])
    sty = BC["td_style_share"]
    pstyle = BC["pf_style_share"]
    body = f"""
<h1>想定位置取り・追走難度・想定上がりの連動（tp-53）</h1>
<p class="muted">ユーザー指摘（2026-10-20）: 想定位置取りと、追走難度、想定上がりがうまく連動していない気がする。結果から読み取れる関係と、予測値との関係の差分を確認する。
対象: 2020〜2025 年の 20,270 レース・28.3 万頭（追走難度側は事前計算の保存物、位置取り側は馬柱から同じ関数で再計算、実測は確定結果）。</p>

<h2>1. 要点</h2>
<ul>
<li><b>連動していません。原因は設計です。</b>この 3 つは別系統で作られ、共通のペース想定がありません。位置取り（隊列）は馬柱の通過順位だけの集計、追走難度・想定上がりは別のルールで、画面の「展開」の見出しは位置取り側のペース、上がりの基準は追走側のペースに由来します。</li>
<li><b>追走側のペースは実測と無関係です。</b>91% のレースが「ミドル」、実測の前半 3F との相関は −0.02（位置取り側の先行勢の圧力は −0.33）。追走側の脚質の分布は 92% が「差し」で、逃げ・先行が 0 頭のレースが 82%（位置取り側は逃げ・先行が平均 4.6 頭）。</li>
<li><b>想定上がりは位置取りにほとんど反応しません。</b>実測では、後ろの馬ほど上がりが速く（全体で最後方と先頭の差 約 0.9 秒、ペースが速いほど大きい: ハイ −1.4 / ミドル −0.8 / スロー −0.3）、追走側の想定上がりでは同じ傾きが −0.15（全期間）で、2024〜25 年はほぼ 0（2023 年は −0.25）、追走側が「ハイ」と判定したレースでは符号が逆です。</li>
<li><b>整合層（既定オフ）で連動を回復できます。</b>位置取り側の先行勢からペースを推定し、位置取り×ペースで上がり偏差を決め、同条件の中央値と自己の過去の上がり偏差を足す。2023〜2025 年の walk-forward で、上がり偏差の相関は 0.07〜0.16 → 0.35〜0.36、MAE は 0.76〜0.93 → 0.69〜0.70、ペース 3 分類の一致は 37〜40%（全部ミドルと答えるだけでも 39〜40%）→ 46〜47%。</li>
<li>V4/V5 の勝率予測への影響はありません（これらの値は V4/V5 の入力ではなく、画面表示用です）。</li>
</ul>

<h2>2. 2 つのペース定義は一致せず、追走側は実測と無関係</h2>
<p>位置取り側（逃げ・先行の頭数から ハイ/ミドル/スロー）と追走側（pace_prediction.pace_type）の一致率は <b>{pc(A['pf_td_agree'])}</b>（追走側の {pc(A['td_mid_share'])} が「ミドル」）。実測のペース（前半 3F の同条件 z で 3 分類）との一致率は、位置取り側 {pc(A['PF_acc'])}、追走側 {pc(A['TD_acc'])} で、どちらも低い（全部『ミドル』と答えるだけで 39〜40% 取れるので、位置取り側 38〜39%・追走側 37〜40% は、その水準を超えていない）。ただし相関で見ると、位置取り側は実測と意味のある関係があり、追走側はほぼ無関係です。</p>
{t1}
{t5}
<p>位置取り側の「逃げが 0 頭のレースは『スロー』」は実測の傾向と合っています（実測 z の平均 +0.50、前半が遅い）。ご指摘の例（202604040101 新潟 1R: 位置取り側スロー・逃げ 0・先行 6、追走側ミドル・先行 3）は、逃げ 0 頭でも先行 3〜6 頭いることが多く、追走側の脚質判定が先行を数えていないために食い違います。</p>
{t10}

<h2>3. 結果から読み取れる関係と、予測値の関係</h2>
<p>実測（レース内で平均を引いた上がり偏差 ~ 初角位置）は、全体で相関 {f3(link['act_l3f~act_early'][0])}、位置が後ろ1.0あたり約 {f3(sl['実測 全体'])} 秒。追走側の想定上がりの傾きは {f3(sl['TD予測 全体'])}（実測の 5 分の 1 以下）。</p>
{t4}
{t3}
{t2}
<p>追走側の「楽さ」（ease_pct）と「温存度」（stamina_index）は、位置取りの言い換えに近い（温存度は追走側の位置と相関 0.98）ため、上がりの予測に新しい情報を足していません。個別の位置取りの当たりも、追走側 {f3(acc['td_early~act_early'][0])}、位置取り側 {f3(acc['pf_early~act_early'][0])} と、位置取り側のほうが大幅に良い。</p>

<h2>4. 原因の切り分け</h2>
<ol>
<li><b>(a) 3 つが独立に作られている</b>: 位置取り側（馬柱だけのルール）と追走側（別のルールとスコア）が別々の脚質判定を持つ。追走側の脚質は 差し {pc(sty.get('差し'))}・先行 {pc(sty.get('先行'))}・追込 {pc(sty.get('追込'))}・逃げ {pc(sty.get('逃げ'))}、位置取り側は 差し(中団) {pc(pstyle.get('mid'))}・先行 {pc(pstyle.get('stalker'))}・追込 {pc(pstyle.get('closer'))}・逃げ {pc(pstyle.get('front'))}。</li>
<li><b>(b) ペースの共有が無い</b>: 位置取り側の先行勢の数が追走側のペース・想定上がりに伝わらない。追走側のペース指数は実測と無相関。</li>
<li><b>(c) 想定上がりの水準が不安定</b>: 追走側の基準上がり（last_3f_baseline_sec）は、実測のレース中央値に対し MAE 1.7〜2.3 秒、ダートで偏り −2.2 秒（芝 −0.1 秒）。同条件の中央値（前年まで）なら MAE 0.74〜0.77 秒・相関 0.87〜0.88。</li>
<li><b>(d) 馬の上がり能力の測り方</b>: 近走の上がり『秒』の加重平均で、条件（芝/ダ・距離・馬場）の補正がなく、サンプルが無い馬（41%）は基準のまま。自己の過去の上がり偏差（同条件の中央値との差）は、単独で実測との相関 0.34〜0.36 と最も効く。</li>
<li>(e) 特徴量の先読み・時点ずれ: 追走側の保存物は pre_race_only=True で対象日より前の走だけを使っており、先読みは見つからなかった（lgbm は heuristic）。</li>
</ol>
{t8}

<h2>5. 整合層（pace_coherence）の検証</h2>
<p>位置取り側の位置取り（枠補正後）と先行勢の数だけを入力にして、(1) 前半 3F の z を推定、(2) 位置取り×ペース z（×距離・芝）で上がり偏差を決め、(3) 自己の過去の上がり偏差（同条件の中央値との差の直近 5 走平均）を足す。同条件の表と係数は学習年より前だけで作り（walk-forward）、最終版は 2020〜2025 年で作る。</p>
{t6}
{t7}
<p><b>読み方</b>: 上がり偏差の改善の大半は『自己の過去の上がり偏差』（単独で相関 0.34〜0.35）から来ており、『位置×ペース』の結合だけでは相関 0.15〜0.16 だが、傾きが実測に近づく（追走側はほぼ 0 → 新は −0.55〜−0.88 と実測 −0.36〜−1.47 の順序（スロー < ミドル < ハイ）と一致）。つまり<b>連動の回復は位置×ペースの項、精度の向上は自己の過去の上がりの項</b>が担う。</p>
{t9}

<h2>6. 実装と導入（画面表示用・既定オフ）</h2>
<ul>
<li><code>src/pipeline/inference/pace_coherence.py</code>（<code>config/pace_coherence.json</code> に係数と同条件の表）。環境変数 <code>KEIBA_PACE_COHERENCE=1</code> のときだけ、予測 API の応答に <code>pace_unified</code>（z・ハイ/ミドル/スロー・想定の前半 3F・基準の上がり）と、各馬の <code>l3f_coherent</code>（上がり偏差・想定の上がり秒・順位）を足す。無効・設定なし・材料不足のときは何もしない。</li>
<li>画面は <code>pace_unified</code> を見出しのペースに使い、<code>l3f_coherent</code> を想定上がりに使えば、位置取り・ペース・上がりが同じ流れになる（UI の編集は keiba-vpn-72 の担当）。追走側の既存の値は残る（併記できる）。</li>
<li>V4/V5 の入力ではない（<code>tracking_difficulty</code>・<code>position_forecast</code> は特徴量にもサービングにも使われていない）ため、再学習は不要。</li>
</ul>
<h2>7. 限界</h2>
<ul>
<li>『整合層』は実測の関係を作るだけで、予測の当たりを保証しない（上がり偏差の相関は 0.35 前後）。追走難度（ease_pct）自体の再設計（stamina と位置の重複の解消）は未実施。</li>
<li>位置取りは枠順補正後の過去走ベースの集計で、実際の隊列とは別。初出走・履歴の浅い馬は位置取りも上がりも出ない。</li>
<li>ペースの 3 分類の閾値（±0.5 SD）は 3 分位に近いが、実測のペースは連続量。</li>
</ul>
"""
    return f"<!doctype html><html lang='ja'><head><meta charset='utf-8'><title>想定位置取り・追走難度・想定上がりの連動（tp-53）</title>{head_style()}</head><body>{body}</body></html>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(DEFAULT_JSON))
    a = ap.parse_args()
    d = json.loads(Path(a.json).read_text(encoding="utf-8"))
    OUT.write_text(build(d), encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    main()
