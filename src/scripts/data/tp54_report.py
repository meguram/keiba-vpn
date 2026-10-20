"""tp-54 の結果（results.json / pace_eval.json / v5_select_tp54/results.json）を章 tracking_difficulty_v2.html にする。

python -m src.scripts.data.tp54_report --results <results.json> --pace <pace_eval.json> [--v5 <v5_select_tp54/results.json>]
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CHAPTER = ROOT / "docs/html/modeling/chapters/tracking_difficulty_v2.html"


def _f(x, nd=3) -> str:
    return "—" if x is None or x != x else f"{x:+.{nd}f}"


def _table(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{th}</tr>{body}</table>"


INTRO = """<p class="muted">ユーザー指示（2026-10-20）: 追走難度を『想定位置取り（position_forecast の想定、枠補正込み）を取るために必要な通過ラップが、その馬にとって楽か大変か』として作り直す。前走比（前走と同じ位置を取るなら今回は前走より楽か）は副次ビュー。</p>
<h2>1. 要点</h2>
<ul>
<li><b>現行の追走難度（ease_pct）は、実測の『終いの落ち』をほとんど予測しません。</b>ease_pct は、馬の『いつもの位置との差（position_deviation）』を LightGBM で予測し、50＋差×200 に変換した値で、位置取りの言い換えです。実測の終いの落ち（Y2）との相関は −0.03〜−0.06（符号は『楽なほど落ちにくい』と逆）、目盛りも年によって大きく動きます（2020〜2025 年の全体では 60 以上 4%・30 未満 51%、評価年別は下の表）。現行の『消耗（stamina_index）』は手書きの式（重みは固定）です。</li>
<li><b>新しい定義</b>: 要求（想定位置取り p̂ × 今回の想定ペース z_hat × コース・距離・馬場・頭数・脚質構成）と、供給（その馬の過去 5 走から作る、位置取りの効果を除いた上がりの力、前で運んだ・後ろで運んだときの上がり、いつもの位置）を入れた LightGBM が、実測の『終いの落ち』Y2 を予測する。Y2 は『実際の上がり偏差（レース内の中央値との差、±4 秒）− その馬の過去の力によるベースライン』で、過去の力が特徴量でもあり目的変数の一部でもある見かけの相関（平均への回帰）を除いている。予測が大きい＝終いが落ちる＝大変。ease_pct_v2 は −予測 を学習期間の分位に直した 0〜100（50 が標準）。</li>
<li><b>結果</b>: 追走に関わる量だけの core モデルで Y2 との相関 0.22〜0.23（2023〜2025 年、walk-forward）。現行 ease は −0.03〜−0.06。予測の十分位ごとの実測 Y2 は −0.30〜−0.33 秒（楽）から +0.44〜+0.49 秒（大変）まで単調。過去の力だけのベースラインで上がり偏差の分散の 8% が説明できていたところに、core が 13% まで、一般の調子の変化（年齢・馬体重・休養・距離変化・前走の上がり）まで入れた full が 17.5% まで説明を足す。</li>
<li><b>位置取り×ペースの整合</b>: 『前×速いペース』は予測 +0.14〜+0.17 秒に対し実測 +0.14〜+0.16 秒、『後×速い』は −0.04〜−0.08 に対し −0.05〜+0.00 秒など、位置・ペースごとの平均がよく合う（現行はこの関係を持たない）。</li>
<li><b>前走比（副次ビュー）</b>: 前走と同じ位置を取った馬（約 1.2 万対/年）で、『前走の条件より今回は楽』の予測が、前走→今走の終いの余裕の改善を相関 0.20〜0.22 で予測する。</li>
</ul>
<h2>2. データ上の制約（通過ラップについて）</h2>
<p>各馬の通過順位（コーナー別）・走破タイム・上がり3F・レースの前半3F／後半3F・ハロンラップは取れる（通過順位 99%、上がり 96%、レースの前半3F 94%）。<b>馬ごとの位置別の通過タイム（コーナー通過時刻）は取れない</b>ので、『その位置を保つのに要る速度』は、位置取りとペースの想定（要求）と、実測の終い（結果）の関係から学習した。</p>"""


OPS = """<h2>運用（フラグ・出力・限界）</h2>
<ul>
<li><b>出力</b>: 環境変数 <code>KEIBA_TRACKING_V2=1</code>（既定は無効）のとき、予測 API（<code>/api/race/{id}/predictions</code>）の各馬に <code>tracking_v2</code>（<code>ease_pct</code>＝0〜100・50 が標準・60 以上が楽・30 未満が大変、<code>label</code>、<code>drain_sec</code>＝過去の力から見た終いの落ち（秒）、<code>demand_sec</code>＝位置取り×ペース×コースの要求、<code>capacity_sec</code>＝その馬の供給による差）が付く。既存の追走難度（ease_pct・消耗）は変えない。説明文の案: 『想定位置取りを取るために必要な通過ラップが、その馬にとって楽か大変か（過去の同じような走から見た終いの余裕）』。</li>
<li><b>ペース想定 v2</b>: <code>KEIBA_PACE_COHERENCE_V2=1</code> で、整合層（pace_coherence）のペース z を新しい係数（<code>config/pace_coherence_v2.json</code>）に差し替える。現行の挙動は変えない。</li>
<li><b>推論の材料</b>: 要求は position_forecast の想定位置取り（枠補正込み）と pace_forecast の先行勢の数・圧力から。供給は馬柱（race_shutuba_past の raw）の直近 5 走（対象日より前）から、学習表と同じ定義で組み立てる。2025 年の実在の馬 4,000 頭で、学習表の集計と馬柱から作った集計を比べると、相関 0.92〜0.97（完全一致は 66〜96%）。差は、馬柱に出ない走（地方・障害など）と日付のずれ。</li>
<li><b>前走比（副次ビュー）</b>: 学習側の評価では出せるが、推論側（馬柱）では前走のペース想定（先行勢の数）が分からないので、API には出していない。</li>
<li><b>V4/V5 への影響</b>: なし（追走難度は V4/V5 の入力ではない）。特徴量としての増分は下の一次選別のとおり、v5 に足す根拠は無かった。</li>
<li><b>限界</b>: ① 馬ごとの位置別の通過タイム（コーナー通過時刻）は取れず、『要求』は想定位置×想定ペースの関数としてモデルが学習したもの。② 過去の走が少ない馬（0〜1 走）は供給の列が欠け、ease が標準寄りになる保証は無い（欠けは木が扱うが、分布が学習時と違いうる）。本番で使う前に、実レースで分布（ease の分位の割合）を確認する。③ 2026 年の評価は、モデルが市場に弱い年で、他の評価とそろえるため行っていない（学習は 2020〜2025 年）。</li>
</ul>"""


def build(results: dict, pace: dict, v5: dict | None, intro: str = "", stack: dict | None = None) -> str:
    parts = ["<h1>追走難度 v2（tp-54）</h1>", intro or INTRO]
    by = results.get("by_year", {})
    rows = []
    for y, r in by.items():
        rows.append([y, str(r["n_test"]), _f(r["v2_full"]["corr"]), _f(r["v2_demand_only"]["corr"]), _f(r["baseline_current_ease"]["corr_neg_ease"]),
                     _f(r["baseline_current_stamina"]["corr_stamina"]), _f(r["baseline_position"]["corr_pf_early"]), f"{r['current_ease_share_ge60']*100:.1f}% / {r['current_ease_share_lt30']*100:.1f}%"])
    parts.append("<h2>実測の『終いの落ち』Y を予測できるか（walk-forward）</h2>")
    parts.append(_table(["評価年", "件数", "v2（要求＋供給）の相関", "v2（要求のみ）", "現行 ease の相関（符号を反転）", "現行 消耗 の相関", "想定位置取り単独", "現行 ease ≥60 / <30 の割合"], rows))
    for y, r in by.items():
        parts.append(f"<h3>{y} 年: 予測の十分位ごとの実測 Y（秒）</h3>")
        d1 = {d['q']: d['mean_Y'] for d in r["deciles_by_pred"]}
        d2 = {d['q']: d['mean_Y'] for d in r["deciles_by_current_ease"]}
        parts.append(_table(["十分位（0=楽 … 9=大変）", "v2 の実測 Y", "現行 ease の実測 Y"], [[str(q), _f(d1.get(q)), _f(d2.get(q))] for q in range(10)]))
    parts.append("<h2>位置取り×ペースでの整合（予測と実測の平均 Y）</h2>")
    for y, r in by.items():
        g = r["position_pace_grid"]
        parts.append(f"<h3>{y} 年</h3>" + _table(["位置取り", "ペース", "予測 Y", "実測 Y", "件数"], [[x["pos"], x["pace"], _f(x["pred"]), _f(x["actual"]), str(x["n"])] for x in g]))
    parts.append("<h2>前走比（前走と同じ位置を取るなら、今回の条件は前走より楽か）</h2>")
    parts.append(_table(["評価年", "対の数", "同じ位置の対", "相関（全）", "相関（同位置）", "順位相関（同位置）"],
                        [[y, str(r["vs_prev"]["n"]), str(r["vs_prev"]["n_same_pos"]), _f(r["vs_prev"]["corr_all"]), _f(r["vs_prev"]["corr_same_pos"]), _f(r["vs_prev"]["spearman_same_pos"])] for y, r in by.items()]))
    pp = pace.get("by_year", {})
    parts.append("<h2>ペース想定の較正し直し</h2>")
    parts.append(_table(["評価年", "現行 相関", "新 相関", "GBM 相関", "現行 3分類一致", "新 3分類一致"],
                        [[y, _f(r["current"]["corr"]), _f(r["ridge_v2"]["corr"]), _f(r.get("gbm", {}).get("corr")), _f(r["current"]["acc3"]), _f(r["ridge_v2"]["acc3"])] for y, r in pp.items()]))
    dis = pace.get("disagreement", {})
    f0 = pace.get("front0", {})
    parts.append(f"<p>位置取り側ラベルと整合層（現行）の向きが真逆のレース: {dis.get('n_opposite_pf_vs_current')} / {dis.get('n_known')} ({dis.get('opposite_rate', 0)*100:.1f}%)。"
                 f"真逆のとき、実測の3分類に当たる割合は 位置取り側 {dis.get('when_opposite_acc', {}).get('pf_level')}、整合層（現行）{dis.get('when_opposite_acc', {}).get('current')}、新 {dis.get('when_opposite_acc', {}).get('ridge_v2')}。"
                 f"逃げ0頭のレース {f0.get('n')} 件の実測の z 平均 {f0.get('mean_z')}（現行の予測 {f0.get('mean_pred_current')}、新 {f0.get('mean_pred_v2')}）。</p>")
    if v5:
        parts.append("<h2>勝率予測への増分（v5 の列セットに足す）</h2>")
        rows = []
        for k, v in (v5.get("pairs") or {}).items():
            o = v["overall"]
            rows.append([html.escape(k), _f(o["Δwin"], 5), _f(o["win_hi"], 5), _f(o["Δtop2"], 5), _f(o["Δtop3"], 5), "採用" if o["採用"] else "不採用"])
        parts.append(_table(["比較", "Δ勝率", "上限", "Δ連対", "Δ複勝", "判定"], rows))
    parts.append(OPS)
    if stack:
        parts.append("<h2>勝率予測への増分の一次選別（v5 構成 d の log p_win に足す条件付きロジット）</h2>")
        parts.append("<p class='muted'>v5 の構成 d（V4＋追い切り再取得版＋CY＋ALL_BEST）の OOF の勝率に、標準化した EASE2 の列を γ 倍して足した softmax。負＝改善。EASE2 は年別 leave-one-year-out（対象年の結果は学習に不使用）。全列を LightGBM に入れる本評価は、一次選別で改善が見えなかったため回していない。</p>")
        for k, rows in stack.get("results", {}).items():
            parts.append(f"<h3>{html.escape(k)}</h3>" + _table(["特徴量", "γ", "Δ対数尤度/レース", "95%下限", "95%上限", "t", "レース数"],
                                                      [[html.escape(r["features"]), html.escape(str(r["gamma"])), _f(r["d_nll"], 5), _f(r["lo95"], 5), _f(r["hi95"], 5), str(r["t"]), str(r["n_races"])] for r in rows]))
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--pace", required=True)
    ap.add_argument("--v5")
    ap.add_argument("--stack")
    ap.add_argument("--intro", default="")
    a = ap.parse_args()
    res = json.loads(Path(a.results).read_text(encoding="utf-8"))
    pace = json.loads(Path(a.pace).read_text(encoding="utf-8"))
    v5 = json.loads(Path(a.v5).read_text(encoding="utf-8")) if a.v5 and Path(a.v5).exists() else None
    stack = json.loads(Path(a.stack).read_text(encoding="utf-8")) if a.stack and Path(a.stack).exists() else None
    body = build(res, pace, v5, a.intro, stack)
    import re

    css = re.search(r"<style>.*?</style>", (CHAPTER.parent / "pace_consistency.html").read_text(encoding="utf-8"), re.S)
    CHAPTER.write_text(f"<!doctype html><html lang='ja'><head><meta charset='utf-8'><title>追走難度 v2（tp-54）</title>{css.group(0) if css else ''}</head><body>\n{body}\n</body></html>", encoding="utf-8")
    print("wrote", CHAPTER)


if __name__ == "__main__":
    main()
