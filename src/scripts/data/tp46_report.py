"""tp-46 の章 docs/html/modeling/chapters/mark_rules.html を result.json から作る。

python -m src.scripts.data.tp46_report <result.json>
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CH = ROOT / "docs/html/modeling/chapters"
SRC = CH / "race_type_marks.html"

POL_JP = {
    "P0_current": "P0 現行（◎=勝率最上位、○=連対確率最大、▲=単穴、△=複勝確率順）",
    "P1_switch": "P1 状況で切替（突出/荒れ/拮抗/通常の4型）",
    "P1a_close_only": "P1a 拮抗型だけ切替（上位2頭のうち期待値の高い方を◎）",
    "P1b_value_only": "P1b 荒れ型だけ切替（期待値の高い別の馬を◎）",
    "P1c_close_1.10": "P1c 拮抗型の境目 1.10",
    "P1d_close_1.25": "P1d 拮抗型の境目 1.25",
    "P1e_value_1.30": "P1e 荒れ型の期待値の境目 1.30",
    "P1f_value_1.05": "P1f 荒れ型の期待値の境目 1.05",
    "P2_maybe_composite": "P2 △を合成点（複勝確率^0.55×複勝EV^0.45）の順にする",
}


def head() -> str:
    t = SRC.read_text(encoding="utf-8")
    return t[: t.index("</style></head>") + len("</style></head>")].replace("レースの性質で変える印と買い方（tp-43）", "印の付け方をレースの状況で切り替えるか（tp-46）")


def f3(x) -> str:
    return "—" if x is None else f"{x:.3f}"


def d(v) -> str:
    return f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"


def table(res: dict, key: str, pols: list[str]) -> str:
    rows = ["<tr><th>方策</th><th>◎の勝率</th><th>◎の複勝率</th><th>◎○の連対カバー</th><th>馬連(◎○)的中</th><th>印が3着内を拾う割合</th><th>3着内3頭すべて印</th><th>平均印数</th><th>△の平均頭数</th>"
            "<th>単勝◎</th><th>複勝◎</th><th>馬連◎○</th><th>ワイド◎○</th><th>3連複◎+△ボックス</th><th>◎が変わる割合*</th><th>オッズ無しで◎が同じ割合</th></tr>"]
    for p in pols:
        v = res[key][p]
        rows.append(f"<tr><td>{html.escape(POL_JP[p])}</td>" + "".join(f"<td class='num'>{f3(v[m])}</td>" for m in (
            "honmei_win", "honmei_top3", "top2cov", "umaren_hit", "cover3", "all3", "n_marks", "n_maybe", "roi_tan", "roi_fuku", "roi_umaren", "roi_wide", "roi_trio_box", "honmei_flip", "noodds_honmei_same")) + "</tr>")
    return "<table>" + "".join(rows) + "</table>"


def diff_table(res: dict, pols: list[str]) -> str:
    rows = ["<tr><th>方策 − 現行</th><th>期間</th><th>◎の勝率</th><th>◎○の連対カバー</th><th>印が3着内を拾う割合</th><th>単勝◎の回収率</th><th>馬連◎○の回収率</th><th>3連複ボックスの回収率</th></tr>"]
    for p in pols:
        for key, lab in (("design_2022_23", "2022〜23（設計）"), ("confirm_2024", "2024（確認）"), ("ref_2025", "2025（参考）")):
            v = res[key][p]
            rows.append(f"<tr><td>{html.escape(POL_JP[p])}</td><td>{lab}</td>" + "".join(f"<td class='num'>{d(v['d_' + m])}</td>" for m in (
                "honmei_win", "top2cov", "cover3", "roi_tan", "roi_umaren", "roi_trio_box")) + "</tr>")
    return "<table>" + "".join(rows) + "</table>"


def main(path: str) -> None:
    res = json.loads(Path(path).read_text(encoding="utf-8"))
    sh = res["situation_shares"]
    shares = "<table><tr><th>年</th><th>突出(dom)</th><th>荒れ(value)</th><th>拮抗(close)</th><th>通常(normal)</th></tr>" + "".join(
        f"<tr><td>{y}</td>" + "".join(f"<td class='num'>{v.get(k, 0):.1%}</td>" for k in ("dom", "value", "close", "normal")) + "</tr>" for y, v in sh.items()) + "</table>"
    main_pols = ["P0_current", "P1_switch", "P1a_close_only", "P1b_value_only", "P2_maybe_composite"]
    all_pols = list(POL_JP)
    body = f"""
<body><a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>印の付け方をレースの状況で切り替えるか（tp-46）</h1>
<p class="muted">依頼: 「◎○▲★各1頭＋△複数」の枠を保ったまま、◎○▲★の選び方を、勝率が拮抗しているレースや期待値の高い馬がいるレースなど、レースの状況で切り替えて最適化したい。
評価する印は本番の <code>assign_mece_marks</code> そのもの（<code>situational=True</code> で切替）。データは tp-36 の入力（V4 の予測・確定オッズ・確定払戻）、{res['races']:,} レース（2022〜2025 年）。</p>
<div class="note"><b>要点.</b><ul>
<li><b>状況で◎を切り替えると、印の質が下がった.</b>◎の勝率は 0.317 → 0.256（設計年）、◎の複勝率は 0.621 → 0.526、◎○の連対カバーは −0.03 で、3 年とも区間が 0 を含まない悪化。回収率の差は、単勝◎が設計年だけ +0.062（区間 +0.003〜+0.126）で 2024 年は +0.045（区間が 0 を含む）、2025 年は +0.001、馬連◎○は −0.010・+0.060・−0.089 と符号が年で入れ替わる。</li>
<li><b>拮抗型だけの切替でも、◎の勝率が −0.011〜−0.019 下がり、回収率は改善しなかった.</b>荒れ型（期待値の高い別の馬を◎にする）は、◎の勝率が −0.06 と大きく下がる。</li>
<li><b>印の安定性も悪化.</b>確定前のオッズの揺れ（σ=0.07 の対数正規）で◎が変わるレースは、現行 0% に対し切替は約 5〜6%。オッズ無しでは、切替の◎は現行と 65% しか一致しない（現行は 100%）。</li>
<li><b>△を合成点の順にしても改善しない.</b>印が3着内を拾う割合が −0.02〜−0.03、3頭すべてが印の割合が 0.64 → 0.58。</li>
<li><b>結論: 採用しない.</b>◎=勝率最上位、○=連対確率最大、▲=複勝妙味の単穴、★=10〜30倍の期待値馬、△=複勝確率の高い順（上限つき）のシンプルな基準を維持する。切替の実装（<code>situational</code>）は既定オフで残し、実オッズ（T-60/T-15）が貯まった後の再評価に使う。2026 年の1回きりの評価は、採用しないので使っていない。</li>
</ul></div>
<h2>1. 状況の判定と切替の内容</h2>
<ul>
<li><b>突出型 (dom)</b>: 勝率最大が均等確率の 2 倍以上、かつ 1 位/2 位が 1.6 倍以上。◎=勝率最上位、○=連対確率最大（現行どおり）。</li>
<li><b>荒れ型 (value)</b>: 複勝確率 25% 以上の馬のうち期待値（勝率×単勝オッズ）が最大の馬が、勝率最上位とは別の馬で、期待値 1.15 以上。◎=その馬、○=勝率最上位。</li>
<li><b>拮抗型 (close)</b>: 勝率 1 位/2 位が 1.15 倍未満。◎=上位 2 頭のうち期待値の高い方、○=もう一方。</li>
<li><b>通常型 (normal)</b>: 上のどれでもない。現行どおり。</li>
<li>期待値は確定オッズから計算した値（本番では確定前は予測オッズ。揺れは安定性で別に見る）。境目は一般的な値で置き、拮抗型と荒れ型だけ数通りを試した（試した方策は {len(res['policies'])} 通り）。</li>
</ul>
{shares}
<p class="muted">荒れ型が約 1/3 のレースに出る。期待値の高い別の馬は珍しくなく、「荒れ型」で◎を切り替えると、多くのレースで◎が勝率最上位でなくなる。</p>
<h2>2. 方策ごとの印の質と回収率（2024 年、確認年）</h2>
{table(res, 'confirm_2024', main_pols)}
<p class="muted">* 確定オッズに σ=0.07 の対数正規ノイズを2回かけ（T-60→T-15 のオッズ変動の模擬）、◎が変わったレースの割合。回収率は 1 レース 1 を賭けた戻り（買い目で均等割り、母数はレース単位、払戻は確定払戻）。</p>
<h2>3. 現行との差（レース単位の対ブートストラップ、95% 区間）</h2>
{diff_table(res, ['P1_switch', 'P1a_close_only', 'P1b_value_only', 'P2_maybe_composite'])}
<h2>4. 境目を変えた方策（設計年 2022〜23）</h2>
{table(res, 'design_2022_23', all_pols)}
<h2>5. 判断と今後</h2>
<ul>
<li>切替は、◎が当たる確率と連対のカバーという印の質を確実に下げ、回収率の改善は年で符号が入れ替わった（tp-43 の荒れ度での切替と同じ結論）。期待値の情報は、◎の選び方ではなく、★（10〜30 倍の期待値馬）と買い目の選び方（推奨ポートフォリオ）で使うのが妥当。</li>
<li>△は 1 レース最大 4 頭（頭数が少ないレースは減らす）で、典型的な印の合計は 6〜7 個（旧ルールは平均 11.7 個）。印が3着内の馬を拾う割合は 0.86（3 頭中約 2.6 頭）で、3 頭すべてに印が付く割合は 0.64。</li>
<li>再評価の条件: T-60/T-15 の実オッズが貯まった後に、確定前のオッズでの印の安定性と質を測り直す。</li>
</ul>
<p class="muted">再現: <code>python -m src.scripts.data.tp46_marks run</code>（環境変数 TP46_DATA に tp36/frame_all.parquet と payoff.parquet の置き場を指定）→ <code>python -m src.scripts.data.tp46_report &lt;result.json&gt;</code>。</p>
</body></html>
"""
    (CH / "mark_rules.html").write_text(head() + body, encoding="utf-8")
    print("wrote", CH / "mark_rules.html")


if __name__ == "__main__":
    main(sys.argv[1])
