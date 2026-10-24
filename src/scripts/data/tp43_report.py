"""tp-43 の結果を docs の 1 章（race_type_marks.html）にする。

python -m src.scripts.data.tp43_report --dir <tp43 の出力ディレクトリ（analysis.json / final2026.json があるところ）>
"""

from __future__ import annotations

import argparse
import html
import json
import statistics as st
from pathlib import Path

from src.scripts.data import tp29_report as r29
from src.scripts.data import tp43_marks as tm

ROOT = Path(__file__).resolve().parents[3]
CHAPTER = ROOT / "docs/html/modeling/chapters/race_type_marks.html"
KEY_SCHEMES = ["tan1", "fuku1", "umaren_n3", "wide_n3", "wide_b4", "umatan_n3", "sanpuku_a4", "sanpuku_b4"]
SHOW_MARKS = ["M1_pwin", "M2_ptop3", "M3m_evwin_model", "M3b_evwin_blend", "M4_evplace", "M5b_t25", "M5b_t35", "M5b_t45", "M5m_t35", "M7_ev_x_top3"]
SHORT = {"M1_pwin": "M1 勝率", "M2_ptop3": "M2 複勝確率", "M3m_evwin_model": "M3m 単勝EV(モデル)", "M3b_evwin_blend": "M3b 単勝EV(混合)", "M4_evplace": "M4 複勝EV",
         "M5b_t25": "M5b 25%", "M5b_t35": "M5b 35%", "M5b_t45": "M5b 45%", "M5m_t35": "M5m 35%", "M7_ev_x_top3": "M7 EV×複勝確率"}
BUCKET = {"0": "堅い", "1": "中", "2": "荒れ"}


def _f(x, nd=3, sign=False):
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _pct(x):
    return "-" if x is None or x != x else f"{100 * x:.1f}%"


def _ci(t):
    return f"{_pct(t[0])} [{_pct(t[1])}, {_pct(t[2])}]"


def baseline_tbl(a: dict) -> str:
    by = {(b["mark"], b["scheme"]): b for b in a["baseline"]}
    rows = []
    for s in KEY_SCHEMES:
        cells = [tm.SCHEME_JP[s]]
        for m in SHOW_MARKS:
            b = by[(m, s)]
            d, t = b["design"]["cap"], b["test"]["cap"]
            cells.append(f"{_f(d, 2)} / <b>{_f(t, 2)}</b>" if t == t and t >= 1.0 else f"{_f(d, 2)} / {_f(t, 2)}")
        rows.append(cells)
    return r29._tbl(["買い方"] + [SHORT[m] for m in SHOW_MARKS], rows, ["s"] + ["n"] * len(SHOW_MARKS))


def excess_tbl(a: dict) -> str:
    rows = []
    for d in tm.DEFS:
        for alt in ("M5b_t35", "M3b_evwin_blend", "M5m_t35", "M7_ev_x_top3"):
            e = a["excess_vs_m1"][d][alt]
            cells = [tm.DEF_JP[d], SHORT[alt]]
            for v in ("0", "1", "2"):
                cells.append(" / ".join(_f(e[v][k], 3, True) for k in ("design", "confirm", "ref")))
            rows.append(cells)
    return r29._tbl(["荒れ度の定義", "印", "堅い（設計/確認/参考）", "中", "荒れ"], rows, ["s", "s", "n", "n", "n"])


def interaction_block(a: dict) -> tuple[str, str]:
    it = a["interaction"]
    sig = [r for r in it if r["lo"] > 0]
    signeg = [r for r in it if r["hi"] < 0]
    by_def = {}
    for r in it:
        by_def.setdefault(r["def"], []).append(r["inter"])
    t1 = r29._tbl(["荒れ度の定義", "検定数", "交互作用の平均（荒れ − 堅い）", "区間が 0 を超えて正", "区間が 0 を下回って負"],
                  [[tm.DEF_JP[d], len(v), _f(st.mean(v), 3, True), sum(1 for r in it if r["def"] == d and r["lo"] > 0), sum(1 for r in it if r["def"] == d and r["hi"] < 0)] for d, v in by_def.items()],
                  ["s", "n", "n", "n", "n"])
    top = sorted(it, key=lambda r: -r["inter"])[:6]
    t2 = r29._tbl(["定義", "印", "買い方", "荒れでの差(対 M1)", "堅いでの差", "交互作用 [95%区間]", "レース数（荒/堅）"],
                  [[tm.DEF_JP[r["def"]], SHORT[r["alt"]], tm.SCHEME_JP[r["scheme"]], _f(r["rough"], 3, True), _f(r["solid"], 3, True),
                    f"{_f(r['inter'], 3, True)} [{_f(r['lo'], 3, True)}, {_f(r['hi'], 3, True)}]", f"{r['n_rough']}/{r['n_solid']}"] for r in top],
                  ["s", "s", "s", "n", "n", "n", "n"])
    return f"全 {len(it)} 検定のうち、区間が 0 を超えて正は {len(sig)}、0 を下回って負は {len(signeg)}（補正なし。偶然でも約 5% ずつ出る）", t1 + "<h3>交互作用が最も大きかった上位</h3>" + t2


def switch_block(a: dict) -> str:
    sv = a["switch_value"]
    rows = []
    jp = {"m6b_vs_global": "M6b（荒れ度の区分ごとに設計年で選んだ印）− 荒れ度を使わない単一の最良印", "m6a_vs_m5": "M6a（荒れ→M5b 35%、他→M1）− M5b 35% を常に使う",
          "m6a_vs_m1": "M6a − M1（勝率順を常に使う）"}
    for k, label in jp.items():
        m = [r[k]["test"][0] for r in sv]
        rows.append([label, _f(st.mean(m), 4, True), sum(1 for r in sv if r[k]["test"][1] > 0), sum(1 for r in sv if r[k]["test"][2] < 0), len(sv),
                     sum(1 for r in sv if r[k]["confirm"] > 0), sum(1 for r in sv if r[k]["ref"] > 0)])
    t1 = r29._tbl(["比較", "検証年（2024〜25）の回収率差の平均", "区間が正", "区間が負", "組数", "2024 年で正", "2025 年で正"], rows, ["s", "n", "n", "n", "n", "n", "n"])
    rows2 = []
    for d in tm.DEFS:
        rr = [r for r in sv if r["def"] == d]
        rows2.append([tm.DEF_JP[d]] + [_f(st.mean(r[k]["test"][0] for r in rr), 4, True) for k in jp])
    t2 = r29._tbl(["荒れ度の定義", "M6b − 単一の最良印", "M6a − 常に M5b 35%", "M6a − 常に M1"], rows2, ["s", "n", "n", "n"])
    return t1 + "<h3>荒れ度の定義別（20 買い方の平均）</h3>" + t2


def rc_block(a: dict) -> str:
    rc, rc2 = a["reality_check_test"], a["reality_check_2023_25"]
    rows = [[" / ".join(r["strategy"]), _f(r["excess_per_race"], 4, True), _f(r["t"], 2), r["bet_races"]] for r in rc["top"][:10]]
    t = r29._tbl(["戦略（条件 / 印 / 買い方）", "1 レースあたりの超過損益", "t 値", "賭けたレース数"], rows, ["s", "n", "n", "n"])
    return (f"<p>固定の戦略 {rc['n_strategies']} 本（印 {len(tm.MARKS)} × 買い方 {len(tm.SCHEME_NAMES)} の基本 {len(tm.MARKS) * len(tm.SCHEME_NAMES)} 本、荒れ度 6 定義 × 堅い/荒れ × 基本、"
            f"荒れ度で M1 から切り替える規則 6 定義 × 4 印 × 買い方 {len(tm.SCHEME_NAMES)}）のうち、検証年 2024〜25 年（{rc['n_races']} レース）の最大の t 値は <b>{_f(rc['t_max'], 2)}</b>、"
            f"reality check の p 値は <b>{_f(rc['p'], 3)}</b>。2023〜25 年にまとめても p = {_f(rc2['p'], 3)}（t 最大 {_f(rc2['t_max'], 2)}）。</p>" + t)


def ruin_block(a: dict) -> str:
    rows = [[" / ".join(r["strategy"]), r["bet_races"], _pct(r["roi_cap_on_bets"]) if r["roi_cap_on_bets"] else "-", _pct(r["ruin"])] for r in a["ruin"]]
    return r29._tbl(["戦略", "賭けたレース数（検証年）", "回収率（賭けたレース）", "破産確率（元手 50 単位、1 単位/レース、500 レース）"], rows, ["s", "n", "n", "n"])


def final_block(f: dict) -> str:
    want = ["M1_pwin", "M5b_t35", "M3b_evwin_blend", "M6a_ent", "M6a_fav", "M6b_ent"]
    sch = ["tan1", "umaren_n3", "wide_n3", "wide_b4", "sanpuku_a4", "sanpuku_b4"]
    by = {(r["scheme"], r["rule"]): r for r in f["rows"]}
    gb = {r["scheme"]: r["rule"] for r in f["rows"] if r["rule"].startswith("global_best")}
    rows = []
    for s in sch:
        cells = [tm.SCHEME_JP[s]]
        for w in want:
            r = by.get((s, w))
            cells.append(f"{_pct(r['roi_cap'])} [{_pct(r['lo'])}, {_pct(r['hi'])}]" if r else "-")
        g = by.get((s, gb.get(s, "")))
        cells.append(f"{gb[s]}: {_pct(g['roi_cap'])} [{_pct(g['lo'])}, {_pct(g['hi'])}]" if g else "-")
        rows.append(cells)
    return r29._tbl(["買い方"] + ["M1 勝率", "M5b 35%", "M3b 単勝EV(混合)", "M6a(エントロピー)", "M6a(上位3人気)", "M6b(エントロピー)", "設計年の最良の単一印"], rows, ["s"] + ["n"] * 7)


def build(d: Path) -> None:
    a = json.loads((d / "analysis.json").read_text(encoding="utf-8"))
    f = json.loads((d / "final2026.json").read_text(encoding="utf-8"))
    sv = a["switch_value"]
    rc = a["reality_check_test"]
    it_line, it_tbl = interaction_block(a)
    m6a_m1 = st.mean(r["m6a_vs_m1"]["test"][0] for r in sv)
    m6a_m5 = st.mean(r["m6a_vs_m5"]["test"][0] for r in sv)
    m6b_g = st.mean(r["m6b_vs_global"]["test"][0] for r in sv)
    best2026 = max(f["rows"], key=lambda r: r["lo"])
    over2026 = [r for r in f["rows"] if r["lo"] > 1.0]
    b = {(x["mark"], x["scheme"]): x for x in a["baseline"]}
    style = __import__("re").search(r"<style>.*?</style>", r29.STYLE_SRC.read_text(encoding="utf-8"), __import__("re").S).group(0)
    n_trials = rc["n_all_strategies"]
    body = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>レースの性質で変える印と買い方（tp-43）— keiba-vpn</title>{style}</head>
<body><a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>レースの性質で変える印と買い方（tp-43）</h1>
<p class="muted">ユーザー提案: 人気（実力）で決着しそうなレースは勝率ベースで◎○▲を決め、荒れやすいレース（実力馬が負ける可能性が基準値より高い）は期待値が高く複勝確率もそこそこ高い馬を◎にして、印上位の馬を馬連・ワイド・3連複で買う。モデル出力（p_win / p_top2 / p_top3）をよりよく使う方法を探る。</p>
<div class="note"><b>要点.</b>
 <ul>
 <li><b>『荒れ度で印を切り替える』ことに価値は見つからなかった.</b>荒れ度の定義 6 種のどれで切り替えても、検証年（2024〜25 年）の回収率は、荒れ度を使わない単一の印に対して平均 {_f(m6b_g, 3, True)}（設計年で区分ごとに選んだ印）、常に M5b を使う場合に対して平均 {_f(m6a_m5, 3, True)}（ユーザー案の切り替え）で、改善しなかった。M1 との比較（{_f(m6a_m1, 3, True)}）が正なのは、切り替え先の M5b 自体が M1 より良いためで、荒れ度の効果ではない。</li>
 <li><b>印の付け方そのものは、組合せ券のうちワイドと3連複で効く傾向がある.</b>検証年の回収率は、勝率順（M1）が 0.87〜0.91 なのに対し、混合の期待値順（M3b）や『期待値が高く複勝確率が閾値以上』の印（M5b）は、ワイド・3連複で 0.97〜1.14 まで上がる組がある（馬連・馬単では一貫せず、馬単は 0.69〜0.88 で M1 以下）。ただしこの差は<b>市場オッズを混ぜた確率</b>（Benter 型）による。市場なしのモデル単体の期待値順（M3m）は、単勝（頭打ち後 1.00）以外では、まれな高配当に偏って組合せ券が 0.3〜0.8 と大きく劣る。</li>
 <li><b>どの条件でも 100% を超える確かな方策は無い.</b>固定の戦略 {n_trials} 本の reality check は p = {_f(rc['p'], 2)}（検証年の最大 t = {_f(rc['t_max'], 2)}）。2026 年（未使用 {f['races']} レース）は、規則のどれも区間下限が 100% を超えなかった（最大の区間下限 {_pct(best2026['lo'])}）。</li>
 <li>『堅い』レースで混合の期待値順（M3b）のワイド 4 頭ボックスなどが 100% を超える場面があり（検証年）、荒れるレースではなかった。ユーザー案の向き（荒れ→期待値◎）とは逆だが、多重比較の補正後は偶然の範囲。</li>
 </ul></div>
<h2>1. 方法</h2>
<ul>
 <li><b>データ</b>: tp-36 の入力（2022〜25 年は V4 の leave-one-year-out と bundle_v4_2024 の予測、確定の単勝・複勝オッズ、確定払戻。2026 年は bundle_v4_2025 の予測）。混合確率 p_blend は walk-forward の Benter 型で 2023 年から（2022 年は無い）。市場オッズは<b>馬券の判断と精算にだけ</b>使い、特徴量にはしない。{a['races']} レース（頭数 6 頭以上、払戻が曖昧なレースは除く）。</li>
 <li><b>期間の使い分け</b>: 荒れ度の区分の境目（3 分位）と、荒れ度ごとの印の選択は 2022〜23 年だけで決め、2024 年で確認、2025 年は参考。2026 年は最後に一度だけ。</li>
 <li><b>印の付け方（10 通り）</b>: """ + "、".join(f"{SHORT[m]} = {tm.MARK_JP[m]}" for m in tm.MARKS) + f"""。◎は順位 1 位、○▲は 2・3 位。M5 系は、複勝確率が閾値以上の馬を先に並べ、その中を期待値順にする（ユーザー案）。</li>
 <li><b>買い方（{len(tm.SCHEME_NAMES)} 通り）</b>: 単勝・複勝の◎1 点、馬連・ワイドの◎流し（○▲／4 頭／5 頭）とボックス（3・4 頭）、馬単の◎→相手、3連複の 3 頭ボックス・◎軸＋相手から 2 頭・4/5 頭ボックス。1 レースの賭け金は一定で、買い目で等分する。母数はレース単位。</li>
 <li><b>荒れ度の定義（6 種）</b>: """ + "、".join(tm.DEF_JP[k] for k in tm.DEFS) + f"""。いずれも値が大きいほど荒れやすい向きにそろえ、2022〜23 年の 3 分位で『堅い（下位 1/3）』『中』『荒れ（上位 1/3）』に分ける。</li>
 <li><b>回収率の頭打ち</b>: 1 レースの戻りを賭け金の 100 倍で頭打ちにした回収率（capped）を主指標にする（全列共通の上限で、頭打ちになったレースは全体の {_pct(a['frac_capped'])}）。まれな超高配当 1〜2 件で平均が動くため。生の値は付表に併記。</li>
 <li><b>検定</b>: レース単位のブートストラップで区間を出し、固定の戦略 {n_trials} 本には White の reality check（スチューデント化、検証年）を使う。その他に試した数: 荒れ度の定義×印×買い方の交互作用 {len(a['interaction'])} 検定、設計年での印の選択（定義 6 × 区分 3 × 買い方 {len(tm.SCHEME_NAMES)} × 印 {len(tm.MARKS)} の比較）。</li>
</ul>
<h2>2. 印の付け方ごとの回収率（capped、設計年 2022〜23 / 検証年 2024〜25）</h2>
<p class="muted">太字は検証年で 100% 以上。M3b・M5b・M7 は混合確率（2023 年以降）を使うので、設計年は 2023 年のみ。複勝 EV（M4）の複勝オッズは中央値で近似。</p>
{baseline_tbl(a)}
<h2>3. 荒れ度によって最適な印は変わるか</h2>
<p class="muted">各印の回収率から M1（勝率順）の回収率を引いた差を、全 {len(tm.SCHEME_NAMES)} 買い方で平均（capped）。荒れるほど M1 以外が有利になるなら、右の列ほど大きくなる。3 つの数字は 設計 / 確認（2024）/ 参考（2025）。</p>
{excess_tbl(a)}
<p>{it_line}。</p>
{it_tbl}
<div class="note">定義によって向きが一致せず（頭数は逆、乖離は正）、年が変わると符号が入れ替わる。『荒れるレースほど期待値の印が有利』という一貫した関係は確認できない。</div>
<h2>4. 荒れ度で切り替える価値</h2>
<p class="muted">設計年（2022〜23）だけで決めた規則を、検証年（2024〜25）で比べる。差は回収率（capped）の差で、区間はレース単位ブートストラップ。</p>
{switch_block(a)}
<h2>5. 多重比較の補正（reality check）と破産確率</h2>
{rc_block(a)}
<h3>破産確率</h3>
<p class="muted">元手 50 単位、1 レース 1 単位、500 レース賭け続けて元手が尽きる確率（検証年のレースから復元抽出）。</p>
{ruin_block(a)}
<h2>6. 2026 年（未使用の hold-out、一度だけ）</h2>
<p class="muted">{f['races']} レース（組合せ券の払戻が壊れた {f['excluded_broken_payoff']} レースと単勝オッズが壊れたレースは除く）。2026 年は複勝オッズが無いので M4 は評価できない。組合せ券の払戻は 2026 年 1〜3 月の連結を補正テーブルで補正したもの。</p>
{final_block(f)}
<p>100% を超える区間下限を持つ組は <b>{len(over2026)}</b> 件、最大の区間下限は {_pct(best2026['lo'])}（{best2026['rule']} / {tm.SCHEME_JP.get(best2026['scheme'], best2026['scheme'])}）。2026 年はモデルが市場に対して弱い（tp-36 §10.1）ことも差し引いて見る。</p>
<h2>7. 結論と UI の印の仕様案</h2>
<ul>
 <li>荒れ度で◎の付け方を切り替える方式は、設計・確認・参考のどの年でも一貫した改善が無く、reality check も通らないので<b>不採用</b>。</li>
 <li>一方、印の付け方としては『混合確率の期待値が高く、複勝確率が閾値以上の馬を◎・○・▲にする』（M5b）が、勝率順（M1）より組合せ券で良い傾向がある。ただし 100% 超の確かな根拠は無く、破産確率も高いので、<b>推奨ポートフォリオを置き換える根拠にはならない</b>。</li>
 <li>UI の印にする場合の案（実装は採用確定後）: 既定は勝率順の◎○▲、別の表示として『期待値◎（混合 EV 順、複勝確率 {int(35)}% 以上を優先）』を併記し、荒れ度の表示は『参考値』に留める。API は <code>p_win / p_top3 / ev_win_blend / marks.{{m1,m5b}}</code> を返す形が自然。</li>
 <li>限界: (1) 混合確率は市場オッズを使うので、印は確定オッズが分かる段階（T-15 以降）でしか作れない。(2) 組合せ券の期待値で買い目を選ぶ方式（ペアごとの EV）は tp-36 で評価済みで、tp-41 の同時確率の補正も回収率を改善しなかった。ここでは印の順位だけで買い目を作った。(3) 荒れ度の定義は 6 種に限った（tp-30 の事前予測の R² は 0.13/0.20 と限定的）。(4) 2022 年は混合確率が無く、混合系の印は 2023 年が設計年の全て。</li>
</ul>
<p class="muted">コード: <code>src/scripts/data/tp43_marks.py</code>（表の作成）、<code>tp43_analysis.py</code>（分析）、<code>tp43_report.py</code>。結果の生データ: <code>data/local/modeling/betting/tp43/</code>。</p>
</body></html>
"""
    CHAPTER.write_text(body, encoding="utf-8")
    print("保存:", CHAPTER)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(tm.OUT))
    build(Path(ap.parse_args().dir))


if __name__ == "__main__":
    main()
