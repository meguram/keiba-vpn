"""tp-34 の結果を docs の 1 章（rest_and_interval.html）にする。

python -m src.scripts.data.tp34_report   # ホストで実行（data/local/modeling/training_pipeline_v4/tp34/ の結果を読む）。出力は docs/html/modeling/chapters/
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

from src.pipeline.features import tp34_rest as tr
from src.scripts.data import tp29_report as r29

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/local/modeling/training_pipeline_v4/tp34"
CHAPTER = ROOT / "docs/html/modeling/chapters/rest_and_interval.html"

GROUP_JP = {"R1_band": "間隔の帯・叩き何戦目・その馬の過去の間隔との比較", "R2_horse_pref": "馬ごとの同じ間隔の帯／休み明け／叩き何戦目での過去成績", "R3_tataki_gain": "馬ごとの叩き 2・3 戦目の上昇度",
            "R4_trainer": "調教師ごとの休み明け／叩き 2 戦目／間隔の帯の成績", "R5_cross": "年齢・クラス・面・性別・距離・季節 × 間隔の帯の全馬横断の成績", "R6_prev": "前走の消耗（着差・上がり・押し上げ）× 間隔",
            "R7_work": "休養中の調教の本数・強度・カバー率", "R8_season": "前走月 × 今回月（夏休み・年末年始）", "ALL_PASSED": "第 1 段を通った群の同時追加"}


def _f(x, nd=4, sign=True):
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _pct(x):
    return "-" if x is None or (isinstance(x, float) and x != x) else f"{x * 100:.1f}%"


def _resid_tbl(rows: list[dict], label: str = "区分", years: bool = True) -> str:
    head = [label, "件数", "勝率 実/予測", "勝率の残差 [95%区間]", "複勝率 実/予測", "複勝率の残差 [95%区間]"] + (["複勝残差 2022/23/24"] if years else [])
    cls = ["", "n", "n", "n", "n", "n"] + (["n"] if years else [])
    body = []
    for r in rows:
        w = f"{_pct(r['win_rate'])} / {_pct(r['win_pred'])}"
        t3 = f"{_pct(r['top3_rate'])} / {_pct(r['top3_pred'])}"
        wr = f"{_f(r['win_resid'])} [{_f(r['win_lo'])}, {_f(r['win_hi'])}]"
        tr_ = f"{_f(r['top3_resid'])} [{_f(r['top3_lo'])}, {_f(r['top3_hi'])}]"
        sig3 = r["top3_lo"] > 0 or r["top3_hi"] < 0
        sigw = r["win_lo"] > 0 or r["win_hi"] < 0
        tr_ = f"<b>{tr_}</b>" if sig3 else tr_
        wr = f"<b>{wr}</b>" if sigw else wr
        row = [html.escape(str(r["cat"])), int(r["n"]), w, wr, t3, tr_]
        if years:
            row.append(" / ".join(_f(r.get(f"top3_resid_{y}")) for y in (2022, 2023, 2024)))
        body.append(row)
    return r29._tbl(head, body, cls)


def _grp_tbl(res: dict) -> str:
    rows = []
    for g, s in (res.get("groups") or {}).items():
        c = s["vs_V4"]
        rows.append([g, html.escape(GROUP_JP.get(g, "")), s["cols"], f"{c['Δwin']:+.5f}", f"{c['win_hi']:+.5f}", f"{c['Δtop2']:+.5f}", f"{c['Δtop3']:+.5f}", "採用" if c["採用"] else "不採用"])
    return r29._tbl(["群", "内容", "列数", "Δ勝率", "95% 上限", "Δ連対", "Δ複勝", "採否"], rows, ["", "", "n", "n", "n", "n", "n", ""]) if rows else "<p>（未実施）</p>"


def build() -> None:
    d = json.loads((OUT / "describe.json").read_text(encoding="utf-8"))
    def load(lab: str) -> dict:
        p_ = OUT / lab / "results.json"
        return json.loads(p_.read_text(encoding="utf-8")) if p_.exists() else {}

    # 第 1 段: stage1（R1〜R5）・stage1b（R6〜R8）のうち、端数のリークを直す前に回した R2・R3・R6・R7 は stage1c / stage1d のやり直しで置き換える
    s1 = load("stage1")
    s1["groups"] = {**(s1.get("groups") or {}), **(load("stage1b").get("groups") or {}), **(load("stage1c").get("groups") or {}), **(load("stage1d").get("groups") or {})}
    s1["groups"] = {g: s1["groups"][g] for g in tr.GROUPS if g in s1["groups"]}
    leaky = {"R6_prev": (load("stage1b").get("groups") or {}).get("R6_prev"), "R2_horse_pref": (load("stage1").get("groups") or {}).get("R2_horse_pref")}
    leaky2 = {"R6_prev": (load("stage2").get("groups") or {}).get("R6_prev"), "R2_horse_pref": (load("stage2").get("groups") or {}).get("R2_horse_pref")}
    s2 = load("stage2c")
    s2["groups"] = {**{g: v for g, v in (load("stage2").get("groups") or {}).items() if g == "R1_band"}, **(s2.get("groups") or {})}
    style = re.search(r"<style>.*?</style>", r29.STYLE_SRC.read_text(encoding="utf-8"), re.S).group(0)
    adopted = [g for g, s in (s2.get("groups") or {}).items() if s["vs_V4"]["採用"]]
    b1 = (s1.get("base") or {}).get("levels", {})
    b2 = (s2.get("base") or {}).get("levels", {})
    hs, ts = d.get("horse_split_half") or {}, d.get("trainer_split") or {}

    def _lk(x):
        return f"{x['vs_V4']['Δwin']:+.5f}（上限 {x['vs_V4']['win_hi']:+.5f}）" if x else "-"
    leak_rows = [[g, _lk(leaky.get(g)), _lk(leaky2.get(g))] for g in ("R6_prev", "R2_horse_pref")]
    leak_tbl = r29._tbl(["群", "第 1 段（修正前）", "第 2 段（修正前）"], leak_rows, ["", "n", "n"])
    concl = ("第 2 段で採用基準を満たした群はありませんでした。V4 には何も足しません。" if not adopted else "第 2 段で採用基準を満たした群: " + "、".join(adopted) + "。")

    def sec(title: str, key: str, label: str, years: bool = True, note: str = "") -> str:
        if key not in d:
            return ""
        return f"<h3>{title}</h3>" + (f'<p class="muted">{note}</p>' if note else "") + _resid_tbl(d[key], label, years)

    subs = ""
    for nm in ("休み明け初戦", "標準的な間隔"):
        subs += (sec(f"{nm}: 年齢別", f"age|{nm}", "年齢", True) + sec(f"{nm}: クラス別", f"cls|{nm}", "クラス", True) + sec(f"{nm}: 距離別", f"dist|{nm}", "距離", True)
                 + sec(f"{nm}: 面別", f"surface|{nm}", "面", True))
    body = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>休養明け・レース間隔の分析（tp-34）— keiba-vpn</title>{style}</head>
<body><a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>休養明け・レース間隔の分析（tp-34）</h1>
<p class="muted">前走からの間隔・休養明けの叩き何戦目・その組み合わせ（前走の結果、年齢、クラス、乗り替わり、休養中の調教）が好走率にどう効くか。V4 アンサンブル（v4_e2）の valid 2022〜2024 の予測 {d['n_rows']:,} 行（{d['n_races']:,} レース）に対する残差（実際の率 − 予測の平均）を、レース単位のブートストラップ 95% 区間つきで見る。特徴量（コード: <code>src/pipeline/features/tp34_rest.py</code>）は、日付が厳密に前の走・調教だけから作る（as-of）。市場オッズ・人気は使わない。</p>
<div class="note"><b>要点.</b> {concl}
 V4 は間隔・休養明けの効果をほぼ吸収している。残る系統的なずれは、休み明け初戦の複勝率を V4 が 0.7 ポイント（相対 4%）ほど高く見積もる点（2022〜2024 の全年で同じ向き）と、半年超・14〜26 週の長い休養の複勝率をやや高く見積もる点。
 馬ごとの『休み明けの得意不得意』には再現性がある（偶数番目と奇数番目の休み明けの相関 r = {hs.get('r', float('nan')):.2f}、全体への補正後 {hs.get('spearman_brown', float('nan')):.2f}）が、調教師ごとの休み明けの仕上げ傾向には再現性がない（前半と後半の相関 r = {ts.get('r_top3', float('nan')):+.2f}）。</div>
<h2>1. 間隔の帯・叩き何戦目別の好走率と V4 に対する残差</h2>
<p class="muted">残差が正 = 実際が V4 の予測より良い。太字は 95% 区間が 0 を含まないもの。複勝残差の年別は 2022 / 2023 / 2024 の順で、全年で同じ向きなら再現性がある。</p>
{sec('間隔の帯', 'band', '前走からの間隔')}
{sec('叩き何戦目', 'tataki', '休養明けからの走数', note='休養明け = 前走から 90 日以上（初戦 = 叩き 1 戦目）。デビュー前後でデータ上の前走が無い走は除く。')}
{sec('休み明け初戦と標準的な間隔の比較', 'lay_vs_ref', '区分')}
<h2>2. 属性との組み合わせ</h2>
{subs}
{sec('前走の結果 × 間隔', 'prev×gap', '間隔 × 前走', False, '前走の着順は、レース内の相対順位（0 = 1 着、1 = 最下位）。')}
{sec('乗り替わり × 間隔', 'jockey×gap', '間隔 × 騎手', False)}
{sec('休養中の調教の本数（休養 60 日以上）', 'work|休養60日以上', '前走から今回までの追い切りの本数', False, 'horse_training の記録。『記録なし』は本数 0 ではなく、調教ページが無い／読めない馬を含む。')}
<h2>3. 再現性（馬・調教師の得意不得意）</h2>
<ul>
 <li><b>馬ごと:</b> {hs.get('horses', 0):,} 頭（休み明け初戦が 4 回以上）。休み明け初戦の成績（同じ馬のそれ以外の走との差）を、偶数番目と奇数番目で分けた平均の相関は r = {hs.get('r', float('nan')):.2f}（スピアマン・ブラウン補正で {hs.get('spearman_brown', float('nan')):.2f}）。馬ごとに休み明けの得意不得意がある。平均では、休み明け初戦は同じ馬のそれ以外の走より fp（正規化着順）が {hs.get('mean_x', float('nan')):+.3f} 悪い。</li>
 <li><b>調教師ごと:</b> {ts.get('trainers', 0)} 人。休み明け初戦の V4 に対する残差を前半（2022〜23）と後半（2024）で比べた相関は、複勝 r = {ts.get('r_top3', float('nan')):+.2f}、勝率 r = {ts.get('r_win', float('nan')):+.2f}。調教師の休み明けの仕上げ傾向は再現しない。</li>
</ul>
<h2>4. 特徴量の増分（V4 全体に対して）</h2>
<p class="muted">群: """ + "、".join(f"{g} = {GROUP_JP[g]}" for g in tr.GROUPS) + f"""。いずれも as-of（日付が厳密に前の走・調教のみ）。基準は V4（新しい土台）の LightGBM。採否は『95% 区間の上限 &lt; 0、効果 ≥ 0.0002、連対・複勝の悪化 0.0002 以内』。G1 の rest 群（前走間隔の日数・フラグ・直近の出走数）、tp-31 の T5_rest_cls（前走間隔×コース）は別。</p>
<h3>第 1 段（valid 2022+2023、seed 1 本）</h3>
<p class="muted">基準 {b1.get('logloss_win', float('nan')):.5f}</p>
{_grp_tbl(s1)}
<h3>第 2 段（valid 2024、新 seed 2 本）</h3>
<p class="muted">基準 {b2.get('logloss_win', float('nan')):.5f}</p>
{_grp_tbl(s2)}
<h2>5. 見かけの大きな改善はリークだった（浮動小数点の端数）</h2>
<p>前走の消耗の群 R6 は、第 1 段で勝率 −0.00329、新しい seed の第 2 段でも −0.00379（上限 −0.00323）と、これまでのどの特徴量より大きな改善に見えた。調べると、
 <code>rs6_prev_fp_vs_mean</code>（前走の正規化着順 − 過去の平均）の 1 列だけで同じ改善が出て（この列を除くと消える）、しかも改善は『過去 1 走だけの馬（11.5%）』の行にすべて集中していた。
 原因は、過去の合計を『累積和 − 自分』で作っていたこと。数学的には『自分より前の合計』だが、浮動小数点では (a + b) − b が a に戻らないことがあり、端数に今回の値 b の痕跡が残る。
 今回が 1 着（正規化着順が厳密に 0）のときだけ端数が 0 になる、というパターンを木モデルが拾い、勝ち馬を当てていた（値の差は 1e-16 で、通常の許容誤差つきの先読み検査では見えない）。</p>
{leak_tbl}
<ul>
 <li><b>確認:</b> 『前走を除いた平均との差』（同じ情報で端数が出ない形）では改善が出ない。累積和を 1 行ずらして作る形（<code>_prior_cumsum</code>）に直すと、R6 は第 1 段で +0.00010（上限 +0.00036）になり、改善は消えた。馬ごとの得意度 R2 も同じ作りの列（<code>*_aff_fp</code>）が影響していた。</li>
 <li><b>直した箇所:</b> <code>factory_self.excl_cum</code>・<code>_same_key_sum</code>・<code>g_timedev</code>・<code>g_attrs</code> の日次の累積、<code>factory_field</code>（calendar_env の同日の先のレースの偏り）。同じ『累積和 − 自分』のパターンは、他にも <code>tp29_course_aptitude</code>・<code>tp31_course_specific</code>・<code>tp32_equipment</code>・<code>factory_slowstart</code> にある。</li>
 <li><b>検査の追加:</b> 先読み検査テストを『厳密一致（端数まで）』にし、<code>excl_cum</code> が今回の値にビットまで依存しないことを確かめるテストを足した。実データ 1 日分（2023-04-15、516 行）で、G1 の採用群（change・experience・lags_margin・pair・rest・seq_stats・timedev）は修正の前から厳密一致だった。</li>
 <li><b>確認が要るもの:</b> 馬主系（factory_conn）、種牡馬 ID×条件（factory_ped）、G3 の他の群、補助ヘッド（tp25）は、この端数の観点では未検査。V4 の採用群に同じ痕跡が無いことを、厳密一致のテストで確かめる必要がある。</li>
</ul>
<h2>6. 結論</h2>
<ul>
 <li>{concl}</li>
 <li>V4 は、間隔・休養明けの効きの大部分をすでに持つ（G1 の rest 群・experience 群、TL_A の追い切り、S3 の前段、補助ヘッド）。休み明け初戦の複勝率を高く見積もるずれは小さく、特徴量として足しても勝率の改善にはならない。</li>
 <li>馬ごとの休み明けの得意不得意には再現性があるので、運用では『その馬の過去の休み明け成績』を見る価値がある（人が読む用）。調教師ごとの仕上げ傾向は再現しない。</li>
</ul>
</body></html>
"""
    CHAPTER.write_text(body, encoding="utf-8")
    print("保存:", CHAPTER)


if __name__ == "__main__":
    build()
