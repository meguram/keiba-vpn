"""tp-31 の結果を docs の 1 章（course_specific_effects.html）にする。

python -m src.scripts.data.tp31_report   # ホストで実行（data/local/modeling/training_pipeline_v4/tp31/ の結果を読む）。出力は docs/html/modeling/chapters/
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd

from src.scripts.data import tp29_report as r29

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/local/modeling/training_pipeline_v4/tp31"
CHAPTER = ROOT / "docs/html/modeling/chapters/course_specific_effects.html"

ATTR_JP = {"sire": "父", "dam_sire": "母父", "jockey": "騎手", "trainer": "調教師", "style": "脚質", "post": "馬番位置", "style_post": "脚質×馬番位置", "sex": "性別", "age": "年齢",
           "carried": "斤量(レース内)", "bodywt": "馬体重", "rest": "前走間隔", "clsstep": "クラス変更"}
VAL_JP = {"s1": "先行寄り", "s2": "中間", "s3": "差し寄り", "in": "内", "mid": "中", "out": "外", "light": "軽い", "avg": "平均", "heavy": "重い", "w1": "〜439kg", "w2": "440〜479kg",
          "w3": "480〜519kg", "w4": "520kg〜", "r1": "〜14日", "r2": "15〜35日", "r3": "36〜70日", "r4": "71日〜", "down": "降級", "same": "同クラス", "up": "昇級",
          "a3": "3歳", "a4": "4歳", "a5": "5歳", "a6p": "6歳以上", "NA": "不明"}


def _course(k: str) -> str:
    p = k.split("|")
    if len(p) == 4:
        v = {"-": "", "不明": "(内外不明)"}.get(p[3], p[3] + "回り")
        return f"{r29.VENUE.get(p[0], p[0])}{p[1]}{p[2]}m{v}"
    if len(p) == 3:
        return f"{r29.VENUE.get(p[0], p[0])}{p[1]}{r29.BAND.get(p[2], p[2])}"
    return k


def build() -> None:
    summ = json.loads((OUT / "audit_summary.json").read_text(encoding="utf-8"))
    cells = pd.read_csv(OUT / "audit_cells.csv", dtype={"value": str})
    s1 = json.loads((OUT / "stage1/results.json").read_text(encoding="utf-8")) if (OUT / "stage1/results.json").exists() else {}
    s2 = json.loads((OUT / "stage2/results.json").read_text(encoding="utf-8")) if (OUT / "stage2/results.json").exists() else {}
    nm = r29.Namer()

    def value_label(feat: str, v: str) -> str:
        if feat == "jockey":
            return nm.jockey.get(v, v)
        if feat == "trainer":
            return nm.trainer.get(v, v)
        if feat == "sire":
            return (nm._anc(nm.by_sire[v], 1, 0) if v in nm.by_sire else "") or v
        if feat == "dam_sire":
            return (nm._anc(nm.by_dam[v], 2, 2) if v in nm.by_dam else "") or v
        if feat == "style_post":
            a, b = v.split("|") if "|" in v else (v, "")
            return VAL_JP.get(a, a) + "・" + VAL_JP.get(b, b)
        return VAL_JP.get(v, v)

    rep = cells[cells["sig_rep"]].copy()
    rep = rep[rep["level"] == "c0"].sort_values("pval")
    rows = []
    for _, x in rep.head(30).iterrows():
        direction = "V4 が過大評価（実際は期待より悪い）" if x["z"] < 0 else "V4 が過小評価（実際は期待より良い）"
        rows.append([html.escape(_course(x["course"])), ATTR_JP.get(x["feature"], x["feature"]), html.escape(str(value_label(x["feature"], str(x["value"])))), int(x["n"]), f"{x['z']:+.2f}", direction])
    tbl_rep = r29._tbl(["コース", "属性", "値", "件数", "z", "向き"], rows, ["", "", "", "n", "n", ""])

    by_f = pd.DataFrame(summ["by_feature"])
    by_f["属性"] = by_f["feature"].map(lambda x: ATTR_JP.get(x, x))
    tbl_f = r29._tbl(["属性", "検定したセル", "BH 有意", "うち再現あり"], [[r["属性"], int(r["tested"]), int(r["sig"]), int(r["sig_rep"])] for _, r in by_f.iterrows()], ["", "n", "n", "n"])

    cross = pd.DataFrame(summ["cross_year"])
    cross["属性"] = cross["feature"].map(lambda x: ATTR_JP.get(x, x))
    tbl_c = r29._tbl(["属性", "2023 κ=50", "2023 κ=200", "2024 κ=50", "2024 κ=200"],
                     [[r["属性"], f"{r['d_logloss_2023_k50']:+.5f}", f"{r['d_logloss_2023_k200']:+.5f}", f"{r['d_logloss_2024_k50']:+.5f}", f"{r['d_logloss_2024_k200']:+.5f}"] for _, r in cross.iterrows()],
                     ["", "n", "n", "n", "n"])

    def _grp_rows(res: dict) -> list[list]:
        out = []
        for g, s in (res.get("groups") or {}).items():
            c = s["vs_V4"]
            out.append([g, s["cols"], f"{c['Δwin']:+.5f}", f"{c['win_hi']:+.5f}", f"{c['Δtop2']:+.5f}", f"{c['Δtop3']:+.5f}", "採用" if c["採用"] else "不採用"])
        return out

    head = ["群", "列数", "Δ勝率", "95% 上限", "Δ連対", "Δ複勝", "採否"]
    cl = ["", "n", "n", "n", "n", "n", ""]
    b1 = s1.get("base", {}).get("levels", {})
    b2 = s2.get("base", {}).get("levels", {})
    t1 = r29._tbl(head, _grp_rows(s1), cl) if s1 else "<p>（第 1 段は未実施）</p>"
    t2 = r29._tbl(head, _grp_rows(s2), cl) if s2 else "<p>第 1 段を通った群はありませんでした（または未実施）。</p>"
    temp = pd.read_csv(OUT / "audit_temperature_by_course.csv") if (OUT / "audit_temperature_by_course.csv").exists() else pd.DataFrame()

    style = re.search(r"<style>.*?</style>", r29.STYLE_SRC.read_text(encoding="utf-8"), re.S).group(0)
    adopted = [g for g, s in (s2.get("groups") or {}).items() if s["vs_V4"]["採用"]]
    concl = ("第 2 段で採用基準を満たした群はありませんでした。V4 には何も足しません。" if not adopted else "第 2 段で採用基準を満たした群: " + "、".join(adopted) + "。")
    body = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>コース特注の要因（tp-31）— keiba-vpn</title>{style}</head>
<body><a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>コース特注の要因（tp-31）</h1>
<p class="muted">各コース（場×面×距離×内外）で、全体と効き方が違う特徴量を探し、効果のあるものは導入する。V4 アンサンブル（v4_e2）の予測残差を、コース×属性で系統的に説明できるかを調べ（監査）、そのうえでコース特注の特徴量を作って V4 の LightGBM に足した増分を測った。</p>
<div class="note"><b>要点.</b> {concl}
 V4 の予測残差はコース×属性でほぼ説明がつかない（6,755 セル中、BH 有意 {summ['sig_bh']}、再現あり {summ['sig_bh_and_replicated']}。前の年で推定したセル効果で補正しても次の年の LogLoss は ±0.0001 以内）。
 コース特注の差は、V4 がすでに持つ情報（b_aff_*、場×面×距離別の枠・脚質の表 course_draw_post、種牡馬×条件、厳密コースの騎手・調教師の成績）でほぼ吸収されている。</div>
<h2>1. 異質性の監査（V4 の予測残差がコース×属性でずれているか）</h2>
<p class="muted">対象: V4 アンサンブル（v4_e2）の valid 2022〜2024 の予測 {140865} 行。厳密コース 131。残差 r = 実際の1着 − 予測勝率。予測はレース内で合計 1 に正規化されているので、レース内の残差の合計は 0。そのため z はレースを単位にしたクラスター頑健な値（Σs/√Σs²、s = レース内のセルの残差の合計）を使い、多重検定は BH-FDR 5%。『再現あり』は、2022+2023 の符号が 2024 でも同じ向きで |z|>1。</p>
<p>検定したセル {summ['cells_tested']}（z の sd {summ['z_sd']:.3f}、|z|>2 の割合 {summ['frac_abs_z_gt2']:.3f}。帰無なら sd 1、0.046）。BH 有意 {summ['sig_bh']}、うち再現あり {summ['sig_bh_and_replicated']}。</p>
{tbl_f}
<h3>再現ありのセル（厳密コース、上位）</h3>
<p class="muted">z が負 = V4 の予測より実際が悪い（V4 がそのコースで過大評価）。同じセルが場×面×距離帯のレベルでも検出されることが多い。個人（騎手・調教師・種牡馬）のセルが多く、件数は 80〜140 と小さい。</p>
{tbl_rep}
<h3>前の年で推定したセル効果で補正したときの LogLoss の変化（負 = 改善）</h3>
<p class="muted">厳密コース×属性のセルごとに、前の年までの残差から縮約した効果（κ は縮約の強さ）を logit に足して、次の年の勝率 LogLoss がどれだけ変わるか。0.0001 を超える改善はない。</p>
{tbl_c}
<h2>2. 候補の特注特徴量の増分</h2>
<p class="muted">特徴量（コード: <code>src/pipeline/features/tp31_course_specific.py</code>）: 厳密コース×属性の as-of 経験ベイズ lift。階層縮約は 全体 → 属性 → 面×距離帯 → 場×面×距離帯 → 厳密コース。目的は着順残差（0.5 −(着順−1)/(頭数−1)）。
 群: T1 父・母父、T2 騎手・調教師、T3 脚質・馬番位置・両者の組、T4 性別・年齢・斤量・馬体重、T5 前走間隔・クラス変更、T6 属性 lift の合計と信頼度重みの平均。基準は V4 の LightGBM（{b1.get('logloss_win', float('nan')):.5f}、valid 2022+2023、seed 1 本）。</p>
<h3>第 1 段（valid 2022+2023、seed 1 本）</h3>
{t1}
<h3>第 2 段（valid 2024、新 seed 2 本）</h3>
<p class="muted">基準 {b2.get('logloss_win', float('nan')):.5f}</p>
{t2}
<h2>3. コース別の温度（p^b をレース内で再正規化）</h2>
<p class="muted">コースごとに確率の尖り具合 b を前の年で選んで次の年に当てると、LogLoss は {('平均 %+.5f（2023）、%+.5f（2024）' % (summ.get('temperature_gain_mean_2023') or 0.0, summ.get('temperature_gain_mean_2024') or 0.0))}（正 = 悪化）。対象 {summ.get('temperature_courses')} コース（走が 3,000 行以上のコース）。コース別の温度調整は使えない。</p>
<h2>4. 結論</h2>
<ul>
 <li>V4 は、コース特注の効きをすでに吸収している。コース×属性の残差は、検定でも、前の年からの補正でも、特徴量としての追加でも、基準を満たさなかった。</li>
 <li>残るコース特注の差は、特定の騎手・調教師・父が特定のコースで V4 の期待より悪い／良いというセルで、件数が小さい（80〜140）。個人の癖としては面白いが、将来の予測には再現しない。</li>
 <li>tp-15（概形類似）、tp-26（成績連動コースの転移）、tp-29（期待外れ・期待以上の差分）と合わせ、コース特注の特徴量は V4 に追加しない。</li>
</ul>
</body></html>
"""
    CHAPTER.write_text(body, encoding="utf-8")
    print("保存:", CHAPTER)


if __name__ == "__main__":
    build()
