"""tp-33 の結果（記述分析・年別の再現性・回帰・特徴量の増分）から、docs の章 transport_and_stay.html と /tmp/tp33_report.md を作る。

python -m src.scripts.data.tp33_report
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/local/modeling/training_pipeline_v4/tp33"
CHAPTER = ROOT / "docs/html/modeling/chapters/transport_and_stay.html"
HEAD = (ROOT / "docs/html/modeling/chapters/course_specific_effects.html").read_text(encoding="utf-8").split("<body>")[0].replace("コース特注の要因（tp-31）", "輸送と滞在（tp-33）")


def _tbl(df: pd.DataFrame, fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    h = "<tr>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns) + "</tr>"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, float):
                cells.append(f'<td class="num">{(fmt.get(c, "{:+.4f}")).format(v) if pd.notna(v) else "-"}</td>')
            else:
                cells.append(f"<td>{html.escape(str(v))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table>{h}{''.join(rows)}</table>"


def _jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> None:
    R = pd.read_csv(OUT / "descr_table.csv")
    Y = pd.read_csv(OUT / "descr_by_year.csv")
    RAW = pd.read_csv(OUT / "descr_raw_by_year.csv")
    LG = pd.read_csv(OUT / "descr_logit.csv", index_col=0) if (OUT / "descr_logit.csv").exists() else pd.DataFrame()
    s1, s2 = _jload(OUT / "stage1/results.json"), _jload(OUT / "stage2/results.json")
    audit = Path("/tmp/tp33_data_audit.md").read_text(encoding="utf-8") if Path("/tmp/tp33_data_audit.md").exists() else ""

    d1 = R[["区分", "出走数", "top3率", "予測top3", "top3残差", "top3残差_lo", "top3残差_hi", "win率", "予測win", "win残差", "win残差_lo", "win残差_hi"]].copy()
    d1.columns = ["区分", "出走数", "複勝率", "V4 予測の複勝率", "複勝残差", "95%下限", "95%上限", "勝率", "V4 予測の勝率", "勝率残差", "下限", "上限"]
    fmt = {"出走数": "{:,.0f}", "複勝率": "{:.3f}", "V4 予測の複勝率": "{:.3f}", "勝率": "{:.3f}", "V4 予測の勝率": "{:.3f}"}

    def stage_rows(res: dict) -> pd.DataFrame:
        rows = []
        for g, s in (res.get("groups") or {}).items():
            c = s["vs_V4"]
            rows.append({"群": g, "列数": s["cols"], "Δ勝率": c["Δwin"], "上限": c["win_hi"], "Δ連対": c["Δtop2"], "Δ複勝": c["Δtop3"], "採否": "採用" if c["採用"] else "不採用"})
        return pd.DataFrame(rows)

    S1, S2 = stage_rows(s1), stage_rows(s2)
    lines = ["# tp-33 輸送の有無・滞在と好走率", ""]
    lines += ["## 記述分析（valid 2022〜24 の V4 アンサンブル予測に対する残差。95% は行ではなくレース単位のブートストラップ）", d1.to_markdown(index=False, floatfmt=".4f") if hasattr(d1, "to_markdown") else d1.to_string()]
    lines += ["", "## 年別の再現性（複勝の残差）", Y.to_string(index=False), "", "## 生の複勝率の差（遠征 − ホーム/近距離、場×面×格で層別）", RAW.to_string(index=False)]
    if len(LG):
        lines += ["", "## 回帰（複勝 ~ 遠征 + 滞在 + 初遠征 + V4 の予測 logit + 場 + 格、レース単位のクラスター頑健）", LG.round(4).to_string()]
    if len(S1):
        lines += ["", "## 特徴量の増分（V4 全体に対して）", "### 第 1 段（valid 2022+2023、seed 1）", S1.round(5).to_string(index=False)]
    if len(S2):
        lines += ["### 第 2 段（valid 2024、新 seed 2）", S2.round(5).to_string(index=False)]
    Path("/tmp/tp33_report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")

    adopted = [r["群"] for _, r in S2.iterrows() if r["採否"] == "採用"] if len(S2) else []
    verdict = ("第 2 段で採用基準を満たした群: " + "、".join(adopted)) if adopted else "第 2 段で採用基準（CI 上限 &lt; 0・効果 0.0002 以上）を満たした群はありませんでした。V4 には何も足しません。"
    body = [f'<body><a href="../../index.html" class="muted">← ドキュメント一覧</a><h1>輸送と滞在（tp-33）</h1>',
            '<p class="muted">輸送のありなし（厩舎の拠点 × 開催場）と、滞在（レース前 28 日の追い切りの場所）で、好走率が変わるかの分析。V4 アンサンブルの予測に対する残差で見るので、馬の能力・クラス・場の差は予測に含まれたうえでの効果です。</p>',
            f'<div class="note"><b>要点.</b> {verdict}</div>',
            "<h2>1. データ上、輸送と滞在を判定できるか</h2><ul>",
            "<li><b>追い切りの場所</b>は、馬別調教ページのコース名（栗坂・CW・南W・美坂・函W・札ダ・小ダ 等）から読める。約 119 万本のうち読めないのは 0.8%、地方競馬場は 0.04%。</li>",
            "<li><b>調教師の拠点（栗東/美浦）</b>は、調教師の馬の追い切りの場所の多数派から決めた。352 人全員が判定でき、純度は 1.0（全員が拠点の施設だけで追い切っている）。栗東 238・美浦 114。</li>",
            "<li><b>輸送区分</b>は、拠点から開催場への概算の道のりで 4 区分（0 ホーム〜120km / 1 近距離〜350km / 2 中〜長距離〜700km / 3 遠方・北海道）。実際の輸送経路・所要時間・到着時刻は取得できない。</li>",
            "<li><b>滞在</b>（拠点以外での追い切り）が実際に多いのは<b>札幌・函館（85〜89%）と小倉（18%）</b>で、他の開催場では 1% 未満。つまり、データ上の滞在は『北海道・小倉の開催に合わせた現地調教』がほぼすべて。福島・新潟・東京・中山・関西の開催では、遠征馬は拠点で追い切って当日（または前日）に輸送している。</li>",
            "<li>被覆: レース前 28 日に追い切りの記録がある出走の割合は 2020 年 83%・2021 年 57%・2022 年 76%・2023 年 85%・2024 年 92%・2025 年 97%。記録が無い行は滞在を判定不可（欠損）。</li></ul>",
            "<h2>2. 記述分析</h2><p class='muted'>残差 = 実際の率 − V4 の予測の率。正なら予測より良く走っている。</p>", _tbl(d1, fmt),
            "<h3>年別の再現性（複勝の残差）</h3>", _tbl(Y, {"出走数": "{:,.0f}", "年": "{:.0f}"}),
            "<h3>生の複勝率の差（遠征 − ホーム/近距離、場×面×格で層別・加重）</h3><p class='muted'>能力の差（遠征に出す馬は選ばれている）を含む。2020〜23 年は +1.5〜2.2 ポイントだが、2024 年 +0.3、2025 年 −0.4 と消えている。</p>", _tbl(RAW, {"年": "{:.0f}", "層の数": "{:.0f}"}),
            ]
    if len(LG):
        body += ["<h3>交絡を制御した回帰</h3><p class='muted'>複勝 ~ 遠征 + 滞在 + 初遠征 + V4 の予測 logit + 場 + 格（レース単位のクラスター頑健 SE）。</p>", _tbl(LG.reset_index().rename(columns={"index": "項"}), {"coef": "{:+.4f}", "se": "{:.4f}", "p": "{:.3f}"})]
    body += ["<h2>3. 特徴量の増分（V4 全体に対して）</h2>", "<p class='muted'>LightGBM、valid の年を前から順に使う expanding。第 1 段は 2022+2023・seed 1 本、第 2 段は 2024・新 seed 2 本。採否は CI 上限 &lt; 0・効果 ≥ 0.0002・連対/複勝の悪化 ≤ 0.0002。</p>"]
    if len(S1):
        body += ["<h3>第 1 段</h3>", _tbl(S1, {"列数": "{:.0f}", "Δ勝率": "{:+.5f}", "上限": "{:+.5f}", "Δ連対": "{:+.5f}", "Δ複勝": "{:+.5f}"})]
    if len(S2):
        body += ["<h3>第 2 段</h3>", _tbl(S2, {"列数": "{:.0f}", "Δ勝率": "{:+.5f}", "上限": "{:+.5f}", "Δ連対": "{:+.5f}", "Δ複勝": "{:+.5f}"})]
    body += ["<h2>4. 解釈</h2><div class='note'>輸送区分・滞在・遠征の連続・初遠征のいずれも、V4 の予測に対する残差はほぼ 0 で、年によって向きも変わる。例外は『美浦所属 × 関西への遠征（区分 2）』の複勝が予測より約 1.0 ポイント低い点だが、約 25 区分を見た中の 1 つで、多重比較を考えると偶然の範囲を超えない。滞在も同様で、北海道・小倉の現地調教の馬は拠点調教の馬と、V4 の予測に対する残差で差がない。</div>",
             "<p class='muted'>コード: <code>src/pipeline/features/tp33_transport.py</code>、<code>src/scripts/data/tp33_eval.py</code>。テスト: <code>tests/pipeline/test_tp33_transport.py</code>（先読みなしの検査を含む）。</p></body></html>"]
    CHAPTER.write_text(HEAD + "".join(body), encoding="utf-8")
    print("章を書いた:", CHAPTER)


if __name__ == "__main__":
    main()
