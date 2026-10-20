"""tp-51（アンサンブルの構造）の結果を docs の 1 章（ensemble_structure.html）にする。

python -m src.scripts.data.tp51_report --dir <結果ディレクトリ（stack/results.json, gm_*.json, gm_run.log などがあるところ）>
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

from src.scripts.data import tp29_report as r29

ROOT = Path(__file__).resolve().parents[3]
CHAPTER = ROOT / "docs/html/modeling/chapters/ensemble_structure.html"

SCHEME_JP = {
    "S1_wlogit_all": "3 対象とも weighted_logit",
    "S1b_greedy_all": "3 対象とも前進選択（greedy）",
    "S2_stack_lr": "ロジスティック回帰（その対象の 7 logit）",
    "S3_lr_cross": "クロス対象 LR（3 対象 × 7 = 21 logit）",
    "S3c_lr_cross_C003": "クロス対象 LR（正則化を強く C=0.03）",
    "S3d_lr_cross_C1": "クロス対象 LR（正則化を弱く C=1）",
    "S3e_lr_ownwin": "LR（その対象の 7 logit + 勝率の 7 logit）",
    "S4_lr_full": "LR（21 logit + 平均・標準偏差・レース内 z + 文脈）",
    "S3w_win_cross_lr": "勝率だけクロス対象 LR、連対・複勝は現行",
    "S4w_win_full_lr": "勝率だけ LR（文脈つき）、連対・複勝は現行",
    "S3x_win_cross_oth_ownwin": "勝率はクロス対象 LR、連対・複勝は LR（own+win）",
    "S5_lgbm_meta": "メタ LightGBM（浅い・文脈つき）",
    "S5b_lgbm_meta_noctx": "メタ LightGBM（文脈なし）",
    "S5c_lgbm_meta_deep": "メタ LightGBM（深め）",
    "S6_nn_meta": "メタ NN（勝率はレース内 softmax、文脈つき）",
    "S6b_nn_meta_noctx": "メタ NN（文脈なし）",
    "S7_group_stack": "2 段（GBM 群・NN 群の平均 logit → LR）",
    "S8_segment": "面 × 距離帯ごとの weighted_logit（全体へ縮約）",
    "S9_temp": "後段: 温度（logit の傾きと切片）",
    "S9b_iso": "後段: 等張回帰",
    "S10_harville_mix": "後段: 連対・複勝を『直接』と『勝率からの割引 Harville』の混合に",
    "S11_L3_lgbm_nn": "L3: 現行 + メタ LightGBM + メタ NN の logit 平均",
    "S11b_L3_lr": "L3: 現行 + クロス対象 LR の logit 平均",
    "S11c_L3_all": "L3: 現行 + LightGBM + NN + LR",
    "S11d_L3_group": "L3: 現行 + 2 段（群）",
    "S11e_L3lr_win2": "L3（現行 + クロス LR）を勝率・連対に、複勝は現行",
    "S11f_L3all_win2": "L3（現行 + LightGBM + NN + LR）を勝率・連対に、複勝は現行",
    "S11g_L3_lr_full": "L3: 現行 + クロス LR + 文脈つき LR",
    "S11h_L3_lr_full_win2": "L3（現行 + 2 つの LR）を勝率・連対に、複勝は現行",
    "S11i_L3_w05": "L3: 現行 0.65 + クロス LR 0.35",
    "S11j_L3_w05_win2": "L3（0.65 : 0.35）を勝率・連対に、複勝は現行",
}


def _f(x, nd=5, sign=True):
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _cell(c: dict | None) -> str:
    if not c:
        return "-"
    ok = "<b>採用</b>" if c.get("採用") else "不採用"
    return f"{_f(c['Δwin'])}（上限 {_f(c['win_hi'])}） / {_f(c['Δtop2'])} / {_f(c['Δtop3'])} / {ok}"


def stack_table(res: dict) -> str:
    names = []
    for k in res:
        n = k.split("|")[0]
        if n not in names and n != "S0_spec":
            names.append(n)
    rows = []
    for n in names:
        jp = SCHEME_JP.get(n, n)
        rows.append([f"<code>{html.escape(n)}</code>", html.escape(jp), _cell(res.get(f"{n}|loyo")), _cell(res.get(f"{n}|forward"))])
    return r29._tbl(["方式", "内容", "LOYO（Δ勝率（95% 上限）/ Δ連対 / Δ複勝 / 判定）", "前進（同）"], rows, ["s", "s", "n", "n"])


def pool_table(res: dict) -> str:
    rows = []
    for k, v in res.items():
        n, mode = k.split("|")
        if n.startswith("P_"):
            rows.append([f"<code>{html.escape(n)}</code>", mode, _cell(v)])
    return r29._tbl(["V4 の 7 メンバーに足したもの", "評価", "Δ勝率（95% 上限）/ Δ連対 / Δ複勝 / 判定"], rows, ["s", "s", "n"]) if rows else "<p>（未実施）</p>"


def gm_table(comb: dict) -> str:
    rows = []
    for k, c in comb.items():
        rows.append([f"<code>{html.escape(k)}</code>", f"{_f(c['Δwin'])}（上限 {_f(c['win_hi'])}） / {_f(c['Δtop2'])} / {_f(c['Δtop3'])}", "<b>採用</b>" if c.get("採用") else "不採用"])
    return r29._tbl(["グルーピング型の合成", "全列モデル比 Δ勝率（95% 上限）/ Δ連対 / Δ複勝", "判定"], rows, ["s", "n", "s"]) if rows else "<p>（未実施）</p>"


def single_models(log: str) -> list[list]:
    rows = []
    for m in re.finditer(r"\] (\S+): win ([0-9.]+) top2 ([0-9.]+) top3 ([0-9.]+) cols=(\d+) \((\d+)s\)", log):
        rows.append([f"<code>{m.group(1)}</code>", m.group(2), m.group(3), m.group(4), m.group(5), m.group(6) + " 秒"])
    return rows


def groups_table(csv_path: Path, imp: dict) -> str:
    import pandas as pd

    t = pd.read_csv(csv_path)
    rows = []
    for g, d in t.groupby("group"):
        ex = ", ".join(d["column"].head(4))
        rows.append([f"<code>{g}</code>", len(d), f"{100 * imp.get(g, float('nan')):.1f}%" if g in imp else "-", html.escape(ex) + " …"])
    rows.sort(key=lambda r: -float(r[2].rstrip("%")) if r[2] != "-" else 0)
    return r29._tbl(["グループ", "列数", "全列モデルの gain 比", "列の例"], rows, ["s", "n", "n", "s"])


def build(d: Path) -> None:
    stack = json.loads((d / "stack/results.json").read_text(encoding="utf-8"))
    comb = json.loads((d / "gm_combine.json").read_text(encoding="utf-8")) if (d / "gm_combine.json").exists() else {}
    imp = json.loads((d / "gm_importance.json").read_text(encoding="utf-8")) if (d / "gm_importance.json").exists() else {}
    corr = json.loads((d / "gm_corr.json").read_text(encoding="utf-8")) if (d / "gm_corr.json").exists() else {}
    log = (d / "gm_run.log").read_text(encoding="utf-8") if (d / "gm_run.log").exists() else ""
    leak = (d / "stack_leak.log").read_text(encoding="utf-8") if (d / "stack_leak.log").exists() else ""
    summary = (d / "summary.html").read_text(encoding="utf-8") if (d / "summary.html").exists() else ""
    n_schemes = len({k.split("|")[0] for k in stack}) + len(comb)
    style = re.search(r"<style>.*?</style>", r29.STYLE_SRC.read_text(encoding="utf-8"), re.S).group(0)
    csv_path = ROOT / "src/scripts/data/tp51_feature_groups.csv"
    singles = single_models(log)
    leak_ok = leak.count("OK（評価年のラベルを使っていない）")
    body = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" /><title>アンサンブルの構造（多層化・バギング・グルーピング）— tp-51</title>{style}</head>
<body><a href="../../index.html" class="muted">← ドキュメント一覧</a>
<h1>アンサンブルの構造（多層化・バギング・特徴量グルーピング）— tp-51</h1>
<p class="muted">ユーザー要望: アンサンブルをもっと多層にしたり、バギングなど様々な表データ予測の技術を考慮する余地があるか。さらに『各学習器で使う特徴量をグルーピングして行う』戦略も。V4（7 メンバー、1,135 列、weighted_logit / greedy の合成）の<b>モデル構造側</b>を、V4 の特徴量のまま評価した。</p>
{summary}
<h2>1. 評価の枠</h2>
<ul>
 <li><b>土台</b>: V4 の 7 メンバーの OOF（valid 2022・2023・2024 の 140,865 行）。現行の合成（勝率 weighted_logit、連対・複勝 greedy）の LOYO を基準（Δ=0）にする。</li>
 <li><b>LOYO</b>: 1 年を外して他の 2 年で合成を学習し、外した年を予測（V4 の採否に使った方式）。<b>前進</b>: 評価年より前の年だけで学習（2023 は 2022 のみ、2024 は 2022〜23）。前進の方が厳密にリークが無い。採否は既定の基準（95% paired-bootstrap・レース単位の CI 上限 &lt; 0、効果 ≥ 0.0002、連対・複勝の悪化 ≤ 0.0002）。</li>
 <li>test 2025・2026 年は使っていない。市場オッズ・人気は使っていない。試した構成の総数: 約 {n_schemes} 通り。</li>
 <li><b>リーク防止の検査</b>: 評価年の着順を並べ替えても、メタ学習の予測が 1e-9 以内で変わらないこと（評価年のラベルを学習に使っていない）を 3 方式で確認（{leak_ok} 件 OK）。単体テスト: <code>tests/pipeline/test_tp51_stack.py</code>。</li>
</ul>
<h2>2. 多層スタッキング・後段（OOF の合成の仕方）</h2>
{stack_table(stack)}
<h2>3. 特徴量グルーピング型のメンバー</h2>
<p class="muted">V4 の 1,062 列（LightGBM の全列モデルの入力）を 16 グループに分けた（列→グループの表: <code>src/scripts/data/tp51_feature_groups.csv</code>、規則: <code>tp51_groups.py</code>）。粗い段階は valid 2022・2023、seed 1 本、学習率 3 倍・ラウンド 1/3 の設定（基準の全列モデルも同じ設定）。</p>
{groups_table(csv_path, imp)}
<h3>単体モデルの性能（valid 2022〜23 の平均 LogLoss）</h3>
{r29._tbl(["モデル", "勝率", "連対", "複勝", "列数", "学習時間"], singles, ["s", "n", "n", "n", "n", "n"]) if singles else "<p>（未実施）</p>"}
<h3>合成の効果（全列モデル比）</h3>
{gm_table(comb)}
<p class="muted">多様性: {html.escape(json.dumps(corr, ensure_ascii=False)) if corr else '（未計算）'}</p>
<h2>4. 別系統のメンバーを V4 の 7 メンバーに足す</h2>
{pool_table(stack)}
<h2>5. 推論の実用性</h2>
<ul>
 <li>LR・L3 の合成は、メンバー 7 × 3 対象の logit の線形結合（係数 21〜22 個 × 3 対象）で、推論時間・バンドルサイズへの影響はほぼ無い。</li>
 <li>メタ LightGBM・メタ NN・グルーピング型は、メンバーの数だけ推論とバンドルが増える。</li>
</ul>
<h2>6. 結論と v5 への提案</h2>
{(d / 'conclusion.html').read_text(encoding='utf-8') if (d / 'conclusion.html').exists() else '<p>（結論は結果が出そろってから記載）</p>'}
</body></html>
"""
    CHAPTER.write_text(body, encoding="utf-8")
    print("保存:", CHAPTER)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    build(Path(ap.parse_args().dir))


if __name__ == "__main__":
    main()
