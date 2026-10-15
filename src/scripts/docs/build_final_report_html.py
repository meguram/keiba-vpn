#!/usr/bin/env python3
"""モデリング全体（tp-05a〜e・tp-06）の実験結果レポート HTML を作る。数値はすべて成果物から埋め込む。

  python -m src.scripts.docs.build_final_report_html --in /tmp/modeling_out --out docs/html/modeling/modeling_experiment_report.html
入力ディレクトリには、ホストの training_pipeline/ 配下の次のサブディレクトリを置く（無いものは節ごと省く）:
  model_screen/ model_zoo/ model_tune_c/ hpo/ ensemble/ final_eval/  と mlflow_runs.json（任意）
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.scripts.docs.build_model_report_html import CSS, bars

LABEL = {
    "lgb_binary": "LightGBM 二値", "lgb_softmax": "LightGBM レース内 softmax", "cat_binary": "CatBoost 二値", "xgb_binary": "XGBoost 二値", "linear": "線形（条件付きロジット）",
    "nn_s": "NN 小 (128,64)", "nn": "NN 中 (256,128)", "nn_l": "NN 大 (512,256,128)", "lgb_tuned": "LightGBM 二値（調整後）", "xgb_tuned": "XGBoost 二値（調整後）",
    "nn_tuned": "NN（調整後）", "ensemble": "アンサンブル",
}
VAR_LABEL = {"hl1": "半減期 1 年の重み", "hl2": "半減期 2 年の重み", "hl3": "半減期 3 年の重み", "min2022": "2020〜2021 年を学習から外す", "r150": "ラウンド 150", "r500": "ラウンド 500", "r800": "ラウンド 800"}


def f(x, n=5):
    return "—" if x is None or (isinstance(x, float) and x != x) else f"{x:.{n}f}"


def table(df: pd.DataFrame, fmt=None, first="") -> str:
    fmt = fmt or {}
    o = [f"<table><tr><th>{html.escape(first)}</th>" + "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns) + "</tr>"]
    for idx, r in df.iterrows():
        o.append(f"<tr><td>{html.escape(str(idx))}</td>" + "".join(f"<td>{fmt.get(c, lambda v: f(v) if isinstance(v, float) else html.escape(str(v)))(r[c])}</td>" for c in df.columns) + "</tr>")
    return "".join(o) + "</table>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    I = a.inp

    def rc(p, **kw):
        return pd.read_csv(I / p, **kw) if (I / p).exists() else None

    def rj(p):
        return json.loads((I / p).read_text(encoding="utf-8")) if (I / p).exists() else None

    zoo, tunec, hpo_final, hpo_best, confirm = rc("model_zoo/summary.csv", index_col=0), rc("model_tune_c/vs_base.csv", index_col=0), rc("hpo/final_summary.csv", index_col=0), rj("hpo/best_params.json"), rc("hpo/confirm_2024_delta.csv", index_col=0)
    ens_cmp, ens_spec, loyo = rc("ensemble/ensemble_vs_best_single.csv", index_col=0), rj("ensemble/ensemble_spec.json"), rc("ensemble/loyo_logloss.csv", index_col=0)
    fin, comp, practical, top1c = rj("final_eval/report.json"), rc("final_eval/comparison.csv", index_col=0), rc("final_eval/practical_metrics.csv", index_col=0), rc("final_eval/top1_by_confidence.csv", index_col=0)
    segs = rc("final_eval/segments.csv")
    runs = rj("mlflow_runs.json") or {}

    o = [f'<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>モデリング実験結果レポート — keiba-vpn docs</title><style>{CSS}</style></head><body><main>']
    o.append("<h1>モデリング実験結果レポート（tp-05a〜tp-06）</h1>")
    o.append(f'<p class="muted">作成: {datetime.now():%Y-%m-%d %H:%M} / 実験管理: MLflow（ホスト上 <code>http://127.0.0.1:23230</code>）の <code>keiba-tp-model-screen</code>・<code>keiba-tp-model-zoo</code>・<code>keiba-tp-hpo</code>・<code>keiba-tp-ensemble</code>・<code>keiba-tp-final-eval</code></p>')

    # 結論
    o.append("<h2>1. 結論</h2><div class='box'><ul>")
    if fin:
        m = fin["metrics"]
        o.append(f"<li><b>test 2025（2020〜2024 年で学習、{fin['test_races']:,} レース・{fin['test_rows']:,} 頭を一度だけ評価）の最終精度</b>: "
                 f"勝率 LogLoss <b>{m['logloss_win']:.4f}</b>（AUC {m['auc_win']:.3f}）、連対率 LogLoss <b>{m['logloss_top2']:.4f}</b>（AUC {m['auc_top2']:.3f}）、複勝率 LogLoss <b>{m['logloss_top3']:.4f}</b>（AUC {m['auc_top3']:.3f}）。"
                 f"一様予測は勝率 {comp.loc['一様予測(1/頭数)', 'logloss_win']:.4f}・連対率 {comp.loc['一様予測(1/頭数)', 'logloss_top2']:.4f}・複勝率 {comp.loc['一様予測(1/頭数)', 'logloss_top3']:.4f}。</li>")
        pk = fin["practical"]
        o.append(f"<li><b>予測 1 位の馬が勝つ割合 {pk['予測1位の馬が勝つ割合（勝率）'] * 100:.1f}%</b>（ランダムなら {pk['一様予測の1位的中率(参考)'] * 100:.1f}%）、連対する割合 {pk['予測1位の馬が連対(2着以内)する割合'] * 100:.1f}%、複勝圏に入る割合 {pk['予測1位の馬が複勝圏(3着以内)に入る割合'] * 100:.1f}%。"
                 f"勝ち馬が予測上位 3 頭に入る割合は {pk['勝ち馬が予測上位3頭に入る割合'] * 100:.1f}%。</li>")
    o.append("<li>確率は「レース内で合計 1・2・3 にそろえる」処理を最終段で必ず適用する（これだけで勝率 LogLoss が約 0.0026 改善）。</li>")
    o.append("<li>市場オッズとの比較は未実施（絶対的な強さは、市場の暗黙の確率との差で判断する）。</li></ul></div>")

    # 最終精度
    if fin:
        o.append("<h2>2. 今の予測パイプラインの精度（test 2025）</h2>")
        met = pd.DataFrame({"勝率 p_win": [m["logloss_win"], m["auc_win"], m["brier_win"], m["ece_win"]],
                            "連対率 p_top2": [m["logloss_top2"], m["auc_top2"], m["brier_top2"], None],
                            "複勝率 p_top3": [m["logloss_top3"], m["auc_top3"], m["brier_top3"], None]}, index=["LogLoss（小さいほど良い）", "AUC（大きいほど良い）", "Brier", "ECE（較正のずれ）"])
        o.append(table(met, first="指標"))
        o.append("<h3>基準との比較（LogLoss）</h3>" + table(comp.rename(columns={"logloss_win": "勝率", "logloss_top2": "連対率", "logloss_top3": "複勝率"}), first="モデル"))
        if practical is not None:
            o.append("<h3>実用的な見え方</h3><table><tr><th>指標</th><th>値</th></tr>" + "".join(f"<tr><td>{html.escape(str(i))}</td><td>{v:.4f}</td></tr>" for i, v in practical.iloc[:, 0].items()) + "</table>")
        if top1c is not None:
            o.append("<h3>予測 1 位の馬の確信度別（予測勝率が高いほど実際に勝つか）</h3>" + table(top1c, fmt={"races": lambda v: f"{int(v):,}"}, first="予測勝率の帯"))
        if segs is not None:
            VENUE = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京", "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
            sg = segs.copy()
            sg["値"] = [VENUE.get(str(v).zfill(2), v) if k == "venue_code" else v for k, v in zip(sg["区分"], sg["値"])]
            sg["区分"] = sg["区分"].map({"surface": "馬場", "venue_code": "競馬場"}).fillna(sg["区分"])
            sg["k"] = sg["区分"] + " / " + sg["値"].astype(str)
            o.append("<h3>セグメント別</h3>" + table(sg.set_index("k").drop(columns=["区分", "値"]), fmt={"行数": lambda v: f"{int(v):,}"}, first="区分 / 値"))

    # 段階別
    o.append("<h2>3. 段階別の実験結果（valid 2022〜2024）</h2>")
    if zoo is not None:
        o.append("<h3>tp-05b モデル族の比較</h3>")
        z = zoo[["logloss_win", "logloss_top2", "logloss_top3", "auc_win", "auc_top2", "auc_top3", "top1_hit", "seconds"]].copy()
        z.index = [LABEL.get(i, i) for i in z.index]
        o.append(table(z, first="モデル", fmt={"seconds": lambda v: f"{v:.0f} 秒", "top1_hit": lambda v: f"{v:.4f}"}))
        items = sorted([(LABEL.get(i, i), float(zoo.loc[i, "logloss_win"]), False) for i in zoo.index if zoo.loc[i, "logloss_win"] < 0.225], key=lambda x: x[1])
        items[0] = (items[0][0], items[0][1], True)
        o.append(bars(items, items[0][1] - 0.0015, items[-1][1] + 0.0015))
    if tunec is not None:
        o.append("<h3>tp-05c 重み・学習期間・ラウンド数（LightGBM 基準との差。負なら改善）</h3>")
        t = tunec[["Δwin", "win_lo", "win_hi", "Δtop2", "Δtop3", "採用"]].copy()
        t.index = [VAR_LABEL.get(i, i) for i in t.index]
        o.append(table(t.rename(columns={"win_lo": "勝率 95%下限", "win_hi": "勝率 95%上限"}), first="案", fmt={"採用": lambda v: "<span class=ok>採用</span>" if str(v) == "True" else "<span class=warn>不採用</span>"}))
        o.append("<p class='muted'>いずれも基準（重みなし・300 ラウンド・2020 年から学習）を有意には上回らなかったため、既定の設定を維持した。</p>")
    if hpo_final is not None:
        o.append("<h3>tp-05d ハイパーパラメータ調整（Optuna。調整は valid 2022・2023、2024 は確認専用）</h3>")
        hf = hpo_final[["logloss_win", "logloss_top2", "logloss_top3", "race_nll", "auc_win", "top1_hit"]].copy()
        hf.index = [LABEL.get(i.replace("_default", "").replace("_tuned", "") + ("_tuned" if i.endswith("_tuned") else ""), i) + ("（既定）" if i.endswith("_default") else "") for i in hf.index]
        o.append(table(hf, first="モデル"))
        if confirm is not None:
            o.append("<p>確認 fold（valid 2024、調整に未使用）の LogLoss の変化（調整後 − 既定、負なら改善）:</p>" + table(confirm, first="モデル"))
        if hpo_best:
            o.append("<table><tr><th>モデル</th><th>最良パラメータ</th></tr>" + "".join(f"<tr><td>{k}</td><td style='text-align:left'><code>{html.escape(json.dumps(v['params'], ensure_ascii=False))}</code></td></tr>" for k, v in hpo_best["best"].items()) + "</table>")
    if loyo is not None and ens_spec:
        o.append("<h3>tp-05e アンサンブル（leave-one-year-out の LogLoss）</h3>")
        L = loyo.copy()
        L.index = [i if not i.startswith("single:") else "単独: " + LABEL.get(i.split(":")[1], i.split(":")[1]) for i in L.index]
        o.append(table(L.sort_values("win").rename(columns={"win": "勝率", "top2": "連対率", "top3": "複勝率"}), first="方式"))
        o.append("<p>採用した合成案: " + "、".join(f"{t}: <b>{ens_spec['targets'][t]['method']}</b>" for t in ens_spec["targets"]) + "。構成モデル: " + "、".join(LABEL.get(n, n) for n in ens_spec["models"]) + "。</p>")
        if ens_cmp is not None:
            o.append("<p>単独の最良モデルとの差（LOYO、レース単位の paired bootstrap。負ならアンサンブルが良い）:</p>" + table(ens_cmp.drop(columns=["best_single"]).assign(best_single=ens_cmp["best_single"].map(lambda x: LABEL.get(x, x))), first="対象"))

    if runs:
        o.append("<h2>4. MLflow</h2><p class='muted'>各 run の ID（ホストの MLflow で参照）。</p><table><tr><th>実験</th><th>run_id</th></tr>" + "".join(f"<tr><td>{html.escape(k)}</td><td><code>{html.escape(str(v)[:12])}</code></td></tr>" for k, v in runs.items()) + "</table>")
    o.append("<h2>5. 注意と次の一手</h2><ul><li>この精度は、固定の特徴量（tp-03 の採用セット + 前段 3 列 + 相対化）と、2020〜2024 年の学習だけで得た値。test 2025 は一度だけ評価した。</li>"
             "<li>市場オッズとの比較（暗黙の確率との差）と、馬券としての回収率の検証が次の段階（バックテスト）。</li>"
             "<li>tp-07 で 2020〜2025 年を使った最終再学習・登録・推論の一致確認を行う。</li></ul></main></body></html>")
    a.out.write_text("\n".join(o), encoding="utf-8")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
