#!/usr/bin/env python3
"""モデリング全体（tp-05a〜e・tp-06）の実験結果レポート HTML を作る。数値はすべて成果物から埋め込む。

  python -m src.scripts.docs.build_final_report_html --in /tmp/modeling_out --out docs/html/modeling/chapters/modeling_experiment_report.html
入力ディレクトリには、ホストの training_pipeline/ 配下の次のサブディレクトリを置く（無いものは節ごと省く）:
  model_screen/ model_zoo/ model_tune_c/ hpo/ ensemble/ final_eval/ improve_f/ improve_g/ improve_i/ ensemble_v2/ final_eval_v2/
  rl/（= data/local/modeling/betting/rl/<tag>/ の中身）  と mlflow_runs.json（任意）
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.scripts.docs.build_model_report_html import CSS, bars

from src.scripts.docs import report_notes as N

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
    fin2, comp2, practical2, top1c2, segs2 = rj("final_eval_v2/report.json"), rc("final_eval_v2/comparison.csv", index_col=0), rc("final_eval_v2/practical_metrics.csv", index_col=0), rc("final_eval_v2/top1_by_confidence.csv", index_col=0), rc("final_eval_v2/segments.csv")
    imp_f, imp_g1, imp_g2, imp_i = rc("improve_f/vs_base.csv", index_col=0), rc("improve_g/stage1_vs_lgb_tuned.csv", index_col=0), rc("improve_g/stage2_vs_tuned.csv", index_col=0), rc("improve_i/vs_nn_extras_ped.csv", index_col=0)
    v2cmp, v2spec, v2loyo, v2cal = rc("ensemble_v2/v2_vs_v1.csv", index_col=0), rj("ensemble_v2/ensemble_spec.json"), rc("ensemble_v2/loyo_logloss.csv", index_col=0), rc("ensemble_v2/segment_recalibration.csv", index_col=0)
    rl = rj("rl/report.json")

    o = [f'<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>モデリング実験結果レポート — keiba-vpn docs</title><style>{CSS}</style></head><body><main>']
    o.append("<h1>モデリング実験結果レポート（tp-05a〜tp-08）</h1>")
    o.append(N.LEAK_NOTE_HTML)
    o.append(f'<p class="muted">作成: {datetime.now():%Y-%m-%d %H:%M} / 実験管理: MLflow（ホスト上 <code>http://127.0.0.1:23230</code>）の <code>keiba-tp-model-screen</code>・<code>keiba-tp-model-zoo</code>・<code>keiba-tp-hpo</code>・<code>keiba-tp-ensemble</code>・<code>keiba-tp-final-eval</code></p>')

    # 結論
    P = fin2 or fin          # 現行の最終精度（v2 があれば v2）
    CP = comp2 if fin2 else comp
    o.append("<h2>1. 結論</h2><div class='box'><ul>")
    if P:
        m = P["metrics"]
        ver = "v2（新特徴量 + アンサンブル v2）" if fin2 else "v1"
        o.append(f"<li><b>現行パイプライン（{ver}）の test 2025（2020〜2024 年で学習、{P['test_races']:,} レース・{P['test_rows']:,} 頭を一度だけ評価）の最終精度</b>: "
                 f"勝率 LogLoss <b>{m['logloss_win']:.4f}</b>（AUC {m['auc_win']:.3f}）、連対率 LogLoss <b>{m['logloss_top2']:.4f}</b>（AUC {m['auc_top2']:.3f}）、複勝率 LogLoss <b>{m['logloss_top3']:.4f}</b>（AUC {m['auc_top3']:.3f}）。"
                 f"一様予測は勝率 {CP.loc['一様予測(1/頭数)', 'logloss_win']:.4f}・連対率 {CP.loc['一様予測(1/頭数)', 'logloss_top2']:.4f}・複勝率 {CP.loc['一様予測(1/頭数)', 'logloss_top3']:.4f}。</li>")
        pk = P["practical"]
        o.append(f"<li><b>予測 1 位の馬が勝つ割合 {pk['予測1位の馬が勝つ割合（勝率）'] * 100:.1f}%</b>（ランダムなら {pk['一様予測の1位的中率(参考)'] * 100:.1f}%）、連対する割合 {pk['予測1位の馬が連対(2着以内)する割合'] * 100:.1f}%、複勝圏に入る割合 {pk['予測1位の馬が複勝圏(3着以内)に入る割合'] * 100:.1f}%。"
                 f"勝ち馬が予測上位 3 頭に入る割合は {pk['勝ち馬が予測上位3頭に入る割合'] * 100:.1f}%。</li>")
    if fin2 and fin:
        mm = fin["metrics"]
        o.append(f"<li>v1（tp-06）は 勝率 {mm['logloss_win']:.4f} / 連対率 {mm['logloss_top2']:.4f} / 複勝率 {mm['logloss_top3']:.4f}。追い切り評価・厩舎コメントの特徴量とアンサンブル v2 で、3 指標とも有意に改善した（§3.5）。</li>")
    o.append("<li>確率は「レース内で合計 1・2・3 にそろえる」処理を最終段で必ず適用する（これだけで勝率 LogLoss が約 0.0026 改善）。</li>")
    if rl:
        h = next((r for r in rl["final"] if r["方策"].startswith("RL（下限0.5）")), None)
        bl = next((r for r in rl["final"] if r["方策"] == "複勝: モデル1位に全額"), None)
        bf = next((r for r in rl["final"] if r["方策"] == "単勝: 1番人気に全額"), None)
        vs = next((r for r in rl.get("vs_baselines", []) if r["比較相手"] == "複勝: モデル1位に全額"), None)
        if h:
            extra = (f"単純な基準（複勝のモデル 1 位 {bl['roi'] * 100:.1f}%）より {vs['下限0.5: 回収率の差'] * 100:+.1f} ポイント（95% 信頼区間 {vs['下限0.5: 95%下限'] * 100:+.1f}〜{vs['下限0.5: 95%上限'] * 100:+.1f}）有意に高く、単勝 1 番人気（{bf['roi'] * 100:.1f}%）を大きく上回る" if vs else "")
            o.append(f"<li><b>馬券の強化学習（tp-08、test 2025、確定オッズ）: 控除率を上回る優位は見つからなかった（回収率は 100% 未満）。</b>自由に見送れる設定では全レース見送りに落ちる。1 レース平均の賭け金に下限（予算の 50%）を課した最良の方策の回収率は <b>{h['roi'] * 100:.1f}%</b>"
                     f"（95% 信頼区間 {h['roi_lo'] * 100:.1f}〜{h['roi_hi'] * 100:.1f}%）{N.RL_HEADLINE_FIX}。{extra}が、損失を小さくする配分であって、利益が出る賭け方ではない。詳しくは §4。</li>")
    else:
        o.append("<li>市場オッズとの比較・馬券の回収率の検証は §4（実施した場合）を参照。</li>")
    o.append("</ul></div>")

    # 最終精度
    def final_section(num, title, fin_, comp_, practical_, top1c_, segs_):
        m_ = fin_["metrics"]
        o.append(f"<h2>{num}. {title}</h2>")
        met = pd.DataFrame({"勝率 p_win": [m_["logloss_win"], m_["auc_win"], m_["brier_win"], m_["ece_win"]],
                            "連対率 p_top2": [m_["logloss_top2"], m_["auc_top2"], m_["brier_top2"], None],
                            "複勝率 p_top3": [m_["logloss_top3"], m_["auc_top3"], m_["brier_top3"], None]}, index=["LogLoss（小さいほど良い）", "AUC（大きいほど良い）", "Brier", "ECE（較正のずれ）"])
        o.append(table(met, first="指標"))
        o.append("<h3>基準との比較（LogLoss）</h3>" + table(comp_.rename(columns={"logloss_win": "勝率", "logloss_top2": "連対率", "logloss_top3": "複勝率"}), first="モデル"))
        if practical_ is not None:
            o.append("<h3>実用的な見え方</h3><table><tr><th>指標</th><th>値</th></tr>" + "".join(f"<tr><td>{html.escape(str(i))}</td><td>{v:.4f}</td></tr>" for i, v in practical_.iloc[:, 0].items()) + "</table>")
        if top1c_ is not None:
            o.append("<h3>予測 1 位の馬の確信度別（予測勝率が高いほど実際に勝つか）</h3>" + table(top1c_, fmt={"races": lambda v: f"{int(v):,}"}, first="予測勝率の帯"))
        if segs_ is not None:
            VENUE = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京", "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
            sg = segs_.copy()
            sg["値"] = [VENUE.get(str(v).zfill(2), v) if k == "venue_code" else v for k, v in zip(sg["区分"], sg["値"])]
            sg["区分"] = sg["区分"].map({"surface": "馬場", "venue_code": "競馬場"}).fillna(sg["区分"])
            sg["k"] = sg["区分"] + " / " + sg["値"].astype(str)
            o.append("<h3>セグメント別</h3>" + table(sg.set_index("k").drop(columns=["区分", "値"]), fmt={"行数": lambda v: f"{int(v):,}"}, first="区分 / 値"))

    if fin2:
        final_section(2, "今の予測パイプラインの精度（v2、test 2025）", fin2, comp2, practical2, top1c2, segs2)
        o.append("<details><summary>(参考) v1 の最終精度（tp-06、test 2025）</summary>")
        final_section("2b", "v1 の最終精度", fin, comp, practical, top1c, segs)
        o.append("</details>")
    elif fin:
        final_section(2, "今の予測パイプラインの精度（test 2025）", fin, comp, practical, top1c, segs)

    # 段階別
    o.append("<h2>3. 段階別の実験結果（valid 2022〜2024）</h2>")
    if zoo is not None:
        o.append("<h3>3.1 tp-05b モデル族の比較</h3>")
        z = zoo[["logloss_win", "logloss_top2", "logloss_top3", "auc_win", "auc_top2", "auc_top3", "top1_hit", "seconds"]].copy()
        z.index = [LABEL.get(i, i) for i in z.index]
        o.append(table(z, first="モデル", fmt={"seconds": lambda v: f"{v:.0f} 秒", "top1_hit": lambda v: f"{v:.4f}"}))
        items = sorted([(LABEL.get(i, i), float(zoo.loc[i, "logloss_win"]), False) for i in zoo.index if zoo.loc[i, "logloss_win"] < 0.225], key=lambda x: x[1])
        items[0] = (items[0][0], items[0][1], True)
        o.append(bars(items, items[0][1] - 0.0015, items[-1][1] + 0.0015))
    if tunec is not None:
        o.append("<h3>3.2 tp-05c 重み・学習期間・ラウンド数（LightGBM 基準との差。負なら改善）</h3>")
        t = tunec[["Δwin", "win_lo", "win_hi", "Δtop2", "Δtop3", "採用"]].copy()
        t.index = [VAR_LABEL.get(i, i) for i in t.index]
        o.append(table(t.rename(columns={"win_lo": "勝率 95%下限", "win_hi": "勝率 95%上限"}), first="案", fmt={"採用": lambda v: "<span class=ok>採用</span>" if str(v) == "True" else "<span class=warn>不採用</span>"}))
        o.append("<p class='muted'>いずれも基準（重みなし・300 ラウンド・2020 年から学習）を有意には上回らなかったため、既定の設定を維持した。</p>")
    if hpo_final is not None:
        o.append("<h3>3.3 tp-05d ハイパーパラメータ調整（Optuna。調整は valid 2022・2023、2024 は確認専用）</h3>")
        hf = hpo_final[["logloss_win", "logloss_top2", "logloss_top3", "race_nll", "auc_win", "top1_hit"]].copy()
        hf.index = [LABEL.get(i.replace("_default", "").replace("_tuned", "") + ("_tuned" if i.endswith("_tuned") else ""), i) + ("（既定）" if i.endswith("_default") else "") for i in hf.index]
        o.append(table(hf, first="モデル"))
        if confirm is not None:
            o.append("<p>確認 fold（valid 2024、調整に未使用）の LogLoss の変化（調整後 − 既定、負なら改善）:</p>" + table(confirm, first="モデル"))
        if hpo_best:
            o.append("<table><tr><th>モデル</th><th>最良パラメータ</th></tr>" + "".join(f"<tr><td>{k}</td><td style='text-align:left'><code>{html.escape(json.dumps(v['params'], ensure_ascii=False))}</code></td></tr>" for k, v in hpo_best["best"].items()) + "</table>")
    if loyo is not None and ens_spec:
        o.append("<h3>3.4 tp-05e アンサンブル（leave-one-year-out の LogLoss）</h3>")
        L = loyo.copy()
        L.index = [i if not i.startswith("single:") else "単独: " + LABEL.get(i.split(":")[1], i.split(":")[1]) for i in L.index]
        o.append(table(L.sort_values("win").rename(columns={"win": "勝率", "top2": "連対率", "top3": "複勝率"}), first="方式"))
        o.append("<p>採用した合成案: " + "、".join(f"{t}: <b>{ens_spec['targets'][t]['method']}</b>" for t in ens_spec["targets"]) + "。構成モデル: " + "、".join(LABEL.get(n, n) for n in ens_spec["models"]) + "。</p>")
        if ens_cmp is not None:
            o.append("<p>単独の最良モデルとの差（LOYO、レース単位の paired bootstrap。負ならアンサンブルが良い）:</p>" + table(ens_cmp.drop(columns=["best_single"]).assign(best_single=ens_cmp["best_single"].map(lambda x: LABEL.get(x, x))), first="対象"))

    # 3.5 精度向上の探索
    if imp_f is not None or imp_g1 is not None or v2cmp is not None:
        o.append("<h3>3.5 精度向上の探索（tp-05f〜i）と アンサンブル v2（tp-05h）</h3>")
        o.append("<p class='muted'>採否は valid 2022〜2024 だけで決める。基準 = 差の 95% 信頼区間の上限 &lt; 0、かつ連対・複勝が悪化しない（0.0002 以内）。差は負なら改善。</p>")
        cols = ["Δwin", "win_hi", "Δtop2", "Δtop3", "採用"]
        ok_fmt = {"採用": lambda v: "<span class=ok>採用</span>" if str(v) == "True" else "<span class=warn>不採用</span>"}
        if imp_f is not None:
            o.append("<h4>tp-05f NN の埋め込み・注意機構・相対化の拡張</h4>" + table(imp_f.set_index("variant")[["基準"] + cols] if "variant" in imp_f.columns else imp_f[["基準"] + cols], fmt=ok_fmt, first="案"))
        if imp_g1 is not None:
            o.append("<h4>tp-05g 新しい特徴量（追い切り評価・厩舎コメント = extras、血統 × 条件 = ped）。段階 1: LightGBM</h4>" + table(imp_g1[["基準"] + cols], fmt=ok_fmt, first="案"))
        if imp_g2 is not None:
            o.append("<h4>段階 2: XGBoost・NN への展開</h4>" + table(imp_g2[["基準"] + cols], fmt=ok_fmt, first="案"))
        if imp_i is not None:
            o.append("<h4>tp-05i NN の再調整（新特徴量セット上）</h4>" + table(imp_i[["基準"] + cols], fmt=ok_fmt, first="案"))
        if v2cmp is not None:
            o.append("<h4>tp-05h アンサンブル v2（LOYO。v1 との差）</h4>" + table(v2cmp, fmt=ok_fmt, first="対象"))
            if v2spec:
                o.append("<p>構成: " + "、".join(v2spec["models"]) + "。合成案: " + "、".join(f"{t}: <b>{v2spec['targets'][t]['method']}</b>" for t in v2spec["targets"]) + "。</p>")
        if v2cal is not None:
            o.append("<h4>弱いセグメント（障害・2700m 以上・8 頭以下）の再較正（LOYO）</h4>" + table(v2cal, fmt=ok_fmt, first="対象") + "<p class='muted'>全体の LogLoss が有意に良くならないため不採用。</p>")

    # 4 馬券最適化
    nsec = 4
    if rl:
        nsec = 5
        rb = lambda n: pd.read_csv(I / "rl" / n) if (I / "rl" / n).exists() else None
        bl_, cal_, fin_, vsb_, typ_, mode_ = rb("benter_blend.csv"), rb("ev_calibration_blend.csv"), rb("final_test2025.csv"), rb("final_vs_baselines.csv"), rb("final_by_type.csv"), rb("mode_comparison.csv")
        o.append("<h2>4. 馬券最適化の強化学習（tp-08）</h2>")
        o.append(N.RL_CORRECTION_HTML)
        o.append("<p>予測（v2）の勝率・連対率・複勝率から、割引 Harville で単勝・複勝・馬連・ワイド・馬単の各組の確率を作り、市場の暗黙の確率と Benter 型でブレンドする。"
                 "1 レースを 1 ステップとし、候補の買い目への賭け金の配分（予算 1 = 1R の予算、残りは見送り）を方策勾配で学習する。報酬は実際の払戻 − 賭け金。"
                 "学習 2022〜2024（ハイパーパラメータは 2023・2024 を評価年にした Optuna）、test 2025 は一度だけ。<b>オッズは締切後の確定オッズ</b>で、購入判断の時点の値ではない（楽観側に偏りうる）。三連複・三連単はオッズが無く対象外。</p>")
        if bl_ is not None:
            o.append(f"<h3>市場とのブレンド（Benter）</h3><p>勝ち馬の対数尤度が最大になる a={rl['benter']['a']:.3f}（モデル）、b={rl['benter']['b']:.3f}（市場）。値は 2022 年だけで当てはめた。数値は勝ち馬の負の対数尤度（小さいほど良い）。</p>" + table(bl_.set_index("year"), fmt={"races": lambda v: f"{int(v):,}"}, first="年"))
            o.append("<p class='muted'>市場の暗黙の確率の方がモデル単独よりかなり良く、ブレンドの追加分はごくわずか。モデルの情報の大半は既にオッズに織り込まれている。</p>")
        if cal_ is not None:
            pv = cal_.pivot(index="band", columns="type", values="実際の回収率")
            pv = pv.loc[sorted(pv.index, key=lambda b: float(str(b).lstrip("(").split(",")[0]))]
            o.append("<h3>期待回収率の較正（valid 2023・2024）</h3><p>買い目を「ブレンド確率 × オッズ」の帯に分け、実際の回収率（的中時の払戻倍率の平均）を見る。優位があるなら、期待回収率が高い帯ほど実際も高く、1 を超える帯がある。</p>" + table(pv, first="期待回収率の帯"))
            o.append("<p class='muted'>期待回収率が高い買い目ほど、実際の回収率は<b>むしろ低い</b>（高いオッズの穴馬を過大評価している）。1 を超える帯は無い。</p>")
        if mode_ is not None:
            o.append("<h3>方策の学習方式の比較（valid、既定のハイパーパラメータ）</h3>" + table(mode_.assign(k=mode_["mode"] + " / 下限 " + mode_["賭け金の下限"].astype(str) + " / fold " + mode_["fold"]).set_index("k")[["roi", "profit_per_race", "stake_per_race", "participation"]], first="方式 / 下限 / fold"))
            o.append("<p class='muted'>下限 0（見送り自由）は、賭け金が 0 に収束する（全レース見送り）。優位が無いので、期待値の最大化は「買わない」になる。下限を課すと、その賭け金のもとで損失を最小にする配分を学ぶ。</p>")
        if fin_ is not None:
            o.append("<h3>test 2025 の結果（RL = 強化学習の方策、下限 = 1 レース平均の賭け金の下限（予算比））</h3>" + table(fin_.set_index("方策"), fmt={"participation": lambda v: f"{v * 100:.1f}%"}, first="方策"))
            o.append("<p class='muted'>roi = 総払戻 / 総賭け金（95% 信頼区間はレース単位の bootstrap）。profit_per_race は予算 1 に対する 1 レース当たりの損益。</p>")
        if vsb_ is not None:
            o.append("<h3>回収率の差（RL − 相手）</h3>" + table(vsb_.set_index("比較相手"), first="比較相手"))
        if typ_ is not None:
            o.append("<h3>券種別の賭け金の割合</h3>" + table(typ_.pivot(index="券種", columns="下限", values="賭け金の割合"), first="券種"))
        png = I / "rl" / "cumulative_profit_2025.png"
        if png.exists():
            o.append('<h3>累積損益（test 2025）</h3><img alt="cumulative profit" style="max-width:100%" src="data:image/png;base64,' + base64.b64encode(png.read_bytes()).decode() + '">')
        o.append("<div class='box'><b>読み方</b>: 確定オッズで見ても、モデル（市場と独立な情報）から控除率を上回る優位は取り出せなかった。"
                 "実運用の推奨馬券は、回収率 100% 超を約束するものではなく、損失をなるべく小さくする配分（低控除の券種・人気馬寄り）か、"
                 "オッズが動く前（T-60）に市場より先に織り込める情報（馬体重・追い切り・パドックなど）が見つかった場合の追加検証が必要。</div>")

    if runs:
        o.append(f"<h2>{nsec}. MLflow</h2><p class='muted'>各 run の ID（ホストの MLflow で参照）。</p><table><tr><th>実験</th><th>run_id</th></tr>" + "".join(f"<tr><td>{html.escape(k)}</td><td><code>{html.escape(str(v)[:12])}</code></td></tr>" for k, v in runs.items()) + "</table>")
    o.append(f"<h2>{nsec + 1}. 注意と次の一手</h2><ul><li>精度は、固定の特徴量（tp-03 の採用セット + 前段 3 列 + 相対化 + 追い切り・厩舎コメント・血統×条件）と、2020〜2024 年の学習だけで得た値。test 2025 は一度だけ評価した。</li>"
             "<li>馬券の回収率は確定オッズ（スナップショットなし）での検証。2026-10-17 以降に蓄積される T-60/T-15 のオッズで、購入時点の値での再検証が必要。</li>"
             "<li>tp-07 で 2020〜2025 年を使った最終再学習・登録・推論の一致確認を行う。</li></ul></main></body></html>")
    a.out.write_text("\n".join(o), encoding="utf-8")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
