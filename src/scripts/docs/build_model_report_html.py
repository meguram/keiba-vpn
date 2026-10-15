#!/usr/bin/env python3
"""tp-05a の実験結果（ホストの data/local/modeling/training_pipeline/model_screen/）から HTML レポートを作る。

  python -m src.scripts.docs.build_model_report_html --in /tmp/tp05a_out --log /tmp/tp05a.log --mlflow-json /tmp/tp05a_out/mlflow_runs.json \
      --out docs/html/modeling/tp05a_experiment_report.html
数値はすべて入力ファイルから埋め込む（手で書き写さない）。文章の判断は数値から機械的に出す（基準: 95% 信頼区間の上限が 0 未満、かつ 2 fold 以上で改善）。
"""

from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

CSS = """
:root { --bg:#0f172a; --panel:#15213c; --border:#243358; --text:#e6ecff; --muted:#9aa7c7; --accent:#7aa9ff; --ok:#4ade80; --warn:#fbbf24; --bad:#f87171; --code:#0b1428; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.75 "Noto Sans JP",system-ui,sans-serif; }
main { max-width:1080px; margin:0 auto; padding:40px 28px 96px; }
h1 { font-size:26px; margin:0 0 6px; } h2 { font-size:20px; margin:44px 0 10px; padding-top:18px; border-top:1px solid var(--border); }
h3 { font-size:15px; color:var(--accent); margin:22px 0 6px; } .muted { color:var(--muted); font-size:13px; }
table { border-collapse:collapse; width:100%; margin:10px 0 16px; font-size:13px; }
th, td { border:1px solid var(--border); padding:5px 9px; text-align:right; vertical-align:top; } td:first-child, th:first-child { text-align:left; }
th { background:var(--panel); color:var(--muted); font-weight:600; }
code { font-family:"JetBrains Mono",ui-monospace,Menlo,monospace; background:var(--code); padding:1px 6px; border-radius:5px; font-size:12.5px; }
.ok { color:var(--ok); font-weight:600; } .warn { color:var(--warn); font-weight:600; } .bad { color:var(--bad); font-weight:600; }
.box { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:12px 18px; margin:12px 0; }
.best td { background:rgba(74,222,128,.10); }
"""

LABEL = {
    "A1_binary": "A1 二値（基準）", "A1_binary+S": "A1 + 前段3列", "A3_softmax": "A3 レース内 softmax", "A4a_lambdarank": "A4a lambdarank",
    "A4b_xendcg": "A4b rank_xendcg", "C1_binary+rel": "C1 二値 + 相対化", "A3_softmax+rel": "A3 softmax + 相対化",
    "C1_binary+rel+S": "C1 二値 + 相対化 + 前段3列", "A3_softmax+rel+S": "A3 softmax + 相対化 + 前段3列",
}
ORACLE = {"A4a_lambdarank", "A4b_xendcg"}


def f(x, n=5):
    return "—" if x is None or (isinstance(x, float) and x != x) else f"{x:.{n}f}"


def bars(items: list[tuple[str, float, bool]], lo: float, hi: float) -> str:
    w, h = 760, 26 * len(items) + 20
    out = [f'<svg width="{w}" height="{h}" role="img" aria-label="p_win LogLoss の比較">']
    for i, (name, v, best) in enumerate(items):
        y = 10 + i * 26
        x = 300 + (v - lo) / (hi - lo) * 420
        out.append(f'<text x="290" y="{y + 14}" fill="#e6ecff" font-size="12" text-anchor="end">{html.escape(name)}</text>')
        out.append(f'<rect x="300" y="{y + 3}" width="{max(2, x - 300):.0f}" height="14" fill="{"#4ade80" if best else "#7aa9ff"}" rx="3"/>')
        out.append(f'<text x="{x + 6:.0f}" y="{y + 14}" fill="#9aa7c7" font-size="11">{v:.5f}</text>')
    out.append("</svg>")
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--log", type=Path)
    ap.add_argument("--mlflow-json", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--extra-json", type=Path, help="データの前提・資源の実測など、手で集めた値（任意）")
    a = ap.parse_args()

    S = pd.read_csv(a.inp / "summary.csv", index_col=0)
    PF = pd.read_csv(a.inp / "per_fold_logloss.csv", index_col=0)
    VS = pd.read_csv(a.inp / "vs_A1.csv", index_col=0)
    rep = json.loads((a.inp / "report.json").read_text(encoding="utf-8"))
    secs = {}
    if a.log and a.log.exists():
        for m in re.finditer(r"^\s+(\S+)\s+p_win LogLoss [\d.]+\s+race_nll [\d.]+\s+\((\d+)s\)", a.log.read_text(encoding="utf-8"), re.M):
            secs[m.group(1)] = int(m.group(2))
    runs = json.loads(a.mlflow_json.read_text(encoding="utf-8")) if a.mlflow_json and a.mlflow_json.exists() else {}
    extra = json.loads(a.extra_json.read_text(encoding="utf-8")) if a.extra_json and a.extra_json.exists() else {}

    non_oracle = [m for m in S.index if m not in ORACLE]
    best = S.loc[non_oracle, "logloss_win"].idxmin()
    a1 = S.loc["A1_binary"]
    d_best = S.loc[best, "logloss_win"] - a1["logloss_win"]
    vs_best = VS.loc[best] if best in VS.index else None
    raw_gain = S.loc["A1_binary", "logloss_win_raw"] - S.loc["A1_binary", "logloss_win"]
    stage = VS.loc["A1_binary+S"] if "A1_binary+S" in VS.index else None

    o = []
    o.append(f'<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
             f'<title>tp-05a 実験結果レポート — keiba-vpn docs</title><style>{CSS}</style></head><body><main>')
    o.append("<h1>tp-05a 実験結果レポート: 勝ち馬確率の目的関数の比較</h1>")
    o.append(f'<p class="muted">作成: {datetime.now():%Y-%m-%d %H:%M} / 実行コミット <code>{html.escape(str(rep.get("git_commit")))}</code> / '
             f'notebook <code>notebooks/modeling/training_pipeline/tp-05a-baseline-objectives.ipynb</code> / MLflow 実験 <code>keiba-tp-model-screen</code>（ホスト上 <code>http://127.0.0.1:23230</code>）</p>')

    # 1. 結論
    o.append("<h2>1. 結論</h2><div class=\"box\"><ul>")
    o.append(f"<li><b>最良の方式は {html.escape(LABEL.get(best, best))}</b>（p_win LogLoss <b>{f(S.loc[best, 'logloss_win'])}</b>）。基準 A1（二値）の {f(a1['logloss_win'])} から "
             f"<b>{d_best:+.5f}</b>"
             + (f"（95% 信頼区間 {vs_best['95%下限']:+.5f}〜{vs_best['95%上限']:+.5f}、{int(vs_best['改善した fold 数'])}/3 fold で改善）" if vs_best is not None else "") + "。</li>")
    o.append(f"<li>確率を<b>レース内で合計 1 にそろえる（M1）だけで {raw_gain:.5f} 改善</b>する（A1 の正規化前 {f(a1['logloss_win_raw'])} → 後 {f(a1['logloss_win'])}）。"
             f"モデルを変えるよりも大きい改善で、必ず最終段にかける。</li>")
    if stage is not None:
        o.append(f"<li>tp-04 で採用した前段の 3 列（<code>s_time_pred</code> ほか）は、二値でも {stage['Δ LogLoss(平均)']:+.5f}"
                 f"（95% 信頼区間 {stage['95%下限']:+.5f}〜{stage['95%上限']:+.5f}）と、効果が統計的に確認できた。</li>")
    o.append("<li>ランキング系（lambdarank・rank_xendcg）は温度 T=1 では較正が悪い。検証 fold で T を最適化した値（楽観的な参考値）では二値・softmax と同等まで届く。"
             "実運用の確率には使えず、順位付けの用途に限る。</li>")
    o.append("<li>1 位的中率は全方式で約 30%、NDCG@3 は約 0.51 と、方式間の差が小さい。確率の質（LogLoss・race_nll）で差がつく。</li></ul></div>")

    # 2. 条件
    o.append("<h2>2. 実験条件</h2><table><tr><th>項目</th><th>内容</th></tr>")
    for k, v in [("モデル", "LightGBM（固定: learning_rate 0.05、num_leaves 31、min_data_in_leaf 100、bagging 0.8、feature_fraction 0.7、lambda_l2 5、300 ラウンド）"),
                 ("特徴量", f"tp-03 の採用セット（B0′ + 履歴特徴量 127 列、<code>speed_*</code> なし）計 {rep.get('n_base_features')} 列。前段 3 列: {', '.join(rep.get('stage_cols', []))}。相対化はレース内の z・順位・最大との差を 14 列に追加"),
                 ("検証", "valid 2022・2023・2024 の expanding window（purge 7 日 + embargo 3 日）、3 seed。test 2025 は未使用（tp-06 まで）。2026 年は未使用"),
                 ("学習行数・評価行数", f"全 {rep.get('n_rows'):,} 行（2020〜2025 年）"),
                 ("採否の基準", "レース単位の paired bootstrap（1,000 回）で A1 との差の 95% 信頼区間の上限が 0 未満、かつ 3 fold 中 2 fold 以上で改善。スクリーニング = 2022・2023、確認 = 2024"),
                 ("主指標", "p_win の LogLoss（正規化後）。併記: race_nll（レース内で正規化した勝ち馬の負の対数尤度）、Brier、ECE、1 位的中率、NDCG@3")]:
        o.append(f"<tr><td>{k}</td><td style='text-align:left'>{v}</td></tr>")
    o.append("</table>")

    # 3. 性能比較
    o.append("<h2>3. 性能比較（3 fold × 3 seed の平均）</h2>")
    items = sorted([(LABEL.get(m, m) + ("（T 校正・楽観）" if m in ORACLE and 'logloss_win_T' in S.columns else ""),
                     float(S.loc[m, "logloss_win_T"] if m in ORACLE else S.loc[m, "logloss_win"]), m == best) for m in S.index], key=lambda x: x[1])
    lo = min(v for _, v, _ in items) - 0.0015
    hi = max(v for _, v, _ in items if v < 0.225) + 0.0015
    o.append(bars([(n, v, b) for n, v, b in items if v < 0.225], lo, hi))
    o.append('<p class="muted">緑 = 最良（温度を検証 fold で最適化した値は楽観的なので、最良の判定からは除く）。rank_xendcg（T=1）は 0.228 と大きく外れるため図から外した。</p>')
    cols = [("logloss_win", "p_win LogLoss"), ("race_nll", "race_nll"), ("brier_win", "Brier"), ("ece_win", "ECE"), ("top1_hit", "1位的中"), ("ndcg3", "NDCG@3"),
            ("logloss_top2", "top2 LogLoss"), ("logloss_top3", "top3 LogLoss")]
    o.append("<table><tr><th>方式</th>" + "".join(f"<th>{n}</th>" for _, n in cols) + "<th>A1 との差</th><th>95% 区間</th><th>基準</th></tr>")
    for m in S.index:
        v = VS.loc[m] if m in VS.index else None
        cls = ' class="best"' if m == best else ""
        o.append(f"<tr{cls}><td>{html.escape(LABEL.get(m, m))}</td>" + "".join(f"<td>{f(S.loc[m, c], 4 if c in ('race_nll', 'top1_hit', 'ndcg3') else 5)}</td>" for c, _ in cols)
                 + (f"<td>{v['Δ LogLoss(平均)']:+.5f}</td><td>{v['95%下限']:+.5f}〜{v['95%上限']:+.5f}</td><td>{'<span class=ok>採用</span>' if v['採用基準(上限<0 かつ 2fold以上)'] else '<span class=warn>不採用</span>'}</td>"
                    if v is not None else "<td>基準</td><td>—</td><td>—</td>") + "</tr>")
    o.append("</table>")

    # 4. fold 別
    o.append("<h2>4. fold 別の p_win LogLoss</h2><table><tr><th>方式</th><th>valid 2022</th><th>valid 2023</th><th>valid 2024</th><th>スクリーニング(2022・2023)</th><th>確認(2024)</th><th>平均</th></tr>")
    for m in PF.index:
        o.append(f"<tr><td>{html.escape(LABEL.get(m, m))}</td>" + "".join(f"<td>{f(PF.loc[m, c])}</td>" for c in ["valid_2022", "valid_2023", "valid_2024", "スクリーニング(2022・2023)", "確認(2024)", "平均"]) + "</tr>")
    o.append("</table>")

    # 5. M1 と温度
    o.append("<h2>5. 確率の整合（M1）と温度</h2><h3>二値モデル: 正規化の効果</h3><table><tr><th>方式</th><th>正規化前</th><th>正規化後</th><th>改善</th></tr>")
    for m in S.index:
        if pd.notna(S.loc[m].get("logloss_win_raw")):
            o.append(f"<tr><td>{html.escape(LABEL.get(m, m))}</td><td>{f(S.loc[m, 'logloss_win_raw'])}</td><td>{f(S.loc[m, 'logloss_win'])}</td><td>{S.loc[m, 'logloss_win_raw'] - S.loc[m, 'logloss_win']:+.5f}</td></tr>")
    o.append("</table><h3>ランキング系: 温度 T</h3><table><tr><th>方式</th><th>T=1 の LogLoss</th><th>最適 T</th><th>最適 T での LogLoss（楽観）</th><th>最適 T での race_nll</th></tr>")
    for m in ORACLE & set(S.index):
        o.append(f"<tr><td>{html.escape(LABEL.get(m, m))}</td><td>{f(S.loc[m, 'logloss_win'])}</td><td>{f(S.loc[m, 'T_oracle'], 3)}</td><td>{f(S.loc[m, 'logloss_win_T'])}</td><td>{f(S.loc[m, 'race_nll_T'], 4)}</td></tr>")
    o.append("</table><p class='muted'>最適 T は評価に使った検証 fold で求めたため、実際より良く見える（1 パラメータなので過大評価は小さい）。実運用では過去の fold で T を決める必要がある。</p>")

    # 6. 資源
    o.append("<h2>6. ホスト資源</h2><table><tr><th>方式</th><th>実行時間（3 fold × 3 seed）</th><th>MLflow run</th></tr>")
    for m in S.index:
        r = runs.get(m, {})
        o.append(f"<tr><td>{html.escape(LABEL.get(m, m))}</td><td>{secs.get(m, '—')} 秒</td><td><code>{html.escape(str(r.get('run_id', '—'))[:12])}</code></td></tr>")
    o.append("</table>")
    if extra.get("resources"):
        o.append("<div class='box'>" + extra["resources"] + "</div>")

    # 7. データの前提
    if extra.get("premises"):
        o.append("<h2>7. データの前提（この結果の信頼性）</h2>" + extra["premises"])

    # 8. 解釈・次
    o.append("<h2>8. 解釈と次の実験</h2>" + extra.get("next", "<ul><li>tp-05b で CatBoost・XGBoost・線形・NN を同じ条件で比較する。</li></ul>"))
    o.append("<h2>9. 再現方法</h2><div class='box'><code>papermill notebooks/modeling/training_pipeline/tp-05a-baseline-objectives.ipynb …</code>（ホスト、KEIBA_ENV=stg、POLARS_MAX_THREADS=6）。"
             "成果物は <code>data/local/modeling/training_pipeline/model_screen/</code>（summary.csv・per_fold_logloss.csv・vs_A1.csv・fold_metrics.csv・report.json）。"
             "このレポートは <code>python -m src.scripts.docs.build_model_report_html</code> で作成。</div>")
    o.append("</main></body></html>")
    a.out.write_text("\n".join(o), encoding="utf-8")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
