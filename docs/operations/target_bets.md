# 推奨馬券の既定方策 target（3連複・目標払戻 5 倍）

2026-10-24 にユーザー決定で既定にした。検証は決定ログ §78、採用の記録は §79。

## 何をするか
1 レースの予算 B（UI の購入上限。既定 2,000 円、空欄は 10,000 円で計画）に対し、**どれか 1 つの 3連複が当たれば払戻が約 5B** になるように、買い目ごとの購入額を決める。
購入額 = `ceil(M · B · k / 想定オッズ)`（100 円単位）、合計 ≤ B、最大 3 点、想定の期待値（的中確率 × 想定オッズ）≥ 1.0。条件を満たす組が無ければ見送り。

**見送り（推奨度 0）のレースは、推奨馬券を一切返さない**（2026-10-26）。`delegates.optimize_betting` の出口（`hide_skipped_bets`）で、買い目・他プランの買い目・金額・的中率・期待値・印の付け直しを落とし、推奨度と見送りの理由だけ残す（`skipped: true`。仮想購入の結果も付けない）。一覧の段の計算など内部用には `hide_skipped=False`。過去の予測成績の仮想購入も、見送りのレースは出さない。

## パラメータ（`config/target_bets.json`）
| 項目 | 値 | 意味 |
|---|---|---|
| mult (M) | 5.0 | 目標払戻の倍率 |
| k | 2.0 | 想定オッズの誤差への余裕（必要購入額を 2 倍） |
| tau | 1.0 | 想定の期待値の下限 |
| max_bets | 3 | 買い目の上限 |
| tier_thresholds | 0.05 / 0.075 / 0.11 | 推奨度 1〜4 の境（どれかが当たる確率） |

係数 `config/odds_pred/odds_pred_config.json`（学習済み成果物。公開リポジトリに置かない。各環境に配置する）。

## 入力と出力
- 入力: 自前の勝率・連対率・複勝率（保存済み予測）、想定単勝・想定複勝（既定は §77 のモデル `sec77`。下の『想定オッズの出所』）。**実オッズは使わない・出さない**。
- 出力（`POST /api/v1/betting/optimize`）: `strategy:"target"`、`status`（bet/skip）、`bets[]`（`combo`、`yen`、`est_multiple`＝想定倍率、`est_payout_yen`＝想定払戻、`hit_rate_pct`）、`any_hit_rate_pct`、`race_tier`（推奨度）。`/betting/race-values` は推奨度の段階（`basis:"target"`）。
- 見せる相手: 開発者・有料会員。ゲスト・無料は従来どおり 401。

## 注意（画面にも出している）
- 当たったときの払戻水準を固定する方式で、**期待値・回収率 100% 超を保証しない**（検証の回収率は 2025 年で 1.215、95% 区間 0.99〜1.46）。的中率は約 6%。約 94% のレースで購入額を失う。
- 想定倍率・想定払戻は AI の予測で、実際のオッズとは異なる。
- 想定単勝・複勝が §77 のモデルで作られていない（キャッシュなしで従来モデルや自前確率に戻った）レースは、検証より入力の質が落ちる。応答の `odds_basis` で区別できる（`sec77` / `predicted`=従来 / `own`=自前確率から）。

## ロールバック
- `.env` に `KEIBA_BET_STRATEGY=legacy` を足して API を再起動（旧方式に戻る。再起動はキューが空のときに）。
- 係数ファイルを退避しても、自動で旧方式になる（`available()` が False）。

## 前向きの追跡の方針
設定を凍結したまま、毎開催日の推奨（買い目・購入額・想定倍率）を保存し、結果の確定後に実払戻で回収率を集計する。賭け数が約 3,000 に達してから、100% 超かどうかを区間で判断する。それまでは『100% 超』と主張しない。

## 想定オッズの出所（sec77）

- 出所の優先順: `sec77`（§77 のモデル）→ `predicted`（従来の final_odds 予測）→ `own`（自前確率から機械的に作る）。応答の `odds_basis` に記録する。
- 作り方: 馬体重の発表後（T-60）の予測で作る当日の特徴量（v6 のサービング経路 `V3Serving.features()`）→ LightGBM（対数シェア・対数複勝。学習は 2020〜2025 年、`mtl_` 列とオッズ・人気は不使用）→ 較正の直線 → レース内で正規化 → 想定単勝・想定複勝（`src/pipeline/inference/odds_sec77.py`）。
- 保存先は従来の最終オッズ予測のキャッシュ（カテゴリ `final_odds_prediction`）。entries の `source` が `sec77`。`odds_source.predicted_single_odds` を通る表示（レース詳細・race-values・推奨馬券・予測成績）が自動で同じ値を使う。従来モデルの結果は、期限内の sec77 を上書きしない。
- 旧モデル用の想定オッズの較正（`paid_zone.calibrate_single`）は sec77 には重ねない（学習時に市場の水準へ較正済み）。
- 学習・保存: `python -m src.scripts.data.odds_sec77_fit fit`（`data/local/modeling/odds_pred/serving/`。再学習のたびに実行）。過去日の算出・比較: `python -m src.scripts.data.odds_sec77_backfill --from … --to … [--dry-run]`。
- 有効化: モデルが存在すれば既定で有効。`KEIBA_ODDS_SEC77=0` で無効（従来の想定オッズに戻る）。特徴量づくりのために、ホストの予測プロセス（weight_triggered_predict）が v6 の V3Serving をもう 1 つ読み込む（v6 の shadow が有効ならそれを共有）。
- 期限: sec77 のキャッシュは期限切れにしない（`FinalOddsCache.load_cached`）。従来モデルの結果だけ 24 時間。

### 基準の固定（2026-10-26。2026 年 1〜8 月の推奨馬券が約 50% に落ちた原因の再発防止）

検証と別の想定オッズ（従来モデル）で推奨・集計が作られると、ほぼ全レースが最注目になり回収率が崩れる。次の 5 つで、基準が違うものは**出さない・数えない・気づける**ようにした。

| 守り | 内容 |
|---|---|
| 推奨を出さない | stg/prod（`KEIBA_ENV` が明示）では既定で `KEIBA_TARGET_REQUIRE_SEC77=1` 相当。sec77 が無いレースは `status=skip`・`odds_pending=true`・`odds_basis=pending`（『想定オッズの準備中』）を返し、従来モデルでその場計算もしない（`target_bets.require_sec77` / `delegates._optimize_target`・`_predicted_singles`）。dev は既定オフ（`=1` で試せる） |
| 集計に数えない | 予測成績の `bets_t` は `basis=sec77` の記録だけ集計。sec77 が無い日は `status=basis` で外し、後で再計算（`prediction_results.TARGET_ODDS_BASIS`） |
| 失効させない | sec77 のキャッシュは期限切れにしない |
| 作れる環境に載せる | モデル `odds_sec77`（`ODDS_SEC77_DIR`）を `config/gcp_job_map.json` の models に追加し、`bundle_v6e` を使う全ジョブ（predict-race・eve-precompute・day-prep）に載せる。ホストからの公開: `bash scripts/gcp/publish_models_from_host.sh odds_sec77`（**再学習のたびに公開し直す**）。`tests/pipeline/test_sec77_status.py` が載せ忘れを検出 |
| 気づける | 予測ジョブのイベントに `sec77`（ok / no_model / disabled / no_numbers / error）をレースごとに記録し、モデルが無いときは ERROR ログ。`prediction-results-refresh`（`--add-target-bets`）は直近 7 日に sec77 が無いレースがあれば通知（`KEIBA_SLACK_NOTIFY=1` で Slack） |

sec77 を再学習・更新したときは、`odds_sec77_fit` → `publish_models_from_host.sh odds_sec77` → 過去分の `backfill_target_bets.sh --sec77` の順で、検証・本番・集計の基準を揃える。

## プラン方式（全券種・縛りなしが既定、2026-10-25）
- 既定プラン `pattern_all`: 単勝・複勝・馬連・ワイド・馬単・3連複・3連単を、券種・馬を自由に組み合わせる。購入額 s_i = ceil(M·B·k/想定オッズ_i)（M=5, k=2.0）で、どれか1つ当たれば払戻 ≥ M·B。上位3着の全順列（MB60）の確率和で『どれか当たる確率』を厳密に数え（`portfolio_pattern.py`）、費用対効果上位50件から最大5点をビームサーチ（幅12、厳密列挙との一致 30/40、平均差 0.0014）で最大化。
- レース選択: パターンの期待回収率（想定オッズ・k で保守化）が `race_ev_threshold`（2.0）以上のレースだけ推奨（2022〜24年で上位約30%、2025年は約2割）。推奨度チップは `race_ev_tiers` による段階。検証（k=2.0, τ=1.0）: 期待値の5分位と実現回収率は 0.88/0.96/1.00/1.20/1.33 と単調。上位30%で 2022〜24年 1.26（1.15〜1.40）、2025年 1.30（1.05〜1.56）。全レースを買うと 1.07。100%超は未確認。
- 補助プラン（3連複型・ワイド型・馬連型・馬単型・単勝型・複勝1点・ミックス）は貪欲法で、`config/target_bets.json` の `plans` に定義。各プランの検証成績は `evidence` に記載（2025年に1.0を割るプランも多い）。
- 3連単の想定オッズ（確率の2次多項式）は、取得済みの実オッズ（2024〜2025、694,119組）と比べて対数RMSE 0.99、平均で実際より約40%低く出る（保守側）。誤差が大きいので、3連単は参考。
- 計算時間: 1レース約 0.01 秒（候補生成＋探索）。検証: `portfolio_pattern_eval.py run`、プラン: `portfolio_plans_eval.py`。
