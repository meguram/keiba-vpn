# めぐ指数の撤廃の計画（調査と計画のみ・2026-11-18）

ユーザーの依頼（2026-11-18、統括役の経由）: 「めぐ指数って結局使われているのですか？予測以外の用途（UI上での表示など）は現状考えていないですので、撤廃してほしいです。」

この文書は調査と計画だけ。実装は統括役とユーザーの承認の後で、本番に効く変更は 11/22（日）の夜以降に行う。データ・GCS は消さない（削除は計画に書くだけ）。

## 0. 結論

- **めぐ指数は予測に効いている。効きは小さいが、学習の乱数を変えても再現した。** 候補 r04 で、めぐ指数の 8 列（`rb_mg_*`）を使う 3 メンバーすべてから列を抜いて学習し直すと、勝ち馬の対数損失（レース単位・主の指標）が悪くなった。
  - 1 本目（基準と同じ乱数）: +0.00107（95% 区間 +0.00030〜+0.00184）
  - 2 本目（乱数を変えた組で、抜いた版と抜かない版を比べる）: +0.00112（+0.00033〜+0.00192）。2022・2023・2024 年のどれも悪化した。
  - 乱数だけを変えた 2 つの版どうしの差（偶然だけで出る差）は −0.00035・−0.00030 で、効きの 3 分の 1 ほど。
  - 大きさの目安: r04 が本番 v6e より良くなった分（−0.0161）の約 7%。副の指標（◎ の勝率・複勝率、回収率 4 種）は記述だけ（多重比較の補正前）。◎ の複勝率は 2 本とも約 −0.3 ポイントで、ほかの副の指標の差は区間が 0 をまたぐ。
- 予測が使うめぐ指数は、走ごとの `megu_index_v3a` から作る `rb_mg_*` の 8 列だけ（本番 v6e・候補 r04・想定オッズ sec77）。前段 S3・補助ヘッドは使っていない。印・推奨馬券は、表示用の `megu_mark`・`expected_megu` を使っておらず、勝率と想定オッズを通して間接的に効くだけ。
- 予測以外（画面・API・表示用の計算・表示用の定期処理）は、予測に影響しないので外せる。会員の画面には今も、`/weekly-predictions` のめぐ指数の表・馬のパネルの「直近のめぐ指数」と「成長曲線」が出ている。
- **推奨: 「予測の特徴量だけ残す」**（統括役の解釈のとおり）。
  - 段 A: 画面・API・表示用の計算を外す。予測の数字は変わらない。11/22 の夜以降に行う。
  - 段 B: 特徴量の 8 列は残す。めぐ指数の計算は、予測の流れ（週次の計算 → flat → RB の状態）の中だけに閉じる。
  - 全部の撤廃（列も外す）を選ぶ場合は、上の精度の損と、学習のやり直し（全メンバー・束・第 2 段・sec77）を受け入れることになる。ユーザーの判断事項として §3 の段 B に書く。
- 今週末（11/21・22）の配備（r04＋第 2 段）は何も変えない。データ・GCS は消さない。
- 効きの測りには弱みが残る: v2 の基準タイムが 2020〜2025 年で当てはめたままで、効きを少し膨らませる向きの先読みがある（§2.3）。B2 を選ぶときは、作り直した版の valid と test 2025 で最終確認する。
- 費用への効きは小さい（Cloud Run Jobs は 11/13 から止まっているので今の減る分は 0 円に近く、再開した場合でも月 10 円前後。増える分は 0 円）。撤廃の理由は費用ではなく、運用の弱さと保守の手間。

## 1. 使われ方の棚卸し

### 1.1 予測（モデルの特徴量）

予測が使うめぐ指数は、走ごとの `megu_index_v3a` 1 列だけ。そこから作る RB の `rb_mg_*` 8 列（直近 5 走の ewm5・max5・median5・trimmed5・huber5・fl_ewm・fl_whuber・n）が入力になる（`src/pipeline/features/serving_v6.py:23-28`）。

| モデル | めぐ指数の列 | 備考 |
|---|---|---|
| 本番 v6e（7 メンバー） | `rb_mg_*` 8 列（lgb_extras_ped・xgb_extras_ped・lgb_extras・lgb_softmax・linear） | netkeiba 指数の RB `rb_nk_*` 8 列も持つ。mtl_nn の 2 つは 0 列 |
| 候補 r04（5 メンバー） | `rb_mg_*` 8 列（xgb_extras_ped・lgb_softmax・linear） | `rb_nk_*` は外してある。RB はめぐ指数だけ |
| 想定オッズ sec77（share・place） | `rb_mg_*` 8 列 | 本番では既定で有効（`KEIBA_ODDS_SEC77=0` で無効）。想定オッズ → 推奨馬券につながる。r04 の影のランナーでは無効（`config/candidate_runtime.json` の `KEIBA_ODDS_SEC77=0`） |

次のものは、v6e・r04・sec77 のどの列にも入っていない（bundle.json の `feature_cols`・stage_meta.json・meta.pkl で確認）。

- 前段 S3（v6e の `final_stage/v6_e_2025`、r04 の `stage_s3t7`）、補助ヘッド（`training_pipeline_v4/final_aux`）
- `tp39_growth`・`tp41_similar`・`tp42_racelevel`・`tp76_asof`（`B_m_megu`）・`factory_src` の `MEGU_hist`（`mg_*`）
- `expected_megu`・`megu_mark`・成長度・クラス基準
- 名前が似ている `f1_sim_*`（`horse_history/match.py`、Elo の残差）と `sr_perf_*`（`horse_history/rating.py`、独自の基準）はめぐ指数を使わない

予測に必要な計算の連鎖:

1. 入力: `race_result_flat`・DB の係数（`megu_regression_params`・`megu_par_time`）・`config/megu_v3a/`（`src/pipeline/megu_index/compute.py:313,330`）
2. v2 の計算（`compute.py:440-626`）→ v3a（`v3a.py:366-409`。v2 の途中の列を使うので v2 も外せない）
3. 保存: `tables/<年>/megu_index_flat.parquet` と GCS の正本（`compute.py:699-778`）。DB へ v2 を upsert
4. 走ごとの表: `src/scripts/data/tp65_prep.py:25-75`（`megu_index_v3a` と netkeiba 指数 `nk`）
5. RB の状態: `build_rb_state` → `data/local/inference/state/rb_v6/`（`src/ops/jobs/rb_state.py`）。動かしているのはホストの cron `rb-state-daily`（08:20 JST、`scripts/cron/rb_state_daily.sh`）。Cloud Run Jobs の `rb-state-daily` は、2026-11-13 に費用のため Scheduler ごと止めてある（`config/gcp_job_map.json` の `scheduler_groups` の `retired`、`docs/operations/deleted_scheduler_keiba_g_20261113.md`）
6. 推論: `src/pipeline/inference/v3_serving.py:81-86,145-146,273-285`（列が欠けると KeyError）

r04（影のランナー）の RB は別の流れで作っている。リポジトリの外の `retrain_out/cycles16_fix3/rb_live/`（`rb_live.sh`・`eve.sh`・`schedule.sh`・`rb_runs_2016.py`、状態は `state_live`）で、2016〜2019 年は `retrain_out/megu_2016` を使う。推論のめぐ指数の基準は `config/candidate_runtime.json` の `KEIBA_MEGU_V3A_DIR=retrain_out/megu_v3a_ref2016-2024`。r04 は年ごとの as-of のめぐ指数で学習したが、推論で読む 2020 年以降の flat は本番の基準（`config/megu_v3a`、2020〜2025 年で当てはめ）で作られている。学習と推論の基準が少しずれる（`retrain_trials/README.md` によると `rb_mg_ewm5` の相関 0.99・平均絶対差 約 1 点）。

### 1.2 印・推奨馬券

- 印（◎○▲△）と推奨馬券（`target_bets`）は、めぐ指数の応答項目（`megu_mark`・`expected_megu`）を使っていない。めぐ指数の項目は `_with_bet_marks` の後で付く（`src/api/app.py:7248` → `:7290-7297`）。
- 「めぐAI印」は印の種類 `mark_type` の名前（`src/api/tier.py:3,34`・`config/mark_rule.json`）で、めぐ指数から作る `megu_mark`（めぐ◎/めぐ○、`config/megu_mark.json`）とは別物。
- ただし、勝率（v6e）と想定オッズ（sec77）の入力に `rb_mg_*` があるので、印・推奨馬券には間接的に効いている（効きの大きさは §2）。

### 1.3 画面（frontend）

| 場所 | 表示 | 今見えているか |
|---|---|---|
| `frontend/components/MeguTable.tsx`・`PastRuns.tsx`（`weekly-predictions/page.tsx:623`） | めぐ指数（直近 5 走）の表・めぐ印・突破ライン | **見えている**（会員。機能フラグの外） |
| `frontend/components/ExpectedMegu.tsx`（`EntryCardPanel.tsx:393-397`） | 馬のパネルの「直近のめぐ指数」 | **見えている** |
| `frontend/components/GrowthCurve.tsx`（`EntryCardPanel.tsx:399-401`、API `entry_card.py:401` → `horse_growth.py`） | 「成長曲線」。縦軸の既定がめぐ指数（`megu_index_v3a_abs_robust`） | **見えている** |
| `frontend/app/megu-index/page.tsx`・`home/homeData.ts:78-83` | 今週のめぐ指数 | 見えない（`features.ts:35` の `megu-index` が無効） |
| `frontend/app/race/[id]/page.tsx`（想定/実測めぐ指数の列） | レース詳細の列 | 見えない（`SHOW_MEGU=false`） |
| `frontend/app/horse/[id]/page.tsx`（指数タブ） | 馬ごとの推移 | 見えない（`horse-detail` が無効） |
| `frontend/lib/{expectedMegu,pastRunsView,classBenchmark,growth}.ts`・`mock.ts:465-579` | 型・表示の関数・モック | 上の部品の部品（`growth.ts` は未使用） |

### 1.4 API

| 場所 | 中身 | 使われているか |
|---|---|---|
| `src/api/app.py:7290-7297`（`/api/race/{id}/predictions`） | `attach_expected_megu` で `expected_megu`・`megu_mark`・`class_benchmark`・`growth` を毎回付ける | **毎回動く**（画面の MeguTable・パネルが使う） |
| `src/api/entry_card.py:401-436` | パネルの `growth`（成長曲線） | **使われている** |
| `src/api/flask_app.py:261-415`（`/api/v1/races/<id>/megu-index`・`/horse/<id>/megu-index-history`・`/races/<id>/megu-index-predicted`）・`src/api/megu_predict_race.py` | DB の `megu_index`（v2）を返す | 無効な画面からだけ |
| `src/api/app.py:3114-3141`（`/api/megu-index-dates`） | 開催日の一覧 | 無効な画面からだけ |
| `src/api/tier.py:34-36,347-348`・`config/public_fields.json:14,22,25,52`・`src/utils/paid_zone.py:196` | 匿名・公開の出し分けの項目とパス | 項目を消すときに一緒に直す |
| `src/api/quality_health.py:358-440`・`monitor_coverage.py:242-282`・`quality_auto_remediation.py:243-355` | 品質の監視の `calculated`（めぐ指数のそろい）。レース詳細の警告バッジ（`app.py:7081`）につながる | 使われている |
| `scripts/ops/vps/cache_warm.py:43,49,66`・`src/scripts/ci/smoke_endpoints.py:192`・`config/http_cache.json:28` | 先読み・スモーク | エンドポイントを消すときに直す |

推論側の付与（表示用）: `src/pipeline/inference/{expected_megu,hem_megu_infer,megu_excusable_attach,megu_mark,class_benchmark,growth_attach,growth_index}.py`、`src/pipeline/megu_index/v3b.py`。`config/retrain_plan.json:98` にも「親モデルの入力ではない」とある。

### 1.5 定期の処理・データ

| 処理 | 中身 | 予測に必要か |
|---|---|---|
| `src/scraper/auto_scrape.py:340,381-436` `finalize_weekly_update`（金曜。ホスト） | 開催日と直近 60 日の欠けのめぐ指数を計算し、flat・PG・GCS に保存 | 必要（flat を作る主な処理。`KEIBA_WEEKLY_MEGU_BATCH=0` で止まる） |
| 品質の自動修復 `src/api/quality_auto_remediation.py:354-363`（`quality_health.py:466` の `maybe_remediate_after_checks`。stg では既定で有効 `:57-64`。API `POST /api/quality-check/remediate` `app.py:2321`） | `calculated` が欠けると `compute_for_date` でめぐ指数を計算し直し、flat・GCS に書く | flat を書くもう一つの道。`compute.py:701-760` は手元に flat が無いと 1 日分の表で GCS の年の正本を上書きしうる |
| ホストの cron `rb-state-daily`（08:20 JST） | RB の状態（本番 v6e 用）の作り直し | 必要（`rb_mg_*`・`rb_nk_*` の両方） |
| `retrain_out/cycles16_fix3/rb_live/`（リポジトリの外） | r04 の影のランナー用の RB の状態 | r04 に必要 |
| ホストの cron `KEIBA-HEM-STATE`（`src.scripts.data.hem_megu state`、月曜 07:30 JST） | 想定めぐ指数の騎手の表 | 表示用だけ |
| `src/ops/gcs_mirror.py:64`（`tables::*/megu_index_flat.parquet`） | Cloud Run の API に flat を配る | 表示用だけ。ただし品質の自動修復が API の中で動くと flat を読み書きする（上の行） |
| Jobs の state_in に flat が入るもの: eve-precompute・predict-race・px-ridge-window・day-prep・rb-state-daily・hem-megu-state | 取り込み指定 | 2026-11-13 から Scheduler ごと止まっている（`retired`）。rb-state-daily 以外は、コード上 flat を読む箇所が見つからない（再開するなら要確認） |
| raceday-evening の `compute_for_date`（`auto_scrape.py:1258-1268`） | キュー版（既定）では動かない | — |
| raceday-weight-runner の案（月曜夜の確定結果からめぐ指数・RB を作る。`rb_state_megu_plan_20261118.md`、レビュー中） | 金曜の遅れの解消 | めぐ指数を撤廃するならめぐ指数の段は不要。RB の段は `rb_nk_*` を使うモデルが残る限り要る |

### 1.6 docs・テスト

- 予測の連鎖: `docs/operations/{pred_model_v6e,v6_state_refresh,megu_v3a_rollout,model_cycles_20261116}.md`、`docs/html/modeling/chapters/{v6_features,index_variance,megu_index_v3a}.html`。テストは `tests/pipeline/test_serving_v6.py`・`test_tp65.py`・`tests/ops/test_rb_state_job.py`・`test_megu_v3a*.py`・`test_megu_weekly_batch.py`・`tests/*/test_weekly_finalize.py` など。
- 表示用: `docs/operations/expected_megu_hem.md`、`chapters/{megu_marking,megu_excusable,class_benchmark,growth_curve_weight}.html`。テストは `test_expected_megu.py`・`test_hem_megu*.py`・`test_megu_mark.py`・`test_horse_growth.py`・`test_megu_predict.py`・`tests/api/test_megu_predict_race_helpers.py`・`test_entry_card.py`・`test_tiered_access.py`、frontend の `pastRunsView.test.mjs`・`classBenchmark.test.mjs`・`growthCurve.test.mjs`。
- 研究用: `test_tp39_growth.py`・`test_tp41_similar.py`・`test_tp42_racelevel.py`・`test_tp66.py`・`test_tp69.py`。`tests/conftest.py:138-139` が `config/megu_v3a` に依存。
- 設定・一覧（レビューで追加）: `config/host_dependency_baseline.json`（`megu_predict_race.py`・`expected_megu.py`・`hem_megu_infer.py` などの行）、`config/fitted_artifacts.txt:6-7,23-26` と `tests/docs/test_docs_consistency.py:22`、`config/retrain_plan.json:95-98`、`config/growth_index/`、`src/data_health/spec.py:134` と `tests/data_health/test_data_health.py:244`（`megu_predict_*.json`）、`config/persona_inputs.json:79`（`B_m_megu`）。
- テスト（レビューで追加）: `tests/ops/test_vps_cache_warm.py:101`（`megu-index-history` を期待）、`tests/api/test_monitor_coverage.py`。
- 表示用の v2 の想定めぐ指数の部品: `src/pipeline/megu_index/{predict,predict_params,optimize_predict}.py`。
- flat を読む研究用: `folklore_build.py:538`・`v6_ext_build.py:138`・`dam_age_analysis.py:215`・`src/research/pedigree/growth_weight_*`・`backup_local_to_gcs.py:65`。
- docs の本の棚: `src/scripts/docs/docs_shelves.py:93,204-205`（`megu_marking`・`megu_excusable`・`class_benchmark` の章を消すと棚が壊れる）。
- コラム: `frontend/content/columns/index-claims.json` と、別のファイル `frontend/content/columns/guides/index-claims.json`。

## 2. 予測への効き（数字）

費用 0 円・手元のデータだけ。評価は valid 2022〜2024（10,366 レース・年ごとに広げる fold、学習は 2016 年から）。2025 年の test は r04 の 1 回きりの評価を使ってしまうので使わない。2026 年の着順は読まない（10/17〜11/15 のめぐ指数の計算し直しは 2026 年の表なので、この評価に関係しない）。
主の指標は model_cycles と同じ「勝ち馬の対数損失」（レースごとに −log p(勝ち馬)、p はレース内で正規化）。区間はレースの再標本化（2,000 回）の 95%。差は「抜いた版 − 基準」で、正は悪化。
道具は `src/scripts/data/model_cycles.py`・`model_cycles_members.py`（model-improve-cycles の worktree のものを読むだけで使用）、出力は `/home/jovyan/work/retrain_out/megu_ablation/`（実行の記録 `run.sh`・`run2.sh`・`run3.sh`・`run4.sh`・`seedwrap.py`、ログは同じ名前の `.log`、採点は `r04/score/`・`v6e16/score/`）。

### 2.1 学習済みモデルの gain の重要度

| モデル・メンバー | 勝ちの重み | `rb_mg_*` の gain の割合 | 最上位の列の順位（全 1,160〜1,231 列） |
|---|---|---|---|
| v6e lgb_extras_ped | 0.24 | 0.56% | 38 位（rb_mg_fl_whuber） |
| v6e lgb_softmax | 0.23 | 0.71% | 41 位 |
| v6e xgb_extras_ped | 0.06 | 0.64% | 57 位 |
| r04 xgb_extras_ped | 0.28 | 0.66% | 34 位（rb_mg_max5） |
| r04 lgb_softmax | 0.21 | 0.72% | 27 位 |
| sec77 share・place | — | 0.16%・0.14% | 88 位・124 位 |

8 列は全列の 0.7% で、gain の割合もほぼ同じ（1 列あたり平均並み）。linear（重み約 0.2）は gain が無いので、この表では分からない。

### 2.2 列を抜いて学習し直す（ablation）

r04 は model-improve-cycles の最終の候補のとおり。xgb は `p01_fix_s3`（元の設定・3 seed）、アンサンブルは `ens5_p01s3`、土台は `base_2016_cyc3`。基準は 0.20709 で、重みも記録と一致することを確かめた。
抜く列は `drop_prefix` に `rb_mg_` を足しただけで、ほかの spec は同じ。lgb_softmax・linear は `model_cycles_members` で作り直した（`CYCLES_MEMBER_OUT=megu_nomg`）。mtl_nn の 2 メンバーは、もともとめぐ指数の列を持たない。
乱数を変えた組は、xgb を `seed_offset=3`、lgb_softmax・linear を seed 3・4 にした（道具のコードは変えず、`seedwrap.py` で `tp27_eval.SEEDS` だけ差し替えた）。この組で「めぐ列あり」（`q_ctlfull`）と「めぐ列なし」（`q_fullctl`）を作り、2 本目の推定にした（レビューの指摘への対応）。

| 版（r04） | 勝ち馬の対数損失の差（基準比） | 95% 区間 | 2022 / 2023 / 2024 | ◎ 勝率 | ◎ 複勝率 | 単勝の回収率 |
|---|---|---|---|---|---|---|
| 基準 `q_base` | 1.92802 | — | — | 33.30% | 64.43% | 90.6% |
| xgb だけから抜く `q_xgbonly` | +0.00021 | −0.00014〜+0.00055 | +0.0002 / +0.0002 / +0.0003 | −0.03pt | −0.03pt | −0.2pt |
| **3 メンバーすべてから抜く `q_full`** | **+0.00107** | **+0.00030〜+0.00184** | +0.0002 / +0.0023 / +0.0006 | −0.17pt（−0.40〜+0.07） | **−0.28pt（−0.55〜−0.03）** | −0.7pt（−1.8〜+0.4） |
| seed の対照（xgb だけ） `q_ctl` | −0.00005 | −0.00049〜+0.00038 | +0.0001 / −0.0001 / −0.0001 | −0.13pt | −0.05pt | −0.6pt |
| seed の対照（3 メンバー） `q_ctlfull` | −0.00035 | −0.00098〜+0.00031 | −0.0005 / +0.0002 / −0.0007 | −0.17pt | +0.01pt | −1.0pt（−2.1〜−0.0） |
| `q_full` − `q_ctlfull`（参考。乱数も違う） | +0.00142 | +0.00062〜+0.00225 | +0.0008 / +0.0021 / +0.0014 | 0.00pt | −0.29pt（−0.58〜0.00） | +0.3pt |
| **2 本目: 乱数を変えた組で抜く `q_fullctl` − `q_ctlfull`** | **+0.00112** | **+0.00033〜+0.00192** | +0.0007 / +0.0014 / +0.0013 | −0.09pt（−0.34〜+0.14） | −0.34pt（−0.63〜−0.06） | +0.1pt（−1.0〜+1.4） |
| 参考: `q_fullctl` − `q_base`（乱数も違う） | +0.00077 | −0.00005〜+0.00164 | +0.0002 / +0.0015 / +0.0006 | −0.26pt | −0.33pt | −0.9pt |

主の指標は勝ち馬の対数損失 1 本。◎・回収率・クラス別は記述だけで、多重比較の補正はしていない（副の指標は 8 つ、クラスは 7 つ）。対照の `q_ctlfull` − `q_base` でも単勝の回収率が −1.0 ポイント（区間 −2.1〜−0.0）と 0 をまたがない差になっており、回収率の差は乱数だけでもこの大きさで動く。

クラス別（記述だけ）: 1 本目（`q_full` − `q_base`）は未勝利 +0.0020（+0.0006〜+0.0032、3,444 レース）が最も大きく、2 本目（`q_fullctl` − `q_ctlfull`）は 2 勝 +0.0024（+0.0001〜+0.0046）が最も大きい。2 本で最も大きいクラスがそろわないので、クラス別の傾向は採らない。

xgb のメンバー単体（3 seed）の差は、抜いた版 +0.00099（−0.00033〜+0.00225）、seed の対照 −0.00012（−0.00168〜+0.00140）。

参考 1（不採用の版での測り）: 最初は学習率の版 `p06_lr3_s3`（前段を作り直した後に効きが消えて不採用になった版）で測っていた。この版では xgb だけ抜くと +0.00014（−0.00009〜+0.00039）で、`q_xgbonly` と同じ傾向だった。

参考 2（本番 v6e と同じ入力の v6e16、xgb 1 seed）: めぐ指数だけ抜くとアンサンブル +0.00027（−0.00060〜+0.00117）、RB 16 列をすべて抜くと +0.00097（+0.00009〜+0.00179）。ただし seed だけ変えた対照が +0.00128（+0.00036〜+0.00220）で、どちらも乱数のぶれの内だった。1 seed では判定できない。v6e は netkeiba 指数の RB `rb_nk_*` も持つので、めぐ指数の列を抜いても `rb_nk_*` が一部を肩代わりすると見られる（要確認）。v6e は r04 に置き換わる予定なので、v6e での作り直しはしない。

### 2.3 事象と根拠

| 項目 | 事象 | 根拠 |
|---|---|---|
| r04 でめぐ指数が効く | 3 メンバーから抜くと、対数損失が +0.0011 前後悪くなる。乱数を変えた 2 組で再現し（+0.00107・+0.00112）、2 本目は 3 年とも同じ向き。乱数だけの差（−0.00035・−0.00030）の約 3 倍 | めぐ指数は、走破タイムを馬場・ペース・斤量で直した能力の値で、出走の 96% にある（`base_2016/tp65/prep_report.json` の `megu_v3a_cover` 0.959）。同じ種類の netkeiba 指数は 49%（`nk_cover`）しかなく、r04 では推論とのずれのために外してある。タイムに基づく能力の値の履歴は、r04 ではほかに代わりが少ない（自前の `sr_perf_*` はある）。どのクラスで効くかは 2 本でそろわず、仮説の段階 |
| xgb だけから抜くと効かない | +0.00021 で偶然の幅の内。アンサンブルの重みが lgb_softmax へ移る（0.21 → 0.24） | 同じ情報を lgb_softmax・linear がまだ持っているので、ほかのメンバーで代わりが利く。効きを測るには、列を持つメンバーをすべて作り直す必要がある |
| 先読みで効きが膨らんでいないか | v3a の基準は年ごとの as-of。ただし v2 の基準タイムは 2020〜2025 年で当てはめたまま | v3a の基準は 2026-11-09 に年ごとの as-of に作り直した（`retrain_trials/README.md`。今の `rob.parquet` は 11/09 09:22 作で、旧版は `rob_megu_ref2025.parquet` に退避）。一方、v3a は `front_split_dev`（`v3a.py:162-163`）を使い、これは `front_split_sec − par_front_split_sec`（`compute.py:536`）。`par_front_split_sec` は DB の `megu_par_time` から来ており、2020〜2025 年で当てはめてある（`config/megu_par_time_v2_meta.json`）。`megu_asof.sh` は v3a の基準だけを作り直したので、検証年の前半のラップの基準にその年の結果が入っている。影響はペースの補正（3 秒でクリップ・係数 1 つ）に限られ小さいと見込むが、効きを膨らませる向き。本番 v6e の表は v3a も旧の基準のまま |
| 判定の基準 | 主の指標の区間が 0 をまたがない、乱数を変えた組でも再現する、3 年で同じ向き、乱数だけの差より大きい | `evidence_policy.txt` の「偶然の見込み・年ごとの安定」。レースの再標本化の区間には学習の乱数のぶれが入らないので、乱数を変えた組を別に作った（v6e16 の 1 seed では、乱数のぶれだけで +0.00128 と区間が 0 をまたがない差が出た）。この基準は 1 本目の結果を見た後に書いたので、事前の登録ではない。2 本目は基準を決めた後に流した |
| 残る弱み | 乱数の組は 2 つだけ。v2 の基準の先読み。test 2025 は未使用。基準の r04 の設定は、めぐ列ありの状態で選ばれた | 乱数のぶれの分布は 2 点でしか見ていない。test 2025 は r04 の 1 回きりの評価なので使わない。抜いた版は設定を選び直していないので、効きをわずかに大きく見せる向きの偏りがある（小さい）。全部の撤廃（B2）を選ぶときは、作り直した版の valid と test 2025 で最終確認する |


## 3. 撤廃の段取り

効いているので、推奨は「予測の特徴量だけ残す」。段 A は、どちらを選んでも同じ内容で行う。

3 段に分ける。段 A は予測に触れないので先に進めてよい。段 B・C は予測に効くので、モデルの担当（model-improve-cycles）の次の版に合わせる。

### 段 A: 画面・API・表示用の計算を外す（予測に影響なし・11/22 の夜以降）

予測の数字（勝率・印・推奨馬券・成績の照合）は変わらない。変わるのは会員の画面の表示と、予測 API の応答の項目だけ。

1. 画面（frontend）を先に外す（API より先に外し、項目が無くても画面が壊れないようにする）
   - `weekly-predictions/page.tsx:623` の `<MeguTable>` と `:328` の受け渡し、型 `expected_megu`・`expected_megu_meta`・`class_benchmark`
   - `EntryCardPanel.tsx` の「直近のめぐ指数」（`:393-397`）と「成長曲線」（`:399-401`。縦軸の既定がめぐ指数なので欄ごと外す）
   - 削除: `components/{MeguTable,PastRuns,ExpectedMegu,GrowthCurve}.tsx`、`lib/{expectedMegu,pastRunsView,classBenchmark,growth,growthCurve}.ts` と各 `.test.mjs`、`app/megu-index/`、`mock.ts:465-579`、`home/homeData.ts:78-83`、`race/[id]/page.tsx` のめぐの列、`horse/[id]/page.tsx` の指数タブ
   - `lib/features.ts` の `megu-index` はページと一緒に消す（ページを残してフラグだけ消すと `/megu-index` が開ける。`middleware.ts:114`）
   - コラムの記事 `content/columns/index-claims.json`・`content/columns/guides/index-claims.json` の「めぐ指数」の文言は、記事の担当（column-design）に確認してから直す
   - 対象外: 別の機能の成長曲線（`/growth-curve`・`/api/growth-curve/*`・`/api/v1/horse/<id>/growth-curve`・`growth_curve_service.py`。めぐ指数を使わない）は消さない
2. API
   - `src/api/app.py:7290-7297` の `attach_expected_megu` の呼び出しを外す（`expected_megu`・`megu_mark`・`class_benchmark`・`growth`・`growth_meta` がまとめて消える。成長度 `KEIBA_GROWTH_INDEX` もここだけから呼ばれる）
   - `src/api/entry_card.py:401-436` の `growth`
   - 削除: `flask_app.py:261-415` の 3 エンドポイント、`megu_predict_race.py`、`app.py:3114-3141` の `/api/megu-index-dates`、`src/scripts/data/eval_megu_predict.py`
   - 先読み・スモークの一覧は同じ commit で直す: `scripts/ops/vps/cache_warm.py:43,49,66`（と `tests/ops/test_vps_cache_warm.py:101`）、`src/scripts/ci/smoke_endpoints.py:192`、`config/http_cache.json:28`、`scripts/ops/vps/probe_stg_api.sh:4`
   - 公開・会員の出し分けの一覧（`src/api/tier.py:34-36,347-348`、`config/public_fields.json:14,22,25,52`、`src/utils/paid_zone.py:196`）は、段 A では消さない。消すと、項目がどこかに残っていた場合に匿名に出る向きに壊れる。残す費用は 0。stg と本番で項目が出なくなったことを確かめてから、後で掃除する
3. 表示用の計算
   - 削除: `src/pipeline/inference/{expected_megu,hem_megu_infer,megu_excusable_attach,megu_mark,class_benchmark,growth_attach,growth_index,horse_growth}.py`、`src/pipeline/megu_index/v3b.py`・`excusable.py`、`config/{megu_mark.json,class_benchmark.json,expected_megu/}`、学習側 `src/scripts/data/{expected_megu_fit,hem_megu,class_benchmark_build}.py`
   - ホストの cron `KEIBA-HEM-STATE`（`hem_megu state`、月曜 07:30 JST）を止める。`config/gcp_job_map.json:4241` の `hem-megu-state` の行を外す（Scheduler はもう無い）
   - `src/ops/gcs_mirror.py:64` の `tables::*/megu_index_flat.parquet` を外す（Cloud Run の API で flat を読むのは表示用だけ）。束（`ops/mirror/_bundle/base.tar.gz`）と目録から外れることを確かめる。**外す前に**、品質の自動修復（`quality_auto_remediation.py:354-363`・`POST /api/quality-check/remediate`）が Cloud Run の中で `compute_for_date` を呼ばないことを確かめるか、その道から `compute_for_date` を外す（手元に flat が無いと、GCS の年の正本を 1 日分で上書きしうる。`compute.py:701-760`）
   - 設定・一覧の掃除: `config/host_dependency_baseline.json`、`config/fitted_artifacts.txt:6-7,23-26`、`config/retrain_plan.json:95-98`、`config/growth_index/`、`src/data_health/spec.py:134`、`src/scripts/docs/docs_shelves.py:93,204-205`（表示用の章を消すなら、棚も同じ commit で直す）
4. テスト: 表示用のテスト（`test_expected_megu.py`・`test_hem_megu*.py`・`test_megu_mark.py`・`test_horse_growth.py`・`test_megu_predict.py`・`tests/api/test_megu_predict_race_helpers.py`）を消し、`test_entry_card.py`・`test_tiered_access.py`・`tests/ops/test_vps_cache_warm.py`・`tests/docs/test_docs_consistency.py`・`tests/data_health/test_data_health.py`・frontend の `features.test.mjs` を直す。CI と同じテストと `npm run lint`・`npx tsc --noEmit`・`npm run build` を通す
5. 確かめ: stg で予測 API の応答の `p_win`・印・`target_bets` が撤廃の前後で同じこと（同じ日の保存済みの予測で比べる）、会員の `/weekly-predictions` と馬のパネルが表示できること

### 段 B: 予測の特徴量の扱い（モデルの次の版で決める）

推奨（B1・特徴量だけ残す）: `rb_mg_*` の 8 列を、r04 とその次の版の入力として残す。予測に要る連鎖（§1.1 の 1〜6）だけを残し、それ以外を外す。

- 残すもの: `src/pipeline/megu_index/`（v2 → v3a の計算。v3a は v2 の途中の列を使う）、`config/megu_v3a/`、DB の係数、`finalize_weekly_update` の週次の計算、`megu_index_flat.parquet`、`tp65_prep`・`build_rb_state`・`serving_v6.rb_features`、ホストの cron `rb-state-daily`、関係するテスト
- r04 の RB の流れも決める（モデルの担当と）: r04 を本番にするとき、RB を `rb_live`（リポジトリの外）と `rb-state-daily` のどちらで作るか、`rb_live` 一式をリポジトリに入れるか、推論のめぐ指数の基準を as-of の運用（その日より前で当てはめる。`retrain_trials/README.md`）にそろえるか
- 外すもの（段 A に加えて）
  - Jobs の state_in の flat: eve-precompute・predict-race・px-ridge-window・day-prep（今は Scheduler ごと止まっている。設定の掃除だけ。再開するなら、各ジョブの入口から grep して flat を読まないことを確かめる）。rb-state-daily は残す
  - PG の `megu_index` への計算結果の書き込み（v2 の upsert、`compute.py:629`）。使うのは表示用の API と品質の監視だけ。**この変更と、品質の監視の `calculated` の直し（PG ではなく flat のそろいを見る形にするか外す）は、同じ commit にする**。別にすると、`calculated` が毎日 fail になり、stg の自動修復が `compute_for_date` を繰り返し呼ぶ（計算と GCS への書き込みの繰り返し）
- 運用の守り（推奨に加える）: RB の状態を作る前に、めぐ指数の検査（クラスの空・被覆・基準の段 `v3a_par_level` の割合）を関門として通す。10/24〜11/08 に、クラスが空のまま粗い基準で計算されためぐ指数が RB に入った（`rb_state_megu_plan_20261118.md`）
  - 研究用の `tp39_growth`・`tp41_similar`・`tp42_racelevel`・`tp76_asof` のめぐ指数の部分・`factory_src.megu_hist` は、モデルに入っていない。消すかどうかはモデルの担当に確認する（復活候補の台帳 `config/feature_sets.json` の revival_r1 から参照されている）
- 当日・週次の流れとの関わり: raceday-weight-runner が、月曜の夜に確定結果から `compute_megu_for_opening_dates` → `rb_state.run()` を動かす案をレビューしている（`docs/operations/rb_state_megu_plan_20261118.md`。速報から作る案は採らない方向）。B1 では予測にめぐ指数を使い続けるので、この案は RB の新しさに効き、計画と食い違わない。なお 10/24〜11/08 のめぐ指数は、クラスが空のまま粗い基準で計算されていたが、11/18 14:12 に 10/17〜11/15 が計算し直された（同じ資料）
- sec77: めぐ指数の列の gain は 0.15% と小さいが、列は持っている。B1 ではそのまま

代わりの案（B1′・計算を簡素化して残す）: v2 の DB の係数（`megu_par_time`・`megu_regression_params`）を年ごとの as-of で固定のファイルにし、v2 → v3a の計算を、RB の状態を作る処理（ホストの cron）の中にまとめる。PG への書き込み・DB の係数への依存・自動修復による再計算が外れ、v2 の基準の先読み（§2.3）も同時に消える。手間は B1 より大きい（係数の書き出し・計算の移し替え・過去日での一致の確かめ）。B1 の段 A・B を終えた後の次の候補とする。

代わりの案（B2・全部の撤廃）: 次の版で `rb_mg_*` を外す。選ぶならユーザーの判断が要る（精度の損があるため）。

- 損: valid 2022〜24 で対数損失 +0.00107（r04 の改善分の約 7%）、◎ 複勝率 −0.28 ポイント
- やること: r04 の 3 メンバー（xgb・lgb_softmax・linear）と最終の束・第 2 段（noise_fit2025）を学習し直し、valid と 2025 年 test の評価をやり直す。bet-optimizer の事前登録の確かめの期間も立て直す。sec77 も `rb_mg_*` を外して作り直す（`python -m src.scripts.data.odds_sec77_fit fit`）。
- 時期: model-improve-cycles の意見では、今週末の配備では外さず、次の版で外すなら反対しない。
- 代わりの列: netkeiba 指数の RB `rb_nk_*` は出走の 49% にしかなく、推論で直近の走の値がほぼ無いずれが知られている（`model_cycles_20261116.md:98`）。めぐ指数の代わりにはならない。

### 段 C: めぐ指数の計算そのものを止める（段 B で外すと決めた版が、本番・stg・影のすべてで動いてから）

B1（推奨）を選んだ場合、段 C は行わない。めぐ指数の計算は予測の流れの中に残す。

B2 を選んだ場合の段取り:

1. めぐ指数の列を持たない版が、本番・stg・影のすべてで動いていることを確かめる。v6e・r04（列あり）・sec77（列あり）の推論が、どこにも残っていないこと。
2. 週次の計算を止める。まず `KEIBA_WEEKLY_MEGU_BATCH=0` で止め、1 週間様子を見てからコードを外す。raceday-weight-runner の月曜夜の案のめぐ指数の段も不要になる。
3. RB の状態（Jobs `rb-state-daily`・`scripts/cron/rb_state_daily.sh`・`install_v6_state_cron.sh`）を止める。`rb_nk_*` を使うモデルも残っていないことが前提で、r04 系は `rb_nk_*` を外してある。
4. Jobs の state_in と `gcs_mirror.py` から flat を外す。品質の監視の `calculated` を外す。
5. `src/pipeline/megu_index/` を外から使う関数（`is_obstacle_race_name`・`PRIZE_TABLE`）を `src/utils/` などへ移し、パッケージと `src/scripts/data/` のめぐ指数の CLI（9 本）、テストを消す。`tests/conftest.py:138-139` の飛ばし一覧も直す。
6. データ・GCS・DB は消さない（下の表）。削除はユーザーの承認の後。

### データ・GCS・DB（消さない。削除はユーザーの承認の後）

| もの | 場所 | 扱い |
|---|---|---|
| めぐ指数の表 | `data/page_reference/tables/<年>/megu_index_flat.parquet`（2020〜2026 年、各 5〜6MB） | 予測に使う間は残す。段 C の後も、戻し方のために当面残す |
| GCS の正本 | `chuou/data/preprocessed/netkeiba/pc/megu_index/<年>/`・`ops/mirror` の tables の flat | 同上。削除はユーザーの承認が要る |
| 学習用の作り直し | `retrain_out/megu_2016`・`megu_asof`・`megu_frames`・`megu_v3a_ref*`・`data/local/modeling/megu_recompute_*` | 同上 |
| 表示用の状態 | `data/local/modeling/hem/`・`config/expected_megu/`・`config/megu_mark.json` | 段 A で使わなくなる。消すのは承認の後 |
| RB の状態 | `data/local/inference/state/rb_v6/`・`retrain_out/base_2016/rb_state` | RB を使うモデルが残る間は要る |
| DB | PG の `megu_index`・`megu_regression_params`・`megu_par_time`・`megu_condition_transfer`（`src/db/models.py:469-600`） | 係数は v2・v3a の計算の入力なので、計算を続ける間は残す。テーブルは消さない |

## 4. 費用

- 減る分は小さい（今は 0 円に近く、Jobs を再開した場合でも月 10 円前後）。
  - Jobs の state_in に入っているめぐ指数の flat（7 ファイル）: eve-precompute・predict-race・day-prep・px-ridge-window・rb-state-daily・hem-megu-state。これらの Scheduler は 2026-11-13 から止まっているので、今の費用は 0 円。再開した場合の上限は、1 日 700 回の実行がすべて 7 ファイルを読むとして 700 × 7 × 30 = 14.7 万回 × 0.06 円/千回 ≒ 月 9 円（ほかに取り込みの CPU・メモリの秒数）。東京リージョンの中なので転送料は無い。
  - Cloud Run の API の起動の束から flat（約 40MB）が外れる。リージョンの中なので転送料は無いが、起動が少し軽くなる。
  - 週次のホストの計算・GCS への書き出し（年の parquet 数個、Class A 数回）とホストの HEM の作り直し（週 1 回）は月 1 円未満。B1 ではホストの計算は残る。
- 増える分は 0 円。学習のやり直しはホストの GPU・CPU（電気代のみ）。GCS を繰り返し読む処理は足さない。
- 費用に効く変更ではない（上限 2,000 円/月に対して誤差）ので、撤廃の判断は費用ではなく、運用の弱さ（週次の遅れ・計算の間違いが RB に引きずられる）と保守の手間で行う。

## 5. リスクと戻し方

| リスク | 起きること | 防ぎ方 | 戻し方 |
|---|---|---|---|
| 段 C を早まる | v6e・r04（めぐ列あり）・sec77 が動いているのに RB の状態やめぐ指数が止まる → `v3_serving.py:283-285` で KeyError、または `rb_mg_n` が減って黙って精度が落ちる | 段 C は、めぐ列のないモデルが本番・stg・影のすべてで動いてから。`rb_state` の検査（件数・cut）を残す | `KEIBA_WEEKLY_MEGU_BATCH=1` に戻し、`compute_megu_for_opening_dates` で欠けた日を計算 |
| 段 A で API の項目を先に消す | 画面が項目を待って崩れる | 画面を先に外し、同じ配備か後で API を外す | git revert |
| 公開・会員の出し分けの一覧のずれ | 項目の漏れ・過剰な遮断 | tier.py・public_fields.json・paid_zone.py を同じ commit で直し、`test_tiered_access.py` を通す | git revert |
| 成績の照合 | 照合は保存済みの予測（`p_win`・印・`target_bets`）を使い、めぐ指数の項目は使わない（2026-11-18 に `src/` を `megu_mark`・`expected_megu`・`megu_index` で grep し、成績・照合のコードは 0 件。該当は研究用の `sire_line_stats.py`・`tp65_report.py` だけ） | 段 A の実装の前にもう一度 grep する | — |
| 品質の監視 | `quality_health.py` の `calculated`（めぐ指数のそろい）が段 C の後、または PG への書き込みを止めた後に fail になり、レース詳細に警告バッジ（`app.py:7081`）が出る。stg では自動修復が `compute_for_date` を繰り返す | PG の書き込みを止める変更・段 C と同じ commit で `calculated` を直すか外す | git revert |
| 品質の自動修復と flat | ミラーから flat を外した後に、自動修復（`quality_auto_remediation.py:354-363`）が Cloud Run の中で動くと、1 日分の表で GCS の年の正本を上書きしうる（`compute.py:701-760`） | ミラーから外す前に、その道が Cloud Run で動かないことを確かめるか、`compute_for_date` を外す | GCS のオブジェクトの世代から戻す（世代管理があるかを先に確かめる） |
| `src/pipeline/megu_index/` を外から使う箇所 | 段 C でパッケージを消すと壊れる: `race_quality_rules.py:12`・`quality_health.py:395`（`is_obstacle_race_name`）、`sire_line_stats.py:38`（`PRIZE_TABLE`）、`auto_scrape.py`、`quality_auto_remediation.py`、`src/scripts/data/` の 9 本 | 共有の関数を `src/utils/` などへ移してから消す | git revert |
| 今週末の配備 | r04＋第 2 段は `rb_mg_*` で学習済み。推論だけで列を空にすると学習とずれる | 今週末（11/21・22）は何も変えない（model-improve-cycles の意見） | — |

## 6. レビュー（別のエージェント）の指摘と対応

計画書を作った本人とは別のエージェントに、根拠・代わりの案・費用・リスク・見落としの観点で読ませた（2026-11-18）。

| 重要度 | 指摘 | 対応 |
|---|---|---|
| 高 | 「効いている」の根拠が、1 回だけの seed の対照との比べに頼っていて弱い（約 2σ）。対照と抜いた版で乱数のゆらし方が対称でない | 直した。乱数を変えた組で「めぐ列あり／なし」を作り、2 本目の推定 +0.00112（+0.00033〜+0.00192、3 年とも悪化）を得た。1 本目 +0.00107 と一致。§0・§2.2・§2.3 の表現を、2 本の推定と、乱数だけの差 2 点で書き直した |
| 高 | 先読みの対策は v3a の基準だけで、v2 の基準タイム（`megu_par_time`、2020〜2025 年で当てはめ）が残る | 直した（記載）。§2.3 に、影響の道（`front_split_dev`）と向き（効きを膨らませる）を書いた。測り直しはしていない。B1′（係数を年ごとの as-of で固定のファイルにする）で同時に解消できる |
| 中 | Jobs（rb-state-daily・hem-megu-state など）は 2026-11-13 から Scheduler ごと止まっている。RB と HEM はホストの cron | 直した。§1.1・§1.5・段 A・§4 を、今動いている場所（ホストの cron）で書き直した。費用は今 0 円、再開した場合の上限を別に書いた |
| 中 | r04 の RB は `retrain_out/cycles16_fix3/rb_live/`（リポジトリの外）で作っており、推論の基準（`config/megu_v3a`）と学習の基準（as-of）がずれる | 直した。§1.1 に流れとずれを書き、段 B の「残すもの」に、r04 の RB の作り方と基準をモデルの担当と決める項目を足した |
| 中 | flat を書く処理は週次だけではない（品質の自動修復）。ミラーから外した後に GCS の年の正本を 1 日分で上書きしうる。PG の書き込みを先に止めると自動修復が繰り返す | 直した。§1.5 に行を足し、段 A の 3（ミラーから外す前の確かめ）、段 B（PG の書き込みと `calculated` の直しを同じ commit に）、§5 のリスク表に書いた |
| 中 | 副の指標・クラス別の多重比較と、事後に決めた判定の基準 | 直した。主の指標は対数損失 1 本と明記し、副の指標とクラス別は記述だけにした。未勝利の説明は、2 本目で再現しなかったので採らないと書いた。判定の基準が事後であることも書いた |
| 中 | 棚卸しの漏れ（設定の一覧・テスト・表示用の v2 の部品・flat を読む研究用・docs の棚・コラムの別ファイル） | 直した。§1.6 と段 A に足した |
| 中 | 公開・会員の出し分けの一覧から項目名を消すと、残っていた項目が匿名に出る向きに壊れる | 直した。段 A では消さず、項目が出なくなったことを確かめてから掃除する |
| 低 | `gcs_mirror.py:60` は `:64` の誤り | 直した |
| 低 | 別の機能の成長曲線（`/growth-curve`）を一緒に消さないよう明記 | 直した（段 A の 1 の「対象外」） |
| 低 | `q_full − q_base` の採点の JSON が上書きされて残っていない | 直した。`q_full_vsbase` の名前で採点し直して残した（`r04/score/q_full_vsbase.json`） |
| 低 | 基準の設定はめぐ列ありで選ばれており、抜いた版は選び直していない | 直した（§2.3 の「残る弱み」に記載） |
| 提案 | B1′（計算を簡素化して残す）と、RB の前のめぐ指数の検査の関門 | 採った。段 B に B1′ を代わりの案として、関門を「運用の守り」として足した |

