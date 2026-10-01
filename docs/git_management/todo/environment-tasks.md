# 環境別TODO（開発PC / 学習PC / VPS）

**最終更新**: 2026-10-02（WSL切断後に再開。現状のコードと確定アーキテクチャにもとづく）
**位置づけ**: 機能領域別のTODO（同ディレクトリの各ファイル）を、「どの環境で作業するか」で横断的に並べ直したもの。
詳細な設計は [vps-gcp-responsibilities.md](../../operations/vps-gcp-responsibilities.md)、調査スクリプトは [diagnostics.md](../../operations/diagnostics.md)。

## 環境の定義

**呼び名の対応**: 学習PC ＝ 旧称「別PC」（本物のデータ・GCP接続・学習をするPC）。開発PC ＝ Claude Code を動かしているこのPC。文書内の「別PC」は「学習PC」に統一済み。

| 環境 | 何を指すか | できること | できないこと |
|---|---|---|---|
| **開発PC** | Claude Code で作業しているPC。コードを書きテストを回す | 実装・単体テスト・モックデータでの動作確認・ドキュメント | 本物の特徴量ストア/モデルが無い。GCS・VPS・GCPに接続しない（GCPに触るコマンドは**書くだけ**で、実行は学習PC） |
| **学習PC** | 本物のデータ・特徴量ストア・学習済みモデルがあり、**現状、GCPへの接続ラインが確保できている唯一の環境** | **データ準備の全部**（スクレイピング、特徴量生成とそのコーディング）、学習、実データでの計測、**GCPに触る作業すべて**（GCS読み書き・Cloud SQL作成・ジョブのデプロイ・Scheduler登録・課金エクスポート・モデル公開） | 常時起動ではない |
| **VPS** | ConoHa 2GB/3コア/SSD 100GB（契約済み） | 常時稼働。ページ配信・API・Redis・監視・開催日の当日取得・T-45推論 | メモリが小さい（推論の余裕は実測待ち）。**GCP接続は未確立**（`GCS_*` の設定が必要。E1） |
| **GCP** | Cloud SQL / GCS / Cloud Run Jobs ＋ Scheduler | 定期集計ジョブ、データの正本（GCS） | 操作は学習PCから（接続ラインがあるため） |
| **ユーザー作業** | 認証情報の設定・契約・コンソール操作 | | コードでは代行できない |

**分担の前提（2026-10-02 確認済み）**
- 過去データの収集（スクレイピング）と特徴量生成は**学習PC**で実行する。保存先はGCS（`HybridStorage`）。
- **開催日の当日分**（出馬表更新・馬体重・オッズ）だけはVPSのcronで取得する（学習PCは常時起動でないため）。
- 定期集計ジョブ（騎手調教師統計・種牡馬集計・レース質など）は従来どおり**GCP Cloud Run Jobs**。
- **GCPへの接続は現状、学習PCのみ**。GCS・Cloud SQL・Cloud Run Jobs・Scheduler・BigQuery に触る作業は学習PCで実行する。開発PCはコマンド・スクリプトを用意するところまで。VPSは `GCS_*` を設定するまで GCS を読み書きできない（ページ配信・T-45推論・当日取得の前提なので、設定を先に行う）。

---

## A. 開発PC（コードで完結するもの）

| # | TODO | 状態・メモ |
|---|---|---|
| A1 | **集計ジョブの出力をGCS化**（追走難度・track-speed・ミオスタチン・騎手調教師統計・血統アーティファクトは出力がローカルファイルのみ） | ⚠️GCPで実行する前に必須。最優先 |
| A2 | `publish_model` のCLI化（学習PCからモデルをGCSへ公開するコマンド） | `model_registry.publish_model` は実装済み。CLIのみ未 |
| A3 | ページ単位の集約オブジェクト＋先読み（prewarm）エンドポイント、`/race/{id}` から着手 | `FeatureStore.load_rows_for_keys` は実装済み（推論の読み込み用） |
| A4 | cronを `docker compose run --rm` 方式へ移行（`scripts/cron/setup_all_cron.sh` は現状ホストのPython前提）＋VPSに置くのは当日分の取得cronだけに絞る | VPSでの動作確認はC列 |
| A5 | `pre_race_predict_trigger.py` の本実装（T-45起動のcron登録含む） | race-detail.md の既存TODO。`race_day_workflow` は実装済み |
| A6 | `/api/race/{id}/predictions` のGCS/PostgreSQL二重経路の整理 | race-detail.md の既存TODO |
| A7 | 特徴量ビルダーの**インターフェース固定**（`build(race_data)`、増分計算の契約、学習と同じ入力処理の共有）とテスト | 本体の実装は学習PC（B1）。ここでは枠と検証用の疑似データのみ |
| A8 | 調査結果の反映：`decide` の出力を各TODO・設計書に反映 | B/Cのレポート待ち |
| A9 | 母馬ページ(種付け日)機能のドキュメント・要件表への行追加（`requirement_row_catalog` と `scrape_process.md`） | 実装・テストは完了。文書の同期が未 |
| A10 | cushion の `admin/sync-*` をGCP Jobs側へ寄せるか判断 | cushion.md の既存TODO |
| A11 | コミット・push（未コミットの変更あり） | 依頼があり次第 |

## B. 学習PC（データ準備・学習・実データの計測）

| # | TODO | 状態・メモ |
|---|---|---|
| B1 | **特徴量ビルダーの本実装**（約1000特徴量、増分計算、全件再計算しない）。コーディングも学習PCで行う | 現状は疑似ビルダー。A7の枠に実装を載せる |
| B2 | **過去データの収集**（レース・馬・血統・調教）。保存先はGCS | 既存のキュー/スクレイパーで実行 |
| B3 | **種付け日の取得**: 2024年以降生まれの馬に `horse_mating_date` タスクを投入（母馬ページ own.netkeiba） | 実装済み。実行のみ。母馬ごとに1リクエスト |
| B4 | 学習（LightGBM/XGBoost/CatBoost/MLP＋LRメタ）→ `publish_model` でGCSへ公開（学習PCはGCPに接続できるため、ここで公開まで実行できる） | A2のCLI待ち |
| B5 | **調査スクリプトの実行**: `diagnose_training_pc`（特徴量ストアの実サイズ、行フィルタ読み込みの実測、モデル容量、ライブラリ版） | **大規模データの全結合（`--feature-sample` を大きくする）はメモリを大量に使う。既定の200列で実行** |
| B6 | **動的特徴量のラベル付け**: `reports/dynamic_feature_labels.csv` の `label` 列に dynamic/static を記入 | T-45にしか確定しない特徴量（オッズ・馬体重・馬場・取消・騎手変更など） |
| B7 | 5代血統の整備率を計測（race-ensure-5gen系が未完了の馬の割合） | bloodline-pedigree.md の既存TODO |
| B8 | レース質推定の精度検証（実際の決着との相関） | race-quality.md の既存TODO |
| B9 | 学習に使うライブラリ版を固定して記録（VPS/Dockerイメージと揃える基準） | B5のレポートで比較される |

## C. VPS（実機でしか分からないこと・運用）

| # | TODO | 状態・メモ |
|---|---|---|
| C1 | **netkeiba試験取得**: `diagnose_vps --netkeiba-trial`（VPSのIPで1リクエスト）。遮断されたら当日取得の方式を再検討 | 当日分の取得をVPSに置く前提の条件 |
| C2 | **推論メモリの実測**: `diagnose_vps --model-dir <モデル>`（開催日の繁忙時間帯にも1回） | 結果で「VPS推論 or GCP切替」を決定（`decide`） |
| C3 | スワップ2GBの追加、推論は `--memory=700m`・`nice` で実行 | C2の結果しだい |
| C4 | GCS/Cloud SQLの遅延を実測（`diagnose_vps`） | ページ集約オブジェクト（A3）の要否判断 |
| C5 | 当日取得cronの設置（A4の方式で）と、ジョブ停止の監視（watchdog＋Slack通知） | A4の後 |
| C6 | kick/recover/stop-and-clear など緊急系エンドポイントの使用頻度、`/api/odds/snapshot` の記録頻度を計測 | scraping-queue.md / odds-final-odds.md の既存TODO。運用ログから |

## D. GCP

| # | TODO | 状態・メモ |
|---|---|---|
**D列はすべて学習PCで実行する**（GCPに接続できるのが学習PCのみのため）。開発PCはスクリプトの整備まで。

| # | TODO（実行場所: 学習PC） | 状態・メモ |
|---|---|---|
| D1 | Cloud SQL作成（`scripts/gcp/setup_cloud_sql.sh`、stg/prod共用1台。まず表示のみで内容確認→`--apply`） | ユーザーの承認後 |
| D2 | 集計ジョブのデプロイ（`scripts/gcp/deploy_cloud_run_jobs.sh`）とScheduler登録 | **A1（出力のGCS化）完了後**。バケットとCloud Run Jobsは同じリージョンにする（GCS読み出しが無料になる） |
| D3 | 課金データのBigQueryエクスポート有効化 → `GCP_BILLING_BQ_TABLE` 設定（日次コストSlack通知） | ユーザー作業（コンソール）＋学習PCで確認 |
| D4 | VPS用のサービスアカウントキー（GCS読み書き権限のみ）を発行し、VPSの `.env.prod` に `GCS_*` を設定 | 最小権限。学習PCで発行→ユーザーがVPSへ設定 |

## E. ユーザー作業（コードでは代行不可）

- E1 `.env` / `.env.stg` / `.env.prod` の `GCS_*`（サービスアカウント）設定。学習PCは接続済み。**VPSは未設定**（D4）。
- E2 GitHub Secrets の登録（CI/CDでのデプロイ用）。
- E3 課金エクスポートの有効化（D3）、Cloud SQLの作成承認（D1）。
- E4 **T-45再予測**（T-15のオッズ取得後にもう一度予測するか）の方針決定。
- E5 動的特徴量のラベル記入（B6）。

---

## 実行順序の目安

1. 開発PC: **A1**（GCS化）・**A2**（公開CLI）・A7（ビルダーの枠）・A4〜A6
2. 学習PC: **B5・B6**（調査とラベル）→ レポートを開発PCで `decide`（A8）。GCP作業（D1〜D4）もここから
3. VPS: **D4の後**に **C1・C2・C4**（試験取得・メモリ・遅延。C4のGCS遅延は `GCS_*` 設定後でないと測れない）→ レポートを `decide`
4. 判断結果しだいで: 推論場所（C3 または GCP切替）、集約オブジェクトの要否（A3）
5. 学習PC: B1→B2/B3→B4、公開後にVPSでT-45を通しで確認

## 注意（負荷の高い作業）

WSL/IDEを巻き込んで落とさないため、開発PCでは**大きな合成データ（数GB・1000列の全結合など）を作らない**。
本番規模の計測は学習PCで、既定の列数（200）のまま実行する。
