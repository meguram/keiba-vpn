# TODO: feature/race-quality

**対象領域**: レース質分析
**関連ドキュメント**: [../feature-race-quality.md](../feature-race-quality.md)
**最終更新**: 2026-09-30

<!--
  このファイルの構成:
  - 「現状の実装」は自動生成＋手動確認した内容。コードを変更したら合わせて更新する。
  - 「既知の課題」は過去の調査で判明済みの問題（無ければ空のまま）。
  - 「TODO」は自由に追記してよい。優先度・担当・期限などは各行に自由書式で書いてよい。
  - 完了したTODOは削除せず [x] にチェックして残す（履歴として）。
-->

## 現状の実装（2026-09-30時点）

- `/race-quality`: レース質分析ページ
- `/api/race-quality/meta`: レース質8軸の定義・セグメントキー一覧などの固定メタデータ
- `/api/race-quality/day`: 指定日（YYYYMMDD）の全JRAレースのレース質を一括推定
- `/api/race-quality/race`: 単一レースのレース質ベクトル（9確率）とメタ情報
- `/api/race-quality/entrants-aptitude`: 指定レースの出走馬ごとの8軸適性スコア（血統＋戦歴キャッシュ利用）。
  血統(`horse_pedigree_5gen`)・戦歴(`horse_result`)が未取得の馬は `storage.load` が `None` を返し、
  `_compute_aptitude_fast` がゼロ血統ベクトル・`_history_features([])` が既定値（`last3f_fast=0.5`等）を
  返すことでエラー落ちせず続行する（`src/research/race/race_quality_model.py` の
  `build_horse_aptitude_cache_payload` / `build_entrants_aptitude_response`）。
  回帰テスト: `tests/research/test_race_quality_model.py` の
  `test_aptitude_payload_missing_pedigree_and_history_has_fallback` 等（2026-10-01追加）。

## 目標（推測）

ユーザが「このレースは実力通りに決まりやすいか、荒れやすいか」を8軸のレース質指標で
事前に把握し、馬券判断の参考にできることが目標と推測される。

## このラインまで実装できたらブランチを消してよい

- 対象レース（少なくとも中央競馬全レース）で当日〜前日にはレース質推定が出ており、
  ユーザが「未計算」に当たらない
- 出走馬ごとの8軸適性スコアが血統・戦歴データの欠損時にもエラー落ちせず表示される
- 上記が実現できていれば、ユーザ向けの実装は完了したとみなせる

## 既知の課題

- 下記TODOの「`/api/race-quality/day`の自動実行（バッチ/cron）整備」は、VPSなら既存のOS
  crontab方式に乗せればよい。GCPへ移行する場合も、Compute Engineでのリフト&シフトなら
  同方式を継続できるが、Cloud Run等のサーバーレス構成を選ぶ場合はCloud Scheduler+Cloud Run
  Jobsでの実装が前提になる（採用するGCPサービスにより対応が変わる）。詳細は
  [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)。
  現時点ではVPS運用継続が方針のため、本ファイルのTODOはそのまま進めてよい。

## TODO（手動追記用）

<!-- 「現状の実装」と「削除してよいライン」の差分から推測したTODO。実態を確認して要不要を判断すること。 -->

### 共通TODO（ホスト方式に関係ない）

- [x] 血統・戦歴データが欠損している馬について、entrants-aptitude がエラー落ちせず
      妥当なフォールバック値を返すか確認する — 2026-10-01対応: コードを読んだ上で
      `_FakeStorage`（`horse_pedigree_5gen`/`horse_result`/`race_index`/`race_barometer`等を
      すべて`None`で返す）を使って `build_horse_aptitude_cache_payload` /
      `get_horse_aptitude_cache` / `build_entrants_aptitude_response` を直接呼び出し、
      さらにローカルサーバー（`python3 main.py --port 8099`）を一時起動して
      `GET /api/race-quality/entrants-aptitude?race_id=NOTAREALRACEID` を curl で確認した
      （検証後サーバー停止）。結果: 血統・戦歴が丸ごと欠損していてもゼロ血統ベクトル＋
      既定の戦歴統計（`last3f_fast=0.5`等）にフォールバックし例外は発生しない。
      存在しない race_id は 500 ではなく
      `{"error": "出走馬データなし（出馬表・レース結果未取得）"}` の 404 を返す（既存実装のまま、
      `src/api/app.py` の例外ハンドラも含めて妥当）。空文字列・短すぎる `horse_id` を持つ
      entry は `continue` で静かにスキップされ出力に含まれない（エラーにはならないが
      スキップされた件数は返却値に出ない。実運用の scraped データでは horse_id が空になる
      ケースは想定しづらく、クラッシュしない以上は許容範囲と判断し、今回はコード修正なし）。
      コード変更は行わず、この挙動を固定化する回帰テストのみ追加
      （`tests/research/test_race_quality_model.py` に4件追加、既存含め
      `tests/research/test_race_quality_model.py` 14件・`make test` 相当のフルスイート
      469件すべてpass）。
- [ ] レース質推定の精度（実際の決着との相関）を検証する — 2026-10-01対応:
      この作業環境では検証不可能と判断してスキップ。理由: (1)
      `analyze_race`/`collect_race_xy_tensors` が必要とする実データ
      (`race_result`/`horse_pedigree_5gen`/`race_barometer`/`race_index`/`race_lap`等)は
      GCSにのみ存在し、ローカル`data/`配下にはキャッシュが一切ない（`data/local`は
      `meta`のみ、`data/page_reference`は空、`data/features/`自体が存在しない）。
      (2) 本セッションの`.env`には`GCS_BUCKET`が設定されておらず
      （`SLACK_WEBHOOK_URL`のみ）、`HybridStorage.gcs_enabled`はFalseになる。
      (3) `gcloud`にADCは存在するが、アクティブなアカウント・プロジェクトは
      本プロジェクトの想定バケット（`.env.example`の`magu-keiba-horse-racing-ai`）とは
      無関係な別アカウント・別プロジェクト（Rakuten法人アカウント／
      `ccbd-ecbdp-bds-prod`）のものであり、ユーザ不在のままこの資格情報を使って
      本プロジェクト外のGCPリソースへアクセスを試みるのは越権と判断し、実施しなかった。
      (4) ローカルに実データのparquetは`notebooks/megu_index/output/`配下に存在する
      （`megu_dataset.parquet`等、302,357行・2020年以降の実`race_id`/`finish_pos`/
      `last_3f`等を含む）が、これは別の時間指数(TSI)研究パイプラインの中間データであり、
      race_quality_model が必要とする血統適性・戦歴・バロメーター指数等の説明変数を
      含まない。これらを使うと全馬が同一のゼロ・フォールバックベクトルになり
      （項目1で確認した挙動）、NNLS混合比の推定自体が無意味になるため、
      「検証した」と称するのは誤解を招くと判断し実施しなかった。
      本項目は、`GCS_BUCKET`を本プロジェクト用に設定したうえで実データにアクセスできる
      環境（ユーザー自身の環境）で改めて実施する必要がある。その際は
      `src/research/race/tune_race_quality_priors.py`（NNLS残差最小化・既存のセグメント
      チューニングスクリプト）のサンプリング方式を参考に、直近の確定済みレース
      （数十〜数百件）で `analyze_race` の `r2_fit`／予測軸と、好走馬の人気・オッズ等の
      実決着傾向を比較する、という方針を推奨する。

### VPS側（サービング）に残るTODO

- [x] `/api/race-quality/day`（日次一括推定）が自動実行（バッチ/cron）されているか確認し、
      無ければ整備する
      — 2026-10-02対応: 現状はAPI呼び出しのみで自動実行なしと確認。本機能はGCP側（役割分担
      マッピング: 配信=VPS、日次一括推定の実行=GCP）に移行する方針のため、OS crontabでの整備は
      行わず、Cloud Scheduler + Cloud Run Jobsの実行コマンドとして
      [`docs/operations/gcp-cloud-run-jobs.md`](../../operations/gcp-cloud-run-jobs.md)の
      ジョブ#6（`python -m src.scripts.cloud_jobs.race_quality_day`）に記録済み。

### GCP側（スクレイピング・ML・スケジュール実行）のTODO

- [x] 上記の自動実行整備は、サーバーレス移行する場合はCloud Scheduler+Cloud Run Jobsでの
      実装が前提になる。詳細は
      [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)
      — 2026-10-01対応: `/api/race-quality/day`が呼ぶ`analyze_date()`をそのまま呼び出す
      CLI エントリポイント`python -m src.scripts.cloud_jobs.race_quality_day`（新規）を追加し、
      実行コマンド・想定頻度（元は固定cron無し。提案値: 毎日19:00 JST）・リソース目安・
      `gcloud scheduler jobs create http`登録コマンド例を
      [`docs/operations/gcp-cloud-run-jobs.md`](../../operations/gcp-cloud-run-jobs.md)に
      まとめた。デプロイ設計図は
      [`scripts/gcp/deploy_cloud_run_jobs.sh`](../../../scripts/gcp/deploy_cloud_run_jobs.sh)。
      実デプロイ・スケジューラ登録はユーザー側作業として残る

## メモ
