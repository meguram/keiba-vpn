# TODO: feature/bloodline-pedigree

**対象領域**: 血統・種牡馬クラスタ
**関連ドキュメント**: [../feature-bloodline-pedigree.md](../feature-bloodline-pedigree.md)
**最終更新**: 2026-09-30

<!--
  このファイルの構成:
  - 「現状の実装」は自動生成＋手動確認した内容。コードを変更したら合わせて更新する。
  - 「既知の課題」は過去の調査で判明済みの問題（無ければ空のまま）。
  - 「TODO」は自由に追記してよい。優先度・担当・期限などは各行に自由書式で書いてよい。
  - 完了したTODOは削除せず [x] にチェックして残す（履歴として）。
  - この領域はエンドポイント数が最も多い（65件）。サブテーマごとに見出しを分けているので、
    実装を追加した場合は該当するサブテーマに追記する（無ければ新しい見出しを作ってよい）。
-->

## 現状の実装（2026-09-30時点）

### ページ
- `/bloodline`（統合Viewer）・`/bloodline-cluster`・`/bloodline-vector`（血統ベクトル空間v2）・
  `/pedigree-map`・`/note-aptitude-race`・`/pedigree-race-stats`
- `/course-bloodline`: 旧ページ、`/bloodline`へ308リダイレクト
- ナビゲーション: 全ページ共通の `templates/partials/_nav.html`（全血統・pedigreeテンプレートが`{% include %}`済み）に
  常設グローバルナビがあり、「血統」ドロップダウンから`/bloodline`・`/bloodline-vector`・`/pedigree-map`・
  `/bloodline-cluster`・`/pedigree-race-stats`・`/myostatin`・`/note-aptitude-race`（2026-10-01追加）に、
  「データ分析」ドロップダウンからも`/note-aptitude-race`に遷移できる（相互リンク済み）。

### 血統分析（bloodline/*）
- `POST /api/bloodline/analyze`（バックグラウンド実行）・`/status`・`/surfaces`・
  `/data/{analysis_type}`（芝・ダート・障害別CSV/JSON読み込み）

### 種牡馬クラスタ（bloodline-cluster/*）
- 検索・参照系: `lookup`（馬名→プロファイル）・`lookup-by-id`・`sire-info`・`suggest`・`horse-name-suggest`・
  `clusters`（全L2メタクラスタ）・`tags`（特殊適性タグ一覧）
- 統計系: `stats`（期間×舞台条件×種牡馬/クラスタ）・`sire-presence-stats`・`sire-heatmap`・
  `sire-summary-card`・`sire-best-conditions`・`sire-presence-horses`・`horse-aptitude`
- 管理系: `POST reload`（アーティファクト再ロード）・`admin/artifact-status`・
  `POST admin/rebuild/{target}`・`admin/job-status`・`POST admin/reload`

### 種牡馬ツリー（stallion-sire-tree/*）
- `GET`（軽量50ノードツリー）・`l1-groups`（4大主流+非主流）・`roots`（父不明ノード）・
  `node/{horse_id}`（子リスト、ページネーション対応）・`bms-stats/{horse_id}`・`search`
- `POST rebuild` + `/rebuild/status`: 全種牡馬ツリーのバックグラウンド再構築

### pedigree-map（適性タグ・条件ランキング）
- `GET`・`tags`・`tags-full`（~400頭全種牡馬タグマップ）・`cluster-hierarchy`（L1×L2×L3階層）・
  `condition-ranking`（探索モードメインAPI）・`progeny-under-condition`（該当馬展開）

### note血統ナレッジ（pedigree/note-aptitude, race-note-3d系）
- `note-aptitude`（父・母父・牝系の簡易スコア）・`note-aptitude/table`（適性表全行）・
  `race-note-3d`・`race-note-3d-compare`・`race-note-3d-v2`（血統メタクラスタベース）・
  `week-races`（今週開催、note-aptitude-raceのデフォルト選択肢）

### 種牡馬因子統計・チューニング
- `POST rebuild-sire-factor-stats`（fast/full）・`POST tune-weights`（scope=global/race）

### 5代血統整備（race-ensure-5gen系）
- `POST race-ensure-5gen`（欠損調査+取得）・`/status`（session_id進捗）・`POST /cancel`・
  `POST batch-race-ensure-5gen`（開催日範囲一括）

### コース×血統（course-bloodline/*）
- `POST analyze`・`/status`・`/surfaces`・`/data/{analysis_type}`、`/api/course-profiles`（ドメインナレッジ）

### pedigree-race-stats
- `/meta`（フィルタ用メタ）・`/query`（血統カテゴリ別種牡馬カウント分布、10世代対応）・`/lineage-meta`

## 目標（推測）

ユーザが「この馬・この種牡馬はどの条件（コース・距離・馬場）で買いか」を血統情報から
判断できること。血統は競馬予想の重要な軸の一つであり、ユーザに直感的に伝わる可視化
（クラスタ・ヒートマップ・条件ランキング）を提供することが目標と推測される。

## このラインまで実装できたらブランチを消してよい

- ユーザが馬名・種牡馬名を入力すれば、条件別の適性・ベスト条件・過去成績が迷わず分かる
- 主要な可視化（bloodline-cluster・pedigree-map・種牡馬ツリー）がアーティファクト未生成で
  「データ無し」表示になる頻度が実質ゼロ（定期rebuildが機能している）
- サブテーマ（note血統ナレッジ・5代血統整備・course-bloodline等）ごとに、少なくとも
  「エラーなく結果が返る」レベルの安定性がある
- 上記が実現できていれば、ユーザ向けの実装は完了したとみなせる

## 既知の課題

（無し。ページURL・API双方とも他ブランチとの分類重複は見つかっていない）
- 下記TODOの「定期rebuildスケジュールの整備」は、VPSなら既存のOS crontab/daemon thread方式
  でそのまま整備できる。GCPへ移行する場合も、Compute Engineでのリフト&シフトなら同じ方式を
  概ねそのまま使えるが、Cloud Run等のサーバーレス構成を選ぶ場合はCloud Scheduler+Cloud Run
  Jobsでの再設計が必要になる（採用するGCPサービスにより対応が変わる）。方式検討前に
  [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)を参照。
  現時点ではVPS運用継続が方針のため、本ファイルのTODOはそのまま進めてよい。

## TODO（手動追記用）

<!-- 「現状の実装」と「削除してよいライン」の差分から推測したTODO。実態を確認して要不要を判断すること。 -->

### 共通TODO（ホスト方式に関係ない）

- [x] `/bloodline`・`/bloodline-cluster`・`/pedigree-map`・`/note-aptitude-race` 等、
      サブテーマごとに別ページに分かれているUIの統合・ナビゲーション改善を検討する
      （`/course-bloodline`は既に`/bloodline`へリダイレクト統合済み）
      — 2026-10-01対応: 調査の結果、全ページは既に共通の`templates/partials/_nav.html`を
      `{% include %}`しており（`bloodline.html`・`bloodline_cluster.html`・`bloodline_vector.html`・
      `pedigree_map.html`・`pedigree_race_stats.html`・`race/note_aptitude_race.html`で確認）、
      常設グローバルナビの「血統」ドロップダウンから相互に1クリックで遷移可能だった。
      新規ページ作成・ルーティング再設計は不要と判断。唯一のギャップとして、
      `/note-aptitude-race`は血統メタクラスタベースの機能であるにもかかわらず
      「データ分析」ドロップダウンにのみ掲載されていたため、`_nav.html`の「血統」ドロップダウンにも
      同ページへのクロスリンクを追加した（`templates/partials/_nav.html`編集、新規ページ・新規ルート無し）。
      結論として「現状のグローバルナビ構成を維持しつつ1件のクロスリンクを追加」が適切な対応。
- [ ] 5代血統整備（race-ensure-5gen系）が未完了の馬の割合を計測する
      （計測自体はホスト方式に依存しない。計測後の定期実行整備は下記を参照）
      — 2026-10-01調査: 計測ロジック自体は既存の`src/scraper/verify_horse_scrape_completeness.py`
      （`verify_horses_for_race_period(..., tasks=["horse_pedigree_5gen"])`、`horse_pedigree_5gen`の
      `ancestors[]`が5件以上あるかで判定）がそのまま使えると確認した。しかし本対応を行った作業環境では
      実データにアクセスできず計測を完走できなかった: (1) `data/`配下に`horse_pedigree_5gen`・
      `data/features/horse/ped_tbl`等の実体が存在しない（`data/local/`は`image`/`meta`のみ、
      `data/page_reference/`も`BUNDLE.md`のみでレース一覧すら無い）、(2) `.env`に`GCS_BUCKET`等の
      GCS認証情報が設定されていない、(3) `gcloud auth list`では既存アカウントが認証済みだったが
      `gsutil`での対象バケット読み取りは403で拒否された、(4) `data/queue/scrape_queue.json`に
      残っていたジョブはヘルスチェック用のダミーレース（`race_id=999901019999`等）のみで実馬データではない。
      以上により母集団（直近1年の出走馬など）を定義して計測する前提のデータ自体が本環境に無く、
      「計測不可能なほどデータが欠けている」ケースと判断してスキップした。GCS権限のあるVPS/本番環境では
      下記コマンドで即座に測定できる（`total_horses`に対する`missing_count_by_task.horse_pedigree_5gen`の
      比率がそのまま未完了率になる）:
      `python3 -m src.scraper.verify_horse_scrape_completeness --start-date 20250101 --end-date 20251001 --tasks horse_pedigree_5gen`

### VPS側（サービング）に残るTODO

（本ファイルでは該当なし。アーティファクトrebuild・5代血統整備はいずれも重い処理のためGCP側へ移動。
VPS側は読み取り系配信のみ）

### GCP側（スクレイピング・ML・スケジュール実行）のTODO

- [x] bloodline-cluster / pedigree-map / stallion-sire-tree 等、複数アーティファクトの
      定期rebuildスケジュールの有無を確認し、無ければ整備する
      — 2026-10-02対応: `src/api/app.py`の`_BLOODLINE_ARTIFACTS`に列挙された各rebuilderモジュール
      （`build_pair_lift_profiles`・`build_role_lift_profiles`等）・種牡馬ツリー
      （`build_full_sire_tree`）はいずれも既存の`python -m`直接実行に対応済みであることを確認し、
      Cloud Scheduler + Cloud Run Jobsの実行コマンドとして
      [`docs/operations/gcp-cloud-run-jobs.md`](../../operations/gcp-cloud-run-jobs.md)の
      ジョブ#11・#12に追記した。種牡馬ツリーの`relevant_stallion_ids`再生成ステップ
      （`src/api/app.py`内のprivate関数`_regenerate_relevant_stallion_ids`）はCLI化未対応のまま
      残っており、別途切り出しが必要（既知の課題として記録）。
- [x] 5代血統整備が未完了の馬が多い場合、`batch-race-ensure-5gen`の定期実行を整備する
      — 2026-10-02対応: 実処理関数`batch_race_pedigree_5gen_date_range`
      （`src/research/pedigree/race_pedigree_5gen_prefetch.py`）を特定し、
      [`docs/operations/gcp-cloud-run-jobs.md`](../../operations/gcp-cloud-run-jobs.md)の
      ジョブ#13として記録。直接呼べるCLIラッパーは未作成（`python -c`呼び出しか
      `src/scripts/cloud_jobs/`への薄いラッパー追加が今後必要）。

## メモ
