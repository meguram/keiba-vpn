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

## TODO（手動追記用）

<!-- 「現状の実装」と「削除してよいライン」の差分から推測したTODO。実態を確認して要不要を判断すること。 -->

- [ ] bloodline-cluster / pedigree-map / stallion-sire-tree 等、複数アーティファクトの
      定期rebuildスケジュールの有無を確認し、無ければ整備する（現状はいずれも手動`POST rebuild`系）
- [ ] `/bloodline`・`/bloodline-cluster`・`/pedigree-map`・`/note-aptitude-race` 等、
      サブテーマごとに別ページに分かれているUIの統合・ナビゲーション改善を検討する
      （`/course-bloodline`は既に`/bloodline`へリダイレクト統合済み）
- [ ] 5代血統整備（race-ensure-5gen系）が未完了の馬の割合を計測し、必要なら
      `batch-race-ensure-5gen`の定期実行を整備する

## メモ
