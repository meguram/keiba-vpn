# TODO: feature/horse-profile

**対象領域**: 馬プロフィール・馬名検索・関係者統計
**関連ドキュメント**: [../feature-horse-profile.md](../feature-horse-profile.md)
**最終更新**: 2026-09-30

<!--
  このファイルの構成:
  - 「現状の実装」は自動生成＋手動確認した内容。コードを変更したら合わせて更新する。
  - 「既知の課題」は過去の調査で判明済みの問題（無ければ空のまま）。
  - 「TODO」は自由に追記してよい。優先度・担当・期限などは各行に自由書式で書いてよい。
  - 完了したTODOは削除せず [x] にチェックして残す（履歴として）。
-->

## 現状の実装（2026-09-30時点）

- `/api/horse/{horse_id}/detail`: 馬詳細情報（netkeiba + SmartRC + 統計）を集約して返す
- `/api/horse/{horse_id}/recent_races`: horse_resultのrace_historyから直近N走（新しい順）
- `/api/horse/{horse_id}/race_performance_history`: 戦績のうちrace_performance生成済みのものだけを返す
- `/api/person/{ptype}/{person_id}/stats`: 騎手・調教師の成績情報
- `/api/horse-names/index-meta`: 馬名インデックスの参照用メタ（パス・頭数・生成時刻）
- `/api/horse-names/search`: 馬名検索・候補返却
- ページURLは無し（`/api/*`のみで構成される機能領域）

## 目標（推測）

ユーザが馬名・騎手名・調教師名を検索すれば、その馬/人物の基本情報と成績がすぐ確認できること。
他の分析系ブランチ（race-detail・bloodline-pedigree等）が参照する基盤データとしての役割も
兼ねると推測される。

## このラインまで実装できたらブランチを消してよい

- 馬名検索・騎手/調教師統計で「見つからない」「情報が古い」といった事象がユーザ体験上
  問題にならないレベルまで解消されている
- 他ブランチ（race-detail等）から参照される際にレスポンスが安定している
  （タイムアウト・欠損データによる連鎖的な表示崩れが無い）
- 上記が実現できていれば、ユーザ向けの実装は完了したとみなせる

## 既知の課題

（無し）
- 騎手・調教師統計（`/api/person/{ptype}/{person_id}/stats`が参照するデータ）の生成は
  `scripts/cron/update_jockey_trainer_stats.sh`によるOS crontab定期実行が前提。GCPへ移行する場合、
  Compute Engineでのリフト&シフトなら同方式を継続できるが、Cloud Run等のサーバーレス構成を
  選ぶ場合はCloud Scheduler+Cloud Run Jobsでの再設計が必要になる（採用するGCPサービスにより
  対応が変わる）。詳細は
  [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)。
  現時点ではVPS運用継続が方針のため、本ファイルのTODOはそのまま進めてよい。

## TODO（手動追記用）

<!-- 「現状の実装」と「削除してよいライン」の差分から推測したTODO。実態を確認して要不要を判断すること。 -->

### 共通TODO（ホスト方式に関係ない）

- [ ] 馬名検索（`/api/horse-names/search`）の表記ゆれ対応（旧馬名・カタカナ表記違い等）を検証する
- [ ] 他ブランチ（race-detail等）からの参照時に、本APIのタイムアウト・エラーが連鎖的に
      表示崩れを起こしていないか確認する

### 常時稼働ホスト（VPS / GCP Compute Engine）の場合のTODO

- [ ] 騎手・調教師統計（`/api/person/{ptype}/{person_id}/stats`）の集計対象期間・更新頻度を明記する
      （`src.pipeline.build_jockey_trainer_stats`は現状OS crontab
      `scripts/cron/update_jockey_trainer_stats.sh`で定期実行。この前提での更新頻度を明記する）

### GCPサーバーレス（Cloud Run等）移行時のTODO

- [ ] 上記の統計更新頻度は、サーバーレス移行する場合はCloud Scheduler+Cloud Run Jobsでの
      実行頻度に置き換わる。詳細は
      [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)

## メモ
