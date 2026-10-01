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
- `/api/race-quality/entrants-aptitude`: 指定レースの出走馬ごとの8軸適性スコア（血統＋戦歴キャッシュ利用）

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

- [ ] 血統・戦歴データが欠損している馬について、entrants-aptitude がエラー落ちせず
      妥当なフォールバック値を返すか確認する
- [ ] レース質推定の精度（実際の決着との相関）を検証する

### 常時稼働ホスト（VPS / GCP Compute Engine）の場合のTODO

- [ ] `/api/race-quality/day`（日次一括推定）が自動実行（バッチ/cron）されているか確認し、
      無ければOS crontab方式で整備する（現状はAPI呼び出しのみで、いつ計算されるかが不明）

### GCPサーバーレス（Cloud Run等）移行時のTODO

- [ ] 上記の自動実行整備は、サーバーレス移行する場合はCloud Scheduler+Cloud Run Jobsでの
      実装が前提になる。詳細は
      [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)

## メモ
