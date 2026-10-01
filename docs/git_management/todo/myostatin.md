# TODO: feature/myostatin

**対象領域**: ミオスタチン遺伝子解析
**関連ドキュメント**: [../feature-myostatin.md](../feature-myostatin.md)
**最終更新**: 2026-09-30

<!--
  このファイルの構成:
  - 「現状の実装」は自動生成＋手動確認した内容。コードを変更したら合わせて更新する。
  - 「既知の課題」は過去の調査で判明済みの問題（無ければ空のまま）。
  - 「TODO」は自由に追記してよい。優先度・担当・期限などは各行に自由書式で書いてよい。
  - 完了したTODOは削除せず [x] にチェックして残す（履歴として）。
-->

## 現状の実装（2026-09-30時点）

- `/myostatin`: ページ表示
- `/api/myostatin`: ミオスタチン遺伝子型ナレッジベース（`MYOSTATIN_GENES_JSON`）の内容を返す
- `POST /api/myostatin/predict`: 父・母父（+距離）を指定し、`MyostatinLookup`で種牡馬情報・産駒予測・特徴量を返す
- `POST /api/myostatin/recalculate`: 全ての未確定馬のミオスタチン遺伝子型を血統から再計算

## 目標（推測）

ユーザが「この馬（配合）は短距離向きか長距離向きか」をミオスタチン遺伝子型という科学的根拠
付きの追加軸で判断できることが目標と推測される。血統・戦績だけでは見えない適性を補完する
位置づけと推測される。

## このラインまで実装できたらブランチを消してよい

- 主要な父・母父についてミオスタチン遺伝子型情報がナレッジベースに揃っている
  （「情報無し」表示になる主要種牡馬が実質無い）
- 未確定馬の遺伝子型再計算が定期的に走り、新規馬が長期間「不明」のままにならない
- 上記が実現できていれば、ユーザ向けの実装は完了したとみなせる

## 既知の課題

- 下記TODOの「`recalculate`の定期実行有無の確認・スケジュール化」は、VPSなら既存のOS crontab
  方式に乗せればよい。GCPへ移行する場合も、Compute Engineでのリフト&シフトなら同方式を
  概ね継続できるが、Cloud Run等のサーバーレス構成を選ぶ場合はCloud Scheduler+Cloud Run Jobs
  での再設計が必要になる（採用するGCPサービスにより対応が変わる）。詳細は
  [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)。
  現時点ではVPS運用継続が方針のため、本ファイルのTODOはそのまま進めてよい。

## TODO（手動追記用）

<!-- 「現状の実装」と「削除してよいライン」の差分から推測したTODO。実態を確認して要不要を判断すること。 -->

### 共通TODO（ホスト方式に関係ない）

- [ ] 主要種牡馬のうちミオスタチン遺伝子型情報が「不明」になっている割合を計測する
- [ ] 予測（`/api/myostatin/predict`）の根拠・信頼度をユーザに分かりやすく提示できているか確認する

### 常時稼働ホスト（VPS / GCP Compute Engine）の場合のTODO

- [ ] `POST /api/myostatin/recalculate`（未確定馬の再計算）が定期実行されているか確認し、
      無ければOS crontab方式でスケジュール化する

### GCPサーバーレス（Cloud Run等）移行時のTODO

- [ ] 上記のスケジュール化は、サーバーレス移行する場合はCloud Scheduler+Cloud Run Jobsでの
      実装が前提になる。詳細は
      [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)

## メモ
