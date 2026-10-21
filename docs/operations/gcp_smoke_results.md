# GCP(Cloud Run Jobs) から netkeiba への到達試験の記録

仕様と合格条件は `architecture_target.md` 7.4、実行手順は `gcp_setup_runbook.md` 9 章。
**実行は Cloud Shell で `bash deploy/gcp/netkeiba_smoke/run.sh <枠>`**（公開ページのみ・ログインなし・30 リクエスト・4.5〜7 秒間隔・403/429 で即中止）。

## 実行前の確認（毎回）
- [ ] ホストの取得ジョブが止まっている（DEC-023: 同時にアクセスしない）: `ssh keiba-remote "pgrep -af '[a]uto_scrape' || echo 動いていない"`
- [ ] 枠が重複していない（同じ日に続けて 2 回やらない。**3 回は日・時間帯を変える**）
- [ ] メンテナンス窓（毎日 JST 03:00〜04:00）を避けている

## 推奨の実行枠（JST）
| 枠 | いつ | 狙い |
|---|---|---|
| `weekday_noon` | 平日の昼（12:00〜14:00） | 通常時 |
| `raceday_morning` | 開催日（土日）の朝（7:00〜9:00） | 開催日の負荷下 |
| `night` | 夜間（23:00〜翌 2:00） | 夜間の取得想定 |

## 基準（VPS の実績: 30/30 成功、`db.netkeiba.com` と `race.netkeiba.com` を交互）
| 項目 | 合格 |
|---|---|
| 成功率（200 かつ本文に目印あり） | 3 回とも 93% 以上（28/30） |
| 403 / 429 | 0 件（1 件でも出たら試験は中止され FAIL） |
| 遮断を示す語（アクセス集中・Access Denied・captcha 等） | 0 件 |
| 応答 | ログイン画面へ飛ばされない（目印 `race_table_01` / `RaceTable01` がある） |

## 記録（`run.sh` が最後に出す行をそのまま貼る）
| 日時 | 枠 | 送信元 IP | 合否 | 成功/計画 | 403/429 | 備考（失敗の理由・気づき） |
|---|---|---|---|---|---:|---|
| | weekday_noon | | | | | |
| | raceday_morning | | | | | |
| | night | | | | | |

## 総合判定（3 回終了後に記入）
- 3 回とも PASS → 取得のうち時間厳守でないもの（`daily-race-lists`・`weekly-update`・`recent-retry`・夜間）を Cloud Run Jobs へ移す検討に進む（`gcp_jobs_migration.md`）。**ログイン依存の取得は含めない**。
- 1 回でも FAIL → 取得は VPS の薄い実行役に集約（`vps_agent.md`）。VPS も遮断された場合はホスト PC（自宅 IP）に戻す。
- 送信元 IP が回ごとに変わるか（変動の幅）も 1 行書く: 

総合判定:

## GCS 監査の記録（`gcp_setup_runbook.md` 6 章）
| 項目 | 値 |
|---|---|
| バケット名 | |
| リージョン | |
| ストレージクラス | |
| 使用量 | |
| バージョニング | |
| ライフサイクル | |
