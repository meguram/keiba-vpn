# メンテナンス時間帯の自動反映（ステージ → 窓の中で適用）

サーバの再起動やフロントの入れ替えを伴う変更を、**毎日 JST 03:00–04:00 の窓の中で、ステージ済みの変更だけ**自動で反映する。
任意のコードを無人で書き換える仕組みではない。**ユーザーが事前にステージした manifest 付きの変更だけ**を、権限を絞って適用する（CI の deploy に近い）。
窓の仕様は [maintenance_window.md](maintenance_window.md)。

## 全体の流れ

```
ローカル: stage_deploy.py create  →  ホスト: data/local/deploy/pending/<id>/
ホスト(root cron, JST 03:05): maintenance_deploy.sh
   窓の中か → 前提条件を待つ → sha256 検証 → バックアップ → 原子的に配置 → 構文チェック
   → 手順(API 再起動 / フロントのビルドと切替) → 検証(/api/health ほか)
   成功: applied/<id>/   失敗: 自動ロールバック → failed/<id>/ + Slack 緊急通知
```

## 1. ステージを作る（ローカル）

```bash
# 作業ツリーの内容をそのまま送る
python -m src.scripts.ops.stage_deploy create --id 20261022-megu-table --desc "めぐ指数の専用テーブル" \
    --steps frontend_build_switch \
    --file frontend/components/MeguTable.tsx --file frontend/lib/pastRunsView.ts

# コミット済みの内容を送る（未コミットの変更を混ぜない。推奨）
python -m src.scripts.ops.stage_deploy create --id 20261022-prob-cal --desc "確率の市場補正" --ref aadc366 \
    --steps api_restart frontend_build_switch \
    --file src/utils/prob_calibration.py --file src/api/app.py --file config/prob_calibration.json \
    --verify "http://127.0.0.1:23220/api/race/202604040201/predictions|status=200|key=pace_type"

python -m src.scripts.ops.stage_deploy list              # pending / applied / failed と直近の監査
python -m src.scripts.ops.stage_deploy cancel <id>       # pending の取り消し（applied には触れない）
```

- `--id`: 英数字・`._-`、3〜80 文字。**同じ id は 2 度適用されない**。
- `--steps`: `api_restart`（`restart_server.sh`、PORT=23220）、`frontend_build_switch`（`frontend_build.sh stg` → `server_watchdog.sh --profile stg --switch-frontend`）。設定やスクリプトだけの変更は steps なし（配置と構文チェックのみ）。
- `--verify "URL|status=200|key=a.b"`: 反映後に確認する URL（繰り返し可）。`api_restart` なら `/api/health`、`frontend_build_switch` ならフロントの `/` は自動で確認する。
- `--expires-hours`（既定 72）: 期限を過ぎたステージは適用せず failed へ。`--max-load`（既定 14）、`--extend-window-minutes`（窓の残りが足りないときに終了を延ばしてよい上限、0〜60）、`--quiet`（成功時の Slack を出さない）。
- 送り方: `ssh` で `pending/.incoming-<id>` に置いてから `pending/<id>` へ名前変更（ランナーが途中の状態を拾わない）。所有者は root。

### manifest の例

```json
{
  "schema": 1, "id": "20261022-prob-cal", "description": "確率の市場補正",
  "created_at": "2026-10-21T18:00:00+09:00", "expires_at": "2026-10-24T18:00:00+09:00",
  "files": [
    {"dest": "src/utils/prob_calibration.py", "sha256": "…64桁…"},
    {"dest": "src/api/app.py", "sha256": "…64桁…"},
    {"dest": "config/prob_calibration.json", "sha256": "…64桁…"}
  ],
  "steps": ["api_restart"],
  "verify": {"urls": [{"url": "http://127.0.0.1:23220/api/race/202604040201/predictions", "expect_status": 200, "json_key": "prob_calibration"}],
             "health_url": "http://127.0.0.1:23220/api/health"},
  "preconditions": {"queue_idle": true, "max_load": 14.0, "min_disk_gb": 3.0, "no_processes": []},
  "rollback": {"method": "restore_backup"}, "extend_window_minutes": 0, "notify": true
}
```

ステージ内の位置は `files/<dest>`。バックエンド変更 = `steps: ["api_restart"]`、フロント変更 = `["frontend_build_switch"]`、両方 = 両方を順に、設定だけ = `steps: []`。

## 2. 安全のための制約（コードで強制。`src/scripts/ops/deploy_lib.py`）

- **manifest に無いパスは置かない**。ステージに未宣言のファイルがあれば拒否。sha256 が合わなければ拒否。
- 配置先は **`src/` `config/` `frontend/` `scripts/` 配下のみ**。絶対パス・`..`・バックスラッシュ・非正規パスは拒否。**シンボリックリンク禁止**（ステージ内のファイル・配置先・親ディレクトリが repo の外を指す場合も）。
- **対象外（人手）**: `.env*`、crontab、`.git`、`node_modules`、`.next*`、ランナー自身（`scripts/ops/maintenance_deploy.sh`、`src/scripts/ops/maintenance_deploy.py`、`deploy_lib.py`）。
- 1 ステージ最大 200 ファイル、1 ファイル 20MB。
- 手順は 2 種類だけ（任意のコマンドは実行しない）。ロールバックの方法は `restore_backup` のみ。
- 窓の外では何もしない（`--force-window` は手動のときだけ）。二重起動は flock で防ぐ。冪等（適用済みの id は再適用しない）。

## 3. ランナーの動き（`scripts/ops/maintenance_deploy.sh` → `src/scripts/ops/maintenance_deploy.py`）

1. 窓（`config/maintenance.json`。臨時の override も含む）の中か判定。外なら終了。
2. pending を古い順に処理。無効（manifest 不正・ハッシュ不一致・期限切れ・重複 id）は `failed/` へ移し、**次のステージは続ける**。
3. 必要な時間（配置・検証 90 秒 + API 再起動 180 秒 + フロントのビルドと切替 720 秒）が窓の残りに収まらなければ**新規に始めない**（`extend_window_minutes` が許可されていれば、実行時の override で終了を延ばす）。収まらなければ次の窓へ持ち越し（以降のステージも順序を保つため待たせる）。
4. 前提条件を窓の終了前まで 30 秒おきに確認: スクレイピングキュー running/queued=0、運用ジョブ（raceday-*、weight_triggered_predict、serving-state-precompute、lap_forecast_daily、tracking_v2_precompute、調教の取り直し、manifest の `no_processes`）が動いていない、load < 上限、ディスクの空き。満たせなければ次の窓へ持ち越し（pending のまま）。
5. バックアップ `data/local/meta/cron_backups/deploy_<id>/`（元のファイルと、新規ファイルかどうかの記録）→ **一時ファイルに書いて `os.replace` で原子的に配置** → 構文チェック（`.py` は ast、`.json` はパース、`.sh` は `bash -n`）。
6. 手順を実行 → 検証。**一度始めた手順は窓の終了で止めない**（途中で止めない）。
7. 失敗（構文エラー・手順の失敗・検証の失敗・例外）は**自動ロールバック**: 退避を戻し（新規ファイルは削除）、実行済みの手順をやり直す（API 再起動、フロントは `KEIBA_FORCE_BUILD=1` で再ビルドして切替）、`/api/health` などを再確認。結果 `failed_rolled_back`（復旧）か `failed_rollback_error`（人手の対応が必要）。Slack に**緊急**通知。
8. 成功は `applied/<id>/`（`result.json` に退避の場所）へ。Slack に開始・完了（`KEIBA_SLACK_NOTIFY=1` のとき）。

監査: `data/local/meta/deploy_audit.jsonl`（`start / placed / step / applied / failed / rollback / deferred / invalid / waiting / window_extended / cancelled`）と `logs/maintenance_deploy.log`。cron の成否は監視ポータルの `maintenance-deploy` ジョブでも見える。

### 失敗したとき

| 状態 | 意味 | 対応 |
|---|---|---|
| `failed_rolled_back` | 反映に失敗したが元に戻り、サービスは復旧している | `failed/<id>/result.json` の `reason` を直して、新しい id でステージし直す |
| `failed_rollback_error` | ロールバック後の再起動・確認にも失敗 | **人手の対応**。退避は `data/local/meta/cron_backups/deploy_<id>/files/`、手動で戻し `restart_server.sh` と `frontend_build.sh stg` → `--switch-frontend` |
| `invalid` | manifest・ハッシュ・期限・id の問題 | `failed/<id>/result.json` を確認。何も配置していない |
| `deferred`（pending のまま） | 前提条件か窓の残りが足りない | 次の窓で再試行。急ぐなら前提（キューや重い計算）を片付けるか、`--extend-window-minutes` |

## 4. cron の登録（ホストで root。人が行う）

```bash
ssh keiba-remote 'cd /home/jovyan/work/keiba-vpn && sudo -n bash scripts/cron/install_maintenance_deploy_cron.sh /home/jovyan/work/keiba-vpn'
# 開始/終了の Slack 通知も入れるとき（config/maintenance.json の notify.slack_start_end も true に）
#   … install_maintenance_deploy_cron.sh /home/jovyan/work/keiba-vpn --with-notify
# 外す: … install_maintenance_deploy_cron.sh /home/jovyan/work/keiba-vpn --remove
```

登録されるのは `5 18 * * *`（18:05 UTC = 03:05 JST、タグ `KEIBA-MAINTENANCE-DEPLOY`）。pending が空なら即終了する。登録前に `crontab -l` を `data/local/meta/cron_backups/` へ退避する。
窓の機能（`KEIBA_MAINTENANCE_WINDOW=1`）とは独立に動く（ランナーは `config/maintenance.json` の窓だけを見る）。**窓の表示を出さずに反映だけ行うこともできる**が、窓の表示を有効にしてから使うのが本来の運用。

## 5. 手動で試す

```bash
# ホストで（root）: 検証だけ（何も変えない。窓の外でも可）
cd /home/jovyan/work/keiba-vpn && sudo -n bash scripts/ops/maintenance_deploy.sh . -- --dry-run
sudo -n bash scripts/ops/maintenance_deploy.sh . -- --list
# 窓の外で今すぐ適用（手動。前提条件の確認と自動ロールバックは同じ）
sudo -n bash scripts/ops/maintenance_deploy.sh . -- --force-window
```

テスト: `tests/scripts/test_maintenance_deploy.py`（一時ディレクトリの疑似 repo とモック。manifest の検証、パス制限、シンボリックリンク、原子的配置、前提条件、窓の判定・延長、構文エラー / 手順の失敗 / 検証の失敗での自動ロールバック、順序、取り消し）。実サーバは再起動しない。
