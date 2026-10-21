# VPS の常駐最小構成：取得の実行役・メモリ上限・ハートビート・pull 型デプロイ

設計の根拠は [architecture_target.md](./architecture_target.md)（3.2 メモリ予算、5.2 時間厳守ジョブ、5.6 デプロイ、7.1 フェーズ 1、フェーズ 4）。
この文書は、リポジトリ側で用意した部品の**使い方・導入順・戻し方**をまとめる。VPS への実接続は、ここに書いた手順でユーザーが行う。

## 1. 全体図

```
GCP（司令塔）                     GCS                               VPS（常駐最小）
Cloud Scheduler / Job ──積む──▶ ops/fetch_tasks/pending/<id>.json ◀─60秒pull── keiba-fetch-agent（1本ずつ）
  (src.ops.fetch_plan)           running/ done/ failed/                          └ 子プロセス src.ops.fetch_child（MemoryMax 250M）
                                 ops/events/weights_announced/<日>/<race>.json ◀─ 馬体重が出たら置く → GCP が予測を起動
ホスト PC ──publish_release.sh──▶ ops/releases/{ui,fetch}/latest.json ──03:10/03:20 JST に pull──▶ pull_release.py → 入れ替え
VPS ──5 分おき（HMAC 署名）──▶ GCP /ops/heartbeat（死活・メモリ・ユニット。VPS は Slack の秘密を持たない）
```

## 2. 部品

| 区分 | ファイル | 役割 |
|---|---|---|
| タスク表 | `src/ops/fetch_task.py` | スキーマ・allowlist・GCS 上のキュー（作成のみ可の条件つき書き込みで排他 claim、停滞回収、期限切れは failed） |
| 計画・投入 | `src/ops/fetch_plan.py` | 開催日の T-60/T-15/直前(T-10/5/2)/馬体重検知を積む。定期タスク（entries-sync 等）を 1 件積む |
| 実行役 | `src/ops/fetch_agent.py` / `fetch_child.py` | pull → 起動 → 結果記録。再試行は `max_attempts` まで。アクセス制限の疑いは再試行せず 30 分 claim を止める |
| 持ち主の切替 | `config/fetch_owner.json` / `src/ops/fetch_owner.py` / `scripts/auto_scrape.sh` | `agent_tasks` に載せたタスクはホストの cron が実行しない（二重取得の防止） |
| ユニット | `scripts/ops/vps/templates/keiba-fetch*.{slice,service}`、`dropin-*-memory.conf` | メモリ上限と超過時の挙動 |
| 導入 | `scripts/ops/vps/vps_03_units.sh`（`--dry-run` 可・冪等）、`vps_01_base_setup.sh`（earlyoom・journald 100MB） | |
| ハートビート | `scripts/ops/vps/heartbeat_client.py`、`templates/keiba-heartbeat.{service,timer}` | 他の担当が作成。取得の実行役のユニット（`keiba-fetch-agent`）を監視対象に追加済み |
| pull 型デプロイ | `src/ops/release_publish.py`、`scripts/ops/vps/publish_release.sh`（配布側）、`pull_release.py` と `templates/keiba-release-pull*`（VPS 側） | |
| 実測 | `scripts/ops/vps/measure_memory.sh` | `free -m` とユニット別の表を出す |
| 依存 | `requirements-fetch.txt` | 取得の実行役だけの最小の依存（UI・ML・API は入れない） |

### タスクの種類（`fetch_task.ALLOWED_TASKS`）
- `race-odds-t60|t15|t10|t05|t02`（引数 `race_id`, `date`）：1 レースのオッズ取得。t10/t05/t02 は `KEIBA_ODDS_LATE_SNAPSHOTS=1` のときだけ実行（`lead_seconds=60` で期限の 60 秒前から待機）。
- `race-weight-check`（`race_id`, `date`）：出馬表を取り直し、馬体重が出たらイベントを置く。出た後の検知タスクは取得せず省略する。
- auto_scrape のモード：`entries-sync` `recent-retry` `barometer-retry` `daily-race-lists` `weekly-update` `catchup-missing` `jra-baba-morning` `raceday-evening` `raceday-comment` `new-horse-pedigree` `raceday-eve-fetch`。
- **含めないもの**：常駐ループ（`raceday-runner` `raceday-weight-runner` `raceday-result-runner`）は実行役を 1 日塞ぐので個別タスクに置き換え。`raceday-eve`（追走難度などの前計算で ML を読む）・`feature-accumulate` は 250M に収まらないので GCP Jobs。
- 秘密（netkeiba のログイン等）はタスクの引数にもログの末尾にも出ない（引数のキー検査と、既知の秘密値の伏せ字）。

## 3. メモリ表（architecture_target.md 3.2 の想定 → 実測欄）

| ユニット | MemoryHigh | MemoryMax | 超過時 | 実測（現在 / ピーク） |
|---|---:|---:|---|---|
| nginx | 32M | 48M | 1 秒で再起動 | （実測を貼る） |
| cloudflared-prod | 96M | 128M | 再起動・再接続 | |
| keiba-ui（Next.js） | 160M | 200M | 5 秒で再起動 | |
| keiba-fetch-agent（slice 共通） | 220M | **250M** | 最初に OOM Kill（`OOMScoreAdjust=500`）→ 失敗を記録 → 再試行 | |
| keiba-heartbeat | — | 48M | 次回に再実行 | |
| 合計（OS を除く上限） | | 約 626M | OS 約 400MB と合わせて最悪でも MemAvailable 約 1.0GB | |

- 参考（開発 PC で `requirements-fetch.txt` だけを入れた venv を作って計測）：実行役の親プロセス 約 25MB、子プロセスが取得モジュールを import した直後 約 44〜63MB。取得中の実測は VPS で行う（未測定）。venv のディスクは 約 380MB。
- 実測のしかた：通常時と、実行役の動作中（`sudo systemctl start keiba-fetch-agent` 後にタスクが走っている間）に `bash scripts/ops/vps/measure_memory.sh --label "通常時" > /tmp/mem.md` を実行し、結果を上の表に貼る。目標は used ≤ 600MB、MemAvailable ≥ 1.0GB、swap 使用 ≤ 200MB（表の下に OK/NG を出す）。
- swap は `vps_01` が 2GB 作る（architecture_target の 1GB より大きいが問題なし）。取得の実行役は `MemorySwapMax=0` で swap を使わない。

## 4. 導入手順（ユーザー作業。上から順に）

前提：GCP 側（バケット、サービスアカウント 2 つ、Cloud Run のハートビート API）は GCP の移行チェックリストの手順で用意する。鍵や秘密の値は**チャットに貼らず**、`set_secret.sh` か標準入力でファイルに入れる。

- 取得の実行役用サービスアカウント：`ops/fetch_tasks/`・`ops/events/` の読み書き。pull 用：`ops/releases/` の読み取りと `ops/releases/*/applied/vps.json` の書き込み。（権限を分けるため鍵は別）

1. **土台**（VPS・root）：`sudo bash vps_01_base_setup.sh --ssh-mode …`（earlyoom と journald 100MB が追加された。既に実行済みなら再実行して反映。冪等）。
2. **ユニット類**：`sudo bash vps_03_units.sh --dry-run` で内容を確認 → `sudo bash vps_03_units.sh --with-pull-venv`。メモリ上限の drop-in が入り、`keibafetch` ユーザーと環境ファイルの雛形（秘密なし）ができる。
3. **ハートビート**（他担当の手順書に従う）：`/etc/keiba-ui/heartbeat.env` に `HEARTBEAT_URL` と `KEIBA_HEARTBEAT_SECRET` → `sudo systemctl enable --now keiba-heartbeat.timer`（`vps_03_units.sh` を再実行すると有効化される）。
4. **pull 型デプロイ**：`/etc/keiba-pull/pull.env` の `GCS_BUCKET`、鍵ファイル `/etc/keiba-pull/sa.json`（`ssh keiba-vps 'sudo install -m 600 /dev/stdin /etc/keiba-pull/sa.json' < 鍵.json`）→ `sudo bash vps_03_units.sh` を再実行（タイマー `keiba-release-pull-ui.timer` 03:10 / `-fetch.timer` 03:20 JST が有効になる）。
5. **取得のコードを VPS に置く**（ホスト）：`bash scripts/ops/vps/publish_release.sh --fetch --yes` →（VPS）`scp requirements-fetch.txt` してから `sudo bash vps_03_units.sh --with-fetch-venv /tmp/requirements-fetch.txt` → `sudo /opt/keiba-pull/venv/bin/python /opt/keiba-ui/bin/pull_release.py --kind fetch --force`（初回だけ窓の外でも入れる）。
6. **確認**：`sudo -u keibafetch env KEIBA_ENV=prod PYTHONPATH=/opt/keiba-fetch/current /opt/keiba-fetch/venv/bin/python -m src.ops.fetch_child selfcheck` が `"missing": []` を返す。
7. **取得の秘密**：`/etc/keiba-fetch/fetch.env` に `GCS_BUCKET`、netkeiba のログイン（`.env.example` の `NETKEIBA_*` と同名）。`/etc/keiba-fetch/sa.json` に鍵（権限 640 root:keibafetch）。
8. **実測**（3 章）。通常時の数字を表に貼る。
9. **ここまでは何も取得しない**（`KEIBA_FETCH_AGENT_ENABLED=0`、`agent_tasks` 空）。移行は次章。

## 5. 取得を段階的に移す（タスクごとに切り替え）

推奨の順（低リスク → 時間厳守）：`entries-sync` → `recent-retry` `barometer-retry` `daily-race-lists` `jra-baba-morning` → `race-odds-t60/t15`（1 開催週末）→ 直前オッズ → `race-weight-check`。

**その前に 5.5 章の影モードで、ホストを止めずに VPS の取得を数日間見届ける（基準を満たすまでコードが担当の移譲を拒否する）。**

1 タスクの切り替え手順（**ホストを先に止め、直後に VPS を始める**。二重取得より 1 回の取りこぼしの方が安全）：
1. GCP の Scheduler（またはホスト）が積む準備をする：定期タスクは `python -m src.ops.fetch_plan recurring --task entries-sync`（10 分おき）、開催日は前日夜〜朝に `python -m src.ops.fetch_plan day --date YYYYMMDD [--late]`。`--dry-run` で件数を確認。
2. `config/fetch_owner.json` の `agent_tasks` に追加してホストに反映（ステージ → メンテナンス窓の自動反映）。以後 `scripts/auto_scrape.sh` が該当タスクをスキップする（`race-odds-*` を移すとホストの `raceday-runner`、`race-weight-check` を移すと `raceday-weight-runner` もスキップ）。
3. VPS：`/etc/keiba-fetch/fetch.env` を `KEIBA_FETCH_AGENT_ENABLED=1` にし、`publish_release.sh --fetch` で同じ `fetch_owner.json` を持つバンドルを配布 → `sudo bash vps_03_units.sh --enable-fetch-agent`。
4. **並走の検証**：その日のスナップショット（`race_odds` の `snapshots` の phase と `snapshot_at`、`late_odds_report.py`）に欠けが無いか、`fetch_plan show --state failed` が空か、ハートビートが正常かを確認する。
5. 問題があれば 6 章で戻す。

## 5.5 影モード（ホストが取得を続けたまま、VPS の取得を数日間見届ける）

**方針**: prod（VPS/GCP）で取得が数日間うまくいくのを見届けるまでは、stg（ホスト PC）の取得は止めない。そのために、ホストが取得を続けたまま、VPS の実行役が**同じ取得を並走**し、結果を本番ではなく `shadow/<日付>/` 以下に保存して、本番（ホストが取ったもの）と比べる。基準を満たすまで、**担当の移譲（`job_owner set`）・適用スクリプトの生成（`cutover_plan make`）・crontab の停止（`host_cron_switch.sh disable`）はコードが拒否する**（`--force` でも外せない）。

### 仕組み
| 部品 | 内容 |
|---|---|
| 有効化 | VPS の `/etc/keiba-fetch/fetch.env` に `KEIBA_FETCH_SHADOW_ENABLED=1`（既定 0）と `KEIBA_FETCH_AGENT_ENABLED=1`。GCP の予約ジョブ（`reserve-race-day` / `reserve-fetch-tasks`）の環境にも同じ `KEIBA_FETCH_SHADOW_ENABLED=1`（影のタスクを積むため）。**直前オッズ（T-10/5/2）を比べるなら VPS の `.env` にも `KEIBA_ODDS_LATE_SNAPSHOTS=1`**（無効だと取得せずスキップ扱いになり、スキップが 10% を超えると不合格） |
| 対象 | `config/fetch_shadow.json` の `tasks`（軽いものだけ）: `entries-sync`、`race-odds-t60/t15/t10/t05/t02`、`race-weight-check`（3 回に 1 回に間引く）、`daily-race-lists`、`jra-baba-morning`、`recent-retry`、`barometer-retry`。`weekly-update` `raceday-evening` `new-horse-pedigree` など重い定期処理は**対象外**（影では実行しない。子プロセスも拒否する） |
| タスク表 | 本番とは別の名前空間 `gs://BUCKET/ops/fetch_tasks_shadow/`。本番のタスク表・イベント（馬体重の検知→予測の起動）には触れない（`PrefixedStore` が他のキーを拒否） |
| 書き込みの隔離 | 子プロセス（`KEIBA_FETCH_SHADOW=1`）は `shadow_storage.install()` で GCS の書き込み先を `shadow/<日付>/<本番と同じパス>` に付け替え、**それ以外の書き込み・削除・コピーは例外**。読み取りは本番をそのまま読む。コードの保証に加えて IAM でも守る（下の「事前」） |
| 実行 | 同じ実行役が、本番のタスクが無いときに影のタスクを 1 本ずつ実行（メモリは 250M の枠内）。本番のタスクが優先。影のタスクは本番の計画に 5〜300 秒（タスクごと）のジッターを足して積むので、ホストと同じ秒には叩かない |
| 上限 | 1 日の見積りリクエスト数が `max_requests_per_day`（既定 2,500）を超えるタスクは**取得せずスキップ**（失敗にはしない。スキップが 10% を超えた日は不合格） |
| アクセス制限 | 403/429 や制限語を検知したら**即停止**し、`access_pause_seconds`（既定 3,600 秒。最低 1,800）claim を止め、Slack に注意を出す。1 日に 2 回で翌朝 6 時まで止める。**本番の実行役も同じ IP なので一緒に止まる**。状態は VPS のローカル（`data/local/meta/fetch_shadow/state.json`）に残り、再起動しても止まったまま |
| 日次集計 | GCP の Run Job `fetch-shadow-report`（毎朝 07:30 JST）が前日分を `ops/fetch_shadow/daily/<日付>.json` に書き、Slack に要約を出す（`src/scripts/ops/shadow_report.py`） |

### 本番との比較の規則（`shadow_report.py`）
- `race_odds`: 同じ phase の本番スナップショット（最新が別の phase なら `snapshots` から探す）と、出走馬の集合が一致し、単勝オッズが相対 15% 以内の馬が 90% 以上（取得時刻の差を許す）。
- `race_pair_odds`: 券種ごとに組合せの集合が一致し、オッズが相対 15% 以内のものが 90% 以上。
- `race_shutuba`: 馬の集合と安定項目（馬番・枠・馬名・騎手・調教師・性齢）が一致。馬体重は両方に値があるときだけ比べる（片方が未発表なら無視）。
- 一致率の基準は 95%、取得の成功率は 95%、アクセス制限の疑い・異常通知は 0 件。

### 合格の基準（`config/fetch_shadow.json` の `pass`。`shadow_eval.py` が判定）
直近 **5 日**（昨日まで）がすべて合格、うち**開催日を 1 日以上含む**こと。さらにジョブのタスクが `min_task_days` 日以上実行され、各日のそのタスクも合格であること。日次レポートが 1 日でも欠ける・読めないときは**不合格**（安全側）。影モードの対象外のタスクを含む VPS ジョブ（例: `weekly-update`）は、`config/gcp_job_map.json` のそのジョブに `"shadow": {"exempt": true, "reason": "…"}` を**人が**書くまで止められない（免除はユーザーの判断）。

### リクエスト増の見積りと DEC-023
- 目安（`est_requests` の合計。実測で置き換える）: 平日 約 250 件/日、開催日（36R）約 1,000 件/日（T-60 が 2、T-15 が 12、馬体重 約 4、直前を有効にすると +3 / レース）。**ホストと合わせて、その間は netkeiba への取得がほぼ 2 倍**になる。上限 2,500 件/日はこの 2.5 倍。
- DEC-023（同じ時刻に叩かない）は、ジッター（ホストの計画時刻に 5〜300 秒を足す）と、1 日の上限、アクセス制限での即停止で守る。**影モードを動かす間は、ホストの重い取得（バックフィル）を同じ時間帯に走らせない**。長い影の期間は不要（5 日 + 開催日）なので、判定が揃ったら止める。

### 手順
**事前（ユーザー・1 回）**
1. GCS の IAM: 影モードの VPS 実行役のサービスアカウントには、通常の書き込み権限（`ops/fetch_tasks/` 等）に加えて、**`shadow/` と `ops/fetch_tasks_shadow/` 以下だけ**に書ける条件つきの `storage.objectUser` を付ける（IAM 条件 `resource.name.startsWith("projects/_/buckets/BUCKET/objects/shadow/")`）。本番のデータのパスには書き込み権限を付けない。これがコードの隔離の外側の保証になる。
2. `config/fetch_shadow.json` の `tasks` と `limits` を確認（対象を減らす・上限を下げるのは安全側）。

**開始**
1. VPS の `fetch.env` に `KEIBA_FETCH_SHADOW_ENABLED=1`（と、直前オッズを見るなら `KEIBA_ODDS_LATE_SNAPSHOTS=1`）。`sudo systemctl restart keiba-fetch-agent`。ホストの `fetch_owner.json` の `agent_tasks` は**空のまま**（ホストが取得を続ける）。
2. GCP の `reserve-race-day` / `reserve-fetch-tasks` の環境に `KEIBA_FETCH_SHADOW_ENABLED=1`。`python -m src.ops.reservations race-day --date YYYYMMDD --dry-run` の出力の `shadow_tasks` が 0 でないことを確認。
3. `job_map` の `fetch-shadow-report` の `ready` を true にして Scheduler に載せる（`python -m src.ops.job_map ready fetch-shadow-report --note …`）。

**確認（毎日）**
- `python -m src.scripts.ops.shadow_report --yesterday --json`（またはその朝の Slack 要約）で `ok`、`success_rate`、`match_ratio`、`access_restrictions`、`skipped` を見る。
- `python -m src.scripts.ops.cron_cutover_status` の各 VPS ジョブの `shadow_days`（足りない日数・不合格の理由が出る）。
- `python -m src.ops.fetch_shadow status`（VPS で。当日の予算の消費・停止状態）。

**終了（判定が揃ったら）**
1. `cron_cutover_status` が `shadow_days` 合格になったジョブから、4 章の手順で担当を渡す（`cutover_plan make` → `_1_owner.sh`。ここで初めて `fetch_owner.json` の `agent_tasks` に載り、ホストの cron が見送る）。
2. 担当を渡したタスクの影は自動で止まる（実行役が担当しているタスクは影の対象から外れる）。全部移し終えたら `KEIBA_FETCH_SHADOW_ENABLED=0` に戻す。`shadow/` 以下は 30 日で削除してよい（ライフサイクル規則を付ける）。

**ロールバック（影モードをやめる）**
- VPS の `KEIBA_FETCH_SHADOW_ENABLED=0` にして実行役を再起動、GCP の予約ジョブの環境からも外す。ホストの取得は最初から止めていないので、ほかの作業は不要。積み済みの影のタスクは放置でよい（期限が来ると失敗として記録されるだけで実行されない）。
- アクセス制限の疑いで止まった場合: 自動で止まる。再開してよいと判断したら `data/local/meta/fetch_shadow/state.json` の `paused_until` を消して実行役を再起動。制限が続く（VPS の IP が制限されている）なら、`architecture_target.md` 5.1 のフォールバック（取得はホストに残す）に切り替える。

## 6. ロールバック

| 状況 | 手順 |
|---|---|
| 取得をホストに戻す | `config/fetch_owner.json` の `agent_tasks` からタスクを外してホストへ反映（以後 cron が動く）。VPS：`sudo systemctl disable --now keiba-fetch-agent`。積み済みの pending は放置でよい（期限が来ると failed に記録され、実行されない） |
| 緊急（ホストに反映できない） | VPS の実行役を止め、ホストで `KEIBA_FETCH_OWNER_FILE=/dev/null` を cron の環境に足す（担当表を空として扱う） |
| UI を前の版へ | `sudo /opt/keiba-ui/bin/rollback.sh`（pull 型でも同じ。`install_release.sh` が切替後の確認に失敗したら自動で戻る） |
| 取得コードを前の版へ | `ln -sfn /opt/keiba-fetch/releases/<前の版> /opt/keiba-fetch/current && sudo systemctl restart keiba-fetch-agent`（5 世代残る） |
| pull 型を止める | `sudo systemctl disable --now keiba-release-pull-ui.timer keiba-release-pull-fetch.timer`（従来の `promote.sh` は引き続き使える） |
| 同じ版の失敗の繰り返し | `/var/lib/keiba-pull/<kind>.json` の `failed_ids` に入った版は再試行しない。直したら新しい版を公開する（`--force` で強制） |

## 7. pull 型デプロイ（SSH を閉じるために）

- 昇格 = `bash scripts/ops/vps/publish_release.sh [--yes]`（ホスト。stg で確認した版を GCS に置き、`latest.json` を最後に書く）。VPS は毎日 JST 03:10 に取得し、メンテナンス窓（03:00〜04:00）の外では何もしない。
- 入れ替えは既存の `install_release.sh`（sha256 検証 → 別ポートで起動テスト → 切替 → 確認 → 失敗なら自動で戻す）。**日時（リリース ID 末尾 12 桁）が現在より新しい版だけ**入れる（戻すのは手動の `rollback.sh`）ので、`promote.sh`（SSH で即時）と共存できる。
- 結果は GCS の `ops/releases/<kind>/applied/vps.json` に VPS が書く：`python -m src.ops.release_publish status --kind ui`。
- SSH を閉じる：pull が 1 回成功し、ハートビートが届いていることを確認したあと `vps_01_base_setup.sh --ssh-mode closed`（管理は Cloudflare Tunnel 経由の SSH か ConoHa のコンソール）。**緊急の `promote.sh` が使えなくなる**ので、窓を待てない修正がある場合の代替（コンソールからの手動導入）を決めてから閉じる。
- 取得コードのバンドルに入るのは `src/`・取得が読む `config/` の数ファイル・`requirements-fetch.txt`・`scripts/auto_scrape.sh` だけ（会員名簿 `members.json`・費用・モデルの重みは入らない）。

## 8. 通知と環境表示

- VPS は Slack の秘密を持たない。失敗は `failed/` に記録され、ハートビートの `keiba-fetch-agent` の状態と GCP 側の監視が通知する。VPS 上で通知が必要になったら Slack を直接叩かず `src.utils.slack_notify.send_payload` を使う（環境名【PROD】と実行場所が付く）。
- 環境ファイルの雛形は `KEIBA_ENV=prod` と `KEIBA_RUN_LOCATION=vps` を持つ（`vps_03_units.sh` が作る）。

## 9. 既知の未解決

- **`race-odds-t15` は取得だけ**：ホストの `raceday-runner` は T-15 取得の後に予測の起動なども行う。VPS の実行役は取得のみで、予測は GCP 側が `race_odds` の更新やイベントを見て起動する設計（GCP 側の実装と突き合わせが必要）。
- `weekly-update` `raceday-evening` などは中で計算（めぐ指数・PG 同期など）を行う。250M に収まるか未測定のため、移す前に VPS で実測する（初期の担当は軽いタスクだけにしてある）。
- 取得の実行役の実メモリ（取得中）と venv の依存の過不足は VPS での実測・`selfcheck` で確認する（開発 PC の clean venv では `missing: []`）。
- `auto_scrape --task` は例外を飲み込んで終了コード 0 を返す。`fetch_child auto` は状態ファイルを読んで error を失敗にするが、タスク自身が `status: error` を書かずに黙って終わる場合は検知できない。
- GCP の IP からの netkeiba 到達試験（architecture_target.md 7.4）が不合格のときは、取得を VPS に集約する前提のまま。VPS の IP も遮断された場合の最後の砦はホスト PC。
- ConoHa のコンソールからの緊急導入手順（SSH を閉じた後）は未整備。
