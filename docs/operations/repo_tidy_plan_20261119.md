# リポジトリの整理の計画（直下のファイルとディレクトリ構成）

2026-11-18 調べ・2026-11-19 版。この段は調査と計画だけ。共有の作業ツリー（`/home/jovyan/work/keiba-vpn`）のファイルは、移していないし消してもいない。
実施は 11/23（月・祝）の開催の後で、段ごとにユーザーの承認を得てから行う。費用は 0 円（GCS・Cloud Run に触れない）。

## 結論

- 直下に散らばっている物の多くは、実行時のファイル（pid 16 個・lock 2 個・profile 1 個）と、生成物と、`.env` の控え 6 個。
- 直下に残すのは、設定・入口・README の類だけにする。実行時のファイルは `run/`（git 外）にまとめる。
- 動かすと壊れる物がはっきりしている。root の crontab・VPS の配布物・Docker・CI・docs の本・Cloud Run Jobs の 6 つ（下の「動かすと壊れる所」）。
- ディレクトリ名と役割が一致していない。`src/research` に本番の部品があり、`src/scripts/data` に研究の物が多い。そのため名前では動かさず、使われ方の分類で決める。
- 段は 3 つに分け、それぞれ別に承認を得る。
  - 段 1: 掃除と文書の食い違いの修正（安全）
  - 段 2: pid・lock の置き場の移動（サービスの再起動を伴う）
  - 段 3: ディレクトリ構成の変更（段 3a と段 3b に分ける）

## 直下の棚卸し

追跡は `git ls-files`、無視は `git status --ignored` で確かめた。参照元は scripts・src・tests・Makefile・Dockerfile・CI・root の crontab を検索して調べた。

### 実行時のファイル（すべて git 外・`.gitignore` 済み）

書くのは `scripts/server/` の 9 本だけ。Python からは読まれない。

| ファイル | 書く物 | 2026-11-18 の状態 |
|---|---|---|
| `.flask.pid` | `service_start.sh:40`・`server_watchdog.sh:132` | 生きている（Flask） |
| `.frontend.pid` | `service_start.sh:41`・`server_watchdog.sh:135` | 生きている（next dev :23100） |
| `.server-23120.pid` | `restart_server.sh:38`・`server_watchdog.sh:109` | 生きている（FastAPI） |
| `.ops-queue.pid` | `server_watchdog.sh:330` | 生きている（:23220） |
| `.monitor.pid` | `start_monitor.sh:60` | 生きている（監視ポータル） |
| `.tcpexposer_tunnel.monitor.pid` | `tunnel_tcpexposer.sh:63` | 生きている（トンネル） |
| `.flask-{dev,stg}.pid`・`.frontend-{dev,stg}.pid` | `server_watchdog.sh --profile` | 古い（プロセスなし） |
| `.server.pid`・`.server-23220.pid` | 旧 `restart_server.sh` | 古い |
| `.tcpexposer_tunnel.pid`・`.tcpexposer_tunnel.stg.pid` | `tunnel_tcpexposer.sh`（旧名と stg） | 古い |
| `.cloudflared.pid` | `tunnel_cloudflared.sh:16` | 古い |
| `.mlflow.pid` | `service_start.sh:42`・`server_watchdog.sh:119` | 古い |
| `.autostart_dev.lock` | `autostart_dev.sh:59`（IDE の起動フック） | flock 用の空ファイル |
| `.ensure_monitor.lock` | `ensure_monitor.sh:36`（cron 5 分おき） | flock 用の空ファイル |
| `.service_start.profile` | `service_start.sh:512`・`server_watchdog.sh:67` が読む | 最終起動のプロファイル |

- 持ち主は混ざっている。Flask・Next.js・FastAPI・ops-queue・監視ポータルの pid は root の物（root の crontab の watchdog が起動する）。トンネルの pid・profile・lock の 2 つは jovyan の物。移す先は、両方が書ける必要がある。
- 「古い」はプロセスの番号と名前を `ps` で照合した結果。番号は使い回されるので、消す当日にもう一度照合する（段 1 の 3）。

### `.env` の控え（6 個・git 外）

- 中身は読んでいない。書くコードは無い（作業のときに手で作った控え）。
- 段 1 で、`.env`・`.env.stg`・控えの権限を 600 にそろえる。
- 消すか、作業ツリーの外の保管場所（`~/` の下の権限 700 のディレクトリ）へ移すかは、ユーザーが決める。`data/` の下（GCS へのミラー・控えの対象）には置かない。
- `.env`・`.env.stg` は使っている物なので動かさない。

### 生成物・キャッシュ（git 外）

| 物 | 由来 | 扱い |
|---|---|---|
| `.coverage` | pytest-cov（2026-07-04） | 消す |
| `catboost_info/` | CatBoost の学習（`train_dir` 未指定で作業ディレクトリに書く。`src/pipeline/models/ensemble_trainer.py`・`zoo.py` ほか） | 消す。再発は `.gitignore` 済みなので害は無い |
| `__pycache__/`（直下）・`.pytest_cache/`・`.ruff_cache/` | Python・pytest・ruff | 残す（無視済み・自動で作られる） |
| `tmp/` の古いログ 2 個 | 旧 `setup_all_cron.sh` の試し（今は `/tmp` に書く） | 消す |
| `logs/`（158MB） | cron・サービスのログ。`rotate_logs.sh` が回す | 残す |
| `mlruns`（`mlflow/runs` への symlink） | `src/pipeline/mlflow/runtime.py:66` が作り直す | 残す |

### 追跡されている直下のファイル

| ファイル | 参照元 | 扱い |
|---|---|---|
| `main.py` | `Dockerfile.serving`・CI（compileall・起動）・`scripts/server/*` | 残す |
| `wsgi.py` | `Dockerfile.serving:73` | 残す |
| `service_start` | `scripts/server/service_start.sh` への入口 | 残す |
| `Makefile` | 手作業の入口（setup・test・dev-mock） | 残す |
| `requirements.txt`・`constraints-jobs.txt` | CI・`Dockerfile.{jobs,gcp-jobs,serving,cloudrun}` | 残す |
| `requirements-fetch.txt` | VPS の取得の実行役（`scripts/ops/vps/vps_03_units.sh`・`src/ops`） | 残す |
| `pytest.ini`・`alembic.ini` | pytest・`alembic upgrade head`（CI） | 残す |
| `docker-compose.{dev,serving}.yml` | 置き場で名前とボリュームが決まる（AGENTS.md） | 残す |
| `.env.example`・`.env.stg.example`・`.env.prod.example` | 手順書 | 残す |
| `keiba-vpn.code-workspace` | 参照なし。存在しない `.venv` を指す | 消す候補（承認） |
| `tmp/.gitkeep` | コードは直下の `tmp/` を使わない | `tmp/` ごと消す候補（承認） |

## ディレクトリ構成の現状

### 置き場の決まりと実態の食い違い

- `scripts/` は「シェルのみ」の決まり（AGENTS.md）だが、`.py` が 9 個ある。
  - `scripts/run_tp26.py`: 一回きりの研究。参照なし。消す候補。
  - `scripts/dev/retrain_ctl.py`・`retrain_upstream.py`: 再学習の管理と上流の実行。
  - `scripts/ops/vps/*.py` の 6 個: VPS で単体で動く。置き場は妥当なので、決まりの方を直す。
- `scripts/` 直下の 11 本は、どれも crontab から呼ばれない。
  - `auto_scrape.sh`: VPS の配布物に入る（`src/ops/release_publish.py:50`）。動かさない。
  - `setup_dev.sh`・`setup_prod.sh`: Makefile が使う。
  - `show_scrape_progress.sh`・`watch_backfill_pace.sh`: 一回限りの進捗表示。古い候補。
- `scripts/cron/` の 61 本のうち、root の crontab が呼ぶのは 21 本。残りの大半は crontab を書く道具（`install_*.sh`。テストが検査する）。
  - 参照なし: `maintenance_notify.sh`（シェルだけ。呼ぶ先の `src/scripts/ops/maintenance_notify.py` はテストが参照する）。
  - `install_lap_forecast_cron.sh` は参照なしに見えるが、今動いている `lap-forecast-eve`・`lap-forecast-morning` の 2 行を書く唯一の道具。消さない。
  - 退役済み: `sporthorse_births_nightly.sh`（`config/gcp_job_map.json:3674` で retire。外す手順 `install_sporthorse_births_cron.sh --remove` が参照）・`jma_weather_run.sh`（`gcp_job_map.json:4326` で enabled=false）。
  - 廃止済みの仕組みの残り: `git_pull_hourly.sh`・`setup_git_pull_cron.sh`（pull 型のランナーは 2026-10-30 に廃止。docs の 4 か所に載る）。
- AGENTS.md と `docs/operations/PROJECT_LAYOUT.md` の記述が古い。
  - scripts に `ops/`・`gcp/`・`dev/`・`ci/` が載っていない。config は settings.yaml だけと書いてある。
  - PROJECT_LAYOUT.md は特徴量ストアを `data/features/` と書く（実際は `data/local/features/`）。
  - 直下の `frontend/`・`deploy/`・`site_public/`・`dev_fixture/` が概要図に無い。
- `src/data_health/spec.py:133-140` は、`config/megu_predict_*.json` と `models/*.pkl` を「git から取得」としている。実際は `config/.gitignore`・`models/.gitignore` で無視している（学習済みの重みは公開リポジトリに置かない方針）。
- `data/skill_logs/` はスキルが作る報告物で、データではない。
- `templates/`・`static/` は旧 FastAPI（`src/api/app.py:902-906`）と `Dockerfile.{cloudrun,serving}` が使う。残す。

### src/ の使われ方（静的な解析・1,405 モジュール）

入口（src/api・src/monitor・src/ops・main.py・root の crontab・scripts/{cron,server,ops,gcp,ci}・Dockerfile・CI・Makefile・Cloud Run Jobs の設定）から、import・`-m`・文字列をたどった。

| 分類 | 件数 |
|---|---:|
| 本番・運用の経路 | 609 |
| 開発補助だけ（scripts/dev ほかから） | 25 |
| 研究（本番以外のコード・研究用の config から） | 307 |
| テストだけが参照 | 117 |
| どこからも参照なし | 347 |

本番以外の 796 件の一覧は `docs/operations/repo_tidy_src_usage_20261119.tsv`（パス・分類・最終更新・docs の言及の数・行数）。

- 名前と役割が合っていない。
  - `src/research` の本番: graded 18/20・race_analysis 12/12・pedigree 34/75・race 21/35。
  - `src/scripts/data` の本番: 72 件（drift_monitor 一式・prediction_results・update_serving_state・hem_megu など）。
  - tp* の名前の 235 件のうち 35 件は本番の経路に入る（例: `pipeline.inference.candidate_runner` → `rebuild_groups_2016` → `tp27_eval`）。
  - 再学習の経路（`config/retrain_plan.json`・`scripts/dev/retrain_upstream.py`）に本番以外の 34 件が入る（`tp53_dataset`・`tp54_*`・`tp61_*`・`fit_final_v4〜v6`・`v6_parity` ほか）。
- 未参照 347 件の内訳: `src/scripts/data` 218・`src/scripts/scraping` 32・`src/research/pedigree` 26・`src/scripts/maintenance` 20 ほか。
  - 2026-07-04（履歴の起点）から更新が無いもの 93 件。docs の言及も無いもの 255 件。両方に当たるもの 74 件。
- 役割の重なり。
  - `src/research/scripts/`（3 件）はキュー投入の古い道具で、`src/scripts/scraping` と同じ役割。3 件とも未参照。
  - `src/scripts/ops/`（19 件）と `src/ops/`（49 件）は名前も役割も重なる。

判定の限界:

- 変数を渡す `importlib`・文字列を連結した `-m`・設定から組み立てる呼び出しは拾えない。
- 「本番」は多めに出る。案内文の中の `python -m`、古い `deploy_v5.sh`、crontab に無い `install_*.sh`、retire のジョブも入口に数えた。
- 「未参照」でも、docs にだけ手順が書かれた手動の道具がある。そのため「未参照」を消す一覧として使わない。消すのは段 3b で、根拠の列を付けてユーザーの承認を得てから。

## 動かすと壊れる所

| 物 | 壊れる先 | 根拠 |
|---|---|---|
| crontab が絶対パスで呼ぶ `scripts/cron/` の 21 本・`scripts/server/{ensure_monitor,server_watchdog,ensure_postgres}.sh` | ホストの root の crontab | `sudo crontab -l` |
| `scripts/auto_scrape.sh`・`scripts/ops/vps/cache_warm.py`・`scripts/ops/vps/templates/` | VPS の取得の配布物と、VPS に入っている `install_ops_files.sh` の許可リスト | `src/ops/release_publish.py:35-50`・`scripts/ops/vps/install_ops_files.sh:41-47`・`tests/ops/test_install_ops_files.py` |
| `frontend/`・`config/ports.env` | UI の配布物（`git archive`） | `scripts/ops/vps/build_release.sh:21` |
| `config/` の 10 個（ports.env・gcp_job_map.json ほか） | VPS の配布物 | `release_publish.py:35-37` |
| `config/` の研究名のファイル（`tp59_frozen_rule.json` など） | Cloud Run Jobs の state_in | `config/gcp_job_map.json:1449` |
| `src/` の中のモジュールの名前 | Jobs の呼び出し（gcp_job_map・scheduled_jobs）・`config/retrain_plan.json`・`config/data_access/ledger.jsonl`・テスト 139 ファイル | 解析の結果 |
| `docs/html/**`・本に載る md・要件定義書 | 本と棚の生成・テスト・update-requirements のスキル | `build_docs_book.py:89-456`・`docs_shelves.py:26-217`・`tests/scripts/test_build_docs_{book,shelves}.py` |
| `main.py`・`wsgi.py`・`templates/`・`static/`・`requirements*.txt`・`constraints-jobs.txt`・`alembic.ini` | Docker・CI | `deploy/docker/Dockerfile.*`・`.github/workflows/ci.yml` |
| 直下の pid・lock | 生きているサービスの管理（watchdog は古い名前に戻って読まない。`server_watchdog.sh:469,574`） | 下の段 2 |

## 案の階層（整理の後）

```
keiba-vpn/
├── README.md AGENTS.md CLAUDE.md
├── main.py wsgi.py service_start Makefile
├── requirements.txt requirements-fetch.txt constraints-jobs.txt
├── pytest.ini alembic.ini docker-compose.{dev,serving}.yml
├── .env.example .env.stg.example .env.prod.example   # 実物の .env・.env.stg は git 外のまま
├── src/ tests/ scripts/ config/ data/ docs/ frontend/
├── deploy/ alembic/ templates/ static/ notebooks/ site_public/ models/ mlflow/ dev_fixture/
├── logs/            # git 外（今のまま）
└── run/             # git 外・新設。pid・lock・profile をここにまとめる
```

直下のファイルは、今の約 60 個（ドットファイルを含む）から 20 個前後になる。

## 段取り

### 実施の時間帯

- ホストの時計は UTC。crontab の時刻も UTC で、`TZ=Asia/Tokyo` はコマンドの中だけに効く。
- 避ける時間帯（JST）:
  - 開催日（土日・祝）の全日と、その前日の夜（金曜の 18:00 以降）。
  - 毎日 03:00〜04:40（メンテナンス・verify-past-values・rotate_logs）、05:00〜09:00（jra-baba-morning・backfill-all-horse・shutuba-audit・daily-race-lists・raceday-runner・barometer-retry）、17:00〜18:00（daily-race-lists・raceday-evening・raceday-eve）。
- 勧める窓: 火〜木の 13:00〜16:00 JST（UTC 04:00〜07:00）。
  - この間に走るのは、短い周期の物だけ: remote-access（2 分）・watchdog（3 分）・ensure_monitor・ensure_postgres・GCS のミラー（5 分）・entries-sync（10 分）・cron-catchup・vps-alive・slack-scrape-alerts（15 分）・backfill-past（毎時 :05）。
- 最も早い実施日は 11/24（火）の 13:00 JST。

### 段 1（安全）: 掃除と文書の食い違いの修正

サービスの再起動は無い。コードの動きも変えない。

1. `.env`・`.env.stg`・控え 6 個を `chmod 600` にする（中身は読まない）。控えを消すか保管かはユーザーが決める。
2. 生成物を消す: `.coverage`・`catboost_info/`・`tmp/` の古いログ 2 個。
3. 古い pid を消す: 上の表の「古い」10 個。当日に `ps -o user,args -p <番号>` と `/proc/<番号>/cwd` で照合し、この作業ツリーのサービスでないことを確かめてから消す。
4. 承認を得た物を git から消す: `keiba-vpn.code-workspace`・`tmp/.gitkeep`・`scripts/run_tp26.py`（`src/research/race/running_profile.py` の docstring の名前も直す）。
5. 文書を直す（1 コミット）。
   - AGENTS.md の「レイアウト」表と `PROJECT_LAYOUT.md`: scripts の下位、config、`data/local/features/`、frontend・deploy・site_public・dev_fixture・run/。
   - `scripts/ops/vps/` の `.py` を決まりの例外として書く。
   - `src/data_health/spec.py:133-140` の「git から取得」を、実際の配り方（`docs/operations/model_distribution.md`）に直す。これはコードの変更なので、critic に差分を見せる。

確かめ方: CI と同じテスト（compileall・check_no_shuffle・pytest）。`git status --ignored` で消した物が無いこと。

戻し方:
- git の分は revert。
- 消した生成物は作り直せる。
- 古い pid は消す前に作業ツリーの外（`~/repo_tidy_removed_<日付>/`）へ写しておき、1 週間後に消す。`run/` は段 2 で権限を決めて作るので、段 1 では作らない。

### 段 2: pid・lock の置き場を `run/` へ移す（サービスの再起動を伴う）

コードの変更:
- pid の置き場を扱う共通のファイルは今は無い（共通の読み込みは `ports.sh` だけ）。新しく `scripts/server/run_dir.sh` を作り、`RUN_DIR="$ROOT/run"` と `mkdir -p "$RUN_DIR"` を置く。
  - 環境変数で置き場を変えられる作りにはしない。root の cron と jovyan とで値が食い違うと、サービスを見失う。
  - `mkdir -p` は worktree や別の PC（`run/` が無い所）で lock・pid の書き込みが落ちないため。
- 8 本（`service_start.sh`・`server_watchdog.sh`・`restart_server.sh`・`start_monitor.sh`・`tunnel_tcpexposer.sh`・`tunnel_cloudflared.sh`・`autostart_dev.sh`・`ensure_monitor.sh`）が `run_dir.sh` を読む。
- 移行の分岐: 新旧の両方を見て、更新時刻の新しい方を採り、新しい置き場へ移してから読む。古い置き場だけにあるときも同じ。外すのは 1 週間後の別のコミット。
- watchdog は、読んだ pid のファイルの名前と、生死の判定をログに出す（確かめに使う）。
- `.gitignore` に `run/` を足す。
- 直す docs: `stg_frontend_prod_build.md`・`raceday_prediction_workflow.html`・`git_management/todo/monitor-quality.md` の pid のファイル名（本は作り直す）。
- テスト: `tests/scripts/test_frontend_build.py` などの pid の扱い。新しい置き場と移行の分岐（新旧の時刻の比べ）のテストを足す。

当日の手順（共有の作業ツリー・統括役の承認の後）:

1. `sudo crontab -l > ~/crontab_backup_<日時>.txt` で控える。
2. `KEIBA_AUTOSTART_DEV=0` にして、IDE・Claude の起動フック（`autostart_dev.sh`・`.cursor/hooks`）がサービスを起動し直さないようにする。手での `service_start` もしない。
3. root の crontab の watchdog と ensure_monitor の 2 行（`KEIBA-VPN-WATCHDOG`・`KEIBA-VPN-MONITOR`）をコメントにする。
4. 走っている watchdog が終わるのを待つ。`pgrep -f server_watchdog` が空で、`logs/` の下の `server_watchdog.running`（`cron_run.sh` の印）が無いこと。フロントのビルドの途中だと 3 分を超える。
5. `sudo git log HEAD..origin/develop` で、取り込む中身（他のセッションのコミットを含む）を確かめる。
6. `sudo install -d -o root -g users -m 2775 run` で作る（root と jovyan の両方が書ける）。pid は写さない。移行の分岐に移させる。
7. `sudo git pull` で取り込む。
8. crontab と同じ形・同じ権限（root）で 1 回流す: `sudo bash scripts/cron/cron_run.sh server_watchdog -- …/server_watchdog.sh`。jovyan で流すと root のプロセスへの `kill -0` が失敗し、生きているサービスを死んだと判定して二重に起動する。
9. 確かめる（正の形で）。
   - `run/` に生きている 6 個の pid がそろい、直下には残っていない。
   - 6 個の番号が `ps` で生きていて、手順 1 の前と変わらない。
   - watchdog のログに、6 個の pid のファイルを `run/` から読み、生きていると判定した行がある。
   - pid のファイルが無いと Flask と Next.js は「監視の対象外」として黙って飛ばされる（`server_watchdog.sh:470,577`）。そのため「起動の行が出ない」だけでは合格にしない。
10. `sudo bash scripts/server/ensure_monitor.sh` を 1 回流し、監視ポータルが二重に立たないことを確かめる。
11. crontab の 2 行のコメントを外す（控え全体で戻さない。間に他のセッションが `install_*.sh` で変えた行を消さないため）。次の 2 周（6 分）のログを見る。
12. `KEIBA_AUTOSTART_DEV` を戻す。

戻し方:
- 手順 3・4 と同じく crontab の 2 行を止め、走っている watchdog の終わりを待つ。
- コミットを revert して `sudo git pull` する。
- `run/` の pid・lock・profile を直下へ写し戻す（必須）。revert すると移行の分岐ごと消え、古いコードは直下しか読まない。写し戻す前に crontab を戻すと、サービスを見失って二重に起動する。
- 写し戻しの後に、手順 8〜10 と同じ形で確かめ、2 行のコメントを外す。

### 段 3: ディレクトリ構成の変更

量が多いので 2 つに分け、別々に承認を得る。

段 3a（小さい物・壊れる先が少ない）:
- `scripts/cron/` の未使用の物を消す。実施の前に crontab（root と jovyan）・gcp_job_map・scheduled_jobs・テストをもう一度検索する。
  - すぐ消せる: `maintenance_notify.sh`（シェルだけ。Python 側は残す）。
  - 参照元と同じコミットで消す: `git_pull_hourly.sh`・`setup_git_pull_cron.sh`（docs の cicd-design.md・monitoring-portal.md・scraping_sla の記述。本は作り直す）。
  - 後回し: `sporthorse_births_nightly.sh`（crontab から外す `--remove` がユーザーの作業待ち）・`jma_weather_run.sh`（gcp_job_map と費用の台帳に残る）。
  - 消さない: `install_lap_forecast_cron.sh`（今動いている 2 行を書く唯一の道具）。
- `scripts/` 直下の一回限りの 2 本（`show_scrape_progress.sh`・`watch_backfill_pace.sh`）を消す。
- 他の運用の道具（`run_after_scrape_missing_5gen_10gen_chain.sh`・`run_all_model_notebooks.sh`・`run_ml_warehouse_prep.sh`・`watch_ml_warehouse.sh`・`setup_claude_user.sh`）を `scripts/dev/` へ移す。
  - 5 本とも `$(dirname "$0")/..` を直下として使うので、`../..` に直す。
  - 参照元も直す: AGENTS.md・`docs/claude_code_multi_pc.md`・`src/scripts/scraping/scrape_missing_5gen_for_10gen.py:45`。
  - `auto_scrape.sh`（VPS の配布物）・`setup_dev.sh`・`setup_prod.sh`（Makefile）は動かさない。
- `src/research/scripts/` の 3 件（未参照・役割は scraping と同じ）を消す。
- `data/skill_logs/` を `docs/skill_logs/` へ移すか git 外にする。書き換え: `.claude/skills/` の SKILL.md 3 つ・`update-requirements/scripts/collect_state.py`・`src/api/flask_app.py:42`・`docs/operations/service-endpoints.md`。

段 3b（src の整理・大きい）:
- 実施の当日に解析をもう一度流し、TSV を作り直してから決める（コードは日々変わるため）。
- 削除の候補は「未参照・docs の言及なし・90 日以上更新なし」の 3 つを満たす物に限る。消す物は一覧でユーザーの承認を得る。消しても git の履歴に残る。
- docs にだけ手順がある物は、docs を直してから消すか、残す。
- 研究の移動（例: `src/research/experiments/`）は、本番の経路に入る tp* の 35 件と、再学習の経路の 34 件を動かさない。移すなら群ごとに小さなコミットにする。
  - 書き換え: テスト 139 ファイル・config 9 ファイル（`retrain_plan.json`・`data_access/ledger.jsonl`・`tp75_frozen_rule_*.json` ほか）・`scripts/dev`。
  - Dockerfile は `COPY src/` なので書き換えは要らない。Jobs のモジュール名を変えるなら、イメージの配備と `live-` タグの付け替えも要る。
- `src/scripts/ops` と `src/ops` の統合は、段 3b の最後に別の計画として扱う。
- docs の木（`docs/html` など）は動かさない。本と棚の生成・テスト・スキルが直書きしているため、動かす効果より壊す危険が大きい。日付付きの一時メモの寄せ先は、docs の整理の担当（`docs_shelves.py`）の方針に合わせる。

確かめ方（段 3 の各コミット）:
- CI と同じテスト一式。
- 動かした物を呼ぶ cron の 1 回の動作（`bash scripts/cron/cron_run.sh <名前> -- …` を手で流す）。
- Jobs に関わるなら stg で 1 回流す。

戻し方: コミットごとの revert。消した物は git の履歴から戻せる。

## 秘密

- `.env`・`.env.stg`・控えの 6 個の中身は読んでいない。この文書にも書かない。
- 秘密に関わる別の気づきは、統括役へ直接報告した（この文書には書かない）。

## 調べた方法と道具

- 直下: `ls -la`・`git ls-files`・`git status --ignored`・`ps`・`sudo crontab -l`、参照元は `git grep`。
- src の分類: import と `-m` と文字列を ast でたどる解析（作業用の一時領域で実行。段 3b の当日に流し直す）。
- scripts・docs・config の地図: 読むだけの調査（別のエージェント）。

## レビューと対応

計画は、作った本人と別のエージェントが点検した（観点: 壊れる所の見落とし・秘密・戻し方・時間帯）。指摘 20 件はすべて取り込んだ。主なもの:

- `install_lap_forecast_cron.sh` は、今動いている cron の 2 行を書く唯一の道具だった → 消す候補から外した。
- 段 2 の確かめを jovyan で流すと、root のサービスを死んだと判定して二重に起動する → root で、crontab と同じ形で流すと書いた。
- pid のファイルが無いと watchdog は黙って監視を飛ばす → 「起動の行が出ない」ではなく、`run/` の 6 個と生死を正の形で確かめるようにした。
- pid を写してから取り込むまでの間にサービスが起動し直すと、古い番号を読む → 写さずに移行の分岐に移させ、新旧は時刻で選ぶ。起動フックも止める。
- 戻し方の「revert の後も古い置き場で動く」は誤り → 写し戻しを必須にし、crontab は 2 行だけ戻す形にした。
- 避ける時間帯が狭かった → 朝 05:00〜09:00・夜中 03:00〜04:40 に広げた。
- `scripts/` 直下の道具を `scripts/dev/` へ移すと、直下の解決（`dirname/..`）が壊れる → `../..` への修正と参照元を書いた。
- 公開する文書に控えのファイル名と権限の状態を書いていた → 消した。権限を直す対象に `.env.stg` を足した。
