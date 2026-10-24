# 学習PC（stg）で実行するデータチェック（2020 年以降の全データポイント）

開発PC（dev）では GCP に接続できないため、**GCS 上の実データが「2020 年以降すべて揃い、スキーマに適合している」か**は学習PCでしか確認できません。
この手順書は、開発PCで整合性を確認したあと（2026-10-06 時点）に、学習PCで実行するものを順番に並べたものです。

- 設計・判定の詳細: `docs/html/data/DATA_HEALTH.html`（確認できること）／`docs/html/data/DATASET_CATALOG.html`（データセット一覧とサンプル）
- 既存の学習PC検証（認証・ミドルウェアなど）は `docs/todos/README.md`。本書はそれとは別の**データ網羅性・スキーマ**の検証

## 0. 前提

| 項目 | 前提 |
|---|---|
| netkeiba | **プレミアム認証は完了している**（`.env.stg` に `netkeiba_id` / `netkeiba_pw`）。タイム指数・調子偏差値・パドック・個別ラップも通常の取得対象 |
| GCS | **アクセスポイント（バケット・prefix）は stg と prod で同一**。学習PCで確認した「データの有無・スキーマ適合」は prod にもそのまま当てはまる。**stg の書き込みは prod が読むデータになる**ので、再取得は上書きに注意 |
| 期間 | 2020-01-01 〜 **実行日の前日**（既定。実行当日以降のレースは対象外） |
| コード | 開発PCの変更（`src/data_health/`、`src/scraper/schema_*`、`tests/` など）が **develop に commit・push 済み**で、学習PCで pull してあること（**開発PCでは未コミット**。先に commit → push する） |

## 1. 準備（学習PCで 1 回）

```bash
cd ~/project/myproject/keiba-vpn
git checkout develop && git pull
pip install -r requirements.txt

# .env.stg（git 管理外）に GCS と netkeiba の認証、KEIBA_ENV を用意。鍵の値はログに出さない
#   GCS_BUCKET / GCS_PRIVATE_KEY / GCS_CLIENT_EMAIL ... / netkeiba_id / netkeiba_pw / DEV_SECRET_KEY(32文字以上)
#   GCS_PRIVATE_KEY は改行を \n にエスケープした 1 行（過去に 1 文字欠落で接続できなかった事例あり）
export KEIBA_ENV=stg

# 任意（.env.stg に書いてもよい）。既定値のままで動く
#   DATA_HEALTH_VALIDATE=full            全件検証（既定）
#   DATA_HEALTH_VALIDATE_BUDGET=5000     1 回の download 上限（0=無制限）
#   DATA_HEALTH_VALIDATE_WORKERS=8
```

- ディスクは数 GB 空けておく（台帳 `ledger/`・キーテーブル・結果ファイル。VPS ではなく学習PC）。
- 全件検証は **GCP の外から download すると送信料金（約 $0.12/GB）** がかかる。量は未実測。手順 3 で件数と GB を記録する。

## 2. 手順

各手順の「期待結果」を満たさなければ、次へ進まず「NG のとき」を見る。結果は手順 7 の表に記録する。

### 手順 1: 定義とテストの確認（数十秒）

```bash
python3 -m pytest tests/data_health tests/scraper/test_schema_infer.py tests/scraper/test_dev_mode.py \
  tests/scripts/test_embed_catalog_samples.py -q -p no:cacheprovider
python3 -c "from src.scraper import schemas; print(schemas.schema_fingerprint(), schemas.SCHEMA_VERSION, len(schemas.SCHEMAS))"
```

- 期待: 全件 PASS。fingerprint が **`ef48c8a2bedc`**（開発PC・2026-10-06 時点。手順 3 の `apply` 後は変わる）と一致＝全環境で同じ定義。
- NG: fingerprint が違う → `git pull` が足りない（`src/scraper/schema_defs.json` を確認）。

### 手順 2: 疎通と軽量チェック（sample モード。download は少量）

```bash
DATA_HEALTH_VALIDATE=sample python3 -m src.data_health --no-horses
```

- 期待: 「インフラ」で **GCS 接続 OK**（バケット名と応答 ms）、**PostgreSQL / Redis** が OK、`env.gcs_bucket` / `env.gcs_credentials` / `env.dev_secret_key` が OK。
- 期待: 「レース存在カバレッジ」が年ごとに出る（2020〜今年）。
- NG（GCS 接続）: `GCS_BUCKET`・鍵・権限を確認（上記の鍵の 1 文字欠落に注意）。PostgreSQL（Cloud SQL 未作成 T-016）・Redis 未起動は FAIL になるが、データの確認自体は続けられる（`--no-infra` で省略可）。
- 結果は `data/local/meta/data_health/stg/latest.html` に出る。

### 手順 3: 実データからスキーマを再構成する（スキーマ定義の見直し）

スキーマ定義（`src/scraper/schema_defs.json`）は手書きの初期定義から始まっている。**実データで食い違いを洗い出し、定義に反映して git で全環境に配る。**

```bash
# 3-1. 実データを走査して観測プロファイルを作る（download: カテゴリごとに最大 200 件）
python3 -m src.scraper.schema_infer collect --source storage --per-category 200 \
  --years 2020,2021,2022,2023,2024,2025,2026

# 3-2. 現行スキーマとの差を見る（適合率・未定義キー・必須の出現率・型の食い違い）
python3 -m src.scraper.schema_infer report

# 3-3. 反映内容を確認してから反映（既定は保守的: 新カテゴリ(advisory)と未定義キーの任意追加のみ）
python3 -m src.scraper.schema_infer apply --dry-run
python3 -m src.scraper.schema_infer apply            # 必要なら --promote（任意→必須）/ --demote（必須→任意）
```

- 期待: `report` で各カテゴリの「N/N 適合」。**不適合・型の食い違い（`type_conflict`）・一度も観測されない定義済みキー（`never_observed`）・必須の出現率不足（`demote_candidate`）** は人が判断する（自動では直さない）。
- 対象: スキーマ未定義だった `race_predictions` / `tracking_difficulty` / `final_odds_prediction` / `finish_order_prediction` / `race_performance` / `jra_cushion` は、ここで **advisory（保存を止めない）** として追加される。
- 反映後は定義が変わる → 手順 1 のテストを再実行し、**`schema_defs.json` と `docs/requirements/data/schemas/observed/` を commit → push**（開発PC・VPS も同じ定義になる）。開発PCでは `python -m src.scripts.docs.embed_catalog_samples` と `make dev-mock` を再実行してサンプルを更新する。
- **厳格化**: advisory のカテゴリは、適合率が十分（下記の合格基準）になってから `schema_defs.json` の `"advisory": true` を外す。外す前に手順 4 で既存データの適合を確認する。
- 保存時に拒否されたデータがあれば `python3 -m src.scraper.schema_violations summary` で「どの項目がどの値か」を見る。隔離データ（`data/local/quarantine/`）も観測の材料になる。

### 手順 4: 2020 年以降の完全性の判定（本番の確認。全件 download）

```bash
# 全件を一気に（download 量が大きい）
bash docs/todos/verify/stg_data_complete.sh

# 1 回 5000 件ずつ、全件になるまで繰り返す（途中で止めても台帳から再開）。初回はこちらを推奨
BUDGET=5000 MAX_RUNS=100 bash docs/todos/verify/stg_data_complete.sh

# 直接呼ぶ場合（0=完全 / 3=未達 / 2=FAIL）
python3 -m src.data_health --require-complete
```

- 判定: 2020-01-01〜前日の JRA レース全件 × 重要度 `required`/`recommended` のカテゴリで、**不足・スキーマ不適合・未検証が 0**、出走馬・派生データ・インフラに FAIL なし、race_lists に不完全な日なし、全件検証（`full`）が最後まで終わっている。
- 実行中、`validated_now`（今回検証した件数）が減っていき、最終的に 0 になる。**件数・download した GB・所要時間を記録する**（課金の見積りに使う）。
- 台帳（`data/local/meta/data_health/stg/ledger/`）は消さない。消すと全件再検証になる。

### 手順 5: 不足・不適合の把握 → スクレイピング → 再チェック（`stg_scrape_missing.sh`）

手順 4 で見つかった「取得すべきデータ」を、**データチェックの結果をもとに**スクレイピングして埋める。1 回の実行で
**チェック → 対象の特定 → 事前確認 → キュー投入 → 取得 → 再チェック**を行い、進捗が無くなるまで（または完全になるまで）繰り返す。

```bash
# 5-1. まずドライラン（既定）。何を・どれだけ取得するか（ジョブ数・取得リクエスト数・所要時間の目安）を表示するだけ。何も書き込まない
bash docs/todos/verify/stg_scrape_missing.sh

# 5-2. 少数で実行して様子を見る（優先度の高い順に 20 件）
EXECUTE=1 MAX_JOBS=20 bash docs/todos/verify/stg_scrape_missing.sh

# 5-3. 本番: チェック → 取得 → 再チェックを最大 5 回（進捗が無ければ自動で止まる）
EXECUTE=1 ROUNDS=5 bash docs/todos/verify/stg_scrape_missing.sh        # 件数が多い・上書きを含むときは確認が出る（非対話なら YES=1）
```

（中身は `python -m src.scripts.scraping.scrape_from_health_plan`。終了コード 0=完全 / 3=まだ不足あり（ドライランも 3）/ 4=事前確認で中止 / 5=アクセス制限で中断 / 6=確認が取れず中止）

**連携の仕組み（データチェック → スクレイピング）**

| 段階 | 内容 |
|---|---|
| 1 事前確認 | `KEIBA_ENV=stg`（dev は書き込めない／prod は拒否）、GCS 接続（保存されずに「成功」扱いになるのを防ぐ）、netkeiba 認証情報、他のワーカーが動いていない、アクセス一時停止中でない |
| 2 チェック | データチェックを実行して最新の計画 `scrape_plan.json` を得る（台帳により更新分だけ検証）。`--plan` で保存済みの計画も使える |
| 3 絞り込み | 実行場所・状態・カテゴリ・期間・推定の有無・件数（下表） |
| 4 確認 | 内訳と見積りを表示。ドライランはここで終了 |
| 5 実行 | `bulk_add_jobs` でキュー投入 → `process_queue()` で取得。**不足は既存を上書きしない**（`smart_skip=true・overwrite=false`）、**スキーマ不適合だけ上書き再取得**（`overwrite=true`）。開催日に 6 レース以上の不足があれば `date_all` 1 本にまとめる |
| 6 再チェック | チェックをやり直し、残りを確認。前回より減っていなければ停止（取得先に存在しない・スキーマ側の問題など、繰り返しても解消しないもの） |

| 絞り込み（環境変数） | 内容 |
|---|---|
| `STATUS=missing\|invalid\|calendar` | 不足のみ／スキーマ不適合の上書き再取得のみ／race_lists の不完全のみ |
| `CATEGORY=race_result,race_index` | カテゴリを限定 |
| `SINCE=2024-01-01` `UNTIL=…` | 対象日の期間 |
| `NO_INFERRED=1` | 連番から推定しただけのレース（race_lists に無い）を除く。存在しない可能性があるので、まず除いて試してもよい |
| `REQUEUE_FAILED=1` | 同じジョブが失敗のまま残っていたら待機に戻す（重複扱いで動かないのを避ける） |

- 連携の注意（確認済み）: (1) `取得できない` ことが確定したデータは、スクレイパーが `_meta.not_available` の**スタブ**を GCS に保存する。
  データチェックはこれを**「取得不可（na）」**として扱い、健全にも不足にも数えない（再取得を繰り返さない）。(2) キューの重複判定は `種別:ID:タスク` で、**失敗のまま残っているジョブは重複扱いで動かない**（`REQUEUE_FAILED=1`）。
  (3) 再取得の計画は全件から作る（レポートの不足一覧は先頭 5000 件に切るが、計画は切らない）。(4) 馬のデータは直近〜出走予定の出走馬のみ計画に入る（古いレースの馬は `date_all` の一式取得に含まれる）。

#### 5-A. スクレイピング中にアクセス制限の疑いが出たとき（即終了 → チェック → 再開）

**方針: スクレイピング中のエラーには敏感に対応する。「ページが存在しない」以外のエラーは、基本的にアクセス制限とみなす。**
スクリプトは netkeiba への全リクエストを監視し、次を検知したら**その場で全リクエストを止めて**終了する（既定は 1 回で即時）。

| 検知するもの | 備考 |
|---|---|
| HTTP ステータス 400 以上（**404 を含む**、403、429、5xx など） | **本文が「ページが見つかりません」「該当データはありません」「No Data」等の応答だけは、存在しないページとして除外**（`--strict-not-found` = `STRICT_NOT_FOUND=1` で除外しない） |
| 通信エラー（接続拒否・タイムアウト・リダイレクト過多など） | 自分側のネットワーク不調でも止まる（敏感な設計。`STOP_AFTER=2` などで緩められる） |
| 2xx でも本文が「アクセスが制限されています」「アクセスが集中」「Access Denied」のページ | |
| キュー側の既存検知（HTTP 400 のブロック疑い）でアクセス一時停止フラグが立った場合 | 同じ後始末をする |

- 監視の調整: `STOP_ON_STATUS=any`（既定）／`404,403` のように限定／空文字で監視しない。`STOP_AFTER=N`（何回連続で止めるか。正常な応答でリセット）。
- 本スクリプトの内部では、`NetkeibaClient` の応答オブザーバ（`src/scraper/access_guard.py`）が `BaseException` 系の例外を送出するため、通常のスクレイパーが `except Exception` で握りつぶして次へ進むことはない。一度検知したら以降のリクエストはすべて同じ例外で即失敗する（送信もしない）。

**検知したときの動き（自動）**

1. 実行中だったジョブを**待機に戻し**、**アクセス一時停止フラグ**を立てる（cron・他のワーカーも止まる）
2. **その時点のデータチェックをやり直す**（取得できた分が反映され、残りの不足が最新になる）
3. 再開用の状態と計画を保存し、終了コード 5 で終了する
   - `data/local/meta/data_health/stg/access_restriction.json` … 検知した URL・ステータス・時刻、今回の実行条件（引数）、残り件数、再開コマンド
   - `data/local/meta/data_health/stg/scrape_plan.after_restriction.json` … **制限が解除された後に実行すべき計画**（最新の状況に基づく）

**制限が解けたあとの再開**

```bash
# 1) すぐに再実行しない。しばらく時間をおき、ブラウザで netkeiba にアクセスできることを確認する（プレミアムのログインも）
# 2) 再開（疎通確認 → 一時停止フラグの解除 → 前回と同じ条件で、最新のチェックに基づいて続きを実行）
RESUME=1 CLEAR_PAUSE=1 bash docs/todos/verify/stg_scrape_missing.sh
#    一時停止フラグは UI の「再開」で解除してもよい（その場合は CLEAR_PAUSE 不要）。疎通確認を省くなら SKIP_PROBE=1
#    条件は前回を引き継ぐ。変えたいものだけ指定する（例: 件数を絞って様子を見る）
RESUME=1 CLEAR_PAUSE=1 MAX_JOBS=10 bash docs/todos/verify/stg_scrape_missing.sh
```

- 疎通確認で繋がらなければ何も実行せず終了する（フラグは解除済み・状態は残る）。再開に成功したら状態は `access_restriction.resolved.<日時>.json` に退避される。
- 再開後にまた制限を検知した場合は、同じ手順でまた即終了して状態を更新する。**短時間に何度も繰り返す場合は、間隔を空けるか、`MAX_JOBS` を小さくして様子を見る**。
- 対応が終わったら、実行ログ（`scrape_runs/<日時>.json` の `restriction`）に検知内容が残っている。頻発するなら `.env.stg` の `NETKEIBA_THROTTLE_MIN/MAX` を大きく、`SCRAPE_QUEUE_PARALLEL` を小さくする。

#### 5-B. スクレイピング中に型エラー等でスキーマに弾かれたサンプルの保存先

保存時にスキーマ検証で引っかかったデータは、**どの項目がどの値で引っかかったか**を次の場所に残す（git 管理外の `data/local/` 配下）。

| 保存先 | 内容 | 見方 |
|---|---|---|
| `data/local/meta/schema_violations/<category>.jsonl` | **1 回の拒否につき 1 行**。日時・key・判断（`rejected`=保存せず / `saved_advisory` / `saved_lenient`）・環境・違反の一覧（項目名・規則・期待・**実際の値**・型・何番目の要素か・馬番など）。5MB で 1 世代ローテーション | `python3 -m src.scraper.schema_violations summary`（項目×規則ごとに、どの値で何回）／`show <category> <key>`／`tail -n 20` |
| `data/local/quarantine/<category>/<key>.json` | **拒否して保存しなかったデータ本体**（違反つき）。同じ key が後で合格して保存されたら自動で削除 | 開いて実際の HTML 由来の値を確認。スキーマの見直し（`schema_infer collect`）の材料にもなる |
| 保存した JSON の `_meta.schema_validation.violations` | advisory のカテゴリ、または `KEIBA_SCHEMA_STRICT=0` で保存した場合は、データ自体にも違反が残る | GCS 上の JSON の `_meta` |
| スクレイパーのログ行 `取得失敗 [<category>/<key>]: [schema_validation] … :: <項目>: type（期待 … / 実際 str '…'）` | コンソール（`stg_scrape_missing.sh` は `~/keiba-scrape-<日時>.log` にも保存）と、ワーカーのログ `data/queue/.worker_log_ring.jsonl`（UI の `/api/scrape-queue/worker-logs`） | ログの検索は `grep 取得失敗 ~/keiba-scrape-*.log` |
| スクリプトの実行ログ `data/local/meta/data_health/stg/scrape_runs/<日時>.json` | `rounds[].schema_rejections`（この実行中に拒否された件数・判断・上位の項目と値）、`outcome`（失敗ジョブと理由）、`restriction`（アクセス制限） | `python3 -c "import json;…"` または直接開く |
| データヘルスのレポート | `race_ids/invalid_*.txt`（保存済みでスキーマ不適合の race_id）・`invalid_detail.json`（原因と値）・`latest.html` の「保存時に引っかかった記録」 | 手順 4・5 の再チェック後に更新される |

- **注意: スキーマ拒否ではジョブは失敗にならないことが多い。** `ScraperRunner._fetch_parse_save` は保存時の例外を 1 カテゴリの失敗としてログに出して次へ進むため、
  キューのジョブは `completed` になる（`failure_reason=schema_validation` でジョブが失敗になるのは、例外が外まで届く経路だけ）。
  **拒否されたかどうかは、ジョブの状態ではなく上の表（特に `schema_violations` の記録と `quarantine/`）で確認する**。再チェックでも「不足」のまま残る。
- 原因の切り分け: 同じ項目・同じ値が多数の key で出る → サイト側の形式変更か定義の誤り（手順 3 に戻る）。特定の key だけ → そのページ固有の値（quarantine の本体を確認）。

### 手順 6: 結果を開発PC・VPS に持ち帰る

```bash
# 学習PC
ls data/local/meta/data_health/stg/latest.json
# 開発PC（結果ディレクトリごと scp 等で持ち込み。GCP には触れない。latest.json 単体でも取り込めるが概要のみ）
python3 -m src.data_health --import ~/Downloads/stg/
# → data/local/meta/data_health/index.html と dashboard.html に stg の最新結果と推移が並ぶ
```

- **ダッシュボードで見る**: 学習PC・開発PCとも `data/local/meta/data_health/dashboard.html` をブラウザで開くだけ（サーバ不要）。dev / stg(=prod) をタブで切り替え、ヒートマップのセルをクリックすると該当のレース一覧に絞り込まれる。
  手順 4・5 の実行中は、健全性の検証やスクレイピングの進捗が自動で表示される（画面は約 30 秒ごとに自動更新。更新されるのは**チェックを実行した時点の結果**で、ブラウザから GCS は見ない）。
  開発PCで stg のレース単位まで見るには、`latest.json` 単体ではなく**結果ディレクトリごと**取り込む: `python3 -m src.data_health --import ~/Downloads/stg/`（`race_keys.csv`・`access_restriction.json`・`scrape_runs/` も入る）。
  常に新しく保ちたい場合は cron で `DATA_HEALTH_VALIDATE=sample python3 -m src.data_health` を定期実行する。
- VPS（prod）は GCS が**同一**なので全件検証は不要。到達性と直近分だけ確認する（別 TODO: VPS の `GCS_*` 設定 T-003 のあと）:
  `KEIBA_ENV=prod DATA_HEALTH_VALIDATE=sample python3 -m src.data_health`

### 手順 7: 結果の記録

| 項目 | 記入 |
|---|---|
| 実行日時 / commit / host | |
| 手順 1: テスト / schema fingerprint（apply 前 → 後） | |
| 手順 2: GCS 接続 / PostgreSQL / Redis | |
| 手順 3: 不適合・型の食い違い・never_observed の件数と対応 / advisory を外したカテゴリ | |
| 手順 4: 対象レース数 / 検証件数 / download した GB / 所要時間 / 回数 | |
| 手順 4: 完全性 OK/NG と未達の理由 | |
| 手順 5: 再取得したジョブ数（不足・上書き）/ ラウンド数 / 再実行後の結果 | |
| 手順 5: 保存時にスキーマで拒否された件数と上位の項目・値（`schema_violations summary`） | |
| 手順 5-A: アクセス制限の検知の有無（URL・ステータス・時刻）/ 再開までの時間 | |
| 課金の実績（送信・操作）と見積りとの差 | |
| 発見した問題（T- 番号を付ける） | |

## 3. 合格基準

- [ ] 手順 1: テスト全件 PASS、schema fingerprint が全環境で一致
- [ ] 手順 2: GCS に接続できる（`gcs.connect` OK）
- [ ] 手順 3: 実データでの食い違いを確認し、定義を反映して push 済み（advisory のまま残すカテゴリは理由を記録）
- [ ] 手順 4: `stg_data_complete.sh` が **終了コード 0（完全性 OK）**
- [ ] 手順 4: `race_ids/` に `missing_*` / `invalid_*` / `unvalidated_*` が無い（または対応済みの記録がある）
- [ ] 手順 5: ドライラン → 少数実行（`MAX_JOBS=20`）で取得・再チェックの流れを確認し、本実行で完全性 OK になった（残りがあれば理由を記録）
- [ ] 手順 5-A: アクセス制限の疑いが出た場合は、即終了 → 再開手順で続きを実行できた（出なかった場合は未確認として記録）
- [ ] 手順 6: stg の結果を開発PCの `index.html`・`dashboard.html` に取り込んだ

## 4. 既知の制約（この検証で分からないこと）

- 検証できるのは**スキーマで表せる範囲**（必須キー・型・非空・最小最大・パターン）。値の意味（確率が 0〜1 か等）や、モデルがロードできるか（`models/keiba_model.pkl` は特徴量次元不一致で読めない既知の問題 T-062）は見ない。
- **race 単位のカテゴリが中心**。馬単位（`horse_result` など）は、直近 30 日〜14 日先の出走馬の**存在**のみで、全馬のスキーマ適合は未対応。
- 派生 11 カテゴリ・`requirement_row_trace`・`horse_name` は対象外（親カテゴリが揃えば再生成できる）。SmartRC は取得中止。
- 連番からの欠損推定は「開催回の日は 1 日目から連続・各日 12R」を仮定（race_lists がある日は race_lists を優先）。
- 全件検証の download 量・所要時間は**未実測**。手順 4 の記録が最初の実測値になる。
- スクレイピング実行スクリプトの取得リクエスト数・所要時間は**目安**（1 リクエスト約 3.1 秒・同時 1 本で計算）。`date_all` は 1 開催日あたり約 300 リクエストと仮定している。
- アクセス制限の検知は**敏感**（「ページが存在しない」以外のエラーで即停止）。誤検知で止まることはあるが、見逃して制限を悪化させるより安全という方針。実機での挙動（netkeiba が制限時に返す実際の応答）は**未確認**なので、初回の実行結果（`scrape_runs` の `guard.responses`）を記録して調整する。
- 派生データ（特徴量 Parquet・血統成果物など）は存在のみ確認し、中身は見ない。dev に無いものは、ここで初めて実在を確認する。

## 5. トラブルシュート

| 症状 | 原因と対処 |
|---|---|
| `gcs.connect` が FAIL | `GCS_BUCKET` / 鍵 / 権限。`GCS_PRIVATE_KEY` の 1 文字欠落（過去事例）。`env.gcs_credentials` が WARN なら ADC にフォールバック中 |
| 完全性が `validate_incomplete` だけ NG | 検証が途中。手順 4 を再実行（台帳から続き）。`BUDGET=0` で一気に進めてもよい |
| `invalid_*` が大量 | 定義の誤りか、サイト構造の変更。`schema_violations summary` の値を見て手順 3 へ（まず定義側を疑う） |
| 日付が「YYYY-??」の行が多い | race_lists が無く、まだ検証していないレース。検証が進むと月に振り分けられる。race_lists の取得（`daily-race-lists`）も確認 |
| 完全性が `range` で NG | `--since` を 2020-01-01 より後に指定している、または `--until` が前日より前 |
| 再取得しても `rejected` | 保存時のスキーマ検証で拒否。`data/local/quarantine/<category>/<key>.json` と `schema_violations show <category> <key>` で値を確認 |
| PostgreSQL / Redis が FAIL | Cloud SQL 未作成（T-016）/ Redis 未起動。データ確認だけなら `--no-infra` |
| `stg_scrape_missing.sh` が「事前確認で中止」 | dev/prod で実行している／GCS に接続できない／`netkeiba_id`・`netkeiba_pw` 未設定／別のキューワーカー（API サーバの常駐ワーカー）が動いている／アクセス一時停止中。表示された `[ERROR]` の行に従う |
| すぐ「アクセス制限の疑い」で止まる | 自分側のネットワーク不調（通信エラーも検知する）、または 404 等が実際に返っている。実行ログ `scrape_runs/*.json` の `restriction`（URL・ステータス）を確認。誤検知が続くなら `STOP_AFTER=2`、特定ステータスだけ見るなら `STOP_ON_STATUS=403,429` |
| 再開で「一時停止フラグが残っています」 | 制限が解けたことを確認して `CLEAR_PAUSE=1` を付ける（または UI の「再開」） |
| 再開で「疎通確認: NG」 | まだ繋がらない。時間をおいて再実行（フラグは解除済みで状態は残る）。確認を省くなら `SKIP_PROBE=1`（非推奨） |
| 取得しても「不足」が減らない | 取得先にそのデータが存在しない（N/A スタブが保存されれば「取得不可」になる）、またはスキーマ拒否（`schema_violations summary` と `data/local/quarantine/` を確認）。推定レースなら `NO_INFERRED=1` で除く |
