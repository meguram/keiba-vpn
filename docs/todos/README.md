# 学習PCでの実環境検証（2026-10-02 対応分）

開発PCで対応した TODO 7 件を、GCS・Redis・PostgreSQL・Next.js が実際に動く**学習PC**で確認するための手順です。
開発PCでは GCP に接続できないため、ここにあるスクリプトは**学習PCで初めて実行される部分**を含みます。

- 対象コミット: `develop` ブランチ（この README を含むコミット以降）
- スクリプトの場所: `docs/todos/verify/`
- 結果は `PASS` / `FAIL` / `SKIP` で出ます。`SKIP` は「前提が無くて検証していない」の意味で、成功ではありません。

## 1. 準備（学習PCで 1 回）

```bash
cd ~/project/myproject/keiba-vpn      # 学習PC上のリポジトリ
git checkout develop && git pull
pip install -r requirements.txt       # redis / psycopg / lightgbm などが入っていること
( cd frontend && npm ci )             # Next.js の検証に必要
make db-up                            # Redis / PostgreSQL（docker）。既に動いていれば不要
```

- `.env` に `GCS_*`（読み取り可能な鍵）と `DEV_PASSWORD` があること。`DEV_SECRET_KEY` は検証スクリプトが一時的に設定するので不要です。
- 実データ確認は環境変数 `RACE_ID` で行います（無くても他の項目は検証できます）。
  - T-013 / T-032: GCS に**予測キャッシュ**があるレース
  - T-029: GCS に**出馬表（race_shutuba）**があるレース
  - 同じIDで両方を満たすレースがあれば、1 つで全項目を確認できます。

> **デプロイ時の必須設定（T-010 の影響）**: `KEIBA_ENV` が `stg` / `prod` の環境では `DEV_SECRET_KEY`（ランダムな 32 文字以上）が
> 未設定だと、ログイン Cookie の署名・検証で `RuntimeError` になり開発者ログインが使えません。
> VPS / stg の `.env.prod` / `.env.stg` に設定済みか、**デプロイ前に必ず確認**してください（`.env.*.example` に項目があります）。
> dev（`KEIBA_ENV` 未設定・dev）は従来どおり警告付きの固定鍵で動きます。

## 2. 実行

```bash
# 全部まとめて（数分〜10 分程度。Next.js の開発サーバーだけ一時ポートで起動して止める）
RACE_ID=202606010101 bash docs/todos/verify/run_all.sh

# Next.js の開発サーバーを起動する項目（T-011）を飛ばす（T-010 / T-032 は TestClient でサーバー不要）
SKIP_SERVERS=1 bash docs/todos/verify/run_all.sh

# 1 項目だけ
bash docs/todos/verify/T-055_circuit_breaker.sh
```

ログは `~/keiba-verify-<日時>.log` に保存されます。**FAIL があればこのファイルを共有してください。**

## 3. 項目ごとの内容と期待結果

| スクリプト | 何を確認するか | 期待結果 | 必要なもの |
|---|---|---|---|
| `T-060_settings.sh` | `config/settings.yaml` の MLflow モデル設定がカタログ 7 件と一致 | 7 モデルが表示され PASS | なし |
| `T-013_harville.sh` | 連対率・複勝率が Harville 式で出る／計算が速い | 18 頭 1 回 1ms 未満（開発PCでは約 0.02ms）。`RACE_ID` 指定時は 連対合計≈2、複勝合計≈3 | 実データ確認のみ GCS 読み取り |
| `T-029_t45_output.sh` | T-45 予測の保存形式に `win_prob` / `place_prob` / `show_prob` が入る | `RACE_ID` 指定時、全頭に 3 項目があり `保存=False` | GCS 読み取りのみ（**保存しない**） |
| `T-055_circuit_breaker.sh` | Redis / PostgreSQL の障害時に待たされない | 下記 | Redis（復旧確認）、`redis` と `psycopg` パッケージ |
| `T-011_middleware.sh` | 開発者限定ページだけ `/login` へ飛ぶ（ゲスト向けページは閉じない） | 22 項目 PASS（開発PCで確認済み） | `frontend/node_modules` |
| `T-010_auth.sh` | FastAPI の書き込み系が未ログインで 401、偽造 Cookie も 401、prod で鍵未設定ならエラー | 12 項目 PASS（開発PCで確認済み） | アプリの import ができる環境（サーバーは起動しない） |
| `T-032_betting.sh` | `/betting` が使う最適化 API の応答形式・エラー応答 | 未ログイン 401／race_id 無し 400／不明レース 404／`RACE_ID` 指定時は 200 と候補一覧（開発PCでは `DEV_PASSWORD` がないため 401 までが対象） | `DEV_PASSWORD`、`RACE_ID` の正常系のみ GCS 読み取り |

### T-055 の期待値（学習PCで最初に実行される部分）

- Redis 無応答（接続は受けるが応答しないローカルサーバーで再現）
  - 失敗 3 回の所要は**各 0.5 秒前後**（1.5 秒未満）
  - ブレーカーが開いた後の 20 回は**最大 20ms 未満**（開発PCでは 0.02ms）
  - 2 秒後に健全な Redis（`REDIS_URL`、既定 `redis://localhost:6379/0`）へ切り替えると往復に成功し、ブレーカーが閉じる
- PostgreSQL 無応答: `DB_CONNECT_TIMEOUT_SEC=2` のとき**約 2 秒で失敗**（6 秒未満）。開発PCで 2.00 秒を確認済み。
- 「復旧確認に失敗」と出た場合は、その直下に例外の詳細が出ます。Redis が起動していない場合も同じ表示になります。

## 4. ブラウザでの目視確認（スクリプトでは確認できない部分）

### T-032: `/betting`

1. Flask API（`bash scripts/server/start_flask_api.sh`）と Next.js（`cd frontend && npm run dev`）を起動し、`/login` から開発者ログインする。
2. `/betting` を開く → **「レースID（12桁）」入力欄**がある。
3. 空のまま「最適化」→ 「レースID は 12 桁の数字で入力してください」と出て、API は呼ばれない（開発者ツールの Network で確認）。
4. 予測キャッシュのあるレースID（例 `RACE_ID`）を入れて「最適化」→ 軍資金・合計賭け金・期待収益・Kelly 比率と、候補の表（馬名または組み合わせ・券種・賭け金・Kelly f・エッジ）が出る。画面が白くならないこと。
5. 存在しないID（`000000000000`）→ 「予測結果がありません…」のエラー文が赤字で出る。
6. ページを再読み込みすると、前回のレースIDが入力欄に復元される。

### T-011: 開発者限定ページ

- ログアウトした状態で `/betting` を開くと `/login?next=/betting` に移動する。`/`、`/races`、`/bloodline` などは通常どおり開ける。

### T-010: 監視ポータルへの影響がないこと

- `http://<学習PC>:9090`（監視ポータル）を開き、キューの状況などが**ログインなしで従来どおり表示される**こと（読み取り系は仕様どおり開放のまま）。
- FastAPI の開発者向けダッシュボード（`/`）で「シミュレーション開始」「オッズ学習」などの POST ボタンを**未ログインで押すと認可エラー（401）になり操作は実行されない**こと。ログイン後は従来どおり動くこと（実行される操作なので、実行してよい状態で試す）。

## 5. スクリプトが行う操作（安全性）

- GCS は**読み取りのみ**。保存・削除はしません（T-029 は `persist=False`）。
- T-010 の「ログイン済み」確認は `POST /api/scrape-queue/add` に空の `{}` を送るだけで、入力検証で 400 になり**キューに何も入りません**（スクリプトは開発PCのネットワーク遮断環境でも全項目 PASS を確認済み）。
- FastAPI と Flask はサーバーを起動せず、アプリをプロセス内で呼びます（FastAPI のキューワーカー等の `lifespan` も動きません）。Next.js の開発サーバー（T-011）だけ一時ポートで起動し、終了時に必ず停止します。
- T-055 の障害再現は 127.0.0.1 の一時 TCP サーバーのみ。本物の Redis / PostgreSQL は止めません。

## 6. この検証の対象外（今回は見送った TODO）

| TODO | 理由 |
|---|---|
| T-005 `publish_model` の CLI 化 | GCS へ公開するコマンドのため、開発PCでは実装・検証しない方針。学習PCで対応する |
| T-025 `/pedigree-race-stats` の API 接続 | 接続先 API が SQLite 経路でスタブ応答（種牡馬別集計は未設計）。設計が先に必要 |
| T-034 SmartRC 除去 | 約 20 ファイルに及ぶため、影響範囲の確認が先 |
| T-061（新規） 勝率 softmax の統一 | 学習済みモデルの出力尺度の確認が必要（学習PCで実モデルを使って判断） |

TODO 全体の一覧と優先順位は `docs/html/index.html` の §14 を参照してください。

## 7. 実行結果（学習PC・2026-10-02〜04）

結果の要約は `docs/todos/verify/results/` にあります（`summary-20261002-163452.md` 初回、`summary-20261004-135912.md` 修復再実行、`summary-20261004-racedata.md` 実データ確認）。

- 7 項目すべて OK（T-011 / T-032 は初回 NG → 原因は Windows マウント上の `node_modules` 破損。ネイティブ Linux 側で `npm ci` して解消）。
- 実データ確認で `POST /api/v1/betting/optimize` の `KeyError: 'roi_pct'` を検出・修正済み。回帰テスト: `tests/api/test_optimize_betting.py`。
- **未実施（次にやること）**: T-055 の「健全な Redis への復旧確認」。Redis が起動していなかったため、復旧の往復確認は失敗表示のまま。
  `make db-up` などで Redis を起動して `bash docs/todos/verify/T-055_circuit_breaker.sh` を再実行し、`復旧後の往復` が表示されることを確認する。
- 結果ファイルに **鍵・パスワード・トークンを書かない**（差分やログを貼るときは値をマスクする）。

## 8. データ網羅性・スキーマの検証（別手順）

2020 年以降の全データポイントが GCS に揃い、スキーマに適合しているかの確認は、本書とは別に
[`learning_env/data_check.md`](learning_env/data_check.md) にまとめています（`bash docs/todos/verify/stg_data_complete.sh`、
`python -m src.scraper.schema_infer`、`python -m src.data_health`）。前提は「netkeiba プレミアム認証は完了」「GCS のアクセスポイントは stg と prod で同一」です。

スクレイピング（不足データの取得）も同じ手順書の手順 5 にあります: `bash docs/todos/verify/stg_scrape_missing.sh`（既定はドライラン）。
エラーには敏感に対応し、「ページが存在しない」以外はアクセス制限とみなして即終了 → その時点のデータチェック → 制限解除後に `RESUME=1` で再開します。
スキーマで弾かれたサンプルの保存先は同手順書の 5-B にまとめています。
