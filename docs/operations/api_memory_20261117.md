# 本番 API のメモリの張り付きと 504 の再発防止（2026-11-17）

担当: api-stability セッション（統括役の依頼）。進捗: `/home/jovyan/work/progress/tasks/api-stability.json`。

## 1. 起きたこと

- 2026-11-15 21:47〜11-16 00:37 JST、本番 `keiba-api` の 1 台（`keiba-api-00205-mur`）のメモリが 2GiB の 99% に張り付き、CPU 100%。
  GCS の時間切れ（`SSL: UNEXPECTED_EOF_WHILE_READING`・BrokenPipe、`Timeout of 10.0s exceeded`）が続き、504 が 54 件。
- 2026-11-16 12:47 JST にも別の 1 台が OOM（`Out-of-memory event detected in container` → signal 9）。
- 毎朝 07 時台の `keiba-cache-warm`（VPS の nginx のキャッシュの取り直し。2026-11-15 は 2,899 件、うち 200 が 1,503 件・401 が 1,396 件）の後、約 1.1GiB 増えたまま下がらない。
- 応急の手当て（統括役が実施・ユーザー承認済み）: 本番の 100% を、メモリ 4Gi の `keiba-api-mem4g-1117`（イメージは `live-prod` と同じ digest）へ切り替えた。

## 2. コードから分かった原因の候補

| 候補 | 内容 |
|---|---|
| 障害検知が効かない | GCS の失敗の記録（`_gcs_last_failure`）とクライアントが `HybridStorage` のインスタンスごと。API の多くの経路は要求のたびに `HybridStorage()` を作るので、毎回「失敗 0 件」から始まり、クライアントも毎回作り直す。`load()` は不調のときも、写しが無ければ GCS を読みに行く。 |
| 時間切れが長い | `HybridStorage` の retry の deadline が 10 秒。ミラー（`GcsBackend`）の list・download は timeout・retry の指定が無い（ライブラリの既定）。1 vCPU では `to_thread` のスレッドが少なく、止まった処理が溜まるだけで 504 になる。504 の後もスレッドの処理は続く。 |
| 取り込みの待ち | `_gate` は「最後に成功してから 5 分以上」で最大 8 秒待つ。失敗が続くと、要求のたびに 8 秒待つ。 |
| メモリの溜まり | load キャッシュ（`LOAD_CACHE_MAX_ENTRIES=2000`。件数の上限で、期限切れの項目は消えない）、`/tmp/keiba-prerender`（上限 300MB）、ミラーで取り込んだファイル。Cloud Run では `/tmp` と書き込み層もメモリに数える（RSS には出ない）。 |
| 病んだ 1 台が残る | liveness probe が無い。cache-warm は 5xx でも次の周（60 秒後）に同じキーを投げ直す。 |

## 3. メモリの量（4Gi）とその判断

- 本番の配備（`scripts/ci/deploy_gcp.sh api-revision --target prod`）は `MEMORY=4Gi`（`PROD_MEMORY` で上書き可）。stg は 2Gi のまま（直しで 2Gi に収まるかを stg で確かめるため）。
- `deploy_cloud_run_api.sh promote` は、メモリが `PROMOTE_MIN_MEMORY`（既定 4Gi）未満のリビジョンへ本番の通信を切り替えない。
- **費用**: 課金エクスポートで、サービスのメモリ代は 2026-11-07〜17 に 1 日 0.8〜4.8 円（stg 込み）。本番を 2 倍にして、最悪 +150 円/月、ふだんは +30〜60 円/月。ユーザーが承認済み（2026-11-17、統括役経由）。
- **2Gi に戻すかは、ユーザーの判断**。条件の目安: 再発防止の直し（下の 4 章）が本番に入り、1 週間（レース日の土日と朝の温めを含む）の `memory.current` の最大が 2GiB の 70% 未満であること。戻すときは `scripts/ci/deploy_gcp.sh` の `${PROD_MEMORY:-4Gi}`（api-revision と promote の 2 か所）を 2Gi にする（ワークフローは `PROD_MEMORY` を渡していないので、既定を直す）。promote の確認の閾値も同じ値になる。手で promote するときは `PROMOTE_MIN_MEMORY=2Gi` を付ける。

## 4. 直し（段ごと）

| 段 | 内容 | 状態 |
|---|---|---|
| 1 | 本番は 4Gi・promote の確認。メモリの内訳の口 `GET /ops/memory`（共有ヘッダが要る。VPS の nginx は `/ops` を通さない） | 実装済み |
| 2 | 障害検知（状態とクライアントの共有・すぐ返す・503 + no-store・クライアントの作り直し・時間切れの短縮）、`_gate` | 予定 |
| 3 | キャッシュの上限（バイト数・期限切れの掃除・prerender） | 予定 |
| 4 | 自分で終わる仕組み、`_fill_predicted_odds`、cache-warm の間隔 | 予定 |

## 5. 測り方と結果（stg）

- 2026-11-15 の 07〜08 時の cache-warm の要求 2,899 件から、等間隔に 300 件を抜き、stg（タグ `rstg`）へ 1 秒に 2 件で流す。
  共有ヘッダ付き・Cookie なし（cache-warm と同じ匿名の扱い）・`X-Keiba-Fresh: 1`。25 件ごとに `/ops/memory` を記録する。
- stg への要求は全体で 1,000 件以内。同じ東京リージョンの中なので GCS の転送料はかからない。

（結果は各段の後に追記する）
