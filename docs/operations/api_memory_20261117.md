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
| 1 | 本番は 4Gi・promote の確認。メモリの内訳の口 `GET /ops/memory`（共有ヘッダが要る。VPS の nginx は `/ops` を通さない） | develop・stg に反映（56270a3b） |
| 2 | GCS の障害: 健全性とクライアントの共有（`KEIBA_GCS_SHARED_STATE=1`）、不調の間はすぐ返す（`KEIBA_GCS_FAIL_FAST=1`、15 秒に 1 件だけ試す）、障害ですぐ返した画面の GET は 503 + Retry-After + no-store・不調の間は全応答 no-store・prerender に置かない、通信の失敗が 60 秒に 5 回でクライアントを作り直す、待ち 4 秒・再試行込み 4 秒、ミラーの timeout・retry、取り込みが失敗している間は `_gate` が待たない | 実装済み |
| 3 | 量の上限: load キャッシュ `LOAD_CACHE_MAX_MB=256`（JSON の長さ × 2.5 で数える）・60 秒ごとに期限切れを消す、レース詳細のキャッシュは期限切れを消し 200 件まで、bundle のキャッシュに期限切れの掃除と 50 件、prerender 100MB、FastAPI と Flask の HybridStorage を 1 つに、`MALLOC_ARENA_MAX=2` | 実装済み |
| 4 | `src/api/memory_guard.py`（下）、会員向けの出し分けを to_thread へ、VPS の cache-warm の間隔 | 実装済み |

### 自分で終わる仕組み（`src/api/memory_guard.py`）

- 外からの ping・liveness probe は足さない。入口（Gateway）が要求の始めに、30 秒に 1 回（同時の要求のうち 1 つだけ）スレッドで点検する。
- メモリの割合は cgroup v2 → cgroup v1 → `/proc/meminfo`（`MemTotal - MemAvailable`。/tmp は Shmem として使用中に入る）の順で読む。
  2026-11-17 の stg（gen2）では cgroup v2 のファイルが無かった。
- 85% 以上: キャッシュ（load・一覧・bundle・レース詳細など。設定の取り直しは含めない）を捨て、prerender を半分に、GC と malloc_trim。5 分に 1 回まで。
- 捨てた後も 92% 以上（起動から 5 分以上）: SIGTERM（uvicorn は `--timeout-graceful-shutdown 15` で処理中の要求を待つ）→ 20 秒後に `os._exit`。Cloud Run が入れ替える。
- GCS: クライアントを作り直してから 600 秒以上成功せず、いまも不調で、かつメモリが 70% 以上のときだけ終わる。GCS だけの障害（地域の障害など）では終わらない
  （新しい 1 台は起動時の取り込みもできず、手元の写しで答えている今の台より悪くなるため）。
- `KEIBA_MEMORY_GUARD=0` で止められる。しきい値は `KEIBA_MEMORY_GUARD_SOFT`・`_HARD` などで変えられる。

### 設計上の限界（承知のうえで残すもの）

- 要求の中の並列処理（`ThreadPoolExecutor.submit`）は文脈を引き継がないため、その中で障害ですぐ返しても要求の印が立たず、503 にならない。
  ただし、その時点で GCS が不調（共有の状態がバックオフ中）なら no-store は付くので、nginx に長く保存されることは無い。
- 本文を送り始めた後（ストリーミング）の障害は 503 にできない。
- load キャッシュの追い出しは挿入順（読まれても順は変わらない）。従来の件数の上限と同じ動き。

## 5. 測り方と結果（stg）

- 2026-11-15 の 07〜08 時の cache-warm の要求 2,899 件から、等間隔に 300 件を抜き、stg（タグ `rstg`）へ 1 秒に 2 件で流す。
  共有ヘッダ付き・Cookie なし（cache-warm と同じ匿名の扱い）・`X-Keiba-Fresh: 1`。25 件ごとに `/ops/memory` を記録する。
- stg への要求は全体で 1,000 件以内。同じ東京リージョンの中なので GCS の転送料はかからない。

### 直す前（段 1 だけの stg、2GiB、2026-11-17 15:50 JST ごろ、同じ 1 台）

| 時点 | RSS | load キャッシュ（件数・概算） | prerender | ミラーの木 |
|---|---|---|---|---|
| 流す前（起動 318 秒） | 452MB | 151 件・5MB | 0.3MB | 412MB |
| 100 件 | 575MB | 1,185 件・43MB | 16MB | 412MB |
| 175 件 | 644MB | 2,000 件（件数の上限）・73MB | 20MB | 412MB |
| 300 件 | 739MB | 2,000 件・約 500MB（大きい smartrc_race などが入ると増える） | 37MB | 412MB |

- 朝の温め（2,899 件）の 1 割で RSS が +288MB。load キャッシュは件数の上限に早く達し、その後は 1 件の大きさで量が決まる（smartrc_race は 1 件約 0.7MB）。
  件数だけの上限では量が抑えられない。
- この版は cgroup の割合を読めなかった（段 2〜4 で `/proc/meminfo` に頼るよう直した）ので、RSS で比べる。
- **ミラーの木 412MB は起動の時点からメモリを占める**。うち `data/local/race_analysis` が 3,062 件・286MB（ほかは `page_reference/tables` 76MB など）。
  Cloud Run の書き込み層はメモリに数えるので、2GiB のうち約 2 割。日付の窓（`WINDOWED_DIRS`。古い日は読まれたときに 1 件だけ取る）に入れれば減らせる。
  ただし、束（`base.tar.gz`）・目録の作りと、古い日の読み出しの費用（1 件ずつの Class B）に関わるため、今回は直さず、別の作業として提案する。

（直した後の値は段 2〜4 の stg の配備の後に追記する）
