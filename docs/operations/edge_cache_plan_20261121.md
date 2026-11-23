# めぐ AI の予測を VPS のキャッシュから返す計画（2026-11-21、edge-cache）

ユーザーの依頼: 本番（https://app.megu-keiba.com/ 、ConoHa の VPS）で、AI の予測の表示が当日も過去の分も遅い。すべてのページをコンマ秒以下で返したい。基本は VPS のキャッシュのフォルダから返す。深夜に直近 1 か月の開催分を入れる。GCP の費用は最小にする。

この段は**調査と計画だけ**です。コード・本番・VPS の設定は何も変えていません。実装は 11/23 の開催の後、統括役の承認を得てから行います。

統括役が承認した方向（この計画の骨組み）:

- ① 作る: ホスト PC が、ページに要る API の応答を区分（匿名・会員・開発者）ごとに JSON に書き出す。
- ② 送る: ホスト → VPS に rsync（ssh）で、変わったファイルだけを送る。GCS を通さない。
- ③ 配る: VPS の nginx がフォルダから返す。区分の判定は VPS の上で署名付きの Cookie を確かめるだけにする。
- ④ 予備: フォルダに無いときだけ、今の Cloud Run の経路へ流す。

## 0. 要点

- **遅いのは会員の要求です。** 匿名の要求は、今も VPS の nginx のキャッシュから返っています（VPS の中で 0.5ms、公開 URL から 130〜150ms）。
  - 11/21（土・開催日）にブラウザから Cloud Run に届いた要求は 107 件でした（iPhone 1 台。会員と見られる）。
  - その予測（`/api/race/{id}/predictions`）は p50 2.0 秒・p95 22.9 秒（n=66）でした。race-values は 15〜29 秒、race-list は 1〜31 秒です。原因はコールドスタート（起動 27.6 秒）と、キャッシュに無い計算です。
  - nginx は Cookie の付いた要求を読まないので、会員の要求は毎回 Cloud Run に届きます。
- **Cloud Run の要求の 94% は VPS の cache-warm です。** 11/21 の 8,796 件のうち 8,312 件でした。
  - そのうち 4,098 件（10 種類）は、会員だけのパス（result・analysis・megu-index-predicted・tracking-difficulty・race-quality/day・bets-summary・bets-by-level・馬の 3 種・megu-index-dates）です。cache-warm は匿名として呼ぶので 401 が返り、保存もされません。無駄な要求です。
- **承認された方向のまま書くと、規約違反と本番との食い違いが 2 つ入ります。この計画では次のとおり直しました（§3.2・§3.3）。**
  1. ホストの作業ツリーは develop で、`.env` は stg 用です（`KEIBA_ENV=stg`・`KEIBA_PUBLIC_MODE=public`・`KEIBA_PROB_CALIBRATION=1`）。本番は master の版と、Cloud Run の環境変数で動いています。そのままホストで作ると、本番と違う中身が出ます。→ 本番に配備中の版を別のツリーに取り出し、環境変数も本番の値に固定して作ります。
  2. CLAUDE.md は「ホストPCの処理に GCS を読ませて速くしない」と定めています。ところが、確定結果・払戻・オッズは GCS にしかありません（ホストの `data/cache` には一部だけ）。→ 作る処理は GCS に触れたら止める守りを入れ、手元のファイルだけで作ります。手元に足りない入力は、ホストが保存するときに手元にも書きます（取得はすべてホストで動いています）。入力が足りないレースは作らず、Cloud Run に任せます。
- **区分の判定は、今のセッションの Cookie では VPS の上でできません。**
  - 今の Cookie の検証は、署名だけでなく名簿（`member_active`・`pw_changed_ts`）も見ています（`src/api/auth.py:83-104`）。
  - VPS に名簿を置かないために、Cloud Run が別に「区分の Cookie」（env・区分・発行時刻・期限・HMAC）を出します。期限は 2 時間にし、迷ったら Cloud Run に流します（§3.5）。
  - 課金を止めた人に、最大 2 時間フォルダの中身が出続けます。ただし会員は共有アカウントで、課金を止めた人も月替わりの切り替えまでは今も入れるので、実質の後退は無いと統括役が判断しました（§9）。
- **VPS の nginx は 1.18 で、njs も `secure_link` もありません。** `auth_request` と `gzip_static` はあります。区分の判定は、`auth_request` で 127.0.0.1 の小さな確認役に聞く形にします。
  - 確認役が落ちても 500 にせず Cloud Run に流す設定（`error_page`）と、deploy ユーザーが置ける全停止の印（`_ctl/OFF`）を必ず入れます（§3.6）。
- **最初に試すのは、確定済みのレース × 会員の区分です。** まだ本番に無い weekend-launch の P1（`race_settled`）に頼らずに済み、効果がいちばん大きい所です。11/21 のブラウザの予測の要求 66 件のうち 61 件は、確定済みの 11/14・15 のレースでした。匿名の分は、P1 が入ってから同じ仕組みで作ります。開発者の区分を VPS に置くかは、承認の項目です（§7）。
- 費用（最悪の回数）:
  - GCP は増えません。作る処理は GCS を読みません。Cloud Run への要求は減ります。
  - 減る方向として、cache-warm のうち新しい仕組みで足りるパスをやめられます。最大で月 約 100 円です（§4）。
  - 直近 1 か月分を最初に埋めるときだけ、GCS を 1 回読む案があります（約 3 円。承認が要る。§4）。
- 目標の「主なページの API の p95 を 200ms 以内」は、フォルダに当たる要求では届く見込みです（公開 URL・新しい接続で 130〜150ms、使い回しで 約 54ms）。フォルダに無い要求（1 か月より前・入力の欠け・確定前の匿名など）は、今の Cloud Run の速さのままです。
  - **auth/status（人ごと）は段 1〜4 では Cloud Run のままで、目標の p95 の対象から外して別に測ります。** 冷えた Cloud Run では auth/status が 14〜28 秒かかります。画面の描画がこれを待つかは段 3 で確かめ、待つなら段 5 以降で確認役が答える形を検討します（§3.5）。
  - 段 1〜4 の対象のパス（predictions・race・result・analysis）は、11/21 のブラウザの要求ではクエリ無しで呼ばれています（クエリ付きは race-values・scrape-dates・news だけ）。§3.6 の「クエリ付きは Cloud Run へ」で当たりは減りません。

## 1. 今の計測（2026-11-22 01:00 JST ごろ）

### 1.1 公開 URL を匿名で（ホスト PC の curl、新しい接続、各 2 回）

対象のレース: 当日 `202608050611`（11/22 京都 11R）、確定済み `202608050411`（11/15 京都 11R）。

| 対象 | 1 回目 TTFB（nginx） | 2 回目 TTFB（nginx） | 大きさ |
|---|---|---|---|
| `/`・`/races`・`/race/{id}`・`/weekly-predictions`・`/analysis`（HTML、Next の事前生成） | 137〜249ms | 140〜200ms | 3〜5KB |
| `/api/scrape-dates?picker_past_days=30&picker_future_days=0` | 229ms（MISS） | 134ms（HIT） | 131B |
| `/api/race-list/20261122`・`/20261115` | 129〜138ms（HIT） | 134〜144ms（HIT） | 0.9〜1.4KB |
| `/api/race-calendar?month=202611` | 277ms（MISS） | 147ms（HIT） | 369B |
| `/api/race/{当日}` | 657ms（MISS） | 126ms（HIT） | 19KB |
| `/api/race/{当日}/predictions` | **1,777ms**（MISS） | 153ms（HIT） | 1.9KB |
| `/api/race/{確定済み}` | 852ms（MISS） | 136ms（HIT） | 27KB |
| `/api/race/{確定済み}/predictions` | 304ms（MISS） | 134ms（HIT） | 2.4KB |
| `/api/prediction-results/news?limit=6` | 412ms（EXPIRED） | 153ms（HIT） | 247B |
| `/api/v1/auth/status` | 254ms（MISS） | 232ms（MISS） | 100B |
| 会員だけ（race-values・analysis・result） | 207〜247ms（401） | 204〜217ms（401） | 130B |

- Cloudflare はどれも `DYNAMIC`（API・HTML は置かれていない。PR #3 の L6 はまだ develop に入っていない）。
- 2 回目の HIT は 130〜150ms です。そのうち 約 100ms は新しい接続（DNS・TCP・TLS）の分です。使い回しの接続では 約 54ms（latency_plan §2）。
- 会員の要求は測れていません（このマシンに会員のログインを持たないため。ログインは試していません）。会員の速さは、下の Cloud Run のログで見ています。

### 1.2 VPS の中から（deploy ユーザーの ssh、各 2 回）

| 対象 | 値 |
|---|---|
| nginx の HIT（`127.0.0.1:8080`、race-list・predictions） | 0.5ms |
| nginx の MISS → Cloud Run（race-list / 当日の predictions） | 124ms / 2,169ms |
| VPS → Cloud Run `/api/health` 新しい接続（latency_plan §7 の 2） | con 8〜63ms・tls 71〜130ms・ttfb 99〜162ms |
| 同 使い回し | ttfb 27ms |

- **L3（nginx → Cloud Run の接続の使い回し）は、見送りの基準（20ms 未満）を大きく超えます。** 新しい接続のたびに 70〜130ms かかっています。ただし nginx 1.18 は upstream の `resolve` が使えません。名前の解決を起動時に固定し、IP が変わったら reload で拾う案になります（latency-plan で扱う）。

### 1.3 Cloud Run のログ（サービスアカウントで Cloud Logging を読んだ。読み出しは無料）

11/21 JST の 1 日の要求（8,796 件）の内訳:

| 送り手 | 件数 | 備考 |
|---|---|---|
| keiba-cache-warm（VPS） | 8,312 | 予測 1,659・レース 1,659・race-list 257 など（200）。会員だけのパス 10 種 計 4,098（401） |
| Next.js Middleware（保守の確認） | 372 | |
| ブラウザ（iPhone） | 107 | |
| curl（今回の計測など） | 5 | |

ブラウザの要求の速さ（11/21）:

| パス | 件数 | p50 | p95 | 最大 |
|---|---|---|---|---|
| `/api/race/{id}/predictions` | 66 | 2.0 秒 | 22.9 秒 | 28.1 秒 |
| `/api/race/{id}/result` | 8 | 0.1 秒 | 13.8 秒 | 18.4 秒 |
| `/api/v1/betting/race-values` | 5 | 15.4 秒 | 28.9 秒 | 28.9 秒 |
| `/api/race-list/{日}` | 6 | 1.1〜4.1 秒 | — | 31.2 秒 |

- 起動（「起動の段」のログ）: 11/17 6 回・11/18 28 回・11/19 18 回・11/20 12 回・11/21 10 回。1 回の起動は 27.6 秒の例があります。
- 11/19〜21 の 0.5 秒を超えた要求 3,000 件の 80% は predictions（p50 1.3 秒・p95 2.0 秒）。キャッシュに無い predictions の計算は 1〜2 秒かかります。

### 1.4 latency_plan §7 の残り

| 項目 | 結果 |
|---|---|
| 1. nginx の版・上流 | nginx 1.18.0（Ubuntu）。モジュール: `auth_request`・`gzip_static`・`geoip2` あり、njs・`secure_link` なし。上流は Cloud Run（`keiba-api-…run.app`） |
| 2. VPS → Cloud Run | §1.2。新しい接続 99〜162ms、使い回し 27ms |
| 3. 実際の利用者のキャッシュの当たり | **未計測。** access.log は adm の持ち物で、deploy ユーザーは読めない。Cloud Run のログから、ブラウザの要求は 1 日 約 100 件と分かった（§1.3） |
| 4. cache-warm の 429 | **未確認。** deploy ユーザーは journal を読めない（`-- No entries --` は権限のため。0 件の根拠にはならない）。ユーザーに `sudo journalctl -u keiba-cache-warm --since today \| grep -c ' 429'` をお願いする |
| 5. Cloud Run のログ | §1.3 で取得 |
| 6. ブラウザの計測（LCP・TLS） | **未計測**（ユーザーの操作） |

- ほかに: VPS の nginx の API キャッシュの置き場を `du` で見ると 68KB でしたが、deploy ユーザーは中のディレクトリを読めないためと見られます（計測では HIT が出ている）。件数は不明です。

## 2. API の棚卸し

画面は Next.js の事前生成の殻で、データはすべてブラウザが `/api` を直接呼びます。有効な画面はホーム・`/races`・`/race/[id]`・`/weekly-predictions`・`/analysis` です（`frontend/lib/features.ts`）。

### 2.1 ページと API

| 画面 | 呼ぶ API（順） | 会員の区分の違い |
|---|---|---|
| 全画面の共通 | `/api/v1/auth/status`（20 秒）、`/api/maintenance/status`（60 秒おき） | auth/status は人ごと |
| ホーム | `prediction-results/news?limit=6`・`prediction-results/bets-by-level`（並列） | bets-by-level は会員だけ |
| `/races`・過去日 | `scrape-dates?picker_past_days=30&picker_future_days=0` → `race-list/{日}`（直列。PR #3 の L11 で並列にする）、週の各日の race-list、`race-calendar?month=`、`v1/betting/race-values?date=`、`prediction-results/bets-summary?date=` | race-values・bets-summary は会員だけ |
| レース詳細 | `race/{id}` と `race/{id}/predictions`（並列）→ RaceNav の `race-list/{日}`・race-values | predictions はサーバが区分で項目を落とす |
| 推奨馬券のタブ | `POST /api/v1/betting/optimize`（本文に race_id・bankroll・max_total_bet・odds_mode・odds_phase・bet_types） | 会員だけ。dev は full、会員は paid の別の中身 |
| 週の予想 | URL から race-list・race-values・predictions を先読み → scrape-dates（`picker_future_days=14`）→ race-list → 選んだレースの predictions、続けて他のレースの predictions を最大 40 本、前後 7 日の race-values を最大 8 日分。押したときに `race/{id}/result`・`tracking-difficulty`・`entry/{horse_id}/card` | result・tracking は会員だけ |
| めぐ分析 | scrape-dates → race-list → `v1/races/{id}/analysis` | analysis は会員だけ（401 で判定） |

定期の問い合わせ: 未完了の間の predictions（60 秒）、発走後 4 時間の result（60 秒）、保守の確認（60 秒。latency_plan の L2b で今のままと決定済み）。

### 2.2 API ごとの性質

| API | 区分で変わる | 時刻・確定で変わる | 人ごと | フォルダの対象 |
|---|---|---|---|---|
| `/api/v1/auth/status`・login・logout・billing | — | — | **人ごと** | 対象外（auth/status だけ §3.5 の確認役が答える） |
| `/api/maintenance/status` | 変わらない | 保守の時刻 | — | 対象外 |
| `/api/scrape-dates?…`・`/api/race-calendar?month=` | 変わらない | 開催日の追加 | — | 対象（クエリを正規化） |
| `/api/race-list/{日}` | 変わらない | 出馬表・結果の取り込み | — | 対象 |
| `/api/race/{id}` | 匿名は項目を落とす（`app.py:6982`） | 出馬表の段階・馬体重・取消 | — | 対象 |
| `/api/race/{id}/predictions` | 匿名・会員・dev で別（`app.py:7154`） | 予測の再計算・T-60/T-15 のオッズ・確定（P1 の後） | — | 対象（本命） |
| `/api/race/{id}/result`・`tracking-difficulty`・`v1/races/{id}/analysis` | 会員だけ（匿名は 401） | 結果の確定・前日の計算 | — | 対象（会員・dev） |
| `/api/v1/betting/race-values?date=` | 会員（paid）・dev（full）で別 | オッズで当日は数分おき（`_value_ttl` 20 秒） | — | 確定済みの日だけ対象。当日は Cloud Run |
| `POST /api/v1/betting/optimize` | 会員（paid）・dev（full） | オッズ・確定 | 本文の bankroll・max_total_bet が人ごとになりうる | 既定の条件だけ GET の口を作って対象（§3.4） |
| `/api/prediction-results/news`・`bets-summary`・`bets-by-level` | 3 通り（`prediction_results_api.py:37-44`） | 開催日の確定ごと | — | 対象 |
| `/api/race/{id}/entry/{horse_id}/card` | 変わらない | 出馬表 | — | 後回し（押したときだけ・数が多い） |

人ごとに変わる応答は auth/status と billing だけです（お気に入りなどは無い。`stallion-notes/custom` はサーバ全体で 1 つ）。

### 2.3 応答の中で、要求の時刻から作る値

- predictions の `freshness.age_hours` は要求のたびに今の時刻から計算します（`app.py:7115-7140`）。ファイルにすると、作った時刻の値で止まります。→ 作った時刻（`generated_at`）を入れ、画面の側で差を出す形に直します（実装の段でフロントを確かめる）。
- `entries_status`・`pipeline_status` は作った時点の値になります。出来事ごとに作り直すので、追いつく遅れは §3.7 の上限の内です。

### 2.4 今ある仕組みと、使い回せる範囲

| 仕組み | 今 | 使い回し |
|---|---|---|
| prerender（Cloud Run の `/tmp`） | 落とす前の全項目の版だけを置く。区分は見ない | 使わない（Cloud Run の中の仕組みのまま残す） |
| prewarm（`KEIBA_PREWARM`） | Cloud Run の中で今週分を先に作る。GCP の外では `egress_billed()` で止まる | 使わない |
| `config/http_cache.json`・`http_cache_headers.py` | 匿名で許可リストのパスだけ public。Vary: Cookie | 匿名の分は今のまま。フォルダの対象のパスの一覧は、この許可リストを土台にする |
| nginx の API キャッシュ・cache-warm | 匿名だけ。cache-warm は GCS の合図を VPS から読む | フォルダの後ろの予備として残す。フォルダで足りるパスは cache-warm の対象から外す（§4） |
| `cache_marker`（保存 → 目印） | GCS に目印を置く | 使わない（ホストの手元の合図を使う。§3.3） |
| gcs_mirror の push（5 分おき、手元の manifest で差分） | `sent` の一覧が分かる | 作り直す対象を決める合図の 1 つにする |
| `inference/events/<日>.jsonl` の ok 行 | 本番の予測（v6e）がレースごとに書く | 予測の出来事の合図にする |
| `httpx.ASGITransport` で自分の app を呼ぶ前例 | `cloudrun_entry._warmup` | 作る処理で、同じ形で app を in-process で呼ぶ |

## 3. 設計

### 3.1 全体の流れ

```
ホスト PC（GCS を読まない）
  合図（events の ok 行・entries_state・mirror の sent・結果の保存・深夜の一括）
    → edge-render（本番の版のツリー・本番の環境変数・手元の入力だけ）
    → /home/…/edge_out/<env>/<版>/<区分>/api/...json（内容が変わったときだけ書き換える）
    → rsync（ssh・差分）
VPS
  /srv/keiba-edge/<env>/<版>/<区分>/...json（と .json.gz）
  nginx: /api/ の許可リストのパス → 区分を判定 → ファイルがあれば返す（0.5ms）
                                              → 無ければ今の Cloud Run の経路（nginx の匿名キャッシュ・cache-warm を含む）
```

### 3.2 作る処理と、本番との一致

- 作る処理（仮称 `edge-render`。`src/ops/edge_render.py`）は、API の app を in-process で呼び（`httpx.ASGITransport`。`cloudrun_entry._warmup` と同じ形）、応答の本文をファイルにします。
  - 区分は Cookie を偽造せず、作る処理の中だけで区分を指定する口から渡します。`token_tier` だけでなく、`is_member`・`is_developer` の判定にも効く形にします（`token_tier` だけでは開発者の判定が変わらない）。この口は、作る処理のプロセスの外では効かない作りにします。
  - **要求の接続元をループバックにしない**。`ASGITransport` の接続元は既定で 127.0.0.1 で、KEIBA_ENV が dev に落ちると開発者の自動ログイン（`auth.py:175-205`）で匿名の作成が開発者の中身になります。接続元をループバック以外の値にし、`KEIBA_DEV_AUTO_LOGIN=0` を固定します。
  - free の区分は作りません（中身は匿名と同じ。§3.5 で anon のファイルを返す）。
- **本番の版で作る**:
  - CI の配備（`scripts/ci/deploy_gcp.sh`）が、配備した版の sha に git のタグ（例 `api-live-prod`・`api-live-stg`）を付けて push します。
  - ホストは GitHub からタグを取り、別のツリー（例 `/home/jovyan/work/keiba-edge/<env>/src`）を、その sha に切り替えます。共有の作業ツリー（develop・ホストの cron が動く）は使いません。
  - ファイルの置き場は版ごと（`<env>/<sha の先頭 12 文字>/`）にします。版が変わったら、新しい版の置き場を作り終えてから切り替えます（§3.6）。
- **本番の環境変数で作る**:
  - Cloud Run の本番・stg の環境変数のうち、応答に効くもの（`KEIBA_ENV`・`KEIBA_PUBLIC_MODE`・`KEIBA_PROB_CALIBRATION`・`KEIBA_LAP_FORECAST`・`KEIBA_TRACKING_V2` など）を、リポジトリの 1 つのファイル（例 `config/edge_render_env.json`）に env ごとに書きます。
  - 作る処理は、ホストの `.env` を読まず、このファイルの値だけで動きます。`app.py:37-38` は import の時点で `.env` を読む（上書きはしない）ので、別のツリーに `.env` を置かず、空の dotenv を指定してから import します。本番の `ENV_VARS` に無いキー（`KEIBA_PUBLIC_MODE` など）も、本番での値（未設定なら空）を明示して固定します。`GCS_BUCKET` は空にします（§3.3）。
  - テストで、配備スクリプトの `ENV_VARS` とこのファイルの食い違いを検出します。
  - 秘密（`DEV_SECRET_KEY` など）は作るのに要りません。
- **別のツリーの入力**: `data/local`・`data/page_reference`・`inference/tracking_v2`・`lap_forecast` などは、共有の作業ツリーの実物をリンクで見せます。リンクが無いと黙って空の応答になるので、作る前に必須の置き場の有無を確かめ、無ければ止めます。

### 3.3 GCS を読まない（CLAUDE.md の決まり）

- 作る処理は、`HybridStorage` を「手元だけ」の形で使います（`GCS_BUCKET` を空にし、GCS のクライアントを作ろうとしたら例外にする）。
- **例外が握りつぶされても気づける形にします。** 会員の想定オッズの読み込み（`tier.py:81-84`）など、GCS の読み出しの失敗を握りつぶして空で続ける箇所があります（`storage.py` にも `except Exception` が多い）。そのため、「GCS に触れようとした」「手元に入力が無かった」をプロセスの印で持ち、印が立った応答は丸ごと捨てます（ファイルを作らず Cloud Run に任せる。`prerender.py` の `degraded_since` と同じ考え）。確定済みのファイルは長く返り続けるので、欠けた中身を置かないことが大事です。
- 応答に要る入力（実装の最初に、1 レースを作って読んだカテゴリを数えて確定する）:
  - 手元にすでにあるもの: 予測・推奨の元（`data/local/inference/…`）、めぐ分析（`data/local/race_analysis`）、出馬表の段階（`entries_state`）、レース一覧（`data/page_reference/race_lists`）。
  - GCS にしか無いもの（見込み）: `race_shutuba`・`race_result`・`race_result_on_time`・払戻・`race_odds`・`race_pair_odds`・過去走。
  - 本番は `KEIBA_PREDICTION_RESULTS_GCS=1`（`deploy_cloud_run_api.sh:67`）で、予測成績（news・bets-summary・bets-by-level）を GCS から読みます。手元の入力で作れない見込みなので、段 6 までは対象から外します。
- **GCS にしか無いものは、ホストが保存するときに手元にも書きます。**
  - 取得はすべてホストで動いています（`config/fetch_owner.json` の agent_tasks は空）。保存の時点で中身は手元にあります。
  - `HybridStorage.save` に「この区分は手元にも書く」の設定を足し、置き場は 35 日分に限ります（例 `data/local/edge_src/<category>/…`。古いものは深夜に消す）。
  - 追加の GCS の操作は 0 回です。
  - **開催日の朝のホストの当日の処理は変えません。** この書き込みは、11/23 の開催の後の平日に入れ、翌週の開催で試します。
- 直近 1 か月分を最初に埋める方法は 2 つです（ユーザーの判断）:
  - 案 A（推奨）: 手元にも書く設定を入れた日から先だけを作る。1 か月より前は Cloud Run のまま。GCS の読み出しは 0。1 か月で埋まる。
  - 案 B: 1 回だけ GCS から直近 1 か月の入力を読む。約 330 レース × 約 0.5MB（見込み）≒ 0.16GiB × 18 円 ≒ **約 3 円**（1 回だけ）。読む量は実装の最初の 1 レースで測り直してから、承認を取る。

### 3.4 フォルダの形

```
/srv/keiba-edge/<env>/current -> <sha12>/
/srv/keiba-edge/<env>/<sha12>/
  _meta/heartbeat.json          ホストが送るたびに書く（作った時刻・版）
/srv/keiba-edge/<env>/_ctl/     版の置き場の外（rsync の --delete・版の切り替えで消えない）
  not_before                    区分の Cookie をこの時刻より前の発行なら使わない（§3.5）
  stale                         確定前のレースを使わない印（§3.7）
  OFF                           全部 Cloud Run に戻す印（§3.6）
  anon/api/race/<race_id>.json（と .json.gz。以下同じ）
  anon/api/race/<race_id>/predictions.json
  member/api/race/<race_id>/predictions.json
  member/api/race/<race_id>/result.json
  member/api/v1/races/<race_id>/analysis.json
  member/api/v1/betting/optimize-default/<race_id>.json
  dev/…（member と同じ形）
  anon/api/race-list/<yyyymmdd>.json
  anon/api/_q/scrape-dates/picker_past_days=30&picker_future_days=0.json
```

- 置き場の区分は anon・member・dev の 3 つです（free の人には anon を返す）。
- 区分で中身が変わらない API（race-list・scrape-dates・race-calendar）は anon にだけ置き、nginx が全区分に anon を返します。
- ファイルは素の `.json` と gzip の `.json.gz` を並べて置きます。nginx の `try_files` は素のファイルの有無で判定し、`gzip_static on` が `.gz` を返します（`try_files` は `.gz` だけの置き場を見つけないため、素のファイルも要る。合わせて 1 か月 約 30〜70MB）。
- **クエリ付きのパス**は許可リストにして、正規化の規則を決めます。
  - 対象は scrape-dates・race-calendar・race-values・news・bets-summary だけです。段 6 までは、クエリ付きの要求はすべて Cloud Run に流します（§3.6）。
  - キーの名前を並べ替え、値は `[0-9A-Za-z_-]` だけを許します。それ以外の文字や、知らないキーがあればフォルダを見ずに Cloud Run に流します。
  - nginx の `map` で `$args` を許可された形だけ受け付けます（正規表現で全体に一致したときだけファイル名にする）。
- **POST の optimize**: フロントに GET の口（例 `GET /api/v1/betting/optimize-default/<race_id>`。クエリにしない）を足します。既定の条件（bankroll・max_total_bet・odds_mode・odds_phase・bet_types が既定）のときだけ使い、それ以外は今の POST（Cloud Run）を使います。当日・確定前は対象にしません（オッズで変わるため）。
- 大きさの見込み: 1 レース × 3 区分 × 約 6 種類 × 平均 約 10KB（gzip 後 約 3KB）≒ 50KB。1 か月 約 330 レースで 約 20MB。日付の単位の分を足しても 50MB 未満です。

### 3.5 区分の判定

今のセッションの Cookie（`keiba_dev_session`、30 日、HMAC）の検証は、署名だけでなく、要求ごとに名簿の `member_active`（課金の状態 `billing_allows` を含む）と `pw_changed_ts` を見ています（`src/api/auth.py:83-104,390-392`）。VPS に名簿を置かないため、別の Cookie を足します。**この Cookie は、今のセッションより失効が遅れます**（下の「失効」）。その遅れの上限を短くするのが、この節の中心です。

- **区分の Cookie**（仮称 `keiba_edge`）:
  - 中身: `env.区分.発行時刻(iat).期限.HMAC-SHA256`。鍵は新しい秘密 `EDGE_TIER_KEY` で、**stg と prod で別の鍵**にし、HMAC の中身に env も入れます（stg の Cookie を本番で使えない。stg には `KEIBA_TEST_USER` がある）。人を特定する値は入れません。
  - 区分は `anon`・`free`・`member`・`dev` の 4 つです。free の人にもファイルは anon を返しますが、Cookie の区分は free のままにします（画面の「ログイン中」の表示を崩さない）。
  - 出す所: Cloud Run の login と auth/status（どちらもセッションの完全な検証を通った後）。`HttpOnly`・`Secure`・`SameSite=Lax`・`Path=/api/`。
  - **期限は 2 時間**（推奨）。期限が切れたら、次の要求は Cloud Run に流れ、auth/status で出し直します。
    - Cloud Run に戻るのは 1 人 2 時間に 1 回です。冷えていればその 1 回は遅くなりますが、起動の回数は最悪の見積もり（96 回/日）の内です。
  - ログアウトでは、セッションと区分の Cookie を両方消します。
- **失効**（今のセッションとの差）:
  - 月替わりのパスワードの切り替え: ホストの `shared_login`（毎日 00:05 JST、月が替わった日だけ動く）の終わりに、`not_before`（その時刻 ＋ Cloud Run に名簿のミラーが届くまでの余裕 15 分）を VPS に送ります。発行時刻が `not_before` より前の Cookie は使えず、Cloud Run に流れます（そこで今どおり 401）。ミラーが届く前に出し直された Cookie も落とすため、`not_before` は 15 分後にもう一度送ります。
  - 課金の停止・会員の無効化: Cloud Run 側（`v1/routes/billing.py`）で起き、ホストは拾えません。**区分の Cookie の期限（2 時間）が、停止した人にフォルダの中身が出続ける上限です。** 今のセッションは次の要求で止まるので、この 2 時間が後退分です（承認の項目。§7）。
  - 急ぐとき（漏れの疑いなど）: `not_before` を今の時刻にして送るか、`EDGE_TIER_KEY` を替えると、全員の区分の Cookie が使えなくなり、Cloud Run に戻ります（害は 1 回の遅さだけ）。
- **VPS の確認役**:
  - 127.0.0.1 の小さな HTTP の処理（Python の標準ライブラリだけ、約 100 行、systemd の unit）。HMAC・env・期限・`not_before` を確かめて、`X-Edge-Tier: anon|member|dev|none` を返します（free は anon、検証に落ちたら none）。
  - セッションの Cookie だけあって区分の Cookie が無い要求は none（Cloud Run へ）。会員が匿名の中身を見て戸惑うことを避けます。
  - **区分の Cookie があって検証に落ちたら none**（anon ではなく Cloud Run へ）。偽の Cookie で区分を探られることも避けます。
  - 答えは nginx が `proxy_cache` で Cookie の値ごとに 60 秒覚えます（`not_before` の効きが最大 60 秒遅れる）。
  - **確認役が落ちた・遅いとき**: nginx の `auth_request` は、確認役が答えない・タイムアウト・5xx のとき 500 を返し、Cloud Run には流しません。そのため §3.6 の `error_page 500 502 503 504 = @cloudrun` で Cloud Run に流し、確認役への接続・読み出しの期限を短く（100ms）します。
- **auth/status**: 最初の段（段 1〜4）は Cloud Run のままにします（画面は `user_name` も使う。`Nav.tsx:30,127`）。auth/status は画面の描画を止めない形か（実装の段で確かめる）、止めるなら、確認役が答える形を段 5 以降で別に検討します。
- **開発者の区分を VPS に置くか**: latency_plan の L7 は「開発者は素通り」としていました。開発者の中身（第 2 段・v6c の試行など、確定前は開発者だけ）を VPS のディスクに置くかは、承認の項目にします（§7）。推奨は、段 1〜4 は member だけで、dev は Cloud Run のまま。

### 3.6 nginx の返し方

```nginx
# 考え方（実装の段で nginx -t と stg で確かめる）。stg と prod はポート（8081・8080）で分かれた server ブロックごとに置く
location ~ ^/api/(race/\d{12}(/predictions|/result|/tracking-difficulty)?|race-list/\d{8}|v1/races/\d{12}/analysis|…)$ {
    # rewrite の段（access の段より前）で、全停止の印・クエリ付き・区分の Cookie の無い要求を auth_request の外に出す
    if (-f /srv/keiba-edge/prod/_ctl/OFF) { return 418; }          # deploy ユーザーが置ける。置けば全部 Cloud Run
    if ($args != "")                       { return 418; }          # 許可していないクエリ（with_result_status・refresh など）
    if ($cookie_keiba_edge = "")           { return 418; }          # 段 1〜5 は会員だけ。匿名は段 6 で足す
    error_page 418 = @cloudrun;
    auth_request /_edge_tier;                                       # 127.0.0.1 の確認役。Cookie の値ごとに 60 秒覚える
    auth_request_set $edge_tier $upstream_http_x_edge_tier;
    error_page 500 502 503 504 = @cloudrun;                         # 確認役が落ちても 500 にしない
    root /srv/keiba-edge/prod/current;
    gzip_static on;
    add_header Cache-Control "private, max-age=15" always;           # 区分つきは共有キャッシュに置かせない
    add_header X-Edge "hit" always;
    try_files /$edge_tier$uri.json @cloudrun;                         # none は存在しない名前なので Cloud Run
}
location @cloudrun { …今の /api/ の location と同じ（nginx の匿名キャッシュを含む）… }
location = /_edge_tier { internal; proxy_pass http://127.0.0.1:<確認役>; proxy_connect_timeout 100ms; proxy_read_timeout 100ms; proxy_cache …; }
```

- 対象は許可リストのパスだけです。それ以外の `/api/` は今の location のまま。
- パスは正規表現で数字の桁まで決め打ちにします。大文字・末尾の `/`・`..` などは一致せず、今の location（Cloud Run）に行きます。
- クエリ付きのパス（scrape-dates・race-calendar・race-values など）は、正規化の規則と `map` を足す段（段 6 以降）まで、すべて Cloud Run に流します。
- `X-Edge: hit` を付け、Cloud Run に流れたものと区別できるようにします（計測と確認用）。
- **全停止**: `_ctl/OFF` を置くだけで、reload なしに全部 Cloud Run に戻ります。deploy ユーザーが置けます（VPS では sudo できないので、戻すのに root の reload を要らない形にする）。
- **Cloudflare・ブラウザのキャッシュ**:
  - フォルダから返す応答は `Cache-Control: private, max-age=15` にします。Cloudflare には置かれず、ブラウザは今の `httpCache.ts` のとおり 15〜60 秒だけ覚えます。
  - 匿名の確定済みの応答を Cloudflare に置くのは、PR #3 の L6 の規則（Cookie と Authorization が無い `/api/` の許可リスト、Edge 300 秒）に任せます。フォルダから匿名に返すかは段 6 で決めます。
- **stg と prod**: 今の nginx は `server_name _` で、待ち受けのポート（prod 8080・stg 8081）で server ブロックを分けています（`nginx-keiba-ui-cloudrun.conf:7-8`）。フォルダも `/srv/keiba-edge/{prod,stg}/` に分け、各 server ブロックが自分の置き場だけを見ます。区分の Cookie の鍵と env も分けます（§3.5）。

### 3.7 作るタイミングと古さの上限

| いつ | 何を | 合図 |
|---|---|---|
| 深夜（03:00〜04:00 の保守の時間の後、04:10 JST） | 直近 35 日の開催分のうち、入力が変わったレースと日付の単位の分 | 入力のファイルの MD5 を手元の記録（`edge_render_manifest.json`）と比べる。入力には、レースのデータに加えて、応答に効く設定と目印（`config/public_fields.json`・`prob_calibration.json`・`bets_stage2/_dev_only.json`・`predictions_v6c/_disabled.json`・モデルの版など）のハッシュも入れる（コードの版のタグでは拾えない変更のため） |
| 出馬表の更新 | そのレースの `race/{id}`・predictions、その日の race-list | `entries_state/<rid>.json` の変化 |
| 予測の再計算（v6e・第 2 段） | そのレースの predictions・推奨 | `inference/events/<日>.jsonl` の ok 行 |
| T-60・T-15 のオッズ | 当日のレースは対象にしない（会員の predictions はオッズで変わる） | — |
| 速報・確定 | そのレースの result・predictions、その日の race-list・成績 | 結果を手元に書いたこと（§3.3） |
| 配備（API の版が変わる） | 直近 35 日を新しい版で作り直す（約 330 レース） | §3.2 のタグ |

- **当日・確定前のレースの会員の predictions は、最初は作りません。** T-60・T-15 のオッズや市場補正で中身が変わり、出来事を取りこぼすと古いオッズの値が出るためです。前日まで（オッズの前）のレースと、確定後のレースだけを作ります。当日の分は、試しで出来事の取りこぼしが無いと確かめてから広げます。
- **出来事を見る処理は、ランナーとは別のプロセスにします。** 開催日のランナー・entries-sync・raceday-weight-runner のコードは変えません。別の処理（仮称 `edge-render watch`。1 分おき、手元のファイルだけを見る）が合図を読みます。止めるときは `config/edge_render.json` の `enabled: false` 1 つで止まります。
- **古さの上限**:
  - 確定済みのレース: レースのデータは変わりませんが、設定・目印・P1 の導入・モデルの差し替えで中身が変わります。設定と目印は上の manifest の入力で、API の版は §3.2 の版の切り替えで扱います。念のため、確定済みのファイルも 7 日ごとに作り直します（深夜。中身が同じなら送らない）。
  - 確定前のレース（前日までの出馬表・予測）: 合図からファイルが VPS に届くまで、見込み 1〜2 分（1 分おきの watch ＋作る時間 ＋rsync）。
  - VPS の見張り（systemd の timer、5 分おき、VPS の中のファイルだけを見る）: `heartbeat.json` が 15 分より古いと、確定前のレースのファイルを使わない印（`_ctl/stale`）を置きます。nginx は印があると確定前のパスを Cloud Run に流します（`if (-f …)` で判定）。確定前と確定後は、置き場を `pre/`・`settled/` に分けて判定できるようにします（実装の段で形を決める）。
  - 見張りの timer は deploy ユーザーの user unit で動かせるか、root の unit が要るかを実装の段で確かめます（root が要ればユーザーの操作）。

### 3.8 rsync の差分の決め方（GCS の一覧で比べない）

- 作る処理は、出来上がった本文の MD5 を手元の記録と比べ、**変わったときだけ**ファイルを書き換えます（一時ファイル → `os.replace`）。変わらないファイルの更新時刻はそのままです。
- rsync は既定の比べ方（大きさと更新時刻）で、手元と VPS のファイルだけを比べます。GCS には触れません。
  - `rsync -a --delay-updates --delete-after -e "ssh keiba-vps" edge_out/prod/<sha12>/ deploy@vps:/srv/keiba-edge/prod/<sha12>/`
  - `--delay-updates` で、1 回の送りの分がそろってから置き換わります。
  - 35 日を過ぎたレースは、手元で消し、`--delete-after` で VPS からも消えます。
- 版の切り替え: 新しい `<sha12>/` を送り終えてから、`current` の向きを付け替えます（`ln -sfn` → `mv -T` で一度に替える）。古い版は 1 つだけ残します（戻し用）。
- 送るのは deploy ユーザーです（VPS では sudo できない）。置き場 `/srv/keiba-edge` は deploy が書け、www-data が読める持ち主・権限にします（作るのはユーザーの操作。§7）。

### 3.9 ホストが止まったとき

- 確定済みのレースのファイルは、そのまま返り続けます（正しい中身）。
- 確定前のレースは、15 分で見張りが止め、Cloud Run に流れます（今と同じ速さに戻るだけ）。
- 区分の Cookie は Cloud Run が出すので、ホストが止まっても判定は続きます。月替わりの `not_before` が送られないと、新しいパスワードの後も古い区分の Cookie が最大 2 時間（期限）使えます。→ 切り替えの日に送れなかったら、統括役に知らせる（`shared_login` の失敗の通知に足す）。

### 3.10 weekend-launch の P1・latency の PR #3 との関係

- **P1（確定の前後での出し分け）**:
  - 作る処理は API のコードをそのまま呼ぶので、P1 が本番に入れば、匿名のファイルは「作った時点の確定の判定」で落とされた中身になります。重ねて作らず、同じ redactor を使います。
  - 確定の出来事で匿名のファイルを作り直すまで、匿名には伏せた版が残ります（漏れの向きではない）。
  - **P1 が本番に入るまで、匿名のファイルは作りません**（試しは会員だけ）。匿名は今の nginx のキャッシュのままです。
  - 実オッズ由来の値（期待値・オッズ・人気・第 2 段の q・r・T-60 オッズ）は、匿名のファイルに入りません（P1 の redactor のとおり）。テストで、匿名の置き場の全ファイルにこれらの項目が無いことを確かめます（`tests/api/test_shared_cache_tiers.py` と同じ項目の一覧を使う）。
  - 確定後の推奨の成績は、P1 のとおり匿名にも出します（prediction-results の匿名の版）。
- **PR #3（L11・L1・L6）**:
  - L11（並べて取る）・L1（前回の内容を先に出す）はフロントの変更で、フォルダと独立に効きます。先に入れてよい。
  - L6（Cloudflare に匿名の応答を 300 秒）: フォルダの応答は `private` にするので、Cloudflare には匿名の Cloud Run の応答だけが置かれます（今の計画のまま）。
  - L3（接続の使い回し）: §1.2 の結果で効果が大きい（70〜130ms）。フォルダに無い要求の速さに効くので、この計画と並べて進めてよい（nginx 1.18 なので `resolve` を使わない形）。
- **cache-warm**: フォルダで返す区分のパスは、cache-warm の取り直しが要らなくなります。まず、会員だけのパス 10 種（401 しか返らない。11/21 は 4,098 件）を cache-warm の対象から外します（独立した小さな直し。すぐやってよい）。

## 4. 費用（最悪の回数）

単価は CLAUDE.md と `cost_ledger.md`: GCS の外への転送 18 円/GiB、Class A 0.8 円/千回、Class B 0.06 円/千回、Cloud Run CPU 4.1 円/千 vCPU 秒、Cloud Run の外への送信 20 円/GiB。

| 項目 | 最悪の回数 | 月額 |
|---|---|---|
| ホストの作る処理の GCS 読み出し | 0 回（GCS に触れたら止める） | 0 円 |
| 手元にも書く設定（保存の時点で手元に書く） | GCS の操作は増えない | 0 円 |
| 直近 1 か月の初回の埋め（案 B を選んだときだけ・1 回） | 約 330 レース × 約 0.5MB ≒ 0.16GiB | **約 3 円**（1 回）。実測してから承認 |
| 区分の Cookie を出す | login・auth/status の応答に Cookie を足すだけ。要求は増えない | 0 円 |
| Cloud Run の要求（利用者） | フォルダに当たる分だけ減る。フォルダに無いときは今と同じ | 0 円（減る方向） |
| コールドスタート | 減る方向（会員の要求がフォルダで返る）。最悪でも今と同じ 96 回/日 × 30 の見積もりの内 | 0 円（増えない） |
| cache-warm の会員だけのパス 10 種をやめる | 4,098 件/日 × 30 ≒ 12 万件/月。1 件 最大 0.1 vCPU 秒とすると 1.2 万 vCPU 秒 | **最大 約 50 円 減る** |
| cache-warm のうちフォルダで足りるパスをやめる（段 6 の後） | 残り 約 4,200 件/日 × 30 × 0.1 vCPU 秒 ≒ 1.3 万 vCPU 秒 | **最大 約 50 円 減る** |
| 区分の Cookie の期限切れで Cloud Run に戻る | 会員 1 人 2 時間に 1 回 × 12 回/日。会員が 10 人でも 120 回/日の auth/status（温まっていれば 0.01 秒） | 数円未満。起動の回数は最悪の 96 回/日の内 |
| VPS が GCS を読む | 0 回（cache-warm の合図の読み出しは、cache-warm を減らす分だけ減る） | 0 円 |
| VPS のディスク | 約 30〜70MB × 版 2 つ | 0 円（空き 71GB） |
| VPS の受信 | 1 日 数 MB（変わったファイルだけ） | 0 円（ConoHa は受信の課金なし） |
| VPS の確認役 | メモリ 約 20MB・CPU はほぼ 0 | 0 円 |
| ホストの負荷 | 深夜: 最悪（版の切り替え）で 330 レース × 約 3 秒 ≒ 17 分（nice 19）。開催日: 出来事 約 200 回 × 約 3 秒 | 0 円（電気代は別） |
| Cloudflare | 規則を足さない | 0 円 |

- 合計: **GCP は 0 円の増加、最大 約 100 円/月の減少**。初回の埋めに案 B を選ぶと、1 回だけ 約 3 円。
- Cloud Run の 1 要求の CPU 秒（0.1 vCPU 秒）は上限の見積もりです。課金は要求の時間の和で、重なれば少なくなります。実装の 1〜2 日後に課金エクスポート（`horse-racing-466415.billing_export`）で確かめます。

## 5. 段取り

| 段 | 中身 | 合格の条件 |
|---|---|---|
| 0（すぐ・小） | cache-warm の対象から、会員だけのパス 10 種（401 しか返らない）を外す | 開催日で、cache-warm の要求が 約 4,000 件/日 減る |
| 1（11/24〜） | 作る処理（手元だけ・版の固定・環境変数の固定・握りつぶしの印）を書く。**確定済みのレース 数十 × 全区分**を作り、本番の Cloud Run の応答と項目ごとに比べる | GCS の読み出し 0 回（印が立った応答は捨てた件数も数える）。応答が `generated_at` などの時刻の値を除いて本番と一致。入力が足りず作れなかった件数と理由が分かる |
| 2 | 区分の Cookie（Cloud Run）と VPS の確認役・`_ctl/OFF`・`error_page`。stg だけで出す | stg で、匿名・free・会員・開発者・期限切れ・`not_before` より前・偽の署名・prod の鍵の Cookie の 8 通りで、返るファイル（または Cloud Run）が表のとおり。**確認役を止めても 500 にならず Cloud Run に流れる。`_ctl/OFF` で reload なしに全部 Cloud Run に戻る** |
| 3 | stg で、確定済みのレースの predictions・race・result・analysis を会員の区分でフォルダから返す | 主な画面（レース一覧・レース詳細・予測・週の予想・めぐ分析）でブラウザが出す API の要求の**すべて**（当たり・Cloud Run 行きを含む）の p95 と、当たり（`X-Edge: hit`）の率を出す。auth/status が画面の描画を待たせるかを確かめる |
| 4 | 本番に出す（master への昇格はユーザーの指示）。同じ範囲 | 本番で、確定済みのレースの画面の API の要求のすべて（auth/status を除く）の p95 が 200ms 以内。当たりの率と、auth/status の p95 も別に出す。会員の内容が anon の置き場に無い（テストと抜き取り） |
| 5 | 手元にも書く設定を入れ、出馬表・前日の予測の分（確定前）を広げる。開催日の朝は変えず、平日に入れる。**`HybridStorage.save` は開催日の取り込みと同じコードなので、手元への書き込みは失敗しても保存そのものを落とさない作り（例外を外に出さない・容量の上限・止める設定 1 つ）にする** | 出来事からファイルが届くまで 2 分以内。見張りが 15 分で止まる。手元の書き込みを壊しても、保存と取り込みが止まらないことをテストで確かめる |
| 6 | P1 の後、匿名の区分を作る | 区分 × 確定の前後の表（P1 のテスト）がフォルダでも成り立つ |
| 7 | 当日・確定前の会員の predictions、optimize の GET の口 | 出来事の取りこぼしが無いことを 2 週の開催で確かめてから |

- 開催日（土日）の朝にはホストの当日の処理を変えません。段 1〜5 の入れ替えは平日に行います。
- 段ごとに critic と CI を通し、統括役に断ってから develop → stg に入れます。

## 6. リスクと戻し方

| リスク | 防ぎ方 | 戻し方 |
|---|---|---|
| 会員の値が anon の置き場に入る（漏れ） | 作る処理は区分ごとに別の app 呼び出しで作る。anon の置き場の全ファイルに会員の項目・実オッズの項目が無いこと、**member の置き場に開発者だけの項目（第 2 段・v6c の試行など）が無いこと**を、送る前に毎回確かめ、1 件でもあれば送らない。テストでも確かめる | `_ctl/OFF` を置く（reload なしに全部 Cloud Run）。漏れたファイルを消し、Cloudflare・nginx の全消去 |
| 区分の取り違え（判定の誤り・開発者の自動ログイン） | 検証に落ちたら none（Cloud Run）。free は anon。作る処理は接続元をループバックにせず、`KEIBA_DEV_AUTO_LOGIN=0` | `_ctl/OFF` |
| 確認役が落ちて対象の API が 500 になる | `error_page 500 502 503 504 = @cloudrun`・確認役への期限 100ms | 確認役を止めても Cloud Run に流れる（段 2 で確かめる） |
| 失効した会員に出続ける | 区分の Cookie の期限 2 時間・`not_before`（版の置き場の外） | `not_before` を今の時刻にする・`EDGE_TIER_KEY` を替える |
| 欠けた中身が長く返る（GCS の守りが握りつぶされる） | プロセスの印が立った応答は捨てる。確定済みも 7 日ごとに作り直す | 該当のファイルを消す（Cloud Run に戻る） |
| 本番と違う中身（版・環境変数の食い違い） | 配備した版のタグのツリー・`edge_render_env.json`・テスト。段 1 で本番の応答と一致を確かめる | 版の置き場を消す（Cloud Run に戻る） |
| 古い中身（出来事の取りこぼし・ホストの停止） | 確定前は前日まで・確定後だけから始める。見張りで 15 分 | `_ctl/stale` を置く。`enabled: false` |
| フロントと応答の形の食い違い（UI だけ先に変わる） | API の版で置き場を分け、配備のときは作り直すまで空にする | 同上 |
| ホストの CLAUDE.md の決まり違反（GCS を読む） | 手元だけの守り（GCS に触れたら例外） | — |
| VPS の権限（deploy は sudo 不可） | 置き場・nginx・確認役の unit はユーザーの操作で入れる。戻すのは deploy が置ける `_ctl/OFF` で足りる形にする | `_ctl/OFF`。最後の手段は nginx の include を外して reload（root） |
| 開催日の取り込みを壊す（段 5） | 手元の書き込みは例外を外に出さない・止める設定 1 つ・平日に入れる | 止める設定を切る |

戻す操作は、どれも「フォルダを使わない → 今の Cloud Run の経路」に倒れます。今の nginx の匿名キャッシュと cache-warm は、段 6 まで残します。

## 7. お願いしたいこと・判断してほしいこと

ユーザーの操作（VPS の root が要る）:

1. `sudo journalctl -u keiba-cache-warm --since today | grep -c ' 429'`（latency_plan §7 の 4）。
2. ブラウザ（会員のログインあり）で `/races`・レース詳細を開き、開発者ツールで API の TTFB と LCP を 1〜2 回（§7 の 6）。
3. 段 2 で: `/srv/keiba-edge` の作成、確認役の unit、nginx の include の追加（下の「7.1 ユーザーの操作の手順」）。

### 7.1 ユーザーの操作の手順（段 2 の前に、統括役から頼む）

VPS の root（ConoHa のコンソールか、root で入れる ssh）で流します。`<…>` は実装の段で決まる値です。

**A. cache-warm の 429 の件数（latency_plan §7 の 4）**

```bash
sudo journalctl -u keiba-cache-warm --since today --no-pager | grep -c ' 429'
sudo journalctl -u keiba-cache-warm --since "-7 days" --no-pager | grep ' 429' | tail -5
```

**B. ブラウザの計測（会員でログインしたブラウザ。latency_plan §7 の 3・6）**

1. Chrome で https://app.megu-keiba.com/races を開き、F12 → Network、「Disable cache」は外す、絞り込みに `/api/` と入れる。
2. 再読み込みし、各行の `Waiting for server response`（TTFB）と、応答の見出しの `X-Cache-Status`・`cf-cache-status` を書き留める。
3. 確定済みのレースの詳細（例 `/race/202608050411`）でも 1〜2 回。
4. Lighthouse（F12 → Lighthouse → Performance・Mobile）で LCP を 1 回。

**C. 段 2 の置き場・確認役・nginx（stg の待ち受け 8081 から）**

```bash
# 1) 置き場（deploy が書き、www-data（nginx）が読む）
sudo install -d -o deploy -g www-data -m 2750 /srv/keiba-edge /srv/keiba-edge/stg /srv/keiba-edge/prod
sudo install -d -o deploy -g www-data -m 2750 /srv/keiba-edge/stg/_ctl /srv/keiba-edge/prod/_ctl

# 2) 区分の Cookie の鍵（Cloud Run の Secret と同じ値。env ごとに別）
sudo install -d -o root -g root -m 755 /etc/keiba-edge
sudo install -o root -g www-data -m 640 /dev/null /etc/keiba-edge/stg.key
sudo sh -c 'printf %s "<stg の EDGE_TIER_KEY>" > /etc/keiba-edge/stg.key'

# 3) 確認役の unit（リポジトリの scripts/ops/vps/ に置く版を配る。名前は実装の段で決める）
sudo install -o root -g root -m 644 <リポジトリ>/scripts/ops/vps/templates/keiba-edge-tier.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now keiba-edge-tier
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:<確認役のポート>/healthz   # 200

# 4) nginx（stg の server ブロックに include を足す。prod は段 4）
sudo install -o root -g root -m 644 <リポジトリ>/scripts/ops/vps/templates/keiba-edge-stg.conf /etc/nginx/snippets/
sudo nginx -t && sudo systemctl reload nginx

# 戻し（root 不要の方）: deploy で置けば reload なしに全部 Cloud Run
touch /srv/keiba-edge/stg/_ctl/OFF
# 戻し（最後の手段・root）: include を外して reload
sudo rm /etc/nginx/snippets/keiba-edge-stg.conf && sudo nginx -t && sudo systemctl reload nginx
```

- 確認役を止めても API が 500 にならず Cloud Run に流れることを、段 2 の合格条件で確かめます（`sudo systemctl stop keiba-edge-tier` → 対象のパスが 200）。

判断してほしいこと（推奨を先に書く）:

1. 直近 1 か月の初回の埋め: **案 A（手元にも書く設定を入れた日から先だけ。GCS 0 回）を推奨**。案 B は 1 回 約 3 円で、すぐ 1 か月分がそろう。
2. 区分の Cookie の期限と失効の後退: **2 時間を推奨**。課金を止めた人に、最大 2 時間フォルダの中身が出続けます（今のセッションは次の要求で止まる）。この後退を受け入れるかを決めてください。受け入れない場合は、課金の停止のたびに Cloud Run が `not_before` を上げる仕組み（Cloud Run → VPS の通知）が要り、別の設計になります。
4. 開発者の区分を VPS のディスクに置くか: **段 1〜4 は置かない（開発者は Cloud Run のまま）を推奨**。開発者の中身には、確定前は開発者だけの試行（第 2 段・v6c）が入るためです。
3. 当日・確定前の会員の predictions をフォルダに入れる時期: **段 7（2 週の開催で取りこぼしが無いと確かめてから）を推奨**。

## 8. 別のエージェントのレビュー（指摘と対応）

草稿を、作った本人とは別のエージェント（general-purpose）に、計画の本文・コード・生の計測（curl の結果・Cloud Run のログ）を材料として渡し、漏れ・費用・見落とし・戻し方の観点で見てもらった（2026-11-22）。

| 重さ | 指摘 | 対応 |
|---|---|---|
| 高 | `auth_request` は確認役が落ちる・期限切れ・5xx のとき 500 を返し、Cloud Run に流れない。戻すのに root の reload が要る | 直した。`error_page 500 502 503 504 = @cloudrun`、確認役への期限 100ms、deploy が置ける `_ctl/OFF`（rewrite の段で判定）を足した。段 2 の合格条件に「確認役を止める」を足した（§3.6・§5） |
| 高 | 「今のセッションもサーバ側の失効が無い」は誤り。今は要求ごとに `member_active`（課金）と `pw_changed_ts` を見る。課金の停止はホストで拾えない。`epoch.json` が版の置き場の中で、rsync の `--delete` で消える | 直した。区分の Cookie に発行時刻を入れ、版の置き場の外の `_ctl/not_before` で落とす。期限を 12 時間 → 2 時間にし、課金の停止で最大 2 時間出続ける後退を承認の項目にした（§3.5・§7） |
| 高 | GCS の守りは、例外の握りつぶし（`tier.py:81-84` など）で消える。KEIBA_ENV=prod では `gcp_forbidden()` が効かない。欠けた中身が確定済みとして長く返る | 直した。「GCS に触れた」「入力が無かった」をプロセスの印で持ち、印が立った応答は捨てる。段 1 の照合を数十レース × 全区分にした。確定済みも 7 日ごとに作り直す（§3.3・§3.7・§5） |
| 高 | `app.py` は import 時に `.env` を読む。`ASGITransport` の接続元は 127.0.0.1 で、開発者の自動ログインで匿名が開発者の中身になりうる。別のツリーのリンクが無いと黙って空 | 直した。空の dotenv・`GCS_BUCKET` 空・`KEIBA_DEV_AUTO_LOGIN=0`・接続元をループバック以外に。区分の差し込みは `is_member`・`is_developer` にも効かせる。必須の置き場が無ければ止める（§3.2） |
| 中 | member・dev の置き場を検査していない。L7 は「開発者は素通り」だった | 直した。member の置き場に開発者だけの項目が無いことを送る前に確かめる。開発者の区分は段 1〜4 では置かない（推奨）を承認の項目にした（§3.5・§6・§7） |
| 中 | free に anon の Cookie を出すと、auth/status を確認役が答える案で「ゲスト」表示になる。画面は `user_name` も使う | 直した。Cookie の区分は free を分け、ファイルだけ anon を返す。auth/status は段 1〜4 では Cloud Run のまま（§3.5） |
| 中 | `try_files` はクエリを見ない。race-list の `with_result_status`、tracking-difficulty の `refresh` が効かなくなる | 直した。`$args` が空でない要求は Cloud Run へ。optimize の GET の口もパスの形にした（§3.4・§3.6） |
| 中 | stg と prod は `server_name _` のポートで分かれている。stg には `KEIBA_TEST_USER` がある | 直した。ポートの server ブロックごとに置き場を分け、鍵も env ごとに分けて HMAC に env を入れる（§3.5・§3.6） |
| 中 | 確定済みでも、P1・試行の公開・`public_fields.json`・目印で中身が変わる。コードの版のタグでは拾えない | 直した。設定と目印のハッシュを manifest の入力に入れる（§3.7） |
| 中 | 段 5 で変える `HybridStorage.save` は開催日の取り込みと同じコード | 直した。例外を外に出さない・上限・止める設定 1 つ。壊しても取り込みが止まらないことをテストで確かめる（§5・§6） |
| 低 | 本番は `KEIBA_PREDICTION_RESULTS_GCS=1` で、予測成績は手元で作れない見込み | 直した。段 6 までは対象から外す（§3.3） |
| 低 | 401 は 4 種でなく 6 種。predictions の 200 は 1,659 件。nginx の置き場の 68KB は権限のせいの見込み | 直した。ログを数え直すと 401 は 10 種・4,098 件だった（レビューの 6 種よりも多い）。数字と費用を直し、68KB の記述を改めた（§0・§1・§4） |

## 9. 統括役の判断（2026-11-22。ユーザーは推奨で進めてよいとの指示）

| 項目 | 判断 |
|---|---|
| ① 1 か月分の埋め方 | 案 A（手元にも書く設定を入れた日から先だけ。GCS 0 回・0 円） |
| ② 区分の Cookie の期限 | 2 時間で可。**理由: 会員は共有アカウントで、課金を止めた人も月替わりのパスワードの切り替えまでは今も同じアカウントで入れる。そのため、区分の Cookie の 2 時間の遅れは、今の失効より実質的に後退しない。** 月替わりの切り替えは `not_before` で効く（§3.5） |
| ③ 当日・確定前の会員の予測 | 2 週の開催で取りこぼしが無いと確かめてから（段 7） |
| ④ 開発者の区分 | 段 1〜4 では VPS に置かない |
| 段 0 の時期 | 11/23 の開催の後（11/24 以降）。開催日に VPS の設定を変えない |
| 実装の前 | §3.5・§3.6 の 2 回目のレビューを通す |
| ユーザーの操作 | 段 2 の前に統括役から頼む（§7.1） |
| L3 の計測の結果 | latency-plan の後継（11/24 以降）に統括役から渡す |
| 次 | ここで区切り、11/24 の合図を待つ |

## 10. 追記（2026-11-23、担当 edge-cache）: 「本番の費用をほぼ 0 円に」を受けた順番の組み替え

ユーザーの指示（11/23）: 本番の費用がほぼ 0 円になるように、VPS の上で基本を解決する。統括役の §9 の判断（案 A・区分の Cookie 2 時間・開発者は置かない）はそのまま使う。

### 10.1 今の費用と、フォルダで消える部分

- 課金エクスポートの 11/16〜22 の 1 日あたり（円）:

  | 項目 | 1 日 | 月の見込み | フォルダで減るか |
  |---|---|---|---|
  | Cloud Run（CPU・メモリ・外への送信） | 15〜21 | 約 450〜630 | **減る**（下） |
  | GCS の保存（Standard Tokyo 約 1.8GiB） | 約 6.7 | 約 200 | 減らない |
  | GCS の操作（Class A・B） | 約 8〜15 | 約 250〜450 | 一部（cache-warm の合図の一覧・Cloud Run の起動の読み） |
  | Artifact Registry | 3〜8 | 約 100〜240 | 減らない |
  | Secret Manager | 約 0〜3 | 約 0〜90 | 減らない |

- Cloud Run の要求の 94%（11/21 は 8,796 件のうち 8,312 件）は VPS の cache-warm でした。edge-fix の E（11/23 19:34）で会員だけの 11 種（1 日 約 4,000 件）は外れました。残りの cache-warm は、匿名の race・predictions・race-list・成績などの取り直しです。
- フォルダで消せるのは、この cache-warm の取り直しと、利用者・SSR の MISS です。**GCS の保存・Artifact Registry・ホストの GCS の操作は残ります。「ほぼ 0 円」の下限は、Cloud Run を除いて月 約 550〜900 円です**（別の担当の範囲。この計画では減らない）。
- 素の基準（edge-fix の後）は、11/24 以降の課金エクスポートと VPS の timed.log で取り直します（11/23 夜の 3 時間分はテストの要求が多く、基準にならない）。

### 10.2 試しの結果（11/23 夜。ホストで API を手元だけで呼び、GCS には届かない形で読もうとしたものを数えた）

- 確定済みのレース 202608050411（11/15）で、匿名と会員（使い捨ての鍵で署名した、試しのプロセスの中だけのトークン）を呼んだ。
- `race/{id}` は GCS の 11 種（race_shutuba・race_result・race_result_on_time・race_result_lap・race_odds・race_pair_odds・race_index・race_barometer・race_paddock・race_trainer_comment・smartrc_race）が無いと、200 のまま 543〜785 バイトに欠ける（本番は 27KB）。**200 は正しさの印にならない。**
- `predictions` は race_predictions・race_shutuba_past・race_odds・race_result・race_shutuba（6 回）を読む。
- `race-list/{日}` は、その日のレースの race_shutuba と、race_result・race_result_on_time の年の一覧（`list_blobs`）と `exists` を使う。
- `tracking-difficulty` は手元だけで作れた（53KB）。`v1/races/{id}/analysis` は Flask 側の口なので、作る処理は Cloud Run と同じ Gateway（FastAPI＋Flask）を組んで呼ぶ。
- 1 本あたりの時間は 0.01〜1.2 秒（入力が欠けた状態）。
- `HybridStorage.load` は、`GCS_BUCKET` が空・1 回失敗した後の 60 秒の待避中は、Blob に触れずに None を返す。作る処理は、手元だけのカテゴリ以外で None を返した `load`・一覧・`exists`・`note_degraded` を要求ごとの印に記録し、印が立った応答は捨てる。1 件ごとに待避の状態を戻す。

### 10.3 順番の組み替え（§3.10・§5 からの逸脱）

**匿名を会員より先に作る。** §3.10 は「P1 が本番に入るまで匿名のファイルは作らない」としていたが、次の理由で変える。

- 費用の大半は匿名の取り直し（cache-warm）で、匿名は区分の Cookie も確認役も要らない（Cookie も Authorization も無い要求にだけ返す）。
- 匿名のファイルは、今 nginx に置いている匿名の応答と同じコードで作るので、中身は今より後退しない。
- P1 が配備されれば、版の置き場（`<env>/<sha12>/`）ごと作り直す（§3.2）。P1 の確定の前後の出し分けは、確定の合図（結果の保存）で作り直す。

| 段 | 中身 | いつ |
|---|---|---|
| E1 | 保存のときに手元にも写す。**コードは変えない。** 既にある `HybridStorage._maybe_write_local_mirror`（GCS への保存が成功した後に `data/local/mirror/<カテゴリ>/<年>/<キー>.json` へ同じ JSON を書く。失敗は警告だけで保存は成功のまま）を、ホストの `data/local/meta/local_mirror_config.json`（git の外）で 13 種に有効にする。35 日より古い写しは E2 の深夜の処理で消す | 11/23 夜（開催日の時間の外）に有効化。11/26 ごろの 11/28 の出馬表から貯まる |
| E2 | 作る処理（`src/ops/edge_render.py`）。本番の版・本番の環境変数・手元だけ・印が立てば捨てる。匿名（race・predictions・race-list）と会員（＋result・tracking-difficulty・analysis）。区分は、作る処理の中だけの使い捨ての鍵で署名したトークンで渡す（auth のコードを変えない） | 11/24〜27 |
| E3 | VPS の stg で匿名をフォルダから返す（Cookie も Authorization も無い要求だけ。無ければ今の経路）。`_ctl/OFF`・rsync（deploy）・置き場 | 11/27 に stg。11/28〜29 のレースで確かめ、本番は 11/30 以降の平日 |
| E4 | 区分の Cookie（Cloud Run）・VPS の確認役・会員のファイル | E3 の後。`EDGE_TIER_KEY` は、値を出さない gcloud のコマンドをユーザーに流してもらう（手元の gcloud は設定のディレクトリに書けず動かない） |
| E5 | 確定前・開催日の匿名を、入力の変化で作り直す。cache-warm のうちフォルダで足りるパスを止める | E3 の本番の後、開催 1 週で取りこぼしを確かめてから |

- 会員の照合: このマシンには会員のログインが無い。会員のファイルは stg の `KEIBA_TEST_USER`（stg の Cloud Run）の応答と比べるか、統括役に手段を頼む。
- nginx: 新しいスニペット（`nginx-edge-files.conf`）に分ける。ただし、サーバの段の `location`（名前付きの予備の経路）を足すには、サイトのテンプレート 3 つに include の 1 行が要り、edge-fix と重なる。順番は統括役に決めてもらう。
- 開催日（土日祝の 07:30〜17:00）は、ホストの開催日のコードと VPS の設定を変えない。

### 10.4 E1 の写しの対象と、書き手の確かめ

- 対象（13 種）: race_shutuba・race_result・race_result_on_time・race_result_lap・race_odds・race_pair_odds・race_index・race_barometer・race_paddock・race_trainer_comment・smartrc_race・race_predictions・race_shutuba_past。
- 書き手（scout の調べ）: どれもホストの cron・開催日のランナーが保存する。VPS は影（`shadow/`）にだけ書く。`/etc/keiba-fetch/fetch.env` に `KEIBA_FETCH_AGENT_TASKS` は無いことを 11/23 に確認した。
  - 例外: Cloud Run の API も、開発者だけの口（`/data-viewer`・`POST /api/race/{id}/predict`）と `GET /api/betting/pair-odds/{id}`（GCS に無いときに取得）で書きうる。ホストの写しは、その分だけ古くなりうる。
  - 対策: E2 は確定済みのレース（結果と払戻がある）だけから作る。確定の後にこれらの口で書き直されることは、まず無い。
- 副作用: ホストの `growth_curve_service` は、race_index の写しがあれば GCS より先に読む（`src/pipeline/inference/growth_curve_service.py:113`）。中身は同じで、ホストの GCS の読み出しが減る向き。
- 費用: GCS の操作は 0 回の増加（保存の成功の後に手元へ書くだけ）。ディスクは 1 レース 13 種で 0.1〜1MB 程度、35 日で 1GB 未満の見込み（空き 5.9TB）。

### 10.5 E2・E3 の作り（11/23 実装）

| 部品 | 場所 | 中身 |
|---|---|---|
| 作る処理 | `src/ops/edge_render.py render` | Cloud Run と同じ Gateway（FastAPI＋Flask）を同じプロセスで呼ぶ。環境は作り直す（ホストの `.env`・`KEIBA_*` を持ち込まない。`config/edge_render.json` の Cloud Run の値だけ） |
| 守り | 同 | `HybridStorage` の置き場を写し（`data/local/mirror`）に差し替える。次のどれかが起きた応答は「印」が立ち、捨てる。200 以外も捨てる。<br>・GCS のクライアントを作る（`GCS_BUCKET` は無いバケット名にして、GCS に行く口は必ずここに当てる）<br>・外へのソケットの接続・HTTP・保存<br>・写しの対象外のカテゴリ、写しに無い馬のキー（読む・有無・更新時刻・一覧）<br>・`exists_gcs`・`get_scraped_at_from_blob`・`list_years`・`note_degraded`<br>・**WARNING 以上のログ**（API の付与の失敗は握りつぶされ、欠けた 200 になるため）<br>印が立ったレースは、同じ回のほかの対象・そのレースの日の一覧・前の回のファイルも消す（プロセスの中のキャッシュで欠けを持ち越さない。VPS からも消えて Cloud Run の経路に戻る） |
| 手元の別の置き場 | `config/edge_render.json` の `local_fallback` | 血統（`data/local/horse_pedigree_5gen`）・調教（`data/local/horse_training`）。GCS と同じ並びの手元の写し |
| 対象 | 同 | 写しに出馬表と結果があるレース（確定済み）の `race/{id}`・`predictions`（匿名）。開催日の全レースが揃った日の `race-list/{日}`。開催日は「写しの有効化＋5 日」より後・直近 35 日・前日まで |
| 作り直し | 同 | 写しのファイルの更新時刻と大きさ（指紋）が変わったものだけ。推論の出力の変化は指紋に入らないので、深夜に `--full` で全部作り直す。本文の MD5 が同じなら書き換えない |
| 送る | `src/ops/edge_render.py push` | 手元の記録（manifest と送った記録）で差分を決め、ssh の tar で送る。VPS では 1 ファイルずつ `os.replace`。最後に `current` を付け替え、古い版は 1 つ残す。ホストに rsync は無い |
| 段取り | `scripts/ops/edge/edge_sync.sh <env>` | 最後に成功した配備（prod は promote.yml、stg は deploy-stg.yml）の commit を別のツリー（`/home/jovyan/work/keiba-edge/<env>/src`）に取り出し、git に無い係数（`config/fitted_artifacts.txt`）をホストから重ねて、作って送る。jovyan で流す |
| VPS | `scripts/ops/vps/vps_14_edge_files.sh --env <env>`・`templates/nginx-edge-files.conf` | 置き場 `/srv/keiba-edge/<env>/`（deploy:www-data 2750）。サイトの `location ~ ^/api/` の前に include の 1 行。匿名（Cookie も Authorization も無い）・GET/HEAD・クエリなし・合言葉なし・`_ctl/OFF` なしのときだけフォルダを見る。無ければ `/api/__edge_miss$uri` に内部の転送をして、今の `/api/` の location（上流へは元の `$request_uri`）が受ける |

試しの結果（11/23 夜）:

- 照合: 11/15 の確定済みの 2 レースで、写し（試しのため、この 2 レースの 17 種と出走馬だけ GCS から 1 回取った。約 2.4MB、0.1 円未満）から作った匿名の応答を本番の公開 URL と比べた。
  - `race/{id}`: 項目まで一致した。
  - `predictions`: 確率まで一致した。違いは `pipeline_status`（進み具合の表示）の時刻と文言だけ。ホストは経過の記録を持つので、作った側の方が正確。
- 作る時間: 2 レース × 2 区分 × 最大 5 種で 8〜11 秒（API の読み込みを含む）。
- stg（8081）に vps_14 を入れて確かめた。
  - フォルダにあるもの: `X-Edge: hit` で 1ms 前後。gzip あり・なし、HEAD のどれも通る。
  - フォルダに無いもの、クエリ付き、Cookie 付き、cache-warm の合言葉つき、`_ctl/OFF` のとき: 今の経路（`X-Cache-Status`）に流れる。
  - 本番（8080）は変わらない。
- 足りなかった入力: 最初は horse_result・horse_training・horse_pedigree_5gen・race_oikiri・final_odds_prediction。
  - 写しを 17 種に広げた（11/23 22:3x JST）。
  - 血統と調教は、手元の別の置き場から読む。

ホストの cron（統括役が共有の作業ツリーを早送りした後に足す。root の cron から jovyan で）:

```
*/15 * * * * nice -n 10 runuser -u jovyan -- sh -c 'bash /home/jovyan/work/keiba-vpn/scripts/ops/edge/edge_sync.sh stg >> /home/jovyan/work/keiba-edge/sync-stg.log 2>&1' # KEIBA_EDGE
40 4 * * *   nice -n 10 runuser -u jovyan -- sh -c 'EDGE_RENDER_ARGS=--full bash /home/jovyan/work/keiba-vpn/scripts/ops/edge/edge_sync.sh stg >> /home/jovyan/work/keiba-edge/sync-stg.log 2>&1' # KEIBA_EDGE
20 4 * * *   cd /home/jovyan/work/keiba-vpn && nice -n 10 /opt/conda/bin/python3 -m src.ops.edge_render prune --mirror-days 40 >> /home/jovyan/work/keiba-vpn/logs/edge_prune.log 2>&1 # KEIBA_EDGE
```

- prod は、stg で 11/28〜29 の確定分を確かめた後（11/30 以降の平日）に同じ形で足す。
- edge_sync は jovyan で動かす。ログも jovyan のシェルが開くので、jovyan の持ち物になる。
- 写しの掃除だけは root で動かす。
  - 理由: 写しは root の cron（取得）が書くので、写しのフォルダは root の 755 で、jovyan では消せない。
  - 掃除は消すだけで、写しの置き場にファイルを作らない。ログは共有の作業ツリーの logs/（root の持ち物）に書く。
  - 消すのはレースのカテゴリだけ。馬のカテゴリは、直近のレースに出た馬でも最後の保存が古いことがあるので消さない。
  - 04:20 に流し、04:40 の `--full` と重ねない。
- ログ: `sync-<env>.log` は edge_sync.sh が 5MB で 1 世代に切り替える。`render-<env>.log` は毎回上書き。`edge_prune.log` は 1 日 1 行。
- 診断: 確定前のレースも作ってみて、捨てる理由を数える。15 分ごとの回と同じツリー・同じロックで流す（途中で版が替わらないように）。

  ```
  T=/home/jovyan/work/keiba-edge/stg/src
  cd $T && KEIBA_PAGE_REFERENCE_DIR=/home/jovyan/work/keiba-vpn/data/page_reference PYTHONPATH=$T PYTHONPYCACHEPREFIX=/home/jovyan/work/keiba-edge/pyc \
    flock /home/jovyan/work/keiba-edge/.lock-stg nice -n 19 ionice -c3 /opt/conda/bin/python3 -m src.ops.edge_render render \
    --env stg --out /home/jovyan/work/keiba-edge/out --base /home/jovyan/work/keiba-vpn --date 20261128 --diagnose --full
  ```

  - `<out>/_diagnose/` に書くだけで、送らない。
  - 11/26 ごろ、11/28 の出馬表が写しに入ったら流す。
  - 終わったら `rm -rf /home/jovyan/work/keiba-edge/out/_diagnose` で消す。

分かっている穴（受け入れる理由）:

- 写しは「ホストが GCS に保存できた後」に書く。次の場合、そのレースは「写しに無い＝GCS に無い」とみなされ、欠けた中身になりうる。
  - 写しの書き込みの失敗（警告だけ）
  - `save_mode=gcs_only`（`skip_local_mirror`）の保存。開発者の手動の取り直しの口だけ。
  - 写しの設定を一時的に切った間の保存
  - 理由: どれもまれで、欠けたレースは深夜の `--full` でも直らない。そのため、気づいたらそのレースの写しを入れ直すか、`_ctl/OFF` を置く。カテゴリを足すときは `mirror_since` を足した日に進める（窓の中の古いレースに写しが無いため）。
- critic（11/23）の指摘で直したもの: gh の `-R`、捨てたときに前の回のファイルも消す、有無・更新時刻の予備と印、`GCS_BUCKET` を無いバケットに、ソケットの塞ぎ、WARNING を印に、同じレース・日の巻き添えで消す、`Vary: Cookie`、rewrite でメソッドを保つ、data/page_reference・data/cache・models のリンク、深夜の回の待ち、`--resync`、写しの掃除の既定を 40 日。

- フォルダから外れたとき（`_ctl/OFF`・`vps_14 --remove`・印が立ったレースのファイルを消したとき）は、nginx の API キャッシュ（確定済みは 30 日）が返る。cache-warm はフォルダにあった間それを取り直していないので、その間に更新があったパスは古い。
  - `_ctl/OFF` を置いた・`--remove` を流したとき（`--remove` は OFF も置く）は、続けて `sudo bash scripts/ops/vps/purge_api_cache.sh` で API キャッシュを全部消す。費用は edge-fix の段 1 と同じく最悪 約 2 円（1 回）。
  - 印でファイルを消したレースは、確定済みで、更新はまれ。気づいたら、そのレースだけ取り直す（cache-warm の合図か purge）。
- 同じプロセスの中のほかのモジュールのキャッシュ（負のキャッシュ）が、別のレースへ欠けを持ち越す道は残る（HybridStorage は None を覚えない）。WARNING を印にしているので、多くはそこで捕まる。
- WARNING 以上をすべて印にしているので、害の無い WARNING が毎回出ると、全部が捨てられる（安全側）。heartbeat の `reasons` で見つけたら、その WARNING の出所を直すか、ここで除く一覧を足す。

**cache-warm との関係（費用が減る条件）:** フォルダは cache-warm（合言葉つき）を素通りさせる。cache-warm が確定済みのレースを取り直す限り、Cloud Run の要求は減らない。匿名の利用者の要求は、もともと nginx の API キャッシュの HIT が多い。そのため、E3 を本番に入れるときに、cache-warm（VPS）の取り直しの前に、フォルダにファイルがあるパスは取り直さない形にする。cache-warm は VPS で動くので、`/srv/keiba-edge/prod/current/anon<パス>.json` の有無を見るだけでよい（GCS も Cloud Run も使わない）。
→ 実装済み（`scripts/ops/vps/cache_warm.py` の `edge_body`）。フォルダにファイルがあり、`_ctl/OFF` が無く、本番のスニペット（`/etc/nginx/snippets/keiba-edge-prod.conf`）があるときだけ取り直さない。本文はファイルから読み、開催日の一覧の取り直しの手がかりに使う。VPS には E3 の本番と同じときに `vps_10_cache_warm.sh` で入れる。

### 10.6 費用（最悪の回数）

| 項目 | 最悪の回数 | 月額 |
|---|---|---|
| 写し（保存の後に手元へ書く） | GCS の操作は増えない | 0 円 |
| 作る処理の GCS | 0 回（クライアントを作ろうとした時点で例外。応答は捨てる） | 0 円 |
| 作る処理の Cloud Run | 0 回（同じプロセスの中で呼ぶ。外へ HTTP を出さない） | 0 円 |
| ホスト → VPS | ssh の tar。15 分ごと × 96 回/日。変わったファイルだけ（確定の日に 1 日 数 MB、版の切り替えで最大 約 50MB） | 0 円（GCP の外どうし） |
| VPS のディスク | 匿名 2 種で 1 レース 約 60KB（gzip を並べて）× 35 日 約 360 レース ≒ 25MB × 版 2 つ | 0 円（空き 71GB） |
| ホストの CPU | 15 分ごとの起動（API の読み込み 約 5〜10 秒）＋変わったレースだけ（1 本 0.1〜1 秒）。深夜の `--full` は最悪 360 レース × 2 種 × 約 1 秒 ≒ 12 分（nice 19） | 電気代のみ |
| Cloud Run | 減る。フォルダに当たった匿名の要求は Cloud Run に届かない。cache-warm の取り直しは、上の「cache-warm との関係」の直しを入れてから減る | 減る方向（条件付き） |

- Cloud Run の月額は、フォルダに無い要求（会員・ログイン・確定前・35 日より前）の分だけ残る。利用者の数に比例して増えるのは、会員の分（E4 まで）と確定前の分（E5 まで）。
- 最悪の起動の回数は今と同じ（96 回/日 × 30 日）。起動 1 回は 約 30 秒 × 1 vCPU・4GiB ≒ 0.15 円 → 月 最大 約 430 円。会員・確定前がフォルダに入れば、起動のきっかけ自体が減る。

## 11. E5 の計画（2026-11-24、担当 edge-cache）: 確定前・開催日の匿名をフォルダから返す

狙い: cache-warm の取り直しの中心（確定前のレースの `race/{id}`・`predictions` と開催日の `race-list`）を、ホストの作り直しに置き換える。11/21 の cache-warm の 200 は 約 4,200 件/日で、そのうち 約 3,300 件がこの 3 種だった。

### 11.1 確定前の応答の、作り直しのきっかけ

入力の変化（scout の調べ。path は 11/24 の develop）:

| 入力 | 置き場 | 粒度 | 匿名の応答への効き |
|---|---|---|---|
| 出馬表・速報・結果・オッズ・指数など 17 種 | `data/local/mirror/<cat>/<年>/<race_id>.json` | レース | 詳細・予測（市場補正後の勝率・pipeline_status のオッズの段）・一覧 |
| 予測 v6e | `data/local/inference/predictions_v6/<日>/<race_id>.parquet`（`weight_triggered_predict.py:383`） | レース | 予測 |
| 予測 v6c（候補） | `data/local/inference/predictions_v6c/<日>/<race_id>.parquet`（`candidate_runner.py:550`） | レース | 予測（未勝利・新馬） |
| 予測の経過 | `data/local/inference/events/<日>.jsonl`・`launch_watch/<日>.jsonl`（追記） | 日（行に race_id） | pipeline_status の段と時刻 |
| 出馬表の段階 | `data/local/meta/entries_state/<race_id>.json`（`entries_sync.py:225`） | レース | entries_status・pipeline_status・一覧 |
| その日のレースの一覧 | `data/page_reference/race_lists/<日>.json` | 日 | 一覧 |
| 設定・係数 | `config/edge_render.json`・git に無い係数・版 | 全体 | 全部 |

時刻で変わる値（入力が同じでも変わる）:

- `pipeline_status.race_over`（発走 ＋5 分）と、予測の段の文言（予測の窓の終わりの前後）（`pipeline_status.py:110・139`）
- `pipeline_status.as_of`（作るたびに変わる）
- `entries_status.is_future`（日付）
- 出馬表の段階の推定（状態ファイルが無いときだけ。木曜 12:00）
- レース詳細の品質の警告（検査が古くなると付く）

### 11.2 作り

- **見張り（新）: `src/ops/edge_watch.py`**
  - API を読み込まない軽いプロセスで、30 秒ごとに上の入力の更新時刻・大きさを見る（stat だけ）。
  - events・launch_watch は、前回の位置から後ろに足された行だけを読み、race_id を拾う。
  - 次のときに、そのレース（と、その日の一覧）を「作り直す」に積む。
    - 変わった入力がある。
    - 時刻の区切りが来た（発走 ＋5 分・予測の窓の終わり・0 時）。
    - 安全網: 当日の確定前のレースは、前回から 10 分。
  - 積んだものがあれば、`edge_render render --race … --date …` を新しいプロセスで起こす（API の中のキャッシュを持ち越さない）。続けて `push` する。
  - 起こすのは 1 度に 1 つ。終わるまで次は積むだけにする（最短 30 秒おき）。
- **作る処理: `edge_render`**
  - 対象に「確定前（写しに出馬表がある・今日から 7 日先まで）」を足す。
  - 本文の比較では `pipeline_status.as_of` を除く。as_of だけが変わったときは書き換えず、送らない。
- **systemd ではなく cron の 1 分ごとの起動＋flock**（ホストはコンテナで systemd が無い）。
  - 1 回の起動で 約 55 秒、30 秒ごとに 2 周して終わる。前の回が動いていれば何もしない。
  - 止める: `config/edge_render.json` の `watch_enabled: false`。
- **範囲:** 匿名だけ（E3 と同じ nginx の口。nginx の設定は変えない）。会員は E4。

### 11.3 量の見積もり（開催日・最悪の置き方）

前提: 1 日 36 レース（3 場）、匿名 2 種（詳細・予測）＋その日の一覧 1 本。1 本を作るのに 0.3〜1 秒（11/23 の試し）、プロセスの起動に 約 5〜10 秒（API の読み込み）。

| 項目 | 回数・量 | 見積もり |
|---|---|---|
| 入力の変化による作り直し | 1 レースあたり 約 12 回（出馬表・馬体重 3、オッズ 2、予測 3、経過 2、速報・結果 2）× 36 × 2 種 | 約 860 本/日 |
| 安全網（10 分ごと、09:00〜17:00 の確定前） | 最悪 48 回 × 36 × 2 種（実際は発走の後は確定に移って減る） | 最悪 約 3,500 本/日 |
| 一覧の作り直し | 変化のたび × 3 日分 | 約 500 本/日 |
| 作る CPU | 約 4,900 本 × 最大 1 秒 | 最悪 約 80 分/日（1 コアの換算） |
| 起動の CPU | 最短 30 秒おき × 8 時間 ＝ 最悪 960 回 × 10 秒 | 最悪 約 160 分/日 |
| 合計の CPU | 最悪 約 4 コア時間/日（nice 19・ionice idle） | ホスト 24 コアの 8 時間（192 コア時間）の 約 2% |
| ディスクの書き込み | 中身が変わったときだけ。約 1,400 本 × (20KB＋gz 5KB) | 約 35MB/日 |
| VPS への転送 | 変わった本文の gz を tar（ssh） | 約 10〜20MB/日。GCP の外どうしで 0 円 |
| GCS の操作 | 0 回（作る処理は GCS のクライアントを作れない。見張りは stat だけ） | 0 円 |
| 平日 | 確定前は 7 日先までの出馬表のあるレースだけ。変化は少なく、安全網は当日だけ | 1 日 数分 |

開催日の処理との重なり:

- raceday-runner・weight-runner・影のランナー（candidate_runner）は、T-60・T-45・T-15・馬体重の時刻に推論で CPU を使う。今は平時の負荷 3.5（24 コア）。
- 見張りの作る処理は nice 19・ionice idle で、取り合いのときは必ず後に回る。最悪 1 コアを使い続けても、24 コアのうち 1 つ。
- 推論の出力（parquet）を読むだけで、書き込みはしない。ロックも取らない。
- 写しの書き込み（取得の保存の後）は今と同じ。見張りは足さない。
- 遅れの上限:
  - 入力が変わってから VPS に届くまでは、見張り 30 秒＋起動 10 秒＋作る数秒＋送る 1〜2 秒で、約 1 分以内。今は s-maxage 30 秒＋swr 30 秒で最大 60 秒。
  - 負荷が高く遅れた分は、nginx が古いファイルを返し続ける。
  - 見張りが止まったときのために、VPS の見張りで `heartbeat.json` が 5 分より古ければ、確定前のファイルを使わない。これは 11.5 の段 3。

### 11.4 cache-warm の止め方と戻し方

- **止まり方（自動）:** E3 の `cache_warm.py` の `edge_body` が、フォルダにファイルがあるパスの取り直しをしない。E5 で確定前の詳細・予測・一覧がフォルダに入れば、その分の cache-warm の Cloud Run の要求が消える。
  - 残るのは、race-calendar・scrape-dates・prediction-results の 3 種と、クエリ付きの一覧。これらは段 6 でクエリの正規化をしてから。
- **戻し方:** `touch /srv/keiba-edge/prod/_ctl/OFF`（deploy で置ける。reload なし）。
  1. nginx がフォルダを見なくなる。
  2. cache-warm も取り直しに戻る（`edge_body` が OFF を見る）。
  3. 続けて `sudo bash scripts/ops/vps/purge_api_cache.sh` で API キャッシュを消す（フォルダにあった間、取り直していないため。最悪 約 2 円・1 回）。
- 作る側を止める: `config/edge_render.json` の `watch_enabled: false`。または crontab の `KEIBA_EDGE` の行をコメントにする。
- **cache-warm そのものは止めない（timer は残す）。** フォルダで足りない時（ホストの停止・OFF）の予備として、今の取り直しが自動で戻る形にしておく。

### 11.5 段取り

| 段 | 中身 | いつ |
|---|---|---|
| 1 | edge_render に確定前の対象・as_of を除く比較。見張り `edge_watch.py`（テスト・critic） | 11/24〜26（平日） |
| 2 | stg で、11/28 の出馬表（11/26 ごろ写しに入る）を使い、診断と見張りを平日に回す | 11/26〜27 |
| 3 | VPS の stg に、heartbeat の古さで確定前を止める見張り（timer）を入れる | 11/27 |
| 4 | stg の見張りを、11/28・29 の開催日に動かすか（統括役の判断。新しい処理を開催日に初めて動かすことになる） | 11/27 の報告で判断を仰ぐ |
| 5 | 本番: vps_14 --env prod・vps_10（cache-warm）・見張りの prod | 11/30 以降の平日。開催日の 07:30〜17:00 には入れない |

- 費用の確かめ: 本番に入れた 1〜2 日後に、課金エクスポートで Cloud Run の CPU・要求と GCS の操作を、入れる前の 1 週と比べる（CLAUDE.md）。
