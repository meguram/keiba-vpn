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
  - **課金を止めた人に、最大 2 時間フォルダの中身が出続けます。** 今のセッションは次の要求で止まるので、ここは今より後退します（承認の項目。§7）。
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
3. 段 2 で: `/srv/keiba-edge` の作成（deploy が書け、www-data が読める）、確認役の unit、nginx の include の追加。手順は実装の段で書く。

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

