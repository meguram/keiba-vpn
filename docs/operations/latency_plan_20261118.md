# 表示の遅延を ms に近づける計画（GCP の費用は据え置き・2026-11-18）

ユーザーの依頼: GCP の費用を変えずに、すべてのアクセスポイントで ms レベルの表示速度に近づけたい。本番は ConoHa の VPS で動いている。
この段は**調査と計画だけ**です。設定・本番・VPS は何も変えていません。実装はユーザーと統括役の承認の後に行います。

調べ方: 3 つの係（経路の地図・計測・費用の監査）をサブエージェントに分けて並行で調べ、食い違いはコードと公開 URL で確かめ直しました。
草稿は、作った本人とは別のエージェントにレビューさせ、指摘を反映しました（§9）。

## 0. 要点

- **いちばん大きい事実**: 今の実際の利用者（ログインした会員・所有者）の要求は、**VPS の nginx のキャッシュも Cloudflare のキャッシュも素通りして、毎回 Cloud Run に届きます。**
  - nginx は Cookie が 1 つでも付いた要求を、読みも保存もしません（`scripts/ops/vps/templates/nginx-zones.conf:10-13`）。
  - 30 日のキャッシュと cache-warm が効くのは、Cookie の無い匿名の訪問者だけです。匿名の訪問者がどれだけいるかは、まだ分かりません（§7 の 3）。
- **本番 app は Cloudflare Access の外にあります**（2026-11-18 に確認: Cookie なしで `/races` が 200。stg だけが Access の 302）。docs の「app は Access の内側」（`go_live.md`・`tiered_access.md:38`・`cdn_cache.md:28`）は古い記述です。
  - **これは表示の速さとは別に、ユーザーの判断が要る事項です。** `config/public_fields.json` は `mode: private`（「Access で所有者のみ」が前提）のままで、匿名の応答が不特定多数に届く状態です。`tiered_access.md:34` は、Access を外す前に `KEIBA_TEST_USER`（test/test）を off にすることを求めています。
  - 配備スクリプトでは `KEIBA_TEST_USER` は `API_TEST_USER=1` のときだけ付きます（`scripts/gcp/deploy_cloud_run_api.sh:78`。既定は付かない）。本番の今の値は、このマシンの gcloud の認証が root 専用のため確かめていません（本番に test/test でログインを試すことはしていません）。
  - 意図した公開かどうかを確かめるまで、匿名の応答を Cloudflare に広げる L6 は入れません。
- **推奨の上位は、どれも GCP の費用が 0 円か、減る方向**の案だけにしました。11 月はすでに上限を超えていて、12 月の余裕は 約 370〜570 円しかありません（§5）。
- **会員に効く手は、ブラウザと画面の作り（L1・L2a・L11）、起動の短縮（A2）、nginx → Cloud Run の往復（L3）に限られます。** nginx と Cloudflare のキャッシュ（L4・L6）は匿名の訪問者向けです。
- 「ms レベル」に届かないもの: コールドスタート（14〜28 秒。最小インスタンス 1 は月 約 4,900 円で ×）と、キャッシュに無い重い計算（0.2〜10 秒）。どちらも、表示は L1 で前回の内容を先に出して見えにくくするのが、費用内でできる限界です。

### 推奨の順番（期待の効果・月額）

| 順 | 案 | 効く相手 | 期待の効果 | GCP の月額 |
|---|---|---|---|---|
| 1 | L2a 保守の確認で HTML・画面の行き来を待たせない | 全員 | HTML・RSC の最悪 +1.5 秒 → 0 | 0 円（要求の回数は同じ） |
| 2 | L11 画面の API の直列をやめ、並べて取る | 全員（とくに会員） | `/races` などで往復 1 回分（会員 120〜300ms）縮む | 0 円 |
| 3 | L1 ブラウザに前回の内容を残して先に出す（SWR）を主な画面に広げる | 同じタブでの再読み込み・戻る・行き来 | 数百 ms〜28 秒 → 数十 ms（裏で更新） | 0 円 |
| 4 | L3 nginx → Cloud Run の接続を使い回す | 会員の API すべて | 10ms 前後〜数十 ms（VPS での計測で決める） | 0 円 |
| 5 | A2 起動の短縮（既存の案。import を遅らせる） | 冷えた時の全員 | 起動の待ち 14 秒のうち数秒 | 0 円（起動 1 回の CPU が減る） |
| 6 | L9 300 秒空いた後の 8 秒待ちを狭める | 全員 | 静かな時間の後の最初の表示: 最悪 8 秒 → 数秒以下（要計測） | 0 円 |
| 7 | L6 Cloudflare の Cache Rule（`/api/` の許可リストだけ・短い Edge の期限）＋アイコンの期限 | 匿名 | 54ms → 約 30ms。アイコンの毎回の確認（約 150ms）が消える | 0 円 |
| 8 | L4 nginx の期限切れを裏で取り直す | 匿名（calendar・scrape-dates など） | 期限切れ直後の 1 人: 最大 1.2 秒 → 約 54ms | 0 円 |
| 9 | L2b 保守のタブの問い合わせを減らす | — | 速さには効かない。決まり（15 分未満の問い合わせ）に合わせる | **×（今の形では）: 最悪 +約 700 円**（コールドスタートが増える）。起動の回数の実測で、最悪でも余裕の内と示せたら見直す。決まりと費用のどちらを優先するかは、ユーザーの判断 |
| 10 | L7 会員の応答を区分ごとに共有キャッシュ | 会員 | 会員の API 120〜300ms → 約 54ms | 0 円（減る方向）。**秘密に関わるので別の段階** |
| 11 | L8 重い計算の応答の事前生成（既存の Jobs に相乗り） | 全員 | 初回 2〜10 秒 → 数百 ms | 約 40 円（△） |
| — | L5 Cloudflare の 0-RTT | 再訪 | 約 14ms（1 往復） | 0 円（任意。HTTP/3 は有効になっている） |

×（費用の上限・決まりに反する）: 最小インスタンス 1、VPS から GCS を直接読む、cache-warm の周期を上げる、15 分未満の ping で温め続ける（§4）。

## 1. 経路の地図

```
ブラウザ ─(TLS)→ Cloudflare（NRT・HTTP/2 と HTTP/3）─ cloudflared トンネル → VPS nginx 127.0.0.1:8080（stg は 8081）
   ├─ /_next/static・public の静的ファイル → Next.js（VPS 上の standalone。CF が拡張子で保存）
   ├─ 画面の HTML・RSC                     → Next.js（middleware が保守の状態を確認 → 事前生成の殻を返す）
   ├─ /data/image/*                        → nginx 画像キャッシュ（7 日。Cookie を見ないので会員にも効く）→ Cloud Run
   └─ /api/*（FastAPI）・/api/v1/*（Flask） → nginx API キャッシュ（匿名・public だけ）→ Cloud Run keiba-api
                                                 └ cloudrun_entry.Gateway → prerender（/tmp）→ メモリ LRU → GCS
```

- Next.js はサーバ側でデータを取りません。画面のデータはすべてブラウザが `/api` を直接呼びます（`lib/api.ts` はどこからも import されない。ISR・revalidate も無い）。
- nginx から Cloud Run へは `proxy_pass https://$upstream_host$request_uri`（`nginx-keiba-ui-cloudrun.conf:21`）で、変数を使うため `upstream{}` の keepalive が効きません。要求ごとに TCP と TLS を張り直していると見られます（L3）。
- middleware の保守の確認と cache-warm は、どちらも 127.0.0.1 から nginx に入るので、同じ流量制限（`keiba_api` 120 回/分・burst 60）の枠を分け合います（`nginx-keiba-ui-cloudrun.conf:10-11,20`、`nginx-zones.conf:2`）。

### 各段のキャッシュ

| 段 | 場所 | 期間 | 鍵 | 無効化 | 根拠 |
|---|---|---|---|---|---|
| ブラウザ（メモリ） | `httpCache.ts` | 既定 15 秒・予測 45 秒・auth/status 20 秒・race-list 60 秒 | URL | 期限 | `frontend/lib/httpCache.ts:57-75` |
| ブラウザ（sessionStorage） | `persist` 指定だけ | 最大 30 分。同じタブの中だけ | URL | 会員区分が変わったら・ログアウトで捨てる | 同 `:28-36`、`Nav.tsx:40`。使っているのは 2 画面・6 か所だけ |
| ブラウザの HTTP キャッシュ | — | 多くの画面が `cache:"no-store"` で使っていない | — | — | `races/page.tsx:176` ほか |
| Cloudflare | 静的ファイルだけ（拡張子） | `immutable` | URL | — | HTML・API はすべて `cf-cache-status: DYNAMIC`（Cache Rule 未設定） |
| nginx API | `/var/cache/nginx/keiba_api` | 上流の `s-maxage`。日付・レース ID の付くパスは当日・未来 6 時間、確定済み 30 日。calendar 120 秒・scrape-dates 600 秒・prediction-results 3,600 秒 | `$request_uri` | `purge_api_cache.sh`、cache-warm の上書き | `nginx-api-cache.conf:15-24`、`config/http_cache.json`、`http_cache_headers.py:113-120` |
| nginx 画像 | `/var/cache/nginx/keiba_img` | 200 は 7 日・404 は 1 分 | URL | `purge_img_cache.sh` | `nginx-zones.conf:6` |
| cache-warm | VPS の timer 60 秒おき | 目印があったパスだけ取り直す | — | GCS の合図のメタデータを 1 回読むだけ。一覧は 5〜30 分に 1 回 | `scripts/ops/vps/cache_warm.py:281-320` |
| Cloud Run prerender | `/tmp/keiba-prerender`（100MB） | 既定 120 秒、確定した過去日は 6 時間 | パス | `X-Keiba-Fresh` | `src/api/prerender.py:84-118` |
| Cloud Run 応答のまとめ | メモリ | race-list ±2 日は 45 秒、ほかは 600 秒 | パス | 期限 | `src/api/response_cache.py:84-91` |
| HybridStorage LRU | メモリ | 900 秒（出馬表 300 秒）・2,000 件・256MB | GCS のパス | 期限 | `src/api/cloudrun_entry.py:39-41` |
| ディスク L2 | — | Cloud Run では実質使わない | — | — | `cloudrun_entry.py:46` |

### 要求の処理中に GCS を同期で読む箇所（Cloud Run の中。同じリージョンなので転送料は 0 円）

- `src/scraper/storage.py:998-1003` — メモリ LRU が外れたとき（4 秒で打ち切り）。
- `src/ops/gcs_mirror.py:550-575`（`fetch_one`・`lazy_fetch`）— 窓（14 日）より古い予測・lap・tracking。
- `src/ops/gcs_mirror.py:743-787` と `cloudrun_entry._gate` — 前回の取り込みの成功から 300 秒以上空いた後の最初の要求は、目録の取り込みを最大 8 秒待つ（L9）。
- `tracking_difficulty_store.py:52`（600 秒ごと）、`prediction_results.py:621-662`（一覧は 60 秒）。

### キャッシュしてはいけない線（ログイン・会員の秘密）

- `/api/v1/auth/*`・`/api/auth`・admin・maintenance は `http_cache.json` の never にあり、nginx も Cookie 付きを素通りさせます。
- **nginx の API スニペットに `proxy_cache_valid` を足さない**（足すと private や auth/status まで保存されうる）。
- Cloudflare は `Vary: Cookie` を見ないので、Cloudflare に置くときの守りは Cache Rule の条件だけになります（L6）。
- 会員名簿（`data/local/auth/members.json`）は束に入れない（`gcs_mirror.py:193`）。
- 会員の応答を共有キャッシュに置くのは、区分を nginx が信用できる形で知れるようになってから（L7）。

## 2. 実測（2026-11-18、ホスト PC（WSL の curl）から Cloudflare NRT 経由。公開 URL へ合計 約 100 回だけ）

測り方（1 件ずつ）:

```
curl -s -o /dev/null -D h.txt -w 'dns=%{time_namelookup} con=%{time_connect} tls=%{time_appconnect} ttfb=%{time_starttransfer} total=%{time_total}\n' https://app.megu-keiba.com/races
```

### 本番 app、新しい接続で 3 回（TTFB・ms）

| URL | 1 回目 / 2 回目 / 3 回目 | キャッシュのヘッダ |
|---|---|---|
| `/` | 438 / 208 / 216 | x-nextjs-cache: HIT、cf: DYNAMIC |
| `/login` | 184 / 233 / 191 | 同上 |
| `/races` | 168 / 172 / 146 | 同上 |
| `/weekly-predictions` | 183 / 137 / 166 | 同上 |
| `/api/v1/auth/status` | 248 / 228 / 301 | nginx MISS（Cloud Run まで） |
| `/api/race-list/20261115`（確定済み） | 137 / 201 / 182 | nginx HIT |
| `/api/race-list/20261122`（来週） | **465** / 151 / 138 | MISS（初回の取得）→ HIT |
| `/api/scrape-dates` | **1,182** / 133 / 142 | MISS → HIT |
| `/api/race/202608050411/predictions` | 224 / 147 / 160 | MISS → HIT |
| `/_next/static/…js` | 287 / 123 / 107 | cf: MISS → HIT |
| `/megu-icon.png` | 362 / 152 / 158 | cf: REVALIDATED（毎回オリジンへ確認） |

### 同じ接続を使い回した時（ブラウザの 2 本目以降に近い）

| 対象 | TTFB |
|---|---|
| 静的 js（CF HIT） | 29ms |
| `/races`（Next の事前生成） | 59ms |
| `/api/race-list/20261115`（nginx HIT） | 54ms |
| `/api/v1/auth/status`（Cloud Run・温） | 120〜140ms |

### 段ごとの内訳（ms）

| 段 | ms | 根拠 |
|---|---|---|
| DNS | 5〜16（初回だけ 91） | curl |
| TCP（端末 → CF NRT） | 約 14 | con − dns |
| TLS | curl では 60〜80（累計 85〜105） | tls − con。TLS 1.3 は本来 1 往復（約 14ms）で済むので、差の多くは手元の curl（証明書の検証など）の処理と見られる。**ブラウザで測り直す**（§7 の 6） |
| CF のエッジで返す | 22〜50 | 使い回し 29ms |
| CF → VPS → nginx HIT | 50〜110 | 使い回し 54ms |
| Next の事前生成 HTML | 55〜130 | 使い回し 59ms |
| nginx → Cloud Run（温）の往復 | 約 70〜90 | auth/status 120〜140 − nginx HIT 54。アプリの処理は 0.1〜0.3ms（`login_latency_20261116.md`）。内訳（接続の張り直し・Cloud Run の手前の処理）は §7 の 2 で測る |
| Cloud Run の中の初回の計算 | 180〜2,900（race-quality/day は 9.7 秒） | 今回の MISS、`capacity_and_latency_20261027.md` |
| Cloud Run のコールドスタート | 中央 28 秒（9〜43 秒）、案 A 後のログインは 約 14 秒 | `login_latency_plan_a.md` |

- gzip: 圧縮あり・なしの両方で `race-list` が nginx HIT になることを確かめました（`Vary: Accept-Encoding, Cookie`）。cache-warm の版と食い違う心配はありません。
- HTTP/2 は使われていて、HTTP/3 も `alt-svc: h3=":443"` で有効です。
- stg は Access の 302 で止まるので、オリジンまでは測れていません。
- **計測は Cookie なしの curl の TTFB だけ**です。実際のブラウザ（Cookie あり）の当たりと、表示の完了（LCP）は測れていません（§7 の 3・6）。
- Cloud Run のログからの遅いエンドポイントの上位は取れていません（このマシンの gcloud の認証が root 専用）。§7 の 5 をユーザーにお願いします。

## 3. 改善の候補

期待の ms は §2 の内訳から見込んだ値です。実装の後に、同じ curl とブラウザの計測で測り直します。

### L2a 保守の確認で HTML・画面の行き来を待たせない — 推奨 1

- 今: `frontend/middleware.ts:67-100` は、プロセスあたり 15 秒に 1 回、`/api/maintenance/status` を `no-store` で取りに行き、返るまで最大 1.5 秒待たせる。対象（`middleware.ts:136`）は HTML だけでなく、画面の行き来の RSC・先読みの要求も含む。Cloud Run が冷えていると毎回 1.5 秒で打ち切りになる。
- 案: 前の値ですぐ答え、期限が切れていたら裏で取り直す（待たない）。取り直しの回数は今と同じ（15 秒に 1 回まで）。
- 効果: HTML・RSC の最悪 +1.5 秒が消える。
- 費用: 0 円（Cloud Run への要求の回数は変わらない。温まり方も変わらない）。
- リスク: 保守の開始の切り替えが最大 15 秒遅れる（今も 15 秒の保持があるので、ほぼ同じ）。
  別の注意として、middleware の確認が cache-warm と同じ流量制限の枠を使い、429 になると「保守ではない」と扱われる（`middleware.ts:82-85`）。L2a のついでに、429 を「前の値を使う」にする。

### L11 画面の API の直列をやめる — 推奨 2（レビューで追加）

- 今: `/races` は `scrape-dates` が返ってから `race-list` を取る（`races/page.tsx:176→202`）。会員では Cloud Run の往復が 2 回続く。
- 案: URL の日付か「今日」を先に使って `race-list` を同時に取り、`scrape-dates` の結果と食い違ったときだけ取り直す。ほかの画面も同じ形の直列を洗い出す。あわせて、最初に必ず取る API は HTML に `<link rel="preload" as="fetch" crossorigin>` を入れ、JS の読み込みと並べる。
- 効果: 往復 1 回分（会員 120〜300ms、匿名 50〜60ms）。
- 費用: 0 円（取り直しが起きた時だけ 1 回増える。Cloud Run の起動回数は変わらない）。
- リスク: 日付の推定が外れたときの無駄な 1 回。

### L1 ブラウザに前回の内容を残して先に出す（SWR）を広げる — 推奨 3

- 今: `httpCache.ts` の `persist`／`readStored`（sessionStorage に 30 分）があるのに、使っているのは分析・週の予想の 2 画面だけ。
- 案: レース一覧・レース詳細・予測・カレンダーなど主な画面で、前回の内容をまず描き、裏で取り直して差し替える。取り直し中は小さな「更新中」の印を出す。
- 効く範囲: **同じタブの中**（再読み込み・戻る・画面の行き来）。sessionStorage と区分の記録（`TIER_KEY`）はタブごとなので（`httpCache.ts:34-36,47`）、新しいタブ・ブックマークから開いた時は効かない。タブを越えて効かせる（localStorage にする）のは、共有端末に会員の内容が残るので採らない。
- 効果: 同じタブの中では、Cloud Run の往復を待たずに数十 ms で表示。同じタブで開き直したときは、コールドスタート中（14〜28 秒）でも前回の内容が見える。**Cookie の付いた会員に効く、費用 0 円の案。**
- 費用: 0 円（要求の回数は今と同じ。増やさない）。
- リスク: 古い内容の表示（印で分かるようにする。オッズ・締切のような時間に敏感な値は、取り直しが済むまで薄く出すなど区別する）。sessionStorage の上限（5MB 前後。大きい応答は残さない）。秘密: 利用者のブラウザの中だけで、区分が変わったら・ログアウトで消える既存の仕組みを使う。

### L3 nginx → Cloud Run の接続を使い回す — 推奨 4（VPS での計測が先）

- 今: `proxy_pass` に変数を使うので keepalive が効かない（`nginx-keiba-ui-cloudrun.conf:21`）。会員の要求のたびに TCP と TLS を張り直していると見られる。
- 案: `upstream keiba_cloudrun { server <run.app のホスト>:443 resolve; keepalive 16; }` ＋ `proxy_pass https://keiba_cloudrun`。`resolve` は nginx 1.27.3 以上。古い版なら、名前の解決を起動時に固定し IP の変わり目は reload で拾うか、見送る。
- 効果: VPS も Cloud Run も東京なので、張り直しの分は 10ms 前後の可能性が高い。70〜90ms の多くは Cloud Run の手前の処理かもしれない（推測）。**§7 の 2 の結果で、効果が 20ms 未満なら見送る。**
- 費用: 0 円。
- リスク: run.app の IP が変わったときに古い接続先へ行く（`resolve` で解決）。SNI・共有鍵のヘッダは今と同じ。

### A2 起動の短縮 — 推奨 5（既存の案）

- `login_latency_plan_a.md:38,80` の A2（FastAPI の app の import を遅らせる）。起動 14 秒の内訳は import 約 6 秒＋名簿・束・目録の取り込み 約 8 秒。
- 効果: 冷えた時の待ちが数秒縮む見込み（取り込みは CPU がある間に済ませる必要があるので、縮むのは import の分だけ）。
- 費用: 0 円（起動 1 回の CPU 秒が減る方向）。
- リスク: 遅らせた import の初回の要求が少し遅くなる。案 A の「ログインを先に答える」と食い違わないように確かめる。

### L9 300 秒空いた後の 8 秒待ちを狭める — 推奨 6（要計測）

- 今: 前回の取り込みの成功から 300 秒以上空くと、温まったインスタンスでも最初の要求が目録の取り込みを最大 8 秒待つ（`gcs_mirror.py:749,759-761`、`cloudrun_entry._gate`）。
- 待つのを全部やめると、出る内容の古さに上限が無くなる（静かな夜の後なら数時間前の内容）。そのため全部はやめず、狭める案から選ぶ:
  1. 待つのを、開催日の当日・前後のパス（予測・結果・race-list）に絞る。それ以外は待たずに答える。
  2. `wait_s` を 8 秒 → 2〜3 秒に縮める（目録の取り込みの実測の中央が収まるなら）。
- 効果: 静かな時間の後の最初の表示: 最悪 8 秒 → 数秒以下。
- 費用: 0 円（目録を読む回数は今と同じ）。
- 先にやること: Cloud Run のログで、実際に何秒・何回待っているかを測る（§7 の 5）。

### L6 Cloudflare の Cache Rule（匿名・`/api/` だけ）とアイコンの期限 — 推奨 7

- 案:
  1. Cache Rule の条件は「パスが `/api/` で始まる」かつ「Cookie が無い」かつ「Authorization が無い」。対象は `http_cache.json` の許可リストに入るパスだけ。
  2. **Edge の期限は全体で短く上書きする（例: 60〜300 秒）。** オリジンの `s-maxage` に従わせると、当日・未来のレース（6 時間）や確定済み（30 日）が Cloudflare に長く残り、cache-warm の上書き（VPS の nginx にしか届かない）が反映されない。長く置きたい確定済みのレースは、上書きのたびに Cloudflare の API で URL 単位に purge する仕組みを作ってからにする。
  3. **HTML・RSC には置かない。** Next の HTML は `s-maxage=31536000`・`vary: rsc,…` を返すので、広い規則にすると 1 年残り、保守の切り替えも効かなくなる。
  4. `megu-icon.png` などの `public` の画像の毎回の REVALIDATED は、`next.config.js:59-75` の headers で期限を付けて直す（リポジトリの変更だけ。ダッシュボードは触らない）。
- 効果: 匿名の API 54ms → 約 30ms（使い回し時）。アイコン 約 150ms → CF HIT。
- 費用: 0 円（Cloudflare の無料プラン）。VPS → Cloud Run の要求は減る方向。
- 価値は、匿名の訪問者の数しだい（§7 の 3 の結果で判断）。
- **入れる条件: 本番 app が Access の外にあるのが意図した公開かを、ユーザーが確かめてから（§0）。** 意図しない公開なら、匿名の応答を Cloudflare のエッジに広げるのは逆向き。

### L4 nginx の期限切れを裏で取り直す — 推奨 8

- 今: `proxy_cache_background_update` が無く、上流も `stale-while-revalidate` を付けない。期限が切れたキーの最初の 1 人が Cloud Run を待つ。
- 効く範囲: 日付・レース ID の付くパスは 6 時間〜30 日で、cache-warm が上書きするので、期限切れはまれ。**効くのは calendar（120 秒）・scrape-dates（600 秒）・megu-index-dates・prediction-results など短いものに限られる。**（今回の `scrape-dates` の MISS 1,182ms がこれに当たる）
- 案: `nginx-api-cache.conf` に `proxy_cache_background_update on;` を足し、上流は public の応答に `stale-while-revalidate` を付ける（`http_cache_headers.py`）。あわせて `proxy_cache_lock_timeout 10s`（`nginx-api-cache.conf:22`）がコールドスタート（14〜28 秒）より短く、10 秒後に待ちきれなかった要求が上流へ行って保存されない点も見直す。
- 効果: 対象のパスの期限切れ直後: 最大 1.2 秒 → 約 54ms。
- 費用: 0 円（取り直しの回数は今と同じ）。
- リスク: 期限切れの後の 1 回だけ古い内容が出る。

### L2b 保守のタブの問い合わせを減らす — ×（今の形では）・ユーザーの判断事項

- **判断が要る対立**: 今ある 60 秒おきの問い合わせは CLAUDE.md の「15 分未満のタブの定期問い合わせ」に反している。一方で、直すとコールドスタートが増え、最悪で費用の上限を超えうる。どちらを優先するかは、ユーザーが決める（この計画では決めない）。

- 今: `MaintenanceShell.tsx:139-141` は、開いているタブごとに 60 秒おき（予告中・保守中は 15 秒）に問い合わせる。`/api/maintenance` は never なので、匿名・会員とも毎回 Cloud Run に届く。**CLAUDE.md の「15 分未満のタブの定期問い合わせを足さない」に、今ある処理が反している。**
- ただし、この問い合わせはタブを開いている間 Cloud Run を温め、L9 の 300 秒の空きも防いでいる。外すとコールドスタートが増える。
  - 最悪の見積もり: 起動 1 回 約 100 vCPU 秒（`cost_ledger.md:93`）× 96 回/日 × 30 日 ＝ 28.8 万 vCPU 秒 × 4.1 円/千 ≒ **+約 700 円/月**。余裕（370〜570 円）を超えうる。
  - 体感でも、タブを開いたままの会員が 14〜28 秒待つ回数が増える。
- 案（L6・A2・L9 の後で）: 初回に取った状態の応答（残り秒）を基に手元で数え、予告の時間帯の前後とフォーカスを得たときだけ問い合わせる。予定は環境変数と `maintenance.json`（30 秒ごとに読み直し）で変わるので、フロントに予定を埋め込まない。
- 判断: Cloud Run のログで起動の回数（今の平日 1〜4 回/日・レース日 12 回/日）を測り、最悪でも上限に収まる見込みが立つまで入れない。

### L7 会員の応答を区分ごとに共有キャッシュ — 推奨 10（別の段階）

- 既存の計画 `docs/todos/vps_cache_30d_plan.md` の 3-4 と同じ（作り直さない）。
- 案: ログイン時に API が「区分の署名つき Cookie」を発行し、nginx（njs）が署名を確かめて鍵に区分を入れる。開発者・管理は今どおり素通り。
- 必ずセットにすること: 上流の `Vary: Cookie` のままでは Cookie の値ごとに保存物が分かれて当たらないので、区分で分ける形に Vary を出し分ける。セキュリティのレビューを通す。
- 効果: 会員の API 120〜300ms → 約 54ms。冷えていても前の保存を出せる。
- 費用: 0 円（Cloud Run への要求が減る）。
- リスク: **高い**。区分の取り違えは、有料会員の内容が匿名に出る漏れになる。別の計画として行う。

### L8 重い計算の応答の事前生成 — 推奨 11（△）

- 対象: `/api/v1/betting/race-values`（初回 1.9〜2.9 秒）、`/api/race-quality/day`（9.7 秒）、`/api/pedigree-race-stats/query`（約 80 秒で 504）。
- 案: 既存の Cloud Run Jobs の実行に相乗りして作り、GCS に置く。API（同じリージョン）が読む。VPS・ホストからは読まない。
- 費用: 1 日 1 回の実行に相乗りで **約 40 円/月**（式は §5。無料枠は 11 月は 11/07〜08 に尽きているので、無料枠を当てにしない）。新しく毎時の実行を足すのは、余裕を削るので見送る。
- リスク: Jobs が止まっている期間は効かない（台帳では Jobs は停止中の扱い。動いている実行を先に確かめる）。

### L5 Cloudflare の 0-RTT（任意）

- HTTP/3 はすでに有効（`alt-svc: h3=":443"`）。残るのは 0-RTT だけで、ダッシュボードで確かめる。
- 効果: 再訪時に 1 往復（約 14ms）。費用 0 円。リスク: 再送（Cloudflare は GET などに限る）。

### L10 ETag／304（参考）

- 0 円。ただし 304 でも Cloud Run は起きるので、減るのは送る量だけ。優先しない。

### 検討して採らない・効かない案

| 案 | 判定 | 理由 |
|---|---|---|
| Cloudflare Argo | × | 有料。Cloudflare の費用も上限に入る |
| Cloudflare Tiered Cache | 効果薄 | 利用者がほぼ NRT だけ |
| HTTP/2 の多重化・preconnect | 済・不要 | HTTP/2 は使われている。同じオリジンなので preconnect は要らない |
| ブラウザのキャッシュを localStorage でタブを越えて残す | 採らない | 共有端末に会員の内容が残る |

## 4. ×（採らない案）

| 案 | 月額の見積もり | 理由 |
|---|---|---|
| 最小インスタンス 1 | 2Gi で 約 2,900 円（`cloud_run_api.md:71`）。本番は今 4Gi なので 約 4,900 円。開催日の 9 時間 × 8 日だけでも 290〜490 円＋Scheduler 30 円 | 上限超え。「最小インスタンス 0」に反する |
| VPS から GCS を直接読む | 1 万要求 × 100KB ≒ 29GiB/月 × 18.86 円 ≒ 550 円＋Class B 19 円。束 0.4GiB を 15 分ごとに読めば 約 2.2 万円 | 「GCP の外から GCS を繰り返し読まない」に反する（2026-10 の超過と同じ構図） |
| cache-warm の周期を上げる | 30 秒にすると合図の読みが +4.3 万回/月。対象を倍にすると 2,899 件 × 0.5 vCPU 秒 × 30 ≒ 4.3 万 vCPU 秒 → +180 円 | 決まり 3・6 の趣旨に反する。本番に API の安定化（段 2〜4）が入るまでは、朝の温めが OOM の引き金にもなる |
| 15 分未満の ping で Cloud Run を温め続ける | 常時起動に近づき、最悪で最小インスタンス 1 と同じ水準 | 決まりで禁止 |
| ホスト・VPS での事前生成の周回 | 2026-10 は 1 日 約 50GiB の転送料 | 決まり 3 で禁止 |

## 5. 費用の見積もり（最悪の回数 × 単価）

単価は `cost_ledger.md:57-63` の実単価: Cloud Run CPU 4.1 円/千 vCPU 秒、Cloud Run の外への送信 20 円/GiB、GCS の外への転送 18.86 円/GiB、Class A 0.8 円/千回、Class B 0.06 円/千回。

- 今の余裕: 12 月の見込み 1,400〜1,480 円（`cost_ledger.md:132-133,176`）＋本番 4Gi の分（ふだん +30〜60 円、最悪 +150 円。`api_memory_20261117.md:27`）→ 約 1,430〜1,630 円。**余裕は 約 370〜570 円。**
- L1・L2a・L3・L4・L5・L6・L7・L9・L10・L11・A2: GCS の操作・転送・Cloud Run の起動回数を増やさないので **0 円**。
  - L2a は取り直しの回数が今と同じ（15 秒に 1 回まで）。温まり方も変わらない。
  - L11 は、日付の推定が外れた時だけ要求が 1 回増える（温まったインスタンスへの要求で、100ms 単位の課金。月に数円未満）。
  - L6・L7 は Cloud Run への要求を減らす方向。A2 は起動 1 回の CPU 秒を減らす方向。
- **L2b: 最悪 +約 700 円**（起動 100 vCPU 秒 × 96 回/日 × 30 日 × 4.1 円/千）。余裕を超えうるので、今の形では ×。
- L8: 1 日 1 回の既存の実行に相乗り: 300 vCPU 秒 × 30 × 2.9 円/千 ≒ 26 円 ＋ 書き込み 500 件 × 30 × 0.8 円/千 ≒ 12 円 → **約 40 円**。
- VPS のディスク: nginx の置き場は今の上限（空きの 30%、1〜20GB）の中。L7 で会員区分が 3 通り増えても 数百 MB。
- Cloudflare: 無料プランの範囲（L5・L6）で 0 円。

## 6. 「ms レベル」に届くもの・届かないもの

| 対象 | 今 | 推奨の後の見込み | 届くか |
|---|---|---|---|
| 静的ファイル（`_next/static`） | 使い回し 29ms・新しい接続 107〜132ms（curl） | 同じ | 届く（新しい接続の初回は TLS の分。ブラウザでの値は §7 の 6 で測る） |
| 画面の HTML・RSC（事前生成の殻） | 59ms（最悪 +1.5 秒） | 59ms。最悪の +1.5 秒が消える（L2a） | 届く |
| 匿名の API（nginx HIT） | 54ms、短い期限のパスの期限切れ直後は 最大 1.2 秒 | 30〜54ms（L4・L6） | 届く |
| 会員の API（Cloud Run・温） | 120〜300ms。直列の画面はその 2 倍 | 直列が消える（L11）。1 回は 110〜130ms 程度（L3。計測しだい）。L7 の後は 約 54ms。同じタブの中は L1 で数十 ms | 同じタブの中の表示は L1 で届く。初めて開く画面は L7 まで届かない |
| キャッシュに無い重い計算 | 0.2〜10 秒 | 一部は L8 で数百 ms | 届かない（計算そのものの時間） |
| ログイン | 約 220ms（会員は 0.5〜0.6 秒の見込み） | 同じ | 届かない（パスワードの照合 124〜137ms は安全のための意図した時間。案 F は採らない） |
| コールドスタート | 14〜28 秒 | A2 で数秒縮む。同じタブなら L1 で前回の内容が出る | 届かない（最小インスタンス 1 が × のため） |
| 静かな時間の後の最初の要求 | 最大 8 秒の待ち | L9 で数秒以下 | 届かない（古さとの兼ね合い） |

## 7. ユーザーにお願いしたい確認（自分では VPS・本番に入っていません）

1. VPS の nginx の版と、API の上流の向き:
   ```
   nginx -v
   sudo bash <VPS 上のリポジトリ>/scripts/ops/vps/switch_api_upstream.sh status
   ```
2. VPS → Cloud Run の、新しい接続と使い回しの時間（L3 の効果の見積もり）:
   ```
   H=<keiba-api の run.app のホスト>
   curl -s -o /dev/null -o /dev/null -w 'con=%{time_connect} tls=%{time_appconnect} ttfb=%{time_starttransfer}\n' https://$H/api/health https://$H/api/health
   ```
   1 行目が新しい接続、2 行目が使い回しです。差が 20ms 未満なら L3 は見送ります。
3. 実際の利用者の nginx キャッシュの当たり（L4・L6 の価値の判断）:
   - いちばん簡単なのは、ふだん使うブラウザ（ログインあり／なし）で開発者ツールを開き、`/api/` の応答の `X-Cache-Status` を数件見ることです（Cookie があれば `BYPASS` か空になるはず）。
   - 匿名の訪問者の数は、nginx の access.log の行数を、`/api/auth/status` の応答の会員区分ごとに分けるのが確実ですが、ログの書式に `$upstream_cache_status` と Cookie の有無が無いため、今のログからは出せません。書式を足すかは、実装の段で相談します。
4. cache-warm の 429（温めの要求は 127.0.0.1 から来るので、120 回/分・burst 60 の制限を 1 つの IP として使う。1 周 200 件を 0.05 秒おきに投げると当たりうる。middleware の保守の確認も同じ枠）:
   ```
   sudo journalctl -u keiba-cache-warm --since today | grep -c ' 429'
   ```
5. Cloud Run のログ（このマシンでは gcloud の認証が root 専用で取れなかった）:
   ```
   # 遅いエンドポイントの上位
   gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="keiba-api" AND httpRequest.latency>"0.5s"' --project horse-racing-466415 --freshness=7d --limit=3000 --format=json
   # 起動の回数と段の時間（L2b・A2・L9 の判断）
   gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="keiba-api" AND textPayload:"起動の段"' --project horse-racing-466415 --freshness=7d --limit=500 --format='value(timestamp,textPayload)'
   ```
6. ブラウザでの計測: 開発者ツールの Network で `/races` を開き直し、HTML の TTFB・TLS の時間（Timing の Initial connection / SSL）と、表示の完了（Lighthouse か Performance の LCP）を 1〜2 回見る。curl の TLS 60〜80ms がブラウザでも同じかを確かめる。

## 8. ついでに見つけた点（直すのは別の作業）

- docs の「app は Cloudflare Access の内側」が古い（`go_live.md` の表、`tiered_access.md:38`、`cdn_cache.md:28`）。本番 app は Access の外で、stg だけが内側。**公開の前提（`public_fields.json` の `mode: private`、`KEIBA_TEST_USER` を off にする件、anti_scraping の enforce）が守られているかを、別に確かめる必要があります。**
- `purge_api_cache.sh` の説明は鍵を「ポート＋ホスト＋URL」としているが、実際の鍵は `$request_uri`。`--match '^8081'` は何も消さない。`docs/todos/vps_cache_30d_plan.md:16` の鍵の記述も古い。
- Flask のセッション Cookie に `Secure` が無い（`flask_auth.py:105-106`。FastAPI 側の `auth.py:266` は付けている）。
- 本番の `/api/v1/auth/status` に `Server-Timing` が付いていない（本番は応急の版 `keiba-api-mem4g-1117` のため、と見られる）。
- `cdn_cache.md:12-17` の TTL の表が `config/http_cache.json` と合っていない。
- 保守のタブの 60 秒おきの問い合わせが、CLAUDE.md の決まりに反している（L2b。費用の兼ね合いで保留）。

## 9. 別のエージェントのレビュー（指摘と対応）

草稿を、作った本人とは別のエージェント（general-purpose）に、計画の本文・コード・計測の生の数字を材料として渡し、根拠・代わりの案・費用・リスク・見落としの観点で見てもらった。

| 指摘 | 対応 |
|---|---|
| HTTP/3 はすでに有効（`alt-svc: h3`）。0-RTT の効果は 15〜40ms ではなく 約 14ms | 直した。L5 を 0-RTT だけの任意の案に下げた |
| 「TLS 85〜105ms は物理的な下限」は誤り。TLS 1.3 は 1 往復で、差の多くは手元の curl の処理 | 直した。推測として書き、ブラウザで測り直す（§7 の 6） |
| L2 のタブの問い合わせを外すと、温めが無くなりコールドスタートが増える。最悪 +約 700 円で余裕を超える | 直した。L2 を L2a（待たない・0 円）と L2b（問い合わせを減らす・最悪 +700 円・保留）に分けた |
| L2 の後に L9 の 8 秒待ちが増えるのに、順番が逆 | 直した。L2b は L9・A2 の後で、起動の回数を測ってから判断 |
| L9 の「最大 5 分ほど古い」は誤り。古さに上限が無い | 直した。全部はやめず、開催日のパスに絞る・`wait_s` を縮める案に替えた |
| L4 の前提（race-list・v1/races は 30〜120 秒で切れる）がコードと違う | 直した。効くのは calendar・scrape-dates などに限ると書き、順番を下げた |
| L6 で Cloudflare に置く期限が抜けている。6 時間・30 日が残り、上書きが届かない | 直した。Edge の期限を全体で短く上書きすることを必須にした |
| L6 の条件に Authorization を足し、`/api/` だけに限る。HTML は `s-maxage=31536000` なので置かない | 直した |
| 余裕の数字が §0 と §5 で食い違う | 直した（370〜570 円に統一） |
| 「匿名のキャッシュは今すぐ効く」は Cookie なしの curl でしか確かめていない。§7 の 3 のコマンドは当たりを数えていない | 直した。価値は匿名の訪問者の数しだいと書き、§7 の 3 をブラウザの開発者ツールで見る形に替えた |
| L1 は同じタブの中でしか効かない | 直した。効く範囲を明記し、localStorage にしない理由を書いた |
| L2 の効果は RSC・先読みにも及ぶ。匿名のタブの問い合わせも Cloud Run に届く | 直した |
| L2 の「手元の時計で判断」は壊れやすい（環境変数と JSON で変わる） | 直した。初回の応答の残り秒を基に数える形にした（L2b） |
| middleware の確認と cache-warm が同じ流量制限の枠を使い、429 で「保守ではない」扱いになる | 直した。L2a で 429 を「前の値を使う」にする |
| 見落とし: A2（起動の短縮）、画面の API の直列、preload、ブラウザの LCP での計測 | 直した。A2・L11 を推奨に加え、§7 の 6 を足した |
| アイコンの REVALIDATED は `next.config.js` の headers で直せる | 直した（L6 の 4） |
| `proxy_cache_lock_timeout 10s` がコールドスタートより短い | 直した（L4 で一緒に見直す） |
| L3 の効果「−20〜60ms」は根拠が弱い（東京どうしなら 10ms 前後） | 直した。効果を下げ、20ms 未満なら見送ると書いた |
| Argo・Tiered Cache・HTTP/2・preconnect の判定を書く | 直した（§3 の「検討して採らない・効かない案」） |
| L8 の「無料枠の中なら 約 12 円」は使えない（無料枠は 11/07〜08 に尽きた） | 直した（約 40 円） |

## 10. 進め方（承認の後）

1. L2a・L11・L1（frontend だけ。stg で確かめる）。
2. §7 の 1・2 の結果で L3 を決める。A2 は §7 の 5 の起動の段の時間を見てから。
3. L9（Cloud Run のログで待ちの実測を見てから。範囲を狭める案）。
4. §7 の 3 の結果で、L6（Cloudflare のダッシュボードはユーザーの操作。アイコンの期限はリポジトリで）と L4（nginx のテンプレート。stg の待ち受けで先に入れる）の価値を判断する。
5. L2b は今の形では ×。起動の回数を測り、最悪でも上限に収まる見込みが立ち、ユーザーが決まりと費用のどちらを優先するか決めてから見直す。L7 は別の計画書とセキュリティのレビュー。L8 は Jobs の実行状況を確かめてから。
6. 各段の後に §2 と同じ curl とブラウザの計測で測り直し、1〜2 日後に課金エクスポートで費用が増えていないことを確かめる。

## 11. 統括役・ユーザーの判断（2026-11-18）

- 本番 app が Access の外にあるのは**意図した公開**（ユーザーの回答）。L6 は進めてよい。条件は「Cookie・Authorization の無い `/api/` の許可リストだけ、Edge の期限 60〜300 秒、HTML は置かない」。会員だけに出す値が匿名の応答に混じらないことを、テストで確かめる。
- 保守のタブの 60 秒おきの問い合わせ（L2b）は**今のまま**。費用を優先する決まりの例外として、`cost_ledger.md` の 4 章「審査: 表示の遅延の計画」に理由を記録した。L2a は進める。
- 実装の順: L2a → L11 → L1 → L3（VPS で計測して 20ms 未満なら見送り）→ A2 → L9 → L6 → L4。L7 は別の計画とセキュリティのレビュー。L8 は後で相談。
- 時期: 11/20（金）に本番の昇格（master）があるので、作業は `feature/latency` で行い、develop へは 11/22（日）の夜以降に入れる。小さく独立して安全なもの（L2a など）は、critic と CI を通し、統括役に断ってから develop に入れてよい。
- §7 のユーザーへの確認は、週末明けに統括役から頼む。

## 12. 実装の状況（2026-11-18、11/24 に develop へ載せ直し）

| 案 | 状態 | 置き場 | 要点 |
|---|---|---|---|
| L2a | develop に入った（統括役の了承）・CI 緑 | 2fe8b86a | 手元の確かめ（next dev ＋偽の保守 API）: 状態の API が 10 秒遅れたとき、画面の TTFB は変更前 2.4〜3.0 秒 → 変更後 0.3〜1.1 秒。保守の目印で 2 本目から 503、保守中の 429 でも 503 のまま、通常の 429 は待たずに 200 |
| L11 | develop（PR #3、11/24） | 3bf53408 | `/races` で開催日の一覧と race-list を並べて取る。外れる日（祝日の月曜開催とその翌日）は 1 本むだになるだけ |
| L1 | develop（PR #3、11/24） | eddd8230 ＋ 11/24 の載せ直し | `/races` の一覧・`/race/[id]` の本体と予測で、このタブに残した前回の内容を先に出す。最新が取れないときは「前回の内容を表示しています」と出し、自動の取り直しを続ける |
| L6 | develop（PR #3、11/24。リポジトリ側） | 00a7d49f ＋ 11/24 の載せ直し | 共有キャッシュに置ける応答に会員の値が混じらないテスト。public の応答に `Cloudflare-CDN-Cache-Control` を付ける。nginx に 5 分以上置き、開催日が前日以前の応答（日付が分からないときは確定済みの 30 日のものだけ）だけ `public, max-age=300`。それ以外（当日・未来のレースと一覧、保守と認証の状態）は `no-store`（11/24 の載せ直しで直した。下の「11/24 の載せ直し」）。アイコン・ブランドの画像に 1 日の期限。Cloudflare の規則は下の手順でユーザーが設定する |
| L4 | **見送り** → edge-fix F（2026-11-23）で別の形で入った | 3ca8fd0a | 期限切れの裏の取り直しは、アプリの `stale-while-revalidate=min(60, s-maxage)` と nginx の `background_update`（古さの上限つき）、`/api/` の `keepalive_timeout 0` で入った（docs/operations/edge_fix_20261123.md の F）。下の見送りの理由のうち、古さの上限と keepalive の待ちはそこで解かれた |
| L3・A2・L9 | 未着手 | — | §7 の VPS の計測と Cloud Run のログ（週明けに統括役からユーザーに依頼）を見てから |

### L4 を見送った理由（critic のレビュー、2026-11-18）

- `proxy_cache_background_update` で返す古い応答は、保存したときの `Cache-Control`（`public, max-age`・`s-maxage`）のまま出る。nginx は `Age` を付けないので、Cloudflare とブラウザがその古い応答をさらに満期まで使い、古さが nginx の期限＋Cloudflare の期限まで延びる。
- 裏の取り直しが終わるまで、Cloudflare からの keepalive の接続の次の要求が待たされる（nginx の既知の挙動）。冷えた Cloud Run では最大 約 28 秒。
- stg（読むだけ）では本番の古い保存を返し、裏の取り直し（rstg）は保存されずに捨てられるので、検証中のコードの応答が見えなくなる。
- `proxy_cache_lock_timeout` を延ばしても、`proxy_cache_lock_age`（既定 5 秒）ごとに上流へ 1 本ずつ抜ける。両方を延ばすと、上流が詰まったときの待ちが最大 30 秒＋読み取りの期限に延びる。
- 代わりに、期限切れの最初の 1 人の待ちは、L1（前回の内容を先に出す）と L6（Cloudflare に 5 分置く）で、利用者からは見えにくくなる。

### L1 で変わる既存の挙動

- 区分（匿名・会員・開発者）の記録は、**このページの中で認証の状態を取って確かめた後だけ**使う。再読み込みの直後・直接開いた直後は、認証の状態が返るまで前回の内容を出さない。
  - 理由: 他のタブでのログアウト・セッション切れの後に再読み込みすると、記録が会員のまま残っていて、会員向けの内容が先に出ていた（既存の分析・週の予想の画面も同じ）。安全側に直した。
  - そのため、前回の内容がすぐ出るのは、主に**同じタブの中の画面の行き来**（クライアント側の遷移）。
- 要求を出したときと返ったときで区分が変わったら（ログイン・ログアウト）、その応答は残さない。
- あふれたときは全部を捨てず、古い半分を消す。1 件 40 万文字を超える応答は残さない（匿名のレース本体は最大 約 270KB。2026-11-18 に 4 レースを実測）。

### 11/24 の載せ直し（11/23 の edge-fix・edge-front の後）

- L1: edge-front（61a91936）は、「未ログイン」をこのタブに 5 分残し、その間は認証の状態をネットワークに聞かずに返す（近道）。近道では区分の確認が立たず、匿名の利用者は再読み込みの後 5 分間、前回の内容が出なかった（週の予想・分析も同じ）。近道でも区分を「匿名」と確かめたことにした。ただし読むだけにし、近道の間に返った応答は残さない（別のタブでログインした後は、Cookie が会員なのに区分が匿名のままになりうるため）。ネットワークで確かめ直した後は残す。
- L6: edge-fix は、当日のレースを nginx に 30 秒＋swr 60 秒置き、古さの上限を 60 秒にした。Cloudflare に一律 5 分置くと、この上限が 5 分を超える。そこで、s-maxage が `cdn_max_age`（300 秒）未満の応答は `no-store` にした。当日・未来と分かるレースと一覧（当日の race-list は 6 時間、速報の印が付いた当日のレースも長く置く）も `no-store` にした。当日は cache-warm が nginx を取り直し続けるので、Cloudflare の分だけ古くしない。
- L6: 保守の状態と Cookie の無い auth/status は、edge-fix で public（`max-age=0, s-maxage=…`）になった。見出しが無いと Cloudflare の規則（オリジンに従う）が s-maxage で置くので、`max-age=0` の応答も `no-store` にした。
- L6 の必要: edge_fix_20261123.md の「残り」の 3 で、HIT の要求に利用者の側で 5〜9 秒の例があり、遅れは nginx より手前（Cloudflare とトンネル）で起きていた。そこに効くのは Cloudflare のエッジに置く L6 だけ。
- 11/23 の nginx の消去（B の purge）は、下の「規則を入れる前に」の代わりにならない。それ以降の保存物にも `Cloudflare-CDN-Cache-Control` が無いため。
- VPS のフォルダから返す応答（edge-cache の `/srv/keiba-edge`、`templates/nginx-edge-files.conf`）は Cloud Run の見出しを通らない。そこにも同じ規則（長く置くものは 5 分、確定前は `no-store`）の `Cloudflare-CDN-Cache-Control` を付ける必要がある。無いと `max-age=30` が Cloudflare に効く。担当 edge-cache と合わせる（統括役経由）。

### L6: Cloudflare の Cache Rule の設定（ユーザーの操作。ダッシュボード）

Edge TTL を「オリジンを上書き」にすると、オリジンが `private`・`no-store` を返す匿名の応答（保守の状態・ログインの状態など）まで置かれうる。そこで Cloudflare はオリジンの指示に従わせ、Cloudflare だけに効く短い期限はオリジンが `Cloudflare-CDN-Cache-Control` で付ける（`config/http_cache.json` の `cdn_max_age`、300 秒。nginx・ブラウザはこのヘッダを見ない）。

**規則を入れる前に（必須。順番: L6 が本番の Cloud Run に入る → nginx の全消去 → 規則。開催日の 07:30〜17:00 は避ける）**: この変更（`Cloudflare-CDN-Cache-Control`）が本番の API に入った後、VPS の nginx の API の保存を一度全部消す（`sudo bash scripts/ops/vps/purge_api_cache.sh` の全消去）。
- 理由: それより前に nginx が保存した応答にはこのヘッダが無い（確定済みは最大 30 日残る）。そのまま規則を入れると、nginx の HIT で返る古い保存の `s-maxage`（最大 2,592,000 秒）が Cloudflare に効き、cache-warm の上書きが届かないまま最大 30 日残る。
- 費用: 消した直後は、匿名の要求が URL ごとに 1 回 Cloud Run に届く（その後は nginx に保存される）。一度きりで、GCS の操作は Cloud Run の中なので転送料は 0 円。Cloud Run の秒は、ふだんの開催日の 1 日分の匿名の要求ほど。

Caching → Cache Rules → Create rule:

1. 条件（Custom filter expression）:
   ```
   (http.host eq "app.megu-keiba.com" and starts_with(http.request.uri.path, "/api/") and not len(http.request.headers["cookie"]) > 0 and not len(http.request.headers["authorization"]) > 0)
   ```
   - 対象を `/api/` に限る。HTML・RSC は入れない（Next の HTML は `s-maxage=31536000` を返すので、入れると 1 年残り、保守の切り替えも効かなくなる）。
2. Cache eligibility: Eligible for cache。
3. Edge TTL: **Use cache-control header if present, bypass cache if not**（オリジンの指示に従う。`private`・`no-store` は置かれない）。
4. Browser TTL: Respect origin（ブラウザの `max-age` は 30 秒のまま）。

確かめ方:
- `curl -sI https://app.megu-keiba.com/api/race-list/<確定済みの日>` を 2 回続けて打ち、2 回目が `cf-cache-status: HIT` で `age` が 300 以下。
- Cookie を付けた要求（`-H 'Cookie: a=b'`）は `DYNAMIC` か `BYPASS` のまま。
- `/api/maintenance/status`・`/api/v1/auth/status`・当日と未来のレース・一覧は `HIT` にならない（`cloudflare-cdn-cache-control: no-store`）。
- 前から保存されていそうな確定済みの古い日（例: 1 か月前の開催日）でも、2 回目の `age` が 300 以下になること（全消去が効いているか。新しく保存された URL だけ見ても分からない）。
- Cloudflare が `Cloudflare-CDN-Cache-Control` を優先しない場合（`age` が 300 を超える）は、規則を外して知らせてほしい（6 時間〜30 日の `s-maxage` が Cloudflare に効いてしまうため）。
- 注意: Cloud Run が落ちている間に nginx が返す古い保存（`proxy_cache_use_stale`）にもこのヘッダが付いているので、Cloudflare はそれをさらに最大 300 秒置く。会員の値の漏れではなく、古い表示が最大 5 分延びるだけ。

会員の値が混じらないことは `tests/api/test_shared_cache_tiers.py` が確かめている。
- `public` は匿名の要求にだけ付く。
- その中身に会員限定の項目が無い。
- どの会員区分にも、セッション以外の Cookie や Authorization が付いた要求にも、`public` と Cloudflare 用の期限を付けない。
