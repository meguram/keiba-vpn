# VPS キャッシュを全エンドポイント 30 日保持にする検討（2026-11-02）

依頼: GCP の費用を最小にするため、基本的にすべてのエンドポイントで 30 日分を VPS のローカルに置く。
その前に「30 日置くと問題のあるエンドポイント」を分類し、持ち方を決める。データ取得（作成）時に自動でキャッシュへ入る仕組みが望ましい。
stg と prod は同じエンドポイントを参照し、キャッシュのディレクトリを共有する。

---

## 1. 今の仕組み（コードから確認）

| 部品 | 中身 |
|---|---|
| 置き場 | VPS の nginx `proxy_cache`。`/var/cache/nginx/keiba_api`、上限 `max_size=1g`、`inactive=35d`（`scripts/ops/vps/templates/nginx-zones.conf`） |
| 何を置くか | API が `Cache-Control: public, s-maxage` を付けた応答だけ。付けるのは `config/http_cache.json` の許可リストに入っていて、**Cookie・Authorization のない匿名の要求**だけ（`src/api/http_cache_headers.py`） |
| 保持期間 | 通常は 30〜600 秒。**開催から 3 日以上たった確定済みのレース**（`/api/race-list/`・`/api/race/`・`/api/v1/races/`）だけ 30 日 |
| 更新時の上書き | データの保存時（`HybridStorage` → `src/utils/cache_marker.py`）に GCS の `ops/cache_dirty/<日>/<race_id か 開催日 か _ALL>` に目印を置く。VPS の `keiba-cache-warm`（60 秒おき、`scripts/ops/vps/cache_warm.py`）が目印を一覧して、該当パスを `X-Keiba-Refresh` 付きで取り直して上書きする（空白の時間なし） |
| 鍵 | `$server_port$host$request_uri`。**stg と prod はホストが違うので別々に保存**（置き場所は同じ） |
| 会員の要求 | **キャッシュを読まない・保存しない**（Cookie があると素通り）。会員が見る予測・出馬表は毎回 Cloud Run に届く |
| stg の上流 | stg は Cloud Run の `rstg`（stg に入れた新しい API コード）、prod は本番のリビジョン |

---

## 2. エンドポイントの分類

画面（`frontend/`）から呼ばれる GET を対象にした。「上書き」は、データが変わったときに取り直してキャッシュを置き換えること。

### A. 30 日置いてよい（確定後は変わらない）

| エンドポイント | 変わるタイミング | 補足 |
|---|---|---|
| `/api/race/{id}/result` | 確定後は変わらない（払戻の訂正はまれ） | 確定前は B |
| 過去レースの `/api/race/{id}`・`/predictions`・`/tracking-difficulty`、`/api/v1/races/{id}/analysis`・`/megu-index-predicted` | 確定後は変わらない。**モデルの入れ替え時だけ**全体が変わる | モデル入れ替えは `_ALL` の目印か `purge_api_cache.sh` |
| 過去日の `/api/race-list/{日付}`、`/api/race-quality/day`・`/race`（過去） | 確定後は変わらない | |
| `/api/archive/all-predictions` | 過去分は変わらない | 範囲指定のクエリごとに別の鍵になる |

### B. 毎日〜毎週変わるが、「作成時に上書き」すれば 30 日置いてよい

| エンドポイント | 変わるタイミング | 上書きのきっかけ（目印のキー） | 今の対応 |
|---|---|---|---|
| 未来・当日の `/api/race/{id}`（出馬表） | 出馬表の発表、枠順確定、当日の馬体重・取消 | race_id | ✅ 目印あり |
| 未来・当日の `/api/race/{id}/predictions`・`/tracking-difficulty`、`/api/v1/races/{id}/*` | 前日・当日の再予測、オッズ・馬体重の反映 | race_id | ✅ 目印あり |
| `/api/race-list/{日付}`（未来・当日） | 出馬表・結果の取り込み | 開催日 | ✅ 目印あり |
| `/api/race-calendar`、`/api/scrape-dates`、`/api/megu-index-dates` | 開催日の追加・取り込み | 開催日 / `_ALL` | ✅（`GLOBAL_PATHS`） |
| `/api/horse/{id}/detail`、`/api/v1/horse/{id}/growth-curve`・`/megu-index-history` | **その馬が出走するたび**（週 1 回程度）、プロフィールの更新 | **horse_id**（今は無い） | ❌ 目印が無い。今は 120 秒 |
| `/api/prediction-results`（`/days`・`/news`・`/bets-summary`・`/bets-by-level`） | 開催日の結果確定ごと | 開催日 / `_ALL` | ❌ 取り直し対象に入っていない |
| `/api/race-quality/day`（当日） | 当日の結果取り込み | 開催日 | ❌ 取り直し対象に入っていない |
| `/api/race/{race_id}/entry/{horse_id}/card` | 出馬表・馬の更新 | race_id | ❌ 取り直し対象に入っていない |

### C. 30 日置くと問題がある（置かない、または短く）

| エンドポイント | 理由 | 持ち方 |
|---|---|---|
| `/api/v1/betting/race-values` | **オッズ連動で当日は数分おきに変わる**。目印での上書きが追いつかない | 当日は短い TTL（今の 30 秒）。確定後は A |
| `/api/v1/auth/*`、`/api/v1/admin/*`、`/api/maintenance/status`、`/api/health`、`/api/_hp/probe`、`/api/v1/billing/*` | 個人・状態・認証 | 置かない（今も `never`） |
| `/api/v1/stallion-notes/custom`・`/overrides` | 利用者が編集する個人データ | 置かない |
| `/api/v1/betting/optimize`（POST） | 計算要求 | 置かない |
| `/api/v1/data-analysis/query` | 任意のクエリで鍵が無限に増える | 置くなら上限つき・`_ALL` で消去。優先度は低い |

---

## 3. 方針（案）

### 3-1. 保持期間: B は「上書きつきで 30 日」、C は今のまま

- `config/http_cache.json` の `s_maxage` を、A と B は 30 日にする（今の `settled` を、日付によらず全レース・全日付に広げる）。
- 「古いまま出る」事故を防ぐのは、保持期間を短くすることではなく、**作成時の上書き**で担保する。
- 当日・未来のものは、上書き漏れがあっても被害が小さいよう、念のための上限（例: 当日は 6 時間）を残す案もある（3-5 の「要相談」）。

### 3-2. 作成時に自動でキャッシュに入る（上書き）仕組みを広げる

今の「保存 → 目印 → VPS が取り直す」をそのまま使い、足りない種類を足す。

| 追加 | 内容 |
|---|---|
| 目印のキーに horse_id | `cache_marker` の正規表現に馬の ID（10 桁）を足す。出走結果を取り込んだとき、その出走馬の horse_id の目印を置く（レース 1 つで十数頭） |
| `cache_warm.py` の取り直し対象 | horse_id → `/api/horse/{id}/detail`、`/api/v1/horse/{id}/growth-curve`、`/megu-index-history`。開催日 → `/api/prediction-results*`、`/api/race-quality/day`。race_id → `entry/{horse_id}/card`（出走馬ぶん） |
| 「消す」ではなく「取り直して上書き」 | 依頼にあった「取得のタイミングでキャッシュから消す」は、消した直後の要求が Cloud Run に流れるので、今と同じ「取り直して上書き」（空白なし）にする |

### 3-3. stg と prod で同じキャッシュを使う

- 鍵からホストを外して `$request_uri`（+ 3-4 の会員区分）にし、stg と prod で同じ保存物を使う（ディレクトリはすでに共有）。
- **注意（安全上いちばん大事）**: stg は新しい API コード（`rstg`）を向いているので、stg が書いた応答を共有すると、**検証中のコードの応答が本番の利用者に出る**。そこで次のどちらかにする。
  - **案 1（推奨）: 書くのは prod だけ。** stg は共有キャッシュを読むだけで、無いときは `rstg` から取るが保存しない（stg のサイトに `proxy_no_cache 1`）。API の変更を stg で確かめたいときは、開発者のログイン（Cookie あり）で見れば素通りして `rstg` の応答が見える。
  - **案 2: stg も本番の API を向ける（ご依頼の「同じエンドポイント」をそのまま実現）。** キャッシュは完全に共有できるが、stg で新しい API を確かめる場所がなくなり、`deploy-stg.yml` の `rstg` が意味を失う（反映の手順の変更が必要）。

### 3-4. 会員の応答もキャッシュする（効果は大きいが、段階を分ける）

- 今は会員の要求は全部 Cloud Run に届く。会員が増えるほど、匿名だけのキャッシュでは費用が下がらない。
- 応答は会員区分（匿名 / 会員 / 開発者）で 3 通りしかないので、**区分ごとに鍵を分ければ**キャッシュできる。ただし区分を nginx が信用できる形で知る必要がある（Cookie の値をそのまま鍵にすると、人数ぶん別の保存物になって効かない）。
- 案: API がログイン時に「区分の署名つき Cookie」を発行し、nginx（njs）が署名を確かめて鍵に入れる。開発者・管理系は今どおり素通り。
- セキュリティに関わるので、3-1〜3-3 の後に別の段階で行う。

### 3-5. ディスク容量の見積もり

- レース関連: 月約 300 レース × 約 6 本 × 数十 KB ≒ 100MB 前後。
- 馬関連（B に入れる場合）: 現役馬 1.5 万頭 × 3 本 × 数十 KB ≒ 1GB 以上。
- 今の上限 `max_size=1g` では足りなくなる見込み。**VPS の空き容量を確認して上限を決める**必要がある（足りないときは馬関連を 7 日にするなど）。
- 上限を超えると古いものから消えるだけで、壊れはしない（消えた分は Cloud Run に取りに行く）。

### 3-6. 費用への影響

- 減るもの: Cloud Run の要求数（と、そこから GCS を読む回数）。
- 増えるもの: 取り直し（目印 1 件につき数本の GET）と、`cache_warm` の GCS 一覧（今も 1 分に 1 回、月約 4.3 万回 × Class A 約 0.8 円/千回 ≒ 35 円/月。変わらない）。
- VPS のディスクは追加費用なし。

---

## 4. 決まったこと（2026-11-02）

1. stg と prod の共有: **案 1（鍵を共有し、書くのは prod だけ）**
2. 会員の応答のキャッシュ: **後の段階**（今回は匿名の応答だけ）
3. 馬のページ: **VPS の容量を見て決める** → 適用時に `vps_09` がディスクの空きの 30%（1〜20GB）を上限にする。30 日保持・馬の目印で上書き
4. 当日・未来のレース: **6 時間の上限を残す**（確定後は 30 日）

## 5. 実装したこと

| 変更 | ファイル |
|---|---|
| 保持期間を三段に（確定済み 30 日 / 日付が分かる当日・未来 6 時間 / 日付不明は短いまま）。馬のページを 30 日。成績集計・レース質を 1 時間で共有キャッシュへ | `config/http_cache.json`、`src/api/http_cache_headers.py` |
| 馬の目印（`h`+馬ID）。戦績・プロフィール・血統の保存時に置く。馬 ID の先頭は生まれ年なので、年での間引きはしない | `src/utils/cache_marker.py` |
| 取り直し係: 馬の目印 → 馬のページ 3 本、開催日の目印 → 成績集計・レース質も。取り直せたパスは 11 分後にもう 1 回（API 内のメモリ上の使い回し 最大 10 分を越える） | `scripts/ops/vps/cache_warm.py` |
| **Cloud Run 内の使い回し対策**: 取り直しのときだけ nginx が `X-Keiba-Fresh: 1` を付け、API は共有鍵が通った要求に限り prerender の使い回しを読まずに作り直す（これが無いと、取り直しても Cloud Run が使い回す古い応答（当日 2 分・過去 6 時間）を長く保存してしまう） | `scripts/ops/vps/templates/nginx-api-cache.conf`、`src/api/middleware/proxy_key.py`、`src/api/prerender.py`、`src/api/app.py`（レース一覧） |
| stg と prod の共有: 鍵を `$request_uri` に。stg の待ち受けポートからの要求は読むだけで保存しない（`$keiba_cache_readonly`。ポートは `vps_09` が stg のサイト設定から読む。読めなければ中止） | `nginx-api-cache.conf`、`nginx-zones.conf`、`vps_09_api_cache.sh` |
| 上限をディスクの空きから決める（`API_CACHE_MAX_MB` で上書き可） | `vps_09_api_cache.sh` |
| テスト・検証 | `tests/ops/test_vps_api_cache.py`・`test_vps_cache_warm.py`、`tests/api/test_http_cache_headers.py`・`test_proxy_key.py`・`test_prerender_prewarm.py`、`tests/utils/test_cache_marker.py`、`scripts/ops/vps/verify_api_cache.py`（stg の読み取り専用・見出しの確認を追加） |

## 6. 適用の手順（この順に）

順序が大事: 馬のページの 30 日（API の見出し）が本番に出る前に、VPS の取り直し係と、目印を書く取得側が新しくなっている必要がある（逆だと、更新されない馬のページが 30 日残る）。

1. **取得側（目印の書き手）を新しくする**: ホストは develop を pull。VPS の取得役・Jobs は通常の反映（`ops/releases/fetch/`・Jobs のイメージ）で入る
2. **VPS（root）**: リポジトリを pull して
   ```bash
   sudo python3 scripts/ops/vps/verify_api_cache.py        # 本番に触れない検証（全部 PASS を確認）
   sudo bash scripts/ops/vps/vps_09_api_cache.sh            # 鍵の共有・stg 読み取り専用・上限の自動決定（表示される上限を確認）
   sudo bash scripts/ops/vps/vps_10_cache_warm.sh           # 取り直し係の更新（馬・集計・2 回目の取り直し）
   ```
   鍵の形が変わるので、今あるキャッシュは読まれなくなる（古いものは上限・35 日で自然に消える。すぐ消すなら `purge_api_cache.sh`）
3. **API を本番へ**（stg → master の承認）。`X-Keiba-Fresh` は古い API に届いても無視されるだけなので、2 と 3 の間に不整合は出ない
4. **1〜2 日後に課金を確認**（BigQuery `billing_export`。Cloud Run の要求数と GCS の操作数）

## 7. 残り

- 会員の応答のキャッシュ（3-4。署名つきの会員区分 Cookie と nginx での確認）
- `/api/race/{id}/entry/{horse_id}/card` は取り直し係の対象外（出走馬の一覧が要るため）。当日・未来は 6 時間、確定後は 30 日
- 開催日一覧・カレンダー（`/api/scrape-dates`・`/api/race-calendar`・`/api/megu-index-dates`）は URL が数本しかなく、30 日にしても節約がほぼないので、今の短い保持のまま
