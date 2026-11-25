# 公開（匿名）と会員の出し分け

方針（2026-11-25 ユーザーの確定。2026-10-21・11-19 の決まりを置き換える）:
- 匿名・無料会員（`is_member()` が偽）には、**レースと出走馬の公式の事実と、印（◎○▲△★消）だけ**を出す。
- **netkeiba 独自の情報・確定前のオッズ・予測の数値（勝率・複勝率・順位）は有料会員から**。
- 確定後の結果の表の確定オッズ・人気・払戻は、JRA が公表する確定の事実なので誰にでも出す。

UI の隠しではなく、**サーバが応答から項目を落とす**（許可リスト方式）。コード: `src/api/tier.py`。テスト: `tests/api/test_tiered_access.py`・`tests/api/test_check_anon_scope.py`。
本番では `config/public_fields.json` の `mode=private`（`KEIBA_PUBLIC_MODE` が空）。下の表は private の経路の出し方。

## 匿名・無料会員で見える / 見えない

| 区分 | 匿名・無料 | 有料会員・開発者 |
|---|---|---|
| レースの見出し・出馬表の事実（馬番・馬名・枠・性齢・騎手・斤量・調教師・馬体重） | 見える | 見える |
| 結果の事実（着順・タイム・着差・上がり・通過・ラップ） | 見える | 見える |
| 確定の結果の単勝オッズ・人気・払戻（`race_result` の元が確定の結果のときだけ。判定は `tier.race_result_confirmed`、応答の `access.result_confirmed`） | 見える | 見える |
| 確定前のオッズ・人気（出馬表の `odds`・`popularity`、速報 `race_result_on_time`、`race_odds`・`race_pair_odds`、速報だけの段の結果のオッズ・払戻） | 見えない | 見える |
| 印（`mark_type`・記号の `mark`）。印は 1 種類（全員に同じめぐAI印）。無料の新しい印 `public_mark_type` は影の記録だけで出さない（2026-11-25） | 見える（馬番順） | 見える |
| 勝率・複勝率・順位・連対率・想定オッズ・期待値・買い判断・印の説明・展開・めぐ指数 | 見えない | 見える |
| 馬柱・近走（`horses` の `race_history`）・タイム指数・各指数・備考・厩舎コメント・追い切り・パドック・バロメーター・SmartRC | 見えない | 見える |
| 出走馬カード（`/api/race/{id}/entry/{horse}/card`） | 近走の公式の事実・同条件の成績・騎手・調教師の成績。追い切り・厩舎コメント・成長曲線（めぐ指数）・近走の人気は落とす（`access.locked`） | 全部 |
| 予測成績（`/api/prediction-results/*`） | 印と着順の突き合わせ。勝率・オッズ・人気・仮想購入は落とす | 仮想購入まで |
| Flask `/api/v1/races/{id}/results` | 確定の結果はそのまま。速報のときはオッズ・人気・払戻を落とす | そのまま |

- 画面は、落とした欄を空欄にせず「🔒 会員限定」と出す（`EntryCardPanel`・レース詳細・週の予想・予測成績）。
- 出し分けを変えて配備したら、nginx の API のキャッシュを purge する（`scripts/ops/vps/purge_api_cache.sh`。確定済みのレースの匿名の応答は s-maxage 30 日で置かれるので、purge しないと前の版が出続ける）。edge のフォルダは、指紋に配備の commit が入るので自動で作り直される。Cloudflare は API を置かない（`cf-cache-status: DYNAMIC`。2026-11-25 に確認）。
- 漏れの検査: `python -m src.scripts.ops.check_anon_scope --base <URL> --date <YYYYMMDD>`。匿名で取得して、出してはいけない欄（`tier.SCOPE_FORBIDDEN_KEYS`）が 1 つも無いことを確かめる。無料会員で確かめるときは `KEIBA_SCOPE_COOKIE` を付ける。周期では流さない（1 回で約 40 件の要求）。

会員限定のパス（匿名・無料会員は 401 `reason: member_required`）: `/api/data/*`（生の JSON）・`/api/horse/*`（馬のページ）・`/api/betting/pair-odds/*`・`/api/odds/*`（2026-11-25 に追加）・`tracking-difficulty`・`/api/race/{id}/bundle`・`bloodline-aptitude`・`/api/race-quality/*`・`/api/v1/betting/*`・`/api/v1/races/{id}/(final-odds|megu-index*|predictions/laps)`・`/api/v1/horse/{id}/(megu-index-history|growth-curve)`・`/api/predictions`・`pedigree-race-stats` / `data-analysis` 系。一覧は `tier.py` の `_MEMBER_ONLY_PATTERNS`。
項目を足す・外すときは `tier.py`（`PUBLIC_ENTRY_FIELDS`・`RACE_HEADER_KEYS`・`SHUTUBA_PUBLIC_ENTRY_KEYS`・`RESULT_PUBLIC_ENTRY_KEYS`・`CARD_RECENT_KEYS`・`SCOPE_FORBIDDEN_KEYS`）とテストを直す。**応答に新しい項目を足しても、許可リストに入れない限り匿名・無料には出ない。**

## ロール

セッション Cookie（署名付き）のロールは `dev`（開発者・全機能）と `member`（会員）。会員は開発者向け機能（`/api/v1/admin/*`・スクレイピング操作等）に入れない。開発者は会員を兼ねる。dev 環境のこの PC からの自動ログインは開発者扱い。

- ログイン: `POST /api/v1/auth/login {id, password, remember}`。`id` が空なら開発者パスワード（`DEV_PASSWORD`）、あれば会員。
- 状態: `GET /api/v1/auth/status` / `GET /api/auth/status` → `tier`（`public` / `member` / `dev`）・`is_member`・`is_developer`。
- 会員の名簿は `data/local/auth/members.json`（正本は GCS。`members_store_gcs.md`）。ハッシュだけを持ち、平文は置かない。
- 動作確認用の test は `config/members.json`。2026-11-19 から git の外で、置いたマシンでだけ有効（CI で作るイメージには入らない。`members_json_purge_20261119.md`）。`requires_env` が指す環境変数が `1` のときだけ有効。
- stg の版（rstg）の test は、Secret Manager の `KEIBA_STG_TEST_USER`（stg 専用）から読む（2026-11-25〜）。下の「stg の test」。

## テストユーザー（id: test / pw: test）

動作確認用。`KEIBA_TEST_USER=1` のときだけログインできる（**既定オフ**）。切替: `bash scripts/ops/test_user_flag.sh on|off|status`（`.env` を書くだけ。反映は API の再起動後）。
**一般公開（Cloudflare Access を外す）前には必ず `off`**。test/test は誰でも推測できるため、公開環境で有効のままだと有料会員の機能が誰にでも開く。

### stg の test（rstg。2026-11-25〜）

- 読む場所: 環境変数 `KEIBA_STG_TEST_USER`（旧形式の 1 件の JSON）。`src/api/auth.py::_stg_test_user`。
- 通る条件: `KEIBA_STG_TEST_USER` があり、かつ `KEIBA_TEST_USER=1`。片方だけでは通らない。ID は `test` だけ。
- 載せるのは `scripts/gcp/deploy_cloud_run_api.sh` で、`API_TEST_USER=1`（`deploy_gcp.sh --target stg`）のときだけ。本番の版には秘密も目印も載らない。
- promote と `rollback_cloud_run_api.sh` は、`KEIBA_TEST_USER`・`KEIBA_STG_TEST_USER` を持つ版を本番の通信に載せない。
- 秘密は配備より先に要る（無いと `gcloud run deploy` が失敗する）。2026-11-25 に作成済み。
- 秘密を作る・作り直す: `bash scripts/ops/stg_test_user_secret.sh`（値は画面に出ない）。反映は次の stg の配備から。
- 名簿の正本（GCS・`data/local/auth/`）は本番と共有なので、test はそこに入れない。

## 公開に向けた残り

- 本番（app.megu-keiba.com）は Cloudflare Access の外で、匿名にも開いている。公開の範囲は上の表の private の経路で決まる。
- 会員は `manage_users`・管理 API・決済で作る（`user_management.md`）。`config/members.json` には足さない。

## 実オッズは開発者以外に出さない（`actual_odds_public`）

netkeiba 由来の市場オッズそのものは、情報元の許可が取れるまで、ゲスト・無料・有料のいずれにも出さない（`config/public_fields.json` の `release.actual_odds_public=false`、既定）。開発者（所有者・私的利用）の応答は変えない。

- 予測（`/api/race/{id}/predictions`）: 有料会員には `win_odds` / `place_odds_*` / `popularity` / `odds_source` / 生の期待値（`ev_win` `ev_place` `ev_top2` `expected_value`。勝率と並べると実オッズを逆算できる）/ `bet_suggestion`（文中に想定単勝）を出さない。代わりに、最終オッズ予測のキャッシュ（有料ゾーンと同じ較正後の想定オッズ）から単勝・複勝・想定人気・期待値を付ける。キャッシュが無いレースは付けない（実オッズへ落とさない）。較正後の期待値（`ev_*_cal`）・印・買い判断は出す。
- レース詳細: `race_odds` / `race_pair_odds` / 出馬表・結果の `odds` `popularity` は元から出さない。確定払戻（`payoff`）は確定オッズそのものなので、同じフラグが true になるまで出さない。
- 予測成績（`/api/prediction-results/*`）: 有料会員にも、印の馬の人気・確定オッズ・平均オッズ/人気、各買い目の `odds`、注目レースの『単勝N倍』を出さない（`tier.redact_prediction_results_member`）。仮想購入の金額・払戻・回収率は出す。
- `/api/race/{id}/final-odds`（想定オッズ）は会員限定。
- 許可が取れたら `actual_odds_public` を true にすると、上の項目が会員（開発者以外）にも出る。検証: `tests/api/test_tiered_access.py`（4 区分 × エンドポイント）、`tests/api/test_public_policy.py`。
