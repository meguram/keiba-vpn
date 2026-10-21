# 公開（匿名）と会員の出し分け

方針（2026-10-21 ユーザー決定）: **勝率・複勝率までは誰でも見られる。想定単勝（想定オッズ）・期待値・めぐAI印（旧「推奨」）・推奨馬券・その他の付随情報は会員のみ。**
UI の隠しではなく、**サーバが匿名の応答から項目を落とす**（許可リスト方式）。コード: `src/api/tier.py`、テスト: `tests/api/test_tiered_access.py`。

## 匿名で見える / 見えない

| 区分 | 匿名 | 会員・開発者 |
|---|---|---|
| 出馬表の事実（馬番・馬名・枠・性齢・騎手・斤量・調教師・馬体重）、レース結果、馬の過去走 | 見える | 見える |
| 勝率（`win_prob`）・複勝率（`top3_prob` / `show_prob`） | 見える（並びは勝率順に振り直し） | 見える |
| 連対率（`top2_prob` / `place_prob`） | 見えない | 見える |
| 想定単勝・想定複勝・人気・確定前オッズ（`race_odds` / `race_pair_odds` / 出馬表の `odds`・`popularity`） | 見えない | 見える |
| 期待値（`ev_*`・`expected_value`・`buy_tier`） | 見えない | 見える |
| めぐAI印（`mark_type`・`recommendation`・`megu_mark`・`kesi_reason`）・`bet_suggestion` | 見えない | 見える |
| 推奨馬券（`POST /api/v1/betting/optimize`）・レース価値（`/betting/race-values`） | 401 | 見える |
| 展開予想・ラップ・追走難度・レース質・想定めぐ指数・成長度・クラス基準・市場補正の注記 | 見えない | 見える |
| 指数・パドック・バロメーター・厩舎コメント・SmartRC（`/api/race/{id}` の付随カテゴリ） | 見えない | 見える |

会員限定のパス（匿名は 401 `reason: member_required`）: `tracking-difficulty`・`/api/race/{id}/bundle`・`bloodline-aptitude`・`/api/race-quality/*`・`/api/v1/betting/*`・`/api/v1/races/{id}/(final-odds|megu-index*|predictions/laps)`・`/api/v1/horse/{id}/(megu-index-history|growth-curve)`・`/api/predictions`・`pedigree-race-stats` / `data-analysis` 系。一覧は `tier.py` の `_MEMBER_ONLY_PATTERNS`。
項目を足す・外すときは `tier.py`（`PUBLIC_ENTRY_FIELDS` / `PUBLIC_TOP_FIELDS` / `LOCKED_GROUPS` / `RACE_DETAIL_MEMBER_KEYS`）とテストを直す。**予測応答に新しい項目を足しても、許可リストに入れない限り匿名には出ない。**

## ロール

セッション Cookie（署名付き）のロールは `dev`（開発者・全機能）と `member`（会員）。会員は開発者向け機能（`/api/v1/admin/*`・スクレイピング操作等）に入れない。開発者は会員を兼ねる。dev 環境のこの PC からの自動ログインは開発者扱い。

- ログイン: `POST /api/v1/auth/login {id, password, remember}`。`id` が空なら開発者パスワード（`DEV_PASSWORD`）、あれば会員。
- 状態: `GET /api/v1/auth/status` / `GET /api/auth/status` → `tier`（`public` / `member` / `dev`）・`is_member`・`is_developer`。
- 会員アカウントは `config/members.json`（ソルト付き PBKDF2-SHA256 のハッシュのみ。平文は置かない）。`requires_env` が指す環境変数が `1` のときだけ有効。

## テストユーザー（id: test / pw: test）

動作確認用。`KEIBA_TEST_USER=1` のときだけログインできる（**既定オフ**）。切替: `bash scripts/ops/test_user_flag.sh on|off|status`（`.env` を書くだけ。反映は API の再起動後）。
**一般公開（Cloudflare Access を外す）前には必ず `off`**。test/test は誰でも推測できるため、公開環境で有効のままだと有料会員の機能が誰にでも開く。

## 公開に向けた残り

- Cloudflare Access（現状は所有者のみ）の緩和は未実施。緩和するときは anti_scraping を enforce にし、`KEIBA_TEST_USER` を `off` にする（`docs/operations/go_live.md`）。
- 有料会員の登録・課金は未実装（`config/members.json` を手で足す段階）。
