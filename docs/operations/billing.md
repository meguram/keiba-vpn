# 有料プラン（Stripe）— 裏側の実装（UI には出さない・既定は無効）

netkeiba のライセンス確認が済むまで、**画面には一切出さず、API も 404 のまま**にしておく。実装だけ先に入れてある。

## 無効の仕組み（三重）
1. `KEIBA_BILLING=1` が無いと全 API が 404（`src/billing/service.py` の `enabled()`）。
2. `STRIPE_SECRET_KEY` / `STRIPE_WEBHOOK_SECRET` / `STRIPE_PRICE_ID` が揃わないと 404。
3. **本番キー（`sk_live_`/`rk_live_`）は `config/billing.json` の `license_confirmed: true` になるまで動かない**。テストキーは確認前でも動かせる（テストモードの動作確認用）。
フロントに入口は無い（`frontend/lib/features.ts` にも未登録）。

## 流れ
1. `POST /api/v1/billing/signup`（ID・パスワード）→ 仮登録（`billing.status=pending`。支払いまでログイン不可）→ Stripe Checkout の URL を返す。Stripe に繋がらなければ仮登録は残さない。24 時間で支払いが無い仮登録は自動削除。
2. 支払い後、Stripe が `POST /api/v1/billing/webhook` を送る。**権限は Webhook だけが変える**（成功ページへの戻りは信用しない）。
3. 解約・カード変更は Stripe の Customer Portal（`POST /api/v1/billing/portal`、会員のみ）。
4. `GET /api/v1/billing/status`：自分の状態（有効時のみ）。

## 安全策
- Webhook: `Stripe-Signature` を HMAC-SHA256 で検証、時刻のずれ 300 秒超は拒否（リプレイ対策）、イベント ID で冪等（`data/local/billing/events.json`）。対象が未登録の先着イベントは 409 を返して Stripe の再送に任せる。
- 個人情報: 会員ファイルには「ログインIDの変換値（idk）」と Stripe の顧客 ID・購読 ID・期間の終わりだけ。メール・カード情報は持たない（Stripe 側）。カード情報はサーバを通らない（Hosted Checkout）。
- 支払い遅延（`past_due`）は期間の終わり＋`grace_days`（既定 3 日）まで利用可。`pending/canceled/incomplete` は不可。管理者が作ったアカウント（billing なし）は対象外。
- CSRF: signup / portal は `Origin` があれば `config/billing.json` の `allowed_origins` に限る。signup は IP ごと 1 時間 5 回まで。
- 秘密鍵は `.env` のみ（git に入れない）。可能なら制限付きキー（rk_）。

## 有効化の手順（将来）
点検: `python3 -m src.billing.preflight`（未達の項目と対処を表示。全部 OK になってから有効化する）。申込み画面（`/signup`）と法的表記（`/legal/{tokushoho,terms,privacy}`）は実装済みで、`frontend/lib/features.ts` の `billing`（既定 false）が false の間は `/` へ戻される。
1. データの権利（netkeiba / JRA-VAN 等）を確認し、`config/billing.json` の `license_confirmed` を `true` に。
2. Stripe でテストモードの商品・価格を作り、`.env` に 3 つの変数と `KEIBA_BILLING=1`。`stripe listen` / テストカードで一巡確認。
3. Webhook のエンドポイント `https://app.megu-keiba.com/api/v1/billing/webhook` を Stripe に登録。**Cloudflare Access はこのパスだけ Bypass が必要**（署名で守る）。
4. 特定商取引法の表記・利用規約・プライバシーポリシーのページを用意（Stripe の審査にも必要）。
5. フロントの登録ボタン・料金カードを `features.ts` で出す。
テスト: `python3 -m pytest tests/billing`
