# ログインの POST を Cloudflare で抑える（手順書。ユーザーの操作）

`login_scale_review_20261117.md` の【高】の 2 件目への対策。

失敗のログインも、照合（PBKDF2 60 万回）で 0.4〜0.5 vCPU 秒を使う。攻撃で 3 台が照合に埋まると、最悪で月 約 3.3 万円になる。アプリのロック（`config/anti_scraping.json`）には 2 つの抜け穴がある。

- ロックはインスタンスごとのメモリにあり、再起動で消える。
- ロックの判定の前に Cloud Run が起動する（起動とリクエストの費用がかかる）。

Cloud Run に届く前に、Cloudflare で止める。

## 1. レート制限のルール（推奨。0 円・コードの変更なし）

Cloudflare のダッシュボード → `megu-keiba.com` のゾーン → **Security → WAF → Rate limiting rules → Create rule**

| 項目 | 値 |
|---|---|
| ルール名 | `login-post-limit` |
| 条件（Expression） | `(http.request.method eq "POST" and http.request.uri.path in {"/api/v1/auth/login" "/login" "/api/auth/login"})` |
| 数える単位 | IP（無料プランは IP だけ） |
| しきい値 | 10 秒に 5 回 |
| 動作 | Block（ブロックする時間は 10 秒。無料プランの上限） |

- 無料プランで作れるレート制限のルールは 1 本で、期間・ブロックの時間は 10 秒の想定。作る画面で選べる値を確かめ、選べる範囲で近い値にする。
- 効果: 1 つの IP からは毎分 30 回まで。1 つの IP が使える照合の CPU は、最大でも毎秒 0.5 回 × 0.45 秒。アプリのロック（15 分に 5 回の失敗）と合わせて、総当たりを抑える。
- 副作用: ふつうの人は、10 秒に 5 回ログインを押すことはない。共有の出口（会社・携帯）の向こうで、同時に 5 人以上がログインするときだけ待たされる（10 秒）。
- 確かめ方:
  - ダッシュボードの **Security → Events** で、ルール名で絞って件数を見る。
  - stg（Access の内側）は対象外でよい。本番のゾーンだけで足りる。
- 戻し方: 同じ画面でルールを **Disable**（無効）にするか、消す。

## 2. Turnstile（必要になったら。コードの変更が要る）

- 何か: Cloudflare の、人かどうかの確認（無料）。ふだんは見えない形で動く。
- 入れ方の案（ログインの速さを落とさないため、疑わしいときだけ）:
  1. アプリのロックの記録で、その IP に直近の失敗が 2 回以上あるときだけ、ログインの応答に `captcha_required` を返す。
  2. 画面は、そのときだけ Turnstile の部品を出し、得た値を付けてログインを送り直す。
  3. API は Cloudflare の確認の口（siteverify）で値を確かめてから照合する。
- 費用: 0 円。確認の通信は約 50 ms で、疑わしいときだけ。
- ユーザーの操作: ダッシュボードの **Turnstile → Add site** でサイトの鍵と秘密の鍵を作る。秘密の鍵は Secret Manager に `TURNSTILE_SECRET` として登録する（`scripts/gcp/put_secret.sh`）。
- 今は入れない。1 のルールで足りなくなったら（Security → Events で、多数の IP からの失敗が続くようになったら）実装する。

## 3. アプリ側の目安

- 1 時間に 30 回を超える失敗で、通知が出る（`config/anti_scraping.json` の `global_failures_per_hour_alert`）。通知が続くなら、1 のしきい値を下げる。
