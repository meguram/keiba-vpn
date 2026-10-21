# スクレイピング対策

> 最終更新: 2026-10-21。設定: `config/anti_scraping.json`（30 秒ごとに読み直し。API の再起動は不要）。実装: `src/utils/anti_scraping.py`（判定）、`src/api/middleware/anti_scraping.py`（FastAPI / Flask への接続）、`frontend/`（robots.txt・ヘッダ・ハニーポット）。

## 1. 脅威モデルと現状の露出

守りたいもの: 予測確率・期待値・推奨馬券・印・めぐ指数・成長度・展開予想などの API 応答（このサイトの価値の中心）。netkeiba 由来のデータ（レース結果・馬の過去走など）も API から返るため、元データの一括取得（再配布にあたる使われ方）を防ぐ意味もある。

想定する相手: (a) 連番の `race_id` を順に叩く収集プログラム、(b) 検索・AI 学習のクローラー、(c) 脆弱性スキャナ、(d) パスワードの総当たり、(e) ログインしたセッションを共有・転売して複数の接続元から使うケース。

**対策前の露出（`python -m src.scripts.maintenance.anti_scraping_audit` で再現できる）**

- FastAPI の GET API 144 本が、**未ログインのまま取れる**状態だった（認証が必要なのは開発者向けの管理系だけで、`src/api/auth.py` の `requires_auth` は旧ルートの読み取りを通す設計）。予測 API（`/api/race/{id}/predictions`）、週間予測、予測成績、血統、データ閲覧などが含まれる。
- そのうち ID・日付で**列挙できる形**のパスが 19 本。連番の `race_id` で全レースを総なめにできる。
- レート制限・bot 判定・ログインの試行回数制限は無かった。robots.txt も無かった。
- 経路: 公開トンネル → Next.js(:23200) → rewrites で `/api/v1/*` は Flask(:23210)、`/api/*` は FastAPI(:23220)。ブラウザの画面は枠だけで、データはブラウザから API を呼んで取る（サーバ側で Cookie 無しに取る画面は無い）ので、API 側で認証を強制しても画面は壊れない。

## 2. 多層防御（層・既定のモード）

`mode` は `off`（無効）/ `detect`（記録と通知だけ。止めない）/ `enforce`（実際にブロック）。**既定は検知のみ**。全体を止める非常口は環境変数 `KEIBA_ANTI_SCRAPING=off`（要 API 再起動）。

| 層 | 内容 | 既定 |
|---|---|---|
| 認証 | `/api/*` の読み取りは未ログインなら 401（例外: 健康確認・認証状態・ログイン・ハニーポット）。内部ジョブは対象外 | detect |
| レート制限 | IP 単位・ユーザー（セッション）単位・種別単位の滑動窓。データ API は 1 分 120 回・10 秒 40 回、ページは 1 分 180 回。**5 分間に異なる race_id を 60 件超**（馬 ID は 120 件超）も制限。超過は 429 + Retry-After → 違反の累積で段階的に一時ブロック（3 回 15 分 / 8 回 24 時間 / 20 回 7 日） | detect |
| bot 判定 | UA なし・既知のスクレイピングライブラリ・ヘッドレス、ブラウザを名乗るのに Sec-Fetch-* が無い、API だけを叩く（ページを見ていない）、連番 ID の総なめ、人間にありえない速度、を点数化（閾値 6、10 分で減衰） | detect |
| ハニーポット | 画面外のリンク（Nav 内）と `/api/archive/`・`/api/_hp/` などを踏む、`/.env`・`/wp-login.php` などのスキャナ探索パスを踏むと、その接続元を 7 日ブロック | **enforce** |
| ログイン試行制限 | 同一接続元で 15 分に 5 回失敗 → 15 分ロック（繰り返すと 1 時間・24 時間）。成功で解除。FastAPI の `/login` と Flask の `/api/v1/auth/login` の両方 | **enforce** |
| セッション異常 | 同じセッションが 10 分に 4 つ以上の接続元から使われたら検知（enforce でそのセッションを 15 分ブロック） | detect |
| robots / ヘッダ | `/robots.txt` で全面 Disallow（主要な AI クローラーを名前でも拒否）、全レスポンスに `X-Robots-Tag: noindex, nofollow, noarchive, noai, noimageai` | 有効 |
| 応答の最小化 | 拒否時の本文は `error` / `reason`（login_required / blocked / rate_limited）/ `retry_after` だけ。しきい値・スコア・検知した信号は返さない | 有効 |

**robots.txt と noai ヘッダは、守る相手にしか効かない。** 実効的な防御は認証・レート制限・ブロックと、次節の前段。

### 内部ジョブを巻き込まない設計
cron・開催日ランナー・監視ポータル・Slack 通知・Next.js のサーバ側取得は、**信頼プロキシ（既定はループバック）からの、転送ヘッダの無い直接アクセス**として扱い、判定を素通しする。公開トンネル経由のリクエストは必ず転送ヘッダ（`X-Forwarded-For` など）が付くので対象になる。許可リスト `allowlist_ips`（IP/CIDR）にも同じ扱い。

### 接続元 IP の決め方（偽装対策）
直接の接続元が信頼プロキシ（`trusted_proxies`）のときだけ転送ヘッダを採用する。`X-Forwarded-For` は**右から見て、信頼プロキシでない最初の値**を採る（先頭値は偽装できるため）。`CF-Connecting-IP` / `X-Real-IP` は設定順に見る。トンネルがループバックから中継し、本当のクライアント IP が付かない場合は、全員が同じ IP に見えて巻き添えにならないよう、UA と言語から作った指紋（`fp:xxxx`）をキーにする。**IP が取れる前段（Cloudflare 等）にすると精度が上がる**。

### ミドルウェアの順序
外側 → 内側: `KeibaDeployment`（環境ヘッダ）→ メンテナンス判定 → **AntiScraping（ブロック判定・レート制限・bot）** → Auth → GZip → ルート。Flask は `before_request` の登録順（メンテナンス → AntiScraping）。判定でエラーが出ても**素通し**にする（対策の不具合でサイトを止めない）。

## 3. 有効化の手順（段階的）

1. **検知のみで 1〜2 週間回す**（反映直後の状態）。ログ: `data/local/meta/anti_scraping/events-YYYYMMDD.jsonl`（時刻・種別・検知/ブロック・接続元・UA・パス・理由。同じイベントは 60 秒に 1 回）。日次サマリー: `python -m src.scripts.maintenance.anti_scraping_report --hours 24`（`--send` で Slack、`KEIBA_SLACK_NOTIFY=1` のときだけ。`--alert-only` は 1 時間のブロックが閾値以上のときだけ）。
2. ログで「正規の利用が検知に引っかかっていないか」を見る（特に `anon_protected` が、あなた以外のログインしていない閲覧者によるものか）。
3. `config/anti_scraping.json` の `mode` を順に `enforce` へ: `auth` → `rate_limit` → `bot`。しきい値は検知ログの分布で調整（正規の画面は 1 回の表示で数十リクエスト。週間予測を開いても異なる race_id は 40 件未満の想定）。
4. 誤ブロックしたときの非常口: `allowlist_ips` に IP/CIDR を足す（30 秒で反映）、またはその mode を `detect` に戻す。ブロックはメモリ上なので、API を再起動すると全て解除される。

## 4. 前段（インフラ）の推奨 — ここが最も効く

アプリ側の対策は「正規の見た目をした機械」を完全には止められない。**そもそも到達させない**のが最強。利用形態で選ぶ。

| 利用形態 | 推奨 | 効果 | 手間 |
|---|---|---|---|
| 自分だけ | **IP 許可リスト** または **Cloudflare Access（ゼロトラスト認証、メール OTP）** をサイトの前に置く | 認証できない相手はアプリに到達しない（収集は事実上不可能） | 小 |
| 限定された数人 | Cloudflare Access のポリシー（許可メール/ドメイン）。アプリ内認証は共有パスワード 1 つなので、Access で人を区別できるとセッション共有の追跡も可能 | 高 | 小〜中 |
| 一般公開 | Cloudflare の Bot Fight Mode / Rate Limiting / WAF + ログイン画面に Turnstile。アプリ内は enforce | 中 | 中 |

現在の公開経路は `tcpexposer` のトンネル（`service-endpoints.md`）。実クライアント IP を渡さない場合は IP 単位の制御が弱い（上の指紋キー）。**Cloudflare Tunnel + Cloudflare Access に移行するのが最優先の推奨**。以下はあなた（Cloudflare アカウントの操作）が行う設定の一覧。

1. **Cloudflare Tunnel** でホストの Next.js（:23200）を公開（`cloudflared tunnel`）。アプリ側は何も変えず、`CF-Connecting-IP` が付くようになる（`client_ip_headers` の先頭に既に入っている）。公開ホスト名は Cloudflare 経由のみにする（オリジンの直公開をやめる）。
2. **Zero Trust > Access > Applications**: 自己ホスト型アプリを追加（サイトのホスト名）。ポリシー `Allow`: メールアドレスが `<自分のメール>`（人を足すときは Emails に追加）、認証は One-time PIN。セッション期間は 24 時間〜7 日。**API の `/api/*` も同じアプリに含める**（パスで分けない）。
3. **Security > Bots**: Bot Fight Mode を有効（Free）。有料なら Super Bot Fight Mode で「Definitely automated: Block」「Likely automated: Managed Challenge」、「Verified bots: Allow」は不要なので Block。
4. **Security > WAF > Rate limiting rules**: ルール 1「`/api/*` に同一 IP から 1 分 120 回超 → 10 分ブロック」、ルール 2「`/login` と `/api/v1/auth/login` への POST が 1 分 10 回超 → 1 時間ブロック」。
5. **Security > WAF > Custom rules**: `(http.user_agent contains "python-requests") or (http.user_agent contains "curl") or (http.user_agent contains "scrapy") or (http.user_agent eq "")` → Block（アプリ内の bot 判定と二重）。`(ip.geoip.country ne "JP")` → Managed Challenge（国内のみの利用なら）。
6. **Turnstile**: ログインフォームに設置する場合は、サイトキーを Next.js の環境変数、シークレットを `.env` に置き、`/api/v1/auth/login` でトークンを検証する（実装は必要になった時点で追加。Access を使うなら不要）。
7. **Security > Settings**: Security Level を Medium 以上、Browser Integrity Check を有効。

`tcpexposer` のまま使う場合: そのサービスが IP 許可リストや基本認証を提供しているか確認する。無ければ、ホストのファイアウォール側で許可 IP だけに絞る（自宅/職場の固定 IP がある場合）。

## 5. データ側の対策（現状と方針）

- **列挙の抑止**: レートに加えて「5 分間の異なる race_id / horse_id の件数」で総なめを検知。連番 ID は netkeiba の形式で変えられないため、署名付きトークン方式は入れていない（導入コストが大きい。Access で足りる）。
- **件数上限**: 一覧系 API のページネーション上限は個別の API ごとの実装。`route_audit.md`（監査スクリプトの出力）で列挙可能な 19 本を見直す候補にする。
- **応答の透かし**（ユーザー別の微小な識別子を応答に混ぜて流出元を追跡）は、応答の整合性・キャッシュ・テストへの影響が大きいので**導入していない**。必要なら Access 導入後（人が特定できる状態）に検討。
- **エラー応答**: 拒否時は内部情報を返さない（上記）。

## 6. 運用

- **ログ**: `data/local/meta/anti_scraping/`（追記のみ）。`route_audit.md` は `python -m src.scripts.maintenance.anti_scraping_audit --out data/local/meta/anti_scraping/route_audit.md` で再生成。
- **日次サマリー/警報（任意）**: cron に `python -m src.scripts.maintenance.anti_scraping_report --send`（日次）と `--alert-only --hours 1`（1 時間おき）。`KEIBA_SLACK_NOTIFY=1` のときだけ送る。
- **誤ブロック**: 上記の非常口。ログイン失敗のロックは 15 分で自動解除（API 再起動でも解除）。
- **限界**: 状態はプロセス内メモリ（再起動で消える。ブロックが短時間中心なので許容）。複数プロセスで動かす場合はプロセスごとに数える。ログインしたブラウザの操作を模倣する本格的なボットは、アプリ内では止めきれない（Access・レート制限・ログの監視で対処）。

## 7. 反映（ホスト）

ファイル: `config/anti_scraping.json`、`src/utils/anti_scraping.py`、`src/api/middleware/__init__.py`、`src/api/middleware/anti_scraping.py`、`src/api/app.py`（登録 4 行）、`src/api/flask_app.py`（登録 4 行）、`src/scripts/maintenance/anti_scraping_report.py`・`anti_scraping_audit.py`、`frontend/app/robots.ts`、`frontend/components/Nav.tsx`、`frontend/middleware.ts`、`frontend/next.config.js`。反映後は API の再起動とフロントのビルド・切り替えが要る。ロールバックは、退避したファイルを戻して再起動、または環境変数 `KEIBA_ANTI_SCRAPING=off`（要再起動）。
