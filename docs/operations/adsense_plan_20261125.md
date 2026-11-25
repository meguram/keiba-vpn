# 無料ゾーンに AdSense の広告を出す計画（2026-11-25、ads）

ユーザーの依頼: 無料ゾーン（匿名・無料会員）に Google AdSense などの広告を出したい。
パブリッシャー ID は `ca-pub-5321507754924902`（秘密ではない）。

## 0. 結論

- **AdSense は出せる見込み。** ギャンブルは禁止ではなく「制限付きコンテンツ（オンラインギャンブル）」の扱いで、日本のユーザーへの配信は制限の対象外と明記されている（§1）。
- **段 1（審査用・この変更）**: 匿名・無料会員の画面にだけ `adsbygoogle.js` を読み込む。全ページに所有の確認の meta、`/ads.txt`、robots で広告の巡回を許す。広告の枠はまだ置かない。
- **審査に要るユーザーの操作が 1 つある**: ルートの `megu-keiba.com` に DNS が無い。ads.txt の巡回はルートから始まるので、Cloudflare でルートから app の ads.txt へ転送する（§6、0 円）。
- **段 2（審査の後）**: 広告ユニットを置く。プライバシーポリシーの追記（Google の必須 3 点・外部送信の公表）は wording の担当と 1 つの変更にし、**広告の枠を出す前に入れる**。
- 費用: GCP・Cloudflare とも 0 円の見込み（§5）。

## 1. 規約（AdSense）

- パブリッシャー制限の「オンライン ギャンブルに関するコンテンツ」: 現金を賭けられるページ（カジノ・ブックメーカー・くじの販売・スポーツくじ）と、それへ送客する集約・アフィリエイトのサイトが対象。https://support.google.com/adsense/answer/10437795
  - 除外: 「関連するコンテンツが…日本…のユーザーに配信されている場合」は制限から外れる（同ページ、日本語版・英語版とも）。
  - このサイトでは投票できない。投票サイトへのアフィリエイトのリンクは置かない。
  - 予想・tipster を名指しした記述は無い。
- プログラム ポリシー・パブリッシャー ポリシーにギャンブルの章は無い。ギャンブルのサイト上の行動をパーソナライズ広告に使うことの禁止だけ。https://support.google.com/adsense/answer/48182 ・https://support.google.com/adsense/answer/10502938
- **依存症の相談窓口の表示を求める記述は、上の 3 文書に無い。** 注意書きは統括役の方針どおり 2 つ（「馬券の購入は 20 歳以上」「予想は参考情報で、結果を保証しない」）。言い換え（推奨馬券 → サンプル買い目など）は wording の担当（`docs/operations/wording_glossary.md`）を前提にする。
- Google 広告（広告主側）の日本の欄は、くじ・スポーツベッティングを免許のある事業者に限って認める。公営競技の名指しは無い。https://support.google.com/adspolicy/answer/6018017
- 審査は数日〜2〜4 週間。https://developers.google.com/adsense/platforms/direct/approvals
- 代わり（AdSense が落ちたとき）: i-mobile・忍者AdMax・nend などは、競馬を許すかの公式の基準を確かめられなかった。落ちたら各社に問い合わせる。投票サービスのアフィリエイトは、上の制限に当たるので使わない。

## 2. 範囲

- 出すのは匿名と無料会員（`is_member` が false）。有料会員（paid）と開発者（dev）には**スクリプトも読み込まない**。
  - 区分は `useAuthStatus`（`/api/v1/auth/status`）で取る。確認中・確認の失敗のときは読まない（会員を匿名と取り違えない）。
  - 判定は `frontend/lib/adsense.ts::shouldLoadAdsense`。テストは `frontend/lib/adsense.test.mjs`。
- ホスト: 本番 `app.megu-keiba.com` と stg `stg.megu-keiba.com`（ユーザーの指示「stg・prod ともに」）。手元の開発・別名のホストでは読まない。
- 止める印: ビルド時の `NEXT_PUBLIC_ADSENSE=off` で誰にも読まない。
- stg で読むのは表示の確かめのため。AdSense のサイトに登録しないので、広告はほぼ出ない。stg で広告を押さない（無効なトラフィックにしない）。
- 承認（統括役、2026-11-25）: `ads_architecture.md` の上書きは可。noindex は外さずに審査を出す。netkeiba の論点は統括役が匿名・無料に見える情報の出どころを監査中で、**結果が出るまで本番への反映と審査の申し込みは止める**（develop・stg までは進める）。
- 段 2 の枠の位置（案）: ページの末尾・記事（めぐコラム）の後。折り返しの上には置かない（会員の表示を動かさないため。§3）。
- 公開の範囲の決まり（`public_scope_plan_20261119.md`）は変えない。広告は画面の外の部品で、API の応答もキャッシュも区分の出し分けも触らない。

### 既存の設計書との違い（承認の項目）

`docs/operations/ads_architecture.md`（2026-10-21）は「限定側（app.）に第三者のスクリプトを入れない。広告は別オリジンの公開側に出す」としていた。今回は app. の匿名・無料会員に出す。上書きする理由:

- セッションの Cookie は `HttpOnly`・`SameSite=Lax`（`src/api/auth.py:266-272`）で、広告のスクリプトからは読めない。
- 有料会員・開発者の画面には読み込まない。
- 匿名・無料会員の画面には、会員だけの値が無い（API が落としてから返す）。
- app. はすでに匿名に開いている（Access の外）。別オリジンの公開側は今は無い。

## 3. 技術

- 入れ方: `frontend/components/ads/AdSenseLoader.tsx`（`next/script`・`afterInteractive`）を `app/layout.tsx` に 1 行で置く。区分が分かった後に読み込むので、**HTML は全区分で同じ**（nginx・Cloudflare の共有キャッシュの中身は変わらない）。
- 所有の確認: `metadata.other` で `<meta name="google-adsense-account" content="ca-pub-…">` を全ページに出す（https://support.google.com/adsense/answer/12169212）。スクリプトは JS で後から入るので、巡回が HTML だけを見ても確認できるようにする。
- ads.txt: `frontend/public/ads.txt`（Next が `/ads.txt` で返す）。`Cache-Control: public, max-age=86400`。保守の時間（03:00〜04:00）でも 503 にしない（`middleware.ts` の通すパスに `/ads.txt`・`/robots.txt` を足した）。
- robots: `Mediapartners-Google`・`Google-Display-Ads-Bot` だけ `Allow: /`。検索の索引は今までどおり拒否（`*` と `X-Robots-Tag: noindex`）。https://support.google.com/adsense/answer/10532
  - 本番の robots.txt は、Cloudflare の管理の robots（`User-agent: * Allow: /` と Content-Signal）が前に付いている。Mediapartners-Google は `*` を見ない（https://developers.google.com/search/docs/crawling-indexing/google-special-case-crawlers ）ので、名指しの許可で確実にする。
  - noindex が審査に効くかは、公式の文書に記述が無い。審査で「サイトが見つからない」と言われたら、ユーザーの判断で外す（検索に出るようになる）。
- CSP: 今は無い（`next.config.js`・nginx のどちらにも無い）。衝突しない。将来入れるときは Google の strict CSP（nonce＋`strict-dynamic`）だけが対応。https://support.google.com/adsense/answer/16283098
- Referrer-Policy は `same-origin` のまま。広告のスクリプトはページの URL を自分で読むので、配信に要らない見込み。段 2 で広告が空なら見直す。
- 表示の速さ: スクリプトは `afterInteractive` で、画面が使えるようになった後に読む。最初の表示（TTFB・LCP）には入らない。段 1 は枠を置かないので、表示のずれ（CLS）は 0。
  - 段 2: 自動広告は使わず、広告ユニットにする（自動広告は位置を Google が決め、ずれを抑えられない）。枠は折り返しの下に、高さを先に取って置く。
- 会員がログインした直後: ログインの画面は `router.push` で移る（全体を読み直さない）。匿名の間にスクリプトを読んでいたら、`window.location.assign` で全体を読み直して捨てる（`app/login/page.tsx`、`lib/adsense.ts::adsenseLoaded`）。
- robots の広告の巡回にも `/api/` は閉じたまま（Nav の罠のリンクで巡回の IP が 7 日止められないように）。
- 残る穴（直さない）: 区分の確認が 429 などの 4xx だと `fetchAuthStatus` は匿名として返すので、その間の有料会員にスクリプトが読まれうる。共有の取得の作りを変えると Nav などにも効くので、段 1 では変えない。害は広告が出るだけで、会員の値は漏れない。

## 4. 法と同意

- プライバシーポリシー（`frontend/app/legal/[doc]/page.tsx`）の今の文言「法令に基づく場合を除き、それ以外の第三者には提供しません」は、広告を出すと守れない。**段 2 の前に直す。** 足すこと:
  - Google の必須 3 点: 第三者（Google）が Cookie で過去の閲覧に基づく広告を出すこと・Google の広告 Cookie の使い方・広告設定（https://adssettings.google.com ）でパーソナライズを止められること。https://support.google.com/adsense/answer/1348695
  - 外部送信の公表（電気通信事業法 第27条の12）: 送る情報（閲覧したページの URL・Cookie の識別子・IP・端末の情報）、送り先（Google LLC）、目的（広告の配信・効果の測定）。https://www.soumu.go.jp/main_sosiki/joho_tsusin/d_syohi/gaibusoushin_kiritsu.html
    - このサイトが規律の対象かは境界にある（会員制の情報提供）。公表しておけば足りる。
  - 個人情報保護法: Cookie の識別子は「個人関連情報」。このサイトは Google に会員の ID を渡さないので、提供の同意（第31条）は要らない見込み。https://www.ppc.go.jp/personalinfo/legal/guidelines_tsusoku/
- **layout.tsx・legal は担当 wording の変更と重なる。** 段 1 の layout.tsx の変更は import 2 行・meta 1 項目・部品 1 行だけにした。プライバシーポリシーの追記は、wording が develop に入った後に、wording と相談して 1 つの変更にする。
- EEA・英国・スイス: パーソナライズ広告に Google 認定の CMP が要る（2024-01〜）。https://support.google.com/adsense/answer/13554116
  - 推奨: AdSense の「プライバシーとメッセージ」で Google の CMP（無料）の GDPR のメッセージを有効にする。コードは今のスクリプトのまま（追加の実装は無い）。日本の閲覧者には出ない。
  - 出さないと、EEA などの閲覧者には制限付きの広告が出るだけで、違反にはならない。

## 5. 費用（最悪の回数で）

- 広告のスクリプト・広告の中身は Google から直接ブラウザへ。VPS・Cloud Run・GCS を通らない。0 円。
- ads.txt・robots.txt・meta: VPS の Next が返す（Cloudflare が ads.txt を拡張子で置く）。Cloud Run の要求は増えない。0 円。
- 区分の確認（`auth/status`）: 既存の Nav・画面と同じ取得を共有する（`fetchAuthStatus` は 1 ページ 1 回）。広告のために増える要求は 0。
- 広告の巡回（Mediapartners-Google）が JS を動かして匿名の API を呼ぶ分: 匿名の API は nginx のキャッシュに当たる。
  - 最悪で 1 日 1,000 ページ × 5 要求が全部外れても、月 15 万回。Cloud Run の要求の無料枠（200 万回）の内で 0 円。コールドスタートは CLAUDE.md の最悪 96 回/日の見積もりの内。
- `adsense_check`: 外への HTTP 5 回と AdSense API 2〜3 回を 1 日 1 回、手で。0 円。
- Secret Manager に置く場合: 秘密 1 つ・版 1 つ（無料枠 6 版の内）、読み出し 1 日 1 回（無料枠 1 万回の内）。0 円。

## 6. ユーザーの操作（順に）

1. **今（段 1 が本番に出る前でよい）**: Cloudflare の DNS で、ルート `megu-keiba.com` にプロキシ付きのレコードを作る（例: `AAAA` `100::`、プロキシ オン）。
2. Cloudflare の Rules → Redirect Rules で、ルートの**全パス**を app へ 301（パスを保つ）: `megu-keiba.com/*` → `https://app.megu-keiba.com/${1}`。ads.txt もこれに含まれる。
   - 根拠: 「ads.txt の巡回はルートから始まり、ルートが ads.txt を返すか、そこへ転送する必要がある」https://support.google.com/adsense/answer/7679060
   - 全パスにするのは、サイトをルートで登録したとき、審査の巡回がルートのトップを開くため（404 や 52x だと「サイトが利用できない」で落ちうる）。転送の先の meta で所有の確認が通るかは公式の記述が無いので、画面で試す。
   - `ads_architecture.md` の「ルートには触らない」は、ルートに別のサイトが無いと分かった（名前が引けない）ので外す。
3. 段 1 が本番に出た後、AdSense の「サイト」に追加する。`megu-keiba.com` で受け付けるか `app.megu-keiba.com` で受け付けるかは、公式の文書に記述が無い。画面で試す（ルートが先の候補）。確認の方法は「meta タグ」か「ads.txt」を選ぶ。**stg はサイトに足さない**（Access の内側で巡回が届かず、審査に落ちて「要注意」が残るだけ）。
4. 審査を申請する（数日〜4 週間）。結果はメールと `adsense_check`（§7）で見る。
5. 通ったら、AdSense の画面で「プライバシーとメッセージ」→ GDPR のメッセージを有効にする（§4）。
6. 広告ユニット（ディスプレイ広告・レスポンシブ）を 1〜2 個作り、ユニット ID（data-ad-slot の数字）を統括役へ渡す。→ 段 2。

## 7. 審査の状態を Claude が確かめる

- 外から見える所: `bash scripts/ops/adsense_check.sh --no-api`（`src/ops/adsense_check.py`）。ルート・本番・stg の ads.txt、robots、匿名のトップと `/races` の meta を 1 画面で出す。統括役が 1 日 1 回、手で流す（cron は足さない）。
  - 有料会員に読み込まないことは、会員の資格を使わずに `frontend/lib/adsense.test.mjs` で確かめる。
- 審査の状態: AdSense Management API v2 の `accounts.sites.list`（state が READY・GETTING_READY・REQUIRES_REVIEW・NEEDS_ATTENTION）。資格が無いうちは「未設定」と出して終わる。値・トークンは画面にもログにも出さない。
- ユーザーの操作（一度だけ）:
  1. Google Cloud（`horse-racing-466415`）で「AdSense Management API」を有効にする。
  2. 「API とサービス」→「OAuth 同意画面」を「外部」・テストユーザーに自分のアカウントで作る（公開の審査は要らない。テストのままだと更新トークンが 7 日で切れるので、**「本番環境に公開」にする**。権限が adsense.readonly だけなら確認の審査なしで公開できる見込み）。
  3. 「認証情報」→ OAuth クライアント ID（デスクトップ アプリ）を作り、JSON を落とす。
  4. ホストで `python -m src.ops.adsense_oauth_setup --client-file <落とした JSON> --out ~/.config/keiba/adsense_oauth.json` を流し、出た URL をブラウザで開いて同意する。値は画面に出ない（権限 0600 で保存）。
  5. 確かめ: `ADSENSE_OAUTH_FILE=~/.config/keiba/adsense_oauth.json bash scripts/ops/adsense_check.sh`。
  - Secret Manager に置く場合は、`gcloud secrets create adsense-oauth --data-file=<同じ JSON>` の後、流す前に一時ファイルへ取り出す。手元のファイルで足りるので、既定はファイル。

## 8. 段の日程

- 段 1（この変更）: develop → stg（読み込みを確かめる）→ 本番は統括役がまとめて昇格。
- 段 2: 審査の通過と wording の変更の後。プライバシーポリシーの追記 → 広告ユニット（枠の高さを先に取る）→ develop → stg → 本番。

## 8.5 審査の申し込みの前に確かめたこと（2026-11-25）

- **匿名で開いた HTML の中身が薄い。** トップは本文 約 29 文字、`/races` は約 260 文字（スクリプト・タグを除く）。画面は `"use client"` で、本文は `/api/` から取る。
  - 巡回に `/api/` を閉じたので、JS を動かしても殻に見える見込み。「価値の低いコンテンツ」で落ちる典型の形。
  - 本文が HTML にあるめぐコラム（`/column`）は開発者だけの機能（`lib/features.ts`）で、匿名には出ていない。
  - 判断の候補（決め打ちしない）: (a) robots の閉じる範囲を罠の接頭辞（`/api/archive/` など）だけに狭める、(b) 匿名に出せる文章の画面（めぐコラムの一部・使い方・用語集）を公開してトップから入れる、(c) 両方。
- スクレイピング対策（`config/anti_scraping.json`）: 回数の制限と bot の判定は detect（記録だけ）。罠（honeypot）は enforce だが、巡回には `/api/` を閉じたので当たらない。巡回を止める設定は今は無い。
- 別件（既存の不具合）: 匿名で `/column` を開くと `307 → https://localhost:3000/` に転送される（middleware の開発者だけの転送が、プロキシの後ろで内側のホスト名を使う）。

## 9. 決めてほしいこと

- `ads_architecture.md` の「app. に第三者のスクリプトを入れない」を上書きしてよいか（§2。推奨: 上書きする）。
- noindex を外すか（推奨: まず外さずに審査を出す。落ちたら判断）。
- netkeiba の利用規約 第15条（営利目的の利用）: 広告は営利にあたる。有料会員がすでにあるので新しい論点ではないが、収益の源が増えることは記録しておく（`public_data_inventory.md`）。
