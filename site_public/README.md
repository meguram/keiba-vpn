# site_public — 公開側サイト（広告用・静的）

限定側のアプリ（`frontend/`、`app.megu-keiba.com`、Cloudflare Access 保護）とは**完全に独立**した静的サイトです。依存は Python 標準ライブラリだけ。設計の根拠は `docs/operations/ads_architecture.md`（フェーズ 2）。

**置かないもの**: 予測確率・期待値・推奨馬券・印・独自指数・外部サイト由来のデータ（成績表・オッズ・指数）・個別のレース/馬の情報。置くのは、自作の解説・用語・検証の考え方・個別を特定しない自前の集計だけ。広告は**未導入**（枠のコードと `ads.txt` の雛形のみ、無効）。

## 構成

```
site_public/
  config.json              サイト名・URL・運営者の雛形・広告フラグ（既定 false）・許可ドメイン・記事の並び
  build.py                 ビルド＋HTML 検証＋リンク切れ検査（標準ライブラリのみ）
  check_public_safety.py   公開してはいけないものの検査（禁止パターン）
  content/pages/*.html     固定ページ（トップ・このサイトについて・用語集・FAQ・お問い合わせ・運営者情報・プライバシー・利用条件・注意）
  content/articles/*.html  解説記事 8 本（先頭に <!--meta title/description/date--> が必要）
  static/                  style.css・favicon.svg（→ dist/assets/）
  dist/                    出力（git 管理外）
```

出力には `robots.txt`（検索は許可、AI の学習・収集系は拒否）、`sitemap.xml`、`ads.txt`（コメントのみ）、`_headers`（Cloudflare Pages 用のセキュリティヘッダー。CSP は `script-src 'none'`）、`404.html` が含まれます。解析タグ・Cookie・外部フォント・外部スクリプトは使いません。

## ローカルでのビルドとプレビュー

```bash
python3 site_public/build.py                      # dist/ を作り、HTML とリンクを検査
python3 site_public/check_public_safety.py --dist site_public/dist   # 公開してはいけないものの検査
python3 -m http.server 8080 -d site_public/dist   # http://localhost:8080/ で確認
python3 site_public/build.py --preview-ads        # 広告枠の位置を灰色の枠で確認（本番には使わない）
python3 site_public/build.py --strict             # 【要入力】【要確認】が残っていたら失敗（公開前の最終確認）
python3 -m pytest tests/site_public -q            # テスト
```

ホスト名を変える（`www` ではなく `pub` にするなど）ときは `config.json` の `site_url` を直すか、`--site-url https://pub.megu-keiba.com` を付けます。

## 記事の書き方

`content/articles/<slug>.html` を追加し、`config.json` の `articles_order` に slug を足す。先頭は次の形:

```html
<!--meta
title: 記事のタイトル
description: 一覧と検索結果に出る説明（1〜2 文）
date: 2026-10-21
-->
<h2>見出し（h1 は自動。h2 が 3 つ以上で目次が付く）</h2>
<p>本文…</p>
```

守ること: 自分の言葉で書く／個別のレース・馬・騎手の名前を出さない／数字は `docs/html/modeling/training_pipeline_decisions.html`（決定ログ）から正しく引き、検証の期間・母数・不確かさを添える／外部サイトへのリンクを貼らない（ビルドが失敗します）／『保証ではない』の注記は自動で付く。

### 記事と素材の対応（社内文書。サイトには出ません）

| 記事 | 主な素材（`docs/html/modeling/chapters/`） |
|---|---|
| なぜ回収率100%超は難しいのか | proven_methods_roi、monthly_roi_search、research_overview |
| 期待値と回収率 | ev_calibration_adopt、ev_calibration_audit |
| 予測確率の較正 | prob_calibration_gap |
| 展開の読み方 | pace_type、tracking_difficulty_v2 |
| ラップ予想 | lap_forecast |
| 指数のノイズ | index_variance、megu_excusable |
| アンサンブル | ensemble_structure |
| 予測を検証する | drift_diagnosis、deployment_readiness |

## Cloudflare Pages へのデプロイ（あなたの操作）

最初は**ビルド済みの dist をアップロードする方法**が簡単です（Git 連携は後からでも可）。

1. ローカルで `python3 site_public/build.py --strict` を実行し、`dist/` を作る。**`--strict` が通る（＝【要入力】【要確認】が無い）ことが公開の条件。**
2. Cloudflare ダッシュボード → 左メニュー **Workers & Pages** → **Create** → タブ **Pages** → **Upload assets**。
3. プロジェクト名（例 `megu-public`）を入力 → **Create project** → `site_public/dist` フォルダをドラッグ＆ドロップ → **Deploy site**。
4. 表示された `*.pages.dev` の URL で見た目を確認する（`noindex` ではないので、確認前に URL を広めない）。
5. プロジェクトの **Custom domains** タブ → **Set up a custom domain** → `www.megu-keiba.com` を入力 → **Continue** → 内容を確認して **Activate domain**（ゾーンが Cloudflare にあるので、CNAME は自動で作られる）。
6. DNS で、ルート（`megu-keiba.com`）の A レコード（既存サイト）と `app.`・`monitor.`（Tunnel）は**触らない**。
7. **Zero Trust → Access → Applications** に、`www.megu-keiba.com` を対象にするアプリが**無い**ことを確認する（公開側は Access の対象外にする。限定側は `app.` と `monitor.` のみ）。
8. Git 連携にする場合: Create → Pages → **Connect to Git** → リポジトリを選び、**Root directory** = `site_public`、**Build command** = `python3 build.py --out dist`、**Build output directory** = `dist`。Python が使えるビルド環境か、最初のビルドログで確認する。

## 公開前チェックリスト（すべて満たすまで公開しない）

- [ ] `python3 site_public/build.py --strict` が通る（運営者情報・連絡先・所在地・相談窓口・準拠法・外部送信規律・控除率の確認が済んでいる）
- [ ] `python3 site_public/check_public_safety.py --dist site_public/dist` が OK
- [ ] **目視**: 馬名・騎手名・調教師名・レース名が出ていない（機械では判定できない）
- [ ] 記事の数字が最新の決定ログと合っている（古い数字・誇張がない）
- [ ] プライバシーポリシー・利用条件・運営者情報を**専門家が確認**した（特に、運営者の個人情報の公開範囲、外部送信規律、将来の広告）
- [ ] 公営競技の注意書き（20歳未満は購入不可、依存症の相談窓口）の連絡先が、公式の最新の案内と一致している
- [ ] データ元の利用規約を確認した（収益化の前に、取得データの利用を規約上どう位置づけるか）
- [ ] 公開側のホスト名に Cloudflare Access が掛かっていない／限定側（`app.`・`monitor.`）は掛かっている
- [ ] Cloudflare の AI クローラー設定（Block）と `robots.txt` が矛盾していない

## 広告を入れるとき（フェーズ 3。今は入れない）

1. 先に `docs/operations/ads_architecture.md` の条件（データ元の規約・広告ネットワークのポリシー・法令確認）を満たす。1 つでも否なら見送る。
2. `config.json` の `ads.enabled` を `true` にする（`ad_slot()` が広告枠の要素を出す）。**広告スクリプトを入れるには `_headers` の CSP（現在 `script-src 'none'`）の緩和が必要**で、`build.py` の `_headers` の生成部分を、ネットワークが指定するドメインだけ許可する形に直す。
3. `ads.txt` に、ネットワークから指定された行を追加する。
4. プライバシーポリシーを改訂する（取得する情報・事業者名・目的・拒否の方法。Cookie 同意の仕組みも検討）。広告であることが分かる表示を付ける。
5. 広告でレイアウトが崩れない（`.ad-slot` の高さを固定している）ことを、`--preview-ads` で確認する。

## `ads_architecture.md` との対応

| 設計書の項目 | このサイトでの実装 |
|---|---|
| 推奨構成（公開側は別オリジンの静的サイト） | `site_public/`（`frontend/` とコードも依存も分離） |
| 公開してよい範囲の線引き | `check_public_safety.py` が禁止パターンを検査（テストに入っている） |
| 広告は審査後 | `ads.enabled=false`、枠はプレビュー時のみ表示、`ads.txt` はコメントのみ |
| robots（公開側は検索を許可、AI は拒否） | `robots.txt` の生成（AI 学習・収集系は個別に Disallow） |
| 法令・表示 | プライバシー・利用条件・運営者情報・注意書きの雛形（【要確認】つき） |
| Cookie は host-only／公開側は Cookie なし | Cookie・解析・外部スクリプトなし（CSP で `script-src 'none'`） |
| Access の対象は限定側のホスト名だけ | 手順 7 のとおり、公開側は Access の対象外 |
