# 匿名向け応答の CDN キャッシュ（費用が増えない閲覧集中対策）

閲覧者が 1,000〜10,000 人になっても Cloud Run を増やさない（費用が増えない）ための仕組み。仕様の位置づけ: `docs/html/operations/production_load.html` §7.3。

## オリジン側（実装済み）

- `src/api/http_cache_headers.py` / `config/http_cache.json`。ルールに `s_maxage` があるパスで、**Cookie も Authorization も無い匿名リクエスト**の 200 応答だけ `Cache-Control: public, max-age=N, s-maxage=S` にする。
- Cookie / Authorization があるとき、dev の自動ログインのとき、`s_maxage` の無いルールは従来どおり `private, max-age=N`（会員の応答は共有キャッシュに載らない）。
- 会員限定パスの匿名応答は 401 なので、200 のみ対象のこの仕組みでは載らない。
- 止めるときは `config/http_cache.json` の該当ルールから `s_maxage` を消す（30 秒で反映）か `enabled=false`。

| パス | max-age | s-maxage |
|------|--------|----------|
| `/api/race-calendar` `/api/scrape-dates` `/api/megu-index-dates` | 300 | 120 |
| `/api/race-list/` | 60 | 60 |
| `/api/v1/races/` `/api/race/`（AI予測・出馬表・馬カード） | 60 / 30 | 30 |
| `/api/horse/` `/api/v1/horse/` | 300 | 120 |

## Cloudflare 側（ダッシュボードでの設定が必要）

オリジンが `public` を返しても、Cloudflare は拡張子の無い `/api/...` を既定ではキャッシュしない。次のキャッシュルールが要る（無料プランで可）。

1. Caching → Cache Rules → 新規ルール。
2. 条件: URI パスが `/api/` で始まる **かつ** Cookie ヘッダが空（`not http.cookie contains "keiba_dev_session"` ではなく、**Cookie が一切無い**ことを条件にする。オリジンも Cookie があれば private にするため）。
3. 動作: キャッシュ対象（Eligible for cache）、Edge TTL は「オリジンの Cache-Control を使う」。
4. Cookie がある要求はルールに合致しないのでキャッシュを素通りする（会員の応答が混ざらない）。

注意: Cloudflare Access（現在の公開経路）の内側ではまだ匿名の一般公開をしていない。public モードへ切り替えるときに合わせて有効にする。確認は `curl -sI https://<host>/api/race/<race_id>/predictions` を 2 回続けて打ち、`cf-cache-status: HIT` になるか見る。

## 画面の遅延呼び出し

推奨馬券（`/api/v1/betting/optimize`）は「推奨馬券」タブを開いたときだけ呼ぶ（`frontend/app/weekly-predictions/page.tsx` の `tab === "bet"`）。レース質（開催日）は専用画面だけで呼ぶ。

## VPS の nginx キャッシュと事前ロード（自動）

- 匿名・許可リストの応答は VPS ローカルの nginx `proxy_cache`（`/var/cache/nginx/keiba_api`、`max_size=1g inactive=35d`）に置く。GCS は読まない。確定済み（3 日以上前）のレースは `s-maxage` 30 日、開催前後・一覧・カレンダーは数十〜120 秒（`config/http_cache.json`）。
- 参照用データが作られた・更新されたとき、書き手（`HybridStorage.save` / `gcs_mirror.push` / `cache_epoch.bump`）が GCS に `ops/cache_dirty/<日付>/<race_id|開催日|_ALL>` の目印を書く（`src/utils/cache_marker.py`。`KEIBA_CACHE_MARKER=0` で停止）。
- VPS の `keiba-cache-warm.timer`（60 秒おき、`scripts/ops/vps/cache_warm.py`）が目印を list 1 回だけで拾い（月 約 0.2 ドル）、対象のエンドポイントを `X-Keiba-Refresh` 付きでローカル nginx から取り直して入れ直す。出馬表カード（private）は対象外。
- 導入・撤去: `sudo bash scripts/ops/vps/vps_10_cache_warm.sh [--dry-run|--remove]`（前提は `vps_09_api_cache.sh` と pull 用 venv）。確認: `journalctl -u keiba-cache-warm -n 5`。詳細は `docs/html/operations/vps_load_and_cache.html` §4-4。
