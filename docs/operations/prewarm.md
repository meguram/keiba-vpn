# 画面の応答の事前生成（prewarm）

## 目的
今週のレースの AI 予測など、ユーザーが訪れやすい画面の応答を、アクセスされる前に作ってディスクに置く。
ユーザーのアクセスはそれを読むだけになり、表示までの待ちが減る（dev 計測: 予測 API 約 0.12〜0.22 秒 → 約 0.002 秒）。

## 有効化
`.env` に `KEIBA_PREWARM=1`（既定オフ。stg/prod の API と Flask の両方に同じ .env が読まれること）。オフのときは従来どおりその場で計算する。

## 仕組み
- 置き場: `data/local/cache/prerender/<kind>/<key>.json`（`KEIBA_PRERENDER_DIR`。遅いディスクなら速いローカルへ）。`src/api/prerender.py`。
  kind は `predictions`・`race_detail`・`race_list`・`race_values`（Flask のレース選択の期待値表示）。
- 置くのは匿名/会員の出し分け前の全項目の版。取り出した後に各 API が従来どおり項目を落とす（`src/api/tier.py`）。
- **読まれなくなる条件**（その場計算に戻る）: ①期限切れ ②保存以外の世代（配備・設定変更・手動 `clear_caches`。`cache_epoch.hard_current`）が進んだ
  ③応答が依存するレース/開催日が、応答を作り始めたあとに更新された（**レース単位**の目印）。
  依存は、予測・詳細＝そのレース、一覧・期待値表示＝開催日と全レース。
- **変更の目印の付き方**
  - `HybridStorage.save`: キーが race_id（12桁）/開催日（8桁）の保存（今年・昨年分）。出馬表の再取得・オッズ・結果・馬体重・`race_lists` など全カテゴリ。
  - ファイル出力: 予測（`predictions_v6`/`predictions_v3`/`predictions_backfill`）・追走難度・ラップ予想・出馬表の状態（`entries_state`）。
    作る側が 15 秒ごとに直近の開催日のファイルの更新時刻を見て拾う（書く側の改修は不要）。
  - 手で付ける: `python3 -c "from src.api import prerender; prerender.mark_dirty(['<race_id>'])"`（`KEIBA_PREWARM=1` が必要）。
- **作る側**: `src/api/prewarm.py`。API(FastAPI) の起動時に始まり、`data/local/meta/prewarm.lock` を取れた 1 ワーカーだけが動く（落ちたら別ワーカーが引き継ぐ）。
  - **変更に反応**: 目印が付いたら 3 秒まとめてから、そのレースの予測・詳細を作り直し、続けてその開催日の一覧と期待値表示（paid/full）を作り直す。
    開催日の目印（`race_lists` の更新）は、一覧を作り直して増えたレースも拾う。窓の外のレース（過去分の補填）は無視。
  - **周回（安全網）**: 60 秒ごと、および配備などで世代が進んだとき。窓内の全開催日を見て、無効・期限の半分を過ぎたものだけ作り直す。
  - 対象: `config/prewarm.json` の窓（既定 過去 14 日〜先 7 日の開催日）。hot＝これからの開催日＋直近 3 日（期限 300 秒）、cold＝それより前（6 時間）。
  - ホストの負荷が高い（1 分平均 > コア数×0.5）周回は見送る。目印のファイルは 3 日で掃除。
  - 状況: `data/local/meta/prewarm_status.json`（周回・変更ごとの件数・作り直し数・エラー数）。

## 既知の限界
- 目印は同じホスト（同じファイルシステム）で保存した分だけ見える。Cloud Run など、保存する側（Jobs）と API が別の場所の環境では、
  変更への反応は効かず、周回と期限（hot 300 秒）で追従する。必要なら API 経由で目印を付ける口を足す。
- 馬単位のデータ（`horse_*` の更新）は目印にしない（馬から全レースへの広がりが大きいため）。期限（hot 300 秒）と周回で追従する。
- 開催日にオッズの保存が続くと、その開催日の期待値表示（全レースの最適化）を頻繁に作り直す。
- 事前生成するのは既定の条件（`odds_mode=auto`・購入上限の既定）の期待値表示だけ。条件を変えたときはその場計算。
- ブラウザ側の先読み（`frontend/lib/httpCache.ts`）は従来どおりホバー時のみ。

## 確認
- `python3 -m pytest tests/api/test_prerender_prewarm.py`
- 動作確認: `KEIBA_PREWARM=1` で API を起動し、`prewarm_status.json` と `data/local/cache/prerender/` を見る。
