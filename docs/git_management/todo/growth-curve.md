# TODO: feature/growth-curve

**対象領域**: 成長曲線
**関連ドキュメント**: [../feature-growth-curve.md](../feature-growth-curve.md)
**最終更新**: 2026-09-30

<!--
  このファイルの構成:
  - 「現状の実装」は自動生成＋手動確認した内容。コードを変更したら合わせて更新する。
  - 「既知の課題」は過去の調査で判明済みの問題（無ければ空のまま）。
  - 「TODO」は自由に追記してよい。優先度・担当・期限などは各行に自由書式で書いてよい。
  - 完了したTODOは削除せず [x] にチェックして残す（履歴として）。
-->

## 現状の実装（2026-09-30時点）

- `/growth-curve`: ページ表示
- `/api/growth-curve/{horse_id}`: 馬の成長曲線データ（calculated_dataローカル優先、計算成功時は随時蓄積）。
  通常GETはローカルJSONのみ（GCS読み取りなし）。`allow_compute=true`で未計算時に1回だけ計算、
  `force_refresh=true`で再計算、`fetch_speed_index=true`でrace_index補完（GCS増）、
  `jra_only`で中央競馬/全会場切替、`limit`で件数制限
- `/api/growth-curve/status`: 成長曲線ローカルストアの件数・パス
- Flask v1 (`/api/v1/horse/<horse_id>/growth-curve`) と2026-09-30にパラメータ完全パリティ化済み
  （legacy/v1どちらも同じ`growth_curve_service.get_growth_curve`を利用。v1側の`HybridStorage()`直接生成も
  `_get_storage()`シングルトンに統一済み）

## 目標（推測）

ユーザが「この馬は成長途上か、もう完成しているか」を過去のレースぶり・タイム推移から
視覚的に把握できることが目標と推測される。特に若馬・休養明け馬の見極めに使う想定と推測される。

## このラインまで実装できたらブランチを消してよい

- ユーザが見たい馬について、ローカル未計算による「データ無し」表示に当たる頻度が実質ゼロ
  （初回アクセス時の自動計算フォールバックで十分カバーできている）
- legacy/v1どちらから見ても成長曲線データが一致している
- 上記が実現できていれば、ユーザ向けの実装は完了したとみなせる（2026-09-30時点でv1パリティ・
  HybridStorageシングルトン化は解決済みのため、実質このラインに近い状態）

## 既知の課題

（無し。2026-09-30時点でv1パリティ・HybridStorageシングルトン化ともに解決済み。`make test` 439 passed で確認）
- `/api/growth-curve/{horse_id}?fetch_speed_index=true`はrace_index補完でGCS増時にスクレイピングが
  発生し得る経路。2026-10-02決定のVPS/GCP役割分担（VPSは直接スクレイピングしない）と矛盾するため、
  GCP側（Cloud Tasks）へのジョブ委譲に変更する必要がある。詳細は
  [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)
  の「ユーザーリクエスト起点の計算処理をどこで実行するか」を参照。

## TODO（手動追記用）

<!-- 「現状の実装」と「削除してよいライン」の差分から推測したTODO。実態を確認して要不要を判断すること。 -->

### 共通TODO（ホスト方式に関係ない）

- [x] 初回アクセス時の自動計算（`allow_compute_on_miss`）のレイテンシを実測し、
      ユーザ待ち時間として許容範囲か確認する（リクエスト同期の計算であり、定期実行や
      常駐プロセスには依存しないためホスト方式に関係ない。ただしGCPサーバーレスへ移行する場合、
      コールドスタートの影響でレイテンシの絶対値が変わる可能性はある）
      — 2026-10-01対応: 衝突回避のため一時ポート（127.0.0.1:8010）でFastAPIを単独起動し、
      本番8000/5000には触れずに計測。`data/local/mirror/horse_result/` に合成の
      horse_result（12走×3頭・100走×1頭、未計算=`data/calculated_data/growth_curve/`に
      該当JSON無し）を一時配置し、`GET /api/growth-curve/{horse_id}?allow_compute=true`を
      cache miss状態で複数回・複数馬IDに対して`curl -w time_total`で実測。
      結果: 12走×15回計測で平均2.07ms・最大9.9ms（初回のみ）・それ以外は1.3〜2.2ms、
      100走1頭でも3.0〜3.7ms。cache hit（2回目以降）も1.8〜3.6msとほぼ同水準で、
      計算本体（`build_growth_curve_response`のソート・集計）はコスト上ほぼ無視できる
      （GCS/スクレイピング非経路、`fetch_speed_index=false`既定のため）。
      ユーザ待ち時間としては一般的なWebAPIの許容基準（数秒以内）を大幅に下回り問題無し
      と判断。なお本番でGCS有効時の`fetch_speed_index=true`経路や`force_refresh=true`
      （実スクレイピングを伴う）は本検証の対象外（別途ネットワーク依存のレイテンシが乗る点は
      既知の注記のまま）。検証後、合成データ・生成キャッシュ・一時サーバーはすべて削除済み。
- [x] legacy/v1で成長曲線データが完全一致することを確認する自動テストを追加する（現状は手動確認のみ）
      — 2026-10-01対応: `tests/api/test_growth_curve_legacy_v1_parity.py`を新規追加
      （既存`tests/api/test_tracking_difficulty_legacy_v1_parity.py`と同構成。
      `get_growth_curve`をモックしlegacy/v1の公開フィールド・races配列の完全一致、
      404ステータス（`no_horse_result`/`not_precomputed`）の一致、デフォルト/
      `fetch_speed_index=true`/`force_refresh=true`時の呼び出しkwargs一致を検証）。
      作成時に実際の不一致を発見・修正: `src/api/flask_app.py`の`/api/v1/horse/<id>/growth-curve`が
      `race_index_gcs`/`enqueue_missing`を`get_growth_curve`に明示的に渡しておらず、
      legacy（`src/api/app.py`）とは異なり環境変数`KEIBA_GROWTH_CURVE_RACE_INDEX_GCS`
      依存のデフォルトに落ちていた（`fetch_speed_index=true`時にlegacyはGCSのrace_index
      補完を行うがv1は行わない、という実動作の差異）。legacyと同じ導出規則
      （`race_index_gcs=fetch_speed_index`, `enqueue_missing=force_refresh`）を明示的に
      渡すよう修正し、本当の意味でのパラメータ完全パリティを確保。
      `python3 -m pytest tests/api/test_growth_curve_legacy_v1_parity.py -v`で6件すべてpass、
      `tests/api/`全体（181件）・リポジトリ全体（`--ignore=tests/scraper/manual
      --ignore=tests/research/manual`で466 passed, 5 skipped）も回帰無しを確認。

## メモ
