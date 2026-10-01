# TODO: feature/track-speed

**対象領域**: トラックスピード指標
**関連ドキュメント**: [../feature-track-speed.md](../feature-track-speed.md)
**最終更新**: 2026-09-30

<!--
  このファイルの構成:
  - 「現状の実装」は自動生成＋手動確認した内容。コードを変更したら合わせて更新する。
  - 「既知の課題」は過去の調査で判明済みの問題（無ければ空のまま）。
  - 「TODO」は自由に追記してよい。優先度・担当・期限などは各行に自由書式で書いてよい。
  - 完了したTODOは削除せず [x] にチェックして残す（履歴として）。
-->

## 現状の実装（2026-09-30時点）

- ページ: `/track-speed`・`/track-speed/dev`（開発用、`/login`必須のリダイレクト付き）
- `/api/track-speed/meta`: baseline/pace_baseline生成状態、日付数、会場一覧
- `/api/track-speed/dates`（会場フィルタ可）・`/api/track-speed/venues`（日付指定）: 集計済み日付・会場の一覧
- `/api/track-speed/day`: 指定日・会場のトラックスピードデータ取得（`track_speed_engine.query_day`）
- `/api/track-speed/status`: baseline再構築ジョブの進行状況・ready状態
- `POST /api/track-speed/rebuild-baselines`: ベースライン再構築をバックグラウンドスレッドで開始
- `POST /api/track-speed/assign`: 指定期間のレースにperf_index割当を実行
- `/api/track-speed/validate-perf`: レースパフォーマンス指数のバリデーション実行
- `/api/track-speed/race-horses`: レースの全馬にperf_indexと速度水準ラベルを付与（2着馬がrace_perfと同値）
- `/api/track-speed/by-category`: カテゴリ別集計

## 目標（推測）

ユーザが「このレースのタイムは早い/遅いのが馬場のせいか、馬の実力か」を切り分けて評価できる
よう、馬場差を補正したperf_index（スピード指標）を提供することが目標と推測される。

## このラインまで実装できたらブランチを消してよい

- 対象会場・条件で baseline が定期的に再構築され、perf_index が古いまま使われる事象が無い
- レース単位でユーザが見たときに、全馬にperf_indexが欠損なく付与されている
- 上記が実現できていれば、ユーザ向けの実装は完了したとみなせる

## 既知の課題

- 下記TODOの「`rebuild-baselines`の定期実行有無の確認」は、VPSなら既存のOS crontab方式に
  乗せればよい。GCPへ移行する場合も、Compute Engineでのリフト&シフトなら同方式を継続できるが、
  Cloud Run等のサーバーレス構成を選ぶ場合はCloud Scheduler+Cloud Run Jobsでの実装が前提になる
  （採用するGCPサービスにより対応が変わる）。詳細は
  [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)。
  現時点ではVPS運用継続が方針のため、本ファイルのTODOはそのまま進めてよい。

## TODO（手動追記用）

<!-- 「現状の実装」と「削除してよいライン」の差分から推測したTODO。実態を確認して要不要を判断すること。 -->

### 共通TODO（ホスト方式に関係ない）

- [x] `perf_index`が付与されていない馬・レースの割合を計測し、`assign`の対象範囲に漏れが無いか確認する
      — 2026-10-01対応: 本サンドボックスには実データが一切存在しない（`data/`は4.5MBのみで`race_result`等のparquetが無く、`GCS_BUCKET`未設定、稼働中のAPIも`GET /api/track-speed/meta`が`baselines_ready:false, dates_count:0`を返す）ため、実測によるパーセンテージ計測は不可能だった。代わりにコードを精査し、構造的な漏れを特定した。
      **主要な漏れ（高確度）**: ベースライン学習期間は`build_track_speed_baselines.py`既定で2020-2025年（改修後）だが、`perf_index`付与（`assign_races`）を呼ぶ経路は①CLI `assign_track_speed.py`（`--date-from`既定`2026-01-01`）と②`/track-speed`画面の「2026〜振り分け」ボタン（`templates/analysis/track_speed.html:1169` `fetch('/api/track-speed/assign?date_from=2026-01-01', ...)`、`date_to`無し）の2つのみで、両方とも2026-01-01以降しか対象にしない。さらに`scripts/`配下にこれを定期実行するcron/daemonは存在せず（`grep -rl assign_track_speed scripts/`で0件）、手動クリック以外に実行経路が無い。したがって2020-2025年の過去レースはベースライン計算の母集団にのみ使われ、`perf_index`自体は（過去に誰かが`--date-from 2020-01-01`等で手動実行していない限り）1件も付与されていない可能性が高い。これは本ファイル「目標（推測）」の『全馬にperf_index欠損なく付与』という前提と、実装が暗黙的に『2026年以降のレースのみ』にスコープを絞っている点の不整合であり、意図的な設計（2026年以降のみをユーザ提供対象とする）か見落としかは要件側で確認が必要。
      **その他の除外要因（`src/research/race/track_speed_engine.py`）**: `load_races_from_parquet`（L823-905）はJRA中央競馬場・芝/ダートのみに絞り障害レースと改修日以前のレースを除外（意図的）。`score_race`（L1077-1216）は`track_condition`が`COND_CANDIDATES={良,稍重,重,不良}`（L65-70）以外の場合、および対象venue×layout×surface×distance×class_band×cond_pool（ALL会場フォールバック・クラス±2ランクフォールバック含む）に`MIN_BASELINE_N=12`件以上のベースラインが無い場合、`None`を返しそのレースを`assign`対象から静かに除外する。両ケースとも同じ`None`返却のため、ログ上で「track_condition不正」と「ベースライン不足」を区別できない。
      **フォローアップ**: 正確な付与率・除外件数は実データにアクセスできるホスト（本番VPS/GCS接続環境）で `load_races_from_parquet()` の母集団件数と `assign_races()` 内で `score_race()` が `None` を返した件数を比較するログを仕込んで計測するのが望ましい（本対応では大規模バッチ実行はしない方針のため見送り）。
- [x] `/track-speed/dev`（開発用ページ、ログイン必須）の役割を整理し、本番ページとの差異を明記する
      — 2026-10-01対応: `src/api/app.py`のルーティング（L12098-12117）と両テンプレートを比較した。
      **`/track-speed`（本番）**: 認証チェック無し（誰でもアクセス可、`track_speed_page`に`is_developer`判定は無い）。インタラクティブな馬場速度ダッシュボード本体で、`templates/analysis/track_speed.html`は`fetch()`を10箇所で呼び`/api/track-speed/meta|dates|venues|day|status|rebuild-baselines|assign|validate-perf|race-horses|by-category`を網羅する。加えて運用操作ボタン「基準データ再構築」（`POST /api/track-speed/rebuild-baselines`）と「2026〜振り分け」（`POST /api/track-speed/assign?date_from=2026-01-01`）がページ上に直接存在し（L419-420）、これらのボタン自体・対応するPOSTハンドラ（`src/api/app.py` L12193, L12299）にも認証チェックが無いため非ログインユーザでも実行トリガーできる。開発者ログイン済みの場合のみ、JS側で`/api/auth/status`の`is_developer`を見て右下に`/track-speed/dev`への浮動リンクを動的追加する（`track_speed.html:1300-1324`）。
      **`/track-speed/dev`（開発用）**: `is_developer`でなければ`/login?next=/track-speed/dev`へリダイレクト（`src/api/app.py:12107-12112`）、開発者専用。`templates/analysis/track_speed_dev.html`は`fetch()`呼び出しが0件の完全な静的解説ページで、日次データ表示・運用操作ボタンを一切持たない。内容はパイプライン概要図とStep1〜5（ペース特徴量抽出→OLS補正→ベースラインZ→テンポラルプーリング→PF指数+速度水準ラベル）の計算ロジック解説、数式・サンプル値、データアーティファクト一覧、設計メモ/制約のドキュメントのみ。
      **差異まとめ**: 想定役割は「本番＝データ閲覧＋運用操作（無認証）」 vs 「dev＝アルゴリズム解説専用の静的ドキュメント（開発者限定）」。ただし運用操作ボタン（rebuild-baselines/assign）が無認証の本番ページ側に置かれている点は、ページの役割分担の前提（本番=閲覧用、dev=開発者専用）とは矛盾しており、別途セキュリティ観点のTODO化を検討する価値がある（本TODOの対応範囲外のため指摘のみ）。

### 常時稼働ホスト（VPS / GCP Compute Engine）の場合のTODO

- [ ] `POST /api/track-speed/rebuild-baselines` の実行トリガーが手動のみか、定期実行があるかを
      現行のOS crontab/daemon thread方式を前提に確認する

### GCPサーバーレス（Cloud Run等）移行時のTODO

- [ ] 上記の定期実行確認・整備は、サーバーレス移行する場合はCloud Scheduler+Cloud Run Jobsの
      実行ログを前提にした確認に置き換わる。詳細は
      [`docs/operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)

## メモ
