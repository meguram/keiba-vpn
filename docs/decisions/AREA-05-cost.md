# AREA-05: コスト設計

> **改訂**: 2026-07-03 — 実装実態に合わせて改訂

---

## 1. 現行インフラ月額（概算）

| 項目 | 月額 | 備考 |
|------|------|------|
| ConoHa VPS 2GB | ¥1,320 | 固定費・常時起動 |
| GCS（`gs://chuou/`） | ¥200〜500 | 読み取り: ローカルキャッシュで最小化 |
| UptimeRobot | ¥0 | Free プラン（5 分間隔×50 監視まで無料） |
| GitHub | ¥0 | Free プラン |
| MLflow（VPS 上 Docker） | 含む | VPS 費用に含む |
| **合計** | **¥1,520〜1,820** | |

---

## 2. GCS コスト削減方針

### L2 ディスクキャッシュ

- HybridStorage の L2 キャッシュ（`data/local/cache/`）により、GCS 読み取りを大幅削減
- 週次アクセスカウント（`disk_l2_weekly_access.json`）でアクセス頻度の低いキャッシュを自動クリーンアップ
- 効果: GCS Egress/読み取りを推定 80〜90% 削減

### `batch_list_blobs` の活用

- `date_coverage` インデックスは GCS を逐一確認せず、`batch_list_blobs` でまとめてスキャン
- `/monitor` ページの表示でも GCS API コールを最小化

---

## 3. スケールアップ判断基準

| 状況 | 対応 | 追加コスト |
|------|------|-----------|
| 同時ユーザー 50 超 | Cloud Run へ移行 | +¥0〜300/月 |
| ETL 処理 30 分超 | Cloud Run Jobs | +¥0〜300/月 |
| モデルサイズ 50MB 超 | VPS メモリ増量（4GB） | +¥660/月 |
| GPU が必要（LSTM） | Vertex AI または GPU VPS | +¥3,000〜10,000/月 |
| GCS 転送 100GB 超 | 削減戦略の見直し | +¥1,000〜/月 |

---

## 4. 推論コスト

現行: VPS 上の LightGBM（MLflow serve） — 追加費用なし（VPS 固定費に含む）

将来フェーズ別:

| フェーズ | 手段 | 月額コスト |
|--------|------|----------|
| Phase 1-3（現行） | VPS 上の `*.lgb` + MLflow | ¥0（追加なし） |
| Phase 4（LSTM） | Cloud Run（CPU）またはコンテナ | +¥0〜375/月 |
| 大規模アンサンブル | Vertex AI Batch（GPU） | +¥750〜3,000/月 |

---

## 5. 削減済みの設計決定

| 決定 | 削減効果 |
|------|---------|
| PostgreSQL を採用しない（GCS + Parquet） | DB 費用ゼロ、Supabase/CloudSQL の ~¥1,000〜5,000/月を削減 |
| Redis を採用しない（ローカルキャッシュ） | Redis Managed の ~¥500〜2,000/月を削減 |
| Vercel を採用しない（VPS 上で Jinja2） | Vercel Pro の ~¥2,500/月を削減 |
| 外部 LLM API を使用しない（セルフホスト推論） | OpenAI 等 API の変動費ゼロ |
| 独自 HTML/CSS（Next.js 不使用） | フロントエンドビルドコスト削減 |
