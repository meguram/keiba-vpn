# AREA-06 — データ管理要件

**Status**: FINAL  
**Last Updated**: 2026-07-03  
**Consolidates**: DEC-002(旧3層構成→廃止), DEC-005(DP要件), DEC-009(ETL), DEC-010(GCS設計/Feature Store)

---

## 1. ストレージ構成

| ストレージ | 役割 | 備考 |
|---|---|---|
| VPS ローカル `/data`（SSD 100GB） | 一時保存、PostgreSQL データファイル | プライマリ |
| GCS `keiba-vpn-data` | Parquet バックアップ、モデル保管、生データ | 長期保存 / フォールバック |
| Redis 7（VPS 内） | キャッシュ（予測結果・レース一覧） | TTL 管理 |
| PostgreSQL（VPS 内） | ユーザー/auth、race_cache、races/horses 等 | |

> **旧 DEC-002 廃止**: Cloud SQL + BigQuery の 3 層構成は DEC-008 で VPS ローカル + GCS のみに集約。BigQuery は月次分析が必要になった場合のみ追加（ADR-002）。

---

## 2. GCS パス管理（SSoT）

**ファイル**: `keiba-vpn/src/scraper/gcs_paths.py`（Single Source of Truth）

```python
GCS_BUCKET = "keiba-vpn-data"  # 環境変数で上書き可

# 生データ
RAW_RACE_CARD_PREFIX  = "raw/race_card/{date}/"
RAW_ODDS_PREFIX       = "raw/odds/{date}/{race_id}/"

# 特徴量
FEATURE_STATIC_PREFIX  = "features/static/dt={date}/"
FEATURE_DYNAMIC_PREFIX = "features/dynamic/dt={date}/race_id={race_id}/"

# モデル
MODEL_PREFIX        = "models/v{version}/"
MODEL_CURRENT_PATH  = "models/current/model.lgb"
MODEL_ROLLBACK_PATH = "models/v{version}/model.lgb"
```

**運用ルール**:
- 全コンポーネント（ETL / Feature Store / Cloud Run Jobs / 推論ワーカー / Flask API）はここからインポート
- ハードコードされたパス文字列は **CI lint で禁止**（違反 → ビルド失敗）
- `test_gcs_paths.py` で全パスの形式を自動検証

---

## 3. GCS ライフサイクルポリシー

| データ種別 | 保持期間 |
|---|---|
| 生データ（`raw/`） | 7 日 |
| Parquet（`features/`） | 90 日 |
| モデル（`models/v{N-2}` 以前） | 削除（N-1 は保持） |

---

## 4. ETL パイプライン

```
cron 06:00 JST（keiba-scraper.service）
  │
  ├─ JRA サイトスクレイピング（requests + BeautifulSoup4）
  │   tenacity リトライ（4s→8s→60s, max 3）
  │   pybreaker Circuit Breaker（5 失敗→OPEN）
  │
  ├─ 生データ → GCS raw/race_card/{date}/ に保存
  │
  ├─ 特徴量変換 → Parquet → GCS features/static/, features/dynamic/
  │
  ├─ PostgreSQL 更新（races / horses / entries テーブル）
  │
  ├─ race_cache テーブル更新
  │
  └─ Redis key: etl:complete:{race_date} をセット（TTL 3,600s）

cron 12:00 JST（追加オッズ取得）
  └─ オッズ差分 → GCS raw/odds/{date}/{race_id}/
     → 差分再推論トリガー（±15% 変動時のみ）
```

---

## 5. Feature Store 設計

| 種別 | 更新頻度 | GCS パス | 内容例 |
|---|---|---|---|
| 静的特徴量 | 週次（日曜夜） | `features/static/dt={date}/` | 血統、コース別成績、騎手統計 |
| 動的特徴量 | レース 30 分前 | `features/dynamic/dt={date}/race_id={race_id}/` | 直前オッズ、馬体重、馬場状態 |

**スキーマオーナーシップ**: data-engineer チーム

---

## 6. Redis データ管理

| キー | TTL | 用途 |
|---|---|---|
| `pred:{race_id}` | レース開始 +30min | 予測結果キャッシュ（L2）|
| `races:today` | 10 分 | レース一覧 |
| `odds:{race_id}` | 1 分 | 直前オッズ |
| `results:{race_id}` | 24 時間 | レース結果 |
| `etl:complete:{race_date}` | 3,600 秒 | 推論バッチの起動フラグ |
| `race_data_ready:{race_id}` | 当日 | ETL 完了通知 |

maxmemory-policy: **allkeys-lru**（256MB 上限超過時に古いキーを自動削除）

---

## 7. 非機能要件（データ系）

| ID | 要件 |
|---|---|
| NFR-DATA-01 | GCS 読み取りリクエスト削減 ≥95%（Redis 導入後比） |
| NFR-DATA-02 | Redis キャッシュヒット率 ≥90%（レース日） |
| NFR-DATA-03 | GCS パスハードコード禁止（CI lint 強制） |
| NFR-DATA-04 | ETL 完了: レース当日 06:00 JST まで |
| NFR-DATA-05 | スクレイピング成功率 ≥95% |
| NFR-DATA-06 | GCS signed URL: 認証済みユーザーのみ、TTL 1h、パストラバーサル保護 |
