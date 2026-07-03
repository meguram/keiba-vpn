# AREA-03 — バックエンド要件

**Status**: FINAL  
**Last Updated**: 2026-07-03  
**Consolidates**: DEC-001(技術選定), DEC-003(サーバ負荷), DEC-007(ADR-001), DEC-008(確定), DEC-009(実装詳細)

---

## 1. 技術選定

| 項目 | 決定内容 | 根拠 |
|---|---|---|
| 言語 | Python 3.11 | AI/ML ライブラリとの親和性 |
| フレームワーク | Flask 3.x | 既存コードベース、軽量、VPS 2GB 制約 |
| WSGI | Gunicorn 21.x（worker_class=gthread, workers=2, threads=4, preload_app=True） | CoW でモデル共有、メモリ節約 |
| リバースプロキシ | Nginx（静的ファイル直配信・keepalive・gzip） | |
| 移行トリガー | 同時接続 500+ で FastAPI への移行を検討（ADR-001） | |

> **DEC-001 矛盾解消**: DEC-001 は FastAPI を選定したが DEC-008 で Flask 確定。FastAPI は 500+ 同時接続時に再検討。

---

## 2. Gunicorn 設定

```python
# gunicorn.conf.py
worker_class       = "gthread"
workers            = 2
threads            = 4
preload_app        = True      # LightGBM Booster を CoW で worker 間共有
max_requests       = 500
max_requests_jitter= 50
bind               = "unix:/tmp/gunicorn.sock"
```

---

## 3. 認証・認可

| 項目 | 決定内容 |
|---|---|
| 認証方式 | Flask-Login セッション認証 |
| ログイン方法 | メール + パスワード（Phase 2 で Google OAuth 追加） |
| Cookie 設定 | `HttpOnly=True`, `Secure=True`, `SameSite=Lax` |
| CSRF 保護 | Flask-WTF CSRF トークン（状態変更リクエスト） |
| パスワード保存 | bcrypt ハッシュ |
| ログイン失敗制限 | 5 回失敗で 15 分ロックアウト |
| VPN | WireGuard（管理 API のアクセス制御）|

### エンドポイント認証要件

| エンドポイント | 認証要否 | ゲスト挙動 |
|---|---|---|
| GET `/health`, `/api/v1/health` | No | — |
| GET `/login`, POST `/login` | No | — |
| POST `/logout` | Yes | 401 |
| GET `/api/v1/races/today` | No | 閲覧可 |
| GET `/api/v1/predictions/{race_id}` | No | TOP3 のみ |
| GET `/api/v1/races/{race_id}/entries` | Yes | 401 |
| POST `/api/v1/filter/stats` | Yes | 401 |
| GET `/api/v1/sires/ranking`, `/api/v1/shap/{race_id}`, `/api/v1/horses/{id}` | Yes | 401 |
| GET `/admin/circuit-status` | Yes（IP 制限） | 403 |

---

## 4. 4 層キャッシュ設計

```
L1: Flask lru_cache   — 60s per-worker（プロセス内、共有なし）
L2: Redis 7           — レース一覧 TTL 10min / 予測 60min / オッズ 1min / 結果 24h
L3: PostgreSQL        — race_cache テーブル（ETL 更新まで有効）
L4: GCS               — フォールバックのみ（503 回避用）
```

### Thundering Herd 防止

```python
# Redis SET NX ミューテックスロック
lock_key = f"lock:pred:{race_id}"
if redis.set(lock_key, "1", nx=True, ex=10):
    # キャッシュ再計算
    ...
else:
    # ロック待機
    time.sleep(0.1)
```

### Redis TTL 一覧

| キー | TTL |
|---|---|
| `pred:{race_id}` | 60 分（レース開始 +30 分で auto-expire） |
| `races:today` | 10 分 |
| `odds:{race_id}` | 1 分 |
| `results:{race_id}` | 24 時間 |
| `etl:complete:{race_date}` | 3,600 秒 |

---

## 5. DB スキーマ（主要テーブル）

```sql
CREATE TABLE races (
  race_id     VARCHAR(20) PRIMARY KEY,
  race_date   DATE NOT NULL,
  venue       VARCHAR(10),
  course      VARCHAR(10),
  distance    INTEGER CHECK (distance BETWEEN 800 AND 3600),
  grade       VARCHAR(5)
);

CREATE TABLE horses (
  horse_id    VARCHAR(20) PRIMARY KEY,
  name        VARCHAR(50) NOT NULL,
  sire_id     VARCHAR(20),   -- 種牡馬ランキング用
  birth_year  INTEGER
);

CREATE TABLE entries (
  race_id     VARCHAR(20) REFERENCES races,
  horse_id    VARCHAR(20) REFERENCES horses,
  post_pos    INTEGER,
  weight      INTEGER CHECK (weight BETWEEN 380 AND 620),
  PRIMARY KEY (race_id, horse_id)
);

CREATE TABLE results (
  race_id     VARCHAR(20) REFERENCES races,
  horse_id    VARCHAR(20) REFERENCES horses,
  finish_pos  INTEGER,
  PRIMARY KEY (race_id, horse_id)
);

CREATE TABLE predictions (
  race_id          VARCHAR(20) REFERENCES races,
  horse_id         VARCHAR(20) REFERENCES horses,
  win_probability  REAL NOT NULL,
  confidence_score INTEGER CHECK (confidence_score BETWEEN 0 AND 100),
  inference_source VARCHAR(10) CHECK (inference_source IN ('local', 'external')),
  model_version    VARCHAR(10),
  created_at       TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (race_id, horse_id)
);

CREATE TABLE race_cache (
  race_id     VARCHAR(20) PRIMARY KEY,
  race_date   DATE NOT NULL,
  predictions JSONB NOT NULL,
  etl_version INTEGER NOT NULL DEFAULT 1,
  cached_at   TIMESTAMPTZ DEFAULT NOW(),
  expires_at  TIMESTAMPTZ NOT NULL
);
CREATE INDEX ON race_cache (race_date);
```

---

## 6. レート制限

| 対象 | 制限 |
|---|---|
| 全 API（IP ベース） | 60 req/min/IP |
| 分析 API（ユーザーベース） | 20 req/min/user |
| `/api/v1/filter/stats`（重い集計） | 10 req/min/user |

---

## 7. パフォーマンス目標

| 項目 | 目標値 |
|---|---|
| Redis HIT 時 P50 | ≤50ms |
| Redis HIT 時 P99 | ≤200ms |
| GCS フォールバック P99 | ≤1,000ms |
| E2E P95 | ≤1,200ms |
| 未認証エンドポイント | ≤100ms |
