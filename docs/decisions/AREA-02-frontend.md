# AREA-02: フロントエンド設計

> **改訂**: 2026-07-03 — 実装実態に合わせて全面改訂

---

## 1. 技術スタック

| 項目 | 採用 | 備考 |
|------|------|------|
| **テンプレートエンジン** | Jinja2 | FastAPI の `TemplateResponse` で描画 |
| **スタイル** | 独自 CSS | `static/css/` |
| **JavaScript** | バニラ JS (ES6+) | インライン `<script>` またはスクリプトタグ |
| **チャート** | 必要に応じてインラインSVG/Chart.js | ページ個別 |
| **ホスティング** | VPS 上の FastAPI が直接サーブ | Nginx リバースプロキシ経由 |

### テンプレート構造

```
templates/
├── admin/
│   ├── monitor.html        ← スクレイプ監視・GCS マトリクス
│   ├── scrape.html         ← 手動スクレイプ操作
│   ├── scrape_control.html ← キュー制御
│   ├── queue_status.html   ← キューステータス
│   └── cron_jobs.html      ← cron 一覧
├── data/
│   └── data_viewer.html    ← データビューア
└── index.html              ← TOP ページ
```

---

## 2. 主要ページ仕様

### 2-1. `/monitor` — スクレイピング監視

**目的**: GCS データ存在マトリクスでスクレイプ状況を可視化し、欠損データを UI から再取得できる。

**主要コンポーネント**:

1. **カバレッジカレンダー**: 月単位でカバレッジ率をヒートマップ表示
   - API: `GET /api/coverage-calendar?month=YYYY-MM`

2. **GCS データ存在マトリクス**:
   - 縦軸: 当日レース（レース番号 + 競馬場）
   - 横軸: GCS カテゴリ（19 カテゴリ）
   - セル値: `✓`（データあり）/ `✗`（未取得）/ `−`（取得不可）
   - API: `GET /api/date-race-matrix?date=YYYYMMDD`

3. **スクレイプアクションバー**:
   - カラム（カテゴリ）を選択→「欠損のみスクレイプ」ボタンが表示
   - API: `POST /api/scrape-missing` にカテゴリ・日付・race_id リストを送信

**マトリクス表示仕様**:

| 状態 | 表示 | 背景色 | 意味 |
|------|------|--------|------|
| `true` | `✓` | 緑 | GCS にデータあり |
| `false` | `✗` | 赤 | 未取得または取得失敗 |
| `null` | `−` | グレー | 取得不可（N/A マーカー） |

**カラムラベル (`SHORT_LABEL`)**:

```javascript
const SHORT_LABEL = {
  "race_shutuba":               "出馬表",
  "race_shutuba_meta":          "出馬表M",
  "race_index":                 "指数",
  "race_paddock":               "P",
  "race_odds":                  "オッズ",
  "race_result_on_time":        "速報",
  "race_result_on_time_payoff": "速報払戻",
  "race_result_on_time_lap":    "速報L",
  "race_result_on_time_corner": "速報C",
  "race_result":                "結果",
  "race_result_meta":           "結果M",
  "race_result_payoff":         "払戻",
  "race_result_track":          "馬場",
  "race_result_corner":         "コーナー",
  "race_result_lap_times":      "ラップT",
  "race_result_lap":            "ラップ",
  "race_barometer":             "走行指数",
};
```

---

### 2-2. `/queue-status` — キューステータス

**目的**: バックグラウンドスクレイプキューの状態確認・操作。

**表示項目**:
- 全体統計: `pending` / `running` / `completed` / `failed` 件数
- ジョブ一覧テーブル（`job_id`, `job_kind`, `target_id`, `tasks`, `status`, `created_at`）
- `schema_validation_failures` カウント

**操作**:
- 失敗ジョブの再キュー（`POST /api/scrape-queue/failed/requeue`）
- キュー一時停止・再開（`POST /api/scrape-queue/stop-and-clear`）
- キューキック（`POST /api/scrape-queue/kick`）

---

### 2-3. `/scrape-control` — スクレイプ制御

**目的**: 手動でスクレイプジョブを投入・確認する。

**機能**:
- レース ID / 馬 ID / 日付を入力してジョブ投入
- タスク種別チェックボックス（出馬表・結果・オッズ等）
- `smart_skip` オプション（既存データをスキップするか）

---

### 2-4. `/cron-jobs` — cron ジョブ管理

**表示項目**:
- 現在登録の cron エントリ一覧（`/api/admin/cron-jobs` から取得）
- JST 換算時刻・タスク名・最終実行ログ

---

## 3. API データ型（フロントエンド視点）

### `GET /api/date-race-matrix`

```typescript
// レスポンス例
{
  "date": "20260615",
  "categories": ["race_shutuba", "race_odds", ...],  // 19 カテゴリ
  "races": [
    {
      "race_id": "202605010101",
      "race_name": "1R",
      "venue": "東京",
      "coverage": {
        "race_shutuba": true,
        "race_odds": true,
        "race_barometer": null,  // N/A
        "race_result": false,    // 未取得
        ...
      }
    },
    ...
  ]
}
```

### `POST /api/scrape-missing`

```typescript
// リクエスト
{
  "date": "20260615",
  "category": "race_result",
  "race_ids": ["202605010101", "202605010102"]  // 省略時は全レース
}

// レスポンス
{
  "queued": 12,
  "skipped": 3,
  "message": "12 件のジョブを投入しました"
}
```

---

## 4. 性能要件

| ページ | 目標 |
|--------|------|
| `/monitor` マトリクス描画 | ≤3 秒（API 取得含む） |
| `/queue-status` 更新 | ≤1 秒（ポーリング間隔 5 秒） |
| API キャッシュヒット | ≤200ms |
| API キャッシュミス | ≤2000ms |

---

## 5. エラー表示

- API エラー時: ページ内トースト通知（赤背景）
- N/A データ: `−` セル + ツールチップ「このデータは取得対象外です」
- ネットワークエラー: 「サーバーに接続できません」バナー表示
