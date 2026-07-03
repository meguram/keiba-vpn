# AREA-01 — アプリケーション要件

**Status**: FINAL  
**Last Updated**: 2026-07-03  
**Consolidates**: DEC-005(要件 v1.0), DEC-008(要件確定), DEC-010(F-01~F-26)

---

## 1. プロジェクト概要

keiba-vpn は日本競馬（JRA）のレースデータをスクレイピングし、LightGBM モデルで予測スコアを生成してマルチユーザーに提供する競馬予測・データ分析 Web アプリケーション。

| 項目 | 内容 |
|---|---|
| 提供価値 | AI 予測スコア付き出馬表・条件別統計分析・コース別種牡馬ランキング |
| ターゲットユーザー | 競馬ファン（登録ユーザー） + 未登録ゲスト（閲覧のみ） |
| インフラ制約 | ConoHa VPS 2GB/100GB SSD（契約済み固定費）|
| 設計優先度 | ① VPS 2GB 制約内での安定稼働 ② スクレイプ・推論・API の完全プロセス分離 ③ コスト固定化 |

---

## 2. ユーザー機能一覧

### ゲスト（未ログイン）

| 機能 | 説明 |
|---|---|
| 今日のレース一覧 | 開催場・レース番号・発走時刻を閲覧 |
| 予測スコア（TOP3 のみ） | 上位 3 頭の AI 予測スコアのみ表示 |

### 登録ユーザー（ログイン済み）

| 機能 | ID | 説明 |
|---|---|---|
| 今日のレース一覧 | US-03 | 全レース閲覧 |
| AI 予測スコア（全頭） | PM-01〜PM-10 | 全頭の勝率・信頼スコア・SHAP 説明 |
| 条件別統計分析 | F-1, F-2 | 単勝率・ROI、馬券種別収支 |
| コース別種牡馬ランキング | F-3 | sire_id 紐付きランキング |
| AND 条件フィルタ | F-4 | 複合条件での馬検索 |
| データ鮮度確認 | F-5 | FreshnessStatus バナー |
| 予測履歴 | US-04 | 直近 30 日間 |
| 追いかけ馬通知 | US-05 | 最大 10 頭、出走時プッシュ通知 |

---

## 3. データフロー（レース当日）

```
06:00 JST  スクレイパー → GCS raw/race_card/{date}/
           ETL 完了フラグ: Redis etl:complete:{race_date}
08:00 JST  特徴量変換 → GCS features/
08:10 JST  LightGBM バッチ推論 → Redis + PostgreSQL predictions
12:00 JST  オッズ追加取得 → 差分再推論（±15% 変動時のみ）
レース後   結果スクレイピング → GCS results/
毎週月曜   Cloud Run Jobs モデル再学習
```

---

## 4. 機能要件

### 4-A. データパイプライン（DP-01〜07）

| ID | 要件 |
|---|---|
| DP-01 | 出馬表データをレース当日 06:00 までに GCS 取込完了 |
| DP-02 | スクレイピング失敗時: 指数バックオフ最大 3 回リトライ（4s→8s→60s） |
| DP-03 | 全リトライ失敗時: GCS 前回データにフォールバック + `data_status: stale` 付与 |
| DP-04 | ETL 完了後 Redis に `etl:complete:{race_date}`（TTL 3600s）をセット |
| DP-05 | ETL 失敗時: 5 分以内に Slack 通知 |
| DP-06 | Circuit Breaker（pybreaker）: 5 回失敗で OPEN, 60s 後 HALF-OPEN |
| DP-07 | Circuit OPEN 時: GCS 最終成功データ配信 + "データ更新中" バナー |

### 4-B. AI 予測（PM-01〜10）

| ID | 要件 |
|---|---|
| PM-01 | 単勝的中率: 初期 ≥20%, 中期目標 ≥30% |
| PM-02 | Gunicorn ワーカー内での `predict()` 呼び出し禁止（ruff lint 強制） |
| PM-03 | 起動時に `MODEL_CURRENT_PATH` からモデルを 1 回のみロード |
| PM-04 | Pandera バリデーション: 距離 800〜3,600m / 頭数 2〜18 / 馬体重 380〜620kg |
| PM-05 | ETL フラグ確認: `etl:complete:{race_date}` がなければ最大 10 分待機後に停止 + Slack |
| PM-06 | SHAP TreeExplainer: 上位 10 頭のみ、推論バッチと時間分離実行 |
| PM-07 | モデルバージョン管理: current + N-1 世代を保持 |
| PM-08 | 予測キャッシュに `model_version` を含める |
| PM-09 | オッズ ±15% 変動で差分再推論（1 日上限 5 回） |
| PM-10 | 週次再学習は Cloud Run Jobs 専用（VPS での学習禁止） |

### 4-C. 分析機能（F-1〜F-5）

| ID | 要件 |
|---|---|
| F-1 | 条件別単勝率・ROI 集計（マルチフィルタ） |
| F-2 | 馬券種別収支分析 |
| F-3 | コース別種牡馬ランキング（horses.sire_id） |
| F-4 | AND 条件フィルタ UI |
| F-5 | データ鮮度表示（FRESH / STALE / STALE_CIRCUIT_OPEN / UNKNOWN） |

---

## 5. 非機能要件

| ID | 分類 | 要件 |
|---|---|---|
| NFR-01 | 可用性 | ≥99.5%（≤3.6h/月ダウンタイム） |
| NFR-02 | レイテンシ | API P50 ≤50ms（Redis HIT） |
| NFR-03 | レイテンシ | API P99 ≤200ms（Redis HIT） |
| NFR-04 | レイテンシ | E2E P95 ≤1,200ms |
| NFR-05 | バッチ | 推論バッチ ≤10分（08:00〜08:10） |
| NFR-06 | セキュリティ | レート制限: 60 req/min/IP, 10 req/min/user（分析 API） |
| NFR-07 | 圧縮 | gzip で転送量 -60% |
| NFR-08 | モデル精度 | Logloss ≤2.2, Calibration Error ≤0.05, ROI ≥-15% |
| NFR-09 | テンポラルリーク | バリデーション日より前で strict カットオフ |

---

## 6. API エンドポイント（確定版）

| メソッド | パス | 認証 | ゲスト |
|---|---|---|---|
| GET | `/health` | No | — |
| GET/POST | `/login` | No | — |
| POST | `/logout` | Yes | 401 |
| GET | `/api/v1/races/today` | No | 閲覧可 |
| GET | `/api/v1/races/{race_id}/entries` | Yes | 401 |
| GET | `/api/v1/predictions/{race_id}` | No | TOP3 のみ |
| POST | `/api/v1/filter/stats` | Yes | 401 |
| GET | `/api/v1/sires/ranking` | Yes | 401 |
| GET | `/api/v1/shap/{race_id}` | Yes | 401 |
| GET | `/api/v1/horses/{id}` | Yes | 401 |
| GET | `/admin/circuit-status` | Yes（IP 制限） | 403 |

### 予測 API レスポンス仕様

```json
{
  "race_id": "R11_20240120",
  "data_status": "fresh",
  "model_version": "v3",
  "predictions": [
    {
      "horse_id": "12345",
      "horse_name": "サンプルホース",
      "win_probability": 0.34,
      "confidence_score": 82,
      "rank_prediction": 1,
      "reasons": ["過去成績: 直近3走平均着順 1.8位", "馬場適性: 良馬場勝率 61%"],
      "inference_source": "local"
    }
  ],
  "generated_at": "2024-01-20T09:15:00+09:00"
}
```

タイムスタンプ: ISO 8601 + JST（+09:00）統一

---

## 7. 主なリスク

| リスク | 対策 |
|---|---|
| JRA スクレイピング法的リスク | **Human Review 必須**: robots.txt / ToS 確認後に ETL 開始 |
| テンポラルリーク | バリデーション日 strict カットオフ（NFR-09） |
| VPS 2GB 超過 | スクレイパーと推論の同時実行禁止、CoW 活用 |
