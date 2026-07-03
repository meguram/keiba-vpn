# DEC-008: 全エージェント合意済みの最終提案を統合した結果、keiba-vpn は **ConoHa VPS 固定費モデル（¥1,520〜1,820/月）** のもと、Flask + LightGBM バッチ推論 + Redis キャッシュを VPS 一台に同居させる構成が最もコスト・パフォーマンスのバランスが高く、本要件定義書はその実装方針を確定した公式ドキュメントとする。

**Date**: 2026-07-03
**Agent**: web-search-agent, decisions-context-agent, proposal-agent, backend-engineer, data-engineer, ai-model-engineer, integration-synthesizer, quality-reviewer
**Task**: TASK-026
**Status**: ACCEPTED

---

## Context

以下2点について人間レビューを推奨します: ①スクレイピング対象サイト（JRA 等）の利用規約・robots.txt との適法性確認（ETL パイプライン実装前に必須）、②VPS 2GB メモリ割り当て表の実測値による検証（SHAP 計算の時間差実行設計が実際の運用で成立するかフェーズ2開始前に負荷テスト実施を推奨）。

---

## Decision

# 改善要件定義書: keiba-vpn — 日本競馬予測・データ分析 Web アプリ

> **ドキュメント番号**: REQ-001
> **バージョン**: v1.0（Loop 4 全エージェント合意済み）
> **作成日**: 2025-01-16
> **参照決定**: DEC-002 / DEC-004 / DEC-006 / DEC-007

---

## サマリー

keiba-vpn は、JRA 競馬データの統計分析と LightGBM 推論モデルを組み合わせた **日本競馬向け予測・データ分析 Web アプリ**である。ユーザは任意条件（距離・馬場・クラス・騎手等）で絞り込んだ勝率・単勝回収率の統計閲覧と、バッチ事前計算済みの AI 予測スコアを出馬表と並べて確認できる。インフラは **ConoHa VPS 固定費モデル（月額 ¥1,520〜1,820）** を採用し、クラウド従量課金によるコスト爆発リスクを排除する。

---

## 1. アプリターゲット（機能要件）

### 1-1. データ分析機能

| # | 機能名 | 具体的な内容 | 主要指標 | 優先度 | 担当エージェント |
|---|---|---|---|---|---|
| F-1 | 条件別勝率分析 | 距離・馬場・クラス・季節・騎手・調教師 × 対象馬のフィルタリング集計 | 勝率 (%)、連対率 (%) | 高 | backend-engineer |
| F-2 | 単勝回収率分析 | F-1 と同条件に対応した単勝・複勝回収率の算出 | 単回収率 (%)、複回収率 (%) | 高 | backend-engineer |
| F-3 | コース別好成績種牡馬 | 指定コース × 任意条件での種牡馬ランキング（`horses.sire_id` カラム必須） | 勝率・回収率・出走数 | 高 | data-engineer |
| F-4 | レース条件複合フィルタ | 開催場所・距離・馬場・天候・クラス・頭数の AND 条件指定 UI | — | 高 | frontend-engineer |
| F-5 | データ鮮度表示 | スクレイピング最終更新タイムスタンプを UI に表示（`data_freshness` フィールド） | 更新時刻 | 中 | fullstack-integrator |

### 1-2. AI モデル結果表示機能

| # | 機能名 | 具体的な内容 | 主要指標 | 優先度 | 担当エージェント |
|---|---|---|---|---|---|
| F-6 | AI 予測スコア一覧 | 事前バッチ計算済み JSON を出馬表と並列表示 | 推定勝率・着順ランク | 高 | ai-model-engineer |
| F-7 | 推奨馬ハイライト | 上位スコア馬を色分け・ランク番号で視覚強調 | — | 高 | frontend-engineer |
| F-8 | SHAP 説明可能 AI | LightGBM SHAP TreeExplainer による上位寄与特徴量の簡易表示（上位 10 頭のみ適用） | 上位 5 特徴量・寄与度 | 中 | ai-model-engineer |
| F-9 | 予測精度フィードバック | レース後実結果と予測の照合・的中率追跡ダッシュボード | 単勝的中率・回収率推移 | 低 | ai-model-engineer |
| F-10 | 推論ソース表示 | バッチ推論 / 外部 WebAPI フォールバック区別を UI に表示（`inference_source` フラグ） | — | 中 | fullstack-integrator |

### 1-3. 認証・共通機能

| # | 機能名 | 内容 | auth_required | 優先度 |
|---|---|---|---|---|
| F-11 | ユーザ認証 | Flask-Login セッション認証（ログイン・ログアウト） | ❌ No（ログインページ自体） | 高 |
| F-12 | 全分析 API 認証ガード | F-1〜F-10 すべてに `@login_required` デコレータ適用 | ✅ Yes | 高 |
| F-13 | ヘルスチェック | `/health` エンドポイント（Nginx ヘルスチェック用） | ❌ No | 高 |

---

## 2. 運用パフォーマンスとコスト

### 2-1. パフォーマンス非機能要件

| # | 要件 | 目標値 | 達成手段 | 担当 |
|---|---|---|---|---|
| N-1 | データ分析 API レスポンス | < 1.5 秒（キャッシュ未使用時） | Redis TTL 3h キャッシュ | backend-engineer |
| N-2 | AI スコア取得レスポンス | < 200 ms | バッチ済み JSON 返却（GCS/ローカル） | ai-model-engineer |
| N-3 | AI 推論バッチ処理時間 | < 5 分 / 日（全レース・約 200 頭） | LightGBM バッチ + cron 23:00 | ai-model-engineer |
| N-4 | 外部 WebAPI フォールバック遅延 | < 3 秒 | DEC-006 参照 | ai-model-engineer |
| N-5 | 同時接続ユーザ数 | 最大 50 名（重賞ピーク想定） | Gunicorn worker=2, threads=4 | operations-engineer |
| N-6 | フロントエンド初期表示 (LCP) | < 2.5 秒 | Vercel CDN + Next.js SSG/ISR | frontend-engineer |
| N-7 | データ鮮度（スクレイピング） | 前日夜〜当日朝 06:00 更新 | cron + WireGuard VPN | data-engineer |
| N-8 | キャッシュ後レスポンス | < 200 ms | Redis インメモリ返却 | backend-engineer |

### 2-2. スクレイピングスケジュール詳細

```
データ種別          取得タイミング              頻度
──────────────────────────────────────────────────────
出馬表              毎朝 06:00（レース3日前〜前日） 日次
オッズ              毎10分（当日 09:00〜発走）      高頻度（当日のみ）
レース結果          毎30分（当日 15:00〜18:00）     レース後
AI バッチ推論       出馬表確定後 23:00             日次
モデル再学習        毎週月曜 02:00                 週次
```

### 2-3. コスト試算表（月次・VPS 固定費モデル）

| # | サービス | 用途 | 月額概算 | 備考 |
|---|---|---|---|---|
| C-1 | **ConoHa VPS 2GB / 100GB SSD** | Flask API・AI 推論バッチ・cron・Redis・WireGuard 全同居 | **¥1,320** | DEC-007 固定費モデル |
| C-2 | **Google Cloud Storage** | Parquet 形式過去レースデータ長期保存・バックアップ | ¥200〜500 | 日次書き込み中心（DEC-002） |
| C-3 | **Vercel (Next.js)** | フロントエンドホスティング | **¥0** | Hobby プラン（月 100GB 帯域内） |
| C-4 | **WireGuard VPN** | スクレイピング出口 IP 管理 | ¥0 | VPS 内同居 |
| C-5 | **Redis 7** | 分析クエリキャッシュ（maxmemory 256MB） | ¥0 | VPS 内同居 |
| C-6 | **Slack Webhook** | スクレイピング失敗・障害アラート | ¥0 | 無料プラン |
| — | **合計（概算）** | — | **¥1,520〜1,820 / 月** | GCS 変動費が幅の要因 |

> 💡 **コスト最適化ポイント**: VPS 固定費モデルにより、重賞週末のアクセス急増時もスケールアウトコストが発生しない。GCS は書き込み中心で読み取りは VPS ローカルキャッシュ経由のため従量課金を最小化。

### 2-4. VPS リソース割り当て（2GB メモリ）

```
コンポーネント           推定使用メモリ
────────────────────────────────────
Flask / Gunicorn         ~300 MB
Nginx                    ~50 MB
Redis (maxmemory 256MB)  ~256 MB
LightGBM バッチ推論      ~400 MB（ピーク時）
SHAP 計算（上位10頭）    ~200 MB（推論とは時間差実行）
OS / その他              ~300 MB
────────────────────────────────────
合計見積               ~1,506 MB / 2,048 MB ✅ 安全圏
```

---

## 3. 実装言語・サービス構成

### 3-1. 技術スタック全体図

```
┌─────────────────────────────────────────────────────────────────┐
│             keiba-vpn システム構成（DEC-007 準拠）              │
├─────────────────┬───────────────────────────────────────────────┤
│  レイヤ         │  技術・サービス                               │
├─────────────────┼───────────────────────────────────────────────┤
│ フロントエンド   │ React / Next.js (TypeScript)                 │
│                 │ Tailwind CSS / Recharts（グラフ表示）         │
│                 │ Vercel（無料ホスティング / CDN）              │
├─────────────────┼───────────────────────────────────────────────┤
│ バックエンド     │ Python 3.11 / Flask                          │
│ API             │ Gunicorn (worker=2, threads=4) + Nginx        │
│                 │ REST API (JSON) / Flask-Login（セッション認証）│
├─────────────────┼───────────────────────────────────────────────┤
│ データ収集      │ Python (requests / BeautifulSoup4)            │
│ ETL             │ VPS cron（スケジュール別実行）                │
│                 │ WireGuard VPN（出口 IP 管理）                 │
│                 │ 失敗時 → Slack Webhook アラート              │
├─────────────────┼───────────────────────────────────────────────┤
│ キャッシュ      │ Redis 7（VPS 内同居）                        │
│                 │ maxmemory 256MB / allkeys-lru ポリシー        │
├─────────────────┼───────────────────────────────────────────────┤
│ データ保存      │ VPS ローカル /data（SSD 100GB）← 主系        │
│                 │ GCS Parquet（長期保存・バックアップ）← DEC-002│
│                 │ SQLite / PostgreSQL（レース結果・予測 DB）    │
├─────────────────┼───────────────────────────────────────────────┤
│ AI / ML         │ scikit-learn / LightGBM（バッチ推論 ← 通常時）│
│ (DEC-006)       │ 外部 WebAPI フォールバック（モデル更新中等）  │
│                 │ SHAP TreeExplainer（説明可能 AI）             │
│                 │ モデルファイル: /models/v{N}/lgbm.pkl         │
├─────────────────┼───────────────────────────────────────────────┤
│ 監視・運用      │ systemd 自動再起動                            │
│                 │ Slack Webhook（障害・ETL 失敗アラート）       │
│                 │ Nginx アクセスログ                            │
└─────────────────┴───────────────────────────────────────────────┘
```

### 3-2. 主要データモデル（スキーマ概要）

```sql
-- 主要テーブル定義（要件レベル）

races          (race_id, date, venue, course, distance, surface, weather, class)
horses         (horse_id, name, sire_id, ...)   -- sire_id: 種牡馬分析に必須
entries        (entry_id, race_id, horse_id, jockey_id, trainer_id, weight, odds)
results        (result_id, entry_id, rank, time, prize)
predictions    (pred_id, race_id, horse_id, win_prob, place_prob,
                model_version, inference_source ENUM('local','external'),
                created_at)

-- インデックス
CREATE INDEX idx_horses_sire_id ON horses(sire_id);
CREATE INDEX idx_entries_race    ON entries(race_id);
CREATE INDEX idx_predictions_race ON predictions(race_id);
```

### 3-3. AI 推論 2 段階戦略（DEC-006）

```
[通常時] VPS バッチ推論（cron 23:00）
  ├── LightGBM .pkl ロード（/models/v{N}/lgbm.pkl）
  ├── pandas 特徴量エンジニアリング
  ├── predict_proba() → predictions テーブル書き込み
  │      inference_source = 'local'
  └── SHAP TreeExplainer → 上位 10 頭のみ計算（時間差実行）

[フォールバック時] 外部 WebAPI 推論（モデル更新中 / VPS 推論失敗時）
  ├── 特徴量 JSON → 外部エンドポイントへ POST
  └── レスポンス → predictions テーブル書き込み
         inference_source = 'external'
```

---

## 4. 実装ロードマップ

### Phase 1（優先度: 高 ／ 目安 4〜6 週）

```
[ ] VPS 環境構築（ConoHa + WireGuard + Nginx + systemd）
[ ] スクレイピング ETL パイプライン（requests/BS4 + cron + Slack アラート）
[ ] DB スキーマ構築（races / horses / entries / results）
    └── horses.sire_id カラム + インデックス追加
[ ] Flask API 基盤（Flask-Login 認証 + @login_required）
[ ] 条件別勝率・単勝回収率 API（F-1 / F-2）
[ ] Redis キャッシュデコレータ実装
[ ] Next.js フロントエンド基盤（認証フロー + 分析フィルタ UI）
```

### Phase 2（優先度: 中 ／ 目安 3〜4 週）

```
[ ] LightGBM モデル学習・バッチ推論パイプライン
[ ] predictions テーブル + inference_source フラグ
[ ] AI スコア一覧・推奨馬ハイライト UI（F-6 / F-7）
[ ] コース別好成績種牡馬 API（F-3）
[ ] GCS バックアップ連携（DEC-002）
[ ] データ鮮度表示（F-5）
```

### Phase 3（優先度: 低 ／ 目安 2〜3 週）

```
[ ] SHAP 説明可能 AI 表示（F-8）— メモリ時間差実行対応
[ ] 推論ソース表示 UI（F-10）
[ ] 予測精度フィードバックダッシュボード（F-9）
[ ] 外部 WebAPI フォールバック実装（DEC-006 完全対応）
[ ] モデル週次再学習 Job（cron 月曜 02:00）
```

---

## 5. 依存関係・リスク

### 依存関係

| 依存元 | 依存先 | 内容 |
|---|---|---|
| AI バッチ推論 | ETL パイプライン | 出馬表確定データが前提（Phase 1 完了後に Phase 2 着手） |
| SHAP 計算 | LightGBM バッチ推論 | 推論完了後に時間差で別 cron ジョブとして実行 |
| フロントエンド AI 表示 | predictions テーブル | バッチ推論 DB 書き込みが完了していること |
| GCS バックアップ | VPS ローカル /data | VPS 側 ETL が正常完了していること |

### リスク一覧

| # | リスク | 発生確率 | 影響度 | 対策 |
|---|---|---|---|---|
| R-1 | JRA サイト HTML 変更によるスクレイピング破綻 | 中 | 高 | Slack アラート + 手動 selector 修正フロー整備 |
| R-2 | VPS 2GB メモリ不足（SHAP + 推論同時実行） | 中 | 中 | cron 時間差実行・SHAP 対象を上位 10 頭に限定 |
| R-3 | Vercel Hobby プラン 100GB 帯域超過 | 低 | 中 | 静的アセット最適化・画像圧縮 |
| R-4 | Redis メモリ溢れによるキャッシュ全消去 | 低 | 低 | allkeys-lru 採用・maxmemory 256MB 設定で安全運用 |
| R-5 | LightGBM モデル精度劣化（データドリフト） | 中 | 中 | 週次再学習 + 精度フィードバック追跡（F-9） |
| R-6 | GCS 認証切れによるバックアップ失敗 | 低 | 低 | Service Account キー定期更新 + アラート |

---

## 付録: エンドポイント一覧

| メソッド | エンドポイント | 機能 | auth_required |
|---|---|---|---|
| GET | `/health` | ヘルスチェック | ❌ No |
| GET/POST | `/login` | ログイン | ❌ No |
| POST | `/logout` | ログアウト | ✅ Yes |
| POST | `/api/v1/filter/stats` | 条件別勝率・回収率 | ✅ Yes |
| GET | `/api/v1/sires/ranking` | コース別種牡馬ランキング | ✅ Yes |
| GET | `/api/v1/predictions/{race_id}` | AI 予測スコア一覧 | ✅ Yes |
| GET | `/api/v1/shap/{race_id}` | SHAP 説明値 | ✅ Yes |
| GET | `/api/v1/races/{race_id}/entries` | 出馬表 + AI スコア統合 | ✅ Yes |

---

## Conclusion

**全エージェント合意済みの最終提案を統合した結果、keiba-vpn は **ConoHa VPS 固定費モデル（¥1,520〜1,820/月）** のもと、Flask + LightGBM バッチ推論 + Redis キャッシュを VPS 一台に同居させる構成が最もコスト・パフォーマンスのバランスが高く、本要件定義書はその実装方針を確定した公式ドキュメントとする。**

---

## Consequences

- この決定はレビュー済みで承認されました
- 実装時はこのドキュメントを参照してください

---

_Approved via Multi-Agent Console — 2026-07-03_
