# DEC-002: decision

**Date**: 2026-07-02
**Agent**: proposal-agent, backend-engineer, data-engineer, ai-model-engineer
**Task**: TASK-007
**Status**: ACCEPTED

---

## Context

「データ管理に関してのベストプラクティスを考える。現状はGCSで管理している。ローカルPCを常時動かしておくようなデータベース管理も可能だが、常にオンにしておかないといけないのでできれば避けたい。」の改善要件定義書です。要件の優先順位と実装可否を確認してください。

---

## Decision

<response>

# 改善要件定義書の作成

以下のドキュメントを生成します（Document Management Protocol に従い、各ファイルパスも明示します）。

---

## 📁 生成ドキュメント一覧

| ドキュメント種別 | パス |
|---|---|
| タスク | `project/keiba-vpn/tasks/TASK-001-data-management-requirements.md` |
| 思考ログ | `project/keiba-vpn/docs/thinking-logs/2025-01-01-orchestrator-TASK-001.md` |
| 決定記録 | `project/keiba-vpn/docs/decisions/DEC-001-data-management-architecture.md` |
| **要件定義書（本体）** | `project/keiba-vpn/docs/requirements/REQ-001-data-management.md` |
| ダッシュボード更新 | `project/keiba-vpn/docs/status/DASHBOARD.md` |

---

## 📄 `project/keiba-vpn/tasks/TASK-001-data-management-requirements.md`

```markdown
# TASK-001: データ管理改善 要件定義書の作成

- **Status**: DONE
- **Created**: 2025-01-01
- **Owner**: orchestrator
- **Related Agents**: backend-engineer, data-engineer, ai-model-engineer, cost-optimizer
- **Input**: Round 1〜3 の自律的議論ログ
- **Output**: REQ-001-data-management.md
```

---

## 📄 `project/keiba-vpn/docs/thinking-logs/2025-01-01-orchestrator-TASK-001.md`

```markdown
# 思考ログ — TASK-001

## エージェント別クリティークの統合判断

### Firestore 廃止の根拠
- 3エージェント共通で「競馬ドメインに根本的に不向き」と判定
- 主理由: JOINなし / 動的複合フィルター不可 / 不等号+ソートの同時使用制限
- 代替: OLTP → Cloud SQL、メタデータ配信 → Cloud SQL キャッシュ層

### BigQuery を最重要提案に格上げした根拠
- ai-model-engineer: テンポラルリーケージ防止のために監査テーブルが必要
- data-engineer: オッズ更新がデイリーバッチでは不十分（1時間インクリメンタルが必要）
- backend-engineer: GCSキャッシュ鮮度管理とParquet形式の提案を採用

### 役割分担の確定ロジック
  Cloud SQL  → OLTP（書き込み整合性・正規化エンティティ）
  BigQuery   → 分析・AI評価（読み取り専用、スケールアウト不要）
  GCS        → ファイル原本・モデルアーティファクト・キャッシュ
  Cloud Workflows → パイプライン制御（Phase 3 で実装）
```

---

## 📄 `project/keiba-vpn/docs/decisions/DEC-001-data-management-architecture.md`

```markdown
# DEC-001: データ管理アーキテクチャの決定

- **Status**: DECIDED
- **Date**: 2025-01-01
- **Decision Makers**: backend-engineer, data-engineer, ai-model-engineer (via Orchestrator)

## 決定事項
1. Firestore の採用を **見送り**（3エージェント一致）
2. Cloud SQL (PostgreSQL) を OLTP 基盤として採用（スケジュール自動停止付き）
3. BigQuery を 分析・AI評価レイヤーとして採用（インクリメンタルロード）
4. GCS 命名規則の標準化を **最優先基盤整備** として即時着手
5. Cloud Workflows によるパイプライン管理は Phase 3 として後続実装

## 否決された案
- Firestore によるメタデータ管理: 競馬ドメインの複合クエリ要件に根本的に不適合
- ローカル PC 常時稼働 DB: 可用性・保守コストの観点で要件から除外
```

---

## 📄 **要件定義書本体** `project/keiba-vpn/docs/requirements/REQ-001-data-management.md`

---

# 改善要件定義書: データ管理アーキテクチャ（keiba-vpn）

> **作成日**: 2025-01-01
> **対象システム**: keiba-vpn（日本競馬予測Webアプリ）
> **策定プロセス**: 3ラウンドの自律エージェント議論（backend-engineer / data-engineer / ai-model-engineer）を Orchestrator が統合
> **関連決定記録**: DEC-001-data-management-architecture.md

---

## サマリー

現行のGCS単層管理から、**GCS（ファイル原本）・Cloud SQL（OLTP）・BigQuery（分析/AI評価）の3層アーキテクチャ**へ移行する。ローカルPCの常時稼働を完全回避しつつ、Cloud SQLのスケジュール自動停止によりコストを最小化する。すべての層はサーバーレス or スケジュール停止で管理され、馬・騎手・レースエンティティの複合クエリ、リアルタイムに近いオッズ更新、AI推論の再現性確保を同時に実現する。

---

## アーキテクチャ概観

```
┌─────────────────────────────────────────────────────────┐
│                  keiba-vpn データ層                       │
│                                                         │
│  [スクレイピング]                                         │
│       │                                                 │
│       ▼                                                 │
│  ┌─────────┐   raw/・processed/   ┌──────────────────┐  │
│  │   GCS   │ ←────────────────── │ モデルアーティファクト│  │
│  │（原本層）│ ──キャッシュ出力────→ │  feature_columns │  │
│  └────┬────┘                     └──────────────────┘  │
│       │ インクリメンタルロード(1h/日次)                    │
│       ▼                                                 │
│  ┌──────────┐  分析・AI評価クエリ  ┌──────────────────┐  │
│  │ BigQuery │ ◀──────────────── │ ai-model-engineer│  │
│  │（分析層） │                    └──────────────────┘  │
│  └──────────┘                                          │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │        Cloud SQL (PostgreSQL)  ← OLTP専用         │   │
│  │  馬/騎手/レースマスタ・出馬表・予測スコア最新版       │   │
│  │  ※ 深夜1時〜朝5時 自動停止（コスト最適化）           │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 機能要件

| # | 要件 | 優先度 | 担当エージェント | 根拠 |
|---|------|--------|----------------|------|
| **F-1** | GCS バケットのパス命名規則を日付パーティション形式に統一する | **高** | data-engineer | 全提案の基盤。他すべての要件の前提条件 |
| **F-2** | `raw/odds/` のパスに時刻（HH）粒度を追加し、数分単位のオッズ更新を記録できる構造にする | **高** | data-engineer | オッズは当日レース中に数分単位で更新されるため日付粒度では不足 |
| **F-3** | GCS バケットのオブジェクトバージョニングを有効化する | **高** | data-engineer / backend-engineer | 誤上書き防止・障害時ロールバックに必須 |
| **F-4** | Cloud SQL (PostgreSQL, db-f1-micro) を OLTP 基盤として導入し、馬・騎手・調教師・レーススケジュール・出馬表・予測スコア最新版を管理する | **高** | backend-engineer | Firestore は本ドメインの複合クエリ要件に非対応と全エージェント一致で判断 |
| **F-5** | Cloud SQL に接続プーリングを設定する（pool_size=5, max_overflow=2, pool_pre_ping=True, pool_recycle=1800） | **高** | backend-engineer | インスタンス停止→再起動後の接続エラーを防止 |
| **F-6** | Cloud SQL の接続方法として Cloud SQL Connector (IAM 認証) を採用する | **高** | backend-engineer | Cloud Run 環境での Unix Socket 不要・運用簡素化 |
| **F-7** | BigQuery を分析・AI評価の専用レイヤーとして導入し、`model_evaluation_log` / `feature_snapshot` / `prediction_vs_result` / `temporal_split_log` テーブルを定義する | **高** | ai-model-engineer | モデル評価・テンポラルリーケージ防止・ROIシミュレーションに必須 |
| **F-8** | GCS → BigQuery のロードをレース当日は1時間ごとのインクリメンタルロード（MERGE）、非レース日は日次バッチに切り替える | **高** | data-engineer | 毎朝1回のバッチでは当日のオッズ変動を推論に反映できない |
| **F-9** | BigQuery の分析結果を GCS に Parquet 形式でエクスポートし、Flask API がキャッシュとして参照できる仕組みを構築する | **中** | backend-engineer / data-engineer | JSON比 60〜70% サイズ削減・読み取り速度向上 |
| **F-10** | GCSキャッシュの鮮度メタデータ（generated_at / valid_until / gcs_path）を Cloud SQL に持たせ、API レスポンス時に TTL チェックを行う | **中** | backend-engineer | レース当日の短 TTL（〜数時間）と非レース日の長 TTL を動的に制御するため |
| **F-11** | モデルアーティファクトのパスに `feature_columns.json`（学習時特徴量リスト）と `training_metadata.json`（train cutoff 日付・評価スコア）を追加する | **中** | ai-model-engineer | 推論時の特徴量整合性確保とテンポラルリーケージ監査に必要 |
| **F-12** | GCS ライフサイクルポリシーを設定する（`raw/`→90日後 Nearline、365日後 Coldline / `cache/`→7日後削除） | **中** | data-engineer / cost-optimizer | ストレージコスト長期的に 20〜40% 削減見込み |
| **F-13** | Cloud Scheduler + Cloud Workflows によりスクレイピング→ETL→推論→通知のパイプラインを自動管理する | **低** | operations-engineer | 実装難易度が高いため Phase 3 で対応 |
| **F-14** | BigQuery テンポラルリーケージ自動検証クエリを定期実行し、モデル学習データへの未来データ混入を検知する | **低** | ai-model-engineer | 高精度維持に重要だが BigQuery 基盤構築後に実装可能 |

---

## 非機能要件

| # | 要件 | 目標値 | 担当 |
|---|------|--------|------|
| **N-1** | Cloud SQL の月間稼働コスト | ≤ $8/月（自動停止スケジュール：深夜1時〜朝5時） | cost-optimizer |
| **N-2** | Flask API のレスポンスタイム（GCS キャッシュ参照時） | P95 ≤ 500ms | operations-engineer |
| **N-3** | Cloud SQL インスタンス Cold start 時間（停止→接続確立） | ≤ 120秒、かつ停止時間帯にスクレイピング・推論ジョブが重ならないよう設計 | backend-engineer / operations-engineer |
| **N-4** | BigQuery クエリコスト（分析・評価用途） | ≤ $5/月（本プロジェクト規模〜数GB/月はほぼ無視できる水準） | cost-optimizer |
| **N-5** | GCS ストレージコスト（Nearline 移行後） | 現状比 ≤ -20%（12ヶ月後） | cost-optimizer / data-engineer |
| **N-6** | BigQuery インクリメンタルロードの遅延（GCS → BigQuery 反映） | レース当日: ≤ 90分 / 非レース日: ≤ 24時間 | data-engineer |
| **N-7** | モデル推論の再現性 | 同一 race_id・model_version で同一スコアを再現できること（feature_columns.json 保存により担保） | ai-model-engineer |
| **N-8** | 常時稼働サーバー | ゼロ（Cloud SQL の停

---

## Conclusion

****

---

## Consequences

- この決定はレビュー済みで承認されました
- 実装時はこのドキュメントを参照してください

---

_Approved via Multi-Agent Console — 2026-07-02_
