# Git ブランチ管理（エンドポイント別 feature ブランチ）

`src/api/app.py`（FastAPI legacy, :8000）の主要機能ごとに `feature/XXX` ブランチを作成し、`main`（`e7e9e17`, 2026-09-30）から分岐した。全ブランチは分岐時点で `main` と同一コミットであり、今後の変更はこのブランチ単位で行い、レビュー後に `main` へマージする運用とする。

Flask `/api/v1`（DEC-013仕様上の正）・monitorポータルは対象外（FastAPI legacyの主要機能別）。

## ブランチ一覧

| ブランチ | 対象領域 | エンドポイント数 | 詳細 |
|---|---|---|---|
| `feature/admin-ops` | 管理者運用（cron・構造チェック・システム統計・ログ） | 18 | [feature-admin-ops.md](./feature-admin-ops.md) |
| `feature/betting` | 馭券戦略 | 3 | [feature-betting.md](./feature-betting.md) |
| `feature/bloodline-pedigree` | 血統・種牡馬クラスタ | 65 | [feature-bloodline-pedigree.md](./feature-bloodline-pedigree.md) |
| `feature/core-platform` | コアプラットフォーム（ヘルス/認証/ダッシュボード） | 10 | [feature-core-platform.md](./feature-core-platform.md) |
| `feature/cushion` | クッション値（トラックコンディション） | 8 | [feature-cushion.md](./feature-cushion.md) |
| `feature/growth-curve` | 成長曲線 | 3 | [feature-growth-curve.md](./feature-growth-curve.md) |
| `feature/horse-profile` | 馬プロフィール・馬名検索・関係者統計 | 6 | [feature-horse-profile.md](./feature-horse-profile.md) |
| `feature/model-training` | モデル学習・シミュレーション・バックフィル | 13 | [feature-model-training.md](./feature-model-training.md) |
| `feature/monitor-quality` | 監視・データ品質チェック | 15 | [feature-monitor-quality.md](./feature-monitor-quality.md) |
| `feature/myostatin` | ミオスタチン遺伝子解析 | 4 | [feature-myostatin.md](./feature-myostatin.md) |
| `feature/odds-final-odds` | オッズ・最終オッズ | 5 | [feature-odds-final-odds.md](./feature-odds-final-odds.md) |
| `feature/race-detail` | レース詳細・予測表示 | 15 | [feature-race-detail.md](./feature-race-detail.md) |
| `feature/race-quality` | レース質分析 | 5 | [feature-race-quality.md](./feature-race-quality.md) |
| `feature/scraping-queue` | スクレイピング・キュー管理 | 43 | [feature-scraping-queue.md](./feature-scraping-queue.md) |
| `feature/track-speed` | トラックスピード指標 | 12 | [feature-track-speed.md](./feature-track-speed.md) |
| `feature/tracking-difficulty` | 追走難度 | 3 | [feature-tracking-difficulty.md](./feature-tracking-difficulty.md) |

合計: 228 エンドポイント（`src/api/app.py` の全228ルート）

## 運用ルール

- 新しいエンドポイントを追加する場合は、まず該当する機能領域のブランチが無いか本ファイルを確認する。
- 該当ブランチが無い場合は `feature/<領域名>` で新規作成し、本ディレクトリに対応するmdファイルを追加する。
- ブランチ命名は `feature/<kebab-case>` で統一する。
- DEC-013により新規APIは本来 `src/api/flask_app.py` / `src/api/v1/` 側に実装すべきであり（`.claude/skills/debug-refactor/SKILL.md` 参照）、legacy機能ブランチでの新規エンドポイント追加は既存機能の修正・移行作業を想定したものである。
