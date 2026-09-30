# Git ブランチ管理（ページURL別 feature ブランチ）

> **2026-09-30 訂正（3回目）**: 数える対象を「ユーザがブラウザで直接開くページURL（`/api/`を含まないHTMLルート）」に絞り込んだ。過去2回の訂正（エンドポイント→ルート→URLパス）はいずれも `/api/*` のJSON APIを含めて数えていたため225件になっていたが、そのうち200件は画面から呼ばれる裏側のAPIであり、ユーザが実際に到達するページではない。ページ単体のURLは**27件**（`src/api/app.py` 全228ルート定義中）。

`src/api/app.py`（FastAPI legacy, :8000）の主要機能ごとに `feature/XXX` ブランチを作成し、`main`（`e7e9e17`, 2026-09-30）から分岐した。全ブランチは分岐時点で `main` と同一コミットであり、今後の変更はこのブランチ単位で行い、レビュー後に `main` へマージする運用とする。

各ブランチの対象は「ページURL」だが、実際の開発ではそのページを支える `/api/*` 実装も同じブランチで扱う（ページ単体では動作しないため）。`/api/*` の全一覧は `.claude/skills/evaluate-keiba-architecture/scripts/collect_endpoints.py` の出力を参照。

Flask `/api/v1`（DEC-013仕様上の正）・monitorポータルは対象外（FastAPI legacyの主要機能別）。

## ブランチ一覧

| ブランチ | 対象領域 | ページURL数 | 詳細 |
|---|---|---|---|
| `feature/admin-ops` | 管理者運用（cron・構造チェック・システム統計・ログ） | 3 | [feature-admin-ops.md](./feature-admin-ops.md) |
| `feature/betting` | 馭券戦略 | 1 | [feature-betting.md](./feature-betting.md) |
| `feature/bloodline-pedigree` | 血統・種牡馬クラスタ | 7 | [feature-bloodline-pedigree.md](./feature-bloodline-pedigree.md) |
| `feature/core-platform` | コアプラットフォーム（ヘルス/認証/ダッシュボード） | 3 | [feature-core-platform.md](./feature-core-platform.md) |
| `feature/cushion` | クッション値（トラックコンディション） | 0 | [feature-cushion.md](./feature-cushion.md) |
| `feature/growth-curve` | 成長曲線 | 1 | [feature-growth-curve.md](./feature-growth-curve.md) |
| `feature/horse-profile` | 馬プロフィール・馬名検索・関係者統計 | 0 | [feature-horse-profile.md](./feature-horse-profile.md) |
| `feature/model-training` | モデル学習・シミュレーション・バックフィル | 0 | [feature-model-training.md](./feature-model-training.md) |
| `feature/monitor-quality` | 監視・データ品質チェック | 2 | [feature-monitor-quality.md](./feature-monitor-quality.md) |
| `feature/myostatin` | ミオスタチン遺伝子解析 | 1 | [feature-myostatin.md](./feature-myostatin.md) |
| `feature/odds-final-odds` | オッズ・最終オッズ | 0 | [feature-odds-final-odds.md](./feature-odds-final-odds.md) |
| `feature/race-detail` | レース詳細・予測表示 | 1 | [feature-race-detail.md](./feature-race-detail.md) |
| `feature/race-quality` | レース質分析 | 1 | [feature-race-quality.md](./feature-race-quality.md) |
| `feature/scraping-queue` | スクレイピング・キュー管理 | 4 | [feature-scraping-queue.md](./feature-scraping-queue.md) |
| `feature/track-speed` | トラックスピード指標 | 2 | [feature-track-speed.md](./feature-track-speed.md) |
| `feature/tracking-difficulty` | 追走難度 | 1 | [feature-tracking-difficulty.md](./feature-tracking-difficulty.md) |

合計: 27 ページURL（`src/api/app.py` 全体では228ルート定義・225ユニークURLパスあり、うち`/api/*`のJSON APIが198件、ページが27件〈`/login`のGET/POSTを1件と数えると27件〉）

### ページが0件のブランチ（削除せず保持）

以下のブランチは対象領域が `/api/*` のみで構成されており、ページURL基準では対象が無い。
削除候補として提示したが、2026-09-30時点でユーザの判断により削除せず保持することとした:

- `feature/odds-final-odds`
- `feature/model-training`
- `feature/cushion`
- `feature/horse-profile`

## 運用ルール

- 新しいページURLを追加する場合は、まず該当する機能領域のブランチが無いか本ファイルを確認する。
- 該当ブランチが無い場合は `feature/<領域名>` で新規作成し、本ディレクトリに対応するmdファイルを追加する。
- ブランチ命名は `feature/<kebab-case>` で統一する。
- DEC-013により新規APIは本来 `src/api/flask_app.py` / `src/api/v1/` 側に実装すべきであり（`.claude/skills/debug-refactor/SKILL.md` 参照）、legacy機能ブランチでの新規ページURL追加は既存機能の修正・移行作業を想定したものである。
