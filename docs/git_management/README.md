# Git ブランチ管理（ページURL別 feature / hotfix ブランチ）

> **2026-09-30 訂正（3回目）**: 数える対象を「ユーザがブラウザで直接開くページURL（`/api/`を含まないHTMLルート）」に絞り込んだ。過去2回の訂正（エンドポイント→ルート→URLパス）はいずれも `/api/*` のJSON APIを含めて数えていたため225件になっていたが、そのうち200件は画面から呼ばれる裏側のAPIであり、ユーザが実際に到達するページではない。ページ単体のURLは**27件**（`src/api/app.py` 全228ルート定義中）。

> **2026-09-30 ブランチ命名の変更**: 検証の過程で機能領域の分類ミス（パス文字列の部分一致による誤分類）が実際に見つかり
> 修正した5ブランチは `feature/XXX` → `hotfix/XXX` にリネームした（`race-detail`・`tracking-difficulty`・
> `odds-final-odds`・`scraping-queue`・`cushion`）。分類ミスが見つからなかった11ブランチは `feature/XXX` のまま。
> `hotfix/XXX` は「分類ミスの修正」だけでなく「その機能領域に残る設計・実装上の問題」も解決した上で
> `feature/XXX` に戻す運用とする（一つずつ対応中: 進捗は下表の「検証結果」列を参照）。
> `race-detail`・`tracking-difficulty`・`odds-final-odds`・`scraping-queue`・`cushion` の5ブランチ全て
> 2026-09-30に解決済みで `feature/XXX` に復帰した。`hotfix/XXX` は現在0本。

`src/api/app.py`（FastAPI legacy, :8000）の主要機能ごとに `feature/XXX` ブランチを作成し、`main`（`e7e9e17`, 2026-09-30）から分岐した。全ブランチは分岐時点で `main` と同一コミットであり、今後の変更はこのブランチ単位で行い、レビュー後に `main` へマージする運用とする。

各ブランチの対象は「ページURL」だが、実際の開発ではそのページを支える `/api/*` 実装も同じブランチで扱う（ページ単体では動作しないため）。`/api/*` の全一覧は `.claude/skills/evaluate-keiba-architecture/scripts/collect_endpoints.py` の出力を参照。

Flask `/api/v1`（DEC-013仕様上の正）・monitorポータルは対象外（FastAPI legacyの主要機能別）。

各ブランチの「現状の実装」「既知の課題」「手動追記用TODO」は [`todo/`](./todo/) 配下に1ファイルずつ用意している
（一覧は [todo/README.md](./todo/README.md)）。

## ブランチ一覧

| ブランチ | 対象領域 | ページURL数 | 検証結果 | 詳細 | TODO |
|---|---|---|---|---|---|
| `feature/admin-ops` | 管理者運用（cron・構造チェック・システム統計・ログ） | 3 | 問題なし | [feature-admin-ops.md](./feature-admin-ops.md) | [todo/admin-ops.md](./todo/admin-ops.md) |
| `feature/betting` | 馭券戦略 | 1 | 問題なし | [feature-betting.md](./feature-betting.md) | [todo/betting.md](./todo/betting.md) |
| `feature/bloodline-pedigree` | 血統・種牡馬クラスタ | 7 | 問題なし | [feature-bloodline-pedigree.md](./feature-bloodline-pedigree.md) | [todo/bloodline-pedigree.md](./todo/bloodline-pedigree.md) |
| `feature/core-platform` | コアプラットフォーム（ヘルス/認証/ダッシュボード） | 3 | 問題なし | [feature-core-platform.md](./feature-core-platform.md) | [todo/core-platform.md](./todo/core-platform.md) |
| `feature/cushion` | クッション値（トラックコンディション） | 0 | 解決済み・feature/に復帰 | [feature-cushion.md](./feature-cushion.md) | [todo/cushion.md](./todo/cushion.md) |
| `feature/growth-curve` | 成長曲線 | 1 | 問題なし | [feature-growth-curve.md](./feature-growth-curve.md) | [todo/growth-curve.md](./todo/growth-curve.md) |
| `feature/horse-profile` | 馬プロフィール・馬名検索・関係者統計 | 0 | 問題なし | [feature-horse-profile.md](./feature-horse-profile.md) | [todo/horse-profile.md](./todo/horse-profile.md) |
| `feature/model-training` | モデル学習・シミュレーション・バックフィル | 0 | 問題なし | [feature-model-training.md](./feature-model-training.md) | [todo/model-training.md](./todo/model-training.md) |
| `feature/monitor-quality` | 監視・データ品質チェック | 2 | 問題なし | [feature-monitor-quality.md](./feature-monitor-quality.md) | [todo/monitor-quality.md](./todo/monitor-quality.md) |
| `feature/myostatin` | ミオスタチン遺伝子解析 | 1 | 問題なし | [feature-myostatin.md](./feature-myostatin.md) | [todo/myostatin.md](./todo/myostatin.md) |
| `feature/odds-final-odds` | オッズ・最終オッズ | 0 | 解決済み・feature/に復帰 | [feature-odds-final-odds.md](./feature-odds-final-odds.md) | [todo/odds-final-odds.md](./todo/odds-final-odds.md) |
| `feature/race-detail` | レース詳細・予測表示 | 1 | 解決済み・feature/に復帰 | [feature-race-detail.md](./feature-race-detail.md) | [todo/race-detail.md](./todo/race-detail.md) |
| `feature/race-quality` | レース質分析 | 1 | 問題なし | [feature-race-quality.md](./feature-race-quality.md) | [todo/race-quality.md](./todo/race-quality.md) |
| `feature/scraping-queue` | スクレイピング・キュー管理 | 4 | 解決済み・feature/に復帰 | [feature-scraping-queue.md](./feature-scraping-queue.md) | [todo/scraping-queue.md](./todo/scraping-queue.md) |
| `feature/track-speed` | トラックスピード指標 | 2 | 問題なし | [feature-track-speed.md](./feature-track-speed.md) | [todo/track-speed.md](./todo/track-speed.md) |
| `feature/tracking-difficulty` | 追走難度 | 1 | 解決済み・feature/に復帰 | [feature-tracking-difficulty.md](./feature-tracking-difficulty.md) | [todo/tracking-difficulty.md](./todo/tracking-difficulty.md) |

合計: 27 ページURL（`src/api/app.py` 全体では228ルート定義・225ユニークURLパスあり、うち`/api/*`のJSON APIが198件、ページが27件〈`/login`のGET/POSTを1件と数えると27件〉）

### ページが0件のブランチ（削除せず保持）

以下のブランチは対象領域が `/api/*` のみで構成されており、ページURL基準では対象が無い。
削除候補として提示したが、2026-09-30時点でユーザの判断により削除せず保持することとした:

- `feature/odds-final-odds`（解決済み・保持継続）
- `feature/model-training`
- `feature/cushion`（解決済み・保持継続）
- `feature/horse-profile`

## 運用ルール

- 新しいページURLを追加する場合は、まず該当する機能領域のブランチが無いか本ファイルを確認する。
- 該当ブランチが無い場合は `feature/<領域名>` で新規作成し、本ディレクトリに対応するmdファイルを追加する。
- ブランチ命名は原則 `feature/<kebab-case>` とする。ただし検証（本ドキュメントの一つずつの確認作業等）で
  実際に分類ミス・設計上の問題が見つかり修正した場合は `hotfix/<kebab-case>` にリネームする
  （問題が見つからなかったブランチは `feature/` のまま）。
- DEC-013により新規APIは本来 `src/api/flask_app.py` / `src/api/v1/` 側に実装すべきであり（`.claude/skills/debug-refactor/SKILL.md` 参照）、legacy機能ブランチでの新規ページURL追加は既存機能の修正・移行作業を想定したものである。
