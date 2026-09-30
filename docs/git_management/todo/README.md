# ブランチ別 TODO 一覧

`feature/XXX` 各ブランチの「現状の実装」「既知の課題」「TODO」を1ファイルずつ管理する。

## 使い方

- 手動でTODOを追記する場合は、該当ブランチのファイルの `## TODO（手動追記用）` セクションに
  `- [ ] 内容` の形式で追記する（優先度・担当・期限などは自由書式でよい）。
- 完了したTODOは削除せず `- [x]` にチェックして残す（対応履歴として）。
- コードを変更して機能が変わった場合は `## 現状の実装` セクションも合わせて更新する。
- 新しいブランチ（機能領域）を追加した場合は、このディレクトリに同じ構成のファイルを追加し、
  下の一覧にも行を追加する。

## ファイル一覧

| ファイル | ブランチ | 対象領域 |
|---|---|---|
| [core-platform.md](./core-platform.md) | `feature/core-platform` | コアプラットフォーム（ヘルス/認証/ダッシュボード） |
| [scraping-queue.md](./scraping-queue.md) | `feature/scraping-queue` | スクレイピング・キュー管理 |
| [monitor-quality.md](./monitor-quality.md) | `feature/monitor-quality` | 監視・データ品質チェック |
| [race-detail.md](./race-detail.md) | `feature/race-detail` | レース詳細・予測表示 |
| [tracking-difficulty.md](./tracking-difficulty.md) | `feature/tracking-difficulty` | 追走難度 |
| [race-quality.md](./race-quality.md) | `feature/race-quality` | レース質分析 |
| [odds-final-odds.md](./odds-final-odds.md) | `feature/odds-final-odds` | オッズ・最終オッズ |
| [admin-ops.md](./admin-ops.md) | `feature/admin-ops` | 管理者運用（cron・構造チェック・システム統計・ログ） |
| [model-training.md](./model-training.md) | `feature/model-training` | モデル学習・シミュレーション・バックフィル |
| [betting.md](./betting.md) | `feature/betting` | 馭券戦略 |
| [bloodline-pedigree.md](./bloodline-pedigree.md) | `feature/bloodline-pedigree` | 血統・種牡馬クラスタ |
| [cushion.md](./cushion.md) | `feature/cushion` | クッション値（トラックコンディション） |
| [track-speed.md](./track-speed.md) | `feature/track-speed` | トラックスピード指標 |
| [myostatin.md](./myostatin.md) | `feature/myostatin` | ミオスタチン遺伝子解析 |
| [growth-curve.md](./growth-curve.md) | `feature/growth-curve` | 成長曲線 |
| [horse-profile.md](./horse-profile.md) | `feature/horse-profile` | 馬プロフィール・馬名検索・関係者統計 |

各ファイルの「対象エンドポイント」の正確な一覧（method/path/handler/行番号）は
`../feature-<branch>.md`（ページURL）および
`.claude/skills/evaluate-keiba-architecture/scripts/collect_endpoints.py` の出力（全`/api/*`含む）を参照。
