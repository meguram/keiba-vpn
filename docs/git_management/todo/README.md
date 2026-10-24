# ブランチ別 TODO 一覧

`feature/XXX` 各ブランチの「現状の実装」「目標（推測）」「削除してよいライン」「既知の課題」「TODO」を
1ファイルずつ管理する。

**VPS/GCPの役割分担について（2026-10-02決定）**: VPS（ConoHa、2GB軽量）とGCPのどちらかを
選ぶのではなく、**役割分担**する方針が決定した。VPSはページ公開・ユーザー向けAPIサービング専用
（必要最低限のファイルデータのみ）、GCPはスクレイピング・ML学習・スケジュール実行（定期バッチ）
専用。機能領域ごとの担当環境マッピング・ブリッジが必要な点（Cloud SQL化・モデル配信同期等）は
[`../../operations/deployment-vps-vs-gcp.md`](../../operations/deployment-vps-vs-gcp.md)を参照。
ジョブ・プロセス単位の責務分担表は[`../../operations/vps-gcp-responsibilities.md`](../../operations/vps-gcp-responsibilities.md)。
該当ファイル（admin-ops・scraping-queue・core-platform・monitor-quality・bloodline-pedigree・
cushion・model-training・race-quality・race-detail・tracking-difficulty・track-speed・myostatin・
horse-profile・odds-final-odds）の「既知の課題」に個別の背景を記載している。各ファイルの
「TODO（手動追記用）」セクションは、担当環境に応じて
**「共通TODO（ホスト方式に関係ない）」「VPS側（サービング）に残るTODO」
「GCP側（スクレイピング・ML・スケジュール実行）のTODO」**の3分類で再構成済み
（依存が無いTODOファイル=betting・growth-curveは共通TODOのみ）。

## 各セクションの意味

- **現状の実装**: 2026-09-30時点でコード上できていること（docstring・実装読解ベース）。
- **目標（推測）**: このブランチ（機能領域）が最終的にユーザに向けて何を実現したいのかを推測したもの。
  断定ではなく推測なので、実態とズレていたら書き換えてよい。
- **このラインまで実装できたらブランチを消してよい**: 「目標（推測）」を受けて、ユーザに向けての
  実装が完了したと言える具体的な状態。ここに書かれた状態が実際に満たされたら、このブランチ
  （＝この機能領域の開発タスク）は完了とみなし、ブランチを削除してよい。
- **既知の課題**: 過去の調査で判明済みの問題。
- **TODO（手動追記用）**: 自由記述のチェックリスト。

## 使い方

- 手動でTODOを追記する場合は、該当ブランチのファイルの `## TODO（手動追記用）` セクションに
  `- [ ] 内容` の形式で追記する（優先度・担当・期限などは自由書式でよい）。
- 完了したTODOは削除せず `- [x]` にチェックして残す（対応履歴として）。
- コードを変更して機能が変わった場合は `## 現状の実装` セクションも合わせて更新する。
- 「目標（推測）」や「削除してよいライン」が実態とズレていると感じたら、遠慮なく書き換えてよい
  （推測ベースのため、認識が変わったら更新するのが正しい運用）。
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

機能領域ではなく**作業する環境（開発PC / 学習PC / VPS / GCP）**で並べ直した一覧は [environment-tasks.md](./environment-tasks.md)。

各ファイルの「対象エンドポイント」の正確な一覧（method/path/handler/行番号）は
`../feature-<branch>.md`（ページURL）および
`.claude/skills/evaluate-keiba-architecture/scripts/collect_endpoints.py` の出力（全`/api/*`含む）を参照。
