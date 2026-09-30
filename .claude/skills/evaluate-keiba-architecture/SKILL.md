---
name: evaluate-keiba-architecture
description: Runs a periodic architecture health check across keiba-vpn's three HTTP route layers (FastAPI legacy src/api/app.py :8000, Flask /api/v1 src/api/flask_app.py+src/api/v1/ :5000, monitor src/monitor/app.py :9090), checks data/config path consistency, flags redundant or duplicated designs, and writes an HTML report to data/skill_logs/architecture_eval/<yyyymmdd>.html. Use when the user asks for an architecture health check, endpoint/route audit, path consistency check, or a periodic review of routing/data-path integrity across the whole app.
---

このリポジトリ（keiba-vpn）は3層のHTTPルーティングとGCS/ローカルの二重データストレージを持つ、複雑度の高いアプリケーションです。
本スキルは**定期的なヘルスチェック**として、全エンドポイントとデータ/パスの整合性、冗長設計の有無を調査し、
わかりやすいHTMLレポートを1本作成することがゴールです。実装の変更は行わず、**調査とレポート作成のみ**を行う。

## 0. 対象とゴール

3つのルーティング層（`AGENTS.md` / `docs/operations/service-endpoints.md` / `docs/decisions/DEC-013-*.html` / `docs/decisions/DEC-015-*.html` が仕様の正）:

| 層 | ファイル | ポート | 位置づけ |
|---|---|---|---|
| FastAPI legacy | `src/api/app.py`（1万行超） | :8000 | DEC-013により段階廃止予定。新規ルート追加禁止 |
| Flask `/api/v1` | `src/api/flask_app.py` + `src/api/v1/services.py` / `delegates.py` | :5000 | 仕様上の正。新規APIはここに追加する |
| monitor | `src/monitor/app.py` | :9090 | 開発者専用監視ポータル（エンドユーザー非公開）。仕様書 `docs/monitoring-portal.md` |

ゴール: 各層の**全エンドポイント一覧** + **ヘルス判定** + **パス/データ整合性チェック** + **冗長設計の指摘と解決策**を1本のHTMLレポートにまとめる。

## 1. エンドポイント一覧の収集（スクリプト実行）

```bash
python3 .claude/skills/evaluate-keiba-architecture/scripts/collect_endpoints.py > /tmp/keiba_endpoints.json
```

AST解析で3ファイルのルート定義（method / path / handler名 / 行番号）をJSONで列挙する（巨大な `app.py` をimportせず静的解析するため安全）。
`src/api/v1/services.py` / `delegates.py` はルート定義を持たないサービス層なので対象外（意図的な設計）。

## 2. パス整合性チェック（スクリプト実行）

```bash
python3 .claude/skills/evaluate-keiba-architecture/scripts/check_paths.py > /tmp/keiba_path_check.json
```

`AGENTS.md` / `docs/operations/service-endpoints.md` のバックティックパスと `config/settings.yaml` 内のパス文字列について、実在確認（`exists: true/false`）を行う。

**既知の判断基準**: `data/` 以下は容量が大きく、開発機によっては一部ディレクトリ（GCS専用カテゴリ等）が意図的に未マテリアライズの場合がある。`exists: false` が出たら即「バグ」と断定せず、次を確認してから重大度を決める。
- そのパスが**コード内で実際に参照されている**か（`Grep`で該当パスのキーを検索）。参照されていない設定・記述なら「デッドコード/デッドドキュメント」として指摘。
- `config/settings.yaml` の `paths:` セクション（`data/raw` / `data/processed` / `data/mock` / `pipelines`）は既知の疑わしい候補 — 現行の `data/features/` 等（`AGENTS.md` 記載）とは異なる旧レイアウトを指している可能性が高い。実際に読んでいるコードがあるか `grep -rn "paths\[.raw_data.\]\|settings\[.paths.\]"` 等で確認する。

## 3. エンドポイント単位のヘルスチェック（手動判断・最重要）

スクリプトは「存在するルートの列挙」のみ。各エンドポイントについて次を確認し、`OK` / `警告` / `重大` を判定する。

- ハンドラ本体（`Read`で該当行を確認）が参照するデータ（`HybridStorage`呼び出し・GCSカテゴリ名・DBテーブル・ローカルパス）が実在/到達可能か。
- **同一・類似パスが複数層に重複していないか**（例: FastAPI legacyとFlask v1に同名エンドポイントが両方ある）。ヘルスチェック(`/api/health` 系)のように層ごとに必要な重複は正常、機能重複は移行負債として指摘する。
- `docs/operations/service-endpoints.md` / DEC-013 / DEC-015 の方針と実装が一致しているか（特に `src/api/app.py` に新しいルートが追加されていないか）。
- monitorのルートが `docs/monitoring-portal.md` の仕様と一致しているか。

判定基準の目安:

| Status | 基準 |
|---|---|
| OK | 参照先データ・仕様書との整合ともに問題なし |
| 警告 | 参照先が疑わしい／仕様書と軽微な不一致／将来的なリスクだが即時の実害なし |
| 重大 | 参照先データが存在しない・仕様（DEC-013等）に明確に違反・重複実装で保守コストが増大している |

## 4. レポート作成

1. テンプレート `.claude/skills/evaluate-keiba-architecture/templates/report_template.html` を読み、構成に沿って**調査結果で埋めた実体のHTML**を新規作成する（テンプレート内のHTMLコメント・サンプル行は削除する）。
   **配色（`<style>`）は変更せず、テンプレートの CSS 変数・クラスをそのまま使う**（下記「配色ルール」を参照。テンプレートが唯一の正）。
2. 出力先: `data/skill_logs/architecture_eval/<yyyymmdd>.html`（日本時間の実行日、例 `20260930.html`）。
   - ディレクトリが無ければ作成する。
   - 同日に複数回実行した場合は上書きする（過去日のファイルは削除しない＝履歴として残す）。
3. コミットハッシュは `git rev-parse --short HEAD` で取得し、`{{COMMIT}}` に埋める。
4. レポートに含める必須セクション（テンプレート参照）: サマリー／層別エンドポイント一覧（3テーブル）／パス整合性チェック結果／冗長設計・移行負債の指摘（各issueに解決策を明記）／次回への引き継ぎ事項。

## 4-1. 配色ルール（テンプレートに実装済み・変更しないこと）

過去に「彩度の高い背景色＋彩度の高いテキスト色」（GitHub風バッジ: 緑背景+緑文字など）でレポートを作ったところ、
特に警告色（`#fab219` 系）が光サーフェス上でコントラスト1.79:1しかなく（`dataviz` skillの `references/palette.md`
「Status palette」で実測済み）、視認性が悪く「配色が悪い」という指摘を受けた。

そのため本テンプレートは **dataviz skill の `references/palette.md`（Status palette / Chart chrome & ink）を
そのまま採用**し、次のルールで実装している。新しいセクションを追加する場合もこのルールに従う。

- ステータス（OK/警告/重大）は**色だけで意味を持たせない**。必ず「8pxの色ドット + 黒系テキストのラベル」の組み合わせ
  （`class="status status-ok|status-warn|status-crit"`）で表現する。彩度の高い色を背景やテキストに直接使わない。
- ページ全体は暖色系ニュートラル（`--page-plane` `#f9f9f7`、カード/テーブルは `--surface` `#fcfcfb`）。
  彩度のあるステータス色は「ドット」「issueの左ボーダー」「カードの小さいラベル横のドット」にのみ使う。
  数字（カードの `.num`）やテーブルのテキストは常に `--ink-primary`（ほぼ黒）。
- 新しい色を追加する場合は `dataviz` skill を呼び、`references/palette.md` の値を使うか、
  `scripts/validate_palette.js` で検証してから採用する（自己判断で色を決めない）。

## 5. 実行後

- レポートの絶対パスをユーザーに提示する。
- 重大判定の件数を一言で要約して伝える。
- 本スキルは調査・レポート作成のみを行う。コードやデータの修正はスコープ外（修正が必要な場合は別途ユーザーに確認の上、`debug-refactor` スキル等で対応する）。
- レポートファイル自体をgitにコミットするかはユーザー判断（本スキルはコミットしない）。
