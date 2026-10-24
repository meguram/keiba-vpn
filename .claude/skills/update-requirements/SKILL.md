---
name: update-requirements
description: Regenerates the single-page product requirements document docs/html/index.html for keiba-vpn from the current git state, code, design docs (DEC/AREA), data layout and task state. Covers requirements, KPIs, architecture diagrams, data/ML workflows, user data-delivery flow, implementation-status matrix (implemented / partial / MOCK / TODO) and a dependency-ordered TODO schedule as the last section. Use when the user runs /update-requirements or asks to update, refresh, or regenerate the requirements/spec overview document.
---

keiba-vpn の**要件定義書（`docs/html/index.html`）を、実行時点の最新状態で再生成する**スキル。
この HTML 1 本を読めば、目的・要件・KPI・構成・ワークフロー・ユーザーへの提供フロー・未実装事項のすべてが把握できる状態にするのがゴール。
書き換える対象は `docs/html/index.html` のみ。**コード・データ・他ドキュメントは変更しない**（コミットもしない）。

> 初回実行時、既存の `index.html`（iframe 型ドキュメントポータル）は要件書に置き換わる。旧ポータルのナビゲーションは §13「関連ドキュメント」が引き継ぐ（旧版は git 履歴に残る）。

## 絶対ルール

1. **事実のみを書く。** 実装状況・KPI 現状値・エンドポイント・モデル一覧は、コード/設計書/評価レポート/git で裏を取れたものだけ。推測で埋めない。
2. **未設計・モック・仮データは隠さず明示する。** コード上で設計が未了のもの、モックデータ/モックモデル（`src/api/stg_mock.py`、`data/mock`、ダミー推論など）は、該当箇所に必ず `MOCK` / `TODO` タグを付け、§14 に TODO として登録する。
3. **ステータスは色だけで伝えない。** `<span class="tag done|partial|mock|todo|dep">ラベル</span>` を使う（ドット + ラベル文字）。テンプレートの CSS を変更しない。
4. **測れていない KPI は「未計測」と書き、計測基盤の構築を TODO にする。**
5. 文書は日本語。固有名（DEC-xxx、パス、ポート、モデルキー）は実物と一字一句一致させる。

## 手順

### 1. 現状の収集

```bash
python3 .claude/skills/update-requirements/scripts/collect_state.py > /tmp/keiba_state.json
```

出力先は任意の作業用パスでよい（並列実行なら別パスにする）。引数は `--doc <前回版パス>` のみ（既定 `docs/html/index.html`）。未知の引数はエラーになる。

JSON の内容: 前回要件書のメタ（`previous_doc`：前回 commit・既存 TODO 行）／git（現 commit・前回からの差分 `since_prev`）／DEC・AREA 一覧とステータス／`src/` パッケージ規模／全 API ルート（3 層）／フロント全ページ／MLflow モデル・settings／`data/` 構成／cron・Docker・CI・alembic／テスト構成／TODO・mock・stub 痕跡／関連ドキュメント一覧。
ここで得るのは**下調べ**。最終判定は手順 2・3 で実物を読んで行う。収集結果は網羅性を保証しない。**収集値とソースの再集計が食い違ったら再集計を採用**し、食い違いがあったことを完了報告に一言書く。件数を文書に書く前に、ソースで裏取りする（例: API ルート数は `src/api/flask_app.py` の `register_blueprint` と `src/api/v1/routes/*.py` を、画面数は `frontend/app/**/page.tsx` を `Grep`/`Glob` で数え直す）。

- `frontend_pages` の `api_paths` / `uses_mock_flag` は画面ごとの API 呼び出しとモック切替の**手がかり**。データ出所・認可の列は、ここから該当コードを読んで自分で埋める。
- `task_docs` は TODO の実態が書かれたタスク管理ドキュメント（`docs/git_management/todo/*.md`。`environment-tasks.md` はユーザー自身の作業・方針項目）。未完チェックボックスや記載タスクは §14 の TODO 候補として必ず拾う。

- `git.prev_commit_valid` が true のときだけ `git.since_prev.changed_by_area` を使い、**変更の大きい領域から重点的に再確認**する（文書は毎回全体を再生成する。差分だけのパッチにしない）。
- `prev_commit_valid` が false（記録された commit が git に無い／未記録）なら差分は使えない。全領域を同じ深さで再確認し、§0 に「前回版の commit が無効のため全面再確認した」と一文で注記する。
- `previous_doc.is_requirements_doc` が false（旧ポータル・前回版なし）なら初回扱い。§0 には「初回生成」と書く（「前回版の commit が無効」とは書かない）。前回版が自分の中で矛盾している（本文の commit 表記と meta が違う、古い §0 注記が残る等）場合は meta と git を信じ、§0 の来歴注記は今回の `since_prev` から書き直す。
- 「前回からの主な変化」を報告するときの基準は、前回版の本文ではなく **git の差分（`since_prev`）と前回 TODO 表**。前回本文が空・スタブなら、その二つだけで書く。

### 2. 設計の正を読む

優先順: `docs/decisions/MASTER.html`（決定の総覧）→ `DEC-*.html` / `AREA-*.html`（特に AREA-01 アプリ要件、DEC-022 予測タスク、DEC-023 コスト、DEC-024 精度目標、DEC-026 馬券戦略）→ `AGENTS.md` → `docs/operations/*.md`（`service-endpoints.md`、`cost-estimate.md`、`vps-gcp-responsibilities.md`）→ `docs/html/design/FULLSTACK_ARCHITECTURE.html`・`ml_pipeline.html`・`inference_services.html`・`mlflow_platform.html` → `docs/html/data/*` → `docs/git_management/todo/*.md`（`task_docs`。タスク状況の一次情報）。
`superseded` / 廃止予定の決定は現行要件に混ぜず、§10 と該当箇所に「廃止予定」タグで残す。設計書と実装が食い違う場合は**実装を事実として採用**し、食い違い自体を §12 のリスクに 1 件 1 行で「設計側（DEC 等）／実装側／採用した側／対応する T-ID」の形式で記録する。

### 3. 実装状況の判定（証拠必須）

各機能領域・画面・モデル・パイプラインについて、関連コードを `Read`/`Grep` で確認し次のいずれかを付ける。**証拠のファイルパスを必ず併記**する。

| タグ | 基準 |
|---|---|
| 実装済 `done` | 実データ/実モデルで end-to-end に動く実装がある（テストまたは実行経路が確認できる。本物の API を呼ぶ経路をコードで追え、モック分岐が無ければ、専用テストが無くても `done`。テスト無しを理由欄に書く） |
| 部分実装 `partial` | 動くが対象範囲が限定／一部ステップ欠落／手動実行のみ |
| MOCK `mock` | モックデータ・ダミー推論・固定値・スタブで代替している（`stg_mock`、`data/mock`、`mock|dummy|stub` 痕跡の実体確認）。**常時固定値を表示する画面・乱数の疑似ビルダー・学習済みモデルの代わりに固定ヒューリスティックで値を出すもの（例: `lap_predictor`）は MOCK** |
| TODO `todo` | 設計書はあるがコード無し、またはコードも設計も無い |
| 廃止予定 `dep` | DEC で段階廃止が決まっている（例: FastAPI legacy `:8000`） |

**証拠の範囲**: この開発 PC には学習済み成果物・`data/features`・本番環境が無いことがある。「実装済」は「コードとテストがあり、この環境で確認できる範囲で動く」を意味する。実データ・成果物・本番でしか確認できない部分が機能の主体なら `partial` とし、未検証の部分を理由欄に書く（§0 にも一文で注記）。

**境界の判定ルール**: 本物の API 経路があり、環境変数（`NEXT_PUBLIC_MOCK` 等）でモックに切り替えられるだけの画面は `partial`（切替の存在を備考に書く）。本物の経路が無く常にモックの画面は `mock`。ユーザー入力に対して実際に計算するシミュレータ（デフォルト値がサンプルなだけ。例: `/betting-simulation`）は `partial`。データ依存のない静的ページは `done`。廃止予定の層に依存して動いているものは、動作状況のタグに加えて依存を TODO 化する（`dep` は廃止予定の対象そのものにだけ付ける）。

`todo_markers` の痕跡は**誤検出が多い**（テスト名・ドキュメント文字列・`import`など）。必ず該当行の文脈を読んでから採否を決める。

### 4. TODO リストの構築（§14 の中身）

- 手順 3 で `mock` / `todo` と判定したもの、KPI 未計測、設計と実装の食い違い、前回要件書の未完了 TODO を集約する。
- **ID は継続する**: `previous_doc.todos` に同じ作業項目があれば、範囲・依存・文言が変わっても同じ `T-xxx` を使う（変更点は項目欄に書く）。完了していれば `data-status="done"` にして残す（削除しない）。新規は最大 ID + 1（ID は安定ラベルで、表の並び順とは無関係）。
- **前回行の判別・修復**: 前回の行は、項目名か根拠パスが実在の作業と一致すれば対応づける。空・プレースホルダ名で一致しない行、または依存が壊れている（存在しない ID・循環）行は、行を消さず `done` + 項目名に「（取り下げ）」とし、`data-depends` を空にして、表の末尾に固定した Phase「取り下げ」にまとめる（波の計算には含めない）。優先度・種別・規模・根拠は前回値のまま（無ければ `—`）。修復内容は §0 と完了報告に書く。
- **既存行か新規行か**: 担当と成果物が同じなら既存行に追記（ID 維持）、違うなら新 ID。前回 `done` の行は、`since_prev` の差分が根拠パスに触れたときだけ再確認する（根拠欄の更新は可）。
- **前回行の判定順**: ① 項目名が実在の作業に対応すれば（根拠パスが `—` でも、具体的で妥当な名前なら対応とみなす）ID 維持 ② 対応しないものを取り下げ ③ 依存の修復は最後に全行へ。ID は見つけた順に採番し、表の並びとは無関係。
- **タスク文書の網羅**: `task_docs` の未完チェックボックスと `environment-tasks.md` の各タスクは、必ずどれかの T-ID に対応づける（項目欄か根拠欄に出所のタスク ID、例 `A3`、を併記して追跡できるようにする。コミット・判断待ちなどユーザー作業も `TODO` 行にし、根拠欄に「ユーザー作業」と書く）（既存行への統合可）。対応先が無いものは新規行にする。
- 各 TODO に 優先度（P0 ブロッカー / P1 KPI・ユーザー価値に直結 / P2 品質・運用 / P3 改善）、種別（`TODO`|`MOCK`）、依存（TODO-ID）、根拠パス、規模（S/M/L）を付ける。
- **並び順**: ① 依存先が必ず依存元より上（トポロジカル順）② 依存の波ごとに `Phase` ヘッダ行（`<tr class="phase-head">`）でグルーピングする。波 = 依存の深さ（依存のない項目が波 1、依存先の最大の波 + 1 がその項目の波）。波内は優先度 P0→P3 ③ 同優先度は、他の TODO の依存先になっているものを先、次に規模 S→L の順。`data-order` は 1 からの連番。
- 行フォーマットはテンプレート内の GUIDE コメント参照。

### 5. 文書の作成

0. 「全体を再生成」とは、テンプレートから全セクションを組み直すこと。前回版の本文のうち変化が無いと再確認できた文章は再利用してよいが、ずれやすい値（件数・行番号・日付・計測値）は必ずソースから再導出する。
1. `templates/requirements_template.html` を読み、**`{{...}}` をすべて実内容に置換し、HTML コメント（GUIDE）をすべて削除した実体の HTML** を `docs/html/index.html` に書き出す（CSS・JS・セクション ID と順序は変更しない。既存ファイルがあれば上書き）。
2. `{{COMMIT}}` = `git rev-parse --short HEAD`、`{{BRANCH}}` = 現ブランチ、`{{DATE}}` = 実行日（JST, `YYYY-MM-DD`）。未コミット変更があれば §0 に「作業ツリーに未コミット変更あり」と注記する。
3. 図は外部ライブラリ不要の CSS 部品で描く: レイヤー図 = `.lanes/.lane`、フロー図 = `.flow .node/.arrow`。未実装ノードは `node is-todo`、モックは `node is-mock`（破線）。各ノードに実在するパス/ポート/カテゴリ名を併記する。
4. セクション構成（固定・この順）:
   `summary` `overview` `requirements` `kpi` `architecture` `data-flow` `ml-flow` `delivery` `infra` `quality` `decisions` `status-matrix` `risks` `related-docs` `scheduling`
   `scheduling`（§14 TODO 一覧）は**必ず最後**。
5. 必須の内容:
   - §0 サマリー: 件数カードは §11 の集計と一致させる。
   - §2 要件: `FR-xx` / `NFR-xx` の ID、根拠（DEC/AREA）、状況タグ、証拠パス。
   - §3 KPI: 目標値（DEC-024 の精度目標など）／現状値／測定方法。
   - §4 構成図: クライアント(Next.js)・API 3 層（FastAPI legacy :8000 / Flask v1 :5000 / monitor :9090）・推論サービス（MLflow serve）・PostgreSQL/Redis・GCS/ローカル・スクレイパ・cron の関係。
   - §5 データ: ソース → スクレイパ → スキーマ検証 → 保存(GCS/ローカル) → 特徴量ストア → 学習データセット。
   - §6 モデル: MLflow catalog の**全キー**を一覧化し、各モデルの学習/推論/モック状況を記載。
   - §7 提供フロー: 画面（`frontend_pages` 全件）× 使用 API × データ出所 × 認可 × 状況タグ、更新タイミングと鮮度。
   - §11 実装状況マトリクス: 1 行 = 1 項目（領域・モデル・画面・パイプライン。画面は 1 ページ 1 行）、各行に状況タグをちょうど 1 つ。§11 内に凡例や他のタグを置かない。§0 のカード（`data-card`）はこの表のタグ数と一致させる（`dep` はカードに含めず本文で別記）。
   - 本文中の件数（ルート数・画面数・モック数など）は、表の行数から数えて同じデータから書く。手書きで別々に数えない。
   - §13 関連ドキュメント: `docs_index` の全件（`docs/decisions/**` と `docs/git_management/**` を含む）を章別に、`docs/html/index.html` からの相対リンク（`../decisions/…`、`../operations/…` 等）で載せる（旧ポータルの代替）。index 外の文書を補足リンクとして足してもよい（バリデータのリンク検査が効く。到達できない補足リンクは落とすかリンクにしない文字列で書く）。載っていない html があればバリデータが警告する。

### 6. 検証（必須・通るまで直す）

```bash
python3 .claude/skills/update-requirements/scripts/validate_requirements.py
```

セクション順・`scheduling` 最後・未置換プレースホルダ/GUIDE 残り・メタ・TODO の ID 一意/依存先実在/循環なし/依存先が上/`data-order` 昇順・リンク切れを検査する。`[error]` が 0 になるまで修正する。`[warn]`（同フェーズ内の優先度逆転）は依存による例外か確認する。
§0 のカードと §11 のタグ数の一致、§11 の MOCK/TODO 行から §14 の TODO-ID（`T-` + 3 桁。`T-45` など業務用語と区別）への参照、§13 の載せ漏れはバリデータが検査する（`info` の `whole_doc_tags` は文書全体の参考値で、比較対象ではない）。タグは §11 の表の行の状況セルにだけ置き、集計文などの本文には置かない。

### 7. 完了報告

ユーザーに次を**短く**伝える: 出力先 `docs/html/index.html`、今回の commit、実装済/部分/MOCK/TODO の件数、P0 の TODO 上位 3 件（= §14 の表順で先頭から 3 件の P0）。依存順なので最重要とは限らない場合は「実際の最重要経路は T-xxx → T-yyy」と一文添える、前回からの主な変化（前回版がある場合）。ブラウザで開く場合の案内は不要。
**コミットはしない**（するかどうかはユーザー判断）。
