# 公開リポジトリから netkeiba の取得データを外した記録

方針: netkeiba 利用規約 第14条（不特定多数に知りうる状態にしない）。**HEAD から外すだけで、履歴は書き換えていない**（履歴には残る）。実値はこの文書に書かない。

## 外したもの
| 対象 | 件数 | 内容 | 代替 |
|---|---:|---|---|
| `dev_fixture/real_day.tar.gz`、`MANIFEST.json` | 2 | UI 最小動線の実データ 1 開催日（出馬表・結果・オッズ） | `dev_fixture/.gitignore`（ローカル専用）。`make dev-data` は合成モックだけで動く。実データ日は各自 `capture_dev_day` → `make dev-fixture-pack` |
| `docs/requirements/data/scrape_process_samples/*.json` | 25 | 実スクレイプ JSON（`nk_*`） | 同ディレクトリの `.gitignore`。README は書き出し手順のみ残した |
| `docs/requirements/data/scrape_process.md` の埋め込みサンプル | 24 ブロック | 上の同一内容 | 置き場の対応表のみ残し、サンプルは注記に置換 |
| `notebooks/megu_index/output/nb01/megu_dataset.parquet`、`nb04/megu_index.parquet`、`nb05/megu_final.parquet` | 3 | 約 30 万行の行単位データ（race_id・horse_id・着順・タイム） | 入れ子 `.gitignore`。ノートブックで再生成 |
| ノートブック 8 本の出力セル | 12 セル | race_id と馬名・オッズ等を含む表出力 | 出力のみ削除（コードは維持） |

## 残したもの（理由）
| 対象 | 理由 |
|---|---|
| `docs/requirements/data/schemas/observed/*.json`（37） | 型・件数・キー範囲・適合数だけで、値のサンプルを含まない（走査で確認）。`schema_infer` が再生成する |
| `models/final_odds_bundle.json`、`config/pace_type_models.json`、`config/tracking_difficulty_v2/*` | 学習済みの係数・特徴量名・指標（自前の成果物）。生データを含まない |
| `docs/operations/slack_samples/*.json` | 合成モックの名前（実データ由来ではない） |
| `data/jra_baba/`、`data/meta/course_master/` | JRA 公式・コース情報（netkeiba 由来ではない想定。要確認欄参照） |

## 要確認
| 項目 | 内容 |
|---|---|
| 履歴に残る実データ | 過去コミットに `real_day.tar.gz`、サンプル JSON、行単位 Parquet が残る。履歴の書き換え（filter-repo＋強制 push）は影響が大きく、すでに clone された分には効かない。**非推奨**。代替は『今後置かない』ことと、必要なら規約上の確認 |
| `data/research/other_predictor_umauma/`（54）、`docs/other_predictor/`（3） | netkeiba ではなく**第三者の記事の写し**（raw_notes など）。著作権・規約の面で別途確認し、不要なら外す |
| `data/research/bloodline/**` | netkeiba 由来データの集計（種牡馬×条件の件数・率）。行単位ではないが、公開の可否を判断する |
| `notebooks/megu_index/output/nb01/track_explorer.html` ほか HTML | 集計の可視化。race_id が数十件埋め込まれている。許容か判断する |
| `data/jra_baba/`、`data/meta/course_master/` | JRA 公式の利用規約を確認（URL をいただければ読む） |
| tests・docs の個別の実 ID | race_id・horse_id は事実の識別子。値サンプル（馬名・オッズ・払戻）の混入は走査で未検出だが、網羅は保証しない |

## 再発防止
- `scripts/dev/secret_scan.sh` が `dev_fixture/*`・`scrape_process_samples/*.json`・行単位 Parquet・`data/dev_mock|local` の追加を拒否する（pre-push、`--tracked`）。
