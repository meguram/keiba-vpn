# 当て込まれた係数・集計表を公開リポジトリに置かない（分類・影響・対策）

方針（2026-10-22 確定）: 学習や集計で**データに当て込んで決めた値**（重み・係数・補正表・集計表）は git で管理しない。**人が決める設定・しきい値**は git に残す。
対象一覧は `config/fitted_artifacts.txt`（機械が読む正）。実ファイルは各環境に残し、配布は `scripts/ops/fitted_artifacts.sh`（束ねる／展開）→ `src/ops/model_publish.py`（GCS の `models/fitted_artifacts/`）。

## 分類
**git から外したもの（当て込まれた値・集計表。(91 ファイル)**
| 種別 | パス | 理由 |
|---|---|---|
| 学習済みの重み・符号化器 | `models/*.pkl`、`models/encoder.json`、`models/final_odds_bundle.json`、`models/pace_predictor/*`、`config/{growth_index,tracking_difficulty_v2}/model*.txt`、`config/expected_megu/model.json`、`config/pace_type_models.json` | 学習で得た重み |
| 当て込まれた係数・表 | `config/growth_index/{params,ped_profile}.json`、`config/megu_predict_{params,condition_weights}.json`、`config/megu_v3a/{centers.json,month_offsets.csv,par.csv}`、`config/gate_bias.json`、`config/class_benchmark.json`、`config/pace_coherence{,_v2}.json`、`config/tracking_difficulty_v2/{meta,position_effect}.json`、`config/megu_excusable.json`、`config/drift_top_features.json` | 係数・補正量・基準表・重要度（データから決めた値） |
| 集計表 | `data/research/bloodline/**`（62 件）、`data/research/course_bloodline`（リンク） | 血統×条件の集計値 |

**git に残すもの（人が決める設定・しきい値・仕様）**: `kesi_mark.json`、`megu_mark.json`、`mark_rule.json`、`maintenance.json`、`ports.env`、`anti_scraping.json`、`behavior_flags.json`、`fetch_owner.json`、`fetch_shadow.json`、`job_owner.json`、`entries_schedule.json`、`paid_zone.json`、`public_fields.json`、`members.json`、`settings.yaml`、`gcp_job_map.json`、`cost_assumptions.json`、`migration_phases.json`、`serving_state.json`、`decoupling_roles.json`、`modeling/*`、`model_feature_groups.json`、`frozen_tracking.json`、`tp59/tp62_frozen_rule.json`（凍結した判断）、`megu_v3a/{spec,meta}.json`、`megu_par_time_v2_meta.json`（学習の記録）、`drift_periods.json` ほか。

## 判断が割れるもの（外していない。ユーザー判断）
| パス | 割れる理由 |
|---|---|
| `config/ev_calibration.json`、`ev_calibration_pc.json`、`prob_calibration.json` | 較正の係数（当て込み）と、段の閾値・有効化の設定（人が決める）が同じファイルに混在。UI と推奨度のテストが閾値を読む。分けるなら、係数と閾値で 2 ファイルに分割する改修が先。 |
| `config/pace_type.json` | モデルの定義、コースの幾何情報、競馬場コードが混在 |
| `data/meta/course_master/jra_courses.json`、`profiles_A.json` | 数値（距離・コーナー・高低差）が中心で、`evidence`／`start_position_A` に出典名つきの短い事実の記述（自分の言葉）。JRA 公式の高低断面図の数値化は事実データだが、出典は JRA。残した |
| `data/jra_baba/cushion_*.json` | JRA 公開情報の蓄積。JRA サイトの規約で事実データの扱いは未記載 |
| `data/meta/dormello/*.csv`、`data/knowledge/horse_peds_domain.html` | 自前の手作業の調査・生成物。外部データの転記を含みうる |
| `docs/requirements/data/schemas/observed/` | 型・件数・キー範囲のみ（値のサンプルは無いことを走査で確認済み） |

## 影響と対策
| 影響を受けるもの | 何が起きるか | 対策 |
|---|---|---|
| ランタイム（ホスト） | ファイルは残るので変化なし。`git pull` しない運用 | 変更なし |
| テスト・CI・新規 clone | 実ファイルが無く、読むテストが失敗する | `tests/conftest.py` の `_FIT_REQUIRES` で、ファイルが無ければ skip。クリーンな worktree で全体 2960 件通ることを確認 |
| 文書の参照整合テスト | 文書が言及するパスが実在しない | `config/fitted_artifacts.txt` に載るパスは不在を許容 |
| データカタログ | E05 のサンプルが読めない | ファイルが無ければ「サンプルなし」と注記 |
| env-parity の `release_manifest`（config のハッシュ） | ハッシュは**実ファイル**から取るので、環境間で版が違えば差分として検出される（望ましい動作）。git 管理外になっても対象から外れない | 変更なし。配布後に照合で揃いを確認 |
| `stage_deploy` | `--ref HEAD` では配れない（git に無い） | **作業ツリー指定（`--ref` なし）**で積む。または `fitted_artifacts.sh pack` → GCS 経由 |
| VPS・Cloud Run・Jobs | イメージの `COPY config/` に入らない。また `gcloud builds submit` は `.gitignore` のファイルを既定で送らない | ビルド前に `model_publish fetch fitted_artifacts` → `fitted_artifacts.sh unpack` でコンテキストへ展開し、`--ignore-file`（`.gitignore` を使わない除外指定）で送る。または起動時に GCS から取得（`decoupling_plan.md` D2） |
| 履歴除去（強制 push）後 | 履歴から元の値を取り戻せない | **除去前に `scripts/ops/purge_preflight.sh check` が全部 OK**（実体が ホスト・GCS・リポジトリ外バックアップ の 3 か所に揃う）ことを必須にする |

## 今の確認結果（`purge_preflight.sh check`、2026-10-22）
| 場所 | 一致 | 無い | 内容が違う | 次にやること |
|---|---:|---:|---:|---|
| (a) ホストの実ファイル | 89/91 | 1 | 1 | `config/drift_top_features.json` がホストに無い → 作業ツリー指定のステージで配置。`models/final_odds_bundle.json` は**ホスト版が正**（開発 PC の複製が古い）→ **ホストで pack する**か、ホスト版を取得して照合 |
| (b) GCS `models/fitted_artifacts/` | 未確認 | - | - | ホストで `fitted_artifacts.sh pack` → `model_publish publish fitted_artifacts <dir>` を行い、`KEIBA_GCS_BUCKET` を指定して再確認 |
| (c) リポジトリ外バックアップ | tar.gz 0/91（bundle は作成済み・検証 OK） | 91 | 0 | `purge_preflight.sh backup`（git bundle と tar.gz）。ホスト版を取り込んだあとに作る |

## コース資料（`data/meta/course_master/**`）の監査（2026-10-22）
第三者サイト（JRA-VAN すごトク、競馬ラボ、Wikipedia など）の**文の抜粋**を含むものは、公開リポジトリから外した（権利の都合。JRA-VAN は公表禁止）。
| ファイル | 内容 | 扱い |
|---|---|---|
| `geo/jravan_start.json` | JRA-VAN すごトクの各コース解説から抜いた発走位置の文（約140件） | **HEAD から外す・履歴から除去**・ローカル保持 |
| `geo/start_descriptions.json` | 独立収集 A/B/C（JRA 公式、netkeiba・競馬ラボ、Wikipedia・JRA-VAN 等）の記述と URL | 同上 |
| `geo/route_audit.json` | 上の 2 つとの突き合わせ表（文を含む） | 同上 |
| `geo/start_overrides.json` | 発走位置の上書き。根拠の `summary`（他サイトの文）を削除、出典名と URL は残した | 整理して残す・履歴から除去して整理版を戻す |
| `corrections.json` | 補正値と根拠。JRA 公式本文・JRA-VAN の引用文を、自分の言葉の事実に書き換え | 同上 |
| `geo/verification_notes.json` | 検証メモ。他サイトの名前とファイル名の言及を整理 | 同上 |
| `geo/routes.json`、`geo/osm/*.json` | OpenStreetMap 由来（ODbL。帰属表示あり） | 残す |
| `jra_courses.json`、`profiles_A.json`、`verification_issues.json` | 数値と自前の短い記述 | 残す（要確認に記載） |
`track_geometry.py` は、独立した記述（`start_descriptions.json`）が無い環境では走行ルートの監査を行わず（利用不可）、既存の結果を変えない。`test_route_audit_*` は `route_audit.json` が無ければ skip。
**未対応**: `docs/html/modeling/chapters/course_master_verification.html` が、`jravan_start.json` の記述と監査表の内容に言及している（`docs/html` の編集は `book.html` の再生成が要るため、触っていない。親側で、該当節の削除または言い換えと再生成が必要）。
