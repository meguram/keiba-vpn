"""データ存在チェックの要件定義（何が・どの環境で・どの程度必要か）。

重要度（level）:
  required     … 無いと FAIL（stg/prod で到達できる必要があるデータ）
  recommended  … 無いと WARN
  optional     … 無くても INFO のみ
  skip         … その環境では評価しない（dev は全データ不要）

dev は「モックで動くこと」だけを要件にする（``make dev-mock`` が全カテゴリのサンプルを生成するので、
レース／馬カテゴリは dev でも required。派生データ（特徴量・血統成果物など）は dev では対象外）。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

ENVS = ("dev", "stg", "prod")
LEVELS = ("required", "recommended", "optional", "skip")


def _lv(stg: str, prod: str | None = None, dev: str = "skip") -> dict[str, str]:
    return {"dev": dev, "stg": stg, "prod": prod or stg}


@dataclass(frozen=True)
class CategorySpec:
    name: str                 # HybridStorage カテゴリ名
    label: str
    scope: str                # race | horse
    due: str                  # shutuba | dayof | weekly | horse  （期限の計算ルール）
    task: str                 # 再取得に使うキュータスク ID（空 = スクレイプ不可の生成物）
    level: dict[str, str] = field(default_factory=dict)
    window_days: int | None = None   # 数値なら「今日±この日数」のレースだけ評価


# 期限ルールは docs/requirements/data/scrape_process.md の SLA に対応:
#   shutuba = SLA1 前日17:00 / dayof = SLA3〜5 当日18:00 / weekly = SLA6 レース後最初の金曜17:00
RACE_CATEGORIES: list[CategorySpec] = [
    CategorySpec("race_shutuba", "出馬表", "race", "shutuba", "race_shutuba", _lv("required", dev="required")),
    CategorySpec("race_result", "確定結果", "race", "weekly", "race_result", _lv("required", dev="required")),
    CategorySpec("race_index", "タイム指数", "race", "shutuba", "race_index", _lv("recommended", dev="required")),
    CategorySpec("race_odds", "単複オッズ", "race", "dayof", "race_odds", _lv("recommended", dev="required")),
    CategorySpec("race_result_lap", "個別ラップ", "race", "weekly", "race_result_lap", _lv("recommended", dev="required")),
    CategorySpec("race_barometer", "調子偏差値", "race", "weekly", "race_barometer", _lv("recommended", dev="required")),
    CategorySpec("race_paddock", "パドック", "race", "dayof", "race_paddock", _lv("optional", dev="required")),
    CategorySpec("race_pair_odds", "2連系オッズ", "race", "dayof", "race_pair_odds", _lv("optional", dev="required")),
    CategorySpec("race_result_on_time", "速報結果", "race", "dayof", "race_result_on_time", _lv("optional", dev="required")),
    # AI 予測は直近分だけ存在する生成物（スクレイプ対象ではない）
    CategorySpec("race_predictions", "AI予測", "race", "shutuba", "", _lv("recommended", "required", dev="required"),
                 window_days=45),
]

HORSE_CATEGORIES: list[CategorySpec] = [
    CategorySpec("horse_result", "馬情報・戦績", "horse", "horse", "horse_profile", _lv("required", dev="required")),
    CategorySpec("horse_pedigree_5gen", "5世代血統", "horse", "horse", "horse_pedigree_5gen",
                 _lv("required", dev="required")),
    CategorySpec("horse_training", "調教", "horse", "horse", "horse_training", _lv("optional", dev="required")),
]

# 馬カテゴリの評価対象（今日から前後この日数のレースの出走馬）
HORSE_WINDOW_PAST_DAYS = 30
HORSE_WINDOW_FUTURE_DAYS = 14
HORSE_RACES_MAX = 300


@dataclass(frozen=True)
class ArtifactSpec:
    """ローカルに置かれる派生物（特徴量・血統成果物・モデル・設定など）。"""

    id: str
    label: str
    group: str
    patterns: tuple[str, ...]          # リポジトリ相対の glob。{Y} は年、{F} は特徴量ルート候補
    level: dict[str, str]
    hint: str = ""
    per_year: bool = False             # True なら年ごとに存在を確認（不足年を可視化）
    min_files: int = 1


FEATURE_ROOTS = ("data/features", "data/local/features")   # 実装は後者、文書は前者（揺れ）

_F = "{F}"
ARTIFACTS: list[ArtifactSpec] = [
    ArtifactSpec("C01", "race_result_flat", "特徴量", ("data/page_reference/tables/{Y}/race_result_flat.parquet",
                 "data/local/tables/{Y}/race_result_flat.parquet"), _lv("required", "recommended"),
                 "python -m src.scraper.export_tables", per_year=True),
    ArtifactSpec("C02", "base_tbl", "特徴量", (f"{_F}/base_tbl/{{Y}}/*.parquet",), _lv("required", "recommended"),
                 "python -m src.pipeline.register_raw_table_features", per_year=True),
    ArtifactSpec("C03a", "race_tbl", "特徴量", (f"{_F}/race_tbl/{{Y}}/*.parquet",), _lv("required", "recommended"),
                 "python -m src.pipeline.register_raw_table_features", per_year=True),
    ArtifactSpec("C03b", "race_horse_tbl", "特徴量", (f"{_F}/race_horse_tbl/{{Y}}/*.parquet",),
                 _lv("required", "recommended"), "python -m src.pipeline.register_raw_table_features", per_year=True),
    ArtifactSpec("C03c", "horse_tbl", "特徴量", (f"{_F}/horse_tbl/**/*.parquet",), _lv("required", "recommended"),
                 "python -m src.pipeline.register_raw_table_features"),
    ArtifactSpec("C04a", "race_jockey_tbl", "特徴量", (f"{_F}/race_jockey_tbl/{{Y}}/*.parquet",),
                 _lv("recommended", "optional"), "python -m src.pipeline.build_jockey_trainer_stats", per_year=True),
    ArtifactSpec("C04b", "race_trainer_tbl", "特徴量", (f"{_F}/race_trainer_tbl/{{Y}}/*.parquet",),
                 _lv("recommended", "optional"), "python -m src.pipeline.build_jockey_trainer_stats", per_year=True),
    ArtifactSpec("C04c", "jockey_tbl / trainer_tbl", "特徴量", (f"{_F}/jockey_tbl/**/*.parquet", f"{_F}/trainer_tbl/**/*.parquet"),
                 _lv("recommended", "optional"), "python -m src.pipeline.build_jockey_trainer_stats", min_files=2),
    ArtifactSpec("C05", "target/rank_tbl（着順ラベル）", "特徴量", (f"{_F}/target/rank_tbl/{{Y}}/rank.parquet",),
                 _lv("required", "optional"), "python -m src.pipeline.build_rank_target", per_year=True),
    ArtifactSpec("C06a", "horse/ped_tbl", "特徴量", (f"{_F}/horse/ped_tbl/*/*.parquet",), _lv("required", "recommended"),
                 "python -m src.pipeline.build_horse_entity_store"),
    ArtifactSpec("C06b", "horse/result_tbl", "特徴量", (f"{_F}/horse/result_tbl/*/*.parquet",), _lv("recommended", "optional"),
                 "python -m src.pipeline.build_horse_entity_store"),
    ArtifactSpec("C07", "layer_a_train", "特徴量", ("data/modeling/layer_a_train.parquet",
                 "data/local/modeling/layer_a_train.parquet"), _lv("required", "optional"),
                 "python3 -m src.pipeline.build_layer_a_dataset"),
    ArtifactSpec("D01", "horse_pedigree_5gen（ローカル）", "血統", ("data/local/horse_pedigree_5gen/*/*.json",),
                 _lv("required", "optional"), "python -m src.scraper.horse_pedigree_5gen_bulk mirror-local"),
    ArtifactSpec("D01b", "horse_pedigree_10gen", "血統", ("data/local/horse_pedigree_10gen/*/*.json",),
                 _lv("required", "optional"), "python -m src.research.pedigree.build_horse_pedigree_10gen --skip-existing"),
    ArtifactSpec("D02", "pedigree_10gen index", "血統", ("data/research/pedigree_10gen/meta.json",),
                 _lv("recommended", "optional"), "python -m src.research.pedigree.build_pedigree_10gen_index"),
    ArtifactSpec("D03", "pedigree_race_index", "血統", ("data/research/pedigree_race_index/*",
                 "data/page_reference/pedigree_race_index/*"), _lv("recommended"),
                 "python -m src.research.pedigree.build_pedigree_race_index"),
    ArtifactSpec("D04", "note_aptitude_race", "血統", ("data/page_reference/note_aptitude_race/*",),
                 _lv("recommended"), "python -m src.research.pedigree.note_aptitude_race_map"),
    ArtifactSpec("D05", "bloodline_meta_cluster", "血統", ("data/research/bloodline_meta_cluster/unified.parquet",),
                 _lv("recommended"), "python -m src.research.pedigree.build_meta_cluster_artifacts"),
    ArtifactSpec("D07", "sire_factor_stats / race_quality_priors", "血統",
                 ("data/research/sire_factor_stats.json", "data/research/race_quality_priors.json"),
                 _lv("recommended", "optional"), "src/research の sire_factor_stats.py / tune_race_quality_priors.py", min_files=2),
    ArtifactSpec("A17", "jra_cushion（年別）", "外部データ", ("data/page_reference/cushion/cushion_{Y}.json",),
                 _lv("recommended"), "JRA 馬場: python -m src.scraper.jra_baba_live（または sync_cushion_from_preprocessed）",
                 per_year=True),
    ArtifactSpec("E01", "myostatin_genes.json（手動）", "手動作成",
                 ("data/calculated_data/knowledge/myostatin_genes.json",), _lv("recommended", dev="optional"),
                 "手動作成ナレッジ。他 PC からコピー"),
    ArtifactSpec("E04", "config（settings.yaml / megu_*.json）", "設定",
                 ("config/settings.yaml", "config/megu_par_time_v2_meta.json",
                  "config/megu_predict_condition_weights.json", "config/megu_predict_params.json"),
                 _lv("required", dev="required"), "git から取得", min_files=4),
    ArtifactSpec("F01", "models/（ローカル Booster）", "モデル",
                 ("models/keiba_model.pkl", "models/label_encoders.pkl", "models/final_odds_bundle.json",
                  "models/pace_predictor/model_1f.txt"), _lv("required", dev="optional"), "git から取得（学習PCで再学習→コミット）",
                 min_files=4),
]

# GCS 側に無ければならないもの（stg/prod のみ。list/exists の 1 回で確認）
GCS_OBJECTS: list[dict] = [
    {"id": "F02", "label": "models/ensemble/latest.json（公開済みアンサンブル）", "blob": "models/ensemble/latest.json",
     "level": _lv("recommended", "required"), "hint": "学習PCで学習→publish_model（T-005/T-006 未実施）"},
]


@contextmanager
def level_overrides(env: str, levels: dict[str, str]):
    """``{ID: level}`` を一時的に適用する（カテゴリ名・派生データ ID・``gcs.<ID>``）。終了時に元へ戻す。"""
    saved: list[tuple[dict, str]] = []
    try:
        if levels:
            targets = [(s.name, s.level) for s in (*RACE_CATEGORIES, *HORSE_CATEGORIES)]
            targets += [(a.id, a.level) for a in ARTIFACTS]
            targets += [(f"gcs.{g['id']}", g["level"]) for g in GCS_OBJECTS]
            for key, lv in targets:
                if key in levels:
                    saved.append((lv, lv[env]))
                    lv[env] = levels[key]
        yield
    finally:
        for lv, old in reversed(saved):
            lv[env] = old
