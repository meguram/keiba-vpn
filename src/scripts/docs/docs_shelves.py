"""docs/html の「棚」（日本語のディレクトリの階層）への振り分け。

元の HTML は今のパスのまま動かさない（生成スクリプトが出力先を固定で持ち、担当のセッションが作り直しているため）。
棚のディレクトリには、build_docs_shelves.py が生成する目次・まとめのページだけを置き、元のパスへリンクする。

振り分けの決め方（上から順に効く）:
  1. OVERRIDE に書いた章は、その棚。
  2. 本の PARTS の status が legacy / snapshot / history の章は「09_過去の資料」。
  3. それ以外は、章が載っている本の部（Part）ごとの既定の棚（PART_DEFAULT）。
  data/meta の研究の HTML は、フォルダごとの棚（META_DIR_SHELF）。
章の説明は本の PARTS の blurb を使う（同じ説明を 2 か所に書かない）。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Shelf:
    key: str
    dir: str           # docs/html からの相対のディレクトリ
    title: str
    lead: str


SHELVES: list[Shelf] = [
    Shelf("best", "01_いまのベスト分析", "いまのベスト分析",
          "データ分析家が最初に見る所。重賞・通常レース・血統・走法軸・馬の予測モデル・推奨馬券について、今の最良の結論・数字・根拠・限界を 1 枚ずつに。"),
    Shelf("spec", "02_本番の仕様", "本番の仕様",
          "今、本番（prod）と stg で実際に動いているものの仕様。全体の構成・データ・モデルと規則・画面と公開。"),
    Shelf("res-graded", "03_調査と検証/重賞", "重賞", "重賞のデータ分析型の予想（ルールベース）の調査・サイクルの記録。"),
    Shelf("res-course", "03_調査と検証/通常レース", "通常レース", "全レース版（コース × クラス）の好走データ抽出の調査・サイクルの記録。"),
    Shelf("res-pedigree", "03_調査と検証/血統", "血統", "血統・種牡馬の系統・遺伝子の調査。"),
    Shelf("res-model", "03_調査と検証/モデル", "モデル", "馬の予測モデル（親モデル・アンサンブル・前段）の改善と検証の記録。"),
    Shelf("res-bet", "03_調査と検証/馬券と印", "馬券と印", "推奨馬券・印・期待値・確率の較正の調査と検証。"),
    Shelf("res-feature", "03_調査と検証/特徴量と要因", "特徴量と要因", "コース・展開・調教・騎手・めぐ指数など、要因と特徴量の調査。"),
    Shelf("res-data", "03_調査と検証/データ品質", "データ品質", "データの欠け・誤り・ずれの調査と検査。"),
    Shelf("ops", "04_運用", "運用", "定期実行・取得・配備・公開基盤・費用・監視の手順と現状。"),
    Shelf("old", "09_過去の資料", "過去の資料", "置き換わった設計・ある時点の記録・決定の履歴。今の正は各ページの冒頭か、本の注記にある。"),
]
SHELF = {s.key: s for s in SHELVES}
RESEARCH_DIR = "03_調査と検証"
RESEARCH_TITLE = "調査と検証"
RESEARCH_LEAD = "テーマごとの調査の記録・サイクルのログ。結論だけを見るなら「いまのベスト分析」から。"

# 本の部（Part.title）→ 既定の棚
PART_DEFAULT: dict[str, str] = {
    "要件と全体像": "spec",
    "データ": "spec",
    "取得と運用": "ops",
    "公開基盤と脱ホスト": "ops",
    "モデリングの方針と分割": "res-model",
    "特徴量と血統": "res-feature",
    "学習パイプラインと実験結果": "res-model",
    "オッズと馬券": "res-bet",
    "推論・MLflow・コスト": "spec",
    "本番運用": "spec",
    "決定記録": "old",
}
OLD_STATUS = {"legacy", "snapshot", "history"}

# 既定と違う棚に置く章（slug → 棚）
OVERRIDE: dict[str, str] = {
    # 本番の仕様（モデル・規則・運用の正）
    "model_versions": "spec", "retrain_order": "spec", "upstream_model_outputs": "spec",
    "training_pipeline_decisions": "spec", "race_performance_rating_system": "spec", "megu_index_v3a": "spec",
    "exotic_bet_probabilities": "spec", "final_odds_model": "spec", "ev_calibration_adopt": "spec",
    "feature_serving_low_memory": "spec", "data_terms_and_public_policy": "spec", "local_dataset_endpoints": "spec",
    "persona_goals": "spec", "persona_status": "spec",
    # 運用
    "DATA_HEALTH": "ops", "production_load": "ops",
    # データ品質
    "data_quality_issues": "res-data", "course_master_verification": "res-data", "drift_diagnosis": "res-data",
    "feature_gap_audit": "res-data", "data_trend": "res-data",
    # 重賞
    "graded_handicapper": "res-graded",
    # 血統
    "NOTE_APTITUDE_5GEN": "res-pedigree", "PEDIGREE_NOTE_APTITUDE": "res-pedigree", "PERFORMANCE_GENES": "res-pedigree",
    "sire_lines_all": "res-pedigree", "horse_pedigree_10gen_merge_design": "res-pedigree", "pedigree_position_cross": "res-pedigree",
    "pedigree_roi": "res-pedigree", "pedigree_model": "res-pedigree", "pedigree_easy": "res-pedigree", "pedigree_side_cohit": "res-pedigree", "dam_age_firstfoal": "res-pedigree",
    "weight_growth_performance": "res-pedigree",
    # 馬券と印（学習・実験の部にある馬券の記録）
    "bet_cycles": "res-bet", "latest_strategy_tables": "res-bet", "megu_why_below_100": "res-bet", "megu_cycles_50": "res-bet", "longshot_course_rules": "res-bet", "rules_council_summary": "res-bet", "rules_council_favorites": "res-bet", "rules_council_t60_band": "res-bet", "rules_council_daybias": "res-bet", "rules_council_debut": "res-bet", "bet_patterns_from_honmei": "res-bet", "bet_optimization_story": "res-bet", "bet_r2_walkthrough": "res-bet", "rules_bet_opt": "res-bet", "odds_rank_accuracy": "res-bet",
    # 特徴量と要因（学習・実験の部や馬券の部にある要因の調査）
    "running_map_course_axis": "res-feature", "running_map_course_axis_rationale": "res-feature", "running_axis_curves": "res-feature", "running_style_pairs": "res-feature",
    "feature_investigation_catalog": "res-feature",
    "pace_consistency": "res-feature", "lap_forecast": "res-feature", "tracking_difficulty_v2": "res-feature",
    "handicapper_viewpoints": "res-feature", "data_acquisition": "res-feature", "feature_cross_search": "res-feature",
    "target_encoding_inventory": "res-feature", "jockey_trainer_roi": "res-feature", "jockey_switch_ev": "res-feature",
    "similar_pairs_multi": "res-feature", "position_pace_course": "res-feature", "jockey_distance_dependence": "res-feature", "jockey_weight_affinity": "res-feature",
    "folklore_hypotheses": "res-feature", "folklore_verification": "res-feature", "feature_screening": "res-feature",
    "pace_type": "res-feature", "pace_benefit": "res-feature", "index_variance": "res-feature",
    "megu_excusable": "res-feature", "megu_marking": "res-feature", "revival_feature_set": "res-feature",
    # モデル（馬券の部にあるモデルの記録）
    "ensemble_v5": "res-model", "ensemble_structure": "res-model", "ensemble_diversity": "res-model",
    "deployment_readiness": "res-model", "v6_features": "res-model", "persona_ensemble_survey": "res-model",
    "final_layer_philosophies": "res-model", "prob_calibration_gap": "res-bet",
    # 古い（status は付いていないが置き換わったもの）
    "tp05a_experiment_report": "old",
}

# data/meta の研究の HTML（フォルダ → 棚）。新しい HTML が足されても自動で入る
META_DIR_SHELF: dict[str, str] = {
    "graded": "res-graded",
    "course": "res-course",
    "sire_lines": "res-pedigree",
    "dormello": "res-pedigree",
}
UNKNOWN_SHELF = "res-data"   # 既定の棚が決まっていない部・フォルダの置き場（警告を出す。テストでは落とす）


@dataclass
class Extra:
    """本に載っていない文書（md など）を、棚のまとめ（群）から参照するための項目。"""
    path: str          # リポジトリのルートからの相対パス
    title: str
    blurb: str = ""


@dataclass
class Group:
    """棚の中の「まとめ」（同じテーマの文書の束）。"""
    title: str
    lead: str
    slugs: list[str] = field(default_factory=list)       # 本の章（どの棚の章でもよい。棚の外の章は「関連」として出る）
    extras: list[Extra] = field(default_factory=list)


def E(path: str, title: str, blurb: str = "") -> Extra:
    return Extra(path, title, blurb)


# 分散している同じテーマの文書を、棚ごとに「まとめ」にする。結論・今の版・経緯（更新日の年表）・詳細のリンクの形で出す。
# 群に入らなかった棚の章は「その他」に入る。
GROUPS: dict[str, list[Group]] = {
    "spec": [
        Group("全体の構成", "今の構成の正は最終アーキテクチャ。要件定義書は /update-requirements で作り直す。",
              ["requirements", "final_architecture", "FULLSTACK_ARCHITECTURE", "LAYER_DESIGN", "server_architecture", "inference_services", "mlflow_platform",
               "persona_goals", "persona_status"]),
        Group("データ", "取得元・保存先・列の仕様。データの流れは data_flow_schedule。",
              ["DATASET_CATALOG", "DATASOURCES", "data_scraping_storage_reference", "database_schemas", "data_features_reference", "data_flow_schedule",
               "local_dataset_endpoints", "data_terms_and_public_policy"]),
        Group("モデルと規則（今、本番で動く版）", "どの版が本番かの正は model_versions（モデルの版と採用状況）。確定事項の正は training_pipeline_decisions（決定ログ）。",
              ["model_versions", "retrain_order", "upstream_model_outputs", "training_pipeline_decisions", "race_performance_rating_system", "megu_index_v3a",
               "final_odds_model", "exotic_bet_probabilities", "ev_calibration_adopt", "feature_serving_low_memory"]),
    ],
    "ops": [
        Group("定期実行と取得", "cron（ホスト）と GCP・VPS の定期実行。登録の正は scheduled_jobs と cron_schedule（どちらも生成）。",
              ["cron_schedule", "scheduled_jobs", "raceday_prediction_workflow", "scraper_architecture", "scraping_sla", "data_flow_schedule",
               "prediction_results", "prediction_backfill", "tracking_v2_precompute"],
              [E("docs/operations/cron_catchup.md", "cron の取りこぼしの追いかけ"), E("docs/operations/fetch_tasks.md", "取得のタスク")]),
        Group("公開基盤・配備", "画面=VPS、API=Cloud Run、Jobs、取得役。到達点は decoupling_status。",
              ["decoupling_status", "decoupling_plan", "architecture_target", "cloud_run_api", "gcp_jobs_migration", "vps_agent", "monitor_cloudrun",
               "publisher_sa", "github_promote", "env_parity", "stg_frontend_prod_build", "megu_v3a_rollout"]),
        Group("費用", "費用の上限と単価の正は config/cost_assumptions.json。台帳と変更の記録は md。cost（GCS のコスト分析）は時点の記録。",
              ["cost"],
              [E("docs/operations/cost_ledger.md", "費用の台帳"), E("docs/operations/cost-estimate.md", "費用の見積もり"),
               E("docs/operations/cost_changes_20261113.md", "費用に効く変更（2026-11-13）"), E("docs/operations/cost_recalc_20261028.md", "費用の再計算（2026-10-28）"),
               E("docs/operations/billing.md", "課金エクスポートの見方"), E("config/cost_assumptions.json", "費用の前提（上限・単価）")]),
        Group("監視・負荷", "監視ポータル・ドリフトの監視・負荷の限界と対応。",
              ["production_load", "vps_load_and_cache", "raceday_audit_and_load_test_20261123", "monkey_test_results", "DATA_HEALTH"],
              [E("docs/monitoring-portal.md", "監視ポータルの仕様"), E("docs/operations/drift_monitor.md", "ドリフトの監視"),
               E("docs/operations/diagnostics.md", "診断の手順")]),
    ],
    "res-model": [
        Group("親モデルの版と比較", "今の本番の版と採用時の精度は「いまのベスト分析 / 馬の予測モデル」。改善サイクルの正は model_cycles（生成）。",
              ["model_cycles", "model_versions", "retrain_order", "model_accuracy_summary", "deployment_readiness", "v6_features",
               "ensemble_v5", "ensemble_structure", "ensemble_diversity"],
              [E("docs/operations/retrain_trials/README.md", "再学習の試しの記録（README）"),
               E("docs/operations/retrain_trials/model_cycles_20261116.md", "親モデルの改善サイクル（2026-11-16〜）"),
               E("docs/operations/retrain_trials/full2026_prod_vs_cand_20261116.md", "2026 年の本番と候補の比較"),
               E("docs/operations/retrain_trials/v6e_2026_gap_20261114.md", "v6e の 2026 年の落ち込み"),
               E("docs/operations/retrain_trials/model_condition_strength_weakness_20261114.md", "条件ごとの得手不得手"),
               E("docs/operations/retrain_trials/rank_label_bakeoff_20261113.md", "順位ラベルの比べ"),
               E("docs/operations/pred_model_v6e.md", "v6e の本番化の手順")]),
    ],
    "res-bet": [
        Group("推奨馬券の最適化", "結論は「いまのベスト分析 / 推奨馬券の最適化」。サイクルの正は bet_cycles（生成）。",
              ["latest_strategy_tables", "megu_why_below_100", "megu_cycles_50", "longshot_course_rules", "rules_council_summary", "rules_council_favorites", "rules_council_t60_band", "rules_council_daybias", "rules_council_debut", "bet_patterns_from_honmei", "bet_cycles", "bet_optimization_story", "bet_r2_walkthrough", "rules_bet_opt", "odds_rank_accuracy", "bet_optimization", "bet_allocation_table", "betting_edge_search", "monthly_roi_search",
               "trifecta_true_odds", "proven_methods_roi"],
              [E("docs/operations/retrain_trials/bet_optimization_20261116.md", "推奨馬券の最適化の記録（2026-11-16〜）")]),
        Group("印と期待値", "印の規則・消し印・期待値と確率の較正。",
              ["race_type_marks", "mark_rules", "kesi_mark", "ev_calibration_audit", "ev_calibration_adopt", "prob_calibration_gap", "edge_factor_mining",
               "rule_personas", "yosou_procedure"],
              [E("docs/operations/mark_rule_v2.md", "印の規則 v2"), E("docs/operations/prob_calibration.md", "確率の較正の運用")]),
    ],
    "res-data": [
        Group("データと文書の食い違い", "データ品質の既知の問題と、文書どうし・文書とコードの食い違いの一覧。",
              ["data_quality_issues", "feature_gap_audit", "drift_diagnosis", "course_master_verification"],
              [E("docs/operations/docs_inconsistencies_20261118.md", "文書の食い違い・古い記述の一覧（2026-11-18）"),
               E("docs/operations/data_audit_20261116.md", "データの監査（2026-11-16）")]),
        Group("データの傾向の変化", "2016〜2025 年の傾向の変化（ドリフト）の大きさと、ルール・モデル・馬券への影響、問題点と改善案。",
              ["data_trend"],
              [E("docs/operations/data_trend_20261118.md", "データの傾向の変化の調査（2026-11-18）")]),
    ],
    "res-pedigree": [
        Group("血統のまとめ", "結論は「いまのベスト分析 / 血統分析のまとめ」。",
              ["pedigree_easy", "pedigree_model", "pedigree_roi", "pedigree_side_cohit", "pedigree_position_cross", "dam_age_firstfoal", "weight_growth_performance", "sire_lines_all",
               "PERFORMANCE_GENES", "PEDIGREE_NOTE_APTITUDE", "NOTE_APTITUDE_5GEN", "horse_pedigree_10gen_merge_design",
               "track_bias_pedigree_pipeline", "pedigree_stage_research_notes"],
              [E("docs/ideas/sire_dam_influence.md", "父と母の影響"), E("docs/ideas/euro_kick_sire_lines.md", "欧州型の瞬発力の系統")]),
    ],
    "res-feature": [
        Group("走法マップ", "走法の軸 × コースの適性・走法や脚質が近い馬が一緒に来るか。結論は「いまのベスト分析 / 走法軸のまとめ」。", ["running_map_course_axis", "running_map_course_axis_rationale", "running_axis_curves", "running_style_pairs"],
              [E("docs/ideas/running_map_axes_fixed.md", "走法マップの軸（固定）"), E("docs/ideas/running_axes_scale_1to10.md", "走法の軸の 1〜10 の尺度"),
               E("docs/ideas/running_style_pairs.md", "走法の組")]),
        Group("めぐ指数", "本番の版は megu_index_v3a（本番の仕様の棚）。",
              ["megu_index_v3a", "megu_excusable", "megu_marking", "index_variance", "class_benchmark", "race_performance_rating_system"]),
        Group("コースと展開", "コースの形・成績の連動・展開と位置取り。",
              ["course_geometry_atlas", "course_performance_similarity", "course_aptitude_groups", "course_specific_effects", "position_pace_course",
               "pace_type", "pace_benefit", "pace_consistency", "lap_forecast", "tracking_difficulty_v2", "race_volatility"]),
        Group("調教・状態・騎手", "追い切り・休養・輸送・騎手と調教師。",
              ["final_week_training_pattern", "training_condition_growth", "workout_remaining_ideas", "lap_groups_and_trainer_stats", "rest_and_interval",
               "transport_and_stay", "equipment_and_change", "geld_after_races", "growth_curve_weight", "jockey_trainer_roi", "jockey_switch_ev",
               "jockey_distance_dependence", "jockey_weight_affinity", "trainer_comment_wordcloud", "slow_start_real_label"]),
    ],
}
