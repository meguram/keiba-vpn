"""tp-75: ルール台帳に取り込む出典の棚卸し（章・ホスト出力・スクリプト・設定ごとに『ルール化できる統計があるか』と、取り込んだ／取り込まなかった理由）。

ingest の値:
  recomputed  共通の表（tp72 の mining.parquet）の上で条件を定義し直し、同じ検定でルール化した（family に台帳の族を書く）
  excluded_market  市場情報（オッズ・人気・市場確率・想定オッズ・期待値）に依存するので取り込まなかった
  baseline    ルールではなく、比較用の基準（モデルの順位・印）として使った
  none        取り込まなかった（理由を reason に書く）
"""

from __future__ import annotations

CH = "docs/html/modeling/chapters/"

# (出典, 区分, 取り込み, 族/用途, ルール化できる統計, 理由)
INVENTORY: list[dict] = []


def _add(src: str, kind: str, ingest: str, family: str, rule_stats: str, reason: str) -> None:
    INVENTORY.append({"source": src, "kind": kind, "ingest": ingest, "family": family, "rule_stats": rule_stats, "reason": reason})


def _ch(name: str, ingest: str, family: str, rule_stats: str, reason: str) -> None:
    _add(CH + name + ".html", "chapter", ingest, family, rule_stats, reason)


# ---- 章（docs/html/modeling/chapters/ と docs/html/modeling/）--------------------------------------------------
_ch("edge_factor_mining", "recomputed", "T72（F1〜F23, 881 型）", "あり", "買う要素の網羅探索そのもの。事前登録の 881 型を、この台帳の基準（95%/90%、両方向、再現、BH）で検定し直した")
_ch("geld_after_races", "recomputed", "G（去勢明け N 戦目）", "あり", "牡→セ の最初の出走から数えた N 戦目を、共通の表の性別の履歴から定義し直した")
_ch("dam_age_firstfoal", "recomputed", "D（母の出産時年齢・初仔・種牡馬の供用年）", "あり", "共有表 foal_lineage を horse_id で結合して定義し直した")
_ch("jockey_switch_ev", "recomputed", "J（乗り替わりの型）", "あり", "jchg_cls（継続/非有力→有力/有力→非有力/有力→有力/非有力→非有力）を使い、前走着順・距離で層別")
_ch("equipment_and_change", "recomputed", "E（馬具・距離替わり・面替わり・斤量・調教師替わり）", "あり", "blk_state、dist_chg、first_surface、jw_chg、chg_trainer で定義")
_ch("rest_and_interval", "recomputed", "R（休養日数の帯、前走着順との組合せ、馬体重との組合せ）", "あり", "lay_days の帯で定義")
_ch("kesi_mark", "excluded_market", "", "あり", "消印の条件は人気（市場由来）と複勝確率の組合せ。大衆情報を条件に使わない方針により取り込まない")
_ch("proven_methods_roi", "excluded_market", "", "あり", "人気帯・オッズ帯・『函館の 3 番人気』は市場由来の条件。取り込まない")
_ch("betting_edge_search", "excluded_market", "", "あり", "モデルの期待値（確率×オッズ）と混合確率（市場を混ぜる Benter 型）は市場由来。取り込まない")
_ch("ev_calibration_adopt", "excluded_market", "", "あり", "期待値の較正は市場オッズに依存。取り込まない")
_ch("ev_calibration_audit", "excluded_market", "", "あり", "期待値の水準の監査は市場オッズに依存。取り込まない")
_ch("mark_rules", "excluded_market", "", "あり", "印は勝率順位（モデル）と、単穴・期待値の印（オッズ由来）の規則。印の規則は取り込まない（モデル順位は比較の基準線にだけ使う）")
_ch("race_type_marks", "excluded_market", "", "なし", "印の切り替えはレース単位で、期待値（オッズ）に依存する印を含む。取り込まない")
_ch("monthly_roi_search", "recomputed", "T72（F1, F2 月・季節）", "一部", "月・季節の条件は T72 の性別×月・季節で取り込み済み。月次過半の戦略自体は買い方であり条件ではない")
_ch("jockey_trainer_roi", "recomputed", "T72（F21〜F23 種牡馬×騎手）", "一部", "種牡馬・母父×騎手は T72 で取り込み済み。凍結規則（EV≥1.5 かつ騎手・調教師の残差が正）は単勝オッズ由来の期待値を条件に含むため取り込まない")
_ch("pedigree_roi", "recomputed", "T72（F5, F21〜F23）", "一部", "血統×条件は T72 の種牡馬・母父の上位 × 騎手/条件で取り込み済み。198 規則の個別の統計は同じ検定の対象外（別表）")
_ch("pedigree_side_cohit", "none", "", "なし", "クラスタ対の同時好走で馬ごとの条件ではない（ペア単位）。補正後 0 本")
_ch("pedigree_position_cross", "none", "", "なし", "特徴量群の評価で、条件ごとの回収率・残差の表ではない")
_ch("position_pace_course", "recomputed", "B9（脚質×面、前走の上がり順位）", "一部", "事前に使える形の 519 セルは補正通過 0。脚質×面と前走の上がり順位で近い条件を定義した")
_ch("pace_benefit", "none", "", "一部", "展開利のラベルは共通の表に無い（tp64 の表が別）。残差との関連は弱く、ルール化の根拠が薄い。次の版で結合を検討")
_ch("pace_type", "none", "", "一部", "展開タイプの確率は別の表。レース単位の性質でもある")
_ch("pace_consistency", "none", "", "なし", "予想同士の連動の検査で、条件別の成績ではない")
_ch("tracking_difficulty_v2", "none", "", "なし", "特徴量の設計・評価章")
_ch("lap_forecast", "none", "", "なし", "予想モデルの章")
_ch("lap_groups_and_trainer_stats", "none", "", "一部", "調教師別・コース別の調教成績は別表。調教師の as-of 成績の表が共通の表に無い")
_ch("course_specific_effects", "recomputed", "C（競馬場×面、距離帯×面、クラス×面）", "一部", "コース特注の要因は、競馬場×面などの粗い条件で取り込んだ。細かいコース別の統計は標本が薄い")
_ch("course_aptitude_groups", "none", "", "一部", "グループ割り当て（成果物）が共通の表に無い")
_ch("course_performance_similarity", "none", "", "なし", "コースの対応づけで馬の条件ではない")
_ch("course_geometry_atlas", "none", "", "なし", "コース概形のマスタ")
_ch("course_master_verification", "none", "", "なし", "マスタの検証記録")
_ch("transport_and_stay", "none", "", "あり", "東西の遠征には、調教師の所属（美浦/栗東）の対応表が要る。共通の表に無い。次の版で結合を検討")
_ch("growth_curve_weight", "recomputed", "T72（F7 馬体重）", "一部", "馬体重の増減・自分の平均との差は T72 の F7 で取り込み済み。成長曲線の補正ポイントは特徴量で、条件ではない")
_ch("weight_growth_performance", "recomputed", "T72（F7 馬体重）", "一部", "同上。体重の増減は『安定が最良』で、条件化は T72 の dw/dev で行った")
_ch("training_condition_growth", "none", "", "一部", "追い切りの時計の出方は別表（tl_）。共通の表に無い")
_ch("workout_remaining_ideas", "none", "", "一部", "調教師ごとに標準化した追い切りは別表")
_ch("slow_start_real_label", "none", "", "一部", "出遅れ率は別表のラベル。次の版で結合を検討")
_ch("megu_marking", "recomputed", "T72（F14 前走の指数 ev_score/ev_delta）", "一部", "めぐ指数の印・過去走の集約は、共通の表の ev_score/ev_delta で近い条件を定義")
_ch("megu_excusable", "none", "", "一部", "無視してよい低下の判別は別表の出力")
_ch("megu_index_v3a", "none", "", "なし", "指数の定義章")
_ch("class_benchmark", "none", "", "一部", "クラス基準（突破ライン）は別表")
_ch("index_variance", "none", "", "なし", "指数のばらつきの扱い")
_ch("race_volatility", "none", "", "なし", "レース単位の荒れ度")
_ch("handicapper_viewpoints", "none", "", "なし", "予想家の観点の調査で、統計表ではない。観点の分類はペルソナの設計に使った")
_ch("yosou_procedure", "excluded_market", "", "一部", "『穴馬』（人気薄）を条件にする手順で、人気に依存。取り込まない")
_ch("similar_pairs_multi", "none", "", "なし", "複数軸が同時に近いペアの分析で、ペア単位（馬ごとの条件ではない）")
_ch("similar_horses_pairs", "none", "", "なし", "ペア単位の分析")
_ch("feature_cross_search", "none", "", "なし", "特徴量の交互作用の探索で、採用なし")
_ch("feature_gap_audit", "none", "", "なし", "棚卸し（索引）。取りこぼしの確認に使った")
_ch("feature_investigation_catalog", "none", "", "なし", "調査の総覧（索引）。取りこぼしの確認に使った")
_ch("prob_calibration_gap", "excluded_market", "", "あり", "確率の市場補正・人気区分別のずれは市場由来。取り込まない")
_ch("v6_features", "none", "", "なし", "モデルの特徴量の評価")
_ch("ensemble_v5", "none", "", "なし", "アンサンブルの構成")
_ch("ensemble_structure", "none", "", "なし", "アンサンブルの構造")
_ch("ensemble_diversity", "none", "", "なし", "アンサンブルの幅出し")
_ch("persona_ensemble_survey", "none", "", "なし", "ペルソナの調査。この台帳は 6 つ目のペルソナの実体")
_ch("bet_optimization", "excluded_market", "", "なし", "推奨ポートフォリオは単勝・複勝オッズから期待値を作る。取り込まない")
_ch("betting_policy_reinforcement_learning", "none", "", "なし", "設計議論")
_ch("betting_backtest_dataset_spec", "none", "", "なし", "データ仕様")
_ch("exotic_bet_probabilities", "none", "", "なし", "組合せ券の確率")
_ch("final_odds_model", "none", "", "なし", "想定オッズのモデル")
_ch("trifecta_true_odds", "none", "", "なし", "3 連単の評価")
_ch("deployment_readiness", "none", "", "なし", "導入可否")
_ch("drift_diagnosis", "none", "", "なし", "ドリフトの診断")
_ch("model_accuracy_summary", "none", "", "なし", "精度の総括")
_ch("training_pipeline_decisions", "baseline", "確定事項ログ（全章の索引・採否の根拠）", "あり", "各節の採否を台帳の突き合わせに使った（条件の統計そのものは各章から取り込んだ）")
_ch("research_overview", "none", "", "なし", "全体像")
_ch("reference_insights", "none", "", "なし", "参考資料のメモ")
for _n in ("adopted_features_v3", "advanced_modeling_techniques", "data_acquisition", "data_flow_schedule", "feature_engineering_exploration",
           "feature_serving_low_memory", "feature_strategy_tp03_tp04", "horse_pedigree_10gen_merge_design", "horse_pre_race_dataset_spec",
           "hyperparameter_tuning_workflow", "modeling_coding_strategy", "modeling_experiment_report", "modeling_strategy_tp04_tp07",
           "pedigree_stage_research_notes", "pre_race_feature_whitelist", "race_performance_rating_system", "staged_probability_pipeline",
           "tp05a_experiment_report", "track_bias_pedigree_pipeline", "train_valid_test_split_strategy"):
    _ch(_n, "none", "", "なし", "設計・仕様・実験メモの章で、条件別の統計表を持たない")
_ch("jockey_distance_dependence", "none", "", "あり", "検証の事前登録のみで、結果が未確定（別の担当が実行中）。次の版で取り込む")

# ---- ホストの保存結果 --------------------------------------------------------------------------------------------
for _o, _i, _r in (
    ("tp72_out/mining.parquet, stage1.json", "recomputed", "T72（共通の表そのもの）"),
    ("geld_out（geld/）", "recomputed", "G"),
    ("dam_out（dam_age/）", "recomputed", "D"),
    ("jsw_out", "recomputed", "J"),
    ("tp59_out（census/ev_final/ev_robust）", "excluded_market", "EV（単勝オッズ）を条件に含む規則は取り込まない"),
    ("tp56_out（features/bet）", "none", "買い方の評価で、馬ごとの条件の統計表ではない"),
    ("tp62_out（overlay）", "none", "月次戦略の探索で、条件ではない"),
    ("tp74pos_out / tp74_out", "none", "位置×コース×ペース: 補正通過 0 のため台帳に入る条件が無い（B9 で近い条件を定義）"),
    ("tp73_out", "none", "クラスタ対・父母系の分析: 補正通過 0 / 馬ごとの条件ではない"),
    ("tp38_out / tp60_out", "none", "血統の特徴量群・埋め込みの評価で、条件別の統計表ではない"),
    ("tp51_out / tp63_out", "none", "アンサンブル"),
    ("tp64_out / tp65_out / tp66_out / tp68_out / tp69_out", "none", "展開利・指数のばらつき・指数の印・excusable: 別表で共通の表に無い（次の版で結合を検討）"),
    ("tp71_out", "none", "想定オッズモデルの評価"),
    ("tp54_out / tp55_out*", "none", "追走難度・ラップ予想の出力で、条件別の成績ではない"),
    ("tp48_2026_out / drift_out / persona_out / readiness_out / jdd_out", "none", "別の目的の出力（3 連単・ドリフト・ペルソナ・導入可否・騎手距離）"),
):
    _add("host:/home/jovyan/work/" + _o, "host_out", _i, _r if _i == "recomputed" else "", "あり" if _i in ("recomputed", "excluded_market") else "なし", _r if _i in ("none", "excluded_market") else "共通の表に結合して再計算した")

# ---- その他のリポジトリ内の分析 -------------------------------------------------------------------------------------
_add("src/research/pedigree/bloodline_roi.py, backtest_*.py, *_evidence.py", "script", "none", "", "一部",
     "血統の回収率・適性の旧バックテストで、評価設計が現行（OOF 残差・BH）と異なる。pedigree_roi（tp-60）が同じ問いを現行の設計で再検定している")
_add("src/research/race/*（race_quality_model, tune_*, evaluate_*）", "script", "none", "", "なし", "レース質・馬場傾向のモデル化で、馬ごとの条件の統計表ではない")
_add("data/analysis/pedigree/（中間 Parquet）", "data", "none", "", "一部", "バックテストの中間成果物。条件別の統計は pedigree_roi に集約済み")
_add("docs/requirements/data/（スキーマ・取得プロセス）", "doc", "none", "", "なし", "データ要件の文書で、分析結果ではない")
_add("config/tp59_frozen_rule.json", "config", "excluded_market", "", "あり", "凍結規則（単勝 EV≥1.5 かつ騎手・調教師の残差が正）は単勝オッズ由来の期待値を条件に含む。市場に依存する規則は取り込まない")
_add("config/tp72_frozen_rule.json", "config", "recomputed", "T72（F3 夏の牝馬）", "あり", "夏の牝馬は T72 の F3 に含まれる")
_add("config/ev_calibration.json, ev_calibration_pc.json, prob_calibration.json, paid_zone.json（B/C）", "config", "excluded_market", "", "なし", "期待値・市場補正・想定オッズに依存。取り込まない（paid_zone の A＝確信度はレース単位で、馬の条件ではないため台帳に入れない）")
