"""復活セットの台帳: 過去の採用検討（revival_scan の結果）から、採用基準を緩めた各段階で復活する群を分類する。

入力: docs/operations/revival/revival_scan.json（revival_scan がホストで作る。学習なし）、group_columns.json（群の列数）。
出力: docs/operations/revival_feature_ledger.csv（群ごと）、config/feature_sets.json（基準・復活セットの定義）。
基準の段階（勝率の LogLoss の差。負が改善）:
  S1 従来: 片側 95% 上限 < 0 かつ 効果 ≤ −0.0002（連対・複勝の悪化は +0.0002 以内）
  S2: 2 期間以上で同符号（年別が取れれば年、取れなければ段階）、かつ点推定 ≤ −0.0002（区間は問わない）
  S3: 2 期間以上で同符号、かつ点推定 ≤ −0.0001
  S4: 懐疑的な事前分布 N(0, τ²)（τ=0.0002）で縮約した事後で P(効果 < 0) ≥ 0.8
  S5: 片側 90% の上限 < 0
いずれも連対・複勝の悪化が +0.0002 以内であること（place_ok）を条件にする。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCAN = ROOT / "docs" / "operations" / "revival" / "revival_scan.json"
TP76_RESULTS = ROOT / "docs" / "operations" / "revival" / "tp76_results.json"
COLUMNS = ROOT / "docs" / "operations" / "revival" / "group_columns.json"
LEDGER_CSV = ROOT / "docs" / "operations" / "revival_feature_ledger.csv"
FEATURE_SETS = ROOT / "config" / "feature_sets.json"

Z95, Z90 = 1.645, 1.2816
MIN_EFFECT_S1 = 0.0002
PLACE_TOL = 0.0002
TAU_PRIOR = 0.0002
OOF_SE_INFLATION = 1.4   # OOF の再集計の標準誤差は、ログの paired bootstrap（seed 間の揺れを含む）より小さいので割り増す

# 検討（tp 番号）ごとの情報: (主題, 構築コード, データの置き場)
THEMES: dict[str, tuple[str, str, str]] = {
    "tp15": ("コース概形だけの類似度", "src/pipeline/features/course_variant.py", "（保存なし）"),
    "tp16": ("季節性・後日談・クッション値・近交係数ほか", "notebooks（tp-16 / tp-16b）", "（保存なし。ノートの出力）"),
    "tp16b": ("後日談の変種・訂正後のクッション値と含水率", "notebooks（tp-16b）", "（保存なし。含水率の補完後は再評価が要る）"),
    "tp21": ("SmartRC のテン1F を直接特徴量に", "src/pipeline/features/smartrc_ten1f.py", "（保存なし）"),
    "tp25": ("位置取り・ペース前段の別形", "src/pipeline/features/tp25a_start.py / tp25b_pace.py", "（保存なし）"),
    "tp26": ("好走傾向が似たコースの対応づけ", "src/pipeline/features/tp26_course_transfer.py", "（保存なし）"),
    "tp28": ("追い切りのパターン別の統計", "src/pipeline/features/tp28_workout_stats.py", "（保存なし）"),
    "tp29": ("コース特注適性（期待外れ・期待以上）", "src/pipeline/features/tp29_course_aptitude.py", "（保存なし）"),
    "tp30": ("荒れ度", "src/pipeline/features/tp30_volatility.py", "（保存なし）"),
    "tp31": ("コース×属性の as-of lift", "src/pipeline/features/tp31_course_specific.py", "（保存なし）"),
    "g3": ("G3 のレース・メンバー構成", "src/pipeline/features/factory_field.py", "（V4 の field7 に採用分が入っている）"),
    "g6b": ("出遅れ率の実ラベル", "src/pipeline/features/slowstart_real.py", "（保存なし）"),
    "tp32": ("馬具・初経験・替わり系（FIRST・COMBO ほか）", "src/pipeline/features/tp32_equipment.py", "training_pipeline_v4/tp32/groups/"),
    "tp33": ("輸送・滞在", "src/pipeline/features/tp33_transport.py", "training_pipeline_v4/tp33/groups/"),
    "tp34": ("休養・間隔", "src/pipeline/features/tp34_rest.py", "training_pipeline_v4/tp34/groups/"),
    "tp35": ("ラップ分類・坂路時計", "src/pipeline/features/tp35_lap_groups.py", "training_pipeline_v4/tp35/groups/"),
    "tp37": ("追い切りの残りの余地", "src/pipeline/features/tp37_workout.py", "training_pipeline_v4/tp37/groups/"),
    "tp38": ("血統の位置・クロス・類似種牡馬", "src/pipeline/features/tp38_pedigree.py", "/home/jovyan/work/tp38_out/groups/"),
    "tp39": ("成長曲線（馬体重×パフォーマンス）", "src/pipeline/features/tp39_growth.py", "training_pipeline_v4/tp39/groups/"),
    "tp40": ("調教の時計の出方（サイクル内の推移）", "src/pipeline/features/tp40_condition.py", "training_pipeline_v4/tp40/groups/"),
    "tp41": ("似た馬の着順（SIM）", "src/pipeline/features/tp41_similar.py", "training_pipeline_v4/tp41/groups/"),
    "tp42": ("めぐ指数のレースレベル", "src/pipeline/features/tp42_racelevel.py", "training_pipeline_v4/tp42/groups/"),
    "tp56": ("ユーザーの予想手順（レース・馬タイプほか）", "src/pipeline/features/tp56_yosou.py", "/home/jovyan/work/tp56_out/features.parquet"),
    "tp57": ("成長指数", "src/scripts/data/（growth_index_*）", "training_pipeline_v4/tp57_growth/groups/"),
    "tp59": ("関係者の残差（rs59）", "src/pipeline/features/tp59_uplift.py", "（V5 の OOF 残差。毎日の更新が要る）"),
    "tp60": ("血統×レース質・埋め込み・近さ", "src/scripts/data/tp60_*.py", "/home/jovyan/work/tp60_out/groups/"),
    "tp64": ("展開利（PB4）", "src/pipeline/features/tp64_pace_benefit.py", "/home/jovyan/work/tp64_out/pb_features.parquet"),
    "acq": ("取得済みの未使用データ（非 JRA 過去走・きょうだい）", "src/pipeline/features/acq_groups.py", "/home/jovyan/work/acq/groups/"),
    "v6": ("v6 への追加候補（クラス基準・展開タイプ・馬主名ほか）", "src/pipeline/features/v6_groups.py / v6_ext.py", "training_pipeline_v4/v6/groups/"),
    "v5sel": ("v5 の構成選択", "src/scripts/data/（v5_select）", "training_pipeline_v4/v5_select/"),
}

# すでに採用（比較の基準側に入っている）群: (theme, group) の接頭辞一致
ADOPTED_PREFIX = [
    ("tp40", "CY"), ("tp40", "CY"),
    ("tp38", "ALL_BEST"), ("tp38", "PX_cell_sd"), ("tp38", "PX_emb"), ("tp38", "PX_ridge_als"),
    ("v5sel", "b_tlfresh"), ("v5sel", "c_CY"), ("v5sel", "d_CY_BEST"),
    ("v6", "x_RB"), ("v6", "z:RB@b"), ("v6", "z:RB@e"), ("v6", "z:RB+-RIDER+-YF@b"), ("v6", "z:RB"), ("v6", "m_S3B"),
]
ADOPTED_EXACT = {("tp40", "CY"), ("tp38", "ALL_BEST"), ("tp38", "PX_cell_sd"), ("tp38", "PX_emb"), ("tp38", "PX_ridge_als"),
                 ("v5sel", "b_tlfresh"), ("v5sel", "c_CY"), ("v5sel", "d_CY_BEST"), ("v6", "x_RB"), ("v6", "z:RB@b"), ("v6", "z:RB@e"),
                 ("v6", "z:RB+-RIDER+-YF@b"), ("v6", "z:RB"), ("v6", "m_S3B")}

# 冗長なクラスタ（同じ情報を複数の群が持つ）
CLUSTERS: list[tuple[str, list[tuple[str, str]]]] = [
    ("指数履歴", [("tp42", ""), ("v6", "s_CB"), ("v6", "z:RB"), ("v6", "x_RB"), ("tp39", "GR_mga"), ("tp39", "GR_meg"), ("tp57", ""), ("tp64", "")]),
    ("追い切り", [("tp40", ""), ("tp37", ""), ("tp35", ""), ("tp28", "")]),
    ("血統", [("tp38", ""), ("tp60", "")]),
    ("成長", [("tp39", "")]),
    ("初経験・替わり", [("tp32", ""), ("tp34", ""), ("tp33", "")]),
    ("位置取り・ペース", [("v6", "h_"), ("v6", "i_"), ("v6", "j_"), ("v6", "k_"), ("v6", "l_"), ("v6", "m_"), ("v6", "n_"), ("v6", "p_"), ("v6", "q_"), ("v6", "r_"), ("tp25", "")]),
    ("似た馬・コース類似", [("tp41", ""), ("tp15", ""), ("tp26", ""), ("tp29", ""), ("tp31", "")]),
    ("取得済みの未使用データ", [("acq", ""), ("v6", "x_NJ")]),
    ("関係者・馬主", [("v6", "z:OWNN"), ("v6", "z:RB+CB+NJ+RS59"), ("tp59", "")]),
]

# 推論側（本番の保存値から同じ列を作る経路）がある群。無いものは『要整備』
SERVING_OK_EXACT = {("tp40", "CY"), ("tp38", "ALL_BEST"), ("tp38", "PX_cell_sd"), ("tp38", "PX_emb"), ("tp38", "PX_ridge_als"),
                    ("v6", "x_RB"), ("v6", "z:RB"), ("v6", "m_S3B")}
SERVING_EFFORT: dict[tuple[str, str], tuple[str, str]] = {
    # (工数の目安 S/M/L, 理由)
    ("tp42", ""): ("M", "RB と同じ指数の履歴状態を使う。as-of のレースレベルの状態表を追加"),
    ("tp39", ""): ("M", "馬ごとの体重×成績の履歴状態を追加（馬体重は T-60 で発表）"),
    ("tp32", ""): ("M", "as-of の構築関数 tp32_equipment は serving_v4 が読み込み済み。FIRST 用の状態表を追加"),
    ("tp41", ""): ("L", "同じレースの全ペアの類似度が要る。出走表確定後に計算する経路を新設"),
    ("tp38", ""): ("M", "PX の祖先表・SVD は serving_v5_px にある。変種は同じ表から作れる"),
    ("tp60", ""): ("L", "構成埋め込みと共有表（race_quality）の更新運用が要る"),
    ("tp37", ""): ("M", "追い切りの調教師標準化・テキスト列の状態表を追加"),
    ("tp35", ""): ("M", "ラップ分類の統計表を追加"),
    ("tp34", ""): ("S", "既存の履歴から作れる（間隔・叩き回数）"),
    ("tp33", ""): ("S", "追い切りの場所と調教師の拠点から作れる"),
    ("tp40", "IN"): ("M", "追い切りの履歴の傾き。CY の状態表を流用"),
    ("tp40", "LY"): ("M", "追い切りの履歴。CY の状態表を流用"),
    ("v6", "s_CB"): ("S", "RB の履歴状態（指数）から作れる"),
    ("v6", "x_NJ"): ("L", "非 JRA 過去走の履歴を状態表に加える"),
    ("acq", ""): ("L", "取得済みデータの履歴状態を加える"),
    ("tp59", ""): ("M", "V5 の OOF 残差。運用側で毎日の更新が要る"),
    ("tp64", ""): ("M", "展開タイプの実測ラベルの履歴状態を追加"),
}

LEAK = {
    ("tp34", "R6_prev"): "前走の消耗の群。見かけの −0.0038 は端数リーク（決定ログ B 表）。修正後は +0.00010",
    ("tp34", "ALL_PASSED"): "R6_prev（端数リーク）を含む束",
    ("g6b", "p_slow"): "ラベルの有無が将来の情報になる（ラベルのある走だけ学習）。第 2 段で再現せず",
}

# 年別の OOF が保存されておらず、決定ログの数字だけが残る検討（決定ログ・各章から転記。pooled と上限）
MANUAL_ROWS: list[dict[str, Any]] = [
    {"theme": "tp16b", "group": "AF（後日談の変種）", "d_win": -0.00015, "hi95": 0.00001, "periods": {"pooled": -0.00015}, "n_cols": None,
     "note": "境界線上（決定ログ §27）。含水率の補完後は再評価が要る"},
    {"theme": "tp16", "group": "後日談ほか 10 群の最良", "d_win": -0.00011, "hi95": 0.00005, "periods": {"pooled": -0.00011}, "n_cols": None, "note": "効果が基準未満（§13）"},
    {"theme": "tp16b", "group": "CU（クッション値・含水率）", "d_win": None, "hi95": None, "periods": {}, "n_cols": None,
     "note": "含水率の欠け補完と 4 コーナー訂正（2020〜24）の後に未再評価。要再評価", "status_override": "要再評価(データ補完後)"},
    {"theme": "g3", "group": "cross_misc", "d_win": -0.00009, "hi95": 0.0001, "periods": {"stage1": -0.00028, "cumulative": -0.00009}, "n_cols": None,
     "note": "第 1 段 −0.00028（−0.00003）。累積では −0.00009 で不採用（§33）"},
    {"theme": "tp28", "group": "追い切りパターン別の統計(wp_f/roi/diff)", "d_win": 0.00008, "hi95": 0.00032, "periods": {"stage2": 0.00008}, "n_cols": 86, "note": "第 1 段の通過は見かけ（§33）"},
    {"theme": "tp31", "group": "コース特注 as-of lift 6 群", "d_win": 0.00002, "hi95": 0.0001, "periods": {"stage2": 0.00002}, "n_cols": None, "note": "V4 が吸収済み（§33）"},
    {"theme": "tp30", "group": "荒れ度の文脈列", "d_win": 0.00013, "hi95": 0.0003, "periods": {"pooled": 0.00013}, "n_cols": None, "note": "温度調整は悪化（§33）"},
    {"theme": "tp26", "group": "好走傾向が似たコース(統計のみ)", "d_win": -0.00014, "hi95": -0.00003, "periods": {"pooled": -0.00014}, "n_cols": None, "note": "木モデルが同じ情報を持つ（§33）", "dup_known": True},
    {"theme": "tp29", "group": "コース特注適性(期待外れ/期待以上)", "d_win": -0.00019, "hi95": -0.00003, "periods": {"pooled": -0.00019}, "n_cols": None, "note": "市場版の全部入り。複勝 −0.00022（§33）。人気に依存", "market_dep": True},
    {"theme": "tp15", "group": "コース概形だけの類似度", "d_win": -0.00006, "hi95": 0.00007, "periods": {"pooled": -0.00006}, "n_cols": None, "note": "b_aff_* と順位相関 0.83（§11）", "dup_known": True},
    {"theme": "tp25", "group": "ペース予測・ペース条件付き能力(tp25b)", "d_win": -0.0002, "hi95": -0.00003, "periods": {"pooled": -0.0002}, "d_top3": 0.00042, "n_cols": None, "note": "現行前段と併用で複勝 +0.00042（§33）"},
    {"theme": "tp25", "group": "位置取り・テン1F の前段(tp25a)", "d_win": -0.00012, "hi95": 0.00008, "periods": {"pooled": -0.00012}, "n_cols": None, "note": "S3 に重ねると情報が重複（§33）", "dup_known": True},
    {"theme": "tp21", "group": "SmartRC テン1F を直接", "d_win": -0.00014, "hi95": None, "periods": {"pooled": -0.00014}, "n_cols": None,
     "note": "出走後の取得で先読みの疑い。前段経由に変更（§16）", "leak": "SmartRC の保存は出走後の取得で先読みの疑い"},
    {"theme": "g6b", "group": "p_slow（出遅れ実ラベル)", "d_win": -0.00040, "hi95": 0.0004, "periods": {"stage1": -0.00073, "stage2": -0.00005}, "n_cols": 5,
     "note": "ラベルの有無が将来の情報", "leak": "ラベルの有無が将来の情報になる（ラベルのある走だけ学習）。第 2 段で再現せず"},
    {"theme": "tp56", "group": "HT（馬タイプ）", "d_win": -0.00016, "hi95": -0.00003, "periods": {"pooled": -0.00016}, "n_cols": None, "note": "基準 −0.0002 に届かず（§33）。タイプの能力差は V5 に入っている", "eval_base": "V5", "dup_known": True},
    {"theme": "tp59", "group": "rs59（関係者の残差 4 列）", "d_win": -0.00015, "hi95": -0.00002, "periods": {"pooled": -0.00015}, "d_top2": -0.00008, "d_top3": -0.00001, "n_cols": 4,
     "note": "境界（v6 章）。RB に対する増分は −0.00010（上限 +0.00007、CB+NJ+rs59 の束）。運用側で毎日の更新が要る", "eval_base": "V5(d_CY_BEST)"},
    {"theme": "tp64", "group": "PB4（展開利 4 列)", "d_win": None, "hi95": None, "periods": {}, "n_cols": 4,
     "note": "V5 への増分は基準未満、RB に足して増分なし（+0.00004）", "status_override": "不採用のまま"},
]


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _theme_of(path: str) -> str:
    p = path.replace("\\", "/")
    for t in ("tp32", "tp33", "tp34", "tp35", "tp37", "tp39", "tp40", "tp41", "tp42", "tp57_growth"):
        if f"/{t}/" in p:
            return "tp57" if t == "tp57_growth" else t
    for key, th in (("/tp38_out/", "tp38"), ("/tp60_out/", "tp60"), ("/acq/", "acq"), ("/v5_select/", "v5sel"), ("/v6/", "v6")):
        if key in p:
            return th
    return "other"


def _stage_of(path: str) -> str:
    p = path.replace("\\", "/").split("/")
    if "stage1_prelim_stale" in p:
        return "skip"
    for s in p:
        if s.startswith("stage") or s in ("eval", "eval_s2", "eval_rep"):
            return s
    return p[-2] if len(p) > 1 else ""


def pooled_observations(scan: dict[str, Any]) -> pd.DataFrame:
    """results.json の群ごとの評価（pooled）を (theme, group, stage) の表にする。基準は V4（または V5 / d_CY_BEST）。"""
    rows = []
    for r in scan["pooled"]:
        theme, stage = _theme_of(r["file"]), _stage_of(r["file"])
        if stage == "skip" or theme == "other":
            continue
        ctx = r["ctx"]
        var = r.get("variant") or ""
        if ctx and ctx[0] == "groups":
            group = ctx[1]
        elif ctx and ctx[0] == "pairs":
            group = ctx[1]
            if theme == "v6" and not var.endswith(" vs d_CY_BEST"):
                continue
            if theme == "acq" and not var.endswith(" vs d"):
                continue
            if theme == "v5sel" and not (" vs a_V4" in ctx[1] and ctx[1].split(" vs ")[0] in ("b_tlfresh", "c_CY", "d_CY_BEST")):
                continue
            group = group.split(" vs ")[0]
        else:
            continue
        rows.append({"theme": theme, "group": group, "stage": stage,
                     "d_win": r["Δwin"], "hi95_win": r["win_hi"], "d_top2": r.get("Δtop2", np.nan), "d_top3": r.get("Δtop3", np.nan)})
    return pd.DataFrame(rows)


def oof_year_table(scan: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for r in scan["oof_years"]:
        d = r["dir"].replace("\\", "/")
        theme = _theme_of(d + "/")
        rows.append({"theme": theme, "group": r["group"], "stage": _stage_of(d + "/x"), "year": r["year"], "n_races": r["n_races"],
                     "d_win": r["d_win"], "d_top2": r["d_top2"], "d_top3": r["d_top3"], "se_win": r.get("se_win", np.nan),
                     "hi95_win": r.get("hi95_win", np.nan)})
    return pd.DataFrame(rows)


def _stage_weight(stage: str) -> float:
    return 2.0 if str(stage).startswith("stage1") else 1.0   # 段階 1 は 2022+2023、段階 2 は 2024


def combine_group(obs: pd.DataFrame, years: pd.DataFrame) -> dict[str, Any]:
    """1 つの群の観測をまとめる。年別（OOF の再集計）が 2 年以上あれば年別、無ければ段階別を期間として扱う。"""
    out: dict[str, Any] = {}
    use_year = len(years) >= 2
    if use_year:
        y = years.sort_values(["year", "stage"]).drop_duplicates("year", keep="last")
        w = y["n_races"].to_numpy(float)
        out.update(granularity="year", periods={int(r.year): float(r.d_win) for r in y.itertuples()},
                   d_win_pool=float(np.average(y["d_win"], weights=w)),
                   d_top2_pool=float(np.average(y["d_top2"], weights=w)), d_top3_pool=float(np.average(y["d_top3"], weights=w)))
    if len(obs):
        w_obs = np.array([_stage_weight(s) for s in obs["stage"]])
        w_obs = w_obs / w_obs.sum()
        se_obs = ((obs["hi95_win"] - obs["d_win"]) / Z95).clip(lower=1e-6).to_numpy()
        se_pool = float(np.sqrt(np.sum((w_obs * se_obs) ** 2)))
        last = obs.sort_values("stage").iloc[-1]
        out["hi95_last"], out["d_win_last"], out["se_source"] = float(last["hi95_win"]), float(last["d_win"]), "ログの paired bootstrap"
        if not use_year:
            o = obs.sort_values("stage")
            out.update(granularity="stage", periods={str(r.stage): float(r.d_win) for r in o.itertuples()},
                       d_win_pool=float(np.average(o["d_win"], weights=w_obs[np.argsort(obs["stage"].to_numpy(), kind="stable")])),
                       d_top2_pool=float(np.average(o["d_top2"].fillna(0), weights=w_obs[np.argsort(obs["stage"].to_numpy(), kind="stable")])),
                       d_top3_pool=float(np.average(o["d_top3"].fillna(0), weights=w_obs[np.argsort(obs["stage"].to_numpy(), kind="stable")])))
        out["se_pool"] = se_pool
    else:   # OOF だけが残る群（v6 の評価など）。年別の bootstrap の標準誤差を割り増して使う
        se = float(np.sqrt(np.sum((y["n_races"] / y["n_races"].sum() * y["se_win"]) ** 2))) * OOF_SE_INFLATION
        out.update(se_pool=se, hi95_last=out["d_win_pool"] + Z95 * se, d_win_last=float(y.sort_values("year").iloc[-1]["d_win"]),
                   se_source=f"OOF 再集計×{OOF_SE_INFLATION}")
    out["n_periods"] = len(out["periods"])
    return out


def classify_levels(info: dict[str, Any]) -> dict[str, Any]:
    d, se = info["d_win_pool"], max(info["se_pool"], 1e-6)
    vals = list(info["periods"].values())
    enough = len(vals) >= 2
    same_neg = enough and all(v < 0 for v in vals)
    place_ok = info["d_top2_pool"] <= PLACE_TOL and info["d_top3_pool"] <= PLACE_TOL
    b = TAU_PRIOR**2 / (TAU_PRIOR**2 + se**2)
    pm, ps = b * d, math.sqrt(b) * se
    p_neg = norm_cdf(-pm / ps)
    return {
        "S1": bool(info["hi95_last"] < 0 and info["d_win_last"] <= -MIN_EFFECT_S1 and place_ok),
        "S2": bool(same_neg and d <= -MIN_EFFECT_S1 and place_ok),
        "S3": bool(same_neg and d <= -1e-4 and place_ok),
        "S4": bool(p_neg >= 0.8 and place_ok),
        "S5": bool(d + Z90 * se < 0 and place_ok),
        "p_neg": float(p_neg), "same_sign_neg": bool(same_neg), "place_ok": bool(place_ok), "enough_periods": bool(enough),
    }


def _match(theme: str, group: str, table: list[tuple[str, str]]) -> bool:
    return any(theme == t and (g == "" or group == g or group.startswith(g)) for t, g in table)


def cluster_of(theme: str, group: str) -> str:
    for name, members in CLUSTERS:
        if _match(theme, group, members):
            return name
    return f"{theme}:単独"


def serving_status(theme: str, group: str) -> str:
    return "あり" if (theme, group) in SERVING_OK_EXACT else "要整備"


def serving_note(theme: str, group: str) -> tuple[str, str]:
    for (t, g), v in SERVING_EFFORT.items():
        if theme == t and (g == "" or group == g or group.startswith(g)):
            return v
    return ("M", "推論側の生成経路を追加する")


EVAL_BASE = {"tp32": "V4", "tp33": "V4", "tp34": "V4", "tp35": "V4", "tp37": "V4", "tp39": "V4", "tp40": "V4", "tp41": "V4", "tp42": "V4",
             "tp57": "V4", "tp38": "V4", "tp60": "V5", "acq": "d(V5相当)", "v6": "V5(d_CY_BEST)", "v5sel": "V4(a_V4)"}


def is_composition(theme: str, group: str) -> bool:
    """群ではなく『構成』（複数群の組合せ、RB の上に足した構成など）の評価行。単独の復活候補にはしない。"""
    return theme in ("v6", "v5sel") and ("+" in group or group.startswith("z:") or "@" in group or group in ("d_CY_BEST", "c_CY", "b_tlfresh")) and (theme, group) not in {("v6", "z:RB"), }


def is_bundle(group: str) -> bool:
    return group.startswith(("ALL", "BATCH[")) or group.endswith(("ALL_PASSED", "_ALL"))


BUNDLE_MEMBERS = {
    ("tp39", "ALL_PASSED"): ["tp39:POP_megu", "tp39:KR_mga", "tp39:KR_pro", "tp39:KR_pt", "tp39:GR_meg", "tp39:GR_mga", "tp39:GR_pro", "tp39:GR_pt", "tp39:GR_comp"],
    ("tp42", "RL_ALL"): ["tp42:RL_A", "tp42:RL_A2", "tp42:RL_B", "tp42:RL_C"],
}


def reasons(info: dict[str, Any], lv: dict[str, Any]) -> str:
    out = []
    d = info["d_win_pool"]
    if d > 0:
        out.append("勝率が悪化")
    elif d > -1e-4:
        out.append("効果が小さい(>-0.0001)")
    elif d > -MIN_EFFECT_S1:
        out.append("最小効果量未満")
    if info["hi95_last"] >= 0:
        out.append("区間上限が0を跨ぐ")
    if not lv["enough_periods"]:
        out.append("期間が1つ(符号の一貫性は未確認)")
    elif not lv["same_sign_neg"]:
        out.append("期間で符号が混在")
    if not lv["place_ok"]:
        out.append("連対/複勝が悪化(>+0.0002)")
    return "／".join(out) or "基準は満たす"


def assign_status(r: pd.Series) -> str:
    """採用済み / R1(S2) / R2(S3 か S4) / 要整備(R1|R2: 統計は満たすが推論側が未整備) / 不採用のまま / 要再評価"""
    if r.get("status_override"):
        return r["status_override"]
    if r["adopted"]:
        return "採用済み"
    if r["leak"]:
        return "不採用のまま(リーク疑い)"
    if r.get("market_dep"):
        return "不採用のまま(市場情報に依存)"
    if r["kind"] == "composition":
        return "構成(参考)"
    if r["kind"] == "bundle":
        return "束(参考)"
    if r.get("dup_known") and not r["S2"]:
        return "不採用のまま(既存と重複)"
    level = "R1" if r["S2"] else ("R2" if (r["S3"] or r["S4"]) else "")
    if not level:
        return "不採用のまま"
    return level if r["serving"] == "あり" else f"要整備({level})"


def build_ledger(scan: dict[str, Any], col_counts: dict[str, dict[str, Any]] | None = None) -> pd.DataFrame:
    obs = pooled_observations(scan)
    yrs = oof_year_table(scan)
    keys = set(map(tuple, obs[["theme", "group"]].drop_duplicates().to_numpy().tolist())) | \
        set(map(tuple, yrs[["theme", "group"]].drop_duplicates().to_numpy().tolist()))
    # v5_select の構成 e/f/g と acq の NJ（v6 の x_NJ と同じ群）は群ではないか重複なので外す
    keys = {k for k in keys if not (k[0] == "v5sel" and k[1] not in ("b_tlfresh", "c_CY", "d_CY_BEST", "a_V4")) and k != ("acq", "NJ") and k != ("v5sel", "a_V4")}
    keys = {k for k in keys if not (k[0] == "v6" and k[1] in ("d_CY_BEST", "d_CY_BEST.v5select"))}
    rows = []
    for theme, group in sorted(keys):
        o = obs[(obs["theme"] == theme) & (obs["group"] == group)]
        y = yrs[(yrs["theme"] == theme) & (yrs["group"] == group)]
        if o.empty and len(y) < 2:
            continue
        info = combine_group(o, y)
        lv = classify_levels(info)
        rows.append(_row(theme, group, info, lv, col_counts))
    for m in MANUAL_ROWS:
        rows.append(_manual_row(m))
    tp76 = load_tp76()
    if tp76:
        rows.extend(tp76_rows(tp76))
    df = pd.DataFrame(rows)
    if "s4_3y" not in df:
        df["s4_3y"] = False
    old = df["s4_3y"].isna()
    df.loc[old, "s4_3y"] = [bool(r["S4"] and r["n_periods"] >= 3 and r["granularity"] == "year" and all(v < 0 for v in json.loads(r["periods"]).values()))
                            for _, r in df[old].iterrows()]
    df["s4_3y"] = df["s4_3y"].astype(bool)
    df["status"] = df.apply(assign_status, axis=1)
    df = pick_representatives(df)
    order = {"採用済み": 0, "R1": 1, "R2": 2, "要整備(R1)": 3, "要整備(R2)": 4, "束(参考)": 5, "要再評価(データ補完後)": 6}
    df["_o"] = df["status"].map(order).fillna(9)
    return df.sort_values(["_o", "d_win"]).drop(columns="_o").reset_index(drop=True)


def _row(theme: str, group: str, info: dict[str, Any], lv: dict[str, Any], col_counts) -> dict[str, Any]:
    topic, code, loc = THEMES.get(theme, ("", "", ""))
    cc = (col_counts or {}).get(f"{theme}:{group}") or (col_counts or {}).get(COLUMN_ALIAS.get(f"{theme}:{group}", ""), {})
    sv = "あり" if (theme, group) in ADOPTED_EXACT else serving_status(theme, group)
    eff, why = serving_note(theme, group)
    leak = LEAK.get((theme, group), "")
    return {
        "id": f"{theme}:{group}", "theme": theme, "group": group, "kind": "composition" if is_composition(theme, group) else ("bundle" if is_bundle(group) else "group"), "topic": topic,
        "eval_base": EVAL_BASE.get(theme, ""),
        "code": code, "data_dir": cc.get("path", loc), "n_cols": cc.get("n"),
        "granularity": info["granularity"], "periods": json.dumps(info["periods"], ensure_ascii=False), "n_periods": info["n_periods"],
        "d_win": round(info["d_win_pool"], 6), "se": round(info["se_pool"], 6), "se_source": info["se_source"],
        "hi95_last": round(info["hi95_last"], 6), "d_top2": round(info["d_top2_pool"], 6), "d_top3": round(info["d_top3_pool"], 6),
        **{k: lv[k] for k in ("S1", "S2", "S3", "S4", "S5")}, "p_neg": round(lv["p_neg"], 3),
        "adopted": (theme, group) in ADOPTED_EXACT, "cluster": cluster_of(theme, group), "serving": sv,
        "serving_effort": "" if sv == "あり" else eff, "serving_note": "" if sv == "あり" else why,
        "needs_recompute": info["granularity"] != "year", "reason": reasons(info, lv), "leak": leak, "source": "評価結果(OOF・results.json)",
        "dup_known": False, "market_dep": False,
        "status_override": "", "members": json.dumps(BUNDLE_MEMBERS.get((theme, group), []), ensure_ascii=False),
    }


def _manual_row(m: dict[str, Any]) -> dict[str, Any]:
    theme, group = m["theme"], m["group"]
    topic, code, loc = THEMES.get(theme, ("", "", ""))
    d = m["d_win"]
    hi = m["hi95"]
    periods = m.get("periods") or {}
    d_top2, d_top3 = m.get("d_top2", 0.0), m.get("d_top3", 0.0)
    place_ok = d_top2 <= PLACE_TOL and d_top3 <= PLACE_TOL
    if d is None:
        s = {k: False for k in ("S1", "S2", "S3", "S4", "S5")}
        se, p_neg = None, None
    else:
        se = (hi - d) / Z95 if hi is not None else None
        s = {"S1": bool(hi is not None and hi < 0 and d <= -MIN_EFFECT_S1 and place_ok), "S2": False, "S3": False,
             "S4": False, "S5": False}
        if se and se > 0:
            b = TAU_PRIOR**2 / (TAU_PRIOR**2 + se**2)
            p_neg = norm_cdf(-(b * d) / (math.sqrt(b) * se))
            s["S4"], s["S5"] = bool(p_neg >= 0.8 and place_ok), bool(d + Z90 * se < 0 and place_ok)
            vals = list(periods.values())
            if len(vals) >= 2 and all(v < 0 for v in vals):
                s["S2"], s["S3"] = bool(d <= -MIN_EFFECT_S1 and place_ok), bool(d <= -1e-4 and place_ok)
        else:
            p_neg = None
    leak = LEAK.get((theme, group), m.get("leak", ""))
    sv = serving_status(theme, group)
    eff, why = serving_note(theme, group)
    return {
        "id": f"{theme}:{group}", "theme": theme, "group": group, "kind": "group", "topic": topic, "code": code, "data_dir": loc,
        "eval_base": m.get("eval_base", EVAL_BASE.get(theme, "")), "dup_known": bool(m.get("dup_known", False)), "market_dep": bool(m.get("market_dep", False)),
        "n_cols": m.get("n_cols"), "granularity": "決定ログ", "periods": json.dumps(periods, ensure_ascii=False), "n_periods": len(periods),
        "d_win": d, "se": None if se is None else round(se, 6), "se_source": "決定ログの上限から逆算", "hi95_last": hi,
        "d_top2": d_top2, "d_top3": d_top3, **s, "p_neg": None if p_neg is None else round(p_neg, 3),
        "adopted": False, "cluster": cluster_of(theme, group), "serving": sv, "serving_effort": "" if sv == "あり" else eff,
        "serving_note": "" if sv == "あり" else why, "needs_recompute": True,
        "reason": m.get("note", ""), "leak": leak, "source": "決定ログの数字", "status_override": m.get("status_override", ""),
        "members": "[]",
    }


def pick_representatives(df: pd.DataFrame) -> pd.DataFrame:
    """同じクラスタに R 候補が複数あれば、効果の大きい 1 つを代表にして、残りは冗長にする（採用済みと同じクラスタも冗長）。"""
    df = df.copy()
    df["redundant_with"] = ""
    df["overlaps_adopted"] = ""
    cand = df["status"].str.contains("R1|R2", regex=True) & (df["kind"] == "group")
    adopted_clusters = df.loc[df["adopted"]].groupby("cluster")["id"].first().to_dict()
    # tp38 の変種は採用済み ALL_BEST（PX_cell_sd・PX_emb・PX_ridge_als）と同じ祖先表から作る派生なので、代表を立てずに冗長として扱う
    variants = cand & (df["theme"] == "tp38")
    df.loc[variants, "redundant_with"] = "tp38:ALL_BEST（採用済み）"
    df["_lv"] = df["status"].str.contains("R1").map({True: 0, False: 1})
    for cl, g in df[cand & ~variants].groupby("cluster"):
        order = g.sort_values(["_lv", "d_win"])
        keep = order.index[0]
        for idx in order.index[1:]:
            df.loc[idx, "redundant_with"] = df.loc[keep, "id"]
    for idx in df[cand].index:
        cl = df.loc[idx, "cluster"]
        if cl in adopted_clusters:
            df.loc[idx, "overlaps_adopted"] = adopted_clusters[cl]
    df = df.drop(columns="_lv")
    return df


# 復活セットの群 → v6_eval の外部群の名前（REV_<tp>_<群>）。既存の v6_eval が読める名前はそのまま使う
V6_EVAL_NAMES = {"v6:x_NJ": "NJ", "v6:s_CB": "CB", "tp59:rs59（関係者の残差 4 列）": "RS59", "tp64:PB4（展開利 4 列)": "PB4"}


COLUMN_ALIAS = {"v6:x_NJ": "acq:NJ", "v6:s_CB": "v6:CB"}


def short_name(gid: str) -> str:
    if gid in V6_EVAL_NAMES:
        return V6_EVAL_NAMES[gid]
    theme, group = gid.split(":", 1)
    return f"REV_{theme}_{group}"


def _clean(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if math.isnan(float(v)) else float(v)
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def build_feature_sets(df: pd.DataFrame, columns: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """基準（今までの基準 = v6e）と復活セット（R1・R1+R2・広い版）の定義。学習はこの定義の列を足して比べる。"""
    groups = df[(df["kind"] == "group") & df["status"].str.contains("R1|R2", regex=True)].copy()
    groups["_rank"] = groups["status"].str.contains("R1").map({True: 0, False: 1})
    rep = groups[groups["redundant_with"].fillna("") == ""]

    def entry(r: pd.Series) -> dict[str, Any]:
        gid = r["id"]
        c = columns.get(gid) or columns.get(f"{r['theme']}:{r['group']}") or columns.get(COLUMN_ALIAS.get(gid, "")) or {}
        return {"id": gid, "name": short_name(gid), "level": "R1" if "R1" in r["status"] else "R2", "n_cols": _clean(r["n_cols"]) or c.get("n"),
                "path": c.get("path") or "", "serving": r["serving"], "serving_effort": r["serving_effort"] if isinstance(r["serving_effort"], str) else "",
                "d_win": _clean(r["d_win"]), "cluster": r["cluster"], "buildable": bool(c.get("path")) or gid in V6_EVAL_NAMES}

    r1_all = [entry(r) for _, r in rep[rep["_rank"] == 0].iterrows()]
    r2_all = [entry(r) for _, r in rep[rep["_rank"] == 1].iterrows()]
    r1, r2 = [x for x in r1_all if x["buildable"]], [x for x in r2_all if x["buildable"]]
    unbuilt = [x["id"] for x in r1_all + r2_all if not x["buildable"]]
    wide = [entry(r) for _, r in groups.iterrows()]
    # 広い版: 学習表に列がある群。tp38 の変種は採用済み ALL_BEST の列と重なるので外す
    uniq_wide = [w for w in wide if w["buildable"] and not w["id"].startswith("tp38:")]
    s76_groups, s76_sets = tp76_sets(df)
    base_sets = {
        "baseline": {"base": None, "add": [], "doc": "今までの基準（v6e の構成）"},
        "revival_r1": {"base": "baseline", "add": [x["name"] for x in r1], "doc": "R1: S2 を満たす群の代表（クラスタごとに 1 つ）"},
        "revival_r1r2": {"base": "baseline", "add": [x["name"] for x in r1 + r2], "doc": "R1 + R2（S3 か S4）の代表"},
        "revival_wide": {"base": "baseline", "add": [x["name"] for x in uniq_wide], "doc": "R1・R2 の全群（冗長を含む。多数の弱いシグナルを束ねる効果を見る）"},
    }
    all_sets = {**base_sets, **s76_sets}
    for name, a, b in (("revival_r1+screen_p1", "revival_r1", "screen_p1"), ("revival_r1r2+screen_p1p2", "revival_r1r2", "screen_p1p2"),
                       ("revival_wide+screen_all", "revival_wide", "screen_all")):
        all_sets[name] = {"base": "baseline", "add": all_sets[a]["add"] + all_sets[b]["add"], "doc": f"{a} + {b}（復活セットと tp-76 の組合せ）"}
    groups = {x["name"]: x for x in {w["name"]: w for w in wide}.values()}
    groups.update(s76_groups)
    for v in all_sets.values():   # 学習表に列が無い群（構築が要るもの）を除いた、すぐ評価できる群
        v["add_ready"] = [a for a in v["add"] if groups.get(a, {}).get("buildable", True)]
    plan_ids = [("A", "baseline"), ("B", "revival_r1"), ("C", "revival_r1r2"), ("D", "revival_wide"), ("E", "screen_p1"), ("F", "screen_p1p2"),
                ("G", "screen_all"), ("H", "revival_r1+screen_p1"), ("I", "revival_r1r2+screen_p1p2"), ("J", "revival_wide+screen_all")]
    return {
        "version": 2, "frozen_on": "2026-10-22",
        "doc": "復活セット: 過去の採用検討で不採用・保留・境界だった群のうち、緩めた基準（S2〜S4）で復活するもの、と tp-76（学習前スクリーニング）の候補。台帳: docs/operations/revival_feature_ledger.csv、章: revival_feature_set.html",
        "criteria": {"S1": "片側95%上限<0 かつ 効果≤-0.0002（従来）", "S2": "2期間以上で同符号 かつ 点推定≤-0.0002", "S3": "2期間以上で同符号 かつ 点推定≤-0.0001",
                     "S4": "懐疑的事前分布 N(0,0.0002^2) の事後で P(効果<0)≥0.8", "S5": "片側90%上限<0", "place_ok": "連対・複勝の悪化は +0.0002 以内"},
        "false_positive_rate": {
            "doc": "シャドウ特徴量 81 単位（日内の並べ替え・レース内で馬をずらす）を tp-76 の手順に通した実測。tp-76 の候補（v6e の logit への少数パラメータの追加）での値で、台帳の群の再集計にそのまま当てはまるとは限らない",
            "measured": {k: {"fpr": TP76_MEASURED_FPR[k], "real_pass": TP76_REAL_PASS[k], "n_real": 27} for k in TP76_MEASURED_FPR},
            "recommend": "特徴量の候補入りは S4 かつ LOYO 3 年同符号（偽陽性率 2.5%）。保守的な代替は S5a（0%）。S2・S3 は偽陽性率 0% だが、要求する効果が候補の実力より大きく通る本物が 0。馬券の規則の基準は緩めない",
        },
        "baseline": {
            "name": "v6e", "n_cols_bundle": 1231, "v6_eval_name": "z:RB@e",
            "note": "V4(owner125+self8 199+field7 451+ped32+aux27、tl_ は再取得版に差し替え)+CY37+ALL_BEST43(+差分)+RB16、前段は e 版。列の一覧は serving_v4_columns.json / serving_v5_columns.json / serving_v6.py",
            "components": [{"name": "V4", "n_cols": 1062}, {"name": "tl_ 再取得版", "n_cols": 0, "note": "差し替え"}, {"name": "CY", "n_cols": 37}, {"name": "ALL_BEST", "n_cols": 43},
                           {"name": "RB", "n_cols": 16}, {"name": "前段 e", "n_cols": 0, "note": "差し替え"}],
        },
        "sets": all_sets,
        "groups": groups,
        "unbuilt": unbuilt + [g for g, v in s76_groups.items() if not v["buildable"]],
        "screen_unassigned": {k: v for k, v in TP76_UNASSIGNED.items()},
        "comparison": {"plan": [{"id": i, "set": n} for i, n in plan_ids],
                       "estimate": "1 構成の LOYO 評価（3 年 × 2 seed × 3 目的）は 0.5〜1 時間（過去の実績からの目安。実測ではない）。10 構成を直列で 5〜10 時間、共有 3 スロットで約 2〜4 時間。"
                                   "合成スコアは、学習前に OOF の列を保存する処理（tp76_screen の composite）が要る。seed 2・3 の確認と通った構成の LOYO アンサンブル・最終学習は別枠（8〜9 時間）"},
        "serving_note": "復活セット・tp-76 の群は、推論側の生成経路が未整備（採用時に整備）。比較は学習表の列でのオフライン評価",
    }


def tp76_sets(df: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """tp-76 の候補を構成の群と構成（screen_p1・screen_p1p2・screen_all）にする。合成スコアは列がまだ無いので『構築が要る』群にする。"""
    rank = {"S": 0, "M": 1, "L": 2}
    t = df[df["theme"] == "tp76"]
    groups: dict[str, dict[str, Any]] = {}
    for name, cols in TP76_SET_COLS.items():
        units = [u for u, g in TP76_SET_GROUP.items() if g == name]
        rows = t[t["group"].isin(units)]
        eff = max((e for e in rows["serving_effort"].dropna()), key=lambda e: rank.get(e, 1), default="M")
        pri = min((TP76_PRIORITY[u] for u in units))
        groups[name] = {"id": name, "name": name, "level": f"screen_p{pri}", "n_cols": len(cols), "cols": cols, "path": TP76_FEATURES, "units": units,
                        "serving": "要整備", "serving_effort": eff, "buildable": True,
                        "rerun_after_complete_10gen": any(u in TP76_RERUN_PED for u in units), "d_win": _clean(rows["d_win"].min()) if len(rows) else None}
    comp = t[t["group"] == "合成スコア"]
    groups["S76_COMPOSITE"] = {
        "id": "S76_COMPOSITE", "name": "S76_COMPOSITE", "level": "screen_p1", "n_cols": 2, "cols": ["comp_win", "comp_top3"], "path": "", "units": ["合成スコア"],
        "serving": "要整備", "serving_effort": "L", "buildable": False, "rerun_after_complete_10gen": True,
        "d_win": _clean(comp["d_win"].iloc[0]) if len(comp) else None,
        "build_note": "弱いシグナルの代表 26 列を L2 ロジスティックに当てはめた線形予測（勝ち用・3 着内用）。係数は A 年、較正は B 年、評価は C 年。年ごとの OOF の列として保存する処理が要る（学習再開前に tp76_screen の composite を保存形式にする）",
    }
    p1 = ["S76_COMPOSITE", "S76_D2", "S76_D5", "S76_D5r"]
    p1p2 = p1 + ["S76_A2", "S76_B3"]
    allp = p1p2 + ["S76_E1", "S76_E2", "S76_D4", "S76_D6", "S76_C", "S76_F", "S76_G"]
    sets = {
        "screen_p1": {"base": "baseline", "add": p1, "doc": "tp-76 優先1: 合成スコア + D2（競馬場×面）+ D5（4 軸の適性）+ D5r（同・種牡馬の事前なし）"},
        "screen_p1p2": {"base": "baseline", "add": p1p2, "doc": "tp-76 優先1+2: + A2（日内の内有利×枠）+ B3（めぐ指数のカルマン）"},
        "screen_all": {"base": "baseline", "add": allp, "doc": "tp-76 優先1〜3 + 束に入れて除去で判断（C・F・G）: + E1・E2・D4・D6 + C・F・G"},
    }
    return groups, sets


def write_feature_sets(df: pd.DataFrame, columns: dict[str, dict[str, Any]], path: Path = FEATURE_SETS) -> dict[str, Any]:
    cfg = build_feature_sets(df, columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return cfg


# ---- tp-76（学習前の特徴量スクリーニング）の候補を台帳と構成に取り込む ----
TP76_FEATURES = "/home/jovyan/work/tp76_out/features.parquet"
TP76_PRIORITY = {   # ユーザー指示（2026-10-22）による優先度
    "D2": 1, "D5": 1, "D5r": 1, "A2": 2, "B3": 2, "E1": 3, "E2": 3, "D4": 3, "D6": 3,
    "C1": 4, "C2": 4, "C3": 4, "F1": 4, "F2": 4, "F3": 4, "G1": 4, "G2": 4, "G3": 4,
}
TP76_PRIORITY_LABEL = {0: "優先1(合成スコア)", 1: "優先1", 2: "優先2", 3: "優先3", 4: "束に入れて除去で判断"}
TP76_UNASSIGNED = {"B1": "tp-76 の判定は『優先』(S4・S5b)", "B2": "tp-76 の判定は『優先』(S4・S5b)", "D1": "tp-76 の判定は『優先』(S4・S5b)",
                   "D3": "tp-76 の判定は『優先』(S4・S5b)", "A1": "tp-76 の判定は『後回し』", "A3": "tp-76 の判定は『後回し』",
                   "E3": "tp-76 の判定は『後回し』", "E4": "tp-76 の判定は『後回し』", "E5": "tp-76 の判定は『後回し』"}
# 構成に入れるときの群（C・F・G は族ごとにまとめる。列の重なりを避ける）
TP76_SET_GROUP = {"D2": "S76_D2", "D5": "S76_D5", "D5r": "S76_D5r", "A2": "S76_A2", "B3": "S76_B3", "E1": "S76_E1", "E2": "S76_E2",
                  "D4": "S76_D4", "D6": "S76_D6", "C1": "S76_C", "C2": "S76_C", "C3": "S76_C", "F1": "S76_F", "F2": "S76_F", "F3": "S76_F",
                  "G1": "S76_G", "G2": "S76_G", "G3": "S76_G"}
TP76_SET_COLS = {"S76_D2": ["D_venue"], "S76_D5": ["D_all"], "S76_D5r": ["D_raw_all"], "S76_A2": ["A_fit_inner"], "S76_B3": ["B_m_megu"],
                 "S76_E1": ["E_abs"], "S76_E2": ["E_day"], "S76_D4": ["D_cush"], "S76_D6": ["D_ag_all"],
                 "S76_C": ["C_elo", "C_rel", "C_unc"], "S76_F": ["F_n30", "F_n90", "F_since", "F_dist90", "F_burden90", "F_tataki"],
                 "S76_G": ["G_n_front", "G_front_rank", "G_inner_front", "G_fit"]}
TP76_RERUN_PED = {"B1", "B2", "B3", "D1", "D2", "D3", "D4", "D5", "D6"}   # D5r は種牡馬の事前なしなので対象外
TP76_EFFORT = {"低": "S", "中": "M", "高": "L"}
TP76_MEASURED_FPR = {"S1": 0.0, "S2": 0.0, "S3": 0.0, "S4": 0.025, "S5a": 0.0, "S5b": 0.012}   # シャドウ 81 単位（tp-76 §4）
TP76_REAL_PASS = {"S1": 0, "S2": 0, "S3": 0, "S4": 13, "S5a": 7, "S5b": 13}


def load_tp76(path: Path = TP76_RESULTS) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _tp76_row(unit: str, u: dict[str, Any], *, composite: bool = False) -> dict[str, Any]:
    d3, se3 = u["d3"], max(u["se3"], 1e-9)
    periods = {y: u[f"d3_{y}"] for y in (2022, 2023, 2024) if f"d3_{y}" in u}
    p = u.get("pass", {})
    pri = 0 if composite else TP76_PRIORITY.get(unit)
    label = "未割当" if pri is None else TP76_PRIORITY_LABEL[pri]
    sv_eff = "L" if composite else TP76_EFFORT.get(u.get("infer_cost", "中"), "M")
    why = ("D の事前表（馬×条件・種牡馬×条件）、B の日次状態、A の日内結果が要る。係数は発見期間の外側で固定して持ち込む" if composite
           else u.get("cost_note", ""))
    s4 = bool(p.get("S4", (u.get("post_neg") or 0) >= 0.8))
    s4_3y = bool(s4 and u.get("signs_neg", 0) >= 3) if not composite else True
    ped = (unit in TP76_RERUN_PED) or composite
    return {
        "id": f"tp76:{unit}", "theme": "tp76", "group": unit, "kind": "group", "topic": "学習前スクリーニング（tp-76）" + f"：{u.get('family', '')}",
        "eval_base": "v6e(LOYO OOF の logit)", "metric": "Δ3(3目的の1頭あたり平均)", "code": "src/pipeline/features/tp76_asof.py",
        "data_dir": "" if composite else TP76_FEATURES, "n_cols": 2 if composite else len(u.get("cols", [])), "cols": ",".join(u.get("cols", [])),
        "granularity": "year", "periods": json.dumps(periods, ensure_ascii=False), "n_periods": len(periods),
        "d_win": d3, "se": se3, "se_source": "tp-76 のクラスタ頑健 SE", "hi95_last": d3 + Z95 * se3,
        "d_top2": u["d_top2"], "d_top3": u["d_top3"],
        "S1": bool(p.get("S1", False)), "S2": bool(p.get("S2", False)), "S3": bool(p.get("S3", False)), "S4": s4,
        "S5": bool(p.get("S5a", False)), "S5b": bool(p.get("S5b", False)), "p_neg": u.get("post_neg"), "s4_3y": s4_3y,
        "d_2025": u.get("d3_2025"), "screen_priority": label, "rerun_after_complete_10gen": bool(ped),
        "adopted": False, "cluster": "tp76:" + str(u.get("family", "合成スコア")), "serving": "要整備", "serving_effort": sv_eff, "serving_note": why,
        "needs_recompute": False, "reason": f"2025 年(1 回) {u.get('d3_2025', float('nan')):+.2e}／LOYO の年の符号一致 {u.get('signs_neg', '')}" + (f"／{TP76_UNASSIGNED[unit]}" if unit in TP76_UNASSIGNED else ""), "leak": "",
        "source": "tp-76 の結果(results.json)", "status_override": f"スクリーニング({label})", "members": "[]",
        "dup_known": False, "market_dep": False, "redundant_with": "", "overlaps_adopted": "",
    }


def tp76_rows(res: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [_tp76_row(k, v) for k, v in res["units"].items()]
    comp = res.get("composite_nested")
    if comp:
        c = dict(comp)
        c.update(family="合成スコア(弱いシグナルの束)", cols=["合成スコア(勝ち用・3着内用)"], infer_cost="高", signs_neg=comp.get("signs_neg", 3))
        c.setdefault("post_neg", None)
        c["pass"] = {"S4": True}   # 合成スコアは LOYO の区間が 0 を外れ、3 年とも負（偽の合成スコアは悪化側）
        rows.append(_tp76_row("合成スコア", c, composite=True))
    return rows


def load_scan(path: Path = SCAN) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_columns(path: Path = COLUMNS) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> None:
    cols = load_columns()
    df = build_ledger(load_scan(), cols)
    LEDGER_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LEDGER_CSV, index=False)
    cfg = write_feature_sets(df, cols)
    print(df["status"].value_counts().to_string())
    for k, v in cfg["sets"].items():
        print(k, len(v["add"]), v["add"])


if __name__ == "__main__":
    main()
