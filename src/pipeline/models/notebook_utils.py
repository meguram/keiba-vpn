"""
予測モデル系ノートブック共通ユーティリティ。

提供機能:
  - load_master() / load_raw_results() : データ読み込み
  - encode_cats()                       : カテゴリカル列を Categorical dtype 化
  - FEATURE_BASE / feature_set()        : 特徴量定義
  - train_lgb()                         : LightGBM 学習（surface 別 + early stopping）
  - oof_predict()                       : GroupKFold OOF 予測（グループ = race_id）
  - eval_classification() / eval_regression() : 評価
  - save_oof() / load_oof()             : OOF 予測の保存・読み込み
  - harville_place() / harville_show()  : Harville 式による連対率・複勝率導出
  - softmax_normalize()                 : レース内 softmax 正規化
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# パス定義
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path("/home/jovyan/work/keiba-vpn")
MODELING_DIR  = PROJECT_ROOT / "data/local/modeling"
OOF_DIR       = MODELING_DIR / "oof"
TABLES_DIR    = PROJECT_ROOT / "data/page_reference/tables"
ALL_YEARS     = [str(y) for y in range(2020, 2026)]
SURFACE_CATS  = ["turf", "dirt", "hurdle"]

OOF_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# カテゴリカル列の定義
# ─────────────────────────────────────────────────────────────────────────────
CAT_COLS = [
    "venue", "surface", "direction", "grade", "race_class",
    "weather", "track_condition", "weight_rule", "course_type",
    "sex", "surface_cat",
]

# ─────────────────────────────────────────────────────────────────────────────
# 特徴量グループ定義
# ─────────────────────────────────────────────────────────────────────────────
FEAT_RACE = [
    "distance", "field_size", "venue_code", "jockey_weight",
    "bracket_number", "horse_number",
    # categorical (handled separately)
    "venue", "surface", "direction", "grade", "track_condition", "weather",
]

FEAT_HORSE = ["weight", "weight_change", "age", "sex"]

FEAT_JK_BASE = [
    "jk_prior_all_win_rate", "jk_prior_all_top3_rate", "jk_prior_all_avg_finish",
    "jk_prior_all_avg_pass_first", "jk_prior_all_avg_pass_norm_first",
    "jk_roll10_wins", "jk_roll10_starts", "jk_roll10_avg_finish",
    "jk_roll10_avg_pass_first",
    "jk_roll30_wins", "jk_roll30_starts", "jk_roll30_avg_finish",
    "jk_at_venue_win_rate", "jk_at_venue_starts",
    "jk_at_surface_win_rate", "jk_at_surface_starts",
    "jk_at_dist_win_rate", "jk_at_dist_starts",
    "jk_at_grade_win_rate", "jk_at_grade_starts",
    "jk_at_track_cond_win_rate",
    "jk_cal90_wins", "jk_cal90_starts", "jk_cal365_wins",
]

FEAT_TR_BASE = [
    "tr_prior_all_win_rate", "tr_prior_all_top3_rate", "tr_prior_all_avg_finish",
    "tr_roll10_wins", "tr_roll10_starts", "tr_roll10_avg_finish",
    "tr_roll30_wins", "tr_roll30_starts",
    "tr_at_venue_win_rate", "tr_at_surface_win_rate",
    "tr_at_dist_win_rate", "tr_at_grade_win_rate",
    "tr_cal90_wins", "tr_cal90_starts",
]

FEAT_SPEED = [
    "speed_max", "speed_avg", "speed_distance",
    "speed_recent_1", "speed_recent_2", "speed_recent_3",
]

FEAT_SIRE = [
    "sire_prior_win_rate", "sire_prior_top3_rate",
    "sire_prior_win_rate_surface", "sire_prior_top3_rate_surface",
    "sire_prior_win_rate_dist", "sire_prior_top3_rate_dist",
    "dam_sire_prior_win_rate", "dam_sire_prior_win_rate_surface",
]


def feature_set(
    df: pd.DataFrame,
    extra: Optional[list[str]] = None,
    exclude: Optional[list[str]] = None,
) -> list[str]:
    """
    データフレームに存在する列だけを返す特徴量リスト。
    extra: 追加したい列, exclude: 除外したい列。
    """
    base = FEAT_RACE + FEAT_HORSE + FEAT_JK_BASE + FEAT_TR_BASE + FEAT_SPEED + FEAT_SIRE
    if extra:
        base = base + extra
    if exclude:
        base = [c for c in base if c not in exclude]
    # df に存在する列のみ
    available = [c for c in base if c in df.columns]
    # 重複を除去して順序を保持
    seen: set[str] = set()
    result: list[str] = []
    for c in available:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# データ読み込み
# ─────────────────────────────────────────────────────────────────────────────

def load_master(v2: bool = False) -> pd.DataFrame:
    """master_dataset_full[_v2].parquet を読み込む。"""
    suffix = "_v2" if v2 else ""
    p = MODELING_DIR / f"master_dataset_full{suffix}.parquet"
    if not p.exists() and v2:
        logger.warning("_v2 ファイルが存在しません。通常版にフォールバックします。")
        p = MODELING_DIR / "master_dataset_full.parquet"
    df = pd.read_parquet(p)
    # surface_cat が存在しない場合は surface 列から導出する
    if "surface_cat" not in df.columns:
        surf_map = {"芝": "turf", "ダート": "dirt", "障": "hurdle"}
        df["surface_cat"] = df["surface"].astype(str).map(surf_map).fillna("other")
    else:
        # Categorical 型の場合は str に変換しておく
        df["surface_cat"] = df["surface_cat"].astype(str)
    # venue_code: '05' などの文字列を整数に変換する
    if "venue_code" in df.columns:
        df["venue_code"] = pd.to_numeric(df["venue_code"], errors="coerce").astype("Int64")
    logger.info("load_master: %s shape=%s", p.name, df.shape)
    return df


def load_raw_results(cols: Optional[list[str]] = None) -> pd.DataFrame:
    """race_result_flat を全年ロード（passing_order / pace など取得用）。"""
    parts = []
    default_cols = ["race_id", "horse_number", "passing_order", "pace",
                    "finish_position", "field_size", "date", "surface", "distance",
                    "venue_code"]
    load_cols = cols or default_cols
    for year in ALL_YEARS:
        p = TABLES_DIR / year / "race_result_flat.parquet"
        if p.exists():
            all_cols = pd.read_parquet(p).columns.tolist()
            available = [c for c in load_cols if c in all_cols]
            parts.append(pd.read_parquet(p, columns=available))
    df = pd.concat(parts, ignore_index=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 前処理
# ─────────────────────────────────────────────────────────────────────────────

def encode_cats(df: pd.DataFrame, cat_cols: Optional[list[str]] = None) -> pd.DataFrame:
    """指定列（デフォルト CAT_COLS）を category dtype に変換する。"""
    df = df.copy()
    for col in (cat_cols or CAT_COLS):
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def derive_running_style(passing_order: Any, field_size: int) -> Optional[str]:
    """
    passing_order (例: '3-2-2-1') → 脚質カテゴリ。
    - 逃げ: 平均順位 <= 1.5
    - 先行: 平均順位 <= field_size × 0.30
    - 差し: 平均順位 <= field_size × 0.65
    - 追い込み: それ以上
    """
    if pd.isna(passing_order):
        return None
    s = str(passing_order).strip()
    if not s or s in ("nan", "None", ""):
        return None
    try:
        parts = [p.strip("()") for p in s.split("-")]
        positions = [float(p) for p in parts if p.replace(".", "").isdigit()]
    except (ValueError, AttributeError):
        return None
    if not positions:
        return None
    avg = sum(positions) / len(positions)
    n = max(1, int(field_size))
    if avg <= 1.5:
        return "逃"
    elif avg <= n * 0.30:
        return "先"
    elif avg <= n * 0.65:
        return "差"
    else:
        return "追"


def derive_pace_category(pace_json: Any) -> Optional[str]:
    """
    pace JSON 文字列 → H/M/S カテゴリ。
    first_half_3f < second_half_3f - 1.0 → S (スロー)
    first_half_3f > second_half_3f + 1.0 → H (ハイ)
    それ以外 → M (ミドル)
    """
    if pd.isna(pace_json):
        return None
    try:
        d = json.loads(pace_json) if isinstance(pace_json, str) else pace_json
        t3f = float(d.get("first_half_3f") or d.get("t3f", 0))
        l3f = float(d.get("second_half_3f") or d.get("l3f", 0))
        if t3f == 0 or l3f == 0:
            return None
        diff = t3f - l3f
        if diff > 1.0:
            return "H"
        elif diff < -1.0:
            return "S"
        else:
            return "M"
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 学習
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PARAMS_BINARY = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "seed": 42,
}

DEFAULT_PARAMS_MULTICLASS = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
}

DEFAULT_PARAMS_REGRESSION = {
    "objective": "regression",
    "metric": ["mae", "mse"],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "seed": 42,
}


def train_lgb(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    features: list[str],
    target: str,
    params: dict[str, Any],
    cat_features: Optional[list[str]] = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 50,
    label_encoder: Optional[dict[str, int]] = None,
) -> lgb.Booster:
    """
    LightGBM モデルを学習する。

    label_encoder: multiclass の場合、ラベル文字列 → int のマッピング。
    """
    feats = [f for f in features if f in df_train.columns]
    cats  = [f for f in (cat_features or []) if f in df_train.columns]

    X_tr = df_train[feats]
    X_vl = df_valid[feats]

    if label_encoder:
        y_tr = df_train[target].map(label_encoder).fillna(-1).astype(int)
        y_vl = df_valid[target].map(label_encoder).fillna(-1).astype(int)
    else:
        y_tr = pd.to_numeric(df_train[target], errors="coerce")
        y_vl = pd.to_numeric(df_valid[target], errors="coerce")

    # NaN ターゲットを除外
    mask_tr = y_tr.notna()
    mask_vl = y_vl.notna()

    ds_tr = lgb.Dataset(X_tr[mask_tr], label=y_tr[mask_tr],
                        categorical_feature=cats or "auto", free_raw_data=False)
    ds_vl = lgb.Dataset(X_vl[mask_vl], label=y_vl[mask_vl],
                        categorical_feature=cats or "auto", free_raw_data=False,
                        reference=ds_tr)

    callbacks = [
        lgb.early_stopping(early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        params,
        ds_tr,
        num_boost_round=num_boost_round,
        valid_sets=[ds_tr, ds_vl],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )
    return model


def oof_predict(
    df_train: pd.DataFrame,
    features: list[str],
    target: str,
    params: dict[str, Any],
    group_col: str = "race_id",
    n_splits: int = 5,
    cat_features: Optional[list[str]] = None,
    label_encoder: Optional[dict[str, int]] = None,
) -> pd.Series:
    """
    GroupKFold OOF 予測。グループ = group_col（同一レースは同一 fold）。
    Returns: 予測値の pd.Series（index は df_train の index）。
    """
    feats    = [f for f in features if f in df_train.columns]
    cats     = [f for f in (cat_features or []) if f in df_train.columns]
    groups   = df_train[group_col].values
    X        = df_train[feats]

    if label_encoder:
        y = df_train[target].map(label_encoder).fillna(-1).astype(int)
    else:
        y = pd.to_numeric(df_train[target], errors="coerce")

    valid_mask = y.notna()
    oof = pd.Series(np.nan, index=df_train.index)

    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr_idx, vl_idx) in enumerate(gkf.split(X, y, groups=groups)):
        # valid 側で NaN ターゲットを除外
        vl_idx_clean = [i for i in vl_idx if valid_mask.iloc[i]]
        tr_idx_clean = [i for i in tr_idx if valid_mask.iloc[i]]
        if not tr_idx_clean or not vl_idx_clean:
            continue

        X_tr, y_tr = X.iloc[tr_idx_clean], y.iloc[tr_idx_clean]
        X_vl, y_vl = X.iloc[vl_idx_clean], y.iloc[vl_idx_clean]

        ds_tr = lgb.Dataset(X_tr, label=y_tr.values,
                            categorical_feature=cats or "auto", free_raw_data=False)
        ds_vl = lgb.Dataset(X_vl, label=y_vl.values,
                            categorical_feature=cats or "auto", free_raw_data=False,
                            reference=ds_tr)

        cb = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=9999)]
        model = lgb.train(params, ds_tr, num_boost_round=500,
                          valid_sets=[ds_vl], callbacks=cb)

        preds = model.predict(X_vl)
        if preds.ndim == 2:  # multiclass
            preds = preds.argmax(axis=1)
        oof.iloc[vl_idx_clean] = preds
        logger.info("fold %d done: vl=%d", fold, len(vl_idx_clean))

    return oof


# ─────────────────────────────────────────────────────────────────────────────
# 評価
# ─────────────────────────────────────────────────────────────────────────────

def eval_classification(y_true, y_pred, task: str = "binary") -> dict[str, float]:
    """task: 'binary' or 'multiclass'"""
    metrics: dict[str, float] = {}
    valid = pd.Series(y_true).notna() & pd.Series(y_pred).notna()
    yt = np.array(y_true)[valid]
    yp = np.array(y_pred)[valid]

    if task == "binary":
        # AUC
        if len(np.unique(yt)) == 2:
            metrics["auc"] = float(roc_auc_score(yt, yp))
        # accuracy at 0.5
        metrics["acc_0.5"] = float(accuracy_score(yt, (yp >= 0.5).astype(int)))
    else:
        metrics["acc"] = float(accuracy_score(yt, yp))
        metrics["f1_macro"] = float(f1_score(yt, yp, average="macro", zero_division=0))
    return metrics


def eval_regression(y_true, y_pred) -> dict[str, float]:
    valid = pd.Series(y_true).notna() & pd.Series(y_pred).notna()
    yt = np.array(y_true)[valid]
    yp = np.array(y_pred)[valid]
    return {
        "mae":  float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "corr": float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 1 else np.nan,
    }


# ─────────────────────────────────────────────────────────────────────────────
# OOF 保存・読み込み
# ─────────────────────────────────────────────────────────────────────────────

def save_oof(
    df: pd.DataFrame,
    pred_col: str,
    model_name: str,
    key_cols: Optional[list[str]] = None,
) -> Path:
    """OOF 予測を parquet に保存して返す。"""
    keys = key_cols or ["race_id", "horse_number"]
    out = df[keys + [pred_col]].copy()
    p = OOF_DIR / f"{model_name}_oof.parquet"
    out.to_parquet(p, index=False)
    logger.info("OOF saved: %s  shape=%s", p, out.shape)
    return p


def load_oof(model_name: str) -> pd.DataFrame:
    """OOF 予測を読み込む。"""
    p = OOF_DIR / f"{model_name}_oof.parquet"
    if not p.exists():
        raise FileNotFoundError(f"OOF file not found: {p}")
    return pd.read_parquet(p)


# ─────────────────────────────────────────────────────────────────────────────
# ポストプロセス（T-1 出力 → T-2/T-3 導出）
# ─────────────────────────────────────────────────────────────────────────────

def softmax_normalize(scores: np.ndarray) -> np.ndarray:
    """数値安定な softmax。"""
    e = np.exp(scores - scores.max())
    return e / e.sum()


def harville_place(win_probs: np.ndarray) -> np.ndarray:
    """Harville 式による連対率（2着以内確率）。"""
    p = np.array(win_probs, dtype=float)
    n = len(p)
    place = np.zeros(n)
    for i in range(n):
        place[i] = p[i]  # 1着
        for j in range(n):
            if j != i:
                denom = 1 - p[j] + 1e-9
                place[i] += p[j] * (p[i] / denom)
    return np.minimum(place, 1.0)


def harville_show(win_probs: np.ndarray) -> np.ndarray:
    """Harville 式による複勝率（3着以内確率）。"""
    p = np.array(win_probs, dtype=float)
    n = len(p)
    show = np.zeros(n)
    for i in range(n):
        s = p[i]  # 1着
        for j in range(n):
            if j == i:
                continue
            pj_given_i = p[j] / (1 - p[i] + 1e-9)
            for k in range(n):
                if k == i or k == j:
                    continue
                s += p[j] * pj_given_i * (p[i] / (1 - p[i] - p[j] + 1e-9))
        show[i] = s
    return np.minimum(show, 1.0)


def apply_harville_per_race(
    df: pd.DataFrame,
    win_prob_col: str,
    race_id_col: str = "race_id",
) -> pd.DataFrame:
    """レースごとに Harville 式を適用して place_prob / show_prob を付与。"""
    df = df.copy()
    df["place_prob"] = np.nan
    df["show_prob"]  = np.nan

    for race_id, grp in df.groupby(race_id_col):
        idx  = grp.index
        raw  = grp[win_prob_col].fillna(0).values
        norm = softmax_normalize(raw)
        df.loc[idx, win_prob_col]  = norm
        df.loc[idx, "place_prob"]  = harville_place(norm)
        df.loc[idx, "show_prob"]   = harville_show(norm)

    return df
