"""
5.3 ラボ: ファーストステップ集計 + グラフ + 全特徴群を用いた勝利予測の精度検証

前提（手動または本スクリプトの --run-prereqs）:
  - ``build_maternal_stallion_stage_stats`` → maternal_stallion_stage_stats.parquet
  - ``build_stallion_ancestor_graph`` → stallion_tree/graph.json

評価:
  - 時系列 split（既定: 2020–2023 学習、2024–2025 テスト）
  - 二値ターゲット: 1 着かどうか
  - 指標: ROC-AUC, Brier, log loss（必要なら校正）
  - Ablation: 特徴プレフィックス群を落としたときの比較

注意:
  - 母系集計 Parquet は全期間合算のため **未来情報リーク**が含まれる。本結果は探索用。
  - 厳密な因果評価には日付以前のみで集計した v_s(t) が必要（別バッチ）。

CLI::
    python3 -m src.research.pedigree.pedigree_stage5_3_eval --max-rows 100000 --run-prereqs
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pipeline.features.track_bias_pedigree import stage_key  # noqa: E402
from src.research.pedigree.build_maternal_stallion_stage_stats import (  # noqa: E402
    extract_maternal_stallion_ids,
    full_stage_key,
    load_stallion_id_set,
)
from src.research.pedigree.pedigree_local_store import load_full_pedigree_index  # noqa: E402
from src.research.pedigree.pedigree_stage5_3_features import (  # noqa: E402
    StageStatsStore,
    build_feature_row,
    build_graph_bundle,
    louvain_like_labels,
)
from src.research.pedigree.sire_factor_stats import load_sire_factor_stats  # noqa: E402
from src.utils.keiba_logging import script_basic_config  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_MATERNAL = Path("data/meta/modeling/maternal_stallion_stage_stats.parquet")
DEFAULT_GRAPH = Path("data/meta/modeling/stallion_tree/graph.json")
OUT_JSON = Path("data/meta/modeling/pedigree_stage5_3_eval.json")


def _parse_year(d: Any) -> int | None:
    try:
        return int(pd.Timestamp(d).year)
    except Exception:
        return None


def _row_win(fp: Any) -> int:
    try:
        return 1 if int(float(fp)) == 1 else 0
    except Exception:
        return 0


def _feature_group(col: str) -> str:
    if col.startswith(("a_raw_", "a_eb_", "a_res_", "a_winsor", "a_logit")):
        return "A_table"
    if col.startswith("m_mean_wr_coarse") or col.startswith("m_max_wr_coarse"):
        return "A_coarse"
    if col.startswith(("m_mean_entropy", "m_mean_gini", "m_mean_pca", "m_mean_nmf", "m_mean_svd")):
        return "A_profile"
    if col.startswith(("f_", "b_")):
        return "B_factor"
    if col.startswith(("g_", "c_")):
        return "C_graph"
    if col.startswith("d_"):
        return "D_meta"
    if col.startswith("e_"):
        return "E_cross"
    return "other"


def _aux_r2_pca_to_consistency(store: StageStatsStore, sire_stats: dict[str, Any]) -> float | None:
    """B: 舞台 PCA から consistency 軸を Ridge で説明できるか（R²）。"""
    sires = sire_stats.get("sires") or {}
    rows_x = []
    rows_y = []
    for sid in store._stallion_rows:
        ent = sires.get(sid)
        if not isinstance(ent, dict):
            continue
        m = ent.get("maternal") or {}
        ax = m.get("axes") or {}
        c = ax.get("consistency")
        if c is None:
            continue
        if sid not in store._stallion_pc.index:
            continue
        x = store._stallion_pc.loc[sid].values[: min(4, store._stallion_pc.shape[1])]
        rows_x.append(x)
        rows_y.append(float(c))
    if len(rows_y) < 50:
        return None
    X = np.asarray(rows_x, dtype=np.float64)
    y = np.asarray(rows_y, dtype=np.float64)
    rid = Ridge(alpha=1.0)
    rid.fit(X, y)
    pred = rid.predict(X)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-15
    return float(1.0 - ss_res / ss_tot)


def run_eval(
    *,
    maternal_path: Path,
    graph_path: Path,
    tables_dir: Path,
    train_max_year: int,
    test_min_year: int,
    max_rows: int | None,
    random_state: int,
) -> dict[str, Any]:
    long_df = pd.read_parquet(maternal_path)
    store = StageStatsStore(long_df)
    gb = build_graph_bundle(graph_path)
    sire_stats = load_sire_factor_stats(None)
    comm = louvain_like_labels(gb)

    aux_r2 = _aux_r2_pca_to_consistency(store, sire_stats)

    index = load_full_pedigree_index(None)
    stallion_set = load_stallion_id_set(None)

    paths = sorted(tables_dir.glob("[0-9][0-9][0-9][0-9]/race_result_flat.parquet"))
    frames = []
    for p in paths:
        frames.append(
            pd.read_parquet(
                p,
                columns=[
                    "horse_id",
                    "date",
                    "venue_code",
                    "surface",
                    "distance",
                    "track_condition",
                    "finish_position",
                ],
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df["year"] = df["date"].map(_parse_year)
    df = df.dropna(subset=["year"])
    df["y_win"] = df["finish_position"].map(_row_win)

    horse_ids = df["horse_id"].astype(str).str.strip().unique()
    horse_m: dict[str, frozenset[str]] = {}
    for hid in horse_ids:
        if hid not in index:
            continue
        horse_m[hid] = frozenset(
            extract_maternal_stallion_ids(hid, index, stallion_ids=stallion_set)
        )

    df = df[df["horse_id"].astype(str).str.strip().map(lambda h: len(horse_m.get(h, ())) > 0)]
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(max_rows, random_state=random_state)

    feat_cache: dict[tuple[str, str, str], dict[str, float]] = {}

    def feats_for_row(r: pd.Series) -> dict[str, float]:
        hid = str(r["horse_id"]).strip()
        fsk = full_stage_key(
            r["venue_code"],
            r["surface"],
            r["distance"],
            r["track_condition"],
        )
        sk = stage_key(r["venue_code"], r["surface"], r["distance"])
        k = (hid, fsk, sk)
        if k not in feat_cache:
            feat_cache[k] = build_feature_row(
                store,
                gb,
                sire_stats,
                horse_m[hid],
                fsk,
                sk,
                comm,
            )
        return feat_cache[k]

    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        fd = feats_for_row(r)
        if not fd:
            continue
        fd["y_win"] = int(r["y_win"])
        fd["year"] = int(r["year"])
        fd["horse_id"] = str(r["horse_id"]).strip()
        rows.append(fd)

    feat_df = pd.DataFrame(rows)
    y = feat_df["y_win"].values.astype(np.int32)
    years = feat_df["year"].values.astype(np.int32)
    horses = feat_df["horse_id"].values
    X = feat_df.drop(columns=["y_win", "year", "horse_id"], errors="ignore")

    train_mask = years <= train_max_year
    test_mask = years >= test_min_year
    X_tr, y_tr = X.loc[train_mask].reset_index(drop=True), y[train_mask]
    X_te, y_te = X.loc[test_mask].reset_index(drop=True), y[test_mask]

    def fit_score(
        name: str,
        Xtr: pd.DataFrame,
        ytr: np.ndarray,
        Xev: pd.DataFrame,
        yev: np.ndarray,
    ) -> dict[str, Any]:
        cols = [c for c in Xtr.columns if Xtr[c].std() > 1e-12 or Xev[c].std() > 1e-12]
        if not cols:
            return {"name": name, "error": "no_variation"}
        pipe = Pipeline(
            [
                ("imp", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("sc", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        pipe.fit(Xtr[cols], ytr)
        prob = pipe.predict_proba(Xev[cols])[:, 1]
        out: dict[str, Any] = {
            "name": name,
            "n_features": len(cols),
            "n_train": int(len(ytr)),
            "n_test": int(len(yev)),
            "roc_auc": float(roc_auc_score(yev, prob)),
            "brier": float(brier_score_loss(yev, prob)),
            "log_loss": float(log_loss(yev, prob, labels=[0, 1])),
        }
        try:
            frac_pos, mean_pred = calibration_curve(yev, prob, n_bins=10, strategy="quantile")
            out["calibration_mean_abs_gap"] = float(
                np.mean(np.abs(frac_pos - mean_pred))
            )
        except Exception:
            pass
        return out

    all_cols = list(X.columns)
    baseline = fit_score("full_all_time_split", X_tr, y_tr, X_te, y_te)

    ablations: dict[str, Any] = {}
    groups = sorted(set(_feature_group(c) for c in all_cols))
    for g in groups:
        keep = [c for c in all_cols if _feature_group(c) != g]
        if len(keep) < 5:
            continue
        ablations[f"drop_{g}"] = fit_score(
            f"drop_{g}",
            X_tr[keep],
            y_tr,
            X_te[keep],
            y_te,
        )

    # G: 馬単位グループ split（全期間・同一特徴）
    X_g = X.reset_index(drop=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
    g_tr, g_te = next(gss.split(X_g, y, groups=horses))
    g_fit = fit_score(
        "group_shuffle_horse_all_years",
        X_g.iloc[g_tr],
        y[g_tr],
        X_g.iloc[g_te],
        y[g_te],
    )

    return {
        "meta": {
            "maternal_path": str(maternal_path),
            "graph_path": str(graph_path),
            "train_max_year": train_max_year,
            "test_min_year": test_min_year,
            "max_rows": max_rows,
            "n_rows_input": int(len(df)),
            "n_rows_with_features": int(len(feat_df)),
            "n_feature_cache_keys": len(feat_cache),
            "leakage_note": (
                "maternal aggregate uses all years; metrics are exploratory only. "
                "Additionally, stallion×stage counts include the current race row in the global "
                "aggregate, so ROC-AUC can be optimistically high (target leakage)."
            ),
        },
        "aux": {
            "r2_pca_to_consistency_maternal_axis": aux_r2,
        },
        "baseline_full": baseline,
        "ablations": ablations,
        "group_split_horse": g_fit,
    }


def main() -> None:
    script_basic_config()
    ap = argparse.ArgumentParser(description="5.3 特徴ラボ評価")
    ap.add_argument("--maternal-parquet", type=Path, default=DEFAULT_MATERNAL)
    ap.add_argument("--graph-json", type=Path, default=DEFAULT_GRAPH)
    ap.add_argument("--tables-dir", type=Path, default=Path("data/local/tables"))
    ap.add_argument("--train-max-year", type=int, default=2023)
    ap.add_argument("--test-min-year", type=int, default=2024)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    ap.add_argument(
        "--run-prereqs",
        action="store_true",
        help="maternal 集計 + グラフ生成を subprocess で実行",
    )
    args = ap.parse_args()

    if args.run_prereqs:
        subprocess.run(
            [sys.executable, "-m", "src.research.pedigree.build_maternal_stallion_stage_stats", "--fetch-pedigree"],
            cwd=_ROOT,
            check=False,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.research.pedigree.build_stallion_ancestor_graph",
                "--from-parquet",
                str(DEFAULT_MATERNAL),
                "--out-dir",
                str(DEFAULT_GRAPH.parent),
                "--expand-parents",
            ],
            cwd=_ROOT,
            check=False,
        )

    if not args.maternal_parquet.exists():
        logger.error("母系集計 Parquet がありません: %s", args.maternal_parquet)
        sys.exit(1)

    report = run_eval(
        maternal_path=args.maternal_parquet,
        graph_path=args.graph_json,
        tables_dir=args.tables_dir,
        train_max_year=args.train_max_year,
        test_min_year=args.test_min_year,
        max_rows=args.max_rows,
        random_state=args.random_state,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("保存: %s", args.out)
    print(json.dumps(report.get("baseline_full"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
