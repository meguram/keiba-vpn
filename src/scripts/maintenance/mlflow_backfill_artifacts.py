#!/usr/bin/env python3
"""学習パイプライン（tp-05a〜）の MLflow 実験に、再現に必要な成果物（特徴量リスト・実行済みノートブック・OOF・集計 CSV）を足す。

各 notebook の実行時に記録していたのは指標とパラメータだけで、特徴量リスト・実行済みノートブック・OOF 予測は MLflow に残っていなかった。
実験ごとに ``<notebook>-artifacts`` という run を作り、次を保存する（既にあれば作り直さない）:
  features/feature_list.json（採用した特徴量名・ハッシュ・列数）, notebooks/<executed>.ipynb, oof/*.parquet, results/*.csv|json
  python -m src.scripts.maintenance.mlflow_backfill_artifacts
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

WORK = Path("data/local/modeling/training_pipeline")
REL_BASE = ["f3_r", "sr_perf_ewm2", "sr_perf_ewm6", "sr_exp", "c_res_ewm2", "c_cls_ewm2", "a_l3fz_ewm2", "a_rk_ewm2", "a_style", "b_aff_all",
            "pc_adv", "f2_pace_match", "horse_prior_win_rate", "jockey_prior_win_rate"]
STAGE = ["s_time_pred", "s_time_gap", "s_rank_pred"]

PLAN = {  # (実験名, run 名): (実行済みノートブック, 結果ディレクトリ, OOF のパターン)
    ("keiba-tp-model-screen", "tp-05a-artifacts"): ("runs/tp-05a-executed.ipynb", "model_screen", []),
    ("keiba-tp-model-zoo", "tp-05b-artifacts"): ("runs/tp-05b-executed.ipynb", "model_zoo", ["lgb_binary", "lgb_softmax", "cat_binary", "xgb_binary", "linear", "nn_s", "nn", "nn_l"]),
    ("keiba-tp-model-zoo", "tp-05c-artifacts"): ("runs/tp-05c-executed.ipynb", "model_tune_c", ["lgb_binary_hl1", "lgb_binary_hl2", "lgb_binary_hl3", "lgb_binary_min2022", "lgb_binary_r150", "lgb_binary_r500", "lgb_binary_r800"]),
    ("keiba-tp-stage", "tp-04-artifacts"): ("oof_executed_tp04_v2.ipynb", "oof", []),
}


def feature_list() -> dict:
    adopted = json.loads((WORK / "ablation_nospeed/adopted_features.json").read_text(encoding="utf-8"))
    base = list(adopted["b0_features"]) + list(adopted["features"])
    rel = [c + s for c in REL_BASE for s in ("_rz", "_rrank", "_rgap")]
    full = base + STAGE + rel
    return {"base(tp-03 採用)": base, "stage(tp-04 採用)": STAGE, "relative(C1)": rel, "all": full, "n_all": len(full),
            "hash": hashlib.md5("|".join(full).encode()).hexdigest()[:10]}


def main() -> int:
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    client = MlflowClient()
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    fl = feature_list()
    for (exp, run_name), (nb, resdir, oofs) in PLAN.items():
        mlflow.set_experiment(exp)
        e = client.get_experiment_by_name(exp)
        if client.search_runs([e.experiment_id], filter_string=f"tags.mlflow.runName = '{run_name}'", max_results=1):
            print("既にあります:", exp, run_name)
            continue
        with mlflow.start_run(run_name=run_name, tags={"kind": "artifacts", "git_commit": git, "notebook": run_name.replace("-artifacts", "")}):
            mlflow.log_dict(fl, "features/feature_list.json")
            mlflow.log_params({"n_features_all": fl["n_all"], "feature_list_hash": fl["hash"]})
            p = WORK / nb
            if p.exists():
                mlflow.log_artifact(str(p), "notebooks")
            for f in sorted((WORK / resdir).glob("*")):
                if f.is_file() and f.suffix in (".csv", ".json"):
                    mlflow.log_artifact(str(f), "results")
            n = 0
            for name in oofs:
                f = WORK / "oof/model_oof" / f"{name}.parquet"
                if f.exists():
                    mlflow.log_artifact(str(f), "oof")
                    n += 1
            print("記録:", exp, run_name, "OOF", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
