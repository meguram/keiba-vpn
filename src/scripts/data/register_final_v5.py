"""アンサンブル v5 の最終モデル（fit_final_v5 の出力）を MLflow に記録して、Registry に登録する（追加のみ。既存のモデルは触らない）。

python -m src.scripts.data.register_final_v5 --fit-end 2025
- 実験 keiba-final-models に run を作り、bundle のディレクトリを artifact に、学習期間・特徴量数・seed・tp-06c の test 2025 の指標を param/metric/tag に記録する。
- Registry の名前: keiba-ensemble-v5（バージョンが追加される）。推論は、この artifact（bundle）を EnsembleBundle.load で読む。
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import mlflow

from src.config.ports import mlflow_tracking_uri
from src.pipeline.mlflow.train_utils import require_tracking_server

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "data/local/modeling/training_pipeline"
# 注意: MLflow への登録（Registry への追加）は、ユーザーへ報告して承認を得てから実行すること。
MODEL_NAME = "keiba-ensemble-v5"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-end", type=int, choices=[2024, 2025], default=2025)
    ap.add_argument("--run-id", default="", help="既に artifact を記録済みの run を使う（記録をやり直さずに登録だけ行う）")
    a = ap.parse_args()
    bundle = WORK / f"final_v5/bundle_v5_{a.fit_end}"
    info = json.loads((bundle / "fit_info.json").read_text(encoding="utf-8"))
    rep = json.loads((WORK / "final_eval_v5/report.json").read_text(encoding="utf-8")) if (WORK / "final_eval_v5/report.json").exists() else {}
    parity = bundle / "parity.json"
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    require_tracking_server(mlflow_tracking_uri())
    mlflow.set_tracking_uri(mlflow_tracking_uri())
    mlflow.set_experiment("keiba-final-models")
    client = mlflow.tracking.MlflowClient()
    if a.run_id:
        run_id = a.run_id
    else:
        with mlflow.start_run(run_name=f"ensemble-v5-fit{a.fit_end}", tags={"model": "ensemble_v5", "git_commit": commit, "fit_end": str(a.fit_end), "notebook": "tp-07-v5"}) as run:
            mlflow.log_params({"fit_end": a.fit_end, "train_rows": info["train_rows"], "seeds": info["seeds"], "n_features": info["n_features"], "members": ",".join(m["name"] for m in json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))["members"])})
            mlflow.log_metrics({f"test2025_logloss_{k}": float(v) for k, v in (rep.get("logloss") or {}).items()})
            mlflow.log_metrics({f"test2025_vs_v4_{k}": float(v) for k, v in (rep.get("vs_v4") or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
            if parity.exists():
                mlflow.log_dict(json.loads(parity.read_text(encoding="utf-8")), "parity_vs_v4.json")
            mlflow.log_artifacts(str(bundle), artifact_path="bundle")
            run_id = run.info.run_id
    uri = f"{client.get_run(run_id).info.artifact_uri}/bundle"
    try:
        client.create_registered_model(MODEL_NAME, description="アンサンブル v5（V4 + tl_ 再取得版 + 採用された追加群）。artifact の bundle を EnsembleBundle.load で読む")
    except mlflow.exceptions.MlflowException:
        pass                                                             # 既にある
    mv = client.create_model_version(MODEL_NAME, source=uri, run_id=run_id, tags={"fit_end": str(a.fit_end), "git_commit": commit})
    print(json.dumps({"run_id": run_id, "model": MODEL_NAME, "version": mv.version, "uri": uri}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
