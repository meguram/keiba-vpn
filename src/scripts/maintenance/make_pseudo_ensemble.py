"""疑似アンサンブルモデルを作って公開する（ワークフロー動作確認用）。

本物の学習は学習PCで行う。これは、ランダムな合成データで LightGBM / XGBoost / CatBoost / MLP
＋ロジスティック回帰（メタ）の小さなモデルを作り、``EnsembleTrainer`` と同じレイアウトで保存して
``model_registry`` のストアへ公開する。推論ワークフロー（取得→検証→予測）を検証する目的で、
予測精度には意味がない。

使い方:
  python -m src.scripts.maintenance.make_pseudo_ensemble --store .pseudo_model_store --version pseudo-v1
  python -m src.scripts.maintenance.make_pseudo_ensemble --n-features 1000 --store gs://bucket/models/ensemble
"""

from __future__ import annotations

import argparse
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np

from src.pipeline.features.pseudo_builder import DEFAULT_N_FEATURES, pseudo_feature_names


def train_pseudo_ensemble(out_dir: str | Path, n_features: int = DEFAULT_N_FEATURES, *, n_rows: int = 1500, seed: int = 0) -> list[str]:
    """``out_dir`` に疑似アンサンブルを保存し、特徴量名を返す。"""
    import lightgbm as lgb
    import pandas as pd
    import xgboost as xgb
    from catboost import CatBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    names = pseudo_feature_names(n_features)
    X = pd.DataFrame(rng.standard_normal((n_rows, n_features)).astype("float32"), columns=names)
    signal = X.iloc[:, 0] + 0.5 * X.iloc[:, min(1, n_features - 1)]
    y = (signal + rng.normal(0, 0.5, n_rows) > 0).astype(int)

    models: dict[str, object] = {}
    preds: dict[str, np.ndarray] = {}

    models["lightgbm"] = lgb.train(
        {"objective": "binary", "verbose": -1, "num_leaves": 15, "seed": seed},
        lgb.Dataset(X, label=y),
        num_boost_round=30,
    )
    preds["lightgbm"] = models["lightgbm"].predict(X)

    models["xgboost"] = xgb.train(
        {"objective": "binary:logistic", "max_depth": 3, "seed": seed},
        xgb.DMatrix(X, label=y),
        num_boost_round=30,
    )
    preds["xgboost"] = models["xgboost"].predict(xgb.DMatrix(X))

    cat = CatBoostClassifier(iterations=30, depth=4, verbose=0, random_seed=seed)
    cat.fit(X, y)
    models["catboost"] = cat
    preds["catboost"] = cat.predict_proba(X)[:, 1]

    scaler = StandardScaler().fit(X)
    mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=60, random_state=seed)
    mlp.fit(scaler.transform(X), y)
    mlp._scaler = scaler
    models["mlp"] = mlp
    preds["mlp"] = mlp.predict_proba(scaler.transform(X))[:, 1]

    learners = list(models)
    meta_X = np.column_stack([preds[n] for n in learners])
    meta = LogisticRegression().fit(meta_X, y)

    for name, model in models.items():
        with open(out / f"{name}_model.pkl", "wb") as f:
            pickle.dump(model, f)
    with open(out / "meta_model.pkl", "wb") as f:
        pickle.dump(meta, f)
    (out / "ensemble_meta.json").write_text(
        json.dumps({"base_learners": learners, "feature_names": names, "n_folds": 0, "pseudo": True}, indent=2),
        encoding="utf-8",
    )
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default=".pseudo_model_store", help="公開先（パス または gs://bucket/prefix）")
    parser.add_argument("--version", default="pseudo-v1")
    parser.add_argument("--n-features", type=int, default=DEFAULT_N_FEATURES)
    args = parser.parse_args(argv)

    from src.pipeline.models.model_registry import open_store, publish_model

    with tempfile.TemporaryDirectory() as tmp:
        names = train_pseudo_ensemble(tmp, args.n_features)
        total = sum(p.stat().st_size for p in Path(tmp).iterdir())
        manifest = publish_model(tmp, args.version, open_store(args.store), feature_names=names, metrics={"pseudo": True})
    print(f"公開: version={args.version} store={args.store} 特徴量={args.n_features} 合計サイズ={total / 1024:.0f}KB")
    print(f"ファイル: {', '.join(manifest['files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
