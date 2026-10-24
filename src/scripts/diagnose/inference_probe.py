"""推論のメモリ・時間を**別プロセス**で測る（モデル学習の分を含めないため）。

モデルを読み込み、ランダムな特徴量（1レース分）を予測し、ステージ別の RSS を JSON で標準出力に出す。
実データが不要なので、学習PCでもVPSでも実行できる。実データでの計測は
``measure_inference_memory`` を使う。

  python -m src.scripts.diagnose.inference_probe --model-dir models/ensemble --rows 18
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def probe(model_dir: str, rows: int = 18, repeat: int = 3) -> dict:
    import numpy as np

    from src.scripts.diagnose.common import current_rss_mb

    out: dict = {"rss_after_python_mb": round(current_rss_mb())}
    import pandas as pd  # noqa: F401

    out["rss_after_pandas_mb"] = round(current_rss_mb())

    from src.pipeline.models.ensemble_predictor import EnsemblePredictor

    t = time.perf_counter()
    predictor = EnsemblePredictor.load(model_dir)
    out["load_sec"] = round(time.perf_counter() - t, 2)
    out["rss_after_load_mb"] = round(current_rss_mb())
    names = list(predictor.feature_names)
    out["n_features"] = len(names)

    rng = np.random.default_rng(0)
    import pandas as pd

    df = pd.DataFrame(rng.standard_normal((rows, len(names))).astype("float32"), columns=names)
    df.insert(0, "horse_name", [f"h{i}" for i in range(rows)])
    df.insert(0, "horse_number", range(1, rows + 1))
    df.insert(0, "horse_id", [f"id{i}" for i in range(rows)])
    times = []
    for _ in range(max(1, repeat)):
        t = time.perf_counter()
        predictor.predict(df)
        times.append(time.perf_counter() - t)
    out["predict_sec_median"] = round(sorted(times)[len(times) // 2], 3)
    out["rss_after_predict_mb"] = round(current_rss_mb())
    try:
        import resource

        out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    except ImportError:  # Windows
        out["peak_rss_mb"] = out["rss_after_predict_mb"]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--rows", type=int, default=18)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args(argv)
    json.dump(probe(args.model_dir, args.rows, args.repeat), sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
