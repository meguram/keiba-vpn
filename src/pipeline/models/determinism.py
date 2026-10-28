"""推論の決定性: 同じ入力なら実行する場所（GPU の有無・スレッド数・実行回数）によらず同じ値を返すための固定。

- NN の推論は CPU に固定する（GPU と CPU では float32 の積和の順序が違い、1e-7 程度ずれる。Cloud Run は CPU だけ）。
  GPU で推論したいときだけ ``KEIBA_INFERENCE_DEVICE=cuda`` を明示する。
- torch のスレッド数は 1 に固定する（CPU でもスレッド数で積和の順序が変わり、1e-7 程度ずれる。1 レースは 18 頭までで速度に影響しない）。
- NN の推論は 1 レースずつ流す（まとめるレース数で行列積のブロック分けが変わり、1e-7 程度ずれる。同じレースの値が一緒に流す他のレースに左右されない）。
- 乱数のシードは ``SEED`` に固定する（推論のコードは乱数を使わないが、将来混ざっても毎回同じ列になるように）。
"""

from __future__ import annotations

import os
import random

import numpy as np

SEED = 0
RACE_CHUNK = 1          # NN の推論で一度に流すレース数


def inference_device(device: str | None = None) -> str:
    return device or (os.environ.get("KEIBA_INFERENCE_DEVICE") or "cpu").strip().lower()


def fix_inference_runtime(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
