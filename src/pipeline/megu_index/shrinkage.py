"""小サンプル推定向けの経験的ベイズ縮約。"""

from __future__ import annotations

import numpy as np


def shrink_scalar(
    estimate: float | None,
    prior: float,
    n: int,
    strength: float = 30.0,
) -> tuple[float, float]:
    """
    estimate を prior へ縮約する。

    Returns:
        (shrunk_value, cell_weight)  cell_weight = n / (n + strength)
    """
    if n <= 0 or estimate is None or (isinstance(estimate, float) and np.isnan(estimate)):
        return prior, 0.0
    w = n / (n + strength)
    return float(w * estimate + (1.0 - w) * prior), w
