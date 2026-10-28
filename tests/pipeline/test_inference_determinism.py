"""推論の決定性: 同じ入力なら、実行回数・行の並び・一緒に流す他のレース・実行する場所によらず同じ値を返す。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline.inference.v3_serving import V3Serving
from src.pipeline.models import determinism as det
from src.pipeline.models import ensemble as en


def test_inference_device_is_cpu_unless_explicit(monkeypatch):
    monkeypatch.delenv("KEIBA_INFERENCE_DEVICE", raising=False)
    assert det.inference_device() == "cpu"
    assert det.inference_device("cuda") == "cuda"
    monkeypatch.setenv("KEIBA_INFERENCE_DEVICE", "cuda")
    assert det.inference_device() == "cuda"


def test_fix_inference_runtime_resets_seeds_and_threads():
    torch = pytest.importorskip("torch")
    det.fix_inference_runtime()
    a = (np.random.rand(3), torch.rand(3).numpy())
    det.fix_inference_runtime()
    b = (np.random.rand(3), torch.rand(3).numpy())
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert torch.get_num_threads() == 1


@pytest.mark.parametrize("entry", [
    {"method": "weighted_logit", "weights": [0.2, 0.5, 0.3]},
    {"method": "greedy", "weights": [0.2, 0.5, 0.3]},
    {"method": "stack_lr", "coef": [0.4, 0.4, 0.2], "intercept": -0.1},
])
def test_apply_spec_does_not_depend_on_row_count(entry):
    P = np.random.default_rng(3).uniform(0.02, 0.9, size=(97, 3))
    whole = en.apply_spec(entry, P)
    parts = np.concatenate([en.apply_spec(entry, P[i:i + 7]) for i in range(0, len(P), 7)])
    assert np.array_equal(whole, parts)


class _FakeBundle:
    """行の並びに依存する（先頭からの累積和）ふりをして、渡された並びを記録する。"""

    def __init__(self):
        self.seen: list[np.ndarray] = []

    def predict(self, X, race):
        self.seen.append(X["x"].to_numpy().copy())
        v = X["x"].to_numpy(dtype=float)
        return pd.DataFrame({"p_win": v / 100, "p_top2": v / 50, "p_top3": v / 25}, index=X.index)


def test_v3_predict_is_independent_of_row_order():
    v = V3Serving.__new__(V3Serving)
    v.bundle, v.feature_cols = _FakeBundle(), ["x"]
    X = pd.DataFrame({"race_id": ["r2", "r1", "r2", "r1"], "horse_id": ["h3", "h2", "h1", "h4"], "x": [3.0, 2.0, 1.0, 4.0]})
    out1 = v.predict(X)
    out2 = v.predict(X.iloc[[2, 0, 3, 1]].reset_index(drop=True))
    assert list(v.bundle.seen[0]) == list(v.bundle.seen[1])                   # モデルへは (レース, 馬) 順で渡る
    key = ["race_id", "horse_id"]
    assert out1.sort_values(key).reset_index(drop=True).equals(out2.sort_values(key).reset_index(drop=True))
    assert list(out1["horse_id"]) == ["h3", "h2", "h1", "h4"]                  # 戻りは呼び出し側の並びのまま
    assert list(out1["p_win"]) == [0.03, 0.02, 0.01, 0.04]
