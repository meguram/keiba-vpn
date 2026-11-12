"""ラップ予想の推論（tp55_build.build_infer）: 対象日の出馬表が無いときは空で返す（並走の Job が KeyError で落ちていた）。"""

import pandas as pd

from src.scripts.data import tp55_build as tb


def test_build_infer_returns_empty_when_no_shutuba_rows(monkeypatch):
    res = pd.DataFrame({"surface": pd.Series(dtype=str), "passing_order": pd.Series(dtype=str), "field_size": pd.Series(dtype=float)})
    monkeypatch.setattr(tb, "load_inputs", lambda years, _x: (res, pd.DataFrame(), pd.DataFrame()))
    cols = ["race_id", "date", "venue", "surface", "distance", "direction", "grade", "race_class", "weather",
            "track_condition", "field_size", "round", "horse_number", "horse_id", "sex_age", "jockey_weight", "jockey_id"]
    monkeypatch.setattr(tb, "_read", lambda y, name, c: pd.DataFrame({"date": ["2026-11-08"], "surface": ["芝"],
                                                                      **{k: [None] for k in cols if k not in ("date", "surface")}}))
    df, rep = tb.build_infer([2026], ["2026-11-12"])
    assert df.empty
    assert rep["skipped"] == "no_shutuba_rows"
