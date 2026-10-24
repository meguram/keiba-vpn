import numpy as np
import pandas as pd

from src.pipeline.features import pedigree_context as pc


def _df(rows):
    d = pd.DataFrame(rows)
    d["date"] = pd.to_datetime(d["date"])
    return d.reset_index(drop=True)


def test_uses_only_strictly_earlier_races_and_shrinks_to_global_mean():
    base = dict(sire="S", dam_sire="D", surface="芝", distance=1600, track_condition="良", venue_code="05", race_id="r", horse_id="h")
    rows = []
    for i, (dt, fin) in enumerate([("2021-01-01", 1), ("2021-01-02", 5), ("2021-01-03", 1), ("2021-01-03", 2), ("2021-01-04", 1)]):
        rows.append({**base, "date": dt, "finish_position": fin, "race_id": f"r{i}", "horse_id": f"h{i}"})
    df = _df(rows)
    labels = pd.DataFrame({"race_id": df["race_id"], "horse_id": df["horse_id"], "pos_early": 0.3, "l3f": 35.0})
    f = pc.build(df, labels, k=0.0)
    n = f["ped_sire_sd_n"].tolist()
    assert n == [0.0, 1.0, 2.0, 2.0, 4.0]                   # 同日の行は数えない
    assert np.isnan(f["ped_sire_sd_win"].iloc[0])           # 最初の日は全体平均も無い
    assert abs(f["ped_sire_sd_win"].iloc[4] - 0.5) < 1e-6   # それ以前の 4 行は 勝ち 2/4（1着, 5着, 1着, 2着 → 勝ち 2）


def test_future_results_do_not_change_past_features():
    base = dict(sire="S", dam_sire="D", surface="芝", distance=1600, track_condition="良", venue_code="05")
    def mk(last_fin):
        rows = [{**base, "date": f"2021-01-0{i + 1}", "finish_position": 1 if i % 2 == 0 else 4, "race_id": f"r{i}", "horse_id": f"h{i}"} for i in range(5)]
        rows[-1]["finish_position"] = last_fin
        df = _df(rows)
        return pc.build(df, pd.DataFrame({"race_id": df["race_id"], "horse_id": df["horse_id"], "pos_early": 0.5, "l3f": 36.0}))
    a, b = mk(1), mk(9)
    assert np.allclose(a.iloc[:4].to_numpy(), b.iloc[:4].to_numpy(), equal_nan=True)
