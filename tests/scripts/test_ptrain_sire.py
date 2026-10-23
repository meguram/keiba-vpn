import numpy as np
import pandas as pd

from src.scripts.data import ptrain_sire as ps


def _df(n=30):
    d = pd.date_range("2022-01-01", periods=n, freq="7D")
    return pd.DataFrame({"race_id": [f"r{i}" for i in range(n)], "horse_id": [f"h{i}" for i in range(n)], "date": d,
                         "distance": 1200 + 200 * (np.arange(n) % 5), "surface": "芝", "fp": 1 + np.arange(n) % 4,
                         "sire": "S", "dam_sire": "D"})


LIN = pd.DataFrame({"stallion_id": ["S", "D"], "sub_group_label": ["A系", "B系"], "main_group_name": ["X系", "Y系"]})


def test_sire_stats_are_asof_and_lines_added():
    df = _df()
    a = ps.build_frame(df, LIN)
    assert {"P77_S_sire_wd", "P77_S_sline_wd", "P77_S_dline_z", "P77_S_smain_t3gap"} <= set(a.columns)
    df2 = df.copy()
    df2.loc[15:, "distance"] += 1000
    df2.loc[15:, "fp"] = 1
    b = ps.build_frame(df2, LIN)
    cols = [c for c in a.columns if c.startswith("P77_S_sire_") and not c.endswith(("gap", "_z", "t3gap"))]
    assert a.loc[:15, cols].equals(b.loc[:15, cols])       # 未来の結果を変えても、それ以前の行（当日を含む）は変わらない
    assert a.loc[0, "P77_S_sire_wn"] == 0
