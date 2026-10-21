import numpy as np
import pandas as pd

from src.scripts.data import v6_stage as st


def test_scenario_values_class_means():
    T = pd.DataFrame({"r_pace_bal": [-2.0, -1.0, 0.0, 0.2, 2.0, 4.0], "r_pace_bal_cls": [0, 0, 1, 1, 2, 2]})
    sv = st.scenario_values(T)
    assert sv == {"slow": -1.5, "mid": 0.1, "high": 3.0}


def test_mixture_sd_zero_when_one_scenario_certain():
    A = np.array([[1.0, 2.0, 3.0]])
    P = np.array([[0.0, 1.0, 0.0]])
    mix = (P * A).sum(axis=1)
    sd = np.sqrt((P * (A - mix[:, None]) ** 2).sum(axis=1))
    assert mix[0] == 2.0 and sd[0] == 0.0
