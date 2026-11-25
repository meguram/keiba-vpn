"""pedigree_lab の as-of の集計が「その日より前・自分を引く」になっているかを、総当たりと比べる。"""
import numpy as np
import pandas as pd

from src.research.pedigree_lab.data import asof, split_meetings


def _brute(df, keys, vals):
    S = np.zeros((len(df), len(vals)))
    N = np.zeros(len(df))
    for i, r in df.iterrows():
        if any(str(r[k]) in ("", "nan") for k in keys):
            continue
        m = (df["date"] < r["date"]) & (df["horse_id"] != r["horse_id"])
        for k in keys:
            m &= df[k].astype(str) == str(r[k])
        S[i] = df.loc[m, vals].sum().to_numpy()
        N[i] = m.sum()
    return S, N


def test_asof_matches_bruteforce():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "date": pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 40, n), unit="D"),
        "horse_id": rng.integers(0, 40, n).astype(str),
        "sire": rng.choice(["a", "b", "c", ""], n),
        "surf": rng.choice(["芝", "ダ"], n),
        "win": rng.integers(0, 2, n),
        "x": rng.normal(size=n),
    })
    df = df.drop_duplicates(["date", "horse_id"]).reset_index(drop=True)   # 1 日 1 走
    for keys in (["sire"], ["sire", "surf"]):
        s, c = asof(df, keys, ["win", "x"])
        bs, bc = _brute(df, keys, ["win", "x"])
        assert np.allclose(c, bc)
        assert np.allclose(s, bs)


def test_split_meetings_is_by_meeting():
    m = pd.Series([f"2020{i:04d}" for i in range(200) for _ in range(12)])
    lab = split_meetings(m)
    assert (pd.DataFrame({"m": m, "l": lab}).groupby("m")["l"].nunique() == 1).all()
    assert set(lab.unique()) == {0, 1, 2}
