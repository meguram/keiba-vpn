"""ts_scale_compare の尺度計算の検算（合成データ）。cur が horse_ts.course_ts と一致し、外れ値でコースの SD が膨らまないこと。"""
import numpy as np
import pandas as pd

from src.research.race import horse_ts as ht
from src.scripts.data import ts_scale_compare as sc


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for venue, dist in (("京都", 1600), ("中山", 1600), ("阪神", 1800)):
        n = 800
        x = rng.normal(11.5, 0.4, n)
        if venue == "京都":
            x[0] = 29.7          # 歩く走（外れ値）
        for i in range(n):
            rows.append({"race_id": f"{venue}{i // 16}", "horse_id": f"{venue}{i}", "venue": venue, "surface": "芝", "distance": dist,
                         "peak": float(x[i]), "offset": 0.0, "finish_order": i % 16 + 1, "nent": 16})
    h = ht.course_ts(pd.DataFrame(rows))
    return h.reset_index(drop=True)


def test_cur_matches_course_ts() -> None:
    h = _frame()
    ts = sc.scale_ts(h, "cur", np.ones(len(h), bool))
    assert float((ts - h["ts"]).abs().max()) < 1e-9


def test_winsor_and_exclude_keep_sd_near_10() -> None:
    h = _frame()
    fit = np.ones(len(h), bool)
    kyoto = (h["venue"] == "京都").to_numpy()
    sd_cur = sc.scale_ts(h, "cur", fit)[kyoto & (h["peak"] < 20).to_numpy()].std()
    assert sd_cur < 7                  # 外れ値 1 走で SD が膨らみ、ts が圧縮される
    for mode in ("ex20", "winabs", "winz", "robwinz"):
        sd = sc.scale_ts(h, mode, fit)[kyoto & (h["peak"] < 20).to_numpy()].std()
        assert 9 < sd < 11, (mode, sd)


def _two_courses(seed: int = 1) -> pd.DataFrame:
    """同じ芝・同じ距離帯（1600〜1999m）に、大きいコース（平均 11.3・3000 走）と小さいコース（平均 11.9・250 走）。"""
    rng = np.random.default_rng(seed)
    rows = []
    for venue, dist, n, mu in (("京都", 1600, 3000, 11.3), ("中山", 1800, 250, 11.9)):
        x = rng.normal(mu, 0.4, n)
        for i in range(n):
            rows.append({"race_id": f"{venue}{i // 16}", "horse_id": f"{venue}{i}", "venue": venue, "surface": "芝", "distance": dist,
                         "peak": float(x[i]), "offset": 0.0, "finish_order": i % 16 + 1, "nent": 16})
    return ht.course_ts(pd.DataFrame(rows)).reset_index(drop=True)


def test_fit_subset_uses_only_fit_runs() -> None:
    """当てる走を全走以外にしたとき、mu・sd は fit の走だけで決まり、fit 外の走にも ts が付く。"""
    h = _frame()
    fit = (np.arange(len(h)) % 2 == 0)
    ts = sc.scale_ts(h, "cur", fit)
    ref = ht.course_ts(h[fit].drop(columns=["ts", "mu_s", "sd_s"]))       # fit の走だけで course_ts を作り直す
    assert float((ts[fit].to_numpy() - ref["ts"].to_numpy()).__abs__().max()) < 1e-9
    assert ts[~fit].notna().all()


def test_small_course_clip_is_not_one_sided() -> None:
    """走数 300 未満のコースを帯の絶対値で切ると、帯の平均とずれたコースで片側だけ切れる。z 版は両側とも小さく、skip は切らない。"""
    h = _two_courses()
    fit = np.ones(len(h), bool)
    small = (h["venue"] == "中山").to_numpy()
    x = h["x"].to_numpy()
    lo, hi = sc.winsor_bounds(h, fit, 0.005, "abs")
    assert (x[small] > hi[small]).mean() > 0.04 and (x[small] < lo[small]).mean() == 0.0       # 遅い側ばかり切れる
    lo, hi = sc.winsor_bounds(h, fit, 0.005, "z")
    assert (x[small] > hi[small]).mean() < 0.03 and (x[small] < lo[small]).mean() < 0.03
    lo, hi = sc.winsor_bounds(h, fit, 0.005, "skip")
    assert np.isnan(lo[small]).all() and np.isnan(hi[small]).all()
    assert np.isfinite(lo[~small]).all()                                                     # 走数の多いコースは切る
    rates = sc.clip_rates(h, "winskip", fit, min_n=50)
    assert rates["n>=300"]["lo_median"] < 1.0


def test_rho_from_r_and_four_runs_drop_nan() -> None:
    assert abs(float(sc.rho_from_r(np.array([0.5]))[0]) - 1 / 3) < 1e-12
    rows = []
    for i in range(sc.MIN_HORSES + 10):
        for k in range(4):
            rows.append({"horse_id": f"h{i}", "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=30 * (3 - k)), "ts_a": float(k)})
    g = pd.DataFrame(rows)
    g.loc[g.index[:1], "ts_a"] = np.nan           # 先頭の馬は NaN の走があり、4 走に満たなくなる → 馬ごと除かれる
    arr = sc.four_runs(g, ["ts_a"])
    assert arr.shape == (sc.MIN_HORSES + 9, 4, 1)
    assert arr[0, :, 0].tolist() == [3.0, 2.0, 1.0, 0.0]       # 古い順に並ぶ（date が古い走の k が大きい）


def test_bootstrap_uses_same_horses_for_all_variants() -> None:
    """尺度どうしは同じ再標本で比べる。ts_b = 2*ts_a+1（一次変換）なら ρ も再標本の ρ も一致し、差の区間は [0, 0]。"""
    rng = np.random.default_rng(3)
    n = 400
    base = rng.normal(size=(n, 1))
    vals = base + rng.normal(scale=1.0, size=(n, 4))
    rows = [{"horse_id": f"h{i}", "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=k), "ts_cur": vals[i, k], "ts_b": 2 * vals[i, k] + 1}
            for i in range(n) for k in range(4)]
    r = sc.reliability(pd.DataFrame(rows), ["cur", "b"], np.random.default_rng(0))
    assert r["b"]["diff_vs_cur"] == 0.0 and r["b"]["diff_ci"] == [0.0, 0.0]
    assert r["b"]["ci"] == r["cur"]["ci"]
