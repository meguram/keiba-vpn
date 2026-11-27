"""aptitude-group の部品の確かめ（手元のデータを読まない小さな表だけ）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.aptitude_group import clogit, common, vector


def _toy(n_days: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        for h in range(8):
            rows.append({"date": pd.Timestamp("2018-01-01") + pd.Timedelta(days=d), "horse_id": f"h{d}_{h}",
                         "sire": f"s{h % 3}", "surf": "T" if d % 2 else "D", "fpct": rng.random()})
    return pd.DataFrame(rows)


def test_dev_is_asof_future_rows_do_not_change_past():
    F = _toy(30)
    A = (F["date"].dt.day % 3 == 0).to_numpy()
    g = -F["fpct"].to_numpy()
    b = vector.base(F, A, ["surf"], g)
    d1, n1 = vector.dev(F, A, ["sire", "surf"], g, b, level=False)
    G = pd.concat([F, _toy(40, seed=1).iloc[len(F):]], ignore_index=True)       # 後ろの日だけ足す
    AG = (G["date"].dt.day % 3 == 0).to_numpy()
    gG = -G["fpct"].to_numpy()
    bG = vector.base(G, AG, ["surf"], gG)
    d2, n2 = vector.dev(G, AG, ["sire", "surf"], gG, bG, level=False)
    assert np.allclose(d1, d2[: len(F)]) and np.allclose(n1, n2[: len(F)])


def test_first_day_has_no_history():
    F = _toy(5)
    g = -F["fpct"].to_numpy()
    A = np.ones(len(F), bool)
    d, n = vector.dev(F, A, ["sire"], g, vector.base(F, A, [], g), level=True)
    first = F["date"] == F["date"].min()
    assert (n[first] == 0).all() and np.allclose(d[first], 0)


def test_clogit_recovers_coefficient():
    rng = np.random.default_rng(3)
    S, J, beta = 4000, 6, np.array([0.8, -0.5])
    X = rng.normal(size=(S * J, 2))
    sid = np.repeat(np.arange(S), J)
    u = X @ beta
    p = np.exp(u).reshape(S, J)
    p /= p.sum(1, keepdims=True)
    pick = np.array([rng.choice(J, p=pp) for pp in p])
    ch = np.zeros(S * J, bool)
    ch[np.arange(S) * J + pick] = True
    b, Hinv, _ = clogit.fit(X, ch, sid)
    se = np.sqrt(np.diag(Hinv))
    assert np.all(np.abs(b - beta) < 4 * se)
    cse = clogit.cluster_se(X, ch, sid, b, Hinv, np.arange(S) // 10)
    assert np.all(cse > 0)


def test_sample_stays_inside_each_set():
    rng = np.random.default_rng(4)
    sid = np.repeat(np.arange(50), 5)
    X = rng.normal(size=(250, 1))
    idx = clogit.sample(X, np.array([1.0]), sid, rng)
    assert len(idx) == 50 and np.all(sid[idx] == np.arange(50))


def test_split_is_fixed_and_about_70():
    m = pd.Series([f"2018{i:04d}" for i in range(1000)])
    a, b = common.split(m), common.split(m.sample(frac=1, random_state=1))
    assert abs((a == 0).mean() - 0.70) < 0.01
    assert dict(zip(m, a)) == dict(zip(m.sample(frac=1, random_state=1), b))


def _toy_races(n_r: int = 6, n_h: int = 7, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_r):
        fp = rng.permutation(n_h) + 1
        for h in range(n_h):
            rows.append({"race_id": f"2018010101{r:02d}", "meeting": f"201801{r // 2:02d}", "date": pd.Timestamp("2018-01-06"),
                         "finish_position": int(fp[h]), "sire": f"s{h % 3}", "ss": f"t{h % 2}", "style": rng.random(),
                         "lq": rng.normal(), "prev_fpct": rng.random(), "s_q5": h % 5, "s": rng.normal(), "popularity": h + 1,
                         "novec": 0.0, "a_pace_h": rng.normal(), "a_pace_s": rng.normal(), "a_wet": rng.normal(), "a_front": rng.normal()})
    return pd.DataFrame(rows)


def test_stages_one_chosen_per_set_and_order():
    from src.research.aptitude_group import check
    V = _toy_races()
    RA = check.Races(V, check.SHOCK_COLS)
    races = np.array([0, 2, 3, 5])
    sid, X, ch, hj = RA.stages(RA.pos[1], RA.pos[2], RA.pos[3], races, check.PAIR_ALL)
    starts = np.r_[0, np.flatnonzero(np.diff(sid)) + 1]
    assert np.array_equal(np.unique(sid), np.arange(2 * len(races)))
    assert np.all(np.add.reduceat(ch.astype(int), starts) == 1)
    st2 = sid % 2 == 0
    assert np.array_equal(hj[st2][ch[st2]], RA.pos[2][races])            # 2 着の段の選ばれた馬 = 本物の 2 着（races の順）
    assert np.array_equal(hj[~st2][ch[~st2]], RA.pos[3][races])
    assert not np.isin(RA.pos[1][races], hj).any()                       # 1 着は選択肢に入らない


def test_pair_string_parse():
    import json
    assert [int(min(json.loads(s))) for s in ["[1, 12]", "[3, 4]"]] == [1, 3]


def test_line_labels_use_ancestors_and_merge(tmp_path, monkeypatch):
    import json
    from src.research.aptitude_group import lines
    chain = {}
    # 9 頭は「B系」の下の「A系」（自分の札は揺れて X系）、9 頭は直に「B系」
    for i in range(9):
        chain[f"a{i}"] = {"chain": [[f"a{i}", "", "X系"], ["p", "", "A系"], ["q", "", "B系"]]}
        chain[f"b{i}"] = {"chain": [[f"b{i}", "", "A系"], ["q", "", "B系"]]}
    p = tmp_path / "origin.json"
    p.write_text(json.dumps(chain), encoding="utf-8")
    monkeypatch.setattr(lines, "ORIGIN", p)
    lab = lines.line_labels(set(chain))
    assert all(v == "B系" for v in lab.values())          # 自分の札（X系・A系）は使わず、A系は直の 9 頭を持つ B系へまとまる
    assert lines.short("Mr. Prospector系") == "ミスプロ"


def test_contrast_is_condition_minus_outside():
    from src.research.aptitude_group import sire_vectors
    rng = np.random.default_rng(7)
    n = 4000
    F = pd.DataFrame({"sire": np.where(np.arange(n) % 2 == 0, "s1", "s2"), "surf": np.where(rng.random(n) < 0.5, "T", "D"),
                      "venue": rng.choice(["東京", "中山", "京都"], n), "distance": 1600, "going": rng.choice(["G", "W"], n),
                      "course_type": "", "direction": "右", "early": rng.random(n), "ts": rng.random(n) - 0.5,
                      "lx": np.log(1600), "acc_z": rng.normal(size=n), "gain_r": rng.normal(size=n)})
    F["gc"] = rng.normal(0, 0.1, n) + np.where((F["sire"] == "s1") & (F["venue"] == "京都"), 0.2, 0.0)
    S = sire_vectors.stats(F, "sire")
    v = S[("d", "kyoto")] / S[("n", "kyoto")]
    # 全体での同じ差（Δ0 ≈ 0.1）を引くので、s1 ≈ +0.1・s2 ≈ −0.1。差は 0.2
    assert v["s1"] > 0.05 > -0.05 > v["s2"] and abs((v["s1"] - v["s2"]) - 0.2) < 0.03
    # 条件を多く走る鍵でも薄まらない: s1 の京都の割合を変えても、ずれは同じくらい
    G = F[~((F["sire"] == "s1") & (F["venue"] != "京都") & (np.arange(n) % 4 == 0))]
    v2 = sire_vectors.stats(G.reset_index(drop=True), "sire")
    assert abs(v2[("d", "kyoto")]["s1"] / v2[("n", "kyoto")]["s1"] - v["s1"]) < 0.03


def test_stats_subset_with_full_base_matches_full():
    """一つ抜きの集計: 鍵の行だけで集計し直しても、全体の基準を渡せば全体の集計のその鍵の値と同じ（critic 5 回目）。"""
    from src.research.aptitude_group import sire_vectors
    rng = np.random.default_rng(11)
    n = 3000
    F = pd.DataFrame({"sire": rng.choice(["s1", "s2", "s3"], n), "surf": np.where(rng.random(n) < 0.5, "T", "D"),
                      "venue": rng.choice(["東京", "中山", "京都", "札幌"], n), "distance": rng.choice([1200, 1600, 2400], n), "going": rng.choice(["G", "W"], n),
                      "course_type": "", "direction": rng.choice(["右", "左"], n), "early": rng.random(n), "ts": rng.random(n) - 0.5,
                      "acc_z": rng.normal(size=n), "gain_r": rng.normal(size=n), "gc": rng.normal(0, 0.3, n)})
    F["lx"] = np.log(F["distance"])
    full = sire_vectors.stats(F, "sire")
    sub = F[F["sire"] == "s2"].reset_index(drop=True)
    part = sire_vectors.stats(sub, "sire", base=sire_vectors.stats_base(F))
    assert np.allclose(full.loc["s2"].to_numpy(float), part.loc["s2"].to_numpy(float))
    bad = sire_vectors.stats(sub, "sire")                       # 基準を渡さないと、芝ダの差は 0 になる（直す前の誤り）
    assert abs(bad.loc["s2", ("d", "surf")]) < 1e-9 and abs(full.loc["s2", ("d", "surf")]) > 1e-6


def test_period_guard_refuses_2025():
    import pytest
    with pytest.raises(AssertionError):
        common.open_token("select", "x", "y", period=("2016-01-01", "2025-01-01"))
