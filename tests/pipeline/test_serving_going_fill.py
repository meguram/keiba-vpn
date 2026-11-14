"""当日の馬場: 出馬表に無い馬場を、同じ日に先に確定したレースで埋める（v3_serving）と、日ごとのキャッシュの馬場の列を
そのレースだけで計算し直す（serving_v4.going_self / _override_going）。"""

import numpy as np
import pandas as pd

from src.pipeline.features import factory_self as fs
from src.pipeline.features import serving_v4 as sv4
from src.pipeline.inference import v3_serving as v3


def _card():
    return pd.DataFrame({"race_id": ["r1", "r1", "r2", "r3", "r4", "r5"], "horse_id": list("abcdef"), "date": "2026-11-15",
                         "venue_code": ["05", "05", "05", "05", "08", "05"], "surface": ["芝", "芝", "ダート", "障", "芝", "芝"],
                         "start_time": ["11:00"] * 6, "track_condition": [None, None, "", None, None, "重"]})


def _history():
    return pd.DataFrame({"race_id": ["p1", "q1", "q2", "q3", "q4", "q5", "q6", "z9"],
                         "date": ["2026-11-14"] + ["2026-11-15"] * 7,
                         "venue_code": ["05", "05", "05", "05", "05", "5", "05", "06"],
                         "surface": ["芝", "芝", "芝", "ダート", "芝", "ダート", "芝", "芝"],
                         "start_time": ["15:00", "08:30", "09:30", "09:00", "10:30", "09:45", None, "08:00"],
                         "track_condition": ["不良", "良", "稍重", "不良", "重", "重", "不良", "良"]})


def test_same_day_done_takes_the_same_day_with_start_time():
    d = v3.same_day_done(_history(), _card())
    assert set(d["race_id"]) == {"q1", "q2", "q3", "q4", "q5", "z9"}  # 前日 p1・発走時刻の無い q6 は使わない（発走の前後は使う側で絞る）
    assert v3.same_day_done(None, _card()) is None


def test_fill_going_ignores_races_after_the_cutoff_and_without_card_time():
    out = v3.fill_going_from_same_day(_card(), v3.same_day_done(_history(), _card()))
    assert out.set_index("horse_id").loc["a", "track_condition"] != "重"            # 10:30 の q4（重）は、11:00 のレース（予測は約 60 分前の 10:00）には使わない
    c = _card().assign(start_time=None)
    assert v3.fill_going_from_same_day(c, v3.same_day_done(_history(), c)).equals(c)   # 出馬表の発走時刻が分からなければ埋めない


def test_fill_going_normalizes_and_handles_surfaces():
    out = v3.fill_going_from_same_day(_card(), v3.same_day_done(_history(), _card()))
    g = out.set_index("horse_id")["track_condition"]
    assert g["a"] == "稍" and g["b"] == "稍"                     # 芝は 09:30 の q2（稍重）が最新 → 出馬表の表記「稍」にそろえる
    assert g["c"] == "重"                                          # ダートは 09:45 の q5（重。場コード "5" も "05" と同じ）が最新
    assert g["d"] == "稍"                                          # 障害は芝の馬場
    assert pd.isna(g["e"])                                         # 京都（08）は確定したレースが無い（朝の 1R と同じ）→ 空のまま
    assert g["f"] == "重"                                          # 既に入っている馬場は変えない


def test_fill_going_uses_only_races_known_60_minutes_before():
    c = _card()
    c["start_time"] = "10:20"                                               # 締め切り 09:20 → 芝の最新は 08:30 の q1（良）。09:30 の q2 は使わない
    out = v3.fill_going_from_same_day(c, v3.same_day_done(_history(), c))
    assert out.set_index("horse_id").loc["a", "track_condition"] == "良"


def test_fill_going_cutoff_is_per_venue():
    c = _card()
    c.loc[c["venue_code"] == "08", "start_time"] = "09:10"                  # 別の場の早いレースに引きずられない
    out = v3.fill_going_from_same_day(c, v3.same_day_done(_history(), c))
    assert out.set_index("horse_id").loc["a", "track_condition"] == "稍"


def test_hurdle_recorded_as_dirt_counts_as_turf():
    h = pd.DataFrame({"race_id": ["j1", "d1"], "date": "2026-11-15", "venue_code": "05", "surface": ["ダート", "ダート"], "distance": [3000, 1400],
                      "start_time": ["09:30", "09:00"], "track_condition": ["良", "重"]})
    c = pd.DataFrame({"race_id": ["r1", "r2"], "horse_id": ["a", "b"], "date": "2026-11-15", "venue_code": "05", "surface": ["ダート", "芝"],
                      "start_time": "11:00", "track_condition": None})
    g = v3.fill_going_from_same_day(c, v3.same_day_done(h, c)).set_index("race_id")["track_condition"]
    assert g["r1"] == "重" and g["r2"] == "良"                              # 3000m のダート（障害）は芝の馬場として扱い、ダートの最新にしない


def test_fill_day_card_actual_uses_each_finished_race_only():
    day = pd.DataFrame({"race_id": ["q1", "q2", "q4", "r9"], "horse_id": list("wxyz"), "venue_code": "05", "track_condition": None})
    done = v3.same_day_done(_history(), _card())
    out = v3.fill_day_card_actual(day, done, _card()).set_index("race_id")["track_condition"]
    assert out["q1"] == "良" and out["q2"] == "稍"                          # 確定したレースには、そのレース自身の実際の馬場
    assert pd.isna(out["q4"]) and pd.isna(out["r9"])                       # 今回より後に発走するレース・未確定のレースは空のまま


def test_fill_going_noop_without_done():
    c = _card()
    assert v3.fill_going_from_same_day(c, None).equals(c)


def _runs():
    rows = []
    for h, seq in {"a": ["良", "稍重", "良"], "b": ["重"], "x": ["良", "良"]}.items():
        for i, g in enumerate(seq):
            rows.append({"race_id": f"{h}{i}", "date": pd.Timestamp("2026-06-01") + pd.Timedelta(days=30 * i), "horse_id": h,
                         "track_condition": g, "finish_position": 1 if i == 0 else 5, "field_size": 10})
    r = pd.DataFrame(rows)
    for c in fs.RUN_COLS:
        if c not in r.columns:
            r[c] = np.nan
    r["surface"], r["venue_code"], r["distance"] = "芝", "05", 1600.0
    return r[fs.RUN_COLS]


class _Hist:
    def __init__(self):
        self.self_runs = _runs()
        self.frozen = None
        self.graded = {}


def _today(going):
    t = pd.DataFrame({"race_id": ["t1", "t1"], "horse_id": ["a", "b"], "date": "2026-11-15", "surface": ["芝", "障"], "venue_code": "05",
                      "distance": 1600, "field_size": 10, "track_condition": [going, going]})
    return t


def test_going_self_matches_full_day_computation():
    g = sv4.V4Groups(_Hist())
    got = g.going_self(_today("稍")).set_index("horse_id")
    card = g.as_result_surface(g.normalize(_today("稍")))                 # 日全体で作るときと同じ前処理・同じ関数で、全馬（x を含む）について作る
    T = sv4._card_rows(card, fs.RUN_COLS)
    T["distance"] = pd.to_numeric(T["distance"], errors="coerce").astype(float)
    allr = pd.concat([_runs(), T], ignore_index=True).sort_values(["horse_id", "date", "race_id"], kind="stable").reset_index(drop=True)
    full = fs.build_groups(allr, only=["change", "experience"])
    ref = pd.concat([f.set_index(["race_id", "horse_id"]) for f in full.values()], axis=1).loc["t1"]
    for c in sv4.GOING_SELF_COLS:
        np.testing.assert_allclose(got[c].astype(float).to_numpy(), ref.loc[got.index, c].astype(float).to_numpy(), equal_nan=True)
    assert got.loc["a", "chg_going"] == 1.0 and got.loc["a", "chg_going_num"] == 1.0          # 前走 良 → 今日 稍（稍重）: 変わった・+1
    assert got.loc["a", "exp_n_going"] == 1.0 and got.loc["b", "exp_first_going"] == 1.0


def test_override_going_picks_owner_variant_and_tolerates_duplicates():
    f = sv4.V4Features.__new__(sv4.V4Features)
    f.groups = sv4.V4Groups(_Hist())
    out = pd.DataFrame({"race_id": ["t1", "t1", "t1", "t2"], "horse_id": ["a", "a", "b", "x"], "o_go_win": [0.5, 0.5, 0.5, 0.5],
                        "o_go_win__going_g": [0.1, 0.1, 0.2, 0.3], "o_go_win__going_h": [0.9, 0.9, 0.8, 0.7], "chg_going_num": np.nan})
    card = pd.concat([_today("良"), _today(None).assign(race_id="t2", horse_id=["x", "x"]).iloc[:1]], ignore_index=True)
    res = f._override_going(out, card)
    assert "o_go_win__going_g" not in res.columns and len(res) == 4               # 馬場ごとの列は落とす。キーが重複しても落ちない
    r = res.drop_duplicates(["race_id", "horse_id"]).set_index("horse_id")
    assert r.loc["a", "o_go_win"] == 0.1 and r.loc["b", "o_go_win"] == 0.2       # 良 → g の値
    assert r.loc["x", "o_go_win"] == 0.5                                          # 馬場が分からないレースはキャッシュのまま
    res2 = f._override_going(out, _today("重"))
    assert res2.drop_duplicates(["race_id", "horse_id"]).set_index("horse_id").loc["a", "o_go_win"] == 0.9   # 良以外 → h の値


def test_override_going_replaces_only_known_races():
    f = sv4.V4Features.__new__(sv4.V4Features)
    f.groups = sv4.V4Groups(_Hist())
    out = pd.DataFrame({"race_id": ["t1", "t1", "t2"], "horse_id": ["a", "b", "x"], "chg_going_num": [np.nan, np.nan, np.nan],
                        "exp_first_going": [1.0, 1.0, 1.0], "o_x": [0.1, 0.2, 0.3]})
    card = pd.concat([_today("稍"), _today(None).assign(race_id="t2", horse_id=["x", "x"]).iloc[:1]], ignore_index=True)
    res = f._override_going(out, card).set_index("horse_id")
    assert res.loc["a", "chg_going_num"] == 1.0 and res.loc["a", "exp_first_going"] == 0.0     # 馬場が分かったレースは置き換える
    assert np.isnan(res.loc["x", "chg_going_num"]) and res.loc["x", "exp_first_going"] == 1.0 # 分からないレースはキャッシュのまま
    assert res["o_x"].tolist() == [0.1, 0.2, 0.3]                                             # ほかの列は変えない


def _conn_hist():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(60):
        d = pd.Timestamp("2026-01-04") + pd.Timedelta(days=7 * (i // 6))
        for j in range(6):
            rows.append({"race_id": f"h{i}", "date": d, "venue_code": "05", "surface": "芝" if i % 2 else "ダート", "distance": 1600,
                         "grade": "", "race_class": "1勝クラス", "track_condition": ["良", "稍重", "重"][i % 3], "start_time": "10:00",
                         "field_size": 6, "round": 1, "finish_position": j + 1, "bracket_number": j + 1, "horse_number": j + 1,
                         "horse_id": f"x{(i * 6 + j) % 40}", "jockey_id": "j", "trainer_id": "t", "jockey_weight": 55, "last_3f": 35.0,
                         "passing_order": f"{j + 1}-{j + 1}", "weight": 480, "sex_age": "牡3",
                         "owner": f"o{rng.integers(4)}", "breeder": f"b{rng.integers(3)}", "birthplace": "日高"})
    return pd.DataFrame(rows)


class _ConnHist:
    def __init__(self):
        self.conn_raw = _conn_hist()
        self.graded = {}
        self.profile = pd.DataFrame({"horse_id": ["x1", "x2", "x3"], "owner": ["o1", "o2", "o1"], "breeder": ["b0", "b1", "b2"],
                                     "birthplace": "日高"})


def test_owner_going_variants_match_owner_with_each_going():
    g = sv4.V4Groups(_ConnHist())
    card = pd.DataFrame({"race_id": ["t1", "t1", "t2"], "horse_id": ["x1", "x2", "x3"], "date": "2026-11-15", "venue_code": "05",
                         "surface": ["芝", "芝", "ダート"], "distance": 1600, "race_class": "1勝クラス", "field_size": 2, "track_condition": None,
                         "horse_number": [1, 2, 1], "start_time": "11:00"})
    var = g.owner_going_variants(card).set_index(["race_id", "horse_id"])
    for v, tc in (("g", "良"), ("h", "重")):
        ref = g.owner(card.assign(track_condition=tc)).set_index(["race_id", "horse_id"])
        cols = [c for c in ref.columns if c.startswith(sv4.OWNER_GOING_PREFIX)]
        assert cols
        for c in cols:
            np.testing.assert_allclose(var[f"{c}{sv4.GOING_VAR_SEP}{v}"].loc[ref.index].to_numpy(), ref[c].to_numpy(), equal_nan=True, rtol=1e-6)
