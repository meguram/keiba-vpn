"""重賞のルールベース予想家: レース名の鍵・別名の表・コースの鍵、集計の as-of。"""

import numpy as np
import pandas as pd

from src.research.graded import race_keys as RK
from src.research.graded import tally as TL


def test_norm_and_aliases():
    assert RK.canonical_key("第66回毎日放送賞スワンステークス", 10) == "スワンS"
    assert RK.canonical_key("第10回 MBS賞スワンS", 10) == "スワンS"
    assert RK.canonical_key("第64回産経大阪杯", 4) == "大阪杯"
    assert RK.canonical_key("第100回報知弥生ディープ記念", 3) == "弥生賞"
    # 府中牝馬S は 10 月の回だけアイルランドT に結ぶ（2025 年からは 6 月の別のレース）
    assert RK.canonical_key("第64回府中牝馬ステークス", 10) == "アイルランドT"
    assert RK.canonical_key("第73回府中牝馬ステークス", 6) == "府中牝馬S"
    assert RK.canonical_key("第40回ジャパンカップ", 11) == "ジャパンC"


def test_alias_targets_are_canonical():
    al, mal = RK._aliases()
    for src, dst in al.items():
        assert RK.norm_name(dst) == dst, dst
        assert dst not in al, f"{dst} がさらに別名になっている"
    for m in mal:
        assert m["to"] not in al


def test_resolve_io_ambiguous_courses():
    vc = pd.Series(["08", "08", "08", "04", "06"])
    surf = pd.Series(["芝"] * 5)
    dist = pd.Series([1600, 1400, 1600, 2000, 1800])
    io = pd.Series([None, None, None, None, "内"], dtype=object)
    ct = pd.Series(["外", "", None, None, None], dtype=object)
    cls = pd.Series([8, 3, 2, 6, 3])
    out = RK.resolve_io(vc, surf, dist, io, ct, cls).tolist()
    assert out == ["外", "内外不明", "内外不明", "外", "内"]
    # OP 以上で course_type が空（内回り）と出るときだけ内
    out = RK.resolve_io(vc[:1], surf[:1], dist[:1], io[:1], pd.Series([""], dtype=object), pd.Series([5])).tolist()
    assert out == ["内"]


def _toy_runs():
    """同じ重賞（同じコース）の 3 回分と、同じコースの別の OP、重賞でない日。"""
    rows = []
    for y, rid in ((2018, "A18"), (2019, "A19"), (2020, "A20")):
        for h in range(4):
            rows.append({"race_id": rid, "dt": pd.Timestamp(f"{y}-05-01"), "rkey": "Aカップ", "course": "東京芝1600", "grade": "G2", "cls": 7,
                         "fin": h + 1, "field": 4})
    for h in range(4):
        rows.append({"race_id": "B19", "dt": pd.Timestamp("2019-06-01"), "rkey": "B特別", "course": "東京芝1600", "grade": "OP", "cls": 5,
                     "fin": h + 1, "field": 4})
    return pd.DataFrame(rows)


def test_tally_counts_only_before_the_day():
    d = _toy_runs()
    lv = pd.Series(["先行", "先行", "後方", "後方"] * 4, index=d.index)
    tm = (d["race_id"] == "A20")
    out = TL.tally_type(TL.base_frame(d), lv, tm.to_numpy())
    first = out.iloc[0]       # 2020 年の 1 着（先行）: 主 = 2018・2019 の先行 4 走（1・2 着 ×2）
    assert first["main_n"] == 4 and first["main_w1"] == 2 and first["main_w2"] == 2 and first["main_ed"] == 2
    assert first["sub_n"] == 2 and first["sub_w1"] == 1        # 副 = 同じコースの別レース（B特別）
    assert first["glob_n"] == 4                                  # 全体 = 重賞だけ（2018・2019 の A）
    # 未来の行を足しても、過去の値は変わらない
    x = d[d["race_id"] == "A20"].copy()
    x["race_id"] = "A21"
    x["dt"] = [pd.Timestamp("2021-05-01")] * len(x)
    d2 = pd.concat([d, x], ignore_index=True)
    lv2 = pd.Series(list(lv) + ["先行", "先行", "後方", "後方"], index=d2.index)
    out2 = TL.tally_type(TL.base_frame(d2), lv2, (d2["race_id"] == "A20").to_numpy())
    pd.testing.assert_frame_equal(out.reset_index(drop=True), out2.reset_index(drop=True))


def test_tally_freeze_stops_at_2024():
    d = _toy_runs()
    d.loc[d["race_id"] == "A20", "dt"] = pd.Timestamp("2025-05-01")
    lv = pd.Series(["先行"] * len(d), index=d.index)
    out = TL.tally_type(TL.base_frame(d), lv, (d["race_id"] == "A20").to_numpy(), freeze=True)
    assert out.iloc[0]["main_n"] == 8 and np.isclose(out.iloc[0]["main_e3"], 8 * 0.75)
