"""拡張ラベル（src/research/model_relabel/labels.py）と分け方の検査。"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.research.model_relabel import folds, mr_labels as L

ROOT = Path(__file__).resolve().parents[2]
CONV = L.load_spec()["margin"]["convert_A"]


def _race(rid, rows):
    return pd.DataFrame([{"race_id": rid, "horse_id": f"{rid}_{i}", "finish_position": fp, "margin": m, "time_sec": t}
                         for i, (fp, m, t) in enumerate(rows)])


@pytest.mark.parametrize("s,v", [("ハナ", 0.05), ("アタマ", 0.1), ("クビ", 0.25), ("1/2", 0.5), ("3/4", 0.75), ("1.1/4", 1.25),
                                 ("2.1/2", 2.5), ("3", 3.0), ("大", 99.0), ("同着", 0.0), ("1/2+ハナ", 0.55), ("クビ+1/2", 0.75)])
def test_margin_lengths(s, v):
    assert math.isclose(L.margin_lengths(s, CONV), v)


def test_margin_blank_is_nan():
    assert np.isnan(L.margin_lengths("", CONV)) and np.isnan(L.margin_lengths(None, CONV))


def test_cumsum_win_and_place():
    # 1 着・2 着クビ(0.25)・3 着アタマ(0.35)・4 着ハナ(0.4)・5 着クビ(0.65)・6 着 1/2(1.15)・7 着 大
    d = _race("202001010101", [(1, "", 60.0), (2, "クビ", 60.0), (3, "アタマ", 60.1), (4, "ハナ", 60.1), (5, "クビ", 60.2),
                               (6, "1/2", 60.3), (7, "大", 64.0)])
    out = L.build_labels(d)
    assert out["y1"].tolist() == [True] + [False] * 6
    assert out["y_w"].tolist() == [True, True, True, True, False, False, False]
    # 3 着の累積 0.35 から: 4 着 0.05・5 着 0.3・6 着 0.8
    assert out["y_p"].tolist() == [True, True, True, True, True, False, False]
    assert out["big"].tolist() == [False] * 6 + [True]


def test_big_needs_6th_and_one_second():
    d = _race("202001010102", [(1, "", 60.0), (2, "大", 62.0), (6, "1", 60.9), (7, "1", 61.0)])
    out = L.build_labels(d)
    assert out["big"].tolist() == [False, False, False, True]


def test_dead_heat_winner_and_third():
    d = _race("202001010103", [(1, "", 60.0), (1, "同着", 60.0), (3, "1", 60.2), (3, "同着", 60.2), (5, "1/2", 60.3), (6, "1/2", 60.4)])
    out = L.build_labels(d)
    assert out["y1"].tolist() == [True, True, False, False, False, False]
    assert out["y_w"].tolist() == [True, True, False, False, False, False]
    assert out["y_p"].tolist() == [True, True, True, True, True, False]


def test_not_finished_is_negative_not_big():
    d = _race("202001010104", [(1, "", 60.0), (2, "クビ", 60.0), (-1, "", 0.0)])
    out = L.build_labels(d)
    assert out["y_w"].tolist() == [True, True, False] and out["big"].tolist() == [False, False, False]


def test_bad_race_not_extended():
    # 2 着以下に空の着差 → 拡張しない
    d = _race("202001010105", [(1, "", 60.0), (2, "", 60.0), (3, "ハナ", 60.0)])
    out = L.build_labels(d)
    assert not out["ext_ok"].any()
    assert out["y_w"].tolist() == [True, False, False] and out["y_p"].tolist() == [True, True, True]


def test_demoted_race_not_extended():
    d = _race("202001010106", [(1, "", 60.0), (2, "クビ", 60.0), (3, "ハナ", 60.0)])
    d["finish_position_raw"] = ["1", "2(降)", "3"]
    out = L.build_labels(d)
    assert not out["ext_ok"].any() and out["y_w"].tolist() == [True, False, False]


def test_input_order_kept():
    d = _race("202001010107", [(3, "アタマ", 60.1), (1, "", 60.0), (2, "クビ", 60.0)])
    out = L.build_labels(d)
    assert out["horse_id"].tolist() == d["horse_id"].tolist()
    assert out["y1"].tolist() == [False, True, False] and out["y_w"].tolist() == [True, True, True]


def test_folds_deterministic_and_disjoint():
    ms = [f"2016{v:02d}{k:02d}" for v in range(1, 11) for k in range(1, 6)]
    a, b = folds.make_folds(ms, 20261128), folds.make_folds(ms, 20261128)
    pd.testing.assert_frame_equal(a, b)
    assert sorted(a["fold"].unique()) == [0, 1, 2, 3, 4]
    for j in range(5):
        assert (a.loc[a["fold"] == j, f"calib_{j}"] == 0).all()     # 外側の折の開催は較正に入らない


def test_committed_folds_match_spec():
    csv = ROOT / "docs/operations/model_relabel/meeting_folds.csv"
    f = pd.read_csv(csv, dtype={"meeting": str})
    sp = L.load_spec()["meeting_split"]
    g = folds.make_folds(f["meeting"], int(sp["seed"]), int(sp["k"]), float(sp["inner_calib_frac"]))
    pd.testing.assert_frame_equal(f.reset_index(drop=True), g.reset_index(drop=True), check_dtype=False)
    assert f["meeting"].str.fullmatch(r"20(1[6-9]|2[0-4])\d{4}").all()      # 2016〜24 年の開催だけ


def test_not_imported_from_pipeline():
    dirs = [ROOT / "src/pipeline", ROOT / "src/api"]
    assert all(d.is_dir() for d in dirs)
    r = subprocess.run(["grep", "-rln", "model_relabel", *map(str, dirs)], capture_output=True, text=True)
    assert r.returncode == 1 and r.stdout.strip() == ""


def test_winner_with_margin_not_extended():
    # 降着の主な形（1 着の行に着差が入る）
    d = _race("202001010108", [(1, "クビ", 60.0), (2, "", 60.0), (3, "ハナ", 60.0)])
    out = L.build_labels(d)
    assert not out["ext_ok"].any() and out["y_w"].tolist() == [True, False, False]


def test_time_reversal_not_extended():
    d = _race("202001010109", [(1, "", 60.0), (2, "クビ", 60.0), (3, "ハナ", 59.8)])
    out = L.build_labels(d)
    assert not out["ext_ok"].any()


def test_second_dead_heat_no_third():
    # 1・2・2・4 着: 3 着内の累積の最大（2 着の 0.5）から 4 着はクビ差 → y_p
    d = _race("202001010110", [(1, "", 60.0), (2, "1/2", 60.1), (2, "同着", 60.1), (4, "クビ", 60.1), (5, "1", 60.3)])
    out = L.build_labels(d)
    assert out["y_p"].tolist() == [True, True, True, True, False]


def test_dead_heat_row_order_reversed():
    # 同着の行が相方より前に来ても、累積は同じ
    a = _race("202001010111", [(1, "", 60.0), (2, "1", 60.2), (3, "1/2", 60.3), (3, "同着", 60.3)])
    b = _race("202001010111", [(1, "", 60.0), (2, "1", 60.2), (3, "同着", 60.3), (3, "1/2", 60.3)])
    ca = L.build_labels(a).set_index("finish_position")["cum"]
    cb = L.build_labels(b).set_index("finish_position")["cum"]
    assert sorted(ca.loc[3].tolist()) == sorted(cb.loc[3].tolist()) == [1.5, 1.5]


def test_ext_ok_shared_with_not_finished():
    d = _race("202001010112", [(-1, "", 0.0), (1, "", 60.0), (2, "", 60.0)])
    out = L.build_labels(d)
    assert not out["ext_ok"].any()
    out["field_size"], out["surface"], out["grade"], out["year"] = 3, "芝", "", 2020
    assert L.rate_table(out)["races_no_extend"] == 1


def test_blocked_race_day_window():
    import datetime as dt

    from src.research.model_relabel.mr_train import blocked
    jst = dt.timezone(dt.timedelta(hours=9))
    at = lambda s: dt.datetime.fromisoformat(s).replace(tzinfo=jst)      # noqa: E731
    assert not blocked(at("2026-11-28 07:29")) and blocked(at("2026-11-28 07:30"))
    assert blocked(at("2026-11-28 16:59")) and not blocked(at("2026-11-28 17:00"))
    assert blocked(at("2026-11-30 12:00"))            # 月曜の開催
    assert not blocked(at("2026-12-01 12:00"))        # 平日
    assert blocked(at("2026-12-12 10:00"))            # 土曜


def test_race_normalize_sum_and_cap():
    from src.research.model_relabel.mr_eval import race_normalize
    rid = np.array([1, 1, 1, 1, 2, 2, 2])
    p = np.array([0.9, 0.5, 0.1, 0.05, 0.3, 0.3, 0.3])
    q = race_normalize(p, rid, 3.0)
    assert q.max() <= 1
    assert math.isclose(q[:4].sum(), 3.0, rel_tol=1e-4) and math.isclose(q[4:].sum(), 3.0, rel_tol=1e-3)
    w = race_normalize(p, rid, 1.0)
    assert math.isclose(w[:4].sum(), 1.0, rel_tol=1e-6)


def test_place_one_length_version():
    # 3 着の累積 0.35 から: 4 着 0.05・5 着 0.3・6 着 0.8（1 馬身以内）・7 着 大
    d = _race("202001010113", [(1, "", 60.0), (2, "クビ", 60.0), (3, "アタマ", 60.1), (4, "ハナ", 60.1), (5, "クビ", 60.2),
                               (6, "1/2", 60.3), (7, "大", 64.0)])
    out = L.build_labels(d)
    assert out["y_p"].tolist() == [True] * 5 + [False, False]
    assert out["y_p1"].tolist() == [True] * 6 + [False]


def test_variant_of_names():
    from src.research.model_relabel.mr_eval import variant_of
    assert variant_of("orig|r800") == ("orig", None)
    assert variant_of("w2|r400") == ("ext_w2", None)
    assert variant_of("k0.5|r200") == ("ext_w1", 0.5)
    assert variant_of("x1w4|r600") == ("ext1_w4", None)
    assert variant_of("x1k1.0|r800") == ("ext1_w1", 1.0)


def test_target_and_weight_ext1_uses_y_p1():
    from src.research.model_relabel.mr_train import target_and_weight
    lab = pd.DataFrame({"y1": [1, 0, 0, 0], "y3": [1, 1, 0, 0], "y_w": [1, 1, 0, 0], "y_p": [1, 1, 0, 0],
                        "y_p1": [1, 1, 1, 0], "big": [False, False, False, True]}).astype(bool)
    y, wt = target_and_weight(lab, "top3", "ext1_w4")
    assert y.tolist() == [1, 1, 1, 0] and wt.tolist() == [1, 1, 1, 4]
    y, wt = target_and_weight(lab, "top3", "ext_w2")
    assert y.tolist() == [1, 1, 0, 0] and wt.tolist() == [1, 1, 1, 2]
    with pytest.raises(AssertionError):
        target_and_weight(lab, "win", "ext1_w1")
    y, wt = target_and_weight(lab, "win", "ext_w2_nc")          # _nc はラベル・重みが同じ（行の除き方だけ違う）
    assert y.tolist() == [1, 1, 0, 0] and wt.tolist() == [1, 1, 1, 2]


def test_manifest_compare(tmp_path):
    import json

    from src.research.model_relabel.mr_train import _check_manifest
    old = {"spec_version": "mr-2", "label_md5": "a", "sets": [{"max_depth": 8, "nthread": 4}], "rounds": [200]}
    (tmp_path / "manifest.json").write_text(json.dumps(old))
    # 版・スレッドの数が違い、列の印が新しく付いても通る
    _check_manifest(tmp_path, {"spec_version": "mr-5", "label_md5": "a", "sets": [{"max_depth": 8, "nthread": 2}], "rounds": [200], "cols_md5": "x"})
    with pytest.raises(SystemExit):     # ラベルが違えば止める
        _check_manifest(tmp_path, {"spec_version": "mr-5", "label_md5": "b", "sets": [{"max_depth": 8, "nthread": 2}], "rounds": [200]})
    with pytest.raises(SystemExit):     # 古い印にあって新しい印に無い鍵（除き方など）があれば止める
        old2 = dict(old, chalk_md5="c")
        (tmp_path / "manifest.json").write_text(json.dumps(old2))
        _check_manifest(tmp_path, {"spec_version": "mr-5", "label_md5": "a", "sets": [{"max_depth": 8}], "rounds": [200]})


def test_chalk_races():
    res = pd.DataFrame({"race_id": ["a"] * 4 + ["b"] * 4, "finish_position": [1, 2, 3, 4, 1, 2, 3, -1],
                        "popularity": [2, 1, 3, 10, 1, 2, 5, 4]})
    assert L.chalk_races(res).tolist() == [True] * 4 + [False] * 4
    # 1 着の同着（1・1・3）・取消（-1）は印に関わらない
    r2 = pd.DataFrame({"race_id": ["c"] * 4, "finish_position": [1, 1, 3, -1], "popularity": [1, 2, 3, 1]})
    assert L.chalk_races(r2).tolist() == [True] * 4
    # 3 着の同着で 4 頭: 全員が 3 番人気以内でないと付かない
    r3 = pd.DataFrame({"race_id": ["d"] * 4, "finish_position": [1, 2, 3, 3], "popularity": [1, 2, 3, 4]})
    assert not L.chalk_races(r3).any()
    # 3 着内の人気が欠けたら付けない
    r4 = pd.DataFrame({"race_id": ["e"] * 3, "finish_position": [1, 2, 3], "popularity": [1, 2, np.nan]})
    assert not L.chalk_races(r4).any()


def test_variant_of_nc():
    from src.research.model_relabel.mr_eval import variant_of
    assert variant_of("ncorig|r800") == ("orig_nc", None)
    assert variant_of("ncw4|r200") == ("ext_w4_nc", None)
