"""重賞の条件の版（src/research/graded/versions.py）のテスト。窓の切り方・古さの判定・リーク（検証と予想の対象の回の結果が、
発見での規則の選び方に入らないこと）・台帳。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.research.graded import cycles as CY
from src.research.graded import versions as VS
from tests.research.test_graded_playbook import NAMES, _table


def test_windows_last_two_are_validation_and_rest_from_2018_is_discovery():
    d = ["2017-05-01", "2019-05-01", "2021-05-01", "2023-05-01", "2025-05-01", "2026-05-01"]
    assert VS.windows(d, "2027-01-01") == (["2019-05-01", "2021-05-01", "2023-05-01"], ["2025-05-01", "2026-05-01"])
    assert VS.windows(d, "2026-05-01") == (["2019-05-01", "2021-05-01"], ["2023-05-01", "2025-05-01"])   # 予想する回（同じ日）は入れない
    assert VS.windows(["2025-05-01"], "2026-01-01") == ([], ["2025-05-01"])                           # 発見が無い → 版を作らない


def test_is_stale_only_when_a_new_edition_result_is_after_the_window():
    v = {"validation": {"to": "2026-11-15"}}
    assert not VS.is_stale(v, ["2025-11-16"], "2026-11-22")
    assert VS.is_stale(v, ["2025-11-16", "2026-11-20"], "2027-11-21")
    assert not VS.is_stale(v, ["2026-11-22"], "2026-11-22")       # 予想する回そのものは、まだ結果が無い扱い
    assert VS.is_stale(None, ["2020-01-01"], "2026-01-01") and not VS.is_stale(None, [], "2026-01-01")


def _shuffle_outcomes(t, mask_races):
    r2 = t.r.copy()
    rng = np.random.default_rng(3)
    cols = ["fin", "y1", "y3", "ret_tan", "ret_fuku"]
    for i in np.flatnonzero(mask_races):
        idx = r2.index[t.race_of == i]
        r2.loc[idx, cols] = r2.loc[idx, cols].to_numpy()[rng.permutation(len(idx))]
    return CY.Table(r2, t.F.copy(), NAMES)


def test_build_uses_only_discovery_results(monkeypatch):
    """検証の回（と予想する回）の着順・払戻を入れ替えても、選ばれる規則は同じ（発見の回だけで選ぶ）。検証の成績は変わりうる。"""
    t = _table()
    before = "2027-01-01"
    v1 = VS.build(t, ["R", "S <x>"], before, cycles=6)
    dt, rk, has = VS._race_cols(t)
    val = np.zeros(len(rk), bool)
    for k in ("R", "S <x>"):
        _, vv = VS.windows(sorted(set(dt[rk == k])), before)
        val |= (rk == k) & np.isin(dt, vv)
    v2 = VS.build(_shuffle_outcomes(t, val), ["R", "S <x>"], before, cycles=6)
    for k in ("R", "S <x>"):
        assert v1[k]["honmei"] == v2[k]["honmei"] and v1[k]["kill"] == v2[k]["kill"] and v1[k]["sha"] == v2[k]["sha"]
        assert v1[k]["validation"]["n"] == 2 and v1[k]["discovery"]["from"] >= "2018"
    assert "N" not in VS.build(t, ["N"], before, cycles=2)        # 発見の回が無い重賞は版を作らない


def test_build_ignores_rows_on_or_after_before():
    """before 以降の回（予想する回・その先の回）の結果を変えても、窓も規則も同じ。"""
    t = _table()
    before = "2026-01-01"
    v1 = VS.build(t, ["R"], before, cycles=4)
    dt, rk, _ = VS._race_cols(t)
    v2 = VS.build(_shuffle_outcomes(t, dt >= before), ["R"], before, cycles=4)
    assert v1["R"]["sha"] == v2["R"]["sha"] and v1["R"]["val_stats"] == v2["R"]["val_stats"]


def test_ledger_and_current(tmp_path, monkeypatch):
    monkeypatch.setenv("GRADED_RULES_VERSIONS", str(tmp_path / "v.json"))
    P = {"frozen_at": "2026-11-15T16:19:26+09:00", "sha256_rules": "s",
         "rules": {"B": {"honmei_by_rkey": {"R": {"terms": {"x_A01": 1}, "k": None}}, "kill_by_rkey": {"R": {"terms": {"x_A01": 1}, "k": 3}}}}}
    v = VS.current("R", P)
    assert v["id"] == VS.V0_ID and v["validation"]["to"] == VS.V0_DATA_END   # 台帳が空なら 20 サイクルの B（検証の終わりは表の結果の終わり）
    assert VS.current("Z", P) is None
    assert VS.current("R", P, date="2024-06-01") is None                      # 最初の版は 2025 年以降の回だけ（規則が 2024 年までで選ばれた）
    VS.append_version("R", {"id": "x", "before": "2027-05-01", "sha": "a", "validation": {"to": "2026-05-01"}})
    VS.append_version("R", {"id": "x", "before": "2027-05-01", "sha": "a", "validation": {"to": "2026-05-01"}})   # 同じ版は足さない
    assert len(VS.load_ledger()["rkeys"]["R"]) == 1
    assert VS.current("R", P)["id"] == "x"
    # 予想する回の日より後に作った版（未来の結果で選んだ版）は使わない
    assert VS.current("R", P, date="2026-11-22")["id"] == VS.V0_ID
    assert VS.current("R", P, date="2027-05-01")["id"] == "x"


def test_material_warning():
    assert VS.material_cutoff() == "2024-12-31"                               # prereg.json の確認の期間の終わり
    assert VS.material_warning("2026-11-22") is None
    assert "2025 年までで決め直す" in VS.material_warning("2027-02-01")
    assert VS.material_warning("2027-02-01", cutoff="2025-12-31") is None     # 決め直した後は警告しない


def test_build_never_picks_day_of_rules():
    from src.research.graded.picks import needs_dayof
    v = VS.build(_table(), ["R", "S <x>"], "2027-01-01", cycles=6)
    for x in v.values():
        for kd in ("honmei", "kill"):
            if x[kd]:
                assert not needs_dayof(x[kd]["terms"])   # score_day を使う決め方も外す


def test_build_skips_entry_stage_races():
    """出馬表の段階の回（着順がすべて空）は、発見にも検証にも入れない。"""
    t = _table()
    r2 = t.r.copy()
    last = r2["race_id"] == "R2026"
    r2.loc[last, "fin"] = np.nan
    t2 = CY.Table(r2, t.F.copy(), NAMES)
    v = VS.build(t2, ["R"], "2027-01-01", cycles=4)["R"]
    assert v["validation"]["to"] < "2026" and v["validation"]["n"] == 2       # 2026 年の回（結果なし）は窓に入らない


def test_best_defaults_unchanged():
    """Best の既定（dv・ev・rkeys を渡さない）は、渡したときと同じ（20 サイクルの作りを変えない）。"""
    t = _table()
    yr = t.race["year"].to_numpy()
    a = CY.Best(t)
    b = CY.Best(t, dv=(yr >= CY.DEV[0]) & (yr <= CY.DEV[1]), ev=(yr >= CY.EVAL[0]) & (yr <= CY.EVAL[1]),
                rkeys=sorted(set(t.race["rkey"].to_numpy()[yr >= CY.EVAL[0]])))
    cand = [({"score_pre": 1}, "p"), ({"x_A01": 1}, "a")]
    for B in (a, b):
        W, lab = CY.matrix(t, cand)
        B.update(W, lab)
    assert json.dumps(a.report(), sort_keys=True, default=str) == json.dumps(b.report(), sort_keys=True, default=str)
