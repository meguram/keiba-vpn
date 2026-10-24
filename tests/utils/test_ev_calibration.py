import json

import numpy as np
import pytest

from src.utils import ev_calibration as ec
from src.utils.ev_calibration import apply_spec, calibrate_portfolio, calibrated_ev, tier_thresholds, value_levels


def test_disabled_or_missing_returns_input():
    assert calibrated_ev(1.5, "win", {}) == 1.5
    assert calibrated_ev(1.5, "win", {"enabled": False, "k": {"win": 0.2}}) == 1.5
    assert calibrated_ev(1.5, "win", {"enabled": True, "k": {}}) == 1.5
    assert calibrated_ev(None, "win", {"enabled": True, "k": {"win": 0.2}}) is None


def test_legacy_shrink_format_still_works():
    cfg = {"enabled": True, "k": {"win": 0.2, "umaren": 0.1}, "base": {"win": 0.8}}
    assert calibrated_ev(1.5, "win", cfg) == pytest.approx(0.8 + 0.2 * 0.7)
    assert calibrated_ev(0.8, "win", cfg) == pytest.approx(0.8)
    assert calibrated_ev(2.0, "umaren", cfg) == pytest.approx(0.775 + 0.1 * (2.0 - 0.775))


def test_apply_spec_methods():
    assert apply_spec({"method": "linear", "a": 0.5, "b": 0.2}, 2.0) == pytest.approx(0.9)
    assert apply_spec({"method": "const", "value": 0.93}, 5.0) == pytest.approx(0.93)
    assert apply_spec({"method": "market", "base": 0.775}, 9.0) == pytest.approx(0.775)
    iso = {"method": "isotonic", "x": [0.5, 1.0, 2.0], "y": [0.6, 0.8, 0.9]}
    assert apply_spec(iso, 0.1) == pytest.approx(0.6) and apply_spec(iso, 9.0) == pytest.approx(0.9)       # 端は外挿しない
    assert apply_spec(iso, 1.5) == pytest.approx(0.85)
    v = apply_spec(iso, np.array([0.5, 0.75, 1.0, 1.5, 2.0]))
    assert np.all(np.diff(v) >= 0)                                                                          # 単調
    with pytest.raises(ValueError):
        apply_spec({"method": "x"}, 1.0)


def test_isotonic_calibration_is_capped_and_never_negative():
    cfg = {"enabled": True, "kinds": {"win": {"method": "linear", "a": -0.5, "b": 1.0}}, "cap": {"ev": 1.3}}
    assert calibrated_ev(0.1, "win", cfg) == 0.0
    assert calibrated_ev(5.0, "win", cfg) == 1.3


def test_unvalidated_candidate_kinds_fall_back_to_market_level():
    cfg = {"enabled": True, "kinds": {"cand_tansho": {"method": "const", "value": 0.9}}}
    assert calibrated_ev(3.0, "tansho", cfg) == pytest.approx(0.9)
    assert calibrated_ev(3.0, "sanrenpuku", cfg) == pytest.approx(0.75)          # 検証していない券種は払戻率（市場並み）
    assert calibrated_ev(3.0, "wide", cfg) == pytest.approx(0.775)               # 設定に無い候補の券種も同様


def test_calibrate_portfolio_is_stake_weighted_and_capped():
    cfg = {"enabled": True, "kinds": {"cand_tansho": {"method": "const", "value": 0.9}, "cand_wide": {"method": "const", "value": 1.2}}, "cap": {"ev": 1.3, "roi": 1.15}}
    out = calibrate_portfolio([{"bet_type": "tansho", "ev": 2.0, "bet_amount": 300}, {"bet_type": "wide", "ev": 2.0, "bet_amount": 100}], cfg)
    assert out["evs"] == [pytest.approx(0.9), pytest.approx(1.2)]
    assert out["expected_roi"] == pytest.approx((0.9 * 300 + 1.2 * 100) / 400, abs=1e-3) and out["expected_return"] == pytest.approx(0.975 * 400)
    capped = calibrate_portfolio([{"bet_type": "wide", "ev": 2.0, "bet_amount": 100}], {**cfg, "kinds": {"cand_wide": {"method": "const", "value": 1.3}}})
    assert capped["expected_roi"] == 1.15 and capped["capped"] is True
    assert calibrate_portfolio([{"bet_type": "wide", "ev": 2.0, "bet_amount": 100}], {"enabled": False}) is None
    assert calibrate_portfolio([], cfg)["expected_roi"] is None


def test_thresholds_follow_config():
    assert tier_thresholds({"enabled": False, "tiers": {"strong": 9}}) == ec.LEGACY_TIERS
    assert tier_thresholds({"enabled": True, "tiers": {"strong": 1.1, "recommend": 1.0}})["strong"] == 1.1
    assert value_levels({"enabled": False}) == ec.LEGACY_LEVELS
    assert dict(value_levels({"enabled": True, "levels": {"hot": 1.03, "good": 1.0, "ok": 0.97}})) == {"hot": 1.03, "good": 1.0, "ok": 0.97}


def test_shipped_config_is_consistent():
    cfg = json.loads(ec.CONFIG_PATH.read_text(encoding="utf-8"))
    assert cfg["version"] == 2 and isinstance(cfg["enabled"], bool)
    for kind in ("win", "top2", "place", "cand_tansho", "cand_fukusho", "cand_umaren", "cand_wide"):
        spec = cfg["kinds"][kind]
        assert np.isfinite(apply_spec(spec, 1.5))
        if spec["method"] == "isotonic":
            assert len(spec["x"]) == len(spec["y"]) >= 2 and np.all(np.diff(spec["x"]) > 0) and np.all(np.diff(spec["y"]) >= 0)
    t = cfg["tiers"]
    assert t["strong"] > t["recommend"] > t["watch"]
    lv = cfg["levels"]
    assert lv["hot"] > lv["good"] > lv["ok"]
    assert cfg["range"]["lo"] < cfg["range"]["hi"]
    # 馬ごとの較正は、EV が高いほど値が下がらず（単調）、極端に高い EV を過大に見せない
    for kind in ("win", "top2", "place"):
        grid = np.linspace(0.1, 6.0, 60)
        v = apply_spec(cfg["kinds"][kind], grid)
        assert np.all(np.diff(v) >= -1e-9) and v.max() <= 1.35
