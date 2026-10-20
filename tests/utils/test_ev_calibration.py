import pytest

from src.utils.ev_calibration import calibrated_ev


def test_disabled_or_missing_k_returns_input():
    assert calibrated_ev(1.5, "win", {}) == 1.5
    assert calibrated_ev(1.5, "win", {"enabled": False, "k": {"win": 0.2}}) == 1.5
    assert calibrated_ev(1.5, "win", {"enabled": True, "k": {}}) == 1.5
    assert calibrated_ev(None, "win", {"enabled": True, "k": {"win": 0.2}}) is None


def test_shrinks_toward_the_payout_rate():
    cfg = {"enabled": True, "k": {"win": 0.2, "umaren": 0.1}, "base": {"win": 0.8}}
    assert calibrated_ev(1.5, "win", cfg) == pytest.approx(0.8 + 0.2 * 0.7)
    assert calibrated_ev(0.8, "win", cfg) == pytest.approx(0.8)             # 損益分岐は動かない
    assert calibrated_ev(2.0, "umaren", cfg) == pytest.approx(0.775 + 0.1 * (2.0 - 0.775))      # base の既定は払戻率（馬連 0.775）
