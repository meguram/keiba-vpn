import numpy as np
import pytest

from src.pipeline.inference import growth_attach as ga
from src.pipeline.inference import growth_index as gi

PARAMS = {
    "level_mean": 40.0, "rest_median": 34.0, "thresholds": [-2.0, -0.8, 0.9, 2.2], "reason_min_points": 0.3,
    "level_table": [{"level": i, "label": gi.LABELS[i], "share": 0.2, "n": 100, "share_resid_up": 0.4 + 0.05 * i, "mean_resid": float(i - 2), "share_up": 0.5, "mean_y_now": 0.0} for i in range(5)],
    "version": "gi-test", "unit": "pt",
}


class FakeBooster:
    """feature_vector と同じ長さ + 1（バイアス）の SHAP を返す。oikiri の最初の列に +2、weight の最初の列に +1 を置く。"""

    def predict(self, v, pred_contrib=False):
        n = v.shape[1]
        out = np.zeros((1, n + 1))
        sl = gi.group_slices()
        out[0, sl["oikiri"].start] = 2.0
        out[0, sl["weight"].start] = 1.0
        out[0, -1] = 0.1
        return out


def _runs(*vals, low=(), start=(2026, 6, 1)):
    from datetime import date, timedelta

    d0 = date(*start)
    return [{"date": (d0 + timedelta(days=30 * i)).strftime("%Y-%m-%d"), "value": v, "low": i in low} for i, v in enumerate(vals)]


def test_level_uses_valid_runs_with_decay_and_skips_stale_previous_run():
    runs = _runs(40.0, 50.0, 60.0)                       # 6/1, 7/1, 7/31
    lv = ga.level_inputs(runs, [r["date"] for r in runs], "2026-10-01")
    assert lv["stale_prev"] == 0 and lv["rest_days"] == 62 and lv["n_valid_prior"] == 3
    w = [1.0, 0.7, 0.49]
    assert lv["level"] == pytest.approx((60 * w[0] + 50 * w[1] + 40 * w[2]) / sum(w))
    st = ga.level_inputs(runs, [r["date"] for r in runs], "2026-08-05")          # 前走(7/31)から 5 日 → 前走を除く
    assert st["stale_prev"] == 1 and st["level"] == pytest.approx((50 * 1.0 + 40 * 0.7) / 1.7)


def test_level_needs_two_valid_runs_and_ignores_low_confidence():
    assert ga.level_inputs(_runs(50.0), ["2026-06-01"], "2026-10-01") is None
    runs = _runs(40.0, 99.0, 60.0, low=(1,))
    lv = ga.level_inputs(runs, [r["date"] for r in runs], "2026-10-01")
    assert lv["n_valid_prior"] == 2 and lv["level"] == pytest.approx((60 + 40 * 0.7) / 1.7)


def test_weight_inputs_before_and_after_announcement():
    assert ga.weight_inputs(None, None, [460]) == {"weight_change": None, "weight_dev": None}
    r = ga.weight_inputs(470, 2, [460, 450, "x", 0])
    assert r["weight_change"] == 10 and r["weight_dev"] == pytest.approx(470 - 455)
    assert ga.weight_inputs(470, 4, [])["weight_change"] == 4                      # 前走の体重が無ければ発表の増減


def _x(weight=470):
    runs = _runs(48.0, 52.0, 55.0)
    return ga.build_x(
        horse_id="2024104426", sex_age="牡2", weight_today=weight, announced_change=2, past_weights=[460, 456], index_runs=runs,
        history_dates=[r["date"] for r in runs], race_date="2026-10-17", distance=1600, surface="芝",
        tl_row={"f_t4z_vs_self": 0.5, "has_training": 1.0}, extras_row={"oikiri_eval": 3.0, "comment_pos": 1.0},
        sire_profile=None, dam_sire_profile=None,
    )


def test_build_x_stage_age_and_oikiri_sources():
    x = _x()
    assert x["stage"] == "2歳後半" and x["surface"] == "芝" and x["sex"] == "牡"
    assert x["f_t4z_vs_self"] == 0.5 and x["oikiri_eval"] == 3.0 and x["comment_pos"] == 1.0 and x["comment_neg"] is None
    assert x["weight_change"] == 10 and x["has_training"] == 1.0


def test_compute_growth_with_fake_booster_and_provisional_drops_weight():
    g = ga.compute_growth(_x(), PARAMS, FakeBooster())
    assert g["provisional"] is False and g["contributions"]["oikiri"] == 2.0 and g["contributions"]["weight"] == 1.0
    assert g["index"] == pytest.approx(3.1) and g["stage"] == 4 and g["label"] == "大きく成長"
    p = ga.compute_growth(_x(weight=None), PARAMS, FakeBooster())
    assert p["provisional"] is True and p["contributions"]["weight"] == 0.0 and p["index"] == pytest.approx(2.1)
    assert p["stage"] == 3 and all("体重" not in t and "kg" not in t for t in p["reasons"])


def test_build_x_returns_none_without_enough_history():
    assert ga.build_x(horse_id="2024104426", sex_age="牡2", weight_today=470, announced_change=0, past_weights=[], index_runs=_runs(50.0), history_dates=[],
                      race_date="2026-10-17", distance=1600, surface="芝", tl_row=None, extras_row=None, sire_profile=None, dam_sire_profile=None) is None


def test_meta_has_level_table_and_attach_is_off_by_default(monkeypatch):
    m = ga.meta_from(PARAMS)
    assert [r["label"] for r in m["level_table"]] == gi.LABELS and "AI の予測ではありません" in m["note"]
    monkeypatch.delenv(ga.ENV_FLAG, raising=False)
    resp = {"predictions": [{"horse_id": "2024104426"}]}
    assert ga.attach_growth(resp, "2026-10-17") is resp and "growth" not in resp["predictions"][0]
    monkeypatch.setenv(ga.ENV_FLAG, "1")
    monkeypatch.setattr(ga, "load_assets", lambda *_: None)                           # 設定が読めなければ何も付けず落ちない
    assert ga.attach_growth(resp, "2026-10-17") is resp and "growth" not in resp["predictions"][0]


def test_missing_training_record_is_not_scored_as_no_improvement():
    """追い切りの記録が無い馬は、寄与を 0 にして評価に含めず、注記を返す（『上積みが見えない』と読ませない）。"""
    x = _x()
    x["f_t4z_vs_self"] = None
    g = ga.compute_growth(x, PARAMS, FakeBooster())
    assert g["contributions"]["oikiri"] == 0.0 and g["coverage"]["oikiri"] is False
    assert g["index"] == pytest.approx(1.1) and g["notes"] and "追い切り" in g["notes"][0]
    assert all("追い切り" not in t for t in g["reasons"])
    assert ga.compute_growth(_x(), PARAMS, FakeBooster())["notes"] == []
