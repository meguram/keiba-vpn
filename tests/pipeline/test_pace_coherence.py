"""tp-53: 位置取り・ペース・想定上がりを結びつける整合層（pace_coherence）のテスト。"""

import pytest

from src.pipeline.inference import pace_coherence as pc

CFG = pc.load_config()

RAW = "2025.06.29\xa0函館 6 2歳新馬 芝1200 1:11.7 良 12頭\xa010番\xa02人 武豊 55.0 2-2\xa0(35.7)\xa0434(0) エスカレイト (1.0)"
RAW2 = "2025.05.10 1東京8 3歳未勝利 ダ1400 1:25.3 稍重 16頭 3番 5人 川田 56.0 5-5 (36.9) 480(+2) ○○ (0.4)"


def test_config_shipped_and_sane():
    assert CFG and CFG["cohort_l3f"] and CFG["cohort_t3f"] and CFG["cohort_l3f_fallback"]
    feats, coef = CFG["l3f"]["features"], CFG["l3f"]["coef"]
    assert coef[feats.index("e")] < 0          # 後ろの馬ほど上がりが速い（実測と同じ向き）
    assert CFG["l3f"]["coef"][feats.index("prior")] > 0


def test_parse_past_run():
    a = pc.parse_past_run(RAW)
    assert a == {"date": "20250629", "venue": "函館", "surface": "芝", "distance": 1200, "going": "良", "l3f": 35.7}
    b = pc.parse_past_run(RAW2)
    assert (b["venue"], b["surface"], b["distance"], b["going"], b["l3f"]) == ("東京", "ダート", 1400, "稍重", 36.9)
    assert pc.parse_past_run("読めない行") is None
    assert pc.parse_past_run(RAW.replace("(35.7)", "(84.0)")) is None      # 上がりが範囲外（30〜50 秒）は捨てる


def test_cohort_keys_and_nan_distance():
    assert pc.cohort_key("東京", "芝", 1600, "良") == "東京|芝|1600|良"
    assert pc.cohort_key("東京", "芝", 1600, "稍重") == "東京|芝|1600|道悪"
    assert pc.cohort_key("東京", "芝", float("nan"), "良") == "東京|芝|0|良"


def test_prior_uses_only_runs_before_race_date():
    cfg = {"cohort_l3f": {pc.cohort_key("函館", "芝", 1200, "良"): 35.0}, "cohort_l3f_fallback": {}}
    past = [{"raw": RAW}]                                               # 2025-06-29 の走（35.7 → 偏差 +0.7）
    assert pc.prior_l3f_dev(past, "20250701", cfg) == pytest.approx(0.7)
    assert pc.prior_l3f_dev(past, "20250629", cfg) is None              # 同日・未来の走は使わない
    assert pc.prior_l3f_dev(past, "20250101", cfg) is None
    assert pc.prior_l3f_dev([{"raw": "読めない"}], "20250701", cfg) is None


def _horses(n=8):
    return [{"horse_number": i + 1, "early_pos_norm": i / (n - 1), "past_races": []} for i in range(n)]


COURSE = {"venue": "新潟", "surface": "芝", "distance": 1600, "track_condition": "良"}
FAST = {"front_runner_count": 3, "stalker_count": 5, "early_pressure": 0.8}
SLOW = {"front_runner_count": 0, "stalker_count": 1, "early_pressure": 0.1}


def test_pace_label_follows_front_pressure():
    fast = pc.coherent_forecast(CFG, _horses(), FAST, COURSE)["pace"]
    slow = pc.coherent_forecast(CFG, _horses(), SLOW, COURSE)["pace"]
    assert fast["z"] < slow["z"]
    assert fast["level"] == "high" and slow["level"] == "slow"


def test_back_horses_are_faster_and_gap_grows_with_pace():
    def gap(pf):
        h = pc.coherent_forecast(CFG, _horses(), pf, COURSE)["horses"]
        return h[1]["dev_sec"] - h[8]["dev_sec"]            # 先頭 - 最後方（正なら先頭が遅い）
    assert gap(FAST) > 0 and gap(SLOW) > 0          # どのペースでも後ろの馬が速い（傾きの上限で、正の傾きに外挿されない）
    assert gap(FAST) > gap(SLOW)                            # ペースが速いほど位置取りの差が大きい（実測と同じ向き）


def test_slope_never_positive_even_for_extreme_slow_pace():
    for z in (-3.0, -1.0, 0.0, 1.0, 3.0):
        for dist in (1000, 1600, 2400):
            for surf in ("芝", "ダート"):
                assert pc.l3f_slope(CFG, z, dist, surf) <= pc.SLOPE_MAX


def test_dev_is_centered_and_ranked():
    out = pc.coherent_forecast(CFG, _horses(), FAST, COURSE)["horses"]
    assert abs(sum(v["dev_sec"] for v in out.values())) < 0.1
    assert sorted(v["rank"] for v in out.values()) == list(range(1, 9))
    assert out[8]["rank"] < out[1]["rank"]                  # 最後方の馬のほうが上がりは速い（順位が小さい）
    assert out[1]["seconds"] is not None and 30 < out[1]["seconds"] < 45


def _resp():
    preds = [{"horse_number": h["horse_number"], "horse_id": f"h{h['horse_number']}", "position_forecast": {"early_pos_norm": h["early_pos_norm"]}} for h in _horses()]
    return {"predictions": preds, "pace_forecast": {"level": "high", **FAST}, "venue": "新潟", "surface": "芝", "distance": 1600, "track_condition": "良"}


def test_attach_is_noop_when_flag_off(monkeypatch):
    monkeypatch.delenv(pc.ENV_FLAG, raising=False)
    r = pc.attach_pace_coherence(_resp(), None, "20261024")
    assert "pace_unified" not in r and "l3f_coherent" not in r["predictions"][0]


def test_attach_adds_fields_when_flag_on(monkeypatch):
    monkeypatch.setenv(pc.ENV_FLAG, "1")
    r = pc.attach_pace_coherence(_resp(), None, "20261024")
    assert r["pace_unified"]["source"] == "pace_coherence" and r["pace_unified"]["label"] in ("ハイ", "ミドル", "スロー")
    assert all("l3f_coherent" in p and p["l3f_coherent"]["seconds"] for p in r["predictions"])


def test_attach_skips_when_pace_unknown(monkeypatch):
    monkeypatch.setenv(pc.ENV_FLAG, "1")
    resp = _resp()
    resp["pace_forecast"] = {"level": "unknown"}
    assert "pace_unified" not in pc.attach_pace_coherence(resp, None, "20261024")
