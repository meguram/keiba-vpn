from src.pipeline.inference import position_forecast as pf


def _run(date, passing, field=16):
    return {"date": date, "raw": f"{date} 東京 5 3歳未勝利 芝1600 1:34.2 良 {field}頭 7番 6人 騎手 57.0 {passing} (34.2) 462(0) 勝馬 (0.4)"}


def test_parse_run_normalizes_first_corner_like_pace_predictor():
    norm, first, field, passing = pf.parse_run(_run("2026.05.01", "1-1-1-1", 16)["raw"])
    assert (norm, first, field, passing) == (0.0, 1, 16, "1-1-1-1")
    norm, first, *_ = pf.parse_run(_run("2026.05.01", "9-9-5-3", 17)["raw"])
    assert first == 9 and abs(norm - 0.5) < 1e-9
    assert pf.parse_run("2026.05.01 東京 芝1600 良") is None           # 通過順位も頭数も無い
    assert pf.parse_run(_run("2026.05.01", "1-1", 3)["raw"]) is None       # 少頭数は除く


def test_style_thresholds():
    assert [pf.style_of(x) for x in (0.0, 0.15, 0.16, 0.35, 0.36, 0.65, 0.66, 1.0)] == [
        "front", "front", "stalker", "stalker", "mid", "mid", "closer", "closer"]


def test_only_runs_before_the_race_date_are_used_and_recent_runs_weigh_more():
    past = [_run("2026.06.01", "1-1-1-1"), _run("2026.05.01", "16-16-16-16"), _run("2026.07.01", "16-16-16-16")]
    f = pf.horse_forecast(past, "20260610")          # 2026.07.01 は対象レースより後（使わない）
    assert f["runs_used"] == 2 and f["style_code"] == "mid" and f["early_pos_norm"] < 0.5   # 新しい走(逃げ)の重みが大きい
    assert f["last_passing"] == "1-1-1-1" and f["confidence"] == "mid"
    assert pf.horse_forecast([_run("2026.07.01", "1-1")], "20260610") is None
    assert pf.horse_forecast([], "20260610") is None


def _entries(n):
    return [{"horse_number": i, "horse_id": f"h{i}"} for i in range(1, n + 1)]


def _doc(styles):
    first = {"F": "1-1-1-1", "S": "4-4-4-4", "M": "9-9-9-9", "C": "16-16-16-16"}
    return {"entries": [{"horse_number": i + 1, "horse_id": f"h{i + 1}", "past_races": [_run("2026.05.01", first[s])]} for i, s in enumerate(styles)]}


def test_rank_and_pace_levels():
    styles = "FFFSSMMMMCCC" + "M" * 4
    by_no, pace = pf.build_position_forecast(_doc(styles), _entries(16), "20260610")
    assert by_no[1]["early_pos_est"] < by_no[4]["early_pos_est"] < by_no[6]["early_pos_est"] < by_no[10]["early_pos_est"]
    assert pace["level"] == "high" and pace["front_runner_count"] == 3 and pace["label"] == "ハイ"
    _, slow = pf.build_position_forecast(_doc("SSMMMMMMCCCC" + "MMMM"), _entries(16), "20260610")
    assert slow["level"] == "slow" and slow["front_runner_count"] == 0
    _, single = pf.build_position_forecast(_doc("FSMMMMMMCCCC" + "MMMM"), _entries(16), "20260610")
    assert single["level"] == "slow" and "単騎逃げ" in single["note"]
    _, mid = pf.build_position_forecast(_doc("FFSSSMMMMMCC" + "MMMM"), _entries(16), "20260610")
    assert mid["level"] == "high" or mid["level"] == "mid"


def test_unknown_when_most_horses_have_no_history_and_attach_keeps_other_fields():
    resp = {"race_id": "r", "predictions": [{"horse_number": 1, "horse_id": "h1", "win_prob": 0.2}, {"horse_number": 2, "horse_id": "x2"}]}
    out = pf.attach_position_forecast(resp, {"entries": [{"horse_number": 1, "horse_id": "h1", "past_races": [_run("2026.05.01", "1-1-1-1")]}]}, "20260610")
    assert out["predictions"][0]["win_prob"] == 0.2 and out["predictions"][0]["position_forecast"]["style"] == "逃げ"
    assert out["predictions"][1]["position_forecast"] is None
    assert out["pace_forecast"]["level"] == "unknown" and out["pace_forecast"]["known_horses"] == 1
    assert pf.attach_position_forecast({"predictions": []}, None, None) == {"predictions": []}
