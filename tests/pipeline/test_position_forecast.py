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
    by_no, pace, _ = pf.build_position_forecast(_doc(styles), _entries(16), "20260610")
    assert by_no[1]["early_pos_est"] < by_no[4]["early_pos_est"] < by_no[6]["early_pos_est"] < by_no[10]["early_pos_est"]
    assert pace["level"] == "high" and pace["front_runner_count"] == 3 and pace["label"] == "ハイ"
    _, slow, _ = pf.build_position_forecast(_doc("SSMMMMMMCCCC" + "MMMM"), _entries(16), "20260610")
    assert slow["level"] == "slow" and slow["front_runner_count"] == 0
    _, single, _ = pf.build_position_forecast(_doc("FSMMMMMMCCCC" + "MMMM"), _entries(16), "20260610")
    assert single["level"] == "slow" and "単騎逃げ" in single["note"]
    _, mid, _ = pf.build_position_forecast(_doc("FFSSSMMMMMCC" + "MMMM"), _entries(16), "20260610")
    assert mid["level"] == "high" or mid["level"] == "mid"


def test_unknown_when_most_horses_have_no_history_and_attach_keeps_other_fields():
    resp = {"race_id": "r", "predictions": [{"horse_number": 1, "horse_id": "h1", "win_prob": 0.2}, {"horse_number": 2, "horse_id": "x2"}]}
    out = pf.attach_position_forecast(resp, {"entries": [{"horse_number": 1, "horse_id": "h1", "past_races": [_run("2026.05.01", "1-1-1-1")]}]}, "20260610")
    assert out["predictions"][0]["win_prob"] == 0.2 and out["predictions"][0]["position_forecast"]["style"] == "逃げ"
    assert out["predictions"][1]["position_forecast"] is None
    assert out["pace_forecast"]["level"] == "unknown" and out["pace_forecast"]["known_horses"] == 1
    assert pf.attach_position_forecast({"predictions": []}, None, None) == {"predictions": []}


# ── 枠順の補正 ────────────────────────────────────────────────

def _all(passing, n):
    return {"entries": [{"horse_number": i + 1, "horse_id": f"h{i + 1}", "past_races": [_run("2026.05.01", passing)]} for i in range(n)]}


def test_gate_rel_edges():
    assert pf.gate_rel(1, 16) == 0.0 and pf.gate_rel(16, 16) == 1.0
    assert abs(pf.gate_rel(8, 15) - 0.5) < 1e-9
    assert pf.gate_rel(None, 16) is None and pf.gate_rel("x", 16) is None
    assert pf.gate_rel(17, 16) is None and pf.gate_rel(0, 16) is None and pf.gate_rel(1, 1) is None


def test_gate_coeff_course_surface_mean_and_unknown():
    inner = pf.gate_coeff("中京", "芝", 1600)
    assert inner["level"] == "inner" and inner["k"] > 0.03 and inner["source"] == "course"
    outer = pf.gate_coeff("新潟", "芝", 1000)
    assert outer["level"] == "outer" and outer["k"] < -0.03
    flat = pf.gate_coeff("東京", "芝", 1600)
    assert flat["level"] == "flat" and abs(flat["k"]) < 0.03
    few = pf.gate_coeff("中京", "芝", 3300)               # 9 レースしかない → 面の平均
    assert few["level"] == "low" and few["source"] == "surface_mean" and abs(few["k"] - 0.0246) < 1e-9
    assert pf.gate_coeff("未知場", "ダート", 1200)["k"] == pf.load_gate_bias()["_meta"]["surface_mean"]["ダート"]
    assert pf.gate_coeff(None, None, None)["level"] == "na" and pf.gate_coeff(None, None, None)["k"] == 0.0


def test_inner_gate_course_moves_inner_horses_forward_and_outer_back():
    course = {"venue": "中京", "surface": "芝", "distance": 1600}
    by_no, pace, gate = pf.build_position_forecast(_all("8-8-8-8", 16), _entries(16), "20260610", course)
    assert gate["level"] == "inner" and gate["applied"] is True and "内枠が前に出やすい" in gate["note"]
    assert by_no[1]["early_pos_norm"] < by_no[16]["early_pos_norm"]
    assert by_no[1]["early_pos_est"] < by_no[16]["early_pos_est"]
    assert by_no[1]["early_pos_raw"] == by_no[16]["early_pos_raw"]       # 補正前は同じ
    assert by_no[1]["gate_shift"] < 0 < by_no[16]["gate_shift"] and by_no[1]["gate_rel"] == 0.0


def test_outer_gate_course_reverses_and_flat_course_keeps_order_by_number():
    out = pf.build_position_forecast(_all("8-8-8-8", 16), _entries(16), "20260610", {"venue": "新潟", "surface": "芝", "distance": 1000})
    assert out[2]["level"] == "outer" and out[0][16]["early_pos_norm"] < out[0][1]["early_pos_norm"]
    flat = pf.build_position_forecast(_all("8-8-8-8", 16), _entries(16), "20260610", {"venue": "東京", "surface": "芝", "distance": 1600})
    assert flat[2]["level"] == "flat" and abs(flat[0][1]["gate_shift"]) < 0.02 and abs(flat[0][16]["gate_shift"]) < 0.02


def test_shift_is_small_and_clipped():
    by_no, _, _ = pf.build_position_forecast(_all("1-1-1-1", 16), _entries(16), "20260610", {"venue": "新潟", "surface": "芝", "distance": 1000})
    assert all(0.0 <= f["early_pos_norm"] <= 1.0 for f in by_no.values())
    assert max(abs(f["gate_shift"]) for f in by_no.values()) <= 0.15 / 2 + 1e-9    # 最大でも k/2


def test_small_field_missing_number_and_no_course_do_not_shift():
    small = pf.build_position_forecast(_all("8-8-8-8", 6), _entries(6), "20260610", {"venue": "中京", "surface": "芝", "distance": 1600})
    assert small[2]["applied"] is False and all(f["gate_shift"] == 0 for f in small[0].values())
    nocourse = pf.build_position_forecast(_all("8-8-8-8", 16), _entries(16), "20260610", None)
    assert nocourse[2]["level"] == "na" and all(f["gate_shift"] == 0 for f in nocourse[0].values())
    ents = _entries(16)
    ents[2]["horse_number"] = None
    no_num = pf.build_position_forecast(_all("8-8-8-8", 16), ents, "20260610", {"venue": "中京", "surface": "芝", "distance": 1600})
    assert len(no_num[0]) == 15


def test_pace_counts_use_pre_gate_style():
    # 補正で脚質の境界をまたいでも、ペースは馬そのものの脚質（補正前）で数える
    doc = _all("3-3-3-3", 16)     # norm = 2/15 = 0.133 → 逃げ
    by_no, pace, _ = pf.build_position_forecast(doc, _entries(16), "20260610", {"venue": "中京", "surface": "芝", "distance": 1600})
    assert by_no[16]["style_raw_code"] == "front" and by_no[16]["style_code"] == "stalker"     # 最外は後ろへ補正される
    assert by_no[1]["style_code"] == "front"
    assert pace["front_runner_count"] == 16


def test_attach_adds_gate_bias_and_uses_response_course_when_not_given():
    resp = {"venue": "中京", "surface": "芝", "distance": 1600, "predictions": [{"horse_number": i, "horse_id": f"h{i}"} for i in range(1, 17)]}
    out = pf.attach_position_forecast(resp, _all("8-8-8-8", 16), "20260610")
    assert out["gate_bias"]["level"] == "inner" and out["predictions"][0]["position_forecast"]["gate_rel"] == 0.0
