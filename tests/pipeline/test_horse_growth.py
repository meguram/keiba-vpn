import pandas as pd

from src.pipeline.inference import horse_growth as hg


def _runs():
    return [
        {"date": "2024-03-10", "race_id": "r1", "race_name": "新馬", "venue": "東京", "surface": "芝", "distance": 1600, "finish": 3, "field": 14},
        {"date": "2024-06-02", "race_id": "r2", "race_name": "未勝利", "venue": "東京", "surface": "芝", "distance": 1600, "finish": 1, "field": 16},
        {"date": "2025-02-09", "race_id": "r3", "race_name": "1勝クラス", "venue": "京都", "surface": "芝", "distance": 1800, "finish": 2, "field": 12},
        {"date": "2025-05-04", "race_id": "r4", "race_name": "2勝クラス", "venue": "京都", "surface": "芝", "distance": 1800, "finish": 5, "field": 10},
    ]


def test_parse_birthday_and_age_months():
    assert hg.parse_birthday("2022年4月5日") == (2022, 4, 5)
    assert hg.parse_birthday("2022-04-05") == (2022, 4, 5)
    assert hg.parse_birthday("") is None and hg.parse_birthday("不明") is None
    bd = (2022, 4, 5)
    assert hg.age_months(bd, "2024-04-05") == 24
    assert hg.age_months(bd, "2024-04-04") == 23          # 誕生日の前日はまだ満月齢が上がらない
    assert hg.age_months(bd, "2022-01-01") is None        # 生まれる前は出さない
    assert hg.age_months(None, "2024-04-05") is None


def test_align_removes_level_shift_between_eras():
    # 2024 年は水準 45、2025 年は水準 95 の版。同じ『平均より +5』の走は、補正後に同じ値になる
    a = hg.align(50.0, 45.0, 50.0)
    b = hg.align(100.0, 95.0, 50.0)
    assert a == b == 55.0
    assert hg.align(70.0, None, 50.0) == 70.0              # 月の基準が無ければ補正しない


def test_trend_labels_and_short_series():
    assert hg.trend([50.0]) is None
    assert hg.trend([40.0, 55.0])["label"] == "上昇中"
    assert hg.trend([50.0, 51.0, 49.0, 50.0, 50.5, 50.0])["label"] == "横ばい"
    assert hg.trend([60.0, 60.0, 60.0, 50.0, 48.0, 49.0])["label"] == "下降中"


def test_build_growth_orders_by_date_excludes_invalid_and_uses_baseline():
    megu = {"r1": (44.0, "valid"), "r2": (50.0, "valid"), "r3": (99.0, "valid"), "r4": (-900.0, "out_of_range")}
    baseline = {"2024-03": 44.0, "2024-06": 45.0, "2025-02": 94.0, "2025-05": 95.0}
    g = hg.build_growth(list(reversed(_runs())), megu, baseline, 50.0, "2022年4月5日")
    assert g["status"] == "ok" and g["n"] == 3 and g["excluded"] == 1
    assert [p["race_id"] for p in g["points"]] == ["r1", "r2", "r3"]
    assert [p["value"] for p in g["points"]] == [50.0, 55.0, 55.0]
    assert g["points"][0]["raw"] == 44.0 and g["points"][0]["age_months"] == 23
    assert g["best"]["value"] == 55.0


def test_build_growth_no_data_when_nothing_usable():
    g = hg.build_growth(_runs(), {"r1": (1.0, "out_of_range")}, {}, 50.0)
    assert g["status"] == "no_data" and g["excluded"] == 1
    assert hg.build_growth(_runs(), {}, {}, 50.0)["status"] == "no_data"


def test_reads_parquet_per_horse_and_builds_baseline(tmp_path):
    for y, shift in ((2024, 0.0), (2025, 50.0)):
        d = tmp_path / str(y)
        d.mkdir()
        rows, res = [], []
        for i in range(300):                                # 月の中央値を出すには 200 件以上が要る
            rid = f"{y}{i:04d}"
            rows.append({"race_id": rid, "horse_id": f"h{i % 50}", "megu_index": 40.0 + shift + (i % 7), "computation_status": "valid"})
            res.append({"race_id": rid, "date": f"{y}-06-0{1 + i % 8}"})
        pd.DataFrame(rows).to_parquet(d / "megu_index_flat.parquet")
        pd.DataFrame(res).to_parquet(d / "race_result_flat.parquet")
    baseline, ref = hg.month_baseline(tmp_path)
    assert abs(baseline["2025-06"] - baseline["2024-06"] - 50.0) < 1e-6
    m = hg.megu_map_for_horse("h3", {2024, 2025}, tmp_path)
    assert m and all(st == "valid" for _, st in m.values())
    runs = [{"date": "2024-06-01", "race_id": "20240003"}, {"date": "2025-06-01", "race_id": "20250003"}]
    g = hg.growth_for_horse("h3", runs, None, tmp_path)
    assert g["status"] == "ok" and g["n"] == 2
    assert abs(g["points"][0]["value"] - g["points"][1]["value"]) < 1e-6      # 年またぎの水準差が消える


def test_growth_for_horse_without_files_returns_no_data(tmp_path):
    g = hg.growth_for_horse("hx", [{"date": "2024-06-01", "race_id": "r"}], None, tmp_path)
    assert g["status"] == "no_data"


def test_compact_dates_from_entry_card_use_the_right_month_key():
    runs = [{"date": "20250209", "race_id": "r3"}, {"date": "20240310", "race_id": "r1"}]
    g = hg.build_growth(runs, {"r1": (44.0, "valid"), "r3": (99.0, "valid")}, {"2024-03": 44.0, "2025-02": 94.0}, 50.0)
    assert [p["value"] for p in g["points"]] == [50.0, 55.0] and g["points"][0]["date"] == "20240310"


def test_axis_can_be_switched_in_one_place(tmp_path, monkeypatch):
    d = tmp_path / "2025"
    d.mkdir()
    pd.DataFrame([{"race_id": "rA", "horse_id": "hz", "rating": 61.0}]).to_parquet(d / "other_flat.parquet")
    other = hg.Axis("other", "別の指数", "other_flat.parquet", "rating", None, align_by_month=False)
    monkeypatch.setitem(hg.AXES, "other", other)
    monkeypatch.setenv("KEIBA_GROWTH_AXIS", "other")
    assert hg.active_axis().name == "別の指数"
    m = hg.megu_map_for_horse("hz", {2025}, tmp_path)
    assert m == {"rA": (61.0, "valid")}
    g = hg.build_growth([{"date": "20250301", "race_id": "rA"}], m, {"2025-03": 10.0}, 50.0)
    assert g["axis"]["name"] == "別の指数" and g["points"][0]["value"] == 61.0, "月補正を切った指数は元の値のまま"


def test_robust_axis_uses_restored_values_and_marks_low_confidence(tmp_path):
    import pandas as pd

    from src.pipeline.inference import horse_growth as hg

    d = tmp_path / "2025"
    d.mkdir()
    pd.DataFrame({
        "race_id": ["r1", "r2", "r3", "r4"], "horse_id": ["h"] * 4,
        "megu_index_robust": [55.0, 30.0, 100.0, float("nan")],
        "megu_index_status": ["ok", "restored_out_of_range", "clipped", "no_par"],
        "low_confidence": [False, True, False, False],
    }).to_parquet(d / "megu_index_flat.parquet")
    ax = hg.AXES["megu_robust"]
    mm = hg.megu_map_for_horse("h", {2025}, tmp_path, ax)
    assert set(mm) == {"r1", "r2", "r3"}            # 値の無い行（基準なし）は入らない
    runs = [{"race_id": f"r{i}", "date": f"2025-0{i}-10", "race_name": f"R{i}", "finish": i} for i in (1, 2, 3)]
    g = hg.build_growth(runs, mm, {}, 50.0, None, ax)
    assert g["status"] == "ok" and g["n"] == 3 and g["excluded"] == 0
    assert [p["value"] for p in g["points"]] == [55.0, 30.0, 100.0]   # 月別の補正はしない
    assert [p["low_confidence"] for p in g["points"]] == [False, True, False]
