"""走法軸の点数別 勝率・複勝率の集計（期待率の表・区分）と、HTML 側の高−低の差の確認。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.scripts.data import axis_score_curves as asc
from src.scripts.docs import build_axis_score_curves_html as page


def test_dist_band_and_class():
    assert asc.dist_band(1200) == "〜1200m" and asc.dist_band(1600) == "1500〜1700m" and asc.dist_band(2500) == "2500m〜"
    assert asc.class_of("G1") == "OP以上" and asc.class_of("2勝") == "2・3勝" and asc.class_of("障害") is None


def test_place_cut_by_field_size():
    assert list(asc.place_cut(np.array([5, 7, 8, 18]))) == [2, 2, 3, 3]


def test_expected_tables_follow_popularity_and_sum_near_one():
    rng = np.random.default_rng(0)
    rows = []
    for r in range(3000):
        n = 12
        order = rng.permutation(n)             # 人気順と無関係なら、どの人気順も同じ率になる
        for pop in range(1, n + 1):
            rows.append((r, pop, n, order[pop - 1] + 1))
    t = pd.DataFrame(rows, columns=["race_id", "popularity", "field_size", "finish_position"])
    tw, tp = asc.expected_tables(t)
    assert abs(tw[12, 1:13].sum() - 1.0) < 0.05
    assert abs(tp[12, 1:13].mean() - 3 / 12) < 0.02
    assert tw[12, 5] > 0 and tw[4, 1] == 0           # 5 頭未満は表に無い


def test_side_stats_adjusted_difference():
    n = [100] * 10
    p = [20] * 4 + [30] + [40] * 5        # 複勝数（低 20%・高 40%）
    arr = [n, [0] * 10, p, [0] * 10, p]       # 期待 = 実績 → 補正後 0
    s = page.side_stats(arr)
    assert abs(s["raw"] - 0.20) < 1e-9 and abs(s["adj"]) < 1e-9
    assert page.side_stats([[10] * 10] + [[1] * 10] * 4) is None     # 走数が少なければ出さない


def test_side_stats_excludes_middle_and_signs_z():
    n = [100] * 10
    p = [20] * 4 + [99] + [40] * 5        # 5 点台の区間は高低どちらにも入らない
    arr = [n, [0] * 10, p, [0] * 10, [30] * 10]   # 期待は一律 30% → 補正後 = 実績の差と同じ
    s = page.side_stats(arr)
    assert abs(s["raw"] - 0.20) < 1e-9 and s["z"] > 0
    arr2 = [n, [0] * 10, p[::-1], [0] * 10, [30] * 10]
    assert page.side_stats(arr2)["z"] < 0


def test_build_embeds_constants_and_escapes_script_end():
    c = {"A": [[200] * 10, [20] * 10, [60] * 10, [20.0] * 10, [60.0] * 10], "B": [[50] * 10, [5] * 10, [15] * 10, [5.0] * 10, [15.0] * 10]}
    d = {"meta": {"built": "2026-11-18", "exp_years": [2016, 2023], "periods": {"A": ["2024-01-01", "2025-12-31"], "B": ["2026-01-01", "2026-12-31"]},
                  "runs": {"A": 2000, "B": 500}, "axis_runs": {"ts": {"A": 2000, "B": 500}}, "course_min_runs": 400, "history_mtime": "2026-11-12T00:00:00"},
         "axes": [{"key": "ts", "label": "</script>TS", "high": "高", "low": "低"}],
         "dims": {"all": {"label": "全体", "groups": [{"key": "全体", "label": "全体", "data": {"ts": c}}]}}}
    h = page.build(d)
    assert "__MINN__" not in h and f"SIDE={page.MIN_SIDE}" in h and f"MINN={page.MIN_N}" in h
    assert h.count("</script>") == 1


def test_parse_course_key_and_grid_union_and_order():
    assert page.parse_course_key("京都芝1600m") == ("京都", "芝", 1600)
    assert page.parse_course_key("中山ダ2400m") == ("中山", "ダ", 2400)
    assert page.parse_course_key("全体") is None and page.parse_course_key("京都障1600m") is None
    ent = [("阪神", "芝", 1600, 0), ("札幌", "芝", 1200, 1), ("京都", "ダ", 1800, 2), ("東京", "芝", 1400, 3), ("謎場", "芝", 1000, 4), ("札幌", "ダ", 1700, 5)]
    g = page.build_course_grid(ent)
    assert g["芝"]["venues"] == ["札幌", "東京", "阪神", "謎場"]                 # 競馬場の定義順、知らない場は末尾
    assert g["芝"]["dists"] == [1000, 1200, 1400, 1600]                          # 距離の和集合を昇順
    assert g["ダ"]["venues"] == ["札幌", "京都"] and g["ダ"]["dists"] == [1700, 1800]
    assert g["芝"]["idx"]["阪神|1600"] == 0 and "阪神|1200" not in g["芝"]["idx"]
    assert sum(len(x["idx"]) for x in g.values()) == len(ent)                    # 芝 + ダの格子の点数 = コース数


def test_axis_course_grid_and_ts_grid_accept_dart_notation():
    d = {"dims": {"course": {"groups": [{"key": "京都芝1600m"}, {"key": "中山ダ1800m"}, {"key": "壊れた"}]}}}
    g = page.axis_course_grid(d)
    assert g["芝"]["idx"] == {"京都|1600": 0} and g["ダ"]["idx"] == {"中山|1800": 1}
    ts = {"courses": [{"venue": "中山", "surface": "ダート", "distance": 1800}, {"venue": "京都", "surface": "芝", "distance": 1600}]}
    tg = page.ts_course_grid(ts)
    assert tg["ダ"]["idx"] == {"中山|1800": 0} and tg["芝"]["idx"] == {"京都|1600": 1}


def test_git_definition_dates_use_injected_runner_and_never_fail():
    log = "2026-11-12 abc1234 feat(軸): 定義を直す\n"
    assert page.parse_log_line("\n" + log) == {"date": "2026-11-12", "hash": "abc1234", "subject": "feat(軸): 定義を直す"}
    assert page.parse_log_line("") is None and page.parse_log_line(None) is None
    calls = []

    def fn_ok(args):
        calls.append(args)
        return log

    r = page.last_change("a/b.py", "func", runner=fn_ok)
    assert r["grain"] == "関数" and r["func"] == "func" and "-L" in calls[0] and ":func:a/b.py" in calls[0]

    def fn_file_only(args):
        return None if "-L" in args else log

    r = page.last_change("a/b.py", "func", runner=fn_file_only)
    assert r["grain"] == "ファイル" and r["date"] == "2026-11-12"
    assert page.last_change("a/b.py", "func", runner=lambda a: None) is None

    out = page.axis_defs(["ts", "basic", "zzz"], runner=lambda a: log)
    assert out["ts"]["hash"] == "abc1234" and out["basic"]["file"].endswith("basic_speed.py") and out["zzz"] is None
    shallow = page.axis_defs(["ts"], runner=lambda a: "true\n" if "--is-shallow-repository" in a else log)
    assert shallow == {"ts": None}                                              # 浅いクローンは日付が当てにならない


def test_def_note_marks_stale_points_and_unknown():
    info = {"date": "2026-11-13", "hash": "abc1234", "subject": "fix", "grain": "関数", "file": "src/x/run_axes.py", "func": "with_ts"}
    assert "点が未更新の可能性" in page.def_note_html(info, "2026-11-12")
    assert "点が未更新の可能性" not in page.def_note_html(info, "2026-11-13")           # 同じ日は印なし（定義 > 点 のときだけ）
    s = page.def_note_html(info, "2026-11-12")
    assert "2026-11-13" in s and "abc1234" in s and "関数単位" in s and "run_axes.py::with_ts" in s
    assert "不明（git なし）" in page.def_note_html(None, "2026-11-12")


def _mini_d(courses=("京都芝1600m", "中山ダ1800m")):
    cell = {"A": [[200] * 10, [20] * 10, [60] * 10, [20.0] * 10, [60.0] * 10], "B": [[50] * 10, [5] * 10, [15] * 10, [5.0] * 10, [15.0] * 10]}
    groups = [{"key": k, "label": k, "data": {"ts": cell}} for k in courses]
    return {"meta": {"built": "2026-11-18", "exp_years": [2016, 2023], "periods": {"A": ["2024-01-01", "2025-12-31"], "B": ["2026-01-01", "2026-12-31"]},
                     "runs": {"A": 2000, "B": 500}, "axis_runs": {"ts": {"A": 2000, "B": 500}}, "course_min_runs": 400, "history_mtime": "2026-11-12T00:00:00"},
            "axes": [{"key": "ts", "label": "TS", "high": "高", "low": "低"}],
            "dims": {"all": {"label": "全体", "groups": [{"key": "全体", "label": "全体", "data": {"ts": cell}}]},
                     "course": {"label": "コース", "groups": groups}}}


def _mini_ts(courses):
    hist = [0] * 22
    hist[11] = 10
    cs = [{"venue": v, "surface": s, "distance": dd, "n": 10, "ts_sd": 9.5 + i / 10, "ts_skew": -1.5 if i == 0 else 0.1, "ts_p05": 35.0, "ts_p25": 45.0, "ts_p50": 50.0,
           "ts_p75": 55.0, "ts_p95": 65.0, "ts_min": 12.0, "ts_max": 80.0, "ts_hist": hist, "peak20": 1 if i == 0 else 0}
          for i, (v, s, dd) in enumerate(courses)]
    band = {"surface": "芝", "band": "1500〜1700m", "n": 100, "races": 10, "ts_sd": 9.8, "ts_skew": -3.5, "ts_p05": 34.0, "ts_p50": 50.6, "ts_p95": 64.0, "peak_mean": 11.4, "front115": 0.67,
            "peak20": 1, "rel": {"rho_split": 0.35, "rho_split_ci": [0.3, 0.4]}}
    return {"meta": {"built": "2026-11-18", "runs": 100, "races": 10, "date_from": "2024-01-06", "date_to": "2026-11-07", "hist": {"lo": 10.0, "hi": 90.0, "step": 4.0, "peak_outlier": 20.0}},
            "all_rel": None, "bands": [band], "courses": cs}


def test_build_without_ts_json_has_no_ts_section_but_has_course_grid_section():
    h = page.build(_mini_d())
    assert 'id="cg-ts"' not in h and "table.sortable" not in h and 'id="cg-axes"' in h and "<h2>6. コース別" in h and "<h2>7. 限界" in h
    assert "<th>定義の最終更新</th>" not in h and "点の作成日" not in h               # defs なしなら日付の列・注記は出ない
    assert '"ts":null' in h


def test_build_with_ts_json_rows_equal_course_count_and_embeds_numbers_from_json():
    courses = [("京都", "芝", 1600), ("中山", "ダート", 1800), ("小倉", "芝", 1200)]
    ts = _mini_ts(courses)
    h = page.build(_mini_d(), ts, {"ts": {"date": "2026-11-13", "hash": "abc1234", "subject": "fix 定義", "grain": "関数", "file": "src/r/run_axes.py", "func": "with_ts"}})
    assert 'id="cg-ts"' in h and "ts-dist" in h and h.count("table.sortable") == 1          # SORT_JS が入る
    detail = h.split("<details class=\"fl\">")[1].split("</details>")[0]
    assert detail.count("<tr>") - 1 == len(ts["courses"])                                  # 見出し行を除いた行 = コース数
    assert "京都芝1600m（1 走・歪み -1.5）" in h                                          # 外れ値の文は JSON の値から作る
    assert "定義の最終更新 2026-11-13" in h and "点が未更新の可能性" in h
    assert h.count("</script>") == 2                                                       # 本体（データ・JS）+ 並べ替え
    assert "<th>定義の最終更新</th>" in h
    assert "打ち切ってから偏差値にしている" in h and "検討中" not in h


def test_ts_block_histogram_sums_to_runs_and_counts_outlier_peaks():
    from src.scripts.data import ts_course_resolution as tcr

    rng = np.random.default_rng(1)
    ts = np.concatenate([rng.normal(50, 10, 500), [3.0, 95.0, -40.0]])        # 両端の外れ値の階級にも入る
    g = pd.DataFrame({"ts": ts, "peak": np.concatenate([rng.normal(11.5, 0.3, 500), [20.5, 11.0, 25.0]])})
    b = tcr.ts_block(g)
    assert len(b["ts_hist"]) == 22 and sum(b["ts_hist"]) == len(g)            # 10〜90 を 4 点刻み 20 階級 + 両端
    assert b["ts_hist"][0] == 2 and b["ts_hist"][-1] == 1                       # 10 未満（3.0, -40.0）と 90 以上（95.0）
    assert b["peak20"] == 2 and b["ts_min"] == -40.0 and b["ts_max"] == 95.0
    assert b["ts_p05"] < b["ts_p25"] < b["ts_p50"] < b["ts_p75"] < b["ts_p95"]


def test_ts_section_box_id_does_not_collide_with_axis_box_ids():
    h = page.build(_mini_d(), _mini_ts([("京都", "芝", 1600)]))
    axis_keys = [a["key"] for a in _mini_d()["axes"]]                  # 軸側の箱は "cgbig-" + 軸キーで JS が作る（キーに ts がある）
    assert 'id="cgbig-tsdist"' in h and all(f'id="cgbig-{k}"' not in h for k in axis_keys)


def test_estimate_fills_sparse_bins_within_observed_range():
    """走数の少ない区間は、走数の多い区間への当てはめで埋め、実測の最小〜最大の範囲に収める（JS の est を node で動かす）。"""
    import json
    import shutil
    import subprocess

    import pytest

    if shutil.which("node") is None:
        pytest.skip("node が無い")
    js = page.JS
    src = js[js.index("function fitLogit"):js.index("function sum(a,i,j)")]
    n = [0, 5, 400, 500, 600, 500, 400, 30, 0, 0]
    rate = [0.0, 0.0, 0.04, 0.06, 0.08, 0.10, 0.12, 0.13, 0.0, 0.0]
    k = [round(a * b) for a, b in zip(n, rate)]
    few_k, few_n = [1, 2, 3, 0, 0, 0, 0, 0, 0, 0], [40, 50, 20, 0, 0, 0, 0, 0, 0, 0]       # 実測の区間が 2 つだけ
    exp_n = [0, 5, 400, 500, 600, 500, 400, 30, 0, 0]
    code = (f"var MINN={page.MIN_N};{src}console.log(JSON.stringify([est({json.dumps(k)},{json.dumps(n)}),"
            f"est({json.dumps(few_k)},{json.dumps(few_n)}),est({json.dumps([x / 10 for x in exp_n])},{json.dumps(exp_n)},true)]));")
    out = json.loads(subprocess.run(["node", "-e", code], capture_output=True, text=True, check=True).stdout)
    est, few, exp = out
    assert len(est) == 10 and all(o is not None for o in est)
    assert [o["e"] for o in est] == [True, True, False, False, False, False, False, False, True, True]
    obs = [o["v"] for o in est if not o["e"]]
    assert all(min(obs) - 1e-9 <= est[i]["v"] <= max(obs) + 1e-9 for i in (0, 8, 9))   # 走数 0 の区間は曲線そのもの（実測の範囲内）
    assert est[8]["v"] == pytest.approx(max(obs)) and est[9]["v"] == pytest.approx(max(obs))                                     # 上り調子の延長は、実測の最大で頭打ち
    assert est[1]["v"] < est[2]["v"]                                                    # 走数 5 で勝ち 0 の区間は、実測に引かれて曲線より低い
    assert all(o is None or not o["e"] for o in few)                                    # 実測の区間が 3 未満なら推定しない
    assert few[2] == {"v": pytest.approx(3 / 20), "e": False, "r": True, "n": 20}       # 推定できないときも、走数不足の区間は実測を参考として出す（省かない）
    assert not few[0].get("r") and few[3] is None                                       # 走数 MINN 以上は通常の実測、走数 0 は描けない
    assert exp[1] == {"v": 0.1, "e": False} and exp[0]["e"] and exp[8]["e"]              # 期待率: 走数が少なくても実測、走数 0 だけ曲線
