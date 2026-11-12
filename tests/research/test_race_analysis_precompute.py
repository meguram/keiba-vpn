"""めぐ分析: 過去位置取りの計算と、事前計算（画面は保存済みの結果を読むだけ）。"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.research.race_analysis import past_position as P
from src.research.race_analysis import precompute, store
from src.research.race_analysis import runs as runs_mod


# ── 過去位置取り（純粋関数） ──

def test_outliers_only_removed_with_five_or_more_runs():
    rs = [0.10, 0.12, 0.15, 0.95, 0.11]                  # 一度だけ大きく後ろ（出遅れ）
    assert P.split_outliers(rs) == {3}
    assert P.split_outliers(rs[:4]) == set()               # 履歴が 4 走なら除かない
    assert P.split_outliers([0.1, 0.9, 0.1, 0.9, 0.1]) == set() or len(P.split_outliers([0.1, 0.9, 0.1, 0.9, 0.1])) <= 1   # 割れている馬は上限まで


def test_exclusion_is_capped_by_fraction_of_history():
    rs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.99, 0.99]   # 外れ値 2 つ / 10 走 → 上限 2 つ
    assert len(P.split_outliers(rs)) <= int(len(rs) * P.MAX_EXCLUDE_FRAC)


def test_weighted_mean_gives_recent_runs_more_weight():
    new_front = P.horse_position([0.1, 0.5, 0.5, 0.5]).value        # 直近だけ前
    old_front = P.horse_position([0.5, 0.5, 0.5, 0.1]).value        # 昔だけ前
    assert new_front < old_front < 0.5
    st = P.horse_position([0.2, 0.4])
    w = [1.0, P.DECAY]
    assert st.value == pytest.approx((0.2 * w[0] + 0.4 * w[1]) / sum(w))
    assert P.horse_position([]).value is None


def test_excluded_run_keeps_recency_weights():
    st = P.horse_position([0.10, 0.12, 0.15, 0.95, 0.11])
    assert [i for i, _ in st.excluded] == [3] and [i for i, _ in st.used] == [0, 1, 2, 4]
    ws = [P.DECAY ** i for i in (0, 1, 2, 4)]
    assert st.value == pytest.approx(sum(w * r for w, r in zip(ws, [0.10, 0.12, 0.15, 0.11])) / sum(ws))


def _field(values, se=0.05):
    return {f"h{i}": P.PositionStat(v, se) for i, v in enumerate(values)}


def test_single_leader_by_default_and_others_by_band():
    vals = [0.07, 0.3, 0.1, 0.5, 0.6, 0.9, 0.7, 0.2, 0.4, 0.8, 0.15, 0.55, 0.65, 0.35]     # 14 頭、先頭 0.07 と 2 番手 0.10 は差が大きい
    lab = P.classify(_field(vals))
    by_val = [lab[f"h{i}"] for i in sorted(range(len(vals)), key=lambda i: vals[i])]
    assert by_val.count("ハナ") == 1 and by_val[0] == "ハナ"
    # 帯の上端を含む: 0.2 → 先行(前)、0.4 → 先行(後ろ)、0.6 → 差し(前)、0.8 → 差し(後ろ)
    assert by_val == ["ハナ", "先行(前)", "先行(前)", "先行(前)", "先行(後ろ)", "先行(後ろ)", "先行(後ろ)",
                      "差し(前)", "差し(前)", "差し(前)", "差し(後ろ)", "差し(後ろ)", "差し(後ろ)", "追込"]


def test_style_and_band_are_one_to_one():
    assert P.STYLES == ("ハナ", "先行(前)", "先行(後ろ)", "差し(前)", "差し(後ろ)", "追込")
    assert list(P.PALETTE) == list(P.STYLES)                                    # 並べ替えの順も脚質の順
    # 先頭の馬は位置がどこでもハナ（メンバーが全頭後ろでも）。ハナ以外は帯だけで決まる（メンバーに依らない）
    lab = P.classify(_field([0.7, 0.85, 0.95]))
    assert lab == {"h0": "ハナ", "h1": "追込", "h2": "追込"}
    assert P.classify(_field([0.05, 0.5]))["h1"] == P.classify(_field([0.05, 0.3, 0.4, 0.5]))["h3"] == "差し(前)"


def test_co_leaders_only_when_gap_is_within_error_and_at_most_three():
    close = _field([0.100, 0.102, 0.103, 0.104, 0.5, 0.6, 0.7, 0.8])     # 先頭 4 頭がほぼ同じ
    lab = P.classify(close)
    assert sum(v == "ハナ" for v in lab.values()) == 3                     # 最大 3 頭
    apart = _field([0.10, 0.20, 0.5, 0.6, 0.7, 0.8])
    assert sum(v == "ハナ" for v in P.classify(apart).values()) == 1
    # 推定誤差が大きい（走数が少ない）馬どうしは、同じ差でも同着扱いになりやすい
    noisy = {"a": P.PositionStat(0.10, 0.12), "b": P.PositionStat(0.12, 0.12), "c": P.PositionStat(0.6, 0.05), "d": P.PositionStat(0.7, 0.05)}
    assert sum(v == "ハナ" for v in P.classify(noisy).values()) == 2


def test_unknown_horses_are_not_ranked():
    lab = P.classify({"a": P.PositionStat(0.1, 0.05), "b": P.PositionStat(None, None), "c": P.PositionStat(0.8, 0.05)})
    assert lab["b"] == "不明" and lab["a"] == "ハナ"


# ── 事前計算 ──

class FakeStorage:
    def __init__(self, data):
        self.data = data
        self.loads = 0

    def load(self, cat, key):
        self.loads += 1
        return self.data.get((cat, key))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from src.research.race_analysis.running_map_module import RunningMapModule

    monkeypatch.setattr(RunningMapModule, "applicable", lambda self, ctx: False)     # 全期間の出走記録を読むので対象外
    rows = [
        {"race_id": f"R{i}", "date": f"2026-0{1 + i // 3}-{10 + i % 3}", "venue": "東京", "surface": "芝", "distance": 1600, "finish_position": 1 + i % 5,
         "jockey_id": "J1", "bracket_number": 1, "horse_id": "h1", "passing_order": f"{1 + i % 4}-{1 + i % 4}", "field_size": 10}
        for i in range(8)
    ]
    d = tmp_path / "flat" / "2026"
    d.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(d / "race_result_flat.parquet")
    monkeypatch.setattr(runs_mod, "FLAT_DIR", tmp_path / "flat")
    monkeypatch.setattr(runs_mod, "SIRE_MAP_PATH", tmp_path / "none.parquet")
    runs_mod._cache.update(sig=None, df=None, sire_sig=None, sire_df=None, roi_sig=None, roi_df=None)
    monkeypatch.setenv("KEIBA_RACE_ANALYSIS_DIR", str(tmp_path / "out"))
    card = {"race_name": "T", "date": "2026-10-31", "venue": "東京", "round": 11, "surface": "芝", "distance": 1600,
            "entries": [{"horse_number": 1, "bracket_number": 1, "horse_name": "イチ", "horse_id": "h1", "jockey_id": "J1"},
                        {"horse_number": 2, "bracket_number": 2, "horse_name": "ニ", "horse_id": "hNew", "jockey_id": "J2"}]}
    return FakeStorage({("race_shutuba", "202605040111"): card}), tmp_path


def test_precompute_writes_once_and_skips_when_inputs_are_unchanged(env):
    storage, _ = env
    assert precompute.precompute_race(storage, "202605040111") == "written"
    saved = store.read("202605040111")
    assert saved["status"] == "ok" and saved["version"] == precompute.ANALYSIS_VERSION and saved["computed_at"]
    assert {m["id"] for m in saved["modules"]} >= {"jockey_course", "gate_course", "past_position"}
    assert precompute.precompute_race(storage, "202605040111") == "unchanged"                  # 入力が同じなら書き直さない
    storage.data[("race_shutuba", "202605040111")]["entries"][1]["jockey_id"] = "J3"            # 騎手変更
    assert precompute.precompute_race(storage, "202605040111") == "written"
    assert precompute.precompute_race(storage, "202605040111", force=True) == "written"


def test_precompute_when_new_results_arrive_for_a_future_race(env, tmp_path):
    storage, base = env
    assert precompute.precompute_race(storage, "202605040111") == "written"
    more = pd.DataFrame([{"race_id": "RX", "date": "2026-10-25", "venue": "東京", "surface": "芝", "distance": 1600, "finish_position": 1, "jockey_id": "J1",
                          "bracket_number": 1, "horse_id": "h1", "passing_order": "1-1", "field_size": 10}])
    more.to_parquet(base / "flat" / "2026" / "race_result_flat.parquet")                       # 結果データが更新された
    assert precompute.precompute_race(storage, "202605040111") == "written"                     # 集計に使う最終日が変わった → 作り直す


def test_own_result_changes_the_fingerprint_so_finish_marks_get_added(env):
    storage, _ = env
    assert precompute.precompute_race(storage, "202605040111") == "written"
    assert precompute.precompute_race(storage, "202605040111") == "unchanged"
    storage.data[("race_result_on_time", "202605040111")] = {"entries": [{"horse_id": "h1", "finish_position": 2}, {"horse_id": "hNew", "finish_position": "中止"}]}
    assert precompute.own_finish(storage, "202605040111") == [["h1", 2]]
    assert precompute.precompute_race(storage, "202605040111") == "written"                     # 速報の着順が入った → 作り直す
    storage.data[("race_result", "202605040111")] = {"entries": [{"horse_id": "h1", "finish_position": 1}, {"horse_id": "hNew", "finish_position": 2}]}
    assert precompute.precompute_race(storage, "202605040111") == "written"                     # 確定で着順が変わった → 作り直す
    assert precompute.precompute_race(storage, "202605040111") == "unchanged"


def test_finish_positions_skip_non_numeric_and_missing():
    from src.research.race_analysis.running_map_module import finish_positions

    card = {"entries": [{"horse_id": "a", "finish_position": 1}, {"horse_id": "b", "finish_position": 5.0}, {"horse_id": "c", "finish_position": None},
                        {"horse_id": "d", "finish_position": "取消"}, {"horse_id": "e", "finish_position": 0}, {"finish_position": 2}]}
    assert finish_positions(card) == {"a": 1, "b": 5}
    assert finish_positions(None) == {}


def test_no_entries_is_not_stored(env):
    storage, _ = env
    assert precompute.precompute_race(storage, "202605040199") == "no_entries"
    assert store.read("202605040199") is None


def test_past_position_module_classifies_and_uses_first_corner(env):
    storage, _ = env
    precompute.precompute_race(storage, "202605040111")
    saved = store.read("202605040111")
    m = next(x for x in saved["modules"] if x["id"] == "past_position")
    assert m["status"] == "ok" and [c["key"] for c in m["columns"]] == ["style", "range", "n", "hist", "band_win", "band_top3", "band_win_roi", "band_place_roi"]
    labels = [c["label"] for c in m["columns"]]
    assert labels[4:] == ["勝率", "複勝率", "単勝回収率", "複勝回収率"] and not any("帯" in x for x in labels)
    cells = {e["horse_name"]: e["analysis"]["past_position"] for e in saved["entries"]}
    assert cells["イチ"]["n"] == 8 and cells["イチ"]["style"] == "ハナ"          # h1 は履歴あり。もう 1 頭は初出走なので分類しない → 1 頭だけ前
    assert cells["ニ"]["style"] == "不明" and cells["ニ"]["pos"] is None and cells["ニ"]["hist"] is None
    assert 0 < cells["イチ"]["pos"] < 0.5 and math.isclose(cells["イチ"]["pos"], cells["イチ"]["pos"])


# ── 補助の計算（Best5・血統）が欠けた結果は、正常な保存済みの結果を上書きしない ──

def test_degraded_reasons_picks_module_errors_and_build_errors():
    assert precompute.degraded_reasons({"modules": [{"id": "a", "status": "ok"}]}) == []
    got = precompute.degraded_reasons({"modules": [{"id": "running_map", "status": "ok", "figure": {"build_errors": ["best5", "ped"]}},
                                                   {"id": "gate_course", "status": "error"}]})
    assert got == ["running_map:best5", "running_map:ped", "module:gate_course"]


def test_degraded_result_never_overwrites_a_good_one_and_is_retried(env, monkeypatch):
    storage, _ = env
    rid = "202605040111"
    assert precompute.precompute_race(storage, rid) == "written"
    good = store.read(rid)
    real = precompute.analyze_race

    def broken(st, r):
        p = real(st, r)
        p["modules"][0]["figure"] = {"build_errors": ["best5_missing"]}
        return p

    monkeypatch.setattr(precompute, "analyze_race", broken)
    assert precompute.precompute_race(storage, rid, force=True) == "degraded"
    assert store.read(rid) == good                                                       # 正常な結果を残す
    # 保存済みが無い場合は、欠けた結果を指紋なしで保存し、直れば次の周回で作り直す
    store.path_of(rid).unlink()
    assert precompute.precompute_race(storage, rid) == "degraded"
    saved = store.read(rid)
    assert len(saved["degraded"]) == 1 and saved["degraded"][0].endswith(":best5_missing")
    assert saved["fingerprint"] == "degraded"
    monkeypatch.setattr(precompute, "analyze_race", real)
    assert precompute.precompute_race(storage, rid) == "written"
    assert "degraded" not in store.read(rid)
    assert precompute.precompute_races(storage, [rid])["degraded"] == 0


# ── API は保存済みの結果を読むだけ ──

def test_api_reads_the_stored_result_and_never_computes(env, monkeypatch):
    storage, _ = env
    precompute.precompute_race(storage, "202605040111")
    import src.research.race_analysis.service as svc

    monkeypatch.setattr(svc, "analyze_race", lambda *a, **k: (_ for _ in ()).throw(AssertionError("リクエストで計算してはいけない")))
    from flask import Flask

    from src.api.v1.routes.race_analysis import bp

    app = Flask(__name__)
    app.register_blueprint(bp, url_prefix="/api/v1")
    c = app.test_client()
    ok = c.get("/api/v1/races/202605040111/analysis").get_json()
    assert ok["status"] == "ok" and "fingerprint" not in ok and ok["computed_at"]
    pending = c.get("/api/v1/races/202605040222/analysis").get_json()                                     # 未計算 → 計算せず『集計待ち』
    assert pending["status"] == "pending" and pending["entries"] == []
    assert c.get("/api/v1/races/bad/analysis").status_code == 400


def test_prune_removes_only_old_files(env):
    import os
    import time

    storage, _ = env
    precompute.precompute_race(storage, "202605040111")
    p = store.path_of("202605040111")
    old = time.time() - 100 * 86400
    os.utime(p, (old, old))
    assert store.prune(90) == 1 and not p.exists()


def test_prune_keeps_current_year_races(env):
    import os
    import time

    storage, _ = env
    precompute.precompute_race(storage, "202605040111")
    p = store.path_of("202605040111")
    old = time.time() - 100 * 86400
    os.utime(p, (old, old))
    assert store.prune(90, keep_year=2026) == 0 and p.exists()
    assert store.prune(90, keep_year=2027) == 1 and not p.exists()


def test_band_helpers_and_stats():
    assert [P.band_of(r) for r in (0.05, 0.2, 0.21, 0.5, 0.79, 1.0)] == ["先行(前)", "先行(前)", "先行(後ろ)", "差し(前)", "差し(後ろ)", "追込"]
    assert P.band_label("ハナ") == "ハナ（初角1番手）" and P.band_label("先行(前)") == "先行(前)（0〜20%）" and P.band_label("追込") == "追込（80〜100%）"
    df = pd.DataFrame({"pos_first": [1, 2, 1, 9, 10], "field_size": [10, 10, 4, 10, 10], "finish_position": [1, 5, 4, 3, 2],
                       "win_ret": [4.0, 0.0, 0.0, float("nan"), 0.0], "place_ret": [1.5, 0.0, 0.0, float("nan"), 1.2]})
    b = P.band_stats(df)
    assert list(b) == list(P.STYLES)
    assert b["ハナ"]["n"] == 2 and b["ハナ"]["win"] == 0.5 and b["ハナ"]["win_roi"] == pytest.approx(2.0)   # 初角 1 番手は頭数に依らずハナ（1/4 も）
    assert b["先行(前)"]["n"] == 1 and b["先行(前)"]["win"] == 0.0                                   # 2/10 はハナではなく 0〜20%
    assert b["追込"]["n"] == 2 and b["追込"]["win_roi"] == 0.0 and b["追込"]["place_roi"] == pytest.approx(1.2)   # 払戻が読めない走は回収率の分母から外す
    assert b["差し(前)"]["n"] == 0 and b["差し(前)"]["win"] is None


def test_range_cell_spans_recent_runs_and_weights_recent_more():
    runs = [(2, 10), (5, 10), (9, 10), (3, 10), (4, 10), (1, 10)]               # 新しい順。6走前は表示範囲外
    st = P.horse_position([p / x for p, x in runs])
    c = P.range_cell(st, runs)
    assert c["lo"] == pytest.approx(0.2) and c["hi"] == pytest.approx(0.9) and len(c["pts"]) == 5
    assert c["pts"][0][1] == 1.0 and c["pts"][0][1] > c["pts"][4][1]             # 前走がいちばん濃い
    assert c["mean"] == pytest.approx(st.value)
    assert P.range_cell(P.horse_position([]), []) is None
