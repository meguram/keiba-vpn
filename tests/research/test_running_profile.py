"""走法データ（src/research/race/running_profile.py）。"""

from __future__ import annotations

import pandas as pd
import pytest

from src.research.race import running_profile as rp

LAPS = "[12.5, 11.0, 11.5, 12.0, 12.0, 11.8, 11.5, 11.9]"     # 1600m


def _race(rid: str, date: str, horses: list[tuple[str, int, float, str]], *, name: str = "3歳未勝利", laps: str | None = LAPS,
          venue: str = "東京", surface: str = "芝", distance: int = 1600) -> list[dict]:
    return [{"race_id": rid, "date": date, "venue": venue, "surface": surface, "distance": distance, "track_condition": "良",
             "race_name": name, "field_size": len(horses), "finish_position": pos, "horse_id": hid, "passing_order": po,
             "last_3f": l3f, "time_sec": 95.0 + pos * 0.2, "lap_times": laps} for hid, pos, l3f, po in horses]


@pytest.fixture
def flat(tmp_path):
    rows = (
        _race("R1", "2025-01-05", [("A", 1, 34.0, "1-1"), ("B", 2, 34.5, "3-2"), ("C", 3, 35.0, "4-4"), ("D", 4, 35.5, "2-3")])
        + _race("R2", "2025-02-05", [("A", 2, 34.2, "2-2"), ("B", 1, 33.8, "4-1"), ("C", 3, 35.1, "1-3"), ("D", 4, 36.0, "3-4")])
        + _race("R3", "2025-03-05", [("A", 1, 33.9, "1-1"), ("B", 3, 34.6, "2-3"), ("C", 2, 34.4, "4-2"), ("D", 4, 35.4, "3-4")],
                laps=None)                                                    # ラップの無い平地戦 → 推定で補う
        + _race("J1", "2025-03-10", [("A", 1, 40.0, "1-1"), ("B", 2, 41.0, "2-2")], name="東京ハイジャンプ(JGII)",
                laps=None, distance=3110)                                     # 障害 → 除外
        + _race("R4", "2025-04-05", [("A", 1, 33.0, "1-1"), ("B", 2, 33.5, "2-2"), ("C", 3, 34.0, "3-3")])
    )
    d = tmp_path / "2025"
    d.mkdir()
    pd.DataFrame(rows).to_parquet(d / "race_result_flat.parquet")
    return tmp_path


@pytest.fixture
def sires():
    return pd.DataFrame({"horse_id": ["A", "B", "C", "D", "NEW"], "sire_id": ["S"] * 5})


def test_jump_races_are_dropped_and_missing_laps_are_imputed(flat):
    runs = rp.load_runs([2025], flat)
    assert "J1" not in set(runs["race_id"])
    m = rp.run_metrics(runs)
    r3 = m[m["race_id"] == "R3"]
    assert r3["laps_imputed"].all() and r3["ts_score"].notna().all() and r3["split_ratio"].notna().all()
    assert not m.loc[m["race_id"] != "R3", "laps_imputed"].any()
    a1 = m[(m["race_id"] == "R1") & (m["horse_id"] == "A")].iloc[0]
    assert a1["front_ratio"] == pytest.approx(1 / 4) and a1["shift_ratio"] == pytest.approx(0.0)
    # ラスト 3F は 11.5, 11.9 … の形をこの馬の上がり 34.0 に伸縮した値の最小
    assert a1["ts_fastest_f"] == pytest.approx(11.5 * 34.0 / (11.8 + 11.5 + 11.9))


def test_profile_uses_only_runs_before_the_date_and_includes_the_latest_run(flat, sires, monkeypatch):
    monkeypatch.setattr(rp, "MIN_SIRE_RUNS", 1)
    m = rp.run_metrics(rp.load_runs([2025], flat))
    p = rp.profiles_as_of(m, ["A"], "2025-04-05", sires).iloc[0]       # R4 当日 → R1〜R3 の 3 走
    assert p["n_runs"] == 3 and p["last_run_date"] == "2025-03-05" and p["n_front_ratio"] == 3
    after = rp.profiles_as_of(m, ["A"], "2025-04-06", sires).iloc[0]  # 翌日 → R4 まで入る（前走が平均から漏れない）
    assert after["n_runs"] == 4 and after["last_run_date"] == "2025-04-05"
    assert after["front_ratio"] != p["front_ratio"]


def test_newcomer_gets_the_sire_average_and_recent_runs_weigh_more(flat, sires, monkeypatch):
    monkeypatch.setattr(rp, "MIN_SIRE_RUNS", 1)
    m = rp.run_metrics(rp.load_runs([2025], flat))
    p = rp.profiles_as_of(m, ["NEW", "A"], "2025-04-05", sires).set_index("horse_id")
    past = m[m["date"] < "2025-04-05"]
    assert p.at["NEW", "n_runs"] == 0 and p.at["NEW", "own_split_ratio"] == 0.0
    assert p.at["NEW", "split_ratio"] == pytest.approx(past["split_ratio"].mean()) and p.at["NEW", "prior_split_ratio"] == "sire"
    # A の先行度: 新しい順に R3=1/4, R2=2/4, R1=1/4 を 0.7 倍ずつの重みで平均し、父の平均へ c=1 で引き寄せる
    decay, c, _ = rp.PROFILE_PARAMS["front_ratio"]
    own, w = [0.25, 0.5, 0.25], [1, decay, decay ** 2]
    prior = past["front_ratio"].mean()
    assert p.at["A", "front_ratio"] == pytest.approx((sum(a * b for a, b in zip(own, w)) + c * prior) / (sum(w) + c))
    assert 0 < p.at["A", "own_front_ratio"] < 1


def test_metrics_are_computed_once_then_only_new_races_are_added(flat, monkeypatch, tmp_path):
    path = tmp_path / "store" / "run_metrics.parquet"
    calls = []
    real = rp.run_metrics
    monkeypatch.setattr(rp, "run_metrics", lambda runs: (calls.append(sorted(set(runs["race_id"]))), real(runs))[1])
    first = rp.load_metrics(path, flat)
    assert calls == [["R1", "R2", "R3", "R4"]]
    again = rp.load_metrics(path, flat)                       # 元のファイルが変わっていない → 読むだけ
    assert calls == [["R1", "R2", "R3", "R4"]] and len(again) == len(first)
    f = flat / "2025" / "race_result_flat.parquet"            # 新しいレースの結果が入った
    df = pd.read_parquet(f)
    pd.concat([df, pd.DataFrame(_race("R5", "2025-05-05", [("A", 1, 33.5, "1-1"), ("B", 2, 34.0, "2-2")]))]).to_parquet(f)
    added = rp.load_metrics(path, flat)
    assert calls[-1] == ["R5"] and set(added["race_id"]) == {"R1", "R2", "R3", "R4", "R5"}


def _axes_file(tmp_path, rows):
    p = tmp_path / "run_axes.parquet"
    base = {c[3:]: float("nan") for c in rp.AX_COLS}
    pd.DataFrame([{**base, "abn": False, **r} for r in rows]).to_parquet(p)
    return p


def test_axes_are_attached_by_race_and_horse_and_abnormal_runs_become_nan(flat, tmp_path):
    ax = _axes_file(tmp_path, [
        {"race_id": "R1", "horse_id": "A", "ts": 60.0, "slope": 55.0},
        {"race_id": "R2", "horse_id": "A", "ts": 99.0, "abn": True},       # 異常走 → 全軸 NaN
    ])
    m = rp.load_metrics(tmp_path / "store" / "run_metrics.parquet", flat, axes_path=ax)
    a = m.set_index(["race_id", "horse_id"])
    assert a.loc[("R1", "A"), "ax_ts"] == 60.0 and a.loc[("R1", "A"), "ax_slope"] == 55.0
    assert pd.isna(a.loc[("R2", "A"), "ax_ts"])
    assert pd.isna(a.loc[("R1", "B"), "ax_ts"])                           # 実測ラップが無い走は NaN（0 にしない）
    assert len(m) == len(rp.load_metrics(tmp_path / "store" / "run_metrics.parquet", flat, axes_path=tmp_path / "none.parquet"))


def test_missing_axes_file_gives_all_nan_columns(flat, tmp_path):
    m = rp.load_metrics(tmp_path / "store" / "run_metrics.parquet", flat, axes_path=tmp_path / "none.parquet")
    assert all(m[c].isna().all() for c in rp.AX_COLS)


def test_axis_profile_counts_only_measured_runs_and_slope_has_no_prior(flat, sires, tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "MIN_SIRE_RUNS", 1)
    ax = _axes_file(tmp_path, [
        {"race_id": "R1", "horse_id": "A", "ts": 60.0, "slope": 70.0},
        {"race_id": "R2", "horse_id": "A", "ts": 50.0},
        {"race_id": "R1", "horse_id": "B", "ts": 40.0},
    ])
    m = rp.load_metrics(tmp_path / "store" / "run_metrics.parquet", flat, axes_path=ax)
    cols = ["ax_ts", "ax_slope"]
    p = rp.profiles_as_of(m, ["A", "NEW"], "2025-04-05", sires, cols=cols).set_index("horse_id")
    assert p.at["A", "n_ax_ts"] == 2 and p.at["NEW", "n_ax_ts"] == 0
    decay, c, _ = rp.PROFILE_PARAMS["ax_ts"]
    prior = m.loc[m["date"] < "2025-04-05", "ax_ts"].mean()               # (60 + 50 + 40) / 3
    assert p.at["A", "ax_ts"] == pytest.approx((50.0 + decay * 60.0 + c * prior) / (1 + decay + c))
    assert p.at["NEW", "ax_ts"] == pytest.approx(prior) and p.at["NEW", "own_ax_ts"] == 0.0
    assert p.at["A", "ax_slope"] == pytest.approx(70.0) and p.at["A", "prior_ax_slope"] is None   # 急坂適性は自分の走だけ
    assert pd.isna(p.at["NEW", "ax_slope"])                                                         # 走が無ければ不明（父平均で埋めない）
