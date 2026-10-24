"""出遅れ率（実ラベル）の特徴量: 先読みが無いこと（ラベルを将来側で書き換えても、その日以前の特徴量が 1 ビットも変わらない）、ラベル不明の走は欠損。"""

import numpy as np
import pytest
import pandas as pd

from src.pipeline.features import factory_slowstart as fs
from src.pipeline.features import slowstart_real as SR


def _synthetic(seed=0, days=70, races_per_day=3, field=9):
    rng = np.random.default_rng(seed)
    rows = []
    rid = 0
    for d in range(days):
        date = pd.Timestamp("2022-01-01") + pd.Timedelta(days=7 * d)
        for _ in range(races_per_day):
            rid += 1
            horses = rng.choice(60, size=field, replace=False)
            fin = rng.permutation(field) + 1
            for i, h in enumerate(horses):
                rows.append({"race_id": f"R{rid:05d}", "horse_id": f"H{h:03d}", "date": date, "horse_number": i + 1, "field_size": field, "distance": int(rng.choice([1200, 1600, 2000])),
                             "surface": str(rng.choice(["芝", "ダ"])), "venue_code": str(rng.choice(["05", "06"])), "sex_age": str(rng.choice(["牡3", "牝4", "牡5"])),
                             "weight": float(rng.integers(420, 540)), "bracket_number": i // 2 + 1, "jockey_id": f"J{rng.integers(0, 8)}", "trainer_id": f"T{rng.integers(0, 6)}",
                             "finish_position": int(fin[i])})
    f = pd.DataFrame(rows)
    f["slow_rem"] = (rng.random(len(f)) < 0.05).astype(float)
    parents = pd.DataFrame({"horse_id": [f"H{h:03d}" for h in range(60)], "sire_id": [f"S{h % 7}" for h in range(60)], "dam_sire_id": [f"D{h % 5}" for h in range(60)]})
    return f, parents


def _build(f, parents):
    f = fs.prepare(f)
    ev = f[["horse_id", "date", SR.LABEL]].copy()
    return SR.build(f, ev, parents)


def _same(a, b):
    return np.array_equal(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True)


@pytest.mark.parametrize("cut", ["2022-07-01", "2022-07-16", "2022-03-20"])      # 月の途中で切ると、同じ月の将来のラベルが全体の率に混ざる先読みも検出できる
def test_future_labels_do_not_change_past_features_exactly(cut, monkeypatch):
    monkeypatch.setattr(SR, "MIN_BASE_EVENTS", 30)
    f, parents = _synthetic()
    g0 = _build(f, parents)
    cut = pd.Timestamp(cut)
    f2 = f.copy()
    rng = np.random.default_rng(1)
    late = (f2["date"] >= cut).to_numpy()
    f2.loc[late, SR.LABEL] = rng.integers(0, 2, late.sum()).astype(float)          # cut 以降（当日を含む）のラベルを全部書き換える
    g1 = _build(f2, parents)
    past = (f["date"] <= cut).to_numpy()                                              # cut 当日の特徴量も、当日のラベルに依存してはいけない
    for k in g0:
        assert list(g0[k].columns) == list(g1[k].columns)
        assert _same(g0[k][past], g1[k][past]), k
    future = ~past
    assert any(not _same(g0[k][future], g1[k][future]) for k in g0)                   # 書き換えが将来側の特徴量には効いている（テストが空振りでない）


def test_same_day_other_races_do_not_leak(monkeypatch):
    monkeypatch.setattr(SR, "MIN_BASE_EVENTS", 30)
    f, parents = _synthetic(seed=3)
    f = f.sort_values(["date", "race_id"]).reset_index(drop=True)
    g0 = _build(f, parents)
    day = f["date"].iloc[len(f) // 2]
    f2 = f.copy()
    f2.loc[f2["date"] == day, SR.LABEL] = 1.0 - f2.loc[f2["date"] == day, SR.LABEL]
    g1 = _build(f2, parents)
    on_day = (f["date"] == day).to_numpy()
    for k in g0:
        assert _same(g0[k][on_day], g1[k][on_day]), k


def test_unknown_label_rows_are_all_nan_and_group_counts_dropped(monkeypatch):
    monkeypatch.setattr(SR, "MIN_BASE_EVENTS", 30)
    f, parents = _synthetic(seed=5)
    unk = (f["horse_id"].isin([f"H{h:03d}" for h in range(10)])).to_numpy()
    f.loc[unk, SR.LABEL] = np.nan
    g = _build(f, parents)
    for k, v in g.items():
        assert v[unk].isna().all().all(), k
    cols = [c for v in g.values() for c in v.columns]
    assert not [c for c in cols if c.endswith("_n") and c != "ss_h_n"]


def test_parse_remarks_and_row_labels():
    rows = SR.parse_remarks("H1", [{"race_id": "A", "date": "2024/01/01", "remarks": "出遅れ", "start_index": 80}, {"race_id": "B", "date": "2024/02/01", "remarks": ""},
                                  {"race_id": "C", "date": "2024/03/01", "remarks": "S不利"}])
    rem = pd.DataFrame(rows)
    assert rem["slow_rem"].tolist() == [1.0, 0.0, 0.0] and rem["trouble_rem"].tolist() == [1.0 * 0, 0.0, 1.0]
    frame = pd.DataFrame({"race_id": ["A", "B", "Z"], "horse_id": ["H1", "H1", "H1"]})
    lab = SR.row_labels(frame, rem)
    assert lab.iloc[:2].tolist() == [1.0, 0.0] and np.isnan(lab.iloc[2])
    frame2 = pd.DataFrame({"race_id": ["A"], "horse_id": ["H9"]})
    assert SR.row_labels(frame2, rem).isna().all()


def test_monthly_base_uses_only_previous_months():
    ev = pd.DataFrame({"date": pd.to_datetime(["2022-01-10"] * 40 + ["2022-02-10"] * 40 + ["2022-03-10"] * 40), SR.LABEL: [0.0] * 40 + [1.0] * 40 + [0.0] * 40})
    b = SR.monthly_base(ev, pd.Series(pd.to_datetime(["2022-02-20", "2022-03-20", "2022-04-20"])), min_events=30)
    assert b[0] == 0.0 and b[1] == 0.5 and abs(b[2] - 1 / 3) < 1e-12              # 2 月は 1 月だけ、3 月は 1〜2 月、4 月は 1〜3 月（当月は含めない）
