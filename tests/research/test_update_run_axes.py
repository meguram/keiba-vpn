"""走法の軸の日次更新（src/scripts/data/update_run_axes.py）: 一覧を取らず足りないレースだけ読む・頭数のそろわない行は足さない・試行の記録。"""
import json
from datetime import date

import pandas as pd

from src.research.race import running_profile as rp
from src.scraper import date_coverage as dc
from src.scripts.data import backfill_horse_lap as bhl
from src.scripts.data import update_run_axes as u


class _Storage:
    def __init__(self, docs, healthy=True):
        self.docs, self.loaded, self._gcs_healthy = docs, [], healthy

    def batch_list_blobs(self, *a):
        raise AssertionError("一覧を取ってはいけない")

    def load(self, cat, key):
        self.loaded.append((cat, key))
        return self.docs.get((cat, key))


def _doc(rid, n):
    return {"race_id": rid, "entries": [{"finish_order": i + 1, "horse_id": f"h{i}", "sections": [{"lap_time": 12.0}]} for i in range(n)],
            "_meta": {"x": 1}}


def _setup(tmp_path, monkeypatch, races):
    """races: [(race_id, date, 頭数)]"""
    monkeypatch.setattr(dc, "NOT_AVAILABLE_DIR", tmp_path / "meta" / "not_available")
    runs = pd.DataFrame([{"race_id": r, "date": pd.Timestamp(d)} for r, d, _ in races])
    monkeypatch.setattr(rp, "load_runs", lambda years, base=None: runs)
    monkeypatch.setattr(bhl, "expected_counts", lambda years: {r: n for r, _, n in races})
    lap = tmp_path / "horse_lap"
    lap.mkdir()
    return lap


def test_appends_complete_and_counts_missing(tmp_path, monkeypatch):
    # 11-07（土）は 11-13（金）18 時に公開 → 11-14 から窓
    lap = _setup(tmp_path, monkeypatch, [("202605010101", "2026-11-07", 2), ("202605010102", "2026-11-07", 3),
                                         ("202605010103", "2026-11-07", 2), ("202605010104", "2026-10-01", 2)])
    st = _Storage({("horse_lap", "202605010101"): _doc("202605010101", 2),
                   ("horse_lap", "202605010102"): _doc("202605010102", 2)})     # 頭数不足
    s = u.append_new(date(2026, 11, 14), [2026], st, lap_dir=lap)
    assert s == {"candidates": 3, "appended": 1, "not_found": 1, "incomplete": 1}
    rows = [json.loads(x) for x in (lap / "horse_lap_2026.jsonl").read_text().splitlines()]
    assert [r["race_id"] for r in rows] == ["202605010101"] and rows[0]["expected"] == 2 and "_meta" not in rows[0]
    from src.scraper import export_tables as et
    assert et._load_attempts("horse_lap", "2026") == {"202605010102": [1, "2026-11-14"], "202605010103": [1, "2026-11-14"]}
    # 翌日: 追記済みは読まない。間隔 1 日なので残りの 2 件を読み直す
    st2 = _Storage({})
    u.append_new(date(2026, 11, 15), [2026], st2, lap_dir=lap)
    assert sorted(k for _, k in st2.loaded) == ["202605010102", "202605010103"]


def test_before_publication_nothing_is_read(tmp_path, monkeypatch):
    lap = _setup(tmp_path, monkeypatch, [("202605010101", "2026-11-07", 2)])
    st = _Storage({})
    assert u.append_new(date(2026, 11, 13), [2026], st, lap_dir=lap)["candidates"] == 0
    assert st.loaded == []


def test_unhealthy_gcs_does_not_count(tmp_path, monkeypatch):
    lap = _setup(tmp_path, monkeypatch, [("202605010101", "2026-11-07", 2)])
    u.append_new(date(2026, 11, 14), [2026], _Storage({}, healthy=False), lap_dir=lap)
    from src.scraper import export_tables as et
    assert et._load_attempts("horse_lap", "2026") == {}


def test_lap_files_skip_incomplete_dumps(tmp_path, monkeypatch):
    lap = tmp_path / "horse_lap"
    lap.mkdir()
    for n in ("horse_lap_2025.jsonl", "horse_lap_2026.jsonl", "horse_lap_2026.incomplete_20261101_000000.jsonl"):
        (lap / n).write_text("")
    monkeypatch.setenv("KEIBA_HORSE_LAP_EXTRA", str(tmp_path / "none.jsonl"))
    assert [p.name for p in u.lap_files(lap)] == ["horse_lap_2025.jsonl", "horse_lap_2026.jsonl"]


def test_recompute_replaces_atomically(tmp_path, monkeypatch):
    from src.research.race import run_axes as ra

    monkeypatch.setattr(ra, "compute", lambda paths, years=None: pd.DataFrame({"race_id": ["r"], "horse_id": ["h"]}))
    out = tmp_path / "rp" / "run_axes.parquet"
    import os
    (tmp_path / "horse_lap_2026.jsonl").write_text("")
    os.utime(tmp_path / "horse_lap_2026.jsonl", (1000, 1000))
    monkeypatch.setenv("KEIBA_HORSE_LAP_EXTRA", str(tmp_path / "none.jsonl"))
    assert u.recompute(tmp_path, out=out) == 1
    assert out.exists() and not out.with_name(out.name + ".tmp").exists()
    assert out.stat().st_mtime == 1000          # 読み始める前の jsonl の時刻（計算中の追記を翌日に拾う）
    assert not u.needs_recompute(0, out, tmp_path)


def test_transient_gcs_failure_is_not_counted_per_read(tmp_path, monkeypatch):
    """読めなかった直後に GCS が不調の印なら、その 1 件は試行に数えない（後の読み込みが成功しても）。"""
    lap = _setup(tmp_path, monkeypatch, [("202605010101", "2026-11-07", 2), ("202605010102", "2026-11-07", 2)])

    class _Flaky(_Storage):
        def load(self, cat, key):
            self._gcs_healthy = key != "202605010101"
            return super().load(cat, key)

    u.append_new(date(2026, 11, 14), [2026], _Flaky({}), lap_dir=lap)
    from src.scraper import export_tables as et
    assert et._load_attempts("horse_lap", "2026") == {"202605010102": [1, "2026-11-14"]}


def test_read_runs_dedupes_and_skips_broken_lines(tmp_path):
    from src.research.race import run_axes as ra

    sec = [{"distance_m": 200 * (i + 1), "lap_time": 12.0, "corner": "", "position": 1} for i in range(6)]
    row = {"race_id": "r1", "entries": [{"finish_order": 1, "horse_id": "h1", "sections": sec}]}
    f = tmp_path / "horse_lap_2026.jsonl"
    empty = {"race_id": "r1", "entries": [], "empty_ok": True}
    f.write_text(json.dumps(row) + "\n" + "{broken\n" + json.dumps(row) + "\n" + json.dumps(empty) + "\n")
    assert len(ra.read_runs([f])) == 1          # 重複は 1 走、中身のある行は空の行で消さない


def test_needs_recompute_when_jsonl_is_newer(tmp_path, monkeypatch):
    import os
    lap = tmp_path / "horse_lap"
    lap.mkdir()
    monkeypatch.setenv("KEIBA_HORSE_LAP_EXTRA", str(tmp_path / "none.jsonl"))
    j = lap / "horse_lap_2026.jsonl"
    j.write_text("")
    axes = tmp_path / "run_axes.parquet"
    assert u.needs_recompute(0, axes, lap)                     # run_axes が無い
    axes.write_text("x")
    os.utime(j, (1, 1))
    assert not u.needs_recompute(0, axes, lap)                 # jsonl のほうが古い
    assert u.needs_recompute(1, axes, lap)                     # 追記があった
    os.utime(j, None)
    os.utime(axes, (2, 2))
    assert u.needs_recompute(0, axes, lap)                     # 手で回した取得が jsonl に足した
