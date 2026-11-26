"""tp48_trifecta collect-snap（確定と発走前の 3連単オッズを phase つきで parquet に）の確認。GCS は読まない。"""

import json
from argparse import Namespace

import pandas as pd

from src.scripts.data import tp48_trifecta as t


def _doc(rid: str, odds: float, phase=None, snaps=None) -> dict:
    d = {"race_id": rid, "trifecta": [[1, 2, 3, odds, 1], [2, 1, 3, odds * 2, 2]], "n_head": 3, "expected_combos": 6,
         "_meta": {"scraped_at_jst": "2026-11-23 17:35:37"}}
    if phase:
        d["phase"] = phase
        d["snapshot_at"] = "2026-11-23T17:35:37+09:00"
    if snaps:
        d["snapshots"] = snaps
    return d


def test_doc_to_phase_frames_top_and_snapshots():
    snaps = [{"phase": "t60", "snapshot_at": "2026-11-23T09:00:17+09:00", "trifecta": [[1, 2, 3, 30.0, 1]], "n_head": 3, "expected_combos": 6},
             {"phase": "t15", "snapshot_at": "2026-11-23T09:30:32+09:00", "trifecta": [[1, 2, 3, 25.0, 1]]},
             {"phase": "final", "snapshot_at": "old", "trifecta": [[1, 2, 3, 99.0, 1]]}]      # 最上位と同じ段階は最上位を採る
    fs = t.doc_to_phase_frames(_doc("202605050611", 20.0, "final", snaps), "gcs")
    got = {f["phase"].iat[0]: f for f in fs}
    assert set(got) == {"final", "t60", "t15"}
    assert got["final"]["odds"].iat[0] == 20.0 and got["final"]["scraped_at_jst"].iat[0] == "2026-11-23 17:35:37"
    assert got["t60"]["snapshot_at"].iat[0] == "2026-11-23T09:00:17+09:00" and got["t60"]["odds"].iat[0] == 30.0
    assert got["t15"]["expected"].iat[0] == 6        # 段階に無い値は文書の値を使う


def test_doc_without_phase_is_backfilled_final():
    fs = t.doc_to_phase_frames(_doc("201610021212", 12.5), "mirror")
    assert len(fs) == 1 and fs[0]["phase"].iat[0] == "final" and fs[0]["phase_raw"].iat[0] is None


def _base(tmp_path, year: int, rids: list[str], date: str | None = None):
    tb = tmp_path / "data/page_reference/tables" / str(year)
    tb.mkdir(parents=True)
    pay = json.dumps({"三連単": {"numbers": "1 → 2 → 3", "payout": "1,250", "popularity": "1"}}, ensure_ascii=False)
    pd.DataFrame({"race_id": rids, "date": [date or f"{year}-06-01"] * len(rids), "payoff": [pay] * len(rids)}).to_parquet(tb / "race_result_flat.parquet")
    return tmp_path


def test_collect_snap_mirror_then_old_parquet_no_gcs(tmp_path):
    rids = ["201610021210", "201610021211", "201610021212"]
    base = _base(tmp_path, 2016, rids)
    mp = t._mirror_path(base, rids[0])
    mp.parent.mkdir(parents=True)
    mp.write_text(json.dumps(_doc(rids[0], 12.5)), encoding="utf-8")
    old = base / "data/local/modeling/betting/tp48"
    old.mkdir(parents=True)
    f = t.doc_to_frame(_doc(rids[1], 12.5))
    f.assign(n_head=3, expected=6).to_parquet(old / "trifecta_2016.parquet")
    a = Namespace(base=str(base), out="", years=[2016], force=False, no_gcs=True, max_gcs_reads=0, max_gcs_errors=20, stop_at="")
    t.collect_snap(a)
    out = base / t.SNAP_OUT
    d = pd.read_parquet(out / "trifecta_2016.parquet")
    assert list(d.columns) == t.SNAP_COLS
    assert dict(d.drop_duplicates("race_id").set_index("race_id")["source"].astype(str)) == {rids[0]: "mirror", rids[1]: "tp48_20261020"}
    assert json.loads((out / "missing_2016.json").read_text()) == [rids[2]]
    rep = json.loads((out / "collect_2016.json").read_text())
    assert rep["gcs_reads_total"] == 0 and rep["races_by_phase"] == {"final": 2}
    # 2 回目は飛ばす（出力が既にある）
    (out / "trifecta_2016.parquet").write_bytes(b"keep")
    t.collect_snap(a)
    assert (out / "trifecta_2016.parquet").read_bytes() == b"keep"


def test_collect_snap_read_limit_stops_before_writing(tmp_path):
    base = _base(tmp_path, 2016, ["201610021210"])
    a = Namespace(base=str(base), out="", years=[2016], force=False, no_gcs=False, max_gcs_reads=0, max_gcs_errors=20, stop_at="")
    try:
        t.collect_snap(a)
        raise AssertionError("止まるはず")
    except SystemExit as e:
        assert e.code == 3
    assert not (base / t.SNAP_OUT / "trifecta_2016.parquet").exists()


def test_qa_on_snap_output_uses_final_and_tables_payoff(tmp_path):
    rids = ["201610021210"]
    base = _base(tmp_path, 2016, rids)
    out = base / t.SNAP_OUT
    out.mkdir(parents=True)
    fs = t.doc_to_phase_frames(_doc(rids[0], 12.5, "final", [{"phase": "t60", "trifecta": [[1, 2, 3, 40.0, 1]]}]), "gcs")
    pd.concat(fs)[t.SNAP_COLS].assign(race_id=rids[0]).to_parquet(out / "trifecta_2016.parquet")
    old_tables, old_root = t.TABLES, t.ROOT
    t.TABLES, t.ROOT = base / "data/page_reference/tables", base
    try:
        t.qa(Namespace(years=[2016], dir=str(out)))
    finally:
        t.TABLES, t.ROOT = old_tables, old_root
    res = json.loads((out / "qa.json").read_text())["2016"]
    assert res["races"] == 1 and res["hit_checked"] == 1 and res["hit_exact_rate"] == 1.0      # 12.5 倍 × 100 = 1,250 円（t60 の 40 倍は使わない）


def _args(base, **kw):
    d = dict(base=str(base), out="", years=[2026], force=False, no_gcs=False, max_gcs_reads=100, max_gcs_errors=20, stop_at="")
    d.update(kw)
    return Namespace(**d)


def _write_mirror(base, rid, doc):
    mp = t._mirror_path(base, rid)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(doc), encoding="utf-8")


def test_mirror_without_final_is_completed_from_old_or_recorded(tmp_path):
    rids = ["202601010101", "202601010102"]
    base = _base(tmp_path, 2026, rids, date="2026-06-01")
    for r in rids:
        _write_mirror(base, r, _doc(r, 9.0, "t15"))
    old = base / "data/local/modeling/betting/tp48"
    old.mkdir(parents=True)
    t.doc_to_frame(_doc(rids[0], 12.5)).assign(n_head=3, expected=6).to_parquet(old / "trifecta_2026.parquet")
    t.collect_snap(_args(base, no_gcs=True))
    out = base / t.SNAP_OUT
    d = pd.read_parquet(out / "trifecta_2026.parquet")
    r0 = d[d["race_id"] == rids[0]].drop_duplicates("phase").set_index("phase")
    assert set(r0.index.astype(str)) == {"t15", "final"} and r0.loc["final", "odds"] == 12.5
    assert json.loads((out / "no_final_2026.json").read_text()) == [rids[1]]


def test_old_parquet_not_used_after_live_start(tmp_path):
    rid = "202605050611"
    base = _base(tmp_path, 2026, [rid], date="2026-11-23")
    old = base / "data/local/modeling/betting/tp48"
    old.mkdir(parents=True)
    t.doc_to_frame(_doc(rid, 12.5)).assign(n_head=3, expected=6).to_parquet(old / "trifecta_2026.parquet")
    t.collect_snap(_args(base, no_gcs=True))
    assert json.loads((base / t.SNAP_OUT / "missing_2026.json").read_text()) == [rid]


class _Blob:
    def __init__(self, store, name):
        self.store, self.name = store, name

    def download_as_bytes(self, timeout=None, retry=None):
        from google.api_core.exceptions import NotFound

        self.store["calls"] += 1
        v = self.store["docs"].get(self.name.rsplit("/", 1)[-1][:-5])
        if v is None:
            raise NotFound("x")
        if isinstance(v, Exception):
            raise v
        return v


class _Bucket:
    def __init__(self, store):
        self.store = store

    def blob(self, name):
        return _Blob(self.store, name)


def test_gcs_read_cached_in_docs_and_errors_counted(tmp_path, monkeypatch):
    rids = ["202601010101", "202601010102", "202601010103"]
    base = _base(tmp_path, 2026, rids)
    store = {"calls": 0, "docs": {rids[0]: json.dumps(_doc(rids[0], 12.5, "final")).encode(), rids[2]: RuntimeError("503")}}
    monkeypatch.setattr(t, "_open_bucket", lambda b: (_Bucket(store), "pc", None))
    t.collect_snap(_args(base))
    out = base / t.SNAP_OUT
    rep = json.loads((out / "collect_2026.json").read_text())
    assert rep["source"]["gcs"] == 1 and rep["gcs_errors_total"] == 1 and rep["missing"] == 2
    assert (out / "_docs/2026" / f"{rids[0]}.json").exists() and (out / "_docs/2026" / f"{rids[1]}.404").exists()
    assert json.loads((out / "errors_2026.json").read_text()) == [rids[2]]
    assert not (out / "trifecta_2026.parquet").exists()      # 一時的な失敗が残る年は書かない
    calls = store["calls"]
    store["docs"][rids[2]] = json.dumps(_doc(rids[2], 7.0, "t15")).encode()      # 2 回目は失敗したレースだけ GCS を読む
    t.collect_snap(_args(base))
    assert store["calls"] == calls + 1
    rep = json.loads((out / "collect_2026.json").read_text())
    assert rep["source"]["docs"] == 1 and rep["source"]["gcs"] == 1 and rep["errors"] == 0 and rep["missing"] == 1 and rep["no_final"] == 1
    assert (out / "trifecta_2026.parquet").exists()
    calls = store["calls"]
    t.collect_snap(_args(base, force=True))      # 3 回目: 確定の無い文書も 404 も _docs にあるので GCS は読まない
    assert store["calls"] == calls


def test_mirror_final_only_does_not_hide_docs_snapshots(tmp_path):
    rid = "202605050611"
    base = _base(tmp_path, 2026, [rid], date="2026-11-23")
    _write_mirror(base, rid, _doc(rid, 20.0))
    dp = base / t.SNAP_OUT / "_docs/2026" / f"{rid}.json"
    dp.parent.mkdir(parents=True)
    dp.write_text(json.dumps(_doc(rid, 20.0, "final", [{"phase": "t60", "trifecta": [[1, 2, 3, 30.0, 1]]}])), encoding="utf-8")
    t.collect_snap(_args(base, no_gcs=True))
    d = pd.read_parquet(base / t.SNAP_OUT / "trifecta_2026.parquet")
    assert set(d["phase"].astype(str)) == {"final", "t60"}


def test_gcs_errors_stop_before_writing(tmp_path, monkeypatch):
    rids = ["202601010101", "202601010102"]
    base = _base(tmp_path, 2026, rids)
    store = {"calls": 0, "docs": {r: RuntimeError("503") for r in rids}}
    monkeypatch.setattr(t, "_open_bucket", lambda b: (_Bucket(store), "pc", None))
    try:
        t.collect_snap(_args(base, max_gcs_errors=2))
        raise AssertionError("止まるはず")
    except SystemExit as e:
        assert e.code == 4
    assert not (base / t.SNAP_OUT / "trifecta_2026.parquet").exists()
