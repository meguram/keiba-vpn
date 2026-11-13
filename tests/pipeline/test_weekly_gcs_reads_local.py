"""金曜の週次で GCS を大量に読まない（2026-11-13）: 馬名インデックスはローカルの flat から、成長曲線は前回からの差分の馬だけ。"""
import json

import pandas as pd


def _tables(tmp_path):
    d = tmp_path / "tables" / "2026"
    d.mkdir(parents=True)
    pd.DataFrame({"horse_id": ["2022100001", "2022100002"], "horse_name": ["アカ", "アオ"]}).to_parquet(d / "race_shutuba_flat.parquet")
    pd.DataFrame({"horse_id": ["2022100001", "2022100003"], "horse_name": ["アカ", "シロ"], "date": ["2026-11-08", "2026-11-01"],
                  "finish_position": [1, 3]}).to_parquet(d / "race_result_flat.parquet")
    return tmp_path / "tables"


def test_horse_name_index_from_local_flat_keeps_existing_and_reads_no_gcs(tmp_path, monkeypatch):
    from src.utils import horse_name_index as H

    tables = _tables(tmp_path)
    written = {}
    monkeypatch.setattr(H, "load_horse_name_index", lambda base: {"horses": [{"horse_id": "2010100009", "horse_name": "フルイ", "name_en": "Old"}]})
    monkeypatch.setattr(H, "_write_horse_name_index_bundle", lambda base, out: written.update(out))
    import src.scraper.storage as S
    monkeypatch.setattr(S, "HybridStorage", lambda *a, **k: (_ for _ in ()).throw(AssertionError("GCS を使ってはいけない")))
    n = H.rebuild_horse_name_index_from_local_flat(tmp_path, tables_dir=tables)
    ids = {h["horse_id"]: h for h in written["horses"]}
    assert n == 4 and set(ids) == {"2010100009", "2022100001", "2022100002", "2022100003"}
    assert ids["2010100009"]["name_en"] == "Old" and written["source"] == "local_flat"     # 既存の馬は消えない・項目も残る


def test_growth_curve_targets_are_the_diff_since_last_compute(tmp_path, monkeypatch):
    from src.pipeline.inference import growth_curve_service as G

    tables = _tables(tmp_path)
    flat_last = G.latest_finished_race_dates(tables)
    assert flat_last == {"2022100001": "2026-11-08", "2022100003": "2026-11-01"}
    saved = {
        "2022100001": {"races": [{"date": "2026/10/25"}]},       # 11/08 に走った → 作り直す（日数の窓ではなく前回からの差分）
        "2022100003": {"races": [{"date": "2026/11/01"}]},       # 変わっていない → 読まない
    }
    monkeypatch.setattr(G, "load_local", lambda hid: saved.get(hid))
    got = G.horses_needing_update(["2022100001", "2022100002", "2022100003"], flat_last)
    assert got == ["2022100001", "2022100002"]                  # 2022100002 は曲線が無いので作る


def test_upcoming_starters_and_failed_backoff(tmp_path, monkeypatch):
    """地方・海外を走った馬は flat に無いので、中央の出馬表に載った時点で作り直す。作れない馬は 28 日あける。"""
    from src.pipeline.inference import growth_curve_service as G

    d = tmp_path / "tables" / "2026"
    d.mkdir(parents=True)
    pd.DataFrame({"horse_id": ["2022100005", "2022100006"], "date": ["2026-11-15", "2026-11-01"]}).to_parquet(d / "race_shutuba_flat.parquet")
    up = G.upcoming_starters(tmp_path / "tables", today="2026-11-13")
    assert up == {"2022100005"}
    monkeypatch.setattr(G, "load_local", lambda hid: {"races": [{"date": "2026/10/01"}]} if hid == "2022100005" else None)
    now = 1_000_000_000.0
    failed = {"2022100007": now - 86400, "2022100008": now - 40 * 86400}
    got = G.horses_needing_update(["2022100005", "2022100007", "2022100008"], {}, up, failed, now=now)
    assert got == ["2022100005", "2022100008"]          # 出馬表に載った馬・28 日たった失敗は試す。最近の失敗は飛ばす
