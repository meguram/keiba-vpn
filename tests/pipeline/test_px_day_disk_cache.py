"""v3_serving._px_for_day の日単位キャッシュを、別プロセス（Jobs の次のレース）でも使えるようにディスクへ残す。"""
from types import SimpleNamespace

import pandas as pd

from src.pipeline.inference import v3_serving as vs


def _srv(tmp_path, calls):
    s = vs.V3Serving.__new__(vs.V3Serving)
    s._px_cache = None
    s._card_day = {}
    s.v4 = SimpleNamespace(state_dir=tmp_path, hist=SimpleNamespace(field_hist=None))

    class FakePX:
        def day(self, card):
            calls.append(len(card))
            return card[vs.KEYS].assign(px_a=1.0)

    s.px = FakePX()
    s._day_card = lambda c, storage: None
    return s


def _card(ids):
    return pd.DataFrame({"race_id": ["202605010101"] * len(ids), "horse_id": ids, "date": ["2026-10-25"] * len(ids)})


def test_px_day_is_computed_once_and_reused_from_disk_by_a_new_process(tmp_path):
    calls = []
    s1 = _srv(tmp_path, calls)
    out1 = s1._px_for_day(_card(["h1", "h2"]), None)
    assert calls == [2] and (tmp_path / "2026-10-25" / "px_day.parquet").exists()
    s1._px_for_day(_card(["h1"]), None)                              # 同じプロセスはメモリから
    assert calls == [2]
    s2 = _srv(tmp_path, calls)                                      # 別プロセス相当（メモリは空）→ ディスクから。計算しない
    out2 = s2._px_for_day(_card(["h1", "h2"]), None)
    assert calls == [2] and sorted(out2["horse_id"]) == sorted(out1["horse_id"])
    s2._px_for_day(_card(["h1", "h3"]), None)                       # キャッシュに無い馬が来たら作り直す
    assert len(calls) == 2


def test_broken_cache_file_is_recomputed(tmp_path):
    calls = []
    p = tmp_path / "2026-10-25"
    p.mkdir(parents=True)
    (p / "px_day.parquet").write_bytes(b"not parquet")
    s = _srv(tmp_path, calls)
    s._px_for_day(_card(["h1"]), None)
    assert calls == [1]
