"""事前生成（src/api/prerender.py・prewarm.py）: 期限・世代・レース単位の変更の目印で読まなくなる／変更があったレースだけ作り直す。"""

import asyncio
import json
import os
import time
from datetime import datetime, timedelta, timezone

import pytest
from starlette.responses import JSONResponse

from src.api import prerender, prewarm
from src.utils import cache_epoch

JST = timezone(timedelta(hours=9))
RID = "202605030201"
RID2 = "202605030202"


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_PREWARM", "1")
    monkeypatch.setenv("KEIBA_PRERENDER_DIR", str(tmp_path / "pre"))
    monkeypatch.setattr(cache_epoch, "_FILE", tmp_path / "epoch.json")
    monkeypatch.setattr(cache_epoch, "_HARD_FILE", tmp_path / "epoch_hard.json")
    cache_epoch.bump("test")
    return tmp_path


def _put(kind, key, deps=(), max_age=300.0, since_ns=None):
    prerender.put(kind, key, b'{"x":1}', since_ns=since_ns if since_ns is not None else time.time_ns(), hard=cache_epoch.hard_current(), deps=deps, max_age=max_age)


def test_disabled_by_default_reads_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("KEIBA_PREWARM", raising=False)
    monkeypatch.setenv("KEIBA_PRERENDER_DIR", str(tmp_path / "pre"))
    _put("k", "a")
    assert prerender.get("k", "a") is None
    assert prerender.mark_dirty([RID]) == 0
    assert not (tmp_path / "pre").exists()


def test_saving_one_race_invalidates_only_that_race(store):
    _put("predictions", RID, deps=[RID])
    _put("predictions", RID2, deps=[RID2])
    assert prerender.get("predictions", RID) == b'{"x":1}'
    prerender.mark_dirty([RID])
    assert prerender.get("predictions", RID) is None
    assert prerender.get("predictions", RID2) == b'{"x":1}'          # 他のレースは有効のまま


def test_date_level_entry_depends_on_its_races_and_date(store):
    _put("race_values", "20261025_x", deps=["20261025", RID, RID2])
    assert prerender.get("race_values", "20261025_x") is not None
    prerender.mark_dirty([RID2])
    assert prerender.get("race_values", "20261025_x") is None
    _put("race_values", "20261025_x", deps=["20261025", RID, RID2])
    assert prerender.get("race_values", "20261025_x") is not None
    prerender.mark_dirty(["20261025"])
    assert prerender.get("race_values", "20261025_x") is None


def test_save_during_compute_invalidates_the_result(store):
    since = time.time_ns()
    prerender.mark_dirty([RID])                                         # 計算を始めたあとに保存された
    _put("predictions", RID, deps=[RID], since_ns=since)
    assert prerender.get("predictions", RID) is None
    _put("predictions", RID, deps=[RID])                                # 保存のあとに作り始めたものは有効
    assert prerender.get("predictions", RID) is not None


def test_hard_epoch_and_expiry_invalidate_everything(store):
    _put("predictions", RID, deps=[RID], max_age=0.01)
    time.sleep(0.05)
    assert prerender.get("predictions", RID) is None
    _put("predictions", RID, deps=[RID])
    cache_epoch.bump("save:race_odds")                                  # 保存の世代では無効にならない
    assert prerender.get("predictions", RID) is not None
    os.utime(cache_epoch._HARD_FILE, ns=(1, 1))
    cache_epoch.bump("deploy")                                          # 配備・手動クリアは全部無効
    assert prerender.get("predictions", RID) is None


def test_mark_from_save_filters_keys_and_old_years(store):
    year = datetime.now().year
    prerender.mark_from_save("race_odds", f"{year}05030201")
    prerender.mark_from_save("race_lists", f"{year}1025")
    prerender.mark_from_save("horse_profile", "2020100001")              # 馬 ID（10 桁）は対象外
    prerender.mark_from_save("race_result", "201905030201")             # 古い年の補填は対象外
    names = set(os.listdir(prerender._dirty_dir()))
    assert names == {f"{year}05030201", f"{year}1025"}


def test_dirty_between_and_prune(store):
    t0 = time.time_ns()
    prerender.mark_dirty([RID])
    t1 = time.time_ns()
    prerender.mark_dirty([RID2])
    assert prerender.dirty_between(t0, t1) == [RID]
    assert set(prerender.dirty_between(t0, time.time_ns())) == {RID, RID2}
    assert prerender.prune_dirty(older_than_sec=-1) == 2


def test_cached_response_computes_once_skips_non_200_and_follows_deps(store):
    calls = []

    async def ok():
        calls.append(1)
        return JSONResponse({"a": 1})

    async def nf():
        calls.append(1)
        return JSONResponse({"status": "not_found"}, status_code=404)

    async def run():
        await prerender.cached_response("p", RID, ok)
        await prerender.cached_response("p", RID, ok)
        await prerender.cached_response("p", "n", nf)
        await prerender.cached_response("p", "n", nf)
        prerender.mark_dirty([RID])
        return await prerender.cached_response("p", RID, ok)

    r = asyncio.run(run())
    assert json.loads(r.body) == {"a": 1}
    assert len(calls) == 4            # ok: 最初の 1 回 + 保存後の 1 回、404 は毎回（置かない）


def test_cached_json_with_race_ids_deps(store):
    n = []

    def comp():
        n.append(1)
        return {"races": [{"race_id": RID}]}

    d = "20261025"
    assert prerender.cached_json("v", "k", comp, deps=prerender.race_ids_deps(d)) == {"races": [{"race_id": RID}]}
    prerender.cached_json("v", "k", comp, deps=prerender.race_ids_deps(d))
    assert len(n) == 1
    prerender.mark_dirty([RID])
    prerender.cached_json("v", "k", comp, deps=prerender.race_ids_deps(d))
    assert len(n) == 2


def test_select_dates_hot_cold_and_window():
    today = datetime(2026, 10, 27, tzinfo=JST)
    stems = ["20261025", "20261024", "20261031", "20261101", "20261020", "20261010", "20261001", "20261120", "bad"]
    hot, cold = prewarm.select_dates(stems, today, {**prewarm._DEFAULTS})
    assert hot == ["20261025", "20261024", "20261031", "20261101"]
    assert cold == ["20261020"]


class FakeApp:
    """app.py の計算の代わり。作り直した回数を数え、本物と同じく prerender に置く。"""

    def __init__(self, day, races):
        self.day, self.races = day, races
        self.built: list[str] = []
        self.values = 0

    def hooks(self, today):
        async def race_list(date, max_age):
            if prewarm.stale("race_list", f"{date}_1", max_age, True):
                since, hard = time.time_ns(), cache_epoch.hard_current()
                prerender.put("race_list", f"{date}_1", json.dumps({"races": [{"race_id": r} for r in self.races]}).encode(), since_ns=since, hard=hard, deps=[date, *self.races], max_age=max_age)
            return list(self.races)

        async def warm_race(rid, max_age, only):
            if not prewarm.stale("predictions", rid, max_age, only):
                return 0
            prerender.put("predictions", rid, b"{}", since_ns=time.time_ns(), hard=cache_epoch.hard_current(), deps=[rid], max_age=max_age)
            self.built.append(rid)
            return 1

        def warm_values(date, view, max_age, only):
            if not prewarm.stale("race_values", f"{date}_{view}", max_age, only):
                return 0
            prerender.put("race_values", f"{date}_{view}", b"{}", since_ns=time.time_ns(), hard=cache_epoch.hard_current(), deps=[date, *self.races], max_age=max_age)
            self.values += 1
            return 1

        return prewarm.Hooks(stems=lambda: [self.day], race_list=race_list, warm_race=warm_race, warm_values=warm_values, now=lambda: today)


def test_cycle_then_only_changed_race_is_rebuilt(store):
    today = datetime.now(JST)
    app = FakeApp(today.strftime("%Y%m%d"), [RID, RID2])
    w = prewarm.Warmer(app.hooks(today), sleep=prewarm._nosleep)
    cfg = {**prewarm._DEFAULTS}
    st = asyncio.run(w.run_cycle(cfg))
    assert (st.dates, st.races) == (1, 2) and app.built == [RID, RID2] and app.values == 2
    assert w.race_dates == {RID: app.day, RID2: app.day}

    assert asyncio.run(w.run_cycle(cfg)).rebuilt == 0                 # 何も変わっていなければ作り直さない

    app.built.clear()
    app.values = 0
    prerender.mark_dirty([RID2])                                      # 予測が出力された／出馬表が再取得された
    st = asyncio.run(w.handle_events([RID2], cfg))
    assert app.built == [RID2]                                        # 変わったレースだけ
    assert app.values == 2                                            # その開催日の期待値表示（paid / full）は作り直す
    assert prerender.get("predictions", RID) is not None and prerender.get("predictions", RID2) is not None

    # 窓の外のレース（過去分の補填）は何もしない
    app.built.clear()
    st = asyncio.run(w.handle_events(["202001010101"], cfg))
    assert app.built == [] and st.rebuilt == 0


def test_handle_date_event_refreshes_list_and_picks_up_new_races(store):
    today = datetime.now(JST)
    app = FakeApp(today.strftime("%Y%m%d"), [RID])
    w = prewarm.Warmer(app.hooks(today), sleep=prewarm._nosleep)
    cfg = {**prewarm._DEFAULTS}
    asyncio.run(w.run_cycle(cfg))
    app.races = [RID, RID2]                                           # race_lists が更新されてレースが増えた
    prerender.mark_dirty([app.day])
    asyncio.run(w.handle_events([app.day], cfg))
    assert RID2 in app.built and w.race_dates[RID2] == app.day


def test_run_cycle_survives_a_failing_date(store):
    today = datetime.now(JST)
    ds = [today.strftime("%Y%m%d"), (today + timedelta(days=1)).strftime("%Y%m%d")]

    async def race_list(date, max_age):
        if date == ds[0]:
            raise RuntimeError("boom")
        return ["r"]

    async def warm_race(*a):
        return 1

    hooks = prewarm.Hooks(stems=lambda: ds, race_list=race_list, warm_race=warm_race, warm_values=lambda *a: 0, now=lambda: today)
    st = asyncio.run(prewarm.Warmer(hooks, sleep=prewarm._nosleep).run_cycle({**prewarm._DEFAULTS}))
    assert st.errors == 1 and st.dates == 1 and st.rebuilt == 1


def test_scan_finds_updated_prediction_and_state_files(store, tmp_path):
    inf = tmp_path / "inf"
    day = "20261025"
    (inf / "predictions_v6" / day).mkdir(parents=True)
    (inf / "predictions_backfill" / "bundle_v6_2025" / day).mkdir(parents=True)
    (inf / "lap_forecast" / "2026-10-25").mkdir(parents=True)
    old = inf / "predictions_v6" / day / f"{RID2}.parquet"
    old.write_bytes(b"x")
    os.utime(old, ns=(1, 1))
    since = time.time_ns() - 1
    (inf / "predictions_v6" / day / f"{RID}.parquet").write_bytes(b"x")
    (inf / "predictions_backfill" / "bundle_v6_2025" / day / f"{RID2}.parquet").write_bytes(b"x")
    (inf / "lap_forecast" / "2026-10-25" / f"{RID}.json").write_text("{}")
    (inf / "predictions_v6" / "20260101").mkdir()
    (inf / "predictions_v6" / "20260101" / f"{RID}.parquet").write_bytes(b"x")     # 対象日の外
    assert sorted(set(prewarm._scan_inference(inf, {day}, since))) == [RID, RID2]
    state = tmp_path / "state"
    state.mkdir()
    (state / f"{RID}.json").write_text("{}")
    (state / "202601010101.json").write_text("{}")
    assert prewarm._scan_flat(state, {RID: day}, since) == [RID]


def test_try_lead_is_exclusive(tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_PREWARM_LOCK", str(tmp_path / "l.lock"))
    monkeypatch.setattr(prewarm, "_lock_fd", None)
    assert prewarm.try_lead() is True
    import fcntl

    fd = os.open(tmp_path / "l.lock", os.O_RDWR)
    with pytest.raises(OSError):
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.close(fd)
    os.close(prewarm._lock_fd)
    monkeypatch.setattr(prewarm, "_lock_fd", None)


def test_cached_json_serves_stale_then_refreshes_in_background(store):
    calls = []

    def compute():
        calls.append(1)
        return {"v": len(calls)}

    kw = dict(deps=[RID], max_age=300.0, stale_ok_sec=1800)
    assert prerender.cached_json("sv", "k", compute, **kw) == {"v": 1}
    assert prerender.cached_json("sv", "k", compute, **kw) == {"v": 1} and len(calls) == 1
    prerender.mark_dirty([RID])                                  # データ更新: 通常は無効になる
    assert prerender.get("sv", "k") is None
    assert prerender.cached_json("sv", "k", compute, **kw) == {"v": 1}      # 古い版を先に返す
    for _ in range(100):
        if len(calls) == 2 and prerender.get("sv", "k") is not None:
            break
        time.sleep(0.02)
    assert prerender.cached_json("sv", "k", compute, **kw) == {"v": 2}      # 裏で作り直された


def test_stale_is_not_served_across_hard_epoch_or_beyond_limit(store):
    prerender.cached_json("sv", "k2", lambda: {"v": 1}, stale_ok_sec=1800)
    prerender.mark_dirty([ "202605030299"])
    assert prerender.get_stale("sv", "k2", 1800) is not None
    assert prerender.get_stale("sv", "k2", 0.0) is None
    cache_epoch.bump("deploy")                                   # 配備: 古い版は返さない
    assert prerender.get_stale("sv", "k2", 1800) is None


def test_store_only_flag_enables_cache_but_not_the_warmer(tmp_path, monkeypatch):
    monkeypatch.delenv("KEIBA_PREWARM", raising=False)
    monkeypatch.setenv("KEIBA_PRERENDER", "1")
    monkeypatch.setenv("KEIBA_PRERENDER_DIR", str(tmp_path / "pre"))
    monkeypatch.setenv("KEIBA_PRERENDER_MAX_AGE_SEC", "7")
    monkeypatch.setattr(cache_epoch, "_FILE", tmp_path / "epoch.json")
    monkeypatch.setattr(cache_epoch, "_HARD_FILE", tmp_path / "epoch_hard.json")
    assert prerender.enabled() and not prerender.warmer_enabled() and prerender.default_max_age() == 7.0
    prerender.put("k", "a", b"{}", since_ns=time.time_ns(), hard=cache_epoch.hard_current())
    assert prerender.get("k", "a") == b"{}" and prerender._read("k", "a")[0]["max_age"] == 7.0


def test_concurrent_identical_misses_compute_once(store):
    calls = []

    async def compute():
        calls.append(1)
        await asyncio.sleep(0.05)
        return JSONResponse({"v": len(calls)})

    async def go():
        rs = await asyncio.gather(*[prerender.cached_response("sf", "x", compute, deps=[RID]) for _ in range(10)])
        return [json.loads(r.body) for r in rs]

    out = asyncio.run(go())
    assert len(calls) == 1 and all(o == {"v": 1} for o in out)


def test_prune_size_removes_oldest_first(store):
    for i in range(5):
        prerender.put("pr", f"k{i}", b"x" * 1000, since_ns=time.time_ns(), hard=cache_epoch.hard_current())
        os.utime(prerender._path("pr", f"k{i}"), (1000 + i, 1000 + i))
    assert prerender.prune_size(max_bytes=3000) >= 2
    left = {p.name for p in (prerender.base_dir() / "pr").iterdir()}
    assert "k4.json" in left and "k0.json" not in left


def test_race_max_age_is_long_only_for_finished_past_races():
    today = "20261027"
    past = prerender.PAST_RACE_MAX_AGE_SEC
    detail = lambda date, result: json.dumps({"race_id": RID, "date": date, "race_result": result}).encode()
    preds = lambda date: json.dumps({"race_id": RID, "entries_status": {"race_date": date}}).encode()
    assert prerender.race_max_age(detail("2026-10-25", {"entries": [1]}), today=today) == past
    assert prerender.race_max_age(detail("2026-10-25", None), today=today) is None      # 結果が未取得の過去日は短いまま
    assert prerender.race_max_age(detail("2026-10-27", {"entries": [1]}), today=today) is None  # 当日は短いまま
    assert prerender.race_max_age(preds("2026-10-25"), today=today) == past
    assert prerender.race_max_age(preds("2026-11-01"), today=today) is None
    assert prerender.race_max_age(json.dumps({"race_id": RID}).encode(), today=today) is None  # 日付が無ければ既定
    assert prerender.race_max_age(b"not json", today=today) is None


def test_cached_response_applies_max_age_function(store):
    body = {"race_id": RID, "entries_status": {"race_date": "2020-01-01"}}

    async def compute():
        return JSONResponse(body)

    asyncio.run(prerender.cached_response("predictions", RID, compute, max_age=prerender.race_max_age))
    meta = json.loads(prerender._path("predictions", RID).read_bytes().split(b"\n", 1)[0])
    assert meta["max_age"] == prerender.PAST_RACE_MAX_AGE_SEC
