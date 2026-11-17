"""メモリ・GCS の長い障害で自分で終わる仕組み（src/api/memory_guard.py）。本当には終わらない（終わり方は偽物に差し替える）。"""

import pytest

from src.api import memory_guard as MG


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def _guard(ratios, *, stuck=False, clock=None):
    seq = list(ratios)
    calls = {"clear": 0, "term": []}

    def read():
        return {"ratio": seq.pop(0) if len(seq) > 1 else seq[0]}

    def clear():
        calls["clear"] += 1

    clk = clock or Clock()
    g = MG.MemoryGuard(read=read, clear=clear, terminate=lambda s: calls["term"].append(s), stuck=lambda d: stuck, clock=clk)
    return g, calls, clk


def test_low_memory_does_nothing_and_checks_only_every_30s():
    g, calls, clk = _guard([0.4])
    clk.t += 400
    assert g.maybe_check() is None
    assert not g.claim()
    clk.t += 10
    assert g.maybe_check() is None and calls == {"clear": 0, "term": []}


def test_soft_threshold_clears_caches():
    g, calls, clk = _guard([0.88, 0.6])
    clk.t += 400
    assert g.maybe_check() == "clear" and calls["clear"] == 1 and calls["term"] == []


def test_still_high_after_clear_exits_once():
    g, calls, clk = _guard([0.95, 0.94])
    clk.t += 400
    assert g.maybe_check() == "exit_memory"
    assert calls["term"] == [g.grace_sec]
    clk.t += 60
    assert g.maybe_check() is None and len(calls["term"]) == 1          # 2 回は終わらない
    assert not g.claim()


def test_never_exits_right_after_start():
    g, calls, clk = _guard([0.99])
    clk.t += 60                                                          # 起動から 5 分未満
    assert g.maybe_check() == "clear" and calls["term"] == []


def test_gcs_stuck_exits_only_with_high_memory_and_after_min_uptime():
    g, calls, clk = _guard([0.75], stuck=True)
    clk.t += 60
    assert g.maybe_check() is None
    clk.t += 400
    assert g.maybe_check() == "exit_gcs" and len(calls["term"]) == 1


def test_gcs_outage_alone_never_exits():
    g, calls, clk = _guard([0.4], stuck=True)                            # 地域の障害など: メモリは低い
    clk.t += 1000
    assert g.maybe_check() is None and calls["term"] == []


def test_clear_at_most_every_5_minutes_and_exit_if_still_high():
    g, calls, clk = _guard([0.88, 0.80, 0.89, 0.93])
    clk.t += 400
    assert g.maybe_check() == "clear" and calls["clear"] == 1
    clk.t += 31
    assert g.maybe_check() is None and calls["clear"] == 1                # 間隔の内は捨て直さない
    clk.t += 31
    assert g.maybe_check() == "exit_memory" and calls["clear"] == 1        # 間隔の内で hard を越えたら終わる


def test_claim_lets_only_one_concurrent_check():
    g, _, clk = _guard([0.1])
    clk.t += 400
    assert g.claim() is True and g.claim() is False


def test_read_errors_never_raise():
    def boom():
        raise OSError("no cgroup")

    g = MG.MemoryGuard(read=boom, clear=lambda: None, terminate=lambda s: None, stuck=lambda d: False, clock=Clock())
    g.started -= 1000
    assert g.maybe_check() is None


def test_unreadable_ratio_is_ignored():
    g, calls, clk = _guard([None])
    clk.t += 400
    assert g.maybe_check() is None and calls["clear"] == 0


def test_gcs_stuck_reads_shared_state(monkeypatch):
    import time

    from src.scraper import storage as S

    S._SHARED_GCS.clear()
    st = S._GcsState()
    S._SHARED_GCS["b"] = st
    now = time.time()
    st.last_success, st.last_failure, st.rebuild_at = now - 700, now - 5, now - 650
    assert MG.gcs_stuck(600)
    st.rebuild_at = 0.0                                                  # 作り直す前は終わらない
    assert not MG.gcs_stuck(600)
    st.rebuild_at, st.last_success = now - 650, now - 100                # 作り直しの後に成功している
    assert not MG.gcs_stuck(600)
    st.last_success, st.rebuild_at = now - 2000, now - 100               # 作り直してから間もない
    assert not MG.gcs_stuck(600)
    st.rebuild_at, st.last_failure = now - 650, now - 120                # いまは不調でない
    assert not MG.gcs_stuck(600)
    S._SHARED_GCS.clear()


def test_default_clear_targets_caches_only(monkeypatch):
    from src.utils import cache_epoch

    seen = []
    monkeypatch.setattr(cache_epoch, "clear_some", lambda names, reason="": seen.append(tuple(names)) or list(names))
    monkeypatch.setattr(cache_epoch, "clear_all", lambda *a, **k: (_ for _ in ()).throw(AssertionError("clear_all は呼ばない")))
    MG.default_clear()
    assert seen and "storage.load_cache" in seen[0] and "public_policy" not in seen[0]


def test_gateway_runs_guard_in_thread_only_when_due(monkeypatch):
    import asyncio

    monkeypatch.setenv("KEIBA_CLOUDRUN_LAZY", "1")
    from src.api import cloudrun_entry as E

    hits = []

    class G:
        def __init__(self):
            self.d = True

        def claim(self):
            d, self.d = self.d, False
            return d

        def check(self):
            hits.append(1)

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    gw = E.Gateway(app, app, app)
    gw.guard = G()

    async def run():
        async def recv():
            return {"type": "http.request"}

        async def send(m):
            pass

        for _ in range(3):
            await gw({"type": "http", "path": "/api/race/1", "method": "GET"}, recv, send)

    asyncio.run(run())
    assert hits == [1]


def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv("KEIBA_MEMORY_GUARD", "0")
    assert not MG.enabled()
    monkeypatch.setenv("KEIBA_MEMORY_GUARD", "1")
    assert MG.enabled()


if __name__ == "__main__":
    pytest.main([__file__])
