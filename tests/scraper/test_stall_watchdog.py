import subprocess
import sys
import textwrap
import time

from src.scraper import stall_watchdog as sw


def test_stalled_detects_long_http_and_idle(monkeypatch):
    now = time.monotonic()
    monkeypatch.setattr(sw, "_last_log", now)
    assert sw._stalled(240, 2400, now) is None
    with sw.track_http("http://x/a"):
        assert sw._stalled(240, 2400, now + 100) is None
        assert "http://x/a" in sw._stalled(240, 2400, now + 300)
    assert sw._inflight == {}
    assert "ログ" in sw._stalled(240, 2400, now + 2500)


def test_hung_http_exits_75_with_stack():
    code = textwrap.dedent("""
        import time
        from src.scraper import stall_watchdog as sw
        sw.start(http_sec=1, idle_sec=100, poll_sec=0.2)
        with sw.track_http("http://hang/"):
            time.sleep(30)
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert r.returncode == sw.EXIT_STALLED
    assert "http://hang/" in r.stderr and "Thread" in r.stderr
