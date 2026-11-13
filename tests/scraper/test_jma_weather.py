from datetime import date
from pathlib import Path

import pytest

from src.scraper import jma_weather as jw

FX = Path(__file__).parent / "fixtures"


def test_parse_hourly_s1_and_a1():
    s1 = jw.parse_hourly((FX / "jma_hourly_s1_sapporo_20261018.html").read_text(encoding="utf-8"))
    assert len(s1) == 24 and s1[0]["hour"] == 1 and s1[-1]["hour"] == 24
    r = s1[12]
    assert r["hour"] == 13 and r["wind_dir"] == "東南東" and r["wind_dir_deg"] == 112.5 and r["wind_speed_ms"] == 4.2 and r["temp_c"] == 15.4
    a1 = jw.parse_hourly((FX / "jma_hourly_a1_toyonaka_20261018.html").read_text(encoding="utf-8"))
    assert len(a1) == 24 and a1[12]["wind_speed_ms"] == 2.8 and a1[12]["wind_dir"] == "東南東" and a1[0]["humidity_pct"] is None      # アメダスの『平均風速』も読める・未観測は None


def test_parse_hourly_errors_on_structure_change():
    with pytest.raises(jw.JmaError):
        jw.parse_hourly("<html><body>メンテナンス中</body></html>")


def test_num_flags_and_missing():
    assert jw._num("5.4 )") == (5.4, ")") and jw._num("12.0 ]") == (12.0, "]")
    assert jw._num("×") == (None, "") and jw._num("///") == (None, "") and jw._num("--") == (0.0, "")


def test_parse_stations_and_nearest():
    st = jw.parse_stations((FX / "jma_prefecture_14.html").read_text(encoding="utf-8", errors="ignore"), 14)
    assert any(s["name"] == "札幌" and s["type"] == "s" and s["block_no"] == "47412" for s in st)
    assert all(s["ended"] for s in st if s["name"] == "阿曽岩")                                  # 終了年が入っている観測所は現行ではない
    near = jw.nearest_station(jw.COURSES[0], st)
    assert near["name"] == "札幌" and near["distance_km"] < 3


def test_wind_directions_cover_16_points():
    assert len(jw.WIND_DEG) == 16 and jw.WIND_DEG["北"] == 0.0 and jw.WIND_DEG["西"] == 270.0


class _Store:
    def __init__(self):
        self.d = {}

    def load(self, c, k, bypass_cache=False):
        return self.d.get((c, k))

    def save(self, c, k, data):
        self.d[(c, k)] = data
        return True

    def list_keys(self, c):
        return [k for (cc, k) in self.d if cc == c]


def test_fetch_range_skips_stored_days_and_saves(monkeypatch):
    store = _Store()
    html = (FX / "jma_hourly_s1_sapporo_20261018.html").read_text(encoding="utf-8")
    calls = []
    monkeypatch.setattr(jw, "_get", lambda url, session: calls.append(url) or html)
    monkeypatch.setattr(jw, "load_stations", lambda: {"01": {"station": {"type": "s", "prec_no": 14, "block_no": "47412", "name": "札幌", "lat": 43.06, "lon": 141.33, "distance_km": 1.5}}})
    r = jw.fetch_range(date(2020, 1, 1), date(2020, 1, 2), store=store)
    assert r == {"fetched": 2, "skipped": 0} and len(calls) == 2
    r2 = jw.fetch_range(date(2020, 1, 1), date(2020, 1, 2), store=store)
    assert r2 == {"fetched": 0, "skipped": 2} and len(calls) == 2                      # 取得済みの日は取り直さない
    assert jw.report(store, years=[2020]) == {"01": {2020: 2}}


def test_save_failure_is_an_error():
    class _Bad(_Store):
        def save(self, c, k, data):
            return False

    with pytest.raises(jw.JmaError):
        jw.save_month({"venue_code": "01", "month": "2020-01", "days": {}}, _Bad())


def test_month_object_layout_and_observed_at():
    html = (FX / "jma_hourly_s1_sapporo_20261018.html").read_text(encoding="utf-8")
    recs = jw.with_observed_at(jw.parse_hourly(html), date(2026, 10, 18))
    assert recs[0]["observed_at"] == "2026-10-18T01:00:00+09:00" and recs[-1]["observed_at"] == "2026-10-19T00:00:00+09:00"      # 24 時は翌日 0 時
    store = _Store()
    import types
    jw_get = lambda url, session: html      # noqa: E731
    mp = pytest.MonkeyPatch()
    mp.setattr(jw, "_get", jw_get)
    mp.setattr(jw, "load_stations", lambda: {"01": {"station": {"type": "s", "prec_no": 14, "block_no": "47412", "name": "札幌", "lat": 43.06, "lon": 141.33, "distance_km": 1.5}}})
    try:
        jw.fetch_range(date(2020, 1, 31), date(2020, 2, 1), store=store)
    finally:
        mp.undo()
    assert sorted(k for _, k in store.d) == ["01_202001", "01_202002"]                 # 月ごとに別オブジェクト
    day = store.d[("jma_weather_hourly", "01_202001")]["days"]["2020-01-31"]
    assert day["station"]["name"] == "札幌" and day["fetched_at"] and len(day["hours"]) == 24


def test_halt_file_blocks_start_and_is_written_on_abort(tmp_path, monkeypatch):
    from src.scraper import jma_weather as J

    halt = tmp_path / "meta" / "jma_weather_halt.json"

    def boom(*a, **k):
        raise J.JmaError("HTTP 503")

    monkeypatch.setattr(J, "fetch_range", boom)
    assert J.main(["--halt-file", str(halt), "daily"]) == 3 and "HTTP 503" in halt.read_text(encoding="utf-8")
    monkeypatch.setattr(J, "fetch_range", lambda *a, **k: (_ for _ in ()).throw(AssertionError("中断印があるのに取得した")))
    assert J.main(["--halt-file", str(halt), "daily"]) == 5


_ST = {"01": {"station": {"type": "s", "prec_no": 14, "block_no": "47412", "name": "札幌", "lat": 43.06, "lon": 141.33, "distance_km": 1.5}}}


def test_missing_table_for_yesterday_is_skipped_not_halted(monkeypatch):
    """日付が変わった直後は昨日のページがまだ出ていないことがある。止めずに飛ばす（2026-10-26 に 18 日間止まった原因）。"""
    from datetime import timedelta
    monkeypatch.setattr(jw, "_get", lambda url, session: "<html>no table</html>")
    monkeypatch.setattr(jw, "load_stations", lambda: _ST)
    y = date.today() - timedelta(days=1)
    assert jw.fetch_range(y, y, store=_Store()) == {"fetched": 0, "skipped": 0}


def test_missing_table_for_old_day_still_halts_with_context(monkeypatch):
    monkeypatch.setattr(jw, "_get", lambda url, session: "<html>no table</html>")
    monkeypatch.setattr(jw, "load_stations", lambda: _ST)
    with pytest.raises(jw.JmaError) as e:
        jw.fetch_range(date(2020, 1, 1), date(2020, 1, 1), store=_Store())
    assert "01 2020-01-01" in str(e.value) and "tablefix1" in str(e.value)


def test_run_script_exits_nonzero_while_halted():
    sh = (Path(__file__).resolve().parents[2] / "scripts" / "cron" / "jma_weather_run.sh").read_text(encoding="utf-8")
    line = next(x for x in sh.splitlines() if 'if [ -e "$HALT" ]' in x)
    assert "exit 5" in line


def _run_sh(tmp_path, hour: int, done: str | None = None, halt: bool = False, fail: str | None = None, rc: int = 0):
    """jma_weather_run.sh を、python の代わりに呼び出しを記録するだけの偽物で動かす。"""
    import os
    import subprocess

    proj = tmp_path / "proj"
    (proj / "data/local/meta").mkdir(parents=True, exist_ok=True)
    calls = tmp_path / "calls.txt"
    fake = tmp_path / "fakepy.sh"
    fake.write_text(f'#!/bin/sh\necho "$@" >> {calls}\nexit {rc}\n')
    fake.chmod(0o755)
    meta = proj / "data/local/meta"
    if done:
        (meta / "jma_weather_daily_done").write_text(done + "\n")
    if fail:
        (meta / "jma_weather_daily_fail").write_text(fail + "\n")
    if halt:
        (meta / "jma_weather_halt.json").write_text("{}")
    sh = Path(__file__).resolve().parents[2] / "scripts" / "cron" / "jma_weather_run.sh"
    env = {**os.environ, "PYTHON": str(fake), "JMA_LOCK": str(tmp_path / "lock"), "JMA_DAILY_FIRST_HOUR": str(hour)}
    r = subprocess.run(["bash", str(sh), str(proj)], env=env, timeout=30).returncode
    return r, (calls.read_text().splitlines() if calls.exists() else []), proj


def _jst(days: int = 0, fmt: str = "%F") -> str:
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone(timedelta(hours=9))) + timedelta(days=days)).strftime(fmt)


def test_run_script_runs_once_a_day_and_never_the_full_fetch(tmp_path):
    """毎時 `fetch --from 20160101` で月ファイル 1,310 件（約 345MiB）を読み直していた（2026-11-13）。1 日 1 回・前回からの範囲だけにする。"""
    rc, calls, proj = _run_sh(tmp_path, hour=0)
    assert rc == 0 and calls == [f"-m src.scraper.jma_weather fetch --from {_jst(-3, '%Y%m%d')}"]   # 記録が無ければ直近 3 日
    assert (proj / "data/local/meta/jma_weather_daily_done").read_text().strip() == _jst()
    rc2, calls2, _ = _run_sh(tmp_path, hour=0, done=_jst())          # 同じ日の 2 回目は何もしない
    assert rc2 == 0 and calls2 == calls


def test_run_script_resumes_from_last_done_day_capped(tmp_path):
    """PC や印で止まっていた日も、次に済んだ回で埋める（前回済んだ日の 2 日前から。31 日より前には戻らない）。"""
    _, calls, _ = _run_sh(tmp_path / "a", hour=0, done=_jst(-10))
    assert calls == [f"-m src.scraper.jma_weather fetch --from {_jst(-12, '%Y%m%d')}"]
    _, calls, _ = _run_sh(tmp_path / "b", hour=0, done=_jst(-90))
    assert calls == [f"-m src.scraper.jma_weather fetch --from {_jst(-31, '%Y%m%d')}"]


def test_run_script_caps_retries_per_day(tmp_path):
    rc, calls, proj = _run_sh(tmp_path / "a", hour=0, rc=1)
    assert rc == 1 and (proj / "data/local/meta/jma_weather_daily_fail").read_text().split() == [_jst(), "1"]
    rc, calls, _ = _run_sh(tmp_path / "b", hour=0, fail=f"{_jst()} 3")
    assert rc == 6 and calls == []                                    # 上限に達した日は読み直さない
    rc, calls, _ = _run_sh(tmp_path / "c", hour=0, fail=f"{_jst(-1)} 3")
    assert rc == 0 and len(calls) == 1                                # 前の日の失敗は数えない


def test_run_script_waits_for_morning_and_halt_is_visible(tmp_path):
    rc, calls, _ = _run_sh(tmp_path, hour=24)                        # まだその時刻でない
    assert rc == 0 and calls == []
    rc, calls, _ = _run_sh(tmp_path / "h", hour=0, halt=True)
    assert rc == 5 and calls == []


def test_load_failure_does_not_become_an_empty_month():
    """GCS の一時的な失敗を空の月とみなすと、数日分だけで月ファイルを上書きしてしまう。"""
    class S:
        _gcs_healthy = False

        def load(self, *a, **k):
            return None

    with pytest.raises(RuntimeError):
        jw.load_month("01", "202611", store=S())

    class Ok(S):
        _gcs_healthy = True
    assert jw.load_month("01", "202611", store=Ok())["days"] == {}     # 本当に無い月は空で始める
