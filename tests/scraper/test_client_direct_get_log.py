"""session を直接使う API の取得（オッズなど）も「GET」の行を出し、取得の状況で数えられること（2026-11-05）。"""
import logging
from datetime import datetime

from src.scraper import client as C


def _fake_get(url, *a, **kw):
    return "resp"


def test_direct_session_get_to_netkeiba_logs_a_get_line_once(caplog):
    g = C._log_direct_get(_fake_get)
    with caplog.at_level(logging.INFO, logger="scraper.client"):
        assert g("https://race.netkeiba.com/api/api_get_jra_odds.html?race_id=201704020206") == "resp"
        g("https://example.com/x")                                   # netkeiba 以外は書かない
        C._IN_CLIENT_GET.on = True
        try:
            g("https://race.netkeiba.com/race/shutuba.html")         # client.get の中（client.get が書く）は書かない
        finally:
            C._IN_CLIENT_GET.on = False
    lines = [r.getMessage() for r in caplog.records]
    assert lines == ["GET https://race.netkeiba.com/api/api_get_jra_odds.html?race_id=201704020206 (api)"]


def test_fetch_status_counts_api_get_lines(tmp_path):
    from src.scripts.scraping import fetch_status as F

    log = tmp_path / "b.log"
    log.write_text("2026-11-05 08:56:54 [INFO] scraper.client: GET https://race.netkeiba.com/api/api_get_jra_odds.html?race_id=1 (api)\n"
                   "2026-11-05 08:56:54 [INFO] scraper.run: 保存: race_odds/201704020206 (15件)\n", encoding="utf-8")
    s = F.scan_log(log, datetime(2026, 11, 5, 9, 0, 0))
    assert s["windows"][1] == 1 and s["kinds"] == {"api/race_odds": 1}      # 保存の行で 1 件（API の GET は二重に数えない）


def test_fetch_status_counts_odds_saves_even_without_get_lines(tmp_path):
    from src.scripts.scraping import fetch_status as F

    log = tmp_path / "b.log"
    log.write_text("".join(f"2026-11-05 08:{m:02d}:00 [INFO] scraper.run: 保存: race_odds/2017040202{m:02d} (15件)\n" for m in range(10, 40))
                   + "2026-11-05 08:50:00 [INFO] scraper.run: 保存: race_pair_odds/201704020206 (3券種)\n"
                   + "2026-11-05 08:51:00 [INFO] scraper.run: スキップ (既存): race_odds/201704020206\n", encoding="utf-8")
    s = F.scan_log(log, datetime(2026, 11, 5, 9, 0, 0))
    assert s["windows"][1] == 31 and round(F.rate_per_hour(s, 1, datetime(2026, 11, 5, 9, 0, 0)), 1) == round(31 / (50 / 60), 1)   # ログが 50 分なので 50 分で割る
