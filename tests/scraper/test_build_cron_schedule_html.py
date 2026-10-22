from src.scripts.docs.build_cron_schedule_html import parse, to_jst

SAMPLE = """PATH=/opt/conda/bin:/usr/bin
*/3 * * * * /w/scripts/server/server_watchdog.sh # KEIBA-VPN-WATCHDOG
30 22 * * * cd /w && TZ=Asia/Tokyo bash /w/scripts/cron/run_auto_scrape_logged.sh /w raceday-runner logs/x.log # K
0 9 * * 5 cd /w && TZ=Asia/Tokyo bash /w/scripts/cron/yield_to_rescrape.sh weekly-update -- bash /w/scripts/cron/run_auto_scrape_logged.sh /w weekly-update logs/w.log # K
*/10 20-23 * * * cd /w && TZ=Asia/Tokyo bash /w/scripts/cron/run_auto_scrape_logged.sh /w jra-baba-morning logs/b.log # K
0 15 * * * cd /w && TZ=Asia/Tokyo bash /w/scripts/cron/yield_to_rescrape.sh backfill -- /opt/conda/bin/python3 -m src.scraper.backfill --year 2026 --phase fast --max-dates 7 >> /w/l 2>&1 # B
0 17 * * 1,4 cd /w && TZ=Asia/Tokyo bash /w/scripts/cron/yield_to_rescrape.sh backfill -- /opt/conda/bin/python3 -m src.scraper.backfill --year 2021 --phase fast >> /w/l 2>&1 # B
*/2 * * * * /home/x/start_remote_access.sh # KEIBA-REMOTE-ACCESS
*/5 * * * * /home/x/unknown_tool.sh # KEIBA-UNKNOWN
"""


def test_utc_to_jst_conversion_including_day_rollover():
    assert to_jst("30", "22", "*")[0] == "毎日 07:30"
    assert to_jst("0", "9", "5")[0] == "金 18:00"
    assert to_jst("0", "17", "1,4")[0] == "火・金 02:00"  # 17:00 UTC は翌日 02:00 JST
    assert to_jst("0", "15", "*")[0] == "毎日 00:00"
    assert to_jst("*/3", "*", "*")[0] == "毎日 終日 3分ごと"
    assert to_jst("*/10", "20-23", "*")[0].startswith("毎日 05:00〜08:50（10分ごと）")


def test_parse_groups_and_flags():
    jobs, others = parse(SAMPLE)
    by = {j.name: j for j in jobs}
    assert by["開催日ランナー（オッズ・出馬表）"].group == "live" and not by["開催日ランナー（オッズ・出馬表）"].yields
    assert by["週次更新"].yields and by["週次更新"].when == "金 18:00"
    assert by["backfill: 2026年 結果+出馬表"].group == "backfill" and by["backfill: 2026年 結果+出馬表"].yields
    assert by["backfill: 2021年 結果+出馬表"].when == "火・金 02:00"
    assert by["サーバ死活監視"].group == "ops"
    assert by["リモートアクセスの起動保証"].group == "ops"
    assert len(others) == 1 and "KEIBA-UNKNOWN" in others[0]


def test_every_job_documents_processing_and_destination():
    from src.scripts.docs.build_cron_schedule_html import JOBS

    for key, (_g, _n, _d, process, dest) in JOBS.items():
        assert process and dest, key
