from datetime import datetime

from src.scripts.scraping import fetch_status as fs


def _prog():
    days = [f"201901{d:02d}" for d in range(1, 11)]          # 10 開催日
    return {
        "calendar_2019": days,
        "done_fast": days[:6],
        "stats_fast": {d: {"races": 30} for d in days[:6]},
        "past_horse_ids": [str(i) for i in range(600)],
        "past_horse_dates": days[:6],                         # 100 頭/日
        "past_horses_done": [str(i) for i in range(200)],
        "done_cards": days[:1],
        "stats_cards": {days[0]: {"races": 30, "fetched": 60}},
        "paternal_anchor_pending": ["a"] * 20,
    }


def test_past_remaining_back_calculates_requests_per_stage():
    rows = {r["tier"]: r for r in fs.past_remaining(_prog(), horse_gets=600, years=[2019])}
    assert list(rows) == fs.PAST_TIERS                                 # 段階の順に並ぶ（見込みは累計で出す）
    assert rows["fast"]["left_req"] == 4 * (30 + 1)                   # 残り 4 日 × (30 レース + 一覧 1)
    horses_left = 400 + 4 * 100 * fs.NEW_HORSE_DECAY                   # 未処理 400 頭 + 未集計 4 日の新しい馬（概算）
    assert rows["horse"]["left_req"] == horses_left * 3.0              # 実績 600 件 / 200 頭
    assert rows["horse"]["approx"]
    assert rows["cards"]["left_req"] == 9 * 30 * 2.0                   # 実績 60 件 / 30 レース
    assert rows["lap"]["left_req"] == 10 * 30 * 1.0                    # 実績なし → 1 件/レース
    assert rows["anchors_f"]["left_req"] == 20 + horses_left * (20 / 200)


def test_scan_log_counts_windows_and_current_run(tmp_path):
    log = tmp_path / "x.log"
    log.write_text("\n".join([
        "2026-10-29 10:00:00 [INFO] scraper.client: GET https://db.netkeiba.com/horse/1/ (attempt 1/4)",
        "===== 2026-10-29T20:00:00+09:00 START =====",
        "2026-10-29 21:30:00 [INFO] scraper.client: GET https://db.netkeiba.com/race/2/ (attempt 1/4)",
        "2026-10-29 21:40:00 [INFO] scraper.backfill: 待機中",
    ]), encoding="utf-8")
    now = datetime(2026, 10, 29, 22, 0, 0)
    s = fs.scan_log(log, now, windows_h=(1, 24))
    assert s["windows"] == {1: 1, 24: 2}
    assert s["kinds"] == {"db/horse": 1, "db/race": 1}
    assert s["run_start"] == datetime(2026, 10, 29, 20, 0, 0) and s["run_gets"] == 1
    assert s["last_msg"].endswith("待機中")
    assert fs.rate_per_hour(s, 24, now) == 2 / 12                      # ログが窓より新しいときは、ログの最初からの時間で割る


def test_render_runs_without_data(tmp_path, monkeypatch):
    for name in ("PROGRESS", "TRAINING_PROGRESS", "PAST_LOG", "TRAINING_LOG"):
        monkeypatch.setattr(fs, name, tmp_path / name)
    monkeypatch.setattr(fs, "_procs", lambda: {"past": [], "training": [], "other": []})
    out = fs.render(datetime(2026, 10, 29, 22, 0, 0))
    assert "停止中" in out and "見込みは出せません" in out and "完了" not in out
