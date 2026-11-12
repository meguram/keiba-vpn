import json
from datetime import datetime

from src.scripts.scraping import fetch_status as fs

PAST_ARGS = "/opt/conda/bin/python3 -m src.scraper.backfill --phase past --cap 0"


def _row(pid, args, ppid=1, etime="01:00"):
    return {"pid": pid, "ppid": ppid, "user": "root", "etime": etime, "args": args}


def _tasks(*rows, entries=None, now_ts=None):
    return fs.collect_tasks(rows=list(rows), entries=entries or {}, now_ts=now_ts or 1_000_000.0)


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


def test_horse_cost_uses_touched_horses_so_batch_saves_do_not_move_it():
    a = {r["tier"]: r for r in fs.past_remaining(_prog(), horse_gets=600, years=[2019], horse_touched=200)}["horse"]["left_req"]
    b = {r["tier"]: r for r in fs.past_remaining(_prog(), horse_gets=630, years=[2019], horse_touched=210)}["horse"]["left_req"]
    assert a == b                                                      # 保存前の 10 頭分を足しても 1 頭あたりは変わらない
    small = {r["tier"]: r for r in fs.past_remaining(_prog(), horse_gets=600, years=[2019], horse_touched=5)}["horse"]["left_req"]
    assert small == a                                                  # 触れた馬が少ない実行は全体の実績（600/200）で求める


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
    assert s["run_horse_gets"] == 0 and s["run_horse_ids"] == set()    # 最後の実行（START の後）には馬の GET が無い
    assert s["last_msg"].endswith("待機中")
    assert fs.rate_per_hour(s, 24, now) == 2 / 12                      # ログが窓より新しいときは、ログの最初からの時間で割る


def test_render_runs_without_data(tmp_path, monkeypatch):
    for name in ("PROGRESS", "TRAINING_PROGRESS", "PAST_LOG", "TRAINING_LOG"):
        monkeypatch.setattr(fs, name, tmp_path / name)
    out = fs.render(datetime(2026, 10, 29, 22, 0, 0), tasks=[])
    assert "実行中・待機中の取得はありません" in out and "停止中" not in out and "補填" not in out   # 完了・停止中のものは出さない
    out = fs.render(datetime(2026, 10, 29, 22, 0, 0), tasks=_tasks(_row(10, PAST_ARGS)))
    assert "[実行中] 2016〜2019 年の補填" in out and "見込みは出せません" in out and "完了" not in out


def test_scan_log_collects_horses_touched_in_last_run(tmp_path):
    log = tmp_path / "y.log"
    g = "2026-10-29 21:00:00 [INFO] scraper.client: GET https://db.netkeiba.com/horse/%s(attempt 1/4)"
    log.write_text("\n".join([
        g % "2012100001/ ", "===== 2026-10-29T20:00:00+09:00 START =====",
        g % "2012100002/ ", g % "result/2012100002/ ", g % "ped/2012100002/ ", g % "ped/1999100003/ ",
    ]), encoding="utf-8")
    s = fs.scan_log(log, datetime(2026, 10, 29, 22, 0, 0))
    assert s["run_horse_gets"] == 4 and s["run_horse_ids"] == {"2012100002", "1999100003"}
    assert s["kinds"]["db/horse"] == 5


def test_scan_log_counts_hex_id_anchors_and_splits_by_kind(tmp_path):
    log = tmp_path / "z.log"
    g = "2026-10-29 21:00:00 [INFO] scraper.client: GET https://db.netkeiba.com/horse/%s(attempt 1/4)"
    log.write_text("\n".join([
        "===== 2026-10-29T20:00:00+09:00 START =====",
        g % "2012100002/ ", g % "result/2012100002/ ", g % "ped/2012100002/ ",
        g % "ped/000a000e7d/ ", g % "ped/1999100003/ ",
    ]), encoding="utf-8")
    s = fs.scan_log(log, datetime(2026, 10, 29, 22, 0, 0))
    assert s["run_horse_gets"] == 5 and "000a000e7d" in s["run_horse_ids"]
    bd = fs.horse_breakdown(s["run_horse_by_id"], {"2012100002"})
    assert bd == {"touched": 0, "profile": 0, "result": 0, "ped": 0, "anchor": 2, "in_flight": 1}   # 最後に触れた馬は途中として除く
    bd = fs.horse_breakdown({**s["run_horse_by_id"], "2012100009": {"profile": 1}}, {"2012100002", "2012100009"})
    assert bd["touched"] == 1 and bd["profile"] == 1 and bd["result"] == 1 and bd["ped"] == 1 and bd["in_flight"] == 1


def test_scan_log_keeps_previous_run_horses_after_restart(tmp_path):
    g = "2026-10-29 21:00:00 [INFO] scraper.client: GET https://db.netkeiba.com/horse/%s(attempt 1/4)"
    lines = ["===== 2026-10-29T20:00:00+09:00 START ====="]
    for i in range(fs.MIN_TOUCHED_HORSES + 1):
        hid = f"20121{i:05d}"
        lines += [g % f"{hid}/ ", g % f"result/{hid}/ ", g % f"ped/{hid}/ "]
    lines += ["===== 2026-10-29T21:30:00+09:00 START =====", g % "2012199999/ "]          # 再起動直後（1 頭だけ）
    log = tmp_path / "r.log"
    log.write_text("\n".join(lines), encoding="utf-8")
    s = fs.scan_log(log, datetime(2026, 10, 29, 22, 0, 0))
    assert list(s["run_horse_by_id"]) == ["2012199999"]
    assert len(s["prev_horse_by_id"]) == fs.MIN_TOUCHED_HORSES + 1


def test_render_keeps_horse_parts_right_after_restart(tmp_path, monkeypatch):
    import json

    prog = _prog()
    prog["past_horse_ids"] = [f"20121{i:05d}" for i in range(600)]
    prog["past_horses_done"] = prog["past_horse_ids"][:200]
    (tmp_path / "prog.json").write_text(json.dumps(prog), encoding="utf-8")
    g = "2026-10-29 21:00:00 [INFO] scraper.client: GET https://db.netkeiba.com/horse/%s(attempt 1/4)"
    lines = ["===== 2026-10-29T20:00:00+09:00 START ====="]
    for hid in prog["past_horse_ids"][:60]:
        lines += [g % f"{hid}/ ", g % f"result/{hid}/ ", g % f"ped/{hid}/ "]
    lines += ["===== 2026-10-29T21:30:00+09:00 START =====", g % f"{prog['past_horse_ids'][200]}/ "]
    (tmp_path / "past.log").write_text("\n".join(lines), encoding="utf-8")
    monkeypatch.setattr(fs, "PROGRESS", tmp_path / "prog.json")
    monkeypatch.setattr(fs, "PAST_LOG", tmp_path / "past.log")
    monkeypatch.setattr(fs, "TRAINING_PROGRESS", tmp_path / "t.json")
    monkeypatch.setattr(fs, "TRAINING_LOG", tmp_path / "t.log")
    out = fs.render(datetime(2026, 10, 29, 22, 0, 0), tasks=_tasks(_row(10, PAST_ARGS)))
    assert "馬のプロフィール" in out and "母方の祖先の血統" in out


def test_horse_parts_sum_to_horse_left_req():
    parts = {"profile": 200, "result": 200, "ped": 200, "anchor": 300}      # 200 頭に触れて 900 件
    rows = {r["tier"]: r for r in fs.past_remaining(_prog(), horse_gets=900, years=[2019], horse_touched=200, horse_parts=parts)}
    h = rows["horse"]
    assert abs(sum(h["parts"].values()) - h["left_req"]) < 1e-6
    assert abs(h["parts"]["anchor"] - (h["total"] - h["done"]) * 1.5) < 1e-6
    assert "parts" not in {r["tier"]: r for r in fs.past_remaining(_prog(), horse_gets=900, years=[2019])}["horse"]


def test_horse_left_does_not_step_with_saves_and_ignores_in_flight():
    base = {r["tier"]: r for r in fs.past_remaining(_prog(), horse_gets=600, years=[2019], horse_touched=200, horse_unsaved=10)}["horse"]
    nxt = {r["tier"]: r for r in fs.past_remaining(_prog(), horse_gets=600, years=[2019], horse_touched=200, horse_unsaved=0)}["horse"]
    assert base["done"] == 210 and nxt["done"] == 200
    assert base["left_req"] == nxt["left_req"] - 10 * 3.0              # 保存前の 10 頭分だけ残りが減る（保存で段にならない）


def test_render_shows_training_past_section(tmp_path, monkeypatch):
    for name in ("PROGRESS", "TRAINING_PROGRESS", "PAST_LOG", "TRAINING_LOG", "TRAINING_PAST_PROGRESS", "TRAINING_PAST_LOG"):
        monkeypatch.setattr(fs, name, tmp_path / name)
    (tmp_path / "TRAINING_PAST_PROGRESS").write_text(json.dumps({
        "done": {f"h{i}": "saved" for i in range(100)}, "counts": {"saved": 100, "empty": 0, "error": 0}, "requests": 300, "total": 1000}), encoding="utf-8")
    g = "2026-10-29 21:%02d:00 [INFO] scraper.client: GET https://db.netkeiba.com/horse/training.html?id=2005105204 (attempt 1/4)"
    (tmp_path / "TRAINING_PAST_LOG").write_text("===== 2026-10-29T20:50:00+09:00 START =====\n" + "\n".join(g % m for m in range(10, 50)), encoding="utf-8")
    args = "python3 -m src.scripts.scraping.backfill_horse_training_past --horses-file x.parquet --limit 0"
    out = fs.render(datetime(2026, 10, 29, 22, 0, 0), tasks=_tasks(_row(10, args)))
    assert "[実行中] 調教の過去分の取り直し" in out and "100/1,000 頭" in out and "[███░░░░░░░░░░░░░░░░░░░░░░░░░░░]  10.0%" in out and "1 頭あたり 3.0 件" in out


def test_render_shows_redo_truncated_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "TRAINING_PAST_PROGRESS", tmp_path / "p.json")
    monkeypatch.setattr(fs, "TRAINING_PAST_LOG", tmp_path / "p.log")
    (tmp_path / "p.json").write_text(json.dumps({"done": {}, "redo": {f"h{i}": "saved" for i in range(130)}}), encoding="utf-8")
    g = "2026-10-29 21:%02d:00 [INFO] scraper.client: GET https://db.netkeiba.com/horse/training.html?id=2005105204 (attempt 1/4)"
    (tmp_path / "p.log").write_text("===== 2026-10-29T20:50:00+09:00 START =====\n"
                                    "打ち切った馬の取り直し: 対象 19119 頭 / 取り直し済み 100 / 残り 2000 頭 / 今回 2000 頭（見積もり 1.0 時間）\n"
                                    + "\n".join(g % m for m in range(10, 50)), encoding="utf-8")
    args = "python3 -m src.scripts.scraping.backfill_horse_training_past --horses-file x.parquet --redo-truncated"
    out = fs.render(datetime(2026, 10, 29, 22, 0, 0), tasks=_tasks(_row(10, args)))
    assert "--redo-truncated" in out and "130/2,100 頭" in out and "残り 1,970 頭" in out
    assert "1 頭あたり 1.3 件" in out                            # この実行で取り直した 30 頭に 40 件


def test_launcher_and_child_are_one_task_and_launcher_alone_is_waiting(tmp_path):
    sh = tmp_path / "run_fetch.sh"
    sh.write_text("#!/bin/bash\n# 進捗: bash scripts/dev/fetch_status.sh\nsleep 300\npython -m src.scripts.scraping.backfill_horse_training_past --limit 0\n")
    launcher = _row(20, f"bash {sh}")
    child = _row(21, "/opt/conda/bin/python3 -m src.scripts.scraping.backfill_horse_training_past --limit 0", ppid=20)
    ts = _tasks(launcher, child)
    assert len(ts) == 1 and ts[0]["state"] == "running" and [m["pid"] for m in ts[0]["members"]] == [20, 21]
    ts = _tasks(launcher)                                         # 子が無い（再開待ちの sleep・前の取得の終了待ち）
    assert len(ts) == 1 and ts[0]["state"] == "waiting"
    ts = _tasks(_row(30, f"bash {tmp_path / 'other.sh'}"), _row(31, "watch -n 1 bash scripts/dev/fetch_status.sh"),
                _row(32, "python -m src.scripts.scraping.fetch_status"), _row(33, "python -m src.scraper.backfill --report"))
    assert ts == []                                               # 取得でない sh・表示そのもの・状況の表示だけの起動は出さない


def test_registry_waiting_and_progress_are_shown():
    now_ts = 1_000_000.0
    e = {"state": "waiting", "waiting_for": "取得スロット待ち", "gets": 0}
    ts = _tasks(_row(40, "python3 -m src.scraper.course_position --from 2016"), entries={40: e}, now_ts=now_ts)
    assert ts[0]["state"] == "waiting" and ts[0]["waiting_for"] == "取得スロット待ち"
    bucket = str(int(now_ts // 60))
    e = {"state": "running", "gets": 200, "last_get_at": now_ts - 5, "buckets": {bucket: 100}, "done": 50, "total": 150, "unit": "頭"}
    ts = _tasks(_row(41, "python3 -c 'research'"), _row(42, "python3 main.py --prod"), entries={41: e, 42: {"state": "running", "gets": 3, "last_get_at": now_ts - 3600}},
                now_ts=now_ts)
    assert [t["top"] for t in ts] == [41]                         # 入口のパターンに無くても直近に GET があれば出す。常駐のサーバーは出さない
    lines = fs._generic_lines(ts[0], now_ts, datetime(2026, 10, 29, 22, 0, 0))
    assert any("直近 1 時間 100 件" in x for x in lines) and any("50/150 頭" in x and " 33.3%" in x and "あと 4.0 時間" in x for x in lines)   # 残り 100 頭 × 4 件 ÷ 100 件/時
    marked = {"state": "waiting", "marked": True, "label": "打ち切った馬の取り直し", "waiting_for": "PID 8219 の終了"}
    ts = _tasks(_row(43, "bash /tmp/x/redo_after.sh"), entries={43: marked}, now_ts=now_ts)
    assert ts[0]["state"] == "waiting" and ts[0]["label"] == "打ち切った馬の取り直し" and ts[0]["waiting_for"] == "PID 8219 の終了"


def test_bar_follows_existing_monitor_style():
    assert fs._bar(1, 4, 8) == "[██░░░░░░]  25.0%"
    assert fs._bar(5, 4, 4) == "[████] 100.0%"
    assert fs._bar(0, None, 4) == "[????]"
    assert fs._bar(1, 2, 4, pct=False) == "[██░░]"


def test_course_position_shows_bar_and_rate_from_samples(tmp_path, monkeypatch):
    from src.scraper import course_position as cp

    monkeypatch.setattr(fs, "CACHE_PATH", tmp_path / "c" / "cache.json")
    (tmp_path / "race_lists").mkdir()
    monkeypatch.setattr(cp, "RACE_LISTS", tmp_path / "race_lists")
    calls = []
    monkeypatch.setattr(cp, "race_dates", lambda y0, y1, before=None: calls.append(1) or ["20160105", "20160106", "20170105", "20170106"])
    store = {2016: {"2016-01-05": []}, 2017: {}}
    monkeypatch.setattr(cp, "load_store", lambda y: {"dates": store.get(y, {})})
    t = {"main": {"pid": 9, "args": "python -m src.scraper.course_position --from 2016 --to 2017"}}
    clock = [1_000_000.0]
    monkeypatch.setattr(fs, "_now_ts", lambda: clock[0])
    lines = fs._course_position_detail(t, datetime(2026, 11, 12, 18, 0))
    assert "1/4 開催日" in lines[0] and " 25.0%" in lines[0] and "測り始め" in lines[1]
    store[2016]["2016-01-06"] = []
    clock[0] += 1800                                                        # 30 分で 1 日 → 2 日/時、残り 2 日で 1 時間
    lines = fs._course_position_detail(t, datetime(2026, 11, 12, 18, 0))
    assert "2/4 開催日" in lines[0] and "2 日/時" in lines[1] and "あと 1.0 時間" in lines[1]
    assert len(calls) == 1                                                  # 総数（race_lists の全読み）は控えを使う


def test_running_task_without_progress_says_why():
    out = fs.render(datetime(2026, 10, 29, 22, 0, 0), tasks=_tasks(_row(10, "python -m src.scraper.jma_weather")))
    assert "[実行中] 気象庁の天気" in out and "進捗 —（総数の分からない取得" in out


def test_course_position_args_forms_and_stalled_and_broken_cache(tmp_path, monkeypatch):
    from src.scraper import course_position as cp

    monkeypatch.setattr(fs, "CACHE_PATH", tmp_path / "cache.json")
    (tmp_path / "cache.json").write_text(json.dumps({"course_position": [], "samples": {"x": [[1]]}}))   # 形の崩れた控え
    seen = []
    monkeypatch.setattr(cp, "race_dates", lambda y0, y1, before=None: seen.append((y0, y1)) or ["20180105", "20180106"])
    monkeypatch.setattr(cp, "result_dates", lambda y0, y1, before=None: ["20180107"], raising=False)
    monkeypatch.setattr(cp, "load_store", lambda y: {"dates": {"2018-01-05": []}})
    t = {"main": {"pid": 9, "args": "python -m src.scraper.course_position --from=2018 --to=2018 --include-results"}}
    clock = [1_000_000.0]
    monkeypatch.setattr(fs, "_now_ts", lambda: clock[0])
    lines = fs._course_position_detail(t, datetime(2026, 11, 12, 18, 0))
    assert seen == [(2018, 2018)] and "1/3 開催日" in lines[0]               # --from= の書き方・結果の表の日付も総数に入れる
    clock[0] += 600
    lines = fs._course_position_detail(t, datetime(2026, 11, 12, 18, 0))
    assert "直近 10 分は増えていない" in lines[1]                            # 止まっているのに「測り始め」と出さない
    clock[0] += 4 * 3600
    fs._course_position_detail({"main": {"pid": 10, "args": t["main"]["args"]}}, datetime(2026, 11, 12, 18, 0))
    assert list(json.loads((tmp_path / "cache.json").read_text())["samples"]) == ["cp:10"]   # 終わった取得の標本は捨てる
