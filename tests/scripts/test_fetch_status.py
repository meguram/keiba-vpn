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
    monkeypatch.setattr(fs, "_procs", lambda: {"past": [], "training": [], "other": []})
    out = fs.render(datetime(2026, 10, 29, 22, 0, 0))
    assert "停止中" in out and "見込みは出せません" in out and "完了" not in out


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
    monkeypatch.setattr(fs, "_procs", lambda: {"past": [], "training": [], "other": []})
    out = fs.render(datetime(2026, 10, 29, 22, 0, 0))
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
