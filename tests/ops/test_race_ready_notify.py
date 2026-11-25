"""レースごとの「予測・推奨が画面に出た」Slack 通知（src/scripts/ops/race_ready_notify.py）。送信は差し替えた関数で受ける。"""

import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.scripts.ops import race_ready_notify as R

DAY = "20261128"
JST = R.JST
STG_URL = "https://hooks.slack.com/services/T000/B000/stgSECRET"
PROD_URL = "https://hooks.slack.com/services/T000/B000/prodSECRET"
ENV = {"SLACK_WEBHOOK_URL_STG": STG_URL, "SLACK_WEBHOOK_URL_PROD": PROD_URL,
       "SLACK_WEBHOOK_URL": "https://hooks.slack.com/services/T000/B000/generic", "KEIBA_ENV": "stg",
       "KEIBA_NOTIFY_ENV_LABEL": "XX"}
CFG = json.loads((Path(R.ROOT) / "config" / "race_ready_notify.json").read_text(encoding="utf-8"))


def at(hm: str) -> datetime:
    h, m = map(int, hm.split(":"))
    return datetime(2026, 11, 28, h, m, tzinfo=JST)


def _w(root: Path, rel: str, obj) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return p


def make_day(root: Path, *, races=(("202605050101", "2歳未勝利", "10:00"), ("202605050111", "ジャパンC", "15:40"))):
    _w(root, f"data/page_reference/race_lists/{DAY}.json",
       {"date": DAY, "races": [{"race_id": rid, "round": int(rid[-2:]), "venue": "東京", "race_name": nm, "date": DAY} for rid, nm, _ in races]})
    for rid, _, st in races:
        _w(root, f"data/local/meta/entries_state/{rid}.json", {"race_id": rid, "date": "2026-11-28", "start_time": st})


def event(root: Path, rid: str, hm: str, *, sec77="ok", status="ok"):
    p = root / "data/local/inference/events" / f"{DAY}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"race_id": rid, "at": at(hm).isoformat(), "status": status, "sec77": sec77}) + "\n")


def pushed(root: Path, tree: str, rids, hm: str, index_ok=True):
    p = root / R.PUSH_LOG / f"{DAY}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": at(hm).isoformat(), "tree": tree, "race_ids": list(rids), "index_ok": index_ok}) + "\n")


class Post:
    def __init__(self, status=200):
        self.calls: list[tuple[str, dict]] = []
        self.status = status

    def __call__(self, url, json=None, timeout=None):
        self.calls.append((url, json))
        return type("Resp", (), {"status_code": self.status, "headers": {}})()


def run(root, now, *, state=None, send=True, post=None, environ=None, cfg=None):
    return R.run_once(DAY, now, root=root, cfg=cfg or CFG, send=send, state={} if state is None else state,
                      environ=ENV if environ is None else environ, post=post or Post(), sleep=lambda s: None)


@pytest.fixture(autouse=True)
def _no_storage(monkeypatch):
    """手元のファイルだけを読む: HybridStorage・GCS のクライアントを作ったら落とす。"""
    import src.scraper.storage as st

    def boom(*a, **k):
        raise AssertionError("GCS・storage を使ってはいけない")

    monkeypatch.setattr(st, "HybridStorage", boom)
    monkeypatch.setenv("KEIBA_STAGE2_START", "20261121")


def test_sends_once_per_env_after_lead_with_links_and_labels(tmp_path):
    make_day(tmp_path)
    rid = "202605050101"
    event(tmp_path, rid, "09:05")
    pushed(tmp_path, "predictions_v6", [rid], "09:06")
    post = Post()
    state: dict = {}
    assert run(tmp_path, at("09:09"), state=state, post=post)["sent"] == []          # 上げてから 4 分たっていない
    r = run(tmp_path, at("09:10"), state=state, post=post)
    assert sorted((x["env"], x["race_id"]) for x in r["sent"]) == [("prod", rid), ("stg", rid)]
    by = {url: body for url, body in post.calls}
    assert set(by) == {STG_URL, PROD_URL}                                            # 汎用の宛先には送らない
    stg, prod = json.dumps(by[STG_URL], ensure_ascii=False), json.dumps(by[PROD_URL], ensure_ascii=False)
    assert f"https://stg.megu-keiba.com/race/{rid}" in stg and "app.megu-keiba.com" not in stg
    assert f"https://app.megu-keiba.com/race/{rid}" in prod and "stg.megu-keiba.com" not in prod
    assert by[STG_URL]["text"].startswith("【STG】") and by[PROD_URL]["text"].startswith("【PROD】")   # KEIBA_NOTIFY_ENV_LABEL を引き継がない
    assert "東京 1R 2歳未勝利" in stg and "発走 10:00" in stg and "見込み 09:10" in stg
    assert run(tmp_path, at("09:11"), state=state, post=post)["sent"] == [] and len(post.calls) == 2   # 重ねて送らない


def test_needs_sec77_for_the_recommendation(tmp_path):
    make_day(tmp_path)
    rid = "202605050111"
    event(tmp_path, rid, "15:00", sec77="no_model")
    pushed(tmp_path, "predictions_v6", [rid], "15:01")
    r = run(tmp_path, at("15:20"))
    assert r["sent"] == []
    event(tmp_path, rid, "15:10", sec77="ok")                                         # やり直しで最後の ok の行が sec77=ok
    assert len(run(tmp_path, at("15:20"))["sent"]) == 2


def test_never_after_post_and_not_when_stale(tmp_path):
    make_day(tmp_path)
    rid = "202605050101"
    event(tmp_path, rid, "09:00")
    pushed(tmp_path, "predictions_v6", [rid], "09:00")
    assert run(tmp_path, at("10:00"))["sent"] == []                                   # 発走後
    state: dict = {}
    r = run(tmp_path, at("09:30"), state=state)                                       # そろって 26 分（15 分超）
    assert r["sent"] == [] and r["skipped_old"] == 2
    assert run(tmp_path, at("09:31"), state=state)["skipped_old"] == 0               # 記録したので数え直さない


def test_dry_run_records_and_never_sends_later(tmp_path):
    make_day(tmp_path)
    rid = "202605050101"
    event(tmp_path, rid, "09:05")
    pushed(tmp_path, "predictions_v6", [rid], "09:06")
    post = Post()
    state: dict = {}
    r = run(tmp_path, at("09:10"), state=state, send=False, post=post)
    assert len(r["dryrun"]) == 2 and post.calls == []
    assert all(STG_URL not in x["text"] and PROD_URL not in x["text"] for x in r["dryrun"])
    assert run(tmp_path, at("09:11"), state=state, send=True, post=post)["sent"] == [] and post.calls == []


def test_missing_prod_webhook_does_not_fall_back(tmp_path):
    make_day(tmp_path)
    rid = "202605050101"
    event(tmp_path, rid, "09:05")
    pushed(tmp_path, "predictions_v6", [rid], "09:06")
    post = Post()
    env = {k: v for k, v in ENV.items() if k != "SLACK_WEBHOOK_URL_PROD"}
    r = run(tmp_path, at("09:10"), post=post, environ=env)
    assert [x["env"] for x in r["sent"]] == ["stg"] and [u for u, _ in post.calls] == [STG_URL]


def test_failed_sends_retry_up_to_three_times_without_leaking_url(tmp_path):
    make_day(tmp_path)
    rid = "202605050101"
    event(tmp_path, rid, "09:05")
    pushed(tmp_path, "predictions_v6", [rid], "09:06")
    post = Post(status=500)
    state: dict = {}
    for m in range(10, 15):
        run(tmp_path, at(f"09:{m}"), state=state, post=post)
    assert len(post.calls) == 6                                                       # 2 環境 × 3 回で止める
    assert STG_URL not in json.dumps(state) and PROD_URL not in json.dumps(state)


def test_index_failure_delays_visibility(tmp_path):
    make_day(tmp_path)
    rid = "202605050101"
    event(tmp_path, rid, "09:00")
    pushed(tmp_path, "predictions_v6", [rid], "09:25", index_ok=False)               # Cloud Run に見えるのは 09:55 とみなす
    r = run(tmp_path, at("09:40"))
    assert r["sent"] == []
    r = run(tmp_path, at("09:51"))                                                    # 発走 10 分前を過ぎた → stg にだけ警告
    assert r["sent"] == [] and [(x["env"], x["race_id"]) for x in r["late"]] == [("stg", rid)]
    assert "目録の書き込みに失敗" in r["late"][0]["why"]


def test_late_warning_is_grouped_and_sent_once_to_stg_only(tmp_path):
    make_day(tmp_path, races=(("202605050101", "2歳未勝利", "10:00"), ("202605050102", "2歳新馬", "10:00")))
    post = Post()
    state: dict = {}
    r = run(tmp_path, at("09:51"), state=state, post=post)
    assert len(r["late"]) == 2 and len(post.calls) == 1 and post.calls[0][0] == STG_URL
    assert "2 レース" in post.calls[0][1]["text"] and "推論なし" in json.dumps(post.calls[0][1], ensure_ascii=False)
    run(tmp_path, at("09:52"), state=state, post=post)
    assert len(post.calls) == 1


def test_stage2_line_dev_only_and_waiting(tmp_path):
    make_day(tmp_path)
    rid = "202605050101"
    inf = tmp_path / R.INF
    _w(tmp_path, f"{R.INF}/bets_stage2/_dev_only.json", {})
    event(tmp_path, rid, "09:05")
    pushed(tmp_path, "predictions_v6", [rid], "09:06")
    post = Post()
    run(tmp_path, at("09:10"), post=post)
    body = json.dumps(post.calls[0][1], ensure_ascii=False)
    assert "第 2 段（試行中）: 判定待ち" in body and "推奨馬券: 出た（今の推奨）" in body
    # 判定が先に上がっていれば、開発者だけの 1 行を足す（会員の推奨は今の推奨のまま）
    _w(tmp_path, f"{R.INF}/bets_stage2/{DAY}/{rid}.json",
       {"strategy": "stage2_r2", "status": "buy", "honmei": {"horse_number": 9}, "bets": [{"combo": [9]}], "as_of": at("09:04").isoformat()})
    pushed(tmp_path, "bets_stage2", [rid], "09:05")
    post = Post()
    run(tmp_path, at("09:10"), post=post)
    body = json.dumps(post.calls[0][1], ensure_ascii=False)
    assert "第 2 段（試行中）: 開発者だけ・単勝 ◎9 1 点（判定 09:04）" in body and "推奨馬券: 出た（今の推奨）" in body
    assert inf.exists()


def test_stage2_for_members_when_not_dev_only_and_not_before_start(tmp_path, monkeypatch):
    make_day(tmp_path)
    rid = "202605050101"
    _w(tmp_path, f"{R.INF}/bets_stage2/_dev_only.json", {"disabled": False})         # 目印が無ければ開発者だけ（stage2_live.dev_only の既定）。会員に出す段
    event(tmp_path, rid, "09:05", sec77="no_model")                                   # 第 2 段が会員に出るなら §77 は要らない
    pushed(tmp_path, "predictions_v6", [rid], "09:06")
    _w(tmp_path, f"{R.INF}/bets_stage2/{DAY}/{rid}.json", {"strategy": "stage2_r2", "status": "skip", "skip_label": "候補なし"})
    pushed(tmp_path, "bets_stage2", [rid], "09:06")
    post = Post()
    run(tmp_path, at("09:10"), post=post)
    body = json.dumps(post.calls[0][1], ensure_ascii=False)
    assert "推奨馬券: 出た（第 2 段（試行中））" in body and "会員に出た・見送り（候補なし）" in body
    monkeypatch.setenv("KEIBA_STAGE2_START", "20261205")                              # 開始日より前の日は第 2 段を読まない
    assert run(tmp_path, at("09:10"))["sent"] == []


def test_non_race_day_returns_quickly(tmp_path):
    assert R.main(["--root", str(tmp_path), "--day", DAY]) == 0
    assert not (tmp_path / R.STATE_DIR).exists()


def test_env_is_read_from_root_without_overriding(tmp_path, monkeypatch):
    monkeypatch.delenv("KEIBA_SKIP_DOTENV", raising=False)
    monkeypatch.delenv("SLACK_WEBHOOK_URL_STG", raising=False)
    monkeypatch.setenv("KEIBA_SLACK_NOTIFY", "0")
    (tmp_path / ".env").write_text("SLACK_WEBHOOK_URL_STG=https://x/$notvar y\nKEIBA_SLACK_NOTIFY=1\n", encoding="utf-8")
    R._load_env(tmp_path)
    assert R.os.environ["SLACK_WEBHOOK_URL_STG"] == "https://x/$notvar y"             # shell のように $・空白を解釈しない
    assert R.os.environ["KEIBA_SLACK_NOTIFY"] == "0"                                  # 既にある値は上書きしない
    monkeypatch.delenv("SLACK_WEBHOOK_URL_STG")


def test_main_reads_env_only_on_race_days(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(R, "_load_env", lambda root: called.append(root))
    assert R.main(["--root", str(tmp_path), "--day", DAY]) == 0 and called == []     # 非開催日は .env も読まない


def test_main_without_mark_does_not_send(tmp_path, monkeypatch):
    make_day(tmp_path)
    monkeypatch.setenv("KEIBA_SLACK_NOTIFY", "1")
    assert R.sending_enabled(tmp_path, {"KEIBA_SLACK_NOTIFY": "1"}) is False
    _w(tmp_path, R.MARK, "x")
    assert R.sending_enabled(tmp_path, {"KEIBA_SLACK_NOTIFY": "1"}) is True
    assert R.sending_enabled(tmp_path, {"KEIBA_SLACK_NOTIFY": "0"}) is False


def test_cron_line_is_utc_for_jst_7_to_17():
    sh = (Path(R.ROOT) / "scripts/cron/install_race_ready_notify_cron.sh").read_text(encoding="utf-8")
    assert 'CRON_TIME="* 22-23,0-8 * * *"' in sh                                     # root の crontab は UTC（JST 07:00〜17:59）
    for f in ("install_race_ready_notify_cron.sh", "race_ready_notify.sh"):
        subprocess.run(["bash", "-n", str(Path(R.ROOT) / "scripts/cron" / f)], check=True)
    assert "flock -n" in (Path(R.ROOT) / "scripts/cron/race_ready_notify.sh").read_text(encoding="utf-8")


def test_config_has_no_secret_and_right_hosts():
    raw = (Path(R.ROOT) / "config" / "race_ready_notify.json").read_text(encoding="utf-8")
    assert "hooks.slack.com" not in raw
    assert CFG["targets"]["prod"]["base_url"] == "https://app.megu-keiba.com"
    assert CFG["targets"]["stg"]["base_url"] == "https://stg.megu-keiba.com"
    assert timedelta(seconds=CFG["lead_sec"]) == timedelta(minutes=4)


def test_v6c_and_stage2_count_only_for_maiden_races(tmp_path):
    """新しい候補の予測と第 2 段は未勝利・新馬だけ（画面の in_scope と同じ）。重賞に v6c があっても「出た」にしない。"""
    make_day(tmp_path)
    rid = "202605050111"                                                              # ジャパンC
    event(tmp_path, rid, "15:00", sec77="no_model")
    pushed(tmp_path, "predictions_v6c", [rid], "15:00")
    _w(tmp_path, f"{R.INF}/bets_stage2/{DAY}/{rid}.json", {"strategy": "stage2_r2", "status": "buy"})
    pushed(tmp_path, "bets_stage2", [rid], "15:00")
    assert run(tmp_path, at("15:10"))["sent"] == []


def test_index_failure_is_lifted_by_the_next_successful_index(tmp_path):
    make_day(tmp_path)
    rid = "202605050101"
    event(tmp_path, rid, "09:00")
    pushed(tmp_path, "predictions_v6", [rid], "09:00", index_ok=False)
    pushed(tmp_path, "_index", [], "09:05", index_ok=True)                            # 次の push で目録を書けた
    r = run(tmp_path, at("09:09"))
    assert sorted(x["env"] for x in r["sent"]) == ["prod", "stg"]


def test_index_recovery_written_on_the_previous_evening_is_seen(tmp_path):
    make_day(tmp_path)
    rid = "202605050101"
    event(tmp_path, rid, "09:00")
    eve = datetime(2026, 11, 27, 21, 0, tzinfo=JST)
    p = tmp_path / R.PUSH_LOG / f"{DAY}.jsonl"                                        # 前日の夜に上げた翌日分（目録は失敗）
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"at": eve.isoformat(), "tree": "predictions_v6", "race_ids": [rid], "index_ok": False}) + "\n", encoding="utf-8")
    (tmp_path / R.PUSH_LOG / "20261127.jsonl").write_text(                            # 回復の行は書いた日（前日）のファイルに入る
        json.dumps({"at": (eve + timedelta(minutes=5)).isoformat(), "tree": "_index", "race_ids": [], "index_ok": True}) + "\n", encoding="utf-8")
    races = {r.race_id: r for r in R.collect(DAY, tmp_path)}
    assert races[rid].pushed["predictions_v6"] == eve + timedelta(minutes=5)          # 30 分の推定ではなく回復の時刻


def test_late_warning_over_budget_is_not_counted_as_failure(tmp_path):
    make_day(tmp_path, races=(("202605050101", "2歳未勝利", "10:00"),))
    cfg = dict(CFG, max_per_run=0)
    state: dict = {}
    post = Post()
    for m in range(51, 56):
        run(tmp_path, at(f"09:{m}"), state=state, post=post, cfg=cfg)
    assert post.calls == [] and not any(k.endswith("|late") for k in state)
    run(tmp_path, at("09:56"), state=state, post=post)
    assert len(post.calls) == 1


def test_state_is_saved_right_after_each_send(tmp_path):
    make_day(tmp_path)
    rid = "202605050101"
    event(tmp_path, rid, "09:05")
    pushed(tmp_path, "predictions_v6", [rid], "09:06")
    snaps: list[dict] = []

    def save(st):
        snaps.append(json.loads(json.dumps(st)))
        raise_after = len(snaps) == 1
        if raise_after:
            raise KeyboardInterrupt                                                  # 1 通送って記録した直後に落ちる

    with pytest.raises(KeyboardInterrupt):
        R.run_once(DAY, at("09:10"), root=tmp_path, cfg=CFG, send=True, state={}, environ=ENV, post=Post(),
                   sleep=lambda s: None, save=save)
    assert snaps and any(v.get("st") == "sent" for v in snaps[0].values())


def test_main_sends_and_saves_state(tmp_path, monkeypatch):
    make_day(tmp_path)
    rid = "202605050101"
    event(tmp_path, rid, "09:05")
    pushed(tmp_path, "predictions_v6", [rid], "09:06")
    _w(tmp_path, R.MARK, "x")
    post = Post()
    import src.utils.slack_notify as SN

    real = SN.send_payload
    monkeypatch.setattr(SN, "send_payload", lambda payload, **kw: real(payload, **dict(kw, post=post, sleep=lambda s: None)))
    monkeypatch.setattr(R.time, "sleep", lambda s: None)
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("KEIBA_SLACK_NOTIFY", "1")

    class FakeNow(datetime):
        @classmethod
        def now(cls, tz=None):
            return at("09:10")

    monkeypatch.setattr(R, "datetime", FakeNow)
    assert R.main(["--root", str(tmp_path)]) == 0
    st = json.loads(R.state_path(tmp_path, DAY).read_text(encoding="utf-8"))
    assert st[f"stg|{rid}|ready"]["st"] == "sent" and st[f"prod|{rid}|ready"]["st"] == "sent" and len(post.calls) == 2
    assert STG_URL not in json.dumps(st)


def test_explicit_dry_run_does_not_touch_state(tmp_path, monkeypatch):
    make_day(tmp_path)
    rid = "202605050101"
    event(tmp_path, rid, "09:05")
    pushed(tmp_path, "predictions_v6", [rid], "09:06")

    class FakeNow(datetime):
        @classmethod
        def now(cls, tz=None):
            return at("09:10")

    monkeypatch.setattr(R, "datetime", FakeNow)
    assert R.main(["--root", str(tmp_path), "--dry-run"]) == 0
    assert not R.state_path(tmp_path, DAY).exists()
