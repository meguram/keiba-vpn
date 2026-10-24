"""config/gcp_job_map.json の整合性と、cron 式の評価器。外部接続なし。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.ops import fetch_owner, fetch_task as FT, job_map as jm

JST = ZoneInfo("Asia/Tokyo")
DOC = jm.load()


def test_job_map_is_valid():
    errs = jm.validate(DOC, known_fetch_tasks=set(FT.ALLOWED_TASKS), agent_tasks=set(FT.DEFAULT_AGENT_TASKS))
    assert errs == []


def test_runtime_env_marks_prod_gcp():
    # 通知の先頭【PROD】・末尾『実行場所: GCP』でホスト(stg)と見分けるため必須
    assert DOC["runtime"]["env"]["KEIBA_ENV"] == "prod"
    assert DOC["runtime"]["env"]["KEIBA_RUN_LOCATION"] == "gcp"


def test_scheduler_group_count_within_budget():
    n = len(DOC["scheduler_groups"])
    assert 6 <= n <= 14


def test_every_gcp_module_exists():
    for j in jm.by_destination("gcp", DOC):
        for st in j["steps"]:
            assert st["argv"][0] == "-m"
            mod = st["argv"][1]
            assert Path(mod.replace(".", "/") + ".py").exists() or Path(mod.replace(".", "/") + "/__init__.py").exists(), f"{j['id']}: {mod}"


def test_host_jobs_are_never_cut_over_targets():
    from src.scripts.ops import cutover_plan as cp

    for cid, u in cp.units(DOC).items():
        if "host" in u["destinations"]:
            with pytest.raises(ValueError):
                cp.render(cid, DOC)


def test_default_config_owner_file_is_empty_and_valid():
    owners = json.loads(Path("config/job_owner.json").read_text(encoding="utf-8"))["owners"]
    ok_ids = {(j.get("origin") or {}).get("cron_job_id") for j in DOC["jobs"] if j["destination"] in ("gcp", "vps", "retire")}
    assert set(owners) <= ok_ids


def test_vps_jobs_cover_all_fetch_tasks_they_declare():
    for j in jm.by_destination("vps", DOC):
        assert j["fetch_task"]["task"] in FT.ALLOWED_TASKS
        for t in j.get("extra_tasks") or []:
            assert t in FT.ALLOWED_TASKS


def test_vps_schedule_matches_host_cron_when_convertible():
    # 取得は時刻が命: VPS の schedule_jst はホストの実 crontab(UTC)と同じ時刻にそろっていること
    checked = 0
    for j in jm.by_destination("vps", DOC):
        org = (j.get("origin") or {}).get("cron_utc")
        sch = j.get("schedule_jst")
        if not org or not sch:
            continue
        try:
            conv = jm.jst_to_utc_cron(sch)
        except ValueError:
            continue
        # 同じ時刻に発火する分(今週のどこか)が一致すること
        start = datetime(2026, 10, 19, 0, 0, tzinfo=ZoneInfo("UTC"))
        a = [t for t in jm.occurrences(conv, start, start + timedelta(days=7))]
        b = [t for t in jm.occurrences(org, start, start + timedelta(days=7))]
        assert a == b, f"{j['id']}: {conv} != {org}"
        checked += 1
    assert checked >= 5


def test_vps_job_ready_only_for_agent_tasks():
    owned = set(FT.DEFAULT_AGENT_TASKS)
    for j in jm.by_destination("vps", DOC):
        if j["ready"]:
            assert j["fetch_task"]["task"] in owned


# ── cron 式 ──

def _t(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=JST)


def test_cron_matches_basic():
    assert jm.cron_matches("30 5 * * *", _t(2026, 10, 21, 5, 30))
    assert not jm.cron_matches("30 5 * * *", _t(2026, 10, 21, 5, 31))
    assert jm.cron_matches("*/15 * * * *", _t(2026, 10, 21, 7, 45))
    assert not jm.cron_matches("*/15 * * * *", _t(2026, 10, 21, 7, 40))
    assert jm.cron_matches("10,40 19 * * *", _t(2026, 10, 21, 19, 40))
    assert jm.cron_matches("* 8-16 * * 0,1,6", _t(2026, 10, 24, 12, 3))     # 土曜
    assert jm.cron_matches("* 8-16 * * 0,1,6", _t(2026, 10, 25, 12, 3))     # 日曜
    assert not jm.cron_matches("* 8-16 * * 0,1,6", _t(2026, 10, 21, 12, 3))  # 水曜
    assert jm.cron_matches("0 10 * * 1", _t(2026, 10, 19, 10, 0))            # 月曜
    assert jm.cron_matches("0 10 * * 7", _t(2026, 10, 25, 10, 0))            # 7 = 日曜


def test_cron_step_with_start_and_ranges():
    assert jm.cron_matches("5/10 * * * *", _t(2026, 10, 21, 1, 25))
    assert not jm.cron_matches("5/10 * * * *", _t(2026, 10, 21, 1, 20))
    assert jm.cron_matches("*/10 8-22 * * *", _t(2026, 10, 21, 22, 50))
    assert not jm.cron_matches("*/10 8-22 * * *", _t(2026, 10, 21, 23, 0))


def test_prev_occurrence_and_occurrences():
    now = _t(2026, 10, 21, 19, 47)
    assert jm.prev_occurrence("10,40 19 * * *", now) == _t(2026, 10, 21, 19, 40)
    occ = jm.occurrences("*/10 * * * *", _t(2026, 10, 21, 0, 0), _t(2026, 10, 21, 1, 0))
    assert [t.minute for t in occ] == [0, 10, 20, 30, 40, 50]


def test_invalid_cron_rejected():
    for bad in ("* * * *", "61 * * * *", "* 25 * * *", "*/0 * * * *"):
        with pytest.raises(ValueError):
            jm.parse_cron(bad)


def test_jst_to_utc_cron():
    assert jm.jst_to_utc_cron("30 5 * * *") == "30 20 * * *"
    assert jm.jst_to_utc_cron("0 10 * * 1") == "0 1 * * 1"
    assert jm.jst_to_utc_cron("30 8 * * 1") == "30 23 * * 0"          # 日をまたぐと曜日も 1 つ戻る
    assert jm.jst_to_utc_cron("10,40 19 * * *") == "10,40 10 * * *"
    with pytest.raises(ValueError):
        jm.jst_to_utc_cron("0 8-9 * * *")


def test_when_jst_filters_are_reachable():
    # *_weekly の when_jst が g-0800 の tick(毎日)のどこかで成立する（月曜）
    j = jm.job_by_id("tracking-v2-weekly", DOC)
    g = jm.group_of(j["group"], DOC)
    assert jm._can_match(g["cron_jst"], j["when_jst"])
    assert not jm._can_match("0 8 * * 2", j["when_jst"])


def test_validate_detects_problems():
    bad = json.loads(json.dumps(DOC))
    bad["jobs"][0]["ready"] = True
    bad["jobs"][0]["blockers"] = ["x"]
    bad["runtime"]["env"]["KEIBA_ENV"] = "stg"
    errs = jm.validate(bad)
    assert any("blockers があるのに ready=true" in e for e in errs)
    assert any("KEIBA_ENV=prod" in e for e in errs)


def test_fetch_owner_default_is_compatible():
    assert fetch_owner.agent_tasks() == set()      # 既定は空（誰も奪わない）
