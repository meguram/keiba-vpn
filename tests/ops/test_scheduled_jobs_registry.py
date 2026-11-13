"""定期実行ジョブの一覧（config/scheduled_jobs.json）と、実物の定義（VPS のタイマーのひな形・GCP のジョブ表）の整合。"""

import json
import re
from pathlib import Path

from src.ops import scheduled_jobs as SJ

REPO = Path(__file__).resolve().parents[2]
TPL = REPO / "scripts/ops/vps/templates"


def _reg():
    return SJ.load()


def test_registry_is_valid():
    assert SJ.validate(_reg()) == []


def test_every_vps_timer_template_is_listed_with_the_same_schedule():
    reg = {j["ref"].get("timer"): j for j in _reg()["jobs"] if j["where"] == "vps" and j["kind"] == "timer"}
    for f in sorted(TPL.glob("*.timer")):
        assert f.name in reg, f"{f.name} が config/scheduled_jobs.json に無い"
        txt = f.read_text(encoding="utf-8")
        sch = reg[f.name]["schedule"]
        cal = re.search(r"^OnCalendar=(.+)$", txt, re.M)
        act = re.search(r"^OnUnitActiveSec=(\d+)(min|s)$", txt, re.M)
        if act:
            n = int(act.group(1)) / (60 if act.group(2) == "s" else 1)
            assert sch == {"every_min": n}, f.name
        elif cal and re.fullmatch(r"\*-\*-\* (\d\d):(\d\d):00", cal.group(1).strip()):
            hh, mm = re.fullmatch(r"\*-\*-\* (\d\d):(\d\d):00", cal.group(1).strip()).groups()
            assert sch == {"cron": f"{int(mm)} {int(hh)} * * *"}, f.name
        elif cal and cal.group(1).strip() == "*-*-* *:50:00":
            assert sch == {"cron": "50 * * * *"}, f.name
        elif cal and re.fullmatch(r"\*-\*-\* \*:00/(\d+):\d\d", cal.group(1).strip()):
            n = int(re.fullmatch(r"\*-\*-\* \*:00/(\d+):\d\d", cal.group(1).strip()).group(1))
            assert sch == {"every_min": n}, f.name
        else:
            raise AssertionError(f"{f.name}: 予定の書き方を整合テストが知らない（テストを足す）")


def test_gcp_groups_match_the_job_map():
    m = json.loads((REPO / "config/gcp_job_map.json").read_text(encoding="utf-8"))
    reg = {j["ref"].get("group"): j for j in _reg()["jobs"] if j["where"] == "gcp" and j["kind"] == "scheduler"}
    assert set(reg) == set(m["scheduler_groups"])
    for gid, g in m["scheduler_groups"].items():
        assert reg[gid]["schedule"] == {"cron": g["cron_jst"]}, gid
        assert reg[gid]["ref"]["run_job"] == m["runtime"]["run_jobs"][g["size"]]["name"], gid
        assert reg[gid]["ref"]["scheduler"] == f"keiba-{gid}"


def test_active_and_paused_jobs_explain_what_they_do():
    for j in _reg()["jobs"]:
        if j["status"] in ("active", "paused"):
            assert j.get("what") and (j.get("outputs") or j["kind"] == "scheduler"), j["id"]
            assert j.get("basis"), j["id"]


def test_sample_urls_carry_no_secrets():
    for j in _reg()["jobs"]:
        u = j.get("sample_url") or ""
        assert not re.search(r"(key|token|secret|password)=", u, re.I), j["id"]
