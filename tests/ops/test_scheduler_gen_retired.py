"""消した Cloud Scheduler を、master への昇格（promote.yml → deploy_gcp.sh jobs → deploy_cloud_run_jobs.sh apply scheduler）で作り直さないこと。
2026-11-13 に費用のため keiba-g-* 13 件を消した。scheduler_gen は「無ければ create」なので、job_map で ready のままだと昇格のたびに復活していた。"""
import json
import re
from pathlib import Path

from src.ops import job_map, scheduler_gen

ROOT = Path(__file__).resolve().parents[2]


def _scheduler_names(text: str) -> set[str]:
    return set(re.findall(r"gcloud scheduler jobs (?:create|update) http (\S+)", text))


def test_only_non_retired_groups_are_created_even_with_include_unready():
    doc = job_map.load()
    for inc in (False, True):
        names = _scheduler_names("\n".join(scheduler_gen.scheduler_commands(doc, scheduler_gen.settings({}), include_unready=inc)))
        want = {f"keiba-{g}" for g, v in doc["scheduler_groups"].items() if not v.get("retired")}
        assert names <= want, f"retired のグループのコマンドが出ている: {sorted(names - want)}"
    assert "keiba-g-0730-shadow" in names


def test_retired_group_is_never_ready():
    doc = {"scheduler_groups": {"g-x": {"jobs": ["a"], "retired": "消した"}}, "jobs": [{"id": "a", "ready": True, "enabled": True}]}
    assert scheduler_gen.group_is_retired(doc, "g-x") and not scheduler_gen.group_is_ready(doc, "g-x")


def test_job_map_agrees_with_scheduled_jobs_registry():
    """config/scheduled_jobs.json（定期ジョブの一覧の正。持ち主は監視のセッション）で retired・paused の Scheduler は、job_map でも retired。
    active のものは retired にしない（一覧と生成のずれで、消したものが復活したり、動かしたいものが作られなかったりしないように）。"""
    reg = json.loads((ROOT / "config" / "scheduled_jobs.json").read_text(encoding="utf-8"))
    groups = job_map.load()["scheduler_groups"]
    bad = []
    for j in reg.get("jobs") or []:
        if j.get("kind") != "scheduler" or not str(j.get("id", "")).startswith("gcp-g-"):
            continue
        gid = j["id"][len("gcp-"):]
        assert gid in groups, f"一覧にある Scheduler {gid} が job_map の scheduler_groups に無い"
        retired = bool(groups[gid].get("retired"))
        if j.get("status") in ("retired", "paused") and not retired:
            bad.append(f"{gid}: 一覧は {j['status']} だが job_map は retired でない（昇格で作り直される）")
        if j.get("status") == "active" and retired:
            bad.append(f"{gid}: 一覧は active だが job_map は retired（作られない）")
    listed = {str(j["id"])[len("gcp-"):] for j in reg.get("jobs") or [] if j.get("kind") == "scheduler" and str(j.get("id", "")).startswith("gcp-g-")}
    bad += [f"{g}: job_map にあるが一覧（scheduled_jobs.json）に Scheduler の行が無い" for g in groups if g not in listed]
    assert not bad, "\n".join(bad)


def test_retired_reason_must_be_one_line_string():
    from copy import deepcopy

    doc = deepcopy(job_map.load())
    gid = next(iter(doc["scheduler_groups"]))
    for bad in (True, "", "a\nb"):
        doc["scheduler_groups"][gid]["retired"] = bad
        assert any("retired は理由の 1 行" in e for e in job_map.validate(doc))
