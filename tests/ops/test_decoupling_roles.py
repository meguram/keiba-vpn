"""ホスト分類ジョブの見直し(config/decoupling_roles.json)が gcp_job_map と食い違っていないか。"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROLES = {"research_only", "ops_required", "retire"}
TARGETS = {"vps", "gcp"}


def _load():
    jobs = json.loads((ROOT / "config" / "gcp_job_map.json").read_text(encoding="utf-8"))["jobs"]
    roles = json.loads((ROOT / "config" / "decoupling_roles.json").read_text(encoding="utf-8"))["roles"]
    return jobs, roles


def test_every_host_or_retire_job_has_a_role():
    jobs, roles = _load()
    need = {j["id"] for j in jobs if j["destination"] in ("host", "retire")}
    assert need - set(roles) == set(), f"役割が未定義: {sorted(need - set(roles))}"
    assert set(roles) - {j["id"] for j in jobs} == set(), "job_map に無い id がある"


def test_roles_are_valid_and_ops_required_have_a_target():
    _, roles = _load()
    for jid, r in roles.items():
        assert r["role"] in ROLES, jid
        assert r.get("note"), jid
        if r["role"] == "ops_required":
            assert r.get("target") in TARGETS, f"{jid}: 移し先(vps/gcp)が必要"


def test_retire_destination_jobs_are_retire_role():
    jobs, roles = _load()
    for j in jobs:
        if j["destination"] == "retire":
            assert roles[j["id"]]["role"] == "retire", j["id"]
