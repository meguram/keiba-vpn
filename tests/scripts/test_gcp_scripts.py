"""scripts/gcp の読み取り・DRY 動作を、偽の gcloud で確認する（GCP には接続しない）。"""
from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GCP = ROOT / "scripts" / "gcp"

DESCRIBE = {"location": "ASIA-NORTHEAST1", "location_type": "region", "default_storage_class": "STANDARD",
            "versioning_enabled": True, "uniform_bucket_level_access": True, "public_access_prevention": "enforced",
            "lifecycle_config": {"rule": [{"action": {"type": "Delete"}, "condition": {"age": 365}}]}}


def _gcloud(tmp: Path) -> dict:
    stub = tmp / "gcloud"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'echo "$*" >> "$GCLOUD_LOG"\n'
        'case "$*" in\n'
        '  "config get-value project") echo test-proj ;;\n'
        '  *"buckets describe"*) cat "$DESCRIBE_JSON" ;;\n'
        '  *describe*) exit 1 ;;\n'
        '  *"get-iam-policy"*) echo "roles/storage.objectUser serviceAccount:x" ;;\n'
        "esac\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    (tmp / "d.json").write_text(json.dumps(DESCRIBE))
    return {**os.environ, "PATH": f"{tmp}:{os.environ['PATH']}", "GCLOUD_LOG": str(tmp / "log"), "DESCRIBE_JSON": str(tmp / "d.json")}


def test_audit_gcs_renders_table_and_only_reads(tmp_path):
    env = _gcloud(tmp_path)
    r = subprocess.run(["bash", str(GCP / "audit_gcs.sh"), "my-bucket"], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert "リージョン" in r.stdout and "OK（東京）" in r.stdout and "バージョニング" in r.stdout and "Delete" in r.stdout
    log = (tmp_path / "log").read_text()
    assert not any(w in log for w in (" create", " delete", " update", " add-iam", " rm ", " enable"))


def test_setup_project_dry_run_lists_expected_resources(tmp_path):
    env = {**_gcloud(tmp_path), "DRY": "1", "BUCKET": "b"}
    r = subprocess.run(["bash", str(GCP / "setup_project.sh")], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    out = r.stdout + r.stderr
    for api in ("run.googleapis.com", "cloudscheduler", "cloudtasks", "secretmanager", "artifactregistry", "cloudbuild", "billingbudgets", "bigquery"):
        assert api in out
    for sa in ("keiba-api", "keiba-jobs", "keiba-scheduler", "keiba-deployer"):
        assert f"service-accounts create {sa}" in out
    assert "roles/storage.objectUser" in out and "run.invoker" in out
    assert "roles/owner" not in out and "roles/editor" not in out


def test_put_secret_rejects_unknown_name_and_never_takes_value_as_argument(tmp_path):
    env = _gcloud(tmp_path)
    r = subprocess.run(["bash", str(GCP / "put_secret.sh"), "NOT_REGISTERED"], env=env, capture_output=True, text=True, input="v", timeout=30)
    assert r.returncode == 2 and "未登録" in r.stderr
    r = subprocess.run(["bash", str(GCP / "put_secret.sh"), "SLACK_WEBHOOK_URL"], env=env, capture_output=True, text=True, input="s3cr3t-value", timeout=30)
    assert r.returncode == 0, r.stderr
    assert "s3cr3t-value" not in r.stdout + r.stderr
    assert "s3cr3t-value" not in (tmp_path / "log").read_text()  # gcloud の引数にも出ない（標準入力で渡す）
