"""scripts/ci/deploy_gcp.sh（CI から呼ぶ反映の手順）の --dry-run の検査。GCP・docker には接続しない。"""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SH = ROOT / "scripts/ci/deploy_gcp.sh"


def _run(step: str, target: str, **env):
    e = {k: v for k, v in os.environ.items() if k not in ("PROJECT_ID", "GCS_BUCKET", "ENABLE_JOBS_DEPLOY", "JOBS_PARALLEL")}
    e.update({"GITHUB_SHA": "0123456789abcdef" * 2 + "01234567", "PROJECT_ID": "p1", "GCS_BUCKET": "b1", **env})
    return subprocess.run(["bash", str(SH), step, "--target", target, "--dry-run"], cwd=ROOT, env=e, capture_output=True, text=True)


def test_build_images_pushes_api_and_jobs_and_checks_job_modules():
    r = _run("build-images", "stg")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "deploy/docker/Dockerfile.cloudrun" in out and "deploy/docker/Dockerfile.jobs" in out
    assert "asia-northeast1-docker.pkg.dev/p1/keiba-vpn/keiba-api:01234567" in out and "keiba-vpn/jobs:01234567" in out
    assert "importlib.import_module" in out                    # イメージの中で全ジョブのモジュールを読み込む
    assert out.index("importlib") < out.rindex("docker push")   # 確かめてから Jobs のイメージを push


def test_stg_uses_fixed_tag_and_never_switches_traffic_or_jobs():
    r = _run("api-revision", "stg")
    assert r.returncode == 0 and "TAG=stg" in r.stdout and "deploy_cloud_run_api.sh deploy --apply" in r.stdout
    for step in ("promote", "jobs"):
        assert _run(step, "stg").returncode != 0              # stg では通信の切り替えも Jobs の更新もできない


def test_prod_revision_is_tagged_with_the_commit():
    r = _run("api-revision", "prod")
    assert r.returncode == 0 and "TAG=c01234567" in r.stdout


def test_smoke_needs_race_id_and_hides_the_key():
    r = _run("smoke", "prod")
    assert r.returncode == 0                                   # dry-run は仮の値で通る
    assert "KEIBA_PROXY_KEY=$(gcloud secrets versions access" in r.stdout and "cloudrun_smoke" in r.stdout
    e = {k: v for k, v in os.environ.items() if k != "SMOKE_RACE_ID"}
    e.update({"PROJECT_ID": "p1", "GCS_BUCKET": "b1"})
    real = subprocess.run(["bash", str(SH), "smoke", "--target", "prod"], cwd=ROOT, env=e, capture_output=True, text=True)
    assert real.returncode != 0                                # 実行時にレース ID が無ければ、通信は切り替えない


def test_promote_is_prod_only_and_jobs_are_opt_in_and_parallel_by_default():
    assert "deploy_cloud_run_api.sh promote --apply" in _run("promote", "prod").stdout
    off = _run("jobs", "prod")
    assert off.returncode == 0 and "更新は行いません" in off.stdout and "apply" not in off.stdout.replace("更新は行いません", "")
    on = _run("jobs", "prod", ENABLE_JOBS_DEPLOY="true")
    assert on.stdout.count("--parallel") == 2 and "apply queue" in on.stdout and "apply scheduler" in on.stdout   # 既定は並走(影モード)
    full = _run("jobs", "prod", ENABLE_JOBS_DEPLOY="true", JOBS_PARALLEL="0")
    assert "--parallel" not in full.stdout
