"""promote.yml（昇格のワークフロー）と WIF 準備・SHA 固定スクリプトの検査。ネットワーク・GCP には接続しない。"""
import os
import re
import stat
import subprocess
import textwrap
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WFDIR = ROOT / ".github/workflows"
WF = WFDIR / "promote.yml"                      # master への push（本番）
WF_STG = WFDIR / "deploy-stg.yml"               # stg への push（stg）
ALL_WF = [p for p in sorted(WFDIR.glob("*.yml")) if p.name != "ci.yml"]     # 反映に関わるもの（ci.yml はテストだけ）


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _wf() -> dict:
    return _load(WF)


def _on(d: dict) -> dict:
    return d.get("on") or d.get(True)             # PyYAML は on を True に読む


def _steps(job: dict) -> list[dict]:
    return job.get("steps", [])


def _all_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in ALL_WF)


def test_branch_flow_stg_then_master():
    # develop → stg（stg 環境）→ master（本番）。それぞれの push で動く
    assert _on(_wf())["push"]["branches"] == ["master"]
    assert _on(_load(WF_STG))["push"]["branches"] == ["stg"]
    on = _on(_wf())
    assert set(on["workflow_dispatch"]["inputs"]) == {"skip_tests", "skip_stg_check"}
    for d in (_wf(), _load(WF_STG)):
        assert "pull_request_target" not in _on(d) and "pull_request" not in _on(d)
        assert d["permissions"] == {"contents": "read"}              # 既定は読み取りのみ
        assert d["concurrency"]["cancel-in-progress"] is False
    assert "stg" in _on(_load(WFDIR / "ci.yml"))["push"]["branches"]


def test_prod_requires_stg_membership_approval_and_single_deploy_job():
    jobs = _wf()["jobs"]
    assert set(jobs) == {"verify-stg", "test", "build-ui", "deploy"}
    dep = jobs["deploy"]
    assert dep["environment"] == "production" and dep["permissions"]["id-token"] == "write"   # 承認は 1 回、この job だけが GCP に入る
    assert set(dep["needs"]) == {"verify-stg", "build-ui"}
    for name in ("verify-stg", "test", "build-ui"):
        assert "id-token" not in (jobs[name].get("permissions") or {})   # ビルド・検証は GCP に入れない
        assert "environment" not in jobs[name]
    run = "\n".join(s.get("run", "") for s in jobs["verify-stg"]["steps"])
    assert "origin/stg" in run and "merge-base --is-ancestor" in run and "exit 1" in run     # stg に無いコミットは止める


def test_prod_deploy_order_smoke_before_promote_and_publish_last():
    steps = _steps(_wf()["jobs"]["deploy"])
    runs = [s.get("run", "") for s in steps]
    order = {k: next(i for i, r in enumerate(runs) if k in r) for k in
             ("sha256sum -c", "build-images", "api-revision", " smoke ", "promote --target prod", "jobs --target prod", "release_publish publish-ui")}
    auth = next(i for i, s in enumerate(steps) if "google-github-actions/auth" in s.get("uses", ""))
    assert order["sha256sum -c"] < auth < order["build-images"] < order["api-revision"] < order[" smoke "]
    assert order[" smoke "] < order["promote --target prod"] < order["jobs --target prod"] < order["release_publish publish-ui"]


def test_stg_deploy_never_switches_traffic_or_touches_prod_assets():
    jobs = _load(WF_STG)["jobs"]
    dep = jobs["deploy"]
    assert dep["environment"] == "stg" and dep["permissions"]["id-token"] == "write"
    text = WF_STG.read_text(encoding="utf-8")
    assert "promote --target" not in text and "jobs --target" not in text and "release_publish" not in text
    assert "api-revision --target stg" in text and "smoke --target stg" in text
    assert "environment" not in jobs["test"] and "environment" not in jobs["build-ui"]


def test_auth_uses_wif_without_key_files_or_secrets():
    text = _all_text()
    assert "credentials_json" not in text and "GCP_SA_KEY" not in text and "secrets." not in text.replace("secrets: inherit", "")
    for path in (WF, WF_STG):
        auths = [s for j in _load(path)["jobs"].values() for s in _steps(j) if "google-github-actions/auth" in s.get("uses", "")]
        assert len(auths) == 1 and "vars.WIF_PROVIDER" in auths[0]["with"]["workload_identity_provider"]


def test_ci_never_scrapes():
    assert not re.search(r"auto_scrape|src\.scraper|netkeiba\.com|scrape_", _all_text())


def test_actions_are_pinned_or_marked_for_pinning():
    refs = re.findall(r"uses:\s*([^\s#]+)(?:\s*#\s*(\S+))?", _all_text())
    for ref, note in refs:
        if ref.startswith("./"):
            continue
        sha = re.search(r"@([0-9a-f]{40})$", ref)
        assert sha or note == "PIN", f"{ref}: SHA 固定か # PIN の印が必要（scripts/ci/pin_actions.sh で固定）"


def _stub_bin(tmp_path: Path, name: str, body: str) -> Path:
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    p = d / name
    p.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return d


def test_setup_wif_dry_run_has_conditions_and_creates_no_key(tmp_path):
    stub = _stub_bin(tmp_path, "gcloud", 'echo "gcloud $*" >> "$STUB_LOG"; exit 0\n')
    env = {**os.environ, "PATH": f"{stub}:{os.environ['PATH']}", "STUB_LOG": str(tmp_path / "log"), "GCS_BUCKET": "b1", "PROJECT_ID": "p1"}
    r = subprocess.run(["bash", str(ROOT / "scripts/gcp/setup_github_wif.sh")], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = r.stdout.replace("\\", "")                           # printf %q の引用符のエスケープを外して読む
    assert "[dry]" in out and not (tmp_path / "log").exists()    # 既定は表示のみ（gcloud を呼ばない）
    assert "assertion.repository=='meguram/keiba-vpn'" in out
    assert "assertion.ref=='refs/heads/master'" in out and "assertion.environment=='production'" in out
    assert "assertion.ref=='refs/heads/stg'" in out and "assertion.environment=='stg'" in out
    assert "refs/heads/main" not in out
    assert "keys create" not in out and "--key" not in out        # 鍵は作らない
    assert "ops/releases/" in out and "roles/storage.objectUser" in out
    assert "roles/iam.workloadIdentityUser" in out


def test_setup_wif_requires_bucket(tmp_path):
    stub = _stub_bin(tmp_path, "gcloud", "exit 0\n")
    env = {k: v for k, v in os.environ.items() if k != "GCS_BUCKET"}
    env.update({"PATH": f"{stub}:{os.environ['PATH']}", "PROJECT_ID": "p1"})
    r = subprocess.run(["bash", str(ROOT / "scripts/gcp/setup_github_wif.sh")], env=env, capture_output=True, text=True)
    assert r.returncode != 0


def test_pin_actions_rewrites_marked_lines_only(tmp_path):
    repo = tmp_path / "r"
    (repo / ".github/workflows").mkdir(parents=True)
    (repo / "scripts/ci").mkdir(parents=True)
    (repo / "scripts/ci/pin_actions.sh").write_text((ROOT / "scripts/ci/pin_actions.sh").read_text(encoding="utf-8"), encoding="utf-8")
    (repo / ".github/workflows/promote.yml").write_text(textwrap.dedent("""\
        jobs:
          a:
            steps:
              - uses: actions/checkout@v4   # PIN
              - uses: actions/setup-node@v4
        """), encoding="utf-8")
    sha = "a" * 40
    stub = _stub_bin(tmp_path, "git", f'echo "{sha}\trefs/tags/v4^{{}}"\n')
    env = {**os.environ, "PATH": f"{stub}:{os.environ['PATH']}"}
    r = subprocess.run(["bash", "scripts/ci/pin_actions.sh", "--write"], cwd=repo, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = (repo / ".github/workflows/promote.yml").read_text(encoding="utf-8")
    assert f"actions/checkout@{sha}   # v4" in out
    assert "actions/setup-node@v4\n" in out                         # 印の無い行は触らない


def test_referenced_scripts_and_modules_exist():
    text = _all_text()
    for p in set(re.findall(r"bash (scripts/[\w./-]+\.sh)", text)):
        assert (ROOT / p).exists(), p
    for m in set(re.findall(r"python -m ([\w.]+)", text)) - {"pytest", "pip"}:
        assert (ROOT / (m.replace(".", "/") + ".py")).exists(), m
    # build.env を書く先と、publish の GCS 接頭辞が既存の部品と同じ
    assert "scripts/ops/vps/build.env" in text
    assert 'PREFIX = "ops/releases"' in (ROOT / "src/ops/release_publish.py").read_text(encoding="utf-8")


def test_publish_dry_run_with_fake_tarball(tmp_path):
    import hashlib
    import json

    from src.ops import release_publish as RP

    t = tmp_path / "keiba-ui-abcdef12-202610221200.tar.gz"
    t.write_bytes(b"fake")
    Path(str(t) + ".sha256").write_text(f"{hashlib.sha256(b'fake').hexdigest()}  {t.name}\n", encoding="utf-8")
    res = RP.publish(None, "ui", t, commit="abcdef12" * 5, dry_run=True)
    assert res["dry_run"] is True and res["latest"]["kind"] == "ui"
    json.dumps(res)


def test_ci_installs_the_production_versions_and_cpu_torch():
    for name in ("_test.yml", "ci.yml"):
        text = (WFDIR / name).read_text(encoding="utf-8")
        assert "-c constraints-jobs.txt" in text and "torch==2.6.0+cpu" in text, name      # 本番(Jobs イメージ)と同じ版で試す


def test_stg_manual_run_can_skip_tests_but_push_cannot():
    d = _load(WF_STG)
    assert set(_on(d)["workflow_dispatch"]["inputs"]) == {"skip_tests"}
    assert "inputs.skip_tests" in d["jobs"]["test"]["if"] and "workflow_dispatch" in d["jobs"]["test"]["if"]      # push では常にテストする
