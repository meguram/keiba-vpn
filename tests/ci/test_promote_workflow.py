"""promote.yml（昇格のワークフロー）と WIF 準備・SHA 固定スクリプトの検査。ネットワーク・GCP には接続しない。"""
import os
import re
import stat
import subprocess
import textwrap
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WF = ROOT / ".github/workflows/promote.yml"


def _wf() -> dict:
    return yaml.safe_load(WF.read_text(encoding="utf-8"))


def _steps(job: dict) -> list[dict]:
    return job.get("steps", [])


def test_triggers_and_least_privilege():
    d = _wf()
    on = d.get("on") or d.get(True)               # PyYAML は on を True に読む
    assert on["push"]["branches"] == ["main"]
    assert set(on["workflow_dispatch"]["inputs"]) == {"skip_tests", "deploy_target"}
    assert "pull_request_target" not in on and "pull_request" not in on
    assert d["permissions"] == {"contents": "read"}              # 既定は読み取りのみ
    assert d["concurrency"]["cancel-in-progress"] is False


def test_only_publish_jobs_get_id_token_and_approval():
    jobs = _wf()["jobs"]
    pub = jobs["publish-ui"]
    assert pub["environment"] == "production" and pub["permissions"]["id-token"] == "write"
    assert pub["needs"] == ["build-ui"]
    for name in ("test", "build-ui"):
        assert "id-token" not in (jobs[name].get("permissions") or {})   # ビルドは GCP に入れない
        assert "environment" not in jobs[name]
    # 通信の切り替えは別の承認
    assert jobs["api-traffic"]["environment"] == "production-traffic"
    assert jobs["api-traffic"]["needs"] == ["api-revision"]


def test_api_jobs_default_off():
    jobs = _wf()["jobs"]
    for n in ("api-revision", "api-traffic"):
        assert "ENABLE_API_PROMOTE" in jobs[n]["if"] and "'true'" in jobs[n]["if"]


def test_auth_uses_wif_without_key_files_or_secrets():
    text = WF.read_text(encoding="utf-8")
    assert "credentials_json" not in text and "GCP_SA_KEY" not in text and "secrets." not in text
    assert text.count("workload_identity_provider:") >= 1
    jobs = _wf()["jobs"]
    auths = [s for s in _steps(jobs["publish-ui"]) if "google-github-actions/auth" in s.get("uses", "")]
    assert len(auths) == 1 and "vars.WIF_PROVIDER" in auths[0]["with"]["workload_identity_provider"]


def test_publish_goes_through_release_publish_and_checksums():
    steps = _steps(_wf()["jobs"]["publish-ui"])
    runs = "\n".join(s.get("run", "") for s in steps)
    assert "sha256sum -c" in runs
    assert "src.ops.release_publish publish-ui" in runs
    # 公開の前に認証、認証の前に検算
    order = [i for i, s in enumerate(steps) if "sha256sum" in s.get("run", "")]
    auth = [i for i, s in enumerate(steps) if "google-github-actions/auth" in s.get("uses", "")]
    pubi = [i for i, s in enumerate(steps) if "publish-ui" in s.get("run", "")]
    assert order[0] < auth[0] < pubi[0]


def test_ci_never_scrapes():
    text = WF.read_text(encoding="utf-8")
    assert not re.search(r"auto_scrape|src\.scraper|netkeiba\.com|scrape_", text)


def test_actions_are_pinned_or_marked_for_pinning():
    text = WF.read_text(encoding="utf-8")
    refs = re.findall(r"uses:\s*([^\s#]+)(?:\s*#\s*(\S+))?", text)
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
    assert "assertion.ref=='refs/heads/main'" in out and "assertion.environment=='production'" in out
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
    text = WF.read_text(encoding="utf-8")
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
