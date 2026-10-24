"""scripts/ci/stg_changes.sh（stg の反映で、api / ui のどちらを動かすかの判定）。"""
import subprocess
from pathlib import Path

SH = Path(__file__).resolve().parents[2] / "scripts/ci/stg_changes.sh"


def _git(cwd, *a):
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _decide(tmp_path, files):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t"); _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "base.txt").write_text("x"); _git(tmp_path, "add", "-A"); _git(tmp_path, "commit", "-qm", "a")
    before = _git(tmp_path, "rev-parse", "HEAD")
    for f in files:
        p = tmp_path / f; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("y")
    _git(tmp_path, "add", "-A"); _git(tmp_path, "commit", "-qm", "b")
    out = subprocess.run(["bash", str(SH), before, "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True).stdout
    return dict(line.split("=") for line in out.split())


def test_frontend_only_skips_api(tmp_path):
    assert _decide(tmp_path, ["frontend/components/A.tsx"]) == {"api": "false", "ui": "true"}


def test_docs_and_tests_only_skip_both(tmp_path):
    assert _decide(tmp_path, ["docs/x.md", "tests/a.py", "README.md"]) == {"api": "false", "ui": "false"}


def test_backend_change_runs_api_not_ui(tmp_path):
    assert _decide(tmp_path, ["src/api/app.py", "config/x.json"]) == {"api": "true", "ui": "false"}


def test_unknown_base_runs_everything(tmp_path):
    _git(tmp_path, "init", "-q")
    for before in ("", "0" * 40, "deadbeef"):
        out = subprocess.run(["bash", str(SH), before, "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True).stdout
        assert "api=true" in out and "ui=true" in out
