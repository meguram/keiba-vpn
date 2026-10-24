"""scripts/dev/secret_scan.sh: 秘密らしい文字列を検出し、ダミー値は通すこと。"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _scan(tmp_path, text):
    repo = tmp_path / "r"
    repo.mkdir()
    run = lambda *a: subprocess.run(a, cwd=repo, check=True, capture_output=True, text=True)  # noqa: E731
    run("git", "init", "-q")
    run("git", "config", "user.email", "t@example.com")
    run("git", "config", "user.name", "t")
    (repo / "a.txt").write_text("ok\n", encoding="utf-8")
    run("git", "add", "a.txt")
    run("git", "commit", "-qm", "base")
    (repo / "b.txt").write_text(text, encoding="utf-8")
    run("git", "add", "b.txt")
    run("git", "commit", "-qm", "add")
    return subprocess.run(["bash", str(ROOT / "scripts/dev/secret_scan.sh"), "HEAD~1..HEAD"], cwd=repo, capture_output=True, text=True)


def test_detects_slack_webhook_without_printing_it(tmp_path):
    secret = "https://hooks.slack.com/services/T0ABCDEFGH1/B0ABCDEFGH1/" + "a1B2c3D4e5F6g7H8i9J0k1L2"
    r = _scan(tmp_path, f"url={secret}\n")
    assert r.returncode == 1 and "slack_webhook" in r.stdout
    assert "a1B2c3D4e5F6g7H8i9J0k1L2" not in r.stdout + r.stderr


def test_allows_placeholders(tmp_path):
    r = _scan(tmp_path, 'GCS_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\\nMASKED\\n-----END PRIVATE KEY-----"\nssh -p <SSH_PORT> a@<VPS_IP>\n')
    assert r.returncode == 0


def test_rejects_netkeiba_data_paths(tmp_path):
    repo = tmp_path / "r2"
    (repo / "dev_fixture").mkdir(parents=True)
    run = lambda *a: subprocess.run(a, cwd=repo, check=True, capture_output=True, text=True)  # noqa: E731
    run("git", "init", "-q")
    run("git", "config", "user.email", "t@example.com")
    run("git", "config", "user.name", "t")
    (repo / "a.txt").write_text("ok\n", encoding="utf-8")
    run("git", "add", "a.txt")
    run("git", "commit", "-qm", "base")
    (repo / "dev_fixture" / "real_day.tar.gz").write_bytes(b"x")
    run("git", "add", "-f", "dev_fixture/real_day.tar.gz")
    run("git", "commit", "-qm", "add")
    r = subprocess.run(["bash", str(ROOT / "scripts/dev/secret_scan.sh"), "HEAD~1..HEAD"], cwd=repo, capture_output=True, text=True)
    assert r.returncode == 1 and "netkeiba の実データ置き場" in r.stdout
