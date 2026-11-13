"""crontab の登録スクリプト: 時刻（ホストの cron は UTC）と、タグの行だけを入れ替えること（偽の crontab で確かめる）。"""
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CRON = ROOT / "scripts" / "cron"


def _line(script: str) -> str:
    text = (CRON / script).read_text(encoding="utf-8")
    return re.search(r'^LINE="([^"]+)"', text, re.M).group(1)


def test_shared_login_runs_at_0005_jst_in_utc():
    """2026-10-23〜11-13 は `5 0`（UTC＝JST 09:05）で、毎月 1 日の朝 09:05 にログインが切れていた。"""
    line = _line("install_shared_login_cron.sh")
    assert line.startswith("5 15 * * * ") and "shared_login_daily.sh" in line


def _fake_crontab(tmp_path: Path, content: str | None):
    store = tmp_path / "crontab.txt"
    if content is not None:
        store.write_text(content)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    fake = bin_dir / "crontab"
    fake.write_text(f"""#!/bin/sh
if [ "$1" = "-l" ]; then
  [ -f {store} ] || {{ echo "no crontab for test" >&2; exit 1; }}
  cat {store}; exit 0
fi
cp "$1" {store}
""")
    fake.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "KEIBA_CRON_BACKUP_DIR": str(tmp_path / "bk")}
    return store, env


def _run(script, env, *args):
    proj = Path(env["KEIBA_CRON_BACKUP_DIR"]).parent / "proj"
    proj.mkdir(exist_ok=True)
    return subprocess.run(["bash", str(CRON / script), str(proj), *args], env=env, capture_output=True, text=True, timeout=30)


FILLER = "".join(f"{i} 1 * * * echo {i} # OTHER\n" for i in range(25))


def test_serving_state_retags_old_lines_and_keeps_the_rest(tmp_path):
    old = ("35 22 * * * cd /proj && TZ=Asia/Tokyo bash scripts/cron/cron_run.sh serving-state-precompute -- env x precompute --date $(TZ=Asia/Tokyo date +\\%Y\\%m\\%d)\n"
           "# 履歴にいて馬主が未登録の馬を補う\n"
           "10 23 * * * cd /proj && TZ=Asia/Tokyo bash scripts/cron/cron_run.sh serving-state-profile -- env x profile\n")
    store, env = _fake_crontab(tmp_path, FILLER + old)
    r = _run("install_serving_state_cron.sh", env)
    assert r.returncode == 0, r.stdout + r.stderr
    new = store.read_text()
    assert new.count("# OTHER") == 25 and new.count("# KEIBA-SERVING-STATE") == 2
    assert "env x" not in new                                                  # タグの無い旧行は消える
    assert "precompute --date $(TZ=Asia/Tokyo date +\\%Y\\%m\\%d) # KEIBA-SERVING-STATE" in new   # % は crontab 用に逃がす
    r2 = _run("install_serving_state_cron.sh", env)                            # 2 回目も同じ（冪等）
    assert r2.returncode == 0 and store.read_text() == new
    assert list((tmp_path / "bk").glob("crontab_before_serving_state_*.txt"))


def test_sporthorse_remove_only_drops_its_line_and_handles_no_crontab(tmp_path):
    tag = "0 13 * * * cd /proj && bash x # KEIBA-SPORTHORSE-BIRTHS\n"
    store, env = _fake_crontab(tmp_path, "1 1 * * * echo keep # OTHER\n" + tag)
    r = _run("install_sporthorse_births_cron.sh", env, "--remove")
    assert r.returncode == 0, r.stdout + r.stderr
    assert store.read_text() == "1 1 * * * echo keep # OTHER\n"
    store2, env2 = _fake_crontab(tmp_path / "empty", None)                    # crontab が無い利用者でも壊さない
    r2 = _run("install_sporthorse_births_cron.sh", env2, "--remove")
    assert r2.returncode == 0 and store2.read_text() == ""


def test_remove_flag_in_any_position_and_bad_args(tmp_path):
    tag = "0 13 * * * cd /proj && bash x # KEIBA-SPORTHORSE-BIRTHS\n# 0 13 * * * old # KEIBA-SPORTHORSE-BIRTHS\n"
    store, env = _fake_crontab(tmp_path, "1 1 * * * echo keep # OTHER\n" + tag)
    r = subprocess.run(["bash", str(CRON / "install_sporthorse_births_cron.sh"), "--remove"], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stdout + r.stderr                          # 先頭の --remove も効き、コメントの行で数がずれない
    assert "echo keep" in store.read_text() and "bash x" not in store.read_text()
    r2 = subprocess.run(["bash", str(CRON / "install_serving_state_cron.sh"), "--bogus"], env=env, capture_output=True, text=True, timeout=30)
    assert r2.returncode == 2
