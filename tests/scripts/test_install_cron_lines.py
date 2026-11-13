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
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = tmp_path / "crontab.txt"
    if content is not None:
        store.write_text(content)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    fake = bin_dir / "crontab"
    fake.write_text(f"""#!/bin/sh
if [ "$1" = "-l" ]; then
  [ -z "$FAKE_CRONTAB_FAIL" ] || {{ echo "crontab: cannot open: Permission denied" >&2; exit 1; }}
  [ -f {store} ] || {{ echo "no crontab for test" >&2; exit 1; }}
  cat {store}; exit 0
fi
if [ "$1" = "-" ]; then cat > {store}; else cp "$1" {store}; fi
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


# ── 共通の安全な入れ替え（scripts/cron/lib_crontab.sh。2026-11-13）を使う登録スクリプト ──
SAFE = ["install_host_postprocess_cron.sh", "install_vps_alive_cron.sh", "install_vps_mirror_pull_cron.sh"]


def test_safe_installers_never_wipe_crontab_when_listing_fails(tmp_path):
    """crontab -l が失敗したら、crontab を変えずに止まる（以前は登録する数行だけで root の crontab 全体を置き換えうる作りだった）。"""
    for name in SAFE:
        d = tmp_path / name
        store, env = _fake_crontab(d, FILLER)
        env = {**env, "KEIBA_CRON_ALLOW_NONROOT": "1", "FAKE_CRONTAB_FAIL": "1"}
        r = _run(name, env)
        assert r.returncode != 0 and store.read_text() == FILLER, name


def test_safe_installers_refuse_non_root(tmp_path):
    for name in SAFE:
        store, env = _fake_crontab(tmp_path / name, FILLER)
        r = _run(name, env)          # KEIBA_CRON_ALLOW_NONROOT なし（テストは root でない）
        if os.getuid() != 0:
            assert r.returncode != 0 and store.read_text() == FILLER, name


def test_safe_installers_replace_only_their_tag_and_are_idempotent(tmp_path):
    for name in SAFE:
        tag = {"install_host_postprocess_cron.sh": "KEIBA-HOST-POSTPROCESS", "install_vps_alive_cron.sh": "KEIBA-VPS-ALIVE",
               "install_vps_mirror_pull_cron.sh": "KEIBA-VPS-MIRROR-PULL"}[name]
        store, env = _fake_crontab(tmp_path / name, FILLER + f"# 0 0 * * * old # {tag}\n")     # コメントにした行は同じタグでも残す
        env = {**env, "KEIBA_CRON_ALLOW_NONROOT": "1"}
        r = _run(name, env)
        assert r.returncode == 0, name + r.stdout + r.stderr
        first = store.read_text()
        assert first.count("# OTHER") == 25 and f"# 0 0 * * * old # {tag}" in first
        assert _run(name, env).returncode == 0 and store.read_text() == first, name     # 2 回目も同じ


def test_safe_installers_refuse_a_suspiciously_small_crontab(tmp_path):
    store, env = _fake_crontab(tmp_path, "1 1 * * * echo only # OTHER\n")
    r = _run("install_vps_alive_cron.sh", {**env, "KEIBA_CRON_ALLOW_NONROOT": "1"})
    assert r.returncode != 0 and store.read_text() == "1 1 * * * echo only # OTHER\n"


def test_setup_all_cron_stops_when_listing_fails_and_dedupes_headers(tmp_path):
    """setup_all_cron.sh: crontab -l が失敗したら止まる。見出しのブロック（タグの無い行）が install のたびに重ならない（2026-11-13 に 6 回分重なっていた）。"""
    header = ("# keiba-vpn 全自動スクレイピング cron (setup_all_cron.sh で管理)\n"
              "PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin\n")
    content = FILLER + header * 3
    store, env = _fake_crontab(tmp_path / "fail", content)
    r = subprocess.run(["bash", str(CRON / "setup_all_cron.sh"), "install"], env={**env, "FAKE_CRONTAB_FAIL": "1"},
                       capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and store.read_text() == content
    store, env = _fake_crontab(tmp_path / "ok", content)
    r = subprocess.run(["bash", str(CRON / "setup_all_cron.sh"), "install"], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    new = store.read_text()
    assert new.count("# keiba-vpn 全自動スクレイピング cron") == 1
    # PATH= は消さない（先頭の PATH= を消すと、その後の行のジョブが既定の PATH で動く）。重なりは dedupe_setup_all_headers.sh で
    assert new.count("PATH=/opt/conda") == 3 + 1           # 既存の 3 行は残し、新しいブロックの 1 行が足される
    assert new.count("# OTHER") == 25


def test_dedupe_headers_keeps_every_job_and_one_header(tmp_path):
    header = ("# keiba-vpn 全自動スクレイピング cron (setup_all_cron.sh で管理)\n"
              "PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin\n")
    content = header + FILLER + header + header
    store, env = _fake_crontab(tmp_path, content)
    env = {**env, "KEIBA_CRON_ALLOW_NONROOT": "1"}
    r = subprocess.run(["bash", str(CRON / "dedupe_setup_all_headers.sh")], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    new = store.read_text()
    assert new.count("# keiba-vpn 全自動スクレイピング cron") == 1 and new.count("PATH=/opt/conda") == 1 and new.count("# OTHER") == 25


def test_verify_past_values_cron_is_daily_and_wrapped():
    text = (CRON / "install_verify_past_values_cron.sh").read_text(encoding="utf-8")
    line = re.search(r'^LINES="([^\n]+)"$', text, re.M).group(1)
    assert line.startswith("25 19 * * * ") and "cron_run.sh" in line and "verify-past-values" in line


def test_installers_on_the_safe_replace_keep_a_minimum_line_count():
    """lib_crontab.sh に移った登録スクリプトは、ホストの root の crontab の取り違いに気付くよう下限 20 を付ける（2026-11-13）。"""
    paths = sorted(CRON.glob("install_*.sh")) + sorted((CRON.parent / "ops").glob("install_*cron*.sh"))
    using = [p for p in paths if "crontab_replace_tag" in p.read_text(encoding="utf-8")]
    assert len(using) >= 14
    for p in using:
        text = p.read_text(encoding="utf-8")
        assert 'KEIBA_CRON_MIN_LINES="${KEIBA_CRON_MIN_LINES:-20}"' in text, p.name
        assert "crontab -l 2>/dev/null |" not in text, p.name          # 失敗を握りつぶす読み方が残っていない
        assert subprocess.run(["bash", "-n", str(p)]).returncode == 0, p.name


def test_converted_installers_never_wipe_crontab_when_listing_fails(tmp_path):
    for name in ["install_odds_drift_cron.sh", "install_gcs_mirror_cron.sh", "install_cost_guard_cron.sh"]:
        store, env = _fake_crontab(tmp_path / name, FILLER)
        r = _run(name, {**env, "KEIBA_CRON_ALLOW_NONROOT": "1", "FAKE_CRONTAB_FAIL": "1"})
        assert r.returncode != 0 and store.read_text() == FILLER, name
        store2, env2 = _fake_crontab(tmp_path / (name + "_ok"), FILLER)
        r2 = _run(name, {**env2, "KEIBA_CRON_ALLOW_NONROOT": "1"})
        assert r2.returncode == 0 and store2.read_text().count("# OTHER") == 25, name + r2.stdout + r2.stderr
