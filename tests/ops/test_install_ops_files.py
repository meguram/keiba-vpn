"""VPS の ops ファイルの入れ替え（scripts/ops/vps/install_ops_files.sh、方法 A）のテスト。

root 以外で KEIBA_OPS_TEST_ROOT を指定すると、置き場所が一時ディレクトリの下に差し替わり、systemctl は呼ばずに記録だけ残る。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from src.ops import release_publish as rp

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/ops/vps/install_ops_files.sh"
SUDOERS = ROOT / "scripts/ops/vps/templates/sudoers-keiba-ops-install"
VER = "fetch-0123abcd-202611131700"

pytestmark = pytest.mark.skipif(os.geteuid() == 0 or not shutil.which("flock"), reason="root 以外・flock が要る")


def _run(tr: Path, *args: str, **extra: str) -> subprocess.CompletedProcess:
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "KEIBA_OPS_TEST_ROOT": str(tr), "LANG": "C.UTF-8", **extra}
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=60)


def _make_release(tr: Path, ver: str = VER) -> Path:
    rel = tr / "opt/keiba-fetch/releases" / ver
    for f in rp.OPS_FILES:
        dst = rel / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / f, dst)
        dst.chmod(0o644)
    (rel / rp.OPS_MANIFEST).write_text(json.dumps(rp.ops_manifest(ROOT)))
    for d in [tr, *sorted({p for p in rel.rglob("*") if p.is_dir()}), *rel.parents]:
        if d == tr or tr in d.parents:
            d.chmod(0o755)
    return rel


@pytest.fixture
def tr(tmp_path):
    root = tmp_path / "root"
    root.mkdir(mode=0o755)
    (root / "run").mkdir()
    rel = _make_release(root)
    os.symlink(rel, root / "opt/keiba-fetch/current")          # release-pull が切り替えた今の版
    return root


def _calls(tr: Path) -> list[str]:
    p = tr / "systemctl.calls"
    return p.read_text().splitlines() if p.exists() else []


def test_allowlist_matches_ops_files():
    """スクリプトの許可リストの元ファイルと、配布物に入れる OPS_FILES が一致する（片方だけに足すと入れ替えられない）。"""
    text = SCRIPT.read_text()
    block = text[text.index("declare -A FILES=("):text.index("declare -A AFTER=(")]
    block = block.replace("$T/", "scripts/ops/vps/templates/")
    srcs = set(re.findall(r'(scripts/ops/vps/[\w./@-]+)\|', block))
    assert srcs == set(rp.OPS_FILES)
    for f in rp.OPS_FILES:
        assert (ROOT / f).is_file(), f


def test_install_copies_files_after_sha_check_and_reloads(tr):
    r = _run(tr, "install", VER, "fetch-timers")
    assert r.returncode == 0, r.stderr
    unit = tr / "etc/systemd/system/keiba-fetch-reserve.timer"
    assert unit.read_bytes() == (ROOT / "scripts/ops/vps/templates/keiba-fetch-reserve.timer").read_bytes()
    assert oct(unit.stat().st_mode & 0o777) == "0o644"
    assert "systemctl daemon-reload" in _calls(tr)
    assert not any("enable" in c for c in _calls(tr))          # fetch-timers は有効化しない
    log = (tr / "var/log/keiba-ops-install.log").read_text()
    assert "install\t" + VER + "\tfetch-timers" in log


def test_install_cache_warm_is_executable_and_status_lists_it(tr):
    assert _run(tr, "install", VER, "cache-warm").returncode == 0
    dst = tr / "opt/keiba-ui/bin/cache_warm.py"
    assert oct(dst.stat().st_mode & 0o777) == "0o755"
    assert _calls(tr) == []                                   # 温め役は入れた後の操作なし
    s = _run(tr, "status", "cache-warm")
    assert s.returncode == 0 and "入っている" in s.stdout and VER in s.stdout


@pytest.mark.parametrize("name", ["evil", "../cache-warm", "cache-warm;id", ""])
def test_rejects_unknown_name(tr, name):
    r = _run(tr, "install", VER, name)
    assert r.returncode != 0
    assert not (tr / "opt/keiba-ui/bin/cache_warm.py").exists()


@pytest.mark.parametrize("ver", ["../../etc", "fetch-0123abcd-2026111317", "fetch-0123ABCD-202611131700", "fetch-0123abcd-202611131700/..", "x"])
def test_rejects_bad_version(tr, ver):
    r = _run(tr, "install", ver, "cache-warm")
    assert r.returncode == 2 and "版の形が不正" in r.stderr


def test_rejects_version_other_than_current(tr):
    _make_release(tr, "fetch-11111111-202601010000")                # 残っている古い版
    r = _run(tr, "install", "fetch-11111111-202601010000", "cache-warm")
    assert r.returncode == 1 and "今の版" in r.stderr
    assert not (tr / "opt/keiba-ui/bin/cache_warm.py").exists()


def test_rejects_symlinked_destination(tr, tmp_path):
    bindir = tr / "opt/keiba-ui/bin"
    bindir.mkdir(parents=True)
    os.symlink(tmp_path / "elsewhere.py", bindir / "cache_warm.py")
    r = _run(tr, "install", VER, "cache-warm")
    assert r.returncode == 1 and "リンク" in r.stderr
    assert not (tmp_path / "elsewhere.py").exists()


def test_failed_after_step_restores_backup(tr):
    unit = tr / "etc/systemd/system/keiba-sched-status.timer"
    unit.parent.mkdir(parents=True)
    unit.write_text("old\n")
    r = _run(tr, "install", VER, "sched-status-units", KEIBA_OPS_TEST_FAIL_SYSTEMCTL="enable")
    assert r.returncode == 1 and "退避から戻した" in r.stderr
    assert unit.read_text() == "old\n"
    assert not (tr / "etc/systemd/system/keiba-sched-status.service").exists()   # 入れる前に無かったものは消す
    assert not list((tr / "var/lib/keiba-ops/backup/sched-status-units").iterdir())   # 失敗した回の退避は残さない


def test_rejects_missing_or_symlinked_version(tr):
    assert _run(tr, "install", "fetch-ffffffff-202601010000", "cache-warm").returncode == 1
    link = tr / "opt/keiba-fetch/releases/fetch-eeeeeeee-202601010000"
    os.symlink(tr / "opt/keiba-fetch/releases" / VER, link)
    (tr / "opt/keiba-fetch/current").unlink()
    os.symlink(link, tr / "opt/keiba-fetch/current")
    r = _run(tr, "install", "fetch-eeeeeeee-202601010000", "cache-warm")
    assert r.returncode == 1 and not (tr / "opt/keiba-ui/bin/cache_warm.py").exists()


def test_rejects_sha_mismatch_and_installs_nothing(tr):
    rel = tr / "opt/keiba-fetch/releases" / VER
    (rel / "scripts/ops/vps/templates/keiba-mirror-vps-push.timer").write_text("[Timer]\nOnCalendar=*:*\n")
    r = _run(tr, "install", VER, "fetch-timers")
    assert r.returncode == 1 and "sha256 の不一致" in r.stderr
    assert not (tr / "etc/systemd/system/keiba-fetch-reserve.service").exists()   # 先頭のファイルも入れない
    assert not any(c.startswith("systemctl") for c in _calls(tr))                 # daemon-reload もしない
    assert "中止" in (tr / "var/log/keiba-ops-install.log").read_text()


@pytest.mark.parametrize("which", ["release", "releases", "file", "parent"])
def test_rejects_writable_dir_or_file(tr, which):
    rel = tr / "opt/keiba-fetch/releases" / VER
    target = {"release": rel, "releases": rel.parent, "file": rel / "scripts/ops/vps/cache_warm.py",
              "parent": tr / "opt"}[which]
    target.chmod(0o777 if target.is_dir() else 0o666)
    r = _run(tr, "install", VER, "cache-warm")
    assert r.returncode == 1 and "書ける" in r.stderr
    assert not (tr / "opt/keiba-ui/bin/cache_warm.py").exists()


def test_rejects_manifest_missing_entry(tr):
    rel = tr / "opt/keiba-fetch/releases" / VER
    (rel / rp.OPS_MANIFEST).write_text(json.dumps({"schema": 1, "files": {}}))
    r = _run(tr, "install", VER, "cache-warm")
    assert r.returncode == 1 and "ops_manifest.json に無い" in r.stderr


def test_rollback_restores_previous_and_removes_new(tr):
    dst = tr / "opt/keiba-ui/bin/cache_warm.py"
    dst.parent.mkdir(parents=True)
    dst.write_text("old\n")
    assert _run(tr, "install", VER, "cache-warm").returncode == 0
    assert dst.read_text() != "old\n"
    r = _run(tr, "rollback", "cache-warm")
    assert r.returncode == 0, r.stderr
    assert dst.read_text() == "old\n"
    # 入れる前に無かったユニットは、rollback で消える
    assert _run(tr, "install", VER, "sched-status-units").returncode == 0
    assert "systemctl enable --now keiba-sched-status.timer" in _calls(tr)
    assert _run(tr, "rollback", "sched-status-units").returncode == 0
    assert not (tr / "etc/systemd/system/keiba-sched-status.timer").exists()
    calls = _calls(tr)
    assert "systemctl disable --now keiba-sched-status.timer" in calls            # 入れる前に無かった timer は止める
    assert calls[-1] == "systemctl daemon-reload"                                 # 消した timer を有効化し直さない
    # 退避が無ければ何もしない
    r = _run(tr, "rollback", "fetch-timers")
    assert r.returncode == 0 and "退避がありません" in r.stdout


def test_requires_root_without_test_root():
    r = subprocess.run(["bash", str(SCRIPT), "status"], env={"PATH": os.environ.get("PATH", "")}, capture_output=True, text=True)
    assert r.returncode == 2 and "root" in r.stderr


def test_sudoers_line_only_allows_the_script():
    lines = [ln for ln in SUDOERS.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    assert lines == ["deploy ALL=(root) NOPASSWD: /opt/keiba-ui/bin/install_ops_files.sh *"]


def test_fetch_bundle_contains_ops_files_and_manifest(tmp_path):
    import tarfile

    tb = rp.build_fetch_bundle(tmp_path, commit="0123abcd" * 5)
    with tarfile.open(tb) as tf:
        names = set(tf.getnames())
        man = json.loads(tf.extractfile(rp.OPS_MANIFEST).read())
        for f in rp.OPS_FILES:
            assert f in names
            assert tf.getmember(f).mode & 0o022 == 0
            assert man["files"][f] == rp.sha256_file(ROOT / f)
