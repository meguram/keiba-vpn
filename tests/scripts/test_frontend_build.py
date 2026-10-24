"""stg の Next 本番ビルド運用: 再ビルドが要るかの判定（scripts/server/frontend_build.sh）。"""

import os
import subprocess
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "server" / "frontend_build.sh"


def _needed(fdir: Path, sig: str) -> tuple[bool, str]:
    r = subprocess.run(
        ["bash", "-c", f'source "{SCRIPT}"; frontend_build_needed "$1" ".next-stg" "$2"', "x", str(fdir), sig],
        capture_output=True, text=True,
    )
    return r.returncode == 0, r.stdout.strip()


def _setup(tmp_path: Path) -> Path:
    f = tmp_path / "frontend"
    (f / "app").mkdir(parents=True)
    (f / "app" / "page.tsx").write_text("x")
    (f / "package.json").write_text("{}")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "ports.env").write_text("A=1\n")
    return f


def _built(f: Path, sig: str) -> None:
    d = f / ".next-stg"
    d.mkdir()
    for p in f.rglob("*"):
        if p.is_file() and ".next-stg" not in p.parts:
            os.utime(p, (time.time() - 100, time.time() - 100))
    os.utime(f.parent / "config" / "ports.env", (time.time() - 100, time.time() - 100))
    (d / "BUILD_ID").write_text("abc")
    (d / ".build-env").write_text(sig)


def test_no_build_id_means_build(tmp_path):
    f = _setup(tmp_path)
    (f / ".next-stg").mkdir()                       # 開発モードの残骸（BUILD_ID なし）
    need, why = _needed(f, "sig")
    assert need and "BUILD_ID" in why


def test_up_to_date_build_is_not_rebuilt(tmp_path):
    f = _setup(tmp_path)
    _built(f, "sig")
    need, _ = _needed(f, "sig")
    assert not need


def test_newer_source_triggers_rebuild(tmp_path):
    f = _setup(tmp_path)
    _built(f, "sig")
    time.sleep(0.05)
    (f / "app" / "page.tsx").write_text("changed")
    need, why = _needed(f, "sig")
    assert need and "app/page.tsx" in why


def test_newer_ports_env_triggers_rebuild(tmp_path):
    f = _setup(tmp_path)
    _built(f, "sig")
    time.sleep(0.05)
    (tmp_path / "config" / "ports.env").write_text("A=2\n")
    need, why = _needed(f, "sig")
    assert need and "ports.env" in why


def test_env_signature_change_triggers_rebuild(tmp_path):
    f = _setup(tmp_path)
    _built(f, "KEIBA_ENV=stg|NEXT_PUBLIC_X=1|")
    need, why = _needed(f, "KEIBA_ENV=stg|NEXT_PUBLIC_X=2|")
    assert need and "環境" in why


def test_signature_includes_public_vars_and_urls():
    r = subprocess.run(
        ["bash", "-c", f'source "{SCRIPT}"; frontend_env_signature'],
        capture_output=True, text=True,
        env={**os.environ, "KEIBA_ENV": "stg", "KEIBA_API_URL": "http://a", "NEXT_PUBLIC_PICKER_PAST_DAYS": "150"},
    )
    out = r.stdout
    assert "KEIBA_ENV=stg|" in out and "KEIBA_API_URL=http://a|" in out and "NEXT_PUBLIC_PICKER_PAST_DAYS=150|" in out


def _bash(cmd: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", f'source "{SCRIPT}"; {cmd}'], capture_output=True, text=True, **kw)


def test_load_guard_boundaries():
    assert _bash("frontend_load_too_high 19.3 24 0.8").returncode == 0     # 閾値 19.2 を超える
    assert _bash("frontend_load_too_high 19.2 24 0.8").returncode == 1     # 超えない（等しい）
    assert _bash("frontend_load_too_high 5 24").returncode == 1            # 係数の既定は 0.8
    assert _bash("frontend_load_too_high 47 24").returncode == 0           # 本番ビルド中に実際に出た負荷


def test_remove_dist_refuses_paths_outside_frontend_next(tmp_path):
    keep = tmp_path / "other"
    keep.mkdir()
    (keep / "x").write_text("1")
    r = _bash(f'frontend_remove_dist "{keep}"')
    assert r.returncode == 1 and (keep / "x").exists()
    r = _bash('frontend_remove_dist ""')
    assert r.returncode == 1
    d = tmp_path / "frontend" / ".next-stg-build"
    d.mkdir(parents=True)
    (d / "BUILD_ID").write_text("a")
    assert _bash(f'frontend_remove_dist "{d}"').returncode == 0 and not d.exists()
    assert _bash(f'frontend_remove_dist "{d}"').returncode == 0           # 無くても成功


# ── 停止処理（/proc ベース。ss / netstat / lsof に依存しない）──────────────────────────
import signal
import socket
import sys


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # ゾンビは死んだものとして扱う
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0] != "Z"
    except Exception:
        return False


def _wait_listening(port: int, timeout: float = 8.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.1)
    return False


def _spawn_listener(port: int, cwd: Path, *, ignore_term: bool = False, orphan: bool = True) -> int:
    """port で LISTEN する HTTP サーバを起動して PID を返す。orphan=True なら親を終了させて ppid=1 に近い状態にする。"""
    code = (
        "import signal,http.server,socketserver,sys\n"
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "")
        + f"socketserver.TCPServer.allow_reuse_address=True\n"
        + f"http.server.test(HandlerClass=http.server.SimpleHTTPRequestHandler, port={port}, bind='127.0.0.1')\n"
    )
    if orphan:
        # sh が子を起動してすぐ終了 → 子は孤児（next-server が 1 の子になって残った状況の再現）
        out = subprocess.run(
            ["sh", "-c", f'cd "{cwd}" && {sys.executable} -c "{code}" >/dev/null 2>&1 & echo $!'],
            capture_output=True, text=True,
        ).stdout.strip()
        pid = int(out)
    else:
        pid = subprocess.Popen([sys.executable, "-c", code], cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).pid
    assert _wait_listening(port), "テスト用サーバが起動しない"
    return pid


def _stop_all(port: int, cwd: Path, pid_file: Path | None = None, wait: int = 1) -> subprocess.CompletedProcess:
    args = f'frontend_stop_all {port} "{cwd}"' + (f' "{pid_file}"' if pid_file else "")
    return subprocess.run(
        ["bash", "-c", f'source "{SCRIPT}"; {args}'], capture_output=True, text=True,
        env={**os.environ, "KEIBA_STOP_WAIT_SEC": str(wait)},
    )


def _cleanup(*pids: int) -> None:
    for p in pids:
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_listener_pids_and_port_in_use(tmp_path):
    port = _free_port()
    pid = _spawn_listener(port, tmp_path, orphan=False)
    try:
        r = subprocess.run(["bash", "-c", f'source "{SCRIPT}"; frontend_port_in_use {port} && echo IN; frontend_listener_pids {port}'],
                           capture_output=True, text=True)
        assert "IN" in r.stdout and str(pid) in r.stdout.split()
    finally:
        _cleanup(pid)
    time.sleep(0.3)
    assert subprocess.run(["bash", "-c", f'source "{SCRIPT}"; frontend_port_in_use {port}']).returncode == 1


def test_descendants_include_grandchildren():
    p = subprocess.Popen(["bash", "-c", "sleep 30 & sleep 30 & wait"])
    time.sleep(0.5)
    try:
        out = subprocess.run(["bash", "-c", f'source "{SCRIPT}"; frontend_descendants {p.pid}'], capture_output=True, text=True).stdout.split()
        assert len(out) == 2 and all(_alive(int(x)) for x in out)
    finally:
        subprocess.run(["pkill", "-P", str(p.pid)])
        p.kill()
        p.wait()


def test_stop_all_kills_orphan_listener_with_matching_cwd(tmp_path):
    fe = tmp_path / "frontend"
    fe.mkdir()
    port = _free_port()
    pid = _spawn_listener(port, fe)                       # 孤児の next-server 相当（pid ファイルの親はもういない）
    pid_file = tmp_path / ".frontend.pid"
    pid_file.write_text("999999\n")                        # 死んだ npm の PID が残っている状況
    try:
        r = _stop_all(port, fe, pid_file)
        assert r.returncode == 0, r.stdout + r.stderr
        assert not _alive(pid) and not pid_file.exists()
    finally:
        _cleanup(pid)


def test_stop_all_force_kills_a_process_that_ignores_term(tmp_path):
    fe = tmp_path / "frontend"
    fe.mkdir()
    port = _free_port()
    pid = _spawn_listener(port, fe, ignore_term=True)
    try:
        r = _stop_all(port, fe, None, wait=1)
        assert r.returncode == 0 and not _alive(pid)
        assert "KILL" in r.stderr
    finally:
        _cleanup(pid)


def test_stop_all_does_not_touch_a_listener_from_another_directory(tmp_path):
    fe = tmp_path / "frontend"
    other = tmp_path / "elsewhere"
    fe.mkdir()
    other.mkdir()
    port = _free_port()
    pid = _spawn_listener(port, other)                    # cwd が違う＝別の起動元
    try:
        r = _stop_all(port, fe, None, wait=1)
        assert r.returncode == 1 and _alive(pid)           # 止められないので中止（旧を残す）
    finally:
        _cleanup(pid)


def test_stop_all_kills_the_whole_tree_under_the_pid_file(tmp_path):
    fe = tmp_path / "frontend"
    fe.mkdir()
    port = _free_port()
    # npm 相当の親（コマンドラインに npm を含む）が子として LISTEN するサーバを持つ
    parent = subprocess.Popen(
        ["bash", "-c", f'exec -a npm-run-start bash -c \'cd "{fe}"; {sys.executable} -m http.server {port} --bind 127.0.0.1 >/dev/null 2>&1 & wait\''],
    )
    assert _wait_listening(port)
    pid_file = tmp_path / ".frontend.pid"
    pid_file.write_text(f"{parent.pid}\n")
    kids = subprocess.run(["bash", "-c", f'source "{SCRIPT}"; frontend_descendants {parent.pid}'], capture_output=True, text=True).stdout.split()
    try:
        assert kids
        r = _stop_all(port, fe, pid_file)
        assert r.returncode == 0, r.stdout + r.stderr
        parent.wait(timeout=5)
        assert not any(_alive(int(k)) for k in kids)
    finally:
        _cleanup(parent.pid, *[int(k) for k in kids])


def test_verify_build_checks_build_specific_url_and_listener_ancestry(tmp_path):
    bid = "BUILDxyz123"
    root = tmp_path / "srv"
    (root / "_next" / "static" / bid).mkdir(parents=True)
    (root / "_next" / "static" / bid / "_buildManifest.js").write_text("x")
    port = _free_port()
    pid = _spawn_listener(port, root, orphan=False)
    try:
        def verify(b, started):
            return subprocess.run(["bash", "-c", f'source "{SCRIPT}"; frontend_verify_build {port} {b} {started}'],
                                  capture_output=True, text=True)
        assert verify(bid, pid).returncode == 0                       # 正しいビルドで、起動した PID 自身が持ち主
        assert verify("OTHERID", pid).returncode == 1                 # 別ビルド ID（旧サーバが応答している状況）
        assert verify(bid, 999999).returncode == 1                    # 持ち主が起動した PID の子孫ではない（旧プロセスが残っている）
    finally:
        _cleanup(pid)
