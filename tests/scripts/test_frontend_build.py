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
