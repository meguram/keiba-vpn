"""VPS の API キャッシュ(nginx proxy_cache)の設定が、上流の切替で外れず、会員の応答を匿名へ出さない作りであることを確かめる。"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

VPS = Path(__file__).resolve().parents[2] / "scripts" / "ops" / "vps"
TPL = VPS / "templates"
SITE_TEMPLATES = ["nginx-keiba-ui.conf", "nginx-keiba-ui-cloudrun.conf", "nginx-keiba-ui-stg.conf"]


def _location(text, head):
    m = re.search(re.escape(head) + r" \{(.*?)\n    \}", text, re.S)
    assert m, f"{head} の location が見つからない"
    return m.group(1)


@pytest.mark.parametrize("name", SITE_TEMPLATES)
def test_api_and_image_locations_each_include_only_their_cache(name):
    text = (TPL / name).read_text(encoding="utf-8")
    api = _location(text, "location ~ ^/api/")
    img = _location(text, "location ~ ^/data/image/")
    assert "snippets/keiba-api-cache*.conf" in api and "keiba-img-cache" not in api
    assert "snippets/keiba-img-cache*.conf" in img and "keiba-api-cache" not in img


@pytest.mark.parametrize("name", SITE_TEMPLATES)
def test_image_location_keeps_upstream_auth(name):
    img = _location((TPL / name).read_text(encoding="utf-8"), "location ~ ^/data/image/")
    auth = "keiba-access.conf" if name == "nginx-keiba-ui.conf" else "keiba-proxykey.conf"
    assert f"include /etc/nginx/{auth};" in img, "画像の取得先に認証ヘッダを付けないと 401/403 になる"


def test_image_snippet_has_only_cache_directives():
    text = (TPL / "nginx-img-cache.conf").read_text(encoding="utf-8")
    directives = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    assert "location" not in directives and "proxy_pass" not in directives and "proxy_set_header" not in directives
    assert "proxy_cache keiba_img;" in directives and "proxy_cache_valid 200 7d;" in directives


def test_snippet_never_serves_member_requests_from_cache():
    text = (TPL / "nginx-api-cache.conf").read_text(encoding="utf-8")
    assert "proxy_cache_bypass $keiba_api_nocache $keiba_refresh;" in text
    assert 'proxy_set_header X-Keiba-Refresh ""' in text, "合言葉の見出しは上流へ渡さない"
    assert "proxy_no_cache $keiba_api_nocache;" in text
    directives = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    assert "proxy_cache_valid" not in directives, "上流が public を付けたものだけ保存する（既定の保存時間を足さない）"
    assert "$server_port" in directives, "stg と本番は同じ nginx で上流が違うのでキーにポートを含める"
    zones = (TPL / "nginx-zones.conf").read_text(encoding="utf-8")
    assert "keys_zone=keiba_apicache" in zones
    m = re.search(r"keys_zone=keiba_apicache:\S+ max_size=(\d+)[mg] inactive=(\d+)d", zones)
    assert m and int(m.group(2)) >= 32, "確定済みレースを1か月置くので、未参照の失効(inactive)は32日以上"
    assert "inactive=35d" in (VPS / "vps_09_api_cache.sh").read_text(encoding="utf-8"), "稼働中のゾーン行も同じ保持設定に直す"
    assert re.search(r'map "\$http_cookie\$http_authorization" \$keiba_api_nocache \{\s*default 1;\s*""\s*0;', zones)


def test_refresh_token_fits_the_map_hash_bucket():
    """合言葉(48 文字)は既定のバケット 64 に収まらず nginx -t が落ち、バケットを足すと既存設定と重複して落ちた(2026-10-29)。
    VPS の既存設定に依存しないよう、合言葉を 32 文字にして既定のまま収める。"""
    zones = (TPL / "nginx-zones.conf").read_text(encoding="utf-8")
    assert not re.search(r"^\s*map_hash_bucket_size", zones, re.M), "既存の nginx 設定と重複するので足さない"
    sh = (VPS / "vps_09_api_cache.sh").read_text(encoding="utf-8")
    assert "head -c 16" in sh and "head -c 24" not in sh
    assert "!= 32" in sh, "長い（旧 48 文字）合言葉が残っていたら作り直す"


@pytest.mark.skipif(shutil.which("nginx") is None, reason="nginx が無い環境では実機検証をしない")
def test_verify_script_passes():
    r = subprocess.run([sys.executable, str(VPS / "verify_api_cache.py")], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


def _fake_entry(d, name, key):
    (d / name).write_bytes(f"\x01\x02junk\nKEY: {key}\n\r\nbody".encode())


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash が無い")
def test_purge_script_match_and_all(tmp_path):
    cache = tmp_path / "cache"; (cache / "a" / "bc").mkdir(parents=True)
    _fake_entry(cache / "a" / "bc", "1", "8080example.com/api/race/202506050811")
    _fake_entry(cache / "a" / "bc", "2", "8081example.com/api/race/202506050811")
    _fake_entry(cache / "a", "3", "8080example.com/api/race-list/20260906")
    shim = tmp_path / "bin"; shim.mkdir()
    (shim / "nginx").write_text("#!/bin/sh\nexit 0\n"); (shim / "nginx").chmod(0o755)   # reload は何もしない
    import os
    env = {**os.environ, "PATH": f"{shim}{os.pathsep}{os.environ['PATH']}"}
    run = lambda *a: subprocess.run(["bash", str(VPS / "purge_api_cache.sh"), *a], capture_output=True, text=True, env=env)  # noqa: E731
    left = lambda: sorted(p.name for p in cache.rglob("*") if p.is_file())  # noqa: E731
    r = run("--match", "^8081", str(cache))
    assert r.returncode == 0 and "1 件" in r.stdout and left() == ["1", "3"], r.stdout + r.stderr
    r = run("--match", "api/race-list/", str(cache))
    assert r.returncode == 0 and left() == ["1"]
    r = run("--match", "nothing-matches", str(cache))
    assert r.returncode == 0 and "0 件" in r.stdout and left() == ["1"]
    r = run(str(cache))
    assert r.returncode == 0 and "1 件" in r.stdout and left() == []
    assert run("--match").returncode == 2
    assert run(str(tmp_path / "missing")).returncode == 1
