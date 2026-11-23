"""vps_09_api_cache.sh のサイト・zones・テンプレートの書き換え（適用・2 回目・--remove）を、偽の nginx / systemctl で確かめる。

本物の nginx の構文の確かめは、VPS 上の verify_api_cache.py と、本番の設定の写しでの試し（docs/operations/edge_fix_20261123.md）。
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

VPS = Path(__file__).resolve().parents[2] / "scripts" / "ops" / "vps"
TPL = VPS / "templates"
OLD = "limit_req zone=keiba_api burst=60 nodelay;"
EXT = "limit_req zone=keiba_api_ext burst=60 nodelay;"
LO = "limit_req zone=keiba_api_lo burst=300 nodelay;"


def _old_form(text: str) -> str:
    """今の VPS の形（/api/ も旧の 1 つの枠）に戻したもの。"""
    out = []
    for line in text.splitlines(keepends=True):
        if LO in line:
            continue
        if EXT in line:
            line = line[:line.index(EXT)] + OLD + "\n"
        out.append(line)
    return "".join(out)


def _api_block(text: str) -> str:
    b = text[text.index("location ~ ^/api/ {"):]
    return b[:b.index("\n    }")]


def _img_block(text: str) -> str:
    b = text[text.index("location ~ ^/data/image/ {"):]
    return b[:b.index("\n    }")]


@pytest.fixture()
def env(tmp_path):
    if shutil.which("bash") is None or shutil.which("awk") is None:
        pytest.skip("bash / awk が無い")
    ngx = tmp_path / "nginx"
    (ngx / "conf.d").mkdir(parents=True)
    (ngx / "sites-available").mkdir()
    zones = (TPL / "nginx-zones.conf").read_text(encoding="utf-8")
    a, b = zones.index("# API の制限の枠（外の実 IP ごと・127.0.0.1 の別枠）"), zones.index("# API の制限の枠ここまで\n") + len("# API の制限の枠ここまで\n")
    (ngx / "conf.d" / "keiba-ui-zones.conf").write_text(zones[:a] + zones[b:], encoding="utf-8")     # 今の VPS: 制限の節が無い
    for name, tpl in (("keiba-ui", "nginx-keiba-ui-cloudrun.conf"), ("keiba-ui-stg", "nginx-keiba-ui-stg.conf")):
        (ngx / "sites-available" / name).write_text(_old_form((TPL / tpl).read_text(encoding="utf-8")), encoding="utf-8")
    tpl_dir = tmp_path / "tpl"
    tpl_dir.mkdir()
    for t in ("nginx-keiba-ui.conf", "nginx-keiba-ui-cloudrun.conf"):
        (tpl_dir / t).write_text(_old_form((TPL / t).read_text(encoding="utf-8")), encoding="utf-8")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for cmd in ("nginx", "systemctl"):
        (bin_ / cmd).write_text("#!/bin/sh\nexit 0\n")
        (bin_ / cmd).chmod(0o755)
    e = {**os.environ, "PATH": f"{bin_}{os.pathsep}{os.environ['PATH']}", "NGX_ETC": str(ngx), "BK_ROOT": str(tmp_path / "bk"),
         "API_CACHE_DIR": str(tmp_path / "cache" / "keiba_api"), "KEIBA_UI_TPL_DIR": str(tpl_dir),
         "REFRESH_TOKEN_FILE": str(tmp_path / "secret" / "refresh.token"), "API_CACHE_MAX_MB": "1024",
         "SITES": f"{ngx}/sites-available/keiba-ui {ngx}/sites-available/keiba-ui-stg"}
    return ngx, tpl_dir, e


def _run(e, *args):
    r = subprocess.run(["bash", str(VPS / "vps_09_api_cache.sh"), *args], env=e, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return r


def _snapshot(ngx: Path, tpl_dir: Path) -> dict:
    files = [p for p in list(ngx.rglob("*")) + list(tpl_dir.glob("*")) if p.is_file()]
    return {str(p): p.read_bytes() for p in files}


def test_apply_rewrites_only_api_location_and_is_idempotent(env):
    ngx, tpl_dir, e = env
    _run(e)
    for name in ("keiba-ui", "keiba-ui-stg"):
        s = (ngx / "sites-available" / name).read_text(encoding="utf-8")
        api = _api_block(s)
        assert EXT in api and LO in api and OLD not in api, name
        assert OLD in _img_block(s) and EXT not in _img_block(s), name       # 画像の location は旧の枠のまま
        assert "keiba-api-cache" in api
    zones = (ngx / "conf.d" / "keiba-ui-zones.conf").read_text(encoding="utf-8")
    assert zones.count("zone=keiba_api_ext:10m") == 1 and "keiba_cache_readonly" in zones
    assert (ngx / "conf.d" / "keiba-api-gate.conf").is_file() and (ngx / "snippets" / "keiba-api-cache.conf").is_file()
    for t in tpl_dir.glob("*.conf"):
        assert EXT in _api_block(t.read_text(encoding="utf-8")), t.name
    tok = Path(e["REFRESH_TOKEN_FILE"]).read_text()
    before = _snapshot(ngx, tpl_dir)
    _run(e)
    after = _snapshot(ngx, tpl_dir)
    assert before == after                                                     # 2 回目は何も変えない
    assert Path(e["REFRESH_TOKEN_FILE"]).read_text() == tok                    # 合言葉も変えない


def test_apply_fails_and_restores_when_api_limit_line_differs(env):
    ngx, tpl_dir, e = env
    site = ngx / "sites-available" / "keiba-ui"
    s = site.read_text(encoding="utf-8")
    api = _api_block(s)
    site.write_text(s.replace(api, api.replace(OLD, "limit_req zone=keiba_api burst=99 nodelay;")), encoding="utf-8")
    orig = site.read_text(encoding="utf-8")
    r = subprocess.run(["bash", str(VPS / "vps_09_api_cache.sh")], env=e, capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "見つかりません" in r.stderr
    assert site.read_text(encoding="utf-8") == orig                            # 画像の location を書き換えたまま成功しない


def test_remove_keeps_limits_and_zone_definitions(env):
    """--remove はキャッシュだけを外す。制限の枠（サイト・テンプレート・zones の節）は残す（テンプレートから作り直す道具が落ちない）。"""
    ngx, tpl_dir, e = env
    _run(e)
    _run(e, "--remove")
    for name in ("keiba-ui", "keiba-ui-stg"):
        s = (ngx / "sites-available" / name).read_text(encoding="utf-8")
        assert EXT in _api_block(s) and LO in _api_block(s) and "keiba-api-cache" not in s, name
    for t in tpl_dir.glob("*.conf"):
        assert EXT in _api_block(t.read_text(encoding="utf-8")), t.name
    zones = (ngx / "conf.d" / "keiba-ui-zones.conf").read_text(encoding="utf-8")
    assert zones.count("zone=keiba_api_ext:10m") == 1 and "keiba_apicache" not in zones
    assert not (ngx / "conf.d" / "keiba-api-gate.conf").exists()


def test_remove_restores_deleted_includes_when_nginx_test_fails(env, tmp_path):
    """--remove で nginx -t が落ちたら、消した include 先（合言葉・stg の読むだけ）も戻す。"""
    ngx, tpl_dir, e = env
    _run(e)
    before = {f: (ngx / f).read_bytes() for f in ("keiba-refresh.conf", "keiba-cache-readonly.conf")}
    (Path(e["PATH"].split(os.pathsep)[0]) / "nginx").write_text("#!/bin/sh\nexit 1\n")
    r = subprocess.run(["bash", str(VPS / "vps_09_api_cache.sh"), "--remove"], env=e, capture_output=True, text=True, timeout=60)
    assert r.returncode != 0
    assert {f: (ngx / f).read_bytes() for f in before} == before


def test_limit_zones_only_adds_zone_definitions(env):
    ngx, tpl_dir, e = env
    _run(e, "--limit-zones")
    zones = (ngx / "conf.d" / "keiba-ui-zones.conf").read_text(encoding="utf-8")
    assert zones.count("zone=keiba_api_ext:10m") == 1 and zones.count("zone=keiba_api_lo:1m") == 1
    s = (ngx / "sites-available" / "keiba-ui").read_text(encoding="utf-8")
    assert OLD in _api_block(s) and EXT not in s                               # サイトはまだ旧の枠のまま
    assert not (ngx / "snippets" / "keiba-api-cache.conf").exists()          # スニペット（段 3）はまだ入れない
    _run(e, "--limit-zones")
    assert (ngx / "conf.d" / "keiba-ui-zones.conf").read_text(encoding="utf-8") == zones
