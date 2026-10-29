"""Cloud Run のデプロイ/ロールバック/セットアップ、VPS の nginx 切替テンプレート。gcloud は偽物に差し替え、実際には何も実行しない。"""

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GCP = ROOT / "scripts" / "gcp"
VPS = ROOT / "scripts" / "ops" / "vps"


@pytest.fixture
def fake_gcloud(tmp_path):
    log = tmp_path / "gcloud.log"
    g = tmp_path / "bin" / "gcloud"
    g.parent.mkdir()
    g.write_text(f"#!/usr/bin/env bash\necho \"$@\" >> {log}\n[ -n \"$FAKE_NO_SERVICE\" ] && [ \"$3\" = describe ] && exit 1\nexit 0\n")
    g.chmod(g.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": f"{g.parent}:{os.environ['PATH']}", "PROJECT_ID": "proj-x", "GCS_BUCKET": "bkt-x", "TAG": "t1"}
    env.pop("KEIBA_ENV", None)
    return env, log


def run(script, *args, env):
    return subprocess.run(["bash", str(script), *args], env=env, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("script", sorted(p.name for p in GCP.glob("*cloud_run_api.sh")) + sorted(p.name for p in VPS.glob("*.sh")))
def test_shell_syntax(script):
    p = (GCP / script) if (GCP / script).exists() else (VPS / script)
    assert subprocess.run(["bash", "-n", str(p)], capture_output=True).returncode == 0


def test_deploy_default_and_dry_run_execute_nothing(fake_gcloud):
    env, log = fake_gcloud
    for args in ([], ["--dry-run"]):
        r = run(GCP / "deploy_cloud_run_api.sh", "all", *args, env=env)
        assert r.returncode == 0, r.stderr
        assert not log.exists(), "gcloud が呼ばれた"
    out = r.stdout
    assert "gcloud run deploy keiba-api" in out
    # 要件: 最小 0 / 最大 3 / 同時実行 20-40 / タイムアウト 60 / KEIBA_ENV=prod / KEIBA_RUN_LOCATION=gcp / 共有ヘッダ必須
    assert "--min-instances 0" in out and "--max-instances 3" in out and "--timeout 60" in out
    assert 20 <= int(re.search(r"--concurrency (\d+)", out).group(1)) <= 40
    assert "KEIBA_ENV=prod" in out and "KEIBA_RUN_LOCATION=gcp" in out and "KEIBA_PROXY_KEY_REQUIRED=1" in out
    assert "--set-secrets" in out and "KEIBA_PROXY_KEY=KEIBA_PROXY_KEY:latest" in out and "--no-traffic" not in out.split("gcloud builds")[0]


def test_deploy_apply_calls_gcloud(fake_gcloud):
    env, log = fake_gcloud
    r = run(GCP / "deploy_cloud_run_api.sh", "deploy", "--apply", env=env)
    assert r.returncode == 0, r.stderr
    calls = log.read_text()
    assert "run deploy keiba-api" in calls and "--no-traffic" in calls       # 既存サービスへは新リビジョンをトラフィック 0% で
    log.unlink()
    r1 = run(GCP / "deploy_cloud_run_api.sh", "deploy", "--apply", env={**env, "FAKE_NO_SERVICE": "1"})
    assert r1.returncode == 0 and "--no-traffic" not in log.read_text()      # 初回は --no-traffic が使えないので付けない
    r2 = run(GCP / "deploy_cloud_run_api.sh", "deploy", "--apply", env={**env, "PROJECT_ID": "YOUR_GCP_PROJECT_ID"})
    assert r2.returncode == 2                                                # プレースホルダのままでは実行しない


def test_deploy_cpu_alloc_switches_billing_mode_and_mirror_mode(fake_gcloud):
    """always=CPU 常時割り当て(インスタンス課金) / request=リクエスト課金。request のときだけ取り込みをリクエスト駆動にする。不正値は拒否。"""
    env, log = fake_gcloud
    for mode, flag, mirror in (("always", "--no-cpu-throttling", False), ("request", "--cpu-throttling", True)):
        log.unlink(missing_ok=True)
        r = run(GCP / "deploy_cloud_run_api.sh", "deploy", "--apply", env={**env, "CPU_ALLOC": mode})
        assert r.returncode == 0, r.stderr
        calls = log.read_text()
        assert (flag in calls) and (("--no-cpu-throttling" in calls) == (mode == "always")), mode
        assert ("KEIBA_MIRROR_MODE=request" in calls) == mirror, mode
    log.unlink(missing_ok=True)
    assert run(GCP / "deploy_cloud_run_api.sh", "deploy", "--apply", env=env).returncode == 0 and "--no-cpu-throttling" not in log.read_text() and "--cpu-throttling" in log.read_text()   # 既定はリクエスト課金
    assert run(GCP / "deploy_cloud_run_api.sh", "deploy", "--apply", env={**env, "CPU_ALLOC": "bogus"}).returncode == 2


def test_ci_uses_request_billing_for_stg_and_keeps_prod_switchable():
    body = (ROOT / "scripts/ci/deploy_gcp.sh").read_text(encoding="utf-8")
    assert 'CPU_ALLOC="$([ "$TARGET" = stg ] && echo request || echo "${PROD_CPU_ALLOC:-request}")"' in body


def test_deploy_apply_refused_in_dev(fake_gcloud):
    env, log = fake_gcloud
    r = run(GCP / "deploy_cloud_run_api.sh", "deploy", "--apply", env={**env, "KEIBA_ENV": "dev"})
    assert r.returncode == 2 and not log.exists()


def test_no_secret_values_in_scripts():
    for p in [*GCP.glob("*cloud_run_api.sh"), VPS / "switch_api_upstream.sh", VPS / "render_nginx_proxykey.sh", VPS / "heartbeat_client.py"]:
        text = p.read_text(encoding="utf-8")
        assert not re.search(r"eyJ[A-Za-z0-9_-]{20,}|hooks\.slack\.com/services/T\w+/B\w+/\w+|-----BEGIN", text), p


def test_rollback_and_setup_default_to_print_only(fake_gcloud):
    env, log = fake_gcloud
    r = run(GCP / "rollback_cloud_run_api.sh", "--to", "keiba-api-00007-abc", env=env)
    assert r.returncode == 0 and "--to-revisions keiba-api-00007-abc=100" in r.stdout and not log.exists()
    r = run(GCP / "setup_cloud_run_api.sh", env=env)
    assert r.returncode == 0 and not log.exists() and "roles/storage.objectAdmin" in r.stdout and "roles/secretmanager.secretAccessor" in r.stdout
    assert "versions add" in r.stdout and "openssl rand" in r.stdout          # 値は標準入力から入れる案内（値そのものは無い）


def test_nginx_templates_render_without_placeholders():
    def render(tpl, subs):
        s = (VPS / "templates" / tpl).read_text(encoding="utf-8")
        for k, v in subs.items():
            s = s.replace(k, v)
        return s

    cr = render("nginx-keiba-ui-cloudrun.conf", {"__CLOUDRUN_HOST_RE__": r"keiba-api-x-an\.a\.run\.app", "__CLOUDRUN_HOST__": "keiba-api-x-an.a.run.app",
                                                  "__PROXY_PORT__": "8080", "__APP_PORT__": "3000"})
    host = render("nginx-keiba-ui.conf", {"__UPSTREAM_HOST_RE__": r"api-internal\.x", "__UPSTREAM_HOST__": "api-internal.x",
                                          "__PROXY_PORT__": "8080", "__APP_PORT__": "3000"})
    for conf in (cr, host):
        assert not re.search(r"__[A-Z_]+__", re.sub(r"#.*", "", conf))
        assert conf.count("{") == conf.count("}")
        assert "location ~ ^/api/" in conf and "location ~ ^/data/image/" in conf and "proxy_pass http://127.0.0.1:3000;" in conf
    assert "include /etc/nginx/keiba-proxykey.conf" in cr and "CF-Access" not in re.sub(r"#.*", "", cr)
    assert "include /etc/nginx/keiba-access.conf" in host and "proxykey" not in re.sub(r"#.*", "", host)
    assert "proxy_set_header X-Keiba-Client-IP $remote_addr;" in cr    # クライアントが送った値ではなく nginx が見た IP を渡す


def test_switch_script_has_safety_nets():
    s = (VPS / "switch_api_upstream.sh").read_text(encoding="utf-8")
    for needle in ("nginx -t", "revert", "/api/health", "cp -a \"$SITE\" \"$BAK\"", "keiba-proxykey.conf"):
        assert needle in s
