"""VPS の systemd ユニット・スクリプトの静的検査（実機なし）。メモリ予算が architecture_target.md 3.2 の表とずれないことを守る。"""
import configparser
import os
import re
import subprocess
from pathlib import Path

T = Path("scripts/ops/vps/templates")
ROOT_DIR = Path(__file__).resolve().parents[2]


def load(name):
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    cp.optionxform = str
    cp.read(T / name, encoding="utf-8")
    return cp


def mb(s):
    m = re.fullmatch(r"(\d+)M", s)
    assert m, s
    return int(m.group(1))


def test_memory_budget_matches_design():
    caps = {
        "agent": load("keiba-fetch-agent.service")["Service"],
        "slice": load("keiba-fetch.slice")["Slice"],
        "nginx": load("dropin-nginx-memory.conf")["Service"],
        "cf": load("dropin-cloudflared-prod-memory.conf")["Service"],
        "ui": load("dropin-keiba-ui-memory.conf")["Service"],
    }
    assert (mb(caps["agent"]["MemoryHigh"]), mb(caps["agent"]["MemoryMax"])) == (220, 250)
    assert (mb(caps["slice"]["MemoryHigh"]), mb(caps["slice"]["MemoryMax"])) == (220, 250)
    assert (mb(caps["nginx"]["MemoryMax"]), mb(caps["cf"]["MemoryMax"]), mb(caps["ui"]["MemoryMax"])) == (48, 128, 200)
    for k, v in caps.items():
        assert mb(v["MemoryHigh"]) < mb(v["MemoryMax"]), k
    assert sum(mb(caps[k]["MemoryMax"]) for k in ("agent", "nginx", "cf", "ui")) <= 640   # OS 約 400MB と合わせて約 1.0GB


def test_agent_unit_is_hardened_and_off_by_default():
    svc = load("keiba-fetch-agent.service")
    s = svc["Service"]
    assert s["OOMScoreAdjust"] == "500" and s["Slice"] == "keiba-fetch.slice" and s["Restart"] == "on-failure"
    assert "KEIBA_ENV=prod" in s["Environment"] and "KEIBA_RUN_LOCATION=vps" in s["Environment"]
    assert s["NoNewPrivileges"] == "true" and s["ProtectSystem"] == "strict"
    text = (T / "keiba-fetch-agent.service").read_text(encoding="utf-8")
    assert "PASSWORD" not in text and "SECRET" not in text      # 秘密はユニットに書かない


def test_pull_timers_ui_is_periodic_and_fetch_sits_inside_maintenance_window():
    ui = load("keiba-release-pull-ui.timer")["Timer"]          # 画面は窓を待たず数分おき(昇格で API が先に切り替わるため)
    assert "OnCalendar" not in ui and ui["OnUnitActiveSec"] == "3min"
    assert ui["Unit"] == "keiba-release-pull@ui.service"
    t = load("keiba-release-pull-fetch.timer")["Timer"]         # 取得の実行役は窓の中(03:20)だけ
    assert t["OnCalendar"].startswith("*-*-* 03:20")
    assert t["Unit"] == "keiba-release-pull@fetch.service"
    assert "--kind %i" in (T / "keiba-release-pull@.service").read_text(encoding="utf-8")


def test_shell_scripts_parse():
    for f in ("vps_03_units.sh", "measure_memory.sh", "publish_release.sh", "vps_01_base_setup.sh"):
        assert subprocess.run(["bash", "-n", f"scripts/ops/vps/{f}"]).returncode == 0, f


def test_vps_03_dry_run_changes_nothing():
    r = subprocess.run(["bash", "scripts/ops/vps/vps_03_units.sh", "--dry-run"], capture_output=True, text=True)
    assert r.returncode == 0 and "完了" in r.stdout
    assert "eyJ" not in r.stdout


# ── stg 環境の画面（VPS の 2 つ目の画面）──

def test_stg_ui_unit_is_separate_and_smaller_than_prod():
    s = load("keiba-ui-stg.service")["Service"]
    assert "/srv/keiba-ui-stg/current" in s["WorkingDirectory"] and s["EnvironmentFile"] == "/etc/keiba-ui/ui-stg.env"
    assert (mb(s["MemoryHigh"]), mb(s["MemoryMax"])) == (100, 160) and s["SyslogIdentifier"] == "keiba-ui-stg"
    assert s["NoNewPrivileges"] == "true" and s["ProtectSystem"] == "full"
    prod = load("dropin-keiba-ui-memory.conf")["Service"]
    assert mb(s["MemoryMax"]) < mb(prod["MemoryMax"])                          # 確認用の環境は本番の画面より小さく
    # 常駐の合計（本番 640 + stg 160）でも、OS を含めて 2GB の VPS に余裕が残る
    assert 640 + mb(s["MemoryMax"]) <= 800


def test_stg_pull_timer_pulls_often_and_has_no_window():
    t = load("keiba-release-pull-ui-stg.timer")["Timer"]
    assert t["OnUnitActiveSec"] == "5min" and t["Unit"] == "keiba-release-pull@ui-stg.service"
    assert "OnCalendar" not in t                                               # 本番の画面は窓（03:10）、stg は公開されたらすぐ


def test_stg_nginx_goes_to_the_stg_tagged_revision_with_its_own_ports():
    text = (T / "nginx-keiba-ui-stg.conf").read_text(encoding="utf-8")
    assert "__PROXY_PORT__" in text and "__APP_PORT__" in text and "__CLOUDRUN_HOST__" in text
    assert "include /etc/nginx/keiba-proxykey.conf" in text and "eyJ" not in text
    assert "server_name _" in text and "listen 127.0.0.1:__PROXY_PORT__" in text


def test_stg_instance_script_parses_and_dry_run_changes_nothing():
    assert subprocess.run(["bash", "-n", "scripts/ops/vps/vps_05_stg_instance.sh"]).returncode == 0
    r = subprocess.run(["bash", "scripts/ops/vps/vps_05_stg_instance.sh", "--dry-run"], capture_output=True, text=True)
    assert r.returncode == 0 and "完了" in r.stdout and "eyJ" not in r.stdout
    assert "/srv/keiba-ui-stg" in r.stdout and "sites-available/keiba-ui-stg" in r.stdout
    assert "sites-available/keiba-ui " not in r.stdout and "keiba-ui.service" not in r.stdout       # 本番の画面・nginx には触れない


def _install_vars(instance: str | None) -> dict:
    """install_release.sh の冒頭（変数を決める部分）だけを実行して、決まった値を返す。"""
    lines = (ROOT_DIR / "scripts/ops/vps/install_release.sh").read_text(encoding="utf-8").splitlines()
    head = "\n".join(lines[: next(i for i, l in enumerate(lines) if l.startswith("APP_ROOT=")) + 1])
    script = head + '\necho "$SVC|$APP_ROOT|$APP_PORT|$PROXY_PORT|$TP|$SFX"'
    env = {k: v for k, v in os.environ.items() if k not in ("INSTANCE", "APP_ROOT", "APP_PORT", "PROXY_PORT")}
    if instance is not None:
        env["INSTANCE"] = instance
    out = subprocess.run(["bash", "-c", script.replace('. "/etc/keiba-ui/vars${SFX}.env"', ': # vars は読まない'), "x", "/dev/null"], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return dict(zip(("svc", "root", "app", "proxy", "tp", "sfx"), out.stdout.strip().split("|")))


def test_install_release_defaults_are_exactly_the_prod_values_and_stg_is_separate():
    prod = _install_vars(None)
    assert prod == {"svc": "keiba-ui", "root": "/srv/keiba-ui", "app": "3000", "proxy": "8080", "tp": "3101", "sfx": ""}     # 本番は今までと同じ
    stg = _install_vars("stg")
    assert stg == {"svc": "keiba-ui-stg", "root": "/srv/keiba-ui-stg", "app": "3002", "proxy": "8081", "tp": "3102", "sfx": "-stg"}
    bad = subprocess.run(["bash", "scripts/ops/vps/install_release.sh", "/x"], env={**os.environ, "INSTANCE": "nope"}, capture_output=True, text=True)
    assert bad.returncode == 2 and "INSTANCE" in bad.stderr


def test_build_release_has_env_label_with_prod_default():
    text = (ROOT_DIR / "scripts/ops/vps/build_release.sh").read_text(encoding="utf-8")
    assert "ENVLABEL=prod" in text and "--env-label)" in text and 'NEXT_PUBLIC_ENV_LABEL="$([ "$ENVLABEL" = prod ]' in text
    assert '"env":"prod"' not in text and "--env prod" not in text           # 固定の prod は残さない


def test_pull_sa_setup_is_least_privilege_and_creates_no_key_by_default():
    env = {**os.environ, "PROJECT_ID": "p1", "GCS_BUCKET": "b1"}
    r = subprocess.run(["bash", str(ROOT_DIR / "scripts/gcp/setup_pull_sa.sh")], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = r.stdout.replace("\\", "")
    assert "[dry]" in out and "keys create" not in out                                  # 既定は表示のみ。鍵は --make-key のときだけ
    assert "roles/storage.objectViewer" in out and "objects/ops/releases/" in out
    assert "endsWith" in out and "/applied/vps.json" in out                            # 書き込めるのは取り込み結果の報告だけ
    k = subprocess.run(["bash", str(ROOT_DIR / "scripts/gcp/setup_pull_sa.sh"), "--make-key", "/tmp/x.json"], env=env, capture_output=True, text=True)
    assert "keys create /tmp/x.json" in k.stdout.replace("\\", "") and not Path("/tmp/x.json").exists()      # --apply なしでは作らない


# ── 監視ポータルの VPS 側の入口 ──

def test_monitor_proxy_script_parses_and_dry_run_renders_without_leftovers(tmp_path):
    assert subprocess.run(["bash", "-n", "scripts/ops/vps/vps_06_monitor_proxy.sh"]).returncode == 0
    vars_file = tmp_path / "vars.env"
    vars_file.write_text("MONITOR_CLOUDRUN_HOST=keiba-monitor-abc.asia-northeast1.run.app\n", encoding="utf-8")
    r = subprocess.run(["bash", "scripts/ops/vps/vps_06_monitor_proxy.sh", "--dry-run"], capture_output=True, text=True,
                       env={**os.environ, "VARS_FILE": str(vars_file)})
    assert r.returncode == 0 and "127.0.0.1:8082" in r.stdout and "keiba-monitor-abc" in r.stdout
    r2 = subprocess.run(["bash", "scripts/ops/vps/vps_06_monitor_proxy.sh", "--dry-run"], capture_output=True, text=True,
                        env={**os.environ, "VARS_FILE": str(tmp_path / "none.env")})
    assert r2.returncode == 2 and "MONITOR_CLOUDRUN_HOST" in r2.stderr          # 設定が無ければ何も置かない


def test_monitor_proxy_template_sends_the_shared_key_and_listens_locally():
    t = (T / "nginx-keiba-monitor.conf").read_text(encoding="utf-8")
    assert "listen 127.0.0.1:__MONITOR_PORT__" in t and "include /etc/nginx/keiba-proxykey.conf" in t
    assert "PASSWORD" not in t and "SECRET" not in t


def test_traffic_units_are_sandboxed_and_hold_no_secret():
    svc = load("keiba-traffic.service")["Service"]
    assert svc["Type"] == "oneshot" and mb(svc["MemoryMax"]) <= 128
    assert svc["ProtectSystem"] == "strict" and svc["ReadWritePaths"] == "/var/lib/keiba-ui/traffic" and svc["NoNewPrivileges"] == "true"
    assert svc["ExecStart"].endswith("/opt/keiba-ui/bin/traffic_client.py")
    text = (T / "keiba-traffic.service").read_text(encoding="utf-8") + (T / "keiba-traffic.timer").read_text(encoding="utf-8")
    assert "SECRET" not in text and "http" not in text.replace("network-online", "")   # 秘密も送り先も /etc/keiba-ui/heartbeat.env にだけ置く
    assert load("keiba-traffic.timer")["Timer"]["OnUnitActiveSec"] in ("15min", "900")


def test_traffic_installer_is_valid_bash_and_installs_both_scripts():
    sh = ROOT_DIR / "scripts/ops/vps/vps_07_traffic_timer.sh"
    assert subprocess.run(["bash", "-n", str(sh)], capture_output=True).returncode == 0
    body = sh.read_text(encoding="utf-8")
    for name in ("traffic_client.py", "vps_traffic_probe.py", "heartbeat_client.py", "keiba-traffic.timer"):
        assert name in body
