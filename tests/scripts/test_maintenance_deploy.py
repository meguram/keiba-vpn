"""メンテナンス時間帯の自動反映（ステージ → ランナー）。実サーバは再起動しない: 一時ディレクトリの疑似 repo とモックで、
manifest 検証・パス制限・原子的配置・自動ロールバック・前提条件・窓の判定を確かめる。"""

import json
import os
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

from src.scripts.ops import deploy_lib as D
from src.scripts.ops import maintenance_deploy as MD
from src.scripts.ops import stage_deploy as SD
from src.utils import maintenance as M

JST = ZoneInfo("Asia/Tokyo")
OLD_PY = "VALUE = 1\n"
NEW_PY = "VALUE = 2\n"


def at(h, m=0, day=22):
    return datetime(2026, 10, day, h, m, tzinfo=JST)


class FakeEnv:
    """コマンド実行・HTTP・通知をまとめて記録するモック。"""

    def __init__(self):
        self.cmds: list[list[str]] = []
        self.notes: list[tuple[str, list[str], str]] = []
        self.rc = {}              # {"restart_server": 1} のように、コマンド名に含まれる語 → 終了コード
        self.health = [200]       # 順に返す（尽きたら最後の値）
        self.queue = {"stats": {"running": 0, "queued": 0}}
        self.procs: list[str] = []
        self.load = 3.0
        self.disk = 100.0
        self._h = 0

    def run(self, cmd, *, timeout=300, env=None, cwd=None):
        self.cmds.append(list(cmd))
        for k, v in self.rc.items():
            if any(k in c for c in cmd):
                return v, f"fake rc={v}"
        return 0, "ok"

    def http_get(self, url, *, timeout=5.0):
        if "/api/scrape-jobs" in url:
            return 200, json.dumps(self.queue)
        if "/api/health" in url or url.endswith("/"):
            st = self.health[min(self._h, len(self.health) - 1)]
            self._h += 1
            return st, json.dumps({"status": "ok", "prob_calibration": {"applied": True}})
        return 200, json.dumps({"prob_calibration": {"applied": True}})

    def notify(self, title, lines, severity):
        self.notes.append((title, lines, severity))


def make_repo():
    root = Path(tempfile.mkdtemp(prefix="repo_"))
    (root / "src" / "api").mkdir(parents=True)
    (root / "src" / "api" / "app.py").write_text(OLD_PY)
    (root / "config").mkdir()
    (root / "config" / "x.json").write_text('{"v": 1}')
    return root


def stage(root, id="20261022-test", files=None, steps=("api_restart",), **kw):
    files = files or {"src/api/app.py": NEW_PY}
    out = root / "data" / "local" / "deploy" / "pending"
    out.mkdir(parents=True, exist_ok=True)
    src = Path(tempfile.mkdtemp(prefix="srcrepo_"))
    for rel, content in files.items():
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(content)
    return SD.create_stage(out, id=id, description="テスト", rels=list(files), steps=list(steps), repo=src, now=kw.pop("now", at(1)), **kw)


def deployer(root, env: FakeEnv, now=None, **kw):
    return MD.Deployer(root, clock=lambda: now or at(3, 5), run=env.run, http_get=env.http_get, sleep=lambda s: None,
                       load_fn=lambda: env.load, disk_gb_fn=lambda: env.disk, processes=lambda: env.procs,
                       notify=env.notify, cfg=M.DEFAULT_CONFIG, env={}, **kw)


class TestManifestValidation(unittest.TestCase):
    def test_denied_and_invalid_paths(self):
        for bad in (".env", "frontend/.env.local", "/etc/passwd", "../x.py", "src/../../x.py", "data/local/x.json", "crontab", "src\\x.py",
                    "scripts/ops/maintenance_deploy.sh", "src/scripts/ops/maintenance_deploy.py", "src/scripts/ops/deploy_lib.py",
                    "frontend/.next-stg/x.js", "frontend/node_modules/a/b.js", "docs/x.md", "README.md", "src//x.py", "src/./x.py"):
            with self.assertRaises(D.ManifestError, msg=bad):
                D.check_dest_allowed(bad)
        for ok in ("src/api/app.py", "config/x.json", "frontend/components/A.tsx", "scripts/cron/x.sh"):
            D.check_dest_allowed(ok)

    def test_validate_shape(self):
        m = D.build_manifest(id="abc-123", description="d", files=[{"dest": "src/a.py", "sha256": "0" * 64}], steps=["api_restart"], now=at(1))
        D.validate_manifest(m)
        for mut in ({"schema": 2}, {"id": "../x"}, {"id": "a"}, {"files": []}, {"steps": ["rm_rf"]}, {"extend_window_minutes": 999},
                    {"rollback": {"method": "custom_script"}}):
            with self.assertRaises(D.ManifestError, msg=mut):
                D.validate_manifest({**m, **mut})
        with self.assertRaises(D.ManifestError):
            D.validate_manifest({**m, "files": [{"dest": "src/a.py", "sha256": "0" * 64}] * 2})
        with self.assertRaises(D.ManifestError):
            D.validate_manifest({**m, "files": [{"dest": "src/a.py", "sha256": "zz"}]})

    def test_expiry(self):
        m = D.build_manifest(id="abc-123", description="d", files=[{"dest": "src/a.py", "sha256": "0" * 64}], steps=[], expires_hours=1, now=at(1))
        D.validate_manifest(m, now=at(1, 30), check_expiry=True)
        with self.assertRaises(D.ManifestError):
            D.validate_manifest(m, now=at(2, 30), check_expiry=True)

    def test_symlink_destination_and_escape(self):
        root = make_repo()
        outside = Path(tempfile.mkdtemp())
        (root / "src" / "link.py").symlink_to(outside / "evil.py")
        with self.assertRaises(D.ManifestError):
            D.resolve_inside(root, "src/link.py")
        (root / "src" / "escdir").symlink_to(outside)
        with self.assertRaises(D.ManifestError):
            D.resolve_inside(root, "src/escdir/new.py")


class TestStageDeploy(unittest.TestCase):
    def test_create_stage_hashes_and_ref(self):
        root = make_repo()
        s = stage(root)
        m = json.loads((s / "manifest.json").read_text())
        self.assertEqual(m["files"][0]["dest"], "src/api/app.py")
        self.assertEqual(m["files"][0]["sha256"], D.sha256_file(s / "files" / "src/api/app.py"))
        D.verify_staged_files(s, m)

    def test_create_stage_rejects_forbidden_and_symlink(self):
        root = make_repo()
        src = Path(tempfile.mkdtemp())
        (src / ".env").write_text("SECRET=1")
        with self.assertRaises(D.ManifestError):
            SD.create_stage(root / "o", id="abc-001", description="x", rels=[".env"], steps=[], repo=src)
        (src / "src").mkdir()
        (src / "src" / "real.py").write_text("x=1\n")
        (src / "src" / "link.py").symlink_to(src / "src" / "real.py")
        with self.assertRaises(SystemExit):
            SD.create_stage(root / "o", id="abc-002", description="x", rels=["src/link.py"], steps=[], repo=src)

    def test_parse_verify(self):
        v = SD.parse_verify("http://x/api|status=503|key=a.b")
        self.assertEqual(v, {"url": "http://x/api", "expect_status": 503, "json_key": "a.b"})

    def test_push_commands_are_atomic_and_quoted(self):
        root = make_repo()
        s = stage(root)
        calls = []
        SD.push(s, "host", "/srv/repo", run=lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0))
        joined = " ".join(" ".join(c) for c in calls)
        self.assertIn("scp", joined)
        self.assertIn(".incoming-20261022-test", joined)   # 途中の状態を pending に見せない
        self.assertIn("mv", joined)

    def test_cancel_rejects_bad_id(self):
        with self.assertRaises(SystemExit):
            SD.remote_cancel("h", "/r", "../../etc", run=lambda *a, **k: None)


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.root = make_repo()
        self.env = FakeEnv()

    def read(self, rel="src/api/app.py"):
        return (self.root / rel).read_text()

    def test_happy_path(self):
        s = stage(self.root, files={"src/api/app.py": NEW_PY, "config/y.json": '{"n": 1}'}, steps=("api_restart", "frontend_build_switch"))
        d = deployer(self.root, self.env)
        res = d.run_all()
        self.assertEqual(res, [("20261022-test", "applied")])
        self.assertEqual(self.read(), NEW_PY)
        self.assertEqual(self.read("config/y.json"), '{"n": 1}')
        self.assertTrue((self.root / "data/local/deploy/applied/20261022-test/result.json").exists())
        self.assertFalse(s.exists())
        # 退避（元のファイル）
        self.assertEqual((self.root / "data/local/meta/cron_backups/deploy_20261022-test/files/src/api/app.py").read_text(), OLD_PY)
        # 手順は API 再起動 → フロントのビルド → 切替の順
        names = [" ".join(c[:3]) for c in self.env.cmds if "restart_server" in " ".join(c) or "frontend_build" in " ".join(c) or "server_watchdog" in " ".join(c)]
        self.assertEqual(len(names), 3)
        self.assertIn("restart_server", names[0])
        self.assertIn("frontend_build", names[1])
        self.assertIn("server_watchdog", names[2])
        events = [json.loads(x)["event"] for x in (self.root / "data/local/meta/deploy_audit.jsonl").read_text().splitlines()]
        self.assertIn("start", events)
        self.assertIn("applied", events)
        self.assertEqual([n[2] for n in self.env.notes], ["normal", "normal"])   # 開始・成功

    def test_outside_window_does_nothing_and_force_applies(self):
        stage(self.root)
        d = deployer(self.root, self.env, now=at(12, 0))
        self.assertEqual(d.run_all(), [])
        self.assertEqual(self.read(), OLD_PY)
        self.assertEqual(self.env.cmds, [])
        self.assertEqual(d.run_all(force_window=True), [("20261022-test", "applied")])

    def test_dry_run_changes_nothing(self):
        stage(self.root)
        d = deployer(self.root, self.env, now=at(12, 0))
        self.assertEqual(d.run_all(dry_run=True), [("20261022-test", "dry_run")])
        self.assertEqual(self.read(), OLD_PY)
        self.assertEqual(self.env.cmds, [])
        self.assertTrue((self.root / "data/local/deploy/pending/20261022-test").exists())

    def test_preconditions_defer(self):
        stage(self.root)
        for setup in (lambda e: e.queue["stats"].update(running=1), lambda e: setattr(e, "load", 20.0),
                      lambda e: setattr(e, "disk", 0.5), lambda e: e.procs.append("python raceday-runner x")):
            self.env = FakeEnv()
            setup(self.env)
            d = deployer(self.root, self.env, max_wait_sec=60)
            self.assertEqual(d.run_all(), [("20261022-test", "deferred")])
            self.assertEqual(self.read(), OLD_PY)
            self.assertEqual(self.env.cmds, [])
            self.assertTrue((self.root / "data/local/deploy/pending/20261022-test").exists())   # 次の窓へ持ち越し

    def test_wait_then_proceed(self):
        stage(self.root)
        busy = {"n": 0}
        env = self.env
        orig = env.http_get

        def http_get(url, *, timeout=5.0):
            if "/api/scrape-jobs" in url:
                busy["n"] += 1
                return 200, json.dumps({"stats": {"running": 1 if busy["n"] < 3 else 0, "queued": 0}})
            return orig(url, timeout=timeout)

        env.http_get = http_get
        self.assertEqual(deployer(self.root, env, max_wait_sec=600).run_all(), [("20261022-test", "applied")])

    def test_sha_mismatch_and_extra_file_and_symlink_are_invalid(self):
        s = stage(self.root)
        (s / "files/src/api/app.py").write_text("tampered\n")
        self.assertEqual(deployer(self.root, self.env).run_all(), [("20261022-test", "invalid")])
        self.assertEqual(self.read(), OLD_PY)
        self.assertTrue((self.root / "data/local/deploy/failed/20261022-test").exists())
        # manifest に無いファイル
        s = stage(self.root, id="20261022-extra")
        (s / "files/src/api/other.py").write_text("x=1\n")
        self.assertEqual(deployer(self.root, self.env).run_all(), [("20261022-extra", "invalid")])
        self.assertFalse((self.root / "src/api/other.py").exists())
        # ステージ内のシンボリックリンク
        s = stage(self.root, id="20261022-link")
        os.remove(s / "files/src/api/app.py")
        (s / "files/src/api/app.py").symlink_to("/etc/hostname")
        self.assertEqual(deployer(self.root, self.env).run_all(), [("20261022-link", "invalid")])
        self.assertEqual(self.env.cmds, [])

    def test_hand_crafted_manifest_with_forbidden_dest_is_invalid(self):
        s = stage(self.root)
        m = json.loads((s / "manifest.json").read_text())
        m["files"] = [{"dest": ".env", "sha256": m["files"][0]["sha256"]}]
        (s / "manifest.json").write_text(json.dumps(m))
        self.assertEqual(deployer(self.root, self.env).run_all(), [("20261022-test", "invalid")])
        self.assertFalse((self.root / ".env").exists())

    def test_expired(self):
        stage(self.root, now=at(1) - timedelta(days=10))
        self.assertEqual(deployer(self.root, self.env).run_all(), [("20261022-test", "invalid")])

    def test_duplicate_applied_id(self):
        stage(self.root)
        d = deployer(self.root, self.env)
        self.assertEqual(d.run_all(), [("20261022-test", "applied")])
        stage(self.root, files={"src/api/app.py": "VALUE = 3\n"})
        self.assertEqual(d.run_all(), [("20261022-test", "invalid")])
        self.assertEqual(self.read(), NEW_PY)

    def test_syntax_error_rolls_back_without_restart(self):
        stage(self.root, files={"src/api/app.py": "def broken(:\n", "src/api/new_module.py": "X = 1\n"})
        d = deployer(self.root, self.env)
        self.assertEqual(d.run_all(), [("20261022-test", "failed_rolled_back")])
        self.assertEqual(self.read(), OLD_PY)                                   # 元に戻る
        self.assertFalse((self.root / "src/api/new_module.py").exists())        # 新規ファイルは消える
        self.assertEqual([c for c in self.env.cmds if any("restart_server" in x for x in c)], [])   # 再起動していない
        self.assertEqual(self.env.notes[-1][2], "critical")
        self.assertTrue((self.root / "data/local/deploy/failed/20261022-test/result.json").exists())

    def test_restart_failure_rolls_back_and_restarts_again(self):
        stage(self.root)
        self.env.rc = {"restart_server": 1}
        d = deployer(self.root, self.env)
        res = d.run_all()
        self.assertEqual(res, [("20261022-test", "failed_rollback_error")])   # ロールバック後の再起動も失敗する設定
        self.assertEqual(self.read(), OLD_PY)
        self.assertEqual(len([c for c in self.env.cmds if any("restart_server" in x for x in c)]), 2)
        self.assertEqual(self.env.notes[-1][2], "critical")

    def test_verify_failure_rolls_back_then_recovers(self):
        stage(self.root)
        # 反映後の検証だけ失敗（18 回 503）し、ロールバック後は 200 に戻る
        self.env.health = [503] * 18 + [200]
        d = deployer(self.root, self.env)
        self.assertEqual(d.run_all(), [("20261022-test", "failed_rolled_back")])
        self.assertEqual(self.read(), OLD_PY)
        self.assertEqual(len([c for c in self.env.cmds if any("restart_server" in x for x in c)]), 2)   # 反映時 + ロールバック時
        self.assertIn("critical", [n[2] for n in self.env.notes])

    def test_frontend_rollback_uses_force_build(self):
        stage(self.root, files={"frontend/components/A.tsx": "export const A = 1;\n"}, steps=("frontend_build_switch",))
        self.env.rc = {"server_watchdog": 1}
        res = deployer(self.root, self.env).run_all()
        self.assertTrue(res[0][1].startswith("failed"))
        builds = [c for c in self.env.cmds if any("frontend_build" in x for x in c)]
        self.assertEqual(len(builds), 2)     # 反映時 + ロールバック時の再ビルド

    def test_window_too_short_defers_or_extends(self):
        stage(self.root, steps=("api_restart", "frontend_build_switch"))
        late = at(3, 55)     # 残り 5 分 < 必要な秒数
        self.assertEqual(deployer(self.root, self.env, now=late).run_all(), [("20261022-test", "deferred")])
        self.assertEqual(self.read(), OLD_PY)
        self.assertEqual(self.env.cmds, [])
        # 延長が許可されていれば、実行時 override で窓を延ばして続行
        for p in (self.root / "data/local/deploy/pending").iterdir():
            import shutil
            shutil.rmtree(p)
        stage(self.root, id="20261022-ext", steps=("api_restart", "frontend_build_switch"), extend_window_minutes=45)
        ov = Path(tempfile.mkdtemp()) / "ov.json"
        with mock.patch.dict(os.environ, {"KEIBA_MAINTENANCE_OVERRIDE": str(ov)}):
            M.clear_cache()
            d = deployer(self.root, self.env, now=late)
            d.cfg = None   # override ファイルを読む設定ロードに
            with mock.patch.object(M, "load_config", side_effect=lambda *a, **k: M._merge(M.DEFAULT_CONFIG, json.loads(ov.read_text()) if ov.exists() else {})):
                res = d.run_all()
            self.assertEqual(res, [("20261022-ext", "applied")])
            self.assertTrue(ov.exists())
            self.assertIn("延長", json.loads(ov.read_text())["override"]["message"])
        M.clear_cache()

    def test_stops_after_deferred_to_keep_order(self):
        stage(self.root, id="20261022-a")
        stage(self.root, id="20261022-b")
        self.env.load = 99.0
        res = deployer(self.root, self.env, max_wait_sec=0).run_all()
        self.assertEqual([r[1] for r in res], ["deferred"])    # 後続は処理しない

    def test_invalid_does_not_block_following(self):
        s = stage(self.root, id="20261022-a")
        (s / "files/src/api/app.py").write_text("tampered\n")
        stage(self.root, id="20261022-b")
        res = dict(deployer(self.root, self.env).run_all())
        self.assertEqual(res, {"20261022-a": "invalid", "20261022-b": "applied"})

    def test_notify_flag_false_is_quiet_on_success(self):
        stage(self.root, notify=True)
        s = self.root / "data/local/deploy/pending/20261022-test"
        m = json.loads((s / "manifest.json").read_text())
        m["notify"] = False
        (s / "manifest.json").write_text(json.dumps(m))
        deployer(self.root, self.env).run_all()
        self.assertEqual(self.env.notes, [])

    def test_cli_list_and_cancel(self):
        stage(self.root)
        d = deployer(self.root, self.env)
        self.assertEqual([p.name for p in d.list_pending()], ["20261022-test"])
        rc = MD.main(["--root", str(self.root), "--cancel", "20261022-test"])
        self.assertEqual(rc, 0)
        self.assertEqual(d.list_pending(), [])
        self.assertTrue((self.root / "data/local/deploy/cancelled/20261022-test").exists())
        self.assertEqual(MD.main(["--root", str(self.root), "--cancel", "../x"]), 3)


if __name__ == "__main__":
    unittest.main()
