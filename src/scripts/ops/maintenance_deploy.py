"""メンテナンス時間帯（毎日 JST 03:00–04:00）に、ステージ済みの変更を安全に反映するランナー（ホストの root cron から）。

  data/local/deploy/pending/<id>/{manifest.json, files/<dest>} を古い順に処理する。**manifest に無いパスは置かない**。
  窓の外では何もしない（--force-window で手動）。前提条件（スクレイピングキューが空・運用ジョブが動いていない・load・ディスク）を
  窓の終了前まで待ち、満たせなければ次の窓に持ち越す。再起動/切替は窓の終了までに終わらせられるときだけ新規に始める。
  sha256 検証 → バックアップ → 原子的に配置 → 構文チェック → 手順（API 再起動、フロントのビルドと切替）→ 検証。
  失敗したら自動でロールバック（退避を戻して再起動・再ビルド）して Slack に緊急通知。成功は applied/、失敗は failed/ へ移す。

  python -m src.scripts.ops.maintenance_deploy            # 窓の中なら pending を処理
  python -m src.scripts.ops.maintenance_deploy --dry-run  # 検証だけ（何も変えない）
  python -m src.scripts.ops.maintenance_deploy --list / --cancel <id> / --force-window
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from src.scripts.ops import deploy_lib as D
from src.utils import maintenance as M

logger = logging.getLogger("ops.maintenance_deploy")
REPO_ROOT = Path(__file__).resolve().parents[3]

API_PORT = int(os.environ.get("KEIBA_DEPLOY_API_PORT", "23220"))
FRONT_PORT = int(os.environ.get("KEIBA_DEPLOY_FRONT_PORT", "23200"))
API_BASE = f"http://127.0.0.1:{API_PORT}"
STEP_NEED_SEC = {"api_restart": 180, "frontend_build_switch": 720}   # 窓の残りがこれ未満なら、その手順を含む変更は新規に始めない
BASE_NEED_SEC = 90                                                    # 配置・検証の余裕
WAIT_POLL_SEC = 30
# 動いていると再起動で困る運用ジョブ（コマンド行にこの文字列を含むプロセス）
BLOCK_PATTERNS = ("raceday-runner", "raceday-result-runner", "raceday-weight-runner", "weight_triggered_predict",
                  "serving-state-precompute", "lap_forecast_daily", "tracking_v2_precompute", "backfill_horse_training_recent")

RunCmd = Callable[..., "tuple[int, str]"]
HttpGet = Callable[..., "tuple[int, str]"]


# ───────────────────────── 既定の実装（テストでは差し替える） ─────────────────────────

def default_run(cmd: list[str], *, timeout: int = 300, env: dict[str, str] | None = None, cwd: str | None = None) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, env={**os.environ, **(env or {})})
        return r.returncode, ((r.stdout or "") + (r.stderr or ""))[-4000:]
    except subprocess.TimeoutExpired as exc:
        return 124, f"timeout {timeout}s: {(exc.stdout or b'')[-500:]!r}"


def default_http_get(url: str, *, timeout: float = 5.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read(200_000).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, ""
    except Exception as exc:  # noqa: BLE001
        return 0, str(exc)[:200]


def default_processes() -> list[str]:
    r = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True)
    return r.stdout.splitlines()


def default_notify(title: str, lines: list[str], severity: str) -> None:
    if os.environ.get("KEIBA_SLACK_NOTIFY") != "1":
        return
    try:
        from src.utils import slack_notify as SN

        SN.notify_event(title, lines, severity=severity, channel="alert" if severity != "normal" else "report")
    except Exception as exc:  # noqa: BLE001  通知の失敗で反映を止めない
        logger.warning("Slack 通知に失敗（無視）: %s", exc)


class Deployer:
    def __init__(self, root: Path = REPO_ROOT, *, clock: Callable[[], datetime] | None = None, run: RunCmd = default_run,
                 http_get: HttpGet = default_http_get, sleep: Callable[[float], None] = time.sleep,
                 load_fn: Callable[[], float] | None = None, disk_gb_fn: Callable[[], float] | None = None,
                 processes: Callable[[], list[str]] = default_processes, notify: Callable[[str, list[str], str], None] = default_notify,
                 cfg: dict | None = None, env: dict[str, str] | None = None, max_wait_sec: float | None = None) -> None:
        self.root = Path(root)
        self.base = self.root / "data" / "local" / "deploy"
        self.pending, self.applied, self.failed = (self.base / n for n in ("pending", "applied", "failed"))
        self.audit_path = self.root / "data" / "local" / "meta" / "deploy_audit.jsonl"
        self.clock = clock or (lambda: datetime.now(D.JST))
        self.run, self.http_get, self.sleep, self.processes, self.notify_fn = run, http_get, sleep, processes, notify
        self.load_fn = load_fn or (lambda: os.getloadavg()[0])
        self.disk_gb_fn = disk_gb_fn or (lambda: shutil.disk_usage(self.root).free / 1e9)
        self.cfg = cfg
        self.env = env if env is not None else dict(os.environ)
        self.max_wait_sec = max_wait_sec

    # ── 監査・通知 ──
    def audit(self, event: str, id: str | None = None, **kw: Any) -> None:
        rec = {"ts": self.clock().isoformat(timespec="seconds"), "event": event, "id": id, **kw}
        logger.info("%s", json.dumps(rec, ensure_ascii=False, default=str))
        try:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            logger.warning("監査ログを書けない: %s", exc)

    def notify(self, m: dict | None, title: str, lines: list[str], severity: str = "normal") -> None:
        if m is not None and not m.get("notify", True):
            return
        self.notify_fn(title, lines, severity)

    # ── 窓 ──
    def window(self) -> dict:
        cfg = self.cfg if self.cfg is not None else M.load_config(ttl=0)
        return M.compute_status(cfg, self.clock(), enabled=True)

    # ── ステージの一覧 ──
    def list_pending(self) -> list[Path]:
        if not self.pending.is_dir():
            return []
        items = [p for p in self.pending.iterdir() if p.is_dir() and not p.name.startswith(".") and not p.is_symlink()]
        return sorted(items, key=lambda p: (p.stat().st_mtime, p.name))

    def move(self, stage: Path, dest_root: Path, result: dict) -> Path:
        dest_root.mkdir(parents=True, exist_ok=True)
        dst = dest_root / stage.name
        if dst.exists():
            dst = dest_root / f"{stage.name}-{int(time.time())}"
        (stage / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        shutil.move(str(stage), str(dst))
        return dst

    # ── 前提条件 ──
    def preconditions_ok(self, m: dict) -> tuple[bool, list[str]]:
        pre = {**D.DEFAULT_PRECONDITIONS, **(m.get("preconditions") or {})}
        bad: list[str] = []
        if pre.get("queue_idle", True):
            st, body = self.http_get(f"{API_BASE}/api/scrape-jobs")
            if st == 200:
                try:
                    s = json.loads(body).get("stats", {})
                    if int(s.get("running", 0)) + int(s.get("queued", 0)) > 0:
                        bad.append(f"スクレイピングキューが空でない（running={s.get('running')} queued={s.get('queued')}）")
                except (ValueError, TypeError):
                    bad.append("スクレイピングキューの応答を読めない")
        load = self.load_fn()
        if load >= float(pre.get("max_load", 14.0)):
            bad.append(f"load が高い（{load:.1f} ≥ {pre.get('max_load')}）")
        free = self.disk_gb_fn()
        if free < float(pre.get("min_disk_gb", 3.0)):
            bad.append(f"ディスクの空きが少ない（{free:.1f}GB）")
        pats = tuple(BLOCK_PATTERNS) + tuple(pre.get("no_processes") or ())
        ps = self.processes()
        hit = sorted({p for p in pats if any(p in line and "maintenance_deploy" not in line for line in ps)})
        if hit:
            bad.append("運用ジョブが動いている: " + ", ".join(hit))
        return (not bad), bad

    def need_seconds(self, m: dict) -> int:
        return BASE_NEED_SEC + sum(STEP_NEED_SEC[s] for s in m.get("steps", []))

    def wait_for_conditions(self, m: dict, deadline_sec: float) -> tuple[bool, list[str]]:
        """窓の中で前提条件を待つ。deadline_sec（残り秒）から必要な秒数を引いた時点まで。"""
        waited = 0.0
        limit = max(0.0, deadline_sec - self.need_seconds(m)) if self.max_wait_sec is None else self.max_wait_sec
        while True:
            ok, why = self.preconditions_ok(m)
            if ok:
                return True, []
            if waited >= limit:
                return False, why
            self.audit("waiting", m["id"], reasons=why)
            self.sleep(WAIT_POLL_SEC)
            waited += WAIT_POLL_SEC

    # ── 配置とロールバック ──
    def place(self, stage: Path, m: dict, backup: Path) -> list[dict]:
        placed: list[dict] = []
        for f in m["files"]:
            dest_rel = f["dest"]
            dst = D.resolve_inside(self.root, dest_rel)
            existed = dst.exists()
            if existed:
                bk = backup / "files" / dest_rel
                bk.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dst, bk)
            placed.append({"dest": dest_rel, "existed": existed})
            (backup / "backup.json").write_text(json.dumps({"id": m["id"], "files": placed}, ensure_ascii=False, indent=1), encoding="utf-8")
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".deploy-tmp")
            shutil.copyfile(D.staged_path(stage, dest_rel), tmp)
            try:
                os.chmod(tmp, (dst.stat().st_mode & 0o777) if existed else 0o644)
            except OSError:
                pass
            with open(tmp, "rb") as fh:
                os.fsync(fh.fileno())
            os.replace(tmp, dst)
        return placed

    def syntax_check(self, m: dict) -> None:
        for f in m["files"]:
            dst = D.resolve_inside(self.root, f["dest"])
            if f["dest"].endswith(".py"):
                ast.parse(dst.read_text(encoding="utf-8"), filename=f["dest"])
            elif f["dest"].endswith(".json"):
                json.loads(dst.read_text(encoding="utf-8"))
            elif f["dest"].endswith(".sh"):
                rc, out = self.run(["bash", "-n", str(dst)], timeout=30)
                if rc != 0:
                    raise SyntaxError(f"{f['dest']}: {out[:200]}")

    def restore(self, backup: Path, placed: list[dict]) -> None:
        for p in placed:
            dst = D.resolve_inside(self.root, p["dest"])
            if p["existed"]:
                bk = backup / "files" / p["dest"]
                tmp = dst.with_name(dst.name + ".deploy-tmp")
                shutil.copyfile(bk, tmp)
                os.replace(tmp, dst)
            elif dst.exists():
                dst.unlink()

    # ── 手順 ──
    def step_cmds(self, step: str, *, rollback: bool = False) -> list[tuple[list[str], int, dict[str, str]]]:
        if step == "api_restart":
            return [(["bash", "scripts/server/restart_server.sh"], 240,
                     {"PYTHON": "/opt/conda/bin/python3", "PORT": str(API_PORT), "TZ": "Asia/Tokyo"})]
        if step == "frontend_build_switch":
            build_env = {"KEIBA_FORCE_BUILD": "1"} if rollback else {}
            return [(["bash", "scripts/server/frontend_build.sh", "stg"], 900, build_env),
                    (["bash", "scripts/server/server_watchdog.sh", "--profile", "stg", "--switch-frontend"], 300,
                     {"PATH": "/opt/conda/bin:/usr/local/bin:/usr/bin:/bin"})]
        raise ValueError(step)

    def exec_step(self, step: str, *, rollback: bool = False) -> tuple[bool, str]:
        for cmd, timeout, env in self.step_cmds(step, rollback=rollback):
            rc, out = self.run(cmd, timeout=timeout, env=env, cwd=str(self.root))
            ok = rc in (0, 2) if cmd[1].endswith("frontend_build.sh") else rc == 0
            self.audit("step", None, step=step, cmd=" ".join(cmd[:3]), rc=rc, rollback=rollback)
            if not ok:
                return False, f"{' '.join(cmd[:3])} rc={rc}: {out[-300:]}"
        return True, ""

    def wait_http(self, url: str, expect: int = 200, key: str | None = None, tries: int = 18, gap: float = 5.0) -> tuple[bool, str]:
        last = ""
        for _ in range(tries):
            st, body = self.http_get(url, timeout=10.0)
            if st == expect:
                if key:
                    try:
                        cur: Any = json.loads(body)
                        for part in key.split("."):
                            cur = cur[part]
                    except (ValueError, KeyError, TypeError, IndexError):
                        last = f"{url}: キー {key} が無い"
                        self.sleep(gap)
                        continue
                return True, ""
            last = f"{url}: status={st}（期待 {expect}）"
            self.sleep(gap)
        return False, last

    def verify(self, m: dict) -> tuple[bool, str]:
        v = m.get("verify") or {}
        steps = m.get("steps", [])
        checks: list[tuple[str, int, str | None]] = []
        if "api_restart" in steps:
            checks.append((v.get("health_url") or f"{API_BASE}/api/health", 200, None))
        for u in v.get("urls") or []:
            checks.append((u["url"], int(u.get("expect_status", 200)), u.get("json_key")))
        if "frontend_build_switch" in steps:
            checks.append((v.get("frontend_url") or f"http://127.0.0.1:{FRONT_PORT}/", 200, None))
        for url, st, key in checks:
            ok, why = self.wait_http(url, st, key)
            if not ok:
                return False, why
        return True, ""

    def extend_window(self, minutes: int, need_sec: int) -> None:
        """窓の残りが足りない変更のために、実行時の override で終了を延ばす（git 管理の設定は書き換えない）。"""
        now = self.clock()
        st = self.window()
        start = st.get("window_start") or now.isoformat(timespec="seconds")
        end = max(datetime.fromisoformat(st["window_end"]), now + timedelta(seconds=need_sec + 300))
        end = min(end, datetime.fromisoformat(st["window_end"]) + timedelta(minutes=minutes))
        p = M.runtime_override_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"override": {"start": start, "end": end.isoformat(timespec="seconds"),
                                              "message": "システムの更新のため、メンテナンスを延長しています。"}}, ensure_ascii=False), encoding="utf-8")
        M.clear_cache()
        self.audit("window_extended", None, until=end.isoformat(timespec="seconds"))

    # ── 1 件の処理 ──
    def process(self, stage: Path, *, dry_run: bool = False, force_window: bool = False) -> str:
        id = stage.name
        try:
            m = D.load_manifest(stage)
            D.validate_manifest(m, now=self.clock(), check_expiry=True)
            D.verify_staged_files(stage, m)
            for f in m["files"]:
                D.resolve_inside(self.root, f["dest"])
        except D.ManifestError as exc:
            self.audit("invalid", id, reason=str(exc))
            if not dry_run:
                self.move(stage, self.failed, {"status": "invalid", "reason": str(exc)})
                self.notify(None, f"反映ステージが不正: {id}", [str(exc), "適用していません（failed/ へ移動）"], "warning")
            return "invalid"
        if (self.applied / id).exists():
            self.audit("duplicate", id)
            if not dry_run:
                self.move(stage, self.failed, {"status": "invalid", "reason": "同じ id がすでに適用済み"})
            return "invalid"
        need = self.need_seconds(m)
        st = self.window()
        if dry_run:
            self.audit("dry_run", id, files=[f["dest"] for f in m["files"]], steps=m["steps"], need_sec=need, window_active=st["active"])
            return "dry_run"
        # 窓の残りが足りない: 延長が許可されていれば延ばし、無ければ持ち越し
        remaining = st.get("remaining_sec") or 0
        if force_window and not st["active"]:
            remaining = need + 1800        # 手動の強制実行（窓の外）: 残り時間の制約は掛けない
        if remaining < need:
            if m.get("extend_window_minutes", 0) > 0 and st["active"]:
                self.extend_window(m["extend_window_minutes"], need)
                st = self.window()
                remaining = st.get("remaining_sec") or 0
            if remaining < need:
                self.audit("deferred", id, reason="window_remaining", remaining=remaining, need=need)
                return "deferred"
        ok, why = self.wait_for_conditions(m, remaining)
        if not ok:
            self.audit("deferred", id, reason="preconditions", detail=why)
            self.notify(m, f"反映を次の窓に持ち越し: {id}", why, "normal")
            return "deferred"
        self.notify(m, f"メンテナンス中の反映を開始: {id}", [m.get("description", ""), f"ファイル {len(m['files'])} 件、手順 {', '.join(m['steps']) or 'なし'}"], "normal")
        self.audit("start", id, description=m.get("description"), files=[f["dest"] for f in m["files"]], steps=m["steps"])
        backup = self.root / "data" / "local" / "meta" / "cron_backups" / f"deploy_{id}"
        backup.mkdir(parents=True, exist_ok=True)
        placed: list[dict] = []
        done_steps: list[str] = []
        fail = ""
        try:
            placed = self.place(stage, m, backup)
            self.syntax_check(m)
            self.audit("placed", id, count=len(placed))
            for s in m["steps"]:
                done_steps.append(s)
                ok, why = self.exec_step(s)
                if not ok:
                    raise RuntimeError(f"手順 {s} が失敗: {why}")
            ok, why = self.verify(m)
            if not ok:
                raise RuntimeError(f"検証に失敗: {why}")
        except Exception as exc:  # noqa: BLE001  どの段階の失敗でも自動ロールバック
            fail = f"{type(exc).__name__}: {exc}"[:600]
        if not fail:
            dst = self.move(stage, self.applied, {"status": "applied", "applied_at": self.clock().isoformat(timespec="seconds"), "backup": str(backup)})
            self.audit("applied", id, moved_to=str(dst))
            self.notify(m, f"メンテナンス中の反映が完了: {id}", [m.get("description", ""), "検証 OK"], "normal")
            return "applied"
        self.audit("failed", id, reason=fail)
        rb_fail = ""
        try:
            self.restore(backup, placed)
            for s in done_steps:
                ok, why = self.exec_step(s, rollback=True)
                if not ok:
                    rb_fail = f"ロールバック後の {s} に失敗: {why}"
                    break
            if not rb_fail and done_steps:
                ok, why = self.verify({**m, "verify": {"urls": []}})
                if not ok:
                    rb_fail = f"ロールバック後の検証に失敗: {why}"
        except Exception as exc:  # noqa: BLE001
            rb_fail = f"ロールバック中に例外: {exc}"[:400]
        status = "failed_rollback_error" if rb_fail else "failed_rolled_back"
        self.audit("rollback", id, ok=not rb_fail, detail=rb_fail)
        self.move(stage, self.failed, {"status": status, "reason": fail, "rollback_error": rb_fail, "backup": str(backup)})
        lines = [fail, "自動でロールバックしました" if not rb_fail else f"ロールバックにも失敗: {rb_fail}（人手の対応が必要）", f"退避: {backup}"]
        self.notify(None, f"メンテナンス中の反映に失敗: {id}", lines, "critical")
        return status

    # ── 全体 ──
    def run_all(self, *, force_window: bool = False, dry_run: bool = False) -> list[tuple[str, str]]:
        results: list[tuple[str, str]] = []
        items = self.list_pending()
        if not items:
            return results
        st = self.window()
        if not st["active"] and not force_window and not dry_run:
            self.audit("outside_window", None, pending=[p.name for p in items], next_start=st.get("window_start"))
            return results
        for stage in items:
            res = self.process(stage, dry_run=dry_run, force_window=force_window)
            results.append((stage.name, res))
            if res == "deferred":
                break       # 前提条件や窓が足りない: 順序を保つため後続も持ち越す
        return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(REPO_ROOT))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-window", action="store_true", help="窓の外でも実行（手動）")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--cancel", metavar="ID")
    ap.add_argument("--max-wait-sec", type=float, help="前提条件を待つ最大秒数（既定: 窓の終了前まで）")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    d = Deployer(Path(a.root), max_wait_sec=a.max_wait_sec)
    if a.list:
        for p in d.list_pending():
            try:
                m = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
                print(f"{p.name}\t{m.get('created_at')}\t{m.get('description')}\tsteps={m.get('steps')}\texpires={m.get('expires_at')}")
            except (OSError, ValueError):
                print(f"{p.name}\t(manifest を読めない)")
        return 0
    if a.cancel:
        src = d.pending / a.cancel
        if not D.ID_RE.match(a.cancel) or not src.is_dir():
            print("pending に無い id です")
            return 3
        d.move(src, d.base / "cancelled", {"status": "cancelled"})
        d.audit("cancelled", a.cancel)
        return 0
    # 同時実行を防ぐ（cron の重複・手動実行）
    lock = open(os.environ.get("KEIBA_DEPLOY_LOCK", "/tmp/keiba_maintenance_deploy.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("別の反映が実行中です")
        return 0
    results = d.run_all(force_window=a.force_window, dry_run=a.dry_run)
    for id, res in results:
        print(f"{id}: {res}")
    return 1 if any(r.startswith("failed") for _, r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
