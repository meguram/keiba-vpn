"""メンテナンス時間帯の自動反映のために、変更をステージする（ローカルで実行）。

  # 変更ファイルを指定して manifest 付きのステージを作り、ホストへ送る
  python -m src.scripts.ops.stage_deploy create --id 20261022-megu-table --desc "めぐ指数の専用テーブル" \\
      --steps frontend_build_switch --file frontend/components/MeguTable.tsx --file frontend/lib/pastRunsView.ts
  # コミット済みの内容をそのまま送る（作業ツリーの未コミット変更を混ぜない）
  python -m src.scripts.ops.stage_deploy create --id ... --ref aadc366 --file src/api/app.py --steps api_restart \\
      --verify "http://127.0.0.1:23220/api/race/202604040201/predictions|status=200|key=prob_calibration"
  python -m src.scripts.ops.stage_deploy list           # ホストの pending / applied / failed
  python -m src.scripts.ops.stage_deploy cancel <id>    # ホストの pending を取り消す（applied には触れない）

ランナー（ホストの root cron、毎日 JST 03:05）が、窓の中で前提条件を待ってから適用する。詳細は docs/operations/maintenance_deploy.md。
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable

from src.scripts.ops import deploy_lib as D

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HOST = "keiba-remote"
DEFAULT_REMOTE_DIR = "/home/jovyan/work/keiba-vpn"
PENDING_REL = "data/local/deploy/pending"

Run = Callable[..., subprocess.CompletedProcess]


def parse_verify(spec: str) -> dict:
    """'URL|status=200|key=a.b' → {"url", "expect_status", "json_key"}。"""
    parts = spec.split("|")
    out: dict = {"url": parts[0], "expect_status": 200}
    for p in parts[1:]:
        k, _, v = p.partition("=")
        if k == "status":
            out["expect_status"] = int(v)
        elif k == "key":
            out["json_key"] = v
        else:
            raise SystemExit(f"--verify のオプションが不明: {p}")
    return out


def read_content(rel: str, ref: str | None, repo: Path) -> bytes:
    D.check_dest_allowed(rel)
    if ref:
        r = subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{rel}"], capture_output=True)
        if r.returncode != 0:
            raise SystemExit(f"{ref}:{rel} を読めません: {r.stderr.decode(errors='replace')[:200]}")
        return r.stdout
    p = repo / rel
    if p.is_symlink() or not p.is_file():
        raise SystemExit(f"通常ファイルではありません（シンボリックリンク不可）: {rel}")
    return p.read_bytes()


def create_stage(out_root: Path, *, id: str, description: str, rels: list[str], steps: list[str], ref: str | None = None,
                 repo: Path = REPO_ROOT, verify: list[dict] | None = None, health_url: str | None = None,
                 frontend_url: str | None = None, expires_hours: float = 72.0, max_load: float | None = None,
                 extend_window_minutes: int = 0, notify: bool = True, now: datetime | None = None) -> Path:
    """out_root/<id>/ に files/<dest> と manifest.json を作る（sha256 つき）。"""
    stage = out_root / id
    if stage.exists():
        raise SystemExit(f"すでに存在します: {stage}")
    files = []
    for rel in rels:
        data = read_content(rel, ref, repo)
        dst = stage / "files" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        import hashlib

        files.append({"dest": rel, "sha256": hashlib.sha256(data).hexdigest()})
    v: dict = {"urls": verify or []}
    if health_url:
        v["health_url"] = health_url
    if frontend_url:
        v["frontend_url"] = frontend_url
    pre = {"max_load": max_load} if max_load else {}
    m = D.build_manifest(id=id, description=description, files=files, steps=steps, verify=v, preconditions=pre,
                         expires_hours=expires_hours, extend_window_minutes=extend_window_minutes, notify=notify, now=now)
    (stage / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    D.verify_staged_files(stage, m)
    return stage


def push(stage: Path, host: str, remote_dir: str, run: Run = subprocess.run) -> None:
    """ホストへ送る。.incoming-<id> に置いてから pending/<id> へ名前変更（ランナーが途中の状態を拾わない）。"""
    id = stage.name
    rd = shlex.quote(remote_dir)
    tmp = f"/tmp/stage_{id}"
    run(["ssh", host, f"rm -rf {shlex.quote(tmp)}"], check=True)
    run(["scp", "-q", "-r", str(stage), f"{host}:{tmp}"], check=True)
    cmd = (f"set -e; P={rd}/{PENDING_REL}; sudo -n mkdir -p $P; "
           f"[ ! -e $P/{shlex.quote(id)} ] || {{ echo 'すでに pending にあります'; exit 3; }}; "
           f"sudo -n rm -rf $P/.incoming-{shlex.quote(id)}; sudo -n cp -a {shlex.quote(tmp)} $P/.incoming-{shlex.quote(id)}; "
           f"sudo -n chown -R root:root $P/.incoming-{shlex.quote(id)}; sudo -n mv $P/.incoming-{shlex.quote(id)} $P/{shlex.quote(id)}; "
           f"rm -rf {shlex.quote(tmp)}; echo staged")
    run(["ssh", host, cmd], check=True)


def remote_list(host: str, remote_dir: str, run: Run = subprocess.run) -> str:
    rd = shlex.quote(remote_dir)
    cmd = (f"cd {rd}/data/local/deploy 2>/dev/null || {{ echo '(ステージ無し)'; exit 0; }}; "
           "for d in pending applied failed; do echo \"== $d\"; ls -1t $d 2>/dev/null | grep -v '^\\.' | head -20; done; "
           "sudo -n tail -n 12 ../meta/deploy_audit.jsonl 2>/dev/null | cut -c1-200")
    r = run(["ssh", host, cmd], capture_output=True, text=True)
    return r.stdout + r.stderr


def remote_cancel(host: str, remote_dir: str, id: str, run: Run = subprocess.run) -> None:
    if not D.ID_RE.match(id):
        raise SystemExit(f"不正な id: {id}")
    rd = shlex.quote(remote_dir)
    cmd = (f"set -e; P={rd}/{PENDING_REL}; [ -d $P/{shlex.quote(id)} ] || {{ echo 'pending にありません'; exit 3; }}; "
           f"sudo -n mkdir -p {rd}/data/local/deploy/cancelled; sudo -n mv $P/{shlex.quote(id)} {rd}/data/local/deploy/cancelled/{shlex.quote(id)}-$(date +%s); echo cancelled")
    run(["ssh", host, cmd], check=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create", help="ステージを作ってホストへ送る")
    c.add_argument("--id", required=True)
    c.add_argument("--desc", required=True)
    c.add_argument("--file", action="append", required=True, help="repo 相対パス（繰り返し可）")
    c.add_argument("--steps", nargs="*", default=[], choices=list(D.STEPS))
    c.add_argument("--ref", help="git の ref から内容を取る（既定: 作業ツリー）")
    c.add_argument("--verify", action="append", default=[], help="'URL|status=200|key=a.b'（繰り返し可）")
    c.add_argument("--health-url", default="http://127.0.0.1:23220/api/health")
    c.add_argument("--frontend-url", default="http://127.0.0.1:23200/")
    c.add_argument("--expires-hours", type=float, default=72.0)
    c.add_argument("--max-load", type=float)
    c.add_argument("--quiet", action="store_true", help="成功時の Slack 通知を出さない（失敗は常に緊急通知）")
    c.add_argument("--extend-window-minutes", type=int, default=0)
    c.add_argument("--host", default=DEFAULT_HOST)
    c.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    c.add_argument("--no-push", action="store_true", help="ホストへ送らず、ローカルのステージだけ作る（--out に出力）")
    c.add_argument("--out", help="--no-push のときの出力先")
    ls = sub.add_parser("list", help="ホストの pending / applied / failed を表示")
    ls.add_argument("--host", default=DEFAULT_HOST)
    ls.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    cn = sub.add_parser("cancel", help="ホストの pending を取り消す")
    cn.add_argument("id")
    cn.add_argument("--host", default=DEFAULT_HOST)
    cn.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    a = ap.parse_args(argv)
    if a.cmd == "list":
        print(remote_list(a.host, a.remote_dir))
        return 0
    if a.cmd == "cancel":
        remote_cancel(a.host, a.remote_dir, a.id)
        return 0
    out_root = Path(a.out) if (a.no_push and a.out) else Path(tempfile.mkdtemp(prefix="stage_"))
    out_root.mkdir(parents=True, exist_ok=True)
    stage = create_stage(out_root, id=a.id, description=a.desc, rels=a.file, steps=a.steps, ref=a.ref,
                         verify=[parse_verify(v) for v in a.verify], health_url=a.health_url if "api_restart" in a.steps else None,
                         frontend_url=a.frontend_url if "frontend_build_switch" in a.steps else None,
                         expires_hours=a.expires_hours, max_load=a.max_load, extend_window_minutes=a.extend_window_minutes, notify=not a.quiet)
    m = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
    print(f"ステージを作成: {stage}（{len(m['files'])} ファイル、steps={m['steps']}、有効期限 {m['expires_at']}）")
    for f in m["files"]:
        print(f"  {f['sha256'][:12]}  {f['dest']}")
    if a.no_push:
        return 0
    push(stage, a.host, a.remote_dir)
    print(f"ホストへ送りました。次の窓（毎日 JST 03:05）で適用されます。確認: python -m src.scripts.ops.stage_deploy list / 取り消し: ... cancel {a.id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
