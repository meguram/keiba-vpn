"""メンテナンス時間帯の自動反映（ステージされた変更だけを適用する）の共通部分: manifest の形・検証・ハッシュ。

方針: 自動で動くのは『ユーザーが事前にステージした manifest 付きの変更』だけ。manifest に無いパスは置かない。
許可パスは src/ config/ frontend/ scripts/ 配下のみ。.env・crontab・ランナー自身は人手（ここでは扱わない）。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = 1
ALLOWED_ROOTS = ("src/", "config/", "frontend/", "scripts/")
# ステージしても置かないもの（人手で行う・ランナー自身・環境ごとの秘密/生成物）
DENIED_PATTERNS = (
    r"(^|/)\.env[^/]*$",
    r"(^|/)\.git(/|$)",
    r"(^|/)node_modules(/|$)",
    r"(^|/)\.next[^/]*(/|$)",
    r"^scripts/ops/maintenance_deploy\.sh$",
    r"^src/scripts/ops/(maintenance_deploy|deploy_lib)\.py$",
    r"(^|/)crontab[^/]*$",
)
STEPS = ("api_restart", "frontend_build_switch")
ID_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._-]{2,79}$")
MAX_FILES = 200
MAX_FILE_BYTES = 20 * 1024 * 1024
DEFAULT_PRECONDITIONS = {"queue_idle": True, "max_load": 14.0, "min_disk_gb": 3.0, "no_processes": []}
JST = timezone(timedelta(hours=9))


class ManifestError(ValueError):
    pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_rel(path: str) -> str:
    """repo 相対の POSIX パスに正規化。絶対パス・..・バックスラッシュ・NUL は拒否。"""
    if not isinstance(path, str) or not path or "\x00" in path or "\\" in path:
        raise ManifestError(f"不正なパス: {path!r}")
    pp = PurePosixPath(path)
    if pp.is_absolute() or any(part in ("..", "") for part in pp.parts) or str(pp) != path:
        raise ManifestError(f"不正なパス（絶対パス・..・非正規）: {path!r}")
    return path


def check_dest_allowed(dest: str) -> None:
    normalize_rel(dest)
    if not dest.startswith(ALLOWED_ROOTS):
        raise ManifestError(f"許可されていない配置先（src/ config/ frontend/ scripts/ のみ）: {dest}")
    for pat in DENIED_PATTERNS:
        if re.search(pat, dest):
            raise ManifestError(f"配置対象外のパス（人手で行う/生成物）: {dest}")


def resolve_inside(root: Path, rel: str) -> Path:
    """root 配下に解決（シンボリックリンクを辿って外へ出る配置を拒否）。root 自体や親が symlink でも、解決後が root 内であること。"""
    base = root.resolve()
    target = (base / rel)
    # 既存の部分を resolve して root 内か確認（親ディレクトリが symlink で外を指す場合を検出）
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if base not in probe.resolve().parents and probe.resolve() != base:
        raise ManifestError(f"root の外に出るパス: {rel}")
    if target.is_symlink():
        raise ManifestError(f"配置先がシンボリックリンク: {rel}")
    return target


def build_manifest(*, id: str, description: str, files: list[dict[str, str]], steps: list[str], verify: dict | None = None,
                   preconditions: dict | None = None, expires_hours: float = 72.0, extend_window_minutes: int = 0,
                   notify: bool = True, now: datetime | None = None) -> dict:
    now = now or datetime.now(JST)
    m = {
        "schema": SCHEMA, "id": id, "description": description, "created_at": now.isoformat(timespec="seconds"),
        "expires_at": (now + timedelta(hours=expires_hours)).isoformat(timespec="seconds"),
        "files": files, "steps": list(steps), "verify": verify or {},
        "preconditions": {**DEFAULT_PRECONDITIONS, **(preconditions or {})},
        "rollback": {"method": "restore_backup"}, "extend_window_minutes": int(extend_window_minutes), "notify": bool(notify),
    }
    validate_manifest(m)
    return m


def validate_manifest(m: dict, *, now: datetime | None = None, check_expiry: bool = False) -> None:
    """形・パス・ステップの検証（ファイルの実在・ハッシュは verify_staged_files で別に見る）。"""
    if not isinstance(m, dict) or m.get("schema") != SCHEMA:
        raise ManifestError("manifest の schema が違います")
    if not isinstance(m.get("id"), str) or not ID_RE.match(m["id"]):
        raise ManifestError(f"不正な id: {m.get('id')!r}")
    files = m.get("files")
    if not isinstance(files, list) or not files or len(files) > MAX_FILES:
        raise ManifestError(f"files は 1〜{MAX_FILES} 件")
    seen: set[str] = set()
    for f in files:
        if not isinstance(f, dict) or not isinstance(f.get("dest"), str) or not re.fullmatch(r"[0-9a-f]{64}", str(f.get("sha256", ""))):
            raise ManifestError(f"files の要素が不正: {f!r}")
        check_dest_allowed(f["dest"])
        if f["dest"] in seen:
            raise ManifestError(f"配置先が重複: {f['dest']}")
        seen.add(f["dest"])
        if f.get("path") is not None and f["path"] != f["dest"]:
            raise ManifestError("path は dest と同じにしてください（ステージ内の位置は files/<dest>）")
    steps = m.get("steps", [])
    if not isinstance(steps, list) or any(s not in STEPS for s in steps):
        raise ManifestError(f"steps は {STEPS} のみ: {steps!r}")
    if "extend_window_minutes" in m and not (isinstance(m["extend_window_minutes"], int) and 0 <= m["extend_window_minutes"] <= 60):
        raise ManifestError("extend_window_minutes は 0〜60")
    pre = m.get("preconditions") or {}
    if not isinstance(pre, dict) or not isinstance(pre.get("max_load", 14.0), (int, float)):
        raise ManifestError("preconditions が不正")
    if m.get("rollback", {}).get("method", "restore_backup") != "restore_backup":
        raise ManifestError("rollback.method は restore_backup のみ")
    if check_expiry:
        now = now or datetime.now(JST)
        try:
            exp = datetime.fromisoformat(m["expires_at"])
        except (KeyError, ValueError) as exc:
            raise ManifestError("expires_at が不正") from exc
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=JST)
        if now > exp:
            raise ManifestError(f"有効期限切れ: {m['expires_at']}")


def load_manifest(stage_dir: Path) -> dict:
    p = stage_dir / "manifest.json"
    if p.is_symlink() or not p.is_file():
        raise ManifestError("manifest.json がありません")
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(f"manifest.json を読めません: {exc}") from exc
    validate_manifest(m)
    if m["id"] != stage_dir.name:
        raise ManifestError(f"manifest の id とディレクトリ名が違います: {m['id']} != {stage_dir.name}")
    return m


def staged_path(stage_dir: Path, dest: str) -> Path:
    return stage_dir / "files" / dest


def verify_staged_files(stage_dir: Path, m: dict) -> None:
    """ステージ内のファイルが manifest 通りか（実在・通常ファイル・シンボリックリンクでない・サイズ・sha256）。manifest に無いファイルがあれば拒否。"""
    root = (stage_dir / "files").resolve()
    declared = {f["dest"] for f in m["files"]}
    for f in m["files"]:
        p = staged_path(stage_dir, f["dest"])
        if p.is_symlink() or not p.is_file():
            raise ManifestError(f"ステージにファイルがない/通常ファイルでない: {f['dest']}")
        if root not in p.resolve().parents:
            raise ManifestError(f"ステージの外を指す: {f['dest']}")
        if p.stat().st_size > MAX_FILE_BYTES:
            raise ManifestError(f"大きすぎるファイル: {f['dest']}")
        if sha256_file(p) != f["sha256"]:
            raise ManifestError(f"sha256 が一致しない: {f['dest']}")
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for fn in filenames:
            rel = (Path(dirpath) / fn).relative_to(root).as_posix()
            if rel not in declared:
                raise ManifestError(f"manifest に無いファイルがステージにある: {rel}")
        for dn in dirnames:
            if (Path(dirpath) / dn).is_symlink():
                raise ManifestError(f"ステージにシンボリックリンクのディレクトリ: {dn}")
