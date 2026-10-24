"""Cloud Run の API が読む『GCS に無いローカル専用データ』の束を作る（push）・取り込む（pull）。

    # ホスト/Jobs: 何が入るか・サイズの確認（GCS に触れない）
    python -m src.scripts.ops.serving_state push --dry-run
    # ホスト/Jobs: 束を作って GCS へ（<prefix>/<group>/<ts>.tar.gz と latest.json。sha256 つき）
    python -m src.scripts.ops.serving_state push --group core
    # Cloud Run の起動時（src/api/cloudrun_entry.py が KEIBA_SERVING_STATE_PULL=1 のとき呼ぶ）。手動でも可
    python -m src.scripts.ops.serving_state pull --group core

束の中身は ``config/serving_state.json``。展開先は repo ルート（``data/`` 配下のみ許可。パスの抜け出しは拒否）。
dev では GCS に触れない（gcp_guard）。``--local-dir`` で GCS の代わりにディレクトリを使える（検証・テスト用）。
"""

from __future__ import annotations

import argparse
import fnmatch
import gzip
import hashlib
import json
import logging
import os
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "config" / "serving_state.json"
logger = logging.getLogger("ops.serving_state")
ALLOWED_TOP = "data"


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def select_files(root: Path, group: dict) -> list[Path]:
    seen: dict[str, Path] = {}
    for pat in group.get("include") or []:
        for p in root.glob(pat):
            if p.is_file() and not p.is_symlink():
                seen[p.relative_to(root).as_posix()] = p
    ex = group.get("exclude") or []
    return [seen[k] for k in sorted(seen) if not any(fnmatch.fnmatch(k, e) for e in ex)]


def summarize(root: Path, files: list[Path]) -> dict:
    total = sum(p.stat().st_size for p in files)
    tops: dict[str, int] = {}
    for p in files:
        rel = p.relative_to(root).parts
        k = "/".join(rel[:3])
        tops[k] = tops.get(k, 0) + p.stat().st_size
    return {"files": len(files), "bytes": total, "by_dir": dict(sorted(tops.items(), key=lambda kv: -kv[1]))}


def check_limits(summary: dict, limits: dict) -> None:
    if summary["bytes"] > int(limits.get("max_total_bytes", 1 << 62)) or summary["files"] > int(limits.get("max_files", 1 << 62)):
        raise SystemExit(f"束が上限を超えています（{summary['files']} ファイル / {summary['bytes']} bytes）。config/serving_state.json の limits か include を見直してください")


def build_archive(root: Path, files: list[Path], out: Path) -> str:
    """tar.gz を作って sha256 を返す。更新時刻は固定しない（内容が同じなら sha が同じになるよう mtime を 0 にそろえる）。"""
    with open(out, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w") as tar:   # gzip ヘッダの時刻も 0 にして、同じ内容なら同じ sha256 になるようにする
        for p in files:
            ti = tar.gettarinfo(str(p), arcname=p.relative_to(root).as_posix())
            ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname = 0, 0, 0, "", ""
            with open(p, "rb") as f:
                tar.addfile(ti, f)
    h = hashlib.sha256()
    with open(out, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    ok = []
    for m in tar.getmembers():
        pp = PurePosixPath(m.name)
        if not m.isfile() or pp.is_absolute() or ".." in pp.parts or not pp.parts or pp.parts[0] != ALLOWED_TOP:
            raise ValueError(f"展開できないパスが含まれています: {m.name}")
        ok.append(m)
    return ok


class Backend(Protocol):
    def put(self, name: str, src: Path, content_type: str) -> None: ...
    def put_bytes(self, name: str, data: bytes) -> None: ...
    def get_bytes(self, name: str) -> bytes | None: ...
    def get_file(self, name: str, dst: Path) -> bool: ...


class DirBackend:
    def __init__(self, d: Path) -> None:
        self.d = Path(d)

    def put(self, name: str, src: Path, content_type: str = "") -> None:
        dst = self.d / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)

    def put_bytes(self, name: str, data: bytes) -> None:
        dst = self.d / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)

    def get_bytes(self, name: str) -> bytes | None:
        p = self.d / name
        return p.read_bytes() if p.exists() else None

    def get_file(self, name: str, dst: Path) -> bool:
        p = self.d / name
        if not p.exists():
            return False
        shutil.copyfile(p, dst)
        return True


class GcsBackend:
    def __init__(self, bucket: str | None = None) -> None:
        from src.config.gcp_guard import assert_gcp_allowed

        assert_gcp_allowed("serving_state（GCS）")
        self.bucket_name = bucket or os.environ.get("GCS_BUCKET", "")
        if not self.bucket_name:
            raise SystemExit("GCS_BUCKET が未設定です")
        from google.cloud import storage

        self.bucket = storage.Client().bucket(self.bucket_name)

    def put(self, name: str, src: Path, content_type: str = "application/gzip") -> None:
        self.bucket.blob(name).upload_from_filename(str(src), content_type=content_type)

    def put_bytes(self, name: str, data: bytes) -> None:
        self.bucket.blob(name).upload_from_string(data, content_type="application/json")

    def get_bytes(self, name: str) -> bytes | None:
        from google.api_core.exceptions import NotFound

        try:
            return self.bucket.blob(name).download_as_bytes()
        except NotFound:
            return None

    def get_file(self, name: str, dst: Path) -> bool:
        from google.api_core.exceptions import NotFound

        try:
            self.bucket.blob(name).download_to_filename(str(dst))
            return True
        except NotFound:
            return False


def push(backend: Backend, cfg: dict, group: str, root: Path = ROOT) -> dict:
    g = cfg["groups"][group]
    files = select_files(root, g)
    summ = summarize(root, files)
    check_limits(summ, cfg.get("limits") or {})
    prefix = f"{cfg.get('object_prefix', 'serving_state')}/{group}"
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    with tempfile.TemporaryDirectory() as td:
        arc = Path(td) / "state.tar.gz"
        sha = build_archive(root, files, arc)
        name = f"{prefix}/{ts}.tar.gz"
        backend.put(name, arc, "application/gzip")
        info = {"object": name, "sha256": sha, "bytes": arc.stat().st_size, "files": summ["files"],
                "uncompressed_bytes": summ["bytes"], "created_at": ts}
        backend.put_bytes(f"{prefix}/latest.json", json.dumps(info).encode())  # 最後に書く（latest が先に新しくならない）
    return info


def pull(backend: Backend, cfg: dict, group: str, root: Path = ROOT) -> dict:
    """latest.json を見て、前回と同じ sha256 なら何もしない。sha256 不一致・不正パスは展開しない。"""
    prefix = f"{cfg.get('object_prefix', 'serving_state')}/{group}"
    raw = backend.get_bytes(f"{prefix}/latest.json")
    if raw is None:
        return {"group": group, "status": "no_state"}
    info = json.loads(raw)
    marker = root / "data" / "local" / "meta" / f"serving_state_{group}.json"
    if marker.exists():
        try:
            if json.loads(marker.read_text(encoding="utf-8")).get("sha256") == info["sha256"]:
                return {"group": group, "status": "unchanged", "sha256": info["sha256"]}
        except (OSError, ValueError):
            pass
    with tempfile.TemporaryDirectory() as td:
        arc = Path(td) / "state.tar.gz"
        if not backend.get_file(info["object"], arc):
            return {"group": group, "status": "missing_object", "object": info["object"]}
        h = hashlib.sha256()
        with open(arc, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != info["sha256"]:
            return {"group": group, "status": "sha_mismatch"}
        with tarfile.open(arc, "r:gz") as tar:
            members = safe_members(tar)
            for m in members:
                dst = root / m.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(m)
                assert src is not None
                tmp = dst.with_name(dst.name + ".part")
                with open(tmp, "wb") as out:
                    shutil.copyfileobj(src, out)
                os.replace(tmp, dst)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"sha256": info["sha256"], "created_at": info.get("created_at"), "pulled_at": time.time()}), encoding="utf-8")
    return {"group": group, "status": "pulled", "files": len(members), "sha256": info["sha256"]}


def make_backend(local_dir: str | None) -> Backend:
    return DirBackend(Path(local_dir)) if local_dir else GcsBackend()


def pull_default_groups(env: dict[str, str] | None = None) -> list[dict]:
    """Cloud Run 起動時の入口。失敗しても例外は投げず結果を返す（API の起動を妨げない）。"""
    e = os.environ if env is None else env
    cfg = load_config()
    groups = [g.strip() for g in (e.get("KEIBA_SERVING_STATE_GROUPS") or ",".join(cfg.get("default_groups") or ["core"])).split(",") if g.strip()]
    results = []
    try:
        backend = make_backend(e.get("KEIBA_SERVING_STATE_LOCAL_DIR"))
    except BaseException as exc:  # noqa: BLE001  SystemExit も含めて起動を止めない
        return [{"status": "backend_error", "error": f"{type(exc).__name__}: {exc}"[:200]}]
    for g in groups:
        try:
            results.append(pull(backend, cfg, g))
        except Exception as exc:  # noqa: BLE001
            logger.exception("serving_state の取り込みに失敗: %s", g)
            results.append({"group": g, "status": "error", "error": f"{type(exc).__name__}: {exc}"[:200]})
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("--group", action="append", help="既定は config の default_groups")
    ap.add_argument("--dry-run", action="store_true", help="push のみ: 入るファイルとサイズを表示して終わる（GCS に触れない）")
    ap.add_argument("--local-dir", help="GCS の代わりに使うディレクトリ（検証用）")
    a = ap.parse_args(argv)
    cfg = load_config()
    groups = a.group or cfg.get("default_groups") or ["core"]
    for g in groups:
        if g not in cfg["groups"]:
            print(f"未知のグループ: {g}", file=sys.stderr)
            return 2
    if a.action == "push" and a.dry_run:
        for g in groups:
            files = select_files(ROOT, cfg["groups"][g])
            s = summarize(ROOT, files)
            print(json.dumps({"group": g, **s}, ensure_ascii=False, indent=1))
        return 0
    backend = make_backend(a.local_dir)
    for g in groups:
        r = push(backend, cfg, g) if a.action == "push" else pull(backend, cfg, g)
        print(json.dumps({"group": g, **r}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
