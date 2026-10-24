#!/usr/bin/env python3
"""pull 型デプロイ（VPS 側）。GCS の latest.json を見て、新しい版があればメンテナンス窓（JST 03:00〜04:00）に取りに行って入れ替える。
root の systemd タイマーから実行する（SSH を閉じても配布できる）。ホストの scripts/ops/vps/publish_release.sh が latest.json を書く＝昇格。

  pull_release.py --kind ui|ui-stg|fetch [--force] [--dry-run] [--window 03:00-04:00]

- ui   : tar.gz と .sha256 を /srv/keiba-ui/incoming に置き、既存の /opt/keiba-ui/bin/install_release.sh に任せる
         （sha256 検証→展開→別ポートで起動テスト→切替→確認→失敗なら自動で直前の版へ）。
- fetch: 取得の実行役のコードバンドル。/opt/keiba-fetch/releases/<id> に展開→selfcheck→current を切替→実行役を再起動→失敗なら戻す。
- 前に進むだけ: latest の日時（ID 末尾 12 桁）が現在より新しいときだけ入れる。戻すのは rollback.sh（手動）。SSH 経由の promote.sh と共存できる。
- 同じ版の失敗を繰り返さない（state の failed_ids。--force で解除）。結果は GCS の ops/releases/<kind>/applied/vps.json に報告（VPS は Slack の秘密を持たない）。
環境: GCS_BUCKET、GOOGLE_APPLICATION_CREDENTIALS（EnvironmentFile=/etc/keiba-pull/pull.env。値は表示しない）。標準ライブラリ + google-cloud-storage。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")
PREFIX = "ops/releases"
_ID = re.compile(r"^[A-Za-z0-9._-]{1,120}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_TS = re.compile(r"-(\d{12})$")
STATE_DIR = Path(os.environ.get("KEIBA_PULL_STATE", "/var/lib/keiba-pull"))
UI_ROOT = Path(os.environ.get("APP_ROOT", "/srv/keiba-ui"))
UI_STG_ROOT = Path(os.environ.get("APP_ROOT_STG", "/srv/keiba-ui-stg"))     # stg 環境の画面（2 つ目。INSTANCE=stg で導入）
FETCH_ROOT = Path(os.environ.get("KEIBA_FETCH_ROOT", "/opt/keiba-fetch"))
INSTALL_UI = "/opt/keiba-ui/bin/install_release.sh"


class PullError(RuntimeError):
    pass


def release_ts(release_id: str) -> str | None:
    m = _TS.search(release_id or "")
    return m.group(1) if m else None


def is_newer(latest_id: str, current_id: str | None) -> bool:
    lt = release_ts(latest_id)
    if lt is None:
        return False
    ct = release_ts(current_id or "")
    return True if ct is None else lt > ct


def in_window(now: datetime, window: str = "03:00-04:00") -> bool:
    a, b = window.split("-")
    cur = now.astimezone(JST).strftime("%H:%M")
    return a <= cur < b


def validate_latest(kind: str, d: object) -> dict:
    if not isinstance(d, dict):
        raise PullError("latest.json が JSON オブジェクトではありません")
    rid, sha, tb = str(d.get("release_id", "")), str(d.get("sha256", "")), str(d.get("tarball", ""))
    if d.get("kind") != kind or not _ID.match(rid) or not _SHA.match(sha):
        raise PullError("latest.json の kind / release_id / sha256 が不正です")
    if tb != f"{PREFIX}/{kind}/{rid}.tar.gz":
        raise PullError("latest.json の tarball のパスが想定外です")
    return d


def decide(kind: str, latest: dict, current_id: str | None, now: datetime, *, force: bool, window: str, failed_ids: list[str]) -> str:
    """'install' | 'up_to_date' | 'older' | 'waiting_for_window' | 'previously_failed'（純粋関数）。"""
    rid = latest["release_id"]
    if rid == current_id:
        return "up_to_date"
    if not is_newer(rid, current_id):
        return "older"
    if rid in failed_ids and not force:
        return "previously_failed"
    if not force and not in_window(now, window):
        return "waiting_for_window"
    return "install"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def safe_extract(tar_path: Path, dest: Path) -> None:
    """パスの抜け出し（../・絶対パス・リンク）を拒否して展開する。"""
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf.getmembers():
            p = Path(m.name)
            if p.is_absolute() or ".." in p.parts or m.issym() or m.islnk() or m.isdev():
                raise PullError(f"tar に不正なエントリ: {m.name}")
        tf.extractall(dest, filter="data") if sys.version_info >= (3, 12) else tf.extractall(dest)


# --- GCS ---------------------------------------------------------------------------------------
class Gcs:
    def __init__(self, bucket: str) -> None:
        from google.cloud import storage

        self.b = storage.Client().bucket(bucket)

    def get_json(self, key: str):
        blob = self.b.get_blob(key)
        return None if blob is None else json.loads(blob.download_as_text())

    def put_json(self, key: str, obj: object) -> None:
        self.b.blob(key).upload_from_string(json.dumps(obj, ensure_ascii=False), content_type="application/json")

    def download(self, key: str, path: Path) -> None:
        self.b.blob(key).download_to_filename(str(path))


# --- 状態 --------------------------------------------------------------------------------------
def load_state(kind: str) -> dict:
    try:
        return json.loads((STATE_DIR / f"{kind}.json").read_text())
    except (OSError, ValueError):
        return {"failed_ids": []}


def save_state(kind: str, st: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / f"{kind}.json").write_text(json.dumps(st, ensure_ascii=False))


def current_id(kind: str) -> str | None:
    link = {"ui": UI_ROOT, "ui-stg": UI_STG_ROOT}.get(kind, FETCH_ROOT) / "current"
    return os.path.basename(os.path.realpath(link)) if link.is_symlink() else None


# --- 導入 --------------------------------------------------------------------------------------
def install_ui(g: Gcs, latest: dict, instance: str = "") -> None:
    rid = latest["release_id"]
    root = UI_STG_ROOT if instance == "stg" else UI_ROOT
    inc = root / "incoming"
    part, final = inc / f"{rid}.tar.gz.partial", inc / f"{rid}.tar.gz"
    g.download(latest["tarball"], part)
    if sha256_file(part) != latest["sha256"]:
        part.unlink(missing_ok=True)
        raise PullError("ダウンロードした tar.gz の sha256 が latest.json と一致しません")
    os.replace(part, final)
    Path(str(final) + ".sha256").write_text(f"{latest['sha256']}  {final.name}\n")
    shutil.chown(final, "deploy", "keibaui")
    shutil.chown(str(final) + ".sha256", "deploy", "keibaui")
    env = dict(os.environ)
    for k in ("APP_ROOT", "APP_PORT", "PROXY_PORT", "INSTANCE"):       # 本番の値を stg の導入に持ち込まない（置き場・ポートは vars-stg.env が決める）
        env.pop(k, None)
    if instance:
        env["INSTANCE"] = instance
    r = subprocess.run([INSTALL_UI, str(final)], capture_output=True, text=True, timeout=900, env=env)
    if r.returncode != 0:
        raise PullError(f"install_release.sh が終了コード {r.returncode}（自動で直前の版に戻っているはず）: {(r.stdout + r.stderr)[-400:]}")


def install_fetch(g: Gcs, latest: dict) -> None:
    rid = latest["release_id"]
    rel = FETCH_ROOT / "releases" / rid
    tmp = FETCH_ROOT / "incoming" / f"{rid}.tar.gz"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    g.download(latest["tarball"], tmp)
    if sha256_file(tmp) != latest["sha256"]:
        tmp.unlink(missing_ok=True)
        raise PullError("ダウンロードした tar.gz の sha256 が latest.json と一致しません")
    part = rel.with_name(rid + ".partial")
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    safe_extract(tmp, part)
    state = Path("/var/lib/keiba-fetch")
    for name in ("data", "logs"):                       # 版をまたいで残す領域は /var/lib/keiba-fetch に置く
        (state / name).mkdir(parents=True, exist_ok=True)
        shutil.rmtree(part / name, ignore_errors=True)
        os.symlink(state / name, part / name)
    shutil.chown(state, "keibafetch", "keibafetch")
    for name in ("data", "logs"):
        shutil.chown(state / name, "keibafetch", "keibafetch")
    py = FETCH_ROOT / "venv" / "bin" / "python"
    r = subprocess.run([str(py), "-m", "src.ops.fetch_child", "selfcheck"], cwd=part, capture_output=True, text=True, timeout=120,
                       env={**os.environ, "PYTHONPATH": str(part), "KEIBA_ENV": "prod"})
    if r.returncode != 0:
        shutil.rmtree(part, ignore_errors=True)
        raise PullError(f"selfcheck に失敗（切り替えません）: {r.stdout[-300:]}")
    os.replace(part, rel)
    cur = FETCH_ROOT / "current"
    prev = os.path.realpath(cur) if cur.is_symlink() else None
    tmp_link = FETCH_ROOT / "current.tmp"
    tmp_link.unlink(missing_ok=True)
    os.symlink(rel, tmp_link)
    os.replace(tmp_link, cur)
    was_active = subprocess.run(["systemctl", "is-active", "--quiet", "keiba-fetch-agent"]).returncode == 0
    if was_active:
        subprocess.run(["systemctl", "restart", "keiba-fetch-agent"], check=False)
        time.sleep(5)
        if subprocess.run(["systemctl", "is-active", "--quiet", "keiba-fetch-agent"]).returncode != 0 and prev:
            tmp_link.unlink(missing_ok=True)
            os.symlink(prev, tmp_link)
            os.replace(tmp_link, cur)
            subprocess.run(["systemctl", "restart", "keiba-fetch-agent"], check=False)
            raise PullError("切替後に実行役が起動しなかったため直前の版へ戻しました")
    releases = sorted((FETCH_ROOT / "releases").iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in releases[5:]:
        if str(old) != os.path.realpath(cur) and str(old) != prev:
            shutil.rmtree(old, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="pull 型デプロイ（VPS）")
    ap.add_argument("--kind", choices=("ui", "ui-stg", "fetch"), required=True)
    ap.add_argument("--force", action="store_true", help="窓の外でも・前回失敗した版でも実行")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--window", default="03:00-04:00")
    a = ap.parse_args(argv)
    bucket = os.environ.get("GCS_BUCKET", "")
    if not bucket:
        print("GCS_BUCKET が未設定です", file=sys.stderr)
        return 2
    g = Gcs(bucket)
    try:
        latest = validate_latest(a.kind, g.get_json(f"{PREFIX}/{a.kind}/latest.json"))
    except PullError as e:
        print(f"latest.json: {e}", file=sys.stderr)
        return 1
    st = load_state(a.kind)
    now = datetime.now(JST)
    cur = current_id(a.kind)
    window = "00:00-24:00" if a.kind == "ui-stg" else a.window       # stg は窓を待たない（確認用の環境なので、公開されたらすぐ入れる）
    verdict = decide(a.kind, latest, cur, now, force=a.force, window=window, failed_ids=st.get("failed_ids", []))
    print(f"kind={a.kind} latest={latest['release_id']} current={cur} -> {verdict}")
    if verdict != "install" or a.dry_run:
        return 0
    result, detail = "ok", ""
    try:
        (install_fetch(g, latest) if a.kind == "fetch" else install_ui(g, latest, "stg" if a.kind == "ui-stg" else ""))
    except Exception as e:  # noqa: BLE001
        result, detail = "failed", f"{type(e).__name__}: {e}"[:500]
        st["failed_ids"] = (st.get("failed_ids", []) + [latest["release_id"]])[-20:]
    st.update({"last_attempt": now.isoformat(timespec="seconds"), "last_release": latest["release_id"], "last_result": result})
    save_state(a.kind, st)
    try:
        g.put_json(f"{PREFIX}/{a.kind}/applied/vps.json", {"release_id": latest["release_id"], "result": result, "detail": detail,
                                                          "at": now.isoformat(timespec="seconds"), "previous": cur})
    except Exception as e:  # noqa: BLE001
        print(f"適用報告の書き込みに失敗: {type(e).__name__}", file=sys.stderr)
    if result != "ok":
        print(detail, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
