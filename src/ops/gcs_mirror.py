"""ローカルの運用データ（予測ファイル・出馬表の状態）を GCS と同期する。ホストと切り離した公開（Cloud Run）のため。

  push（ホスト／Jobs）: ローカルのファイルのうち GCS と内容が違うものだけをアップロードする（cron で数分おき）。書き込み側のコードは変えない。
  pull（Cloud Run）   : 起動時に全部、以降は 1 分ごとに直近の分だけをローカルへ取り込む。読み込み側のコードは変えない（ローカルのファイルを読むまま）。

置き場: gs://$GCS_BUCKET/ops/mirror/<リポジトリルートからの相対パス>。差分の判定は MD5。小さいファイルが数千個（約 12MB）なので費用は最小課金の範囲。
予測成績の日次集計（prediction_results）は別（ops/prediction_results/、prediction_results.py が直接読み書きする）。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

logger = logging.getLogger("ops.gcs_mirror")

PREFIX = "ops/mirror"
# 会員名簿（ハッシュだけ。ログインの照合に Cloud Run が使う）。ホストでは root だけが読める場所にあるため、通常の同期(jovyan)は読めずに飛ばし、
# root の cron（scripts/cron/gcs_mirror_push_auth.sh）が上げる。Cloud Run は TREES として取り込み、src/api/auth.py の MEMBERS_LOCAL_FILE と同じ場所に置く。
AUTH_TREES = ("data/local/auth::members.json",)
TREES = (
    "data/local/inference/predictions_v3",
    "data/local/inference/predictions_v5",
    "data/local/inference/predictions_v6",
    "data/local/inference/predictions_backfill",
    "data/local/inference/snapshots",
    "data/local/inference/lap_forecast",
    "data/local/inference/tracking_v2",
    "data/local/meta/entries_state",
    *AUTH_TREES,
    # 画面が読む参照データ（取得で作る・更新が速い／ほぼ固定のもの。旧 serving_state の core 群）。"<フォルダ>::<glob>" はフォルダの中の一致するファイルだけ
    "data/page_reference/race_lists",
    "data/page_reference/race_day_schedule",
    "data/page_reference/meta",
    "data/page_reference/cushion",
    "data/page_reference/track_speed",
    "data/page_reference/knowledge",
    "data/page_reference/pedigree_map",
    "data/page_reference/note_aptitude_race_l3",
    "data/page_reference/bloodline_vector",
    "data/page_reference/pedigree_race_index",
    "data/page_reference/tables::*/megu_index_flat.parquet",
    "data/page_reference/tables::*/race_result_flat.parquet",
)
# GCS には全部置くが、Cloud Run の起動時には取り込まない（全体で約 1GB。メモリ上のファイルシステムに載せない）もの。
# 定期の取り込み(recent)にも含めない（一括アップロードで更新時刻が揃い、直近扱いで全部が降りてしまうため）。読まれたときに 1 件だけ取りに行く（fetch_one。一定時間より古ければ取り直す）。
LAZY_TREES = (
    "data/calculated_data/tracking_difficulty::20*.json",     # 追走難度・想定ラップの事前計算（/api/race/{id}/tracking-difficulty）。1 レース約 35KB・全体で約 1GB
)
PUSH_TREES = TREES + LAZY_TREES
_SKIP = re.compile(r"(\.tmp|\.lock)$")
REPO_ROOT = Path(__file__).resolve().parents[2]


class Backend(Protocol):
    def list(self, prefix: str) -> dict[str, tuple[str, float]]: ...      # key -> (md5 の base64, 更新時刻 epoch 秒)
    def download(self, key: str) -> bytes: ...
    def upload(self, key: str, data: bytes) -> None: ...


class GcsBackend:
    def __init__(self) -> None:
        from src.config.gcp_credentials import storage_client
        from src.config.gcp_guard import assert_gcp_allowed

        assert_gcp_allowed("運用データの GCS ミラー")
        self._client = storage_client()
        self._bucket = self._client.bucket(os.environ["GCS_BUCKET"])

    def list(self, prefix: str) -> dict[str, tuple[str, float]]:
        return {b.name: (b.md5_hash or "", b.updated.timestamp() if b.updated else 0.0)
                for b in self._client.list_blobs(self._bucket, prefix=prefix)}

    def download(self, key: str) -> bytes:
        return self._bucket.blob(key).download_as_bytes()

    def upload(self, key: str, data: bytes) -> None:
        self._bucket.blob(key).upload_from_string(data, content_type="application/octet-stream")


def _md5(data: bytes) -> str:
    return base64.b64encode(hashlib.md5(data).digest()).decode()


def split_tree(tree: str) -> tuple[str, str]:
    """``"dir"`` または ``"dir::glob"``（glob はフォルダからの相対）。"""
    d, _, g = tree.partition("::")
    return d, g or "**/*"


def _files(root: Path, tree: str) -> list[Path]:
    d, g = split_tree(tree)
    base = root / d
    try:
        return [p for p in base.glob(g) if p.is_file() and not _SKIP.search(p.name)] if base.exists() else []
    except PermissionError:          # root だけが読める場所（会員名簿）は、読める実行者の同期に任せて飛ばす
        return []


def push(backend: Backend, root: Path = REPO_ROOT, trees: tuple[str, ...] = PUSH_TREES, *, recent_days: float | None = None,
         dry_run: bool = False, workers: int = 8, batch_bytes: int = 64 * 1024 * 1024) -> dict[str, int]:
    """内容が違う（または GCS に無い）ローカルのファイルだけを上げる。recent_days を付けると、その日数以内に更新されたファイルだけを見る。
    全体が大きくても、メモリに溜めるのは batch_bytes まで（溜まったら上げて空にする）。"""
    cutoff = time.time() - recent_days * 86400 if recent_days is not None else 0.0
    todo: list[tuple[str, bytes]] = []
    seen = uploaded = would = 0
    held = 0

    def flush() -> None:
        nonlocal todo, held, uploaded
        if todo:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(lambda kv: backend.upload(*kv), todo))
            uploaded += len(todo)
        todo, held = [], 0

    for tree in trees:
        remote = backend.list(f"{PREFIX}/{split_tree(tree)[0]}/")
        for p in _files(root, tree):
            if p.stat().st_mtime < cutoff:
                continue
            seen += 1
            data = p.read_bytes()
            key = f"{PREFIX}/{p.relative_to(root).as_posix()}"
            if remote.get(key, ("",))[0] != _md5(data):
                if dry_run:
                    would += 1
                    continue
                todo.append((key, data)); held += len(data)
                if held >= batch_bytes:
                    flush()
    flush()
    return {"seen": seen, "uploaded": uploaded, "would_upload": would}


def pull(backend: Backend, root: Path = REPO_ROOT, trees: tuple[str, ...] = TREES, *, recent_days: float | None = None,
         workers: int = 16) -> dict[str, int]:
    """GCS のファイルのうち、ローカルに無い・内容が違うものをローカルへ取り込む。recent_days を付けると、GCS で直近に更新されたものだけを見る。"""
    cutoff = time.time() - recent_days * 86400 if recent_days is not None else 0.0
    todo: list[str] = []
    seen = 0
    for d in dict.fromkeys(split_tree(t)[0] for t in trees):      # tables のように 1 フォルダに複数の glob があっても、一覧は 1 回
        for key, (md5, updated) in backend.list(f"{PREFIX}/{d}/").items():
            if updated < cutoff:
                continue
            seen += 1
            local = root / key[len(PREFIX) + 1:]
            if not local.is_file() or _md5(local.read_bytes()) != md5:
                todo.append(key)

    def _get(key: str) -> None:
        local = root / key[len(PREFIX) + 1:]
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_name(local.name + ".tmp")
        tmp.write_bytes(backend.download(key))
        os.replace(tmp, local)

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_get, todo))
    return {"seen": seen, "downloaded": len(todo)}


_MISS: dict[str, float] = {}


def fetch_one(rel_path: str, root: Path = REPO_ROOT, backend: Backend | None = None, *, miss_ttl: float = 60.0, max_age: float | None = None) -> bool:
    """GCS のミラーから 1 ファイルだけ取りに行く（LAZY_TREES 用。起動時に取り込まない古いデータを、読まれたときに使う）。
    ミラーが無効（KEIBA_MIRROR_PULL が 1 でない）なら何もしない。GCS に無かった結果は miss_ttl 秒だけ覚えて、同じファイルを何度も問い合わせない。"""
    if backend is None and os.environ.get("KEIBA_MIRROR_PULL", "").strip() != "1":
        return False
    now = time.time()
    if now - _MISS.get(rel_path, 0.0) < miss_ttl:
        return False
    local = root / rel_path
    if max_age is not None and local.is_file() and now - local.stat().st_mtime < max_age:
        return False                                  # 取って間もない（max_age 秒以内）ので取り直さない
    try:
        be = backend or GcsBackend()
        data = be.download(f"{PREFIX}/{rel_path}")
    except Exception:  # noqa: BLE001  無い・通信失敗は『まだ無い』として扱う
        _MISS[rel_path] = now
        return False
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_name(local.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, local)
    return True


def start_refresher(backend: Backend | None = None, root: Path = REPO_ROOT, *, interval: float = 60.0, recent_days: float = 2.0) -> threading.Thread:
    """Cloud Run 用: 起動時に全部を取り込み、以降は interval 秒ごとに直近の更新分だけを取り込む常駐スレッド。失敗しても API は止めない。"""
    def run() -> None:
        be = backend
        first = True
        while True:
            try:
                be = be or GcsBackend()
                if first:     # 起動直後は直近分を先に（表示に要る分を数秒で）、続けて全体
                    logger.info("gcs_mirror pull(直近): %s", pull(be, root, recent_days=recent_days))
                    logger.info("gcs_mirror pull(全体): %s", pull(be, root))
                    first = False
                else:
                    r = pull(be, root, recent_days=recent_days)
                    if r["downloaded"]:
                        logger.info("gcs_mirror pull: %s", r)
            except Exception as exc:  # noqa: BLE001
                logger.warning("gcs_mirror pull に失敗（次回に再試行）: %s", exc)
            time.sleep(interval)

    t = threading.Thread(target=run, name="gcs-mirror-pull", daemon=True)
    t.start()
    return t


def pull_trees(trees: list[str] | tuple[str, ...], *, recent_days: float | None = 14.0, backend: Backend | None = None, root: Path = REPO_ROOT) -> dict[str, int]:
    """Jobs が読む参照データ（race_lists・race_day_schedule など）を ops/mirror からローカルへ取り込む。読み取りだけ。recent_days 以内に更新されたものだけ。"""
    return pull(backend or GcsBackend(), root, tuple(trees), recent_days=recent_days)


def tree_known(tree: str) -> bool:
    """job_map の mirror_out に書ける木か（PUSH_TREES のどれかの配下）。"""
    d = split_tree(tree)[0].rstrip("/")
    return any(d == split_tree(t)[0] or d.startswith(split_tree(t)[0] + "/") for t in PUSH_TREES)


def push_trees(trees: list[str] | tuple[str, ...], *, since_epoch: float | None = None, backend: Backend | None = None, root: Path = REPO_ROOT) -> dict[str, int]:
    """Jobs が作った出力（予測・ラップ予想など）を ops/mirror へ上げる。since_epoch 以降に更新されたファイルだけ。"""
    be = backend or GcsBackend()
    recent = (time.time() - since_epoch) / 86400 if since_epoch else None
    return push(be, root, tuple(trees), recent_days=recent)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("--recent-days", type=float, default=None, help="直近この日数に更新されたものだけ（省略=全部）")
    ap.add_argument("--dry-run", action="store_true", help="push のみ: 上げるファイルの数を表示して終わる")
    ap.add_argument("--only-auth", action="store_true", help="会員名簿(AUTH_TREES)だけを上げる（root の cron 用）")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    be = GcsBackend()
    trees = AUTH_TREES if a.only_auth else PUSH_TREES
    print(push(be, trees=trees, recent_days=a.recent_days, dry_run=a.dry_run) if a.action == "push" else pull(be, recent_days=a.recent_days))
    return 0


if __name__ == "__main__":
    sys.exit(main())
