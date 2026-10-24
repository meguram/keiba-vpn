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
TREES = (
    "data/local/inference/predictions_v3",
    "data/local/inference/predictions_v5",
    "data/local/inference/predictions_v6",
    "data/local/inference/predictions_backfill",
    "data/local/inference/snapshots",
    "data/local/inference/lap_forecast",
    "data/local/inference/tracking_v2",
    "data/calculated_data/tracking_difficulty::20*.json",     # 追走難度・想定ラップの事前計算（/api/race/{id}/tracking-difficulty）。1 レース約 35KB・1 日 36 件。過去分は --recent-days で段階的に
    "data/local/meta/entries_state",
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
    return [p for p in base.glob(g) if p.is_file() and not _SKIP.search(p.name)] if base.exists() else []


def push(backend: Backend, root: Path = REPO_ROOT, trees: tuple[str, ...] = TREES, *, recent_days: float | None = None,
         dry_run: bool = False, workers: int = 8) -> dict[str, int]:
    """内容が違う（または GCS に無い）ローカルのファイルだけを上げる。recent_days を付けると、その日数以内に更新されたファイルだけを見る。"""
    cutoff = time.time() - recent_days * 86400 if recent_days is not None else 0.0
    todo: list[tuple[str, bytes]] = []
    seen = 0
    for tree in trees:
        remote = backend.list(f"{PREFIX}/{split_tree(tree)[0]}/")
        for p in _files(root, tree):
            if p.stat().st_mtime < cutoff:
                continue
            seen += 1
            data = p.read_bytes()
            key = f"{PREFIX}/{p.relative_to(root).as_posix()}"
            if remote.get(key, ("",))[0] != _md5(data):
                todo.append((key, data))
    if not dry_run and todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(lambda kv: backend.upload(*kv), todo))
    return {"seen": seen, "uploaded": 0 if dry_run else len(todo), "would_upload": len(todo) if dry_run else 0}


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("--recent-days", type=float, default=None, help="直近この日数に更新されたものだけ（省略=全部）")
    ap.add_argument("--dry-run", action="store_true", help="push のみ: 上げるファイルの数を表示して終わる")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    be = GcsBackend()
    print(push(be, recent_days=a.recent_days, dry_run=a.dry_run) if a.action == "push" else pull(be, recent_days=a.recent_days))
    return 0


if __name__ == "__main__":
    sys.exit(main())
