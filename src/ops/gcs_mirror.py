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
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol

logger = logging.getLogger("ops.gcs_mirror")

PREFIX = "ops/mirror"
# 会員名簿（ハッシュだけ。ログインの照合に Cloud Run が使う）。ホストでは root だけが読める場所にあるため、通常の同期(jovyan)は読めずに飛ばし、
# root の cron（scripts/cron/gcs_mirror_push_auth.sh）が上げる。Cloud Run は TREES として取り込み、src/api/auth.py の MEMBERS_LOCAL_FILE と同じ場所に置く。
AUTH_TREES = ("data/local/auth::members.json",)
_FEATURES = "data/local/features"
TREES = (
    "data/local/inference/predictions_v3",
    "data/local/inference/predictions_v5",
    "data/local/inference/predictions_v6",
    "data/local/inference/predictions_backfill",
    "data/local/inference/snapshots",
    "data/local/inference/lap_forecast",
    "data/local/inference/tracking_v2",
    "data/local/race_analysis",        # めぐ分析の事前計算の結果（レースごと。画面はこれを読むだけ）
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
    # 出馬表の騎手・調教師カード（条件別の成績）の集計。毎朝の再生成（export_tables_daily の後）で更新され、5 分おきの push で Cloud Run に届く（約 560KB）
    f"{_FEATURES}/jockey_tbl::lookup_jockey*.parquet",
    f"{_FEATURES}/trainer_tbl::lookup_trainer*.parquet",
    f"{_FEATURES}/jockey_trainer_stats::_manifest.json",
    "data/page_reference/tables::*/megu_index_flat.parquet",
    "data/page_reference/tables::*/race_result_flat.parquet",
)
# GCS には全部置くが、Cloud Run の起動時には取り込まない（全体で約 1GB。メモリ上のファイルシステムに載せない）もの。
# 定期の取り込み(recent)にも含めない（一括アップロードで更新時刻が揃い、直近扱いで全部が降りてしまうため）。読まれたときに 1 件だけ取りに行く（fetch_one。一定時間より古ければ取り直す）。
LAZY_TREES = (
    "data/calculated_data/tracking_difficulty::20*.json",     # 追走難度・想定ラップの事前計算（/api/race/{id}/tracking-difficulty）。1 レース約 35KB・全体で約 1GB
)
PUSH_TREES = TREES + LAZY_TREES
# レースごとのファイルが日付フォルダに積み上がる木。起動時と定期の取り込みでは、日付が直近 PULL_WINDOW_DAYS 日以内のフォルダだけ降ろす
# （全体で 1 万件超 → 起動時の GCS 読み出しを約 1/10 に）。古い日は読まれたときに lazy_fetch で 1 件だけ取る。push は全体のまま。
WINDOWED_DIRS = tuple(t for t in TREES if t.rsplit("/", 1)[-1] in ("predictions_backfill", "lap_forecast", "tracking_v2"))
PULL_WINDOW_DAYS = 14.0
_DATE_SEG = re.compile(r"^(\d{4})-?(\d{2})-?(\d{2})$")
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

    def read_gen(self, key: str) -> tuple[bytes | None, int]:
        """中身と世代（無ければ (None, 0)）。目録の読み書きを、他の書き手と競合しても失わないために使う。"""
        from google.api_core.exceptions import NotFound

        b = self._bucket.blob(key)
        try:
            data = b.download_as_bytes()
        except NotFound:
            return None, 0
        return data, int(b.generation or 0)

    def upload_if(self, key: str, data: bytes, generation: int) -> bool:
        """世代が generation のときだけ上書きする（0 は「まだ無いときだけ」）。他が先に書いていたら False。"""
        from google.api_core.exceptions import PreconditionFailed

        try:
            self._bucket.blob(key).upload_from_string(data, content_type="application/json", if_generation_match=generation)
            return True
        except PreconditionFailed:
            return False


# ── 更新の目録（_index/） ────────────────────────────────────────────────
# 費用の上限（CLAUDE.md）: Cloud Run の API が毎分 24 か所を一覧すると、一覧（Class A）だけで 1 日 3〜7 万回・月 700〜1,600 円になる。
# そこで書き手（ホストの push・会員名簿の push・Jobs）が「直近に上げたファイルの目録」を 1 つずつ置き、API は毎分それを読むだけにする
# （1 回の読み出し。月 10 円未満）。目録が無い・古いとき、と FULL_LIST_SEC ごとには従来どおり一覧して取りこぼしを防ぐ。
INDEX_PREFIX = f"{PREFIX}/_index"
INDEX_WRITERS = ("host", "auth", "jobs")
INDEX_KEEP_SEC = 2 * 86400            # 目録に残す期間（API の定期取り込みの直近 2 日と同じ）
INDEX_STALE_SEC = 20 * 60             # ホストの目録がこれより古ければ（push が止まっている）一覧に戻す。push は 5 分おき
FULL_LIST_SEC = 30 * 60               # 目録を使っていても、この間隔で一覧して取りこぼし（目録を書かない書き手など）を拾う


def index_key(writer: str) -> str:
    return f"{INDEX_PREFIX}/{writer}.recent.json"


def _index_doc(writer: str, entries: dict[str, list]) -> bytes:
    return json.dumps({"version": 1, "writer": writer, "written_at": time.time(), "entries": entries},
                      ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _prune(entries: dict[str, list], now: float) -> dict[str, list]:
    return {k: v for k, v in entries.items() if float(v[1]) >= now - INDEX_KEEP_SEC}


def write_index_merge(backend: Backend, writer: str, new: dict[str, list], *, tries: int = 6) -> bool:
    """他の書き手（同時に動く Jobs など）と競合しても失わないよう、世代を確かめて目録へ追記する。世代を扱えない backend は上書き。"""
    now = time.time()
    if not hasattr(backend, "read_gen"):
        backend.upload(index_key(writer), _index_doc(writer, _prune(dict(new), now)))
        return True
    for _ in range(tries):
        raw, gen = backend.read_gen(index_key(writer))
        try:
            cur = (json.loads(raw).get("entries") or {}) if raw else {}
        except ValueError:
            cur = {}
        cur.update(new)
        if backend.upload_if(index_key(writer), _index_doc(writer, _prune(cur, now)), gen):
            return True
        time.sleep(0.2)
    return False


# ── 起動時の取り込みの束（_bundle/） ─────────────────────────────────────
# 費用の上限（CLAUDE.md）: API は起動のたびに参照データ約 6,000 ファイルを 1 つずつ読んでいた（1 回約 0.4 円）。アクセスが 15 分以上
# あくたびに起動し直すので、最悪 1 日 90 回・月 1,100 円になりうる。ホストが 1 日 1 回、起動時に要る分を 1 つの束にして置き、
# API は起動時にそれを 1 回読む（＋その後の更新は目録で）。束が無い・古い・壊れているときは従来どおり一覧で取り込む。
BUNDLE_KEY = f"{PREFIX}/_bundle/base.tar.gz"
BUNDLE_EVERY_SEC = 20 * 3600          # ホストが束を作り直す間隔
BUNDLE_MAX_AGE_SEC = 36 * 3600        # これより古い束は使わない（目録は直近 2 日分なので、その内側に収める）


BUNDLE_TREES = tuple(t for t in TREES if t not in AUTH_TREES)   # 会員名簿は束に入れない（従来どおり個別に取り込む）


def build_bundle(root: Path = REPO_ROOT, trees: tuple[str, ...] = BUNDLE_TREES, *, window: float | None = None, now: float | None = None) -> bytes:
    """起動時の取り込み（pull(window=...) と同じ範囲）のファイルを tar.gz に束ねる。先頭の _bundle.json に作った時刻と含む木。
    会員名簿（AUTH_TREES）は渡されても入れない。"""
    trees = tuple(t for t in trees if t not in AUTH_TREES)
    import io
    import tarfile

    now = time.time() if now is None else now
    window = window_days() if window is None else window
    oldest = date.fromtimestamp(now) - timedelta(days=window)
    buf = io.BytesIO()
    included: list[str] = []
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=6) as tf:
        files: list[tuple[str, Path]] = []
        for tree in trees:
            d = split_tree(tree)[0]
            got = _files(root, tree)
            if got:
                included.append(tree)
            for f in got:
                rel = f.relative_to(root).as_posix()
                if d in WINDOWED_DIRS:
                    kd = _date_in_path(rel[len(d) + 1:].split("/"))
                    if kd is not None and kd < oldest:
                        continue
                files.append((rel, f))
        meta = json.dumps({"built_at": now, "trees": included, "files": len(files)}).encode()
        ti = tarfile.TarInfo("_bundle.json"); ti.size = len(meta); ti.mtime = int(now)
        tf.addfile(ti, io.BytesIO(meta))
        for rel, f in files:
            tf.add(str(f), arcname=rel, recursive=False)
    return buf.getvalue()


def extract_bundle(data: bytes, root: Path = REPO_ROOT, trees: tuple[str, ...] = TREES, *, now: float | None = None) -> dict | None:
    """束を root に展開する。木の外のパス・絶対パス・.. を含むものは捨てる。古い・壊れている束は None（何も書かない）。"""
    import io
    import tarfile

    now = time.time() if now is None else now
    dirs = [split_tree(t)[0] for t in trees]
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            mf = tf.extractfile("_bundle.json")
            meta = json.loads(mf.read()) if mf else {}
            if now - float(meta.get("built_at") or 0) > BUNDLE_MAX_AGE_SEC:
                return None
            n = 0
            for m in tf.getmembers():
                name = m.name
                if not m.isfile() or name == "_bundle.json" or name.startswith("/") or ".." in name.split("/"):
                    continue
                if not any(name.startswith(d + "/") for d in dirs):
                    continue
                src = tf.extractfile(m)
                if src is None:
                    continue
                local = root / name
                local.parent.mkdir(parents=True, exist_ok=True)
                tmp = local.with_name(local.name + ".tmp")
                tmp.write_bytes(src.read())
                os.replace(tmp, local)
                n += 1
    except (tarfile.TarError, OSError, ValueError, KeyError):
        return None
    return {"files": n, "built_at": meta.get("built_at"), "trees": meta.get("trees") or []}


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


def _load_manifest(path: Path | None) -> dict[str, str] | None:
    doc = _read_manifest_doc(path)
    return None if doc is None else {str(k): str(v) for k, v in (doc.get("md5") or {}).items()}


def _read_manifest_doc(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001  壊れていたら一覧で作り直す
        return None


def _save_manifest(path: Path, known: dict[str, str], recent: dict[str, list] | None = None, bundle_at: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"version": 1, "md5": known, "recent": recent or {}, "bundle_at": bundle_at},
                              ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


def push(backend: Backend, root: Path = REPO_ROOT, trees: tuple[str, ...] = PUSH_TREES, *, recent_days: float | None = None,
         dry_run: bool = False, workers: int = 8, batch_bytes: int = 64 * 1024 * 1024,
         manifest: Path | None = None, verify_remote: bool = False, index_writer: str | None = None) -> dict[str, int]:
    """内容が違う（または GCS に無い）ローカルのファイルだけを上げる。recent_days を付けると、その日数以内に更新されたファイルだけを見る。
    全体が大きくても、メモリに溜めるのは batch_bytes まで（溜まったら上げて空にする）。

    manifest（ホストの定期 push 用）: 上げた内容の MD5 をローカルに記録し、次回からは GCS を一覧せずに記録と比べる。
    GCS の一覧は 1 回で約 4 万件・約 40MB の応答になり、GCP の外から 5 分おきに取ると転送料だけで月 6,000 円を超えるため
    （CLAUDE.md の「費用の上限」）。記録が無い・壊れている・verify_remote のときだけ一覧し、その結果で記録を作り直す。

    index_writer: 上げたファイルを更新の目録（_index/<writer>.recent.json）に載せる。API は毎分これを読むだけで更新を知る。
    manifest があれば目録は手元の記録から丸ごと書き直し（毎回。書いた時刻が「push が生きている」印になる）、無ければ世代を確かめて追記する。"""
    now = time.time()
    cutoff = now - recent_days * 86400 if recent_days is not None else 0.0
    mdoc = None if verify_remote else _read_manifest_doc(manifest)
    known = None if mdoc is None else {str(k): str(v) for k, v in (mdoc.get("md5") or {}).items()}
    recent: dict[str, list] = dict((mdoc or {}).get("recent") or {})
    use_list = known is None
    if known is None:
        known = {}
    todo: list[tuple[str, bytes]] = []
    seen = uploaded = would = 0
    held = 0
    sent: list[str] = []

    def flush() -> None:
        nonlocal todo, held, uploaded
        if todo:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(lambda kv: backend.upload(*kv), todo))
            uploaded += len(todo)
            sent.extend(k[len(PREFIX) + 1:] for k, _ in todo)
            t = time.time()
            for k, data in todo:
                known[k] = _md5(data)
                recent[k] = [known[k], t]
        todo, held = [], 0

    listed: set[str] = set()
    for tree in trees:
        d = split_tree(tree)[0]
        if use_list and d not in listed:              # 同じフォルダに複数の glob があっても一覧は 1 回
            listed.add(d)
            for k, (md5, upd) in backend.list(f"{PREFIX}/{d}/").items():
                known[k] = md5
                if upd >= now - INDEX_KEEP_SEC:
                    recent[k] = [md5, upd]
        for p in _files(root, tree):
            if p.stat().st_mtime < cutoff:
                continue
            seen += 1
            data = p.read_bytes()
            key = f"{PREFIX}/{p.relative_to(root).as_posix()}"
            if known.get(key, "") != _md5(data):
                if dry_run:
                    would += 1
                    continue
                todo.append((key, data)); held += len(data)
                if held >= batch_bytes:
                    flush()
    flush()
    recent = _prune(recent, time.time())
    bundle_at = float((mdoc or {}).get("bundle_at") or 0.0)
    if index_writer == "host" and manifest is not None and not dry_run and time.time() - bundle_at >= BUNDLE_EVERY_SEC:
        try:      # 1 日 1 回、API の起動時の取り込みを 1 回の読み出しで済ませる束を置く
            backend.upload(BUNDLE_KEY, build_bundle(root))
            bundle_at = time.time()
        except Exception:  # noqa: BLE001  束は補助（無ければ API は従来どおり一覧で取り込む）
            logger.warning("起動時の取り込みの束を置けませんでした", exc_info=True)
    if manifest is not None and not dry_run:
        _save_manifest(manifest, known, recent, bundle_at)
    if index_writer and not dry_run:
        try:
            if manifest is not None:
                backend.upload(index_key(index_writer), _index_doc(index_writer, recent))
            elif sent:
                write_index_merge(backend, index_writer, {f"{PREFIX}/{k}": recent[f"{PREFIX}/{k}"] for k in sent})
        except Exception:  # noqa: BLE001  目録は補助（無ければ API は一覧に戻る）
            logger.warning("更新の目録を書けませんでした: %s", index_writer, exc_info=True)
    if sent:
        try:      # 画面に出る応答の元（予測・ラップ・分析など）が変わったことを、VPS の事前ロード係へ目印で伝える（最善努力）
            from src.utils import cache_marker

            cache_marker.mark_from_paths(sent)
        except Exception:  # noqa: BLE001
            pass
    return {"seen": seen, "uploaded": uploaded, "would_upload": would}


def window_days() -> float:
    try:
        return float(os.environ.get("KEIBA_MIRROR_WINDOW_DAYS") or PULL_WINDOW_DAYS)
    except ValueError:
        return PULL_WINDOW_DAYS


def _date_in_path(parts: list[str]) -> date | None:
    for seg in parts:
        m = _DATE_SEG.match(seg)
        if m:
            try:
                return date(int(m[1]), int(m[2]), int(m[3]))
            except ValueError:
                return None
    return None


def outside_window(day: str | date, window: float | None = None) -> bool:
    """日付（YYYYMMDD / YYYY-MM-DD）が、取り込み窓（今日から window 日前まで。未来は窓内）より古いか。読めない日付は窓内扱い。"""
    d = day if isinstance(day, date) else _date_in_path([str(day)[:10]])
    if d is None:
        return False
    return d < date.today() - timedelta(days=window_days() if window is None else window)


def lazy_fetch(path: Path, day: str | date) -> bool:
    """窓より古い日のファイルがローカルに無いとき、GCS から 1 件だけ取る。ファイルがローカルにある状態になれば True。
    窓の内側（定期の取り込みが降ろす）・ミラー無効・リポジトリ外のパスでは何もしない。"""
    if path.is_file():
        return True
    try:
        if not outside_window(day):
            return False
        return fetch_one(path.resolve().relative_to(REPO_ROOT).as_posix())
    except Exception:  # noqa: BLE001  読み出し側を止めない（リポジトリ外のパス・通信失敗は『まだ無い』）
        return False


_DOWNLOAD_HOOKS: list = []


def on_download(fn) -> None:
    """pull が 1 件以上降ろした後に fn(降ろしたキーの一覧) を呼ぶ。API 側のメモリキャッシュを、ファイルが入れ替わった時点で無効にするために使う。"""
    if fn not in _DOWNLOAD_HOOKS:
        _DOWNLOAD_HOOKS.append(fn)


def pull(backend: Backend, root: Path = REPO_ROOT, trees: tuple[str, ...] = TREES, *, recent_days: float | None = None,
         workers: int = 16, window: float | None = None) -> dict[str, int]:
    """GCS のファイルのうち、ローカルに無い・内容が違うものをローカルへ取り込む。recent_days を付けると、GCS で直近に更新されたものだけを見る。
    window を付けると、WINDOWED_DIRS の木は日付フォルダが今日から window 日より前のものを降ろさない（読まれたときに lazy_fetch）。"""
    cutoff = time.time() - recent_days * 86400 if recent_days is not None else 0.0
    todo: list[str] = []
    seen = 0
    oldest = date.today() - timedelta(days=window) if window is not None else None
    for d in dict.fromkeys(split_tree(t)[0] for t in trees):      # tables のように 1 フォルダに複数の glob があっても、一覧は 1 回
        windowed = oldest is not None and d in WINDOWED_DIRS
        for key, (md5, updated) in backend.list(f"{PREFIX}/{d}/").items():
            if updated < cutoff:
                continue
            if windowed:
                kd = _date_in_path(key[len(PREFIX) + len(d) + 2:].split("/"))
                if kd is not None and kd < oldest:
                    continue
            seen += 1
            local = root / key[len(PREFIX) + 1:]
            if not local.is_file() or _md5(local.read_bytes()) != md5:
                todo.append(key)

    _download_all(backend, root, todo, workers)
    return {"seen": seen, "downloaded": len(todo)}


def pull_via_index(backend: Backend, root: Path = REPO_ROOT, trees: tuple[str, ...] = TREES, *, recent_days: float = 2.0,
                   window: float | None = None, workers: int = 16, now: float | None = None) -> dict[str, int] | None:
    """更新の目録（書き手ごとに 1 つ）だけを読んで、直近 recent_days 日に上がったファイルのうちローカルと違うものを取り込む。
    一覧（Class A）を使わない。ホストの目録が無い・読めない・古い（push が止まっている）ときは None（呼び出し側が一覧に戻る）。"""
    now = time.time() if now is None else now
    entries: dict[str, list] = {}
    host_ok = False
    for w in INDEX_WRITERS:
        try:
            raw = backend.download(index_key(w))
            doc = json.loads(raw)
        except Exception:  # noqa: BLE001  無い書き手は飛ばす
            continue
        if w == "host":
            host_ok = now - float(doc.get("written_at") or 0) <= INDEX_STALE_SEC
        for k, v in (doc.get("entries") or {}).items():
            if k not in entries or float(v[1]) > float(entries[k][1]):
                entries[k] = v
    if not host_ok:
        return None
    dirs = list(dict.fromkeys(split_tree(t)[0] for t in trees))
    cutoff = now - recent_days * 86400
    oldest = date.today() - timedelta(days=window) if window is not None else None
    todo: list[str] = []
    seen = 0
    for key, (md5, upd) in entries.items():
        if float(upd) < cutoff:
            continue
        rel = key[len(PREFIX) + 1:]
        d = next((x for x in dirs if rel.startswith(x + "/")), None)
        if d is None:
            continue
        if oldest is not None and d in WINDOWED_DIRS:
            kd = _date_in_path(rel[len(d) + 1:].split("/"))
            if kd is not None and kd < oldest:
                continue
        seen += 1
        local = root / rel
        if not local.is_file() or _md5(local.read_bytes()) != md5:
            todo.append(key)
    _download_all(backend, root, todo, workers)
    return {"seen": seen, "downloaded": len(todo), "via": "index"}


def _download_all(backend: Backend, root: Path, todo: list[str], workers: int) -> None:
    def _get(key: str) -> None:
        local = root / key[len(PREFIX) + 1:]
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_name(local.name + ".tmp")
        tmp.write_bytes(backend.download(key))
        os.replace(tmp, local)

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_get, todo))
        for hook in list(_DOWNLOAD_HOOKS):
            try:
                hook(todo)
            except Exception:  # noqa: BLE001  フックの失敗で取り込みを止めない
                logger.warning("gcs_mirror の取り込み後フックが失敗", exc_info=True)


class Refresh:
    """定期の取り込み 1 回分: ふだんは目録だけを読み（pull_via_index）、目録が使えないときと FULL_LIST_SEC ごとに一覧（pull）。"""

    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._last_full = clock()           # 起動時の取り込み（一覧）の直後から数える

    def __call__(self, backend: Backend, root: Path, *, recent_days: float, window: float | None) -> dict[str, int]:
        if self._clock() - self._last_full < FULL_LIST_SEC:
            r = pull_via_index(backend, root, recent_days=recent_days, window=window)
            if r is not None:
                return r
        self._last_full = self._clock()
        return {**pull(backend, root, recent_days=recent_days, window=window), "via": "list"}


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


initial_recent_pulled = threading.Event()      # 起動直後の「直近分」の取り込みが終わった（成否を問わず）。Cloud Run の起動時ウォームアップが待つ
initial_full_pulled = threading.Event()        # 起動直後の「全体」の取り込みまで終わった（成否を問わず）。リクエスト課金では、起動の完了前に済ませる


def _initial_from_bundle(be: Backend, root: Path, recent_days: float) -> dict | None:
    """束から起動時の取り込みを済ませる（読み出し 1 回＋目録）。束に無い木（会員名簿など）は一覧で取る。使えなければ None。"""
    try:
        got = extract_bundle(be.download(BUNDLE_KEY), root)
    except Exception:  # noqa: BLE001  束が無い・読めない
        return None
    if got is None:
        return None
    try:
        after = pull_via_index(be, root, recent_days=recent_days, window=window_days())
        if after is None:                 # ホストの目録が使えない → 直近分は一覧で
            after = pull(be, root, recent_days=recent_days, window=window_days())
        missing = tuple(t for t in TREES if t not in set(got["trees"]))
        rest = pull(be, root, missing, window=window_days()) if missing else {"downloaded": 0}
    except Exception:  # noqa: BLE001  束の後の追いつきに失敗 → 呼び出し側が従来の一覧で取り直す
        logger.warning("束の後の取り込みに失敗。一覧で取り直します", exc_info=True)
        return None
    return {"bundle_files": got["files"], "recent": after, "not_in_bundle": rest}


def _initial_pull(be: Backend, root: Path, recent_days: float) -> None:
    """起動直後は束（あれば）、無ければ直近分を先に（表示に要る分を数秒で）続けて全体。失敗しても API は止めない。"""
    try:
        b = _initial_from_bundle(be, root, recent_days)
        if b is not None:
            initial_recent_pulled.set()
            logger.info("gcs_mirror pull(束): %s", b)
            return
        try:
            logger.info("gcs_mirror pull(直近): %s", pull(be, root, recent_days=recent_days, window=window_days()))
        finally:
            initial_recent_pulled.set()
        logger.info("gcs_mirror pull(全体): %s", pull(be, root, window=window_days()))
    finally:
        initial_full_pulled.set()


def start_refresher(backend: Backend | None = None, root: Path = REPO_ROOT, *, interval: float = 60.0, recent_days: float = 2.0) -> threading.Thread:
    """Cloud Run 用（CPU 常時割り当て）: 起動時に全部を取り込み、以降は interval 秒ごとに直近の更新分だけを取り込む常駐スレッド。失敗しても API は止めない。
    リクエスト課金（CPU はリクエスト中だけ）では、リクエストの合間にスレッドが止まるので ``start_initial`` と ``RequestRefresher`` を使う。"""
    refresh = Refresh()

    def run() -> None:
        be = backend
        first = True
        while True:
            try:
                be = be or GcsBackend()
                if first:
                    _initial_pull(be, root, recent_days)
                    first = False
                else:
                    r = refresh(be, root, recent_days=recent_days, window=window_days())
                    if r["downloaded"]:
                        logger.info("gcs_mirror pull: %s", r)
            except Exception as exc:  # noqa: BLE001
                logger.warning("gcs_mirror pull に失敗（次回に再試行）: %s", exc)
            time.sleep(interval)

    t = threading.Thread(target=run, name="gcs-mirror-pull", daemon=True)
    t.start()
    return t


def start_initial(backend: Backend | None = None, root: Path = REPO_ROOT, *, recent_days: float = 2.0) -> threading.Thread:
    """リクエスト課金（CPU はリクエスト中だけ）用: 起動時に直近分 → 全体を 1 回だけ取り込む。起動処理（CPU が常にある間）が ``initial_full_pulled`` を待つ。
    以降の更新は ``RequestRefresher`` がリクエストをきっかけに取り込む。"""
    def run() -> None:
        try:
            _initial_pull(backend or GcsBackend(), root, recent_days)
        except Exception as exc:  # noqa: BLE001
            logger.warning("gcs_mirror の起動時の取り込みに失敗: %s", exc)
            initial_recent_pulled.set()
            initial_full_pulled.set()

    t = threading.Thread(target=run, name="gcs-mirror-initial", daemon=True)
    t.start()
    return t


class RequestRefresher:
    """リクエスト課金向けの更新: CPU はリクエスト中だけ割り当てられ、常駐スレッドは合間に止まる。そこで『前回から interval 秒以上経っていれば、
    リクエストをきっかけに直近の更新分を取り込む』。最後の成功から ``wait_gap`` 秒以上空いていたら（長く止まっていた後の最初のリクエスト）は、
    古い表示を避けるため最大 ``wait_s`` 秒だけ取り込みの完了を待つ。それ以外は待たずに始めるだけ（取り込みはほかのリクエストの間に進む）。"""

    def __init__(self, backend: Backend | None = None, root: Path = REPO_ROOT, *, interval: float = 60.0, recent_days: float = 2.0,
                 wait_gap: float = 300.0, wait_s: float = 8.0, clock=time.monotonic) -> None:
        self.backend, self.root, self.interval, self.recent_days = backend, root, interval, recent_days
        self.wait_gap, self.wait_s, self._clock = wait_gap, wait_s, clock
        self._lock = threading.Lock()
        self._running = False
        self._last_start = clock()      # 起動時の取り込み直後から数える
        self._last_ok = clock()
        self._refresh = Refresh(clock)

    def should_wait(self) -> bool:
        return self._clock() - self._last_ok >= self.wait_gap

    def start_if_due(self) -> threading.Thread | None:
        """取り込みが必要なら別スレッドで始めて、そのスレッドを返す（不要・実行中なら None）。"""
        with self._lock:
            now = self._clock()
            if self._running or now - self._last_start < self.interval:
                return None
            self._running, self._last_start = True, now

        def run() -> None:
            try:
                be = self.backend = self.backend or GcsBackend()
                r = self._refresh(be, self.root, recent_days=self.recent_days, window=window_days())
                self._last_ok = self._clock()
                if r["downloaded"]:
                    logger.info("gcs_mirror pull: %s", r)
            except Exception as exc:  # noqa: BLE001
                logger.warning("gcs_mirror pull に失敗（次回に再試行）: %s", exc)
            finally:
                with self._lock:
                    self._running = False

        t = threading.Thread(target=run, name="gcs-mirror-request", daemon=True)
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
    return push(be, root, tuple(trees), recent_days=recent, index_writer="jobs")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("--recent-days", type=float, default=None, help="直近この日数に更新されたものだけ（省略=全部）")
    ap.add_argument("--dry-run", action="store_true", help="push のみ: 上げるファイルの数を表示して終わる")
    ap.add_argument("--only-auth", action="store_true", help="会員名簿(AUTH_TREES)だけを上げる（root の cron 用）")
    ap.add_argument("--manifest", default=None,
                    help="push のみ: 上げた内容の記録（既定 data/local/meta/gcs_mirror_push_manifest*.json）。'none' で記録を使わず毎回 GCS を一覧する")
    ap.add_argument("--verify-remote", action="store_true", help="push のみ: 記録を使わず GCS を一覧して比べ、記録を作り直す")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    be = GcsBackend()
    trees = AUTH_TREES if a.only_auth else PUSH_TREES
    if a.manifest == "none":
        manifest = None
    else:
        manifest = Path(a.manifest) if a.manifest else REPO_ROOT / "data/local/meta" / (
            "gcs_mirror_push_manifest_auth.json" if a.only_auth else "gcs_mirror_push_manifest.json")
    writer = "auth" if a.only_auth else "host"
    print(push(be, trees=trees, recent_days=a.recent_days, dry_run=a.dry_run, manifest=manifest, verify_remote=a.verify_remote,
               index_writer=None if manifest is None else writer)
          if a.action == "push" else pull(be, recent_days=a.recent_days))
    return 0


if __name__ == "__main__":
    sys.exit(main())
