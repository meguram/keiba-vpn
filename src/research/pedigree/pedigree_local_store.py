"""
5世代血統 JSON のローカル集約（GCS read 負荷をホットパスで抑える）。

方針
----
- **本番・特徴計算・類似度バッチ**では ``storage.list_keys("horse_pedigree_5gen")`` や
  馬ごとの ``storage.load`` を繰り返さない。
- 既存の GCS スナップショット ``chuou/data/snapshots/sire_factor/horse_pedigree_5gen.jsonl.gz``
  を **1 blob だけ**取得し、ローカル ``.jsonl.gz`` に保存して以降はディスクのみ読む。
- メモリ上は ``horse_id -> レコード`` の辞書を **プロセス内で共有**（TTL 付き）。
- 母系を深く見る Phase1/2 では、祖先に現れる ``horse_id`` を BFS でたどるが、
  **同一 ID の取得は辞書ルックアップ 1 回**に集約される（重複祖先・インブリードで効く）。

環境変数
--------
- ``KEIBA_PEDIGREE_JSONL_GZ``: ローカルミラーのパス（既定: data/research/_ped_snapshot_cache.jsonl.gz）
- ``data/local/horse_pedigree_5gen/{4桁}/{horse_id}.json`` … ``load_full_pedigree_index(..., path=ディレクトリ)``
  でシャード JSON から索引を構築可（``horse_pedigree_5gen_bulk mirror-local`` 出力）

関連
----
- ``research/sire_factor_stats.build_and_upload_snapshot`` … スナップショット生成（オフライン・重い）
- ``research/sire_factor_race_map`` … 種牡馬因子用スリムキャッシュ（本モジュール経由に統合可）
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Iterator

logger = logging.getLogger(__name__)

# sire_factor_race_map と同じ既定パス（既存キャッシュをそのまま使う）
DEFAULT_LOCAL_GZ = Path(
    os.environ.get(
        "KEIBA_PEDIGREE_JSONL_GZ",
        "data/research/_ped_snapshot_cache.jsonl.gz",
    )
)

DISK_MAX_AGE_SEC = 86400.0
MEM_TTL_SEC = 3600.0

_mem_lock = threading.Lock()
_mem_index: dict[str, dict] | None = None
_mem_ts: float = 0.0
_mem_index_source: str | None = None


def local_gz_path() -> Path:
    return DEFAULT_LOCAL_GZ


def _download_snapshot_gz(storage: Any) -> bytes | None:
    from src.research.pedigree.sire_factor_stats import _download_snapshot_blob

    return _download_snapshot_blob(storage, "horse_pedigree_5gen.jsonl.gz")


def ensure_local_pedigree_gz(
    storage: Any,
    *,
    path: Path | None = None,
    disk_max_age_sec: float = DISK_MAX_AGE_SEC,
) -> Path | None:
    """
    ローカル ``.jsonl.gz`` が無い・古い場合のみ GCS スナップショットを1回ダウンロードする。
    storage が無効・ダウンロード失敗時は None。
    """
    p = path or local_gz_path()
    now = time.time()
    if p.exists():
        age = now - p.stat().st_mtime
        if age <= disk_max_age_sec:
            return p
        try:
            p.unlink()
        except OSError:
            pass

    raw = _download_snapshot_gz(storage)
    if not raw:
        return None
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(raw)
    logger.info("pedigree_local_store: GCS スナップショットを保存 %s (%.1f MB)", p, len(raw) / 2**20)
    return p


def iter_pedigree_records_from_gz(path: Path) -> Iterator[dict[str, Any]]:
    """ローカル gzip JSONL を 1 行ずつ yield（全件メモリに載せないバッチ向け）。"""
    raw = gzip.decompress(path.read_bytes())
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and rec.get("horse_id"):
            yield rec


def build_horse_id_index(path: Path) -> dict[str, dict]:
    """JSONL.gz 全件を ``horse_id -> レコード`` に読み込む。"""
    out: dict[str, dict] = {}
    for rec in iter_pedigree_records_from_gz(path):
        hid = str(rec.get("horse_id", "")).strip()
        if hid:
            out[hid] = rec
    return out


def local_pedigree_json_dir_has_records(root: Path) -> bool:
    """``data/local/horse_pedigree_5gen`` 等に 1 件以上の血統 JSON があるか。"""
    if not root.is_dir():
        return False
    for _ in root.rglob("*.json"):
        return True
    return False


def iter_pedigree_records_from_json_dir(root: Path) -> Iterator[dict[str, Any]]:
    """``mirror-local`` 等で書いた ``**/*.json`` を 1 件ずつ yield（全件メモリに載せない）。"""
    for f in sorted(root.rglob("*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(rec, dict) and rec.get("horse_id"):
            yield rec


def build_horse_id_index_from_json_dir(root: Path) -> dict[str, dict]:
    """シャード済み JSON 木を ``horse_id -> レコード`` に読み込む。"""
    out: dict[str, dict] = {}
    for rec in iter_pedigree_records_from_json_dir(root):
        hid = str(rec.get("horse_id", "")).strip()
        if hid:
            out[hid] = rec
    return out


def load_full_pedigree_index(
    storage: Any,
    *,
    path: Path | str | None = None,
    force_refresh: bool = False,
    mem_ttl_sec: float = MEM_TTL_SEC,
) -> dict[str, dict]:
    """
    メモリキャッシュ付きで ``horse_id -> 血統レコード`` を返す。

    - ``path`` が **ディレクトリ**のとき: 配下の ``*.json`` から索引（GCS は使わない）。
    - ``path`` が **.jsonl.gz ファイル**のとき: 従来どおり。無ければ ``ensure_local_pedigree_gz``。
    - ``path`` が None のとき: 環境変数 / 既定の gz パス。
    - ``force_refresh`` 時はメモリのみ無視し、ディスクがあればディスクから再読込。
    """
    global _mem_index, _mem_ts, _mem_index_source
    now = time.time()
    p = Path(path) if path else local_gz_path()
    try:
        src_key = str(p.resolve())
    except OSError:
        src_key = str(p)

    with _mem_lock:
        if (
            not force_refresh
            and _mem_index
            and _mem_index_source == src_key
            and (now - _mem_ts) < mem_ttl_sec
        ):
            return _mem_index

    if p.is_dir():
        if not local_pedigree_json_dir_has_records(p):
            logger.warning(
                "pedigree_local_store: %s に horse_pedigree_5gen の *.json がありません",
                p,
            )
            idx: dict[str, dict] = {}
        else:
            t0 = time.time()
            idx = build_horse_id_index_from_json_dir(p)
            logger.info(
                "pedigree_local_store: ディレクトリからインデックス構築 %d 頭 (%.1fs)",
                len(idx),
                time.time() - t0,
            )
        with _mem_lock:
            _mem_index = idx
            _mem_ts = time.time()
            _mem_index_source = src_key
        return idx

    if not p.exists() or force_refresh:
        if storage is not None:
            ensure_local_pedigree_gz(storage, path=p)
        if not p.exists():
            logger.warning(
                "pedigree_local_store: %s が無い（GCS から取得するか build_and_upload_snapshot を実行）",
                p,
            )
            return {}

    t0 = time.time()
    idx = build_horse_id_index(p)
    with _mem_lock:
        _mem_index = idx
        _mem_ts = time.time()
        _mem_index_source = src_key
    logger.info(
        "pedigree_local_store: インデックス構築 %d 頭 (%.1fs)",
        len(idx),
        time.time() - t0,
    )
    return idx


def ancestor_horse_ids_from_record(rec: dict | None) -> list[str]:
    """1 件の血統レコードから、祖先ノードの horse_id を列挙（空文字除外）。"""
    if not rec:
        return []
    out: list[str] = []
    for anc in rec.get("ancestors") or []:
        hid = str(anc.get("horse_id") or "").strip()
        if hid:
            out.append(hid)
    return out


# === 10gen 統合データ用ヘルパー (build_horse_pedigree_10gen.py で生成) ===

LOCAL_10GEN_DIR = Path("data/local/horse_pedigree_10gen")


def local_10gen_path(horse_id: str) -> Path:
    """horse_id から 10gen 統合 JSON のローカルパスを返す。"""
    prefix = horse_id[:4] if len(horse_id) >= 4 else "0000"
    return LOCAL_10GEN_DIR / prefix / f"{horse_id}.json"


def load_10gen_record(horse_id: str) -> dict | None:
    """1 馬の 10gen 統合データを直接ロード (キャッシュ無し)。

    形式は ``build_horse_pedigree_10gen.py`` 参照。
    存在しない場合は None。
    """
    p = local_10gen_path(horse_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        logger.warning("10gen ロード失敗 %s: %s", horse_id, e)
        return None


def load_full_pedigree_10gen_index(
    *,
    path: Path | str | None = None,
    force_refresh: bool = False,
    mem_ttl_sec: float = MEM_TTL_SEC,
) -> dict[str, dict]:
    """10gen 統合データを ``horse_id -> レコード`` でロード (メモリキャッシュ付き)。

    引数 ``path`` は 10gen 統合ディレクトリ (default: data/local/horse_pedigree_10gen)。
    ``load_full_pedigree_index`` と同じインタフェースで使える。
    """
    p = Path(path) if path else LOCAL_10GEN_DIR
    return load_full_pedigree_index(
        storage=None,
        path=p,
        force_refresh=force_refresh,
        mem_ttl_sec=mem_ttl_sec,
    )


def expand_10gen_ancestors_from_record(rec: dict | None) -> dict[str, set[str]]:
    """10gen レコードから 4 視点別の祖先 ID 集合を返す。

    Returns:
        {'father': set, 'dam_sire_line': set, 'dam_dam_line': set, 'dam': set}
        (g=1 の母自身は 'dam')

    生成時にすでに ``ancestors[].side`` が付与されているため、ここでは単純集計のみ。
    """
    out = {"father": set(), "dam_sire_line": set(), "dam_dam_line": set(), "dam": set()}
    if not rec:
        return out
    for anc in rec.get("ancestors") or []:
        hid = str(anc.get("horse_id") or "").strip()
        if not hid:
            continue
        side = anc.get("side") or "unknown"
        if side in out:
            out[side].add(hid)
    return out


def bfs_pedigree_closure(
    seed_horse_ids: Iterable[str],
    index: dict[str, dict],
    *,
    max_nodes: int = 10_000,
) -> tuple[set[str], int]:
    """
    インデックス上で、種牡馬・母馬の ``horse_id`` から到達可能な祖先 ID を BFS で収集。

    各 ``horse_id`` は高々 1 回だけ ``index`` を参照するので、
    「母系10頭 × それぞれ5世代」でも **重複 ID は共有**され、GCS 相当の N 回取得にならない。

    Returns:
        (visited_ids, missing_count) … index に無くスキップした参照数
    """
    q: deque[str] = deque()
    seen: set[str] = set()
    missing = 0
    for s in seed_horse_ids:
        s = (s or "").strip()
        if s and s not in seen:
            seen.add(s)
            q.append(s)

    while q and len(seen) < max_nodes:
        hid = q.popleft()
        rec = index.get(hid)
        if not rec:
            missing += 1
            continue
        for nxt in ancestor_horse_ids_from_record(rec):
            if nxt not in seen:
                seen.add(nxt)
                if len(seen) >= max_nodes:
                    break
                q.append(nxt)
    return seen, missing


def stallion_pedigree_closure_horse_ids(
    seed_horse_ids: Iterable[str],
    index: dict[str, dict],
    *,
    max_nodes: int = 500_000,
) -> tuple[set[str], bool]:
    """
    5 世代表の牡スロット（各世代で position が偶数）に現れる horse_id を BFS で収集する。

    1 頭の完全な 5 世代表には牡が ``research.pedigree.pedigree_similarity.MALE_PEDIGREE_SLOTS_5GEN`` (=31)
    スロットあり（種牡馬因子の 6 スロット加権とは別。こちらは表の牡をすべて列挙）。

    インデックスに無い馬は集合に含めるが、そこからは展開しない（未取得はワーカー取得後に
    再投入バッチで辿れる）。
    """
    from src.research.pedigree.pedigree_similarity import male_pedigree_slot_horse_ids_from_ancestors

    pending: deque[str] = deque()
    seen: set[str] = set()
    for s in seed_horse_ids:
        s = (s or "").strip()
        if s and s not in seen:
            seen.add(s)
            pending.append(s)

    hit_cap = False
    while pending:
        if len(seen) >= max_nodes:
            hit_cap = True
            break
        hid = pending.popleft()
        rec = index.get(hid)
        if not rec:
            continue
        for nxt in male_pedigree_slot_horse_ids_from_ancestors(rec.get("ancestors")):
            if nxt not in seen:
                seen.add(nxt)
                pending.append(nxt)
                if len(seen) >= max_nodes:
                    hit_cap = True
                    break
        if hit_cap:
            break

    return seen, hit_cap
