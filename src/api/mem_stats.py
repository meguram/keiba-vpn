"""API のプロセスのメモリの内訳（Cloud Run のメモリの張り付きの調査・再発の監視用。読み取りだけ・GCS に触れない）。

Cloud Run（gen2）は cgroup v2 の ``memory.current`` がメモリの上限（``memory.max``）に達すると OOM で落とす。
``/tmp`` やコンテナの書き込み層（tmpfs）もここに数えられ、プロセスの RSS には出ないので、RSS ではなく cgroup の値を見る。
2026-11-15〜16 に本番の 1 台が 2GiB の 99% に張り付いた（docs/operations/api_memory_20261117.md）。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

CGROUP = Path("/sys/fs/cgroup")
REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_int(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    return int(raw) if raw.isdigit() else None       # memory.max は上限なしのとき "max"


MEMINFO = Path("/proc/meminfo")
_NO_LIMIT = 1 << 60          # cgroup v1 の「上限なし」はとても大きな数


def _meminfo(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="ascii").splitlines():
            k, _, rest = line.partition(":")
            parts = rest.split()
            if parts and parts[0].isdigit():
                out[k] = int(parts[0]) * 1024
    except OSError:
        pass
    return out


def cgroup_memory(root: Path = CGROUP, meminfo: Path = MEMINFO) -> dict[str, Any]:
    """使用量・上限（バイト）と割合。cgroup v2 → cgroup v1 → /proc/meminfo の順に読めたものを使う（source に記録）。
    Cloud Run（gen2）は 1 台ごとの VM で、2026-11-17 の stg では cgroup v2 のファイルが無かった。VM の MemTotal はインスタンスのメモリに近く、
    /tmp（tmpfs）は Shmem として「使用中」に入る（MemAvailable に含まれない）ので、used = MemTotal - MemAvailable で張り付きを見られる。"""
    cur, mx = _read_int(root / "memory.current"), _read_int(root / "memory.max")
    source = "cgroup2"
    if cur is None:
        cur, mx = _read_int(root / "memory" / "memory.usage_in_bytes"), _read_int(root / "memory" / "memory.limit_in_bytes")
        source = "cgroup1"
        if mx is not None and mx >= _NO_LIMIT:
            mx = None
    mi = _meminfo(meminfo)
    if (cur is None or not mx) and mi.get("MemTotal") and "MemAvailable" in mi:
        cur, mx, source = mi["MemTotal"] - mi["MemAvailable"], mi["MemTotal"], "meminfo"
    if cur is None:
        source = None
    out: dict[str, Any] = {"current": cur, "max": mx, "source": source}
    try:
        for line in (root / "memory.stat").read_text(encoding="ascii").splitlines():
            k, _, v = line.partition(" ")
            if k in ("anon", "file", "shmem", "kernel"):
                out[k] = int(v)
    except (OSError, ValueError):
        pass
    used = cur
    if source == "cgroup2" and cur is not None and "file" in out and "shmem" in out:
        used = max(0, cur - max(0, out["file"] - out["shmem"]))      # 回収できるページキャッシュ（shmem=/tmp 以外のファイル）は除く
    elif source == "cgroup1" and cur is not None:
        # v1 の usage_in_bytes はページキャッシュを含む。v2 の分岐と同じく、ファイルの LRU（active_file + inactive_file）を除く。
        # tmpfs（/tmp）・shmem のページはファイルではなく anon の LRU に載るので、使用中のまま数える。dirty・writeback もファイルの LRU に
        # 入るため、書き込みが多いときは少し少なく見積もる。前提が外れたときの安全策として、下で /proc/meminfo の割合と大きいほうを使う
        out["used_basis"] = "usage"
        try:
            st1 = {}
            for line in (root / "memory" / "memory.stat").read_text(encoding="ascii").splitlines():
                k, _, v = line.partition(" ")
                if v.strip().isdigit():
                    st1[k] = int(v)

            def pick(name: str) -> int | None:
                return st1.get(f"total_{name}", st1.get(name))

            active, inactive = pick("active_file"), pick("inactive_file")
            if active is not None and inactive is not None:
                vals = {"anon": pick("rss"), "shmem": pick("shmem"), "cache": pick("cache")}
                out.update({k: v for k, v in vals.items() if v is not None and k not in out})
                out["file_lru"] = active + inactive
                used = max(0, cur - active - inactive)
                out["used_basis"] = "usage-file_lru"
        except OSError:
            pass
    out["used"] = used
    out["ratio"] = round(used / mx, 4) if used is not None and mx else None
    if source in ("cgroup1", "cgroup2") and mi.get("MemTotal") and "MemAvailable" in mi:
        # 安全策: cgroup の前提（書き込み層・/tmp が cgroup に数えられる等）が外れても点検が止まらないよう、VM 全体の割合と大きいほうを使う
        mr = round((mi["MemTotal"] - mi["MemAvailable"]) / mi["MemTotal"], 4)
        out["meminfo_ratio"] = mr
        if out["ratio"] is None or mr > out["ratio"]:
            out["ratio"] = mr
            out["ratio_basis"] = "meminfo"
    if mi:
        out["shmem"] = out.get("shmem", mi.get("Shmem"))
    return out


def rss_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


class Budget:
    """1 回の統計で、lstat の件数と時間の合計に上限を設ける（統計の口が 1 vCPU を長く占めないように）。"""

    def __init__(self, max_files: int = 100_000, max_sec: float = 3.0) -> None:
        import time

        self.files_left, self.deadline, self._clock = max_files, time.monotonic() + max_sec, time.monotonic

    def take(self) -> bool:
        self.files_left -= 1
        return self.files_left >= 0 and self._clock() < self.deadline


def dir_bytes(path: Path, max_files: int = 50_000, budget: Budget | None = None) -> dict[str, Any]:
    """ディレクトリの合計バイト数とファイル数（max_files か全体の予算で打ち切り。打ち切ったら truncated）。"""
    total = n = 0
    truncated = False
    if not path.is_dir():
        return {"bytes": 0, "files": 0}
    for dirpath, _dirs, files in os.walk(path):
        for f in files:
            if n >= max_files or (budget is not None and not budget.take()):
                truncated = True
                break
            try:
                total += os.lstat(os.path.join(dirpath, f)).st_size
            except OSError:
                continue
            n += 1
        if truncated:
            break
    out: dict[str, Any] = {"bytes": total, "files": n}
    if truncated:
        out["truncated"] = True
    return out


def obj_bytes(obj: Any, max_nodes: int = 20_000) -> tuple[int, bool]:
    """JSON 風の値（dict・list・str・数）のおおよそのバイト数を、文字列を作らずに数える（sys.getsizeof の合計）。
    max_nodes で打ち切る（打ち切ったら 2 つ目が True）。メモリが張り付いた時点で呼ばれても、新たに大きな確保をしない。"""
    import sys

    total = 0
    stack = [obj]
    seen = 0
    while stack:
        o = stack.pop()
        seen += 1
        if seen > max_nodes:
            return total, True
        total += sys.getsizeof(o)
        if isinstance(o, dict):
            stack.extend(o.keys())
            stack.extend(o.values())
        elif isinstance(o, (list, tuple)):
            stack.extend(o)
    return total, False


def approx_bytes(values: list[Any], total_count: int, sample: int = 10) -> int:
    """キャッシュの中身のおおよその量（先頭 sample 件の obj_bytes の平均 × 件数）。"""
    if not values or total_count <= 0:
        return 0
    got = [obj_bytes(v)[0] for v in values[:sample]]
    return int(sum(got) / len(got) * total_count) if got else 0


def storage_stats() -> dict[str, Any]:
    """生きている HybridStorage の数と、各 load キャッシュの件数・おおよその量。"""
    from src.scraper.storage import HybridStorage

    insts = list(HybridStorage._instances)
    caches = []
    total = 0
    for st in insts:
        with st._load_cache_lock:
            items = list(st._load_cache.values())
            n = len(items)
            sample = [v[1] for v in items[:10]]
        b = st.load_cache_bytes() if hasattr(st, "load_cache_bytes") else approx_bytes(sample, n)
        total += b
        caches.append({"entries": n, "approx_bytes": b})
    caches.sort(key=lambda c: -c["approx_bytes"])
    gcs = insts[0].gcs_health() if insts and hasattr(insts[0], "gcs_health") else None
    return {"instances": len(insts), "load_cache_total_bytes": total, "load_caches": caches[:10], "gcs": gcs}


def tmp_stats(budget: Budget | None = None) -> dict[str, Any]:
    tmp = Path(tempfile.gettempdir())
    pr = Path(os.environ.get("KEIBA_PRERENDER_DIR") or tmp / "keiba-prerender")
    return {"prerender": dir_bytes(pr, budget=budget), "tmp_total": dir_bytes(tmp, budget=budget)}


def mirror_stats(root: Path = REPO_ROOT, budget: Budget | None = None) -> dict[str, Any]:
    """GCS のミラーで取り込んだ木（コンテナの書き込み層 = メモリに数える）の量。同じフォルダは 1 回だけ数える。"""
    try:
        from src.ops.gcs_mirror import PUSH_TREES, split_tree
    except Exception:  # noqa: BLE001
        return {}
    dirs = sorted({split_tree(t)[0] for t in PUSH_TREES})
    dirs = [d for d in dirs if not any(d != o and d.startswith(o.rstrip("/") + "/") for o in dirs)]    # 親が並んでいれば子は数えない
    out: dict[str, Any] = {}
    total = 0
    for d in dirs:
        st = dir_bytes(root / d, budget=budget)
        total += st["bytes"]
        if st["bytes"] or st.get("truncated"):
            out[d] = st
    out["_total_bytes"] = total
    return out


def snapshot() -> dict[str, Any]:
    """メモリの内訳を 1 回まとめて返す（GET /ops/memory）。ファイルの走査は全体で 10 万件・3 秒まで。"""
    out: dict[str, Any] = {"cgroup": cgroup_memory(), "rss": rss_bytes()}
    budget = Budget()
    for name, fn in (("storage", storage_stats), ("tmp", lambda: tmp_stats(budget)), ("mirror", lambda: mirror_stats(budget=budget))):
        try:
            out[name] = fn()
        except Exception as exc:  # noqa: BLE001  統計の失敗で応答を落とさない（型の名前だけ返す）
            out[name] = {"error": type(exc).__name__}
    return out
