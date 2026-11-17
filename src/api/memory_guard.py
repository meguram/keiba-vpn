"""Cloud Run の API が、メモリの張り付きや GCS の長い障害から自分で抜ける仕組み（外からの ping・liveness probe は足さない）。

2026-11-15〜16 に本番の 1 台が 2GiB の 99% に張り付き、CPU 100%・GCS の時間切れで 504 が続いた。入れ替わらずに 3 時間近く残った。
（docs/operations/api_memory_20261117.md）

- CPU はリクエスト中だけ割り当てられる（リクエスト課金）ので、常駐スレッドではなく、入口（src/api/cloudrun_entry.py の Gateway）が
  要求の始めに ``maybe_check()`` を呼ぶ。実際に確かめるのは ``check_every`` 秒に 1 回だけ（cgroup のファイルを 2 つ読む程度）。
- メモリ（src/api/mem_stats.cgroup_memory の ratio。回収できるページキャッシュは除き、/tmp は数える）:
  - ``soft``（既定 0.85）以上: キャッシュ（load・一覧・bundle・レース詳細など）を捨て、応答の置き場（prerender）を半分に削り、GC と
    malloc_trim でメモリを OS に返す。捨てるのは ``clear_min_sec``（既定 300 秒）に 1 回まで（捨て続けて取り直しの GCS の読み出しが増えないように）。
  - 捨てた後も、または捨てる間隔の内で ``hard``（既定 0.92）以上: 自分で終わる（下）。
- GCS: 共有の状態（KEIBA_GCS_SHARED_STATE=1）で、クライアントを作り直してから ``gcs_dead_sec``（既定 600 秒）以上 1 件も成功せず、
  いまも不調で、かつメモリも ``gcs_exit_ratio``（既定 0.7）以上のときだけ終わる（2026-11-15 の 1 台はメモリの張り付きと GCS の失敗が同時だった）。
  GCS だけの障害（地域の障害など）では終わらない: 新しい 1 台は起動時の取り込みもできず、手元の写しで答えている今の台より悪くなるため。
- 終わり方: SIGTERM（uvicorn が処理中の要求を終えてから止まる）→ ``grace_sec``（既定 20 秒）後に os._exit。Cloud Run が新しい 1 台を起こす。
  起動から ``min_uptime``（既定 300 秒）未満では終わらない（起動直後の読み込みで高いだけのことがある・入れ替わりの繰り返しを防ぐ）。
- ``KEIBA_MEMORY_GUARD=0`` で止められる。各値は ``KEIBA_MEMORY_GUARD_SOFT`` などで変えられる。
"""

from __future__ import annotations

import gc
import logging
import os
import signal
import threading
import time
from typing import Any, Callable

logger = logging.getLogger("api.memory_guard")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def enabled() -> bool:
    return (os.environ.get("KEIBA_MEMORY_GUARD") or "1").strip() != "0"


# メモリの圧迫時に捨てるキャッシュ（src/utils/cache_epoch に登録された名前）。設定の取り直し（public_policy・較正の設定など）は含めない
CLEAR_NAMES = ("storage.load_cache", "app.bundle_cache", "app.race_list_cache", "app.race_list_memos", "app.status_cache",
               "entry_card", "delegates.value_cache", "prediction_results_api")


def malloc_trim() -> None:
    """glibc に、解放済みのメモリを OS へ返させる（dict を消しても arena に残り、memory.current が下がらないことが多い）。"""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001  glibc 以外では何もしない
        pass


def default_clear() -> None:
    """キャッシュを捨てる（CLEAR_NAMES・レース詳細）、応答の置き場を半分に、GC と malloc_trim。世代（cache_epoch）は進めない。"""
    from src.utils import cache_epoch

    cache_epoch.clear_some(CLEAR_NAMES, "メモリが高いため")
    try:
        from src.api import app as _app

        with _app._race_detail_cache_lock:
            _app._race_detail_cache.clear()
    except Exception:  # noqa: BLE001
        pass
    try:
        from src.api import prerender

        prerender.prune_size(prerender._max_bytes() // 2)
    except Exception:  # noqa: BLE001
        pass
    gc.collect()
    malloc_trim()


def default_terminate(grace_sec: float) -> None:
    """SIGTERM で uvicorn に処理中の要求を終えさせ、grace_sec 後に必ず終わる。"""
    timer = threading.Timer(grace_sec, os._exit, args=(1,))
    timer.daemon = True
    timer.start()
    os.kill(os.getpid(), signal.SIGTERM)


def gcs_stuck(dead_sec: float) -> bool:
    """共有の GCS の状態が、クライアントを作り直してから dead_sec 以上 1 件も成功せず、いまも不調か。"""
    from src.scraper import storage as S

    now = time.time()
    with S._SHARED_GCS_LOCK:
        states = list(S._SHARED_GCS.values())
    return any(st.rebuild_at and st.last_success < st.rebuild_at and now - st.rebuild_at >= dead_sec
               and st.last_failure and now - st.last_failure <= S._GCS_BACKOFF_SEC for st in states)


class MemoryGuard:
    def __init__(self, *, read: Callable[[], dict[str, Any]] | None = None, clear: Callable[[], None] = default_clear,
                 terminate: Callable[[float], None] = default_terminate, stuck: Callable[[float], bool] = gcs_stuck,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if read is None:
            from src.api.mem_stats import cgroup_memory

            read = cgroup_memory
        self.read, self.clear, self.terminate, self.stuck, self.clock = read, clear, terminate, stuck, clock
        self.soft = _env_float("KEIBA_MEMORY_GUARD_SOFT", 0.85)
        self.hard = _env_float("KEIBA_MEMORY_GUARD_HARD", 0.92)
        self.check_every = _env_float("KEIBA_MEMORY_GUARD_EVERY_SEC", 30)
        self.min_uptime = _env_float("KEIBA_MEMORY_GUARD_MIN_UPTIME_SEC", 300)
        self.grace_sec = _env_float("KEIBA_MEMORY_GUARD_GRACE_SEC", 20)
        self.gcs_dead_sec = _env_float("KEIBA_MEMORY_GUARD_GCS_DEAD_SEC", 600)
        self.gcs_exit_ratio = _env_float("KEIBA_MEMORY_GUARD_GCS_RATIO", 0.7)
        self.clear_min_sec = _env_float("KEIBA_MEMORY_GUARD_CLEAR_MIN_SEC", 300)
        self.started = clock()
        self._last = -1e18
        self._last_clear = -1e18
        self._lock = threading.Lock()
        self.exiting = False
        self.clears = 0

    def claim(self) -> bool:
        """確かめる番を取る（取れたら True。同時に来た要求のうち 1 つだけが取れる）。入口は取れたときだけスレッドで check を呼ぶ。"""
        now = self.clock()
        if self.exiting or now - self._last < self.check_every:      # ロックの前の軽い判定（ほとんどの要求はここで帰る）
            return False
        with self._lock:
            if self.exiting or now - self._last < self.check_every:
                return False
            self._last = now
            return True

    def maybe_check(self) -> str | None:
        """確かめる時刻なら確かめて、した処理（"clear"・"exit_memory"・"exit_gcs"）を返す。何もしなければ None。例外は出さない。"""
        return self.check() if self.claim() else None

    def check(self) -> str | None:
        """確かめる（claim の後に呼ぶ）。例外は出さない。"""
        now = self.clock()
        try:
            return self._check(now)
        except Exception as exc:  # noqa: BLE001  点検の失敗で要求を落とさない
            logger.warning("メモリの点検に失敗: %r", exc)
            return None

    def _ratio(self) -> float | None:
        r = self.read().get("ratio")
        return float(r) if isinstance(r, (int, float)) else None

    def _check(self, now: float) -> str | None:
        old_enough = now - self.started >= self.min_uptime
        ratio = self._ratio()
        if ratio is not None and ratio >= self.soft:
            if now - self._last_clear < self.clear_min_sec:
                if ratio >= self.hard and old_enough:          # 少し前に捨てても下がっていない
                    return self._exit("exit_memory", f"キャッシュを捨てた後もメモリが上限の {ratio * 100:.0f}%")
                return None
            logger.warning("メモリが上限の %.0f%%。キャッシュを捨てます", ratio * 100)
            self._last_clear = now
            self.clear()
            self.clears += 1
            after = self._ratio()
            if after is not None and after >= self.hard and old_enough:
                return self._exit("exit_memory", f"キャッシュを捨てた後もメモリが上限の {after * 100:.0f}%")
            return "clear"
        if old_enough and ratio is not None and ratio >= self.gcs_exit_ratio and self.stuck(self.gcs_dead_sec):
            return self._exit("exit_gcs", f"メモリが上限の {ratio * 100:.0f}% で、GCS の読み出しがクライアントを作り直してから "
                                          f"{self.gcs_dead_sec:.0f} 秒以上成功しない")
        return None

    def _exit(self, kind: str, why: str) -> str:
        with self._lock:
            if self.exiting:
                return kind
            self.exiting = True
        logger.error("このインスタンスを終了します（Cloud Run が入れ替える）: %s", why)
        self.terminate(self.grace_sec)
        return kind
