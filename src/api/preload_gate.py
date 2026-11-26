"""起動時の先読み（src/api/cloudrun_entry.py の preload_models）が済むまで、予測の計算を少しだけ待たせる印。

起動直後の 1 本目の予測が、裏の先読みと同時に同じ読み込み（LightGBM のモデル・めぐ指数の表など）を払い、10〜11 秒かかっていた
（2026-11-26 の stg の計測）。先読みが動いている間は、計算の側がその完了を待ってから読む（二重に読まない）。
待ちには上限があり、超えたら自分で読む（先読みが失敗・止まったときに要求を待たせ過ぎない）。

先読みを始めない環境（ホスト・テスト）では begin() が呼ばれないので、wait() は何もしない。
"""
from __future__ import annotations

import threading
import time

WAIT_SEC = 8.0                     # 予測の計算が先読みを待つ上限（秒）
EXPIRE_SEC = 60.0                  # begin() からこれだけ経てば、finish() が呼ばれていなくても待たない（呼び忘れ・例外の道の安全弁）

_done = threading.Event()
_active = False
_began = 0.0


def begin() -> None:
    """先読みを始める。finish() を必ず対で呼ぶ（呼ばないと、要求ごとに WAIT_SEC 待つ）。"""
    global _active, _began
    _done.clear()
    _began = time.monotonic()
    _active = True


def finish() -> None:
    _done.set()


def pending() -> bool:
    return _active and not _done.is_set() and time.monotonic() - _began < EXPIRE_SEC


def wait(timeout: float = WAIT_SEC) -> float:
    """先読みが動いていれば、その完了を最大 timeout 秒待つ。待った秒数を返す。"""
    if not pending():
        return 0.0
    t0 = time.monotonic()
    _done.wait(max(0.0, min(timeout, EXPIRE_SEC - (t0 - _began))))
    return time.monotonic() - t0


def reset() -> None:
    """テスト用: 先読みが無い状態に戻す。"""
    global _active
    _active = False
    _done.clear()


def memory_note() -> str:
    """ログ用のメモリの値（MB）: プロセスの最大の常駐（VmHWM）と、コンテナ（cgroup）の今と最大。読めないものは出さない。"""
    out = []
    try:
        for line in open("/proc/self/status", encoding="ascii"):
            if line.startswith("VmHWM:"):
                out.append(f"rss_peak={int(line.split()[1]) // 1024}MB")
    except OSError:
        pass
    for name, path in (("cg_now", "/sys/fs/cgroup/memory.current"), ("cg_peak", "/sys/fs/cgroup/memory.peak"),
                       ("cg_now", "/sys/fs/cgroup/memory/memory.usage_in_bytes"), ("cg_peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes")):
        if any(x.startswith(name + "=") for x in out):
            continue
        try:
            out.append(f"{name}={int(open(path, encoding='ascii').read().strip()) // (1024 * 1024)}MB")
        except (OSError, ValueError):
            pass
    return " ".join(out)
