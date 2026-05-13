"""
キュー 1 ジョブの実行中だけ有効にしたい永続化ポリシー（ContextVar）。

例: ジョブの ``skip_local_mirror`` が真のとき、GCS 保存は行うが
    ``data/local/mirror/`` への常設コピーは行わない。
"""

from __future__ import annotations

from contextvars import ContextVar, Token

_ctx_skip_local_mirror: ContextVar[bool] = ContextVar("skip_local_mirror", default=False)


def is_skip_local_mirror() -> bool:
    return _ctx_skip_local_mirror.get()


def set_skip_local_mirror(v: bool) -> Token:
    return _ctx_skip_local_mirror.set(bool(v))


def reset_skip_local_mirror(t: Token) -> None:
    _ctx_skip_local_mirror.reset(t)
