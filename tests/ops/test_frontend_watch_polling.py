"""next dev の起動経路が、9p の作業ツリーでも変更を拾えるようポーリングを有効にしているか（2026-11-12: 監視役の経路に無く、置き換えたファイルが再コンパイルされなかった）。
ポーリングは dev のときだけ（next start には付けない）。export が dev の if の内側にあることを行単位で確かめる。"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPORT = re.compile(r'^\s*export WATCHPACK_POLLING="\$\{WATCHPACK_POLLING:-true\}"')
IF_DEV = re.compile(r'^(\s*)if .*FRONTEND_NPM_SCRIPT"? *==? *"?dev"?')


def _enclosing_if(lines: list[str], i: int) -> str | None:
    """i 行目を囲む直近の if 行（同じか浅い字下げの fi で閉じられていないもの）。"""
    indent = len(lines[i]) - len(lines[i].lstrip())
    closed_at = indent   # これより浅い（小さい）字下げの fi・if だけを見る
    for j in range(i - 1, -1, -1):
        s = lines[j]
        ind = len(s) - len(s.lstrip())
        if ind >= closed_at or not s.strip():
            continue
        if re.match(r"^\s*fi\b", s):
            closed_at = ind   # 兄弟の if ブロックの終わり。その if を飛ばす
            continue
        if re.match(r"^\s*if\b", s):
            return s
        closed_at = ind
    return None


@pytest.mark.parametrize("script", ["scripts/server/service_start.sh", "scripts/server/server_watchdog.sh"])
def test_polling_is_exported_only_inside_dev_branch(script):
    lines = (ROOT / script).read_text(encoding="utf-8").splitlines()
    hits = [i for i, s in enumerate(lines) if EXPORT.match(s)]
    assert hits, f"{script}: next dev にポーリングの設定が無い"
    for i in hits:
        enc = _enclosing_if(lines, i)
        assert enc is not None and IF_DEV.match(enc), f"{script}:{i + 1}: ポーリングの export が dev の if の外にある（next start にも付く）"


def test_enclosing_if_detects_export_outside_the_dev_branch():
    inside = ['  if [ "$FRONTEND_NPM_SCRIPT" = "dev" ]; then', "    # コメント", '    export WATCHPACK_POLLING="${WATCHPACK_POLLING:-true}"', "  fi"]
    outside = ['  if [ "$FRONTEND_NPM_SCRIPT" = "dev" ]; then', "    :", "  fi", '  export WATCHPACK_POLLING="${WATCHPACK_POLLING:-true}"']
    assert IF_DEV.match(_enclosing_if(inside, 2) or "")
    assert _enclosing_if(outside, 3) is None
