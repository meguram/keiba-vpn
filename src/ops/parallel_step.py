"""GCP Jobs の並走期間（``KEIBA_PARALLEL_RUN=1``）に、job の step を本番 GCS へ書かせずに実行する入口。

    python -m src.ops.parallel_step <module> [引数…]      # job_dispatch が step の ``-m <module>`` をこれで包む

並走中はホストが本番の取得・計算を続けている。Jobs の step は ``KEIBA_ENV=prod`` で同じコードを通すため、``HybridStorage.save`` などが
本番のキー（``chuou/…``・``ops/…``）に書くと、ホストの成果物を上書きする（2026-10-27 に predict_race が想定オッズを上書きした件）。
そこで VPS の影モードと同じガード（``src.ops.shadow_storage``）を ``root="parallel"`` で入れてから step のモジュールを実行する:
GCS への書き込みは ``parallel/<JST 日付>/<同じキー>`` に付け替わり、削除・コピーなど他の操作は例外で止まる。読み取りは本番のまま。
ローカルのファイル（state_out で ``state/`` へ上がるもの）には影響しない。``ops/mirror`` へは並走中は job_dispatch が上げない。
step が起こす子プロセスの python にも同じガードが入るよう、``parallel_site/sitecustomize.py`` を ``PYTHONPATH`` の先頭に足す。
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

from src.ops import shadow_storage

SITE_DIR = Path(__file__).resolve().parent / "parallel_site"
REPO_ROOT = Path(__file__).resolve().parents[2]


def child_env(env: dict[str, str]) -> dict[str, str]:
    """子プロセスにもガードが入る環境変数（KEIBA_PARALLEL_GUARD=1・PYTHONPATH の先頭に parallel_site とリポジトリ）。"""
    parts = [str(SITE_DIR), str(REPO_ROOT)] + [p for p in (env.get("PYTHONPATH") or "").split(os.pathsep) if p]
    return {**env, "KEIBA_PARALLEL_GUARD": "1", "PYTHONPATH": os.pathsep.join(dict.fromkeys(parts))}


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit("使い方: python -m src.ops.parallel_step <module> [引数…]")
    shadow_storage.install(root=shadow_storage.PARALLEL_ROOT)
    os.environ.update(child_env(dict(os.environ)))
    module, rest = args[0], args[1:]
    sys.argv = [module, *rest]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
