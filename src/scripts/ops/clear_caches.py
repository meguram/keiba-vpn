"""API のメモリ上のキャッシュを、再起動なしで空にする（世代ファイルを進める。次の要求で各 API プロセスが反映する）。

使い方: python -m src.scripts.ops.clear_caches [--reason 理由]
"""

from __future__ import annotations

import argparse

from src.utils import cache_epoch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reason", default="manual")
    args = ap.parse_args()
    cache_epoch.bump(args.reason)
    print(f"キャッシュの世代を進めました（{args.reason}）。API は次の要求で空にします。")


if __name__ == "__main__":
    main()
