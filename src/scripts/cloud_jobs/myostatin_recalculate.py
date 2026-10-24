#!/usr/bin/env python3
"""
全ての未確定馬のミオスタチン遺伝子型を血統から再計算し、ナレッジベース JSON を更新する CLI。

元は ``POST /api/myostatin/recalculate``（src/api/app.py）のハンドラ内に直接書かれていた
処理を ``src.research.genes.myostatin.recalculate_myostatin_genotypes`` に切り出し、
同エンドポイントとこの CLI の両方から共通で呼び出す。処理ロジック自体はこの CLI では
実装しない。

Cloud Scheduler + Cloud Run Jobs から定期実行する想定
（docs/operations/gcp-cloud-run-jobs.md 参照）。

Usage:
  python -m src.scripts.cloud_jobs.myostatin_recalculate
  python -m src.scripts.cloud_jobs.myostatin_recalculate --json-path /path/to/myostatin_genes.json
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-path",
        default=None,
        help="myostatin_genes.json の明示パス（省略時は src.config.data_paths.MYOSTATIN_GENES_JSON）",
    )
    args = parser.parse_args(argv)

    from src.research.genes.myostatin import recalculate_myostatin_genotypes

    result = recalculate_myostatin_genotypes(args.json_path)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
