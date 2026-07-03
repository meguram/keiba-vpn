"""
MLflow プラットフォーム全体のヘルスチェック CLI。

例:
  python -m src.scripts.maintenance.mlflow_platform_health
  python -m src.scripts.maintenance.mlflow_platform_health --json
"""

from __future__ import annotations

import argparse
import json
import sys

from src.pipeline.mlflow.runtime import ensure_mlruns_symlink, platform_health


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MLflow 全モデルヘルス")
    parser.add_argument("--json", action="store_true", help="JSON で出力")
    parser.add_argument(
        "--force-serve",
        action="store_true",
        help="Serving /health をキャッシュせず再確認",
    )
    args = parser.parse_args(argv)

    ensure_mlruns_symlink()
    report = platform_health(force_serve_check=args.force_serve)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        tr = report["mlflow_tracking"]
        print(f"Tracking: {tr.get('uri')} ok={tr.get('ok')} ({tr.get('ms')}ms)")
        for m in report["models"]:
            serve = "OK" if m.get("serve_ok") else "-"
            local = "loaded" if m.get("local_booster_loaded") else "-"
            print(
                f"  [{m['lifecycle']}] {m['key']}: "
                f"serve={serve} local={local} backend={m.get('infer_backend')}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
