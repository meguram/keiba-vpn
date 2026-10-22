"""prod（VPS）の状況を収集して data/local/meta/prod_status/latest.json に書く（ホストの cron で 1〜5 分おき）。

    python -m src.scripts.ops.collect_prod_status            # 収集して書く
    python -m src.scripts.ops.collect_prod_status --print    # 書かずに結果を表示
    python -m src.scripts.ops.collect_prod_status --only vps releases

読み取りだけ（VPS へは固定の採取スクリプトを ssh で流し、GCP は読み取り専用の gcloud_ro.sh）。画面（監視ポータル）はこのキャッシュを読むだけ。
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print", dest="show", action="store_true", help="書かずに結果を表示")
    ap.add_argument("--only", nargs="*", default=None, help="取るセクションを限定")
    ns = ap.parse_args(argv)
    from src.monitor import prod_status as P
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    res = P.collect_all(only=ns.only)
    if ns.show:
        print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
        return 0
    path = P.write_cache(res)
    notified = P.maybe_notify(res)
    ov = res["overall"]
    print(f"{res['collected_at']} {ov['level']} → {path}（{res['duration_s']}s）" + ("  Slack 通知あり" if notified else ""))
    for r in ov["reasons"]:
        print(f"  [{r['level']}] {r['message']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
