"""tp-37: 馬別調教（horse_training）の『記者コメント』（comment 列。追い切り表には落としていた）を取り出して表にする。

python -m src.scripts.data.tp37_comments extract   # tp37/training_comments.parquet（コメントのある追い切りだけ。horse_id・train_date・race_date・comment）
読み取りだけ（GCS の horse_training を読む。書き換えない）。生のコメントは公開リポジトリに入れない（保存先は data/local の中）。
"""

from __future__ import annotations

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/local/modeling/training_pipeline_v4/tp37"
RACE_DATE = re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})")


def _rows(rec: dict | None, hid: str) -> tuple[int, list[dict]]:
    if not rec:
        return 0, []
    rows = []
    es = rec.get("entries") or []
    for e in es:
        c = e.get("comment")
        m = RACE_DATE.search(str(e.get("race_info", "")))
        if c and m and e.get("date"):
            rows.append({"horse_id": str(rec.get("horse_id") or hid), "train_date": e["date"], "race_date": f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}", "comment": str(c)})
    return len(es), rows


def cmd_extract(threads: int) -> None:
    from src.scraper.run import ScraperRunner

    st = ScraperRunner(auto_login=False).storage
    keys = st.list_keys("horse_training")
    print(f"{len(keys):,} 頭", flush=True)
    t0 = time.time()
    rows, n_entries, done = [], 0, 0

    def work(k):
        try:
            return _rows(st.load("horse_training", k), k)
        except Exception as e:  # noqa: BLE001
            return 0, [{"horse_id": k, "error": str(e)[:80]}]

    with ThreadPoolExecutor(threads) as ex:
        for n, r in ex.map(work, keys):
            n_entries += n
            rows += [x for x in r if "error" not in x]
            done += 1
            if done % 3000 == 0:
                print(f"{done:,}/{len(keys):,} コメント付き {len(rows):,} / 追い切り {n_entries:,} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "training_comments.parquet", index=False)
    print(f"保存: {len(df):,} 行（追い切り {n_entries:,} 本のうち {len(df) / max(n_entries, 1):.3f}）", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract"])
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    cmd_extract(a.threads)


if __name__ == "__main__":
    main()
