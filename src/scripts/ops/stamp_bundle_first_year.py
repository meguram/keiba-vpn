"""束（bundle.json）に履歴の起点年を書き足す。起点年の記録（first_year_stamp）を入れる前に作った束のための道具。

    python -m src.scripts.ops.stamp_bundle_first_year --bundle-dir <束> --first-year 2016 [--train-year-min 2016] [--dry-run]

- 書くのは first_year・train_year_min と、出どころが無いときだけ first_year_source（"stamp"）。ほかの鍵は変えない。一時名に書いて fsync してから置き換える。
- 手元の束を書き換えるだけ。ジョブが取得する GCS の版に入れるには、このあと model_publish で公開し直す（manifest の sha256 が変わる）。
- 既に別の first_year・train_year_min が書かれている、または fit_info.json の first_year と食い違うときは止める。同じ値なら何もしない。
- 推論が束の起点年と実行時の KEIBA_FIRST_YEAR を照合するときの元になる（training_years.bundle_first_year / check_first_year）。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def stamp(bundle_dir: Path, first_year: int, train_year_min: int | None = None, dry_run: bool = False) -> dict:
    p = Path(bundle_dir) / "bundle.json"
    meta = json.loads(p.read_text(encoding="utf-8"))
    fi = Path(bundle_dir) / "fit_info.json"
    fit_fy = json.loads(fi.read_text(encoding="utf-8")).get("first_year") if fi.exists() else None
    if fit_fy is not None and int(fit_fy) != int(first_year):
        raise SystemExit(f"{fi}: first_year={fit_fy} が書かれている（{first_year} と食い違う）")
    have = meta.get("first_year")
    if have is not None and int(have) != int(first_year):
        raise SystemExit(f"{p}: 既に first_year={have} が書かれている（{first_year} で上書きしない）")
    have_min = meta.get("train_year_min")
    if train_year_min is not None and have_min is not None and int(have_min) != int(train_year_min):
        raise SystemExit(f"{p}: 既に train_year_min={have_min} が書かれている（{train_year_min} で上書きしない）")
    add = {"first_year": int(first_year)}
    if train_year_min is not None:
        add["train_year_min"] = int(train_year_min)
    if "first_year_source" not in meta:          # 元の出どころ（"env"/"default"）が書かれていれば残す
        add["first_year_source"] = "stamp"
    would_change = any(meta.get(k) != v for k, v in add.items())
    if would_change and not dry_run:
        meta.update(add)
        tmp = p.with_suffix(".json.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(json.dumps(meta, ensure_ascii=False, indent=1))
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(p)
        finally:
            tmp.unlink(missing_ok=True)
    return {"path": str(p), "would_change": would_change, "changed": would_change and not dry_run, **add}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle-dir", required=True)
    ap.add_argument("--first-year", type=int, required=True)
    ap.add_argument("--train-year-min", type=int)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    print(json.dumps(stamp(Path(a.bundle_dir), a.first_year, a.train_year_min, a.dry_run), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
