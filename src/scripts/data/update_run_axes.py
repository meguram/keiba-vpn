"""めぐ分析の走法の軸（data/local/running_profile/run_axes.parquet）を日次で更新する（ホスト。毎朝の export_tables_daily.sh から）。

python -m src.scripts.data.update_run_axes [--dry-run]

1. 足りないレースを決める（ローカルだけ。GCS の一覧は取らない）:
   ローカルの race_result_flat（平地・着順あり）のうち、馬別ラップの公開（バロメーターと同じ「レース翌週の金曜 18 時」）の翌日から
   21 日の窓にあり、jsonl（horse_lap_<年>.jsonl）にまだ頭数のそろった行が無いもの。試行の間隔・上限は export_tables と同じ記録を使う。
2. GCS の horse_lap（バロメーターの取得が同じページから保存する。run.py::_save_horse_lap_from_html）を名前で読み、
   結果表と頭数がそろったものだけ jsonl に追記する（backfill_horse_lap と同じ形）。netkeiba へは取りに行かない。
3. 追記があったとき（または run_axes.parquet が無いとき）だけ、全件で run_axes を計算し直して置き換える
   （各軸は全期間の統計に依存するので増分では計算しない。全件で約 16 秒・約 650MB）。

jsonl の場所: KEIBA_HORSE_LAP_DIR（既定 /home/jovyan/acq/horse_lap）。2023 年の重賞の jsonl は KEIBA_HORSE_LAP_EXTRA（既定 <親>/horse_lap23/horse_lap_2023.jsonl）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import date, timedelta
from pathlib import Path

from src.scripts.data import backfill_horse_lap as bhl

LAP_DIR = Path(os.environ.get("KEIBA_HORSE_LAP_DIR") or "/home/jovyan/acq/horse_lap")
WINDOW_DAYS = 21
CATEGORY = "horse_lap"


def lap_files(lap_dir: Path = LAP_DIR) -> list[Path]:
    """計算に使う jsonl（horse_lap_<年>.jsonl と 2023 年の重賞分）。退避した incomplete_* は使わない。"""
    files = sorted(p for p in lap_dir.glob("horse_lap_*.jsonl") if re.fullmatch(r"horse_lap_\d{4}\.jsonl", p.name))
    extra = Path(os.environ.get("KEIBA_HORSE_LAP_EXTRA") or lap_dir.parent / "horse_lap23" / "horse_lap_2023.jsonl")
    if extra.exists() and extra not in files:
        files.append(extra)
    return files


def candidates(today: date, years: list[int], expect: dict[str, int], done: set[str], attempts: dict[str, list]) -> list[tuple[int, str]]:
    from src.research.race import running_profile as rp
    from src.scraper.export_tables import _attempt_due
    from src.scraper.run import ScraperRunner

    out: list[tuple[int, str]] = []
    for y in years:
        runs = rp.load_runs([y])
        if runs.empty:
            continue
        r = runs.drop_duplicates("race_id")
        for rid, d in zip(r["race_id"].astype(str), r["date"]):
            if rid in done or not expect.get(rid):
                continue
            start = ScraperRunner.barometer_publish_at(d.date()).date() + timedelta(days=1)
            if start <= today <= start + timedelta(days=WINDOW_DAYS) and _attempt_due(attempts.get(rid), today):
                out.append((y, rid))
    return sorted(out)


def append_new(today: date, years: list[int], storage, lap_dir: Path = LAP_DIR, dry_run: bool = False) -> dict:
    from src.scraper import export_tables as et

    expect = bhl.expected_counts(years)
    done = bhl.done_ids([lap_dir / f"horse_lap_{y}.jsonl" for y in years], expect)
    stats = {"candidates": 0, "appended": 0, "not_found": 0, "incomplete": 0}
    for y in years:
        attempts = et._load_attempts(CATEGORY, str(y))
        todo = [rid for yy, rid in candidates(today, [y], expect, done, attempts) if yy == y]
        stats["candidates"] += len(todo)
        if dry_run or not todo:
            continue
        lines: list[str] = []
        failed: list[str] = []
        for rid in todo:
            d = storage.load(CATEGORY, rid)
            if not d:
                stats["not_found"] += 1
                if getattr(storage, "_gcs_healthy", True):     # 読めなかった直後に GCS が不調の印なら、一時的な失敗かもしれないので数えない
                    failed.append(rid)
                continue
            if not bhl.is_complete(d, expect):
                stats["incomplete"] += 1
                failed.append(rid)
                continue
            d = {k: v for k, v in d.items() if k != "_meta"}
            d["expected"] = expect[rid]
            lines.append(json.dumps(d, ensure_ascii=False) + "\n")
        if lines:
            _append_lines(lap_dir / f"horse_lap_{y}.jsonl", lines)
            stats["appended"] += len(lines)
        if failed:
            for rid in failed:
                attempts[rid] = [int(attempts.get(rid, [0, ""])[0]) + 1, today.isoformat()]
        live = {rid for _, rid in candidates(today, [y], expect, done | {json.loads(s)["race_id"] for s in lines}, {})}
        et._save_attempts(CATEGORY, str(y), {k: v for k, v in attempts.items() if k in live})
    return stats


def _append_lines(path: Path, lines: list[str]) -> None:
    """jsonl に追記する。新しく作ったファイル（年が替わったときなど）は、置き場所のディレクトリの持ち主に合わせる
    （cron は root で動くが、手で回す backfill_horse_lap は jovyan で同じファイルに追記するため）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", encoding="utf-8") as f:
        f.writelines(lines)
    if new and hasattr(os, "geteuid") and os.geteuid() == 0:
        try:
            st = path.parent.stat()
            os.chown(path, st.st_uid, st.st_gid)
        except OSError:
            pass


def recompute(lap_dir: Path = LAP_DIR, out: Path | None = None) -> int:
    from src.research.race import run_axes as ra
    from src.research.race import running_profile as rp

    out = out or rp.AXES_PATH
    df = ra.compute(lap_files(lap_dir))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    return len(df)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="走法の軸（run_axes）の日次更新")
    ap.add_argument("--dry-run", action="store_true", help="候補の数だけ出す（GCS を読まない・書かない）")
    a = ap.parse_args(argv)
    from src.research.race import running_profile as rp

    today = date.today()
    years = [today.year - 1, today.year] if today.month == 1 else [today.year]
    storage = None
    if not a.dry_run:
        from src.scraper.storage import HybridStorage

        storage = HybridStorage()
    st = append_new(today, years, storage, dry_run=a.dry_run)
    print(json.dumps(st, ensure_ascii=False))
    if a.dry_run:
        return 0
    if st["appended"] or not rp.AXES_PATH.exists():
        n = recompute()
        print(f"run_axes を作り直した: {n} 走 → {rp.AXES_PATH}")
    else:
        print("追記なし。run_axes はそのまま")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
