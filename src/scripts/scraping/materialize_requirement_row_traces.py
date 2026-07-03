"""
要件書 netkeiba 表の「1 行 = 1 JSON」参照（``requirement_row_trace``）と
発走時刻スナップショット（``race_day_schedule``）をマテリアライズする。

- ``requirement_row_trace``: GCS ``chuou/data/others/requirement_row_trace/{trace_key}.json``
  （正本 JSON/HTML を分割せず、HybridStorage / HtmlArchive へのポインタのみ）
- ``race_day_schedule``: ``data/page_reference/race_day_schedule/{YYYYMMDD}.json``（local_only）

既存 GCS の canonical を書き換えず、追記・上書きするだけなのでバックフィルに安全。

使用例::

    # ドキュメントのサンプル ID で trace 全行 + 発走表 1 日
    python3 -m src.scripts.scraping.materialize_requirement_row_traces --doc-sample

    # 開催日だけ発走表（race_lists が読める環境で）
    python3 -m src.scripts.scraping.materialize_requirement_row_traces --schedule-date 20230625

    # 任意の race / horse / date で trace のみ（GCS 要）
    python3 -m src.scripts.scraping.materialize_requirement_row_traces \\
        --race-id 202309030811 --horse-id 2019105219 --date 20230625 --traces-only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from src.scraper.requirement_row_catalog import (
    NETKEIBA_REQUIREMENT_ROWS,
    build_trace_payload,
    probe_canonical_presence,
    resolve_storage_key,
)
from src.scraper.race_day_schedule import synthesize_race_day_schedule_payload
from src.scraper.storage import HybridStorage

logger = logging.getLogger(__name__)

_DOC_RACE = "202309030811"
_DOC_HORSE = "2019105219"
_DOC_DATE = "20230625"


def _probe_raw_presence(
    archive,
    spec,
    *,
    race_id: str,
    horse_id: str,
    date_fmt: str,
) -> dict[tuple[str, str], bool]:
    out: dict[tuple[str, str], bool] = {}
    if archive is None:
        return out
    exists_fn = getattr(archive, "exists", None)
    if not callable(exists_fn):
        return out
    for h in spec.raw_html:
        k = resolve_storage_key(
            h.key_field, race_id=race_id, horse_id=horse_id, date_fmt=date_fmt
        )
        if h.role == "paged" and h.key_field == "horse_id":
            k = f"{horse_id}_p1"
        try:
            out[(h.category, k)] = bool(exists_fn(h.category, k))
        except Exception:
            out[(h.category, k)] = False
    return out


def materialize_schedules(storage: HybridStorage, dates: list[str], *, dry_run: bool) -> int:
    n = 0
    for d in dates:
        payload = synthesize_race_day_schedule_payload(storage, d)
        if dry_run:
            logger.info("[dry-run] race_day_schedule/%s slots=%d", d, len(payload.get("slots") or []))
            n += 1
            continue
        storage.save("race_day_schedule", d, payload)
        logger.info("saved race_day_schedule/%s slots=%d", d, len(payload.get("slots") or []))
        n += 1
    return n


def materialize_traces(
    storage: HybridStorage,
    archive,
    *,
    race_id: str,
    horse_id: str,
    date_fmt: str,
    dry_run: bool,
    gcs_traces: bool,
) -> int:
    n = 0
    for spec in NETKEIBA_REQUIREMENT_ROWS():
        pc = probe_canonical_presence(storage, spec, race_id, horse_id, date_fmt)
        pr = _probe_raw_presence(
            archive, spec, race_id=race_id, horse_id=horse_id, date_fmt=date_fmt
        )
        presence = {**pc, **pr}
        payload = build_trace_payload(
            spec,
            race_id=race_id,
            horse_id=horse_id,
            date_fmt=date_fmt,
            presence=presence,
        )
        tk = str(payload["trace_key"])
        if dry_run:
            logger.info("[dry-run] requirement_row_trace/%s", tk)
            n += 1
            continue
        if gcs_traces and storage.gcs_enabled:
            storage.save("requirement_row_trace", tk, payload)  # type: ignore[arg-type]
            logger.info("saved requirement_row_trace/%s", tk)
        elif gcs_traces:
            logger.warning("GCS 無効のため trace スキップ: %s", tk)
        n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", default=".", help="リポジトリルート（HybridStorage の base）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--doc-sample",
        action="store_true",
        help=f"race_id={_DOC_RACE} horse_id={_DOC_HORSE} date={_DOC_DATE} で trace + schedule",
    )
    ap.add_argument("--race-id", default="", help="trace 用レース ID")
    ap.add_argument("--horse-id", default="", help="trace 用馬 ID")
    ap.add_argument("--date", default="", help="YYYYMMDD（race_lists / schedule）")
    ap.add_argument(
        "--schedule-date",
        action="append",
        default=[],
        metavar="YYYYMMDD",
        help="発走表だけ書く（複数指定可）",
    )
    ap.add_argument(
        "--traces-only",
        action="store_true",
        help="race_day_schedule は書かず requirement_row_trace のみ",
    )
    ap.add_argument(
        "--no-gcs-traces",
        action="store_true",
        help="trace を GCS に送らず dry-run 相当のログのみ（ローカル検証用）",
    )
    args = ap.parse_args(argv)

    base = os.path.abspath(args.base_dir)
    storage = HybridStorage(base_dir=base)
    try:
        from src.scraper.html_archive import HtmlArchive

        archive = HtmlArchive()
    except Exception as e:
        logger.warning("HtmlArchive 初期化に失敗（raw_html.present は未検査）: %s", e)
        archive = None

    race_id = args.race_id or _DOC_RACE
    horse_id = args.horse_id or _DOC_HORSE
    date_fmt = args.date or _DOC_DATE

    if args.doc_sample:
        race_id, horse_id, date_fmt = _DOC_RACE, _DOC_HORSE, _DOC_DATE

    dates = list(args.schedule_date)
    if args.doc_sample and date_fmt not in dates:
        dates.append(date_fmt)

    gcs_traces = not args.no_gcs_traces

    if dates and not args.traces_only:
        materialize_schedules(storage, dates, dry_run=args.dry_run)

    want_traces = (
        args.doc_sample
        or args.traces_only
        or bool(args.race_id or args.horse_id or args.date)
    )
    if want_traces:
        materialize_traces(
            storage,
            archive,
            race_id=race_id,
            horse_id=horse_id,
            date_fmt=date_fmt,
            dry_run=args.dry_run,
            gcs_traces=gcs_traces and not args.dry_run,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
