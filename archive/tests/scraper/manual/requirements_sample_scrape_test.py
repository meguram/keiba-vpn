#!/usr/bin/env python3
"""
docs/requirements/data/scrape_process.md のサンプル
(race_date=20230625, race_id=202309030811, horse_id=2019105219) で
データ取得要件に対応する ScraperRunner 等の取得をスモークする。

前提: リポジトリルートで実行し、.env に netkeiba ログイン情報があること。
  python3 tests/scraper/manual/requirements_sample_scrape_test.py

--quick: skip_existing=True のみ（既存ストレージがあればネット省略）
--json:  結果を JSON のみ stdout（マークダウン列更新用）
--export-samples: 各取得の戻り値 dict を `docs/requirements/data/scrape_process_samples/*.json` に保存し、
  `scrape_process.md` の折りたたみサンプルを再生成（未取得カテゴリは省略されるので、初回は --quick なし推奨）

取得後に `src.scraper.schemas.validate` を可能なカテゴリで実行し、detail に `schema=PASS` / `schema=FAIL ...` を付与する。
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

# リポジトリルート
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.WARNING)


@dataclass
class RowResult:
    key: str
    status: str  # PASS | FAIL | WARN | SKIP | NA
    detail: str = ""

    def to_cell(self) -> str:
        s = f"{self.status}"
        if self.detail:
            s += f" — {self.detail}"
        return s


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    p = REPO_ROOT / ".env"
    if p.is_file():
        load_dotenv(p)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="skip_existing=True のみ試行")
    parser.add_argument("--json", action="store_true", help="JSON のみ出力")
    parser.add_argument(
        "--export-samples",
        action="store_true",
        help="実データ JSON を docs/requirements/data/scrape_process_samples/ に書き、scrape_process.md を更新",
    )
    args = parser.parse_args()
    force_fetch = not args.quick
    skip_existing = not force_fetch

    _load_dotenv()

    from src.scraper import schemas
    from src.scraper.auto_scrape import _fetch_race_schedule
    from src.scraper.jra_baba_live import JRABabaLiveScraper
    from src.scraper.run import ScraperRunner

    race_date = "20230625"
    race_id = "202309030811"
    horse_id = "2019105219"

    results: list[RowResult] = []
    samples: dict[str, dict] = {}
    samples_dir = REPO_ROOT / "docs" / "requirements" / "data" / "scrape_process_samples"

    def record_sample(schema_category: str | None, data: object) -> None:
        """--export-samples 用: 戻り値 dict をカテゴリ別に保持（horse_result は race_history が長い方を優先）。"""
        if not args.export_samples or not schema_category or not isinstance(data, dict):
            return
        snap = copy.deepcopy(data)
        if schema_category == "horse_result":
            prev = samples.get(schema_category)
            if prev is None or len(snap.get("race_history") or []) > len(
                prev.get("race_history") or []
            ):
                samples[schema_category] = snap
            return
        samples[schema_category] = snap

    runner = ScraperRunner(interval=1.0, cache=False, auto_login=True)
    logged_in = runner.client.login()

    def add(
        key: str,
        fn,
        ok_check,
        schema_category: str | None = None,
    ) -> None:
        try:
            data = fn()
            record_sample(schema_category, data)
            ok, detail = ok_check(data)
            if ok:
                st = "PASS"
            elif data is not None:
                st = "WARN"
            else:
                st = "FAIL"
            if schema_category and isinstance(data, dict):
                vr = schemas.validate(schema_category, data)
                if vr.get("passed"):
                    detail = f"{detail}; schema=PASS(v{vr.get('schema_version')})"
                else:
                    detail = f"{detail}; schema=FAIL {vr}"
                    if ok:
                        st = "WARN"
            elif schema_category:
                detail = f"{detail}; schema=SKIP(non-dict)"
            results.append(RowResult(key, st, detail))
        except Exception as e:
            results.append(RowResult(key, "FAIL", str(e)[:200]))

    # --- netkeiba 表と同順（タイトルキー） ---
    add(
        "出馬表HTML",
        lambda: runner.scrape_race_card(race_id, skip_existing=skip_existing),
        lambda d: (
            bool(d and d.get("entries")),
            f"entries={len((d or {}).get('entries') or [])}",
        ),
        "race_shutuba",
    )
    add(
        "レース情報HTML",
        lambda: runner.scrape_race_card(race_id, skip_existing=skip_existing),
        lambda d: (
            bool(d and (d.get("race_name") or d.get("venue"))),
            f"race_name={bool((d or {}).get('race_name'))}",
        ),
        "race_shutuba",
    )
    add(
        "レースタイム指数HTML",
        lambda: runner.scrape_speed_index(race_id, skip_existing=skip_existing),
        lambda d: (
            bool(d and d.get("entries")),
            f"entries={len((d or {}).get('entries') or [])}",
        ),
        "race_index",
    )
    add(
        "レース調子偏差値HTML",
        lambda: runner.scrape_barometer(race_id, skip_existing=skip_existing),
        lambda d: (
            bool(d and d.get("entries")),
            f"entries={len((d or {}).get('entries') or [])}",
        ),
        "race_barometer",
    )
    add(
        "レースパドックHTML",
        lambda: runner.scrape_paddock(race_id, skip_existing=skip_existing),
        lambda d: (
            bool(d and d.get("entries")),
            f"entries={len((d or {}).get('entries') or [])}",
        ),
        "race_paddock",
    )
    add(
        "レースオッズHTML",
        lambda: runner.scrape_odds(race_id, skip_existing=skip_existing),
        lambda d: (
            bool(d and d.get("entries")),
            f"entries={len((d or {}).get('entries') or [])}",
        ),
        "race_odds",
    )

    on_time = {"ref": None}

    def load_on_time():
        on_time["ref"] = runner.scrape_race_result_on_time(
            race_id, skip_existing=skip_existing, opening_date=race_id[:8]
        )
        return on_time["ref"]

    add(
        "レース結果HTML",
        load_on_time,
        lambda d: (
            bool(d and d.get("entries")),
            f"entries={len((d or {}).get('entries') or [])}",
        ),
        "race_result_on_time",
    )
    add(
        "レース払戻HTML",
        load_on_time,
        lambda d: (
            bool(d and isinstance(d.get("payoff"), dict) and len(d.get("payoff") or {}) > 0),
            f"payoff_keys={len((d or {}).get('payoff') or {})}",
        ),
        "race_result_on_time",
    )
    add(
        "レースラップHTML",
        load_on_time,
        lambda d: (
            bool(d and (d.get("lap_times") or [])),
            f"lap_times={len((d or {}).get('lap_times') or [])}",
        ),
        "race_result_on_time",
    )
    add(
        "レース通過順位HTML",
        load_on_time,
        lambda d: (
            bool(d and (d.get("corner_passing") or [])),
            f"corner_passing={len((d or {}).get('corner_passing') or [])}",
        ),
        "race_result_on_time",
    )

    add(
        "レース個別ラップHTML",
        lambda: runner.scrape_race_result_lap(race_id, skip_existing=skip_existing),
        lambda d: (
            bool(
                d
                and (
                    (d.get("entries_lap") or [])
                    or (d.get("lap_times") or [])
                )
            ),
            f"entries_lap={len((d or {}).get('entries_lap') or [])} lap_times={len((d or {}).get('lap_times') or [])}",
        ),
        "race_result_lap",
    )

    add(
        "馬プロフィール",
        lambda: runner.scrape_horse(
            horse_id, skip_existing=skip_existing, with_history=False, skip_pedigree=True
        ),
        lambda d: (bool(d and d.get("horse_name")), f"name={(d or {}).get('horse_name', '')[:20]!r}"),
        "horse_result",
    )
    add(
        "馬過去成績",
        lambda: runner.scrape_horse(
            horse_id, skip_existing=skip_existing, with_history=True, skip_pedigree=True
        ),
        lambda d: (
            bool(d and (d.get("race_history") or [])),
            f"race_history={len((d or {}).get('race_history') or [])}",
        ),
        "horse_result",
    )
    add(
        "馬血統データ",
        lambda: runner.scrape_horse_pedigree_5gen(horse_id, skip_existing=skip_existing),
        lambda d: (
            bool(d and len(d.get("ancestors") or []) >= 5),
            f"ancestors={len((d or {}).get('ancestors') or [])}",
        ),
        "horse_pedigree_5gen",
    )
    add(
        "馬調教",
        lambda: runner.scrape_horse_training(
            horse_id, skip_existing=skip_existing, max_pages=2, force=False, smart_skip=False
        ),
        lambda d: (
            d is not None and isinstance((d or {}).get("entries"), list),
            f"entries={len((d or {}).get('entries') or [])}",
        ),
        "horse_training",
    )

    add(
        "レースID一覧",
        lambda: {"date": race_date, "races": runner.scrape_race_list(race_date)},
        lambda d: (
            bool(isinstance(d, dict) and len(d.get("races") or []) > 0),
            f"n_races={len((d or {}).get('races') or [])}",
        ),
        "race_lists",
    )

    add(
        "レース発走時間",
        lambda: _fetch_race_schedule(runner, race_date),
        lambda sched: (
            bool(isinstance(sched, list) and len(sched or []) > 0),
            f"n_slots={len(sched or [])}",
        ),
    )

    add(
        "レース結果DB",
        lambda: runner.scrape_race_result(race_id, skip_existing=skip_existing),
        lambda d: (
            bool(d and d.get("entries")),
            f"entries={len((d or {}).get('entries') or [])}",
        ),
        "race_result",
    )
    add(
        "レース情報DB",
        lambda: runner.scrape_race_result(race_id, skip_existing=skip_existing),
        lambda d: (
            bool(d and (d.get("race_name") or d.get("track_condition"))),
            "race_meta_ok",
        ),
        "race_result",
    )
    add(
        "レース払戻DB",
        lambda: runner.scrape_race_result(race_id, skip_existing=skip_existing),
        lambda d: (
            bool(d and d.get("payoff")),
            f"payoff_keys={len((d or {}).get('payoff') or {})}",
        ),
        "race_result",
    )
    add(
        "レース馬場情報DB",
        lambda: runner.scrape_race_result(race_id, skip_existing=skip_existing),
        lambda d: (
            bool(d and d.get("track_condition")),
            f"track_condition={((d or {}).get('track_condition') or '')!r}",
        ),
        "race_result",
    )
    add(
        "レース通過順位DB",
        lambda: runner.scrape_race_result_lap(race_id, skip_existing=skip_existing),
        lambda d: (
            bool(d and (d.get("corner_passing") or [])),
            f"corner_passing={len((d or {}).get('corner_passing') or [])}",
        ),
        "race_result_lap",
    )
    add(
        "レースラップDB",
        lambda: runner.scrape_race_result_lap(race_id, skip_existing=skip_existing),
        lambda d: (
            bool(d and (d.get("lap_times") or [])),
            f"lap_times={len((d or {}).get('lap_times') or [])}",
        ),
        "race_result_lap",
    )
    add(
        "レース個別ラップDB",
        lambda: runner.scrape_race_result_lap(race_id, skip_existing=skip_existing),
        lambda d: (
            bool(d and (d.get("entries_lap") or [])),
            f"entries_lap={len((d or {}).get('entries_lap') or [])}",
        ),
        "race_result_lap",
    )

    results.append(RowResult("separator", "NA", "表の区切り行"))

    # JRA
    try:
        jra = JRABabaLiveScraper(output_dir=str(REPO_ROOT / "data" / "page_reference" / "cushion"))
        jdata = jra.scrape()
        results.append(
            RowResult(
                "馬場情報(JRA)",
                "PASS" if jdata else "WARN",
                f"records={len(jdata or [])}",
            )
        )
    except Exception as e:
        results.append(RowResult("馬場情報(JRA)", "FAIL", str(e)[:200]))

    results.append(RowResult("smartrc", "NA", "運用中止（要件）"))

    out = [
        {
            "key": r.key,
            "status": r.status,
            "detail": r.detail,
            "cell": r.to_cell(),
        }
        for r in results
    ]
    out.insert(
        0,
        {
            "key": "_meta",
            "status": "PASS" if logged_in else "FAIL",
            "detail": "NetkeibaClient.login()" + (" OK" if logged_in else " failed"),
            "cell": ("ログイン済" if logged_in else "ログイン失敗"),
        },
    )
    if args.export_samples:
        samples_dir.mkdir(parents=True, exist_ok=True)
        for p in samples_dir.glob("*.json"):
            p.unlink()
        for stem in sorted(samples):
            p = samples_dir / f"{stem}.json"
            p.write_text(
                json.dumps(samples[stem], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"export: {p.relative_to(REPO_ROOT)}")
        from src.scripts.docs.gen_scrape_process_samples import patch_scrape_process_doc

        patch_scrape_process_doc(samples_dir)
        print(
            "scrape_process.md のサンプル範囲を更新しました（未取得カテゴリは HTML コメントで欠損記載）。"
        )

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    for r in results:
        print(f"{r.key}\t{r.status}\t{r.detail}")
    return 0 if logged_in else 2


if __name__ == "__main__":
    raise SystemExit(main())
