"""開催日 T-15 プレレース取得バンドル + 完了後 AI 予測トリガ（モック含む）。

運用方針（docs/html/operations/scraping_sla.html と整合）:
  - 出馬表 / 馬柱・調教 / 追い切りは **前日 18:00 の raceday-eve** で取得する。
    T-15 バンドルでは skip_existing=True のため、取得済みならスキップ（フォールバック用に残す）。
  - オッズ / JRA 馬場は **発走 T-15 分** で取得（直前に確定するデータ）。
  - JRA 馬場ライブ (jra_baba_live) は **同じタイミング** に統合（別 systemd 5 分ポーリングは廃止方向）。
  - AI 予測は **T-15 バッチ完了** をトリガに起動（発走 10 分前時点の予測物として扱う）。

本モジュールは最終版の足場。`KEIBA_PRE_RACE_PREDICT_ENABLED=1` で実予測を有効化可能。

Usage:
    from src.scraper.raceday_pre_race_pipeline import run_t15_pre_race_bundle, trigger_pre_race_predict

    result = run_t15_pre_race_bundle(runner, race_dict)
    trigger_pre_race_predict(race_dict, bundle_result=result)
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("scraper.raceday_pre_race_pipeline")

# T-15 バンドル内での JRA 馬場再取得の最小間隔（秒）。同一バッチで複数 R を処理するときの重複抑制。
_BABA_MIN_INTERVAL_SEC = float(os.environ.get("KEIBA_BABA_REFRESH_MIN_SEC", "120"))
_last_baba_refresh_ts: float = 0.0


@dataclass
class PreRaceBundleResult:
    """1 レース分の T-15 バンドル実行結果。"""

    race_id: str
    ok: bool = True
    netkeiba_tasks: dict[str, bool] = field(default_factory=dict)
    jra_baba_refreshed: bool = False
    errors: list[str] = field(default_factory=list)
    elapsed_sec: float = 0.0


def refresh_jra_baba_live(*, force: bool = False) -> bool:
    """JRA 馬場ライブを 1 回取得。直近で取得済みならスキップ（force で強制）。"""
    global _last_baba_refresh_ts
    now = time.time()
    if not force and (now - _last_baba_refresh_ts) < _BABA_MIN_INTERVAL_SEC:
        logger.debug("JRA 馬場ライブ: 直近 %.0fs 以内のためスキップ", now - _last_baba_refresh_ts)
        return False
    try:
        from src.scraper.jra_baba_live import JRABabaLiveScraper

        records = JRABabaLiveScraper().scrape()
        _last_baba_refresh_ts = now
        logger.info("JRA 馬場ライブ: %d レコード更新", len(records))
        return True
    except Exception as e:
        logger.warning("JRA 馬場ライブ取得失敗: %s", e)
        return False


def run_t15_pre_race_bundle(runner: Any, race: dict) -> PreRaceBundleResult:
    """発走 T-15 分相当の netkeiba + JRA 馬場を 1 レース分取得する。"""
    rid = str(race.get("race_id") or "")
    t0 = time.time()
    out = PreRaceBundleResult(race_id=rid)

    tasks: list[tuple[str, Callable[[], Any]]] = [
        ("出馬表",    lambda: runner.scrape_race_card(rid, skip_existing=True)),
        ("単複オッズ", lambda: runner.scrape_odds(rid, skip_existing=False)),
        ("2連オッズ", lambda: runner.scrape_pair_odds(rid, skip_existing=False)),
        # 馬柱・調教 / 追い切りは前日 raceday-eve で取得済みが前提。
        # 前日取得に失敗した場合のフォールバックとして skip_existing=True で残す。
        ("馬柱・調教", lambda: runner.scrape_shutuba_past(rid, skip_existing=True)),
        ("追い切り",  lambda: runner.scrape_oikiri(rid, skip_existing=True)),
        ("SmartRC",  lambda: runner.scrape_smartrc(rid)),
    ]

    for name, fn in tasks:
        try:
            result = fn()
            out.netkeiba_tasks[name] = bool(result)
            if not result:
                logger.warning("  - %s (%s): データなし", name, rid)
        except Exception as e:
            out.netkeiba_tasks[name] = False
            out.errors.append(f"{name}:{e}")
            logger.error("  ✗ %s (%s): %s", name, rid, e)

    out.jra_baba_refreshed = refresh_jra_baba_live()
    out.ok = not out.errors
    out.elapsed_sec = time.time() - t0
    return out


def trigger_pre_race_predict(
    race: dict,
    *,
    bundle_result: PreRaceBundleResult | None = None,
) -> None:
    """T-15 バンドル完了後に AI 予測トリガを起動（デフォルトはモック）。"""
    from src.pipeline.inference.pre_race_predict_trigger import (
        PreRacePredictTrigger,
        TriggerConfig,
    )

    enabled = os.environ.get("KEIBA_PRE_RACE_PREDICT_ENABLED", "").strip() in (
        "1",
        "true",
        "yes",
    )
    cfg = TriggerConfig(
        mock=not enabled,
        use_fresh_scrape=enabled,
    )
    PreRacePredictTrigger(cfg).on_pre_race_scrape_complete(race, bundle_result=bundle_result)
