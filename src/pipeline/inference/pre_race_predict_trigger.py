"""T-15 プレレース取得完了をトリガとする AI 予測パイプライン（モック / 本番切替）。

最終版は後補完。現状:
  - デフォルト: ログのみ（モック）
  - KEIBA_PRE_RACE_PREDICT_ENABLED=1: 内部で予測ロジックを呼ぶ（API サーバ不要）

将来:
  - POST /api/race/{rid}/predict への HTTP 呼び出し
  - Slack / UI 通知
  - T-10 時刻までの待機が必要なら `post_time - 10min` でスケジュール
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger("pipeline.inference.pre_race_predict_trigger")


@dataclass
class TriggerConfig:
    mock: bool = True
    """True のとき予測本体は呼ばずログのみ。"""
    use_fresh_scrape: bool = False
    """本番時: predict 前に smart_skip=False 相当で再取得する想定（後補完）。"""
    api_base_url: str = ""
    """空ならプロセス内直呼び。将来 http://127.0.0.1:8000 等。"""


class PreRacePredictTrigger:
    """T-15 バンドル完了 → AI 予測のオーケストレーション。"""

    def __init__(self, config: TriggerConfig | None = None):
        self.config = config or TriggerConfig()

    def on_pre_race_scrape_complete(
        self,
        race: dict,
        *,
        bundle_result: Any = None,
    ) -> dict[str, Any]:
        rid = str(race.get("race_id") or "")
        label = self._race_label(race)
        now = datetime.now().strftime("%H:%M:%S")

        if self.config.mock:
            logger.info(
                "[MOCK pre_race_predict] %s | race_id=%s | T-15 バンドル完了をトリガに予測起動（未実行）",
                label,
                rid,
            )
            if bundle_result is not None:
                err_n = len(getattr(bundle_result, "errors", []) or [])
                baba = getattr(bundle_result, "jra_baba_refreshed", False)
                logger.info(
                    "  bundle: ok=%s errors=%d jra_baba=%s elapsed=%.1fs",
                    getattr(bundle_result, "ok", "?"),
                    err_n,
                    baba,
                    getattr(bundle_result, "elapsed_sec", 0),
                )
            return {
                "status": "mock",
                "race_id": rid,
                "triggered_at": now,
                "message": "KEIBA_PRE_RACE_PREDICT_ENABLED=1 で本番予測を有効化",
            }

        return self._run_prediction(rid, race)

    def _run_prediction(self, race_id: str, race: dict) -> dict[str, Any]:
        """本番予測: inference_pipeline 経由で DB / Redis / GCS へ保存。"""
        logger.info("[pre_race_predict] 本番予測開始 race_id=%s", race_id)
        try:
            from src.pipeline.inference.inference_pipeline import run_inference_for_race

            payload = run_inference_for_race(
                race_id,
                allow_scrape=self.config.use_fresh_scrape,
            )
            if not payload.get("horses") and payload.get("error"):
                return {"status": "error", "race_id": race_id, "error": payload.get("error")}
            return {
                "status": "ok",
                "race_id": race_id,
                "model_version": payload.get("model_version"),
                "horses": len(payload.get("horses") or []),
            }
        except Exception as exc:
            logger.exception("[pre_race_predict] 失敗 race_id=%s", race_id)
            return {"status": "error", "race_id": race_id, "error": str(exc)}

    @staticmethod
    def _race_label(race: dict) -> str:
        venue = race.get("venue", "")
        rnd = race.get("round", "")
        name = (race.get("race_name") or "")[:16]
        return f"{venue} {rnd}R {name}".strip()
