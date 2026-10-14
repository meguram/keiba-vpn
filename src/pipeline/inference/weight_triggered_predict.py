"""馬体重の検知後に呼ぶ、当日の特徴量作成（推論モード）と予測の起動。

``raceday-weight-runner`` が各レースの馬体重を検知したら ``run_for_races`` を呼ぶ。

  1. 保存済みの出馬表（``race_shutuba``）から当日の出走馬の行を作る
  2. 履歴（``race_result_flat``）の末尾に、結果欄がプレースホルダーの走として足し、学習と同じ計算で特徴量を作る
     （``horse_history.inference``。履歴の読み込みは ``InferenceContext.warm()`` でランナー起動時に 1 回だけ行う）
  3. 特徴量を ``data/local/inference/features/<日付>/<race_id>.parquet`` に保存し、イベントを ``events/<日付>.jsonl`` に追記する
  4. ``KEIBA_WEIGHT_TRIGGER_MODE=predict`` のときだけ、既存の予測パイプライン（``race_day_workflow.predict_race``）も呼ぶ

最終モデル（tp-04 以降）の学習が済むまでは、既定の ``features`` で特徴量の作成までを行う。
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

logger = logging.getLogger("pipeline.inference.weight_triggered_predict")

_JST = ZoneInfo("Asia/Tokyo")
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LAYER_A = "data/local/modeling/training_pipeline/layer_a.parquet"


def inference_dir() -> Path:
    return Path(os.environ.get("KEIBA_INFERENCE_DIR") or REPO_ROOT / "data" / "local" / "inference")


def trigger_mode() -> str:
    m = (os.environ.get("KEIBA_WEIGHT_TRIGGER_MODE") or "features").strip().lower()
    return m if m in ("features", "predict") else "features"


class InferenceContext:
    """履歴と補修表を保持する。ランナーの起動時に ``warm()`` で読み、以降は特徴量の計算だけを行う。"""

    def __init__(self, tables_dir: str | Path | None = None, layer_a: str | Path | None = None, years: list[int] | None = None):
        self.tables_dir = tables_dir
        self.layer_a = Path(layer_a) if layer_a else REPO_ROOT / DEFAULT_LAYER_A
        self.years = years
        self.raw_hist: pl.DataFrame | None = None
        self.repair_map: pl.DataFrame | None = None
        self.history_last_date: str | None = None
        self.graded_names: dict[str, str] = {}

    def warm(self) -> dict:
        from src.pipeline.features.horse_history.common import build_repair_map, read_runs_raw

        t0 = time.perf_counter()
        years = self.years or list(range(2020, datetime.now(_JST).year + 1))
        self.raw_hist = read_runs_raw(self.tables_dir, years).filter(pl.col("finish_position") > 0)
        if self.layer_a.exists():
            self.repair_map = build_repair_map(self.layer_a)
        else:
            logger.warning("layer_a が無いので last_3f・馬体重の補修表なしで進める: %s", self.layer_a)
            self.repair_map = pl.DataFrame(schema={"horse_id": pl.Utf8, "run_date": pl.Date, "l3f_fix": pl.Float64,
                                                   "weight_fix": pl.Float64, "src_date": pl.Date})
        self.history_last_date = str(self.raw_hist["date"].max())
        from src.pipeline.features.horse_history.inference import graded_race_names

        self.graded_names = graded_race_names(self.tables_dir, years)
        info = {"history_rows": self.raw_hist.height, "history_last_date": self.history_last_date,
                "warm_sec": round(time.perf_counter() - t0, 1)}
        logger.info("推論用の履歴を読み込み: %s", info)
        return info

    def build(self, card: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
        from src.pipeline.features.horse_history.inference import build_inference_features

        if self.raw_hist is None:
            self.warm()
        return build_inference_features(self.raw_hist, card, self.repair_map, graded_names=self.graded_names)


def card_frame(storage: Any, race_ids: list[str]) -> pl.DataFrame:
    """保存済みの ``race_shutuba`` から、``race_shutuba_flat`` と同じ形の行を作る。"""
    from src.scraper.export_tables import _data_to_flat_rows
    from src.pipeline.features.horse_history.inference import CARD_COLS

    rows: list[dict] = []
    for rid in race_ids:
        data = storage.load("race_shutuba", rid, bypass_cache=True)  # 取得直後の最新を読む
        if not data:
            logger.warning("出馬表が保存されていない: %s", rid)
            continue
        data = {**data, "race_id": data.get("race_id") or rid}
        rows.extend(_data_to_flat_rows(data, "race_shutuba"))
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame([{c: r.get(c) for c in CARD_COLS + ["race_name"]} for r in rows], infer_schema_length=None)
    ints = [c for c in ("distance", "horse_number", "weight") if c in df.columns]
    return df.with_columns([pl.col(c).cast(pl.Int64, strict=False) for c in ints]
                           + [pl.col("jockey_weight").cast(pl.Float64, strict=False)])


def _append_event(day: str, event: dict) -> None:
    d = inference_dir() / "events"
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"{day}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def run_for_races(
    races: list[dict], storage: Any, ctx: InferenceContext, *, weights_known: dict[str, bool] | None = None, mode: str | None = None
) -> list[dict]:
    """``races``（発走時刻表の行）の特徴量を 1 回の計算でまとめて作る。結果はレースごとのイベント（dict）の一覧。"""
    mode = mode or trigger_mode()
    weights_known = weights_known or {}
    ids = [r["race_id"] for r in races]
    now = datetime.now(_JST)
    events: list[dict] = []
    try:
        card = card_frame(storage, ids)
        if card.height == 0:
            raise ValueError("出馬表の行がありません")
        # 馬体重は出走馬全頭に同時に発表される。揃ったレースで体重の無い馬は、出走取消・除外とみなして出走馬から外す。
        announced = [rid for rid, k in weights_known.items() if k]
        no_weight = (pl.col("race_id").is_in(announced)) & ~(pl.col("weight").cast(pl.Float64, strict=False) > 0).fill_null(False)
        dropped = {r[0]: r[1] for r in card.filter(no_weight).group_by("race_id").len().iter_rows()} if announced else {}
        if dropped:
            logger.info("[weight-trigger] 馬体重のない馬を除外: %s", dropped)
            card = card.filter(~no_weight)
        F, info = ctx.build(card)
    except Exception as exc:
        logger.exception("特徴量の作成に失敗: %s", ids)
        for r in races:
            ev = {"race_id": r["race_id"], "at": now.isoformat(timespec="seconds"), "status": "error", "error": str(exc), "mode": mode}
            _append_event(now.strftime("%Y%m%d"), ev)
            events.append(ev)
        return events

    for r in races:
        rid = r["race_id"]
        day = str(F.filter(pl.col("race_id") == rid)["date"].min() or now.date()).replace("-", "")
        out_dir = inference_dir() / "features" / day
        out_dir.mkdir(parents=True, exist_ok=True)
        part = F.filter(pl.col("race_id") == rid)
        part.write_parquet(out_dir / f"{rid}.parquet")
        ev = {
            "race_id": rid, "at": now.isoformat(timespec="seconds"), "status": "ok", "mode": mode, "rows": part.height,
            "weights_known": bool(weights_known.get(rid, info["weights_known_ratio"] >= 0.9)),
            "going_known": info["going_known"], "build_sec": info["build_sec"], "batch_races": info["races"],
            "history_last_date": ctx.history_last_date, "dropped_no_weight": int(dropped.get(rid, 0)),
        }
        if mode == "predict":
            try:
                from src.pipeline.inference.race_day_workflow import predict_race

                res = predict_race(rid, storage, source="weight_t60")
                ev["predict_status"] = res.get("status")
                ev["predict_error"] = res.get("error")
            except Exception as exc:
                logger.exception("予測の起動に失敗: %s", rid)
                ev["predict_status"], ev["predict_error"] = "error", str(exc)
        _append_event(day, ev)
        events.append(ev)
        logger.info("[weight-trigger] %s ok rows=%d weights_known=%s build=%.1fs", rid, part.height, ev["weights_known"], info["build_sec"])
    return events
