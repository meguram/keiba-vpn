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


def v3_mode() -> str:
    """off: 使わない / shadow: 予測をローカルに残すだけ（画面・GCS には出さない） / predict: 予測を保存して画面・API に出す。"""
    m = (os.environ.get("KEIBA_V3_SERVING") or "off").strip().lower()
    return m if m in ("off", "shadow", "predict") else "off"


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
        self.v3 = None

    def engine(self) -> str:
        """bundle: 出走馬の履歴だけを読む（小さなメモリで動く）。full: 全履歴から作る。auto: バンドルがあれば bundle。"""
        e = (os.environ.get("KEIBA_INFERENCE_ENGINE") or "auto").strip().lower()
        if e == "full":
            return "full"
        from src.pipeline.features.horse_history.serving import default_bundle_dir

        if (default_bundle_dir() / "meta.json").exists():
            return "bundle"
        if e == "bundle":
            raise FileNotFoundError(f"サービング用バンドルがありません: {default_bundle_dir()}（feature-accumulate を先に実行）")
        return "full"

    def warm(self) -> dict:
        info = self._warm_history()
        if v3_mode() != "off":
            try:
                from src.pipeline.inference.v3_serving import V3Serving

                self.v3 = V3Serving()
                info["v3"] = self.v3.warm()
            except Exception as exc:  # v3 が使えなくても履歴系の特徴量づくりは続ける
                logger.exception("v3 の準備に失敗: %s", exc)
                self.v3, info["v3"] = None, {"error": str(exc)}
        return info

    def _warm_history(self) -> dict:
        t0 = time.perf_counter()
        self.bundle = None
        if self.engine() == "bundle":
            from src.pipeline.features.horse_history.serving import Bundle, default_bundle_dir

            self.bundle = Bundle(default_bundle_dir())
            self.graded_names = self.bundle.graded_names
            self.history_last_date = self.bundle.meta["history_last_date"]
            info = {"engine": "bundle", "history_last_date": self.history_last_date, "bundle_built_at": self.bundle.meta["built_at"],
                    "warm_sec": round(time.perf_counter() - t0, 1)}
            age = (datetime.now(_JST).date() - datetime.fromisoformat(self.history_last_date).date()).days
            if age > 10:  # 確定結果の取得が遅れている（recent-retry と feature-accumulate を確認）
                logger.warning("バンドルの履歴が古い: 最終日 %s（%d 日前）", self.history_last_date, age)
            info["history_age_days"] = age
            logger.info("サービング用バンドルを読み込み: %s", info)
            return info

        from src.pipeline.features.horse_history.common import build_repair_map, read_runs_raw

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
        info = {"engine": "full", "history_rows": self.raw_hist.height, "history_last_date": self.history_last_date,
                "warm_sec": round(time.perf_counter() - t0, 1)}
        logger.info("推論用の履歴を読み込み: %s", info)
        return info

    def build(self, card: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
        if self.raw_hist is None and getattr(self, "bundle", None) is None:
            self.warm()
        if getattr(self, "bundle", None) is not None:
            from src.pipeline.features.horse_history.serving import compute_entrants

            return compute_entrants(self.bundle, card, self.graded_names)
        from src.pipeline.features.horse_history.inference import build_inference_features

        out, info = build_inference_features(self.raw_hist, card, self.repair_map, graded_names=self.graded_names)
        return out, info | {"engine": "full"}


V3_CARD_EXTRA = ["bracket_number", "weight_change", "field_size", "round", "weather", "weight_rule", "course_type", "jockey_name"]


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
    extra = [c for c in V3_CARD_EXTRA if c not in CARD_COLS]  # v3 の LayerA 列（枠番・馬体重増減・頭数・R・天候・重量種別）
    df = pl.DataFrame([{c: r.get(c) for c in CARD_COLS + ["race_name"] + extra} for r in rows], infer_schema_length=None)
    ints = [c for c in ("distance", "horse_number", "weight", "bracket_number", "weight_change", "field_size", "round") if c in df.columns]
    return df.with_columns([pl.col(c).cast(pl.Int64, strict=False) for c in ints]
                           + [pl.col("jockey_weight").cast(pl.Float64, strict=False)])


def fill_card_meta(card: pl.DataFrame, races: list[dict]) -> pl.DataFrame:
    """出馬表に日付・発走時刻が無いとき、発走時刻表の値で補う（面・距離は補えないので、空なら呼び出し側で除く）。"""
    sched = pl.DataFrame([
        {"race_id": r["race_id"], "_date": r["post_time"].strftime("%Y-%m-%d"),
         "_start": (r.get("start_time_str") or r["post_time"].strftime("%H:%M"))}
        for r in races if r.get("post_time")
    ], schema={"race_id": pl.Utf8, "_date": pl.Utf8, "_start": pl.Utf8})
    df = card.join(sched, on="race_id", how="left")
    df = df.with_columns(
        pl.when(pl.col("date").cast(pl.Utf8).fill_null("") == "").then(pl.col("_date")).otherwise(pl.col("date")).alias("date"),
        pl.when(pl.col("start_time").cast(pl.Utf8).fill_null("") == "").then(pl.col("_start")).otherwise(pl.col("start_time")).alias("start_time"),
    )
    return df.drop("_date", "_start")


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
    events_bad: list[dict] = []
    try:
        card = card_frame(storage, ids)
        if card.height == 0:
            raise ValueError("出馬表の行がありません")
        card = fill_card_meta(card, races)
        unusable = card.filter(
            ~(pl.col("distance").cast(pl.Float64, strict=False) > 0) | pl.col("surface").cast(pl.Utf8).fill_null("").eq("")
        )["race_id"].unique().to_list()
        if unusable:
            logger.error("[weight-trigger] 出馬表のレース情報（面・距離）が空のため除外: %s", unusable)
            card = card.filter(~pl.col("race_id").is_in(unusable))
            for r in races:
                if r["race_id"] in unusable:
                    pt = r.get("post_time")
                    ev = {"race_id": r["race_id"], "at": now.isoformat(timespec="seconds"), "status": "error", "mode": mode,
                          "error": "出馬表のレース情報（面・距離）が空"}
                    _append_event(pt.strftime("%Y%m%d") if pt else now.strftime("%Y%m%d"), ev)
                    events_bad.append(ev)
            races = [r for r in races if r["race_id"] not in unusable]
            if card.height == 0:
                return events_bad
        # 枠順確定前の出馬表（馬番が全頭 0）は特徴量の対象にならない。他のレースを巻き込まないよう、先に除く。
        no_number = (card.group_by("race_id").agg((pl.col("horse_number").fill_null(0) > 0).sum().alias("k")).filter(pl.col("k") == 0)["race_id"].to_list())
        if no_number:
            logger.error("[weight-trigger] 馬番が確定していない出馬表のため除外: %s", no_number)
            card = card.filter(~pl.col("race_id").is_in(no_number))
            for r in races:
                if r["race_id"] in no_number:
                    pt = r.get("post_time")
                    ev = {"race_id": r["race_id"], "at": now.isoformat(timespec="seconds"), "status": "error", "mode": mode,
                          "error": "出馬表の馬番が未確定（枠順確定前）"}
                    _append_event(pt.strftime("%Y%m%d") if pt else now.strftime("%Y%m%d"), ev)
                    events_bad.append(ev)
            races = [r for r in races if r["race_id"] not in no_number]
            if card.height == 0:
                return events_bad
        # 馬体重は出走馬全頭に同時に発表される。揃ったレースで体重の無い馬は、出走取消・除外とみなして出走馬から外す。
        announced = [rid for rid, k in weights_known.items() if k]
        no_weight = (pl.col("race_id").is_in(announced)) & ~(pl.col("weight").cast(pl.Float64, strict=False) > 0).fill_null(False)
        dropped = {r[0]: r[1] for r in card.filter(no_weight).group_by("race_id").len().iter_rows()} if announced else {}
        if dropped:
            logger.info("[weight-trigger] 馬体重のない馬を除外: %s", dropped)
            card = card.filter(~no_weight)
        F, info = ctx.build(card)
        v3p, v3_err, v3_sec = {}, None, 0.0
        if getattr(ctx, "v3", None) is not None:
            try:
                t_v3 = time.perf_counter()
                P = ctx.v3.predict(ctx.v3.features(card, F, storage))
                v3p = {k: g for k, g in P.groupby("race_id")}
                v3_sec = time.perf_counter() - t_v3
            except Exception as exc:
                logger.exception("v3 の予測に失敗: %s", ids)
                v3_err = str(exc)
    except Exception as exc:
        logger.exception("特徴量の作成に失敗: %s", ids)
        for r in races:
            ev = {"race_id": r["race_id"], "at": now.isoformat(timespec="seconds"), "status": "error", "error": str(exc), "mode": mode}
            pt = r.get("post_time")
            _append_event(pt.strftime("%Y%m%d") if pt else now.strftime("%Y%m%d"), ev)
            events.append(ev)
        return events_bad + events

    for r in races:
        rid = r["race_id"]
        day = str(F.filter(pl.col("race_id") == rid)["date"].min() or now.date()).replace("-", "")
        out_dir = inference_dir() / "features" / day
        out_dir.mkdir(parents=True, exist_ok=True)
        part = F.filter(pl.col("race_id") == rid)
        part.write_parquet(out_dir / f"{rid}.parquet")
        # 予測に使った時点の出馬表（馬体重・馬場状態・取消を反映したもの）を残す。出馬表は後の取得で上書きされるため。
        snap_dir = inference_dir() / "snapshots" / day
        snap_dir.mkdir(parents=True, exist_ok=True)
        (snap_dir / f"{rid}.json").write_text(json.dumps(
            {"race_id": rid, "detected_at": now.isoformat(timespec="seconds"), "weights_known": bool(weights_known.get(rid, info["weights_known_ratio"] >= 0.9)),
             "entries": card.filter(pl.col("race_id") == rid).to_dicts()}, ensure_ascii=False, default=str), encoding="utf-8")
        ev = {
            "race_id": rid, "at": now.isoformat(timespec="seconds"), "status": "ok", "mode": mode, "rows": part.height,
            "weights_known": bool(weights_known.get(rid, info["weights_known_ratio"] >= 0.9)),
            "going_known": info["going_known"], "build_sec": info["build_sec"], "batch_races": info["races"], "engine": info.get("engine", "full"),
            "history_last_date": ctx.history_last_date, "dropped_no_weight": int(dropped.get(rid, 0)),
        }
        if getattr(ctx, "v3", None) is not None:
            ev["v3_status"] = "error" if v3_err else ("ok" if rid in v3p else "missing")
            if v3_err:
                ev["v3_error"] = v3_err
            elif rid in v3p:
                v_dir = inference_dir() / "predictions_v3" / day
                v_dir.mkdir(parents=True, exist_ok=True)
                v3p[rid].to_parquet(v_dir / f"{rid}.parquet", index=False)
                ev["v3_sec"] = round(v3_sec, 1)
                if v3_mode() == "predict":
                    try:
                        from src.pipeline.inference.race_day_workflow import v3_payload
                        from src.pipeline.inference.race_prediction_service import save_cached

                        payload = v3_payload(rid, storage, v3p[rid], model_version=ctx.v3.bundle_dir.name, source="weight_t60", elapsed=v3_sec)
                        ev["v3_persisted"] = bool(save_cached(storage, rid, payload, source="weight_t60"))
                    except Exception as exc:
                        logger.exception("v3 の予測の保存に失敗: %s", rid)
                        ev["v3_status"], ev["v3_error"] = "error", str(exc)
        if mode == "predict" and not (v3_mode() == "predict" and ev.get("v3_persisted")):
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
    return events_bad + events
