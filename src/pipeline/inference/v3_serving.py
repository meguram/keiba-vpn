"""アンサンブル v3 を当日のレースに使うための特徴量の組み立てと予測。

履歴系 127 列（``InferenceContext.build``）に、LayerA 48 列・前段 3 列・相対化 42 列・血統 24 列・追加 7 列・追い切り 50 列を足して、
バンドルが要求する 301 列を作る。各部分の定義は ``serving_*`` の各モジュール（学習表との一致検証つき）を参照。

``KEIBA_V3_BUNDLE``（既定: final_v3/bundle_2025）と ``KEIBA_V3_STAGE``（既定: final_stage/fixed）で使うファイルを切り替える。
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from src.pipeline.features import serving_layer_a as sla
from src.pipeline.features import serving_ped_extras as spe
from src.pipeline.features import serving_stage_rel as ssr
from src.pipeline.features.serving_v5 import v5_columns as sv5_columns

logger = logging.getLogger("pipeline.inference.v3_serving")

REPO_ROOT = Path(__file__).resolve().parents[3]
WORK = REPO_ROOT / "data/local/modeling/training_pipeline"
KEYS = ["race_id", "horse_id"]


def default_bundle_dir() -> Path:
    return Path(os.environ.get("KEIBA_V3_BUNDLE") or WORK / "final_v3/bundle_2025")


def default_stage_dir(bundle_dir: Path | None = None) -> Path:
    """前段モデルの置き場。v4 のバンドルは新土台（training_pipeline_v4）で作り直した前段（final_stage/v4_fixed。検証用 bundle_v4_2024 は v4_verify_2024）を使う。"""
    if os.environ.get("KEIBA_V3_STAGE"):
        return Path(os.environ["KEIBA_V3_STAGE"])
    name = Path(bundle_dir).name if bundle_dir else ""
    if name.startswith(("bundle_v4", "bundle_v5")):
        return WORK / "final_stage" / ("v4_verify_2024" if name.endswith("2024") else "v4_fixed")
    if name.startswith("bundle_v6"):
        return WORK / "final_stage" / ("v6_e_2024" if name.endswith("2024") else "v6_e_2025")
    return WORK / "final_stage/fixed"


def default_aux_dir(bundle_dir: Path | None = None) -> Path:
    """補助ヘッドの最終モデルの置き場（v4 は新土台のもの）。"""
    if os.environ.get("KEIBA_V3_AUX"):
        return Path(os.environ["KEIBA_V3_AUX"])
    name = Path(bundle_dir).name if bundle_dir else ""
    return (WORK.parent / "training_pipeline_v4/final_aux") if name.startswith(("bundle_v4", "bundle_v5", "bundle_v6")) else WORK / "final_aux"


def v5_enabled() -> bool:
    """v5 を v4 の shadow と並走させる（KEIBA_V5_SHADOW=1）。出力は predictions_v5/<日>/<race_id>.parquet。v4 の経路には影響しない。"""
    return (os.environ.get("KEIBA_V5_SHADOW") or "").strip().lower() in ("1", "true", "on", "yes")


def v5_paths() -> dict:
    """v5 のバンドル・前段・追い切りの基準表（KEIBA_V5_BUNDLE / KEIBA_V5_STAGE / KEIBA_V5_TL_REF）。"""
    bundle = Path(os.environ.get("KEIBA_V5_BUNDLE") or WORK / "final_v5/bundle_v5_2025")
    stage = Path(os.environ.get("KEIBA_V5_STAGE") or WORK / "final_stage" / ("v4_verify_2024" if bundle.name.endswith("2024") else "v4_fixed"))
    tl_ref = Path(os.environ.get("KEIBA_V5_TL_REF") or WORK.parent / "training_pipeline_v4/tp35/tl_reference_v2.parquet")
    return {"bundle": bundle, "stage": stage, "tl_reference": tl_ref}


def v6_enabled() -> bool:
    """v6 を v4/v5 の shadow と並走させる（KEIBA_V6_SHADOW=1）。出力は predictions_v6/<日>/<race_id>.parquet。v4・v5 の経路には影響しない。"""
    return (os.environ.get("KEIBA_V6_SHADOW") or "").strip().lower() in ("1", "true", "on", "yes")


def v6_paths() -> dict:
    """v6 のバンドル・前段 S3B・追い切りの基準表・RB の履歴状態（KEIBA_V6_BUNDLE / KEIBA_V6_STAGE / KEIBA_V5_TL_REF / KEIBA_V6_RB_STATE）。"""
    bundle = Path(os.environ.get("KEIBA_V6_BUNDLE") or WORK / "final_v6e/bundle_v6_2025")
    stage = Path(os.environ.get("KEIBA_V6_STAGE") or WORK / "final_stage" / ("v6_e_2024" if bundle.name.endswith("2024") else "v6_e_2025"))
    tl_ref = Path(os.environ.get("KEIBA_V5_TL_REF") or WORK.parent / "training_pipeline_v4/tp35/tl_reference_v2.parquet")
    rb = Path(os.environ.get("KEIBA_V6_RB_STATE") or Path(os.environ.get("KEIBA_INFERENCE_DIR") or REPO_ROOT / "data/local/inference") / "state" / "rb_v6")
    return {"bundle": bundle, "stage": stage, "tl_reference": tl_ref, "rb_state": rb}


class V3Serving:
    def __init__(self, bundle_dir: str | Path | None = None, stage_dir: str | Path | None = None, tables_dir: str | Path | None = None,
                 tl_reference: str | Path | None = None):
        self.bundle_dir = Path(bundle_dir) if bundle_dir else default_bundle_dir()
        self.stage_dir = Path(stage_dir) if stage_dir else default_stage_dir(self.bundle_dir)
        self.tables_dir = tables_dir
        self.tl_reference_path = Path(tl_reference) if tl_reference else ssr.tl_reference_path(self.stage_dir.parent)
        self.bundle = None
        self.stage = None
        self.history: pd.DataFrame | None = None
        self.ped: spe.PedState | None = None
        self.tl_ref: pd.DataFrame | None = None
        self.aux = None
        self.rb_state: Path | None = None          # v6: RB の履歴状態の場所（v6_paths()["rb_state"]。None なら RB 列は使わない）
        self.rb_hist: pd.DataFrame | None = None
        self.rb_params: dict | None = None
        self.feature_cols: list[str] = []
        self.v4 = None
        self._card_day: dict[str, pd.DataFrame] = {}

    def warm(self) -> dict:
        from src.pipeline.models.ensemble_bundle import EnsembleBundle
        from src.pipeline.features.horse_history.common import load_runs, apply_repair
        from src.pipeline.features.horse_history.serving import EMPTY_RM

        t0 = time.perf_counter()
        self.bundle = EnsembleBundle.load(self.bundle_dir)
        self.feature_cols = list(dict.fromkeys(c for m in self.bundle.members for c in m.feature_cols))
        self.stage = ssr.StageModels.load(self.stage_dir)
        self.tl_ref = pd.read_parquet(self.tl_reference_path)
        if any(x.startswith("mtl_") for x in self.feature_cols):           # 補助タスクつき NN の補助ヘッド予測を使うバンドル（v4 候補）
            from src.pipeline.features.serving_aux import AuxModels

            self.aux = AuxModels.load(default_aux_dir(self.bundle_dir))
        from src.pipeline.features import serving_v4 as sv4

        if sv4.needed_groups(self.feature_cols):                           # v4 の群（馬主・馬自身・メンバー構成/コース環境・種牡馬 ID）を要求するバンドル
            tt = Path(os.environ.get("KEIBA_V4_TRAIN_TABLE") or WORK.parent / "training_pipeline_v4/features/train_table.parquet")
            if not tt.exists():
                tt = WORK.parent / "training_pipeline_fixed/features/train_table.parquet"
            self.v4 = sv4.V4Features(self.tables_dir, tt)
            self.v4.warm()
        from src.pipeline.features import serving_v6 as sv6

        if self.rb_state is not None and any(x in sv6.COLUMNS for x in self.feature_cols):            # v6: めぐ指数/netkeiba 指数の履歴 RB（tp-65）
            self.rb_hist, self.rb_params = sv6.load_state(self.rb_state)
        self.history = sla.load_history(self.tables_dir)
        runs = apply_repair(load_runs(self.tables_dir), EMPTY_RM)
        from src.pipeline.features import serving_parents as spar

        hist_ped = spe.history_from_runs(runs)
        if spar.enabled():              # 学習表の sire / dam_sire は血統 JSON 由来の ID（血統キー復旧後）。履歴・当日とも同じキーで数える
            self.ped = spe.PedState.from_history(spar.get_map().attach(hist_ped), blank_pedigree=False)
        else:
            self.ped = spe.PedState.from_history(hist_ped)
        info = {"bundle": str(self.bundle_dir), "columns": len(self.feature_cols), "history_rows": len(self.history),
                "history_last_date": str(self.history["date"].max()), "warm_sec": round(time.perf_counter() - t0, 1)}
        logger.info("v3 の推論準備: %s", info)
        return info

    # ------------------------------------------------------------------ 特徴量
    def features(self, card: pl.DataFrame, F: pl.DataFrame, storage: Any, *, history: pd.DataFrame | None = None, workers: int = 8) -> pd.DataFrame:
        """card: ``card_frame`` の出馬表（馬体重・日付・発走時刻つき）、F: ``InferenceContext.build`` の履歴系。
        history: LayerA の履歴の差し替え（省略時は起動時の履歴 + 同日に確定済みのレースの結果）。戻り値: race_id・horse_id + バンドルの全列。"""
        t0 = time.perf_counter()
        c = card.to_pandas()
        race_ids = sorted(c["race_id"].astype(str).unique())
        if history is None:
            same_day = sla.same_day_results(storage, race_ids_of_day(c, storage)) if storage is not None else None
            history = self.history if same_day is None or len(same_day) == 0 else pd.concat([self.history, same_day], ignore_index=True)
        A = sla.build_layer_a_features(c, history)
        Fp = F.to_pandas()
        Fp["race_id"], Fp["horse_id"] = Fp["race_id"].astype(str), Fp["horse_id"].astype(str)
        fam = [x for x in Fp.columns if x not in KEYS and x not in A.columns and x != "date"]
        base = A.merge(Fp[KEYS + fam], on=KEYS, how="left")
        cm = c.assign(race_id=c["race_id"].astype(str), horse_id=c["horse_id"].astype(str)).drop_duplicates(KEYS)
        base = base.merge(cm[KEYS + ["date"]], on=KEYS, how="left")
        base = base.sort_values(["race_id", "horse_number"], kind="stable").reset_index(drop=True)

        records = None
        stage_in = base
        extra_names = [x for x in self.stage.feature_cols if x not in base.columns]
        if extra_names:                                                                       # v6: 前段 S3B は tl_・wc_・追走難度 v2・ラップ予想・位置取り予想も入力にする
            from src.pipeline.features import serving_v6 as sv6
            from src.pipeline.features import serving_v5 as sv5

            records = self._training_records(base["horse_id"].unique().tolist(), storage, workers) if storage is not None else {}
            parts = [base]
            tl_n = [x for x in extra_names if x.startswith(ssr.TL_PREFIX)]
            wc_n = [x for x in extra_names if x.startswith("wc_")]
            if tl_n:
                tl0 = ssr.tl_columns_for_races(base[KEYS + ["date"]], records, self.tl_ref)
                tl0 = base[KEYS].merge(tl0, on=KEYS, how="left").drop(columns=KEYS).reindex(columns=tl_n)
                tl0.index = base.index
                parts.append(tl0)
            if wc_n:
                wc0 = sv5.cy_columns_for_races(base[KEYS + ["date"]], records, self.tl_ref)
                wc0 = base[KEYS].merge(wc0, on=KEYS, how="left").drop(columns=KEYS).reindex(columns=wc_n)
                wc0.index = base.index
                parts.append(wc0)
            rest = [x for x in extra_names if x not in tl_n and x not in wc_n]
            if rest:
                inf = Path(os.environ.get("KEIBA_INFERENCE_DIR") or REPO_ROOT / "data/local/inference")
                parts.append(sv6.stage_forecast_columns(base, rest, inf))
            stage_in = pd.concat(parts, axis=1)
        stage = self.stage.predict(stage_in, base["race_id"])
        cur = pd.concat([base, stage] + ([self.aux.predict(base, base["race_id"])] if self.aux is not None else []), axis=1)
        v4cols = None
        if self.v4 is not None:
            v4cols = self.v4.features(c, cur, self.feature_cols, storage, self._day_card(c, storage))
            v4cols.index = pd.RangeIndex(len(v4cols))
            v4cols = base[KEYS].astype(str).merge(v4cols.drop_duplicates(KEYS), on=KEYS, how="left").drop(columns=KEYS)
            v4cols.index = base.index

        tgt = base[KEYS + ["date"]].merge(cm[KEYS + ["surface", "distance", "track_condition", "venue_code"]], on=KEYS, how="left")
        from src.pipeline.features import serving_parents as spar

        ped = self.ped.features(spar.get_map().attach(tgt) if (spar.enabled() and not self.ped.blank_pedigree) else tgt)
        extras = spe.extras_features(base[KEYS], spe.fetch_extras_rows(race_ids, storage, workers=min(workers, 4))) if storage is not None else spe.extras_features(base[KEYS], None)
        ped.index = extras.index = base.index

        if records is None:
            records = self._training_records(base["horse_id"].unique().tolist(), storage, workers) if storage is not None else {}
        tl = ssr.tl_columns_for_races(base[KEYS + ["date"]], records, self.tl_ref)
        tl = base[KEYS].merge(tl, on=KEYS, how="left").drop(columns=KEYS)
        # 調教の記録が 1 頭も無いと列が作られない。学習表でも記録の無い馬は欠損（tl_has_training も欠損）なので、欠損で埋める
        tl = tl.reindex(columns=[x for x in self.feature_cols if x.startswith(ssr.TL_PREFIX)])
        tl.index = base.index

        wc_names = [x for x in self.feature_cols if x.startswith("wc_")]                    # v5: 追い切りサイクルの推移 CY（tp-40）
        if wc_names:
            from src.pipeline.features import serving_v5 as sv5

            wc = sv5.cy_columns_for_races(base[KEYS + ["date"]], records, self.tl_ref)
            wc = base[KEYS].merge(wc, on=KEYS, how="left").drop(columns=KEYS).reindex(columns=wc_names)
            wc.index = base.index
        else:
            wc = pd.DataFrame(index=base.index)

        px_names = [x for x in self.feature_cols if x in set(sv5_columns().get("PX", []))]                 # v5: 血統の位置別セル・SVD 射影・リッジ（tp-38 の ALL_BEST）
        if px_names and self.v4 is not None:
            pxf = self._px_for_day(c, storage)
            px = base[KEYS].merge(pxf, on=KEYS, how="left").drop(columns=KEYS).reindex(columns=px_names)
            px.index = base.index
        else:
            px = pd.DataFrame(index=base.index)

        from src.pipeline.features import serving_v6 as sv6

        rb_names = [x for x in self.feature_cols if x in sv6.COLUMNS]                                    # v6: めぐ指数/netkeiba 指数の履歴 RB（16 列）
        if rb_names and self.rb_hist is not None:
            rb = sv6.rb_features(self.rb_hist, base[KEYS + ["date"]], self.rb_params)
            rb = base[KEYS].astype(str).merge(rb.assign(race_id=rb["race_id"].astype(str), horse_id=rb["horse_id"].astype(str)), on=KEYS, how="left").drop(columns=KEYS).reindex(columns=rb_names)
            rb.index = base.index
        else:
            rb = pd.DataFrame(index=base.index)

        rel_names = [x for x in self.feature_cols if ssr.split_relative_name(x) is not None and x not in cur.columns]
        rel = ssr.relative_by_names(pd.concat([cur, ped, extras], axis=1), rel_names) if rel_names else pd.DataFrame(index=base.index)

        allc = pd.concat([cur, ped, extras, tl, wc, px, rb, rel] + ([v4cols] if v4cols is not None else []), axis=1)
        allc = allc.loc[:, ~allc.columns.duplicated()]
        missing = [x for x in self.feature_cols if x not in allc.columns]
        if missing:
            raise KeyError(f"v3 の特徴量が足りない列: {len(missing)} 列 {missing[:12]}")
        out = pd.concat([base[KEYS], allc[self.feature_cols]], axis=1)
        logger.info("v3 の特徴量: %d 行 × %d 列 %.1fs", len(out), len(self.feature_cols), time.perf_counter() - t0)
        return out

    def _px_for_day(self, c: pd.DataFrame, storage: Any) -> pd.DataFrame:
        """その日の全レースの PX 列（v5）。日単位でキャッシュ（学習側の関数を履歴 + 当日の行に当てるので、1 日 15〜20 秒）。"""
        from src.pipeline.features import serving_v5_px as sv5px

        day = str(pd.to_datetime(c["date"]).min().date())
        if getattr(self, "_px_cache", None) is None:
            self._px_cache = {}
        want = c[KEYS].astype(str)
        have = self._px_cache.get(day)
        if have is not None and len(want.merge(have[KEYS], on=KEYS, how="left", indicator=True).query("_merge == 'left_only'")) == 0:
            return have
        if getattr(self, "px", None) is None:
            self.px = sv5px.PXFeatures(self.v4.hist.field_hist)
        allc = self._day_card(c, storage)
        card = (allc if allc is not None and len(allc) else c).copy()
        card["race_id"], card["horse_id"] = card["race_id"].astype(str), card["horse_id"].astype(str)
        card = pd.concat([card, c[[x for x in c.columns if x in card.columns]].astype({"race_id": str, "horse_id": str})], ignore_index=True).drop_duplicates(KEYS)
        if "field_size" not in card.columns or card["field_size"].isna().any():
            card["field_size"] = card.groupby("race_id")["horse_id"].transform("count")
        out = self.px.day(card)
        self._px_cache[day] = out
        return out

    def _day_card(self, c: pd.DataFrame, storage: Any) -> pd.DataFrame | None:
        """その日の全レースの出馬表（馬主・馬自身の群を日単位で作るため）。日程から引けなければ None（今回の出馬表だけで作る）。"""
        if storage is None:
            return None
        day = str(pd.to_datetime(c["date"]).min().date())
        if day in self._card_day:
            return self._card_day[day]
        try:
            from src.pipeline.inference.race_day_workflow import load_day_schedule
            from src.pipeline.inference.weight_triggered_predict import card_frame

            ids = [r["race_id"] for r in load_day_schedule(storage, day.replace("-", ""))]
            allc = card_frame(storage, ids).to_pandas()
            if len(allc):
                self._card_day[day] = allc
                return allc
        except Exception as exc:
            logger.warning("その日の全レースの出馬表を作れなかった（今回の分だけで作る）: %s", exc)
        return None

    @staticmethod
    def _training_records(horse_ids: list[str], storage: Any, workers: int) -> dict[str, dict | None]:
        def one(h: str):
            try:
                return h, storage.load("horse_training", h)
            except Exception:
                return h, None

        with ThreadPoolExecutor(max_workers=workers) as ex:
            return dict(ex.map(one, horse_ids))

    # ------------------------------------------------------------------ 予測
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """戻り値: race_id・horse_id・p_win・p_top2・p_top3。"""
        P = self.bundle.predict(X[self.feature_cols], X["race_id"].to_numpy()).reset_index(drop=True)
        return pd.concat([X[KEYS].reset_index(drop=True), P], axis=1)


def race_ids_of_day(card: pd.DataFrame, storage: Any) -> list[str]:
    """同じ開催日に同じ場で走るレース（今回の対象も含む）の race_id。結果が保存済みのものだけ ``same_day_results`` が拾う。"""
    out: set[str] = set()
    for rid in card["race_id"].astype(str).unique():
        out.update(f"{rid[:10]}{n:02d}" for n in range(1, 13))
    return sorted(out - set(card["race_id"].astype(str)))
