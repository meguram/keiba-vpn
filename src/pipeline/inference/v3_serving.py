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

logger = logging.getLogger("pipeline.inference.v3_serving")

REPO_ROOT = Path(__file__).resolve().parents[3]
WORK = REPO_ROOT / "data/local/modeling/training_pipeline"
KEYS = ["race_id", "horse_id"]


def default_bundle_dir() -> Path:
    return Path(os.environ.get("KEIBA_V3_BUNDLE") or WORK / "final_v3/bundle_2025")


def default_stage_dir() -> Path:
    return Path(os.environ.get("KEIBA_V3_STAGE") or WORK / "final_stage/fixed")


class V3Serving:
    def __init__(self, bundle_dir: str | Path | None = None, stage_dir: str | Path | None = None, tables_dir: str | Path | None = None,
                 tl_reference: str | Path | None = None):
        self.bundle_dir = Path(bundle_dir) if bundle_dir else default_bundle_dir()
        self.stage_dir = Path(stage_dir) if stage_dir else default_stage_dir()
        self.tables_dir = tables_dir
        self.tl_reference_path = Path(tl_reference) if tl_reference else ssr.tl_reference_path(self.stage_dir.parent)
        self.bundle = None
        self.stage = None
        self.history: pd.DataFrame | None = None
        self.ped: spe.PedState | None = None
        self.tl_ref: pd.DataFrame | None = None
        self.aux = None
        self.feature_cols: list[str] = []

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

            self.aux = AuxModels.load(Path(os.environ.get("KEIBA_V3_AUX") or WORK / "final_aux"))
        self.history = sla.load_history(self.tables_dir)
        runs = apply_repair(load_runs(self.tables_dir), EMPTY_RM)
        self.ped = spe.PedState.from_history(spe.history_from_runs(runs))
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

        stage = self.stage.predict(base, base["race_id"])
        cur = pd.concat([base, stage] + ([self.aux.predict(base, base["race_id"])] if self.aux is not None else []), axis=1)

        tgt = base[KEYS + ["date"]].merge(cm[KEYS + ["surface", "distance", "track_condition", "venue_code"]], on=KEYS, how="left")
        ped = self.ped.features(tgt)
        extras = spe.extras_features(base[KEYS], spe.fetch_extras_rows(race_ids, storage, workers=min(workers, 4))) if storage is not None else spe.extras_features(base[KEYS], None)
        ped.index = extras.index = base.index

        records = self._training_records(base["horse_id"].unique().tolist(), storage, workers) if storage is not None else {}
        tl = ssr.tl_columns_for_races(base[KEYS + ["date"]], records, self.tl_ref)
        tl = base[KEYS].merge(tl, on=KEYS, how="left").drop(columns=KEYS)
        # 調教の記録が 1 頭も無いと列が作られない。学習表でも記録の無い馬は欠損（tl_has_training も欠損）なので、欠損で埋める
        tl = tl.reindex(columns=[x for x in self.feature_cols if x.startswith(ssr.TL_PREFIX)])
        tl.index = base.index

        rel_names = [x for x in self.feature_cols if ssr.split_relative_name(x) is not None and x not in cur.columns]
        rel = ssr.relative_by_names(pd.concat([cur, ped, extras], axis=1), rel_names) if rel_names else pd.DataFrame(index=base.index)

        allc = pd.concat([cur, ped, extras, tl, rel], axis=1)
        allc = allc.loc[:, ~allc.columns.duplicated()]
        missing = [x for x in self.feature_cols if x not in allc.columns]
        if missing:
            raise KeyError(f"v3 の特徴量が足りない列: {len(missing)} 列 {missing[:12]}")
        out = pd.concat([base[KEYS], allc[self.feature_cols]], axis=1)
        logger.info("v3 の特徴量: %d 行 × %d 列 %.1fs", len(out), len(self.feature_cols), time.perf_counter() - t0)
        return out

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
