"""
レースデイ予測パイプライン

大衆指標 (オッズ・人気) を完全排除した予測モデルを使用。

フロー:
  日付指定 → レースID一覧取得 → レースごとに{
    出馬表 + タイム指数 + 馬柱・調教 + パドック + バロメーター + 追い切り
    → 出走馬の戦績取得 → 特徴量テーブル構築 → カテゴリ変数エンコード
    → モデル予測 → 結果集約
  }

Usage:
  pipeline = RaceDayPipeline()
  results = pipeline.run("20260315")
  for r in results:
      print(r["race_name"], r["predictions"])
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger("pipeline.race_day")


@dataclass
class PipelineConfig:
    """パイプラインの動作設定"""
    scrape_interval: float = 1.0
    cache_scrape: bool = True
    auto_login: bool = True
    skip_existing_scrape: bool = True
    model_name: str = "keiba-lgbm-nopi"
    model_stage: str = "latest"
    mlflow_tracking_uri: str = "http://localhost:5000"
    output_dir: str = "data/predictions"
    parallel_horses: int = 1
    max_retries: int = 2


@dataclass
class RaceResult:
    """1レースの予測結果"""
    race_id: str
    race_name: str
    venue: str
    round_num: int
    surface: str
    distance: int
    track_condition: str
    field_size: int
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    scrape_time_sec: float = 0
    feature_time_sec: float = 0
    predict_time_sec: float = 0
    error: str = ""


class RaceDayPipeline:
    """日付を起点にした統合予測パイプライン"""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._runner = None
        self._model = None
        self._encoder = None
        self._ensemble: Any | None = None
        self._use_ensemble: bool | None = None

    @property
    def runner(self):
        if self._runner is None:
            from scraper.run import ScraperRunner
            self._runner = ScraperRunner(
                interval=self.config.scrape_interval,
                cache=self.config.cache_scrape,
                auto_login=self.config.auto_login,
            )
        return self._runner

    def run(
        self,
        date: str,
        race_ids: list[str] | None = None,
        callback: Callable[[str, str], None] | None = None,
    ) -> list[RaceResult]:
        """
        指定日付の全レースを処理する。

        Args:
            date: YYYYMMDD 形式
            race_ids: 処理するレースIDのリスト（指定なしなら全レース）
            callback: 進捗コールバック (phase, detail)
        """
        def _cb(phase: str, detail: str = ""):
            if callback:
                callback(phase, detail)
            logger.info("[%s] %s", phase, detail)

        _cb("init", f"レースデイパイプライン開始: {date}")

        # Phase 1: レースID一覧取得
        if not race_ids:
            _cb("race_list", f"日付 {date} のレース一覧を取得中...")
            race_list = self.runner.scrape_race_list(date)
            race_ids = [r["race_id"] for r in (race_list or {}).get("races", [])]
            _cb("race_list", f"{len(race_ids)}レース検出")

        if not race_ids:
            _cb("error", "レースが見つかりません")
            return []

        # Phase 2: レースごとに処理
        results: list[RaceResult] = []
        for idx, rid in enumerate(race_ids, 1):
            _cb("race", f"[{idx}/{len(race_ids)}] レース {rid} を処理中...")

            result = self._process_single_race(rid, _cb)
            results.append(result)

            if result.error:
                _cb("race_error", f"  レース {rid} エラー: {result.error}")
            else:
                _cb("race_done", f"  レース {rid} 完了 ({len(result.predictions)}頭)")

        # Phase 3: 結果を保存
        _cb("save", "予測結果を保存中...")
        self._save_results(date, results)

        # Phase 4: サマリ表示
        total_horses = sum(len(r.predictions) for r in results)
        ok_races = sum(1 for r in results if not r.error)
        _cb("done", f"完了: {ok_races}/{len(results)}レース, {total_horses}頭")

        return results

    def _process_single_race(
        self,
        race_id: str,
        callback: Callable[[str, str], None],
    ) -> RaceResult:
        """1レースのスクレイピング→特徴量構築→予測のフルサイクル"""

        result = RaceResult(race_id=race_id, race_name="", venue="", round_num=0,
                            surface="", distance=0, track_condition="", field_size=0)

        # Step 1: 全データ取得
        t0 = time.time()
        try:
            race_data = self._scrape_race(race_id, callback)
        except Exception as e:
            result.error = f"スクレイピング失敗: {e}"
            return result
        result.scrape_time_sec = time.time() - t0

        card = race_data.get("race_card") or {}
        result.race_name = card.get("race_name", "")
        result.venue = card.get("venue", "")
        result.round_num = card.get("round", 0)
        result.surface = card.get("surface", "")
        result.distance = card.get("distance", 0)
        result.track_condition = card.get("track_condition", "")
        result.field_size = card.get("field_size", 0) or len(card.get("entries", []))

        # Step 2: 特徴量テーブル構築
        t1 = time.time()
        try:
            from pipeline.feature_builder import build_race_features
            features_df = build_race_features(race_data)
        except Exception as e:
            result.error = f"特徴量構築失敗: {e}"
            return result
        result.feature_time_sec = time.time() - t1

        if features_df.empty:
            result.error = "特徴量テーブルが空です"
            return result

        # Step 3: 予測
        t2 = time.time()
        try:
            predictions_df = self._predict(features_df)
        except Exception as e:
            result.error = f"予測失敗: {e}"
            return result
        result.predict_time_sec = time.time() - t2

        result.predictions = predictions_df
        return result

    def _scrape_race(
        self,
        race_id: str,
        callback: Callable[[str, str], None],
    ) -> dict:
        """
        1レースの全データを効率的に取得する。
        出馬表 → タイム指数 → 馬柱・調教 → パドック → バロメーター → 追い切り → 各馬戦績
        オッズは取得しない（大衆指標排除ポリシー）。
        """
        race_data: dict[str, Any] = {"race_id": race_id}

        callback("scrape", f"  出馬表を取得中... ({race_id})")
        race_data["race_card"] = self.runner.scrape_race_card(
            race_id, skip_existing=self.config.skip_existing_scrape)

        callback("scrape", f"  タイム指数を取得中... ({race_id})")
        race_data["speed_index"] = self.runner.scrape_speed_index(
            race_id, skip_existing=self.config.skip_existing_scrape)

        callback("scrape", f"  馬柱・調教を取得中... ({race_id})")
        race_data["shutuba_past"] = self.runner.scrape_shutuba_past(
            race_id, skip_existing=self.config.skip_existing_scrape)

        for label, key, method_name in [
            ("パドック", "paddock", "scrape_paddock"),
            ("追い切り", "oikiri", "scrape_oikiri"),
        ]:
            callback("scrape", f"  {label}を取得中... ({race_id})")
            try:
                method = getattr(self.runner, method_name, None)
                if method:
                    race_data[key] = method(
                        race_id, skip_existing=self.config.skip_existing_scrape
                    )
            except Exception as e:
                logger.warning("%s取得失敗 [%s]: %s", label, race_id, e)

        callback("scrape", f"  バロメーターを取得中... ({race_id})")
        try:
            barometer = getattr(self.runner, "scrape_barometer", None)
            if barometer:
                race_data["barometer"] = barometer(
                    race_id, skip_existing=self.config.skip_existing_scrape
                )
        except Exception as e:
            logger.warning("バロメーター取得失敗 [%s]: %s", race_id, e)

        horse_ids = set()
        for source_key in ("race_card", "speed_index"):
            source = race_data.get(source_key)
            if source:
                for entry in source.get("entries", []):
                    hid = entry.get("horse_id", "")
                    if hid:
                        horse_ids.add(hid)

        callback("scrape", f"  出走馬 {len(horse_ids)}頭 の戦績を取得中...")
        race_data["horses"] = {}
        for i, hid in enumerate(sorted(horse_ids), 1):
            if i % 5 == 0:
                callback("scrape", f"    馬情報 {i}/{len(horse_ids)}...")
            try:
                horse = self.runner.scrape_horse(
                    hid,
                    skip_existing=self.config.skip_existing_scrape,
                    with_history=True,
                )
                if horse:
                    race_data["horses"][hid] = horse
            except Exception as e:
                logger.warning("馬情報取得失敗 [%s]: %s", hid, e)

        return race_data

    def _predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        特徴量テーブルからモデル予測を行う。
        アンサンブルモデル → 単体LightGBM → ヒューリスティック の順に試行。
        """
        meta_cols = ["race_id", "horse_number", "horse_name", "horse_id",
                     "venue", "race_date", "race_round"]
        meta = features_df[[c for c in meta_cols if c in features_df.columns]].copy()

        predicted = False

        ensemble = self._load_ensemble()
        if ensemble is not None:
            try:
                X = self._prepare_model_input(features_df)
                meta["pred_score"] = ensemble.predict(X)
                meta["model_type"] = "ensemble"
                predicted = True
                logger.info("アンサンブルモデルで予測完了")
            except Exception as e:
                logger.warning("アンサンブル予測失敗: %s", e)

        if not predicted:
            model = self._load_model()
            if model is not None:
                try:
                    X = self._prepare_model_input(features_df)
                    meta["pred_score"] = model.predict(X)
                    meta["model_type"] = "lightgbm"
                    predicted = True
                except Exception as e:
                    logger.warning("LightGBM 予測失敗: %s", e)

        if not predicted:
            meta["pred_score"] = self._fallback_score(features_df)
            meta["model_type"] = "heuristic"

        meta = meta.sort_values("pred_score", ascending=False).reset_index(drop=True)
        meta["pred_rank"] = range(1, len(meta) + 1)

        return meta

    def _load_ensemble(self):
        """アンサンブルモデルが存在すればロードする。"""
        if self._use_ensemble is False:
            return None
        if self._ensemble is not None:
            return self._ensemble

        try:
            from pipeline.ensemble_trainer import EnsembleTrainer
            from pathlib import Path as _Path
            if _Path("models/ensemble/ensemble_meta.json").exists():
                ens = EnsembleTrainer()
                ens._load_models()
                self._ensemble = ens
                self._use_ensemble = True
                logger.info("アンサンブルモデルをロード完了")
                return ens
        except Exception as e:
            logger.info("アンサンブルモデルなし: %s", e)

        self._use_ensemble = False
        return None

    def _load_model(self):
        """MLflow からモデルを読み込む。存在しない場合は None。"""
        if self._model is not None:
            return self._model

        try:
            import mlflow
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)

            client = mlflow.MlflowClient()
            versions = client.search_model_versions(
                f"name='{self.config.model_name}'"
            )
            if versions:
                latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
                model_uri = f"models:/{self.config.model_name}/{latest.version}"
                self._model = mlflow.pyfunc.load_model(model_uri)
                logger.info("MLflow モデルをロード: %s (v%s)", model_uri, latest.version)
                return self._model
            else:
                logger.info("MLflow に登録モデルなし: %s", self.config.model_name)
                return None
        except Exception as e:
            logger.info("MLflow モデルロード失敗 (代替予測使用): %s", e)
            return None

    def _prepare_model_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """モデル入力用にエンコーダーを適用し、数値カラムのみを返す。"""
        if self._encoder is None:
            self._encoder = self._load_encoder()

        from pipeline.feature_builder import PUBLIC_INDICATOR_SET
        import numpy as np

        if self._encoder is not None:
            encoded = self._encoder.transform(df)
        else:
            drop_cols = {"race_id", "horse_name", "horse_id", "race_date"}
            encoded = df.drop(
                columns=[c for c in drop_cols if c in df.columns], errors="ignore"
            )
            str_cols = {"surface", "direction", "weather", "track_condition", "sex",
                        "venue", "sire", "dam_sire", "jockey_name", "trainer_name"}
            for i in range(1, 6):
                str_cols.add(f"prev{i}_surface")
                str_cols.add(f"prev{i}_track_cond")
            for col in str_cols:
                if col in encoded.columns:
                    encoded[col] = encoded[col].astype("category").cat.codes

        if "finish_position" in encoded.columns:
            encoded.drop(columns=["finish_position"], inplace=True)

        numeric_cols = [
            c for c in encoded.select_dtypes(include=[np.number]).columns
            if c not in PUBLIC_INDICATOR_SET
        ]

        non_feature = {"horse_number", "race_round"}
        numeric_cols = [c for c in numeric_cols if c not in non_feature]

        return encoded[numeric_cols].fillna(0)

    def _load_encoder(self):
        """保存済みエンコーダーを読み込む。"""
        encoder_path = "models/encoder.json"
        try:
            from pipeline.encoder import FeatureEncoder
            from pathlib import Path
            if Path(encoder_path).exists():
                enc = FeatureEncoder()
                enc.load(encoder_path)
                logger.info("エンコーダーをロード: %s", encoder_path)
                return enc
        except Exception as e:
            logger.warning("エンコーダーロード失敗: %s", e)
        return None

    def _fallback_score(self, df: pd.DataFrame) -> pd.Series:
        """
        モデル不在時のヒューリスティックスコア。
        大衆指標 (オッズ・人気) を一切使用しない。

        構成:
          タイム指数 (35%) + 直近成績 (25%) + 上がり3F (20%) +
          戦績統計 (10%) + バロメーター偏差値 (10%)
        """
        scores = pd.Series(0.0, index=df.index)

        if "speed_max" in df.columns:
            sp = df["speed_max"].fillna(0)
            scores += (sp / sp.max() * 35) if sp.max() > 0 else 0

        finish_cols = [f"prev{i}_finish" for i in range(1, 6)]
        available = [c for c in finish_cols if c in df.columns]
        if available:
            finish_data = df[available].copy()
            finish_data[finish_data == 0] = pd.NA
            avg_f = finish_data.mean(axis=1).fillna(18)
            scores += (1 - avg_f / 18).clip(0, 1) * 25

        if "avg_last_3f_5" in df.columns:
            l3f = df["avg_last_3f_5"].fillna(40)
            scores += (1 - (l3f - 33) / 7).clip(0, 1) * 20

        if "career_top3_rate" in df.columns:
            t3r = df["career_top3_rate"].fillna(0)
            scores += t3r.clip(0, 1) * 10

        if "barometer_total" in df.columns:
            bt = df["barometer_total"].fillna(50)
            scores += ((bt - 30) / 40).clip(0, 1) * 10

        return scores

    def _save_results(self, date: str, results: list[RaceResult]):
        """予測結果を日付ごとのCSVとサマリに保存する。"""
        out_dir = Path(self.config.output_dir) / date
        out_dir.mkdir(parents=True, exist_ok=True)

        all_preds = []
        for r in results:
            if r.error or r.predictions.empty:
                continue
            pred = r.predictions.copy()
            pred["race_name"] = r.race_name
            pred["venue"] = r.venue
            pred["surface"] = r.surface
            pred["distance"] = r.distance
            pred["track_condition"] = r.track_condition
            all_preds.append(pred)

        if all_preds:
            combined = pd.concat(all_preds, ignore_index=True)
            combined.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")

        summary_lines = [
            f"# レースデイ予測: {date}",
            f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        for r in results:
            status = "✅" if not r.error else "❌"
            summary_lines.append(f"## {status} {r.venue}{r.round_num}R {r.race_name}")
            summary_lines.append(f"  {r.surface}{r.distance}m / {r.track_condition} / {r.field_size}頭")

            if r.error:
                summary_lines.append(f"  エラー: {r.error}")
            elif not r.predictions.empty:
                summary_lines.append("")
                summary_lines.append("| 予想順 | 馬番 | 馬名 | スコア |")
                summary_lines.append("|--------|------|------|--------|")
                for _, row in r.predictions.head(5).iterrows():
                    summary_lines.append(
                        f"| {int(row.get('pred_rank', 0))} | "
                        f"{int(row.get('horse_number', 0))} | "
                        f"{row.get('horse_name', '')} | "
                        f"{row.get('pred_score', 0):.1f} |"
                    )
            summary_lines.append("")

            timing = (f"  ⏱ scrape={r.scrape_time_sec:.1f}s "
                      f"feature={r.feature_time_sec:.1f}s "
                      f"predict={r.predict_time_sec:.1f}s")
            summary_lines.append(timing)
            summary_lines.append("")

        (out_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
        logger.info("結果保存: %s", out_dir)

    def close(self):
        if self._runner:
            self._runner.close()
