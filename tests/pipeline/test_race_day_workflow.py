"""開催日ワークフロー（発走45分前に予測）のエンドツーエンド検証。

疑似特徴量ビルダー + 疑似アンサンブル + モックの開催日データで、
「発走時刻表 → T-45起動 → 特徴量 → アンサンブル推論 → 結果保存 → 保存結果を読む」が
最後まで動くことを確認する。GCP には接続しない（Cloud Tasks は呼び出しを記録する偽物）。
"""
from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from src.pipeline.features.pseudo_builder import PseudoFeatureBuilder
from src.pipeline.inference import race_day_workflow as wf
from src.pipeline.models.ensemble_predictor import EnsemblePredictor
from src.pipeline.models.model_registry import LocalModelStore, fetch_latest, publish_model
from src.scripts.maintenance.make_pseudo_ensemble import train_pseudo_ensemble

JST = ZoneInfo("Asia/Tokyo")
DATE = "20261004"
N_FEATURES = 40


class InMemoryStorage:
    """GCS 相当の保存先（カテゴリ/キー → データ）のメモリ上の代用品。

    ``HybridStorage`` は GCS 未接続だと race_shutuba 等を保存しない（GCSが正本の設計）ため、
    ワークフローの検証にはこちらを使う。``writable=False`` で保存不可（GCS未接続）を再現する。
    """

    def __init__(self, writable: bool = True):
        self.data: dict[tuple[str, str], dict] = {}
        self.writable = writable

    def load(self, category: str, key: str):
        return self.data.get((category, key))

    def save(self, category: str, key: str, data: dict) -> bool:
        if not self.writable:
            return False
        self.data[(category, key)] = data
        return True

    def seed(self, category: str, key: str, data: dict) -> None:
        self.data[(category, key)] = data


def _entries(n: int, race_id: str) -> list[dict]:
    return [
        {"horse_id": f"H{race_id[-4:]}{i:02d}", "horse_number": i + 1, "horse_name": f"馬{race_id[-2:]}-{i + 1}"}
        for i in range(n)
    ]


class RaceDayWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        # 疑似アンサンブルを作って「モデルストア」へ公開し、現行版として取得する
        model_dir = cls.root / "trained"
        names = train_pseudo_ensemble(model_dir, N_FEATURES, n_rows=400)
        cls.store = LocalModelStore(cls.root / "store")
        publish_model(model_dir, "pseudo-v1", cls.store, feature_names=names)
        cls.model_dir = fetch_latest(cls.store, cls.root / "cache")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        self._env = patch.dict(os.environ, {})
        self._env.start()
        self.addCleanup(self._env.stop)
        self.storage = InMemoryStorage()
        self.predictor = EnsemblePredictor.load(self.model_dir)
        self.builder = PseudoFeatureBuilder(n_features=N_FEATURES)
        self.races = self._make_day()

    def _make_day(self) -> list[tuple[str, str]]:
        """東京・京都の各3レース分の開催日データ（race_lists / race_shutuba / horse_result）を作る。"""
        races: list[tuple[str, str]] = []
        slots = {"05": ["09:55", "10:25", "10:55"], "08": ["10:00", "10:30", "11:00"]}
        rl = []
        for venue_code, times in slots.items():
            for rnd, hhmm in enumerate(times, start=1):
                rid = f"20260{venue_code[-1]}0301{rnd:02d}"
                races.append((rid, hhmm))
                entries = _entries(12 + rnd, rid)
                rl.append({"race_id": rid, "venue": venue_code, "round": rnd, "race_name": f"R{rnd}"})
                self.storage.seed("race_shutuba", rid, {
                    "race_id": rid, "race_name": f"R{rnd}", "venue": venue_code, "round": rnd,
                    "surface": "芝", "distance": 1600, "start_time": hhmm, "entries": entries,
                })
                for e in entries:
                    self.storage.seed("horse_result", e["horse_id"], {"horse_id": e["horse_id"], "race_history": []})
        self.storage.seed("race_lists", DATE, {"date": DATE, "races": rl})
        return races

    # ── 発走時刻表 → 起動時刻 ─────────────────────────────

    def test_plan_is_45_minutes_before_post_time(self):
        plan = wf.plan_race_day(self.storage, DATE)
        self.assertEqual(len(plan), 6)
        for row in plan:
            self.assertEqual(row["post_time"] - row["run_at"], timedelta(minutes=45))
        self.assertEqual(plan[0]["run_at"].strftime("%H:%M"), "09:10")  # 09:55 発走の45分前
        self.assertEqual([r["run_at"] for r in plan], sorted(r["run_at"] for r in plan))

    # ── 1レースの予測 ─────────────────────────────────────

    def test_predict_race_saves_result_that_can_be_read_back(self):
        rid = self.races[0][0]
        res = wf.predict_race(rid, self.storage, builder=self.builder, predictor=self.predictor)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["model_type"], "ensemble_stacking")
        self.assertEqual(res["model_version"], "pseudo-v1")
        self.assertEqual(res["feature_count"], N_FEATURES)
        preds = res["predictions"]
        self.assertEqual(len(preds), res["total_horses"])
        self.assertEqual([p["pred_rank"] for p in preds], list(range(1, len(preds) + 1)))
        self.assertAlmostEqual(sum(p["normalized_score"] for p in preds), 1.0, places=1)

        # 保存結果を画面側と同じ経路（load_cached）で読める
        from src.pipeline.inference.race_prediction_service import load_cached

        cached = load_cached(self.storage, rid)
        self.assertIsNotNone(cached)
        self.assertEqual(len(cached["predictions"]), len(preds))

    def test_prediction_is_deterministic(self):
        rid = self.races[0][0]
        a = wf.predict_race(rid, self.storage, builder=self.builder, predictor=self.predictor)
        b = wf.predict_race(rid, self.storage, builder=self.builder, predictor=self.predictor)
        self.assertEqual([p["horse_id"] for p in a["predictions"]], [p["horse_id"] for p in b["predictions"]])

    def test_missing_shutuba_returns_error_without_saving(self):
        from src.pipeline.inference.race_prediction_service import load_cached

        res = wf.predict_race("999999999999", self.storage, builder=self.builder, predictor=self.predictor)
        self.assertEqual(res["status"], "error")
        self.assertIsNone(load_cached(self.storage, "999999999999"))

    def test_feature_mismatch_is_detected(self):
        from src.pipeline.models.ensemble_predictor import FeatureMismatchError

        wrong_builder = PseudoFeatureBuilder(n_features=N_FEATURES - 5)
        with self.assertRaises(FeatureMismatchError):
            wf.predict_race(self.races[0][0], self.storage, builder=wrong_builder, predictor=self.predictor)

    # ── 起動方法1: 予約配信 ───────────────────────────────

    def test_enqueue_registers_scheduled_tasks_for_every_race(self):
        calls = []

        def fake_enqueue(payload, *, schedule_time, task_id, **kw):
            calls.append((payload, schedule_time, task_id))
            return f"projects/p/locations/l/queues/q/tasks/{task_id}"

        now = datetime(2026, 10, 4, 7, 0, tzinfo=JST)
        res = wf.enqueue_race_day_tasks(self.storage, DATE, enqueue_fn=fake_enqueue, now_fn=lambda: now)
        self.assertEqual([r["status"] for r in res], ["enqueued"] * 6)
        self.assertEqual(len(calls), 6)
        payload, when, task_id = calls[0]
        self.assertEqual(payload["job_kind"], "predict_race")
        self.assertEqual(when.strftime("%H:%M"), "09:10")
        self.assertTrue(task_id.startswith("predict-"))

    def test_enqueue_is_idempotent_on_already_exists(self):
        class AlreadyExists(Exception):
            pass

        def dup(payload, **kw):
            raise AlreadyExists("task exists")

        now = datetime(2026, 10, 4, 7, 0, tzinfo=JST)
        res = wf.enqueue_race_day_tasks(self.storage, DATE, enqueue_fn=dup, now_fn=lambda: now)
        self.assertEqual({r["status"] for r in res}, {"exists"})

    def test_enqueue_catches_up_late_registration_and_skips_past_races(self):
        calls = []
        # 10:40: 09:55/10:00 発走は発走済み、10:25・10:30 は起動時刻(09:40/09:45)を過ぎているが発走前ではない…
        now = datetime(2026, 10, 4, 10, 10, tzinfo=JST)
        res = wf.enqueue_race_day_tasks(
            self.storage, DATE,
            enqueue_fn=lambda payload, *, schedule_time, task_id, **kw: calls.append((task_id, schedule_time)) or "n",
            now_fn=lambda: now, catch_up_delay_sec=10,
        )
        by_status = {}
        for r in res:
            by_status.setdefault(r["status"], []).append(r["race_id"])
        self.assertEqual(len(by_status["skipped_past"]), 2)      # 09:55 と 10:00 は発走済み
        self.assertEqual(len(by_status["enqueued"]), 4)
        # 起動時刻を過ぎたレースは「今+10秒」で即実行、まだのレースは 発走-45分
        times = sorted(t for _, t in calls)
        self.assertEqual(times[0], now + timedelta(seconds=10))

    # ── 起動方法2: 待機して順に実行 ────────────────────────

    def test_run_local_waits_until_t_minus_45_then_predicts_all(self):
        clock = {"now": datetime(2026, 10, 4, 8, 30, tzinfo=JST)}
        sleeps = []

        def fake_sleep(sec):
            sleeps.append(sec)
            clock["now"] += timedelta(seconds=sec)

        res = wf.run_race_day_local(
            self.storage, DATE, now_fn=lambda: clock["now"], sleep_fn=fake_sleep,
            builder=self.builder, predictor=self.predictor,
        )
        self.assertEqual([r["status"] for r in res], ["success"] * 6)
        self.assertAlmostEqual(sleeps[0], 40 * 60, delta=1)      # 08:30 → 09:10 まで待つ
        # 全レースの結果が保存されている
        from src.pipeline.inference.race_prediction_service import load_cached

        for rid, _ in self.races:
            self.assertIsNotNone(load_cached(self.storage, rid), rid)

    def test_run_local_continues_after_one_race_fails(self):
        # 1レースだけ出馬表を消して失敗させても、残りは実行される
        bad = self.races[2][0]
        del self.storage.data[("race_shutuba", bad)]
        clock = {"now": datetime(2026, 10, 4, 8, 30, tzinfo=JST)}
        res = wf.run_race_day_local(
            self.storage, DATE, now_fn=lambda: clock["now"],
            sleep_fn=lambda s: clock.__setitem__("now", clock["now"] + timedelta(seconds=s)),
            builder=self.builder, predictor=self.predictor,
        )
        statuses = {r["race_id"]: r["status"] for r in res}
        self.assertNotEqual(statuses.get(bad), "success")
        self.assertEqual(sum(1 for s in statuses.values() if s == "success"), 5)

    def test_run_local_on_non_race_day_exits_without_loading_model(self):
        with patch.object(wf, "load_predictor", side_effect=AssertionError("モデルを読み込んだ")):
            self.assertEqual(wf.run_race_day_local(self.storage, "20261005"), [])

    # ── 保存できなかった場合（GCS未接続など）────────────────

    def test_unsaved_result_is_flagged_and_fails_in_worker(self):
        self.storage.writable = False
        res = wf.predict_race(self.races[0][0], self.storage, builder=self.builder, predictor=self.predictor)
        self.assertEqual(res["status"], "success")
        self.assertFalse(res["persisted"])              # 計算はできたが保存されていない
        with patch.object(wf, "load_predictor", return_value=self.predictor), \
             patch("src.pipeline.features.pseudo_builder.get_feature_builder", return_value=self.builder):
            job = wf.handle_predict_job({"race_id": self.races[0][0]}, self.storage)
        self.assertEqual(job["status"], "error")       # ワーカーは失敗を返し Cloud Tasks に再試行させる

    def test_run_local_reports_not_persisted(self):
        self.storage.writable = False
        clock = {"now": datetime(2026, 10, 4, 8, 30, tzinfo=JST)}
        res = wf.run_race_day_local(
            self.storage, DATE, now_fn=lambda: clock["now"],
            sleep_fn=lambda s: clock.__setitem__("now", clock["now"] + timedelta(seconds=s)),
            builder=self.builder, predictor=self.predictor,
        )
        self.assertEqual({r["status"] for r in res}, {"not_persisted"})

    # ── ワーカー（Cloud Tasks が呼ぶ側）───────────────────

    def test_handle_predict_job(self):
        with patch.object(wf, "load_predictor", return_value=self.predictor), \
             patch("src.pipeline.features.pseudo_builder.get_feature_builder", return_value=self.builder):
            ok = wf.handle_predict_job({"job_kind": "predict_race", "race_id": self.races[0][0]}, self.storage)
            self.assertEqual(ok["status"], "success")
        self.assertEqual(wf.handle_predict_job({}, self.storage)["status"], "error")


if __name__ == "__main__":
    unittest.main()
