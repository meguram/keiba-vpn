import numpy as np
import pandas as pd

from src.pipeline.megu_index import v3a_db as D


class _Session:
    def __init__(self, fail=False):
        self.calls, self.commits, self.rollbacks, self.fail = [], 0, 0, fail

    def execute(self, sql, params=None):
        if self.fail:
            raise RuntimeError("db down")
        self.calls.append(params)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def _result():
    return pd.DataFrame({
        "race_id": ["r1", "r1", "r2"], "horse_id": ["h1", "h2", "h3"], "finish_time_sec": [95.1, 95.4, 110.0],
        "megu_index_v3a_abs_robust": [52.34, np.nan, 61.0], "megu_v3a_status": ["ok", "ok", "steeplechase"],
        "v3a_par_sec": [96.0, 96.0, 111.0], "v3a_track_sec": [0.1, 0.1, 0.0], "v3a_corrected_sec": [95.0, 95.3, 110.0],
    })


def test_hook_is_off_by_default_and_without_v3a():
    s = _Session()
    assert D.maybe_upsert_for_result(s, _result(), True, environ={}) == 0
    assert D.maybe_upsert_for_result(s, _result(), False, environ={"KEIBA_MEGU_V3A_DB": "1"}) == 0
    assert s.calls == [] and s.commits == 0


def test_hook_upserts_only_usable_rows_when_enabled():
    s = _Session()
    n = D.maybe_upsert_for_result(s, _result(), True, environ={"KEIBA_MEGU_V3A_DB": "1"})
    assert n == 1 and s.commits == 1                                     # 絶対指数が NaN の行と障害は入れない
    rec = s.calls[0][0]
    assert rec["race_id"] == "r1" and rec["horse_id"] == "h1" and rec["megu_index"] == 52.3 and rec["mv"] == "v3a"
    assert rec["delta_pace_sec"] is None and rec["computation_status"] == "valid"


def test_db_failure_does_not_stop_the_daily_job():
    s = _Session(fail=True)
    assert D.maybe_upsert_for_result(s, _result(), True, environ={"KEIBA_MEGU_V3A_DB": "1"}) == 0
    assert s.rollbacks == 1


def test_daily_rows_match_bulk_conversion():
    # 一括（parquet）と日次（計算結果）は同じ変換を通るので、同じ入力なら同じ行になる
    df = _result()
    bulk = D.build_index_rows(df)
    daily = D.build_index_rows(df.sample(frac=1, random_state=1).reset_index(drop=True)).sort_values(["race_id", "horse_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(bulk.sort_values(["race_id", "horse_id"]).reset_index(drop=True), daily)
