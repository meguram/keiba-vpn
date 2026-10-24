import time

from src.utils import race_result_availability as rra
from src.utils.race_card_merge import load_merged_race_card


class _Storage:
    def __init__(self, delay=0.0, confirmed=(), flash=()):
        self.delay, self.confirmed, self.flash = delay, set(confirmed), set(flash)
        self.loads, self.lists = [], []

    def list_keys(self, category, year=None):
        self.lists.append(category)
        return list(self.confirmed if category == "race_result" else self.flash)

    def load(self, category, race_id):
        time.sleep(self.delay)
        self.loads.append((category, race_id))
        if category == "race_shutuba":
            return {"race_id": race_id, "race_name": "テスト", "distance": 1600, "surface": "芝", "entries": []}
        return None

    def exists(self, category, race_id):
        return False


def test_keys_are_cached_for_a_short_time_across_dates():
    rra._keys_cache.clear()
    st = _Storage(confirmed={"a"}, flash={"b"})
    rra.batch_race_result_status(st, ["a", "b", "c"], date="20261024")
    rra.batch_race_result_status(st, ["a", "b"], date="20261025")     # 同じ年 → 一覧は再取得しない
    assert st.lists.count("race_result") == 1 and st.lists.count("race_result_on_time") == 1


def test_keys_cache_expires():
    rra._keys_cache.clear()
    st = _Storage(confirmed={"a"})
    rra._list_keys_cached(st, "race_result", "2026", now=0.0)
    rra._list_keys_cached(st, "race_result", "2026", now=rra._KEYS_TTL_SEC - 1)
    rra._list_keys_cached(st, "race_result", "2026", now=rra._KEYS_TTL_SEC + 1)
    assert st.lists.count("race_result") == 2


def test_load_merged_race_card_skips_result_when_told():
    st = _Storage()
    load_merged_race_card("r1", storage=st, with_result=False)
    assert ("race_result", "r1") not in st.loads and ("race_shutuba", "r1") in st.loads
    st2 = _Storage()
    load_merged_race_card("r1", storage=st2)
    assert ("race_result", "r1") in st2.loads


def test_parallel_card_loading_matches_serial_and_is_faster():
    from concurrent.futures import ThreadPoolExecutor

    ids = [f"r{i}" for i in range(16)]
    st = _Storage(delay=0.05)

    def one(rid):
        return rid, load_merged_race_card(rid, storage=st, with_result=False)

    t0 = time.time()
    serial = [one(r) for r in ids]
    t_serial = time.time() - t0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        parallel = list(pool.map(one, ids))
    t_parallel = time.time() - t0
    assert serial == parallel and t_parallel < t_serial / 3
