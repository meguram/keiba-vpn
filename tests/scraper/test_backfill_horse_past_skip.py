import tempfile
import unittest
from pathlib import Path

from src.scraper import backfill as bf


def _ped(anchors_m, anchors_f):
    ancs = [{"generation": 5, "position": 16 + i, "horse_id": a} for i, a in enumerate(anchors_m)]
    ancs += [{"generation": 5, "position": i, "horse_id": a} for i, a in enumerate(anchors_f)]
    return {"ancestors": ancs}


class _Storage:
    def __init__(self, have):
        self.have = {k: set(v) for k, v in have.items()}
        self.docs = {}
        self.exists_calls = 0
        self.list_calls = []

    def batch_list_blobs(self, cat, year):
        self.list_calls.append((cat, year))
        return {k: 1.0 for k in self.have.get(cat, set()) if k.startswith(year)}

    def exists(self, cat, key):
        self.exists_calls += 1
        return key in self.have.get(cat, set())

    def load(self, cat, key):
        return self.docs.get(key)


class _Runner:
    def __init__(self, storage):
        self.storage = storage
        self.scraped, self.ped5 = [], []

    def scrape_horse(self, hid, skip_existing=False, with_history=False, save_pedigree_5gen=False):
        self.scraped.append((hid, save_pedigree_5gen))
        self.storage.have.setdefault("horse_result", set()).add(hid)
        if save_pedigree_5gen:
            self.storage.have.setdefault("horse_pedigree_5gen", set()).add(hid)
            self.storage.docs[hid] = _ped(["2010M%d" % len(hid)], ["2010F1"])

    def scrape_horse_pedigree_5gen(self, hid, skip_existing=False):
        self.ped5.append(hid)
        self.storage.have.setdefault("horse_pedigree_5gen", set()).add(hid)


class HorsePastSkipTest(unittest.TestCase):
    def _run(self, ids, storage):
        with tempfile.TemporaryDirectory() as t:
            prog = bf.BackfillProgress(Path(t) / "p.json")
            prog._data["past_horse_ids"] = ids
            runner = _Runner(storage)
            stats = bf._run_phase_horse_past(runner, prog)
            return runner, stats, prog

    def test_skips_listed_horses_without_individual_exists(self):
        ids = ["2015100001", "2015100002"]
        st = _Storage({"horse_result": ids, "horse_pedigree_5gen": ids + ["2010M1"]})
        for h in ids:
            st.docs[h] = _ped(["2010M1"], ["2010F1"])
        runner, stats, prog = self._run(ids, st)
        self.assertEqual(stats["horses_skipped"], 2)
        self.assertEqual(stats["horses_processed"], 0)
        self.assertEqual(runner.scraped, [])
        self.assertEqual(runner.ped5, [])
        self.assertEqual(st.exists_calls, 0)
        self.assertEqual(sorted(prog._data["past_horses_done"]), ids)
        self.assertEqual(prog._data["paternal_anchor_pending"], ["2010F1"])

    def test_new_horse_saves_pedigree_in_same_fetch(self):
        st = _Storage({})
        runner, stats, _ = self._run(["2015100009"], st)
        self.assertEqual(runner.scraped, [("2015100009", True)])
        self.assertEqual(runner.ped5.count("2015100009"), 0)
        self.assertEqual(stats["horses_processed"], 1)
        self.assertEqual(stats["anchors_m"], 1)

    def test_stale_listing_falls_back_to_exists(self):
        ids = ["2015100003"]
        st = _Storage({"horse_result": ids, "horse_pedigree_5gen": []})
        st.batch_list_blobs = lambda cat, year: {}
        st.docs[ids[0]] = _ped([], [])
        runner, stats, _ = self._run(ids, st)
        self.assertEqual(stats["horses_skipped"], 1)
        self.assertEqual(runner.ped5, ids)


if __name__ == "__main__":
    unittest.main()
