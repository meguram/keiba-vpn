import json
import re

from src.research.pedigree import manual_checklist as mc


def test_slug():
    assert mc.slug("Shecky Greene") == "shecky-greene"
    assert mc.slug("Harlan\xa0(米)") == "harlan"
    assert mc.slug("Devil's Bag") == "devil-s-bag"
    assert mc.slug("ノーザンテースト") == ""


def test_build_rows_orders_by_impact_and_skips_resolved():
    anc = {h: {"precision": "year", "birthday": "1970", "name": n} for h, n in (("a", "A One"), ("b", "B Two"), ("c", "C Three"))}
    anc["d"] = {"precision": "day", "birthday": "1970-01-02", "name": "D"}
    zone = [{"stallion_id": "a", "min_gen": "1", "n_horses": "5"}, {"stallion_id": "b", "min_gen": "2", "n_horses": "9"},
            {"stallion_id": "c", "min_gen": "5", "n_horses": "99"}, {"stallion_id": "d", "min_gen": "1", "n_horses": "50"}]
    rows = mc.build_rows(anc, zone, {"b": {"name": "Bee Two"}}, {"a"}, 10, "https://example.test/ped/")
    assert [r["horse_id"] for r in rows] == ["b"]                 # a は手入力済み、c は 4 代以降、d は日付つき
    assert rows[0]["url"] == "https://example.test/ped/bee-two" and rows[0]["impact"] == 9


def test_render_embeds_json_safely():
    html_text = mc.render([{"horse_id": "x", "name": "</script>", "name_en": "", "year": 1970, "impact": 1, "url": "u"}])
    data = re.search(r"const DATA = (.*?);\nconst MON", html_text, re.S)[1]
    assert "</" not in data and json.loads(data.replace("<\\/", "</"))[0]["horse_id"] == "x"
