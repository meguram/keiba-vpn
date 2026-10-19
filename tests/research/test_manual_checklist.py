import json
import re

from src.research.pedigree import manual_checklist as mc


def test_slug():
    assert mc.slug("Shecky Greene") == "shecky-greene"
    assert mc.slug("Harlan\xa0(米)") == "harlan"
    assert mc.slug("Devil's Bag") == "devils-bag"
    assert mc.slug("Man o'War") == "man-owar" and mc.slug("Mr. Prospector") == "mr-prospector" and mc.slug("Kris S.") == "kris-s"
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


def test_slug_candidates_cover_known_exceptions():
    c = mc.slug_candidates("Key to the Mint")
    assert c[0] == "key-mint" and "key-mint-0" in c and "key-to-the-mint" in c
    assert mc.slug_candidates("The Tetrarch")[0] == "tetrarch"
    assert mc.slug_candidates("Kampala")[:3] == ["kampala", "kampala-0", "kampala-1"]
    assert mc.slug_candidates("Devil's Bag")[0] == "devils-bag"
    assert len(set(mc.slug_candidates("Hold Your Peace"))) == len(mc.slug_candidates("Hold Your Peace"))


def test_slug_drops_of():
    assert mc.slug_candidates("Tale of the Cat")[0] == "tale-cat"
    assert mc.slug_candidates("Cox's Ridge")[0] == "coxs-ridge"


def test_slug_drops_in_and_lists_wider_variants():
    c = mc.slug_candidates("In The Wings")
    assert c[0] == "wings" and "in-the-wings" in c
    h = mc.slug_candidates("Hold Your Peace")
    assert h[0] == "hold-peace" and "hold-your-peace" in h          # your も取り除いた形が先、そのままの形も候補に残す


def test_slug_drops_all_stopwords_first_but_keeps_fallbacks():
    assert mc.slug_candidates("Wild Again")[0] == "wild"
    assert mc.slug_candidates("Be My Guest")[0] == "guest"
    assert mc.slug_candidates("Who Is It")[0] == "who-is-it"            # すべてがストップワードならそのままの形
    c = mc.slug_candidates("Key to the Mint")
    assert c[0] == "key-mint" and "key-to-the-mint" in c
    c = mc.slug_candidates("Our Emblem")
    assert c[0] == "emblem" and "our-emblem" in c
    assert len(mc.slug_candidates("Hold Your Peace")) <= 14


def test_latin_name_prefers_a_field_with_letters():
    assert mc.latin_name({"name": "ヨハネスブルグ", "name_en": "Johannesburg"}, "ヨハネスブルグ") == "Johannesburg"
    assert mc.latin_name({"name": "Kris", "name_en": "クリス"}, "Kris") == "Kris"
    assert mc.latin_name({}, "ノーザンテーストNorthern Taste(加)") == "Northern Taste(加)"
    assert mc.latin_name({}, "ハイセイコー") == ""
    rows = mc.build_rows({"a": {"precision": "year", "birthday": "1999", "name": "ヨハネスブルグ"}},
                         [{"stallion_id": "a", "min_gen": "1", "n_horses": "9"}], {"a": {"name": "ヨハネスブルグ", "name_en": "Johannesburg"}}, set(), 10, "https://x.test/p")
    assert rows[0]["candidates"] and rows[0]["candidates"][0] == "https://x.test/p/johannesburg"
