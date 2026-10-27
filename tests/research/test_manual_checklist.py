"""manual_checklist.py のテスト（サイトへのアクセスは一切行わない純粋ロジックのみ）。"""
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


def test_build_rows_includes_sire_name():
    anc = {"a": {"precision": "year", "birthday": "1976", "name": "Kampala"}}
    zone = [{"stallion_id": "a", "min_gen": "1", "n_horses": "9"}]
    rows = mc.build_rows(anc, zone, {}, set(), 10, "https://x.test/p", {"a": {"sire": "Kalamoun", "sire_id": "s"}})
    assert rows[0]["sire"] == "Kalamoun"
    assert mc.build_rows(anc, zone, {}, set(), 10, "https://x.test/p")[0]["sire"] == ""


# --- a05e360（レビュー時に追加したテスト） ---


def test_slug_strips_apostrophes_and_symbols():
    assert mc.slug("Devil's Bag") == "devils-bag"
    assert mc.slug("Northern Dancer (can)") == "northern-dancer"


def test_slug_candidates_removes_stopwords_in_priority_order():
    cands = mc.slug_candidates("The Tetrarch")
    assert cands[0] == "tetrarch"
    assert "the-tetrarch" in cands  # ストップワード無しの候補も残る


def test_slug_candidates_removes_known_examples():
    assert mc.slug_candidates("Key to the Mint")[0] == "key-mint"
    assert mc.slug_candidates("Tale of the Cat")[0] == "tale-cat"
    assert mc.slug_candidates("In The Wings")[0] == "wings"


def test_slug_candidates_includes_numbered_suffixes():
    cands = mc.slug_candidates("Kampala")
    assert "kampala" in cands
    assert "kampala-1" in cands


def test_latin_name_prefers_name_en_when_it_has_latin_chars():
    rec = {"name": "サンデーサイレンス", "name_en": "Sunday Silence"}
    assert mc.latin_name(rec, "サンデーサイレンス") == "Sunday Silence"


def test_latin_name_falls_back_to_stripping_katakana_from_ancestor_name():
    assert mc.latin_name({}, "ノーザンテーストNorthern Taste") == "Northern Taste"


def test_latin_name_returns_empty_when_no_latin_chars_anywhere():
    assert mc.latin_name({}, "サンデーサイレンス") == ""


def test_build_rows_filters_year_precision_and_min_year_and_sorts_by_impact():
    ancestors = {
        "a": {"name": "A", "birthday": "1970", "precision": "year"},
        "b": {"name": "B", "birthday": "1970", "precision": "year"},
        "c": {"name": "C", "birthday": "1970-01-02", "precision": "day"},  # 除外: day precision
        "d": {"name": "D", "birthday": "1890", "precision": "year"},  # 除外: 1900 未満
    }
    zone = [
        {"stallion_id": "a", "min_gen": "2", "n_horses": "10"},
        {"stallion_id": "b", "min_gen": "3", "n_horses": "99"},
        {"stallion_id": "c", "min_gen": "1", "n_horses": "50"},
        {"stallion_id": "d", "min_gen": "1", "n_horses": "50"},
    ]
    rows = mc.build_rows(ancestors, zone, {}, set(), limit=10, base_url="https://sporthorse-data.com/pedigree")
    assert [r["horse_id"] for r in rows] == ["b", "a"]  # 影響度(n_horses)降順
    assert rows[0]["url"].startswith("https://sporthorse-data.com/pedigree/")


def test_build_rows_excludes_manually_already_done_horses():
    ancestors = {"a": {"name": "A", "birthday": "1970", "precision": "year"}}
    zone = [{"stallion_id": "a", "min_gen": "1", "n_horses": "10"}]
    rows = mc.build_rows(ancestors, zone, {}, {"a"}, limit=10, base_url="https://sporthorse-data.com/pedigree")
    assert rows == []


def test_render_embeds_json_data_and_escapes_closing_script_tags():
    rows = [{"horse_id": "a", "name": "A</script>", "name_en": "", "year": 1970, "impact": 1,
             "url": "https://example.com/a", "candidates": [], "sire": ""}]
    html_out = mc.render(rows)
    assert "<\\/script>" in html_out
    assert "</script>" not in html_out.split("const DATA")[1].split(";")[0]
