from src.scripts.data.acq_coverage import is_jra_race_id
from src.scripts.data.acq_history_extract import is_jra_race_id as is_jra2
from src.scripts.data.acq_profile_pilot import extract_rows


def test_jra_race_id_codes():
    assert is_jra_race_id("202206010406") and is_jra2("202510030811")
    assert not is_jra_race_id("202544122909")  # 地方（場コード 44）
    assert not is_jra_race_id("") and not is_jra_race_id("2022")


def test_extract_profile_rows_keeps_headers_text_and_links_only():
    html = (
        '<div class="horse_title"><h1>テスト馬</h1></div>'
        '<table class="db_prof_table"><tr><th>馬主</th><td><a href="https://db.netkeiba.com/owner/123/">A社</a></td></tr>'
        '<tr><th>セリ取引価格</th><td>1,000万円 (2020年 セレクトセール)</td></tr></table>'
    )
    d = extract_rows(html)
    assert d["name"] == "テスト馬"
    assert d["rows"][0] == {"th": "馬主", "td": "A社", "hrefs": ["https://db.netkeiba.com/owner/123/"]}
    assert d["rows"][1]["td"].startswith("1,000万円")
    assert extract_rows("<html></html>")["rows"] == []
