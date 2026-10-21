from src.scraper.profile_extra import id_from_href, parse_profile_rows, parse_sale_price


def test_sale_price_formats():
    assert parse_sale_price("-")["price_man"] is None
    d = parse_sale_price("1,836万円 (2014年 北海道セレクションセール)")
    assert (d["price_man"], d["sale_year"], d["sale_name"]) == (1836.0, 2014, "北海道セレクションセール")
    assert parse_sale_price("1億2,000万円 (2020年 セレクトセール)")["price_man"] == 12000.0
    assert parse_sale_price("1.1億円 (2019年 セレクトセール)")["price_man"] == 11000.0
    d = parse_sale_price("不明な書式")
    assert d["price_man"] is None and d["raw"] == "不明な書式"


def test_ids_and_rows():
    assert id_from_href("https://db.netkeiba.com/owner/x0a92b/", "owner") == "x0a92b"
    assert id_from_href("", "owner") == ""
    rows = [
        {"th": "馬主", "td": "A", "hrefs": ["https://db.netkeiba.com/owner/505007/"]},
        {"th": "セリ取引価格", "td": "500万円 (2015年 サマーセール)", "hrefs": []},
        {"th": "近親馬", "td": "x", "hrefs": ["https://db.netkeiba.com/horse/2009101869/", "https://db.netkeiba.com/horse/2010101667/"]},
    ]
    o = parse_profile_rows(rows)
    assert o["owner_id"] == "505007" and o["breeder_id"] == "" and o["sale_listed"] == 1 and o["sale_price_man"] == 500.0
    assert o["relatives"] == ["2009101869", "2010101667"]
    assert parse_profile_rows([])["sale_listed"] == 0
