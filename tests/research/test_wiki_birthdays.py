from src.research.pedigree import wiki_birthdays as wb


def test_citation_date_is_not_picked():
    # Spearmint: 出典内の 11 July 1906 ではなく本文の April 6, 1903
    raw = ('April 6, 1903<ref name="a">{{cite news|url=http://x/OW19060711 |title=T '
           '|date=11 July 1906 |access-date=2012-02-15}}</ref>\n| country = UK')
    assert wb.parse_date(raw) == (1903, 4, 6)


def test_formats():
    assert wb.parse_date("{{Birth date|1913|5|16}}") == (1913, 5, 16)
    assert wb.parse_date("{{birth date|df=yes|1920|4|4}}") == (1920, 4, 4)
    assert wb.parse_date("24 January 1935") == (1935, 1, 24)
    assert wb.parse_date("January 24, 1935") == (1935, 1, 24)
    assert wb.parse_date("March 1920") == (1920, 3, None)
    assert wb.parse_date("1927") == (1927, None, None)
    assert wb.parse_date("unknown") is None


def test_foaled_field_ignores_following_fields():
    wt = "{{Infobox racehorse\n| sire = X\n| foaled = 1912\n| country = [[United States]]\n| death_date = 1930\n}}"
    assert wb.foaled_field(wt) == (1912, None, None)


def test_norm():
    assert wb.norm("Spearmint (horse)") == wb.norm("Spearmint")
