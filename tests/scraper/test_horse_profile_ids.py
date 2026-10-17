"""馬のプロフィール表から馬主・生産者の ID（リンク）を取る。"""

from bs4 import BeautifulSoup

from src.scraper.parsers import HorseParser

HTML = """
<html><body><div class="db_head_name"><h1>テストホース</h1></div>
<table class="db_prof_table">
<tr><th>生年月日</th><td>2021年3月1日</td></tr>
<tr><th>調教師</th><td><a href="/trainer/01088/">調教師A</a></td></tr>
<tr><th>馬主</th><td><a href="/owner/005509/">サンデーレーシング</a></td></tr>
<tr><th>生産者</th><td><a href="/breeder/373126/">ノーザンファーム</a></td></tr>
<tr><th>産地</th><td>安平町</td></tr>
</table></body></html>
"""


def test_owner_and_breeder_ids_are_parsed():
    info = HorseParser()._parse_profile(BeautifulSoup(HTML, "html.parser"))
    assert info["owner"] == "サンデーレーシング" and info["owner_id"] == "005509"
    assert info["breeder"] == "ノーザンファーム" and info["breeder_id"] == "373126"


def test_no_link_means_no_id():
    html = HTML.replace('<a href="/owner/005509/">サンデーレーシング</a>', "個人名")
    info = HorseParser()._parse_profile(BeautifulSoup(html, "html.parser"))
    assert info["owner"] == "個人名" and "owner_id" not in info
