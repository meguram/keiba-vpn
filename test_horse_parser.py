#!/usr/bin/env python3
"""
馬ページのパーサーをテスト
"""
import sys
sys.path.insert(0, '.')

from scraper.client import NetkeibaClient
from bs4 import BeautifulSoup

horse_id = "2020103111"  # アスターブジエ

print("Fetching horse result page...")
client = NetkeibaClient(auto_login=True)

# ログイン状態を確認
if client.session.cookies.get('netkeiba_pc', domain='.netkeiba.com'):
    print("✓ Logged in (netkeiba_pc cookie exists)")
else:
    print("✗ Not logged in")

result_url = f"https://db.netkeiba.com/horse/result/{horse_id}/"
print(f"\nFetching: {result_url}")
html = client.fetch(result_url)

print(f"HTML length: {len(html)} bytes")

# テーブル構造を確認
soup = BeautifulSoup(html, "html.parser")
table = soup.select_one("table.db_h_race_results, table[class*='race_results']")

if table:
    print("\n=== Table found ===")

    # ヘッダー行を確認
    header_row = table.select_one("tr")
    if header_row:
        headers = [th.get_text(strip=True) for th in header_row.select("th")]
        print(f"\nHeaders ({len(headers)} columns):")
        for i, h in enumerate(headers):
            print(f"  {i}: {h}")

    # 最初のデータ行を確認
    data_rows = table.select("tr")[1:]
    if data_rows:
        first_row = data_rows[0]
        cols = first_row.select("td")
        print(f"\nFirst race data ({len(cols)} columns):")
        for i, col in enumerate(cols):
            text = col.get_text(strip=True)[:30]
            print(f"  {i}: {text}")

        # タイム指数を探す
        print("\n=== Looking for 指数 column ===")
        for i, col in enumerate(cols):
            text = col.get_text(strip=True)
            if text.isdigit() and len(text) == 2:  # 2桁の数字
                print(f"  Column {i}: {text} (candidate)")
else:
    print("✗ Table not found")
    print("\nSearching for any table...")
    all_tables = soup.select("table")
    print(f"Found {len(all_tables)} tables")
