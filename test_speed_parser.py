#!/usr/bin/env python3
"""
タイム指数パーサーのテストスクリプト
"""
import sys
sys.path.insert(0, '.')

from scraper.client import NetkeibaClient
from scraper.parsers import SpeedIndexParser

race_id = "202606020409"
url = f"https://race.netkeiba.com/race/speed.html?race_id={race_id}"

print(f"Fetching: {url}")
client = NetkeibaClient(auto_login=True)
html = client.fetch(url)

print(f"\nHTML length: {len(html)} bytes")
print("\nParsing...")

parser = SpeedIndexParser()
data = parser.parse(html, race_id=race_id)

print(f"\nTotal entries parsed: {len(data['entries'])}")

if len(data['entries']) > 0:
    print("\n=== First 5 entries ===")
    for i, entry in enumerate(data['entries'][:5]):
        print(f"\n{i+1}. 馬番{entry['horse_number']}: {entry['horse_name']}")
        print(f"   time_index_m: {entry.get('time_index_m', 'N/A')}")
        print(f"   speed_max: {entry['speed_max']}")
        print(f"   speed_recent: {entry['speed_recent']}")

    # アスターブジエ（馬番4）を探す
    print("\n=== Looking for 馬番4 (アスターブジエ) ===")
    for entry in data['entries']:
        if entry['horse_number'] == 4:
            print(f"Found: {entry['horse_name']}")
            print(f"time_index_m: {entry.get('time_index_m', 'N/A')}")
            print(f"speed_recent: {entry['speed_recent']}")
            break
    else:
        print("Not found!")

# HTMLの一部を表示（デバッグ用）
print("\n=== HTML structure (first HorseList row) ===")
from bs4 import BeautifulSoup
soup = BeautifulSoup(html, "html.parser")
first_row = soup.select_one("tr.HorseList")
if first_row:
    print("Columns in first row:")
    for i, td in enumerate(first_row.select("td")):
        classes = td.get('class', [])
        text = td.get_text(strip=True)[:20]
        print(f"  {i}: class={classes} text='{text}'")
