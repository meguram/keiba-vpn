"""馬別調教ページの表記（旧: 曜日つき、新: 時刻つき）を両方読めることのテスト。"""

from __future__ import annotations

import unittest

from src.scraper.parsers import TrainingParser

ROW_NEW = """<table class="race_table_01"><caption>2026/10/17  東京11R富士S(GII)結果 ： 着</caption>
<tr><th>日付</th><th>コース</th></tr>
<tr><td colspan="9">[短評]２頭を追走し、直線で</td></tr>
<tr><td class="txt_center">2026/10/14 06:21:00(水)</td><td>美Ｗ</td><td>良</td><td>戸崎圭</td>
<td class="TrainingTimeData txt_l"><ul class="TrainingTimeDataList"><li>83.6</li><li>67.0</li><li>51.9</li><li>37.7</li><li>12.1</li></ul>
<p class="TrainingHeisou">内トナリティ馬也と併せ０秒２遅れ</p><p class="TrainingHeisou">外トゥーザムーン馬也と併せ０秒１遅れ</p></td>
<td>8</td><td>強め</td><td>平行線</td><td>C</td><td></td></tr></table>"""

ROW_OLD = """<table class="race_table_01"><caption>2023/10/29  東京11R 天皇賞(秋) 結果 ： 1着</caption>
<tr><td class="txt_center">2023/10/25(水)</td><td>美Ｗ</td><td>良</td><td>助手</td>
<td class="TrainingTimeData"><ul class="TrainingTimeDataList"><li>84.7</li><li>67.6</li><li>52.5</li><li>37.3</li><li>11.3</li></ul></td>
<td>4</td><td>馬也</td><td>気配抜群</td><td>A</td><td></td></tr></table>"""


class LayoutTest(unittest.TestCase):
    def test_new_layout_with_time_of_day(self):
        d = TrainingParser().parse(ROW_NEW, horse_id="H")
        self.assertEqual(len(d["entries"]), 1)
        e = d["entries"][0]
        self.assertEqual((e["date"], e["day_of_week"], e["time_of_day"]), ("2026-10-14", "水", "06:21"))
        self.assertEqual(e["lap_times"], [83.6, 67.0, 51.9, 37.7, 12.1])
        self.assertIn("トナリティ", e["partner_notes"])
        self.assertEqual(e["comment"], "２頭を追走し、直線で")

    def test_old_layout_still_parses(self):
        e = TrainingParser().parse(ROW_OLD, horse_id="H")["entries"][0]
        self.assertEqual((e["date"], e["day_of_week"]), ("2023-10-25", "水"))
        self.assertNotIn("time_of_day", e)

    def test_weekday_filled_when_missing(self):
        html = ROW_NEW.replace("(水)", "")
        e = TrainingParser().parse(html, horse_id="H")["entries"][0]
        self.assertEqual(e["day_of_week"], "水")


if __name__ == "__main__":
    unittest.main()
