"""馬柱の馬名の右の印（B = ブリンカー）を読む。"""

from __future__ import annotations

import unittest

from src.scraper.parsers import ShutubaPastParser

ROW = '''<tr class="HorseList"><td class="Waku1"><span>1</span></td><td class="Waku">1</td>
<td class="Horse_Info"><div class="Horse02"><a href="https://db.netkeiba.com/horse/2023104856" target="_blank"> マイケルバローズ</a>{marks}</div></td></tr>'''


def _html(*marks: str) -> str:
    rows = "".join(ROW.format(marks="".join(f'<span class="Mark">{m}</span>' for m in ms)) for ms in marks)
    return f'<table class="Shutuba_Table Shutuba_Past5_Table table02">{rows}</table>'


class BlinkerMarkTest(unittest.TestCase):
    def test_blinker_mark(self):
        out = ShutubaPastParser().parse(_html(("B",), ()), "202601010101")["entries"]
        self.assertEqual([e["blinker"] for e in out], [1, 0])
        self.assertEqual(out[0]["marks"], ["B"])
        self.assertEqual(out[1]["marks"], [])

    def test_other_marks_kept_and_not_blinker(self):
        out = ShutubaPastParser().parse(_html(("S",), ("B", "H")), "202601010101")["entries"]
        self.assertEqual([e["blinker"] for e in out], [0, 1])
        self.assertEqual(out[1]["marks"], ["B", "H"])

    def test_no_name_cell_means_no_key(self):
        html = '<table class="Shutuba_Past5_Table"><tr class="HorseList"><td class="Waku1">1</td><td class="Waku">1</td><td><a href="https://db.netkeiba.com/horse/2023104856">x</a></td></tr></table>'
        out = ShutubaPastParser().parse(html, "1")["entries"]
        self.assertEqual(len(out), 1)
        self.assertNotIn("blinker", out[0])


if __name__ == "__main__":
    unittest.main()
