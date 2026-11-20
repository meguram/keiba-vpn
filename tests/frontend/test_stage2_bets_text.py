"""第 2 段の推奨の画面（Stage2Bets.tsx）の決め打ちの文言に、回収率の数字を書かない（試行中のあいだ。bet-optimizer の申し送り 2026-11-20）。"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "frontend/components/Stage2Bets.tsx"


def test_stage2_screen_has_no_return_rate_numbers():
    text = SRC.read_text(encoding="utf-8")
    assert "回収率は確かめ中" in text
    for line in text.splitlines():
        if "回収率" in line:
            assert not re.search(r"\d+(\.\d+)?\s*[%％倍]|〜\s*\d", line), line
    assert "107" not in text
