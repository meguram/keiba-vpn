"""VPS の見張り（keiba-sched-status）の有効化の条件: 今の版に見張りのコードと一覧があるときだけ（古い版で有効にすると毎回 import に失敗する）。"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_sched_status_timer_is_enabled_only_when_the_release_has_the_code():
    s = (REPO / "scripts/ops/vps/vps_03_units.sh").read_text(encoding="utf-8")
    i = s.index("systemctl enable --now keiba-sched-status.timer")
    cond = s[s.rindex("if ", 0, i):i]
    for need in ("/opt/keiba-fetch/current/src/ops/sched_status.py", "/opt/keiba-fetch/current/config/scheduled_jobs.json", "GCS_BUCKET"):
        assert need in cond, need
    assert re.search(r"install -d .*sched_status", cond + s[i - 200:i])
