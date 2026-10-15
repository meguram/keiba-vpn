"""データの収集〜統合フロー図(build_data_flow_html)が、cron の登録内容と食い違っていないことの検査。"""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.scripts.docs import build_data_flow_html as F
from src.scripts.docs.build_cron_schedule_html import JOBS, parse

ROOT = Path(__file__).resolve().parents[2]


def _jobs():
    text = subprocess.run(["bash", str(ROOT / "scripts/cron/setup_all_cron.sh"), "show"], capture_output=True, text=True, check=True).stdout
    return parse(text)[0]


def test_every_job_key_in_the_flow_exists():
    keys = {k for r in F.ROWS for k, _ in r.collect + r.process if k} | {k for _, ks in F.LANES for k in ks}
    assert keys <= set(JOBS), f"cron の定義に無いジョブキー: {keys - set(JOBS)}"


def test_every_referenced_job_is_registered_in_cron():
    times = F.job_times(_jobs())
    missing = {k for r in F.ROWS for k, _ in r.collect + r.process if k and k != "backfill"} - set(times)
    assert not missing, f"cron に登録されていないジョブを図が参照している: {missing}"


def test_rendered_page_contains_the_cron_times():
    html = F.render(_jobs(), "test")
    for needle in ("毎日 07:30", "毎日 07:40", "毎日 08:00", "金 18:00", "<svg", "（cron 未登録）"[:0] or "データの収集〜加工〜統合フロー"):
        assert needle in html
    assert "（cron 未登録）" not in html


def test_compact_merges_times_per_day():
    assert F.compact(["毎日 19:40", "毎日 06:10"]) == "毎日 06:10・19:40"
    assert F.compact(["毎日 08:30", "金 19:30"]) == "毎日 08:30 ／ 金 19:30"
