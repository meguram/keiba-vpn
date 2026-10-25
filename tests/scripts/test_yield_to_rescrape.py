"""yield_to_rescrape.sh: 優先の取り直しが本当に動いているときだけ見送る（コマンドラインに名前が含まれるだけでは見送らない）。"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "scripts" / "cron" / "yield_to_rescrape.sh"


class TestYieldToRescrape(unittest.TestCase):
    def setUp(self) -> None:
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.root = Path(d.name)
        (self.root / "scripts" / "cron").mkdir(parents=True)
        shutil.copy(SRC, self.root / "scripts" / "cron" / "yield_to_rescrape.sh")
        self.script = self.root / "scripts" / "cron" / "yield_to_rescrape.sh"

    def _run(self) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(self.script), "t", "--", "echo", "RAN"], capture_output=True, text=True, timeout=20)

    def _spawn(self, *args: str) -> subprocess.Popen:
        p = subprocess.Popen(list(args), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: (p.kill(), p.wait()))
        time.sleep(0.4)
        return p

    def test_runs_when_nothing_is_rescraping(self) -> None:
        self.assertIn("RAN", self._run().stdout)

    def test_name_only_in_commandline_does_not_skip(self) -> None:
        self._spawn("bash", "-c", "sleep 30  # run_priority_rescrape.sh")
        self._spawn("sleep", "30", "run_priority_rescrape.sh")
        out = self._run().stdout
        self.assertIn("RAN", out)
        self.assertNotIn("SKIP", out)

    def test_skips_when_the_rescrape_script_is_running(self) -> None:
        runner = self.root / "run_priority_rescrape.sh"
        runner.write_text("#!/bin/bash\nsleep 30\n")
        self._spawn("bash", str(runner))
        out = self._run().stdout
        self.assertIn("SKIP", out)
        self.assertNotIn("RAN", out)

    def test_yield_disabled_file_wins(self) -> None:
        runner = self.root / "run_priority_rescrape.sh"
        runner.write_text("#!/bin/bash\nsleep 30\n")
        self._spawn("bash", str(runner))
        d = self.root / "data" / "local" / "meta" / "rescrape"
        d.mkdir(parents=True)
        (d / "YIELD_DISABLED").write_text("")
        self.assertIn("RAN", self._run().stdout)


if __name__ == "__main__":
    unittest.main()
