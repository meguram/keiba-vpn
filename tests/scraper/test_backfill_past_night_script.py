"""scripts/cron/backfill_past_night.sh: 通信エラーのみで止まったら少し置いて同じ起動の中で再開し、HTTP エラーでは再開しない。"""

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/cron/backfill_past_night.sh"


def _run(tmp_path: Path, halts: list[dict | None]) -> tuple[int, int, str]:
    """補填の本体を偽物に差し替えて起動スクリプトを走らせる。halts[i] は i 回目の実行が書く HALT（None なら書かない）。"""
    meta = tmp_path / "data/local/meta"
    meta.mkdir(parents=True)
    (tmp_path / "halts.json").write_text(json.dumps(halts))
    fake = tmp_path / "fake_python"
    fake.write_text(f"""#!/usr/bin/env bash
if [ "$1" = "-c" ]; then exec {sys.executable} "$@"; fi
exec {sys.executable} - <<'PY'
import json, pathlib
root = pathlib.Path({str(tmp_path)!r})
n = int((root / "calls").read_text()) if (root / "calls").exists() else 0
(root / "calls").write_text(str(n + 1))
halts = json.loads((root / "halts.json").read_text())
h = halts[n] if n < len(halts) else None
if h is not None:
    (root / "data/local/meta/backfill_past_halt.json").write_text(json.dumps(h))
PY
""")
    fake.chmod(0o755)
    env = {**os.environ, "PYTHON": str(fake), "BACKFILL_PAST_TRANSPORT_COOLOFF": "0", "BACKFILL_PAST_LOCK": str(tmp_path / "lock")}
    rc = subprocess.run(["bash", str(SCRIPT), str(tmp_path)], env=env, timeout=60).returncode
    calls = int((tmp_path / "calls").read_text())
    return rc, calls, (tmp_path / "logs/backfill_past.log").read_text()


def test_transport_only_halt_restarts_in_same_invocation(tmp_path):
    rc, calls, log = _run(tmp_path, [{"bad": 0, "transport_errors": 3}, {"bad": 0, "transport_errors": 3}, None])
    assert rc == 0 and calls == 3
    assert log.count("HALT 自動解除") == 2
    assert not (tmp_path / "data/local/meta/backfill_past_halt.json").exists()


def test_http_error_halt_is_kept_for_a_human(tmp_path):
    rc, calls, log = _run(tmp_path, [{"bad": 1, "transport_errors": 0}])
    assert calls == 1
    assert "HALT 継続" in log
    assert (tmp_path / "data/local/meta/backfill_past_halt.json").exists()
