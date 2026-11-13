"""VPS 全体の停止の検知（src/ops/vps_alive.py）: 2 回続けて失敗で 1 回だけ知らせ、戻ったら 1 回知らせる。"""

import json

from src.ops import vps_alive as V


def test_two_failures_alert_once_then_recovery_once(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "STATE", tmp_path / "s.json")
    sent = []

    def note(title, lines, severity="warning"):
        sent.append((title, severity))

    seq = [(False, "timeout"), (False, "timeout"), (False, "timeout"), (True, "keiba-ui-1"), (True, "keiba-ui-1")]
    for ok, d in seq:
        V.main([], probe_fn=lambda ok=ok, d=d: (ok, d), notify=note, now=0)
    assert [s[1] for s in sent] == ["critical", "normal"] and "停止の疑い" in sent[0][0]


def test_one_blip_does_not_alert(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "STATE", tmp_path / "s.json")
    sent = []
    for ok in (True, False, True):
        V.main([], probe_fn=lambda ok=ok: (ok, "x"), notify=lambda *a, **k: sent.append(a), now=0)
    assert sent == []
    assert json.loads((tmp_path / "s.json").read_text())["fails"] == 0


def test_probe_rejects_non_prod_or_non_json(monkeypatch):
    class R:
        status = 200

        def __init__(self, body):
            self.body = body

        def read(self, n):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(V.urllib.request, "urlopen", lambda req, timeout: R(b"<html>Cloudflare Access</html>"))
    assert V.probe()[0] is False
    monkeypatch.setattr(V.urllib.request, "urlopen", lambda req, timeout: R(json.dumps({"env": "prod", "release": "r1"}).encode()))
    assert V.probe() == (True, "r1")


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "STATE", tmp_path / "s.json")
    V.main(["--dry-run"], probe_fn=lambda: (False, "x"), notify=lambda *a, **k: None)
    assert not (tmp_path / "s.json").exists()


def test_failed_notification_is_retried_next_time(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "STATE", tmp_path / "s.json")
    calls = []

    def note(title, lines, severity="warning"):
        calls.append(title)
        return {"ok": len(calls) > 1, "error": "webhook_not_configured"}

    for _ in range(3):
        V.main([], probe_fn=lambda: (False, "timeout"), notify=note, now=0)
    assert len(calls) == 2 and json.loads((tmp_path / "s.json").read_text())["down"] is True   # 1 回目は送れず、2 回目で送れた
