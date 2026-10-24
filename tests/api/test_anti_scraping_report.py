import json
import time

from src.scripts.maintenance import anti_scraping_report as R


def test_summarize_and_load(tmp_path):
    now = time.time()
    day = time.strftime("%Y%m%d", time.gmtime(now + 9 * 3600))
    rows = [
        {"ts": now - 10, "kind": "rate", "action": "block", "ip": "203.0.113.1", "ua": "python-requests/2", "reason": "d60"},
        {"ts": now - 20, "kind": "honeypot", "action": "block", "ip": "203.0.113.1", "ua": "python-requests/2", "reason": "honeypot"},
        {"ts": now - 30, "kind": "bot", "action": "detect", "ip": "203.0.113.2", "ua": "curl/8", "reason": "bot_score:ua_missing"},
        {"ts": now - 99999, "kind": "rate", "action": "block", "ip": "x", "reason": "old"},
    ]
    (tmp_path / f"events-{day}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\nnot-json\n", encoding="utf-8")
    ev = R.load_events(tmp_path, now - 3600, now)
    assert len(ev) == 3
    s = R.summarize(ev, 1.0)
    assert s["blocks"] == 2 and s["top_ips"][0] == ("203.0.113.1", 2)
    assert any("ブロック 2" in line for line in R.render_lines(s))


def test_main_dry_run_prints_and_does_not_send(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("KEIBA_SLACK_NOTIFY", raising=False)
    assert R.main(["--dir", str(tmp_path), "--hours", "1"]) == 0
    assert "記録 0 件" in capsys.readouterr().out
