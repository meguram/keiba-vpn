"""会議のライブ（src/scripts/docs/council_live.py）: JSONL の解析・ラベル付け・伏せ字・差分の読み、と SSE。"""

from __future__ import annotations

import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from src.scripts.docs import council_live as cl


def _a(ts: str, *blocks: dict) -> dict:
    return {"type": "assistant", "timestamp": ts, "message": {"role": "assistant", "content": list(blocks)}}


def _u(ts: str, content, tur=None) -> dict:
    o = {"type": "user", "timestamp": ts, "message": {"role": "user", "content": content}}
    if tur is not None:
        o["toolUseResult"] = tur
    return o


def _handback(aid: str, body: str) -> str:
    report = "\n".join("  " + l for l in body.splitlines())
    return (f'Another Claude session sent a message:\n<agent-message from="{aid}">\n[Subagent hand-back] ... '
            f"The report follows:\n{report}\n</agent-message>\n<system-reminder>x</system-reminder>")


# ── ラベル・伏せ字 ──

@pytest.mark.parametrize("atype,desc,label,cls", [
    ("roi-plus-man-1", "RC-6 R1 1号", "1 号（リーダー）", "s1"),
    ("roi-plus-man-3", "", "3 号（運用）", "s3"),
    ("general-purpose", "RC-6 R1 2号 の代わり", "2 号（市場）・代理", "s2"),
    ("critic", "レビュー", "critic", "sx"),
])
def test_speaker_for_agent(atype, desc, label, cls):
    assert cl.speaker_for_agent(atype, desc) == (label, cls)


def test_mask_secrets_hides_whole_lines():
    text = "ふつうの行\nGCS_TOKEN=abc\n.env を読む\nAuthorization: Bearer x\nwebhook の URL\npassword=1\nAPI key は\n残る行"
    out = cl.mask_secrets(text).splitlines()
    assert out[0] == "ふつうの行" and out[-1] == "残る行"
    assert all(l == cl.MASKED for l in out[1:-1])


def test_stage_and_decision():
    assert cl.stage_of("RC-6 R1 1号") == "R1"
    assert cl.stage_of("", "## R3 の確定") == "R3"
    assert cl.stage_of("RC-10 の話") == ""
    assert cl.is_decision("s1", "RC-6 の決定", "x")
    assert not cl.is_decision("s2", "RC-6 の決定", "x")


# ── 本体の記録の解析 ──

def test_parse_main_flow():
    ag = cl.AgentDirectory()
    ev = []
    ev += cl.parse_main_line(_a("2026-11-25T01:00:00Z", {"type": "thinking", "thinking": "内緒"},
                                {"type": "text", "text": "R0 の資料を出します"}), ag)
    ev += cl.parse_main_line(_a("2026-11-25T01:00:01Z", {"type": "tool_use", "id": "tu1", "name": "Agent",
                                "input": {"subagent_type": "roi-plus-man-1", "description": "RC-6 R1 1号", "prompt": "長い"}}), ag)
    ev += cl.parse_main_line(_u("2026-11-25T01:00:02Z", [{"type": "tool_result", "tool_use_id": "tu1",
                                "content": "Async agent launched. agentId: a1234567890abcdef"}]), ag)
    ev += cl.parse_main_line(_a("2026-11-25T01:00:03Z", {"type": "tool_use", "id": "tu2", "name": "Bash",
                                "input": {"command": "cat .env", "description": "設定を見る"}}), ag)
    ev += cl.parse_main_line(_u("2026-11-25T01:00:04Z", [{"type": "tool_result", "tool_use_id": "tu2", "content": "SECRET=1\n生の出力"}]), ag)
    ev += cl.parse_main_line(_u("2026-11-25T01:00:05Z", _handback("a1234567890abcdef", "## 決定\n採る。\ntoken=xyz")), ag)
    ev += cl.parse_main_line(_u("2026-11-25T01:00:06Z", "<task-notification>\n<summary>Agent \"x\" finished</summary>\n</task-notification>"), ag)
    ev += cl.parse_main_line(_u("2026-11-25T01:00:07Z", "<system-reminder>長い規約</system-reminder>統括役から: 次へ"), ag)
    ev += cl.parse_main_line({"type": "attachment", "timestamp": "x"}, ag)

    kinds = [(e["speaker"], e["kind"]) for e in ev]
    assert kinds == [
        ("担当 rules-council", "text"), ("担当 rules-council", "call"), ("担当 rules-council", "tool"),
        ("1 号（リーダー）", "text"), ("通知", "notice"), ("統括役・ユーザー", "text"),
    ]
    assert ev[0]["stage"] == "R0"
    assert "1 号（リーダー）" in ev[1]["text"] and "長い" not in ev[1]["text"]
    assert ev[2]["text"] == "Bash: 設定を見る"
    rep = ev[3]
    assert rep["decision"] and rep["stage"] == "R1" and rep["cls"] == "s1"
    assert rep["text"].startswith("## 決定\n採る。") and "xyz" not in rep["text"] and cl.MASKED in rep["text"]
    assert all("生の出力" not in e["text"] and "内緒" not in e["text"] for e in ev)
    assert ev[5]["text"] == "統括役から: 次へ"


def test_parse_sub_line_uses_meta_label_and_skips_handback(tmp_path: Path):
    ag = cl.AgentDirectory()
    meta = tmp_path / "agent-abc1234567.meta.json"
    meta.write_text(json.dumps({"agentType": "general-purpose", "description": "RC-6 R2 3号（代わり）"}), encoding="utf-8")
    ag.load_meta(meta)
    ev = cl.parse_sub_line(_a("t", {"type": "text", "text": "見ます"}, {"type": "tool_use", "name": "Read", "input": {"file_path": "/x.md"}},
                              {"type": "tool_use", "name": "SubagentHandback", "input": {"message": "報告"}}), "abc1234567", ag)
    assert [(e["speaker"], e["kind"], e["stage"]) for e in ev] == [("3 号（運用）・代理", "text", "R2"), ("3 号（運用）・代理", "tool", "R2")]
    assert ev[1]["text"] == "Read: /x.md"
    assert cl.parse_sub_line(_u("t", [{"type": "tool_result", "content": "生"}]), "abc1234567", ag) == []


# ── 差分の読み ──

def test_tail_reads_only_appended_complete_lines(tmp_path: Path):
    f = tmp_path / "s.jsonl"
    f.write_bytes(b'{"a":1}\n{"a":2}\n{"a":')
    t = cl.TailFile(f, initial_tail=10_000)
    assert t.read_new_lines() == [b'{"a":1}', b'{"a":2}']
    assert t.read_new_lines() == []
    with f.open("ab") as h:
        h.write(b'3}\n{"a":4}\n')
    assert t.read_new_lines() == [b'{"a":3}', b'{"a":4}']
    f.write_bytes(b'{"b":1}\n')  # 切り詰め
    assert t.read_new_lines() == [b'{"b":1}']


def test_tail_initial_skips_partial_first_line(tmp_path: Path):
    f = tmp_path / "s.jsonl"
    f.write_bytes(b'{"x":"' + b"z" * 50 + b'"}\n{"a":1}\n')
    t = cl.TailFile(f, initial_tail=15)
    assert t.read_new_lines() == [b'{"a":1}']


def _session(tmp_path: Path):
    proj = tmp_path / "projects" / "p"
    sid = "11111111-2222-3333-4444-555555555555"
    sub = proj / sid / "subagents"
    sub.mkdir(parents=True)
    main = proj / f"{sid}.jsonl"
    lines = [
        _a("2026-11-25T01:00:00Z", {"type": "tool_use", "id": "tu1", "name": "Agent",
           "input": {"subagent_type": "roi-plus-man-2", "description": "RC-7 R1 2号"}}),
        _u("2026-11-25T01:00:01Z", [{"type": "tool_result", "tool_use_id": "tu1", "content": "ok"}], {"agentId": "a2222222222"}),
    ]
    main.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in lines), encoding="utf-8")
    subf = sub / "agent-a2222222222.jsonl"
    subf.write_text(json.dumps(_a("2026-11-25T01:00:02Z", {"type": "text", "text": "控除率から見る"}), ensure_ascii=False) + "\n", encoding="utf-8")
    tmpb = tmp_path / "tmp"
    tasks = tmpb / "-proj" / sid / "tasks"
    tasks.mkdir(parents=True)
    (tasks / "a2222222222.output").symlink_to(subf)  # 同じ実体は 1 本にまとめる
    (tasks / "bash123.output").write_text("生の出力\n", encoding="utf-8")
    return proj, tmpb, main, subf


def test_feed_dedupes_symlinks_and_streams_deltas(tmp_path: Path):
    proj, tmpb, main, subf = _session(tmp_path)
    feed = cl.LiveFeed(proj, tmpb)
    assert feed.poll_once() == 2
    assert len(feed.tails) == 2
    evs = feed.since(0)
    assert [(e["speaker"], e["kind"]) for e in evs] == [("担当 rules-council", "call"), ("2 号（市場）", "text")]
    assert feed.poll_once() == 0
    with subf.open("a", encoding="utf-8") as h:
        h.write(json.dumps(_a("2026-11-25T01:00:09Z", {"type": "text", "text": "R2 で反対"}), ensure_ascii=False) + "\n")
    assert feed.poll_once() == 1
    new = feed.since(evs[-1]["seq"])
    assert len(new) == 1 and new[0]["text"] == "R2 で反対"


def test_feed_follows_new_session_file(tmp_path: Path):
    proj, tmpb, main, _ = _session(tmp_path)
    feed = cl.LiveFeed(proj, tmpb)
    feed.poll_once()
    new_main = proj / "99999999-0000-0000-0000-000000000000.jsonl"
    new_main.write_text(json.dumps(_a("2026-11-25T02:00:00Z", {"type": "text", "text": "立て直し後"}), ensure_ascii=False) + "\n", encoding="utf-8")
    assert feed.poll_once() == 1
    assert feed.since(0)[-1]["text"] == "立て直し後"


def test_default_project_dir_name(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    d = cl.default_project_dir(Path("/home/jovyan/work/keiba-vpn"))
    assert d == tmp_path / "projects" / "-home-jovyan-work-keiba-vpn--claude-worktrees-rules-council"


# ── SSE ──

def test_sse_stream_sends_backlog_and_new_events(tmp_path: Path):
    pytest.importorskip("markdown")
    from src.scripts.docs import council_viewer as cv

    proj, tmpb, main, subf = _session(tmp_path)
    feed = cl.LiveFeed(proj, tmpb, poll_sec=0.1)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), cv.make_handler(tmp_path, feed))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
        c.request("GET", "/live")
        r = c.getresponse()
        body = r.read().decode("utf-8")
        assert r.status == 200 and "EventSource('/live/stream')" in body
        assert "connect-src 'self'" in r.getheader("Content-Security-Policy")
        assert "http://" not in body and "https://" not in body

        c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
        c.request("GET", "/live/stream")
        r = c.getresponse()
        assert r.getheader("Content-Type").startswith("text/event-stream")
        got = []
        while len(got) < 2:
            line = r.fp.readline().decode("utf-8")
            if line.startswith("data: "):
                got.append(json.loads(line[6:]))
        with subf.open("a", encoding="utf-8") as h:
            h.write(json.dumps(_a("2026-11-25T01:00:09Z", {"type": "text", "text": "追記"}), ensure_ascii=False) + "\n")
        while len(got) < 3:
            line = r.fp.readline().decode("utf-8")
            if line.startswith("data: "):
                got.append(json.loads(line[6:]))
        assert got[-1]["text"] == "追記" and got[-1]["seq"] == 3
        c.close()
    finally:
        srv.shutdown()
        srv.server_close()


def test_mask_private_key_block_and_url_credentials():
    text = "前\n-----BEGIN PRIVATE KEY-----\nMIIEvQIBADAN\nQWERTY\n-----END PRIVATE KEY-----\n後\npostgres://u:p@h/db\nAIzaSyA1234567890abc\n"
    out = cl.mask_secrets(text).splitlines()
    assert out[0] == "前" and out[5] == "後"
    assert all(l == cl.MASKED for l in out[1:5] + out[6:8])


def test_raw_command_output_and_meta_are_not_streamed():
    ag = cl.AgentDirectory()
    ev = cl.parse_main_line(_u("t", "<bash-input>cat x</bash-input><bash-stdout>VALUE=1</bash-stdout>"), ag)
    assert len(ev) == 1 and ev[0]["kind"] == "notice" and "VALUE" not in ev[0]["text"]
    meta = _u("t", "skill の本文")
    meta["isMeta"] = True
    assert cl.parse_main_line(meta, ag) == []


def test_bad_line_does_not_drop_the_batch(tmp_path: Path):
    proj, tmpb, main, subf = _session(tmp_path)
    with main.open("a", encoding="utf-8") as h:
        h.write(json.dumps({"type": "assistant", "timestamp": "z", "message": {"content": ["文字だけ", 3]}}) + "\n")
        h.write(json.dumps(_a("2026-11-25T01:00:05Z", {"type": "text", "text": "残る"}), ensure_ascii=False) + "\n")
    feed = cl.LiveFeed(proj, tmpb)
    feed.poll_once()
    assert "残る" in [e["text"] for e in feed.since(0)]


def test_stale_last_event_id_after_restart(tmp_path: Path):
    proj, tmpb, _, _ = _session(tmp_path)
    feed = cl.LiveFeed(proj, tmpb)
    feed.poll_once()
    assert len(feed.since(2500)) == 2  # 前の起動の番号でも、直近の分から流し直す
    assert feed.wait(2500, 0.01) == []  # 待ちで回り続けない


def test_tail_detects_replaced_file(tmp_path: Path):
    f = tmp_path / "s.jsonl"
    f.write_bytes(b'{"a":1}\n')
    t = cl.TailFile(f, initial_tail=10_000)
    assert t.read_new_lines() == [b'{"a":1}']
    g = tmp_path / "s2.jsonl"
    g.write_bytes(b'{"b":1}\n{"b":2}\n')
    g.replace(f)
    assert t.read_new_lines() == [b'{"b":1}', b'{"b":2}']
