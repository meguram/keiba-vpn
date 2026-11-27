"""会議のライブ（src/scripts/docs/council_live.py）: JSONL の解析・ラベル付け・伏せ字・差分の読み・会議のセッションの探し方と、/live/api。"""

from __future__ import annotations

import http.client
import json
import os
import threading
import time
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
    (sub / "agent-a2222222222.meta.json").write_text(json.dumps({"agentType": "roi-plus-man-2", "description": "RC-7 R1 2号"}),
                                                     encoding="utf-8")
    subf = sub / "agent-a2222222222.jsonl"
    subf.write_text(json.dumps(_a("2026-11-25T01:00:02Z", {"type": "text", "text": "控除率から見る"}), ensure_ascii=False) + "\n", encoding="utf-8")
    tmpb = tmp_path / "tmp"
    tasks = tmpb / "-proj" / sid / "tasks"
    tasks.mkdir(parents=True)
    (tasks / "a2222222222.output").symlink_to(subf)  # 同じ実体は 1 本にまとめる
    (tasks / "bash123.output").write_text("生の出力\n", encoding="utf-8")
    return proj, tmpb, main, subf


def _hub(proj: Path, tmpb: Path | None = None, hours: float = 24.0) -> cl.CouncilHub:
    return cl.CouncilHub(lambda: [proj], tmpb, hours)


def test_feed_dedupes_symlinks_and_reads_deltas(tmp_path: Path):
    proj, tmpb, main, subf = _session(tmp_path)
    feed = cl.SessionFeed(main, tmpb)
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


def test_since_limits_to_last_n_talks_and_keeps_tools_between(tmp_path: Path):
    f = tmp_path / "s.jsonl"
    lines = []
    for i in range(10):
        lines.append(_a(f"2026-11-25T01:{i:02d}:00Z", {"type": "text", "text": f"発言 {i}"}))
        lines.append(_a(f"2026-11-25T01:{i:02d}:30Z", {"type": "tool_use", "name": "Bash", "input": {"description": f"道具 {i}"}}))
    f.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in lines), encoding="utf-8")
    feed = cl.SessionFeed(f, None, "担当 x")
    feed.poll_once()
    got = feed.since(0, 3)
    assert [e["text"] for e in got if e["kind"] == "text"] == ["発言 7", "発言 8", "発言 9"]
    assert [e["text"] for e in got if e["kind"] == "tool"] == ["Bash: 道具 7", "Bash: 道具 8", "Bash: 道具 9"]
    assert got[0]["speaker"] == "担当 x"
    assert len(feed.since(10_000, 3)) == len(got)  # 先の番号（起動し直し）は初回と同じ扱い


def test_project_prefix_and_dirs(tmp_path: Path):
    assert cl.project_prefix(Path("/home/jovyan/work/keiba-vpn")) == "-home-jovyan-work-keiba-vpn"
    assert cl.project_prefix(Path("/home/jovyan/work/keiba-vpn/.claude/worktrees/live-bubbles")) == "-home-jovyan-work-keiba-vpn"
    root = tmp_path / "projects"
    for name in ("-home-jovyan-work-keiba-vpn", "-home-jovyan-work-keiba-vpn--claude-worktrees-rules-council",
                 "-home-jovyan-work-keiba-vpn-history", "-home-jovyan-other"):
        (root / name).mkdir(parents=True)
    got = [d.name for d in cl.project_dirs(tmp_path, "-home-jovyan-work-keiba-vpn")]
    assert got == ["-home-jovyan-work-keiba-vpn", "-home-jovyan-work-keiba-vpn--claude-worktrees-rules-council"]


def test_claude_base_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    assert cl.claude_base() == tmp_path


def test_scan_name_takes_last_and_decodes_escapes():
    data = (b'{"type":"agent-name","agentName":"old","sessionId":"s"}\n'
            b'{"type":"custom-title","customTitle":"\\u8840\\u7d71-lab","sessionId":"s"}\n'
            b'{"type": "agent-name", "agentName": "rules-council"}\n')
    assert cl.scan_name(data) == "rules-council"
    assert cl.scan_name(data[:-40]) == "血統-lab"
    assert cl.scan_name(b'{"type":"user","message":"agentName"}') == ""


def _write(path: Path, objs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in objs), encoding="utf-8")


def test_hub_lists_only_recent_council_sessions(tmp_path: Path):
    proj = tmp_path / "p"
    talk = _a("2026-11-26T01:00:00Z", {"type": "text", "text": "こんにちは"})
    # 名前で会議（号をまだ呼んでいない）
    _write(proj / "aaaaaaaa-0000-0000-0000-000000000001.jsonl", [{"type": "agent-name", "agentName": "rules-council"}, talk])
    # 顔ぶれで会議（名前は別）
    _write(proj / "aaaaaaaa-0000-0000-0000-000000000002.jsonl", [{"type": "agent-name", "agentName": "train-serve-skew"}, talk])
    _write(proj / "aaaaaaaa-0000-0000-0000-000000000002" / "subagents" / "agent-a1.meta.json", [{"agentType": "pedigree-1"}])
    # 会議でない（統括役など）
    _write(proj / "aaaaaaaa-0000-0000-0000-000000000003.jsonl", [{"type": "agent-name", "agentName": "claude-orchestrator"}, talk])
    _write(proj / "aaaaaaaa-0000-0000-0000-000000000003" / "subagents" / "agent-a2.meta.json", [{"agentType": "critic"}])
    # 古い会議
    old = proj / "aaaaaaaa-0000-0000-0000-000000000004.jsonl"
    _write(old, [{"type": "agent-name", "agentName": "pedigree-lab"}, talk])
    t = time.time() - 3 * 86400
    os.utime(old, (t, t))
    hub = _hub(proj)
    rows = hub.sessions()
    assert {r["name"] for r in rows} == {"rules-council", "train-serve-skew"}
    assert set(rows[0]) == {"id", "name", "mtime", "members"}  # cwd・最後の指示などは渡さない
    assert next(r for r in rows if r["name"] == "train-serve-skew")["members"] == ["pedigree-1"]
    assert hub.feed("aaaaaaaa-0000-0000-0000-000000000003") is None  # 会議でないセッションは読まない
    assert hub.feed("aaaaaaaa-0000-0000-0000-000000000004") is None
    f = hub.feed("aaaaaaaa-0000-0000-0000-000000000001")
    assert f is not None and f.since(0)[-1]["speaker"] == "担当 rules-council"


def test_hub_follows_new_session_file(tmp_path: Path):
    proj, tmpb, main, _ = _session(tmp_path)
    hub = _hub(proj, tmpb)
    assert [r["id"] for r in hub.sessions()] == [main.stem]
    new_main = proj / "99999999-0000-0000-0000-000000000000.jsonl"
    _write(new_main, [{"type": "agent-name", "agentName": "rules-council"},
                      _a("2026-11-25T02:00:00Z", {"type": "text", "text": "立て直し後"})])
    hub.discover(force=True)
    rows = hub.sessions()
    assert rows[0]["id"] == new_main.stem  # 新しい順
    assert hub.feed(new_main.stem).since(0)[-1]["text"] == "立て直し後"


def test_hub_name_rescans_only_appended_bytes(tmp_path: Path):
    f = tmp_path / "p" / "aaaaaaaa-0000-0000-0000-000000000009.jsonl"
    _write(f, [_a("t", {"type": "text", "text": "x"})])
    info = cl.SessionInfo(f)
    st = f.stat()
    info.refresh(st.st_mtime, st.st_size)
    assert info.name == "" and not info.council
    with f.open("a", encoding="utf-8") as h:
        h.write(json.dumps({"type": "agent-name", "agentName": "pedigree-lab"}) + "\n")
    st = f.stat()
    info.refresh(st.st_mtime, st.st_size)
    assert info.name == "pedigree-lab" and info.council and info.scanned == st.st_size


# ── /live/api（数秒おきに取りに来る） ──

def test_live_api_sessions_and_events(tmp_path: Path):
    pytest.importorskip("markdown")
    from src.scripts.docs import council_viewer as cv

    proj, tmpb, main, subf = _session(tmp_path)
    hub = _hub(proj, tmpb)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), cv.make_handler(tmp_path, hub))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]

    def get(path: str):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("GET", path)
        r = c.getresponse()
        body = r.read().decode("utf-8")
        c.close()
        return r, body

    try:
        r, body = get("/live")
        assert r.status == 200 and "/live/api/events" in body and "EventSource" not in body
        assert "connect-src 'self'" in r.getheader("Content-Security-Policy")
        assert "http://" not in body and "https://" not in body  # 外部の CDN は使わない

        r, body = get("/live/api/sessions")
        d = json.loads(body)
        assert r.status == 200 and r.getheader("Content-Type").startswith("application/json")
        assert [s["id"] for s in d["sessions"]] == [main.stem]

        r, body = get(f"/live/api/events?session={main.stem}&since=0&limit=30")
        d = json.loads(body)
        assert [e["kind"] for e in d["events"]] == ["call", "text"]
        assert d["events"][1]["html"] == "<p>控除率から見る</p>" and "text" not in d["events"][1]
        assert d["events"][0]["text"].startswith("呼び出し → 2 号")
        epoch, seq = d["epoch"], d["seq"]

        with subf.open("a", encoding="utf-8") as h:
            h.write(json.dumps(_a("2026-11-25T01:00:09Z", {"type": "text", "text": "**追記**"}), ensure_ascii=False) + "\n")
        time.sleep(cl.POLL_MIN_SEC + 0.1)
        r, body = get(f"/live/api/events?session={main.stem}&since={seq}&limit=30")
        d = json.loads(body)
        assert d["epoch"] == epoch and [e["html"] for e in d["events"]] == ["<p><strong>追記</strong></p>"]
        assert d["events"][0]["seq"] == seq + 1

        assert get("/live/api/events?session=../../etc&since=0")[0].status == 404
        assert get("/live/api/events?session=ffffffff-0000-0000-0000-000000000000")[0].status == 404
        assert get("/live/stream")[0].status == 404  # SSE はやめた
        assert get(f"/live/api/events?session={main.stem}&since=%C2%B2&limit=%C2%B2")[0].status == 200  # ² で 500 にしない
        r, body = get("/live")
        csp = r.getheader("Content-Security-Policy")
        assert "'unsafe-inline'" not in csp.split("script-src", 1)[1].split(";")[0]
        nonce = csp.split("'nonce-", 1)[1].split("'", 1)[0]
        assert f'<script nonce="{nonce}">' in body and get("/live")[1] != body  # 応答ごとに替わる
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
    feed = cl.SessionFeed(main, tmpb)
    feed.poll_once()
    assert "残る" in [e["text"] for e in feed.since(0)]


def test_tail_detects_replaced_file(tmp_path: Path):
    f = tmp_path / "s.jsonl"
    f.write_bytes(b'{"a":1}\n')
    t = cl.TailFile(f, initial_tail=10_000)
    assert t.read_new_lines() == [b'{"a":1}']
    g = tmp_path / "s2.jsonl"
    g.write_bytes(b'{"b":1}\n{"b":2}\n')
    g.replace(f)
    assert t.read_new_lines() == [b'{"b":1}', b'{"b":2}']


# ── 11/26 の形（号の報告・統括役の連絡が isMeta で来る、続きの依頼、議題の番号） ──

def test_stage_ignores_agenda_number_and_topic():
    assert cl.stage_of("R0-17 の R1（3 号）") == "R1"
    assert cl.stage_of("R0-17 複勝の期待値") == ""
    assert cl.topic_of("R0-17 の R1（3 号）") == "R0-17"
    assert cl.topic_of("", "RC-6 の決定") == "RC-6"
    assert cl.is_decision("s1", "R0-17 の R3", "x", "R3")
    assert not cl.is_decision("s2", "R0-17 の R3", "x", "R3")


def test_meta_agent_message_and_cross_session_are_shown():
    ag = cl.AgentDirectory()
    cl.parse_main_line(_a("t0", {"type": "tool_use", "id": "tu1", "name": "Agent",
                                 "input": {"subagent_type": "roi-plus-man-1", "description": "R0-17 の R1（1 号）"}}), ag, "担当 rules-council")
    cl.parse_main_line(_u("t1", [{"type": "tool_result", "tool_use_id": "tu1", "content": "agentId: a1234567890abcdef"}]), ag)
    ev = cl.parse_main_line(_a("t2", {"type": "tool_use", "id": "tu2", "name": "SendMessage",
                                      "input": {"to": "a1234567890abcdef", "summary": "R0-17 の R3（まとめ）", "message": "長い本文"}}), ag)
    assert ev[0]["kind"] == "call" and "続きを頼む → 1 号（リーダー）: R0-17 の R3" in ev[0]["text"] and "長い本文" not in ev[0]["text"]
    rep = _u("t3", _handback("a1234567890abcdef", "採る。"))
    rep["isMeta"] = True
    ev = cl.parse_main_line(rep, ag)
    assert len(ev) == 1 and ev[0]["speaker"] == "1 号（リーダー）" and ev[0]["stage"] == "R3" and ev[0]["decision"]
    assert ev[0]["topic"] == "R0-17"
    x = _u("t4", 'Another Claude session sent a message:\n<cross-session-message from="uds:/tmp/x.sock" '
                 'from-name="claude-orchestrator-handover" from-mode="prompting">\n次へ進めてよい\n</cross-session-message>')
    x["isMeta"] = True
    ev = cl.parse_main_line(x, ag)
    assert [(e["speaker"], e["text"]) for e in ev] == [("統括役", "次へ進めてよい")]
    skill = _u("t5", "skill の本文")
    skill["isMeta"] = True
    assert cl.parse_main_line(skill, ag) == []


def test_pedigree_speakers_and_notice_unescape():
    assert cl.speaker_for_agent("pedigree-1", "") == ("血統 1 号", "p1")
    assert cl.speaker_for_agent("pedigree-2", "") == ("血統 2 号", "p2")
    ev = cl.parse_main_line(_u("t", "<task-notification>\n<summary>Background command \"a &gt; b\" completed</summary>\n</task-notification>"),
                            cl.AgentDirectory())
    assert ev[0]["text"] == 'Background command "a > b" completed'


def test_meta_with_tag_text_but_no_match_is_dropped():
    o = _u("t", "規約: <agent-message の書き方 と <cross-session-message の書き方")
    o["isMeta"] = True
    assert cl.parse_main_line(o, cl.AgentDirectory()) == []


def test_names_are_masked():
    assert cl.safe_label("deploy-token-abc") == cl.MASKED
    x = _u("t", '<cross-session-message from="uds:/x" from-name="my-secret-runner">やあ</cross-session-message>')
    assert cl.parse_main_line(x, cl.AgentDirectory())[0]["speaker"] == cl.MASKED
    assert cl.speaker_for_agent("api-key-helper", "") == (cl.MASKED, "sx")
    assert cl.council_name("bet-optimizer-1124") and not cl.council_name("bet-optimizerx")


def test_name_in_middle_of_large_file_and_split_line(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cl, "NAME_SCAN_HEAD", 100)
    monkeypatch.setattr(cl, "NAME_SCAN_TAIL", 200)
    f = tmp_path / "aaaaaaaa-0000-0000-0000-00000000000a.jsonl"
    pad = json.dumps({"type": "api-request", "x": "z" * 300}) + "\n"
    f.write_text(pad * 3 + json.dumps({"type": "agent-name", "agentName": "pedigree-lab"}) + "\n" + pad * 3, encoding="utf-8")
    info = cl.SessionInfo(f)
    st = f.stat()
    info.refresh(st.st_mtime, st.st_size)
    assert info.name == "pedigree-lab"
    with f.open("a", encoding="utf-8") as h:  # 名前の行の書きかけ
        h.write('{"type":"agent-name","agentName":"rules-')
    st = f.stat()
    info.refresh(st.st_mtime, st.st_size)
    assert info.name == "pedigree-lab" and info.scanned < st.st_size
    with f.open("a", encoding="utf-8") as h:
        h.write('council"}\n')
    st = f.stat()
    info.refresh(st.st_mtime, st.st_size)
    assert info.name == "rules-council" and info.scanned == st.st_size


def test_feed_rebuilt_after_eviction_changes_epoch(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cl, "MAX_FEEDS", 1)
    proj = tmp_path / "p"
    ids = [f"aaaaaaaa-0000-0000-0000-00000000000{i}" for i in (1, 2)]
    for sid in ids:
        _write(proj / f"{sid}.jsonl", [{"type": "agent-name", "agentName": "rules-council"},
                                       _a("2026-11-26T01:00:00Z", {"type": "text", "text": "x"})])
    hub = _hub(proj)
    first = hub.feed(ids[0])
    time.sleep(0.01)
    hub.feed(ids[1])  # 1 本目を捨てる
    again = hub.feed(ids[0])
    assert again is not first and again.created != first.created


# ── 伏せ字（メール・IP） ──

def test_mask_email_and_ip_inline():
    out = cl.mask_secrets("連絡は someone@example.com へ。番地は 10.1.2.3、手元は 127.0.0.1。版は 1.2.3")
    assert "someone@example.com" not in out and "10.1.2.3" not in out
    assert "［メールを伏せた］" in out and "［IP を伏せた］" in out
    assert "127.0.0.1" in out and "版は 1.2.3" in out  # 行ごとは伏せない
    assert cl.safe_label("ops@example.org の担当") == "［メールを伏せた］ の担当"


# ── 動いているセッションの一覧（claude agents --json） ──

REPO = "/home/u/work/repo"
ORCH = "dddddddd-0000-0000-0000-000000000001"
WORK = "eeeeeeee-0000-0000-0000-000000000002"
NONAME = "ffffffff-0000-0000-0000-000000000003"


def _agents_json():
    return [
        {"id": "eeeeeeee", "sessionId": WORK, "cwd": f"{REPO}/.claude/worktrees/track-bias", "name": "track-bias",
         "state": "working", "status": "idle", "startedAt": 1_000_000},
        {"id": "dddddddd", "sessionId": ORCH, "cwd": REPO, "name": "claude-orchestrator-handover",
         "state": "working", "status": "busy", "startedAt": 2_000_000},
        {"id": "ffffffff", "sessionId": NONAME, "cwd": REPO, "state": "blocked", "startedAt": 3_000_000},
        {"id": "99999999", "sessionId": "99999999-0000-0000-0000-000000000009", "cwd": "/home/u/other", "name": "x",
         "state": "working"},  # このリポジトリの外は出さない
        {"id": "zz", "sessionId": "not-a-uuid", "cwd": REPO},
    ]


def test_agent_row_keeps_repo_only_and_hides_cwd():
    rows = [r for r in (cl.agent_row(x, REPO) for x in _agents_json()) if r]
    assert [r["sid"] for r in rows] == [WORK, ORCH, NONAME]
    assert rows[0]["place"] == "track-bias" and rows[1]["place"] == "本体"
    assert rows[2]["name"] == "" and rows[2]["state"] == "blocked" and rows[2]["status"] == ""
    assert all("cwd" not in r for r in rows) and rows[0]["started"] == 1000.0


def test_agent_list_calls_at_most_every_30_sec_and_keeps_last_on_failure():
    now = [100.0]
    calls = []

    def fn():
        calls.append(now[0])
        if len(calls) == 3:
            raise RuntimeError("down")
        return _agents_json()

    al = cl.AgentList(fn, REPO, clock=lambda: now[0])
    assert cl.AGENTS_SEC == 30.0 and al.interval == 30.0
    rows, ok = al.get()
    assert ok and len(rows) == 3 and len(calls) == 1
    now[0] += 29.9
    al.get()
    al.get()
    assert len(calls) == 1  # 30 秒の内は呼ばない
    now[0] += 0.2
    al.get()
    assert len(calls) == 2
    now[0] += 30
    rows, ok = al.get()  # 失敗: 前の一覧を残す
    assert not ok and len(rows) == 3 and len(calls) == 3
    al.get()
    assert len(calls) == 3  # 失敗も 30 秒は覚える


def test_run_state_four_states():
    now = 10_000.0
    assert cl.run_state("done", "busy", now, now) == "done"
    assert cl.run_state("blocked", "busy", now, now) == "blocked"
    assert cl.run_state("working", "busy", 0, now) == "running"  # 長いコマンドの最中
    assert cl.run_state("working", "idle", now - 10, now) == "waiting"  # idle と明かしていればそれを使う
    assert cl.run_state("working", "", now - 179, now) == "running"  # status が無いときは直近 3 分の動き
    assert cl.run_state("working", "", now - 181, now) == "waiting"
    assert cl.run_state("working", "", 0, now) == "waiting"


def _agents_hub(tmp_path: Path, fn=_agents_json):
    base = tmp_path / "projects"
    main_dir, wt_dir = base / "-repo", base / "-repo--claude-worktrees-track-bias"
    talk = _a("2026-11-27T01:00:00Z", {"type": "text", "text": "作業中"})
    _write(main_dir / f"{ORCH}.jsonl", [{"type": "agent-name", "agentName": "claude-orchestrator-handover"}, talk])
    _write(main_dir / f"{NONAME}.jsonl", [talk])
    old = wt_dir / f"{WORK}.jsonl"  # worktree のフォルダ・24 時間より古い記録
    _write(old, [{"type": "agent-name", "agentName": "track-bias"}, talk])
    t = time.time() - 3 * 86400
    os.utime(old, (t, t))
    # 載っていない会議（終わった）と、載っていない普通のセッション
    _write(main_dir / "aaaaaaaa-0000-0000-0000-00000000000a.jsonl", [{"type": "agent-name", "agentName": "rules-council"}, talk])
    _write(main_dir / "bbbbbbbb-0000-0000-0000-00000000000b.jsonl", [{"type": "agent-name", "agentName": "secret-work"}, talk])
    al = cl.AgentList(fn, REPO)
    return cl.CouncilHub(lambda: [main_dir, wt_dir], None, 24.0, al, None), al


def test_hub_tabs_orchestrator_first_and_past_council_last(tmp_path: Path):
    hub, al = _agents_hub(tmp_path)
    tabs, ok = hub.tabs()
    assert ok
    assert [t["id"] for t in tabs] == [ORCH, WORK, NONAME, "aaaaaaaa-0000-0000-0000-00000000000a"]
    by = {t["id"]: t for t in tabs}
    assert by[NONAME]["name"] == "ffffffff" and by[NONAME]["state"] == "blocked"  # 名前の無いセッションは短い id
    assert by[WORK]["found"] and by[WORK]["state"] == "waiting" and by[WORK]["place"] == "track-bias"
    assert by[ORCH]["state"] == "running"
    assert by["aaaaaaaa-0000-0000-0000-00000000000a"]["state"] == "past"
    assert set(tabs[0]) == {"id", "short", "name", "state", "agent_state", "place", "started", "mtime", "members", "found"}
    hub.tabs()
    assert al.calls == 1  # 画面の問い合わせが重なっても 30 秒に 1 回


def test_hub_feed_only_listed_or_council(tmp_path: Path):
    hub, _ = _agents_hub(tmp_path)
    hub.tabs()
    f = hub.feed(WORK)  # worktree のフォルダの古い記録も、載っていれば読む
    assert f is not None and f.since(0)[-1]["speaker"] == "担当 track-bias"
    assert hub.feed("aaaaaaaa-0000-0000-0000-00000000000a") is not None  # 会議は今までどおり
    assert hub.feed("bbbbbbbb-0000-0000-0000-00000000000b") is None  # 載っていない・会議でない → 出さない


def test_hub_tabs_fall_back_to_council_when_agents_fail(tmp_path: Path):
    def boom():
        raise FileNotFoundError("claude")

    hub, _ = _agents_hub(tmp_path, boom)
    tabs, ok = hub.tabs()
    assert not ok and [t["id"] for t in tabs] == ["aaaaaaaa-0000-0000-0000-00000000000a"]
    assert hub.feed(ORCH) is None


# ── ゴール（最初の利用者の文） ──

def test_first_user_text_skips_meta_reminders_and_summaries(tmp_path: Path):
    f = tmp_path / "s.jsonl"
    _write(f, [
        {"type": "custom-title", "customTitle": "x"},
        {"type": "file-history-snapshot", "snapshot": {"user": "x"}},
        {"type": "user", "isMeta": True, "message": {"content": "skill の本文"}},
        _u("t1", "<command-name>/model</command-name>"),
        _u("t2", "This session is being continued from a previous conversation."),
        _u("t3", [{"type": "tool_result", "tool_use_id": "x", "content": "結果"}]),
        _u("t4", "<system-reminder>規約</system-reminder>\nあなたは担当「x」です。連絡 a@b.example.com"),
        _u("t5", "2 回目の指示"),
    ])
    goal = cl.first_user_text(f)
    assert goal.startswith("あなたは担当「x」です。") and "a@b.example.com" not in goal and "規約" not in goal


# ── 待っている理由（記録の末尾から作る規則） ──

def _own(kind, text, **h):
    return {"ts": "t", "speaker": "担当", "cls": "sc", "kind": kind, "text": text, **h}


def _evs(*objs):
    agents = cl.AgentDirectory()
    out = []
    for o in objs:
        out.extend(cl.parse_main_line(o, agents, "担当"))
    return out, agents


def test_wait_reason_approval_screen_after_exit_plan_mode():
    evs = [_own("text", "計画です"), _own("tool", "ExitPlanMode", tool="ExitPlanMode")]
    assert cl.wait_reasons(evs, "blocked", "1ac8c069") == ["ユーザーの承認の画面で止まっている（claude attach 1ac8c069）"]
    evs = [_own("text", "続けます"), _own("tool", "Bash: x", tool="Bash")]
    assert cl.wait_reasons(evs, "blocked", "bad id; rm")[0] == "承認の画面などで止まっている可能性"  # 形の違う id は出さない


def test_wait_reason_background_bash_from_record():
    evs, agents = _evs(
        _a("t1", {"type": "tool_use", "id": "b1", "name": "Bash",
                  "input": {"command": "pytest -q", "description": "テストを流す", "run_in_background": True}}),
        _u("t2", [{"type": "tool_result", "tool_use_id": "b1", "content": "running"}], {"backgroundTaskId": "bxyz1234"}),
        _a("t3", {"type": "text", "text": "終わるまで待ちます。"}),
    )
    assert evs[0]["bg"] and agents.bg_tasks == {"bxyz1234": ("Bash: テストを流す", "t2")}
    assert cl.wait_reasons(evs, "waiting") == ["背景の処理の終わりを待っている: Bash: テストを流す"]
    done = evs + cl.parse_main_line(_u("t4", "<task-notification><task-id>bxyz1234</task-id><summary>done</summary></task-notification>"),
                                    agents, "担当")
    assert done[-1]["task"] == "bxyz1234" and evs[0]["task"] == "bxyz1234"
    other = evs + cl.parse_main_line(_u("t4", "<task-notification><task-id>bother99</task-id><summary>x</summary></task-notification>"),
                                     agents, "担当")
    assert cl.wait_reasons(other, "waiting")[0].startswith("背景の処理の終わりを待っている")  # 別の番号の通知では解けない
    assert cl.wait_reasons(done, "waiting") == [cl.NO_REASON]  # その番号の通知の後は待っていない


def test_background_line_is_masked():
    evs, agents = _evs(
        _a("t1", {"type": "tool_use", "id": "b1", "name": "Bash",
                  "input": {"command": "x", "description": "ops@example.com に送る token を確かめる", "run_in_background": True}}),
        _u("t2", [{"type": "tool_result", "tool_use_id": "b1", "content": "r"}], {"backgroundTaskId": "bmask123"}),
    )
    line = agents.bg_tasks["bmask123"][0]
    assert "example.com" not in line and "token" not in line and line == cl.MASKED


def test_wait_reason_monitor_and_sleep_loop():
    evs, _ = _evs(_a("t", {"type": "tool_use", "id": "m", "name": "Monitor", "input": {"description": "CI を待つ"}}))
    assert cl.wait_reasons(evs, "waiting") == ["背景の処理の終わりを待っている: Monitor: CI を待つ"]
    evs, _ = _evs(_a("t", {"type": "tool_use", "id": "s", "name": "Bash",
                           "input": {"command": "until [ -f x ]; do sleep 5; done", "description": "印を待つ", "run_in_background": True}}))
    assert evs[0]["wait"] and "Bash: 印を待つ" in cl.wait_reasons(evs, "waiting")[0]
    assert "until" not in evs[0]["text"]  # 命令の中身は出さない
    fg, _ = _evs(_a("t", {"type": "tool_use", "id": "f", "name": "Bash",
                          "input": {"command": "ls | while read l; do echo $l; done", "description": "一覧"}}))
    assert cl.wait_reasons(fg, "waiting") == [cl.NO_REASON]  # 前景の while は待ちにしない


def test_wait_reason_live_process_wins():
    bg = [{"task": "bq", "line": "Bash: 学習", "ts": "t", "pids": 2, "minutes": 12}]
    assert cl.wait_reasons([_own("text", "流しました")], "waiting", "", bg) == ["背景の処理の終わりを待っている: Bash: 学習（12 分）"]


def test_wait_reason_send_message_to_orchestrator():
    evs, _ = _evs(_a("t", {"type": "text", "text": "報告します"},
                     {"type": "tool_use", "id": "s", "name": "SendMessage", "input": {"to": "claude-orchestrator-handover", "summary": "完了"}}))
    assert evs[-1]["text"] == "SendMessage: → claude-orchestrator-handover 完了"
    assert cl.wait_reasons(evs, "waiting") == ["統括役の返事を待っている"]
    reply = evs + [{"ts": "t2", "speaker": "統括役", "cls": "so", "kind": "text", "text": "了解"}]
    assert cl.wait_reasons(reply, "waiting") == [cl.NO_REASON]
    sub, _ = _evs(_a("t", {"type": "tool_use", "id": "s", "name": "SendMessage", "input": {"to": "a1234567", "summary": "続き"}}))
    assert cl.wait_reasons(sub, "waiting") == ["a1234567 の返事を待っている"]
    sub_reply = sub + [{"ts": "t2", "speaker": "critic", "cls": "sx", "kind": "text", "text": "直した"}]
    assert cl.wait_reasons(sub_reply, "waiting") == [cl.NO_REASON]


def test_wait_reason_asks_and_time():
    evs = [_own("text", "直しました。\n承認をお願いします（返事が無ければ推奨の案で進めます）。12/1 以降に入れます")]
    r = cl.wait_reasons(evs, "waiting")
    assert r[0] == "統括役・ユーザーの返事を待っている: 承認をお願いします（返事が無ければ推奨の案で進めます）。"
    assert r[1] == "時刻を待っている: 12/1 以降"
    assert cl.wait_reasons([_own("text", "17:00 以降に流す")], "waiting") == ["時刻を待っている: 17:00 以降"]
    assert cl.wait_reasons([_own("text", "直しました。")], "waiting") == [cl.NO_REASON]


def test_wait_reason_subagent_call():
    evs, _ = _evs(_a("t", {"type": "tool_use", "id": "c", "name": "Agent", "input": {"subagent_type": "critic", "description": "差分を見る"}}))
    assert cl.wait_reasons(evs, "waiting") == ["サブエージェントの報告を待っている: 呼び出し → critic: 差分を見る"]


# ── 背景の処理（/proc の fd/1 と記録の番号の照合） ──

def test_proc_scanner_groups_by_task_and_reads_no_cmdline(tmp_path: Path):
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "stat").write_text("cpu 1\nbtime 1000\n")
    for pid, task, start in ((11, "bq1234", 500), (12, "bq1234", 300), (13, None, 100)):
        d = proc / str(pid)
        (d / "fd").mkdir(parents=True)
        target = f"/tmp/claude-1/-p/s/tasks/{task}.output" if task else "/dev/null"
        (d / "fd" / "1").symlink_to(target)
        (d / "stat").write_text(f"{pid} (py thon) S " + " ".join(["0"] * 18) + f" {start} 0 0\n")
    (proc / "self").mkdir()
    ps = cl.ProcScanner(proc, clock=lambda: 0.0)
    hz = os.sysconf("SC_CLK_TCK")
    found = ps.get()
    assert set(found) == {"bq1234"} and found["bq1234"]["pids"] == 2
    assert found["bq1234"]["started"] == pytest.approx(1000 + 300 / hz)
    bg = cl.background_tasks({"bq1234": ("Bash: 学習", "t"), "gone": ("Bash: 終わった", "t")}, found, 1000 + 300 / hz + 600)
    assert bg == [{"task": "bq1234", "line": "Bash: 学習", "ts": "t", "pids": 2, "minutes": 10}]


# ── /live/api のタブと帯 ──

def test_live_api_tabs_and_summary(tmp_path: Path):
    pytest.importorskip("markdown")
    from src.scripts.docs import council_viewer as cv

    hub, _ = _agents_hub(tmp_path)
    blocked = tmp_path / "projects" / "-repo" / f"{NONAME}.jsonl"
    _write(blocked, [_u("t0", "ゴールは **画面** を作ること"),
                     _a("t1", {"type": "text", "text": "計画です"}, {"type": "tool_use", "id": "p", "name": "ExitPlanMode", "input": {}})])
    srv = ThreadingHTTPServer(("127.0.0.1", 0), cv.make_handler(tmp_path, hub))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]

    def get(path: str):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("GET", path)
        r = c.getresponse()
        body = r.read().decode("utf-8")
        c.close()
        return r, body

    try:
        d = json.loads(get("/live/api/sessions")[1])
        assert d["agents_ok"] and d["sessions"][0]["id"] == ORCH
        d = json.loads(get(f"/live/api/events?session={NONAME}&since=0&limit=20")[1])
        s = d["summary"]
        assert s["state"] == "blocked" and s["reasons"] == ["ユーザーの承認の画面で止まっている（claude attach ffffffff）"]
        assert s["goal_head"] == "ゴールは **画面** を作ること" and s["goal_html"] == "<p>ゴールは <strong>画面</strong> を作ること</p>"
        assert s["text"]["head"] == "計画です" and s["tool"]["text"] == "ExitPlanMode"
        assert "tool" not in d["events"][-1] or "bg" not in d["events"][-1]  # 判定の手がかりは渡さない
        assert all(set(e) <= set(cv.EVENT_FIELDS) | {"html", "text"} for e in d["events"])
        assert get("/live/api/events?session=bbbbbbbb-0000-0000-0000-00000000000b&since=0")[0].status == 404
        r, body = get("/live")
        assert 'id="tabs"' in body and 'id="sum"' in body and "claude attach" in body
        assert "http://" not in body and "https://" not in body
    finally:
        srv.shutdown()
        srv.server_close()


# ── 研究の大枠（config/council_viewer/themes.json） ──

def test_parse_themes_masks_and_drops_bad_rows():
    rows = cl.parse_themes({"themes": [
        {"id": "pedigree", "name": "血統", "goal": "連絡 a@b.example.com\n次の行", "sessions": ["aptitude-group", "", 3], "milestone": "12/4 A の門"},
        {"id": "../x", "name": "名前だけ"},
        {"goal": "名前が無い"}, "文字列",
    ]})
    assert [r["id"] for r in rows] == ["pedigree", "t1"]
    assert rows[0]["goal"] == "連絡 ［メールを伏せた］ 次の行" and rows[0]["sessions"] == ["aptitude-group"]
    assert rows[0]["milestone"] == "12/4 A の門" and rows[1]["sessions"] == [] and rows[1]["milestone"] == ""
    assert cl.parse_themes([]) == [] and cl.parse_themes({"themes": "x"}) == []


def test_repo_themes_file_is_valid():
    path = Path(__file__).resolve().parents[2] / cl.THEMES_REL
    rows, err = cl.ThemeFile(path).get()
    assert err == "" and len(rows) == 3
    assert rows[2]["sessions"] == ["pair-bets", "rules-council"]


def test_theme_file_rereads_when_changed(tmp_path: Path):
    f = tmp_path / "themes.json"
    tf = cl.ThemeFile(f)
    assert tf.get() == ([], "themes.json がありません")
    f.write_text(json.dumps({"themes": [{"id": "a", "name": "A", "sessions": ["x"]}]}), encoding="utf-8")
    assert [r["name"] for r in tf.get()[0]] == ["A"]
    f.write_text("{壊れた", encoding="utf-8")
    os.utime(f, (time.time() + 5, time.time() + 5))
    rows, err = tf.get()
    assert rows == [] and err.startswith("themes.json を読めません")


def test_theme_cards_states_and_none(tmp_path: Path):
    def agents():
        rows = _agents_json()
        rows.append({"id": "cccccccc", "sessionId": "cccccccc-0000-0000-0000-00000000000c", "cwd": REPO, "name": "finished",
                     "state": "done", "startedAt": 1})
        return rows

    hub, _ = _agents_hub(tmp_path, agents)
    tabs, _ = hub.tabs()
    themes = cl.parse_themes({"themes": [{"id": "x", "name": "X", "sessions": ["track-bias", "finished", "missing"]}]})
    card = cl.theme_cards(hub, themes, tabs)[0]
    by = {s["name"]: s for s in card["sessions"]}
    assert by["track-bias"]["state"] == "waiting" and by["track-bias"]["id"] == WORK and by["track-bias"]["head"] == "作業中"
    assert by["track-bias"]["reason"] == cl.NO_REASON
    assert by["finished"]["state"] == "none" and by["finished"]["note"] == "終わった"
    assert by["missing"] == {"name": "missing", "id": "", "state": "none", "note": "動いているセッションが無い"}


def test_live_api_includes_themes(tmp_path: Path):
    pytest.importorskip("markdown")
    from src.scripts.docs import council_viewer as cv

    hub, _ = _agents_hub(tmp_path)
    root = tmp_path / "repo"
    (root / cl.THEMES_REL).parent.mkdir(parents=True)
    (root / cl.THEMES_REL).write_text(json.dumps({"themes": [{"id": "t", "name": "T", "goal": "g", "sessions": ["track-bias"]}]}),
                                      encoding="utf-8")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), cv.make_handler(root, hub))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
        c.request("GET", "/live/api/sessions")
        d = json.loads(c.getresponse().read().decode("utf-8"))
        assert d["themes_error"] == "" and d["themes"][0]["sessions"][0]["id"] == WORK
        c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
        c.request("GET", "/live")
        body = c.getresponse().read().decode("utf-8")
        assert body.index('id="themes"') < body.index('class="bar"')  # 一番上の段
    finally:
        srv.shutdown()
        srv.server_close()


def test_theme_cards_do_not_open_feeds_and_pick_newest(tmp_path: Path):
    def agents():
        rows = _agents_json()
        old = dict(rows[0], sessionId="eeeeeeee-0000-0000-0000-0000000000e0", id="eeeeee00", state="done", startedAt=1)
        return [old] + rows  # 同じ名前の終わった古いセッションが先に並ぶ

    hub, _ = _agents_hub(tmp_path, agents)
    tabs, _ = hub.tabs()
    themes = cl.parse_themes({"themes": [{"id": "x", "name": "X", "sessions": ["track-bias"]}]})
    card = cl.theme_cards(hub, themes, tabs)[0]
    assert card["sessions"][0]["id"] == WORK and card["sessions"][0]["state"] == "waiting"
    assert hub.feeds == {}  # 開いているタブの feed を追い出さない
    assert cl.parse_themes({"themes": [{"name": "A", "sessions": "abc"}]})[0]["sessions"] == []


def test_theme_file_prefers_local_copy(tmp_path: Path):
    repo_f, local_f = tmp_path / "config.json", tmp_path / "local.json"
    repo_f.write_text(json.dumps({"themes": [{"name": "版"}]}), encoding="utf-8")
    tf = cl.ThemeFile(local_f, repo_f)
    assert tf.get()[0][0]["name"] == "版"
    local_f.write_text(json.dumps({"themes": [{"name": "手元"}]}), encoding="utf-8")
    assert tf.get()[0][0]["name"] == "手元"


def test_card_cache_rereads_only_on_change(tmp_path: Path):
    f = tmp_path / "s.jsonl"
    _write(f, [_a("t1", {"type": "text", "text": "一"})])
    cc = cl.CardCache()
    first = cc.events(f)
    assert [e["text"] for e in first] == ["一"] and cc.events(f) is first
    with f.open("a", encoding="utf-8") as h:
        h.write(json.dumps(_a("t2", {"type": "text", "text": "二"}), ensure_ascii=False) + "\n")
    assert [e["text"] for e in cc.events(f)] == ["一", "二"]
