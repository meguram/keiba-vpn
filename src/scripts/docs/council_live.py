"""会議（担当 rules-council）のセッションの記録（JSONL）を tail して、話し手のラベル付きの出来事に直す。

council_viewer の /live・/live/stream（Server-Sent Events）が使う。手元のファイルを読むだけ（GCS・外の網は読まない。費用 0 円）。

- 元: ``<claude の設定>/projects/<rules-council の worktree の名前>/<session>.jsonl``（新しい順。立て直されると別のファイル）と、
  そのセッションのサブエージェントの記録 ``<session>/subagents/agent-*.jsonl``。
  ``/tmp/claude-<uid>/*/<session>/tasks/a*.output`` は同じファイルへのシンボリックリンクなので、実体のパスで 1 本にまとめる。
- 話し手: 担当 rules-council の地の文、各号（roi-plus-man-1〜3、号を名乗って代わりに呼んだ general-purpose）、統括役、通知。
- ツールの呼び出しは 1 行にたたみ、ツールの生の出力（tool_result）は出さない。秘密が混ざりうる行は伏せる。
- 読むのは差分だけ（ファイルごとに位置を覚える）。読むのは 1 本のスレッドだけで、接続が増えても読む量は増えない。
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

POLL_SEC = 2.5
MAX_EVENTS = 3000
TEXT_LIMIT = 6000
LINE_LIMIT = 160
INITIAL_TAIL_MAIN = 2 * 1024 * 1024  # 初回に読む末尾の量（12MB 級の記録を全部は読まない）
INITIAL_TAIL_SUB = 512 * 1024
MAIN_WINDOW_SEC = 24 * 3600  # いちばん新しい記録から 24 時間以内に更新された記録も追う（立て直しの前後）
MAX_MAIN_FILES = 3

SECRET_RE = re.compile(
    r"(?i)(\.env\b|token|api[_\- ]?key|\bkeys?\b|secret|password|passwd|webhook|bearer|credential|private[_ ]key|"
    r"MONITOR_PASSWORD|sk-[A-Za-z0-9]{8,}|gh[pousr]_[A-Za-z0-9]{8,}|AKIA[0-9A-Z]{8,}|AIza[0-9A-Za-z_\-]{10,}|"
    r"xox[abpors]-|eyJ[A-Za-z0-9_\-]{10,}\.|://[^/\s:@]+:[^/\s@]+@|-----BEGIN|-----END)"
)
BEGIN_RE = re.compile(r"-----BEGIN [A-Z ]+-----")
END_RE = re.compile(r"-----END [A-Z ]+-----")
RAW_OUTPUT_TAGS = ("<bash-stdout", "<bash-stderr", "<bash-input", "<local-command-stdout", "<local-command-stderr",
                   "<command-name", "<command-message", "<command-args")
MASKED = "（秘密を含みうる行を伏せた）"
STAGE_RE = re.compile(r"(?<![A-Za-z0-9])R([0-3])(?![0-9])")
GOU_RE = re.compile(r"([123])\s*号")
REMINDER_RE = re.compile(r"<system-reminder[^>]*>.*?</system-reminder[^>]*>", re.S)
AGENT_MSG_RE = re.compile(r'<agent-message from="([^"]+)">(.*?)</agent-message>', re.S)
XSESSION_RE = re.compile(r"<cross-session-message[^>]*>(.*?)</cross-session-message>", re.S)
NOTIF_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.S)
AGENT_ID_RE = re.compile(r"agentId:\s*(a[0-9a-f]{6,})")
TASK_OUTPUT_RE = re.compile(r"^a[0-9a-f]{6,}\.output$")

GOU_LABEL = {"1": "1 号（リーダー）", "2": "2 号（市場）", "3": "3 号（運用）"}


# ── 伏せ字・切り詰め ──

def mask_secrets(text: str) -> str:
    out, in_block = [], False
    for line in text.splitlines():
        if BEGIN_RE.search(line):
            in_block = True
        if in_block or SECRET_RE.search(line):
            out.append(MASKED)
        else:
            out.append(line)
        if in_block and END_RE.search(line):
            in_block = False
    return "\n".join(out)


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…（続きは省略）"


def one_line(text: str, limit: int = LINE_LIMIT) -> str:
    return clip(" ".join(str(text).split()), limit)


# ── 話し手のラベル ──

def speaker_for_agent(agent_type: str, description: str) -> tuple[str, str]:
    """(ラベル, 色の組)。roi-plus-man-N はその号、号を名乗る代理（general-purpose など）も同じ号に寄せる。"""
    m = re.fullmatch(r"roi-plus-man-([123])", agent_type or "")
    if m:
        return GOU_LABEL[m.group(1)], f"s{m.group(1)}"
    g = GOU_RE.search(description or "")
    if g:
        return GOU_LABEL[g.group(1)] + "・代理", f"s{g.group(1)}"
    return agent_type or "サブエージェント", "sx"


def stage_of(*texts: str) -> str:
    for t in texts:
        m = STAGE_RE.search((t or "")[:400])
        if m:
            return f"R{m.group(1)}"
    return ""


def is_decision(cls: str, description: str, text: str) -> bool:
    return cls == "s1" and ("決定" in (description or "") or "裁定" in text[:600] or "## 決定" in text or "### 決定" in text)


def _tool_line(block: dict) -> str:
    name = block.get("name", "?")
    inp = block.get("input") if isinstance(block.get("input"), dict) else {}
    what = inp.get("description") or inp.get("file_path") or inp.get("pattern") or inp.get("query") or ""
    return one_line(f"{name}: {what}" if what else name)


def _agent_report(body: str) -> str:
    """SubagentHandback の枠（説明の行・字下げ）を外して、報告の本文だけにする。"""
    i = body.find("The report follows:")
    if i >= 0:
        body = body[i + len("The report follows:"):]
    lines = body.splitlines()
    return "\n".join(l[2:] if l.startswith("  ") else l for l in lines).strip()


class AgentDirectory:
    """agentId → (種類, 説明)。meta.json と、本体の記録の Agent の呼び出し・結果から集める。"""

    def __init__(self) -> None:
        self.by_id: dict[str, tuple[str, str]] = {}
        self._pending: dict[str, tuple[str, str]] = {}  # tool_use_id → (種類, 説明)

    def note_call(self, tool_use_id: str, agent_type: str, description: str) -> None:
        self._pending[tool_use_id] = (agent_type, description)

    def note_result(self, tool_use_id: str, agent_id: str) -> None:
        info = self._pending.pop(tool_use_id, None)
        if info and agent_id:
            self.by_id.setdefault(agent_id, info)

    def load_meta(self, meta: Path) -> None:
        aid = meta.name.removeprefix("agent-").removesuffix(".meta.json")
        if aid in self.by_id:
            return
        try:
            d = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        self.by_id[aid] = (str(d.get("agentType", "")), str(d.get("description", "")))

    def get(self, agent_id: str) -> tuple[str, str]:
        return self.by_id.get(agent_id, ("", ""))


# ── 1 行の JSONL → 出来事 ──

def _event(ts: str, speaker: str, cls: str, kind: str, text: str, stage: str = "", decision: bool = False) -> dict:
    limit = TEXT_LIMIT if kind == "text" else LINE_LIMIT
    text = mask_secrets(text)
    return {"ts": ts or "", "speaker": speaker, "cls": cls, "kind": kind,
            "text": clip(text, limit) if kind == "text" else one_line(text, limit),
            "stage": stage, "decision": decision}


def parse_main_line(obj: dict, agents: AgentDirectory) -> list[dict]:
    """担当 rules-council の本体の記録の 1 行。"""
    t = obj.get("type")
    ts = obj.get("timestamp", "")
    msg = obj.get("message")
    if t not in ("assistant", "user") or not isinstance(msg, dict):
        return []
    content = msg.get("content")
    out: list[dict] = []
    if t == "assistant" and isinstance(content, list):
        for b in content:
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text" and b.get("text", "").strip():
                txt = b["text"].strip()
                out.append(_event(ts, "担当 rules-council", "sc", "text", txt, stage_of(txt)))
            elif bt == "tool_use" and b.get("name") in ("Agent", "Task"):
                inp = b.get("input") or {}
                atype, desc = str(inp.get("subagent_type", "")), str(inp.get("description", ""))
                agents.note_call(b.get("id", ""), atype, desc)
                label, _ = speaker_for_agent(atype, desc)
                out.append(_event(ts, "担当 rules-council", "sc", "call", f"呼び出し → {label}: {desc}", stage_of(desc)))
            elif bt == "tool_use":
                out.append(_event(ts, "担当 rules-council", "sc", "tool", _tool_line(b)))
        return out
    if t == "user" and isinstance(content, list):
        for b in content:  # ツールの結果は出さない。Agent の結果からは agentId だけ拾う
            if b.get("type") == "tool_result":
                tur = obj.get("toolUseResult")
                aid = tur.get("agentId") if isinstance(tur, dict) else None
                if not aid:
                    c = b.get("content")
                    s = c if isinstance(c, str) else " ".join(x.get("text", "") for x in c or [] if isinstance(x, dict))
                    m = AGENT_ID_RE.search(s or "")
                    aid = m.group(1) if m else None
                if aid:
                    agents.note_result(b.get("tool_use_id", ""), aid)
        return out
    if t == "user" and isinstance(content, str):
        raw = content
        if obj.get("isMeta"):
            return []  # skill の本文・規約などの差し込み
        if any(tag in raw for tag in RAW_OUTPUT_TAGS):
            return [_event(ts, "通知", "sn", "notice", "（手で打ったコマンドとその出力。中身は出さない）")]
        for m in AGENT_MSG_RE.finditer(raw):  # 号などの最終の報告
            atype, desc = agents.get(m.group(1))
            label, cls = speaker_for_agent(atype, desc)
            rep = _agent_report(m.group(2))
            out.append(_event(ts, label, cls, "text", rep, stage_of(desc, rep), is_decision(cls, desc, rep)))
        if out:
            return out
        if "<task-notification" in raw:
            s = NOTIF_SUMMARY_RE.search(raw)
            return [_event(ts, "通知", "sn", "notice", s.group(1) if s else "通知")]
        x = XSESSION_RE.search(raw)
        body = (x.group(1) if x else REMINDER_RE.sub("", raw)).strip()
        if not body or body.startswith("This session is being continued"):
            return [_event(ts, "通知", "sn", "notice", "会話の要約から再開")] if body else []
        return [_event(ts, "統括役・ユーザー", "so", "text", body, stage_of(body))]
    return []


def parse_sub_line(obj: dict, agent_id: str, agents: AgentDirectory) -> list[dict]:
    """サブエージェントの記録の 1 行。地の文とツールの 1 行だけ（最終の報告は本体の記録の側で出す）。"""
    if obj.get("type") != "assistant" or not isinstance(obj.get("message"), dict):
        return []
    content = obj["message"].get("content")
    if not isinstance(content, list):
        return []
    atype, desc = agents.get(agent_id)
    label, cls = speaker_for_agent(atype, desc)
    ts = obj.get("timestamp", "")
    out = []
    for b in content:
        if b.get("type") == "text" and b.get("text", "").strip():
            txt = b["text"].strip()
            out.append(_event(ts, label, cls, "text", txt, stage_of(desc, txt), is_decision(cls, desc, txt)))
        elif b.get("type") == "tool_use" and b.get("name") != "SubagentHandback":
            out.append(_event(ts, label, cls, "tool", _tool_line(b), stage_of(desc)))
    return out


# ── 差分の読み ──

class TailFile:
    """ファイルの大きさと位置を覚えて、足された完全な行だけを返す。"""

    def __init__(self, path: Path, initial_tail: int) -> None:
        self.path = path
        self.off: int | None = None
        self.ino: int | None = None
        self.buf = b""
        self.initial_tail = initial_tail

    def read_new_lines(self) -> list[bytes]:
        try:
            st = self.path.stat()
        except OSError:
            return []
        size = st.st_size
        if self.ino is not None and st.st_ino != self.ino:  # 差し替えられた
            self.off, self.buf = 0, b""
        self.ino = st.st_ino
        skip_partial = False
        if self.off is None:
            self.off = max(0, size - self.initial_tail)
            skip_partial = self.off > 0
        if size < self.off:  # 切り詰められた
            self.off, self.buf = 0, b""
        if size == self.off:
            return []
        try:
            with self.path.open("rb") as f:
                f.seek(self.off)
                data = f.read(size - self.off)
        except OSError:
            return []
        self.off += len(data)
        data = self.buf + data
        if skip_partial:  # 途中から読み始めたときは、最初の半端な行を捨てる
            nl = data.find(b"\n")
            data = data[nl + 1:] if nl >= 0 else b""
        *lines, self.buf = data.split(b"\n")
        return [l for l in lines if l.strip()]


def default_project_dir(repo_root: Path) -> Path:
    cfg = os.environ.get("CLAUDE_CONFIG_DIR")
    base = Path(cfg) if cfg else (Path.home() / ".claude-jovyan" if (Path.home() / ".claude-jovyan").is_dir() else Path.home() / ".claude")
    name = re.sub(r"[^A-Za-z0-9]", "-", str(repo_root / ".claude" / "worktrees" / "rules-council"))
    return base / "projects" / name


def default_tmp_base() -> Path:
    return Path(tempfile.gettempdir()) / f"claude-{os.getuid()}"


class LiveFeed:
    def __init__(self, project_dir: Path, tmp_base: Path | None = None, poll_sec: float = POLL_SEC) -> None:
        self.project_dir = project_dir
        self.tmp_base = tmp_base
        self.poll_sec = poll_sec
        self.agents = AgentDirectory()
        self.tails: dict[str, tuple[TailFile, str, str]] = {}  # 実体のパス → (tail, "main"|"sub", agentId)
        self.events: deque[dict] = deque(maxlen=MAX_EVENTS)
        self.seq = 0
        self.cond = threading.Condition()
        self.clients = 0
        self.last_client = 0.0
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # 見つける
    def discover(self) -> None:
        stamped = []
        for p in self.project_dir.glob("*.jsonl"):
            try:
                stamped.append((p.stat().st_mtime, p))
            except OSError:
                continue  # 周回の間に消えた
        if not stamped:
            return
        stamped.sort(reverse=True)
        newest = stamped[0][0]
        for mtime, p in stamped[:MAX_MAIN_FILES]:
            if mtime < newest - MAIN_WINDOW_SEC:
                continue
            self._add(p, "main", "", INITIAL_TAIL_MAIN)
            sid = p.stem
            sub_dir = self.project_dir / sid / "subagents"
            for meta in sub_dir.glob("agent-*.meta.json"):
                self.agents.load_meta(meta)
            for f in sub_dir.glob("agent-*.jsonl"):
                self._add(f, "sub", f.stem.removeprefix("agent-"), INITIAL_TAIL_SUB)
            if self.tmp_base is not None:
                for f in self.tmp_base.glob(f"*/{sid}/tasks/*.output"):
                    if TASK_OUTPUT_RE.match(f.name):
                        self._add(f, "sub", f.name.removesuffix(".output"), INITIAL_TAIL_SUB)

    def _add(self, path: Path, kind: str, agent_id: str, initial_tail: int) -> None:
        try:
            real = str(path.resolve(strict=True))
        except OSError:
            return
        if real not in self.tails:
            self.tails[real] = (TailFile(Path(real), initial_tail), kind, agent_id)

    # 読む
    def poll_once(self) -> int:
        with self._lock:
            self.discover()
            batch: list[dict] = []
            # 本体を先に読む（Agent の呼び出しから号の名前を引くため）
            for real, (tail, kind, aid) in sorted(self.tails.items(), key=lambda kv: kv[1][1] != "main"):
                for raw in tail.read_new_lines():
                    try:
                        obj = json.loads(raw)
                    except ValueError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    try:
                        batch.extend(parse_main_line(obj, self.agents) if kind == "main" else parse_sub_line(obj, aid, self.agents))
                    except Exception:  # noqa: BLE001  形の違う 1 行で、同じ周回のほかの行を捨てない
                        continue
            if not batch:
                return 0
            batch.sort(key=lambda e: e["ts"])
            with self.cond:
                for e in batch:
                    self.seq += 1
                    e["seq"] = self.seq
                    self.events.append(e)
                self.cond.notify_all()
            return len(batch)

    def since(self, seq: int, initial: int = 400) -> list[dict]:
        with self.cond:
            if seq <= 0 or seq > self.seq:  # 起動し直して番号が戻ったときは、直近の分から流し直す
                return list(self.events)[-initial:]
            return [e for e in self.events if e["seq"] > seq]

    # 1 本のスレッドで回す。見ている人がいない間（60 秒）は読まない
    def ensure_started(self) -> None:
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="council-live", daemon=True)
                self._thread.start()

    def _run(self) -> None:
        while True:
            if self.clients > 0 or time.time() - self.last_client < 60:
                try:
                    self.poll_once()
                except Exception:  # noqa: BLE001  1 回の失敗で止めない
                    pass
            time.sleep(self.poll_sec)

    def attach(self) -> None:
        with self.cond:
            self.clients += 1
            self.last_client = time.time()
        self.ensure_started()

    def detach(self) -> None:
        with self.cond:
            self.clients = max(0, self.clients - 1)
            self.last_client = time.time()

    def wait(self, seq: int, timeout: float) -> list[dict]:
        with self.cond:
            if self.seq <= seq:
                self.cond.wait(timeout)
            if self.seq <= seq:
                return []
        return self.since(seq)
