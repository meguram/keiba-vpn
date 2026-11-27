"""会議（rules-council・pedigree-lab など）のセッションの記録（JSONL）を差分で読み、話し手のラベル付きの出来事に直す。

council_viewer の /live と /live/api/*（画面が数秒おきに取りに来る）が使う。手元のファイルを読むだけ（GCS・外の網は読まない。費用 0 円）。

- タブ: ``claude agents --json``（30 秒に 1 回まで）に載っている、このリポジトリのセッション（統括役が先頭）と、
  最近（既定 24 時間）の会議のセッション（会議の顔ぶれ roi-plus-man-1〜3・pedigree-1〜2 を呼んだか、
  名前が rules-council・pedigree-lab・bet-optimizer のもの）。中身を出すのはこのどちらかだけ。
- 記録の場所: ``<claude の設定>/projects/`` の下の、このリポジトリ（本体と .claude/worktrees/*）のフォルダの ``<sessionId>.jsonl``。
- 帯: 状態（実行中・待機中・止まっている・終わった）・待っている理由（記録の末尾から作る規則。wait_reasons）・
  ゴール（最初の利用者の文）・今の作業・背景の処理（``/proc/<pid>/fd/1`` と記録の背景の番号の照合）。
- 読む: 本体の記録と、そのサブエージェントの記録 ``<session>/subagents/agent-*.jsonl``。
  ``/tmp/claude-<uid>/*/<session>/tasks/a*.output`` は同じファイルへのシンボリックリンクなので、実体のパスで 1 本にまとめる。
- 話し手: 担当の地の文、各号（roi-plus-man-1〜3、号を名乗って代わりに呼んだ general-purpose）、血統の 1・2 号、統括役、通知。
- ツールの呼び出しは 1 行にたたみ、ツールの生の出力（tool_result）は出さない。秘密が混ざりうる行は伏せる。
- 読むのは差分だけ（ファイルごとに位置を覚える）。画面の問い合わせが重なっても、同じセッションを読み直すのは 1.5 秒に 1 回まで。
"""
from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

POLL_MIN_SEC = 1.5  # 同じセッションを読み直す最短の間隔（画面の問い合わせが重なっても読む量を増やさない）
DISCOVER_SEC = 10.0  # セッションを探し直す間隔
MAX_EVENTS = 3000
TEXT_LIMIT = 6000
LINE_LIMIT = 160
# 初回に読む末尾の量。記録の大半は api-request などの会話でない行で、末尾 2MB では発言が十数件しか無かった（11/26 実測）
INITIAL_TAIL_MAIN = 12 * 1024 * 1024
INITIAL_TAIL_SUB = 2 * 1024 * 1024
WINDOW_HOURS = 24.0  # この時間内に更新された記録だけを探す（KEIBA_COUNCIL_LIVE_HOURS で替える）
MAX_FEEDS = 8  # 同時に覚えておくセッションの数（古く見たものから捨てる）
NAME_SCAN_HEAD = 1024 * 1024
NAME_SCAN_TAIL = 4 * 1024 * 1024
SKIP_PREFIXES = (b'{"type":"api-request', b'{"type":"data-flow', b'{"type":"file-history')  # 会話でない大きな行は解析しない

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
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+")
IPV4_RE = re.compile(r"(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9]|\.[0-9])")
IP_OK = frozenset({"127.0.0.1", "0.0.0.0"})  # 手元の番地は伏せない
STAGE_RE = re.compile(r"(?<![A-Za-z0-9])R([0-3])(?![0-9]|-[0-9])")  # R0-17 は議題の番号なので段にしない
TOPIC_RE = re.compile(r"(?<![A-Za-z0-9])(R0-[0-9]+|RC-[0-9]+)")
GOU_RE = re.compile(r"([123])\s*号")
REMINDER_RE = re.compile(r"<system-reminder[^>]*>.*?</system-reminder[^>]*>", re.S)
AGENT_MSG_RE = re.compile(r'<agent-message from="([^"]+)">(.*?)</agent-message>', re.S)
XSESSION_RE = re.compile(r"<cross-session-message([^>]*)>(.*?)</cross-session-message>", re.S)
FROM_NAME_RE = re.compile(r'from-name="([^"]{1,80})"')
NOTIF_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.S)
AGENT_ID_RE = re.compile(r"agentId:\s*(a[0-9a-f]{6,})")
TASK_OUTPUT_RE = re.compile(r"^a[0-9a-f]{6,}\.output$")
NOTIF_TASK_RE = re.compile(r"<task-id>([A-Za-z0-9]{4,40})</task-id>")
TASK_ID_RE = re.compile(r"[A-Za-z0-9]{4,40}")

GOU_LABEL = {"1": "1 号（リーダー）", "2": "2 号（市場）", "3": "3 号（運用）"}
PEDIGREE_LABEL = {"1": "血統 1 号", "2": "血統 2 号"}
# 会議の顔ぶれ（この種類のサブエージェントを呼んだセッションを会議とみなす）
COUNCIL_AGENT_TYPES = frozenset({"roi-plus-man-1", "roi-plus-man-2", "roi-plus-man-3", "pedigree-1", "pedigree-2"})
# 名前でも会議とみなす担当（立て直した直後で、まだ号を呼んでいないときも出す）
COUNCIL_NAME_PREFIXES = ("rules-council", "pedigree-lab", "bet-optimizer")
OWNER_DEFAULT = "担当 rules-council"


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
    text = EMAIL_RE.sub("［メールを伏せた］", "\n".join(out))
    return IPV4_RE.sub(lambda m: m.group(0) if m.group(0) in IP_OK else "［IP を伏せた］", text)


def safe_label(text: str, limit: int = 80) -> str:
    """話し手・セッションの名前など、画面に出す短い名前（1 行にして、秘密を含みうるなら伏せる）。"""
    return mask_secrets(one_line(text, limit)) or "？"


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
    m = re.fullmatch(r"pedigree-([12])", agent_type or "")
    if m:
        return PEDIGREE_LABEL[m.group(1)], f"p{m.group(1)}"
    g = GOU_RE.search(description or "")
    if g:
        return GOU_LABEL[g.group(1)] + "・代理", f"s{g.group(1)}"
    return safe_label(agent_type, 40) if agent_type else "サブエージェント", "sx"


def stage_of(*texts: str) -> str:
    for t in texts:
        m = STAGE_RE.search((t or "")[:400])
        if m:
            return f"R{m.group(1)}"
    return ""


def topic_of(*texts: str) -> str:
    """議題の番号（R0-17・RC-6）。説明・本文の先頭から。"""
    for t in texts:
        m = TOPIC_RE.search((t or "")[:400])
        if m:
            return m.group(1)
    return ""


def is_decision(cls: str, description: str, text: str, stage: str = "") -> bool:
    """1 号の R3・決定・裁定を決定とみなす。"""
    return cls == "s1" and (stage == "R3" or "決定" in (description or "") or "裁定" in text[:600] or "## 決定" in text or "### 決定" in text)


def _tool_line(block: dict) -> str:
    name = block.get("name", "?")
    inp = block.get("input") if isinstance(block.get("input"), dict) else {}
    what = inp.get("description") or inp.get("file_path") or inp.get("pattern") or inp.get("query") or inp.get("summary") or ""
    if name == "SendMessage" and inp.get("to"):
        what = f"→ {inp['to']}" + (f" {what}" if what else "")
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
        self.latest: dict[str, str] = {}  # agentId → SendMessage で続きを頼んだときの要約（段・議題をこちらから取る）
        self.bg_pending: dict[str, dict] = {}  # tool_use_id → 背景で流した Bash の出来事（結果の番号を書き足す）
        self.bg_tasks: dict[str, tuple[str, str]] = {}  # 背景の処理の番号 → (Bash の 1 行, 時刻)

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

    def topic_text(self, agent_id: str) -> str:
        """段・議題を拾う文（続きを頼んだ要約があればそれ、無ければ最初の説明）。"""
        return self.latest.get(agent_id) or self.get(agent_id)[1]


# ── 1 行の JSONL → 出来事 ──

def _event(ts: str, speaker: str, cls: str, kind: str, text: str, stage: str = "", decision: bool = False,
           topic: str = "", **hints) -> dict:
    """``hints`` は状態の判定の手がかり（tool・bg・wait・to・task）。画面には渡さない（council_viewer.EVENT_FIELDS）。"""
    limit = TEXT_LIMIT if kind == "text" else LINE_LIMIT
    text = mask_secrets(text)
    e = {"ts": ts or "", "speaker": speaker, "cls": cls, "kind": kind,
         "text": clip(text, limit) if kind == "text" else one_line(text, limit),
         "stage": stage, "decision": decision, "topic": topic}
    e.update(hints)
    return e


WAIT_CMD_RE = re.compile(r"(^|[;&|(\s])(sleep\s|until\s|while\s)")


def _tool_hints(block: dict) -> dict:
    """ツールの呼び出しの手がかり。命令の中身は画面に出さず、待つ形かどうかだけ見る。"""
    inp = block.get("input") if isinstance(block.get("input"), dict) else {}
    h = {"tool": str(block.get("name", ""))}
    if inp.get("run_in_background") is True:
        h["bg"] = True
    cmd = inp.get("command")
    if isinstance(cmd, str) and WAIT_CMD_RE.search(cmd):
        h["wait"] = True
    if h["tool"] == "SendMessage":
        h["to"] = safe_label(str(inp.get("to", "")), 60)
    return h


def parse_main_line(obj: dict, agents: AgentDirectory, owner: str = OWNER_DEFAULT) -> list[dict]:
    """担当（rules-council・pedigree-lab など）の本体の記録の 1 行。``owner`` は担当の話し手の名前。"""
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
                out.append(_event(ts, owner, "sc", "text", txt, stage_of(txt), topic=topic_of(txt)))
            elif bt == "tool_use" and b.get("name") in ("Agent", "Task"):
                inp = b.get("input") or {}
                atype, desc = str(inp.get("subagent_type", "")), str(inp.get("description", ""))
                agents.note_call(b.get("id", ""), atype, desc)
                label, _ = speaker_for_agent(atype, desc)
                out.append(_event(ts, owner, "sc", "call", f"呼び出し → {label}: {desc}", stage_of(desc), topic=topic_of(desc)))
            elif bt == "tool_use" and b.get("name") == "SendMessage" and isinstance(b.get("input"), dict) \
                    and str(b["input"].get("to", "")) in agents.by_id:
                aid, summ = str(b["input"]["to"]), str(b["input"].get("summary", ""))
                agents.latest[aid] = summ
                label, _ = speaker_for_agent(*agents.get(aid))
                out.append(_event(ts, owner, "sc", "call", f"続きを頼む → {label}: {summ}", stage_of(summ), topic=topic_of(summ)))
            elif bt == "tool_use":
                ev = _event(ts, owner, "sc", "tool", _tool_line(b), **_tool_hints(b))  # 伏せ字は _event の中で通る
                if ev.get("bg"):
                    agents.bg_pending[b.get("id", "")] = ev
                out.append(ev)
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
                task = tur.get("backgroundTaskId") if isinstance(tur, dict) else None
                ev = agents.bg_pending.pop(b.get("tool_use_id", ""), None)
                if isinstance(task, str) and TASK_ID_RE.fullmatch(task) and ev is not None:
                    ev["task"] = task  # 通知（task-notification）の番号と突き合わせる
                    agents.bg_tasks[task] = (ev["text"], ts)
        return out
    if t == "user" and isinstance(content, str):
        raw = content
        meta = bool(obj.get("isMeta"))  # skill の本文・規約などの差し込み。号の報告・統括役の連絡も isMeta で来るので、形が合うものだけ通す
        if any(tag in raw for tag in RAW_OUTPUT_TAGS):
            return [_event(ts, "通知", "sn", "notice", "（手で打ったコマンドとその出力。中身は出さない）")]
        for m in AGENT_MSG_RE.finditer(raw):  # 号などの最終の報告
            atype, desc = agents.get(m.group(1))
            label, cls = speaker_for_agent(atype, desc)
            rep = _agent_report(m.group(2))
            about = agents.topic_text(m.group(1))
            stg = stage_of(about, rep)
            out.append(_event(ts, label, cls, "text", rep, stg, is_decision(cls, about, rep, stg), topic_of(about, desc, rep)))
        if out:
            return out
        if meta and not XSESSION_RE.search(raw):
            return []
        if "<task-notification" in raw:
            s = NOTIF_SUMMARY_RE.search(raw)
            tid = NOTIF_TASK_RE.search(raw)
            return [_event(ts, "通知", "sn", "notice", html.unescape(s.group(1)) if s else "通知",  # 通知の中は &gt; などで書かれる
                           task=tid.group(1) if tid else "")]
        x = XSESSION_RE.search(raw)
        body = (x.group(2) if x else REMINDER_RE.sub("", raw)).strip()
        if not body or body.startswith("This session is being continued"):
            return [_event(ts, "通知", "sn", "notice", "会話の要約から再開")] if body else []
        who = "統括役・ユーザー"
        if x:
            fn = FROM_NAME_RE.search(x.group(1))
            name = fn.group(1) if fn else ""
            who = "統括役" if (not name or "orchestrator" in name) else safe_label(name, 40)
        return [_event(ts, who, "so", "text", body, stage_of(body), topic=topic_of(body))]
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
    desc = agents.topic_text(agent_id)
    ts = obj.get("timestamp", "")
    out = []
    for b in content:
        if b.get("type") == "text" and b.get("text", "").strip():
            txt = b["text"].strip()
            stg = stage_of(desc, txt)
            out.append(_event(ts, label, cls, "text", txt, stg, is_decision(cls, desc, txt, stg), topic_of(desc, txt)))
        elif b.get("type") == "tool_use" and b.get("name") != "SubagentHandback":
            out.append(_event(ts, label, cls, "tool", _tool_line(b), stage_of(desc), topic=topic_of(desc)))
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


def claude_base() -> Path:
    cfg = os.environ.get("CLAUDE_CONFIG_DIR")
    if cfg:
        return Path(cfg)
    home = Path.home()
    return home / ".claude-jovyan" if (home / ".claude-jovyan").is_dir() else home / ".claude"


def project_prefix(repo_root: Path) -> str:
    """リポジトリの根 → claude の projects の下のフォルダ名（worktree の中から動かしても本体の名前にそろえる）。"""
    s = str(repo_root)
    if "/.claude/worktrees/" in s:
        s = s.split("/.claude/worktrees/", 1)[0]
    return re.sub(r"[^A-Za-z0-9]", "-", s)


def project_dirs(base: Path, prefix: str) -> list[Path]:
    """本体（cwd がリポジトリの根）と、その worktree（.claude/worktrees/*）の projects のフォルダ。"""
    root = base / "projects"
    try:
        kids = list(root.iterdir())
    except OSError:
        return []
    wt = prefix + "--claude-worktrees-"
    return sorted(d for d in kids if d.is_dir() and (d.name == prefix or d.name.startswith(wt)))


def default_tmp_base() -> Path:
    return Path(tempfile.gettempdir()) / f"claude-{os.getuid()}"


NAME_RE = re.compile(rb'"type":\s*"(?:agent-name|custom-title)".{0,200}?"(?:agentName|customTitle)":\s*"((?:[^"\\\n]|\\.){1,200})"')


def scan_name(data: bytes) -> str:
    """記録のバイト列から、いちばん後のセッションの名前（agent-name・custom-title）を拾う。"""
    found = ""
    for m in NAME_RE.finditer(data):
        try:
            found = str(json.loads(b'"' + m.group(1) + b'"'))  # \uXXXX の名前も戻す
        except ValueError:
            continue
    return one_line(found, 80)


def council_name(name: str) -> bool:
    return any(name == p or name.startswith(p + "-") for p in COUNCIL_NAME_PREFIXES)


class SessionInfo:
    """1 本の記録の名前と、呼んだサブエージェントの種類（会議かどうかの判定に使う）。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.sid = path.stem
        self.name = ""
        self.scanned = 0  # 名前を探し終えた位置
        self.agent_types: set[str] = set()
        self.mtime = 0.0

    def refresh(self, mtime: float, size: int) -> None:
        self.mtime = mtime
        if size > self.scanned:
            # 名前は後から付け直されるので、いちばん後の名前を取る。初回は尻 → 頭 → 全体の順に探す（全体は 1 回だけ）
            try:
                with self.path.open("rb") as f:
                    if self.scanned == 0 and size > NAME_SCAN_HEAD + NAME_SCAN_TAIL:
                        f.seek(size - NAME_SCAN_TAIL)
                        got = scan_name(f.read(NAME_SCAN_TAIL))
                        if not got:
                            f.seek(0)
                            while not got and f.tell() < size:  # 尻に無いときだけ、1 回だけ全体を読む
                                chunk = f.read(NAME_SCAN_TAIL)
                                got = scan_name(chunk) or got
                        data = b""
                        end = size
                    else:
                        f.seek(self.scanned)
                        data = f.read(size - self.scanned)
                        got = ""
                        end = size
            except OSError:
                return
            if data:
                nl = data.rfind(b"\n")  # 書きかけの行は次の回に読む（名前の行が 2 回にまたがって取りこぼさない）
                if nl < 0:
                    return
                data = data[:nl + 1]
                end = self.scanned + len(data)
                got = scan_name(data)
            if got:
                self.name = got
            self.scanned = end
        sub = self.path.parent / self.sid / "subagents"
        for meta in sub.glob("agent-*.meta.json"):
            try:
                t = str(json.loads(meta.read_text(encoding="utf-8")).get("agentType", ""))
            except (OSError, ValueError):
                continue
            if t:
                self.agent_types.add(t)

    @property
    def council(self) -> bool:
        return bool(self.agent_types & COUNCIL_AGENT_TYPES) or council_name(self.name)

    def public(self) -> dict:
        """画面に渡す分だけ（cwd・会員の識別子・最後の指示などは渡さない）。"""
        return {"id": self.sid, "name": safe_label(self.name) if self.name else self.sid[:8], "mtime": self.mtime,
                "members": sorted(self.agent_types & COUNCIL_AGENT_TYPES)}


class SessionFeed:
    """1 本のセッション（本体の記録とそのサブエージェントの記録）の出来事。読むのは差分だけ。"""

    def __init__(self, main: Path, tmp_base: Path | None = None, owner: str = OWNER_DEFAULT) -> None:
        self.main = main
        self.sid = main.stem
        self.tmp_base = tmp_base
        self.owner = owner
        self.agents = AgentDirectory()
        self.tails: dict[str, tuple[TailFile, str, str]] = {}  # 実体のパス → (tail, "main"|"sub", agentId)
        self.events: deque[dict] = deque(maxlen=MAX_EVENTS)
        self.seq = 0
        self.last_poll = 0.0
        self.created = time.time()
        self.last_used = self.created  # 作ったばかりのものを先に捨てない  # 作り直すと番号が 1 から振り直しになる（画面はこれで取り直しを知る）
        self._goal: str | None = None
        self._lock = threading.Lock()

    def discover(self) -> None:
        self._add(self.main, "main", "", INITIAL_TAIL_MAIN)
        sub_dir = self.main.parent / self.sid / "subagents"
        for meta in sub_dir.glob("agent-*.meta.json"):
            self.agents.load_meta(meta)
        for f in sub_dir.glob("agent-*.jsonl"):
            self._add(f, "sub", f.stem.removeprefix("agent-"), INITIAL_TAIL_SUB)
        if self.tmp_base is not None:
            for f in self.tmp_base.glob(f"*/{self.sid}/tasks/*.output"):
                if TASK_OUTPUT_RE.match(f.name):
                    self._add(f, "sub", f.name.removesuffix(".output"), INITIAL_TAIL_SUB)

    def _add(self, path: Path, kind: str, agent_id: str, initial_tail: int) -> None:
        try:
            real = str(path.resolve(strict=True))
        except OSError:
            return
        if real not in self.tails:
            self.tails[real] = (TailFile(Path(real), initial_tail), kind, agent_id)

    def poll_once(self) -> int:
        with self._lock:
            self.last_poll = time.time()
            self.discover()
            batch: list[dict] = []
            # 本体を先に読む（Agent の呼び出しから号の名前を引くため）
            for real, (tail, kind, aid) in sorted(self.tails.items(), key=lambda kv: kv[1][1] != "main"):
                for raw in tail.read_new_lines():
                    if raw.startswith(SKIP_PREFIXES):
                        continue
                    try:
                        obj = json.loads(raw)
                    except ValueError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    try:
                        batch.extend(parse_main_line(obj, self.agents, self.owner) if kind == "main"
                                     else parse_sub_line(obj, aid, self.agents))
                    except Exception:  # noqa: BLE001  形の違う 1 行で、同じ周回のほかの行を捨てない
                        continue
            if not batch:
                return 0
            batch.sort(key=lambda e: e["ts"])
            for e in batch:
                self.seq += 1
                e["seq"] = self.seq
                self.events.append(e)
            return len(batch)

    def poll_if_due(self, min_sec: float = POLL_MIN_SEC) -> None:
        self.last_used = time.time()
        if time.time() - self.last_poll >= min_sec:
            self.poll_once()

    def goal(self) -> str:
        """最初の利用者の文（初めの指示）。頭だけ 1 回読んで覚える。"""
        if not self._goal:  # 始めたばかりで、まだ書かれていないときは次の回に読み直す
            self._goal = first_user_text(self.main)
        return self._goal

    def last_activity(self) -> float:
        """本体とサブエージェントの記録の、いちばん新しい更新の時刻（UNIX 秒）。"""
        best = 0.0
        for real in list(self.tails):
            try:
                best = max(best, os.stat(real).st_mtime)
            except OSError:
                continue
        return best

    def summary(self, agent: dict | None, procs: dict | None = None, now: float | None = None) -> dict:
        """選んだセッションの帯（状態・待っている理由・ゴール・今の作業・背景の処理）。"""
        now = time.time() if now is None else now
        with self._lock:
            evs = list(self.events)
            bg_tasks = dict(self.agents.bg_tasks)  # poll_once が別のスレッドで足すので、ロックの中で写す
        last = activity_mtime(self.main)  # タブと同じ材料（本体＋サブエージェントの記録）
        agent = agent or {}
        state = run_state(agent.get("state", ""), agent.get("status", ""), last, now) if agent else "past"
        bg = background_tasks(bg_tasks, procs or {}, now)
        own_text = next((e for e in reversed(evs) if e["cls"] == "sc" and e["kind"] == "text"), None)
        own_tool = next((e for e in reversed(evs) if e["cls"] == "sc" and e["kind"] == "tool"), None)
        orch = next((e for e in reversed(evs) if e["cls"] == "so" and e["kind"] == "text"), None)
        return {"state": state, "agent_state": agent.get("state", ""), "short": agent.get("short", ""),
                "reasons": wait_reasons(evs, state, agent.get("short", ""), bg) if state in ("waiting", "blocked") else [],
                "goal": self.goal(), "orch": orch, "text": own_text, "tool": own_tool, "last": last, "bg": bg}

    def since(self, seq: int, limit: int = 30) -> list[dict]:
        """``seq`` より後の出来事。``seq`` が 0（初回）・先の番号（起動し直し）なら、発言（ツール以外）が
        ``limit`` 件になるところから後ろを返す（間のツールの行も含める）。"""
        with self._lock:
            evs = list(self.events)
            last = self.seq
        if 0 < seq <= last:
            return [e for e in evs if e["seq"] > seq]
        n, start = 0, 0
        for i in range(len(evs) - 1, -1, -1):
            if evs[i]["kind"] != "tool":
                n += 1
                if n >= limit:
                    start = i
                    break
        return evs[start:]


class CouncilHub:
    """タブの一覧（動いているセッション＋最近の会議）と、セッションごとの出来事を渡す。

    中身を出すのは、``claude agents`` に載っている（このリポジトリの）セッションか、会議と判定したセッションだけ。
    """

    def __init__(self, dirs_fn, tmp_base: Path | None = None, window_hours: float = WINDOW_HOURS,
                 agents: "AgentList | None" = None, procs: "ProcScanner | None" = None) -> None:
        self.dirs_fn = dirs_fn  # () → 探すフォルダの一覧（呼ぶたびに探し直す。worktree が増えても追う）
        self.tmp_base = tmp_base
        self.window_sec = window_hours * 3600
        self.agents = agents
        self.procs = procs
        self.infos: dict[str, SessionInfo] = {}
        self.feeds: dict[str, SessionFeed] = {}
        self.listed: dict[str, dict] = {}  # sessionId → claude agents の 1 行（最後に取れた分）
        self.paths: dict[str, Path] = {}  # sessionId → 本体の記録（見つけたら覚える）
        self.last_discover = 0.0
        self.epoch = f"{time.time():.3f}"  # 起動し直すと番号が戻るので、画面はこれが替わったら初めから取り直す
        self._lock = threading.Lock()

    def discover(self, force: bool = False) -> None:
        with self._lock:
            now = time.time()
            if not force and now - self.last_discover < DISCOVER_SEC:
                return
            self.last_discover = now
            seen = set()
            for d in self.dirs_fn():
                for p in d.glob("*.jsonl"):
                    try:
                        st = p.stat()
                    except OSError:
                        continue
                    if st.st_mtime < now - self.window_sec:
                        continue
                    key = str(p)
                    seen.add(key)
                    info = self.infos.get(key)
                    if info is None:
                        info = self.infos[key] = SessionInfo(p)
                    info.refresh(st.st_mtime, st.st_size)
            for key in [k for k in self.infos if k not in seen]:
                del self.infos[key]

    def sessions(self) -> list[dict]:
        """最近の会議のセッション（古い形の一覧。tabs の材料）。"""
        self.discover()
        with self._lock:  # discover が別のスレッドで agent_types を足している間に回さない
            rows = [i.public() for i in self.infos.values() if i.council]
        rows.sort(key=lambda r: r["mtime"], reverse=True)
        return rows

    def find_main(self, sid: str) -> Path | None:
        """sessionId → 本体の記録。24 時間より古くても探す（止まったままのセッションも出す）。"""
        p = self.paths.get(sid)
        if p is not None and p.is_file():
            return p
        for d in self.dirs_fn():
            q = d / f"{sid}.jsonl"
            if q.is_file():
                self.paths[sid] = q
                return q
        return None

    def tabs(self, now: float | None = None) -> tuple[list[dict], bool]:
        """(タブの行, claude agents を読めたか)。統括役を先頭に、動いているセッション → 最近の会議の順。"""
        now = time.time() if now is None else now
        council = {r["id"]: r for r in self.sessions()}
        rows, ok = self.agents.get() if self.agents else ([], False)
        listed = {r["sid"]: r for r in rows}
        with self._lock:
            if ok or rows:
                self.listed = listed
        out = []
        for r in rows:
            main = self.find_main(r["sid"])
            last = activity_mtime(main) if main else 0.0
            c = council.get(r["sid"], {})
            out.append({"id": r["sid"], "short": r["short"], "name": r["name"] or c.get("name") or r["short"],
                        "state": run_state(r["state"], r["status"], last, now), "agent_state": r["state"],
                        "place": r["place"], "started": r["started"], "mtime": last, "members": c.get("members", []),
                        "found": main is not None})
        for sid, c in council.items():
            if sid not in listed:
                out.append({"id": sid, "short": sid[:8], "name": c["name"], "state": "past", "agent_state": "",
                            "place": "", "started": 0, "mtime": c["mtime"], "members": c["members"], "found": True})
        out.sort(key=lambda t: (not t["name"].startswith("claude-orchestrator"), t["state"] == "past",
                                t["started"] if t["state"] != "past" else -t["mtime"]))
        return out, ok

    def _info(self, sid: str) -> SessionInfo | None:
        for i in self.infos.values():
            if i.sid == sid and i.council:
                return i
        return None

    def agent_row(self, sid: str) -> dict | None:
        return self.listed.get(sid)

    def feed(self, sid: str) -> SessionFeed | None:
        """claude agents に載っているか、会議と判定したセッションだけ読む（ほかのセッションの中身は出さない）。"""
        self.discover()
        with self._lock:
            info = self._info(sid)
            row = self.listed.get(sid)
        if info is None and row is None:
            return None
        found = self.find_main(sid) if info is None else None  # ファイルを探す間はロックを持たない
        with self._lock:
            if info is not None:
                path, name = info.path, info.name
            else:
                if found is None:
                    return None
                path, name = found, row["name"]
            owner = f"担当 {safe_label(name)}" if name else "担当"
            f = self.feeds.get(sid)
            if f is None:
                f = SessionFeed(path, self.tmp_base, owner)
                self.feeds[sid] = f
                while len(self.feeds) > MAX_FEEDS:
                    old = min(self.feeds.values(), key=lambda x: x.last_used)
                    del self.feeds[old.sid]
            elif name and f.owner != owner:
                f.owner = owner
        f.poll_if_due()
        return f

    def summary(self, sid: str, feed: SessionFeed) -> dict:
        procs = self.procs.get() if self.procs else {}
        return feed.summary(self.listed.get(sid), procs)


# ── 動いているセッションの一覧（claude agents --json） ──

AGENTS_SEC = 30.0  # claude agents を呼ぶのは 30 秒に 1 回まで（画面が何枚開いていても）
ACTIVE_SEC = 180.0  # 直近 3 分に記録の動きがあれば「実行中」
SID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
SHORT_RE = re.compile(r"[0-9a-f]{8}")
AGENT_STATES = ("working", "blocked", "done")


def run_claude_agents() -> list:
    """``claude agents --json``（動いている裏のセッションの一覧）。読むだけ（セッションには何も送らない）。"""
    path = os.pathsep.join([str(Path.home() / ".local" / "bin"), os.environ.get("PATH") or "/usr/bin:/bin"])
    exe = shutil.which("claude", path=path)
    if not exe:
        raise FileNotFoundError("claude")
    env = dict(os.environ, PATH=path, CLAUDE_CONFIG_DIR=str(claude_base()))
    r = subprocess.run([exe, "agents", "--json"], capture_output=True, timeout=20, env=env,
                       stdin=subprocess.DEVNULL, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"claude agents: {r.returncode}")
    d = json.loads(r.stdout)
    if not isinstance(d, list):
        raise ValueError("claude agents: not a list")
    return d


def agent_row(x, repo_root: str | None) -> dict | None:
    """claude agents の 1 行 → タブに使う分だけ（cwd は worktree の名前だけにする）。このリポジトリの外は捨てる。"""
    if not isinstance(x, dict):
        return None
    sid = str(x.get("sessionId", ""))
    if not SID_RE.fullmatch(sid):
        return None
    cwd = str(x.get("cwd", ""))
    if repo_root and not (cwd == repo_root or cwd.startswith(repo_root.rstrip("/") + "/")):
        return None
    short = str(x.get("id", ""))
    if not SHORT_RE.fullmatch(short):
        short = sid[:8]
    place = "本体"
    if "/.claude/worktrees/" in cwd:
        place = safe_label(cwd.split("/.claude/worktrees/", 1)[1].split("/", 1)[0], 60)
    state = str(x.get("state", ""))
    started = x.get("startedAt")
    return {"sid": sid, "short": short, "name": safe_label(str(x.get("name", "")), 80) if x.get("name") else "",
            "state": state if state in AGENT_STATES else "other",
            "status": str(x.get("status", "")) if x.get("status") in ("busy", "idle") else "",
            "place": place, "started": float(started) / 1000 if isinstance(started, (int, float)) else 0.0}


class AgentList:
    """claude agents の結果を覚える。呼ぶのは ``interval`` 秒に 1 回まで・同時に 1 本まで（失敗も覚える）。"""

    def __init__(self, fn=run_claude_agents, repo_root: str | None = None, interval: float = AGENTS_SEC,
                 clock=time.monotonic) -> None:
        self.fn = fn
        self.repo_root = repo_root
        self.interval = interval
        self.clock = clock
        self.rows: list[dict] = []
        self.ok = False
        self.last: float | None = None
        self.calls = 0
        self._lock = threading.Lock()

    def get(self) -> tuple[list[dict], bool]:
        if not self._lock.acquire(blocking=self.last is None):  # 呼んでいる間は前の結果を返す（初回だけは待つ）
            return list(self.rows), self.ok
        try:
            now = self.clock()
            if self.last is None or now - self.last >= self.interval:
                self.last = now
                self.calls += 1
                try:
                    rows = [r for r in (agent_row(x, self.repo_root) for x in self.fn()) if r]
                    self.rows, self.ok = rows, True
                except Exception:  # noqa: BLE001  前の一覧を残す（画面を空にしない）
                    self.ok = False
            return list(self.rows), self.ok
        finally:
            self._lock.release()


def run_state(agent_state: str, status: str, last: float, now: float) -> str:
    """4 つの状態: running（実行中）・waiting（待機中）・blocked（止まっている）・done（終わった）。

    blocked・done は claude agents の state のまま。それ以外は、claude agents の status（busy → 実行中・idle → 待機中）。
    status が無いときは、記録が直近 ``ACTIVE_SEC`` 秒に動いていれば実行中、そうでなければ待機中。
    """
    if agent_state == "done":
        return "done"
    if agent_state == "blocked":
        return "blocked"
    if status:  # claude agents が busy・idle を明かしていれば、それを使う
        return "running" if status == "busy" else "waiting"
    return "running" if (last and now - last <= ACTIVE_SEC) else "waiting"


def activity_mtime(main: Path) -> float:
    """本体の記録と、そのサブエージェントの記録のいちばん新しい更新の時刻。"""
    best = 0.0
    for p in [main, *(main.parent / main.stem / "subagents").glob("agent-*.jsonl")]:
        try:
            best = max(best, p.stat().st_mtime)
        except OSError:
            continue
    return best


GOAL_SCAN_HEAD = 2 * 1024 * 1024


def first_user_text(path: Path, head: int = GOAL_SCAN_HEAD) -> str:
    """記録の頭から、最初の利用者の文（skill の差し込み・通知・要約からの再開・手で打ったコマンドを除く）。"""
    try:
        with path.open("rb") as f:
            data = f.read(head)
    except OSError:
        return ""
    lines = data.split(b"\n")
    if len(data) >= head:
        lines = lines[:-1]  # 書きかけ・切れた行は捨てる
    for raw in lines:
        if raw.startswith(SKIP_PREFIXES) or b'"user"' not in raw:
            continue
        try:
            obj = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(obj, dict) or obj.get("type") != "user" or obj.get("isMeta"):
            continue
        msg = obj.get("message")
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            content = "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
        if not isinstance(content, str):
            continue
        if any(tag in content for tag in RAW_OUTPUT_TAGS) or "<task-notification" in content or "<agent-message" in content:
            continue
        x = XSESSION_RE.search(content)
        body = (x.group(2) if x else REMINDER_RE.sub("", content)).strip()
        if not body or body.startswith("This session is being continued"):
            continue
        return clip(mask_secrets(body), TEXT_LIMIT)
    return ""


# ── 待っている理由（記録の末尾から作る規則。推測で作らない） ──

ASK_RE = re.compile(r"承認をお願い|承認をください|承認を待|判断をください|ご判断|判断をお願い|判断を仰|判断を待|"
                    r"返事が(?:無|な)ければ|返事を待|指示をください|指示を待|確認をお願い|ご確認ください|お待ちしています")
TIME_WAIT_RE = re.compile(r"((?:[0-9]{1,2}/[0-9]{1,2}|[0-9]{1,2}:[0-9]{2}|[0-9]{1,2} ?時)(?:（[^）]{1,4}）)? ?"
                          r"(?:以降|を待|まで待|になったら|になってから))")
APPROVAL_TOOLS = frozenset({"ExitPlanMode", "AskUserQuestion"})
WAIT_TOOLS = frozenset({"Monitor"})
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?])|\n")
NO_REASON = "理由は記録から読み取れない"


def _sentence(text: str, m: re.Match) -> str:
    """一致した場所を含む 1 文。"""
    start = max(text.rfind("。", 0, m.start()), text.rfind("\n", 0, m.start())) + 1
    ends = [i for i in (text.find("。", m.end()), text.find("\n", m.end())) if i >= 0]
    end = min(ends) + 1 if ends else len(text)
    return one_line(text[start:end].strip(), 120)


def wait_reasons(events: list[dict], state: str, short: str = "", bg: list[dict] | None = None) -> list[str]:
    """待機中・止まっているときの「何を待っているか・理由」（強い順）。規則は tests/scripts/test_council_live.py で固定。

    1. 止まっている（blocked）: 最後の自分の動きが ExitPlanMode・AskUserQuestion → 承認の画面。ほかは承認の画面の可能性。
    2. 背景の処理が ps に生きている → その処理の終わりを待っている。
       生きているか分からなくても、最後の自分のツールが背景の Bash（run_in_background。sleep・until の待ちも含む）・
       Monitor で、その後にその番号の通知が来ていない → 同じ。前景の sleep・until は、終わってから待機中になるので見ない。
    3. 最後の自分のツールが SendMessage で、その後に返事（統括役宛てなら統括役の発言、ほかの宛先ならほかの話し手の発言）
       が無い → 宛先の返事を待っている。
    4. 最後の自分の動きがサブエージェントの呼び出し → その報告を待っている。
    5. 最後の自分の発言に「承認をお願い」「判断をください」「返事が無ければ」など → 統括役・ユーザーの返事を待っている。
    6. 最後の自分の発言に「12/1 以降」「17:00 以降」など → 時刻を待っている。
    どれにも当たらなければ「理由は記録から読み取れない」。
    """
    out: list[str] = []
    sid = short if SHORT_RE.fullmatch(short or "") else ""
    attach = f"（claude attach {sid}）" if sid else ""
    own = [(i, e) for i, e in enumerate(events) if e.get("cls") == "sc"]
    last_i, last = own[-1] if own else (-1, None)
    tool_i, tool = next(((i, e) for i, e in reversed(own) if e["kind"] == "tool"), (-1, None))
    text_i, text = next(((i, e) for i, e in reversed(own) if e["kind"] == "text"), (-1, None))
    after = lambda i, cls: any(e.get("cls") == cls for e in events[i + 1:])  # noqa: E731
    if state == "blocked":
        if last is not None and last["kind"] == "tool" and last.get("tool") in APPROVAL_TOOLS:
            out.append(f"ユーザーの承認の画面で止まっている{attach}")
        else:
            out.append(f"承認の画面などで止まっている可能性{attach}")
    for b in (bg or [])[:3]:
        out.append(f"背景の処理の終わりを待っている: {b['line']}（{b['minutes']} 分）")
    if not bg and tool is not None and (tool.get("bg") or tool.get("tool") in WAIT_TOOLS):
        task = tool.get("task")  # 番号が分かれば、その番号の通知が来たら終わり。分からなければ（Monitor など）どの通知でも
        done = any(e.get("cls") == "sn" and (not task or e.get("task") == task) for e in events[tool_i + 1:])
        if not done:
            out.append(f"背景の処理の終わりを待っている: {tool['text']}")
    if tool is not None and tool.get("tool") == "SendMessage":
        to = tool.get("to") or ""
        orch = not to or "orchestrator" in to or to == "main"
        replied = any(e.get("cls") not in ("sc", "sn") and (e.get("cls") == "so" or not orch) for e in events[tool_i + 1:])
        if not replied:
            out.append("統括役の返事を待っている" if orch else f"{to} の返事を待っている")
    if last is not None and last["kind"] == "call":
        out.append(f"サブエージェントの報告を待っている: {one_line(last['text'], 100)}")
    if text is not None and not after(text_i, "so"):
        m = ASK_RE.search(text["text"])
        if m:
            out.append(f"統括役・ユーザーの返事を待っている: {_sentence(text['text'], m)}")
        t = TIME_WAIT_RE.search(text["text"])
        if t:
            out.append(f"時刻を待っている: {t.group(1)}")
    return out or [NO_REASON]


# ── 背景の処理（そのセッションが run_in_background で流したもの。ps と記録の照合。読むだけ） ──

TASK_LINK_RE = re.compile(r"/tasks/([A-Za-z0-9]{4,40})\.output$")
PROCS_SEC = 10.0


class ProcScanner:
    """``/proc/<pid>/fd/1`` が ``…/tasks/<番号>.output`` を指す処理を、番号ごとにまとめる（命令行・環境変数は読まない）。"""

    def __init__(self, proc_root: Path = Path("/proc"), interval: float = PROCS_SEC, clock=time.monotonic) -> None:
        self.root = proc_root
        self.interval = interval
        self.clock = clock
        self.last: float | None = None
        self.found: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _boot(self) -> float:
        try:
            for line in (self.root / "stat").read_text().splitlines():
                if line.startswith("btime "):
                    return float(line.split()[1])
        except (OSError, ValueError):
            pass
        return 0.0

    def scan(self) -> dict[str, dict]:
        boot, hz = self._boot(), os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        out: dict[str, dict] = {}
        try:
            pids = [d for d in self.root.iterdir() if d.name.isascii() and d.name.isdigit()]
        except OSError:
            return out
        for d in pids:
            try:
                m = TASK_LINK_RE.search(os.readlink(d / "fd" / "1"))
            except OSError:
                continue
            if not m:
                continue
            started = 0.0
            try:
                stat = (d / "stat").read_text()
                fields = stat[stat.rindex(")") + 2:].split()
                started = boot + int(fields[19]) / hz if boot else 0.0
            except (OSError, ValueError, IndexError):
                pass
            t = out.setdefault(m.group(1), {"pids": 0, "started": started})
            t["pids"] += 1
            if started and (not t["started"] or started < t["started"]):
                t["started"] = started
        return out

    def get(self) -> dict[str, dict]:
        with self._lock:
            now = self.clock()
            if self.last is None or now - self.last >= self.interval:
                self.last = now
                self.found = self.scan()
            return dict(self.found)


def background_tasks(bg_tasks: dict[str, tuple[str, str]], procs: dict[str, dict], now: float) -> list[dict]:
    """記録の背景の処理のうち、ps に生きているもの（名前は記録の Bash の 1 行。経過は処理の始まりから）。"""
    out = []
    for task, (line, ts) in bg_tasks.items():
        p = procs.get(task)
        if not p:
            continue
        mins = int(max(0.0, now - p["started"]) // 60) if p.get("started") else 0
        out.append({"task": task, "line": line, "ts": ts, "pids": p["pids"], "minutes": mins})
    out.sort(key=lambda b: b["ts"])
    return out


def default_hub(repo_root: Path) -> CouncilHub:
    live_dir = os.environ.get("KEIBA_COUNCIL_LIVE_DIR")  # 決め打ちのフォルダ（試験・手で見るとき）
    try:
        hours = float(os.environ.get("KEIBA_COUNCIL_LIVE_HOURS", "") or WINDOW_HOURS)
    except ValueError:
        hours = WINDOW_HOURS
    if live_dir:
        dirs = lambda: [Path(live_dir)]  # noqa: E731
    else:
        base, prefix = claude_base(), project_prefix(repo_root)
        dirs = lambda: project_dirs(base, prefix)  # noqa: E731
    root = str(repo_root)
    if "/.claude/worktrees/" in root:
        root = root.split("/.claude/worktrees/", 1)[0]
    agents = None if os.environ.get("KEIBA_COUNCIL_LIVE_AGENTS") == "0" else AgentList(run_claude_agents, root)
    return CouncilHub(dirs, default_tmp_base(), hours, agents, ProcScanner())
