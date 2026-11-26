"""会議（rules-council・pedigree-lab など）のセッションの記録（JSONL）を差分で読み、話し手のラベル付きの出来事に直す。

council_viewer の /live と /live/api/*（画面が数秒おきに取りに来る）が使う。手元のファイルを読むだけ（GCS・外の網は読まない。費用 0 円）。

- 探す: ``<claude の設定>/projects/`` の下の、このリポジトリ（本体と .claude/worktrees/*）のフォルダの ``*.jsonl`` のうち、
  最近（既定 24 時間）更新されたもの。会議の顔ぶれ（roi-plus-man-1〜3・pedigree-1〜2）を呼んだセッションか、
  名前（agent-name・custom-title）が rules-council・pedigree-lab・bet-optimizer のものだけを出す（ほかのセッションの中身は出さない）。
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
    return "\n".join(out)


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
        self.latest: dict[str, str] = {}  # agentId → SendMessage で続きを頼んだときの要約（段・議題をこちらから取る）

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
           topic: str = "") -> dict:
    limit = TEXT_LIMIT if kind == "text" else LINE_LIMIT
    text = mask_secrets(text)
    return {"ts": ts or "", "speaker": speaker, "cls": cls, "kind": kind,
            "text": clip(text, limit) if kind == "text" else one_line(text, limit),
            "stage": stage, "decision": decision, "topic": topic}


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
                out.append(_event(ts, owner, "sc", "tool", _tool_line(b)))
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
            return [_event(ts, "通知", "sn", "notice", html.unescape(s.group(1)) if s else "通知")]  # 通知の中は &gt; などで書かれる
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
    """最近更新された記録から会議のセッションを探し、セッションごとの出来事を渡す。"""

    def __init__(self, dirs_fn, tmp_base: Path | None = None, window_hours: float = WINDOW_HOURS) -> None:
        self.dirs_fn = dirs_fn  # () → 探すフォルダの一覧（呼ぶたびに探し直す。worktree が増えても追う）
        self.tmp_base = tmp_base
        self.window_sec = window_hours * 3600
        self.infos: dict[str, SessionInfo] = {}
        self.feeds: dict[str, SessionFeed] = {}
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
        self.discover()
        with self._lock:  # discover が別のスレッドで agent_types を足している間に回さない
            rows = [i.public() for i in self.infos.values() if i.council]
        rows.sort(key=lambda r: r["mtime"], reverse=True)
        return rows

    def _info(self, sid: str) -> SessionInfo | None:
        for i in self.infos.values():
            if i.sid == sid and i.council:
                return i
        return None

    def feed(self, sid: str) -> SessionFeed | None:
        """会議と判定したセッションだけ読む（ほかのセッションの中身は出さない）。"""
        self.discover()
        with self._lock:
            info = self._info(sid)
            if info is None:
                return None
            f = self.feeds.get(sid)
            if f is None:
                f = SessionFeed(info.path, self.tmp_base, f"担当 {safe_label(info.name)}" if info.name else "担当")
                self.feeds[sid] = f
                while len(self.feeds) > MAX_FEEDS:
                    old = min(self.feeds.values(), key=lambda x: x.last_used)
                    del self.feeds[old.sid]
            elif info.name and f.owner != f"担当 {safe_label(info.name)}":
                f.owner = f"担当 {safe_label(info.name)}"
        f.poll_if_due()
        return f


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
    return CouncilHub(dirs, default_tmp_base(), hours)
