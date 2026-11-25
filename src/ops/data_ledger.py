"""データの「見た回数」の台帳（D1。計画: progress/ledger_plan_20261118.md）。

どの期間を・何の目的で・どの対象（系統＋版）で・誰が見たかを、追記だけの JSONL に 1 行ずつ残し、
評価の道具が結果（着順・払戻）を読む前に `guard()` を呼んで、決まりの外の読み出しを警告・拒否する。

- 判定の鍵は「系統（family）＋版（version。凍結の hash）＋期間（日付の区間の重なり）＋目的」。出力先・環境変数は鍵に入れない。
- 実行時の正はホストの共有の写し（全 worktree が同じホストで動く）。同じ行を作業ツリーの git の台帳にも書く（ホストの故障に備える）。
- 決まり（期間・目的ごとの扱い・enforce の系統）は `config/data_access/policy.json`。
- 既定は「警告＋記録」。`policy.json` の `enforce_families` に入れた系統、または `KEIBA_LEDGER_MODE=enforce` で拒否する
  （環境変数は enforce に上げるだけで、warn に下げることはできない）。

費用 0 円（ホストのファイルと git だけ。GCS・GCP に触れない）。
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import secrets
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
JST = timezone(timedelta(hours=9))
POLICY = ROOT / "config/data_access/policy.json"
GIT_LEDGER = ROOT / "config/data_access/ledger.jsonl"
SHARED_REL = Path("work/progress/data_access/ledger.jsonl")


def _shared_default(env: Any = None, euid: int | None = None) -> Path:
    """ホストの共有の写しの場所（全 worktree・clone から同じ場所）。

    SUDO_USER は、本当に sudo で root として動いているとき（euid 0）だけ使い、元の利用者の HOME にする。
    root でない処理に SUDO_USER が残っているとき（統括役が立てたセッションの SUDO_USER=root など）は使わない
    （2026-11-25: SUDO_USER=root で /root/... を指し、読めない台帳を空とみなして「0 回」と記録した）。
    """
    env = os.environ if env is None else env
    euid = os.geteuid() if euid is None else euid
    su = str(env.get("SUDO_USER") or "").strip()
    if euid == 0 and su and su != "root":
        try:
            home = Path(f"~{su}").expanduser()
        except (RuntimeError, KeyError):   # 知らない利用者（版により例外か ~名前 のまま）
            home = None
        if home is not None and home.is_absolute():
            return home / SHARED_REL
    return _own_home() / SHARED_REL


def _own_home() -> Path:
    """いま動いている利用者（euid）の家。HOME は見ない（SUDO_USER=root と一緒に HOME=/root が残っていても取り違えない）。"""
    try:
        import pwd

        return Path(pwd.getpwuid(os.geteuid()).pw_dir)
    except (ImportError, KeyError):          # passwd に無い uid のコンテナなど
        return Path.home()


SHARED_DEFAULT = _shared_default()

PURPOSES = ("select", "confirm", "eval_only", "reproduce", "forward", "fit")
# 見た回数に数える目的（reproduce・fit は別に数える）
_SEEN = ("select", "confirm", "eval_only", "forward")
_PERIOD_RULES = ("free", "record", "sealed", "forward")


class LedgerDenied(SystemExit):
    """決まりの外の読み出しを拒否した（enforce）。SystemExit なので、道具はそのまま止まる。"""


# ---------------------------------------------------------------- 置き場

# テストの印。tests/conftest.py だけが True にする（環境変数では切り替えない: 本番で空の台帳を指す抜け道を塞ぐ）
TESTING = False


def _under_pytest() -> bool:
    return TESTING


_TEST_DIR: Path | None = None


def _test_dir() -> Path:
    global _TEST_DIR
    if _TEST_DIR is None:
        import atexit
        import shutil

        _TEST_DIR = Path(tempfile.mkdtemp(prefix="keiba-ledger-test-"))
        atexit.register(shutil.rmtree, _TEST_DIR, True)
    return _TEST_DIR


def shared_path() -> Path:
    """実行時の正（ホストの共有の写し）。場所を環境変数で変えられるのはテストのときだけ（本番で空の台帳を指す抜け道を塞ぐ）。"""
    if _under_pytest():
        env = os.environ.get("KEIBA_LEDGER_SHARED")
        return Path(env) if env else _test_dir() / "shared.jsonl"
    return SHARED_DEFAULT


def git_path() -> Path:
    """作業ツリーの git の台帳。テストのときは一時ディレクトリ（テストで本物の台帳を書かない）。"""
    if _under_pytest():
        env = os.environ.get("KEIBA_LEDGER_GIT")
        return Path(env) if env else _test_dir() / "git.jsonl"
    return GIT_LEDGER


def policy_path() -> Path:
    if _under_pytest() and os.environ.get("KEIBA_LEDGER_POLICY"):
        return Path(os.environ["KEIBA_LEDGER_POLICY"])
    return POLICY


def load_policy(path: Path | None = None) -> dict:
    return json.loads((path or policy_path()).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 読み書き

def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            text = f.read()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    lines = text.split("\n")
    if lines and lines[-1].strip():
        _say(f"[台帳] {path} の最後の行が改行で終わっていない（書きかけ・強制終了）。読み飛ばす")
    for i, line in enumerate(lines[:-1], 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            _say(f"[台帳] {path}:{i} が JSON として読めない。読み飛ばす（手で直す）: {exc}")
    return rows


def load_rows(paths: Iterable[Path] | None = None, errors: list | None = None) -> list[dict]:
    """共有の写し ∪ git の台帳（id で重複を除く。順は at の昇順）。

    errors に list を渡すと、読めない置き場を (path, 例外) で足して残りを読む（渡さなければ例外をそのまま上げる）。
    """
    seen: dict[str, dict] = {}
    for p in (paths if paths is not None else (shared_path(), git_path())):
        try:
            got = _read_jsonl(Path(p))
        except OSError as exc:
            if errors is None:
                raise
            errors.append((Path(p), exc))
            continue
        for r in got:
            rid = r.get("id")
            if rid and rid not in seen:
                seen[rid] = r
    return sorted(seen.values(), key=lambda r: (str(r.get("at", "")), str(r.get("id"))))


def _append(path: Path, row: dict, *, make_parent: bool = True) -> None:
    if make_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()                      # fsync はしない（1 回 0.2 秒ほどかかり、同時の書き込みで待ちが積もる。追記の 1 行は flock で守る）
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _shared_usable(path: Path) -> bool:
    """共有の写しの置き場が、読み書きできるディレクトリか（無い・ファイル・権限が無いなら False）。"""
    try:
        if not path.parent.is_dir() or not os.access(path.parent, os.R_OK | os.W_OK | os.X_OK):
            return False
        return not path.exists() or os.access(path, os.R_OK | os.W_OK)
    except OSError:
        return False


def write_row(row: dict, *, strict_shared: bool) -> list[str]:
    """共有の写しと git の台帳の両方に 1 行を足す。書けなかった置き場の説明を返す（strict_shared なら共有に書けないとき止める）。"""
    errs: list[str] = []
    for label, p in (("git の台帳", git_path()), ("共有の写し", shared_path())):   # git を先に（共有で止めても行を残す）
        try:
            # 共有の置き場は作らない（無い場所に作ると、次からは空の写しを正とみなして回数を少なく数える）
            _append(p, row, make_parent=(label != "共有の写し"))
        except OSError as exc:
            errs.append(f"{label} {p} に書けない: {exc}")
            if strict_shared and label == "共有の写し":
                raise LedgerDenied(f"[台帳] {errs[-1]}（enforce では、記録できない読み出しを拒否する）") from exc
    return errs


# ---------------------------------------------------------------- 期間

def _d(x: str | date | None) -> date | None:
    if x is None or x == "":
        return None
    if isinstance(x, date):
        return x
    return date.fromisoformat(str(x)[:10])


def overlaps(a_from, a_to, b_from, b_to) -> bool:
    """閉区間 [a_from, a_to] と [b_from, b_to] が重なるか（None は無限）。"""
    af, at, bf, bt = _d(a_from), _d(a_to), _d(b_from), _d(b_to)
    lo_ok = at is None or bf is None or bf <= at
    hi_ok = bt is None or af is None or af <= bt
    return lo_ok and hi_ok


def periods_hit(period: tuple, policy: dict) -> list[dict]:
    return [p for p in policy["periods"] if overlaps(period[0], period[1], p.get("from"), p.get("to"))]


# ---------------------------------------------------------------- 判定

@dataclass
class Decision:
    decision: str                      # allowed / downgraded / warned / denied / skipped
    purpose: str                       # 実際に記録する目的（eval_only に落ちることがある）
    mode: str                          # warn / enforce
    violations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    periods: list[str] = field(default_factory=list)


def _starts(rows: list[dict]) -> list[dict]:
    """「読む前」の行（kind=start か backfill・申告）と、その後の終わりの行の状態をまとめる。"""
    fin: dict[str, dict] = {}
    amend: dict[str, dict] = {}
    base = {r.get("id"): r for r in rows if r.get("kind") in ("start", "record")}
    pol = None
    for r in rows:
        if r.get("kind") == "finish" and r.get("ref"):
            fin[r["ref"]] = r
        if r.get("kind") == "amend":
            if pol is None:
                pol = load_policy()
            why = _amend_problem(r, base.get(r.get("ref")), pol, amend)
            if why:
                _say(f"[台帳] 直しの行 {r.get('id')} を使わない: {why}")
            else:
                amend[r["ref"]] = r                 # 1 つの行への直しは 1 回だけ
    out = []
    for r in rows:
        if r.get("kind") in ("start", "record"):
            f = fin.get(r["id"], {})
            a = amend.get(r["id"])
            extra = {"purpose": a["purpose"], "_amended_from": r.get("purpose"), "_amend_id": a["id"]} if a else {}
            out.append({**r, **extra, "_status": f.get("status", "started"), "_result": f.get("result_path"), "_auto_error": f.get("auto", False)})
    return out


# 許す直し: confirm → eval_only だけ（見た回数は減らさない。「採否に使った確かめ」を「評価だけ」に改めるだけ）
_AMEND_OK = {("confirm", "eval_only")}


def _amend_problem(a: dict, orig: dict | None, pol: dict, done: dict) -> str | None:
    if orig is None:
        return "直す行（start・record）が台帳に無い"
    if a.get("ref") in done:
        return "同じ行の直しは 1 回だけ"
    if (orig.get("purpose"), a.get("purpose")) not in _AMEND_OK:
        return f"直せるのは confirm→eval_only だけ（{orig.get('purpose')}→{a.get('purpose')}）"
    fam = (orig.get("target") or {}).get("family")
    if fam in (pol.get("amend_locked_families") or []):
        return f"系統 {fam} は事前登録の 1 回だけの確かめなので直せない"
    ap = str(a.get("approved_by") or "").strip()
    if ap not in (pol.get("amend_approvers") or []):
        return f"承認者 {ap or '（空）'} が policy.json の amend_approvers に無い"
    if ap == a.get("session"):
        return "承認者と直した担当が同じ"
    if not str(a.get("reason") or "").strip():
        return "理由が無い"
    return None


def amend(*, ref: str, purpose: str, reason: str, approved_by: str, session: str | None = None) -> dict:
    """過去の行の目的を直す（追記だけの台帳なので、直しの行を足す）。直せるのは confirm→eval_only だけで、承認者は
    policy.json の amend_approvers（統括役・ユーザー）に限る。直しは、すでに結果の JSON に埋め込んだ summary は変えない。"""
    sess = session_name(session)
    row = {"id": _new_id(sess) + "-amend", "kind": "amend", "ref": ref, "at": _now(), "session": sess,
           "purpose": purpose, "reason": reason, "approved_by": approved_by}
    lock_f = open(shared_path().parent / ".ledger.lock", "a") if _shared_usable(shared_path()) else None
    try:
        if lock_f is not None:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
        rows = load_rows()
        base = {r.get("id"): r for r in rows if r.get("kind") in ("start", "record")}
        done = {r.get("ref"): r for r in rows if r.get("kind") == "amend"}
        why = _amend_problem(row, base.get(ref), load_policy(), done)
        if why:
            raise ValueError(why)
        for e in write_row(row, strict_shared=False):
            _say("[台帳] " + e)
    finally:
        if lock_f is not None:
            fcntl.flock(lock_f, fcntl.LOCK_UN)
            lock_f.close()
    return row


def _counted(r: dict) -> bool:
    """数える行か。道具が自分で書いた error（例外）で結果の置き場が無い行だけは数えない（落ちた後の再実行を許す）。"""
    if r.get("decision") == "denied":
        return False
    if r["_status"] == "error" and r["_auto_error"] and not r["_result"]:
        return False
    return True


def tally(rows: list[dict], period: tuple, family: str | None = None) -> dict:
    """period に重なる行の数（全体・family・目的ごと）。count_unknown の行は「以上」の印を付ける。"""
    c = {"total": 0, "family": 0, "family_confirm": 0, "lower_bound": False, "by_purpose": {}}
    for r in _starts(rows):
        if not _counted(r):
            continue
        p = r.get("period") or {}
        if not overlaps(period[0], period[1], p.get("from"), p.get("to")):
            continue
        pur = r.get("purpose", "不明")
        c["by_purpose"][pur] = c["by_purpose"].get(pur, 0) + 1
        if pur not in _SEEN:
            continue
        c["total"] += 1
        if r.get("count_unknown"):
            c["lower_bound"] = True
        if family and (r.get("target") or {}).get("family") == family:
            c["family"] += 1
            if pur == "confirm":
                c["family_confirm"] += 1
    return c


def mode_for(family: str, policy: dict) -> str:
    if os.environ.get("KEIBA_LEDGER_MODE", "").strip().lower() == "enforce":
        return "enforce"
    if family in (policy.get("enforce_families") or []) or policy.get("default_mode") == "enforce":
        return "enforce"
    return "warn"


def decide(rows: list[dict], *, family: str, version: str, period: tuple, purpose: str, reason: str,
           policy: dict, frozen_dirty: Sequence[str] = (), frozen_given: bool = True) -> Decision:
    """読み出しの可否を決める（純粋な関数。書き込みはしない）。"""
    if purpose not in PURPOSES:
        raise ValueError(f"purpose は {PURPOSES} のどれか: {purpose}")
    pf, pt = _d(period[0]), _d(period[1])
    if pf and pt and pf > pt:
        raise ValueError(f"期間の始まりが終わりより後: {period}")
    mode = mode_for(family, policy)
    hit = periods_hit(period, policy)
    rules = {p["rule"] for p in hit}
    d = Decision("allowed", purpose, mode, periods=[p["name"] for p in hit])
    if not hit or rules <= {"free"}:
        d.decision = "skipped"            # 開発の期間だけ: 記録しない
        return d
    d.counts = tally(rows, period, family)
    sealed = rules & {"sealed", "forward"}
    prior = [r for r in _starts(rows) if _counted(r)
             and overlaps(period[0], period[1], (r.get("period") or {}).get("from"), (r.get("period") or {}).get("to"))
             and (r.get("target") or {}).get("family") == family and r.get("purpose") in _SEEN]
    same = [r for r in prior if str((r.get("target") or {}).get("version")) == str(version)]

    if family not in (policy.get("families") or {}):
        d.violations.append(f"unknown_family: 系統 {family} が config/data_access/policy.json の families に無い（名前を変えて数を逃れる道を塞ぐ）")
    if purpose == "fit" and any(p["rule"] == "forward" or (p["rule"] == "sealed" and p["name"] != "test") for p in hit):
        d.violations.append("fit_outside: fit（学習に入れる）は test（2025 年）の期間だけ")
    if purpose == "reproduce" and not frozen_given:
        d.violations.append("reproduce_unfrozen: reproduce は凍結・事前登録のファイル（frozen_paths）が必須")
    if purpose == "select" and sealed:
        d.violations.append(f"select_sealed: 封印・test・前向きの期間（{', '.join(d.periods)}）を「選ぶ」目的で読もうとした")
    if purpose == "forward" and "sealed" in rules:
        d.violations.append("forward_outside: forward は前向きの期間だけ（封印・test の期間に重なる）")
    if purpose == "confirm" and sealed:
        if any(r.get("purpose") == "confirm" for r in same):
            d.violations.append(f"confirm_again: 同じ対象（{family}・{version}）で、この期間の確かめは済んでいる（出力先を変えても同じ）")
        elif same or prior:
            d.purpose = "eval_only"
            d.decision = "downgraded"
            who = "同じ対象" if same else f"同じ系統（{family}）の別の版"
            d.notes.append(f"{who}がこの期間をすでに見ている（系統で {d.counts['family']} 回{'以上' if d.counts['lower_bound'] else ''}）。"
                           "確かめを「評価だけ」に落とした: 採否・選び直しに使わない。採否に使うなら新しい確かめの期間を立てる")
            if not reason:
                reason = "自動: 確かめから評価だけに落とした"
    if d.purpose in ("eval_only", "reproduce") and not reason.strip():
        d.violations.append(f"reason_missing: {d.purpose} は理由（reason）の文が必須")
    if purpose in ("confirm", "reproduce") and frozen_dirty:
        d.violations.append("frozen_dirty: 凍結・事前登録のファイルが commit されていないか変更がある: " + ", ".join(frozen_dirty))
    if d.violations:
        d.decision = "denied" if mode == "enforce" else "warned"
    return d


# ---------------------------------------------------------------- git

def _git_dirty(paths: Sequence[Path | str]) -> list[str]:
    bad = []
    for p in paths:
        p = Path(p).resolve()
        try:
            rel = str(p.relative_to(ROOT))
        except ValueError:
            bad.append(f"{p}（リポジトリの外）")
            continue
        tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel], capture_output=True).returncode == 0
        clean = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", rel], capture_output=True).returncode == 0
        if not (tracked and clean):
            bad.append(rel)
    return bad


def _git_head() -> str:
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or "不明"


def session_name(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if os.environ.get("KEIBA_SESSION"):
        return os.environ["KEIBA_SESSION"]
    parts = ROOT.parts
    if ".claude" in parts and "worktrees" in parts:
        return parts[parts.index("worktrees") + 1] if parts.index("worktrees") + 1 < len(parts) else "不明"
    return "不明"


# ---------------------------------------------------------------- 道具が呼ぶ口

def _now() -> str:
    return datetime.now(JST).isoformat(timespec="seconds")


def _new_id(session: str) -> str:
    return f"{datetime.now(JST):%Y%m%dT%H%M%S%f}-{session}-{secrets.token_hex(3)}"   # マイクロ秒まで（同じ秒の行を書いた順に並べる）


def _say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@dataclass
class Token:
    id: str | None
    family: str
    version: str
    period: tuple
    purpose: str
    decision: Decision
    mode: str
    _finished: bool = False

    def summary(self) -> dict:
        """結果の JSON に載せる台帳の要約（「この期間を何回見たか」を報告に必ず残す）。"""
        c = self.decision.counts or {}
        unk = bool(c.get("unknown"))         # 共有の写しが使えず、回数が分からない（0 と書かない）
        return {"ledger_id": self.id, "purpose": self.purpose, "decision": self.decision.decision,
                "periods": self.decision.periods,
                "seen_total": None if unk else c.get("total", 0), "seen_family": None if unk else c.get("family", 0),
                "seen_lower_bound": unk or c.get("lower_bound", False), "seen_unknown": unk,
                "seen_at_least": {"total": c.get("total", 0), "family": c.get("family", 0)} if unk else None,
                "notes": self.decision.notes,
                "violations": self.decision.violations,
                "use_for_decision": self.purpose in ("select", "confirm", "forward") and not self.decision.violations}

    def finish(self, result_path: str | Path | None = None, status: str = "done", error: str = "", auto: bool = False) -> None:
        if self._finished or self.id is None:
            self._finished = True
            return
        self._finished = True
        row = {"id": f"{self.id}-fin", "kind": "finish", "ref": self.id, "at": _now(), "status": status,
               "result_path": str(result_path) if result_path else None, "auto": auto}
        if error:
            row["error"] = error[:500]
        for e in write_row(row, strict_shared=False):
            _say("[台帳] " + e)

    def check_dates(self, dates: Iterable[Any], what: str = "読んだ表") -> None:
        """実際に読んだ日付の範囲が、申告した期間からはみ出していないかを確かめる（期間を小さく申告して広く読む道を塞ぐ）。"""
        ds = sorted({_d(str(x)) for x in dates if x is not None and str(x) not in ("", "NaT", "nan", "None")})
        if not ds:
            return
        lo, hi = ds[0], ds[-1]
        pf, pt = _d(self.period[0]), _d(self.period[1])
        if (pf and lo < pf) or (pt and hi > pt):
            msg = f"[台帳] {what} の日付 {lo}〜{hi} が、申告した期間 {self.period[0]}〜{self.period[1]} からはみ出している"
            if self.id is not None:
                write_row({"id": f"{self.id}-range-{secrets.token_hex(2)}", "kind": "violation", "ref": self.id, "at": _now(),
                           "violation": "range_outside", "range": [str(lo), str(hi)]}, strict_shared=False)
            if self.mode == "enforce":
                raise LedgerDenied(msg)
            _say(msg + "（warn: 記録して続ける）")

    def check_race_ids(self, race_ids: Iterable[Any], what: str = "読んだ表") -> None:
        """race_id（先頭 4 桁が年）しか無い表の確かめ。年の範囲で申告した期間と比べる。"""
        ys = sorted({int(str(r)[:4]) for r in race_ids if r is not None and str(r)[:4].isdigit()})
        if not ys:
            return
        pf, pt = _d(self.period[0]), _d(self.period[1])
        lo_ok = pf is None or ys[0] >= pf.year
        hi_ok = pt is None or ys[-1] <= pt.year
        if not (lo_ok and hi_ok):
            self.check_dates([f"{ys[0]}-01-01", f"{ys[-1]}-12-31"], what)

    def __enter__(self) -> "Token":
        return self

    def __exit__(self, et, ev, tb) -> bool:
        if et is None or (issubclass(et, SystemExit) and not issubclass(et, LedgerDenied) and ev.code in (0, None)):
            self.finish()
        elif issubclass(et, LedgerDenied):
            self.finish(status="aborted", error=str(ev))
        elif issubclass(et, Exception):
            # 道具の例外: 数えない（落ちた後の再実行を許す）
            self.finish(status="error", error=f"{et.__name__}: {ev}", auto=True)
        else:
            # Ctrl-C・0 以外の SystemExit: 結果を見た後に止めた可能性があるので数える
            self.finish(status="aborted", error=f"{et.__name__}: {ev}")
        return False


def guard(*, tool: str, family: str, version: str, period: tuple[str, str | None], purpose: str,
          reads: Sequence[str] = ("着順", "払戻"), frozen_paths: Sequence[Path | str] = (), reason: str = "",
          session: str | None = None, note: str = "", target_name: str = "", meta: dict | None = None) -> Token:
    """結果を読む直前に呼ぶ。決まりの外なら warn は警告＋記録、enforce は LedgerDenied（SystemExit）で止める。

    period は (YYYY-MM-DD, YYYY-MM-DD)（終わりは None で無限）。family は系統、version は凍結の hash。
    meta には仮説の数・config の sha など、道具ごとの補足を入れる（そのまま行に残る）。
    返す Token の purpose は、同じ期間をすでに見ているとき eval_only に落ちていることがある（結果の採否に使わない）。
    `with guard(...) as tok:` で使うと、例外で落ちたときに error の行を自動で書く（落ちた後の再実行を許すため）。
    """
    policy = load_policy()
    sess = session_name(session)
    mode = mode_for(family, policy)
    lock_f = None
    try:
        lock_f = open(shared_path().parent / ".ledger.lock", "a")
        fcntl.flock(lock_f, fcntl.LOCK_EX)          # 読む→決める→書くを 1 つに（同時の 2 つの confirm を両方通さない）
    except OSError as exc:              # 置き場が無い: enforce では下の判定で拒否、warn では記録の時に作る
        lock_f = None
        _say(f"[台帳] 共有の置き場のロックが取れない: {exc}")
    try:
        return _guard_locked(tool=tool, family=family, version=version, period=period, purpose=purpose, reads=reads,
                             frozen_paths=frozen_paths, reason=reason, sess=sess, note=note, target_name=target_name,
                             meta=meta, policy=policy, mode=mode)
    finally:
        if lock_f is not None:
            fcntl.flock(lock_f, fcntl.LOCK_UN)
            lock_f.close()


def _guard_locked(*, tool, family, version, period, purpose, reads, frozen_paths, reason, sess, note, target_name,
                  meta, policy, mode) -> Token:
    errs: list = []
    rows = load_rows(errors=errs)            # 読めない置き場は飛ばして、読めた分だけで数える（下限になる）
    for p, exc in errs:
        _say(f"[台帳] {p} が読めない: {exc}")
    # 共有の写しが無い・読めない・書けない（別のホスト・取り違え・権限）: 回数は「不明」。enforce では通さない
    shared_ok = _shared_usable(shared_path()) and not any(p == shared_path() for p, _ in errs)
    d_unknown = not shared_ok
    dirty = _git_dirty(frozen_paths) if frozen_paths else []
    d = decide(rows, family=family, version=str(version), period=period, purpose=purpose, reason=reason,
               policy=policy, frozen_dirty=dirty, frozen_given=bool(frozen_paths))
    if not shared_ok and d.decision != "skipped":
        d.violations.append(f"shared_missing: 共有の写し {shared_path()} が無い・読めない・書けない（見た回数は不明）")
        d.decision = "denied" if mode == "enforce" else "warned"
        d.counts["unknown"] = True
        if d.notes:                          # 上の注記の回数は、読めた台帳だけで数えた下限
            d.notes.append("上の注記の回数は、共有の写しが使えず、読めた台帳だけで数えた下限（実際は以上）")
    if d.decision == "skipped":
        return Token(None, family, str(version), period, purpose, d, mode)
    if d_unknown:
        # 0 と書かない（2025・26 年の eval_only の決まりを回数で守っている）。読めた分（git の台帳など）は下限として残す
        counts_before = {"total": None, "family": None, "unknown": True,
                         "at_least": {"total": d.counts.get("total", 0), "family": d.counts.get("family", 0)}}
    else:
        counts_before = {"total": d.counts.get("total", 0), "family": d.counts.get("family", 0)}
    row = {"id": _new_id(sess), "kind": "start", "at": _now(), "session": sess, "tool": tool,
           "target": {"family": family, "version": str(version), "name": target_name},
           "period": {"from": period[0], "to": period[1], "names": d.periods},
           "reads": list(reads), "purpose": d.purpose, "requested_purpose": purpose, "reason": reason or None,
           "decision": d.decision, "mode": mode, "violations": d.violations, "notes": d.notes,
           "git_head": _git_head(), "note": note or None, "meta": meta or None, "source": "tool",
           "counts_before": counts_before}
    # 共有が使えないと分かっているときは、下の違反の一覧（shared_missing）で拒否する
    for e in write_row(row, strict_shared=(mode == "enforce" and shared_ok)):
        _say("[台帳] " + e)
    head = (f"[台帳] {family}・{version} の {period[0]}〜{period[1]}（{'・'.join(d.periods)}）: 目的 {purpose}"
            f"{' → ' + d.purpose if d.purpose != purpose else ''}。")
    if d_unknown:
        head += (f"この期間を見た回数は不明（共有の写しが使えない。読めた台帳だけで全体 {d.counts.get('total', 0)} 回以上・"
                 f"この系統で {d.counts.get('family', 0)} 回以上。今回を除く）")
    else:
        head += (f"この期間は全体で {d.counts.get('total', 0)} 回{'以上' if d.counts.get('lower_bound') else ''}・"
                 f"この系統で {d.counts.get('family', 0)} 回見ている（今回を除く）")
    _say(head)
    for n in d.notes:
        _say("[台帳] " + n)
    if d.violations:
        msg = "[台帳] 決まりの外: " + " / ".join(d.violations)
        if d.decision == "denied":
            raise LedgerDenied(msg + "（拒否。記録は残した）")
        _say(msg + "（warn: 記録して続ける。enforce に切り替えた後は拒否される）")
    return Token(row["id"], family, str(version), period, d.purpose, d, mode)


def declare(*, family: str, version: str, period: tuple, purpose: str, reason: str, session: str | None = None,
            tool: str = "manual", reads: Sequence[str] = ("着順", "払戻"), at: str | None = None, source: str = "manual",
            count_unknown: bool = False, result_path: str | None = None, note: str = "", target_name: str = "",
            evidence: str = "") -> dict:
    """道具の外で見た分の申告・過去の分の起こし（判定せず、そのまま 1 行を足す）。"""
    sess = session_name(session)
    pol = load_policy()
    row = {"id": _new_id(sess) if source != "backfill" else f"bf-{family}-{version}-{period[0]}-{purpose}-{(at or '')[:16]}".replace(":", ""),
           "kind": "record", "at": at or _now(), "session": sess, "tool": tool,
           "target": {"family": family, "version": str(version), "name": target_name},
           "period": {"from": period[0], "to": period[1], "names": [p["name"] for p in periods_hit(period, pol)]},
           "reads": list(reads), "purpose": purpose, "reason": reason, "decision": "recorded", "source": source,
           "count_unknown": bool(count_unknown), "result_path": result_path, "note": note or None, "evidence": evidence or None}
    write_row(row, strict_shared=False)
    return row


# ---------------------------------------------------------------- 追記だけの検査

def check_append_only(base_text: str, now_text: str) -> list[str]:
    """base の行が now の先頭からそのまま残っているか（順は merge=union で入れ替わりうるので、行の集合で見る）。消えた・変わった行を返す。"""
    now = set(l for l in now_text.splitlines() if l.strip())
    return [l[:160] for l in base_text.splitlines() if l.strip() and l not in now]


# ---------------------------------------------------------------- CLI

def _cmd_show(a) -> int:
    rows = load_rows()
    period = (a.date_from, a.date_to)
    c = tally(rows, period)
    print(json.dumps({"period": period, **c}, ensure_ascii=False))
    for r in _starts(rows):
        p = r.get("period") or {}
        if not overlaps(period[0], period[1], p.get("from"), p.get("to")):
            continue
        t = r.get("target") or {}
        print(f"{r.get('at', '')[:16]}  {r.get('session', '不明'):<22} {t.get('family', ''):<14} {str(t.get('version', ''))[:14]:<14} "
              f"{p.get('from')}〜{p.get('to')}  {r.get('purpose'):<9} {r.get('decision', ''):<10} {r['_status'] if r.get('kind') == 'start' else '-':<7}"
              f"{' 以上' if r.get('count_unknown') else ''}  {r.get('source', '')}"
              f"{'  （直し: ' + str(r['_amended_from']) + '→' + str(r['purpose']) + '）' if r.get('_amend_id') else ''}")
    return 0


def _cmd_amend(a) -> int:
    try:
        row = amend(ref=a.ref, purpose=a.purpose, reason=a.reason, approved_by=a.approved_by, session=a.session)
    except ValueError as exc:
        print(f"直せない: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(row, ensure_ascii=False))
    return 0


def _cmd_declare(a) -> int:
    row = declare(family=a.family, version=a.version, period=(a.date_from, a.date_to), purpose=a.purpose,
                  reason=a.reason, session=a.session, note=a.note or "", result_path=a.result_path)
    print(json.dumps(row, ensure_ascii=False))
    return 0


def _cmd_sync(a) -> int:
    """共有の写しと git の台帳の行をそろえる（片方にしか無い行を、もう片方に足す）。"""
    sh, gi = shared_path(), git_path()
    a_rows, b_rows = _read_jsonl(sh), _read_jsonl(gi)
    a_ids, b_ids = {r["id"] for r in a_rows}, {r["id"] for r in b_rows}
    n1 = n2 = 0
    for r in a_rows:
        if r["id"] not in b_ids:
            _append(gi, r)
            n1 += 1
    for r in b_rows:
        if r["id"] not in a_ids:
            _append(sh, r, make_parent=False)   # 共有の置き場は作らない（無いなら止まる）
            n2 += 1
    print(f"git へ {n1} 行・共有へ {n2} 行を足した")
    return 0


def _cmd_backfill(a) -> int:
    """手書きの一覧（config/data_access/backfill_*.json）から過去の分の行を作る。--write で台帳に足す（同じ id は足さない）。"""
    src = json.loads(Path(a.src).read_text(encoding="utf-8"))
    have = {r["id"] for r in load_rows()}
    n = 0
    for e in src["entries"]:
        rid = f"bf-{e['id']}"
        at = datetime.fromisoformat(e["at"]).astimezone(JST).isoformat(timespec="seconds")   # 並べるため JST にそろえる
        row = {"id": rid, "kind": "record", "at": at, "session": e.get("session", "不明"), "tool": e.get("tool", "不明"),
               "target": {"family": e["family"], "version": str(e.get("version", "不明")), "name": e.get("name", "")},
               "period": {"from": e["period"][0], "to": e["period"][1]},
               "reads": e.get("reads", ["着順", "払戻"]), "purpose": e["purpose"], "reason": e.get("reason"),
               "decision": "recorded", "source": "backfill", "count_unknown": bool(e.get("count_unknown")),
               "result_path": e.get("result_path"), "evidence": e.get("evidence"), "note": e.get("note")}
        if rid in have:
            continue
        n += 1
        if a.write:
            write_row(row, strict_shared=False)
        else:
            print(json.dumps(row, ensure_ascii=False))
    print(f"{'足した' if a.write else '足す候補'}: {n} 行（既にある行は除いた）", file=sys.stderr)
    return 0


def _cmd_check_append_only(a) -> int:
    rel = str(GIT_LEDGER.relative_to(ROOT))
    if subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{a.base}^{{commit}}"], capture_output=True).returncode != 0:
        print(f"基準 {a.base} が無い（取り込み・名前を確かめる）")
        return 2
    r = subprocess.run(["git", "-C", str(ROOT), "show", f"{a.base}:{rel}"], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"基準 {a.base} に {rel} が無い（初めての追加）: 検査なし")
        return 0
    now = GIT_LEDGER.read_text(encoding="utf-8") if GIT_LEDGER.exists() else ""
    gone = check_append_only(r.stdout, now)
    if gone:
        print(f"台帳は追記だけ。{a.base} にあった行が消えた・変わった: {len(gone)} 行")
        for g in gone[:20]:
            print("  - " + g)
        return 1
    print(f"OK: {rel} は {a.base} からの追記だけ")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.ops.data_ledger", description="データの見た回数の台帳")
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("show", help="期間に重なる行と回数")
    s.add_argument("--from", dest="date_from", default="2025-01-01")
    s.add_argument("--to", dest="date_to", default=None)
    s.set_defaults(fn=_cmd_show)
    s = sp.add_parser("declare", help="道具の外で見た分の申告")
    s.add_argument("--family", required=True)
    s.add_argument("--version", required=True)
    s.add_argument("--from", dest="date_from", required=True)
    s.add_argument("--to", dest="date_to", required=True)
    s.add_argument("--purpose", required=True, choices=PURPOSES)
    s.add_argument("--reason", required=True)
    s.add_argument("--session")
    s.add_argument("--note")
    s.add_argument("--result-path")
    s.set_defaults(fn=_cmd_declare)
    s = sp.add_parser("amend", help="過去の行の目的を直す（統括役の承認が必須）")
    s.add_argument("--ref", required=True)
    s.add_argument("--purpose", required=True, choices=PURPOSES)
    s.add_argument("--reason", required=True)
    s.add_argument("--approved-by", required=True)
    s.add_argument("--session")
    s.set_defaults(fn=_cmd_amend)
    s = sp.add_parser("sync", help="共有の写しと git の台帳をそろえる")
    s.set_defaults(fn=_cmd_sync)
    s = sp.add_parser("backfill", help="過去の分を手書きの一覧から起こす")
    s.add_argument("--src", default=str(ROOT / "config/data_access/backfill_20261118.json"))
    s.add_argument("--write", action="store_true")
    s.set_defaults(fn=_cmd_backfill)
    s = sp.add_parser("check-append-only", help="git の台帳が基準から追記だけか（CI）")
    s.add_argument("--base", default="origin/develop")
    s.set_defaults(fn=_cmd_check_append_only)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
