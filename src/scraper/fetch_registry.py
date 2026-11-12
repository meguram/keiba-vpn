"""このPCで動くデータ取得の一覧（``scripts/dev/fetch_status.sh`` の元）。読み書きはローカルの /proc と /tmp だけ（GCS・netkeiba には触れない）。

取得が「見えるかどうか」は ps の走査で決める（:func:`is_fetch_args` / :func:`launcher_script`）。心拍ファイルの無い取得
（古いコードのまま動いている取得、NetkeibaClient を通らない JRA・気象庁・smartrc の取得）も漏らさないため。
心拍ファイル（``/tmp/keiba_fetch_tasks/<pid>-<uid>.json``）は補足で、GET の件数・待機中かどうか・進捗を足す。

新しい取得を足すときの決まり（docs/operations/fetch_tasks.md）:
  - 入口のモジュールを :data:`FETCH_MODULE_PATTERN` に足す（sh から起動するものは tests/scraper/test_fetch_registry.py が分類を強制する）。
  - NetkeibaClient を通る GET は自動で数える。長い取得は :func:`progress` で進捗（done/total）を書くと、終了見込みが出る。
  - sh の待ち受け（前の取得の終了を待ってから起動する、など）は ``python -m src.scraper.fetch_registry mark --pid $$ ...`` で登録する。

注意: ``backfill_guard.OTHER_FETCH_PATTERN``（補填が譲る相手の判定）とは別物。このモジュール・表示のプロセス名がそれに当たらないようにする。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

REGISTRY_DIR = Path(os.environ.get("KEIBA_FETCH_REGISTRY_DIR") or "/tmp/keiba_fetch_tasks")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLUSH_EVERY_SEC = 5.0          # 心拍ファイルの書き込みの間隔（GET のたびには書かない）
BUCKET_SEC = 60                # GET 件数の 1 分ごとの集計（直近 1 時間の件数に使う）
KEEP_BUCKETS = 60

# 取得の入口（ps の引数に対する正規表現）。表示の対象を決めるだけで、取得の動きは変えない。
# 常駐のプロセス（src.scraper.scheduler・src.ops.fetch_agent）は入れない（取っていない間も「実行中」に出てしまう。取得中は心拍で出る）
FETCH_MODULE_PATTERN = (
    r"-m\s+src\.(?:"
    r"scraper\.(?:auto_scrape|backfill|jma_weather|course_position|entries_sync|horse_profile_eve|recent_retry|shutuba_audit"
    r"|availability_watch|structure_monitor|jra_\w+|smartrc_\w+)"
    r"|scripts\.scraping\.(?!fetch_status\b|migrate_|audit_|check_and_build_|fix_sex_|gcs_coverage_|scrape_progress_monitor\b)\w+"
    r"|scripts\.data\.(?:backfill_(?:all|horse_lap|race_barometer_gcs|race_lists_kaisai_since_2020|race_result_pace|sanrenpuku_odds|trifecta_odds)\b|entries_probe|acq_\w+|tp37_comments|megu_index_scrape_\w+)"
    r"|research\.pedigree\.(?:sporthorse_\w+|jbis_births|wiki_birthdays|birth_subplans|pedigree_similarity)"
    r"|monitor\.page_quality_check"
    r"|ops\.fetch_child"
    r")\b"
)
_FETCH_RE = re.compile(FETCH_MODULE_PATTERN)
_NOT_FETCH_RE = re.compile(r"--report\b|--dry-run\b|--help\b")   # 状況の表示・見積もりだけの起動は取得に数えない
_SH_TOKEN = re.compile(r"(\S+\.sh)\b")

_lock = threading.Lock()
_state: dict[str, Any] = {}
_last_flush = 0.0
_cleaned = [False]


def _after_fork_in_child() -> None:
    # 他のスレッドが書き込みの途中で fork すると、子では _lock が取れなくなる。子では作り直す
    global _lock, _last_flush
    _lock = threading.Lock()
    _last_flush = 0.0


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


def is_fetch_args(args: str) -> bool:
    """ps の引数が取得の入口の python か。"""
    return bool(_FETCH_RE.search(args)) and not _NOT_FETCH_RE.search(args)


def module_of(args: str) -> str | None:
    m = re.search(r"-m\s+(src\.[\w.]+)", args)
    return m.group(1) if m else None


def _strip_comments(text: str) -> str:
    """コメント行を除き、行末の \\ で続く行を 1 行につなぐ（bash と同じく、コメントの \\ では継続しない）。"""
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return re.sub(r"\\\n", " ", text)


_launcher_cache: dict[tuple[str, float], bool] = {}


def script_launches_fetch(path: Path) -> bool:
    """sh の中身（コメントを除く）に取得の入口の起動があるか。中身で判定するので、新しい sh・作業用の待ち受けも自動で拾える。"""
    try:
        key = (str(path), path.stat().st_mtime)
    except OSError:
        return False
    if key not in _launcher_cache:
        try:
            body = _strip_comments(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            body = ""
        _launcher_cache[key] = any(is_fetch_args(line) for line in body.splitlines())
    return _launcher_cache[key]


def launcher_script(args: str, cwd: str | None = None) -> str | None:
    """sh のプロセスの引数から、取得を起動する sh のパスを返す（無ければ None）。cron_run.sh などの包みの中も見る。"""
    if not re.match(r"^(?:\S*/)?(?:ba|da)?sh\s", args) and ".sh" not in args.split(None, 1)[0]:
        return None
    for tok in _SH_TOKEN.findall(args):
        p = Path(tok)
        # 相対パスはプロセスの cwd で解決する。root のプロセスの cwd は jovyan から読めないので、リポジトリの直下でも試す
        cands = [p] if p.is_absolute() else [Path(b) / p for b in (cwd, PROJECT_ROOT) if b]
        if any(script_launches_fetch(c) for c in cands):
            return tok
    return None


def proc_start_ticks(pid: int) -> int | None:
    """/proc/<pid>/stat の開始時刻（起動からのクロック数）。PID の使い回しを見分けるのに使う。"""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return int(stat.rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _file_for(pid: int) -> Path:
    # ファイル名にユーザーを入れる（1777 のディレクトリでは他のユーザーのファイルを上書き・削除できないので、PID の使い回しで書けなくなるのを防ぐ）
    return REGISTRY_DIR / f"{pid}-{os.geteuid()}.json"


def _cleanup_dead() -> None:
    """自分のユーザーの、死んだプロセスの心拍・書きかけの一時ファイルを消す（他のユーザーの分は消せないので触らない）。"""
    uid = os.geteuid()
    try:
        for f in REGISTRY_DIR.iterdir():
            try:
                st = f.lstat()
                if f.is_symlink() or st.st_uid != uid:
                    continue
                if f.name.startswith("."):
                    if time.time() - st.st_mtime > 3600:
                        f.unlink()
                elif f.suffix == ".json" and proc_start_ticks(int(f.stem.split("-")[0])) is None:
                    f.unlink()
            except (OSError, ValueError):
                continue
    except OSError:
        pass


def _ensure_dir() -> bool:
    try:
        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        if REGISTRY_DIR.is_symlink():      # 置かれたシンボリックリンクをたどって chmod・書き込みをしない
            return False
        st = REGISTRY_DIR.stat()
        if st.st_uid == os.geteuid() and (st.st_mode & 0o7777) != 0o1777:
            os.chmod(REGISTRY_DIR, 0o1777)   # root（cron）と jovyan の両方が書く。umask 022 のままだと片方が書けない
        if not _cleaned[0]:
            _cleaned[0] = True
            _cleanup_dead()
        return os.access(REGISTRY_DIR, os.W_OK)
    except OSError:
        return False


def _disabled() -> bool:
    # テストの中では、ディレクトリを明示したときだけ書く（/tmp に心拍を散らかさない）
    return "PYTEST_CURRENT_TEST" in os.environ and "KEIBA_FETCH_REGISTRY_DIR" not in os.environ


def _write(entry: dict[str, Any], pid: int) -> None:
    if not _ensure_dir():
        return
    tmp = None
    try:
        # 一時ファイルは O_EXCL で作る（名前を予測したシンボリックリンクをたどって、他のファイルを上書きしない）
        fd, tmp = tempfile.mkstemp(prefix=f".{pid}.", suffix=".tmp", dir=REGISTRY_DIR)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            os.fchmod(fh.fileno(), 0o644)
            fh.write(json.dumps(entry, ensure_ascii=False))
        os.replace(tmp, _file_for(pid))
        tmp = None
    except OSError:
        pass
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _main_module() -> str | None:
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    return getattr(spec, "name", None)


def _base() -> dict[str, Any]:
    pid = os.getpid()
    if _state.get("pid") != pid:      # fork した子は自分の分を持ち直す
        _state.clear()
        _state.update({"pid": pid, "start_ticks": proc_start_ticks(pid), "argv": " ".join(sys.argv)[:300],
                       "module": _main_module(),
                       "state": "running", "gets": 0, "buckets": {}, "started_at": time.time()})
    return _state


def _snapshot(force: bool = False) -> dict[str, Any] | None:
    """書き込む中身（_lock の中で呼ぶ）。間引きで書かないときは None。"""
    global _last_flush
    now = time.time()
    if not force and now - _last_flush < FLUSH_EVERY_SEC:
        return None
    _last_flush = now
    st = dict(_state)
    st["buckets"] = dict(st.get("buckets", {}))
    st["updated_at"] = now
    return st


def _emit(snap: dict[str, Any] | None) -> None:
    # ファイルの書き込みは _lock の外で行う（他のスレッドの GET を待たせない）
    if snap:
        _write(snap, snap["pid"])


def note_get(n: int = 1) -> None:
    """GET を n 件数える（NetkeibaClient が自動で呼ぶ）。失敗しても取得は止めない。"""
    if _disabled():
        return
    try:
        with _lock:
            st = _base()
            now = time.time()
            st["gets"] += n
            st["last_get_at"] = now
            b = st["buckets"]
            key = str(int(now // BUCKET_SEC))
            b[key] = b.get(key, 0) + n
            if len(b) > KEEP_BUCKETS + 5:
                for k in sorted(b)[:-KEEP_BUCKETS]:
                    del b[k]
            resumed = st.get("state") == "waiting"
            if resumed:                       # GET が出たら待機は終わり。表示が待機中のまま残らないように、すぐに書く
                st["state"], st["waiting_for"] = "running", None
            snap = _snapshot(force=resumed)
        _emit(snap)
    except Exception:  # noqa: BLE001
        pass


def set_state(state: str, waiting_for: str | None = None) -> None:
    """"waiting"（待機中。waiting_for に待っている相手）か "running" に切り替える。すぐに書く。"""
    if _disabled():
        return
    try:
        with _lock:
            st = _base()
            if st.get("state") == state and st.get("waiting_for") == waiting_for:
                return
            st["state"], st["waiting_for"] = state, waiting_for
            st["state_since"] = time.time()
            snap = _snapshot(force=True)
        _emit(snap)
    except Exception:  # noqa: BLE001
        pass


def progress(done: int | float, total: int | float | None, unit: str = "件", label: str | None = None,
             req_per_unit: float | None = None) -> None:
    """進捗を書く（終了見込みは表示の側で GET の速度から出す。req_per_unit は 1 単位あたりの GET 件数の見込み）。"""
    if _disabled():
        return
    try:
        with _lock:
            st = _base()
            first = st.get("total") != total or (label is not None and st.get("label") != label)   # 始まり・対象の変化はすぐに書く
            st.update({"done": done, "total": total, "unit": unit, "progress_at": time.time()})
            if label:
                st["label"] = label
            if req_per_unit:
                st["req_per_unit"] = req_per_unit
            snap = _snapshot(force=first)
        _emit(snap)
    except Exception:  # noqa: BLE001
        pass


def mark(pid: int, state: str = "waiting", label: str | None = None, waiting_for: str | None = None) -> bool:
    """別のプロセス（sh の待ち受けなど）を登録する。そのプロセスが終われば、表示から自動で消える。
    登録した sh が取得の子を起動すれば、表示はその子に合わせて「実行中」になる（登録し直さなくてよい）。"""
    ticks = proc_start_ticks(pid)
    if ticks is None:
        return False
    entry = {"pid": pid, "start_ticks": ticks, "state": state, "waiting_for": waiting_for, "label": label,
             "marked": True, "updated_at": time.time(), "state_since": time.time()}
    try:
        entry["argv"] = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()[:300]
    except OSError:
        pass
    _write(entry, pid)
    return _file_for(pid).exists()


def gets_within(entry: dict[str, Any], seconds: float, now: float | None = None) -> int:
    now = now or time.time()
    lo = int((now - seconds) // BUCKET_SEC)
    return sum(n for k, n in (entry.get("buckets") or {}).items() if int(k) > lo)


def live_entries() -> dict[int, dict[str, Any]]:
    """生きているプロセスの心拍だけを返す（死んだ pid・開始時刻の合わない pid・壊れたファイルは飛ばす）。"""
    out: dict[int, dict[str, Any]] = {}
    try:
        files = []
        for f in REGISTRY_DIR.glob("*.json"):
            try:
                files.append((f.stat().st_mtime, f))
            except OSError:
                continue
    except OSError:
        return out
    for _, f in sorted(files):
        try:
            pid = int(f.stem.split("-")[0])
            e = json.loads(f.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        ticks = proc_start_ticks(pid)
        if ticks is None or not isinstance(e, dict) or (e.get("start_ticks") is not None and e["start_ticks"] != ticks):
            continue
        out[pid] = e          # 同じ pid に複数のユーザーの分があれば新しい方（mtime の順に上書き）
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="取得の一覧（fetch_status）への登録")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mark", help="プロセスを待機中・実行中として登録する（sh の待ち受けなど）")
    m.add_argument("--pid", type=int, required=True)
    m.add_argument("--state", choices=("waiting", "running"), default="waiting")
    m.add_argument("--label")
    m.add_argument("--waiting-for")
    sub.add_parser("list", help="生きている登録を JSON で出す")
    a = ap.parse_args(argv)
    if a.cmd == "mark":
        ok = mark(a.pid, a.state, a.label, a.waiting_for)
        print("登録した" if ok else f"登録できなかった（PID {a.pid} が無いか、{REGISTRY_DIR} に書けない）")
        return 0 if ok else 1
    print(json.dumps(live_entries(), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
