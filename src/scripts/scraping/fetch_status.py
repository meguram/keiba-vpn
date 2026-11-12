"""このPCで実行中・待機中のデータ取得と、終わる見込みを 1 画面に出す（読み取りだけ。netkeiba・GCS へはアクセスしない）。

    watch -n 5 bash scripts/dev/fetch_status.sh

対象はこのPCのすべての取得（src/scraper/fetch_registry.py の判定。完了・停止中のものは出さない）:
  - 取得の入口の python（fetch_registry.FETCH_MODULE_PATTERN）と、それを起動する sh（中身で判定）を ps から拾い、
    起動元の sh ごとに 1 つのタスクにまとめる。sh だけが生きていて取得の子が無いときは「待機中」。
  - 心拍ファイル（/tmp/keiba_fetch_tasks/<pid>.json）があれば、GET の件数・待機中かどうか・進捗（progress）を足す。
以下の既知の取得は、動いているときに詳しい行を足す:
  - 2016〜2019 年の補填（``src.scraper.backfill --phase past``）: 段階ごとの残り件数を逆算し、実績の速度から段階ごとの終了見込みを出す。
    段階は PAST_TIERS の順に全年分を通して進むので、見込みは段階の累計で出す。
  - 調教の補填（``backfill_horse_training_recent``）: 今回の実行の残り頭数と終了見込み。

残り件数の係数（1 開催日のレース数、1 レース・1 頭あたりの件数）は、取得済みの分の実績から求める。実績が無い段階は
docs/operations/backfill_2016.md §4 の値を使う。出走馬の総数は fast が終わるまで確定しないので、未取得の開催日の新しい馬は
NEW_HORSE_DECAY を掛けた概算（年をさかのぼるほど既に集めた馬と重なる）。
速度は補填のログの GET 行から数える。他の取得に譲って待っている時間も含めた実効の速度で、見込みには直近 ETA_WINDOWS_H 時間の平均を使う。
1 頭あたりの件数は、最後の実行で触れた馬の数（25 頭ごとの保存を待たない）で割り、保存の間に残り件数が増減しないようにする。
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.scraper import backfill_guard as bg
from src.scraper import fetch_registry as fr
from src.scraper.backfill import PAST_TIERS, PAST_YEARS

ROOT = Path(__file__).resolve().parents[3]
PROGRESS = ROOT / "data/local/meta/backfill_progress.json"
TRAINING_PROGRESS = ROOT / "data/local/meta/horse_training_backfill_progress.json"
PAST_LOG = ROOT / "logs/backfill_past.log"
TRAINING_LOG = ROOT / "logs/horse_training_backfill.log"
TRAINING_PAST_PROGRESS = ROOT / "data/local/meta/horse_training_past_progress.json"
TRAINING_PAST_LOG = ROOT / "logs/horse_training_past.log"

JST = timezone(timedelta(hours=9))     # ログの時刻は JST（OS の時刻は UTC のことがある）
ETA_WINDOWS_H = (3,)                   # 見込みに使う速度の窓（時間）。譲り・中断・夜の切れ目も含めた実効の平均
SHOW_WINDOWS_H = (1, 3, 24)            # 速度の表示に使う窓（時間）
MIN_TOUCHED_HORSES = 30                # この頭数に満たない実行では、1 頭あたりの件数を全体の実績で求める
REQ_PER_RACE = {"cards": 2.0}          # 実績が無い段階の 1 レースあたりの件数（§4）。ここに無い段階は 1
REQ_PER_HORSE_DEFAULT = 4.0            # 実績が無いときの 1 頭あたりの件数（§4）
RACES_PER_DAY_DEFAULT = 32.0
NEW_HORSE_DECAY = 0.5                  # 未取得の開催日の新しい馬の数 ＝ 取得済みの開催日あたりの頭数 × この係数（概算）
TRAINING_REQ_PER_HORSE = 1.0
TRAINING_PAST_PAGES_PER_HORSE = 3.0   # 全ページ取り直し（10 走 1 ページ）の 1 頭あたりの件数の初期値（実績が 25 頭たまれば実績を使う）
STAGE_LABEL = {"fast": "結果", "horse": "馬・血統", "cards": "出馬表・追い切り", "lap": "ラップ", "odds": "オッズ", "pair": "2連系オッズ",
               "index": "指数", "past": "馬柱", "on_time": "速報", "anchors_f": "父方アンカー"}
REQUIRED_FOR_TRAINING = ("fast", "horse", "cards", "lap")   # 学習に必須の段階（§4）

_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[\w+\] (.*)$")
_RUN_START = re.compile(r"^===== (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\S* START")   # 起動スクリプトが書く実行の区切り（JST）
_GET = re.compile(r"scraper\.client: GET https://([a-z.]*netkeiba\.com)/(\w+)")
# オッズ・2連系・3連複・指数は JSON API で取り、client.get を通らないので GET の行が無い（新しい版は「GET … (api)」を書くが、
# 1 レースで複数回の API を呼ぶ段階もある）。残り件数の単位（1 レース = 1 件）に合わせ、保存の行を 1 件として数える
_API_SAVE = re.compile(r"scraper\.run: 保存: (race_odds|race_pair_odds|race_trifecta_odds|race_barometer)/")
_HORSE_GET = re.compile(r"scraper\.client: GET https://db\.netkeiba\.com/horse/(?:(result|ped)/)?(\w{10})/")   # 祖先の ID は英数字（000a…）のこともある


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def scan_log(path: Path, now: datetime, windows_h: tuple[int, ...] = SHOW_WINDOWS_H) -> dict:
    """ログの GET 行を数える: 窓ごとの件数、種類別の総数、最後の実行の開始からの件数と馬 GET の件数・触れた馬、最後の行の時刻と内容。"""
    out = {"windows": {w: 0 for w in windows_h}, "kinds": {}, "first": None, "last_ts": None, "last_msg": None,
           "run_start": None, "run_gets": 0, "run_horse_gets": 0, "run_horse_ids": set(), "run_horse_by_id": {},
           "prev_horse_by_id": {}}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return out
    cut = {w: now - timedelta(hours=w) for w in windows_h}
    for line in lines:
        st = _RUN_START.match(line)
        if st:
            if len(out["run_horse_by_id"]) >= MIN_TOUCHED_HORSES:      # 再起動直後は今回の実行が少ないので、直前の十分な実行を残す
                out["prev_horse_by_id"] = out["run_horse_by_id"]
            out["run_start"], out["run_gets"] = datetime.strptime(st.group(1), "%Y-%m-%dT%H:%M:%S"), 0
            out["run_horse_gets"], out["run_horse_ids"], out["run_horse_by_id"] = 0, set(), {}
            continue
        m = _TS.match(line)
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        out["first"] = out["first"] or ts
        out["last_ts"], out["last_msg"] = ts, m.group(2)
        g = _GET.search(line)
        if g and line.rstrip().endswith("(api)"):
            continue                       # API の GET は保存の行で数える（二重に数えない）
        a = None if g else _API_SAVE.search(line)
        if not g and not a:
            continue
        kind = f"{g.group(1).split('.')[0]}/{g.group(2)}" if g else f"api/{a.group(1)}"
        out["kinds"][kind] = out["kinds"].get(kind, 0) + 1
        out["run_gets"] += 1
        h = _HORSE_GET.search(line) if g else None
        if h:
            out["run_horse_gets"] += 1
            out["run_horse_ids"].add(h.group(2))
            by = out["run_horse_by_id"].setdefault(h.group(2), {})
            by[h.group(1) or "profile"] = by.get(h.group(1) or "profile", 0) + 1
        for w, c in cut.items():
            if ts >= c:
                out["windows"][w] += 1
    return out


def horse_breakdown(by_id: dict, target_ids: set) -> dict:
    """最後の実行の馬 GET を種類別に数える。対象の馬（プロフィール・戦績・血統）と、それ以外の ID ＝ 5 世代目の祖先（母方アンカーの血統）。
    取得の途中の馬（最後に触れた対象の馬）は 3 件そろっていないので数えず、``in_flight`` に分ける（1 頭あたりの値が揺れないように）。"""
    out = {"touched": 0, "profile": 0, "result": 0, "ped": 0, "anchor": 0, "in_flight": 0}
    last = next((h for h in reversed(list(by_id)) if h in target_ids), None)
    for hid, kinds in by_id.items():
        if hid in target_ids:
            if hid == last:
                out["in_flight"] = 1
                continue
            out["touched"] += 1
            for k, n in kinds.items():
                out[k] += n
        else:
            out["anchor"] += sum(kinds.values())
    return out


def rate_per_hour(scan: dict, window_h: int, now: datetime) -> float:
    """窓の中の件数 ÷ 窓の長さ（ログが窓より新しければ、ログの最初からの時間で割る）。"""
    first = scan.get("first")
    if not first:
        return 0.0
    hours = min(window_h, max((now - first).total_seconds() / 3600, 1e-6))
    return scan["windows"].get(window_h, 0) / hours


def past_remaining(prog: dict, horse_gets: int, years: list[int] = PAST_YEARS, horse_touched: int | None = None,
                   horse_parts: dict | None = None, horse_unsaved: int = 0) -> list[dict]:
    """段階ごとの残り件数（PAST_TIERS の順）。horse_touched は horse_gets を数えた期間に触れた馬の数（保存前の分を含む）。
    horse_parts（種類 → その期間の GET 数）があれば、horse の行に種類別の残り件数 ``parts`` を付ける。
    horse_unsaved は取得したが 25 頭ごとの保存をまだ待っている馬の数で、残りから引く（残りが 25 頭ずつ段になって動かないように）。"""
    dates = [d for y in years for d in prog.get(f"calendar_{y}", [])]
    n_days = len(dates)
    sf = prog.get("stats_fast", {})
    races_known = [sf[d]["races"] for d in dates if d in sf and sf[d].get("races")]
    races_per_day = sum(races_known) / len(races_known) if races_known else RACES_PER_DAY_DEFAULT

    fast_done = set(prog.get("done_fast", []))
    horse_ids = prog.get("past_horse_ids", [])
    horses_done = len(prog.get("past_horses_done", []))
    collected_dates = set(prog.get("past_horse_dates", []))
    uncollected = sum(1 for d in dates if d not in collected_dates)
    per_date = len(horse_ids) / len(collected_dates) if collected_dates else 0.0
    horses_left = max(len(horse_ids) - horses_done - horse_unsaved, 0) + uncollected * per_date * NEW_HORSE_DECAY
    if horse_touched and horse_touched >= MIN_TOUCHED_HORSES:
        req_per_horse = horse_gets / horse_touched
    else:
        req_per_horse = horse_gets / horses_done if horses_done else REQ_PER_HORSE_DEFAULT

    rows = []
    for tier in PAST_TIERS:
        if tier == "horse":
            total_units, left_units = len(horse_ids) + uncollected * per_date * NEW_HORSE_DECAY, horses_left
            row = {"tier": tier, "unit": "頭", "done": horses_done + horse_unsaved, "total": total_units, "left_req": left_units * req_per_horse,
                   "approx": uncollected > 0}
            if horse_parts and horse_touched and horse_touched >= MIN_TOUCHED_HORSES:
                row["parts"] = {k: left_units * n / horse_touched for k, n in horse_parts.items()}
            rows.append(row)
            continue
        if tier == "anchors_f":
            pend = len(prog.get("paternal_anchor_pending", []))
            ratio = pend / horses_done if horses_done else 0.0
            left = pend + horses_left * ratio           # 馬を取るたびに積まれる。既にある祖先は取らないので上限側の見込み
            rows.append({"tier": tier, "unit": "頭", "done": 0, "total": left, "left_req": left, "approx": True})
            continue
        done_set = fast_done if tier == "fast" else set(prog.get(f"done_{tier}", []))
        left_days = sum(1 for d in dates if d not in done_set)
        st = prog.get(f"stats_{tier}", {})
        if tier == "fast":
            per_race = 1.0
        else:
            races = sum(v.get("races", 0) for v in st.values())
            per_race = sum(v.get("fetched", 0) for v in st.values()) / races if races else REQ_PER_RACE.get(tier, 1.0)
        extra = 1.0 if tier == "fast" else 0.0     # fast は開催日ごとのレース一覧が 1 件
        rows.append({"tier": tier, "unit": "日", "done": n_days - left_days, "total": n_days,
                     "left_req": left_days * (races_per_day * per_race + extra), "approx": False})
    return rows


def _fmt_eta(now: datetime, hours: float | None) -> str:
    if hours is None:
        return "—（速度の実績なし）"
    if hours <= 0:
        return "完了"
    t = now + timedelta(hours=hours)
    d = hours / 24
    span = f"{hours:.1f} 時間" if hours < 48 else f"{d:.1f} 日" if d < 60 else f"{d / 7:.1f} 週"
    return f"{t:%Y-%m-%d %H:%M} ごろ（あと {span}）"


def _w(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _ljust(text: str, width: int) -> str:
    return text + " " * max(width - _w(text), 0)


def _rjust(text: str, width: int) -> str:
    return " " * max(width - _w(text), 0) + text


BAR_WIDTH = 30


def _bar(done: float, total: float | None, width: int = BAR_WIDTH, pct: bool = True) -> str:
    """進捗のバー。書式は既存の監視（scrape_progress_monitor.bar・scripts/dev/watch_backfill.sh）に合わせる: [████░░░░]  41.3%"""
    if not total or total <= 0:
        return "[" + "?" * width + "]"
    p = max(0.0, min(done / total, 1.0))
    fill = int(p * width)
    return "[" + "█" * fill + "░" * (width - fill) + "]" + (f" {p * 100:5.1f}%" if pct else "")


def _age(now: datetime, ts: datetime | None) -> str:
    if not ts:
        return "—"
    s = int((now - ts).total_seconds())
    return f"{s} 秒前" if s < 120 else f"{s // 60} 分前" if s < 7200 else f"{s // 3600} 時間前"


PART_LABEL = (("profile", "馬のプロフィール"), ("result", "馬の戦績"), ("ped", "血統（本体・5世代表）"),
              ("anchor", "5世代さかのぼり: 母方の祖先の血統"))


def _horse_parts_lines(row: dict, anchors_f: dict | None) -> list[str]:
    """horse の行の内訳（種類別の残り件数と 1 頭あたりの件数）。父方の祖先は別の段階（anchors_f）で取るので、その件数を添える。"""
    left_units = max(row["total"] - row["done"], 1e-9)
    lines = []
    for k, label in PART_LABEL:
        lines.append("     └ " + _ljust(label, 36) + _rjust(f"{row['parts'][k]:,.0f}", 10) + f"   （1頭あたり {row['parts'][k] / left_units:.2f} 件）")
    if anchors_f:
        lines.append("     └ " + _ljust("5世代さかのぼり: 父方（別の段階）", 36) + _rjust(f"{anchors_f['left_req']:,.0f}", 10) + "   （anchors_f の行。下に記載）")
    return lines




# ---------------------------------------------------------------------------
# このPCで動いている取得（実行中・待機中）を集める。完了・停止中のものは出さない
# ---------------------------------------------------------------------------

STALE_SEC = 600                      # 実行中なのに GET がこの秒数ないときは「止まっている?」を添える

TASK_LABEL = {
    "src.scraper.backfill": "過去分の補填（backfill）",
    "src.scripts.scraping.backfill_horse_training_recent": "調教の補填",
    "src.scripts.scraping.backfill_horse_training_past": "調教の過去分の取り直し（2016〜2019 年の馬・全ページ）",
    "src.scripts.scraping.backfill_horse_race_remarks": "馬の備考の補填",
    "src.scraper.auto_scrape": "定期取得（auto_scrape）",
    "src.scraper.jma_weather": "気象庁の天気",
    "src.scraper.course_position": "コースの位置取り",
    "src.scripts.scraping.fetch_sire_lines": "父系の系統",
    "src.research.pedigree.sporthorse_births": "海外の生年（sporthorse）",
    "src.monitor.page_quality_check": "ページの品質確認",
    "src.ops.fetch_agent": "取得エージェント",
    "src.ops.fetch_child": "取得エージェントの子",
}


def _ps() -> list[dict]:
    import subprocess

    out = subprocess.run(["ps", "-eo", "pid,ppid,user,etime,args"], capture_output=True, text=True, timeout=5).stdout.splitlines()[1:]
    rows = []
    for line in out:
        parts = line.strip().split(None, 4)
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        rows.append({"pid": int(parts[0]), "ppid": int(parts[1]), "user": parts[2], "etime": parts[3], "args": parts[4]})
    return rows


def _cwd(pid: int) -> str | None:
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return None


def _member_kind(row: dict, entry: dict | None, now_ts: float) -> str | None:
    """fetch（取得の python）・launcher（取得を起動する sh）・registered（心拍だけで分かる取得）のどれか。どれでもなければ None。"""
    args = row["args"]
    if re.search(r"fetch_status|-m\s+src\.scraper\.fetch_registry\b", args):   # 表示そのもの・登録の CLI
        return None
    if fr.is_fetch_args(args):
        return "fetch"
    if fr.launcher_script(args, _cwd(row["pid"])):
        return "launcher"
    if entry:
        # 入口のパターンに無い取得（研究用の python -c など）。登録したもの・待機中・直近に GET があるものだけを取得とみなす
        # （API のサーバーなど常駐のプロセスが一度取っただけで、ずっと「実行中」に出ないように）
        if (entry.get("marked") or entry.get("state") == "waiting" or entry.get("label") or entry.get("total") is not None
                or now_ts - (entry.get("last_get_at") or 0) < STALE_SEC):
            return "registered"
    return None


def collect_tasks(rows: list[dict] | None = None, entries: dict[int, dict] | None = None, now_ts: float | None = None) -> list[dict]:
    """取得のプロセスを、起動元の sh ごとに 1 つのタスクにまとめる。state は running（実行中）か waiting（待機中）。"""
    rows = _ps() if rows is None else rows
    entries = fr.live_entries() if entries is None else entries
    now_ts = now_ts or time.time()
    me = os.getpid()
    by_pid = {r["pid"]: r for r in rows}
    kinds = {}
    for r in rows:
        if r["pid"] == me:
            continue
        k = _member_kind(r, entries.get(r["pid"]), now_ts)
        if k:
            kinds[r["pid"]] = k

    def top_of(pid: int) -> tuple[int, int]:
        top, depth, cur, seen = pid, 0, by_pid.get(pid), set()
        while cur and cur["ppid"] in by_pid and cur["ppid"] not in seen:
            seen.add(cur["ppid"])
            cur = by_pid[cur["ppid"]]
            if cur["pid"] in kinds:
                top = cur["pid"]
            depth += 1
        return top, depth

    groups: dict[int, list[tuple[int, int]]] = {}
    for pid in kinds:
        top, depth = top_of(pid)
        groups.setdefault(top, []).append((depth, pid))

    children: dict[int, list[dict]] = {}
    for r in rows:
        children.setdefault(r["ppid"], []).append(r)

    tasks = []
    for top, members in groups.items():
        members.sort()
        mem = [dict(by_pid[p], kind=kinds[p], entry=entries.get(p)) for _, p in members]
        fetchers = [m for m in mem if m["kind"] in ("fetch", "registered") and not (m["entry"] or {}).get("marked")]
        if not fetchers and not any((m["entry"] or {}).get("marked") for m in mem):
            # sh だけが生きている: 子が sleep か無ければ待機中（再開・前の取得の終了を待っている）。
            # 取得でない子（取得の後の push・集計など）が動いていれば、取得は終わっているので出さない
            ids = {m["pid"] for m in mem}
            others = [c for p in ids for c in children.get(p, []) if c["pid"] not in ids]
            if any(not re.match(r"^(?:\S*/)?sleep\b", c["args"]) for c in others):
                continue
        # 待機中: 取得の python の心拍が waiting、または取得の python が無い。
        # 登録（mark）した sh の waiting は、その sh が取得の子を起動した後は数えない（子に合わせて実行中にする）
        waiting = [m for m in mem if (m["entry"] or {}).get("state") == "waiting" and (not fetchers or m in fetchers)]
        main = fetchers[-1] if fetchers else mem[0]
        marked_running = any((m["entry"] or {}).get("marked") and (m["entry"] or {}).get("state") == "running" for m in mem)
        state = "waiting" if waiting or not (fetchers or marked_running) else "running"
        waiting_for = next(((m["entry"] or {}).get("waiting_for") for m in waiting if (m["entry"] or {}).get("waiting_for")), None)
        if state == "waiting" and not waiting_for and not fetchers:
            waiting_for = "取得の起動待ち（前の取得の終了・再開の待ち）"
        label = next(((m["entry"] or {}).get("label") for m in mem if (m["entry"] or {}).get("label")), None)
        mod = fr.module_of(main["args"]) or (main["entry"] or {}).get("module")
        if not label:
            script = next((fr.launcher_script(m["args"], _cwd(m["pid"])) for m in mem if m["kind"] == "launcher"), None)
            label = TASK_LABEL.get(mod or "", mod or (Path(script).name if script else main["args"][:60]))
            if mod == "src.scraper.backfill" and "--phase past" in main["args"]:
                label = "2016〜2019 年の補填（--phase past）"
            t = re.search(r"--task\s+(\S+)", main["args"])
            if t:
                label += f": {t.group(1)}"
        tasks.append({"top": top, "members": mem, "main": main, "module": mod, "state": state,
                      "waiting_for": waiting_for, "label": label,
                      "entry": next((m["entry"] for m in reversed(fetchers) if m["entry"]), None)})
    tasks.sort(key=lambda t: (t["state"] != "running", t["top"]))
    return tasks


def _generic_lines(task: dict, now_ts: float, now: datetime) -> list[str]:
    L = []
    for m in task["members"]:
        tag = {"launcher": "起動元", "fetch": "取得", "registered": "取得"}[m["kind"]]
        L.append(f"   {tag} PID {m['pid']}（{m['user']}）経過 {m['etime']}  {m['args'].split(' -m ', 1)[-1][:90]}")
    if task["state"] == "waiting" and task["waiting_for"]:
        L.append(f"   待っている相手: {task['waiting_for']}")
    e = task["entry"]
    if e and e.get("gets"):
        last = e.get("last_get_at")
        age = _age(now, now - timedelta(seconds=now_ts - last)) if last else "—"
        L.append(f"   GET: 直近 1 時間 {fr.gets_within(e, 3600, now_ts):,} 件 ・ 最後の GET {age}（この実行で計 {e['gets']:,} 件）")
    if e and e.get("total"):
        done, total = float(e.get("done") or 0), float(e["total"])
        per = e.get("req_per_unit") or (e["gets"] / done if done and e.get("gets") else None)
        rate = fr.gets_within(e, 3600, now_ts)
        hours = (total - done) * per / rate if per and rate > 0 else None
        L.append(f"   進捗 {_bar(done, total)}  {done:,.0f}/{total:,.0f} {e.get('unit', '件')} → 終了: {_fmt_eta(now, hours) if done < total else '完了'}")
    return L


def _past_detail(now: datetime) -> tuple[list[str], bool]:
    """2016〜2019 年の補填の段階ごとの残りと終了見込み。2 つ目はログが「他の取得に譲って待機中」か。"""
    prog = _load_json(PROGRESS)
    past = scan_log(PAST_LOG, now)
    msg = past["last_msg"] or ""
    L = [f"   最後のログ: {_age(now, past['last_ts'])}  {msg[:80]}"]
    rates_show = {w: rate_per_hour(past, w, now) for w in SHOW_WINDOWS_H}
    L.append("   速度（他の取得への譲りを含む実効）: 直近 " + " ・ ".join(f"{w}時間 {r:.0f} 件/時" for w, r in rates_show.items()))

    targets = set(prog.get("past_horse_ids", []))
    by_id = past["run_horse_by_id"]
    bd = horse_breakdown(by_id, targets)
    from_prev = bd["touched"] < MIN_TOUCHED_HORSES and bool(past["prev_horse_by_id"])
    if from_prev:     # 再起動直後: 直前の実行の内訳を使う（内訳が消えないように）。その未保存分は取り直しなので差し引かない
        by_id = past["prev_horse_by_id"]
        bd = horse_breakdown(by_id, targets)
    touched, _ = bd.pop("touched"), bd.pop("in_flight")
    parts = bd
    unsaved = 0 if from_prev else len((set(by_id) & targets) - set(prog.get("past_horses_done", [])))
    if touched >= MIN_TOUCHED_HORSES:
        gets_for_ratio = sum(parts.values())
    else:
        gets_for_ratio, touched, parts, unsaved = past["kinds"].get("db/horse", 0), None, None, 0
    rows = past_remaining(prog, gets_for_ratio, horse_touched=touched, horse_parts=parts, horse_unsaved=unsaved) if any(f"calendar_{y}" in prog for y in PAST_YEARS) else []
    if not rows:
        L.append(f"   開催日のカレンダーが進捗ファイルにありません（{PROGRESS}）。見込みは出せません")
        return L, "待機中" in msg
    rates = {w: rate_per_hour(past, w, now) for w in ETA_WINDOWS_H}
    heads = [f"終了見込み（{w}時間平均）" for w in ETA_WINDOWS_H]
    L += ["", "   " + _ljust("段階", 26) + _ljust("", 15) + _rjust("進捗", 22) + _rjust("残り件数", 10) + "   " + "  ".join(_ljust(h, 34) for h in heads)]
    cum = 0.0
    milestone = None
    for row in rows:
        cum += row["left_req"]
        etas = ["完了" if row["left_req"] <= 0 else _fmt_eta(now, cum / rates[w] if rates[w] > 0 else None) for w in ETA_WINDOWS_H]
        pct = row["done"] / row["total"] * 100 if row["total"] else 0.0
        mark = "≈" if row["approx"] else ""
        prog_s = f"{row['done']:,}/{mark}{row['total']:,.0f}{row['unit']} {pct:3.0f}%" if row["tier"] != "anchors_f" else f"残り{mark}{row['total']:,.0f}頭"
        label = f"{row['tier']}（{STAGE_LABEL.get(row['tier'], '')}）"
        tier_bar = _bar(row["done"], row["total"], 12, pct=False) if row["tier"] != "anchors_f" else ""
        L.append("   " + _ljust(label, 26) + _ljust(tier_bar, 15) + _rjust(prog_s, 22) + _rjust(f"{row['left_req']:,.0f}", 10) + "   " + "  ".join(_ljust(e, 34) for e in etas))
        if row.get("parts"):
            L += _horse_parts_lines(row, rows[-1] if rows[-1]["tier"] == "anchors_f" else None)
        if row["tier"] == REQUIRED_FOR_TRAINING[-1]:
            milestone = cum
    L.append("   " + _ljust("合計", 26) + _ljust("", 15) + _rjust("", 22) + _rjust(f"{cum:,.0f}", 10))
    for name, req in ((f"学習に必須の段階（{'・'.join(REQUIRED_FOR_TRAINING)}）", milestone or 0.0), ("すべての段階", cum)):
        etas = [f"{w}時間平均 {_fmt_eta(now, req / rates[w] if rates[w] > 0 else None) if req else '完了'}" for w in ETA_WINDOWS_H]
        L.append(f"   ▶ {name}: " + " ／ ".join(etas))
    L.append("   ≈ は概算（出走馬の総数は fast の完了後に確定。父方アンカーは既存分を除く前の上限側）")
    return L, "待機中" in msg


def _training_detail(now: datetime) -> list[str]:
    tp = _load_json(TRAINING_PROGRESS)
    tr = scan_log(TRAINING_LOG, now, windows_h=(1,))
    run_h = (now - tr["run_start"]).total_seconds() / 3600 if tr["run_start"] else 0.0
    t1 = tr["run_gets"] / run_h if run_h > 0 else 0.0     # 今回の実行の開始から（起動直後の対象の集計の時間も含む）
    left = tp.get("remaining")
    c = tp.get("counts", {})
    return [f"   累計 保存 {c.get('saved', 0):,} ・ 読めない {c.get('unreadable', 0)} ・ 空 {c.get('empty', 0)} ・ 失敗 {c.get('error', 0)}"
            f"  ／ 最後のログ: {_age(now, tr['last_ts'])}",
            f"   今回の残り {left if left is not None else '?'} 頭（25 頭ごとに更新）・ 今回の実行 {t1:.0f} 件/時"
            f" → 今回の終了: {_fmt_eta(now, left * TRAINING_REQ_PER_HORSE / t1 if left and t1 > 0 else None)}"]


_REDO_START = re.compile(r"打ち切った馬の取り直し: 対象 \d+ 頭 / 取り直し済み (\d+) / 残り (\d+) 頭")


def _training_past_detail(now: datetime, redo: bool) -> list[str]:
    """調教の過去分の取り直し。redo（--redo-truncated）のときは、打ち切った馬の全ページの取り直しの進捗を出す。"""
    pp = _load_json(TRAINING_PAST_PROGRESS)
    pl_ = scan_log(TRAINING_PAST_LOG, now, windows_h=(1, 3))
    rate = rate_per_hour(pl_, 3, now)
    run_h = (now - pl_["run_start"]).total_seconds() / 3600 if pl_["run_start"] else 0.0
    if rate <= 0 and run_h > 0:
        rate = pl_["run_gets"] / run_h
    if redo:
        before, left0 = 0, None
        try:
            for line in TRAINING_PAST_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
                m = _REDO_START.search(line)
                if m:
                    before, left0 = int(m.group(1)), int(m.group(2))
        except OSError:
            pass
        rd = pp.get("redo", {})
        n_done = len(rd)
        total = before + left0 if left0 is not None else None
        run_done = n_done - before
        per_horse = pl_["run_gets"] / run_done if run_done >= 25 else 5.0   # 打ち切った馬は 4 ページ以上（起動時の見積もりと同じ 5）
        left = max(total - n_done, 0) if total is not None else None
        cnt = {s: sum(1 for v in rd.values() if v == s) for s in ("saved", "empty", "error")}
        eta = _fmt_eta(now, left * per_horse / rate if left and rate > 0 else None)
        return [f"   打ち切った馬の取り直し（--redo-truncated）: {_bar(n_done, total)}  {n_done:,}/{f'{total:,}' if total is not None else '?'} 頭",
                f"   保存 {cnt['saved']:,} ・ 調教なし {cnt['empty']} ・ 失敗 {cnt['error']}  ／ 最後のログ: {_age(now, pl_['last_ts'])}",
                f"   1 頭あたり {per_horse:.1f} 件 ・ 速度 {rate:.0f} 件/時（直近 3 時間） → 残り {f'{left:,}' if left is not None else '?'} 頭の終了: {eta if left != 0 else '完了'}"]
    pc = pp.get("counts", {})
    n_done, total = len(pp.get("done", {})), pp.get("total")
    reqs = pp.get("requests", 0)
    per_horse = reqs / n_done if n_done >= 25 else TRAINING_PAST_PAGES_PER_HORSE
    left = max(total - n_done, 0) if total else None
    eta = _fmt_eta(now, left * per_horse / rate if left and rate > 0 else None)
    return [f"   進捗 {_bar(n_done, total)}  {n_done:,}/{f'{total:,}' if total else '?'} 頭",
            f"   保存 {pc.get('saved', 0):,} ・ 調教なし {pc.get('empty', 0)} ・ 失敗 {pc.get('error', 0)}"
            f"  ／ 最後のログ: {_age(now, pl_['last_ts'])}",
            f"   1 頭あたり {per_horse:.1f} 件 ・ 速度 {rate:.0f} 件/時（直近 3 時間） → 残り {f'{left:,}' if left is not None else '?'} 頭の終了: {'完了' if left == 0 else eta}"]


CACHE_PATH = Path.home() / ".cache" / "keiba_fetch_status.json"   # 表示のための控え（重い総数の数え直しを避ける・速度の標本）。ユーザーごと
RATE_WINDOW_SEC = 3 * 3600


def _now_ts() -> float:
    return time.time()


def _cache_load() -> dict:
    """控えを読む。形が崩れていれば捨てる（崩れたまま毎回例外になるのを防ぐ）。"""
    try:
        d = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(d, dict):
        return {}
    if not isinstance(d.get("course_position", {}), dict):
        d.pop("course_position")
    smp = d.get("samples", {})
    if not isinstance(smp, dict):
        d.pop("samples")
    else:
        d["samples"] = {k: v for k, v in smp.items()
                        if isinstance(v, list) and all(isinstance(s, list) and len(s) == 2 and all(isinstance(x, (int, float)) for x in s) for s in v)}
    return d


def _cache_save(d: dict, now_ts: float) -> None:
    # 窓より古い標本しか無い鍵（終わった取得の分）は捨てる（控えが増え続けないように）
    d["samples"] = {k: v for k, v in d.get("samples", {}).items() if v and now_ts - v[-1][0] <= RATE_WINDOW_SEC}
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_PATH.with_name(CACHE_PATH.name + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, CACHE_PATH)
    except OSError:
        pass


def _rate_from_samples(cache: dict, key: str, done: float, now_ts: float) -> tuple[float | None, float]:
    """表示のたびに (時刻, 済みの数) を控え、直近 3 時間の増え方から 1 時間あたりの速度を出す（心拍の無い取得の終了見込み用）。
    戻り値は (速度、標本の長さ（秒）)。速度が None で長さが 120 秒以上なら「増えていない」。"""
    samples = [s for s in cache.get("samples", {}).get(key, []) if now_ts - s[0] <= RATE_WINDOW_SEC and s[1] <= done]
    if not samples or now_ts - samples[-1][0] >= 60:
        samples.append([now_ts, done])
    cache.setdefault("samples", {})[key] = samples[-200:]
    t0, d0 = samples[0]
    span = now_ts - t0
    return ((done - d0) / (span / 3600) if span >= 120 and done > d0 else None), span


def _course_position_detail(task: dict, now: datetime) -> list[str]:
    """柵の位置（course_position）: 対象の開催日のうち、保存済みの日の数。
    総数は取得の本体と同じ数え方（race_lists、--include-results ならレース結果の表の日付も）。全部読むと十数秒かかるので、
    同じ引数・同じ日のうちは控えを使う（その日のうちに race_lists が増えても総数は翌日まで変わらない）。"""
    from datetime import date as _date

    from src.scraper import course_position as cp

    args = task["main"]["args"]
    m0, m1 = re.search(r"--from[=\s]+(\d{4})", args), re.search(r"--to[=\s]+(\d{4})", args)
    today = _date.today().strftime("%Y%m%d")          # 本体（course_position.main）と同じく、この機械の日付
    y0, y1 = int(m0.group(1)) if m0 else 2016, int(m1.group(1)) if m1 else int(today[:4])
    incl = "--include-results" in args
    cache = _cache_load()
    key = f"cp:{y0}:{y1}:{today}:{int(incl)}"
    dates = cache.get("course_position", {}).get(key)
    if not isinstance(dates, list):
        dates = cp.race_dates(y0, y1, before=today)
        if incl and hasattr(cp, "result_dates"):
            dates = sorted(set(dates) | set(cp.result_dates(y0, y1, before=today)))
        cache["course_position"] = {key: dates}
    stores = {y: cp.load_store(y)["dates"] for y in range(y0, y1 + 1)}
    done = sum(1 for d in dates if f"{d[:4]}-{d[4:6]}-{d[6:8]}" in stores.get(int(d[:4]), {}))
    total = len(dates)
    now_ts = _now_ts()
    rate, span = _rate_from_samples(cache, f"cp:{task['main']['pid']}", done, now_ts)
    _cache_save(cache, now_ts)
    if done >= total:
        speed, eta = "—", "完了"
    elif rate:
        speed, eta = f"{rate:.0f} 日/時", _fmt_eta(now, (total - done) / rate)
    elif span >= 120:
        speed, eta = f"直近 {span / 60:.0f} 分は増えていない（他の取得を待っているか、止まっている）", "—"
    else:
        speed, eta = "—（測り始め。数分後に出る）", "—"
    return [f"   進捗 {_bar(done, total)}  {done:,}/{total:,} 開催日（{y0}〜{y1}・20 日ごとに保存）",
            f"   速度 {speed} → 終了: {eta}"]


def _detail(task: dict, now: datetime) -> tuple[list[str], bool]:
    """既知の取得の詳しい行（実行中のときだけ）。2 つ目は「待機中」とみなすか。"""
    if task["state"] != "running":
        return [], False
    args, mod = task["main"]["args"], task["module"]
    if mod == "src.scraper.course_position":
        try:
            return _course_position_detail(task, now), False
        except Exception as e:  # noqa: BLE001 — 表示の失敗で一覧全体を止めない
            return [f"   進捗 —（読めない: {e}）"], False
    if mod == "src.scraper.backfill" and "--phase past" in args:
        return _past_detail(now)
    if mod == "src.scripts.scraping.backfill_horse_training_recent":
        return _training_detail(now), False
    if mod == "src.scripts.scraping.backfill_horse_training_past":
        return _training_past_detail(now, "--redo-truncated" in args), False
    return [], False


def render(now: datetime | None = None, tasks: list[dict] | None = None) -> str:
    now = now or datetime.now(JST).replace(tzinfo=None)
    now_ts = time.time()
    tasks = collect_tasks() if tasks is None else tasks
    blocks = []
    for t in tasks:
        detail, yielding = _detail(t, now)
        if t["state"] == "running" and not detail and not (t["entry"] or {}).get("total"):
            detail = ["   進捗 —（総数の分からない取得。docs/operations/fetch_tasks.md）"]
        if yielding:
            t["state"], t["waiting_for"] = "waiting", t["waiting_for"] or "他の取得に譲って待機中（ログより）"
        state = "実行中" if t["state"] == "running" else "待機中"
        e = t["entry"]
        if t["state"] == "running" and e and e.get("last_get_at") and now_ts - e["last_get_at"] > STALE_SEC:
            state += f"（{int((now_ts - e['last_get_at']) // 60)} 分 GET なし。止まっている?）"
        blocks.append((t, [f"■ [{state}] {t['label']}"] + _generic_lines(t, now_ts, now) + detail))
    n_run = sum(1 for t, _ in blocks if t["state"] == "running")
    L: list[str] = []
    bar = "━" * 72
    L += [bar, f" 取得の状況  {now:%Y-%m-%d %H:%M:%S}  実行中 {n_run} ・ 待機中 {len(blocks) - n_run}（終わったものは出さない・読み取りのみ）", bar]

    # 止める印（完了ではなく「止められている」状態なので、取得が無くても出す）
    halt, pause = bg.halted(), None
    try:
        from src.scraper.scrape_access_pause import read_access_pause

        pause = read_access_pause() or {}
    except Exception:  # noqa: BLE001
        pass
    if halt:
        L.append(f" 🚨 補填の中断印あり: {halt.get('reason') or halt.get('status_counts')}（確認後 {bg.HALT_PATH} を削除）")
    if pause and pause.get("active"):
        L.append(f" 🚨 アクセス保留が有効: {pause.get('reason')}（{pause.get('paused_at')}）")

    if not blocks:
        L += ["", " 実行中・待機中の取得はありません"]
    for _, lines in blocks:
        L += [""] + lines
    L.append(bar)
    return "\n".join(L)


def main() -> int:
    print(render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
