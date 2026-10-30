"""取得の状況と、すべて終わる見込みを 1 画面に出す（読み取りだけ。netkeiba・GCS へはアクセスしない）。

    watch -n 5 python -m src.scripts.scraping.fetch_status

対象:
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
import re
import sys
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.scraper import backfill_guard as bg
from src.scraper.backfill import PAST_TIERS, PAST_YEARS

ROOT = Path(__file__).resolve().parents[3]
PROGRESS = ROOT / "data/local/meta/backfill_progress.json"
TRAINING_PROGRESS = ROOT / "data/local/meta/horse_training_backfill_progress.json"
PAST_LOG = ROOT / "logs/backfill_past.log"
TRAINING_LOG = ROOT / "logs/horse_training_backfill.log"

JST = timezone(timedelta(hours=9))     # ログの時刻は JST（OS の時刻は UTC のことがある）
ETA_WINDOWS_H = (3,)                   # 見込みに使う速度の窓（時間）。譲り・中断・夜の切れ目も含めた実効の平均
SHOW_WINDOWS_H = (1, 3, 24)            # 速度の表示に使う窓（時間）
MIN_TOUCHED_HORSES = 30                # この頭数に満たない実行では、1 頭あたりの件数を全体の実績で求める
REQ_PER_RACE = {"cards": 2.0}          # 実績が無い段階の 1 レースあたりの件数（§4）。ここに無い段階は 1
REQ_PER_HORSE_DEFAULT = 4.0            # 実績が無いときの 1 頭あたりの件数（§4）
RACES_PER_DAY_DEFAULT = 32.0
NEW_HORSE_DECAY = 0.5                  # 未取得の開催日の新しい馬の数 ＝ 取得済みの開催日あたりの頭数 × この係数（概算）
TRAINING_REQ_PER_HORSE = 1.0
STAGE_LABEL = {"fast": "結果", "horse": "馬・血統", "cards": "出馬表・追い切り", "lap": "ラップ", "odds": "オッズ", "pair": "2連系オッズ",
               "index": "指数", "past": "馬柱", "on_time": "速報", "anchors_f": "父方アンカー"}
REQUIRED_FOR_TRAINING = ("fast", "horse", "cards", "lap")   # 学習に必須の段階（§4）

_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[\w+\] (.*)$")
_RUN_START = re.compile(r"^===== (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\S* START")   # 起動スクリプトが書く実行の区切り（JST）
_GET = re.compile(r"scraper\.client: GET https://([a-z.]*netkeiba\.com)/(\w+)")
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
        if not g:
            continue
        kind = f"{g.group(1).split('.')[0]}/{g.group(2)}"
        out["kinds"][kind] = out["kinds"].get(kind, 0) + 1
        out["run_gets"] += 1
        h = _HORSE_GET.search(line)
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


def _age(now: datetime, ts: datetime | None) -> str:
    if not ts:
        return "—"
    s = int((now - ts).total_seconds())
    return f"{s} 秒前" if s < 120 else f"{s // 60} 分前" if s < 7200 else f"{s // 3600} 時間前"


def _procs() -> dict[str, list[str]]:
    import subprocess

    out = subprocess.run(["ps", "-eo", "pid,etime,args"], capture_output=True, text=True, timeout=5).stdout.splitlines()[1:]
    found: dict[str, list[str]] = {"past": [], "training": [], "other": []}
    for line in out:
        pid, etime, args = (line.strip().split(None, 2) + ["", ""])[:3]
        if "fetch_status" in args or "grep" in args or "--report" in args or "python" not in args:
            continue
        key = "past" if "--phase past" in args else "training" if "backfill_horse_training_recent" in args else \
            "other" if re.search(bg.OTHER_FETCH_PATTERN, args) else None
        if key:
            found[key].append(f"PID {pid} 経過 {etime}  {args.split(' -m ', 1)[-1][:90]}")
    return found


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


def render(now: datetime | None = None) -> str:
    now = now or datetime.now(JST).replace(tzinfo=None)
    prog = _load_json(PROGRESS)
    past = scan_log(PAST_LOG, now)
    procs = _procs()
    L: list[str] = []
    bar = "━" * 72
    L += [bar, f" 取得の状況  {now:%Y-%m-%d %H:%M:%S}（読み取りのみ）", bar]

    # 止める印
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

    # 補填（2016〜2019）
    rates_show = {w: rate_per_hour(past, w, now) for w in SHOW_WINDOWS_H}
    state = "停止中（プロセスなし）"
    if procs["past"]:
        msg = past["last_msg"] or ""
        state = "他の取得に譲って待機中" if "待機中" in msg else "取得中"
        if past["last_ts"] and (now - past["last_ts"]).total_seconds() > 600 and "待機中" not in msg:
            state = "ログが止まっている（他の取得を待っているか、固まっている）"
    L += ["", f"■ 2016〜2019 年の補填（--phase past）: {state}"]
    for p in procs["past"]:
        L.append(f"   {p}")
    L.append(f"   最後のログ: {_age(now, past['last_ts'])}  {(past['last_msg'] or '')[:80]}")
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
    else:
        rates = {w: rate_per_hour(past, w, now) for w in ETA_WINDOWS_H}
        heads = [f"終了見込み（{w}時間平均）" for w in ETA_WINDOWS_H]
        L += ["", "   " + _ljust("段階", 26) + _rjust("進捗", 22) + _rjust("残り件数", 10) + "   " + "  ".join(_ljust(h, 34) for h in heads)]
        cum = 0.0
        milestone = None
        for row in rows:
            cum += row["left_req"]
            etas = ["完了" if row["left_req"] <= 0 else _fmt_eta(now, cum / rates[w] if rates[w] > 0 else None) for w in ETA_WINDOWS_H]
            pct = row["done"] / row["total"] * 100 if row["total"] else 0.0
            mark = "≈" if row["approx"] else ""
            prog_s = f"{row['done']:,}/{mark}{row['total']:,.0f}{row['unit']} {pct:3.0f}%" if row["tier"] != "anchors_f" else f"残り{mark}{row['total']:,.0f}頭"
            label = f"{row['tier']}（{STAGE_LABEL.get(row['tier'], '')}）"
            L.append("   " + _ljust(label, 26) + _rjust(prog_s, 22) + _rjust(f"{row['left_req']:,.0f}", 10) + "   " + "  ".join(_ljust(e, 34) for e in etas))
            if row.get("parts"):
                L += _horse_parts_lines(row, rows[-1] if rows[-1]["tier"] == "anchors_f" else None)
            if row["tier"] == REQUIRED_FOR_TRAINING[-1]:
                milestone = cum
        L.append("   " + _ljust("合計", 26) + _rjust("", 22) + _rjust(f"{cum:,.0f}", 10))
        for name, req in ((f"学習に必須の段階（{'・'.join(REQUIRED_FOR_TRAINING)}）", milestone or 0.0), ("すべての段階", cum)):
            etas = [f"{w}時間平均 {_fmt_eta(now, req / rates[w] if rates[w] > 0 else None) if req else '完了'}" for w in ETA_WINDOWS_H]
            L.append(f"   ▶ {name}: " + " ／ ".join(etas))
        L.append("   ≈ は概算（出走馬の総数は fast の完了後に確定。父方アンカーは既存分を除く前の上限側）")

    # 調教の補填
    tp = _load_json(TRAINING_PROGRESS)
    tr = scan_log(TRAINING_LOG, now, windows_h=(1,))
    run_h = (now - tr["run_start"]).total_seconds() / 3600 if tr["run_start"] else 0.0
    t1 = tr["run_gets"] / run_h if run_h > 0 else 0.0     # 今回の実行の開始から（起動直後の対象の集計の時間も含む）
    left = tp.get("remaining")
    t_state = "取得中" if procs["training"] else "停止中（プロセスなし）"
    L += ["", f"■ 調教の補填（backfill_horse_training_recent）: {t_state}"]
    for p in procs["training"]:
        L.append(f"   {p}")
    c = tp.get("counts", {})
    L.append(f"   累計 保存 {c.get('saved', 0):,} ・ 読めない {c.get('unreadable', 0)} ・ 空 {c.get('empty', 0)} ・ 失敗 {c.get('error', 0)}"
             f"  ／ 最後のログ: {_age(now, tr['last_ts'])}")
    if procs["training"]:
        L.append(f"   今回の残り {left if left is not None else '?'} 頭（25 頭ごとに更新）・ 今回の実行 {t1:.0f} 件/時"
                 f" → 今回の終了: {_fmt_eta(now, left * TRAINING_REQ_PER_HORSE / t1 if left and t1 > 0 else None)}")
    elif tp.get("aborted"):
        L.append(f"   前回の中断理由: {tp['aborted']}")
    elif tp.get("finished_at"):
        L.append(f"   前回の終了: {tp['finished_at']}")

    if procs["other"]:
        L += ["", "■ その他の取得（補填はこれらに譲る）"] + [f"   {p}" for p in procs["other"]]
    L.append(bar)
    return "\n".join(L)


def main() -> int:
    print(render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
