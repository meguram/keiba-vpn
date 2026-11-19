"""開催日に、レースごとの「AI 予測と推奨馬券が画面に出た（見込み）」を、stg・本番の Slack に 1 回ずつ送る。

計画とレビュー: docs/operations/slack_race_ready_plan_20261119.md（案 A）。

- 読むのはホストの手元のファイルだけ（GCS・HybridStorage・API には触れない。費用 0 円）:
  - レース一覧: data/page_reference/race_lists/<日>.json（無ければ非開催日としてすぐ終わる）
  - 発走時刻: data/local/meta/entries_state/<race_id>.json の start_time
  - 本体の推論と §77: data/local/inference/events/<日>.jsonl（レースごとの最後の status=ok の行の sec77）
  - 予測・判定を上げた時刻: data/local/meta/race_push_log/<日>.jsonl（src/ops/gcs_mirror.py の push が書く）
  - 第 2 段の判定と止めの目印: data/local/inference/bets_stage2/…（目印の解釈は src/api/stage2_live.py を使い回す）
- 「画面に出た」は推定: 予測と推奨の元がそろって（上げて）から lead_sec（240 秒 = 目録 60＋prerender 120＋画面 45 秒）後。
  目録の書き込みに失敗した回の push は、Cloud Run が一覧に戻る 30 分後まで見えないものとして扱う。
- 1 レース・1 環境に 1 回。発走後と、そろってから max_age_min（15 分）を過ぎたものは送らない（目印を後から置いても溜まった分が出ない）。
- 送るのは KEIBA_SLACK_NOTIFY=1 かつ目印 data/local/meta/race_ready_notify.on があるときだけ。無いとき（と --dry-run）は送らずにログに残し、
  そのレースは「見送り（dry-run）」として記録する（後から送らない）。
- 宛先は環境ごとの Webhook の変数（config/race_ready_notify.json の webhook_env）だけ。空ならその環境には送らない（汎用の宛先に落とさない）。
  Webhook の値はログ・記録に出さない。
- 遅れの警告: 発走 late_warn_min（10 分）前を過ぎても出ないレースを、1 回の実行で 1 通にまとめて、late=true の環境（stg）にだけ送る。

  python -m src.scripts.ops.race_ready_notify                         # cron（毎分。scripts/cron/race_ready_notify.sh）
  python -m src.scripts.ops.race_ready_notify --day 20261115 --replay # 過去の日の当て直し（送らない・記録しない。push の時刻はファイルの更新時刻＋5 分）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[3]
JST = timezone(timedelta(hours=9))
CONFIG = ROOT / "config" / "race_ready_notify.json"
MARK = "data/local/meta/race_ready_notify.on"
STATE_DIR = "data/local/meta/slack_sent"
PUSH_LOG = "data/local/meta/race_push_log"      # src/ops/gcs_mirror.py の RACE_PUSH_LOG と同じ
INF = "data/local/inference"
INDEX_FAIL_DELAY = timedelta(minutes=30)        # 目録の書き込みに失敗した回は、Cloud Run の全体の一覧（30 分ごと）まで見えない
REPLAY_PUSH_LAG = timedelta(minutes=5)          # 当て直し: ファイルの更新時刻＋5 分（push の周期）を上げた時刻の代わりにする
MAX_TRIES = 3

logger = logging.getLogger("ops.race_ready_notify")


# ───────────────────────── 読み込み（手元のファイルだけ） ─────────────────────────

def _jsonl(p: Path) -> list[dict]:
    out: list[dict] = []
    try:
        for ln in p.read_text(encoding="utf-8").splitlines():
            try:
                x = json.loads(ln)
            except ValueError:
                continue
            if isinstance(x, dict):
                out.append(x)
    except OSError:
        pass
    return out


def _json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _t(s: Any) -> datetime | None:
    try:
        d = datetime.fromisoformat(str(s))
    except (TypeError, ValueError):
        return None
    return (d if d.tzinfo else d.replace(tzinfo=JST)).astimezone(JST)


def load_config(path: Path = CONFIG) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class Race:
    race_id: str
    venue: str = ""
    round: int | None = None
    name: str = ""
    post: datetime | None = None
    pushed: dict[str, datetime] = field(default_factory=dict)   # 木 → 画面で見えうる最早の時刻（上げた時刻。目録の失敗は +30 分）
    index_failed: bool = False
    main_ok_at: datetime | None = None      # 本体の推論の最後の status=ok の時刻
    sec77: str | None = None                # その行の sec77（ok なら推奨 target の元がある）
    decision: dict | None = None            # 第 2 段の判定ファイル

    @property
    def maiden(self) -> bool:
        return "未勝利" in self.name or "新馬" in self.name


def collect(day: str, root: Path, *, replay: bool = False) -> list[Race]:
    """その日のレースの状態を手元のファイルから集める。レース一覧が無ければ空。"""
    rl = _json(root / "data/page_reference/race_lists" / f"{day}.json")
    if not isinstance(rl, dict) or not rl.get("races"):
        return []
    iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    races: dict[str, Race] = {}
    for r in rl["races"]:
        rid = str(r.get("race_id") or "")
        if len(rid) != 12:
            continue
        race = Race(rid, str(r.get("venue") or ""), r.get("round"), str(r.get("race_name") or ""))
        es = _json(root / "data/local/meta/entries_state" / f"{rid}.json")
        if isinstance(es, dict) and str(es.get("date") or iso) == iso:
            try:
                hh, mm = str(es.get("start_time") or "").split(":")[:2]
                race.post = datetime(int(day[:4]), int(day[4:6]), int(day[6:]), int(hh), int(mm), tzinfo=JST)
            except ValueError:
                pass
        races[rid] = race
    inf = root / INF
    for ev in _jsonl(inf / "events" / f"{day}.jsonl"):          # 後の行ほど新しい（推論のやり直しは最後の ok を使う）
        race = races.get(str(ev.get("race_id") or ""))
        if race is not None and ev.get("status") == "ok" and ev.get("v6_status") in (None, "ok"):
            race.main_ok_at = _t(ev.get("at")) or race.main_ok_at
            race.sec77 = ev.get("sec77")
    if replay:
        for race in races.values():
            for tree, ext in (("predictions_v6", "parquet"), ("predictions_v6c", "parquet"), ("bets_stage2", "json")):
                p = inf / tree / day / f"{race.race_id}.{ext}"
                if p.exists():
                    race.pushed[tree] = datetime.fromtimestamp(p.stat().st_mtime, JST) + REPLAY_PUSH_LAG
    else:
        rows = _jsonl(root / PUSH_LOG / f"{day}.jsonl")
        index_ok_at = sorted(t for t in (_t(r.get("at")) for r in rows if r.get("tree") == "_index" and r.get("index_ok") is True) if t)
        for row in rows:
            at = _t(row.get("at"))
            tree = str(row.get("tree") or "")
            if at is None or not tree or tree == "_index":
                continue
            seen = at
            if row.get("index_ok") is False:    # 目録の失敗: 次に目録を書けた回か、Cloud Run の全体の一覧（30 分）の早いほう
                seen = min([t for t in index_ok_at if t > at][:1] + [at + INDEX_FAIL_DELAY])
            for rid in row.get("race_ids") or []:
                race = races.get(str(rid))
                if race is None or tree in race.pushed:             # 最初に上げた時刻だけ使う（書き直しは表示の変化にしない）
                    continue
                race.pushed[tree] = seen
                race.index_failed = race.index_failed or row.get("index_ok") is False
    for race in races.values():
        if "bets_stage2" in race.pushed:
            d = _json(inf / "bets_stage2" / day / f"{race.race_id}.json")
            race.decision = d if isinstance(d, dict) else None
    return list(races.values())


# ───────────────────────── 環境ごとの判定 ─────────────────────────

@dataclass
class EnvView:
    """その環境の画面の決まり（stage2_live と同じ解釈）。"""
    stage2_on: bool        # 第 2 段の判定を読む（機能が入っていて、開始日を過ぎ、全体の止めが無い）
    v6c_on: bool           # 新しい候補の予測を出す
    dev_only: bool         # 第 2 段は開発者だけ（会員は今の推奨）


def env_view(day: str, target: dict, inf: Path) -> EnvView:
    from src.api import stage2_live as S

    feats = target.get("features") or {}
    start = str(target.get("stage2_start") or S.start_day())
    on = bool(feats.get("stage2", True)) and day >= start and not S.disabled(inf)
    return EnvView(stage2_on=on, v6c_on=on and not S.v6c_disabled(inf), dev_only=S.dev_only(inf))


@dataclass
class Ready:
    at: datetime            # 画面に出た見込みの時刻
    pred: str               # 予測の中身
    rec: str                # 推奨馬券の中身（会員に出るもの）
    stage2: str | None      # 第 2 段の行（未勝利・新馬で第 2 段を読む環境のときだけ）


def _decision_text(d: dict | None) -> str:
    if not d:
        return "判定待ち"
    if d.get("status") == "buy":
        hm = (d.get("honmei") or {}).get("horse_number")
        n = len(d.get("bets") or [])
        return f"単勝 ◎{hm} {n} 点" if hm is not None else f"買い {n} 点"
    return f"見送り（{d.get('skip_label') or d.get('skip_reason') or '規則'}）"


def judge(race: Race, view: EnvView, now: datetime, lead: timedelta) -> tuple[Ready | None, str]:
    """(そろっていれば Ready, そろっていなければ止まっている段の説明)。時刻はすべて now 以前の事象だけで決める。"""
    vis = {k: v for k, v in race.pushed.items() if v <= now}
    # 予測: 本体（v6e）が先に出る。新しい候補（v6c）は来れば置き換わる
    v6c_on = view.v6c_on and race.maiden          # 新しい候補の予測は未勝利・新馬だけ（local_predictions の in_scope）
    pred_ts = [t for k, t in vis.items() if k == "predictions_v6" or (k == "predictions_v6c" and v6c_on)]
    if not pred_ts:
        if race.main_ok_at is None:
            return None, "推論なし"
        return None, "予測の push 待ち" + ("（目録の書き込みに失敗）" if race.index_failed else "")
    pred_at = min(pred_ts)
    pred = "新しい候補（試行中）" if v6c_on and "predictions_v6c" in vis else "v6e"
    # 推奨: §77 があれば今の推奨（target）。第 2 段が会員に出る環境では、判定が上がればそれに置き換わる
    rec_ts: list[tuple[datetime, str]] = []
    if race.sec77 == "ok" and race.main_ok_at is not None and race.main_ok_at <= now:
        rec_ts.append((max(race.main_ok_at, pred_at), "今の推奨"))
    member_stage2 = view.stage2_on and not view.dev_only and race.maiden and "bets_stage2" in vis and (race.decision or {}).get("strategy") == "stage2_r2"
    if member_stage2:
        rec_ts.append((vis["bets_stage2"], "第 2 段（試行中）"))
    if not rec_ts:
        return None, f"推奨の元なし（§77: {race.sec77 or '未'}）"
    rec_at = min(t for t, _ in rec_ts)
    rec = "第 2 段（試行中）" if member_stage2 else "今の推奨"
    ready_at = max(pred_at, rec_at) + lead
    if ready_at > now:
        return None, "画面への反映待ち"
    s2 = None
    if view.stage2_on and race.maiden:
        if "bets_stage2" in vis and race.decision:
            who = "開発者だけ" if view.dev_only else "会員に出た"
            s2 = f"{who}・{_decision_text(race.decision)}"
            at = _t(race.decision.get("as_of"))
            if at:
                s2 += f"（判定 {at:%H:%M}）"
        else:
            s2 = "判定待ち"
    return Ready(ready_at, pred, rec, s2), ""


# ───────────────────────── 文面 ─────────────────────────

def race_title(race: Race) -> str:
    r = f"{race.round}R " if race.round else ""
    return f"{race.venue} {r}{race.name}".strip()


def ready_payload(race: Race, ready: Ready, base_url: str) -> dict:
    from src.utils import slack_notify as SN

    link = f"{base_url.rstrip('/')}/race/{race.race_id}"
    title = f"予測・推奨が画面に出ました（見込み {ready.at:%H:%M}）"
    head = f"*{SN.esc(race_title(race))}* ・ 発走 {race.post:%H:%M} ・ race_id {race.race_id}" if race.post else \
        f"*{SN.esc(race_title(race))}* ・ race_id {race.race_id}"
    lines = [f"AI 予測: 出た（{ready.pred}）", f"推奨馬券: 出た（{ready.rec}）"]
    if ready.stage2:
        lines.append(f"第 2 段（試行中）: {ready.stage2}")
    blocks = [SN.section(f"*{SN.esc(title)}*\n{head}"), SN.section("\n".join("• " + SN.esc(x) for x in lines)),
              SN.context([f"<{link}|レースを開く>"])]
    return SN.build_message(title, blocks, severity="normal", color=SN.SEVERITY_COLOR["good"],
                            fallback=f"{title} {race_title(race)}")


def late_payload(items: list[tuple[Race, str]], base_url: str, warn_min: int) -> dict:
    from src.utils import slack_notify as SN

    title = f"発走 {warn_min} 分前を過ぎても、予測・推奨が出ていません（{len(items)} レース）"
    rows = []
    for race, why in sorted(items, key=lambda x: (x[0].post or datetime.max.replace(tzinfo=JST), x[0].race_id)):
        post = f"{race.post:%H:%M}" if race.post else "—"
        rows.append(f"• {post} <{base_url.rstrip('/')}/race/{race.race_id}|{SN.esc(race_title(race))}> ・ {SN.esc(why)}")
    return SN.build_message(title, [SN.section(f"*[注意] {SN.esc(title)}*"), SN.section("\n".join(rows))],
                            severity="warning", fallback=f"[注意] {title}")


# ───────────────────────── 送信済みの記録 ─────────────────────────

def state_path(root: Path, day: str) -> Path:
    return root / STATE_DIR / f"race_ready_{day}.json"


def load_state(p: Path) -> dict:
    d = _json(p)
    return d if isinstance(d, dict) else {}


def save_state(p: Path, state: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)


def _done(rec: dict | None) -> bool:
    return bool(rec) and (rec.get("st") != "failed" or int(rec.get("n") or 0) >= MAX_TRIES)


# ───────────────────────── 1 回の実行 ─────────────────────────

def webhook_env(target: dict, environ: dict[str, str]) -> dict[str, str] | None:
    """送信に渡す最小の環境（その環境の Webhook と KEIBA_ENV だけ）。Webhook が空なら None（汎用の宛先に落とさない）。"""
    url = (environ.get(str(target.get("webhook_env") or "")) or "").strip()
    if not url:
        return None
    ke = str(target.get("keiba_env") or "").strip().lower()
    return {"KEIBA_ENV": ke, f"SLACK_WEBHOOK_URL_{ke.upper()}": url, "KEIBA_RUN_LOCATION": "host"}


def run_once(day: str, now: datetime, *, root: Path = ROOT, cfg: dict | None = None, send: bool = False,
             state: dict | None = None, environ: dict[str, str] | None = None, replay: bool = False,
             post: Callable[..., Any] | None = None, sleep: Callable[[float], None] = time.sleep,
             races: list[Race] | None = None, save: Callable[[dict], None] | None = None) -> dict:
    """1 回分。戻り値: {"sent": [...], "dryrun": [...], "late": [...], "skipped_old": n, "failed": n}。state は呼び出し側が保存する。"""
    from src.utils import slack_notify as SN

    cfg = cfg if cfg is not None else load_config()
    environ = dict(os.environ) if environ is None else environ
    state = {} if state is None else state
    races = collect(day, root, replay=replay) if races is None else races
    out: dict[str, Any] = {"sent": [], "dryrun": [], "late": [], "skipped_old": 0, "failed": 0}
    if not races:
        return out
    lead = timedelta(seconds=int(cfg.get("lead_sec", 240)))
    max_age = timedelta(minutes=int(cfg.get("max_age_min", 15)))
    warn = timedelta(minutes=int(cfg.get("late_warn_min", 10)))
    budget = int(cfg.get("max_per_run", 20))
    gap = float(cfg.get("send_gap_sec", 1.2))
    stamp = now.isoformat(timespec="seconds")
    sends = 0

    def deliver(payload: dict, env: dict[str, str] | None) -> tuple[bool, str | None]:
        nonlocal sends
        if not send or env is None:
            return False, None
        if sends:
            sleep(gap)
        sends += 1
        r = SN.send_payload(payload, channel="default", env=env, retries=1, post=post)
        return bool(r.get("ok")), (None if r.get("ok") else SN.redact(str(r.get("error")), env))

    for name, target in (cfg.get("targets") or {}).items():
        view = env_view(day, target, root / INF)
        env = webhook_env(target, environ)
        late_items: list[tuple[Race, str]] = []
        for race in sorted(races, key=lambda r: (r.post or datetime.max.replace(tzinfo=JST), r.race_id)):
            if race.post is not None and now >= race.post:
                continue                                            # 発走後は送らない
            key = f"{name}|{race.race_id}|ready"
            if _done(state.get(key)):
                continue
            ready, why = judge(race, view, now, lead)
            if ready is None:
                if target.get("late") and race.post is not None and now >= race.post - warn:
                    lk = f"{name}|{race.race_id}|late"
                    if not _done(state.get(lk)):
                        late_items.append((race, why))
                continue
            if now - ready.at > max_age:
                state[key] = {"st": "skipped_old", "at": stamp}
                out["skipped_old"] += 1
                continue
            payload = ready_payload(race, ready, str(target["base_url"]))
            if not send or env is None:     # 送らない回も「見送り（dry-run）」として記録し、後から送らない
                state[key] = {"st": "skipped_dryrun", "at": stamp, "ready_at": ready.at.isoformat(timespec="seconds")}
                out["dryrun"].append({"env": name, "race_id": race.race_id, "ready_at": ready.at, "at": now,
                                      "text": SN.render_preview_text(SN.with_env_label(payload, {"KEIBA_ENV": str(target.get("keiba_env") or "")}))})
                continue
            if sends >= budget:
                break
            ok, err = deliver(payload, env)
            prev = int((state.get(key) or {}).get("n") or 0)
            if ok:
                state[key] = {"st": "sent", "at": stamp, "ready_at": ready.at.isoformat(timespec="seconds")}
                out["sent"].append({"env": name, "race_id": race.race_id})
            else:
                state[key] = {"st": "failed", "n": prev + 1, "at": stamp, "error": err}
                out["failed"] += 1
            if save:                    # 送るたびに保存する（送った後に落ちても、重ねて送るのは最大 1 通）
                save(state)
        if late_items:
            payload = late_payload(late_items, str(target["base_url"]), int(warn.total_seconds() // 60))
            ok = False
            if send and env is not None and sends >= budget:
                continue                # 上限で送れなかった回は記録に触れず、次の回に回す
            if send and env is not None:
                ok, _ = deliver(payload, env)
            for race, why in late_items:
                lk = f"{name}|{race.race_id}|late"
                if ok or not send or env is None:
                    state[lk] = {"st": "sent" if ok else "skipped_dryrun", "at": stamp, "why": why}
                else:
                    state[lk] = {"st": "failed", "n": int((state.get(lk) or {}).get("n") or 0) + 1, "at": stamp}
                out["late"].append({"env": name, "race_id": race.race_id, "why": why, "sent": ok})
            if save:
                save(state)
    return out


def sending_enabled(root: Path, environ: dict[str, str]) -> bool:
    return environ.get("KEIBA_SLACK_NOTIFY") == "1" and (root / MARK).exists()


def replay_day(day: str, *, root: Path = ROOT, cfg: dict | None = None, start: str = "07:00", end: str = "17:59") -> list[dict]:
    """過去の日を 1 分きざみで当て直す（送らない・記録しない）。返り値は送るはずだった通知の一覧。"""
    cfg = cfg if cfg is not None else load_config()
    races = collect(day, root, replay=True)
    y, m, d = int(day[:4]), int(day[4:6]), int(day[6:])
    t = datetime(y, m, d, *map(int, start.split(":")), tzinfo=JST)
    stop = datetime(y, m, d, *map(int, end.split(":")), tzinfo=JST)
    state: dict = {}
    got: list[dict] = []
    while t <= stop:
        r = run_once(day, t, root=root, cfg=cfg, send=False, state=state, environ={}, replay=True, races=races)
        got += [dict(x, kind="ready") for x in r["dryrun"]] + [dict(x, kind="late", at=t) for x in r["late"]]
        t += timedelta(minutes=1)
    return got


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--day", default="", help="YYYYMMDD（既定は JST の今日）")
    ap.add_argument("--dry-run", action="store_true", help="送らない（記録は「見送り（dry-run）」で残す）")
    ap.add_argument("--replay", action="store_true", help="過去の日を 1 分きざみで当て直す（送らない・記録しない）")
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = Path(a.root)
    now = datetime.now(JST)
    day = a.day or now.strftime("%Y%m%d")
    if not (root / "data/page_reference/race_lists" / f"{day}.json").exists():
        return 0                                                    # 非開催日（重い処理の前に終える）
    if a.replay:
        for x in replay_day(day, root=root):
            if x["kind"] == "ready":
                print(f"{x['at']:%H:%M} {x['env']:<4} {x['race_id']} 見込み {x['ready_at']:%H:%M}")
                print(x["text"])
            else:
                print(f"{x['at']:%H:%M} {x['env']:<4} {x['race_id']} 遅れの警告: {x['why']}")
        return 0
    environ = dict(os.environ)
    send = sending_enabled(root, environ) and not a.dry_run
    sp = state_path(root, day)
    sp.parent.mkdir(parents=True, exist_ok=True)
    import fcntl

    try:
        lock = open(sp.parent / ".race_ready_notify.lock", "w")     # 手で流したときも cron と同時に記録を読み書きしない
    except OSError:
        logger.warning("ロックファイルを開けない（記録は root の持ち物。sudo で流す）")
        return 1
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        logger.info("ほかの実行が動いているので終わる")
        return 0
    state = load_state(sp)
    before = json.dumps(state, sort_keys=True)
    # --dry-run（手で確かめる）は記録を書かない。目印が無いだけの cron の回は「見送り（dry-run）」を記録し、後から送らない
    def saver(st: dict) -> None:
        try:
            save_state(sp, st)
        except OSError:          # 保存に失敗しても、その回の残り（もう一方の環境を含む）は続ける。次の分に重ねて送りうるのでログに残す
            logger.warning("送信済みの記録を保存できませんでした: %s", sp, exc_info=True)

    if a.dry_run:
        saver = None  # type: ignore[assignment]
    r = run_once(day, now, root=root, send=send, state=state, environ=environ, save=saver)
    if saver and json.dumps(state, sort_keys=True) != before:
        save_state(sp, state)
    for x in r["dryrun"]:
        logger.info("送らずに記録（dry-run）: %s %s 見込み %s\n%s", x["env"], x["race_id"], f"{x['ready_at']:%H:%M}", x["text"])
    for x in r["sent"]:
        logger.info("送った: %s %s", x["env"], x["race_id"])
    for x in r["late"]:
        logger.info("遅れの警告%s: %s %s %s", "（送った）" if x["sent"] else "", x["env"], x["race_id"], x["why"])
    if r["failed"] or r["skipped_old"]:
        logger.info("送れなかった %d 件・古くて見送り %d 件", r["failed"], r["skipped_old"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
