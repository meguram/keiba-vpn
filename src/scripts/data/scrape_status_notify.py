"""スクレイピング（定期取得・開催日ランナー・前日取得・夜間・バックフィル）の完了ステータスを Slack に通知する。

3 種類（--mode）:
  alerts    異常の即時通知。期待した時刻に始まらない/止まった/連続失敗/スキーマ違反の急増/取得の停止（403/429 の連続など）/調教被覆の低下。
            同じ異常は復旧するまで REPEAT_MIN 分おきに最大 MAX_REPEATS 回まで。復旧時に『復旧しました』を 1 回。
  summary   日次のサマリー（信号: 緑/黄/赤、タスク別の表、取得/失敗/スキーマ違反/欠損、次回予定）。開催日は夜、非開催日は朝に cron から。
  coverage  週次のカバレッジ要約（data_health の欠損、調教などの運用必須データの被覆）。

材料（読み取りのみ）: cron_status.build_status（crontab + cron_run.sh の実行記録）、/api/scrape-jobs、data/local/meta/{schema_violations,
auto_scrape_status.json,data_health/<env>/latest.json,drift_monitor/latest.json}。DEC-023 のキュー待ちは異常にしない（yield するジョブは未開始でも異常にしない）。
``--source prod``（GCP Jobs から）: ホストの crontab の代わりに ``config/gcp_job_map.json`` の GCP/VPS のジョブを、GCS の実行記録
（GCP は ``ops/job_runs/<job>/``、VPS は ``ops/fetch_tasks/{done,failed}/``。並走中 ``KEIBA_FETCH_SHADOW_ENABLED=1`` は ``ops/fetch_tasks_shadow/``）で見る。
キュー・スキーマ違反・data_health はホストのローカルにしか無いので prod では扱わない。通知の抑制状態はローカルのファイル（Jobs は state_in/state_out で持ち回る）。

  python -m src.scripts.data.scrape_status_notify --mode summary --dry-run     # 既定は dry-run（JSON と HTML プレビュー）
  python -m src.scripts.data.scrape_status_notify --mode alerts --send         # 実送信（KEIBA_SLACK_NOTIFY=1 と Webhook が必要）
終了コード: 0=完了, 13=無効, 14=送信失敗, 1=その他。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from src.utils import slack_notify as SN

REPO_ROOT = Path(__file__).resolve().parents[3]
JST = timezone(timedelta(hours=9))
META = REPO_ROOT / "data" / "local" / "meta"

REPEAT_MIN = 60          # 同じ異常の再通知の間隔（分）
MAX_REPEATS = 3          # 最初の通知を除く再通知の最大回数
FAIL_WARN = 2            # 連続失敗がこの回数で『注意』
FAIL_CRIT = 3            # 連続失敗がこの回数で『緊急』
OVERDUE_GRACE_MIN = 20   # 予定時刻からこの分数を過ぎても始まらなければ未開始とみなす
OVERDUE_CRIT_MIN = 180   # 未開始がこの分数を超えたら緊急
STUCK_HOURS = 6          # 実行中がこの時間を超えたら停止の疑い
SCHEMA_SPIKE_MIN = 20    # スキーマ違反の急増: 直近 24 時間の件数がこの値以上で、過去の中央値の 3 倍以上
BLOCK_MIN = 5            # 403/429 の失敗がこの件数以上で取得停止の兆候
COVERAGE_FLOOR = 0.90    # 調教の被覆の下限

# 期待する開催日条件: raceday（その日が開催日）/ eve（翌日が開催日）/ always
EXPECT = (("raceday-eve", "eve"), ("raceday-runner", "raceday"), ("raceday-result-runner", "raceday"), ("raceday-evening", "raceday"),
          ("raceday-comment", "raceday"), ("raceday-weight-runner", "raceday"))


def expectation_of(job_id: str) -> str:
    for prefix, kind in EXPECT:
        if job_id.startswith(prefix):
            return kind
    return "always"


# ───────────────────────── cron 式（直前の予定時刻） ─────────────────────────

def prev_fire(expr: str, before: datetime, horizon_days: int = 8) -> datetime | None:
    """before 以前で最後に cron 式に合った時刻（before と同じタイムゾーン）。"""
    from src.monitor.cron_status import parse_expr

    mi, ho, dom, mon, dows, dom_star, dow_star = parse_expr(expr)
    t = before.replace(second=0, microsecond=0)
    end = t - timedelta(days=horizon_days)
    while t > end:
        if t.month in mon and t.hour in ho and t.minute in mi:
            cdow = (t.weekday() + 1) % 7
            if dom_star and dow_star:
                ok = True
            elif dom_star:
                ok = cdow in dows
            elif dow_star:
                ok = t.day in dom
            else:
                ok = t.day in dom or cdow in dows
            if ok:
                return t
        t -= timedelta(minutes=1)
    return None


# ───────────────────────── 収集（すべて読み取り。失敗は None/空で返す） ─────────────────────────

def read_runs(job_id: str, runs_dir: Path, limit: int = 30) -> list[tuple[float, float, int]]:
    """cron_run.sh の記録 <job>.tsv（開始epoch 終了epoch 終了コード pid）の直近 limit 行。"""
    try:
        lines = (runs_dir / f"{job_id}.tsv").read_text(encoding="utf-8").strip().splitlines()[-limit:]
    except OSError:
        return []
    out = []
    for ln in lines:
        p = ln.split("\t")
        try:
            out.append((float(p[0]), float(p[1]), int(p[2])))
        except (IndexError, ValueError):
            continue
    return out


def consecutive_failures(runs: list[tuple[float, float, int]]) -> int:
    n = 0
    for _s, _e, ec in reversed(runs):
        if ec == 0:
            break
        n += 1
    return n


def collect_cron(now: datetime, *, status_fn: Callable[..., dict] | None = None, runs_dir: Path | None = None,
                 raceday: Callable[[str], bool] | None = None) -> list[dict]:
    """全 cron ジョブの状態。各ジョブ: id/name/group/state/last/next/consecutive_failures/overdue_min/cause。"""
    from src.monitor import cron_status as CSM

    runs_dir = runs_dir or CSM.RUNS_DIR
    st = status_fn() if status_fn else CSM.build_status(now=now)
    cnow = now.astimezone(CSM.cron_timezone())          # cron 式はサーバ（ホストは UTC）の時刻で評価する
    jobs: list[dict] = []
    for g in st.get("groups", []):
        for j in g.get("jobs", []):
            runs = read_runs(j["id"], runs_dir)
            last = j.get("last_run") or {}
            exp = expectation_of(j["id"])
            overdue_min = None
            if j.get("state") != "running" and not j.get("yields") and j.get("cron_exprs"):
                fires = [f for f in (prev_fire(e, cnow) for e in j["cron_exprs"]) if f]
                if fires:
                    pf = max(fires)
                    day_ok = True
                    if raceday is not None and exp != "always":
                        target = pf.astimezone(JST) if exp == "raceday" else pf.astimezone(JST) + timedelta(days=1)
                        day_ok = raceday(target.strftime("%Y%m%d"))
                    started = last.get("started_at")
                    last_start = datetime.fromisoformat(started).timestamp() if started else (runs[-1][0] if runs else 0.0)
                    late = (cnow - pf).total_seconds() / 60
                    if day_ok and late > OVERDUE_GRACE_MIN and last_start < pf.timestamp() - 60:
                        overdue_min = round(late)
            running_since = j.get("running_since")
            running_h = None
            if j.get("state") == "running" and running_since:
                try:
                    running_h = (now - datetime.fromisoformat(running_since)).total_seconds() / 3600
                except ValueError:
                    running_h = None
            jobs.append({
                "id": j["id"], "name": j.get("name") or j["id"], "group": g.get("name") or g.get("key"),
                "state": j.get("state"), "result": last.get("result"), "started_at": last.get("started_at"),
                "ended_at": last.get("ended_at"), "duration_sec": last.get("duration_sec"), "exit_code": last.get("exit_code"),
                "next_run": j.get("next_run"), "consecutive_failures": consecutive_failures(runs),
                "overdue_min": overdue_min, "running_hours": running_h, "yields": bool(j.get("yields")),
                "cause": ((j.get("cause") or {}).get("label")), "cause_id": ((j.get("cause") or {}).get("pattern_id")),
            })
    return jobs


def _jst(s: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(s).astimezone(JST) if s else None
    except (TypeError, ValueError):
        return None


def collect_prod(now: datetime, io: Any, *, doc: dict | None = None, shadow: bool = False, raceday: Callable[[str], bool] | None = None,
                 limit: int = 30) -> list[dict]:
    """prod（GCP Jobs・VPS 実行役）のジョブの状態を、collect_cron と同じ形で返す。ready で有効なジョブだけ。"""
    from src.ops import job_map as JM
    from src.ops.gcs_io import read_json as gread

    doc = doc or JM.load()
    groups = doc.get("scheduler_groups") or {}
    tasks_prefix = "ops/fetch_tasks_shadow" if shadow else "ops/fetch_tasks"
    now = now.astimezone(JST)
    out: list[dict] = []
    for j in doc.get("jobs") or []:
        dest = j.get("destination")
        if dest not in ("gcp", "vps") or not j.get("enabled", True) or not j.get("ready"):
            continue
        runs: list[dict] = []          # 新しい順 {ok, started, ended, detail}
        if dest == "gcp":
            keys = sorted((k for k in io.list(f"ops/job_runs/{j['id']}/") if not k.endswith("latest.json")), reverse=True)[:limit]
            for k in keys:
                r = gread(io, k)
                if isinstance(r, dict) and r.get("status") not in ("skipped", "waiting"):
                    runs.append({"ok": r.get("status") in ("ok", "partial"), "started": r.get("started_at"), "ended": r.get("finished_at"),
                                 "detail": r.get("reason"), "sec": r.get("duration_sec")})
            cron = (groups.get(j.get("group") or "") or {}).get("cron_jst")
        else:
            if j.get("reserved_by"):
                pres = [f"{t}-" for t in [j["fetch_task"]["task"], *(j.get("extra_tasks") or [])]]
            else:
                pres = [f"{j['id']}-"]
            for state in ("done", "failed"):
                for pre in pres:
                    for k in sorted(io.list(f"{tasks_prefix}/{state}/{pre}"), reverse=True)[:limit]:
                        t = gread(io, k)
                        if not isinstance(t, dict):
                            continue
                        oc = t.get("outcome") or {}
                        if (oc.get("result") or {}).get("skipped"):
                            continue
                        runs.append({"ok": state == "done", "started": t.get("claimed_at") or t.get("not_before"), "ended": t.get("finished_at"),
                                     "detail": oc.get("reason") or (f"rc={oc.get('rc')}" if oc.get("rc") else None), "sec": oc.get("elapsed_sec"),
                                     "restricted": bool(oc.get("access_restriction"))})
            runs.sort(key=lambda r: r.get("started") or "", reverse=True)
            runs = runs[:limit]
            cron = j.get("schedule_jst")
        cf = 0
        for r in runs:
            if r["ok"]:
                break
            cf += 1
        last = runs[0] if runs else {}
        overdue_min = None
        exp = expectation_of(j["id"])
        if cron:
            pf = prev_fire(cron.strip(), now)
            when = j.get("when_jst") if dest == "gcp" else None      # グループの起動のうち when_jst に合う回だけ動く
            for _ in range(2000):
                if pf is None or not when or prev_fire(when, pf) == pf:
                    break
                pf = prev_fire(cron.strip(), pf - timedelta(minutes=1))
            if pf is not None:
                day_ok = True
                if raceday is not None and exp != "always":
                    target = pf if exp == "raceday" else pf + timedelta(days=1)
                    day_ok = raceday(target.strftime("%Y%m%d"))
                ls = _jst(last.get("started"))
                late = (now - pf).total_seconds() / 60
                if day_ok and late > OVERDUE_GRACE_MIN and (ls is None or ls.timestamp() < pf.timestamp() - 60):
                    overdue_min = round(late)
        restricted = sum(1 for r in runs[:5] if r.get("restricted"))
        out.append({
            "id": j["id"], "name": j.get("desc") or j["id"], "group": "GCP Jobs" if dest == "gcp" else ("VPS（影）" if shadow else "VPS"),
            "state": None, "result": None if not last else ("ok" if last["ok"] else "failed"), "started_at": last.get("started"),
            "ended_at": last.get("ended"), "duration_sec": last.get("sec"), "exit_code": None if not last or last["ok"] else 1,
            "next_run": None, "consecutive_failures": cf, "overdue_min": overdue_min, "running_hours": None, "yields": False,
            "cause": last.get("detail") if last and not last["ok"] else None, "cause_id": "403" if restricted else None,
        })
    return out


def fetch_queue(base_url: str, timeout: float = 5.0, get: Callable[..., Any] | None = None) -> dict | None:
    """/api/scrape-jobs（stats と失敗の内訳）。取れなければ None（異常にはせず、サマリーに『取得不可』と出す）。"""
    try:
        if get is None:
            import requests

            get = requests.get
        r = get(base_url.rstrip("/") + "/api/scrape-jobs", timeout=timeout)
        if getattr(r, "status_code", 200) != 200:
            return None
        d = r.json()
    except Exception:  # noqa: BLE001
        return None
    jobs = d.get("jobs") or []
    reasons: dict[str, int] = {}
    for j in jobs:
        fr = j.get("failure_reason")
        if fr and j.get("status") in ("failed", "error", "dead"):
            reasons[str(fr)] = reasons.get(str(fr), 0) + 1
    return {"stats": d.get("stats") or {}, "failed_reasons": reasons, "n_jobs": len(jobs),
            "schema_failures": len(d.get("schema_validation_failures") or [])}


def collect_schema(base_dir: Path, now: datetime) -> dict | None:
    """直近 24 時間と過去 7 日（日ごと）のスキーマ違反の件数。"""
    try:
        from src.scraper import schema_violations as SV

        day = SV.read_log(base_dir, None, days=8)
    except Exception:  # noqa: BLE001
        return None
    by_day: dict[str, int] = {}
    last24 = 0
    cats: dict[str, int] = {}
    for rec in day:
        at = str(rec.get("at", ""))
        n = len(rec.get("violations") or []) or 1
        by_day[at[:10]] = by_day.get(at[:10], 0) + n
        try:
            t = datetime.fromisoformat(at)
            if t.tzinfo is None:
                t = t.replace(tzinfo=now.tzinfo)
            if now - t <= timedelta(hours=24):
                last24 += n
                cats[rec.get("category", "?")] = cats.get(rec.get("category", "?"), 0) + n
        except ValueError:
            continue
    return {"last24": last24, "by_day": by_day, "categories": dict(sorted(cats.items(), key=lambda kv: -kv[1])[:5])}


def read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def collect_misc(meta: Path, env_key: str) -> dict:
    """auto_scrape_status / data_health / drift_monitor（調教の被覆）。"""
    out: dict[str, Any] = {"auto": read_json(meta / "auto_scrape_status.json") or {}}
    dh = read_json(meta / "data_health" / env_key / "latest.json")
    if dh:
        rid_dir = meta / "data_health" / env_key / "race_ids"
        files = {}
        if rid_dir.is_dir():
            for f in sorted(rid_dir.glob("*.txt")):
                try:
                    files[f.stem] = sum(1 for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip())
                except OSError:
                    pass
        out["health"] = {"overall": (dh.get("summary") or {}).get("overall"), "counts": (dh.get("summary") or {}).get("counts"),
                         "complete": (dh.get("completeness") or {}).get("complete"),
                         "reasons": [r.get("message") for r in ((dh.get("completeness") or {}).get("reasons") or [])][:4],
                         "findings": [f"{x.get('area')}: {x.get('message')}"[:120] for x in (dh.get("findings") or []) if x.get("severity") in ("fail", "warn")][:4],
                         "generated_at": dh.get("generated_at"), "missing_lists": files}
    dm = read_json(meta / "drift_monitor" / "latest.json")
    if dm:
        cov = dm.get("coverage") or {}
        out["training_coverage"] = {"status": cov.get("status"), "block": cov.get("block"), "month": cov.get("latest_month"),
                                    "floor": cov.get("floor", COVERAGE_FLOOR)}
    return out


# ───────────────────────── 異常の判定 ─────────────────────────

def detect_anomalies(jobs: list[dict], queue: dict | None, schema: dict | None, misc: dict) -> dict[str, dict]:
    """現在の異常 {key: {severity, title, detail}}。キーが同じなら同じ異常として扱い、通知を抑制する。"""
    out: dict[str, dict] = {}
    for j in jobs:
        jid = j["id"]
        cf = j["consecutive_failures"]
        if cf >= FAIL_WARN and j.get("state") != "running":
            out[f"cron:{jid}:fail"] = {"severity": "critical" if cf >= FAIL_CRIT else "warning", "title": f"{j['name']} が連続 {cf} 回失敗",
                                      "detail": j.get("cause") or f"終了コード {j.get('exit_code')}"}
        if j.get("overdue_min"):
            m = j["overdue_min"]
            out[f"cron:{jid}:overdue"] = {"severity": "critical" if m >= OVERDUE_CRIT_MIN else "warning",
                                         "title": f"{j['name']} が予定時刻から {m} 分たっても始まっていない",
                                         "detail": f"最終実行 {j.get('started_at') or '記録なし'}"}
        if j.get("running_hours") and j["running_hours"] >= STUCK_HOURS:
            out[f"cron:{jid}:stuck"] = {"severity": "warning", "title": f"{j['name']} が {j['running_hours']:.1f} 時間実行中（停止の疑い）", "detail": ""}
        if j.get("cause_id") and any(k in str(j["cause_id"]).lower() for k in ("dns", "network", "403", "429", "block")) and cf >= 1:
            out["net:cause"] = {"severity": "critical", "title": "取得の停止の兆候（ネットワーク/アクセス拒否）",
                                "detail": f"{j['name']}: {j.get('cause')}"}
    if queue:
        blk = sum(v for k, v in (queue.get("failed_reasons") or {}).items() if any(x in k for x in ("403", "429", "dns", "blocked")))
        if blk >= BLOCK_MIN:
            out["net:block"] = {"severity": "critical", "title": f"取得の停止の兆候: 403/429/DNS 系の失敗が {blk} 件",
                                "detail": ", ".join(f"{k}={v}" for k, v in (queue.get("failed_reasons") or {}).items())}
    if schema:
        hist = [v for d, v in schema["by_day"].items()]
        base = statistics.median(hist[:-1]) if len(hist) > 2 else 0
        if schema["last24"] >= SCHEMA_SPIKE_MIN and schema["last24"] >= 3 * max(base, 1):
            out["schema:spike"] = {"severity": "warning", "title": f"スキーマ違反が急増（24時間で {schema['last24']} 件、通常の中央値 {base:.0f}）",
                                   "detail": ", ".join(f"{k} {v}" for k, v in schema["categories"].items())}
    tc = misc.get("training_coverage") or {}
    if tc.get("block") is not None and tc["block"] < float(tc.get("floor") or COVERAGE_FLOOR):
        out["coverage:training"] = {"severity": "critical", "title": f"調教データの被覆が低下（{tc.get('month')}: {tc['block'] * 100:.0f}%）",
                                    "detail": f"下限 {float(tc.get('floor') or COVERAGE_FLOOR) * 100:.0f}%。調教系の特徴量が欠損のまま予測される"}
    h = misc.get("health") or {}
    if h.get("overall") == "fail":
        out["health:fail"] = {"severity": "warning", "title": "データヘルスが fail", "detail": "; ".join(h.get("findings") or [])}
    return out


def step_state(state: dict, current: dict[str, dict], now: datetime, *, repeat_min: int = REPEAT_MIN,
               max_repeats: int = MAX_REPEATS) -> tuple[list[dict], list[dict], dict]:
    """通知の抑制。戻り値: (今送る異常, 復旧した異常, 新しい状態)。

    - 新規: すぐ送る。継続: repeat_min 分たつごとに最大 max_repeats 回まで再通知（その後は黙る）。
    - 重要度が上がったら（注意→緊急）すぐ送る。復旧（現在の異常に無い）: 『復旧』を 1 回送って状態から消す。
    """
    new_state: dict[str, dict] = {}
    notify, resolved = [], []
    ts = now.timestamp()
    for key, a in current.items():
        prev = state.get(key)
        item = {**a, "key": key}
        if not prev:
            notify.append({**item, "kind": "new"})
            new_state[key] = {"first": ts, "last_notified": ts, "repeats": 0, "severity": a["severity"]}
            continue
        s = dict(prev)
        escalated = SN.SEVERITIES.index(a["severity"]) > SN.SEVERITIES.index(prev.get("severity", "normal"))
        due = (ts - prev.get("last_notified", 0)) >= repeat_min * 60 and prev.get("repeats", 0) < max_repeats
        if escalated or due:
            notify.append({**item, "kind": "escalated" if escalated else "repeat"})
            s["last_notified"] = ts
            s["repeats"] = prev.get("repeats", 0) + (0 if escalated else 1)
        s["severity"] = a["severity"]
        new_state[key] = s
    for key, prev in state.items():
        if key not in current:
            resolved.append({"key": key, "since": prev.get("first")})
    return notify, resolved, new_state


# ───────────────────────── メッセージ ─────────────────────────

def _hm(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso).strftime("%m/%d %H:%M")
    except ValueError:
        return str(iso)[:16]


def _dur(sec: float | None) -> str:
    if sec is None:
        return "—"
    return f"{int(sec // 3600)}h{int(sec % 3600 // 60):02d}m" if sec >= 3600 else f"{int(sec // 60)}m{int(sec % 60):02d}s"


def signal_of(jobs: list[dict], anomalies: dict[str, dict], schema: dict | None, queue: dict | None) -> str:
    """green / yellow / red。赤=未実行・連続失敗・緊急の異常、黄=一部失敗（リトライ済み含む）・スキーマ違反・キューの失敗。"""
    if any(a["severity"] == "critical" for a in anomalies.values()) or any(j.get("overdue_min") for j in jobs):
        return "red"
    if any(j["consecutive_failures"] >= FAIL_WARN for j in jobs):
        return "red"
    if anomalies or any(j.get("result") in ("failed", "warning") for j in jobs) or (schema and schema["last24"]) \
            or (queue and queue.get("failed_reasons")):
        return "yellow"
    return "green"


SIGNAL_COLOR = {"green": SN.SEVERITY_COLOR["good"], "yellow": SN.SEVERITY_COLOR["warning"], "red": SN.SEVERITY_COLOR["critical"]}
SIGNAL_WORD = {"green": "正常", "yellow": "一部に注意", "red": "要対応"}


def build_summary(now: datetime, jobs: list[dict], queue: dict | None, schema: dict | None, misc: dict, anomalies: dict[str, dict],
                  *, race_day: bool) -> dict:
    sig = signal_of(jobs, anomalies, schema, queue)
    title = f"スクレイピング状況 {now.strftime('%m/%d')}（{SN.jp_weekday(now.strftime('%Y%m%d'))}）{'開催日' if race_day else '非開催日'}"
    ok = sum(1 for j in jobs if j.get("result") == "success")
    bad = [j for j in jobs if j.get("result") == "failed" or j["consecutive_failures"] or j.get("overdue_min")]
    blocks = [SN.section(f"*{SN.esc(title)}*　［{SIGNAL_WORD[sig]}］")]
    kp = [("ジョブ", f"{len(jobs)} 件（成功 {ok} / 失敗・未開始 {len(bad)}）" if jobs else "取得不可（crontab を読めません。root で実行してください）")]
    if queue:
        st = queue["stats"]
        kp.append(("キュー", f"実行 {st.get('running', 0)} / 待機 {st.get('queued', 0)} / 失敗理由 {sum(queue['failed_reasons'].values())} 件"))
    else:
        kp.append(("キュー", "取得不可（API 未応答）"))
    kp.append(("スキーマ違反（24h）", f"{schema['last24']} 件" if schema else "—"))
    tc = misc.get("training_coverage") or {}
    if tc.get("block") is not None:
        kp.append(("調教の被覆", f"{tc['block'] * 100:.0f}%（{tc.get('month')}）"))
    h = misc.get("health") or {}
    if h:
        kp.append(("データヘルス", f"{h.get('overall')}（欠損リスト {sum((h.get('missing_lists') or {}).values())} 件）"))
    blocks.append(SN.fields(kp))

    groups: dict[str, list[dict]] = {}
    for j in jobs:
        groups.setdefault(j["group"], []).append(j)
    rows = []
    for g, js in groups.items():
        gok = sum(1 for j in js if j.get("result") == "success")
        gbad = sum(1 for j in js if j.get("result") == "failed" or j["consecutive_failures"] or j.get("overdue_min"))
        rows.append([g[:14], str(len(js)), str(gok), str(gbad)])
    blocks.append(SN.section("*タスク別*\n" + SN.code_table(["グループ", "件", "成功", "失敗/未開始"], rows)))

    if bad:
        prow = [[j["name"][:22], "未開始" if j.get("overdue_min") else ("失敗×" + str(j["consecutive_failures"]) if j["consecutive_failures"] else "失敗"),
                 _hm(j.get("started_at")), _dur(j.get("duration_sec"))] for j in bad[:10]]
        blocks.append(SN.section("*問題のあるジョブ*\n" + SN.code_table(["ジョブ", "状態", "最終開始", "所要"], prow, aligns="llll")))
    auto = misc.get("auto") or {}
    lines = []
    for k, v in auto.items():
        if isinstance(v, dict):
            bits = [v.get("status", "")]
            for f in ("races", "shutuba", "saved", "pending", "error_count"):
                if f in v:
                    bits.append(f"{f}={v[f]}")
            if v.get("reason"):
                bits.append(f"reason={v['reason']}")
            lines.append(f"• {SN.esc(k)}: " + " ".join(str(x) for x in bits))
    if lines:
        blocks.append(SN.section("*取得結果（最終実行）*\n" + "\n".join(lines[:8])))
    if schema and schema["categories"]:
        blocks.append(SN.section("*スキーマ違反の多いカテゴリ（24h）*\n" + ", ".join(f"{SN.esc(k)} {v}" for k, v in schema["categories"].items())))
    if h.get("missing_lists"):
        blocks.append(SN.section("*欠損（データヘルス）*\n" + ", ".join(f"{SN.esc(k)} {v}" for k, v in h["missing_lists"].items())))
    nxt = sorted([j for j in jobs if j.get("next_run")], key=lambda j: j["next_run"])[:4]
    if nxt:
        blocks.append(SN.context(["次回予定: " + " / ".join(f"{SN.esc(j['name'][:16])} {_hm(j['next_run'])}" for j in nxt)]))
    if anomalies:
        blocks.append(SN.section("*検知中の異常*\n" + "\n".join(f"• [{SN.SEVERITY_LABEL[a['severity']]}] {SN.esc(a['title'])}" for a in anomalies.values())))
    return SN.build_message(title, blocks, severity="normal", fallback=f"{title} ［{SIGNAL_WORD[sig]}］", color=SIGNAL_COLOR[sig])


def build_alert(items: list[dict], resolved: list[dict], now: datetime) -> dict | None:
    """送る異常と復旧をまとめて 1 通に。何も無ければ None。"""
    if not items and not resolved:
        return None
    top = "critical" if any(i["severity"] == "critical" for i in items) else "warning"
    blocks: list[dict] = []
    if items:
        head = f"*[{SN.SEVERITY_LABEL[top]}] スクレイピングの異常 {len(items)} 件*"
        blocks.append(SN.section(head))
        for i in items[:8]:
            tag = {"new": "新規", "repeat": "継続", "escalated": "悪化"}.get(i["kind"], "")
            blocks.append(SN.section(f"*{SN.esc(i['title'])}*　［{SN.SEVERITY_LABEL[i['severity']]}・{tag}］\n{SN.esc(i.get('detail') or '')}".strip()))
    if resolved:
        names = ", ".join(SN.esc(r["key"]) for r in resolved[:8])
        blocks.append(SN.section(f"*復旧しました*　{names}"))
    blocks.append(SN.context([now.strftime("%Y-%m-%d %H:%M JST")]))
    sev = top if items else "normal"
    fb = f"[{SN.SEVERITY_LABEL[sev]}] スクレイピングの異常 {len(items)} 件" if items else f"スクレイピングが復旧しました（{len(resolved)} 件）"
    return SN.build_message("スクレイピングの異常", blocks, severity=sev, fallback=fb, color=SN.SEVERITY_COLOR["good"] if not items else None)


def build_coverage(now: datetime, misc: dict, schema: dict | None) -> dict:
    h = misc.get("health") or {}
    tc = misc.get("training_coverage") or {}
    blocks = [SN.section(f"*週次カバレッジ {now.strftime('%m/%d')}*")]
    kp = []
    if h:
        kp.append(("データヘルス", f"{h.get('overall')}（fail {h.get('counts', {}).get('fail', 0)} / warn {h.get('counts', {}).get('warn', 0)}）"))
        kp.append(("2020年以降の充足", "満たす" if h.get("complete") else "未達"))
    if tc.get("block") is not None:
        kp.append(("調教の被覆", f"{tc['block'] * 100:.0f}%（{tc.get('month')}、下限 {float(tc.get('floor') or COVERAGE_FLOOR) * 100:.0f}%）"))
    else:
        kp.append(("調教の被覆", "未計測（drift_monitor の実行後に表示）"))
    if schema:
        kp.append(("スキーマ違反（7日）", f"{sum(schema['by_day'].values())} 件"))
    blocks.append(SN.fields(kp))
    if h.get("reasons"):
        blocks.append(SN.section("*未達の理由*\n" + "\n".join("• " + SN.esc(x) for x in h["reasons"])))
    if h.get("missing_lists"):
        blocks.append(SN.section("*欠損・不適合・未検証の race_id*\n" + SN.code_table(["リスト", "件数"], [[k, str(v)] for k, v in h["missing_lists"].items()], aligns="lr")))
    if h.get("findings"):
        blocks.append(SN.section("*指摘*\n" + "\n".join("• " + SN.esc(x) for x in h["findings"])))
    sev = "critical" if (tc.get("block") is not None and tc["block"] < float(tc.get("floor") or COVERAGE_FLOOR)) else ("warning" if not h.get("complete", True) else "normal")
    return SN.build_message("週次カバレッジ", blocks, severity=sev, fallback=f"週次カバレッジ {now.strftime('%m/%d')}")


# ───────────────────────── 実行 ─────────────────────────

MAINTENANCE_GRACE_MIN = 10   # 窓の終了後、再起動した API・ジョブが戻るまでの猶予（分）


def in_maintenance(now: datetime, env: dict[str, str] | None = None) -> bool:
    """メンテナンス時間帯（毎日 JST 03:00–04:00、KEIBA_MAINTENANCE_WINDOW=1 のとき）か、その終了直後の猶予中か。"""
    from datetime import timedelta

    from src.utils import maintenance as M

    try:
        cfg = M.load_config()
        if not M.is_enabled(cfg, env):
            return False
        return any(M.compute_status(cfg, t, enabled=True)["active"] for t in (now, now - timedelta(minutes=MAINTENANCE_GRACE_MIN)))
    except Exception:  # noqa: BLE001  判定の失敗で通知を止めない
        return False


def run(mode: str, now: datetime, *, send: bool = False, state_path: Path | None = None, out_dir: Path | None = None,
        inputs: dict | None = None, poster: Callable[..., Any] | None = None, env: dict[str, str] | None = None, source: str = "host") -> dict:
    """mode ごとにメッセージを作り、dry-run なら出力、send なら送る。inputs でテストから材料を差し込める。"""
    e = os.environ if env is None else env
    if send and str(e.get("KEIBA_SLACK_NOTIFY", "")).strip() != "1":
        return {"code": 13, "reason": "KEIBA_SLACK_NOTIFY!=1"}
    inp = inputs or {}
    if (inp["maintenance"] if "maintenance" in inp else in_maintenance(now, e)):
        # 毎日のメンテナンス時間帯（と終了後の猶予）は『予定された停止』。API の 503・再起動・ジョブの停止を異常にしない（状態も更新しない）
        return {"code": 0, "mode": mode, "reason": "maintenance"}
    meta = Path(inp.get("meta") or META)
    env_key = inp.get("env_key") or (e.get("KEIBA_ENV") or "stg")
    raceday_fn = inp.get("raceday") or _default_raceday
    if source == "prod":
        if "jobs" not in inp:
            from src.ops.gcs_io import default_io

            inp = {**inp, "jobs": collect_prod(now, default_io(), shadow=str(e.get("KEIBA_FETCH_SHADOW_ENABLED", "")).strip() == "1", raceday=raceday_fn)}
        inp = {"queue": None, "schema": None, "misc": {}, **inp}      # キュー・スキーマ違反・data_health はホストのローカルにだけある
    jobs = inp["jobs"] if "jobs" in inp else collect_cron(now, raceday=raceday_fn)
    queue = inp["queue"] if "queue" in inp else fetch_queue(e.get("KEIBA_NOTIFY_API_BASE") or "http://127.0.0.1:23220")
    schema = inp["schema"] if "schema" in inp else collect_schema(REPO_ROOT, now)
    misc = inp["misc"] if "misc" in inp else collect_misc(meta, env_key)
    anomalies = detect_anomalies(jobs, queue, schema, misc)
    res: dict[str, Any] = {"code": 0, "mode": mode, "anomalies": list(anomalies)}
    payload = None
    new_state = None
    if mode == "alerts":
        sp = state_path or meta / "slack_sent" / "scrape_status_state.json"
        state = read_json(sp) or {}
        items, resolved, new_state = step_state(state, anomalies, now)
        payload = build_alert(items, resolved, now)
        res.update(to_notify=[i["key"] for i in items], resolved=[r["key"] for r in resolved])
    elif mode == "summary":
        payload = build_summary(now, jobs, queue, schema, misc, anomalies, race_day=bool(raceday_fn(now.strftime("%Y%m%d"))))
    elif mode == "coverage":
        payload = build_coverage(now, misc, schema)
    else:
        return {"code": 1, "reason": f"unknown mode {mode}"}
    res["payload"] = payload
    if payload and out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = f"{mode}_{now.strftime('%Y%m%d_%H%M')}"
        (out_dir / f"scrape_{stamp}.json").write_text(SN.dumps(payload), encoding="utf-8")
        (out_dir / f"scrape_{stamp}.html").write_text(SN.render_preview_html(payload, title=f"スクレイピング状況 {mode}"), encoding="utf-8")
    if send and payload:
        chan = "alert" if mode == "alerts" else "report"
        kw: dict[str, Any] = {"env": dict(e)}
        if poster:
            kw["post"] = poster
        r = SN.send_payload(payload, channel=chan, **kw)
        res["send"] = r
        if not r["ok"]:
            res["code"] = 14
            return res           # 状態は更新しない（次回また送る）
    if new_state is not None and send:
        sp.parent.mkdir(parents=True, exist_ok=True)
        tmp = sp.with_suffix(".tmp")
        tmp.write_text(json.dumps(new_state, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, sp)
    return res


def _default_raceday(date: str) -> bool:
    """その日が開催日か。レース一覧（race_lists）の保存を見る。無ければ土日を開催日とみなす。"""
    for base in (REPO_ROOT / "data" / "page_reference" / "race_lists", REPO_ROOT / "data" / "local" / "race_lists"):
        for p in (base / f"{date}.json", base / date / "race_lists.json"):
            d = read_json(p)
            if d is not None:
                return bool(d.get("races"))
    return datetime.strptime(date, "%Y%m%d").weekday() >= 5


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=["alerts", "summary", "coverage"], default="summary")
    ap.add_argument("--when", choices=["any", "raceday", "nonraceday"], default="any",
                    help="summary のとき、今日が開催日(raceday)/非開催日(nonraceday)のときだけ送る（開催日は夜、非開催日は朝）")
    ap.add_argument("--now", help="現在時刻の上書き（ISO。テスト/再現用）")
    ap.add_argument("--send", action="store_true")
    ap.add_argument("--out", default="data/local/meta/slack_preview")
    ap.add_argument("--source", choices=["host", "prod"], default="host",
                    help="prod: GCP Jobs・VPS 実行役の実行記録（GCS）を見る。host: このホストの crontab と cron_run の記録")
    a = ap.parse_args(argv)
    now = datetime.fromisoformat(a.now).astimezone(JST) if a.now else datetime.now(JST)
    if a.when != "any" and a.mode == "summary":
        is_rd = _default_raceday(now.strftime("%Y%m%d"))
        if (a.when == "raceday") != is_rd:
            print(json.dumps({"code": 0, "mode": a.mode, "skipped": f"today raceday={is_rd}, when={a.when}"}, ensure_ascii=False))
            return 0
    res = run(a.mode, now, send=a.send, out_dir=Path(a.out), source=a.source)
    print(json.dumps({k: v for k, v in res.items() if k != "payload"}, ensure_ascii=False, default=str))
    if res["code"] == 0 and not a.send:
        print(f"dry-run: {a.out} にプレビューを出力しました（送信していません）")
    return res["code"]


if __name__ == "__main__":
    sys.exit(main())
