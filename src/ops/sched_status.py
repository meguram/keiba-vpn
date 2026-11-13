"""VPS の定期実行の見張り（keiba-sched-status.timer から 15 分おき）。Cloud Run を起こさない。

    python -m src.ops.sched_status            # 見て・書いて・必要なら Slack
    python -m src.ops.sched_status --dry-run  # 見て判定を表示するだけ（GCS にも Slack にも出さない）

1 回の流れ:
1. 一覧（config/scheduled_jobs.json）の VPS のタイマー・常駐の状態を ``systemctl show --timestamp=unix`` 1 回で読む。
2. 手元の履歴（/var/lib/keiba-fetch/sched_status/state.json）に、新しい起動・終わった回を足す（直近 10 回）。
3. ルール（src/ops/schedule_rules.py）で VPS の分を判定する。
4. GCS の ``ops/heartbeat/vps/status.json`` に 1 回書く（VPS → GCS の書き込みは転送料 0。月 2,880 回・Class A 約 2.3 円）。
   監視ポータルと毎朝の点検（g-0730-shadow）はこれを読む。45 分更新が無ければ「VPS の記録の途絶」とみなされる。
5. 赤になった・赤が解消したときだけ Slack（/etc/keiba-ui/slack.env の SLACK_WEBHOOK_URL。同じ赤は繰り返し送らない）。
GCS に書けなくても Slack の判定は続ける（GCS の権限は ops/heartbeat/vps/status.json だけの条件つき）。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from src.ops import schedule_rules as R
from src.ops import scheduled_jobs as SJ

STATE = Path(os.environ.get("KEIBA_SCHED_STATE", "/var/lib/keiba-fetch/sched_status/state.json"))
OBJECT = "ops/heartbeat/vps/status.json"
KEEP_RUNS = 10
SERVICE_PROPS = ("Id", "ActiveState", "SubState", "Result", "ExecMainStatus", "ExecMainStartTimestamp", "ExecMainExitTimestamp")
TIMER_PROPS = ("Id", "LastTriggerUSec", "NextElapseUSecRealtime", "UnitFileState", "ActiveState")


def vps_jobs(reg: dict[str, Any]) -> list[dict[str, Any]]:
    return [j for j in reg["jobs"] if j["where"] == "vps" and (j.get("ref") or {}).get("service")]


def _ts(v: str | None) -> float | None:
    """``@1794500000`` → 1794500000.0。未設定（空・n/a・0）は None。"""
    v = (v or "").strip()
    if v.startswith("@"):
        try:
            t = float(v[1:])
            return t if t > 0 else None
        except ValueError:
            return None
    return None


def parse_show(text: str) -> dict[str, dict[str, str]]:
    """``systemctl show`` の出力（ユニットごとに空行で区切られた KEY=VALUE）→ Id → 属性。"""
    out: dict[str, dict[str, str]] = {}
    for block in text.strip().split("\n\n"):
        d = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
        if d.get("Id"):
            out[d["Id"]] = d
    return out


def read_systemd(jobs: list[dict[str, Any]], run: Any = subprocess.run) -> dict[str, dict[str, str]]:
    units: list[str] = []
    for j in jobs:
        units.append(j["ref"]["service"])
        if j["ref"].get("timer"):
            units.append(j["ref"]["timer"])
    props = sorted(set(SERVICE_PROPS) | set(TIMER_PROPS))
    cmd = ["systemctl", "show", "--timestamp=unix", *[f"-p{p}" for p in props], *units]
    res = run(cmd, capture_output=True, text=True, timeout=30)
    if getattr(res, "returncode", 0) != 0 or not (res.stdout or "").strip():
        # systemd 248 未満（--timestamp=unix が無い）なども含め、読めなかったことを隠さない（記録に error として書く）
        raise RuntimeError(f"systemctl show が失敗（rc={getattr(res, 'returncode', '?')}）: {(res.stderr or '').strip()[:160]}")
    return parse_show(res.stdout or "")


def observe(jobs: list[dict[str, Any]], units: dict[str, dict[str, str]], hist: dict[str, list[dict[str, Any]]], now: float) -> dict[str, dict[str, Any]]:
    """systemd の今の状態を、ジョブごとの記録（schedule_rules の obs の形）にする。hist（job id → 直近の回）を更新して返す。"""
    obs: dict[str, dict[str, Any]] = {}
    for j in jobs:
        svc = units.get(j["ref"]["service"]) or {}
        tmr = units.get(j["ref"].get("timer") or "") or {}
        runs = list(hist.get(j["id"]) or [])
        start = _ts(svc.get("ExecMainStartTimestamp"))
        end = _ts(svc.get("ExecMainExitTimestamp"))
        active = svc.get("ActiveState")
        if start is not None:
            running = active in ("activating", "active") and (end is None or end < start) and j["kind"] == "timer"
            ok = None if running else (svc.get("Result") == "success" and svc.get("ExecMainStatus", "0") in ("0", ""))
            code = None if running else _int(svc.get("ExecMainStatus"))
            rec = {"start": start, "end": None if running else (end if end and end >= start else None), "ok": ok, "code": code}
            if runs and abs(runs[0]["start"] - start) < 1:
                runs[0] = rec              # 同じ回の更新（終わった・結果が出た）
            else:
                runs.insert(0, rec)        # 新しい起動
        trig = _ts(tmr.get("LastTriggerUSec"))
        if start is None and trig is not None and (not runs or trig - runs[0]["start"] > 5):
            # サービス側に時刻が無い（再起動直後など）ときだけ、タイマーの最後の起動で補う。結果は分からない
            runs.insert(0, {"start": trig, "end": trig, "ok": None, "code": None})
        runs = runs[:KEEP_RUNS]
        hist[j["id"]] = runs
        obs[j["id"]] = {"runs": runs, "active": active, "timer_enabled": tmr.get("UnitFileState"), "next": _ts(tmr.get("NextElapseUSecRealtime"))}
    return obs


def _int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def load_state() -> dict[str, Any]:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"runs": {}, "red": {}}


def save_state(st: dict[str, Any]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE)


def write_gcs(doc: dict[str, Any]) -> str | None:
    """ops/heartbeat/vps/status.json に 1 回書く。失敗の理由を返す（成功は None）。"""
    try:
        from src.config.gcp_credentials import storage_client

        blob = storage_client().bucket(os.environ["GCS_BUCKET"]).blob(OBJECT)
        blob.upload_from_string(json.dumps(doc, ensure_ascii=False), content_type="application/json")
        return None
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {str(exc)[:160]}"


def slack(text: str, url: str | None = None) -> bool:
    url = url or os.environ.get("SLACK_WEBHOOK_URL") or ""
    if not url:
        return False
    try:
        req = urllib.request.Request(url, data=json.dumps({"text": text}).encode("utf-8"), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:   # noqa: S310  Slack の webhook（設定ファイルの URL）
            return 200 <= r.status < 300
    except Exception:  # noqa: BLE001
        return False


def run_once(*, now: float | None = None, read: Any = None, write: Any = write_gcs, notify: Any = slack, dry_run: bool = False) -> dict[str, Any]:
    now = time.time() if now is None else now
    reg = SJ.load()
    jobs = vps_jobs(reg)
    st = load_state()
    try:
        units = (read or read_systemd)(jobs)
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {str(exc)[:200]}"
        out = {"overall": "unknown", "counts": {}, "gcs_error": None, "sent": [], "recovered": [], "error": err}
        if not dry_run:
            out["gcs_error"] = write({"v": 1, "written_at": now, "host": "vps", "jobs": {}, "error": err})
        return out
    obs = observe(jobs, units, st.setdefault("runs", {}), now)
    res = R.evaluate(jobs, obs, now, sources={"vps": {"seen_at": now}})
    doc = {"v": 1, "written_at": now, "host": "vps", "jobs": obs,
           "result": {"overall": res["overall"], "counts": res["counts"], "rows": [{k: r[k] for k in ("id", "level", "reasons")} for r in res["rows"]]}}
    out = {"overall": res["overall"], "counts": res["counts"], "gcs_error": None, "sent": [], "recovered": []}
    if dry_run:
        out["rows"] = doc["result"]["rows"]
        return out
    out["gcs_error"] = write(doc)
    new, gone = R.transitions(st.get("red") or {}, res)
    red = dict(st.get("red") or {})
    if new:
        if notify("🔴 VPS の定期実行に異常\n" + "\n".join(f"・{v}" for v in new.values())):
            red.update(new)
            out["sent"] = list(new)
    if gone:
        if notify("✅ VPS の定期実行の異常が解消\n" + "\n".join(f"・{red.get(k, k)}" for k in gone)):
            for k in gone:
                red.pop(k, None)
            out["recovered"] = gone
    st["red"] = red
    st["last_run"] = now
    save_state(st)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    out = run_once(dry_run=a.dry_run)
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
