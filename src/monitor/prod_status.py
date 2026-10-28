"""監視ポータルの『prod（VPS）の状況』: 収集（ホストの cron が 1〜5 分おき）と、画面向けの読み出し。

- **収集** ``collect_all()``（CLI は ``python -m src.scripts.ops.collect_prod_status``）が、ソース別アダプタで各セクションを取り、
  ``data/local/meta/prod_status/latest.json`` に原子的に書く。画面（Flask）はこのキャッシュを読むだけで、リクエスト時に ssh / gcloud を叩かない。
- **アダプタ** は ``run(cmd, input_text, timeout) -> (rc, stdout, stderr)`` を受け取るだけで、実体を差し替えられる。
  暫定の実体: ホストから ``ssh keiba-vpn``（読み取りだけの固定スクリプト）・``gcloud_ro.sh``（読み取り専用のサービスアカウント）。
  最終形（ホストと無関係に公開）では、VPS のハートビート（GCS）→ GCP 上のモニターに、同じセクション形で差し替える。
- 各セクションは ``{"state": "ok"|"missing"|"error", "at": 取得時刻, "reason": 取れない理由, "data": {...}}``。取れない項目は落とさず『未取得（理由）』にする。
- 全体の信号 ``overall.level`` は green / yellow / red / unknown。目標は VPS のメモリ『通常 600MB 以下・MemAvailable 1GB 以上』。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(os.environ.get("KEIBA_REPO_ROOT") or Path(__file__).resolve().parents[2])
JST = timezone(timedelta(hours=9))
SCHEMA = 1

# 目標・閾値（docs/operations/architecture_target.md 3.2 と heartbeat_schema に合わせる）
MEM_USED_TARGET_MB = 600.0
MEM_AVAIL_WARN_MB = 1000.0
MEM_AVAIL_CRIT_MB = 600.0
SWAP_WARN_MB = 200.0
DISK_WARN_PCT = 80
DISK_CRIT_PCT = 85
RESTART_WARN = 3
HEARTBEAT_STALE_S = 15 * 60
CACHE_STALE_S = 15 * 60  # 収集が止まっているとみなす経過

REQUIRED_UNITS = ("keiba-ui", "nginx", "cloudflared-prod")
OPTIONAL_UNITS = ("keiba-fetch-agent", "keiba-ui-health.timer", "keiba-heartbeat.timer", "keiba-release-pull-ui.timer", "keiba-release-pull-fetch.timer")

Runner = Callable[[list, "str | None", float], "tuple[int, str, str]"]


def status_dir() -> Path:
    return Path(os.environ.get("KEIBA_PROD_STATUS_DIR") or REPO_ROOT / "data" / "local" / "meta" / "prod_status")


def cache_file() -> Path:
    return status_dir() / "latest.json"


def now_iso() -> str:
    return datetime.now(JST).isoformat(timespec="seconds")


def default_runner(cmd: list, input_text: str | None = None, timeout: float = 30.0) -> tuple[int, str, str]:
    try:
        r = subprocess.run(cmd, input=input_text, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout or "", r.stderr or ""
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout {timeout:.0f}s"
    except Exception as exc:  # noqa: BLE001
        return 127, "", f"{type(exc).__name__}: {exc}"


def ok(data: Any, at: str | None = None) -> dict:
    return {"state": "ok", "at": at or now_iso(), "data": data}


def missing(reason: str) -> dict:
    """まだ導入されていない・権限が無いなど、取れない（異常ではない）。"""
    return {"state": "missing", "at": now_iso(), "reason": reason, "data": None}


def error(reason: str) -> dict:
    """取ろうとして失敗した（接続できない・壊れた応答）。"""
    return {"state": "error", "at": now_iso(), "reason": reason[:300], "data": None}


# ───────────────────────── アダプタ（ソース別） ─────────────────────────

def add_restart_delta(data: dict, prev: dict | None) -> None:
    """NRestarts は unit を作り直すまでの累計で、過去の失敗（例: 初回リリースの起動失敗）が残り続ける。
    警告には『前回の収集からの増分』を使い、累計は参考表示にする。前回が無ければ増分は不明（None）。"""
    prev_units = ((prev or {}).get("units") or {}) if isinstance(prev, dict) else {}
    for name, u in (data.get("units") or {}).items():
        if not isinstance(u, dict) or not u.get("installed"):
            continue
        p = prev_units.get(name) or {}
        u["restart_delta"] = (u.get("restarts", 0) - p.get("restarts", 0)) if p.get("installed") and u.get("restarts", 0) >= p.get("restarts", 0) else None


def vps_adapter(run: Runner, probe: Path | None = None, ssh_host: str = "keiba-vps", prev: dict | None = None) -> dict:
    """VPS の資源・ユニット・UI の版・上流 API への疎通。ssh で読み取りだけの採取スクリプトを流す（VPS に何も入れない）。"""
    probe = probe or REPO_ROOT / "src" / "scripts" / "ops" / "vps_status_probe.py"
    try:
        script = probe.read_text(encoding="utf-8")
    except OSError as exc:
        return error(f"採取スクリプトが読めない: {exc}")
    rc, out, err = run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", ssh_host, "python3", "-"], script, 45.0)
    if rc != 0 or not out.strip():
        return error(f"VPS に ssh できない/採取失敗 (rc={rc}): {err.strip()[:160]}")
    try:
        data = json.loads(out.strip().splitlines()[-1])
    except ValueError:
        return error("VPS の採取結果が JSON ではない")
    add_restart_delta(data, prev)
    return ok(data)


def release_history(path: Path | None = None, n: int = 5) -> dict:
    """昇格履歴（promote.sh が追記する jsonl）の直近 n 件。"""
    path = path or REPO_ROOT / "data" / "local" / "meta" / "vps_promote_history.jsonl"
    try:
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        return missing("昇格履歴がまだありません（promote.sh の実行後に出ます）")
    rows = []
    for ln in lines[-n:]:
        try:
            rows.append(json.loads(ln))
        except ValueError:
            continue
    return ok({"recent": list(reversed(rows))})


def public_entry_probe(get: Callable[[str], tuple[int | None, str]] | None = None, url: str = "https://app.megu-keiba.com/") -> dict:
    """公開 URL の入口（Cloudflare Access）。Access の背後なので 302 でログインへ飛ぶのが正常。
    トンネルが落ちていても 302 になるため『入口が生きている』ことしか分からない（UI の生死は VPS 内部の採取で見る）。"""
    def _get(u: str) -> tuple[int | None, str]:
        import urllib.error
        import urllib.request

        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k):  # noqa: ANN002, ANN003
                return None

        try:
            opener = urllib.request.build_opener(_NoRedirect)
            with opener.open(urllib.request.Request(u, headers={"User-Agent": "keiba-monitor/1"}), timeout=8) as r:
                return r.status, r.headers.get("location", "")[:120]
        except urllib.error.HTTPError as e:
            return e.code, (e.headers.get("location", "") if e.headers else "")[:120]
        except Exception as exc:  # noqa: BLE001
            return None, type(exc).__name__

    code, loc = (get or _get)(url)
    if code is None:
        return error(f"入口に届かない: {loc}")
    return ok({"url": url, "status": code, "redirect": loc, "note": "Cloudflare Access の背後のため 302 が正常。トンネルの生死までは分からない"})


def _gcloud(run: Runner, gcloud_ro: str, args: list, timeout: float = 60.0) -> tuple[bool, Any, str]:
    rc, out, err = run([gcloud_ro, *args, "--format=json"], None, timeout)
    if rc != 0:
        return False, None, (err.strip() or out.strip())[:200]
    try:
        return True, json.loads(out or "null"), ""
    except ValueError:
        return False, None, "gcloud の出力が JSON ではない"


def gcp_scheduler_adapter(run: Runner, gcloud_ro: str, region: str = "asia-northeast1") -> dict:
    ok_, data, why = _gcloud(run, gcloud_ro, ["scheduler", "jobs", "list", "--location", region])
    if not ok_:
        return error(f"Cloud Scheduler を読めない: {why}")
    rows = [{"name": str(j.get("name", "")).rsplit("/", 1)[-1], "schedule": j.get("schedule"), "state": j.get("state"),
             "last_attempt": j.get("lastAttemptTime"), "status_code": (j.get("status") or {}).get("code")} for j in (data or [])]
    return ok({"jobs": rows, "count": len(rows)})


def _summarize_smoke(run: Runner, gcloud_ro: str, execution: str, repo: Path) -> str:
    ok_, logs, _ = _gcloud(run, gcloud_ro, ["logging", "read",
                                              f'resource.type="cloud_run_job" AND labels."run.googleapis.com/execution_name"="{execution}"',
                                              "--limit", "400", "--freshness", "14d", "--order", "asc"])
    if not ok_:
        return "ログを読めない"
    rc, out, _ = run([sys.executable, str(repo / "deploy" / "gcp" / "netkeiba_smoke" / "summarize.py")], json.dumps(logs or []), 30.0)
    for ln in out.splitlines():
        if ln.startswith("RESULT:"):
            return ln[:200]
    return "結果なし"


def gcp_smoke_adapter(run: Runner, gcloud_ro: str, region: str = "asia-northeast1", limit: int = 3, repo: Path | None = None) -> dict:
    """netkeiba 到達試験（netkeiba-smoke）の直近の実行と、各回の合否の 1 行。"""
    ok_, data, why = _gcloud(run, gcloud_ro, ["run", "jobs", "executions", "list", "--job", "netkeiba-smoke", "--region", region, "--limit", str(limit)])
    if not ok_:
        return error(f"Cloud Run Jobs を読めない: {why}")
    rows = []
    for e in data or []:
        name = (e.get("metadata") or {}).get("name") or e.get("name", "")
        st = e.get("status") or {}
        rows.append({"execution": str(name).rsplit("/", 1)[-1], "start": st.get("startTime"), "succeeded": st.get("succeededCount", 0),
                     "failed": st.get("failedCount", 0), "result": _summarize_smoke(run, gcloud_ro, str(name).rsplit("/", 1)[-1], repo or REPO_ROOT)})
    return ok({"executions": rows})


def _bucket() -> str:
    return os.environ.get("GCS_BUCKET", "").strip()


def heartbeat_adapter(run: Runner, gcloud_ro: str, now: float | None = None) -> dict:
    """VPS のハートビート（GCS ops/heartbeat/latest.json）の鮮度と現在の異常。クライアント導入前は『未導入』。"""
    b = _bucket()
    if not b:
        return missing("GCS_BUCKET が未設定")
    rc, out, err = run([gcloud_ro, "storage", "cat", f"gs://{b}/ops/heartbeat/latest.json"], None, 40.0)
    if rc != 0:
        txt = (err + out).lower()
        if "not found" in txt or "no urls matched" in txt or "matched no objects" in txt or "404" in txt:
            return missing("未導入（VPS のハートビート送信が始まると表示されます）")
        return error(f"ハートビートを読めない: {err.strip()[:160]}")
    try:
        rec = json.loads(out)
    except ValueError:
        return error("ハートビートが JSON ではない")
    from src.ops.heartbeat_schema import evaluate

    t = now if now is not None else time.time()
    age = t - float(rec.get("received_at") or 0) if rec.get("received_at") else None
    return ok({"age_s": round(age) if age is not None else None, "alerts": evaluate(rec, t), "latest": rec.get("latest")})


def shadow_adapter(run: Runner, gcloud_ro: str) -> dict:
    """影モード（fetch-shadow）の日次レポート。導入前は『未導入』。"""
    b = _bucket()
    if not b:
        return missing("GCS_BUCKET が未設定")
    rc, out, err = run([gcloud_ro, "storage", "ls", f"gs://{b}/ops/fetch_shadow/daily/"], None, 40.0)
    names = [ln.strip() for ln in out.splitlines() if ln.strip().endswith(".json")]
    if rc != 0 or not names:
        return missing("未導入（影モードを始めると日次レポートが出ます）")
    last = sorted(names)[-1]
    rc, body, err = run([gcloud_ro, "storage", "cat", last], None, 40.0)
    try:
        rep = json.loads(body)
    except ValueError:
        return error("影モードのレポートが読めない")
    return ok({"date": last.rsplit("/", 1)[-1][:-5], "days_listed": len(names), "report": {k: v for k, v in rep.items() if isinstance(v, (int, float, str, bool))}})


def cutover_adapter(run: Runner, python: str | None = None) -> dict:
    """ホスト cron 停止の判定（cron_cutover_status）。合格/不合格/未着手の件数と、止めてよい単位。"""
    rc, out, err = run([python or sys.executable, "-m", "src.scripts.ops.cron_cutover_status", "--json"], None, 90.0)
    if rc != 0:
        return error(f"cutover 判定を実行できない: {(err.strip() or out.strip())[-160:]}")
    try:
        res = json.loads(out[out.index("{"):])
    except ValueError:
        return error("cutover 判定が JSON ではない")
    rows = res.get("rows") or []
    count: dict[str, int] = {}
    for r in rows:
        count[r.get("state", "?")] = count.get(r.get("state", "?"), 0) + 1
    ready = [u for u, v in (res.get("cut_units") or {}).items() if v.get("ready_to_stop")]
    return ok({"counts": count, "total": len(rows), "ready_to_stop": ready})


def parity_adapter(path: Path | None = None) -> dict:
    """stg と prod の差分（env_parity_check の保存結果）。保存先が無ければ未導入。"""
    path = path or REPO_ROOT / "data" / "local" / "meta" / "env_parity" / "latest.json"
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return missing("未導入（env_parity_check の日次実行を登録すると表示されます）")
    return ok({"summary": {k: d.get(k) for k in ("ok", "diff_count", "diffs", "at") if k in d}})


def cost_adapter() -> dict:
    return missing("費用は読み取り専用の鍵に BigQuery の権限が無いため未取得（Slack の日次コスト報告を参照）")


# ───────────────────────── 全体の信号 ─────────────────────────

def _lvl(a: str, b: str) -> str:
    order = {"green": 0, "unknown": 1, "yellow": 2, "red": 3}
    return a if order[a] >= order[b] else b


def evaluate_overall(sections: dict) -> dict:
    """セクションから全体の信号と理由の一覧を作る。取れない項目は異常にしない（VPS 本体が取れないときだけ unknown/黄）。"""
    level = "green"
    reasons: list[dict] = []

    def add(lv: str, msg: str) -> None:
        nonlocal level
        level = _lvl(level, lv)
        reasons.append({"level": lv, "message": msg})

    vps = sections.get("vps") or {}
    if vps.get("state") != "ok":
        add("yellow", f"VPS の状況を取得できていません（{vps.get('reason') or '未取得'}）")
    else:
        d = vps["data"]
        mem = d.get("mem") or {}
        avail, used, swap = mem.get("available_mb"), mem.get("used_mb"), mem.get("swap_used_mb")
        if avail is not None:
            if avail < MEM_AVAIL_CRIT_MB:
                add("red", f"空きメモリ {avail:.0f}MB（{MEM_AVAIL_CRIT_MB:.0f}MB 未満）")
            elif avail < MEM_AVAIL_WARN_MB:
                add("yellow", f"空きメモリ {avail:.0f}MB（目標 {MEM_AVAIL_WARN_MB:.0f}MB 以上）")
        if used is not None and used > MEM_USED_TARGET_MB:
            add("yellow", f"使用メモリ {used:.0f}MB（通常時の目標 {MEM_USED_TARGET_MB:.0f}MB 以下）")
        if swap is not None and swap > SWAP_WARN_MB:
            add("yellow", f"swap {swap:.0f}MB 使用（緊急用のみの想定）")
        pct = (d.get("disk") or {}).get("used_pct")
        if pct is not None and pct >= DISK_CRIT_PCT:
            add("red", f"ディスク {pct}%")
        elif pct is not None and pct >= DISK_WARN_PCT:
            add("yellow", f"ディスク {pct}%")
        units = d.get("units") or {}
        for name in REQUIRED_UNITS:
            u = units.get(name) or {}
            if not u.get("installed") or u.get("active") != "active":
                add("red", f"{name} が稼働していません")
            elif (u.get("restart_delta") or 0) >= 1:
                add("red" if (u.get("restart_delta") or 0) >= RESTART_WARN else "yellow",
                    f"{name} が前回の収集から {u.get('restart_delta')} 回再起動しています（累計 {u.get('restarts')}）")
        for name in OPTIONAL_UNITS:
            u = units.get(name) or {}
            if u.get("installed") and u.get("active") in ("failed",):
                add("red", f"{name} が failed")
        ui = d.get("ui") or {}
        if ui.get("code") != 200:
            add("red", f"UI が応答しません（{ui.get('code')}）")
        api = d.get("upstream_api") or {}
        if api.get("code") != 200:
            add("red", f"VPS の nginx 経由の応答が 200 ではありません（{api.get('code')}）。API の死活は見ていません（課金のため叩かない）")
        elif (api.get("ms") or 0) > 3000:
            add("yellow", f"nginx 経由の応答が遅い（{api.get('ms')}ms）")
        tun = d.get("tunnel") or {}
        if tun.get("readable") and tun.get("recently_lost"):
            add("yellow", "トンネルの接続が直近で切れた記録があります")

    hb = sections.get("heartbeat") or {}
    if hb.get("state") == "ok":
        for a in (hb["data"] or {}).get("alerts") or []:
            add("red" if a.get("severity") == "critical" else "yellow", f"ハートビート: {a.get('title')}")
    if (sections.get("public_entry") or {}).get("state") == "error":
        add("yellow", "公開 URL の入口に届きません")
    if not reasons:
        reasons.append({"level": "green", "message": "異常は検出されていません"})
    return {"level": level, "reasons": reasons}


# ───────────────────────── 収集と読み出し ─────────────────────────

def collect_all(run: Runner | None = None, *, gcloud_ro: str | None = None, repo: Path | None = None, only: list | None = None) -> dict:
    """全セクションを取って結果を返す（書き込みはしない）。1 つのアダプタの失敗が他に影響しない。"""
    run = run or default_runner
    gro = gcloud_ro or os.environ.get("KEIBA_GCLOUD_RO") or str(Path.home() / ".config" / "keiba-gcp" / "gcloud_ro.sh")
    t0 = time.time()
    gro_ok = Path(gro).exists() or run is not default_runner
    plan: dict[str, Callable[[], dict]] = {
        "vps": lambda: vps_adapter(run),
        "releases": release_history,
        "public_entry": public_entry_probe,
        "heartbeat": lambda: heartbeat_adapter(run, gro) if gro_ok else missing("gcloud_ro.sh がありません"),
        "gcp_scheduler": lambda: gcp_scheduler_adapter(run, gro) if gro_ok else missing("gcloud_ro.sh がありません"),
        "gcp_smoke": lambda: gcp_smoke_adapter(run, gro, repo=repo) if gro_ok else missing("gcloud_ro.sh がありません"),
        "shadow": lambda: shadow_adapter(run, gro) if gro_ok else missing("gcloud_ro.sh がありません"),
        "cutover": lambda: cutover_adapter(run),
        "parity": parity_adapter,
        "cost": cost_adapter,
    }
    try:
        prev_vps = ((json.loads(cache_file().read_text(encoding="utf-8")).get("sections") or {}).get("vps") or {}).get("data")
    except (OSError, ValueError):
        prev_vps = None
    plan["vps"] = lambda: vps_adapter(run, prev=prev_vps)
    sections: dict[str, dict] = {}
    for key, fn in plan.items():
        if only and key not in only:
            continue
        try:
            sections[key] = fn()
        except Exception as exc:  # noqa: BLE001
            sections[key] = error(f"{type(exc).__name__}: {exc}")
    return {"schema": SCHEMA, "collected_at": now_iso(), "duration_s": round(time.time() - t0, 1), "sections": sections,
            "overall": evaluate_overall(sections)}


def write_cache(result: dict, path: Path | None = None) -> Path:
    path = path or cache_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load(path: Path | None = None, now: float | None = None) -> dict:
    """画面向け。キャッシュが無い・壊れている・古いときは、その旨を overall に入れて返す（例外で落とさない）。"""
    path = path or cache_file()
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(d, dict) or "sections" not in d:
            raise ValueError("形式が違う")
    except (OSError, ValueError):
        return {"status": "not_collected", "sections": {}, "overall": {"level": "unknown", "reasons": [
            {"level": "unknown", "message": "まだ収集されていません"}]}, "hint": "python -m src.scripts.ops.collect_prod_status を実行（cron 登録は scripts/ops/install_prod_status_cron.sh）"}
    try:
        age = (now if now is not None else time.time()) - path.stat().st_mtime
    except OSError:
        age = None
    d["status"] = "ok"
    d["age_s"] = round(age) if age is not None else None
    if age is not None and age > CACHE_STALE_S:
        d["status"] = "stale"
        ov = dict(d.get("overall") or {"level": "unknown", "reasons": []})
        ov["level"] = _lvl(ov.get("level", "unknown"), "yellow")
        ov["reasons"] = [{"level": "yellow", "message": f"収集が {int(age // 60)} 分止まっています（表示は古い値）"}, *(ov.get("reasons") or [])]
        d["overall"] = ov
    return d


def summary(d: dict) -> dict:
    """/services の上部カード用の短い要約。"""
    vps = ((d.get("sections") or {}).get("vps") or {})
    data = vps.get("data") or {}
    ui = (data.get("ui") or {}).get("json") or {}
    return {"status": d.get("status"), "level": (d.get("overall") or {}).get("level", "unknown"),
            "reasons": (d.get("overall") or {}).get("reasons", [])[:4], "collected_at": d.get("collected_at"), "age_s": d.get("age_s"),
            "release": data.get("current_release") or ui.get("release"), "env": ui.get("env"),
            "mem_available_mb": (data.get("mem") or {}).get("available_mb"), "mem_used_mb": (data.get("mem") or {}).get("used_mb")}


def maybe_notify(result: dict, state_path: Path | None = None) -> bool:
    """赤・黄の変化があったときだけ Slack へ（KEIBA_PROD_STATUS_NOTIFY=1 のときだけ。既定は送らない）。同じ内容は 1 時間に 1 回まで。"""
    if os.environ.get("KEIBA_PROD_STATUS_NOTIFY", "").strip() != "1":
        return False
    level = (result.get("overall") or {}).get("level")
    if level not in ("red", "yellow"):
        return False
    msgs = [r["message"] for r in (result.get("overall") or {}).get("reasons", []) if r.get("level") in ("red", "yellow")]
    key = f"{level}:{'|'.join(sorted(msgs))}"
    state_path = state_path or status_dir() / "notify_state.json"
    try:
        st = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        st = {}
    if st.get("key") == key and time.time() - float(st.get("at", 0)) < 3600:
        return False
    from src.utils import slack_notify as SN

    payload = SN.build_message("prod の状況", [SN.section("*prod（VPS）の状況: " + ("赤（要対応）" if level == "red" else "黄（注意）") + "*\n" + "\n".join("• " + SN.esc(m) for m in msgs[:8]))],
                               severity="critical" if level == "red" else "warning", fallback=f"prod の状況: {level}")
    res = SN.send_payload(payload, channel="alert")
    if res.get("ok"):
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({"key": key, "at": time.time()}), encoding="utf-8")
    return bool(res.get("ok"))
