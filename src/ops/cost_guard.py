"""月の GCP 費用の歯止め。今月の累計か月末の見込みが上限に近づいたら、並走中の GCP Jobs の Scheduler を一時停止して知らせる。

費用の上限は GCP + Cloudflare で月 ``config/cost_assumptions.json`` の ``budget_jpy_per_month``（2,000 円）。CLAUDE.md の最優先の決まり。
2026-11 に GCP Jobs の全ジョブの並走で 1 日 120〜250 円（月 4,000〜5,000 円ペース）になったため、見込みの段階で止める仕組みを置く。

- 材料: 課金エクスポート（BigQuery）の日別の請求額（クレジット差し引き後、円）。反映は約 1 日遅れるので、今日と昨日は平均に使わず、
  その前の ``RECENT_DAYS`` 日（前の月の末の日も使う）の平均を、今月の残りの日数（今日・昨日を含む）に掛けて月末の見込みにする。
  平均に使う日のデータが無い（課金エクスポートの遅れ・停止）ときは見込みを信用せず『異常』として知らせる。
- 判定: 今月の累計か見込みが上限 × ``RATIO``（既定 0.8）以上なら『上限に近い』。
- 止める対象: 名前が ``keiba-g-`` で始まる Cloud Scheduler のジョブのうち ``KEEP`` 以外（費用の日報のグループは残す）。
  場所は環境変数 ``REGION``（既定 asia-northeast1。scheduler_gen と同じ）。**自動では再開しない**。
  上限に近いあいだは、人が再開しても次の実行（最長 12 時間後）でまた止める。月が替わって見込みが下がるまで再開を続けたいときは、
  ``KEIBA_COST_GUARD_PAUSE=0`` で止める動作だけを外せる（判定と通知は続く）。
- 止められないもの: 本番の API（Cloud Run のサービス）・GCS の転送と操作（VPS・ホストからの読み出し）・Artifact Registry・
  Scheduler 以外から起動した Jobs・Cloudflare。これらで増えたときは止められないが、数字には入るので通知で気づける。
- 通知: 上限に近いあいだは 1 日 1 回（止めたときはその都度）。歯止め自体が動かなかったとき（課金・Scheduler の問い合わせの失敗）も送る。
- 記録: GCS の ``ops/cost_guard/<YYYYMMDD>.json``（その日の最新の判定）。

  python -m src.ops.cost_guard            # 判定だけ（何も止めない・送らない）
  python -m src.ops.cost_guard --apply    # 判定して、必要なら止めて知らせる（ホストの cron から 1 日 2 回）
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
JST = timezone(timedelta(hours=9))
RATIO = 0.8
RECENT_DAYS = 3
PREFIX = "keiba-g-"
KEEP = frozenset({"keiba-g-0730-shadow"})      # 費用の日報（gcp-daily-cost-report）のグループ
BILLING_DATASET = "billing_export"
HTTP_TIMEOUT = 30


def budget_jpy() -> float:
    d = json.loads((ROOT / "config" / "cost_assumptions.json").read_text(encoding="utf-8"))
    return float(d["budget_jpy_per_month"])


def project_id() -> str:
    from src.config.gcp_credentials import gcp_project_id

    return gcp_project_id() or os.environ.get("GCS_PROJECT_ID", "") or "horse-racing-466415"


def location() -> str:
    return os.environ.get("REGION", "").strip() or "asia-northeast1"


def billing_table() -> str:
    """課金エクスポートの表。GCP_BILLING_BQ_TABLE が無ければ billing_export の中の gcp_billing_export_v1_* を探す。"""
    t = os.environ.get("GCP_BILLING_BQ_TABLE", "").strip()
    if t:
        return t
    from google.cloud import bigquery

    from src.config.gcp_credentials import build_gcp_credentials

    proj = project_id()
    client = bigquery.Client(credentials=build_gcp_credentials(), project=proj)
    names = sorted(x.table_id for x in client.list_tables(f"{proj}.{BILLING_DATASET}") if x.table_id.startswith("gcp_billing_export_v1_"))
    if not names:
        raise RuntimeError(f"{proj}.{BILLING_DATASET} に課金エクスポートの表が無い")
    return f"{proj}.{BILLING_DATASET}.{names[0]}"


def assess(daily: dict[str, float], today: date, budget: float, *, ratio: float = RATIO, recent_days: int = RECENT_DAYS) -> dict[str, Any]:
    """日別の請求額（今月分と前の月の末の数日）から累計・見込み・判定を出す（純粋関数）。

    今日と昨日は課金の反映が途中なので平均に使わない。平均に使う日は前の月にまたがってもよい（月初でも見込みが効くように）。
    見込み = 今月の『平均に使った日まで』の合計 + 平均 × その後の今月の日数。平均に使う日のデータが無ければ missing に入れる。"""
    first = today.replace(day=1)
    last = date(today.year, today.month, calendar.monthrange(today.year, today.month)[1])
    mtd = sum(v for d, v in daily.items() if first.isoformat() <= d < today.isoformat())
    complete_end = today - timedelta(days=2)
    days = [complete_end - timedelta(days=i) for i in range(recent_days)]
    missing = sorted(d.isoformat() for d in days if d.isoformat() not in daily)
    vals = [daily[d.isoformat()] for d in days if d.isoformat() in daily]
    avg = sum(vals) / len(vals) if vals else 0.0
    counted = sum(v for d, v in daily.items() if first.isoformat() <= d <= complete_end.isoformat())
    remaining = (last - max(complete_end, first - timedelta(days=1))).days
    projection = counted + avg * remaining
    limit = budget * ratio
    return {"today": today.isoformat(), "budget": budget, "limit": round(limit, 1), "month_to_date": round(mtd, 1),
            "recent_avg": round(avg, 1), "projection": round(projection, 1), "missing_days": missing,
            "over": mtd >= limit or projection >= limit, "over_budget": mtd >= budget}


def fetch_daily(table: str, today: date, lookback: int = RECENT_DAYS + 2) -> dict[str, float]:
    """今月の 1 日（と、平均に使う前の月の末の日）から今日までの日別の請求額（円）。円以外の通貨が来たら例外。"""
    from src.scripts.cloud_jobs.gcp_daily_cost_report import fetch_cost_history

    start = min(today.replace(day=1), today - timedelta(days=lookback))
    out: dict[str, float] = {}
    for r in fetch_cost_history(table, start.isoformat(), today.isoformat()):
        if (r.get("currency") or "JPY") != "JPY":
            raise RuntimeError(f"課金エクスポートの通貨が JPY ではない: {r.get('currency')}（上限は円）")
        out[r["date"]] = out.get(r["date"], 0.0) + float(r["cost"])
    return out


def _session() -> Any:
    from google.auth.transport.requests import AuthorizedSession

    from src.config.gcp_credentials import build_gcp_credentials

    return AuthorizedSession(build_gcp_credentials().with_scopes(["https://www.googleapis.com/auth/cloud-platform"]))


def _base(proj: str, loc: str) -> str:
    return f"https://cloudscheduler.googleapis.com/v1/projects/{proj}/locations/{loc}/jobs"


def scheduler_jobs(session: Any, proj: str, loc: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    token = ""
    while True:
        r = session.get(_base(proj, loc), params={"pageSize": 100, **({"pageToken": token} if token else {})}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        d = r.json()
        out += d.get("jobs") or []
        token = d.get("nextPageToken") or ""
        if not token:
            return out


def _short(j: dict[str, Any]) -> str:
    return str(j.get("name", "")).rsplit("/", 1)[-1]


def pause_targets(jobs: list[dict[str, Any]], keep: frozenset[str] = KEEP) -> list[str]:
    """止める Scheduler のジョブ名（純粋関数）: keiba-g- で始まり、KEEP 以外で、いま有効なもの。"""
    return sorted(_short(j) for j in jobs if _short(j).startswith(PREFIX) and _short(j) not in keep and j.get("state") == "ENABLED")


def paused_groups(jobs: list[dict[str, Any]], keep: frozenset[str] = KEEP) -> list[str]:
    """すでに止まっている対象（KEEP は数えない）。"""
    return sorted(_short(j) for j in jobs if _short(j).startswith(PREFIX) and _short(j) not in keep and j.get("state") == "PAUSED")


def pause(session: Any, proj: str, loc: str, names: list[str]) -> dict[str, str]:
    res = {}
    for n in names:
        try:
            r = session.post(f"{_base(proj, loc)}/{n}:pause", timeout=HTTP_TIMEOUT)
            res[n] = "PAUSED" if r.status_code == 200 else f"error {r.status_code}: {str(r.text)[:120]}"
        except Exception as exc:  # noqa: BLE001  1 件の失敗で残りを止めないことはしない
            res[n] = f"error {type(exc).__name__}: {str(exc)[:120]}"
    return res


def message(rec: dict[str, Any]) -> str:
    ok = [k for k, v in rec["paused"].items() if v == "PAUSED"]
    bad = {k: v for k, v in rec["paused"].items() if v != "PAUSED"}
    if ok:
        head = f"上限に近いため GCP Jobs の Scheduler を {len(ok)} 件止めました"
    elif bad:
        head = "上限に近いので Scheduler を止めようとしましたが、止められませんでした（要対応）"
    elif rec["already_paused"]:
        head = "上限に近づいています（GCP Jobs の Scheduler は止まっています。止められない課金源で増えていないか確認）"
    elif rec.get("pause_disabled"):
        head = "上限に近づいています（KEIBA_COST_GUARD_PAUSE=0 のため止めていません）"
    else:
        head = "上限に近づいていますが、止める対象の Scheduler（keiba-g-*）が見つかりません（要確認）"
    lines = ["[費用の歯止め] " + head,
             f"今月の累計 {rec['month_to_date']:.0f} 円 / 月末の見込み {rec['projection']:.0f} 円（上限 {rec['budget']:.0f} 円、基準 {rec['limit']:.0f} 円）",
             f"直近 {RECENT_DAYS} 日の平均 {rec['recent_avg']:.0f} 円/日（課金の反映は約 1 日遅れ）"]
    if rec.get("missing_days"):
        lines.append(f"課金データが無い日: {', '.join(rec['missing_days'])}（見込みが低く出ている可能性）")
    if ok:
        lines.append("止めたもの: " + ", ".join(ok))
    if bad:
        lines.append("止められなかったもの: " + ", ".join(f"{k}({v})" for k, v in bad.items()))
    if rec["already_paused"]:
        lines.append(f"止まっているもの: {len(rec['already_paused'])} 件")
    lines.append("再開は人が確かめてから（自動では再開しない）。止められないもの: API・GCS の転送と操作・Artifact Registry・Cloudflare。")
    return "\n".join(lines)


def _already_notified(io: Any, key: str) -> bool:
    try:
        from src.ops.gcs_io import read_json

        r = read_json(io, key)
        return bool(isinstance(r, dict) and r.get("notified"))
    except Exception:  # noqa: BLE001
        return False


def run(*, apply: bool, today: date | None = None, daily: dict[str, float] | None = None, session: Any = None,
        io: Any = None, notify: Callable[[str], Any] | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    e = os.environ if env is None else env
    today = today or datetime.now(JST).date()
    daily = daily if daily is not None else fetch_daily(billing_table(), today)
    a = assess(daily, today, budget_jpy())
    rec: dict[str, Any] = {**a, "applied": False, "paused": {}, "already_paused": [], "targets": [], "notified": False,
                           "checked_at": datetime.now(JST).isoformat(timespec="seconds")}
    alert = a["over"] or bool(a["missing_days"])
    if a["over"]:
        session = session or _session()
        proj, loc = project_id(), location()
        jobs = scheduler_jobs(session, proj, loc)
        rec["targets"] = pause_targets(jobs)
        rec["already_paused"] = paused_groups(jobs)
        rec["pause_disabled"] = str(e.get("KEIBA_COST_GUARD_PAUSE", "1")).strip() == "0"
        if apply and rec["targets"] and not rec["pause_disabled"]:
            rec["paused"] = pause(session, proj, loc, rec["targets"])
            rec["applied"] = True
    if apply:
        if io is None:
            from src.ops.gcs_io import default_io

            io = default_io()
        key = f"ops/cost_guard/{today.strftime('%Y%m%d')}.json"
        # 止めた・止められなかったときはその都度。上限に近いだけ・データが欠けているだけなら 1 日 1 回
        if rec["paused"] or (alert and not _already_notified(io, key)):
            (notify or _notify)(message(rec))
            rec["notified"] = True
        elif alert:
            rec["notified"] = True        # 今日すでに送った（記録を上書きしても送った印を残す）
        try:
            io.put_bytes(key, json.dumps(rec, ensure_ascii=False, indent=1).encode(), content_type="application/json")
        except Exception as exc:  # noqa: BLE001  記録の失敗で歯止めを止めない
            rec["record_error"] = str(exc)[:200]
    return rec


def _notify(text: str) -> None:
    from src.utils.notify import notify_slack

    notify_slack(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="月の GCP 費用の歯止め")
    ap.add_argument("--apply", action="store_true", help="上限に近ければ Scheduler を止めて知らせる（無ければ判定だけ）")
    args = ap.parse_args(argv)
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    try:
        rec = run(apply=args.apply)
    except Exception as exc:  # noqa: BLE001  歯止めが動かなかったことを黙らせない
        if args.apply:
            _notify(f"[費用の歯止め] 判定できませんでした（要対応）: {type(exc).__name__}: {str(exc)[:300]}")
        print(f"cost_guard の失敗: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(rec, ensure_ascii=False))
    bad = [v for v in rec["paused"].values() if v != "PAUSED"]
    return 2 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
