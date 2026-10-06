"""データ格納状況のダッシュボード（サーバ不要の静的 HTML + データファイル）。

  <保存ルート>/dashboard.html       … 画面（固定。ブラウザで開くだけ。file:// でも動く）
  <保存ルート>/dashboard_data.js    … 全環境の最新結果（チェックを実行するたびに更新）
  <保存ルート>/run_status.js        … 実行中の進捗（チェック・検証・スクレイピング中に数秒ごとに更新）

画面は約 30 秒ごとにデータファイルを読み直し（進捗は約 3 秒ごと）、実行結果の更新が自動で反映される。
専用サーバは持たない。ブラウザから GCS を直接は見ない（認証情報を持たない）ので、「最新の状態」は
データチェック（``python -m src.data_health``）を実行した時点のもの。cron で定期実行すれば、画面が自動で追従する。

環境の見せ方: dev / stg(= prod と同じ GCS) / prod（VPS からの確認結果があれば）/ ``x@y``（y で x の要件を評価した dry-run）。
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

JST = timezone(timedelta(hours=9))
TEMPLATE = Path(__file__).with_name("dashboard.html")
STATUS_CODES = {"h": "healthy", "p": "present", "i": "invalid", "u": "unvalidated", "m": "missing", "w": "pending", "n": "na"}
_CODE_OF = {v: k for k, v in STATUS_CODES.items()}
ORDER = {"dev": 0, "stg": 1, "prod": 2}
LABELS = {
    "dev": "dev（開発PC・モック）",
    "stg": "stg ＝ prod（学習PC・同一の GCS）",
    "prod": "prod（VPS からの確認結果）",
}
MAX_PLAN_SPECS = 300
MAX_INVALID_DETAIL = 2000
RUN_STALE_SEC = 600


def env_label(key: str) -> str:
    if key in LABELS:
        return LABELS[key]
    if "@" in key:
        want, actual = key.split("@", 1)
        return f"{want} の要件を {actual} で評価（dry-run）"
    return key


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _races(env_dir: Path) -> dict[str, Any] | None:
    p = env_dir / "race_keys.csv"
    if not p.is_file():
        return None
    with p.open(encoding="utf-8-sig", newline="") as f:
        rd = csv.reader(f)
        cols = next(rd, [])
        n_fixed = cols.index("key_confidence") + 1 if "key_confidence" in cols else 10
        rows = []
        for r in rd:
            rows.append(r[:n_fixed] + [_CODE_OF.get(v, v[:1] if v else "") for v in r[n_fixed:]])
    return {"columns": cols, "fixed": n_fixed, "rows": rows}


def _history(env_dir: Path, limit: int = 30) -> list[dict]:
    p = env_dir / "history.jsonl"
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out[-limit:]


def _scrape_runs(env_dir: Path, limit: int = 8) -> list[dict]:
    d = env_dir / "scrape_runs"
    out = []
    for p in sorted(d.glob("*.json"), reverse=True)[:limit] if d.is_dir() else []:
        rec = _read_json(p) or {}
        rounds = rec.get("rounds") or []
        out.append({"file": p.name, "started_at": rec.get("started_at"), "finished_at": rec.get("finished_at"),
                    "exit_code": rec.get("exit_code"), "execute": rec.get("execute"), "resume": rec.get("resume"),
                    "rounds": len(rounds),
                    "selected_jobs": sum((r.get("selected") or {}).get("jobs", 0) for r in rounds),
                    "statuses": [((r.get("outcome") or {}).get("statuses")) for r in rounds if r.get("outcome")],
                    "rejections": sum((r.get("schema_rejections") or {}).get("records", 0) for r in rounds),
                    "restriction": rec.get("restriction"), "final": rec.get("final")})
    return out


def env_payload(key: str, env_dir: Path) -> dict[str, Any] | None:
    rep = _read_json(env_dir / "latest.json")
    if not isinstance(rep, dict) or "summary" not in rep:
        return None
    rc = rep.get("race_coverage", {})
    plan = rep.get("plan", {})
    invalid_detail: dict[str, dict[str, list[str]]] = {}
    n = 0
    for cat, g in (rep.get("race_ids") or {}).items():
        d = {rid: v for rid, v in (g.get("invalid_detail") or {}).items()}
        for rid in list(d)[: max(0, MAX_INVALID_DETAIL - n)]:
            invalid_detail.setdefault(cat, {})[rid] = d[rid]
            n += 1
    report = {k: rep.get(k) for k in ("env", "generated_at", "meta", "settings", "summary", "completeness", "findings", "checks", "artifacts",
                                      "calendar", "horse_coverage", "schema", "schema_log", "key_table", "diff", "range")}
    report["horse_coverage"] = {k: v for k, v in (rep.get("horse_coverage") or {}).items() if k != "gaps"}
    report["calendar"] = {**(rep.get("calendar") or {}), "anomalies": (rep.get("calendar") or {}).get("anomalies", [])[:30]}
    report["race_coverage"] = {k: rc.get(k) for k in ("categories", "by_year", "by_period", "universe", "freshness", "gap_total",
                                                       "invalid_total", "unvalidated_total")}
    report["plan"] = {"counts": plan.get("counts"), "notes": plan.get("notes", [])[:10], "specs": plan.get("specs", [])[:MAX_PLAN_SPECS],
                      "total": len(plan.get("specs", []))}
    return {"key": key, "label": env_label(key), "order": ORDER.get(key.split("@")[0], 9) + (0.5 if "@" in key else 0),
            "report": report, "races": _races(env_dir), "invalid_detail": invalid_detail, "history": _history(env_dir),
            "restriction": _read_json(env_dir / "access_restriction.json"), "scrape_runs": _scrape_runs(env_dir)}


def build_payload(base: Path) -> dict[str, Any]:
    envs = []
    for d in sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []:
        pl = env_payload(d.name, d)
        if pl:
            envs.append(pl)
    envs.sort(key=lambda e: (e["order"], e["key"]))
    return {"generated_at": datetime.now(JST).isoformat(timespec="seconds"), "status_codes": STATUS_CODES, "envs": envs}


def write_dashboard(base: Path) -> dict[str, Path]:
    """画面（固定）とデータを書く。チェックを実行するたびに呼ばれる。"""
    base.mkdir(parents=True, exist_ok=True)
    paths = {"html": base / "dashboard.html", "data": base / "dashboard_data.js"}
    paths["html"].write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    payload = json.dumps(build_payload(base), ensure_ascii=False, separators=(",", ":"))
    tmp = paths["data"].with_suffix(".js.tmp")
    tmp.write_text("window.__DATA_HEALTH__=" + payload + ";\n", encoding="utf-8")
    tmp.replace(paths["data"])                       # 読み込み中の画面が壊れたファイルを掴まないよう、置き換えで更新
    if not (base / "run_status.js").exists():
        _write_status(base, {})
    return paths


# ── 実行中の進捗（画面が約 3 秒ごとに読む）───────────────────────────────────────

def _write_status(base: Path, status: dict[str, Any]) -> None:
    (base / "run_status.json").write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")
    tmp = base / "run_status.js.tmp"
    tmp.write_text("window.__RUN_STATUS__=" + json.dumps(status, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    tmp.replace(base / "run_status.js")


def write_run_status(base: Path, key: str, phase: str, done: int | None = None, total: int | None = None, detail: str = "",
                     state: str = "running") -> None:
    """実行中の段階を記録する（例: 健全性の検証 1200/24000）。失敗しても本処理は止めない。"""
    try:
        base.mkdir(parents=True, exist_ok=True)
        cur = _read_json(base / "run_status.json") or {}
        now = datetime.now(JST).isoformat(timespec="seconds")
        prev = cur.get(key) or {}
        cur[key] = {"state": state, "phase": phase, "done": done, "total": total, "detail": detail, "updated_at": now,
                    "started_at": prev.get("started_at") if prev.get("state") == "running" and state == "running" else now}
        _write_status(base, cur)
    except Exception:  # noqa: BLE001
        pass


def finish_run_status(base: Path, key: str, phase: str = "完了", detail: str = "") -> None:
    write_run_status(base, key, phase, detail=detail, state="idle")
