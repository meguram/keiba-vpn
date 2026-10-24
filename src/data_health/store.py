"""チェック結果の環境別管理（保存・履歴・前回比・全環境の一覧・他 PC の結果の取り込み）。

  <base>/<env>/latest.{json,html} / scrape_plan.json / report_*.json / history.jsonl
  <base>/index.{html,json}        … 全環境の最新結果の一覧

<env> は評価プロファイル。実行環境と違う場合（dev PC で stg 要件を評価）は ``stg@dev`` のように分け、
本物の stg の結果と混ざらないようにする。dev は GCP に接続できないため、学習PC／VPS の結果は
``latest.json`` を持ち込んで ``--import`` で取り込む（GCP には触れない）。
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config.data_paths import ROOT
from src.data_health import report as R
from src.data_health.spec import ENVS

DEFAULT_BASE = ROOT / "data" / "local" / "meta" / "data_health"
STALE_HOURS = 48
HISTORY_SHOWN = 10


def base_dir(out_dir: str | Path | None = None) -> Path:
    """保存ルート。引数 > 環境変数 DATA_HEALTH_OUT_DIR > 既定。"""
    import os

    return Path(out_dir or os.environ.get("DATA_HEALTH_OUT_DIR") or DEFAULT_BASE)


def env_key(profile: str, actual: str) -> str:
    return profile if profile == actual else f"{profile}@{actual}"


def git_commit(root: Path = ROOT) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, timeout=3)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def build_meta(profile: str, actual: str, root: Path = ROOT) -> dict[str, str]:
    from src.scraper import schemas

    return {"env": profile, "actual_env": actual, "key": env_key(profile, actual), "host": socket.gethostname(),
            "commit": git_commit(root), "schema_fingerprint": schemas.schema_fingerprint(),
            "schema_version": schemas.SCHEMA_VERSION}


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ── 前回比 ───────────────────────────────────────────────────────────────

def _fkey(f: dict) -> str:
    return f"{f['area']}|{f['message'].split(':')[0].split('（')[0]}"


def compute_diff(prev: dict | None, cur: dict) -> dict | None:
    if not prev or "summary" not in prev:
        return None
    pf = {_fkey(f): f for f in prev.get("findings", [])}
    cf = {_fkey(f): f for f in cur.get("findings", [])}
    pc, cc = prev["summary"]["counts"], cur["summary"]["counts"]
    return {
        "previous_at": prev.get("generated_at"), "overall_before": prev["summary"]["overall"],
        "counts_delta": {k: cc.get(k, 0) - pc.get(k, 0) for k in ("fail", "warn", "info")},
        "new": [f["message"] for k, f in cf.items() if k not in pf][:30],
        "resolved": [f["message"] for k, f in pf.items() if k not in cf][:30],
        "changed": [{"before": pf[k]["message"], "after": f["message"]} for k, f in cf.items()
                    if k in pf and pf[k]["message"] != f["message"]][:30],
    }


# ── 履歴 ────────────────────────────────────────────────────────────────

def history_line(report: dict) -> dict[str, Any]:
    rc = report.get("race_coverage", {})
    return {"generated_at": report["generated_at"], "overall": report["summary"]["overall"],
            "counts": report["summary"]["counts"], "planned_jobs": report["summary"]["planned_jobs"],
            "missing_races": sum(rc.get("gap_total", {}).values()), "host": report.get("meta", {}).get("host", ""),
            "commit": report.get("meta", {}).get("commit", ""),
            "schema_fingerprint": report.get("meta", {}).get("schema_fingerprint", "")}


def read_history(env_dir: Path, limit: int = HISTORY_SHOWN) -> list[dict]:
    p = env_dir / "history.jsonl"
    if not p.is_file():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows[-limit:]


def _append_history(env_dir: Path, report: dict) -> None:
    line = history_line(report)
    if any(h.get("generated_at") == line["generated_at"] for h in read_history(env_dir, 10_000)):
        return
    with (env_dir / "history.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


# ── 保存・取り込み・一覧 ──────────────────────────────────────────────────

def _key_of(report: dict) -> str:
    meta = report.get("meta") or {}
    return meta.get("key") or env_key(report["env"], meta.get("actual_env", report["env"]))


def save(report: dict, out_dir: str | Path | None = None) -> dict[str, Path]:
    """最新結果として保存し、前回比・履歴・全環境の一覧を更新する。"""
    base = base_dir(out_dir)
    env_dir = base / _key_of(report)
    report["diff"] = compute_diff(load_json(env_dir / "latest.json"), report)
    paths = R.write_outputs(report, env_dir)
    _append_history(env_dir, report)
    idx = rebuild_index(base)
    paths["index"], paths["dashboard"] = idx["html"], idx["dashboard"]
    return paths


def import_report(path: str | Path, out_dir: str | Path | None = None) -> dict[str, Any]:
    """他 PC で得た latest.json を環境別ディレクトリへ取り込む。古いものは履歴にだけ残す。"""
    src = Path(path)
    if src.is_dir():                        # 結果ディレクトリごと渡す（race_keys.csv など副ファイルも一緒に取り込める）
        src = src / "latest.json"
    data = load_json(src)
    if not data or data.get("env") not in ENVS or "summary" not in data or "generated_at" not in data:
        raise ValueError(f"{path} はデータヘルスの latest.json ではありません（env / summary / generated_at が必要）")
    base = base_dir(out_dir)
    env_dir = base / _key_of(data)
    current = load_json(env_dir / "latest.json")
    newer = not current or datetime.fromisoformat(data["generated_at"]) >= datetime.fromisoformat(current["generated_at"])
    if newer:
        data["diff"] = compute_diff(current, data)
        paths = R.write_outputs(data, env_dir)
        for name in ("race_keys.csv", "access_restriction.json"):          # ダッシュボードの掘り下げ・制限バナーに使う副ファイル
            if (src.parent / name).is_file():
                shutil.copy2(src.parent / name, env_dir / name)
        if (src.parent / "scrape_runs").is_dir():
            shutil.copytree(src.parent / "scrape_runs", env_dir / "scrape_runs", dirs_exist_ok=True)
    else:
        env_dir.mkdir(parents=True, exist_ok=True)
        stamp = data["generated_at"].replace(":", "").replace("-", "")[:15]
        paths = {"history": env_dir / f"report_{stamp}.json"}
        paths["history"].write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    _append_history(env_dir, data)
    rebuild_index(base)
    return {"key": _key_of(data), "latest_updated": newer, "paths": paths}


def collect_entries(base: Path) -> list[dict]:
    entries = []
    for d in sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []:
        rep = load_json(d / "latest.json")
        if not rep or "summary" not in rep:
            continue
        entries.append({"key": d.name, "report": rep, "history": read_history(d)})
    order = {k: i for i, k in enumerate(ENVS)}
    entries.sort(key=lambda e: (order.get(e["key"].split("@")[0], 99), "@" in e["key"], e["key"]))
    return entries


def rebuild_index(base: Path) -> dict[str, Path]:
    base.mkdir(parents=True, exist_ok=True)
    entries = collect_entries(base)
    summary = [{"key": e["key"], "env": e["report"]["env"], "overall": e["report"]["summary"]["overall"],
                "counts": e["report"]["summary"]["counts"], "generated_at": e["report"]["generated_at"],
                "meta": e["report"].get("meta", {})} for e in entries]
    paths = {"json": base / "index.json", "html": base / "index.html"}
    paths["json"].write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    paths["html"].write_text(R.render_index(entries), encoding="utf-8")
    from src.data_health import dashboard

    paths["dashboard"] = dashboard.write_dashboard(base)["html"]       # 全環境を切り替えて見られるダッシュボード（サーバ不要）
    return paths
