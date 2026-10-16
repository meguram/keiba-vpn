"""
開発者専用 監視ポータル — ポートは config/ports.env（KEIBA_PORT_MONITOR）

既存の FastAPI / Flask / Next.js とは完全独立した
スタンドアロン Flask アプリ。MONITOR_PASSWORD による Cookie 認証で保護する。

起動:
    bash scripts/server/start_monitor.sh
    python3 -m src.monitor.app            # ポートは config/ports.env の KEIBA_PORT_MONITOR
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from src.utils.project_env import load_project_dotenv

load_project_dotenv()

# ── 設定 ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent

from src.config.ports import local_url as _local_url, monitor_port as _monitor_port  # noqa: E402

_secret_key = os.environ.get("MONITOR_SECRET_KEY", "")
if not _secret_key:
    import secrets
    _secret_key = secrets.token_hex(32)

MONITOR_PASSWORD = os.environ.get("MONITOR_PASSWORD", "")
FASTAPI_URL = os.environ.get("KEIBA_FASTAPI_URL") or _local_url("fastapi")
FLASK_URL = os.environ.get("KEIBA_API_URL") or _local_url("flask")
NEXT_URL = _local_url("frontend")
LOG_DIR = ROOT / "logs"

# 環境ごとのサービスポート定義
_SHARED_DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://keiba:keiba@localhost:5432/keiba",
)
ENV_CONFIGS = {
    "dev": {
        "label": "DEV",
        "next_url": _local_url("frontend", "dev"),
        "flask_url": _local_url("flask", "dev"),
        "fastapi_url": _local_url("fastapi", "dev"),
        "mlflow_url": _local_url("mlflow", "dev"),
        "public_url": "https://meguai-dev.tcpexposer.com/",
        "color": "#58a6ff",
        # dev は NEXT_PUBLIC_MOCK=true のためフロントがAPIを呼ばない。DB接続チェック対象外。
        "db_url": None,
        "mock_frontend": True,
    },
    "stg": {
        "label": "STG",
        "next_url": _local_url("frontend", "stg"),
        "flask_url": _local_url("flask", "stg"),
        "fastapi_url": _local_url("fastapi", "stg"),
        "mlflow_url": _local_url("mlflow", "stg"),
        "public_url": "https://meguai-stg.tcpexposer.com/",
        "color": "#3fb950",
        # stg はネイティブ PostgreSQL :5432 / DB=keiba_db_stg（scripts/server/setup_stg.sh 参照）
        "db_url": os.environ.get(
            "STG_DATABASE_URL",
            "postgresql+psycopg://keiba_user:keiba_pass@localhost:5432/keiba_db_stg",
        ),
        "mock_frontend": False,
    },
    "prod": {
        "label": "PROD",
        "next_url": _local_url("frontend", "prod"),
        "flask_url": _local_url("flask", "prod"),
        "fastapi_url": _local_url("fastapi", "prod"),
        "mlflow_url": _local_url("mlflow", "prod"),
        "public_url": None,
        "color": "#d29922",
        "db_url": os.environ.get("PROD_DATABASE_URL", _SHARED_DB_URL),
        "mock_frontend": False,
    },
}

app = Flask(__name__, template_folder="templates")
app.secret_key = _secret_key
app.config["SESSION_COOKIE_SAMESITE"] = "Strict"
app.config["PERMANENT_SESSION_LIFETIME"] = 43200  # 12 時間


@app.context_processor
def _inject_ports() -> dict:
    return {
        "ports": {
            "monitor": _monitor_port(),
            "fastapi": int(FASTAPI_URL.rsplit(":", 1)[-1].split("/")[0]),
            "flask": int(FLASK_URL.rsplit(":", 1)[-1].split("/")[0]),
            "frontend": int(NEXT_URL.rsplit(":", 1)[-1].split("/")[0]),
        }
    }

# ── 認証 ────────────────────────────────────────────────────────────────────


def _is_authenticated() -> bool:
    return session.get("monitor_auth") is True


def require_auth(f):
    from functools import wraps

    @wraps(f)
    def wrapper(*args, **kwargs):
        if not _is_authenticated():
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return wrapper


# ── ページルート ────────────────────────────────────────────────────────────


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form.get("password") == MONITOR_PASSWORD:
            session.permanent = True
            session["monitor_auth"] = True
            return redirect(url_for("dashboard"))
        error = "パスワードが正しくありません"
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@require_auth
def dashboard():
    return render_template("dashboard.html", active="dashboard")


@app.route("/services")
@require_auth
def services():
    return render_template("services.html", active="services")


@app.route("/scraping")
@require_auth
def scraping():
    return render_template("scraping.html", active="scraping")


@app.route("/data")
@require_auth
def data():
    return render_template("data.html", active="data")


@app.route("/system")
@require_auth
def system():
    return render_template("system.html", active="system")


@app.route("/logs")
@require_auth
def logs():
    log_files = sorted(
        [f.name for f in LOG_DIR.glob("*.log") if f.is_file()],
        key=lambda n: (LOG_DIR / n).stat().st_mtime,
        reverse=True,
    ) if LOG_DIR.exists() else []
    return render_template("logs.html", active="logs", log_files=log_files)


@app.route("/git")
@require_auth
def git():
    return render_template("git.html", active="git")


# ── 内部 JSON API（JS ポーリング用）─────────────────────────────────────────


def _http_get(url: str, timeout: float = 5.0) -> tuple[int, Any, float]:
    """(status_code, json_body_or_None, elapsed_ms) を返す。エラー時は (0, None, -1)。"""
    try:
        t0 = time.monotonic()
        resp = requests.get(url, timeout=timeout)
        elapsed = (time.monotonic() - t0) * 1000
        try:
            body = resp.json()
        except Exception:
            body = None
        return resp.status_code, body, round(elapsed, 1)
    except Exception:
        return 0, None, -1.0


def _db_ping(db_url: str) -> dict:
    """PostgreSQL への軽量 ping チェック（接続確立のみ、行数取得なし）。"""
    import re
    if not db_url:
        return {"ok": False, "error": "URL未設定", "elapsed_ms": -1, "dsn": ""}
    # DSN マスク（パスワード部分を *** に置換）—成功・失敗に関わらず返す
    try:
        dsn_masked = re.sub(r"://([^:@/]+):([^@/]+)@", r"://\1:***@", db_url)
    except Exception:
        dsn_masked = "***"
    t0 = time.monotonic()
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(
            db_url,
            connect_args={"connect_timeout": 3},
            pool_pre_ping=True,
            pool_size=1,
            max_overflow=0,
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        elapsed = round((time.monotonic() - t0) * 1000, 1)
        return {"ok": True, "elapsed_ms": elapsed, "dsn": dsn_masked}
    except Exception as exc:
        elapsed = round((time.monotonic() - t0) * 1000, 1)
        return {"ok": False, "error": str(exc)[:200], "elapsed_ms": elapsed, "dsn": dsn_masked}


def _check_model_data_mode(fastapi_url: str) -> dict:
    """
    stg/prod 環境向け: 予測・追走難度・レース質の各エンドポイントが
    モックデータか実データかをサンプルチェックする。

    代表レース ID は DB / GCS から最近の有効レースを取得する。
    Returns:
      {
        "predictions":          {"mode": "mock"|"real"|"no_data"|"error", "race_id": ...},
        "tracking_difficulty":  {"mode": "mock"|"real"|"no_data"|"error", "race_id": ...},
        "race_quality":         {"mode": "mock"|"real"|"no_data"|"error", "race_id": ...},
        "sample_race_id":       "202602010801",
      }
    """
    import json as _json

    # ① 代表レース ID を取得（race-list から直近の開催日の最初のレース）
    sample_race_id = None
    try:
        code, body, _ = _http_get(
            f"{fastapi_url}/api/scrape-dates?picker_past_days=30", timeout=5.0
        )
        if code == 200 and isinstance(body, dict):
            dates = body.get("dates") or []
            # 直近日から試す
            for d in dates[:5]:
                rc, rb, _ = _http_get(f"{fastapi_url}/api/race-list/{d}", timeout=5.0)
                if rc == 200 and isinstance(rb, dict):
                    races = rb.get("races") or []
                    if races:
                        sample_race_id = races[0].get("race_id")
                        if sample_race_id:
                            break
    except Exception:
        pass

    if not sample_race_id:
        return {
            "predictions": {"mode": "no_data", "race_id": None},
            "tracking_difficulty": {"mode": "no_data", "race_id": None},
            "race_quality": {"mode": "no_data", "race_id": None},
            "sample_race_id": None,
        }

    def _check_endpoint(path: str) -> str:
        """エンドポイントを叩き mock/real/no_data/error を返す。"""
        try:
            code, body, _ = _http_get(f"{fastapi_url}{path}", timeout=8.0)
            if code == 404:
                return "no_data"
            if code != 200 or not isinstance(body, dict):
                return "error"
            if body.get("_mock") is True:
                return "mock"
            # ステータスが not_found / no_entries の場合は no_data
            if body.get("status") in ("not_found", "no_entries", "not_precomputed"):
                return "no_data"
            return "real"
        except Exception:
            return "error"

    pred_mode = _check_endpoint(f"/api/race/{sample_race_id}/predictions")
    td_mode   = _check_endpoint(f"/api/race/{sample_race_id}/tracking-difficulty")
    rq_mode   = _check_endpoint(f"/api/race-quality/race?race_id={sample_race_id}")

    return {
        "predictions":         {"mode": pred_mode, "race_id": sample_race_id},
        "tracking_difficulty": {"mode": td_mode,   "race_id": sample_race_id},
        "race_quality":        {"mode": rq_mode,   "race_id": sample_race_id},
        "sample_race_id": sample_race_id,
    }


def _check_env(env_key: str) -> dict:
    """指定環境（dev/stg/prod）の各サービス死活を返す。"""
    cfg = ENV_CONFIGS[env_key]
    next_code, _, next_ms = _http_get(cfg["next_url"])
    flask_code, flask_body, flask_ms = _http_get(f"{cfg['flask_url']}/api/v1/health")
    fastapi_code, fastapi_body, fastapi_ms = _http_get(f"{cfg['fastapi_url']}/api/health")
    mlflow_code, _, mlflow_ms = _http_get(cfg["mlflow_url"])
    is_mock = cfg.get("mock_frontend", False)
    if is_mock:
        db_result = {"ok": None, "mock": True}  # ok=None → モック構成（チェック対象外）
    else:
        db_result = _db_ping(cfg.get("db_url") or "")
    all_ok = next_code in (200, 304) and flask_code == 200 and fastapi_code == 200

    # stg/prod のみモデルデータモードをチェック（dev はフロントモックなので省略）
    model_data_mode: dict | None = None
    if env_key in ("stg", "prod") and not is_mock:
        try:
            model_data_mode = _check_model_data_mode(cfg["fastapi_url"])
        except Exception:
            model_data_mode = None

    from src.config.deployment import keiba_env

    return {
        "env": env_key,
        "is_local": env_key == keiba_env(),     # この監視サーバが動いているマシンの環境（他の環境のサーバは別のマシンで動くので、停止は異常ではない）
        "label": cfg["label"],
        "color": cfg["color"],
        "public_url": cfg["public_url"],
        "mock_frontend": is_mock,
        "next": {
            "url": cfg["next_url"],
            "status": next_code,
            "ok": next_code in (200, 304),
            "elapsed_ms": next_ms,
        },
        "flask": {
            "url": cfg["flask_url"],
            "status": flask_code,
            "ok": flask_code == 200,
            "elapsed_ms": flask_ms,
            "body": flask_body,
        },
        "fastapi": {
            "url": cfg["fastapi_url"],
            "status": fastapi_code,
            "ok": fastapi_code == 200,
            "elapsed_ms": fastapi_ms,
            "body": fastapi_body,
        },
        "mlflow": {
            "url": cfg["mlflow_url"],
            "status": mlflow_code,
            "ok": mlflow_code == 200,
            "elapsed_ms": mlflow_ms,
        },
        "db": db_result,
        "all_ok": all_ok,
        "model_data_mode": model_data_mode,
    }


@app.route("/api/internal/status")
@require_auth
def internal_status():
    # FastAPI は全環境共通
    fastapi_code, fastapi_body, fastapi_ms = _http_get(f"{FASTAPI_URL}/api/health")

    # 全環境チェック（並列化せず順次、タイムアウト 5s × 3env × 2svc = max 30s だが通常即応）
    envs = {key: _check_env(key) for key in ("dev", "stg", "prod")}
    from src.config.deployment import keiba_env
    local_env = keiba_env() if keiba_env() in envs else "stg"

    # スクレイピングキュー概要
    queue_summary: dict = {}
    try:
        _, queue_data, _ = _http_get(f"{FASTAPI_URL}/api/scrape-jobs", timeout=8.0)
        if isinstance(queue_data, dict):
            queue_summary = {
                "pending": queue_data.get("pending_count", 0),
                "running": queue_data.get("running_count", 0),
                "done": queue_data.get("done_count", 0),
                "failed": queue_data.get("failed_count", 0),
                "schema_validation_failures": queue_data.get("schema_validation_failures", 0),
            }
    except Exception:
        pass

    # Git HEAD
    git_head = _git_head_short()

    return jsonify(
        {
            "timestamp": datetime.now().isoformat(),
            "services": {
                "fastapi": {
                    "status": fastapi_code,
                    "ok": fastapi_code == 200,
                    "elapsed_ms": fastapi_ms,
                    "body": fastapi_body,
                },
                # 後方互換のため旧 flask/nextjs キーも残す（このマシンの環境の値。以前は dev 固定で、stg のホストで常に「停止」と出ていた）
                "flask": envs[local_env]["flask"],
                "nextjs": envs[local_env]["next"],
            },
            "local_env": local_env,
            "envs": envs,
            "queue": queue_summary,
            "git_head": git_head,
        }
    )


@app.route("/api/internal/system")
@require_auth
def internal_system():
    try:
        import psutil

        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        swap = psutil.swap_memory()

        # keiba 関連プロセス
        keiba_procs = []
        for p in psutil.process_iter(["pid", "name", "cmdline", "cpu_percent", "memory_info"]):
            try:
                cmd = " ".join(p.info["cmdline"] or [])
                if any(kw in cmd for kw in ("main.py", "uvicorn", "next", "monitor")):
                    keiba_procs.append(
                        {
                            "pid": p.info["pid"],
                            "name": p.info["name"],
                            "cmd": cmd[:120],
                            "cpu_pct": p.info["cpu_percent"],
                            "mem_mb": round(
                                (p.info["memory_info"].rss if p.info["memory_info"] else 0)
                                / 1024
                                / 1024,
                                1,
                            ),
                        }
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        return jsonify(
            {
                "cpu_pct": cpu,
                "memory": {
                    "used_gb": round(mem.used / 1e9, 2),
                    "total_gb": round(mem.total / 1e9, 2),
                    "pct": mem.percent,
                },
                "disk": {
                    "used_gb": round(disk.used / 1e9, 1),
                    "total_gb": round(disk.total / 1e9, 1),
                    "pct": disk.percent,
                },
                "swap": {
                    "used_gb": round(swap.used / 1e9, 2),
                    "total_gb": round(swap.total / 1e9, 2),
                    "pct": swap.percent,
                },
                "processes": keiba_procs,
            }
        )
    except ImportError:
        return jsonify({"error": "psutil が未インストールです: pip install psutil>=5.9"}), 500


def _git_head_short() -> dict:
    try:
        log = subprocess.check_output(
            ["git", "log", "-1", "--format=%H|%an|%ae|%ai|%s"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        parts = log.split("|", 4)
        return {
            "hash": parts[0][:8] if parts else "",
            "author": parts[1] if len(parts) > 1 else "",
            "email": parts[2] if len(parts) > 2 else "",
            "date": parts[3] if len(parts) > 3 else "",
            "message": parts[4] if len(parts) > 4 else "",
        }
    except Exception:
        return {}


@app.route("/api/internal/git")
@require_auth
def internal_git():
    head = _git_head_short()

    # ブランチ
    branch = ""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
    except Exception:
        pass

    # dirty ファイル
    dirty: list[str] = []
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode()
        dirty = [line for line in out.splitlines() if line.strip()]
    except Exception:
        pass

    # 最終 git pull ログ
    pull_log = ""
    pull_log_path = LOG_DIR / "git_pull_hourly.log"
    if pull_log_path.exists():
        try:
            lines = pull_log_path.read_text(errors="replace").splitlines()
            pull_log = "\n".join(lines[-30:])
        except Exception:
            pass

    return jsonify(
        {
            "branch": branch,
            "head": head,
            "dirty": dirty,
            "dirty_count": len(dirty),
            "pull_log": pull_log,
        }
    )


@app.route("/api/internal/git/pull", methods=["POST"])
@require_auth
def internal_git_pull():
    """git pull origin <branch> を実行して結果を返す。"""
    import time as _time_mod
    started_at = _time_mod.time()
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=ROOT, stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
    except Exception:
        branch = "main"

    try:
        result = subprocess.run(
            ["git", "pull", "origin", branch],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        output = "\n".join(filter(None, [stdout, stderr]))
        success = result.returncode == 0
    except subprocess.TimeoutExpired:
        output = "タイムアウト（60秒）"
        success = False
    except Exception as e:
        output = str(e)
        success = False

    elapsed = round(_time_mod.time() - started_at, 1)

    # ログに記録
    try:
        log_path = LOG_DIR / "git_pull.log"
        from datetime import datetime as _dt
        entry = f"[{_dt.now().strftime('%Y-%m-%d %H:%M:%S')}] [monitor-manual] branch={branch} ok={success} ({elapsed}s)\n{output}\n"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except Exception:
        pass

    # pull 後の HEAD を取得
    head_after = _git_head_short()

    return jsonify({
        "ok": success,
        "branch": branch,
        "output": output,
        "elapsed_s": elapsed,
        "head": head_after,
    })


@app.route("/api/internal/logs")
@require_auth
def internal_logs():
    filename = request.args.get("file", "")
    if not filename:
        return jsonify({"error": "file パラメータが必要です"}), 400

    # パストラバーサル防止
    target = (LOG_DIR / filename).resolve()
    if not str(target).startswith(str(LOG_DIR.resolve())):
        return jsonify({"error": "不正なファイルパスです"}), 403
    if not target.exists():
        return jsonify({"error": f"{filename} が見つかりません"}), 404

    try:
        lines = target.read_text(errors="replace").splitlines()
        tail = lines[-200:]
        return jsonify(
            {
                "file": filename,
                "total_lines": len(lines),
                "lines": tail,
                "mtime": datetime.fromtimestamp(target.stat().st_mtime).isoformat(),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/internal/coverage/month")
@require_auth
def internal_coverage_month():
    """年・月ごとのカレンダーデータ（日別ステータス）を返す。"""
    try:
        year = int(request.args.get("year", datetime.now().year))
        month = int(request.args.get("month", datetime.now().month))
    except ValueError:
        return jsonify({"error": "year/month は整数で指定してください"}), 400

    import calendar

    _, days_in_month = calendar.monthrange(year, month)

    # FastAPI から coverage-calendar を取得（年単位キャッシュ）
    _, cal_data, _ = _http_get(
        f"{FASTAPI_URL}/api/coverage-calendar?year={year}", timeout=15.0
    )

    # 日付 → coverage データの辞書を作成
    date_map: dict[str, dict] = {}
    if isinstance(cal_data, dict):
        for entry in cal_data.get("dates", []):
            date_map[entry["date"]] = entry

    days = []
    for day in range(1, days_in_month + 1):
        date_str = f"{year}{month:02d}{day:02d}"
        entry = date_map.get(date_str)

        if entry is None:
            status = "gray"  # レースなし（データなし）
            total = 0
            pct = 0.0
            per_cat = {}
        elif entry.get("total_races", 0) == 0:
            status = "gray"
            total = 0
            pct = 0.0
            per_cat = {}
        elif entry.get("pct", 0) >= 100.0:
            status = "green"
            total = entry["total_races"]
            pct = entry["pct"]
            per_cat = entry.get("per_cat", {})
        else:
            status = "red"
            total = entry["total_races"]
            pct = round(entry.get("pct", 0), 1)
            per_cat = entry.get("per_cat", {})

        days.append({
            "date": date_str,
            "day": day,
            "status": status,
            "total_races": total,
            "pct": pct,
            "per_cat": per_cat,
        })

    categories = cal_data.get("categories", []) if isinstance(cal_data, dict) else []

    return jsonify({
        "year": year,
        "month": month,
        "days_in_month": days_in_month,
        "first_weekday": calendar.monthrange(year, month)[0],  # 0=月曜
        "days": days,
        "categories": categories,
    })


@app.route("/api/internal/coverage/date-matrix")
@require_auth
def internal_coverage_date_matrix():
    """指定日の race×category マトリクスを返す（FastAPI /api/date-race-matrix を中継）。"""
    date = request.args.get("date", "")
    if not date:
        return jsonify({"error": "date パラメータが必要です (YYYYMMDD)"}), 400

    _, data, elapsed = _http_get(
        f"{FASTAPI_URL}/api/date-race-matrix?date={date}", timeout=30.0
    )
    if data is None:
        return jsonify({"error": f"FastAPI からデータ取得失敗 (date={date})"}), 502

    # カテゴリのラベル付け（表示名）
    CAT_LABELS = {
        "race_shutuba":             "出馬表",
        "race_shutuba_meta":        "レース情報",
        "race_index":               "レース指数",
        "race_paddock":             "パドック",
        "race_odds":                "オッズ",
        "race_result_on_time":      "速報結果",
        "race_result_on_time_payoff": "速報払戻",
        "race_result_on_time_lap":  "速報ラップ",
        "race_result_on_time_corner": "速報通過順",
        "race_result":              "確定結果",
        "race_result_meta":         "確定情報",
        "race_result_payoff":       "確定払戻",
        "race_result_track":        "馬場情報",
        "race_result_corner":       "通過順(DB)",
        "race_result_lap_times":    "ラップ(DB)",
        "race_result_lap":          "個別ラップ",
        "race_barometer":           "馬場指数",
    }

    CAT_URLS = {
        "race_shutuba":             "https://race.netkeiba.com/race/shutuba.html?race_id={rid}",
        "race_shutuba_meta":        "https://race.netkeiba.com/race/shutuba.html?race_id={rid}",
        "race_result_on_time":      "https://race.netkeiba.com/race/result.html?race_id={rid}",
        "race_result_on_time_payoff": "https://race.netkeiba.com/race/result.html?race_id={rid}",
        "race_result_on_time_lap":  "https://race.netkeiba.com/race/result.html?race_id={rid}",
        "race_result_on_time_corner": "https://race.netkeiba.com/race/result.html?race_id={rid}",
        "race_result":              "https://db.netkeiba.com/race/{rid}/",
        "race_result_meta":         "https://db.netkeiba.com/race/{rid}/",
        "race_result_payoff":       "https://db.netkeiba.com/race/{rid}/",
        "race_result_track":        "https://db.netkeiba.com/race/{rid}/",
        "race_result_corner":       "https://db.netkeiba.com/race/{rid}/",
        "race_result_lap_times":    "https://db.netkeiba.com/race/{rid}/",
        "race_result_lap":          "https://db.netkeiba.com/race/{rid}/",
    }

    categories_with_meta = [
        {
            "key": cat,
            "label": CAT_LABELS.get(cat, cat),
            "url_template": CAT_URLS.get(cat, ""),
        }
        for cat in data.get("categories", [])
    ]

    return jsonify({
        "date": date,
        "race_ids": data.get("race_ids", []),
        "categories": categories_with_meta,
        "race_meta": data.get("race_meta", {}),
        "matrix": data.get("matrix", {}),
        "elapsed_ms": elapsed,
    })


@app.route("/api/internal/scraping")
@require_auth
def internal_scraping():
    _, jobs, _ = _http_get(f"{FASTAPI_URL}/api/scrape-jobs", timeout=10.0)
    _, calendar, _ = _http_get(
        f"{FASTAPI_URL}/api/coverage-calendar?year={datetime.now().year}", timeout=10.0
    )
    return jsonify({"jobs": jobs, "calendar": calendar})


GCS_BUCKET = os.environ.get("GCS_BUCKET", "magu-keiba-horse-racing-ai")
GCS_PREFIX = "chuou/data/preprocessed/netkeiba/pc"
GCS_OTHERS = "chuou/data/others"






# ── ページ品質チェック結果 ──────────────────────────────────────────────────────


@app.route("/api/internal/page-quality")
@require_auth
def internal_page_quality():
    """
    最新のページ品質チェック結果を返す。
    data/local/page_quality/latest.json から読み込む。
    クエリパラメータ:
      ?date=YYYY-MM-DD  : 特定日の履歴を取得（省略時は latest）
    """
    from pathlib import Path

    base = Path(__file__).parent.parent.parent
    result_dir = base / "data" / "local" / "page_quality"
    date_param = request.args.get("date")

    if date_param:
        target = result_dir / "history" / f"{date_param}.json"
    else:
        target = result_dir / "latest.json"

    if not target.exists():
        return jsonify({
            "run_at": None,
            "summary_tag": "unknown",
            "total": 0,
            "safe_count": 0,
            "caution_count": 0,
            "out_count": 0,
            "results": [],
            "note": "結果ファイルがまだ存在しません。cron または手動実行を行ってください。",
        })

    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        # 利用可能な履歴日付リストを付与
        history_dates = sorted(
            [p.stem for p in (result_dir / "history").glob("*.json")],
            reverse=True,
        )[:30]
        data["history_dates"] = history_dates
        return jsonify(data)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/internal/page-quality/run", methods=["POST"])
@require_auth
def internal_page_quality_run():
    """
    ページ品質チェックをオンデマンド実行する（管理用）。
    """
    import subprocess
    import sys

    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "src.monitor.page_quality_check"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).parent.parent.parent),
            text=True,
        )
        return jsonify({"status": "started", "pid": proc.pid})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── データストアモニタリング ────────────────────────────────────────────────────

def _pg_status() -> dict:
    """PostgreSQL の接続チェックと主要テーブル行数を返す。"""
    try:
        from sqlalchemy import create_engine, text
        from src.db.session import database_url
        url = database_url()
        engine = create_engine(url, connect_args={"connect_timeout": 3}, pool_pre_ping=True)

        TABLES = [
            "races", "entries", "race_results",
            "horses", "jockeys", "trainers",
            "horse_stats_snapshot", "jockey_stats_snapshot", "trainer_stats_snapshot",
            "prediction_results", "scrape_runs",
            "race_lap_times",
        ]
        rows: dict[str, int | None] = {}
        with engine.connect() as conn:
            # 存在するテーブルのみカウント
            existing = {
                r[0] for r in conn.execute(
                    text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
                )
            }
            for tbl in TABLES:
                if tbl not in existing:
                    rows[tbl] = None
                    continue
                try:
                    rows[tbl] = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
                except Exception:
                    rows[tbl] = None

            # 最終更新（scrape_runs が存在する場合）
            last_etl: str | None = None
            if "scrape_runs" in existing:
                try:
                    r = conn.execute(
                        text("SELECT MAX(started_at) FROM scrape_runs")
                    ).scalar()
                    last_etl = str(r) if r else None
                except Exception:
                    pass

        # DSN をマスク
        try:
            dsn_masked = "@".join(
                ["***:***".join(url.split("@")[0].split("//", 1)[-1].split(":")[-2:][0:1]),
                 url.split("@")[1]]
            ) if "@" in url else url
        except Exception:
            dsn_masked = "***"

        return {
            "ok": True,
            "dsn_masked": dsn_masked,
            "tables": rows,
            "last_etl_at": last_etl,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:300]}


def _sqlite_status(db_path: Path, label: str, key_tables: list[str]) -> dict:
    """SQLite DB の存在チェックと指定テーブル行数を返す。"""
    if not db_path.exists():
        return {"ok": False, "path": str(db_path), "label": label, "error": "ファイルなし"}
    try:
        import sqlite3
        stat = db_path.stat()
        conn = sqlite3.connect(str(db_path), timeout=3)
        existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        rows: dict[str, int | None] = {}
        for tbl in key_tables:
            if tbl not in existing:
                rows[tbl] = None
                continue
            try:
                rows[tbl] = conn.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()[0]
            except Exception:
                rows[tbl] = None
        conn.close()
        return {
            "ok": True,
            "path": str(db_path),
            "label": label,
            "size_mb": round(stat.st_size / 1024 / 1024, 1),
            "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            "tables": rows,
        }
    except Exception as exc:
        return {"ok": False, "path": str(db_path), "label": label, "error": str(exc)[:200]}


def _parquet_status(features_dir: Path) -> list[dict]:
    """data/local/features/ 配下のブロックをスキャンして統計を返す。"""
    blocks: list[dict] = []
    if not features_dir.exists():
        return blocks

    try:
        import pyarrow.parquet as pq
    except ImportError:
        return [{"block": "pyarrow なし", "ok": False}]

    def scan_block(path: Path, label: str) -> dict:
        files = list(path.rglob("*.parquet"))
        if not files:
            return {"block": label, "ok": True, "files": 0, "rows": 0, "size_mb": 0, "mtime": None}
        total_rows = 0
        total_size = 0
        latest_mtime = 0.0
        for f in files:
            try:
                meta = pq.read_metadata(f)
                total_rows += meta.num_rows
            except Exception:
                pass
            st = f.stat()
            total_size += st.st_size
            latest_mtime = max(latest_mtime, st.st_mtime)
        return {
            "block": label,
            "ok": True,
            "files": len(files),
            "rows": total_rows,
            "size_mb": round(total_size / 1024 / 1024, 1),
            "mtime": datetime.fromtimestamp(latest_mtime).isoformat(timespec="seconds") if latest_mtime else None,
        }

    for block_dir in sorted(features_dir.iterdir()):
        if not block_dir.is_dir():
            continue
        sub_dirs = [d for d in sorted(block_dir.iterdir()) if d.is_dir() and not d.name.startswith("_")]
        parquet_files_direct = list(block_dir.glob("*.parquet"))
        if sub_dirs and not parquet_files_direct:
            # ブロックが年別 or シャード別のサブディレクトリ構造
            blocks.append(scan_block(block_dir, block_dir.name))
        else:
            blocks.append(scan_block(block_dir, block_dir.name))

    return blocks


def _jt_stats_manifest(features_dir: Path) -> dict | None:
    """jockey_trainer_stats マニフェストを読む。"""
    manifest_path = features_dir / "jockey_trainer_stats" / "_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "generated_at": m.get("generated_at"),
            "years_all_computed": m.get("years_all_computed"),
            "years_jt_features": m.get("years_jt_race_features_output"),
        }
    except Exception:
        return None


@app.route("/cron")
@require_auth
def cron_page():
    return render_template("cron.html", active="cron")


@app.route("/api/internal/cron-status")
@require_auth
def internal_cron_status():
    """全 cron ジョブの状態（待機中/実行中）・最終実行・結果・失敗の原因。作りは src/monitor/cron_status.py。"""
    from src.monitor.cron_status import build_status

    return jsonify(build_status())


@app.route("/data-stores")
@require_auth
def page_data_stores():
    return render_template("data_stores.html")


DOCS_DIR = ROOT / "docs" / "decisions"


def _specs_file_groups() -> dict:
    """docs/decisions/ の .md ファイルを AREA / DEC / その他 に分類して返す。"""
    if not DOCS_DIR.exists():
        return {"master": [], "area": [], "dec": []}
    all_files = sorted(f.name for f in DOCS_DIR.glob("*.md") if f.name != "DEC-TEMPLATE.md")
    return {
        "master": [f for f in all_files if not f.startswith("AREA-") and not f.startswith("DEC-")],
        "area":   [f for f in all_files if f.startswith("AREA-")],
        "dec":    [f for f in all_files if f.startswith("DEC-")],
    }


@app.route("/specs")
@require_auth
def specs():
    return render_template(
        "specs.html",
        active="specs",
        groups=_specs_file_groups(),
        content=None,
        current_file=None,
    )


@app.route("/specs/<path:filename>")
@require_auth
def specs_file(filename: str):
    target = (DOCS_DIR / filename).resolve()
    if not str(target).startswith(str(DOCS_DIR.resolve())):
        return "不正なパスです", 403
    if not target.exists() or target.suffix != ".md":
        return "ファイルが見つかりません", 404

    import markdown as md_lib
    raw = target.read_text(encoding="utf-8")
    content_html = md_lib.markdown(
        raw,
        extensions=["tables", "fenced_code", "toc", "nl2br"],
    )

    return render_template(
        "specs.html",
        active="specs",
        groups=_specs_file_groups(),
        content=content_html,
        current_file=filename,
    )


@app.route("/api/internal/data-stores")
@require_auth
def internal_data_stores():
    """
    PostgreSQL / SQLite / Parquet の各データストア状態を返す。

    Returns:
        {
          "postgresql": { ok, dsn_masked, tables, last_etl_at, error? },
          "sqlite": [ { ok, label, path, size_mb, mtime, tables } ],
          "parquet": [ { block, ok, files, rows, size_mb, mtime } ],
          "jt_manifest": { generated_at, years_all_computed, years_jt_features },
          "timestamp": "ISO"
        }
    """
    base = ROOT

    # PostgreSQL
    pg = _pg_status()

    # SQLite 群
    bloodline_db = base / "data" / "research" / "bloodline" / "bloodline.db"
    mlflow_db    = base / "mlflow" / "runs" / "mlflow.db"

    sqlite_stores = [
        _sqlite_status(bloodline_db, "bloodline.db（血統・レース索引）", [
            "races", "race_results", "horse_names",
            "pedigree_cats", "race_result_slim",
            "horse_bms", "stallion_lineage",
        ]),
        _sqlite_status(mlflow_db, "MLflow tracking DB", [
            "experiments", "runs", "registered_models",
        ]),
    ]

    # ML warehouse catalog（存在すれば）
    ml_catalog = base / "data" / "ml" / "warehouse" / "sqlite" / "catalog.sqlite3"
    if ml_catalog.exists():
        sqlite_stores.append(
            _sqlite_status(ml_catalog, "ML Warehouse catalog", ["horse_catalog"])
        )

    # Parquet 特徴量
    features_dir = base / "data" / "local" / "features"
    parquet_blocks = _parquet_status(features_dir)

    # JT マニフェスト
    jt_manifest = _jt_stats_manifest(features_dir)

    return jsonify({
        "postgresql": pg,
        "sqlite": sqlite_stores,
        "parquet": parquet_blocks,
        "jt_manifest": jt_manifest,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })


# ── エントリポイント ─────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="keiba-vpn 監視ポータル")
    parser.add_argument("--port", type=int, default=_monitor_port())
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    if not MONITOR_PASSWORD:
        print("[monitor] 警告: MONITOR_PASSWORD が未設定です。.env に設定してください。")

    print(f"[monitor] 起動: http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
