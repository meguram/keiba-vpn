"""インフラのヘルスチェックとローカル派生物（特徴量・血統・モデル等）の存在チェック。

各チェックは ``{id, label, status, detail, hint, level}`` を返す。status は
ok / warn / fail / info / skip。level（required/recommended/optional）は環境ごとの重要度で、
問題があったときの status（fail / warn / info）を決める。
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config.data_paths import ROOT
from src.data_health.spec import ARTIFACTS, FEATURE_ROOTS, GCS_OBJECTS, ArtifactSpec

_PROBLEM_STATUS = {"required": "fail", "recommended": "warn", "optional": "info"}

INFRA_CHECK_IDS = (
    "env.keiba_env", "env.gcs_bucket", "env.gcs_credentials", "env.dev_secret_key", "dev.gcp_blocked", "dev.mock", "dev.mock_samples",
    "gcs.connect", "infra.db", "infra.redis", "infra.queue", "infra.disk", "data.upcoming_calendar",
)


def check(cid: str, label: str, status: str, detail: str = "", hint: str = "", level: str = "required") -> dict:
    return {"id": cid, "label": label, "status": status, "detail": detail, "hint": hint, "level": level}


def _problem(level: str) -> str:
    return _PROBLEM_STATUS.get(level, "info")


def _short(e: BaseException) -> str:
    return (str(e).strip().splitlines() or [type(e).__name__])[0][:200]


# ── インフラ ──────────────────────────────────────────────────────────────

def check_env_config(env: str) -> list[dict]:
    out = []
    raw = os.environ.get("KEIBA_ENV", "").strip()
    if not raw:
        out.append(check("env.keiba_env", "KEIBA_ENV", "warn",
                         "未設定（コード上は prod 扱い）", "dev なら .env に KEIBA_ENV=dev、stg/prod は起動時に指定"))
    else:
        out.append(check("env.keiba_env", "KEIBA_ENV", "ok", f"{raw}（解釈: {env}）"))
    if env in ("stg", "prod"):
        from src.config.gcp_credentials import gcp_credentials_available

        bucket = os.environ.get("GCS_BUCKET", "").strip()
        out.append(check("env.gcs_bucket", "GCS_BUCKET", "ok" if bucket else "fail",
                         "設定あり" if bucket else "未設定", "" if bucket else ".env.<env> に GCS_BUCKET を設定"))
        adc = Path.home().joinpath(".config/gcloud/application_default_credentials.json").is_file()
        if gcp_credentials_available():
            out.append(check("env.gcs_credentials", "GCS 認証情報", "ok", "GCS_* サービスアカウントあり"))
        elif adc:
            out.append(check("env.gcs_credentials", "GCS 認証情報", "warn", "GCS_* 無し（ADC にフォールバック）",
                             "本番は GCS_* サービスアカウントを .env.<env> に設定", "recommended"))
        else:
            out.append(check("env.gcs_credentials", "GCS 認証情報", "fail", "GCS_PRIVATE_KEY 未設定・ADC 無し",
                             "VPS 用サービスアカウントを発行して .env.<env> に転記（T-003）"))
        key = os.environ.get("DEV_SECRET_KEY", "")
        out.append(check("env.dev_secret_key", "DEV_SECRET_KEY（32文字以上）", "ok" if len(key) >= 32 else "fail",
                         "OK" if len(key) >= 32 else f"{len(key)} 文字", "開発者ログインの署名鍵（T-010）"))
    return out


def check_dev_guard() -> dict:
    from src.config.gcp_guard import GcpAccessForbidden, assert_gcp_allowed

    try:
        assert_gcp_allowed("data_health probe")
    except GcpAccessForbidden:
        return check("dev.gcp_blocked", "dev の GCP 遮断", "ok", "GcpAccessForbidden が有効（GCS/Cloud SQL/Tasks/BigQuery を遮断）")
    return check("dev.gcp_blocked", "dev の GCP 遮断", "fail", "遮断が効いていない", "KEIBA_ENV=dev を確認")


def check_dev_store(root: Path) -> dict:
    cats = sorted(d.name for d in root.iterdir() if d.is_dir()) if root.is_dir() else []
    n = sum(1 for _ in root.rglob("*.json")) if root.is_dir() else 0
    if n == 0:
        return check("dev.mock", "dev モックデータ", "fail", f"{root} が空", "make dev-mock")
    return check("dev.mock", "dev モックデータ", "ok", f"{n} ファイル / {len(cats)} カテゴリ（{root}）")


DEV_SAMPLE_MIN = 2
DEV_SAMPLE_MIN_OVERRIDE = {"jra_cushion": 1, "requirement_row_trace": 1}


def dev_sample_categories() -> list[str]:
    """dev モックに最低限ほしいカテゴリ = スキーマ定義済み全カテゴリ + スキーマ未定義でも生成している生成物。"""
    from src.scraper import schemas

    extra = {"race_predictions", "tracking_difficulty", "final_odds_prediction", "finish_order_prediction",
             "race_performance", "jra_cushion"}
    return sorted(set(schemas.SCHEMAS) | extra)


def check_dev_samples(root: Path, page_ref: Path | None = None) -> dict:
    """全データポイントのモックサンプルが揃っているか（スキーマ適合は生成時に検証済み）。"""
    from src.scraper.dev_store import DevStore
    from src.scraper.storage import HybridStorage

    if page_ref is None:
        from src.config.data_paths import page_reference_root
        page_ref = page_reference_root()
    store = DevStore(root)
    short = []
    for cat in dev_sample_categories():
        id_type = HybridStorage.CATEGORY_MAP.get(cat, "race")
        n = len(list((page_ref / cat).glob("*.json"))) if id_type == "local_only" else len(store.list_keys(cat, id_type))
        need = DEV_SAMPLE_MIN_OVERRIDE.get(cat, DEV_SAMPLE_MIN)
        if n < need:
            short.append(f"{cat}({n}/{need})")
    total = len(dev_sample_categories())
    if short:
        return check("dev.mock_samples", "dev モックの全カテゴリ", "fail", f"{total - len(short)}/{total} カテゴリ。不足: {', '.join(short[:8])}",
                     "make dev-mock")
    return check("dev.mock_samples", "dev モックの全カテゴリ", "ok", f"{total} カテゴリすべてにサンプルあり（各 {DEV_SAMPLE_MIN} 件以上）")


def check_gcs(env: str, storage: Any) -> list[dict]:
    out = []
    if not storage.gcs_enabled:
        return [check("gcs.connect", "GCS 接続", "fail", "接続できない（GCS_BUCKET / 認証情報を確認）",
                      "VPS は GCS_* 未設定だと到達不可（T-003）")]
    try:
        bucket = storage._get_bucket()
        t0 = time.time()
        ok = bucket.exists(timeout=8)
        ms = int((time.time() - t0) * 1000)
        if not ok:
            return [check("gcs.connect", "GCS 接続", "fail", f"バケット {bucket.name} が存在しない/権限なし")]
        first = list(bucket.list_blobs(prefix=storage.GCS_BASE + "/", max_results=1, timeout=8))
        out.append(check("gcs.connect", "GCS 接続", "ok" if first else "warn",
                         f"{bucket.name}（{ms}ms）" + ("" if first else " / データ prefix が空")))
        for spec in GCS_OBJECTS:
            lvl = spec["level"][env]
            if lvl == "skip":
                continue
            exists = bucket.blob(spec["blob"]).exists(timeout=8)
            out.append(check(f"gcs.{spec['id']}", spec["label"], "ok" if exists else _problem(lvl),
                             "あり" if exists else "無し", "" if exists else spec["hint"], lvl))
    except Exception as e:
        out.append(check("gcs.connect", "GCS 接続", "fail", _short(e)))
    return out


def check_database(env: str) -> dict:
    level = "recommended" if env == "dev" else "required"
    try:
        from sqlalchemy import text

        from src.db.session import init_engine

        engine = init_engine()
        t0 = time.time()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        backend = os.environ.get("KEIBA_DB_BACKEND", "") or "DATABASE_URL"
        return check("infra.db", "PostgreSQL", "ok", f"{backend} 接続OK（{int((time.time() - t0) * 1000)}ms）", level=level)
    except Exception as e:
        hint = "make db-up（dev）" if env == "dev" else "Cloud SQL インスタンス未作成（T-016）/ 接続設定を確認"
        return check("infra.db", "PostgreSQL", _problem(level), _short(e), hint, level)


def check_redis(env: str) -> dict:
    level = "optional" if env == "dev" else "required"
    url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    try:
        import redis

        redis.Redis.from_url(url, socket_connect_timeout=1, socket_timeout=1).ping()
        return check("infra.redis", "Redis", "ok", f"{url.split('@')[-1]} PING OK", level=level)
    except ImportError:
        return check("infra.redis", "Redis", _problem(level), "redis パッケージ未導入", "pip install redis", level)
    except Exception as e:
        return check("infra.redis", "Redis", _problem(level), _short(e), "docker compose で Redis を起動", level)


def check_queue(env: str, root: Path = ROOT) -> dict:
    if os.environ.get("KEIBA_QUEUE_BACKEND", "").strip().lower() == "cloud_tasks" and env != "dev":
        return check("infra.queue", "スクレイプキュー", "info", "Cloud Tasks バックエンド（ローカルキューは対象外）", level="optional")
    p = root / "data" / "queue" / "scrape_queue.json"
    if not p.is_file():
        return check("infra.queue", "スクレイプキュー", "info", "scrape_queue.json なし", level="optional")
    try:
        jobs = json.loads(p.read_text(encoding="utf-8")).get("jobs", [])
    except (OSError, ValueError) as e:
        return check("infra.queue", "スクレイプキュー", "warn", f"読み込み失敗: {_short(e)}", level="recommended")
    cnt: dict[str, int] = {}
    for j in jobs:
        cnt[j.get("status", "?")] = cnt.get(j.get("status", "?"), 0) + 1
    failed = cnt.get("failed", 0)
    detail = " / ".join(f"{k}={v}" for k, v in sorted(cnt.items())) or "ジョブなし"
    return check("infra.queue", "スクレイプキュー", "warn" if failed else "ok", detail,
                 "failed を確認（schema_validation は /monitor）" if failed else "", "recommended")


def check_disk(root: Path = ROOT) -> dict:
    u = shutil.disk_usage(root)
    free_gb, pct = u.free / 1e9, u.free / u.total * 100
    bad = free_gb < 10 or pct < 10
    return check("infra.disk", "ディスク空き", "warn" if bad else "ok", f"空き {free_gb:.0f}GB（{pct:.0f}%） / 全体 {u.total / 1e9:.0f}GB",
                 "キャッシュ上限を下げる/不要データを削除" if bad else "", "recommended")


def check_upcoming_calendar(env: str, missing_weekends: list[str]) -> dict:
    level = "skip" if env == "dev" else "recommended"
    if level == "skip":
        return check("data.upcoming_calendar", "直近7日の race_lists", "skip", "dev は評価しない", level=level)
    if missing_weekends:
        return check("data.upcoming_calendar", "直近7日の race_lists", "warn",
                     f"土日の記録なし: {', '.join(missing_weekends)}", "python -m src.scraper.run race-list（daily-race-lists cron）", level)
    return check("data.upcoming_calendar", "直近7日の race_lists", "ok", "直近の土日は記録あり（開催あり/なし）", level=level)


def apply_overrides(results: list[dict], levels: dict[str, str]) -> list[dict]:
    """``DATA_HEALTH_LEVELS`` / ``DATA_HEALTH_SKIP`` をインフラチェックの結果へ反映する。"""
    for c in results:
        lv = levels.get(c["id"])
        if not lv:
            continue
        if lv == "skip":
            c.update(status="skip", detail=c["detail"] + "（設定で対象外）", level="skip")
        elif c["status"] in ("fail", "warn", "info"):
            c.update(status=_problem(lv), level=lv)
    return results


def run_infra_checks(env: str, storage: Any, *, dev_root: Path | None, missing_weekends: list[str],
                     root: Path = ROOT, actual_env: str | None = None, levels: dict[str, str] | None = None
                     ) -> list[dict]:
    actual_env = actual_env or env
    if actual_env != env:
        # 実行環境と評価プロファイルが違う（例: dev PC で stg 要件を評価）と、接続系の結果は意味を持たない
        return [check("infra.profile", "インフラ疎通", "skip",
                      f"実行環境 {actual_env} と評価プロファイル {env} が異なるため省略（データの存在のみ評価）",
                      level="optional")]
    out = check_env_config(env)
    if env == "dev":
        out.append(check_dev_guard())
        out.append(check_dev_store(dev_root) if dev_root else check("dev.mock", "dev モックデータ", "skip"))
        if dev_root:
            out.append(check_dev_samples(dev_root))
    else:
        out += check_gcs(env, storage)
    out += [check_database(env), check_redis(env), check_queue(env, root), check_disk(root),
            check_upcoming_calendar(env, missing_weekends)]
    return apply_overrides(out, levels or {})


# ── ローカル派生物 ────────────────────────────────────────────────────────

def _expand(patterns: tuple[str, ...], year: str | None) -> list[str]:
    out = []
    for p in patterns:
        variants = [p.replace("{F}", f) for f in FEATURE_ROOTS] if "{F}" in p else [p]
        for v in variants:
            out.append(v.replace("{Y}", year) if year else v)
    return out


def _glob(root: Path, patterns: list[str]) -> list[Path]:
    found: list[Path] = []
    for pat in patterns:
        found += [p for p in root.glob(pat) if p.is_file()]
    return found


def check_artifact(spec: ArtifactSpec, env: str, years: list[str], root: Path = ROOT) -> dict:
    level = spec.level.get(env, "skip")
    base = {"id": spec.id, "label": spec.label, "group": spec.group, "level": level, "hint": spec.hint}
    if level == "skip":
        return {**base, "status": "skip", "detail": "この環境では評価しない", "found": 0, "missing_years": []}
    if spec.per_year:
        have, missing = [], []
        for y in years:
            (have if _glob(root, _expand(spec.patterns, y)) else missing).append(y)
        ok = not missing and bool(years)
        detail = (f"{len(have)}/{len(years)} 年分" + (f"（不足: {_fmt_years(missing)}）" if missing else "")) if years else "対象年なし"
        return {**base, "status": "ok" if ok or not years else _problem(level), "detail": detail,
                "found": len(have), "missing_years": missing}
    files = _glob(root, _expand(spec.patterns, None))
    uniq = {p.resolve() for p in files}
    ok = len(uniq) >= spec.min_files
    roots = sorted({f for f in FEATURE_ROOTS if any(str(p).startswith(str(root / f)) for p in files)})
    newest = max((p.stat().st_mtime for p in uniq), default=0)
    detail = f"{len(uniq)} ファイル" + (f"（更新 {datetime.fromtimestamp(newest).strftime('%Y-%m-%d')}）" if newest else "")
    if roots:
        detail += f" @ {', '.join(roots)}"
    return {**base, "status": "ok" if ok else _problem(level), "detail": detail if uniq else "無し",
            "found": len(uniq), "missing_years": []}


def _fmt_years(ys: list[str]) -> str:
    return ", ".join(ys) if len(ys) <= 6 else f"{ys[0]}–{ys[-1]} の {len(ys)} 年"


def check_artifacts(env: str, years: list[str], root: Path = ROOT) -> list[dict]:
    return [check_artifact(s, env, years, root) for s in ARTIFACTS]
