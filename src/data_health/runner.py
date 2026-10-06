"""データ存在チェックの実行本体。環境（KEIBA_ENV）ごとに要件を切り替えて 1 つのレポート dict を返す。"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.config.data_paths import ROOT
from src.config.deployment import keiba_env
from src.data_health import checks as C
from src.data_health import coverage as V
from src.data_health import keytable as KT
from src.scraper import schema_violations
from src.data_health import schema_check, store
from src.data_health.config import Settings
from src.data_health.plan import build_plan
from src.data_health.spec import HORSE_CATEGORIES, RACE_CATEGORIES, level_overrides

JST = timezone(timedelta(hours=9))
DEFAULT_SINCE = date(2020, 1, 1)      # 要件書・row 派生カテゴリの保存範囲（2020-2026）に合わせる
UPCOMING_DAYS = 14                    # daily-race-lists が先読みする日数（馬の「出走予定」確認と直近カレンダー確認用）


def _rank(status: str) -> int:
    return {"fail": 3, "warn": 2, "info": 1}.get(status, 0)


def run_health(*, env: str | None = None, storage: Any = None, now: datetime | None = None,
               since: date | None = None, until: date | None = None, root: Path = ROOT,
               infra: bool = True, include_optional_in_plan: bool = False, dev_root: Path | None = None,
               horses: bool = True, settings: Settings | None = None, actual_env: str | None = None,
               ledger_dir: Path | None = None, progress: Any = None) -> dict[str, Any]:
    """``env`` は評価する要件プロファイル、``actual_env`` は実際の実行環境（GCP 遮断・接続可否を決める）。"""
    actual = actual_env or keiba_env()
    env = env or (settings.env if settings else None) or actual
    levels = dict(settings.levels) if settings else {}
    with level_overrides(env, levels):
        cfg = settings or Settings(env=env)
        if ledger_dir is None and settings is not None:
            ledger_dir = store.base_dir(cfg.out_dir) / store.env_key(env, actual) / "ledger"
        return _run(env, actual, storage, now, since, until, root, infra, include_optional_in_plan, dev_root, horses,
                    cfg, levels, ledger_dir, progress)


def _run(env: str, actual: str, storage: Any, now: datetime | None, since: date | None, until: date | None,
         root: Path, infra: bool, include_optional_in_plan: bool, dev_root: Path | None, horses: bool,
         cfg: Settings, levels: dict[str, str], ledger_dir: Path | None, progress: Any = None) -> dict[str, Any]:
    def step(phase: str, done: int | None = None, total: int | None = None, detail: str = "") -> None:
        if progress is not None:
            progress(phase, done, total, detail)

    now = now or datetime.now(JST)
    since = since or cfg.since
    until = until or cfg.until
    today = now.date()
    if since is None:
        since = today - timedelta(days=90) if env == "dev" else DEFAULT_SINCE
    # 評価期間の終わりは「実行日の前日」。当日以降のレース（結果・オッズ等がまだ揃わない）は 2020 年以降の完全性に含めない。
    until = until or today - timedelta(days=1)
    until_ymd = until.strftime("%Y%m%d")
    ahead = max(until, today + timedelta(days=UPCOMING_DAYS))          # 馬の出走予定・直近カレンダーの確認は先まで見る
    years = [str(y) for y in range(since.year, ahead.year + 1)]
    if storage is None:
        from src.scraper.storage import HybridStorage

        storage = HybridStorage(base_dir=str(root))
    if dev_root is None and actual == "dev":
        from src.scraper.dev_store import dev_mock_root

        dev_root = dev_mock_root(root)

    cal = V.load_calendar(since, ahead)
    keytable = KT.KeyTable(ledger_dir / "race_keys.json" if ledger_dir else None)
    keytable.update_from_calendar(cal.dates, cal.race_meta)
    cats = [c.name for c in RACE_CATEGORIES if c.level[env] != "skip"]
    for needed in ("race_shutuba", "race_result"):
        if needed not in cats:
            cats.append(needed)
    step("格納状況の一覧を取得（GCS の list）")
    present = V.collect_presence(storage, cats, years)
    beyond = {rid for rid, row in keytable.rows.items() if (row.get("date") or "") > until_ymd}      # 評価期間より後のレース
    health, vstats = schema_check.validate_scope(
        env, storage, present, mode=cfg.validate, sample=cfg.schema_sample, budget=cfg.validate_budget,
        workers=cfg.validate_workers, ledger_dir=ledger_dir, keytable=keytable, skip_keys=beyond, progress=progress)
    step("集計・計画の作成")
    keytable.save()                                            # 検証の副産物（日付・場・R・レース名）を保存
    universe_all = V.build_universe(cal, present, years, infer=(actual != "dev" and env != "dev"), keytable=keytable)
    universe = {rid: r for rid, r in universe_all.items() if not (r.date and r.date > until_ymd)}
    race_cov = V.race_coverage(env, universe, present, now, health=health, max_gaps=10**9,          # 再取得計画は全件から作る
                               unknown_state="unvalidated" if cfg.validate == "full" else "present")
    race_status = race_cov.pop("race_status")
    cat_names = [c["name"] for c in race_cov["categories"]]
    race_cov["freshness"] = V.freshness(present, now)
    horse_cov = (V.horse_coverage(env, storage, universe_all, now, past_days=cfg.horse_past_days,
                                  future_days=cfg.horse_future_days, races_max=cfg.horse_races_max)
                 if horses and any(h.level[env] != "skip" for h in HORSE_CATEGORIES)
                 else {"categories": [], "gaps": [], "horses": 0, "window": {}})

    data_years = sorted(y for y, keys in present.get("race_result", {}).items() if keys)
    missing_weekends = V.weekend_coverage(cal, today)
    checks = (C.run_infra_checks(env, storage, dev_root=dev_root, missing_weekends=missing_weekends, root=root,
                                 actual_env=actual, levels=levels) if infra else [])
    artifacts = C.check_artifacts(env, data_years, root)
    # dev PC はスクレイピングしない（stg 要件を評価する dry-run でも計画は出さない）
    plan = build_plan("dev" if actual == "dev" else env, race_cov, horse_cov, cal.anomalies, now,
                      include_optional=include_optional_in_plan)
    plan["env"] = env
    race_cov["gaps_total_count"] = len(race_cov["gaps"])
    race_cov["gaps"] = race_cov["gaps"][:REPORT_GAPS_MAX]                 # レポート(JSON/HTML)には先頭だけ載せる

    report = {
        "env": env, "generated_at": now.isoformat(timespec="seconds"), "meta": store.build_meta(env, actual, root),
        "settings": {"levels": levels, "since": since.isoformat(), "until": until.isoformat(), "warnings": cfg.warnings,
                     "fail_on": cfg.fail_on, "sources": cfg.sources,
                     "schema_sample": cfg.schema_sample, "validate": cfg.validate, "validate_budget": cfg.validate_budget,
                     "horse_window": {"past_days": cfg.horse_past_days, "future_days": cfg.horse_future_days,
                                      "races_max": cfg.horse_races_max}},
        "range": {"since": since.isoformat(), "until": until.isoformat(), "years": years},
        "checks": checks, "artifacts": artifacts,
        "calendar": {"dates": len(cal.dates), "no_race_dates": len(cal.no_race_dates), "anomalies": cal.anomalies,
                     "source_dirs": cal.source_dirs, "missing_weekends": missing_weekends},
        "race_coverage": race_cov, "horse_coverage": horse_cov, "plan": plan,
    }
    report["schema"] = schema_check.schema_report(env, storage, present, cfg.schema_sample, health=health, stats=vstats,
                                                  mode=cfg.validate)
    report["race_ids"] = {c: {"missing": v["missing"], "invalid": v["invalid"], "invalid_detail": v["invalid_detail"],
                              "unvalidated": v["unvalidated"]} for c, v in race_cov.pop("race_ids").items()}
    report["schema_log"] = schema_violations.summarize(root, days=30)      # 保存時に引っかかった記録（どの項目がどの値か）
    report["key_table"] = {"races": len(universe), "with_date": sum(1 for r in universe.values() if r.date),
                           "rows_in_ledger": len(keytable.rows), "categories": cat_names}
    cols, rows = KT.csv_table(universe, keytable, cat_names, race_status)
    report["_side"] = {"race_keys": {"columns": cols, "rows": rows}}      # CSV 用（latest.json には書かない）
    report["findings"] = build_findings(report)
    report["completeness"] = assess_completeness(report, since, cfg, until, today)
    report["summary"] = summarize(report)
    return report


COMPLETE_SINCE = date(2020, 1, 1)
REPORT_GAPS_MAX = 5000


def assess_completeness(r: dict[str, Any], since: date, cfg: Settings, until: date, today: date) -> dict[str, Any]:
    """「2020 年以降のすべてのデータポイントが、スキーマに適合した状態で揃っているか」の合否と、満たしていない理由。

    対象は重要度が cfg.complete_levels（既定 required / recommended）のカテゴリ・派生データ・インフラ。
    全件を検証していない（off/sample）、検証が途中、評価期間が 2020 年より後、も「完全とは言えない」理由にする。
    """
    lv = set(cfg.complete_levels)
    reasons: list[dict[str, Any]] = []

    def add(code: str, message: str, count: int = 1) -> None:
        reasons.append({"code": code, "message": message, "count": count})

    if since > COMPLETE_SINCE:
        add("range", f"評価期間の開始が {COMPLETE_SINCE} より後（{since}）")
    if until < today - timedelta(days=1):
        add("range", f"評価期間の終了が実行日の前日（{today - timedelta(days=1)}）より前（{until}）")
    if cfg.validate != "full":
        add("validate_mode", f"健全性を全件では検証していない（DATA_HEALTH_VALIDATE={cfg.validate}）")
    rc = r["race_coverage"]
    levels = {c["name"]: c["level"] for c in rc["categories"]}
    for key, label in (("gap_total", "不足"), ("invalid_total", "スキーマ不適合"), ("unvalidated_total", "未検証")):
        for cat, n in rc.get(key, {}).items():
            if n and levels.get(cat) in lv:
                add(key.split("_")[0], f"{cat}: {label} {n} レース", n)
    stats = (r.get("schema") or {}).get("stats") or {}
    if cfg.validate == "full" and stats and not stats.get("complete"):
        add("validate_incomplete", "健全性の検証が途中（再実行で続きから検証）")
    for c in r["horse_coverage"].get("categories", []):
        if c["missing"] and c["level"] in lv:
            add("horse", f"{c['name']}: 直近/出走予定の馬 {c['missing']} 頭が不足", c["missing"])
    for a in r["artifacts"]:
        if a["status"] in ("fail", "warn") and a["level"] in lv:
            add("artifact", f"{a['id']} {a['label']}: {a['detail']}")
    for c in r["checks"]:
        if c["status"] == "fail":
            add("infra", f"{c['label']}: {c['detail']}")
    in_range = [a for a in r["calendar"]["anomalies"] if a["date"] <= r["range"]["until"].replace("-", "")]
    if in_range and r["env"] != "dev":
        add("calendar", f"race_lists の不完全 {len(in_range)} 日", len(in_range))
    return {"complete": not reasons, "since": since.isoformat(), "levels": sorted(lv), "reasons": reasons,
            "scope": f"{since} 〜 {r['range']['until']}"}


def build_findings(r: dict[str, Any]) -> list[dict]:
    """人が最初に読む「要対応」一覧（重大なものから）。"""
    out: list[dict] = []
    env = r["env"]
    for c in r["checks"]:
        if c["status"] in ("fail", "warn"):
            out.append({"severity": c["status"], "area": "インフラ", "message": f"{c['label']}: {c['detail']}", "hint": c["hint"]})
    for a in r["artifacts"]:
        if a["status"] in ("fail", "warn"):
            out.append({"severity": a["status"], "area": "派生データ", "message": f"{a['label']}: {a['detail']}", "hint": a["hint"]})
    rc = r["race_coverage"]
    for cat in rc["categories"]:
        name, lvl = cat["name"], cat["level"]
        total_missing = rc["gap_total"].get(name, 0)
        if not total_missing:
            continue
        worst = sorted(((p, v[name]["missing"]) for p, v in rc["by_period"].items() if name in v and v[name]["missing"]),
                       key=lambda x: -x[1])[:5]
        sev = {"required": "fail", "recommended": "warn"}.get(lvl, "info")
        out.append({"severity": sev, "area": "レース", "message":
                    f"{name}（{cat['label']}）: {total_missing} レース不足 / 多い期間 " + ", ".join(f"{p}:{n}" for p, n in worst),
                    "hint": "plan（scrape_plan.json）の spec をキューへ投入"})
    for c in r["horse_coverage"].get("categories", []):
        if c["missing"]:
            sev = {"required": "fail", "recommended": "warn"}.get(c["level"], "info")
            out.append({"severity": sev, "area": "馬", "message":
                        f"{c['name']}: 直近/出走予定の {c['expected']} 頭中 {c['missing']} 頭が不足", "hint": "horse ジョブを投入"})
    out += schema_check.schema_findings(env, r["schema"])
    log = r.get("schema_log") or {}
    rej = (log.get("decisions") or {}).get("rejected", 0)
    if rej:
        top = (log.get("groups") or [{}])[0]
        vals = ", ".join(f"{v}×{n}" for v, n in top.get("values", [])[:2])
        out.append({"severity": "warn", "area": "スキーマ", "message":
                    f"保存時に拒否: 直近30日で {rej} 回（隔離中 {sum(log.get('quarantined', {}).values())} 件）。最多: {top.get('category')} "
                    f"{top.get('field')} [{top.get('rule')}] {vals}",
                    "hint": "python -m src.scraper.schema_violations summary ／ 隔離データは data/local/quarantine/"})
    for a in r["calendar"]["anomalies"][:20]:
        if env == "dev" and a["problem"].startswith("race_lists 不完全"):     # dev のモックは 1 日 8 レースと小さく、取得もできないので対象外
            continue
        out.append({"severity": "info" if env == "dev" else "warn", "area": "カレンダー",
                    "message": f"{a['date']}: {a['problem']}", "hint": "race_list を再取得"})
    out.sort(key=lambda f: -_rank(f["severity"]))
    return out


def summarize(r: dict[str, Any]) -> dict[str, Any]:
    counts = {"fail": 0, "warn": 0, "info": 0}
    for f in r["findings"]:
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1
    overall = "fail" if counts["fail"] else "warn" if counts["warn"] else "ok"
    return {"overall": overall, "counts": counts, "planned_jobs": r["plan"]["counts"]["jobs"],
            "complete": (r.get("completeness") or {}).get("complete")}
