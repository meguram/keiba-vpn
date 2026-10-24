"""stg（学習PC）: データチェック → 不足データの特定 → スクレイピング → 再チェック。

``python -m src.data_health`` が「どのデータが足りない／壊れているか」を調べて再取得の計画（scrape_plan.json）を作る。
このスクリプトはその計画を実際に実行する側で、1 回の実行で次を順に行う。

  1. 事前確認  … KEIBA_ENV=stg、GCS 接続、netkeiba 認証情報、他のワーカーが動いていないか、アクセス一時停止中でないか
  2. チェック  … データチェックを実行し、最新の計画を得る（``--plan`` で保存済みの計画を使うことも可）
  3. 絞り込み  … 実行場所・状態(不足/スキーマ不適合)・カテゴリ・期間・推定の有無・件数で対象を選ぶ
  4. 確認      … 対象の内訳と、取得リクエスト数・所要時間の目安を表示（**既定はここまで = ドライラン**）
  5. 実行      … キューへ投入（既存を上書きしない smart_skip。スキーマ不適合だけ overwrite）→ ワーカーで処理
  6. 再チェック … データチェックをやり直して（台帳により更新分だけ）、残りを確認。進捗があれば次のラウンドへ

安全策: ドライランが既定／``--execute`` でも KEIBA_ENV=stg 以外は拒否（dev は GCP に書けない。prod は ``--allow-prod``）／
アクセス制限(block)の疑いで一時停止されたら中断／ラウンドで進捗が無ければ停止／件数が多い・上書きがあるときは確認が必要（``--yes``）。

  python -m src.scripts.scraping.scrape_from_health_plan                      # ドライラン: 何を取得するかを表示するだけ
  python -m src.scripts.scraping.scrape_from_health_plan --execute --max-jobs 200
  python -m src.scripts.scraping.scrape_from_health_plan --execute --rounds 5 --status missing

**スクレイピング中のエラーには敏感に対応する。「ページが存在しない」以外のエラーは、基本的にアクセス制限とみなし**、
検知したら（HTTP 400 以上（404 を含む）／接続エラー・タイムアウト／アクセス制限のページ。既定は 1 回で即時）
**直ちに全リクエストを止めて**実行を終了する（存在しないページ=本文が「ページが見つかりません」等の応答だけは除外）:
  実行中ジョブを待機に戻す → アクセス一時停止フラグを立てる → その時点のデータチェックをやり直して最新の不足を記録 →
  再開用の状態(``access_restriction.json``)と計画(``scrape_plan.after_restriction.json``)を保存。
  制限が解けたら ``--resume`` で続きから実行できる（疎通確認 → 一時停止の解除 → 保存した条件で、最新のチェックに基づいて再取得）。

  python -m src.scripts.scraping.scrape_from_health_plan --resume              # 制限解除後の再開（疎通確認つき）
  python -m src.scripts.scraping.scrape_from_health_plan --resume --clear-pause  # 一時停止フラグも自分で解除する

終了コード: 0=完全（不足なし） / 3=まだ不足あり / 4=事前確認に失敗 / 5=アクセス制限で中断 / 6=確認が取れず中止 / 2=その他の失敗
注意: **GCS に書き込む＝stg と prod は同一バケットなので prod が読むデータも変わる**。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from src.scraper.access_guard import AccessGuard, AccessRestrictionDetected

JST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parents[3]
STATE_NAME = "access_restriction.json"

EXIT_COMPLETE, EXIT_REMAINING, EXIT_PREFLIGHT, EXIT_BLOCKED, EXIT_NOT_CONFIRMED, EXIT_ERROR = 0, 3, 4, 5, 6, 2
CONFIRM_ABOVE = 200                     # これを超える件数、または上書きを含むときは確認が必要
AVG_REQUEST_SEC = 3.1                   # NETKEIBA_THROTTLE_MIN/MAX(2.2〜4.0) の平均 + 処理。同時 1 本(DEC-023)
DATE_ALL_REQUESTS = 300                 # 1 開催日の一式のおおよそのリクエスト数（約 36 レース × 8 ほか。目安）
RUNNERS = ("all", "learning", "vps")


# ── アクセス制限: 状態の保存・再開 ──────────────────────────────────────────────

def state_dir() -> Path:
    from src.data_health import store

    return store.base_dir(None) / "stg"


def state_path() -> Path:
    return state_dir() / STATE_NAME


def load_state() -> dict[str, Any] | None:
    try:
        return json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def parse_statuses(text: str) -> set[int] | None:
    """``any``（既定）= 400 以上のすべて → None。``404,403`` のように指定すればそのステータスだけ。空文字 = 監視しない（空集合）。"""
    if str(text or "").strip().lower() in ("any", "all", "*"):
        return None
    out = set()
    for part in str(text or "").replace(" ", "").split(","):
        if part:
            if not part.isdigit() or not 100 <= int(part) <= 599:
                raise ValueError(f"HTTP ステータス {part!r} が不正です（例: 404 または 404,403）")
            out.add(int(part))
    return out


def default_probe() -> tuple[bool, str]:
    """制限が解けたかの疎通確認（netkeiba のトップを 1 回だけ取得）。"""
    try:
        from src.scraper.client import NetkeibaClient

        with AccessGuard():
            with NetkeibaClient() as c:
                html = c.fetch("https://db.netkeiba.com/", use_cache=False)
        return (bool(html) and len(html) > 200), "取得できた" if html else "空の応答"
    except AccessRestrictionDetected as e:
        return False, str(e)
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:150]}"


def handle_restriction(args: argparse.Namespace, queue: Any, check: Callable, log: dict[str, Any], out: Callable[[str], None], *,
                       reason: str, url: str = "", status: int | None = None, env: str = "stg") -> int:
    """アクセス制限を検知したときの後始末。

    1) 実行中ジョブを待機に戻し、アクセス一時停止フラグを立てる（cron・他のワーカーも止まる）
    2) 今の時点のデータチェックをやり直す（取得できた分が反映され、残りの不足が最新になる）
    3) 再開用の状態と、制限解除後に実行すべき計画を保存する
    """
    out(f"\n!!! アクセス制限の疑いを検知: {reason}")
    out("    直ちに実行を終了します（これ以上リクエストを送りません）。")
    try:
        n = queue.pause_queue_for_access_error(reason)
        out(f"    実行中だったジョブ {n} 件を待機に戻し、アクセス一時停止フラグを立てました。")
    except Exception as e:  # noqa: BLE001
        n = None
        out(f"    (注) キューの一時停止処理に失敗: {e}")
    log["restriction"] = {"reason": reason, "url": url, "status": status, "running_requeued": n}
    out("\n----- この時点のデータチェック（最新の不足を把握） -----")
    plan: dict[str, Any] = {}
    comp: dict[str, Any] | None = None
    try:
        report = check(args)
        plan, comp = report["plan"], report.get("completeness")
        out("完全性: " + ("OK" if comp and comp["complete"] else f"NG（未達 {len(comp['reasons']) if comp else '?'} 項目）"))
        out(f"再取得の残り: {plan['counts']['jobs']} 件 {plan['counts'].get('by_runner', {})}")
    except Exception as e:  # noqa: BLE001
        out(f"    (注) データチェックに失敗: {type(e).__name__}: {str(e)[:150]}（再開時にやり直します）")
        log["restriction"]["check_error"] = str(e)[:300]
    d = state_dir()
    d.mkdir(parents=True, exist_ok=True)
    plan_path = d / "scrape_plan.after_restriction.json"
    if plan:
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
    state = {"detected_at": datetime.now(JST).isoformat(timespec="seconds"), "env": env, "reason": reason, "url": url, "status": status,
             "argv": getattr(args, "_argv", None), "plan_path": str(plan_path) if plan else None,
             "remaining": (plan.get("counts") if plan else None), "complete": (comp or {}).get("complete"),
             "resume_command": "python -m src.scripts.scraping.scrape_from_health_plan --resume"}
    state_path().write_text(json.dumps(state, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    out(f"\n再開用の状態を保存: {state_path()}")
    if plan:
        out(f"制限解除後に実行する計画: {plan_path}")
    out("\n制限が解除されたことを確認してから、次のコマンドで続きを実行してください（疎通確認 → 最新のチェック → 残りを再取得）:")
    out("    " + state["resume_command"] + "        # 一時停止フラグが残っていれば --clear-pause も付ける")
    return EXIT_BLOCKED


def resume_preconditions(args: argparse.Namespace, out: Callable[[str], None], probe: Callable[[], tuple[bool, str]]) -> int | None:
    """--resume の前提確認。問題があれば終了コードを返す（None なら続行）。"""
    from src.scraper import scrape_access_pause as pause_mod

    st = load_state()
    if not st:
        out(f"  [ERROR] 再開する状態がありません（{state_path()}）。アクセス制限で中断した実行の後にだけ使えます")
        return EXIT_PREFLIGHT
    out(f"再開: {st.get('detected_at')} に検知した制限（{st.get('reason', '')[:80]}）の続き。前回の残り {st.get('remaining')}")
    pause = pause_mod.read_access_pause()
    if pause.get("active"):
        if not args.clear_pause:
            out("  [ERROR] アクセス一時停止フラグが残っています。制限が解けたことを確認してから --clear-pause を付けて再実行"
                "（または UI の「再開」で解除）してください")
            return EXIT_BLOCKED
        pause_mod.clear_access_pause()
        out("  一時停止フラグを解除しました。")
    if not args.skip_probe:
        ok, detail = probe()
        out(f"  疎通確認: {'OK' if ok else 'NG'}（{detail}）")
        if not ok:
            out("  まだアクセスできない可能性があります。時間をおいて再実行してください（確認を省くなら --skip-probe）。")
            return EXIT_BLOCKED
    return None


def status_writer(env: str) -> Callable[..., None]:
    """ダッシュボード(run_status.js)へ進捗を書く関数を返す。"""
    from src.data_health import dashboard, store

    base = store.base_dir(None)

    def write(phase: str, done: int | None, total: int | None, detail: str, finished: bool = False) -> None:
        if finished:
            dashboard.finish_run_status(base, env, phase, detail)
        else:
            dashboard.write_run_status(base, env, phase, done, total, detail)

    return write


# ── 事前確認 ─────────────────────────────────────────────────────────────

def preflight(env: str, *, allow_prod: bool, queue: Any, storage: Any = None, environ: dict[str, str] | None = None
              ) -> list[tuple[str, str]]:
    """[(level, message)]。level = error（続行不可）/ warn。"""
    environ = os.environ if environ is None else environ
    out: list[tuple[str, str]] = []
    if env == "dev":
        out.append(("error", "dev では実行できません（GCP に書けず、スクレイピング結果が data/dev_mock に入ってしまう）。学習PC(stg)で実行してください"))
    elif env == "prod" and not allow_prod:
        out.append(("error", "prod では実行しません（VPS の当日取得は cron の役割）。意図して実行するなら --allow-prod"))
    elif env not in ("stg", "prod"):
        out.append(("error", f"KEIBA_ENV={env!r} は想定外です（stg を指定）"))
    if not (environ.get("netkeiba_id", "").strip() and environ.get("netkeiba_pw", "").strip()):
        out.append(("error", "netkeiba の認証情報（netkeiba_id / netkeiba_pw）が未設定です（プレミアム認証が必要なカテゴリがある）"))
    try:
        if storage is None:
            from src.scraper.storage import HybridStorage

            storage = HybridStorage(base_dir=str(ROOT))
        if not storage.gcs_enabled:
            out.append(("error", "GCS に接続できません（GCS_BUCKET / 鍵）。保存されずに「成功」扱いになるのを防ぐため中止します"))
        elif not storage._get_bucket().exists(timeout=8):
            out.append(("error", "GCS バケットが存在しない、または権限がありません"))
    except Exception as e:  # noqa: BLE001
        out.append(("error", f"GCS の確認に失敗: {str(e).splitlines()[0][:120] if str(e) else type(e).__name__}"))
    try:
        from src.scraper.scrape_access_pause import read_access_pause

        pause = read_access_pause()
        if pause.get("active"):
            out.append(("error", f"アクセス一時停止中です（block の疑い）: {pause.get('reason', '')}。解除されるまで実行しません"))
    except Exception:  # noqa: BLE001
        pass
    if queue.is_locked():
        out.append(("error", "別のキューワーカーが動いています（API サーバの常駐ワーカー等）。停止してから実行してください"))
    return out


# ── 絞り込み・集計 ───────────────────────────────────────────────────────

def spec_date(s: dict[str, Any]) -> str:
    d = str(s.get("date") or "")
    if d:
        return d.replace("-", "")[:8]
    return s["target_id"] if s.get("job_kind") == "date" and len(s["target_id"]) == 8 else ""


def select_specs(specs: list[dict[str, Any]], *, runner: str = "all", status: str = "all", categories: list[str] | None = None,
                 since: str | None = None, until: str | None = None, include_inferred: bool = True,
                 max_jobs: int | None = None) -> list[dict[str, Any]]:
    out = []
    for s in specs:
        r = str(s.get("runner", ""))
        if runner == "learning" and not r.startswith("学習PC"):
            continue
        if runner == "vps" and not r.startswith("VPS"):
            continue
        if status != "all" and s.get("status", "missing") != status:
            continue
        if categories and not set(categories) & set(s.get("categories") or []):
            continue
        d = spec_date(s)
        if (since and d and d < since.replace("-", "")) or (until and d and d > until.replace("-", "")):
            continue
        if not include_inferred and s.get("confidence") == "inferred":
            continue
        out.append(s)
    out.sort(key=lambda s: (-int(s.get("priority") or 0), spec_date(s) or "0", s["target_id"]))   # 優先度の高い順（同順位は日付・ID 順）
    return out[:max_jobs] if max_jobs else out


def estimate(specs: list[dict[str, Any]]) -> dict[str, Any]:
    req = 0
    for s in specs:
        if s["job_kind"] == "date":
            req += DATE_ALL_REQUESTS if "date_all" in s["tasks"] else len(s["tasks"]) * 2
        elif s["job_kind"] == "horse":
            req += len(s["tasks"]) * 2
        else:
            req += len(s["tasks"])
    return {"jobs": len(specs), "requests": req, "hours": round(req * AVG_REQUEST_SEC / 3600, 1),
            "by_kind": dict(Counter(s["job_kind"] for s in specs)), "by_status": dict(Counter(s.get("status", "missing") for s in specs)),
            "by_runner": dict(Counter(s.get("runner", "") for s in specs)),
            "by_confidence": dict(Counter(s.get("confidence", "") for s in specs)),
            "overwrite": sum(1 for s in specs if s.get("overwrite"))}


def format_estimate(e: dict[str, Any]) -> str:
    return (f"ジョブ {e['jobs']} 件（{e['by_kind']}）／状態 {e['by_status']}／実行場所 {e['by_runner']}／根拠 {e['by_confidence']}\n"
            f"  取得リクエストの目安 約 {e['requests']:,} 回 ≒ {e['hours']} 時間（同時 1 本・1 回 約 {AVG_REQUEST_SEC} 秒）"
            + (f"\n  ※ 上書き再取得（overwrite）を {e['overwrite']} 件含む" if e["overwrite"] else ""))


# ── キュー ───────────────────────────────────────────────────────────────

def dedupe_key(s: dict[str, Any]) -> str:
    return f"{s['job_kind']}:{s['target_id']}:{':'.join(sorted(set(s['tasks'])))}"


def enqueue(queue: Any, specs: list[dict[str, Any]], *, chunk: int = 500) -> dict[str, int]:
    total: Counter[str] = Counter()
    for i in range(0, len(specs), chunk):
        total.update(queue.bulk_add_jobs(specs[i:i + chunk]))
    return dict(total)


def requeue_failed(queue: Any, specs: list[dict[str, Any]]) -> int:
    """同じ dedupe_key で failed のまま残っているジョブは、投入しても重複扱いで動かないので pending に戻す。"""
    keys = {dedupe_key(s) for s in specs}
    ids = [j["job_id"] for j in queue.load_queue()
           if j.get("status") == "failed" and (j.get("dedupe_key") in keys) and j.get("job_id")]
    if not ids:
        return 0
    n, _ = queue.requeue_failed_jobs(job_ids=ids)
    return n


def outcome(queue: Any, specs: list[dict[str, Any]]) -> dict[str, Any]:
    keys = {dedupe_key(s) for s in specs}
    jobs = [j for j in queue.load_queue() if j.get("dedupe_key") in keys]
    failed = [j for j in jobs if j.get("status") == "failed"]
    return {"statuses": dict(Counter(j.get("status", "?") for j in jobs)),
            "failed": [{"job_id": j.get("job_id"), "target": j.get("target_id"), "tasks": j.get("tasks"),
                        "reason": j.get("failure_reason") or "", "error": str(j.get("error") or "")[:300]} for j in failed[:50]],
            "failed_reasons": dict(Counter(j.get("failure_reason") or "other" for j in failed))}


# ── チェック（データヘルス）──────────────────────────────────────────────

def default_check(args: argparse.Namespace) -> dict[str, Any]:
    """データチェックを実行して保存し、レポート（plan / completeness を含む）を返す。"""
    from src.config.deployment import keiba_env
    from src.data_health import store
    from src.data_health.config import load_settings
    from src.data_health.runner import run_health

    cfg = load_settings(profile="stg")
    if args.since:
        cfg.since = datetime.strptime(args.since.replace("-", ""), "%Y%m%d").date()
    report = run_health(settings=cfg, actual_env=keiba_env(), infra=False, horses=not args.no_horses)
    store.save(report, cfg.out_dir)
    return report


def load_plan_file(path: str) -> dict[str, Any]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if "plan" in d and "specs" not in d:                      # latest.json を渡された場合
        d = {"plan": d["plan"], "completeness": d.get("completeness"), "env": d.get("env")}
        return d
    return {"plan": d, "completeness": None, "env": d.get("env")}


# ── 本体 ────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace, *, queue: Any = None, check: Callable[[argparse.Namespace], dict] | None = None,
        env: str | None = None, storage: Any = None, confirm: Callable[[str], bool] | None = None,
        out: Callable[[str], None] = print, sleep: Callable[[float], None] = time.sleep,
        probe: Callable[[], tuple[bool, str]] | None = None) -> int:
    from src.config.deployment import keiba_env

    env = env or keiba_env()
    if queue is None:
        from src.scraper.job_queue import ScrapeJobQueue

        queue = ScrapeJobQueue()
    check = check or default_check
    started_iso = datetime.now(JST).isoformat(timespec="seconds")
    log: dict[str, Any] = {"started_at": started_iso, "env": env, "execute": bool(args.execute), "resume": bool(args.resume),
                           "filters": {k: getattr(args, k) for k in ("runner", "status", "category", "since", "until", "max_jobs", "rounds")},
                           "rounds": []}

    _progress = status_writer(env) if args.execute else None      # ドライランは「実行中」表示を出さない

    def _status(kind: str, **kw: Any) -> None:
        """ダッシュボードの「実行中」表示を更新する（失敗しても本処理は止めない）。"""
        if _progress is None:
            return
        try:
            if kind == "start":
                _progress("スクレイピング実行を開始", None, None, "事前確認 OK")
            elif kind == "check":
                _progress(f"データチェック（ラウンド {kw['rnd']}/{args.rounds}）", None, None, "")
            elif kind == "scrape":
                _progress(f"スクレイピング中（ラウンド {kw['rnd']}/{args.rounds}）", 0, kw["total"], "ジョブ")
            elif kind == "tick":
                _progress(f"スクレイピング中（ラウンド {kw['rnd']}/{args.rounds}）", kw["done"], kw["total"], kw.get("detail", ""))
            elif kind == "done":
                ok = kw["code"] in (EXIT_COMPLETE, EXIT_REMAINING)
                _progress("完了" if ok else "中断", None, None, f"終了コード {kw['code']}", finished=True)
        except Exception:  # noqa: BLE001
            pass

    def finish(code: int) -> int:
        _status("done", code=code)
        log["exit_code"] = code
        log["finished_at"] = datetime.now(JST).isoformat(timespec="seconds")
        if code in (EXIT_COMPLETE, EXIT_REMAINING) and args.resume and state_path().exists():
            done = state_path().with_name(f"access_restriction.resolved.{datetime.now(JST).strftime('%Y%m%dT%H%M%S')}.json")
            state_path().replace(done)                                   # 再開できたので「未解決の制限」の状態は片付ける
        if not args.no_log:
            d = state_dir() / "scrape_runs"
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"{datetime.now(JST).strftime('%Y%m%dT%H%M%S')}.json"
            p.write_text(json.dumps(log, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            out(f"実行ログ: {p}")
        return code

    if args.resume:
        code = resume_preconditions(args, out, probe or default_probe)
        if code is not None:
            return finish(code)
    problems = preflight(env, allow_prod=args.allow_prod, queue=queue, storage=storage)
    for level, msg in problems:
        out(f"  [{level.upper()}] 事前確認: {msg}")
    if any(level == "error" for level, _ in problems):
        log["preflight"] = problems
        return finish(EXIT_PREFLIGHT)
    out("事前確認: OK（GCS 接続・netkeiba 認証情報・ワーカー停止中・アクセス制限なし）")
    _status("start")

    prev_remaining: int | None = None
    confirmed = bool(args.yes)
    for rnd in range(1, max(1, args.rounds) + 1):
        out(f"\n===== ラウンド {rnd}/{args.rounds}: データチェック =====")
        _status("check", rnd=rnd)
        if args.plan and rnd == 1:
            report = load_plan_file(args.plan)
            out(f"保存済みの計画を使用: {args.plan}")
        else:
            report = check(args)
        plan, comp = report["plan"], report.get("completeness")
        if comp:
            out("完全性: " + ("OK（対象範囲のすべてが揃い、スキーマに適合）" if comp["complete"] else f"NG（未達 {len(comp['reasons'])} 項目）"))
            for r in comp["reasons"][:10]:
                out(f"  - {r['message']}")
        if comp and comp["complete"]:
            log["rounds"].append({"round": rnd, "complete": True})
            out("\n不足はありません。")
            return finish(EXIT_COMPLETE)

        specs = select_specs(plan["specs"], runner=args.runner, status=args.status, categories=args.category, since=args.since,
                             until=args.until, include_inferred=not args.no_inferred, max_jobs=args.max_jobs)
        est = estimate(specs)
        out(f"\n再取得の対象（計画 {len(plan['specs'])} 件のうち絞り込み後）:\n  " + format_estimate(est))
        for n in plan.get("notes", [])[:5]:
            out(f"  (注) {n}")
        rnd_log: dict[str, Any] = {"round": rnd, "remaining_before": len(plan["specs"]), "selected": est}
        log["rounds"].append(rnd_log)
        if not specs:
            out("\n絞り込み後の対象がありません。未達の理由が「取得で解消できないもの」（検証が途中・派生データ・インフラ等）の可能性があります。")
            return finish(EXIT_REMAINING)
        if prev_remaining is not None and len(plan["specs"]) >= prev_remaining:
            out(f"\n進捗がありません（残り {len(plan['specs'])} 件 ≥ 前回 {prev_remaining} 件）。これ以上繰り返しても解消しないため停止します。")
            out("  → 残りの race_id は data/local/meta/data_health/stg/race_ids/ と scrape_runs のログを確認（取得先に存在しない可能性、"
                "スキーマ側の問題は schema_violations summary）")
            rnd_log["stopped"] = "no_progress"
            return finish(EXIT_REMAINING)
        prev_remaining = len(plan["specs"])

        if not args.execute:
            out("\nドライランです（何も投入していません）。実行するには --execute を付けてください。")
            for s in specs[:15]:
                out(f"    {s['job_kind']:5s} {s['target_id']:14s} {','.join(s['tasks']):40s} [{s.get('status')}] {s.get('reason', '')[:60]}")
            if len(specs) > 15:
                out(f"    … 他 {len(specs) - 15} 件")
            return finish(EXIT_REMAINING)

        need_confirm = (est["jobs"] > CONFIRM_ABOVE or est["overwrite"]) and not confirmed
        if need_confirm:
            msg = (f"\nGCS に書き込みます（stg と prod は同一バケットなので prod が読むデータも変わります）。"
                   f"{est['jobs']} 件" + (f"・うち上書き {est['overwrite']} 件" if est["overwrite"] else "") + " を実行してよいですか？ [yes/no] ")
            ok = confirm(msg) if confirm else (sys.stdin.isatty() and input(msg).strip().lower() == "yes")
            if not ok:
                out("確認が取れないため中止しました（非対話で実行するなら --yes）。")
                return finish(EXIT_NOT_CONFIRMED)
            confirmed = True

        pre = queue.get_status()
        others = (pre.get("pending", 0) or 0) + (pre.get("precheck", 0) or 0)
        if others:
            out(f"  (注) キューに既に待機中のジョブが {others} 件あります。ワーカーはそれも処理します。")
        res = enqueue(queue, specs)
        rnd_log["enqueue"] = res
        out(f"\nキューへ投入: {res}")
        if args.requeue_failed:
            n = requeue_failed(queue, specs)
            rnd_log["requeued_failed"] = n
            out(f"  失敗のまま残っていたジョブを待機に戻した: {n} 件")
        elif res.get("duplicate", 0):
            out(f"  (注) 重複 {res['duplicate']} 件は、既に待機・実行中、または失敗のまま残っています。失敗を再実行するなら --requeue-failed")

        stop_statuses = parse_statuses(args.stop_on_status)
        watch = ("HTTP 400 以上のすべて" if stop_statuses is None else f"HTTP {sorted(stop_statuses)}") if stop_statuses != set() else "なし"
        out(f"\nワーカーを起動して処理します（netkeiba へは同時 1 本・間隔を空けて取得。Ctrl-C で中断できます）…\n"
            f"  アクセス制限の監視: {watch}"
            + ("（存在しないページを除く）" if not args.strict_not_found else "（存在しないページも含む）")
            + f"・通信エラー・制限ページを {args.stop_after} 回連続で受けたら直ちに終了します")
        t0 = time.time()
        _status("scrape", rnd=rnd, total=len(specs))
        guard = AccessGuard(stop_statuses, args.stop_after, exempt_not_found=not args.strict_not_found) if stop_statuses != set() else None
        poll_stop = threading.Event()

        def poll() -> None:                       # 取得中、数秒ごとにジョブの進み具合をダッシュボードへ
            while not poll_stop.wait(5.0):
                try:
                    st = outcome(queue, specs)["statuses"]
                    done = sum(st.get(k, 0) for k in ("completed", "failed"))
                    _status("tick", rnd=rnd, done=done, total=len(specs), detail=" / ".join(f"{k} {v}" for k, v in sorted(st.items())))
                except Exception:  # noqa: BLE001
                    pass

        poller = threading.Thread(target=poll, daemon=True)
        poller.start()
        try:
            try:
                if guard is None:
                    queue.process_queue()
                else:
                    with guard:
                        queue.process_queue()
            finally:
                poll_stop.set()
        except AccessRestrictionDetected as e:
            rnd_log["stopped"] = "access_restriction"
            rnd_log["seconds"] = round(time.time() - t0)
            rnd_log["outcome"] = outcome(queue, specs)
            return finish(handle_restriction(args, queue, check, log, out, reason=str(e), url=e.url, status=e.status, env=env))
        except KeyboardInterrupt:
            out("中断されました。")
            rnd_log["interrupted"] = True
            return finish(EXIT_ERROR)
        rnd_log["seconds"] = round(time.time() - t0)
        if guard is not None:
            rnd_log["guard"] = {"responses": dict(guard.seen), "page_not_found_ignored": guard.not_found}
            if guard.not_found:
                out(f"  (参考) 存在しないページとして見逃した応答: {guard.not_found} 件（アクセス制限とはみなしていません）")
        oc = outcome(queue, specs)
        rnd_log["outcome"] = oc
        out(f"処理結果（{rnd_log['seconds']} 秒）: {oc['statuses']}" + (f" ／ 失敗の内訳 {oc['failed_reasons']}" if oc["failed"] else ""))
        for f in oc["failed"][:5]:
            out(f"    失敗: {f['target']} {f['tasks']} [{f['reason'] or 'other'}] {f['error'][:100]}")
        try:
            from src.scraper.scrape_access_pause import read_access_pause

            pause = read_access_pause()
        except Exception:  # noqa: BLE001
            pause = {}
        try:                                                       # この実行中に保存時のスキーマ検証で拒否されたデータ（どの値か）
            from src.scraper import schema_violations as SV

            rej = SV.summarize(".", since=started_iso, top=5)
            rnd_log["schema_rejections"] = {"records": rej["records"], "decisions": rej["decisions"], "top": rej["groups"][:5]}
            if rej["records"]:
                out(f"保存時のスキーマ検証で引っかかったデータ: {rej['decisions']}"
                    "（スクレイパーは1カテゴリの失敗として続行するため、ジョブ自体は completed になることがある）")
                for g in rej["groups"][:5]:
                    out(f"    {g['category']} {g['field']} [{g['rule']}] {g['count']}回: " + " ; ".join(f"{v} ×{n}" for v, n in g["values"][:2]))
                out("    → data/local/meta/schema_violations/<category>.jsonl（違反の中身）と data/local/quarantine/（拒否したデータ）")
        except Exception:  # noqa: BLE001
            pass
        if pause.get("active"):
            rnd_log["stopped"] = "access_pause"
            return finish(handle_restriction(args, queue, check, log, out, reason=str(pause.get("reason") or "アクセス一時停止"), env=env))
        if rnd < args.rounds:
            sleep(args.pause_sec)

    out("\n===== 最終チェック =====")
    final = check(args) if not args.plan else None
    if final:
        comp = final["completeness"]
        log["final"] = {"complete": comp["complete"], "reasons": comp["reasons"][:20], "planned_jobs": final["plan"]["counts"]["jobs"]}
        out("完全性: " + ("OK" if comp["complete"] else f"NG（未達 {len(comp['reasons'])} 項目 / 残りの再取得 {final['plan']['counts']['jobs']} 件）"))
        for r in comp["reasons"][:10]:
            out(f"  - {r['message']}")
        return finish(EXIT_COMPLETE if comp["complete"] else EXIT_REMAINING)
    return finish(EXIT_REMAINING)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="stg: データチェック → 不足の特定 → スクレイピング → 再チェック",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--execute", action="store_true", help="実際にキュー投入・スクレイピングを行う（既定はドライラン）")
    ap.add_argument("--yes", action="store_true", help="確認プロンプトを省略（非対話実行用）")
    ap.add_argument("--plan", help="データチェックを実行せず、保存済みの scrape_plan.json（または latest.json）を使う（1 ラウンド目のみ）")
    ap.add_argument("--runner", choices=RUNNERS, default="all",
                    help="対象の実行場所: learning=学習PC(過去分の補完) / vps=VPS cron(直近14日) / all（既定）。stg では all でよい")
    ap.add_argument("--status", choices=("all", "missing", "invalid", "calendar"), default="all",
                    help="missing=不足 / invalid=スキーマ不適合(上書き再取得) / calendar=race_lists の不完全")
    ap.add_argument("--category", action="append", help="対象カテゴリ（複数可。例: --category race_result --category race_index）")
    ap.add_argument("--since", help="対象期間の開始 YYYY-MM-DD（データチェックの開始日にも使う）")
    ap.add_argument("--until", help="対象期間の終了 YYYY-MM-DD")
    ap.add_argument("--no-inferred", action="store_true", help="連番から推定しただけのレース（race_lists に無い）を除く")
    ap.add_argument("--max-jobs", type=int, help="1 ラウンドで実行するジョブ数の上限（優先度順）。まず少数で試すのに使う")
    ap.add_argument("--rounds", type=int, default=1, help="チェック→取得→再チェックの最大ラウンド数（進捗が無ければ自動停止）")
    ap.add_argument("--pause-sec", type=float, default=30.0, help="ラウンド間の待機秒")
    ap.add_argument("--requeue-failed", action="store_true", help="同じジョブが失敗のまま残っていたら待機に戻す")
    ap.add_argument("--no-horses", action="store_true", help="データチェックで馬の確認（出馬表の download）を省く")
    ap.add_argument("--allow-prod", action="store_true", help="KEIBA_ENV=prod でも実行を許可（通常は不要）")
    ap.add_argument("--no-log", action="store_true", help="実行ログ(JSON)を保存しない")
    ap.add_argument("--stop-on-status", default="any",
                    help="アクセス制限とみなして直ちに終了するHTTPステータス。any（既定）= 400 以上のすべて（存在しないページを除く）／"
                         "404,403 のように指定／空文字で監視しない。接続エラー・タイムアウト・制限ページは常に監視（空文字の場合を除く）")
    ap.add_argument("--strict-not-found", action="store_true",
                    help="「ページが見つかりません」の応答も制限の疑いとして扱う（既定は除外。より敏感にしたいとき）")
    ap.add_argument("--stop-after", type=int, default=1, help="何回連続でエラーを受けたら終了するか（既定 1 = 即時。正常な応答でリセット）")
    ap.add_argument("--resume", action="store_true",
                    help="アクセス制限で中断した実行の続き。疎通確認 → 最新のデータチェック → 残りを再取得（前回の条件を引き継ぐ。指定した引数が優先）")
    ap.add_argument("--clear-pause", action="store_true", help="--resume のとき、アクセス一時停止フラグも解除する（制限が解けたと確認した後に）")
    ap.add_argument("--skip-probe", action="store_true", help="--resume のときの疎通確認を省く")
    return ap


def parse_args(argv: list[str]) -> argparse.Namespace:
    """--resume のときは、前回（アクセス制限で中断した実行）の条件を引き継ぎ、今回指定した引数で上書きする。"""
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.resume:
        st = load_state()
        if st and st.get("argv"):
            args = ap.parse_args(argv, namespace=ap.parse_args(st["argv"]))
    args._argv = [a for a in argv if a not in ("--resume", "--clear-pause", "--skip-probe")]     # 次の再開でも使えるよう保存する条件
    return args


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()            # KEIBA_ENV・GCS_*・netkeiba_* を .env（＋.env.<env>）から読む
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("中断されました。")
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
