"""2016〜2019 年だけに走った馬の調教ページ（horse_training）を全ページ取り直す。

  python -m src.scripts.scraping.backfill_horse_training_past --horses-file missing_ids.parquet --dry-run
  python -m src.scripts.scraping.backfill_horse_training_past --horses-file missing_ids.parquet --limit 40      # パイロット
  python -m src.scripts.scraping.backfill_horse_training_past --horses-file missing_ids.parquet                  # 本番（再開できる）

過去分補填（--phase past）の段階に horse_training は無く、2016〜2019 年の最終出走の馬は調教が未取得（最終追い切りの被覆が 10〜66%）。
馬別ページは 1 ページ 10 走分なので、1 頭あたり複数ページ（平均 約 3）。保存は ``ScraperRunner.scrape_horse_training(force=True)``
（全ページを結合して GCS とローカルに保存。調教なしの馬は entries 空で保存）。GCS は書き込みのみで読み出さない。
スクレイピング規則: 他の取得が動いている間は待つ、1 頭ごとに上乗せ 0.5〜1.5 秒、100 頭ごとに 30〜60 秒、
アクセス制限（HTTP 400/403/429・制限ページ）の検知・アクセス保留の印・連続 8 件の失敗で中断。再開できる（進捗: data/local/meta/horse_training_past_progress.json）。
2 ページ目以降が取れなかった馬は、途中までの記録で上書きしない（scrape_horse_training は途中で諦めても保存するため、保存の直前で止める）。

打ち切った馬の全ページの取り直し（全頭が終わってから）:
  python -m src.scripts.scraping.backfill_horse_training_past --horses-file missing_ids.parquet --redo-truncated
  取得済みのうち手元の記録に truncated_before がある馬だけを、打ち切りなし（--cutoff ""）で取り直す。
  進み具合は進捗ファイルの redo に残す（再開できる。失敗した馬は元の打ち切った記録のまま残る）。
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from src.scripts.scraping.backfill_horse_training_recent import EXTRA_SLEEP, wait_ok

ROOT = Path(__file__).resolve().parents[3]
PROGRESS = ROOT / "data/local/meta/horse_training_past_progress.json"
STATUSES = ("saved", "empty", "error")
CUTOFF = "2015-01-01"     # 最終追い切りの集計は 2016 年以降。ページ内の最古の調教がこれより前になったら、次のページは取らない（1 年の余白）
MAX_EMPTY_STREAK = 20     # 調教なしの馬が続くのは異常（ログイン切れ・軟ブロックの疑い）
MAX_WAIT_SEC = 6 * 3600   # 開催日ランナー等が長く動く日があるので、待つ上限は長めにとる


def read_horses(path: Path) -> list[str]:
    """馬 ID の一覧（parquet は horse_id 列、それ以外は空白区切り）。重複は除き、並びは保つ。"""
    if path.suffix == ".parquet":
        import pandas as pd

        ids = [str(x) for x in pd.read_parquet(path, columns=["horse_id"])["horse_id"]]
    else:
        ids = path.read_text(encoding="utf-8").split()
    return list(dict.fromkeys(i for i in ids if i))


def load_progress() -> dict:
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text(encoding="utf-8"))   # 壊れていたら止める（空から始めると全頭を取り直してしまう）
    return {"done": {}, "counts": {s: 0 for s in STATUSES}, "requests": 0}


def save_progress(p: dict) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    p["counts"] = {s: sum(1 for v in p["done"].values() if v == s) for s in STATUSES}
    tmp = PROGRESS.with_suffix(".tmp")
    tmp.write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(PROGRESS)


def is_truncated(horse_id: str) -> bool:
    """手元の記録が打ち切りの日付で止めたもの（truncated_before あり）か。記録が無い・読めないときは False。"""
    from src.scraper.horse_training_local import horse_training_json_path

    path = horse_training_json_path(ROOT, horse_id)
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("truncated_before"))
    except (OSError, ValueError):
        return False


def pages_of(result: dict | None) -> int:
    """保存した記録に要したリクエスト数（ページ数）。調教なしでも 1 回は取る。"""
    return max(1, int((result or {}).get("pages_fetched") or 1))


class StopPaging(RuntimeError):
    """打ち切りの日付に届いたので、次のページは取らない（scrape_horse_training は 2 ページ目以降の例外で止まる）。"""


class PartialFetch(RuntimeError):
    """2 ページ目以降が取れなかった・アクセス制限を検知した。途中までの記録で上書きしないための例外。"""


def guard_persist(runner, guard, cutoff: str = "", reject_empty: bool = False) -> None:
    """runner の取得と保存を包む。1 ページ以上取れたあとに取得が失敗した馬、制限を検知した馬は保存しない。
    cutoff（YYYY-MM-DD）を渡すと、ページ内の最古の調教日がそれより前になった時点で、次のページを取らずに打ち切って保存する。
    reject_empty=True（打ち切った馬の取り直し）は、調教 0 件の結果を保存しない（ログイン切れ・軟ブロックで、保存済みの記録を空で消さない）。"""
    state = {"ok": 0, "failed": False, "cut": False, "stopped": False}
    fetch, persist = runner.client.fetch, runner._persist_horse_training
    storage = getattr(runner, "storage", None)
    if storage is not None:
        save = storage.save

        def save_(*a, **k):
            r = save(*a, **k)
            if r is False:   # GCS 未接続・バックオフ中は例外でなく False が返る
                raise RuntimeError("GCS に保存できなかった")
            return r

        storage.save = save_

    parser = getattr(runner, "training_parser", None)
    if cutoff and parser is not None:
        parse = parser.parse

        def parse_(*a, **k):
            data = parse(*a, **k)
            dates = [str(e.get("date") or "")[:10] for e in (data or {}).get("entries", [])]
            dates = [d for d in dates if len(d) == 10]
            state["cut"] = state["cut"] or bool(dates and min(dates) < cutoff)
            return data

        parser.parse = parse_

    def fetch_(*a, **k):
        if state["cut"] and state["ok"] > 0:
            state["stopped"] = True
            raise StopPaging("打ち切りの日付に届いた")
        try:
            r = fetch(*a, **k)
        except BaseException:
            state["failed"] = state["ok"] > 0
            raise
        state["ok"] += 1
        return r

    def persist_(hid, result):
        failed, ok, stopped = state["failed"], state["ok"], state["stopped"]
        state.update(ok=0, failed=False, cut=False, stopped=False)
        if guard.tripped is not None:
            raise guard.tripped
        if reject_empty and not (result or {}).get("entries"):
            raise PartialFetch(f"{hid}: 取り直しの結果が 0 件なので、保存済みの記録を上書きしない")
        if stopped:   # 意図した打ち切り: 取れたページ数と打ち切りの日付を記録に残す
            failed = False
            result["pages_fetched"], result["truncated_before"] = ok, cutoff
            result.pop("incomplete", None)   # 打ち切りは取り損ねではない（印は truncated_before で残す）
        elif failed or (ok and int((result or {}).get("pages_fetched") or 1) > ok):   # 例外なしの空ページで打ち切った場合も含む
            raise PartialFetch(f"{hid}: 2 ページ目以降が取れなかったので保存しない")
        persist(hid, result)

    runner.client.fetch, runner._persist_horse_training = fetch_, persist_
    runner._guard_state = state


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horses-file", required=True, help="対象の馬 ID（parquet の horse_id 列、または 1 行 1 頭のテキスト）")
    ap.add_argument("--limit", type=int, default=0, help="今回取得する馬数の上限（パイロット用。0 は残り全部）")
    ap.add_argument("--cutoff", default=CUTOFF, help="この日付より前の調教しか無いページは取らない（空文字で全ページ取る）")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--redo-truncated", action="store_true",
                    help="取得済みで打ち切った馬（記録に truncated_before あり）だけを全ページ取り直す（--cutoff は空になる）")
    ap.add_argument("--dry-run", action="store_true", help="対象数と所要時間の見積もりだけ")
    a = ap.parse_args(argv)

    horses = read_horses(Path(a.horses_file))
    prog = load_progress()
    if a.redo_truncated:
        # 本取得が正常に終わってから（並走すると同じ進捗ファイルを上書きし合う）
        if not prog.get("finished_at") or prog.get("aborted") or any(h not in prog["done"] for h in horses):
            print("本取得が終わっていない（finished_at が無い・中断・未取得あり）ので、打ち切った馬の取り直しは始めない", flush=True)
            return 5
        a.cutoff = ""
        redo = prog.setdefault("redo", {})
        todo = [h for h in horses if prog["done"].get(h) == "saved"
                and (h not in redo or (a.retry_failed and redo[h] == "error")) and is_truncated(h)]
    else:
        todo = [h for h in horses if h not in prog["done"] or (a.retry_failed and prog["done"][h] == "error")]
    n = len(todo) if not a.limit else min(len(todo), a.limit)
    est_h = n * (5 if a.redo_truncated else 3) * 5.0 / 3600   # 1 頭 約 3 ページ（打ち切った馬は 4 ページ以上）× 約 5 秒
    if a.redo_truncated:
        print(f"打ち切った馬の取り直し: 対象 {len(horses)} 頭 / 取り直し済み {len(prog['redo'])} / 残り {len(todo)} 頭 / 今回 {n} 頭（見積もり {est_h:.1f} 時間）", flush=True)
    else:
        print(f"対象 {len(horses)} 頭 / 取得済み {len(prog['done'])} / 未取得 {len(todo)} 頭 / 今回 {n} 頭（見積もり {est_h:.1f} 時間）", flush=True)
    if a.dry_run:
        return 0

    from src.scraper import backfill_guard as bg
    from src.scraper.access_guard import AccessGuard, AccessRestrictionDetected
    from src.scraper.run import ScraperRunner
    from src.scraper.scrape_access_pause import read_access_pause

    runner = ScraperRunner()
    guard = AccessGuard(statuses={400, 403, 429}, watch_transport_errors=False)
    guard.__enter__()
    guard_persist(runner, guard, a.cutoff, reject_empty=a.redo_truncated)
    st = prog.setdefault("redo_meta", {}) if a.redo_truncated else prog   # 取り直しの経過は本取得の finished_at などと分けて残す

    def tally() -> dict:
        if not a.redo_truncated:
            return prog["counts"]
        return {s_: sum(1 for v in prog["redo"].values() if v == s_) for s_ in STATUSES}

    try:
        prog["total"] = len(horses)
        st.pop("aborted", None)
        st.pop("finished_at", None)
        t0, done, streak, empties = time.time(), 0, 0, 0
        for hid in todo[:n]:
            halt, pause = bg.halted(), read_access_pause()
            if halt or pause.get("active") or guard.tripped is not None:
                st["aborted"] = f"アクセス保留／中断印あり: {(halt or {}).get('reason') or pause.get('reason') or guard.tripped}"
                save_progress(prog)
                print(st["aborted"], flush=True)
                return 3
            if not wait_ok(MAX_WAIT_SEC):
                st["aborted"] = "他の取得が終わらず待ちきれなかった"
                save_progress(prog)
                print(st["aborted"], flush=True)
                return 4
            status, req = "error", 1
            try:
                res = runner.scrape_horse_training(hid, force=True)
                status = "saved" if (res or {}).get("entries") else "empty"
                req = pages_of(res)
            except AccessRestrictionDetected as e:
                st["aborted"] = f"アクセス制限の疑い: {e}"
                save_progress(prog)
                print(st["aborted"], flush=True)
                return 3
            except Exception as e:  # noqa: BLE001
                runner._guard_state.update(ok=0, failed=False, cut=False, stopped=False)
                print(f"失敗 {hid}: {e}", flush=True)
            done += 1
            if a.redo_truncated:   # 失敗しても done は saved のまま（打ち切った記録が残っている）
                prog["redo"][hid] = status
            else:
                prog["done"][hid] = status
                prog["counts"][status] = prog["counts"].get(status, 0) + 1
            prog["requests"] = prog.get("requests", 0) + req
            streak = streak + 1 if status == "error" else 0
            empties = empties + 1 if status == "empty" else 0
            if streak >= 8 or empties >= MAX_EMPTY_STREAK:
                st["aborted"] = f"連続 {streak or empties} 件が{'失敗' if streak else '調教なし'}（アクセス制限・ログイン切れ・ページ構造の変化の疑い）: {hid}"
                save_progress(prog)
                print(st["aborted"], flush=True)
                return 3
            if done % 25 == 0:
                st.update({"seconds": round(time.time() - t0), "remaining": len(todo) - done})
                save_progress(prog)
                print(f"{done}/{n} {tally()} ({hid})", flush=True)
            time.sleep(random.uniform(30, 60) if done % 100 == 0 else random.uniform(*EXTRA_SLEEP))
        st.update({"finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - t0), "remaining": len(todo) - done})
        save_progress(prog)
        print(json.dumps(tally(), ensure_ascii=False), flush=True)
        return 0
    finally:
        guard.__exit__(None, None, None)
        save_progress(prog)


if __name__ == "__main__":
    raise SystemExit(main())
