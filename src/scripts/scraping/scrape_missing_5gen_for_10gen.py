"""10gen 構築に必要な 5gen 末端祖先のみをスクレイピングする最適化版。

【背景】
    `scrape_pedigree_10gen_ancestors.py` は 5gen 内に登場する全祖先 (gen 1-5)
    を todo にするため、すでに JOIN で構築可能なデータも再取得していた。

    しかし `build_horse_pedigree_10gen.py` は既に「5gen 末端 (gen=5) 祖先 X
    の 5gen データを展開して gen 6-10 を構築する」JOIN 方式を実装しており、
    新規取得が必要なのは "5gen 末端 X のうちローカル 5gen を持たない馬" のみ。

【効果】
    - 旧 todo: 約 39,722 件 (50h)
    - 新 todo: 約 13,287 件 (16h)
    - 削減率: 約 66%

【動作】
    1. race_result_flat（既定: 2020 年以降）から対象競走馬 horse_id を収集
    2. その競走馬の 5gen JSON だけを走査し、gen=5 末端祖先 ID を集合化
    3. ローカル 5gen 既存集合と差分を取り、todo を確定
    4. NetkeibaClient + HybridStorage で順次取得
    5. 取得時の sex も `parse_horse_sex_from_ped_html` (修正版) で正しく取得

Usage:
    # 通常起動 (バックグラウンド)
    nohup setsid python3 -m src.scripts.scraping.scrape_missing_5gen_for_10gen \
        --interval 3.0 \
        > logs/scraping/scrape_missing_5gen.log 2>&1 < /dev/null &
    disown

    # 動作確認 (先頭 5 馬)
    python3 -m src.scripts.scraping.scrape_missing_5gen_for_10gen --limit 5

    # 再帰的に追加祖先も取りたい場合
    python3 -m src.scripts.scraping.scrape_missing_5gen_for_10gen --recursive

【再開】
    ローカル horse_pedigree_5gen の有無で未取得を判定するため、停止後は
    同コマンドを再実行するだけで続きから再開できる (進捗 JSON は累計表示用)。
    進捗をリセットして最初から数え直す場合は --reset-progress を付ける。

進捗:
    data/research/pedigree_10gen_3view/scrape_missing_5gen_progress.json

【完了後】10gen JSON / 研究用 parquet / 祖先名マップの再生成は次を実行:
    bash scripts/run_after_scrape_missing_5gen_10gen_chain.sh
    # バックグラウンドで「5gen 取得終了まで待ってから」連鎖:
    # nohup bash scripts/run_after_scrape_missing_5gen_10gen_chain.sh \\
    #   >> logs/scraping/after_5gen_10gen_chain.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.pedigree.pedigree_similarity import (  # noqa: E402
    parse_blood_table_5gen,
    parse_horse_sex_from_ped_html,
)
from src.research.pedigree.verify_race_horses_pedigree_local import (  # noqa: E402
    collect_race_horse_ids,
)
from src.scraper.client import NetkeibaClient  # noqa: E402
from src.scraper.storage import HybridStorage  # noqa: E402

logger = logging.getLogger("scrape_missing_5gen")

PED_URL = "https://db.netkeiba.com/horse/ped/{horse_id}/"
PED_DIR = ROOT / "data/local/horse_pedigree_5gen"
TABLES_DIR = ROOT / "data/page_reference/tables"
PROGRESS_PATH = ROOT / "data/local/research/pedigree_10gen_3view/scrape_missing_5gen_progress.json"
DEFAULT_YEARS = "2020,2021,2022,2023,2024,2025"


def _prefix_of(horse_id: str) -> str:
    return horse_id[:4] if len(horse_id) >= 4 else "0000"


def _local_path(horse_id: str) -> Path:
    return PED_DIR / _prefix_of(horse_id) / f"{horse_id}.json"


def _save_local(horse_id: str, data: dict) -> None:
    p = _local_path(horse_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def collect_gen5_ids(file_path: Path) -> set[str]:
    """1 つの 5gen ファイルから、gen=5 ancestors の horse_id を集合化。"""
    try:
        data = json.loads(file_path.read_text())
    except Exception:
        return set()
    out: set[str] = set()
    for a in data.get("ancestors") or []:
        if a.get("generation") != 5:
            continue
        hid = (a.get("horse_id") or "").strip()
        if hid:
            out.add(hid)
    return out


def _parse_years(s: str) -> list[int]:
    out: list[int] = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return sorted(set(out))


def load_scope_race_horses(years: list[int]) -> set[str]:
    """race_result_flat から対象競走馬 ID を収集。"""
    ids = collect_race_horse_ids(TABLES_DIR, years)
    print(f"[scope] race_result_flat years={years[0]}–{years[-1]} "
          f"→ {len(ids):,} 頭", flush=True)
    return ids


def scan_state(
    target_gen: int = 5,
    *,
    scope_race_horses: set[str] | None = None,
) -> tuple[set[str], set[str], int]:
    """ローカル 5gen を走査して、(既存集合, 未取得集合, needed件数) を返す。

    scope_race_horses が指定された場合、当該競走馬の 5gen から gen=5 末端のみ集める
    （10gen JOIN に必要な分だけ。他馬の 5gen は走査しない）。
    """
    existing: set[str] = set()
    needed: set[str] = set()
    n_files = 0
    n_scoped = 0
    t0 = time.time()
    for d in sorted(PED_DIR.iterdir()):
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.suffix != ".json":
                continue
            hid = f.stem
            existing.add(hid)
            n_files += 1
            if scope_race_horses is not None and hid not in scope_race_horses:
                continue
            n_scoped += 1
            if target_gen == 5:
                needed.update(collect_gen5_ids(f))
            else:
                try:
                    data = json.loads(f.read_text())
                    for a in data.get("ancestors") or []:
                        anc_hid = (a.get("horse_id") or "").strip()
                        if anc_hid:
                            needed.add(anc_hid)
                except Exception:
                    pass
    missing = needed - existing
    scope_note = (
        f"  scope_race_horses={len(scope_race_horses):,}  scanned={n_scoped:,}"
        if scope_race_horses is not None else ""
    )
    print(
        f"[scan] local={len(existing):,}  needed_gen{target_gen}={len(needed):,}  "
        f"missing={len(missing):,}{scope_note} ({time.time()-t0:.1f}s)",
        flush=True,
    )
    return existing, missing, len(needed)


def build_record(horse_id: str, ancestors: list[dict], sex: str) -> dict:
    sire = dam = dam_sire = ""
    for a in ancestors:
        if a["generation"] == 1 and a["position"] == 0:
            sire = a["name"]
        elif a["generation"] == 1 and a["position"] == 1:
            dam = a["name"]
        elif a["generation"] == 2 and a["position"] == 2:
            dam_sire = a["name"]
    return {
        "horse_id": horse_id,
        "sex": sex,
        "sire": sire,
        "dam": dam,
        "dam_sire": dam_sire,
        "ancestors": ancestors,
        "ancestor_count": len(ancestors),
        "source": "phase_10gen_join_optimized",
    }


class AccessError(Exception):
    pass


def _ped_page_has_blood_table(html: str) -> bool:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    return bool(soup.select_one(
        "table.blood_table, table[class*='blood'], table[summary*='血統']"
    ))


def scrape_one(
    horse_id: str,
    client: NetkeibaClient,
    storage: HybridStorage,
) -> tuple[bool, set[str], str]:
    """1 頭分をスクレイピング。

    Returns:
        (success, newly_found_gen5_ids, error_msg)
    """
    try:
        html = client.fetch(PED_URL.format(horse_id=horse_id))
        ancestors = parse_blood_table_5gen(html)
        if len(ancestors) < 5:
            if len(ancestors) == 0 and not _ped_page_has_blood_table(html):
                return False, set(), "血統表なし"
            logger.info(
                "祖先%d件（netkeiba 血統未登録または空白表）— 保存: %s",
                len(ancestors), horse_id,
            )
        sex = parse_horse_sex_from_ped_html(html, horse_id)
        rec = build_record(horse_id, ancestors, sex)
        # GCS + ローカル両方に保存
        storage.save("horse_pedigree_5gen", horse_id, rec)
        _save_local(horse_id, rec)
        # 再帰オプション用に gen=5 ancestors を返す
        new_gen5 = {
            (a.get("horse_id") or "").strip()
            for a in ancestors
            if a.get("generation") == 5 and (a.get("horse_id") or "").strip()
        }
        return True, new_gen5, ""
    except Exception as e:
        s = str(e)
        if any(c in s for c in (
            "403", "429", "503", "Forbidden", "Too Many",
            "Service Unavailable", "ConnectionError",
            "Connection aborted", "Network is unreachable",
        )):
            raise AccessError(f"アクセスエラー ({horse_id}): {s}") from e
        return False, set(), s


def load_progress() -> dict:
    if not PROGRESS_PATH.exists():
        return {}
    try:
        return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_progress(state: dict) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _merge_counts(base: dict, session: dict) -> dict:
    keys = ("success", "failed", "skipped", "added_recursive")
    return {k: int(base.get(k, 0)) + int(session.get(k, 0)) for k in keys}


def _progress_payload(
    *,
    needed_count: int,
    run_started_at: float,
    session_started_at: float,
    remaining_at_session_start: int,
    index_done: int,
    todo_len: int,
    counts_session: dict,
    counts_cumulative_base: dict,
    target_gen: int,
    recursive: bool,
    resumed: bool,
    scope_years: list[int] | None = None,
    scope_race_horses_count: int | None = None,
    rate_per_hour: int | None = None,
    eta_hours: float | None = None,
    stopped_at: float | None = None,
    reason: str | None = None,
    finished_at: float | None = None,
) -> dict:
    done_cumulative = needed_count - remaining_at_session_start + index_done
    counts_cumulative = _merge_counts(counts_cumulative_base, counts_session)
    out: dict = {
        "updated_at": time.time(),
        "run_started_at": run_started_at,
        "session_started_at": session_started_at,
        "needed_count": needed_count,
        "initial_missing": needed_count,
        "remaining_at_session_start": remaining_at_session_start,
        "done_cumulative": done_cumulative,
        "done": index_done,
        "n_total": needed_count,
        "n_remaining": max(todo_len - index_done, 0),
        "counts": counts_session,
        "counts_cumulative": counts_cumulative,
        "target_gen": target_gen,
        "recursive": recursive,
        "resumed": resumed,
        "scope_years": scope_years or [],
        "scope_race_horses": scope_race_horses_count,
        "all_horses": scope_race_horses_count is None,
    }
    if rate_per_hour is not None:
        out["rate_per_hour"] = rate_per_hour
    if eta_hours is not None:
        out["eta_hours"] = eta_hours
    if stopped_at is not None:
        out["stopped_at"] = stopped_at
    if reason is not None:
        out["reason"] = reason
    if finished_at is not None:
        out["finished_at"] = finished_at
    # ダッシュボード互換
    out["started_at"] = run_started_at
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=3.0,
                        help="スクレイピング間隔 (秒) — netkeiba は >=3.0 推奨")
    parser.add_argument("--limit", type=int, default=None,
                        help="先頭 N 頭で動作確認")
    parser.add_argument("--target-gen", type=int, default=5,
                        help="集める世代範囲 (5=末端のみ最適化, 0=旧挙動全 gen)")
    parser.add_argument("--recursive", action="store_true",
                        help="新規取得馬の gen=5 末端も追加 todo に積む (10gen 完全化)")
    parser.add_argument("--save-progress-every", type=int, default=50,
                        help="N 頭ごとに進捗 JSON を更新")
    parser.add_argument("--reset-progress", action="store_true",
                        help="進捗 JSON の累計カウンタをリセットして新規セッション扱い")
    parser.add_argument(
        "--years",
        type=str,
        default=DEFAULT_YEARS,
        help="race_result_flat の対象年（カンマ区切り）。10gen 対象競走馬の範囲",
    )
    parser.add_argument(
        "--all-horses",
        action="store_true",
        help="全年・全 5gen を走査（旧挙動。通常は使わない）",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    target_gen = args.target_gen if args.target_gen > 0 else 0
    scope_race_horses: set[str] | None = None
    scope_years: list[int] = []
    if not args.all_horses:
        scope_years = _parse_years(args.years)
        if not scope_years:
            print("[main] --years が空です", flush=True)
            return 1
        scope_race_horses = load_scope_race_horses(scope_years)
    existing, missing, needed_count = scan_state(
        target_gen=target_gen if target_gen >= 1 else 5,
        scope_race_horses=scope_race_horses,
    )
    todo = sorted(missing)
    if args.limit:
        todo = todo[:args.limit]
        print(f"[main] --limit により {len(todo):,} に絞る", flush=True)

    if not todo:
        print("[main] 未取得馬なし、終了", flush=True)
        return 0

    prior = load_progress()
    scope_meta_match = (
        prior.get("all_horses") == args.all_horses
        and prior.get("scope_years") == scope_years
    )
    can_resume = (
        not args.reset_progress
        and prior
        and not prior.get("finished_at")
        and (args.all_horses or scope_meta_match)
    )
    if prior and not prior.get("finished_at") and not can_resume and not args.reset_progress:
        print(
            "[main] 進捗 JSON のスコープが変わったため --reset-progress 扱いで再開します",
            flush=True,
        )
    done_at_start = needed_count - len(todo)
    if can_resume:
        run_started_at = float(prior.get("run_started_at") or prior.get("started_at") or time.time())
        counts_cumulative_base = prior.get("counts_cumulative") or prior.get("counts") or {}
        print(
            f"[resume] 続きから再開: 累計 {done_at_start:,}/{needed_count:,} 完了, "
            f"今回 {len(todo):,} 頭 (interval={args.interval}s)",
            flush=True,
        )
        resumed = True
    else:
        run_started_at = time.time()
        counts_cumulative_base = {
            "success": 0, "failed": 0, "skipped": 0, "added_recursive": 0,
        }
        resumed = False

    remaining_at_session_start = len(todo)
    session_started_at = time.time()
    counts_session = {"success": 0, "failed": 0, "skipped": 0, "added_recursive": 0}
    scope_race_horses_count = (
        len(scope_race_horses) if scope_race_horses is not None else None
    )

    storage = HybridStorage()
    client = NetkeibaClient(interval=args.interval, auto_login=True)
    t0 = time.time()
    print(f"[main] スクレイピング開始: {len(todo):,} 頭 (interval={args.interval}s, "
          f"推定 {len(todo)*args.interval/3600:.1f}h, target_gen={target_gen}, "
          f"recursive={args.recursive}, resumed={resumed})", flush=True)
    save_progress(_progress_payload(
        needed_count=needed_count,
        run_started_at=run_started_at,
        session_started_at=session_started_at,
        remaining_at_session_start=remaining_at_session_start,
        index_done=0,
        todo_len=len(todo),
        counts_session=counts_session,
        counts_cumulative_base=counts_cumulative_base,
        target_gen=target_gen,
        recursive=args.recursive,
        resumed=resumed,
        scope_years=scope_years,
        scope_race_horses_count=scope_race_horses_count,
    ))
    i = 0
    try:
        # 動的拡張対応: list は処理中に append される可能性
        while i < len(todo):
            hid = todo[i]
            i += 1
            if hid in existing:
                counts_session["skipped"] += 1
                continue
            try:
                ok, new_gen5, err = scrape_one(hid, client, storage)
            except AccessError as e:
                logger.error("★ アクセスエラー — 停止: %s", e)
                save_progress(_progress_payload(
                    needed_count=needed_count,
                    run_started_at=run_started_at,
                    session_started_at=session_started_at,
                    remaining_at_session_start=remaining_at_session_start,
                    index_done=i,
                    todo_len=len(todo),
                    counts_session=counts_session,
                    counts_cumulative_base=counts_cumulative_base,
                    target_gen=target_gen,
                    recursive=args.recursive,
                    resumed=resumed,
                    scope_years=scope_years,
                    scope_race_horses_count=scope_race_horses_count,
                    stopped_at=time.time(),
                    reason=str(e),
                ))
                return 1
            if ok:
                counts_session["success"] += 1
                existing.add(hid)
                if args.recursive:
                    # 新規 gen=5 末端のうち、まだローカル不在のものを todo に追加
                    for nid in new_gen5:
                        if nid not in existing and nid not in set(todo):
                            todo.append(nid)
                            counts_session["added_recursive"] += 1
            else:
                counts_session["failed"] += 1
                if err:
                    logger.warning("失敗 %s: %s", hid, err)
            if i % args.save_progress_every == 0 or i == len(todo):
                el = time.time() - t0
                rate = i / el if el > 0 else 0
                remaining = len(todo) - i
                eta = remaining / rate / 3600 if rate > 0 else 0
                cum = _merge_counts(counts_cumulative_base, counts_session)
                done_cum = needed_count - remaining_at_session_start + i
                print(f"[main]  今回 {i}/{len(todo)} "
                      f"累計 {done_cum}/{needed_count} "
                      f"success={counts_session['success']} failed={counts_session['failed']} "
                      f"+rec={counts_session['added_recursive']} "
                      f"rate={rate*3600:.0f}/h eta={eta:.1f}h", flush=True)
                save_progress(_progress_payload(
                    needed_count=needed_count,
                    run_started_at=run_started_at,
                    session_started_at=session_started_at,
                    remaining_at_session_start=remaining_at_session_start,
                    index_done=i,
                    todo_len=len(todo),
                    counts_session=counts_session,
                    counts_cumulative_base=counts_cumulative_base,
                    target_gen=target_gen,
                    recursive=args.recursive,
                    resumed=resumed,
                    scope_years=scope_years,
                    scope_race_horses_count=scope_race_horses_count,
                    rate_per_hour=int(rate * 3600),
                    eta_hours=round(eta, 2),
                ))
    except KeyboardInterrupt:
        print("[main] Ctrl+C で停止 (再実行で続きから再開)", flush=True)
        save_progress(_progress_payload(
            needed_count=needed_count,
            run_started_at=run_started_at,
            session_started_at=session_started_at,
            remaining_at_session_start=remaining_at_session_start,
            index_done=i,
            todo_len=len(todo),
            counts_session=counts_session,
            counts_cumulative_base=counts_cumulative_base,
            target_gen=target_gen,
            recursive=args.recursive,
            resumed=resumed,
            scope_years=scope_years,
            scope_race_horses_count=scope_race_horses_count,
            stopped_at=time.time(),
            reason="KeyboardInterrupt",
        ))
    finally:
        client.close()

    el = time.time() - t0
    cum = _merge_counts(counts_cumulative_base, counts_session)
    print(f"[main] セッション完了: {counts_session} 累計={cum} ({el/3600:.2f}h)", flush=True)
    if i >= len(todo):
        save_progress(_progress_payload(
            needed_count=needed_count,
            run_started_at=run_started_at,
            session_started_at=session_started_at,
            remaining_at_session_start=remaining_at_session_start,
            index_done=i,
            todo_len=len(todo),
            counts_session=counts_session,
            counts_cumulative_base=counts_cumulative_base,
            target_gen=target_gen,
            recursive=args.recursive,
            resumed=resumed,
            scope_years=scope_years,
            scope_race_horses_count=scope_race_horses_count,
            finished_at=time.time(),
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
