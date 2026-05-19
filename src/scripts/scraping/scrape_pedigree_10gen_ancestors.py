"""10gen 展開のために、5gen 内に登場する未取得祖先をスクレイピング。

実行ロジック:
    1. data/local/horse_pedigree_5gen/ をスキャンし、登場する全祖先 ID 集合を作る
    2. ローカルにも GCS にも無い祖先を todo として抽出 (= スクレイピング対象)
    3. NetkeibaClient(interval=3.0) で順次取得し、ローカル + GCS に保存
    4. 取得時に新たに発見した祖先 (= 取得馬の 5gen 内に出てくる祖先) も todo に追加
       → 真の 10gen フル覆蓋まで再帰的に取得

進捗ログ:
    /tmp/scrape_10gen_ancestors.log (主) と stderr / stdout の併用
    /home/hirokiakataoka/project/myproject/keiba-vpn/data/research/pedigree_10gen_3view/scrape_progress.json

Usage:
    nohup python -m src.scripts.scraping.scrape_pedigree_10gen_ancestors \
        > /tmp/scrape_10gen_ancestors.log 2>&1 &
    disown

    # 1 頭で動作確認
    python -m src.scripts.scraping.scrape_pedigree_10gen_ancestors --limit 3

    # 進捗確認
    cat data/research/pedigree_10gen_3view/scrape_progress.json
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
from src.scraper.client import NetkeibaClient  # noqa: E402
from src.scraper.storage import HybridStorage  # noqa: E402

logger = logging.getLogger("scrape_10gen_ancestors")

PED_URL = "https://db.netkeiba.com/horse/ped/{horse_id}/"
PED_DIR = ROOT / "data/local/horse_pedigree_5gen"
PROGRESS_PATH = ROOT / "data/research/pedigree_10gen_3view/scrape_progress.json"


def _prefix_of(horse_id: str) -> str:
    return horse_id[:4] if len(horse_id) >= 4 else "0000"


def _local_path(horse_id: str) -> Path:
    return PED_DIR / _prefix_of(horse_id) / f"{horse_id}.json"


def _save_local(horse_id: str, data: dict) -> None:
    p = _local_path(horse_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def collect_ancestor_ids(file_path: Path) -> set[str]:
    try:
        data = json.loads(file_path.read_text())
    except Exception:
        return set()
    return {
        (a.get("horse_id") or "").strip()
        for a in data.get("ancestors") or []
        if (a.get("horse_id") or "").strip()
    }


def scan_state() -> tuple[set[str], set[str]]:
    """ローカル 5gen を走査して、保存済み horse_id と未取得祖先 ID を返す。"""
    existing: set[str] = set()
    all_anc: set[str] = set()
    n = 0
    t0 = time.time()
    for d in sorted(PED_DIR.iterdir()):
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.suffix != ".json":
                continue
            existing.add(f.stem)
            all_anc.update(collect_ancestor_ids(f))
            n += 1
    missing = all_anc - existing
    print(f"[scan] local={len(existing):,} ancestors={len(all_anc):,} "
          f"missing={len(missing):,} ({time.time()-t0:.1f}s)", flush=True)
    return existing, missing


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
        "source": "phase_10gen_ancestors",
    }


class AccessError(Exception):
    pass


def scrape_one(
    horse_id: str,
    client: NetkeibaClient,
    storage: HybridStorage,
) -> tuple[bool, set[str], str]:
    """1 頭分をスクレイピング。

    Returns:
        (success, newly_found_ancestor_ids, error_msg)
    """
    try:
        html = client.fetch(PED_URL.format(horse_id=horse_id))
        ancestors = parse_blood_table_5gen(html)
        if len(ancestors) < 5:
            return False, set(), f"祖先不足({len(ancestors)})"
        sex = parse_horse_sex_from_ped_html(html, horse_id)
        rec = build_record(horse_id, ancestors, sex)
        # GCS + ローカル両方に保存
        storage.save("horse_pedigree_5gen", horse_id, rec)
        _save_local(horse_id, rec)
        anc_ids = {
            (a.get("horse_id") or "").strip()
            for a in ancestors
            if (a.get("horse_id") or "").strip()
        }
        return True, anc_ids, ""
    except Exception as e:
        s = str(e)
        if any(c in s for c in (
            "403", "429", "503", "Forbidden", "Too Many",
            "Service Unavailable", "ConnectionError",
            "Connection aborted", "Network is unreachable",
        )):
            raise AccessError(f"アクセスエラー ({horse_id}): {s}") from e
        return False, set(), s


def save_progress(state: dict) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=3.0,
                        help="スクレイピング間隔 (秒) — netkeiba は >=3.0 推奨")
    parser.add_argument("--limit", type=int, default=None,
                        help="先頭 N 頭で動作確認")
    parser.add_argument("--max-rounds", type=int, default=20,
                        help="再帰的に新祖先取得を繰り返す最大ラウンド数")
    parser.add_argument("--save-progress-every", type=int, default=50,
                        help="N 頭ごとに進捗 JSON を更新")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    existing, missing = scan_state()
    todo = sorted(missing)
    if args.limit:
        todo = todo[:args.limit]
        print(f"[main] --limit により {len(todo):,} に絞る", flush=True)

    if not todo:
        print("[main] 未取得馬なし、終了", flush=True)
        return 0

    storage = HybridStorage()
    client = NetkeibaClient(interval=args.interval, auto_login=True)
    counts = {"success": 0, "failed": 0, "skipped": 0}
    t0 = time.time()
    started_at = time.time()
    print(f"[main] スクレイピング開始: {len(todo):,} 頭 (interval={args.interval}s, "
          f"推定 {len(todo)*args.interval/3600:.1f}h)", flush=True)
    try:
        for i, hid in enumerate(todo):
            if hid in existing:
                counts["skipped"] += 1
                continue
            try:
                ok, new_anc, err = scrape_one(hid, client, storage)
            except AccessError as e:
                logger.error("★ アクセスエラー — 停止: %s", e)
                save_progress({
                    "stopped_at": time.time(),
                    "reason": str(e),
                    "counts": counts,
                    "i": i,
                    "n_total": len(todo),
                })
                return 1
            if ok:
                counts["success"] += 1
                existing.add(hid)
            else:
                counts["failed"] += 1
                if err:
                    logger.warning("失敗 %s: %s", hid, err)
            done = i + 1
            if done % args.save_progress_every == 0 or done == len(todo):
                el = time.time() - t0
                rate = done / el if el > 0 else 0
                eta = (len(todo) - done) / rate / 3600 if rate > 0 else 0
                print(f"[main]  {done}/{len(todo)} "
                      f"success={counts['success']} failed={counts['failed']} "
                      f"rate={rate*3600:.0f}/h eta={eta:.1f}h", flush=True)
                save_progress({
                    "updated_at": time.time(),
                    "started_at": started_at,
                    "done": done,
                    "n_total": len(todo),
                    "counts": counts,
                    "rate_per_hour": int(rate * 3600),
                    "eta_hours": round(eta, 2),
                })
    except KeyboardInterrupt:
        print("[main] Ctrl+C で停止", flush=True)
    finally:
        client.close()

    el = time.time() - t0
    print(f"[main] 完了: {counts} ({el/3600:.2f}h)", flush=True)
    save_progress({
        "finished_at": time.time(),
        "started_at": started_at,
        "n_total": len(todo),
        "counts": counts,
        "elapsed_hours": round(el / 3600, 2),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
