"""ローカル 5gen データを再帰展開して、1 馬 1 ファイルの 10gen JSON を生成。

既存 5gen データ (``data/local/horse_pedigree_5gen/{prefix}/{horse_id}.json``)
の 5 代目祖先 X それぞれについて、X 自身の 5gen データを連結することで
最大 10 世代分の祖先情報を 1 ファイルにまとめる。

出力:
    data/local/horse_pedigree_10gen/{prefix}/{horse_id}.json
    {
      "horse_id": str,
      "sex": str,
      "sire": str, "dam": str, "dam_sire": str,
      "ancestors": [
        // gen 1-5: 5gen 内データ
        {"generation": 1, "position": 0, "name": "...", "horse_id": "...",
         "side": "father"|"dam"|"dam_sire_line"|"dam_dam_line"},
        ...
        // gen 6-10: 5gen 目祖先の 5gen データを展開
        {"generation": 6, "position": 64, "name": "...", "horse_id": "...",
         "side": ..., "via": "<5gen 目祖先 horse_id>"},
        ...
      ],
      "n_ancestors": int,
      "coverage": {
        "n_5gen_ancestors": int,
        "n_5gen_ancestors_resolved": int,
        "missing_5gen_data": [<horse_id>, ...]   # 5 代目祖先で 5gen データを持たない馬
      },
      "source": "10gen_merge",
      "generated_at": ISO timestamp
    }

side 判定:
    - g=1, p=0      : father
    - g=1, p=1      : dam (母そのもの)
    - g>=2:
        * bit (g-1) of p == 0 → father (1 代目が父)
        * bit (g-1) of p == 1 →
            bit (g-2) of p == 0 → dam_sire_line (1 代目=母, 2 代目=父)
            bit (g-2) of p == 1 → dam_dam_line  (1 代目=母, 2 代目=母)

Usage:
    python -m src.research.pedigree.build_horse_pedigree_10gen [--limit N]
        [--workers 8] [--target-list path]

性能:
    ローカル 5gen 51,048 馬 / 全 ancestor 集合の precompute → 1 馬約 1-2 ms。
    全体で約 1-2 分の見込み。
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("build_10gen")

PED_DIR = ROOT / "data/local/horse_pedigree_5gen"
OUT_DIR = ROOT / "data/local/horse_pedigree_10gen"


def _prefix_of(horse_id: str) -> str:
    return horse_id[:4] if len(horse_id) >= 4 else "0000"


def _local_path_5gen(horse_id: str) -> Path:
    return PED_DIR / _prefix_of(horse_id) / f"{horse_id}.json"


def _local_path_10gen(horse_id: str) -> Path:
    return OUT_DIR / _prefix_of(horse_id) / f"{horse_id}.json"


def _side_of(g: int, p: int) -> str:
    """generation g, position p から side を判定 (4 値)。"""
    if g < 1 or p < 0:
        return "unknown"
    is_dam_side = ((p >> (g - 1)) & 1) == 1
    if not is_dam_side:
        return "father"
    if g == 1:
        return "dam"
    is_dam_dam = ((p >> (g - 2)) & 1) == 1
    return "dam_dam_line" if is_dam_dam else "dam_sire_line"


def load_5gen(horse_id: str) -> dict | None:
    p = _local_path_5gen(horse_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def build_10gen_record(horse_id: str, idx: dict[str, dict]) -> dict | None:
    """1 馬の 10gen ancestors を構築。

    idx: {horse_id -> 5gen data dict} の precompute インデックス
    """
    rec = idx.get(horse_id)
    if not rec:
        return None
    ancestors_out: list[dict] = []
    seen: set[tuple[int, int]] = set()
    missing_5gen: list[str] = []
    n_5gen_resolved = 0
    n_5gen_total = 0

    # gen 1-5
    for anc in rec.get("ancestors") or []:
        try:
            g = int(anc.get("generation", 0))
            p = int(anc.get("position", -1))
        except (TypeError, ValueError):
            continue
        if g < 1 or g > 5 or p < 0:
            continue
        hid = (anc.get("horse_id") or "").strip()
        nm = anc.get("name") or ""
        side = _side_of(g, p)
        if (g, p) in seen:
            continue
        seen.add((g, p))
        ancestors_out.append({
            "generation": g,
            "position": p,
            "name": nm,
            "horse_id": hid,
            "side": side,
        })

    # gen 6-10: 5gen 目祖先 X (g=5) の 5gen を展開
    for anc in rec.get("ancestors") or []:
        try:
            g = int(anc.get("generation", 0))
            p_X = int(anc.get("position", -1))
        except (TypeError, ValueError):
            continue
        if g != 5 or p_X < 0:
            continue
        X_hid = (anc.get("horse_id") or "").strip()
        if not X_hid:
            continue
        n_5gen_total += 1
        sub = idx.get(X_hid)
        if not sub:
            missing_5gen.append(X_hid)
            continue
        n_5gen_resolved += 1
        X_side = _side_of(5, p_X)
        for sa in sub.get("ancestors") or []:
            try:
                sg = int(sa.get("generation", 0))
                sp = int(sa.get("position", -1))
            except (TypeError, ValueError):
                continue
            if sg < 1 or sp < 0:
                continue
            # 元馬から見た gen = 5 + sg, position = p_X * 2^sg + sp
            mg = 5 + sg
            mp = (p_X << sg) | sp
            if mg > 10:
                continue
            if (mg, mp) in seen:
                continue
            seen.add((mg, mp))
            ancestors_out.append({
                "generation": mg,
                "position": mp,
                "name": sa.get("name") or "",
                "horse_id": (sa.get("horse_id") or "").strip(),
                "side": X_side,
                "via": X_hid,
            })

    ancestors_out.sort(key=lambda x: (x["generation"], x["position"]))
    out = {
        "horse_id": horse_id,
        "sex": rec.get("sex", ""),
        "sire": rec.get("sire", ""),
        "dam": rec.get("dam", ""),
        "dam_sire": rec.get("dam_sire", ""),
        "ancestors": ancestors_out,
        "n_ancestors": len(ancestors_out),
        "coverage": {
            "n_5gen_ancestors": n_5gen_total,
            "n_5gen_ancestors_resolved": n_5gen_resolved,
            "missing_5gen_data": sorted(set(missing_5gen)),
        },
        "source": "10gen_merge",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return out


def _process_horse(horse_id: str, idx: dict[str, dict]) -> tuple[str, bool, int]:
    out = build_10gen_record(horse_id, idx)
    if not out:
        return horse_id, False, 0
    p = _local_path_10gen(horse_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return horse_id, True, out["n_ancestors"]


def build_all_index() -> dict[str, dict]:
    """5gen のローカルファイルから {horse_id -> data} の dict を構築。"""
    idx: dict[str, dict] = {}
    n = 0
    t0 = time.time()
    print(f"[idx] ローカル 5gen を構築中... {PED_DIR}", flush=True)
    for d in PED_DIR.iterdir():
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.suffix != ".json":
                continue
            try:
                idx[f.stem] = json.loads(f.read_text())
            except Exception:
                pass
            n += 1
            if n % 10000 == 0:
                print(f"[idx]   {n:,} files ({time.time()-t0:.1f}s)", flush=True)
    print(f"[idx] 完了: {len(idx):,} 馬 ({time.time()-t0:.1f}s)", flush=True)
    return idx


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="先頭 N 馬で動作確認")
    parser.add_argument("--target-list", type=str, default=None,
                        help="対象 horse_id を 1 行ずつ含むファイル")
    parser.add_argument("--target-id", type=str, default=None,
                        help="特定 1 馬の horse_id のみ生成")
    parser.add_argument("--skip-existing", action="store_true",
                        help="既存 10gen ファイルをスキップ (差分更新)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 対象 horse_id 決定
    if args.target_id:
        ids = [args.target_id]
    elif args.target_list:
        ids = [x.strip() for x in Path(args.target_list).read_text().splitlines() if x.strip()]
    else:
        ids = []
        for d in sorted(PED_DIR.iterdir()):
            if not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                if f.suffix == ".json":
                    ids.append(f.stem)
    if args.limit:
        ids = ids[:args.limit]
    if args.skip_existing:
        before = len(ids)
        ids = [h for h in ids if not _local_path_10gen(h).exists()]
        print(f"[main] --skip-existing: {before} -> {len(ids)}", flush=True)
    print(f"[main] 対象 horse_id: {len(ids):,}", flush=True)

    idx = build_all_index()

    counts = {"success": 0, "failed": 0}
    n_anc_sum = 0
    t0 = time.time()
    for i, hid in enumerate(ids):
        _, ok, n = _process_horse(hid, idx)
        if ok:
            counts["success"] += 1
            n_anc_sum += n
        else:
            counts["failed"] += 1
        if (i + 1) % 5000 == 0:
            print(f"[main]   {i+1:,}/{len(ids):,} success={counts['success']} "
                  f"avg_n_anc={n_anc_sum/max(1,counts['success']):.0f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
            gc.collect()
    elapsed = time.time() - t0
    avg_anc = n_anc_sum / max(1, counts["success"])
    print(f"[main] 完了: {counts} avg_n_anc={avg_anc:.0f} ({elapsed:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
