"""全祖先 horse_id → name の辞書を 5gen/10gen ローカルから構築する。

既存 ``sid_to_name.json`` (4,773 件) は主流種牡馬中心で、L3 の海外祖先名が
拾えない。本スクリプトでは:
    - data/local/horse_pedigree_5gen/ の全 ancestors を走査
    - 各祖先の最初に出てきた name を採用 (重複時は文字数の長い方を優先)
    - 結果を data/research/bloodline_meta_cluster/sid_to_name_full.json に保存

Usage:
    python -m src.research.pedigree.build_full_sid_to_name
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PED_DIR = ROOT / "data/local/horse_pedigree_5gen"
OUT_PATH = ROOT / "data/page_reference/note_aptitude_race/sid_to_name_full.json"


def main() -> int:
    name_map: dict[str, str] = {}
    # 既存 sid_to_name を seed として読み込む
    seed = ROOT / "data/page_reference/note_aptitude_race/sid_to_name.json"
    if seed.exists():
        try:
            name_map.update(json.loads(seed.read_text()))
        except Exception:
            pass
    print(f"[seed] {len(name_map):,} entries", flush=True)

    n_files = 0
    t0 = time.time()
    for d in PED_DIR.iterdir():
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.suffix != ".json":
                continue
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue
            for a in data.get("ancestors") or []:
                hid = (a.get("horse_id") or "").strip()
                nm = (a.get("name") or "").strip()
                if not hid or not nm:
                    continue
                cur = name_map.get(hid)
                # より長い (=完全な) name を優先
                if cur is None or len(nm) > len(cur):
                    name_map[hid] = nm
            n_files += 1
            if n_files % 10000 == 0:
                print(f"[scan] {n_files:,} files / map={len(name_map):,} "
                      f"({time.time()-t0:.1f}s)", flush=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(name_map, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {len(name_map):,} entries → {OUT_PATH} ({time.time()-t0:.1f}s)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
