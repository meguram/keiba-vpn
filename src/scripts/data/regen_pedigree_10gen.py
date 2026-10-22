"""不完全な 10 世代 JSON の再生成（派生データの更新。netkeiba へはアクセスしない）。

手順: 計画（対象の一覧・件数・サイズを記録）→ ステージング領域に作り直す → 完全性の検査（2,046 位置・世代別の埋まり率）→
      検査に通ったものだけ、旧版を退避してから置き換える（新規は追加）。完全なもの（既に 2,046 位置）は触らない。
使い方:
  python -m src.scripts.data.regen_pedigree_10gen --plan            # 一覧だけ作る（何も変えない）
  python -m src.scripts.data.regen_pedigree_10gen --apply           # 作り直し → 検査 → 退避 → 置き換え
退避先: data/local/backup/horse_pedigree_10gen_<日付>/<接頭辞>/<horse_id>.json、記録: data/local/meta/pedigree_10gen_regen_<日付>/manifest.json
元に戻す: 退避先のファイルを data/local/horse_pedigree_10gen/ へコピーし戻す（--restore <退避先>）。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.pedigree import build_horse_pedigree_10gen as bld  # noqa: E402

LOCAL = ROOT / "data" / "local"
FINAL = LOCAL / "horse_pedigree_10gen"
BIRTH_RANGE = ("2008", "2024")        # 出走馬（5 世代血統がある馬のうち、生年が 2008〜2024 年）


def _targets() -> list[str]:
    ids = []
    for d in sorted(bld.PED_DIR.iterdir()):
        if d.is_dir() and BIRTH_RANGE[0] <= d.name[:4] <= BIRTH_RANGE[1]:
            ids.extend(f.stem for f in sorted(d.iterdir()) if f.suffix == ".json")
    return ids


def _existing_n(path: Path) -> int | None:
    try:
        return int(json.loads(path.read_text(encoding="utf-8")).get("n_ancestors", 0))
    except (OSError, ValueError):
        return None


def plan() -> dict:
    ids = _targets()
    full, incomplete, missing = [], [], []
    inc_bytes = 0
    for h in ids:
        p = FINAL / h[:4] / f"{h}.json"
        if not p.is_file():
            missing.append(h)
            continue
        n = _existing_n(p)
        if n is not None and n >= bld.EXPECTED_POSITIONS:
            full.append(h)
        else:
            incomplete.append(h)
            inc_bytes += p.stat().st_size
    return {"targets": len(ids), "full_existing": full, "incomplete_existing": incomplete, "missing": missing,
            "incomplete_existing_bytes": inc_bytes, "location": str(FINAL)}


CRITERION = "maternal"      # 父方は分析に使わないため、既定は母方の完全性（世代 1〜10 の母方 1,023 位置）で判定する


def passes(doc: dict, criterion: str | None = None) -> bool:
    if (criterion or CRITERION) == "maternal":
        return bool(doc.get("maternal_complete")) and doc.get("n_maternal") == bld.EXPECTED_MATERNAL and all(
            abs(float(v) - 1.0) < 1e-9 for v in (doc.get("maternal_fill") or {}).values()) and len(doc.get("maternal_fill") or {}) == 10
    return bool(doc.get("complete")) and doc.get("n_ancestors") == bld.EXPECTED_POSITIONS and all(
        abs(float(v) - 1.0) < 1e-9 for v in (doc.get("fill") or {}).values()) and len(doc.get("fill") or {}) == 10


def apply(stamp: str) -> dict:
    pl = plan()
    meta_dir = LOCAL / "meta" / f"pedigree_10gen_regen_{stamp}"
    meta_dir.mkdir(parents=True, exist_ok=True)
    backup = LOCAL / "backup" / f"horse_pedigree_10gen_{stamp}"
    stage = LOCAL / f"horse_pedigree_10gen_staging_{stamp}"
    todo = pl["incomplete_existing"] + pl["missing"]
    (meta_dir / "plan.json").write_text(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in pl.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    (meta_dir / "replace_targets.txt").write_text("\n".join(pl["incomplete_existing"]), encoding="utf-8")
    print(f"[plan] 対象 {pl['targets']}（完全な既存 {len(pl['full_existing'])} / 不完全な既存 {len(pl['incomplete_existing'])} / 無い {len(pl['missing'])}）", flush=True)

    bld.OUT_DIR = stage                                  # ビルダーの出力先をステージングに
    stage.mkdir(parents=True, exist_ok=True)
    idx = bld.build_all_index()
    t0 = time.time()
    built = 0
    for i, h in enumerate(todo):
        _, ok, _n = bld._process_horse(h, idx)
        built += int(ok)
        if (i + 1) % 5000 == 0:
            print(f"[build] {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    replaced, added, partial, failed = [], [], [], []
    backup_bytes = 0
    for h in todo:
        sp = stage / h[:4] / f"{h}.json"
        if not sp.is_file():
            failed.append(h)
            continue
        doc = json.loads(sp.read_text(encoding="utf-8"))
        if not passes(doc):
            partial.append({"horse_id": h, "n_ancestors": doc.get("n_ancestors"), "missing_5gen": (doc.get("coverage") or {}).get("missing_5gen_data", [])[:5]})
            continue
        fp = FINAL / h[:4] / f"{h}.json"
        if fp.is_file():
            bp = backup / h[:4] / f"{h}.json"
            bp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fp, bp)
            backup_bytes += bp.stat().st_size
            replaced.append(h)
        else:
            added.append(h)
        fp.parent.mkdir(parents=True, exist_ok=True)
        tmp = fp.with_suffix(".tmp")
        shutil.copy2(sp, tmp)
        tmp.replace(fp)                                  # 原子的に置き換え
    result = {"stamp": stamp, "todo": len(todo), "built": built, "replaced": len(replaced), "added": len(added),
              "partial_not_installed": len(partial), "failed": len(failed), "backup_dir": str(backup), "backup_bytes": backup_bytes,
              "staging_dir": str(stage), "skipped_full_existing": len(pl["full_existing"])}
    (meta_dir / "manifest.json").write_text(json.dumps({**result, "partial": partial[:200], "failed": failed[:200]}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[done]", json.dumps(result, ensure_ascii=False), flush=True)
    return result


def fill_report() -> dict:
    """出走馬（2008〜2024 年生まれ）の 10 世代の充足率。完全（2,046 位置）・母方が完全（1,023 位置）の割合と、世代別の平均埋まり率。"""
    ids = _targets()
    n = full = mat = absent = 0
    gen_sum = {str(g): 0.0 for g in range(1, 11)}
    mat_sum = {str(g): 0.0 for g in range(1, 11)}
    for h in ids:
        p = FINAL / h[:4] / f"{h}.json"
        if not p.is_file():
            absent += 1
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            absent += 1
            continue
        n += 1
        is_full = d.get("n_ancestors", 0) >= bld.EXPECTED_POSITIONS
        full += int(is_full)
        mat += int(bool(d.get("maternal_complete")) or is_full)
        fill = d.get("fill") or ({str(g): 1.0 for g in range(1, 11)} if is_full else {})
        mfill = d.get("maternal_fill") or ({str(g): 1.0 for g in range(1, 11)} if is_full else {})
        for g in gen_sum:
            gen_sum[g] += float(fill.get(g, 0.0))
            mat_sum[g] += float(mfill.get(g, 0.0))
    out = {"targets": len(ids), "docs": n, "absent": absent,
           "complete_2046": full, "complete_2046_rate": round(full / max(1, len(ids)), 4),
           "maternal_complete_1023": mat, "maternal_complete_rate": round(mat / max(1, len(ids)), 4),
           "fill_by_generation": {g: round(v / max(1, n), 4) for g, v in gen_sum.items()},
           "maternal_fill_by_generation": {g: round(v / max(1, n), 4) for g, v in mat_sum.items()}}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def restore(backup_dir: str) -> int:
    n = 0
    for f in Path(backup_dir).glob("*/*.json"):
        dst = FINAL / f.parent.name / f.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        n += 1
    print(f"[restore] {n} 件を戻しました")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--restore", metavar="BACKUP_DIR")
    ap.add_argument("--report", action="store_true", help="10 世代の充足率（完全・母方が完全・世代別の埋まり率）を出す")
    ap.add_argument("--stamp", default=datetime.now().strftime("%Y%m%d"))
    ap.add_argument("--criterion", choices=["maternal", "full"], default="maternal", help="置き換えの合格基準（既定: 母方の完全性）")
    a = ap.parse_args()
    global CRITERION
    CRITERION = a.criterion
    if a.restore:
        restore(a.restore)
        return 0
    if a.report:
        fill_report()
        return 0
    if a.apply:
        apply(a.stamp)
        return 0
    pl = plan()
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in pl.items()}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
