"""連結されて保存された払戻（複勝・ワイドが 1 つに潰れた 2024-01〜2026-03 の race_result）を、補正テーブル（payoff_fix）で複数行に直して焼き込む。

python -m src.scripts.maintenance.bake_payoff_fix --years 2024 2025 2026            # dry-run（件数と検証結果だけ）
python -m src.scripts.maintenance.bake_payoff_fix --years 2024 2025 2026 --apply    # 退避してから保存（HybridStorage.save）

- 対象: payoff_fix で曖昧（ambiguous）・解析不能でないレースのうち、保存済み race_result の払戻が連結形式のもの。
- 焼き込む前に、直した複勝・ワイドの組合せが、着順（同着を含む）から求めた期待集合（winning_combos）と一致することを確認し、一致しないレースは保存しない。
- 元の払戻は data/local/backup/payoff_fix_<日時>/ に JSON で退避。派生カテゴリ race_result_payoff も、同じ内容で作り直して保存する。
- 結果は data/local/meta/payoff_bake_report.json。曖昧なレースは別途 db.netkeiba から再取得する（src.scraper.run race-result <race_id>）。
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.pipeline.betting import winning_combos as W
from src.scraper import payoff_repair as pr
from src.scraper.row_data_extractor import extract_result_payoff
from src.scraper.storage import HybridStorage

ROOT = Path(__file__).resolve().parents[3]


def _finish(doc: dict) -> dict[int, int]:
    out = {}
    for e in doc.get("entries") or []:
        try:
            out[int(e.get("horse_number"))] = int(e.get("finish_position") or 0)
        except (TypeError, ValueError):
            continue
    return out


def _valid(doc: dict, payoff: dict) -> tuple[bool, str]:
    fin = _finish(doc)
    groups = W.finish_groups(fin)
    if len(groups) < 3:
        return False, "着順が3着未満"
    n = len(doc.get("entries") or [])
    got = W.normalize_payoff(payoff)
    wide_exp = W.winning_combos(groups, "wide")
    if set(got.get("wide", {})) != wide_exp:
        return False, f"ワイドの組が期待と不一致 got={sorted(got.get('wide', {}))} exp={sorted(wide_exp)}"
    fk = set(got.get("fukusho", {}))
    if fk != {c for c in W.winning_combos(groups, "fukusho", n_runners=n)} and fk != {(h,) for h in W.top_set(groups, 3)}:
        return False, f"複勝の組が期待と不一致 got={sorted(fk)}"
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2024, 2025, 2026])
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    s = HybridStorage()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    bdir = ROOT / f"data/local/backup/payoff_fix_{stamp}"
    todo: list[tuple[str, list[dict]]] = []
    for y in a.years:
        for rid, rows in pr.load_fix(y).items():
            todo.append((rid, rows))
    stats = {"candidates": len(todo), "already_ok": 0, "changed": 0, "invalid": 0, "saved": 0, "errors": 0, "no_doc": 0}
    invalid: dict[str, str] = {}
    changed_ids: list[str] = []
    if a.apply:
        bdir.mkdir(parents=True, exist_ok=True)

    def work(item: tuple[str, list[dict]]) -> None:
        rid, rows = item
        try:
            doc = s.load("race_result", rid)
            if not isinstance(doc, dict):
                stats["no_doc"] += 1
                return
            if not pr.is_concatenated(doc.get("payoff")):
                stats["already_ok"] += 1
                return
            new_payoff = pr.apply_fix(doc["payoff"], rows)
            ok, why = _valid(doc, new_payoff)
            if not ok:
                stats["invalid"] += 1
                invalid[rid] = why
                return
            stats["changed"] += 1
            changed_ids.append(rid)
            if not a.apply:
                return
            (bdir / f"{rid}.json").write_text(json.dumps({"race_id": rid, "payoff": doc["payoff"]}, ensure_ascii=False), encoding="utf-8")
            latest = s.load("race_result", rid)  # 並走する取得で更新されていないか、保存直前に読み直す
            if not pr.is_concatenated(latest.get("payoff")):
                return
            new_doc = dict(latest)
            new_doc["payoff"] = pr.apply_fix(latest["payoff"], rows)
            s.save("race_result", rid, new_doc)
            try:
                s.save("race_result_payoff", rid, extract_result_payoff(new_doc))
            except Exception:  # noqa: BLE001  派生カテゴリは補助。失敗しても本体は保存済み
                pass
            stats["saved"] += 1
        except Exception as exc:  # noqa: BLE001
            stats["errors"] += 1
            invalid[rid] = f"error: {exc}"

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        for i, _ in enumerate(ex.map(work, todo), 1):
            if i % 500 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(todo)} 変更={stats['changed']} 保存={stats['saved']} 不一致={stats['invalid']} err={stats['errors']}", flush=True)
    rep = {"mode": "apply" if a.apply else "dry-run", "backup_dir": str(bdir) if a.apply else None, "seconds": round(time.time() - t0), **stats,
           "invalid_races": invalid, "changed_sample": sorted(changed_ids)[:20]}
    out = ROOT / "data/local/meta/payoff_bake_report.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("invalid_races", "changed_sample")}, ensure_ascii=False), flush=True)
    if invalid:
        print(f"不一致/失敗 {len(invalid)} 件 → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
