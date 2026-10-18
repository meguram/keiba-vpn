"""ドルメロ流（8年周期）の活性値が予測に効くかを tp-16 と同じ手順で測る。対照として周期 7・9・10 年でも同じ特徴量を作る。

活性値は (馬の受胎年 - 祖先種牡馬の生年) mod 周期。馬ごとに固定の情報なので、レース日以降の情報は使っていない（リークなし）。
余りが 0 の組は 0/8 の境界が未確定なので、優先祖先の判定からは外し、個数と父の 0 フラグだけを特徴量にする。

    python -m src.scripts.data.dormello_bakeoff --horse-births /tmp/horse_birthdays.csv \
        --ancestor-births /tmp/ancestor_birthdays.csv [--smoke] [--threads 8]
採用基準は tp-16 と同じ（valid 2022〜2024 × 2 seed、勝率 LogLoss の差の 95% 信頼区間の上限 < 0、効果 ≥ 0.0002、連対・複勝が 0.0002 を超えて悪化しない）。
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd



def _find_root() -> Path:
    for q in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (q / "AGENTS.md").exists() and (q / "src").is_dir():
            return q
    raise RuntimeError("リポジトリルートが見つかりません（リポジトリ内で実行してください）")


REPO_ROOT = _find_root()
os.environ.setdefault("POLARS_MAX_THREADS", "4")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.research.pedigree import dormello_activity as da  # noqa: E402
except ImportError:  # 未反映のホストでは、このファイルと同じ場所の dormello_activity.py を読む
    import importlib.util

    _spec = importlib.util.spec_from_file_location("dormello_activity", Path(__file__).resolve().with_name("dormello_activity.py"))
    da = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(da)

FEATURE_NAMES = ["act_sire", "act_dsire", "act_max3", "act_max3_gen", "act_max3_is_sire", "act_n_zero3", "act_sire_zero", "act_n_known3"]


def load_births(paths: list[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in paths:
        out.update(da.load_birthdays(p))
    return out


def male_ancestors_gen3(ped_dirs: list[Path], horse_ids: list[str]) -> dict[str, list[tuple[int, int, str]]]:
    """horse_id -> [(世代, 位置, 祖先ID)]（世代 3 以内の種牡馬 = 位置が偶数）。10 世代 → 5 世代の順に探す。"""
    out: dict[str, list[tuple[int, int, str]]] = {}
    for h in horse_ids:
        for d in ped_dirs:
            f = d / h[:4] / f"{h}.json"
            if not f.exists():
                continue
            try:
                anc = json.loads(f.read_text(encoding="utf-8")).get("ancestors") or []
            except Exception:
                continue
            out[h] = [(int(a["generation"]), int(a["position"]), a["horse_id"]) for a in anc
                      if a.get("horse_id") and int(a["generation"]) <= 3 and int(a["position"]) % 2 == 0]
            break
    return out


def conception_year_of(h: str, births: dict[str, str]) -> int | None:
    b = births.get(h)
    by = da.birth_year_of(h, births)
    if by is None:
        return None
    return da.conception_year(by, date.fromisoformat(b) if b and len(b) == 10 else None)


def activity_features(males: list[tuple[int, int, str]], cy: int, births: dict[str, str], cycle: int) -> dict:
    ph: dict[tuple[int, int], int | None] = {}
    for g, p, a in males:
        by = da.birth_year_of(a, births)
        ph[(g, p)] = None if by is None else (cy - by) % cycle
    known = {k: v for k, v in ph.items() if v is not None}
    nonzero = [(v, g, p) for (g, p), v in known.items() if v > 0]
    f = {"act_sire": ph.get((1, 0)), "act_dsire": ph.get((2, 2)), "act_n_zero3": sum(1 for v in known.values() if v == 0),
         "act_sire_zero": None if ph.get((1, 0)) is None else float(ph[(1, 0)] == 0), "act_n_known3": len(known)}
    if nonzero:
        mx = max(v for v, _, _ in nonzero)
        best = [(g, p) for v, g, p in nonzero if v == mx]
        g_best = min(g for g, _ in best)
        f.update(act_max3=mx, act_max3_gen=g_best, act_max3_is_sire=float((1, 0) in best and g_best == 1))
    return f


def build_group(horse_ids, males, cys, births, cycle) -> pd.DataFrame:
    rows = {}
    for h in horse_ids:
        if h in males and cys.get(h) is not None:
            rows[h] = activity_features(males[h], cys[h], births, cycle)
    t = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=FEATURE_NAMES)
    return t.astype(float).reindex(horse_ids)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horse-births", required=True, type=Path)
    ap.add_argument("--ancestor-births", required=True, type=Path)
    ap.add_argument("--ped-dirs", nargs="+", type=Path, default=[REPO_ROOT / "data/local/horse_pedigree_10gen", REPO_ROOT / "data/local/horse_pedigree_5gen"])
    ap.add_argument("--cycles", nargs="+", type=int, default=[8, 7, 9, 10])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None, help="結果 CSV の保存先（既定: work/bakeoff。書けなければ /tmp/dormello_bakeoff）")
    a = ap.parse_args()

    import mlflow
    from src.utils.project_env import load_project_dotenv
    load_project_dotenv()
    from src.pipeline.mlflow.train_utils import require_tracking_server
    from src.pipeline.models import feature_bakeoff as fb, tp05_data, zoo, zoo_runner

    work = REPO_ROOT / "data/local/modeling/training_pipeline"
    out_dir = a.out_dir or work / "bakeoff"
    if not (out_dir.exists() and os.access(out_dir, os.W_OK)):
        out_dir = Path("/tmp/dormello_bakeoff")
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [0, 1]
    mlflow.set_experiment("keiba-tp-feature-bakeoff")
    require_tracking_server()
    D = tp05_data.load_dataset(work)
    df, meta, folds = D["df"], D["meta"], D["folds"]
    EP, _EXTRAS, _PED = tp05_data.load_extras_ped(df, work)
    X0 = pd.concat([df[D["FEATS"]["base+S+C1"]], EP], axis=1)
    best = json.loads((work / "hpo/best_params.json").read_text(encoding="utf-8"))["best"]
    LP = zoo.tuned_params("lgb", best["lgb"]["params"])
    if a.smoke:
        folds = folds[:1]

    births = load_births([a.horse_births, a.ancestor_births])
    hids = pd.unique(df["horse_id"].astype(str)).tolist()
    t0 = time.time()
    males = male_ancestors_gen3(a.ped_dirs, hids)
    cys = {h: conception_year_of(h, births) for h in hids}
    print(f"血統 {len(males)}/{len(hids)} 頭、受胎年 {sum(v is not None for v in cys.values())} 頭 ({time.time() - t0:.0f}s)", flush=True)

    key = ["race_id", "horse_id"]
    fin_map = meta.set_index(key)["finish_position"]
    res0, OOF0 = zoo_runner.run_oof("lgb_dm_base", "lgb_binary", X0, meta, folds, seeds, a.threads, params=dict(LP))
    print(f"基準: win {res0['logloss_win'].mean():.5f} top2 {res0['logloss_top2'].mean():.5f} top3 {res0['logloss_top3'].mean():.5f}", flush=True)
    rows = []
    cycles = a.cycles[:1] if a.smoke else a.cycles
    for c in cycles:
        name = f"DM{c}"
        g = build_group(hids, males, cys, births, c)
        feat = g.reindex(df["horse_id"].astype(str)).set_axis(df.index)
        print(f"{name}: 列 {feat.shape[1]}, 欠損率 {feat.isna().mean().mean():.3f}", flush=True)
        X = pd.concat([X0, feat.add_prefix(f"bk_{name}_")], axis=1)
        t1 = time.time()
        res, oof = zoo_runner.run_oof(f"lgb_dm_{name}", "lgb_binary", X, meta, folds, seeds, a.threads, params=dict(LP))
        rows.append(fb.compare_oof(OOF0, oof, fin_map, name))
        r = rows[-1]
        print(f"{name}: win {res['logloss_win'].mean():.5f} Δ {r['Δwin']:+.5f} (上限 {r['win_hi']:+.5f}) top2 {r['Δtop2']:+.5f} top3 {r['Δtop3']:+.5f} 採用={r['採用']} ({time.time() - t1:.0f}s)", flush=True)
        pd.DataFrame(rows).set_index("variant").to_csv(out_dir / "dormello_vs_base.csv")
        gc.collect()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
