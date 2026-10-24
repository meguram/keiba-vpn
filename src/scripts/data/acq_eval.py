"""新規データ候補（非 JRA 過去走 NJ・母側のきょうだい SIB・セリ価格 SALE・馬主 ID OWN）を v5 の構成 d に累積して評価する（v6_eval と同じ手順）。

python -m src.scripts.data.acq_eval build            # 群の特徴量を ACQ_DIR/groups/*.parquet に保存（runs_all.parquet / ped_dam.parquet が前提）
python -m src.scripts.data.acq_eval list
python -m src.scripts.data.acq_eval run NAME [--tag s1]     # 構成 NAME の OOF（既定: valid 2022〜24 expanding・seed 0,1。基準 d_CY_BEST は v6 の OOF を使う）
python -m src.scripts.data.acq_eval compare [--tag s1]       # 基準との対の比較（採否は feature_bakeoff.compare_oof の基準）
test 2025・2026 年は使わない。市場オッズ・人気は使わない。環境変数 ACQ_DIR（既定 data/local/modeling/acq）, ACQ_BASE_OOF（基準 OOF の場所）。
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd

os.environ.setdefault("TP40_THREADS", os.environ.get("ACQ_THREADS", "4"))
os.environ.setdefault("TP40_LOAD_LIMIT", os.environ.get("ACQ_LOAD_LIMIT", "14"))

from src.pipeline.features import acq_groups as ag  # noqa: E402
from src.scripts.data import tp27_eval as t  # noqa: E402
from src.scripts.data import tp40_eval as e40  # noqa: E402
from src.scripts.data import v5_select_eval as sel  # noqa: E402

KEY = ["race_id", "horse_id"]
ROOT = Path(__file__).resolve().parents[3]
ACQ = Path(os.environ.get("ACQ_DIR") or ROOT / "data/local/modeling/acq")
BASE_OOF = Path(os.environ.get("ACQ_BASE_OOF") or e40.V / "v6/eval/oof_d_CY_BEST.parquet")
YEARS = [2022, 2023, 2024]
SEEDS = [0, 1]
BASE_CFG = {"base": "fresh", "groups": ["CY", "BEST"]}
CONFIGS: dict[str, list[str]] = {
    "NJ": ["NJ"],
    "SIB": ["SIB"],
    "NJ_SIB": ["NJ", "SIB"],
    "SALE": ["SALE"],
    "OWN": ["OWN"],
    "SALE_SIB": ["SALE", "SIB"],
    "PT": ["PT"],
}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cmd_build(a) -> None:
    df, base, meta, folds, lp = t.setup()
    keys = df[KEY + ["date"]].copy()
    runs = pd.read_parquet(ACQ / "runs_all.parquet")
    ped = pd.read_parquet(ACQ / "ped_dam.parquet")
    (ACQ / "groups").mkdir(parents=True, exist_ok=True)
    fns = [("NJ", lambda: ag.nonjra_features(runs, keys)), ("SIB", lambda: ag.sibling_features(runs, ped, keys))]
    if (ACQ / "workouts_partner.parquet").exists():
        fns.append(("PT", lambda: ag.partner_features(pd.read_parquet(ACQ / "workouts_partner.parquet"), keys)))
    for name, fn in fns:
        if a.only and name not in a.only:
            continue
        G = fn()
        G.insert(0, "horse_id", keys["horse_id"].to_numpy())
        G.insert(0, "race_id", keys["race_id"].to_numpy())
        G.to_parquet(ACQ / "groups" / f"{name}.parquet", index=False)
        _log(f"{name}: {G.shape[1] - 2} 列 × {len(G)} 行")


def load_group(name: str, df: pd.DataFrame) -> pd.DataFrame:
    G = pd.read_parquet(ACQ / "groups" / f"{name}.parquet")
    G["race_id"], G["horse_id"] = G["race_id"].astype(str), G["horse_id"].astype(str)
    m = df[KEY].astype(str).merge(G, on=KEY, how="left")
    assert len(m) == len(df)
    m.index = df.index
    return m.drop(columns=KEY)


def build_frame(groups: list[str], df: pd.DataFrame) -> pd.DataFrame:
    X = sel.build_frame(BASE_CFG | {}, df)
    G = pd.concat([load_group(g, df) for g in groups], axis=1)
    dup = set(G.columns) & set(X.columns)
    assert not dup, f"列名が重なる: {sorted(dup)[:5]}"
    G.index = X.index
    return pd.concat([X, G], axis=1)


def _paths(tag: str) -> tuple[Path, Path]:
    d = ACQ / ("eval" if not tag else f"eval_{tag}")
    d.mkdir(parents=True, exist_ok=True)
    return d, d / "levels.json"


def cmd_run(a) -> None:
    d, resp = _paths(a.tag)
    years = [int(x) for x in a.years.split(",")] if a.years else YEARS
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else SEEDS
    res = e40._jload(resp) or {}
    if a.name in res and (d / f"oof_{a.name}.parquet").exists():
        _log(f"{a.name}: 既に完了")
        return
    df, base, meta, folds, lp = t.setup()
    X = build_frame(CONFIGS[a.name], df)
    _log(f"{a.name}: 列 {X.shape[1]}（基準 d は 1,142 列）")
    lv, oof = e40._run_subset(a.name, X, meta, folds, lp, years, seeds)
    oof.to_parquet(d / f"oof_{a.name}.parquet", index=False)
    res = e40._jload(resp) or {}
    res[a.name] = {"levels": lv, "cols": int(X.shape[1]), "groups": CONFIGS[a.name], "years": years, "seeds": seeds}
    e40._jsave(resp, res)


def cmd_compare(a) -> None:
    d, resp = _paths(a.tag)
    df, base, meta, folds, lp = t.setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    rm = pd.read_parquet(e40.OUT / "run_meta.parquet")
    rm["race_id"], rm["horse_id"] = rm["race_id"].astype(str), rm["horse_id"].astype(str)
    b = pd.read_parquet(BASE_OOF)
    out: dict = {"pairs": {}}
    for n in CONFIGS:
        f = d / f"oof_{n}.parquet"
        if not f.exists():
            continue
        o = pd.read_parquet(f)
        c = t._cmp(b, o, fin_map, f"{n} vs d")
        by_year = {}
        for y in YEARS:
            ids = set(meta.loc[meta["year"] == y, "race_id"].astype(str))
            ma, mb = b["race_id"].astype(str).isin(ids), o["race_id"].astype(str).isin(ids)
            if ma.sum() > 1000 and mb.sum() > 1000:
                cy = t._cmp(b[ma], o[mb], fin_map, f"{n} vs d {y}")
                by_year[str(y)] = {"Δwin": cy["Δwin"], "win_hi": cy["win_hi"], "Δtop2": cy["Δtop2"], "Δtop3": cy["Δtop3"]}
        sub = e40._subset_cmp(b, o, fin_map, rm, f"{n} vs d")
        out["pairs"][n] = {"overall": c, "by_year": by_year, "subsets": sub}
        _log(f"{n} vs d: Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f} 採用={c['採用']}")
    out["levels"] = e40._jload(resp)
    e40._jsave(d / "results.json", out)


def cmd_list(a) -> None:
    for n, g in CONFIGS.items():
        print(n, g)


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--only", nargs="*", default=[], help="作る群を限る（既存の群は書き換えない）")
    r = sp.add_parser("run")
    r.add_argument("name", choices=list(CONFIGS))
    r.add_argument("--tag", default="")
    r.add_argument("--years", default="")
    r.add_argument("--seeds", default="")
    c = sp.add_parser("compare")
    c.add_argument("--tag", default="")
    sp.add_parser("list")
    a = ap.parse_args()
    {"build": cmd_build, "run": cmd_run, "compare": cmd_compare, "list": cmd_list}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
