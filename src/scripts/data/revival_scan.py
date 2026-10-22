"""復活セットの台帳づくり（読み取りのみ・学習なし）: 保存済みの評価結果から、群ごとの増分を集める。

1. extract_pooled: 各段階（stage1/stage2/v6/…）の results.json から、群ごとの Δ勝率・Δ連対・Δ複勝と片側 95% 上限（pooled）を集める。
2. oof_year_deltas: 保存済みの OOF（oof_<群>.parquet と base.parquet）から、年別の Δ（paired、レース単位ブートストラップ）を再集計する。
   学習は走らせない。OOF が保存されていない群は年別を出せず、台帳では『要再集計』にする。
使い方（ホスト）: python -m src.scripts.data.revival_scan --out /tmp/revival_scan.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["race_id", "horse_id"]
EPS = 1e-15
N_BOOT = 1000
SEED = 20261022

MODELING = Path(os.environ.get("KEIBA_MODELING_DIR", "/home/jovyan/work/keiba-vpn/data/local/modeling"))
V4 = MODELING / "training_pipeline_v4"
WORK = Path(os.environ.get("KEIBA_WORK_DIR", "/home/jovyan/work"))
RANK_DIR = Path(os.environ.get("KEIBA_RANK_DIR", "/home/jovyan/work/keiba-vpn/data/local/features/target/rank_tbl"))

STAGE_DIRS = ["tp32", "tp33", "tp34", "tp35", "tp37", "tp39", "tp40", "tp41", "tp42", "tp57_growth"]


def walk_records(o, ctx: list[str], src: str, out: list[dict]) -> None:
    """results.json の中から Δwin を持つ辞書（群ごとの評価）を再帰的に拾う。"""
    if isinstance(o, dict):
        if "Δwin" in o:
            r = {"file": src, "ctx": ctx[-3:]}
            for k in ("variant", "win_base", "win_new", "Δwin", "win_hi", "Δtop2", "top2_hi", "Δtop3", "top3_hi"):
                if k in o:
                    r[k] = o[k]
            out.append(r)
            return
        for k, v in o.items():
            walk_records(v, ctx + [str(k)], src, out)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk_records(v, ctx + [str(i)], src, out)


def result_files() -> list[Path]:
    files: list[Path] = []
    for d in STAGE_DIRS:
        files += [Path(p) for p in glob.glob(str(V4 / d / "stage*" / "results.json"))]
    files += [Path(p) for p in glob.glob(str(WORK / "tp38_out" / "stage*" / "results.json"))]
    files += [Path(p) for p in glob.glob(str(WORK / "tp60_out" / "eval" / "*" / "results.json"))]
    files += [Path(p) for p in glob.glob(str(WORK / "acq" / "eval" / "*.json"))]
    files += [Path(p) for p in glob.glob(str(V4 / "v6" / "eval*" / "results*.json"))]
    files += [V4 / "v5_select" / "results.json"]
    return [p for p in files if p.exists()]


def extract_pooled() -> list[dict]:
    recs: list[dict] = []
    for f in result_files():
        try:
            walk_records(json.loads(f.read_text(encoding="utf-8")), [], str(f), recs)
        except (OSError, ValueError):
            continue
    return recs


def load_labels(years: list[int]) -> pd.DataFrame:
    parts = []
    for y in years:
        p = RANK_DIR / str(y) / "rank.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p, columns=["race_id", "horse_id", "rank"]))
    lab = pd.concat(parts, ignore_index=True)
    lab["race_id"] = lab["race_id"].astype(str)
    lab["horse_id"] = lab["horse_id"].astype(str)
    return lab


def _ll(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_delta(base: pd.DataFrame, new: pd.DataFrame, lab: pd.DataFrame, rng: np.random.Generator,
                 n_boot: int = N_BOOT) -> dict | None:
    """base と new の OOF を (race_id, horse_id) で結合し、勝・連対・複勝の LogLoss 差（new − base、負が改善）と
    レース単位ブートストラップの片側 95% 上限・90% 上限を返す。"""
    cols = ["p_win", "p_top2", "p_top3"]
    m = base[KEY + cols].merge(new[KEY + cols], on=KEY, suffixes=("_b", "_n")).merge(lab, on=KEY)
    if m.empty:
        return None
    ys = {"win": (m["rank"] == 1).to_numpy(float), "top2": (m["rank"] <= 2).to_numpy(float), "top3": (m["rank"] <= 3).to_numpy(float)}
    pk = {"win": "p_win", "top2": "p_top2", "top3": "p_top3"}
    races, inv = np.unique(m["race_id"].to_numpy(), return_inverse=True)
    n_r = len(races)
    cnt = np.bincount(inv, minlength=n_r).astype(float)
    out: dict = {"n_rows": int(len(m)), "n_races": int(n_r)}
    idx = rng.integers(0, n_r, size=(n_boot, n_r))
    for k, y in ys.items():
        d = _ll(m[pk[k] + "_n"].to_numpy(float), y) - _ll(m[pk[k] + "_b"].to_numpy(float), y)
        ds = np.bincount(inv, weights=d, minlength=n_r)
        delta = float(ds.sum() / cnt.sum())
        boot = ds[idx].sum(axis=1) / cnt[idx].sum(axis=1)
        out[f"d_{k}"] = delta
        out[f"hi95_{k}"] = float(np.quantile(boot, 0.95))
        out[f"hi90_{k}"] = float(np.quantile(boot, 0.90))
        out[f"se_{k}"] = float(boot.std(ddof=1))
    return out


def oof_pairs() -> list[dict]:
    """(群の OOF, 基準の OOF, 評価の文脈) の組を列挙する。"""
    pairs: list[dict] = []
    # stage ごとの base.parquet と同じ場所の oof_*.parquet
    stage_dirs: list[Path] = []
    for d in STAGE_DIRS:
        stage_dirs += [Path(p) for p in glob.glob(str(V4 / d / "stage*"))]
    stage_dirs += [Path(p) for p in glob.glob(str(WORK / "tp60_out" / "eval" / "*"))]
    stage_dirs += [Path(p) for p in glob.glob(str(WORK / "acq" / "eval"))]
    stage_dirs += [Path(p) for p in glob.glob(str(WORK / "tp38_out" / "stage*"))]
    for sd in sorted(set(stage_dirs)):
        base = sd / "base.parquet"
        if not base.exists():
            continue
        for f in sorted(sd.glob("oof_*.parquet")):
            pairs.append({"group": f.stem[4:], "dir": str(sd), "new": str(f), "base": str(base), "base_name": "stage_base"})
    # v6/eval*: 基準は同じ場所の oof_d_CY_BEST
    for sd in sorted(glob.glob(str(V4 / "v6" / "eval*"))):
        sd = Path(sd)
        base = sd / "oof_d_CY_BEST.parquet"
        if not base.exists():
            continue
        for f in sorted(sd.glob("oof_*.parquet")):
            if f.name == base.name or f.stem.endswith(".v5select"):
                continue
            pairs.append({"group": f.stem[4:], "dir": str(sd), "new": str(f), "base": str(base), "base_name": "d_CY_BEST"})
    # v5_select: 基準は a_V4
    sd = V4 / "v5_select"
    base = sd / "oof_a_V4.parquet"
    if base.exists():
        for f in sorted(sd.glob("oof_*.parquet")):
            if f.name == base.name:
                continue
            pairs.append({"group": f.stem[4:], "dir": str(sd), "new": str(f), "base": str(base), "base_name": "a_V4"})
    return pairs


def oof_year_deltas(years: tuple[int, ...] = (2022, 2023, 2024)) -> list[dict]:
    rng = np.random.default_rng(SEED)
    lab = load_labels(list(years))
    rows: list[dict] = []
    for pr in oof_pairs():
        try:
            base = pd.read_parquet(pr["base"])
            new = pd.read_parquet(pr["new"])
        except (OSError, ValueError):
            continue
        for df in (base, new):
            df["race_id"] = df["race_id"].astype(str)
            df["horse_id"] = df["horse_id"].astype(str)
        for y in years:
            b, n = base[base["year"] == y], new[new["year"] == y]
            if b.empty or n.empty:
                continue
            res = paired_delta(b, n, lab, rng)
            if res:
                rows.append({**{k: pr[k] for k in ("group", "dir", "base_name")}, "year": y, **res})
    return rows


def check_sets(cfg_path: str, frame: str | None = None) -> dict:
    """復活セットの群の parquet を読み取り専用で点検する: 行キー・行数・列名の重なり（基準の V4 フレーム／群どうし）。"""
    import pyarrow.parquet as pq

    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    base_cols: set[str] = set()
    if frame and Path(frame).exists():
        base_cols = set(pq.ParquetFile(frame).schema_arrow.names)
    seen: dict[str, str] = {}
    out: dict = {"groups": {}, "collisions": []}
    for name, g in cfg["groups"].items():
        path = g.get("path") or ""
        if not path:
            continue
        p = Path(path) if path.startswith("/") else MODELING.parent.parent.parent / path
        if not p.exists():
            out["groups"][name] = {"exists": False, "path": str(p)}
            continue
        f = pq.ParquetFile(p)
        cols = [c for c in f.schema_arrow.names if c not in KEY]
        if g.get("cols"):   # 1 つの parquet に複数の群の列がある場合（tp-76）は、群の列だけを調べる
            cols = [c for c in cols if c in g["cols"]]
        coll = sorted(set(cols) & base_cols)
        for c in cols:
            if c in seen and seen[c] != name:
                out["collisions"].append({"column": c, "groups": [seen[c], name]})
            seen[c] = name
        out["groups"][name] = {"exists": True, "rows": f.metadata.num_rows, "has_keys": all(k in f.schema_arrow.names for k in KEY),
                               "n_cols": len(cols), "collide_with_base": coll[:6], "n_collide_with_base": len(coll)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-oof", action="store_true")
    ap.add_argument("--check-sets", default="", help="config/feature_sets.json のパス。群の parquet を点検して --out に書く")
    ap.add_argument("--frame", default="", help="基準の V4 フレーム parquet（列名の重なりを調べる）")
    args = ap.parse_args()
    if args.check_sets:
        res = check_sets(args.check_sets, args.frame or None)
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"groups {len(res['groups'])}, collisions {len(res['collisions'])}")
        return
    out = {"pooled": extract_pooled(), "oof_years": [] if args.skip_oof else oof_year_deltas()}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"pooled {len(out['pooled'])} 件, oof_years {len(out['oof_years'])} 件")


if __name__ == "__main__":
    main()
