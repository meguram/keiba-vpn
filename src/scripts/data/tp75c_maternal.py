"""tp-75 第 2 ラウンドの追加（2c）: 完全な母方 10 世代表（data/local/horse_pedigree_10gen/）から、母系の系統・母方の高頻度種牡馬・母方クラスタの列を作る。

python -m src.scripts.data.tp75c_maternal build [--out DIR] [--limit N]    # 出走馬ごとの列（共有表 maternal_lineage.parquet）
python -m src.scripts.data.tp75c_maternal check                            # 作った表の充足率の確認

構造（血統表）だけから作る。成績・人気・オッズは使わない（リークも市場情報も無い）。
母方 = path_fm が 'M' で始まる位置（1,023 位置）。母方の牡馬 = そのうち path_fm が 'F' で終わる位置。世代 = path_fm の長さ。
- 母父 = 'MF'、母母父 = 'MMF'、母母母父 = 'MMMF'。系統は stallion_lineage（main 5 / coarse 12+その他 / fine 36）で分類。
- 近い母系 = 世代 2〜5 の牡馬、深い母系 = 世代 6〜10 の牡馬。系統ごとの本数と、深い母系での割合（share）。
- 母方の高頻度種牡馬 = 出走馬の母方 10 世代に現れる割合が 5〜60% の種牡馬のうち、割合の高い順に 20 頭（構造だけで決める。割合が 60% 超の祖先は、ほぼ全頭にいて区別にならない）。
  現れる／2 回以上（クロス）／深い世代（6〜10）だけに現れる、の 3 種類。
- 母方クラスタ = 近い・深い母系の系統の割合（コース K 個の標準化したベクトル）の KMeans（K=8, 24。乱数の種は固定）。
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

TEN = Path(os.environ.get("TP75C_10GEN", "/home/jovyan/work/keiba-vpn/data/local/horse_pedigree_10gen"))
LINEAGE_DIR = Path(os.environ.get("TP75B_LINEAGE", "/home/jovyan/work/keiba-vpn/data/page_reference/pedigree_race_index"))
MINING = Path(os.environ.get("TP72_OUT", "/home/jovyan/work/tp72_out")) / "mining.parquet"
OUT = Path(os.environ.get("TP75C_OUT", "/home/jovyan/work/keiba-vpn/data/local/modeling/shared_cache/maternal_lineage"))
HF_LO, HF_HI, N_HF = 0.05, 0.60, 20
KS = (8, 24)
SEED = 20261022
N_COARSE = 12
_L: dict = {}


def lineage_maps() -> dict:
    sl = pd.read_parquet(LINEAGE_DIR / "stallion_lineage.parquet")
    meta = json.loads((LINEAGE_DIR / "stallion_lineage_meta.json").read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for m in meta:
        counts[m["anchor_name"]] = counts.get(m["anchor_name"], 0) + int(m.get("count") or 0)
    top = [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:N_COARSE]]
    sl["sub"] = sl["sub_group_label"].astype(str)
    sl["coarse"] = np.where(sl["sub"].isin(top), sl["sub"], "その他")
    by_id = {str(r.stallion_id): (r.main_group_name, r.coarse, r.sub) for r in sl.itertuples()}
    by_name = {r.stallion_name: (r.main_group_name, r.coarse, r.sub) for r in sl.itertuples()}
    return {"by_id": by_id, "by_name": by_name, "coarse_levels": top + ["その他"]}


def _init(maps: dict) -> None:
    _L.update(maps)


def parse_one(horse_id: str) -> dict | None:
    """1 頭の 10 世代 JSON から、母方の牡馬の祖先（path_fm, 世代, id, 名前）を取り出して要約する。"""
    p = TEN / horse_id[:4] / f"{horse_id}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    anc = d.get("ancestors") or []
    mat_total = 0
    males: list[tuple[str, int, str, str]] = []
    for a in anc:
        pf = a.get("path_fm") or ""
        if pf.startswith("M"):
            mat_total += 1
            if pf.endswith("F"):
                males.append((pf, len(pf), str(a.get("horse_id") or ""), a.get("name") or ""))
    out = {"horse_id": horse_id, "mat_positions": mat_total, "mat_complete": bool(mat_total == 1023)}
    by_id, by_name = _L["by_id"], _L["by_name"]

    def line(i: str, nm: str):
        return by_id.get(i) or by_name.get(nm)

    for key, path in (("mmf", "MMF"), ("mmmf", "MMMF"), ("mff", "MFF")):
        hit = next((m for m in males if m[0] == path), None)
        ln = line(hit[2], hit[3]) if hit else None
        out[f"{key}_name"] = hit[3] if hit else None
        out[f"{key}_main"], out[f"{key}_coarse"], out[f"{key}_fine"] = (ln if ln else (None, None, None))
    near = defaultdict(int); deep = defaultdict(int)
    anc_count: dict[str, list] = {}
    unk_near = unk_deep = 0
    for pf, g, i, nm in males:
        ln = line(i, nm)
        key = ln[1] if ln else None
        if 2 <= g <= 5:
            near[key] += 1
            if key is None:
                unk_near += 1
        elif g >= 6:
            deep[key] += 1
            if key is None:
                unk_deep += 1
        if i:
            rec = anc_count.setdefault(i, [0, 99, nm])
            rec[0] += 1
            rec[1] = min(rec[1], g)
    out["near"] = {k: v for k, v in near.items() if k}
    out["deep"] = {k: v for k, v in deep.items() if k}
    out["n_near"] = sum(near.values()); out["n_deep"] = sum(deep.values())
    out["anc"] = {i: (c, mg) for i, (c, mg, _n) in anc_count.items()}
    out["anc_names"] = {i: n for i, (_c, _mg, n) in anc_count.items()}
    return out


def build(out: Path, limit: int | None = None, workers: int = 3) -> pd.DataFrame:
    maps = lineage_maps()
    ids = sorted(pd.read_parquet(MINING, columns=["horse_id"])["horse_id"].astype(str).unique())
    if limit:
        ids = ids[:limit]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(maps,)) as ex:
        recs = [r for r in ex.map(parse_one, ids, chunksize=200) if r]
    prev = defaultdict(int)
    names: dict[str, str] = {}
    for r in recs:
        for i in r["anc"]:
            prev[i] += 1
        names.update(r["anc_names"])
    n = len(recs)
    cand = [(i, c / n) for i, c in prev.items() if HF_LO <= c / n <= HF_HI]
    hf = [i for i, _ in sorted(cand, key=lambda t: (-t[1], t[0]))[:N_HF]]
    levels = maps["coarse_levels"]
    rows = []
    for r in recs:
        row = {k: r[k] for k in ("horse_id", "mat_positions", "mat_complete", "mmf_name", "mmf_main", "mmf_coarse", "mmf_fine", "mmmf_name", "mmmf_main", "mmmf_coarse", "mmmf_fine", "mff_name", "mff_main", "mff_coarse", "mff_fine", "n_near", "n_deep")}
        for li, lv in enumerate(levels):                    # 列名は系統名ではなく番号（式で使える名前にする。対応は README の coarse_levels）
            row[f"near_L{li:02d}"] = r["near"].get(lv, 0)
            row[f"deep_L{li:02d}"] = r["deep"].get(lv, 0)
            row[f"share_near_L{li:02d}"] = r["near"].get(lv, 0) / r["n_near"] if r["n_near"] else np.nan
            row[f"share_deep_L{li:02d}"] = r["deep"].get(lv, 0) / r["n_deep"] if r["n_deep"] else np.nan
        for k, i in enumerate(hf):
            c, mg = r["anc"].get(i, (0, 99))
            row[f"hf{k:02d}_present"] = int(c >= 1)
            row[f"hf{k:02d}_cross"] = int(c >= 2)
            row[f"hf{k:02d}_deeponly"] = int(c >= 1 and mg >= 6)
        rows.append(row)
    df = pd.DataFrame(rows)
    from sklearn.cluster import KMeans
    cols = [f"share_near_L{li:02d}" for li in range(len(levels))] + [f"share_deep_L{li:02d}" for li in range(len(levels))]
    X = df[cols].fillna(0.0).to_numpy()
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    ok = df["mat_complete"].to_numpy()
    for K in KS:
        km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(X[ok])
        lab = np.full(len(df), -1)
        lab[ok] = km.labels_
        df[f"mat_cl{K}"] = lab
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "maternal_lineage.parquet", index=False)
    readme = {
        "rows": int(len(df)), "key": "horse_id", "source": str(TEN), "complete_rate": float(df["mat_complete"].mean()),
        "note": "構造（血統表）だけから作った列。成績・オッズは使っていない。母方 = path_fm が 'M' で始まる 1,023 位置。完全な母方 10 世代表に基づく。",
        "hf_sires": [{"slot": f"hf{k:02d}", "horse_id": i, "name": names.get(i), "prevalence": float(prev[i] / n)} for k, i in enumerate(hf)],
        "coarse_levels": levels, "kmeans": {"K": list(KS), "seed": SEED, "features": cols, "fit_on": "母方が完全な全出走馬（構造だけ）"},
        "columns": {"mmf_/mmmf_/mff_*": "母母父・母母母父・母父父の系統（name/main/coarse/fine）", "near_*/deep_*": "母方の牡馬（近い = 世代 2〜5、深い = 世代 6〜10）の系統（coarse）ごとの本数",
                    "share_near_*/share_deep_*": "同じ割合", "hfNN_present/cross/deeponly": "母方の高頻度種牡馬が母方にいる／2 回以上／世代 6〜10 にだけいる", "mat_cl8/mat_cl24": "母方の系統構成の KMeans クラスタ"},
    }
    (out / "README.json").write_text(json.dumps(readme, ensure_ascii=False, indent=1), encoding="utf-8")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check"])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if a.cmd == "build":
        df = build(Path(a.out), a.limit, a.workers)
        print(len(df), "rows; complete", float(df["mat_complete"].mean()))
    else:
        df = pd.read_parquet(Path(a.out) / "maternal_lineage.parquet")
        print(len(df), df["mat_complete"].mean(), df[["mmf_main", "mmmf_main"]].notna().mean().to_dict())


if __name__ == "__main__":
    main()
