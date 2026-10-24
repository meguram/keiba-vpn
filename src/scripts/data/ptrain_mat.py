"""母方（母系）の完全な 10 世代血統表から、血統予想家の入力を作る（学習なし。成績は使わない＝リークの余地がない構造だけの特徴量）。

python -m src.scripts.data.ptrain_mat scan    # 10 世代 JSON を走査し、母方の祖先を (馬, 世代, 位置, 祖先ID) の縦長表に落とす（シャード単位）
python -m src.scripts.data.ptrain_mat build   # 残存数（その祖先を母方に持つ馬の数）で高頻度の祖先を選び、血量・クロス・クラスタ・SVD を馬ごとに作る

位置の読み方（build_horse_pedigree_10gen の仕様）: 世代 g の位置 p は g ビット。ビット (g-1) が 0 なら父方、1 なら母方。
母方 = 世代 1 で p=1（母）、世代 g>=2 で ((p >> (g-1)) & 1) == 1。母方の位置数は 世代 g で 2^(g-1)、1〜10 世代の合計 1,023。
出力: PTRAIN_OUT/mat/{long_*.parquet, freq.parquet, features.parquet, meta.json}
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = Path(os.environ.get("PTRAIN_PED10", ROOT / "data/local/horse_pedigree_10gen"))
OUT = Path(os.environ.get("PTRAIN_OUT", ROOT / "data/local/modeling/ptrain")) / "mat"
TOP_N = int(os.environ.get("PTRAIN_MAT_TOPN", 300))
SHARE_MIN = float(os.environ.get("PTRAIN_MAT_SHARE_MIN", 0.03))      # 残存数の下限（母集団の馬のうち、母方にその祖先を持つ割合）
SHARE_MAX = float(os.environ.get("PTRAIN_MAT_SHARE_MAX", 0.60))      # 上限（ほぼ全馬が持つ創始者級の祖先は区別に役立たない）
K_CLUSTERS = (16, 40)
N_SVD = 8
SEED = 77


def is_maternal(g: int, p: int) -> bool:
    return ((p >> (g - 1)) & 1) == 1


def _scan_prefix(prefix_dir: str) -> tuple[str, int]:
    """1 プレフィックスのディレクトリを走査し、母方の祖先を long_<prefix>.parquet に書く。"""
    d = Path(prefix_dir)
    rows_h, rows_g, rows_p, rows_a = [], [], [], []
    n = 0
    for f in d.glob("*.json"):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        hid = j.get("horse_id") or f.stem
        for a in j.get("ancestors") or []:
            g, p, aid = a.get("generation"), a.get("position"), a.get("horse_id")
            if not aid or g is None or p is None or not is_maternal(int(g), int(p)):
                continue
            rows_h.append(hid)
            rows_g.append(int(g))
            rows_p.append(int(p))
            rows_a.append(aid)
        n += 1
    out = OUT / f"long_{d.name}.parquet"
    pd.DataFrame({"horse_id": rows_h, "g": np.array(rows_g, dtype="int8"), "pos": np.array(rows_p, dtype="int16"), "anc": rows_a}).to_parquet(out, index=False)
    return d.name, n


def cmd_scan(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dirs = sorted(str(p) for p in SRC.iterdir() if p.is_dir())
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=int(a.workers)) as ex:
        for name, n in ex.map(_scan_prefix, dirs):
            done += n
            if done % 5000 < n:
                print(f"[{time.time() - t0:.0f}s] {done} horses", flush=True)
    print("scan done", done, f"{time.time() - t0:.0f}s")


def _load_long(restrict: set[str] | None = None) -> pd.DataFrame:
    parts = []
    for f in sorted(OUT.glob("long_*.parquet")):
        d = pd.read_parquet(f)
        if restrict is not None:
            d = d[d["horse_id"].isin(restrict)]
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def cmd_build(a) -> None:
    from scipy import sparse
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD

    t0 = time.time()
    L = _load_long()
    L["horse_id"] = L["horse_id"].astype("category")
    L["anc"] = L["anc"].astype("category")
    hs = L["horse_id"].cat.categories
    n_h = len(hs)
    hi = L["horse_id"].cat.codes.to_numpy()
    ai = L["anc"].cat.codes.to_numpy()
    print(f"long rows {len(L):,} horses {n_h:,} ancestors {len(L['anc'].cat.categories):,}", flush=True)
    # 残存数: その祖先を母方に持つ馬の数（馬×祖先の重複を除く）
    pairs = pd.DataFrame({"h": hi, "a": ai}).drop_duplicates()
    prev = pairs.groupby("a").size()
    freq = pd.DataFrame({"anc": L["anc"].cat.categories[prev.index], "n_horses": prev.to_numpy()})
    freq["share"] = freq["n_horses"] / n_h
    freq = freq.sort_values("n_horses", ascending=False).reset_index(drop=True)
    freq.to_parquet(OUT / "freq.parquet", index=False)
    band = freq[(freq["share"] >= SHARE_MIN) & (freq["share"] <= SHARE_MAX)]
    top = band.head(TOP_N)["anc"].tolist()          # 残存数の多い順（ただし創始者級を除く）
    pd.DataFrame({"anc": top, "rank": range(len(top))}).merge(freq, on="anc").to_parquet(OUT / "top_ancestors.parquet", index=False)
    top_idx = {x: i for i, x in enumerate(top)}
    cat_to_top = np.full(len(L["anc"].cat.categories), -1, dtype=np.int32)
    for i, x in enumerate(L["anc"].cat.categories):
        j = top_idx.get(x)
        if j is not None:
            cat_to_top[i] = j
    col = cat_to_top[ai]
    m = col >= 0
    g = L["g"].to_numpy()
    share = np.where(m, 2.0 ** (-g.astype(float)), 0.0)          # 血量（世代 g の位置は 2^-g）
    S = sparse.csr_matrix((share[m], (hi[m], col[m])), shape=(n_h, TOP_N))     # 馬×高頻度祖先の血量の合計（クロスは足し合わせ）
    C = sparse.csr_matrix((np.ones(m.sum()), (hi[m], col[m])), shape=(n_h, TOP_N))   # 登場回数
    gmin = np.full((n_h, TOP_N), 99, dtype=np.int16)
    np.minimum.at(gmin, (hi[m], col[m]), g[m].astype(np.int16))
    # 完全性: 母方の位置のうち祖先IDが埋まっている数（最大 1,023）
    filled = np.bincount(hi, minlength=n_h)
    per_gen = {gg: np.bincount(hi[g == gg], minlength=n_h) for gg in range(1, 11)}
    feats = pd.DataFrame({"horse_id": hs.astype(str)})
    feats["mat_filled"] = filled
    feats["mat_fill_frac"] = filled / 1023.0
    for gg in (6, 7, 8, 9, 10):
        feats[f"mat_fill_g{gg}"] = per_gen[gg] / (2 ** (gg - 1))
    feats["mat_n_top"] = np.asarray((C > 0).sum(axis=1)).ravel()
    feats["mat_cross_n"] = np.asarray((C > 1).sum(axis=1)).ravel()                  # 母方に 2 回以上登場する高頻度祖先の数（クロス）
    feats["mat_share_top"] = np.asarray(S.sum(axis=1)).ravel()
    # SVD 座標とクラスタ（血量の行列。成績は使わない）
    X = S.toarray().astype("float32")
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)          # 列ごとに標準化（頻度の高い祖先に引きずられない）
    svd = TruncatedSVD(n_components=N_SVD, random_state=SEED)
    Z = svd.fit_transform(Xs)
    for k in range(N_SVD):
        feats[f"mat_svd{k}"] = Z[:, k].astype("float32")
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    for K in K_CLUSTERS:
        feats[f"mat_cl{K}"] = KMeans(n_clusters=K, n_init=4, random_state=SEED).fit_predict(Zn).astype("int16")
    # 高頻度祖先上位 40 の血量（解釈しやすい列）
    for i in range(40):
        feats[f"mat_sh_{i}"] = X[:, i]
    feats["mat_gmin_top1"] = gmin[np.arange(n_h), np.argmax(X, axis=1)]
    # 母父（世代 2・母方・p=2*? ）: 世代 2 の母方位置は p=2,3 のうち ((p>>1)&1)==1 → p in {2,3}。母父は 世代2 のうち 父側(下位ビット0)= p=2
    dam = L[(L["g"] == 2) & (L["pos"] == 2)][["horse_id", "anc"]].astype({"horse_id": str, "anc": str}).rename(columns={"anc": "mat_dam_sire"})
    feats = feats.merge(dam.drop_duplicates("horse_id"), on="horse_id", how="left")
    # 母方クラスタ（K=16）に属する馬の数が少ないクラスタは「その他」にまとめる必要はない（カテゴリとして使う）
    feats.to_parquet(OUT / "features.parquet", index=False)
    meta = {"n_horses": int(n_h), "top_n": TOP_N, "svd_explained": [float(x) for x in svd.explained_variance_ratio_], "top20": band.head(20).to_dict("records"), "n_band": int(len(band)), "share_band": [SHARE_MIN, SHARE_MAX],
            "seconds": round(time.time() - t0, 1)}
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("build done", meta["seconds"], "s; top1", freq.iloc[0].to_dict())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--workers", default="4")
    s.set_defaults(fn=cmd_scan)
    b = sub.add_parser("build")
    b.set_defaults(fn=cmd_build)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
