"""めぐ分析「基本 5 軸」の地図の評価（読むだけ・JSON を書く。ホストのローカルだけ。GCS は読まない）。

A. 血統側の距離の得点分布: 父・母父の産駒の平均（縮小あり）が、デビューが 2025 年以降の馬の実際の分布（点数の割合）をどれだけ当てるか。
   学習 = 2025-01-01 より前の走。検証 = 初出走が 2025 年で、2025 年中に点の付いた走が 3 以上の馬（血統は学習側の産駒だけで作るので漏れない）。
   比べるもの: 全体の分布（基準）/ 父だけ / 父+母父（重み 1・0.5）× 縮小の強さ κ / tp-85 の dist_good（階層ベイズ。モデルが 2025 年を学習に含む可能性があり有利な側）。
B. 地図の置き方: 2025-07〜12 の平地レースから無作為に N レースを取り、5 軸（TS・持続力・ギア・距離適性（適）・血統のダート割合）の z で
   PCA / MDS / t-SNE / Isomap / 選択馬中心の放射配置（PCA・MDS の向き）を、「5 軸の距離の順位相関」と「5 軸で近い 5 頭の再現率」で比べる（レース単位。区間はレースの引き直し）。

使い方（ホストで）: cd <repo> && nice -n 19 /opt/conda/bin/python -I src/scripts/data/basic5_map_eval.py OUT.json [--races 300] [--skip-a] [--skip-b]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path.cwd()
sys.path.insert(0, str(REPO))

from src.research.race import good_distance as gd  # noqa: E402
from src.research.race import running_map as rm  # noqa: E402
from src.research.race import running_map_basic5 as b5  # noqa: E402

SPLIT = pd.Timestamp("2025-01-01")
B_FROM, B_TO = pd.Timestamp("2025-07-01"), pd.Timestamp("2025-12-31")
BOOT = 300


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ci(v: np.ndarray, rng: np.random.Generator, n: int = BOOT) -> list[float]:
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return [float("nan")] * 3
    m = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return [round(float(v.mean()), 4), round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


# ── A. 血統側の距離分布 ────────────────────────────────────────────────────

def eval_a(rng: np.random.Generator) -> dict:
    from src.pipeline.features.pedigree_keys import build_parent_table

    runs = rm._gd_runs()
    runs = runs.assign(horse_id=runs["horse_id"].astype(str), surface=runs["surface"].astype(str))
    runs = runs[runs["points"].notna()]
    train = runs[runs["date"] < SPLIT]
    y25 = runs[(runs["date"] >= SPLIT) & (runs["date"] <= pd.Timestamp("2025-12-31"))]
    first = runs.groupby("horse_id")["date"].min()
    debut25 = first[(first >= SPLIT) & (first <= pd.Timestamp("2025-12-31"))].index
    own = y25[y25["horse_id"].isin(debut25)]
    n_own = own.groupby("horse_id").size()
    test_ids = list(n_own[n_own >= 3].index)
    log("学習の走", len(train), "検証の馬", len(test_ids))

    pt = gd.profile_table(train, by_surface=False)
    ids_tr, P = b5.horse_points(pt)
    tab = build_parent_table(set(ids_tr) | set(test_ids)).set_index("horse_id")
    sire_tr = tab["sire_id"].reindex(ids_tr).fillna("").to_numpy(str)
    dam_tr = tab["dam_sire_id"].reindex(ids_tr).fillna("").to_numpy(str)
    log("親表", len(tab), "父あり", int((sire_tr != "").sum()))

    po = gd.profile_table(own[own["horse_id"].isin(test_ids)], by_surface=False)
    ids_te, Pte = b5.horse_points(po)
    own_rel = Pte / Pte.sum(axis=1, keepdims=True)
    centers = b5.BINS.astype(float)
    own_best = np.array([gd.best_distance(r, b5.BINS) for r in own_rel], float)
    own_mean = own_rel @ centers
    sire_te = tab["sire_id"].reindex(ids_te).fillna("").to_numpy(str)
    dam_te = tab["dam_sire_id"].reindex(ids_te).fillna("").to_numpy(str)

    out: dict = {"n_train_horses": len(ids_tr), "n_test_horses": len(ids_te)}
    glob = P.sum(axis=0) / P.sum()
    cands: dict[str, np.ndarray] = {"全体の分布（基準）": np.tile(glob, (len(ids_te), 1))}
    for kappa in (3.0, 10.0, 30.0):
        ts_, g = b5.parent_tables(P, sire_tr, kappa)
        td_, _ = b5.parent_tables(P, dam_tr, kappa)
        for name, w in ((f"父のみ κ={kappa:g}", (1.0, 0.0)), (f"父+母父(1:0.5) κ={kappa:g}", (1.0, 0.5))):
            H = np.array([b5.ped_hist(ts_.get(s), td_.get(d) if w[1] > 0 else None, g, w) for s, d in zip(sire_te, dam_te)])
            cands[name] = H
    base = cands["全体の分布（基準）"]
    ce_base = -(own_rel * np.log(base + 1e-6)).sum(axis=1)
    res = {}
    for name, H in cands.items():
        ce = -(own_rel * np.log(H + 1e-6)).sum(axis=1)
        pred_best = np.array([gd.best_distance(h, b5.BINS) for h in H], float)
        pred_mean = H @ centers
        from scipy.stats import spearmanr

        res[name] = {
            "クロスエントロピー": ci(ce, rng), "基準との差(負=良い)": ci(ce - ce_base, rng),
            "適の誤差m(平均)": ci(np.abs(pred_best - own_best), rng), "適が±200m以内": ci((np.abs(pred_best - own_best) <= 200).astype(float), rng),
            "平均距離の順位相関": round(float(spearmanr(pred_mean, own_mean)[0]), 4) if np.ptp(pred_mean) > 0 else None,
        }
    out["血統の距離分布"] = res

    # tp-85 の dist_good（同じ馬の無作為 3000 頭）
    from src.research.race import running_map_ped as rmp

    sub = rng.choice(len(ids_te), min(3000, len(ids_te)), replace=False)
    ents = [{"horse_id": ids_te[i], "sex_age": "牡3"} for i in sub]   # 性別は不明のため全馬を牡として渡す（牝は tp-85 に不利）
    pf = rmp.pedigree_for(ents)
    got = [i for i in sub if pf and ids_te[i] in pf["horses"]]
    if got:
        from scipy.stats import spearmanr

        dg = np.array([pf["horses"][ids_te[i]]["v"]["dist_good"][0] for i in got])
        best_name = "父+母父(1:0.5) κ=10"
        out["tp85比較"] = {"n": len(got), "tp85 dist_good vs 実際の平均距離(順位相関)": round(float(spearmanr(dg, own_mean[got])[0]), 4),
                          "父+母父 κ=10 の平均距離 vs 実際の平均距離(同じ馬)": round(float(spearmanr(cands[best_name][got] @ centers, own_mean[got])[0]), 4),
                          "注": "tp-85 のモデルが 2025 年の走を学習に含むなら tp-85 に有利"}
    return out


# ── B. 地図の置き方 ────────────────────────────────────────────────────────

def sample_races(n: int, seed: int) -> pd.DataFrame:
    cols = ["race_id", "date", "horse_id", "sex_age", "surface"]
    d = pd.read_parquet(rm.rp.FLAT_DIR / "2025" / "race_result_flat.parquet", columns=cols)
    d = d[d["surface"].isin(gd.GROUND)].copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d[(d["date"] >= B_FROM) & (d["date"] <= B_TO) & d["horse_id"].notna()]
    d["horse_id"] = d["horse_id"].astype(str)
    d["race_id"] = d["race_id"].astype(str)
    cnt = d.groupby("race_id")["horse_id"].transform("nunique")
    d = d[cnt >= 8].drop_duplicates(["race_id", "horse_id"])
    rids = np.random.default_rng(seed).permutation(sorted(d["race_id"].unique()))[:n]
    return d[d["race_id"].isin(set(rids))]


def normal_score(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """ref（全体の値）の中での位置を正規分布の z に（rm._to_normal と同じ流儀。SD で割って全軸を同じ幅にする）。"""
    q = np.quantile(ref[np.isfinite(ref)], np.linspace(0, 1, 1001))
    z = rm._to_normal(np.asarray(x, float), q)
    return z / float(np.nanstd(rm._to_normal(ref[np.isfinite(ref)], q)))


def race_z(g: pd.DataFrame, runs: pd.DataFrame, qs: dict, d: dict, ped_cache: dict) -> pd.DataFrame:
    """1 レースの馬ごとの 5 軸（z の前の生値）。ts・sustain・gear は地図と同じ z、dist は（適）m、dirt は血統の割合。"""
    from src.research.race import running_map_ped as rmp

    as_of = str(g["date"].iloc[0].date())
    ids = list(g["horse_id"])
    prof = rm.rp.profiles_as_of(d["metrics"], ids, as_of, d["sires"], cols=rm.MAP_COLS, priors=rm._priors(as_of)).set_index("horse_id")
    row = {}
    for h in ids:
        r = {}
        for k in ("ts", "sustain", "gear"):
            j = rm.AXIS_KEYS.index(k)
            col = rm.AXES[j][1]
            v = float(prof.loc[h, col]) if h in prof.index else np.nan
            r[k] = float(rm._to_normal(np.array([v]), qs[k])[0] / qs["_sd"][j] * rm.SIGNS[j]) if np.isfinite(v) else np.nan
        row[h] = r
    z = pd.DataFrame.from_dict(row, orient="index")
    prof_d = gd.profile_before(runs, ids, as_of, rm.GD_TAU, rm.GD_MIN_RUNS)
    z["dist_raw"] = prof_d["best_m"].reindex(z.index) if len(prof_d) else np.nan
    need = [h for h in ids if h not in ped_cache]
    if need:
        pf = rmp.pedigree_for([{"horse_id": h, "sex_age": s} for h, s in zip(g["horse_id"], g["sex_age"]) if h in need])
        for h in need:
            ped_cache[h] = float(pf["horses"][h]["v"]["dirt_share"][0]) if pf and h in pf["horses"] else np.nan
    z["dirt_raw"] = [ped_cache[h] for h in z.index]
    return z


def eval_b(n_races: int, rng: np.random.Generator) -> dict:
    from sklearn.manifold import TSNE, Isomap

    d, qs = rm._data(), rm._quantiles()
    runs = rm._gd_runs()
    runs = runs.assign(horse_id=runs["horse_id"].astype(str), surface=runs["surface"].astype(str))
    smp = sample_races(n_races, 0)
    log("レース", smp["race_id"].nunique(), "馬", len(smp))
    ped_cache: dict = {}
    frames = {}
    for i, (rid, g) in enumerate(smp.groupby("race_id")):
        frames[rid] = race_z(g, runs, qs, d, ped_cache)
        if i % 50 == 0:
            log("軸の計算", i)
    allz = pd.concat(frames.values())
    ref_d, ref_r = allz["dist_raw"].to_numpy(float), allz["dirt_raw"].to_numpy(float)
    consts = {"dist_mean": float(np.nanmean(ref_d)), "dist_sd": float(np.nanstd(ref_d)), "dirt_mean": float(np.nanmean(ref_r)), "dirt_sd": float(np.nanstd(ref_r)),
              "dist_missing_rate": float(np.isnan(ref_d).mean()), "dirt_missing_rate": float(np.isnan(ref_r).mean())}
    methods = ["PCA", "MDS", "t-SNE", "Isomap", "放射(PCA向き)", "放射(MDS向き)"]
    acc = {m: {"rho": [], "recall": []} for m in methods}
    n_ok = 0
    for rid, z in frames.items():
        Z = np.column_stack([z["ts"], z["sustain"], z["gear"], normal_score(z["dist_raw"].to_numpy(), ref_d), normal_score(z["dirt_raw"].to_numpy(), ref_r)])
        keep = np.isnan(Z).sum(axis=1) <= 1
        Z = Z[keep]
        if len(Z) < 8:
            continue
        D = b5.pair_dist(Z)
        n_ok += 1
        Df = b5.fill_missing(D)
        Df = (Df + Df.T) / 2
        lay = {"PCA": b5.layout_pca(Z), "MDS": b5.layout_mds(Z)}
        lay["t-SNE"] = TSNE(n_components=2, metric="precomputed", init="random", perplexity=min(5.0, (len(Z) - 1) / 3), random_state=0).fit_transform(Df)
        lay["Isomap"] = Isomap(n_neighbors=min(5, len(Z) - 1), n_components=2, metric="precomputed").fit_transform(Df)
        for m, Y in lay.items():
            acc[m]["rho"].append(b5.dist_rho(D, Y))
            acc[m]["recall"].append(b5.knn_recall(D, Y))
        for m, base_name in (("放射(PCA向き)", "PCA"), ("放射(MDS向き)", "MDS")):
            rhos, recs = [], []
            for c in range(len(Z)):
                R = b5.radial(Df[c], lay[base_name], c)
                rhos.append(b5.dist_rho(D, R))
                row = D[c].copy()
                row[c] = np.inf
                row = np.where(np.isnan(row), np.inf, row)
                top5 = set(np.argsort(row, kind="stable")[:5])
                r2 = np.hypot(R[:, 0] - R[c, 0], R[:, 1] - R[c, 1])
                r2[c] = np.inf
                recs.append(len(top5 & set(np.argsort(r2, kind="stable")[:5])) / 5)
            acc[m]["rho"].append(float(np.nanmean([x for x in rhos if x is not None])))
            acc[m]["recall"].append(float(np.mean(recs)))
    res = {}
    for m, v in acc.items():
        res[m] = {"距離の順位相関": ci(np.array([x for x in v["rho"] if x is not None], float), rng),
                  "近い5頭の再現率": ci(np.array([x for x in v["recall"] if x is not None], float), rng)}
    return {"races_used": n_ok, "consts": consts, "layouts": res,
            "注": "放射の再現率は半径が真の距離なので作りの上でほぼ 1（比較の材料にならない。距離の順位相関を見る）。放射は『選んだ馬を中心にしたとき』の値を馬ごとに出して平均。再現率は選んだ馬の近い5頭が、画面でも選んだ馬に近い5頭に入る割合（ランダムだと約 5/(n-1)）"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--races", type=int, default=300)
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    out: dict = {}
    if not a.skip_a:
        out["A"] = eval_a(rng)
        print(json.dumps(out["A"], ensure_ascii=False, indent=1), flush=True)
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    if not a.skip_b:
        out["B"] = eval_b(a.races, rng)
        print(json.dumps(out["B"], ensure_ascii=False, indent=1), flush=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    log("wrote", a.out)


if __name__ == "__main__":
    main()
