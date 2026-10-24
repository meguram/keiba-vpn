"""tp-60 構成埋め込みから、近さと適性の関係の実測・クラスタ統計・kNN 型の統計（すべて as-of、ID キー、オッズ非依存）を作る。

python -m src.scripts.data.tp60_embfeat neighbors --variant A_t2_T10_g100     # 競走馬どうしの近傍（コサイン、上位 K）
python -m src.scripts.data.tp60_embfeat affinity --variant ...                # 近さの十分位ごとの適性の相関、全兄弟・同父・同母の一致度（天井）
python -m src.scripts.data.tp60_embfeat groups --variant ...                  # 群 CLU_<v>（クラスタ×条件）, KNN_<v>（近い馬の成績）を groups/ に書く（V5 の学習表の行単位）

適性の定義（分析用）: 馬ごとの着順スコア res（レース内の着順を [-0.5, 0.5]）の、その馬の平均からの偏差を条件別に平均し、
  芝−ダの差（芝・ダ各 3 走以上）、長距離（1800m 以上）−短距離（それ未満）の差（各 3 走以上）。能力そのもの（平均 res）も併記する。
kNN の重みは『近さ別の適性の回帰係数 β(近さ)』（2021 年までの結果から推定、以後固定）を使い、近さが 1 に近いほど強く信じる固定の重みにはしない。
事前（近い馬の成績）は馬自身の過去走で更新する（縮約）。クラスタ定義は 2021 年までにデビューした馬で固定し、それ以降の馬は最近傍の中心に割り当てる。
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.scripts.data import tp60_lib as L

EMB = L.OUTD / "embed"
GROUPS = L.OUTD / "groups"
K_NN = int(os.environ.get("TP60_KNN", 30))
KS = [64, 256]


def _log(msg: str) -> None:
    L.log(msg)


def load_emb(variant: str) -> tuple[np.ndarray, pd.Index]:
    E = np.load(EMB / f"E32_{variant}.npy").astype(np.float32)
    horses = pd.Index(pd.read_parquet(EMB / "horses.parquet")["horse_id"].astype(str))
    return E, horses


def unit(E: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(E, axis=1, keepdims=True)
    return np.where(n > 1e-9, E / np.maximum(n, 1e-9), 0.0).astype(np.float32)


def knn_table(E: np.ndarray, k: int, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """各馬の近傍 k 頭（自分を除く）とコサイン類似度。同点（全兄弟など）はそのまま並ぶ。"""
    U = unit(E)
    n = len(U)
    idx = np.zeros((n, k), dtype=np.int32)
    sim = np.zeros((n, k), dtype=np.float32)
    for lo in range(0, n, chunk):
        S = U[lo:lo + chunk] @ U.T
        r = np.arange(S.shape[0])
        S[r, lo + r] = -2.0
        part = np.argpartition(-S, k, axis=1)[:, :k]
        ps = np.take_along_axis(S, part, 1)
        o = np.argsort(-ps, axis=1)
        idx[lo:lo + chunk] = np.take_along_axis(part, o, 1)
        sim[lo:lo + chunk] = np.take_along_axis(ps, o, 1)
    return idx, sim


def horse_aptitude(m: pd.DataFrame, years: tuple[int, int] | None = None, parity: int | None = None) -> pd.DataFrame:
    """馬ごとの能力（平均 res）、芝−ダ、長−短、距離の傾き（log 距離に対する偏差の傾き）の適性。分析用（全期間または指定年）。parity: 馬ごとの出走の偶奇で半分に分ける（信頼性の推定用）。"""
    d = m[["horse_id", "date", "year", "surface", "distance", "finish_position", "field_size"]].copy()
    if years:
        d = d[d["year"].between(*years)]
    fp = pd.to_numeric(d["finish_position"], errors="coerce")
    n = np.maximum(pd.to_numeric(d["field_size"], errors="coerce") - 1.0, 1.0)
    d["res"] = 0.5 - (fp - 1.0) / n
    d = d[fp > 0].sort_values(["horse_id", "date"])
    if parity is not None:
        d = d[(d.groupby("horse_id").cumcount() % 2) == parity]
    d["dev"] = d["res"] - d.groupby("horse_id")["res"].transform("mean")
    d["turf"] = d["surface"].astype(str) == "芝"
    d["long"] = pd.to_numeric(d["distance"], errors="coerce") >= 1800
    d["ld"] = np.log(pd.to_numeric(d["distance"], errors="coerce"))
    g = d.groupby("horse_id")
    out = pd.DataFrame({"n": g.size(), "ability": g["res"].mean()})
    mn = 3 if parity is None else 2
    for nm, col in (("a_surf", "turf"), ("a_dist", "long")):
        a = d[d[col]].groupby("horse_id")["dev"].agg(["mean", "size"])
        b = d[~d[col]].groupby("horse_id")["dev"].agg(["mean", "size"])
        j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
        j = j[(j["size_a"] >= mn) & (j["size_b"] >= mn)]
        out[nm] = (j["mean_a"] - j["mean_b"]).reindex(out.index)
    ldc = d["ld"] - g["ld"].transform("mean")
    cov = (ldc * d["dev"]).groupby(d["horse_id"]).sum()
    var = (ldc ** 2).groupby(d["horse_id"]).sum()
    nn = g.size()
    ok = (nn >= (5 if parity is None else 3)) & (var > 1e-3)
    out["a_slope"] = (cov / var).where(ok).reindex(out.index)
    return out


def corr_pairs(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 100:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def cmd_neighbors(a) -> None:
    E, horses = load_emb(a.variant)
    idx, sim = knn_table(E, K_NN)
    np.save(EMB / f"nn_idx_{a.variant}.npy", idx)
    np.save(EMB / f"nn_sim_{a.variant}.npy", sim)
    _log(f"{a.variant}: 近傍 {idx.shape}、sim 平均 {sim.mean():.3f}、第 1 近傍が 0.999 以上の馬 {(sim[:, 0] > 0.999).mean():.3f}")


def _sb(r: float) -> float:
    return float(2 * r / (1 + r)) if np.isfinite(r) and r > -1 else np.nan


def cmd_affinity(a) -> None:
    m = L.load_master(jump=False)
    E, horses = load_emb(a.variant)
    U = unit(E)
    hs = pd.read_parquet(L.PED / "horse_static.parquet").drop_duplicates("horse_id").set_index("horse_id").reindex(horses)
    idx = np.load(EMB / f"nn_idx_{a.variant}.npy")
    sim = np.load(EMB / f"nn_sim_{a.variant}.npy")
    cols = ("ability", "a_surf", "a_dist", "a_slope")
    ap_all = horse_aptitude(m).reindex(horses)
    A = {c: ap_all[c].to_numpy(dtype=float) for c in cols}
    n_start = ap_all["n"].fillna(0).to_numpy()
    # 信頼性（偶奇の分割の相関 → Spearman-Brown）
    p0, p1 = horse_aptitude(m, parity=0).reindex(horses), horse_aptitude(m, parity=1).reindex(horses)
    rel = {c: _sb(corr_pairs(p0[c].to_numpy(dtype=float), p1[c].to_numpy(dtype=float))) for c in cols}
    out: dict = {"variant": a.variant, "reliability": rel, "deciles_random": {}, "neighbor_bands": {}, "pairs": {}}
    ok_h = n_start >= 6
    rng = np.random.default_rng(5)
    cand = np.flatnonzero(ok_h)
    r1, r2 = rng.choice(cand, 3000000), rng.choice(cand, 3000000)
    keep = r1 != r2
    r1, r2 = r1[keep], r2[keep]
    s_rand = np.einsum("ij,ij->i", U[r1], U[r2])
    qs = np.quantile(s_rand, np.linspace(0, 1, 11))
    qs[0] -= 1e-6
    qs[-1] += 1e-6
    bk = np.clip(np.searchsorted(qs, s_rand, side="right") - 1, 0, 9)
    for b in range(10):
        mk = bk == b
        e = {"sim_lo": float(qs[b]), "sim_hi": float(qs[b + 1]), "pairs": int(mk.sum())}
        for c in cols:
            e[f"corr_{c}"] = corr_pairs(A[c][r1[mk]], A[c][r2[mk]])
        out["deciles_random"][str(b + 1)] = e
    ii = np.repeat(np.arange(len(horses)), idx.shape[1])
    jj = idx.reshape(-1)
    ss = sim.reshape(-1)
    kp = ok_h[ii] & ok_h[jj]
    ii, jj, ss = ii[kp], jj[kp], ss[kp]
    for lab, lo, hi in (("<0.85", -1, 0.85), ("0.85-0.90", 0.85, 0.90), ("0.90-0.95", 0.90, 0.95), ("0.95-0.99", 0.95, 0.99), ("0.99-0.999", 0.99, 0.999), (">=0.999", 0.999, 2.0)):
        mk = (ss >= lo) & (ss < hi)
        out["neighbor_bands"][lab] = {"pairs": int(mk.sum()), **{f"corr_{c}": corr_pairs(A[c][ii[mk]], A[c][jj[mk]]) for c in cols}}
    out["pairs"]["random"] = {"pairs": int(len(r1)), **{f"corr_{c}": corr_pairs(A[c][r1], A[c][r2]) for c in cols}}

    def pair_set(groups, name, maxn=60000):
        pr = []
        for lst in groups:
            lst = [x for x in lst if ok_h[x]]
            for k_ in range(len(lst) - 1):
                pr.append((lst[k_], lst[k_ + 1]))
        if not pr:
            return
        pr = np.array(pr)[:maxn]
        out["pairs"][name] = {"pairs": int(len(pr)), **{f"corr_{c}": corr_pairs(A[c][pr[:, 0]], A[c][pr[:, 1]]) for c in cols}}
    hs2 = hs.reset_index().assign(code=np.arange(len(horses))).dropna(subset=["sire_id", "dam_id"])
    pair_set(hs2.groupby(["sire_id", "dam_id"])["code"].apply(list).tolist(), "full_siblings(同父同母)")
    pair_set(hs2.groupby("sire_id")["code"].apply(lambda s_: list(s_.sample(min(len(s_), 60), random_state=1))).tolist(), "同父(母は別)")
    pair_set(hs2.groupby("dam_id")["code"].apply(list).tolist(), "同母(父は別も含む)")
    # 補正: 真の相関 ≈ 観測 / sqrt(信頼性_i × 信頼性_j) = 観測 / 信頼性（同じ指標どうし）
    for nm, e in out["pairs"].items():
        e["disattenuated"] = {c: (e[f"corr_{c}"] / rel[c] if rel[c] and np.isfinite(rel[c]) and rel[c] > 0.05 and np.isfinite(e[f"corr_{c}"]) else None) for c in cols}
    _save_json(f"affinity_{a.variant}.json", out)
    _log(f"affinity {a.variant}: 信頼性 {rel}、全兄弟 {out['pairs'].get('full_siblings(同父同母)')}、無作為 {out['pairs']['random']}")


def _save_json(name: str, obj) -> None:
    (L.RES).mkdir(parents=True, exist_ok=True)
    (L.RES / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)), encoding="utf-8")


def estimate_beta_curve(variant: str, n_bins: int = 10) -> dict:
    """2021 年までの結果だけで、近さ（sim）の十分位ごとの適性の回帰係数 β（近い馬の偏差 → 自分の偏差）を推定する。"""
    m = L.load_master()
    E, horses = load_emb(variant)
    idx = np.load(EMB / f"nn_idx_{variant}.npy")
    sim = np.load(EMB / f"nn_sim_{variant}.npy")
    ap = horse_aptitude(m, years=(2020, 2021)).reindex(horses)
    ab = ap["ability"].to_numpy(dtype=float)
    ns = ap["n"].fillna(0).to_numpy()
    ok = ns >= 4
    ii = np.repeat(np.arange(len(horses)), idx.shape[1])
    jj = idx.reshape(-1)
    ss = sim.reshape(-1)
    keep = ok[ii] & ok[jj]
    ii, jj, ss = ii[keep], jj[keep], ss[keep]
    qs = np.quantile(ss, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = qs[0] - 1e-6, qs[-1] + 1e-6
    bucket = np.clip(np.searchsorted(qs, ss, side="right") - 1, 0, n_bins - 1)
    mu = np.nanmean(ab[ok])
    x, y = ab[jj] - mu, ab[ii] - mu
    beta = []
    for b in range(n_bins):
        mk = bucket == b
        den = (x[mk] ** 2).sum()
        beta.append(float(max((x[mk] * y[mk]).sum() / den, 0.0)) if den > 0 else 0.0)
    return {"edges": qs.tolist(), "beta": beta, "mu": float(mu)}


def cmd_curve(a) -> None:
    c = estimate_beta_curve(a.variant)
    _save_json(f"beta_curve_{a.variant}.json", c)
    _log(f"β 曲線: {np.round(c['beta'], 3)}")


def _asof_sn(date: np.ndarray, key: np.ndarray, v: np.ndarray):
    import polars as pl

    base = pl.DataFrame({"_i": np.arange(len(date)), "date": date, "k": key, "v": v})
    vv = pl.col("v").cast(pl.Float64)
    daily = (base.group_by(["k", "date"]).agg(vv.sum().alias("s"), vv.is_not_null().sum().alias("n")).sort("date")
             .with_columns(pl.col("s").cum_sum().over("k").alias("S"), pl.col("n").cum_sum().over("k").alias("N")).select("k", "date", "S", "N").sort("date"))
    return base, daily


def build_groups(variant: str) -> None:
    import polars as pl

    from src.pipeline.features import factory_ped as fp
    from src.pipeline.features import factory_src as fs
    from src.scripts.data import tp27_eval as t

    df, base, meta, folds, lp = t.setup()
    if os.environ.get("TP60_SMOKE"):
        df = df.sample(40000, random_state=0).sort_values(["date", "race_id", "horse_number"]).reset_index(drop=True)
    E, horses = load_emb(variant)
    U = unit(E)
    hix = horses.get_indexer(df["horse_id"].astype(str))
    has = (hix >= 0) & (np.linalg.norm(E[np.where(hix >= 0, hix, 0)], axis=1) > 1e-9)
    res = fs.res_score(df)
    fpv = pd.to_numeric(df["finish_position"], errors="coerce").to_numpy(dtype=float)
    win = np.where(fpv > 0, (fpv == 1).astype(float), np.nan)
    surf = df["surface"].astype(str)
    sd = (surf + "|" + fp._dist_band(df)).to_numpy()
    allk = np.array(["all"] * len(df))
    date = pd.to_datetime(df["date"]).to_numpy().astype("datetime64[D]")
    out = pd.DataFrame({"race_id": df["race_id"].astype(str), "horse_id": df["horse_id"].astype(str)})
    tag = variant.split("_")[0]
    # ---- クラスタ × 条件（定義は 2021 年までにデビューした馬で固定）
    from sklearn.cluster import MiniBatchKMeans

    first_year = df.groupby("horse_id")["year"].min()
    fit_h = np.array([first_year.get(h, 9999) <= 2021 for h in horses])
    fit_h &= np.linalg.norm(E, axis=1) > 1e-9
    for K in KS:
        km = MiniBatchKMeans(n_clusters=K, random_state=0, n_init=3, batch_size=4096).fit(U[fit_h])
        lab = np.full(len(horses), -1)
        okh = np.linalg.norm(E, axis=1) > 1e-9
        lab[okh] = km.predict(U[okh])
        cl = np.where(has, lab[np.where(hix >= 0, hix, 0)], -1)
        for cn, ck in (("all", allk), ("sd", sd)):
            key = pd.Series(np.where(cl >= 0, [f"{c}|{k}" for c, k in zip(cl, ck)], None), index=df.index)
            ck_key = pd.Series(ck, index=df.index)
            for vn, v in (("win", win), ("res", res)):
                S, N = fp.asof_sums(df, key, v)
                Sc, Nc = fp.asof_sums(df, ck_key, v)
                So, No = fp.asof_sums(df, (df["horse_id"].astype(str) + "|" + pd.Series(ck, index=df.index)), v)   # 馬自身の過去走（同条件）
                S, N = S - So, N - No
                mu = np.where(Nc > 0, Sc / np.maximum(Nc, 1), np.nan)
                val = (S + 40.0 * mu) / (N + 40.0)
                out[f"clu60_{tag}_K{K}_{cn}_{vn}"] = np.where(cl >= 0, val - mu, np.nan).astype("float32")
            out[f"clu60_{tag}_K{K}_{cn}_n"] = np.where(cl >= 0, N, np.nan).astype("float32")
        _log(f"クラスタ K={K} 完了")
    # ---- kNN: 近い馬の as-of 成績（近さ別の β で重み付け）
    idx = np.load(EMB / f"nn_idx_{variant}.npy")
    sim = np.load(EMB / f"nn_sim_{variant}.npy")
    curve = estimate_beta_curve(variant)
    edges, beta = np.array(curve["edges"]), np.array(curve["beta"])
    nb = np.clip(np.searchsorted(edges, sim, side="right") - 1, 0, len(beta) - 1)
    Wn = beta[nb].astype(np.float32)                                            # 馬 × K の重み
    rows = np.flatnonzero(has)
    horse_of = df["horse_id"].astype(str).to_numpy()
    for cn, ck in (("all", allk), ("sd", sd)):
        # 近傍馬の（馬, 条件）別の累積成績
        sub = pl.DataFrame({"h": horse_of, "c": ck, "date": date, "v": res}, nan_to_null=True)
        vv = pl.col("v").cast(pl.Float64)
        daily = (sub.group_by(["h", "c", "date"]).agg(vv.sum().alias("s"), vv.is_not_null().sum().alias("n")).sort("date")
                 .with_columns(pl.col("s").cum_sum().over(["h", "c"]).alias("S"), pl.col("n").cum_sum().over(["h", "c"]).alias("N")).select("h", "c", "date", "S", "N").sort("date"))
        Sg, Ng = fp.asof_sums(df, pd.Series(ck, index=df.index), res)
        mu_c = np.where(Ng > 0, Sg / np.maximum(Ng, 1), 0.0)
        acc_num = np.zeros(len(df))
        acc_w = np.zeros(len(df))
        sim_mean = np.full(len(df), np.nan)
        sim_top = np.full(len(df), np.nan)
        step = 60000
        for lo in range(0, len(rows), step):
            r = rows[lo:lo + step]
            hi_ = hix[r]
            nbr = idx[hi_]                                                      # (len, K) 近傍のコード
            P = pl.DataFrame({"_i": np.repeat(r, nbr.shape[1]), "h": horses.to_numpy()[nbr.reshape(-1)], "c": np.repeat(ck[r], nbr.shape[1]), "date": np.repeat(date[r], nbr.shape[1]),
                              "w": Wn[hi_].reshape(-1)}, nan_to_null=True).sort("date")
            J = P.join_asof(daily, on="date", by=["h", "c"], strategy="backward", allow_exact_matches=False).with_columns(pl.col("S").fill_null(0.0), pl.col("N").fill_null(0))
            Jd = J.to_pandas()
            dev = np.where(Jd["N"] > 0, Jd["S"] / np.maximum(Jd["N"], 1) - mu_c[Jd["_i"].to_numpy()], 0.0)
            rel = Jd["N"].to_numpy() / (Jd["N"].to_numpy() + 4.0)
            w = Jd["w"].to_numpy() * rel
            num = pd.Series(w * dev).groupby(Jd["_i"].to_numpy()).sum()
            den = pd.Series(w).groupby(Jd["_i"].to_numpy()).sum()
            acc_num[num.index.to_numpy()] = num.to_numpy()
            acc_w[den.index.to_numpy()] = den.to_numpy()
            sim_mean[r] = sim[hi_].mean(1)
            sim_top[r] = sim[hi_, 0]
        prior = np.where(has, acc_num / (acc_w + 0.5), np.nan)                  # 近い馬の成績の加重平均（偏差）、重みが小さいほど 0 へ縮約
        out[f"knn60_{tag}_{cn}_prior"] = prior.astype("float32")
        out[f"knn60_{tag}_{cn}_wsum"] = np.where(has, acc_w, np.nan).astype("float32")
        # 事前を馬自身の同条件の過去走で更新（縮約）
        So, No = fp.asof_sums(df, (df["horse_id"].astype(str) + "|" + pd.Series(ck, index=df.index)), res)
        post = (So + 6.0 * (mu_c + np.nan_to_num(prior))) / (No + 6.0) - mu_c
        out[f"knn60_{tag}_{cn}_post"] = np.where(has, post, np.nan).astype("float32")
        _log(f"kNN {cn} 完了")
    out[f"knn60_{tag}_sim_mean"] = sim_mean.astype("float32")
    out[f"knn60_{tag}_sim_top"] = sim_top.astype("float32")
    GROUPS.mkdir(parents=True, exist_ok=True)
    if os.environ.get("TP60_SMOKE"):
        _log(f"smoke 完了: {out.shape}, 欠損率 {out.isna().mean().round(3).to_dict()}")
        return
    clu_cols = [c for c in out.columns if c.startswith("clu60_")]
    knn_cols = [c for c in out.columns if c.startswith("knn60_")]
    out[["race_id", "horse_id"] + clu_cols].to_parquet(GROUPS / f"CLU_{tag}.parquet", index=False)
    out[["race_id", "horse_id"] + knn_cols].to_parquet(GROUPS / f"KNN_{tag}.parquet", index=False)
    _save_json(f"beta_curve_{variant}.json", curve)
    _log(f"書き出し CLU_{tag} {len(clu_cols)} 列、KNN_{tag} {len(knn_cols)} 列")


def cmd_groups(a) -> None:
    build_groups(a.variant)


def cmd_sirepairs(a) -> None:
    """構成が同一（全兄弟）の種牡馬どうしで、産駒の成績がどれだけ違うか（種牡馬固有の差＝固有分の残差）。"""
    from src.scripts.data import tp60_embed as te

    m = L.load_master()
    m = m[m["year"] <= 2025]
    G, code, _, _ = te.load_graph()
    Ef = np.load(EMB / f"Efull_{a.variant}.npy").astype(np.float32)
    cnt = m.groupby("sire_id").size()
    sires = [s_ for s_ in cnt[cnt >= 200].index if s_ in code]
    V = Ef[[code[s_] for s_ in sires]]
    U = unit(V)
    S = U @ U.T
    names = json.loads((EMB / "node_names.json").read_text(encoding="utf-8"))
    rows = []
    for i in range(len(sires)):
        for j in range(i + 1, len(sires)):
            if S[i, j] >= 0.999:
                a_, b_ = sires[i], sires[j]
                da, db = m[m["sire_id"] == a_], m[m["sire_id"] == b_]
                ra, rb = L.roi_ci(m.reset_index(drop=True), (m["sire_id"] == a_).to_numpy(), "ret_win", B=300), L.roi_ci(m.reset_index(drop=True), (m["sire_id"] == b_).to_numpy(), "ret_win", B=300)
                def z(x1, n1, x2, n2):
                    p = (x1 + x2) / (n1 + n2)
                    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
                    return float((x1 / n1 - x2 / n2) / se) if se > 0 else np.nan
                rows.append({"a": a_, "b": b_, "name_a": names.get(a_, "")[:14], "name_b": names.get(b_, "")[:14], "n_a": len(da), "n_b": len(db),
                             "win_a": float(da["win"].mean()), "win_b": float(db["win"].mean()), "top3_a": float(da["top3"].mean()), "top3_b": float(db["top3"].mean()),
                             "z_win": z(da["win"].sum(), len(da), db["win"].sum(), len(db)), "z_top3": z(da["top3"].sum(), len(da), db["top3"].sum(), len(db)),
                             "roi_win_a": ra["roi"], "roi_win_b": rb["roi"]})
    zs = np.array([r["z_top3"] for r in rows if np.isfinite(r["z_top3"])])
    out = {"variant": a.variant, "n_sires": len(sires), "n_identical_pairs": len(rows), "pairs": rows,
           "z_top3_sd": float(zs.std()) if len(zs) > 2 else None, "share_abs_z_gt2": float((np.abs(zs) > 2).mean()) if len(zs) else None}
    _save_json(f"sirepairs_{a.variant}.json", out)
    _log(f"同一構成の種牡馬ペア {len(rows)}、複勝率の z の標準偏差 {out['z_top3_sd']}（偶然なら 1）")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("neighbors", "affinity", "curve", "groups", "sirepairs"):
        p = sub.add_parser(n)
        p.add_argument("--variant", required=True)
    a = ap.parse_args()
    {"neighbors": cmd_neighbors, "affinity": cmd_affinity, "curve": cmd_curve, "groups": cmd_groups, "sirepairs": cmd_sirepairs}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
