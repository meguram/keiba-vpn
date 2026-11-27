"""詳細系統（大系統の中の、種牡馬の血統ベクトルの群）の作りと、as-of の割り当て（記述。検定や買い方ではない）。

    KEIBA_SESSION=aptitude-group OMP_NUM_THREADS=2 nice -n 19 python -m src.research.aptitude_group.detailed_lines build    # 版 v1 を作る
    ... detailed_lines assign   # 毎年 1 月 1 日の切り口の割り当て・初年度と 3 年後の一致の確かめ

規則（数を見る前に固定。log.md「詳細系統の固定」）:
- 大系統: 種牡馬の祖先の鎖（data/local/pedigree_lines/sire_line_origin.json の netkeiba の「○○系」）を父から上へたどり、
  走数 300 以上の種牡馬のうち、その札を鎖に持つものが 8 頭以上ある最初の札。
- 群: 大系統の中で、13 本の z（sire_vectors の版の尺度で固定）を Ward の階層クラスタで k = 1〜4 に分ける。種牡馬 6 頭未満の大系統は 1 群。
  k は、シルエット ≥ 0.15・ブートストラップ（開催の単位の引き直し 20 回）の ARI の平均 ≥ 0.6・2 期（2016〜19・2020〜24）の ARI ≥ 0.4・
  どの群も 2 頭以上、を満たす一番大きい k（満たさなければ 1）。
- 名前: 大系統の短い名前 ＋ 群の中心の、大系統の平均からのずれが一番大きい次元の言葉（重なるときだけ 2 つ目を足す）。
- 割り当て assign(F, t, ver): t より前の走りだけで縮めた x̂（版の尺度の z）と、測りの分散 K/(n+K) を作り、
  事後 ∝ 事前 × Π_d N(x̂_d; μ_cd, σ²_cd + K/(n_d+K))。事前（v2）= 同じ父の父の、ほかの息子（同じ大系統）の 1 回目の事後の平均 × 0.7 ＋ 均等 × 0.3。
  v3 は、父の父が同じ版で走りを持てばその 1 回目の事後を先に使う。どちらも無ければ均等。学習と推論で同じ関数を使う。
- v2 の群の次元は主の 6 本（13 本は v1）。
- 注意: 群の中心は 2016〜24 の全期間で作るので、過去の時点の割り当てには群の作りの先取りが残る（前向きの使い道では無い）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.research.aptitude_group import common
from src.research.aptitude_group.lines import MIN_LINE, line_labels, short
from src.research.aptitude_group.sire_vectors import DIMS, K, MIN_RUNS, runs, shrink
from src.research.pedigree_lab import data

ROOT = Path(__file__).resolve().parents[3]
VERSION = 2              # main() の --version で替える（2 = 大系統の中・3 = 大系統の枠を外す）
VER_PATH = ROOT / f"config/aptitude_group/detailed_lines_v{VERSION}.json"
GLOBAL = "全体"            # v3 の群の入れ物の名前
MAIN6 = ["dist", "surf", "wet", "pos", "ts", "gear"]      # v2 で群を作る次元（log.md「詳細系統 v2 の固定」）
N_NEIGHBORS = 10
MIN_GROUP = 6
N_BOOT = 20
TAGS = {"dist": ("長", "短"), "surf": ("ダ", "芝"), "wet": ("道悪", "良"), "pos": ("先行", "差し"), "ts": ("瞬発", "持続"),
        "gear": ("ギア", "平坦"), "steep": ("急坂", "平坦"), "small": ("小回り", "大箱"), "straight": ("直線", "短直"),
        "left": ("左", "右"), "yoshiba": ("洋芝", "野芝"), "kyoto": ("京都", "京都苦")}


def z_fixed(V: pd.DataFrame, mu: dict, sd: dict) -> pd.DataFrame:
    return pd.DataFrame({d: (V[d] - mu[d]) / sd[d] for d in DIMS}, index=V.index)


def weighted_scale(V: pd.DataFrame) -> tuple[dict, dict]:
    w = V["runs"].to_numpy(float)
    mu, sd = {}, {}
    for d in DIMS:
        x = V[d].to_numpy(float)
        mu[d] = float(np.average(x, weights=w))
        sd[d] = float(np.sqrt(np.average((x - mu[d]) ** 2, weights=w))) or 1.0
    return mu, sd


def feats(Z: pd.DataFrame) -> np.ndarray:
    """群を作る次元（v2 は主の 6 本。道悪は芝とダートの平均）。"""
    W = Z.copy()
    W["wet"] = (W["wet_T"] + W["wet_D"]) / 2
    return W[MAIN6].to_numpy(float)


def ward(X: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return np.ones(len(X), int)
    return fcluster(linkage(X, "ward"), k, criterion="maxclust")


def name_clusters(label: str, centers: list[np.ndarray], line_mean: np.ndarray, reps: list[str] | None = None) -> list[tuple[str, str]]:
    """中心（MAIN6）の、大系統の平均からのずれが一番大きい次元の言葉。重なるときだけ 2 つ目を足す。"""
    def feat(c):
        diff = {k: float(c[i] - line_mean[i]) for i, k in enumerate(MAIN6)}
        return sorted(MAIN6, key=lambda k: -abs(diff[k])), diff
    names, descs = [], []
    ff = [feat(c) for c in centers]
    base = [TAGS[o[0]][0 if d[o[0]] > 0 else 1] for o, d in ff]
    for i, (o, d) in enumerate(ff):
        nm = base[i]
        if base.count(nm) > 1:
            nm += TAGS[o[1]][0 if d[o[1]] > 0 else 1]
        if reps is None:
            names.append(f"{short(label)}・{nm}")
        else:                                            # v3: 特徴 1〜2 つ（2 つ目は |ずれ| ≥ 0.5 のとき）＋ 代表の種牡馬
            if nm == base[i] and abs(d[o[1]]) >= 0.5:
                nm += TAGS[o[1]][0 if d[o[1]] > 0 else 1]
            names.append(f"{nm}（{reps[i]}型）")
        descs.append("、".join(f"{k} {d[k]:+.2f}" for k in o[:4]))
    seen: dict[str, int] = {}
    out = []
    for nm, ds in zip(names, descs):
        seen[nm] = seen.get(nm, 0) + 1
        out.append((nm if seen[nm] == 1 else f"{nm}{seen[nm]}", ds))
    return out


def boot_frames(F: pd.DataFrame, n: int, seed: int = 4242):
    meets = F["meeting"].astype(str).to_numpy()
    idx = pd.Series(np.arange(len(F))).groupby(meets).indices
    keys = np.array(sorted(idx))
    for b in range(n):
        rng = np.random.default_rng(seed + b)
        pick = rng.choice(len(keys), len(keys), replace=True)
        yield F.iloc[np.concatenate([idx[keys[i]] for i in pick])].reset_index(drop=True)


def cmd_build() -> dict:
    F = runs()
    V, _ = shrink(F, "sire", "ss", "L4")
    V = V[V["runs"] >= MIN_RUNS]
    mu, sd = weighted_scale(V)
    Z = z_fixed(V, mu, sd)
    lab = line_labels(set(V.index))
    V["line"] = V.index.map(lab)
    if VERSION >= 3:
        groups = {GLOBAL: list(V.index)}
    else:
        groups = {g: list(ix) for g, ix in V.dropna(subset=["line"]).groupby("line").groups.items()}
    from src.research.aptitude_group.sire_vectors import names as horse_names
    nmap = horse_names(set(V.index), F)
    # ブートストラップと 2 期のベクトル（版の尺度で z）
    BZ = []
    for Fb in boot_frames(F, N_BOOT):
        Vb, _ = shrink(Fb, "sire", "ss", "L4")
        BZ.append(z_fixed(Vb, mu, sd))
        print("boot", len(BZ), flush=True)
    PZ = []
    for m in (F["date"].dt.year <= 2019, F["date"].dt.year >= 2020):
        Vp, _ = shrink(F[m].reset_index(drop=True), "sire", "ss", "L4")
        PZ.append(z_fixed(Vp[Vp["runs"] >= 150], mu, sd))
    ver: dict = {"version": VERSION, "cluster_dims": MAIN6, "made": pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d %H:%M"), "dims": DIMS, "k_shrink": K, "min_runs": MIN_RUNS,
                 "z_mu": mu, "z_sd": sd, "rules": __doc__, "groups": {}, "sire_line": {h: l for h, l in lab.items()}}
    for g, ids in sorted(groups.items(), key=lambda t: -len(t[1])):
        X = feats(Z.loc[ids])
        tried = {}
        best = 1
        if len(ids) >= MIN_GROUP:
            for k in (range(3, 13) if VERSION >= 3 else (2, 3, 4)):
                labs = ward(X, k)
                if np.bincount(labs)[1:].min() < 2:
                    tried[k] = {"min_size": int(np.bincount(labs)[1:].min())}
                    continue
                sil = float(silhouette_score(X, labs))
                aris = []
                for B in BZ:
                    common_ids = [h for h in ids if h in B.index]
                    if len(common_ids) == len(ids):
                        aris.append(adjusted_rand_score(labs, ward(feats(B.loc[ids]), k)))
                both = [h for h in ids if h in PZ[0].index and h in PZ[1].index]
                pari = float("nan")
                if len(both) >= k + 2:
                    pari = float(adjusted_rand_score(ward(feats(PZ[0].loc[both]), k), ward(feats(PZ[1].loc[both]), k)))
                ok = sil >= 0.15 and aris and np.mean(aris) >= 0.6 and np.isfinite(pari) and pari >= 0.4
                tried[k] = {"sil": round(sil, 3), "boot_ari": round(float(np.mean(aris)), 3) if aris else None, "period_ari": round(pari, 3),
                            "period_n": len(both), "ok": bool(ok)}
                if ok:
                    best = k
        labs = ward(X, best)
        centers = [X[labs == c].mean(0) for c in range(1, best + 1)]
        var = [np.maximum(X[labs == c].var(0), 0.1) if (labs == c).sum() > 1 else np.full(X.shape[1], 0.5) for c in range(1, best + 1)]
        reps = None
        if VERSION >= 3:
            reps = [nmap.get(max((ids[i] for i in np.flatnonzero(labs == c)), key=lambda h: V.loc[h, "runs"]), "?") for c in range(1, best + 1)]
        nm = name_clusters(g, centers, X.mean(0), reps)
        ver["groups"][g] = {"short": short(g), "n_sires": len(ids), "k": best, "tried": tried,
                            "clusters": [{"cid": f"{g}#{c + 1}", "name": nm[c][0], "desc": nm[c][1], "center": centers[c].round(4).tolist(),
                                          "var": var[c].round(4).tolist(), "members": [ids[i] for i in np.flatnonzero(labs == c + 1)]}
                                         for c in range(best)]}
        if VERSION >= 3:                                  # 新しい群と元の大系統の対応表（どの系統がどの群に散ったか）
            lab_of = {h: lab.get(h, "札なし") for h in ids}
            ver["groups"][g]["crosstab"] = {c["name"]: pd.Series([lab_of[h] for h in c["members"]]).value_counts().to_dict()
                                            for c in ver["groups"][g]["clusters"]}
            ver["names"] = {h: nmap.get(h, h) for h in ids}
        print(g, len(ids), best, tried, [c["name"] for c in ver["groups"][g]["clusters"]], flush=True)
    VER_PATH.parent.mkdir(parents=True, exist_ok=True)
    VER_PATH.write_text(json.dumps(ver, ensure_ascii=False, indent=1))
    return {"groups": {g: {"k": v["k"], "n": v["n_sires"], "names": [c["name"] for c in v["clusters"]]} for g, v in ver["groups"].items()},
            "unlabeled": int(V["line"].isna().sum())}


def n_main6(row: pd.Series) -> np.ndarray:
    """主の 6 本の件数（道悪は芝とダートの件数の和）。"""
    return np.array([row["n_wet_T"] + row["n_wet_D"] if d == "wet" else row[f"n_{d}"] for d in MAIN6], float)


def group_of(sire: str, ss: str, ver: dict) -> str | None:
    """種牡馬の大系統（版の表 → 父の父の大系統 の順。どちらも無ければ None）。"""
    if ver.get("version", 1) >= 3:
        return GLOBAL
    sl = ver["sire_line"]
    g = sl.get(sire) or sl.get(ss)
    return g if g in ver["groups"] else None


def assign(F: pd.DataFrame, t: pd.Timestamp, ver: dict, sires: pd.DataFrame) -> pd.DataFrame:
    """t より前の走りだけで、sires（列 sire・ss）の各種牡馬に詳細系統の事後の確率を付ける。学習と推論で同じ関数。"""
    Fb = F[F["date"] < t].reset_index(drop=True)
    V, _ = shrink(Fb, "sire", "ss", "L4") if len(Fb) else (pd.DataFrame(), None)
    mu, sd = ver["z_mu"], ver["z_sd"]
    Z = z_fixed(V, mu, sd) if len(V) else pd.DataFrame(columns=DIMS)
    rows = []
    for s, ss in zip(sires["sire"], sires["ss"]):
        g = group_of(s, ss, ver)
        if g is None:
            continue
        cl = ver["groups"][g]["clusters"]
        k = len(cl)
        n_runs = int(V.loc[s, "runs"]) if s in V.index else 0
        if s in Z.index and k > 1:
            x = feats(Z.loc[[s]])[0]
            n = n_main6(V.loc[s])
            se2 = K / (n + K)
            ll = np.array([-0.5 * np.sum((x - np.array(c["center"])) ** 2 / (np.array(c["var"]) + se2) + np.log(np.array(c["var"]) + se2)) for c in cl])
        else:
            ll = np.zeros(k)
        rows.append({"sire": s, "ss": ss, "group": g, "runs": n_runs, "ll": ll})
    R = pd.DataFrame(rows)
    if R.empty:
        return R

    def post(ll, prior):
        a = ll + np.log(prior)
        a = np.exp(a - a.max())
        return a / a.sum()
    R["p1"] = [post(ll, np.full(len(ll), 1 / len(ll))) for ll in R["ll"]]
    # 事前: 父の父が同じ群の入れ物にいて走りがあれば、その 1 回目の事後（v3）。無ければ同じ父の父のほかの息子の平均（v2）。どちらも無ければ均等。
    # ss が空の種牡馬どうしは兄弟にしない（critic 2 回目の 14）。父の父の 1 回目の事後には、本人の走りは入らない（本人は息子の側）。
    by_sire = {s: (g, p1, n) for s, g, p1, n in zip(R["sire"], R["group"], R["p1"], R["runs"])}
    pri = []
    for i, r in R.iterrows():
        k = len(r["ll"])
        par = by_sire.get(r["ss"]) if r["ss"] else None
        bro = R[(R["ss"] == r["ss"]) & (R["group"] == r["group"]) & (R["sire"] != r["sire"]) & (R["runs"] > 0)] if r["ss"] else R.iloc[0:0]
        if ver.get("version", 1) >= 3 and par is not None and par[0] == r["group"] and par[2] > 0:   # 父の父の事後を使うのは v3 だけ（v2 の固定は兄弟だけ。critic 3 回目の 1）
            pri.append(0.7 * par[1] + 0.3 / k)
        elif len(bro):
            pri.append(0.7 * np.mean(np.stack(bro["p1"].to_numpy()), 0) + 0.3 / k)
        else:
            pri.append(np.full(k, 1 / k))
    R["prior"] = pri
    R["p"] = [post(ll, p) for ll, p in zip(R["ll"], R["prior"])]
    R["label"] = [ver["groups"][g]["clusters"][int(np.argmax(p))]["cid"] for g, p in zip(R["group"], R["p"])]
    # 均等な事前の argmax は群の番号で決まるので、札を付けない（critic 2 回目の 5）
    R["prior_label"] = [None if np.ptp(p) < 1e-12 else ver["groups"][g]["clusters"][int(np.argmax(p))]["cid"] for g, p in zip(R["group"], R["prior"])]
    R["as_of"] = t.strftime("%Y-%m-%d")
    return R.drop(columns=["ll", "p1"])


def neighbors(F: pd.DataFrame, t: pd.Timestamp, ver: dict, sires: pd.DataFrame, k: int = N_NEIGHBORS) -> pd.DataFrame:
    """離れた群が無いときの割り当て（v2 の固定の形）。t より前の走りだけで、各種牡馬に近い種牡馬の上位 k 頭（全体・同じ大系統）と、
    その重み付きのベクトルを返す。比べる相手も t の時点で走数 300 以上の種牡馬（as-of）。学習と推論で同じ関数。
    距離² = Σ (x − y)² / (1 + 測りの分散 x + 測りの分散 y)（主の 6 本・版の尺度の z）。重み exp(−距離² / 2)。"""
    Fb = F[F["date"] < t].reset_index(drop=True)
    if not len(Fb):
        return pd.DataFrame()
    V, _ = shrink(Fb, "sire", "ss", "L4")
    Z = z_fixed(V, ver["z_mu"], ver["z_sd"])
    X = feats(Z)
    SE = np.stack([K / (n_main6(V.loc[h]) + K) for h in V.index])
    pos = {h: i for i, h in enumerate(V.index)}
    ref = np.flatnonzero(V["runs"].to_numpy() >= ver["min_runs"])
    rows = []
    for s, ss in zip(sires["sire"], sires["ss"]):
        g = group_of(s, ss, ver)
        if s not in pos:
            rows.append({"sire": s, "group": g, "runs": 0, "as_of": t.strftime("%Y-%m-%d"), "near": [], "near_line": [], "wvec": None})
            continue
        i = pos[s]
        r = ref[ref != i]
        d2 = (((X[r] - X[i]) ** 2) / (1 + SE[r] + SE[i])).sum(1)
        w = np.exp(-d2 / 2)
        o = np.argsort(d2)[:k]
        same = [j for j in np.argsort(d2) if group_of(V.index[r[j]], "", ver) == g and g is not None][:k]
        wv = (w[o, None] * X[r[o]]).sum(0) / max(w[o].sum(), 1e-12)
        rows.append({"sire": s, "group": g, "runs": int(V["runs"].iloc[i]), "as_of": t.strftime("%Y-%m-%d"),
                     "near": [V.index[r[j]] for j in o], "near_d": [round(float(np.sqrt(d2[j])), 3) for j in o],
                     "near_line": [V.index[r[j]] for j in same], "wvec": [round(float(x), 3) for x in wv],
                     "vec": [round(float(x), 3) for x in X[i]]})
    return pd.DataFrame(rows)


def cmd_assign() -> dict:
    ver = json.loads(VER_PATH.read_text(encoding="utf-8"))
    F = runs()
    S = F[["sire", "ss"]].drop_duplicates("sire")
    first_by = F.groupby("sire")["horse_id"].agg(lambda s: int(s.str[:4].min()))
    first_run = F.groupby("sire")["date"].min().dt.year
    if all(v["k"] == 1 for v in ver["groups"].values()):
        return cmd_neighbors(F, S, ver, first_by, first_run)
    snaps = {}
    for y in range(2017, 2026):
        snaps[y] = assign(F, pd.Timestamp(f"{y}-01-01"), ver, S)
        print("snap", y, len(snaps[y]), flush=True)
    # 確かめ: 初年度の産駒が 2014 年生まれ以降・k ≥ 2 の大系統。初年度の末（y0+1 の 1 月 1 日）と 3 年後（y0+4 の 1 月 1 日）
    res = []
    for s in S["sire"]:
        if s not in first_by.index or first_by[s] < 2014:
            continue
        y0 = int(first_run[s])
        if y0 + 4 > 2025:
            continue
        a, b = snaps[y0 + 1], snaps[y0 + 4]
        ra, rb = a[a["sire"] == s], b[b["sire"] == s]
        if ra.empty or rb.empty or len(ver["groups"][ra["group"].iloc[0]]["clusters"]) < 2:
            continue
        member = any(s in c["members"] for c in ver["groups"][ra["group"].iloc[0]]["clusters"])
        res.append({"sire": s, "member": member, "group": ra["group"].iloc[0], "y0": y0, "runs_a": int(ra["runs"].iloc[0]), "runs_b": int(rb["runs"].iloc[0]),
                    "a": ra["label"].iloc[0], "prior_a": ra["prior_label"].iloc[0], "b": rb["label"].iloc[0]})
    R = pd.DataFrame(res)
    out: dict = {"n": int(len(R))}
    if len(R):
        sizes = {g: np.array([len(c["members"]) for c in v["clusters"]], float) for g, v in ver["groups"].items()}
        rand = float(np.mean([((sizes[g] / sizes[g].sum()) ** 2).sum() for g in R["group"]]))
        # 群の作りに入った種牡馬（member）は、中心に本人の先の走りが入るので一致が上ぶれする（critic 2 回目の 3）。入らない種牡馬を主に読む
        for nm, sub in (("all", R), ("non_member", R[~R["member"]]), ("member", R[R["member"]])):
            if len(sub):
                pa = sub.dropna(subset=["prior_a"])
                out[nm] = {"n": int(len(sub)), "agree_first_vs_3y": round(float((sub["a"] == sub["b"]).mean()), 3),
                           "prior_n": int(len(pa)), "agree_prior_vs_3y": round(float((pa["prior_a"] == pa["b"]).mean()), 3) if len(pa) else None,
                           "median_runs_first": int(sub["runs_a"].median()), "median_runs_3y": int(sub["runs_b"].median())}
        out["agree_random"] = round(rand, 3)
    # 可視化用の年ごとの割り当て
    tl = pd.concat(snaps.values(), ignore_index=True)
    tl["p"] = tl["p"].map(lambda a: [round(float(x), 3) for x in a])
    tl[["sire", "group", "as_of", "runs", "label", "p"]].to_json(common.OUT / f"detailed_timeline_v{VERSION}.json", orient="records", force_ascii=False)
    R.to_csv(common.OUT / f"detailed_check_v{VERSION}.csv", index=False)
    return out


def cmd_neighbors(F, S, ver, first_by, first_run) -> dict:
    """近傍の割り当ての確かめ: 初年度の末と 3 年後の、近い 10 頭の重なり（Jaccard）。比べの相手は無作為の 10 頭どうしの重なり。"""
    snaps = {}
    for y in range(2017, 2026):
        snaps[y] = neighbors(F, pd.Timestamp(f"{y}-01-01"), ver, S).set_index("sire")
        print("snap", y, len(snaps[y]), flush=True)
    res = []
    for s in S["sire"]:
        if s not in first_by.index or first_by[s] < 2014:
            continue
        y0 = int(first_run[s])
        if y0 + 4 > 2025 or s not in snaps[y0 + 1].index or s not in snaps[y0 + 4].index:
            continue
        a, b = set(snaps[y0 + 1].loc[s, "near"]), set(snaps[y0 + 4].loc[s, "near"])
        if not a or not b:
            continue
        nref = int((snaps[y0 + 4]["runs"] >= ver["min_runs"]).sum())
        res.append({"sire": s, "y0": y0, "runs_a": int(snaps[y0 + 1].loc[s, "runs"]), "runs_b": int(snaps[y0 + 4].loc[s, "runs"]),
                    "jacc": len(a & b) / len(a | b), "rand": N_NEIGHBORS / max(2 * nref - N_NEIGHBORS, 1)})
    R = pd.DataFrame(res)
    out = {"mode": "neighbors", "n": int(len(R))}
    if len(R):
        out |= {"jaccard_mean": round(float(R["jacc"].mean()), 3), "jaccard_random": round(float(R["rand"].mean()), 3),
                "median_runs_first": int(R["runs_a"].median()), "median_runs_3y": int(R["runs_b"].median())}
    tl = pd.concat([v.reset_index() for v in snaps.values()], ignore_index=True)
    tl.to_json(common.OUT / f"detailed_neighbors_v{VERSION}.json", orient="records", force_ascii=False)
    R.to_csv(common.OUT / f"detailed_check_v{VERSION}.csv", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "assign"])
    ap.add_argument("--version", type=int, default=2, choices=[2, 3])
    a = ap.parse_args()
    global VERSION, VER_PATH
    VERSION = a.version
    VER_PATH = ROOT / f"config/aptitude_group/detailed_lines_v{VERSION}.json"
    common.open_token("select", f"detailed_lines_{a.cmd}_v{VERSION}", "詳細系統（記述・着順の集計は種牡馬のベクトルだけ・開発の行 2016〜24）")
    out = cmd_build() if a.cmd == "build" else cmd_assign()
    p = common.OUT / f"detailed_{a.cmd}_v{VERSION}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(out, ensure_ascii=False, default=str)[:3000])
    data.finish_all(str(p))


if __name__ == "__main__":
    main()
