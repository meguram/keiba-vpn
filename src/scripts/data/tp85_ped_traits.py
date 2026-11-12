"""tp-85: 父・母父・母母父・母母母父の 4 頭から、馬の観点（走法軸・距離・コース・馬場・早熟・休み明け・初戦）がどこまで言えるかを、80% 区間つきで測る。ホストで実行。

python -m src.scripts.data.tp85_ped_traits [--traits dirt_share,ax_ts] [--quick] [--force]
出力: data/local/modeling/tp85_ped_traits/results/<観点>.json（観点ごとに保存。あれば飛ばす）、summary.json。

モデル（混合モデル）: 馬の観点の値 y = 切片 + Σ(位置の重み × 祖先の効果) + 馬ごとのずれ + 測定誤差。
  祖先の効果は血統の木に沿って足し合わせる（大系統 → 小系統 → 父系の祖先 → 種牡馬そのもの）。細かい層は上の層へ縮む（階層ベイズ）。
  層ごとの分散は EM（最尤）で推定する。測定誤差の分散は馬ごとの走数から決める（走数が少ない馬ほど大きい）。
  A: 同じ種牡馬の効果を 4 つの位置で共有する（位置の重み 1, 1/2, 1/4, 1/8 = 遺伝子の割合）。A1〜A4 は使う位置を増やしたもの。
  B: 位置ごとに別の効果にする（母父と父で効き方が違ってよい）。
評価は時間を分ける: 学習 = 初出走が 2022 年末までの馬、検証 = 2023 年以降に初出走した馬（新しい馬への当てはめ）。2026 年は使わない。市場オッズ・人気は使わない。
指標: 真の値の分散のうち血統で説明できる割合（測定誤差を除いた R²）、80% 区間の被覆率（目標 0.80）、区間幅の比（血統なし = 1）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("TP39_THREADS", "6"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ.get("TP39_THREADS", "6"))
os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("TP39_THREADS", "6"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.linalg import cho_factor, cho_solve  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / ("data/local/modeling/tp85_ped_traits" + os.environ.get("TP85_OUT_SUFFIX", ""))
FLAT = ROOT / "data/page_reference/tables/{y}/race_result_flat.parquet"
AXES_PATH = ROOT / "data/local/running_profile/run_axes.parquet"
BASIC_PATH = ROOT / "data/local/running_profile/basic_speed.parquet"
BASIC_SCALE = 90.0
ORIGIN_PATH = ROOT / "data/meta/pedigree_lines/sire_line_origin.json"
PED_DIRS = [ROOT / "data/local/horse_pedigree_5gen", ROOT / "data/local/horse_pedigree_10gen"]
LAST_DATE = pd.Timestamp("2025-12-31")
SPLIT = pd.Timestamp("2022-12-31")
SPLIT_AX = pd.Timestamp("2025-02-28")                    # 走法軸の実測ラップは 2024 年からなので、軸の初出走日で分ける
POS = {1: (1, 0), 2: (2, 2), 3: (3, 6), 4: (4, 14)}      # 父 / 母父 / 母母父 / 母母母父（generation, position）
POS_NAME = {1: "父", 2: "母父", 3: "母母父", 4: "母母母父"}
POS_COEF = {1: 1.0, 2: 0.5, 3: 0.25, 4: 0.125}
Z80 = 1.2815515655446004
AXIS_TRAITS = ["ts", "sustain", "debut", "gear", "endure", "front", "cr", "makuri"]
AXIS_LABEL = {"ts": "TS（末脚の速さ）", "sustain": "TS持続力", "debut": "出脚", "gear": "ギアチェンジ", "endure": "Hペース耐性",
              "front": "先行力", "cr": "Cr性能", "makuri": "まくり特性"}
VARIANTS = [("A1", [1], "shared"), ("A2", [1, 2], "shared"), ("A3", [1, 2, 3], "shared"), ("A4", [1, 2, 3, 4], "shared"), ("B4", [1, 2, 3, 4], "pos")]
SEX_VARIANTS = [("A2s", [1, 2], "shared"), ("A2i", [1, 2], "shared"), ("A4s", [1, 2, 3, 4], "shared"), ("A4i", [1, 2, 3, 4], "shared"), ("B4s", [1, 2, 3, 4], "pos")]   # s = 性別の平均を引く / i = さらに牡牝ごとのずれ
SEXMODE = {"A2s": "fix", "A2i": "inter", "A4s": "fix", "A4i": "inter", "B4s": "fix"}


def _log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ───────────────────────── 1 走ごとの表
def load_runs() -> pd.DataFrame:
    cols = ["race_id", "date", "surface", "distance", "direction", "track_condition", "finish_position", "horse_id", "sex_age", "passing_order"]
    d = pd.concat([pd.read_parquet(str(FLAT).format(y=y), columns=cols) for y in range(2020, 2026)], ignore_index=True)
    d = d[d["surface"].isin(["芝", "ダート"])].copy()
    d["rank"] = pd.to_numeric(d["finish_position"], errors="coerce")
    d = d[d["rank"].notna() & d["horse_id"].notna()].copy()
    d["horse_id"] = d["horse_id"].astype(str)
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] <= LAST_DATE]
    d["N"] = d.groupby("race_id")["rank"].transform("size")
    d = d[d["N"] >= 5].copy()
    d["res"] = (d["N"] + 1 - 2 * d["rank"]) / (2 * (d["N"] - 1))                   # 大きいほど好走（-0.5〜0.5）
    d["age"] = pd.to_numeric(d["sex_age"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    d["sex"] = d["sex_age"].astype(str).str.extract(r"(牡|牝|セ)")[0].replace({"セ": "牡"})        # 去勢は牡として扱う
    d = d.sort_values(["horse_id", "date", "race_id"], kind="stable").reset_index(drop=True)
    g = d.groupby("horse_id", sort=False)
    d["idx"] = g.cumcount()
    d["first_date"] = g["date"].transform("first")
    d["interval"] = g["date"].diff().dt.days
    d["is_debut"] = (d["idx"] == 0) & ((d["first_date"] >= "2021-01-01") | (d["age"] == 2))
    d["dist_rel"] = d["distance"] - d.groupby("surface")["distance"].transform("mean")
    first = pd.to_numeric(d["passing_order"].astype(str).str.split("-").str[0], errors="coerce")
    d["q_now"] = ((first - 1) / (d["N"] - 1)).where(first.between(1, 99))
    d["dirt"] = (d["surface"] == "ダート").astype(float)
    d["left"] = (d["direction"] == "左").astype(float).where(d["direction"].isin(["右", "左"]))
    d["wet"] = d["track_condition"].isin(["稍重", "重", "不良"])
    d["good"] = d["track_condition"] == "良"
    return d


def _pooled_var(d: pd.DataFrame, col: str, minn: int) -> tuple[pd.DataFrame, float]:
    x = d[d[col].notna()]
    g = x.groupby("horse_id")[col]
    n, m = g.size(), g.mean()
    ss = ((x[col] - x["horse_id"].map(m)) ** 2).groupby(x["horse_id"]).sum()
    ok = n >= minn
    s2 = float(ss[ok].sum() / (n[ok] - 1).sum())
    return pd.DataFrame({"y": m[ok], "n": n[ok]}), s2


def mean_trait(d: pd.DataFrame, col: str, minn: int = 3) -> pd.DataFrame:
    t, s2 = _pooled_var(d, col, minn)
    t["v"] = s2 / t["n"]
    return t.reset_index()[["horse_id", "y", "v", "n"]]


def contrast_trait(d: pd.DataFrame, a: pd.Series, b: pd.Series, strat: str | None, s2: float) -> pd.DataFrame:
    """馬の中で A の走の res の平均 − B の走の res の平均（strat があるときは芝・ダートなど層ごとに出して逆分散で合算）。"""
    keys = ["horse_id"] + ([strat] if strat else [])
    ga = d[a].groupby([d[k] for k in keys])["res"].agg(["mean", "size"]).add_prefix("a_")
    gb = d[b].groupby([d[k] for k in keys])["res"].agg(["mean", "size"]).add_prefix("b_")
    j = ga.join(gb, how="inner")
    j["diff"] = j["a_mean"] - j["b_mean"]
    j["var"] = s2 * (1 / j["a_size"] + 1 / j["b_size"])
    j["w"] = 1 / j["var"]
    j = j.reset_index()
    j["wd"] = j["w"] * j["diff"]
    r = j.groupby("horse_id").agg(wd=("wd", "sum"), w=("w", "sum"), n=("a_size", "sum"))
    return pd.DataFrame({"horse_id": r.index, "y": (r["wd"] / r["w"]).to_numpy(), "v": (1 / r["w"]).to_numpy(), "n": r["n"].to_numpy()})


def build_traits(d: pd.DataFrame) -> dict[str, pd.DataFrame]:
    s2_res = float(((d["res"] - d.groupby("horse_id")["res"].transform("mean")) ** 2).sum() / (len(d) - d["horse_id"].nunique()))
    T: dict[str, pd.DataFrame] = {}
    T["dirt_share"] = mean_trait(d, "dirt")
    T["dist_run"] = mean_trait(d, "dist_rel")
    top3 = d[d["rank"] <= 3]
    T["dist_good"] = mean_trait(top3.assign(x=top3["dist_rel"]), "x", minn=2)
    T["left_share"] = mean_trait(d, "left")
    T["style_q"] = mean_trait(d, "q_now")
    T["wet_diff"] = contrast_trait(d, d["wet"], d["good"], "surface", s2_res)
    T["mature"] = contrast_trait(d, d["age"] >= 4, d["age"] <= 3, None, s2_res)
    nd = ~d["is_debut"]
    T["rest_diff"] = contrast_trait(d, nd & (d["interval"] >= 84), nd & (d["interval"] <= 56), None, s2_res)
    later = d["idx"].between(1, 4)
    deb = d[d["is_debut"]][["horse_id", "res"]].rename(columns={"res": "r0"})
    lg = d[later].groupby("horse_id")["res"].agg(["mean", "size"])
    j = deb.merge(lg, left_on="horse_id", right_index=True)
    j = j[j["size"] >= 2]
    T["debut_diff"] = pd.DataFrame({"horse_id": j["horse_id"], "y": j["r0"] - j["mean"], "v": s2_res * (1 + 1 / j["size"]), "n": j["size"] + 1}).reset_index(drop=True)
    ax = pd.read_parquet(AXES_PATH)
    ax["date"] = pd.to_datetime(ax["date"])
    ax = ax[(ax["date"] <= LAST_DATE) & ~ax["abn"].fillna(False).astype(bool)].copy()
    ax["horse_id"] = ax["horse_id"].astype(str)
    grp = ax.groupby(["surface", "band"])
    for c in AXIS_TRAITS:
        ax[c] = ax[c] - grp[c].transform("mean")            # 芝ダ・距離帯の平均を引く（面の違いを走法の違いと混ぜない）
        T["ax_" + c] = mean_trait(ax, c, minn=2)
    if BASIC_PATH.exists():                                # 基礎スピード = 90×p×s の馬ごとの平均（良馬場のみ。2020 年から）
        bs = pd.read_parquet(BASIC_PATH)
        bs["race_id"], bs["horse_id"] = bs["race_id"].astype(str), bs["horse_id"].astype(str)
        bs = bs.drop_duplicates(["race_id", "horse_id"]).merge(d[["race_id", "horse_id", "surface", "distance"]], on=["race_id", "horse_id"])
        bs["basic"] = BASIC_SCALE * bs["bs_p"] * bs["bs_s"]
        bs["basic"] = bs["basic"] - bs.groupby(["surface", np.where(bs["distance"] <= 1400, "短", np.where(bs["distance"] < 1800, "1600", "1800+"))])["basic"].transform("mean")
        T["basic"] = mean_trait(bs.dropna(subset=["basic"]), "basic", minn=2)
    fd = d.groupby("horse_id")["first_date"].first()
    fax = ax.groupby("horse_id")["date"].min()
    sxm = d.dropna(subset=["sex"]).groupby("horse_id")["sex"].first()
    for k, t in T.items():
        t["sex"] = t["horse_id"].map(sxm).fillna("牡")
        t["first_date"] = t["horse_id"].map(fax if k.startswith("ax_") else fd)
        T[k] = t[t["first_date"].notna() & np.isfinite(t["y"]) & np.isfinite(t["v"]) & (t["v"] > 0)].reset_index(drop=True)
    return T


# ───────────────────────── 血統
def load_ped(hids: list[str]) -> dict[str, dict[int, str]]:
    ped: dict[str, dict[int, str]] = {}
    want = {v: k for k, v in POS.items()}
    for h in hids:
        for dr in PED_DIRS:
            f = dr / h[:4] / f"{h}.json"
            if not f.exists():
                continue
            try:
                anc = json.loads(f.read_text(encoding="utf-8")).get("ancestors") or []
            except Exception:
                continue
            r: dict[int, str] = {}
            for a in anc:
                k = want.get((a.get("generation"), a.get("position")))
                if k and a.get("horse_id"):
                    r[k] = str(a["horse_id"])
            if r:
                ped[h] = r
                break
    return ped


def line_path(lines: list) -> list[str]:
    out: list[str] = []
    for ln in reversed(lines):
        if ln and ln not in out:
            out.append(ln)
    return out


def make_nodes(origin: dict):
    cache: dict[str, list] = {}

    def nodes(sid: str) -> list[tuple[str, str]]:
        if sid in cache:
            return cache[sid]
        r = origin.get(sid)
        out: list[tuple[str, str]] = []
        if r:
            path = line_path(r["lines"])
            out += [("L:" + "/".join(path[: i + 1]), "line") for i in range(len(path))]
            out += [("H:" + hid, "branch") for hid, _ in reversed(r["chain"][:-1])]
        out.append(("S:" + sid, "sire"))
        cache[sid] = out
        return out

    return nodes


def entries(ped, hids, nodes_fn, positions, mode, sexes=None, inter=False) -> pd.DataFrame:
    """inter=True のとき、各祖先の効果に『牡牝ごとのずれ』（キーに性別を付けた別の効果、層は x_ を付ける）を足す。"""
    rows, keys, coef, pos, blk = [], [], [], [], []
    for i, h in enumerate(hids):
        p = ped.get(h)
        if not p:
            continue
        for k in positions:
            a = p.get(k)
            if not a:
                continue
            for nk, kind in nodes_fn(a):
                rows.append(i)
                keys.append(nk if mode == "shared" else f"{k}|{nk}")
                coef.append(POS_COEF[k] if mode == "shared" else 1.0)
                pos.append(k)
                blk.append(kind if mode == "shared" else f"{k}|{kind}")
                if inter:
                    rows.append(i)
                    keys.append(f"X{sexes[i]}|{nk}")
                    coef.append(POS_COEF[k])
                    pos.append(k)
                    blk.append("x_" + kind)
    return pd.DataFrame({"row": rows, "key": keys, "coef": coef, "pos": pos, "blk": blk})


# ───────────────────────── 混合モデルの EM
def rowquad(X: sp.csr_matrix, C: np.ndarray, chunk: int = 4000) -> np.ndarray:
    out = np.empty(X.shape[0])
    for i in range(0, X.shape[0], chunk):
        Xc = X[i:i + chunk]
        out[i:i + chunk] = np.asarray(Xc.multiply(Xc @ C).sum(1)).ravel()
    return out


def fit_em(X: sp.csr_matrix, y: np.ndarray, v: np.ndarray, blk: np.ndarray, nblk: int, iters: int = 40, tol: float = 2e-3) -> dict:
    """X の 0 列目は切片。blk は列ごとの層の番号（切片は -1）。y・v は標準化済み。"""
    n, p = X.shape
    cnt = np.bincount(blk[1:], minlength=nblk).astype(float)
    sig = np.full(nblk, 0.02)
    tau = 0.5
    Xt = X.T.tocsr()
    for it in range(iters):
        w = 1.0 / (tau + v)
        Xw = X.multiply(w[:, None]).tocsr()
        A = (Xt @ Xw).toarray()
        A[np.diag_indices(p)] += np.where(blk >= 0, 1.0 / np.maximum(sig[np.maximum(blk, 0)], 1e-9), 1e-4)
        C = cho_solve(cho_factor(A, lower=True), np.eye(p))
        u = C @ (Xt @ (w * y))
        res = y - X @ u
        zcz = rowquad(X, C)
        g = tau / (tau + v)
        tau_new = float(np.mean(g ** 2 * (res ** 2 + zcz) + g * v))
        u2 = u ** 2 + np.diag(C)
        s_new = np.bincount(blk[1:], u2[1:], minlength=nblk) / np.maximum(cnt, 1)
        s_new = np.where(cnt > 0, s_new, sig)
        ch = max(abs(tau_new - tau) / max(tau, 1e-6), float(np.max(np.abs(s_new - sig) / np.maximum(sig, 1e-6))))
        tau, sig = tau_new, s_new
        if ch < tol:
            break
    return {"u": u, "C": C, "tau": tau, "sig": sig, "iters": it + 1}


def calibration(yte: np.ndarray, mu: np.ndarray, vte: np.ndarray, tau: float, s: float) -> dict:
    """検証の馬を予測の大きさで並べ、端（下位・上位 10%）で予測どおりに実測がずれるかを見る。傾き 1 = 予測の大きさがそのまま当たる、0 = 当たらない。"""
    w = 1.0 / (tau + vte)
    mc, yc = mu - mu.mean(), yte - yte.mean()
    den = float((w * mc ** 2).sum())
    out = {"slope": round(float((w * mc * yc).sum() / den), 3) if den > 0 else None, "slope_se": round(float(1 / np.sqrt(den)), 3) if den > 0 else None}
    lo_q, hi_q = np.quantile(mu, [0.1, 0.9])
    for nm, msk in (("bottom10", mu <= lo_q), ("top10", mu >= hi_q)):
        n = int(msk.sum())
        out[nm] = {"n": n, "pred_diff": round(float((mu[msk].mean() - mu.mean()) * s), 4), "actual_diff": round(float((yte[msk].mean() - yte.mean()) * s), 4),
                   "actual_se": round(float(yte[msk].std() / np.sqrt(max(n, 1)) * s), 4)}
    return out


def fit_variant(name, positions, mode, ped, nodes_fn, tr: pd.DataFrame, te: pd.DataFrame, minh: int = 2, sexmode: str = "none") -> tuple[dict, dict]:
    """sexmode: none = 性別を使わない / fix = 性別ごとの平均を引く / inter = fix + 祖先の効果に牡牝ごとのずれを足す。"""
    t0 = time.time()
    m, s = float(tr["y"].mean()), float(tr["y"].std())
    sex_mean = tr.groupby("sex")["y"].mean() if sexmode != "none" else None
    off_tr = tr["sex"].map(sex_mean).to_numpy() if sexmode != "none" else np.full(len(tr), m)
    off_te = te["sex"].map(sex_mean).fillna(m).to_numpy() if sexmode != "none" else np.full(len(te), m)
    ytr, vtr = ((tr["y"].to_numpy() - off_tr) / s), (tr["v"] / s ** 2).to_numpy()
    yte, vte = ((te["y"] - m) / s).to_numpy(), (te["v"] / s ** 2).to_numpy()
    shift_te = (off_te - m) / s                       # 性別の平均の分（予測に足す）
    inter = sexmode == "inter"
    etr = entries(ped, tr["horse_id"].tolist(), nodes_fn, positions, mode, tr["sex"].tolist(), inter)
    ete = entries(ped, te["horse_id"].tolist(), nodes_fn, positions, mode, te["sex"].tolist(), inter)
    nh = etr.groupby("key")["row"].nunique()
    keep = nh[nh >= minh].index
    col = {k: i + 1 for i, k in enumerate(keep)}
    blocks = sorted(etr["blk"].unique())
    bid = {b: i for i, b in enumerate(blocks)}
    kb = etr.drop_duplicates("key").set_index("key")["blk"]
    blk = np.concatenate([[-1], [bid[kb[k]] for k in keep]]).astype(int)

    def mat(e: pd.DataFrame, n: int):
        e = e.assign(col=e["key"].map(col))
        seen = e[e["col"].notna()]
        Z = sp.coo_matrix((seen["coef"], (seen["row"], seen["col"].astype(int))), shape=(n, len(col) + 1)).tocsr()
        Z = Z + sp.csr_matrix((np.ones(n), (np.arange(n), np.zeros(n, int))), shape=Z.shape)
        return Z, e[e["col"].isna()]

    Xtr, _ = mat(etr, len(tr))
    Xte, unseen = mat(ete, len(te))
    f = fit_em(Xtr, ytr, vtr, blk, len(blocks))
    u, C, tau, sig = f["u"], f["C"], f["tau"], f["sig"]
    mu = Xte @ u + shift_te
    var_u = rowquad(Xte, C)
    un = np.zeros(len(te))
    if len(unseen):
        add = (unseen["coef"] ** 2) * unseen["blk"].map(lambda b: sig[bid[b]] if b in bid else float(sig.mean()))
        un = np.bincount(unseen["row"], add, minlength=len(te))
    var_true = var_u + un + tau
    var_obs = var_true + vte
    r = yte - mu
    tot = float(((yte - 0.0) ** 2).sum())
    noise = float(vte.sum())
    pop_raw = float(yte.var() - vte.mean())
    pop_true = max(pop_raw, 0.05 * float(yte.var()))
    ceiling = max(0.0, 1 - float(vte.mean()) / float(yte.var()))               # 測定誤差のため、どんなモデルでも超えられない R² の上限
    cov = float((np.abs(r) <= Z80 * np.sqrt(var_obs)).mean())
    base_cov = float((np.abs(yte) <= Z80 * np.sqrt(ytr.var() + vte)).mean())
    # 分散の内訳（学習の馬の平均）。位置別・層別
    Etr = etr.assign(vv=etr["coef"] ** 2 * etr["blk"].map(lambda b: sig[bid[b]]))
    n_tr = len(tr)
    by_pos = (Etr.groupby("pos")["vv"].sum() / n_tr).to_dict()
    by_kind = (Etr.assign(kind=Etr["blk"].str.split("|").str[-1]).groupby("kind")["vv"].sum() / n_tr).to_dict()
    expl = float(sum(by_pos.values()))
    out = {
        "n_train": int(len(tr)), "n_test": int(len(te)), "q": int(len(col)), "iters": f["iters"], "sec": round(time.time() - t0, 1),
        "tau_share": round(tau / (tau + expl), 4), "explained_share": round(expl / (tau + expl), 4),
        "share_by_position": {POS_NAME[int(k)]: round(float(x) / (tau + expl), 4) for k, x in by_pos.items()},
        "share_by_kind": {k: round(float(x) / (tau + expl), 4) for k, x in by_kind.items()},
        "r2_true": round(1 - float(((r ** 2).sum() - noise)) / max(tot - noise, 1e-9), 4) if pop_raw > 0.2 * float(yte.var()) else None,
        "ceiling_r2": round(ceiling, 4),
        "r2_obs": round(1 - float((r ** 2).sum()) / tot, 4),
        "cover80": round(cov, 4), "cover80_baseline": round(base_cov, 4),
        "width_ratio": round(float(np.sqrt(var_true).mean() / np.sqrt(pop_true)), 4),
        "halfwidth80_natural": round(float(Z80 * np.sqrt(var_true).mean() * s), 4),
        "pop_sd_true_natural": round(float(np.sqrt(pop_true) * s), 4),
        "unseen_share": round(float(len(unseen) / max(len(ete), 1)), 4),
    }
    out["calib"] = calibration(yte, mu, vte, tau, s)
    pos_prior_var = Etr.groupby(["row", "pos"])["vv"].sum().groupby("pos").mean().to_dict()      # 位置の祖先が分からない馬に足す、事前の分散
    keep_state = {"u": u, "C": C, "col": col, "sig": sig, "bid": bid, "s": s, "m": m, "tau": tau, "pos_prior_var": {int(k): float(v) for k, v in pos_prior_var.items()},
                  "sex_mean": sex_mean.to_dict() if sex_mean is not None else {},
                  "te": {"mu": mu, "var_true": var_true, "vte": vte, "yte": yte}}
    return out, keep_state


def sire_table(state: dict, nodes_fn, ped, tr: pd.DataFrame, origin: dict, topn: int = 8) -> dict:
    u, C, col, s = state["u"], state["C"], state["col"], state["s"]
    cnt = pd.Series([ped.get(h, {}).get(1) for h in tr["horse_id"]]).value_counts()
    rows = []
    for sid, c in cnt.items():
        if c < 40 or not sid:
            continue
        idx = [col[k] for k, _ in nodes_fn(sid) if k in col]
        if not idx:
            continue
        mean = float(u[idx].sum() * POS_COEF[1])
        sd = float(np.sqrt(C[np.ix_(idx, idx)].sum()) * POS_COEF[1])
        rows.append({"sire": origin.get(sid, {}).get("name", sid), "horses": int(c), "effect": round(mean * s, 4), "lo80": round((mean - Z80 * sd) * s, 4), "hi80": round((mean + Z80 * sd) * s, 4)})
    rows.sort(key=lambda r: r["effect"])
    lines = []
    for k, i in col.items():
        if k.startswith("L:"):
            lines.append({"line": k[2:], "effect": round(float(u[i]) * s, 4), "lo80": round(float(u[i] - Z80 * np.sqrt(C[i, i])) * s, 4), "hi80": round(float(u[i] + Z80 * np.sqrt(C[i, i])) * s, 4)})
    lines.sort(key=lambda r: r["effect"])
    return {"low": rows[:topn], "high": rows[-topn:][::-1], "n_sires": len(rows), "lines": lines}


def run_trait(name, t, ped, nodes_fn, origin, quick: bool) -> dict:
    t = t[t["horse_id"].isin(ped)].reset_index(drop=True)
    sp_ = SPLIT_AX if name.startswith("ax_") else SPLIT
    tr, te = t[t["first_date"] <= sp_].reset_index(drop=True), t[t["first_date"] > sp_].reset_index(drop=True)
    _log(f"{name}: 学習 {len(tr):,} 頭 / 検証 {len(te):,} 頭")
    res: dict = {"trait": name, "n_train": len(tr), "n_test": len(te), "variants": {}}
    if len(tr) < 1500 or len(te) < 500:
        res["skipped"] = "頭数が少ない"
        return res
    only = [x for x in os.environ.get("TP85_VARIANTS", "").split(",") if x]
    allv = VARIANTS + SEX_VARIANTS
    for vn, pos, mode in ([v for v in allv if v[0] in only] if only else VARIANTS[3:4] if quick else VARIANTS):
        o, st = fit_variant(vn, pos, mode, ped, nodes_fn, tr, te, sexmode=SEXMODE.get(vn, "none"))
        res["variants"][vn] = o
        _log(f"  {vn}: R²(真)={o['r2_true']} 観測R²={o['r2_obs']:+.3f} 上限={o['ceiling_r2']:.2f} 被覆={o['cover80']:.3f}(血統なし {o['cover80_baseline']:.3f}) 幅比={o['width_ratio']:.3f} 反復={o['iters']} {o['sec']}s q={o['q']}")
        c = o["calib"]
        _log(f"    端の当てはまり: 傾き={c['slope']}±{c['slope_se']} 下位10%(予測{c['bottom10']['pred_diff']:+.3f}→実測{c['bottom10']['actual_diff']:+.3f}±{c['bottom10']['actual_se']:.3f}) 上位10%(予測{c['top10']['pred_diff']:+.3f}→実測{c['top10']['actual_diff']:+.3f}±{c['top10']['actual_se']:.3f})")
        if vn == "A4":
            res["sires"] = sire_table(st, nodes_fn, ped, tr, origin)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traits", default="")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    d = load_runs()
    _log(f"走 {len(d):,} / 馬 {d['horse_id'].nunique():,}")
    T = build_traits(d)
    want = [x for x in a.traits.split(",") if x] or list(T)
    hids = sorted(set().union(*[set(T[k]["horse_id"]) for k in want]))
    ped = load_ped(hids)
    origin = json.loads(ORIGIN_PATH.read_text(encoding="utf-8"))
    nodes_fn = make_nodes(origin)
    cov = {POS_NAME[k]: round(float(np.mean([k in p for p in ped.values()])), 3) for k in POS}
    known = {POS_NAME[k]: round(float(np.mean([p[k] in origin for p in ped.values() if k in p])), 3) for k in POS}
    _log(f"血統あり {len(ped):,}/{len(hids):,} 頭 / 位置別の有り {cov} / 系統が分かる割合 {known}")
    (OUT / "coverage.json").write_text(json.dumps({"horses": len(hids), "with_ped": len(ped), "position_present": cov, "line_known": known}, ensure_ascii=False, indent=1), encoding="utf-8")
    for k in want:
        f = OUT / "results" / f"{k}.json"
        if f.exists() and not a.force:
            _log(f"{k}: 済み")
            continue
        r = run_trait(k, T[k], ped, nodes_fn, origin, a.quick)
        f.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    sm = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((OUT / "results").glob("*.json"))}
    (OUT / "summary.json").write_text(json.dumps(sm, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
