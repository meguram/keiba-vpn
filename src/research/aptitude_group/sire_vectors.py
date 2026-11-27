"""種牡馬の血統ベクトル（記述と可視化。検定や買い方ではない。開発の行 2016〜24 だけ・費用 0 円）。

    KEIBA_SESSION=aptitude-group OMP_NUM_THREADS=2 nice -n 19 python -m src.research.aptitude_group.sire_vectors

ユーザーの依頼（11/27・統括役経由）: 同じ系統でも、非共通の部分の影響で適性の統計量が違う。種牡馬の単位で、系統の中の近い・遠いをベクトルで見る。
- 父の側は「その種牡馬の産駒の走り」、母父の側は「その種牡馬を母父に持つ馬の走り」で、同じ作りの集計をする。
- 次元（どれも擬似件数 K で、4 代上の父系の祖先（L4）→ 父の父 → 種牡馬 の順に上の段の値へ縮める。表示の大系統〔lines.py〕とは別）:
  dist 距離（log 距離への傾き）・surf 芝ダ（ダート − 芝）・wet_T 芝の道悪・wet_D ダートの道悪・pos 前の位置（序盤の位置の割合の逆）・
  ts トップスピード（上がり 3F のレースの中の順位）・gear ギアチェンジ（後半の加速が大きいレースでの 4 角から入線の位置の上がり方）・
  コースの 6 本（steep 急坂・small 小回り・straight 長い直線・left 左回り・yoshiba 洋芝・kyoto 京都）。
  群の次元（コース・道悪・芝ダ）は「その条件の平均 − 条件の外の平均」（芝ダの中）− 全体での同じ差を、調和の件数で縮める。
- 主の 6 本（地図・レーダー）: 距離・芝ダ・道悪（芝とダートの平均）・前の位置・TS・ギア。
- 尺度: 対象の種牡馬（産駒の走数 300 以上）の間の、走数の重み付きの z。
- 共通の部分 = 中系統（ss）の走数の重み付き平均。非共通の部分 = 種牡馬 − 中系統の平均（と − 大系統の平均）。
- 全期間（2016〜24）の集計で作る。as-of の版は、走りを「その時点より前」に絞って同じ stats・shrink を当てる（detailed_lines の assign・neighbors）。
  加速の z は前の年までの分布で作るので、絞った走りだけで値が決まる（z の尺度だけは版の全期間の値を使う）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.aptitude_group import common
from src.research.aptitude_group.vector import SMALL_VENUES, frame
from src.research.pedigree_lab import data
from src.research.pedigree_lab.explore5 import long_straight

K = 200.0
MIN_RUNS = 300            # 尺度（z）と群の中心を作る種牡馬（中核）
SHOW_RUNS = 50            # 地図・近傍・表に出す種牡馬（ユーザーの指定・11/27）。50〜299 走は中核の尺度で z にするだけ
PED5 = Path("/home/jovyan/work/keiba-vpn/data/local/horse_pedigree_5gen")
COURSE = ["steep", "small", "straight", "left", "yoshiba", "kyoto"]
DIMS = ["dist", "surf", "wet_T", "wet_D", "pos", "ts", "gear"] + COURSE
RADAR = ["dist", "wet", "pos", "ts", "gear", "course"]
DBANDS = [(0, 1400, "〜1400"), (1401, 1700, "1500〜1700"), (1701, 1900, "1800"), (1901, 2100, "2000"), (2101, 2499, "2200〜2400"), (2500, 9999, "2500〜")]


def runs() -> pd.DataFrame:
    F = frame()
    F["gc"] = 0.5 - F["fpct"]                            # レースの中で中心化した −分位（1 着 +0.5・最下位 −0.5）
    po, l3, laps = [], [], []
    for y in range(2016, 2025):
        r = pd.read_parquet(f"{common.TABLES}/{y}/race_result_flat.parquet", columns=["race_id", "horse_id", "passing_order", "last_3f"])
        po.append(r)
        laps.append(pd.read_parquet(f"{common.TABLES}/{y}/race_result_race.parquet", columns=["race_id", "lap_times", "direction"]))
    P = pd.concat(po, ignore_index=True)
    P["horse_id"] = P["horse_id"].astype(str)
    P = P.drop_duplicates(["race_id", "horse_id"]).rename(columns={"passing_order": "po2"})
    F = F.merge(P, on=["race_id", "horse_id"], how="left")
    first = pd.to_numeric(F["po2"].fillna("").str.split("-").str[0], errors="coerce")
    F["early"] = (first - 1) / (F["n_run"] - 1).clip(lower=1)
    F.loc[(F["early"] < 0) | (F["early"] > 1), "early"] = np.nan
    l3v = pd.to_numeric(F["last_3f"], errors="coerce")
    l3v[(l3v < 30) | (l3v > 50)] = np.nan
    F["ts"] = 0.5 - (l3v.groupby(F["race_id"]).rank(method="average") - 1) / (l3v.groupby(F["race_id"]).transform("count") - 1).clip(lower=1)
    F["gain"] = F["corner"] - F["fpct"]                  # 4 角から入線までに上げた位置の割合（正 = 上げた）
    L = pd.concat(laps, ignore_index=True).drop_duplicates("race_id")

    def accel(s):
        try:
            v = [float(x) for x in json.loads(s)]
        except Exception:
            return np.nan
        if len(v) < 6:
            return np.nan
        return float(np.mean(v[-5:-3]) - np.mean(v[-3:]))     # 後半の加速（秒・200m あたり。正 = 終いが速い）
    L["acc"] = L["lap_times"].map(accel)
    F = F.merge(L[["race_id", "acc", "direction"]], on="race_id", how="left")
    # 加速の z は、芝ダ × 距離帯ごとに前の年までのレースの平均と sd で（2016 年はその年。critic 2 回目の 6）
    key = (F["surf"] + "_" + pd.cut(F["distance"], [0, 1400, 1800, 2200, 9999]).astype(str)).to_numpy()
    yr = F["date"].dt.year.to_numpy()
    R = pd.DataFrame({"key": key, "yr": yr, "acc": F["acc"].to_numpy(), "race_id": F["race_id"].to_numpy()}).drop_duplicates("race_id")
    accz = np.full(len(F), np.nan)
    for y in np.unique(yr):
        ref = R[(R["yr"] < y)] if (R["yr"] < y).any() else R[R["yr"] == y]
        st = ref.groupby("key")["acc"].agg(["mean", "std"])
        m = yr == y
        mu = pd.Series(key[m]).map(st["mean"]).to_numpy()
        sd = pd.Series(key[m]).map(st["std"]).to_numpy()
        accz[m] = (F["acc"].to_numpy()[m] - mu) / sd
    F["acc_z"] = accz
    # 位置の上がり方は、4 角で後ろにいた馬ほど機械的に大きいので、レースの中で 4 角の位置に回帰した残りを使う（critic 2 回目の 8）
    c, gn = F["corner"], F["gain"]
    ok = c.notna() & gn.notna()
    rid = F["race_id"]
    cm = c.where(ok).groupby(rid).transform("mean")
    gm = gn.where(ok).groupby(rid).transform("mean")
    cov = ((c - cm) * (gn - gm)).where(ok).groupby(rid).transform("sum")
    var = ((c - cm) ** 2).where(ok).groupby(rid).transform("sum")
    F["gain_r"] = (gn - gm) - (cov / var.replace(0, np.nan)).fillna(0) * (c - cm)
    F["lx"] = np.log(F["distance"])
    ped = data.load_ped()[["horse_id", "anc5"]]
    want = {(3, 4): "bms_ss", (5, 16): "bms_L4"}
    rows = []
    for h, a in zip(ped["horse_id"], ped["anc5"].fillna("")):
        d = {}
        for t in a.split(";"):
            if t:
                g, p, i = t.split(":")
                if (int(g), int(p)) in want:
                    d[want[(int(g), int(p))]] = i
        rows.append((h, d.get("bms_ss", ""), d.get("bms_L4", "")))
    F = F.merge(pd.DataFrame(rows, columns=["horse_id", "bms_ss", "bms_L4"]), on="horse_id", how="left")
    return F


def flags(F: pd.DataFrame) -> dict[str, np.ndarray]:
    v, d, s, g = F["venue"].to_numpy(), F["distance"].to_numpy(), F["surf"].to_numpy(), F["going"].to_numpy()
    ct = F["course_type"].fillna("").to_numpy()
    return {"steep": np.isin(v, ["中山", "阪神", "中京"]),
            "small": np.isin(v, SMALL_VENUES) | ((v == "中山") & (ct != "外")),
            "straight": (s == "T") & np.array([long_straight(a, b) for a, b in zip(v, d)]),
            "left": F["direction"].fillna("").to_numpy() == "左",
            "yoshiba": (s == "T") & np.isin(v, ["札幌", "函館"]),
            "kyoto": v == "京都",
            "wet_T": (s == "T") & (g == "W"),
            "wet_D": (s == "D") & np.isin(g, ["S", "W"])}


def stats(F: pd.DataFrame, key: str) -> pd.DataFrame:
    """鍵（父・中系統・大系統など）ごとの、縮める前のずれの和と件数（次元ごとに ("d", 次元)・("n", 次元)）。
    F に列 w があれば、行の重み（サイアーライン・牝系のベクトルの血量の重み）として使う。無ければ 1。"""
    fl = flags(F)
    gc = F["gc"].to_numpy(float)
    wt = F["w"].to_numpy(float) if "w" in F.columns else np.ones(len(F))
    out = {}
    k = F[key].fillna("").to_numpy()
    surf = F["surf"].to_numpy()

    def wsum(x, by):
        return pd.Series(x).groupby(by).sum()

    def contrast(A: np.ndarray, by_surf: bool):
        """「その条件の平均 − 条件の外の平均」（芝ダの中で）− 全体での同じ差。件数は調和の件数 nA·nB/(nA+nB)（critic 2 回目の 2）。"""
        cell = surf if by_surf else np.full(len(F), "all")
        T = pd.DataFrame({"k": k, "c": cell, "A": A, "gw": gc * wt, "w": wt})
        g = T.groupby(["k", "c", "A"])[["gw", "w"]].sum().unstack("A", fill_value=0)
        g = g.reindex(columns=pd.MultiIndex.from_product([["gw", "w"], [False, True]]), fill_value=0)   # 条件の行が無いときも列をそろえる
        nA, nB, sA, sB = g[("w", True)], g[("w", False)], g[("gw", True)], g[("gw", False)]
        P = T.groupby(["c", "A"])[["gw", "w"]].sum()
        P = (P["gw"] / P["w"]).unstack("A").reindex(columns=[False, True])
        d0 = (P[True] - P[False]).fillna(0).reindex(g.index.get_level_values("c")).to_numpy()
        ne = (nA * nB / (nA + nB).replace(0, np.nan)).fillna(0)
        diff = (sA / nA.replace(0, np.nan) - sB / nB.replace(0, np.nan)).fillna(0) - d0
        dd = (ne * diff).groupby(level="k").sum()
        return dd, ne.groupby(level="k").sum()
    for nm in COURSE + ["wet_T", "wet_D"]:
        out[("d", nm)], out[("n", nm)] = contrast(fl[nm], True)
    out[("d", "surf")], out[("n", "surf")] = contrast(surf == "D", False)
    # 前の位置・TS（水準）
    for nm, col, sign in (("pos", "early", -1.0), ("ts", "ts", 1.0)):
        x = sign * F[col].to_numpy(float)
        ok = np.isfinite(x)
        mu = np.average(x[ok], weights=wt[ok])
        out[("d", nm)] = wsum(np.where(ok, (x - mu) * wt, 0.0), k)
        out[("n", nm)] = wsum(ok * wt, k)
    # 距離の傾き: 鍵 × 芝ダの中で中心化した log 距離への gc の傾き（critic 2 回目の 7）。件数は Σw·x² ÷ 全体（芝ダの中）の x の分散
    lx = F["lx"].to_numpy(float)
    ks = pd.Series(k).astype(str) + "|" + pd.Series(surf).astype(str)
    sw = pd.Series(wt).groupby(ks).transform("sum").to_numpy()
    xc = lx - pd.Series(lx * wt).groupby(ks).transform("sum").to_numpy() / sw
    gcc = gc - pd.Series(gc * wt).groupby(ks).transform("sum").to_numpy() / sw
    sws = pd.Series(wt).groupby(surf).transform("sum").to_numpy()
    xs = lx - pd.Series(lx * wt).groupby(surf).transform("sum").to_numpy() / sws
    vx = float(np.average(xs ** 2, weights=wt))          # 重み付き（w が無いときは前と同じ・critic 4 回目）
    out[("d", "dist")] = wsum(xc * gcc * wt, k) / vx
    out[("n", "dist")] = wsum(xc * xc * wt, k) / vx
    # ギア: 後半の加速（as-of の z）への、4 角の位置を除いた上がり方の傾き。どちらも鍵の中で中心化（critic 2 回目の 8）
    a, gn = F["acc_z"].to_numpy(float), F["gain_r"].to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(gn)
    a0, g0 = np.where(ok, a, 0.0), np.where(ok, gn, 0.0)
    w = ok * wt
    na = pd.Series(w).groupby(k).transform("sum").to_numpy()
    am = np.where(na > 0, pd.Series(a0 * w).groupby(k).transform("sum").to_numpy() / np.maximum(na, 1e-12), 0)
    gm = np.where(na > 0, pd.Series(g0 * w).groupby(k).transform("sum").to_numpy() / np.maximum(na, 1e-12), 0)
    ac, gc2 = (a0 - am) * (w > 0), (g0 - gm) * (w > 0)
    out[("d", "gear")] = wsum(ac * gc2 * w, k)
    out[("n", "gear")] = wsum(ac * ac * w, k)
    S = pd.DataFrame(out)
    S["runs"] = wsum(wt, k)
    return S.drop(index="", errors="ignore")


def shrink(F: pd.DataFrame, k_sire: str, k_mid: str, k_top: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """大系統 → 中系統 → 種牡馬 の順に 1 回ずつ縮めた値（種牡馬の行）と、種牡馬 → 中系統・大系統の対応。"""
    St, Sm, Ss = stats(F, k_top), stats(F, k_mid), stats(F, k_sire)
    link = F[[k_sire, k_mid, k_top]].replace("", np.nan).dropna().drop_duplicates(k_sire).set_index(k_sire)
    Ss = Ss.loc[Ss.index.intersection(link.index)]
    vals = {}
    mid_top = F[[k_mid, k_top]].replace("", np.nan).dropna().drop_duplicates(k_mid).set_index(k_mid)[k_top]
    for nm in DIMS:
        vt = St[("d", nm)] / (St[("n", nm)] + K)
        mid = link.loc[Ss.index, k_mid]
        top = link.loc[Ss.index, k_top]
        vm = (Sm[("d", nm)] + K * Sm.index.to_series().map(mid_top).map(vt).fillna(0.0)) / (Sm[("n", nm)] + K)
        vs = (Ss[("d", nm)].to_numpy() + K * mid.map(vm).fillna(top.map(vt)).fillna(0.0).to_numpy()) / (Ss[("n", nm)].to_numpy() + K)
        vals[nm] = vs
        vals[f"n_{nm}"] = Ss[("n", nm)].to_numpy()
    V = pd.DataFrame(vals, index=Ss.index)
    V["runs"] = Ss["runs"]
    V["mid"], V["top"] = link.loc[Ss.index, k_mid], link.loc[Ss.index, k_top]
    return V, link


def best_band(F: pd.DataFrame, key: str, ids: pd.Index) -> pd.Series:
    b = pd.Series(pd.cut(F["distance"], [x[0] - 1 for x in DBANDS] + [9999], labels=[x[2] for x in DBANDS]).astype(str))
    T = pd.DataFrame({"k": F[key], "b": b, "gc": F["gc"]})
    T = T[T["k"].isin(ids)]
    g = T.groupby(["k", "b"])["gc"].agg(["sum", "size"])
    g["v"] = g["sum"] / (g["size"] + 50)
    g = g[g["size"] >= 30]
    return g["v"].groupby(level=0).idxmax().map(lambda t: t[1])


def names(ids: set[str], F: pd.DataFrame) -> dict[str, str]:
    """血統の 5 代の JSON から、祖先の id の名前を拾う（各 id について子孫の 1 頭の JSON を読む）。"""
    out: dict[str, str] = {}
    need = set(ids)
    for col in ("sire", "ss", "L4", "ds", "bms_ss", "bms_L4"):
        if not need:
            break
        sub = F[F[col].isin(need)].drop_duplicates(col)
        for h in sub["horse_id"]:
            p = PED5 / h[:4] / f"{h}.json"
            if not p.exists():
                continue
            try:
                for a in json.loads(p.read_text(encoding="utf-8")).get("ancestors", []):
                    i = a.get("horse_id")
                    if i in need and a.get("name"):
                        out[i] = clean(a["name"])
            except Exception:
                continue
            need -= set(out)
    return out


def clean(s: str) -> str:
    m = re.match(r"^([^A-Za-z(（]+)[A-Za-z]", s)
    if m and re.search(r"[ぁ-んァ-ヶ一-龥]", m.group(1)):
        return m.group(1).strip()
    return re.sub(r"\(.*?\)$", "", s).strip()


def zscore(V: pd.DataFrame) -> pd.DataFrame:
    """中核（走数 MIN_RUNS 以上）の、走数の重み付きの平均と sd で z にする（50〜299 走の種牡馬も同じ尺度）。"""
    core = (V["runs"] >= MIN_RUNS).to_numpy()
    w = V["runs"].to_numpy(float)[core]
    Z = pd.DataFrame(index=V.index)
    for nm in DIMS:
        x = V[nm].to_numpy(float)
        mu = np.average(x[core], weights=w)
        sd = np.sqrt(np.average((x[core] - mu) ** 2, weights=w))
        Z[nm] = (x - mu) / (sd if sd > 0 else 1.0)
    Z["wet"] = (Z["wet_T"] + Z["wet_D"]) / 2
    Z["course"] = Z[COURSE].abs().mean(1)
    return Z


def uncertainty(V: pd.DataFrame) -> pd.Series:
    """主の 6 本の測りの不確かさ（縮めの割合 K/(n+K) の平方根の平均。0 = 確か・1 = 上の段の値のまま）。"""
    n = pd.DataFrame({d: (V["n_wet_T"] + V["n_wet_D"]) if d == "wet" else V[f"n_{d}"] for d in ["dist", "surf", "wet", "pos", "ts", "gear"]})
    return np.sqrt(K / (n + K)).mean(1)


def side(F: pd.DataFrame, k_sire: str, k_mid: str, k_top: str) -> dict:
    V, _ = shrink(F, k_sire, k_mid, k_top)
    V = V[V["runs"] >= SHOW_RUNS]
    Z = zscore(V)
    vec = Z[DIMS]
    w = V["runs"]
    w = w.where(V["runs"] >= MIN_RUNS, 0) + 1e-9          # 系統の平均は中核の種牡馬で作る（50〜299 走は引かれる側だけ）
    mid_mean = vec.mul(w, axis=0).groupby(V["mid"]).sum().div(w.groupby(V["mid"]).sum(), axis=0)
    top_mean = vec.mul(w, axis=0).groupby(V["top"]).sum().div(w.groupby(V["top"]).sum(), axis=0)
    nonc_mid = vec - mid_mean.loc[V["mid"]].to_numpy()
    nonc_top = vec - top_mean.loc[V["top"]].to_numpy()
    X = vec.to_numpy()
    Xc = X - X.mean(0)
    U, S, Wt = np.linalg.svd(Xc, full_matrices=False)
    pc = Xc @ Wt[:2].T
    Nc = nonc_mid.to_numpy()
    U2, S2, W2 = np.linalg.svd(Nc - Nc.mean(0), full_matrices=False)
    pcn = (Nc - Nc.mean(0)) @ W2[:2].T
    bb = best_band(F, k_sire, V.index)
    return {"V": V, "Z": Z, "vec": vec, "unc": uncertainty(V), "nonc_mid": nonc_mid, "nonc_top": nonc_top, "pc": pc, "pcn": pcn,
            "pc_var": (S[:2] ** 2 / (S ** 2).sum()).round(3).tolist(), "pc_load": Wt[:2].round(3).tolist(),
            "pcn_var": (S2[:2] ** 2 / (S2 ** 2).sum()).round(3).tolist(), "best": bb}


def to_json(sd: dict, nm: dict[str, str]) -> dict:
    V, Z = sd["V"], sd["Z"]
    rows = []
    for i, h in enumerate(V.index):
        rows.append({"id": h, "name": nm.get(h, h), "runs": int(V.loc[h, "runs"]),
                     "mid": V.loc[h, "mid"], "mid_name": nm.get(V.loc[h, "mid"], V.loc[h, "mid"]),
                     "top": V.loc[h, "top"], "top_name": nm.get(V.loc[h, "top"], V.loc[h, "top"]),
                     "best": str(sd["best"].get(h, "")), "core": bool(V.loc[h, "runs"] >= MIN_RUNS),
                     "unc": round(float(sd["unc"].loc[h]), 3),
                     "v": [round(float(Z.loc[h, d]), 3) for d in DIMS],
                     "radar": [round(float(Z.loc[h, d]), 3) for d in RADAR],
                     "nm": [round(float(x), 3) for x in sd["nonc_mid"].loc[h].to_numpy()],
                     "nt": [round(float(x), 3) for x in sd["nonc_top"].loc[h].to_numpy()],
                     "pc": [round(float(x), 3) for x in sd["pc"][i]], "pcn": [round(float(x), 3) for x in sd["pcn"][i]]})
    return {"rows": rows, "pc_var": sd["pc_var"], "pc_load": sd["pc_load"], "pcn_var": sd["pcn_var"]}


def boot_main(n_boot: int = 50, seed: int = 8080) -> None:
    """誤差の楕円と距離の区間のための、開催の単位のブートストラップ（縮めの作りも各回で同じ）。
    各回のベクトルを、全データの中核の平均と sd（sire_vectors.json を作った時と同じ尺度）で z にし、主の 6 本を保存する。"""
    common.open_token("select", "sire_vectors_boot_v1", "種牡馬の血統ベクトルの誤差（ブートストラップ・記述）")
    F = runs()
    F["L4"] = F["L4"].fillna("")
    out = {}
    for nm_side, keys in (("sire", ("sire", "ss", "L4")), ("bms", ("ds", "bms_ss", "bms_L4"))):
        V0, _ = shrink(F, *keys)
        V0 = V0[V0["runs"] >= SHOW_RUNS]
        core = V0["runs"] >= MIN_RUNS
        w = V0.loc[core, "runs"].to_numpy(float)
        mu = {d: float(np.average(V0.loc[core, d], weights=w)) for d in DIMS}
        sd = {d: float(np.sqrt(np.average((V0.loc[core, d] - mu[d]) ** 2, weights=w))) or 1.0 for d in DIMS}
        ids = list(V0.index)
        B = np.full((n_boot, len(ids), 6), np.nan)
        meets = F["meeting"].astype(str).to_numpy()
        idx = pd.Series(np.arange(len(F))).groupby(meets).indices
        mk = np.array(sorted(idx))
        for b in range(n_boot):
            rng = np.random.default_rng(seed + b)
            pick = rng.choice(len(mk), len(mk), replace=True)
            Fb = F.iloc[np.concatenate([idx[mk[i]] for i in pick])].reset_index(drop=True)
            Vb, _ = shrink(Fb, *keys)
            Vb = Vb.reindex(ids)
            Z = pd.DataFrame({d: (Vb[d] - mu[d]) / sd[d] for d in DIMS}, index=ids)
            Z["wet"] = (Z["wet_T"] + Z["wet_D"]) / 2
            B[b] = Z[["dist", "surf", "wet", "pos", "ts", "gear"]].to_numpy()
            print(nm_side, "boot", b + 1, flush=True)
        out[nm_side] = {"ids": ids, "B": B}
    np.savez_compressed(common.OUT / "sire_vectors_boot.npz", sire_ids=np.array(out["sire"]["ids"]), sire_B=out["sire"]["B"],
                        bms_ids=np.array(out["bms"]["ids"]), bms_B=out["bms"]["B"])
    data.finish_all(str(common.OUT / "sire_vectors_boot.npz"))


def main() -> None:
    common.open_token("select", "sire_vectors_v1", "種牡馬の血統ベクトル（記述・可視化。開発の行 2016〜24 の全期間の集計）")
    F = runs()
    F["L4"] = F["L4"].fillna("")
    sire = side(F, "sire", "ss", "L4")
    bms = side(F, "ds", "bms_ss", "bms_L4")
    ids = set()
    for sd in (sire, bms):
        ids |= set(sd["V"].index) | set(sd["V"]["mid"]) | set(sd["V"]["top"])
    nm = names(ids, F)
    out = {"made": pd.Timestamp.now(tz="Asia/Tokyo").strftime("%Y-%m-%d %H:%M"), "period": "2016〜2024", "dims": DIMS, "radar": RADAR,
           "min_runs": MIN_RUNS, "show_runs": SHOW_RUNS, "k": K, "sire": to_json(sire, nm), "bms": to_json(bms, nm)}
    p = common.OUT / "sire_vectors.json"
    p.write_text(json.dumps(out, ensure_ascii=False))
    print({"sire": len(sire["V"]), "bms": len(bms["V"]), "named": len(nm), "of": len(ids),
           "pc_var": sire["pc_var"], "bytes": p.stat().st_size})
    data.finish_all(str(p))


if __name__ == "__main__":
    import sys
    boot_main() if "--boot" in sys.argv else main()
