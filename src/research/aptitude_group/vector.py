"""適性のベクトル（段 1-a。規則・as-of・手作業の表なし。開発の行 2016〜24 だけ）。

    KEIBA_SESSION=aptitude-group OMP_NUM_THREADS=4 nice -n 19 python -m src.research.aptitude_group.vector [--persist]

次元（数を見る前に固定。docs/operations/aptitude_group/log.md「段 1 の固定」）:
  揺れの次元（組の近さに使う）: pace_h・pace_s（過去のレースのペースの 3 分位。2017 年以降のレースだけで作る）、
    wet（芝は重・不良、ダートは稍重〜不良）、front（4 角の位置の −分位の水準）。どれも今回の芝ダの鍵で引く。
  平均の次元（1 頭の点 s に使う）: dirt・dist（2000m 以上 − 1400m 以下）・steep・small・straight。
  コースの型（段 1-d）: stamina（2500m 以上）・sprint_T（芝 1200m 以下）・kyoto・yoshiba（small・steep は上と同じ列）。
1 頭の値: ½·父 ＋ ¼·母父 ＋ ¼·牝系 3〜5 代（血量 2^−代 の重みの平均）。欠けは 0（和を 1 に直さない）。
  父は、群 A のずれの和を 1 回だけ縮める: 大系統（ssss）→ 中系統（ss）→ 父 の順に、擬似件数 200 で上の段へ。
  母父・牝系の祖先は 0 へ 1 回だけ縮める。ずれの作りは pedigree_lab.aptitude.apt_by と同じ（群 A の −分位の平均 − 全条件の平均 − 全体の同じ差）。
  front は群 A を持たない水準の次元で、ずれ = 4 角の位置の −分位の和 − 件数 × 全体の as-of の平均。
尺度: 次元ごとに、前の年まで（2017 年以降）の行の sd で割る。2016 年は慣らしの年（組の検定に使わない）。
費用 0 円（手元の parquet だけ）。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.research.aptitude_group import common
from src.research.pedigree_lab import data
from src.research.pedigree_lab.aptitude import maternal_long, pace_labels
from src.research.pedigree_lab.data import asof
from src.research.pedigree_lab.explore5 import long_straight
from src.research.pedigree_lab.lineage import line_ids

K = 200.0
SHOCK = ["pace_h", "pace_s", "wet", "front"]
MEAN = ["dirt", "dist", "steep", "small", "straight"]
COURSE = ["stamina", "sprint_T", "kyoto", "yoshiba", "small", "steep"]
SMALL_VENUES = ("札幌", "函館", "福島", "小倉")


def frame() -> pd.DataFrame:
    cols = ["race_id", "horse_id", "meeting", "date", "win", "fpct", "odds", "popularity", "horse_number", "finish_position",
            "n_run", "sire", "ss", "ds", "surf", "venue", "distance", "going", "course_type"]
    F = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=cols)
    F = F[F["date"] <= data.DEV_END].reset_index(drop=True)
    F = F.merge(line_ids()[["horse_id", "L4"]], on="horse_id", how="left")
    F["L4"] = F["L4"].fillna("")
    po = []
    for y in range(2016, 2025):
        r = pd.read_parquet(f"{common.TABLES}/{y}/race_result_flat.parquet", columns=["race_id", "horse_id", "passing_order"])
        po.append(r)
    P = pd.concat(po, ignore_index=True)
    P["horse_id"] = P["horse_id"].astype(str)
    P = P.drop_duplicates(["race_id", "horse_id"])
    F = F.merge(P, on=["race_id", "horse_id"], how="left")
    last = pd.to_numeric(F["passing_order"].fillna("").str.split("-").str[-1], errors="coerce")
    F["corner"] = (last - 1) / (F["n_run"] - 1).clip(lower=1)          # 0 = 4 角で先頭、1 = 最後方
    F.loc[(F["corner"] < 0) | (F["corner"] > 1), "corner"] = np.nan
    F = F.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)
    return F


def group_a(F: pd.DataFrame, what: str, pace_t: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    v, d, s = F["venue"].to_numpy(), F["distance"].to_numpy(), F["surf"].to_numpy()
    turf, dirt = s == "T", s == "D"
    g = F["going"].to_numpy()
    if what in ("pace_h", "pace_s"):
        yr = F["date"].dt.year.to_numpy()
        return (pace_t == (0 if what == "pace_h" else 2)) & (yr >= 2017), ["surf"]
    if what == "wet":
        return (turf & (g == "W")) | (dirt & np.isin(g, ["S", "W"])), ["surf"]
    if what == "dirt":
        return dirt, []
    if what == "dist_long":
        return d >= 2000, ["surf"]
    if what == "dist_short":
        return d <= 1400, ["surf"]
    if what == "steep":
        return np.isin(v, ["中山", "阪神", "中京"]), ["surf"]
    if what == "small":
        ct = F["course_type"].fillna("").to_numpy()
        return np.isin(v, SMALL_VENUES) | ((v == "中山") & (ct != "外")), ["surf"]
    if what == "straight":
        return turf & np.array([long_straight(a, b) for a, b in zip(v, d)]), ["surf"]
    if what == "stamina":
        return d >= 2500, ["surf"]
    if what == "sprint_T":
        return turf & (d <= 1200), ["surf"]
    if what == "kyoto":
        return v == "京都", ["surf"]
    if what == "yoshiba":
        return turf & np.isin(v, ["札幌", "函館"]), ["surf"]
    raise ValueError(what)


def _sums(D: pd.DataFrame, A: np.ndarray, keys: list[str], g: np.ndarray, w: np.ndarray) -> np.ndarray:
    ok = np.isfinite(g) & (w > 0)
    g0, w0, A = np.where(ok, g, 0.0), np.where(ok, w, 0.0), A.astype(float)
    E = D[["date", "horse_id"] + keys].assign(_a=g0 * A * w0, _na=A * w0, _t=g0 * w0, _nt=w0)
    s, _ = asof(E, keys, ["_a", "_na", "_t", "_nt"])
    return s


def base(F: pd.DataFrame, A: np.ndarray, keys: list[str], g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """全体（馬の行）での as-of の群 A の平均と全条件の平均（鍵に芝ダがあれば芝ダごと）。"""
    s = _sums(F.assign(_k="1"), A, ["_k"] + [k for k in keys if k == "surf"], g, np.ones(len(F)))
    return s[:, 0] / np.maximum(s[:, 1], 1e-9), s[:, 2] / np.maximum(s[:, 3], 1e-9)


def dev(D: pd.DataFrame, A: np.ndarray, keys: list[str], g: np.ndarray, b: tuple[np.ndarray, np.ndarray], level: bool,
        w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """縮める前のずれの和と件数。level=False は群 A のずれ（apt_by と同じ）、True は水準（front）。"""
    s = _sums(D, A, keys, g, np.ones(len(D)) if w is None else w)
    mA0, mT0 = b
    if level:
        return s[:, 2] - s[:, 3] * mT0, s[:, 3]
    rawT = np.where(s[:, 3] > 0, s[:, 2] / np.maximum(s[:, 3], 1e-9), 0.0)
    return s[:, 0] - s[:, 1] * rawT - s[:, 1] * (mA0 - mT0), s[:, 1]


def horse_value(F: pd.DataFrame, L: pd.DataFrame, A: np.ndarray, keys: list[str], g: np.ndarray, level: bool) -> dict[str, np.ndarray]:
    b = base(F, A, keys, g)
    d4, n4 = dev(F, A, ["L4"] + keys, g, b, level)
    d2, n2 = dev(F, A, ["ss"] + keys, g, b, level)
    ds_, ns = dev(F, A, ["sire"] + keys, g, b, level)
    v4 = d4 / (n4 + K)
    v2 = (d2 + K * v4) / (n2 + K)
    vs = (ds_ + K * v2) / (ns + K)
    db, nb = dev(F, A, ["ds"] + keys, g, b, level)
    vb = db / (nb + K)
    row = L["row"].to_numpy()
    dm, nm = dev(L, A[row], ["anc"] + keys, g[row], (b[0][row], b[1][row]), level)
    wt = L["bw"].to_numpy()
    agg = pd.DataFrame({"row": row, "wd": wt * dm / (nm + K), "w": wt}).groupby("row")[["wd", "w"]].sum()
    vm = (agg["wd"] / agg["w"]).reindex(np.arange(len(F))).fillna(0.0).to_numpy()
    return {"sire": vs, "bms": vb, "mat": vm, "all": 0.5 * vs + 0.25 * vb + 0.25 * vm}


def scale_asof(F: pd.DataFrame, x: np.ndarray) -> np.ndarray:
    """前の年まで（2017 年以降の行だけ）の sd で割る。2016 年は履歴の慣らしの年で、尺度の元に入れない（critic の 2）。
    2016・2017 年は、その年の行の sd で割る。組の検定は 2017 年以降のレースだけを使う。"""
    yr = F["date"].dt.year.to_numpy()
    out = np.empty_like(x)
    for y in np.unique(yr):
        m = yr == y
        prior = (yr < y) & (yr >= 2017)
        ref = x[prior] if prior.any() else x[m]
        sd = float(np.std(ref))
        out[m] = x[m] / (sd if sd > 0 else 1.0)
    return out


def pre_race(F: pd.DataFrame) -> pd.DataFrame:
    """発走前に分かる馬の値: 前 3 走の 4 角の位置の平均（脚質）と前走の −分位（陰性の対照）。"""
    H = F[["horse_id", "date", "corner", "fpct"]].sort_values(["horse_id", "date"], kind="stable")
    g = H.groupby("horse_id")
    style = g["corner"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    prev = g["fpct"].shift(1)
    return pd.DataFrame({"style": style, "prev_fpct": prev}, index=H.index).reindex(F.index)


def persistence(F: pd.DataFrame, A: np.ndarray, g: np.ndarray, level: bool, n_null: int = 20) -> dict:
    """父の「群 A の平均 − 全体の平均」（level は水準 − 全体）を 2016〜19 と 2020〜24 で別々に作った相関。帰無は生年の中の父の入れ替え。"""
    per = np.where(F["date"].dt.year <= 2019, 1, 2)
    ok = np.isfinite(g)

    def table(sire):
        T = pd.DataFrame({"sire": sire, "per": per, "g": g, "A": A})[ok]
        allm = T.groupby(["sire", "per"])["g"].agg(["mean", "size"])
        if level:
            J = allm[allm["size"] >= 100]
            mu = T.groupby("per")["g"].mean()
            out = J["mean"] - mu.reindex(J.index.get_level_values("per")).to_numpy()
        else:
            a = T[T["A"]].groupby(["sire", "per"])["g"].agg(["mean", "size"])
            J = a.join(allm["mean"].rename("all"), how="inner")
            J = J[J["size"] >= 100]
            out = J["mean"] - J["all"]
        return out.unstack("per").dropna()
    W = table(F["sire"].to_numpy())
    if len(W) < 10 or W.shape[1] < 2:
        return {"sires": int(len(W))}
    r = float(W[1].corr(W[2]))
    H = F.drop_duplicates("horse_id")[["horse_id", "sire"]].reset_index(drop=True)
    H["by"] = H["horse_id"].str[:4]
    nulls = []
    for k in range(n_null):
        rng = np.random.default_rng(900 + k)
        perm = np.arange(len(H))
        for ii in H.groupby("by").indices.values():
            perm[ii] = rng.permutation(ii)
        fake = dict(zip(H["horse_id"], H["sire"].to_numpy()[perm]))
        Wn = table(F["horse_id"].map(fake).to_numpy())
        if len(Wn) >= 10 and Wn.shape[1] == 2:
            nulls.append(float(Wn[1].corr(Wn[2])))
    return {"sires": int(len(W)), "corr": round(r, 3), "null_q95": round(float(np.percentile(nulls, 95)), 3) if nulls else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persist", action="store_true", help="揺れの次元の父の 2 期の持続（帰無 20 回）も出す")
    a = ap.parse_args()
    common.open_token("select", "vector_v1", "段 1-a 適性のベクトル（as-of・開発の行 2016〜24。組の検定の前）")
    F = frame()
    pace_t = pace_labels(F)
    L = maternal_long(F, drop_common=False, min_gen=3)
    gf = -F["fpct"].to_numpy(float)
    gc = -F["corner"].to_numpy(float)
    out = F[["race_id", "horse_id", "meeting", "date", "odds", "popularity", "horse_number", "finish_position", "n_run",
             "sire", "ss", "surf", "venue", "distance", "going", "course_type"]].copy()
    parts = {}
    report: dict = {"rows": int(len(F)), "corner_missing": round(float(F["corner"].isna().mean()), 4)}
    for what in ["pace_h", "pace_s", "wet", "front", "dirt", "dist_long", "dist_short", "steep", "small", "straight",
                 "stamina", "sprint_T", "kyoto", "yoshiba"]:
        if what == "front":
            A, keys, g, level = np.ones(len(F), bool), ["surf"], gc, True
        else:
            A, keys = group_a(F, what, pace_t)
            g, level = gf, False
        hv = horse_value(F, L, A, keys, g, level)
        parts[what] = hv["all"]
        report[f"{what}_A_rows"] = int(A.sum())
        if a.persist and what in SHOCK:
            report[f"{what}_persist"] = persistence(F, A, g, level)
        print(what, report.get(f"{what}_persist", ""), flush=True)
    parts["dist"] = parts.pop("dist_long") - parts.pop("dist_short")
    for k, x in parts.items():
        out[f"a_{k}"] = scale_asof(F, x)
        out[f"raw_{k}"] = x
    pr = pre_race(F)
    out["style"], out["prev_fpct"] = pr["style"].to_numpy(), pr["prev_fpct"].to_numpy()
    cols = [f"a_{k}" for k in parts]
    report["corr"] = out[cols].corr().round(2).to_dict()
    out.to_parquet(common.OUT / "vec_v1.parquet", index=False)
    (common.OUT / "vector_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "corr"}, ensure_ascii=False, indent=1, default=str))
    data.finish_all(str(common.OUT / "vector_v1.json"))


if __name__ == "__main__":
    main()
