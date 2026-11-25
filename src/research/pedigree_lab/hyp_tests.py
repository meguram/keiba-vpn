"""段 1: 事前に固定した仮説 H1〜H10 の検定（log.md の「事前に固定した仮説」）。

各仮説は「基準の列」を入れた条件付きロジット（レースの中の softmax・勝ち）に「調べる列」を足し、その係数 β を見る。
- 列はレースの中で平均を引き、全体の sd で割る（1 sd あたりの対数オッズ）。
- 区間は開催（race_id の先頭 8 桁）の単位のブートストラップ（200 回）。
- 探す側 2017〜2022、確かめる側 2023〜2024。仮説ごとの向きは事前に固定（期待 +）。
- クラス別（新馬・未勝利・1 勝・2 勝以上）・年齢別（2・3・4 歳以上）にも β を出し、効きが消える形を見る。
出力: OUT/hyp_tests.json と標準出力の表。費用 0 円。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.research.pedigree_lab import data

HYP = [  # (名前, 調べる列, 基準の列, 行の絞り込み)
    ("H1 父×芝ダ", ["sire_surf_top3"], ["sire_all_top3"], None),
    ("H2 父×距離帯", ["sire_sd_top3"], ["sire_surf_top3"], None),
    ("H2b 母父×距離帯", ["bms_sd_top3"], ["bms_all_top3", "sire_sd_top3"], None),
    ("H3 母父", ["bms_all_top3"], ["sire_all_top3"], None),
    ("H4 きょうだい（母の 1 頭 1 票）", ["dam_v_win"], ["sire_all_top3", "bms_all_top3"], None),
    ("H4b 全きょうだい（上乗せ）", ["full_v_win"], ["sire_all_top3", "bms_all_top3", "dam_v_win"], None),
    ("H5 父×馬場（重・不良）", ["sire_sg_top3"], ["sire_surf_top3"], ("going", "W")),
    ("H5s 父×馬場（稍重）", ["sire_sg_top3"], ["sire_surf_top3"], ("going", "S")),
    ("H6 父×年齢×芝ダ", ["sire_agesurf_top3"], ["sire_surf_top3"], None),
    ("H7 牝系（2 代母）", ["dd_v_win"], ["sire_all_top3", "bms_all_top3", "dam_v_win"], None),
    ("H8 父×場×芝ダ", ["sire_vs_top3"], ["sire_surf_top3"], None),
    ("H9 ニックス", ["nick_all_top3"], ["sire_all_top3", "bms_all_top3"], None),
    ("H10 近交係数", ["ib_F"], ["sire_all_top3", "bms_all_top3"], None),
    ("H10b SS のクロスの有無", ["ib_SS_on"], ["sire_all_top3", "bms_all_top3"], None),
]


HYP4 = [  # 第 4 版で足した列（2 体の段 1 の R1。数を見る前に向きを固定: wet_e +・rasurf +・quick +・surf_e +・dz −）
    ("H5' 父の道悪寄り（同じ産駒の中の差・重不良）", ["sire_wet_e"], ["sire_surf_top3"], ("going", "W")),
    ("H5's 同（稍重。予想 ≈0）", ["sire_wet_e"], ["sire_surf_top3"], ("going", "S")),
    ("H6' 父×レースの年齢条件×芝ダ", ["sire_rasurf_top3"], ["sire_surf_top3"], None),
    ("H12 母の産駒の 5 走以内の勝ち上がり", ["dam_v_quick"], ["sire_all_top3", "bms_all_top3", "dam_v_win"], None),
    ("H13 母父のダート寄り（ダートの行）", ["bms_surf_e"], ["sire_surf_top3", "bms_all_top3"], ("surf", "D")),
    ("H14 父の距離の幅で割ったずれ（予想 −）", ["sire_dz"], ["sire_sd_top3"], None),
]


def clogit(Xm: np.ndarray, y: np.ndarray, rid_codes: np.ndarray, n_iter: int = 25, w: np.ndarray | None = None) -> np.ndarray:
    """条件付きロジット（各レースの勝ち馬 1 頭）。ニュートン法。w はレースの重み（ブートストラップ用、行ごとに展開済み）。"""
    k = Xm.shape[1]
    b = np.zeros(k)
    if w is None:
        w = np.ones(len(y))
    for _ in range(n_iter):
        eta = Xm @ b
        mx = pd.Series(eta).groupby(rid_codes).transform("max").to_numpy()
        e = np.exp(eta - mx)
        den = np.bincount(rid_codes, e)
        p = e / den[rid_codes]
        # 勾配: Σ w (x_win − E[x])
        Ex = np.vstack([np.bincount(rid_codes, p * Xm[:, j]) for j in range(k)]).T
        g = (w * y) @ Xm - (np.bincount(rid_codes, w * y)[:, None] * Ex).sum(0)
        H = np.zeros((k, k))
        wr = np.bincount(rid_codes, w * y)
        for a in range(k):
            for c in range(a, k):
                Exy = np.bincount(rid_codes, p * Xm[:, a] * Xm[:, c])
                H[a, c] = H[c, a] = -np.sum(wr * (Exy - Ex[:, a] * Ex[:, c]))
        step = np.linalg.solve(H - 1e-9 * np.eye(k), g)
        b = b - step
        if np.max(np.abs(step)) < 1e-7:
            break
    return b


def prep(X: pd.DataFrame, cols: list[str]) -> np.ndarray:
    M = np.empty((len(X), len(cols)))
    rid = X["race_id"].to_numpy()
    for j, c in enumerate(cols):
        v = X[c].astype(float).fillna(X[c].astype(float).mean()).to_numpy()
        v = v - pd.Series(v).groupby(rid).transform("mean").to_numpy()
        sd = v.std()
        M[:, j] = v / (sd if sd > 0 else 1.0)
    return M


def boot(X: pd.DataFrame, cols: list[str], n_boot: int = 200, seed: int = 0) -> tuple[float, float, float]:
    rc = pd.factorize(X["race_id"])[0]
    y = X["win"].to_numpy(float)
    M = prep(X, cols)
    b0 = clogit(M, y, rc)[-1]
    mt = pd.factorize(X["meeting"])[0]
    nm = mt.max() + 1
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, nm, nm), minlength=nm).astype(float)
        bs.append(clogit(M, y, rc, n_iter=8, w=cnt[mt])[-1])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(b0), float(lo), float(hi)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v3")
    ap.add_argument("--mode", default="random", choices=["random", "forward"],
                    help="random: 開催の無作為 70/30（主。ユーザーの指示）/ forward: 2017〜22 と 2023〜24（参考）")
    ap.add_argument("--hyp", default="base", choices=["base", "v4"])
    a = ap.parse_args()
    X = pd.read_parquet(data.OUT / f"feat_{a.tag}.parquet")
    X = X[(X["date"].dt.year >= 2017) & (X["date"].dt.year <= 2024)].reset_index(drop=True)
    X["ib_SS_on"] = (X["ib_SS"] > 0).astype(float)
    X["clsg"] = X["cls"].map({"NEW": "1新馬", "MDN": "2未勝利", "C1": "3一勝"}).fillna("4二勝以上")
    X["ageg"] = X["ageb"].map({"2": "2歳", "3": "3歳"}).fillna("4歳以上")
    yr = X["date"].dt.year
    explore_mask = (X["split"] == 0) if a.mode == "random" else (yr <= 2022)
    out = []
    for name, test, basec, filt in (HYP if a.hyp == "base" else HYP4):
        cols = basec + test
        D = X if filt is None else X[X[filt[0]] == filt[1]]
        em = explore_mask.loc[D.index]
        ex, cf = D[em], D[~em]
        r = {"hyp": name, "test": test[0]}
        r["explore"] = boot(ex, cols, 100)
        r["confirm"] = boot(cf, cols, 100, seed=1)
        # 層ごと（探す側・点推定だけ）
        for g in ("clsg", "ageg"):
            r[g] = {}
            for lv, sub in ex.groupby(g):
                if sub["race_id"].nunique() < 300:
                    continue
                rc = pd.factorize(sub["race_id"])[0]
                r[g][lv] = round(float(clogit(prep(sub, cols), sub["win"].to_numpy(float), rc)[-1]), 3)
        out.append(r)
        e, c = r["explore"], r["confirm"]
        print(f"{name:28s} 探す {e[0]:+.3f} [{e[1]:+.3f},{e[2]:+.3f}]  確かめ {c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]  "
              f"クラス {r['clsg']}  年齢 {r['ageg']}", flush=True)
    (data.OUT / f"hyp_tests_{a.tag}_{a.mode}_{a.hyp}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
