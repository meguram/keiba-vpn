"""種牡馬の産駒の脚質（着順を引いた位置・上がり）の as-of の列（log.md「調べ L」と「流す前の直し」。開発の行・時間の順）。

    KEIBA_SESSION=pedigree-lab OMP_NUM_THREADS=6 nice -n 19 python -m src.research.pedigree_lab.sire_style

- 元: data/page_reference/tables/{2016..2024}/race_result_flat.parquet の passing_order・last_3f・finish_position（2025・26 年は開けない）。
  走った馬（finish_position > 0・上がり > 0）だけで順位を付け、分母は走った頭数 − 1（critic）。
- 各走の 3 つの値（どれも走った馬の中の順位の割合・0 が先頭・1 が最後）: sty_l3 = 上がりの順位、sty_f4 = 最後の通過順、
  sty_mv = 最後の通過順 − 最初の通過順（負なら位置を上げた）。
  強さと条件の分は、芝ダ × 距離の帯 × 着順の帯（10 段）ごとの、その日より前の平均を引いて除く（同じ着順の中で、上がりが速いか・前にいたか。
  「順位 − 着順」の引き算は平均への回帰で強さが逆向きに残るのでやめた。critic）。
- 列: 父・母父の産駒の、その日より前の走りの平均（擬似件数 200 走で 0 へ縮める）6 列（sty_*_sire・sty_*_bms）。
- (A) ＋ 6 列を時間の順（学習は 2017 年から・試す年は 2019〜24 年）で当て、行ごとの予測を sire_style_wf.parquet に残す。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.research.pedigree_lab import ablate, data
from src.research.pedigree_lab.data import asof

ROOT = Path("/home/jovyan/work/keiba-vpn/data/page_reference/tables")
VALS = ["sty_l3", "sty_f4", "sty_mv"]
NEW = [f"{v}_{k}" for k in ("sire", "bms") for v in VALS]
K = 200.0


def style() -> pd.DataFrame:
    F = pd.concat([pd.read_parquet(ROOT / str(y) / "race_result_flat.parquet",
                                   columns=["race_id", "horse_id", "passing_order", "last_3f", "finish_position"]) for y in range(2016, 2025)],
                  ignore_index=True)
    F["race_id"], F["horse_id"] = F["race_id"].astype(str), F["horse_id"].astype(str)
    fp = pd.to_numeric(F["finish_position"], errors="coerce")
    l3 = pd.to_numeric(F["last_3f"], errors="coerce")
    F = F[(fp > 0) & (l3 > 0)].copy()                                      # 走った馬だけ（取消・中止の 0 を除く。critic）
    F["_fp"], F["_l3"] = fp[F.index], l3[F.index]
    n = F.groupby("race_id")["race_id"].transform("size")
    den = (n - 1).where(n > 1)
    po = F["passing_order"].astype(str).str.split("-")
    F["_first"] = pd.to_numeric(po.str[0], errors="coerce")
    F["_last"] = pd.to_numeric(po.str[-1], errors="coerce")
    g = F.groupby("race_id")
    rf = (g["_fp"].rank(method="average") - 1) / den
    rl3 = (g["_l3"].rank(method="average") - 1) / den
    rfirst = (g["_first"].rank(method="average") - 1) / den              # 走った馬の中で順位を付け直す（critic）
    rlast = (g["_last"].rank(method="average") - 1) / den
    # 値は順位の割合そのもの。強さの分は、あとで「着順の帯」の中で中心を合わせて除く（引き算は平均への回帰で強さが逆向きに残る。critic）
    F["sty_l3"], F["sty_f4"], F["sty_mv"] = rl3, rlast, rlast - rfirst
    F["rfb"] = np.floor(rf.clip(0, 0.999) * 10).astype("Int64").astype(str)
    return F[["race_id", "horse_id", "rfb"] + VALS].drop_duplicates(["race_id", "horse_id"])


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "sire_style", "種牡馬の産駒の脚質の列の時間の順の比べ（尤度だけ・記述）"))
    X = pd.read_parquet(data.OUT / "feat_v4.parquet")
    X = X[X["date"] <= pd.Timestamp(data.DEV_END)].reset_index(drop=True)
    X = X.merge(style(), on=["race_id", "horse_id"], how="left")
    print("付く割合", {c: round(float(X[c].notna().mean()), 3) for c in VALS}, flush=True)
    E = X.assign(_k="1", sire=X["sire"].fillna("none"), ds=X["ds"].fillna("none"))
    agg = []
    for v in VALS:
        E[f"_{v}"], E[f"_{v}n"] = E[v].fillna(0.0), E[v].notna().astype(float)
        agg += [f"_{v}", f"_{v}n"]
    # 芝ダ × 距離の帯 × 着順の帯の、その日より前の平均で中心を合わせる（1 号・critic）
    E["rfb"] = E["rfb"].fillna("na")
    cs, _ = asof(E, ["_k", "surf", "dband", "rfb"], agg)                   # 芝ダ × 距離の帯 × 着順の帯（10 段）の中で中心を合わせる（critic）
    for i, v in enumerate(VALS):
        okc = cs[:, 2 * i + 1] > 30
        mu = np.where(okc, cs[:, 2 * i] / np.maximum(cs[:, 2 * i + 1], 1), 0.0)
        use_v = E[v].notna().to_numpy() & okc                              # 中心を引けない行は数えない（critic）
        E[f"_{v}"] = np.where(use_v, E[v].fillna(0.0) - mu, 0.0)
        E[f"_{v}n"] = use_v.astype(float)
    for key, nm in (("sire", "sire"), ("ds", "bms")):
        s, _ = asof(E, [key], agg)
        none = (E[key] == "none").to_numpy()
        s[none] = 0.0
        for i, v in enumerate(VALS):
            X[f"{v}_{nm}"] = s[:, 2 * i] / (s[:, 2 * i + 1] + K)
    print("mean", X[NEW].mean().round(4).to_dict(), "sd", X[NEW].std().round(4).to_dict(), flush=True)
    print("父の勝ち率の列との相関", {c: round(float(X[c].corr(X["sire_all_win"])), 3) for c in NEW if "sire_all_win" in X}, flush=True)
    X = X[X["date"].dt.year >= 2017].reset_index(drop=True)
    from src.research.pedigree_lab.walkforward import fit_year
    use = list(dict.fromkeys([c for c in ablate.col_sets(X)["pure"] if not c.startswith("ib_")] + NEW))
    yr = X["date"].dt.year.to_numpy()
    p = np.full(len(X), np.nan)
    for t in range(2019, 2025):
        p[yr == t] = fit_year(X, use, "win", t, 2000)
        print(t, flush=True)
    te = yr >= 2019
    o = X.loc[te, ["race_id", "horse_id"]].copy()
    o["p_TB"] = p[te]
    path = data.OUT / "sire_style_wf.parquet"
    o.to_parquet(path, index=False)
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
