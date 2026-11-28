"""血統の札の族と、共通と特殊の境目（段 1-c・1-d）。着順を使わない。

    KEIBA_SESSION=day-pedigree OMP_NUM_THREADS=4 nice -n 19 python -m src.research.day_pedigree.features

族（数を見る前に固定。docs/operations/day_pedigree/log.md「段 1 の固定」）:
  S 牝系の特殊な祖先: 母の側の牡の祖先のうち、母父の線（2:2・3:4・4:8・5:16）を除いた位置（3:6・4:10/12/14・5:18〜30）。
    値は血量（2^−代 の和）× log(1/p)。0/1 は帯の計算にだけ使う。
  C 共通の血の濃さ: 帯の上の祖先の、同じ位置の血量の和（1 本）。
  M 中系統: 父の父（ss）と父の父の父（sss）の札。B 母父の系統: 母父の父系の 4 代上（4:8）の札。
境目（統括役の指定）:
  登場率 p(祖先, 芝ダ, 年) = 前の年のその芝ダの出走馬（馬の単位）のうち、その祖先をその位置に持つ割合。
  探す側（part 0）の 2017〜24 年の日ごとに、その日の最後のレースより前のレースの情報量
    I_day = Σ_r [Σ_{持つ} w − (Σ_{持つ} w)² / Σ w]（w = π5H(1 − π5H)・Harville・着順なし）
  を出し、検出力 = Φ(δ√I_day − z_α)。帯 = 日の検出力の中央値 ≥ 0.25（δ = log 2・片側 α 0.10）。
  帯が 5 頭未満なら検出力の上位 5 頭。N = Šidák の α = 1 − 0.9^{1/N} で N 番目の中央値が 0.15 以上の最大（5〜30）。
  帯の上（帯から外れ、p が帯の中の最大の p より大きい）は C へ。帯の下は使わない。
特殊の版（11/27 の帯の結果の後に追加。着順を見ていない）:
  上の規則では、検出力が p = 0.5 で最大なので、ノーザンダンサーなど共通の祖先が選ばれ、上で切れる祖先も無かった。
  そこで「共通」= 1 レースに平均 2 頭以上いる祖先（p × 平均の頭数 ≥ 2）とし、C の 1 本へまとめる。
  「特殊」= p が共通の線より下で、δ = log 3 の日の検出力の中央値が 0.25 以上の祖先のうち、検出力の上位 10 頭。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.research.day_pedigree import data as dd
from src.research.pedigree_lab import data as pl

S_POS = {(3, 6)} | {(4, p) for p in (10, 12, 14)} | {(5, p) for p in range(18, 32, 2)}
B_POS = (4, 8)
DELTAS = {"log1.3": np.log(1.3), "log1.5": np.log(1.5), "log2": np.log(2.0), "log3": np.log(3.0)}
MAIN_DELTA = "log2"
Z_A = norm.ppf(0.90)
BAND_POWER, N_POWER, N_MIN, N_MAX = 0.25, 0.15, 5, 30
COMMON_PER_RACE, SPECIAL_DELTA, SPECIAL_N = 2.0, "log3", 10


def parse_anc(ped: pd.DataFrame) -> pd.DataFrame:
    """馬 × 祖先（S の位置）の血量。同じ祖先が何度も出たら血量を足す。"""
    rows = []
    for h, a in zip(ped["horse_id"], ped["anc5"].fillna("")):
        for t in a.split(";"):
            if t:
                g, p, i = t.split(":")
                if (int(g), int(p)) in S_POS and i:
                    rows.append((h, i, 0.5 ** int(g)))
    L = pd.DataFrame(rows, columns=["horse_id", "anc", "bw"])
    return L.groupby(["horse_id", "anc"], as_index=False)["bw"].sum()


def label_table(ped: pd.DataFrame) -> pd.DataFrame:
    """馬ごとの M・B の札（ss・sss・母父の父系 4 代上）。"""
    b = []
    for a in ped["anc5"].fillna(""):
        d = dict(((int(t.split(":")[0]), int(t.split(":")[1])), t.split(":")[2]) for t in a.split(";") if t)
        b.append(d.get(B_POS, ""))
    return pd.DataFrame({"horse_id": ped["horse_id"], "M_ss": ped["ss"].fillna(""), "M_sss": ped["sss"].fillna(""), "B_ds4": b})


def prevalence(R: pd.DataFrame, L: pd.DataFrame, key: str = "anc") -> pd.DataFrame:
    """p(祖先, 芝ダ, 年) = 前の年のその芝ダの出走馬（馬の単位）のうち持つ割合。年 y の行には y−1 年の値を付ける。"""
    H = R[["horse_id", "surf", "year"]].drop_duplicates()
    n = H.groupby(["surf", "year"]).size().rename("n_h")
    J = H.merge(L[["horse_id", key]].drop_duplicates(), on="horse_id")
    c = J.groupby([key, "surf", "year"]).size().rename("c").reset_index().merge(n.reset_index(), on=["surf", "year"])
    c["p"] = c["c"] / c["n_h"]
    c["year"] = c["year"] + 1                     # as-of: 前の年の値を次の年に使う
    return c[[key, "surf", "year", "p"]]


def day_info(R: pd.DataFrame, L: pd.DataFrame, key: str = "anc") -> pd.DataFrame:
    """探す側の 2017〜24 年の日ごと × 祖先の I_day（その日の最後のレースより前）。着順を使わない。"""
    S = R[(R["part"] == 0) & (R["year"] >= 2017) & (R["k"] < R["k_last"])]
    S = S[["race_id", "day", "surf", "horse_id", "w"]]
    sw = S.groupby("race_id")["w"].sum().rename("sw")
    J = S.merge(L[["horse_id", key]].drop_duplicates(), on="horse_id")
    a = J.groupby(["race_id", key], as_index=False).agg(swx=("w", "sum"))
    a = a.merge(sw.reset_index(), on="race_id")
    a["I"] = a["swx"] - a["swx"] ** 2 / a["sw"]
    a = a.merge(S.drop_duplicates("race_id")[["race_id", "day", "surf"]], on="race_id")
    return a.groupby(["surf", "day", key], as_index=False)["I"].sum()


def power_table(R: pd.DataFrame, DI: pd.DataFrame, P: pd.DataFrame, key: str = "anc") -> pd.DataFrame:
    """祖先ごとの日の検出力の中央値（その祖先がいない日は I = 0 として数える）と、2017〜24 年の平均の p。"""
    days = R[(R["part"] == 0) & (R["year"] >= 2017)].groupby("surf")["day"].nunique()
    rows = []
    for (s, a), g in DI.groupby(["surf", key]):
        nd = int(days[s])
        I = np.concatenate([g["I"].to_numpy(), np.zeros(max(nd - len(g), 0))])
        rec = {"surf": s, key: a, "days_with": len(g), "days": nd, "I_med": float(np.median(I))}
        for dn, dv in DELTAS.items():
            rec[f"pow_{dn}"] = float(np.median(norm.cdf(dv * np.sqrt(I) - Z_A)))
        rows.append(rec)
    T = pd.DataFrame(rows)
    pm = P[(P["year"] >= 2017) & (P["year"] <= 2024)].groupby(["surf", key])["p"].mean().rename("p_mean").reset_index()
    return T.merge(pm, on=["surf", key], how="left").fillna({"p_mean": 0.0})


def choose(T: pd.DataFrame, delta: str) -> dict:
    """1 つの芝ダ・1 つの δ の帯と N。T は 1 つの芝ダの power_table。"""
    col = f"pow_{delta}"
    T = T.sort_values(col, ascending=False).reset_index(drop=True)
    band = T[T[col] >= BAND_POWER]
    fallback = len(band) < N_MIN
    if fallback:
        band = T.head(N_MIN)
    dv = DELTAS[delta]
    n_ok = 0
    for N in range(1, min(N_MAX, len(band)) + 1):
        za = norm.ppf(1 - (1 - 0.9 ** (1.0 / N)))
        I = band.loc[N - 1, "I_med"]
        if norm.cdf(dv * np.sqrt(I) - za) >= N_POWER:
            n_ok = N
    N = int(min(max(n_ok, N_MIN), len(band), N_MAX))
    p_hi = float(band["p_mean"].max()) if len(band) else 1.0
    common = T[(~T.index.isin(band.index)) & (T["p_mean"] > p_hi)]
    return {"band": band, "N": N, "fallback": bool(fallback), "p_lo": float(band["p_mean"].min()) if len(band) else None,
            "p_hi": p_hi, "common": common}


def choose_special(T: pd.DataFrame, n_bar: float) -> dict:
    """特殊の版。T は 1 つの芝ダの power_table。共通の線 p ≥ COMMON_PER_RACE / n_bar。"""
    p_common = COMMON_PER_RACE / n_bar
    col = f"pow_{SPECIAL_DELTA}"
    sp = T[(T["p_mean"] < p_common) & (T[col] >= BAND_POWER)].sort_values(col, ascending=False).head(SPECIAL_N)
    cm = T[T["p_mean"] >= p_common].sort_values("p_mean", ascending=False)
    return {"p_common": p_common, "special": sp, "common": cm}


def main() -> None:
    dd.open_token("select", "features_v1", "段 1 の帯と祖先の一覧（着順を使わない。取消の判定に着順の列を読むだけ）")
    R = dd.load_rows()
    print("行", len(R), "日", R["day"].nunique(), flush=True)
    R["pi5h"] = dd.harville_top(R)
    R["w"] = R["pi5h"] * (1 - R["pi5h"])
    ped = pl.load_ped()
    ped = ped[ped["horse_id"].isin(R["horse_id"].unique())]
    L = parse_anc(ped)
    LB = label_table(ped)
    out = {"S": {}, "M_ss": {}, "M_sss": {}, "B_ds4": {}}
    n_bar = R.drop_duplicates("race_id").groupby("surf")["n_run"].mean()      # 平均の頭数（9 頭以上のレース）
    keep = {}
    for fam in out:
        if fam == "S":
            Lf, key = L, "anc"
        else:
            Lf = LB[["horse_id", fam]].rename(columns={fam: "anc"})
            Lf = Lf[Lf["anc"] != ""]
            key = "anc"
        P = prevalence(R, Lf, key)
        DI = day_info(R, Lf, key)
        PT = power_table(R, DI, P, key)
        keep[fam] = PT
        for s in ("T", "D"):
            Ts = PT[PT["surf"] == s]
            res = {}
            for dn in DELTAS:
                c = choose(Ts, dn)
                res[dn] = {"band_size": int(len(c["band"])), "N": c["N"], "fallback": c["fallback"],
                           "p_lo": c["p_lo"], "p_hi": c["p_hi"], "common_size": int(len(c["common"]))}
            c = choose(Ts, MAIN_DELTA)
            res["chosen"] = c["band"].head(c["N"])[["anc", "p_mean", "I_med", f"pow_{MAIN_DELTA}", "days_with"]].to_dict("records")
            res["common"] = c["common"].sort_values("p_mean", ascending=False)[["anc", "p_mean"]].head(40).to_dict("records")
            cs = choose_special(Ts, n_bar[s])
            res["special_p_common"] = cs["p_common"]
            res["special"] = cs["special"][["anc", "p_mean", "I_med", f"pow_{SPECIAL_DELTA}", f"pow_{MAIN_DELTA}", "days_with"]].to_dict("records")
            res["common_by_count"] = cs["common"][["anc", "p_mean"]].to_dict("records")
            res["top_power"] = Ts.sort_values(f"pow_{MAIN_DELTA}", ascending=False).head(15)[["anc", "p_mean", f"pow_{MAIN_DELTA}"]].to_dict("records")
            out[fam][s] = res
            print(fam, s, {k: v for k, v in res.items() if k not in ("chosen", "common", "top_power", "special", "common_by_count")}, flush=True)
    ids = set()
    for fam in out:
        for s in ("T", "D"):
            for k in ("chosen", "common", "top_power", "special", "common_by_count"):
                ids |= {r["anc"] for r in out[fam][s][k]}
    from src.research.pedigree_lab.dirt_deep import names
    nm = names(ids)
    for fam in out:
        for s in ("T", "D"):
            for k in ("chosen", "common", "top_power", "special", "common_by_count"):
                for r in out[fam][s][k]:
                    r["name"] = nm.get(r["anc"], "")
    dd.OUT.mkdir(parents=True, exist_ok=True)
    pd.concat([v.assign(fam=k) for k, v in keep.items()]).to_parquet(dd.OUT / "power_table.parquet", index=False)
    path = dd.OUT / "features_band.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("書いた", path)
    dd.finish_all(str(path))


if __name__ == "__main__":
    main()
