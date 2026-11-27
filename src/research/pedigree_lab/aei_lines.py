"""賞金で見た血の価値を、小系統・大系統の単位で見る（log.md「調べ Y」。記述・持続の表だけ）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.aei_lines

- 馬ごとの値 b: aei_cols の b（本賞金 ÷ 前の年 × 年齢の区分の 1 走あたり）の馬ごとの平均を、全体の平均で割った値（1 頭 1 票）。
- 単位: 父・小系統（父の父）・大系統（父の父の父）。母父の側は 母父・母父の父・母父の父の父（血統表の anc5 の 3:4・4:8）。
  組: 父の小系統 × 母父の小系統（母父の父）・父の大系統 × 母父の大系統。組は生の平均と、足し算の見込み（父側 + 母父側 − 1）を引いた残差の 2 つ。
- 持続: 生年 2014〜16 年・2017〜20 年の 2 期で、両方に 20 頭以上いる単位の平均の Spearman。
  帰無は、単位の札を生年の中で馬の間で入れ替える（200 回・95% 点）。母の id で分けた版（同じ母の子は同じ半分）も出す。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.research.pedigree_lab import data

MIN_H = 20


def horses() -> pd.DataFrame:
    X = pd.read_parquet(data.OUT / "feat_v4.parquet", columns=["race_id", "horse_id", "date", "split", "sire", "ss", "sss", "ds", "dam"])
    C = pd.read_parquet(data.OUT / "aei_cols.parquet", columns=["race_id", "horse_id", "b"])
    T = X.merge(C, on=["race_id", "horse_id"])
    T = T[T["split"].isin([0, 1]) & (T["date"] >= "2018-01-01") & (T["date"] <= data.DEV_END) & T["b"].notna()]
    H = T.groupby("horse_id").agg(b=("b", "mean"), sire=("sire", "first"), ss=("ss", "first"), sss=("sss", "first"),
                                  ds=("ds", "first"), dam=("dam", "first")).reset_index()
    ped = data.load_ped()[["horse_id", "anc5"]]
    a = ped["anc5"].fillna("")
    ped["ms"] = a.str.extract(r"(?:^|;)3:4:([^;]+)")[0]
    ped["mss"] = a.str.extract(r"(?:^|;)4:8:([^;]+)")[0]
    H = H.merge(ped[["horse_id", "ms", "mss"]], on="horse_id", how="left")
    H["by"] = H["horse_id"].str[:4].astype(int)
    H["b"] = H["b"] / H["b"].mean()
    for c in ("sire", "ss", "sss", "ds", "ms", "mss", "dam"):
        H[c] = H[c].fillna("")
    return H


def unit_means(H: pd.DataFrame, key: np.ndarray) -> pd.DataFrame:
    g = pd.DataFrame({"k": key, "b": H["b"].to_numpy()}).groupby("k")["b"].agg(["mean", "size"])
    return g[g.index != ""]


def persist(H: pd.DataFrame, key: np.ndarray, per: np.ndarray, resid_of: tuple | None = None) -> tuple[float, int]:
    out = []
    for p_ in (1, 2):
        m = per == p_
        P = H[m]
        u = unit_means(P, key[m])
        if resid_of is not None:
            ka, kb = resid_of
            va = unit_means(P, ka[m])["mean"]; vb = unit_means(P, kb[m])["mean"]
            parts = pd.Series(u.index).str.split("|", expand=True)
            u = u.assign(mean=u["mean"].to_numpy() - (parts[0].map(va).to_numpy() + parts[1].map(vb).to_numpy() - 1.0))
        out.append(u[u["size"] >= MIN_H]["mean"])
    k = out[0].index.intersection(out[1].index)
    if len(k) < 3:
        return np.nan, len(k)
    return float(spearmanr(out[0].loc[k], out[1].loc[k])[0]), int(len(k))


def main() -> None:
    data.TOKENS.append(data._guard(("2016-01-01", "2024-12-31"), "select", "aei_lines", "賞金の価値の小系統・大系統の持続（記述）"))
    H = horses()
    H = H[H["by"].between(2014, 2020)].reset_index(drop=True)
    per_by = np.where(H["by"] <= 2016, 1, 2)
    dl = H["dam"].str[-1].fillna("0").map(lambda c: int(c, 16) if c in "0123456789abcdef" else 0).to_numpy()
    per_dam = 1 + (dl % 2)
    keys = {"父": H["sire"], "父の小系統（父の父）": H["ss"], "父の大系統（父の父の父）": H["sss"],
            "母父": H["ds"], "母父の小系統（母父の父）": H["ms"], "母父の大系統（母父の父の父）": H["mss"]}
    combos = {"組: 小系統 × 小系統": ("父の小系統（父の父）", "母父の小系統（母父の父）"),
              "組: 大系統 × 大系統": ("父の大系統（父の父の父）", "母父の大系統（母父の父の父）")}
    rng = np.random.default_rng(43)
    yb = H["by"].to_numpy()
    res = {"馬": int(len(H)), "単位ごと": {}}

    def run(name, key, resid_of=None):
        key = key.to_numpy().astype(str) if hasattr(key, "to_numpy") else key
        r_by, n_by = persist(H, key, per_by, resid_of)
        r_dm, n_dm = persist(H, key, per_dam, resid_of)
        nul = []
        for _ in range(200):
            kp = key.copy()
            for y in np.unique(yb):
                ii = np.flatnonzero(yb == y)
                kp[ii] = key[rng.permutation(ii)]
            # 注意: 組の残差の帰無は、組の札だけを入れ替え、両側の札（resid_of）は入れ替えていない。
            # そのため残差が両側の値の写しになり、帰無が高く出る（作り誤り・使えない。log.md 調べ Y・critic）
            nul.append(persist(H, kp, per_by, resid_of)[0])
        u = unit_means(H, key)
        top = u[u["size"] >= 2 * MIN_H].sort_values("mean", ascending=False).head(8)
        res["単位ごと"][name] = {"生年の 2 期の ρ": round(r_by, 3), "単位の数": n_by, "帰無の 95% 点": round(float(np.nanpercentile(nul, 95)), 3),
                               "母の id で分けた ρ": round(r_dm, 3), "母の id で分けた単位の数": n_dm,
                               "上位（40 頭以上）": [[k, round(float(v), 3), int(n)] for k, (v, n) in top.iterrows()]}
        print(name, json.dumps({k: v for k, v in res["単位ごと"][name].items() if k != "上位（40 頭以上）"}, ensure_ascii=False), flush=True)

    for nm, k in keys.items():
        run(nm, k)
    for nm, (a, b) in combos.items():
        ka, kb = keys[a].to_numpy().astype(str), keys[b].to_numpy().astype(str)
        kc = np.where((ka != "") & (kb != ""), np.char.add(np.char.add(ka, "|"), kb), "")
        run(nm + "（生の平均）", kc)
        run(nm + "（足し算の見込みを引いた残差）", kc, (ka, kb))
    path = data.OUT / "aei_lines.json"
    path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
