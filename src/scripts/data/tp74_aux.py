"""tp-74 補助: 『今走に前へ行けたか』（普段の位置取りとの差）の残差と、事前の情報でそれを当てる力（AUC）。記述のみで検定はしない。

ホストで: python tp74_aux.py（tp74_pos.py と同じ場所。出力 <TP74POS_OUT>/aux.json）
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tp74_pos import DISC, OUT, load  # noqa: E402
m = load()
d = m[m["bp"].notna() & (m["pf_src"] == "pf") & m["chg3"].notna() & m["pp3"].notna()]
disc = d[d["year"].isin(DISC)]
out = {}
tab = disc.groupby(["bp", "pp3"])["chg3"].agg(fwd=lambda s: float((s == 0).mean()), back=lambda s: float((s == 2).mean()), n="size").reset_index()
out["move_by_band_pp3"] = tab.to_dict("records")
# 前へ行ける確率の事前予測（セル平均）の識別力
def auc(y, p):
    from scipy.stats import rankdata
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
tr = disc; te = d[d["year"] == 2025]
res = {}
for lab, keys in (("band", ["bp"]), ("band+pace予想", ["bp", "pp3"]), ("band+コース+ペース予想", ["bp", "c5", "pp3"]), ("band+コース+馬場+ペース予想", ["bp", "c5", "g3", "pp3"])):
    x = tr.dropna(subset=keys).copy(); x["f"] = (x["chg3"] == 0).astype(float); x["b"] = (x["chg3"] == 2).astype(float)
    g = x.groupby(keys)[["f", "b"]].agg(["sum", "count"])
    mu_f, mu_b = x["f"].mean(), x["b"].mean(); k = 100
    est = pd.DataFrame({"pf": (g[("f", "sum")] + k * mu_f) / (g[("f", "count")] + k), "pb": (g[("b", "sum")] + k * mu_b) / (g[("b", "count")] + k)}).reset_index()
    t = te.dropna(subset=keys).merge(est, on=keys, how="left").fillna({"pf": mu_f, "pb": mu_b})
    res[lab] = {"auc_forward": auc((t["chg3"] == 0).astype(int).to_numpy(), t["pf"].to_numpy()), "auc_back": auc((t["chg3"] == 2).astype(int).to_numpy(), t["pb"].to_numpy()), "n": int(len(t))}
out["move_auc_2025"] = res
# 実際の動きの影響（残差）: 前へ/同/後ろ
x = d[d["year"].isin(DISC)].groupby("chg3").agg(n=("r3", "size"), r3=("r3", "mean"), t3=("t3", "mean"), p=("p_top3", "mean")).reset_index()
out["chg3_resid"] = x.to_dict("records")
(OUT / "aux.json").write_text(json.dumps(out, ensure_ascii=False, default=float), encoding="utf-8")
print(json.dumps(out["move_auc_2025"], ensure_ascii=False)); print(x.to_string())
