"""段 1 の第 4 版: 第 2 版（features2。レースの年齢条件の鍵を足した）に、2 体の R1（段 1 の結果へのレビュー）の列を足す。

足す列（どれも as-of・自分を引く・血統の役ごと。馬の年齢・性は使わない）
- 同じ産駒の中の差（能力を消した適性）: 各走の分位から「その馬のそれまでの走りの分位の平均」を引いた残り e を作り、
  役（父・母父・父父）× 条件（芝ダ × 馬場・芝ダ）ごとに、ほかの産駒の e の平均を縮める。
  道悪: 芝ダごとに「重・不良の e − 良の e」（H5 の作り直し 1 回）。ダート寄り: 「ダートの e − 芝の e」。
  縮小の強さは e の分散で決めた固定の 200 走（段 1 の k の目安。選び直さない）。
- 母の産駒の勝ち上がりの速さ（1 頭 1 票）: 5 走を終えた（または 5 走以内に勝った）上のきょうだいのうち、5 走以内に勝った割合。
- 父の産駒の距離の幅: 3 着内の距離の平均と標準偏差（as-of）から、|今回 − 平均| ÷ 標準偏差。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.pedigree_lab import features2
from src.research.pedigree_lab.data import asof

K_RES = 200.0


def own_resid(r: pd.DataFrame) -> np.ndarray:
    """各走の分位 −（その馬のそれまでの走りの分位の平均）。初出走は NaN。分位は小さいほど良いので、符号を返して「大きいほど良い」にする。"""
    o = r.sort_values(["horse_id", "date"], kind="stable")
    g = o.groupby("horse_id")["fpct"]
    prev_mean = (g.cumsum() - o["fpct"]) / g.cumcount().replace(0, np.nan)
    e = -(o["fpct"] - prev_mean)
    return e.reindex(r.index).to_numpy()


def resid_apt(r: pd.DataFrame) -> pd.DataFrame:
    r = r.copy()
    r["_e"] = own_resid(r)
    has = ~np.isnan(r["_e"].to_numpy())
    r["_e0"] = np.nan_to_num(r["_e"].to_numpy())
    r["_one"] = has.astype(float)
    out = {}
    for role, cols in {"sire": ["sire"], "bms": ["ds"], "ss": ["ss"]}.items():
        # 芝ダ × 馬場（G/S/W）の e の平均（縮めて 0 へ）
        s, _ = asof(r, cols + ["surf", "going"], ["_e0", "_one"])
        m_sg = s[:, 0] / (s[:, 1] + K_RES)
        # 同じ役・同じ芝ダの、良の e の平均（比べの元）
        rg = r.assign(going="G")
        sG, _ = asof(rg.assign(_e0=np.where(r["going"] == "G", r["_e0"], 0.0), _one=np.where(r["going"] == "G", r["_one"], 0.0)),
                     cols + ["surf"], ["_e0", "_one"])
        m_g = sG[:, 0] / (sG[:, 1] + K_RES)
        out[f"{role}_wet_e"] = np.where(r["going"].to_numpy() == "G", 0.0, m_sg - m_g).astype("float32")
        # 芝ダの e の差（ダート寄り）: 今回の芝ダの e − もう一方の芝ダの e
        s_s, _ = asof(r, cols + ["surf"], ["_e0", "_one"])
        m_s = s_s[:, 0] / (s_s[:, 1] + K_RES)
        other = r.assign(surf=np.where(r["surf"] == "T", "D", "T"))
        # もう一方の芝ダの値は「同じ役・もう一方の芝ダの鍵」を、自分の行の日付で引く
        s_o, _ = asof(pd.concat([r, other.assign(_e0=0.0, _one=0.0)], ignore_index=True), cols + ["surf"], ["_e0", "_one"])
        s_o = s_o[len(r):]
        m_o = s_o[:, 0] / (s_o[:, 1] + K_RES)
        out[f"{role}_surf_e"] = (m_s - m_o).astype("float32")
        out[f"{role}_res_n"] = np.log1p(np.maximum(s_s[:, 1], 0)).astype("float32")
    return pd.DataFrame(out, index=r.index)


def dam_quick(r: pd.DataFrame) -> pd.DataFrame:
    """母の産駒の 5 走以内の勝ち上がり（1 頭 1 票）。状態が決まった日 = 5 走以内に勝った日か、5 走目の日。"""
    D0 = np.datetime64("2010-01-01")
    o = r.sort_values(["horse_id", "date"], kind="stable").copy()
    o["_i"] = o.groupby("horse_id").cumcount()
    o["_d"] = (o["date"].to_numpy("datetime64[D]") - D0).astype(np.int64)
    fw = o[(o["win"] == 1) & (o["_i"] < 5)].groupby("horse_id")["_d"].min()
    d5 = o[o["_i"] == 4].set_index("horse_id")["_d"]
    H = pd.DataFrame({"fw": fw}).join(d5.rename("d5"), how="outer")
    H["known"] = H[["fw", "d5"]].min(axis=1)
    H["quick"] = H["fw"].notna()
    day = (r["date"].to_numpy("datetime64[D]") - D0).astype(np.int64)
    out = {}
    for role, col in {"dam": "dam", "dd": "dd"}.items():
        hk = r.drop_duplicates("horse_id").set_index("horse_id")[col].astype(str)
        codes, _ = pd.factorize(hk)
        kmap = pd.Series(codes, index=hk.index)
        kq = kmap.reindex(r["horse_id"]).to_numpy()
        res = {}
        for nm, msk in (("known", np.ones(len(H), bool)), ("quick", H["quick"].to_numpy())):
            e = H.loc[msk, "known"]
            cnt = features2._events_count(kmap.reindex(e.index).to_numpy(), e.to_numpy().astype(np.int64), kq, day)
            mine = H.reindex(r["horse_id"])
            mine_b = (np.nan_to_num(mine["known"].to_numpy(), nan=1e12) < day) & (mine["quick"].fillna(False).to_numpy() if nm == "quick" else True)
            res[nm] = cnt - mine_b.astype(int)
        bad = (hk == "").reindex(r["horse_id"]).to_numpy()
        kn = np.where(bad, 0, res["known"])
        qu = np.where(bad, 0, res["quick"])
        out[f"{role}_v_quick"] = ((qu + 2 * 0.3) / (kn + 2)).astype("float32")
    return pd.DataFrame(out, index=r.index)


def dist_width(r: pd.DataFrame) -> pd.DataFrame:
    r = r.copy()
    r["_d1"] = r["distance"] * r["top3"]
    r["_d2"] = (r["distance"] ** 2) * r["top3"]
    out = {}
    for role, cols in {"sire": ["sire"], "bms": ["ds"]}.items():
        s, _ = asof(r, cols + ["surf"], ["_d1", "_d2", "top3"])
        n = s[:, 2]
        mu = np.where(n > 0, s[:, 0] / np.maximum(n, 1), np.nan)
        var = np.where(n > 1, s[:, 1] / np.maximum(n, 1) - mu ** 2, np.nan)
        sd = np.sqrt(np.clip(var, 100.0 ** 2, None))
        # 件数が少ないと sd が不安定: 10 頭分の擬似件数で 400m に寄せる
        sd = (sd * n + 400.0 * 10) / (n + 10)
        out[f"{role}_dz"] = (np.abs(r["distance"].to_numpy() - mu) / sd).astype("float32")
        out[f"{role}_dsd"] = sd.astype("float32")
    return pd.DataFrame(out, index=r.index)


def build(r: pd.DataFrame, ped: pd.DataFrame, sub: bool = True) -> tuple[pd.DataFrame, dict]:
    X, kinfo = features2.build(r, ped, sub=sub)
    X = X.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)
    ex = pd.concat([resid_apt(X), dam_quick(X), dist_width(X)], axis=1)
    X = pd.concat([X, ex], axis=1)
    rid = X["race_id"].to_numpy()
    for c in ["sire_rasurf_top3", "bms_rasurf_top3", "sire_surf_e", "bms_surf_e", "dam_v_quick", "sire_dz"]:
        X[f"rel_{c}"] = (X[c] - X.groupby(rid)[c].transform("mean")).astype("float32")
    return X, kinfo
