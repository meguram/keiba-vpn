"""段 1: 血統だけの特徴量（as-of）。

列の種類
- 鍵（集計を分けるだけ。主のモデルには入れない）: 芝ダ・距離帯・馬場・場・クラス帯・年齢帯・性・初出走・休み明け・初の芝ダ・距離の延長短縮。
- 特徴量: 祖先（父・母父・父父・系統・母・2 代母・父 × 母父）の産駒の、条件別の成績（縮小した率と件数）、
  産駒の適性の平均（距離・ダート率）と今回の条件との差、5 代のインブリード。
すべて `data.asof`（その日より前・自分を引く）で作る。祖先の id はそのまま特徴量に入れない（開催の無作為の分けで未来の成績を覚えるため）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.pedigree_lab.data import asof

DIST_BINS = [0, 1300, 1600, 2000, 2400, 9999]
DIST_LAB = ["S", "M", "I", "L", "E"]          # ～1300・1400-1600・1700-2000・2100-2400・2500～


def add_context(r: pd.DataFrame) -> pd.DataFrame:
    """鍵の列（血統ではない。集計を分ける条件）。馬の過去は「走った事実」だけ（着順は使わない）。"""
    r = r.copy()
    r["surf"] = np.where(r["surface"] == "芝", "T", "D")
    r["dband"] = pd.cut(r["distance"], DIST_BINS, labels=DIST_LAB).astype(str)
    r["going"] = np.where(r["track_condition"].isin(["稍", "重", "不"]), "W", "G")   # 道悪 / 良
    r["cls"] = r["grade"].map({"新馬": "NEW", "未勝利": "MDN", "1勝": "C1", "2勝": "C2", "3勝": "C3"}).fillna("OP")
    r["ageb"] = r["age"].clip(upper=5).fillna(5).astype(int).astype(str)
    r["sexb"] = np.where(r["sex"] == "牝", "F", "M")
    r["by"] = r["horse_id"].str[:4]
    r = r.sort_values(["horse_id", "date"], kind="stable")
    g = r.groupby("horse_id")
    r["career"] = g.cumcount()
    r["prev_date"] = g["date"].shift()
    r["prev_dist"] = g["distance"].shift()
    r["n_surf_before"] = r.groupby(["horse_id", "surf"]).cumcount()
    r = r.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)
    exact = r["by"].astype(int) >= 2014              # 2016 年より前の出走が無い世代だけ、キャリアの数が正確
    r["debut"] = np.where(exact & (r["career"] == 0), "Y", "N")
    gap = (r["date"] - r["prev_date"]).dt.days
    r["layoff"] = np.where(gap.isna(), "X", np.where(gap >= 90, "Y", "N"))
    r["first_surf"] = np.where(exact & (r["career"] > 0) & (r["n_surf_before"] == 0), "Y", "N")
    dd = r["distance"] - r["prev_dist"]
    r["dchg"] = np.where(dd.isna(), "X", np.where(dd >= 200, "U", np.where(dd <= -200, "D", "S")))
    return r


def _rate(s, c, prior, m):
    return (s + m * prior) / (c + m)


ROLES = {   # 祖先の役 → ped の列（1 列か、2 列の組）
    "sire": ["sire"], "bms": ["ds"], "ss": ["ss"], "sss": ["sss"], "bms2": ["dds"],
    "dam": ["dam"], "dd": ["dd"], "nick": ["sire", "ds"], "nickl": ["ss", "ds"],
}
CONDS = {   # 条件の組（鍵）
    "all": [], "surf": ["surf"], "sd": ["surf", "dband"], "sg": ["surf", "going"], "vs": ["venue", "surf"],
    "vsd": ["venue", "surf", "dband"], "cls": ["cls"], "age": ["ageb"], "sex": ["sexb"],
    "deb": ["debut"], "lay": ["layoff"], "fs": ["first_surf", "surf"], "dchg": ["dchg"], "agesurf": ["ageb", "surf"],
}
# 役ごとに作る条件（きょうだい・牝系は件数が少ないので粗い条件だけ）
ROLE_CONDS = {
    "sire": list(CONDS), "bms": list(CONDS), "ss": ["all", "surf", "sd", "sg", "age", "deb", "fs"],
    "sss": ["all", "surf", "sd", "sg"], "bms2": ["all", "surf", "sd"],
    "dam": ["all", "surf", "sd"], "dd": ["all", "surf", "sd"], "nick": ["all", "surf", "sd"], "nickl": ["all", "surf", "sd"],
}
VALS = ["win", "top3", "ex_win", "ex_top3", "fpct"]
M = {"win": 30.0, "top3": 30.0}    # 縮小の強さ（擬似件数）


def role_stats(r: pd.DataFrame) -> pd.DataFrame:
    """役 × 条件 ごとの as-of の成績。率は「1 つ粗い条件の同じ役」へ縮小し、最後は全体の条件別の平均へ。"""
    out = {}
    # 条件ごとの全体の平均（as-of・自分を引く。事前の床）
    base = {}
    for cn, ck in CONDS.items():
        if not ck:
            s, c = asof(r.assign(_k="1"), ["_k"], VALS)
        else:
            s, c = asof(r, ck, VALS)
        base[cn] = np.where(c[:, None] > 0, s / np.maximum(c, 1)[:, None], np.nan)
    gl = np.nanmean(base["all"], axis=0)
    for cn in base:
        b = base[cn]
        base[cn] = np.where(np.isnan(b), gl, b)
    parent = {"surf": "all", "sd": "surf", "sg": "surf", "vs": "surf", "vsd": "vs", "cls": "all", "age": "all",
              "sex": "all", "deb": "all", "lay": "all", "fs": "surf", "dchg": "all", "agesurf": "age"}
    for role, cols in ROLES.items():
        keep = {}
        for cn in ROLE_CONDS[role]:
            ck = cols + CONDS[cn]
            s, c = asof(r, ck, VALS)
            # 縮小の先: 同じ役の親の条件の率（無ければ全体の条件別）を、条件の全体の差で補正
            if cn == "all":
                prior = base["all"]
            else:
                p = keep.get(parent[cn])
                prior = base[cn] if p is None else p + (base[cn] - base[parent[cn]])
            rate = (s + 30.0 * prior) / (c[:, None] + 30.0)
            keep[cn] = rate
            for j, v in enumerate(VALS):
                if v in ("win", "top3", "fpct", "ex_win"):
                    out[f"{role}_{cn}_{v}"] = rate[:, j].astype("float32")
            out[f"{role}_{cn}_n"] = np.log1p(c).astype("float32")
            if cn != "all":    # 親との差（条件の効きの強さ）
                out[f"{role}_{cn}_dtop3"] = (rate[:, 1] - keep[parent[cn]][:, 1] if parent[cn] in keep else rate[:, 1] - prior[:, 1]).astype("float32")
    return pd.DataFrame(out, index=r.index)


def aptitude(r: pd.DataFrame) -> pd.DataFrame:
    """産駒の適性の平均（距離・ダート）。勝ち・3 着内の産駒の距離の平均と今回の距離との差。"""
    out = {}
    r = r.copy()
    r["_dist_t3"] = r["distance"] * r["top3"]
    r["_dirt_t3"] = (r["surf"] == "D") * r["top3"]
    r["_dirt"] = (r["surf"] == "D").astype(float)
    r["_dist"] = r["distance"].astype(float)
    r["_wet_t3"] = (r["going"] == "W") * r["top3"]
    r["_wet"] = (r["going"] == "W").astype(float)
    for role in ["sire", "bms", "dam", "dd", "ss"]:
        cols = ROLES[role]
        s, c = asof(r, cols, ["_dist_t3", "top3", "_dirt_t3", "_dirt", "_dist", "_wet_t3", "_wet"])
        t3 = s[:, 1]
        md = np.where(t3 > 0, s[:, 0] / np.maximum(t3, 1), np.nan)
        out[f"{role}_apt_dist"] = md.astype("float32")
        out[f"{role}_apt_ddiff"] = (r["distance"].to_numpy() - md).astype("float32")
        out[f"{role}_apt_absdd"] = np.abs(r["distance"].to_numpy() - md).astype("float32")
        # ダートでの好走の割合 ÷ ダートを走った割合（ダート寄りの度合い）
        dirt_share_t3 = (s[:, 2] + 3 * 0.5) / (t3 + 3)
        dirt_share_run = (s[:, 3] + 3 * 0.5) / (c + 3)
        out[f"{role}_apt_dirt"] = (dirt_share_t3 - dirt_share_run).astype("float32")
        wet_t3 = (s[:, 5] + 3 * 0.2) / (t3 + 3)
        wet_run = (s[:, 6] + 3 * 0.2) / (c + 3)
        out[f"{role}_apt_wet"] = (wet_t3 - wet_run).astype("float32")
        out[f"{role}_apt_meanrun"] = np.where(c > 0, s[:, 4] / np.maximum(c, 1), np.nan).astype("float32")
    return pd.DataFrame(out, index=r.index)


KEY_ANC = {   # 名の知れた祖先（インブリードの鍵）。id は 5 代血統の表のもの
    "SS": "000a00033a",   # サンデーサイレンス
}


def inbreeding(p: pd.DataFrame) -> pd.DataFrame:
    """5 代のインブリード: 父方と母方の両方に出る祖先の数・最も近い組の代の和・Wright の近交係数の近似。"""
    rows = []
    for h, a in zip(p["horse_id"], p["anc5"].fillna("")):
        side = {}
        for t in a.split(";"):
            if not t:
                continue
            g, pos, i = t.split(":")
            g, pos = int(g), int(pos)
            sd = "S" if pos < 2 ** (g - 1) else "D"
            side.setdefault(i, []).append((g, sd))
        n_cross, best, F, ss_cross = 0, 99, 0.0, 0
        for i, lst in side.items():
            S = [g for g, s in lst if s == "S"]
            D = [g for g, s in lst if s == "D"]
            if S and D:
                n_cross += 1
                for a1 in S:
                    for b1 in D:
                        F += 0.5 ** (a1 + b1 - 1)
                best = min(best, min(S) + min(D))
                if i == KEY_ANC["SS"]:
                    ss_cross = min(S) + min(D)
        rows.append((h, n_cross, best if best < 99 else np.nan, F, ss_cross))
    return pd.DataFrame(rows, columns=["horse_id", "ib_n", "ib_best", "ib_F", "ib_SS"])


def build(r: pd.DataFrame, ped: pd.DataFrame) -> pd.DataFrame:
    r = add_context(r)
    pc = ["horse_id", *sorted({c for cols in ROLES.values() for c in cols})]
    r = r.merge(ped[pc], on="horse_id", how="left")
    for c in pc[1:]:
        r[c] = r[c].fillna("").astype(str)
    st = role_stats(r)
    ap = aptitude(r)
    ib = inbreeding(ped[ped["horse_id"].isin(r["horse_id"].unique())])
    X = pd.concat([r, st, ap], axis=1).merge(ib, on="horse_id", how="left")
    return X
