"""段 1 の第 2 版の特徴量（2 体の R2 を反映。log.md の「段 1 R2」）。

第 1 版からの変更:
- 馬場は 3 段（G 良・S 稍重・W 重と不良）。第 1 版は「重」だけを道悪にしていた（不具合）。
- 縮小の強さ k は、役 × 条件 × 物差しごとにモーメント法で決める（2019 年末までの行。同じ馬の走りの相関を design effect で割り引く）。
  縮める先は「1 つ粗い条件の同じ役の率 ＋ 条件の全体の差（ロジットで足す）」。ニックスは「父 + 母父 − 全体」（ロジット）へ。
- 母・2 代母・全きょうだい・父は「1 頭 1 票」の数（走った産駒の頭数・勝ち上がった頭数・2 勝クラスに上がった頭数）も作る。
- 主の版の鍵は、レースの条件（芝ダ・距離帯・馬場・場・クラス）と年齢だけ。馬の状態の鍵（初出走・休み明け・初の芝ダ・距離の延長短縮）は副の版。
- レースの中の偏差（その列 − レースの平均）を主な率に付ける。
- 初出走は「2014 年以降の生まれ・新馬か未勝利・キャリア 0」だけ（地方からの転入を外す）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.pedigree_lab.data import asof

DIST_BINS = [0, 1300, 1600, 2000, 2400, 9999]
DIST_LAB = ["S", "M", "I", "L", "E"]
K_END = pd.Timestamp("2019-12-31")       # k を決める行の終わり（2017 年末では母・条件の組が薄すぎた。k は 1 つの超母数で、2017〜19 の行へのわずかな先取りを log に記録）


def add_context(r: pd.DataFrame) -> pd.DataFrame:
    r = r.copy()
    r["surf"] = np.where(r["surface"] == "芝", "T", "D")
    r["dband"] = pd.cut(r["distance"], DIST_BINS, labels=DIST_LAB).astype(str)
    r["going"] = r["track_condition"].map({"良": "G", "稍重": "S", "重": "W", "不良": "W"}).fillna("G")
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
    exact = pd.to_numeric(r["by"], errors="coerce").fillna(0) >= 2014
    r["debut"] = np.where(exact & (r["career"] == 0) & r["cls"].isin(["NEW", "MDN"]), "Y", "N")
    gap = (r["date"] - r["prev_date"]).dt.days
    r["layoff"] = np.where(gap.isna(), "X", np.where(gap >= 90, "Y", "N"))
    r["first_surf"] = np.where(exact & (r["career"] > 0) & (r["n_surf_before"] == 0), "Y", "N")
    dd = r["distance"] - r["prev_dist"]
    r["dchg"] = np.where(dd.isna(), "X", np.where(dd >= 200, "U", np.where(dd <= -200, "D", "S")))
    return r


ROLES = {"sss": ["sss"], "ss": ["ss"], "sire": ["sire"], "bms": ["ds"], "dd": ["dd"], "dam": ["dam"],
         "full": ["sire", "dam"], "nick": ["sire", "ds"]}      # 親（上の階層）が先に来る順
CONDS = {"all": [], "surf": ["surf"], "sd": ["surf", "dband"], "sg": ["surf", "going"], "vs": ["venue", "surf"],
         "cls": ["cls"], "agesurf": ["ageb", "surf"],
         # 副の版（馬の状態の鍵）
         "deb": ["debut"], "lay": ["layoff"], "fs": ["first_surf", "surf"], "dchg": ["dchg"]}
PARENT = {"surf": "all", "sd": "surf", "sg": "surf", "vs": "surf", "cls": "all", "agesurf": "surf",
          "deb": "all", "lay": "all", "fs": "surf", "dchg": "all"}
MAIN_CONDS = {
    "sire": ["all", "surf", "sd", "sg", "vs", "cls", "agesurf"],
    "bms": ["all", "surf", "sd", "sg", "cls", "agesurf"],
    "ss": ["all", "surf", "sd", "sg"], "sss": ["all", "surf"],
    "dam": ["all", "surf"], "dd": ["all", "surf"], "full": ["all"], "nick": ["all", "surf"],
}
SUB_CONDS = {"sire": ["deb", "lay", "fs", "dchg"], "bms": ["deb", "fs", "dchg"], "ss": ["deb", "fs"]}
VALS = ["win", "top3", "fpct"]
BINOM = {"win", "top3"}


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _combine(parent: np.ndarray, d_base: np.ndarray, binom: bool) -> np.ndarray:
    """親の率に条件の全体の差を足す（二項はロジットで、分位はそのまま）。"""
    if binom:
        return _sig(_logit(parent) + d_base)
    return parent + d_base


def _rho(r: pd.DataFrame, v: str) -> float:
    """同じ馬の走りどうしの相関（一元配置のモーメント法）。"""
    g = r.groupby("horse_id")[v].agg(["mean", "count"])
    g = g[g["count"] >= 2]
    s2 = r[v].var()
    tau = g["mean"].var() - np.mean(s2 / g["count"])
    return float(np.clip(tau / s2, 0.0, 0.9))


def estimate_k(r: pd.DataFrame, keys: list[str], v: str, prior_cols: list[str] | None, rho: float,
               cond_cols: list[str] | None = None, pcond_cols: list[str] | None = None) -> float:
    """鍵の間の真のばらつき τ² をモーメント法で。k（走数の単位）= σ²·deff ÷ τ²。

    prior_cols があれば、親の鍵の平均を引いた残り（条件の効きの上乗せ）で測る。2019 年末までの行だけ。
    cond_cols・pcond_cols（条件の鍵・親の条件の鍵）があれば、条件の全体の差を「親の条件の中の差」で引く（縮める先と同じ形。critic 1）。
    """
    d = r[r["date"] <= K_END]
    for c in keys:
        d = d[d[c].astype(str) != ""]
    if len(d) == 0:
        return 100.0
    s2 = d[v].var()
    hier = bool(prior_cols) and not set(prior_cols) <= set(keys)     # 上の階層（父 → 父父など）: 残り = 平均 − 親の平均
    gk = keys + [c for c in (prior_cols or []) if c not in keys] if hier else keys
    g = d.groupby(gk).agg(m=(v, "mean"), n=(v, "size"), h=("horse_id", "nunique"))
    if hier:
        pm = d.groupby(prior_cols)[v].mean().rename("pm")
        g = g.reset_index().merge(pm.reset_index(), on=prior_cols, how="left")
        g["m"] = g["m"] - g["pm"]
    elif prior_cols:
        pm = d.groupby(prior_cols)[v].mean().rename("pm")
        cond = cond_cols if cond_cols is not None else [k for k in keys if k not in prior_cols]
        cm = d.groupby(cond)[v].mean().rename("cm") if cond else None
        g = g.reset_index().merge(pm.reset_index(), on=prior_cols, how="left")
        if cm is not None:
            g = g.merge(cm.reset_index(), on=cond, how="left")
            if pcond_cols:
                pc = d.groupby(pcond_cols)[v].mean().rename("pcm")
                g = g.merge(pc.reset_index(), on=pcond_cols, how="left")
                g["m"] = g["m"] - g["pm"] - (g["cm"] - g["pcm"])
            else:
                g["m"] = g["m"] - g["pm"] - (g["cm"] - d[v].mean())
        else:
            g["m"] = g["m"] - g["pm"]
    g = g[g["n"] >= 15]
    if prior_cols:                   # 親の中に子の組が 2 つ以上あるものだけ（1 つだけだと残りが 0 になり τ² を小さく見せる）
        g = g[g.groupby(prior_cols)["m"].transform("size") >= 2]
    if len(g) < 10:
        return 100.0
    deff = 1 + (g["n"] / g["h"] - 1) * rho
    noise = s2 * deff / g["n"]
    w = g["n"] / g["n"].sum()
    mu = np.sum(w * g["m"])
    tau2 = np.sum(w * (g["m"] - mu) ** 2) - np.sum(w * noise)
    deff_typ = float(np.median(deff))
    if tau2 <= 1e-9:
        return 5000.0
    return float(np.clip(s2 * deff_typ / tau2, 5.0, 5000.0))


def role_stats(r: pd.DataFrame, sub: bool) -> tuple[pd.DataFrame, dict]:
    out, kinfo = {}, {}
    rho = {v: _rho(r[r["date"] <= K_END], v) for v in VALS}
    kinfo["rho"] = rho
    conds_used = sorted({c for cs in MAIN_CONDS.values() for c in cs} | ({c for cs in SUB_CONDS.values() for c in cs} if sub else set()))
    base = {}
    for cn in conds_used:
        ck = CONDS[cn]
        s, c = asof(r.assign(_k="1"), ["_k"] + ck, VALS)
        gm = r.loc[r["date"].dt.year == 2016, VALS].mean().to_numpy()
        base[cn] = np.where(c[:, None] > 30, s / np.maximum(c, 1)[:, None], gm)   # 30 件までは 2016 年（助走）の平均
    rates = {}
    for role, cols in ROLES.items():
        cl = list(MAIN_CONDS[role]) + (SUB_CONDS.get(role, []) if sub else [])
        for cn in cl:
            ck = cols + CONDS[cn]
            s, c = asof(r, ck, VALS)
            for j, v in enumerate(VALS):
                binom = v in BINOM
                if role == "nick" and cn == "all":
                    # 父 + 母父 − 全体（ロジット）
                    a, b = rates[("sire", "all", v)], rates[("bms", "all", v)]
                    prior = _sig(_logit(a) + _logit(b) - _logit(base["all"][:, j])) if binom else a + b - base["all"][:, j]
                    pcols = ["sire"]          # 母父の列が条件の側になり、足し算の残りで k を測る
                elif cn == "all":
                    up = {"sire": "ss", "ss": "sss", "full": "dam", "dam": "dd"}.get(role)
                    if up and (up, "all", v) in rates:
                        prior = rates[(up, "all", v)]
                    else:
                        prior = base["all"][:, j]
                    # 母は 2 代母の中に 1〜3 頭しかおらず、親の平均を引くと残りが消える。母・全きょうだいは生のばらつきで測る
                    pcols = ROLES[up] if up and role not in ("dam", "full") else None
                else:
                    pa = PARENT[cn]
                    par = rates.get((role, pa, v))
                    if par is None:
                        par = base[pa][:, j]
                    db = (_logit(base[cn][:, j]) - _logit(base[pa][:, j])) if binom else base[cn][:, j] - base[pa][:, j]
                    prior = _combine(par, db, binom)
                    pcols = cols + CONDS[pa]
                if cn != "all":
                    k = estimate_k(r, ck, v, pcols, rho[v], cond_cols=CONDS[cn], pcond_cols=CONDS[PARENT[cn]] or None)
                else:
                    k = estimate_k(r, ck, v, pcols, rho[v])
                kinfo[f"{role}_{cn}_{v}"] = round(k, 1)
                rate = (s[:, j] + k * prior) / (c + k)
                rates[(role, cn, v)] = rate
                out[f"{role}_{cn}_{v}"] = rate.astype("float32")
            out[f"{role}_{cn}_n"] = np.log1p(np.maximum(c, 0)).astype("float32")
            if cn != "all" and role in ("sire", "bms", "ss"):
                pa = PARENT[cn]
                out[f"{role}_{cn}_dfp"] = (rates[(role, cn, "fpct")] - rates.get((role, pa, "fpct"), base[pa][:, 2])).astype("float32")
    return pd.DataFrame(out, index=r.index), kinfo


def _events_count(key_ev: np.ndarray, day_ev: np.ndarray, key_q: np.ndarray, day_q: np.ndarray) -> np.ndarray:
    """鍵ごとに「その日より前」の出来事の数（鍵は整数、日は整数）。"""
    code_ev = key_ev.astype(np.int64) * 100000 + day_ev
    code_ev.sort()
    code_q = key_q.astype(np.int64) * 100000 + day_q
    lo = np.searchsorted(code_ev, key_q.astype(np.int64) * 100000, "left")
    hi = np.searchsorted(code_ev, code_q, "left")
    return hi - lo


def horse_votes(r: pd.DataFrame) -> pd.DataFrame:
    """1 頭 1 票: 鍵（父・母・2 代母・全きょうだい）ごとに、その日より前に「走った」「1 年以上前に走り始めた」
    「勝ち上がった」「1 勝クラス以上で勝った（2 勝クラスに上がった）」産駒の頭数。自分は引く。"""
    D0 = np.datetime64("2010-01-01")
    day = (r["date"].to_numpy("datetime64[D]") - D0).astype(np.int64)
    hh = r.assign(_d=day)
    first_run = hh.groupby("horse_id")["_d"].min()
    first_win = hh[hh["win"] == 1].groupby("horse_id")["_d"].min()
    first_hi = hh[(hh["win"] == 1) & hh["cls"].isin(["C1", "C2", "C3", "OP"])].groupby("horse_id")["_d"].min()
    H = pd.DataFrame({"fr": first_run}).join(first_win.rename("fw")).join(first_hi.rename("fh"))
    out = {}
    hid = r["horse_id"].to_numpy()
    own = H.reindex(hid)
    for role, cols in {"sire": ["sire"], "dam": ["dam"], "dd": ["dd"], "full": ["sire", "dam"], "bms": ["ds"]}.items():
        hk = r.drop_duplicates("horse_id").set_index("horse_id")[cols]
        key = hk.astype(str).agg("|".join, axis=1) if len(cols) > 1 else hk[cols[0]].astype(str)
        empty = (hk == "").any(axis=1)
        codes, uniq = pd.factorize(key)
        kmap = pd.Series(codes, index=hk.index)
        kq = kmap.reindex(hid).to_numpy()
        res = {}
        for ev, shift in (("fr", 0), ("mat", 365), ("fw", 0), ("fh", 0)):
            col = "fr" if ev == "mat" else ev
            e = H[col].dropna()
            ke = kmap.reindex(e.index).to_numpy()
            de = e.to_numpy().astype(np.int64) + shift
            cnt = _events_count(ke, de, kq, day)
            mine = own[col].to_numpy()
            mine_before = np.nan_to_num(mine + shift, nan=1e12) < day
            res[ev] = cnt - mine_before.astype(int)
        bad = empty.reindex(hid).to_numpy()
        for ev in res:
            res[ev] = np.where(bad, 0, res[ev])
        # 1 年以上前に走り始めた産駒のうち勝ち上がった割合（若い産駒は分母に入れない）。勝ちの数は全体から（上限は mat）
        mat = res["mat"]
        out[f"{role}_v_nrun"] = np.log1p(res["fr"]).astype("float32")
        out[f"{role}_v_win"] = ((np.minimum(res["fw"], mat) + 2 * 0.45) / (mat + 2)).astype("float32")
        out[f"{role}_v_hi"] = ((np.minimum(res["fh"], mat) + 2 * 0.15) / (mat + 2)).astype("float32")
    return pd.DataFrame(out, index=r.index)


KEY_ANC = {"SS": "000a00033a"}


def inbreeding(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, a in zip(p["horse_id"], p["anc5"].fillna("")):
        side = {}
        for t in a.split(";"):
            if not t:
                continue
            g, pos, i = t.split(":")
            g, pos = int(g), int(pos)
            side.setdefault(i, []).append((g, "S" if pos < 2 ** (g - 1) else "D"))
        F, ss = 0.0, 0
        for i, lst in side.items():
            S = [g for g, s in lst if s == "S"]
            D = [g for g, s in lst if s == "D"]
            if S and D:
                for a1 in S:
                    for b1 in D:
                        F += 0.5 ** (a1 + b1 - 1)
                if i == KEY_ANC["SS"]:
                    ss = min(S) + min(D)
        rows.append((h, F, ss))
    return pd.DataFrame(rows, columns=["horse_id", "ib_F", "ib_SS"])


REL = ["sire_all_top3", "sire_sd_top3", "sire_sg_top3", "sire_agesurf_top3", "bms_all_top3", "bms_sd_top3",
       "ss_sd_top3", "dam_all_fpct", "dd_all_fpct", "nick_all_top3", "sire_all_win", "sire_sd_win", "bms_sd_win",
       "sire_vs_top3", "sire_cls_top3", "dam_v_win", "dd_v_win", "full_v_win", "sire_v_win", "sire_sd_fpct", "bms_sd_fpct"]


def build(r: pd.DataFrame, ped: pd.DataFrame, sub: bool = False) -> tuple[pd.DataFrame, dict]:
    r = add_context(r)
    pc = ["horse_id", *sorted({c for cols in ROLES.values() for c in cols})]
    r = r.merge(ped[pc], on="horse_id", how="left")
    for c in pc[1:]:
        r[c] = r[c].fillna("").astype(str)
    st, kinfo = role_stats(r, sub)
    hv = horse_votes(r)
    X = pd.concat([r, st, hv], axis=1)
    rid = X["race_id"].to_numpy()
    for c in REL:
        if c in X:
            X[f"rel_{c}"] = (X[c] - X.groupby(rid)[c].transform("mean")).astype("float32")
    ib = inbreeding(ped[ped["horse_id"].isin(r["horse_id"].unique())])
    X = X.merge(ib, on="horse_id", how="left")
    return X, kinfo
