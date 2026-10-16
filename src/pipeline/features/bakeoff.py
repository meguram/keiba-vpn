"""tp-16 特徴量 bake-off の候補群。どれも「そのレースの日より前に分かる情報だけ」で作る（同日の結果は使わない）。

入力 df は race_id・horse_id・date・venue_code・surface・distance・finish_position・field_size を持つ表（並び順は問わない）。
戻り値は df と同じ index の特徴量表。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _res(df: pd.DataFrame) -> np.ndarray:
    fs = df["finish_position"].to_numpy(dtype=float)
    n = np.maximum(df["field_size"].to_numpy(dtype=float) - 1.0, 1.0)
    return 0.5 - (fs - 1.0) / n


def _horse_order(df: pd.DataFrame) -> np.ndarray:
    return np.lexsort((df["race_id"].astype(str).to_numpy(), pd.to_datetime(df["date"]).to_numpy(), df["horse_id"].astype(str).to_numpy()))


def _blocks(order: np.ndarray, hid: np.ndarray):
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and hid[order[j]] == hid[order[i]]:
            j += 1
        yield i, j
        i = j


# ---------------------------------------------------------------- 季節性

def calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """日付の周期（年の何日目を角度にした sin/cos の 1・2 次）、月、曜日、開催回・開催日（race_id から）。"""
    d = pd.to_datetime(df["date"])
    ang = 2 * np.pi * (d.dt.dayofyear.to_numpy(dtype=float) - 1.0) / 365.25
    rid = df["race_id"].astype(str)
    return pd.DataFrame({
        "cal_sin1": np.sin(ang), "cal_cos1": np.cos(ang), "cal_sin2": np.sin(2 * ang), "cal_cos2": np.cos(2 * ang),
        "cal_month": d.dt.month.to_numpy(), "cal_weekday": d.dt.weekday.to_numpy(),
        "cal_kai": pd.to_numeric(rid.str[6:8], errors="coerce").to_numpy(), "cal_nichi": pd.to_numeric(rid.str[8:10], errors="coerce").to_numpy(),
    }, index=df.index)


def horse_season_features(df: pd.DataFrame, ridge: float = 4.0, kappa: float = 3.0, conc: float = 2.0) -> pd.DataFrame:
    """馬ごとの季節適性。(1) 年周期の sin/cos へのリッジ回帰の予測値と不確かさ、(2) 近い季節の過去走だけを重みづけした平均（von Mises 重み）。"""
    n = len(df)
    res = _res(df)
    ang_all = 2 * np.pi * (pd.to_datetime(df["date"]).dt.dayofyear.to_numpy(dtype=float) - 1.0) / 365.25
    dates = pd.to_datetime(df["date"]).to_numpy()
    hid = df["horse_id"].astype(str).to_numpy()
    order = _horse_order(df)
    out = np.full((n, 4), np.nan)
    for i, j in _blocks(order, hid):
        A = np.eye(3) * ridge
        b = np.zeros(3)
        past_ang: list[float] = []
        past_res: list[float] = []
        pending: list[tuple[float, float]] = []
        cur = None
        for t in range(i, j):
            r = order[t]
            if cur is not None and dates[r] != cur:
                for a_, y_ in pending:
                    z = np.array([1.0, np.sin(a_), np.cos(a_)])
                    A += np.outer(z, z); b += z * y_
                    past_ang.append(a_); past_res.append(y_)
                pending = []
            cur = dates[r]
            z = np.array([1.0, np.sin(ang_all[r]), np.cos(ang_all[r])])
            out[r, 0] = float(z @ np.linalg.solve(A, b))
            out[r, 1] = float(np.sqrt(z @ np.linalg.solve(A, z)))
            if past_res:
                w = np.exp(conc * (np.cos(np.array(past_ang) - ang_all[r]) - 1.0))
                pr = np.array(past_res)
                parent = pr.mean()
                out[r, 2] = (float(w @ pr) + kappa * parent) / (float(w.sum()) + kappa)
                out[r, 3] = float(w.sum())
            if not np.isnan(res[r]):
                pending.append((ang_all[r], res[r]))
    return pd.DataFrame(out, columns=["hs_pred", "hs_se", "hs_vm_aff", "hs_vm_n"], index=df.index)


# ---------------------------------------------------------------- 京都改修

KYOTO_CLOSE = pd.Timestamp("2020-11-30")      # 改修による閉鎖（2020 年秋開催の終了後）
KYOTO_REOPEN = pd.Timestamp("2023-04-22")


def venue_era_features(df: pd.DataFrame) -> pd.DataFrame:
    """京都は改修後（2023-04 の再開以降）を別枠にする。馬が改修後の京都を走った経験数も付ける。"""
    d = pd.to_datetime(df["date"])
    kyoto = df["venue_code"].astype(str).eq("08")
    era = np.where(~kyoto, 0, np.where(d >= KYOTO_REOPEN, 2, 1))
    post = (kyoto & (d >= KYOTO_REOPEN)).to_numpy()
    hid = df["horse_id"].astype(str).to_numpy()
    dates = d.to_numpy()
    order = _horse_order(df)
    n_post = np.zeros(len(df))
    for i, j in _blocks(order, hid):
        cnt = 0
        pending = 0
        cur = None
        for t in range(i, j):
            r = order[t]
            if cur is not None and dates[r] != cur:
                cnt += pending; pending = 0
            cur = dates[r]
            n_post[r] = cnt
            pending += int(post[r])
    return pd.DataFrame({"ve_kyoto_era": era, "ve_horse_n_kyoto_post": n_post}, index=df.index)


# ---------------------------------------------------------------- Glicko 型の不確実性

def glicko_rd_features(df: pd.DataFrame, per_week: float = 0.01, q: float = 1.5) -> pd.DataFrame:
    """能力推定の不確実性 RD。レースを走るたびに分散が 1/(1/v+q) へ縮み、休むと週あたり per_week だけ増える（上限 1）。"""
    n = len(df)
    dates = pd.to_datetime(df["date"]).to_numpy()
    hid = df["horse_id"].astype(str).to_numpy()
    order = _horse_order(df)
    rd = np.ones(n)
    for i, j in _blocks(order, hid):
        v = 1.0
        last = None
        for t in range(i, j):
            r = order[t]
            if last is not None:
                weeks = (dates[r] - last) / np.timedelta64(7, "D")
                v = min(1.0, v + per_week * float(weeks))
            rd[r] = np.sqrt(v)
            v = 1.0 / (1.0 / v + q)
            last = dates[r]
    out = pd.DataFrame({"g_rd": rd}, index=df.index)
    g = out.groupby(df["race_id"].to_numpy())["g_rd"]
    out["g_rd_gap"] = out["g_rd"] - g.transform("mean")
    out["g_rd_field_max"] = g.transform("max")
    return out


# ---------------------------------------------------------------- 後日談（相手の次走の成績）

def aftermath_features(df: pd.DataFrame, lags: int = 3, day_key: int = 100000) -> pd.DataFrame:
    """馬の直近 lags 走それぞれについて「そのレースの相手（自分以外）が、その後の出走でどれだけ走ったか」を、今回の日より前に分かる分だけ平均する。
    相手の次走の結果は、その次走の日に初めて分かる（当日は含めない）。"""
    n = len(df)
    res = _res(df)
    win = (df["finish_position"].to_numpy(dtype=float) == 1).astype(float)
    dnum = pd.to_datetime(df["date"]).to_numpy("datetime64[D]").astype(np.int64)
    hid = df["horse_id"].astype(str).to_numpy()
    rcode = pd.factorize(df["race_id"].astype(str))[0].astype(np.int64)
    order = _horse_order(df)
    same = hid[order[1:]] == hid[order[:-1]]
    nxt = np.full(n, -1, dtype=np.int64)
    nxt[order[:-1][same]] = order[1:][same]
    prv = np.full(n, -1, dtype=np.int64)
    prv[order[1:][same]] = order[:-1][same]

    valid = (nxt >= 0)
    valid[valid] = np.isfinite(res[nxt[valid]])
    src = np.flatnonzero(valid)
    key = rcode[src] * day_key + dnum[nxt[src]]
    o = np.argsort(key, kind="stable")
    src, key = src[o], key[o]
    c_res = np.concatenate([[0.0], np.cumsum(res[nxt[src]])])
    c_win = np.concatenate([[0.0], np.cumsum(win[nxt[src]])])

    S_r = np.zeros(n); S_w = np.zeros(n); C = np.zeros(n)
    first = None
    cur = np.arange(n)
    for lag in range(1, lags + 1):
        cur = np.where(cur >= 0, prv[np.maximum(cur, 0)], -1)           # lag 走前の行
        ok = cur >= 0
        p = np.maximum(cur, 0)
        lo = np.searchsorted(key, rcode[p] * day_key, side="left")
        hi = np.searchsorted(key, rcode[p] * day_key + dnum, side="left")      # 次走の日が今回の日より前のものだけ
        s_r = c_res[hi] - c_res[lo]
        s_w = c_win[hi] - c_win[lo]
        cnt = (hi - lo).astype(float)
        self_next = nxt[p]                                                 # 自分自身の次走（=相手ではない）を引く
        sn_ok = ok & (self_next >= 0)
        sn_ok[sn_ok] = (dnum[self_next[sn_ok]] < dnum[sn_ok]) & np.isfinite(res[self_next[sn_ok]])
        s_r = s_r - np.where(sn_ok, res[np.maximum(self_next, 0)], 0.0)
        s_w = s_w - np.where(sn_ok, win[np.maximum(self_next, 0)], 0.0)
        cnt = cnt - sn_ok.astype(float)
        s_r, s_w, cnt = np.where(ok, s_r, 0.0), np.where(ok, s_w, 0.0), np.where(ok, cnt, 0.0)
        if lag == 1:
            first = (s_r, cnt)
        S_r += s_r; S_w += s_w; C += cnt
    with np.errstate(invalid="ignore", divide="ignore"):
        out = pd.DataFrame({
            "af_res_l1": np.where(first[1] >= 3, first[0] / np.maximum(first[1], 1), np.nan),
            "af_res_3": np.where(C >= 3, S_r / np.maximum(C, 1), np.nan),
            "af_win_3": np.where(C >= 3, S_w / np.maximum(C, 1), np.nan),
            "af_cnt_3": C,
        }, index=df.index)
    return out


# ---------------------------------------------------------------- クッション値・含水率

CUSHION_COLS = ["cushion_value", "turf_moisture_goal", "turf_moisture_4corner", "dirt_moisture_goal", "dirt_moisture_4corner"]


def load_cushion(path: str | Path) -> pd.DataFrame:
    rows: list[dict] = []
    for p in sorted(Path(path).glob("cushion_20??.json")):
        rows += json.loads(Path(p).read_text(encoding="utf-8"))
    c = pd.DataFrame(rows)
    c = c[c["date"].notna() & c["venue_code"].notna()].copy()
    c["date"] = pd.to_datetime(c["date"]).dt.strftime("%Y-%m-%d")
    c["venue_code"] = c["venue_code"].astype(str).str.zfill(2)
    if "is_race_day" in c.columns:
        c = c[c["is_race_day"].fillna(False).astype(bool)]
    c = c.drop_duplicates(["date", "venue_code"], keep="last")
    return c[["date", "venue_code"] + [x for x in CUSHION_COLS if x in c.columns]]


def cushion_features(df: pd.DataFrame, cushion: pd.DataFrame, fit_end: str = "2021-12-21", ridge: float = 6.0) -> pd.DataFrame:
    """その日の馬場の硬さ・含水率（当日朝に分かる）と、馬ごとの「硬さ・水分への反応」（面ごとのリッジ回帰の予測値）。"""
    d = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    m = pd.DataFrame({"date": d.to_numpy(), "venue_code": df["venue_code"].astype(str).str.zfill(2).to_numpy()}).merge(cushion, on=["date", "venue_code"], how="left")
    turf = df["surface"].astype(str).eq("芝").to_numpy()
    cush = np.where(turf, m["cushion_value"].to_numpy(dtype=float), np.nan)
    goal = np.where(turf, m["turf_moisture_goal"].to_numpy(dtype=float), m["dirt_moisture_goal"].to_numpy(dtype=float))
    c4 = np.where(turf, m["turf_moisture_4corner"].to_numpy(dtype=float), m["dirt_moisture_4corner"].to_numpy(dtype=float))
    fit = pd.to_datetime(df["date"]).to_numpy() <= np.datetime64(fit_end)
    def st(x, mask):
        v = x[mask & np.isfinite(x)]
        return (float(v.mean()), float(v.std()) if len(v) > 1 and v.std() > 0 else 1.0)
    cz = (cush - st(cush, fit & turf)[0]) / st(cush, fit & turf)[1]
    gz_t = (goal - st(goal, fit & turf)[0]) / st(goal, fit & turf)[1]
    gz_d = (goal - st(goal, fit & ~turf)[0]) / st(goal, fit & ~turf)[1]
    gz = np.where(turf, gz_t, gz_d)
    out = pd.DataFrame({"cu_cushion": cush, "cu_moist_goal": goal, "cu_moist_4c": c4, "cu_moist_diff": goal - c4, "cu_cushion_z": cz, "cu_moist_z": gz}, index=df.index)

    res = _res(df)
    dates = pd.to_datetime(df["date"]).to_numpy()
    hid = df["horse_id"].astype(str).to_numpy()
    order = _horse_order(df)
    pred = np.full(len(df), np.nan)
    se = np.full(len(df), np.nan)
    czf = np.where(np.isfinite(cz), cz, 0.0)
    gzf = np.where(np.isfinite(gz), gz, 0.0)
    ok_x = np.isfinite(gz)
    for i, j in _blocks(order, hid):
        st_ = {}
        pending = []
        cur = None
        for t in range(i, j):
            r = order[t]
            if cur is not None and dates[r] != cur:
                for (s_, z_, y_) in pending:
                    A, b = st_.setdefault(s_, (np.eye(3) * ridge, np.zeros(3)))
                    A += np.outer(z_, z_); b += z_ * y_
                pending = []
            cur = dates[r]
            s_ = bool(turf[r])
            z = np.array([1.0, czf[r], gzf[r]])
            if ok_x[r]:
                A, b = st_.get(s_, (np.eye(3) * ridge, np.zeros(3)))
                pred[r] = float(z @ np.linalg.solve(A, b))
                se[r] = float(np.sqrt(z @ np.linalg.solve(A, z)))
                if not np.isnan(res[r]):
                    pending.append((s_, z, res[r]))
    out["cu_horse_pred"] = pred
    out["cu_horse_se"] = se
    return out


# ---------------------------------------------------------------- 近交係数（Wright）

def inbreeding_from_ancestors(anc: list[dict], max_gen: int = 8) -> dict:
    """anc: [{generation, position, horse_id, name}]。父側（position が 2^(g-1) 未満）と母側の両方に現れる祖先について、
    F = Σ 0.5^(ds+dd-1)（ds, dd は父側・母側の世代）を足す。同じ共通祖先を二度通る経路は除く。"""
    side_occ: dict[str, dict[int, list[tuple[int, int]]]] = {}
    ident: dict[tuple[int, int], str] = {}
    for a in anc:
        g, p = int(a["generation"]), int(a["position"])
        if g > max_gen:
            continue
        k = (a.get("horse_id") or "").strip() or (a.get("name") or "").strip()
        if not k:
            continue
        ident[(g, p)] = k
        s = 0 if p < 2 ** (g - 1) else 1
        side_occ.setdefault(k, {0: [], 1: []})[s].append((g, p))
    common = {k for k, v in side_occ.items() if v[0] and v[1]}
    if not common:
        return {"F": 0.0, "n_common": 0, "max_contrib": 0.0, "min_depth": np.nan}

    def chain_commons(g: int, p: int) -> set:
        cs = set()
        for kk in range(1, g):
            c = ident.get((g - kk, p >> kk))
            if c in common:
                cs.add(c)
        return cs

    F = 0.0
    best = 0.0
    mind = np.inf
    for k in common:
        tot = 0.0
        for (gs, ps) in side_occ[k][0]:
            cs_ = chain_commons(gs, ps)
            for (gd, pd_) in side_occ[k][1]:
                if cs_ & chain_commons(gd, pd_):
                    continue
                tot += 0.5 ** (gs + gd - 1)
                mind = min(mind, gs + gd)
        F += tot
        best = max(best, tot)
    return {"F": F, "n_common": len(common), "max_contrib": best, "min_depth": float(mind) if np.isfinite(mind) else np.nan}


def inbreeding_features(df: pd.DataFrame, ped_dir: str | Path, max_gen: int = 8) -> pd.DataFrame:
    """馬ごとの近交係数。血統は固定なのでリークは無い（10 世代血統 JSON を使う）。"""
    ped_dir = Path(ped_dir)
    rows = {}
    for h in pd.unique(df["horse_id"].astype(str)):
        f = ped_dir / h[:4] / f"{h}.json"
        if not f.exists():
            continue
        try:
            anc = json.loads(f.read_text(encoding="utf-8")).get("ancestors") or []
        except Exception:
            continue
        if anc:
            rows[h] = inbreeding_from_ancestors(anc, max_gen)
    t = pd.DataFrame.from_dict(rows, orient="index")
    t.columns = ["ib_F", "ib_n_common", "ib_max_contrib", "ib_min_depth"]
    return t.reindex(df["horse_id"].astype(str).to_numpy()).set_index(df.index)


# ---------------------------------------------------------------- 最終追い切りの日（調教から何日前か）

def parse_training_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str).str.extract(r"(\d{4}/\d{1,2}/\d{1,2})")[0], format="%Y/%m/%d", errors="coerce")


def oikiri_timing_features(df: pd.DataFrame, oikiri: pd.DataFrame) -> pd.DataFrame:
    """最終追い切りが何日前か（レース週の調整の長さ）。oikiri: race_id・horse_id・training_date。"""
    o = oikiri[["race_id", "horse_id", "training_date"]].copy()
    o["td"] = parse_training_date(o["training_date"])
    o = o.dropna(subset=["td"]).sort_values("td").drop_duplicates(["race_id", "horse_id"], keep="last")
    m = df[["race_id", "horse_id"]].astype(str).merge(o.assign(race_id=o["race_id"].astype(str), horse_id=o["horse_id"].astype(str))[["race_id", "horse_id", "td"]], on=["race_id", "horse_id"], how="left")
    days = (pd.to_datetime(df["date"]).to_numpy() - m["td"].to_numpy()) / np.timedelta64(1, "D")
    days = np.where((days >= 0) & (days <= 30), days, np.nan)
    return pd.DataFrame({"ok_days_before": days}, index=df.index)
