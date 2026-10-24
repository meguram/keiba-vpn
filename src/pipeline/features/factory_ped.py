"""特徴量ファクトリー G4: 血統の深掘り（キーはすべて ID。名前は使わない）。

血統は馬別の JSON（horse_pedigree_5gen / 10gen）の ancestors（generation・position・horse_id）から作る。
  gen1: pos0=父, pos1=母。gen2: pos0=父父, pos1=父母, pos2=母父, pos3=母母。
成績が入る群（父の条件別成績・ニックス・半きょうだい・祖先の産駒成績）はすべて「そのレースの日より前」の結果だけを使い、
自分自身の過去走は引く（きょうだい・祖先の成績に自分が混ざらないように）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features import factory_src as fs

ROOT = fs.ROOT
CACHE = ROOT / "data/local/modeling/training_pipeline/factory/src"
PED_DIRS = [ROOT / "data/local/horse_pedigree_5gen", ROOT / "data/local/horse_pedigree_10gen"]
MAX_GEN = 5


def build_ped_tables(horse_ids, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(祖先の長い表 horse_id・gen・pos・anc_id, 馬ごとの表 horse_id・mating_date)。キャッシュする。"""
    pa, ph = CACHE / "ped_anc.parquet", CACHE / "ped_horse.parquet"
    if pa.exists() and ph.exists() and not force:
        return pd.read_parquet(pa), pd.read_parquet(ph)
    anc, hs = [], []
    for h in sorted({str(x) for x in horse_ids}):
        data = None
        for d in PED_DIRS:
            f = d / h[:4] / f"{h}.json"
            if f.exists():
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    if data.get("ancestors"):
                        break
                except Exception:
                    data = None
        if not data:
            continue
        for a in data.get("ancestors") or []:
            g, p, i = a.get("generation"), a.get("position"), str(a.get("horse_id") or "")
            if g and g <= MAX_GEN and i:
                anc.append((h, int(g), int(p), i))
        hs.append((h, str(data.get("mating_date") or ""), str(data.get("mating_year") or ""), str(data.get("sex") or "")))
    A = pd.DataFrame(anc, columns=["horse_id", "gen", "pos", "anc_id"])
    H = pd.DataFrame(hs, columns=["horse_id", "mating_date", "mating_year", "sex"])
    CACHE.mkdir(parents=True, exist_ok=True)
    A.to_parquet(pa, index=False)
    H.to_parquet(ph, index=False)
    return A, H


def _pick(A: pd.DataFrame, gen: int, pos: int, name: str) -> pd.Series:
    s = A[(A["gen"] == gen) & (A["pos"] == pos)].drop_duplicates("horse_id").set_index("horse_id")["anc_id"]
    return s.rename(name)


def parent_ids(B: pd.DataFrame, A: pd.DataFrame) -> pd.DataFrame:
    P = pd.concat([_pick(A, 1, 0, "sire"), _pick(A, 1, 1, "dam"), _pick(A, 2, 0, "sire_sire"), _pick(A, 2, 1, "sire_dam"), _pick(A, 2, 2, "dam_sire"), _pick(A, 2, 3, "dam_dam")], axis=1)
    out = P.reindex(B["horse_id"].astype(str).to_numpy())
    out.index = B.index
    return out


def asof_sums(B: pd.DataFrame, key: pd.Series, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """key ごとの value の合計と件数を、その日より前の行だけで（同日は含めない）。key 欠損は 0。"""
    import polars as pl

    kk = key.astype(str).where(key.notna() & (key.astype(str) != ""), "__na__").to_numpy()
    base = pl.DataFrame({"_i": np.arange(len(B)), "date": pd.to_datetime(B["date"]).to_numpy().astype("datetime64[D]"), "k": kk, "v": value})
    v = pl.col("v").cast(pl.Float64)
    daily = (base.group_by(["k", "date"]).agg(v.sum().alias("s"), v.is_not_null().sum().alias("n")).sort("date")
             .with_columns(pl.col("s").cum_sum().over("k").alias("S"), pl.col("n").cum_sum().over("k").alias("N")).select("k", "date", "S", "N").sort("date"))
    j = base.sort("date").join_asof(daily, on="date", by="k", strategy="backward", allow_exact_matches=False).sort("_i")
    S = j["S"].fill_null(0.0).to_numpy().astype(float)
    N = j["N"].fill_null(0).to_numpy().astype(float)
    bad = kk == "__na__"
    S[bad], N[bad] = 0.0, 0.0
    return S, N


def _global_mean(B: pd.DataFrame, value: np.ndarray) -> np.ndarray:
    S, N = asof_sums(B, pd.Series("g", index=B.index), value)
    with np.errstate(invalid="ignore", divide="ignore"):
        return S / N


def _dist_band(B: pd.DataFrame) -> pd.Series:
    d = pd.to_numeric(B["distance"], errors="coerce")
    return pd.cut(d, [0, 1300, 1500, 1900, 2400, 99999], labels=False).astype("float").fillna(-1).astype(int).astype(str)


def _going(B: pd.DataFrame) -> pd.Series:
    return np.where(B["track_condition"].astype(str).isin(["良"]), "good", "heavy")


def ped_sire_cond(B: pd.DataFrame, A=None, H=None) -> pd.DataFrame:
    """父・母父・父父・母母父…の ID × 条件（距離帯・面・馬場・競馬場・クラス帯）の産駒成績（レース内着順スコア）の as-of 縮約と件数。"""
    A, H = build_ped_tables(B["horse_id"].unique()) if A is None else (A, H)
    P = parent_ids(B, A)
    surf = B["surface"].astype(str)
    conds = {"sd": surf + "|" + _dist_band(B), "gd": surf + "|" + _going(B), "vn": surf + "|" + B["venue_code"].astype(str), "sf": surf, "all": pd.Series("x", index=B.index)}
    res = fs.res_score(B)
    fp = pd.to_numeric(B["finish_position"], errors="coerce").to_numpy(dtype=float)
    win = np.where(fp > 0, (fp == 1).astype(float), np.nan)
    out = {}
    for role in ("sire", "dam_sire", "sire_sire", "dam_dam"):
        for cn, cv in conds.items():
            if role in ("sire_sire", "dam_dam") and cn not in ("sf", "sd"):
                continue
            key = (P[role].astype(str) + "|" + cv.astype(str)).where(P[role].notna())
            r = fs.asof_rate(B, key, res, 30.0, f"pd_{role}_{cn}_res")
            out[f"pd_{role}_{cn}_res"] = r.iloc[:, 0]
            out[f"pd_{role}_{cn}_n"] = r.iloc[:, 1]
            if cn in ("sf", "sd") and role in ("sire", "dam_sire"):
                w = fs.asof_rate(B, key, win, 30.0, f"pd_{role}_{cn}_win")
                out[f"pd_{role}_{cn}_win"] = w.iloc[:, 0]
    return pd.DataFrame(out, index=B.index).astype("float32")


def ped_nick(B: pd.DataFrame, A=None, H=None) -> pd.DataFrame:
    A, H = build_ped_tables(B["horse_id"].unique()) if A is None else (A, H)
    P = parent_ids(B, A)
    res = fs.res_score(B)
    surf = B["surface"].astype(str)
    out = {}
    key = (P["sire"].astype(str) + "|" + P["dam_sire"].astype(str)).where(P["sire"].notna() & P["dam_sire"].notna())
    r = fs.asof_rate(B, key, res, 15.0, "pn_nick")
    out["pn_nick_res"], out["pn_nick_n"] = r.iloc[:, 0], r.iloc[:, 1]
    key2 = (key.astype(str) + "|" + surf).where(key.notna())
    r2 = fs.asof_rate(B, key2, res, 15.0, "pn_nick_sf")
    out["pn_nick_sf_res"], out["pn_nick_sf_n"] = r2.iloc[:, 0], r2.iloc[:, 1]
    key3 = (P["sire_sire"].astype(str) + "|" + P["dam_sire"].astype(str)).where(P["sire_sire"].notna() & P["dam_sire"].notna())
    r3 = fs.asof_rate(B, key3, res, 15.0, "pn_gs_ds")
    out["pn_gs_ds_res"], out["pn_gs_ds_n"] = r3.iloc[:, 0], r3.iloc[:, 1]
    return pd.DataFrame(out, index=B.index).astype("float32")


def ped_dam_sib(B: pd.DataFrame, A=None, H=None) -> pd.DataFrame:
    """母の他の産駒（半きょうだい）、同じ父母の産駒（全きょうだい）の as-of 成績。自分自身の過去走は引く。"""
    A, H = build_ped_tables(B["horse_id"].unique()) if A is None else (A, H)
    P = parent_ids(B, A)
    res = fs.res_score(B)
    fp = pd.to_numeric(B["finish_position"], errors="coerce").to_numpy(dtype=float)
    win = np.where(fp > 0, (fp == 1).astype(float), np.nan)
    t3 = np.where(fp > 0, (fp <= 3).astype(float), np.nan)
    sh, nh = {}, {}
    for nm, v in (("res", res), ("win", win), ("t3", t3)):
        sh[nm], nh[nm] = asof_sums(B, B["horse_id"].astype(str), v)
    out = {}
    for tag, key in (("half", P["dam"]), ("full", (P["dam"].astype(str) + "|" + P["sire"].astype(str)).where(P["dam"].notna() & P["sire"].notna()))):
        for nm, v in (("res", res), ("win", win), ("t3", t3)):
            S, N = asof_sums(B, key, v)
            S, N = S - sh[nm], N - nh[nm]
            gm = np.nanmean(v)
            k = 8.0
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"sib_{tag}_{nm}"] = np.where(N > 0, (S + k * gm) / (N + k), np.nan)
            if nm == "res":
                out[f"sib_{tag}_n"] = N
    return pd.DataFrame(out, index=B.index).astype("float32")


def ped_blood(B: pd.DataFrame, A=None, H=None, top: int = 30) -> pd.DataFrame:
    """特定の祖先（出現の多い上位 top 頭）の血量（Σ 0.5^世代）と血の集中度・祖先の種類数。静的な特徴。"""
    A, H = build_ped_tables(B["horse_id"].unique()) if A is None else (A, H)
    A = A.assign(w=0.5 ** A["gen"].astype(float))
    cnt = A.groupby("anc_id")["horse_id"].nunique().sort_values(ascending=False)
    topids = cnt.index[:top].tolist()
    W = A[A["anc_id"].isin(topids)].pivot_table(index="horse_id", columns="anc_id", values="w", aggfunc="sum").reindex(columns=topids)
    W.columns = [f"bl_{i}" for i in range(len(topids))]
    g = A.groupby("horse_id")
    extra = pd.DataFrame({"bl_conc": g["w"].apply(lambda s: float((s.groupby(A.loc[s.index, "anc_id"]).sum() ** 2).sum())), "bl_n_distinct": g["anc_id"].nunique(), "bl_n_anc": g.size()})
    T = pd.concat([W.fillna(0.0), extra], axis=1)
    out = T.reindex(B["horse_id"].astype(str).to_numpy())
    out.index = B.index
    return out.astype("float32")


def ped_anc_apt(B: pd.DataFrame, A=None, H=None, max_gen: int = 4, k: float = 30.0) -> pd.DataFrame:
    """祖先ごとに「その祖先の他の子孫（自分を除く）」の as-of 平均成績を作り、世代ごと・重み付き全体で集約する。"""
    import polars as pl

    A, H = build_ped_tables(B["horse_id"].unique()) if A is None else (A, H)
    res = fs.res_score(B)
    Sh, Nh = asof_sums(B, B["horse_id"].astype(str), res)
    gm = _global_mean(B, res)
    base = pl.DataFrame({"_i": np.arange(len(B)), "horse_id": B["horse_id"].astype(str).to_numpy(), "date": pd.to_datetime(B["date"]).to_numpy().astype("datetime64[D]"), "v": res})
    An = pl.from_pandas(A[A["gen"] <= max_gen][["horse_id", "gen", "pos", "anc_id"]])
    L = base.join(An, on="horse_id", how="inner")                      # 行 × 祖先（最大 30）
    v = pl.col("v").cast(pl.Float64)
    daily = (L.group_by(["anc_id", "date"]).agg(v.sum().alias("s"), v.is_not_null().sum().alias("n")).sort("date")
             .with_columns(pl.col("s").cum_sum().over("anc_id").alias("S"), pl.col("n").cum_sum().over("anc_id").alias("N")).select("anc_id", "date", "S", "N").sort("date"))
    # 祖先 a の子孫のうち、その行の日付より前の結果の合計（自分自身の過去走も含むので後で引く）
    j = L.select("_i", "date", "anc_id", "gen", "pos").sort("date").join_asof(daily, on="date", by="anc_id", strategy="backward", allow_exact_matches=False)
    j = j.with_columns(pl.col("S").fill_null(0.0), pl.col("N").fill_null(0)).to_pandas()
    j["Sx"] = j["S"] - Sh[j["_i"].to_numpy()]
    j["Nx"] = j["N"] - Nh[j["_i"].to_numpy()]
    g = gm[j["_i"].to_numpy()]
    j["m"] = np.where(j["Nx"] > 0, (j["Sx"] + k * g) / (j["Nx"] + k), np.nan)
    j["n"] = j["Nx"]
    out = {}
    for gen in range(1, max_gen + 1):
        s = j[j["gen"] == gen].groupby("_i")
        out[f"aa_g{gen}_m"] = s["m"].mean()
        out[f"aa_g{gen}_max"] = s["m"].max()
        out[f"aa_g{gen}_min"] = s["m"].min()
        out[f"aa_g{gen}_n"] = s["n"].mean()
    w = 0.5 ** j["gen"].astype(float)
    ok = j["m"].notna()
    num = (j["m"].where(ok, 0.0) * w).groupby(j["_i"]).sum()
    den = w.where(ok, 0.0).groupby(j["_i"]).sum()
    out["aa_w_m"] = num / den.replace(0, np.nan)
    f = pd.DataFrame(out).reindex(np.arange(len(B)))
    f.index = B.index
    # 父・母それぞれの側（gen 2〜4 の祖先の位置が前半か後半か）
    half = j[(j["gen"] >= 2)].copy()
    half["side"] = (half["pos"] // (2 ** (half["gen"] - 1))).astype(int)
    for sd, nm in ((0, "sire_side"), (1, "dam_side")):
        s = half[half["side"] == sd].groupby("_i")["m"].mean()
        f[f"aa_{nm}"] = s.reindex(np.arange(len(B))).to_numpy()
    return f.astype("float32")


def load_profiles(horse_ids, workers: int = 4, force: bool = False) -> pd.DataFrame:
    """馬別の静的な情報（誕生日・毛色・生産地・馬主・生産者）。horse_result（保存データの読み取りだけ）から取り、キャッシュする。"""
    from concurrent.futures import ThreadPoolExecutor

    p = CACHE / "horse_profile.parquet"
    if p.exists() and not force:
        return pd.read_parquet(p)
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    from src.scraper.storage import HybridStorage

    st = HybridStorage()

    def one(h):
        try:
            d = st.load("horse_result", h) or {}
        except Exception:
            d = {}
        return (h, str(d.get("birthday") or ""), str(d.get("color") or ""), str(d.get("birthplace") or ""), str(d.get("owner") or ""), str(d.get("breeder") or ""))

    ids = sorted({str(x) for x in horse_ids})
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(one, ids))
    D = pd.DataFrame(rows, columns=["horse_id", "birthday", "color", "birthplace", "owner", "breeder"])
    CACHE.mkdir(parents=True, exist_ok=True)
    D.to_parquet(p, index=False)
    return D


def tv_birth(B: pd.DataFrame, A=None, H=None) -> pd.DataFrame:
    """誕生日（horse_result）から日齢・生まれ月・同じレースの出走馬との日齢差。静的な情報（誕生日は変わらない）。"""
    D = load_profiles(B["horse_id"].unique()).drop_duplicates("horse_id").set_index("horse_id")
    bd = pd.to_datetime(D["birthday"].astype(str).str.extract(r"(\d{4})年(\d{1,2})月(\d{1,2})日").apply(lambda r: "-".join(r.dropna().astype(str)) if r.notna().all() else None, axis=1), errors="coerce")
    m = bd.reindex(B["horse_id"].astype(str).to_numpy())
    m.index = B.index
    d = pd.to_datetime(B["date"])
    out = pd.DataFrame(index=B.index)
    out["tb_age_days"] = (d - m).dt.days
    out["tb_birth_month"] = m.dt.month
    out["tb_birth_doy"] = m.dt.dayofyear
    out["tb_age_days_vs_field"] = out["tb_age_days"] - out["tb_age_days"].groupby(B["race_id"].to_numpy()).transform("mean")
    out["tb_age_days_rank"] = out["tb_age_days"].groupby(B["race_id"].to_numpy()).rank(ascending=False)
    return out.astype("float32")


def tv_profile(B: pd.DataFrame, A=None, H=None) -> pd.DataFrame:
    """生産地（教師なしの整数コード）と、生産地ごとの as-of 成績。
    毛色は使わない: 2020 年は 99% 欠損、2025 年は 54% 欠損で、値がある行の勝率は 14%（無い行は 6%）。『最近の馬ほどプロフィールが取れている』という取得の偏りで、特徴量としては成立しない。"""
    D = load_profiles(B["horse_id"].unique()).drop_duplicates("horse_id").set_index("horse_id")
    out = {}
    for c in ("birthplace",):
        v = D[c].astype(str).str.strip().replace({"": np.nan})
        key = v.reindex(B["horse_id"].astype(str).to_numpy())
        key.index = B.index
        out[f"tp_{c}_code"] = pd.Series(pd.factorize(key)[0], index=B.index).where(key.notna())
        r = fs.asof_rates(B, key, f"tp_{c}", k=40.0)
        for cc in r.columns:
            out[cc] = r[cc]
    return pd.DataFrame(out, index=B.index).astype("float32")


GROUPS_PED = {"TV_profile": tv_profile, "PED_sire_cond": ped_sire_cond, "PED_nick": ped_nick, "PED_dam_sib": ped_dam_sib, "PED_blood": ped_blood, "PED_anc_apt": ped_anc_apt, "TV_birth": tv_birth}
