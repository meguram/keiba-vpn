"""tp-41: 同じレースの出走馬どうしの『似ている度』（力・適性・位置取り・血統・陣営）を特徴量・ペア統計にする。

すべて『レース日より前の自分の走』と『出走表の段階で分かる情報』だけから作る（当日の結果は使わない）。
同じレースの他馬の値も、その馬のレース日より前の値なので、先読みにならない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.features import factory_self as fs

KEY = ["race_id", "horse_id"]
FRONT = 0.25          # 先行（最初のコーナーの相対位置がこれ未満）とみなす閾値
KS_POS, KS_APT_D, KS_APT_T, KS_STR = 0.15, 0.12, 0.4, 0.5
GROUPS = ["SIM_pos", "SIM_apt", "SIM_ped", "SIM_stable", "SIM_str", "SIM_prod"]


def load_megu(tables_dir, years: list[int]) -> pd.DataFrame:
    """めぐ指数 v3a の予測用列（下側圧縮）。low_confidence と障害戦は NaN。"""
    from pathlib import Path

    parts = []
    for y in years:
        f = Path(tables_dir) / str(y) / "megu_index_flat.parquet"
        if not f.exists():
            continue
        m = pd.read_parquet(f, columns=["race_id", "horse_id", "megu_index_v3a_feat", "megu_v3a_status", "megu_v3a_low_confidence"])
        ok = ~m["megu_v3a_low_confidence"].fillna(True).astype(bool) & ~m["megu_v3a_status"].astype(str).str.contains("steeple|no_par", regex=True)
        m["megu"] = m["megu_index_v3a_feat"].where(ok)
        parts.append(m[["race_id", "horse_id", "megu"]])
    m = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["race_id", "horse_id", "megu"])
    m["race_id"], m["horse_id"] = m["race_id"].astype(str), m["horse_id"].astype(str)
    return m.drop_duplicates(KEY)


def runner_table(runs: pd.DataFrame, megu: pd.DataFrame | None) -> pd.DataFrame:
    """1 走 = 1 行。列は『その走より前』の値（as-of）だけ。runs は load_runs の出力（馬・日付順）。"""
    d = fs.prep(runs)
    d["logd"] = np.log(d["dist"].clip(lower=800))
    d["turf"] = (d["surface"].astype(str) == "芝").astype(float)
    d["megu"] = np.nan
    if megu is not None and len(megu):
        d = d.drop(columns=["megu"]).merge(megu, on=KEY, how="left")
        d = d.sort_values(["horse_id", "date", "race_id"], kind="stable").reset_index(drop=True)
    out = d[KEY + ["date", "dist", "surface", "horse_number", "bracket_number", "nfield", "jockey_id", "trainer_id", "fin"]].copy()
    out["n_prior"] = fs._g(d).cumcount().to_numpy().astype(float)
    out["fp_ewm"] = fs.ewm_prior(d, "fp", 3.0).to_numpy()
    out["megu_m3"] = fs.roll_prior(d, "megu", 3, "mean").to_numpy()
    out["sty"] = fs.ewm_prior(d, "pos_first", 3.0).to_numpy()
    out["apt_logd"] = fs.roll_prior(d, "logd", 5, "mean").to_numpy()
    out["apt_turf"] = fs.roll_prior(d, "turf", 10, "mean").to_numpy()
    return out


def add_static(T: pd.DataFrame, A: pd.DataFrame) -> pd.DataFrame:
    """父・母父の ID を足す（血統は生まれたときから決まっているので as-of ではない）。A は build_ped_tables の祖先の長い表。"""
    A = A.copy()
    A["horse_id"] = A["horse_id"].astype(str)
    s = A[(A["gen"] == 1) & (A["pos"] == 0)].drop_duplicates("horse_id").set_index("horse_id")["anc_id"]
    ds = A[(A["gen"] == 2) & (A["pos"] == 2)].drop_duplicates("horse_id").set_index("horse_id")["anc_id"]
    T = T.copy()
    T["sire_id"] = T["horse_id"].map(s)
    T["dsire_id"] = T["horse_id"].map(ds)
    return T


def standardize_strength(T: pd.DataFrame) -> pd.DataFrame:
    """『強さ』= 着順位置の指数平滑（低いほど良い）の符号反転と、めぐ指数 3 走平均の標準化の平均。全期間共通の平均・標準偏差で固定
    （レースごとの標準化は他馬の値を使うので避ける）。"""
    T = T.copy()
    z1 = -(T["fp_ewm"] - 0.5) / 0.22
    z2 = (T["megu_m3"] - 42.0) / 12.0
    z = pd.concat([z1, z2], axis=1)
    T["s_fp"] = z1
    T["s_megu"] = z2
    T["s"] = z.mean(axis=1, skipna=True)
    return T


def _ped_cos(Xn, h_idx: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Xn: 行を L2 正規化した疎行列（馬 × 祖先）。h_idx は行番号（-1 は不明）。"""
    out = np.full(len(i), np.nan)
    ok = (h_idx[i] >= 0) & (h_idx[j] >= 0)
    if ok.any():
        a, b = Xn[h_idx[i[ok]]], Xn[h_idx[j[ok]]]
        out[ok] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


def race_pairs(T: pd.DataFrame, Xn=None, horse_index: pd.Index | None = None) -> pd.DataFrame:
    """レースごとの全ペア (i<j) と類似度の各軸。T は standardize_strength 済み + add_static 済みで、レース内の行が揃っていること。"""
    h_idx = horse_index.get_indexer(T["horse_id"].astype(str)) if horse_index is not None else np.full(len(T), -1)
    rows = []
    sty, logd, turf, s = T["sty"].to_numpy(float), T["apt_logd"].to_numpy(float), T["apt_turf"].to_numpy(float), T["s"].to_numpy(float)
    hn, nf = T["horse_number"].to_numpy(float), T["nfield"].to_numpy(float)
    tr, sr, ds = T["trainer_id"].astype(str).to_numpy(), T["sire_id"].astype(str).to_numpy(), T["dsire_id"].astype(str).to_numpy()
    rid = T["race_id"].to_numpy()
    starts = np.r_[0, np.flatnonzero(rid[1:] != rid[:-1]) + 1, len(rid)]
    I, J = [], []
    for a, b in zip(starts[:-1], starts[1:]):
        ii, jj = np.triu_indices(b - a, 1)
        I.append(ii + a)
        J.append(jj + a)
    i, j = np.concatenate(I), np.concatenate(J)
    P = pd.DataFrame({"i": i, "j": j, "race_id": rid[i]})
    P["d_str"] = np.abs(s[i] - s[j])
    P["d_sty"] = np.abs(sty[i] - sty[j])
    P["both_front"] = ((sty[i] < FRONT) & (sty[j] < FRONT)).astype(float)
    P.loc[np.isnan(sty[i]) | np.isnan(sty[j]), "both_front"] = np.nan
    P["d_gate"] = np.abs(hn[i] - hn[j]) / np.maximum(nf[i], 1.0)
    P["d_apt"] = np.sqrt(((logd[i] - logd[j]) / KS_APT_D) ** 2 + ((turf[i] - turf[j]) / KS_APT_T) ** 2)
    P["cos_ped"] = _ped_cos(Xn, h_idx, i, j) if Xn is not None else np.nan
    P["same_trainer"] = (tr[i] == tr[j]).astype(float)
    P["same_sire"] = ((sr[i] == sr[j]) & (sr[i] != "nan")).astype(float)
    P["same_dsire"] = ((ds[i] == ds[j]) & (ds[i] != "nan")).astype(float)
    return P


def runner_features(T: pd.DataFrame, Xn=None, horse_index: pd.Index | None = None) -> pd.DataFrame:
    """出走馬ごとの『同じレースの似た馬』の特徴量。T は 1 日分以上のレース（レース内の行が連続）。"""
    h_idx = horse_index.get_indexer(T["horse_id"].astype(str)) if horse_index is not None else np.full(len(T), -1)
    n = len(T)
    cols = {k: np.full(n, np.nan) for k in [
        "sp_front_n", "sp_front_load", "sp_own_front", "sp_same_n", "sp_rel_s", "sp_front_rel_s",
        "sa_n", "sa_rel_s", "sa_dd", "sa_turf_gap",
        "sd_close_n", "sd_max_cos", "sd_rel_s", "sd_near_n",
        "st_trainer_n", "st_trainer_rank", "st_sire_n", "st_dsire_n", "st_trainer_gap",
        "ss_dens", "ss_pct", "ss_gap_up", "ss_gap_dn", "ss_z",
        "sp_prod_n", "sp_prod_rel_s"]}
    sty, logd, turf, s = T["sty"].to_numpy(float), T["apt_logd"].to_numpy(float), T["apt_turf"].to_numpy(float), T["s"].to_numpy(float)
    tr, sr, ds = T["trainer_id"].astype(str).to_numpy(), T["sire_id"].astype(str).to_numpy(), T["dsire_id"].astype(str).to_numpy()
    rid = T["race_id"].to_numpy()
    starts = np.r_[0, np.flatnonzero(rid[1:] != rid[:-1]) + 1, n]
    for a, b in zip(starts[:-1], starts[1:]):
        m = b - a
        if m < 2:
            continue
        sl = slice(a, b)
        st, sg, sg_ok = sty[sl], s[sl], ~np.isnan(s[sl])
        eye = np.eye(m, dtype=bool)
        # ポジション
        dsty = np.abs(st[:, None] - st[None, :])
        Kp = np.exp(-0.5 * (dsty / KS_POS) ** 2)
        Kp[np.isnan(Kp)] = 0.0
        Kp[eye] = 0.0
        isf = (st < FRONT).astype(float)
        isf[np.isnan(st)] = 0.0
        cols["sp_front_n"][sl] = isf.sum() - isf
        cols["sp_front_load"][sl] = np.nansum(np.clip(0.30 - st, 0, None)) - np.nan_to_num(np.clip(0.30 - st, 0, None))
        cols["sp_own_front"][sl] = isf
        cols["sp_same_n"][sl] = Kp.sum(1)
        sgz = np.where(sg_ok, sg, 0.0)
        w = Kp * sg_ok[None, :]
        den = w.sum(1)
        cols["sp_rel_s"][sl] = np.where(den > 0.5, sg - (w @ sgz) / np.maximum(den, 1e-9), np.nan)
        wf = (isf[None, :] * (~eye)) * sg_ok[None, :]
        cols["sp_front_rel_s"][sl] = np.where(wf.sum(1) > 0, sg - (wf @ sgz) / np.maximum(wf.sum(1), 1e-9), np.nan)
        # 適性
        da = np.sqrt(((logd[sl][:, None] - logd[sl][None, :]) / KS_APT_D) ** 2 + ((turf[sl][:, None] - turf[sl][None, :]) / KS_APT_T) ** 2)
        Ka = np.exp(-0.5 * da ** 2)
        Ka[np.isnan(Ka)] = 0.0
        Ka[eye] = 0.0
        cols["sa_n"][sl] = Ka.sum(1)
        w = Ka * sg_ok[None, :]
        den = w.sum(1)
        cols["sa_rel_s"][sl] = np.where(den > 0.5, sg - (w @ sgz) / np.maximum(den, 1e-9), np.nan)
        cols["sa_dd"][sl] = logd[sl] - np.nanmean(logd[sl])
        cols["sa_turf_gap"][sl] = turf[sl] - np.nanmean(turf[sl])
        # 血統
        if Xn is not None and (h_idx[sl] >= 0).sum() >= 2:
            hh = h_idx[sl]
            C = np.full((m, m), np.nan)
            okh = hh >= 0
            sub = Xn[hh[okh]]
            Cs = (sub @ sub.T).toarray()
            idx = np.flatnonzero(okh)
            C[np.ix_(idx, idx)] = Cs
            C[eye] = np.nan
            cols["sd_max_cos"][sl] = np.where(okh, np.nanmax(np.where(np.isnan(C), -1.0, C), axis=1), np.nan)
            close = np.nan_to_num(C, nan=0.0) > 0.25
            cols["sd_close_n"][sl] = np.where(okh, close.sum(1), np.nan)
            cols["sd_near_n"][sl] = np.where(okh, (np.nan_to_num(C, nan=0.0) ** 2).sum(1), np.nan)
            wc = (np.nan_to_num(C, nan=0.0) ** 2) * sg_ok[None, :]
            den = wc.sum(1)
            cols["sd_rel_s"][sl] = np.where(okh & (den > 0.05), sg - (wc @ sgz) / np.maximum(den, 1e-9), np.nan)
        # 陣営・血縁
        t_ = tr[sl]
        same_t = (t_[:, None] == t_[None, :]) & ~eye
        cols["st_trainer_n"][sl] = same_t.sum(1)
        better = (same_t & sg_ok[None, :] & (np.where(sg_ok, sg, -9)[None, :] > np.where(sg_ok, sg, -9)[:, None])).sum(1)
        cols["st_trainer_rank"][sl] = np.where(same_t.sum(1) > 0, better, np.nan)
        gap = np.where(same_t & sg_ok[None, :], sgz[None, :] - sgz[:, None], np.nan)
        cols["st_trainer_gap"][sl] = np.where(same_t.sum(1) > 0, np.nanmax(np.where(np.isnan(gap), -9, gap), axis=1), np.nan)
        sr_ = sr[sl]
        cols["st_sire_n"][sl] = (((sr_[:, None] == sr_[None, :]) & (sr_[:, None] != "nan")) & ~eye).sum(1)
        ds_ = ds[sl]
        cols["st_dsire_n"][sl] = (((ds_[:, None] == ds_[None, :]) & (ds_[:, None] != "nan")) & ~eye).sum(1)
        # 力の近さ（周りの密度）
        dd = np.abs(sg[:, None] - sg[None, :])
        dd[eye] = np.nan
        cols["ss_dens"][sl] = np.where(sg_ok, (np.nan_to_num(dd, nan=9.0) < 0.25).sum(1), np.nan)
        cols["ss_pct"][sl] = np.where(sg_ok, (sg_ok[None, :] & (sg[None, :] < sg[:, None])).sum(1) / max(sg_ok.sum() - 1, 1), np.nan)
        up = np.where(sg_ok[None, :] & (sg[None, :] > sg[:, None]), sg[None, :] - sg[:, None], np.inf).min(1)
        dn = np.where(sg_ok[None, :] & (sg[None, :] < sg[:, None]), sg[:, None] - sg[None, :], np.inf).min(1)
        cols["ss_gap_up"][sl] = np.where(sg_ok & np.isfinite(up), up, np.nan)
        cols["ss_gap_dn"][sl] = np.where(sg_ok & np.isfinite(dn), dn, np.nan)
        if sg_ok.sum() >= 3:
            cols["ss_z"][sl] = np.where(sg_ok, (sg - np.nanmean(sg)) / (np.nanstd(sg) + 1e-6), np.nan)
        # 積カーネル（位置 × 適性 × 力）
        Kall = Kp * Ka * np.exp(-0.5 * (dd / KS_STR) ** 2)
        Kall[np.isnan(Kall)] = 0.0
        cols["sp_prod_n"][sl] = Kall.sum(1)
        w = Kall * sg_ok[None, :]
        den = w.sum(1)
        cols["sp_prod_rel_s"][sl] = np.where(den > 0.3, sg - (w @ sgz) / np.maximum(den, 1e-9), np.nan)
    F = pd.DataFrame(cols, index=T.index)
    F.insert(0, "horse_id", T["horse_id"].to_numpy())
    F.insert(0, "race_id", T["race_id"].to_numpy())
    return F


GROUP_COLS = {
    "SIM_pos": ["sp_front_n", "sp_front_load", "sp_own_front", "sp_same_n", "sp_rel_s", "sp_front_rel_s"],
    "SIM_apt": ["sa_n", "sa_rel_s", "sa_dd", "sa_turf_gap"],
    "SIM_ped": ["sd_close_n", "sd_max_cos", "sd_rel_s", "sd_near_n"],
    "SIM_stable": ["st_trainer_n", "st_trainer_rank", "st_sire_n", "st_dsire_n", "st_trainer_gap"],
    "SIM_str": ["ss_dens", "ss_pct", "ss_gap_up", "ss_gap_dn", "ss_z"],
    "SIM_prod": ["sp_prod_n", "sp_prod_rel_s"],
}


def sort_for_races(T: pd.DataFrame) -> pd.DataFrame:
    """レース内の行を連続にし、馬番順に並べる。"""
    return T.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)
