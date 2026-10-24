"""tp-35: 追い切りの『ラップ種別 A1〜B3』と『前週の土日の坂路最速時計』から作る、調教師別・条件別の成績の特徴量（as-of）。

ラップ種別（終い 1F = l1、その一つ前の 1F = l2。小数 1 位に丸めて判定）:
  A3: l1 が 12 秒未満（11 秒台以下）で加速（l1 <= l2）          B3: l2 が 12 秒未満で減速（l1 > l2）
  A2: l1・l2 ともに 12 秒台で加速                                B2: l1・l2 ともに 12 秒台で減速
  A1: l1 だけが 12 秒台で加速（l2 は 13 秒以上）                 B1: l2 だけが 12 秒台で減速（l1 は 13 秒以上）
  X : 終い 2F がともに 13 秒以上      NA: ラップが読めない
坂路は l2 = t2 − t1 で正確に出せる。ウッド・ポリは 2F の時計が載らないので l2 = (t3 − t1) / 2（2F〜3F の平均）で代用し、近似フラグを付ける。

前週の土日: レース日 D の曜日から、レースの週の月曜（月曜開催は前の週に数える）を ws とし、前の週の土曜 = ws − 2、日曜 = ws − 1。
  土曜開催の D は 7・6 日前、日曜開催の D は 8・7 日前、月曜開催の D は 9・8 日前。

すべて as-of（当該レース日より前に確定したレースの結果だけ）。回収率は過去の確定払戻の統計（現在の市場オッズ・人気は使わない）。
集計は tp28_workout_stats の as-of 累積・階層縮約（日付が厳密に前。同日の結果は入らない）を再利用する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.features import tp28_workout_stats as T

CLASSES = ["A1", "B1", "A2", "B2", "A3", "B3", "X"]
CLASS_CODE = {c: i for i, c in enumerate(CLASSES)}
SLOT_F, SLOT_W1 = (1, 6), (7, 13)
SLOPE_BINS = [50.0, 52.0, 54.0, 56.0, 58.0]            # 坂路 4F（最終追い切りの時計階級）
WOOD_BINS = [11.0, 12.0, 13.0, 14.0]                    # ウッド・ポリの終い 1F
WK_BINS = [54.0, 56.0, 58.0, 60.0]                      # 前週土日の坂路最速 4F: 〜53.9 / 54.0〜55.9 / 56.0〜57.9 / 58.0〜59.9 / 60.0〜
MIN_GLOBAL = T.MIN_GLOBAL


# ---------------------------------------------------------------- ラップ種別
def classify_lap(l1, l2) -> np.ndarray:
    """終い 1F（l1）と一つ前の 1F（l2）から A1〜B3 / X / NA を返す。"""
    a = np.round(np.asarray(l1, dtype=float), 1)
    b = np.round(np.asarray(l2, dtype=float), 1)
    ok = np.isfinite(a) & np.isfinite(b)
    accel = a <= b
    out = np.full(a.shape, "NA", dtype=object)
    in12_a = (a >= 12.0) & (a < 13.0)
    in12_b = (b >= 12.0) & (b < 13.0)
    cls = np.select(
        [(a < 12.0) & accel, (b < 12.0) & ~accel, in12_a & in12_b & accel, in12_a & in12_b & ~accel, in12_a & accel, in12_b & ~accel],
        ["A3", "B3", "A2", "B2", "A1", "B1"], default="X")
    out[ok] = cls[ok]
    return out


def _bin_edges(x: np.ndarray, edges: list[float]) -> np.ndarray:
    return np.where(np.isfinite(x), np.digitize(x, edges), -1)


def prepare_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """training_laps（1 本 1 行）に、l1・l2・近似フラグ・ラップ種別・時計階級・曜日・前週土日フラグを付ける。"""
    t = laps.copy()
    for c in ("t1", "t2", "t3", "t4"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    slope = t["kind"].eq("slope").to_numpy()
    t["l1"] = np.round(t["t1"], 1)
    l2_slope = np.round(t["t2"] - t["t1"], 1)
    l2_wood = np.round((t["t3"] - t["t1"]) / 2.0, 1)
    t["l2"] = np.where(slope, l2_slope, l2_wood)
    t["l2_approx"] = (~slope & t["kind"].isin(["wood", "poly"])).astype(int)
    ok_kind = t["kind"].isin(["slope", "wood", "poly"]).to_numpy()
    t["cls"] = np.where(ok_kind, classify_lap(t["l1"].to_numpy(), t["l2"].to_numpy()), "NA")
    t["kind_c"] = t["kind"].astype(str) + "|" + t["center"].astype(str)
    tb = np.where(slope, _bin_edges(t["t4"].to_numpy(dtype=float), SLOPE_BINS), _bin_edges(t["l1"].to_numpy(dtype=float), WOOD_BINS))
    t["tbin"] = np.where(ok_kind, tb, -1)
    t["dow"] = pd.to_datetime(t["train_date"]).dt.dayofweek
    rd = pd.to_datetime(t["race_date"])
    wd = rd.dt.weekday.to_numpy()
    wdp = np.where(wd == 0, 7, wd)
    ws = rd - pd.to_timedelta(wdp, unit="D")
    td = pd.to_datetime(t["train_date"])
    t["is_prev_weekend"] = ((td == ws - pd.Timedelta(days=2)) | (td == ws - pd.Timedelta(days=1))).astype(int)
    return t


def pick_slot(t: pd.DataFrame, lo_hi: tuple[int, int], tag: str) -> pd.DataFrame:
    """(horse_id, race_date) ごとに、日数が lo〜hi の範囲で最も新しい（レースに近い）追い切りを 1 本。"""
    s = t[t["days_before"].between(*lo_hi) & t["cls"].ne("NA")].sort_values(["horse_id", "race_date", "days_before"], kind="stable")
    g = s.groupby(["horse_id", "race_date"], sort=False).head(1)
    g = g[["horse_id", "race_date", "cls", "kind_c", "course", "tbin", "l1", "l2", "l2_approx"]].copy()
    g.columns = ["horse_id", "race_date"] + [f"{tag}_{c}" for c in g.columns[2:]]
    return g


def prev_weekend(t: pd.DataFrame) -> pd.DataFrame:
    """前週の土日の追い切りの要約: 坂路の最速 4F、ウッド・ポリの最速終い 1F、本数、最速坂路のラップ種別。"""
    w = t[t["is_prev_weekend"].eq(1)]
    if w.empty:
        return pd.DataFrame(columns=["horse_id", "race_date"])
    key = ["horse_id", "race_date"]
    sl = w[w["kind"].eq("slope") & w["t4"].notna()].sort_values(key + ["t4"], kind="stable")
    best = sl.groupby(key, sort=False).head(1)[key + ["t4", "kind_c", "cls"]].rename(columns={"t4": "wk_t4", "kind_c": "wk_kind_c", "cls": "wk_cls"})
    n_sl = sl.groupby(key).size().rename("wk_n_slope").reset_index()
    wo = w[w["kind"].isin(["wood", "poly"]) & w["l1"].notna()]
    wb = wo.groupby(key)["l1"].min().rename("wk_wood_l1").reset_index()
    n_all = w.groupby(key).size().rename("wk_n_all").reset_index()
    out = n_all.merge(best, on=key, how="left").merge(n_sl, on=key, how="left").merge(wb, on=key, how="left")
    return out


# ---------------------------------------------------------------- 集計の道具
class Agg:
    """レースごとの結果行列 M と日付 day に対する、as-of の階層縮約。"""

    def __init__(self, d: pd.DataFrame, returns: pd.DataFrame | None):
        self.d = d
        self.day = pd.to_datetime(d["date"]).to_numpy("datetime64[D]").astype(np.int64)
        fin = d["finish_position"].to_numpy(dtype=float)
        wr = returns["win_ret"].to_numpy() if returns is not None else np.full(len(d), np.nan)
        pr = returns["place_ret"].to_numpy() if returns is not None else np.full(len(d), np.nan)
        self.M = T.run_metrics(fin, d["field_size"].to_numpy(), wr, pr)
        self.has_res = np.isfinite(fin) & (fin > 0)
        self.M[~self.has_res] = 0.0
        self._gl: dict[bytes, np.ndarray] = {}

    def glob(self, valid: np.ndarray) -> np.ndarray:
        k = np.packbits(valid).tobytes()
        if k not in self._gl:
            self._gl[k] = T.global_rates(self.day, self.M, valid)
        return self._gl[k]

    def stat(self, key: pd.Series, parent: "T.Shrunk | None", glob: np.ndarray, m: float = 30.0) -> "T.Shrunk":
        valid = key.notna().to_numpy()
        codes = T._codes(key, valid)
        S = T._asof_lookup(np.where(valid & self.has_res & (codes >= 0), codes, -1), codes, self.day, self.M)
        return T.Shrunk(S, parent, glob, m=m)

    @staticmethod
    def put(store: dict, name: str, st: "T.Shrunk", glob: np.ndarray, valid_feat: np.ndarray, roi: dict | None = None) -> None:
        vf = valid_feat & (glob[:, 0] >= MIN_GLOBAL)
        for k_, col, gj in (("win", "win", 1), ("top3", "top3", 3), ("res", "res", 4)):
            store[f"{name}_{col}"] = np.where(vf, st.rates[k_] - glob[:, gj], np.nan)
        store[f"{name}_logn"] = np.where(vf, np.log1p(st.n), np.nan)
        if roi is not None:
            roi[f"{name}_roiw"] = np.where(vf, st.rates["roi_w"] - glob[:, 5], np.nan)
            roi[f"{name}_roip"] = np.where(vf, st.rates["roi_p"] - glob[:, 7], np.nan)


def _frame(d: pd.DataFrame, store: dict) -> pd.DataFrame:
    f = pd.DataFrame({k: np.asarray(v, dtype="float32") for k, v in store.items()})
    f.insert(0, "horse_id", d["horse_id"].to_numpy())
    f.insert(0, "race_id", d["race_id"].astype(str).to_numpy())
    return f


def dband(distance: pd.Series) -> pd.Series:
    x = pd.to_numeric(distance, errors="coerce")
    return pd.cut(x, [0, 1300, 1500, 1900, 2400, 99999], labels=False).astype("float").fillna(-1).astype(int).astype(str)


# ---------------------------------------------------------------- 特徴量
def build_features(runs: pd.DataFrame, laps: pd.DataFrame, returns: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """runs: race_id・horse_id・date・trainer_id・finish_position・field_size・venue_code・surface・distance（結果は as-of 集計にだけ使う）。
    laps: training_laps 相当。returns: tp28.runner_returns の結果（runs と同じ並び）。戻り値: 群名 → race_id・horse_id + 特徴量。"""
    runs = runs.reset_index(drop=True)
    t = prepare_laps(laps)
    F, W1 = pick_slot(t, SLOT_F, "F"), pick_slot(t, SLOT_W1, "W1")
    wk = prev_weekend(t)
    d = runs[["race_id", "horse_id", "date", "trainer_id", "finish_position", "field_size", "venue_code", "surface", "distance"]].copy()
    d["race_date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    d["horse_id"] = d["horse_id"].astype(str)
    for part in (F, W1, wk):
        d = d.merge(part, on=["horse_id", "race_date"], how="left")
    assert len(d) == len(runs)
    ag = Agg(d, returns)
    tid = d["trainer_id"].astype(str)
    hid = d["horse_id"]
    cond_a = d["surface"].astype(str) + "|" + dband(d["distance"])
    cond_b = d["venue_code"].astype(str) + "|" + d["surface"].astype(str) + "|" + pd.to_numeric(d["distance"], errors="coerce").fillna(-1).astype(int).astype(str)

    g_cls, g_stat, g_roi, g_horse, g_cond = {}, {}, {}, {}, {}
    # ---- A. ラップ種別（最終 F・1 週前 W1）
    for slot in ("F", "W1"):
        c = d[f"{slot}_cls"]
        g_cls[f"cls_{slot}"] = c.map(CLASS_CODE).to_numpy(dtype=float)
        g_cls[f"cls_{slot}_approx"] = np.where(c.notna(), d[f"{slot}_l2_approx"].to_numpy(dtype=float), np.nan)
    g_cls["cls_F_is_A"] = np.where(d["F_cls"].isin(["A1", "A2", "A3"]), 1.0, np.where(d["F_cls"].notna(), 0.0, np.nan))
    g_cls["cls_F_level"] = d["F_cls"].str[1].map({"1": 1.0, "2": 2.0, "3": 3.0}).to_numpy(dtype=float)
    valid_all = d["F_cls"].notna().to_numpy()
    gl = ag.glob(valid_all & ag.has_res)
    st0 = ag.stat(d["F_cls"], None, gl, m=30.0)                                    # 種別だけ（全馬）
    st_c = ag.stat(d["F_cls"] + "#" + d["F_course"].astype(str), st0, gl, m=30.0)         # 調教コース×種別
    st_t = ag.stat(tid + "#" + d["F_cls"], st0, gl, m=10.0)                       # 調教師×種別
    st_tc = ag.stat(tid + "#" + d["F_course"].astype(str) + "#" + d["F_cls"], st_t, gl, m=10.0)   # 調教師×調教コース×種別
    for nm, st in (("lg_cls", st0), ("lg_course_cls", st_c), ("lg_tr_cls", st_t), ("lg_tr_course_cls", st_tc)):
        ag.put(g_stat, nm, st, gl, valid_all, roi=g_roi if nm != "lg_cls" else None)
    # 調教師の中でその種別をどれくらいの割合で使うか（珍しい型か）
    k_tr = ag.stat(tid.where(valid_all), None, gl, m=10.0)
    g_stat["lg_tr_share"] = np.where(valid_all & (gl[:, 0] >= MIN_GLOBAL), st_t.n / np.maximum(k_tr.n, 1.0), np.nan)
    # 1 週前 W1: 種別×調教師
    vw = d["W1_cls"].notna().to_numpy()
    glw = ag.glob(vw & ag.has_res)
    st0w = ag.stat(d["W1_cls"], None, glw, m=30.0)
    st_tw = ag.stat(tid + "#" + d["W1_cls"], st0w, glw, m=10.0)
    ag.put(g_stat, "lg_w1_tr_cls", st_tw, glw, vw)
    # 馬自身×種別
    st_h = ag.stat(hid + "#" + d["F_cls"], st0, gl, m=3.0)
    ag.put(g_horse, "lg_horse_cls", st_h, gl, valid_all)
    # ---- B. レース条件×調教コース×時計階級（最終追い切り）
    vcond = d["F_tbin"].notna().to_numpy() & (d["F_tbin"].fillna(-1).to_numpy() >= 0)
    glc = ag.glob(vcond & ag.has_res)
    key_bin = (d["F_kind_c"].astype(str) + "|" + d["F_tbin"].astype("Int64").astype(str)).where(vcond)
    st_g = ag.stat(key_bin, None, glc, m=30.0)
    st_a = ag.stat(cond_a + "#" + key_bin, st_g, glc, m=30.0)
    st_b = ag.stat(cond_b + "#" + key_bin, st_a, glc, m=30.0)
    st_tb = ag.stat(tid + "#" + key_bin, st_g, glc, m=10.0)
    for nm, st in (("cd_global", st_g), ("cd_cond_band", st_a), ("cd_cond_exact", st_b), ("cd_trainer_bin", st_tb)):
        ag.put(g_cond, nm, st, glc, vcond, roi=g_roi if nm != "cd_global" else None)
    g_cond["cd_tbin"] = np.where(vcond, d["F_tbin"].to_numpy(dtype=float), np.nan)
    # ---- C. 前週の土日の坂路最速 4F
    g_wk, g_wk_roi = {}, {}
    has_wk = d["wk_n_all"].notna().to_numpy()
    wk_t4 = d["wk_t4"].to_numpy(dtype=float)
    wbin = np.where(np.isfinite(wk_t4), np.digitize(wk_t4, WK_BINS), -1)               # 0:〜53.9 … 4:60.0〜、-1: 坂路なし
    g_wk["wk_t4"] = wk_t4
    g_wk["wk_t4_bin"] = np.where(np.isfinite(wk_t4), wbin.astype(float), np.where(has_wk, -1.0, -2.0))        # -1: 土日に坂路なし（他の調教のみ）、-2: 土日に調教なし
    g_wk["wk_none_slope"] = np.where(np.isfinite(wk_t4), 0.0, 1.0)
    g_wk["wk_n_slope"] = d["wk_n_slope"].fillna(0).to_numpy(dtype=float)
    g_wk["wk_n_all"] = d["wk_n_all"].fillna(0).to_numpy(dtype=float)
    g_wk["wk_wood_l1"] = d["wk_wood_l1"].to_numpy(dtype=float)
    g_wk["wk_cls"] = d["wk_cls"].map(CLASS_CODE).to_numpy(dtype=float)
    wbin_s = pd.Series(np.where(np.isfinite(wk_t4), wbin.astype(str), "none"), index=d.index)       # 『無し』も一つの区分
    wkey = (d["wk_kind_c"].fillna("none").astype(str) + "|" + wbin_s)
    allv = np.ones(len(d), dtype=bool)
    glk = ag.glob(allv & ag.has_res)
    s_g = ag.stat(wkey, None, glk, m=30.0)                                                  # 全体（馬横断）×階級
    s_c = ag.stat(cond_a + "#" + wkey, s_g, glk, m=30.0)                                    # レース条件×階級
    s_t = ag.stat(tid + "#" + wkey, s_g, glk, m=10.0)                                       # 調教師×階級
    for nm, st in (("wk_global", s_g), ("wk_cond", s_c), ("wk_trainer", s_t)):
        ag.put(g_wk, nm, st, glk, allv, roi=g_wk_roi)
    return {"LG_cls": _frame(d, g_cls), "LG_stat": _frame(d, g_stat), "LG_roi": _frame(d, g_roi), "LG_horse": _frame(d, g_horse), "LG_cond": _frame(d, g_cond),
            "WK_stat": _frame(d, g_wk), "WK_roi": _frame(d, g_wk_roi)}
