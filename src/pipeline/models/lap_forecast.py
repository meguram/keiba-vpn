"""tp-55: ラップ傾向（シナリオ）別のハロンごとのタイム予測と、各シナリオの発生確率（混合分布）。

レースごとに
  P(k | 特徴量)        … シナリオ k（スロー/ミドル/ハイ、または形状クラスタ）の発生確率（LightGBM 多クラス＋温度較正）
  lap_j | k, 特徴量   … シナリオ k のときの j 番目のラップ（j = スタートから数えた 0 始まり。最初のラップが『テン1F』。
                          1700m のように 200m で割り切れない距離は最初のラップが短い区間（d0 m）になる）
の混合として出す。シナリオごと・ラップごとに LightGBM 回帰（par = コース別の基準ラップからの偏差を予測）。
不確実性は、3 分割の OOF 残差から求めたシナリオ×ラップごとの標準偏差と、シナリオ間のばらつきの和。
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features import tp55_lapfeat as lf

MAXJ = 14                    # スタートから数えて回帰モデルを持つラップの数（2600m まで全ラップ）
MIN_CELL = 120               # シナリオ×ラップの学習件数がこれ未満なら平均で代用
CAT_COLS = ["venue", "surface", "direction", "weather", "track_condition", "grade", "race_class", "course_key"]
EXTRA_NUM = ["distance", "n_laps", "d0", "kai", "nichime", "round_no", "month", "field_size"]
SD_COLS = ["sd_n", "sd_q_mean", "sd_f3_mean", "sd_l3_mean", "sd_q_last"]
SCEN_NAMES_A = ["スロー", "ミドル", "ハイ"]


def feature_columns(drop_groups: tuple[str, ...] = ()) -> tuple[list[str], list[str]]:
    num: list[str] = []
    for g, cols in lf.FEATURE_GROUPS.items():
        if g in drop_groups:
            continue
        num += [c for c in cols if c not in ("course_code", "venue_code_c", "surface_c", "direction_c", "weather_c", "track_c", "grade_c", "class_c")]
    num = list(dict.fromkeys(num + [c for c in EXTRA_NUM if "course" not in drop_groups]))
    cats = [] if "course" in drop_groups else [c for c in CAT_COLS if c in ("venue", "surface", "direction", "course_key")]
    if "cond" not in drop_groups:
        cats += ["weather", "track_condition", "grade", "race_class"]
    return num, cats


@dataclass
class Vocab:
    cats: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, df: pd.DataFrame, cols: list[str]) -> "Vocab":
        return cls({c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in cols})

    def apply(self, df: pd.DataFrame, num: list[str], cats: list[str]) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        for c in num:
            X[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
        for c in cats:
            X[c] = pd.Categorical(df[c].astype(str).where(df[c].notna()), categories=self.cats.get(c, []))
        return X


def _lgb_reg(threads: int, seed: int):
    import lightgbm as lgb

    return lgb.LGBMRegressor(n_estimators=150, learning_rate=0.06, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.7, reg_lambda=3.0, n_jobs=threads, verbose=-1, random_state=seed, min_data_per_group=30,
                             cat_smooth=20)


def _lgb_clf(threads: int, seed: int, k: int):
    import lightgbm as lgb

    return lgb.LGBMClassifier(objective="multiclass", num_class=k, n_estimators=250, learning_rate=0.04, num_leaves=15, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, n_jobs=threads, verbose=-1,
                              random_state=seed, min_data_per_group=30, cat_smooth=20)


def laps_matrix(laps_list: list[list[float]], width: int = lf.MAX_LAPS) -> np.ndarray:
    M = np.full((len(laps_list), width), np.nan)
    for i, a in enumerate(laps_list):
        M[i, : len(a)] = a
    return M


def par_matrix(df: pd.DataFrame, par: lf.Par, width: int = lf.MAX_LAPS) -> np.ndarray:
    M = np.full((len(df), width), np.nan)
    for i, (v, s, d, di) in enumerate(zip(df["venue"], df["surface"], df["distance"], df["direction"])):
        p = par.vec(v, s, d, di)
        if p is not None:
            M[i, : len(p)] = p
    return M


def scenario_stats(dev: np.ndarray, n_laps: np.ndarray) -> np.ndarray:
    """偏差行列 → [前半3F の偏差, 中盤の平均偏差, 後半3F の偏差]。"""
    n = len(dev)
    out = np.full((n, 3), np.nan)
    for i in range(n):
        m = int(n_laps[i])
        d = dev[i, :m]
        f3 = d[:3].sum()
        l3 = d[-3:].sum()
        mid = d[3:-3].mean() if m > 6 else 0.0
        out[i] = (f3, mid, l3)
    return out


class LapForecast:
    """scenario='A'（前半-後半の差の分位でスロー/ミドル/ハイ）または 'B'（形状クラスタ K 個）。"""

    def __init__(self, scenario: str = "A", k_cluster: int = 5, drop_groups: tuple[str, ...] = (), threads: int = 3, seed: int = 0,
                 n_oof: int = 3):
        self.scenario = scenario
        self.k_cluster = k_cluster
        self.drop_groups = tuple(drop_groups)
        self.threads = threads
        self.seed = seed
        self.n_oof = n_oof
        self.num, self.cats = feature_columns(self.drop_groups)

    # ───────── ラベル
    def _labels(self, df: pd.DataFrame, dev: np.ndarray, n_laps: np.ndarray, fit: bool) -> np.ndarray:
        st = scenario_stats(dev, n_laps)
        if self.scenario == "A":
            q = st[:, 2] - st[:, 0]
            if fit:
                self.q_lo, self.q_hi = np.nanpercentile(q, [25, 75])
            lab = np.where(q <= self.q_lo, 0, np.where(q >= self.q_hi, 2, 1))
            return lab
        if fit:
            from sklearn.cluster import KMeans

            self.st_mu = np.nanmean(st, axis=0)
            self.st_sd = np.nanstd(st, axis=0) + 1e-9
            km = KMeans(n_clusters=self.k_cluster, n_init=10, random_state=self.seed).fit((st - self.st_mu) / self.st_sd)
            # クラスタを『後半3F−前半3F』の大きい順（=ハイペース寄り）に並べ替える
            cen = km.cluster_centers_ * self.st_sd + self.st_mu
            order = np.argsort(cen[:, 2] - cen[:, 0])
            remap = np.empty(self.k_cluster, dtype=int)
            remap[order] = np.arange(self.k_cluster)
            self.km = km
            self.km_remap = remap
            self.cen_sorted = cen[order]
        raw = self.km.predict((st - self.st_mu) / self.st_sd)
        return self.km_remap[raw]

    def scenario_names(self) -> list[str]:
        if self.scenario == "A":
            return SCEN_NAMES_A
        return [f"形状{k+1}" for k in range(self.k_cluster)]

    @property
    def K(self) -> int:
        return 3 if self.scenario == "A" else self.k_cluster

    # ───────── 学習
    def fit(self, df: pd.DataFrame) -> "LapForecast":
        from sklearn.model_selection import GroupKFold

        df = df.reset_index(drop=True)
        laps = [json.loads(s) for s in df["laps_json"]]
        n_laps = np.array([len(a) for a in laps])
        L = laps_matrix(laps)
        self.par = lf.fit_par(pd.DataFrame({"venue": df["venue"], "surface": df["surface"], "distance": df["distance"],
                                            "direction": df["direction"], "laps": laps}))
        P = par_matrix(df, self.par)
        dev = L - P
        ok = ~np.isnan(P[np.arange(len(df)), 0]) & ~np.isnan(dev[np.arange(len(df)), 0])
        df, dev, n_laps, L = df[ok].reset_index(drop=True), dev[ok], n_laps[ok], L[ok]
        lab = self._labels(df, dev, n_laps, fit=True)
        self.prior = np.bincount(lab, minlength=self.K) / len(lab)
        self.vocab = Vocab.fit(df, self.cats)
        X = self.vocab.apply(df, self.num, self.cats)
        groups = df["race_id"].values
        gkf = GroupKFold(n_splits=self.n_oof)
        folds = list(gkf.split(X, lab, groups))
        # シナリオ分類器（OOF の対数オッズで温度を決める）
        K = self.K
        oof = np.zeros((len(df), K))
        for tr, va in folds:
            m = _lgb_clf(self.threads, self.seed, K).fit(X.iloc[tr], lab[tr])
            p = np.full((len(va), K), 1e-6)
            for ci, c in enumerate(m.classes_):
                p[:, c] = m.predict_proba(X.iloc[va])[:, ci]
            oof[va] = p
        self.clf = _lgb_clf(self.threads, self.seed, K).fit(X, lab)
        self.temp = self._fit_temperature(oof, lab)
        # シナリオ×ラップの回帰
        self.reg: dict = {}
        self.sd: dict = {}
        self.mean: dict = {}
        for k in range(K):
            idx_k = np.where(lab == k)[0]
            for j in range(MAXJ):
                rows = idx_k[n_laps[idx_k] > j]
                if len(rows) < 8:
                    continue
                y = dev[rows, j]
                self.mean[(k, j)] = float(np.mean(y))
                if len(rows) < MIN_CELL:
                    self.sd[(k, j)] = float(np.std(y)) if len(y) > 1 else 1.0
                    continue
                Xr = X.iloc[rows]
                oofp = np.zeros(len(rows))
                gk = GroupKFold(n_splits=min(self.n_oof, len(set(groups[rows]))))
                for tr, va in gk.split(Xr, y, groups[rows]):
                    oofp[va] = _lgb_reg(self.threads, self.seed).fit(Xr.iloc[tr], y[tr]).predict(Xr.iloc[va])
                self.sd[(k, j)] = float(np.std(y - oofp))
                self.reg[(k, j)] = _lgb_reg(self.threads, self.seed).fit(Xr, y)
        # 直接回帰（シナリオなし）のベースライン用
        self.direct: dict = {}
        self.direct_sd: dict = {}
        for j in range(MAXJ):
            rows = np.where(n_laps > j)[0]
            if len(rows) < MIN_CELL:
                continue
            y = dev[rows, j]
            Xr = X.iloc[rows]
            oofp = np.zeros(len(rows))
            gk = GroupKFold(n_splits=self.n_oof)
            for tr, va in gk.split(Xr, y, groups[rows]):
                oofp[va] = _lgb_reg(self.threads, self.seed).fit(Xr.iloc[tr], y[tr]).predict(Xr.iloc[va])
            self.direct_sd[j] = float(np.std(y - oofp))
            self.direct[j] = _lgb_reg(self.threads, self.seed).fit(Xr, y)
        self.q_scale = None
        return self

    @staticmethod
    def _fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
        lp = np.log(np.clip(p, 1e-9, 1))
        best, bt = 1e9, 1.0
        for t in np.linspace(0.6, 1.8, 25):
            z = lp / t
            z = z - z.max(axis=1, keepdims=True)
            q = np.exp(z)
            q /= q.sum(axis=1, keepdims=True)
            ll = -np.mean(np.log(np.clip(q[np.arange(len(y)), y], 1e-9, 1)))
            if ll < best:
                best, bt = ll, float(t)
        return bt

    # ───────── 予測
    def predict(self, df: pd.DataFrame) -> dict:
        df = df.reset_index(drop=True)
        n = len(df)
        K = self.K
        X = self.vocab.apply(df, self.num, self.cats)
        n_laps = df["distance"].map(lf.n_laps_of).values
        P = par_matrix(df, self.par)
        proba = np.full((n, K), 1e-6)
        pr = self.clf.predict_proba(X)
        for ci, c in enumerate(self.clf.classes_):
            proba[:, c] = pr[:, ci]
        z = np.log(np.clip(proba, 1e-9, 1)) / self.temp
        z -= z.max(axis=1, keepdims=True)
        proba = np.exp(z)
        proba /= proba.sum(axis=1, keepdims=True)
        mu = np.full((n, K, MAXJ), np.nan)
        sd = np.full((K, MAXJ), np.nan)
        for k in range(K):
            for j in range(MAXJ):
                if (k, j) in self.reg:
                    mu[:, k, j] = self.reg[(k, j)].predict(X)
                elif (k, j) in self.mean:
                    mu[:, k, j] = self.mean[(k, j)]
                if (k, j) in self.sd:
                    sd[k, j] = self.sd[(k, j)]
        direct = np.full((n, MAXJ), np.nan)
        for j, m in self.direct.items():
            direct[:, j] = m.predict(X)
        for j in range(MAXJ):                                   # 長い距離で存在しないラップ
            miss = n_laps <= j
            mu[miss, :, j] = np.nan
            direct[miss, j] = np.nan
        return {"proba": proba, "mu_dev": mu, "sd": sd, "direct_dev": direct, "par": P[:, :MAXJ], "n_laps": n_laps, "temp": self.temp}

    # ───────── 保存
    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "LapForecast":
        with open(path, "rb") as f:
            return pickle.load(f)


def mixture_summary(out: dict, i: int, names: list[str], n_draw: int = 400, seed: int = 0) -> dict:
    """予測結果 out の i 番目のレースを API 用の dict にする（par + 偏差 = 秒）。

    scenarios[k]: 発生確率、ハロンごとの予想タイム（スタートから。最初が『テン1F』）、前半3F・後半3F。
    expected_laps: 確率で重み付けした期待ラップ。band80_lo/hi: 混合からのサンプリングによる 10%・90% 点（ラップ内は独立と仮定）。
    MAXJ より長いラップ（3000m 以上）は回帰を持たないので、par にシナリオの平均偏差が無い分は par のまま（band なし）。
    """
    P = out["par"][i]
    m = int(out["n_laps"][i])
    K = out["proba"].shape[1]
    J = min(m, MAXJ)
    pr = out["proba"][i]
    rng = np.random.default_rng(seed)
    mu_k = np.full((K, m), np.nan)
    sd_k = np.full((K, m), np.nan)
    for k in range(K):
        d = out["mu_dev"][i, k, :J]
        mu_k[k, :J] = P[:J] + np.where(np.isnan(d), 0.0, d)
        sd_k[k, :J] = out["sd"][k, :J]
        if m > J:
            mu_k[k, J:] = np.nan
    scen = []
    for k in range(K):
        v = mu_k[k]
        f3 = float(np.nansum(v[:3])) if not np.isnan(v[:3]).any() else None
        l3 = float(np.nansum(v[m - 3:m])) if not np.isnan(v[m - 3:m]).any() else None
        scen.append({"name": names[k], "prob": round(float(pr[k]), 4), "laps": [None if np.isnan(x) else round(float(x), 2) for x in v],
                     "sd": [None if np.isnan(x) else round(float(x), 2) for x in sd_k[k]],
                     "first3f": None if f3 is None else round(f3, 2), "last3f": None if l3 is None else round(l3, 2)})
    w = pr[:, None]
    mean = np.nansum(w * mu_k, axis=0) / np.where(np.sum(w * ~np.isnan(mu_k), axis=0) > 0, np.sum(w * ~np.isnan(mu_k), axis=0), np.nan)
    pick = np.minimum((rng.random(n_draw)[:, None] > np.cumsum(pr)[None, :]).sum(axis=1), K - 1)
    lo, hi = [], []
    for j in range(m):
        if j >= J or np.isnan(mu_k[pick, j]).any() or np.isnan(sd_k[pick, j]).any():
            lo.append(None); hi.append(None); continue
        x = mu_k[pick, j] + rng.standard_normal(n_draw) * sd_k[pick, j]
        a, b = np.quantile(x, [0.1, 0.9])
        lo.append(round(float(a), 2)); hi.append(round(float(b), 2))
    top = int(np.argmax(pr))
    ef3 = float(np.nansum(mean[:3])) if not np.isnan(mean[:3]).any() else None
    el3 = float(np.nansum(mean[m - 3:m])) if not np.isnan(mean[m - 3:m]).any() else None
    return {"n_laps": m, "scenarios": scen, "expected_laps": [None if np.isnan(x) else round(float(x), 2) for x in mean],
            "band80_lo": lo, "band80_hi": hi, "expected_first3f": None if ef3 is None else round(ef3, 2),
            "expected_last3f": None if el3 is None else round(el3, 2), "likely_scenario": names[top], "likely_prob": round(float(pr[top]), 4)}
