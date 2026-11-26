"""基準の走破時計の関数 T̂ = f(コース, クラス, 年齢の条件, 新馬, 頭数, 前半のペース)。馬場状態は入れない（馬場の速さを測るため）。

形（すべての変数で単調。ユーザーの指定 2026-11-26「基本は単調増加か単調減少。これを破らない」）:
    T̂ = a_c + s_cls·Σ_j B^cls_{c,j} I_j(クラス) + s_pace·Σ_j B^pace_{c,j} I_j(前半の差) + s_fld·Σ_j B^fld_{c,j} I_j(頭数)
         − (d1·[年齢≥3歳] + d2·[年齢≥古馬]) + b_new·[新馬]
- I_j は 3 次の B-spline を右から足した I-spline（C2 で滑らか・0→1 の単調。DEGREE=2 は「次数 +1 の B-spline を使う」の意味）。係数 B ≥ 0 なので、各コースの合計の曲線が必ず単調（係数の制約で保証）。
- 向き: クラスは上がるほど速い（s=−1）、前半が遅いほど遅い（s=+1）、年齢は古馬ほど速い（d≥0）。
  頭数の向きは、増と減の両方で当てはめて損失（検証の期間の残差の MAD）の小さい方に固定する（数字を見る前に決めた規則）。
- コースごとの傾き B_c は、共通の形 θ へ縮める（ridge。λ の選び方は src/research/race_time/__main__.py。v2 は学習の中で {1,5,20,100,500}）。
  水準 a_c は、芝ダ × 距離帯の水準へ縮める（λa=20）。滑らかさ: 隣の係数の差に小さい罰則（ρ=2）。
- 損失: Huber（δ = 1.345 × 最初の当てはめの残差の MAD×1.4826）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.optimize import minimize

from src.pipeline.race_time.features import distance_band

DEGREE = 2


def ispline_basis(x: np.ndarray, knots_inner: list[float], lo: float, hi: float) -> np.ndarray:
    """2 次の I-spline の基底（列は 0→1 の単調な関数）。x は [lo, hi] に切る。"""
    x = np.clip(np.asarray(x, float), lo, hi)
    t = np.r_[[lo] * (DEGREE + 2), knots_inner, [hi] * (DEGREE + 2)]   # 次数 DEGREE+1 の B-spline の節点
    k = DEGREE + 1
    n = len(t) - k - 1
    B = BSpline.design_matrix(np.clip(x, lo, hi - 1e-9), t, k).toarray()   # (len(x), n)
    I = np.cumsum(B[:, ::-1], axis=1)[:, ::-1]                         # I_j = Σ_{m≥j} B_m
    return I[:, 1:]                                                     # I_0 ≡ 1（定数）は水準と重なるので落とす


@dataclass
class Term:
    name: str
    knots: list[float]
    lo: float
    hi: float
    sign: int                      # +1: 値が大きいほど遅い / −1: 値が大きいほど速い

    def basis(self, x: np.ndarray) -> np.ndarray:
        return ispline_basis(x, self.knots, self.lo, self.hi)


@dataclass
class RaceTimeModel:
    target: str = "y_top5"
    lam: float = 100.0
    lam_level: float = 20.0
    rho: float = 2.0
    field_sign: int = 1
    monotone: bool = True          # False は比べの版（係数の下限を外す。採用しない）
    spec: str = "v1"               # v2: 節点を細かく（クラスは順位の段ごと・前半と頭数も増やす）・年齢の条件の水準をコースごとに（単調・縮める）
    terms: dict[str, Term] = field(default_factory=dict)
    courses: list[str] = field(default_factory=list)
    course_band: dict[str, str] = field(default_factory=dict)
    pace_ref: dict[str, float] = field(default_factory=dict)       # コースごとの前半 600m の中央値（前半の差の基準）
    pace_sd: dict[str, float] = field(default_factory=dict)        # コースごとの前半 600m の頑健なばらつき（ペースの区分の z）
    params: dict[str, Any] = field(default_factory=dict)
    resid_q: dict[str, list[float]] = field(default_factory=dict)  # コース（無ければ芝ダ×距離帯）ごとの残差の分位
    meta: dict[str, Any] = field(default_factory=dict)

    # ── 入力 ──
    def _setup(self, df: pd.DataFrame) -> None:
        d = df.dropna(subset=["front600"])
        self.courses = sorted(d["course"].unique())
        self.course_band = {c: f"{c.split('|')[1]}|{distance_band(int(c.split('|')[2]))}" for c in self.courses}
        g = d.groupby("course")["front600"]
        self.pace_ref = g.median().round(4).to_dict()
        self.pace_sd = (g.quantile(0.75) - g.quantile(0.25)).div(1.349).clip(lower=0.2).round(4).to_dict()
        dev = d["front600"] - d["course"].map(self.pace_ref)
        q = np.nanquantile(dev, [0.005, 0.2, 0.4, 0.6, 0.8, 0.995])
        fq = np.nanquantile(d["field_size"], [0.005, 0.995])
        if self.spec == "v2":
            pq = np.nanquantile(dev, [0.005, *np.arange(0.1, 0.91, 0.1), 0.995])
            lo_f, hi_f = float(max(5, fq[0])), float(min(18, fq[1]))
            self.terms = {
                "cls": Term("cls", [2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 1.0, 8.0, -1),
                "pace": Term("pace", [float(v) for v in pq[1:-1]], float(pq[0]), float(pq[-1]), +1),
                "fld": Term("fld", [float(k) for k in (7, 9, 11, 13, 15, 17) if lo_f < k < hi_f], lo_f, hi_f, self.field_sign),
            }
            return
        self.terms = {
            "cls": Term("cls", [2.0, 3.0, 4.0, 5.0, 6.5], 1.0, 8.0, -1),
            "pace": Term("pace", [float(v) for v in q[1:5]], float(q[0]), float(q[5]), +1),
            "fld": Term("fld", [8.0, 11.0, 14.0], float(max(5, fq[0])), float(min(18, fq[1])), self.field_sign),
        }

    def _band_of(self, course: str) -> str:
        if course in self.course_band:
            return self.course_band[course]
        v, s, dist = course.split("|")
        return f"{s}|{distance_band(int(dist))}"

    def _design(self, df: pd.DataFrame) -> dict[str, Any]:
        dev = (df["front600"] - df["course"].map(self.pace_ref).fillna(df["front600"].median())).to_numpy(float)
        X = {"cls": self.terms["cls"].basis(df["class_ord"].to_numpy(float)),
             "pace": self.terms["pace"].basis(dev),
             "fld": self.terms["fld"].basis(df["field_size"].to_numpy(float))}
        cidx = df["course"].map({c: i for i, c in enumerate(self.courses)}).fillna(-1).astype(int).to_numpy()
        age = df["age_ord"].to_numpy(int)
        return {"X": X, "cidx": cidx, "a1": (age >= 1).astype(float), "a2": (age >= 2).astype(float),
                "new": df["newcomer"].to_numpy(float), "band": df["course"].map(self._band_of).to_numpy()}

    # ── 係数の詰め替え ──
    def _shapes(self) -> list[tuple[str, tuple[int, ...]]]:
        K, bands = len(self.courses), sorted(set(self.course_band.values()))
        sh = [("lvl_band", (len(bands),)), ("lvl_dev", (K,)), ("age", (2,)), ("new", (1,))]
        if self.spec == "v2":
            sh.append(("B_age", (K, 2)))                                  # コースごとの年齢の段（≥0。共通の age へ縮める）
        for t, term in self.terms.items():
            J = self.terms[t].basis(np.array([term.lo])).shape[1]
            sh += [(f"th_{t}", (J,)), (f"B_{t}", (K, J))]
        return sh

    def _unpack(self, v: np.ndarray) -> dict[str, np.ndarray]:
        out, i = {}, 0
        for name, shp in self._shapes():
            n = int(np.prod(shp))
            out[name] = v[i:i + n].reshape(shp)
            i += n
        return out

    def _bounds(self) -> list[tuple[float | None, float | None]]:
        b: list[tuple[float | None, float | None]] = []
        for name, shp in self._shapes():
            n = int(np.prod(shp))
            nonneg = self.monotone and (name.startswith(("th_", "B_")) or name == "age")   # B_age も含む
            b += [(0.0, None) if nonneg else (None, None)] * n
        return b

    # ── 予測 ──
    def _predict_design(self, P: dict[str, np.ndarray], D: dict[str, Any]) -> np.ndarray:
        bands = sorted(set(self.course_band.values()))
        bidx = np.array([bands.index(b) if b in bands else -1 for b in D["band"]])
        c = D["cidx"]
        known = c >= 0
        lvl = np.where(bidx >= 0, P["lvl_band"][np.maximum(bidx, 0)], np.mean(P["lvl_band"])) + np.where(known, P["lvl_dev"][np.maximum(c, 0)], 0.0)
        if "B_age" in P:
            ag = np.where(known[:, None], P["B_age"][np.maximum(c, 0)], P["age"][None, :])
            pred = lvl - D["a1"] * ag[:, 0] - D["a2"] * ag[:, 1] + D["new"] * P["new"][0]
        else:
            pred = lvl - D["a1"] * P["age"][0] - D["a2"] * P["age"][1] + D["new"] * P["new"][0]
        for t, term in self.terms.items():
            coef = np.where(known[:, None], P[f"B_{t}"][np.maximum(c, 0)], P[f"th_{t}"][None, :])
            pred = pred + term.sign * np.sum(coef * D["X"][t], axis=1)
        return pred

    def predict(self, df: pd.DataFrame, parts: bool = False) -> np.ndarray | dict[str, np.ndarray]:
        P = {k: np.asarray(v, float) for k, v in self.params.items()}
        D = self._design(df)
        if not parts:
            return self._predict_design(P, D)
        D0 = dict(D)
        D0["X"] = dict(D["X"])
        D0["X"]["pace"] = self.terms["pace"].basis(np.zeros(len(df)))     # 前半の差 0（そのコースの平均的なペース）
        return {"full": self._predict_design(P, D), "no_pace": self._predict_design(P, D0)}

    # ── 当てはめ ──
    def fit(self, df: pd.DataFrame, maxiter: int = 400, init: dict[str, np.ndarray] | None = None) -> "RaceTimeModel":
        d = df.dropna(subset=["front600", self.target]).reset_index(drop=True)
        if not self.terms:
            self._setup(d)
        D = self._design(d)
        y = d[self.target].to_numpy(float)
        K = len(self.courses)
        bands = sorted(set(self.course_band.values()))
        bidx = np.array([bands.index(b) for b in D["band"]])
        C = sparse.csr_matrix((np.ones(len(d)), (np.arange(len(d)), D["cidx"])), shape=(len(d), K))
        Bm = sparse.csr_matrix((np.ones(len(d)), (np.arange(len(d)), bidx)), shape=(len(d), len(bands)))
        shapes = self._shapes()
        n_par = sum(int(np.prod(s)) for _, s in shapes)
        v0 = np.zeros(n_par)
        P0 = self._unpack(v0)
        if init:
            for k, val in init.items():
                if k in P0 and P0[k].shape == np.shape(val):
                    P0[k][...] = val
        else:
            P0["lvl_band"][...] = pd.Series(y).groupby(bidx).median().reindex(range(len(bands))).to_numpy()
            for t in self.terms:
                P0[f"th_{t}"][...] = 0.1
                P0[f"B_{t}"][...] = 0.1
            if "B_age" in P0:
                P0["B_age"][...] = 0.1
        delta = [1.0]

        def obj(v: np.ndarray) -> tuple[float, np.ndarray]:
            P = self._unpack(v)
            G = {k: np.zeros_like(a) for k, a in P.items()}
            pred = self._predict_design(P, D)
            r = y - pred
            a = np.abs(r)
            dl = delta[0]
            loss = np.where(a <= dl, 0.5 * r * r, dl * (a - 0.5 * dl)).sum()
            psi = np.clip(r, -dl, dl)
            gp = -psi                                                     # d loss / d pred
            G["lvl_band"] += Bm.T @ gp
            G["lvl_dev"] += C.T @ gp
            if "B_age" in P:
                G["B_age"][:, 0] += -(C.T @ (gp * D["a1"]))
                G["B_age"][:, 1] += -(C.T @ (gp * D["a2"]))
                da = P["B_age"] - P["age"][None, :]
                loss += self.lam * np.sum(da ** 2)
                G["B_age"] += 2 * self.lam * da
                G["age"] += -2 * self.lam * da.sum(axis=0)
            else:
                G["age"][0] += -(gp * D["a1"]).sum()
                G["age"][1] += -(gp * D["a2"]).sum()
            G["new"][0] += (gp * D["new"]).sum()
            for t, term in self.terms.items():
                G[f"B_{t}"] += term.sign * (C.T @ (gp[:, None] * D["X"][t]))
                diff = P[f"B_{t}"] - P[f"th_{t}"][None, :]
                loss += self.lam * np.sum(diff ** 2)
                G[f"B_{t}"] += 2 * self.lam * diff
                G[f"th_{t}"] += -2 * self.lam * diff.sum(axis=0)
                for key in (f"B_{t}", f"th_{t}"):                       # 隣の係数の差（滑らかさ）
                    arr = P[key]
                    dd = np.diff(arr, axis=-1)
                    loss += self.rho * np.sum(dd ** 2)
                    g = np.zeros_like(arr)
                    g[..., :-1] -= 2 * self.rho * dd
                    g[..., 1:] += 2 * self.rho * dd
                    G[key] += g
            loss += self.lam_level * np.sum(P["lvl_dev"] ** 2)
            G["lvl_dev"] += 2 * self.lam_level * P["lvl_dev"]
            return float(loss), np.concatenate([G[k].ravel() for k, _ in shapes])

        v = np.concatenate([P0[k].ravel() for k, _ in shapes])
        used_delta = delta[0]
        for stage in range(2):                                            # 1 回目の残差の MAD から δ を決めて当てはめ直す
            used_delta = delta[0]
            res = minimize(obj, v, jac=True, method="L-BFGS-B", bounds=self._bounds(), options={"maxiter": maxiter})
            v = res.x
            r = y - self._predict_design(self._unpack(v), D)
            delta[0] = 1.345 * 1.4826 * float(np.median(np.abs(r - np.median(r))))
        self.params = {k: a.tolist() for k, a in self._unpack(v).items()}
        r = y - self._predict_design(self._unpack(v), D)
        qs = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]
        rq = pd.DataFrame({"course": d["course"], "band": D["band"], "r": r})
        self.resid_q = {}
        for key, g in list(rq.groupby("course")) + list(rq.groupby("band")):
            if len(g) >= 50:
                self.resid_q[str(key)] = [round(float(x), 4) for x in np.quantile(g["r"], qs)]
        self.resid_q["_all"] = [round(float(x), 4) for x in np.quantile(r, qs)]
        self.meta.update({"n_train": int(len(d)), "converged": bool(res.success), "huber_delta": round(used_delta, 4),
                          "train_mad": round(float(np.median(np.abs(r))), 4), "n_courses": K})
        return self

    # ── 曲線（コースごとの合計の効き。単調の確かめ・図・explorer） ──
    def term_curve(self, course: str | None, t: str, x: np.ndarray) -> np.ndarray:
        """そのコースの項 t の効き（秒）。コースが学習に無ければ共通の形。"""
        term = self.terms[t]
        ci = self.courses.index(course) if course in self.courses else -1
        coef = np.asarray(self.params[f"B_{t}"][ci] if ci >= 0 else self.params[f"th_{t}"], float)
        return term.sign * term.basis(np.asarray(x, float)) @ coef

    def monotone_violations(self, n_grid: int = 200, tol: float = 1e-9) -> dict[str, dict[str, int]]:
        """全コース × 全項で、決めた向きに逆らう隣どうしの差の数（係数の制約で 0 になるはず）。年齢の項も含む。"""
        out: dict[str, dict[str, int]] = {}
        for c in [*self.courses, "_common"]:
            row = {}
            for t, term in self.terms.items():
                xs = np.linspace(term.lo, term.hi, n_grid)
                d = np.diff(self.term_curve(None if c == "_common" else c, t, xs)) * term.sign
                row[t] = int(np.sum(d < -tol))
            ci = self.courses.index(c) if c in self.courses else -1
            age = np.asarray(self.params["B_age"][ci] if ("B_age" in self.params and ci >= 0) else self.params["age"], float)
            row["age"] = int(np.sum(age < -tol))
            out[c] = row
        return out

    # ── 保存 ──
    def to_json(self) -> dict[str, Any]:
        return {
            "target": self.target, "lam": self.lam, "lam_level": self.lam_level, "rho": self.rho, "field_sign": self.field_sign, "spec": self.spec,
            "degree": DEGREE, "terms": {k: {"knots": t.knots, "lo": t.lo, "hi": t.hi, "sign": t.sign} for k, t in self.terms.items()},
            "courses": self.courses, "course_band": self.course_band, "bands": sorted(set(self.course_band.values())),
            "pace_ref": self.pace_ref, "pace_sd": self.pace_sd, "params": self.params, "resid_q": self.resid_q, "meta": self.meta,
        }

    @classmethod
    def from_json(cls, doc: dict[str, Any]) -> "RaceTimeModel":
        m = cls(target=doc["target"], lam=doc["lam"], lam_level=doc["lam_level"], rho=doc["rho"], field_sign=doc["field_sign"], spec=doc.get("spec", "v1"))
        m.terms = {k: Term(k, list(v["knots"]), float(v["lo"]), float(v["hi"]), int(v["sign"])) for k, v in doc["terms"].items()}
        m.courses, m.course_band = list(doc["courses"]), dict(doc["course_band"])
        m.pace_ref, m.pace_sd = dict(doc["pace_ref"]), dict(doc["pace_sd"])
        m.params, m.resid_q, m.meta = dict(doc["params"]), dict(doc["resid_q"]), dict(doc.get("meta") or {})
        return m

    def save(self, path) -> None:
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "RaceTimeModel":
        from pathlib import Path

        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))
