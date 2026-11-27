"""血統構成から、種牡馬の父としての主の 6 本（地図の位置）を当てるモデル（記述。作りは log.md「予測のモデルの固定」）。

    KEIBA_SESSION=aptitude-group OMP_NUM_THREADS=2 flock /tmp/claude-1000/heavy.lock nice -n 19 python -m src.research.aptitude_group.predict

- 入力: 父の (A)（確かめでは S を通る直系の走りを抜いた一つ抜き）・母父の母父としてと父としての 6 本・母の側 3〜5 代の牡の祖先の (B) の血量の平均・
  大系統の札・5 代の中の重複の血量。産駒の成績（目標そのもの）は使わない。
- 次元ごとのリッジ回帰。父の単位の群で 5 分割の交差検証。出力は common.OUT/predict.json（確かめの数・予測・混ぜた値・残りの共分散）。
- 学習と推論で同じ入力の作り: 父の (A) は、学習の目標の種牡馬にも、50 走未満の種牡馬にも、同じ一つ抜き（father_a_loo）で作る。
  推論は predict_vector（保存したモデルの係数）で行う。assign の事前への組み込みはまだ（記録だけ）。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from src.research.aptitude_group import common
from src.research.aptitude_group.line_vectors import LINE_POS, build, long_frame, pedigree_links, zfix
from src.research.aptitude_group.lines import line_labels
from src.research.aptitude_group.sire_vectors import DIMS, K, MIN_RUNS, PED5, names, runs, shrink, stats, stats_base
from src.research.aptitude_group.sire_vectors_html import line_of
from src.research.pedigree_lab import data
from src.research.pedigree_lab.aptitude import MAT_POS

M6 = ["dist", "surf", "wet", "pos", "ts", "gear"]
ALPHAS = [0.1, 1.0, 10.0, 100.0]
CHI2_6_95 = 12.592


def m6(z: pd.Series | dict) -> np.ndarray:
    g = (lambda k: z[k])
    return np.array([(g("wet_T") + g("wet_D")) / 2 if d == "wet" else g(d) for d in M6], float)


def own_pedigree(h: str) -> dict[tuple[int, int], str]:
    p = PED5 / h[:4] / f"{h}.json"
    if not p.exists():
        return {}
    try:
        return {(a["generation"], a["position"]): a["horse_id"] for a in json.loads(p.read_text(encoding="utf-8")).get("ancestors", []) if a.get("horse_id")}
    except (OSError, ValueError, KeyError):
        return {}


def inbreeding(ped: dict[tuple[int, int], str]) -> float:
    cnt: dict[str, list[int]] = {}
    for (g, _), i in ped.items():
        cnt.setdefault(i, []).append(g)
    return float(sum(sum(0.5 ** g for g in gs) for gs in cnt.values() if len(gs) > 1))


class Inputs:
    """入力の作り（学習と推論で同じ）。A・B は z の 6 本の辞書、sire・bms は sire_vectors の z の 6 本の辞書。"""

    def __init__(self, A: dict, B: dict, SIRE: dict, BMS: dict, lines: list[str], lab: dict[str, str], parent: dict[str, str]):
        self.A, self.B, self.SIRE, self.BMS, self.lines, self.lab, self.parent = A, B, SIRE, BMS, lines, lab, parent

    def row(self, h: str, father_a: np.ndarray | None = None) -> dict:
        ped = own_pedigree(h)
        f = ped.get((1, 0))
        out = {}
        fa = father_a                                  # 父の (A) は呼ぶ側が一つ抜きで作って渡す（学習と推論で同じ・critic 5 回目）
        out |= {f"fa_{d}": (fa[i] if fa is not None else 0.0) for i, d in enumerate(M6)} | {"fa_miss": float(fa is None)}
        bms = ped.get((2, 2))
        for nm, src in (("bb", self.BMS), ("bs", self.SIRE)):
            v = src.get(bms)
            out |= {f"{nm}_{d}": (v[i] if v is not None else 0.0) for i, d in enumerate(M6)} | {f"{nm}_miss": float(v is None)}
        acc, wsum = np.zeros(6), 0.0
        for g in (3, 4, 5):
            for p in MAT_POS[g]:
                v = self.B.get(ped.get((g, p)))
                if v is not None:
                    acc += 0.5 ** g * v
                    wsum += 0.5 ** g
        out |= {f"mb_{d}": (acc[i] / wsum if wsum else 0.0) for i, d in enumerate(M6)} | {"mb_miss": float(wsum == 0)}
        ln = line_of(h, self.lab) or ""
        out |= {f"line_{x}": float(ln == x) for x in self.lines}
        out["inbreed"] = inbreeding(ped)
        out["no_ped"] = float(not ped)
        return out


def father_a_loo(h: str, f: str | None, L: pd.DataFrame, base: dict, Vfull: pd.DataFrame, parent: dict, amu: dict, asd: dict) -> np.ndarray | None:
    """種牡馬 h の父 f の (A) を、h を通る直系の走り（h 自身・h の産駒・h の父系の孫）を抜いて作り直す（z の 6 本）。
    父の父 gf の値も同じく抜いて作り直し、gf はその父の全体の値へ、f は gf へ縮める。基準は全体の L から（base）。"""
    if f is None:
        return None
    keep = (L["horse_id"] != h) & (L["p1"] != h) & (L["p2"] != h)

    def raw(key, prior):
        sub = L[(L["key"] == key) & keep]
        if not len(sub):
            return None
        S = stats(sub, "key", base=base)
        if key not in S.index:
            return None
        return {d: (S.loc[key, ("d", d)] + K * prior.get(d, 0.0)) / (S.loc[key, ("n", d)] + K) for d in DIMS}
    gf = parent.get(f)
    gff = parent.get(gf) if gf else None
    pg = {d: float(Vfull.loc[gff, d]) for d in DIMS} if gff in Vfull.index else {}
    vg = raw(gf, pg) if gf else None
    vf = raw(f, vg or {})
    if vf is None:
        return None
    return m6({d: (vf[d] - amu[d]) / asd[d] for d in DIMS})


def predict_vector(x_row: dict, model: dict) -> np.ndarray:
    """保存したモデル（predict.json の model）で、入力の行（Inputs.row の辞書）から 6 本を出す。"""
    x = (np.array([x_row.get(c, 0.0) for c in model["cols"]], float) - np.array(model["mu"])) / np.array(model["sd"])
    return np.array([model["dims"][d]["intercept"] + float(np.dot(model["dims"][d]["coef"], x)) for d in M6])


def fit_dim(X: np.ndarray, y: np.ndarray, w: np.ndarray, groups: np.ndarray) -> float:
    """学習の側の中の 5 分割（父の群）で α を選ぶ。"""
    best, best_e = ALPHAS[0], np.inf
    gk = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for a in ALPHAS:
        e = 0.0
        for tr, te in gk.split(X, y, groups):
            m = Ridge(alpha=a).fit(X[tr], y[tr], sample_weight=w[tr])
            e += float(np.sum(w[te] * (m.predict(X[te]) - y[te]) ** 2))
        if e < best_e:
            best, best_e = a, e
    return best


def wcorr(a, b, w):
    am, bm = np.average(a, weights=w), np.average(b, weights=w)
    return float(np.sum(w * (a - am) * (b - bm)) / np.sqrt(np.sum(w * (a - am) ** 2) * np.sum(w * (b - bm) ** 2)))


def main() -> None:
    common.open_token("select", "predict_v1", "血統構成から種牡馬の 6 本を当てるモデル（記述・開発の行 2016〜24）")
    sv = json.loads((common.OUT / "sire_vectors.json").read_text())
    lv = json.loads((common.OUT / "line_vectors.json").read_text())
    dims = sv["dims"]
    SIRE = {r["id"]: m6(dict(zip(dims, r["v"]))) for r in sv["sire"]["rows"]}
    BMS = {r["id"]: m6(dict(zip(dims, r["v"]))) for r in sv["bms"]["rows"]}
    A = {h: m6(dict(zip(dims, x["v"]))) for h, x in lv["A"].items()}
    Bv = {h: m6(dict(zip(dims, x["v"]))) for h, x in lv["B"].items()}
    target = {r["id"]: r for r in sv["sire"]["rows"]}
    core = {r["id"] for r in sv["sire"]["rows"] if r.get("core", True)}
    lab = line_labels(core)
    F = runs()
    P, parent = pedigree_links(set(F["horse_id"]))
    lines = sorted({line_of(h, lab) or "" for h in target} - {""})
    inp = Inputs(A, Bv, SIRE, BMS, lines, lab, parent)
    # 父の (A) の一つ抜き: S を通る直系（S 自身・S の産駒・S の孫）の行を抜いて、父の鍵だけで集計し直す
    Fa = F[["horse_id", "meeting", "gc", "surf", "venue", "distance", "going", "course_type", "direction", "early", "ts", "lx", "acc_z", "gain_r"]].copy()
    L = long_frame(Fa, P, LINE_POS)
    pos = P[P["gen"].isin([1, 2]) & (P["pos"] == 0)].pivot_table(index="horse_id", columns="gen", values="anc", aggfunc="first")
    L["p1"], L["p2"] = L["horse_id"].map(pos.get(1, pd.Series(dtype=object))), L["horse_id"].map(pos.get(2, pd.Series(dtype=object)))
    Vfull = build(Fa, P, parent, LINE_POS)
    Vshow = Vfull[Vfull["runs"] >= 50]
    _, amu, asd = zfix(Vshow)
    base = stats_base(L)
    rows, ids, fathers = [], [], []
    for h in target:
        f = own_pedigree(h).get((1, 0))
        fa = father_a_loo(h, f, L, base, Vfull, parent, amu, asd)
        rows.append(inp.row(h, father_a=fa))
        ids.append(h)
        fathers.append(f or h)
    X = pd.DataFrame(rows, index=ids).fillna(0.0)
    cols = list(X.columns)
    Y = np.stack([SIRE[h] for h in ids])
    w = np.array([target[h]["runs"] / (target[h]["runs"] + K) for h in ids])
    groups = pd.factorize(pd.Series(fathers))[0]
    Xn = X.to_numpy(float)
    mu, sd = Xn.mean(0), Xn.std(0) + 1e-9           # 全員のモデル（推論用）の標準化。交差検証は分割の中で取り直す
    oof = np.zeros_like(Y)
    base_ = np.zeros_like(Y)
    LC = X[[c for c in cols if c.startswith("line_")]].to_numpy()
    linecol = np.where(LC.max(1) > 0, LC.argmax(1), -1)    # 札なしは −1（全体の平均で当てる・critic 5 回目）
    for tr, te in GroupKFold(n_splits=5).split(Xn, Y, groups):
        m_, s_ = Xn[tr].mean(0), Xn[tr].std(0) + 1e-9      # 標準化は学習の側だけで（critic 5 回目）
        Xs = (Xn - m_) / s_
        for j in range(6):
            a = fit_dim(Xs[tr], Y[tr, j], w[tr], groups[tr])
            oof[te, j] = Ridge(alpha=a).fit(Xs[tr], Y[tr, j], sample_weight=w[tr]).predict(Xs[te])
            # 比べ: 大系統の平均だけ（学習の側の重み付き平均。札の無い馬は全体の平均）
            T = pd.DataFrame({"l": linecol[tr], "y": Y[tr, j], "w": w[tr]})
            lm = (T["y"] * T["w"]).groupby(T["l"]).sum() / T["w"].groupby(T["l"]).sum()
            lm = lm.drop(-1, errors="ignore")
            base_[te, j] = pd.Series(linecol[te]).map(lm).fillna(np.average(Y[tr, j], weights=w[tr])).to_numpy()
    res = Y - oof
    out: dict = {"n": len(ids), "dims": M6,
                 "corr": {d: round(wcorr(oof[:, j], Y[:, j], w), 3) for j, d in enumerate(M6)},
                 "corr_line_only": {d: round(wcorr(base_[:, j], Y[:, j], w), 3) for j, d in enumerate(M6)},
                 "rmse": {d: round(float(np.sqrt(np.average(res[:, j] ** 2, weights=w))), 3) for j, d in enumerate(M6)},
                 "rmse_line_only": {d: round(float(np.sqrt(np.average((Y[:, j] - base_[:, j]) ** 2, weights=w))), 3) for j, d in enumerate(M6)}}
    # 実測の 95% の楕円（ブートストラップの 6 次元の共分散）の中に入る割合
    bp = common.OUT / "sire_vectors_boot.npz"
    if bp.exists():
        z = np.load(bp, allow_pickle=False)
        bid = {h: i for i, h in enumerate(z["sire_ids"].astype(str))}
        inside, inside_b, nb = 0, 0, 0
        for k, h in enumerate(ids):
            if h not in bid:
                continue
            b = z["sire_B"][:, bid[h], :]
            b = b[np.isfinite(b).all(1)]
            if len(b) < 20:
                continue
            C = np.cov(b.T) + 1e-6 * np.eye(6)
            Ci = np.linalg.inv(C)
            for arr, tag in ((oof, "m"), (base_, "b")):
                dv = arr[k] - Y[k]
                if float(dv @ Ci @ dv) <= CHI2_6_95:
                    if tag == "m":
                        inside += 1
                    else:
                        inside_b += 1
            nb += 1
        out |= {"inside_95": round(inside / nb, 3) if nb else None, "inside_95_line_only": round(inside_b / nb, 3) if nb else None, "inside_n": nb}
    out["resid_cov"] = np.cov(res.T, aweights=w).round(4).tolist()
    # 全員で学び直したモデル（推論用）
    Xs = (Xn - mu) / sd
    coef = {}
    for j, d in enumerate(M6):
        a = fit_dim(Xs, Y[:, j], w, groups)
        m = Ridge(alpha=a).fit(Xs, Y[:, j], sample_weight=w)
        coef[d] = {"alpha": a, "coef": m.coef_.round(5).tolist(), "intercept": round(float(m.intercept_), 5)}
    out["model"] = {"cols": cols, "mu": mu.round(5).tolist(), "sd": sd.round(5).tolist(), "dims": coef}
    out["oof"] = {h: [round(float(x), 3) for x in oof[k]] for k, h in enumerate(ids)}
    # 50 走未満（1〜49 走）の種牡馬: 父の (A) を同じ一つ抜きで作り、全員のモデルで予測する（産駒の成績を入力に使わない）。
    # 混ぜた値は、縮める前のずれの和 d と件数 n を、予測（生の尺度に戻した値）へ K で縮める: (d + K·予測) / (n + K)（二重の縮めを避ける・critic 5 回目）
    V, _ = shrink(F, "sire", "ss", "L4")
    Ssire = stats(F, "sire")
    small = V[(V["runs"] >= 1) & (V["runs"] < 50)]
    mu_s, sd_s = scale_of(V)
    pred = {}
    for h in small.index:
        f = own_pedigree(h).get((1, 0))
        fa = father_a_loo(h, f, L, base, Vfull, parent, amu, asd)
        p = predict_vector(inp.row(h, father_a=fa), out["model"])
        n = float(small.loc[h, "runs"])
        obs = m6({d: (small.loc[h, d] - mu_s[d]) / sd_s[d] for d in DIMS})
        # 予測（z の 6 本）を生の尺度へ: wet は芝・ダートの両方に同じ z を当てる
        pz = dict(zip(M6, p))
        praw = {d: (pz["wet"] if d in ("wet_T", "wet_D") else pz.get(d, 0.0)) * sd_s[d] + mu_s[d] for d in DIMS}
        bl = {d: (Ssire.loc[h, ("d", d)] + K * praw[d]) / (Ssire.loc[h, ("n", d)] + K) if h in Ssire.index else praw[d] for d in DIMS}
        blend = m6({d: (bl[d] - mu_s[d]) / sd_s[d] for d in DIMS})
        pred[h] = {"runs": n, "pred": [round(float(v), 3) for v in p], "obs": [round(float(v), 3) for v in obs],
                   "blend": [round(float(v), 3) for v in blend]}
    nm_small = names(set(pred), F)                    # 名前（子の血統の JSON から。small_names.json の作り）
    (common.OUT / "small_names.json").write_text(json.dumps(nm_small, ensure_ascii=False))
    out["small"] = pred
    p = common.OUT / "predict.json"
    p.write_text(json.dumps(clean_nan(out), ensure_ascii=False, allow_nan=False))
    print(json.dumps({k: out[k] for k in ("n", "corr", "corr_line_only", "rmse", "rmse_line_only", "inside_95", "inside_95_line_only", "inside_n") if k in out}, ensure_ascii=False))
    print("small", len(pred))
    data.finish_all(str(p))


def clean_nan(o):
    """NaN・inf を None にする（ページの JSON.parse が落ちないように・critic 5 回目）。"""
    if isinstance(o, dict):
        return {k: clean_nan(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean_nan(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def scale_of(V: pd.DataFrame) -> tuple[dict, dict]:
    """sire_vectors の z と同じ尺度（走数 50 以上の中から中核 300 以上の、走数の重み付きの平均と sd）。"""
    show = V[V["runs"] >= 50]
    core = show[show["runs"] >= MIN_RUNS]
    w = core["runs"].to_numpy(float)
    mu = {d: float(np.average(core[d], weights=w)) for d in DIMS}
    sd = {d: float(np.sqrt(np.average((core[d] - mu[d]) ** 2, weights=w))) or 1.0 for d in DIMS}
    return mu, sd


if __name__ == "__main__":
    main()
