"""段 1 の組の検定（探す側 70% の開催だけ・開発の行 2016〜24・確定の単勝オッズ）。作りは log.md「段 1 の固定」。

    KEIBA_SESSION=aptitude-group OMP_NUM_THREADS=4 nice -n 19 python -m src.research.aptitude_group.check main [--null 200] [--boot 200]
    ... check course   # 段 1-d コースの型の揃い（6 本）
    ... check money    # お金の線（馬連の確定オッズ m に対する「同じ群」の A/E。2020〜24）

- 主の検定: 2 着・3 着の段の条件付きロジット。段ごとの λ（log q）・(log q)²・s の 5 分位が 1 頭の強さを吸い、
  組の項（近さ・同じ父・同じ中系統・脚質の近さ・人気の近さ〔陽性の対照〕・前走の近さ〔陰性の対照〕）は 2 段で共通。
- 近さ = −‖v_j − v_先‖（揺れの次元）。3 着の段は 1・2 着との平均。全組の sd で割る（係数は 1 sd あたり）。
- 帰無 (i): レースの中で s の 3 分位 × 人気の帯の中で v を入れ替える。(ii): 近さの項を抜いた当てはめから 2・3 着を引き直す。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.research.aptitude_group import clogit, common
from src.research.aptitude_group.vector import SMALL_VENUES
from src.research.pedigree_lab import data
from src.research.pedigree_lab.explore5 import long_straight

SHOCK_COLS = ["a_pace_h", "a_pace_s", "a_wet", "a_front"]
MARGINAL = True          # False は直す前の作り（記録の再現用）
PAIR_ALL = ["sim", "same_sire", "same_ss", "style_close", "q_close", "prev_close", "novec"]
FIRST_YEAR = 2017          # 2016 年は as-of の履歴の慣らしの年（critic の 2）
COURSE_SCENES = {
    "stamina": lambda R: R["distance"] >= 2500,
    "sprint_T": lambda R: (R["surf"] == "T") & (R["distance"] <= 1200),
    "kyoto": lambda R: R["venue"] == "京都",
    "small": lambda R: R["venue"].isin(SMALL_VENUES) | ((R["venue"] == "中山") & (R["course_type"].fillna("") != "外")),
    "yoshiba": lambda R: (R["surf"] == "T") & R["venue"].isin(["札幌", "函館"]),
    "steep": lambda R: R["venue"].isin(["中山", "阪神", "中京"]),
}


def horses(shock: list[str] | None = None) -> pd.DataFrame:
    V = pd.read_parquet(common.OUT / "vec_v1.parquet")
    V["part"] = common.split(V["meeting"])
    V = V[(V["part"] == 0) & (V["date"].dt.year >= FIRST_YEAR)]
    # レースの条件: 1〜3 着の同着が無い・オッズが全頭ある・4 頭以上
    g = V.groupby("race_id")
    top = V[V["finish_position"] <= 3].groupby("race_id")["finish_position"].agg(lambda s: sorted(s.tolist()) == [1, 2, 3])
    okodds = g["odds"].apply(lambda s: bool((s > 0).all()))
    n = g.size()
    good = top.index[top & okodds.reindex(top.index) & (n.reindex(top.index) >= 4)]
    V = V[V["race_id"].isin(good)].copy()
    V = V.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)
    inv = 1.0 / V["odds"]
    V["q"] = inv / inv.groupby(V["race_id"]).transform("sum")
    V["lq"] = np.log(V["q"])
    V["lq"] = V["lq"] - V.groupby("race_id")["lq"].transform("mean")
    d, s, v = V["distance"].to_numpy(), V["surf"].to_numpy(), V["venue"].to_numpy()
    ct = V["course_type"].fillna("").to_numpy()
    small = np.isin(v, SMALL_VENUES) | ((v == "中山") & (ct != "外"))
    straight = (s == "T") & np.array([long_straight(a, b) for a, b in zip(v, d)])
    V["s"] = (np.where(s == "D", 1.0, -1.0) * V["a_dirt"] + np.where(d >= 2000, 1.0, np.where(d <= 1400, -1.0, 0.0)) * V["a_dist"]
              + np.isin(v, ["中山", "阪神", "中京"]) * V["a_steep"] + small * V["a_small"] + straight * V["a_straight"])
    V["s_q5"] = quint(V, "s")
    V["style"] = V["style"].fillna(0.5)
    V["prev_fpct"] = V["prev_fpct"].fillna(0.5)
    cols = SHOCK_COLS if shock is None else shock
    V[cols] = V[cols].fillna(0.0)
    V["novec"] = (np.abs(V[cols].to_numpy()).sum(1) < 1e-9).astype(float)     # 血統の履歴がどの段にも無い馬（近さが 0 で最も近く見える。critic の 3）
    return V


def quint(V: pd.DataFrame, col: str) -> np.ndarray:
    r = V.groupby("race_id")[col].rank(method="first", pct=True).to_numpy()
    return np.minimum((r * 5 - 1e-9).astype(int), 4)


class Races:
    """レースの単位の配列（行はレースの順に連続）。"""

    def __init__(self, V: pd.DataFrame, shock: list[str]):
        self.V = V
        self.rid = pd.factorize(V["race_id"])[0]
        self.R = int(self.rid.max()) + 1
        self.start = np.r_[0, np.flatnonzero(np.diff(self.rid)) + 1]
        fp = V["finish_position"].to_numpy()
        self.pos = {k: self.start + np.array([np.flatnonzero(fp[a:b] == k)[0] for a, b in
                                              zip(self.start, np.r_[self.start[1:], len(V)])]) for k in (1, 2, 3)}
        self.v = V[shock].to_numpy(float)
        self.sire = pd.factorize(V["sire"])[0]
        self.ss = pd.factorize(V["ss"])[0]
        self.style = V["style"].to_numpy(float)
        self.lq = V["lq"].to_numpy(float)
        self.prev = V["prev_fpct"].to_numpy(float)
        self.q5 = V["s_q5"].to_numpy()
        self.meet = pd.factorize(V.drop_duplicates("race_id")["meeting"])[0]
        self.year = V.drop_duplicates("race_id")["date"].dt.year.to_numpy()
        self.novec = V["novec"].to_numpy(float)
        self.extra: dict[str, np.ndarray] = {}       # 組の項（積）に使う値（帰無で入れ替えうる）
        self.marg: dict[str, np.ndarray] = {}        # 1 頭の値として段ごとの 1 次・2 次に入れる値（常に本物）
        # 中心化した揺れの次元（内積の近さ用。pedigree-1 の最後の 1 本）。血統の履歴の無い馬は 0
        self.vc = (self.v - pd.DataFrame(self.v).groupby(self.rid).transform("mean").to_numpy()) * (1 - self.novec)[:, None]

    def typicality(self) -> dict[str, np.ndarray]:
        """その馬と他の出走馬全頭との近さの平均（典型さ。pedigree-2 の最後の 1 本）。血統の揺れ・脚質・前走の 3 つ。"""
        n = len(self.V)
        out = {k: np.zeros(n) for k in ("typ_sim", "typ_style", "typ_prev")}
        ends = np.r_[self.start[1:], n]
        for a, b in zip(self.start, ends):
            v = self.v[a:b]
            D = np.sqrt(((v[:, None, :] - v[None, :, :]) ** 2).sum(2))
            m = b - a - 1
            out["typ_sim"][a:b] = -D.sum(1) / max(m, 1)
            for k, x in (("typ_style", self.style[a:b]), ("typ_prev", self.prev[a:b])):
                out[k][a:b] = -np.abs(x[:, None] - x[None, :]).sum(1) / max(m, 1)
        return out

    def pair(self, j: np.ndarray, p: np.ndarray, v: np.ndarray | None = None) -> dict[str, np.ndarray]:
        v = self.v if v is None else v
        out = {"sim": -np.sqrt(((v[j] - v[p]) ** 2).sum(1)),
               "same_sire": (self.sire[j] == self.sire[p]).astype(float),
               "same_ss": ((self.ss[j] == self.ss[p]) & (self.sire[j] != self.sire[p])).astype(float),
               "style_close": -np.abs(self.style[j] - self.style[p]),
               "q_close": -np.abs(self.lq[j] - self.lq[p]),
               "prev_close": -np.abs(self.prev[j] - self.prev[p]),
               "novec": np.maximum(self.novec[j], self.novec[p]),
               "dot": (self.vc[j] * self.vc[p]).sum(1)}
        for k, a in self.extra.items():
            out[k] = a[j] * a[p]
        return out

    def stages(self, w: np.ndarray, x: np.ndarray, z: np.ndarray | None, races: np.ndarray, pair_cols: list[str],
               q5: list[np.ndarray] | None = None, v: np.ndarray | None = None):
        """選ばれたレース races の 2 着・3 着の段の行。z が None なら 3 着の段の chosen は 0（引き直し用）。"""
        q5 = [self.q5] if q5 is None else q5
        mask = np.isin(self.rid, races)
        h = np.flatnonzero(mask)
        r = self.rid[h]
        rmap = np.full(self.R, -1)
        rmap[races] = np.arange(len(races))
        k = rmap[r]
        blocks = []
        for st in (2, 3):
            if st == 2:
                keep = h != w[r]
                hj = h[keep]
                rr = r[keep]
                P = self.pair(hj, w[rr], v)
                ch = (hj == x[rr]) if x is not None else np.zeros(len(hj), bool)
            else:
                keep = (h != w[r]) & (h != x[r])
                hj = h[keep]
                rr = r[keep]
                P1, P2 = self.pair(hj, w[rr], v), self.pair(hj, x[rr], v)
                P = {c: 0.5 * (P1[c] + P2[c]) for c in P1}
                ch = (hj == z[rr]) if z is not None else np.zeros(len(hj), bool)
            sid = 2 * k[keep] + (st - 2)
            cols = [self.lq[hj], self.lq[hj] ** 2]
            if MARGINAL:
                # 1 頭の値そのものの 1 次・2 次（近さの項が 1 頭の強さの残りを拾わないように。陰性の対照の失敗の直し・11/27）
                vv = self.v                      # 1 頭の値は常に本物（帰無で入れ替えるのは近さの項だけ）
                for a in [self.prev[hj], self.style[hj]] + [vv[hj, i] for i in range(vv.shape[1])] + [e[hj] for e in self.marg.values()]:
                    cols += [a, a ** 2]
                cols.append(self.novec[hj])
            for qq in q5:
                cols += [(qq[hj] == b).astype(float) for b in range(1, 5)]
            S = np.column_stack(cols)
            Z = np.zeros_like(S)
            stage_part = np.hstack([S, Z]) if st == 2 else np.hstack([Z, S])
            X = np.hstack([stage_part, np.column_stack([P[c] for c in pair_cols])]) if pair_cols else stage_part
            blocks.append((sid, X, ch, hj))
        sid = np.r_[blocks[0][0], blocks[1][0]]
        X = np.vstack([blocks[0][1], blocks[1][1]])
        ch = np.r_[blocks[0][2], blocks[1][2]]
        hj = np.r_[blocks[0][3], blocks[1][3]]
        o = np.argsort(sid, kind="stable")
        return sid[o], X[o], ch[o], hj[o]


def scale_pairs(RA: Races, races: np.ndarray, pair_cols: list[str]) -> dict[str, float]:
    """組の項の sd（2 着の段の行で測る。係数を 1 sd あたりにそろえる）。"""
    _, X, _, _ = RA.stages(RA.pos[1], RA.pos[2], RA.pos[3], races, pair_cols)
    m = X.shape[1] - len(pair_cols)
    return {c: float(X[:, m + i].std() or 1.0) for i, c in enumerate(pair_cols)}


def scale_sim(RA: Races, races: np.ndarray, v: np.ndarray) -> float:
    _, X, _, _ = RA.stages(RA.pos[1], RA.pos[2], RA.pos[3], races, ["sim"], v=v)
    return float(X[:, -1].std() or 1.0)


def strata(RA: Races) -> list[np.ndarray]:
    """帰無 (i) の層: レースの中の s の 3 分位 × 人気の帯（1〜3・4〜6・7〜）。"""
    V = RA.V
    t3 = np.minimum((V.groupby("race_id")["s"].rank(method="first", pct=True).to_numpy() * 3 - 1e-9).astype(int), 2)
    pop = V["popularity"].to_numpy()
    pb = np.where(pop <= 3, 0, np.where(pop <= 6, 1, 2))
    st = pd.factorize(pd.Series(RA.rid).astype(str) + "_" + pd.Series(t3).astype(str) + "_" + pd.Series(pb).astype(str))[0]
    return [ii for ii in pd.Series(np.arange(len(V))).groupby(st).indices.values() if len(ii) > 1]


def perm_in(gl: list[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    perm = np.arange(n)
    for ii in gl:
        perm[ii] = ii[rng.permutation(len(ii))]
    return perm


def run_fit(RA: Races, races: np.ndarray, pair_cols: list[str], sd: dict[str, float], z_null=None, q5=None, v=None,
            w_set=None, x=None, z=None):
    x = RA.pos[2] if x is None else x
    z = RA.pos[3] if z is None else z
    sid, X, ch, _ = RA.stages(RA.pos[1], x, z, races, pair_cols, q5=q5, v=v)
    m = X.shape[1] - len(pair_cols)
    for i, c in enumerate(pair_cols):
        X[:, m + i] /= sd[c]
    b, Hinv, ll = clogit.fit(X, ch, sid, w_set=w_set)
    return b, Hinv, ll, (sid, X, ch, m)


def summarize(RA: Races, races, pair_cols, sd, q5=None, v=None) -> dict:
    b, Hinv, ll, (sid, X, ch, m) = run_fit(RA, races, pair_cols, sd, q5=q5, v=v)
    k = np.arange(len(races))
    cl = np.repeat(RA.meet[races], 2)          # set = 2·k + 段
    cl = pd.factorize(cl)[0]
    se = clogit.cluster_se(X, ch, sid, b, Hinv, cl)
    out = {"races": int(len(races)), "ll": round(ll, 1)}
    for i, c in enumerate(pair_cols):
        out[c] = [round(float(b[m + i]), 4), round(float(se[m + i]), 4)]
    out["lambda2_3"] = [round(float(b[0]), 3), round(float(b[m // 2]), 3)]
    del k
    return out


def null_perm(RA: Races, races, pair_cols, sd, n: int, seed0: int = 5000) -> list[float]:
    """(i) レースの中で s の 3 分位 × 人気の帯の中だけで v を入れ替える（近さの項だけ。1 頭の値は本物のまま）。"""
    gl = strata(RA)
    res = []
    for s in range(n):
        perm = perm_in(gl, len(RA.V), np.random.default_rng(seed0 + s))
        b, _, _, (_, _, _, m) = run_fit(RA, races, pair_cols, sd, v=RA.v[perm])
        res.append(float(b[m + pair_cols.index("sim")]))
    return res


def null_sim(RA: Races, races, pair_cols, sd, n: int, seed0: int = 7000) -> list[float]:
    """(ii) 近さの項を抜いた当てはめ（β = 0）から 2・3 着を引き直し、同じ推定で近さの係数を測る。"""
    cols0 = [c for c in pair_cols if c != "sim"]
    b0, _, _, (_, _, _, m0) = run_fit(RA, races, cols0, sd)
    res = []
    for s in range(n):
        rng = np.random.default_rng(seed0 + s)
        x = RA.pos[2].copy()
        z = RA.pos[3].copy()
        # 2 着の段: 1 着は本物のまま、2 着を引く
        sid, X, _, hj = RA.stages(RA.pos[1], x, z, races, cols0)
        for i, c in enumerate(cols0):
            X[:, m0 + i] /= sd[c]
        st2 = (sid % 2) == 0
        pick = clogit.sample(X[st2], b0, sid[st2] // 2, rng)
        x[races] = hj[st2][pick]
        # 3 着の段: 引いた 2 着で選択肢と組の項を作り直して引く
        sid, X, _, hj = RA.stages(RA.pos[1], x, z, races, cols0)
        for i, c in enumerate(cols0):
            X[:, m0 + i] /= sd[c]
        st3 = (sid % 2) == 1
        pick = clogit.sample(X[st3], b0, sid[st3] // 2, rng)
        z[races] = hj[st3][pick]
        b, _, _, (_, _, _, m) = run_fit(RA, races, pair_cols, sd, x=x, z=z)
        res.append(float(b[m + pair_cols.index("sim")]))
    return res


def boot(RA: Races, races, pair_cols, sd, n: int, seed0: int = 9000) -> list[float]:
    meets = RA.meet[races]
    um = np.unique(meets)
    res = []
    for s in range(n):
        rng = np.random.default_rng(seed0 + s)
        draw = rng.choice(um, len(um), replace=True)
        cnt = pd.Series(draw).value_counts()
        w_r = pd.Series(meets).map(cnt).fillna(0).to_numpy(float)
        w_set = np.repeat(w_r, 2)
        keep = w_r > 0
        rr = races[keep]
        b, _, _, (_, _, _, m) = run_fit(RA, rr, pair_cols, sd, w_set=np.repeat(w_r[keep], 2))
        del w_set
        res.append(float(b[m + pair_cols.index("sim")]))
    return res


def scenes(RA: Races) -> dict[str, np.ndarray]:
    R = RA.V.drop_duplicates("race_id").reset_index(drop=True)
    s, d, g = R["surf"].to_numpy(), R["distance"].to_numpy(), R["going"].to_numpy()
    shaky = ((s == "D") & np.isin(g, ["S", "W"])) | ((s == "T") & (g == "W")) | ((s == "T") & (d >= 2400))
    idx = np.arange(len(R))
    return {"all": idx, "turf": idx[s == "T"], "dirt": idx[s == "D"], "shaky": idx[shaky], "calm": idx[~shaky],
            "p2017_19": idx[RA.year <= 2019], "p2020_24": idx[RA.year >= 2020]}


def cmd_main(a) -> dict:
    V = horses()
    RA = Races(V, SHOCK_COLS)
    sc = scenes(RA)
    allr = sc["all"]
    sd = scale_pairs(RA, allr, PAIR_ALL)
    out: dict = {"sd": sd, "horses": int(len(V))}
    variants = {
        "main": PAIR_ALL,
        "sim_only": ["sim"],
        "no_controls": ["sim", "same_sire", "same_ss"],
        "sire_only": ["same_sire", "same_ss", "style_close", "q_close", "prev_close"],
    }
    for name, cols in variants.items():
        out[f"fit_{name}"] = summarize(RA, allr, cols, sd)
        print(name, out[f"fit_{name}"], flush=True)
    for name, idx in sc.items():
        if name == "all":
            continue
        out[f"scene_{name}"] = summarize(RA, idx, PAIR_ALL, sd)
        print(name, out[f"scene_{name}"], flush=True)
    # 次元ごとの近さ（記述）
    for i, c in enumerate(SHOCK_COLS):
        RA1 = RA.v
        vi = RA1[:, [i]]
        sdi = sd | {"sim": scale_sim(RA, allr, vi)}       # 組の項 −|差| の sd（critic の 5）
        out[f"dim_{c}"] = summarize(RA, allr, PAIR_ALL, sdi, v=vi)
        print(c, out[f"dim_{c}"], flush=True)
    if a.boot:
        bb = boot(RA, allr, PAIR_ALL, sd, a.boot)
        out["boot_sim"] = {"n": len(bb), "q025": round(float(np.percentile(bb, 2.5)), 4), "q975": round(float(np.percentile(bb, 97.5)), 4)}
        print("boot", out["boot_sim"], flush=True)
    if a.null:
        n1 = null_perm(RA, allr, PAIR_ALL, sd, a.null)
        out["null_perm"] = {"n": len(n1), "mean": round(float(np.mean(n1)), 4), "q95": round(float(np.percentile(n1, 95)), 4)}
        print("null_perm", out["null_perm"], flush=True)
        n2 = null_sim(RA, allr, PAIR_ALL, sd, a.null)
        out["null_sim"] = {"n": len(n2), "mean": round(float(np.mean(n2)), 4), "q95": round(float(np.percentile(n2, 95)), 4)}
        print("null_sim", out["null_sim"], flush=True)
    return out


def cmd_course(a) -> dict:
    """段 1-d: その場面のレースだけで、主のモデルに β_c·（中心化した適性の積）を足す。帰無は主と同じ層の中の入れ替え（積の項だけ）。"""
    V = horses()
    out: dict = {"sidak_t": 2.64}
    for c, f in COURSE_SCENES.items():
        col = f"a_{c}"
        V[col] = V[col].fillna(0.0)
        cen = (V[col] - V.groupby("race_id")[col].transform("mean")).to_numpy(float)
        RA = Races(V, SHOCK_COLS)
        R = V.drop_duplicates("race_id").reset_index(drop=True)
        idx = np.flatnonzero(f(R).to_numpy())
        RA.extra, RA.marg = {"course": cen}, {"course": V[col].to_numpy(float)}
        q5c = quint(V, col)
        cols = ["course"] + PAIR_ALL
        sd = scale_pairs(RA, idx, cols)
        out[c] = summarize(RA, idx, cols, sd, q5=[RA.q5, q5c])
        b, se = out[c]["course"]
        out[c]["t"] = round(b / se, 2) if se > 0 else None
        out[c]["pass_sidak"] = bool(se > 0 and b / se >= 2.64)
        if a.null:
            gl = strata(RA)
            n1 = []
            for s in range(a.null):
                perm = perm_in(gl, len(V), np.random.default_rng(11000 + s))
                RA.extra = {"course": cen[perm]}
                bb, _, _, (_, _, _, m) = run_fit(RA, idx, cols, sd, q5=[RA.q5, q5c])
                n1.append(float(bb[m]))
            RA.extra = {"course": cen}
            out[c]["null_perm_q95"] = round(float(np.percentile(n1, 95)), 4)
            out[c]["null_perm_mean"] = round(float(np.mean(n1)), 4)
        print(c, out[c], flush=True)
    return out


def cmd_final(a) -> dict:
    """閉じる前の最後の 1 本（log.md に事前に固定）: 典型さを 1 頭の値に足し、(A) 距離の近さ (B) 中心化した内積 を当てる。"""
    V = horses()
    RA = Races(V, SHOCK_COLS)
    RA.marg = RA.typicality()
    sc = scenes(RA)
    out: dict = {}
    for ver, key in (("A_dist", "sim"), ("B_dot", "dot")):
        cols = [key] + [c for c in PAIR_ALL if c != "sim"]
        for name in ("all", "shaky"):
            idx = sc[name]
            sd = scale_pairs(RA, idx, cols)
            r = summarize(RA, idx, cols, sd)
            b, se = r[key]
            r["ci95"] = [round(b - 1.96 * se, 4), round(b + 1.96 * se, 4)]
            r["pass"] = bool(b - 1.96 * se > 0)
            out[f"{ver}_{name}"] = r
            print(ver, name, r, flush=True)
    out["reopen"] = any(v["pass"] for v in out.values())
    return out


def cmd_money(a) -> dict:
    """お金の線: 2020〜24 の探す側のレースで、近さの上位 1/4 の組（同じ群）の、馬連の確定オッズ m に対する A/E。"""
    V = horses()
    V = V[V["date"].dt.year >= 2020].reset_index(drop=True)
    fr = []
    for y in range(2020, 2025):
        P = pd.read_parquet(f"{common.TABLES}/{y}/race_pair_odds_flat.parquet", columns=["race_id", "odds_type", "pair", "odds"])
        fr.append(P[P["odds_type"] == "umaren"])
    P = pd.concat(fr, ignore_index=True)
    P = P[P["race_id"].isin(V["race_id"].unique()) & (P["odds"] > 0)]
    pr = P["pair"].map(lambda s: json.loads(s) if isinstance(s, str) else list(s))     # 文字列 '[1, 2]'（critic の 1）
    P["a"] = pr.map(lambda t: int(min(t)))
    P["b"] = pr.map(lambda t: int(max(t)))
    P["m"] = 1.0 / P["odds"]
    P["m"] = P["m"] / P.groupby("race_id")["m"].transform("sum")
    H = V[["race_id", "horse_number", "finish_position", "popularity", "meeting", "sire", "novec"] + SHOCK_COLS]
    X = P.merge(H.add_suffix("_a").rename(columns={"race_id_a": "race_id"}), left_on=["race_id", "a"], right_on=["race_id", "horse_number_a"])
    X = X.merge(H.add_suffix("_b").rename(columns={"race_id_b": "race_id"}), left_on=["race_id", "b"], right_on=["race_id", "horse_number_b"])
    X = X[(X["novec_a"] == 0) & (X["novec_b"] == 0)].reset_index(drop=True)      # 血統の履歴の無い馬の組は外す（critic の 3）
    X["sim"] = -np.sqrt(sum((X[f"{c}_a"] - X[f"{c}_b"]) ** 2 for c in SHOCK_COLS))
    X["grp"] = X.groupby("race_id")["sim"].rank(pct=True, method="first") > 0.75
    X["hit"] = (X[["finish_position_a", "finish_position_b"]].max(axis=1) == 2) & (X[["finish_position_a", "finish_position_b"]].min(axis=1) == 1)
    lo = X[["popularity_a", "popularity_b"]].min(axis=1)
    hi = X[["popularity_a", "popularity_b"]].max(axis=1)
    band = lambda p: np.where(p <= 1, 0, np.where(p <= 3, 1, np.where(p <= 6, 2, np.where(p <= 9, 3, 4))))
    X["cell"] = band(lo) * 10 + band(hi)
    out = {"pairs": int(len(X)), "races": int(X["race_id"].nunique())}
    meet = X["meeting_a"].astype(str).to_numpy()

    def ae(mask):
        O = X["hit"].to_numpy(float) * mask
        E = X["m"].to_numpy() * mask
        return O, E
    O1, E1 = ae(X["grp"].to_numpy())
    O0, E0 = ae(~X["grp"].to_numpy())
    # 升の中の比（同じ群の E で重み付け）
    T = pd.DataFrame({"cell": X["cell"], "g": X["grp"], "O": X["hit"].astype(float), "E": X["m"]}).groupby(["cell", "g"])[["O", "E"]].sum().unstack("g")
    T = T.dropna()
    T = T[(T[("E", True)] > 0) & (T[("E", False)] > 0) & (T[("O", False)] > 0)]   # 両側がある升だけ（critic の 4。O の条件は 0 割りを避けるため。結果で升を選ぶ小さな偏りが残る・critic 2 回目の 12）
    rat = (T[("O", True)] / T[("E", True)]) / (T[("O", False)] / T[("E", False)])
    wt = T[("E", True)]
    out["ae_grp"] = round(float(O1.sum() / E1.sum()), 4)
    out["ae_other"] = round(float(O0.sum() / E0.sum()), 4)
    out["ratio_in_cell"] = round(float((rat * wt).sum() / wt.sum()), 4)
    D = pd.DataFrame({"meet": meet, "O1": O1, "E1": E1, "O0": O0, "E0": E0}).groupby("meet").sum()
    rng = np.random.default_rng(31)
    bs = []
    for _ in range(1000):
        ii = rng.integers(0, len(D), len(D))
        s = D.iloc[ii].sum()
        bs.append(s["O1"] / s["E1"])
    out["ae_grp_ci"] = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
    out["roi_grp_flat"] = round(float((X["hit"] * X["odds"])[X["grp"]].sum() / X["grp"].sum()), 4)
    out["roi_all_flat"] = round(float((X["hit"] * X["odds"]).sum() / len(X)), 4)
    print(out, flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["main", "course", "money", "final"])
    ap.add_argument("--null", type=int, default=0)
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-marginal", action="store_true", help="直す前の作り（1 頭の値の 1 次・2 次を入れない）")
    a = ap.parse_args()
    global MARGINAL
    MARGINAL = not a.no_marginal
    common.open_token("select", f"check_{a.cmd}_v1", f"段 1 の組の検定 {a.cmd}（探す側 70% の開催だけ）")
    out = {"main": cmd_main, "course": cmd_course, "money": cmd_money, "final": cmd_final}[a.cmd](a)
    p = common.OUT / f"check_{a.cmd}{a.tag}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    data.finish_all(str(p))


if __name__ == "__main__":
    main()
