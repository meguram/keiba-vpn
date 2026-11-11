"""任意の観点の as-of ターゲットエンコーディング（TE）と、単独で効かなかった軸の交互作用が、親モデルの予測の残差に残っているか。再学習 #4（2026-11）。

    python interaction_te_search.py plan     # 候補の一覧と本数 K（結果を見る前に数える）→ <OUT>/manifest.json
    python interaction_te_search.py build    # as-of の TE 列と軸・条件の列を作る → <OUT>/inter_features.parquet（2022〜2025 年）
    python interaction_te_search.py score    # 第 1 段(1): 残差のスコア検定
    python interaction_te_search.py trees    # 第 1 段(2): 残差の木（対照との差）
    python interaction_te_search.py tier2    # 第 2 段: 第 1 段を通ったものを束にして test 2025 に 1 回だけ適用

合格線は docs/operations/retrain_trials/README.md「任意の観点の TE・交互作用」に測る前に固定した。
2026 年は使わない。市場オッズ・人気は使わない。統計量は『そのレース日より前』だけ（同日を含まない累積）。障害は除く（平地のみ）。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("/home/jovyan/work/retrain_out")
BASE = R / "base_2016"
OUT = R / "inter16"
PED = Path("/home/jovyan/work/keiba-vpn/data/local/modeling/training_pipeline_ped/pedigree_keys.parquet")
OOF = BASE / "tp27_v6e/ens/v6.parquet"
T25 = BASE / "final_eval_v6e16/test2025_predictions.parquet"
APRQ = R / "aprq16/aprq_features.parquet"
WIND = R / "wind16/race_wind.parquet"
AXES = R / "axes16/simple_axes.parquet"
EVAL_YEARS = (2022, 2023, 2024)
M_STD, M_ID = 20.0, 40.0
MIN_COND_ROWS = 2000
NOISE = 40

PARTNERS = ["a_style", "a_l3fz_ewm2", "a_rk_ewm2", "c_res_ewm2", "c_cls_step", "b_dist_gap", "d_rest", "d_nprior", "d_wt_chg_now", "d_wt_dev",
            "jockey_weight", "weight", "sr_load_eq", "sr_exp", "sr_perf_ewm2", "f2_pace_adv", "pc_adv", "f3_r_gap", "gate", "speed_max",
            "b_aff_course", "b_aff_going", "c_standing_ewm2", "c_rq_ewm2", "f5_win", "a_top3_freq", "horse_prior_win_rate", "jockey_prior_win_rate",
            "trainer_prior_win_rate", "sire_prior_win_rate_surface", "distance", "field_size"]
CAT_CTX = ["venue_code", "surface", "grade", "track_condition"]
BASE_COLS = ["race_id", "horse_id", "date", "venue_code", "surface", "distance", "grade", "track_condition", "field_size", "horse_number", "sex_age",
             "jockey_id", "trainer_id", "finish_position", "sr_load_chg", "d_rest", "d_nprior", "c_cls_step", "b_dist_gap"]
WIND_COLS = ["head", "cross", "wind_speed_ms"]
APRQ_COLS = ["pred_b_F", "pred_b_D", "st_b_F", "st_b_D", "sw_b_F", "sw_b_D", "d_b_F", "d_b_D",
             "apt_course_win", "apt_going_win", "apt_sum_win", "apt_course_t3", "apt_going_t3", "apt_sum_t3",
             "rq_p_high", "rq_p_slow", "rq_p_l3f_fast", "rq_p_l3f_slow", "rq_p_slow_kick", "rq_p_high_collapse", "rq_te_win", "rq_te_t3"]
SIMPLE = ["s_split", "s_front", "s_cr", "s_makuri", "s_endure"]


def log(m: str) -> None:
    print(f"[{time.strftime('%T')}] {m}", flush=True)


# ------------------------------------------------------------------ データ
def digit(x, edges) -> np.ndarray:
    x = np.asarray(x, float)
    c = np.digitize(x, edges, right=True).astype(np.int64)
    c[np.isnan(x)] = -1
    return c


def factor(*cols) -> np.ndarray:
    s = pd.Series(cols[0]).astype(str)
    for c in cols[1:]:
        s = s + "|" + pd.Series(c).astype(str)
    return pd.factorize(s)[0].astype(np.int64)


def load_table() -> pd.DataFrame:
    cols = sorted(set(BASE_COLS + PARTNERS) - {"gate"} | {"horse_number"})
    t = pd.read_parquet(BASE / "features/train_table.parquet", columns=cols)
    for c in ("race_id", "horse_id"):
        t[c] = t[c].astype(str)
    for c in ("venue_code", "surface", "grade", "track_condition", "jockey_id", "trainer_id", "sex_age"):
        t[c] = t[c].astype(object).where(t[c].notna(), "").astype(str)
    t["date"] = pd.to_datetime(t["date"].astype(str).str[:10])
    t["year"] = t["date"].dt.year
    t = t[t["surface"].isin(["芝", "ダート"]) & (t["year"] <= 2025)].drop_duplicates(["race_id", "horse_id"]).reset_index(drop=True)
    ped = pd.read_parquet(PED).drop_duplicates("horse_id")[["horse_id", "sire_id", "dam_sire_id"]]
    ped["horse_id"] = ped["horse_id"].astype(str)
    t = t.merge(ped, on="horse_id", how="left")
    fs = pd.to_numeric(t["field_size"], errors="coerce")
    t["gate"] = (pd.to_numeric(t["horse_number"], errors="coerce") - 1) / (fs - 1).replace(0, np.nan)
    return t


def add_axes(t: pd.DataFrame) -> dict[str, np.ndarray]:
    """軸（連続値）。風（レース単位）、走法軸の履歴（前走・前 3 走平均）、APT/BIAS/レース質（aprq16）。"""
    ax: dict[str, np.ndarray] = {}
    w = pd.read_parquet(WIND, columns=["race_id"] + WIND_COLS).drop_duplicates("race_id")
    w["race_id"] = w["race_id"].astype(str)
    m = t[["race_id"]].merge(w, on="race_id", how="left")
    for c in WIND_COLS:
        ax["wind_" + c] = m[c].to_numpy(float)
    a = pd.read_parquet(AXES, columns=["race_id", "horse_id", "date"] + SIMPLE)
    a["race_id"], a["horse_id"] = a["race_id"].astype(str), a["horse_id"].astype(str)
    a["date"] = pd.to_datetime(a["date"])
    a = a.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    g = a.groupby("horse_id", sort=False)
    for c in SIMPLE:
        a[c + "_p1"] = g[c].shift(1)
        a[c + "_p3"] = g[c].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    keep = [c + sfx for c in SIMPLE for sfx in ("_p1", "_p3")]
    m = t[["race_id", "horse_id"]].merge(a[["race_id", "horse_id"] + keep].drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    for c in keep:
        ax["ax_" + c] = m[c].to_numpy(float)
    f = pd.read_parquet(APRQ, columns=["race_id", "horse_id"] + APRQ_COLS)
    f["race_id"], f["horse_id"] = f["race_id"].astype(str), f["horse_id"].astype(str)
    m = t[["race_id", "horse_id"]].merge(f.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    for c in APRQ_COLS:
        ax["aq_" + c] = m[c].to_numpy(float)
    return ax


def level_of(x: np.ndarray, race: np.ndarray) -> str:
    s = pd.Series(x).groupby(race)
    sd = s.std()
    n = s.count()
    ok = n >= 2
    if ok.sum() == 0:
        return "race"
    return "race" if float((sd[ok].fillna(0) > 1e-9).mean()) < 0.01 else "horse"


def axis_codes(t: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """A 軸（馬・出走の属性、整数コード、欠測 -1）と S 軸（舞台、欠測なし）。"""
    pre = t["year"].to_numpy() <= 2019
    sa = t["sex_age"].astype(str)
    age = pd.to_numeric(sa.str[1:], errors="coerce").to_numpy(float)
    A: dict[str, np.ndarray] = {}
    A["age"] = np.where(np.isnan(age), -1, np.clip(age, 2, 6) - 2).astype(np.int64)
    A["sex"] = sa.str[:1].map({"牡": 0, "牝": 1, "セ": 2}).fillna(-1).to_numpy(np.int64)
    A["jw"] = digit(t["jockey_weight"], [52, 54, 56, 58])
    lr = t["sr_load_eq"].to_numpy(float) if "sr_load_eq" in t else np.full(len(t), np.nan)
    q = np.nanquantile(lr[pre], [0.2, 0.4, 0.6, 0.8])
    A["lr"] = digit(lr, q)
    A["bw"] = digit(t["weight"], [440, 480, 520])
    A["wc"] = digit(t["d_wt_chg_now"], [-6, -2, 2, 6])
    A["lc"] = digit(t["sr_load_chg"], [-2, -0.5, 0.5, 2])
    A["style"] = digit(t["a_style"], [0.25, 0.5, 0.75])
    third = digit(t["gate"], [0.34, 0.67])
    A["styledraw"] = np.where((A["style"] >= 0) & (third >= 0), A["style"] * 3 + third, -1).astype(np.int64)
    rest = digit(t["d_rest"], [14, 35, 70, 140])
    rest = np.where(pd.to_numeric(t["d_nprior"], errors="coerce").to_numpy() == 0, 5, rest)
    A["rest"] = rest
    A["cs"] = digit(t["c_cls_step"], [-0.5, 0.5])
    A["dg"] = digit(t["b_dist_gap"], [-300, 300])
    for k, c in (("jockey", "jockey_id"), ("trainer", "trainer_id"), ("sire", "sire_id"), ("damsire", "dam_sire_id")):
        s = t[c].astype(object).where(t[c].notna(), "").astype(str)
        code = pd.factorize(s)[0].astype(np.int64)
        code[(s == "").to_numpy()] = -1
        A[k] = code
    going = t["track_condition"].replace({"稍": "稍重", "不": "不良"}).astype(str)
    sv = t["surface"].astype(str)
    db = digit(t["distance"], [1300, 1600, 2000, 2400])
    S: dict[str, np.ndarray] = {}
    S["k3"] = factor(sv, db)
    S["k2"] = factor(t["venue_code"], sv, db)
    S["k1"] = factor(t["venue_code"], sv, t["distance"])
    S["going"] = factor(sv, going)
    S["grade"] = factor(sv, t["grade"])
    S["fs"] = factor(sv, digit(t["field_size"], [10, 13, 16]))
    S["venue"] = factor(t["venue_code"], sv)
    return A, S


A_PLAIN = ["age", "sex", "jw", "lr", "bw", "wc", "lc", "style", "rest", "cs", "dg"]
A_ID = ["jockey", "trainer", "sire", "damsire"]
S_ALL = ["k3", "k2", "k1", "going", "grade", "fs", "venue"]
# 騎手×コース・調教師×競馬場は tp-58 で評価済み・不採用、種牡馬/母父×(面距離帯・馬場・場) は親の ped_* に入っている → 除く
S_ID = {"jockey": ["going", "grade"], "trainer": ["going", "grade"], "sire": ["grade"], "damsire": ["grade"]}
A_SPECIAL = {"styledraw": ["k1", "k2"]}
PAIRS = [("wc", "lc"), ("lr", "age"), ("lr", "sex"), ("jw", "bw"), ("age", "sex"), ("jw", "age"), ("jw", "sex"), ("bw", "sex")]


def te_specs() -> list[tuple[str, str, str, str]]:
    """(列名の幹, 種類, A, S)。種類 = cell（A×S）/ pair（A×A）。win と t3 の 2 通りは列名の末尾。"""
    sp = []
    for a in A_PLAIN:
        for s in S_ALL:
            sp.append((f"te__{a}__{s}", "cell", a, s))
    for a in A_ID:
        for s in S_ID[a]:
            sp.append((f"te__{a}__{s}", "cell", a, s))
    for a, ss in A_SPECIAL.items():
        for s in ss:
            sp.append((f"te__{a}__{s}", "cell", a, s))
    for a, b in PAIRS:
        sp.append((f"te__{a}x{b}", "pair", a, b))
    return sp


def cond_defs(t: pd.DataFrame, A: dict[str, np.ndarray], S: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, str]]:
    """条件（ダミー化の元）。名前 → (整数コード, 水準 race/horse)。"""
    going = t["track_condition"].replace({"稍": "稍重", "不": "不良"}).astype(str)
    c: dict[str, tuple[np.ndarray, str]] = {}
    c["surface"] = (pd.factorize(t["surface"].astype(str))[0].astype(np.int64), "race")
    c["distband"] = (digit(t["distance"], [1300, 1600, 2000, 2400]), "race")
    c["venue"] = (pd.factorize(t["venue_code"].astype(str))[0].astype(np.int64), "race")
    c["going"] = (pd.factorize(going)[0].astype(np.int64), "race")
    c["grade"] = (pd.factorize(t["grade"].astype(str))[0].astype(np.int64), "race")
    c["fsband"] = (digit(t["field_size"], [10, 13, 16]), "race")
    for k in A_PLAIN:
        c[k] = (A[k], "horse")
    c["gatebin"] = (digit(t["gate"], [1 / 3, 2 / 3]), "horse")
    return c


def dummies(codes: dict[str, tuple[np.ndarray, str]], rows: np.ndarray) -> tuple[list[str], list[str], np.ndarray]:
    names, levels, cols = [], [], []
    for k, (cd, lv) in codes.items():
        for v in np.unique(cd[rows]):
            if v < 0:
                continue
            if int((cd[rows] == v).sum()) < MIN_COND_ROWS:
                continue
            names.append(f"{k}={int(v)}")
            levels.append(lv)
            cols.append((cd == v).astype(np.float32))
    return names, levels, np.column_stack(cols)


def standardize(x: np.ndarray, pre: np.ndarray) -> np.ndarray:
    v = x[pre & np.isfinite(x)]
    mu, sd = (float(v.mean()), float(v.std())) if len(v) > 10 else (0.0, 1.0)
    z = (x - mu) / (sd if sd > 0 else 1.0)
    return np.where(np.isfinite(z), z, 0.0).astype(np.float32)


# ------------------------------------------------------------------ as-of の累積
def asof_sums(key: np.ndarray, day: np.ndarray, vals: list[np.ndarray]) -> list[np.ndarray]:
    """key ごとに、その日より前（同日を含まない）の値の累積。key<0 の行は 0。"""
    order = np.lexsort((day, key))
    k, d = key[order], day[order]
    start = np.r_[True, (k[1:] != k[:-1]) | (d[1:] != d[:-1])]
    seg = np.cumsum(start) - 1
    segkey = k[start]
    keystart = np.r_[True, segkey[1:] != segkey[:-1]]
    kid = np.cumsum(keystart) - 1
    outs = []
    for v in vals:
        segsum = np.bincount(seg, weights=v[order])
        cs = np.cumsum(segsum)
        excl = cs - segsum
        off = excl[keystart][kid]
        res = np.empty(len(key))
        res[order] = (excl - off)[seg]
        res[key < 0] = 0.0
        outs.append(res)
    return outs


class Lifts:
    def __init__(self, t: pd.DataFrame):
        fin = pd.to_numeric(t["finish_position"], errors="coerce").to_numpy(float)
        n = pd.to_numeric(t["field_size"], errors="coerce").to_numpy(float)
        ok = (fin > 0) & (n > 0)
        self.W = np.where(ok & (fin == 1), 1.0, 0.0)
        self.E = np.where(ok, 1.0 / np.where(n > 0, n, 1), 0.0)
        self.T = np.where(ok & (fin <= 3), 1.0, 0.0)
        self.E3 = np.where(ok, np.minimum(3.0, n) / np.where(n > 0, n, 1), 0.0)
        self.day = (t["date"].astype("int64") // 86_400_000_000_000).to_numpy(np.int64)

    def lift(self, key: np.ndarray, M: float) -> tuple[np.ndarray, np.ndarray]:
        w, e, t3, e3 = asof_sums(key, self.day, [self.W, self.E, self.T, self.E3])
        return np.log((w + M) / (e + M)), np.log((t3 + M) / (e3 + M))


def cell_key(a: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.where(a >= 0, a.astype(np.int64) * 1_000_000 + s, -1)


def build_te(t: pd.DataFrame, A, S) -> dict[str, np.ndarray]:
    L = Lifts(t)
    out: dict[str, np.ndarray] = {}
    marg: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}

    def m(a: str, M: float):
        if (a, M) not in marg:
            marg[(a, M)] = L.lift(np.where(A[a] >= 0, A[a], -1), M)
        return marg[(a, M)]

    for stem, kind, a, s in te_specs():
        M = M_ID if a in A_ID else M_STD
        if kind == "cell":
            cw, c3 = L.lift(cell_key(A[a], S[s]), M)
            mw, m3 = m(a, M)
            out[stem + "__win"] = (cw - mw).astype(np.float32)
            out[stem + "__t3"] = (c3 - m3).astype(np.float32)
        else:
            b = s
            key = np.where((A[a] >= 0) & (A[b] >= 0), A[a].astype(np.int64) * 1000 + A[b], -1)
            cw, c3 = L.lift(key, M_STD)
            aw, a3 = m(a, M_STD)
            bw_, b3 = m(b, M_STD)
            out[stem + "__win"] = (cw - aw - bw_).astype(np.float32)
            out[stem + "__t3"] = (c3 - a3 - b3).astype(np.float32)
        log(f"TE {stem}")
    return out


def verify_asof(t: pd.DataFrame, A, S, te: dict[str, np.ndarray], n: int = 150) -> float:
    """age×k3 / lr×going の win を、総当たりで日付を見て作り直した値と比べる。"""
    fin = pd.to_numeric(t["finish_position"], errors="coerce").to_numpy(float)
    nn = pd.to_numeric(t["field_size"], errors="coerce").to_numpy(float)
    ok = (fin > 0) & (nn > 0)
    date = t["date"].to_numpy()
    rng = np.random.default_rng(1)
    idx = rng.choice(np.flatnonzero(t["year"].to_numpy() >= 2022), n, replace=False)
    worst = 0.0
    for stem, a, s in (("te__age__k3__win", "age", "k3"), ("te__lr__going__win", "lr", "going")):
        for i in idx:
            if A[a][i] < 0:
                continue
            prior = (date < date[i]) & ok
            cell = prior & (A[a] == A[a][i]) & (S[s] == S[s][i])
            mg = prior & (A[a] == A[a][i])
            lc = np.log((((fin == 1) & cell).sum() + M_STD) / ((1.0 / nn)[cell].sum() + M_STD))
            lm = np.log((((fin == 1) & mg).sum() + M_STD) / ((1.0 / nn)[mg].sum() + M_STD))
            worst = max(worst, abs(float(lc - lm) - float(te[stem][i])))
    return worst


# ------------------------------------------------------------------ plan / build
def assemble():
    t = load_table()
    A, S = axis_codes(t)
    ax = add_axes(t)
    return t, A, S, ax


def plan_lists(t, A, S, ax):
    rows_eval = t["year"].isin(EVAL_YEARS).to_numpy()
    race = t["race_id"].to_numpy()
    cd = cond_defs(t, A, S)
    cn, cl, _ = dummies(cd, rows_eval)
    pl = {c: level_of(t[c].to_numpy(float), race) for c in PARTNERS}
    al = {k: level_of(v, race) for k, v in ax.items()}
    return cn, cl, pl, al


def cross_names(cn, cl, pl, al) -> list[tuple[str, str, str]]:
    """(列名, 軸, 相手)。軸が race 水準のものは、相手が horse 水準のときだけ。"""
    out = []
    for a, lv in al.items():
        if lv == "horse":
            out.append((f"x__{a}__single", a, ""))
        for n, l in zip(cn, cl):
            if lv == "race" and l == "race":
                continue
            out.append((f"x__{a}__{n}", a, "c:" + n))
        for p, l in pl.items():
            if lv == "race" and l == "race":
                continue
            out.append((f"x__{a}__{p}", a, "p:" + p))
    return out


def plan() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t, A, S, ax = assemble()
    cn, cl, pl, al = plan_lists(t, A, S, ax)
    te = [s[0] + sfx for s in te_specs() for sfx in ("__win", "__t3")]
    cr = cross_names(cn, cl, pl, al)
    man = {"K_te": len(te), "K_cross": len(cr), "K": len(te) + len(cr), "te": te, "conds": dict(zip(cn, cl)), "partners": pl, "axes": al,
           "n_axes_horse": sum(v == "horse" for v in al.values()), "n_axes_race": sum(v == "race" for v in al.values())}
    (OUT / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"K_te={man['K_te']} K_cross={man['K_cross']} K={man['K']}  軸 horse={man['n_axes_horse']} race={man['n_axes_race']}  条件 {len(cn)}  相手 {len(pl)}")


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t, A, S, ax = assemble()
    te = build_te(t, A, S)
    w = verify_asof(t, A, S, te)
    log(f"as-of 確認: 総当たりとの最大絶対差 {w:.2e}")
    rows = t["year"].to_numpy() >= 2022
    pre = t["year"].to_numpy() <= 2021
    F = t.loc[rows, ["race_id", "horse_id", "year", "finish_position", "field_size", "distance"] + CAT_CTX].reset_index(drop=True)
    for k, v in te.items():
        F[k] = v[rows]
    for k, v in ax.items():
        F["ax__" + k] = standardize(v, pre)[rows]
    for c in PARTNERS:
        F["pt__" + c] = standardize(t[c].to_numpy(float), pre)[rows]
    cd = cond_defs(t, A, S)
    for k, (cdv, lv) in cd.items():
        F["cd__" + k] = cdv[rows].astype(np.int16)
    rng = np.random.default_rng(11)
    for i in range(NOISE):
        F[f"noise__{i}"] = rng.standard_normal(len(F)).astype(np.float32)
    F["gate"] = t.loc[rows, "gate"].to_numpy(np.float32)
    F["a_style"] = t.loc[rows, "a_style"].to_numpy(np.float32)
    F["a_l3fz_ewm2"] = t.loc[rows, "a_l3fz_ewm2"].to_numpy(np.float32)
    F.to_parquet(OUT / "inter_features.parquet", index=False)
    (OUT / "asof_check.json").write_text(json.dumps({"max_abs_diff": w}), encoding="utf-8")
    log(f"書き出し {F.shape}")


# ------------------------------------------------------------------ 評価の土台
def load_eval(which: str = "oof") -> pd.DataFrame:
    F = pd.read_parquet(OUT / "inter_features.parquet")
    p = pd.read_parquet(OOF if which == "oof" else T25)
    p["race_id"], p["horse_id"] = p["race_id"].astype(str), p["horse_id"].astype(str)
    keep = ["race_id", "horse_id", "p_win"]
    d = F.merge(p[keep].drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="inner")
    d = d[d["year"].isin(EVAL_YEARS) if which == "oof" else d["year"] == 2025]
    d = d.sort_values("race_id", kind="stable").reset_index(drop=True)
    d["y"] = (pd.to_numeric(d["finish_position"], errors="coerce") == 1).astype(float)
    d["res"] = d["y"] - d["p_win"]
    return d


class RaceSums:
    def __init__(self, d: pd.DataFrame):
        r = d["race_id"].to_numpy()
        self.starts = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
        self.ry = d["year"].to_numpy()[self.starts]
        self.res = d["res"].to_numpy(float)
        self.w = (d["p_win"] * (1 - d["p_win"])).to_numpy(float)

    def stat(self, X: np.ndarray) -> dict:
        P = X * self.res[:, None]
        S = np.add.reduceat(P, self.starts, axis=0)
        z = S.sum(0) / np.sqrt((S ** 2).sum(0) + 1e-12)
        zy = {int(y): S[self.ry == y].sum(0) / np.sqrt((S[self.ry == y] ** 2).sum(0) + 1e-12) for y in EVAL_YEARS}
        beta = P.sum(0) / ((X ** 2) * self.w[:, None]).sum(0).clip(1e-12)
        return {"z": z, "zy": zy, "beta": beta}


def zthr(K: int) -> float:
    from scipy.stats import norm

    return float(norm.ppf(1 - 0.005 / K))


def score() -> None:
    man = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    K = man["K"]
    thr = zthr(K)
    d = load_eval("oof")
    d = d[d["surface"].isin(["芝", "ダート"])].reset_index(drop=True)
    rs = RaceSums(d)
    rows = []

    def emit(names: list[str], st: dict, X: np.ndarray, kind: str, extra=None):
        for j, nm in enumerate(names):
            yz = {y: round(float(st["zy"][y][j]), 2) for y in EVAL_YEARS}
            sg = {np.sign(v) for v in yz.values()}
            rows.append({"name": nm, "kind": kind, "n": int(len(d)), "sd_x": round(float(X[:, j].std()), 4), "z": round(float(st["z"][j]), 2),
                         "beta": round(float(st["beta"][j]), 4), "z_by_year": yz, "pass": bool(abs(st["z"][j]) >= thr and len(sg) == 1)})

    te_names = man["te"]
    X = d[te_names].to_numpy(np.float32)
    emit(te_names, rs.stat(X.astype(np.float64)), X, "te")
    log(f"TE {len(te_names)} 本 完了")
    nz = [f"noise__{i}" for i in range(NOISE)]
    Xn = d[nz].to_numpy(np.float32)
    st = rs.stat(Xn.astype(np.float64))
    noise = {"sd_z": round(float(np.std(st["z"])), 3), "max_abs_z": round(float(np.abs(st["z"]).max()), 2)}
    log(f"対照（乱数 {NOISE} 列）: {noise}")

    cn = list(man["conds"])
    cl = man["conds"]
    cmat = {}
    for n in cn:
        k, v = n.split("=")
        cmat[n] = (d["cd__" + k].to_numpy() == int(v)).astype(np.float32)
    pn = list(man["partners"])
    pmat = {p: d["pt__" + p].to_numpy(np.float32) for p in pn}
    for a, lv in man["axes"].items():
        x = d["ax__" + a].to_numpy(np.float32)
        names, cols = [], []
        if lv == "horse":
            names.append(f"x__{a}__single")
            cols.append(x)
        for n in cn:
            if lv == "race" and cl[n] == "race":
                continue
            names.append(f"x__{a}__{n}")
            cols.append(x * cmat[n])
        for p in pn:
            if lv == "race" and man["partners"][p] == "race":
                continue
            names.append(f"x__{a}__{p}")
            cols.append(x * pmat[p])
        X = np.column_stack(cols)
        emit(names, rs.stat(X.astype(np.float64)), X, "cross")
        log(f"交差 {a} ({lv}) {len(names)} 本")
    npass = sum(r["pass"] for r in rows)
    out = {"K": K, "z_threshold": round(thr, 3), "rows": len(rows), "n_pass": npass, "noise": noise, "results": rows}
    (OUT / "score_test.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    top = sorted(rows, key=lambda r: -abs(r["z"]))[:15]
    log(f"K={K} 閾値 z≥{thr:.2f}  通過 {npass}  上位:")
    for r in top:
        print(r["name"], r["z"], r["z_by_year"], "PASS" if r["pass"] else "", flush=True)


# ------------------------------------------------------------------ 残差の木
PRM = dict(objective="binary", learning_rate=0.03, num_leaves=8, min_data_in_leaf=400, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           lambda_l2=50.0, verbose=-1, num_threads=4, seed=7)


def _ll(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _fit_predict(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str]) -> np.ndarray:
    import lightgbm as lgb

    a, b = tr[cols].copy(), te[cols].copy()
    for c in CAT_CTX:
        if c in cols:
            cat = sorted(set(tr[c].astype(str)) | set(te[c].astype(str)))
            a[c] = pd.Categorical(tr[c].astype(str), categories=cat)
            b[c] = pd.Categorical(te[c].astype(str), categories=cat)
    lg = lambda p: np.log(p.clip(1e-6, 1 - 1e-6) / (1 - p.clip(1e-6, 1 - 1e-6)))
    m = lgb.train(PRM, lgb.Dataset(a, tr["y"], init_score=lg(tr["p_win"]).to_numpy()), num_boost_round=300)
    return 1 / (1 + np.exp(-(lg(te["p_win"]).to_numpy() + m.predict(b, raw_score=True))))


def groups(man: dict) -> dict[str, list[str]]:
    te = man["te"]
    g: dict[str, list[str]] = {"TE_ALL": te}
    for a in A_PLAIN + A_ID + list(A_SPECIAL):
        g[f"TE_{a}"] = [c for c in te if c.startswith(f"te__{a}__")]
    g["TE_PAIRS"] = [c for c in te if "x" in c.split("__")[1] and c.split("__")[1] not in A_PLAIN + A_ID + list(A_SPECIAL) and not c.startswith("te__style")]
    g["AX_WIND"] = ["ax__" + a for a in man["axes"] if a.startswith("wind_")]
    g["AX_SIMPLE"] = ["ax__" + a for a in man["axes"] if a.startswith("ax_")]
    g["AX_BIAS"] = ["ax__" + a for a in man["axes"] if a.startswith("aq_") and any(s in a for s in ("pred_b", "st_b", "sw_b", "d_b"))]
    g["AX_APT"] = ["ax__" + a for a in man["axes"] if a.startswith("aq_apt")]
    g["AX_RQ"] = ["ax__" + a for a in man["axes"] if a.startswith("aq_rq")]
    g["AX_ALL"] = ["ax__" + a for a in man["axes"]]
    return {k: v for k, v in g.items() if v}


def control_cols() -> list[str]:
    return CAT_CTX + ["distance", "field_size", "a_style", "a_l3fz_ewm2", "gate"] + ["pt__" + p for p in PARTNERS if p not in ("distance", "field_size", "gate", "a_style", "a_l3fz_ewm2")]


def trees() -> None:
    man = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    d = load_eval("oof")
    ctrl = control_cols()
    gains = {}
    for name, extra in [("control", [])] + list(groups(man).items()):
        cols = ctrl + extra
        g = []
        for hold in EVAL_YEARS:
            tr, te = d[d["year"] != hold], d[d["year"] == hold]
            pr = _fit_predict(tr, te, cols)
            g.append(_ll(te["y"].to_numpy(), pr) - _ll(te["y"].to_numpy(), te["p_win"].to_numpy()))
        gains[name] = g
        log(f"{name}: {[round(x, 6) for x in g]}")
    res = {}
    for name in gains:
        if name == "control":
            continue
        diff = [a - b for a, b in zip(gains[name], gains["control"])]
        res[name] = {"diff_vs_control": [round(x, 6) for x in diff], "mean_diff": round(float(np.mean(diff)), 6), "n_cols": len(groups(man)[name]),
                     "pass": bool(np.mean(diff) <= -0.0002 and all(x < 0 for x in diff))}
        print(name, res[name], flush=True)
    res["control"] = [round(x, 6) for x in gains["control"]]
    (OUT / "trees.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ 第 2 段
def tier2() -> None:
    man = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    sc = json.loads((OUT / "score_test.json").read_text(encoding="utf-8"))
    tj = json.loads((OUT / "trees.json").read_text(encoding="utf-8"))
    gmap = groups(man)
    bundle = [r["name"] for r in sc["results"] if r["pass"]]
    for g, v in tj.items():
        if g != "control" and v["pass"]:
            bundle += gmap[g]
    bundle = sorted(set(bundle))
    if not bundle:
        (OUT / "tier2_NOT_RUN.json").write_text(json.dumps({"reason": "第 1 段を通った候補が無いので test 2025 は使わない"}, ensure_ascii=False), encoding="utf-8")
        log("第 1 段を通った候補なし。test 2025 は使わない。")
        return
    tr = load_eval("oof")
    te = load_eval("t25")
    # 交差の列は組み立てる
    def mat(d: pd.DataFrame, names: list[str]) -> pd.DataFrame:
        cols = {}
        for nm in names:
            if nm.startswith("x__"):
                _, a, o = nm.split("__", 2)
                x = d["ax__" + a].to_numpy(np.float32)
                if o == "single":
                    cols[nm] = x
                elif o in man["partners"]:
                    cols[nm] = x * d["pt__" + o].to_numpy(np.float32)
                else:
                    k, v = o.split("=")
                    cols[nm] = x * (d["cd__" + k].to_numpy() == int(v)).astype(np.float32)
            else:
                cols[nm] = d[nm].to_numpy(np.float32)
        return pd.DataFrame(cols, index=d.index)

    ctrl = control_cols()
    ax_needed = sorted({nm.split("__")[1] for nm in bundle if nm.startswith("x__")})
    tr2 = pd.concat([tr, mat(tr, bundle)], axis=1)
    te2 = pd.concat([te, mat(te, bundle)], axis=1)
    tr2, te2 = tr2.loc[:, ~tr2.columns.duplicated()], te2.loc[:, ~te2.columns.duplicated()]
    y, base = te2["y"].to_numpy(), te2["p_win"].to_numpy()
    p0 = _fit_predict(tr2, te2, ctrl)
    p1 = _fit_predict(tr2, te2, ctrl + bundle)
    ll_i = lambda p: -(y * np.log(np.clip(p, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1)))
    dif = ll_i(p1) - ll_i(p0)
    races = te2["race_id"].to_numpy()
    uniq, inv = np.unique(races, return_inverse=True)
    num, cnt = np.bincount(inv, weights=dif), np.bincount(inv)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        ix = rng.integers(0, len(uniq), len(uniq))
        boots.append(num[ix].sum() / cnt[ix].sum())
    res = {"rows": int(len(te2)), "bundle_cols": len(bundle), "ll_base": _ll(y, base), "ll_control": _ll(y, p0), "ll_bundle": _ll(y, p1),
           "diff_vs_control": float(dif.mean()), "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))], "adopt": bool(dif.mean() <= -0.0002),
           "bundle": bundle[:200], "axes_in_bundle": ax_needed}
    (OUT / "tier2_TEST2025_EVALUATED.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log(json.dumps({k: v for k, v in res.items() if k not in ("bundle",)}, ensure_ascii=False))


if __name__ == "__main__":
    {"plan": plan, "build": build, "score": score, "trees": trees, "tier2": tier2}[sys.argv[1]]()
