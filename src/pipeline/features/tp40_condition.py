"""tp-40: 調教（追い切り）の時計の出方から、馬の『調子・成長』を読む特徴量。

追い切り 1 本ごとに、(種別×所属×強度)ごとの直前 12 か月の分布で標準化した時計（training_patterns.add_reference_z）を『良さ』
g = -(t4z, t1z の平均)（大きい = 速い）、g_t1 = -t1z、g_acc = -accelz（大きい = 終いに加速）にする。調教師の癖を引いた g_tr も作る。

群:
  CY  今回の追い切りサイクル（前走後〜今回）の中の推移: 最終・1 本前・2 本前、上昇度 d1、サイクル全体の上昇 dcyc、傾き、強い追い切りだけの推移
  CYt 同上を調教師の癖で補正した g_tr で
  SF  自分の過去の追い切り（前のサイクルまで）との比較: 平均比・自己ベスト比・同じコース/種別比・自己ベスト更新
  TJ  サイクルごとの最終追い切りの推移（直近 4 サイクルの傾き・前サイクル比）
  LY  休み明けの仕上がり曲線: 先頭 4 本の時計、強い追い切りまでの本数、強度の上がり方、休養帯×本数ごとの母集団との差
  PO  母集団曲線: 年齢×性別×（最終の良さ・サイクルの上昇）の帯ごとの、過去のレースの『着順パーセンタイル』の平均（履歴の無い馬にも使える）
  IN  馬ごとの傾き: その馬の過去のレースで『最終の良さ（またはサイクルの上昇）と着順パーセンタイル』がどう結びついていたか（縮約）× 今回の値

すべて as-of: 追い切りの標準化はその月より前の 12 か月だけ、サイクルの特徴はその追い切りのみ、過去との比較は『前のサイクルまで』、
成績に基づく集計（PO・IN）は『日付が厳密に前』のレースだけ。累積は『日ごとに合計 → キーごとに cumsum → 1 つずらす』（累積和から自分を引かない）。
キー: 馬は horse_id、調教師は trainer_id。市場オッズ・人気は使わない。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.pipeline.features import training_patterns as TP

KEYC = ["horse_id", "race_date"]
KAPPA = 5.0                      # IN の縮約の強さ（観測 5 件分）
M_POP = 60.0                     # PO の縮約の強さ
M_TRN = 30.0                     # 調教師の癖の縮約の強さ
Z_CUTS = (-1.0, -0.35, 0.35, 1.0)
REST_CUTS = (28, 56, 120)


# ---------------------------------------------------------------- as-of の道具
def _codes(*cols) -> np.ndarray:
    """複数の列を組にしたキーの整数コード。いずれかが欠損なら -1。"""
    parts = [pd.Series(c).reset_index(drop=True) for c in cols]
    ok = np.ones(len(parts[0]), dtype=bool)
    for p in parts:
        ok &= p.notna().to_numpy()
    out = np.full(len(parts[0]), -1, dtype=np.int64)
    if ok.any():
        mi = pd.MultiIndex.from_arrays([p[ok].astype(str).to_numpy() for p in parts])
        out[ok] = pd.factorize(mi)[0]
    return out


def prior_sums(code: np.ndarray, day: np.ndarray, V: np.ndarray) -> np.ndarray:
    """各行に、同じキー（code >= 0）で『日付が厳密に前』の行の V の合計を返す（code < 0 の行は 0）。日ごとに合計 → cumsum → 1 つずらす。"""
    n, k = V.shape
    out = np.zeros((n, k))
    use = code >= 0
    if not use.any():
        return out
    df = pd.DataFrame(V[use], columns=range(k))
    df["k"] = code[use]
    df["d"] = day[use]
    g = df.groupby(["k", "d"], sort=True).sum()
    cum = g.groupby(level=0).cumsum()
    prior = cum.groupby(level=0).shift(1).fillna(0.0)
    out[use] = prior.reindex(pd.MultiIndex.from_arrays([code[use], day[use]])).to_numpy()
    return out


def _bin(x: pd.Series, cuts=Z_CUTS) -> pd.Series:
    """帯の番号（0..len(cuts)）。欠損は -1。"""
    b = np.digitize(x.to_numpy(dtype=float), np.asarray(cuts))
    return pd.Series(np.where(x.notna().to_numpy(), b, -1), index=x.index)


# ---------------------------------------------------------------- 追い切り 1 本ごと
def prepare_workouts(laps: pd.DataFrame, runs: pd.DataFrame, with_ref: bool = True) -> pd.DataFrame:
    """追い切り（時計あり）1 本 1 行。g（良さ）・g_t1・g_acc・g_tr・強度・サイクルのキー（horse_id, race_date）・調教師。
    with_ref=False のときは laps が t4z・t1z・accelz を持つこと（テスト用）。"""
    t = TP.prepare(laps)
    t = t[t["race_date"] >= "2011-01-01"].copy()
    if with_ref:
        t = TP.add_reference_z(t)
    t["horse_id"] = t["horse_id"].astype(str)
    t["train_day"] = pd.to_datetime(t["train_date"]).to_numpy("datetime64[D]").astype(np.int64)
    t["g_t4"], t["g_t1"], t["g_acc"] = -t["t4z"], -t["t1z"], -t["accelz"]
    t["g"] = t[["g_t4", "g_t1"]].mean(axis=1)
    t["is_hard"] = (t["intensity"].fillna(0) >= 1).astype(int)
    t = t[t["timed"] & t["g"].notna()].copy()
    tr = runs[["horse_id", "date", "trainer_id"]].copy()
    tr["race_date"] = pd.to_datetime(tr["date"]).dt.strftime("%Y-%m-%d")
    tr["horse_id"] = tr["horse_id"].astype(str)
    tr = tr.drop_duplicates(["horse_id", "race_date"])[["horse_id", "race_date", "trainer_id"]]
    t = t.merge(tr, on=KEYC, how="left")
    t["trainer_id"] = t["trainer_id"].astype("string")
    t = add_trainer_adjusted(t)
    return t.reset_index(drop=True)


def add_trainer_adjusted(T: pd.DataFrame) -> pd.DataFrame:
    """g_tr = g - 調教師×種別の『いつもの良さ』。基準は『そのレースの日より前に走ったレースに紐づく追い切り』の平均を 0 へ縮約したもの
    （保存済みの履歴だけで再現できる定義。まだ走っていないレースに紐づく追い切りは、先に行われていても使わない）。"""
    T = T.copy()
    code = np.where(T["trainer_id"].notna().to_numpy(), _codes(T["trainer_id"], T["kind"]), -1)
    V = np.column_stack([np.ones(len(T)), T["g"].to_numpy(dtype=float)])
    S = prior_sums(code, pd.to_datetime(T["race_date"]).to_numpy("datetime64[D]").astype(np.int64), V)
    habit = S[:, 1] / (S[:, 0] + M_TRN)
    T["habit"] = np.where(code >= 0, habit, np.nan)
    T["g_tr"] = T["g"] - T["habit"]
    return T


# ---------------------------------------------------------------- サイクルの中の推移
def _order(T: pd.DataFrame) -> pd.DataFrame:
    T = T.sort_values(KEYC + ["train_day", "days_before", "t1"], ascending=[True, True, True, False, True], kind="stable")
    g = T.groupby(KEYC, sort=False)
    T = T.assign(k_start=g.cumcount().to_numpy(), k_end=g.cumcount(ascending=False).to_numpy())
    return T


def _pick(T: pd.DataFrame, mask: pd.Series, cols: list[str], suffix: str) -> pd.DataFrame:
    s = T.loc[mask, KEYC + cols].copy()
    s.columns = KEYC + [f"{c}_{suffix}" for c in cols]
    return s


def _slope(T: pd.DataFrame, m: str, min_n: int = 3) -> pd.DataFrame:
    """サイクル内の m の傾き（1 週あたり）。本数が min_n 未満、または日付の散らばりが無ければ欠損。"""
    t = (T["train_day"] - T.groupby(KEYC)["train_day"].transform("min")).to_numpy(dtype=float) / 7.0
    y = T[m].to_numpy(dtype=float)
    d = pd.DataFrame({"horse_id": T["horse_id"].to_numpy(), "race_date": T["race_date"].to_numpy(), "n": 1.0, "t": t, "y": y, "ty": t * y, "tt": t * t})
    s = d.groupby(KEYC, sort=False).sum().reset_index()
    den = s["n"] * s["tt"] - s["t"] ** 2
    num = s["n"] * s["ty"] - s["t"] * s["y"]
    sl = np.where((s["n"] >= min_n) & (den > 1e-9), num / np.where(den > 1e-9, den, 1.0), np.nan)
    return pd.DataFrame({"horse_id": s["horse_id"], "race_date": s["race_date"], "slope": sl})


def cycle_metric(T: pd.DataFrame, m: str, p: str, slopes: bool = True) -> pd.DataFrame:
    """サイクル（horse_id, race_date）ごとの m の推移。p は列名の接頭辞。"""
    S = _order(T[T[m].notna()])
    base = S.groupby(KEYC, sort=False).agg(n=(m, "size"), best=(m, "max"), tot=(m, "sum"), first_day=("train_day", "min"), last_day=("train_day", "max")).reset_index()
    out = base[KEYC].copy()
    out[f"{p}_n"] = base["n"].to_numpy()
    last = _pick(S, S["k_end"] == 0, [m, "train_day"], "last")
    prev = _pick(S, S["k_end"] == 1, [m, "train_day"], "prev")
    prev2 = _pick(S, S["k_end"] == 2, [m, "train_day"], "prev2")
    first = _pick(S, S["k_start"] == 0, [m], "first")
    out = out.merge(last, on=KEYC, how="left").merge(prev, on=KEYC, how="left").merge(prev2, on=KEYC, how="left").merge(first, on=KEYC, how="left")
    out = out.merge(base[KEYC + ["best", "tot", "n"]], on=KEYC, how="left")
    L, P1, P2, F = out[f"{m}_last"], out[f"{m}_prev"], out[f"{m}_prev2"], out[f"{m}_first"]
    res = out[KEYC].copy()
    res[f"{p}_n"] = out[f"{p}_n"].to_numpy()
    res[f"{p}_last"] = L.to_numpy()
    res[f"{p}_d1"] = (L - P1).to_numpy()
    res[f"{p}_d2"] = (P1 - P2).to_numpy()
    res[f"{p}_curv"] = ((L - P1) - (P1 - P2)).to_numpy()
    res[f"{p}_dcyc"] = np.where(out["n"] >= 2, L - F, np.nan)
    res[f"{p}_vsmean"] = np.where(out["n"] >= 2, L - (out["tot"] - L) / (out["n"] - 1).clip(lower=1), np.nan)
    res[f"{p}_best"] = out["best"].to_numpy()
    dd = (out["train_day_last"] - out["train_day_prev2"]).to_numpy(dtype=float)
    res[f"{p}_slope3"] = np.where(dd > 0, (L - P2).to_numpy() / np.where(dd > 0, dd, 1.0) * 7.0, np.nan)
    res[f"{p}_gap"] = (out["train_day_last"] - out["train_day_prev"]).to_numpy(dtype=float)
    if slopes:
        res = res.merge(_slope(S, m).rename(columns={"slope": f"{p}_slope"}), on=KEYC, how="left")
    return res


def cycle_group(T: pd.DataFrame, metrics: dict[str, str]) -> pd.DataFrame:
    out = None
    for m, p in metrics.items():
        r = cycle_metric(T, m, p)
        out = r if out is None else out.merge(r, on=KEYC, how="outer")
    hard = cycle_metric(T[T["is_hard"] == 1], "g", "ccH", slopes=False)[KEYC + ["ccH_n", "ccH_last", "ccH_d1", "ccH_dcyc"]]
    return out.merge(hard, on=KEYC, how="left")


# ---------------------------------------------------------------- 自分の過去との比較
def _cycle_table(T: pd.DataFrame) -> pd.DataFrame:
    S = _order(T)
    s = S.assign(gh=S["g"].where(S["is_hard"] == 1))
    c = s.groupby(KEYC, sort=False).agg(n=("g", "size"), tot=("g", "sum"), mx=("g", "max"), nh=("is_hard", "sum"), mxh=("gh", "max"), toth=("gh", "sum")).reset_index()
    last = S[S["k_end"] == 0][KEYC + ["g", "course", "kind", "days_before", "intensity"]].rename(columns={"g": "g_last", "course": "c_last", "kind": "k_last", "days_before": "db_last", "intensity": "int_last"})
    return c.merge(last, on=KEYC, how="left").sort_values(KEYC, kind="stable").reset_index(drop=True)


def _prior_cum(df: pd.DataFrame, by: list[str], col: str, kind: str = "sum") -> pd.Series:
    """by ごと（行は race_date 昇順）に、『前の行まで』の累積（sum / max）。累積から自分を引かず、1 行ずらす。"""
    g = df.groupby(by, sort=False)[col]
    cum = g.cumsum() if kind == "sum" else g.cummax()
    keys = [df[b] for b in by]
    if kind != "sum":
        cum = cum.groupby(keys, sort=False).ffill()          # 強い追い切りの無いサイクルの NaN で、それ以前の最大値を落とさない
    return cum.groupby(keys, sort=False).shift(1)


def self_group(T: pd.DataFrame, C: pd.DataFrame | None = None) -> pd.DataFrame:
    """C: _cycle_table(T)。自分の前のサイクルまでの追い切りとの比較。"""
    C = _cycle_table(T) if C is None else C
    c = C.copy()
    c["toth0"] = c["toth"].fillna(0.0)
    out = c[KEYC].copy()
    hn = _prior_cum(c, ["horse_id"], "n")
    out["sf_n"] = hn.fillna(0.0).to_numpy()
    hmean = _prior_cum(c, ["horse_id"], "tot") / hn
    hbest = _prior_cum(c, ["horse_id"], "mx", "max")
    hnh = _prior_cum(c, ["horse_id"], "nh")
    hbesth = _prior_cum(c.assign(mxh=c["mxh"]), ["horse_id"], "mxh", "max")
    hmeanh = _prior_cum(c, ["horse_id"], "toth0") / hnh.where(hnh > 0)
    out["sf_last_vs_mean"] = (c["g_last"] - hmean).to_numpy()
    out["sf_last_vs_best"] = (c["g_last"] - hbest).to_numpy()
    out["sf_cycbest_vs_best"] = (c["mx"] - hbest).to_numpy()
    out["sf_hard_vs_best_hard"] = (c["mxh"] - hbesth).to_numpy()
    out["sf_hard_vs_mean_hard"] = (c["mxh"] - hmeanh).to_numpy()
    out["sf_is_pr"] = np.where(hbest.notna() & c["g_last"].notna(), (c["g_last"] > hbest).astype(float), np.nan)
    out["sf_hist_best"] = hbest.to_numpy()
    out["sf_hist_mean"] = hmean.to_numpy()
    return out


def _same_key_vs(T: pd.DataFrame, C: pd.DataFrame, key: str, last_key: str) -> pd.DataFrame:
    """最終追い切りと同じ key（コース名 or 種別）の、自分の前のサイクルまでの追い切りの平均との差と件数。"""
    g = T.groupby(["horse_id", key, "race_date"], sort=False).agg(n=("g", "size"), tot=("g", "sum")).reset_index().sort_values(["horse_id", key, "race_date"], kind="stable").reset_index(drop=True)
    g["pn"] = _prior_cum(g, ["horse_id", key], "n")
    g["pt"] = _prior_cum(g, ["horse_id", key], "tot")
    q = C[KEYC + ["g_last", last_key]].rename(columns={last_key: key})
    q = q.merge(g[["horse_id", key, "race_date", "pn", "pt"]], on=["horse_id", key, "race_date"], how="left")
    tag = "course" if key == "course" else "kind"
    q[f"sf_{tag}_n"] = q["pn"].fillna(0.0)
    q[f"sf_{tag}_vs_mean"] = q["g_last"] - q["pt"] / q["pn"].where(q["pn"] > 0)
    return q[KEYC + [f"sf_{tag}_n", f"sf_{tag}_vs_mean"]]


def self_group_full(T: pd.DataFrame) -> pd.DataFrame:
    C = _cycle_table(T)
    out = self_group(T, C)
    out = out.merge(_same_key_vs(T, C, "course", "c_last"), on=KEYC, how="left").merge(_same_key_vs(T, C, "kind", "k_last"), on=KEYC, how="left")
    return out


# ---------------------------------------------------------------- サイクルごとの最終追い切りの推移
def traj_group(T: pd.DataFrame, C: pd.DataFrame | None = None) -> pd.DataFrame:
    C = _cycle_table(T) if C is None else C
    c = C[KEYC + ["g_last"]].copy()
    c["rd"] = pd.to_datetime(c["race_date"]).to_numpy("datetime64[D]").astype(np.int64)
    g = c.groupby("horse_id", sort=False)
    Y = np.column_stack([g["g_last"].shift(3), g["g_last"].shift(2), g["g_last"].shift(1), c["g_last"]]).astype(float)
    idx = np.arange(4, dtype=float)
    ok = ~np.isnan(Y)
    n = ok.sum(1)
    sx = (ok * idx).sum(1)
    sy = np.where(ok, Y, 0.0).sum(1)
    sxy = (np.where(ok, Y, 0.0) * idx).sum(1)
    sxx = (ok * idx * idx).sum(1)
    den = n * sxx - sx ** 2
    sl = np.where((n >= 3) & ok[:, 3] & (den > 1e-9), (n * sxy - sx * sy) / np.where(den > 1e-9, den, 1.0), np.nan)
    out = c[KEYC].copy()
    out["tj_vs_prev"] = (c["g_last"] - g["g_last"].shift(1)).to_numpy()
    out["tj_slope4"] = sl
    allnan = np.isnan(Y[:, :3]).all(1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prior3 = np.where(allnan, np.nan, np.nanmean(np.where(allnan[:, None], 0.0, Y[:, :3]), axis=1))
    out["tj_vs_mean3"] = (c["g_last"].to_numpy() - prior3)
    out["tj_prior_max3"] = np.where(allnan, np.nan, np.nanmax(np.where(np.isnan(Y[:, :3]), -np.inf, Y[:, :3]), axis=1))
    out["tj_days_prev"] = (c["rd"] - g["rd"].shift(1)).to_numpy(dtype=float)
    out["tj_n_cycles"] = g.cumcount().to_numpy(dtype=float)
    return out


# ---------------------------------------------------------------- 休み明けの仕上がり曲線
def layoff_group(T: pd.DataFrame, runs_rest: pd.DataFrame | None = None) -> pd.DataFrame:
    """先頭 4 本の時計・強い追い切りまでの本数・強度の上がり方。runs_rest（horse_id, race_date, d_rest）があれば、休養帯×本数ごとの母集団の最終の良さとの差も。"""
    S = _order(T)
    out = S.groupby(KEYC, sort=False).agg(ly_n=("g", "size"), ly_hard_n=("is_hard", "sum")).reset_index()
    for k in range(4):
        out = out.merge(_pick(S, S["k_start"] == k, ["g"], f"k{k + 1}").rename(columns={f"g_k{k + 1}": f"ly_g{k + 1}"}), on=KEYC, how="left")
    fh = S[S["is_hard"] == 1].sort_values(KEYC + ["train_day"], kind="stable").groupby(KEYC, sort=False).head(1)[KEYC + ["k_start", "days_before"]]
    out = out.merge(fh.rename(columns={"k_start": "ly_n_before_hard", "days_before": "ly_first_hard_db"}), on=KEYC, how="left")
    it = S[S["intensity"].notna()].copy()
    t = it["k_start"].to_numpy(dtype=float)
    y = it["intensity"].to_numpy(dtype=float)
    d = pd.DataFrame({"horse_id": it["horse_id"].to_numpy(), "race_date": it["race_date"].to_numpy(), "n": 1.0, "t": t, "y": y, "ty": t * y, "tt": t * t}).groupby(KEYC, sort=False).sum().reset_index()
    den = d["n"] * d["tt"] - d["t"] ** 2
    out = out.merge(pd.DataFrame({"horse_id": d["horse_id"], "race_date": d["race_date"], "ly_ramp": np.where((d["n"] >= 3) & (den > 1e-9), (d["n"] * d["ty"] - d["t"] * d["y"]) / np.where(den > 1e-9, den, 1.0), np.nan)}), on=KEYC, how="left")
    last = S[S["k_end"] == 0][KEYC + ["intensity"]].rename(columns={"intensity": "ly_int_last"})
    prev = S[S["k_end"] == 1][KEYC + ["intensity"]].rename(columns={"intensity": "ly_int_prev"})
    out = out.merge(last, on=KEYC, how="left").merge(prev, on=KEYC, how="left")
    out["ly_int_step"] = out["ly_int_last"] - out["ly_int_prev"]
    if runs_rest is not None:
        lastg = S[S["k_end"] == 0][KEYC + ["g"]].rename(columns={"g": "g_last"})
        q = out[KEYC + ["ly_n"]].merge(lastg, on=KEYC, how="left").merge(runs_rest, on=KEYC, how="left")
        band = pd.Series(np.digitize(q["d_rest"].to_numpy(dtype=float), REST_CUTS), index=q.index).where(q["d_rest"].notna(), -1)
        nb = q["ly_n"].clip(upper=6)
        code = np.where(q["g_last"].notna(), _codes(band, nb), -1)
        day = pd.to_datetime(q["race_date"]).to_numpy("datetime64[D]").astype(np.int64)
        V = np.column_stack([np.ones(len(q)), q["g_last"].fillna(0.0).to_numpy()])
        Sg = prior_sums(code, day, V)
        pm = np.where(Sg[:, 0] >= 50, Sg[:, 1] / np.maximum(Sg[:, 0], 1.0), np.nan)
        q["ly_res"] = q["g_last"].to_numpy() - pm
        out = out.merge(q[KEYC + ["ly_res"]], on=KEYC, how="left")
    return out


# ---------------------------------------------------------------- 成績との結びつき（母集団・馬ごと）
def _age_sex(sex_age: pd.Series) -> tuple[pd.Series, pd.Series]:
    s = sex_age.astype(str)
    age = pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce").clip(upper=5)
    sex = s.str.extract(r"^([^\d]+)")[0]
    return age, sex


def link_frame(runs: pd.DataFrame, cyc: pd.DataFrame) -> pd.DataFrame:
    """runs（race_id, horse_id, date, finish_position, field_size, sex_age, d_rest）に、サイクルの x1（最終の良さ）・x2（サイクルの上昇）と、
    y（着順パーセンタイル 1=1 着, 0=最下位）を付ける。"""
    r = runs[["race_id", "horse_id", "date", "finish_position", "field_size", "sex_age", "d_rest"]].copy()
    r["horse_id"] = r["horse_id"].astype(str)
    r["race_id"] = r["race_id"].astype(str)
    r["race_date"] = pd.to_datetime(r["date"]).dt.strftime("%Y-%m-%d")
    r["day"] = pd.to_datetime(r["date"]).to_numpy("datetime64[D]").astype(np.int64)
    fs = r["field_size"].to_numpy(dtype=float)
    r["y"] = np.where(fs > 1, 1.0 - (r["finish_position"].to_numpy(dtype=float) - 1.0) / np.maximum(fs - 1.0, 1.0), np.nan)
    r["age"], r["sex"] = _age_sex(r["sex_age"])
    c = cyc[KEYC + ["cc_last", "cc_dcyc"]].rename(columns={"cc_last": "x1", "cc_dcyc": "x2"})
    r = r.merge(c, on=KEYC, how="left")
    return r.sort_values(["day", "race_id", "horse_id"], kind="stable").reset_index(drop=True)


def pop_group(R: pd.DataFrame) -> pd.DataFrame:
    """R: link_frame。年齢×性別×（x1 の帯, x2 の帯）ごとの、日付が厳密に前のレースの y の平均（親の帯へ縮約）から全体平均を引いた値。"""
    day = R["day"].to_numpy()
    ok = (R["x1"].notna() & R["y"].notna()).to_numpy()
    b1, b2 = _bin(R["x1"]), _bin(R["x2"])
    rb = pd.Series(np.digitize(R["d_rest"].to_numpy(dtype=float), REST_CUTS), index=R.index).where(R["d_rest"].notna(), -1)
    V = np.column_stack([np.ones(len(R)), R["y"].fillna(0.0).to_numpy()]) * ok[:, None]
    keyed = lambda *cols: np.where(R["x1"].notna().to_numpy(), _codes(*cols), -1)   # noqa: E731
    S_all = prior_sums(np.zeros(len(R), dtype=np.int64), day, V)
    mu = S_all[:, 1] / np.maximum(S_all[:, 0], 1.0)
    out = R[["race_id", "horse_id"]].copy()
    for tag, parent_cols, cell_cols in (("po", (R["age"], b1), (R["age"], R["sex"], b1, b2)), ("po2", (rb, b1), (rb, b1, b2))):
        Sp = prior_sums(keyed(*parent_cols), day, V)
        Sc = prior_sums(keyed(*cell_cols), day, V)
        ep = (Sp[:, 1] + M_POP * mu) / (Sp[:, 0] + M_POP)
        ec = (Sc[:, 1] + M_POP * ep) / (Sc[:, 0] + M_POP)
        have = (S_all[:, 0] >= 200) & R["x1"].notna().to_numpy()
        out[f"{tag}_cell"] = np.where(have, ec - mu, np.nan)
        out[f"{tag}_n"] = np.where(have, Sc[:, 0], np.nan)
    return out


def indiv_group(R: pd.DataFrame, kappa: float = KAPPA) -> pd.DataFrame:
    """馬ごとの傾き: その馬の過去のレース（前のレースまで）で、x と y の共分散 / (分散 + kappa)。補正ポイント = 傾き × (今回の x - その馬の過去の x の平均)。"""
    R = R.sort_values(["horse_id", "day", "race_id"], kind="stable").reset_index()
    out = R[["race_id", "horse_id"]].copy()
    for tag, xc in (("in1", "x1"), ("in2", "x2")):
        ok = (R[xc].notna() & R["y"].notna()).to_numpy().astype(float)
        x = R[xc].fillna(0.0).to_numpy() * ok
        y = R["y"].fillna(0.0).to_numpy() * ok
        inc = pd.DataFrame({"horse_id": R["horse_id"].to_numpy(), "n": ok, "sx": x, "sy": y, "sxy": x * y, "sxx": x * x})
        pri = inc.groupby("horse_id", sort=False).cumsum().groupby(inc["horse_id"], sort=False).shift(1).fillna(0.0)
        n, sx, sy, sxy, sxx = (pri[c].to_numpy() for c in ("n", "sx", "sy", "sxy", "sxx"))
        nn = np.maximum(n, 1.0)
        cov = sxy - sx * sy / nn
        var = np.maximum(sxx - sx * sx / nn, 0.0)
        beta = np.where(n >= 2, cov / (var + kappa), 0.0)
        xnow = R[xc].to_numpy(dtype=float)
        out[f"{tag}_cp"] = np.where((n >= 2) & ~np.isnan(xnow), beta * (xnow - sx / nn), np.nan)
        out[f"{tag}_beta"] = np.where(n >= 2, beta, np.nan)
        out[f"{tag}_rel"] = np.where(n >= 2, var / (var + kappa), np.nan)
        out[f"{tag}_n"] = n
    return out.set_index(R["index"]).sort_index()


# ---------------------------------------------------------------- 全体
GROUP_PREFIX = {g: "wc_" for g in ("CY", "CYt", "SF", "TJ", "LY", "PO", "IN")}


def build_features(runs: pd.DataFrame, laps: pd.DataFrame, with_ref: bool = True) -> dict[str, pd.DataFrame]:
    """群 → (race_id, horse_id) を持つ表。runs: race_id, horse_id, date, trainer_id, finish_position, field_size, sex_age, d_rest。"""
    T = prepare_workouts(laps, runs, with_ref=with_ref)
    r = runs.copy()
    r["horse_id"] = r["horse_id"].astype(str)
    r["race_id"] = r["race_id"].astype(str)
    r["race_date"] = pd.to_datetime(r["date"]).dt.strftime("%Y-%m-%d")
    keymap = r[["race_id", "horse_id", "race_date"]].drop_duplicates(["horse_id", "race_date"])
    cy = cycle_group(T, {"g": "cc", "g_t1": "c1", "g_acc": "ca"})
    cyt = cycle_metric(T, "g_tr", "ct")
    sf = self_group_full(T)
    tj = traj_group(T)
    ly = layoff_group(T, r[["horse_id", "race_date", "d_rest"]].drop_duplicates(KEYC))
    R = link_frame(r, cy)
    po = pop_group(R)
    ind = indiv_group(R)

    out: dict[str, pd.DataFrame] = {}
    for name, t in (("CY", cy), ("CYt", cyt), ("SF", sf), ("TJ", tj), ("LY", ly)):
        m = r[["race_id", "horse_id", "race_date"]].merge(t, on=KEYC, how="left").drop(columns=["race_date"])
        m = m.rename(columns={c: GROUP_PREFIX[name] + c for c in m.columns if c not in ("race_id", "horse_id")})
        out[name] = m
    for name, t in (("PO", po), ("IN", ind)):
        m = r[["race_id", "horse_id"]].merge(t, on=["race_id", "horse_id"], how="left")
        m = m.rename(columns={c: GROUP_PREFIX[name] + c for c in m.columns if c not in ("race_id", "horse_id")})
        out[name] = m
    return out
