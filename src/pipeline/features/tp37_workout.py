"""tp-37: 追い切りの『残りの余地』を特徴量にする（調教師ごとの時計の標準化・交互作用・本数/間隔/前走比・追い切り騎手・短評）。

すべて as-of。追い切りの時計は『その追い切りの日より前』の追い切りだけで標準化する（同日の他馬の時計も使わない）。
成績（勝・複勝・着順残差）の集計は『そのレース日より前』に確定したレースだけ（tp28/tp35 の Agg を再利用）。
累積は『日ごとに合計 → キーごとに cumsum → 1 つずらす』で作る（累積和から自分を引かない。端数リークを避ける）。
キー: 調教師は trainer_id、馬は horse_id。追い切りの騎手は名前しか無い（ID なし）ので騎手名の文字列を、追い切り表の中のカテゴリとしてだけ使う
（レースの騎手とは結合しない）。市場オッズ・人気は使わない。
"""

from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

from src.pipeline.features import tp28_workout_stats as T
from src.pipeline.features import tp35_lap_groups as L35
from src.pipeline.features import training_patterns as TP

TIMED = ("slope", "wood", "poly")
METRICS = ("t4", "t1", "acc")                          # 4F の時計 / 終い 1F / 加速度（終い 1F − 直前 2F の 1F 平均。負 = 加速）
CENTER = {"t4": 54.0, "t1": 12.8, "acc": 0.0}          # 累積を中心化する固定の定数（データに依存しない）
SLOT_F, SLOT_W1 = (1, 6), (7, 13)
COND = {"良": "g", "稍": "y", "重": "h", "不": "h"}
MIN_KEY0 = 200                                          # 標準化の基準（種別×強度×馬場）の最低件数
MIN_TRAINER = 20                                        # 調教師の過去の追い切りがこれ未満なら調教師内の値は出さない
M_TRAINER, M_CELL = 30.0, 15.0                          # 縮約の強さ（調教師全体 / 調教師×種別×強度）
N_CLUSTER, CLUSTER_MIN = 5, 100
Z_CUTS = (-1.0, -0.35, 0.35, 1.0)
HABIT_CUTS = (-0.15, 0.15)
STYLE_CUTS = (0.27, 0.45, 0.64)                          # a_style の四分位（固定値）
RIDER_CLASS = {"": 0, "助手": 1, "師": 2, "見習": 3}


# ---------------------------------------------------------------- as-of の道具
def prior_sums(code: np.ndarray, day: np.ndarray, V: np.ndarray) -> np.ndarray:
    """各行に、同じキー（code >= 0）で『日付が厳密に前』の行の V の合計を返す（code < 0 の行は 0）。
    日ごとに合計 → キーごとに cumsum → 1 つずらす。同日・自分の行は入らない。累積和から自分を引くことはしない。"""
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


def _codes(*cols) -> np.ndarray:
    """複数の列（文字列/数値。欠損なら -1）を組にしたキーの整数コード。"""
    parts = [pd.Series(c) for c in cols]
    ok = np.ones(len(parts[0]), dtype=bool)
    for p in parts:
        ok &= p.notna().to_numpy()
    s = parts[0].astype(str)
    for p in parts[1:]:
        s = s + "|" + p.astype(str)
    codes, _ = pd.factorize(s.where(ok))
    return codes.astype(np.int64)


def _moments(code: np.ndarray, day: np.ndarray, X: np.ndarray, centers: np.ndarray):
    """X（n × k、欠損は NaN）の as-of の件数・合計・2 乗和（centers で中心化）。"""
    ok = np.isfinite(X)
    x0 = np.where(ok, X - centers, 0.0)
    S = prior_sums(code, day, np.hstack([ok.astype(float), x0, x0 * x0]))
    k = X.shape[1]
    return S[:, :k], S[:, k:2 * k], S[:, 2 * k:]


# ---------------------------------------------------------------- 追い切り表
def prepare_workouts(laps: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """追い切り 1 本 1 行（レースの前日まで）。ラップ種別・強度・馬場・曜日・調教師を付ける。"""
    t = laps[laps["days_before"] >= 1].copy()
    t = L35.prepare_laps(t)
    t["train_day"] = pd.to_datetime(t["train_date"]).to_numpy("datetime64[D]").astype(np.int64)
    t["intensity"] = t["leg_color"].map(TP.INTENSITY)
    t["timed"] = t["kind"].isin(TIMED) & t["t1"].notna()
    t["condc"] = t["cond"].map(COND).fillna("x")
    t["x_t4"] = t["t4"]
    t["x_t1"] = t["t1"]
    t["x_acc"] = t["accel"]
    for c in ("x_t4", "x_t1", "x_acc"):
        t.loc[~t["timed"], c] = np.nan
    tr = runs[["horse_id", "date", "trainer_id"]].copy()
    tr["race_date"] = pd.to_datetime(tr["date"]).dt.strftime("%Y-%m-%d")
    tr["horse_id"] = tr["horse_id"].astype(str)
    tr = tr.drop_duplicates(["horse_id", "race_date"])[["horse_id", "race_date", "trainer_id"]]
    t["horse_id"] = t["horse_id"].astype(str)
    t = t.merge(tr, on=["horse_id", "race_date"], how="left")
    t["trainer_id"] = t["trainer_id"].astype("string")
    t["rider"] = t["rider"].fillna("").astype(str).str.strip()
    t["rider_class"] = t["rider"].map(RIDER_CLASS).fillna(4).astype(int)
    t["dow"] = pd.to_datetime(t["train_date"]).dt.dayofweek
    return t.reset_index(drop=True)


def global_z(W: pd.DataFrame) -> np.ndarray:
    """時計の標準化: (種別×所属×強度×馬場) ごとの as-of の平均・標準偏差。件数が少なければ馬場を外す。小さい = 速い。"""
    day = W["train_day"].to_numpy()
    X = W[[f"x_{m}" for m in METRICS]].to_numpy(dtype=float)
    C = np.array([CENTER[m] for m in METRICS])
    ok_row = (W["timed"] & W["intensity"].notna()).to_numpy()
    kc, it, cd = W["kind_c"].astype(str), W["intensity"], W["condc"]
    st = []
    for code in (np.where(ok_row, _codes(kc, it, cd), -1), np.where(ok_row, _codes(kc, it), -1)):
        n, s, ss = _moments(code, day, X, C)
        mean = s / np.maximum(n, 1.0)
        var = np.maximum(ss / np.maximum(n, 1.0) - mean ** 2, 1e-4)
        st.append((n, C + mean, np.sqrt(var)))
    (n0, m0, s0), (n1, m1, s1) = st
    use0 = n0 >= MIN_KEY0
    use1 = ~use0 & (n1 >= MIN_KEY0)
    z0 = (X - m0) / s0
    z1 = (X - m1) / s1
    out = np.where(use0, z0, np.where(use1, z1, np.nan))
    out[~ok_row] = np.nan
    return out


def trainer_habit(Z: np.ndarray, tcode: np.ndarray, bcode: np.ndarray, day: np.ndarray, parent: np.ndarray | None = None):
    """調教師ごとの『いつもの速さ』（標準化した時計 Z の as-of 平均を階層縮約）。
    parent: (n × k) 調教師全体の縮約先（既定 0 = 全体の平均）。戻り: dev（調教師×種別×強度）、sd、調教師全体の件数 nA。"""
    k = Z.shape[1]
    C = np.zeros(k)
    nA, sA, ssA = _moments(tcode, day, Z, C)
    nB, sB, ssB = _moments(bcode, day, Z, C)
    par = np.zeros_like(Z) if parent is None else parent
    devA = (sA + M_TRAINER * par) / (nA + M_TRAINER)
    dev1 = (sB + M_CELL * devA) / (nB + M_CELL)
    varA = (ssA + M_TRAINER * 1.0) / (nA + M_TRAINER)
    var1 = (ssB + M_CELL * varA) / (nB + M_CELL) - dev1 ** 2
    sd = np.sqrt(np.maximum(var1, 0.25))
    bad = nA < MIN_TRAINER
    dev1 = np.where(bad, np.nan, dev1)
    return dev1, np.where(bad, np.nan, sd), nA


def trainer_style_vectors(W: pd.DataFrame, Z: np.ndarray, tcode: np.ndarray):
    """調教師の『時計の出し方』のベクトル（as-of。追い切りの行ごと）: 強め以上の割合・一杯の割合・坂路の割合・t4/t1/加速度の標準化平均。"""
    day = W["train_day"].to_numpy()
    timed = W["timed"].to_numpy() & (tcode >= 0)
    it = W["intensity"].to_numpy(dtype=float)
    hard = np.where(np.isfinite(it) & (it >= 1), 1.0, 0.0)
    full = np.where(np.isfinite(it) & (it >= 2), 1.0, 0.0)
    slope = W["kind"].eq("slope").to_numpy().astype(float)
    z = np.nan_to_num(Z)
    zok = np.isfinite(Z).astype(float)
    t = timed.astype(float)
    V = np.column_stack([t, hard * t, full * t, slope * t, z * t[:, None], zok * t[:, None]])
    S = prior_sums(np.where(timed, tcode, -1), day, V)
    n = S[:, 0]
    vec = np.column_stack([S[:, 1] / np.maximum(n, 1), S[:, 2] / np.maximum(n, 1), S[:, 3] / np.maximum(n, 1),
                           S[:, 4] / np.maximum(S[:, 7], 1), S[:, 5] / np.maximum(S[:, 8], 1), S[:, 6] / np.maximum(S[:, 9], 1)])
    return vec, n


def fit_centroids(vec: np.ndarray, n: np.ndarray, tcode: np.ndarray, day: np.ndarray, fit_before: int):
    """fit_before（日）より前の調教師のベクトル（追い切りが CLUSTER_MIN 以上。調教師ごとに最後の行）から k-means の中心を作る。"""
    from sklearn.cluster import KMeans

    sel = (tcode >= 0) & (day < fit_before) & (n >= CLUSTER_MIN)
    if not sel.any():
        return None
    df = pd.DataFrame({"t": tcode[sel], "d": day[sel], "i": np.flatnonzero(sel)})
    idx = df.loc[df.groupby("t")["d"].idxmax(), "i"].to_numpy()
    if len(idx) < N_CLUSTER * 3:
        return None
    X = vec[idx]
    mu, sd = X.mean(axis=0), np.maximum(X.std(axis=0), 1e-6)
    km = KMeans(n_clusters=N_CLUSTER, n_init=10, random_state=0).fit((X - mu) / sd)
    order = np.argsort(km.cluster_centers_[:, 0])                     # 『強めの割合』の小さい順に 0..K-1（再現できる番号）
    return km.cluster_centers_[order], mu, sd


def assign_cluster(vec: np.ndarray, n: np.ndarray, model) -> np.ndarray:
    if model is None:
        return np.full(len(vec), -1)
    cent, mu, sd = model
    z = (vec - mu) / sd
    d = ((z[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
    return np.where(n >= CLUSTER_MIN, d.argmin(axis=1), -1)


def add_standardized(W: pd.DataFrame, fit_before: str = "2021-01-01") -> pd.DataFrame:
    """W に z0_*（基準で標準化）、dev_*（調教師のいつもの速さ）、ztr_*（調教師比の差）、ztrs_*（同を標準偏差で割った値）、
    クラスタ版 devh_*・zh_*、clu（調教師の時計の出し方のクラスタ）を付ける。"""
    W = W.copy()
    day = W["train_day"].to_numpy()
    Z = global_z(W)
    for j, m in enumerate(METRICS):
        W[f"z0_{m}"] = Z[:, j]
    tcode = np.where(W["trainer_id"].notna().to_numpy(), _codes(W["trainer_id"]), -1)
    ok = (W["timed"] & W["intensity"].notna()).to_numpy()
    bcode = np.where(ok & (tcode >= 0), _codes(W["trainer_id"], W["kind_c"].astype(str), W["intensity"]), -1)
    dev, sd, nA = trainer_habit(Z, np.where(ok, tcode, -1), bcode, day)
    for j, m in enumerate(METRICS):
        W[f"dev_{m}"] = dev[:, j]
        W[f"ztr_{m}"] = Z[:, j] - dev[:, j]
        W[f"ztrs_{m}"] = (Z[:, j] - dev[:, j]) / sd[:, j]
    W["trn_n"] = nA[:, 0]
    # ---- クラスタ
    vec, n = trainer_style_vectors(W, Z, tcode)
    fb = int(np.datetime64(fit_before, "D").astype(np.int64))
    model = fit_centroids(vec, n, tcode, day, fb)
    clu = assign_cluster(vec, n, model)
    W["clu"] = clu
    cc_all = np.where((clu >= 0) & ok, clu, -1)
    nC, sC, _ = _moments(cc_all.astype(np.int64), day, Z, np.zeros(Z.shape[1]))
    par = np.where(nC >= 300, sC / np.maximum(nC, 1.0), 0.0)            # 同じクラスタの調教師全体の『いつもの速さ』（縮約先）
    devh, sdh, _ = trainer_habit(Z, np.where(ok, tcode, -1), bcode, day, parent=par)
    for j, m in enumerate(METRICS):
        W[f"devh_{m}"] = devh[:, j]
        W[f"zh_{m}"] = Z[:, j] - devh[:, j]
    # ---- 追い切り騎手の癖（名前の文字列。調教師比の差に対する as-of 平均を縮約）
    rider = W["rider"].where(W["rider_class"].eq(4))
    rcode = np.where(rider.notna().to_numpy() & ok, _codes(rider), -1)
    nR, sR, _ = _moments(rcode.astype(np.int64), day, W[["ztr_t4", "ztr_t1", "ztr_acc"]].to_numpy(dtype=float), np.zeros(3))
    W["rider_bias_t4"] = np.where(rcode >= 0, sR[:, 0] / (nR[:, 0] + 50.0), np.nan)
    W["rider_bias_t1"] = np.where(rcode >= 0, sR[:, 1] / (nR[:, 1] + 50.0), np.nan)
    W["rider_n"] = np.where(rcode >= 0, nR[:, 0], np.nan)
    return W


# ---------------------------------------------------------------- 短評
def eval_vocab(W: pd.DataFrame, min_count: int = 300, before: str = "2022-01-01") -> set[str]:
    e = W.loc[pd.to_datetime(W["train_date"]) < pd.Timestamp(before), "evaluation"].fillna("").map(lambda s: unicodedata.normalize("NFKC", str(s)).strip())
    vc = e[e != ""].value_counts()
    return set(vc[vc >= min_count].index)


def eval_norm(s: pd.Series, vocab: set[str]) -> pd.Series:
    x = s.fillna("").map(lambda v: unicodedata.normalize("NFKC", str(v)).strip())
    return x.where(x.isin(vocab), np.where(x == "", None, "他")).astype(object)



# ---------------------------------------------------------------- 記者コメント（追い切りに付く談話。併せ馬の結果も書かれる）
POS_WORDS = ("好", "上々", "軽快", "鋭", "力強", "機敏", "素軽", "良化", "上向", "万全", "合格点", "及第点", "抜群", "絶好", "シャープ", "活気", "迫力", "文句なし")
NEG_WORDS = ("平凡", "物足りな", "地味", "ひと息", "一息", "疑問", "平行線", "重い", "鈍", "反応が", "見劣", "欠く", "欠け", "不安", "力んで", "硬", "まだ")
PARTNER = {"sensen": ("先着",), "okure": ("遅れ", "後れ"), "doonyu": ("同入", "併入"), "yusei": ("優勢",), "ressei": ("劣勢",), "tsuiso": ("追走", "先行", "追いかけ"), "awase": ("併せ", "併走", "相手", "挟ま", "食らい")}
FIRST = ("初戦", "叩き", "叩いて", "上積み", "休み明け", "久々")


def text_flags(c: pd.Series) -> pd.DataFrame:
    """コメントの語から作る数値（語の有無・個数）。文章そのものは使わない。"""
    x = c.fillna("").astype(str).map(lambda v: unicodedata.normalize("NFKC", v))
    out = pd.DataFrame(index=x.index)
    out["pos"] = sum(x.str.count(w) for w in POS_WORDS).astype(float)
    out["neg"] = sum(x.str.count(w) for w in NEG_WORDS).astype(float)
    out["net"] = out["pos"] - out["neg"]
    for k, ws in PARTNER.items():
        out[k] = sum(x.str.contains(w, regex=False).astype(float) for w in ws).clip(upper=1.0)
    out["first"] = sum(x.str.contains(w, regex=False).astype(float) for w in FIRST).clip(upper=1.0)
    out["len"] = x.str.len().astype(float)
    return out


def verdict_of(c: pd.Series) -> pd.Series:
    """コメントの最後の一文（『好仕上がり』『平行線』『初戦から』のような定型の見立て）。"""
    def last(v):
        ss = [x for x in unicodedata.normalize("NFKC", str(v)).split("。") if x.strip()]
        return ss[-1].strip() if ss else ""
    return c.fillna("").map(last)


def verdict_vocab(c: pd.DataFrame, min_count: int = 100, before: str = "2022-01-01") -> set[str]:
    v = verdict_of(c.loc[pd.to_datetime(c["race_date"]) < pd.Timestamp(before), "comment"])
    vc = v[v != ""].value_counts()
    return set(vc[vc >= min_count].index)


def comment_slots(comments: pd.DataFrame) -> pd.DataFrame:
    """(horse_id, race_date) ごとの、最終週（1〜6 日前）・1 週前（7〜13 日前）の最新のコメントの語の特徴。"""
    c = comments.copy()
    c["horse_id"] = c["horse_id"].astype(str)
    c["days_before"] = (pd.to_datetime(c["race_date"]) - pd.to_datetime(c["train_date"])).dt.days
    fl = text_flags(c["comment"])
    vv = verdict_of(c["comment"])
    vocab = verdict_vocab(comments)
    fl["verdict"] = vv.where(vv.isin(vocab), np.where(vv == "", None, "他")).astype(object)
    c = pd.concat([c[["horse_id", "race_date", "days_before"]], fl], axis=1)
    out = None
    for tag, lo_hi in (("F", SLOT_F), ("W1", SLOT_W1)):
        s = c[c["days_before"].between(*lo_hi)].sort_values(["horse_id", "race_date", "days_before"], kind="stable")
        g = s.groupby(["horse_id", "race_date"], sort=False).head(1).drop(columns=["days_before"]).copy()
        g.columns = ["horse_id", "race_date"] + [f"{tag}c_{k}" for k in g.columns[2:]]
        out = g if out is None else out.merge(g, on=["horse_id", "race_date"], how="outer")
    return out


def text_groups(d: pd.DataFrame, ag: "L35.Agg", tid: pd.Series) -> dict[str, dict]:
    """d は build_features の表（Fc_* / W1c_* を含む）。コメントがある行だけに値が入る。"""
    g: dict = {}
    for tag in ("Fc", "W1c"):
        for k in ("pos", "neg", "net", "sensen", "okure", "doonyu", "yusei", "ressei", "tsuiso", "awase", "first", "len"):
            if f"{tag}_{k}" in d.columns:
                g[f"{tag.lower()}_{k}"] = d[f"{tag}_{k}"].to_numpy(dtype=float)
    g["has_comment_f"] = d["Fc_net"].notna().to_numpy().astype(float)
    g["has_comment_w1"] = d["W1c_net"].notna().to_numpy().astype(float)
    st: dict = {}
    net = d["Fc_net"]
    nb = pd.Series(np.where(net.isna(), np.nan, np.where(net <= -1, 0.0, np.where(net >= 1, 2.0, 1.0))))
    res = pd.Series(np.where(d["Fc_sensen"] == 1, "先着", np.where(d["Fc_okure"] == 1, "遅れ", np.where(d["Fc_doonyu"] == 1, "同入", np.where(d["Fc_net"].notna(), "なし", None)))), dtype=object)
    edge = pd.Series(np.where(d["Fc_yusei"] == 1, "優勢", np.where(d["Fc_ressei"] == 1, "劣勢", np.where(d["Fc_net"].notna(), "なし", None))), dtype=object)
    _lift(ag, st, "t1_net", nb.astype(object), None)
    _lift(ag, st, "t2_partner", res, None)
    _lift(ag, st, "t3_edge", edge, None)
    _lift(ag, st, "t4_net_trn", _cat(tid, nb), nb.astype(object), m_f=10.0)
    _lift(ag, st, "t5_net_kind", _cat(nb, d["F_kind_c"]), nb.astype(object))
    if "Fc_verdict" in d.columns:
        _lift(ag, st, "t6_verdict", d["Fc_verdict"].astype(object), None)
        _lift(ag, st, "t7_verdict_trn", _cat(tid, d["Fc_verdict"]), d["Fc_verdict"].astype(object), m_f=10.0)
    return {"D7_text_flags": g, "D7_text_lift": st, "D8_text_ctx": text_context(d, tid)}


def text_context(d: pd.DataFrame, tid: pd.Series) -> dict:
    """コメントの文脈: 調教師ごとの『いつもの口調』との差、馬自身の前走・過去のコメントとの差（どちらも日付が厳密に前）。"""
    g: dict = {}
    day = pd.to_datetime(d["date"]).to_numpy("datetime64[D]").astype(np.int64)
    has = d["Fc_net"].notna().to_numpy()
    tcode = np.where(tid.notna().to_numpy(), _codes(tid), -1)
    X = d[["Fc_net", "Fc_pos", "Fc_neg"]].to_numpy(dtype=float)
    n, s_, _ = _moments(np.where(has, tcode, -1).astype(np.int64), day, X, np.zeros(3))
    mean = s_ / np.maximum(n, 1.0)
    ok = has & (n[:, 0] >= 30)
    for j, nm in enumerate(("net", "pos", "neg")):
        g[f"fc_{nm}_vs_trn"] = np.where(ok, X[:, j] - mean[:, j], np.nan)
    g["trn_comment_n"] = np.where(tcode >= 0, np.log1p(n[:, 0]), np.nan)
    # 馬自身の過去のコメント（前走までの平均・前走との差）。ソートした上で『1 行前までの累積』を使う（累積和から自分を引かない）
    dd = d.assign(_i=np.arange(len(d))).sort_values(["horse_id", "date", "race_id"], kind="stable")
    grp = dd["horse_id"]
    v = dd["Fc_net"]
    ok_v = v.notna().astype(float)
    v0 = v.fillna(0.0)
    cs = v0.groupby(grp).cumsum().groupby(grp).shift(1)
    cn = ok_v.groupby(grp).cumsum().groupby(grp).shift(1)
    prev_mean = cs / cn.where(cn > 0)
    last_net = v.groupby(grp).ffill().groupby(grp).shift(1)                      # 直近のコメントのある前走の net
    out = pd.DataFrame({"prev_net_mean": prev_mean, "prev_net_last": last_net, "prev_n": cn, "net_vs_self": v - prev_mean, "net_vs_last": v - last_net}, index=dd.index)
    out = out.loc[dd.sort_values("_i").index]
    for c in out.columns:
        g[f"fc_{c}"] = out[c].to_numpy(dtype=float)
    return g

# ---------------------------------------------------------------- レース単位
def pick_slot(W: pd.DataFrame, lo_hi: tuple[int, int], tag: str, cols: list[str]) -> pd.DataFrame:
    s = W[W["timed"] & W["days_before"].between(*lo_hi)].sort_values(["horse_id", "race_date", "days_before"], kind="stable")
    g = s.groupby(["horse_id", "race_date"], sort=False).head(1)[["horse_id", "race_date"] + cols].copy()
    g.columns = ["horse_id", "race_date"] + [f"{tag}_{c}" for c in cols]
    return g


def between_races(W: pd.DataFrame) -> pd.DataFrame:
    """前のレースの後から今回のレースまでの追い切り（種類を問わず）の本数・間隔・始まりの早さ。"""
    key = ["horse_id", "race_date"]
    s = W.sort_values(key + ["days_before"], kind="stable")
    g = s.groupby(key, sort=False)
    out = g.agg(n_all=("days_before", "size"), first_days=("days_before", "max"), n_distinct=("train_date", "nunique"),
                n_hard_all=("intensity", lambda x: int((x.fillna(0) >= 1).sum()))).reset_index()
    dd = s.drop_duplicates(key + ["days_before"])
    gap = dd.groupby(key, sort=False)["days_before"].apply(lambda x: np.diff(np.sort(x.to_numpy())))
    out = out.merge(pd.DataFrame({"gap_last2": gap.map(lambda a: float(a[0]) if len(a) else np.nan), "gap_mean": gap.map(lambda a: float(a.mean()) if len(a) else np.nan),
                                  "gap_max": gap.map(lambda a: float(a.max()) if len(a) else np.nan)}).reset_index(), on=key, how="left")
    hard3 = s[(s["days_before"] <= 3) & (s["intensity"].fillna(0) >= 1)].groupby(key).size().rename("hard_3d").reset_index()
    out = out.merge(hard3, on=key, how="left")
    out["hard_3d"] = out["hard_3d"].fillna(0)
    return out


def _cat(*cols) -> pd.Series:
    ok = np.ones(len(cols[0]), dtype=bool)
    for c in cols:
        ok &= pd.Series(c).notna().to_numpy()
    s = pd.Series(cols[0]).astype(str).reset_index(drop=True)
    for c in cols[1:]:
        s = s + "#" + pd.Series(c).astype(str).reset_index(drop=True)
    return s.where(ok)


def _bin(x: pd.Series, cuts) -> pd.Series:
    a = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    return pd.Series(np.where(np.isfinite(a), np.digitize(a, cuts).astype(float), np.nan))


def _frame(d: pd.DataFrame, store: dict) -> pd.DataFrame:
    f = pd.DataFrame({k: np.asarray(v, dtype="float32") for k, v in store.items()})
    f.insert(0, "horse_id", d["horse_id"].to_numpy())
    f.insert(0, "race_id", d["race_id"].astype(str).to_numpy())
    return f


def _lift(ag: "L35.Agg", store: dict, name: str, fine: pd.Series, coarse: pd.Series | None, m_f: float = 30.0, m_c: float = 30.0) -> None:
    valid = fine.notna().to_numpy()
    gl = ag.glob(valid & ag.has_res)
    parent = ag.stat(coarse.where(fine.notna()), None, gl, m=m_c) if coarse is not None else None
    st = ag.stat(fine, parent, gl, m=m_f)
    L35.Agg.put(store, name, st, gl, valid)


def build_features(runs: pd.DataFrame, laps: pd.DataFrame, returns: pd.DataFrame | None = None, fit_before: str = "2021-01-01", comments: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """runs: race_id・horse_id・date・trainer_id・finish_position・field_size・venue_code・surface・distance・a_style・grade（結果は成績の as-of 集計にだけ使う）。
    laps: training_laps 相当。returns: runner_returns（runs と同じ並び。使わなくてもよい）。戻り値: 群名 → race_id・horse_id + 特徴量。"""
    runs = runs.reset_index(drop=True)
    W = add_standardized(prepare_workouts(laps, runs), fit_before)
    vocab = eval_vocab(W)
    W["eval_n"] = eval_norm(W["evaluation"], vocab)
    W["rank_l"] = W["rank"].where(W["rank"].isin(["A", "B", "C", "D"]))
    W["pos_n"] = W["position"]
    zc = [f"{p}_{m}" for m in METRICS for p in ("z0", "dev", "ztr", "ztrs")]
    extra = ["cls", "kind_c", "intensity", "days_before", "rider_class", "rider_bias_t4", "rider_bias_t1", "rider_n", "clu", "dow", "eval_n", "rank_l", "pos_n", "trn_n",
             "devh_t4", "devh_t1", "devh_acc", "zh_t4", "zh_t1", "zh_acc"]
    cols = zc + extra
    F, W1 = pick_slot(W, SLOT_F, "F", cols), pick_slot(W, SLOT_W1, "W1", cols)
    tm = W[W["timed"] & (W["days_before"] <= 14)]
    a14 = tm.groupby(["horse_id", "race_date"]).agg(m_ztr_t4=("ztr_t4", "mean"), min_ztr_t1=("ztr_t1", "min"), m_ztr_acc=("ztr_acc", "mean"),
                                                  n_fast14=("ztr_t4", lambda x: float((x < -0.5).sum())), n_slow14=("ztr_t4", lambda x: float((x > 0.5).sum()))).reset_index()
    btw = between_races(W)
    d = runs[["race_id", "horse_id", "date", "trainer_id", "finish_position", "field_size", "venue_code", "surface", "distance", "a_style", "grade"]].copy()
    d["race_date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    d["horse_id"] = d["horse_id"].astype(str)
    parts = [F, W1, a14, btw] + ([comment_slots(comments)] if comments is not None else [])
    for part in parts:
        d = d.merge(part, on=["horse_id", "race_date"], how="left")
    assert len(d) == len(runs)
    ag = L35.Agg(d, returns)
    tid = d["trainer_id"].astype(str)
    cond_a = d["surface"].astype(str) + "|" + L35.dband(d["distance"])
    cond_b = d["venue_code"].astype(str) + "|" + d["surface"].astype(str) + "|" + pd.to_numeric(d["distance"], errors="coerce").fillna(-1).astype(int).astype(str)
    surf = d["surface"].astype(str)
    out: dict[str, dict] = {}

    # ---- TRZ: 調教師ごとの標準化
    g = {}
    for tag in ("F", "W1"):
        for c in zc:
            g[f"{tag.lower()}_{c}"] = d[f"{tag}_{c}"].to_numpy(dtype=float)
    for c in ("m_ztr_t4", "min_ztr_t1", "m_ztr_acc", "n_fast14", "n_slow14"):
        g[c] = d[c].to_numpy(dtype=float)
    g["f_trn_logn"] = np.log1p(d["F_trn_n"].to_numpy(dtype=float))
    out["TRZ"] = g
    # ---- TRZ_SELF: 馬自身の過去・前走との比較（標準化した時計）
    g = {}
    dd = d.sort_values(["horse_id", "date", "race_id"], kind="stable")
    for m in METRICS:
        v = dd[f"F_ztr_{m}"]
        ok = v.notna().astype(float)
        v0 = v.fillna(0.0)
        grp = dd["horse_id"]
        cs = v0.groupby(grp).cumsum().groupby(grp).shift(1)
        cn = ok.groupby(grp).cumsum().groupby(grp).shift(1)
        # cumsum はその馬の過去の行だけを足していく。shift(1) で自分を含めない（引き算はしない）
        prev_mean = cs / cn.where(cn > 0)
        g[f"f_ztr_{m}_vs_self"] = (v - prev_mean).reindex(d.index).to_numpy(dtype=float)
        g[f"f_ztr_{m}_vs_prev"] = (v - v.groupby(grp).shift(1)).reindex(d.index).to_numpy(dtype=float)
    out["TRZ_SELF"] = g
    # ---- CLZ: 調教師の『時計の出し方』クラスタ
    g = {"f_clu": d["F_clu"].to_numpy(dtype=float), "w1_clu": d["W1_clu"].to_numpy(dtype=float)}
    for tag in ("F", "W1"):
        for m in METRICS:
            g[f"{tag.lower()}_zh_{m}"] = d[f"{tag}_zh_{m}"].to_numpy(dtype=float)
            if tag == "F":
                g[f"f_devh_{m}"] = d[f"F_devh_{m}"].to_numpy(dtype=float)
    out["CLZ"] = g

    # ---- 交互作用（成績の as-of 集計。レース条件 × 追い切りの型）
    zb4 = _bin(d["F_ztr_t4"], Z_CUTS)
    zb1 = _bin(d["F_ztr_t1"], Z_CUTS)
    zba = _bin(d["F_ztr_acc"], Z_CUTS)
    hab = _bin(d["F_dev_t4"], HABIT_CUTS)
    cls_f = d["F_cls"].where(d["F_cls"].isin(L35.CLASSES[:-1]))
    inten = d["F_intensity"]
    style = _bin(d["a_style"], STYLE_CUTS)                              # 脚質（0 = 逃げ・先行寄り … 1 = 差し・追込寄り）を 4 区分
    grade = d["grade"].astype(str).replace("", "上").where(d["grade"].notna())
    clu = d["F_clu"].where(d["F_clu"] >= 0)
    gi: dict[str, dict] = {k: {} for k in ("I1_cls_cond", "I2_zt4_cond_int", "I3_zt4_style_surf", "I4_cls_grade", "I5_zt1_courseexact", "I6_clu_zt4_cond",
                                          "I7_kind_int_days_cond", "I8_trainer_zt4", "I9_zacc_style_dist", "I10_habit_cond", "I11_clu_course")}
    _lift(ag, gi["I1_cls_cond"], "i1", _cat(cls_f, cond_a), cls_f.astype(object))
    _lift(ag, gi["I2_zt4_cond_int"], "i2", _cat(zb4, cond_a, inten), zb4.astype(object))
    _lift(ag, gi["I3_zt4_style_surf"], "i3", _cat(zb4, style, surf), zb4.astype(object))
    _lift(ag, gi["I4_cls_grade"], "i4", _cat(cls_f, grade), cls_f.astype(object))
    _lift(ag, gi["I5_zt1_courseexact"], "i5", _cat(zb1, cond_b), _cat(zb1, cond_a))
    _lift(ag, gi["I6_clu_zt4_cond"], "i6", _cat(clu, zb4, cond_a), _cat(clu, zb4))
    daysb = pd.cut(pd.to_numeric(d["F_days_before"], errors="coerce"), [0, 2, 4, 6], labels=False).astype(float)
    _lift(ag, gi["I7_kind_int_days_cond"], "i7", _cat(d["F_kind_c"], inten, daysb, cond_a), _cat(d["F_kind_c"], inten, daysb))
    _lift(ag, gi["I8_trainer_zt4"], "i8", _cat(tid, zb4), zb4.astype(object), m_f=10.0)
    _lift(ag, gi["I9_zacc_style_dist"], "i9", _cat(zba, style, L35.dband(d["distance"])), zba.astype(object))
    _lift(ag, gi["I10_habit_cond"], "i10", _cat(hab, cond_a), hab.astype(object))
    _lift(ag, gi["I11_clu_course"], "i11", _cat(clu, cond_b), _cat(clu, cond_a))
    for k, v in gi.items():
        out[k] = v
    if comments is not None:
        out.update(text_groups(d, ag, tid))

    # ---- 本数・間隔・前走比
    g = {c: d[c].to_numpy(dtype=float) for c in ("n_all", "first_days", "n_distinct", "n_hard_all", "gap_last2", "gap_mean", "gap_max", "hard_3d")}
    out["D1_between"] = g
    g = {}
    dd = d.sort_values(["horse_id", "date", "race_id"], kind="stable")
    grp = dd["horse_id"]
    for c, nm in (("F_ztr_t4", "d_ztr_t4"), ("F_ztr_t1", "d_ztr_t1"), ("F_intensity", "d_int"), ("n_all", "d_nall"), ("F_days_before", "d_fdays"), ("first_days", "d_first"), ("F_z0_t4", "d_z0_t4")):
        v = dd[c].astype(float)
        g[nm] = (v - v.groupby(grp).shift(1)).reindex(d.index).to_numpy(dtype=float)
    kc = dd["F_kind_c"].astype(object)
    prev_kc = kc.groupby(grp).shift(1)
    g["same_kind_prev"] = pd.Series(np.where(kc.notna() & prev_kc.notna(), (kc == prev_kc).astype(float), np.nan), index=dd.index).reindex(d.index).to_numpy()
    out["D2_change"] = g

    # ---- 追い切り騎手（名前の文字列。ID なし）
    g = {"f_rider_class": d["F_rider_class"].to_numpy(dtype=float), "w1_rider_class": d["W1_rider_class"].to_numpy(dtype=float),
         "f_rider_bias_t4": d["F_rider_bias_t4"].to_numpy(dtype=float), "f_rider_bias_t1": d["F_rider_bias_t1"].to_numpy(dtype=float),
         "f_rider_logn": np.log1p(d["F_rider_n"].to_numpy(dtype=float)), "w1_rider_bias_t4": d["W1_rider_bias_t4"].to_numpy(dtype=float)}
    rc = d["F_rider_class"].where(d["F_rider_class"].notna())
    _lift(ag, g, "d3_trn_rc", _cat(tid, rc), rc.astype(object))
    out["D3_rider"] = g

    # ---- 短評・追い切りランク
    g = {}
    _lift(ag, g, "d4_eval", d["F_eval_n"].astype(object), None)
    _lift(ag, g, "d4_eval_trn", _cat(tid, d["F_eval_n"]), d["F_eval_n"].astype(object), m_f=10.0)
    _lift(ag, g, "d4_rank_kind", _cat(d["F_rank_l"], d["F_kind_c"]), d["F_rank_l"].astype(object))
    _lift(ag, g, "d4_eval_w1", d["W1_eval_n"].astype(object), None)
    out["D4_eval"] = g

    # ---- 追い切りの曜日 × 調教師、走路位置 × 種別
    g = {"f_dow": d["F_dow"].to_numpy(dtype=float)}
    _lift(ag, g, "d5_trn_dow", _cat(tid, d["F_dow"]), d["F_dow"].astype(object), m_f=10.0)
    _lift(ag, g, "d6_pos_kind", _cat(d["F_kind_c"], d["F_pos_n"]), d["F_kind_c"].astype(object))
    out["D5_dow_pos"] = g
    res = {k: _frame(d, v) for k, v in out.items()}
    keys = pd.DataFrame({"race_id": d["race_id"].astype(str).to_numpy(), "horse_id": d["horse_id"].to_numpy(), "cond_a": cond_a.to_numpy(), "zb4": zb4.to_numpy(), "zb1": zb1.to_numpy(),
                         "zba": zba.to_numpy(), "habit": hab.to_numpy(), "clu": d["F_clu"].to_numpy(), "style_bin": style.to_numpy(), "f_int": inten.to_numpy(),
                         "f_eval": d["F_eval_n"].to_numpy(dtype=object), "f_rank": d["F_rank_l"].to_numpy(dtype=object), "f_rider_class": d["F_rider_class"].to_numpy()})
    res["KEYS"] = keys
    return res
