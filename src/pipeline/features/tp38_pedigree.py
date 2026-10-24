"""tp-38: 血統の『位置・存在割合・クロス・類似種牡馬』の特徴量（すべて as-of。キーは ID、市場オッズ・人気は使わない）。

祖先 a が馬 h の血統表のどの位置にいるか（gen=世代、pos=位置。gen1: 0=父 1=母 / gen2: 0=父父 1=父母 2=母父 3=母母 / 一般に pos を gen ビットで書いたとき
先頭ビットが 0=父側・1=母側）を使い、次の統計量を作る。成績の指標は y = res_score（レース内の着順を [-0.5, 0.5] にしたもの。レース内の平均は 0）。

  F1 cell     祖先 × 位置の区分 × 条件の産駒成績を、その日より前だけで集め、階層縮約（条件の全体 → 祖先の全位置 → 祖先の特定位置）してから、
              馬ごとに『位置の重み 2^-gen』で合算する。馬自身の過去走は引く（祖先の成績に自分が混ざらない）。
  cross       近交係数（Wright 型の経路和）、クロスの数・強さ、クロス祖先の『クロスしたときの上乗せ』。
  emb / clu   血統表の祖先ベクトル（TF-IDF）を SVD で圧縮した座標、k-means の血統クラスタと、クラスタ × 条件の成績。
  sim         種牡馬自身の血統構成のコサイン類似度で近い種牡馬を求め、産駒の少ない種牡馬の成績を近い種牡馬のカーネル加重で縮約する。
リッジ（F2）は tp38_ridge.py。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

GEN_MAX = 5
# 位置の区分（pcls）: gen1 は父/母（0,1）、gen2 は 4 通り（2..5）、gen3 以降は (gen, 側, 直系) で 4 通りずつ（6..17）。
# 直系 = 父父父…（pos==0）か 母母母…（pos==2^gen-1）。
PC_GEN = np.array([1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5])
PC_SIDE = np.array([0, 1, 0, 0, 1, 1] + [0, 0, 1, 1] * 3)
PC_TM = np.array([1, 0, 1, 0, 0, 0] + [0, 1, 0, 0] * 3)       # 父系の直系（tail-male）
PC_TF = np.array([0, 1, 0, 0, 0, 1] + [0, 0, 0, 1] * 3)       # 母系の直系（tail-female）
N_PC = len(PC_GEN)
ROLE_PCLS = {"sire": 0, "ssire": 2, "dsire": 4}               # 父 / 父父 / 母父


def pos_class(gen, pos) -> np.ndarray:
    gen, pos = np.asarray(gen, dtype=np.int64), np.asarray(pos, dtype=np.int64)
    side = pos >> (gen - 1)
    tail = ((pos == 0) | (pos == (1 << gen) - 1)).astype(np.int64)
    return np.where(gen == 1, pos, np.where(gen == 2, 2 + pos, 6 + (gen - 3) * 4 + side * 2 + tail)).astype(np.int16)


def pos_name(gen: int, pos: int) -> str:
    return "".join("DS"[1 - ((pos >> (gen - 1 - i)) & 1)] for i in range(gen))


# ───────────────────────── 血統表の長い表
def prep_ancestors(A: pd.DataFrame, horse_index: pd.Index) -> tuple[pd.DataFrame, np.ndarray]:
    """A: horse_id, gen, pos, anc_id → (h(馬のコード), a(祖先のコード), gen, pos, pcls, side, w, male)。"""
    A = A[A["gen"] <= GEN_MAX].copy()
    A["h"] = horse_index.get_indexer(A["horse_id"].astype(str)).astype(np.int32)
    A = A[A["h"] >= 0]
    A["gen"], A["pos"] = A["gen"].astype(np.int64), A["pos"].astype(np.int64)
    codes, uniq = pd.factorize(A["anc_id"].astype(str))
    A["a"] = codes.astype(np.int32)
    A["pcls"] = pos_class(A["gen"], A["pos"])
    A["side"] = np.right_shift(A["pos"].to_numpy(), (A["gen"] - 1).to_numpy()).astype(np.int8)
    A["w"] = 0.5 ** A["gen"].astype(float)
    A["male"] = ((A["pos"] & 1) == 0)
    return A[["h", "a", "gen", "pos", "pcls", "side", "w", "male"]].reset_index(drop=True), np.asarray(uniq)


# ───────────────────────── as-of の部品
def daily_cum(L: pl.DataFrame, by: list[str], vcol: str = "y", wcol: str | None = None) -> pl.DataFrame:
    """by ごと・日ごとの合計を累積した表（by・date・S・N）。vcol が null の行は入れない。"""
    dv = L.filter(pl.col(vcol).is_not_null())
    if wcol is None:
        agg = [pl.col(vcol).sum().alias("s"), pl.len().cast(pl.Float64).alias("n")]
    else:
        agg = [(pl.col(vcol) * pl.col(wcol)).sum().alias("s"), pl.col(wcol).sum().alias("n")]
    return (dv.group_by(by + ["date"]).agg(agg).sort("date")
            .with_columns(pl.col("s").cum_sum().over(by).alias("S"), pl.col("n").cum_sum().over(by).alias("N")).select(by + ["date", "S", "N"]))


def lookup_prior(L: pl.DataFrame, daily: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """L の各行に、daily（by・date の累積）から『同じ by で date が厳密に前』の最新の S・N を付ける（_r 順）。同じ日は入らない。"""
    j = L.select(["_r", "date"] + by).sort("date").join_asof(daily, on="date", by=by, strategy="backward", allow_exact_matches=False).sort("_r")
    return j.select(["_r", pl.col("S").fill_null(0.0), pl.col("N").fill_null(0.0)])


def prior_cells(L: pl.DataFrame, by: list[str], vcol: str = "y", wcol: str | None = None) -> pl.DataFrame:
    return lookup_prior(L, daily_cum(L, by, vcol, wcol), by)


def res_y(B: pd.DataFrame) -> np.ndarray:
    fp = pd.to_numeric(B["finish_position"], errors="coerce").to_numpy(dtype=float)
    fp = np.where(fp > 0, fp, np.nan)
    n = np.maximum(pd.to_numeric(B["field_size"], errors="coerce").to_numpy(dtype=float) - 1.0, 1.0)
    return 0.5 - (fp - 1.0) / n


def win_y(B: pd.DataFrame) -> np.ndarray:
    fp = pd.to_numeric(B["finish_position"], errors="coerce").to_numpy(dtype=float)
    return np.where(fp > 0, (fp == 1).astype(float), np.nan)


def dist_band(B: pd.DataFrame) -> pd.Series:
    d = pd.to_numeric(B["distance"], errors="coerce")
    return pd.cut(d, [0, 1300, 1500, 1900, 2400, 99999], labels=False).astype("float").fillna(-1).astype(int).astype(str)


def cond_keys(B: pd.DataFrame) -> dict[str, pd.Series]:
    surf = B["surface"].astype(str)
    going = pd.Series(np.where(B["track_condition"].astype(str).isin(["良"]), "good", "heavy"), index=B.index)
    return {"all": pd.Series("x", index=B.index), "sf": surf, "sd": surf + "|" + dist_band(B), "gd": surf + "|" + going, "vn": surf + "|" + B["venue_code"].astype(str)}


class Runs:
    """走（B の各行）の整数表現。horse_index は祖先表と同じ馬のコード（祖先表に無い馬は -1）。"""

    def __init__(self, B: pd.DataFrame, horse_index: pd.Index, y: np.ndarray | None = None):
        self.B = B
        self.n = len(B)
        self.date = pd.to_datetime(B["date"]).to_numpy().astype("datetime64[D]")
        self.h = horse_index.get_indexer(B["horse_id"].astype(str)).astype(np.int32)
        self.y = res_y(B) if y is None else y
        self.cond_codes: dict[str, np.ndarray] = {k: pd.factorize(v)[0].astype(np.int32) for k, v in cond_keys(B).items()}

    def frame(self, cond: str, y: np.ndarray | None = None) -> pl.DataFrame:
        yy = self.y if y is None else y
        return pl.DataFrame({"_r": np.arange(self.n, dtype=np.int64), "date": self.date, "h": self.h, "c": self.cond_codes[cond], "y": yy}, nan_to_null=True)


def own_and_mu(R: Runs, cond: str, y: np.ndarray | None = None, k_mu: float = 50.0) -> pd.DataFrame:
    """走ごとに、(1) その馬自身の過去走の合計・件数（同じ条件）、(2) 条件の全体平均 μ（その日より前、全体平均へ縮約）。"""
    F = R.frame(cond, y)
    own = prior_cells(F, ["h", "c"])
    cm = prior_cells(F, ["c"])
    gm = prior_cells(F.with_columns(pl.lit(0).alias("g")), ["g"])
    o = pd.DataFrame({"S_own": own["S"].to_numpy(), "N_own": own["N"].to_numpy(), "S_c": cm["S"].to_numpy(), "N_c": cm["N"].to_numpy(), "S_g": gm["S"].to_numpy(), "N_g": gm["N"].to_numpy()})
    mu0 = np.where(o["N_g"] > 0, o["S_g"] / np.maximum(o["N_g"], 1.0), 0.0)
    o["mu"] = (o["S_c"] + k_mu * mu0) / (o["N_c"] + k_mu)
    o["mu0"] = mu0
    return o


def _long(F: pl.DataFrame, tbl: pl.DataFrame) -> pl.DataFrame:
    """走 F（_r が走の番号）と馬 × 祖先の表 tbl を h で結び、結合後の行番号を _r、走の番号を _i にする。"""
    return F.rename({"_r": "_i"}).join(tbl, on="h", how="inner").with_row_index("_r").with_columns(pl.col("_r").cast(pl.Int64))


def _own_frame(O: pd.DataFrame) -> pl.DataFrame:
    return pl.DataFrame({"_i": np.arange(len(O), dtype=np.int64), "So": O["S_own"].to_numpy(), "No": O["N_own"].to_numpy(), "mu": O["mu"].to_numpy()})


# ───────────────────────── F1: 祖先 × 位置 × 条件
def cell_features(R: Runs, anc: pd.DataFrame, cond: str, tag: str, y: np.ndarray | None = None, k1: float = 30.0, k2: float = 30.0) -> pd.DataFrame:
    """祖先 a の産駒の条件別成績を、(a, 条件) → (a, 位置の区分, 条件) の階層縮約で作り、馬ごとに位置の重み 2^-gen で合算する。馬自身の過去走は引く。"""
    O = own_and_mu(R, cond, y)
    own = _own_frame(O)
    F = R.frame(cond, y).filter(pl.col("h") >= 0)
    ap = pl.from_pandas(anc[["h", "a", "pcls", "w"]]).group_by(["h", "a", "pcls"]).agg(pl.col("w").sum())          # 同じ区分に同じ祖先が複数回出るときは w を合算
    aa = pl.from_pandas(anc[["h", "a"]]).unique()
    La = _long(F, aa)
    Ca = prior_cells(La, ["a", "c"])
    La = (La.with_columns(Ca["S"].alias("S"), Ca["N"].alias("N")).join(own, on="_i", how="left")
          .with_columns(((pl.col("S") - pl.col("So") + k1 * pl.col("mu")) / (pl.col("N") - pl.col("No") + k1)).alias("m_a")).select(["_i", "a", "m_a"]))
    Lp = _long(F, ap)
    Cp = prior_cells(Lp, ["a", "pcls", "c"])
    Lp = (Lp.with_columns(Cp["S"].alias("S"), Cp["N"].alias("N")).select(["_i", "a", "pcls", "w", "S", "N"]).join(own, on="_i", how="left").join(La, on=["_i", "a"], how="left")
          .with_columns((pl.col("N") - pl.col("No")).alias("nn"), ((pl.col("S") - pl.col("So") + k2 * pl.col("m_a")) / (pl.col("N") - pl.col("No") + k2)).alias("m_ap"))
          .with_columns((pl.col("m_ap") - pl.col("mu")).alias("d"), (pl.col("m_ap") - pl.col("m_a")).alias("dp"), (pl.col("nn") / (pl.col("nn") + k2)).alias("r")))
    return _aggregate(Lp, R.n, tag)


def _aggregate(Lp: pl.DataFrame, n_runs: int, tag: str) -> pd.DataFrame:
    pcg = pl.DataFrame({"pcls": np.arange(N_PC, dtype=np.int16), "gen": PC_GEN.astype(np.int8), "side": PC_SIDE.astype(np.int8), "tm": PC_TM.astype(np.int8), "tf": PC_TF.astype(np.int8)})
    Lp = Lp.with_columns(pl.col("pcls").cast(pl.Int16)).join(pcg, on="pcls", how="left")
    w, d, r = pl.col("w"), pl.col("d"), pl.col("r")
    exprs = [((w * d).sum() / w.sum()).alias(f"{tag}_w"), ((w * r * d).sum() / w.sum()).alias(f"{tag}_r"), ((w * pl.col("dp")).sum() / w.sum()).alias(f"{tag}_pos")]
    for g in range(1, GEN_MAX + 1):
        exprs.append(d.filter(pl.col("gen") == g).mean().alias(f"{tag}_g{g}"))
    for s in (0, 1):
        exprs.append(((w * d).filter(pl.col("side") == s).sum() / w.filter(pl.col("side") == s).sum()).alias(f"{tag}_side{s}"))
    exprs += [d.filter(pl.col("tm") == 1).mean().alias(f"{tag}_tm"), d.filter(pl.col("tf") == 1).mean().alias(f"{tag}_tf"),
              d.filter((pl.col("gen") <= 3) & (pl.col("nn") >= 10)).max().alias(f"{tag}_max3"), d.filter((pl.col("gen") <= 3) & (pl.col("nn") >= 10)).min().alias(f"{tag}_min3"),
              pl.col("nn").filter(pl.col("gen") <= 2).mean().alias(f"{tag}_n12")]
    gp = Lp.group_by("_i").agg(exprs).to_pandas().set_index("_i")
    return pd.DataFrame(index=np.arange(n_runs)).join(gp).astype("float32")


# ───────────────────────── クロス（近交）
def cross_static(anc: pd.DataFrame, n_horses: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """馬ごとの近交の静的な量と、(馬, 祖先) ごとのクロスの強さ・種類の表を返す。
    近交係数は Wright の経路和: 父側 g1 世代・母側 g2 世代に同じ祖先が出るとき 0.5^(g1+g2-1) を全ての組で足す（祖先自身の近交は無視）。"""
    P = pl.from_pandas(anc[["h", "a", "gen", "side", "w"]])
    s0, s1 = P.filter(pl.col("side") == 0), P.filter(pl.col("side") == 1)
    pr = s0.join(s1, on=["h", "a"], suffix="_2").with_columns((0.5 ** (pl.col("gen") + pl.col("gen_2") - 1).cast(pl.Float64)).alias("c"), (pl.col("gen") + pl.col("gen_2")).alias("gs"),
                                                              pl.max_horizontal("gen", "gen_2").alias("gm"))
    per_a = pr.group_by(["h", "a"]).agg(pl.col("c").sum().alias("sa"), pl.col("gs").min().alias("gs"))
    per_h = per_a.group_by("h").agg(pl.col("sa").sum().alias("x_F"), pl.len().alias("x_n"), pl.col("sa").max().alias("x_max"), pl.col("gs").min().alias("x_gsmin"))
    gm = pr.group_by("h").agg(pl.col("gm").min().alias("x_gmin"))
    # 同じ側の中で繰り返し出る祖先（片側集中）: 側ごとに、2 回以上出る祖先の (Σw − max w)
    dup = (P.group_by(["h", "a", "side"]).agg(pl.col("w").sum().alias("sw"), pl.col("w").max().alias("mw"), pl.len().alias("k")).filter(pl.col("k") >= 2)
           .with_columns((pl.col("sw") - pl.col("mw")).alias("dm")).group_by(["h", "side"]).agg(pl.col("dm").sum()))
    d0 = dup.filter(pl.col("side") == 0).select(["h", pl.col("dm").alias("x_dup_sire")])
    d1 = dup.filter(pl.col("side") == 1).select(["h", pl.col("dm").alias("x_dup_dam")])
    H = pl.DataFrame({"h": np.arange(n_horses, dtype=np.int32)}).join(per_h, on="h", how="left").join(gm, on="h", how="left").join(d0, on="h", how="left").join(d1, on="h", how="left")
    H = H.with_columns(pl.col("x_F").fill_null(0.0), pl.col("x_n").fill_null(0), pl.col("x_max").fill_null(0.0), pl.col("x_dup_sire").fill_null(0.0), pl.col("x_dup_dam").fill_null(0.0))
    # 片側の重複だけの祖先も含めた (馬, 祖先) の種類: 2 = 両側にクロス, 1 = 片側で重複, 0 = 1 回のみ
    kinds = (P.group_by(["h", "a"]).agg(pl.len().alias("k")).join(per_a.select(["h", "a", "sa"]), on=["h", "a"], how="left")
             .with_columns(pl.when(pl.col("sa").is_not_null()).then(2).when(pl.col("k") >= 2).then(1).otherwise(0).alias("kind"), pl.col("sa").fill_null(0.0)))
    return H.to_pandas().set_index("h"), kinds.to_pandas()


def cross_asof(R: Runs, anc: pd.DataFrame, kinds: pd.DataFrame, cond: str = "all", k: float = 30.0, tag: str = "xe") -> pd.DataFrame:
    """クロスした祖先ごとの『クロスしたときの上乗せ』: (祖先, 種類) の産駒成績 − (祖先, 全種類) の産駒成績（どちらも縮約）。馬自身の過去走は引く。
    もう 1 つ、クロスの強さの帯（Σ 経路和の対数帯）ごとの全体成績 − 全体平均（近交の強さが成績に効くか）。"""
    O = own_and_mu(R, cond)
    own = _own_frame(O)
    F = R.frame(cond).filter(pl.col("h") >= 0)
    kd = pl.from_pandas(kinds[["h", "a", "kind", "sa"]])
    Lk = _long(F, kd.select(["h", "a", "kind", "sa"]))
    Ca = prior_cells(Lk, ["a", "c"])
    Cx = prior_cells(Lk, ["a", "kind", "c"])
    Lk = (Lk.with_columns(Ca["S"].alias("Sa"), Ca["N"].alias("Na"), Cx["S"].alias("Sx"), Cx["N"].alias("Nx")).join(own, on="_i", how="left")
          .with_columns(((pl.col("Sa") - pl.col("So") + k * pl.col("mu")) / (pl.col("Na") - pl.col("No") + k)).alias("m_a"))
          .with_columns(((pl.col("Sx") - pl.col("So") + k * pl.col("m_a")) / (pl.col("Nx") - pl.col("No") + k)).alias("m_x"))
          .with_columns((pl.col("m_x") - pl.col("m_a")).alias("dx"), (pl.col("Nx") - pl.col("No")).alias("nx")))
    cr = Lk.filter(pl.col("kind") == 2)
    g2 = cr.group_by("_i").agg(((pl.col("sa") * pl.col("dx")).sum() / pl.col("sa").sum()).alias("xe_cross"), pl.col("dx").max().alias("xe_cross_max"), pl.col("nx").mean().alias("xe_cross_n"))
    g1 = Lk.filter(pl.col("kind") == 1).group_by("_i").agg(pl.col("dx").mean().alias("xe_dup"))
    # 強さの帯（祖先に依らない全体）: 経路和の対数帯
    band = (-np.log2(np.maximum(kinds["sa"].to_numpy(), 1e-9))).round().clip(0, 12).astype(np.int8)
    kb = pl.from_pandas(kinds.assign(band=np.where(kinds["kind"] == 2, band, -1))[["h", "a", "band", "sa"]])
    Lb = _long(F, kb.filter(pl.col("band") >= 0).group_by(["h", "band"]).agg(pl.col("sa").sum().alias("sa")))
    Cb = prior_cells(Lb, ["band", "c"])
    Lb = (Lb.with_columns(Cb["S"].alias("Sb"), Cb["N"].alias("Nb")).join(own, on="_i", how="left")
          .with_columns((((pl.col("Sb") - pl.col("So") + k * pl.col("mu")) / (pl.col("Nb") - pl.col("No") + k)) - pl.col("mu")).alias("db")))
    gb = Lb.group_by("_i").agg(((pl.col("sa") * pl.col("db")).sum() / pl.col("sa").sum()).alias("xe_band"))
    # クロス祖先の『その条件での適性』: クロスの強さで重み付けした (祖先, 条件) の産駒成績 − 条件平均
    ac = cr.group_by("_i").agg(((pl.col("sa") * (pl.col("m_a") - pl.col("mu"))).sum() / pl.col("sa").sum()).alias("xe_apt"))
    out = pd.DataFrame(index=np.arange(R.n))
    for g in (g2, g1, gb, ac):
        out = out.join(g.to_pandas().set_index("_i"))
    out.columns = [c.replace("xe_", f"{tag}_") for c in out.columns]
    return out.astype("float32")


# ───────────────────────── 血統空間（埋め込み・クラスタ）
def horse_vectors(anc: pd.DataFrame, n_horses: int, n_anc: int):
    """馬 × 祖先の TF-IDF（tf = Σ 2^-gen）の疎行列（行を L2 正規化）。"""
    import scipy.sparse as sp

    tf = anc.groupby(["h", "a"], sort=False)["w"].sum().reset_index()
    X = sp.csr_matrix((tf["w"].to_numpy(), (tf["h"].to_numpy(), tf["a"].to_numpy())), shape=(n_horses, n_anc))
    df_ = np.asarray((X > 0).sum(axis=0)).ravel()
    idf = np.log((1.0 + n_horses) / (1.0 + df_)) + 1e-6
    X = X.multiply(idf).tocsr()
    nrm = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    X = sp.diags(1.0 / np.maximum(nrm, 1e-12)) @ X
    return X.tocsr(), idf


def embed_and_cluster(X, n_dim: int = 32, ks: tuple[int, ...] = (64, 256), seed: int = 0):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import TruncatedSVD

    E = TruncatedSVD(n_components=n_dim, random_state=seed).fit_transform(X)
    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    cl = {k: MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=4096).fit_predict(En[:, :16]).astype(np.int32) for k in ks}
    return E.astype(np.float32), cl


def cluster_asof(R: Runs, cl: np.ndarray, cond: str, tag: str, k: float = 30.0) -> pd.DataFrame:
    """血統クラスタ × 条件の産駒成績（その日より前、馬自身は引く、条件の平均へ縮約）。"""
    O = own_and_mu(R, cond)
    F = R.frame(cond)
    cc = np.where(R.h >= 0, cl[np.maximum(R.h, 0)], -1).astype(np.int32)
    F = F.with_columns(pl.Series("cl", cc)).filter(pl.col("cl") >= 0)
    Cc = prior_cells(F, ["cl", "c"])
    S, N = Cc["S"].to_numpy(), Cc["N"].to_numpy()
    idx = F["_r"].to_numpy()
    so, no, mu = O["S_own"].to_numpy()[idx], O["N_own"].to_numpy()[idx], O["mu"].to_numpy()[idx]
    nn = N - no
    m = (S - so + k * mu) / (nn + k)
    out = pd.DataFrame(index=np.arange(R.n), columns=[f"{tag}_m", f"{tag}_n"], dtype="float32")
    out.loc[idx, f"{tag}_m"] = (m - mu).astype("float32")
    out.loc[idx, f"{tag}_n"] = nn.astype("float32")
    return out


# ───────────────────────── 類似種牡馬
def stallion_vectors(anc: pd.DataFrame, n_anc: int):
    """男系の祖先（pos が偶数）ごとに、その馬自身の血統表（相対世代 j=1..4、重み 2^-j）を、その馬が最も上の世代に出る馬の血統表から復元する。
    戻り値: (stallion の祖先コード, 疎行列(行=stallion, 列=祖先, TF-IDF を L2 正規化), 持てた世代数)"""
    import scipy.sparse as sp

    nodes = anc[anc["male"]].sort_values(["gen", "h"], kind="stable").drop_duplicates("a")[["a", "h", "gen", "pos"]].rename(columns={"gen": "g0", "pos": "p0"})
    nodes = nodes[nodes["g0"] < GEN_MAX]
    m = nodes.merge(anc[["h", "a", "gen", "pos"]].rename(columns={"a": "a2", "gen": "g2", "pos": "p2"}), on="h", how="inner")
    m = m[(m["g2"] > m["g0"]) & (np.right_shift(m["p2"].to_numpy(), (m["g2"] - m["g0"]).clip(lower=0).to_numpy()) == m["p0"].to_numpy())]
    m["j"] = m["g2"] - m["g0"]
    m["w"] = 0.5 ** m["j"].astype(float)
    tf = m.groupby(["a", "a2"], sort=False)["w"].sum().reset_index()
    sa = np.sort(tf["a"].unique())
    row = pd.Series(np.arange(len(sa)), index=sa)
    r = row.loc[tf["a"]].to_numpy()
    X = sp.csr_matrix((tf["w"].to_numpy(), (r, tf["a2"].to_numpy())), shape=(len(sa), n_anc))
    df_ = np.asarray((X > 0).sum(axis=0)).ravel()
    idf = np.log((1.0 + len(sa)) / (1.0 + df_)) + 1e-6
    X = X.multiply(idf).tocsr()
    nrm = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    X = (sp.diags(1.0 / np.maximum(nrm, 1e-12)) @ X).tocsr()
    depth = m.groupby("a")["j"].max().reindex(sa).to_numpy()
    return sa, X, depth


def neighbors(sa: np.ndarray, X, cand: np.ndarray, targets: np.ndarray, topk: int = 10, min_sim: float = 0.2, power: float = 2.0) -> pd.DataFrame:
    """targets の各 stallion について、cand の中でコサイン類似度が高い上位 topk（自分を除く、min_sim 以上）。κ = sim^power。"""
    pos = pd.Series(np.arange(len(sa)), index=sa)
    ti = pos.reindex(targets).dropna().astype(int)
    ci = pos.reindex(cand).dropna().astype(int)
    rows = []
    C = X[ci.to_numpy()]
    cand_codes = ci.index.to_numpy()
    for s in range(0, len(ti), 1000):
        blk = ti.iloc[s:s + 1000]
        S = (X[blk.to_numpy()] @ C.T).toarray()
        for r, a in enumerate(blk.index):
            sim = S[r]
            sim[cand_codes == a] = -1.0
            top = np.argpartition(-sim, min(topk, len(sim) - 1))[:topk]
            top = top[np.argsort(-sim[top])]
            for t in top:
                if sim[t] >= min_sim:
                    rows.append((int(a), int(cand_codes[t]), float(sim[t])))
    nb = pd.DataFrame(rows, columns=["a", "j", "sim"])
    nb["kappa"] = nb["sim"] ** power
    return nb


def sim_features(R: Runs, anc: pd.DataFrame, nb_by_role: dict[str, pd.DataFrame], cond: str, tag: str, k: float = 30.0, k_nb: float = 30.0) -> pd.DataFrame:
    """役割（父・父父・母父）ごとに、(a) その種牡馬自身の産駒成績（縮約先を条件平均にした従来型）、(b) 近い種牡馬の産駒成績のカーネル加重平均、
    (c) (b) を縮約先にした自身の産駒成績、を作る。馬自身の過去走は引く。"""
    O = own_and_mu(R, cond)
    own = _own_frame(O)
    F = R.frame(cond).filter(pl.col("h") >= 0)
    out = pd.DataFrame(index=np.arange(R.n))
    for role, pc in ROLE_PCLS.items():
        rt = pl.from_pandas(anc[anc["pcls"] == pc][["h", "a"]].drop_duplicates("h"))
        Lr = _long(F, rt)
        daily = daily_cum(Lr, ["a", "c"])
        Cs = lookup_prior(Lr, daily, ["a", "c"])
        base = Lr.with_columns(Cs["S"].alias("S"), Cs["N"].alias("N")).select(["_i", "a", "c", "date", "S", "N"]).join(own, on="_i", how="left")
        nb = pl.from_pandas(nb_by_role[role][["a", "j", "kappa", "sim"]]).with_columns(pl.col("a").cast(pl.Int32), pl.col("j").cast(pl.Int32))
        Q = base.select(["_i", "a", "c", "date"]).join(nb, on="a", how="inner").with_row_index("_r").with_columns(pl.col("_r").cast(pl.Int64))
        Q = Q.drop("a").rename({"j": "a"})
        Cn = lookup_prior(Q, daily, ["a", "c"])
        Q = Q.with_columns(Cn["S"].alias("Sj"), Cn["N"].alias("Nj"))
        qa = Q.group_by("_i").agg((pl.col("kappa") * pl.col("Sj")).sum().alias("KS"), (pl.col("kappa") * pl.col("Nj")).sum().alias("KN"), pl.col("sim").max().alias("smax"))
        base = base.join(qa, on="_i", how="left").with_columns(pl.col("KS").fill_null(0.0), pl.col("KN").fill_null(0.0), pl.col("smax").fill_null(0.0))
        base = base.with_columns(((pl.col("KS") + k_nb * pl.col("mu")) / (pl.col("KN") + k_nb)).alias("m_nb"))
        base = base.with_columns(((pl.col("S") - pl.col("So") + k * pl.col("m_nb")) / (pl.col("N") - pl.col("No") + k)).alias("m_sim"),
                                 ((pl.col("S") - pl.col("So") + k * pl.col("mu")) / (pl.col("N") - pl.col("No") + k)).alias("m_self"))
        f = base.select(["_i", (pl.col("m_sim") - pl.col("mu")).alias(f"{tag}_{role}_sim"), (pl.col("m_nb") - pl.col("mu")).alias(f"{tag}_{role}_nb"),
                         (pl.col("m_sim") - pl.col("m_self")).alias(f"{tag}_{role}_gain"), pl.col("KN").alias(f"{tag}_{role}_nbn"), pl.col("smax").alias(f"{tag}_{role}_smax"),
                         (pl.col("N") - pl.col("No")).alias(f"{tag}_{role}_n")]).to_pandas().set_index("_i")
        out = out.join(f)
    return out.astype("float32")
