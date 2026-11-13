"""父（○○系）と母側（母父・母母父・母母母父の○○系）の系統が、馬の適性・走法のどこに効くか（2026-11-13 ユーザー依頼）。

対象: 2 勝クラス以上（2勝・3勝・OP・L・重賞）を 1 回でも走った馬（2016〜2026 の平地）。その馬の全レースを使う。
形質（1 頭 1 行）:
  - 同じ馬の中の差（適性）: 芝ダ・芝/ダの距離の傾き・芝/ダの道悪・回り・急坂・成長。成績はクラス内の相対着順（adj）。経験ベイズで 0 に縮める。
  - 水準（脚質）: 先行性（初角の位置）・上がりの速さ（上がり 3F のレース内順位）。
  - 走法軸（2023〜。run_axes.parquet）: 芝ダ × 距離帯 × 会場 × クラスの中で標準化した値の馬の平均と、TS上限（上位 2 走の平均）。
  - 出走の配分（参考）: 芝の出走割合・平均の距離（調教師の選択で決まる面がある）。
血統: 5 代血統の (1,0)=父・(2,2)=母父・(3,6)=母母父・(4,14)=母母母父 を smartrc の小系統に。表に無い古い種牡馬は、父系をさかのぼって
最初に表にある祖先の系統、それも無ければ息子（種牡馬）の系統の多数決（割れたら大系統の名前）で補う。
分析:
  A. 影響度: 4 つの系統を同時に入れたリッジの交差検証 R² を Shapley 値で分け合う（系統どうしの相関を二重に数えない）。
     y の並べ替えで帰無分布（95% 点）、父（種牡馬）単位のブートストラップで 90% 区間。4 つとも分かる馬だけの版、種牡馬 ID（父・母父）の版も。
  B. 教師なし: PCA・因子分析（varimax）・KMeans・GMM・Ward。クラスタと系統の調整済み相互情報量（AMI）、主成分ごとの系統の寄与、
     系統 × クラスタのコレスポンデンス分析。
  C. 教師あり（確認）: LightGBM の系統ごとの並べ替え重要度。
  D. 向き: 形質ごとの小系統の縮小平均。
  E. 小系統の距離配置: 走法軸のプロファイル（縮小平均）で Ward の樹形図・MDS の 2 次元配置・近い系統（ブートストラップでの安定度）。
  F. 形質の信頼性（折半法）: 系統で説明できる割合の上限の目安。
使い方: python -m src.research.pedigree.sire_dam_influence [--fast]
  手元のデータだけを読む（GCS・GCP に触れない）。スレッドは既定で 8 に絞る（他の学習と CPU を分け合う）。
出力: data/local/pedigree/sire_dam_influence.json（data/meta/sire_lines/sire_line_analysis.html の材料。build_sire_line_hub_html.py が読む）
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ.setdefault(_v, "8")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402
from itertools import combinations  # noqa: E402
from math import factorial  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
FLAT = REPO / "data/page_reference/tables"
AXES = REPO / "data/local/running_profile/run_axes.parquet"
PED5 = REPO / "data/local/horse_pedigree_5gen"
COLORS = REPO / "data/meta/sire_lines/sire_smartrc_colors.csv"
OUT = REPO / "data/local/pedigree/sire_dam_influence.json"
JST = timezone(timedelta(hours=9))

TOP = {"2勝", "3勝", "OP", "L", "G3", "G2", "G1"}
GRADES = ("新馬", "未勝利", "1勝", "2勝", "3勝", "OP", "L", "G3", "G2", "G1")
STEEP = {"中山", "阪神", "中京"}
FLATV = {"京都", "新潟", "札幌", "函館", "小倉", "福島"}
POS = ("父", "母父", "母母父", "母母母父")
PATHS = {"父": (1, 0), "母父": (2, 2), "母母父": (3, 6), "母母母父": (4, 14)}
GENE_SHARE = {"父": 1 / 2, "母父": 1 / 4, "母母父": 1 / 8, "母母母父": 1 / 16}   # 遺伝子の期待の割合（分散では 2 乗）
BROAD = ["芝ダ（芝−ダ）", "芝の距離（長いほど＋）", "ダの距離（長いほど＋）", "芝の道悪（道悪−良）", "ダの道悪（道悪−良）",
         "回り（右−左）", "急坂（急坂−平坦）", "成長（4歳以上−3歳まで）", "先行性", "上がりの速さ"]
AXIS_NAMES = {"ts": "TS", "sustain": "TS持続力", "gear": "ギア", "endure": "Hペース耐性", "front": "先行力(軸)", "debut": "出脚",
              "cr": "Cr", "makuri": "まくり"}
FINE = ["TS", "TS上限（上位2走）", "TS持続力", "ギア", "Hペース耐性", "先行力(軸)", "出脚", "Cr", "まくり"]
PLACE = ["芝の出走割合", "平均の距離"]
TRAITS = BROAD + FINE + PLACE
ALPHAS = np.logspace(0, 4, 9)


# ---------------------------------------------------------------- 走と形質
def load_runs(years=range(2016, 2027)) -> pd.DataFrame:
    cols = ["race_id", "date", "horse_id", "surface", "distance", "direction", "track_condition", "venue", "grade",
            "finish_position", "field_size", "passing_order", "last_3f", "sex_age"]
    parts = [pd.read_parquet(FLAT / f"{y}/race_result_flat.parquet", columns=cols) for y in years if (FLAT / f"{y}/race_result_flat.parquet").exists()]
    return prepare_runs(pd.concat(parts, ignore_index=True))


def prepare_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """平地・着順あり・5 頭以上に絞り、2 勝クラス以上を 1 回でも走った馬だけにする。adj = クラス内の相対着順（クラスの平均を引く）"""
    r = runs.copy()
    r["horse_id"] = r.horse_id.astype(str)
    r["race_id"] = r.race_id.astype(str)
    r = r[r.surface.isin(["芝", "ダート"]) & r.grade.isin(GRADES)]
    r["pos"] = pd.to_numeric(r.finish_position, errors="coerce")
    r["field"] = pd.to_numeric(r.field_size, errors="coerce")
    r = r[(r.pos >= 1) & (r.field >= 5)].copy()
    # 上がりの順位は、馬を絞る前に全頭の中で付ける（絞った後だと、出走の少ないクラスで順位が上に膨らむ）
    l3f = pd.to_numeric(r.last_3f, errors="coerce")
    r["closing"] = 1 - (l3f.groupby(r.race_id).rank(method="average") - 1) / (r.field - 1)   # 1 = 上がり最速
    r = r[r.horse_id.isin(set(r.loc[r.grade.isin(TOP), "horse_id"]))].copy()
    r["pct"] = 1 - (r.pos - 1) / (r.field - 1)
    r["adj"] = r.pct - r.groupby("grade").pct.transform("mean")
    r["age"] = pd.to_numeric(r.sex_age.astype(str).str[1:], errors="coerce")
    first = pd.to_numeric(r.passing_order.astype(str).str.split("-").str[0], errors="coerce")
    r["front_ratio"] = 1 - (first - 1) / (r.field - 1)   # 大きいほど前
    return r


def within_var(runs: pd.DataFrame) -> float:
    """馬の中の走ごとの adj の分散（2 走以上の馬の、自由度で重みを付けた平均）"""
    g = runs.groupby("horse_id").adj
    n, v = g.size(), g.var(ddof=1)
    ok = n >= 2
    return float((v[ok] * (n[ok] - 1)).sum() / (n[ok] - 1).sum()) if ok.any() else float(runs.adj.var())


def shrink(d: pd.Series, v: pd.Series) -> pd.Series:
    """経験ベイズ: 全体の平均へ、精度に応じて縮める。τ² = 推定値の分散 − 誤差の分散の平均（下限あり）"""
    m = float(d.mean())
    tau2 = max(float(d.var(ddof=0) - v.mean()), 1e-6)
    return m + (d - m) * tau2 / (tau2 + v)


def contrast(runs: pd.DataFrame, mask_a, mask_b, name: str, min_n: int = 2) -> pd.Series:
    """同じ馬の中で、条件 A − 条件 B の adj の差。両方の条件を min_n 走以上走った馬だけ。経験ベイズで全体の平均へ縮める"""
    a = runs[mask_a].groupby("horse_id").adj.agg(["mean", "count"])
    b = runs[mask_b].groupby("horse_id").adj.agg(["mean", "count"])
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    j = j[(j.count_a >= min_n) & (j.count_b >= min_n)]
    if j.empty:
        return pd.Series(dtype=float, name=name)
    s2 = within_var(runs)
    d = j.mean_a - j.mean_b
    v = s2 * (1 / j.count_a + 1 / j.count_b)
    return shrink(d, v).rename(name)


def slope(runs: pd.DataFrame, mask, name: str, min_n: int = 4) -> pd.Series:
    """同じ馬・同じ馬場の中で、距離 400m あたりの adj の傾き（＋＝長いほど走る）。全体の平均へ縮める"""
    r = runs[mask].copy()
    r["x"] = r.distance / 400.0
    g = r.groupby("horse_id")
    r["xc"] = r.x - g.x.transform("mean")
    r["yc"] = r.adj - g.adj.transform("mean")
    agg = r.assign(xy=r.xc * r.yc, xx=r.xc ** 2).groupby("horse_id").agg(xy=("xy", "sum"), xx=("xx", "sum"), n=("x", "size"))
    agg = agg[(agg.n >= min_n) & (agg.xx >= 0.5)]
    if agg.empty:
        return pd.Series(dtype=float, name=name)
    b = agg.xy / agg.xx
    v = within_var(runs) / agg.xx
    return shrink(b, v).rename(name)


def axis_traits(axes: pd.DataFrame, runs: pd.DataFrame, min_runs: int = 3) -> pd.DataFrame:
    """走法軸を 芝ダ × 距離帯 × 会場 × クラス の中で標準化し、馬の平均と TS上限（上位 2 走）。min_runs 走未満の馬は NaN"""
    ax = axes.copy()
    ax["horse_id"] = ax.horse_id.astype(str)
    ax["race_id"] = ax.race_id.astype(str)
    if "abn" in ax:
        ax = ax[ax.abn != True]  # noqa: E712
    ax = ax.merge(runs[["race_id", "horse_id", "grade"]], on=["race_id", "horse_id"], how="inner")
    ax["dband"] = pd.cut(ax.distance, [0, 1400, 1800, 2200, 4000], labels=False)
    key = [ax.surface, ax.dband, ax.venue, ax.grade]
    for c in AXIS_NAMES:
        g = ax.groupby(key)[c]
        ax[c + "_r"] = (ax[c] - g.transform("mean")) / g.transform("std")
    fine = ax.groupby("horse_id").agg(**{v: (k + "_r", "mean") for k, v in AXIS_NAMES.items()}, n_ax=("ts", "size"))
    fine["TS上限（上位2走）"] = ax.groupby("horse_id").ts_r.apply(lambda s: s.nlargest(2).mean())
    fine.loc[fine.n_ax < min_runs] = np.nan
    return fine.drop(columns="n_ax").replace([np.inf, -np.inf], np.nan)


def horse_traits(runs: pd.DataFrame, axes: pd.DataFrame | None, min_c: int = 2, min_s: int = 4, min_ax: int = 3) -> pd.DataFrame:
    wet = runs.track_condition.isin(["稍重", "重", "不良"])
    turf, dirt = runs.surface == "芝", runs.surface == "ダート"
    perf = pd.concat([
        contrast(runs, turf, dirt, BROAD[0], min_c),
        slope(runs, turf, BROAD[1], min_s), slope(runs, dirt, BROAD[2], min_s),
        contrast(runs, turf & wet, turf & ~wet, BROAD[3], min_c), contrast(runs, dirt & wet, dirt & ~wet, BROAD[4], min_c),
        contrast(runs, runs.direction == "右", runs.direction == "左", BROAD[5], min_c),
        contrast(runs, runs.venue.isin(STEEP), runs.venue.isin(FLATV), BROAD[6], min_c),
        contrast(runs, runs.age >= 4, runs.age <= 3, BROAD[7], min_c)], axis=1)
    lvl = runs.groupby("horse_id").agg(n_runs=("adj", "size"), 先行性=("front_ratio", "mean"), 上がりの速さ=("closing", "mean"),
                                       芝の出走割合=("surface", lambda s: (s == "芝").mean()), 平均の距離=("distance", "mean"))
    lvl.loc[lvl.n_runs < 3, ["先行性", "上がりの速さ"]] = np.nan
    lvl["平均の距離"] = np.log(lvl["平均の距離"])
    out = lvl.join(perf)
    if axes is not None:
        out = out.join(axis_traits(axes, runs, min_ax))
    return out.reindex(columns=["n_runs"] + [t for t in TRAITS if t in out.columns or t in FINE])


# ---------------------------------------------------------------- 血統
def _read_ped(h: str):
    p = PED5 / h[:4] / f"{h}.json"
    try:
        d = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return h, None
    return h, {(a["generation"], a["position"]): a for a in d.get("ancestors", [])}


def sire_edges(peds: dict) -> dict[str, str]:
    """どの馬の 5 代血統でも (g, p) の馬の父は (g+1, 2p)。子 → 父の対応"""
    sire_of: dict[str, str] = {}
    for h, anc in peds.items():
        if not anc:
            continue
        if anc.get((1, 0)) and anc[(1, 0)].get("horse_id"):
            sire_of.setdefault(h, anc[(1, 0)]["horse_id"])
        for (gen, pos), a in anc.items():
            f = anc.get((gen + 1, pos * 2))
            if f and a.get("horse_id") and f.get("horse_id"):
                sire_of.setdefault(a["horse_id"], f["horse_id"])
    return sire_of


def line_classifier(lab: dict[str, tuple], sire_of: dict[str, str], rounds: int = 4):
    """lab: 種牡馬 ID → (小系統, 大系統, 国系統)。返す関数: ID → ((小, 大, 国) | None, 由来)。
    表に無ければ父系をさかのぼる。それも無ければ、系統の分かる息子の多数決（小系統が割れたら大系統の名前）"""
    kids: dict[str, set] = {}
    for c, f in sire_of.items():
        kids.setdefault(f, set()).add(c)
    est: dict[str, tuple] = {}
    n_sons: dict[str, int] = {}
    for _ in range(rounds):
        add = {}
        for f, cs in kids.items():
            if f in lab or f in est:
                continue
            labs = [x for x in (lab.get(c) or est.get(c) for c in cs) if x]
            if not labs:
                continue
            sm, cnt = Counter(x[0] for x in labs).most_common(1)[0]
            lg = Counter(x[1] for x in labs).most_common(1)[0][0]
            ty = Counter(x[2] for x in labs).most_common(1)[0][0]
            # 小系統は、息子 2 頭以上が全員一致したときだけ（息子 1 頭だと、その息子の名を冠した小系統が父に付きうる）。それ以外は大系統の名前
            add[f] = (sm if cnt == len(labs) and len(labs) >= 2 else lg, lg, ty)
            n_sons[f] = len(labs)
        if not add:
            break
        est.update(add)

    def classify(sid):
        """表 → 父系をさかのぼって表にある祖先 → 本人か父系の祖先の息子からの推定（近い順）"""
        seen, k, cur, first_est = set(), 0, sid, None
        while cur and cur not in seen and k <= 12:
            if cur in lab:
                return lab[cur], ("表" if k == 0 else "父系をさかのぼり")
            if first_est is None and cur in est:
                first_est = (est[cur], f"息子から推定（{n_sons.get(cur, 0)} 頭）" if k == 0 else f"父系の祖先を息子から推定（{n_sons.get(cur, 0)} 頭）")
            seen.add(cur)
            cur, k = sire_of.get(cur), k + 1
        return first_est if first_est else (None, None)
    return classify


def pedigree_lines(horse_ids, peds_all: dict | None = None) -> pd.DataFrame:
    col = pd.read_csv(COLORS, dtype=str)
    col = col[col.status == "ok"]
    lab = {r.sire_id: (r.small_name, r.large_name, r.type_name) for r in col.itertuples()}
    if peds_all is None:
        files = sorted(x.stem for x in PED5.glob("*/*.json"))
        with ThreadPoolExecutor(16) as ex:
            peds_all = dict(ex.map(_read_ped, files))
    classify = line_classifier(lab, sire_edges(peds_all))
    rows = []
    for h in horse_ids:
        anc = peds_all.get(h)
        r = {"horse_id": h}
        for k, gp in PATHS.items():
            a = anc.get(gp) if anc else None
            v, how = classify(a.get("horse_id")) if a else (None, None)
            r[k + "_id"] = a.get("horse_id") if a else None
            r[k + "_小"], r[k + "_大"], r[k + "_国"] = v if v else (None, None, None)
            r[k + "_how"] = how.split("（")[0] if how else None
            r[k + "_how_detail"] = how
        rows.append(r)
    return pd.DataFrame(rows).set_index("horse_id")


def factor_frame(df: pd.DataFrame, level: str = "小", min_n: int = 30) -> pd.DataFrame:
    """系統の列（不明は「不明」、min_n 頭未満は「その他」）"""
    f = pd.DataFrame({p: df[f"{p}_{level}"].fillna("不明") for p in POS}, index=df.index)
    for p in POS:
        vc = f[p].value_counts()
        f.loc[f[p].isin(vc[vc < min_n].index), p] = "その他"
    return f


# ---------------------------------------------------------------- A. 影響度
def _onehot(s: pd.Series) -> np.ndarray:
    return pd.get_dummies(s.astype(str)).to_numpy(float)


def cv_r2(Xs: list[np.ndarray], y: np.ndarray, folds) -> float:
    """交差検証 R²（リッジ。α の候補を正規方程式でまとめて解き、最良を取る）。負は 0。
    α を検証側で選ぶ分だけ楽観が入る。検定（帰無の 95% 点）は帰無も同じ選び方なので公平。
    点の値の楽観は、帰無の平均（負を 0 で切るのでほぼ 0）では測れない。α の候補 9 つ・数千頭なので小さいと見込む"""
    if not Xs:
        return 0.0
    X = np.hstack(Xs)
    sst = ((y - y.mean()) ** 2).sum()
    preds = {a: np.zeros_like(y) for a in ALPHAS}
    for tr, te in folds:
        mx, my = X[tr].mean(0), y[tr].mean()
        Xc = X[tr] - mx
        G, r = Xc.T @ Xc, Xc.T @ (y[tr] - my)
        eye = np.eye(G.shape[0])
        for a in ALPHAS:
            preds[a][te] = (X[te] - mx) @ np.linalg.solve(G + a * eye, r) + my
    return float(max(0.0, max(1 - ((y - p) ** 2).sum() / sst for p in preds.values())))


def shapley(F: pd.DataFrame, y: np.ndarray, folds) -> tuple[dict, float, dict]:
    """系統（F の列）ごとの R² の Shapley 値。和は全部入れた R² に一致する"""
    cols = list(F.columns)
    enc = {c: _onehot(F[c]) for c in cols}
    v = {S: cv_r2([enc[c] for c in S], y, folds) for k in range(len(cols) + 1) for S in combinations(cols, k)}
    n = len(cols)
    out = {}
    for c in cols:
        rest = [x for x in cols if x != c]
        s = 0.0
        for k in range(n):
            for S in combinations(rest, k):
                Sc = tuple(x for x in cols if x in S or x == c)
                s += factorial(k) * factorial(n - k - 1) / factorial(n) * (v[Sc] - v[S])
        out[c] = s
    return out, v[tuple(cols)], {c: v[(c,)] for c in cols}


def _folds(n: int, k: int, seed: int):
    idx = np.random.default_rng(seed).permutation(n)
    parts = np.array_split(idx, k)
    return [(np.setdiff1d(idx, p), p) for p in parts]


def influence(y_s: pd.Series, F: pd.DataFrame, cluster: pd.Series | None, nperm: int, nboot: int, rng) -> dict:
    m = y_s.notna() & F.notna().all(axis=1)
    y = y_s[m].to_numpy(float)
    y = (y - y.mean()) / y.std()
    Fm = F.loc[m].reset_index(drop=True)
    folds = _folds(len(y), 5, 0)
    sh, full, alone = shapley(Fm, y, folds)
    null = [shapley(Fm, rng.permutation(y), folds)[0] for _ in range(nperm)]
    cols = list(F.columns)
    res = {"n": int(m.sum()), "full_r2": full, "shapley": sh, "alone": alone,
           "p": {c: (1 + sum(z[c] >= sh[c] for z in null)) / (1 + len(null)) for c in cols},
           "null95": {c: float(np.quantile([z[c] for z in null], 0.95)) if null else 0.0 for c in cols},
           "null_mean": {c: float(np.mean([z[c] for z in null])) if null else 0.0 for c in cols}, "ci90": {}}
    # 正味の寄与 = 点の値 − 帰無の平均（区分の数の多さによる見かけの寄与を差し引く。α の選び方の楽観はここでは測れない）
    res["net"] = {c: res["shapley"][c] - res["null_mean"][c] for c in cols}
    if nboot and cluster is not None:
        cl = cluster[m].fillna("不明").astype(str).to_numpy()
        us = np.unique(cl)
        idx = {s: np.where(cl == s)[0] for s in us}
        bs = []
        for _ in range(nboot):
            pick = np.concatenate([idx[s] for s in rng.choice(us, len(us))])
            # 交差検証は元の馬ごとにまとめて分ける（同じ父を重複して引いたとき、同じ馬の写しが学習と検証の両方に入らないように。
            # 兄弟は点の推定と同じく別々の側に入ってよい）
            g = pick
            ug = np.random.default_rng(1).permutation(np.unique(g))
            fold_of = {s: i % 5 for i, s in enumerate(ug)}
            fid = np.array([fold_of[s] for s in g])
            bf = [(np.where(fid != k)[0], np.where(fid == k)[0]) for k in range(5)]
            bs.append(shapley(Fm.iloc[pick].reset_index(drop=True), y[pick], bf)[0])
        # 父を重複して引くと系統の間の分散が膨らみ、分布が上に偏る。基本区間（2θ − 分位点）で偏りを打ち消す。
        # 偏りが一定という仮定に頼り、0 で切るので、点が区間の外に出ることがある（目安として使う）
        res["ci90"] = {c: [max(0.0, 2 * sh[c] - float(np.quantile([b[c] for b in bs], 0.95))), max(0.0, 2 * sh[c] - float(np.quantile([b[c] for b in bs], 0.05)))] for c in cols}
        res["boot_mean"] = {c: float(np.mean([b[c] for b in bs])) for c in cols}
    return res


# ---------------------------------------------------------------- B〜F
def unsupervised(df: pd.DataFrame, F: pd.DataFrame, cols: list[str], rows: pd.Series, rng, nperm: int) -> dict:
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.decomposition import PCA, FactorAnalysis
    from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
    from sklearn.mixture import GaussianMixture
    Z = df.loc[rows, cols]
    Z = ((Z - Z.mean()) / Z.std()).fillna(0.0).to_numpy()   # 測れない対照（例: 芝しか走らない馬の芝ダ）は 0（縮小の事前の値）
    pca = PCA().fit(Z)
    k_pc = int(max(2, (pca.explained_variance_ > 1).sum()))
    pcs = pca.transform(Z)[:, :k_pc]
    fa = FactorAnalysis(n_components=k_pc, rotation="varimax", random_state=0).fit(Z)
    sub = rng.choice(len(Z), min(len(Z), 4000), replace=False)
    sil = {k: float(silhouette_score(pcs[sub], KMeans(k, n_init=5, random_state=0).fit_predict(pcs)[sub])) for k in range(3, 9)}
    kbest = max(sil, key=sil.get)
    km = KMeans(kbest, n_init=10, random_state=0).fit(pcs)
    bic = {k: float(GaussianMixture(k, random_state=0, n_init=2).fit(pcs).bic(pcs)) for k in range(2, 9)}
    gk = min(bic, key=bic.get)
    gm = GaussianMixture(gk, random_state=0, n_init=3).fit(pcs).predict(pcs)
    wsub = rng.choice(len(Z), min(len(Z), 5000), replace=False)
    ward = AgglomerativeClustering(kbest, linkage="ward").fit_predict(pcs[wsub])
    Fr = F.loc[rows].reset_index(drop=True)
    ami = {}
    for name, lab in {"KMeans": km.labels_, "GMM": gm}.items():
        ami[name] = {}
        for p in POS:
            nulls = [adjusted_mutual_info_score(rng.permutation(Fr[p].to_numpy()), lab) for _ in range(10)]
            ami[name][p] = [float(adjusted_mutual_info_score(Fr[p], lab)), float(np.quantile(nulls, 0.95))]
    ami["Ward（5千頭）"] = {p: [float(adjusted_mutual_info_score(Fr[p].iloc[wsub], ward)), None] for p in POS}
    pc_inf = {}
    fsc = fa.transform(Z)   # 因子の名前（fa_load）と同じ、回転後の因子の得点で寄与を測る
    for i in range(k_pc):
        y = pd.Series(np.nan, index=df.index)
        y.loc[rows] = fsc[:, i]
        pc_inf[f"F{i+1}"] = influence(y, F, None, nperm, 0, rng)
    prof = pd.DataFrame(Z, columns=cols).groupby(km.labels_).mean().round(2)
    ca = {}
    for p in ("父", "母父"):
        ct = pd.crosstab(Fr[p], km.labels_)
        ct = ct[~ct.index.isin(["不明", "その他"])]
        P = ct.values / ct.values.sum()
        rm, cm = P.sum(1), P.sum(0)
        S = (P - np.outer(rm, cm)) / np.sqrt(np.outer(rm, cm))
        U, s, Vt = np.linalg.svd(S, full_matrices=False)
        e = np.outer(ct.sum(1), ct.sum(0)) / ct.values.sum()
        ca[p] = {"resid": {line: {int(c): round(float(v), 2) for c, v in row.items()} for line, row in ((ct - e) / np.sqrt(e)).iterrows()},
                 "rows": {line: ((U[i, :2] * s[:2]) / np.sqrt(rm[i])).round(3).tolist() for i, line in enumerate(ct.index)},
                 "cols": {int(c): ((Vt[:2, i] * s[:2]) / np.sqrt(cm[i])).round(3).tolist() for i, c in enumerate(ct.columns)},
                 "inertia": (s[:2] ** 2 / (s ** 2).sum()).round(3).tolist()}
    # 因子の名前は、因子分析の負荷が最大の形質の向きで付ける（画面で使う）
    return {"n": int(rows.sum()), "cols": cols, "pca_var": pca.explained_variance_ratio_[:k_pc].round(4).tolist(),
            "fa_load": {f"F{i+1}": dict(zip(cols, fa.components_[i].round(3).tolist())) for i in range(k_pc)},
            "kmeans_k": kbest, "silhouette": sil, "gmm_k": gk, "bic": bic, "ami": ami, "factor_influence": pc_inf,
            "cluster_profile": {int(k): v for k, v in prof.to_dict(orient="index").items()},
            "cluster_size": {int(k): int(v) for k, v in pd.Series(km.labels_).value_counts().sort_index().items()}, "ca": ca}


def lgb_importance(df: pd.DataFrame, F: pd.DataFrame, rng) -> dict:
    import lightgbm as lgb
    out = {}
    for t in TRAITS:
        m = df[t].notna()
        X = F.loc[m, list(POS)].astype("category")
        y = df.loc[m, t].to_numpy(float)
        tr, te = _folds(len(y), 4, 0)[0]
        mdl = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40, verbose=-1, n_jobs=8)
        mdl.fit(X.iloc[tr], y[tr])
        base = ((y[te] - mdl.predict(X.iloc[te])) ** 2).mean()
        imp = {}
        for c in POS:
            ds = []
            for _ in range(5):
                Xp = X.iloc[te].copy()
                Xp[c] = pd.Categorical(rng.permutation(Xp[c].to_numpy()), categories=X[c].cat.categories)
                ds.append(((y[te] - mdl.predict(Xp)) ** 2).mean() - base)
            imp[c] = float(np.mean(ds) / y[te].var())
        out[t] = imp
    return out


def shrunk_means(df: pd.DataFrame, F: pd.DataFrame, t: str, p: str, k: int = 20, min_n: int = 30) -> dict:
    """系統の平均（標準化、頭数で 0 に縮める）。その形質を持つ馬が min_n 頭未満の系統は出さない"""
    m = df[t].notna() & ~F[p].isin(["不明", "その他"])
    z = (df.loc[m, t] - df.loc[m, t].mean()) / df.loc[m, t].std()
    g = z.groupby(F.loc[m, p]).agg(["mean", "count"])
    g = g[g["count"] >= min_n]
    g["shr"] = g["mean"] * g["count"] / (g["count"] + k)
    return {i: [round(float(r.shr), 3), int(r["count"])] for i, r in g.sort_values("shr").iterrows()}


def line_map(df: pd.DataFrame, F: pd.DataFrame, p: str, cols: list[str], rng, nboot: int, min_n: int = 30, k: int = 20,
             cluster: pd.Series | None = None) -> dict:
    """小系統の距離配置。プロファイル = 形質（標準化）の系統平均を頭数で 0 に縮めたもの。距離はユークリッド"""
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist, squareform
    from sklearn.manifold import MDS
    m = df[cols].notna().all(1) & ~F[p].isin(["不明", "その他"])
    Z = df.loc[m, cols]
    Z = (Z - Z.mean()) / Z.std()
    lines = F.loc[m, p]
    n = lines.value_counts()
    keep = sorted(n[n >= min_n].index)
    if len(keep) < 3:
        return {"cols": cols, "lines": keep, "n": {}, "profile": {}, "xy": {}, "dendro_order": keep, "linkage": [], "near": {}, "nn_stability": {}}

    def prof(Zx, lx):
        g = Zx.groupby(lx.to_numpy())
        nn = g.size().reindex(keep).fillna(0)
        return g.mean().reindex(keep).fillna(0).mul(nn / (nn + k), axis=0)
    shr = prof(Z, lines)
    D = squareform(pdist(shr.to_numpy()))
    L = linkage(shr.to_numpy(), "ward")
    order = dendrogram(L, labels=keep, no_plot=True)["ivl"]
    xy = MDS(2, dissimilarity="precomputed", random_state=0, n_init=8, normalized_stress="auto").fit_transform(D)
    stab = {a: {} for a in keep}
    cl = (cluster.loc[Z.index].fillna("不明").astype(str).to_numpy() if cluster is not None else np.arange(len(Z)).astype(str))
    us = np.unique(cl)
    idx = {s: np.where(cl == s)[0] for s in us}
    for _ in range(nboot):
        pick = np.concatenate([idx[s] for s in rng.choice(us, len(us))])   # 父（種牡馬）単位で引き直す（産駒は束で似る）
        Db = squareform(pdist(prof(Z.iloc[pick], lines.iloc[pick]).to_numpy()))
        np.fill_diagonal(Db, np.inf)
        for i, a in enumerate(keep):
            b = keep[int(np.argmin(Db[i]))]
            stab[a][b] = stab[a].get(b, 0) + 1
    Dd = D.copy()
    np.fill_diagonal(Dd, np.inf)
    return {"cols": cols, "lines": keep, "n": {a: int(n[a]) for a in keep}, "profile": shr.round(3).to_dict(orient="index"),
            "xy": {a: xy[i].round(3).tolist() for i, a in enumerate(keep)}, "dendro_order": order, "linkage": L.round(4).tolist(),
            "near": {a: [[keep[j], round(float(Dd[i, j]), 3)] for j in np.argsort(Dd[i])[:3]] for i, a in enumerate(keep)},
            "nn_stability": {a: {b: c / max(nboot, 1) for b, c in v.items()} for a, v in stab.items()}}


def reliability(runs: pd.DataFrame, axes: pd.DataFrame | None) -> dict:
    """折半法: 馬ごとの走を奇数番目・偶数番目に分けて形質を 2 回作り、Spearman の相関を Spearman-Brown で補正"""
    r = runs.sort_values(["horse_id", "date"]).copy()
    r["k"] = r.groupby("horse_id").cumcount()
    a = horse_traits(r[r.k % 2 == 0], axes, 1, 2, 2)
    b = horse_traits(r[r.k % 2 == 1], axes, 1, 2, 2)
    out = {}
    for c in TRAITS:
        if c not in a or c not in b:
            continue
        j = pd.concat([a[c], b[c]], axis=1).dropna()
        if len(j) < 100:
            continue
        rho = float(j.corr(method="spearman").iloc[0, 1])
        out[c] = {"n": int(len(j)), "half_r": round(rho, 3), "reliability": round(2 * rho / (1 + rho), 3) if rho > 0 else round(rho, 3)}
    return out


# ---------------------------------------------------------------- 実行
def run(fast: bool = False, out: Path = OUT) -> dict:
    warnings.filterwarnings("ignore")
    nperm, nboot = (5, 5) if fast else (20, 20)
    rng = np.random.default_rng(0)
    runs = load_runs()
    axes = pd.read_parquet(AXES) if AXES.exists() else None
    print(f"対象 {runs.horse_id.nunique()} 頭・{len(runs)} 走", file=sys.stderr)
    df = horse_traits(runs, axes)
    df = df.join(pedigree_lines(list(df.index)))
    F = factor_frame(df)
    res: dict = {"built": datetime.now(JST).strftime("%Y-%m-%d %H:%M"), "fast": fast, "n_horses": int(len(df)), "n_runs": int(len(runs)),
                 "gene_share": GENE_SHARE, "traits": {"broad": BROAD, "fine": FINE, "place": PLACE},
                 "coverage": {p: round(float(df[f"{p}_小"].notna().mean()), 3) for p in POS},
                 "how": {p: df[f"{p}_how"].fillna("なし").value_counts().to_dict() for p in POS},
                 "n_sons": {p: df[f"{p}_how_detail"].dropna()[lambda s: s.str.contains("息子")].value_counts().head(20).to_dict() for p in POS}}
    print("F 信頼性", file=sys.stderr)
    res["reliability"] = reliability(runs, axes)
    print("A 影響度", file=sys.stderr)
    res["A_line"] = {t: influence(df[t], F, df["父_id"], nperm, nboot, rng) for t in TRAITS}
    known = df[[f"{p}_小" for p in POS]].notna().all(axis=1)
    res["n_complete"] = int(known.sum())
    res["A_line_complete"] = {t: influence(df[t], F.where(known), None, nperm, 0, rng) for t in TRAITS}
    Fi = pd.DataFrame({"父(種牡馬)": df["父_id"], "母父(種牡馬)": df["母父_id"]}, index=df.index)
    for c in Fi:
        vc = Fi[c].value_counts()
        Fi.loc[Fi[c].isin(vc[vc < 20].index), c] = "その他"
    res["A_stallion"] = {t: influence(df[t], Fi, None, nperm, 0, rng) for t in TRAITS}
    print("C LightGBM", file=sys.stderr)
    try:
        res["C_lgb"] = lgb_importance(df, F, rng)
    except Exception as e:  # noqa: BLE001
        res["C_lgb_error"] = repr(e)
    print("B 教師なし", file=sys.stderr)
    res["B_broad"] = unsupervised(df, F, BROAD, df[BROAD].notna().sum(1) >= 5, rng, nperm)
    res["B_all"] = unsupervised(df, F, BROAD + FINE, df[FINE].notna().all(1), rng, nperm)
    print("D・E", file=sys.stderr)
    res["D"] = {t: {p: shrunk_means(df, F, t, p) for p in ("父", "母父")} for t in TRAITS}
    axes8 = [c for c in FINE if c != "TS上限（上位2走）"]   # TS上限は TS と相関が高く、距離の中で TS を二重に数えるので外す
    # 引き直しは、その位置の種牡馬単位（父の地図は父、母父の地図は母父。産駒・孫はその種牡馬ごとに束で似る）
    res["E"] = {p: line_map(df, F, p, axes8, rng, nboot, cluster=df[f"{p}_id"]) for p in ("父", "母父")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"保存: {out}", file=sys.stderr)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fast", action="store_true", help="並べ替え・ブートストラップを 5 回に減らす（試し用）")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    run(a.fast, a.out)


if __name__ == "__main__":
    main()
