"""
5.3 アイデア群に対応する種牡馬×舞台特徴の前計算（評価ラボ用）

母系種牡馬集合 S_h と当日 full_stage_key に対し、集約スカラ・グラフ平滑・因子接続などを返す。
各関数は docs/html/modeling/pedigree_stage_research_notes.html §5.3 の記号にコメントで対応付け。
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

logger = logging.getLogger(__name__)


def _safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p.astype(np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _entropy_from_probs(p: np.ndarray) -> float:
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p + 1e-15)))


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if n < 2:
        return 0.0
    return float(np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / (n * np.sum(x) + 1e-15))


def _winsorize(a: np.ndarray, lo: float = 0.05, hi: float = 0.95) -> np.ndarray:
    if a.size == 0:
        return a
    low, high = np.quantile(a, [lo, hi])
    return np.clip(a, low, high)


# --- データ構造 -----------------------------------------------------------------


class StageStatsStore:
    """long Parquet (stallion_id, full_stage_key, starts, wins, ...) から派生テーブル。"""

    def __init__(self, long_df: pd.DataFrame):
        self.df = long_df.copy()
        self._cell: dict[tuple[str, str], dict[str, float]] = {}
        self._by_stage: dict[str, list[str]] = defaultdict(list)
        for _, r in self.df.iterrows():
            sid = str(r["stallion_id"]).strip()
            fsk = str(r["full_stage_key"]).strip()
            st = int(r["starts"])
            w = int(r["wins"])
            t3 = int(r["top3"])
            wr = float(r["win_rate"])
            tr = float(r["top3_rate"])
            af = float(r["avg_finish"])
            self._cell[(sid, fsk)] = {
                "starts": float(st),
                "wins": float(w),
                "top3": float(t3),
                "win_rate": wr,
                "top3_rate": tr,
                "avg_finish": af,
            }
            self._by_stage[fsk].append(sid)

        # A: 多解像度（粗い stage_key）— 母系集計と同じキーで集約
        self._cell_coarse: dict[tuple[str, str], float] = {}
        if "stage_key" in self.df.columns:
            cg = self.df.groupby(["stallion_id", "stage_key"], as_index=False).agg(
                starts=("starts", "sum"),
                wins=("wins", "sum"),
            )
            cg["wr"] = cg["wins"] / cg["starts"].clip(lower=1)
            for _, r in cg.iterrows():
                self._cell_coarse[(str(r["stallion_id"]).strip(), str(r["stage_key"]).strip())] = float(
                    r["wr"]
                )

        # A: グローバル平均（残差・EB 用）
        g = self.df.groupby("full_stage_key", as_index=False).agg(
            g_starts=("starts", "sum"),
            g_wins=("wins", "sum"),
        )
        g["global_wr"] = g["g_wins"] / g["g_starts"].clip(lower=1)
        self.global_wr_by_stage = dict(zip(g["full_stage_key"], g["global_wr"]))

        # EB: セル (sid, fsk) の勝率を beta 近似で縮小（簡易）
        mu0 = float(self.df["wins"].sum() / self.df["starts"].sum().clip(min=1))
        k0 = 20.0  # 疑似カウント
        self._mu0 = mu0
        self._k0 = k0
        self._eb_wr: dict[tuple[str, str], float] = {}
        for (sid, fsk), d in self._cell.items():
            n = d["starts"]
            w = d["wins"]
            self._eb_wr[(sid, fsk)] = (w + k0 * mu0) / (n + k0)

        # A: 種牡馬ごとに「舞台分布のエントロピー・ジニ」（win_rate 質量で正規化）
        self.stallion_entropy: dict[str, float] = {}
        self.stallion_gini: dict[str, float] = {}
        for sid, sub in self.df.groupby("stallion_id"):
            wrs = sub["win_rate"].values * sub["starts"].values
            ssum = wrs.sum() + 1e-9
            p = wrs / ssum
            self.stallion_entropy[str(sid)] = _entropy_from_probs(p)
            self.stallion_gini[str(sid)] = _gini(sub["win_rate"].values)

        # 行列 (stallion x stage) win_rate for PCA/NMF/SVD
        pivot = self.df.pivot_table(
            index="stallion_id",
            columns="full_stage_key",
            values="win_rate",
            aggfunc="mean",
        ).fillna(self._mu0)
        self._pivot_wr = pivot
        self._stage_cols = list(pivot.columns)
        self._stallion_rows = list(pivot.index.astype(str))

        X = pivot.values
        self._X_wr = X
        n_comp = min(8, min(X.shape) - 1, X.shape[1], X.shape[0])
        n_comp = max(1, n_comp)
        self.pca = PCA(n_components=min(6, n_comp), random_state=42)
        self.pca.fit(X)
        self.nmf = NMF(n_components=min(6, n_comp), init="nndsvda", random_state=42, max_iter=800)
        try:
            self.nmf.fit(np.clip(X, 1e-6, None))
        except Exception as e:
            logger.warning("NMF fit skipped: %s", e)
            self.nmf = None
        self.svd = TruncatedSVD(n_components=min(6, n_comp), random_state=42)
        Xt = self.svd.fit_transform(X)
        svd_dim = self.svd.components_.shape[0]

        self._stallion_pc = pd.DataFrame(
            self.pca.transform(X),
            index=self._stallion_rows,
            columns=[f"pca_{i}" for i in range(self.pca.n_components_)],
        )
        if self.nmf is not None:
            self._stallion_nmf = pd.DataFrame(
                self.nmf.transform(np.clip(X, 1e-6, None)),
                index=self._stallion_rows,
                columns=[f"nmf_{i}" for i in range(self.nmf.n_components_)],
            )
        else:
            self._stallion_nmf = None
        self._stallion_svd = pd.DataFrame(
            Xt,
            index=self._stallion_rows,
            columns=[f"svd_{i}" for i in range(svd_dim)],
        )

    def get_cell(self, sid: str, fsk: str) -> dict[str, float] | None:
        return self._cell.get((sid, fsk))

    def eb_wr(self, sid: str, fsk: str) -> float | None:
        return self._eb_wr.get((sid, fsk))

    def residual_wr(self, sid: str, fsk: str) -> float | None:
        d = self.get_cell(sid, fsk)
        if not d:
            return None
        g = self.global_wr_by_stage.get(fsk, self._mu0)
        return float(d["win_rate"] - g)

    def mean_aggregates(
        self,
        stallion_ids: frozenset[str],
        fsk: str,
        *,
        mode: str = "raw",
    ) -> dict[str, float]:
        """D: 平均・最大・最小・logsumexp 系。mode ∈ raw | eb | residual"""
        wrs, trs, afs, sts = [], [], [], []
        for sid in stallion_ids:
            d = self.get_cell(sid, fsk)
            if not d:
                continue
            if mode == "eb":
                e = self.eb_wr(sid, fsk)
                wr = e if e is not None else d["win_rate"]
            elif mode == "residual":
                r = self.residual_wr(sid, fsk)
                wr = float(r) if r is not None else 0.0
            else:
                wr = d["win_rate"]
            wrs.append(wr)
            trs.append(d["top3_rate"])
            afs.append(d["avg_finish"])
            sts.append(d["starts"])
        if not wrs:
            return {}
        a = np.array(wrs, dtype=np.float64)
        return {
            "m_mean_wr": float(np.mean(a)),
            "m_max_wr": float(np.max(a)),
            "m_min_wr": float(np.min(a)),
            "m_logsumexp_wr": float(np.log(np.exp(a).sum() + 1e-15)),
            "m_mean_tr": float(np.mean(trs)),
            "m_mean_af": float(np.mean(afs)),
            "m_sum_starts": float(np.sum(sts)),
            "m_n_anc": float(len(wrs)),
        }

    def mean_aggregates_coarse(self, stallion_ids: frozenset[str], stage_key: str) -> dict[str, float]:
        """A: 粗い stage_key のみ（馬場×芝ダ×距離、cond 無し）。"""
        wrs = []
        for sid in stallion_ids:
            w = self._cell_coarse.get((sid, stage_key))
            if w is not None:
                wrs.append(w)
        if not wrs:
            return {}
        a = np.array(wrs, dtype=np.float64)
        return {
            "m_mean_wr_coarse": float(np.mean(a)),
            "m_max_wr_coarse": float(np.max(a)),
        }

    def stallion_profile_features(self, stallion_ids: frozenset[str]) -> dict[str, float]:
        """A: エントロピー・ジニの母系平均 + PCA/NMF/SVD 係数の平均。"""
        ents, ginis = [], []
        for sid in stallion_ids:
            ents.append(self.stallion_entropy.get(sid, 0.0))
            ginis.append(self.stallion_gini.get(sid, 0.0))
        out: dict[str, float] = {}
        if ents:
            out["m_mean_entropy"] = float(np.mean(ents))
            out["m_mean_gini"] = float(np.mean(ginis))
        pc_cols = [c for c in self._stallion_pc.columns]
        for c in pc_cols:
            vals = [self._stallion_pc.loc[sid, c] for sid in stallion_ids if sid in self._stallion_pc.index]
            if vals:
                out[f"m_mean_{c}"] = float(np.mean(vals))
        if self._stallion_nmf is not None:
            for c in self._stallion_nmf.columns:
                vals = [
                    self._stallion_nmf.loc[sid, c]
                    for sid in stallion_ids
                    if sid in self._stallion_nmf.index
                ]
                if vals:
                    out[f"m_mean_{c}_nmf"] = float(np.mean(vals))
        for c in self._stallion_svd.columns:
            vals = [self._stallion_svd.loc[sid, c] for sid in stallion_ids if sid in self._stallion_svd.index]
            if vals:
                out[f"m_mean_{c}_svd"] = float(np.mean(vals))
        return out


def build_graph_bundle(path: Path | None) -> dict[str, Any]:
    """C: グラフ JSON から隣接・辺タイプ。"""
    out: dict[str, Any] = {
        "neighbors": defaultdict(set),
        "neighbors_sire": defaultdict(set),
        "neighbors_damsire": defaultdict(set),
        "nodes": set(),
    }
    if path is None or not path.exists():
        return out
    data = json.loads(path.read_text(encoding="utf-8"))
    for n in data.get("nodes", []):
        if isinstance(n, dict) and n.get("id"):
            out["nodes"].add(str(n["id"]))
        elif isinstance(n, str):
            out["nodes"].add(n)
    for e in data.get("edges", []):
        s, t = str(e["source"]), str(e["target"])
        rel = e.get("relation", "")
        out["nodes"].add(s)
        out["nodes"].add(t)
        out["neighbors"][s].add(t)
        out["neighbors"][t].add(s)
        if rel == "sire":
            out["neighbors_sire"][s].add(t)
            out["neighbors_sire"][t].add(s)
        elif rel == "dam_sire":
            out["neighbors_damsire"][s].add(t)
            out["neighbors_damsire"][t].add(s)
    return out


def graph_smooth_wr(
    store: StageStatsStore,
    gb: dict[str, Any],
    stallion_ids: frozenset[str],
    fsk: str,
    *,
    hops: int = 1,
) -> dict[str, float]:
    """C: 近傍平滑・1/2ホップ（同一 fsk の win_rate を隣接で平均）。"""
    neigh = gb.get("neighbors") or defaultdict(set)

    def collect(ids: set[str], depth: int) -> set[str]:
        cur = set(ids)
        for _ in range(depth):
            nxt = set(cur)
            for u in cur:
                nxt |= neigh.get(u, set())
            cur = nxt
        return cur

    seeds = set(stallion_ids)
    if not seeds:
        return {}
    bag = collect(seeds, hops)
    wrs = []
    for sid in bag:
        d = store.get_cell(sid, fsk)
        if d:
            wrs.append(d["win_rate"])
    if not wrs:
        return {}
    a = np.array(wrs)
    return {
        f"g_smooth_h{hops}_mean_wr": float(np.mean(a)),
        f"g_smooth_h{hops}_max_wr": float(np.max(a)),
    }


def ppr_scores(
    gb: dict[str, Any],
    seeds: frozenset[str],
    *,
    alpha: float = 0.85,
    iters: int = 24,
    max_nodes: int = 320,
) -> dict[str, float]:
    """C: Personalized PageRank（シード周辺サブグラフのみで密行列）。"""
    neigh = gb.get("neighbors") or defaultdict(set)
    if not seeds:
        return {}
    sub: set[str] = set(seeds)
    for s in list(seeds):
        sub |= neigh.get(s, set())
        for v in list(neigh.get(s, set())):
            sub |= neigh.get(v, set())
    nodes = sorted(sub)[:max_nodes]
    if not nodes:
        return {}
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n), dtype=np.float64)
    for u in nodes:
        vs = list(neigh.get(u, set()))
        if not vs:
            continue
        for v in vs:
            if v in idx:
                A[idx[u], idx[v]] = 1.0
    row_sums = A.sum(axis=1, keepdims=True) + 1e-15
    P = A / row_sums
    s = np.zeros(n)
    for u in seeds:
        if u in idx:
            s[idx[u]] = 1.0 / max(len(seeds), 1)
    r = np.ones(n) / n
    for _ in range(iters):
        r = alpha * P.T @ r + (1 - alpha) * s
    top = float(np.max(r))
    ent = float(-np.sum(r * np.log(r + 1e-15)))
    return {"g_ppr_max": top, "g_ppr_entropy": ent}


def shortest_path_kernel(
    gb: dict[str, Any],
    stallion_ids: frozenset[str],
    *,
    lam: float = 0.35,
) -> dict[str, float]:
    """C: シード間の最短距離カーネル風スカラ。"""
    neigh = gb.get("neighbors") or defaultdict(set)
    seeds = list(stallion_ids)
    if len(seeds) < 2:
        return {"g_sp_kernel": 1.0}
    from collections import deque

    def dist(a: str, b: str) -> int:
        if a == b:
            return 0
        q = deque([(a, 0)])
        seen = {a}
        while q:
            u, d = q.popleft()
            for v in neigh.get(u, set()):
                if v == b:
                    return d + 1
                if v not in seen:
                    seen.add(v)
                    q.append((v, d + 1))
        return 10**6

    ds = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            ds.append(dist(seeds[i], seeds[j]))
    if not ds:
        return {}
    md = float(np.median(ds))
    return {"g_sp_kernel": float(np.exp(-lam * md))}


def louvain_like_labels(gb: dict[str, Any]) -> dict[str, int]:
    """C: 連結成分ラベル（Louvain 代替の簡易クラスタ ID）。"""
    neigh = gb.get("neighbors") or defaultdict(set)
    all_nodes: set[str] = set(gb.get("nodes") or [])
    all_nodes |= set(neigh.keys())
    for vs in neigh.values():
        all_nodes |= vs
    seen: set[str] = set()
    labels: dict[str, int] = {}
    cid = 0
    for start in sorted(all_nodes):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        while stack:
            u = stack.pop()
            labels[u] = cid
            for v in neigh.get(u, set()):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        cid += 1
    return labels


def factor_features(
    sire_stats: dict[str, Any],
    stallion_ids: frozenset[str],
    axes: list[str],
) -> dict[str, float]:
    """B: 種牡馬因子軸の母系平均（蒸留・マルチタスクの代理特徴）。"""
    sires = sire_stats.get("sires") or {}
    vecs: list[list[float]] = []
    for sid in stallion_ids:
        ent = sires.get(sid)
        if not isinstance(ent, dict):
            continue
        row = []
        for ax in axes:
            m = ent.get("maternal") or {}
            axd = m.get("axes") or {}
            row.append(float(axd.get(ax, 0.0)))
        if row:
            vecs.append(row)
    if not vecs:
        return {}
    M = np.array(vecs, dtype=np.float64)
    out = {f"f_mean_{ax}": float(np.mean(M[:, j])) for j, ax in enumerate(axes)}
    out["f_mean_l2"] = float(np.linalg.norm(np.mean(M, axis=0)))
    return out


def orthogonal_residual_to_factors(
    store: StageStatsStore,
    stallion_ids: frozenset[str],
    fsk: str,
    factor_means: dict[str, float],
    axes: list[str],
) -> dict[str, float]:
    """B: 因子ベクトルに直交する残差（舞台 win_rate 平均 - 因子から説明される部分の代理）。"""
    # 簡易: 残差 = mean_wr - k * mean_consistency（k は訓練で決めるべきだがラボでは固定スケール）
    agg = store.mean_aggregates(stallion_ids, fsk, mode="raw")
    mw = agg.get("m_mean_wr", 0.0)
    fc = factor_means.get("f_mean_consistency", 0.0)
    return {"b_ortho_wr_proxy": float(mw - 0.1 * fc)}


def bilinear_random_projection(
    mean_wr: float,
    mean_tr: float,
    n_anc: float,
    seed: int = 42,
) -> dict[str, float]:
    """E: 低ランク双線形の代理（固定乱数射影）。"""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1, size=(4,))
    x = np.array([mean_wr, mean_tr, n_anc, 1.0])
    return {"e_bilin_rp": float(np.dot(w, x))}


def fm_poly_features(x_vec: np.ndarray) -> np.ndarray:
    """E: Factorization machines の簡易代理（2次多項式）。"""
    poly = PolynomialFeatures(degree=2, include_bias=True)
    return poly.fit_transform(x_vec.reshape(1, -1))[0]


def rbf_kernel_features(x_vec: np.ndarray, seed: int = 42) -> np.ndarray:
    """E: RBF カーネル特徴（Random Fourier）。"""
    rbf = RBFSampler(gamma=1.0, n_components=6, random_state=seed)
    return rbf.fit_transform(x_vec.reshape(1, -1))[0]


def build_feature_row(
    store: StageStatsStore,
    gb: dict[str, Any],
    sire_stats: dict[str, Any],
    stallion_ids: frozenset[str],
    fsk: str,
    stage_key_coarse: str,
    community_labels: dict[str, int],
) -> dict[str, float]:
    """1 出走行分の特徴（5.3 各項目を可能な範囲で積む）。"""
    feats: dict[str, float] = {}

    # A: raw / EB / residual / logit / winsor
    raw = store.mean_aggregates(stallion_ids, fsk, mode="raw")
    eb = store.mean_aggregates(stallion_ids, fsk, mode="eb")
    res = store.mean_aggregates(stallion_ids, fsk, mode="residual")
    feats.update({f"a_raw_{k}": v for k, v in raw.items()})
    feats.update({f"a_eb_{k}": v for k, v in eb.items()})
    feats.update({f"a_res_{k}": v for k, v in res.items()})

    wrs = []
    for sid in stallion_ids:
        d = store.get_cell(sid, fsk)
        if d:
            wrs.append(d["win_rate"])
    if wrs:
        wa = _winsorize(np.array(wrs))
        feats["a_winsor_mean_wr"] = float(np.mean(wa))
        feats["a_logit_mean_wr"] = float(np.mean(_safe_logit(np.array(wrs))))

    feats.update(store.stallion_profile_features(stallion_ids))

    # 多解像度: coarse stage_key（cond 無し）
    if stage_key_coarse:
        feats.update(store.mean_aggregates_coarse(stallion_ids, stage_key_coarse))

    # B: 因子
    axes = [a["id"] for a in sire_stats.get("axes", [])][:9]
    if not axes:
        axes = ["ts", "consistency", "dirt", "wet_turf"]
    ff = factor_features(sire_stats, stallion_ids, axes)
    feats.update(ff)
    feats.update(orthogonal_residual_to_factors(store, stallion_ids, fsk, ff, axes))

    # C: グラフ
    feats.update(graph_smooth_wr(store, gb, stallion_ids, fsk, hops=1))
    feats.update(graph_smooth_wr(store, gb, stallion_ids, fsk, hops=2))
    feats.update(ppr_scores(gb, stallion_ids))
    feats.update(shortest_path_kernel(gb, stallion_ids))
    ns = gb.get("neighbors_sire") or defaultdict(set)
    nd = gb.get("neighbors_damsire") or defaultdict(set)
    n_sire_e = sum(len(ns.get(s, set())) for s in stallion_ids)
    n_ds_e = sum(len(nd.get(s, set())) for s in stallion_ids)
    feats["c_edge_sire_deg"] = float(n_sire_e)
    feats["c_edge_damsire_deg"] = float(n_ds_e)
    labs = [float(community_labels.get(sid, -1.0)) for sid in stallion_ids]
    if labs:
        feats["c_comm_label_mean"] = float(np.mean(labs))

    # D: log-sumexp は raw に含む。世代重み近似: 深い祖先ほど薄い仮定で同じ集合に対し 0.5**2 を掛けたダミー（集合に世代情報が無いため弱い代理）
    feats["d_depth_decay_proxy"] = float(len(stallion_ids))

    # E: bilinear / FM / RBF
    mw = raw.get("m_mean_wr", 0.0)
    mt = raw.get("m_mean_tr", 0.0)
    na = raw.get("m_n_anc", 0.0)
    feats.update(bilinear_random_projection(mw, mt, na))
    try:
        fm = fm_poly_features(np.array([mw, mt, na, float(feats.get("a_logit_mean_wr", 0.0))]))
        for i, v in enumerate(fm):
            feats[f"e_fm_{i}"] = float(v)
    except Exception:
        pass
    try:
        rk = rbf_kernel_features(np.array([mw, mt, na]))
        for i, v in enumerate(rk):
            feats[f"e_rbf_{i}"] = float(v)
    except Exception:
        pass

    return feats

