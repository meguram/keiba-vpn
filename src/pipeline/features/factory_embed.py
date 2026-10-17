"""特徴量ファクトリー G4: 埋め込み（as-of の結果行列の低ランク分解）。

年 Y の行には「Y-01-01 より前の結果だけ」で作った埋め込みを付ける（年内は更新しない＝先読みなし）。2020 年は欠損。
- 文脈埋め込み: 実体（騎手・調教師・父・馬）× 文脈（競馬場×面×距離帯）の、縮約した着順スコアの行列を SVD。
- 共起埋め込み: 同じレースに出走した馬どうしの共起行列（PPMI）を SVD（馬の『強さの近さ』を相手関係から表す）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.features import factory_src as fs
from src.pipeline.features import factory_ped as fp_


def _ctx(B: pd.DataFrame) -> pd.Series:
    d = pd.cut(pd.to_numeric(B["distance"], errors="coerce"), [0, 1300, 1500, 1900, 2400, 99999], labels=False).astype("float").fillna(-1).astype(int).astype(str)
    return B["venue_code"].astype(str) + "|" + B["surface"].astype(str) + "|" + d


def _svd(M, dim: int):
    from sklearn.decomposition import TruncatedSVD

    dim = max(1, min(dim, min(M.shape) - 1))
    svd = TruncatedSVD(n_components=dim, random_state=0)
    Z = svd.fit_transform(M)
    return Z


def _align(Z: np.ndarray, ids: np.ndarray, ref: dict) -> np.ndarray:
    """年ごとに SVD の向き（符号・回転）が変わるので、前年の埋め込みに直交 Procrustes で合わせる（共通の実体で）。スケールは標準偏差でそろえる。"""
    Z = Z / (Z.std() + 1e-9)
    common = [i for i, e in enumerate(ids) if e in ref]
    if len(common) >= Z.shape[1] * 3:
        A = Z[common]
        Bm = np.array([ref[ids[i]] for i in common])
        U, _, Vt = np.linalg.svd(A.T @ Bm)
        Z = Z @ (U @ Vt)
    ref.clear()
    ref.update({e: Z[i] for i, e in enumerate(ids)})
    return Z


def entity_context_svd(B: pd.DataFrame, entity: pd.Series, dim: int, prefix: str, k: float = 5.0) -> pd.DataFrame:
    from scipy import sparse

    res = fs.res_score(B)
    year = pd.to_datetime(B["date"]).dt.year.to_numpy()
    ent = entity.astype(str).where(entity.notna() & (entity.astype(str) != ""), "__na__").to_numpy()
    ctx = _ctx(B).to_numpy()
    out = np.full((len(B), dim), np.nan, dtype="float32")
    ref: dict = {}
    for Y in sorted(set(year)):
        past = (year < Y) & np.isfinite(res) & (ent != "__na__")
        if past.sum() < 5000:
            continue
        e_ids, e_idx = np.unique(ent[past], return_inverse=True)
        c_ids, c_idx = np.unique(ctx[past], return_inverse=True)
        gm = float(res[past].mean())
        r = res[past] - gm
        S = sparse.coo_matrix((r, (e_idx, c_idx)), shape=(len(e_ids), len(c_ids))).tocsr()
        N = sparse.coo_matrix((np.ones_like(r), (e_idx, c_idx)), shape=S.shape).tocsr()
        # 縮約: (合計) / (件数 + k)
        M = S.copy()
        M.data = S.data / (N.data + k)
        Z = _align(_svd(M, dim), e_ids, ref)
        pos = {e: i for i, e in enumerate(e_ids)}
        now = np.flatnonzero(year == Y)
        idx = np.array([pos.get(e, -1) for e in ent[now]])
        ok = idx >= 0
        out[now[ok], : Z.shape[1]] = Z[idx[ok]]
    return pd.DataFrame(out, columns=[f"{prefix}{i}" for i in range(dim)], index=B.index)


def emb_jockey_trainer(B: pd.DataFrame, **_) -> pd.DataFrame:
    return pd.concat([entity_context_svd(B, B["jockey_id"], 6, "em_jk"), entity_context_svd(B, B["trainer_id"], 6, "em_tr")], axis=1)


def emb_sire(B: pd.DataFrame, **_) -> pd.DataFrame:
    A, H = fp_.build_ped_tables(B["horse_id"].unique())
    P = fp_.parent_ids(B, A)
    return pd.concat([entity_context_svd(B, P["sire"], 8, "em_sire"), entity_context_svd(B, P["dam_sire"], 6, "em_dsire")], axis=1)


def emb_horse_ctx(B: pd.DataFrame, **_) -> pd.DataFrame:
    return entity_context_svd(B, B["horse_id"], 8, "em_hc", k=3.0)


def emb_horse_cooc(B: pd.DataFrame, dim: int = 12, **_) -> pd.DataFrame:
    from scipy import sparse

    year = pd.to_datetime(B["date"]).dt.year.to_numpy()
    hid = B["horse_id"].astype(str).to_numpy()
    rid = B["race_id"].astype(str).to_numpy()
    out = np.full((len(B), dim), np.nan, dtype="float32")
    sim = np.full(len(B), np.nan, dtype="float32")
    ref: dict = {}
    for Y in sorted(set(year)):
        past = year < Y
        if past.sum() < 5000:
            continue
        h_ids, h_idx = np.unique(hid[past], return_inverse=True)
        r_ids, r_idx = np.unique(rid[past], return_inverse=True)
        R = sparse.coo_matrix((np.ones(len(h_idx), dtype="float32"), (r_idx, h_idx)), shape=(len(r_ids), len(h_ids))).tocsr()
        C = (R.T @ R).tocsr()
        C.setdiag(0)
        C.eliminate_zeros()
        tot = float(C.sum())
        rs = np.asarray(C.sum(axis=1)).ravel()
        C = C.tocoo()
        pmi = np.log((C.data * tot) / (rs[C.row] * rs[C.col] + 1e-9) + 1e-12)
        keep = pmi > 0
        P = sparse.coo_matrix((pmi[keep], (C.row[keep], C.col[keep])), shape=C.shape).tocsr()
        Z = _align(_svd(P, dim), h_ids, ref)
        pos = {h: i for i, h in enumerate(h_ids)}
        now = np.flatnonzero(year == Y)
        idx = np.array([pos.get(h, -1) for h in hid[now]])
        ok = idx >= 0
        out[now[ok], : Z.shape[1]] = Z[idx[ok]]
        # 同じレースの相手の平均ベクトルとの類似度（相手関係の近さ）
        Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        df = pd.DataFrame(out[now][:, : Z.shape[1]])
        grp = df.groupby(rid[now]).transform("mean").to_numpy()
        num = np.nansum(out[now][:, : Z.shape[1]] * grp, axis=1)
        den = np.linalg.norm(np.nan_to_num(out[now][:, : Z.shape[1]]), axis=1) * np.linalg.norm(np.nan_to_num(grp), axis=1) + 1e-9
        sim[now] = (num / den).astype("float32")
        sim[now[~ok]] = np.nan
    f = pd.DataFrame(out, columns=[f"em_hco{i}" for i in range(dim)], index=B.index)
    f["em_hco_field_sim"] = sim
    return f


GROUPS_EMBED = {"EMB_jockey_trainer": emb_jockey_trainer, "EMB_sire": emb_sire, "EMB_horse_ctx": emb_horse_ctx, "EMB_horse_cooc": emb_horse_cooc}
