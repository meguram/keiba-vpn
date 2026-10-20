"""v5 の推論用: ALL_BEST（tp-38 の PX_cell_sd・PX_emb・PX_ridge_als）の列を、当日の出走馬に付ける。

方針（二重実装にしない）: 学習側の関数（tp38_pedigree.cell_features、tp38_ridge.fit_one / _features）を、『履歴（その日より厳密に前の確定結果）+ 当日の出走馬の行（結果なし）』に
対してそのまま呼び、当日の行だけを返す。日単位で作ってキャッシュする想定（V4Features.day_groups と同じ）。
状態（STATE_DIR = data/local/inference/state/px_v5/）:
  anc.parquet / horses.parquet / ancestors.parquet … 学習時の祖先表（馬・祖先のコード。約 30 MB）。新しい馬は血統 JSON から足す（コードは末尾に追加）
  svd.npz … PX_emb の SVD の成分（32 × 祖先数）と IDF（約 5 MB）。出走馬の血統の TF-IDF を射影するだけ（静的）
  ridge_<T0>.npz … 半年ごとの作り直し日 T0 より前の走で学習したリッジ（fold ごとの φ・δ_G・δ_C・条件のコード表、約 1 MB）。``build_state`` / ``fit_ridge_window`` で作る
as-of: セル統計は日付が厳密に前の走だけ（tp38_pedigree.lookup_prior）、リッジは T0 より前の走だけ。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features import factory_ped as fp
from src.pipeline.features import tp38_pedigree as P
from src.pipeline.features import tp38_ridge as RG

logger = logging.getLogger("pipeline.features.serving_v5_px")

ROOT = Path(__file__).resolve().parents[3]
STATE_DIR = Path(os.environ.get("KEIBA_INFERENCE_DIR") or ROOT / "data/local/inference") / "state" / "px_v5"
KEY = ["race_id", "horse_id"]
EMB_DIM = 16
RUN_COLS = ["date", "race_id", "horse_id", "venue_code", "surface", "distance", "track_condition", "field_size", "finish_position"]


def add_rel(F: pd.DataFrame, race: pd.Series, cols: list[str]) -> pd.DataFrame:
    """列の『同じレースの平均との差』を足す（tp38_eval.add_rel と同じ）。"""
    g = F[cols].groupby(race.to_numpy())
    rel = F[cols] - g.transform("mean")
    rel.columns = [f"{c}_rel" for c in cols]
    return pd.concat([F, rel], axis=1)


def window_start(day: pd.Timestamp) -> str:
    """作り直し日 T0 = その日を含む半年の初日（1/1 または 7/1。tp38_ridge_run.windows と同じ 6MS の刻み）。"""
    d = pd.Timestamp(day)
    return f"{d.year}-{'01' if d.month <= 6 else '07'}-01"


def window_end(t0: str) -> str:
    d = pd.Timestamp(t0)
    return f"{d.year + 1}-01-01" if d.month == 7 else f"{d.year}-07-01"


# ---------------------------------------------------------------- 状態の作成（半年ごと・オフライン）
def build_state(static_dir: str | Path, out_dir: str | Path = STATE_DIR) -> dict:
    """tp38 の静的な表（anc / horses / ancestors）を状態にコピーし、PX_emb の SVD の成分を作る（``emb.npy`` と一致することを確かめる）。"""
    from sklearn.decomposition import TruncatedSVD

    sd, od = Path(static_dir), Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    anc = pd.read_parquet(sd / "anc.parquet")
    horses = pd.read_parquet(sd / "horses.parquet")
    uniq = pd.read_parquet(sd / "ancestors.parquet")
    anc.to_parquet(od / "anc.parquet", index=False)
    horses.to_parquet(od / "horses.parquet", index=False)
    uniq.to_parquet(od / "ancestors.parquet", index=False)
    X, idf = P.horse_vectors(anc, len(horses), len(uniq))
    svd = TruncatedSVD(n_components=32, random_state=0)
    E = svd.fit_transform(X)
    ref = np.load(sd / "emb.npy") if (sd / "emb.npy").exists() else None
    err = float(np.abs(E - ref).max()) if ref is not None else float("nan")
    np.savez_compressed(od / "svd.npz", comp=svd.components_.astype(np.float32), idf=idf.astype(np.float64))
    info = {"horses": int(len(horses)), "ancestors": int(len(uniq)), "emb_max_abs_err_vs_tp38": err}
    (od / "state_meta.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    return info


def fit_ridge_window(state: "PXState", hist: pd.DataFrame, t0: str, out_dir: str | Path = STATE_DIR) -> Path:
    """T0 より前の走（学習時の馬だけ）でリッジを fold ごとに学習して保存する（tp38_ridge.ridge_features の学習の部分と同じ手順）。"""
    spec = RG.Spec
    h = hist[pd.to_datetime(hist["date"]) < pd.Timestamp(t0)][RUN_COLS].reset_index(drop=True)
    R = P.Runs(h, state.horse_index)
    n_h, n_a = len(state.horse_index), len(state.uniq)
    ga, ca = RG.universes(state.anc, spec.min_desc_g, spec.min_desc_c)
    PAg = RG.pos_ancestor_matrix(state.anc, n_h, ga, n_a)
    PAc = RG.pos_ancestor_matrix(state.anc, n_h, ca, n_a)
    kcodes = R.cond_codes["sd"]
    levels = np.array([str(x) for x in pd.factorize(P.cond_keys(h)["sd"])[1]], dtype="U32")
    n_k = int(kcodes.max()) + 1
    fold_of_h = state.fold_static
    tr_base = (R.h >= 0) & ~np.isnan(R.y)
    arrs: dict[str, np.ndarray] = {"levels": levels, "ga": ga, "ca": ca}
    for f in range(spec.n_fold):
        tr = np.flatnonzero(tr_base & (fold_of_h[np.maximum(R.h, 0)] != f))
        hist_f, scale_c = RG.fit_one(R.h[tr], kcodes[tr], R.y[tr], PAg, PAc, len(ga), len(ca), n_k, spec)
        phi, th_g, th_c = hist_f[-1]
        arrs.update({f"phi{f}": phi, f"th_g{f}": th_g, f"th_c{f}": th_c, f"scale_c{f}": np.array([scale_c]), f"n_k{f}": np.array([n_k]), f"n_train{f}": np.array([len(tr)])})
    p = Path(out_dir) / f"ridge_{t0}.npz"
    np.savez_compressed(p, **arrs)
    return p


# ---------------------------------------------------------------- 状態の読み込み・当日の列
class Ext:
    """学習時の祖先表に、新しい馬（血統 JSON から）を足したもの。馬・祖先のコードは学習時のまま、新しいものは末尾に足す。"""

    def __init__(self, anc: pd.DataFrame, horse_index: pd.Index, uniq: np.ndarray, n_static_h: int, n_static_a: int):
        self.anc, self.horse_index, self.uniq, self.n_static_h, self.n_static_a = anc, horse_index, uniq, n_static_h, n_static_a


class PXState:
    def __init__(self, d: str | Path = STATE_DIR):
        d = Path(d)
        self.dir = d
        self.anc = pd.read_parquet(d / "anc.parquet")
        self.horse_index = pd.Index(pd.read_parquet(d / "horses.parquet")["horse_id"].astype(str))
        self.uniq = pd.read_parquet(d / "ancestors.parquet")["anc_id"].astype(str).to_numpy()
        z = np.load(d / "svd.npz")
        self.comp, self.idf = z["comp"], z["idf"]
        rng = np.random.default_rng(0)
        self.fold_static = rng.integers(0, RG.Spec.n_fold, size=len(self.horse_index))        # 学習時と同じ馬の fold（馬のコード順）
        self._anc_index = pd.Index(self.uniq)

    def extend(self, horse_ids) -> Ext:
        new = [h for h in dict.fromkeys(map(str, horse_ids)) if h not in self.horse_index]
        if not new:
            return Ext(self.anc, self.horse_index, self.uniq, len(self.horse_index), len(self.uniq))
        rows = []
        for h in new:
            for d in fp.PED_DIRS:
                f = d / h[:4] / f"{h}.json"
                if not f.exists():
                    continue
                try:
                    js = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    continue
                anc_rows = [(h, int(a["generation"]), int(a["position"]), str(a.get("horse_id") or "")) for a in (js.get("ancestors") or []) if a.get("generation") and int(a["generation"]) <= P.GEN_MAX and a.get("horse_id")]
                if anc_rows:
                    rows += anc_rows
                    break
        hi = self.horse_index
        if not rows:
            return Ext(self.anc, hi, self.uniq, len(hi), len(self.uniq))
        A = pd.DataFrame(rows, columns=["horse_id", "gen", "pos", "anc_id"])
        got = sorted(A["horse_id"].unique())
        hi = self.horse_index.append(pd.Index(got))
        uniq = list(self.uniq)
        pos = {a: i for i, a in enumerate(self.uniq)}
        codes = []
        for a in A["anc_id"]:
            if a not in pos:
                pos[a] = len(uniq)
                uniq.append(a)
            codes.append(pos[a])
        A["h"] = hi.get_indexer(A["horse_id"]).astype(np.int32)
        A["a"] = np.array(codes, dtype=np.int32)
        A["pcls"] = P.pos_class(A["gen"], A["pos"])
        A["side"] = np.right_shift(A["pos"].to_numpy(), (A["gen"] - 1).to_numpy()).astype(np.int8)
        A["w"] = 0.5 ** A["gen"].astype(float)
        A["male"] = ((A["pos"] & 1) == 0)
        new_rows = A[["h", "a", "gen", "pos", "pcls", "side", "w", "male"]].astype(self.anc.dtypes.to_dict())
        return Ext(pd.concat([self.anc, new_rows], ignore_index=True), hi, np.array(uniq), len(self.horse_index), len(self.uniq))

    def embed(self, ext: Ext, h: np.ndarray) -> pd.DataFrame:
        """走の馬のコード h（-1 は欠損）→ pxe_0..15。学習時の IDF・成分で血統の TF-IDF を射影する（学習側は emb.npy の先頭 16 列）。"""
        import scipy.sparse as sp

        n_a = len(ext.uniq)
        idf = np.concatenate([self.idf, np.full(n_a - len(self.idf), np.log((1.0 + self.n_h_train) / 1.0) + 1e-6)]) if n_a > len(self.idf) else self.idf
        comp = np.concatenate([self.comp, np.zeros((self.comp.shape[0], n_a - self.comp.shape[1]), dtype=self.comp.dtype)], axis=1) if n_a > self.comp.shape[1] else self.comp
        tf = ext.anc.groupby(["h", "a"], sort=False)["w"].sum().reset_index()
        uh = np.unique(h[h >= 0])
        tf = tf[tf["h"].isin(uh)]
        X = sp.csr_matrix((tf["w"].to_numpy(), (tf["h"].to_numpy(), tf["a"].to_numpy())), shape=(len(ext.horse_index), n_a)).multiply(idf).tocsr()
        nrm = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        X = sp.diags(1.0 / np.maximum(nrm, 1e-12)) @ X
        E = (X @ comp.T)[:, :EMB_DIM]
        out = np.asarray(E)[np.where(h >= 0, h, 0)].astype(np.float32)
        out[h < 0] = np.nan
        return pd.DataFrame(out, columns=[f"pxe_{i}" for i in range(EMB_DIM)])

    @property
    def n_h_train(self) -> int:
        return len(self.horse_index)

    def ridge(self, ext: Ext, h: np.ndarray, conds: pd.Series, t0: str) -> pd.DataFrame:
        """作り直し日 t0 のリッジ（fold ごと）で、走（馬のコード h・条件の文字列）の z 系の列を作る。"""
        z = np.load(self.dir / f"ridge_{t0}.npz", allow_pickle=False)
        levels = pd.Index(z["levels"].astype(str))
        kr = levels.get_indexer(conds.astype(str)).astype(np.int32)
        ga, ca = z["ga"], z["ca"]
        PAg = RG.pos_ancestor_matrix(ext.anc, len(ext.horse_index), ga, len(ext.uniq))
        PAc = RG.pos_ancestor_matrix(ext.anc, len(ext.horse_index), ca, len(ext.uniq))
        fold = np.concatenate([self.fold_static, np.array([int(str(x)[-6:]) % RG.Spec.n_fold for x in ext.horse_index[ext.n_static_h:]], dtype=np.int64)])
        cols = ["z", "zg", "zc", "z_s0", "z_s1", "z_g12", "z_g3p", "z_tm"]
        out = pd.DataFrame(np.nan, index=np.arange(len(h)), columns=cols)
        for f in range(RG.Spec.n_fold):
            idx = np.flatnonzero((h >= 0) & (fold[np.maximum(h, 0)] == f))
            if len(idx) == 0:
                continue
            F, _ = RG._features(h[idx], kr[idx], PAg, PAc, z[f"th_g{f}"], z[f"th_c{f}"], len(ca), float(z[f"scale_c{f}"][0]), z[f"phi{f}"])
            out.loc[idx, F.columns] = F.to_numpy()
        return out.astype("float32")


class PXFeatures:
    """V4History の field_hist（日付・race_id・horse_id・venue_code・surface・distance・track_condition・field_size・finish_position）を履歴にして、当日の ALL_BEST の列を作る。"""

    def __init__(self, hist_runs: pd.DataFrame, state_dir: str | Path = STATE_DIR):
        self.state = PXState(state_dir)
        self.hist = hist_runs[RUN_COLS].copy()
        self.hist["date"] = pd.to_datetime(self.hist["date"])
        self.hist["race_id"], self.hist["horse_id"] = self.hist["race_id"].astype(str), self.hist["horse_id"].astype(str)

    def day(self, card: pd.DataFrame) -> pd.DataFrame:
        """card: その日の出馬表（race_id・horse_id・date・venue_code・surface・distance・track_condition・field_size）。戻り値: race_id・horse_id + 列。"""
        c = card[[x for x in RUN_COLS if x != "finish_position" and x in card.columns]].drop_duplicates(KEY).copy()
        c["race_id"], c["horse_id"], c["date"] = c["race_id"].astype(str), c["horse_id"].astype(str), pd.to_datetime(c["date"])
        c["finish_position"] = np.nan
        day = c["date"].min()
        H = self.hist[self.hist["date"] < day]
        B = pd.concat([H, c[RUN_COLS]], ignore_index=True)
        ext = self.state.extend(B["horse_id"].iloc[len(H):])
        R = P.Runs(B, ext.horse_index)
        Fc = P.cell_features(R, ext.anc, "sd", "pxc_sd").iloc[len(H):].reset_index(drop=True)
        race = c["race_id"].reset_index(drop=True)
        Fc = add_rel(Fc, race, [x for x in Fc.columns if x.endswith(("_w", "_r"))])
        h_t = R.h[len(H):]
        E = self.state.embed(ext, h_t)
        t0 = window_start(day)
        if not (self.state.dir / f"ridge_{t0}.npz").exists():          # その半年のリッジが無ければ作る（1〜2 分。通常は半年ごとの更新ジョブで作っておく）
            logger.info("リッジを学習する: T0=%s", t0)
            fit_ridge_window(self.state, self.hist, t0, self.state.dir)
        conds = P.cond_keys(B.iloc[len(H):].reset_index(drop=True))["sd"]
        Fr = add_rel(self.state.ridge(ext, h_t, conds, t0), race, ["z", "zc"])
        return pd.concat([c[KEY].reset_index(drop=True), Fc, E, Fr], axis=1)
