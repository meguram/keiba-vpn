"""tp-85: 血統（父・母父・母母父・母母母父）から、馬の観点（芝ダ・距離・走法軸など）の『本来の傾向』と 80% 区間を出す。

走数が増えても持ち続ける値（馬自身の実測とは別に持ち、『血統の割に…』のずれを見るための基準）。
モデルは tp85_ped_traits の階層混合モデル（系統 → 種牡馬の効果を足し合わせる）。観点ごとに説明量が最大の単位（位置の組み合わせ・性別の扱い）を採り、
検証（新しい馬）で求めた傾き k で『平均からのずれ』を補正する（端の予測の大きさが当たるように）。
区間は『その馬の真の値』の 80%（測定誤差を含めない）。血統が分からない馬は平均と全体の幅を返す（ped_n = 0）。
市場オッズ・人気は使わない。キーは horse_id。
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.scripts.data import tp85_ped_traits as t85

MODEL_DIR = t85.ROOT / "data/local/modeling/tp85_ped_traits/model"


class PedPrior:
    """meta: trait / positions / mode / sexmode / slope / unit。st: fit_variant の状態（u, C, col, sig, bid, s, m, tau, sex_mean）。"""

    def __init__(self, meta: dict, st: dict):
        self.meta = meta
        self.st = st
        self._nodes = None

    @classmethod
    def load(cls, trait: str, model_dir: Path = MODEL_DIR) -> "PedPrior":
        with open(model_dir / f"{trait}.pkl", "rb") as f:
            d = pickle.load(f)
        d["st"]["C"] = d["st"]["C"].astype(np.float64)
        return cls(d["meta"], d["st"])

    def save(self, model_dir: Path = MODEL_DIR) -> Path:
        model_dir.mkdir(parents=True, exist_ok=True)
        st = {k: v for k, v in self.st.items() if k != "te"}
        st["C"] = st["C"].astype(np.float32)
        p = model_dir / f"{self.meta['trait']}.pkl"
        with open(p, "wb") as f:
            pickle.dump({"meta": self.meta, "st": st}, f)
        return p

    def _nodes_fn(self):
        if self._nodes is None:
            self._nodes = t85.make_nodes(json.loads(t85.ORIGIN_PATH.read_text(encoding="utf-8")))
        return self._nodes

    def predict(self, ped: dict[str, dict[int, str]], horses: pd.DataFrame) -> pd.DataFrame:
        """ped: {horse_id: {1: 父, 2: 母父, 3: 母母父, 4: 母母母父}}（load_ped の形）。horses: horse_id と sex（牡・牝・セ）。
        返す列: horse_id, mean（自然な単位）, lo80, hi80, sd（真の値の標準偏差）, ped_n（使えた位置の数）。"""
        mt, st = self.meta, self.st
        hids = horses["horse_id"].astype(str).tolist()
        sexes = horses["sex"].replace({"セ": "牡"}).fillna("牡").tolist()
        n = len(hids)
        e = t85.entries(ped, hids, self._nodes_fn(), mt["positions"], mt["mode"], sexes, mt["sexmode"] == "inter")
        col, u, C, sig, bid, tau, s, m = st["col"], st["u"], st["C"], st["sig"], st["bid"], st["tau"], st["s"], st["m"]
        e = e.assign(col=e["key"].map(col))
        seen = e[e["col"].notna()]
        Z = sp.coo_matrix((seen["coef"], (seen["row"], seen["col"].astype(int))), shape=(n, len(col) + 1)).tocsr()
        Z = Z + sp.csr_matrix((np.ones(n), (np.arange(n), np.zeros(n, int))), shape=Z.shape)
        if mt["sexmode"] != "none":
            shift = (np.array([st["sex_mean"].get(x, m) for x in sexes]) - m) / s
        else:
            shift = np.zeros(n)
        mu = Z @ u + shift
        var_u = t85.rowquad(Z, C)
        unseen = e[e["col"].isna()]
        un = np.zeros(n)
        if len(unseen):
            add = (unseen["coef"] ** 2) * unseen["blk"].map(lambda b: sig[bid[b]] if b in bid else float(sig.mean()))
            un = np.bincount(unseen["row"], add, minlength=n)
        have = e.drop_duplicates(["row", "pos"]).groupby("row")["pos"].agg(set) if len(e) else pd.Series(dtype=object)
        miss = np.zeros(n)
        for i in range(n):
            got = have.get(i, set())
            miss[i] = sum(st["pos_prior_var"].get(k, 0.0) for k in mt["positions"] if k not in got)   # 分からない位置は事前の分散を足す（区間が狭くならないように）
        sd = np.sqrt(var_u + un + miss + tau)
        mean = m + s * mt["slope"] * mu
        half = t85.Z80 * sd * s
        ped_n = np.bincount(e[e["pos"].notna()].drop_duplicates(["row", "pos"])["row"], minlength=n) if len(e) else np.zeros(n, int)
        return pd.DataFrame({"horse_id": hids, "mean": mean, "lo80": mean - half, "hi80": mean + half, "sd": sd * s, "ped_n": ped_n})


TRAIT_LABEL = {**{"ax_" + k: v for k, v in t85.AXIS_LABEL.items()}, "basic": "基礎スピード", "style_q": "位置取り（0=先頭〜1=後方）",
               "dirt_share": "ダート比率", "dist_run": "距離（芝ダ平均との差 m）", "dist_good": "好走する距離（同）"}


class Derived:
    """血統が予測した別の観点（inputs）の期待値から線形に出す観点（例: ギア・TS・Hペース耐性 → TS持続力）。
    entry: inputs / beta（切片, 係数…）/ tau_res（真の値の残差分散）/ tau_pop（真の値の母集団分散）。血統が全く分からない馬は母集団の幅を返す。"""

    def __init__(self, entry: dict, inputs: list["PedPrior"]):
        self.meta = entry
        self.inputs = inputs

    def predict(self, ped: dict[str, dict[int, str]], horses: pd.DataFrame) -> pd.DataFrame:
        parts = [m.predict(ped, horses) for m in self.inputs]
        b = np.asarray(self.meta["beta"], dtype=float)
        mean = b[0] + np.column_stack([p["mean"].to_numpy() for p in parts]) @ b[1:]
        ped_n = np.minimum.reduce([p["ped_n"].to_numpy() for p in parts])
        sd = np.where(ped_n == 0, np.sqrt(self.meta["tau_pop"]), np.sqrt(self.meta["tau_res"]))
        half = t85.Z80 * sd
        return pd.DataFrame({"horse_id": parts[0]["horse_id"], "mean": mean, "lo80": mean - half, "hi80": mean + half, "sd": sd, "ped_n": ped_n})


def load_models(model_dir: Path = MODEL_DIR) -> dict[str, "PedPrior | Derived"]:
    """manifest で採用された観点のモデルを読む（派生の観点は入力の観点が揃っているときだけ）。"""
    man = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    out: dict = {k: PedPrior.load(k, model_dir) for k, v in man.items() if v.get("adopted") and v.get("kind") != "derived"}
    for k, v in man.items():
        if v.get("adopted") and v.get("kind") == "derived" and all(i in out for i in v["inputs"]):
            out[k] = Derived(v, [out[i] for i in v["inputs"]])
    return out


def predict_parents(sex: str, sire_id: str, damsire_id: str | None, models: dict | None = None) -> pd.DataFrame:
    """性別・父・母父（horse_id）から、採用された全観点の期待値と 80% 区間を返す。"""
    models = models or load_models()
    ped = {"x": {1: str(sire_id), **({2: str(damsire_id)} if damsire_id else {})}}
    h = pd.DataFrame({"horse_id": ["x"], "sex": [sex]})
    rows = []
    for name, m in models.items():
        r = m.predict(ped, h).iloc[0]
        rows.append({"trait": name, "label": TRAIT_LABEL.get(name, name), "mean": r["mean"], "lo80": r["lo80"], "hi80": r["hi80"], "ped_n": int(r["ped_n"])})
    return pd.DataFrame(rows)
