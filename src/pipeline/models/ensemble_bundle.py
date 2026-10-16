"""最終モデル（アンサンブル v3 など）の学習・保存・推論。

学習時の `zoo.fit_predict` は学習と予測を一度に行うので、そのままでは推論用に保存できない。ここでは、同じ関数でメンバーを学習し、
学習済みのモデル（LightGBM のブースター、XGBoost、NN の重みと前処理）を seed ごとに保存して、特徴量の表から予測できるようにする。
予測の計算は `zoo_runner.run_oof` と同じ（seed ごとに合計をそろえた確率の平均。softmax 型は勝率の平均から割引 Harville）。
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.pipeline.models import ensemble as en
from src.pipeline.models import win_models as wm
from src.pipeline.models import zoo

TOT = {"win": 1.0, "top2": 2.0, "top3": 3.0}


@dataclass
class MemberSpec:
    name: str
    recipe: str                      # lgb_binary / lgb_softmax / xgb_binary / linear / nn
    params: dict = field(default_factory=dict)
    feature_cols: list[str] = field(default_factory=list)


def _json_params(p: dict) -> dict:
    out = {}
    for k, v in p.items():
        if k == "weight":
            continue
        out[k] = list(v) if isinstance(v, tuple) else v
    return out


class EnsembleBundle:
    def __init__(self, members: list[MemberSpec], spec: dict | None = None, gammas: dict[str, tuple[float, float]] | None = None):
        self.members = members
        self.spec = spec or {}
        self.gammas = gammas or {}
        self.cat_levels: dict[str, list[str]] = {}
        self.models: dict[str, list[dict]] = {}      # member 名 → seed ごとのモデル

    # ------------------------------------------------------------ 学習
    def fit(self, X: pd.DataFrame, y: dict, race: np.ndarray, threads: int, seeds: list[int]) -> "EnsembleBundle":
        for c in X.columns:
            if str(X[c].dtype) == "category":
                self.cat_levels[c] = [str(v) for v in X[c].cat.categories]
        for m in self.members:
            Xm = X[m.feature_cols]
            self.models[m.name] = []
            for s in seeds:
                mo: dict = {}
                zoo.fit_predict(m.recipe, Xm, y, race, Xm.iloc[:1], race[:1], threads, seed=s, params=dict(m.params), model_out=mo)
                self.models[m.name].append(mo)
        return self

    # ------------------------------------------------------------ 推論
    def _cast(self, X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        Xm = X[cols].copy()
        for c in cols:
            if c in self.cat_levels:
                s = Xm[c].astype(str).where(Xm[c].notna())
                if "(なし)" in self.cat_levels[c]:       # 学習時（ablation.load_table）は欠損を「(なし)」にしている
                    s = s.fillna("(なし)")
                Xm[c] = pd.Categorical(s, categories=self.cat_levels[c])
        return Xm

    def predict_member(self, m: MemberSpec, X: pd.DataFrame, race: np.ndarray) -> dict[str, np.ndarray]:
        Xm = self._cast(X, m.feature_cols)
        seeds = self.models[m.name]
        n = len(seeds)
        if m.recipe == "lgb_softmax":
            win = np.zeros(len(Xm))
            for mo in seeds:
                score = mo["boosters"]["win"].predict(Xm, raw_score=True)
                win += wm.normalize_to_total(race, wm.softmax_by_race(race, score), 1.0)
            win /= n
            g2, g3 = self.gammas.get(m.name, (1.0, 1.0))
            p2, p3 = zoo.harville_topk(race, win, g2, g3)
            return {"win": win, "top2": wm.normalize_to_total(race, p2, 2.0), "top3": wm.normalize_to_total(race, p3, 3.0)}
        acc = {t: np.zeros(len(Xm)) for t in TOT}
        for mo in seeds:
            if m.recipe == "lgb_binary":
                out = {t: mo["boosters"][t].predict(Xm, raw_score=False) for t in TOT}
            elif m.recipe == "xgb_binary":
                out = {t: mo["xgb"][t].predict_proba(Xm)[:, 1] for t in TOT}
            elif m.recipe in ("nn", "linear"):
                o = zoo.nn_predict(mo, Xm, race)
                out = {"win": wm.softmax_by_race(race, o["win_score"]), "top2": zoo.sigmoid(o["top2_logit"]), "top3": zoo.sigmoid(o["top3_logit"])}
            else:
                raise ValueError(m.recipe)
            fp = zoo.finalize(race, out)
            for t in TOT:
                acc[t] += fp[t]
        return {t: acc[t] / n for t in TOT}

    def predict(self, X: pd.DataFrame, race: np.ndarray) -> pd.DataFrame:
        """メンバーの確率を spec の方法（重み付き logit・前進選択）で合成し、レース内の合計をそろえる。"""
        race = np.asarray(race)
        per = {m.name: self.predict_member(m, X, race) for m in self.members}
        out = {}
        for t, total in TOT.items():
            P = np.stack([per[m.name][t] for m in self.members], axis=1)
            out[f"p_{t}"] = wm.normalize_to_total(race, en.apply_spec(self.spec["targets"][t], P), total)
        return pd.DataFrame(out, index=X.index)

    # ------------------------------------------------------------ 保存・読み込み
    def save(self, d: Path) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        meta = {"members": [{"name": m.name, "recipe": m.recipe, "params": _json_params(m.params), "feature_cols": m.feature_cols, "n_seeds": len(self.models.get(m.name, []))} for m in self.members],
                "spec": self.spec, "gammas": {k: list(v) for k, v in self.gammas.items()}, "cat_levels": self.cat_levels}
        (d / "bundle.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        for m in self.members:
            for i, mo in enumerate(self.models[m.name]):
                sd = d / m.name / f"seed{i}"
                sd.mkdir(parents=True, exist_ok=True)
                if "boosters" in mo:
                    for t, b in mo["boosters"].items():
                        b.save_model(str(sd / f"lgb_{t}.txt"))
                if "xgb" in mo:
                    for t, b in mo["xgb"].items():
                        b.save_model(str(sd / f"xgb_{t}.json"))
                if "state" in mo:
                    import torch

                    torch.save({"arch": mo["arch"], "state": mo["state"]}, sd / "nn.pt")
                    with open(sd / "prep.pkl", "wb") as f:
                        pickle.dump(mo["prep"], f)

    @classmethod
    def load(cls, d: Path) -> "EnsembleBundle":
        import lightgbm as lgb

        d = Path(d)
        meta = json.loads((d / "bundle.json").read_text(encoding="utf-8"))
        members = [MemberSpec(m["name"], m["recipe"], {k: (tuple(v) if k == "hidden" else v) for k, v in m["params"].items()}, m["feature_cols"]) for m in meta["members"]]
        b = cls(members, meta["spec"], {k: tuple(v) for k, v in meta["gammas"].items()})
        b.cat_levels = meta["cat_levels"]
        for mm in meta["members"]:
            b.models[mm["name"]] = []
            for i in range(mm["n_seeds"]):
                sd = d / mm["name"] / f"seed{i}"
                mo: dict = {}
                if (sd / "lgb_win.txt").exists():
                    mo["boosters"] = {t: lgb.Booster(model_file=str(sd / f"lgb_{t}.txt")) for t in ("win", "top2", "top3") if (sd / f"lgb_{t}.txt").exists()}
                if (sd / "xgb_win.json").exists():
                    import xgboost as xgb

                    mo["xgb"] = {}
                    for t in ("win", "top2", "top3"):
                        x = xgb.XGBClassifier(enable_categorical=True)
                        x.load_model(str(sd / f"xgb_{t}.json"))
                        mo["xgb"][t] = x
                if (sd / "nn.pt").exists():
                    import torch

                    z = torch.load(sd / "nn.pt", map_location="cpu", weights_only=False)
                    mo.update({"arch": z["arch"], "state": z["state"]})
                    with open(sd / "prep.pkl", "rb") as f:
                        mo["prep"] = pickle.load(f)
                b.models[mm["name"]].append(mo)
        return b
