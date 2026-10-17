"""v4 候補の推論用: 補助タスクつき NN（tp-25c の mtl_C）の補助ヘッド予測 mtl_*（9 列）と、そのレース内相対 *_rk / *_gap（18 列）を作る。

学習時は年ごとの expanding OOF（検証年の予測は、その年より前だけで学習したモデル）。推論は 2020〜2025 の全行で学習した最終モデルを使う
（保存: ``fit_final_aux``）。入力は学習と同じ base 175 列（``tp25c_eval.base_columns``）。補助の予測は標準化した単位のまま（学習時の OOF と同じ）。
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

AUX_NAMES = ["t_pos_early", "t_pos_late", "t_l3f_dev", "t_time_dev", "t_rank_norm", "ten_rel_actual", "c1_rel_actual", "r_pace_bal", "r_rpci"]
MISSING_LEVEL = "(なし)"


def aux_columns() -> list[str]:
    names = ["mtl_" + x for x in AUX_NAMES]
    return names + [n + s for n in names for s in ("_rk", "_gap")]


def relative(df: pd.DataFrame, race_id) -> pd.DataFrame:
    """tp25e_eval.rel_cols と同じ式（レース内のパーセンタイル順位と平均との差）。"""
    names = ["mtl_" + x for x in AUX_NAMES]
    t = df[names].copy()
    t["race_id"] = np.asarray(race_id)
    g = t.groupby("race_id", sort=False)
    out = {}
    for c in names:
        out[c + "_rk"] = g[c].rank(pct=True)
        out[c + "_gap"] = t[c] - g[c].transform("mean")
    return pd.DataFrame(out, index=df.index)


class AuxModels:
    def __init__(self, models: list[dict], base_cols: list[str]):
        self.models, self.base_cols = models, base_cols

    @classmethod
    def load(cls, directory: str | Path) -> "AuxModels":
        d = Path(directory)
        meta = pickle.loads((d / "meta.pkl").read_bytes())
        models = [pickle.loads(p.read_bytes()) for p in sorted(d.glob("model_s*.pkl"))]
        if not models:
            raise FileNotFoundError(f"補助モデルがありません: {d}")
        return cls(models, meta["base_cols"])

    def _prep_input(self, X: pd.DataFrame) -> pd.DataFrame:
        miss = [c for c in self.base_cols if c not in X.columns]
        if miss:
            raise KeyError(f"補助モデルの入力列がありません: {miss[:10]}（{len(miss)} 列）")
        Xm = X[self.base_cols].copy()
        for c in self.models[0]["prep"].cat_cols:        # 学習時は欠損を「(なし)」にしている（ablation.load_table）
            Xm[c] = Xm[c].astype("string").fillna(MISSING_LEVEL).astype(str)
        return Xm

    def predict(self, X: pd.DataFrame, race_id) -> pd.DataFrame:
        """戻り値: ``aux_columns()`` の 27 列（X と同じ index）。seed 平均した補助予測 + レース内相対。"""
        from src.pipeline.models.tp25c_mtl import mtl_infer

        Xm = self._prep_input(X)
        race = np.asarray(race_id)
        acc = np.mean([mtl_infer(m, Xm, race) for m in self.models], axis=0)
        out = pd.DataFrame(acc, columns=["mtl_" + x for x in AUX_NAMES], index=X.index).astype("float32")
        return pd.concat([out, relative(out, race)], axis=1)[aux_columns()]
