"""tp-85: 走法マップの血統欄を『10 点満点』『距離 m』で見せるための基準を作る。

直近 2 年に走った馬（性別つき）の血統予測値の分布（0〜100 の 101 分位）と、芝・ダートの平均距離を
data/local/modeling/tp85_ped_traits/model/reference.json に保存する。
使い方: python -m src.scripts.data.tp85_ped_reference
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.pipeline.features.tp85_pedprior import MODEL_DIR, load_models
from src.scripts.data import tp85_ped_traits as t85


def main() -> None:
    d = t85.load_runs()
    surface_mean = {str(k): round(float(v), 1) for k, v in d.groupby("surface")["distance"].mean().items()}
    recent = d[d["date"] >= d["date"].max() - pd.Timedelta(days=730)]
    hs = recent.groupby("horse_id", as_index=False).agg(sex=("sex", "first"))
    ped = t85.load_ped(hs["horse_id"].tolist())
    hs = hs[hs["horse_id"].isin(ped)].reset_index(drop=True)
    print(f"基準の馬: {len(hs)} 頭 / 芝ダの平均距離: {surface_mean}")
    pop = {}
    for k, m in load_models().items():
        mean = m.predict(ped, hs)["mean"].to_numpy()
        pop[k] = [round(float(x), 5) for x in np.percentile(mean, np.linspace(0, 100, 101))]
        print(k, "5%/50%/95%:", pop[k][5], pop[k][50], pop[k][95])
    out = {"surface_mean": surface_mean, "n_horses": int(len(hs)), "pop": pop}
    (MODEL_DIR / "reference.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("保存:", MODEL_DIR / "reference.json")


if __name__ == "__main__":
    main()
