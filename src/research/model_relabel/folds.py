"""開催単位の無作為の 5 分割（副の分け。報告だけに使う。rules-council 1 号が条件付きで了承、11/27）。

- 単位は race_id[:8]（年・場・回）。2016〜24 年の開催だけ。seed は config/model_relabel/spec.json の meeting_split.seed。
- 並べた開催の一覧を rng.permutation で並べ替え、順に 0〜4 の折を当てる。
- 外側の折 k ごとに、学びの側の開催から同じ rng の続きで 20% を較正の側に選ぶ（calib_k = 1）。
- 着順・払戻は読まない（meta の race_id と年だけ）。一覧は docs/operations/model_relabel/meeting_folds.csv に書いて commit する。
  python -m src.research.model_relabel.folds
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "config/model_relabel/spec.json"
FOLDS_CSV = ROOT / "docs/operations/model_relabel/meeting_folds.csv"


def make_folds(meetings, seed: int, k: int = 5, calib_frac: float = 0.2) -> pd.DataFrame:
    ms = np.array(sorted(set(str(m) for m in meetings)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ms))
    fold = np.empty(len(ms), int)
    fold[perm] = np.arange(len(ms)) % k
    out = pd.DataFrame({"meeting": ms, "fold": fold})
    for j in range(k):
        tr = np.flatnonzero(fold != j)
        pick = rng.choice(tr, size=int(round(len(tr) * calib_frac)), replace=False)
        c = np.zeros(len(ms), int)
        c[pick] = 1
        out[f"calib_{j}"] = c
    return out


def main() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))["meeting_split"]
    meta = pd.read_parquet("/home/jovyan/work/retrain_out/cycles16_fix3/meta.parquet", columns=["race_id", "year"])
    meta = meta[pd.to_numeric(meta["year"]) <= 2024]
    f = make_folds(meta["race_id"].astype(str).str[:8], int(spec["seed"]), int(spec["k"]), float(spec["inner_calib_frac"]))
    FOLDS_CSV.parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(FOLDS_CSV, index=False)
    print(len(f), f["fold"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
