"""v5 の LOYO アンサンブル（tp27_v5/ens/<名前>.parquet）を、v4 の LOYO アンサンブル（tp27/ens/<基準>.parquet）と同じ行で対比較する。

python -m src.scripts.data.v5_ens_compare --v5 v5 --base v4_e2 --out <json>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

V = Path(__file__).resolve().parents[3] / "data/local/modeling/training_pipeline_v4"
os.environ.setdefault("TP27_OUT", str(V / "tp27_v5"))
for _k, _v in {"TP_BASE_DIR": V, "TP_WORK": V, "TP25C_OUT": V / "tp25c", "TP25E_OUT": V / "tp25e", "TP27_FAC_FIELD": V / "factory_field"}.items():
    os.environ.setdefault(_k, str(_v))

from src.pipeline.models import feature_bakeoff as fb  # noqa: E402
from src.pipeline.models import tp05_data  # noqa: E402

KEY = ["race_id", "horse_id"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5", default="v5")
    ap.add_argument("--base", default="v4_e2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = tp05_data.load_dataset(V)
    fin_map = D["meta"].set_index(KEY)["finish_position"]
    new = pd.read_parquet(V / "tp27_v5/ens" / f"{a.v5}.parquet")
    old = pd.read_parquet(V / "tp27/ens" / f"{a.base}.parquet")
    for d in (new, old):
        d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    cols = KEY + ["p_win", "p_top2", "p_top3"]
    m = new[KEY].merge(old[KEY], on=KEY, how="inner")
    out = {"rows": int(len(m))}
    out["overall"] = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in fb.compare_oof(old[cols].merge(m, on=KEY), new[cols].merge(m, on=KEY), fin_map, f"{a.v5} vs {a.base}").items()}
    yr = pd.to_datetime(D["meta"].set_index(KEY)["date"]).dt.year
    for y in (2022, 2023, 2024):
        ids = m.merge(yr.rename("y").reset_index(), on=KEY)
        mm = ids[ids["y"] == y][KEY]
        if len(mm) > 1000:
            c = fb.compare_oof(old[cols].merge(mm, on=KEY), new[cols].merge(mm, on=KEY), fin_map, f"{y}")
            out[str(y)] = {k: float(v) for k, v in c.items() if isinstance(v, (float, np.floating))}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: (v if k != "overall" else {kk: round(vv, 5) if isinstance(vv, float) else vv for kk, vv in v.items()}) for k, v in out.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
