"""推論側の ALL_BEST（serving_v5_px）が作る列を、学習側の群（tp38_out/groups の PX_cell_sd・PX_emb・PX_ridge_als）と過去の日で比べる。

KEIBA_INFERENCE_DIR=/tmp/v5_state_test python -m src.scripts.data.verify_v5_serving_px --days 20250614 20240921 --out /tmp/verify_v5_px.json
手順: (1) tp38 の静的な表から状態を作る（状態の置き場は KEIBA_INFERENCE_DIR/state/px_v5。本番の状態は触らない）、(2) 日を含む半年の作り直し日のリッジを学習、
(3) 学習表の行から『結果なしの出馬表』を作って serving の列を作り、群の値と比べる。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features import serving_v5_px as sv

ROOT = Path(__file__).resolve().parents[3]
TRAIN = ROOT / "data/local/modeling/training_pipeline_v4/features/train_table.parquet"
TP38 = Path(os.environ.get("V5_TP38_OUT") or "/home/jovyan/work/tp38_out")
KEY = sv.KEY


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-build", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    cols = sv.RUN_COLS
    tr = pd.read_parquet(TRAIN, columns=cols)
    tr["race_id"], tr["horse_id"] = tr["race_id"].astype(str), tr["horse_id"].astype(str)
    tr["date"] = pd.to_datetime(tr["date"])
    if not (sv.STATE_DIR / "svd.npz").exists() or not a.skip_build:
        info = sv.build_state(TP38 / "static", sv.STATE_DIR)
        print("状態:", json.dumps(info), f"{time.time() - t0:.0f}s", flush=True)
    st = sv.PXState(sv.STATE_DIR)
    groups = {g: pd.read_parquet(TP38 / "groups" / f"{g}.parquet") for g in ("PX_cell_sd", "PX_emb", "PX_ridge_als")}
    rep: dict = {}
    for day in a.days:
        d = pd.Timestamp(f"{day[:4]}-{day[4:6]}-{day[6:]}")
        w0 = sv.window_start(d)
        if not (sv.STATE_DIR / f"ridge_{w0}.npz").exists():
            p = sv.fit_ridge_window(st, tr, w0, sv.STATE_DIR)
            print("リッジ:", p.name, f"{time.time() - t0:.0f}s", flush=True)
        card = tr[tr["date"] == d].copy()
        px = sv.PXFeatures(tr, sv.STATE_DIR)
        t1 = time.time()
        got = px.day(card)
        sec = round(time.time() - t1, 1)
        res = {"rows": int(len(got)), "sec": sec}
        for g, G in groups.items():
            G = G.copy()
            G["race_id"], G["horse_id"] = G["race_id"].astype(str), G["horse_id"].astype(str)
            m = got.merge(G.drop_duplicates(KEY), on=KEY, how="inner", suffixes=("", "__r"))
            cs = [c for c in G.columns if c not in KEY and c in got.columns and c + "__r" in m.columns]
            bad = {}
            for c in cs:
                x, y = m[c].to_numpy(dtype=float), m[c + "__r"].to_numpy(dtype=float)
                ok = (np.isnan(x) & np.isnan(y)) | np.isclose(x, y, rtol=1e-3, atol=1e-4)
                if not ok.all():
                    bad[c] = float(1 - ok.mean())
            res[g] = {"rows": int(len(m)), "cols": len(cs), "bad": len(bad), "worst": dict(sorted(bad.items(), key=lambda kv: -kv[1])[:5]),
                      "missing": sorted(c for c in G.columns if c not in KEY and c not in got.columns)[:6]}
        rep[day] = res
        print(day, json.dumps(res, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
