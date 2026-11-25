"""段 2 の試し: 特徴量の版 × 目的 × 分け方 で当てて、対数損失を JSONL に追記する（試した数の記録を兼ねる）。

    nice -n 19 python -m src.research.pedigree_lab.run_model --tag v1 --targets win,top3 --splits random,forward
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from src.research.pedigree_lab import data, model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--targets", default="win,top3")
    ap.add_argument("--splits", default="random,forward")
    ap.add_argument("--drop", default="", help="特徴量の名前の正規表現（, 区切り）で外す")
    ap.add_argument("--note", default="")
    ap.add_argument("--save-pred", action="store_true")
    a = ap.parse_args()
    X = pd.read_parquet(data.OUT / f"feat_{a.tag}.parquet")
    drop = tuple(d for d in a.drop.split(",") if d)
    for tg in a.targets.split(","):
        for sp in a.splits.split(","):
            t = time.time()
            o = model.fit_eval(X, tg, sp, drop=drop)
            r = o["res"] | {"tag": a.tag, "drop": a.drop, "note": a.note, "sec": round(time.time() - t)}
            with open(data.OUT / "trials.jsonl", "a") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print("top imp:", ", ".join(f"{k}={v:.0f}" for k, v in o["imp"].head(15).items()))
            if a.save_pred:
                pd.DataFrame({"race_id": o["X"]["race_id"], "horse_id": o["X"]["horse_id"], "p": o["pred"],
                              "split": o["split"]}).to_parquet(data.OUT / f"pred_{a.tag}_{tg}_{sp}.parquet", index=False)


if __name__ == "__main__":
    main()
