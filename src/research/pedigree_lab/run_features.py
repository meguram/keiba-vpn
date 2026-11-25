"""段 1 の特徴量を開発の期間（2016〜2024）で作って保存する。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.run_features --tag v2 --gen 2 [--sub]
出力: OUT/feat_<tag>.parquet（と第 2 版は kinfo_<tag>.json: 縮小の強さ k と同じ馬の相関）。費用 0 円。
"""
from __future__ import annotations

import argparse
import json
import time

from src.research.pedigree_lab import data, features, features2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--gen", type=int, default=2)
    ap.add_argument("--sub", action="store_true", help="副の版（馬の状態の鍵）の列も作る")
    a = ap.parse_args()
    t = time.time()
    r = data.load_dev(reason=f"段 1 の特徴量 {a.tag} を作る（as-of の集計の元）", version=f"feat_{a.tag}")
    ped = data.load_ped()
    if a.gen == 1:
        X = features.build(r, ped)
    else:
        X, kinfo = features2.build(r, ped, sub=a.sub)
        (data.OUT / f"kinfo_{a.tag}.json").write_text(json.dumps(kinfo, ensure_ascii=False, indent=1))
    X["split"] = data.split_meetings(X["meeting"])
    X.to_parquet(data.OUT / f"feat_{a.tag}.parquet", index=False)
    data.finish_all(str(data.OUT / f"feat_{a.tag}.parquet"))
    print("feat", X.shape, f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
