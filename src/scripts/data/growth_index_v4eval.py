"""成長指数を V4 の特徴量に足したときの勝率 LogLoss の変化（tp35 と同じ第 1 段・第 2 段の手順）。ホストで実行する。

  python -m src.scripts.data.growth_index_v4eval stage1 --gi <gi_group.parquet>   # valid 2022+2023、seed 0。V4 に対する増分
  python -m src.scripts.data.growth_index_v4eval stage2                            # 通過群を valid 2024、seed 2・3 で確認

指数（gi_group.parquet）は各年『その年より前』のデータだけで学習したもの（2020〜2021 年は欠損）。V4 の基準値は tp35 のキャッシュ（同じ分割・同じ seed）を使う。
test 2025・2026 は使わない。採否の目安は既存と同じ（最小効果量 0.0002、第 1 段は Δwin<−0.00012 かつ上限<0.0001）。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from src.scripts.data import tp35_eval as T

OUT = T.V / "tp57_growth"
SETS = {
    "GI_idx": ["gi_index"],
    "GI_parts": ["gi_stage", "gi_weight", "gi_oikiri", "gi_ped"],
    "GI_all": ["gi_index", "gi_stage", "gi_weight", "gi_oikiri", "gi_ped", "gi_weight_dev"],
}


def prepare(gi_path: Path) -> None:
    g = pd.read_parquet(gi_path)
    g["race_id"], g["horse_id"] = g["race_id"].astype(str), g["horse_id"].astype(str)
    g = g.drop_duplicates(["race_id", "horse_id"])
    (OUT / "groups").mkdir(parents=True, exist_ok=True)
    for name, cols in SETS.items():
        g[["race_id", "horse_id"] + cols].to_parquet(OUT / "groups" / f"{name}.parquet", index=False)
    for stage in ("stage1", "stage2"):
        (OUT / stage).mkdir(parents=True, exist_ok=True)
        src = T.OUT / stage
        if (src / "base.parquet").exists() and not (OUT / stage / "base.parquet").exists():
            shutil.copy(src / "base.parquet", OUT / stage / "base.parquet")
            base = json.loads((src / "results.json").read_text(encoding="utf-8")).get("base")
            (OUT / stage / "results.json").write_text(json.dumps({"base": base}, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage1", "stage2"])
    ap.add_argument("--gi", default="data/analysis/pedigree/growth_weight/gi_group.parquet")
    a = ap.parse_args()
    prepare(Path(a.gi))
    T.OUT = OUT                                     # tp35 の関数は OUT を参照して群の読み込みと結果の保存をする
    if a.cmd == "stage1":
        T._stage("stage1", [2022, 2023], [0], list(SETS))
    else:
        r = json.loads((OUT / "stage1/results.json").read_text(encoding="utf-8"))
        passed = [g for g, s in r.get("groups", {}).items() if s["vs_V4"]["Δwin"] < -0.00012 and s["vs_V4"]["win_hi"] < 0.0001]
        T._log(f"第 1 段を通過: {passed}")
        (OUT / "stage1/passed.json").write_text(json.dumps(passed, ensure_ascii=False), encoding="utf-8")
        if passed:
            T._stage("stage2", [2024], [2, 3], passed)


if __name__ == "__main__":
    main()
