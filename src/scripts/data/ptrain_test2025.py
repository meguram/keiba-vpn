"""tp-77: 選んだ 1 構成を test 2025 に 1 回だけ当てる（基準 z:RB@e と同じ手順・同じ seed で、2020〜2024 で学習 → 2025 を予測）。

python -m src.scripts.data.ptrain_test2025 --set <feature_sets の構成名>
一度でも実行したら印（TEST2025_EVALUATED_tp77.json）を書き、2 回目は拒否する（選択に test 2025 を使わないため）。2026 年は使わない。市場オッズ・人気は使わない。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from src.pipeline.features import feature_sets as fs
from src.pipeline.models.time_split import yearly_expanding_folds
from src.scripts.data import tp27_eval as t
from src.scripts.data import tp40_eval as e40
from src.scripts.data import v6_eval as ev

OUT = Path(os.environ.get("PTRAIN_OUT", "/home/jovyan/work/ptrain_out"))
MARK = OUT / "TEST2025_EVALUATED_tp77.json"
KEY = ["race_id", "horse_id"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--seeds", default="0,1")
    a = ap.parse_args()
    if MARK.exists():
        raise SystemExit(f"test 2025 は既に評価済み: {MARK}")
    seeds = [int(x) for x in a.seeds.split(",")]
    name = fs.v6_eval_name(a.set)
    base_name = ev.BASELINE_NAME if hasattr(ev, "BASELINE_NAME") else "z:RB@e"
    df, base, meta, folds, lp = t.setup()
    f25 = list(yearly_expanding_folds(df, valid_years=("2025",)))
    fin_map = meta.set_index(KEY)["finish_position"]
    res = {}
    oofs = {}
    for label, nm in (("基準", base_name), ("選んだ構成", name)):
        X = ev.build_frame(ev.config_of(nm), df)
        lv, oof = e40._run_subset(nm, X, meta, f25, lp, [2025], seeds)
        oofs[label] = oof
        res[label] = {"name": nm, "cols": int(X.shape[1]), "levels": lv}
    cmp = t._cmp(oofs["基準"], oofs["選んだ構成"], fin_map, f"{a.set} vs 基準（test 2025）")
    out = {"set": a.set, "seeds": seeds, "evaluated_at": time.strftime("%F %T"), **res, "compare": cmp}
    MARK.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
