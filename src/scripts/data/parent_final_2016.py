"""親モデル本体（段 10）を 2016〜2025 年の表で学習する入口（fit_final_v6 と同じ手順・同じ CLI）。

    python -m src.scripts.data.parent_final_2016 --fit-end 2024   # 2016〜2024 で学習 → test 2025 を 1 回だけ評価
    python -m src.scripts.data.parent_final_2016 --fit-end 2025   # 2016〜2025 の全行で学習（2026 年は使わない）

- 読み先・書き先は BASE（retrain_out/base_2016）。出力 BASE/final_v6e16/bundle_v6_{fit_end}、評価 BASE/final_eval_v6e16/。
  本番の final_v6e・final_eval_v6e（test 2025 の評価済みの印つき）は読むだけで書き換えない。
- test 2025 の比較相手は fit_final_v6 の作りどおり本番の final_eval_v5（v5）。本番の v6e（final_eval_v6e）との比較は別に行う。
- アンサンブルの重みは BASE/tp27_v6e/ens/v6.json（段 9 の LOYO）。補助予測は BASE/final_aux/aux_train.parquet（aux_B と同じ。2025 年は 2024 年までで学習した値）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BASE = Path(os.environ.get("REBUILD_BASE") or "/home/jovyan/work/retrain_out/base_2016")
TAG = "e16"


def main() -> None:
    from src.scripts.data import parent_2016 as p

    p.prepare()
    for k, v in {"TP_BASE_DIR": BASE, "TP_WORK": BASE, "TP25C_OUT": BASE / "tp25c", "TP25E_OUT": BASE / "tp25e",
                 "TP27_FAC_FIELD": BASE / "factory/field", "V6_OUT": BASE / "v6", "V5_TP38_OUT": BASE / "tp38",
                 "TP27_OUT": BASE / "tp27_v6e"}.items():
        os.environ[k] = str(v)
    os.environ.setdefault("V5_GROUPS", "CY,BEST,RB")
    os.environ.pop("V6_STAGE", None)
    os.environ["V6_TAG"] = TAG
    from src.scripts.data import rebuild_groups_2016 as rg

    rg.redirect()
    from src.scripts.data import fit_final_v4 as f4
    from src.scripts.data import fit_final_v5 as f5
    from src.scripts.data import fit_final_v6 as f6
    from src.scripts.data import tp27_eval as t27
    from src.scripts.data import v5_common

    f4.V4 = BASE
    f6.OUT, f6.EVAL = BASE / f"final_v6{TAG}", BASE / f"final_eval_v6{TAG}"
    f6.TP27_V6 = f5.TP27_V5 = BASE / "tp27_v6e"
    t27.OUT = BASE / "tp27_v6e"
    os.environ["TP27_OUT"] = str(t27.OUT)
    v5_common.V = BASE
    f6.EVAL.mkdir(parents=True, exist_ok=True)
    f6.main()


if __name__ == "__main__":
    sys.exit(main())
