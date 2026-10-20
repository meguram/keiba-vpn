"""v5 のアンサンブル用に、7 メンバーの OOF（valid 2022〜24）と LOYO アンサンブルを tp27_v5/ に作る。

python -m src.scripts.data.v5_members member NAME        # NAME は lgb_extras_ped / xgb_extras_ped / lgb_extras / lgb_softmax / linear（tp27_eval member と同じ）
python -m src.scripts.data.v5_members mtl nn_extras_ped  # 補助損失つき NN（nn_extras_ped / nn_s）
python -m src.scripts.data.v5_members ens --name v5 --members v4:lgb_extras_ped ...   # LOYO アンサンブル（tp27_eval ens と同じ。入力は tp27_v5/oof）
環境変数: V5_GROUPS（追加する群。例 CY,BEST）, TP27_THREADS。出力先は tp27_v5/（V4 の tp27/ は読むだけ・上書きしない）。
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

V = Path(__file__).resolve().parents[3] / "data/local/modeling/training_pipeline_v4"
OUT = V / "tp27_v5"
os.environ["TP27_OUT"] = str(OUT)
for _k, _v in {"TP_BASE_DIR": V, "TP_WORK": V, "TP25C_OUT": V / "tp25c", "TP25E_OUT": V / "tp25e", "TP27_FAC_FIELD": V / "factory_field"}.items():
    os.environ.setdefault(_k, str(_v))

from src.scripts.data import tp27_eval as t27  # noqa: E402
from src.scripts.data import v5_common  # noqa: E402


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    src = V / "tp27/importance_added.json"
    if src.exists() and not (OUT / "importance_added.json").exists():
        shutil.copy2(src, OUT / "importance_added.json")      # NN の追加列の絞り込み（V4 と同じ重要度）
    v5_common.patch()
    sys.argv = ["tp27_eval"] + sys.argv[1:]
    t27.main()


if __name__ == "__main__":
    main()
