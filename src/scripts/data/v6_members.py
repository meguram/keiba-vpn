"""v6 のアンサンブル用に、7 メンバーの OOF（valid 2022〜24）と LOYO アンサンブルを tp27_v6/ に作る（v5_members と同じ手順・同じ CLI）。

python -m src.scripts.data.v6_members member NAME        # NAME は lgb_extras_ped / xgb_extras_ped / lgb_extras / lgb_softmax / linear
python -m src.scripts.data.v6_members mtl nn_extras_ped  # 補助損失つき NN
python -m src.scripts.data.v6_members ens --name v6 --members ...
環境変数: V5_GROUPS（v5 の群 + v6 の群。例 CY,BEST,RB,CB,NJ）, V6_STAGE（前段 S3 を作り直した版 b など。v6_work が作業ディレクトリを差し替える。空なら v4 のまま）, TP27_THREADS。
出力先は tp27_v6/（tp27_v5/ は読むだけ・上書きしない）。NN には追加群を足さない（v5 と同じ設計）が、前段の差し替えは全メンバーに効く。
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pandas as pd

V = Path(__file__).resolve().parents[3] / "data/local/modeling/training_pipeline_v4"
TAG = os.environ.get("V6_TAG", "")                     # 前段の版などで出力先を分ける（例 e → tp27_v6e）
OUT = V / f"tp27_v6{TAG}"
os.environ["TP27_OUT"] = str(OUT)
if os.environ.get("V6_STAGE"):
    from src.scripts.data import v6_work

    v6_work.use(os.environ["V6_STAGE"])               # 前段 S3 を差し替えた作業ディレクトリ（tp27_eval / tp25e_eval / tp05_data はそこから前段を読む）
for _k, _v in {"TP_BASE_DIR": V, "TP_WORK": V, "TP25C_OUT": V / "tp25c", "TP25E_OUT": V / "tp25e", "TP27_FAC_FIELD": V / "factory_field"}.items():
    os.environ.setdefault(_k, str(_v))

from src.scripts.data import tp27_eval as t27  # noqa: E402
from src.scripts.data import v5_common  # noqa: E402
from src.scripts.data import v5_select_eval as sel  # noqa: E402
from src.scripts.data import v6_eval as ev  # noqa: E402

KEY = ["race_id", "horse_id"]
V6_GROUPS = {"PT", "GB", "CB", "OWNN", "RS59", *ev.EXTERNAL}


def patch() -> None:
    v5_common.patch()
    orig_group = sel.load_group

    def load_group(name: str, df: pd.DataFrame) -> pd.DataFrame:
        if name in V6_GROUPS:
            return ev.load_new(name, df, pd.DataFrame(index=df.index))
        return orig_group(name, df)

    sel.load_group = load_group


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("importance_added.json",):
        src = V / "tp27_v5" / name
        if src.exists() and not (OUT / name).exists():
            shutil.copy2(src, OUT / name)
    patch()
    sys.argv = ["tp27_eval"] + sys.argv[1:]
    t27.main()


if __name__ == "__main__":
    main()
