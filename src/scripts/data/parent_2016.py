"""親モデル（v6e のメンバー・アンサンブル）を 2016〜2025 年の表で回す入口（再学習 2026-11）。v6_members と同じ CLI。

    python -m src.scripts.data.parent_2016 member lgb_extras_ped
    python -m src.scripts.data.parent_2016 mtl nn_extras_ped
    python -m src.scripts.data.parent_2016 ens --name v6 --members ...

- 読み先・書き先をすべて BASE（既定 /home/jovyan/work/retrain_out/base_2016）に向ける。本番の training_pipeline_v4 は読むだけ（hpo・採用列・旧 TL など）。
- 前段 S3 は BASE/v6/s3x_e.parquet（2016〜 の表で作った試行 7 の作り方）を、TP_WORK の oof/stage_features_expanding.parquet として置く。
- 年ごとの fold の学習の始まりを 2016 年にする（rebuild_base_2016._patch_years）。検証年は本番と同じ 2022〜2024 年。
- 出力: BASE/tp27_v6e/oof/*.parquet ほか。V5_GROUPS の既定は本番の v6e と同じ CY,BEST,RB。
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

BASE = Path(os.environ.get("REBUILD_BASE") or "/home/jovyan/work/retrain_out/base_2016")
PROD_V4 = Path(__file__).resolve().parents[3] / "data/local/modeling/training_pipeline_v4"
TAG = os.environ.get("V6_TAG", "e")


def prepare() -> None:
    (BASE / "oof").mkdir(parents=True, exist_ok=True)
    src, dst = BASE / "v6/s3x_e.parquet", BASE / "oof/stage_features_expanding.parquet"
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        tmp = dst.with_suffix(".tmp")
        shutil.copy2(src, tmp)
        tmp.replace(dst)                    # 途中で止まっても壊れたファイルを残さない
    for name in ("hpo", "training_laps_b", "ablation_nospeed"):
        link = BASE / name
        if not link.exists() and not link.is_symlink():
            target = PROD_V4 / name if (PROD_V4 / name).exists() else PROD_V4.parent / "training_pipeline" / name
            link.symlink_to(target)


def main() -> None:
    prepare()
    for k, v in {"TP_BASE_DIR": BASE, "TP_WORK": BASE, "TP25C_OUT": BASE / "tp25c", "TP25E_OUT": BASE / "tp25e",
                 "TP27_FAC_FIELD": BASE / "factory/field", "V6_OUT": BASE / "v6", "V5_TP38_OUT": BASE / "tp38",
                 "TP27_OUT": BASE / f"tp27_v6{TAG}"}.items():
        os.environ[k] = str(v)
    os.environ.setdefault("V5_GROUPS", "CY,BEST,RB")
    os.environ.pop("V6_STAGE", None)                 # 前段は TP_WORK の oof から読む（v6_work の作業ディレクトリは使わない）
    from src.scripts.data import rebuild_groups_2016 as rg

    rg.redirect()
    from src.scripts.data import tp27_eval as t27
    from src.scripts.data import v5_common
    from src.scripts.data import v6_members as m

    m.OUT = t27.OUT = BASE / f"tp27_v6{TAG}"
    os.environ["TP27_OUT"] = str(m.OUT)            # v6_members の import が本番の tp27_v6{TAG} を入れるので戻す
    v5_common.V = BASE
    m.main()


if __name__ == "__main__":
    sys.exit(main())
