"""v6: 前段 S3 を差し替えた『作業ディレクトリ』を作る。

tp27_eval / tp25e_eval / tp05_data は前段を <TP_WORK>/oof/stage_features_expanding.parquet から読み、相対化列もそこから作る。
V4 の作業ディレクトリ（training_pipeline_v4）を書き換えずに差し替えるため、v6/work/ に『V4 の各項目への symlink + oof/ だけ実体
（stage_features_expanding.parquet を s3x_<variant>.parquet に差し替え）』を作り、TP_WORK をそこへ向ける。
使い方: import より前に `from src.scripts.data import v6_work; v6_work.use("b")`。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

V = Path(__file__).resolve().parents[3] / "data/local/modeling/training_pipeline_v4"
OUT = Path(os.environ.get("V6_OUT") or V / "v6")


def _link(link: Path, target: Path) -> None:
    try:
        if not link.exists() and not link.is_symlink():
            link.symlink_to(target)
    except FileExistsError:                       # 同時に起動した別プロセスが先に作った
        pass


def use(stage: str) -> Path:
    farm = OUT / f"work_{stage}" if stage != "b" else OUT / "work"
    (farm / "oof").mkdir(parents=True, exist_ok=True)
    for e in sorted(V.iterdir()):
        if e.name in ("v6", "oof"):
            continue
        link = farm / e.name
        _link(link, e)
    for e in sorted((V / "oof").iterdir()):
        if e.name == "stage_features_expanding.parquet":
            continue
        _link(farm / "oof" / e.name, e)
    src = OUT / f"s3x_{stage}.parquet"
    dst = farm / "oof/stage_features_expanding.parquet"
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        shutil.copy2(src, dst)
    os.environ["TP_WORK"] = str(farm)
    return farm
