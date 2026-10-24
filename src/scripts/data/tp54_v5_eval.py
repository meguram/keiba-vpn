"""tp-54: 追走難度 v2（EASE2 の 5 列）が v5 の列セットに足したとき、勝率・連対・複勝の LogLoss を改善するか。

python -m src.scripts.data.tp54_v5_eval run g_CY_BEST_EASE2     # v5 構成 d（V4 + tl_ 再取得 + CY + ALL_BEST）に EASE2 を足す
python -m src.scripts.data.tp54_v5_eval run h_CY_EASE2          # 構成 c（V4 + tl_ 再取得 + CY）に EASE2 を足す
python -m src.scripts.data.tp54_v5_eval compare                 # g vs d、h vs c を対で比べる（採否の基準は v5_select_eval と同じ）

v5_select_eval の仕組み（tp-27 方式、LightGBM、valid 2022・2023・2024 expanding、2 seed）をそのまま使う。出力は別ディレクトリ
v5_select_tp54/ に置き、v5-retrain の v5_select/ は書き換えない（基準の OOF だけコピーして読む）。EASE2 は年別 leave-one-year-out の予測
（tp54_model oof）で、対象の年の結果は学習に使っていない。市場オッズ・人気は使わない。test 2025・2026 年は使わない。
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pandas as pd

from src.scripts.data import v5_select_eval as v5

TP54_OUT = Path(os.environ.get("TP54_OUT") or "/home/jovyan/work/tp54_out")
OUT = v5.OUT.parent / "v5_select_tp54"

v5.CONFIGS["g_CY_BEST_EASE2"] = {"base": "fresh", "groups": ["CY", "BEST", "EASE2"]}
v5.CONFIGS["h_CY_EASE2"] = {"base": "fresh", "groups": ["CY", "EASE2"]}
v5.PAIRS = [("g_CY_BEST_EASE2", "d_CY_BEST"), ("h_CY_EASE2", "c_CY")]
_orig_load_group = v5.load_group


def _load_group(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if name == "EASE2":
        return v5._merge_keys(df, pd.read_parquet(TP54_OUT / "groups" / "EASE2.parquet"))
    return _orig_load_group(name, df)


v5.load_group = _load_group


def _prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for n in ("d_CY_BEST", "c_CY"):
        src, dst = v5.OUT / f"oof_{n}.parquet", OUT / f"oof_{n}.parquet"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    lv_src, lv_dst = v5.OUT / "levels.json", OUT / "levels.json"
    if lv_src.exists() and not lv_dst.exists():
        shutil.copy2(lv_src, lv_dst)


def main() -> int:
    _prepare()
    v5.OUT = OUT
    return v5.main()


if __name__ == "__main__":
    sys.exit(main())
