"""保存済み horse_result の race_history で、通過順(passing_order)と上り3F(last_3f)がずれた値になっている件の補正。

戦績ページが会員表示の 33 列の表になった後も固定位置で読んでいたため、保存済み horse_result の通過順は '66' や '**'、上り3F は 88.0 や 0.0 のような値になっている
（race_result_flat との一致率は通過順 0.6%、上り3F 1.1%。馬体重・着順・タイム・馬番は一致）。パーサーは列名ベースに修正済み（2026-10-17）。
2020 年以降のレースは race_result_flat（正）から同じ (馬, レース) の値を引けるので、再取得せずに補正する。

python -m src.scraper.horse_result_fix        # data/local/meta/horse_result_fix/overlay.parquet を作る
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "data/page_reference/tables"
OVERLAY = ROOT / "data/local/meta/horse_result_fix/overlay.parquet"
FIELDS = ("passing_order", "last_3f")


def build_overlay(years: range = range(2020, 2027)) -> pd.DataFrame:
    parts = []
    for y in years:
        f = TABLES / str(y) / "race_result_flat.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f, columns=["race_id", "horse_id", *FIELDS]))
    d = pd.concat(parts, ignore_index=True)
    d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    return d.drop_duplicates(["horse_id", "race_id"], keep="last")


def load_lookup(path: Path = OVERLAY) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists() or os.environ.get("KEIBA_HORSE_RESULT_FIX", "1") in ("0", "false"):
        return {}
    d = pd.read_parquet(path)
    return {(h, r): {"passing_order": p, "last_3f": l} for h, r, p, l in zip(d["horse_id"], d["race_id"], d["passing_order"], d["last_3f"])}


def apply_overlay(hr: dict, lookup: dict[tuple[str, str], dict[str, Any]]) -> dict:
    """horse_result の 1 馬分に補正を当てたコピーを返す（race_result_flat に同じレースがある走だけ。元の辞書は変えない）。"""
    if not lookup or not hr.get("race_history"):
        return hr
    hid = str(hr.get("horse_id") or "")
    fixed = []
    for e in hr["race_history"]:
        fx = lookup.get((hid, str(e.get("race_id") or "")))
        if fx is None:
            fixed.append(e)
            continue
        e2 = dict(e)
        po, l3 = fx["passing_order"], fx["last_3f"]
        e2["passing_order"] = "" if po is None or (isinstance(po, float) and pd.isna(po)) else str(po)
        e2["last_3f"] = 0.0 if l3 is None or pd.isna(l3) else float(l3)
        fixed.append(e2)
    return {**hr, "race_history": fixed}


def main() -> None:
    d = build_overlay()
    OVERLAY.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OVERLAY, index=False)
    print("overlay", len(d), "行 →", OVERLAY)


if __name__ == "__main__":
    main()
