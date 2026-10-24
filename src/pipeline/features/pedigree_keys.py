"""馬ごとの父・母父を horse_id（種牡馬の ID）で引く。

出馬表のパーサは ``span.Pedigree`` を探すが、出馬表ページにはその要素が無く、``race_shutuba`` の ``sire`` / ``dam_sire`` は
全期間で空になる（学習表も同じ）。血統は馬別の血統 JSON（``horse_pedigree_5gen`` / ``_10gen``）の ``ancestors`` にあり、
generation=1・position=0 が父、generation=2・position=2 が母父（名前欄と一致することを確認済み）。
名前は表記ゆれ（日英併記など）があるので、キーは ID にする。ID が無い祖先は空文字（不明）にする。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PED_DIRS = [ROOT / "data/local/horse_pedigree_5gen", ROOT / "data/local/horse_pedigree_10gen"]


def parents_from_ancestors(ancestors: list[dict]) -> tuple[str, str]:
    sire = dam_sire = ""
    for a in ancestors or []:
        g, p = a.get("generation"), a.get("position")
        if g == 1 and p == 0:
            sire = str(a.get("horse_id") or "")
        elif g == 2 and p == 2:
            dam_sire = str(a.get("horse_id") or "")
    return sire, dam_sire


def build_parent_table(horse_ids: Iterable[str], dirs: list[Path] | None = None) -> pd.DataFrame:
    """horse_id → (sire_id, dam_sire_id)。先に見つかったディレクトリを優先（5 世代 → 10 世代）。不明は空文字。"""
    dirs = [Path(d) for d in (dirs or PED_DIRS)]
    rows = []
    for h in sorted({str(x) for x in horse_ids}):
        sire = dam_sire = ""
        for d in dirs:
            f = d / h[:4] / f"{h}.json"
            if not f.exists():
                continue
            try:
                s, ds = parents_from_ancestors(json.loads(f.read_text(encoding="utf-8")).get("ancestors") or [])
            except Exception:
                continue
            if s or ds:
                sire, dam_sire = s, ds
                break
        rows.append((h, sire, dam_sire))
    return pd.DataFrame(rows, columns=["horse_id", "sire_id", "dam_sire_id"])


def attach_parent_ids(df: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """df（horse_id を持つ）の ``sire`` / ``dam_sire`` を ID で埋めた複製を返す。"""
    t = table.drop_duplicates("horse_id").set_index("horse_id")
    out = df.copy()
    hid = out["horse_id"].astype(str)
    out["sire"] = hid.map(t["sire_id"]).fillna("").astype(str).to_numpy()
    out["dam_sire"] = hid.map(t["dam_sire_id"]).fillna("").astype(str).to_numpy()
    return out
