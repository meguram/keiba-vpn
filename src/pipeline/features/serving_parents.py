"""推論用: 馬ごとの父・母父の ID（学習表の sire / dam_sire と同じキー）を引く。

学習表の sire / dam_sire は、出馬表が血統名を持たないので、馬別の血統 JSON から作った ID（``pedigree_keys.build_parent_table``。generation=1・position=0 が父、
generation=2・position=2 が母父。不明は空文字）で埋めている（2026-10-17 の血統キー復旧）。推論側も同じキーを使う。

- 基底の対応表は学習時に作ったもの（``training_pipeline_ped/pedigree_keys.parquet``、KEIBA_PED_KEYS で変更）。学習表の行はこの表で埋まっているので、
  表にある馬は（空文字でも）この表の値を使う（履歴の行の扱いが学習側と同じになる）。
- 表に無い馬（学習後に現れた馬）は血統 JSON（5 世代 → 10 世代）から引く。JSON が無い馬は空文字（学習側の『血統が取れない馬』と同じ扱い）。
  空の結果は覚えない（新馬の血統 JSON は後から取れるので、次の呼び出しで引き直す）。
- 無効化（学習表が血統名を空として作られていた頃の挙動に戻す）: 環境変数 KEIBA_PED_PARENT_KEYS=0。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.pipeline.features import pedigree_keys as pk

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_KEYS = ROOT / "data/local/modeling/training_pipeline_ped/pedigree_keys.parquet"


def enabled() -> bool:
    return (os.environ.get("KEIBA_PED_PARENT_KEYS") or "1").strip().lower() not in ("0", "false", "off", "no")


class ParentMap:
    def __init__(self, base: pd.DataFrame | None = None, dirs: list[Path] | None = None):
        self.dirs = dirs
        self._lock = threading.Lock()
        self._base: dict[str, tuple[str, str]] = {}
        if base is not None and len(base):
            b = base.drop_duplicates("horse_id")
            self._base = dict(zip(b["horse_id"].astype(str), zip(b["sire_id"].fillna("").astype(str), b["dam_sire_id"].fillna("").astype(str))))
        self._live: dict[str, tuple[str, str]] = {}

    @classmethod
    def load(cls, path: str | Path | None = None, dirs: list[Path] | None = None) -> "ParentMap":
        p = Path(path or os.environ.get("KEIBA_PED_KEYS") or DEFAULT_KEYS)
        return cls(pd.read_parquet(p) if p.exists() else None, dirs)

    def lookup(self, horse_ids: Iterable[str]) -> pd.DataFrame:
        """horse_id → sire_id, dam_sire_id（``pedigree_keys.build_parent_table`` と同じ形）。"""
        ids = sorted({str(x) for x in horse_ids})
        need = [h for h in ids if h not in self._base and not self._live.get(h, ("", ""))[0] and not self._live.get(h, ("", ""))[1]]
        if need:
            t = pk.build_parent_table(need, self.dirs)
            with self._lock:
                for h, s, d in zip(t["horse_id"], t["sire_id"], t["dam_sire_id"]):
                    if s or d:
                        self._live[h] = (s, d)
        rows = []
        for h in ids:
            s, d = self._base.get(h) or self._live.get(h) or ("", "")
            rows.append((h, s, d))
        return pd.DataFrame(rows, columns=["horse_id", "sire_id", "dam_sire_id"])

    def attach(self, df: pd.DataFrame) -> pd.DataFrame:
        """df（horse_id を持つ）の sire / dam_sire を ID で埋めた複製（``pedigree_keys.attach_parent_ids`` と同じ）。"""
        return pk.attach_parent_ids(df, self.lookup(df["horse_id"].astype(str).unique()))


_MAP: ParentMap | None = None


def get_map() -> ParentMap:
    global _MAP
    if _MAP is None:
        _MAP = ParentMap.load()
    return _MAP
