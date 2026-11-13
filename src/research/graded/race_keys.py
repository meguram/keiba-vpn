"""重賞のレース名とコースの鍵（重賞のルールベース予想家、docs/ideas/graded_rule_based_handicapper.md の第 3 節）。

- レース名: 「第 N 回」と空白を除き、表記をそろえる（ステークス→S、カップ→C、トロフィー→T）。冠の付け替え・改名は race_aliases.json で結ぶ。
- コース: 会場 × 芝ダ × 距離 × 内外回り。内外が決まらない 京都の芝 1400・1600 と 新潟の芝 2000 は、
  OP 以上のレースは外回りとする（course_type のある 2016〜2019 年の OP 以上はすべて外回りだった。course_type が空＝内回りと出るときだけ内）。
  それより下のクラスは、course_type が 2020〜2025 年に無く年によって切り方が変わるのを避けるため、年によらず「内外不明」の別のコースにする。
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

ALIASES = Path(__file__).with_name("race_aliases.json")
AMBIG_IO = {("08", "芝", 1400), ("08", "芝", 1600), ("04", "芝", 2000)}   # (会場コード, 芝ダ, 距離)


def norm_name(name) -> str:
    """「第 N 回」と空白を除き、表記をそろえる。"""
    s = re.sub(r"[\s　\xa0]", "", str(name or ""))
    s = re.sub(r"^第\d+回", "", s)
    return s.replace("ステークス", "S").replace("カップ", "C").replace("トロフィー", "T")


@lru_cache(maxsize=1)
def _aliases() -> tuple[dict, list]:
    j = json.loads(ALIASES.read_text(encoding="utf-8"))
    return j["aliases"], j.get("month_aliases", [])


def canonical_key(name, month: int | None = None) -> str:
    """そろえたレース名（別名の表を当てた後）。"""
    s = norm_name(name)
    al, mal = _aliases()
    for m in mal:
        if s == m["name"] and month is not None and int(month) in m["months"]:
            return m["to"]
    return al.get(s, s)


def canonical_keys(names: pd.Series, months: pd.Series) -> pd.Series:
    """canonical_key を列にまとめて当てる（名前と月の組ごとに 1 回だけ計算する）。"""
    df = pd.DataFrame({"n": names.astype(str).to_numpy(), "m": pd.to_numeric(months, errors="coerce").fillna(0).astype(int).to_numpy()})
    u = df.drop_duplicates().copy()
    u["k"] = [canonical_key(n, m) for n, m in zip(u["n"], u["m"])]
    return pd.Series(df.merge(u, on=["n", "m"], how="left")["k"].to_numpy(), index=names.index)


def resolve_io(vc: pd.Series, surf: pd.Series, dist: pd.Series, io: pd.Series, course_type: pd.Series, cls: pd.Series) -> pd.Series:
    """内外回り。folklore の io（内外が 1 つに決まる会場・距離）に、内外が決まらないコースの判定を足す。"""
    out = io.astype(object).where(io.notna(), None)
    key = list(zip(vc.astype(str), surf.astype(str), pd.to_numeric(dist, errors="coerce").fillna(0).astype(int)))
    amb = pd.Series([k in AMBIG_IO for k in key], index=vc.index)
    ct = course_type.astype(object)
    op = pd.to_numeric(cls, errors="coerce") >= 5
    says_in = ct.notna() & (ct.astype(str).str.strip() == "") & op
    out[amb & op] = "外"
    out[amb & says_in] = "内"
    out[amb & ~op] = "内外不明"   # course_type は 2016〜2019・2026 にしか無いので、年によって切り方が変わらないよう使わない
    return out


def course_key(venue: pd.Series, surf: pd.Series, dist: pd.Series, io: pd.Series) -> pd.Series:
    """コースの鍵（例: 京都芝1600外、中山ダ1200）。"""
    d = pd.to_numeric(dist, errors="coerce").fillna(0).astype(int).astype(str)
    return venue.astype(str) + surf.astype(str) + d + io.fillna("").astype(str).replace({"None": ""})
