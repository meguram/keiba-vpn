"""『○○ × コース』の成績。出走馬ごとに、今回と同じコース（同じ競馬場・同じ芝/ダ・同じ距離）でのこれまでの成績を出す。

○○（軸）は Dimension で差し替える: 騎手 / 枠 / 種牡馬。軸を足すときは Dimension を 1 つ作って registry.py に足す。
対象は、このレースの開催日より前の JRA 全レース（当日以降の結果は入れない）。キーは ID（名前では結合しない）。
単勝オッズが ODDS_CUT（50 倍）以上の出走は、大穴の外れ値として集計から除く（出走数・着順・率・回収率のすべて）。
各馬の行からは、その馬自身の過去の出走を除く（父の産駒成績に自分の成績が混ざり、出走の少ない父で個体の成績がそのまま出てしまうのを避ける）。
色の基準は、勝率・複勝率はその条件の全馬の平均、回収率は 100%（損益分岐）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from src.research.race_analysis.base import Column, ColumnGroup, ModuleOutput, RaceContext
from src.research.race_analysis.runs import SOURCE_DATASET, SOURCE_PROVIDER, ODDS_CUT, drop_long_shots, load_runs_roi, load_runs_with_sire, load_sire_map, sire_map_available

GROUP = "course"
_ROI_SCALE = {"lo": 0.5, "mid": 1.0, "hi": 1.5}   # 回収率の色: 100% で無色、150% 以上で最も緑、50% 以下で最も赤

# 平均に対する倍率で色の端を決める（lo=平均×0.25 で最も赤、hi=平均×倍率 で最も緑。hi は 100% が上限）
_HEAT = {"win": 3.0, "top3": 2.0}


@dataclass(frozen=True)
class Dimension:
    id: str
    label: str
    description: str
    run_col: str                                    # 出走表（runs）でその軸を表す列
    min_n: int                                      # これ未満の出走数の群は参考値として薄く表示し、色を付けない
    entry_key: Callable[[dict[str, Any], dict[str, Any]], str | None]   # (出走馬, 補助表) -> キー（分からなければ None）
    key_header: str | None = None                   # 軸の名前を表の列として出すときの見出し（表は馬番・馬名とこの列だけを出し、性齢・騎手などは出さない）
    key_text: Callable[[dict[str, Any], dict[str, Any]], str | None] | None = None
    needs_sire_map: bool = False
    unknown_note: str | None = None                 # 軸が分からない馬がいるときの注記


def _jockey_key(e: dict[str, Any], _: dict[str, Any]) -> str | None:
    return str(e["jockey_id"]) if e.get("jockey_id") else None


def _bracket_key(e: dict[str, Any], _: dict[str, Any]) -> str | None:
    try:
        b = int(e.get("bracket_number") or 0)
    except (TypeError, ValueError):
        return None
    return str(b) if b > 0 else None


def _jockey_text(e: dict[str, Any], _: dict[str, Any]) -> str | None:
    return e.get("jockey_name") or None


def _bracket_text(e: dict[str, Any], aux: dict[str, Any]) -> str | None:
    b = _bracket_key(e, aux)
    return f"{b}枠" if b else None


def _sire_key(e: dict[str, Any], aux: dict[str, Any]) -> str | None:
    row = aux.get("sire", {}).get(str(e.get("horse_id") or ""))
    return str(row[0]) if row and row[0] else None


# 外国産の父は『マジェスティックウォリアーMajestic Warrior(米)』のようにカナ名と英名が続けて入っている。表では英名を落とす
_KANA_THEN_LATIN = re.compile(r"^(.*[\u30A0-\u30FF])\s*[A-Za-z][A-Za-z0-9 .'&-]*\s*(\(.+?\))?$")


def display_sire_name(name: str | None) -> str | None:
    if not name:
        return name
    m = _KANA_THEN_LATIN.match(name)
    return f"{m.group(1)}{m.group(2) or ''}" if m else name


def _sire_text(e: dict[str, Any], aux: dict[str, Any]) -> str | None:
    row = aux.get("sire", {}).get(str(e.get("horse_id") or ""))
    return display_sire_name(row[1]) if row else None


JOCKEY = Dimension(
    id="jockey_course", label="騎手×コース", min_n=10, run_col="jockey_id", entry_key=_jockey_key, key_header="騎手", key_text=_jockey_text,
    description="出走馬の騎手が、今回と同じコース（同じ競馬場・芝/ダート・距離）でこれまで残してきた成績と単勝・複勝回収率。",
)
GATE = Dimension(
    id="gate_course", label="枠×コース", min_n=30, run_col="bracket_number", entry_key=_bracket_key, key_header="枠", key_text=_bracket_text,
    description="出走馬の枠（1〜8枠）が、今回と同じコース（同じ競馬場・芝/ダート・距離）でこれまで残してきた成績と単勝・複勝回収率。馬の強さに関係なく、枠そのものの傾向です。",
    unknown_note="枠順がまだ確定していない馬は、確定後に表示されます。",
)
SIRE = Dimension(
    id="sire_course", label="種牡馬×コース", min_n=10, run_col="sire_id", entry_key=_sire_key, key_header="父（種牡馬）", key_text=_sire_text,
    needs_sire_map=True,
    description="出走馬の父（種牡馬）の産駒が、今回と同じコース（同じ競馬場・芝/ダート・距離）でこれまで残してきた勝率・複勝率と、単勝・複勝回収率。",
    unknown_note="父を特定できない馬は表示されません。",
)

def _mask(runs: pd.DataFrame, ctx: RaceContext) -> pd.Series:
    return (runs["surface"] == ctx.surface) & (runs["venue"] == ctx.venue) & (runs["distance"] == ctx.distance)


# 集計の 1 組 = (出走, 1着, 2着, 3着, 払戻が読めた出走, 単勝払戻の合計, 複勝払戻の合計)。払戻は 100 円あたり/100
Tally = tuple[int, int, int, int, int, float, float]
_ZERO: Tally = (0, 0, 0, 0, 0, 0.0, 0.0)


def _frame(sub: pd.DataFrame, keys: dict[str, Any]) -> pd.DataFrame:
    f = sub["finish_position"]
    ok = sub["win_ret"].notna() & sub["place_ret"].notna()
    return pd.DataFrame({**keys, "n": 1, "w1": (f == 1).to_numpy(), "w2": (f == 2).to_numpy(), "w3": (f == 3).to_numpy(),
                         "rn": ok.to_numpy(), "wr": sub["win_ret"].where(ok, 0.0).to_numpy(), "pr": sub["place_ret"].where(ok, 0.0).to_numpy()})


def _row(r: Any) -> Tally:
    return (int(r.n), int(r.w1), int(r.w2), int(r.w3), int(r.rn), float(r.wr), float(r.pr))


def _tally(sub: pd.DataFrame, col: str) -> dict[str, Tally]:
    """キー -> 集計。"""
    if sub.empty:
        return {}
    agg = _frame(sub, {"k": sub[col].astype(str).to_numpy()}).groupby("k").sum()
    return {k: _row(r) for k, r in agg.iterrows()}


def _tally_own(sub: pd.DataFrame, col: str, horse_ids: set[str]) -> dict[tuple[str, str], Tally]:
    """(キー, 馬) -> 集計。出走馬自身の分を各馬の集計から差し引くため。"""
    sub = sub[sub["horse_id"].astype(str).isin(horse_ids)]
    if sub.empty:
        return {}
    agg = _frame(sub, {"k": sub[col].astype(str).to_numpy(), "h": sub["horse_id"].astype(str).to_numpy()}).groupby(["k", "h"]).sum()
    return {(str(k), str(h)): _row(r) for (k, h), r in agg.iterrows()}


def _baseline(sub: pd.DataFrame) -> dict[str, float] | None:
    """その条件の全馬の平均（勝率・複勝率・単勝回収率・複勝回収率）。色の基準と見出しの説明。"""
    n = len(sub)
    if n == 0:
        return None
    f = sub["finish_position"]
    out = {"win": float((f == 1).sum()) / n, "top3": float((f <= 3).sum()) / n}
    ok = sub["win_ret"].notna() & sub["place_ret"].notna()
    if ok.any():
        out["win_roi"], out["place_roi"] = float(sub.loc[ok, "win_ret"].mean()), float(sub.loc[ok, "place_ret"].mean())
    return out


def _scale(mid: float | None, key: str) -> dict[str, float] | None:
    if not mid:
        return None
    return {"lo": mid * 0.25, "mid": mid, "hi": min(1.0, mid * _HEAT[key])}


class CourseStatsModule:
    def __init__(self, dim: Dimension, *, label: str | None = None, description: str | None = None) -> None:
        self.dim = dim
        self.id = dim.id
        self.label = label or dim.label
        self.description = description or dim.description

    def applicable(self, ctx: RaceContext) -> bool:
        if self.dim.needs_sire_map and not sire_map_available():
            return False
        return bool(ctx.venue and ctx.surface and ctx.distance)

    def _aux(self) -> dict[str, Any]:
        if not self.dim.needs_sire_map:
            return {}
        m = load_sire_map()
        if m is None:
            return {"sire": {}}
        cols = ["horse_id", "sire_id", "sire_name"]
        return {"sire": {str(r.horse_id): (r.sire_id, r.sire_name) for r in m[cols].itertuples(index=False)}}

    def compute(self, ctx: RaceContext) -> ModuleOutput:
        dim = self.dim
        dated = load_runs_with_sire(roi=True) if dim.needs_sire_map else load_runs_roi()
        if ctx.date:
            dated = dated[dated["date"] < ctx.date]
        dated = drop_long_shots(dated[dated["race_id"] != ctx.race_id])
        until = dated["date"].max() if len(dated) else None
        source = {
            "provider": SOURCE_PROVIDER,
            "dataset": SOURCE_DATASET,
            "period_from": dated["date"].min() if len(dated) else None,
            "period_to": until,
            "races": int(dated["race_id"].nunique()),
            "rows": int(len(dated)),
        }

        aux = self._aux()
        s = ctx.surface or ""
        base = _baseline(dated[_mask(dated, ctx)])
        stats = [("win", "勝率", None), ("top3", "複勝率", "3着以内の率")]
        columns: list[Column] = [Column(f"{GROUP}_rec", "成績", "record", GROUP, "1着-2着-3着-着外")]
        for key, label, desc in stats:
            mid = (base or {}).get(key)
            hint = f"{desc}。" if desc else ""
            title = f"{hint}この条件の全馬の平均は {mid * 100:.1f}%（緑=平均より高い／赤=低い）" if mid else desc
            columns.append(Column(f"{GROUP}_{key}", label, "pct", GROUP, title, _scale(mid, key)))
        for key, label in (("win_roi", "単勝回収率"), ("place_roi", "複勝回収率")):
            mid = (base or {}).get(key)
            title = f"100円ずつ買い続けたときの戻り（100%超で黒字。緑=100%超／赤=100%未満）" + (f"。この条件の全馬の平均は {mid * 100:.1f}%" if mid else "")
            columns.append(Column(f"{GROUP}_{key}", label, "pct", GROUP, title, _ROI_SCALE))
        groups = [ColumnGroup(key=GROUP, label="同コース", sub=f"{ctx.venue or ''} {s}{ctx.distance or ''}m", n_key=f"{GROUP}_n", min_n=dim.min_n)]

        cells: dict[str, dict[str, Any]] = {e["key"]: {} for e in ctx.entries}
        keys = {e["key"]: dim.entry_key(e, aux) for e in ctx.entries}
        unknown = any(k is None for k in keys.values())
        wanted = {k for k in keys.values() if k}
        sub_all = dated[dated[dim.run_col].astype(str).isin(wanted)] if wanted else dated.iloc[0:0]
        scoped = sub_all[_mask(sub_all, ctx)]
        tally = _tally(scoped, dim.run_col)
        own = _tally_own(scoped, dim.run_col, {str(e.get("horse_id")) for e in ctx.entries if e.get("horse_id")})
        for e in ctx.entries:
            k = keys[e["key"]]
            row = cells[e["key"]]
            t = tally.get(k, _ZERO) if k else _ZERO
            if k and e.get("horse_id"):
                o = own.get((k, str(e["horse_id"])), _ZERO)
                t = tuple(a - b for a, b in zip(t, o))   # type: ignore[assignment]
            n, w1, w2, w3, rn, wr, pr = t
            row[f"{GROUP}_n"] = n
            row[f"{GROUP}_rec"] = [w1, w2, w3, n - w1 - w2 - w3] if k else None
            row[f"{GROUP}_win"] = w1 / n if n else None
            row[f"{GROUP}_top3"] = (w1 + w2 + w3) / n if n else None
            row[f"{GROUP}_win_roi"] = wr / rn if rn else None
            row[f"{GROUP}_place_roi"] = pr / rn if rn else None
            if dim.key_text:
                row["_key"] = dim.key_text(e, aux)

        notes = [
            "このレースの開催日より前の JRA 全レースの結果から集計しています。各馬の行は、その馬自身の過去の出走を除いた成績です。",
            f"単勝オッズが{ODDS_CUT:.0f}倍以上の出走は、外れ値として出走数・着順・率・回収率のすべてから除いています。",
            f"出走数が{dim.min_n}未満の群は、参考値として薄く表示し、色は付けません。",
            "回収率は、その条件の出走に100円ずつ買い続けたときの払戻÷賭け金です（払戻が読めない出走は回収率の計算から除く）。色は、勝率・複勝率はその条件の全馬の平均、回収率は100%が基準です。",
        ]
        if dim.unknown_note and unknown:
            notes.append(dim.unknown_note)
        return ModuleOutput(columns=columns, groups=groups, cells=cells, notes=notes, data_until=until, source=source, key_header=dim.key_header)
