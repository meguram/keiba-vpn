"""『○○ × コース』の成績。出走馬ごとに、今回と同じ条件でのこれまでの成績を 2 つの広さで出す。

  同コース           : 同じ競馬場・同じ芝/ダ・同じ距離
  同競馬場・同芝ダ   : 距離は問わない

○○（軸）は Dimension で差し替える: 騎手 / 枠 / 種牡馬。軸を足すときは Dimension を 1 つ作って registry.py に足す。
対象は、このレースの開催日より前の JRA 全レース（当日以降の結果は入れない）。キーは ID（名前では結合しない）。
色の基準は、その条件の全馬の平均（勝率・連対率・複勝率）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from src.research.race_analysis.base import Column, ColumnGroup, ModuleOutput, RaceContext
from src.research.race_analysis.runs import SOURCE_DATASET, SOURCE_PROVIDER, load_runs, load_runs_with_sire, load_sire_map, sire_map_available

SCOPES = ("course", "venue_surface")

# 平均に対する倍率で色の端を決める（lo=平均×0.25 で最も赤、hi=平均×倍率 で最も緑。hi は 100% が上限）
_HEAT = {"win": 3.0, "top2": 2.4, "top3": 2.0}


@dataclass(frozen=True)
class Dimension:
    id: str
    label: str
    description: str
    run_col: str                                    # 出走表（runs）でその軸を表す列
    min_n: int                                      # これ未満の出走数の群は参考値として薄く表示し、色を付けない
    entry_key: Callable[[dict[str, Any], dict[str, Any]], str | None]   # (出走馬, 補助表) -> キー（分からなければ None）
    key_header: str | None = None                   # 軸の名前を表の列として出すときの見出し（騎手・枠は既存の列で見えるので不要）
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


def _sire_key(e: dict[str, Any], aux: dict[str, Any]) -> str | None:
    row = aux.get("sire", {}).get(str(e.get("horse_id") or ""))
    return str(row[0]) if row and row[0] else None


def _sire_text(e: dict[str, Any], aux: dict[str, Any]) -> str | None:
    row = aux.get("sire", {}).get(str(e.get("horse_id") or ""))
    return row[1] if row else None


JOCKEY = Dimension(
    id="jockey_course", label="騎手×コース", min_n=10, run_col="jockey_id", entry_key=_jockey_key,
    description="出走馬の騎手が、今回と同じコース・同じ競馬場・同じ芝/ダートでこれまで残してきた成績。",
)
GATE = Dimension(
    id="gate_course", label="枠×コース", min_n=30, run_col="bracket_number", entry_key=_bracket_key,
    description="出走馬の枠（1〜8枠）が、今回と同じコース・同じ競馬場・同じ芝/ダートでこれまで残してきた成績。馬の強さに関係なく、枠そのものの傾向です。",
    unknown_note="枠順がまだ確定していない馬は、確定後に表示されます。",
)
SIRE = Dimension(
    id="sire_course", label="種牡馬×コース", min_n=10, run_col="sire_id", entry_key=_sire_key, key_header="父（種牡馬）", key_text=_sire_text,
    needs_sire_map=True,
    description="出走馬の父（種牡馬）の産駒が、今回と同じコース・同じ競馬場・同じ芝/ダートでこれまで残してきた成績。",
    unknown_note="父を特定できない馬は表示されません。",
)


def _scope_labels(ctx: RaceContext) -> dict[str, tuple[str, str]]:
    s = ctx.surface or ""
    return {
        "course": ("同コース", f"{ctx.venue or ''} {s}{ctx.distance or ''}m"),
        "venue_surface": ("同競馬場・同芝ダ", f"{ctx.venue or ''} {s}（距離問わず）"),
    }


def _mask(runs: pd.DataFrame, ctx: RaceContext, scope: str) -> pd.Series:
    m = (runs["surface"] == ctx.surface) & (runs["venue"] == ctx.venue)
    if scope == "course":
        m &= runs["distance"] == ctx.distance
    return m


def _tally(sub: pd.DataFrame, col: str) -> dict[str, tuple[int, int, int, int]]:
    """キー -> (出走, 1着, 2着, 3着)。"""
    if sub.empty:
        return {}
    f = sub["finish_position"]
    g = pd.DataFrame({"k": sub[col].astype(str).to_numpy(), "n": 1, "w1": (f == 1).to_numpy(), "w2": (f == 2).to_numpy(), "w3": (f == 3).to_numpy()})
    agg = g.groupby("k").sum()
    return {k: (int(r.n), int(r.w1), int(r.w2), int(r.w3)) for k, r in agg.iterrows()}


def _baseline(sub: pd.DataFrame) -> dict[str, float] | None:
    """その条件の全馬の平均（勝率・連対率・複勝率）。色の基準。"""
    n = len(sub)
    if n == 0:
        return None
    f = sub["finish_position"]
    return {"win": float((f == 1).sum()) / n, "top2": float((f <= 2).sum()) / n, "top3": float((f <= 3).sum()) / n}


def _scale(mid: float | None, key: str) -> dict[str, float] | None:
    if not mid:
        return None
    return {"lo": mid * 0.25, "mid": mid, "hi": min(1.0, mid * _HEAT[key])}


class CourseStatsModule:
    def __init__(self, dim: Dimension) -> None:
        self.dim = dim
        self.id = dim.id
        self.label = dim.label
        self.description = dim.description

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
        return {"sire": {str(r.horse_id): (r.sire_id, r.sire_name) for r in m[["horse_id", "sire_id", "sire_name"]].itertuples(index=False)}}

    def compute(self, ctx: RaceContext) -> ModuleOutput:
        dim = self.dim
        labels = _scope_labels(ctx)
        dated = load_runs_with_sire() if dim.needs_sire_map else load_runs()
        if ctx.date:
            dated = dated[dated["date"] < ctx.date]
        dated = dated[dated["race_id"] != ctx.race_id]
        until = dated["date"].max() if len(dated) else None
        source = {
            "provider": SOURCE_PROVIDER,
            "dataset": SOURCE_DATASET,
            "period_from": dated["date"].min() if len(dated) else None,
            "period_to": until,
            "races": int(dated["race_id"].nunique()),
            "rows": int(len(dated)),
        }
        baselines = {sc: _baseline(dated[_mask(dated, ctx, sc)]) for sc in SCOPES}

        columns: list[Column] = []
        groups: list[ColumnGroup] = []
        for sc in SCOPES:
            name, sub = labels[sc]
            groups.append(ColumnGroup(key=sc, label=name, sub=sub, n_key=f"{sc}_n", min_n=dim.min_n))
            columns.append(Column(f"{sc}_rec", "成績", "record", sc, "1着-2着-3着-着外"))
            for key, label, desc in (("win", "勝率", None), ("top2", "連対率", "2着以内の率"), ("top3", "複勝率", "3着以内の率")):
                mid = (baselines[sc] or {}).get(key)
                hint = f"{desc}。" if desc else ""
                title = f"{hint}この条件の全馬の平均は {mid * 100:.1f}%（緑=平均より高い／赤=低い）" if mid else desc
                columns.append(Column(f"{sc}_{key}", label, "pct", sc, title, _scale(mid, key)))

        aux = self._aux()
        keys = {e["key"]: dim.entry_key(e, aux) for e in ctx.entries}
        wanted = {k for k in keys.values() if k}
        sub_all = dated[dated[dim.run_col].astype(str).isin(wanted)] if wanted else dated.iloc[0:0]
        tallies = {sc: _tally(sub_all[_mask(sub_all, ctx, sc)], dim.run_col) for sc in SCOPES}

        cells: dict[str, dict[str, Any]] = {}
        for e in ctx.entries:
            k = keys[e["key"]]
            row: dict[str, Any] = {}
            for sc in SCOPES:
                n, w1, w2, w3 = tallies[sc].get(k, (0, 0, 0, 0)) if k else (0, 0, 0, 0)
                row[f"{sc}_n"] = n
                row[f"{sc}_rec"] = [w1, w2, w3, n - w1 - w2 - w3] if k else None
                row[f"{sc}_win"] = w1 / n if n else None
                row[f"{sc}_top2"] = (w1 + w2) / n if n else None
                row[f"{sc}_top3"] = (w1 + w2 + w3) / n if n else None
            if dim.key_text:
                row["_key"] = dim.key_text(e, aux)
            cells[e["key"]] = row

        notes = [
            "このレースの開催日より前の JRA 全レースの結果から集計しています。",
            f"出走数が{dim.min_n}未満の群は、参考値として薄く表示し、色は付けません。",
            "セルの色: その条件の全馬の平均より高いほど緑、低いほど赤（濃いほど差が大きい）。",
        ]
        if dim.unknown_note and any(k is None for k in keys.values()):
            notes.append(dim.unknown_note)
        return ModuleOutput(columns=columns, groups=groups, cells=cells, notes=notes, data_until=until, source=source, key_header=dim.key_header)
