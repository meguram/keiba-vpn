"""騎手 × コースの成績。各出走馬の騎手が、今回と同じ条件でこれまでどんな成績かを 3 つの広さで出す。

  同コース           : 同じ競馬場・同じ芝/ダ・同じ距離
  同競馬場・同芝ダ   : 距離は問わない
  同芝ダ（全場）     : 競馬場も距離も問わない（その騎手の基準）

対象は、このレースの開催日より前の JRA 全レース（2020 年以降。当日以降の結果は入れない）。キーは jockey_id（名前では結合しない）。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.research.race_analysis.base import Column, ColumnGroup, ModuleOutput, RaceContext
from src.research.race_analysis.runs import SOURCE_DATASET, SOURCE_PROVIDER, load_runs

MIN_N = 10   # これ未満の出走数は参考値として薄く表示する

SCOPES = ("course", "venue_surface", "surface")


def _scope_labels(ctx: RaceContext) -> dict[str, tuple[str, str]]:
    s = ctx.surface or ""
    return {
        "course": ("同コース", f"{ctx.venue or ''} {s}{ctx.distance or ''}m"),
        "venue_surface": ("同競馬場・同芝ダ", f"{ctx.venue or ''} {s}（距離問わず）"),
        "surface": ("同芝ダ（全場）", f"{s}（競馬場・距離問わず）"),
    }


def _mask(runs: pd.DataFrame, ctx: RaceContext, scope: str) -> pd.Series:
    m = runs["surface"] == ctx.surface
    if scope in ("course", "venue_surface"):
        m &= runs["venue"] == ctx.venue
    if scope == "course":
        m &= runs["distance"] == ctx.distance
    return m


def _tally(sub: pd.DataFrame) -> dict[str, tuple[int, int, int, int]]:
    """jockey_id -> (出走, 1着, 2着, 3着)。"""
    if sub.empty:
        return {}
    f = sub["finish_position"]
    g = pd.DataFrame({"j": sub["jockey_id"].to_numpy(), "n": 1, "w1": (f == 1).to_numpy(), "w2": (f == 2).to_numpy(), "w3": (f == 3).to_numpy()})
    agg = g.groupby("j", observed=True).sum()
    return {j: (int(r.n), int(r.w1), int(r.w2), int(r.w3)) for j, r in agg.iterrows()}


class JockeyCourseModule:
    id = "jockey_course"
    label = "騎手×コース"
    description = "出走馬の騎手が、今回と同じコース・同じ競馬場・同じ芝/ダートでこれまで残してきた成績。"

    def applicable(self, ctx: RaceContext) -> bool:
        return bool(ctx.venue and ctx.surface and ctx.distance)

    def compute(self, ctx: RaceContext) -> ModuleOutput:
        labels = _scope_labels(ctx)
        columns: list[Column] = []
        groups: list[ColumnGroup] = []
        for sc in SCOPES:
            name, sub = labels[sc]
            groups.append(ColumnGroup(key=sc, label=name, sub=sub, n_key=f"{sc}_n", min_n=MIN_N))
            columns += [
                Column(f"{sc}_rec", "成績", "record", sc, "1着-2着-3着-着外"),
                Column(f"{sc}_win", "勝率", "pct", sc),
                Column(f"{sc}_top2", "連対率", "pct", sc, "2着以内の率"),
                Column(f"{sc}_top3", "複勝率", "pct", sc, "3着以内の率"),
            ]
        runs = load_runs()
        before = runs
        if ctx.date:
            before = runs[runs["date"] < ctx.date]
        before = before[before["race_id"] != ctx.race_id]
        until = before["date"].max() if len(before) else None
        source = {
            "provider": SOURCE_PROVIDER,
            "dataset": SOURCE_DATASET,
            "period_from": before["date"].min() if len(before) else None,
            "period_to": until,
            "races": int(before["race_id"].nunique()),
            "rows": int(len(before)),
        }
        ids = {str(e["jockey_id"]) for e in ctx.entries if e.get("jockey_id")}
        before = before[before["jockey_id"].isin(ids)]
        tallies = {sc: _tally(before[_mask(before, ctx, sc)]) for sc in SCOPES}

        cells: dict[str, dict[str, Any]] = {}
        for e in ctx.entries:
            jid = str(e["jockey_id"]) if e.get("jockey_id") else None
            row: dict[str, Any] = {}
            for sc in SCOPES:
                n, w1, w2, w3 = tallies[sc].get(jid, (0, 0, 0, 0)) if jid else (0, 0, 0, 0)
                row[f"{sc}_n"] = n
                row[f"{sc}_rec"] = [w1, w2, w3, n - w1 - w2 - w3] if jid else None
                row[f"{sc}_win"] = w1 / n if n else None
                row[f"{sc}_top2"] = (w1 + w2) / n if n else None
                row[f"{sc}_top3"] = (w1 + w2 + w3) / n if n else None
            cells[e["key"]] = row

        notes = [
            "このレースの開催日より前の JRA 全レースの結果から集計しています。",
            f"出走数が{MIN_N}未満の群は、参考値として薄く表示しています。",
        ]
        return ModuleOutput(columns=columns, groups=groups, cells=cells, notes=notes, data_until=until, source=source)
