"""騎手 × コースの成績。各出走馬の騎手が、今回と同じ条件でこれまでどんな成績かを 2 つの広さで出す。

  同コース           : 同じ競馬場・同じ芝/ダ・同じ距離
  同競馬場・同芝ダ   : 距離は問わない

対象は、このレースの開催日より前の JRA 全レース（2020 年以降。当日以降の結果は入れない）。キーは jockey_id（名前では結合しない）。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.research.race_analysis.base import Column, ColumnGroup, ModuleOutput, RaceContext
from src.research.race_analysis.runs import SOURCE_DATASET, SOURCE_PROVIDER, load_runs

MIN_N = 10   # これ未満の出走数は参考値として薄く表示する

SCOPES = ("course", "venue_surface")


def _scope_labels(ctx: RaceContext) -> dict[str, tuple[str, str]]:
    s = ctx.surface or ""
    return {
        "course": ("同コース", f"{ctx.venue or ''} {s}{ctx.distance or ''}m"),
        "venue_surface": ("同競馬場・同芝ダ", f"{ctx.venue or ''} {s}（距離問わず）"),
    }


def _mask(runs: pd.DataFrame, ctx: RaceContext, scope: str) -> pd.Series:
    m = runs["surface"] == ctx.surface
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


# 平均に対する倍率で色の端を決める（lo=平均×0.25 で最も赤、hi=平均×倍率 で最も緑。hi は 100% が上限）
_HEAT = {"win": 3.0, "top2": 2.4, "top3": 2.0}


def _baseline(sub: pd.DataFrame) -> dict[str, float] | None:
    """その条件の全騎手の平均（勝率・連対率・複勝率）。色の基準。"""
    n = len(sub)
    if n == 0:
        return None
    f = sub["finish_position"]
    return {"win": float((f == 1).sum()) / n, "top2": float((f <= 2).sum()) / n, "top3": float((f <= 3).sum()) / n}


def _scale(mid: float | None, key: str) -> dict[str, float] | None:
    if not mid:
        return None
    return {"lo": mid * 0.25, "mid": mid, "hi": min(1.0, mid * _HEAT[key])}


class JockeyCourseModule:
    id = "jockey_course"
    label = "騎手×コース"
    description = "出走馬の騎手が、今回と同じコース・同じ競馬場・同じ芝/ダートでこれまで残してきた成績。"

    def applicable(self, ctx: RaceContext) -> bool:
        return bool(ctx.venue and ctx.surface and ctx.distance)

    def compute(self, ctx: RaceContext) -> ModuleOutput:
        labels = _scope_labels(ctx)
        dated = load_runs()
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
            groups.append(ColumnGroup(key=sc, label=name, sub=sub, n_key=f"{sc}_n", min_n=MIN_N))
            columns.append(Column(f"{sc}_rec", "成績", "record", sc, "1着-2着-3着-着外"))
            for key, label, desc in (("win", "勝率", None), ("top2", "連対率", "2着以内の率"), ("top3", "複勝率", "3着以内の率")):
                mid = (baselines[sc] or {}).get(key)
                hint = f"{desc}。" if desc else ""
                title = f"{hint}この条件の全騎手の平均は {mid * 100:.1f}%（緑=平均より高い／赤=低い）" if mid else desc
                columns.append(Column(f"{sc}_{key}", label, "pct", sc, title, _scale(mid, key)))

        ids = {str(e["jockey_id"]) for e in ctx.entries if e.get("jockey_id")}
        before = dated[dated["jockey_id"].isin(ids)]
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
            f"出走数が{MIN_N}未満の群は、参考値として薄く表示し、色は付けません。",
            "セルの色: その条件の全騎手の平均より高いほど緑、低いほど赤（濃いほど差が大きい）。",
        ]
        return ModuleOutput(columns=columns, groups=groups, cells=cells, notes=notes, data_until=until, source=source)
