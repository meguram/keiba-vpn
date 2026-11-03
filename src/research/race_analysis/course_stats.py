"""『○○ × コース』の成績。出走馬ごとに、今回と同じ条件でのこれまでの成績を 2 つの広さで出す。

  同コース           : 同じ競馬場・同じ芝/ダ・同じ距離
  同競馬場・同芝ダ   : 距離は問わない

○○（軸）は Dimension で差し替える: 騎手 / 枠 / 種牡馬 / 種牡馬の系統。軸を足すときは Dimension を 1 つ作って registry.py に足す。
対象は、このレースの開催日より前の JRA 全レース（当日以降の結果は入れない）。キーは ID（名前では結合しない）。
各馬の行からは、その馬自身の過去の出走を除く（父の産駒成績に自分の成績が混ざり、出走の少ない父で個体の成績がそのまま出てしまうのを避ける）。
色の基準は、その条件の全馬の平均（勝率・連対率・複勝率）。
"""

from __future__ import annotations

import re
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


def _line_key(e: dict[str, Any], aux: dict[str, Any]) -> str | None:
    row = aux.get("sire", {}).get(str(e.get("horse_id") or ""))
    return str(row[2]) if row and len(row) > 2 and row[2] else None


def _line_text(e: dict[str, Any], aux: dict[str, Any]) -> str | None:
    row = aux.get("sire", {}).get(str(e.get("horse_id") or ""))
    return str(row[3]) if row and len(row) > 3 and row[3] else None


JOCKEY = Dimension(
    id="jockey_course", label="騎手×コース", min_n=10, run_col="jockey_id", entry_key=_jockey_key, key_header="騎手", key_text=_jockey_text,
    description="出走馬の騎手が、今回と同じコース・同じ競馬場・同じ芝/ダートでこれまで残してきた成績。",
)
GATE = Dimension(
    id="gate_course", label="枠×コース", min_n=30, run_col="bracket_number", entry_key=_bracket_key, key_header="枠", key_text=_bracket_text,
    description="出走馬の枠（1〜8枠）が、今回と同じコース・同じ競馬場・同じ芝/ダートでこれまで残してきた成績。馬の強さに関係なく、枠そのものの傾向です。",
    unknown_note="枠順がまだ確定していない馬は、確定後に表示されます。",
)
SIRE = Dimension(
    id="sire_course", label="種牡馬×コース", min_n=10, run_col="sire_id", entry_key=_sire_key, key_header="父（種牡馬）", key_text=_sire_text,
    needs_sire_map=True,
    description="出走馬の父（種牡馬）の産駒が、今回と同じコース・同じ競馬場・同じ芝/ダートでこれまで残してきた成績。",
    unknown_note="父を特定できない馬は表示されません。",
)
SIRE_LINE = Dimension(
    id="sire_line_course", label="系統×コース", min_n=30, run_col="sire_line_id", entry_key=_line_key, key_header="父の系統", key_text=_line_text,
    needs_sire_map=True,
    description="出走馬の父の系統（サンデーサイレンス系など）の産駒の成績。",
    unknown_note="父の系統を特定できない馬は表示されません。系統は父から父の父…（5代まで）をさかのぼって決めています。",
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


def _tally_own(sub: pd.DataFrame, col: str, horse_ids: set[str]) -> dict[tuple[str, str], tuple[int, int, int, int]]:
    """(キー, 馬) -> (出走, 1着, 2着, 3着)。出走馬自身の分を各馬の集計から差し引くため。"""
    sub = sub[sub["horse_id"].astype(str).isin(horse_ids)]
    if sub.empty:
        return {}
    f = sub["finish_position"]
    g = pd.DataFrame({"k": sub[col].astype(str).to_numpy(), "h": sub["horse_id"].astype(str).to_numpy(), "n": 1, "w1": (f == 1).to_numpy(), "w2": (f == 2).to_numpy(), "w3": (f == 3).to_numpy()})
    agg = g.groupby(["k", "h"]).sum()
    return {(str(k), str(h)): (int(r.n), int(r.w1), int(r.w2), int(r.w3)) for (k, h), r in agg.iterrows()}


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


def lines_ready() -> bool:
    """系統の群を出してよいか = netkeiba の『○○系』を全種牡馬ぶん取り終えている。"""
    from src.scraper import sire_line_store

    return sire_line_store.is_complete()


@dataclass(frozen=True)
class Block:
    """表の 1 群 = ある軸（Dimension）を、ある範囲（scope）で集計したもの。列のキーは『<key>_rec』のように key を前置する。"""
    dim: Dimension
    scope: str
    key: str
    label: str | None = None          # 群の見出し（省略すると範囲の名前: 同コース／同競馬場・同芝ダ）
    extra_text: str | None = None     # 群の先頭に出す文字の列の見出し（例: 系統名）
    needs_lines: bool = False         # netkeiba の系統の取得が完全に終わるまでは出さない群


def default_blocks(dim: Dimension) -> tuple[Block, ...]:
    return tuple(Block(dim, sc, sc) for sc in SCOPES)


class CourseStatsModule:
    def __init__(self, dim: Dimension, blocks: tuple[Block, ...] | None = None, *, label: str | None = None, description: str | None = None) -> None:
        self.dim = dim
        self.all_blocks = blocks or default_blocks(dim)
        self.id = dim.id
        self.label = label or dim.label
        self.description = description or dim.description

    @property
    def blocks(self) -> tuple[Block, ...]:
        if lines_ready():
            return self.all_blocks
        return tuple(b for b in self.all_blocks if not b.needs_lines)

    def applicable(self, ctx: RaceContext) -> bool:
        if any(b.dim.needs_sire_map for b in self.blocks) and not sire_map_available():
            return False
        return bool(ctx.venue and ctx.surface and ctx.distance)

    def _aux(self) -> dict[str, Any]:
        if not any(b.dim.needs_sire_map for b in self.blocks):
            return {}
        m = load_sire_map()
        if m is None:
            return {"sire": {}}
        for c in ("sire_line_id", "sire_line_name"):
            if c not in m.columns:
                m = m.assign(**{c: None})
        cols = ["horse_id", "sire_id", "sire_name", "sire_line_id", "sire_line_name"]
        return {"sire": {str(r.horse_id): (r.sire_id, r.sire_name, r.sire_line_id, r.sire_line_name) for r in m[cols].itertuples(index=False)}}

    def compute(self, ctx: RaceContext) -> ModuleOutput:
        dim = self.dim
        blocks = self.blocks
        labels = _scope_labels(ctx)
        dated = load_runs_with_sire() if any(b.dim.needs_sire_map for b in blocks) else load_runs()
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

        aux = self._aux()
        columns: list[Column] = []
        groups: list[ColumnGroup] = []
        cells: dict[str, dict[str, Any]] = {e["key"]: {} for e in ctx.entries}
        unknown = False
        for b in blocks:
            bd, sc = b.dim, b.scope
            name, sub = labels[sc]
            base = _baseline(dated[_mask(dated, ctx, sc)])
            groups.append(ColumnGroup(key=b.key, label=b.label or name, sub=sub, n_key=f"{b.key}_n", min_n=bd.min_n))
            if b.extra_text:
                columns.append(Column(f"{b.key}_name", b.extra_text, "text", b.key, None))
            columns.append(Column(f"{b.key}_rec", "成績", "record", b.key, "1着-2着-3着-着外"))
            for key, label, desc in (("win", "勝率", None), ("top2", "連対率", "2着以内の率"), ("top3", "複勝率", "3着以内の率")):
                mid = (base or {}).get(key)
                hint = f"{desc}。" if desc else ""
                title = f"{hint}この条件の全馬の平均は {mid * 100:.1f}%（緑=平均より高い／赤=低い）" if mid else desc
                columns.append(Column(f"{b.key}_{key}", label, "pct", b.key, title, _scale(mid, key)))

            keys = {e["key"]: bd.entry_key(e, aux) for e in ctx.entries}
            unknown = unknown or any(k is None for k in keys.values())
            wanted = {k for k in keys.values() if k}
            sub_all = dated[dated[bd.run_col].astype(str).isin(wanted)] if wanted else dated.iloc[0:0]
            scoped = sub_all[_mask(sub_all, ctx, sc)]
            tally = _tally(scoped, bd.run_col)
            own = _tally_own(scoped, bd.run_col, {str(e.get("horse_id")) for e in ctx.entries if e.get("horse_id")})
            for e in ctx.entries:
                k = keys[e["key"]]
                row = cells[e["key"]]
                n, w1, w2, w3 = tally.get(k, (0, 0, 0, 0)) if k else (0, 0, 0, 0)
                if k and e.get("horse_id"):
                    on, o1, o2, o3 = own.get((k, str(e["horse_id"])), (0, 0, 0, 0))
                    n, w1, w2, w3 = n - on, w1 - o1, w2 - o2, w3 - o3
                row[f"{b.key}_n"] = n
                row[f"{b.key}_rec"] = [w1, w2, w3, n - w1 - w2 - w3] if k else None
                row[f"{b.key}_win"] = w1 / n if n else None
                row[f"{b.key}_top2"] = (w1 + w2) / n if n else None
                row[f"{b.key}_top3"] = (w1 + w2 + w3) / n if n else None
                if b.extra_text and bd.key_text:
                    row[f"{b.key}_name"] = bd.key_text(e, aux)
                if b is blocks[0] and bd.key_text:
                    row["_key"] = bd.key_text(e, aux)

        mins = " ・ ".join(f"{b.label or labels[b.scope][0]}は{b.dim.min_n}" for b in blocks) if len(blocks) > 1 and len({b.dim.min_n for b in blocks}) > 1 else None
        notes = [
            "このレースの開催日より前の JRA 全レースの結果から集計しています。各馬の行は、その馬自身の過去の出走を除いた成績です。",
            f"出走数が少ない群（{mins}未満）は、参考値として薄く表示し、色は付けません。" if mins else f"出走数が{blocks[0].dim.min_n}未満の群は、参考値として薄く表示し、色は付けません。",
            "セルの色: その条件の全馬の平均より高いほど緑、低いほど赤（濃いほど差が大きい）。",
        ]
        for d in dict.fromkeys(b.dim for b in blocks):
            if d.unknown_note and unknown:
                notes.append(d.unknown_note)
        return ModuleOutput(columns=columns, groups=groups, cells=cells, notes=notes, data_until=until, source=source, key_header=dim.key_header)
