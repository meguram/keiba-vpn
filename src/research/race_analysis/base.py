"""分析モジュールの共通の型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

# 列の表示形式（frontend/lib/raceAnalysis.ts の formatCell と対応させる）
#   record = [1着, 2着, 3着, 着外] を "1-2-3-4" に / pct = 0〜1 を % に / int / text
FORMATS = ("record", "pct", "int", "text")


@dataclass(frozen=True)
class Column:
    key: str
    label: str
    fmt: str = "text"
    group: str | None = None
    title: str | None = None       # ホバーで出す説明


@dataclass(frozen=True)
class ColumnGroup:
    key: str
    label: str
    sub: str | None = None         # 見出しの下に出す補足（条件など）
    n_key: str | None = None       # この群の標本数を持つ列のキー
    min_n: int = 0                 # 標本数がこれ未満の群は薄く表示する


@dataclass
class RaceContext:
    """分析に渡すレースの情報。分析は、このレースの開催日より前のデータだけを使うこと（リーク防止）。"""

    race_id: str
    date: str | None               # "YYYY-MM-DD"
    venue: str | None
    surface: str | None
    distance: int | None
    entries: list[dict[str, Any]] = field(default_factory=list)   # 出走馬。key は馬ごとの一意キー


@dataclass
class ModuleOutput:
    columns: list[Column]
    groups: list[ColumnGroup]
    cells: dict[str, dict[str, Any]]      # entry key -> {column key: 値}
    notes: list[str] = field(default_factory=list)
    data_until: str | None = None         # 使ったデータの最終日


class AnalysisModule(Protocol):
    id: str
    label: str
    description: str

    def applicable(self, ctx: RaceContext) -> bool: ...
    def compute(self, ctx: RaceContext) -> ModuleOutput: ...
