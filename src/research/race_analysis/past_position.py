"""過去位置取り: 各馬の直近の序盤の位置取りから、メンバーの中での脚質（ハナ・先行(前)・先行(後ろ)・差し・追込）を出す。

位置取り = 初角（そのレースで最初に通過するコーナー）の通過順位 N ÷ 頭数 X（0に近いほど前）。馬ごとに直近 WINDOW 走を、新しい走ほどやや重い重み（DECAY^何走前）で加重平均する。
外れ値（偶然の出遅れ・一度だけの逃げなど）は、その馬の履歴が MIN_OUTLIER_RUNS 走以上あるときだけ除く
（中央値から MAD の OUTLIER_K 倍以上離れた走。除くのは履歴の MAX_EXCLUDE_FRAC まで）。履歴が少ない馬は、全走をそのまま使う。
5 段階はメンバー全体での相対順位で決める（順位を前から 1/3 ずつ 先行・差し・追込 に分け、先行をハナ・前・後ろに分ける）:
  ハナ        : 最も前の馬。2・3番目の馬も、先頭の馬との差が推定誤差の CO_LEADER_Z 倍未満（= ほぼ差が無い）なら加える。最大 MAX_LEADERS 頭
  先行(前)    : 先行の三分の一に入る、ハナ以外の前半
  先行(後ろ) : 〃 後半
  差し        : 順位が全体の 1/3〜2/3
  追込        : 2/3 より後ろ
障害と平地は別に数える。履歴が無い馬（初出走など）は分類しない。
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.research.race_analysis.base import Column, ModuleOutput, RaceContext
from src.research.race_analysis.runs import SOURCE_DATASET, SOURCE_PROVIDER, load_runs

WINDOW = 8
DECAY = 0.85
MIN_OUTLIER_RUNS = 5
OUTLIER_K = 4.0
MIN_SCALE = 0.08
MAX_EXCLUDE_FRAC = 0.2
SIGMA_WITHIN = 0.15          # 同じ馬の走ごとの位置取りのばらつき（標準偏差）。ハナの同着判定の誤差に使う（検証値は docs/設計メモ参照）
CO_LEADER_Z = 0.15
MAX_LEADERS = 3
LEADER = "ハナ"
UNKNOWN = "不明"
PALETTE = {
    "ハナ": "rgba(248,81,73,0.55)",
    "先行(前)": "rgba(240,131,0,0.45)",
    "先行(後ろ)": "rgba(210,153,34,0.40)",
    "差し": "rgba(96,165,250,0.35)",
    "追込": "rgba(59,130,246,0.45)",
}


@dataclass
class PositionStat:
    value: float | None                       # 加重平均の位置取り（0〜1。小さいほど前）
    se: float | None                          # 推定誤差の目安
    used: list[tuple[int, float]] = field(default_factory=list)       # 使った走（何走前, 位置取り）
    excluded: list[tuple[int, float]] = field(default_factory=list)   # 外れ値として除いた走


def split_outliers(ratios: list[float]) -> set[int]:
    """外れ値の添字。履歴が MIN_OUTLIER_RUNS 走未満なら除かない。"""
    n = len(ratios)
    if n < MIN_OUTLIER_RUNS:
        return set()
    med = statistics.median(ratios)
    mad = statistics.median(abs(r - med) for r in ratios)
    scale = max(1.4826 * mad, MIN_SCALE)
    far = sorted((i for i, r in enumerate(ratios) if abs(r - med) > OUTLIER_K * scale), key=lambda i: -abs(ratios[i] - med))
    return set(far[: int(n * MAX_EXCLUDE_FRAC)])


def horse_position(ratios_latest_first: list[float]) -> PositionStat:
    """直近の位置取り（新しい順）から、外れ値を除いた加重平均と推定誤差を出す。"""
    rs = ratios_latest_first[:WINDOW]
    if not rs:
        return PositionStat(None, None)
    out = split_outliers(rs)
    used = [(i, r) for i, r in enumerate(rs) if i not in out]
    ws = [DECAY ** i for i, _ in used]
    value = sum(w * r for w, (_, r) in zip(ws, used)) / sum(ws)
    n_eff = sum(ws) ** 2 / sum(w * w for w in ws)
    return PositionStat(value, SIGMA_WITHIN / math.sqrt(n_eff), used, [(i, rs[i]) for i in sorted(out)])


def classify(field_stats: dict[str, PositionStat]) -> dict[str, str]:
    """メンバー全員の位置取りから 5 段階に分ける。値が無い馬は UNKNOWN。"""
    known = sorted(((k, s) for k, s in field_stats.items() if s.value is not None), key=lambda kv: (kv[1].value, kv[0]))
    out = {k: UNKNOWN for k in field_stats}
    n = len(known)
    if n == 0:
        return out
    lead_k, lead = known[0]
    leaders = {lead_k}
    for k, s in known[1:MAX_LEADERS]:
        z = (s.value - lead.value) / math.sqrt(lead.se ** 2 + s.se ** 2)
        if z >= CO_LEADER_Z:
            break
        leaders.add(k)
    # 順位の三分割: 前の 1/3 = 先行（ハナ + 先行(前) + 先行(後ろ)）、真ん中の 1/3 = 差し、後ろの 1/3 = 追込。先行の中のハナ以外は前半を先行(前)、後半を先行(後ろ)にする
    h = len(leaders)
    front_end = max(h, round(n / 3))
    mid_end = max(front_end, round(2 * n / 3))
    front_split = h + math.ceil((front_end - h) / 2)
    for i, (k, _) in enumerate(known):
        if k in leaders:
            out[k] = LEADER
        elif i < front_split:
            out[k] = "先行(前)"
        elif i < front_end:
            out[k] = "先行(後ろ)"
        elif i < mid_end:
            out[k] = "差し"
        else:
            out[k] = "追込"
    return out


def _fmt_hist(stat: PositionStat, runs: list[tuple[int, int]], limit: int = 6) -> str:
    """直近の位置取りを『通過順/頭数』で（新しい順）。除いた外れ値は [ ] で囲む。"""
    ex = {i for i, _ in stat.excluded}
    parts = [f"[{p}/{x}]" if i in ex else f"{p}/{x}" for i, (p, x) in enumerate(runs[:limit])]
    return " ".join(parts)


class PastPositionModule:
    id = "past_position"
    label = "過去位置取り"
    description = "直近の初角通過順位（通過順位÷頭数）を、新しい走ほどやや重く平均して、メンバーの中でのハナ・先行・差し・追込に分けます。"

    def applicable(self, ctx: RaceContext) -> bool:
        return bool(ctx.entries)

    def compute(self, ctx: RaceContext) -> ModuleOutput:
        runs = load_runs()
        dated = runs
        if ctx.date:
            dated = dated[dated["date"] < ctx.date]
        dated = dated[dated["race_id"] != ctx.race_id]
        until = dated["date"].max() if len(dated) else None
        hurdle = (ctx.surface or "").startswith("障")
        ids = {str(e["horse_id"]) for e in ctx.entries if e.get("horse_id")}
        h = dated[dated["horse_id"].isin(ids) & dated["pos_first"].notna() & dated["field_size"].notna()]
        h = h[(h["surface"].astype(str).str.startswith("障")) == hurdle].sort_values(["date", "race_id"], ascending=False)
        source = {
            "provider": SOURCE_PROVIDER, "dataset": SOURCE_DATASET,
            "period_from": dated["date"].min() if len(dated) else None, "period_to": until,
            "races": int(h["race_id"].nunique()), "rows": int(len(h)),
        }
        hist: dict[str, list[tuple[int, int]]] = {}
        for hid, grp in h.groupby("horse_id", sort=False):
            hist[str(hid)] = [(int(p), int(x)) for p, x in zip(grp["pos_first"].head(WINDOW), grp["field_size"].head(WINDOW))]
        stats: dict[str, PositionStat] = {}
        for e in ctx.entries:
            r = hist.get(str(e.get("horse_id") or ""), [])
            stats[e["key"]] = horse_position([p / x for p, x in r])
        labels = classify(stats)

        cells: dict[str, dict[str, Any]] = {}
        for e in ctx.entries:
            st, r = stats[e["key"]], hist.get(str(e.get("horse_id") or ""), [])
            cells[e["key"]] = {
                "style": labels[e["key"]],
                "pos": st.value,
                "n": len(r),
                "hist": _fmt_hist(st, r) if r else None,
            }
        columns = [
            Column("style", "脚質", "text", None, "メンバー全体での相対的な位置（ハナ／先行(前)／先行(後ろ)／差し／追込）", palette=PALETTE),
            Column("pos", "平均位置", "pct", None, "直近の位置取りの加重平均（前から何%。小さいほど前）"),
            Column("n", "走数", "int", None, f"使った直近の走数（最大{WINDOW}）。少ないほど信頼できません"),
            Column("hist", "直近の初角通過順位/頭数", "text", None, "新しい順。[ ] は外れ値として平均から除いた走"),
        ]
        notes = [
            f"位置取り=初角（最初のコーナー）の通過順位÷頭数。直近{WINDOW}走を、新しい走ほどやや重く（1走前ごとに×{DECAY}）平均しています。",
            f"履歴が{MIN_OUTLIER_RUNS}走以上ある馬は、偶然の出遅れ・一度だけの逃げなどの外れ値（[ ] の走）を除いて平均しています。",
            "脚質はこのレースのメンバーの中での相対評価です。ハナは原則1頭で、先頭2・3番手との差が誤差の範囲のときだけ最大3頭まで増えます。",
            "走数が少ない馬（初出走は分類なし）は、位置取りが安定していないため参考程度に見てください。",
        ]
        return ModuleOutput(columns=columns, groups=[], cells=cells, notes=notes, data_until=until, source=source)
