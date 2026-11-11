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
from src.research.race_analysis.course_stats import _ROI_SCALE, _scale
from src.research.race_analysis.runs import SOURCE_DATASET, SOURCE_PROVIDER, drop_long_shots, load_runs, load_runs_roi

WINDOW = 8
SHOW = 5                     # 表に出す直近の走の数（平均は WINDOW 走で計算する）
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


# 位置取り帯（初角の通過順位÷頭数）。今回と同じコースの過去の全出走を、この帯ごとに集計して、各馬の平均位置が入る帯の成績を並べる
BANDS = (("最前", 0.2), ("前", 0.4), ("中", 0.6), ("後", 0.8), ("最後方", 1.0))


def band_of(ratio: float) -> int:
    """位置取り（0〜1）が入る帯の番号。"""
    for i, (_, hi) in enumerate(BANDS):
        if ratio <= hi:
            return i
    return len(BANDS) - 1


def band_label(i: int) -> str:
    lo = BANDS[i - 1][1] if i else 0.0
    name, hi = BANDS[i]
    return f"{name}（{lo * 100:.0f}〜{hi * 100:.0f}%）"


def band_stats(course_runs: pd.DataFrame) -> list[dict[str, float | int | None]]:
    """同じコースの過去の出走（pos_first・field_size あり）を位置取り帯ごとに集計。回収率は払戻が読めた出走だけで割る。"""
    out: list[dict[str, float | int | None]] = [{"n": 0, "win": None, "top3": None, "win_roi": None, "place_roi": None} for _ in BANDS]
    if course_runs.empty:
        return out
    ratio = course_runs["pos_first"].astype(float) / course_runs["field_size"].astype(float)
    idx = pd.cut(ratio, [-1.0] + [hi for _, hi in BANDS[:-1]] + [10.0], labels=False)
    for i in range(len(BANDS)):
        x = course_runs[(idx == i).to_numpy()]
        if x.empty:
            continue
        f = x["finish_position"]
        ok = x["win_ret"].notna() & x["place_ret"].notna()
        out[i] = {"n": int(len(x)), "win": float((f == 1).mean()), "top3": float((f <= 3).mean()),
                  "win_roi": float(x.loc[ok, "win_ret"].mean()) if ok.any() else None, "place_roi": float(x.loc[ok, "place_ret"].mean()) if ok.any() else None}
    return out


def _pct(v: float | None) -> str:
    return "–" if v is None else f"{v * 100:.0f}%"


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


def range_cell(stat: PositionStat, runs: list[tuple[int, int]]) -> dict[str, Any] | None:
    """直近 SHOW 走の位置取り（前から何%）の幅。外れ値の走は除き、新しい走ほど濃く（weight=DECAY^何走前）描く。mean は WINDOW 走の加重平均。"""
    ex = {i for i, _ in stat.excluded}
    pts = [(p / x, DECAY ** i) for i, (p, x) in enumerate(runs[:SHOW]) if i not in ex]
    if not pts or stat.value is None:
        return None
    return {"lo": min(r for r, _ in pts), "hi": max(r for r, _ in pts), "mean": stat.value, "pts": [[round(r, 4), round(w, 3)] for r, w in pts]}


def _runs_cell(stat: PositionStat, runs: list[tuple[int, int]]) -> list[list[int]]:
    """直近の位置取りを新しい順に [通過順位, 頭数, 外れ値として除いたか(1/0)] で。"""
    ex = {i for i, _ in stat.excluded}
    return [[p, x, 1 if i in ex else 0] for i, (p, x) in enumerate(runs[:SHOW])]


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
            "period_from": h["date"].min() if len(h) else None, "period_to": h["date"].max() if len(h) else None,   # 出走馬の履歴として使った走の範囲
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

        bands: list[dict[str, float | int | None]] | None = None
        if ctx.venue and ctx.surface and ctx.distance:
            cr = load_runs_roi()
            cr = cr[(cr["date"] < ctx.date) if ctx.date else slice(None)]
            cr = drop_long_shots(cr[(cr["race_id"] != ctx.race_id) & (cr["venue"] == ctx.venue) & (cr["surface"] == ctx.surface) & (cr["distance"] == ctx.distance)])
            bands = band_stats(cr[cr["pos_first"].notna() & cr["field_size"].notna()])
        total_n = sum(int(b["n"] or 0) for b in bands) if bands else 0

        cells: dict[str, dict[str, Any]] = {}
        for e in ctx.entries:
            st, r = stats[e["key"]], hist.get(str(e.get("horse_id") or ""), [])
            c: dict[str, Any] = {
                "style": labels[e["key"]],
                "pos": st.value,
                "range": range_cell(st, r) if r else None,
                "n": len(r),
                "hist": _runs_cell(st, r) if r else None,
                "band": None, "band_n": None, "band_win": None, "band_top3": None, "band_win_roi": None, "band_place_roi": None,
            }
            if bands and st.value is not None:
                i = band_of(st.value)
                b = bands[i]
                c.update(band=band_label(i), band_n=b["n"], band_win=b["win"], band_top3=b["top3"], band_win_roi=b["win_roi"], band_place_roi=b["place_roi"])
            cells[e["key"]] = c
        columns = [
            Column("style", "脚質", "text", None, "メンバー全体での相対的な位置（ハナ／先行(前)／先行(後ろ)／差し／追込）", palette=PALETTE),
            Column("range", f"直近{SHOW}走の位置取りの幅（前から%）", "range", None,
                   f"直近{SHOW}走の初角位置（通過順位÷頭数。0%=先頭、100%=最後方）の最前〜最後方の幅。点は各走で、新しい走ほど濃く表示。縦線は直近{WINDOW}走を新しい走ほど重くした平均。外れ値の走は除く"),
            Column("n", "走数", "int", None, f"使った直近の走数（最大{WINDOW}）。少ないほど信頼できません"),
            Column("hist", f"直近{SHOW}走の初角通過順位（左が前走 → 右が{SHOW}走前）", "runs", None,
                   f"左端が前走（枠つき）、右へ 2走前・3走前…{SHOW}走前。数字は初角の通過順位、/の後は頭数。色は前（赤）〜後ろ（青）。点線の走は外れ値として平均から除いた走"),
        ]
        if bands:
            avg = {k: (sum((b[k] or 0) * (b["n"] or 0) for b in bands) / total_n if total_n else None) for k in ("win", "top3")}
            columns += [
                Column("band", "位置取り帯", "text", None, "平均位置が入る帯（初角の通過順位÷頭数。最前=前から0〜20%）。この帯は、今回と同じコースの過去の全出走を位置取りで分けたもの"),
                Column("band_n", "帯の走数", "int", None, "今回と同じコースで、その位置取り帯だった過去の出走数（単勝50倍以上は除く）"),
                Column("band_win", "帯の勝率", "pct", None, "同じコース・同じ位置取り帯の勝率" + (f"（このコース全体の平均 {avg['win'] * 100:.1f}%）" if avg["win"] else ""), _scale(avg["win"], "win")),
                Column("band_top3", "帯の複勝率", "pct", None, "同じコース・同じ位置取り帯の複勝率" + (f"（このコース全体の平均 {avg['top3'] * 100:.1f}%）" if avg["top3"] else ""), _scale(avg["top3"], "top3")),
                Column("band_win_roi", "帯の単勝回収率", "pct", None, "同じコース・同じ位置取り帯に100円ずつ買い続けた単勝の戻り（100%超が緑）", _ROI_SCALE),
                Column("band_place_roi", "帯の複勝回収率", "pct", None, "同じコース・同じ位置取り帯に100円ずつ買い続けた複勝の戻り（100%超が緑）", _ROI_SCALE),
            ]
        notes = [
            f"位置取り=初角（最初のコーナー）の通過順位÷頭数。直近{WINDOW}走を、新しい走ほどやや重く（1走前ごとに×{DECAY}）平均しています。"
            f"表に並べるのは直近{SHOW}走（左が前走）で、位置取りの幅の点は新しい走ほど濃く描きます。",
            f"履歴が{MIN_OUTLIER_RUNS}走以上ある馬は、偶然の出遅れ・一度だけの逃げなどの外れ値（点線の走）を除いて平均しています。",
            "脚質はこのレースのメンバーの中での相対評価です。ハナは原則1頭で、先頭2・3番手との差が誤差の範囲のときだけ最大3頭まで増えます。",
            "走数が少ない馬（初出走は分類なし）は、位置取りが安定していないため参考程度に見てください。",
        ]
        if bands:
            notes.append(f"今回のコース（{ctx.venue} {ctx.surface}{ctx.distance}m）の位置取り帯別の成績（走数 勝率 複勝率 単勝回収率 複勝回収率。単勝50倍以上は除く）: "
                         + " ／ ".join(f"{band_label(i)} {b['n']}走 {_pct(b['win'])} {_pct(b['top3'])} {_pct(b['win_roi'])} {_pct(b['place_roi'])}" for i, b in enumerate(bands)))
        return ModuleOutput(columns=columns, groups=[], cells=cells, notes=notes, data_until=until, source=source)
