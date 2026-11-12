"""過去位置取り: 各馬の直近の序盤の位置取りから、脚質（ハナ・先行(前)・先行(後ろ)・差し(前)・差し(後ろ)・追込）を出す。

位置取り = 初角（そのレースで最初に通過するコーナー）の通過順位 N ÷ 頭数 X（0に近いほど前）。馬ごとに直近 WINDOW 走を、新しい走ほどやや重い重み（DECAY^何走前）で加重平均する。
外れ値（偶然の出遅れ・一度だけの逃げなど）は、その馬の履歴が MIN_OUTLIER_RUNS 走以上あるときだけ除く
（中央値から MAD の OUTLIER_K 倍以上離れた走。除くのは履歴の MAX_EXCLUDE_FRAC まで）。履歴が少ない馬は、全走をそのまま使う。
脚質は位置取り帯と 1 対 1（6 段階）。ハナだけはメンバーの中での相対で、それ以外は加重平均の位置が入る帯で決める:
  ハナ        : 最も前の馬。2・3番目の馬も、先頭の馬との差が推定誤差の CO_LEADER_Z 倍未満（= ほぼ差が無い）なら加える。最大 MAX_LEADERS 頭
  先行(前)    : ハナ以外で、平均の位置が前から 0〜20%
  先行(後ろ) : 20〜40%
  差し(前)    : 40〜60%
  差し(後ろ) : 60〜80%
  追込        : 80〜100%
同じコースの過去の成績も同じ 6 区分で集計する（過去の走のハナ = 初角を 1 番手で通過した走。それ以外は通過順位÷頭数の帯）。
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
# 位置取り帯（初角の通過順位÷頭数。上端を含む）と脚質の対応。ハナは帯の外で別に決める。今回と同じコースの過去の全出走も、この 6 区分で集計する
BANDS = (("先行(前)", 0.2), ("先行(後ろ)", 0.4), ("差し(前)", 0.6), ("差し(後ろ)", 0.8), ("追込", 1.0))
STYLES = (LEADER,) + tuple(name for name, _ in BANDS)
PALETTE = {
    "ハナ": "rgba(248,81,73,0.55)",
    "先行(前)": "rgba(240,131,0,0.45)",
    "先行(後ろ)": "rgba(210,153,34,0.40)",
    "差し(前)": "rgba(96,165,250,0.30)",
    "差し(後ろ)": "rgba(96,165,250,0.45)",
    "追込": "rgba(59,130,246,0.55)",
}


def band_of(ratio: float) -> str:
    """位置取り（0〜1）が入る帯の脚質名（ハナ以外）。"""
    for name, hi in BANDS:
        if ratio <= hi:
            return name
    return BANDS[-1][0]


def band_label(style: str) -> str:
    """脚質と帯の範囲（注記用）。"""
    if style == LEADER:
        return f"{LEADER}（初角1番手）"
    i = [name for name, _ in BANDS].index(style)
    lo = BANDS[i - 1][1] if i else 0.0
    return f"{style}（{lo * 100:.0f}〜{BANDS[i][1] * 100:.0f}%）"


def band_stats(course_runs: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    """同じコースの過去の出走（pos_first・field_size あり）を脚質の 6 区分で集計。初角 1 番手がハナで、それ以外は通過順位÷頭数の帯。
    回収率は払戻が読めた出走だけで割る。"""
    out: dict[str, dict[str, float | int | None]] = {k: {"n": 0, "win": None, "top3": None, "win_roi": None, "place_roi": None} for k in STYLES}
    if course_runs.empty:
        return out
    pos = course_runs["pos_first"].astype(float)
    key = (pos / course_runs["field_size"].astype(float)).map(band_of).where(pos != 1, LEADER).to_numpy()
    for k in STYLES:
        x = course_runs[key == k]
        if x.empty:
            continue
        f = x["finish_position"]
        ok = x["win_ret"].notna() & x["place_ret"].notna()
        out[k] = {"n": int(len(x)), "win": float((f == 1).mean()), "top3": float((f <= 3).mean()),
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
    """メンバー全員の位置取りから 6 段階に分ける（ハナはメンバー内の相対、それ以外は平均の位置が入る帯）。値が無い馬は UNKNOWN。"""
    known = sorted(((k, s) for k, s in field_stats.items() if s.value is not None), key=lambda kv: (kv[1].value, kv[0]))
    out = {k: UNKNOWN for k in field_stats}
    if not known:
        return out
    lead_k, lead = known[0]
    leaders = {lead_k}
    for k, s in known[1:MAX_LEADERS]:
        z = (s.value - lead.value) / math.sqrt(lead.se ** 2 + s.se ** 2)
        if z >= CO_LEADER_Z:
            break
        leaders.add(k)
    for k, s in known:
        out[k] = LEADER if k in leaders else band_of(s.value)
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
    description = "直近の初角通過順位（通過順位÷頭数）を、新しい走ほどやや重く平均して、ハナ（メンバーで最も前）と前から 20% ごとの脚質に分け、同じコースでの同じ脚質の成績を並べます。"

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

        bands: dict[str, dict[str, float | int | None]] | None = None
        if ctx.venue and ctx.surface and ctx.distance:
            cr = load_runs_roi()
            cr = cr[(cr["date"] < ctx.date) if ctx.date else slice(None)]
            cr = drop_long_shots(cr[(cr["race_id"] != ctx.race_id) & (cr["venue"] == ctx.venue) & (cr["surface"] == ctx.surface) & (cr["distance"] == ctx.distance)])
            bands = band_stats(cr[cr["pos_first"].notna() & cr["field_size"].notna()])
        total_n = sum(int(b["n"] or 0) for b in bands.values()) if bands else 0

        cells: dict[str, dict[str, Any]] = {}
        for e in ctx.entries:
            st, r = stats[e["key"]], hist.get(str(e.get("horse_id") or ""), [])
            c: dict[str, Any] = {
                "style": labels[e["key"]],
                "pos": st.value,
                "range": range_cell(st, r) if r else None,
                "n": len(r),
                "hist": _runs_cell(st, r) if r else None,
                "band_win": None, "band_top3": None, "band_win_roi": None, "band_place_roi": None,
            }
            if bands and labels[e["key"]] in bands:      # 脚質と同じ区分の成績（脚質と帯は 1 対 1。不明は空）
                b = bands[labels[e["key"]]]
                c.update(band_win=b["win"], band_top3=b["top3"], band_win_roi=b["win_roi"], band_place_roi=b["place_roi"])
            cells[e["key"]] = c
        columns = [
            Column("style", "脚質", "text", None, "ハナ＝メンバーで最も前。それ以外は平均の位置（前から%）の帯: 先行(前) 0〜20%／先行(後ろ) 20〜40%／差し(前) 40〜60%／差し(後ろ) 60〜80%／追込 80〜100%", palette=PALETTE),
            Column("range", f"直近{SHOW}走の位置（前から%）", "range", None,
                   f"直近{SHOW}走の初角位置（通過順位÷頭数。0%=先頭、100%=最後方）の最前〜最後方の幅。点は各走で、新しい走ほど濃く表示。縦線は直近{WINDOW}走を新しい走ほど重くした平均。外れ値の走は除く"),
            Column("n", "走数", "int", None, f"使った直近の走数（最大{WINDOW}）。少ないほど信頼できません"),
            Column("hist", f"直近{SHOW}走の初角通過順位（左が前走 → 右が{SHOW}走前）", "runs", None,
                   f"左端が前走（枠つき）、右へ 2走前・3走前…{SHOW}走前。数字は初角の通過順位、/の後は頭数。色は前（赤）〜後ろ（青）。点線の走は外れ値として平均から除いた走"),
        ]
        if bands:
            avg = {k: (sum((b[k] or 0) * (b["n"] or 0) for b in bands.values()) / total_n if total_n else None) for k in ("win", "top3")}
            same = ("今回と同じコースで、初角の位置がこの馬の脚質と同じ区分（ハナ＝初角1番手、それ以外は通過順位÷頭数の帯）だった過去の出走"
                    "（ハナの馬は、平均の位置によらず初角1番手で通過した走の成績）")
            columns += [
                Column("band_win", "勝率", "pct", None, f"{same}の勝率" + (f"（このコース全体の平均 {avg['win'] * 100:.1f}%）" if avg["win"] else ""), _scale(avg["win"], "win")),
                Column("band_top3", "複勝率", "pct", None, f"{same}の複勝率" + (f"（このコース全体の平均 {avg['top3'] * 100:.1f}%）" if avg["top3"] else ""), _scale(avg["top3"], "top3")),
                Column("band_win_roi", "単勝回収率", "pct", None, f"{same}に100円ずつ買い続けた単勝の戻り（単勝50倍以上は除く。100%超が緑）", _ROI_SCALE),
                Column("band_place_roi", "複勝回収率", "pct", None, f"{same}に100円ずつ買い続けた複勝の戻り（単勝50倍以上は除く。100%超が緑）", _ROI_SCALE),
            ]
        notes = [
            f"位置取り=初角（最初のコーナー）の通過順位÷頭数。直近{WINDOW}走を、新しい走ほどやや重く（1走前ごとに×{DECAY}）平均しています。"
            f"表に並べるのは直近{SHOW}走（左が前走）で、位置取りの幅の点は新しい走ほど濃く描きます。",
            f"履歴が{MIN_OUTLIER_RUNS}走以上ある馬は、偶然の出遅れ・一度だけの逃げなどの外れ値（点線の走）を除いて平均しています。",
            "脚質のハナはこのレースのメンバーで最も前の馬です（原則1頭。先頭2・3番手との差が誤差の範囲のときだけ最大3頭）。"
            "それ以外は平均の位置（前から%）で、先行(前) 0〜20%／先行(後ろ) 20〜40%／差し(前) 40〜60%／差し(後ろ) 60〜80%／追込 80〜100% です。",
            "位置取りマトリックスは、平均の位置を横（左が先頭）に、1頭分の体長を1マスとして並べます。重ならない馬はラチ沿い、重なるときは馬番の小さい馬からラチ側に置きます。",
            "走数が少ない馬（初出走は分類なし）は、位置取りが安定していないため参考程度に見てください。",
        ]
        if bands:
            notes.append(f"今回のコース（{ctx.venue} {ctx.surface}{ctx.distance}m）の脚質（初角の位置）別の成績（走数 勝率 複勝率 単勝回収率 複勝回収率。単勝50倍以上は除く）: "
                         + " ／ ".join(f"{band_label(k)} {b['n']}走 {_pct(b['win'])} {_pct(b['top3'])} {_pct(b['win_roi'])} {_pct(b['place_roi'])}" for k, b in bands.items()))
        return ModuleOutput(columns=columns, groups=[], cells=cells, notes=notes, data_until=until, source=source)
