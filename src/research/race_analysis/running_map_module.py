"""走法マップ: 出走馬の走法（末脚・加速・前後半・先行力・押し上げ・レース質との相性の 6 方面）を 2 次元に写した図。

計算は src/research/race/running_map.py（走法データは src/research/race/running_profile.py が出走記録から作る）。
表ではなく図なので、ModuleOutput の view="map" と figure で返す。各馬はエントリーの key で表と対応づける。
"""

from __future__ import annotations

from typing import Any

from src.research.race import running_map as rm
from src.research.race_analysis.base import ModuleOutput, RaceContext

# 図に要る馬の項目（それ以外は保存しない）
_HORSE_KEYS = ("horse_id", "horse_number", "horse_name", "x", "y", "group", "percentiles", "n_runs", "estimated", "own_weight", "last_run_date")


class RunningMapModule:
    id = "running_map"
    label = "走法マップ"
    description = ("出走馬の走り方（末脚・加速・前後半のペース・先行力・道中の押し上げ・このレースの質との相性）を、"
                   "似た馬が近くに来るように 2 次元に並べた図。色は走り方の近いグループ。")

    def applicable(self, ctx: RaceContext) -> bool:
        return bool(ctx.venue and ctx.surface and ctx.distance and ctx.entries)

    def compute(self, ctx: RaceContext) -> ModuleOutput:
        conditions = {
            "date": rm._norm_date(ctx.date),
            "venue": ctx.venue,
            "surface": rm._norm_surface(ctx.surface),
            "distance": ctx.distance,
            "track_condition": rm._norm_condition(ctx.track_condition),
        }
        out = rm.build_running_map([e for e in ctx.entries if e.get("horse_id")], ctx.race_id, conditions)
        key_of = {str(e.get("horse_id")): e["key"] for e in ctx.entries if e.get("horse_id")}
        horses = [{**{k: h.get(k) for k in _HORSE_KEYS}, "key": key_of.get(str(h.get("horse_id")))} for h in out["horses"]]
        unmapped = [{"key": key_of.get(str(u.get("horse_id"))), "horse_name": u.get("horse_name"), "horse_number": u.get("horse_number"),
                     "reason": u.get("reason")} for u in out["unmapped"]]
        figure: dict[str, Any] = {
            "status": out.get("status"),
            "axes": out["axes"],
            "horses": horses,
            "groups": out["groups"],
            "unmapped": unmapped,
            "fidelity": out["fidelity"],
            "rq_source": out["conditions"].get("rq_source"),
        }
        notes = [
            "各馬の直近 10 走（新しい走ほど重く数える）から走り方を数値にしています。出走が少ない馬は父の産駒の平均に寄せ、新馬は父の産駒の平均です。",
            "近くにいる馬ほど走り方が似ています。矢印は、その向きに行くほどその特徴が強いことを示します（矢印が短い特徴は、この図では見分けにくい）。",
            "このレースの開催日より前の走だけを使っています。",
        ] + list(out.get("notes") or [])
        return ModuleOutput(columns=[], groups=[], cells={}, notes=notes, data_until=out.get("data_until"), view="map", figure=figure)
