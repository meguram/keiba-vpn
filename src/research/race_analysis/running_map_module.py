"""走法マップ: 出走馬の走法（実測ラップ版の 9 方面）を、全馬共通の 2 次元に写した図。
コース（競馬場・芝ダ・距離）ごとに好走しやすいゾーンを色で重ねる。

計算は src/research/race/running_map.py（走法データは src/research/race/running_profile.py が出走記録から作る）。
表ではなく図なので、ModuleOutput の view="map" と figure で返す。各馬はエントリーの key で表と対応づける。
"""

from __future__ import annotations

from typing import Any

from src.research.race import running_map as rm
from src.research.race_analysis.base import ModuleOutput, RaceContext

# 図に要る馬の項目（それ以外は保存しない）
_HORSE_KEYS = ("horse_id", "horse_number", "horse_name", "x", "y", "group", "percentiles", "n_runs", "n_lap_runs", "axis_runs", "from_prior", "estimated", "own_weight", "extras", "last_run_date")


class RunningMapModule:
    id = "running_map"
    label = "走法マップ"
    description = ("出走馬の走り方（末脚の速さと持続力・前後半のペース・出脚・ギアチェンジ・消耗耐性・先行力・コーナーと道中の押し上げ）を、似た馬が近くに来るように 2 次元に並べた図。"
                   "背景の色は、このコースで好走しやすい走り方のゾーン（濃いほど好走しやすい）。")

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
            "extra_axes": out.get("extra_axes", []),
            "horses": horses,
            "groups": out["groups"],
            "unmapped": unmapped,
            "fidelity": out["fidelity"],
            "extent": out.get("extent"),
            "aptitude": out.get("aptitude"),
        }
        notes = [
            "各馬の直近 10 走（新しい走ほど重く数える）から走り方を数値にしています。実測ラップ（馬ごとの 200m ラップ）のある走だけが対象で、"
            "それが少ない馬は父の産駒の平均に寄せ、1 走も無い馬は父の産駒の平均（推定）です。方面ごとの実測の走数は馬の詳細に出ます。",
            "急坂適性（中山・阪神のみ）と道悪の軸（芝=芝道悪適性、ダート=ダートタフ適正）は地図の位置には入れず、馬の詳細にだけ出します。"
            "判定できない馬は「不明」で、0 や平均とは扱いません。",
            "近くにいる馬ほど走り方が似ています。矢印は、その向きに行くほどその特徴が強いことを示します（矢印が短い特徴は、この図では見分けにくい）。",
            "このレースの開催日より前の走だけを使っています。",
            "座標は全レース共通で、同じ位置は同じ走り方です。背景の色は 2020 年以降の同じコースの全出走で測った好走しやすさ（3 着以内に入った割合を頭数から決まる期待と比べた倍率）で、"
            "測った期間の結果を含む値です。",
        ] + list(out.get("notes") or [])
        return ModuleOutput(columns=[], groups=[], cells={}, notes=notes, data_until=out.get("data_until"), view="map", figure=figure)
