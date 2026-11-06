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
_HORSE_KEYS = ("horse_id", "horse_number", "horse_name", "x", "y", "group", "percentiles", "n_runs", "n_lap_runs", "axis_runs", "from_prior", "estimated", "own_weight", "extras", "similar", "last_run_date", "finish_position")


def _load_result(race_id: str) -> dict | None:
    """その日の前のレースの結果（馬場傾向に使う）。"""
    from src.scraper.storage import HybridStorage

    st = HybridStorage()
    for cat in ("race_result", "race_result_on_time"):   # 確定の結果が無ければ速報（当日は速報が先に入る）
        try:
            card = st.load(cat, race_id)
        except Exception:
            card = None
        if card and card.get("entries"):
            return card
    return None


def finish_positions(card: dict | None) -> dict[str, int]:
    """結果（確定または速報）から horse_id → 着順。着順が数字でない馬（中止・除外など）は入れない。"""
    out: dict[str, int] = {}
    for e in (card or {}).get("entries") or []:
        v = e.get("finish_position")
        if e.get("horse_id") and isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 1:
            out[str(e["horse_id"])] = int(v)
    return out


class RunningMapModule:
    id = "running_map"
    label = "走法マップ"
    description = ("出走馬の走り方（TSとTS持続力・出脚・ギアチェンジ・Hペース耐性・先行力・Cr性能（3〜4角の押し上げ）・まくり特性（道中の押し上げ））のうち、"
                   "ギアチェンジ・TS・TS持続力・Cr性能・Hペース耐性・基礎スピードの 6 方面で、走法の近い馬ほど近くに来るように 2 次元に並べた図（縦・横の軸に意味は無く、上ほどこのコースで好走しやすい走法）。"
                   "背景の色は、このコースで好走しやすい走り方のゾーン（濃いほど好走しやすい）。その日の馬場傾向の色にも切り替えられる。")

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
        out = rm.build_running_map([e for e in ctx.entries if e.get("horse_id")], ctx.race_id, conditions, load_result=_load_result)
        key_of = {str(e.get("horse_id")): e["key"] for e in ctx.entries if e.get("horse_id")}
        finish = finish_positions(_load_result(ctx.race_id))

        def fig(o: dict[str, Any]) -> dict[str, Any]:
            horses = [{**{k: h.get(k) for k in _HORSE_KEYS if k != "finish_position"}, "key": key_of.get(str(h.get("horse_id"))),
                       "finish_position": finish.get(str(h.get("horse_id")))} for h in o["horses"]]
            unmapped = [{"key": key_of.get(str(u.get("horse_id"))), "horse_name": u.get("horse_name"), "horse_number": u.get("horse_number"),
                         "reason": u.get("reason")} for u in o["unmapped"]]
            return {
                "status": o.get("status"),
                "axes": o["axes"],
                "extra_axes": o.get("extra_axes", []),
                "horses": horses,
                "groups": o["groups"],
                "unmapped": unmapped,
                "fidelity": o["fidelity"],
                "extent": o.get("extent"),
                "aptitude": o.get("aptitude"),
                "course_map": o.get("course_map"),
                "day_trend": o.get("day_trend"),
                "best5": o.get("best5"),
            }

        figure: dict[str, Any] = fig(out)
        figure["wet"] = fig(out["wet"]) if out.get("wet") else None
        notes = [
            "各馬の直近 10 走（新しい走ほど重く数える）から走り方を数値にしています。実測ラップ（馬ごとの 200m ラップ）のある走だけが対象で、"
            "それが少ない馬は父の産駒の平均に寄せ、1 走も無い馬は父の産駒の平均（推定）です。方面ごとの実測の走数は馬の詳細に出ます。",
            "急坂適性（中山・阪神のみ）と道悪の軸（芝=芝道悪適性、ダート=ダートタフ適正）は地図の位置には入れず、馬の詳細にだけ出します。"
            "判定できない馬は「不明」で、0 や平均とは扱いません。",
            "地図に使う方面は、ギアチェンジ・TS・TS持続力・Cr性能・Hペース耐性・基礎スピードの 6 つに固定しています。その 6 方面で走法の近い馬ほど近くに置いています。"
            "このコースで好走（1〜3 着=1、4〜7 着は 0.2 ずつ下げ、8 着以下=0 の得点）への寄与率が高い方面ほど距離に大きく効くよう重み付けします（均等との半々）。"
            "縦・横の軸に意味は無く、上に行くほどこのコースで好走しやすい走法の向きです。"
            "寄与率は 2024 年以降の同じコースの全出走で、各馬のそのレースより前の走法から得点を説明したときの寄与（他の方面と重なる分は数えない）。"
            "出走が少ないコースは、同じ競馬場・芝ダートの距離帯（1400m 以下・1500〜1700m・1800〜2500m・2600m 以上の長距離）でまとめ、"
            "それでも少なければ全競馬場の同じ芝ダ・距離帯でまとめて測っています。",
            "地図の近さは 6 方面だけの近さです。馬を押すと、その 6 方面で本当に近い馬（このレースの中で 3 頭）を線で結びます。",
            "Hペース耐性は、ペースと位置をそろえたうえで、ハイペースの追走で脚を使わず終いに落ちなかったか（追走効率）を表します。",
            "基礎スピードは、時計の速い馬場（良馬場で、その日に速い時計が出ている）でいつも通りに走れるか、遅い馬場で成績が落ちるかの差です。"
            "Hペース耐性（基礎スピードの範囲を超えた速い流れで、終いに脚がなくなる中でも走れるか）とは別物です。",
            "既定は良馬場の地図です（座標も背景の色も良馬場のレースだけで測っています）。「道悪の地図」に切り替えると、基礎スピードの代わりに道悪適性"
            "（同じ芝ダの直近 30 走で、馬場が重いほど着順が上がるか）を入れた 6 方面で、稍重・重・不良のレースだけで測った地図を出します（稍重以上は件数が少ないので 3 つをまとめています）。"
            "道悪の走が無い馬の道悪適性は中立です。",
            "このレースの開催日より前の走だけを使っています。",
            "「走法が合う Best5」は、地図の背景の色（コース特注ゾーン＋0.5×馬場特注ゾーン。隣り合う格子と混ぜて読む）が濃い所にいる馬の順で、1 パターンだけです。人気は考えません。"
            "配分はクラス・レース番号・当日のレース数・芝ダ・地図の水準ごとに変えても、2025〜2026 年の外挿で定数より良くならなかったので共通です（当日のデータが少ないと馬場の倍率は中立に寄ります）。"
            "3 着以内は 5 頭中約 1.6 頭、人気を除いた上乗せはごくわずかでした。",
        ] + list(out.get("notes") or [])
        return ModuleOutput(columns=[], groups=[], cells={}, notes=notes, data_until=out.get("data_until"), view="map", figure=figure)
