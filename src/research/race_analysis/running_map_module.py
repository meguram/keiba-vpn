"""走法マップ: 出走馬の走法（実測ラップ版の 8 方面）を、全馬共通の 2 次元に写した図。
コース（競馬場・芝ダ・距離）ごとに好走しやすいゾーンを色で重ねる。

計算は src/research/race/running_map.py（走法データは src/research/race/running_profile.py が出走記録から作る）。
表ではなく図なので、ModuleOutput の view="map" と figure で返す。各馬はエントリーの key で表と対応づける。
"""

from __future__ import annotations

from typing import Any

from src.research.race import running_map as rm
from src.research.race_analysis.base import ModuleOutput, RaceContext

# 図に要る馬の項目（それ以外は保存しない）
_HORSE_KEYS = ("horse_id", "horse_number", "horse_name", "x", "y", "group", "percentiles", "n_runs", "n_lap_runs", "axis_runs", "from_prior", "estimated", "own_weight", "extras", "similar", "last_run_date")


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


class RunningMapModule:
    id = "running_map"
    label = "走法マップ"
    description = ("出走馬の走り方（末脚の速さと持続力・出脚・ギアチェンジ・Hペース耐性・先行力・Cr性能（3〜4角の押し上げ）・まくり特性（道中の押し上げ））のうち、"
                   "このコースで好走への寄与率が高い方面（コースごとに 3〜6）で、走法の近い馬ほど近くに来るように 2 次元に並べた図（縦・横の軸に意味は無く、上ほどこのコースで好走しやすい走法）。"
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
            "course_map": out.get("course_map"),
            "day_trend": out.get("day_trend"),
        }
        notes = [
            "各馬の直近 10 走（新しい走ほど重く数える）から走り方を数値にしています。実測ラップ（馬ごとの 200m ラップ）のある走だけが対象で、"
            "それが少ない馬は父の産駒の平均に寄せ、1 走も無い馬は父の産駒の平均（推定）です。方面ごとの実測の走数は馬の詳細に出ます。",
            "急坂適性（中山・阪神のみ）と道悪の軸（芝=芝道悪適性、ダート=ダートタフ適正）は地図の位置には入れず、馬の詳細にだけ出します。"
            "判定できない馬は「不明」で、0 や平均とは扱いません。",
            "地図は、このコースで好走（1〜3 着=1、4〜7 着は 0.2 ずつ下げ、8 着以下=0 の得点）への寄与率が高い方面だけを使い、その方面で走法の近い馬ほど近くに置いています。"
            "使う方面の数はコースごとに 3〜6 で、寄与率を上から足して 90% に届く数です"
            "（寄与率の高い方面ほど距離に大きく効くよう重み付け）。縦・横の軸に意味は無く、上に行くほどこのコースで好走しやすい走法の向きです。"
            "寄与率は 2024 年以降の同じコースの全出走で、各馬のそのレースより前の走法から得点を説明したときの寄与（他の方面と重なる分は数えない）。地図に使った上位の方面は、丸の色の選択肢と馬の詳細で強調します。"
            "出走が少ないコースは、同じ競馬場・芝ダートの距離帯（1400m 以下・1500〜1700m・1800〜2500m・2600m 以上の長距離）でまとめ、"
            "それでも少なければ全競馬場の同じ芝ダ・距離帯でまとめて測っています。",
            "地図の近さは上位の方面だけの近さです。馬を押すと、8 方面すべてで本当に近い馬（このレースの中で 3 頭）を線で結びます。",
            "Hペース耐性は、ペースと位置をそろえたうえで、ハイペースの追走で脚を使わず終いに落ちなかったか（追走効率）を表します。",
            "このレースの開催日より前の走だけを使っています。",
            "座標は同じコース（またはまとめた距離帯）のレースで共通です。背景の色は 2024 年以降の同じコースの全出走で測った好走しやすさ（好走の得点を頭数から決まる期待と比べた倍率）で、"
            "測った期間の結果を含む値です。",
            "「今日の馬場」は、同じ日・同じ競馬場・同じ芝ダートの、このレースより前のレースの結果から、どの走り方（地図の 8 方面と道悪・急坂の軸）の馬が好走しているかを見たものです。"
            "レース単位の検定と多重検定の補正を通った傾向だけを出します。1 日のレース数は少ないので目安です。",
        ] + list(out.get("notes") or [])
        return ModuleOutput(columns=[], groups=[], cells={}, notes=notes, data_until=out.get("data_until"), view="map", figure=figure)
