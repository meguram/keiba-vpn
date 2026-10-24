"""予測 API の expected_megu に、v3b の集約値と『直近走の無視してよい低下（excusable）』の判定を足す（tp-69）。

使い方（expected_megu.attach_expected_megu の、race_details を recent に反映した後で呼ぶ）:
    from src.pipeline.inference.megu_excusable_attach import annotate
    annotate(ex, runs)      # ex: expected_from_runs の返り値（recent に結果の詳細が入った後）、runs: その馬の全走のタプル（runs_for_horse の返り値）

付くもの:
  ex["v3b"] = {"value", "n", "method"}          … 下限なし（API では勝ち馬の指数が無い）＋低信頼の重み 0.5 ＋ 加重 Huber。表示用の参考値
  ex["recent"][i] に low_run / excusable / excusable_p / excusable_reasons … 判定できる走（直前に 3 走以上ある走）だけ
判定は参考表示用。tp-69 の検証で、除外して集約すると次走予測が悪化したので、v3b の値は excusable を除かない。
API では備考・ペース区分（race_quality）が無いため、それらの信号は欠損（中立）で判定する。判定器が無い・信号が足りないときは何も付けない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.megu_index import excusable as ex_mod
from src.pipeline.megu_index import v3b

LOW_Z = 1.0
MIN_SIGNALS = 4         # 結果の詳細（着順・頭数・通過順・上がり・着差）のうち、これ以上揃わない走は判定しない


def _detail_frame(runs: list, recent: list[dict]) -> pd.DataFrame:
    det = {str(r.get("race_id")): r for r in recent if r.get("race_id")}
    rows = []
    for t in runs:
        ymd, v, lo, surf, dist, _venue, _cl, jw, _age, rid = t[:10]
        d = det.get(str(rid), {})
        rows.append({
            "horse_id": "h", "race_id": str(rid), "date": f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}", "x": float(v), "low_conf": 1.0 if lo else 0.0,
            "fp": d.get("finish"), "field_size": d.get("field"), "passing_order": d.get("passing"), "last_3f": d.get("l3f"), "margin": d.get("margin"),
            "distance": d.get("distance") if d.get("distance") is not None else dist, "surface": d.get("surface") or surf,
            "weight_change": d.get("bw_diff"), "jockey_weight": d.get("jw") if d.get("jw") is not None else jw, "track_condition": d.get("going"),
        })
    return pd.DataFrame(rows)


def annotate(ex: dict | None, runs: list, model: ex_mod.ExcusableModel | None = None) -> None:
    if not ex or not runs:
        return
    recent = ex.get("recent") or []
    vals = [float(r["value"]) for r in recent if r.get("value") is not None]
    if vals:
        agg = v3b.aggregate(vals, low_conf=[bool(r.get("low")) for r in recent if r.get("value") is not None])
        if agg is not None:
            ex["v3b"] = {"value": round(agg, 1), "n": len(vals),
                         "method": "直近5走を、新しい走ほど重い加重 Huber（低信頼の走は重み半分）で集約。大差負けの下限は API では未適用"}
    model = model if model is not None else ex_mod.load_default_model()
    if model is None:
        return
    df = _detail_frame(sorted(runs, key=lambda t: (t[0], str(t[9]))), recent)
    df["field_size"] = pd.to_numeric(df["field_size"], errors="coerce").fillna(16.0)
    df["fp"] = pd.to_numeric(df["fp"], errors="coerce")
    sig = ex_mod.run_signals(df, lift=None, scale=v3b.SCALE)
    thr = float(model.meta.get("threshold_excl20", model.meta.get("threshold_excl35", 1.0)))
    detail_cols = ["fp", "field_size", "passing_order", "last_3f", "margin"]
    have = df[detail_cols].notna().sum(axis=1)
    p = model.score(sig)
    by_race = {df.loc[i, "race_id"]: i for i in df.index}
    for r in recent:
        i = by_race.get(str(r.get("race_id")))
        if i is None or r.get("finish") is None or int(have[i]) < MIN_SIGNALS:
            continue
        od = sig.loc[i, "own_drop"]
        if pd.isna(od):
            continue
        low = bool(od >= LOW_Z)
        r["low_run"] = low
        if low:
            flag = bool(p[i] >= thr)
            r["excusable"] = flag
            r["excusable_p"] = round(float(p[i]), 2)
            if flag:
                reasons = model.reasons(sig.loc[i])
                if reasons:
                    r["excusable_reasons"] = reasons
