"""tp-61: 展開タイプ（ペース × 隊列の凝縮度）の予測と、脚質・各馬の展開の向き不向き（画面表示用、既定オフ）。

実測のラップを文献の分類（前後半の差・RPCI/PCI・中緩み・瞬発/ロングスパート/消耗/持続・隊列の縦長/凝縮）の軸に分解した MECE の規則（lap_type_rule）で
6 つのラップ型に分け、事前（前日〜T-60）に分かる量から型と隊列の確率を出して、脚質ごとの有利度に写す。有効化は環境変数 KEIBA_PACE_TYPE=1。
係数・しきい値・脚質別の表は config/pace_type.json（tp61_* スクリプトが作る）。
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any

ENV_FLAG = "KEIBA_PACE_TYPE"
CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "pace_type.json"
VERSION = "tp61-1"

# 規則の木（排他的・網羅的）: ①前後半のバランス（後傾=スロー／均衡=ミドル／前傾=ハイ）→ ②スロー側は加速の長さ、ミドル・ハイ側は中盤の緩みの有無
LAP_TYPES = ["slow_kick", "slow_spurt", "even_kick", "even_flat", "high_breather", "high_collapse"]
LAP_LABELS = {
    "slow_kick": "超スロー・ヨーイドン（直線だけの瞬発戦）",
    "slow_spurt": "スロー・ロングスパート（右肩上がりの加速戦）",
    "even_kick": "均衡・中緩み（中盤で緩み、終いで再加速）",
    "even_flat": "均衡・平坦（持続戦）",
    "high_breather": "前傾・息が入るハイ（中盤で緩むハイ）",
    "high_collapse": "前傾・一本調子のハイ（消耗戦）",
}
SHAPES = ["compact", "normal", "stretched"]
SHAPE_LABELS = {"compact": "凝縮", "normal": "中間", "stretched": "縦長"}
STYLES = [("front", "nige", "逃げ"), ("stalker", "senko", "先行"), ("mid", "sashi", "差し"), ("closer", "oikomi", "追込")]


def lap_type_rule(bal: float, lull_amp: float, accel_len: float, th: dict, band: str | None = None) -> str:
    """bal: 後半3F − 前半3F の par との差（秒、小さい=スロー寄り）。lull_amp: 中盤の最も遅いラップが前後の平均より遅い秒数。
    accel_len: 後半で中盤より 0.2 秒以上速いラップの割合。lull_amp / accel_len のしきい値は距離帯ごと（th['by_band'][band]）。"""
    bt = (th.get("by_band") or {}).get(band) or th
    if bal <= th["bal_q25"]:
        return "slow_spurt" if accel_len > bt["accel_len"] else "slow_kick"
    if bal >= th["bal_q75"]:
        return "high_breather" if lull_amp >= bt["lull_amp"] else "high_collapse"
    return "even_kick" if lull_amp >= bt["lull_amp"] else "even_flat"


def enabled() -> bool:
    return os.environ.get(ENV_FLAG, "0").strip().lower() in ("1", "true", "yes", "on")


@functools.lru_cache(maxsize=2)
def load_config(path: str | None = None) -> dict | None:
    p = Path(path) if path else CONFIG_PATH
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ───────────────────────── 事前予測（学習側 src/scripts/data/tp61_pre.py / tp61_hana.py と同じ定義。テストで一致を確認）
MODELS_PATH = Path(__file__).resolve().parents[3] / "config" / "pace_type_models.json"
PACE = ["slow", "mid", "high"]
PACE_JA = {"slow": "スロー", "mid": "ミドル", "high": "ハイ"}
STYLE_KEYS = {"front": "nige", "stalker": "senko", "mid": "sashi", "closer": "oikomi"}
STYLE_JA = {"front": "逃げ", "stalker": "先行", "mid": "差し", "closer": "追込"}
SCENARIO_NOTE = {
    "slow": "前に行く馬が残りやすく、後ろの馬は届きにくい",
    "mid": "脚質による有利不利は小さい",
    "high": "前に行く馬が苦しく、後ろから差せる馬が有利",
}
SHAPE_NOTE = {"compact": "馬群が詰まったまま直線に入る", "normal": "隊列は標準的", "stretched": "隊列が縦長になりやすい"}
NAN = float("nan")


def scenario_key(pace: str, shape: str) -> str:
    return f"{pace}_{shape}"


def scenario_label(pace: str, shape: str) -> str:
    return f"{PACE_JA[pace]}・{SHAPE_LABELS[shape]}"


def race_pf_stats(early: list) -> dict:
    """想定位置取り（0=先頭〜1=後方、欠損は None）の分布の特徴。tp61_pre.race_pf_features と同じ。"""
    import numpy as np

    e = np.sort(np.array([x for x in early if x is not None and x == x], dtype=float))
    n = len(e)
    rec: dict = {"n_known": n}
    if n >= 4:
        pad = np.pad(e[:5], (0, max(0, 5 - len(e[:5]))), constant_values=np.nan)
        rec.update({f"e{i + 1}": float(pad[i]) for i in range(5)})
        rec["d12"] = float(e[1] - e[0])
        rec["d15"] = float(e[min(4, n - 1)] - e[0])
        rec["std5"] = float(np.std(e[:5]))
        rec["n_le15"], rec["n_le25"], rec["n_le35"], rec["n_le50"] = (int((e <= t).sum()) for t in (0.15, 0.25, 0.35, 0.50))
        rec["near_lead10"] = int((e <= e[0] + 0.10).sum())
        rec["mean_e"], rec["std_e"] = float(e.mean()), float(e.std())
        rec["q25"], rec["q50"], rec["q75"] = (float(np.percentile(e, q)) for q in (25, 50, 75))
    return rec


def horse_lead_stats(past_races: list, race_date: str | None) -> dict | None:
    """馬柱の過去走（対象日より前の直近 5 走）から、1 角先頭率など。tp61_hana.horse_asof と同じ指標。"""
    from src.pipeline.inference.position_forecast import _digits, parse_run

    rd = _digits(race_date)
    runs = []
    for p in past_races or []:
        if not isinstance(p, dict):
            continue
        d = _digits(p.get("date"))
        if rd and d and d >= rd:
            continue
        r = parse_run(str(p.get("raw") or ""))
        if r:
            runs.append((d, r))
    runs.sort(key=lambda x: x[0], reverse=True)
    runs = runs[:5]
    n = len(runs)
    if n == 0:
        return {"lead5": NAN, "n_led5": 0.0, "n5": 0.0, "front5": NAN}
    led = [1.0 if r[1][1] == 1 else 0.0 for r in runs]
    f15 = [1.0 if r[1][0] <= 0.15 else 0.0 for r in runs]
    return {"lead5": sum(led) / n, "n_led5": float(sum(led)), "n5": float(n), "front5": sum(f15) / n}


def hana_race_features(stats: list[dict], numbers: list[int], field: int) -> dict:
    """ハナ主張馬の頭数と強さ。tp61_hana.race_hana と同じ定義（直近 5 走の 1 角先頭率が 0.4 以上・履歴 3 走以上を『主張』、0.6 以上を『強い』、5 走中 4 回以上先頭を『非常に強い』）。"""
    import numpy as np

    lead = np.array([s["lead5"] for s in stats], dtype=float)
    n5 = np.array([s["n5"] for s in stats], dtype=float)
    nl = np.array([s["n_led5"] for s in stats], dtype=float)
    no = np.array(numbers, dtype=float)
    ok = (n5 >= 3) & ~np.isnan(lead)
    claim, strong, vstrong = ok & (lead >= 0.4), ok & (lead >= 0.6), ok & (nl >= 4)
    order = np.sort(np.where(ok, lead, -1.0))[::-1]
    rel = (no - 1) / max(field - 1, 1)
    cn = np.sort(no[claim])
    pairs = int((np.diff(cn) <= 2).sum()) if len(cn) >= 2 else 0
    top = lambda i: max(float(order[i]), 0.0) if len(order) > i else 0.0  # noqa: E731
    front5 = np.array([s["front5"] for s in stats], dtype=float)
    return {"hn_claim": int(claim.sum()), "hn_claim_strong": int(strong.sum()), "hn_claim_vstrong": int(vstrong.sum()),
            "hn_l1": top(0), "hn_l2": top(1), "hn_l3": top(2), "hn_sum3": float(sum(top(i) for i in range(3))),
            "hn_pairs": pairs, "hn_outer": int((claim & (rel >= 0.65)).sum()), "hn_inner": int((claim & (rel <= 0.35)).sum()),
            "hn_minrel": float(rel[claim].min()) if claim.any() else NAN, "hn_front5_n": float(np.nansum(front5)),
            "hn_uncertain": int((n5 < 3).sum())}


@functools.lru_cache(maxsize=1)
def _boosters() -> dict | None:
    try:
        import lightgbm as lgb

        raw = json.loads(MODELS_PATH.read_text(encoding="utf-8"))
        return {k: lgb.Booster(model_str=v) for k, v in raw.items()}
    except Exception:
        return None


def build_features(resp: dict, past_doc: dict | None, race_date: str | None, cfg: dict) -> dict | None:
    preds = [p for p in resp.get("predictions") or [] if isinstance(p, dict) and p.get("horse_number") is not None]
    field = len(preds)
    if field < 8:
        return None
    early = [(p.get("position_forecast") or {}).get("early_pos_norm") for p in preds]
    row = race_pf_stats(early)
    if row["n_known"] < 4:
        return None
    pf = resp.get("pace_forecast") or {}
    venue, surface, distance = resp.get("venue"), resp.get("surface"), resp.get("distance")
    ck = f"{venue}|{surface}|{int(float(distance or 0))}"
    crs = (cfg.get("course") or {}).get("by_course", {}).get(ck) or {}
    glob = (cfg.get("course") or {}).get("global", {})
    row.update({"pf_front": pf.get("front_runner_count"), "pf_stalker": pf.get("stalker_count"), "pf_pressure": pf.get("early_pressure"),
                "field_size": field, "distance": float(distance or 0), "turf": 1.0 if surface == "芝" else 0.0,
                "heavy": 0.0 if str(resp.get("track_condition") or "良") == "良" else 1.0,
                "venue_code": (cfg.get("venue_code") or {}).get(venue, NAN), "grade_code": (cfg.get("grade_code") or {}).get(str(resp.get("grade") or ""), NAN),
                "crs_gap": crs.get("crs_gap", glob.get("crs_gap", NAN)), "crs_bal": crs.get("crs_bal", glob.get("crs_bal", NAN)),
                "crs_t3": crs.get("crs_t3", glob.get("crs_t3", NAN)), "crs_lead": crs.get("crs_lead", glob.get("crs_lead", NAN)), "crs_n": crs.get("n", 0)})
    geo = (cfg.get("geo") or {}).get(ck) or [None, None]
    row["hn_firstcurve"], row["hn_start_geom"] = (NAN if geo[0] is None else geo[0]), (NAN if geo[1] is None else geo[1])
    rows = {}
    for e in (past_doc or {}).get("entries") or []:
        if isinstance(e, dict):
            rows[str(e.get("horse_id") or "")] = e
    stats = [horse_lead_stats((rows.get(str(p.get("horse_id") or "")) or {}).get("past_races") or [], race_date) or {"lead5": NAN, "n_led5": 0.0, "n5": 0.0, "front5": NAN} for p in preds]
    row.update(hana_race_features(stats, [int(p["horse_number"]) for p in preds], field))
    return row


def _own_probs(row: dict, cfg: dict) -> dict | None:
    import numpy as np

    b = _boosters()
    if not b:
        return None
    x = np.array([[row.get(f, NAN) if row.get(f) is not None else NAN for f in cfg["features"]]], dtype=float)
    return {"pace": [float(v) for v in b["pace"].predict(x)[0]], "shape": [float(v) for v in b["shape"].predict(x)[0]]}


def _logit(p):
    import numpy as np

    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _renorm(p, k: float):
    import numpy as np

    p = np.asarray(p, dtype=float)
    if k == 1.0:
        return p / p.sum()
    z = _logit(p)
    lo, hi = -6.0, 6.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid)))).sum() < k:
            lo = mid
        else:
            hi = mid
    return 1 / (1 + np.exp(-(z + (lo + hi) / 2)))


def _theta_shift(theta: list[float], e: float | None, pace_i: int, shape_i: int) -> float:
    """tp61_scen.design（mode='add'）と同じ: φ=(e−0.5, (e−0.5)², 欠損)。係数の並びは [ペース 3×3, 隊列 3×3]。"""
    unk = 1.0 if e is None or e != e else 0.0
    x = 0.0 if unk else float(e) - 0.5
    phi = (x, x * x, unk)
    return sum(theta[pace_i * 3 + j] * phi[j] for j in range(3)) + sum(theta[9 + shape_i * 3 + j] * phi[j] for j in range(3))


def predict_pace_type(resp: dict, past_doc: dict | None, race_date: str | None) -> tuple[dict, dict] | None:
    cfg = load_config()
    if not cfg:
        return None
    row = build_features(resp, past_doc, race_date, cfg)
    if row is None:
        return None
    own = _own_probs(row, cfg)
    if not own:
        return None
    pace_p, shape_p = own["pace"], own["shape"]
    used_lap = False
    lap = resp.get("lap_forecast")
    if isinstance(lap, dict) and lap.get("scenarios"):
        by = {str(s.get("name")): float(s.get("prob") or 0.0) for s in lap["scenarios"]}
        lp = [by.get("スロー"), by.get("ミドル"), by.get("ハイ")]
        if all(v is not None for v in lp) and sum(lp) > 0:
            w = float(cfg.get("own_weight_with_lap", 0.35))
            tot = sum(lp)
            pace_p = [w * a + (1 - w) * (b / tot) for a, b in zip(pace_p, lp)]
            used_lap = True
    joint = {scenario_key(p, s): pace_p[i] * shape_p[j] for i, p in enumerate(PACE) for j, s in enumerate(SHAPES)}
    top_key = max(joint, key=joint.get)
    tp, ts = top_key.split("_")
    table = cfg.get("style_fit") or {}

    def lift(key: str, st: str) -> float | None:
        v = (table.get(key) or {}).get(st)
        return None if not v else v["top3_lift"]

    style_fit: dict = {}
    for st, short, ja in STYLES:
        exp = sum(joint[k] * (lift(k, st) or 1.0) for k in joint)
        top = lift(top_key, st)
        style_fit[short] = {"label": ja, "score": round(max(-1.0, min(1.0, (exp - 1.0) / 0.2)), 3), "expected_lift": round(exp, 3),
                            "top_scenario_lift": top, "top_scenario_score": None if top is None else round(max(-1.0, min(1.0, (top - 1.0) / 0.35)), 3)}
    mp, ms = max(pace_p), max(shape_p)
    conf = "high" if mp >= 0.55 and ms >= 0.55 else "mid" if max(mp, ms) >= 0.5 else "low"
    pace_type = {
        "version": VERSION, "key": top_key, "label": scenario_label(tp, ts), "note": f"{SCENARIO_NOTE[tp]}。{SHAPE_NOTE[ts]}。",
        "probs": {k: round(v, 4) for k, v in joint.items()},
        "pace_probs": {p: round(pace_p[i], 4) for i, p in enumerate(PACE)}, "shape_probs": {s: round(shape_p[j], 4) for j, s in enumerate(SHAPES)},
        "style_fit": style_fit, "confidence": conf,
        "basis": {"pace_level": (resp.get("pace_forecast") or {}).get("level"), "strong_front_runners": row.get("hn_claim_strong"), "front_runners_claim": row.get("hn_claim"),
                  "course_stretched_tendency": row.get("crs_gap"), "lap_forecast_used": used_lap},
        "scenarios": [{"key": k, "label": scenario_label(*k.split("_")), "prob": round(v, 4), "pace": k.split("_")[0], "shape": k.split("_")[1]} for k, v in joint.items()],
    }
    return pace_type, {"row": row, "joint": joint, "cfg": cfg}


def scenario_forecast(resp: dict, ctx: dict) -> dict | None:
    import numpy as np

    cfg, joint = ctx["cfg"], ctx["joint"]
    preds = [p for p in resp.get("predictions") or [] if isinstance(p, dict) and p.get("horse_number") is not None]
    base = {k: np.array([float(p.get(f"{k}_prob") or 0.0) for p in preds]) for k in ("win", "top2", "top3")}
    if any((v <= 0).all() for v in base.values()):
        return None
    ks = {"win": 1.0, "top2": 2.0, "top3": 3.0}
    theta = cfg["scenario_theta"]
    early = [(p.get("position_forecast") or {}).get("early_pos_norm") for p in preds]
    per: dict[str, dict[str, np.ndarray]] = {}
    for i, pc in enumerate(PACE):
        for j, sh in enumerate(SHAPES):
            per[scenario_key(pc, sh)] = {}
            for kind in ("win", "top2", "top3"):
                shift = np.array([_theta_shift(theta[kind], e, i, j) for e in early])
                z = _logit(base[kind]) + shift
                per[scenario_key(pc, sh)][kind] = _renorm(1 / (1 + np.exp(-z)), ks[kind])
    keys = list(joint)
    w = np.array([joint[k] for k in keys])
    marg = {kind: sum(joint[k] * per[k][kind] for k in keys) for kind in ("win", "top2", "top3")}
    # 馬ごと: シナリオ別の確率、展開のばらつき（勝率の logit の標準偏差）、ペース別の印（ペース内の隊列は事前確率で平均）
    pace_marg = {}
    for pc in PACE:
        ws = {k: joint[k] for k in keys if k.startswith(pc + "_")}
        tot = sum(ws.values()) or 1.0
        pace_marg[pc] = sum(ws[k] / tot * per[k]["win"] for k in ws)
    marks = {pc: np.argsort(-pace_marg[pc]) for pc in PACE}
    mark_sym = ["◎", "○", "▲", "△", "△"]
    horses = []
    for idx, p in enumerate(preds):
        by_sc = {}
        scores = []
        for k in keys:
            fit = float(np.clip((_logit(per[k]["top3"][idx]) - _logit(base["top3"][idx])) / 0.3, -1, 1))
            scores.append(fit)
            by_sc[k] = {"win": round(float(per[k]["win"][idx]), 4), "top2": round(float(per[k]["top2"][idx]), 4), "top3": round(float(per[k]["top3"][idx]), 4), "fit_score": round(fit, 3)}
        lw = np.array([_logit(per[k]["win"][idx]) for k in keys])
        mean = float((w * lw).sum() / w.sum())
        sens = float(np.sqrt((w * (lw - mean) ** 2).sum() / w.sum()))
        best = keys[int(np.argmax(scores))]
        marks_by_pace = {}
        for pc in PACE:
            rank = int(np.where(marks[pc] == idx)[0][0])
            marks_by_pace[pc] = mark_sym[rank] if rank < len(mark_sym) else ""
        lab = "・".join(f"{PACE_JA[pc]}{marks_by_pace[pc]}" for pc in ("high", "mid", "slow") if marks_by_pace[pc] in ("◎", "○"))
        horses.append({"horse_number": int(p["horse_number"]), "by_scenario": by_sc,
                       "marginal": {"win": round(float(marg["win"][idx]), 4), "top2": round(float(marg["top2"][idx]), 4), "top3": round(float(marg["top3"][idx]), 4)},
                       "sensitivity": round(sens, 3), "sensitivity_label": "展開次第" if sens >= 0.12 else "展開に左右されにくい" if sens < 0.05 else "ふつう",
                       "best_scenario": best, "best_scenario_label": scenario_label(*best.split("_")), "marks_by_pace": marks_by_pace, "label": lab})
    return {"version": VERSION, "scenarios": [{"key": k, "label": scenario_label(*k.split("_")), "prob": round(joint[k], 4)} for k in keys], "horses": horses,
            "basis": {"base": "最終モデルの勝率・連対率・複勝率に、シナリオごとの想定位置取りの効果を足してレース内で合計を戻した値（表示・説明用。馬券の判断には使わない）"}}


def attach_pace_type(resp: dict, past_doc: dict | None = None, race_date: str | None = None, extra: dict | None = None) -> dict:
    """予測 API の応答に pace_type と scenario_forecast を足す（KEIBA_PACE_TYPE=1 のときだけ。失敗・材料不足のときは何もしない）。
    position_forecast / pace_forecast（あれば lap_forecast）が付いたあとで呼ぶ。extra: track_condition / grade など応答に無い項目。"""
    if not enabled() or not isinstance(resp, dict):
        return resp
    try:
        r = {**resp, **{k: v for k, v in (extra or {}).items() if v}}
        out = predict_pace_type(r, past_doc, race_date)
        if not out:
            return resp
        pt, ctx = out
        resp = {**resp, "pace_type": pt}
        sf = scenario_forecast(r, ctx)
        if sf:
            resp["scenario_forecast"] = sf
    except Exception:  # noqa: BLE001 - 表示用の付加情報なので、失敗しても応答は返す
        return resp
    return resp
