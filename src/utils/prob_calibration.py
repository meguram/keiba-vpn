"""確率の較正（市場情報を使う）。モデルの勝率・連対率・複勝率を、単勝オッズから作る市場の確率と対数線形にブレンドして、実績に合わせる。

背景と検証: docs/html/modeling/chapters/prob_calibration_gap.html（tp-70）。要点: 実績はモデルと市場の“間”にあり（モデルが市場から離れるほど自信過剰）、
logit(p_cal) = a + w1×logit(p_model) + w2×logit(p_market) をレース内で合計 1 / 2 / 3 に合わせると、LogLoss・人気別のずれ・EV の信頼性が改善する。
市場オッズは『モデルの特徴量』ではなく、較正（と馬券の判断）にだけ使う。係数は config/prob_calibration.json（src/scripts/data/tp70_fit.py export で再作成）。

有効化は環境変数 KEIBA_PROB_CALIBRATION=1（既定オフ）。次のときは何もしない（None を返し、呼び出し側は従来の確率を使う）:
  無効 / 設定ファイル無し / 8 頭未満 / 単勝オッズが全頭揃っていない / 市場が予測オッズ（モデルの出力から作った値で独立な市場ではない）。
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from src.utils.race_probabilities import harville_top2_prob, harville_top3_prob

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "prob_calibration.json"
EPS = 1e-6
MIN_FIELD = 8
KSUM = {"win": 1.0, "top2": 2.0, "top3": 3.0}
POP_BINS = ((1, "1人気"), (2, "2人気"), (3, "3人気"), (5, "4-5人気"), (9, "6-9人気"), (10**6, "10人気以下"))


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def scale_to_sum(p: np.ndarray, k: float) -> np.ndarray:
    """レース内の確率の合計を k に合わせる（logit に共通のオフセットを足して二分法）。"""
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    k = min(k, len(p) - 1e-3)
    z = _logit(p)
    lo, hi = -12.0, 12.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if _sigmoid(z + mid).sum() > k:
            hi = mid
        else:
            lo = mid
    return _sigmoid(z + (lo + hi) / 2)


@lru_cache(maxsize=1)
def load_config() -> dict:
    try:
        d = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


from src.utils import cache_epoch as _cache_epoch  # noqa: E402

_cache_epoch.register("prob_calibration.config", lambda: load_config.cache_clear())


def is_enabled() -> bool:
    return os.environ.get("KEIBA_PROB_CALIBRATION", "") == "1" and bool(load_config().get("targets"))


def market_probs(win_odds: Sequence[float]) -> dict[str, np.ndarray]:
    """単勝オッズ → 市場の勝率（正規化 1/odds）と、Harville の連対率・複勝率。"""
    inv = 1.0 / np.asarray(win_odds, float)
    q = inv / inv.sum()
    return {"win": q, "top2": harville_top2_prob(q), "top3": harville_top3_prob(q)}


def _pop_classes(win_odds: Sequence[float]) -> list[str]:
    order = np.argsort(np.argsort(np.asarray(win_odds, float), kind="stable"), kind="stable") + 1   # 1 始まりの人気（オッズ昇順、同じなら並び順）
    out = []
    for r in order:
        for hi, name in POP_BINS:
            if r <= hi:
                out.append(name)
                break
    return out


def calibrate_arrays(p_model: dict[str, Sequence[float]], win_odds: Sequence[float], method: dict[str, str] | None = None) -> dict[str, np.ndarray] | None:
    """p_model = {'win','top2','top3': 配列}。較正後の同じ形の辞書、または None（適用しない）。"""
    cfg = load_config()
    n = len(win_odds)
    if not cfg.get("targets") or n < MIN_FIELD:
        return None
    wo = np.asarray(win_odds, float)
    if not np.all(np.isfinite(wo)) or np.any(wo <= 0):
        return None
    mk = market_probs(wo)
    meth = {**(cfg.get("default_method") or {}), **(method or {})}
    classes = _pop_classes(wo)
    out: dict[str, np.ndarray] = {}
    for t in ("win", "top2", "top3"):
        spec = cfg["targets"][t]
        lp, lm = _logit(p_model[t]), _logit(mk[t])
        if meth.get(t, "blend") == "blend_pop":
            z = np.array([spec["blend_pop"][c]["intercept"] + spec["blend_pop"][c]["coef"][0] * a + spec["blend_pop"][c]["coef"][1] * b for c, a, b in zip(classes, lp, lm)])
        else:
            s = spec["blend"]
            z = s["intercept"] + s["coef"][0] * lp + s["coef"][1] * lm
        out[t] = scale_to_sum(_sigmoid(z), KSUM[t])
    return out


def calibrate_entries(entries: Sequence[dict], odds_by_horse: dict[int, float], *, market_is_real: bool = True) -> list[dict] | None:
    """画面・API 用。entries は horse_number と win_prob / top2_prob / top3_prob を持つ辞書の並び。
    戻り値は entries と同順の [{'win_prob','top2_prob','top3_prob'}（較正後）]、適用しないときは None。
    呼び出し側は、有効なとき元の確率を *_model に退避し、較正後を win_prob などに入れて EV・印・推奨に使う。"""
    if not is_enabled() or not market_is_real:
        return None
    try:
        odds = [float(odds_by_horse.get(int(e.get("horse_number") or 0)) or 0) for e in entries]
        pm = {"win": [float(e["win_prob"]) for e in entries], "top2": [float(e["top2_prob"]) for e in entries], "top3": [float(e["top3_prob"]) for e in entries]}
    except (KeyError, TypeError, ValueError):
        return None
    cal = calibrate_arrays(pm, odds)
    if cal is None:
        return None
    return [{"win_prob": round(float(cal["win"][i]), 4), "top2_prob": round(float(cal["top2"][i]), 4), "top3_prob": round(float(cal["top3"][i]), 4)} for i in range(len(entries))]


# ───────────────────────── 画面・API の応答に適用する（読み出し時に計算し、保存済みの予測は書き換えない） ─────────────────────────

_PRED_ODDS_WORDS = ("predicted", "予測")


def _is_predicted_source(e: dict) -> bool:
    s = str(e.get("odds_source") or "").lower()
    return any(w in s for w in _PRED_ODDS_WORDS)


def calibration_status(preds: Sequence[dict]) -> tuple[bool, str | None]:
    """応答の出走馬一覧に較正を適用できるか。(適用できる, 適用できない理由)。"""
    if not load_config().get("targets"):
        return False, "較正の設定ファイルがありません"
    if len(preds) < MIN_FIELD:
        return False, f"{MIN_FIELD}頭未満のため市場補正なし"
    if any(_is_predicted_source(e) for e in preds):
        return False, "予測オッズのため市場補正なし（実オッズが出てから補正します）"
    try:
        odds = [float(e.get("win_odds") or 0) for e in preds]
    except (TypeError, ValueError):
        return False, "単勝オッズが読めないため市場補正なし"
    if any(o <= 0 or not np.isfinite(o) for o in odds):
        return False, "単勝オッズが全頭に揃っていないため市場補正なし（オッズが出る前は補正しません）"
    for e in preds:
        if any(e.get(k) is None for k in ("win_prob", "top2_prob", "top3_prob")):
            return False, "確率が揃っていないため市場補正なし"
    return True, None


def apply_prob_calibration(resp: dict, *, odds_phase: str | None = None, alpha: float = 0.55) -> dict:
    """予測の応答（predictions = 出走馬の辞書の並び）に市場補正を適用して返す（フラグ KEIBA_PROB_CALIBRATION=1 のときだけ。無効なら入力そのまま）。

    適用できたとき: 各馬の win_prob / top2_prob / top3_prob（と place_prob / show_prob）を補正後に置き換え、元の値を *_model に残す。
    EV（ev_win / ev_top2 / ev_place）・合成スコアと順位・印（◎○▲★△消）・較正後 EV（ev_*_cal）・買い推奨度・推奨の文言を、補正後の確率で付け直す。
    適用できないとき（オッズ未確定・予測オッズ・8 頭未満）は確率も印も変えず、理由だけ ``prob_calibration`` に入れる。
    応答に ``prob_calibration`` が既にあれば二重に適用しない。
    """
    if not is_enabled() or not isinstance(resp, dict):
        return resp
    preds = resp.get("predictions")
    if not isinstance(preds, list) or not preds or not all(isinstance(e, dict) for e in preds) or not resp.get("has_prediction", True):
        return resp
    if "prob_calibration" in resp:
        return resp
    cfg = load_config()
    meta = {"applied": False, "reason": None, "version": cfg.get("version"), "odds_phase": odds_phase, "method": cfg.get("default_method")}
    ok, reason = calibration_status(preds)
    if not ok:
        return {**resp, "prob_calibration": {**meta, "reason": reason}}
    cal = calibrate_arrays(
        {"win": [float(e["win_prob"]) for e in preds], "top2": [float(e["top2_prob"]) for e in preds], "top3": [float(e["top3_prob"]) for e in preds]},
        [float(e["win_odds"]) for e in preds],
    )
    if cal is None:
        return {**resp, "prob_calibration": {**meta, "reason": "補正できませんでした"}}
    from src.utils.race_probabilities import assign_kesi_mark, assign_mece_marks, attach_calibrated_ev, buy_recommendation_tier

    labels = {"honmei": "◎ 1着優位", "pair": "○ 連対候補", "anchor": "△ 馬券内の候補", "show_val": "▲ 単穴", "star": "★ 中穴妙味", "kesi": "消 3着内に来る確率が低い", "none": ""}
    out = []
    for i, e in enumerate(preds):
        e = dict(e)
        for k in ("win_prob", "top2_prob", "top3_prob"):
            e[f"{k}_model"] = e[k]
        pw, p2, p3 = round(float(cal["win"][i]), 4), round(float(cal["top2"][i]), 4), round(float(cal["top3"][i]), 4)
        e.update({"win_prob": pw, "top2_prob": p2, "top3_prob": p3, "estimated_prob": p3, "prob_calibrated": True})
        if "place_prob" in e:
            e["place_prob"] = p2
        if "show_prob" in e:
            e["show_prob"] = p3
        wo = float(e["win_odds"])
        t2 = e.get("top2_odds_est")
        pe = e.get("place_odds_eff")
        e["ev_win"] = round(pw * wo, 3) if pw > 0 else None
        e["ev_top2"] = round(p2 * float(t2), 3) if t2 and p2 > 0 else None
        e["ev_place"] = round(p3 * float(pe), 3) if pe and p3 > 0 else None
        e["expected_value"] = e["ev_place"]
        if p3 > 0 and e["ev_place"]:
            comp = (p3 ** alpha) * (e["ev_place"] ** (1 - alpha))
        else:
            comp = p3 if p3 > 0 else 0.0
        e["composite_score"] = round(comp, 4)
        out.append(e)
    out.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, e in enumerate(out):
        e["composite_rank"] = i + 1
    assign_mece_marks(out)
    assign_kesi_mark(out)
    for e in out:
        e["recommendation"] = labels.get(e.get("mark_type") or "none", "")
        for k in ("ev_win_cal", "ev_top2_cal", "ev_place_cal"):
            e.pop(k, None)
        attach_calibrated_ev(e)
        e["buy_tier"] = buy_recommendation_tier(e)
    new = {**resp, "predictions": out, "prob_calibration": {**meta, "applied": True}}
    if isinstance(resp.get("bet_suggestion"), (dict, list)):
        try:
            from src.utils.race_bet_suggestion import suggest_race_bets

            new["bet_suggestion"] = suggest_race_bets(out)
        except Exception:  # noqa: BLE001
            pass
    return new


def calibrate_prediction_list(preds: list[dict], single_odds: dict | None, *, odds_phase: str | None = None) -> tuple[list[dict], dict | None]:
    """推奨ポートフォリオ用: 保存済みの予測の一覧を、その時点の実オッズ（race_odds の entries）で補正して返す。
    (一覧, prob_calibration のメタ)。無効のとき・補正できないときは入力のまま（メタは無効なら None）。"""
    if not is_enabled():
        return preds, None
    odds_by_hn: dict[int, float] = {}
    for o in (single_odds or {}).get("entries") or []:
        try:
            odds_by_hn[int(o.get("horse_number"))] = float(o.get("win_odds") or 0)
        except (TypeError, ValueError):
            continue
    merged = []
    for e in preds:
        e2 = dict(e)
        try:
            hn = int(e.get("horse_number"))
        except (TypeError, ValueError):
            hn = None
        if hn in odds_by_hn and odds_by_hn[hn] > 0:
            e2["win_odds"] = odds_by_hn[hn]
            e2.pop("odds_source", None)
        merged.append(e2)
    resp = apply_prob_calibration({"predictions": merged, "has_prediction": True}, odds_phase=odds_phase)
    meta = resp.get("prob_calibration")
    return (resp["predictions"] if meta and meta.get("applied") else preds), meta
