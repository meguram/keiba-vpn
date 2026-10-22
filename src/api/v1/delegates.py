"""Flask /api/v1 向けビジネスロジック（FastAPI 非依存）。"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

logger = logging.getLogger(__name__)

# 1 レースの購入合計の既定上限（円）。UI の既定値と揃える（frontend/components/BetRecommend.tsx）。
DEFAULT_MAX_TOTAL_BET = 2000


def _predicted_odds_for(c, predicted: dict | None) -> float | None:
    """単勝・複勝候補の予測オッズ（複勝は下限、既存の fukusho_use_min_odds 方針）。"""
    if not predicted or c.bet_type not in ("tansho", "fukusho"):
        return None
    for e in predicted.get("entries") or []:
        if e.get("horse_number") == c.pair[0]:
            v = e.get("win_odds") if c.bet_type == "tansho" else e.get("place_odds_min")
            return float(v) if v else None
    return None


def optimize_betting(body: dict[str, Any], *, view: str = "full", market: bool = True, forced_odds: dict | None = None,
                     _ctx: dict | None = None) -> tuple[dict[str, Any], int]:
    """推奨ポートフォリオ。view="paid" は会員向け（推奨度の段階と買い目の区分だけ。オッズ・期待値・金額の数値は出さない）。
    market=False は実オッズに触れない（市場補正もしない）。forced_odds={"single":…, "pairs":…} は想定オッズで計算する（有料ゾーン案B）。"""
    if view == "paid":
        return _optimize_paid(body)
    from src.pipeline.inference.betting import BettingConfig, BettingOptimizer
    from src.pipeline.inference.odds_source import predicted_single_odds, resolve_odds
    from src.pipeline.inference.race_prediction_service import load_cached
    from src.scraper.storage import HybridStorage

    race_id = body.get("race_id", "")
    if not race_id:
        return {"error": "race_id is required"}, 400

    storage = HybridStorage()
    cached = load_cached(storage, race_id)
    predictions = (cached or {}).get("predictions") or []
    already_calibrated = False
    if not predictions:
        # 画面（/api/race/{id}/predictions）と同じく、GCS に無ければホストのローカル予測（shadow / バックフィル）を使う
        try:
            from src.api.local_predictions import load_local_prediction_payload

            local = load_local_prediction_payload(race_id, storage, market=market)
        except Exception as exc:  # noqa: BLE001
            logger.warning("推奨用のローカル予測の読み込みに失敗 %s: %s", race_id, exc)
            local = None
        predictions = (local or {}).get("predictions") or []
        already_calibrated = bool(predictions)  # 読み出し時に市場補正済み（二重に補正しない）
    if not predictions:
        return {"error": "予測結果がありません。先に推論を実行してください"}, 404
    if _ctx is not None:
        _ctx["predictions"] = predictions
    if forced_odds is not None:  # 想定オッズで計算するときは、予測に付いている実オッズ由来の項目を使わない
        from src.utils.paid_zone import strip_market

        predictions = strip_market(predictions)

    odds_mode = str(body.get("odds_mode") or "auto").lower()
    odds_phase = str(body.get("odds_phase") or "latest").lower()
    if forced_odds is not None:
        odds = {"source": "predicted", "phase": None, "as_of": None, "available": [], "note": "想定オッズで計算", "allowed_bet_types": None,
                "single_odds": forced_odds["single"], "pair_odds": forced_odds["pairs"]}
    else:
        odds = resolve_odds(storage, race_id, mode=odds_mode, phase=odds_phase)
    if odds["source"] is None and odds_mode in ("actual", "predicted"):
        what = "実オッズ" if odds_mode == "actual" else "予測オッズ"
        return {
            "error": f"{what}がありません（odds_mode={odds_mode}）。{odds['note']}",
            "available_odds": odds["available"],
        }, 404

    bet_types = body.get("bet_types", ["tansho", "fukusho", "umaren", "wide"])
    allowed = odds["allowed_bet_types"]
    if allowed is not None:
        bet_types = [b for b in bet_types if b in allowed]
        if not bet_types:
            return {
                "error": "予測オッズでは単勝・複勝のみ評価できます。bet_types に tansho / fukusho を含めてください",
                "available_odds": odds["available"],
            }, 400

    pair_odds = odds["pair_odds"]
    single_odds = odds["single_odds"]
    if odds["source"] is None and forced_odds is None:  # auto でオッズ無し: 従来どおり predictions 内のオッズ欄にフォールバック
        pair_odds = storage.load("race_pair_odds", race_id) or {}
        single_odds = storage.load("race_odds", race_id)

    # 市場補正（KEIBA_PROB_CALIBRATION=1 のときだけ）: 実オッズがあるとき、推奨の確率を補正後にする（画面の確率・EV・印と同じ）
    prob_cal_meta = None
    if odds["source"] == "actual" and not already_calibrated:
        try:
            from src.utils.prob_calibration import calibrate_prediction_list

            predictions, prob_cal_meta = calibrate_prediction_list(predictions, single_odds, odds_phase=odds_phase)
        except Exception as exc:  # noqa: BLE001
            logger.warning("推奨の確率の市場補正に失敗 %s: %s", race_id, exc)

    # 1 レースの購入合計の上限（円）。未指定は 2000 円。null / 0 は上限なし。
    raw_cap = body.get("max_total_bet", DEFAULT_MAX_TOTAL_BET)
    max_total_bet: int | None = None
    if raw_cap not in (None, 0):
        try:
            max_total_bet = int(raw_cap)
        except (TypeError, ValueError):
            max_total_bet = -1
        if max_total_bet < 100:
            return {"error": "max_total_bet は 100 円以上の整数で指定してください（上限なしは null か 0）"}, 400

    config = BettingConfig(
        max_total_bet=max_total_bet,
        bet_types=bet_types,
        kelly_fraction=body.get("kelly_fraction", 0.25),
        min_ev=body.get("min_ev", 1.05),
        min_prob=body.get("min_prob", 0.02),
        max_candidates=body.get("max_candidates", 15),
        top_n_for_pairs=body.get("top_n_for_pairs", 6),
    )
    optimizer = BettingOptimizer(config)
    portfolio = optimizer.optimize(
        predictions,
        pair_odds,
        body.get("bankroll", 100000),
        single_odds=single_odds,
    )

    predicted = None
    if odds["source"] == "actual" and forced_odds is None:
        predicted = predicted_single_odds(storage, race_id)

    candidates_out = []
    for c in portfolio["candidates"]:
        item = {
            "bet_type": c.bet_type,
            "pair": list(c.pair),
            "pair_label": c.pair_label,
            "horse_names": list(c.horse_names),
            "odds": c.odds,
            "prob": c.prob,
            "ev": c.ev,
            "kelly_fraction": c.kelly_fraction,
            "bet_amount": c.bet_amount,
            "expected_return": c.expected_return,
        }
        pred_odds = _predicted_odds_for(c, predicted)
        if pred_odds:
            item["predicted_odds"] = pred_odds
            item["odds_gap"] = round(c.odds / pred_odds - 1, 3)
        candidates_out.append(item)

    # 想定オッズ（案B）の EV は、実オッズで当てはめた較正に通さない
    calibrated = {} if forced_odds is not None else _calibrate_candidates(candidates_out, bool(prob_cal_meta and prob_cal_meta.get("applied")))
    return {
        "race_id": race_id,
        "bankroll": body.get("bankroll", 100000),
        "total_bet": portfolio["total_bet"],
        "expected_return": portfolio["expected_return"],
        "roi_pct": round(portfolio["expected_roi"] * 100, 1),
        **calibrated,
        "max_total_bet": max_total_bet,
        "probability_model": portfolio.get("probability_model"),
        "candidates": candidates_out,
        "odds_source": odds["source"],
        "odds_phase": odds["phase"],
        "odds_as_of": odds["as_of"],
        "available_odds": odds["available"],
        "odds_note": odds["note"],
        **({"prob_calibration": prob_cal_meta} if prob_cal_meta else {}),
    }, 200


def _paid_error(payload: dict[str, Any], code: int) -> tuple[dict[str, Any], int]:
    return {"view": "paid", "error": str(payload.get("error") or "推奨馬券を計算できません")}, code


def _predicted_singles(storage, race_id: str) -> dict | None:
    """想定単勝・複勝。キャッシュに無ければその場で推論して保存する（final-odds API と同じ get_or_compute。約 0.6 秒/レース）。
    キャッシュが空だと有料ゾーンが黙って案A（確率のみ）に落ち、推奨度が「推奨しない」に偏る（2026-10-22 の不具合）。"""
    from src.pipeline.inference.odds_source import predicted_single_odds

    single = predicted_single_odds(storage, race_id)
    if single:
        return single
    try:
        from src.pipeline.inference import final_odds_service

        final_odds_service.get_or_compute(storage, race_id)
    except Exception as exc:  # noqa: BLE001  推論できないレースは案Aに落とす
        logger.warning("想定オッズの推論に失敗 %s: %s", race_id, exc)
        return None
    return predicted_single_odds(storage, race_id)


def _paid_marks(preds: list[dict], single: dict | None) -> list[dict]:
    """有料ゾーン案A・Bの印: 実オッズ由来の項目を落とし、想定オッズ（B）または確率だけ（A）で付け直す。"""
    from src.api.local_predictions import score_entries
    from src.utils import paid_zone as PZ

    try:
        return score_entries(PZ.strip_market(preds), PZ.predicted_odds_map(single))
    except Exception as exc:  # noqa: BLE001  印が付けられなくても買い目は出す
        logger.warning("有料ゾーンの印の計算に失敗: %s", exc)
        return PZ.strip_market(preds)


def _optimize_paid(body: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """会員向けの推奨馬券。レース単位の推奨度（推奨ポートフォリオ全体の期待回収率を段階にしたもの）と、
    買い目（券種ごとの組み合わせ＋区分 本線/押さえ）だけを返す。オッズ・期待値・確率・金額（Kelly 配分）の数値は出さない。
    方式は config/paid_zone.json の mode（A=確率のみ／B=想定オッズ／C=実オッズを内部で使う）。"""
    from src.scraper.storage import HybridStorage
    from src.utils import paid_zone as PZ

    race_id = body.get("race_id", "")
    if not race_id:
        return {"view": "paid", "error": "race_id is required"}, 400
    raw_cap = body.get("max_total_bet", PZ.default_max_total_yen())
    cap: int | None = None
    if raw_cap not in (None, 0):
        try:
            cap = int(raw_cap)
        except (TypeError, ValueError):
            cap = -1
        if cap < 100:
            return {"view": "paid", "error": "max_total_bet は 100 円以上の整数で指定してください（上限なしは null か 0）"}, 400

    m = PZ.mode()
    bts = PZ.bet_types()
    base = {k: v for k, v in body.items() if k in ("race_id", "odds_phase", "min_prob", "kelly_fraction", "max_candidates", "top_n_for_pairs")}
    base.update({"bet_types": bts, "max_total_bet": None, "bankroll": 100000})
    ctx: dict[str, Any] = {}
    storage = HybridStorage()

    if m == "C":
        full, code = optimize_betting({**base, "odds_mode": "actual"}, _ctx=ctx)
        if code == 200 and full.get("odds_source") == "actual":
            total = full.get("total_bet") or 0
            roi = full.get("expected_roi_calibrated")
            if roi is None and total:
                roi = (full.get("expected_return") or 0) / total
            preds = ctx.get("predictions") or []
            return PZ.build_paid_payload(full.get("candidates") or [], preds, race_id=race_id, roi=roi, m="C", max_total_bet=cap), 200
        m = "B"  # 実オッズが無いレースは想定オッズで

    single = PZ.calibrate_single(_predicted_singles(storage, race_id)) if m == "B" else None
    if m == "B" and not single:
        logger.warning("有料ゾーン: %s は想定単勝が得られないため案A（確率のみ）で出します", race_id)
        m = "A"
    if m == "B":
        win = {int(e["horse_number"]): float(e["win_odds"]) for e in single["entries"] if e.get("win_odds")}
        forced = {"single": single, "pairs": PZ.synth_pair_odds(win)}
        full, code = optimize_betting({**base}, market=False, forced_odds=forced, _ctx=ctx)
        if code != 200:
            return _paid_error(full, code)
        total = full.get("total_bet") or 0
        roi = (full.get("expected_return") or 0) / total if total else None
        preds = _paid_marks(ctx.get("predictions") or [], single)
        return PZ.build_paid_payload(full.get("candidates") or [], preds, race_id=race_id, roi=roi, m="B", max_total_bet=cap), 200

    # A: 確率のみ
    from src.api.local_predictions import load_local_prediction_payload
    from src.pipeline.inference.race_prediction_service import load_cached

    cached = load_cached(storage, race_id)
    preds = (cached or {}).get("predictions") or []
    if not preds:
        local = load_local_prediction_payload(race_id, storage, market=False)
        preds = (local or {}).get("predictions") or []
    if not preds:
        return {"view": "paid", "error": "予測結果がありません。先に推論を実行してください"}, 404
    marks = _paid_marks(preds, None)
    return PZ.build_paid_payload(PZ.prob_only_candidates(marks), marks, race_id=race_id, roi=PZ.confidence_index(marks), m="A", max_total_bet=cap), 200


def _levels_now():
    from src.utils.ev_calibration import value_levels

    return value_levels()


def _calibrate_candidates(candidates_out: list[dict], prob_calibrated: bool = False) -> dict:
    """推奨買い目に、較正後の EV（ev_calibrated）と購入額加重の期待回収率を付ける。EV の較正が無効のときは何も付けない。
    確率を市場補正したレース（prob_calibrated）は、補正後の確率から作った EV に当てはめ直した版（config/ev_calibration_pc.json）を使う（二重に補正しない）。"""
    from src.utils.ev_calibration import LABEL, active_config, calibrate_portfolio

    cfg = active_config(prob_calibrated)
    cal = calibrate_portfolio(candidates_out, cfg)
    if cal is None:
        return {}
    kinds = cfg.get("kinds") or {}
    for item, e in zip(candidates_out, cal["evs"]):
        item["ev_calibrated"] = round(e, 3)
        spec = kinds.get(f"cand_{item['bet_type']}")
        item["ev_calibration_validated"] = bool(spec and spec.get("validated"))
    rng = cfg.get("range") or {}
    return {"expected_roi_calibrated": cal["expected_roi"], "expected_return_calibrated": cal["expected_return"],
            "roi_pct_calibrated": None if cal["expected_roi"] is None else round(cal["expected_roi"] * 100, 1),
            "roi_calibrated_capped": bool(cal.get("capped")), "ev_label": LABEL,
            "roi_range_calibrated": [rng["lo"], rng["hi"]] if "lo" in rng and "hi" in rng else None}


# ── レースの「価値」一覧（レース選択の見た目用） ──

# 推奨ポートフォリオの期待回収率（= Σ金額×期待値 / 購入合計）でレースを 4 段階にする。
# hot: 特に高い / good: 高い / ok: プラス期待 / pass: 見送り水準（買い目なし・期待回収率が基準未満）
VALUE_LEVELS: tuple[tuple[str, float], ...] = (("hot", 1.5), ("good", 1.25), ("ok", 1.05))
_VALUE_CACHE: dict[tuple, tuple[float, dict]] = {}

from src.utils import cache_epoch as _cache_epoch  # noqa: E402

_cache_epoch.register("delegates.value_cache", _VALUE_CACHE.clear)
_VALUE_TTL_SEC = 20.0
_VALUE_TTL_PAST_SEC = 6 * 3600.0


def value_level(expected_roi: float | None, n_bets: int, prob_calibrated: bool = False) -> str:
    """expected_roi は呼び出し側で較正した値を渡す（EV の較正が有効なら較正後の期待回収率、無効なら従来の値）。閾値も較正の設定に従う。
    確率を市場補正したレース（prob_calibrated）は、補正後のポートフォリオで作り直した段（config/ev_calibration_pc.json）を使う。"""
    from src.utils.ev_calibration import active_config, value_levels

    if not n_bets or expected_roi is None:
        return "pass"
    for name, floor in value_levels(active_config(prob_calibrated)):
        if expected_roi >= floor:
            return name
    return "pass"


def _value_row(race: dict[str, Any], payload: dict[str, Any], code: int) -> dict[str, Any]:
    rid = race["race_id"]
    base = {"race_id": rid, "round": race.get("round"), "venue": race.get("venue"), "race_name": race.get("race_name")}
    if code != 200:
        return {**base, "level": "none", "reason": payload.get("error", "計算できません"), "expected_roi": None, "n_bets": 0}
    cands = payload.get("candidates") or []
    total = payload.get("total_bet") or 0
    roi_raw = (payload.get("expected_return") or 0) / total if total else None
    roi_cal = payload.get("expected_roi_calibrated")             # EV の較正が有効なときだけ入る
    roi = roi_cal if roi_cal is not None else roi_raw
    ev_key = "ev_calibrated" if roi_cal is not None else "ev"
    top = max(cands, key=lambda c: c.get(ev_key) or 0) if cands else None
    pcal = bool((payload.get("prob_calibration") or {}).get("applied"))
    try:  # 開発者の表示にも、会員と同じ 5 段階の推奨度（見送り〜最注目）を添える（較正後の期待回収率から）
        from src.utils import paid_zone as PZ

        tl = PZ.level_for_roi(roi if cands else None, "C")
        tier = next((dict(t) for t in PZ.tiers() if int(t.get("level", -1)) == tl), {"level": tl, "label": ""})
    except Exception:  # noqa: BLE001
        tier = None
    return {
        **base,
        "tier": tier,
        "level": value_level(roi, len(cands), pcal),
        "prob_calibrated": pcal,
        "expected_roi": None if roi is None else round(roi, 3),
        "expected_roi_raw": None if roi_raw is None else round(roi_raw, 3),
        "calibrated": roi_cal is not None,
        "max_ev": None if top is None else round(top.get(ev_key) or 0, 3),
        "n_bets": len(cands),
        "total_bet": total,
        "top_bet": None if top is None else {"bet_type": top["bet_type"], "label": top["pair_label"], "ev": round(top.get(ev_key) or 0, 3)},
        "odds_source": payload.get("odds_source"),
        "odds_phase": payload.get("odds_phase"),
    }


HOT_TOP_FRACTION = 0.2  # hot にできるのは、その日の期待回収率が上位 20% のレースだけ


def _apply_day_ranking(rows: list[dict[str, Any]]) -> None:
    """その日の期待回収率で順位を付け、hot は「絶対水準を超え、かつ上位 20%」に絞る（全レースが hot になって区別できなくなるのを防ぐ）。"""
    # 確率を市場補正したレースとしないレースは、期待回収率の水準（ポートフォリオ・較正・段の閾値）が違うので、順位は別々に付ける（補正の機能が無効なら全レースが同じ組）
    for flag in (False, True):
        group = [r for r in rows if bool(r.get("prob_calibrated")) == flag]
        ranked = sorted((r for r in group if r.get("expected_roi") is not None), key=lambda r: -r["expected_roi"])
        cutoff = max(1, math.ceil(len(ranked) * HOT_TOP_FRACTION))
        for i, r in enumerate(ranked, 1):
            r["rank"] = i
            if r["level"] == "hot" and i > cutoff:
                r["level"] = "good"
    for r in rows:
        r.setdefault("rank", None)


_PAID_LEVEL_STYLE = {4: "hot", 3: "good", 2: "ok", 1: "ok", 0: "pass"}


def _value_row_paid(race: dict[str, Any], payload: dict[str, Any], code: int) -> dict[str, Any]:
    """会員向け: 期待回収率などの数値は出さず、レース単位の推奨度（段階）だけ。"""
    base = {"race_id": race["race_id"], "round": race.get("round"), "venue": race.get("venue"), "race_name": race.get("race_name")}
    if code != 200:
        return {**base, "level": "none", "reason": "計算できません", "n_bets": 0, "tier": None}
    tier = payload.get("race_tier") or {"level": 0, "label": "見送り"}
    return {**base, "level": _PAID_LEVEL_STYLE.get(int(tier.get("level") or 0), "pass"), "tier": tier,
            "n_bets": len(payload.get("bets") or [])}


def _value_ttl(date: str) -> float:
    """終わった開催日は結果が動かないので長く使い回す（データが保存されれば cache_epoch で空になる）。当日・先の日は短く。"""
    from datetime import datetime, timedelta, timezone

    today = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d")
    return _VALUE_TTL_SEC if date >= today else _VALUE_TTL_PAST_SEC


_VALUE_WORKERS = 6


def _map_races(fn, races: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """レースごとの計算（ストレージの読み込みで待つ時間が大半）を並列にして、順序は保つ。"""
    from concurrent.futures import ThreadPoolExecutor

    todo = [r for r in races if r.get("race_id")]
    if len(todo) <= 1:
        return [fn(r) for r in todo]
    with ThreadPoolExecutor(max_workers=min(_VALUE_WORKERS, len(todo))) as ex:
        return list(ex.map(fn, todo))


def race_value_summary(
    date: str,
    *,
    odds_mode: str = "auto",
    odds_phase: str = "latest",
    max_total_bet: int | None = DEFAULT_MAX_TOTAL_BET,
    view: str = "full",
) -> dict[str, Any]:
    """開催日の全レースについて、推奨買い目の期待回収率とレベルを返す（レース選択の強調表示に使う）。
    view="paid" は会員向け（推奨度の段階だけ。期待回収率の数値は出さない）。"""
    import time

    from src.scraper.storage import HybridStorage

    key = (date, odds_mode, odds_phase, max_total_bet, view)
    hit = _VALUE_CACHE.get(key)
    if hit and time.time() - hit[0] < _value_ttl(date):
        return hit[1]

    storage = HybridStorage()
    races = (storage.load("race_lists", date) or {}).get("races") or []
    if view == "paid":
        from src.utils import paid_zone as PZ

        def _one_paid(r: dict[str, Any]) -> dict[str, Any]:
            try:
                payload, code = optimize_betting({"race_id": r["race_id"], "max_total_bet": max_total_bet}, view="paid")
            except Exception as exc:  # 1 レースの失敗で一覧全体を落とさない
                logger.warning("race value(paid) 計算失敗 %s: %s", r["race_id"], exc)
                payload, code = {"error": str(exc)}, 500
            return _value_row_paid(r, payload, code)

        prow = _map_races(_one_paid, races)
        out = {"date": date, "view": "paid", "legend": PZ.legend(), "max_total_bet": max_total_bet, "races": prow}
        _VALUE_CACHE[key] = (time.time(), out)
        return out
    def _one_full(r: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "race_id": r["race_id"], "bankroll": 100000, "odds_mode": odds_mode, "odds_phase": odds_phase,
            "max_total_bet": max_total_bet, "bet_types": ["tansho", "fukusho", "umaren", "wide", "sanrenpuku"],
        }
        try:
            payload, code = optimize_betting(body)
        except Exception as exc:  # 1 レースの失敗で一覧全体を落とさない
            logger.warning("race value 計算失敗 %s: %s", r["race_id"], exc)
            payload, code = {"error": str(exc)}, 500
        return _value_row(r, payload, code)

    rows = _map_races(_one_full, races)
    _apply_day_ranking(rows)
    out = {
        "date": date,
        "levels": [{"level": n, "min_roi": f} for n, f in _levels_now()],
        "max_total_bet": max_total_bet,
        "races": rows,
    }
    n_pc = sum(1 for r in rows if r.get("prob_calibrated"))
    if n_pc:       # 市場補正が有効なとき: 補正したレース（実オッズあり）としないレース（予測オッズ・オッズ未確定）の水準が混ざらないよう、段と順位を別に付けた旨を返す
        from src.utils.ev_calibration import active_config, value_levels

        out["levels_prob_calibrated"] = [{"level": n, "min_roi": f} for n, f in value_levels(active_config(True))]
        out["prob_calibration"] = {"races_applied": n_pc, "races_not_applied": len(rows) - n_pc, "mixed": 0 < n_pc < len(rows),
                                   "note": "市場補正した確率のレース（実オッズあり）としないレースは、期待回収率の水準が違うため、段の閾値と当日の順位を別に付けています。"}
    _VALUE_CACHE[key] = (time.time(), out)
    return out


def get_final_odds(race_id: str, refresh: bool = False) -> dict[str, Any]:
    from src.pipeline.inference.final_odds_service import get_or_compute
    from src.scraper.storage import HybridStorage

    return get_or_compute(
        HybridStorage(),
        race_id,
        force_refresh=refresh,
        allow_scrape=refresh,
    )


def get_horse_aptitude(horse_id: str | None, horse_name: str | None) -> tuple[dict, int]:
    if not horse_id and not horse_name:
        return {"error": "horse_id or horse_name required"}, 400
    try:
        from src.research.pedigree.horse_aptitude_profile import HorseAptitudeProfileCalc

        calc = HorseAptitudeProfileCalc()
        result = calc.compute(horse_id=horse_id, horse_name=horse_name)
    except FileNotFoundError as exc:
        return {"error": f"artifacts not found: {exc}"}, 500
    if "error" in result:
        return result, 404
    return result, 200


def get_race_note_3d_v2(race_id: str) -> tuple[dict, int]:
    from src.research.pedigree.race_note_3d_v2 import build_race_note_v2
    from src.scraper.storage import HybridStorage

    try:
        payload = build_race_note_v2(HybridStorage(), race_id)
    except Exception as exc:
        logger.exception("race-note-3d-v2 failed")
        return {"error": str(exc)}, 500
    if payload.get("error"):
        return payload, 404
    return payload, 200


def query_pedigree_race_stats(params: dict[str, Any]) -> tuple[dict, int]:
    """血統×レース条件クエリ — 内部 FastAPI ブリッジまたは直接 SQLite。"""
    bridge = os.environ.get("KEIBA_LEGACY_API", "").strip()
    if bridge:
        import httpx

        r = httpx.get(
            f"{bridge.rstrip('/')}/api/pedigree-race-stats/query",
            params=params,
            timeout=60.0,
        )
        return r.json(), r.status_code

    try:
        from src.pipeline.data.bloodline_sqlite import get_connection, load_race_result_slim
        import pandas as pd

        slim = load_race_result_slim(get_connection())
        if slim is None or slim.empty:
            return {
                "error": "インデックス未生成。build_pedigree_race_index + migrate_bloodline_to_sqlite を実行してください。"
            }, 503

        df = slim.copy()
        if params.get("date_from"):
            df = df[df["date"] >= params["date_from"]]
        if params.get("date_to"):
            df = df[df["date"] <= params["date_to"]]
        if params.get("venues"):
            vlist = [v.strip() for v in str(params["venues"]).split(",") if v.strip()]
            df = df[df["venue"].isin(vlist)]

        top_n = int(params.get("top_n", 50))
        return {
            "total_entries": int(len(df)),
            "unique_horses": int(df["horse_id"].nunique()) if "horse_id" in df.columns else 0,
            "cat1": [],
            "cat2": [],
            "cat3": [],
            "note": "簡易集計。フル血統クロス集計は KEIBA_LEGACY_API で FastAPI ブリッジを使用",
        }, 200
    except Exception as exc:
        return {"error": str(exc)}, 500
