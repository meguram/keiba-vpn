"""凍結した買い方（第 2 回 frozen_20261117_r2、hash 70ba29fa7b55）の当日の判定。

第 2 段のモデルの p（勝率）・q（T-60 の単勝オッズの市場の勝率）・r=log(p/q) と、T-60 の人気から、
レースごとに「買う（◎ の単勝 100 円）／見送り（理由つき）」を決める。

規則はバックテストの道具 ``src/scripts/data/bet_cycles.py`` の関数（normalize_probs・_r_features・apply_qcuts・
_skip・pick_honmei・play）をそのまま呼んで、バックテストと 1 行ずつ同じになるようにする（写さない）。
見送りの理由だけは、play が 1 種類（「見送り（条件）」）で返すものを、同じ順で分けて返す。

事象と根拠: 凍結の買い方は「未勝利・新馬で、2 番人気以下・勝率 5% 以上の馬の r の最大が訓練の 5 分位の 3 段目以上
（r_max ≥ 0.4917）のレースだけ、その馬の単勝を買う」。市場（T-60 の単勝オッズ）より第 2 段のモデルが高く見る馬ほど
回収率が高い、という訓練・評価・2025 年 test の結果（test 107.0%・区間 94.0〜119.6%）に基づく。回収率は確かめ中。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FROZEN_PATH = ROOT / "config/bet_cycles/frozen_20261117_r2.json"
FROZEN_HASH = "70ba29fa7b55"
STRATEGY = "stage2_r2"

# 見送りの理由（コード → 画面の文言）。判定の順もこの順（バックテストの _skip → pick_honmei と同じ）
SKIP_REASONS = {
    "field": "見送り（頭数が 5 頭未満）",
    "r_max": "見送り（市場とのずれが小さい）",
    "class": "見送り（未勝利・新馬でない）",
    "no_candidate": "見送り（候補の馬なし）",
    "no_odds_t60": "見送り（T-60 オッズ未着）",
    "no_prediction": "見送り（予測なし）",
    "odds_broken": "見送り（オッズの取り込みの崩れ）",
    "stage2_error": "見送り（第 2 段の計算に失敗）",
    "deadline": "見送り（発走前の期限までに材料がそろわない）",
}


class FrozenMismatch(RuntimeError):
    pass


def load_frozen(path: Path | str | None = None, want_hash: str = FROZEN_HASH) -> dict:
    """凍結の JSON を読み、中身から hash を計算し直して、記録の hash と want_hash に一致することを確かめる。"""
    from src.scripts.data.bet_cycles import strategy_hash

    p = Path(path or os.environ.get("KEIBA_STAGE2_FROZEN") or FROZEN_PATH)
    body = json.loads(p.read_text(encoding="utf-8"))
    h = strategy_hash({k: v for k, v in body.items() if k not in ("frozen_at", "hash")})
    if h != body.get("hash") or h != want_hash:
        raise FrozenMismatch(f"凍結の hash が合わない: 計算 {h}・記録 {body.get('hash')}・期待 {want_hash}（{p}）")
    return body


def skip_result(reason: str, *, race_id: str, frozen_hash: str = FROZEN_HASH, **extra: Any) -> dict:
    """材料がそろわないときなどの見送り（判定の関数を通さない）。"""
    return {"strategy": STRATEGY, "race_id": str(race_id), "frozen_hash": frozen_hash, "status": "skip",
            "skip_reason": reason, "skip_label": SKIP_REASONS.get(reason, f"見送り（{reason}）"), "bets": [], "horse_marks": {}, **extra}


def decide(*, race_id: str, horse_numbers, p_win, q, r, pop, odds, grade: str, frozen: dict,
           p_top2=None, p_top3=None) -> dict:
    """1 レースの判定。配列は出走馬（取消を除く）の並び（順は問わない。中で予測の降順に並べ替える）。

    - p_win・p_top2・p_top3: 第 2 段のモデルの予測。q: T-60 の市場の勝率。r: log(p_win/q)（無い馬は NaN）。
    - pop: T-60 の人気（q の順位。オッズの無い馬は 99）。odds: T-60 の単勝オッズ（無い馬は NaN）。
    - grade: 出馬表の grade（「新馬」「未勝利」「1勝」…）。class_level でバックテストと同じクラスにする。
    """
    from src.scripts.data import bet_cycles as bc
    from src.scripts.data.rank_condition_analysis import class_level

    hn = np.asarray(horse_numbers, dtype=int)
    pw0 = np.asarray(p_win, dtype=float)
    if len(hn) == 0 or not np.all(np.isfinite(pw0)):
        return skip_result("no_prediction", race_id=race_id, frozen_hash=frozen["hash"])
    order = np.lexsort((hn, -pw0))                       # 勝率の降順・同じ値は馬番の小さい順（load_runners と同じ）
    take = lambda a: None if a is None else np.asarray(a, dtype=float)[order]  # noqa: E731
    hn, pw0 = hn[order], pw0[order]
    p3_in, p2_in = take(p_top3), take(p_top2)
    pw, p3 = bc.normalize_probs(pw0, p3_in)
    p2 = bc.p2_from(pw, p3, p2_in)
    qv, rv, popv, oddv = take(q), take(r), take(pop), take(odds)
    popv = np.where(np.isfinite(popv), popv, 99.0)
    race = bc.Race(str(race_id), "", hn, pw, p3, oddv, popv, np.full(len(hn), 99.0), {bt: {} for bt in bc.BET_TYPES}, p2, qv, rv)
    f: dict[str, Any] = {"field": int(len(hn)), **bc._r_features(race, pw, popv)}
    cuts = {k: v for k, v in (frozen.get("qcuts") or {}).items() if k in f}
    f.update({k: v for k, v in bc.apply_qcuts(pd.DataFrame([f]), cuts).iloc[0].to_dict().items() if k.startswith("q_")})
    f["c_class"] = class_level(grade)
    spec = frozen["spec"]
    res = bc.play(race, f, spec)                         # バックテストと同じ実行（買うか・◎・買い目）
    base = {"strategy": STRATEGY, "race_id": str(race_id), "frozen_hash": frozen["hash"], "field": f["field"], "c_class": f["c_class"],
            "r_max": _num(f.get("r_max")), "q_r_max": int(f.get("q_r_max", 0)),
            "horses": [{"horse_number": int(hn[i]), "p_win": _num(pw[i]), "q": _num(qv[i]), "r": _num(rv[i]), "pop": _num(popv[i]),
                        "odds": _num(oddv[i])} for i in range(len(hn))]}
    if res.get("H") is None:
        code = _skip_code(f, spec.get("skip") or {}, res.get("reason", ""))
        return {**base, "status": "skip", "skip_reason": code, "skip_label": SKIP_REASONS.get(code, res.get("reason", "見送り")),
                "reason_bt": res.get("reason"), "bets": [], "horse_marks": {}}
    H = int(res["H"])
    marks = {str(int(hn[i])): bc.MARK_NAMES[min(j, len(bc.MARK_NAMES) - 1)] for j, i in enumerate([H] + list(res.get("O") or []))}
    return {**base, "status": "buy", "skip_reason": None, "skip_label": None, "reason_bt": res.get("reason"),
            "honmei": {"horse_number": int(hn[H]), "p_win": _num(pw[H]), "q": _num(qv[H]), "r": _num(rv[H]), "pop": _num(popv[H]),
                       "odds": _num(oddv[H])},
            "bets": [{"bet_type": b["bet_type"], "combo": [int(x) for x in b["combo"]], "yen": int(b["yen"])} for b in res["bets"]],
            "horse_marks": marks}


def _skip_code(f: dict, sk: dict, reason_bt: str) -> str:
    """play の見送りを、_skip と同じ順で理由のコードに分ける（判定の結果は play のもの）。"""
    from src.scripts.data import bet_cycles as bc

    if f["field"] < int(sk.get("min_field", 5)):
        return "field"
    if bc._skip(f, {"min_field": 0, "when": sk.get("when") or {}}):
        return "r_max"
    if bc._skip(f, {"min_field": 0, "in": sk.get("in") or {}, "not_in": sk.get("not_in") or {}}):
        return "class"
    if "候補なし" in reason_bt:
        return "no_candidate"
    return "no_candidate"


def market_q(odds, beta: float = 1.0) -> np.ndarray:
    """T-60 の単勝オッズ（1 レースの出走馬）から市場の勝率 q（合計 1）。オッズの無い馬・0 以下は NaN（model_cycles_market.q_from_odds と同じ）。"""
    o = np.asarray(odds, dtype=float)
    inv = np.where(np.isfinite(o) & (o > 0), 1.0 / np.where(o > 0, o, 1.0), np.nan) ** beta
    s = np.nansum(inv)
    return inv / s if s > 0 else inv


def popularity_from_q(q) -> np.ndarray:
    """q の降順の順位（1 始まり・同じ値は前の馬から）。q の無い馬は 99（apply2026 の rank(method="first") と同じ）。"""
    qv = np.asarray(q, dtype=float)
    out = np.full(len(qv), 99.0)
    ok = np.flatnonzero(np.isfinite(qv))
    if len(ok):
        rk = pd.Series(qv[ok]).rank(ascending=False, method="first").to_numpy()
        out[ok] = rk
    return out


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None
