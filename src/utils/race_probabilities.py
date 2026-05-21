"""レース内の勝率・連対率・複勝率（Harville / Plackett-Luce 近似）。"""

from __future__ import annotations

import numpy as np


def _softmax_win_probs(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return np.array([])
    centered = scores - float(np.max(scores))
    exp_s = np.exp(centered)
    s = float(exp_s.sum())
    return exp_s / s if s > 0 else np.ones(len(scores)) / len(scores)


def harville_top2_prob(win_probs: np.ndarray) -> np.ndarray:
    """各馬の連対（1〜2着以内）確率。"""
    n = len(win_probs)
    top2 = np.copy(win_probs)
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            denom = 1.0 - win_probs[j]
            if denom > 1e-9:
                top2[i] += win_probs[j] * win_probs[i] / denom
    return np.minimum(top2, 1.0)


def harville_top3_prob(win_probs: np.ndarray) -> np.ndarray:
    """各馬の複勝（1〜3着以内）確率。"""
    top2 = harville_top2_prob(win_probs)
    n = len(win_probs)
    top3 = np.copy(top2)
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            d1 = 1.0 - win_probs[j]
            if d1 < 1e-9:
                continue
            for k in range(n):
                if k in (i, j):
                    continue
                d2 = 1.0 - win_probs[j] - win_probs[k]
                if d2 > 1e-9:
                    top3[i] += win_probs[j] * (win_probs[k] / d1) * (win_probs[i] / d2)
    return np.minimum(top3, 1.0)


def derive_race_probabilities(scores: list[float]) -> list[dict[str, float]]:
    """
    モデルスコアから勝率・連対率・複勝率を算出する。

    Returns:
        [{"win_prob", "top2_prob", "top3_prob"}, ...]（scores と同順）
    """
    if not scores:
        return []
    arr = np.array(scores, dtype=float)
    win = _softmax_win_probs(arr)
    top2 = harville_top2_prob(win)
    top3 = harville_top3_prob(win)
    out: list[dict[str, float]] = []
    for i in range(len(win)):
        out.append(
            {
                "win_prob": round(float(win[i]), 4),
                "top2_prob": round(float(top2[i]), 4),
                "top3_prob": round(float(top3[i]), 4),
            }
        )
    return out


def estimate_top2_payout_odds(win_odds: float) -> float | None:
    """連対系の参考オッズ（単勝予測オッズからの経験的換算）。"""
    if not win_odds or win_odds <= 0:
        return None
    return round(float(win_odds) * 1.65, 1)


# ★: 想定単勝10〜30倍帯の妙味馬（期待値基準）
_STAR_WIN_ODDS_MIN = 10.0
_STAR_WIN_ODDS_MAX = 30.0
_STAR_EV_WIN_MIN = 1.0  # 勝率×単勝オッズ >= 1.0（100円あたり回収期待が損益分岐以上）
_STAR_EV_PLACE_MIN = 1.08  # 複勝のみ高EVの代替（単勝帯内で複勝妙味が強い場合）


def qualifies_star_mark(entry: dict, base_prob: float) -> bool:
    """想定単勝10〜30倍かつ期待値が高い馬か。"""
    win_odds = float(entry.get("win_odds") or 0)
    if win_odds < _STAR_WIN_ODDS_MIN or win_odds > _STAR_WIN_ODDS_MAX:
        return False
    ev_win = float(entry.get("ev_win") or 0)
    if ev_win >= _STAR_EV_WIN_MIN:
        return True
    ev_place = float(entry.get("ev_place") or 0)
    top3 = float(entry.get("top3_prob") or 0)
    if ev_place >= _STAR_EV_PLACE_MIN and top3 >= base_prob * 1.15:
        return True
    return False


def assign_mece_marks(entries: list[dict]) -> None:
    """
    印を付与（in-place）。◎・★は各1頭まで、○・✓・▲は複数頭可（1頭1印）。

    mark_type:
      honmei   … 1着優位（◎）※1頭のみ
      pair     … 2連系の相手優位（○）※複数可
      anchor   … 3列目の紐で優位（✓）※複数可
      show_val … 勝率低・複勝率高（▲）※複数可
      star     … 10〜30倍帯で期待値が高い（★）※1頭のみ
      none     … 印なし
    """
    n = len(entries)
    if n == 0:
        return
    base = 1.0 / max(n, 1)
    for e in entries:
        e["mark_type"] = "none"

    def _anchor_score(entry: dict) -> float:
        return float(entry.get("top3_prob") or 0) - 0.35 * float(
            entry.get("win_prob") or 0
        )

    def _show_val_gap(entry: dict) -> float:
        return float(entry.get("top3_prob") or 0) - float(entry.get("win_prob") or 0)

    # ◎ 1着優位（1頭）
    by_win = sorted(entries, key=lambda x: float(x.get("win_prob") or 0), reverse=True)
    honmei_hn: int | None = None
    if by_win and float(by_win[0].get("win_prob") or 0) >= base * 1.1:
        by_win[0]["mark_type"] = "honmei"
        honmei_hn = int(by_win[0].get("horse_number") or 0)

    # ★ 中穴妙味（1頭・◎以外）
    star_candidates = [
        e
        for e in entries
        if e.get("mark_type") == "none" and qualifies_star_mark(e, base)
    ]
    if star_candidates:
        star = max(
            star_candidates,
            key=lambda x: (
                float(x.get("ev_win") or 0),
                float(x.get("ev_place") or 0),
            ),
        )
        star["mark_type"] = "star"

    star_hn = next(
        (int(e.get("horse_number") or 0) for e in entries if e.get("mark_type") == "star"),
        None,
    )

    # ○ 2連系相手（複数）
    for e in entries:
        if e.get("mark_type") != "none":
            continue
        hn = int(e.get("horse_number") or 0)
        if hn and hn in (honmei_hn, star_hn):
            continue
        if float(e.get("top2_prob") or 0) >= base * 1.25:
            e["mark_type"] = "pair"

    # ✓ 3列紐（複数）
    for e in entries:
        if e.get("mark_type") != "none":
            continue
        if float(e.get("top3_prob") or 0) >= base * 1.35 and _anchor_score(e) >= base * 0.5:
            e["mark_type"] = "anchor"

    # ▲ 複勝妙味（複数）
    for e in entries:
        if e.get("mark_type") != "none":
            continue
        gap = _show_val_gap(e)
        if gap >= base * 0.45 and float(e.get("win_prob") or 0) < base * 1.05:
            e["mark_type"] = "show_val"


def buy_recommendation_tier(entry: dict) -> str:
    """買い判断推奨度（期待値の最大を基準）。"""
    evs = [
        float(entry.get("ev_win") or 0),
        float(entry.get("ev_top2") or 0),
        float(entry.get("ev_place") or 0),
    ]
    best = max(evs) if evs else 0.0
    if best >= 1.2:
        return "強推奨"
    if best >= 1.05:
        return "推奨"
    if best >= 0.9:
        return "様子見"
    return "—"
