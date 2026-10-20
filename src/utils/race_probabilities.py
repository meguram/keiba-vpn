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


_EPS = 1e-9


def harville_top2_prob(win_probs: np.ndarray) -> np.ndarray:
    """各馬の連対（1〜2着以内）確率。O(n)。"""
    p = np.asarray(win_probs, dtype=float)
    d = 1.0 - p
    ratio = np.where(d > _EPS, p / np.where(d > _EPS, d, 1.0), 0.0)  # p_j / (1 - p_j)
    # top2_i = p_i + p_i * sum_{j != i} p_j / (1 - p_j)
    top2 = p + p * (ratio.sum() - ratio)
    return np.minimum(top2, 1.0)


def harville_top3_prob(win_probs: np.ndarray) -> np.ndarray:
    """各馬の複勝（1〜3着以内）確率。三重ループを避けた O(n^2)。

    3着項 sum_{j!=i} sum_{k!=i,j} p_j * p_k/(1-p_j) * p_i/(1-p_j-p_k) は、
    i に依存するのは p_i だけなので、ペア行列 A[j,k] を 1 度作って
    i を含むペアの行和・列和を引けばよい。
    """
    p = np.asarray(win_probs, dtype=float)
    top2 = harville_top2_prob(p)
    n = len(p)
    if n < 3:
        return top2
    d1 = 1.0 - p[:, None]                       # 1 - p_j
    d2 = 1.0 - p[:, None] - p[None, :]          # 1 - p_j - p_k
    ok = (d1 > _EPS) & (d2 > _EPS) & ~np.eye(n, dtype=bool)
    a = np.where(ok, p[:, None] * p[None, :] / np.where(ok, d1 * d2, 1.0), 0.0)
    extra = p * (a.sum() - a.sum(axis=1) - a.sum(axis=0))
    return np.minimum(top2 + extra, 1.0)


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


# △（mark_type=anchor）の上限。頭数が少ないレースでは減らす（頭数の下限, 上限）。
MAYBE_LIMITS: tuple[tuple[int, int], ...] = ((13, 4), (8, 3), (5, 2), (0, 1))


def maybe_limit(n_runners: int) -> int:
    """頭数に応じた △ の最大頭数（13頭以上 4、8〜12頭 3、5〜7頭 2、4頭以下 1）。"""
    for min_n, limit in MAYBE_LIMITS:
        if n_runners >= min_n:
            return limit
    return 0


def assign_mece_marks(entries: list[dict]) -> None:
    """
    印を付与（in-place）。◎○▲★は各1頭まで、△は複数頭（上限は maybe_limit）。1頭1印。

    mark_type（表示記号）:
      honmei   … 1着優位（◎）※1頭のみ
      pair     … 連対確率が◎の次に高い（○）※1頭のみ
      show_val … 勝率低・複勝率高の単穴（▲）※1頭のみ
      star     … 10〜30倍帯で期待値が高い（★）※1頭のみ
      anchor   … 馬券内に来るかもしれない（△）※複数。3連系の相手候補。保存データ互換のため内部名は anchor のまま
      none     … 印なし
    旧ルールで保存された予測には ○・▲ が複数付いているものがあるが、保存済みの mark_type は書き換えない。
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

    def _free() -> list[dict]:
        return [e for e in entries if e.get("mark_type") == "none"]

    # ◎ 1着優位（1頭）
    by_win = sorted(entries, key=lambda x: float(x.get("win_prob") or 0), reverse=True)
    if by_win and float(by_win[0].get("win_prob") or 0) >= base * 1.1:
        by_win[0]["mark_type"] = "honmei"

    # ★ 中穴妙味（1頭・◎以外）
    star_candidates = [e for e in _free() if qualifies_star_mark(e, base)]
    if star_candidates:
        star = max(
            star_candidates,
            key=lambda x: (
                float(x.get("ev_win") or 0),
                float(x.get("ev_place") or 0),
            ),
        )
        star["mark_type"] = "star"

    # ○ 連対確率が最も高い馬（1頭・◎★以外、閾値を満たすときだけ）
    pair_candidates = [e for e in _free() if float(e.get("top2_prob") or 0) >= base * 1.25]
    if pair_candidates:
        max(pair_candidates, key=lambda x: float(x.get("top2_prob") or 0))["mark_type"] = "pair"

    # ▲ 複勝妙味の単穴（1頭）
    show_candidates = [
        e for e in _free()
        if _show_val_gap(e) >= base * 0.45 and float(e.get("win_prob") or 0) < base * 1.05
    ]
    if show_candidates:
        max(show_candidates, key=_show_val_gap)["mark_type"] = "show_val"

    # △ 馬券内に来るかもしれない馬（複数・上限あり・複勝確率の高い順）
    maybe_candidates = [
        e for e in _free()
        if float(e.get("top3_prob") or 0) >= base * 1.35 and _anchor_score(e) >= base * 0.5
    ]
    maybe_candidates.sort(key=lambda x: float(x.get("top3_prob") or 0), reverse=True)
    for e in maybe_candidates[: maybe_limit(n)]:
        e["mark_type"] = "anchor"


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
