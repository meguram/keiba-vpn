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


# 払戻率（JRA: 単勝・複勝 80%、馬連 77.5%）と、確定オッズから当てはめた補正の定数。
# 根拠と検証は docs/html/modeling/chapters/ev_calibration_audit.html、当てはめは audit_ev_calibration.py calib。
RETURN_RATE_UMAREN = 0.775
TOP2_MARKET_LAMBDA = 0.9      # 単勝の市場確率 q^λ の Harville が、馬連の確定オッズから作る連対確率に最もよく合う指数（λ=1 は人気馬に寄りすぎ）
PLACE_RANGE_WEIGHT = 0.3      # 複勝の実効オッズ = 下限 + w×(上限−下限)。確定オッズの Σ(1/odds)=3/0.8 を満たす w（的中時の払戻の位置の平均も 0.31）


def place_effective_odds(place_min: float | None, place_max: float | None) -> float | None:
    """複勝の期待払戻（倍）。中央値 (min+max)/2 は実際の払戻より約 9% 高い（Σ(1/odds) が 3/0.8 に届かない）ので、下限寄りに取る。"""
    if not place_min or place_min <= 0:
        return None
    hi = place_max if place_max and place_max >= place_min else place_min
    return place_min + PLACE_RANGE_WEIGHT * (hi - place_min)


def market_top2_fair_odds(win_odds: dict[int, float]) -> dict[int, float]:
    """連対そのものの馬券は無いので、単勝オッズから作る『馬連プールと整合する連対の公正オッズ』（控除率 22.5% 込み）を返す。

    q = (1/単勝オッズ) を正規化、連対確率 P2 = Harville(q^λ)、公正オッズ = 0.775 / P2。モデルの連対確率 p2 との EV は p2×この値で、
    p2 が市場の P2 に等しいとき 0.775 になる（以前の『単勝オッズ×1.65』は平均で約 4.4 倍高く、ほぼ全馬が EV≥1 になっていた）。
    頭数分の単勝オッズが揃わないときは空の辞書。
    """
    items = [(int(h), float(o)) for h, o in win_odds.items() if o and float(o) > 0]
    if len(items) < 3 or len(items) != len(win_odds):
        return {}
    hs = [h for h, _ in items]
    q = np.array([1.0 / o for _, o in items], dtype=float)
    q = q / q.sum()
    w = q ** TOP2_MARKET_LAMBDA
    p2 = np.clip(harville_top2_prob(w / w.sum()), 1e-4, 1.0)
    return {h: round(float(max(RETURN_RATE_UMAREN / p, 1.0)), 2) for h, p in zip(hs, p2)}


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

# レースの状況の判定と、状況別の印の基準（situational=True のときだけ使う）。値は設計年（2022〜23 年）で決める。
SITUATION_CFG: dict[str, float] = {
    "dom_mult": 2.0,      # 突出型: 勝率最大が均等確率のこの倍以上
    "dom_ratio": 1.6,     # 突出型: 勝率1位/2位がこの倍以上
    "close_ratio": 1.15,  # 拮抗型: 勝率1位/2位がこの倍未満
    "value_ev": 1.15,     # 荒れ型: 期待値が最大の馬のこの値以上の期待値
    "value_top3": 0.25,   # 荒れ型: 期待値で選ぶ馬の複勝確率の下限
    "value_min_win_mult": 0.8,  # 荒れ型の◎にする馬の勝率の下限（均等確率の倍）
    "maybe_by_composite": 0.0,  # 1 なら △ を複勝確率ではなく合成点（複勝確率^0.55 × 複勝EV^0.45）の高い順に選ぶ
}
SITUATIONAL_DEFAULT = False


def maybe_limit(n_runners: int) -> int:
    """頭数に応じた △ の最大頭数（13頭以上 4、8〜12頭 3、5〜7頭 2、4頭以下 1）。"""
    for min_n, limit in MAYBE_LIMITS:
        if n_runners >= min_n:
            return limit
    return 0


def _f(e: dict, key: str) -> float:
    return float(e.get(key) or 0)


def race_situation(entries: list[dict], cfg: dict[str, float] | None = None) -> str:
    """レースの状況: dom（突出）/ value（期待値の高い別の馬がいる＝荒れ型）/ close（拮抗）/ normal。
    オッズが無く期待値が出ないときは value にならない（dom / close / normal のみ）。"""
    c = {**SITUATION_CFG, **(cfg or {})}
    n = len(entries)
    if n < 2:
        return "normal"
    base = 1.0 / n
    by_win = sorted(entries, key=lambda x: _f(x, "win_prob"), reverse=True)
    p1, p2 = _f(by_win[0], "win_prob"), _f(by_win[1], "win_prob")
    if p1 >= base * c["dom_mult"] and p2 > 0 and p1 / p2 >= c["dom_ratio"]:
        return "dom"
    if p2 <= 0 and p1 > 0:
        return "dom"
    ev_cands = [e for e in entries if e.get("ev_win") and _f(e, "top3_prob") >= c["value_top3"]]
    if ev_cands:
        best = max(ev_cands, key=lambda x: _f(x, "ev_win"))
        if best is not by_win[0] and _f(best, "ev_win") >= c["value_ev"] and _f(best, "win_prob") >= base * c["value_min_win_mult"]:
            return "value"
    if p2 > 0 and p1 / p2 < c["close_ratio"]:
        return "close"
    return "normal"


def assign_mece_marks(entries: list[dict], situational: bool | None = None, cfg: dict[str, float] | None = None) -> None:
    """
    印を付与（in-place）。◎○▲★は各1頭まで、△は複数頭（上限は maybe_limit）。1頭1印。

    mark_type（表示記号）:
      honmei   … 1着優位（◎）※1頭のみ
      pair     … 連対確率が◎の次に高い（○）※1頭のみ
      show_val … 勝率低・複勝率高の単穴（▲）※1頭のみ
      star     … 10〜30倍帯で期待値が高い（★）※1頭のみ
      anchor   … 馬券内に来るかもしれない（△）※複数。3連系の相手候補。保存データ互換のため内部名は anchor のまま
      none     … 印なし
    situational=True のとき ◎○ の選び方をレースの状況（race_situation）で切り替える:
      value（期待値の高い別の馬がいる）… ◎=その馬、○=勝率最上位 / close（拮抗）… ◎=上位2頭のうち期待値の高い方、○=もう一方
      dom（突出）・normal … 通常（◎=勝率最上位、○=連対確率の最も高い馬）。
    旧ルールで保存された予測には ○・▲ が複数付いているものがあるが、保存済みの mark_type は書き換えない。
    """
    n = len(entries)
    if n == 0:
        return
    if situational is None:
        situational = SITUATIONAL_DEFAULT
    c = {**SITUATION_CFG, **(cfg or {})}
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

    by_win = sorted(entries, key=lambda x: float(x.get("win_prob") or 0), reverse=True)
    situation = race_situation(entries, cfg) if situational else "normal"
    pair_done = False

    # ◎ 1着優位（1頭）
    if situation == "value":
        ev_cands = [e for e in entries if e.get("ev_win") and _f(e, "top3_prob") >= c["value_top3"]]
        honmei = max(ev_cands, key=lambda x: _f(x, "ev_win"))
        honmei["mark_type"] = "honmei"
        top = by_win[0]
        if top is not honmei and _f(top, "top2_prob") >= base * 1.25:
            top["mark_type"] = "pair"
            pair_done = True
    elif situation == "close":
        a, b = by_win[0], by_win[1]
        pick, other = (a, b)
        if a.get("ev_win") and b.get("ev_win") and _f(b, "ev_win") > _f(a, "ev_win"):
            pick, other = b, a
        if _f(pick, "win_prob") >= base * 1.1:
            pick["mark_type"] = "honmei"
            if _f(other, "top2_prob") >= base * 1.25:
                other["mark_type"] = "pair"
                pair_done = True
    elif by_win and float(by_win[0].get("win_prob") or 0) >= base * 1.1:
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
    if not pair_done:
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
    if c.get("maybe_by_composite"):
        def _comp(x: dict) -> float:
            p3, evp = _f(x, "top3_prob"), _f(x, "ev_place")
            return (p3 ** 0.55) * (evp ** 0.45) if evp > 0 else p3 * 0.5
        maybe_candidates.sort(key=_comp, reverse=True)
    else:
        maybe_candidates.sort(key=lambda x: float(x.get("top3_prob") or 0), reverse=True)
    for e in maybe_candidates[: maybe_limit(n)]:
        e["mark_type"] = "anchor"


def _cal_horse_ev(entry: dict, key: str, kind: str) -> float | None:
    """馬ごとの EV を較正した値。無効なら None（従来の EV を使う）。EV が無い（0 以下）ものは較正しない。"""
    from src.utils.ev_calibration import calibrated_ev, is_enabled

    v = entry.get(key)
    if not is_enabled() or v is None or float(v) <= 0:
        return None
    return round(float(calibrated_ev(float(v), kind)), 3)


def attach_calibrated_ev(entry: dict) -> None:
    """ev_win / ev_top2 / ev_place に対する較正後の値（ev_*_cal）を付ける（無効のときは None）。画面は ev_*_cal があればそれを表示する。"""
    for key, kind in (("ev_win", "win"), ("ev_top2", "top2"), ("ev_place", "place")):
        entry[f"{key}_cal"] = _cal_horse_ev(entry, key, kind)


def buy_recommendation_tier(entry: dict) -> str:
    """買い判断推奨度（期待値の最大を基準）。EV の較正（config/ev_calibration.json）が有効なら、較正後の EV と較正後の閾値で判定する。"""
    from src.utils.ev_calibration import is_enabled, tier_thresholds

    th = tier_thresholds()
    if is_enabled():
        evs = [_cal_horse_ev(entry, k, kind) for k, kind in (("ev_win", "win"), ("ev_top2", "top2"), ("ev_place", "place"))]
        best = max([e for e in evs if e is not None], default=0.0)
    else:
        best = max(float(entry.get("ev_win") or 0), float(entry.get("ev_top2") or 0), float(entry.get("ev_place") or 0))
    if best >= th["strong"]:
        return "強推奨"
    if best >= th["recommend"]:
        return "推奨"
    if best >= th["watch"]:
        return "様子見"
    return "—"
