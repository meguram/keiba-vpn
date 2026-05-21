"""印・期待値（勝/連/複EV）からレース単位の馬券提案を生成する。"""

from __future__ import annotations

from typing import Any

BET_TYPE_LABELS: dict[str, str] = {
    "tansho": "単勝",
    "fukusho": "複勝",
    "umaren": "馬連",
    "wide": "ワイド",
    "umatan": "馬単",
    "trio": "3連複",
    "tierce": "3連単",
}

MARK_SYMBOLS: dict[str, str] = {
    "honmei": "◎",
    "pair": "○",
    "anchor": "✓",
    "show_val": "▲",
    "star": "★",
    "none": "",
}

_EV_HIGH = 1.1
_EV_OK = 1.05
_EV_WATCH = 0.95


def _hn(entry: dict) -> int:
    return int(entry.get("horse_number") or 0)


def _mark(entry: dict) -> str:
    return str(entry.get("mark_type") or "none")


def _ev(entry: dict, key: str) -> float:
    v = entry.get(key)
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _confidence(ev: float, has_mark: bool) -> str:
    if ev >= _EV_HIGH or (has_mark and ev >= _EV_OK):
        return "high"
    if ev >= _EV_OK or (has_mark and ev >= _EV_WATCH):
        return "medium"
    if ev >= _EV_WATCH:
        return "low"
    return "low"


def _pick_by_mark(entries: list[dict], mark: str) -> dict | None:
    for e in entries:
        if _mark(e) == mark:
            return e
    return None


def _all_by_mark(entries: list[dict], mark: str) -> list[dict]:
    return [e for e in entries if _mark(e) == mark]


def _pair_selection(h1: int, h2: int, *, ordered: bool = False) -> str:
    if ordered:
        return f"{h1}→{h2}"
    return f"{h1}-{h2}"


def _make_pick(
    *,
    bet_type: str,
    horses: list[int],
    ev: float,
    ev_kind: str,
    marks: list[str],
    reason: str,
    priority: int,
    ordered: bool = False,
) -> dict[str, Any]:
    labels = [str(h) for h in horses]
    if bet_type in ("tansho", "fukusho"):
        selection = labels[0] if labels else ""
    elif bet_type == "umatan" and len(horses) >= 2:
        selection = _pair_selection(horses[0], horses[1], ordered=True)
    elif bet_type == "tierce" and len(horses) >= 3:
        selection = f"{horses[0]}→{horses[1]}→{horses[2]}"
    elif len(horses) >= 2:
        selection = _pair_selection(horses[0], horses[1], ordered=ordered)
    elif len(horses) == 3:
        selection = f"{horses[0]}-{horses[1]}-{horses[2]}"
    else:
        selection = "-".join(labels)

    has_mark = bool(marks and marks[0])
    return {
        "bet_type": bet_type,
        "bet_type_label": BET_TYPE_LABELS.get(bet_type, bet_type),
        "horses": horses,
        "horse_labels": labels,
        "selection": selection,
        "ev": round(ev, 3) if ev else None,
        "ev_kind": ev_kind,
        "marks": marks,
        "reason": reason,
        "confidence": _confidence(ev, has_mark),
        "priority": priority,
    }


def suggest_race_bets(entries: list[dict]) -> dict[str, Any]:
    """
    レース単位の馬券提案（印 × 勝/連/複EV）。

    Returns:
        strategy_label, headline, picks[], notes
    """
    if not entries:
        return {
            "strategy_label": "—",
            "headline": "出馬データ不足のため判定できません",
            "picks": [],
            "notes": [],
        }

    honmei = _pick_by_mark(entries, "honmei")
    pairs = _all_by_mark(entries, "pair")
    pair = pairs[0] if pairs else None
    anchors = _all_by_mark(entries, "anchor")
    anchor = anchors[0] if anchors else None
    show_vals = _all_by_mark(entries, "show_val")
    show_val = max(show_vals, key=lambda x: _ev(x, "ev_place")) if show_vals else None
    star = _pick_by_mark(entries, "star")

    picks: list[dict[str, Any]] = []
    notes: list[str] = []

    # ── 単勝 ──
    tansho_candidates: list[tuple[dict, float, str, list[str]]] = []
    for e, label, marks in (
        (honmei, "◎本命", ["◎"]),
        (star, "★中穴", ["★"]),
    ):
        if not e:
            continue
        ev = _ev(e, "ev_win")
        if ev >= _EV_WATCH:
            tansho_candidates.append((e, ev, label, marks))

    if not tansho_candidates:
        best_win = max(entries, key=lambda x: _ev(x, "ev_win"))
        if _ev(best_win, "ev_win") >= _EV_OK:
            tansho_candidates.append((best_win, _ev(best_win, "ev_win"), "勝EV最大", []))

    if tansho_candidates:
        e, ev, lbl, marks = max(tansho_candidates, key=lambda x: x[1])
        picks.append(
            _make_pick(
                bet_type="tansho",
                horses=[_hn(e)],
                ev=ev,
                ev_kind="勝EV",
                marks=marks,
                reason=f"{lbl}の勝期待値が高い（想定単勝{e.get('win_odds') or '—'}倍）",
                priority=1,
            )
        )

    # ── 複勝（▲複数・その他） ──
    fukusho_candidates: list[tuple[dict, float, str, list[str]]] = []
    for e in show_vals:
        ev = _ev(e, "ev_place")
        if ev >= _EV_WATCH:
            fukusho_candidates.append((e, ev, "▲複勝妙味", ["▲"]))
    for e, label, marks in (
        (honmei, "◎本命の複勝守り", ["◎"]),
        (star, "★の複勝", ["★"]),
    ):
        if not e:
            continue
        ev = _ev(e, "ev_place")
        if ev >= _EV_WATCH:
            fukusho_candidates.append((e, ev, label, marks))
    for e in anchors:
        ev = _ev(e, "ev_place")
        if ev >= _EV_WATCH:
            fukusho_candidates.append((e, ev, "✓紐の複勝安定", ["✓"]))

    seen_fukusho: set[int] = set()
    for e, ev, lbl, marks in sorted(fukusho_candidates, key=lambda x: -x[1])[:4]:
        hn = _hn(e)
        if hn in seen_fukusho:
            continue
        seen_fukusho.add(hn)
        already_tansho = any(p["bet_type"] == "tansho" and hn in p["horses"] for p in picks)
        if not already_tansho or ev >= _EV_OK:
            picks.append(
                _make_pick(
                    bet_type="fukusho",
                    horses=[hn],
                    ev=ev,
                    ev_kind="複EV",
                    marks=marks,
                    reason=f"{lbl}の複勝期待値が高い",
                    priority=2,
                )
            )

    # ── 2連系（◎＋○…） ──
    if honmei and pairs:
        h1 = _hn(honmei)
        pair_sorted = sorted(pairs, key=lambda x: _ev(x, "ev_top2"), reverse=True)[:3]
        for i, p2 in enumerate(pair_sorted):
            h2 = _hn(p2)
            pair_ev = (_ev(honmei, "ev_top2") + _ev(p2, "ev_top2")) / 2
            if pair_ev < _EV_WATCH:
                continue
            suffix = f"（{i + 1}点目）" if i > 0 else ""
            picks.append(
                _make_pick(
                    bet_type="umaren",
                    horses=[h1, h2],
                    ev=pair_ev,
                    ev_kind="連EV",
                    marks=["◎", "○"],
                    reason=f"◎○の連対期待が高い（馬連）{suffix}",
                    priority=3 + i,
                )
            )
            if i == 0:
                wide_ev = (_ev(honmei, "ev_place") + _ev(p2, "ev_place")) / 2
                if wide_ev >= _EV_WATCH:
                    picks.append(
                        _make_pick(
                            bet_type="wide",
                            horses=[h1, h2],
                            ev=wide_ev,
                            ev_kind="複EV",
                            marks=["◎", "○"],
                            reason="◎○が共に複勝圏の期待が高い（ワイド）",
                            priority=4,
                        )
                    )
                if _ev(honmei, "ev_win") >= _ev(p2, "ev_win"):
                    picks.append(
                        _make_pick(
                            bet_type="umatan",
                            horses=[h1, h2],
                            ev=_ev(honmei, "ev_top2"),
                            ev_kind="連EV",
                            marks=["◎", "○"],
                            reason="◎→○の順で連対期待（馬単）",
                            priority=5,
                            ordered=True,
                        )
                    )

    # ── 3連系（◎○✓） ──
    if honmei and pair and anchor:
        h1, h2, h3 = _hn(honmei), _hn(pair), _hn(anchor)
        trio_ev = (
            _ev(honmei, "ev_place") + _ev(pair, "ev_place") + _ev(anchor, "ev_place")
        ) / 3
        if trio_ev >= _EV_WATCH:
            picks.append(
                _make_pick(
                    bet_type="trio",
                    horses=[h1, h2, h3],
                    ev=trio_ev,
                    ev_kind="複EV",
                    marks=["◎", "○", "✓"],
                    reason="◎○✓の3列紐で3連複圏の期待",
                    priority=6,
                )
            )
        if _ev(honmei, "ev_win") >= _EV_OK:
            tierce_ev = _ev(honmei, "ev_win")
            picks.append(
                _make_pick(
                    bet_type="tierce",
                    horses=[h1, h2, h3],
                    ev=tierce_ev,
                    ev_kind="勝EV",
                    marks=["◎", "○", "✓"],
                    reason="◎軸の3連単（○→✓）",
                    priority=7,
                    ordered=True,
                )
            )

    # 重複・弱い候補の整理
    picks.sort(key=lambda p: (p["priority"], -(p["ev"] or 0)))
    seen: set[tuple[str, str]] = set()
    filtered: list[dict[str, Any]] = []
    for p in picks:
        key = (p["bet_type"], p["selection"])
        if key in seen:
            continue
        seen.add(key)
        if (p["ev"] or 0) < _EV_WATCH and p["confidence"] == "low":
            continue
        filtered.append(p)

    # 上位4件まで（レース単位の提案を簡潔に）
    filtered = filtered[:4]

    if not filtered:
        notes.append("期待値・印の基準を満たす馬券は見つかりませんでした（様子見推奨）。")
        return {
            "strategy_label": "様子見",
            "headline": "明確な買い目なし",
            "picks": [],
            "notes": notes,
        }

    bet_types = {p["bet_type"] for p in filtered}
    if "tierce" in bet_types or "trio" in bet_types:
        strategy = "3連系フォロー型"
    elif "umaren" in bet_types or "wide" in bet_types:
        strategy = "本命軸＋2連フォロー型"
    elif filtered[0]["bet_type"] == "fukusho" and "tansho" not in bet_types:
        strategy = "複勝中心型"
    elif "tansho" in bet_types and len(filtered) == 1:
        strategy = "単勝絞り型"
    else:
        strategy = "ミックス型"

    headline = " / ".join(
        f"{p['bet_type_label']}{p['selection']}" for p in filtered[:3]
    )

    high = [p for p in filtered if p["confidence"] == "high"]
    if high:
        notes.append(f"推奨度「高」: {', '.join(p['bet_type_label'] + p['selection'] for p in high)}")
    notes.append("印なし・期待値低めの馬は買い目から外しています。")

    return {
        "strategy_label": strategy,
        "headline": headline,
        "picks": filtered,
        "notes": notes,
    }
