"""有料ゾーン（会員向け）の推奨馬券の出力変換。

画面に出すのは『レース単位の推奨度』（推奨ポートフォリオ全体の期待回収率を段階にしたもの）と、買い目（券種ごとの
組み合わせ＋区分 本線/押さえ）だけ。オッズ・期待値・確率・Kelly 配分の金額は出さない（逆算を防ぐ）。

方式（config/paid_zone.json の mode、環境変数 KEIBA_PAID_ZONE_MODE で上書き）:
  A … 確率のみ。オッズを一切使わない最小構成（組み合わせは確率の高い順、推奨度はレースの確信度）。
  B … 承認前の既定。実オッズを使わず、AI の確率と『想定オッズ』（単勝・複勝のモデル予測、馬連・ワイド・馬単・3連複は
      想定単勝から Harville で導いた公正オッズ）で期待値を作る。想定単勝が無いレースは A に落とす。
  C … 内部で全券種の実オッズを使う（netkeiba の承認後）。画面の出力は A・B と同じ形。
仕様と評価: docs/operations/paid_zone_tiers.md
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "paid_zone.json"

BET_LABELS = {"tansho": "単勝", "fukusho": "複勝", "umaren": "馬連", "wide": "ワイド", "umatan": "馬単", "sanrenpuku": "3連複"}
BET_ORDER = ["tansho", "fukusho", "umaren", "wide", "umatan", "sanrenpuku"]
MARK_SYMBOLS = {"honmei": "◎", "pair": "○", "show_val": "▲", "star": "★", "anchor": "△", "kesi": "消"}

# 画面・API の応答に出してはいけないキー（オッズ・期待値・確率・Kelly 配分の金額）。テストで応答を再帰検査する
FORBIDDEN_KEYS = frozenset({
    "odds", "win_odds", "place_odds_min", "place_odds_max", "place_odds_avg", "place_odds_eff", "top2_odds_est", "predicted_odds",
    "odds_gap", "odds_source", "odds_phase", "odds_as_of", "available_odds", "odds_note", "popularity",
    "ev", "ev_calibrated", "ev_raw", "ev_win", "ev_place", "ev_top2", "ev_label", "ev_calibration_validated", "expected_value",
    "expected_return", "expected_return_calibrated", "expected_roi", "expected_roi_calibrated", "roi_pct", "roi_pct_calibrated",
    "roi_range_calibrated", "roi_calibrated_capped", "kelly_fraction", "bet_amount", "total_bet", "bankroll", "prob", "probability_model",
    "prob_calibration", "estimated_prob", "calibrated",
})

_cache: dict[str, Any] = {}


def config() -> dict[str, Any]:
    try:
        mtime = CONFIG_PATH.stat().st_mtime
    except OSError:
        mtime = 0.0
    if _cache.get("mtime") != mtime:
        try:
            _cache["cfg"] = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _cache["cfg"] = {}
        _cache["mtime"] = mtime
    return _cache["cfg"]


def reload() -> None:
    _cache.clear()


def enabled() -> bool:
    if os.environ.get("KEIBA_PAID_ZONE", "").strip() == "0":
        return False
    return bool(config().get("enabled", True))


def mode() -> str:
    m = (os.environ.get("KEIBA_PAID_ZONE_MODE") or config().get("mode") or "B").strip().upper()
    return m if m in ("A", "B", "C") else "B"


def bet_types() -> list[str]:
    bts = [b for b in (config().get("bet_types") or BET_ORDER) if b in BET_LABELS]
    return bts or list(BET_ORDER)


def tiers() -> list[dict[str, Any]]:
    t = config().get("tiers")
    if t:
        return [dict(x) for x in t]
    return [{"level": 0, "label": "推奨しない", "short": "見送り"}, {"level": 1, "label": "元返し程度", "short": "元返し"},
            {"level": 2, "label": "推奨", "short": "推奨"}, {"level": 3, "label": "強く推奨", "short": "強推奨"},
            {"level": 4, "label": "最も推奨", "short": "最推奨"}]


def thresholds(m: str | None = None) -> list[float]:
    th = (config().get("thresholds") or {}).get(m or mode()) or [0.97, 1.0, 1.03, 1.07]
    return [float(x) for x in th]


def level_for_roi(roi: float | None, m: str | None = None) -> int:
    """推奨ポートフォリオの期待回収率 → 推奨度（0〜）。None・非数は 0（推奨しない）。"""
    if roi is None or roi != roi:
        return 0
    lv = 0
    for i, t in enumerate(thresholds(m), 1):
        if roi >= t:
            lv = i
    return lv


def tier_info(level: int) -> dict[str, Any]:
    for t in tiers():
        if t["level"] == level:
            return {"level": level, "label": t["label"], "short": t.get("short", t["label"])}
    return {"level": level, "label": "", "short": ""}


def legend() -> list[dict[str, Any]]:
    return [{"level": t["level"], "label": t["label"], "short": t.get("short", t["label"])} for t in tiers()]


def disclaimer() -> str:
    return str(config().get("disclaimer") or "")


def unit_yen() -> int:
    return int((config().get("units") or {}).get("unit_yen") or 100)


def default_max_total_yen() -> int:
    return int((config().get("units") or {}).get("default_max_total_yen") or 2000)


# ───────────────────────── 案A: 想定オッズだけから全券種の想定オッズを作る ─────────────────────────

def synth_pair_odds(win_odds_by_hn: dict[int, float]) -> dict[str, list[dict]]:
    """想定単勝オッズ → 馬連・ワイド・馬単・3連複の『想定オッズ』（公正オッズ＝控除後の払戻率 ÷ 確率）。実オッズは使わない。

    単勝オッズの逆数を正規化して Harville の着順分布を作り、各組み合わせの確率から公正オッズを出す。頭数が3未満なら空。
    """
    import numpy as np

    from src.pipeline.inference.exotic_probabilities import RankDistribution

    items = sorted((int(h), float(o)) for h, o in win_odds_by_hn.items() if o and float(o) > 0)
    if len(items) < 3:
        return {}
    hns = [h for h, _ in items]
    q = np.array([1.0 / o for _, o in items], dtype=float)
    q = q / q.sum()
    dist = RankDistribution(q)
    pb = config().get("payback") or {}
    pu, pw, pt, ps = (float(pb.get("umaren", 0.775)), float(pb.get("wide", 0.775)), float(pb.get("umatan", 0.75)), float(pb.get("sanrenpuku", 0.75)))

    cal = config().get("odds_calibration") or {}

    def odds(p: float, payback: float, kind: str = "") -> float:
        fair = max(payback / max(p, 1e-6), 1.0)
        c = cal.get(kind) or {}
        a, b = float(c.get("a", 0.0)), float(c.get("b", 1.0))
        return round(max(float(np.exp(a + b * np.log(fair))), 1.0), 1)

    out: dict[str, list[dict]] = {"umaren": [], "wide": [], "umatan": [], "sanrenpuku": []}
    n = len(hns)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = hns[i], hns[j]
            out["umaren"].append({"pair": [a, b], "odds": odds(dist.umaren(i, j), pu, "umaren")})
            o_w = odds(dist.wide(i, j), pw, "wide")
            out["wide"].append({"pair": [a, b], "odds_min": o_w, "odds_max": o_w})
    for i in range(n):
        for j in range(n):
            if i != j:
                out["umatan"].append({"pair": [hns[i], hns[j]], "odds": odds(dist.umatan(i, j), pt, "umatan")})
    for (i, j, k), p in dist.all_sanrenpuku().items():
        out["sanrenpuku"].append({"combo": sorted([hns[i], hns[j], hns[k]]), "odds": odds(p, ps, "sanrenpuku")})
    return out


def calibrate_single(single: dict | None) -> dict | None:
    """想定単勝・複勝（final_odds_prediction）を、実際の水準・ばらつきに合わせて較正する（config の single_calibration。無ければそのまま）。"""
    import numpy as np

    cal = config().get("single_calibration") or {}
    cw, cp = cal.get("win"), cal.get("place_min")
    if not single or not (cw or cp):
        return single
    out = []
    for e in single.get("entries") or []:
        e = dict(e)
        w, pmin, pmax = e.get("win_odds"), e.get("place_odds_min"), e.get("place_odds_max")
        if cw and w and float(w) > 0:
            e["win_odds"] = round(max(float(np.exp(float(cw["a"]) + float(cw["b"]) * np.log(float(w)))), 1.0), 2)
        if cp and pmin and float(pmin) > 0:
            new_min = max(float(np.exp(float(cp["a"]) + float(cp["b"]) * np.log(float(pmin)))), 1.0)
            ratio = (float(pmax) / float(pmin)) if (pmax and float(pmax) >= float(pmin)) else 1.5
            e["place_odds_min"], e["place_odds_max"] = round(new_min, 2), round(new_min * ratio, 2)
        out.append(e)
    return {**single, "entries": out}


_MARKET_KEYS = ("win_odds", "place_odds_min", "place_odds_max", "place_odds_avg", "place_odds_eff", "top2_odds_est", "popularity",
                "ev_win", "ev_place", "ev_top2", "ev_win_cal", "ev_place_cal", "ev_top2_cal", "expected_value", "estimated_prob",
                "mark_type", "recommendation", "buy_tier", "composite_score", "composite_rank", "odds_source", "prob_calibrated",
                "kesi_reason", "megu_mark")


def strip_market(entries: list[dict]) -> list[dict]:
    """予測の出走馬から、市場（実オッズ）由来の項目を落とす。確率・馬番・馬名などは残す。"""
    return [{k: v for k, v in e.items() if k not in _MARKET_KEYS} for e in entries]


def predicted_odds_map(single: dict | None) -> dict[int, dict]:
    """想定オッズの文書 → 馬番ごとの辞書（想定の人気順を popularity として付ける）。"""
    ents = [e for e in ((single or {}).get("entries") or []) if e.get("horse_number") and e.get("win_odds")]
    ranked = sorted(ents, key=lambda e: float(e["win_odds"]))
    pop = {int(e["horse_number"]): i + 1 for i, e in enumerate(ranked)}
    return {int(e["horse_number"]): {**e, "popularity": pop[int(e["horse_number"])]} for e in ents}


# ───────────────────────── 出力の変換 ─────────────────────────

def _combo_key(bt: str, pair: list[int]) -> tuple[int, ...]:
    return tuple(pair) if bt == "umatan" else tuple(sorted(pair))


def build_paid_payload(
    candidates: list[dict],
    predictions: list[dict],
    *,
    race_id: str,
    roi: float | None,
    m: str,
    max_total_bet: int | None,
) -> dict[str, Any]:
    """推奨ポートフォリオの候補（オッズ・期待値つき）→ 会員向けの出力。数値のキーは出さない。

    candidates: {bet_type, pair, horse_names, ev, ev_calibrated?...}。roi: ポートフォリオの期待回収率（C は較正後）。
    """
    level = level_for_roi(roi, m)
    by_hn = {int(p.get("horse_number") or 0): p for p in predictions}
    rank_of = {hn: i for i, hn in enumerate(sorted(by_hn, key=lambda h: -(float(by_hn[h].get("win_prob") or 0))))}
    main_min = float(((config().get("role") or {}).get("main_min_ev") or {}).get(m, 1.0))
    units_cfg = config().get("units") or {}
    u_main, u_cover = int(units_cfg.get("main", 3)), int(units_cfg.get("cover", 1))

    bets: list[dict[str, Any]] = []
    if level >= 1:
        for c in candidates:
            bt = c.get("bet_type")
            if bt not in BET_LABELS:
                continue
            ev = c.get("ev_calibrated") if (m == "C" and c.get("ev_calibrated") is not None) else c.get("ev")
            hns = [int(h) for h in c.get("pair") or []]
            role = c.get("role") or ("main" if (ev or 0) >= main_min else "cover")
            legs = []
            for h in hns:
                p = by_hn.get(h, {})
                mk = MARK_SYMBOLS.get(str(p.get("mark_type") or ""), "")
                legs.append({"horse_number": h, "horse_name": p.get("horse_name") or "", "mark": mk})
            sep = "→" if bt == "umatan" else "-"
            bets.append({
                "bet_type": bt, "bet_label": BET_LABELS[bt], "combo": hns, "combo_label": sep.join(str(h) for h in hns),
                "legs": legs, "role": role, "role_label": "本線" if role == "main" else "押さえ",
                "_units": u_main if role == "main" else u_cover,
                "_order": (BET_ORDER.index(bt), sum(rank_of.get(h, 99) for h in hns), tuple(hns)),
            })
    cap_units = None
    if max_total_bet is not None and max_total_bet > 0:
        cap_units = max(int(max_total_bet) // unit_yen(), 0)
    # 上限を超えるときは、押さえ → 本線の順に、確率の低い（順位の悪い）組み合わせから外す
    bets.sort(key=lambda b: b["_order"])
    if cap_units is not None:
        def total() -> int:
            return sum(b["_units"] for b in bets)

        for role in ("cover", "main"):
            for b in sorted([b for b in bets if b["role"] == role], key=lambda b: b["_order"], reverse=True):
                if total() <= cap_units:
                    break
                bets.remove(b)
    total_units = sum(b["_units"] for b in bets)
    out_bets = []
    for b in bets:
        d = {k: v for k, v in b.items() if not k.startswith("_")}
        d["units"] = b["_units"]
        d["yen"] = b["_units"] * unit_yen()
        out_bets.append(d)
    return {
        "view": "paid",
        "race_id": race_id,
        "race_tier": tier_info(level if out_bets else 0),
        "bets": out_bets,
        "total_units": total_units,
        "total_yen": total_units * unit_yen(),
        "unit_yen": unit_yen(),
        "max_total_bet": max_total_bet,
        "legend": legend(),
        "disclaimer": disclaimer(),
        "note": "オッズと期待値の数値は表示しません。推奨度は、推奨馬券全体（組み合わせ）の見通しを段階にしたものです。",
    }


def find_forbidden_keys(obj: Any, path: str = "") -> list[str]:
    """応答の中に、出してはいけないキーが無いか再帰的に調べる（テスト用）。"""
    found: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN_KEYS:
                found.append(f"{path}/{k}")
            found += find_forbidden_keys(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found += find_forbidden_keys(v, f"{path}[{i}]")
    return found


# ───────────────────────── 案A: 確率のみ（オッズを使わない） ─────────────────────────

# 券種ごとに選ぶ組み合わせ数（確率の高い順。上位 5 頭の中から）
_A_PICKS = {"tansho": 1, "fukusho": 1, "umaren": 2, "wide": 2, "umatan": 1, "sanrenpuku": 2}


def confidence_index(predictions: list[dict]) -> float:
    """レースの確信度＝勝率最大馬の勝率 × 頭数（均等確率の何倍か）。"""
    ws = [float(p.get("win_prob") or 0) for p in predictions]
    return max(ws) * len(ws) if ws else 0.0


def prob_only_candidates(predictions: list[dict]) -> list[dict]:
    """AI の確率だけで、券種ごとの組み合わせを選ぶ（オッズ・期待値は使わない）。1 位を本線、それ以外を押さえにする。"""
    import itertools

    import numpy as np

    from src.pipeline.inference.exotic_probabilities import RankDistribution

    ps = [p for p in predictions if int(p.get("horse_number") or 0) > 0 and float(p.get("win_prob") or 0) > 0]
    if len(ps) < 3:
        return []
    ps.sort(key=lambda p: -float(p["win_prob"]))
    hn = [int(p["horse_number"]) for p in ps]
    dist = RankDistribution(np.array([float(p["win_prob"]) for p in ps], dtype=float))
    top = list(range(min(5, len(ps))))
    out: list[dict] = []

    def add(bt: str, scored: list[tuple[float, tuple[int, ...]]]) -> None:
        scored.sort(key=lambda x: -x[0])
        for rank, (_, idx) in enumerate(scored[: _A_PICKS[bt]]):
            out.append({"bet_type": bt, "pair": [hn[i] for i in idx], "role": "main" if rank == 0 else "cover"})

    bts = bet_types()
    if "tansho" in bts:
        add("tansho", [(float(ps[i]["win_prob"]), (i,)) for i in top])
    if "fukusho" in bts:
        add("fukusho", [(float(ps[i].get("top3_prob") or 0), (i,)) for i in top])
    if "umaren" in bts:
        add("umaren", [(dist.umaren(i, j), tuple(sorted((i, j)))) for i, j in itertools.combinations(top, 2)])
    if "wide" in bts:
        add("wide", [(dist.wide(i, j), tuple(sorted((i, j)))) for i, j in itertools.combinations(top, 2)])
    if "umatan" in bts:
        add("umatan", [(dist.umatan(i, j), (i, j)) for i, j in itertools.permutations(top, 2)])
    if "sanrenpuku" in bts and len(ps) >= 3:
        add("sanrenpuku", [(dist.sanrenpuku(i, j, k), tuple(sorted((i, j, k)))) for i, j, k in itertools.combinations(top, 3)])
    return out
