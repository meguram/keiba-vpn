"""公開（匿名）と会員の出し分け。

匿名で見られるのは、出馬表の事実と『勝率・複勝率』まで。想定単勝・期待値・めぐAI印（旧『推奨』）・推奨馬券・
その他の付随情報は会員（または開発者）だけ。UI の隠しではなく、サーバが匿名の応答から項目を落とす（許可リスト方式）。
仕様・項目表: docs/operations/tiered_access.md
"""

from __future__ import annotations

import re
from typing import Any

# ── 予測 API（/api/race/{id}/predictions, /predict）──

PUBLIC_ENTRY_FIELDS = (
    "horse_number", "horse_name", "horse_id", "bracket_number", "sex_age",
    "jockey_name", "jockey_weight", "trainer_name", "weight", "weight_change",
    "win_prob", "top3_prob", "show_prob",          # 勝率・複勝率
    "pred_rank",                                   # 匿名では勝率の順位に振り直す
)

PUBLIC_TOP_FIELDS = (
    "race_id", "race_name", "venue", "round", "date", "start_time", "grade", "surface", "distance",
    "track_condition", "status", "has_prediction", "total_horses", "predictions",
    "model_type", "model_version", "model_description", "prediction_source", "prediction_note",
    "entries_status", "pipeline_status", "provisional_entries", "freshness", "error", "message", "_mock",
)

# 匿名の応答から落とす（= 会員限定）の説明。UI の鍵表示と docs の表の元
LOCKED_GROUPS = {
    "連対率": ["top2_prob", "place_prob"],
    "想定オッズ": ["win_odds", "place_odds_min", "place_odds_max", "place_odds_avg", "place_odds_eff", "top2_odds_est", "popularity"],
    "期待値": ["ev_win", "ev_place", "ev_top2", "ev_win_cal", "ev_place_cal", "ev_top2_cal", "expected_value", "estimated_prob"],
    "めぐAI印・買い判断": ["mark_type", "recommendation", "buy_tier", "megu_mark", "kesi_reason", "bet_suggestion"],
    "展開・ラップ・追走難度": ["position_forecast", "tracking_v2", "l3f_coherent", "pace_forecast", "gate_bias", "pace_unified", "pace_type", "scenario_forecast", "lap_forecast"],
    "めぐ指数・成長度・クラス基準": ["expected_megu", "expected_megu_meta", "growth", "growth_meta", "class_benchmark"],
    "総合スコア": ["pred_score", "normalized_score", "composite_score", "composite_rank", "prob_calibrated", "prob_calibration"],
}


def _rank_by_win_prob(rows: list[dict]) -> list[dict]:
    if rows and all(isinstance(r.get("win_prob"), (int, float)) for r in rows):
        rows = sorted(rows, key=lambda r: -float(r["win_prob"]))
        for i, r in enumerate(rows, 1):
            r["pred_rank"] = i
    else:
        for r in rows:
            r.pop("pred_rank", None)
    return rows


def _redact_predictions_public_mode(resp: dict[str, Any], role: str) -> dict[str, Any]:
    """公開モード（config/public_fields.json の mode=public）: 開発者以外には A 分類（自前計算）だけ。B/C/O は release の許可まで出さない。"""
    from src.utils import public_policy as P

    top_ok, entry_ok = P.top_allowed_fields(role), P.entry_allowed_fields(role)
    out = {k: v for k, v in resp.items() if k in top_ok}
    rows = [{k: e[k] for k in entry_ok if k in e} for e in (resp.get("predictions") or []) if isinstance(e, dict)]
    if role == "member" and P.odds_allowed() and not P.actual_odds_allowed():
        _fill_predicted_odds(rows, str(resp.get("race_id") or ""))
    if "predictions" in resp:
        out["predictions"] = _rank_by_win_prob(rows)
    out["access"] = {"tier": "public" if role == "anonymous" else "member", "public_mode": True, "member_only": sorted(LOCKED_GROUPS),
                     "withheld": _withheld()}
    return out


def _fill_predicted_odds(rows: list[dict], race_id: str) -> None:
    """実オッズを出さない間（actual_odds_public=false）の会員向け: 想定オッズ（最終オッズ予測のキャッシュ）で単勝・複勝・想定人気・期待値を付ける。
    キャッシュが無いレースは何も付けない（実オッズへは落とさない）。"""
    if not race_id or not rows:
        return
    try:
        from src.pipeline.inference.odds_source import predicted_single_odds
        from src.scraper.storage import HybridStorage
        from src.utils import paid_zone as PZ
        from src.utils.race_probabilities import place_effective_odds

        single = PZ.calibrate_single(predicted_single_odds(HybridStorage(), race_id))
        omap = PZ.predicted_odds_map(single)
    except Exception:  # noqa: BLE001  想定オッズが取れなくても確率などは返す
        return
    for r in rows:
        o = omap.get(int(r.get("horse_number") or 0))
        if not o:
            continue
        w, pmin, pmax = float(o["win_odds"]), float(o.get("place_odds_min") or 0), float(o.get("place_odds_max") or 0)
        r["win_odds"] = round(w, 1)
        r["popularity"] = o.get("popularity")
        r["odds_source"] = "predicted"
        if pmin and pmax:
            eff = place_effective_odds(pmin, pmax)
            r["place_odds_min"], r["place_odds_max"], r["place_odds_avg"] = round(pmin, 1), round(pmax, 1), round((pmin + pmax) / 2, 1)
            if eff:
                r["place_odds_eff"] = round(eff, 2)
                p3 = r.get("top3_prob")
                if isinstance(p3, (int, float)) and p3 > 0:
                    r["ev_place"] = r["expected_value"] = round(float(p3) * eff, 3)
        pw = r.get("win_prob")
        if isinstance(pw, (int, float)) and pw > 0:
            r["ev_win"] = round(float(pw) * w, 3)


def _withheld() -> list[str]:
    """画面に『公開準備中』と出す分類（B=事実・C=取得元固有・O=市場オッズ由来のうち、まだ出していないもの）。"""
    from src.utils import public_policy as P

    w = ["C"]
    if not P.b_allowed():
        w.append("B")
    if not P.odds_allowed():
        w.append("O")
    return w


def redact_predictions_member(resp: dict[str, Any]) -> dict[str, Any]:
    """会員向け。公開モードのときだけ、B/C/O 分類を落とす。private モードでは従来どおり何も落とさない。"""
    from src.utils import public_policy as P

    return _redact_predictions_public_mode(resp, "member") if P.is_public_mode() else resp


def redact_predictions(resp: dict[str, Any]) -> dict[str, Any]:
    """匿名向けの予測応答。許可リストの項目だけ残し、会員限定の項目は『無い』ものとして返す。"""
    from src.utils import public_policy as P

    if P.is_public_mode():
        return _redact_predictions_public_mode(resp, "anonymous")
    out = {k: v for k, v in resp.items() if k in PUBLIC_TOP_FIELDS}
    rows = []
    for e in resp.get("predictions") or []:
        if isinstance(e, dict):
            rows.append({k: e[k] for k in PUBLIC_ENTRY_FIELDS if k in e})
    if "predictions" in resp:
        out["predictions"] = _rank_by_win_prob(rows)
    out["access"] = {"tier": "public", "member_only": sorted(LOCKED_GROUPS)}
    return out


# ── レース詳細（/api/race/{id}）──

RACE_DETAIL_MEMBER_KEYS = (
    "race_odds", "race_pair_odds", "race_index", "race_paddock", "race_barometer", "race_trainer_comment", "smartrc_race",
)
SHUTUBA_MEMBER_ENTRY_KEYS = ("odds", "popularity")


def _redact_race_detail_public_mode(resp: dict[str, Any]) -> dict[str, Any]:
    """公開モードのレース詳細（開発者以外）。許可リスト方式: 名簿に無いキーは出さない。C/O は常に落とし、B は release の許可まで出さない。"""
    from src.utils import public_policy as P

    rd = P.section("race_detail")
    out: dict[str, Any] = {}
    if P.identifiers_allowed():
        out.update({k: resp[k] for k in rd.get("header_identifiers", []) if k in resp})
    else:
        out["race_id"] = resp.get("race_id")
    if P.b_allowed():
        card = resp.get("race_shutuba")
        if isinstance(card, dict) and isinstance(card.get("entries"), list):
            keep = set(rd.get("shutuba_entry_B", []))
            out["race_shutuba"] = {**{k: v for k, v in card.items() if k != "entries"},
                                   "entries": [{k: v for k, v in e.items() if k in keep} for e in card["entries"] if isinstance(e, dict)]}
        res = resp.get("race_result")
        if isinstance(res, dict) and isinstance(res.get("entries"), list):
            keep = set(rd.get("result_entry_B", []))
            out["race_result"] = {**{k: v for k, v in res.items() if k not in ("entries", "payoff")},
                                  "entries": [{k: v for k, v in e.items() if k in keep} for e in res["entries"] if isinstance(e, dict)]}
            if P.odds_allowed() and P.actual_odds_allowed() and "payoff" in res:
                out["race_result"]["payoff"] = res["payoff"]     # 確定払戻は確定オッズそのもの（単勝払戻÷100）なので、実オッズと同じ扱い
    out["access"] = {"tier": "public", "public_mode": True, "member_only": sorted(LOCKED_GROUPS), "withheld": _withheld()}
    return out


def redact_race_detail_member(resp: dict[str, Any]) -> dict[str, Any]:
    from src.utils import public_policy as P

    return _redact_race_detail_public_mode(resp) if P.is_public_mode() else resp


def redact_race_detail(resp: dict[str, Any]) -> dict[str, Any]:
    """匿名向けのレース詳細。出馬表・結果・馬の事実は残し、オッズ・指数・パドック等の付随情報は落とす。"""
    from src.utils import public_policy as P

    if P.is_public_mode():
        return _redact_race_detail_public_mode(resp)
    out = {k: v for k, v in resp.items() if k not in RACE_DETAIL_MEMBER_KEYS}
    card = out.get("race_shutuba")
    if isinstance(card, dict) and isinstance(card.get("entries"), list):
        card = dict(card)
        card["entries"] = [
            {k: v for k, v in e.items() if k not in SHUTUBA_MEMBER_ENTRY_KEYS} if isinstance(e, dict) else e
            for e in card["entries"]
        ]
        out["race_shutuba"] = card
    avail = out.get("data_availability")
    if isinstance(avail, dict):
        out["data_availability"] = {k: v for k, v in avail.items() if k not in RACE_DETAIL_MEMBER_KEYS}
    out["access"] = {"tier": "public", "member_only": sorted(LOCKED_GROUPS)}
    return out


# ── 予測成績（/api/prediction-results/*）──
# ゲスト・無料ログインには、◎などの予測と着順の突き合わせ（件数・率）と勝率までを出し、オッズ・人気・仮想購入（買い目・投資・払戻・回収率）は落とす。
# 会員（有料）・開発者はそのまま。許可リスト方式: 列挙した項目以外は出さない。

_PR_MARK_EXTRA = ("pop_sum", "pop_n", "odds_sum", "odds_n", "w_pop_sum", "w_pop_n", "w_odds_sum", "w_odds_n")
_PR_PICK_KEYS = ("horse_number", "horse_name", "finish", "win_prob", "mark")
_PR_HIGHLIGHT_PUBLIC_KINDS = ("big_miss",)            # ana_hit（単勝オッズ）・big_return（推奨馬券の金額）はオッズ由来
_PR_NEWS_PUBLIC_KINDS = ("day", "week", "big_miss")
_PR_BETS_ZERO = {"invested": 0, "returned": 0, "races_bought": 0, "races_hit": 0, "estimated_races": 0, "by_type": {}}
_PR_TOP_KEYS = ("scope", "date", "from", "to", "days", "statuses", "status", "label", "notes", "prediction_sources", "models", "built_at")
_PR_SUMMARY_KEYS = ("races", "races_settled", "races_no_prediction", "races_no_result", "top1", "logloss", "logloss_model")
_PR_RACE_KEYS = ("race_id", "venue", "round", "race_name", "grade", "included", "reason", "result_status", "n_runners", "result_top3",
                 "winner_logloss", "winner_logloss_model")


def _pr_counts(c: Any) -> Any:
    return {k: v for k, v in c.items() if k not in _PR_MARK_EXTRA} if isinstance(c, dict) else c


def _pr_pick(p: Any) -> Any:
    return {k: p[k] for k in _PR_PICK_KEYS if k in p} if isinstance(p, dict) else p


def _pr_summary(s: Any) -> Any:
    if not isinstance(s, dict):
        return s
    out = {k: s[k] for k in _PR_SUMMARY_KEYS if k in s}
    if isinstance(s.get("marks"), dict):
        out["marks"] = {m: _pr_counts(c) for m, c in s["marks"].items()}
    out["bets"] = dict(_PR_BETS_ZERO, by_type={})
    return out


def _pr_kpi(k: Any) -> Any:
    if not isinstance(k, dict):
        return k
    out = {key: k[key] for key in ("honmei", "top1", "winner_logloss", "winner_logloss_model") if key in k}
    if isinstance(k.get("marks"), dict):
        out["marks"] = {m: _pr_counts(c) for m, c in k["marks"].items()}
    out.update(roi=None, **_PR_BETS_ZERO)
    out["by_type"] = {}
    return out


def _pr_race(r: Any) -> Any:
    if not isinstance(r, dict):
        return r
    out = {k: r[k] for k in _PR_RACE_KEYS if k in r}
    if "top1" in r:
        out["top1"] = _pr_pick(r["top1"])
    if isinstance(r.get("marks"), dict):
        out["marks"] = {m: [_pr_pick(p) for p in ps] for m, ps in r["marks"].items()}
    return out


def redact_prediction_results(path: str, data: Any) -> Any:
    """予測成績の応答から、オッズ・人気・仮想購入（回収率）を落とす（ゲスト・無料ログイン向け）。入力は書き換えない。"""
    if not isinstance(data, dict):
        return data
    tail = path.rstrip("/").rsplit("/", 1)[-1]
    if tail == "days":
        return data
    if tail == "news":
        items = []
        for it in data.get("items") or []:
            if not isinstance(it, dict) or it.get("kind") not in _PR_NEWS_PUBLIC_KINDS:
                continue
            pub = it.get("subtitle_public", it.get("subtitle"))
            items.append({**{k: v for k, v in it.items() if k not in ("roi", "subtitle")}, "subtitle": pub, "subtitle_public": pub})
        return {**{k: v for k, v in data.items() if k != "items"}, "items": items}
    if tail == "mark-card":
        drop = ("avg_popularity", "avg_odds", "win_avg_popularity", "win_avg_odds")
        periods = [{**pd, "marks": [{k: v for k, v in m.items() if k not in drop} for m in pd.get("marks") or []]} if isinstance(pd, dict) else pd
                   for pd in data.get("periods") or []]
        return {**data, "periods": periods}
    if tail == "bets-summary":
        return dict(MEMBER_REQUIRED_BODY)
    out = {k: data[k] for k in _PR_TOP_KEYS if k in data}
    if "kpi" in data:
        out["kpi"] = _pr_kpi(data["kpi"])
    if "summary" in data:
        out["summary"] = _pr_summary(data["summary"])
    if "series" in data:
        out["series"] = [{k: pt[k] for k in ("date", "status", "honmei_win", "races") if k in pt} if isinstance(pt, dict) else pt for pt in data["series"] or []]
    if "highlights" in data:
        out["highlights"] = [h for h in data["highlights"] or [] if isinstance(h, dict) and h.get("kind") in _PR_HIGHLIGHT_PUBLIC_KINDS]
    if "races" in data:
        out["races"] = [_pr_race(r) for r in data["races"] or []]
    return out


# 実オッズ（市場オッズそのもの）に由来するキー。actual_odds_public=false の間、開発者以外の予測成績の応答から再帰的に落とす
ACTUAL_ODDS_KEYS = frozenset({
    "odds", "win_odds", "final_odds", "popularity", "odds_sum", "odds_n", "pop_sum", "pop_n", "w_odds_sum", "w_odds_n", "w_pop_sum", "w_pop_n",
    "avg_odds", "avg_popularity", "win_avg_odds", "win_avg_popularity", "odds_phase", "odds_source", "pair_odds", "single_odds",
    "ev_win",                                                    # 印の馬の勝ち期待値（較正前は確率×実オッズ。確率が見えるので逆算できる）
})
_ODDS_PHRASE = re.compile(r"（単勝[0-9.]+倍）")


def _drop_keys(obj: Any, keys: frozenset[str]) -> Any:
    if isinstance(obj, dict):
        return {k: _drop_keys(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [_drop_keys(v, keys) for v in obj]
    return obj


def redact_prediction_results_member(data: Any) -> Any:
    """有料会員向けの予測成績: 仮想購入の結果（金額・回収率）は残し、実オッズ・人気・実オッズ由来の文言と値を落とす。入力は書き換えない。"""
    from src.utils import public_policy as P

    if not P.is_public_mode() or P.actual_odds_allowed() or not isinstance(data, (dict, list)):
        return data
    out = _drop_keys(data, ACTUAL_ODDS_KEYS)
    if isinstance(out, dict):
        for it in out.get("items") or []:                       # news
            if isinstance(it, dict) and it.get("kind") == "ana_hit":
                for k in ("subtitle", "subtitle_public"):
                    if isinstance(it.get(k), str):
                        it[k] = _ODDS_PHRASE.sub("", it[k])
        for h in out.get("highlights") or []:                   # 本体の注目レース
            if isinstance(h, dict) and h.get("kind") == "ana_hit":
                h.pop("value", None)
                if isinstance(h.get("text"), str):
                    h["text"] = _ODDS_PHRASE.sub("", h["text"])
    return out


# 会員（開発者以外）向けの redactor。private モードでは何も落とさない（従来どおり）
MEMBER_REDACTORS = {redact_predictions: redact_predictions_member, redact_race_detail: redact_race_detail_member}

# ── 会員限定のパス（匿名は 401）──

_MEMBER_ONLY_PATTERNS = [re.compile(p) for p in (
    r"^/api/race/[^/]+/(tracking-difficulty|bundle|bloodline-aptitude|final-odds)",
    r"^/api/predictions$",
    r"^/api/growth-curve/(?!status)",
    r"^/api/race-quality/",
    r"^/api/v1/betting/",
    r"^/api/prediction-results/bets-summary",
    r"^/api/v1/races/[^/]+/(final-odds|megu-index|megu-index-predicted|tracking-difficulty|predictions/laps)",
    r"^/api/v1/horse/[^/]+/(megu-index-history|growth-curve)",
    r"^/api/v1/(pedigree-race-stats|data-analysis)/",
    r"^/api/v1/(bloodline-cluster/horse-aptitude|pedigree/race-note-3d-v2)",
)]


def member_only_path(path: str) -> bool:
    return any(p.search(path) for p in _MEMBER_ONLY_PATTERNS)


def owner_only_path(path: str) -> bool:
    """公開モードで、開発者（所有者・私的利用）だけが入れるパス（取得元固有・市場オッズ由来の情報）。"""
    from src.utils import public_policy as P

    return P.owner_only_path(path)


OWNER_REQUIRED_BODY = {
    "error": "この情報は公開準備中です。",
    "reason": "not_public_yet",
}

MEMBER_REQUIRED_BODY = {
    "error": "この情報は会員限定です。ログインしてください。",
    "reason": "member_required",
    "login_url": "/login",
}
