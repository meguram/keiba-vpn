"""公開（匿名）と会員の出し分け。

匿名で見られるのは、出馬表の事実と『勝率・複勝率』まで。想定単勝・期待値・めぐAI印（旧『推奨』）・推奨馬券・
その他の付随情報は会員（または開発者）だけ。UI の隠しではなく、サーバが匿名の応答から項目を落とす（許可リスト方式）。
仕様・項目表: docs/operations/tiered_access.md
"""

from __future__ import annotations

import re
from typing import Any

# ── 予測 API（/api/race/{id}/predictions, /predict）──

# 匿名・無料会員に出すのは、出馬表の事実と印（◎○▲△★消）だけ。勝率・複勝率・順位などの予測の数値は有料会員から（2026-11-25 ユーザーの判断）
PUBLIC_ENTRY_FIELDS = (
    "horse_number", "horse_name", "horse_id", "bracket_number", "sex_age",
    "jockey_name", "jockey_weight", "trainer_name", "weight", "weight_change",
    "mark_type", "mark",                           # 印（語と記号）。全員に同じめぐAI印（mark_type）
)
MARK_SYMBOLS = {"honmei": "◎", "pair": "○", "show_val": "▲", "anchor": "△", "star": "★", "kesi": "消"}

PUBLIC_TOP_FIELDS = (
    "race_id", "race_name", "venue", "round", "date", "start_time", "grade", "surface", "distance",
    "track_condition", "status", "has_prediction", "total_horses", "predictions",
    "model_type", "model_version", "model_description", "prediction_source", "prediction_note",
    "entries_status", "pipeline_status", "provisional_entries", "freshness", "error", "message", "_mock",
)

# 匿名の応答から落とす（= 会員限定）の説明。UI の鍵表示と docs の表の元
LOCKED_GROUPS = {
    "勝率・複勝率・順位": ["win_prob", "top3_prob", "show_prob", "pred_rank", "win_prob_model", "top3_prob_model"],
    "連対率": ["top2_prob", "place_prob"],
    "想定オッズ": ["win_odds", "place_odds_min", "place_odds_max", "place_odds_avg", "place_odds_eff", "top2_odds_est", "popularity"],
    "期待値": ["ev_win", "ev_place", "ev_top2", "ev_win_cal", "ev_place_cal", "ev_top2_cal", "expected_value", "estimated_prob"],
    "めぐAI印の説明・買い判断": ["recommendation", "buy_tier", "megu_mark", "kesi_reason", "bet_suggestion"],
    "展開・ラップ・追走難度": ["position_forecast", "tracking_v2", "l3f_coherent", "pace_forecast", "gate_bias", "pace_unified", "pace_type", "scenario_forecast", "lap_forecast"],
    "めぐ指数・成長度・クラス基準": ["expected_megu", "expected_megu_meta", "growth", "growth_meta", "class_benchmark"],
    "総合スコア": ["pred_score", "normalized_score", "composite_score", "composite_rank", "prob_calibrated", "prob_calibration"],
}


def locked_group_names() -> list[str]:
    """予測の応答の access.member_only に出す会員限定の機能の名前。予測を軽くした版（src/api/pred_slim.py）では、予測の応答に無い機能を出さない。
    レース詳細（出走馬カードの成長曲線などを含む）は今のまま sorted(LOCKED_GROUPS)。"""
    from src.api import pred_slim

    drop = set(pred_slim.DROPPED_GROUPS) if pred_slim.enabled() else set()
    return sorted(k for k in LOCKED_GROUPS if k not in drop)


def _public_mark(e: dict) -> dict[str, str]:
    """匿名・無料会員に出す印。印は 1 種類（全員に同じめぐAI印 mark_type。ユーザーの判断 2026-11-25）。
    無料の新しい印（public_mark_type）は影の記録だけで、画面・応答には出さない。"""
    mt = e.get("mark_type")
    return {"mark_type": mt, "mark": MARK_SYMBOLS.get(mt, "")} if isinstance(mt, str) and mt else {}


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
    if role == "anonymous":
        for row, e in zip(rows, [e for e in (resp.get("predictions") or []) if isinstance(e, dict)]):
            row.update(_public_mark(e))
    if "predictions" in resp:
        out["predictions"] = _rank_by_win_prob(rows) if role != "anonymous" else sorted(rows, key=lambda r: int(r.get("horse_number") or 99))
    out["access"] = {"tier": "public" if role == "anonymous" else "member", "public_mode": True, "member_only": locked_group_names(),
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

        # 要求ごとの新しいインスタンス（想定オッズを毎回取り直す鮮度を保つ）。Cloud Run では GCS のクライアントと健全性は
        # KEIBA_GCS_SHARED_STATE=1 で共有されるので、作り直しの費用は小さい。呼び出し元は to_thread（src/api/app.py の _tiered_json_async）
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


# 会員（有料。開発者は除く）にも見せない値（ユーザーの判断 2026-11-25・rules-council R0-10）。会員に見せる数値は AI の確率とオッズまで。
# 期待値（p×オッズ）・その段の買い判断・Kelly の目安・期待値の文を含む推奨の文。
MEMBER_HIDDEN_ENTRY_KEYS = frozenset({
    "ev_win", "ev_place", "ev_top2", "ev_win_cal", "ev_place_cal", "ev_top2_cal", "expected_value", "buy_tier", "kelly_fraction", "kelly",
    "recommendation",                                   # 印の説明文（「★ 中穴妙味」など、禁止語を含む）
    "composite_score", "composite_rank",                # p3^0.55 × EV^0.45。確率と並べると期待値を逆算できる
})
MEMBER_HIDDEN_TOP_KEYS = frozenset({"bet_suggestion", "ev_calibrated"})


def _drop_member_hidden(resp: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in resp.items() if k not in MEMBER_HIDDEN_TOP_KEYS}
    if isinstance(resp.get("predictions"), list):
        out["predictions"] = [{k: v for k, v in e.items() if k not in MEMBER_HIDDEN_ENTRY_KEYS} if isinstance(e, dict) else e
                              for e in resp["predictions"]]
    return out


def redact_predictions_member(resp: dict[str, Any]) -> dict[str, Any]:
    """会員向け。公開モードでは B/C/O 分類を落とす。どちらのモードでも、期待値・買い判断・Kelly の目安は落とす（開発者は素通り）。"""
    from src.utils import public_policy as P

    return _drop_member_hidden(_redact_predictions_public_mode(resp, "member") if P.is_public_mode() else resp)


def redact_predictions(resp: dict[str, Any]) -> dict[str, Any]:
    """匿名向けの予測応答。許可リストの項目だけ残し、会員限定の項目は『無い』ものとして返す。"""
    from src.utils import public_policy as P

    if P.is_public_mode():
        return _redact_predictions_public_mode(resp, "anonymous")
    out = {k: v for k, v in resp.items() if k in PUBLIC_TOP_FIELDS}
    rows = []
    for e in resp.get("predictions") or []:
        if isinstance(e, dict):
            rows.append({**{k: e[k] for k in PUBLIC_ENTRY_FIELDS if k in e and k not in ("mark_type", "mark")}, **_public_mark(e)})
    if "predictions" in resp:
        out["predictions"] = sorted(rows, key=lambda r: int(r.get("horse_number") or 99))    # 予測の順位が分からないよう馬番順
    out["access"] = {"tier": "public", "member_only": locked_group_names()}
    return out


# ── レース詳細（/api/race/{id}）──

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


# 匿名・無料会員に出すレース詳細の範囲（2026-11-25 ユーザーの確定）。許可リスト方式: 名簿に無いキーは出さない。
# 出すもの: レースの見出し・出馬表の事実・結果の事実（着順・タイム・上がり・通過・ラップ）。確定の結果だけ、確定の単勝オッズ・人気・払戻も出す。
# 出さないもの: 確定前のオッズ・人気（出馬表・速報 race_result_on_time）、馬柱・近走（horses の race_history）、指数・追い切り・厩舎コメントなど。
RACE_HEADER_KEYS = (
    "race_id", "race_name", "venue", "round", "date", "start_time", "surface", "distance", "grade", "track_condition", "direction", "weather",
    "quality_health_warning",
)
_ENTRY_FACT_KEYS = (
    "horse_number", "number", "bracket_number", "horse_name", "horse_id", "sex_age", "sex", "age", "jockey_name", "jockey", "jockey_id",
    "jockey_weight", "impost", "trainer_name", "trainer", "trainer_id", "weight", "weight_change", "sire", "dam_sire",
)
SHUTUBA_PUBLIC_ENTRY_KEYS = _ENTRY_FACT_KEYS
RESULT_PUBLIC_ENTRY_KEYS = _ENTRY_FACT_KEYS + (
    "finish_position", "finish_position_raw", "position", "finish_status", "finish_time", "time", "time_sec", "margin", "last_3f", "last3f",
    "passing_order",
)
RESULT_CONFIRMED_ENTRY_KEYS = ("odds", "popularity")          # 確定の結果だけ出す（JRA が公表する確定の単勝オッズ・人気）
_CARD_DROP_KEYS = frozenset({"entries", "payoff", "payoffs", "_meta"})
_LAP_KEYS = ("race_id", "lap_times", "pace", "corner_passing")
_LAP_ENTRY_KEYS = ("horse_number", "horse_id", "horse_name", "passing_order", "last_3f")


def race_result_confirmed(resp: dict[str, Any]) -> bool:
    """レース詳細の応答の結果が確定か（判定はここ 1 か所）。表示の結果（prepare_race_result_display）の元が確定の結果（race_result）で、
    着順の entries があるときだけ True。速報（race_result_on_time）だけのとき・結果が無いときは False（確定前扱い）。"""
    rr = resp.get("race_result") if isinstance(resp, dict) else None
    return (isinstance(rr, dict) and rr.get("_display_source") == "race_result"
            and isinstance(rr.get("entries"), list) and len(rr["entries"]) > 0)


def _pick_rows(rows: Any, keys: tuple[str, ...]) -> list:
    return [{k: e[k] for k in keys if k in e} for e in rows if isinstance(e, dict)] if isinstance(rows, list) else []


def redact_race_detail(resp: dict[str, Any]) -> dict[str, Any]:
    """匿名・無料会員向けのレース詳細。許可リストの項目だけを残す。入力は書き換えない。"""
    from src.utils import public_policy as P

    if P.is_public_mode():
        return _redact_race_detail_public_mode(resp)
    confirmed = race_result_confirmed(resp)
    out: dict[str, Any] = {k: resp[k] for k in RACE_HEADER_KEYS if k in resp}
    card = resp.get("race_shutuba")
    if isinstance(card, dict):
        out["race_shutuba"] = {**{k: v for k, v in card.items() if k not in _CARD_DROP_KEYS},
                               "entries": _pick_rows(card.get("entries"), SHUTUBA_PUBLIC_ENTRY_KEYS)}
    res = resp.get("race_result")
    if isinstance(res, dict):
        keys = RESULT_PUBLIC_ENTRY_KEYS + (RESULT_CONFIRMED_ENTRY_KEYS if confirmed else ())
        out["race_result"] = {**{k: v for k, v in res.items() if k not in _CARD_DROP_KEYS},
                              "entries": _pick_rows(res.get("entries"), keys)}
        if confirmed and "payoff" in res:
            out["race_result"]["payoff"] = res["payoff"]      # 確定の払戻（確定後の結果の表は公式と同じ事実）
    lap = resp.get("race_result_lap")
    if isinstance(lap, dict):
        out["race_result_lap"] = {**{k: lap[k] for k in _LAP_KEYS if k in lap}, "entries_lap": _pick_rows(lap.get("entries_lap"), _LAP_ENTRY_KEYS)}
    avail = resp.get("data_availability")
    if isinstance(avail, dict):
        out["data_availability"] = {k: v for k, v in avail.items() if k in ("race_shutuba", "race_result", "race_result_lap")}
    out["access"] = {"tier": "public", "member_only": sorted(LOCKED_GROUPS), "result_confirmed": confirmed,
                     "locked": ["race_odds", "horses"]}
    return out


# ── 出走馬カード（/api/race/{id}/entry/{horse_id}/card）──
# 匿名・無料会員には、近走の公式の事実と同条件の成績などを出す。追い切り・厩舎コメント・めぐ指数（成長曲線）・近走の人気・オッズは落とす。

CARD_RECENT_KEYS = (
    "date", "race_id", "venue", "race_name", "surface", "distance", "going", "field", "post", "finish", "time",
    "jockey", "jockey_id", "jockey_weight", "passing", "last3f", "weight", "weight_change", "margin",
)
CARD_LOCKED = ("training", "trainer_comment", "growth")


def redact_entry_card(resp: dict[str, Any]) -> dict[str, Any]:
    """匿名・無料会員向けの出走馬カード。入力は書き換えない。"""
    out = dict(resp)
    h = resp.get("horse")
    if isinstance(h, dict):
        h = {k: v for k, v in h.items() if k not in CARD_LOCKED}
        if "recent" in h:
            h["recent"] = _pick_rows(h.get("recent"), CARD_RECENT_KEYS)
        out["horse"] = h
    t = resp.get("trainer")
    if isinstance(t, dict):
        out["trainer"] = {k: v for k, v in t.items() if k != "comment"}
    out["access"] = {"tier": "public", "locked": list(CARD_LOCKED)}
    return out


# ── 漏れの検査（tests と src/scripts/ops/check_anon_scope.py が使う）──
# 匿名・無料会員の応答に、どこにも出てはいけないキー（netkeiba 独自の情報・確定前のオッズ・予測の数値）
SCOPE_FORBIDDEN_KEYS = frozenset({
    "training", "trainer_comment", "race_oikiri", "race_trainer_comment", "race_shutuba_past", "race_index", "race_paddock",
    "race_barometer", "smartrc_race", "race_odds", "race_pair_odds", "race_history", "time_index", "start_index", "chase_index",
    "closing_index", "remarks", "stable_comment", "race_result_on_time", "growth",
    "win_odds", "place_odds_min", "place_odds_max", "final_odds", "ev_win", "ev_place", "ev_top2", "expected_value",
    "win_prob", "top3_prob", "show_prob", "top2_prob", "place_prob", "pred_rank", "pred_score", "composite_score", "composite_rank",
})
# レースごとの勝ち馬の logloss（exp(-値) で勝ち馬の予測の勝率になる）。期間全体の集計（kpi・summary）は逆算できないので出してよい
_PER_RACE_LOGLOSS = frozenset({"winner_logloss", "winner_logloss_model"})
# 文字列の中に出てはいけない言い回し（予測の勝率・確定前の単勝オッズ）
SCOPE_FORBIDDEN_TEXT = re.compile(r"勝率\s*[0-9.]+\s*%|単勝\s*[0-9.]+\s*倍")
# 確定の結果に限って出してよい（race_result の中だけ）
SCOPE_CONFIRMED_ONLY_KEYS = frozenset({"odds", "popularity", "payoff", "payoffs"})


def find_scope_leaks(data: Any, *, confirmed: bool = False, path: str = "") -> list[str]:
    """応答の中の、匿名・無料会員に出てはいけないキーの位置の一覧。確定の結果なら race_result の下の odds・popularity・payoff だけは許す。"""
    out: list[str] = []
    if isinstance(data, dict):
        for k, v in data.items():
            q = f"{path}.{k}"
            if k in SCOPE_FORBIDDEN_KEYS or (k in _PER_RACE_LOGLOSS and ".races[]" in q):
                out.append(q)
            elif k in SCOPE_CONFIRMED_ONLY_KEYS and not (confirmed and q.startswith(".race_result.")):
                out.append(q)
            else:
                out += find_scope_leaks(v, confirmed=confirmed, path=q)
    elif isinstance(data, list):
        for v in data:
            out += find_scope_leaks(v, confirmed=confirmed, path=path + "[]")
    elif isinstance(data, str) and SCOPE_FORBIDDEN_TEXT.search(data):
        out.append(f"{path}（文）")
    return out


# ── 予測成績（/api/prediction-results/*）──
# ゲスト・無料ログインには、◎などの予測と着順の突き合わせ（件数・率）と勝率までを出し、オッズ・人気・仮想購入（買い目・投資・払戻・回収率）は落とす。
# 会員（有料）・開発者はそのまま。許可リスト方式: 列挙した項目以外は出さない。

_PR_MARK_EXTRA = ("pop_sum", "pop_n", "odds_sum", "odds_n", "w_pop_sum", "w_pop_n", "w_odds_sum", "w_odds_n")
_PR_PICK_KEYS = ("horse_number", "horse_name", "finish", "mark")    # 勝率は有料会員から（2026-11-25）
_PR_HIGHLIGHT_PUBLIC_KINDS = ("big_miss",)            # ana_hit（単勝オッズ）・big_return（推奨馬券の金額）はオッズ由来
_PR_NEWS_PUBLIC_KINDS = ("day", "week", "big_miss")
_PR_BETS_ZERO = {"invested": 0, "returned": 0, "races_bought": 0, "races_hit": 0, "estimated_races": 0, "by_type": {}}
_PR_TOP_KEYS = ("scope", "date", "from", "to", "days", "statuses", "status", "label", "notes", "prediction_sources", "models", "built_at")
_PR_SUMMARY_KEYS = ("races", "races_settled", "races_no_prediction", "races_no_result", "top1", "logloss", "logloss_model")
_PR_RACE_KEYS = ("race_id", "venue", "round", "race_name", "grade", "included", "reason", "result_status", "n_runners", "result_top3")
# レースごとの勝ち馬の logloss は exp(-値) で勝ち馬の予測の勝率になるので出さない（2026-11-25）
_WIN_PCT_PHRASE = re.compile(r"（勝率[0-9.]+%）")


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


def _pr_highlight(h: dict) -> dict:
    """big_miss の value は ◎の勝率、text に「（勝率xx%）」が入る。どちらも落とす。"""
    out = {k: v for k, v in h.items() if k != "value"}
    if isinstance(out.get("text"), str):
        out["text"] = _WIN_PCT_PHRASE.sub("", out["text"])
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
            pub = _WIN_PCT_PHRASE.sub("", pub) if isinstance(pub, str) else pub
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
        out["highlights"] = [_pr_highlight(h) for h in data["highlights"] or [] if isinstance(h, dict) and h.get("kind") in _PR_HIGHLIGHT_PUBLIC_KINDS]
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
MEMBER_REDACTORS = {redact_predictions: redact_predictions_member, redact_race_detail: redact_race_detail_member}   # 出走馬カードは会員にはそのまま

# ── 会員限定のパス（匿名は 401）──

_MEMBER_ONLY_PATTERNS = [re.compile(p) for p in (
    r"^/api/race/[^/]+/(tracking-difficulty|bundle|bloodline-aptitude|final-odds)",
    r"^/api/race/[^/]+/result$",
    r"^/api/predictions$",
    r"^/api/growth-curve/(?!status)",
    r"^/api/race-quality/",
    r"^/api/v1/betting/",
    r"^/api/prediction-results/bets-(summary|by-level)",
    r"^/api/v1/races/[^/]+/(final-odds|megu-index|megu-index-predicted|tracking-difficulty|predictions/laps|running-map)",
    r"^/api/v1/horse/[^/]+/(megu-index-history|growth-curve)",
    r"^/api/v1/(pedigree-race-stats|data-analysis)/",
    r"^/api/v1/(bloodline-cluster/horse-aptitude|pedigree/race-note-3d-v2)",
    r"^/api/v1/races/[^/]+/analysis$",
    # 2026-11-25: 匿名・無料会員に netkeiba 独自の情報・確定前のオッズを出さない（docs/operations/tiered_access.md）
    r"^/api/data/",                                   # 保存してある生の JSON（追い切り・厩舎コメント・オッズ・馬柱）
    r"^/api/horse/",                                  # 馬のページ（馬柱の人気・オッズ・タイム指数・SmartRC）
    r"^/api/betting/pair-odds/",
    r"^/api/odds/",
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


# ── 研究（選んだ期間）の回収率と、サンプル買い目の期待値（2026-11-25 ユーザーの判断）──
# 開発者以外の応答から落とす。選んだ期間の回収率は将来を表さないので、画面にも API にも出さない。
ROI_NOTICE = "選んだ期間の回収率は、将来を表さないので出していません。"
_ROI_KEY = re.compile(r"(^|_)roi(_|$)")
_ROI_SENTENCE = re.compile(r"[^。]*(回収率|払戻率)[^。]*。?")


def _is_roi_key(k: Any) -> bool:
    return isinstance(k, str) and bool(_ROI_KEY.search(k))


def strip_research_roi(obj: Any) -> Any:
    """めぐ分析（/api/v1/races/<id>/analysis）の応答から、回収率・払戻率の列・値・文を落とす（入力は書き換えない）。
    列（columns の key が *_roi）は列ごと、値（*_roi・roi_*）はキーごと、notes の文は回収率・払戻率を含む文だけを落とし、落とした部品には 1 行の注記を足す。"""
    if isinstance(obj, list):
        return [strip_research_roi(v) for v in obj]
    if not isinstance(obj, dict):
        return obj
    out: dict[str, Any] = {}
    removed = False
    for k, v in obj.items():
        if _is_roi_key(k):
            removed = True
            continue
        if k == "columns" and isinstance(v, list):
            kept = [c for c in v if not (isinstance(c, dict) and _is_roi_key(c.get("key")))]
            removed = removed or len(kept) != len(v)
            out[k] = strip_research_roi(kept)
            continue
        if k == "notes" and isinstance(v, list):
            notes = []
            for t in v:
                if isinstance(t, str):
                    t2 = _ROI_SENTENCE.sub("", t).strip()
                    removed = removed or t2 != t
                    if t2:
                        notes.append(t2)
                else:
                    notes.append(strip_research_roi(t))
            out[k] = notes
            continue
        out[k] = strip_research_roi(v)
    if removed and ("notes" in out or "columns" in out):
        notes = out.get("notes") if isinstance(out.get("notes"), list) else []
        if ROI_NOTICE not in notes:
            out["notes"] = [*notes, ROI_NOTICE]
    return out


BET_HIDDEN_KEYS = frozenset({
    "race_ev", "portfolio_ev", "expected_payout_yen", "ev_line_multiple", "ev", "ev_calibrated", "expected_return", "expected_roi",
    "roi_pct", "roi_pct_calibrated", "roi_range_calibrated", "kelly_fraction", "evidence",
})
_EV_PHRASE = re.compile(r"[^。]*期待値[^。]*。?")


def strip_bet_ev(obj: Any) -> Any:
    """サンプル買い目の応答（開発者以外）から、期待値・期待回収率・Kelly・検証の期間の回収率を落とす（入力は書き換えない）。
    仮想購入の結果（result の invested・returned・roi）は実際の記録なので残す。期待値を書いた文（見送りの理由など）は、その文だけ落とす。"""
    if isinstance(obj, list):
        return [strip_bet_ev(v) for v in obj]
    if not isinstance(obj, dict):
        return obj
    out: dict[str, Any] = {}
    for k, v in obj.items():
        if k == "result":
            out[k] = v
        elif k in BET_HIDDEN_KEYS:
            continue
        elif isinstance(v, str) and "期待値" in v:
            t = _EV_PHRASE.sub("", v).strip()
            out[k] = t or "このレースはサンプル買い目の条件に届きません（見送り）。"
        else:
            out[k] = strip_bet_ev(v)
    return out
