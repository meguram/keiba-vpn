"""予測結果の集計（予測成績）API。日次の集計ファイル（data/local/prediction_results/）を読んで返す。

  GET /api/prediction-results?scope=day|week|month|season&date=YYYYMMDD   期間の集計・日別推移・レース別
  GET /api/prediction-results/news?limit=N                                 ホーム画面のニュース風カード用
  GET /api/prediction-results/days                                         集計のある日の一覧（日付移動用）

公開の GET だが、オッズ・人気・仮想購入（買い目・投資・払戻・回収率）は、有料会員・開発者以外（ゲスト・無料ログイン）の応答からサーバ側で落とす
（src/api/tier.py の redact_prediction_results）。仕様: docs/operations/tiered_access.md
dev（KEIBA_ENV が stg/prod 以外）では、集計ファイルが無い日を dev ストアから都度計算して見せる（ホストの稼働には影響しない）。
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from src.api.auth import is_developer, is_member
from src.api.tier import MEMBER_REQUIRED_BODY, redact_prediction_results, redact_prediction_results_member
from src.pipeline.inference import prediction_results as pr

router = APIRouter()
_CACHE: dict[tuple, tuple[float, Any]] = {}
_TTL = 20.0

from src.utils import cache_epoch as _cache_epoch  # noqa: E402

_cache_epoch.register("prediction_results_api", _CACHE.clear)
_cache_epoch.register("prediction_results_gcs", pr.clear_remote_cache)
_WEEKDAY = "月火水木金土日"


def _respond(request: Request, data: Any) -> JSONResponse:
    """開発者はそのまま、有料会員は実オッズ・人気を落とし、それ以外（ゲスト・無料）はオッズ由来の項目を全部落として返す（キャッシュは落とす前の値のまま共有）。"""
    if is_developer(request):
        return JSONResponse(data)
    if is_member(request):
        return JSONResponse(redact_prediction_results_member(data))      # 有料会員: 実オッズ・人気は出さない（actual_odds_public=false の間）
    return JSONResponse(redact_prediction_results(request.url.path, data))


def _is_dev() -> bool:
    return (os.environ.get("KEIBA_ENV") or "dev").strip().lower() not in ("stg", "prod")


def _sig() -> tuple:
    base = pr.store_dir()
    try:
        files = list(base.glob("[0-9]" * 8 + ".json"))
        return (len(files), max((p.stat().st_mtime for p in files), default=0))
    except OSError:
        return (0, 0)


def _cached(key: tuple, build):
    now = time.time()
    k = key + (_sig(),)
    hit = _CACHE.get(k)
    if hit and now - hit[0] < _TTL:
        return hit[1]
    val = build()
    _CACHE[k] = (now, val)
    if len(_CACHE) > 200:
        for old in sorted(_CACHE, key=lambda x: _CACHE[x][0])[:50]:
            _CACHE.pop(old, None)
    return val


def _kpi(s: dict) -> dict:
    h, t1, b = s["marks"]["honmei"], s["top1"], s["bets"]
    ll = s["logloss"]
    return {
        "honmei": {"n": h["n"], "win": pr.rate(h["f1"], h["n"]), "top2": pr.rate(h["f1"] + h["f2"], h["n"]),
                   "top3": pr.rate(h["f1"] + h["f2"] + h["f3"], h["n"])},
        "top1": {"n": t1["n"], "win": pr.rate(t1["f1"], t1["n"]), "top3": pr.rate(t1["f1"] + t1["f2"] + t1["f3"], t1["n"])},
        "marks": {m: {**c, "rates": {k: pr.rate(c[k], c["n"]) for k in ("f1", "f2", "f3", "f4p")}} for m, c in s["marks"].items()},
        "roi": pr.rate(b["returned"], b["invested"]), "invested": b["invested"], "returned": b["returned"],
        "races_bought": b["races_bought"], "races_hit": b["races_hit"], "estimated_races": b["estimated_races"],
        "by_type": {bt: {**v, "label": pr.BET_LABELS.get(bt, bt), "roi": pr.rate(v["returned"], v["invested"])}
                    for bt, v in b["by_type"].items()},
        "winner_logloss": round(ll["sum"] / ll["n"], 4) if ll["n"] else None,
        # 市場補正した確率のレースだけ: 同じレースの元のモデルの確率での勝ち馬 LogLoss と、補正後の値（小さいほど良い）
        "winner_logloss_model": ({"n": lm["n"], "model": round(lm["sum"] / lm["n"], 4), "calibrated": round(lm["sum_calibrated"] / lm["n"], 4)}
                                 if (lm := s.get("logloss_model")) and lm.get("n") else None),
    }


def _ensure_dev_day(date: str) -> None:
    """dev だけ: 集計ファイルが無い日を dev ストアから計算して保存する。"""
    if not _is_dev() or pr.read_day(date):
        return
    try:
        from src.scraper.storage import HybridStorage

        pr.aggregate_day(date, HybridStorage())
    except Exception:  # noqa: BLE001
        pass


def _label(date: str) -> str:
    d = datetime.strptime(date, "%Y%m%d")
    return f"{d.month}/{d.day}({_WEEKDAY[d.weekday()]})"


@router.get("/api/prediction-results/days", response_class=JSONResponse)
async def prediction_result_days(request: Request):
    def _run():
        days = pr.list_days()
        out = []
        for d in days:
            doc = pr.read_day(d)
            if doc:
                out.append({"date": d, "status": doc["status"], "races": doc["summary"]["races_settled"]})
        return {"days": out, "latest": out[-1]["date"] if out else None}
    return _respond(request, await asyncio.to_thread(lambda: _cached(("days",), _run)))


@router.get("/api/prediction-results", response_class=JSONResponse)
async def prediction_results(request: Request, scope: str = Query("day", pattern="^(day|week|month|season)$"),
                             date: str | None = Query(None, pattern=r"^\d{8}$")):
    def _run():
        d = date
        if not d:
            days = pr.list_days()
            d = days[-1] if days else datetime.now().strftime("%Y%m%d")
        _ensure_dev_day(d)
        ps = pr.period_summary(scope, d)
        out: dict[str, Any] = {**{k: ps[k] for k in ("scope", "date", "from", "to", "days", "statuses", "status", "series", "highlights", "notes")},
                               "kpi": _kpi(ps["summary"]), "summary": ps["summary"], "label": _label(d)}
        if scope == "day":
            doc = pr.read_day(d)
            out["races"] = (doc or {}).get("races", [])
            out["prediction_sources"] = (doc or {}).get("prediction_sources", {})
            out["models"] = (doc or {}).get("models", [])
            out["built_at"] = (doc or {}).get("built_at")
        return out
    return _respond(request, await asyncio.to_thread(lambda: _cached(("pr", scope, date), _run)))


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.0f}%"


def build_news(limit: int = 6) -> dict:
    days = pr.list_days()
    if not days:
        return {"items": []}
    latest = days[-1]
    doc = pr.read_day(latest)
    if not doc:
        return {"items": []}
    badge = "確定" if doc["status"] == "final" else "暫定"
    k = _kpi(doc["summary"])
    h = k["honmei"]
    items: list[dict] = []
    f1 = doc["summary"]["marks"]["honmei"]["f1"]
    pub = f"本命 {f1}/{h['n']}R が1着・複勝圏 {_pct(h['top3'])}" if h["n"] else "集計できるレースがありません"
    sub = pub + (f"・回収率 {_pct(k['roi'])}" if k["roi"] is not None else "")
    items.append({"kind": "day", "title": f"{_label(latest)} 結果まとめ", "subtitle": sub, "subtitle_public": pub, "badge": badge,
                  "date": latest, "link": f"/prediction-results?scope=day&date={latest}", "roi": k["roi"]})
    wk = pr.period_summary("week", latest)
    if len(wk["days"]) >= 1:
        wk_k = _kpi(wk["summary"])
        wh = wk_k["honmei"]
        wpub = f"本命1着率 {_pct(wh['win'])}（{wh['n']}R）・複勝圏 {_pct(wh['top3'])}"
        items.append({"kind": "week", "title": "今週の累計", "subtitle": wpub + (f"・回収率 {_pct(wk_k['roi'])}" if wk_k["roi"] is not None else ""),
                      "subtitle_public": wpub, "badge": "確定" if wk["status"] == "final" else "暫定", "date": latest,
                      "link": f"/prediction-results?scope=week&date={latest}", "roi": wk_k["roi"]})
    shown = [hl for hl in doc.get("highlights", []) if hl["kind"] != "big_miss"]  # 外れた予想は速報に出さない
    for hl in shown[: max(0, limit - len(items))]:
        items.append({"kind": hl["kind"], "title": {"ana_hit": "注目: 穴馬が好走", "big_return": "注目: 大きな払戻"}.get(hl["kind"], "注目"),
                      "subtitle": hl["text"], "subtitle_public": hl["text"], "badge": badge, "date": latest,
                      "link": f"/race/{hl['race_id']}", "roi": None})
    return {"items": items[:limit], "latest": latest, "status": doc["status"]}


@router.get("/api/prediction-results/mark-card", response_class=JSONResponse)
async def prediction_results_mark_card(request: Request):
    """ホームの『印の成績』カード（◎○▲★消の 1着-2着-3着-4着以下。先週・直近5週間・年）。"""
    today = datetime.now(pr.JST).date()
    return _respond(request, await asyncio.to_thread(lambda: _cached(("mark-card", today.isoformat()), lambda: pr.mark_card(today))))


@router.get("/api/prediction-results/bets-by-level", response_class=JSONResponse)
async def prediction_results_bets_by_level(request: Request):
    """ホーム用: 印の成績と同じ期間（先週・直近5週間・年）の、推奨馬券の仮想購入の回収率（推奨度 1〜4 別。見送りは含めない）。会員限定（tier.py）。"""
    if not is_member(request):
        return JSONResponse(MEMBER_REQUIRED_BODY, status_code=401)
    today = datetime.now(pr.JST).date()
    return _respond(request, await asyncio.to_thread(lambda: _cached(("bets-by-level", today.isoformat()), lambda: pr.bets_level_card(today))))


@router.get("/api/prediction-results/news", response_class=JSONResponse)
async def prediction_results_news(request: Request, limit: int = Query(6, ge=1, le=20)):
    def _run():
        if _is_dev() and not pr.list_days():
            for back in range(0, 120):  # dev: 結果のある直近の日を探して一度だけ作る
                d = (datetime.now() - timedelta(days=back)).strftime("%Y%m%d")
                _ensure_dev_day(d)
                if pr.read_day(d):
                    break
        return build_news(limit)
    return _respond(request, await asyncio.to_thread(lambda: _cached(("news", limit), _run)))

def _bets_block(s: dict) -> dict:
    b = s["bets"]
    return {"invested": b["invested"], "returned": b["returned"], "roi": pr.rate(b["returned"], b["invested"]),
            "races_bought": b["races_bought"], "races_hit": b["races_hit"], "estimated_races": b["estimated_races"]}


@router.get("/api/prediction-results/bets-summary", response_class=JSONResponse)
async def prediction_results_bets_summary(request: Request, date: str = Query(..., pattern=r"^\d{8}$")):
    """レース一覧用: 指定日の各レースの仮想購入の結果（的中フラグ）と、その日・その週の回収率。会員限定（tier.py）。"""
    def _run():
        _ensure_dev_day(date)
        doc = pr.read_day(date) or {}
        wk_lo, wk_hi = pr.week_range(date)
        wk = pr.period_summary("week", date)
        day = pr.period_summary("day", date)
        races = {}
        for r in doc.get("races") or []:
            b = r.get("bets") or {}
            tb = b.get("total_bet") or 0
            if not r.get("included") or not tb:
                continue
            ret = b.get("total_return") or 0
            races[r["race_id"]] = {"total_bet": tb, "total_return": round(ret), "hit": ret > 0, "roi": pr.rate(ret, tb), "estimated": bool(b.get("estimated"))}
        return {"date": date, "label": _label(date), "week": {"from": wk_lo, "to": wk_hi, "days": wk["days"], "status": wk["status"], **_bets_block(wk["summary"])},
                "day": {"status": day["status"], **_bets_block(day["summary"])}, "races": races,
                "note": "推奨馬券を、保存済みの最新オッズで購入したとみなした仮想購入（払戻は確定払戻）。実際の購入ではありません。"}
    if not is_member(request):
        return JSONResponse(MEMBER_REQUIRED_BODY, status_code=401)
    return _respond(request, await asyncio.to_thread(lambda: _cached(("bets", date), _run)))


