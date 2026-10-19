"""予測結果の集計（予測成績）API。日次の集計ファイル（data/local/prediction_results/）を読んで返す。

  GET /api/prediction-results?scope=day|week|month|season&date=YYYYMMDD   期間の集計・日別推移・レース別
  GET /api/prediction-results/news?limit=N                                 ホーム画面のニュース風カード用
  GET /api/prediction-results/days                                         集計のある日の一覧（日付移動用）

認証は予測 API（/api/race/{id}/predictions）と同じ公開の GET。回収率（仮想購入）の表示を会員向けにするかは画面側で決める。
dev（KEIBA_ENV が stg/prod 以外）では、集計ファイルが無い日を dev ストアから都度計算して見せる（ホストの稼働には影響しない）。
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from src.pipeline.inference import prediction_results as pr

router = APIRouter()
_CACHE: dict[tuple, tuple[float, Any]] = {}
_TTL = 20.0
_WEEKDAY = "月火水木金土日"


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
async def prediction_result_days():
    def _run():
        days = pr.list_days()
        out = []
        for d in days:
            doc = pr.read_day(d)
            if doc:
                out.append({"date": d, "status": doc["status"], "races": doc["summary"]["races_settled"]})
        return {"days": out, "latest": out[-1]["date"] if out else None}
    return JSONResponse(await asyncio.to_thread(lambda: _cached(("days",), _run)))


@router.get("/api/prediction-results", response_class=JSONResponse)
async def prediction_results(scope: str = Query("day", pattern="^(day|week|month|season)$"),
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
    return JSONResponse(await asyncio.to_thread(lambda: _cached(("pr", scope, date), _run)))


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
    for hl in doc.get("highlights", [])[: max(0, limit - len(items))]:
        items.append({"kind": hl["kind"], "title": {"ana_hit": "注目: 穴馬が好走", "big_return": "注目: 大きな払戻", "big_miss": "注目: 大きく外れた"}.get(hl["kind"], "注目"),
                      "subtitle": hl["text"], "subtitle_public": hl["text"], "badge": badge, "date": latest,
                      "link": f"/race/{hl['race_id']}", "roi": None})
    return {"items": items[:limit], "latest": latest, "status": doc["status"]}


@router.get("/api/prediction-results/news", response_class=JSONResponse)
async def prediction_results_news(limit: int = Query(6, ge=1, le=20)):
    def _run():
        if _is_dev() and not pr.list_days():
            for back in range(0, 120):  # dev: 結果のある直近の日を探して一度だけ作る
                d = (datetime.now() - timedelta(days=back)).strftime("%Y%m%d")
                _ensure_dev_day(d)
                if pr.read_day(d):
                    break
        return build_news(limit)
    return JSONResponse(await asyncio.to_thread(lambda: _cached(("news", limit), _run)))
