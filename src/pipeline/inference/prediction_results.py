"""予測結果の日次集計（予測 vs 実際の着順・仮想購入の回収率）。

その日のレースが終わって結果が揃った時点で 1 日分を集計し、``data/local/prediction_results/<YYYYMMDD>.json`` に残す
（日次が正本。週・月・シーズンは日次の合算）。結果は 2 段階: 速報結果だけ=``provisional``（暫定）、確定結果が入ったら
``final``（確定）で上書きする。確定済みの日を暫定で上書きしない。

指標: ①予測 1 位（勝率最大）と ◎ の的中 ②◎・○・★ の着順分布（1着/2着/3着/4着以下）③推奨馬券を既定上限で購入した
場合の仮想の回収率（券種別）④注目レース（高オッズの的中・大きな払戻・大外れ）。仮想購入の買い目は、その時点で保存済みの
最新（過去のレースは確定）オッズで選ぶ。払戻は確定払戻があればそれ、無ければオッズからの概算（``estimated``）。
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from collections import defaultdict
from datetime import date as _date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("pipeline.inference.prediction_results")

JST = timezone(timedelta(hours=9))
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MAX_TOTAL_BET = 2000
BET_TYPES = ["tansho", "fukusho", "umaren", "wide", "sanrenpuku"]
BET_LABELS = {"tansho": "単勝", "fukusho": "複勝", "umaren": "馬連", "wide": "ワイド", "umatan": "馬単", "sanrenpuku": "3連複"}
MARKS = ("honmei", "pair", "star")
MARK_SYMBOLS = {"honmei": "◎", "pair": "○", "star": "★", "anchor": "✓", "show_val": "▲"}
_PAYOFF_KEYS = {"単勝": "tansho", "複勝": "fukusho", "馬連": "umaren", "ワイド": "wide", "馬単": "umatan",
                "三連複": "sanrenpuku", "3連複": "sanrenpuku"}


def enabled() -> bool:
    return os.environ.get("KEIBA_PREDICTION_RESULTS", "").strip().lower() in ("1", "true", "yes")


def store_dir() -> Path:
    return Path(os.environ.get("KEIBA_PREDICTION_RESULTS_DIR") or REPO_ROOT / "data" / "local" / "prediction_results")


# ───────────────────────── 結果・払戻の読み取り ─────────────────────────

def _to_int(x: Any) -> int | None:
    try:
        return int(str(x).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def parse_payoff(payoff: Any) -> dict[str, dict[tuple[int, ...], int]]:
    """保存形式の払戻（券種名 → {numbers, payout} か、その配列）を {券種: {組合せ: 100円あたりの払戻}} にする。"""
    out: dict[str, dict[tuple[int, ...], int]] = {}
    if not isinstance(payoff, dict):
        return out
    for jp, val in payoff.items():
        bt = _PAYOFF_KEYS.get(str(jp))
        if not bt:
            continue
        rows = val if isinstance(val, list) else [val]
        for r in rows:
            if not isinstance(r, dict):
                continue
            nums = [_to_int(p) for p in str(r.get("numbers", "")).replace("→", "-").split("-")]
            pay = _to_int(r.get("payout"))
            if not nums or None in nums or not pay:
                continue
            key = tuple(nums) if bt in ("tansho", "fukusho", "umatan") else tuple(sorted(nums))  # type: ignore[arg-type]
            out.setdefault(bt, {})[key] = pay
    return out


def _finish_map(entries: list[dict]) -> dict[int, int]:
    fm: dict[int, int] = {}
    for e in entries or []:
        hn, pos = _to_int(e.get("horse_number")), _to_int(e.get("finish_position"))
        if hn and pos and pos >= 1:          # 0・負（失格や取消の表記）は着順として扱わない
            fm[hn] = pos
    return fm


def load_race_result(storage: Any, race_id: str) -> dict | None:
    """確定結果（race_result）があればそれ、無ければ速報（race_result_on_time）。

    戻り値: {status: final|provisional, finish: {馬番: 着順}, n_runners, payoff: {...}, official: bool}。結果が無ければ None。
    """
    for cat, status in (("race_result", "final"), ("race_result_on_time", "provisional")):
        try:
            doc = storage.load(cat, race_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s の読み込み失敗 %s: %s", cat, race_id, exc)
            continue
        if not isinstance(doc, dict):
            continue
        fm = _finish_map(doc.get("entries") or [])
        if not fm:
            continue
        payoff = parse_payoff(doc.get("payoff"))
        if not payoff:
            for pcat in (f"{cat}_payoff",):
                try:
                    pdoc = storage.load(pcat, race_id)
                except Exception:  # noqa: BLE001
                    pdoc = None
                if isinstance(pdoc, dict):
                    payoff = parse_payoff(pdoc.get("payoff"))
        return {"status": status, "finish": fm, "n_runners": len(doc.get("entries") or []),
                "payoff": payoff, "official": bool(payoff), "date": doc.get("date")}
    return None


# ───────────────────────── 予測・仮想購入 ─────────────────────────

def load_predictions(storage: Any, race_id: str) -> dict | None:
    """GCS の race_predictions があればそれ、無ければローカルの shadow / バックフィル（画面と同じ経路）。"""
    try:
        doc = storage.load("race_predictions", race_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("race_predictions の読み込み失敗 %s: %s", race_id, exc)
        doc = None
    if isinstance(doc, dict) and doc.get("predictions"):
        out = dict(doc)
        out.setdefault("prediction_source", "gcs")
        return out
    try:
        from src.api.local_predictions import load_local_prediction_payload
    except ImportError:
        return None
    try:
        return load_local_prediction_payload(race_id, storage)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ローカル予測の読み込み失敗 %s: %s", race_id, exc)
        return None


def settle_candidate(bt: str, pair: tuple[int, ...], amount: int, odds: float, result: dict) -> tuple[bool, float, bool]:
    """1 つの買い目を精算する。戻り値 (的中, 払戻[円], 概算か)。"""
    fm: dict[int, int] = result["finish"]
    n = result.get("n_runners") or len(fm)
    by_pos = defaultdict(list)
    for hn, pos in fm.items():
        by_pos[pos].append(hn)
    top3 = {h for h, p in fm.items() if p <= 3}
    if bt == "tansho":
        hit = fm.get(pair[0]) == 1
        key: tuple[int, ...] = (pair[0],)
    elif bt == "fukusho":
        hit = fm.get(pair[0], 99) <= (3 if n >= 8 else 2)
        key = (pair[0],)
    elif bt == "umaren":
        hit = sorted(pair) == sorted(by_pos.get(1, []) + by_pos.get(2, [])) and len(pair) == 2
        key = tuple(sorted(pair))
    elif bt == "wide":
        hit = all(fm.get(h, 99) <= 3 for h in pair)
        key = tuple(sorted(pair))
    elif bt == "sanrenpuku":
        hit = len(top3) == 3 and set(pair) == top3
        key = tuple(sorted(pair))
    else:
        return False, 0.0, False
    if not hit:
        return False, 0.0, False
    per100 = (result.get("payoff") or {}).get(bt, {}).get(key)
    if per100:
        return True, amount / 100 * per100, False
    return True, amount * float(odds or 0), True  # 公式払戻が無い（暫定）→ オッズからの概算


def simulate_bets(predictions: list[dict], storage: Any, race_id: str, result: dict,
                  max_total_bet: int | None = DEFAULT_MAX_TOTAL_BET) -> dict:
    """推奨馬券（API と同じ最適化）を既定上限で購入したとして精算する。オッズが無いレースは買い目なし。"""
    from src.pipeline.inference.betting import BettingConfig, BettingOptimizer
    from src.pipeline.inference.odds_source import resolve_odds

    odds = resolve_odds(storage, race_id, mode="actual", phase="latest")
    if odds.get("source") is None:
        return {"status": "no_odds", "total_bet": 0, "total_return": 0.0, "by_type": {}, "candidates": [], "estimated": False}
    cfg = BettingConfig(max_total_bet=max_total_bet, bet_types=list(BET_TYPES))
    portfolio = BettingOptimizer(cfg).optimize(predictions, odds["pair_odds"], 100000, single_odds=odds["single_odds"])
    total_bet = 0
    total_ret = 0.0
    estimated = False
    by_type: dict[str, dict] = {}
    cands_out = []
    for c in portfolio.get("candidates") or []:
        hit, ret, est = settle_candidate(c.bet_type, tuple(c.pair), int(c.bet_amount), float(c.odds), result)
        estimated = estimated or (hit and est)
        total_bet += int(c.bet_amount)
        total_ret += ret
        b = by_type.setdefault(c.bet_type, {"invested": 0, "returned": 0.0, "n": 0, "hit": 0})
        b["invested"] += int(c.bet_amount)
        b["returned"] += ret
        b["n"] += 1
        b["hit"] += int(hit)
        cands_out.append({"bet_type": c.bet_type, "pair": list(c.pair), "amount": int(c.bet_amount),
                          "odds": float(c.odds), "hit": bool(hit), "return": round(ret)})
    for b in by_type.values():
        b["returned"] = round(b["returned"])
    return {"status": "ok", "total_bet": total_bet, "total_return": round(total_ret), "by_type": by_type,
            "candidates": cands_out, "estimated": estimated, "odds_phase": odds.get("phase")}


# ───────────────────────── 1 レース・1 日の集計 ─────────────────────────

def _bucket(pos: int | None) -> str | None:
    if pos is None:
        return None
    return "f1" if pos == 1 else "f2" if pos == 2 else "f3" if pos == 3 else "f4p"


def build_race_record(race_id: str, meta: dict, storage: Any, max_total_bet: int | None = DEFAULT_MAX_TOTAL_BET) -> dict:
    """1 レースの予測と結果の突き合わせ。予測・結果のどちらかが無いときは理由つきで返す（集計から除外）。"""
    rec: dict[str, Any] = {
        "race_id": race_id, "venue": meta.get("venue", ""), "round": meta.get("round") or int(str(race_id)[-2:] or 0),
        "race_name": meta.get("race_name", ""), "grade": meta.get("grade") or None,
    }
    pred = load_predictions(storage, race_id)
    res = load_race_result(storage, race_id)
    if not pred or not pred.get("predictions"):
        return {**rec, "included": False, "reason": "no_prediction"}
    if not res:
        return {**rec, "included": False, "reason": "no_result", "prediction_source": pred.get("prediction_source")}
    entries = [e for e in pred["predictions"] if e.get("win_prob") is not None]
    if not entries:
        return {**rec, "included": False, "reason": "no_prediction"}
    fm = res["finish"]
    rec.update(included=True, result_status=res["status"], payoff_official=res["official"],
               prediction_source=pred.get("prediction_source") or "gcs", model_version=pred.get("model_version") or pred.get("model_type"),
               n_runners=res.get("n_runners"))
    top1 = max(entries, key=lambda e: float(e.get("win_prob") or 0))

    def pick(e: dict) -> dict:
        hn = int(e.get("horse_number") or 0)
        return {"horse_number": hn, "horse_name": e.get("horse_name", ""), "finish": fm.get(hn),
                "win_prob": round(float(e.get("win_prob") or 0), 4), "win_odds": e.get("win_odds"),
                "ev_win": e.get("ev_win"), "mark": e.get("mark_type")}

    rec["top1"] = pick(top1)
    rec["marks"] = {m: [pick(e) for e in entries if e.get("mark_type") == m] for m in MARKS}
    order = sorted(fm.items(), key=lambda kv: kv[1])
    rec["result_top3"] = [{"horse_number": h, "finish": p} for h, p in order if p <= 3]
    ps = [float((next((x for x in entries if int(x.get("horse_number") or 0) == h), {}) or {}).get("win_prob") or 0)
          for h, p in fm.items() if p == 1]
    tot = sum(float(e.get("win_prob") or 0) for e in entries) or 1.0
    rec["winner_logloss"] = round(-math.log(max(max(ps, default=0) / tot, 1e-6)), 4) if ps else None
    try:
        rec["bets"] = simulate_bets(entries, storage, race_id, res, max_total_bet)
    except Exception as exc:  # noqa: BLE001
        logger.warning("仮想購入の計算失敗 %s: %s", race_id, exc)
        rec["bets"] = {"status": "error", "total_bet": 0, "total_return": 0.0, "by_type": {}, "candidates": [], "estimated": False}
    return rec


def _blank_counts() -> dict:
    return {"n": 0, "f1": 0, "f2": 0, "f3": 0, "f4p": 0}


def summarize(records: list[dict]) -> dict:
    """レース記録の一覧から、合算できる件数ベースの集計を作る（率は読む側で計算する）。"""
    inc = [r for r in records if r.get("included")]
    s: dict[str, Any] = {
        "races": len(records), "races_settled": len(inc),
        "races_no_prediction": sum(1 for r in records if r.get("reason") == "no_prediction"),
        "races_no_result": sum(1 for r in records if r.get("reason") == "no_result"),
        "top1": _blank_counts(), "marks": {m: _blank_counts() for m in MARKS},
        "bets": {"invested": 0, "returned": 0, "races_bought": 0, "races_hit": 0, "estimated_races": 0, "by_type": {}},
        "logloss": {"n": 0, "sum": 0.0},
    }
    for r in inc:
        for key, p in (("top1", r.get("top1")),):
            b = _bucket((p or {}).get("finish"))
            if b:
                s[key]["n"] += 1
                s[key][b] += 1
        for m in MARKS:
            for p in r["marks"].get(m, []):
                b = _bucket(p.get("finish"))
                if b:
                    s["marks"][m]["n"] += 1
                    s["marks"][m][b] += 1
        bets = r.get("bets") or {}
        if bets.get("total_bet"):
            sb = s["bets"]
            sb["invested"] += bets["total_bet"]
            sb["returned"] += bets["total_return"]
            sb["races_bought"] += 1
            sb["races_hit"] += int(bets["total_return"] > 0)
            sb["estimated_races"] += int(bool(bets.get("estimated")))
            for bt, v in (bets.get("by_type") or {}).items():
                t = sb["by_type"].setdefault(bt, {"invested": 0, "returned": 0, "n": 0, "hit": 0})
                for k in t:
                    t[k] += v[k]
        if r.get("winner_logloss") is not None:
            s["logloss"]["n"] += 1
            s["logloss"]["sum"] += r["winner_logloss"]
    s["logloss"]["sum"] = round(s["logloss"]["sum"], 4)
    return s


def merge_summaries(parts: list[dict]) -> dict:
    """日次の集計を合算する（週・月・シーズン）。"""
    out = summarize([])
    for p in parts:
        for k in ("races", "races_settled", "races_no_prediction", "races_no_result"):
            out[k] += p.get(k, 0)
        for k in ("n", "f1", "f2", "f3", "f4p"):
            out["top1"][k] += p["top1"][k]
            for m in MARKS:
                out["marks"][m][k] += p["marks"][m][k]
        pb, ob = p["bets"], out["bets"]
        for k in ("invested", "returned", "races_bought", "races_hit", "estimated_races"):
            ob[k] += pb[k]
        for bt, v in pb["by_type"].items():
            t = ob["by_type"].setdefault(bt, {"invested": 0, "returned": 0, "n": 0, "hit": 0})
            for k in t:
                t[k] += v[k]
        out["logloss"]["n"] += p["logloss"]["n"]
        out["logloss"]["sum"] = round(out["logloss"]["sum"] + p["logloss"]["sum"], 4)
    return out


def highlights(records: list[dict], limit_each: int = 3) -> list[dict]:
    """注目レース: 高オッズの的中・大きな払戻・大外れ。"""
    ana, big, miss = [], [], []
    for r in records:
        if not r.get("included"):
            continue
        label = f"{r['venue']}{r['round']}R"
        for m in MARKS:
            for p in r["marks"].get(m, []):
                f = p.get("finish")
                if f and f <= 3 and (p.get("win_odds") or 0) >= 10:
                    ana.append({"kind": "ana_hit", "race_id": r["race_id"], "value": p["win_odds"],
                                "text": f"{label} {MARK_SYMBOLS[m]}{p['horse_name']}（単勝{p['win_odds']}倍）が{f}着に"})
        b = r.get("bets") or {}
        if b.get("total_bet") and b["total_return"] >= max(5000, 3 * b["total_bet"]):
            big.append({"kind": "big_return", "race_id": r["race_id"], "value": b["total_return"],
                        "text": f"{label} 推奨馬券が{b['total_bet']:,}円→{b['total_return']:,}円"})
        h = (r.get("marks") or {}).get("honmei") or []
        for p in h:
            if p.get("win_prob", 0) >= 0.30 and (p.get("finish") or 0) >= 6:
                miss.append({"kind": "big_miss", "race_id": r["race_id"], "value": p["win_prob"],
                             "text": f"{label} ◎{p['horse_name']}（勝率{p['win_prob']*100:.0f}%）は{p['finish']}着"})
    ana.sort(key=lambda x: -x["value"])
    big.sort(key=lambda x: -x["value"])
    miss.sort(key=lambda x: -x["value"])
    return ana[:limit_each] + big[:limit_each] + miss[:limit_each]


def build_day(date: str, storage: Any, max_total_bet: int | None = DEFAULT_MAX_TOTAL_BET,
              race_filter: Callable[[dict], bool] | None = None) -> dict:
    """1 日分（YYYYMMDD）を集計する。status は全レースに確定結果があれば final、無ければ provisional。"""
    rl = storage.load("race_lists", date) or {}
    races = [r for r in (rl.get("races") or []) if r.get("race_id") and (race_filter is None or race_filter(r))]
    races.sort(key=lambda r: (r.get("venue", ""), int(r.get("round") or 0), r["race_id"]))
    records = [build_race_record(r["race_id"], r, storage, max_total_bet) for r in races]
    inc = [r for r in records if r.get("included")]
    all_final = bool(inc) and all(r.get("result_status") == "final" and r.get("payoff_official") for r in inc) \
        and not any(r.get("reason") == "no_result" for r in records)
    srcs = defaultdict(int)
    models = set()
    for r in inc:
        srcs[r.get("prediction_source") or "gcs"] += 1
        if r.get("model_version"):
            models.add(str(r["model_version"]))
    notes = ["仮想購入は、推奨馬券（上限 {:,} 円/R）をその時点で保存済みの最新オッズで選び、払戻は確定払戻（無ければオッズからの概算）で精算".format(max_total_bet or 0)]
    if srcs.get("backfill") or srcs.get("shadow"):
        notes.append("バックフィル/shadow で算出した予測を含む（レース当日の確定した予測ではない）")
    return {
        "date": date, "status": "final" if all_final else "provisional",
        "built_at": datetime.now(JST).isoformat(timespec="seconds"),
        "max_total_bet": max_total_bet, "prediction_sources": dict(srcs), "models": sorted(models),
        "summary": summarize(records), "highlights": highlights(records), "races": records, "notes": notes,
    }


# ───────────────────────── 保存・読み出し ─────────────────────────

def day_path(date: str, base: Path | None = None) -> Path:
    return (base or store_dir()) / f"{date}.json"


def read_day(date: str, base: Path | None = None) -> dict | None:
    p = day_path(date, base)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_day(doc: dict, base: Path | None = None, *, force: bool = False) -> Path | None:
    """アトミックに保存する。確定済みの日を暫定で上書きしない（force を除く）。書かなかったときは None。"""
    base = base or store_dir()
    base.mkdir(parents=True, exist_ok=True)
    prev = read_day(doc["date"], base)
    if prev and prev.get("status") == "final" and doc.get("status") != "final" and not force:
        return None
    p = day_path(doc["date"], base)
    fd, tmp = tempfile.mkstemp(dir=str(base), prefix=f".{doc['date']}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return p


def aggregate_day(date: str, storage: Any, *, max_total_bet: int | None = DEFAULT_MAX_TOTAL_BET,
                  base: Path | None = None) -> dict:
    """集計して保存する。レース記録が 1 件も無い日（開催なし）は保存しない。"""
    doc = build_day(date, storage, max_total_bet)
    if doc["summary"]["races_settled"] == 0:
        doc["saved"] = False
        return doc
    doc["saved"] = write_day(doc, base) is not None
    return doc


def week_range(date: str) -> tuple[str, str]:
    d = datetime.strptime(date, "%Y%m%d").date()
    mon = d - timedelta(days=d.weekday())
    return mon.strftime("%Y%m%d"), (mon + timedelta(days=6)).strftime("%Y%m%d")


def period_range(scope: str, date: str) -> tuple[str, str]:
    d = datetime.strptime(date, "%Y%m%d").date()
    if scope == "day":
        return date, date
    if scope == "week":
        return week_range(date)
    if scope == "month":
        first = d.replace(day=1)
        nxt = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
        return first.strftime("%Y%m%d"), (nxt - timedelta(days=1)).strftime("%Y%m%d")
    return f"{d.year}0101", f"{d.year}1231"


def list_days(base: Path | None = None) -> list[str]:
    b = base or store_dir()
    return sorted(p.stem for p in b.glob("[0-9]" * 8 + ".json"))


def period_summary(scope: str, date: str, base: Path | None = None) -> dict:
    """期間の合算と、日別の推移（回収率・本命の勝率など）。"""
    lo, hi = period_range(scope, date)
    days = [d for d in list_days(base) if lo <= d <= hi]
    docs = [x for x in (read_day(d, base) for d in days) if x]
    summ = merge_summaries([x["summary"] for x in docs]) if docs else summarize([])
    series = []
    for x in docs:
        s = x["summary"]
        inv = s["bets"]["invested"]
        series.append({"date": x["date"], "status": x["status"], "roi": (s["bets"]["returned"] / inv) if inv else None,
                       "honmei_win": (s["marks"]["honmei"]["f1"] / s["marks"]["honmei"]["n"]) if s["marks"]["honmei"]["n"] else None,
                       "races": s["races_settled"]})
    return {"scope": scope, "date": date, "from": lo, "to": hi, "days": [x["date"] for x in docs],
            "statuses": {x["date"]: x["status"] for x in docs},
            "status": "final" if docs and all(x["status"] == "final" for x in docs) else "provisional",
            "summary": summ, "series": series,
            "highlights": [h for x in docs for h in x.get("highlights", [])][:12],
            "notes": (docs[-1].get("notes") if docs else [])}


def rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def refresh_for_dates(dates: list[str], *, reason: str = "") -> dict:
    """指定した開催日を集計して保存する（自動実行用。KEIBA_PREDICTION_RESULTS=1 のときだけ。失敗しても呼び出し元を止めない）。

    暫定/確定は結果の状態で自動判定する（確定済みの日は暫定で上書きしない）。
    """
    if not enabled():
        return {"status": "disabled"}
    out: dict[str, str] = {}
    try:
        from src.scraper.storage import HybridStorage

        storage = HybridStorage()
        for d in sorted({str(x).replace("-", "")[:8] for x in dates if x}):
            try:
                doc = aggregate_day(d, storage)
                out[d] = f"{doc['status']}:{doc['summary']['races_settled']}R" + ("" if doc.get("saved") else ":not_saved")
            except Exception as exc:  # noqa: BLE001
                logger.warning("予測結果の集計に失敗 %s (%s): %s", d, reason, exc)
                out[d] = "error"
    except Exception as exc:  # noqa: BLE001
        logger.warning("予測結果の集計を始められない (%s): %s", reason, exc)
        return {"status": "error", "error": str(exc)[:200]}
    logger.info("予測結果の集計 (%s): %s", reason, out)
    return {"status": "ok", "days": out}
