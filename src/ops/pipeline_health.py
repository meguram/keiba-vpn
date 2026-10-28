"""開催日パイプラインの点検: 直近の開催日ごとに、取得 → 予測 → 結果 → 成績集計が最後まで完了したかを段階別に判定する。

ホストを見ない。材料は GCS だけ（``GcsSources``）: データの実在（``chuou/data/preprocessed/netkeiba/pc/<カテゴリ>/<年>/<race_id>.json``）、
予測と実行記録（``state/data/local/inference/{predictions_v3,v5,v6,events}``。ホストでも Jobs でも、ここに上がったものを見る）、
予測成績（``ops/prediction_results/<日>.json``）。判定の純粋な部分（``day_health`` / ``build``）は ``Sources`` を差し替えてテストできる。

段階（開催日 D）:
  eve      前日夕方の出馬表の取得          期限 D-1 20:00   race_shutuba が全レース（1 開催日 1 場 12 レースを期待する）
  odds     当日のオッズ（T-60/T-15）         期限 D 18:00     race_odds が全レース
  predict  馬体重の検知 → 予測               期限 D 18:00     予測ファイル（v6、無ければ v5/v3）が全レースにある。events に失敗があれば警告
  on_time  速報結果                         期限 D 20:00     race_result_on_time が全レース
  index    レース指数                       期限 D+1 12:00   race_index が全レース
  final    確定結果                         期限 D+3 00:00   race_result が全レース（公開は数日遅れ。期限前は『待ち』）
  results  予測成績の集計                    期限 D+3 00:00   prediction_results/D.json が final（暫定は期限前なら待ち）
期限前に未完了なら『待ち』（異常ではない）、期限後に未完了なら『要対応』。
限界: 開催日は GCS の出馬表の日付から見つける。出馬表が 1 件も無い開催日（前日の取得が丸ごと失敗）は一覧に出ないので、
直近・直後の土日で出馬表が無い日は ``weekend_gaps`` で別に警告する（開催のない週末なら誤報になるため警告止まり）。
events は GCS への反映が遅れる／一部のことがある（その場合はエンジン別の件数を出さない）。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

JST = timezone(timedelta(hours=9))
PC = "chuou/data/preprocessed/netkeiba/pc"
INF = "state/data/local/inference"
ENGINES = ("v6", "v5", "v3")  # 画面に出る側から順に
STAGES = [
    ("eve", "前日の出馬表", "race_shutuba"),
    ("odds", "オッズ", "race_odds"),
    ("predict", "予測", None),
    ("on_time", "速報結果", "race_result_on_time"),
    ("index", "レース指数", "race_index"),
    ("final", "確定結果", "race_result"),
    ("results", "予測成績", None),
]
CATS = tuple(c for _, _, c in STAGES if c)
DEADLINE_H = {"eve": -4, "odds": 18, "predict": 18, "on_time": 20, "index": 36, "final": 72, "results": 72}  # 開催日 0:00 JST からの時間
LEVEL_RANK = {"ok": 0, "na": 0, "wait": 1, "warn": 2, "ng": 3}
VENUES = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京", "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
RACES_PER_DAY = 12
_RID = re.compile(r"^\d{12}$")


class Sources(Protocol):
    def race_days(self, now: datetime) -> list[str]: ...
    def matrix(self, date: str) -> dict | None: ...
    def events(self, date: str) -> list[dict]: ...
    def pred_files(self, engine: str, date: str) -> set[str]: ...
    def results_summary(self, date: str) -> dict | None: ...


# ───────────────────────── GCS から読む実体 ─────────────────────────

class GcsSources:
    """``BlobIO``（``src.ops.gcs_io``）だけで読む。1 回の ``build`` の間は一覧を使い回す（インスタンスは都度作る）。"""

    def __init__(self, io: Any) -> None:
        self.io = io
        self._ids: dict[tuple[str, str], set[str]] = {}
        self._group_date: dict[str, str | None] = {}
        self._by_date: dict[str, list[str]] = {}

    def _race_ids(self, cat: str, year: str) -> set[str]:
        k = (cat, year)
        if k not in self._ids:
            pre = f"{PC}/{cat}/{year}/"
            self._ids[k] = {key[len(pre):-5] for key in self.io.list(pre) if key.endswith(".json") and _RID.match(key[len(pre):-5])}
        return self._ids[k]

    def _date_of_group(self, group: str, year: str) -> str | None:
        """開催日（年・場・回・日目の組）の日付。組の中で最初に見つかった出馬表の ``date`` を読む（変わらないので覚える）。"""
        if group not in self._group_date:
            date = None
            for rid in sorted(r for r in self._race_ids("race_shutuba", year) if r[:10] == group)[:2]:
                raw = self.io.get_bytes(f"{PC}/race_shutuba/{year}/{rid}.json")
                try:
                    date = str(json.loads((raw or b"{}").decode("utf-8")).get("date") or "").replace("-", "") or None
                except (ValueError, UnicodeDecodeError):
                    date = None
                if date:
                    break
            self._group_date[group] = date
        return self._group_date[group]

    def race_days(self, now: datetime) -> list[str]:
        lo, hi = (now - timedelta(days=21)).strftime("%Y%m%d"), (now + timedelta(days=7)).strftime("%Y%m%d")
        by_date: dict[str, list[str]] = {}
        for year in sorted({str(now.year), str((now - timedelta(days=21)).year)}):
            by_venue: dict[str, list[str]] = {}
            for g in sorted({r[:10] for r in self._race_ids("race_shutuba", year)}):
                by_venue.setdefault(g[4:6], []).append(g)
            for groups in by_venue.values():
                for g in groups[-6:]:  # 場ごとに直近 6 開催日だけ日付を読む（特別登録だけが先に出た未来の日も含む）
                    d = self._date_of_group(g, year)
                    if d and lo <= d <= hi:
                        by_date.setdefault(d, []).append(g)
        self._by_date = by_date
        return sorted(by_date)

    def matrix(self, date: str) -> dict | None:
        """中央競馬は 1 開催日 1 場 12 レース。出馬表が一部しか出ていない日（特別登録だけ先に出る）を『足りない』と判定できるよう、12 レース分を期待する。"""
        groups = self._by_date.get(date)
        if groups is None:
            return None
        year = date[:4]
        race_ids = sorted(f"{g}{n:02d}" for g in groups for n in range(1, RACES_PER_DAY + 1))
        grid = {r: {c: r in self._race_ids(c, year) for c in CATS} for r in race_ids}
        meta = {r: {"venue": VENUES.get(r[4:6], r[4:6]), "race_num": int(r[10:12])} for r in race_ids}
        return {"race_ids": race_ids, "race_meta": meta, "matrix": grid}

    def events(self, date: str) -> list[dict]:
        raw = self.io.get_bytes(f"{INF}/events/{date}.jsonl")
        rows: list[dict] = []
        for ln in (raw or b"").decode("utf-8", "replace").splitlines():
            try:
                rows.append(json.loads(ln))
            except ValueError:
                continue
        return rows

    def pred_files(self, engine: str, date: str) -> set[str]:
        pre = f"{INF}/predictions_{engine}/{date}/"
        return {k[len(pre):-8] for k in self.io.list(pre) if k.endswith(".parquet")}

    def results_summary(self, date: str) -> dict | None:
        raw = self.io.get_bytes(f"ops/prediction_results/{date}.json")
        if not raw:
            return None
        try:
            d = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        return {"status": d.get("status"), "built_at": d.get("built_at"), "models": d.get("models")}


# ───────────────────────── 判定 ─────────────────────────

def _d0(date: str) -> datetime:
    return datetime(int(date[:4]), int(date[4:6]), int(date[6:8]), tzinfo=JST)


def _stage(state: str, done: int | None, total: int | None, detail: str = "") -> dict:
    return {"state": state, "done": done, "total": total, "detail": detail}


def _judge(key: str, date: str, now: datetime, done: int, total: int, detail: str = "") -> dict:
    """件数と期限から状態を決める。全部そろえば ok。足りないとき、期限前は待ち・期限後は要対応（9 割以上は警告）。"""
    if total and done >= total:
        return _stage("ok", done, total, detail)
    late = now >= _d0(date) + timedelta(hours=DEADLINE_H[key])
    if not late:
        return _stage("wait", done, total, detail or "期限前")
    return _stage("warn" if total and done >= total * 0.9 else "ng", done, total, detail or "期限を過ぎても未完了")


def day_health(date: str, now: datetime, src: Sources) -> dict:
    mx = src.matrix(date)
    race_ids = (mx or {}).get("race_ids") or []
    meta, grid = (mx or {}).get("race_meta") or {}, (mx or {}).get("matrix") or {}
    total = len(race_ids)
    stages: dict[str, dict] = {}
    notes: list[str] = []
    races: list[dict] = []
    if mx is not None and not race_ids:
        for k, _, _ in STAGES:
            stages[k] = _stage("wait", None, None, "レース一覧がまだありません")
        notes.append("この日のレース一覧がまだありません")
    elif mx is None:
        for k, _, _ in STAGES:
            stages[k] = _stage("na", None, None, "取得状況を読めません")
        notes.append("GCS から取得状況を読めませんでした")
    else:
        for k, _, cat in STAGES:
            if cat:
                n = sum(1 for r in race_ids if (grid.get(r) or {}).get(cat))
                stages[k] = _judge(k, date, now, n, total)
        events = {e["race_id"]: e for e in src.events(date) if e.get("race_id")}
        files = {eng: src.pred_files(eng, date) for eng in ENGINES}
        primary = next((eng for eng in ENGINES if files[eng]), "v6")
        eng_ok = {eng: sum(1 for r in race_ids if (events.get(r) or {}).get(f"{eng}_status") == "ok") for eng in ENGINES}
        fail = [(r, e) for r, e in events.items() if r in race_ids and any((e.get(f"{eng}_status") not in (None, "ok")) for eng in ENGINES)]
        have = sum(1 for r in race_ids if r in files[primary])
        n_ev = sum(1 for r in race_ids if r in events)
        det = f"{primary} {have}/{total} 本" + (" ・ok: " + " / ".join(f"{eng} {eng_ok[eng]}" for eng in ENGINES) if n_ev >= total else f"（実行記録 {n_ev}/{total} 件）")
        st = _judge("predict", date, now, have, total, det)
        if fail and st["state"] in ("ok", "wait"):
            st = _stage("warn", have, total, det + f"・エラー {len(fail)} レース")
        elif st["state"] != "ok":
            st["detail"] = det
        stages["predict"] = st
        no_event = [r for r in race_ids if r in files[primary] and r not in events]
        if no_event and events:
            notes.append(f"予測ファイルはあるが実行記録（events）がないレースが {len(no_event)} 件（記録の反映が遅れているか、バックフィル由来）")
        for r, e in fail[:5]:
            errs = [f"{eng}: {e.get(f'{eng}_error') or e.get(f'{eng}_status')}" for eng in ENGINES if e.get(f"{eng}_status") not in (None, "ok")]
            notes.append(f"{meta.get(r, {}).get('venue', '')}{meta.get(r, {}).get('race_num', '')}R（{r}）の予測エラー: " + "; ".join(errs)[:160])
        late_res = now >= _d0(date) + timedelta(hours=DEADLINE_H["results"])
        rs = src.results_summary(date)
        if rs and rs.get("status") == "final":
            stages["results"] = _stage("ok", total, total, f"確定 {(rs.get('built_at') or '')[:16]}")
        else:
            why = f"{rs.get('status')}（{(rs.get('built_at') or '')[:16]}）" if rs else "集計ファイルがありません"
            stages["results"] = _stage("ng" if late_res else "wait", None, None, why)
        if stages["final"]["state"] == "wait" and stages["on_time"]["state"] == "ok":
            stages["final"]["detail"] = "確定結果の公開待ち（開催の翌々日の夕方ごろ）"
        for r in race_ids:
            row, ev = grid.get(r) or {}, events.get(r) or {}
            races.append({"race_id": r, "venue": meta.get(r, {}).get("venue"), "round": meta.get(r, {}).get("race_num"),
                          "eve": bool(row.get("race_shutuba")), "odds": bool(row.get("race_odds")), "predict": r in files[primary],
                          "pred_status": {eng: ev.get(f"{eng}_status") for eng in ENGINES}, "pred_sec": ev.get(f"{primary}_sec"), "pred_at": ev.get("at"),
                          "on_time": bool(row.get("race_result_on_time")), "index": bool(row.get("race_index")), "final": bool(row.get("race_result"))})
        races.sort(key=lambda x: (x["venue"] or "", x["round"] or 0))
    stages = {k: stages[k] for k, _, _ in STAGES if k in stages}  # 表示順（出馬表 → … → 予測成績）
    worst = max((LEVEL_RANK[s["state"]] for s in stages.values()), default=0)
    level = {0: "ok", 1: "wait", 2: "warn", 3: "ng"}[worst]
    headline = {"ok": "最後まで完了", "wait": "進行中（期限前）", "warn": "一部に不足あり", "ng": "要対応（期限後に未完了）"}[level]
    if mx is None:
        level, headline = "na", "取得状況を読めません"
    return {"date": date, "weekday": "月火水木金土日"[_d0(date).weekday()], "races": total, "level": level, "headline": headline, "stages": stages,
            "notes": notes, "race_rows": races}


def weekend_gaps(days: list[str], now: datetime) -> list[dict]:
    """直近・直後の土日で、出馬表が見つからない日（開催がない週末なら誤報なので警告止まり）。前日 20:00 を過ぎたものだけ。"""
    out = []
    for back in range(-3, 4):
        d = now + timedelta(days=back)
        if d.weekday() not in (5, 6):
            continue
        key = d.strftime("%Y%m%d")
        if key in days or now < _d0(key) - timedelta(hours=4):
            continue
        out.append({"date": key, "weekday": "月火水木金土日"[d.weekday()], "message": "この日の出馬表が GCS に見つかりません（開催がない週末なら問題ありません）"})
    return out


def build(src: Sources, now: datetime | None = None, n_days: int = 4) -> dict:
    now = now or datetime.now(JST)
    days = sorted(src.race_days(now))
    today = now.strftime("%Y%m%d")
    past = [d for d in days if d <= today][-n_days:]
    nxt = next((d for d in days if d > today), None)
    out = [day_health(d, now, src) for d in reversed(past)]
    nxt_h = day_health(nxt, now, src) if nxt else None
    latest_done = next((d for d in out if d["date"] < today), out[0] if out else None)
    return {"built_at": now.isoformat(timespec="seconds"), "days": out, "next": nxt_h, "today": today, "gaps": weekend_gaps(days, now),
            "overall": (latest_done or {}).get("level") or "na", "latest_date": (latest_done or {}).get("date")}


def summary(d: dict | None) -> dict:
    """ホームのカード用。最新の（今日より前の）開催日の判定と、次の開催日。"""
    if not d or not d.get("days"):
        return {"status": "not_collected", "gaps": (d or {}).get("gaps", [])}
    latest = next((x for x in d["days"] if x["date"] < d["today"]), d["days"][0])
    stg = {k: v["state"] for k, v in latest["stages"].items()}
    return {"status": "ok", "date": latest["date"], "weekday": latest["weekday"], "level": latest["level"], "headline": latest["headline"], "races": latest["races"],
            "stages": stg, "problem": [k for k, v in stg.items() if v in ("ng", "warn")], "notes": latest["notes"][:2], "gaps": d.get("gaps", []),
            "next": ({"date": d["next"]["date"], "level": d["next"]["level"], "eve": d["next"]["stages"].get("eve", {}).get("state")} if d.get("next") else None)}
