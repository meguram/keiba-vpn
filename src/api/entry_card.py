"""出馬表の『馬・騎手・調教師』をタップしたときに出す情報（1回の取得でまとめて返す）。

予測に依らない実データの事実だけを返す:
  馬   … 近5走（馬柱 race_shutuba_past）、通算・父母父（horse_result）、同条件の成績、前走からの間隔・距離・騎手の変更、最終追い切り（race_oikiri）、厩舎コメント
  騎手 … 今回の競馬場・芝ダ・距離帯・クラスでの成績、直近 90/365 日、直近 10 走、通算、この馬への騎乗歴、プロフィール
  調教師 … 同上（騎乗歴の代わりに、この馬の厩舎コメント）

騎手・調教師の成績は、ホストの特徴量（data/local/features/{jockey,trainer}_tbl/lookup_*.parquet）の集計済みの値を読む
（タップごとに全レースを集計しない）。これは集計時点までの通算で、レース当日より後の結果を含むことがあるため、
集計時点（as_of）を必ず添える。馬の近走・同条件の成績は、レース当日より前のものだけを使う（先読みなし）。
キーは馬名ではなく horse_id / jockey_id / trainer_id。取れない項目は status="no_data" で返す（空欄にしない）。
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
VENUE_JA = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京", "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
_ID = re.compile(r"^[0-9A-Za-z]{1,12}$")
_TTL_SEC = 300.0
_NEG_TTL_SEC = 60.0


def features_dir() -> Path:
    return Path(os.environ.get("KEIBA_FEATURES_DIR") or ROOT / "data" / "local" / "features")


def person_dir() -> Path:
    try:
        from src.config.data_paths import page_reference_root

        return page_reference_root() / "meta" / "person"
    except Exception:
        return ROOT / "data" / "page_reference" / "meta" / "person"


# ── 小さな TTL キャッシュ（ストレージの読み込みをタップごとにやり直さない）──────────────

_cache: dict[tuple, tuple[float, Any]] = {}
_lock = threading.Lock()


def _cached(key: tuple, fn, ttl: float = _TTL_SEC, neg_ttl: float = _NEG_TTL_SEC):
    now = time.time()
    with _lock:
        hit = _cache.get(key)
    if hit and hit[0] > now:
        return hit[1]
    try:
        val = fn()
    except Exception:
        val = None
    with _lock:
        _cache[key] = (now + (ttl if val is not None else neg_ttl), val)
        if len(_cache) > 4000:
            for k in [k for k, v in _cache.items() if v[0] <= now][:2000]:
                _cache.pop(k, None)
    return val


def clear_cache() -> None:
    with _lock:
        _cache.clear()


from src.utils import cache_epoch as _cache_epoch  # noqa: E402

_cache_epoch.register("entry_card", clear_cache)


def _load(storage: Any, cat: str, key: str) -> dict | None:
    return _cached(("load", id(storage), cat, key), lambda: storage.load(cat, key))


# ── 日付・数値の整形 ─────────────────────────────────────────────────────────

def ymd(s: Any) -> str:
    """'2026/10/03' '2026.10.03' '2026-10-03' '20261003' → '20261003'。分からなければ ''。"""
    d = re.sub(r"\D", "", str(s or ""))[:8]
    return d if len(d) == 8 else ""


def _days_between(a: str, b: str) -> int | None:
    try:
        return (date(int(b[:4]), int(b[4:6]), int(b[6:8])) - date(int(a[:4]), int(a[4:6]), int(a[6:8]))).days
    except Exception:
        return None


def _num(x: Any) -> float | None:
    try:
        v = float(x)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def _rate(n: float | None, d: float | None) -> float | None:
    return round(n / d * 100, 1) if (n is not None and d) else None


# ── 集計区分のキー（jockey_trainer_stats.py と同じ定義）──────────────────────────

def surface_coarse(surface: str | None) -> str:
    s = surface or ""
    if "芝" in s or "草" in s:
        return "turf"
    if "ダ" in s or "障" in s:
        return "dirt_ob"
    return "other"


def distance_band_key(distance: Any, surface: str | None) -> str:
    d = _num(distance)
    sc = surface_coarse(surface)
    if d is None:
        return f"{sc}_unknown"
    b = "S_<1600" if d < 1600 else "M_1600_1999" if d < 2000 else "I_2000_2399" if d < 2400 else "L_ge2400"
    return f"{sc}_{b}"


def grade_bucket_key(grade: str | None, race_class: str | None) -> str:
    g = (grade or "").upper()
    rc = race_class or ""
    if "G1" in g:
        return "G1"
    if "G2" in g:
        return "G2"
    if "G3" in g:
        return "G3"
    if "L" in g or "Ｌ" in g:
        return "L"
    if "OP" in g or "オープン" in g or "公開" in g:
        return "OP"
    if re.search(r"勝クラ|3勝|2勝|1勝|未勝|新馬", rc) or re.search(r"^(3勝|2勝|1勝|未勝利|新馬)$", grade or ""):
        return "class_nov_cond"
    return "other"


_GRADE_JA = {"G1": "G1", "G2": "G2", "G3": "G3", "L": "リステッド", "OP": "オープン", "class_nov_cond": "条件戦（未勝利〜3勝）", "other": "その他"}
_BAND_JA = {"S_<1600": "短距離（〜1599m）", "M_1600_1999": "マイル〜（1600〜1999m）", "I_2000_2399": "中距離（2000〜2399m）", "L_ge2400": "長距離（2400m〜）"}


def _band_label(key: str) -> str:
    sc, _, rest = key.partition("_")
    if key.startswith("dirt_ob_"):
        sc, rest = "dirt_ob", key[len("dirt_ob_"):]
    return f"{'芝' if sc == 'turf' else 'ダート・障害'} {_BAND_JA.get(rest, rest)}"


# ── 騎手・調教師の集計（特徴量の lookup）────────────────────────────────────

_lookup_cache: dict[str, tuple[tuple, dict | None]] = {}


def _read_parquet(path: Path):
    try:
        import polars as pl

        return pl.read_parquet(path).to_dicts()
    except Exception:
        try:
            import pandas as pd

            return pd.read_parquet(path).to_dict("records")
        except Exception:
            return None


def _lookup(kind: str) -> dict | None:
    """{'entity': {id: row}, 'slices': {(id, slice_kind, slice_key): row}, 'as_of': 'YYYY-MM-DD'}。ファイルが無ければ None。"""
    base = features_dir() / f"{kind}_tbl"
    f1, f2 = base / f"lookup_{kind}.parquet", base / f"lookup_{kind}_slices.parquet"
    if not (f1.is_file() and f2.is_file()):
        return None
    sig = (f1.stat().st_mtime, f2.stat().st_mtime)
    hit = _lookup_cache.get(kind)
    if hit and hit[0] == sig:
        return hit[1]
    idc = f"{kind}_id"
    rows, srows = _read_parquet(f1), _read_parquet(f2)
    if rows is None or srows is None:
        return None
    as_of = ""
    try:
        gen = json.loads((features_dir() / "jockey_trainer_stats" / "_manifest.json").read_text(encoding="utf-8")).get("generated_at", "")
        as_of = gen[:10]
    except Exception:
        pass
    out = {
        "entity": {str(r[idc]): r for r in rows},
        "slices": {(str(r[idc]), r["slice_kind"], str(r["slice_key"])): r for r in srows},
        "as_of": as_of,
    }
    _lookup_cache[kind] = (sig, out)
    return out


def _stat(starts: Any, wins: Any, top3: Any, avg: Any = None) -> dict:
    s, w, t = _num(starts), _num(wins), _num(top3)
    return {"starts": int(s) if s is not None else None, "wins": int(w) if w is not None else None, "top3": int(t) if t is not None else None,
            "win_rate": _rate(w, s), "top3_rate": _rate(t, s), "avg_finish": round(_num(avg), 1) if _num(avg) is not None else None}


def _person_profile(kind: str, pid: str) -> dict | None:
    if not _ID.match(pid or ""):
        return None
    p = person_dir() / f"{kind}_{pid}.json"
    d = _cached(("person", str(p)), lambda: json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None, ttl=3600)
    if not d:
        return None
    ct = d.get("career_total") or {}
    return {
        "name": d.get("name"), "affiliation": d.get("affiliation"), "debut_year": d.get("debut_year"),
        "career": {"rides": ct.get("rides"), "wins": ct.get("wins"), "win_rate": ct.get("win_rate"), "top2_rate": ct.get("top2_rate"), "top3_rate": ct.get("top3_rate")} if ct else None,
        "this_year_text": (d.get("current_year_wins_text") or "").replace("\n", " ").replace("\t", "") or None,
        "g1_wins_text": d.get("g1_wins_text"),
        "graded_wins_text": d.get("graded_wins_text"),
    }


def _entity_card(kind: str, pid: str, name: str | None, ctx: dict) -> dict:
    """騎手(kind='jockey') / 調教師(kind='trainer') の成績カード。"""
    pre = "jk" if kind == "jockey" else "tr"
    lk = _lookup(kind)
    prof = _person_profile(kind, pid)
    card: dict[str, Any] = {"status": "no_data", "id": pid, "name": (prof or {}).get("name") or name, "profile": prof, "conditions": [], "recent": {}, "career": None, "as_of": None}
    if not pid or lk is None:
        return card
    row = lk["entity"].get(pid)
    sl = lk["slices"]
    conds = []
    for label, skind, skey in (
        (f"この競馬場（{VENUE_JA.get(ctx['venue_code'], ctx['venue_code'])}）", "venue_code", ctx["venue_code"]),
        (f"この馬場（{'芝' if ctx['surface_c'] == 'turf' else 'ダート・障害'}）", "surface_coarse", ctx["surface_c"]),
        (f"この距離（{_band_label(ctx['band'])}）", "distance_band", ctx["band"]),
        (f"このクラス（{_GRADE_JA.get(ctx['grade_b'], ctx['grade_b'])}）", "grade_bucket", ctx["grade_b"]),
    ):
        r = sl.get((pid, skind, skey))
        conds.append({"label": label, **(_stat(r["starts"], r["wins"], r["top3"], r.get("avg_finish")) if r else _stat(None, None, None)), "has_data": bool(r)})
    card["conditions"] = conds
    if row:
        for key, lab in (("cal90", "直近90日"), ("cal365", "直近365日"), ("roll10", "直近10走")):
            card["recent"][key] = {"label": lab, **_stat(row.get(f"{pre}_{key}_starts"), row.get(f"{pre}_{key}_wins"), row.get(f"{pre}_{key}_top3"))}
        card["career"] = {"label": "2020年以降の通算", **_stat(row.get(f"{pre}_prior_all_starts"), row.get(f"{pre}_prior_all_wins"), row.get(f"{pre}_prior_all_top3"), row.get(f"{pre}_prior_all_avg_finish"))}
    if row or any(c["has_data"] for c in conds):
        card["status"] = "ok"
        card["as_of"] = lk["as_of"] or None
        # 集計時点が出走日より後なら、その日以降の結果を含むことを明示（先読みの注意）
        card["includes_after_race"] = bool(lk["as_of"] and ctx["date"] and ymd(lk["as_of"]) > ctx["date"])
    return card


# ── 馬柱の1行（raw）の解釈 ──────────────────────────────────────────────────

_PASS = re.compile(r"(?<![\d.])(\d{1,2}(?:-\d{1,2})+)(?![\d.])")


def parse_past_run(p: dict) -> dict:
    """race_shutuba_past の past_races 1件 → 表示用の1走。raw（netkeiba の馬柱の1行）から取れる項目を取る。"""
    raw = str(p.get("raw") or "")
    toks = raw.split()
    out = {
        "date": ymd(p.get("date")), "race_id": p.get("race_id"), "surface": p.get("surface"), "distance": p.get("distance"),
        "time": p.get("finish_time") or None, "finish": p.get("finish_position"),
        "venue": toks[1] if len(toks) > 1 else None, "race_name": None, "going": None, "field": None, "post": None, "popularity": None,
        "jockey": None, "jockey_weight": None, "passing": None, "last3f": None, "weight": None, "weight_change": None, "margin": None,
    }
    m = re.search(r"^\S+\s+\S+\s+\d+\s+(.+?)\s+[芝ダ障]\d+", raw)
    if m:
        out["race_name"] = m.group(1)
    for k, pat in (("going", r"\s(良|稍重|重|不良)\s"), ("field", r"(\d+)頭"), ("post", r"(\d+)番"), ("popularity", r"(\d+)人")):
        mm = re.search(pat, raw)
        if mm:
            out[k] = mm.group(1) if k == "going" else int(mm.group(1))
    mm = re.search(r"\d+人\s+(\S+)\s+(\d+(?:\.\d)?)", raw)
    if mm:
        out["jockey"], out["jockey_weight"] = mm.group(1), _num(mm.group(2))
    mm = _PASS.search(raw)
    if mm:
        out["passing"] = mm.group(1)
    mm = re.search(r"\((\d{2}\.\d)\)", raw)
    if mm:
        out["last3f"] = _num(mm.group(1))
    mm = re.search(r"\b(\d{3})\(([+-]?\d+)\)", raw)
    if mm:
        out["weight"], out["weight_change"] = int(mm.group(1)), int(mm.group(2))
    mm = re.findall(r"\((-?\d+\.\d)\)", raw)
    if mm:
        out["margin"] = _num(mm[-1])
    return out


def _history_run(h: dict) -> dict:
    """horse_result.race_history の1件 → 同じ形（馬柱が無いときの代替）。"""
    v = re.sub(r"^\d+", "", str(h.get("venue") or ""))
    v = re.sub(r"\d+$", "", v)
    return {
        "date": ymd(h.get("date")), "race_id": h.get("race_id"), "surface": h.get("surface"), "distance": h.get("distance"), "time": h.get("finish_time"),
        "finish": h.get("finish_position"), "venue": v or None, "race_name": h.get("race_name"), "going": h.get("track_condition"), "field": h.get("field_size"),
        "post": h.get("horse_number"), "popularity": h.get("popularity"), "jockey": h.get("jockey_name"), "jockey_weight": h.get("jockey_weight"),
        "passing": None, "last3f": None, "weight": h.get("weight"), "weight_change": h.get("weight_change"), "margin": None,
    }


def _top3(f: Any) -> bool:
    n = _num(f)
    return n is not None and 1 <= n <= 3


def _same_cond(runs: list[dict], ctx: dict) -> list[dict]:
    """過去の全走（レース当日より前）から、今回と同じ条件での成績。"""
    def agg(label: str, pred) -> dict:
        sel = [r for r in runs if pred(r)]
        wins = sum(1 for r in sel if _num(r.get("finish")) == 1)
        t3 = sum(1 for r in sel if _top3(r.get("finish")))
        return {"label": label, "starts": len(sel), "wins": wins, "top3": t3, "win_rate": _rate(wins, len(sel)), "top3_rate": _rate(t3, len(sel))}

    sc = surface_coarse(ctx["surface"])
    band = distance_band_key(ctx["distance"], ctx["surface"])
    venue = ctx["venue"]
    return [
        agg(f"この競馬場（{venue}）", lambda r: bool(venue) and r.get("venue") == venue),
        agg(f"この馬場（{'芝' if sc == 'turf' else 'ダート・障害'}）", lambda r: surface_coarse(r.get("surface")) == sc),
        agg(f"この距離（{_band_label(band)}）", lambda r: distance_band_key(r.get("distance"), r.get("surface")) == band),
    ]


def _entry_by_horse(rows: list[dict] | None, horse_id: str) -> dict | None:
    for r in rows or []:
        if isinstance(r, dict) and str(r.get("horse_id")) == horse_id:
            return r
    return None


def _past_jockey_id(storage: Any, race_id: str | None, horse_id: str) -> tuple[str | None, str | None]:
    """過去のレースでのこの馬の (jockey_id, jockey_name)。出馬表（無ければ結果）から引く。"""
    if not race_id:
        return None, None
    for cat, key in (("race_shutuba", "entries"), ("race_result", "results")):
        c = _load(storage, cat, race_id)
        e = _entry_by_horse((c or {}).get(key) or (c or {}).get("entries"), horse_id)
        if e and e.get("jockey_id"):
            return str(e["jockey_id"]), e.get("jockey_name")
    return None, None


# ── 本体 ───────────────────────────────────────────────────────────────────

def build_entry_card(race_id: str, horse_id: str, storage: Any) -> dict:
    if not (_ID.match(race_id or "") and _ID.match(horse_id or "")):
        return {"status": "bad_request"}
    shutuba = _load(storage, "race_shutuba", race_id)
    entry = _entry_by_horse((shutuba or {}).get("entries"), horse_id)
    if not shutuba or not entry:
        return {"status": "not_found", "race_id": race_id, "horse_id": horse_id}
    rdate = ymd(shutuba.get("date"))
    surface = shutuba.get("surface") or ""
    ctx = {
        "date": rdate, "venue": shutuba.get("venue") or VENUE_JA.get(race_id[4:6], ""), "venue_code": race_id[4:6], "surface": surface, "surface_c": surface_coarse(surface),
        "distance": shutuba.get("distance"), "band": distance_band_key(shutuba.get("distance"), surface),
        "grade_b": grade_bucket_key(shutuba.get("grade"), shutuba.get("race_class")),
    }
    jid, tid = str(entry.get("jockey_id") or ""), str(entry.get("trainer_id") or "")

    # 馬: 近5走（馬柱）・全履歴（horse_result）
    past_doc = _load(storage, "race_shutuba_past", race_id)
    prow = _entry_by_horse((past_doc or {}).get("entries"), horse_id)
    hist = _load(storage, "horse_result", horse_id) or {}
    all_runs = [_history_run(h) for h in (hist.get("race_history") or []) if isinstance(h, dict)]
    all_runs = [r for r in all_runs if r["date"] and (not rdate or r["date"] < rdate)]            # レース当日より前だけ（先読みなし）
    recent_src = [parse_past_run(p) for p in ((prow or {}).get("past_races") or []) if isinstance(p, dict)]
    recent = [r for r in recent_src if r["date"] and (not rdate or r["date"] < rdate)] or sorted(all_runs, key=lambda r: r["date"], reverse=True)
    recent = sorted(recent, key=lambda r: r["date"], reverse=True)[:5]

    # 前走からの間隔・距離の増減・騎手の乗り替わり
    last = recent[0] if recent else None
    layoff = _days_between(last["date"], rdate) if (last and rdate) else None
    dist_chg = (int(ctx["distance"]) - int(last["distance"])) if (last and _num(last.get("distance")) and _num(ctx["distance"])) else None
    # 近5走それぞれの騎手 id を引いて、乗り替わり・この騎手との騎乗歴を判定する
    with ThreadPoolExecutor(max_workers=5) as ex:
        past_j = list(ex.map(lambda r: _past_jockey_id(storage, r.get("race_id"), horse_id), recent))
    for r, (pj, _pn) in zip(recent, past_j):
        r["jockey_id"] = pj
    prev_jid, prev_jname = (past_j[0] if past_j else (None, None))
    changed = None if not (prev_jid and jid) else prev_jid != jid
    rides = [(r, pj) for r, (pj, _) in zip(recent, past_j) if pj and pj == jid]

    oik = _entry_by_horse((_load(storage, "race_oikiri", race_id) or {}).get("entries"), horse_id)
    com = _entry_by_horse((_load(storage, "race_trainer_comment", race_id) or {}).get("entries"), horse_id)

    from src.pipeline.inference.horse_growth import growth_for_horse

    growth = _cached(("growth", horse_id, rdate), lambda: growth_for_horse(horse_id, all_runs, hist.get("birthday"))) if all_runs else None

    horse = {
        "status": "ok" if (recent or hist or oik) else "no_data",
        "id": horse_id, "name": entry.get("horse_name"), "sex_age": entry.get("sex_age"), "weight": entry.get("weight") or None, "weight_change": entry.get("weight_change") if entry.get("weight") else None,
        "profile": {
            "sire": hist.get("sire") or entry.get("sire") or None, "dam": hist.get("dam") or None, "dam_sire": hist.get("dam_sire") or entry.get("dam_sire") or None,
            "owner": hist.get("owner") or None, "breeder": hist.get("breeder") or None, "birthday": hist.get("birthday") or None,
            "career": hist.get("career") or None, "earnings": hist.get("total_earnings"),
        } if hist else None,
        "recent": recent,
        "same_conditions": _same_cond(all_runs, {**ctx, "venue": ctx["venue"]}) if all_runs else [],
        "changes": {"layoff_days": layoff, "distance_change": dist_chg, "jockey_changed": changed, "previous_jockey": prev_jname,
                    "last_run": {"date": last["date"], "race_name": last.get("race_name"), "finish": last.get("finish")} if last else None},
        "training": ({k: oik.get(k) or None for k in ("training_date", "course", "condition", "rider", "lap_times", "impression", "evaluation", "comment")} if oik else None),
        "trainer_comment": ({"comment": com.get("comment"), "evaluation": com.get("evaluation") or None, "trainer": com.get("trainer_name")} if com and com.get("comment") else None),
        "weights": [{"date": r["date"], "weight": r["weight"], "change": r["weight_change"]} for r in recent if r.get("weight")],
        "growth": growth,
    }

    jockey = _entity_card("jockey", jid, entry.get("jockey_name"), ctx)
    jockey["with_horse"] = {
        "rides_in_last5": len(rides), "wins": sum(1 for r, _ in rides if _num(r.get("finish")) == 1), "top3": sum(1 for r, _ in rides if _top3(r.get("finish"))),
        "checked": len(recent), "changed": changed, "previous_jockey": prev_jname,
    } if recent else None
    jockey["weight_today"] = entry.get("jockey_weight")
    trainer = _entity_card("trainer", tid, entry.get("trainer_name"), ctx)
    trainer["comment"] = horse["trainer_comment"]

    return {
        "status": "ok", "race_id": race_id, "horse_id": horse_id, "as_of": rdate or None,
        "race": {"name": shutuba.get("race_name"), "venue": ctx["venue"], "surface": surface, "distance": ctx["distance"], "grade": shutuba.get("grade")},
        "horse": horse, "jockey": jockey, "trainer": trainer,
        "sources": ["race_shutuba", "race_shutuba_past", "horse_result", "megu_index_flat", "race_oikiri", "race_trainer_comment", "jockey_tbl/trainer_tbl lookup", "person meta"],
    }
