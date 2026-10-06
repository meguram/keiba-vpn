"""dev 用の最小モックデータを生成する（GCS の代わりにローカルを読むための元データ）。

生成物（すべて架空。実在の馬・騎手・レースではない）:
  * 開催日 2 日分（直近の過去日曜＝結果あり / 次の土曜＝結果なし）× 2 場 × 4 レース × 8 頭
  * **スキーマ定義（schema_defs.json）の全カテゴリ**のサンプルをスキーマに適合する形で生成する
    - 元データ: race_shutuba / race_odds / race_index / race_pair_odds / race_paddock / race_barometer / race_oikiri /
      race_trainer_comment / race_shutuba_past / race_detail / race_result / race_result_on_time / race_result_lap / smartrc_race
    - 馬: horse_result / horse_pedigree_5gen / horse_training（出走馬 64 頭）、broodmare_mating（母馬）
    - 派生カテゴリ（11 種）は本番と同じ抽出ロジック（row_data_extractor）で元データから生成
    - requirement_row_trace / race_lists / race_day_schedule（local_only。data/page_reference/ 配下）
  * スキーマ未定義の生成物も形式は暫定で用意する: race_predictions / tracking_difficulty / final_odds_prediction /
    finish_order_prediction / race_performance / jra_cushion

書き込み先:
  * GCS 相当カテゴリ … ``data/dev_mock/``（``KEIBA_DEV_MOCK_DIR`` で変更）
  * local_only カテゴリ … ``data/page_reference/``（既存の実データは上書きしない）

実行: ``python -m src.scripts.data.make_dev_mock [--today YYYY-MM-DD] [--clean]``（``make dev-mock``）
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.scraper import schemas
from src.scraper.dev_store import DevStore, dev_mock_root
from src.scraper.storage import HybridStorage

JST = timezone(timedelta(hours=9))
SEED = 20261004

VENUES = [("05", "東京"), ("09", "阪神")]
RACE_SPECS: dict[int, dict[str, Any]] = {
    1: {"name": "3歳以上未勝利", "surface": "ダ", "distance": 1400, "grade": "", "start": "10:05"},
    5: {"name": "3歳以上1勝クラス", "surface": "芝", "distance": 1600, "grade": "", "start": "12:25"},
    9: {"name": "3歳以上2勝クラス", "surface": "芝", "distance": 2000, "grade": "", "start": "14:20"},
    11: {"name": "モックステークス", "surface": "芝", "distance": 1800, "grade": "G3", "start": "15:45"},
}
N_HORSES = 64
FIELD = 8
JOCKEYS = [(f"9{i:04d}", f"モック騎手{c}") for i, c in enumerate("アイウエオカキク", 1)]
TRAINERS = [(f"91{i:03d}", f"モック調教師{c}") for i, c in enumerate("アイウエオカキク", 1)]
SIRES = 24
DAMS = 24


def _fmt_time(sec: float) -> str:
    return f"{int(sec // 60)}:{sec % 60:04.1f}"


def _margin_text(gap_sec: float) -> str:
    lengths = gap_sec / 0.17
    table = [(0.15, "ハナ"), (0.4, "クビ"), (0.75, "1/2"), (1.25, "1"), (1.75, "1 1/2"),
             (2.25, "2"), (3.0, "2 1/2"), (4.0, "3 1/2"), (6.0, "5")]
    for lim, txt in table:
        if lengths < lim:
            return txt
    return "大差"


def make_horses(rng: random.Random) -> list[dict[str, Any]]:
    horses = []
    for i in range(N_HORSES):
        yr = 2021 + i % 3
        jk = rng.choice(JOCKEYS)
        tr = TRAINERS[i % len(TRAINERS)]
        horses.append({
            "horse_id": f"{yr}10{i + 1:04d}",
            "horse_name": f"モックウマ{i + 1:02d}",
            "name_en": f"Mock Horse {i + 1:02d}",
            "birth_year": yr,
            "sex": rng.choice(["牡", "牡", "牝", "セ"]),
            "ability": rng.gauss(0, 1),
            "style": rng.choice(["逃", "先", "差", "追"]),
            "trainer": tr,
            "jockey": jk,
            "seed": rng.randrange(1 << 30),
        })
    return horses


def _anc(h: dict[str, Any], g: int, p: int) -> tuple[str, str, str]:
    r = random.Random(h["seed"] * 1000 + g * 100 + p)
    if p % 2 == 0:
        idx = r.randrange(SIRES)
        return f"1990{idx:06d}", f"モックサイアー{idx:02d}", "牡"
    idx = r.randrange(DAMS)
    return f"1991{idx:06d}", f"モックダム{idx:02d}", "牝"


def pedigree(h: dict[str, Any]) -> dict[str, Any]:
    ancestors = []
    for g in range(1, 6):
        for p in range(2 ** g):
            hid, name, sex = _anc(h, g, p)
            ancestors.append({"generation": g, "position": p, "name": name, "horse_id": hid, "sex": sex})
    sire, dam, dam_sire = ancestors[0]["name"], ancestors[1]["name"], next(
        a["name"] for a in ancestors if a["generation"] == 2 and a["position"] == 2)
    h["sire"], h["dam"], h["dam_sire"] = sire, dam, dam_sire
    return {"horse_id": h["horse_id"], "sex": h["sex"], "sire": sire, "dam": dam, "dam_sire": dam_sire,
            "ancestors": ancestors, "ancestor_count": len(ancestors), "source": "dev_mock"}


def _softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es)
    return [e / s for e in es]


def build_race(rng: random.Random, rid: str, day: date, venue: str, rr: int, field: list[dict[str, Any]],
               with_result: bool) -> dict[str, Any]:
    spec = RACE_SPECS[rr]
    n = len(field)
    probs = _softmax([h["ability"] for h in field])
    odds = [max(1.1, round(0.8 / p, 1)) for p in probs]
    order_pop = sorted(range(n), key=lambda i: odds[i])
    pop = {i: order_pop.index(i) + 1 for i in range(n)}
    weights = [rng.choice([54.0, 55.0, 56.0, 57.0, 58.0]) for _ in range(n)]
    body = [rng.randint(440, 520) for _ in range(n)]
    wchg = [rng.randint(-6, 6) for _ in range(n)]
    meta = {
        "race_id": rid, "race_name": spec["name"], "surface": spec["surface"], "distance": spec["distance"],
        "direction": "右", "weather": "晴",
        "track_condition": "良", "start_time": spec["start"], "venue": venue, "field_size": n,
        "date": day.isoformat(), "round": rr, "grade": spec["grade"], "race_class": spec["name"],
        "weight_rule": "馬齢", "course_type": "",
    }
    entries = []
    for i, h in enumerate(field):
        entries.append({
            "horse_number": i + 1, "bracket_number": i + 1, "horse_name": h["horse_name"],
            "horse_id": h["horse_id"], "sex_age": f"{h['sex']}{day.year - h['birth_year']}",
            "jockey_weight": weights[i], "jockey_name": h["jockey"][1], "jockey_id": h["jockey"][0],
            "trainer_name": h["trainer"][1], "trainer_id": h["trainer"][0], "weight": body[i],
            "weight_change": wchg[i], "odds": odds[i], "popularity": pop[i],
            "sire": h["sire"], "dam_sire": h["dam_sire"],
        })
    out: dict[str, Any] = {
        "shutuba": {**meta, "entries": entries},
        "odds": {"race_id": rid, "entries": [
            {"horse_number": i + 1, "win_odds": odds[i], "place_odds_min": round(max(1.0, odds[i] / 3.2), 1),
             "place_odds_max": round(max(1.1, odds[i] / 2.2), 1), "popularity": pop[i]} for i in range(n)]},
        "index": {"race_id": rid, "entries": [
            {"horse_number": i + 1, "horse_name": h["horse_name"], "horse_id": h["horse_id"],
             "time_index_m": int(90 + h["ability"] * 8 + rng.randint(-3, 3)),
             "speed_max": int(95 + h["ability"] * 8), "speed_avg": int(88 + h["ability"] * 7),
             "speed_distance": int(92 + h["ability"] * 8), "speed_course": int(92 + h["ability"] * 8),
             "speed_recent": [int(88 + h["ability"] * 7 + rng.randint(-5, 5)) for _ in range(4)],
             "odds": odds[i], "popularity": pop[i]} for i, h in enumerate(field)]},
        "meta": meta, "result": None, "lap": None, "finish": None, "on_time": None,
    }
    out.update(_extra_prerace(rng, rid, meta, field, entries, odds, pop))
    if not with_result:
        return out

    perf = [h["ability"] + rng.gauss(0, 0.9) for h in field]
    finish = sorted(range(n), key=lambda i: -perf[i])
    base = spec["distance"] / 16.4 + rng.uniform(-1.0, 1.0)
    t, times = base, {}
    for k, i in enumerate(finish):
        if k:
            t += rng.uniform(0.0, 0.5)
        times[i] = t
    laps_n = spec["distance"] // 200
    raw = [rng.uniform(11.4, 13.2) for _ in range(laps_n)]
    scale = round(base, 1) / sum(raw)
    laps = [round(x * scale, 1) for x in raw]
    pace = {"first_half_3f": round(sum(laps[:3]), 1), "second_half_3f": round(sum(laps[-3:]), 1),
            "t1f": laps[0], "t3f": round(sum(laps[:3]), 1), "l1f": laps[-1], "l3f": round(sum(laps[-3:]), 1)}
    last3 = {i: round(34.0 + rng.random() * 3 - (0.6 if h["style"] in ("差", "追") else 0), 1)
             for i, h in enumerate(field)}
    res_entries, passing_by_corner = [], [[] for _ in range(4)]
    pos = {i: finish.index(i) for i in range(n)}
    for i, h in enumerate(field):
        bias = {"逃": -3, "先": -1.5, "差": 1.5, "追": 3}[h["style"]]
        seq = []
        for c in range(4):
            w = (3 - c) / 3
            seq.append(max(1, min(n, round(pos[i] + 1 + bias * w + rng.uniform(-1, 1)))))
        for c in range(4):
            passing_by_corner[c].append((seq[c] + rng.random() * 0.1, i + 1))
        gap = times[i] - times[finish[max(0, pos[i] - 1)]] if pos[i] else 0.0
        res_entries.append({
            "finish_position": pos[i] + 1, "bracket_number": i + 1, "horse_number": i + 1,
            "horse_name": h["horse_name"], "horse_id": h["horse_id"],
            "sex_age": f"{h['sex']}{day.year - h['birth_year']}", "jockey_weight": weights[i],
            "jockey_name": h["jockey"][1], "jockey_id": h["jockey"][0],
            "finish_time": _fmt_time(times[i]), "time_sec": round(times[i], 1),
            "margin": _margin_text(gap) if pos[i] else "", "passing_order": "-".join(map(str, seq)),
            "last_3f": last3[i], "odds": odds[i], "popularity": pop[i], "weight": body[i],
            "weight_change": wchg[i], "trainer_name": h["trainer"][1], "trainer_id": h["trainer"][0],
        })
    res_entries.sort(key=lambda e: e["finish_position"])
    corner = [{"corner": c + 1, "label": f"{c + 1}角",
               "order_text": ",".join(str(num) for _, num in sorted(passing_by_corner[c]))} for c in range(4)]

    def yen(v: float) -> str:
        return f"{int(v * 100) // 10 * 10:,}"

    w1, w2, w3 = (finish[0] + 1, finish[1] + 1, finish[2] + 1)
    def o(num: int) -> float:
        return odds[num - 1]

    payoff = {
        "単勝": {"numbers": str(w1), "payout": yen(o(w1)), "popularity": str(pop[finish[0]])},
        "複勝": [{"numbers": str(x), "payout": yen(max(1.1, o(x) / 2.6)), "popularity": str(pop[x - 1])}
                for x in (w1, w2, w3)],
        "枠連": {"numbers": f"{min(w1, w2)} - {max(w1, w2)}", "payout": yen(o(w1) * o(w2) * 0.6), "popularity": "5"},
        "馬連": {"numbers": f"{min(w1, w2)} - {max(w1, w2)}", "payout": yen(o(w1) * o(w2) * 0.7), "popularity": "6"},
        "ワイド": [{"numbers": f"{min(a, b)} - {max(a, b)}", "payout": yen(max(1.1, o(a) * o(b) * 0.25)),
                  "popularity": "4"} for a, b in ((w1, w2), (w1, w3), (w2, w3))],
        "馬単": {"numbers": f"{w1} → {w2}", "payout": yen(o(w1) * o(w2) * 1.3), "popularity": "9"},
        "三連複": {"numbers": " - ".join(map(str, sorted((w1, w2, w3)))), "payout": yen(o(w1) * o(w2) * o(w3) * 0.5),
                "popularity": "12"},
        "三連単": {"numbers": f"{w1} → {w2} → {w3}", "payout": yen(o(w1) * o(w2) * o(w3) * 2.5), "popularity": "30"},
    }
    out["result"] = {**meta, "entries": res_entries, "payoff": payoff, "lap_times": laps, "pace": pace,
                     "corner_passing": corner}
    out["on_time"] = {k: v for k, v in out["result"].items() if k != "lap_times"} | {
        "lap_times": laps, "pace": pace, "result_schema_kind": "on_time"}
    out["lap"] = {"race_id": rid, "lap_times": laps, "pace": pace, "corner_passing": corner,
                  "entries_lap": [{"horse_number": e["horse_number"], "horse_id": e["horse_id"],
                                   "horse_name": e["horse_name"], "passing_order": e["passing_order"],
                                   "last_3f": e["last_3f"]} for e in res_entries]}
    by_no = {e["horse_number"]: e for e in res_entries}
    out["finish"] = {h["horse_id"]: by_no[i + 1] for i, h in enumerate(field)}
    return out


def _extra_prerace(rng: random.Random, rid: str, meta: dict[str, Any], field: list[dict[str, Any]],
                   entries: list[dict[str, Any]], odds: list[float], pop: dict[int, int]) -> dict[str, Any]:
    """出走前に存在するその他のレース単位データ（パドック・偏差値・2連系オッズ・追い切り等）。"""
    def base(i: int, h: dict[str, Any]) -> dict[str, Any]:
        return {"horse_number": i + 1, "horse_name": h["horse_name"], "horse_id": h["horse_id"]}

    n = len(field)
    pairs = [(a, b) for a in range(1, n + 1) for b in range(a + 1, n + 1)]
    ranked = sorted(pairs, key=lambda ab: odds[ab[0] - 1] * odds[ab[1] - 1])[:6]
    return {
        "paddock": {"race_id": rid, "entries": [
            {**base(i, h), "paddock_rank": rng.choice("ABCD"), "paddock_comment": "模擬コメント: 落ち着いている"} for i, h in enumerate(field)]},
        "barometer": {"race_id": rid, "entries": [
            {**base(i, h), "finish_order": 0, "index_total": rng.randint(80, 115), "index_start": rng.randint(80, 115),
             "index_chase": rng.randint(80, 115), "index_closing": rng.randint(80, 115)} for i, h in enumerate(field)]},
        "oikiri": {"race_id": rid, "entries": [
            {**base(i, h), "oikiri_course": "美Ｗ", "oikiri_time": f"{rng.uniform(80, 90):.1f}", "oikiri_eval": rng.choice("ABC")}
            for i, h in enumerate(field)]},
        "trainer_comment": {"race_id": rid, "entries": [
            {"horse_name": h["horse_name"], "horse_number": i + 1, "comment": "模擬コメント: 順調に調整できている"} for i, h in enumerate(field)]},
        "shutuba_past": {"race_id": rid, "entries": [
            {"horse_name": h["horse_name"], "horse_id": h["horse_id"], "horse_number": i + 1, "past_races": [
                {"date": "2026/09/06", "race_name": "模擬戦", "finish_position": rng.randint(1, 10), "distance": meta["distance"]}
                for _ in range(2)]} for i, h in enumerate(field)]},
        "pair_odds": {"race_id": rid,
                      "umaren": [{"pair": [a, b], "odds": round(odds[a - 1] * odds[b - 1] * 0.7, 1), "popularity": k + 1}
                                 for k, (a, b) in enumerate(ranked)],
                      "wide": [{"pair": [a, b], "odds_min": round(odds[a - 1] * odds[b - 1] * 0.25, 1),
                                "odds_max": round(odds[a - 1] * odds[b - 1] * 0.4, 1), "popularity": k + 1}
                               for k, (a, b) in enumerate(ranked)],
                      "umatan": [{"pair": [a, b], "odds": round(odds[a - 1] * odds[b - 1] * 1.4, 1), "popularity": k + 1}
                                 for k, (a, b) in enumerate(ranked)]},
        "detail": {**meta, "entries": [{**e, "time_index": rng.randint(80, 115)} for e in entries]},
        "smartrc": {"race_id": rid, "source": "dev_mock", "runners": [], "horses": [], "fullresults": []},
    }


def horse_training(rng: random.Random, h: dict[str, Any], past_day: date) -> dict[str, Any]:
    ents = []
    for k in range(3):
        d = past_day - timedelta(days=7 * (k + 1))
        laps = [round(rng.uniform(11.0, 13.5), 1) for _ in range(4)]
        ents.append({"race_info": "模擬レース前の追い切り", "date": d.isoformat(), "day_of_week": "水", "course": "美Ｗ",
                     "track_condition": "良", "rider": "助手", "time_raw": " ".join(map(str, laps)), "lap_times": laps,
                     "position": "3", "leg_color": "馬也", "evaluation": "模擬評価", "rank": rng.choice("ABC"),
                     "comment": "模擬コメント: 動きは良好"})
    return {"horse_id": h["horse_id"], "total_items": len(ents), "pages_fetched": 1, "entries": ents}


def horse_result(rng: random.Random, h: dict[str, Any], ran: list[dict[str, Any]], past_day: date) -> dict[str, Any]:
    hist = list(ran)
    for k in range(rng.randint(2, 5)):
        d = past_day - timedelta(days=28 * (k + 1) + rng.randint(0, 9))
        place, vname = rng.choice(VENUES)
        pos = max(1, min(14, int(round(rng.gauss(4 - h["ability"] * 2, 3)))))
        surface = rng.choice(["芝", "ダ"])
        dist = rng.choice([1200, 1400, 1600, 1800, 2000, 2400])
        t = dist / 16.4 + rng.uniform(-1, 2)
        hist.append({
            "date": d.strftime("%Y/%m/%d"), "venue": f"{int(place)}{vname}{rng.randint(1, 8)}",
            "weather": "晴", "race_round": rng.choice([1, 5, 9, 11]), "race_name": "モック過去戦",
            "race_id": f"{d.year}{place}0{rng.randint(1, 4)}0{rng.randint(1, 8)}{rng.choice([1, 5, 9, 11]):02d}",
            "field_size": 12, "bracket_number": rng.randint(1, 8), "horse_number": rng.randint(1, 12),
            "odds": round(rng.uniform(1.5, 40), 1), "popularity": rng.randint(1, 12), "finish_position": pos,
            "jockey_name": h["jockey"][1], "jockey_weight": 56.0, "surface": surface, "distance": dist,
            "time_index": int(85 + h["ability"] * 8), "track_condition": "良", "finish_time": _fmt_time(t),
            "time_sec": round(t, 1), "margin": "1", "passing_order": "5-5-4-4", "last_3f": round(34 + rng.random() * 3, 1),
            "weight": rng.randint(440, 520), "weight_change": rng.randint(-6, 6), "winner": "モック勝ち馬",
        })
    hist.sort(key=lambda r: r["date"], reverse=True)
    pos_list = [r["finish_position"] for r in hist]
    rec = [sum(1 for p in pos_list if p == k) for k in (1, 2, 3)]
    rec.append(len(pos_list) - sum(rec))
    return {
        "horse_id": h["horse_id"], "horse_name": h["horse_name"], "name_en": h["name_en"], "sex": h["sex"],
        "birthday": f"{h['birth_year']}年{(h['seed'] % 5) + 2}月{(h['seed'] % 27) + 1}日",
        "trainer": h["trainer"][1], "owner": "モックオーナー", "breeder": "モック牧場", "birthplace": "モック町",
        "total_earnings": sum(max(0, 1200 - (p - 1) * 300) for p in pos_list),
        "career": f"{len(pos_list)}戦{rec[0]}勝 [{rec[0]}-{rec[1]}-{rec[2]}-{rec[3]}]", "career_record": rec,
        "major_wins": [], "sire": h["sire"], "dam": h["dam"], "dam_sire": h["dam_sire"], "race_history": hist,
    }


def history_row(e: dict[str, Any], rid: str, meta: dict[str, Any], day: date, place: str, venue: str, field_size: int,
                dayn: int) -> dict[str, Any]:
    return {
        "date": day.strftime("%Y/%m/%d"), "venue": f"{int(place)}{venue}{dayn}", "weather": meta["weather"],
        "race_round": meta["round"], "race_name": meta["race_name"], "race_id": rid, "field_size": field_size,
        "bracket_number": e["bracket_number"], "horse_number": e["horse_number"], "odds": e["odds"],
        "popularity": e["popularity"], "finish_position": e["finish_position"], "jockey_name": e["jockey_name"],
        "jockey_weight": e["jockey_weight"], "surface": meta["surface"], "distance": meta["distance"],
        "time_index": 95, "track_condition": meta["track_condition"], "finish_time": e["finish_time"],
        "time_sec": e["time_sec"], "margin": e["margin"], "passing_order": e["passing_order"],
        "last_3f": e["last_3f"], "weight": e["weight"], "weight_change": e["weight_change"], "winner": "",
    }


def _put_schemaless(put: Any, rid: str, field: list[dict[str, Any]], rng: random.Random) -> None:
    """スキーマ未定義（pending）の生成物。形式は暫定で、実データの観測後に置き換える。"""
    from src.api.stg_mock import mock_tracking_difficulty

    class _Shim:
        def load(self, category: str, key: str) -> dict[str, Any] | None:
            return {"entries": [{"horse_id": h["horse_id"]} for h in field]} if category == "race_shutuba" else None

    put("tracking_difficulty", rid, mock_tracking_difficulty(rid, _Shim()))
    put("final_odds_prediction", rid, {"race_id": rid, "model_version": "dev-mock-v1", "predictions": [
        {"horse_number": i + 1, "horse_id": h["horse_id"], "predicted_final_win_odds": round(rng.uniform(1.5, 60), 1)}
        for i, h in enumerate(field)]})
    put("finish_order_prediction", rid, {"race_id": rid, "model_version": "dev-mock-v1", "predictions": [
        {"horse_number": i + 1, "horse_id": h["horse_id"], "predicted_rank": i + 1} for i, h in enumerate(field)]})
    put("race_performance", rid, {"race_id": rid, "horses": [
        {"horse_number": i + 1, "horse_id": h["horse_id"], "performance_rating": round(rng.uniform(80, 120), 1)}
        for i, h in enumerate(field)]})


def pick_days(today: date) -> tuple[date, date]:
    upcoming = today + timedelta(days=(5 - today.weekday()) % 7)
    return upcoming - timedelta(days=6), upcoming


def _page_ref_dir() -> Path:
    override = os.environ.get("KEIBA_PAGE_REFERENCE_DIR", "").strip()
    return Path(override).expanduser().resolve() if override else Path("data") / "page_reference"


def _is_mock_file(p: Path) -> bool:
    try:
        return bool(json.loads(p.read_text(encoding="utf-8")).get("_meta", {}).get("dev_mock"))
    except (OSError, ValueError):
        return False


def clean(root: Path) -> None:
    """生成済みモック（``_meta.dev_mock``）だけを消す。dev で save() した実データ相当は残す。"""
    dirs = [root] + [_page_ref_dir() / c for c in ("race_lists", "race_day_schedule")]
    for d in dirs:
        for p in d.rglob("*.json") if d.is_dir() else []:
            if _is_mock_file(p):
                p.unlink()
    if root.is_dir():
        for d in sorted((x for x in root.rglob("*") if x.is_dir()), reverse=True):
            try:
                d.rmdir()
            except OSError:
                pass


def generate(today: date, root: Path) -> dict[str, int]:
    rng = random.Random(SEED)
    cmap = HybridStorage.CATEGORY_MAP
    store = DevStore(root)
    root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    stamp = datetime.now(JST)
    base_meta = {"dev_mock": True, "scraped_at": stamp.timestamp(), "scraped_at_jst": stamp.strftime("%Y-%m-%d %H:%M:%S")}

    payloads: dict[tuple[str, str], dict[str, Any]] = {}

    def put(category: str, key: str, data: dict[str, Any]) -> None:
        payloads[(category, key)] = data
        data = {**data, "_meta": {**data.get("_meta", {}), **base_meta}}
        vr = schemas.validate(category, data)
        if not vr["passed"]:
            raise SystemExit(f"モックがスキーマ不合格: {category}/{key}: {vr}")
        if category not in cmap:
            raise SystemExit(f"CATEGORY_MAP に無いカテゴリ: {category}")
        if cmap[category] == "local_only":
            d = _page_ref_dir() / category
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"{key}.json"
            if p.exists() and not _is_mock_file(p):
                print(f"skip（実データあり）: {p}", file=sys.stderr)
                return
            p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        else:
            store.write(category, key, cmap[category], data)
        counts[category] = counts.get(category, 0) + 1

    horses = make_horses(rng)
    peds = {h["horse_id"]: pedigree(h) for h in horses}
    past_day, upcoming = pick_days(today)
    plan = [(past_day, 3, 2, True), (upcoming, 4, 1, False)]
    ran: dict[str, list[dict[str, Any]]] = {h["horse_id"]: [] for h in horses}

    from src.api.stg_mock import mock_predictions

    for day, kai, dayn, with_result in plan:
        order = list(range(N_HORSES))
        rng.shuffle(order)
        slots = iter(range(0, N_HORSES, FIELD))
        races_listing, schedule_slots = [], []
        for place, venue in VENUES:
            for rr in RACE_SPECS:
                start = next(slots)
                field = [horses[j] for j in order[start:start + FIELD]]
                rid = f"{day.year}{place}{kai:02d}{dayn:02d}{rr:02d}"
                r = build_race(rng, rid, day, venue, rr, field, with_result)
                put("race_shutuba", rid, r["shutuba"])
                put("race_odds", rid, r["odds"])
                put("race_index", rid, r["index"])
                for cat, k in (("race_paddock", "paddock"), ("race_barometer", "barometer"), ("race_oikiri", "oikiri"),
                               ("race_trainer_comment", "trainer_comment"), ("race_shutuba_past", "shutuba_past"),
                               ("race_pair_odds", "pair_odds"), ("race_detail", "detail"), ("smartrc_race", "smartrc")):
                    put(cat, rid, r[k])
                _put_schemaless(put, rid, field, rng)
                if with_result:
                    put("race_result", rid, r["result"])
                    put("race_result_on_time", rid, r["on_time"])
                    put("race_result_lap", rid, r["lap"])
                    for h in field:
                        ran[h["horse_id"]].append(
                            history_row(r["finish"][h["horse_id"]], rid, r["meta"], day, place, venue, FIELD, dayn))
                pred = mock_predictions(rid, [h["horse_id"] for h in field])
                pred["model_version"] = "dev-mock-v1"
                put("race_predictions", rid, pred)
                races_listing.append({"race_id": rid, "round": rr, "venue": venue, "race_name": RACE_SPECS[rr]["name"]})
                schedule_slots.append({
                    "race_id": rid, "venue": venue, "round": rr, "race_name": RACE_SPECS[rr]["name"],
                    "start_time_str": RACE_SPECS[rr]["start"], "time_source": "dev_mock",
                    "post_time_iso": f"{day.isoformat()}T{RACE_SPECS[rr]['start']}:00+09:00"})
        put("race_lists", day.strftime("%Y%m%d"), {"date": day.strftime("%Y%m%d"), "races": races_listing})
        put("race_day_schedule", day.strftime("%Y%m%d"),
            {"date_fmt": day.strftime("%Y%m%d"), "iso_date": day.isoformat(), "slots": schedule_slots})

    for h in horses:
        put("horse_result", h["horse_id"], horse_result(rng, h, ran[h["horse_id"]], past_day))
        put("horse_pedigree_5gen", h["horse_id"], peds[h["horse_id"]])
        put("horse_training", h["horse_id"], horse_training(rng, h, past_day))
    dam_ids = sorted({p["ancestors"][1]["horse_id"] for p in peds.values()})
    for i, dam in enumerate(dam_ids):
        put("broodmare_mating", dam, {"horse_id": dam, "matings": [
            {"mating_year": 2020 + i % 5, "mating_date": f"{2020 + i % 5}-0{1 + i % 6}-{10 + i % 15}"}], "mating_count": 1,
            "source": "dev_mock"})
    # 派生カテゴリは本番と同じ抽出ロジックで元データから作る
    from src.scraper.row_data_extractor import DERIVED_CATEGORY_MAP

    for derived, (parent, fn) in DERIVED_CATEGORY_MAP.items():
        for (cat, key), data in list(payloads.items()):
            if cat == parent:
                put(derived, key, fn(data))
    races_for_trace = [k for (c, k) in payloads if c == "race_shutuba"][:4]
    for rid in races_for_trace:
        put("requirement_row_trace", f"race_{rid}_nk_shutuba_entries", {
            "row_id": "nk_shutuba_entries", "trace_key": f"race_{rid}_nk_shutuba_entries", "scope": "race", "primary_id": rid,
            "canonical": [{"category": "race_shutuba", "key": rid}], "title_ja": "出馬表HTML（モック）"})
    put("jra_cushion", str(upcoming.year), {"records": [
        {"year": upcoming.year, "venue_code": pl, "venue_name": v, "kai": 3, "is_race_day": True, "date": d.isoformat(),
         "cushion_value": 9.4, "turf_moisture_goal": 11.0, "turf_moisture_4corner": 11.6, "dirt_moisture_goal": 2.4,
         "dirt_moisture_4corner": 2.9} for d in (past_day, upcoming) for pl, v in VENUES]})
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--today", help="基準日 YYYY-MM-DD（既定: 今日）")
    ap.add_argument("--clean", action="store_true", help="生成済みモックを削除して終了")
    args = ap.parse_args()
    root = dev_mock_root()
    if args.clean:
        clean(root)
        print(f"削除しました: {root}")
        return 0
    today = date.fromisoformat(args.today) if args.today else datetime.now(JST).date()
    clean(root)
    counts = generate(today, root)
    past, up = pick_days(today)
    print(f"dev モックを生成しました: {root}（結果あり {past} / 結果なし {up}）")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
