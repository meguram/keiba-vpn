"""過去分補填（2016〜2019 年）で取得した race_result の『値』の検査。古いページ構造をパーサが想定しておらず、スキーマは通るが値がおかしい事故を拾う。

見るもの（レース単位）:
  1. スキーマ検証（保存時と同じ schemas.validate。strict 保存を通っているはずなので、通常は 0 件）
  2. 値の妥当性: レース ID・日付・競馬場・面・距離・天候・馬場・発走時刻・頭数と出走表の一致・馬番/枠番/馬 ID・着順・走破時間と秒数の一致・
     上がり 3F・単勝オッズ/人気・馬体重・払戻（単勝の馬番が 1 着と一致）・ラップ（本数と合計が走破時間と合う）
  3. 見たことのない値: カテゴリ項目（面・天候・馬場・着順の状態・性齢の記号）が、基準年（既定 2023 年）の標本に無い値を出していないか
結果は data/local/meta/past_value_check/latest.json。検査済みのレースは state に覚え、既定では未検査分だけを見る（--all で全件）。
使い方: python -m src.scripts.data.verify_past_values [--years 2019 2018 2017 2016] [--ref-year 2023 --ref-n 400] [--all] [--max-fail-rate 0.01]
終了コード: 0=問題なし、1=いずれかの年で『失敗レース率』が --max-fail-rate を超えた（読み取りのみ。保存・削除はしない）
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(os.environ.get("KEIBA_PAST_VALUE_CHECK_DIR") or ROOT / "data" / "local" / "meta" / "past_value_check")
VENUES = {"札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"}
CATEGORICAL = ("surface", "weather", "track_condition", "direction", "grade")
ENTRY_CATEGORICAL = ("finish_status",)


def _tsec(s: Any) -> float | None:
    m = re.fullmatch(r"(\d+):(\d{2})\.(\d)", str(s or ""))
    return int(m[1]) * 60 + int(m[2]) + int(m[3]) / 10 if m else None


def _num(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def check_race(d: dict[str, Any], rid: str) -> list[str]:
    """失敗した検査の名前の一覧（空なら問題なし）。"""
    f: list[str] = []

    def bad(code: str, cond: bool) -> None:
        if cond:
            f.append(code)

    bad("race_id_format", not re.fullmatch(r"\d{12}", rid) or d.get("race_id") != rid)
    date = str(d.get("date") or "")
    bad("date_format", not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date))
    bad("date_year_vs_race_id", bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", date)) and date[:4] != rid[:4])
    bad("venue", d.get("venue") not in VENUES)
    dist = d.get("distance")
    bad("distance_range", not isinstance(dist, int) or not 800 <= dist <= 4500 or dist % 10 != 0)
    bad("grade_unmapped", d.get("grade") in ("その他", "", None))     # 旧クラス名などをパーサが読めていない兆候
    bad("start_time_format", not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", str(d.get("start_time") or "")))
    bad("round_range", not isinstance(d.get("round"), int) or not 1 <= d["round"] <= 12)
    ents = d.get("entries") or []
    bad("no_entries", not ents)
    if not ents:
        return f
    bad("field_size_vs_entries", d.get("field_size") != len(ents))
    nums = [e.get("horse_number") for e in ents]
    bad("horse_number_dup_or_range", len(set(nums)) != len(nums) or any(not isinstance(n, int) or not 1 <= n <= 18 for n in nums))
    bad("bracket_range", any(not isinstance(e.get("bracket_number"), int) or not 1 <= e["bracket_number"] <= 8 for e in ents))
    bad("horse_id_format", any(not re.fullmatch(r"\d{10}", str(e.get("horse_id") or "")) for e in ents))
    bad("jockey_or_trainer_id_missing", any(not e.get("jockey_id") or not e.get("trainer_id") for e in ents))
    live = [e for e in ents if e.get("finish_status") not in ("excluded", "scratched")]       # 取消・除外はオッズ・人気・馬体重が無いのが通常
    fin = [e for e in ents if e.get("finish_status") in (None, "finished") and isinstance(e.get("finish_position"), int)]   # 状態の項目が無い旧形式（2020 年以降の保存分）も対象
    pos = [e.get("finish_position") for e in fin]
    bad("finish_position_invalid", any(not isinstance(p, int) or not 1 <= p <= len(ents) for p in pos))
    bad("winner_missing", 1 not in pos)
    secs = []
    for e in fin:
        t = _tsec(e.get("finish_time"))
        if t is None or _num(e.get("time_sec")) is None or abs(t - float(e["time_sec"])) > 0.05:
            f.append("time_vs_time_sec")
            break
    for e in fin:
        t = _num(e.get("time_sec"))
        if t is not None:
            secs.append((e.get("finish_position"), t))
    win_t = min((t for p, t in secs if p == 1), default=None)
    bad("winner_not_fastest", win_t is not None and any(t < win_t - 0.05 for _, t in secs))
    bad("time_sec_range", any(not 40 <= t <= 400 for _, t in secs))
    l3 = [_num(e.get("last_3f")) for e in fin if e.get("last_3f") not in (None, "")]
    hurdle = d.get("surface") == "障" or "障害" in str(d.get("race_name") or "")     # 障害戦は 1F 相当の値で、surface の表記も年代でゆれる
    bad("last_3f_range", not hurdle and any(v is None or not 28 <= v <= 55 for v in l3))
    odds = [_num(e.get("odds")) for e in live if e.get("odds") not in (None, "")]
    bad("odds_range", any(v is None or v < 1.0 or v > 9999 for v in odds))
    pops = [e.get("popularity") for e in live if e.get("popularity") not in (None, "")]
    bad("popularity_range", any(not isinstance(p, int) or not 1 <= p <= len(ents) for p in pops))
    wts = [e.get("weight") for e in live if e.get("weight") not in (None, "")]
    bad("weight_range", any(not isinstance(w, int) or not 300 <= w <= 700 for w in wts))
    bad("sex_age_format", any(not re.fullmatch(r"[牡牝セ]\d{1,2}", str(e.get("sex_age") or "")) for e in live))
    bad("jockey_weight_range", any(_num(e.get("jockey_weight")) is None or not 40 <= float(e["jockey_weight"]) <= 66 for e in live))
    pay = (d.get("payoff") or {}).get("単勝")
    if isinstance(pay, list) and pay and all(isinstance(x, dict) for x in pay):       # 同着は複数行
        pay = {"numbers": ",".join(str(x.get("numbers")) for x in pay), "payout": pay[0].get("payout")}
    winners = {str(e.get("horse_number")) for e in fin if e.get("finish_position") == 1}
    if isinstance(pay, dict):
        got = {x for x in re.split(r"[,\-/ ]+", str(pay.get("numbers") or "")) if x}
        bad("payoff_win_vs_winner", bool(winners) and not (got & winners))
        bad("payoff_win_amount", _num(str(pay.get("payout")).replace(",", "")) is None or not 100 <= float(str(pay["payout"]).replace(",", "")) <= 99999999)
    else:
        f.append("payoff_win_missing")
    laps = d.get("lap_times") or []
    if laps and isinstance(dist, int) and d.get("surface") in ("芝", "ダート") and not hurdle:
        n = math.ceil(dist / 200)
        bad("lap_count", len(laps) not in (n - 1, n, n + 1))
        if win_t is not None:
            bad("lap_sum_vs_winner_time", abs(sum(float(x) for x in laps) - win_t) > 2.0)
    return f


def _cat_values(d: dict[str, Any]) -> dict[str, set[str]]:
    out = {k: {str(d.get(k))} for k in CATEGORICAL}
    out["finish_status"] = {str(e.get("finish_status")) for e in d.get("entries") or []}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--years", type=int, nargs="+", default=[2019, 2018, 2017, 2016])
    ap.add_argument("--ref-years", type=int, nargs="+", default=[2020, 2021, 2022, 2023, 2024, 2025])
    ap.add_argument("--ref-n", type=int, default=100, help="基準年ごとの標本数")
    ap.add_argument("--all", action="store_true", help="検査済みも含めて全件を見る")
    ap.add_argument("--max-fail-rate", type=float, default=0.01)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    from src.scraper import schemas
    from src.scraper.storage import HybridStorage

    s = HybridStorage()
    state_path = OUT_DIR / "state.json"
    state = {}
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    checked: dict[str, list[str]] = state.get("checked", {})

    ref_vals: dict[str, set[str]] = defaultdict(set)
    for ry in args.ref_years:
        ref_keys = sorted(s.list_keys("race_result", str(ry)))
        step = max(1, len(ref_keys) // max(1, args.ref_n))
        for rid in ref_keys[::step][: args.ref_n]:
            d = s.load("race_result", rid)
            if d:
                for k, v in _cat_values(d).items():
                    ref_vals[k] |= v
    ref_vals["finish_status"] -= {"None"}      # 旧形式（状態の項目なし）は基準にしない。新形式の値は『新しい値』として情報だけ出す

    report: dict[str, Any] = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "ref": {"years": args.ref_years, "values": {k: sorted(v) for k, v in ref_vals.items()}}, "years": {}}
    worst = 0.0
    for y in args.years:
        keys = sorted(s.list_keys("race_result", str(y)))
        todo = [k for k in keys if args.all or k not in checked]
        fails: Counter[str] = Counter()
        samples: dict[str, list[str]] = defaultdict(list)
        unseen: dict[str, Counter[str]] = defaultdict(Counter)
        failed_races = 0
        for rid in todo:
            d = s.load("race_result", rid)
            if not d:
                fails["load_failed"] += 1
                failed_races += 1
                samples["load_failed"].append(rid)
                continue
            codes = check_race(d, rid)
            v = schemas.validate("race_result", d)
            if not v.get("passed", True):
                codes.append("schema_validate")
            checked[rid] = list(codes)
            for k, vals in _cat_values(d).items():
                for val in vals - ref_vals.get(k, vals):
                    if k == "finish_status" and not ref_vals.get(k):
                        continue
                    unseen[k][val] += 1
                    if len(samples[f"info_unseen:{k}={val}"]) < 3:
                        samples[f"info_unseen:{k}={val}"].append(rid)
            if codes:
                failed_races += 1
            for c in codes:
                fails[c] += 1
                if len(samples[c]) < 3:
                    samples[c].append(rid)
        rate = failed_races / len(todo) if todo else 0.0
        worst = max(worst, rate)
        report["years"][str(y)] = {"races_in_store": len(keys), "checked_now": len(todo), "failed_races": failed_races, "fail_rate": round(rate, 4),
                                   "failures": dict(fails.most_common()), "info_unseen_values": {k: dict(v) for k, v in unseen.items()}, "samples": dict(samples)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state["checked"] = checked
    state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    text = json.dumps(report, ensure_ascii=False, indent=1)
    (OUT_DIR / "latest.json").write_text(text, encoding="utf-8")
    with open(OUT_DIR / "history.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"at": report["at"], "years": {y: {k: v[k] for k in ("checked_now", "failed_races", "fail_rate")} for y, v in report["years"].items()}}, ensure_ascii=False) + "\n")
    if args.json_out:
        Path(args.json_out).write_text(text, encoding="utf-8")
    for y, v in report["years"].items():
        print(f"{y}: 検査 {v['checked_now']} / 蓄積 {v['races_in_store']}  失敗レース {v['failed_races']} ({v['fail_rate']:.2%})  {dict(list(v['failures'].items())[:8])}  新しい値(情報): {v['info_unseen_values']}")
    return 1 if worst > args.max_fail_rate else 0


if __name__ == "__main__":
    sys.exit(main())
