#!/usr/bin/env python3
"""
実データを取りに行くシナリオ型スクレイピングテスト（unittest / CI 対象外。手動実行専用）。

「開催日のレース1件」を題材に、発走 T-60 / T-15 のオッズ運用で必要な取得を順に通して検証する:

  S1 race_list    日付のレース一覧を取得できる
  S2 race_card    出馬表（頭数・発走時刻）を取得できる
  S3 odds_t60     単複オッズを phase=t60 で取得・保存できる（全馬分・複勝 min<=max）
  S4 pair_t60     馬連・ワイド・馬単・3連複を phase=t60 で取得でき、組合せ数が頭数と整合する
  S5 odds_t15     phase=t15 で再取得すると最新が t15 になり、t60 が snapshots に残る
  S6 schema       保存物が schemas.validate に適合する
  S7 recommend    取得したオッズから推奨馬券（odds_mode=auto/actual/predicted）が計算できる
  S8 timing       各データ点（出馬表・単勝/複勝・馬連・ワイド・馬単・3連複）の取得所要時間を --repeat 回計測し、
                  T-X（発走何分前に回すか）の判断材料として「必要リードタイム」を見積もる

使い方（リポジトリルートで。.env に KEIBA_ENV=dev と netkeiba ログイン情報が必要）:
  # 実アクセスの明示許可が必要（--yes または KEIBA_RUN_LIVE_SCRAPE_TESTS=1）
  python3 tests/scraper/manual/live_odds_scenarios.py --date 20261017 --yes
  python3 tests/scraper/manual/live_odds_scenarios.py --race-id 202605040811 --yes
  python3 tests/scraper/manual/live_odds_scenarios.py --date 20261017 --max-races 3 --yes --capture-dev

  --capture-dev  取得した実データを dev ストア（data/dev_mock）に保存したまま残す。モックの代わりに
                 実データで UI（NEXT_PUBLIC_MOCK なし）を確認したいときに使う。KEIBA_ENV=dev 必須
                 （stg/prod では GCS に書いてしまうため拒否する）。
  --offline      ネットへは出ず、dev ストアに既にあるデータへ S3〜S7 の検証だけをかける
                 （検証ロジック自体の動作確認用: python3 ... --race-id <id> --offline）
  --json         結果を JSON で stdout へ
  --measure-all  多数のデータ点（レース系 14 種 + 馬系 4 種）を --repeat 回ずつ取得し、実測時間を蓄積する。
                 記録は ScraperRunner 側で常時行われ（src/scraper/timing_stats.py）、集計は
                 python -m src.scraper.timing_stats report で見る。--horses N（既定2）で馬系の対象頭数。
  --timing-only  S8（所要時間計測）だけ実行する。--repeat N（既定5）/ --timing-gap 秒（既定3）
                 見積り条件: --same-time-races（同時刻に重なる最大レース数, 既定3）/ --predict-sec（AI予測の所要秒, 既定0）
                 / --margin-sec（余裕, 既定120）。結果は data/local/meta/scrape_timing/ に JSON で蓄積する。

実アクセス時の配慮: 1 レースあたり約 10 リクエスト。--max-races の既定は 1。間隔は ScraperRunner の
interval（既定）に従う。3 連複 API(type=7) の応答形式は未検証のため、S4 が失敗したときは生の応答キーを
診断出力する（パーサ修正の手掛かり）。
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.WARNING)


@dataclass
class Result:
    scenario: str
    status: str  # PASS | FAIL | WARN | SKIP
    detail: str = ""


@dataclass
class RaceReport:
    race_id: str
    results: list[Result] = field(default_factory=list)

    def add(self, scenario: str, status: str, detail: str = "") -> None:
        self.results.append(Result(scenario, status, detail))

    @property
    def failed(self) -> bool:
        return any(r.status == "FAIL" for r in self.results)


# ── 共通ヘルパ ──────────────────────────────────────────────────────────────


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    p = REPO_ROOT / ".env"
    if p.is_file():
        load_dotenv(p)


def _expected_combos(n: int) -> dict[str, int]:
    return {
        "umaren": math.comb(n, 2),
        "wide": math.comb(n, 2),
        "umatan": n * (n - 1),
        "sanrenpuku": math.comb(n, 3),
    }


def _entries(card: dict | None) -> list[dict]:
    return list((card or {}).get("entries") or [])


# ── 所要時間の計測 ───────────────────────────────────────────────────────────


ODDS_TYPE_LABEL = {"1": "単勝", "2": "複勝", "4": "馬連", "5": "ワイド", "6": "馬単", "7": "3連複"}


class Timing:
    """HTTP 1 回ごと（requests.Session.get）と runner の 1 ステップごとの所要時間を記録する。"""

    def __init__(self) -> None:
        self.http: list[dict[str, Any]] = []
        self.steps: list[dict[str, Any]] = []

    @staticmethod
    def label(url: str) -> str:
        m = re.search(r"api_get_jra_odds\.html.*[?&]type=(\d+)", url)
        if m:
            return f"オッズAPI/{ODDS_TYPE_LABEL.get(m.group(1), 'type' + m.group(1))}"
        m = re.search(r"/([A-Za-z_]+)\.html", url)
        return f"HTML/{m.group(1)}" if m else url[:60]

    def install(self, runner) -> None:
        sess = runner.client._session
        orig = sess.get

        def timed_get(url, *a, **kw):
            t0 = time.perf_counter()
            status, size = None, 0
            try:
                resp = orig(url, *a, **kw)
                status, size = resp.status_code, len(resp.content or b"")
                return resp
            finally:
                self.http.append({"label": self.label(str(url)), "sec": time.perf_counter() - t0, "status": status, "bytes": size})

        sess.get = timed_get

    def step(self, name: str, fn: Callable[[], Any]) -> Any:
        n0 = len(self.http)
        t0 = time.perf_counter()
        try:
            return fn()
        finally:
            wall = time.perf_counter() - t0
            http_sum = sum(r["sec"] for r in self.http[n0:])
            self.steps.append({"name": name, "wall": wall, "http": http_sum, "wait": max(0.0, wall - http_sum),
                               "requests": len(self.http) - n0})


def _stats(xs: list[float]) -> dict[str, float]:
    xs = sorted(xs)
    p95 = xs[min(len(xs) - 1, math.ceil(len(xs) * 0.95) - 1)]
    return {"n": len(xs), "min": xs[0], "med": statistics.median(xs), "p95": p95, "max": xs[-1], "mean": statistics.fmean(xs)}


def summarize_timing(tm: Timing) -> dict[str, Any]:
    by_http: dict[str, list[float]] = {}
    for r in tm.http:
        by_http.setdefault(r["label"], []).append(r["sec"])
    by_step: dict[str, list[dict]] = {}
    for s in tm.steps:
        by_step.setdefault(s["name"], []).append(s)
    return {
        "http": {k: _stats(v) for k, v in by_http.items()},
        "http_bytes": {k: max((r["bytes"] for r in tm.http if r["label"] == k), default=0) for k in by_http},
        "steps": {k: {**_stats([x["wall"] for x in v]), "http_med": statistics.median(x["http"] for x in v),
                      "wait_med": statistics.median(x["wait"] for x in v), "requests": v[0]["requests"]} for k, v in by_step.items()},
    }


def estimate_lead(summary: dict[str, Any], *, same_time_races: int, predict_sec: float, margin_sec: float) -> dict[str, Any]:
    """T-X 見積り。1 レースの「オッズ一式」(単複+2連系+3連複) の p95 を直列実行し、同時刻レース数ぶん積む。"""
    steps = summary["steps"]
    bundle = [steps[k]["p95"] for k in ("単複オッズ", "2連系+3連複オッズ") if k in steps]
    if not bundle:
        return {}
    per_race = sum(bundle)
    batch = per_race * max(1, same_time_races)
    need = batch + predict_sec + margin_sec
    return {
        "per_race_odds_bundle_p95_sec": round(per_race, 1),
        "batch_sec": round(batch, 1),
        "predict_sec": predict_sec,
        "margin_sec": margin_sec,
        "min_lead_sec": round(need, 1),
        "min_lead_min_rounded_up_to_5": int(math.ceil(need / 60 / 5) * 5),
    }


def scenario_timing(rep: RaceReport, runner, tm: Timing, repeat: int, gap: float) -> dict[str, Any]:
    for i in range(repeat):
        tm.step("出馬表", lambda: runner.scrape_race_card(rep.race_id, skip_existing=False))
        tm.step("単複オッズ", lambda: runner.scrape_odds(rep.race_id, skip_existing=False))
        tm.step("2連系+3連複オッズ", lambda: runner.scrape_pair_odds(rep.race_id, skip_existing=False))
        if i < repeat - 1:
            time.sleep(gap)
    summary = summarize_timing(tm)
    rep.add("S8 timing", "PASS", f"{repeat}回計測 / HTTP {len(tm.http)}回")
    return summary


def print_timing(summary: dict[str, Any], est: dict[str, Any]) -> None:
    print("\n--- S8 所要時間（秒）---")
    print(f"{'ステップ(runner 1 呼び出し)':<22}{'n':>3}{'min':>7}{'med':>7}{'p95':>7}{'max':>7}  HTTP中央値 待機中央値 req数")
    for k, v in summary["steps"].items():
        print(f"{k:<22}{v['n']:>3}{v['min']:>7.2f}{v['med']:>7.2f}{v['p95']:>7.2f}{v['max']:>7.2f}  {v['http_med']:>9.2f} {v['wait_med']:>9.2f} {v['requests']:>5}")
    print(f"\n{'HTTP 1 回ごと':<22}{'n':>3}{'min':>7}{'med':>7}{'p95':>7}{'max':>7}  最大bytes")
    for k, v in summary["http"].items():
        print(f"{k:<22}{v['n']:>3}{v['min']:>7.2f}{v['med']:>7.2f}{v['p95']:>7.2f}{v['max']:>7.2f}  {summary['http_bytes'][k]:>9}")
    if est:
        print(f"\n--- T-X 見積り ---\n  1 レースのオッズ一式(p95)={est['per_race_odds_bundle_p95_sec']}s × 同時刻レース → バッチ {est['batch_sec']}s"
              f"\n  + AI予測 {est['predict_sec']}s + 余裕 {est['margin_sec']}s = 最小リード {est['min_lead_sec']}s"
              f" → 発走 T-{est['min_lead_min_rounded_up_to_5']}分より前に開始が必要（5分刻み切り上げ）")
        print("  ※ 待機中央値は ScraperRunner の間隔制御（意図的な sleep）。短縮できるのは HTTP 部分のみ。")


def save_timing(race_id: str, summary: dict[str, Any], est: dict[str, Any], tm: Timing) -> Path:
    from datetime import datetime

    d = REPO_ROOT / "data" / "local" / "meta" / "scrape_timing"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{race_id}.json"
    path.write_text(json.dumps({"race_id": race_id, "measured_at": datetime.now().isoformat(timespec="seconds"),
                                "summary": summary, "estimate": est, "raw_http": tm.http, "raw_steps": tm.steps},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ── シナリオ本体（fetch=True で実アクセス、False は保存済みデータの検証のみ） ──────


def scenario_card(rep: RaceReport, storage, runner, fetch: bool) -> dict | None:
    card = runner.scrape_race_card(rep.race_id, skip_existing=not fetch) if fetch else storage.load("race_shutuba", rep.race_id)
    n = len(_entries(card))
    if n < 4:
        rep.add("S2 race_card", "FAIL", f"出走馬が {n} 頭（出馬表が取れていない）")
        return None
    rep.add("S2 race_card", "PASS", f"{n}頭 発走 {card.get('start_time', '?')}")
    return card


def scenario_odds(rep: RaceReport, storage, runner, fetch: bool, phase: str, n_horses: int) -> None:
    label = f"S3 odds_{phase}"
    doc = runner.scrape_odds(rep.race_id, skip_existing=False, phase=phase) if fetch else storage.load("race_odds", rep.race_id)
    if not doc or not doc.get("entries"):
        rep.add(label, "FAIL", "単複オッズが取得できない（発売開始前か API 変更の可能性）")
        return
    ents = doc["entries"]
    problems = []
    if len(ents) < n_horses - 2:
        problems.append(f"頭数不足 {len(ents)}/{n_horses}")
    for e in ents:
        wo = e.get("win_odds")
        lo, hi = e.get("place_odds_min"), e.get("place_odds_max")
        if not isinstance(wo, (int, float)) or wo <= 0:
            problems.append(f"馬番{e.get('horse_number')}: win_odds 不正 {wo!r}")
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo > hi:
            problems.append(f"馬番{e.get('horse_number')}: place min>max")
    if fetch and doc.get("phase") != phase:
        problems.append(f"phase が {doc.get('phase')!r}（期待 {phase!r}）")
    if problems:
        rep.add(label, "FAIL", "; ".join(problems[:4]))
    else:
        rep.add(label, "PASS", f"{len(ents)}頭 snapshot_at={doc.get('snapshot_at')}")


def scenario_pair(rep: RaceReport, storage, runner, fetch: bool, phase: str, n_horses: int) -> None:
    label = f"S4 pair_{phase}"
    doc = runner.scrape_pair_odds(rep.race_id, skip_existing=False, phase=phase) if fetch else storage.load("race_pair_odds", rep.race_id)
    if not doc:
        rep.add(label, "FAIL", "2連系/3連複オッズが取得できない")
        return
    exp = _expected_combos(n_horses)
    got = {k: len(doc.get(k) or []) for k in exp}
    bad = []
    for k, want in exp.items():
        # 取消・除外馬があると少なくなる。0 は取得失敗、想定の 5 割未満は異常とみなす。
        if got[k] == 0:
            bad.append(f"{k}=0")
        elif got[k] < want * 0.5:
            bad.append(f"{k}={got[k]}/{want}")
    detail = " ".join(f"{k}={got[k]}/{exp[k]}" for k in exp)
    if "sanrenpuku=0" in bad and fetch:
        detail += " | " + _diagnose_sanrenpuku(runner, rep.race_id)
    if bad:
        rep.add(label, "FAIL", f"{detail} 不足: {', '.join(bad)}")
        return
    # 3連複は combo が 3 頭・昇順・重複なしであること
    for c in (doc.get("sanrenpuku") or [])[:50]:
        combo = c.get("combo") or []
        if len(combo) != 3 or sorted(set(combo)) != list(combo):
            rep.add(label, "FAIL", f"sanrenpuku combo 不正: {combo}")
            return
    rep.add(label, "PASS", detail)


def _diagnose_sanrenpuku(runner, race_id: str) -> str:
    """type=7 の生応答キー構造を返す（パーサ修正の手掛かり）。"""
    try:
        url = runner.odds_parser.ODDS_API.format(race_id=race_id, type=7)
        d = runner.client._session.get(url, timeout=15).json()
        odds = (d.get("data") or {}).get("odds") or {}
        sample = {k: (list(v.items())[:2] if isinstance(v, dict) else v) for k, v in odds.items()}
        return f"raw status={d.get('status')!r} odds keys={list(odds)} sample={json.dumps(sample, ensure_ascii=False)[:300]}"
    except Exception as e:  # noqa: BLE001
        return f"診断失敗: {e}"


def scenario_snapshots(rep: RaceReport, storage) -> None:
    for cat in ("race_odds", "race_pair_odds"):
        doc = storage.load(cat, rep.race_id) or {}
        phases = [s.get("phase") for s in doc.get("snapshots", [])]
        if doc.get("phase") != "t15":
            rep.add(f"S5 snapshots/{cat}", "FAIL", f"最新 phase={doc.get('phase')!r}（t15 期待）")
        elif "t60" not in phases:
            rep.add(f"S5 snapshots/{cat}", "FAIL", f"履歴に t60 が無い（snapshots phases={phases}）")
        else:
            rep.add(f"S5 snapshots/{cat}", "PASS", f"latest=t15 history={phases}")


def scenario_schema(rep: RaceReport, storage) -> None:
    from src.scraper import schemas

    for cat in ("race_odds", "race_pair_odds"):
        doc = storage.load(cat, rep.race_id)
        if not doc:
            rep.add(f"S6 schema/{cat}", "SKIP", "データなし")
            continue
        r = schemas.validate(cat, doc)
        if r["passed"]:
            rep.add(f"S6 schema/{cat}", "PASS")
        else:
            rep.add(f"S6 schema/{cat}", "FAIL", str(r.get("violations"))[:300])


def scenario_recommend(rep: RaceReport, storage) -> None:
    """取得したオッズ + 既存の予測結果（race_predictions）から推奨馬券を計算する。"""
    from src.api.v1.delegates import optimize_betting

    if not storage.load("race_predictions", rep.race_id):
        rep.add("S7 recommend", "SKIP", "race_predictions が無い（先に AI 予測を実行すると検証できる）")
        return
    for mode in ("auto", "actual", "predicted"):
        body = {"race_id": rep.race_id, "bankroll": 100000, "odds_mode": mode,
                "bet_types": ["tansho", "fukusho", "umaren", "wide", "sanrenpuku"]}
        try:
            payload, code = optimize_betting(body)
        except Exception as e:  # noqa: BLE001
            rep.add(f"S7 recommend/{mode}", "FAIL", f"例外: {e}")
            continue
        if code != 200:
            status = "WARN" if mode == "predicted" and code == 404 else "FAIL"
            rep.add(f"S7 recommend/{mode}", status, f"HTTP {code}: {payload.get('error')}")
            continue
        kinds = sorted({c["bet_type"] for c in payload["candidates"]})
        rep.add(f"S7 recommend/{mode}", "PASS",
                f"source={payload.get('odds_source')} phase={payload.get('odds_phase')} 候補{len(payload['candidates'])}件 券種={kinds}")


# ── 実行制御 ────────────────────────────────────────────────────────────────


def pick_races(runner, date: str, max_races: int) -> list[str]:
    """日付のレース一覧から、発走が遅い方（まだ買える可能性が高い）を優先して返す。"""
    races = runner.scrape_race_list(date) or []
    ids = [r["race_id"] for r in races if r.get("race_id")]
    return ids[::-1][:max_races]


def run_race(race_id: str, storage, runner, *, fetch: bool, t60_to_t15_sleep: float,
             timing: dict[str, Any] | None = None) -> RaceReport:
    rep = RaceReport(race_id)
    card = scenario_card(rep, storage, runner, fetch)
    if card is None:
        return rep
    n = len(_entries(card))
    if timing is not None:
        # S8 を先に実行（phase 無しで上書き保存するので、履歴を検証する S3〜S5 より前に置く）
        tm = Timing()
        tm.install(runner)
        summary = scenario_timing(rep, runner, tm, timing["repeat"], timing["gap"])
        est = estimate_lead(summary, same_time_races=timing["same_time_races"],
                            predict_sec=timing["predict_sec"], margin_sec=timing["margin_sec"])
        timing["result"] = (summary, est, tm)
        if timing["only"]:
            return rep
    scenario_odds(rep, storage, runner, fetch, "t60", n)
    scenario_pair(rep, storage, runner, fetch, "t60", n)
    if fetch:
        # 実運用の T-60 → T-15 を再現（間隔を空けて同一phaseでない再取得を行う）
        time.sleep(t60_to_t15_sleep)
        scenario_odds(rep, storage, runner, fetch, "t15", n)
        scenario_pair(rep, storage, runner, fetch, "t15", n)
    scenario_snapshots(rep, storage)
    scenario_schema(rep, storage)
    scenario_recommend(rep, storage)
    return rep


RACE_POINTS: list[tuple[str, Callable[..., Any]]] = [
    ("scrape_race_card", lambda r, rid: r.scrape_race_card(rid, skip_existing=False)),
    ("scrape_shutuba_past", lambda r, rid: r.scrape_shutuba_past(rid, skip_existing=False)),
    ("scrape_oikiri", lambda r, rid: r.scrape_oikiri(rid, skip_existing=False)),
    ("scrape_odds", lambda r, rid: r.scrape_odds(rid, skip_existing=False)),
    ("scrape_pair_odds", lambda r, rid: r.scrape_pair_odds(rid, skip_existing=False)),
    ("scrape_speed_index", lambda r, rid: r.scrape_speed_index(rid, skip_existing=False)),
    ("scrape_paddock", lambda r, rid: r.scrape_paddock(rid, skip_existing=False)),
    ("scrape_barometer", lambda r, rid: r.scrape_barometer(rid, skip_existing=False)),
    ("scrape_trainer_comment", lambda r, rid: r.scrape_trainer_comment(rid, skip_existing=False)),
    ("scrape_race_result", lambda r, rid: r.scrape_race_result(rid, skip_existing=False)),
    ("scrape_race_result_on_time", lambda r, rid: r.scrape_race_result_on_time(rid, skip_existing=False)),
    ("scrape_race_result_lap", lambda r, rid: r.scrape_race_result_lap(rid, skip_existing=False)),
    ("scrape_race_detail", lambda r, rid: r.scrape_race_detail(rid)),
    ("scrape_smartrc", lambda r, rid: r.scrape_smartrc(rid)),
]
HORSE_POINTS: list[tuple[str, Callable[..., Any]]] = [
    ("scrape_horse", lambda r, hid: r.scrape_horse(hid, skip_existing=False)),
    ("scrape_horse_pedigree_5gen", lambda r, hid: r.scrape_horse_pedigree_5gen(hid, skip_existing=False)),
    ("scrape_horse_training", lambda r, hid: r.scrape_horse_training(hid, skip_existing=False, force=True)),
    ("scrape_horse_mating_date", lambda r, hid: r.scrape_horse_mating_date(hid, skip_existing=False)),
]


def measure_all(runner, storage, race_id: str, repeat: int, gap: float, n_horses: int) -> int:
    """多数のデータ点を repeat 回ずつ取得する。計測の記録は timing_stats が自動で行う（ここでは呼ぶだけ）。"""
    card = runner.scrape_race_card(race_id, skip_existing=False) or storage.load("race_shutuba", race_id) or {}
    horse_ids = [e["horse_id"] for e in _entries(card) if e.get("horse_id")][:n_horses]
    calls = 0
    for i in range(repeat):
        for name, fn in RACE_POINTS:
            try:
                fn(runner, race_id)
            except Exception as e:  # noqa: BLE001 — 失敗も ok=false として記録済み。続行する
                print(f"  ! {name}: {type(e).__name__}: {e}", file=sys.stderr)
            calls += 1
        for hid in horse_ids:
            for name, fn in HORSE_POINTS:
                try:
                    fn(runner, hid)
                except Exception as e:  # noqa: BLE001
                    print(f"  ! {name}({hid}): {type(e).__name__}: {e}", file=sys.stderr)
                calls += 1
        print(f"  [{i + 1}/{repeat}] 完了（累計 {calls} 呼び出し）", file=sys.stderr)
        if i < repeat - 1:
            time.sleep(gap)
    return calls


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="実データ取得シナリオテスト（手動）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--date", help="開催日 YYYYMMDD（その日の遅い発走から --max-races 件）")
    g.add_argument("--race-id", help="対象 race_id（12桁）")
    ap.add_argument("--max-races", type=int, default=1)
    ap.add_argument("--yes", action="store_true", help="実アクセスを許可（KEIBA_RUN_LIVE_SCRAPE_TESTS=1 でも可）")
    ap.add_argument("--offline", action="store_true", help="ネットへ出ず保存済みデータだけ検証")
    ap.add_argument("--capture-dev", action="store_true", help="取得データを dev ストアに残す（KEIBA_ENV=dev 必須）")
    ap.add_argument("--t60-to-t15-sleep", type=float, default=5.0, help="t60 と t15 取得の間隔（秒）")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--measure-all", action="store_true", help="多数のデータ点の実測時間を蓄積する")
    ap.add_argument("--horses", type=int, default=2, help="--measure-all で計測する馬の頭数")
    ap.add_argument("--timing-only", action="store_true", help="S8（所要時間計測）だけ実行")
    ap.add_argument("--timing", action="store_true", help="S3〜S7 に加えて S8 も実行")
    ap.add_argument("--repeat", type=int, default=5, help="S8 の計測回数")
    ap.add_argument("--timing-gap", type=float, default=3.0, help="S8 の各回の間隔（秒）")
    ap.add_argument("--same-time-races", type=int, default=3, help="T-X 見積り: 同時刻に重なる最大レース数")
    ap.add_argument("--predict-sec", type=float, default=0.0, help="T-X 見積り: AI 予測の所要秒")
    ap.add_argument("--margin-sec", type=float, default=120.0, help="T-X 見積り: 余裕秒")
    args = ap.parse_args(argv)
    if args.timing_only:
        args.timing = True

    _load_dotenv()
    fetch = not args.offline
    if fetch and not (args.yes or os.environ.get("KEIBA_RUN_LIVE_SCRAPE_TESTS") == "1"):
        print("実アクセスを伴います。--yes か KEIBA_RUN_LIVE_SCRAPE_TESTS=1 を指定してください。", file=sys.stderr)
        return 2
    env = (os.environ.get("KEIBA_ENV") or "").strip().lower()
    if fetch and env != "dev":
        print(f"KEIBA_ENV={env!r}: 取得データが GCS に保存されてしまうため dev 以外では実行しません。", file=sys.stderr)
        return 2

    tmp_dir = None
    if fetch and not args.capture_dev:
        # 既存のモックデータを上書きしないよう、使い捨ての dev ストアに取得する
        import tempfile

        tmp_dir = tempfile.mkdtemp(prefix="keiba_live_scenarios_")
        os.environ["KEIBA_DEV_MOCK_DIR"] = tmp_dir

    from src.scraper.storage import HybridStorage

    storage = HybridStorage()
    runner = None
    if fetch:
        from src.scraper.run import ScraperRunner

        runner = ScraperRunner()
    else:
        class _Offline:  # 実アクセス系は呼ばれない（fetch=False 分岐）
            pass

        runner = _Offline()

    race_ids = [args.race_id] if args.race_id else pick_races(runner, args.date, args.max_races) if fetch else []
    if not race_ids:
        print("対象レースがありません（--offline では --race-id が必須）", file=sys.stderr)
        return 2

    if args.measure_all:
        if not fetch:
            print("--measure-all は実アクセスが必要です（--offline 不可）", file=sys.stderr)
            return 2
        for rid in race_ids:
            n = measure_all(runner, storage, rid, args.repeat, args.timing_gap, args.horses)
            print(f"{rid}: {n} 呼び出しを計測・蓄積しました。集計: python -m src.scraper.timing_stats report --days 1")
        return 0

    timings: dict[str, dict[str, Any]] = {}
    reports = []
    for rid in race_ids:
        topt = None
        if args.timing and fetch:
            topt = {"repeat": args.repeat, "gap": args.timing_gap, "only": args.timing_only,
                    "same_time_races": args.same_time_races, "predict_sec": args.predict_sec, "margin_sec": args.margin_sec}
            timings[rid] = topt
        reports.append(run_race(rid, storage, runner, fetch=fetch, t60_to_t15_sleep=args.t60_to_t15_sleep, timing=topt))

    saved: dict[str, str] = {}
    for rid, topt in timings.items():
        if "result" in topt:
            summary, est, tm = topt["result"]
            saved[rid] = str(save_timing(rid, summary, est, tm))

    if args.json:
        out = []
        for r in reports:
            item = {"race_id": r.race_id, "results": [asdict(x) for x in r.results]}
            if r.race_id in timings and "result" in timings[r.race_id]:
                item["timing"] = {"summary": timings[r.race_id]["result"][0], "estimate": timings[r.race_id]["result"][1],
                                  "saved_to": saved.get(r.race_id)}
            out.append(item)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        for r in reports:
            print(f"\n=== {r.race_id} ===")
            for x in r.results:
                print(f"  [{x.status:4}] {x.scenario}: {x.detail}")
            if r.race_id in timings and "result" in timings[r.race_id]:
                print_timing(timings[r.race_id]["result"][0], timings[r.race_id]["result"][1])
                print(f"  計測データ保存: {saved.get(r.race_id)}")
    if tmp_dir:
        print(f"\n（取得データは使い捨て領域 {tmp_dir} に保存。dev の UI に残すなら --capture-dev）")
    elif fetch:
        print("\n（取得データを dev ストア data/dev_mock に保存しました。UI は NEXT_PUBLIC_MOCK なしで実データ表示になります）")
    return 1 if any(r.failed for r in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
