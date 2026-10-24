"""各エンドポイントのデータが「いつ初めて取得できるようになるか」を観測して記録する（SLA の実測）。

「何曜何時に枠順・馬体重・オッズ・追い切り・厩舎コメントが出るか」「結果は発走何分後か」「走行データは何曜何時か」は、
1 回の取得では分からない。このツールを開催前週の水曜〜開催後の金曜まで定期的（30 分ごと、開催日は 10 分ごと）に回し、
対象レース × シグナルごとの first_seen_at を記録する。記録から SLA（`sla_spec.py`）の根拠を更新できる。

  python3 -m src.scraper.availability_watch --auto --yes           # 直近の開催（前後）から対象レースを自動選択して 1 回観測
  python3 -m src.scraper.availability_watch --races 202605040511 --yes
  python3 -m src.scraper.availability_watch --report               # 記録済みの first_seen_at を集計表示

記録先: data/local/meta/availability_watch/{state.json, observations.jsonl}（git 管理外）
dev 専用（KEIBA_ENV=dev）。取得結果は使い捨てディレクトリに保存し、dev ストア・GCS を汚さない。
1 回の観測は 1 レースあたり約 12 リクエスト（既定の取得間隔を守る）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
JST = timezone(timedelta(hours=9))
WATCH_DIR = REPO_ROOT / "data" / "local" / "meta" / "availability_watch"

# シグナル名 → (判定関数名, 説明)。各関数は取得結果 dict から True/False を返す。
SIGNALS: dict[str, str] = {
    "card.entries": "出馬表に出走馬がいる",
    "card.draw_fixed": "出馬表の馬番が全頭確定（枠順確定）",
    "card.weights": "出馬表に馬体重がある（当日）",
    "odds.win": "単勝オッズが出ている（発売開始）",
    "pair_odds.umaren": "馬連オッズが出ている",
    "pair_odds.sanrenpuku": "3連複オッズが出ている",
    "shutuba_past": "馬柱がある",
    "oikiri": "追い切り評価がある",
    "trainer_comment": "厩舎コメント本文がある",
    "paddock": "パドック評価がある",
    "speed_index": "タイム指数がある",
    "barometer": "バロメーター(走行データ)がある",
    "barometer.not_published": "バロメーターは「公開予定」表示（未公開）",
    "result": "レース結果（着順）がある",
    "result_on_time": "速報結果がある",
    "result_lap": "ラップがある",
    "smartrc": "SmartRC のデータがある",
}


def _n(d: Any, key: str = "entries") -> int:
    return len((d or {}).get(key) or []) if isinstance(d, dict) else 0


def evaluate_signals(raw: dict[str, Any]) -> dict[str, bool]:
    """endpoint 名 → 取得結果 から、シグナルごとの真偽を作る（純関数）。"""
    card = raw.get("card") or {}
    entries = card.get("entries") or []
    odds = raw.get("odds") or {}
    pair = raw.get("pair_odds") or {}
    bar = raw.get("barometer") or {}
    comments = (raw.get("trainer_comment") or {}).get("entries") or []
    return {
        "card.entries": len(entries) > 0,
        "card.draw_fixed": len(entries) > 0 and all(int(e.get("horse_number") or 0) > 0 for e in entries),
        "card.weights": any(float(e.get("weight") or 0) > 0 for e in entries),
        "odds.win": any(float(e.get("win_odds") or 0) > 0 for e in (odds.get("entries") or [])),
        "pair_odds.umaren": len(pair.get("umaren") or []) > 0,
        "pair_odds.sanrenpuku": len(pair.get("sanrenpuku") or []) > 0,
        "shutuba_past": _n(raw.get("shutuba_past")) > 0,
        "oikiri": _n(raw.get("oikiri")) > 0,
        "trainer_comment": any(str(e.get("comment") or "").strip() for e in comments),
        "paddock": _n(raw.get("paddock")) > 0,
        "speed_index": _n(raw.get("speed_index")) > 0,
        "barometer": _n(bar) > 0,
        "barometer.not_published": bool(bar.get("not_published")),
        "result": _n(raw.get("result")) > 0,
        "result_on_time": _n(raw.get("result_on_time")) > 0,
        "result_lap": len((raw.get("result_lap") or {}).get("lap_times") or []) > 0 or _n(raw.get("result_lap")) > 0,
        "smartrc": bool(raw.get("smartrc")),
    }


def update_state(state: dict[str, Any], race_id: str, signals: dict[str, bool], now: datetime) -> list[dict[str, Any]]:
    """first_seen_at / last_seen_at を更新し、今回初めて True になった（または消えた）変化を返す（純関数）。"""
    changes: list[dict[str, Any]] = []
    race = state.setdefault("races", {}).setdefault(race_id, {})
    ts = now.isoformat(timespec="seconds")
    for name, val in signals.items():
        s = race.setdefault(name, {"first_seen_at": None, "last_state": False, "last_checked_at": None})
        if val and s["first_seen_at"] is None:
            s["first_seen_at"] = ts
            changes.append({"race_id": race_id, "signal": name, "event": "first_seen", "at": ts})
        elif s["last_state"] and not val:
            changes.append({"race_id": race_id, "signal": name, "event": "disappeared", "at": ts})
        s["last_state"], s["last_checked_at"] = val, ts
    return changes


def observe_race(runner: Any, race_id: str) -> dict[str, Any]:
    """1 レース分の全エンドポイントを取得する（失敗したものは None）。"""
    calls: dict[str, Callable[[], Any]] = {
        "card": lambda: runner.scrape_race_card(race_id, skip_existing=False),
        "odds": lambda: runner.scrape_odds(race_id, skip_existing=False),
        "pair_odds": lambda: runner.scrape_pair_odds(race_id, skip_existing=False),
        "shutuba_past": lambda: runner.scrape_shutuba_past(race_id, skip_existing=False),
        "oikiri": lambda: runner.scrape_oikiri(race_id, skip_existing=False),
        "trainer_comment": lambda: runner.scrape_trainer_comment(race_id, skip_existing=False),
        "paddock": lambda: runner.scrape_paddock(race_id, skip_existing=False),
        "speed_index": lambda: runner.scrape_speed_index(race_id, skip_existing=False),
        "result": lambda: runner.scrape_race_result(race_id, skip_existing=False),
        "result_on_time": lambda: runner.scrape_race_result_on_time(race_id, skip_existing=False),
        "result_lap": lambda: runner.scrape_race_result_lap(race_id, skip_existing=False),
        "smartrc": lambda: runner.scrape_smartrc(race_id),
    }
    raw: dict[str, Any] = {}
    for k, fn in calls.items():
        try:
            raw[k] = fn()
        except Exception as e:  # noqa: BLE001 — 取得失敗は「無い」として扱い、観測は続ける
            raw[k] = None
            print(f"  ! {race_id} {k}: {type(e).__name__}: {e}", file=sys.stderr)
    # バロメーターは「未公開」表示の検知のため、パーサーを直接呼ぶ（公開前の無駄なリクエスト抑止を通さない）
    try:
        raw["barometer"] = runner.barometer_parser.parse_from_api(runner.client._session, race_id)
    except Exception as e:  # noqa: BLE001
        raw["barometer"] = None
        print(f"  ! {race_id} barometer: {type(e).__name__}: {e}", file=sys.stderr)
    return raw


def pick_auto_races(runner: Any, today: datetime) -> list[str]:
    """直近の開催（前後）から観測対象を選ぶ: 今後 7 日の開催日と過去 5 日の開催日それぞれの先頭場の 11R。"""
    out: list[str] = []
    for delta in list(range(-5, 8)):
        d = (today + timedelta(days=delta)).strftime("%Y%m%d")
        try:
            races = runner.scrape_race_list(d) or []
        except Exception:  # noqa: BLE001
            races = []
        if not races:
            continue
        pick = next((r for r in races if int(r.get("round") or 0) == 11), races[len(races) // 2])
        out.append(pick["race_id"])
    return out


def load_state() -> dict[str, Any]:
    p = WATCH_DIR / "state.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def save_state(state: dict[str, Any], changes: list[dict[str, Any]]) -> None:
    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    (WATCH_DIR / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    if changes:
        with open(WATCH_DIR / "observations.jsonl", "a", encoding="utf-8") as f:
            for c in changes:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")


def report(state: dict[str, Any]) -> str:
    lines = ["レース × シグナル の first_seen_at（JST）。未出現は —"]
    for rid, sigs in sorted(state.get("races", {}).items()):
        lines.append(f"\n{rid}")
        for name, s in sigs.items():
            lines.append(f"  {name:<24}{(s['first_seen_at'] or '—'):<26}{SIGNALS.get(name, '')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--auto", action="store_true", help="直近の開催（前後）から対象レースを自動選択")
    g.add_argument("--races", help="race_id をカンマ区切りで指定")
    g.add_argument("--report", action="store_true", help="記録済みの first_seen_at を表示")
    ap.add_argument("--yes", action="store_true", help="実アクセスを許可")
    args = ap.parse_args(argv)

    if args.report:
        print(report(load_state()))
        return 0
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    if (os.environ.get("KEIBA_ENV") or "").strip().lower() != "dev":
        print("KEIBA_ENV=dev 以外では実行しません（取得結果が GCS に書かれるため）", file=sys.stderr)
        return 2
    if not args.yes:
        print("実アクセスを伴います。--yes を指定してください。", file=sys.stderr)
        return 2
    # 取得結果は使い捨て領域へ（dev ストアを汚さない）
    tmp = tempfile.mkdtemp(prefix="keiba_watch_")
    os.environ["KEIBA_DEV_MOCK_DIR"] = os.path.join(tmp, "mock")
    os.environ["KEIBA_PAGE_REFERENCE_DIR"] = os.path.join(tmp, "ref")
    os.environ["KEIBA_SCRAPE_TIMING"] = "steps"

    from src.scraper.run import ScraperRunner

    runner = ScraperRunner()
    now = datetime.now(JST)
    race_ids = [r.strip() for r in args.races.split(",")] if args.races else pick_auto_races(runner, now)
    state, all_changes = load_state(), []
    for rid in race_ids:
        sig = evaluate_signals(observe_race(runner, rid))
        ch = update_state(state, rid, sig, now)
        all_changes += ch
        print(f"{rid}: " + " ".join(f"{k}={'○' if v else '×'}" for k, v in sig.items()) + (f"  ★新規 {[c['signal'] for c in ch if c['event'] == 'first_seen']}" if ch else ""))
    save_state(state, all_changes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
