"""見張りのまとめ: 開催日のレースごとに「T-10 に第 2 段の判定（買い目・見送り）が API で見える状態だったか」を、ホストの手元の記録だけで出す。

読むもの（すべてホストの手元。GCS は読まない・API も叩かない）:
  - 発走時刻: data/local/meta/entries_state/<race_id>.json の start_time
  - 本体の推論の完了: data/local/inference/events/<日>.jsonl（status=ok の最初の at）
  - 影のランナー: data/local/inference/launch_watch/<日>.jsonl（odds_t60_seen・v6c・decision）
  - push の完了: data/local/inference/launch_watch/push_<日>.jsonl（gcs_mirror の push が判定ファイルを上げた時刻）
「API で見える見込み」= push の完了 ＋ 60 秒（Cloud Run の目録の読み直し）。prerender（最大 120 秒）と画面の 45 秒は別に足して判定する。

  python -m src.scripts.data.launch_watch_report --day 20261121 [--json out.json]

画面に出たかは、VPS の nginx・Cloud Run のリクエストのログで別に見る（docs/operations/weekend_launch_20261121.md §5）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_JST = timezone(timedelta(hours=9))
API_LAG = timedelta(seconds=60)
SCREEN_LAG = timedelta(seconds=120 + 45)


def _jsonl(p: Path) -> list[dict]:
    out = []
    try:
        for ln in p.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(ln))
            except ValueError:
                continue
    except OSError:
        pass
    return out


def _t(s) -> datetime | None:
    try:
        return datetime.fromisoformat(str(s)).astimezone(_JST)
    except (TypeError, ValueError):
        return None


def build(day: str, inf: Path, entries_dir: Path) -> list[dict]:
    watch = _jsonl(inf / "launch_watch" / f"{day}.jsonl")
    pushes = _jsonl(inf / "launch_watch" / f"push_{day}.jsonl")
    events = _jsonl(inf / "events" / f"{day}.jsonl")
    rids = sorted({str(x["race_id"]) for x in watch + events if x.get("race_id")} | {r for p in pushes for r in p.get("race_ids", [])})
    first = lambda rows, rid, cond=lambda x: True: min((_t(x["at"]) for x in rows if str(x.get("race_id")) == rid and cond(x) and _t(x.get("at"))), default=None)  # noqa: E731
    rows = []
    for rid in rids:
        st = None
        try:
            es = json.loads((entries_dir / f"{rid}.json").read_text(encoding="utf-8"))
            hh, mm = str(es.get("start_time") or "").split(":")[:2]
            st = datetime(int(day[:4]), int(day[4:6]), int(day[6:]), int(hh), int(mm), tzinfo=_JST)
        except (OSError, ValueError):
            pass
        dec = next((x for x in watch if x.get("event") == "decision" and str(x.get("race_id")) == rid), {})
        pushed = min((_t(p["at"]) for p in pushes if rid in p.get("race_ids", []) and _t(p.get("at"))), default=None)
        api = pushed + API_LAG if pushed else None
        t10 = st - timedelta(minutes=10) if st else None
        rows.append({
            "race_id": rid, "post": st, "main_pred": first(events, rid, lambda x: x.get("status") == "ok"),
            "odds_t60": first(watch, rid, lambda x: x.get("event") == "odds_t60_seen"), "v6c": first(watch, rid, lambda x: x.get("event") == "v6c"),
            "decision": _t(dec.get("at")), "status": dec.get("status"), "skip_reason": dec.get("skip_reason"), "honmei": dec.get("honmei"),
            "pushed": pushed, "api_visible": api,
            "ok_t10_api": (api is not None and t10 is not None and api <= t10),
            "ok_t10_screen": (api is not None and t10 is not None and api + SCREEN_LAG <= t10),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True)
    ap.add_argument("--inference-dir", default=str(ROOT / "data/local/inference"))
    ap.add_argument("--entries-dir", default=str(ROOT / "data/local/meta/entries_state"))
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    rows = build(a.day, Path(a.inference_dir), Path(a.entries_dir))
    f = lambda d: d.strftime("%H:%M:%S") if d else "—"  # noqa: E731
    print("race_id       発走   本体推論  T-60     候補      判定      push     API見込み  T-10API 状態")
    for r in sorted(rows, key=lambda x: (x["post"] or datetime.max.replace(tzinfo=_JST), x["race_id"])):
        print(f"{r['race_id']} {f(r['post'])[:5]}  {f(r['main_pred'])} {f(r['odds_t60'])} {f(r['v6c'])} {f(r['decision'])} {f(r['pushed'])} "
              f"{f(r['api_visible'])}  {'○' if r['ok_t10_api'] else '×'}      {r['status'] or '—'} {r['skip_reason'] or ''}")
    n = len(rows)
    ok = sum(r["ok_t10_api"] for r in rows)
    print(f"\n{a.day}: {n} レース、T-10 に API で見える見込み {ok}/{n}（画面まで含む見込み {sum(r['ok_t10_screen'] for r in rows)}/{n}）。"
          f"買い {sum(r['status'] == 'buy' for r in rows)}・見送り {sum(r['status'] == 'skip' for r in rows)}・判定なし {sum(r['status'] is None for r in rows)}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
