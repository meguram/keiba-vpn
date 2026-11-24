"""影のランナー（candidate_runner）の材料の予約: 開催の暦から、前夜の前段 S3 と、当日の朝の RB の状態の作り直しを決める。

    python -m src.pipeline.inference.candidate_rb_schedule eve       # 前夜（18:30 JST）: 翌日が開催日なら「日<TAB>めぐ指数を待つ日<TAB>期限」
    python -m src.pipeline.inference.candidate_rb_schedule morning   # 当日の朝（07:05〜07:40 JST）: 「動き<TAB>MMDD<TAB>理由」

- それまでの予約（retrain_out/cycles16_fix3/rb_live/schedule.sh）は日付を手で書く一度きりの形で、11/23（祝日）の分が抜けていた。
  ここでは JRA の暦（data/page_reference/cushion/race_calendar_<年>.json。祝日・月曜の開催を含む）を読む。
- 前夜の 18:30 に RB を作っても、その日の結果は翌朝 07:00 の export_tables_daily（速報から race_result_flat への橋渡し）まで表に入らない。
  そのため 11/22・23 の朝に cut が古く、影のランナーが止まりかけた（StaleInputs）。朝は「前の開催日の全レースが結果の表に入った」ら作り直す。
- 週の最初の開催日の前夜は、めぐ指数の週次（金 18:00 開始・書き終わり 18:45〜23:24）が前の週の開催日を書くのを待つ（eve.sh の第 3・4 引数）。
- 読むのは手元のファイルだけ（GCS は読まない。費用 0）。判断は純粋関数（テスト: tests/pipeline/test_candidate_rb_schedule.py）。
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
JST = timezone(timedelta(hours=9))
CAL_DIR = Path(os.environ.get("KEIBA_RACE_CALENDAR_DIR") or ROOT / "data/page_reference/cushion")
TABLES = Path(os.environ.get("KEIBA_TABLES_DIR") or ROOT / "data/page_reference/tables")
RB_STATE = Path(os.environ.get("CANDIDATE_RB_STATE") or "/home/jovyan/work/retrain_out/cycles16_fix3/rb_live/state_live")
MISSING_OK = 2              # 結果がそろったとみなす欠けの上限（中止のレース）
MEGU_WAIT_DEADLINE = "03:00"     # めぐ指数を待つ期限（対象日の JST）。期限を過ぎたら入っていなくても作る（eve.sh の決まり）


def race_days(years: list[int], cal_dir: Path = CAL_DIR) -> list[date]:
    """暦の開催日（中止は暦から消える前提。暦が無い年は空）。"""
    out: set[date] = set()
    for y in years:
        p = cal_dir / f"race_calendar_{y}.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text(encoding="utf-8")).get("race_days") or []:
            try:
                out.add(date.fromisoformat(str(r["date"])[:10]))
            except (KeyError, ValueError):
                continue
    return sorted(out)


def eve_target(today: date, days: list[date]) -> date | None:
    """前夜: 翌日が開催日ならその日。"""
    nxt = today + timedelta(days=1)
    return nxt if nxt in days else None


def megu_wait_days(target: date, days: list[date]) -> list[date]:
    """週の最初の開催日（前の開催日から 3 日以上あく）の前夜だけ、前の週の開催日のめぐ指数を待つ。続きの開催日の前夜は待たない
    （その日の結果は翌朝まで表に入らず、めぐ指数の週次も金曜だけ）。"""
    prev = [d for d in days if d < target]
    if not prev or (target - prev[-1]).days < 3:
        return []
    last = prev[-1]
    run = [last]
    for d in reversed(prev[:-1]):             # 前の週のひと続きの開催日（土日・祝日の月曜）
        if (run[-1] - d).days > 1:
            break
        run.append(d)
    return sorted(run)


def morning_action(today: date, days: list[date], cut: str | None, done_through: str | None) -> tuple[str, str]:
    """当日の朝の判断。返り値は (動き, 理由)。動きは rebuild（作り直す）・wait（表がまだ）・skip（不要）。
    done_through は「全レースの結果が表に入った最後の日」。前の開催日に届くまでは待つ（作っても cut が古い・一部のレースが欠ける）。"""
    if today not in days:
        return "skip", "開催日でない"
    prev = [d for d in days if d < today]
    if not prev:
        return "skip", "前の開催日が暦に無い"
    need = prev[-1].isoformat()
    if cut and str(cut)[:10] >= need:
        return "skip", f"cut {cut} は前の開催日 {need} 以降"
    if not done_through or str(done_through)[:10] < need:
        return "wait", f"前の開催日 {need} の結果が表にそろっていない（そろった最後の日 {done_through}）"
    return "rebuild", f"cut {cut} → 前の開催日 {need} の結果がそろった"


def read_cut(state: Path = RB_STATE) -> str | None:
    try:
        return json.loads((state / "params.json").read_text(encoding="utf-8")).get("cut")
    except (OSError, ValueError):
        return None


def _day_races(path: Path, day: str) -> set[str] | None:
    import pandas as pd

    if not path.exists():
        return None
    d = pd.read_parquet(path, columns=["race_id", "date"])
    ds = pd.to_datetime(d["date"].astype(str).str.replace(r"[./]", "-", regex=True), errors="coerce")
    return set(d.loc[ds == pd.Timestamp(day), "race_id"].astype(str))


def results_complete(day: date, tables: Path = TABLES) -> bool:
    """その日の全レースの結果が race_result_flat に入ったか。数は出馬表（race_shutuba_flat）のレースと比べる。
    出馬表に無い（読めない）ときは、結果が 1 レースでもあれば入った扱い。出馬表の後に中止になったレースは結果に入らないので、
    欠けが MISSING_OK 件までならそろった扱いにする（橋渡しは 1 日分をまとめて入れるので、一部だけのときは欠けがもっと多い）。"""
    y = tables / str(day.year)
    res = _day_races(y / "race_result_flat.parquet", day.isoformat()) or set()
    card = _day_races(y / "race_shutuba_flat.parquet", day.isoformat())
    if not res:
        return False
    return not card or len(card - res) <= MISSING_OK


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else ""
    today = datetime.now(JST).date()
    days = race_days([today.year - 1, today.year, today.year + 1])     # 年の変わり目（1 月の最初の開催日の前の開催日・12 月末の翌日）
    if not any(d.year == today.year for d in days):
        print(f"警告: {today.year} 年の暦（race_calendar_{today.year}.json）が無い。予約は何もしない", file=sys.stderr)
    if mode == "eve":
        t = eve_target(today, days)
        if t:
            w = megu_wait_days(t, days)
            print(f"{t.isoformat()}\t{','.join(d.isoformat() for d in w)}\t{t.isoformat()} {MEGU_WAIT_DEADLINE} JST" if w else t.isoformat())
        return 0
    if mode == "morning":
        prev = [d for d in days if d < today]
        done = prev[-1].isoformat() if prev and results_complete(prev[-1]) else None
        act, why = morning_action(today, days, read_cut(), done)
        print(f"{act}\t{today.strftime('%y%m%d')}\t{why}")
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
