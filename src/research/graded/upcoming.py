"""重賞の「データで予想」の前日の計算（2 段目）: 出馬表の確定した重賞に、凍結した B の規則と前日版の ◎ を当てはめる。

流れ（日ごとに 1 回。ホストの手元だけで、GCS は読まない。1 回 25 分ほど）:
  1. 出馬表の行（race_result_flat と同じ列。結果の列は空）を <work>/<日付>/entries.parquet に書く
  2. folklore の master を、その日の前日までの手元の結果 ＋ 出馬表の行で作る（FOLKLORE_ENTRIES・FOLKLORE_CUTOFF）
  3. frame・水準・集計（tally）を作る（どれも as-of。出馬表の行は数えない）
  4. run picks_upcoming で、<GRADED_PICKS_DIR>/races/<race_id>.json に該当馬を書く
  5. 保存済みのめぐ分析に「データで予想」タブを入れ直す（graded_picks.patch_saved）

使い方:
  python -m src.research.graded.upcoming --races 202605050811,...          # 出馬表（<GRADED_PICKS_DIR>/entries/<race_id>.json）から
  python -m src.research.graded.upcoming --sim --races 202608040911         # 試し: 結果のある過去の回を「出馬表の段階」として計算し、
                                                                            #       記録（history.json）の該当馬と比べる（リークの検査）
entries-sync からは save_entries() と kick()（裏で 1 本だけ起動。失敗しても出馬表の同期は止めない）。

分かっている限界（前日の計算と、結果のある表で作った記録との違い）:
- 当日の値（馬体重・馬場・騎手のその日の勝ち数など）の型を使う規則は計算しない。
- ブリンカー（G08）は SmartRC のレースの表から取るので、出馬表の段階では「不明」になる（前日版の点が記録と少しずれうる）。
- 前日（土曜）の結果は翌朝の書き出し（export_tables_daily）まで手元の表に無いので、日曜の重賞は金・土の時点の結果まで（data_until）で計算する。
  開催の何日目（H08）も 1 日少なくなりうる。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[3]
JST = timezone(timedelta(hours=9))
FLAT_COLS = ["race_id", "date", "venue", "venue_code", "surface", "distance", "direction", "grade", "race_class", "weather", "track_condition",
             "start_time", "field_size", "race_name", "round", "pace", "lap_times", "finish_position", "bracket_number", "horse_number", "horse_id",
             "horse_name", "sex_age", "jockey_weight", "jockey_name", "jockey_id", "time_sec", "passing_order", "last_3f", "odds", "popularity",
             "weight", "weight_change", "trainer_id", "weight_rule"]


def picks_dir() -> Path:
    return Path(os.environ.get("GRADED_PICKS_DIR") or REPO / "data/local/graded_picks")


ENTRY_KEYS = ("horse_number", "bracket_number", "horse_name", "horse_id", "sex_age", "jockey_weight", "jockey_name", "jockey_id", "trainer_id")
CARD_KEYS = ("race_id", "race_name", "date", "venue", "surface", "distance", "direction", "race_class", "start_time", "round", "weight_rule")


def entries_sig(race_id: str) -> str | None:
    """書き置いた出馬表（<GRADED_PICKS_DIR>/entries/<race_id>.json）の指紋。無ければ None。"""
    import hashlib
    try:
        return hashlib.sha256((picks_dir() / "entries" / f"{race_id}.json").read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def confirmed(card: dict) -> bool:
    """出馬表が確定しているか（馬番が全頭に付き、重ならない）。数えるのは計算に使える馬（horse_id のある馬）だけ。"""
    hs = [e for e in (card.get("entries") or []) if isinstance(e, dict) and e.get("horse_id")]
    nums = [int(e.get("horse_number") or 0) for e in hs]
    return bool(hs) and all(n > 0 for n in nums) and len(set(nums)) == len(nums)


def rows_from_card(card: dict, grade: str, race_name: str | None = None) -> list[dict]:
    """出馬表（race_shutuba）から、race_result_flat と同じ列の行を作る。格は記録の重賞の格（出馬表の grade は当てにならない）。
    レース名は記録の重賞の鍵（結果の表の名前。出馬表の略称のままだと、主の層・規則の重賞が引けない）。
    オッズ・人気・馬体重・馬場など、結果や当日に決まる値は入れない。"""
    hs = [e for e in card.get("entries") or [] if isinstance(e, dict) and e.get("horse_id")]
    out = []
    for e in hs:
        out.append({"race_id": str(card.get("race_id")), "date": str(card.get("date"))[:10], "venue": card.get("venue"),
                    "venue_code": str(card.get("race_id"))[4:6], "surface": card.get("surface"), "distance": card.get("distance"),
                    "direction": card.get("direction"), "grade": grade, "race_class": card.get("race_class"), "start_time": card.get("start_time"),
                    "field_size": len(hs), "race_name": race_name or card.get("race_name"), "round": card.get("round"),
                    "bracket_number": e.get("bracket_number"), "horse_number": e.get("horse_number"), "horse_id": str(e.get("horse_id")),
                    "horse_name": e.get("horse_name"), "sex_age": e.get("sex_age"), "jockey_weight": e.get("jockey_weight"),
                    "jockey_name": e.get("jockey_name"), "jockey_id": e.get("jockey_id"), "trainer_id": e.get("trainer_id"),
                    "weight_rule": card.get("weight_rule")})
    return out


def entries_frame(race_ids: list[str], sim: bool) -> Any:
    """race_ids の出馬表の行（sim なら race_result_flat の行。結果の列は folklore が空にする）。"""
    import pandas as pd

    from src.research.race_analysis.graded_picks import load_history, rkey_of
    h = load_history() or {"rkeys": {}}
    rows = []
    if sim:
        from src.research.graded.features import TABLES
        for y in sorted({r[:4] for r in race_ids}):
            f = pd.read_parquet(TABLES / y / "race_result_flat.parquet")
            f["race_id"] = f["race_id"].astype(str)
            f = f[f["race_id"].isin(race_ids)].copy()
            # 試しでは、発走しなかった馬（取消・除外）を外して記録と同じ頭にそろえる（本番の前日の計算では、出馬表の全頭で計算する）
            f = f[pd.to_numeric(f["finish_position"], errors="coerce").notna() | (pd.to_numeric(f["odds"], errors="coerce") > 0)]
            for rid, g in f.groupby("race_id"):
                rk = rkey_of(str(g["race_name"].iloc[0]), str(g["date"].iloc[0]))
                g = g.assign(grade=h["rkeys"][rk]["grade"] if rk else g["grade"])
                rows += g.to_dict("records")
    else:
        for rid in race_ids:
            p = picks_dir() / "entries" / f"{rid}.json"
            card = json.loads(p.read_text(encoding="utf-8"))
            rk = rkey_of(card.get("race_name"), card.get("date"))
            if not rk or not confirmed(card):
                logger.warning("重賞でないか、出馬表が確定していない: %s", rid)
                continue
            rows += rows_from_card(card, h["rkeys"][rk]["grade"], rk)
    e = pd.DataFrame(rows)
    if len(e):
        e = e[[c for c in FLAT_COLS if c in e.columns]]
        e["date"] = e["date"].astype(str).str[:10]
    return e


def _run(cmd: list[str], env: dict, log) -> None:
    t0 = time.time()
    r = subprocess.run(cmd, env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    log.write(f"\n== {' '.join(cmd[2:])}: {time.time() - t0:.0f} 秒 rc={r.returncode}\n".encode())
    log.flush()
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[2:])} が失敗（rc={r.returncode}）")


def run(race_ids: list[str], sim: bool = False, work: Path | None = None) -> dict[str, Any]:
    """日ごとに master → frame・tally → 該当馬。戻り値は日ごとの所要時間と書いたレース。"""
    work = work or picks_dir() / "work"
    e = entries_frame(race_ids, sim)
    if not len(e):
        return {"races": 0}
    out: dict[str, Any] = {"days": {}, "failed": {}}
    py = sys.executable
    sig = json.dumps({rid: entries_sig(rid) for rid in e["race_id"].unique()}) if not sim else "{}"
    ok_ids: list[str] = []
    for date, g in e.groupby("date"):
        t0 = time.time()
        wd = work / "sim_" / date if sim else work / date
        (wd / "master").mkdir(parents=True, exist_ok=True)
        ep = wd / "entries.parquet"
        g.to_parquet(ep, index=False)
        po = wd / "picks" if sim else picks_dir() / "races"
        env = dict(os.environ, FOLKLORE_OUT=str(wd / "master"), FOLKLORE_YEAR_END=date[:4], FOLKLORE_KEEP_DNF="1", FOLKLORE_RESUME="0",
                   FOLKLORE_ENTRIES=str(ep), FOLKLORE_CUTOFF=date, GRADED_OUT=str(wd), GRADED_ENTRIES=str(ep), GRADED_PICKS_OUT=str(po),
                   GRADED_ENTRIES_SIG=sig, PYTHONPATH=str(REPO), **({"GRADED_RULES_VERSIONS": str(wd / "rules_versions.json")} if sim else {}), OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")   # レース日の取得・予測と CPU を取り合わない
        ids = sorted(g["race_id"].unique())
        try:
            with open(wd / "run.log", "ab") as log:
                log.write(f"\n==== {datetime.now(JST).isoformat(timespec='seconds')} {date} races={ids} sim={sim}\n".encode())
                _run([py, "-m", "src.scripts.data.folklore_build"], env, log)
                _run([py, "-m", "src.research.graded.run", "frame"], env, log)
                _run([py, "-m", "src.research.graded.run", "tally"], env, log)
                _run([py, "-m", "src.research.graded.run", "picks_upcoming"], env, log)
        except Exception as exc:   # ある日が失敗しても、ほかの日は続ける
            logger.exception("データで予想の前日の計算に失敗 %s %s", date, ids)
            out["failed"][date] = str(exc)
            continue
        if not sim:   # 大きな中間ファイルは消す（ログは残す）
            import shutil
            shutil.rmtree(wd / "master", ignore_errors=True)
            for f in ("frame.parquet", "levels.parquet", "tally.parquet", "tally_frozen.parquet"):
                (wd / f).unlink(missing_ok=True)
        ok_ids += ids
        out["days"][date] = {"seconds": round(time.time() - t0), "races": ids, "out": str(po)}
        logger.info("データで予想の前日の計算 %s: %d 秒 %s", date, round(time.time() - t0), ids)
    if not sim and ok_ids:
        from src.research.race_analysis.graded_picks import patch_saved
        out["patched"] = patch_saved(ok_ids)
    return out


def compare_sim(race_ids: list[str], work: Path | None = None) -> list[dict]:
    """試しの計算の該当馬と、記録（結果のある表で作った history.json）の該当馬を比べる。一致しなければ、点に未来の値が入っている（または
    出馬表の段階で欠ける値がある）おそれがある。"""
    from src.research.race_analysis.graded_picks import load_history
    h = load_history() or {"rkeys": {}}
    by = {x["race_id"]: x for v in h["rkeys"].values() for x in v["races"]}
    work = work or picks_dir() / "work"
    res = []
    for p in sorted(work.glob("sim_/*/picks/*.json")):
        if p.stem not in race_ids:
            continue
        s = json.loads(p.read_text(encoding="utf-8"))
        x = by.get(p.stem)
        if x is None:
            continue
        r = {"race_id": p.stem, "rkey": s["rkey"], "base": (s["base"]["hn"], x["base"]["hn"])}
        if s["B"]["honmei"] is not None:
            r["B_honmei"] = (s["B"]["honmei"]["hn"], x["B"]["honmei"]["hn"])
        if s["B"]["kill"] is not None:
            r["B_kill"] = (sorted(q["hn"] for q in s["B"]["kill"]), sorted(q["hn"] for q in x["B"]["kill"] or []))
        r["match"] = all(r[k][0] == r[k][1] for k in ("base", "B_honmei", "B_kill") if k in r)
        res.append(r)
    return res


LOCK = "upcoming.lock"


def _lock_alive(lock: Path) -> bool:
    """ロック（中身は計算のプロセス番号）の計算がまだ動いているか。番号の無い・読めない・終わったロックは古いとみなす。"""
    try:
        pid = int(lock.read_text(encoding="utf-8").split()[0])
        os.kill(pid, 0)
        return True
    except (OSError, ValueError, IndexError):
        return False


def stale(race_id: str) -> bool:
    """書き置いた出馬表があり、その出馬表での計算がまだ無いか（出馬表が変わった・計算が失敗した・まだ動かしていない）。"""
    sig = entries_sig(race_id)
    if sig is None:
        return False
    try:
        x = json.loads((picks_dir() / "races" / f"{race_id}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True
    return x.get("entries_sig") != sig


def kick(race_ids: list[str] | None = None, today: str | None = None) -> dict[str, Any]:
    """entries-sync から呼ぶ: 計算がまだの重賞（今回取り直したレースと、書き置き済みで今日以降のレース）の計算を、裏で 1 本だけ起動する（待たない）。
    動いている間は起動しない（次の同期で拾う）。"""
    d = picks_dir()
    today = today or datetime.now(JST).strftime("%Y-%m-%d")
    cand = set(race_ids or [])
    for p in (d / "entries").glob("*.json") if (d / "entries").exists() else []:
        try:
            if str(json.loads(p.read_text(encoding="utf-8")).get("date"))[:10] >= today:
                cand.add(p.stem)
        except (OSError, ValueError):
            continue
    todo = sorted(r for r in cand if stale(r))
    if not todo:
        return {"kicked": []}
    lock = d / LOCK
    if lock.exists() and _lock_alive(lock):
        return {"kicked": [], "running": True, "waiting": todo}
    log = open(d / "upcoming.log", "ab")
    proc = subprocess.Popen(["nice", "-n", "10", sys.executable, "-m", "src.research.graded.upcoming", "--races", ",".join(todo), "--unlock"],
                            cwd=REPO, env=dict(os.environ, PYTHONPATH=str(REPO)), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    lock.write_text(f"{proc.pid} {datetime.now(JST).isoformat(timespec='seconds')}", encoding="utf-8")
    return {"kicked": todo}


def _ymd(s: Any) -> str:
    s = str(s or "")
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 and s.isdigit() else s[:10]


def save_entries(storage: Any, races: list[dict]) -> list[str]:
    """出馬表の確定した重賞のカードを <GRADED_PICKS_DIR>/entries/<race_id>.json に書く（計算に使う列だけ。中身が変わったときだけ）。
    races は entries-sync の計画の行（race_id・race_name・date・list_grade_icon）。重賞の名前か、一覧の格のアイコンが平地の重賞の行だけ
    カードを読む（ほかのレースは読まない）。一覧の名前は出馬表とも結果とも違う略称のことがある（2026-11-23 の「東スポ2歳S」。
    出馬表は「東スポ杯2歳S」）ので、名前で当たらなくてもアイコンで拾い、カードの名前で重賞かを決める。書いた race_id を返す。"""
    from src.api.race_calendar import grade_of
    from src.research.race_analysis.graded_picks import rkey_of
    from src.research.race_analysis.service import _load_card
    out = []
    d = picks_dir() / "entries"
    for r in races:
        rid = str(r.get("race_id"))
        if not rkey_of(r.get("race_name"), _ymd(r.get("date"))) and grade_of({"list_grade_icon": r.get("list_grade_icon"), "race_name": r.get("race_name")}) not in ("G1", "G2", "G3"):
            continue
        card, _ = _load_card(storage, rid)
        if not card or not rkey_of(card.get("race_name"), card.get("date")) or not confirmed(card):
            continue
        card = dict(card, race_id=rid)
        body = {k: card.get(k) for k in CARD_KEYS}
        body["entries"] = [{k: e.get(k) for k in ENTRY_KEYS} for e in card.get("entries") or [] if isinstance(e, dict) and e.get("horse_id")]
        body = json.dumps(body, ensure_ascii=False, sort_keys=True, default=str)
        p = d / f"{rid}.json"
        if p.exists() and p.read_text(encoding="utf-8") == body:
            continue
        d.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, p)
        out.append(rid)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--races", required=True)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--unlock", action="store_true", help="終わったらロックを外す（kick から起動したとき）")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ids = [x for x in a.races.split(",") if x]
    t0 = time.time()
    try:
        res = run(ids, sim=a.sim)
        if a.sim:
            res["compare"] = compare_sim(ids)
        print(json.dumps(res, ensure_ascii=False, default=str), flush=True)
    finally:
        logger.info("データで予想の前日の計算: 全体 %d 秒", round(time.time() - t0))
        if a.unlock:
            try:
                (picks_dir() / LOCK).unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
