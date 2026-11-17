"""重賞の予想家の材料（使う型・縮める強さ・前日版の点の型・消しの当たりの数・回収率版）の年 1 回の決め直し（計画書 §27、2 段目）。

使い方（ホストの手元だけ。GCS は読まない。1〜2 時間。entries-sync・cron からは起動しない）:
  python -m src.research.graded.material --end 2025 --out /home/jovyan/work/graded_material_2025
  - 区切りの年 Y（--end）までの結果で、発見 2018〜(Y-3)・確認 (Y-2)〜Y に分けて、今の材料と同じ作法で選び直す。
  - 作るもの: <out>/prereg.json・prereg_roi.json・prereg_kill.json（リポジトリの材料は書き換えない）と <out>/material_diff.json（今の材料との違い）。
  - 写す前の注意（まだ写さない。2027 年 1 月の決め直しまでに 3 段目で作る）: 材料を写すと、前日版の点・型の水準・消しの当たりの数が変わる。
    最初の版（20 サイクルの B、prereg_cycles.json）は今の材料の点で作った規則なので、そのまま当てはめると別の馬を選び、記録の照合
    （picks_history・playbook の _check）でも止まる。材料に版を付け、版ごとに使う材料を分ける作りが要る（計画書 §27）。
流れ: folklore の master（Y まで）→ frame → tally → select → select_roi → select_kill（どれも run.py の同じコマンド。期間は GRADED_MATERIAL_END で渡す）。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
JST = timezone(timedelta(hours=9))
FILES = ("prereg.json", "prereg_roi.json", "prereg_kill.json")
STEPS = ("frame", "tally", "select", "select_roi", "select_kill")


def env_for(end: int, out: Path, resume: bool = False) -> dict[str, str]:
    """決め直しの環境: 期間は end まで、置き場と材料の書き先は out の下（リポジトリの材料・作業場所を書き換えない）。
    resume=False なら folklore の途中の表から再開しない（別の年までの master を混ぜない）。--from-step のときは同じ置き場の選び直しを許す。"""
    e = {k: v for k, v in os.environ.items() if k not in ("FOLKLORE_ENTRIES", "FOLKLORE_CUTOFF", "GRADED_ENTRIES", "GRADED_RESELECT")}
    e |= {"GRADED_MATERIAL_END": str(end), "GRADED_YEAR_END": str(end), "FOLKLORE_YEAR_END": str(end), "FOLKLORE_OUT": str(out / "master"),
          "FOLKLORE_KEEP_DNF": "1", "FOLKLORE_RESUME": "1" if resume else "0", "GRADED_OUT": str(out), "GRADED_PREREG": str(out / "prereg.json"),
          "GRADED_PREREG_ROI": str(out / "prereg_roi.json"), "GRADED_PREREG_KILL": str(out / "prereg_kill.json"),
          "OMP_NUM_THREADS": e.get("OMP_NUM_THREADS", "8")}
    if resume:
        e["GRADED_RESELECT"] = "1"
    return e


MARK = "material.json"


def check_out(out: Path, end: int, from_step: str | None) -> None:
    """置き場の守り: リポジトリの中・共有の作業場所・ほかの用途の置き場（frame や評価の印があり、この印が無い）は使わない。
    --from-step は、同じ区切りの年で始めた置き場だけ。"""
    if out == REPO or REPO in out.parents:
        raise SystemExit("--out はリポジトリの外にする")
    from src.research.graded.features import GRADED_OUT
    if out in (Path("/home/jovyan/work/graded_out").resolve(), GRADED_OUT.resolve()):
        raise SystemExit("--out に共有の作業場所は使わない（新しい置き場にする）")
    mark = out / MARK
    if mark.exists():
        m = json.loads(mark.read_text(encoding="utf-8"))
        if int(m.get("end", -1)) != end:
            raise SystemExit(f"この置き場は --end {m.get('end')} で始めたもの（--end {end} は新しい置き場で）")
        if not from_step:
            raise SystemExit("この置き場はすでに使った。続きは --from-step、やり直しは新しい置き場で")
    else:
        if from_step:
            raise SystemExit("--from-step は、この コマンドで始めた置き場だけ")
        if out.exists() and any((out / f).exists() for f in ("frame.parquet", "tally.parquet", "EVALUATED.json", "EVALUATED_KILL.json", "master")):
            raise SystemExit("ほかの用途の置き場（frame や評価の印がある）には書かない")


def check_tables(end: int) -> None:
    """区切りの年までの結果の表が手元にあるか（無い年は features・payout が黙って飛ばすので、先に止める）。"""
    miss = [y for y in range(2016, end + 1) for f in ("race_result_flat.parquet", "race_result_race.parquet")
            if not (REPO / "data/page_reference/tables" / str(y) / f).exists()]
    if miss:
        raise SystemExit(f"結果の表が無い年: {sorted(set(miss))}（作業ツリーのリンクを確かめる）")


def check_end(end: int, today: str | None = None) -> None:
    """区切りの年は、結果のそろった年（今年より前）で、最初の材料（2024 年）以降。"""
    year = int((today or datetime.now(JST).strftime("%Y-%m-%d"))[:4])
    if end >= year:
        raise SystemExit(f"--end {end} はまだ結果がそろっていない（{year - 1} 年まで）")
    if end < 2024:
        raise SystemExit("--end は 2024 年以降")


def diff(old: dict, new: dict, old_kill: dict | None = None, new_kill: dict | None = None,
         old_roi: dict | None = None, new_roi: dict | None = None) -> dict:
    """材料の違い: 前日版の使う型の増減と期間、消し・回収率版の主に見る版の名前と使う型の増減。"""
    def used(P):
        s = (P.get("sets") or {}).get("preday") or {}
        return set(s.get("used") or [])

    def prim(P):
        if not P:
            return None, set()
        v = next((v for v in P.get("versions", []) if v.get("primary")), None) or {}
        return v.get("name") or P.get("primary"), set(v.get("used") or [])
    a, b = used(old), used(new)
    out = {"periods": {"old": old.get("periods"), "new": new.get("periods")}, "preday_used_added": sorted(b - a), "preday_used_removed": sorted(a - b),
           "preday_used_kept": len(a & b)}
    for k, (o, n) in {"kill": (old_kill, new_kill), "roi": (old_roi, new_roi)}.items():
        (on, ou), (nn, nu) = prim(o), prim(n)
        out[k] = {"primary_old": on, "primary_new": nn, "used_added": sorted(nu - ou), "used_removed": sorted(ou - nu)}
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", type=int, required=True, help="区切りの年（この年の末までの結果で選ぶ）")
    ap.add_argument("--out", type=Path, required=True, help="置き場（リポジトリの外。master・frame・材料を書く）")
    ap.add_argument("--from-step", choices=STEPS, default=None, help="途中から（master と前の段の出力があるとき）")
    a = ap.parse_args(argv)
    check_end(a.end)
    out = a.out.resolve()
    check_out(out, a.end, a.from_step)
    check_tables(a.end)
    out.mkdir(parents=True, exist_ok=True)
    e = env_for(a.end, out, resume=bool(a.from_step))
    mp = out / MARK
    log = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {"end": a.end, "out": str(out),
                                                                             "started_at": datetime.now(JST).isoformat(timespec="seconds"), "steps": []}

    def save():
        mp.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    save()
    steps = list(STEPS)
    if a.from_step:
        steps = steps[steps.index(a.from_step):]
    todo = ([("master", [sys.executable, "-m", "src.scripts.data.folklore_build"])] if not a.from_step else []) + \
        [(st, [sys.executable, "-m", "src.research.graded.run", st]) for st in steps]
    for st, cmd in todo:
        t0 = time.time()
        r = subprocess.run(cmd, check=False, env=e, cwd=REPO)
        log["steps"].append({"step": st, "seconds": round(time.time() - t0), "rc": r.returncode,
                             "at": datetime.now(JST).isoformat(timespec="seconds")})
        save()       # どの段まで済んだかを毎回残す（--from-step の手がかり）
        if r.returncode != 0:
            raise SystemExit(f"{st} が失敗（rc={r.returncode}）。直してから --from-step {st} で続ける")
        print(f"{st} 済み {time.time() - t0:.0f}s", flush=True)
    rd = lambda p: json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    G = REPO / "src/research/graded"
    log["diff"] = diff(rd(G / "prereg.json"), rd(out / "prereg.json"), rd(G / "prereg_kill.json"), rd(out / "prereg_kill.json"),
                       rd(G / "prereg_roi.json"), rd(out / "prereg_roi.json"))
    log["finished_at"] = datetime.now(JST).isoformat(timespec="seconds")
    log["next"] = "まだ写さない（写す前の注意を参照。材料の版ごとの使い分けが入ってから）"
    save()
    (out / "material_diff.json").write_text(json.dumps(log["diff"], ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(log["diff"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
