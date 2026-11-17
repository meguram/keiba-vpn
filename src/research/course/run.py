"""全レース版（コース × クラスの群 × 良馬場）の条件の抽出（docs/ideas/course_class_handicapper.md）。

作法（ユーザーの指示）:
- 条件は 2016〜2024 年の全体に合わせて選ぶ（過学習してよい）。A（評価の期間に合わせた版）は作らない。
- 20 サイクルの採否・ゴールの判定は 2016〜2024 年の値だけ。参考に確認の期間（2022〜2024）の値も並べる。
- 2025・2026 年は 20 サイクルの間は一切見ない: 計算の表は 2024 年末で切り、2025・2026 年の表のファイルを開かない
  （FOLKLORE_YEAR_END・GRADED_YEAR_END を 2024 にする。tests/research/test_course_run.py で確かめる）。
- 最後の条件を prereg に凍結・commit してから、2025・2026 年で 1 回だけ評価する（評価の後に変えない）。
条件の選び方（重賞と同じ）:
- 買い（◎ の側）: 切り口 × 水準の AND で、複勝率 70% 以上かつ単勝か複勝の回収率 100% 超。件数の少ない条件も可（1〜2 走は「極小」）。
- 消し: 3 着以内がほぼ 0 の条件（[0-0-0-n]）。◎ = 当てはまった買いの条件の数（重み）が最大の馬。
- オッズ・人気は条件に使わない（catalog の型だけ）。as-of（レースの前日まで）の列だけで水準を作る。
計算は手元だけ（費用 0 円）。重い計算は load が 25 を下回ってから（統括役の指示）。
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

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("COURSE_OUT") or "/home/jovyan/work/course_out")
YEAR_END = 2024                  # 20 サイクルの間に読む最後の年
JST = timezone(timedelta(hours=9))
GROUPS = (("新馬・未勝利", 0, 1), ("1勝・2勝", 2, 3), ("3勝・OP", 4, 5))   # クラスの番号（folklore の CLS）。重賞（6 以上）は単位から外す
GOAL = {"win": 0.30, "roi_t": 1.20, "kill_precision": 0.95, "kill_per_race": 3}


def group_of(cls) -> str | None:
    try:
        c = float(cls)
    except (TypeError, ValueError):
        return None
    for name, lo, hi in GROUPS:
        if lo <= c <= hi:
            return name
    return None


def env(year_end: int = YEAR_END) -> dict:
    """計算に渡す環境変数。表を読む年の上限を year_end にする（2025・2026 年のファイルを開かない）。"""
    base = {k: v for k, v in os.environ.items() if k not in ("FOLKLORE_ENTRIES", "FOLKLORE_CUTOFF", "GRADED_ENTRIES")}   # 出馬表の段階の行を足さない
    return dict(base, FOLKLORE_OUT=str(OUT / "master"), FOLKLORE_YEAR_END=str(year_end), FOLKLORE_KEEP_DNF="1", FOLKLORE_RESUME="0",
                GRADED_OUT=str(OUT), GRADED_YEAR_END=str(year_end), PYTHONPATH=str(REPO),
                OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8")


def cmd_frame() -> None:
    """master → frame → 水準（全クラス）。2024 年末までの表だけで作る。"""
    OUT.mkdir(parents=True, exist_ok=True)
    e = env()
    t0 = time.time()
    mp = OUT / "master" / "master.parquet"
    if not (os.environ.get("COURSE_REUSE_MASTER") == "1" and master_ok(mp)):   # 作り済みの master を使い回すのは、2024 年までで同じ設定のときだけ
        subprocess.run([sys.executable, "-m", "src.scripts.data.folklore_build"], env=e, cwd=REPO, check=True)
        (OUT / "master" / "course_master.json").write_text(json.dumps({"year_end": YEAR_END, "keep_dnf": True}), encoding="utf-8")
    subprocess.run([sys.executable, "-m", "src.research.course.run", "_frame_inner"], env=e, cwd=REPO, check=True)
    print("frame", round(time.time() - t0), "秒", flush=True)


def _frame_inner() -> None:
    from src.research.graded import catalog as CAT
    from src.research.graded import features as F
    from src.research.graded.run import with_returns
    if max(F.YEARS) > YEAR_END:
        sys.exit("GRADED_YEAR_END が 2024 になっていない（2025・2026 年の表を読んでしまう）")
    d = F.build(out=None)
    d = with_returns(d)
    if (d["year"] > YEAR_END).any():
        sys.exit("2025 年以降の行が入っている")
    d, L = frame_and_levels(d)
    d.to_parquet(OUT / "frame.parquet", index=False)
    L.to_parquet(OUT / "levels.parquet", index=False)
    print("frame", d.shape, "levels", L.shape, "years", int(d["year"].min()), int(d["year"].max()), flush=True)


MIN_AVAIL_GB = float(os.environ.get("COURSE_MIN_AVAIL_GB", "40"))
MAX_LOAD = float(os.environ.get("COURSE_MAX_LOAD", "20"))


def _meminfo() -> dict[str, float]:
    out = {}
    try:
        for ln in Path("/proc/meminfo").read_text().splitlines():
            k, v = ln.split(":", 1)
            out[k] = float(v.split()[0]) / 1024 / 1024     # GiB
    except (OSError, ValueError):
        pass
    return out


def mem_now() -> dict[str, float]:
    """この処理のメモリ（いま・最大、GiB）と、計算機の空き（GiB）。止められたときの原因を追うためにログに出す。"""
    import resource
    rss = 0.0
    try:
        for ln in Path("/proc/self/status").read_text().splitlines():
            if ln.startswith("VmRSS:"):
                rss = float(ln.split()[1]) / 1024 / 1024
    except OSError:
        pass
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    return {"rss_gb": round(rss, 1), "peak_gb": round(peak, 1), "avail_gb": round(_meminfo().get("MemAvailable", 0.0), 1)}


def wait_resources(poll: int = 60) -> None:
    """空きメモリ MIN_AVAIL_GB 以上・load（1 分）MAX_LOAD 未満になるまで待つ（統括役の決まり。ほかの重い計算と重ねない）。"""
    while True:
        av = _meminfo().get("MemAvailable", 0.0)
        ld = os.getloadavg()[0]
        if av >= MIN_AVAIL_GB and ld < MAX_LOAD:
            return
        print(f"待ち: 空き {av:.1f}GB・load {ld:.1f}（{MIN_AVAIL_GB:.0f}GB 以上・{MAX_LOAD:.0f} 未満で始める）", flush=True)
        time.sleep(poll)


def frame_and_levels(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """凍結の側と評価の側で共通の前処理: 芝ダに絞り、文字の列を string にし、欠けありの数値型を float にしてから、全年・全クラス・全馬場の
    表のまま水準を作る（騎手・厩舎の率の境目は 2017 年の重賞の行の分位で決まるので、単位に絞る前に作る。catalog._ref）。"""
    from src.research.graded import catalog as CAT
    d = d[d["surf"].isin(["芝", "ダ"])].reset_index(drop=True)
    for c in d.columns:
        if d[c].dtype == object:
            d[c] = d[c].astype("string")
    L = CAT.levels(_plain_numeric(d))
    L["__race_id"], L["__horse_id"] = d["race_id"].astype(str).to_numpy(), d["horse_id"].astype(str).to_numpy()   # 行の対応の検査用
    return d, L


def _plain_numeric(d: pd.DataFrame) -> pd.DataFrame:
    """欠けありの数値型（pandas の Int64・Float64 など）を float64（欠けは NaN）にした写し。"""
    d = d.copy()
    for c in d.columns:
        if pd.api.types.is_extension_array_dtype(d[c].dtype) and pd.api.types.is_numeric_dtype(d[c].dtype) and not pd.api.types.is_bool_dtype(d[c].dtype):
            d[c] = d[c].astype("float64")
    return d


def master_ok(path: Path) -> bool:
    """使い回してよい master か: 年の列の最大（parquet の統計だけを読む）が YEAR_END 以下で、作ったときの設定の記録が今と同じ。"""
    import pyarrow.parquet as pq
    if not path.exists():
        return False
    try:
        rec = json.loads((path.parent / "course_master.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if rec != {"year_end": YEAR_END, "keep_dnf": True}:
        return False
    md = pq.ParquetFile(path).metadata
    j = pq.ParquetFile(path).schema_arrow.get_field_index("year")
    mx = [md.row_group(i).column(j).statistics.max for i in range(md.num_row_groups) if md.row_group(i).column(j).statistics]
    return bool(mx) and max(mx) <= YEAR_END


def load_units(min_year: int = 2016) -> tuple[pd.DataFrame, pd.DataFrame]:
    """単位の表: 良馬場・芝ダ・重賞以外・2016〜2024 年の行と、その水準。unit = コース|クラスの群。"""
    d = pd.read_parquet(OUT / "frame.parquet")
    L = pd.read_parquet(OUT / "levels.parquet")
    if (pd.to_numeric(d["year"]) > YEAR_END).any():
        raise ValueError("2025 年以降の行が入っている")
    if "__race_id" in L.columns:
        if len(L) != len(d) or not ((L["__race_id"].astype(str).to_numpy() == d["race_id"].astype(str).to_numpy()).all()
                                    and (L["__horse_id"].astype(str).to_numpy() == d["horse_id"].astype(str).to_numpy()).all()):
            raise ValueError("frame と levels の行がずれている")
        L = L.drop(columns=["__race_id", "__horse_id"])
    g = d["cls"].map(group_of)
    m = (d["track"].astype(str) == "良") & g.notna() & (pd.to_numeric(d["year"]) >= min_year)
    d, L = d[m].reset_index(drop=True), L[m.to_numpy()].reset_index(drop=True)
    d["group"] = g[m].to_numpy()
    d["unit"] = d["course"].astype(str) + "|" + d["group"].astype(str)
    return d, L


def regroup(d: pd.DataFrame, groups: int) -> pd.Series:
    """単位の鍵。groups=4 は新馬を未勝利から分ける（コースごとのもっと良い分け方の候補。2016〜2024 年の中だけで比べる）。"""
    g = d["group"].astype(str)
    if groups == 4:
        newcomer = d["race_class"].astype(str).str.contains("新馬") if "race_class" in d.columns else pd.Series(False, index=d.index)
        g = g.where(~((g == "新馬・未勝利") & newcomer), "新馬")
    return d["course"].astype(str) + "|" + g


# 20 サイクルの仮説。毎回、それまでに採った版から 1 つだけ変える（delta）。採否は 2016〜2024 年の当てはめの値だけで行う
CYCLES = [
    {"hyp": "基準: 深さ 1・前日版・走数 2 以上", "delta": {}},
    {"hyp": "2 つの AND まで", "delta": {"depth": 2}},
    {"hyp": "3 つの AND まで（主の層だけ。副は 2 まで）", "delta": {"depth": 3}},
    {"hyp": "買いの走数の下限 3", "delta": {"n_min": 3}},
    {"hyp": "買いの走数の下限 5", "delta": {"n_min": 5}},
    {"hyp": "◎ の点を条件の複勝率の和にする", "delta": {"weight": "rate"}},
    {"hyp": "主の回数の下限 10（少ない単位も主で探す）", "delta": {"min_races": 10}},
    {"hyp": "主の回数の下限 30（少ない単位はコース全体で探す）", "delta": {"min_races": 30}},
    {"hyp": "クラスの群を 4 つ（新馬を分ける）", "delta": {"groups": 4}},
    {"hyp": "消しの走数の下限 5", "delta": {"kill_n": 5}},
    {"hyp": "消しの走数の下限 20", "delta": {"kill_n": 20}},
    {"hyp": "消しを 2 つの AND まで", "delta": {"kill_depth": 2}},
    {"hyp": "買いの走数の下限 2 に戻す", "delta": {"n_min": 2}},
    {"hyp": "買いの走数の下限 8", "delta": {"n_min": 8}},
    {"hyp": "深さ 2 に戻す", "delta": {"depth": 2}},
    {"hyp": "◎ の点を条件の数に戻す", "delta": {"weight": "count"}},
    {"hyp": "主の回数の下限 20 に戻す", "delta": {"min_races": 20}},
    {"hyp": "クラスの群を 3 つに戻す", "delta": {"groups": 3}},
    {"hyp": "消しの走数の下限 15", "delta": {"kill_n": 15}},
    {"hyp": "当日版（当日の値の型も使う。前日版と並べるための参考の版）", "delta": {"dayof": True}},
]
MIN_COVERAGE = 0.30      # 単位の達成に要る、買ったレースの割合（1〜2 回だけ買って勝つ版を達成に数えない）
MIN_BET = 5              # 単位の達成に要る、買ったレースの数


def _closeness(m: dict) -> float:
    """ゴールまでの近さ。買ったレースの割合が MIN_COVERAGE に足りなければ、その分だけ割り引く。"""
    if not m.get("win") or m.get("roi_t") is None:
        return 0.0
    return min(m["win"] / GOAL["win"], m["roi_t"] / GOAL["roi_t"]) * min(1.0, (m.get("coverage") or 0) / MIN_COVERAGE)


def run_config(d: pd.DataFrame, L: pd.DataFrame, p, keep_cells: bool = False) -> dict:
    """1 つの版を全単位に当てはめた 2016〜2024 年の値（当てはめ）と、2022〜2024 年の値（参考）、単位ごとの値。"""
    from src.research.course import search as SE
    from src.research.graded import buy as BY
    from src.research.graded import catalog as CAT
    types = [t.id for t in CAT.TYPES if not t.optional and t.id not in ("M01", "M02") and (p.dayof or not t.day_of) and t.id in L.columns]
    X, names = BY.encode(L, types)
    Y = SE.y_matrix(d)
    unit = regroup(d, p.groups).to_numpy()
    course = d["course"].astype(str).to_numpy()
    race = d["race_id"].astype(str).to_numpy()
    hn = pd.to_numeric(d["hn"], errors="coerce").fillna(99).to_numpy()
    yr = pd.to_numeric(d["year"]).to_numpy()
    H, K = [], np.zeros(len(d), bool)
    per_unit, out_cells, sub_cache = [], {}, {}
    for u in sorted(set(unit)):
        idx = np.flatnonzero(unit == u)
        nr = len(set(race[idx]))
        if nr < p.min_races:
            c0 = course[idx[0]]
            if c0 not in sub_cache:   # 副: 同じコースの全群（コースごとに 1 回だけ探す）
                sub = np.flatnonzero(course == c0)
                sub_cache[c0] = SE.find_cells(X[sub], Y[sub], p, sub=True)
            cells, layer = sub_cache[c0], "副"
        else:
            cells, layer = SE.find_cells(X[idx], Y[idx], p), "主"
        hm, kl = SE.apply_cells(X[idx], race[idx], hn[idx], cells, p.weight)
        hrows = idx[list(hm.values())] if hm else np.array([], int)
        H.append(hrows)
        K[idx[kl]] = True
        m = SE.metrics(Y, race, hrows, idx[kl], nr)
        ach = bool(m["win"] is not None and m["win"] >= GOAL["win"] and (m["roi_t"] or 0) >= GOAL["roi_t"]
                   and (m["coverage"] or 0) >= MIN_COVERAGE and m["bet_races"] >= MIN_BET)
        per_unit.append({"unit": u, "layer": layer, "races": nr, "n_buy": len(cells["buy"]), "n_kill": len(cells["kill"]),
                         "capped": cells["capped"], **m, "achieved": ach})
        if keep_cells:
            fmt = lambda c: {"types": [types[t] for t in c["ts"]], "levels": [names[types[t]][v] for t, v in zip(c["ts"], c["cs"])]} \
                | {k: c[k] for k in c if k not in ("ts", "cs")}   # noqa: E731
            out_cells[u] = {"layer": layer, "buy": [fmt(c) for c in cells["buy"]], "kill": [fmt(c) for c in cells["kill"]]}
    H = np.concatenate(H) if H else np.array([], int)
    allm = SE.metrics(Y, race, H, np.flatnonzero(K), len(set(race)))
    conf = (yr >= SE.CONF[0]) & (yr <= SE.CONF[1])
    cm = SE.metrics(Y, race, H[conf[H]], np.flatnonzero(K & conf), len(set(race[conf])))
    out = {"params": SE.params_dict(p), "all": allm, "conf_2022_2024": cm, "units": len(per_unit),
           "achieved_units": sum(x["achieved"] for x in per_unit), "closeness": _closeness(allm), "per_unit": per_unit}
    if keep_cells:
        out["cells"] = out_cells
    return out


def cmd_cycles() -> None:
    """20 サイクル。各サイクルの版を全単位に当てはめ、2016〜2024 年の値で採否する（当てはめ。過学習してよい）。2025・2026 年は読まない。"""
    from src.research.course import search as SE
    d, L = load_units()
    log = {"note": "全レース版の 20 サイクル（2016〜2024 年の中だけ。2025・2026 年は見ていない）", "goal": GOAL,
           "started_at": datetime.now(JST).isoformat(timespec="seconds"), "rows": int(len(d)), "races": int(d["race_id"].nunique()), "cycles": []}
    best = None
    cur = {}
    path = OUT / "course_cycles.json"
    for c, cy in enumerate(CYCLES, 1):
        wait_resources()
        t0 = time.time()
        cand = cur | cy["delta"]
        p = SE.Params(**cand)
        r = run_config(d, L, p)
        # 当日版は前日版と並べる参考の版なので、採らない（凍結するのは前日版。当日版の値は記録に並べる）
        adopt = not cand.get("dayof") and (best is None or (r["achieved_units"], r["closeness"]) > (best["achieved_units"], best["closeness"]))
        if adopt:
            best = r | {"cycle": c}
            cur = cand
        log["cycles"].append({"cycle": c, "hyp": cy["hyp"], "params": r["params"], "all": r["all"], "conf_2022_2024": r["conf_2022_2024"],
                              "achieved_units": r["achieved_units"], "units": r["units"], "closeness": r["closeness"], "adopted": adopt,
                              "seconds": round(time.time() - t0, 1), "mem": mem_now()})
        log["best_cycle"] = best["cycle"]
        path.write_text(json.dumps(log, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        a = r["all"]
        print(f"cycle {c} {cy['hyp']} 達成 {r['achieved_units']}/{r['units']} ◎ 勝率 {a['win']} 単回収 {a['roi_t']} 買う回 {a['coverage']} "
              f"消しのレース {a['kill_coverage']} 採用 {adopt} {time.time() - t0:.0f}s メモリ {mem_now()}", flush=True)
    log["finished_at"] = datetime.now(JST).isoformat(timespec="seconds")
    path.write_text(json.dumps(log, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    # 最後に採った版の条件を、単位ごとに書き出す（凍結の材料）
    p = SE.Params(**{k: v for k, v in best["params"].items() if k != "name"})
    fin = run_config(d, L, p, keep_cells=True)
    (OUT / "course_best.json").write_text(json.dumps(fin, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


PREREG = Path(__file__).with_name("prereg_course.json")              # 凍結（commit する）。差し替えの環境変数は持たない
EVAL_MARK = Path(__file__).with_name("EVALUATED_COURSE.json")       # 評価済みの印（commit する。作業場所ではなくリポジトリに置く）


def _sha_file(p: Path) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cmd_freeze() -> None:
    """最後に採った版（course_best.json）を凍結する: 条件の表を <COURSE_OUT>/course_cells.json に書き、版・sha を prereg_course.json に書く
    （prereg は commit する。評価はこの後に 1 回だけ）。凍結済み・評価済みなら止める。"""
    if PREREG.exists() and os.environ.get("COURSE_RESELECT") != "1":
        sys.exit(f"{PREREG} は凍結済み")
    if EVAL_MARK.exists():
        sys.exit(f"評価済みの印 {EVAL_MARK} がある。凍結し直さない")
    B = json.loads((OUT / "course_best.json").read_text(encoding="utf-8"))
    L = json.loads((OUT / "course_cycles.json").read_text(encoding="utf-8"))
    if not L.get("finished_at"):
        sys.exit("サイクルが最後まで回っていない（途中で止まった記録は凍結しない）")
    bc = next((c for c in L["cycles"] if c["cycle"] == L.get("best_cycle")), None)
    if bc is None or {k: v for k, v in bc["params"].items()} != B["params"]:
        sys.exit("course_best.json の版が、記録の最良のサイクルの版と違う")
    cells = OUT / "course_cells.json"
    cells.write_text(json.dumps(B["cells"], ensure_ascii=False, sort_keys=True), encoding="utf-8")
    body = {"frozen_at": datetime.now(JST).isoformat(timespec="seconds"),
            "note": "全レース版の最後に採った版。2016〜2024 年に合わせて選んだ（過学習してよい）。2025・2026 年は見ていない。この後に 1 回だけ評価し、評価の後に変えない",
            "params": B["params"], "best_cycle": L.get("best_cycle"), "cycles_finished_at": L.get("finished_at"),
            "fit_2016_2024": B["all"], "achieved_units": B["achieved_units"], "units": B["units"],
            "cells_file": "course_cells.json", "sha256_cells": _sha_file(cells),
            "track_rule": "単位は良馬場で行われたレース。評価では実際に良馬場だったレースに当てはめる（前日の予想では馬場は見込み）"}
    PREREG.write_text(json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8")
    print("frozen", PREREG)


def evaluate_frozen(d: pd.DataFrame, L: pd.DataFrame, P: dict, cells: dict) -> dict:
    """凍結した条件を、2025・2026 年の単位（同じ群の分け方）に当てはめた値。条件の型・水準の名前を、評価の表の番号に直して、
    凍結の側と同じ選び方（search.apply_cells）で ◎ と消しを決める。評価の表に一度も出ない水準・型の列が無いことは記録する（型の列が無ければ止める）。"""
    from src.research.course import search as SE
    from src.research.graded import buy as BY
    p = SE.Params(**{k: v for k, v in P["params"].items() if k != "name"})
    types = sorted({t for cu in cells.values() for kind in ("buy", "kill") for c in cu[kind] for t in c["types"]})
    miss = [t for t in types if t not in L.columns]
    if miss:
        raise ValueError(f"評価の表に型の列が無い: {miss}")
    X, names = BY.encode(L, types) if types else (np.zeros((len(L), 0), np.int32), {})
    code = {t: {v: i for i, v in enumerate(names[t])} for t in types}
    unmatched = 0

    def conv(c):
        nonlocal unmatched
        cs = []
        for t, lv in zip(c["types"], c["levels"]):
            if lv not in code[t]:
                unmatched += 1
            cs.append(code[t].get(lv, -2))      # 評価の表に出ない水準は、どの行にも当たらない番号
        return {"ts": tuple(types.index(t) for t in c["types"]), "cs": tuple(cs), "rate": c.get("rate", 1.0)}
    Y = SE.y_matrix(d)
    unit = regroup(d, p.groups).to_numpy()
    race = d["race_id"].astype(str).to_numpy()
    hn = pd.to_numeric(d["hn"], errors="coerce").fillna(99).to_numpy()
    H, K, per = [], np.zeros(len(d), bool), []
    for u in sorted(set(unit)):
        idx = np.flatnonzero(unit == u)
        nr = len(set(race[idx]))
        cu = cells.get(u)
        if cu is None:   # 2016〜2024 年に無い単位は買わない
            per.append({"unit": u, "races": nr, "no_rule": True})
            continue
        cc = {"buy": [conv(c) for c in cu["buy"]], "kill": [conv(c) for c in cu["kill"]]}
        hm, kl = SE.apply_cells(X[idx], race[idx], hn[idx], cc, p.weight)
        hrows = idx[list(hm.values())] if hm else np.array([], int)
        H.append(hrows)
        K[idx[kl]] = True
        per.append({"unit": u, **SE.metrics(Y, race, hrows, idx[kl], nr)})
    H = np.concatenate(H) if H else np.array([], int)
    return {"all": SE.metrics(Y, race, H, np.flatnonzero(K), len(set(race))), "per_unit": per, "unmatched_levels": unmatched}


def cmd_evaluate() -> None:
    import hashlib
    """凍結の後に 1 回だけ: 2025・2026 年の表（別の置き場 <COURSE_OUT>/eval）で、凍結した条件を当てはめる。凍結が commit されていなければ止める。"""
    P = json.loads(PREREG.read_text(encoding="utf-8"))
    cells_p = OUT / P["cells_file"]
    if _sha_file(cells_p) != P["sha256_cells"]:
        sys.exit("条件の表が凍結の sha と違う")
    r = subprocess.run(["git", "-C", str(REPO), "ls-files", "--error-unmatch", str(PREREG)], capture_output=True)
    if r.returncode != 0 or subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", str(PREREG)]).returncode != 0:
        sys.exit("prereg_course.json が commit されていない（凍結を commit してから評価する）")
    import secrets
    if EVAL_MARK.exists():
        sys.exit("評価は済んでいる（リポジトリの印。1 回だけ）")
    token = secrets.token_hex(16)
    # 結果を出す前に、評価済みの印を先に立てる（重賞版と同じ。途中で落ちても、結果を見たうえでのやり直しをさせない）
    EVAL_MARK.write_text(json.dumps({"started_at": datetime.now(JST).isoformat(timespec="seconds"), "prereg_sha_cells": P["sha256_cells"],
                                     "token_sha": hashlib.sha256(token.encode()).hexdigest()}), encoding="utf-8")
    ev_out = OUT / "eval"
    e = env(2026) | {"FOLKLORE_OUT": str(ev_out / "master"), "GRADED_OUT": str(ev_out), "COURSE_EVAL_TOKEN": token}
    subprocess.run([sys.executable, "-m", "src.scripts.data.folklore_build"], env=e, cwd=REPO, check=True)
    subprocess.run([sys.executable, "-m", "src.research.course.run", "_eval_inner"], env=e | {"COURSE_OUT": str(OUT)}, cwd=REPO, check=True)


def _eval_inner() -> None:
    """cmd_evaluate からだけ呼ぶ（一度きりのトークンと、リポジトリの印・凍結の sha を照らす）。"""
    import hashlib
    from src.research.graded import features as F
    from src.research.graded.run import with_returns
    tok = os.environ.get("COURSE_EVAL_TOKEN", "")
    mark = json.loads(EVAL_MARK.read_text(encoding="utf-8")) if EVAL_MARK.exists() else {}
    if not tok or mark.get("token_sha") != hashlib.sha256(tok.encode()).hexdigest() or "evaluated_at" in mark:
        sys.exit("評価は cmd_evaluate から 1 回だけ流す")
    P = json.loads(PREREG.read_text(encoding="utf-8"))
    if _sha_file(OUT / P["cells_file"]) != P["sha256_cells"] or mark.get("prereg_sha_cells") != P["sha256_cells"]:
        sys.exit("条件の表が凍結の sha と違う")
    d, L = frame_and_levels(with_returns(F.build(out=None)))       # 凍結の側と同じ前処理（全年の表で水準を作ってから絞る）
    keep = ((pd.to_numeric(d["year"]) >= 2025) & (d["track"].astype(str) == "良") & d["cls"].map(group_of).notna()).to_numpy()
    d, L = d[keep].reset_index(drop=True), L[keep].reset_index(drop=True).drop(columns=["__race_id", "__horse_id"])
    d["group"] = d["cls"].map(group_of)
    cells = json.loads((OUT / P["cells_file"]).read_text(encoding="utf-8"))
    res = evaluate_frozen(d, L, P, cells) | {"evaluated_at": datetime.now(JST).isoformat(timespec="seconds"), "fit_2016_2024": P["fit_2016_2024"],
                                              "params": P["params"], "races": int(d["race_id"].nunique())}
    (OUT / "eval_course.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    EVAL_MARK.write_text(json.dumps(mark | {"evaluated_at": res["evaluated_at"]}), encoding="utf-8")   # この印を commit する
    a, f = res["all"], P["fit_2016_2024"]
    print(f"評価 2025・2026: ◎ 勝率 {a['win']} 単回収 {a['roi_t']}（当てはめ 2016〜2024: {f['win']} / {f['roi_t']}）", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["frame", "_frame_inner", "cycles", "freeze", "evaluate", "_eval_inner"])
    a = ap.parse_args()
    {"frame": cmd_frame, "_frame_inner": _frame_inner, "cycles": cmd_cycles, "freeze": cmd_freeze, "evaluate": cmd_evaluate,
     "_eval_inner": _eval_inner}[a.cmd]()


if __name__ == "__main__":
    main()
