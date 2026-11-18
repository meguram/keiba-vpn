"""ルールの一覧の ◎ の買う／見送るの最適化（rules-bet-opt。計画・レビュー: docs/operations/rules_bet_opt_20261118.md）。

道具は作り直さない: 通常の ◎ は第 3 版（src/research/course/v3.py）の fit_cells・add_market・select_cells・pick、
発走前オッズの再現は bet_cycles（src/scripts/data/bet_cycles.py）の load_drift（2026 年の T-60 と確定の比をレースごとに借りる作法）、
指標は bet_cycles.metrics・poisson_weights・paired_diff を import して使う（どのファイルも書き換えない）。

  python -m src.research.course.bet_opt wf        # walk-forward の ◎（訓練 2016〜Y-1 → Y 年、Y = 2020〜2024。2025 年以降の行は読み込みの直後に落とす）
  python -m src.research.course.bet_opt diag      # γ の符号の診断（オッズだけ。t60 → t15 → 確定の勢いの相関。着順は使わない）
  python -m src.research.course.bet_opt search    # 探す（2020〜22）→ 選ぶ → 評価（2023〜24）。config/rules_bet_opt/search_spec.json のとおり
  python -m src.research.course.bet_opt final     # 凍結（prereg_final.json が commit 済み）の後に 1 回だけ: 2025・2026 年（通常・重賞）

置き場: <COURSE_OUT>/bet_opt/。費用 0 円（手元の表だけ。GCS は読まない）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.course import run as CR
from src.research.course import v2 as V2
from src.research.course import v3 as V3

OUT = CR.OUT / "bet_opt"
SPEC = CR.REPO / "config/rules_bet_opt/search_spec.json"
PREREG = CR.REPO / "config/rules_bet_opt/prereg_final.json"
MARK = CR.REPO / "config/rules_bet_opt/EVALUATED.json"
WF_YEARS = range(2020, 2025)
TRAIN_FROM = 2016
KEEP_COLS = ["race_id", "date", "year", "hn", "course", "surf", "dist", "group", "race_class", "field", "fin", "pay_ok", "ret_tan", "ret_fuku",
             "odds_final", "pop_final"]


def frozen_c3() -> "V3.C3":
    P = json.loads(V3.PREREG3.read_text(encoding="utf-8"))
    return V3.C3(**{k: v for k, v in P["params"].items() if k != "name"})


def thirds(y0: int, y1: int) -> list[tuple[int, int]]:
    """訓練の窓 y0〜y1 を年で三等分した区切り（第 3 版の BLOCKS は 2016〜2024 の 3 年 × 3。窓が短い年でも区切りを 3 つにする）。"""
    ys = list(range(y0, y1 + 1))
    k = len(ys)
    cut = [ys[round(i * k / 3):round((i + 1) * k / 3)] for i in range(3)]
    return [(c[0], c[-1]) for c in cut if c]


def honmei_score(d: pd.DataFrame, L: pd.DataFrame, kept: dict, c: "V3.C3", H: np.ndarray) -> np.ndarray:
    """◎ の行の点（当たった買いの条件の rate の和。pick と同じ計算を ◎ の行だけでやり直す）。"""
    X, types, code = V2._codes(L, kept)
    unit = V3.units_of(d, c)
    sc = np.zeros(len(H))
    for i, h in enumerate(H):
        cu = kept.get(unit[h]) or {}
        sc[i] = sum(b["rate"] for b in cu.get("buy", []) if V2._mask_names(X[h:h + 1], types, code, b)[0])
    return sc


def load_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    """第 3 版の表（2016〜2025）を読み、2025 年以降の行を落とす（評価は凍結の後だけ）。"""
    d, L, _, _ = V3.load3()
    keep = (pd.to_numeric(d["year"]).to_numpy() <= 2024)
    return d[keep].reset_index(drop=True), L[keep].reset_index(drop=True)


def honmei_rows(d: pd.DataFrame, L: pd.DataFrame, kept: dict, c: "V3.C3", tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """◎ の行（点・勝率の見込みつき）と、◎ のあるレースの全頭の行（再現・基準の 1 番人気に使う）。"""
    H, Pw, _ = V3.pick(d, L, kept, c)
    sc = honmei_score(d, L, kept, c, H)
    hm = d.iloc[H][[x for x in KEEP_COLS if x in d.columns]].copy()
    hm["score"], hm["pw"], hm["unit"], hm["src"] = sc, Pw, V3.units_of(d, c)[H], tag
    rid = set(hm["race_id"].astype(str))
    fld = d[d["race_id"].astype(str).isin(rid)][[x for x in KEEP_COLS if x in d.columns]].copy()
    return hm.reset_index(drop=True), fld.reset_index(drop=True)


def cmd_wf() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d, L = load_frame()
    c = frozen_c3()
    yr = pd.to_numeric(d["year"]).to_numpy()
    summ = {}
    blocks0 = list(V3.BLOCKS)
    ver = hashlib.sha256(json.dumps({"params": json.loads(V3.PREREG3.read_text(encoding="utf-8"))["params"], "train_from": TRAIN_FROM,
                                     "code": "bet_opt.wf.v1"}, sort_keys=True).encode()).hexdigest()[:12]
    old = json.loads((OUT / "wf_summary.json").read_text(encoding="utf-8")) if (OUT / "wf_summary.json").exists() else {}
    try:
        _wf_years(d, L, c, yr, ver, old, summ, t0)
    finally:
        V3.BLOCKS = blocks0
    _wf_frozen(d, L, c, yr, old, summ, t0)


def _wf_years(d, L, c, yr, ver, old, summ, t0) -> None:
    for Y in WF_YEARS:
        p, pf = OUT / f"wf_{Y}.parquet", OUT / f"field_{Y}.parquet"
        if p.exists() and pf.exists() and (old.get(str(Y)) or {}).get("ver") == ver:
            print("済み", Y, flush=True)
            continue
        V3.BLOCKS = thirds(TRAIN_FROM, Y - 1)            # 実行時の差し替え（道具のファイルは直さない。fit_cells はモジュールの大域を実行時に読む）
        tr = (yr >= TRAIN_FROM) & (yr <= Y - 1)
        dt, Lt = d[tr].reset_index(drop=True), L[tr].reset_index(drop=True)
        cells = V3.fit_cells(dt, Lt, c)
        cells = V3.add_market(cells, dt, Lt, c)
        kept = V3.select_cells(cells, c)
        te = yr == Y
        dy, Ly = d[te].reset_index(drop=True), L[te].reset_index(drop=True)
        hm, fld = honmei_rows(dy, Ly, kept, c, f"wf{Y}")
        fld.to_parquet(pf, index=False)                  # 全頭 → ◎ の順に書く（◎ があれば全頭もある）
        hm.to_parquet(p, index=False)
        nrow = fld.groupby("race_id").size()
        fsz = pd.to_numeric(fld.groupby("race_id")["field"].first(), errors="coerce")
        nb = [len(cu["buy"]) for cu in kept.values()]
        summ[str(Y)] = {"ver": ver, "train": [TRAIN_FROM, Y - 1], "blocks": V3.BLOCKS,
                        "field_incomplete_share": float((nrow < fsz.reindex(nrow.index)).mean()), "races": int(dy["race_id"].nunique()), "honmei": int(len(hm)),
                        "coverage": float(len(hm) / max(dy["race_id"].nunique(), 1)), "units_with_buy": int(sum(x > 0 for x in nb)),
                        "buy_cells": int(sum(nb)), "sub_layer_share": float(np.mean([cu.get("layer") == "副" for cu in kept.values()])) if kept else None,
                        "pop_final_dist": {str(k): int(v) for k, v in pd.to_numeric(hm["pop_final"], errors="coerce").clip(upper=10).value_counts().sort_index().items()},
                        "elapsed_s": round(time.time() - t0)}
        print(Y, json.dumps(summ[str(Y)], ensure_ascii=False), flush=True)
        old.update(summ)
        (OUT / "wf_summary.json").write_text(json.dumps(old, ensure_ascii=False, indent=1), encoding="utf-8")


def _wf_frozen(d, L, c, yr, old, summ, t0) -> None:
    # 凍結の ◎（2016〜2024 で作った条件）を 2024 年に当てた行（当てはめ。walk-forward の 2024 年の ◎ との一致を見るためだけ）
    P = json.loads(V3.PREREG3.read_text(encoding="utf-8"))
    kept = json.loads((V3.OUT3 / P["cells_file"]).read_text(encoding="utf-8"))
    te = yr == 2024
    hf, _ = honmei_rows(d[te].reset_index(drop=True), L[te].reset_index(drop=True), kept, c, "frozen2024")
    hf.to_parquet(OUT / "frozen_2024.parquet", index=False)
    w = pd.read_parquet(OUT / "wf_2024.parquet")
    a = set(zip(w["race_id"].astype(str), w["hn"]))
    b = set(zip(hf["race_id"].astype(str), hf["hn"]))
    old.update(summ)
    old["frozen_vs_wf_2024"] = {"frozen": len(b), "wf": len(a), "same_pick": len(a & b), "same_race": len({x[0] for x in a} & {x[0] for x in b}),
                                "frozen_pop_dist": {str(k): int(v) for k, v in pd.to_numeric(hf["pop_final"], errors="coerce").clip(upper=10).value_counts().sort_index().items()}}
    (OUT / "wf_summary.json").write_text(json.dumps(old, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wf 終わり", round(time.time() - t0), "秒", json.dumps(old["frozen_vs_wf_2024"], ensure_ascii=False), flush=True)


# ---------------------------------------------------------------- 発走前オッズの再現（bet_cycles の作法 + γ の感度・頭数の近いレースから借りる）
def drift_pool(phase: str = "t60") -> list[np.ndarray]:
    """2026 年の、確定の人気の順に並べたレースごとのずれ d = log(確定 / 発走前)（bet_cycles.load_drift の races。着順は見ない）。"""
    from src.scripts.data import bet_cycles as BC
    return BC.load_drift(phase)["races"]


def sim_t60(field: pd.DataFrame, pool: list[np.ndarray], seed: int, gamma: float = 0.0, basis: str = "win", near: int = 2) -> pd.DataFrame:
    """全頭の確定オッズに、借りた 2026 年のレースの「確定の人気の順位ごとのずれ」を逆に当てて T-60 を作る（1/オッズの合計は元に合わせる）。
    γ ≠ 0 のときは、借りるレースを重み exp(−γ・d_w) で引く。d_w は対象レースの勝ち馬（basis=place は 3 着内）の確定の順位にある、借りる側のずれ。
    γ > 0 は「締切前に売れた馬（d < 0）が勝ちやすい」とみなす再現。頭数が ±near 頭のレースから借りる（無ければ全体から）。"""
    rng = np.random.default_rng(7919 + seed)
    lens = np.array([len(x) for x in pool])
    f = field.sort_values(["race_id", "pop_final", "hn"]).reset_index(drop=True)
    rid = f["race_id"].astype(str).to_numpy()
    st = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
    en = np.r_[st[1:], len(f)]
    odds = pd.to_numeric(f["odds_final"], errors="coerce").to_numpy(float)
    pop = pd.to_numeric(f["pop_final"], errors="coerce").to_numpy(float)
    fin = pd.to_numeric(f["fin"], errors="coerce").to_numpy(float)
    pre = np.full(len(f), np.nan)
    prp = np.full(len(f), 99.0)
    for a, b in zip(st, en):
        o = odds[a:b]
        ok = np.isfinite(o) & (o > 0)
        if ok.sum() < 2:
            continue
        n = int(ok.sum())                                 # 物差しは pool と同じ「オッズのそろった頭数」
        cand = np.flatnonzero(np.abs(lens - n) <= near)
        if not len(cand):
            cand = np.arange(len(pool))
        u = rng.random()                                  # 共通の乱数: γ によらず同じ u を累積の重みで逆変換する（γ の差に引き直しの雑音を混ぜない）
        rk_all = np.where(np.isfinite(pop[a:b]) & (pop[a:b] < 99), pop[a:b], 99).astype(int) - 1
        if gamma:
            tgt = ((fin[a:b] == 1) if basis == "win" else (fin[a:b] <= 3)) & ok
            dw = np.array([pool[j][np.clip(rk_all[tgt], 0, len(pool[j]) - 1)].sum() for j in cand])
            w = np.exp(-gamma * (dw - dw.mean()))
        else:
            w = np.ones(len(cand))
        cw = np.cumsum(w) / w.sum()
        j = cand[min(int(np.searchsorted(cw, u, side="right")), len(cand) - 1)]
        src = pool[j]
        dd = src[np.clip(rk_all, 0, len(src) - 1)]       # 順位が借りる側の頭数を超える馬は、その最下位のずれ（bet_cycles と同じ）
        x = np.where(ok, o * np.exp(-dd), np.nan)
        s0, s1 = np.nansum(1.0 / o[ok]), np.nansum(1.0 / x[ok])
        x = x * (s1 / s0)
        pre[a:b] = x
        prp[a:b] = np.where(np.isfinite(x), pd.Series(x).rank(method="min").to_numpy(), 99.0)
    return pd.DataFrame({"race_id": rid, "hn": f["hn"].to_numpy(), "pre": pre, "pre_pop": prp})


def cmd_diag() -> None:
    """γ の符号の診断（オッズだけ）: 正規化した市場の勝率 q の、t60 → t15 と t15 → 確定の変化の相関（確定の人気帯ごと）。
    正なら「締切前の勢いが確定まで続く＝確定に織り込みが遅れる」向き（γ > 0 寄り）。着順・払戻は読まない。"""
    from src.config.data_paths import TABLES_DIR
    from src.scripts.data import bet_cycles as BC
    pre = pd.read_parquet(BC.PRE_ODDS)
    pre["race_id"] = pre["race_id"].astype(str)
    pre["hn"] = pd.to_numeric(pre["horse_number"], errors="coerce")
    pre["o"] = pd.to_numeric(pre["win_odds"], errors="coerce")
    pre = pre.sort_values("snapshot_at")                # 同じ時点のスナップショットが複数あれば、取得時刻の最後
    w = pre[pre["phase"].isin(["t60", "t15", "t05"]) & (pre["o"] > 0)].pivot_table(index=["race_id", "hn"], columns="phase", values="o", aggfunc="last")
    fl = pd.read_parquet(Path(TABLES_DIR) / "2026" / "race_result_flat.parquet", columns=["race_id", "horse_number", "odds", "popularity"])
    fl["race_id"] = fl["race_id"].astype(str)
    fl["hn"] = pd.to_numeric(fl["horse_number"], errors="coerce")
    m = w.reset_index().merge(fl[["race_id", "hn", "odds", "popularity"]], on=["race_id", "hn"])
    m = m[(pd.to_numeric(m["odds"], errors="coerce") > 0)].dropna(subset=["t60", "t15", "t05"]).copy()
    for c in ("t60", "t15", "t05", "odds"):
        inv = 1.0 / pd.to_numeric(m[c])
        m["q_" + c] = inv / inv.groupby(m["race_id"]).transform("sum")
    m["a"] = np.log(m["q_t15"] / m["q_t60"])
    m["b"] = np.log(m["q_odds"] / m["q_t05"])         # t15 を共有しない区間（共有すると T-15 の雑音だけで負の相関が出る）
    m["band"] = pd.cut(pd.to_numeric(m["popularity"]), [0, 1, 3, 6, 9, 99], labels=["1", "2-3", "4-6", "7-9", "10-"])
    out = {"races": int(m["race_id"].nunique()), "pairs": int(len(m)), "corr_all": float(m["a"].corr(m["b"])),
           "by_band": {str(k): {"n": int(len(g)), "corr": float(g["a"].corr(g["b"])) if len(g) > 10 else None}
                       for k, g in m.groupby("band", observed=True)},
           "note": "t60→t15 の q の変化と t05→確定の変化の相関（中間の時点を共有しない）。着順・払戻は使っていない。pre_odds の T-60・T-15・T-05 がそろうレースだけ（2026-10-17 以降）"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diag_gamma.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["wf", "diag", "search", "final"])
    a = ap.parse_args(argv)
    if a.cmd == "wf":
        cmd_wf()
    elif a.cmd == "diag":
        cmd_diag()
    elif a.cmd == "search":
        from src.research.course import bet_opt_search as S
        S.cmd_search()
    else:
        from src.research.course import bet_opt_search as S
        S.cmd_final()


if __name__ == "__main__":
    main()
