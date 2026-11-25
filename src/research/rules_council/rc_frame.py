"""全走の表（2016〜24、全クラス、芝ダ）。議題ごとの母集団は、この表を発走前に分かる値（T-60 の再現オッズ・クラスなど）で絞って作る。

- 列: 確定の q（1/確定オッズ をレースの中で和 1）・確定オッズ・確定の人気・T-60 の再現オッズ（simulate_pre の種 1000〜1000+n-1）、
  着順・単複の払戻（払戻のある走だけ）、開催（race_id の先頭 8 桁）、開催の 5 分割（種 20261124）、確定オッズの帯（層・物差し用）。
- 水準: src/research/graded/catalog.py の水準（v3 の levels.parquet）から、longshot_rules と同じ規則の型（L・M の族、start_year 2019 以上、I09 を除く）。
- 2025 年以降の行は開かない（year<=2024 で絞り assert）。費用 0 円（手元の表だけ）。

使い方（nice -n 19）: python -m src.research.rules_council.rc_frame [--n-pre 5]
出力: <COURSE_OUT>/rules_council/full.parquet・full_levels.parquet・full_pre.npy・full_info.json
"""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from src.research.course import longshot_rules as LR
from src.research.course import run as CR

OUTW = CR.OUT / "rules_council"
YEAR_END = 2024
NFOLD = 5
SPLIT_SEED = 20261124
# 確定オッズの帯（層・物差し用。判定には使わない）
ODDS_EDGES = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5, 8.0, 10.0, 13.0, 16.0, 20.0, 25.0, 32.0, 40.0, 50.0, 70.0, 100.0, 150.0, 1e9])
FIELD_EDGES = [8, 13]
COLS = LR.COLS


def odds_bin(o: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(ODDS_EDGES, o, side="right") - 1, 0, len(ODDS_EDGES) - 2)


def build(n_pre: int) -> None:
    from src.research.course.v2 import attach_final_odds
    from src.research.megu_cycles.mc_data import simulate_pre
    OUTW.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(LR.FRAME, columns=COLS, filters=[("year", "<=", YEAR_END)])
    yr_all = pd.to_numeric(pd.read_parquet(LR.FRAME, columns=["year"])["year"]).to_numpy()
    keep_rows = np.flatnonzero(yr_all <= YEAR_END)
    assert len(keep_rows) == len(d) and int(pd.to_numeric(d["year"]).max()) <= YEAR_END, "2025 年以降の行が入っている"
    types = LR.cut_types()
    L = pd.read_parquet(LR.LEVELS, columns=types + ["__race_id", "__horse_id"]).iloc[keep_rows].reset_index(drop=True)
    if not ((d["race_id"].astype(str).to_numpy() == L["__race_id"].to_numpy()) & (d["horse_id"].astype(str).to_numpy() == L["__horse_id"].to_numpy())).all():
        raise ValueError("frame と levels の行がずれている")
    m = d["surf"].astype(str).isin(["芝", "ダ"]).to_numpy()
    d, L = d[m].reset_index(drop=True), L[m].reset_index(drop=True)
    d = attach_final_odds(d, range(2016, YEAR_END + 1))
    ok = (d["odds_final"] > 0).to_numpy()
    d, L = d[ok].reset_index(drop=True), L[ok].reset_index(drop=True)
    d["race_id"] = d["race_id"].astype(str)
    order = np.lexsort((pd.to_numeric(d["hn"], errors="coerce").to_numpy(), d["race_id"].to_numpy()))
    d, L = d.iloc[order].reset_index(drop=True), L.iloc[order].reset_index(drop=True)
    inv = 1.0 / d["odds_final"].to_numpy(float)
    d["q"] = inv / pd.Series(inv).groupby(d["race_id"].to_numpy()).transform("sum").to_numpy()
    # simulate_pre は内部で 2026 年の確定オッズ・人気と本物の T-60 のオッズを読む（ずれの元。着順・払戻は読まない）ので台帳に残す
    from src.ops import data_ledger as DL
    from src.research.rules_council import engine as E
    E.check_stop()
    with DL.guard(tool="rules_council.rc_frame", family="rules_council", version="frame_simulate_pre", period=("2026-10-18", "2026-11-23"),
                  purpose="eval_only", reads=("確定オッズ", "確定の人気", "T-60 のオッズ"),
                  reason="T-60 の再現のずれの元（2026 の発走前オッズ）。着順・払戻は読まない", session="rules-council") as tok:
        pre = simulate_pre(d, n=n_pre, phase="t60").astype(np.float32)
        tok.finish()
    d["pre"] = pre[0]
    d["kaisai"] = d["race_id"].str[:8]
    keys = np.array(sorted(d["kaisai"].unique()))
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(keys))
    fold_of = {k: i % NFOLD for i, k in enumerate(keys[perm])}
    d["fold"] = d["kaisai"].map(fold_of).to_numpy(int)
    fsha = hashlib.sha256(json.dumps(sorted(fold_of.items()), ensure_ascii=False).encode()).hexdigest()
    d["unit"] = d["venue"].astype(str) + d["surf"].astype(str) + d["dist"].astype(int).astype(str)
    d["vs"] = d["venue"].astype(str) + d["surf"].astype(str)
    d["cls"] = pd.to_numeric(d["cls"], errors="coerce").fillna(-1).astype(int)
    d["obin"] = odds_bin(d["odds_final"].to_numpy(float))
    d["field"] = pd.to_numeric(d["field"], errors="coerce").fillna(12).astype(int)
    d["fgrp"] = np.searchsorted(FIELD_EDGES, d["field"].to_numpy(), side="right")
    fin = pd.to_numeric(d["fin"], errors="coerce").to_numpy()
    pay = d["pay_ok"].fillna(False).astype(bool).to_numpy()
    d["rt"] = np.where(pay, pd.to_numeric(d["ret_tan"], errors="coerce").fillna(0).to_numpy(), 0.0)
    d["rf"] = np.where(pay, pd.to_numeric(d["ret_fuku"], errors="coerce").fillna(0).to_numpy(), 0.0)
    d["win"] = ((fin == 1) & pay).astype(np.int8)
    d["top3"] = ((fin <= 3) & (d["rf"].to_numpy() > 0)).astype(np.int8)
    d["pay_ok"] = pay
    L = L.drop(columns=["__race_id", "__horse_id"])
    d.to_parquet(OUTW / "full.parquet", index=False)
    L.to_parquet(OUTW / "full_levels.parquet", index=False)
    np.save(OUTW / "full_pre.npy", pre)
    info = {"rows": int(len(d)), "races": int(d["race_id"].nunique()), "kaisai": int(len(keys)), "fold_sha256": fsha,
            "n_pre": n_pre, "pre_seeds": [1000, 1000 + n_pre - 1], "types": list(L.columns),
            "pre_sha256": hashlib.sha256(np.round(pre[0].astype(float), 4).tobytes()).hexdigest(), "years": [2016, YEAR_END]}
    (OUTW / "full_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print({k: v for k, v in info.items() if k != "types"}, flush=True)


def load(cols: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_parquet(OUTW / "full.parquet", columns=cols)
    L = pd.read_parquet(OUTW / "full_levels.parquet")
    assert len(d) == len(L)
    if "year" in d:
        assert int(pd.to_numeric(d["year"]).max()) <= YEAR_END
    return d, L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pre", type=int, default=5)
    a = ap.parse_args()
    build(a.n_pre)


if __name__ == "__main__":
    main()
