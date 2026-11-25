"""前向きの記録（1 号の指示 1。宣言には使わない・将来の評価の材料を貯めるだけ）。手で流す道具（cron・常駐は入れない）。

    KEIBA_SESSION=pedigree-lab nice -n 19 python -m src.research.pedigree_lab.forward_record --date 2026-11-28

- その日より前の結果（race_result_flat）だけで as-of の特徴量を作り、その日の出馬表（race_shutuba_flat）の馬に、
  凍結の第 4 版（2 代母の単位の 5 折の平均）の勝率を当てる。初ダート（FD）・初の距離帯（DB0）の印も付ける。
- 出力: docs/operations/pedigree_lab/forward/<日付>.csv（commit の時刻で発走前を示す）。オッズは使わない。
- 0 円（手元の表と 5 代血統の JSON だけ。GCS を読まない）。開催日の 07:30〜17:00 には流さない。本番・画面には出さない。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.pedigree_lab import build_base, data, features4, final_eval, gates, scene_dist

ROOT = Path(__file__).resolve().parents[3]
OUTDIR = ROOT / "docs/operations/pedigree_lab/forward"
JST = timezone(timedelta(hours=9))


def entries(day: pd.Timestamp) -> pd.DataFrame:
    S = pd.read_parquet(build_base.TABLES / str(day.year) / "race_shutuba_flat.parquet")
    S["date"] = pd.to_datetime(S["date"])
    S = S[(S["date"] == day) & S["surface"].isin(["芝", "ダート"])].copy()
    S["horse_id"] = S["horse_id"].astype(str)
    S["finish_position"] = 99            # 未確定（as-of の集計では、その日より後の行が無いので使われない）
    S["finish_status"] = None
    S["popularity"] = np.nan
    S["payoff"] = None
    S["odds"] = np.nan                   # オッズは使わない
    S["meeting"] = S["race_id"].str[:8]
    S["sex"] = S["sex_age"].str[0]
    S["age"] = pd.to_numeric(S["sex_age"].str[1:], errors="coerce")
    return S.reindex(columns=[*build_base.COLS, "meeting", "sex", "age"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    a = ap.parse_args()
    day = pd.Timestamp(a.date)
    data.TOKENS.append(data._guard(("2016-01-01", str((day - pd.Timedelta(days=1)).date())), "forward", f"forward_{a.date}",
                                   f"前向きの記録 {a.date}（as-of の特徴量の元。凍結の第 4 版で当てる。評価ではない）"))
    runs = build_base.load_runs()
    runs = runs[runs["date"] < day]
    E = entries(day)
    if E.empty:
        raise SystemExit(f"{a.date} の出馬表が無い")
    r = data._prep(pd.concat([runs, E], ignore_index=True))
    ped = data.load_ped()
    miss = sorted(set(E["horse_id"]) - set(ped["horse_id"]))
    if miss:
        ped = pd.concat([ped, build_base.load_ped(miss)], ignore_index=True)
    X, _ = features4.build(r, ped, sub=True)
    X["split"] = -1                      # gates.FCOLS の列をそろえる（前向きの行は分けに入らない）
    F = gates.scenes(X[gates.FCOLS].copy())
    D = scene_dist.marks(X[list(dict.fromkeys(gates.FCOLS + ["dband", "going"]))].copy())
    X["FD"], X["DB0"] = F["FD"].to_numpy(), D["DB0"].to_numpy()
    T = X[X["date"] == day].copy()
    p = final_eval.predict(T, "v4")
    T["p_win"] = p
    T["rank"] = T.groupby("race_id")["p_win"].rank(ascending=False, method="first").astype(int)
    out = T[["race_id", "date", "venue", "surface", "distance", "horse_number", "horse_id", "p_win", "rank", "FD", "DB0"]].copy()
    out["p_win"] = out["p_win"].round(5)
    out["date"] = out["date"].dt.date.astype(str)
    out["no_pedigree"] = out["horse_id"].isin(miss).to_numpy() & T["sire"].eq("").to_numpy()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"{a.date}.csv"
    meta = {"made_at": datetime.now(JST).isoformat(timespec="seconds"), "model": "v4_pure_dd（freeze_v4_gate.sha256）",
            "races": int(out["race_id"].nunique()), "horses": len(out), "no_pedigree": int(out["no_pedigree"].sum())}
    out.sort_values(["race_id", "rank"]).to_csv(path, index=False)
    (OUTDIR / f"{a.date}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(json.dumps(meta, ensure_ascii=False))
    data.finish_all(str(path))


if __name__ == "__main__":
    main()
