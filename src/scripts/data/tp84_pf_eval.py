"""tp-84: 予想ペース × 馬のタイプ・感応度の特徴量（tp84_pacefit）を、tp-83 と同じ土台（V4 の特徴量表・基準 OOF・段階評価）で評価する。ホストで実行。

python -m src.scripts.data.tp84_pf_eval groups   # 群を作って tp84_pf/groups/*.parquet に保存（先に tp84_pace_to_type を実行して stage1_pred.parquet を作る）
python -m src.scripts.data.tp84_pf_eval stage1   # 第 1 段: valid 2022+2023、seed 1 本
python -m src.scripts.data.tp84_pf_eval stage2   # 第 2 段: 第 1 段を通った群を valid 2024、新 seed 2 本
環境変数: TP84_PP（予想ペースの表。既定 /tmp/tp84_out/stage1_pred.parquet）, TP84_GROUPS, TP84_LABEL, TP39_THREADS, TP39_LOAD_LIMIT, TP39_MAX_HEAVY。
test 2025・2026 年は使わない。市場オッズ・人気は使わない。
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from src.pipeline.features import factory_self as fs  # noqa: E402
from src.pipeline.features import tp84_pacefit as pf  # noqa: E402
from src.scripts.data import tp39_eval as te  # noqa: E402
from src.scripts.data import tp42_eval as te42  # noqa: E402
from src.scripts.data import tp84_pace_eval as ev  # noqa: E402

OUT = te.V / "tp84_pf"
te.OUT = OUT
te42.OUT = OUT
GROUPS = ["PF_A", "PF_B", "PF_ALL"]


def _log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def real_pace() -> tuple[pd.DataFrame, dict]:
    rt = pd.read_parquet(ev.FEAT)
    rt = rt[rt["surface"].isin(["芝", "ダート"])].copy()
    rt = ev.race_table(rt)
    fit = rt[pd.to_datetime(rt["date"]) <= ev.PAR_END]
    for c in ("F5", "RPCI"):
        rt["d_" + c] = rt[c].to_numpy() - ev.par_scalar(fit, rt, c)
    sd = {c: float(rt.loc[fit.index, "d_" + c].std()) for c in ("F5", "RPCI")}
    return rt[["race_id", "d_F5", "d_RPCI"]], sd


def cmd_groups(a) -> None:
    runs = fs.load_runs(te.TABLES, list(range(2020, 2026)))
    d = fs.prep(runs).sort_values(["horse_id", "dnum", "race_id"], kind="stable").reset_index(drop=True)
    real, sd = real_pace()
    pace = pd.read_parquet(os.environ.get("TP84_PP", "/tmp/tp84_out/stage1_pred.parquet"))
    _log(f"走 {len(d):,} 行 / 予想ペースのレース {len(pace):,} / 標準偏差 {sd}")
    gs = pf.build(d, pace, real, sd)
    gdir = OUT / "groups"
    gdir.mkdir(parents=True, exist_ok=True)
    yr = d["date"].dt.year.to_numpy()
    cov = {}
    for n, t in gs.items():
        t.to_parquet(gdir / f"{n}.parquet", index=False)
        cols = [c for c in t.columns if c not in pf.KEY]
        by = {int(y): round(float(t.loc[yr == y, cols].notna().mean().mean()), 3) for y in (2021, 2022, 2023, 2024, 2025)}
        cov[n] = {"cols": cols, "nonnull_rate_by_year": by}
        _log(f"{n}: {len(cols)} 列 被覆 {by}")
    te._jsave(OUT / "consts.json", {"sd": sd, "params": {"L": pf.L, "DECAY": pf.DECAY, "LAM_SLOPE": pf.LAM_SLOPE, "LAM_KERN": pf.LAM_KERN, "BW": pf.BW}})
    te._jsave(OUT / "coverage.json", cov)
    shutil.copy(os.environ.get("TP84_PP", "/tmp/tp84_out/stage1_pred.parquet"), OUT / "pace_pred.parquet")


def _groups() -> list[str]:
    return [g for g in os.environ.get("TP84_GROUPS", "").split(",") if g] or GROUPS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["groups", "stage1", "stage2"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "groups":
        cmd_groups(a)
    elif a.cmd == "stage1":
        te._stage(os.environ.get("TP84_LABEL") or "stage1", [2022, 2023], [0], _groups(), "stage1")
    else:
        te._stage(os.environ.get("TP84_LABEL") or "stage2", [2024], [2, 3], _groups(), "stage2")


if __name__ == "__main__":
    main()
