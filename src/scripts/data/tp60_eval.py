"""tp-60 特徴量の群を V5（1,142 列）に累積して 2 段階で評価する（tp-58 / v5_select_eval と同じ基準）。

python -m src.scripts.data.tp60_eval stage1 --groups DEEP_TOT,DEEP_6_10,...     # 検証 2022+2023・seed 0（粗い選抜）
python -m src.scripts.data.tp60_eval stage2 --groups ...                        # 検証 2024・新 seed(2,3)（通過群のみ）
python -m src.scripts.data.tp60_eval full --groups ...                          # 検証 2022〜2024・seed 0,1 を V5 の OOF と比べる（最終候補のみ）

群の置き場: <TP60_OUT>/groups/<NAME>.parquet。race_id・horse_id があれば行単位、horse_id だけなら馬単位で結合。'A+B' は群の連結。
基準: compare_oof（勝率 95% ブートストラップ上限 < 0 かつ効果 ≥ 0.0002、連対・複勝の悪化 ≤ 0.0002）。test 2025・2026 は使わない。市場オッズ・人気は使わない。
環境変数: TP60_THREADS（既定 4）, TP60_LOAD_LIMIT（既定 14）。
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd

os.environ.setdefault("TP40_THREADS", os.environ.get("TP60_THREADS", "4"))
os.environ.setdefault("TP40_LOAD_LIMIT", os.environ.get("TP60_LOAD_LIMIT", "14"))

from src.scripts.data import tp27_eval as t  # noqa: E402
from src.scripts.data import tp40_eval as e40  # noqa: E402
from src.scripts.data import v5_select_eval as v5  # noqa: E402

KEY = ["race_id", "horse_id"]
OUTD = Path(os.environ.get("TP60_OUT") or "/home/jovyan/work/tp60_out")
GROUPS = OUTD / "groups"
EVAL = OUTD / "eval"
BASE_OOF = v5.OUT / "oof_d_CY_BEST.parquet"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_group(name: str, df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for n in name.split("+"):
        T = pd.read_parquet(GROUPS / f"{n}.parquet")
        T["horse_id"] = T["horse_id"].astype(str)
        if "race_id" in T.columns:
            T["race_id"] = T["race_id"].astype(str)
            T = T.drop_duplicates(KEY)
            m = df[KEY].merge(T, on=KEY, how="left")
        else:
            T = T.drop_duplicates("horse_id")
            m = df[["horse_id"]].merge(T, on="horse_id", how="left")
        m = m[[c for c in m.columns if c not in KEY]]
        m.index = df.index
        parts.append(m)
    G = pd.concat(parts, axis=1)
    return G.loc[:, ~G.columns.duplicated()]


def _stage(label: str, years: list[int], seeds: list[int], groups: list[str]) -> None:
    df, base, meta, folds, lp = t.setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    rm = pd.read_parquet(e40.OUT / "run_meta.parquet")
    rm["race_id"], rm["horse_id"] = rm["race_id"].astype(str), rm["horse_id"].astype(str)
    X0 = v5.build_frame(v5.CONFIGS["d_CY_BEST"], df)
    tag = os.environ.get("TP60_EVAL_TAG", "")
    d = EVAL / (label + tag)
    d.mkdir(parents=True, exist_ok=True)
    resp = d / "results.json"
    res = e40._jload(resp)
    bp = d / "base.parquet"
    if tag and not bp.exists() and (EVAL / label / "base.parquet").exists():                 # 並列に回す別タグは、同じ土台（V5）の OOF をコピーして使う
        import shutil

        shutil.copy(EVAL / label / "base.parquet", bp)
        res["base"] = e40._jload(EVAL / label / "results.json").get("base", {})
        e40._jsave(resp, res)
    if label == "full":
        oof0 = pd.read_parquet(BASE_OOF)
        oof0["race_id"], oof0["horse_id"] = oof0["race_id"].astype(str), oof0["horse_id"].astype(str)
        res["base"] = {"cols": int(X0.shape[1]), "from": str(BASE_OOF)}
    elif bp.exists() and "base" in res:
        oof0 = pd.read_parquet(bp)
    else:
        lv, oof0 = e40._run_subset("V5", X0, meta, folds, lp, years, seeds)
        oof0.to_parquet(bp, index=False)
        res["base"] = {"levels": lv, "cols": int(X0.shape[1])}
        e40._jsave(resp, res)
    steps = res.setdefault("groups", {})
    for g in groups:
        if g in steps:
            continue
        G = load_group(g, df)
        dup = set(G.columns) & set(X0.columns)
        assert not dup, f"V5 の列と名前が重なる: {sorted(dup)[:5]}"
        lv, oof = e40._run_subset(f"V5+{g}", pd.concat([X0, G], axis=1), meta, folds, lp, years, seeds)
        oof.to_parquet(d / f"oof_{g}.parquet", index=False)
        c = t._cmp(oof0, oof, fin_map, g)
        steps[g] = {"cols": int(G.shape[1]), "levels": lv, "vs_V5": c, "subsets": e40._subset_cmp(oof0, oof, fin_map, rm, g)}
        e40._jsave(resp, res)
        _log(f"  {g}: Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f} 採用={c['採用']}")
        for nm, s in steps[g]["subsets"].items():
            _log(f"      {nm}（{s['rows']:,} 行）: Δwin {s['Δwin']:+.5f} (hi {s['win_hi']:+.5f})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage1", "stage2", "full"])
    ap.add_argument("--groups", required=True)
    a = ap.parse_args()
    groups = a.groups.split(",")
    if a.stage == "stage1":
        _stage("stage1", [2022, 2023], [0], groups)
    elif a.stage == "stage2":
        _stage("stage2", [2024], [2, 3], groups)
    else:
        _stage("full", [2022, 2023, 2024], [0, 1], groups)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
