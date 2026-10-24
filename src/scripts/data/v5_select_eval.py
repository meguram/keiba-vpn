"""v5 の構成を、V4 全体への累積（tp-27 方式、LightGBM、valid 2022・2023・2024 expanding、2 seed）で決める。

python -m src.scripts.data.v5_select_eval run NAME      # 構成 NAME の OOF を作る（再開可）。NAME は CONFIGS のキー
python -m src.scripts.data.v5_select_eval compare       # 構成どうしを対で比べる（採否の基準は feature_bakeoff.compare_oof）。results.json / report.md
python -m src.scripts.data.v5_select_eval list          # 構成の一覧

構成: a_V4（基準） / b_tlfresh（tl_ を再取得版に） / c_CY（b + tp-40 の CY） / d_CY_BEST（c + tp-38 の ALL_BEST） / e_CY_RLC（c + tp-42 の RL_C） / f_CY_BEST_RLC。
test 2025・2026 年は使わない。市場オッズ・人気は使わない。環境変数: V5_THREADS（既定 4）, V5_LOAD_LIMIT（既定 14）, V5_TP38_OUT（tp-38 の群の置き場）。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

os.environ.setdefault("TP40_THREADS", os.environ.get("V5_THREADS", "4"))
os.environ.setdefault("TP40_LOAD_LIMIT", os.environ.get("V5_LOAD_LIMIT", "14"))

from src.scripts.data import tp27_eval as t  # noqa: E402
from src.scripts.data import tp40_eval as e40  # noqa: E402

KEY = ["race_id", "horse_id"]
V = e40.V
OUT = V / "v5_select"
TP38_OUT = Path(os.environ.get("V5_TP38_OUT") or "/home/jovyan/work/tp38_out")
YEARS = [2022, 2023, 2024]
SEEDS = [0, 1]
BEST_PARTS = ["PX_cell_sd", "PX_emb", "PX_ridge_als"]

CONFIGS: dict[str, dict] = {
    "a_V4": {"base": "old", "groups": []},
    "b_tlfresh": {"base": "fresh", "groups": []},
    "c_CY": {"base": "fresh", "groups": ["CY"]},
    "d_CY_BEST": {"base": "fresh", "groups": ["CY", "BEST"]},
    "e_CY_RLC": {"base": "fresh", "groups": ["CY", "RLC"]},
    "f_CY_BEST_RLC": {"base": "fresh", "groups": ["CY", "BEST", "RLC"]},
}
# 対の比較（新, 基準）。累積の採否は『直前の構成に対して』と『V4 に対して』の両方を見る
PAIRS = [("b_tlfresh", "a_V4"), ("c_CY", "b_tlfresh"), ("d_CY_BEST", "c_CY"), ("e_CY_RLC", "c_CY"), ("f_CY_BEST_RLC", "c_CY"), ("f_CY_BEST_RLC", "d_CY_BEST"),
         ("d_CY_BEST", "b_tlfresh"), ("e_CY_RLC", "b_tlfresh"), ("f_CY_BEST_RLC", "b_tlfresh"), ("f_CY_BEST_RLC", "e_CY_RLC"),
         ("c_CY", "a_V4"), ("d_CY_BEST", "a_V4"), ("e_CY_RLC", "a_V4"), ("f_CY_BEST_RLC", "a_V4")]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _merge_keys(df: pd.DataFrame, T: pd.DataFrame) -> pd.DataFrame:
    T = T.drop_duplicates(KEY).copy()
    T["race_id"], T["horse_id"] = T["race_id"].astype(str), T["horse_id"].astype(str)
    m = df[KEY].merge(T, on=KEY, how="left")
    assert len(m) == len(df), (len(m), len(df))
    m = m[[c for c in m.columns if c not in KEY]]
    m.index = df.index
    return m


def load_group(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if name == "CY":
        return e40._load_group("CY", df)
    if name == "BEST":
        parts = [_merge_keys(df, pd.read_parquet(TP38_OUT / "groups" / f"{p}.parquet")) for p in BEST_PARTS]
        G = pd.concat(parts, axis=1)
        return G.loc[:, ~G.columns.duplicated()]
    if name == "RLC":
        return _merge_keys(df, pd.read_parquet(V / "tp42/groups/RL_C.parquet"))
    raise KeyError(name)


def build_frame(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    os.environ["TP40_BASE"] = cfg["base"]
    X = e40._frame(df)
    if cfg["groups"]:
        G = pd.concat([load_group(g, df) for g in cfg["groups"]], axis=1)
        dup = set(G.columns) & set(X.columns)
        assert not dup, f"V4 の列と名前が重なる: {sorted(dup)[:5]}"
        X = pd.concat([X, G], axis=1)
    return X


def cmd_run(a) -> None:
    cfg = CONFIGS[a.name]
    d = OUT
    d.mkdir(parents=True, exist_ok=True)
    resp = d / "levels.json"
    res = e40._jload(resp)
    if a.name in res and (d / f"oof_{a.name}.parquet").exists():
        _log(f"{a.name}: 既に完了")
        return
    df, base, meta, folds, lp = t.setup()
    X = build_frame(cfg, df)
    lv, oof = e40._run_subset(a.name, X, meta, folds, lp, YEARS, SEEDS)
    oof.to_parquet(d / f"oof_{a.name}.parquet", index=False)
    res = e40._jload(resp)
    res[a.name] = {"levels": lv, "cols": int(X.shape[1]), "groups": cfg["groups"], "base": cfg["base"]}
    e40._jsave(resp, res)


def cmd_compare(a) -> None:
    df, base, meta, folds, lp = t.setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    rm = pd.read_parquet(e40.OUT / "run_meta.parquet")
    rm["race_id"], rm["horse_id"] = rm["race_id"].astype(str), rm["horse_id"].astype(str)
    oofs = {n: pd.read_parquet(OUT / f"oof_{n}.parquet") for n in CONFIGS if (OUT / f"oof_{n}.parquet").exists()}
    out: dict = {"pairs": {}, "n_configs": len(CONFIGS)}
    for new, old in PAIRS:
        if new not in oofs or old not in oofs:
            continue
        c = t._cmp(oofs[old], oofs[new], fin_map, f"{new} vs {old}")
        by_year = {}
        for y in YEARS:
            ids = set(meta.loc[meta["year"] == y, "race_id"].astype(str))
            a0, b0 = oofs[old], oofs[new]
            ma, mb = a0["race_id"].astype(str).isin(ids), b0["race_id"].astype(str).isin(ids)
            if ma.sum() > 1000 and mb.sum() > 1000:
                cy = t._cmp(a0[ma], b0[mb], fin_map, f"{new} vs {old} {y}")
                by_year[str(y)] = {"Δwin": cy["Δwin"], "win_hi": cy["win_hi"], "Δtop2": cy["Δtop2"], "Δtop3": cy["Δtop3"]}
        sub = e40._subset_cmp(oofs[old], oofs[new], fin_map, rm, f"{new} vs {old}")
        out["pairs"][f"{new} vs {old}"] = {"overall": c, "by_year": by_year, "subsets": sub}
        _log(f"{new} vs {old}: Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f} 採用={c['採用']}")
    out["levels"] = e40._jload(OUT / "levels.json")
    e40._jsave(OUT / "results.json", out)


def cmd_list(a) -> None:
    for n, c in CONFIGS.items():
        print(n, c)


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("name", choices=list(CONFIGS))
    sp.add_parser("compare")
    sp.add_parser("list")
    a = ap.parse_args()
    {"run": cmd_run, "compare": cmd_compare, "list": cmd_list}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
