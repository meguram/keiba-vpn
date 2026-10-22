"""v6 の候補（新しい知見の使い方）を v5 の構成 d（V4 + tl_ 再取得 + CY + ALL_BEST）への累積で評価する（v5_select_eval と同じ手順）。

python -m src.scripts.data.v6_eval list
python -m src.scripts.data.v6_eval run NAME [--tag s2 --years 2024 --seeds 2,3]    # 構成 NAME の OOF（既定: valid 2022〜24 expanding・seed 0,1）。基準 d_CY_BEST も同じ設定で作れる
python -m src.scripts.data.v6_eval compare [--tag s2]                              # d_CY_BEST との対の比較（採否は feature_bakeoff.compare_oof の基準）

構成（すべて d_CY_BEST に足す）:
  h_E2     追走難度 v2（EASE2: drain / demand / full / capacity / pct / vs_prev）
  i_E2R    h + 追走難度・位置取り予想・上がり能力のレース内相対（順位・平均との差）
  j_LAPX   ラップ予想（LAP 12 列）+ シナリオの不確実性（エントロピー等）+ 位置取り×シナリオ・追走難度×シナリオの交互作用
  k_ALLN   h + i + j のすべて
  l_S3A    前段 S3 を新しい入力（tl_ 再取得 + CY）で作り直したものに差し替え
  m_S3B    前段 S3 を a + 追走難度 v2・ラップ予想・位置取り予想の入力で作り直したものに差し替え
  n_S3C    前段 S3 をシナリオ条件付き（スロー/ミドル/ハイで予測し、ラップ予想の確率で混合）に差し替え + 混合の幅・感度の列
  o_S3C_ALLN  n + k
  q_PT     展開タイプ系（tp-61）: 事前の展開確率・ハナ主張馬・脚質×展開の期待有利度（as-of）
  r_GB     コース別の枠順バイアス（年ごとの as-of 係数）: gb_shift・補正後の想定位置・枠×最初のコーナーまでの距離・外枠の逃げ馬
  s_CB     クラス基準（年齢×クラスの 2 着馬の指数の突破ライン、年ごとの as-of）: 直近5走・同条件走の最高指数−ライン
  p_S3D    ペース補正後の走破力（s_adj_time / s_adj_l3f: 時計・上がりからレース全体のペースの効果を引いた目的変数を予測）を追加列として足す
test 2025・2026 年は使わない。市場オッズ・人気は使わない。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TP40_THREADS", os.environ.get("V6_THREADS", "4"))
os.environ.setdefault("TP40_LOAD_LIMIT", os.environ.get("V6_LOAD_LIMIT", "14"))

from src.pipeline.features import v6_groups as vg  # noqa: E402
from src.scripts.data import tp27_eval as t  # noqa: E402
from src.scripts.data import tp40_eval as e40  # noqa: E402
from src.scripts.data import v5_select_eval as sel  # noqa: E402
from src.scripts.data import v6_stage as st  # noqa: E402

KEY = ["race_id", "horse_id"]
V = e40.V
OUT = Path(os.environ.get("V6_OUT") or V / "v6")
YEARS = [2022, 2023, 2024]
SEEDS = [0, 1]
BASE_CFG = {"base": "fresh", "groups": ["CY", "BEST"]}
CONFIGS: dict[str, dict] = {
    "d_CY_BEST": {"new": [], "stage": None},
    "h_E2": {"new": ["E2"], "stage": None},
    "i_E2R": {"new": ["E2", "E2R"], "stage": None},
    "j_LAPX": {"new": ["LAP", "LPX"], "stage": None},
    "k_ALLN": {"new": ["E2", "E2R", "LAP", "LPX"], "stage": None},
    "l_S3A": {"new": [], "stage": "a"},
    "m_S3B": {"new": [], "stage": "b"},
    "n_S3C": {"new": [], "stage": "c"},
    "o_S3C_ALLN": {"new": ["E2", "E2R", "LAP", "LPX"], "stage": "c"},
    "p_S3D": {"new": [], "stage": "d"},
    "q_PT": {"new": ["PT"], "stage": None},
    "r_GB": {"new": ["GB"], "stage": None},
    "s_CB": {"new": ["CB"], "stage": None},
}
STAGE_COLS = ["s_time_pred", "s_time_gap", "s_rank_pred"]
CAL_COLS = ["fe_year_frac", "fe_month", "fe_week", "fe_dom", "fe_weekday", "fe_sin1", "fe_cos1", "fe_sin2", "fe_cos2", "fe_sin3", "fe_cos3"]
WORK = Path(os.environ.get("V6_WORK") or "/home/jovyan/work")
# 他の fork が作った群（行キー race_id, horse_id）。名前 → (parquet, 使う列 or None=キー以外すべて)
EXTERNAL: dict[str, tuple[Path, list[str] | None]] = {
    "NJ": (WORK / "acq/groups/NJ.parquet", None),
    "RB": (WORK / "tp65_out/rob.parquet", None),
    "PB4": (WORK / "tp64_out/pb_features.parquet", ["pb_ben_m3", "pb_ben_1", "pb_exp_rel", "pb_pos_sd5"]),
}


try:    # 復活セット（config/feature_sets.json）の群を、REV_ で始まる名前の外部群として読めるようにする。既存の名前は上書きしない
    from src.pipeline.features import feature_sets as _fs

    for _n, _v in _fs.external_groups().items():
        EXTERNAL.setdefault(_n, _v)
except (OSError, KeyError, ValueError):
    pass


def config_of(name: str) -> dict:
    """NAME が CONFIGS にあればそれ、無ければ 'z:群1+群2@段' の形（段は a/b/c/d、省略可）をその場で解釈する。"""
    if name in CONFIGS:
        return CONFIGS[name]
    if name.startswith("z:"):
        body, _, stage = name[2:].partition("@")
        return {"new": [g for g in body.split("+") if g], "stage": stage or None}
    raise KeyError(name)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _forecast_frame(df: pd.DataFrame) -> pd.DataFrame:
    F = st.forecast_inputs(df)
    F.insert(0, "race_id", df["race_id"].to_numpy())
    F.insert(1, "horse_id", df["horse_id"].to_numpy())
    return F


def load_new(name: str, df: pd.DataFrame, F: pd.DataFrame) -> pd.DataFrame:
    ease = [c for c in F.columns if c.startswith("EASE2_")]
    lap = [c for c in F.columns if c.startswith("lp_")]
    if name == "E2":
        return F[ease].astype("float32")
    if name == "E2R":
        cols = ["EASE2_drain", "EASE2_capacity", "EASE2_demand", "EASE2_full", "pf_early", "td_l3f_ability"]
        return vg.race_rel(F, cols)
    if name == "RS59":
        from src.scripts.data import tp59_feat as f59

        out = f59.rs_features(df)
        out.index = df.index
        return out
    if name == "LAP":
        return F[lap].astype("float32")
    if name == "LPX":
        return pd.concat([vg.lap_entropy(F), vg.interactions(F)], axis=1)
    if name in ("PT", "GB", "CB", "OWNN") or name in EXTERNAL:
        G = pd.read_parquet(EXTERNAL[name][0] if name in EXTERNAL else OUT / "groups" / f"{name}.parquet")
        if name in EXTERNAL and EXTERNAL[name][1]:
            G = G[KEY + EXTERNAL[name][1]]
        G["race_id"], G["horse_id"] = G["race_id"].astype(str), G["horse_id"].astype(str)
        m = df[KEY].astype(str).merge(G, on=KEY, how="left")
        assert len(m) == len(df)
        m = m[[c for c in m.columns if c not in KEY]]
        m.index = df.index
        return m
    raise KeyError(name)


def build_frame(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    X = sel.build_frame(BASE_CFG | {}, df)
    if cfg["stage"]:
        S = pd.read_parquet(OUT / f"s3x_{cfg['stage']}.parquet")
        S["race_id"], S["horse_id"] = S["race_id"].astype(str), S["horse_id"].astype(str)
        m = df[KEY].astype(str).merge(S, on=KEY, how="left")
        assert len(m) == len(df)
        m.index = df.index
        for c in STAGE_COLS:
            if c in m.columns:
                X[c] = m[c].to_numpy()
        extra = [c for c in m.columns if c not in KEY + STAGE_COLS]
        for c in extra:
            X[c] = m[c].to_numpy()
    drop = [n for n in cfg["new"] if n.startswith("-")]
    if drop:
        from src.scripts.data import tp59_feat as f59

        if "-RIDER" in drop:
            X = X.drop(columns=list(f59.RIDER_COLS))
        if "-YF" in drop:
            X = X.drop(columns=["fe_year_frac"])
        if "-CAL" in drop:
            X = X.drop(columns=list(CAL_COLS))
    cfg = {**cfg, "new": [n for n in cfg["new"] if not n.startswith("-")]}
    if cfg["new"]:
        F = _forecast_frame(df)
        G = pd.concat([load_new(n, df, F) for n in cfg["new"]], axis=1)
        G = G.loc[:, ~G.columns.duplicated()]
        dup = set(G.columns) & set(X.columns)
        assert not dup, f"列名が重なる: {sorted(dup)[:5]}"
        G.index = X.index
        X = pd.concat([X, G], axis=1)
    return X


def _paths(tag: str) -> tuple[Path, Path]:
    d = OUT / ("eval" if not tag else f"eval_{tag}")
    d.mkdir(parents=True, exist_ok=True)
    return d, d / "levels.json"


def cmd_run(a) -> None:
    cfg = config_of(a.name)
    d, resp = _paths(a.tag)
    years = [int(x) for x in a.years.split(",")] if a.years else YEARS
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else SEEDS
    res = e40._jload(resp) or {}
    if a.name in res and (d / f"oof_{a.name}.parquet").exists():
        _log(f"{a.name}: 既に完了")
        return
    df, base, meta, folds, lp = t.setup()
    X = build_frame(cfg, df)
    _log(f"{a.name}: 列 {X.shape[1]}（基準 d は 1,142 列）")
    lv, oof = e40._run_subset(a.name, X, meta, folds, lp, years, seeds)
    oof.to_parquet(d / f"oof_{a.name}.parquet", index=False)
    res = e40._jload(resp) or {}
    res[a.name] = {"levels": lv, "cols": int(X.shape[1]), "new": cfg["new"], "stage": cfg["stage"], "years": years, "seeds": seeds}
    e40._jsave(resp, res)


def cmd_compare(a) -> None:
    d, resp = _paths(a.tag)
    df, base, meta, folds, lp = t.setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    rm = pd.read_parquet(e40.OUT / "run_meta.parquet")
    rm["race_id"], rm["horse_id"] = rm["race_id"].astype(str), rm["horse_id"].astype(str)
    oofs = {p.stem[4:]: pd.read_parquet(p) for p in sorted(d.glob("oof_*.parquet")) if not p.stem.endswith(".v5select")}
    base = a.base
    if base not in oofs:
        raise SystemExit(f"基準 {base} の OOF がありません")
    out: dict = {"pairs": {}}
    only = {x for x in a.only.split(",") if x}
    for n, o in oofs.items():
        if n == base or (only and n not in only):
            continue
        c = t._cmp(oofs[base], o, fin_map, f"{n} vs {base}")
        by_year = {}
        for y in YEARS:
            ids = set(meta.loc[meta["year"] == y, "race_id"].astype(str))
            a0, b0 = oofs[base], o
            ma, mb = a0["race_id"].astype(str).isin(ids), b0["race_id"].astype(str).isin(ids)
            if ma.sum() > 1000 and mb.sum() > 1000:
                cy = t._cmp(a0[ma], b0[mb], fin_map, f"{n} vs {base} {y}")
                by_year[str(y)] = {"Δwin": cy["Δwin"], "win_hi": cy["win_hi"], "Δtop2": cy["Δtop2"], "Δtop3": cy["Δtop3"]}
        sub = e40._subset_cmp(oofs[base], o, fin_map, rm, f"{n} vs {base}")
        out["pairs"][n] = {"overall": c, "by_year": by_year, "subsets": sub}
        _log(f"{n} vs {base}: Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f} 採用={c['採用']}")
    out["levels"] = e40._jload(resp)
    e40._jsave(d / ("results.json" if base == "d_CY_BEST" else f"results_vs_{base}.json"), out)


def cmd_list(a) -> None:
    for n, c in CONFIGS.items():
        print(n, c)


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("name")
    r.add_argument("--tag", default="")
    r.add_argument("--years", default="")
    r.add_argument("--seeds", default="")
    c = sp.add_parser("compare")
    c.add_argument("--tag", default="")
    c.add_argument("--base", default="d_CY_BEST")
    c.add_argument("--only", default="", help="カンマ区切りの名前だけを比べる")
    sp.add_parser("list")
    a = ap.parse_args()
    {"run": cmd_run, "compare": cmd_compare, "list": cmd_list}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
