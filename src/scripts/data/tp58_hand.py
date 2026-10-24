"""tp-58 手設計のカテゴリ掛け合わせ（as-of 目的変数統計 + 階層縮約）を作り、V5 の OOF に対するスコア検定で上乗せの有無を見る。

python -m src.scripts.data.tp58_hand build   # $TP58_DIR/out/hand_crosses.parquet（全行、キー + x58_* 列）
python -m src.scripts.data.tp58_hand score   # 発見期間 2022–23・確認期間 2024、親列を除いた有効スコアも
各クロスは『子キーの組』と『親キーの組』（子は親へ縮約）。同日は集計に入れない。市場オッズ・人気は使わない。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features.tp58_cross import shrunk_cross
from src.pipeline.features.tp58_score import ScoreTest

ROOT = Path(__file__).resolve().parents[3]
WORKDIR = Path(os.environ.get("TP58_DIR", ROOT / "data/local/modeling/shared_cache/tp58"))
CACHE, OUT = WORKDIR / "cache", WORKDIR / "out"
OOF = ROOT / "data/local/modeling/training_pipeline_v4/v5_select/oof_d_CY_BEST.parquet"
KEY = ["race_id", "horse_id"]
DISC, CONF = (2022, 2023), (2024,)

# (名前, 子キー, 親キー)  ※ 派生列: dist_band, month, course = 場×面×距離, vsd = 場×面×距離帯
CROSSES = [
    ("jk_tr", ["jockey_id", "trainer_id"], ["jockey_id"]),
    ("jk_course", ["jockey_id", "course"], ["jockey_id"]),
    ("jk_vsd", ["jockey_id", "vsd"], ["jockey_id"]),
    ("tr_class", ["trainer_id", "race_class", "grade"], ["trainer_id"]),
    ("tr_venue", ["trainer_id", "venue"], ["trainer_id"]),
    ("sire_cond", ["sire", "surface", "track_condition"], ["sire"]),
    ("damsire_dist", ["dam_sire", "surface", "dist_band"], ["dam_sire"]),
    ("age_season_sex", ["age", "season", "sex"], ["age", "sex"]),
    ("frame_course", ["bracket_number", "course"], ["course"]),
    ("frame_vsd_field", ["bracket_number", "vsd", "field_band"], ["vsd", "field_band"]),
    ("sire_vsd", ["sire", "vsd"], ["sire"]),
    ("jk_sire", ["jockey_id", "sire"], ["jockey_id"]),
]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _sire_map(horse_ids) -> pd.DataFrame:
    """馬ごとの父・母父の ID。ped_tbl の世代 1・2（path_fm が F / MF）から。学習表の sire/dam_sire は空のため。結果は out/sire_map.parquet に保存。"""
    f = OUT / "sire_map.parquet"
    if f.exists():
        return pd.read_parquet(f)
    from concurrent.futures import ThreadPoolExecutor

    base = ROOT / "data/local/features/horse/ped_tbl"

    def one(h):
        fp = base / h[:4] / f"{h}.parquet"
        try:
            d = pd.read_parquet(fp, columns=["path_fm", "ancestor_horse_id"])
        except Exception:
            return (h, None, None)
        g = dict(zip(d["path_fm"], d["ancestor_horse_id"]))
        return (h, g.get("F"), g.get("MF"))

    ids = sorted(set(horse_ids))
    with ThreadPoolExecutor(4) as ex:
        rows = list(ex.map(one, ids))
    m = pd.DataFrame(rows, columns=["horse_id", "sire", "dam_sire"])
    OUT.mkdir(parents=True, exist_ok=True)
    m.to_parquet(f, index=False)
    return m


def _frame() -> pd.DataFrame:
    F = pd.read_parquet(CACHE / "v5_frame_num.parquet", columns=KEY + ["finish_position", "year", "date"])
    R = pd.read_parquet(CACHE / "raw_cols.parquet")
    for d in (F, R):
        d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    R = R.drop_duplicates(KEY)
    D = F.merge(R.drop(columns=["sire", "dam_sire"], errors="ignore"), on=KEY, how="left", suffixes=("", "_raw"))
    D = D.merge(_sire_map(D["horse_id"].unique()), on="horse_id", how="left")
    D = D.sort_values(["date", "race_id"], kind="stable").reset_index(drop=True)
    dist = pd.to_numeric(D["distance"], errors="coerce")
    D["dist_band"] = pd.cut(dist, [0, 1300, 1500, 1700, 1900, 2200, 2700, 9999], labels=False).astype("string")
    D["month"] = pd.to_datetime(D["date"]).dt.month
    D["season"] = ((D["month"] % 12) // 3).astype("string")
    D["course"] = D["venue"].astype("string") + "|" + D["surface"].astype("string") + "|" + dist.astype("string")
    D["vsd"] = D["venue"].astype("string") + "|" + D["surface"].astype("string") + "|" + D["dist_band"].astype("string")
    fs = pd.to_numeric(D["field_size"], errors="coerce")
    D["field_band"] = pd.cut(fs, [0, 9, 13, 16, 99], labels=False).astype("string")
    return D


def cmd_build(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    D = _frame()
    fp = pd.to_numeric(D["finish_position"], errors="coerce").to_numpy()
    valid = ~np.isnan(fp)
    res = D[KEY].copy()
    for name, ck, pk in CROSSES:
        t0 = time.time()
        for tgt, y, prior in (("win", (fp == 1).astype(float), 0.07), ("top3", (fp <= 3).astype(float), 0.2)):
            o = shrunk_cross(D, ck, pk, y, valid, prior=prior)
            res[f"x58_{name}_{tgt}_m"] = o["m"].astype("float32")
            res[f"x58_{name}_{tgt}_dev"] = o["dev"].astype("float32")
            res[f"x58_{name}_{tgt}_parent"] = o["parent"].astype("float32")
        res[f"x58_{name}_n"] = o["n"].astype("float32")
        _log(f"{name}: {time.time() - t0:.1f}s")
    res.to_parquet(OUT / "hand_crosses.parquet", index=False)
    _log(f"保存 {res.shape}")


def cmd_score(a) -> None:
    H = pd.read_parquet(OUT / "hand_crosses.parquet")
    D = _frame()[KEY + ["finish_position", "year", "date"]]
    O = pd.read_parquet(OOF)
    O["race_id"], O["horse_id"] = O["race_id"].astype(str), O["horse_id"].astype(str)
    D = D[D["year"].isin(DISC + CONF)].merge(H, on=KEY, how="left").merge(O[KEY + ["p_win", "p_top2", "p_top3"]], on=KEY, how="inner")
    D = D.sort_values(["date", "race_id"], kind="stable").reset_index(drop=True)
    year = D["year"].to_numpy()
    periods = {"disc": np.isin(year, DISC), "conf": np.isin(year, CONF)}
    fp = pd.to_numeric(D["finish_position"], errors="coerce").to_numpy()
    races = D["race_id"].to_numpy()
    tests = {
        "win": ScoreTest((fp == 1).astype(float), D["p_win"].to_numpy(), races, periods, "win"),
        "top2": ScoreTest((fp <= 2).astype(float), D["p_top2"].to_numpy(), races, periods, "binary"),
        "top3": ScoreTest((fp <= 3).astype(float), D["p_top3"].to_numpy(), races, periods, "binary"),
    }
    rows = []
    for name, _, _ in CROSSES:
        for tgt in ("win", "top3"):
            for kind in ("m", "dev"):
                c = f"x58_{name}_{tgt}_{kind}"
                z = D[c].to_numpy(dtype="float64")
                z = np.nan_to_num((z - np.nanmean(z)) / (np.nanstd(z) + 1e-12))
                par = np.nan_to_num((D[f"x58_{name}_{tgt}_parent"].to_numpy(dtype="float64") - D[f"x58_{name}_{tgt}_parent"].mean()) / (D[f"x58_{name}_{tgt}_parent"].std() + 1e-12))
                r = {"cross": name, "stat": f"{tgt}_{kind}"}
                for t, st in tests.items():
                    zz = st.z(z[:, None])
                    ez = st.efficient_z(z, par[:, None])
                    for p in ("disc", "conf"):
                        r[f"z_{t}_{p}"] = round(float(zz[p][0]), 2)
                        r[f"ez_{t}_{p}"] = round(float(ez[p]), 2)
                rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "hand_score.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
    print(T.to_string(index=False))
    (OUT / "hand_score_n.json").write_text(json.dumps({"n_tests": int(len(T) * 3)}), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "score"])
    a = ap.parse_args()
    {"build": cmd_build, "score": cmd_score}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
