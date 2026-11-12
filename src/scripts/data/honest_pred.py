"""2016〜2025 年の全レースについて、その年を学習に使っていない予測（親モデル v6e 相当）の統合表を作り、舞台別の精度を集計する。

    python -m src.scripts.data.honest_pred combine   # 2016〜2021 の LOYO OOF にアンサンブルの重みを当て、2022〜25 と統合
    python -m src.scripts.data.honest_pred analyze   # 統合表を舞台別に集計（基準 = 頭数による一様予測）

由来 source: loyo_noS = 検証年だけを外した leave-one-year-out（OOF_LOYO_YEARS、メンバー 5 つ、前段 S3 の 3 列を入力から外した版 = 親モデル v6e
そのものではない近似）／expanding = 2022〜24 の段 9 の OOF アンサンブル（重みは年を外して当てはめ）／test2025 = 2024 年までで学習した
親モデルの test 2025 予測。重みは tp27_v6e/ens/v6.json を再推定せずそのまま使う。
S3 を外す理由: 2016 年始まりの表では 2017〜19 年の S3 に未来の情報が混ざり（勝ち馬の相対順位 0.157〜0.246、通常 0.27）、LOYO の 2017 年の
1 番手的中率が 50〜68% に跳ねた。S3 を外すと 2017 年で約 32% に戻る。2016 年は S3 が無く追い切り等の欠損も多いため「参考」、2017〜19 年は信頼度が低い。
reliability 列: 参考（2016）／低（2017〜19）／中（2020〜21）／高（2022〜25）。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(os.environ.get("REBUILD_BASE") or "/home/jovyan/work/retrain_out/base_2016")
LOYO_OOF = BASE / os.environ.get("HONEST_OOF_DIR", "tp27_v6e_loyo_early_noS") / "oof"
OUT = BASE / "honest_pred"
KEY = ["race_id", "horse_id"]
TOT = {"win": 1.0, "top2": 2.0, "top3": 3.0}
EPS = 1e-15


def _ll_vec(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _read(p: Path) -> pd.DataFrame:
    d = pd.read_parquet(p)
    d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    return d


def combine() -> None:
    from src.pipeline.models import ensemble as en
    from src.pipeline.models import win_models as wm

    spec = json.loads((BASE / "tp27_v6e/ens/v6.json").read_text(encoding="utf-8"))["spec"]
    tt = _read(BASE / "features/train_table.parquet")[KEY + ["date", "finish_position"]].drop_duplicates(KEY)
    tt["year"] = tt["race_id"].str[:4].astype(int)
    early = tt[tt["year"] <= 2021]

    parts = []
    for t in TOT:
        names = [n for n, w in spec[t]["weights"].items() if w > 0]
        weights = [spec[t]["weights"][n] for n in names]
        base = early[KEY].copy()
        for n in names:
            o = _read(LOYO_OOF / f"{n}.parquet")[KEY + [f"p_{t}"]].rename(columns={f"p_{t}": n})
            base = base.merge(o, on=KEY, how="left")
        if base[names].isna().any().any():
            raise SystemExit(f"{t}: 2016〜2021 の OOF に欠けがあります: {base[names].isna().sum().to_dict()}")
        P = base[names].to_numpy(dtype="float64")
        raw = en.apply_spec({"method": spec[t]["method"], "weights": weights}, P)
        base[f"p_{t}"] = wm.normalize_to_total(base["race_id"].to_numpy(), raw, TOT[t])
        parts.append(base[KEY + [f"p_{t}"]].set_index(KEY))
    e16 = pd.concat(parts, axis=1).reset_index()
    e16["source"] = "loyo_noS"

    e22 = _read(BASE / "tp27_v6e/ens/v6.parquet")[KEY + ["p_win", "p_top2", "p_top3"]]
    e22["source"] = "expanding"
    t25 = _read(BASE / "final_eval_v6e16/test2025_predictions.parquet")[KEY + ["p_win", "p_top2", "p_top3"]]
    t25["source"] = "test2025"

    allp = pd.concat([e16, e22, t25], ignore_index=True)
    out = tt.merge(allp, on=KEY, how="left")
    if len(out) != len(tt) or out["p_win"].isna().any() or out.duplicated(KEY).any():
        raise SystemExit(f"統合に失敗: rows={len(out)} train_table={len(tt)} na={int(out['p_win'].isna().sum())}")
    out["reliability"] = pd.cut(out["year"], [2015, 2016, 2019, 2021, 2025], labels=["参考", "低", "中", "高"]).astype(str)
    out = out.sort_values(["date", "race_id", "horse_id"], kind="stable").reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "honest_pred_2016_2025.parquet", index=False)

    fin = out["finish_position"].to_numpy()
    Y = {"win": (fin == 1).astype(float), "top2": (fin <= 2).astype(float), "top3": (fin <= 3).astype(float)}
    rows = []
    for y, g in out.groupby("year"):
        ix = g.index.to_numpy()
        rows.append({"year": y, "source": g["source"].iloc[0], "reliability": g["reliability"].iloc[0], "rows": len(g), "races": g["race_id"].nunique(),
                     **{f"ll_{t}": float(_ll_vec(g[f"p_{t}"].to_numpy(), Y[t][ix]).mean()) for t in TOT}})
    yr = pd.DataFrame(rows)
    yr.to_csv(OUT / "year_logloss.csv", index=False)
    (OUT / "NOTES.txt").write_text(__doc__ + "\n", encoding="utf-8")
    print(yr.to_string(index=False))
    print(f"rows={len(out)} (train_table と一致) → {OUT / 'honest_pred_2016_2025.parquet'}")


def _bands(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_numeric(df["distance"], errors="coerce")
    df["dist_band"] = pd.cut(d, [0, 1200, 1600, 2000, 2400, 9999], labels=["~1200", "1300-1600", "1700-2000", "2100-2400", "2500~"]).astype(str)
    df["surface_dist"] = df["surface"].astype(str) + " " + df["dist_band"]
    df["field_band"] = pd.cut(df["field_size"], [0, 8, 12, 15, 99], labels=["~8頭", "9-12頭", "13-15頭", "16頭~"]).astype(str)
    df["month"] = pd.to_datetime(df["date"]).dt.month.astype(str).str.zfill(2)
    df["venue_surface"] = df["venue"].astype(str) + " " + df["surface"].astype(str)
    df["era"] = pd.cut(df["year"], [2015, 2016, 2019, 2021, 2024, 2025], labels=["2016(参考)", "2017-19(低)", "2020-21(中)", "2022-24", "2025"]).astype(str)
    return df


DIMS = ["venue", "surface", "dist_band", "surface_dist", "venue_surface", "race_class", "grade", "field_band", "track_condition", "month", "era", "year"]


def analyze(min_races: int = 150, boots: int = 400, include_2016: bool = False) -> None:
    pred = pd.read_parquet(OUT / "honest_pred_2016_2025.parquet")
    attr = _read(BASE / "features/train_table.parquet")[KEY + ["venue", "surface", "distance", "grade", "race_class", "track_condition", "field_size"]].drop_duplicates(KEY)
    df = _bands(pred.merge(attr, on=KEY, how="left"))
    df = df.dropna(subset=["field_size"])
    if not include_2016:
        df = df[df["year"] != 2016]       # 2016 年は参考（S3 なし・欠損多）。舞台の強弱の順位づけには入れない
    df = df.reset_index(drop=True)
    fin = df["finish_position"].to_numpy()
    n = df["field_size"].to_numpy(dtype="float64")
    yw, y3 = (fin == 1).astype(float), (fin <= 3).astype(float)
    df["ll_win"] = _ll_vec(df["p_win"].to_numpy(), yw)
    df["ll_win_base"] = _ll_vec(1.0 / n, yw)
    df["ll_top3"] = _ll_vec(df["p_top3"].to_numpy(), y3)
    df["ll_top3_base"] = _ll_vec(np.minimum(1.0, 3.0 / n), y3)
    # 1 番手予想（p_win 最大の馬）が勝つか。基準の期待値は 1/頭数
    top = df.groupby("race_id")["p_win"].transform("max") == df["p_win"]
    df["pick"] = (top & (yw == 1)).astype(float)
    df["pick_flag"] = top.astype(float)

    race = df.groupby("race_id").agg(
        year=("year", "first"), n=("field_size", "first"),
        w=("ll_win", "sum"), wb=("ll_win_base", "sum"), t=("ll_top3", "sum"), tb=("ll_top3_base", "sum"),
        h=("pick", "sum"), cnt=("p_win", "size"),
        **{d: (d, "first") for d in DIMS if d != "year"},
    )
    race["hb"] = 1.0 / race["n"]
    rng = np.random.default_rng(0)
    out = []
    for d in DIMS:
        for val, g in race.groupby(d, observed=True):
            r = len(g)
            nh = g["cnt"].sum()
            rec = {"dim": d, "value": val, "races": r, "horses": int(nh),
                   "ll_win": g["w"].sum() / nh, "ll_win_base": g["wb"].sum() / nh,
                   "ll_top3": g["t"].sum() / nh, "ll_top3_base": g["tb"].sum() / nh,
                   "top_hit": g["h"].mean(), "top_hit_base": g["hb"].mean()}
            rec["gain_win"] = rec["ll_win_base"] - rec["ll_win"]          # 正 = 基準より良い（対数損失の減少量）
            rec["gain_win_rel"] = rec["gain_win"] / rec["ll_win_base"]
            rec["gain_top3"] = rec["ll_top3_base"] - rec["ll_top3"]
            rec["gain_top3_rel"] = rec["gain_top3"] / rec["ll_top3_base"]
            rec["hit_lift"] = rec["top_hit"] - rec["top_hit_base"]
            if r >= 30:
                w, wb, t, tb, c = (g[k].to_numpy() for k in ("w", "wb", "t", "tb", "cnt"))
                h, hb = g["h"].to_numpy(), g["hb"].to_numpy()
                idx = rng.integers(0, r, size=(boots, r))
                cs = c[idx].sum(axis=1)
                gw = ((wb[idx].sum(axis=1) - w[idx].sum(axis=1)) / cs) / (wb[idx].sum(axis=1) / cs)
                g3 = ((tb[idx].sum(axis=1) - t[idx].sum(axis=1)) / cs) / (tb[idx].sum(axis=1) / cs)
                hl = h[idx].mean(axis=1) - hb[idx].mean(axis=1)
                for k, a in (("gain_win_rel", gw), ("gain_top3_rel", g3), ("hit_lift", hl)):
                    rec[f"{k}_lo"], rec[f"{k}_hi"] = (float(x) for x in np.percentile(a, [2.5, 97.5]))
            rec["small"] = r < min_races
            out.append(rec)
    res = pd.DataFrame(out)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / ("stage_accuracy_incl2016.csv" if include_2016 else "stage_accuracy.csv"), index=False)
    tot = race
    print(f"全体: races={len(tot)} 勝率 LL 基準比 {1 - ((tot['w'].sum()) / tot['wb'].sum()):.4f} / 複勝 LL 基準比 {1 - tot['t'].sum() / tot['tb'].sum():.4f} / 1番手的中 {tot['h'].mean():.4f} (基準 {tot['hb'].mean():.4f})")
    cols = ["dim", "value", "races", "gain_win_rel", "gain_win_rel_lo", "gain_win_rel_hi", "gain_top3_rel", "hit_lift"]
    big = res[~res["small"] & ~res["dim"].isin(["year", "era"])]
    print("== 勝率: 基準比の改善率が高い 上位 15 ==")
    print(big.sort_values("gain_win_rel", ascending=False)[cols].head(15).round(4).to_string(index=False))
    print("== 勝率: 基準比の改善率が低い 下位 15 ==")
    print(big.sort_values("gain_win_rel")[cols].head(15).round(4).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["combine", "analyze"])
    ap.add_argument("--include-2016", action="store_true", help="analyze に 2016 年（参考）も含める")
    a = ap.parse_args()
    combine() if a.cmd == "combine" else analyze(include_2016=a.include_2016)


if __name__ == "__main__":
    main()
