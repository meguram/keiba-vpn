"""tp-55: ラップ予想（シナリオ確率・期待ラップ）を勝率予測の特徴量にしたときの増分（V5 の列セットへの累積）。ホストで実行。

python -m src.scripts.data.tp55_win_eval make-group     # 年ごとの leave-one-year-out のラップ予想から LAP 群（race_id, horse_id）を作る
python -m src.scripts.data.tp55_win_eval run            # v5_select_eval と同じ手順（valid 2022〜24 expanding、2 seed）で d_CY_BEST + LAP の OOF を作る
python -m src.scripts.data.tp55_win_eval compare        # d_CY_BEST との対の比較（採否の基準は既定）

LAP 群の列: レース単位（lp_p_slow/mid/high, lp_exp_f3_dev, lp_exp_l3_dev, lp_exp_lap0_dev）＋馬単位の交互作用（lp_front_x_high など。馬の脚質×ペース）。
ラップモデルは各年を『その年を含まない年』で学習した予測（OOF）。勝ち馬・着順は使わない（ラップだけが目的変数）。市場オッズ・人気は使わない。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.models import lap_forecast as lfm

OUT = Path(os.environ.get("TP55_OUT", "/home/jovyan/work/tp55_out3"))
KEY = ["race_id", "horse_id"]
THREADS = int(os.environ.get("TP55_THREADS", "3"))
GROUP_PATH = OUT / "groups" / "LAP.parquet"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_group() -> None:
    d = pd.read_parquet(OUT / "races_feat.parquet")
    d["year"] = d["date"].str[:4].astype(int)
    d = d[d["year"] <= 2025].reset_index(drop=True)
    parts = []
    for y in range(2020, 2026):
        tr, te = d[d["year"] != y], d[d["year"] == y]
        m = lfm.LapForecast(scenario="A", threads=THREADS, n_oof=2).fit(tr)
        out = m.predict(te)
        K = m.K
        mu = out["mu_dev"]
        mix = np.nansum(out["proba"][:, :, None] * mu, axis=1) / np.maximum(np.sum(out["proba"][:, :, None] * ~np.isnan(mu), axis=1), 1e-9)
        n_laps = out["n_laps"]
        f3 = np.nansum(mix[:, :3], axis=1)
        l3 = np.array([np.nansum(mix[i, max(n_laps[i] - 3, 0):n_laps[i]]) if n_laps[i] <= lfm.MAXJ else np.nan for i in range(len(te))])
        parts.append(pd.DataFrame({"race_id": te["race_id"].values, "lp_p_slow": out["proba"][:, 0], "lp_p_mid": out["proba"][:, 1], "lp_p_high": out["proba"][:, 2],
                                   "lp_exp_f3_dev": f3, "lp_exp_l3_dev": l3, "lp_exp_lap0_dev": mix[:, 0]}))
        _log(f"{y}: {len(te)} races")
    R = pd.concat(parts, ignore_index=True)
    # 馬単位: 脚質（過去走の最初のコーナーの正規化位置 h_early_w）×ペース。ここで使うのは as-of の馬特徴
    from src.scripts.data import tp55_build as tb

    res, shu, smr = tb.load_inputs(list(range(2020, 2026)), None)
    ent = shu[["race_id", "horse_id", "horse_number"]].drop_duplicates(KEY).copy()
    ent = ent.merge(d[["race_id", "date"]], on="race_id", how="inner")
    ent["date"] = pd.to_datetime(ent["date"])
    res = res[res["surface"].isin(["芝", "ダート"])].copy()
    from src.pipeline.features import tp55_lapfeat as lf

    res["early"] = [lf.first_corner_norm(p, f) for p, f in zip(res["passing_order"], res["field_size"])]
    hf = lf.asof_horse_features(ent, res[["race_id", "horse_id", "date", "early"]].assign(date=lambda x: pd.to_datetime(x["date"])))
    H = ent[["race_id", "horse_id"]].reset_index(drop=True).merge(hf[["race_id", "horse_id", "h_early_w"]], on=KEY, how="left").merge(R, on="race_id", how="inner")
    e = H["h_early_w"]
    H["lp_front_x_high"] = (1 - e) * H["lp_p_high"]             # 前に行く馬ほど、ハイペースの確率が高いと負荷が大きい
    H["lp_front_x_slow"] = (1 - e) * H["lp_p_slow"]             # 前に行く馬ほど、スローの確率が高いと楽
    H["lp_closer_x_slow"] = e * H["lp_p_slow"]                  # 後ろの馬ほど、スローの確率が高いと不利
    H["lp_pos_x_f3"] = (1 - e) * H["lp_exp_f3_dev"]
    H = H.drop(columns=["h_early_w"])
    GROUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    H.to_parquet(GROUP_PATH, index=False)
    _log(f"LAP 群: {H.shape}, 欠損 {H.isna().mean().round(3).to_dict()}")


def _patch_harness():
    from src.scripts.data import v5_select_eval as v5

    v5.CONFIGS["g_CY_BEST_LAP"] = {"base": "fresh", "groups": ["CY", "BEST", "LAP"]}
    orig = v5.load_group

    def load_group(name, df):
        if name == "LAP":
            return v5._merge_keys(df, pd.read_parquet(GROUP_PATH))
        return orig(name, df)

    v5.load_group = load_group
    return v5


def cmd_run() -> None:
    v5 = _patch_harness()

    class A:
        name = "g_CY_BEST_LAP"

    v5.cmd_run(A())


def cmd_compare() -> None:
    v5 = _patch_harness()
    from src.scripts.data import tp27_eval as t
    from src.scripts.data import tp40_eval as e40

    df, base, meta, folds, lp = t.setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    res = {}
    for old in ("d_CY_BEST", "b_tlfresh"):
        a0 = pd.read_parquet(v5.OUT / f"oof_{old}.parquet")
        b0 = pd.read_parquet(v5.OUT / "oof_g_CY_BEST_LAP.parquet")
        c = t._cmp(a0, b0, fin_map, f"g_CY_BEST_LAP vs {old}")
        by_year = {}
        for y in v5.YEARS:
            ids = set(meta.loc[meta["year"] == y, "race_id"].astype(str))
            ma, mb = a0["race_id"].astype(str).isin(ids), b0["race_id"].astype(str).isin(ids)
            cy = t._cmp(a0[ma], b0[mb], fin_map, f"{old} {y}")
            by_year[str(y)] = {"Δwin": cy["Δwin"], "win_hi": cy["win_hi"], "Δtop2": cy["Δtop2"], "Δtop3": cy["Δtop3"]}
        res[f"g_CY_BEST_LAP vs {old}"] = {"overall": c, "by_year": by_year}
        _log(f"vs {old}: Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f} 採用={c['採用']}")
    (OUT / "win_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make-group", "run", "compare"])
    a = ap.parse_args()
    {"make-group": make_group, "run": cmd_run, "compare": cmd_compare}[a.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
