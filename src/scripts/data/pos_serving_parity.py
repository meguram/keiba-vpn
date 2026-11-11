"""位置取り予想の推論（pos_serving）が学習の特徴量表（pos_model の features.parquet）と同じ値を作るかを、過去の日で確かめる。ホストで実行。

    POS_MODEL_OUT=data/local/modeling/pos_model python -m src.scripts.data.pos_serving_parity --days 2025-04-05 2025-10-12 [--model retrain_out/base_2016/final_pos] [--no-weight]

- 各日について、その日より前の走だけの PosState を作り、その日の出走馬（結果表の行を出馬表の代わりに使う）から ``card_features`` を作って、
  学習の表の同じ行と列ごとに比べる（一致しない行の割合・最大差）。
- ``--model`` を渡すと pos_pred も ``pos_oof.parquet`` と比べる（oof は年の開始前の学習なので値は少し違う。差の大きさの目安として出す）。
- ``--no-weight`` は前夜の出馬表に馬体重が無い場合の劣化を見る（wt・wt_chg を欠損にして pos_pred を学習表の入力のまま作った値と比べる）。
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.pipeline.inference import pos_serving as ps
from src.scripts.data import pos_model as pm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="*", default=[])
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-weight", action="store_true")
    ap.add_argument("--fill-weight", action="store_true", help="馬体重の欠損を前走の体重で埋める（--no-weight と併用）")
    ap.add_argument("--no-going", action="store_true", help="馬場を欠損にして出す（前夜の出馬表を想定）。--fill-going で良に補完")
    ap.add_argument("--fill-going", action="store_true")
    ap.add_argument("--no-oikiri", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--write-consts", default=None, help="推論用の定数（pos_consts.json）をこのディレクトリ（final_pos）に書く")
    a = ap.parse_args()

    ref = pd.read_parquet(pm.OUT / "features.parquet")
    consts = ps.make_consts(ref)
    if a.write_consts:
        from pathlib import Path

        (Path(a.write_consts) / ps.CONSTS_FILE).write_text(json.dumps(consts), encoding="utf-8")
        print(f"定数を書いた: {a.write_consts}/{ps.CONSTS_FILE}")
    runs = pm.load_runs()
    runs["f3_dev"] = pm.race_front_pace(runs)
    runs["j_pf"] = pm.asof_mean(runs, "jockey_id", k=30.0, m=consts["m_pf"])
    oik = pd.concat([pd.read_parquet(pm.TABLES / str(y) / "race_oikiri_flat.parquet", columns=["race_id", "horse_id", "date", "course", "rider", "lap_times", "evaluation", "training_date"])
                     for y in pm.YEARS if (pm.TABLES / str(y) / "race_oikiri_flat.parquet").exists()], ignore_index=True)
    oik["race_id"] = oik["race_id"].astype(str)
    model = ps.PosModel.load(a.model) if a.model else None
    oof = pd.read_parquet(pm.OUT / "pos_oof.parquet") if model is not None and (pm.OUT / "pos_oof.parquet").exists() else None
    ref["race_id"], ref["horse_id"] = ref["race_id"].astype(str), ref["horse_id"].astype(str)
    report: dict = {}
    for day in a.days:
        t = pd.Timestamp(day)
        card = runs[runs["date"] == t].copy()
        if card.empty:
            print(f"{day}: 開催なし")
            continue
        if a.no_weight:
            card["weight"], card["weight_change"] = np.nan, np.nan
        if a.no_going:
            card["track_condition"] = np.nan
        state = ps.PosState.from_runs(runs, consts, before=t)
        o = None if a.no_oikiri else oik[oik["race_id"].isin(set(card["race_id"]))]
        got = ps.card_features(card, state, o, blank_pedigree=False, fill_weight=a.fill_weight, fill_going=a.fill_going or not a.no_going)
        want = got[["race_id", "horse_id"]].merge(ref[["race_id", "horse_id"] + pm.FEATURES], on=["race_id", "horse_id"], how="left")
        cols = {}
        for c in pm.FEATURES:
            x, y = got[c], want[c]
            if c in ps.CAT_FEATURES:
                bad = (x.astype(str) != y.astype(str)).mean()
                cols[c] = {"bad": float(bad)}
                continue
            x, y = x.astype(float).to_numpy(), y.astype(float).to_numpy()
            both = np.isfinite(x) & np.isfinite(y)
            nan_mis = float((np.isnan(x) != np.isnan(y)).mean())
            diff = np.abs(x - y)[both]
            cols[c] = {"nan_mismatch": nan_mis, "bad": float((diff > 1e-6).mean()) if len(diff) else 0.0, "max_diff": float(diff.max()) if len(diff) else 0.0}
        report[day] = {"rows": len(got), "cols": cols}
        worst = sorted(cols.items(), key=lambda kv: -(kv[1].get("bad", 0) + kv[1].get("nan_mismatch", 0)))[:8]
        print(f"{day}: {len(got)} 行。ずれの大きい列 → " + ", ".join(f"{k}(bad={v.get('bad', 0):.3f}, nan={v.get('nan_mismatch', 0):.3f}, max={v.get('max_diff', 0):.3g})" for k, v in worst))
        if model is not None:
            pred = ps.pos_pred_for_card(card, state, model, o, blank_pedigree=False, fill_weight=a.fill_weight, fill_going=a.fill_going or not a.no_going)
            report[day]["pos_pred"] = {"mean": float(pred["pos_pred"].mean())}
            if oof is not None:
                oo = oof.assign(race_id=oof["race_id"].astype(str), horse_id=oof["horse_id"].astype(str))
                m = pred.merge(oo[["race_id", "horse_id", "pos_pred"]], on=["race_id", "horse_id"], how="inner", suffixes=("", "_oof"))
                if len(m):
                    d = (m["pos_pred"] - m["pos_pred_oof"]).abs()
                    report[day]["pos_pred"].update({"vs_oof_mean_abs": float(d.mean()), "vs_oof_corr": float(m["pos_pred"].corr(m["pos_pred_oof"])), "n": len(m)})
                    print(f"  pos_pred: oof との平均絶対差 {d.mean():.4f}、相関 {m['pos_pred'].corr(m['pos_pred_oof']):.4f}")
            if not a.no_weight and not a.no_going and "pos_first" in card.columns:
                mm = pred.merge(card[["race_id", "horse_id", "pos_first"]], on=["race_id", "horse_id"])
                mm = mm.dropna(subset=["pos_first"])
                report[day]["pos_pred"]["mae_vs_actual"] = float((mm["pos_pred"] - mm["pos_first"]).abs().mean())
                print(f"  実際の初角位置との MAE {report[day]['pos_pred']['mae_vs_actual']:.4f}")
            if a.no_weight or a.no_going:
                mm = pred.merge(card[["race_id", "horse_id", "pos_first"]], on=["race_id", "horse_id"]).dropna(subset=["pos_first"])
                report[day]["pos_pred"]["mae_vs_actual_degraded"] = float((mm["pos_pred"] - mm["pos_first"]).abs().mean())
                print(f"  （劣化条件）実際の初角位置との MAE {report[day]['pos_pred']['mae_vs_actual_degraded']:.4f}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
