"""追走難度 v2（最終モデル試行 6）を指定日の全レースについて作って保存する。ホストで実行（結果表の全期間を読むため重い。heavy_run.sh 経由・直列）。

  PYTHONPATH=. python -m src.scripts.data.tracking_t6_backfill --dates 2025-04-05 --pos-model retrain_out/base_2016/final_pos \\
      --t6-model retrain_out/base_2016/final_tracking_t6 [--source shutuba|result] [--dry-run]

- 出力: tracking_v2_service.path_for（KEIBA_TRACKING_V2_DIR 配下の <日>/<race_id>.json）。ラップ予想は KEIBA_LAP_FORECAST_DIR の保存値を読む（無いレースは欠損）。
- --source result は、出馬表の代わりに結果表の行を使う（過去日の突き合わせ用）。shutuba は race_shutuba_flat（前夜の運用に近い）。
- 保存先は KEIBA_TRACKING_V2_DIR を必ず指定する（未設定だとミラー対象の木に旧エンジンの保存を上書きしてしまうため、保存は拒否する）。
  実行時は KEIBA_MIRROR_PULL を外す（ラップ予想の読み出しで GCP の外から GCS を 1 件ずつ読まないため）。
- 学習側の母集団に合わせ、位置取り予想の無い馬と 8 頭立て未満のレースは保存しない（t6.make_docs）。
- --parity は PYTHONPATH=. TP54_OUT=<学習表の場所（retrain_out/tp54_2016）> を export して実行する（tp54_model の読み込み時に決まる）。学習側の特徴量（fit_final_tracking_t6 と同じ式）と日ごとに列で突き合わせる。位置取り・ラップ・想定ペースを学習側の保存値に揃えて、
  追走難度の組み立て（過去走の集計・終いの落ち・掛け合わせ）だけの一致を見る。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.inference import lap_forecast_service as lsvc
from src.pipeline.inference import pos_serving as ps
from src.pipeline.inference import tracking_t6 as t6
from src.pipeline.inference import tracking_v2_service as tsvc
from src.pipeline.features import serving_v6 as sv6
from src.scripts.data import pos_model as pm

TABLES = pm.TABLES
YEARS = list(range(2016, 2027))


def load_results(years: list[int]) -> pd.DataFrame:
    fs = [TABLES / str(y) / "race_result_flat.parquet" for y in years]
    return pd.concat([pd.read_parquet(f, columns=t6.RESULT_COLS) for f in fs if f.exists()], ignore_index=True)


def day_entrants(day: pd.Timestamp, source: str, runs: pd.DataFrame) -> pd.DataFrame:
    if source == "result":
        return runs[runs["date"] == day].copy()
    f = TABLES / str(day.year) / "race_shutuba_flat.parquet"
    d = pd.read_parquet(f)
    d["date"] = pd.to_datetime(d["date"].astype(str).str.replace(r"[./]", "-", regex=True), errors="coerce")
    d = d[d["date"] == day].copy()
    miss = [c for c in ps.REQUIRED_CARD if c not in d.columns]
    if miss:
        raise KeyError(f"出馬表に無い列: {miss}")
    return d


def race_cols_for(card: pd.DataFrame, day: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for rid in card["race_id"].astype(str).unique():
        j = lsvc.load_forecast(rid, day)
        if not j:
            continue
        out[rid] = {**sv6.lap_race_columns(j), **sv6.lapd_race_columns(j)}
    return out


def parity(model: t6.T6Model, days: list[str], hist_a: pd.DataFrame, hist_b: pd.DataFrame) -> None:
    """学習側の XC（drain の入力）と、day_features の XC を列ごとに比べる。"""
    from src.pipeline.models import tracking_difficulty_v2 as td2
    from src.scripts.data import fit_final_tracking_t6 as ft
    from src.scripts.data import tp54_model as tm

    h = tm.load()
    h["race_id"], h["horse_id"] = h["race_id"].astype(str), h["horse_id"].astype(str)
    X = td2.make_features(h)
    L, D, P, pe = ft.build_inputs(h)
    _, y = tm.make_target(h, np.asarray(model.meta["baseline_coef"], float))
    Hi = td2.past_drop_inputs(h, y, pe)
    XC = pd.concat([X[tm.CORE], L, D, P, Hi], axis=1)[model.meta["drain_features"]]
    po = pd.read_parquet(ft.POS_OOF)
    po["race_id"], po["horse_id"] = po["race_id"].astype(str), po["horse_id"].astype(str)
    po = pd.concat([po, ft.pp_columns(po)], axis=1)
    lap = pd.read_parquet(ft.LAP_GROUP).assign(race_id=lambda d: d["race_id"].astype(str)).drop_duplicates("race_id").set_index("race_id")
    ld = pd.read_parquet(ft.LAPD_GROUP).assign(race_id=lambda d: d["race_id"].astype(str)).drop_duplicates("race_id").set_index("race_id")
    rc_all = pd.concat([lap[list(t6.LP_BASE)], ld[list(t6.LD_BASE)]], axis=1)
    dates = pd.to_datetime(h["date"])
    for day in days:
        m = (dates == pd.Timestamp(day)) & h["pf_early"].notna()
        if not m.any():
            print(f"{day}: 学習表に行なし")
            continue
        sub = h[m]
        ent = sub[["race_id", "date", "horse_id", "horse_number", "venue", "surface", "distance", "track_condition", "field_size"]].copy()
        pos = po[po["race_id"].isin(set(ent["race_id"]))][["race_id", "horse_id", "pos_pred", *t6.PP_COLS]]
        rc = rc_all.reindex(ent["race_id"].unique()).to_dict(orient="index")
        z = sub.drop_duplicates("race_id").set_index("race_id")["z_hat"].to_dict()
        e, got, _ = t6.day_features(model, ent, hist_a, hist_b, pos, rc, z_override=z)
        want = XC.loc[sub.index].reset_index(drop=True)
        got = got.reset_index(drop=True)
        rows = []
        for c in want.columns:
            a, b = got[c].astype(float).to_numpy(), want[c].astype(float).to_numpy()
            both = np.isfinite(a) & np.isfinite(b)
            diff = np.abs(a - b)[both]
            rows.append((c, float((np.isnan(a) != np.isnan(b)).mean()), float((diff > 1e-6).mean()) if len(diff) else 0.0, float(diff.max()) if len(diff) else 0.0))
        rows.sort(key=lambda r: -(r[1] + r[2]))
        print(f"{day}: {len(want)} 行。ずれの大きい列 → " + ", ".join(f"{c}(nan={n:.3f}, bad={bd:.3f}, max={mx:.3g})" for c, n, bd, mx in rows[:10]))
        okc = sum(1 for r in rows if r[1] == 0 and r[2] == 0)
        fin = want.notna().mean()
        print(f"  完全一致の列 {okc}/{len(rows)}。値のある割合（追加列）: " + ", ".join(f"{c}={fin[c]:.2f}" for c in [*model.meta["lap_columns"][:1], "lp_front_x_high", "ld_f3", "ld_rear_x_q", "pp_rank", "hy3_x_front", "hy1_x_front", "w_adj"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--pos-model", help="final_pos のディレクトリ（--parity では不要）")
    ap.add_argument("--t6-model", required=True)
    ap.add_argument("--source", choices=["shutuba", "result"], default="shutuba")
    ap.add_argument("--years", type=int, nargs="+", default=YEARS)
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    model = t6.T6Model.load(a.t6_model)
    results = load_results(a.years)
    hist_a, hist_b = t6.history_a(results), t6.history_b(results)
    if a.parity:
        parity(model, a.dates, hist_a, hist_b)
        return
    if not a.dry_run and not os.environ.get(tsvc.ENV_DIR):
        raise SystemExit(f"{tsvc.ENV_DIR} が未設定。保存先を指定してから実行する（既定の木はミラー対象）")
    if os.environ.get("KEIBA_MIRROR_PULL", "").strip() == "1":
        raise SystemExit("KEIBA_MIRROR_PULL を外してから実行する（GCP の外から GCS を読まない）")

    runs = pm.load_runs(None, a.years)
    runs["f3_dev"] = pm.race_front_pace(runs, tables=None, years=a.years)
    consts = ps.load_consts(a.pos_model)
    pmodel = ps.PosModel.load(a.pos_model)
    n_docs = 0
    for day in a.dates:
        t = pd.Timestamp(day)
        card = day_entrants(t, a.source, runs)
        if card.empty:
            print(f"{day}: 開催なし")
            continue
        state = ps.PosState.from_runs(runs, consts, before=t)
        oik_f = TABLES / str(t.year) / "race_oikiri_flat.parquet"
        oik = None
        if oik_f.exists():
            oik = pd.read_parquet(oik_f, columns=["race_id", "horse_id", "date", "course", "rider", "lap_times", "evaluation", "training_date"])
            oik["race_id"] = oik["race_id"].astype(str)
            oik = oik[oik["race_id"].isin(set(card["race_id"].astype(str)))]
        pos = ps.pos_pred_for_card(card, state, pmodel, oik, blank_pedigree=a.source == "shutuba")
        ent = card[["race_id", "date", "horse_id", "horse_number", "venue", "surface", "distance", "track_condition", "field_size"]]
        e, XC, XD = t6.day_features(model, ent, hist_a, hist_b, pos, race_cols_for(card, day))
        docs = t6.make_docs(model, e, t6.predict(model, XC, XD))
        if not a.dry_run:
            for d in docs:
                tsvc._write(tsvc.path_for(d["race_id"], day), d)
        n_docs += len(docs)
        print(json.dumps({"day": day, "races": len(docs), "horses": int(sum(d["n_with_v2"] for d in docs)), "dry_run": a.dry_run}, ensure_ascii=False))
    print(f"保存 {n_docs} レース分")


if __name__ == "__main__":
    main()
