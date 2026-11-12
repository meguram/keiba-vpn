"""v6 の推論経路（V3Serving.features）が作る列を、学習側の行列（fit_final_v6 と同じ build_matrices）と同じ行で突き合わせる（監査用・読み取りのみ）。

（再学習の親モデル=2016〜2025 年版は V6_STAGE を付けず --rebuild-2016 を足す。bundle は final_v6e16/bundle_v6_2025、stage は final_s3t7、rb-state は base_2016/rb_state。
 前段の入力は KEIBA_TRACKING_V2_DIR / KEIBA_LAP_FORECAST_DIR に、tracking_t6_backfill と tp55_backfill で作った保存値を向ける。めぐ指数は KEIBA_MEGU_V3A_DIR=retrain_out/megu_v3a_ref2016-2024）
（日単位のキャッシュ state/<日> は KEIBA_INFERENCE_DIR の下に残る。2016 年版は専用の場所を指定し、入力を変えたら 2025-* を消す。--rebuild-2016 は prepare() で土台の中に書き込む）
KEIBA_INFERENCE_ENGINE=full V6_STAGE=e V6_TAG=e V5_GROUPS=CY,BEST,RB python -m src.scripts.data.v6_parity --days 20250202,20250614 --max-races 6 --out <json> \
    --bundle <final_v6e/bundle_v6_2024> --stage <final_stage/v6_e_2024> --rb-state <rb_state>
列ごとに、欠損の位置の不一致率と、両方に値がある行での不一致率（|差| > 1e-3×(1+|学習側|)）を出す。前段 e の入力（追走難度 v2・ラップ予想・位置取り予想）は
2025 年の保存値が無いため、その由来の列（s_*, EASE2/lp/pf, それらの相対化）は比べても意味がなく別枠で出す。
"""

from __future__ import annotations

import os

if os.environ.get("V6_STAGE"):
    from src.scripts.data import v6_work

    v6_work.use(os.environ["V6_STAGE"])

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

KEY = ["race_id", "horse_id"]


GROUPS = {"ped_24": ("ped_",), "sire_prior": ("sire_", "dam_sire_"), "mtl": ("mtl_",), "stage": ("s_time_", "s_rank_"), "rb_mg": ("rb_mg_",),
          "day_agg": ("fe_", "sr_var_", "f1_", "chg_", "exp_", "fd_clsstep"), "o_ob": ("o_", "ob_")}


def _predict_compare(v3, m: pd.DataFrame, S: pd.DataFrame) -> dict:
    """推論側の列で出した確率と、学習側の列で出した確率の差。群ごとに、その群だけ推論側の値・残りは学習側の値にして、どの群がどれだけ確率を動かすかも出す。"""
    cols = [c for c in v3.feature_cols if f"{c}_s" in m.columns and f"{c}_t" in m.columns]
    base = m[["race_id", "horse_id"]].copy()

    def prob(use_t: set[str]) -> pd.Series:
        X = base.copy()
        for c in v3.feature_cols:
            if c in cols:
                X[c] = m[f"{c}_t" if c in use_t else f"{c}_s"].to_numpy()
            else:  # 片側にしか無い列は、そのまま（接尾辞が付かない）
                X[c] = m[c].to_numpy() if c in m.columns else np.nan
        return v3.predict(X).set_index(["race_id", "horse_id"])["p_win"]

    def stat(a: pd.Series, b: pd.Series) -> dict:
        la, lb = np.log(a.clip(1e-6)), np.log(b.clip(1e-6))
        top_a, top_b = a.groupby(level=0).idxmax(), b.groupby(level=0).idxmax()
        return {"mean_abs_dlogp": float((la - lb).abs().mean()), "p95_abs_dlogp": float((la - lb).abs().quantile(0.95)), "mean_abs_dp": float((a - b).abs().mean()),
                "top1_same": float((top_a == top_b).mean()), "corr": float(np.corrcoef(a, b.reindex(a.index))[0, 1])}

    p_s, p_t = prob(set()), prob(set(cols))
    out = {"n_cols": len(cols), "serving_vs_train": stat(p_s, p_t), "only_group_from_serving": {}}
    for g, pre in GROUPS.items():
        use = {c for c in cols if c.startswith(pre)}
        if use:
            out["only_group_from_serving"][g] = {"n": len(use), **stat(prob(set(cols) - use), p_t)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", required=True)
    ap.add_argument("--max-races", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--rb-state", required=True)
    ap.add_argument("--rebuild-2016", action="store_true", help="2016〜2025 年の土台（再学習の親モデル）と突き合わせる。V6_STAGE は付けない")
    a = ap.parse_args()
    if a.rebuild_2016:
        os.environ.setdefault("KEIBA_FIRST_YEAR", "2016")  # 推論の履歴の起点年（既定 2020）。2016 年の土台と合わせる
        from src.scripts.data import parent_final_2016

        parent_final_2016.setup_env()
        from src.scripts.data import rebuild_groups_2016

        rebuild_groups_2016.redirect_serving()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    from src.pipeline.inference import weight_triggered_predict as wtp
    from src.pipeline.inference.v3_serving import V3Serving
    from src.scraper.storage import HybridStorage
    from src.scripts.data import fit_final_v4 as f4
    from src.scripts.data import fit_final_v6 as f6
    from src.scripts.data.backfill_predictions import race_days

    f6.patch()
    df, meta, mats, A, _ = f4.build_matrices()
    names = [m[0] for m in f4.MEMBERS]
    X_all = pd.concat([mats[n][2] for n in names], axis=1)
    X_all = X_all.loc[:, ~X_all.columns.duplicated()]
    X_all.insert(0, "horse_id", meta["horse_id"].to_numpy())
    X_all.insert(0, "race_id", meta["race_id"].astype(str).to_numpy())

    tables = Path(os.environ.get("KEIBA_TABLES_DIR") or "data/page_reference/tables")
    storage = HybridStorage()
    days_req = [d.strip() for d in a.days.split(",") if d.strip()]
    days = race_days([2025], tables, min(days_req), max(days_req))
    days = {d: ids for d, ids in days.items() if d in days_req}
    layer_a = None
    if a.rebuild_2016:
        from src.scripts.data import parent_final_2016

        layer_a = parent_final_2016.BASE / "layer_a.parquet"
    ctx = wtp.InferenceContext(layer_a=layer_a)
    ctx.warm()
    raw_full = ctx.raw_hist
    v3 = V3Serving(Path(a.bundle), Path(a.stage), tl_reference=os.environ.get("KEIBA_V5_TL_REF") or None)
    v3.rb_state = Path(a.rb_state)
    v3.warm()
    hist_full = v3.history.copy()
    train_keys = pl.DataFrame({"race_id": X_all["race_id"].astype(str).to_numpy(), "horse_id": X_all["horse_id"].astype(str).to_numpy()})
    parts = []
    for day, ids in days.items():
        all_ids = list(ids)
        ids = ids[: a.max_races]
        d = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        card = wtp.fill_card_meta(wtp.card_frame(storage, ids), [])
        ok_ids = (card.filter(pl.col("horse_number").fill_null(0) > 0).filter(pl.col("distance").cast(pl.Float64, strict=False) > 0)
                  .filter(pl.col("surface").cast(pl.Utf8).fill_null("") != "")["race_id"].unique().to_list())
        card = card.filter(pl.col("race_id").is_in(ok_ids))
        card = card.with_columns(pl.col("race_id").cast(pl.Utf8), pl.col("horse_id").cast(pl.Utf8)).join(train_keys, on=["race_id", "horse_id"], how="semi")  # 取消・除外馬は学習表に無い。頭数で決まる列がずれるので学習側の出走馬に揃える
        ok_ids = card["race_id"].unique().to_list()
        v3._card_day[d] = wtp.fill_card_meta(wtp.card_frame(storage, all_ids), []).to_pandas() if len(all_ids) > len(ids) else card.to_pandas()  # 本番と同じく、その日の全レースの出馬表を渡す（同日・同場の集計に要る。--max-races で絞っても全レース）
        cut =datetime.date.fromisoformat(d) if raw_full.schema["date"] == pl.Date else d
        ctx.raw_hist = raw_full.filter(pl.col("date") < cut)
        F, _ = ctx.build(card)
        for rid in sorted(ok_ids):
            cr = card.filter(pl.col("race_id") == rid)
            st = str(cr["start_time"][0])
            h = hist_full[(hist_full["date"].astype(str) < d) | ((hist_full["date"].astype(str) == d) & (hist_full["start_time"].astype(str) < st) & (hist_full["race_id"].astype(str) != rid))]
            X = v3.features(cr, F.filter(pl.col("race_id") == rid), storage, history=h)
            parts.append(X)
    S = pd.concat(parts, ignore_index=True)
    S["race_id"], S["horse_id"] = S["race_id"].astype(str), S["horse_id"].astype(str)
    X_all["horse_id"] = X_all["horse_id"].astype(str)
    m = S.merge(X_all, on=KEY, suffixes=("_s", "_t"))
    stage_like = lambda c: c.startswith(("s_", "EASE2_", "lp_", "pf_", "z_hat", "td_")) or "_s_" in c or c.split("_r")[0].startswith("s_")  # noqa: E731
    rows = []
    for c in [x for x in S.columns if x not in KEY]:
        if f"{c}_t" not in m.columns:
            rows.append({"col": c, "note": "学習側に無い"})
            continue
        s, t = pd.to_numeric(m[f"{c}_s"], errors="coerce").to_numpy(float), pd.to_numeric(m[f"{c}_t"], errors="coerce").to_numpy(float)
        both = ~np.isnan(s) & ~np.isnan(t)
        rows.append({"col": c, "n": int(len(m)), "nan_mismatch": float((np.isnan(s) != np.isnan(t)).mean()),
                     "val_mismatch": float((np.abs(s - t)[both] > 1e-3 * (1 + np.abs(t[both]))).mean()) if both.any() else 0.0,
                     "corr": float(np.corrcoef(s[both], t[both])[0, 1]) if both.sum() > 5 and np.std(s[both]) > 0 and np.std(t[both]) > 0 else None,
                     "mad_over_sd": float(np.mean(np.abs(s - t)[both]) / (np.std(t[both]) + 1e-12)) if both.any() else 0.0,
                     "stage_like": bool(stage_like(c))})
    rep = pd.DataFrame(rows)
    pred_cmp = _predict_compare(v3, m, S)
    big = rep[(rep["mad_over_sd"] > 0.2) | (rep["nan_mismatch"] > 0.05)]
    out = {"n_big_diff": int(len(big)), "big_cols": big["col"].tolist(), "rows": int(len(m)), "races": int(m["race_id"].nunique()), "columns": int(len(rep)),
           "worst": rep.sort_values(["mad_over_sd", "nan_mismatch"], ascending=False).head(80).to_dict("records"),
           "all": {r["col"]: {k: r.get(k) for k in ("nan_mismatch", "val_mismatch", "corr", "mad_over_sd")} for r in rows if "n" in r},
           "n_bad_nan_gt5pct": int((rep["nan_mismatch"] > 0.05).sum()), "n_bad_val_gt5pct": int((rep["val_mismatch"] > 0.05).sum()),
           "missing_in_train": rep[rep.get("note").notna()]["col"].tolist() if "note" in rep else [], "predict_compare": pred_cmp}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "worst"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
