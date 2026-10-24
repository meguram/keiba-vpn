"""ドリフト診断用: 推論経路（V3Serving）で作った特徴量を、月ごとに抜き取った日・レースについて保存する。

  KEIBA_INFERENCE_ENGINE=full python -m src.scripts.data.drift_dump_features --years 2025,2026 --months 1-10 --races-per-day 10 \
      --out data/local/inference/drift_features

目的は『同じ推論経路』で 2025 年と 2026 年の特徴量の分布を比べること（学習表との構成差と、年の移り変わりによる変化を切り分ける）。
同じ行の予測（p_win/p_top2/p_top3）も付ける（学習表経路の予測との差＝推論経路の誤差の見積もり用）。モデルの選択には使わない。書き込み先は新しいディレクトリのみ。出力済みの日はスキップ。
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import polars as pl

logger = logging.getLogger("scripts.drift_dump_features")


def pick_days(tables: Path, year: int, months: list[int]) -> dict[str, list[str]]:
    """月ごとに『レース数が最も多い日』を 1 日選ぶ（開催日 → race_id 一覧）。"""
    from src.scripts.data.backfill_predictions import race_days

    days = race_days([year], tables, f"{year}0101", f"{year}1231")
    by_month: dict[int, list[tuple[str, list[str]]]] = {}
    for d, ids in days.items():
        by_month.setdefault(int(d[4:6]), []).append((d, ids))
    out: dict[str, list[str]] = {}
    for m in months:
        cand = by_month.get(m, [])
        if cand:
            d, ids = max(cand, key=lambda x: (len(x[1]), x[0]))
            out[d] = ids
    return out


def spread(ids: list[str], n: int) -> list[str]:
    if n <= 0 or len(ids) <= n:
        return ids
    step = len(ids) / n
    return [ids[int(i * step)] for i in range(n)]


AUX_OOF = "data/local/modeling/training_pipeline_v4/final_aux/aux_oof_2025.parquet"


def swap_in_aux_oof(X: pd.DataFrame, aux: pd.DataFrame) -> pd.DataFrame:
    """2025 年の行の補助ヘッド(mtl_*)を、2024 年までで学習した OOF の値に差し替える。最終の補助モデルは 2025 年まで学習済みなので、
    そのまま使うと 2025 年の行では学習内の予測になり、2026 年（学習外）と比べられない。"""
    key = X[["race_id", "horse_id"]].astype(str)
    sub = key.merge(aux, on=["race_id", "horse_id"], how="left")
    for c in [c for c in aux.columns if c.startswith("mtl_") and c in X.columns]:
        X[c] = sub[c].to_numpy()
    return X


def drop_training_predict(v3, X: pd.DataFrame, storage, frac: float, seed: int = 0) -> pd.DataFrame:
    """X の tl_・wc_ を、馬の調教記録の frac を落とした状態で作り直し、予測し直す（相対化の列も、残った馬だけで作られる）。戻り値は p_win_d・p_top2_d・p_top3_d。"""
    import hashlib

    from src.pipeline.features import serving_stage_rel as ssr
    from src.pipeline.features import serving_v5 as sv5

    keys = ["race_id", "horse_id"]
    hs = X["horse_id"].astype(str).unique().tolist()
    recs = v3._training_records(hs, storage, 8)

    def keep(h: str) -> bool:
        return int(hashlib.md5(f"{seed}:{h}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF >= frac

    sub = {h: (r if keep(h) else None) for h, r in recs.items()}
    base = X[keys + ["day"]].copy()
    base["date"] = pd.to_datetime(base["day"].astype(str), format="%Y%m%d")
    tl_cols = [c for c in v3.feature_cols if c.startswith(ssr.TL_PREFIX)]
    wc_cols = [c for c in v3.feature_cols if c.startswith("wc_")]
    X2 = X.copy()
    tl = ssr.tl_columns_for_races(base[keys + ["date"]], sub, v3.tl_ref)
    tl = base[keys].merge(tl, on=keys, how="left").drop(columns=keys).reindex(columns=tl_cols)
    X2[tl_cols] = tl.to_numpy()
    if wc_cols:
        wc = sv5.cy_columns_for_races(base[keys + ["date"]], sub, v3.tl_ref)
        wc = base[keys].merge(wc, on=keys, how="left").drop(columns=keys).reindex(columns=wc_cols)
        X2[wc_cols] = wc.to_numpy()
    P = v3.predict(X2)
    return pd.DataFrame({"p_win_d": P["p_win"].to_numpy(), "p_top2_d": P["p_top2"].to_numpy(), "p_top3_d": P["p_top3"].to_numpy()}, index=X.index)


def parse_months(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2025,2026")
    ap.add_argument("--months", default="1-10")
    ap.add_argument("--days", default="", help="YYYYMMDD,YYYYMMDD…: 月ごとの選択ではなく、この開催日を抜き取る（週次監視用。--years/--months は無視）")
    ap.add_argument("--races-per-day", type=int, default=10)
    ap.add_argument("--out", default="data/local/inference/drift_features")
    ap.add_argument("--max-load", type=float, default=14.0)
    ap.add_argument("--bundle", default="", help="省略時は v5 の既定（KEIBA_V5_BUNDLE）")
    ap.add_argument("--drop-training", type=float, default=0.0,
                    help="0〜1: 馬ごとの調教記録をこの割合だけ落として tl_・wc_ を作り直した予測（p_win_d など）も付ける。2026 年 4 月以降の『調教ページが揃っていない』状態の再現（マスク実験より実際に近い）")
    ap.add_argument("--stage", default="", help="前段モデルの置き場。省略時は v5 の既定（2025 年の日は学習に入っていない v4_verify_2024 を指定する）")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()
    from src.pipeline.inference import weight_triggered_predict as wtp
    from src.pipeline.inference import v3_serving as vs
    from src.scraper.storage import HybridStorage
    from src.scripts.data.backfill_predictions import wait_for_load

    pth = vs.v5_paths()
    bundle = Path(a.bundle) if a.bundle else pth["bundle"]
    tables = Path(os.environ.get("KEIBA_TABLES_DIR") or "data/page_reference/tables")
    root_dir = Path(".").resolve()
    out_root = Path(a.out)
    plan: dict[str, list[str]] = {}
    if a.days:
        from src.scripts.data.backfill_predictions import race_days

        for d in [x for x in a.days.split(",") if x]:
            plan.update(race_days([int(d[:4])], tables, d, d))
    else:
        for y in [int(x) for x in a.years.split(",")]:
            plan.update(pick_days(tables, y, parse_months(a.months)))
    plan = {d: spread(ids, a.races_per_day) for d, ids in sorted(plan.items())}
    todo = {d: ids for d, ids in plan.items() if not (out_root / d[:4] / f"{d}.parquet").exists()}
    print(json.dumps({"days": len(plan), "todo": len(todo), "races": sum(len(v) for v in plan.values())}))
    if not todo:
        return

    storage = HybridStorage()
    ctx = wtp.InferenceContext()
    ctx.warm()
    assert ctx.raw_hist is not None, "KEIBA_INFERENCE_ENGINE=full で実行する"
    raw_full = ctx.raw_hist
    v3 = vs.V3Serving(bundle, Path(a.stage) if a.stage else pth["stage"], tl_reference=pth["tl_reference"])
    v3.warm()
    hist_full = v3.history.copy()
    aux_oof = None
    if max(int(d[:4]) for d in todo) <= 2025 and (root_dir / AUX_OOF).exists():
        aux_oof = pd.read_parquet(root_dir / AUX_OOF)
        aux_oof["race_id"], aux_oof["horse_id"] = aux_oof["race_id"].astype(str), aux_oof["horse_id"].astype(str)
        logger.info("2025 年の補助ヘッドは OOF（2024 年まで学習）に差し替える: %d 行", len(aux_oof))
    for day, ids in todo.items():
        wait_for_load(a.max_load)
        d = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        t0 = time.time()
        try:
            card = wtp.fill_card_meta(wtp.card_frame(storage, ids), [])
            if card.height == 0:
                logger.warning("%s: 出馬表が無い", day)
                continue
            ok_ids = (card.filter(pl.col("horse_number").fill_null(0) > 0)
                      .filter(pl.col("distance").cast(pl.Float64, strict=False) > 0)
                      .filter(pl.col("surface").cast(pl.Utf8).fill_null("") != "")["race_id"].unique().to_list())
            card = card.filter(pl.col("race_id").is_in(ok_ids))
            if card.height == 0:
                continue
            cut = datetime.date.fromisoformat(d) if raw_full.schema["date"] == pl.Date else d
            ctx.raw_hist = raw_full.filter(pl.col("date") < cut)
            F, _info = ctx.build(card)
            parts = []
            for rid in sorted(ok_ids):
                cr = card.filter(pl.col("race_id") == rid)
                st = str(cr["start_time"][0])
                h = hist_full[(hist_full["date"].astype(str) < d)
                              | ((hist_full["date"].astype(str) == d) & (hist_full["start_time"].astype(str) < st) & (hist_full["race_id"].astype(str) != rid))]
                X = v3.features(cr, F.filter(pl.col("race_id") == rid), storage, history=h)
                X["day"] = day
                if aux_oof is not None:
                    X = swap_in_aux_oof(X, aux_oof)
                P = v3.predict(X)
                for k in ("p_win", "p_top2", "p_top3"):
                    X[k] = P[k].to_numpy()
                if a.drop_training > 0:
                    Pd = drop_training_predict(v3, X, storage, a.drop_training)
                    for k in Pd.columns:
                        X[k] = Pd[k].to_numpy()
                parts.append(X)
            if parts:
                (out_root / day[:4]).mkdir(parents=True, exist_ok=True)
                tmp = out_root / day[:4] / f".{day}.parquet.tmp"
                pd.concat(parts, ignore_index=True).to_parquet(tmp, index=False)
                tmp.replace(out_root / day[:4] / f"{day}.parquet")
            logger.info("%s: %d レース %.0f 秒", day, len(parts), time.time() - t0)
        except Exception:  # noqa: BLE001
            logger.exception("%s: 失敗（次の日へ）", day)


if __name__ == "__main__":
    main()
