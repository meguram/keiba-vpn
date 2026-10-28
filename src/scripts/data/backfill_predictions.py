"""画面に出す予測が無い過去のレースの予測を、推論経路（v3_serving）で算出して保存する（バックフィル）。

  KEIBA_INFERENCE_ENGINE=full python -m src.scripts.data.backfill_predictions --list --from 20260101 --to 20261018
  KEIBA_INFERENCE_ENGINE=full nice -n 19 python -m src.scripts.data.backfill_predictions --from 20260101 --to 20261018 --max-days 2

方針:
  - 対象は『GCS の race_predictions・ローカルの shadow（predictions_v3）・バックフィル済み』のどれも無いレース。
  - 特徴量は verify_v3_serving と同じ手順で、その日より前の履歴だけ（full エンジン。バンドルの Elo は現在値なので使わない）に、
    同日で先に発走したレースの結果だけを足して作る。モデルは 2020〜2025 年で学習したバンドルなので、2026 年は未学習の期間。
    2026 年より前の日付は学習期間に入るため、既定で対象外（--allow-in-sample で解除）。
  - 保存先は ``<推論ディレクトリ>/predictions_backfill/<モデル名>/<日>/<race_id>.parquet``（shadow と同じ列）。出力済みはスキップ。
  - ホストの負荷が高いとき（load1 が --max-load を超える間）は待つ。
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

logger = logging.getLogger("scripts.backfill_predictions")

JRA_VENUES = {f"{i:02d}" for i in range(1, 11)}
TRAIN_LAST_YEAR = 2025


def backfill_root(label: str | None = None) -> Path:
    from src.pipeline.inference import weight_triggered_predict as wtp

    root = wtp.inference_dir() / "predictions_backfill"
    return root / label if label else root


def model_label(bundle_dir: str | Path) -> str:
    return Path(bundle_dir).name


def race_days(years: list[int], tables_dir: Path, since: str, until: str) -> dict[str, list[str]]:
    """開催日 → race_id 一覧（race_shutuba_flat から。JRA の競馬場コードだけ）。"""
    out: dict[str, list[str]] = {}
    for y in years:
        p = tables_dir / str(y) / "race_shutuba_flat.parquet"
        if not p.exists():
            continue
        df = pl.read_parquet(p, columns=["race_id", "date"]).unique().to_pandas()
        df["day"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y%m%d")
        for day, g in df.dropna(subset=["day"]).groupby("day"):
            if since <= day <= until:
                ids = sorted(r for r in g["race_id"].astype(str) if len(r) == 12 and r[4:6] in JRA_VENUES)
                if ids:
                    out[day] = ids
    return dict(sorted(out.items()))


def existing_predictions(storage, inference_dir: Path, label: str, years: list[int]) -> dict[str, set[str]]:
    """予測がすでにある race_id（GCS / shadow / バックフィル）。"""
    gcs: set[str] = set()
    for y in years:
        try:
            gcs |= set(storage.list_keys("race_predictions", str(y)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("GCS の race_predictions 一覧を取れなかった(%s): %s", y, exc)
    shadow = {p.stem for p in (inference_dir / "predictions_v3").glob("*/*.parquet")}
    back = {p.stem for p in (inference_dir / "predictions_backfill" / label).glob("*/*.parquet")}
    return {"gcs": gcs, "shadow": shadow, "backfill": back}


def todo_by_day(days: dict[str, list[str]], have: dict[str, set[str]]) -> dict[str, list[str]]:
    skip = have["gcs"] | have["shadow"] | have["backfill"]
    return {d: [r for r in ids if r not in skip] for d, ids in days.items() if any(r not in skip for r in ids)}


def wait_for_load(max_load: float, sleep_sec: float = 60.0) -> None:
    while True:
        try:
            load1 = os.getloadavg()[0]
        except OSError:
            return
        if load1 <= max_load:
            return
        logger.info("ホストの負荷が高いので待つ: load1=%.1f > %.1f", load1, max_load)
        time.sleep(sleep_sec)


def compare_with_shadow(new_dir: Path, shadow_dir: Path) -> dict:
    """バックフィル(as-of の full エンジン)と shadow(当時の bundle エンジン)の確率を、同じ馬で突き合わせる。"""
    import numpy as np

    rows = []
    for f in sorted(new_dir.glob("*.parquet")):
        s = shadow_dir / f.name
        if not s.exists():
            continue
        a_, b_ = pd.read_parquet(f), pd.read_parquet(s)
        m = a_.merge(b_, on=["race_id", "horse_id"], suffixes=("", "_shadow"))
        rows.append(m)
    if not rows:
        print(json.dumps({"compare": "no_overlap"}))
        return {}
    m = pd.concat(rows, ignore_index=True)
    out = {"races": int(m["race_id"].nunique()), "rows": int(len(m))}
    for k in ("p_win", "p_top2", "p_top3"):
        x, y = m[k].to_numpy(float), m[f"{k}_shadow"].to_numpy(float)
        out[k] = {"corr": float(np.corrcoef(x, y)[0, 1]), "mean_abs_diff": float(np.mean(np.abs(x - y))), "max_abs_diff": float(np.max(np.abs(x - y)))}
    print(json.dumps({"compare_with_shadow": out}, ensure_ascii=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="since", default="20260101")
    ap.add_argument("--to", dest="until", default=datetime.date.today().strftime("%Y%m%d"))
    ap.add_argument("--list", action="store_true", help="対象を数えるだけ（計算しない）")
    ap.add_argument("--max-days", type=int, default=0, help="この日数だけ処理して終わる（0=すべて）")
    ap.add_argument("--max-races", type=int, default=0, help="1 日あたり先頭の何レースだけ算出するか（確認用。0=すべて）")
    ap.add_argument("--oldest-first", action="store_true", help="既定は新しい日から")
    ap.add_argument("--max-load", type=float, default=18.0)
    ap.add_argument("--bundle", default="")
    ap.add_argument("--stage", default="")
    ap.add_argument("--check-shadow", default="", help="この日(YYYYMMDD)を、shadow が既にあっても算出し(保存先は predictions_backfill_check)、shadow と突き合わせる（リーク/一致の確認用）")
    ap.add_argument("--model", default="", choices=["", "v6e"], help="v6e: バンドル・前段・追い切り基準表・RB 状態を v3_serving.v6_paths() から取る")
    ap.add_argument("--all", action="store_true", help="GCS・shadow・他モデルのバックフィルに予測があるレースも算出する（モデルの入れ替え用）")
    ap.add_argument("--out-root", default="", help="出力先の root（既定: <推論ディレクトリ>/predictions_backfill）。差し替え前の検証用に別の場所へ出せる")
    ap.add_argument("--allow-in-sample", action="store_true", help="学習期間(≦2025)の日付も対象にする（予測は学習済みデータに対するもの）")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from src.utils.project_env import load_project_dotenv

    load_project_dotenv()                                   # KEIBA_V3_BUNDLE / KEIBA_V3_STAGE（ホストの .env）を読む
    from src.pipeline.inference import weight_triggered_predict as wtp
    from src.pipeline.inference.v3_serving import V3Serving, default_bundle_dir, default_stage_dir
    from src.scraper.storage import HybridStorage

    v6p = None
    if a.model == "v6e":
        from src.pipeline.inference.v3_serving import v6_paths

        v6p = v6_paths()
    bundle = Path(a.bundle) if a.bundle else (v6p["bundle"] if v6p else default_bundle_dir())
    stage = Path(a.stage) if a.stage else (v6p["stage"] if v6p else default_stage_dir(bundle))
    label = model_label(bundle)
    out_root = Path(a.out_root) if a.out_root else None
    years = sorted({int(a.since[:4]) + i for i in range(int(a.until[:4]) - int(a.since[:4]) + 1)})
    if not a.allow_in_sample:
        years = [y for y in years if y > TRAIN_LAST_YEAR]
    tables = Path(os.environ.get("KEIBA_TABLES_DIR") or "data/page_reference/tables")
    storage = HybridStorage()
    inf = wtp.inference_dir()

    days = race_days(years, tables, a.since, a.until)
    have = existing_predictions(storage, inf, label, years)
    if out_root is not None:
        have["backfill"] = {p.stem for p in (out_root / label).glob("*/*.parquet")}
    if a.all:
        have = {"gcs": set(), "shadow": set(), "backfill": have["backfill"]}
    if a.check_shadow:
        d0 = a.check_shadow
        days = race_days([int(d0[:4])], tables, d0, d0)
        have = {"gcs": set(), "shadow": set(), "backfill": {p.stem for p in (inf / "predictions_backfill_check" / label / d0).glob("*.parquet")}}
    todo = todo_by_day(days, have)
    n_all = sum(len(v) for v in days.values())
    n_todo = sum(len(v) for v in todo.values())
    summary = {"label": label, "days": len(days), "races": n_all, "have_gcs": len(have["gcs"]), "have_shadow": len(have["shadow"]),
               "have_backfill": len(have["backfill"]), "todo_days": len(todo), "todo_races": n_todo}
    print(json.dumps(summary, ensure_ascii=False))
    if a.list:
        for d, ids in todo.items():
            print(d, len(ids))
        return
    if not todo:
        return

    ctx = wtp.InferenceContext()
    ctx.warm()
    assert ctx.raw_hist is not None, "KEIBA_INFERENCE_ENGINE=full で実行する"
    raw_full = ctx.raw_hist
    v3 = V3Serving(bundle, stage, tl_reference=v6p["tl_reference"]) if v6p else V3Serving(bundle, stage)
    if v6p:
        v3.rb_state = v6p["rb_state"]
    v3.warm()
    hist_full = v3.history.copy()
    order = sorted(todo, reverse=not a.oldest_first)
    done_days = 0
    for day in order:
        if a.max_days and done_days >= a.max_days:
            break
        wait_for_load(a.max_load)
        ids = todo[day][: a.max_races] if a.max_races else todo[day]
        d = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        t0 = time.time()
        try:
            card = wtp.fill_card_meta(wtp.card_frame(storage, ids), [])
            if card.height == 0:
                logger.warning("%s: 出馬表が無いので算出不能（%d レース）", day, len(ids))
                continue
            ok_ids = (card.filter(pl.col("horse_number").fill_null(0) > 0)
                      .filter(pl.col("distance").cast(pl.Float64, strict=False) > 0)
                      .filter(pl.col("surface").cast(pl.Utf8).fill_null("") != "")["race_id"].unique().to_list())
            card = card.filter(pl.col("race_id").is_in(ok_ids))
            if card.height == 0:
                logger.warning("%s: 面・距離・馬番が揃った出馬表が無い", day)
                continue
            cut = datetime.date.fromisoformat(d) if raw_full.schema["date"] == pl.Date else d
            ctx.raw_hist = raw_full.filter(pl.col("date") < cut)
            F, _info = ctx.build(card)
            out_dir = (inf / "predictions_backfill_check" / label / day) if a.check_shadow else ((out_root / label / day) if out_root else (backfill_root(label) / day))
            out_dir.mkdir(parents=True, exist_ok=True)
            n_ok = 0
            for rid in sorted(ok_ids):
                cr = card.filter(pl.col("race_id") == rid)
                st = str(cr["start_time"][0])
                h = hist_full[(hist_full["date"].astype(str) < d)
                              | ((hist_full["date"].astype(str) == d) & (hist_full["start_time"].astype(str) < st) & (hist_full["race_id"].astype(str) != rid))]
                X = v3.features(cr, F.filter(pl.col("race_id") == rid), storage, history=h)
                P = v3.predict(X)
                tmp = out_dir / f".{rid}.parquet.tmp"
                P.to_parquet(tmp, index=False)
                tmp.replace(out_dir / f"{rid}.parquet")
                n_ok += 1
            logger.info("%s: %d/%d レース算出 %.0f 秒", day, n_ok, len(ids), time.time() - t0)
            done_days += 1
            if a.check_shadow:
                compare_with_shadow(out_dir, inf / "predictions_v3" / day)
        except Exception:  # noqa: BLE001
            logger.exception("%s: 算出に失敗（次の日へ）", day)


if __name__ == "__main__":
    main()
