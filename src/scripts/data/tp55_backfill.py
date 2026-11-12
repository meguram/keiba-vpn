"""tp-55: 指定日の全レースについて、ラップ傾向別のハロン予想を作って保存する（予測 API が KEIBA_LAP_FORECAST=1 で読む）。ホストで実行。

python -m src.scripts.data.tp55_backfill --dates 2026-10-17 2026-10-18 [--model-dir /home/jovyan/work/tp55_out] [--dry-run]
python -m src.scripts.data.tp55_backfill --from 2026-01-01 --to 2026-10-18
python -m src.scripts.data.tp55_backfill --dates 2025-04-05 --model-dir retrain_out/base_2016/final_lap --model-full model_A_full.pkl --model-noten model_A_noten.pkl --lapd-dir retrain_out/base_2016/final_lap --years 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026   # 再学習後の新モデル（学習側の races_feat は 2016〜なので years も合わせる）

必要: その日の race_shutuba_flat（出馬表の書き出し）と、それ以前の race_result_flat（過去走）。テン1F（SmartRC）があるレースは A_full、
無いレース（2026 年の表が無い、取得前）は A_noten のモデルで予測する。出力: data/local/inference/lap_forecast/<日付>/<race_id>.json。
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.inference import lap_forecast_service as svc
from src.pipeline.models import lap_forecast as lfm
from src.scripts.data import tp55_build as tb


def _dates(a: argparse.Namespace) -> list[str]:
    if a.dates:
        return a.dates
    d0, d1 = date.fromisoformat(a.date_from), date.fromisoformat(a.date_to)
    return [(d0 + timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]


def forecast_frame(df: pd.DataFrame, models: dict, names_by_model: dict, lapd: svc.LapdModel | None = None) -> list[dict]:
    docs: list[dict] = []
    for variant, sub in (("A_full", df[df["ten_cnt"] > 0]), ("A_noten", df[~(df["ten_cnt"] > 0)])):
        m = models.get(variant)
        if m is None or not len(sub):
            continue
        sub = sub.reset_index(drop=True)
        out = m.predict(sub)
        ld = lapd.predict(sub) if lapd is not None else None
        names = names_by_model[variant]
        for i in range(len(sub)):
            if np.isnan(out["par"][i, 0]):
                continue
            summ = lfm.mixture_summary(out, i, names, seed=i)
            par = [None if np.isnan(x) else round(float(x), 2) for x in out["par"][i, : int(out["n_laps"][i])]]
            docs.append(svc.make_doc(str(sub.at[i, "race_id"]), str(sub.at[i, "date"]), int(sub.at[i, "distance"]), summ, variant, par,
                                 {k: v[i] for k, v in ld.items()} if ld else None))
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="*")
    ap.add_argument("--from", dest="date_from")
    ap.add_argument("--to", dest="date_to")
    ap.add_argument("--model-dir", default="/home/jovyan/work/tp55_out")
    ap.add_argument("--model-full", default="model_final2026_A_full.pkl")
    ap.add_argument("--model-noten", default="model_final2026_A_noten.pkl")
    ap.add_argument("--lapd-dir", help="final_lap のディレクトリ（lapd_meta.json と lapd/）。指定すると ld_*（前後半 3F の直接予測）を doc の lapd に保存する")
    ap.add_argument("--years", type=int, nargs="+", default=tb.YEARS)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    md = Path(a.model_dir)
    models = {"A_full": lfm.LapForecast.load(md / a.model_full), "A_noten": lfm.LapForecast.load(md / a.model_noten)}
    names = {k: v.scenario_names() for k, v in models.items()}
    dates = _dates(a)
    df, rep = tb.build_infer(a.years, dates)
    if df.empty:
        print(json.dumps({"dates": dates, "n_features": 0, "n_forecast": 0, "skipped": rep.get("skipped") or "no_rows"}, ensure_ascii=False))
        return
    docs = forecast_frame(df, models, names, svc.LapdModel(a.lapd_dir) if a.lapd_dir else None)
    if not a.dry_run:
        for d in docs:
            svc.save_forecast(d)
    print(json.dumps({"dates": dates, "n_features": rep.get("n_races_features"), "n_forecast": len(docs), "dry_run": a.dry_run}, ensure_ascii=False))


if __name__ == "__main__":
    main()
