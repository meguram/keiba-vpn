"""推論経路（v3_serving）の特徴量 X と、学習時の同じ行の特徴量を列単位で比べ、列群ごとに学習側の値へ差し替えたときの LogLoss の戻りを測る（診断のみ）。

python -m src.scripts.data.diag_v3_serving_gap --x /tmp/serve_v3_X.parquet --out /tmp/diag_v3_gap.json
  X は verify_v3_serving --dump-x の出力。バンドルは bundle_2024（2020〜2024 で学習）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.models import tp05_data
from src.pipeline.models.ensemble_bundle import EnsembleBundle

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "data/local/modeling/training_pipeline"
KEYS = ["race_id", "horse_id"]

PRIOR = ("jockey_prior", "trainer_prior", "horse_prior")
BIAS = ("track_bias", "sire_prior", "dam_sire_prior", "sire_x", "dam_sire_x")


def group_of(c: str, fam: set[str]) -> str:
    if c.startswith("tl_"):
        return "tl"
    if c in tp05_data.STAGE_COLS:
        return "stage"
    if c.endswith(("_rz", "_rrank", "_rgap")):
        return "rel"
    if c.startswith("ped_"):
        return "ped"
    if c in ("oikiri_eval", "has_comment", "comment_len", "comment_pos", "comment_neg", "oikiri_eval_rank", "oikiri_eval_gap"):
        return "extras"
    if c in fam:
        return "history"
    if c.startswith(("jockey_prior", "trainer_prior")):
        return "layera_jt_prior"
    if c.startswith("horse_prior"):
        return "layera_horse_prior"
    if c.startswith(BIAS) or c.startswith("sire") or c.startswith("dam_sire"):
        return "layera_bias_sire"
    return "layera_race_cond"


def ll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train_x() -> tuple[pd.DataFrame, pd.DataFrame]:
    D = tp05_data.load_dataset(WORK)
    df, meta = D["df"], D["meta"]
    EP, _EX, _PED = tp05_data.load_extras_ped(df, WORK)
    TL = pd.read_parquet(WORK / "training_laps_b/cand_TL_A.parquet")
    TLm = df[KEYS].merge(TL, on=KEYS, how="left")
    TLm.index = df.index
    TLX = TLm[[c for c in TLm.columns if c not in KEYS]].add_prefix("tl_")
    X = pd.concat([df[D["FEATS"]["base+S+C1"]], EP, TLX], axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    X = pd.concat([df[KEYS].astype(str), X], axis=1)
    return X, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True)
    ap.add_argument("--bundle", default=str(WORK / "final_v3/bundle_2024"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    xs = pd.read_parquet(a.x)
    xs[KEYS] = xs[KEYS].astype(str)
    Xt, meta = train_x()
    meta = meta.assign(race_id=meta["race_id"].astype(str), horse_id=meta["horse_id"].astype(str))
    bundle = EnsembleBundle.load(a.bundle)
    cols = list(dict.fromkeys(c for m in bundle.members for c in m.feature_cols))
    fam_cols = set(json.loads((WORK / "features/feature_families.json").read_text(encoding="utf-8")).get("__all__", [])) or {
        c for v in json.loads((WORK / "features/feature_families.json").read_text(encoding="utf-8")).values() for c in v}

    j = xs[KEYS].merge(Xt, on=KEYS, how="left", suffixes=("", "")).merge(meta[KEYS + ["finish_position"]], on=KEYS, how="left")
    ok = j[cols[0]].notna() | True
    xs = xs.reset_index(drop=True)
    j = j.reset_index(drop=True)
    assert len(j) == len(xs)
    y = (j["finish_position"] == 1).to_numpy().astype(float)
    have = j["finish_position"].notna().to_numpy()
    xs, j, y = xs[have].reset_index(drop=True), j[have].reset_index(drop=True), y[have]

    # 列ごとの比較
    cmp = []
    for c in cols:
        a_, b_ = xs[c], j[c]
        if str(a_.dtype) in ("object", "string", "category") or str(b_.dtype) == "category":
            sa, sb = a_.astype(str).where(a_.notna(), "(なし)"), b_.astype(str).where(b_.notna(), "(なし)")
            cmp.append({"col": c, "group": group_of(c, fam_cols), "match": float((sa == sb).mean()), "nan_one_side": 0.0, "corr": None, "mean_diff": None})
            continue
        x, t = a_.to_numpy(dtype=float), b_.to_numpy(dtype=float)
        both = np.isfinite(x) & np.isfinite(t)
        eq = (np.isnan(x) & np.isnan(t)) | np.isclose(x, t, rtol=1e-4, atol=1e-6)
        corr = float(np.corrcoef(x[both], t[both])[0, 1]) if both.sum() > 3 and np.std(x[both]) > 0 and np.std(t[both]) > 0 else None
        cmp.append({"col": c, "group": group_of(c, fam_cols), "match": float(eq.mean()), "nan_one_side": float((np.isnan(x) ^ np.isnan(t)).mean()),
                    "corr": corr, "mean_diff": float(np.mean(x[both] - t[both])) if both.any() else None, "mean_abs_diff": float(np.mean(np.abs(x[both] - t[both]))) if both.any() else None})
    cdf = pd.DataFrame(cmp)
    race = j["race_id"].to_numpy()

    def pred(X: pd.DataFrame) -> np.ndarray:
        return bundle.predict(X[cols], race)["p_win"].to_numpy()

    base_serv = ll(pred(xs), y)
    base_train = ll(pred(j), y)
    groups = sorted(cdf["group"].unique())
    swap = {}
    for g in groups:
        gc = cdf.loc[cdf["group"] == g, "col"].tolist()
        X2 = xs.copy()
        for c in gc:
            X2[c] = j[c].to_numpy() if str(j[c].dtype) != "category" else j[c].astype(object).to_numpy()
        swap[g] = {"n_cols": len(gc), "ll": ll(pred(X2), y)}
        swap[g]["gain_vs_serving"] = base_serv - swap[g]["ll"]
    # 差が大きい列だけ（match < 0.99）を差し替え
    bad = cdf.loc[cdf["match"] < 0.99, "col"].tolist()
    X3 = xs.copy()
    for c in bad:
        X3[c] = j[c].to_numpy() if str(j[c].dtype) != "category" else j[c].astype(object).to_numpy()
    swap["__mismatched_cols_only__"] = {"n_cols": len(bad), "ll": ll(pred(X3), y), "gain_vs_serving": base_serv - ll(pred(X3), y)}
    rep = {"rows": int(len(y)), "ll_serving": base_serv, "ll_training_features": base_train, "swap_group": swap,
           "group_match": cdf.groupby("group")["match"].agg(["mean", "min"]).round(4).to_dict("index"),
           "worst_cols": cdf.sort_values("match").head(40).round(4).to_dict("records")}
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("rows", "ll_serving", "ll_training_features", "swap_group", "group_match")}, ensure_ascii=False, indent=1, default=float))
    print(cdf.sort_values("match").head(40).to_string())


if __name__ == "__main__":
    main()
