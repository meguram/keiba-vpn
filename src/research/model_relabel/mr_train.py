"""拡張ラベルの学習（担当 model-relabel）: v5bag と v6e の二値の代理を、ラベルと重みだけ替えて学ぶ。

- 特徴量の表は cycles16_fix3 の X_xgb（読むだけ）。v5bag の列は megu_cycles の `_rich_matrix("v5")`、v6e の代理は
  final_v6e16 の束の xgb_extras_ped の feature_cols（1,231 列）。木の設定は v5bag の BAG_SETS・XGB_PARAMS、
  v6e は xgb_extras_ped の params に固定する（選び直さない。config/model_relabel/spec.json）。
- 目標: win（元 y1・拡張 y_w）・top3（元 y3・拡張 y_p）・notbig（P-ord の m_a = 大敗でない）。
- 変種: orig（元のラベル）・ext_w1・ext_w2・ext_w4（拡張のラベル、大敗の負例の重み w）。
- 分け:
  - time: 年 t（2017〜24）は t より前の年で学ぶ。
  - meeting: 開催の 5 分割（docs/operations/model_relabel/meeting_folds.csv）。学びの側の calib_k=1 の開催は学びに入れず較正に回す。
- 途中の木の予測（spec の rounds）を全部保存する（打ち切りを L で選ぶため）。部品ごとに保存して再開できる。
- v6e の代理の注意: 束の 1,231 列には、v5 で外した列が 148 列ある（rb_*・mtl_*・s_* は未来の年を含む上流の学習・集計、
  tl_f_untimed_later はその走の後の値）。本番の列のまま比べるので外さないが、値の水準と Benter の改善分は汚れる。主の判定は v5bag で行う。
  乱数は 1 本（束は 5 本）、カテゴリの番号は X_xgb の cat.codes（束の cat_levels は当てない）。
- 2025 年以降の行は読み込んだ直後に落とす。費用 0 円（手元の表だけ。GCS を読まない）。nice -n 19 で流す。
  KEIBA_SESSION=model-relabel nice -n 19 python -m src.research.model_relabel.mr_train run --fs v5bag --target win --variant orig
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.model_relabel import mr_labels as LB

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "config/model_relabel/spec.json"
OUT = LB.OUT
PARTS = OUT / "parts"
V6E_BUNDLE = Path(os.environ.get("MODEL_RELABEL_V6E_BUNDLE")
                  or "/home/jovyan/work/retrain_out/base_2016/final_v6e16/bundle_v6_2025/bundle.json")
FOLDS_CSV = ROOT / "docs/operations/model_relabel/meeting_folds.csv"
YEARS = range(2017, 2025)
NTHREAD = int(os.environ.get("MODEL_RELABEL_NTHREAD") or 4)


def spec() -> dict:
    return json.loads(SPEC.read_text(encoding="utf-8"))


def fs_configs(fs: str) -> tuple[list[dict], list[int]]:
    """木の設定（固定）と保存する本数の一覧。"""
    from src.research.megu_cycles import mc_data as MC
    sp = spec()["select"]
    if fs == "v5bag":
        sets = [dict(MC.XGB_PARAMS, **bs) for bs in MC.BAG_SETS]
        rounds = [int(r) for r in sp["rounds_v5bag"]]
    elif fs == "v6e":
        b = json.loads(V6E_BUNDLE.read_text(encoding="utf-8"))
        m = next(x for x in b["members"] if x["name"] == "xgb_extras_ped")
        p = m["params"]
        sets = [dict(objective="binary:logistic", tree_method="hist", device="cuda", eval_metric="logloss", seed=0,
                     max_depth=int(p["max_depth"]), learning_rate=float(p["learning_rate"]), subsample=float(p["subsample"]),
                     colsample_bytree=float(p["colsample_bytree"]), reg_lambda=float(p["reg_lambda"]),
                     min_child_weight=float(p["min_child_weight"]))]
        rounds = [int(r) for r in sp["rounds_v6e"]]
        assert rounds[-1] == int(p["n_estimators"])
    else:
        raise ValueError(fs)
    for s in sets:
        s["nthread"] = NTHREAD
    return sets, rounds


def load_matrix(fs: str):
    """X（float32）と meta（race_id・horse_id・year・finish_position）を year<=2024 の行だけ返す。"""
    from src.research.megu_cycles import mc_data as MC
    if fs == "v5bag":
        X, meta, names = MC._rich_matrix("v5")
    else:
        import pyarrow.parquet as pq
        cols = json.loads(V6E_BUNDLE.read_text(encoding="utf-8"))
        cols = next(x for x in cols["members"] if x["name"] == "xgb_extras_ped")["feature_cols"]
        yr_all = pd.to_numeric(pq.read_table(MC.META_RICH, columns=["year"]).column("year").to_pandas()).to_numpy()
        keep = yr_all <= 2024
        meta = pd.read_parquet(MC.META_RICH, filters=[("year", "<=", 2024)])
        assert len(meta) == keep.sum()
        X = pd.read_parquet(MC.X_RICH, columns=cols)
        X = X[keep].reset_index(drop=True)
        for c in X.columns:
            if isinstance(X[c].dtype, pd.CategoricalDtype):
                X[c] = X[c].cat.codes.where(X[c].notna(), np.nan)
            elif not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = pd.to_numeric(X[c], errors="coerce")
        X, names = X.astype(np.float32), cols
    meta = meta.reset_index(drop=True)
    assert pd.to_numeric(meta["year"]).max() <= 2024
    return X, meta, names


def attach_labels(meta: pd.DataFrame, conv_key: str = "convert_A", max_year: int = 2024) -> pd.DataFrame:
    """meta の行の順に、結果表の拡張ラベルを付ける（race_id・horse_id で結ぶ。行数と y1 の一致を確かめる）。"""
    path = OUT / f"labels_{conv_key}.parquet"
    lab = pd.read_parquet(path, filters=[("year", "<=", max_year)]) if path.exists() else LB.labels_dev(conv_key)
    lab = lab[lab["year"] <= max_year].drop_duplicates(["race_id", "horse_id"])
    key = meta[["race_id", "horse_id"]].astype(str)
    m = key.merge(lab[["race_id", "horse_id", "y1", "y3", "y_w", "y_p", "big", "ext_ok"]], on=["race_id", "horse_id"], how="left")
    assert len(m) == len(meta)
    miss = m["y1"].isna()
    fin = pd.to_numeric(meta["finish_position"], errors="coerce").to_numpy()
    # 結果表に無い行（取消など）は負例・大敗でない
    assert (fin[miss.to_numpy()] != 1).all(), "結果表に無い勝ち馬がいる"
    for c in ("y1", "y3", "y_w", "y_p", "big"):
        m[c] = m[c].fillna(False).astype(bool)
    m["ext_ok"] = m["ext_ok"].fillna(True).astype(bool)
    agree = (m["y1"].to_numpy() == (fin == 1)).mean()
    assert agree > 0.999, f"meta と結果表の 1 着の一致が低い: {agree}"
    m["year"] = pd.to_numeric(meta["year"]).to_numpy(int)
    m["meeting"] = m["race_id"].str[:8]
    return m


def target_and_weight(lab: pd.DataFrame, target: str, variant: str) -> tuple[np.ndarray, np.ndarray]:
    big = lab["big"].to_numpy()
    if target == "notbig":
        assert variant == "orig"
        return (~big).astype(np.float32), np.ones(len(lab), np.float32)
    col = {"win": ("y1", "y_w"), "top3": ("y3", "y_p")}[target]
    if variant == "orig":
        return lab[col[0]].to_numpy(np.float32), np.ones(len(lab), np.float32)
    w = float(variant.split("_w")[1])
    y = lab[col[1]].to_numpy(np.float32)
    wt = np.where(big & (y == 0), w, 1.0).astype(np.float32)
    return y, wt


def _part_path(split, fs, target, variant, conv, bag, unit) -> Path:
    return PARTS / split / f"{fs}_{target}_{variant}_{conv}" / f"bag{bag}_{unit}.npy"


RACE_DAYS = {"1128", "1129", "1130", "1205", "1206"}      # 土日のほかの開催日（月曜の開催など）


def blocked(now=None) -> bool:
    """開催日（土日と RACE_DAYS）の 07:30〜17:00（JST）は重い学習を止める。"""
    import datetime as dt
    now = now or dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))
    day = now.weekday() >= 5 or now.strftime("%m%d") in RACE_DAYS
    hm = now.hour * 100 + now.minute
    return day and 730 <= hm < 1700


def _fingerprint(lab: pd.DataFrame, target: str, variant: str, sets: list[dict], rounds: list[int]) -> dict:
    """部品の作りの印（spec の版・ラベルの中身・木の設定・本数）。違う印の部品は混ぜない。"""
    import hashlib
    y, wt = target_and_weight(lab, target, variant)
    h = hashlib.md5(np.ascontiguousarray(np.stack([y, wt])).tobytes()).hexdigest()
    return {"spec_version": spec()["version"], "label_md5": h, "sets": sets, "rounds": rounds}


def _check_manifest(d: Path, fp: dict) -> None:
    m = d / "manifest.json"
    d.mkdir(parents=True, exist_ok=True)
    if m.exists():
        old = json.loads(m.read_text(encoding="utf-8"))
        if old != json.loads(json.dumps(fp)):
            raise SystemExit(f"部品の作りの印が違う（古い部品を混ぜない）: {d}")
    else:
        m.write_text(json.dumps(fp, ensure_ascii=False, indent=1), encoding="utf-8")


def run(fs: str, target: str, variant: str, split: str = "time", conv: str = "convert_A", only_unit: str | None = None) -> Path:
    import xgboost as xgb
    from src.ops import data_ledger as DL
    t0 = time.time()
    tok = DL.guard(tool="src/research/model_relabel/mr_train.py:run", family="model_relabel", version=f"train-{split}-{fs}-{target}-{variant}-{conv}",
                   period=("2016-01-01", "2024-12-31"), purpose="reproduce", reads=("着順",), frozen_paths=[SPEC],
                   reason="拡張ラベルの前へ進める学習の教師（着順・着差・時計）を読むだけ。成績は数えない")
    X, meta, names = load_matrix(fs)
    lab = attach_labels(meta, conv)
    y, wt = target_and_weight(lab, target, variant)
    yr = lab["year"].to_numpy()
    sets, rounds = fs_configs(fs)
    if split == "time":
        units = [(str(t), yr < t, yr == t) for t in YEARS]
    else:
        f = pd.read_csv(FOLDS_CSV, dtype={"meeting": str}).set_index("meeting")
        mt = lab["meeting"].map(f["fold"]).to_numpy()
        assert not np.isnan(mt.astype(float)).any()
        units = []
        for j in range(5):
            cal = lab["meeting"].map(f[f"calib_{j}"]).to_numpy() == 1
            units.append((f"f{j}", (mt != j) & ~cal, (mt == j) | cal))     # 予測は外側の折と較正の側の両方
    todo = [u for u in units if u[0] == only_unit] if only_unit else units
    _check_manifest(_part_path(split, fs, target, variant, conv, 0, "x").parent, _fingerprint(lab, target, variant, sets, rounds))
    for bi, params in enumerate(sets):
        for unit, tr, te in todo:
            p = _part_path(split, fs, target, variant, conv, bi, unit)
            if p.exists():
                continue
            if blocked():
                tok.finish(status="error", error="開催日の日中で止めた")
                raise SystemExit("開催日の 07:30〜17:00 なので止める（部品は保存済み。続きから流せる）")
            m = xgb.train(params, xgb.DMatrix(X[tr], label=y[tr], weight=wt[tr]), rounds[-1])
            dte = xgb.DMatrix(X[te])
            P = np.stack([m.predict(dte, iteration_range=(0, r)) for r in rounds]).astype(np.float32)
            tmp = p.with_suffix(".tmp.npy")
            np.save(tmp, P)
            os.replace(tmp, p)
            print(f"  {split} {fs} {target} {variant} bag{bi} {unit}: 学習 {int(tr.sum())} → 予測 {int(te.sum())}（{round(time.time() - t0)} 秒）",
                  flush=True)
    dest = assemble(split, fs, target, variant, conv, lab, units, len(sets), rounds)
    tok.finish(result_path=dest)
    return dest


def assemble(split, fs, target, variant, conv, lab, units, n_bags, rounds) -> Path:
    """部品を束ねて、行ごとの予測（本数ごと・袋の平均）の表にする。"""
    if split == "meeting":
        return _assemble_meeting(fs, target, variant, conv, lab, units, n_bags, rounds)
    out = lab[["race_id", "horse_id", "year", "meeting"]].copy()
    P = np.full((len(rounds), len(lab)), np.nan, np.float32)
    done = True
    for unit, tr, te in units:
        parts = [_part_path(split, fs, target, variant, conv, bi, unit) for bi in range(n_bags)]
        if not all(p.exists() for p in parts):
            done = False
            continue
        P[:, te] = sum(np.load(p) for p in parts) / n_bags
    for k, r in enumerate(rounds):
        out[f"p_r{r}"] = P[k]
    dest = OUT / "pred" / split / f"{fs}_{target}_{variant}_{conv}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not done:
        dest = dest.with_suffix(".partial.parquet")      # 未完の表は評価に入れない（mr_eval は完了の名前だけを読む）
    out.to_parquet(dest, index=False)
    print("assembled", dest, "完了" if done else "途中", flush=True)
    return dest


def _assemble_meeting(fs, target, variant, conv, lab, units, n_bags, rounds) -> Path:
    """meeting の分けは、折 j ごとに（外側の予測・較正の側の予測）を縦に積んだ表にする（列 fold・role）。"""
    f = pd.read_csv(FOLDS_CSV, dtype={"meeting": str}).set_index("meeting")
    fold = lab["meeting"].map(f["fold"]).to_numpy()
    rows = []
    done = True
    for unit, tr, te in units:
        j = int(unit[1:])
        parts = [_part_path("meeting", fs, target, variant, conv, bi, unit) for bi in range(n_bags)]
        if not all(p.exists() for p in parts):
            done = False
            continue
        acc = sum(np.load(p) for p in parts) / n_bags
        sub = lab.loc[te, ["race_id", "horse_id", "year", "meeting"]].copy()
        sub["fold"] = j
        sub["role"] = np.where(fold[te] == j, "outer", "calib")
        for k, r in enumerate(rounds):
            sub[f"p_r{r}"] = acc[k]
        rows.append(sub)
    dest = OUT / "pred" / "meeting" / f"{fs}_{target}_{variant}_{conv}.parquet"
    if not done:
        dest = dest.with_suffix(".partial.parquet")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return dest
    pd.concat(rows, ignore_index=True).to_parquet(dest, index=False)
    print("assembled", dest, flush=True)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "labels"])
    ap.add_argument("--fs", choices=["v5bag", "v6e"], default="v5bag")
    ap.add_argument("--target", choices=["win", "top3", "notbig"], default="win")
    ap.add_argument("--variant", default="orig")
    ap.add_argument("--split", choices=["time", "meeting"], default="time")
    ap.add_argument("--conv", default="convert_A")
    ap.add_argument("--only-unit", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "labels":
        for k in ("convert_A", "convert_B_sensitivity"):
            LB.labels_dev(k).to_parquet(OUT / f"labels_{k}.parquet", index=False)
        print("labels saved")
        return
    run(a.fs, a.target, a.variant, a.split, a.conv, a.only_unit)


if __name__ == "__main__":
    main()
