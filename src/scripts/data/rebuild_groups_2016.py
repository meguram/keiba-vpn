"""親モデル・前段 S3 が読む群を、2016〜2025 年の土台（rebuild_base_2016 の出力）で作り直す（再学習 2026-11）。

    python -m src.scripts.data.rebuild_groups_2016 labels   # 前段 S3 のラベル（stage_labels）
    python -m src.scripts.data.rebuild_groups_2016 conn     # G2 関係者（馬主など）
    python -m src.scripts.data.rebuild_groups_2016 self     # G1 馬自身の履歴・状態
    python -m src.scripts.data.rebuild_groups_2016 field    # G3 メンバー構成・コース・環境
    python -m src.scripts.data.rebuild_groups_2016 ped      # G4 種牡馬 ID × 条件（PED_sire_cond）

- 出力は BASE（既定 /home/jovyan/work/retrain_out/base_2016）の下に、本番と同じ並び（factory/{conn,self,field,src}/…, oof/stage_labels.parquet）で置く。
  本番の data/local/modeling は書き換えない（各モジュールの出力先の定数をこの実行の中だけ差し替える）。
- 列は本番の群と同じ列名にそろえる（本番に無い列は落とし、本番にあって作れない列は欠損で足す）。
- 馬の静的プロファイルは fetch_horse_profile_2016 の出力（<BASE>/profile）を使う。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(os.environ.get("REBUILD_BASE") or "/home/jovyan/work/retrain_out/base_2016")
PROD = ROOT / "data/local/modeling/training_pipeline"
TABLES = ROOT / "data/page_reference/tables"
YEARS = list(range(2016, 2026))
KEY = ["race_id", "horse_id"]


def log(m: str) -> None:
    print(f"[{time.strftime('%T')}] {m}", flush=True)


def patch_years() -> None:
    from src.scripts.data import rebuild_base_2016 as rb

    rb._patch_years(YEARS)


def base_table() -> pd.DataFrame:
    from src.pipeline.models import ablation as ab

    df = ab.load_table(BASE / "features/train_table.parquet")
    df["race_id"], df["horse_id"] = df["race_id"].astype(str), df["horse_id"].astype(str)
    return df.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)


def keys() -> pd.DataFrame:
    import polars as pl

    return pl.read_parquet(BASE / "features/train_table.parquet", columns=KEY).to_pandas().astype(str)


def align_cols(g: pd.DataFrame, prod_file: Path, keep_keys: bool = True) -> tuple[pd.DataFrame, dict]:
    """本番の群と同じ列（と順）にそろえる。"""
    want = [c for c in pq.read_schema(prod_file).names if c not in KEY]
    have = [c for c in g.columns if c not in KEY]
    missing = [c for c in want if c not in have]
    extra = [c for c in have if c not in want]
    for c in missing:
        g[c] = np.nan
    cols = (KEY if keep_keys and all(k in g.columns for k in KEY) else []) + want
    return g[cols], {"cols": len(want), "missing_filled_nan": missing, "dropped_extra": extra}


def cmd_labels() -> None:
    import polars as pl

    patch_years()
    from src.pipeline.models.stage_models import stage_labels

    on_time = PROD / "features/on_time_repair.parquet"
    lab = stage_labels(BASE / "layer_a.parquet", pl.read_parquet(on_time) if on_time.exists() else None)
    (BASE / "oof").mkdir(parents=True, exist_ok=True)
    lab.to_parquet(BASE / "oof/stage_labels.parquet", index=False)
    log(f"stage_labels {lab.shape} 年別 {pd.to_datetime(lab['date']).dt.year.value_counts().sort_index().to_dict()}")


def cmd_conn() -> None:
    from src.pipeline.features import factory_conn as fc

    fc.PROFILE = BASE / "profile/horse_profile.parquet"
    out = BASE / "factory/conn/groups"
    out.mkdir(parents=True, exist_ok=True)
    h = fc.prepare(fc.load_history(TABLES, YEARS))
    log(f"履歴 {len(h)} 行 / 馬主あり {h['owner'].notna().mean() if 'owner' in h.columns else 0:.3f}")
    groups = fc.build_all(h)
    k = keys()
    meta = {}
    for g, cols in groups.items():
        prod = PROD / "factory/conn/groups" / f"{g}.parquet"
        if not prod.exists():
            continue
        F = pd.DataFrame({c: np.asarray(a, dtype="float32") for c, a in cols.items()})
        F[KEY[0]], F[KEY[1]] = h["race_id"].to_numpy(), h["horse_id"].to_numpy()
        m = k.merge(F.drop_duplicates(KEY), on=KEY, how="left")
        assert len(m) == len(k)
        m, info = align_cols(m, prod, keep_keys=False)      # 本番の conn 群はキー列なし（行の並び = 学習表）
        m.to_parquet(out / f"{g}.parquet", index=False)
        meta[g] = info
        log(f"conn {g}: {info['cols']} 列 欠損で足した {len(info['missing_filled_nan'])}")
    (BASE / "factory/conn/groups_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_self() -> None:
    from src.pipeline.features import factory_self as fs

    out = BASE / "factory/self"
    out.mkdir(parents=True, exist_ok=True)
    runs = fs.load_runs(TABLES, YEARS)
    log(f"runs {len(runs)}")
    at = pd.read_parquet(BASE / "profile/horse_attrs.parquet").drop_duplicates("horse_id").set_index("horse_id")
    k = keys()
    meta = {}
    for name in fs.GROUPS:
        prod = PROD / "factory/self" / f"{name}.parquet"
        if not prod.exists():
            continue
        g = fs.build_groups(runs, birthday=at["birthday"], attrs=at, only=[name]).get(name)
        if g is None:
            continue
        g = k.merge(g, on=KEY, how="left")
        g, info = align_cols(g, prod)
        g.to_parquet(out / f"{name}.parquet", index=False)
        meta[name] = info
        log(f"self {name}: {info['cols']} 列 欠損で足した {len(info['missing_filled_nan'])}")
    (out / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_field() -> None:
    from src.pipeline.features import factory_field as ff

    out = BASE / "factory/field"
    out.mkdir(parents=True, exist_ok=True)
    df = base_table()
    pace = ff.load_pace_table(TABLES, YEARS)
    prod_dir = Path(os.environ.get("REBUILD_FIELD_PROD") or ROOT / "data/local/modeling/training_pipeline_v4/factory_field")
    meta = {}
    for g in ff.GROUPS:
        prod = prod_dir / f"{g}.parquet"
        if not prod.exists():
            continue
        f = ff.build_group(g, df, **({"pace_table": pace} if g == "race_type" else {}))
        f = f.replace([np.inf, -np.inf], np.nan)
        f = pd.concat([df[KEY].reset_index(drop=True), f.reset_index(drop=True)], axis=1)
        f, info = align_cols(f, prod)
        f.to_parquet(out / f"{g}.parquet", index=False)
        meta[g] = info
        log(f"field {g}: {info['cols']} 列 欠損で足した {len(info['missing_filled_nan'])}")
    (out / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_ped() -> None:
    from src.pipeline.features import factory_ped as fp
    from src.pipeline.features import factory_src as fsrc

    out = BASE / "factory/src"
    out.mkdir(parents=True, exist_ok=True)
    fp.CACHE = out                       # 祖先の表のキャッシュ（本番の factory/src のキャッシュは 2020〜 の馬だけ）を新しく作る
    fsrc.YEARS = YEARS
    B = base_table()
    meta = {}
    for g in ("PED_sire_cond",):
        prod = PROD / "factory/src" / f"cand_{g}.parquet"
        f = fsrc.build_group(g, B, None)
        f = f.assign(race_id=B["race_id"].to_numpy(), horse_id=B["horse_id"].to_numpy())
        f, info = align_cols(f, prod)
        f.to_parquet(out / f"cand_{g}.parquet", index=False)
        meta[g] = info
        log(f"src {g}: {info['cols']} 列 欠損で足した {len(info['missing_filled_nan'])}")
    (out / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def install_setup() -> None:
    """tp27_eval.setup（本番の表を読む）を、新しい土台を読むものに差し替える。前段 S3 の列は <BASE>/oof/stage_features_expanding.parquet があれば付ける。"""
    patch_years()
    from src.pipeline.models import zoo
    from src.pipeline.models.time_split import yearly_expanding_folds
    from src.scripts.data import tp27_eval as t

    def setup():
        adopted = json.loads((PROD / "ablation_nospeed/adopted_features.json").read_text(encoding="utf-8"))
        base = list(adopted["b0_features"]) + list(adopted["features"])
        best = json.loads((PROD / "hpo/best_params.json").read_text(encoding="utf-8"))["best"]
        lp = zoo.tuned_params("lgb", best["lgb"]["params"])
        df = base_table()
        sf = BASE / "oof/stage_features_expanding.parquet"
        if sf.exists():
            st = pd.read_parquet(sf)[KEY + t.STAGE_COLS]
            st["race_id"], st["horse_id"] = st["race_id"].astype(str), st["horse_id"].astype(str)
            df = df.merge(st, on=KEY, how="left")
        else:
            for c in t.STAGE_COLS:
                df[c] = np.nan
        df = df.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)
        meta = df[KEY + ["finish_position", "year", "date"]].copy()
        return df, base, meta, list(yearly_expanding_folds(df)), lp

    t.setup = setup


def laps() -> pd.DataFrame:
    f = BASE / "laps/training_laps_2016.parquet"
    if not f.exists():
        a = pd.read_parquet(ROOT / "data/local/modeling/training_pipeline_v4/tp35/training_laps_fresh.parquet")
        b = pd.read_parquet(BASE / "laps/training_laps_add.parquet")
        m = pd.concat([a, b[[c for c in a.columns if c in b.columns]]], ignore_index=True)
        m.to_parquet(f, index=False)
        log(f"追い切り表 {len(a)} + {len(b)} = {len(m)} 本")
    return pd.read_parquet(f)


def cmd_tl() -> None:
    from src.pipeline.features import training_patterns as tp
    from src.pipeline.features import training_serving as ts

    install_setup()
    from src.scripts.data import tp27_eval as t

    df = t.setup()[0]
    L = laps()
    tt = tp.add_reference_z(tp.prepare(L))
    feat = tp.add_self_relative(tp.race_features(tt))
    RD = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    M = pd.DataFrame({"horse_id": df["horse_id"].astype(str).to_numpy(), "race_date": RD.to_numpy()}, index=df.index).merge(feat, on=["horse_id", "race_date"], how="left")
    M.index = df.index
    cols = ts.tl_a_columns(M.columns)
    new = pd.concat([df[KEY].reset_index(drop=True), M[cols].reset_index(drop=True)], axis=1)
    new, info = align_cols(new, ROOT / "data/local/modeling/training_pipeline_v4/tp35/cand_TL_A_v2.parquet")
    (BASE / "tp35").mkdir(parents=True, exist_ok=True)
    new.to_parquet(BASE / "tp35/cand_TL_A_v2.parquet", index=False)
    yr = pd.to_datetime(df["date"]).dt.year.to_numpy()
    log(f"TL: {info['cols']} 列 被覆（has_training）年別 {M['has_training'].notna().groupby(yr).mean().round(3).to_dict() if 'has_training' in M else ''}")


def cmd_cy() -> None:
    from src.pipeline.features import tp40_condition as C

    install_setup()
    from src.scripts.data import tp27_eval as t
    from src.scripts.data import tp40_eval as e40

    df = t.setup()[0]
    feats = C.build_features(e40._runs(df), laps())
    d = BASE / "tp40/groups"
    d.mkdir(parents=True, exist_ok=True)
    for g, f in feats.items():
        prod = ROOT / "data/local/modeling/training_pipeline_v4/tp40/groups" / f"{g}.parquet"
        if not prod.exists():
            continue
        f = df[KEY].merge(f.drop_duplicates(KEY), on=KEY, how="left")
        f, info = align_cols(f, prod)
        f.to_parquet(d / f"{g}.parquet", index=False)
        log(f"tp40 {g}: {info['cols']} 列 欠損で足した {len(info['missing_filled_nan'])}")


def cmd_best() -> None:
    """tp38 の PX_cell_sd・PX_emb（静的な血統表から作り直す）と PX_ridge_als（窓は 2017 年から）。"""
    import argparse

    from src.pipeline.features import factory_ped as fp

    install_setup()
    from src.scripts.data import tp38_eval as E
    from src.scripts.data import tp38_ridge_run as RR

    E.OUT = BASE / "tp38"
    E.STATIC = E.OUT / "static"
    fp.CACHE = BASE / "factory/src"
    E._wait = lambda *a, **k: None
    if not (E.STATIC / "emb.npy").exists():
        E.cmd_static(argparse.Namespace())
    E.cmd_build(argparse.Namespace(only="PX_cell_sd,PX_emb"))
    w = RR.windows
    RR.windows = lambda start="2017-01-01", end="2026-01-01": w(start, end)
    RR.main()
    for g in ("PX_cell_sd", "PX_emb", "PX_ridge_als"):
        f = E.OUT / "groups" / f"{g}.parquet"
        x, info = align_cols(pd.read_parquet(f), Path("/home/jovyan/work/tp38_out/groups") / f"{g}.parquet")
        x.to_parquet(f, index=False)
        log(f"tp38 {g}: {info['cols']} 列 欠損で足した {len(info['missing_filled_nan'])} 余分 {len(info['dropped_extra'])}")


def cmd_rb() -> None:
    """tp65 の RB（めぐ指数・netkeiba 指数の履歴）。2016〜2019 年のめぐ指数は recompute で作り直したもの（<retrain_out>/megu_2016、
    2020 年は本番と完全一致を確認済み）。netkeiba 指数は出走前ページ（race_index）と馬柱がある年だけ（2016〜2017 年は取得元に無い）。"""
    import argparse

    from src.scripts.data import tp65_feat as F
    from src.scripts.data import tp65_prep as P

    megu = Path(os.environ.get("REBUILD_MEGU") or "/home/jovyan/work/retrain_out/megu_2016")
    orig = P._read

    def _read(year: int, name: str, cols: list[str]):
        if name == "megu_index_flat" and year < 2020 and (megu / str(year) / f"{name}.parquet").exists():
            d = pd.read_parquet(megu / str(year) / f"{name}.parquet")
            for c in cols:
                if c not in d.columns:
                    d[c] = np.nan
            return d[cols]
        return orig(year, name, cols)

    P._read = _read
    out = BASE / "tp65"
    out.mkdir(parents=True, exist_ok=True)
    runs, rep = P.build(list(range(2016, 2027)))
    runs.to_parquet(out / "runs.parquet", index=False)
    (out / "prep_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"tp65 runs {len(runs)} めぐ被覆 {rep.get('megu_v3a_cover')} nk 被覆 {rep.get('nk_cover', rep.get('both_cover'))}")
    F.OUT = out
    F.cmd_build(argparse.Namespace())
    x, info = align_cols(pd.read_parquet(out / "rob.parquet"), Path("/home/jovyan/work/tp65_out/rob.parquet"))
    x.to_parquet(out / "rob.parquet", index=False)
    log(f"RB: {info['cols']} 列 欠損で足した {len(info['missing_filled_nan'])}")


def cmd_aux() -> None:
    """tp25c の補助ヘッド（mtl_C、目的 9 本）の OOF → tp25e の aux_B（mtl_* + レース内の順位・差 = 27 列）。
    本番と同じく年ごとに『その年の 10 日前まで』で学習（2017〜2025 年）。最初の 2016 年は年内を日付順の 4 ブロックに分けて他のブロックで学習
    （本番の 2020 年の扱いと同じ。2016 年は学習側の古い行にしか使わない）。本番の aux_B は 2025 年が欠損だったが、ここでは 2024 年までで学習した値を入れる。"""
    import torch

    from src.pipeline.models.tp25c_mtl import mtl_fit_predict
    from src.scripts.data import tp25c_eval as c
    from src.scripts.data import tp25e_eval as e

    patch_years()
    for name in ("ablation_nospeed", "smartrc_ten1f"):
        link = BASE / name
        if not link.exists():
            link.symlink_to(PROD / name)
    c.FIX, c.TP, c.SX, c.OUT = BASE, BASE, BASE / "smartrc_ten1f", BASE / "tp25c"
    c.OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(c.THREADS)
    df, base, _sx = c.load_frame()
    v = c.VARIANTS["mtl_C"]
    X, A = df[base], df[v["aux"]].to_numpy(dtype="float64")
    race, fin, dates, years = df["race_id"].to_numpy(), df["finish_position"].to_numpy(), pd.to_datetime(df["date"]), df["year"].to_numpy()
    Y = {"win": (fin == 1).astype(int), "top2": (fin <= 2).astype(int), "top3": (fin <= 3).astype(int)}
    k = len(v["aux"])
    cache = c.OUT / "oof_aux_mtl_C.npy"
    oof = np.load(cache) if cache.exists() else np.full((len(df), k), np.nan, dtype="float32")
    gap = pd.Timedelta(days=10)
    first = int(min(years))
    for Yr in range(first + 1, 2026):
        va = np.flatnonzero(years == Yr)
        if np.isfinite(oof[va, 0]).all():
            continue
        tr = np.flatnonzero((years >= first) & (dates < pd.Timestamp(f"{Yr}-01-01") - gap).to_numpy())
        acc = np.zeros((len(va), k))
        for seed in c.SEEDS:
            o = mtl_fit_predict(X.iloc[tr], {q: x[tr] for q, x in Y.items()}, A[tr], race[tr], X.iloc[va], race[va], seed=seed, aux_w=v.get("w", 0.5), interact=v.get("interact", False))
            acc += o["aux"] / len(c.SEEDS)
        oof[va] = acc
        np.save(cache, oof)
        log(f"oof aux {Yr} 完了（学習 {len(tr)} 行）")
    idx = np.flatnonzero(years == first)
    if not np.isfinite(oof[idx, 0]).all():
        order = idx[np.argsort(dates.to_numpy()[idx], kind="stable")]
        for i, vb in enumerate(np.array_split(order, 4)):
            lo, hi = dates.iloc[vb].min() - gap, dates.iloc[vb].max() + gap
            tr = idx[~((dates.iloc[idx] >= lo) & (dates.iloc[idx] <= hi)).to_numpy()]
            acc = np.zeros((len(vb), k))
            for seed in c.SEEDS:
                o = mtl_fit_predict(X.iloc[tr], {q: x[tr] for q, x in Y.items()}, A[tr], race[tr], X.iloc[vb], race[vb], seed=seed, aux_w=0.5)
                acc += o["aux"] / len(c.SEEDS)
            oof[vb] = acc
            log(f"kfold {first} block {i + 1}/4 完了（学習 {len(tr)} 行）")
        np.save(cache, oof)
    names = ["mtl_" + x for x in c.AUX_C]
    t = pd.DataFrame(oof, columns=names, index=df.index)
    t = pd.concat([t, e.rel_cols(pd.concat([df[["race_id"]], t], axis=1), names)], axis=1)
    t.insert(0, "horse_id", df["horse_id"].to_numpy())
    t.insert(0, "race_id", df["race_id"].to_numpy())
    t, info = align_cols(t, ROOT / "data/local/modeling/training_pipeline_v4/tp25e/aux_B.parquet")
    (BASE / "tp25e").mkdir(parents=True, exist_ok=True)
    t.to_parquet(BASE / "tp25e/aux_B.parquet", index=False)
    log(f"aux_B {t.shape} 被覆 年別 {pd.Series(np.isfinite(oof[:, 0])).groupby(years).mean().round(3).to_dict()}")


def redirect() -> None:
    """tp27 / tp40 / tp07 / v5_select / v6 系の読み先（本番の表の定数）を、この実行の中だけ BASE に向ける。
    S3・オッズ予測・親モデルの評価を新しい表で回すときに、各試行のスクリプトの先頭で呼ぶ。"""
    install_setup()
    from src.scripts.data import tp27_eval as t

    t.FIXED, t.FAC, t.FAC_FIELD, t.AUX_DIR = BASE, BASE / "factory", BASE / "factory/field", BASE / "tp25e"

    def tl_frame(df):
        TL = pd.read_parquet(BASE / "tp35/cand_TL_A_v2.parquet")
        TL["race_id"], TL["horse_id"] = TL["race_id"].astype(str), TL["horse_id"].astype(str)
        m = df[KEY].merge(TL, on=KEY, how="left").drop(columns=KEY)
        m.index = df.index
        return m.add_prefix("tl_")

    t.tl_frame = tl_frame
    from src.scripts.data import tp07_base as tb
    from src.scripts.data import tp40_eval as e40

    for name in ("features/fit_params.json",):
        link = BASE / name
        if not link.exists():
            link.symlink_to(PROD / name)
    tb.V4, tb.OLD = BASE, BASE
    e40.V, e40.OUT = BASE, BASE / "tp40"
    try:
        from src.scripts.data import v5_select_eval as sel

        sel.TP38_OUT = BASE / "tp38"
    except ImportError:
        pass
    try:
        from src.scripts.data import v6_eval as v6e

        v6e.EXTERNAL["RB"] = (BASE / "tp65/rob.parquet", None)
    except ImportError:
        pass


def cmd_frame() -> None:
    """親モデル・オッズ予測の土台の表（tp31 の v4_frame）を新しい群から組み立てる。列は本番の v4_frame と同じ。"""
    redirect()
    from src.scripts.data import tp27_eval as t

    df, base, meta, folds, lp = t.setup()
    raw = t._raw_keys()
    X = t._v4_frame(df, base, raw, t.v4_groups("core"))
    X = X.loc[:, ~X.columns.duplicated()]
    prod = ROOT / "data/local/modeling/training_pipeline_v4/tp31/v4_frame.parquet"
    want = pq.read_schema(prod).names
    miss = [c for c in want if c not in X.columns]
    for c in miss:
        X[c] = np.nan
    X = X[want]
    cat = [c for c in X.columns if str(X[c].dtype) in ("category", "object", "string")]
    num = X.drop(columns=cat)
    out = pd.concat([num.astype("float32"), X[cat]], axis=1)[want] if cat else num.astype("float32")
    (BASE / "tp31").mkdir(parents=True, exist_ok=True)
    out.to_parquet(BASE / "tp31/v4_frame.parquet", index=False)
    yr = pd.to_datetime(df["date"]).dt.year
    nn = out.select_dtypes(include=[np.number]).notna().groupby(yr.to_numpy()).mean().mean(axis=1).round(3).to_dict()
    log(f"v4_frame {out.shape}（欠損で足した列 {len(miss)}: {miss[:10]}）年別の平均の非欠損率 {nn}")
    # 2020〜2025 年の行で本番の v4_frame と比べる（行の順は学習表の並び）
    import polars as pl

    from src.pipeline.models import ablation as ab

    pk = ab.load_table(ROOT / "data/local/modeling/training_pipeline_v4/features/train_table.parquet")[KEY + ["date", "horse_number"]]
    pk["race_id"], pk["horse_id"] = pk["race_id"].astype(str), pk["horse_id"].astype(str)
    pk = pk.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)      # 本番の v4_frame の行の並び（t.setup と同じ）
    old = pd.read_parquet(prod)
    old.insert(0, "horse_id", pk["horse_id"].to_numpy()); old.insert(0, "race_id", pk["race_id"].to_numpy())
    new = pd.concat([df[KEY].reset_index(drop=True), out.reset_index(drop=True)], axis=1)
    j = new.merge(old, on=KEY, suffixes=("", "_o"))
    rows = []
    for c in want:
        if c + "_o" in j.columns and pd.api.types.is_numeric_dtype(j[c]) and pd.api.types.is_numeric_dtype(j[c + "_o"]):
            a, b = j[c].to_numpy(float), j[c + "_o"].to_numpy(float)
            same = np.isclose(a, b, rtol=1e-4, atol=1e-6) | (np.isnan(a) & np.isnan(b))
            ok = np.isfinite(a) & np.isfinite(b)
            rows.append({"col": c, "changed": round(float(1 - same.mean()), 4),
                         "corr": round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 4) if ok.sum() > 10 and a[ok].std() > 0 and b[ok].std() > 0 else None,
                         "null_new": round(float(np.isnan(a).mean()), 4), "null_old": round(float(np.isnan(b).mean()), 4)})
    d = pd.DataFrame(rows).sort_values("changed", ascending=False)
    d.to_csv(BASE / "tp31/diff_vs_prod_2020_2025.csv", index=False)
    log(f"本番と比べた行 {len(j)}、値が変わった列 {(d['changed'] > 0.001).sum()} / {len(d)}")


if __name__ == "__main__":
    {"frame": cmd_frame, "aux": cmd_aux, "rb": cmd_rb, "labels": cmd_labels, "conn": cmd_conn, "self": cmd_self, "field": cmd_field, "ped": cmd_ped,
     "tl": cmd_tl, "cy": cmd_cy, "best": cmd_best}[sys.argv[1]]()
