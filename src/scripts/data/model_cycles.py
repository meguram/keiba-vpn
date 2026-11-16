"""親モデルの改善サイクル（2026-11-16〜、統括役の依頼「20 サイクル回して一番良いものを候補に」）の道具。

    python -m src.scripts.data.model_cycles cache                       # v6e16 の xgb_extras_ped と同じ入力（X・meta・fold）を 1 回だけ保存
    python -m src.scripts.data.model_cycles member NAME --spec '{...}' [--seeds 1]   # 勝ちの二値（XGBoost・GPU）の OOF（valid 2022〜2024）
    python -m src.scripts.data.model_cycles ens NAME                    # v6e16 の 7 メンバーの xgb_extras_ped の勝率を NAME に差し替えた LOYO アンサンブル
    python -m src.scripts.data.model_cycles score NAME [--ref REF]      # 勝ち馬の対数損失（レース単位）・◎の勝率・複勝率・回収率 4 種と、対の差の区間

- 入力は rank_bakeoff（RANK_BASE=2016）と同じ作り（2016〜2025 年の土台 retrain_out/base_2016、本番は書き換えない）。
- fold は valid 2022〜2024（年ごとに広げる。学習は 2016 年から）。2026 年は学習・選択に使わない。2025 年（test）は最後に 1 回だけ（ここでは扱わない）。
- spec（サイクルの変更）: drop（列名）・drop_prefix（接頭辞）・params（XGBoost の設定の上書き）・halflife_years・min_train_year
  ・blank_newcomer（新馬戦の行だけ空にする列。推論で空になる列を学習でも空にそろえる）。
- 主の指標は、アンサンブルの出力の「勝ち馬の対数損失」（レースごとに −log p(勝ち馬)、p はレース内で正規化。v6e_2026_gap.race_ll と同じ定義）。
- 出力は CYC（既定 /home/jovyan/work/retrain_out/cycles16）の oof/・ens/・score/。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CYC = Path(os.environ.get("CYCLES_OUT") or "/home/jovyan/work/retrain_out/cycles16")
BASE = Path(os.environ.get("REBUILD_BASE") or "/home/jovyan/work/retrain_out/base_2016")
ROOT = Path(__file__).resolve().parents[3]
PROD_ENS = ROOT / "data/local/modeling/training_pipeline_v4/tp27_v6e/ens/v6.parquet"   # 本番 v6e の LOYO（valid 2022〜2024）
CAND_ENS = BASE / "tp27_v6e/ens/v6.parquet"                                          # 候補 v6e16 の LOYO
V6E_MEMBERS = ["lgb_extras_ped", "xgb_extras_ped", "lgb_extras", "lgb_softmax", "linear", "mtl_nn_extras_ped", "mtl_nn_s"]
KEY = ["race_id", "horse_id"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cmd_cache(_a) -> None:
    os.environ["RANK_BASE"] = "2016"
    from src.scripts.data import rank_bakeoff as rb

    t0 = time.time()
    df, meta, folds, params, X = rb._inputs("xgb_extras_ped", nn=False)
    CYC.mkdir(parents=True, exist_ok=True)
    X = X.reset_index(drop=True)
    X.to_parquet(CYC / "X_xgb.parquet", index=False)
    m = meta.reset_index(drop=True).copy()
    extra = [c for c in ("date", "horse_number", "field_size", "race_class", "is_newcomer") if c in df.columns and c not in m.columns]
    for c in extra:
        m[c] = df[c].to_numpy()
    m.to_parquet(CYC / "meta.parquet", index=False)
    info = {"params": params, "rows": len(X), "cols": X.shape[1], "seconds": round(time.time() - t0),
            "folds": [{"train_n": len(f.train_index), "valid_n": len(f.valid_index),
                       "train_years": sorted({int(y) for y in m["year"].iloc[f.train_index].unique()}),
                       "valid_years": sorted({int(y) for y in m["year"].iloc[f.valid_index].unique()})} for f in folds]}
    (CYC / "cache_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log(f"cache: {X.shape} {info['seconds']}s folds={[(f['train_years'][0], f['valid_years']) for f in info['folds']]}")


# ---------------------------------------------------------------- 入力
def load_cache() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    X = pd.read_parquet(CYC / "X_xgb.parquet")
    meta = pd.read_parquet(CYC / "meta.parquet")
    meta["race_id"], meta["horse_id"] = meta["race_id"].astype(str), meta["horse_id"].astype(str)
    info = json.loads((CYC / "cache_info.json").read_text(encoding="utf-8"))
    return X, meta, info


def newcomer_mask(meta: pd.DataFrame) -> np.ndarray:
    """新馬戦の行（race_class に「新馬」）。"""
    return meta["race_class"].astype(str).str.contains("新馬").to_numpy()


def apply_spec(X: pd.DataFrame, meta: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """サイクルの変更のうち、入力の列に関わるもの（落とす・新馬戦で空にする）。推論では束の feature_cols と前処理で同じことをする。"""
    drop = set(spec.get("drop", []))
    for p in spec.get("drop_prefix", []):
        drop |= {c for c in X.columns if c.startswith(p)}
    X = X.drop(columns=[c for c in X.columns if c in drop])
    bn = [c for c in spec.get("blank_newcomer", []) if c in X.columns]
    if bn:
        X = X.copy()
        nm = newcomer_mask(meta)
        for c in bn:
            if str(X[c].dtype) == "category" and "(なし)" in list(X[c].cat.categories):
                X.loc[nm, c] = "(なし)"          # カテゴリの欠けは学習の表・束の _cast と同じく「(なし)」
            else:
                X.loc[nm, c] = np.nan
    return X


GAP_DAYS = 7 + 3            # time_split.yearly_expanding_folds の purge 7 日 + embargo 3 日


def folds_from_meta(meta: pd.DataFrame, valid_years=(2022, 2023, 2024), first_year: int = 2016) -> list[tuple[np.ndarray, np.ndarray]]:
    """年ごとに広げる fold（time_split.yearly_expanding_folds と同じ: 学習は first_year から、検証の始まりの 10 日前より前。検証は 1 暦年）。"""
    y = meta["race_id"].astype(str).str[:4].astype(int).to_numpy()
    dates = pd.to_datetime(meta["date"]).to_numpy()
    out = []
    for vy in valid_years:
        cut = np.datetime64(pd.Timestamp(f"{vy}-01-01") - pd.Timedelta(days=GAP_DAYS))
        tr = np.flatnonzero((y >= first_year) & (y < vy) & (dates < cut))
        va = np.flatnonzero(y == vy)
        out.append((tr, va))
    return out


def norm_race(race: np.ndarray, p: np.ndarray) -> np.ndarray:
    s = pd.Series(p).groupby(race).transform("sum").to_numpy()
    return p / s


# ---------------------------------------------------------------- メンバー（勝ちの二値・XGBoost）
def fit_win(X_tr, y_tr, X_va, params: dict, seed: int, weight=None, objective: str = "binary") -> np.ndarray:
    import xgboost as xgb

    p = dict(params)
    common = dict(n_estimators=p.get("n_estimators", 400), learning_rate=p.get("learning_rate", 0.05), max_depth=p.get("max_depth", 6),
                  subsample=p.get("subsample", 0.8), colsample_bytree=p.get("colsample_bytree", 0.7), reg_lambda=p.get("reg_lambda", 5.0),
                  min_child_weight=p.get("min_child_weight", 20), tree_method="hist", device=p.get("device", "cuda"),
                  enable_categorical=True, random_state=seed, n_jobs=4, verbosity=0)
    for k in ("max_bin", "gamma", "reg_alpha", "colsample_bylevel", "colsample_bynode", "max_cat_to_onehot"):
        if k in p:
            common[k] = p[k]
    m = xgb.XGBClassifier(**common)
    m.fit(X_tr, y_tr, sample_weight=weight)
    return m.predict_proba(X_va)[:, 1]


def cmd_member(a) -> None:
    spec = json.loads(a.spec) if a.spec else {}
    out = CYC / "oof" / f"{a.name}.parquet"
    if out.exists() and not a.force:
        log(f"{a.name} 既存（--force で作り直す）")
        return
    X, meta, info = load_cache()
    X = apply_spec(X, meta, spec)
    params = {**info["params"], **spec.get("params", {})}
    race = meta["race_id"].to_numpy()
    year = meta["year"].to_numpy()
    fin = meta["finish_position"].to_numpy().astype(float)
    win = (fin == 1).astype(int)
    pred = np.full(len(meta), np.nan)
    t0 = time.time()
    for tr, va in folds_from_meta(meta, first_year=spec.get("min_train_year", 2016)):
        w = None
        if spec.get("halflife_years"):
            age = (year[va].min() - 1) - year[tr]
            w = 0.5 ** (np.clip(age, 0, None) / float(spec["halflife_years"]))
        acc = np.zeros(len(va))
        for s in range(a.seeds):
            acc += norm_race(race[va], fit_win(X.iloc[tr], win[tr], X.iloc[va], params, seed=s, weight=w))
        pred[va] = acc / a.seeds
        log(f"{a.name} fold {int(year[va][0])} 済み（{time.time() - t0:.0f}s）")
    ok = np.isfinite(pred)
    o = meta.loc[ok, KEY + ["year"]].copy().reset_index(drop=True)
    o["p_win"] = pred[ok]
    out.parent.mkdir(parents=True, exist_ok=True)
    o.to_parquet(out, index=False)
    (out.with_suffix(".json")).write_text(json.dumps({"name": a.name, "spec": spec, "seeds": a.seeds, "cols": int(X.shape[1]),
                                                      "seconds": round(time.time() - t0)}, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"{a.name}: cols={X.shape[1]} {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- アンサンブル（xgb の勝率を差し替え）
def load_member_oofs(replace: dict[str, Path] | None = None) -> pd.DataFrame:
    keys = pd.read_parquet(BASE / "tp_rank/keys.parquet")
    keys["race_id"], keys["horse_id"] = keys["race_id"].astype(str), keys["horse_id"].astype(str)
    base = None
    for m in V6E_MEMBERS:
        path = (replace or {}).get(m, BASE / "tp27_v6e/oof" / f"{m}__v4.parquet")
        o = pd.read_parquet(path)
        o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
        cols = [c for c in ("p_win", "p_top2", "p_top3") if c in o.columns]
        o = o[KEY + cols].rename(columns={c: f"{m}__{c[2:]}" for c in cols})
        base = o if base is None else base.merge(o, on=KEY, how="inner")
    base = base.merge(keys[KEY + ["year", "finish_position", "horse_number"]], on=KEY, how="inner")
    return base


def ens_win(base: pd.DataFrame, members: list[str]) -> tuple[np.ndarray, dict]:
    from src.pipeline.models import ensemble as en

    race, years = base["race_id"].to_numpy(), base["year"].to_numpy()
    y = (base["finish_position"].to_numpy() == 1).astype(float)
    P = np.clip(np.stack([base[f"{m}__win"].to_numpy() for m in members], axis=1), 1e-6, 1 - 1e-6)
    llv, pred = en.loyo_eval("weighted_logit", P, years, race, y, 1.0)
    st = en.fit_method("weighted_logit", P, en.logit(P), y)
    return pred, {"loyo_row_ll": float(llv), "weights": dict(zip(members, [round(float(x), 4) for x in st]))}


def cmd_ens(a) -> None:
    replace = {"xgb_extras_ped": CYC / "oof" / f"{a.name}.parquet"} if a.name != "base" else {}
    for kv in a.replace or []:
        k, v = kv.split("=")
        replace[k] = CYC / "oof" / f"{v}.parquet"
    base = load_member_oofs(replace)
    members = list(V6E_MEMBERS) + [n for n in (a.add or [])]
    for n in a.add or []:
        o = pd.read_parquet(CYC / "oof" / f"{n}.parquet")[KEY + ["p_win"]]
        o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
        base = base.merge(o.rename(columns={"p_win": f"{n}__win"}), on=KEY, how="inner")
    pred, info = ens_win(base, members)
    out = base[KEY + ["year"]].copy()
    out["p_win"] = pred
    d = CYC / "ens"
    d.mkdir(parents=True, exist_ok=True)
    tag = a.tag or a.name
    out.to_parquet(d / f"{tag}.parquet", index=False)
    (d / f"{tag}.json").write_text(json.dumps({**info, "replace": {k: str(v) for k, v in replace.items()}, "add": a.add or []},
                                              ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"ens {tag}: 行 {len(out)} 行単位 LL {info['loyo_row_ll']:.5f} 重み {info['weights']}")


# ---------------------------------------------------------------- 評価
def race_metrics(pred: pd.DataFrame, keys: pd.DataFrame, pay: dict) -> pd.DataFrame:
    """レースごと: 勝ち馬の対数損失（勝ち馬がちょうど 1 頭。3 頭未満のレースは race_table の側で除く）と、◎の勝率・複勝率・回収率 4 種（rank_bakeoff_eval と同じ作り）。"""
    from src.scripts.data import rank_bakeoff_eval as rbe

    d = pred[KEY + ["p_win"]].merge(keys, on=KEY, how="inner")
    d["p"] = d["p_win"] / d.groupby("race_id")["p_win"].transform("sum")
    w = d[d["finish_position"] == 1]
    nwin = w.groupby("race_id").size()
    w = w[w["race_id"].map(nwin) == 1]
    ll = pd.DataFrame({"race_id": w["race_id"].to_numpy(), "ll": -np.log(w["p"].clip(1e-6)).to_numpy()})
    t = rbe.race_table(pred[KEY + ["p_win"]], keys, pay)
    return t.merge(ll, on="race_id", how="left")


def _keys() -> pd.DataFrame:
    k = pd.read_parquet(BASE / "tp_rank/keys.parquet")
    k["race_id"], k["horse_id"] = k["race_id"].astype(str), k["horse_id"].astype(str)
    meta = pd.read_parquet(CYC / "meta.parquet", columns=KEY + ["race_class", "date"])
    meta["race_id"], meta["horse_id"] = meta["race_id"].astype(str), meta["horse_id"].astype(str)
    return k.merge(meta, on=KEY, how="left")


def paired_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"))
    bs = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def read_pred(name: str) -> pd.DataFrame:
    p = {"prod_v6e": PROD_ENS, "cand_v6e16": CAND_ENS}.get(name)
    if p is None:
        p = CYC / "ens" / f"{name}.parquet"
        if not p.exists():
            p = CYC / "oof" / f"{name}.parquet"
    d = pd.read_parquet(p)
    d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    return d


def cmd_score(a) -> None:
    from src.scripts.data import rank_bakeoff_eval as rbe

    keys = _keys()
    pay = rbe.load_payoff()
    names = [a.name] + [r for r in a.ref.split(",") if r]
    races = None
    T = {}
    for n in names:
        T[n] = race_metrics(read_pred(n), keys, pay)
        races = set(T[n]["race_id"]) if races is None else races & set(T[n]["race_id"])
    rc = keys.drop_duplicates("race_id").set_index("race_id")
    res = {"races": len(races), "metrics": {}, "diff": {}}
    for n in names:
        t = T[n][T[n]["race_id"].isin(races)].copy()
        T[n] = t
        s = rbe.summarize(t)
        s["ll"] = float(t["ll"].mean())
        res["metrics"][n] = s
    a0 = T[a.name].set_index("race_id")
    for r in names[1:]:
        b0 = T[r].set_index("race_id").reindex(a0.index)
        dd = (a0["ll"] - b0["ll"]).to_numpy()
        e = {"ll": paired_ci(dd)}
        yr = a0.index.str[:4]
        for y in sorted(set(yr)):
            e[f"ll_{y}"] = paired_ci(dd[yr == y])
        cls = rc.reindex(a0.index)["race_class"].astype(str)
        e["ll_newcomer"] = paired_ci(dd[cls.str.contains("新馬").to_numpy()])
        e["ll_2yo"] = paired_ci(dd[cls.str.contains("２歳").to_numpy()])
        bd = rbe.boot_diff(T[a.name], T[r], n_boot=1000)
        e.update({k: v for k, v in bd.items() if k != "n"})
        res["diff"][f"{a.name} - {r}"] = e
    d = CYC / "score"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{a.name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for n, s in res["metrics"].items():
        log(f"{n}: LL {s['ll']:.5f} ◎勝 {s['top1_win']:.4f} ◎複 {s['top1_place']:.4f} 単 {s['roi_tansho']:.3f} 複 {s['roi_fukusho']:.3f} 連 {s['roi_umaren']:.3f} ワ {s['roi_wide']:.3f}")
    for k, e in res["diff"].items():
        log(f"{k}: LL {e['ll'][0]:+.5f}（{e['ll'][1]:+.5f}〜{e['ll'][2]:+.5f}） " + " ".join(f"{y}:{e[y][0]:+.4f}" for y in e if y.startswith("ll_")))


# ---------------------------------------------------------------- 学習と推論の入力のずれ
INF_X = Path(os.environ.get("CYCLES_INF_X") or "/home/jovyan/work/retrain_out/attrs_fix_check/full2_cand/inf/X")
TABLES = ROOT / "data/page_reference/tables"


def newcomer_races_2026() -> set[str]:
    """2026 年の新馬戦の race_id（race_class か、空ならレース名に「新馬」）。
    学習の表は race_class だけで判定する（学習の表で race_class が空なのは障害戦だけ。2026 年の結果の表は race_class の空が多く、レース名で補う）。"""
    r = pd.read_parquet(TABLES / "2026/race_result_flat.parquet", columns=["race_id", "race_class", "race_name"]).drop_duplicates("race_id")
    s = r["race_class"].fillna("").astype(str) + " " + r["race_name"].fillna("").astype(str)
    return set(r.loc[s.str.contains("新馬"), "race_id"].astype(str))


def load_inf_x(limit_days: int | None = None) -> pd.DataFrame:
    days = sorted(p for p in INF_X.glob("2026-*") if p.is_dir())
    if limit_days:
        days = days[-limit_days:]
    fs = [f for d in days for f in sorted(d.glob("*.parquet"))]
    if not fs:
        raise SystemExit(f"推論の入力の書き出しが無い: {INF_X}")
    x = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    x["race_id"], x["horse_id"] = x["race_id"].astype(str), x["horse_id"].astype(str)
    x.attrs["days"] = len(days)
    return x


def parity_table(spec: dict, train_year: int = 2025) -> tuple[pd.DataFrame, dict]:
    """モデルが使う列ごとに、学習（train_year の行）と推論（2026 年の書き出し）の欠けの割合を、新馬戦／それ以外で並べる。
    学習側にはサイクルの変更（apply_spec）を当てる。推論側には、推論で同じ処理（新馬戦で空にする）を当てた後の値を出す。"""
    X, meta, _ = load_cache()
    sel = meta["year"].to_numpy() == train_year
    Xt = apply_spec(X.loc[sel].reset_index(drop=True), meta.loc[sel].reset_index(drop=True), spec)
    nt = newcomer_mask(meta.loc[sel].reset_index(drop=True))
    xi = load_inf_x()
    ni = xi["race_id"].isin(newcomer_races_2026()).to_numpy()
    cols = list(Xt.columns)
    meta_i = pd.DataFrame({"race_class": np.where(ni, "新馬", "")})
    xi_m = apply_spec(cast_like(xi, X), meta_i, spec)[cols]     # 学習に無い水準・数値にならない値は、推論と同じく欠けにする
    tab = pd.DataFrame({
        "train_new": missing_rate(Xt.loc[nt]), "inf_new": missing_rate(xi_m.loc[ni]) if ni.any() else np.nan,
        "train_oth": missing_rate(Xt.loc[~nt]), "inf_oth": missing_rate(xi_m.loc[~ni]),
    })
    tab["gap_new"] = (tab["inf_new"] - tab["train_new"]).abs()
    tab["gap_oth"] = (tab["inf_oth"] - tab["train_oth"]).abs()
    info = {"train_rows": len(Xt), "train_new_rows": int(nt.sum()), "inf_rows": len(xi), "inf_new_rows": int(ni.sum()),
            "inf_races": int(xi["race_id"].nunique()), "inf_days": int(xi.attrs.get("days", 0))}
    return tab, info


def missing_rate(X: pd.DataFrame) -> pd.Series:
    """欠けの割合。カテゴリの「(なし)」（学習の表で欠損を埋めた値）も欠けとして数える。"""
    out = {}
    for c in X.columns:
        s = X[c]
        m = s.isna()
        if str(s.dtype) in ("category", "object"):
            m = m | (s.astype(str) == "(なし)")
        out[c] = float(m.mean()) if len(s) else np.nan
    return pd.Series(out)


# ---------------------------------------------------------------- 2026 年（推論の経路の入力）での参考
CAND_BUNDLE = BASE / "final_v6e16/bundle_v6_2025"


def inf_members(xi: pd.DataFrame) -> pd.DataFrame:
    """候補の束の 7 メンバーの勝率（推論の入力の書き出しに当てる）。1 回作って保存し、使い回す。"""
    from src.pipeline.models.ensemble_bundle import EnsembleBundle

    p = CYC / "inf2026" / "members.parquet"
    src = {"bundle": str(CAND_BUNDLE), "inf_x": str(INF_X)}
    pj = p.with_suffix(".json")
    if p.exists() and pj.exists() and json.loads(pj.read_text(encoding="utf-8")) == src:
        m = pd.read_parquet(p)
        if set(zip(m["race_id"], m["horse_id"])) >= set(zip(xi["race_id"], xi["horse_id"])):
            return m
    b = EnsembleBundle.load(CAND_BUNDLE)
    xi = xi.sort_values(["race_id", "horse_id"]).reset_index(drop=True)
    race = xi["race_id"].to_numpy()
    out = xi[KEY].copy()
    for mem in b.members:
        out[f"{mem.name}__win"] = b.predict_member(mem, xi, race)["win"]
        log(f"inf2026 メンバー {mem.name} 済み")
    out["bundle_p_win"] = b.predict(xi[sorted(set().union(*[m.feature_cols for m in b.members]))], race)["p_win"].to_numpy()
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(p, index=False)
    pj.write_text(json.dumps(src, ensure_ascii=False), encoding="utf-8")
    return out


def cast_like(xi: pd.DataFrame, Xtr: pd.DataFrame) -> pd.DataFrame:
    """推論の入力を、学習の表と同じ列・型（カテゴリは同じ水準の並び、欠損は「(なし)」）にそろえる。"""
    out = xi.reindex(columns=Xtr.columns).copy()
    for c in Xtr.columns:
        if str(Xtr[c].dtype) == "category":
            cats = list(Xtr[c].cat.categories)
            s = out[c].astype(str).where(out[c].notna())
            if "(なし)" in cats:
                s = s.fillna("(なし)")
            out[c] = pd.Categorical(s, categories=cats)
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def cmd_inf2026(a) -> None:
    """変更を入れた xgb（2016〜2025 年の全行で学習）を 2026 年の推論の入力に当て、束のほかのメンバーと合わせて採点する（参考。選択には使わない）。"""
    from src.pipeline.models import ensemble as en
    from src.scripts.data import v6e_2026_gap as gap

    spec = json.loads(a.spec) if a.spec else {}
    xi = load_inf_x()
    mem = inf_members(xi)
    nr = newcomer_races_2026()
    X, meta, info = load_cache()
    fin = meta["finish_position"].to_numpy().astype(float)
    ok = np.isfinite(fin) & (meta["year"].to_numpy() >= spec.get("min_train_year", 2016))
    Xtr = apply_spec(X, meta, spec)
    xi = xi.sort_values(["race_id", "horse_id"]).reset_index(drop=True)
    meta_i = pd.DataFrame({"race_class": np.where(xi["race_id"].isin(nr), "新馬", "")})
    Xi = apply_spec(cast_like(xi, X), meta_i, spec)
    params = {**info["params"], **spec.get("params", {})}
    w = None
    if spec.get("halflife_years"):
        age = 2025 - meta["year"].to_numpy()[ok]
        w = 0.5 ** (np.clip(age, 0, None) / float(spec["halflife_years"]))
    race = xi["race_id"].to_numpy()
    pv = np.zeros(len(xi))
    for s in range(a.seeds):
        pv += norm_race(race, fit_win(Xtr.loc[ok], (fin[ok] == 1).astype(int), Xi, params, seed=s, weight=w))
    pv /= a.seeds
    d = xi[KEY].copy()
    d["var_xgb"] = pv
    d = d.merge(mem, on=KEY, how="left")
    ens = json.loads((CYC / "ens" / f"{a.ens}.json").read_text(encoding="utf-8")) if a.ens else None
    if ens and (ens.get("add") or set(ens.get("replace", {})) - {"xgb_extras_ped"}):
        raise SystemExit("inf2026 は xgb_extras_ped だけを差し替えた ens に限る（--add・ほかの --replace のメンバーは 2026 年の確率が無い）")
    if ens:
        wts = ens["weights"]
    else:
        bj = json.loads((CAND_BUNDLE / "bundle.json").read_text(encoding="utf-8"))
        wts = dict(zip([m["name"] for m in bj["members"]], bj["spec"]["targets"]["win"]["weights"]))
    cols = []
    for n in wts:
        cols.append("var_xgb" if n == "xgb_extras_ped" else f"{n}__win")
    P = np.clip(d[cols].to_numpy(), 1e-6, 1 - 1e-6)
    d["p_win"] = norm_race(race, en.apply_spec({"method": "weighted_logit", "weights": [wts[n] for n in wts]}, P))
    res = gap.results([2026])
    f_var = gap.race_frame(d[KEY + ["p_win"]], res)
    f_bun = gap.race_frame(d[KEY + ["bundle_p_win"]].rename(columns={"bundle_p_win": "p_win"}), res)
    out = {"races": int(f_var["race_id"].nunique()), "ll_var": float(gap.race_ll(f_var)["ll_m"].mean()),
           "ll_bundle": float(gap.race_ll(f_bun)["ll_m"].mean()), "ll_market": float(gap.race_ll(f_var)["ll_q"].mean()),
           "var_minus_bundle": gap.paired(f_var, f_bun)}
    dd = CYC / "inf2026"
    dd.mkdir(parents=True, exist_ok=True)
    d.to_parquet(dd / f"{a.name}.parquet", index=False)
    (dd / f"{a.name}.json").write_text(json.dumps({"spec": spec, "seeds": a.seeds, "ens": a.ens, **out}, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"inf2026 {a.name}: {out}")


PARITY_GAP = 0.10           # 欠けの割合の差がこれ以上の列は「ずれ」（新馬戦・それ以外のどちらか）


def cmd_parity(a) -> None:
    spec = json.loads(a.spec) if a.spec else {}
    tab, info = parity_table(spec)
    allow = set(json.loads(a.allow)) if a.allow else set()
    bad = tab[((tab["gap_new"] >= PARITY_GAP) | (tab["gap_oth"] >= PARITY_GAP)) & ~tab.index.isin(allow)]
    d = CYC / "parity"
    d.mkdir(parents=True, exist_ok=True)
    tab.to_csv(d / f"{a.name}.csv")
    (d / f"{a.name}.json").write_text(json.dumps({**info, "gap": PARITY_GAP, "n_cols": len(tab), "n_bad": len(bad),
                                                  "bad": bad.round(3).reset_index(names="col").to_dict("records")}, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"parity {a.name}: 列 {len(tab)} ずれ {len(bad)} 件 {info}")
    with pd.option_context("display.width", 200, "display.max_rows", 400):
        print(bad.sort_values(["gap_oth", "gap_new"], ascending=False).round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cache")
    m = sub.add_parser("member")
    m.add_argument("name")
    m.add_argument("--spec", default="")
    m.add_argument("--seeds", type=int, default=1, help="篩いは 1（相手は同じ 1 seed の c00）。cand_v6e16・prod_v6e（3 seed）と比べるときは 3")
    m.add_argument("--force", action="store_true")
    e = sub.add_parser("ens")
    e.add_argument("name", help="xgb_extras_ped の勝率を差し替える OOF の名前（base なら差し替えなし）")
    e.add_argument("--replace", nargs="*", help="ほかのメンバーの差し替え MEMBER=NAME")
    e.add_argument("--add", nargs="*", help="足すメンバー（勝率だけ）")
    e.add_argument("--tag", default="")
    s = sub.add_parser("score")
    s.add_argument("name")
    s.add_argument("--ref", default="cand_v6e16,prod_v6e")
    p = sub.add_parser("parity")
    p.add_argument("name")
    p.add_argument("--spec", default="")
    p.add_argument("--allow", default="", help="構造上のずれとして許す列（JSON の配列）")
    i = sub.add_parser("inf2026")
    i.add_argument("name")
    i.add_argument("--spec", default="")
    i.add_argument("--seeds", type=int, default=1)
    i.add_argument("--ens", default="", help="勝率の重み（ens/<名前>.json）。省くと候補の束の重み")
    a = ap.parse_args()
    {"cache": cmd_cache, "member": cmd_member, "ens": cmd_ens, "score": cmd_score, "parity": cmd_parity, "inf2026": cmd_inf2026}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
