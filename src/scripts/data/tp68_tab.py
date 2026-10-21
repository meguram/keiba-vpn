"""tp-68: 表形式データの定番手法のうち未実施だったものの検証（null importance・RFE 簡易版・欠損数/暦の特徴・疑似ラベル・ID 系の件数）。

python -m src.scripts.data.tp68_tab nullimp [--n-null 20 --rounds 250]   # ≤2021 年の行で、実ラベルと『レース内で並べ替えたラベル』の gain 重要度を比べる
python -m src.scripts.data.tp68_tab variants                              # 列集合（null 基準・RFE 簡易版）を作る → <OUT>/variants.json
python -m src.scripts.data.tp68_tab run --name NAME [--add-miss --add-cal]  # V5（1,142 列）の列集合 NAME / 足した列で学習し、V5 の OOF と対で比べる
python -m src.scripts.data.tp68_tab pseudo                                # 疑似ラベルのシミュレーション（≤2022 学習 → 2023 を疑似ラベル → 2024 で評価）
python -m src.scripts.data.tp68_tab idcheck                               # ID 系（馬主・生産者・厩舎・外厩）の as-of 件数/頻度の有無
採用基準は feature_bakeoff.compare_oof（足す場合）。列を減らす場合は『勝率の 95% 区間の上限 ≤ +0.0002、連対・複勝の悪化 ≤ 0.0002』を軽量化の基準にする。
test 2025・2026 は使わない。市場オッズ・人気は使わない。環境変数: TP40_THREADS（既定 4）, TP68_OUT。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TP40_THREADS", "3")

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("TP68_OUT", "/home/jovyan/work/tp68_out"))
BASE_OOF = ROOT / "data/local/modeling/training_pipeline_v4/v5_select/oof_d_CY_BEST.parquet"
KEY = ["race_id", "horse_id"]
YEARS = [2022, 2023, 2024]
SEEDS = [0, 1]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _jsave(p: Path, d) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def _jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load():
    from src.scripts.data import tp40_eval as e40
    from src.scripts.data import v5_select_eval as v5

    t = e40.t
    df, base, meta, folds, lp = t.setup()
    X = v5.build_frame(v5.CONFIGS["d_CY_BEST"], df)
    assert not X.columns.duplicated().any()
    return e40, t, df, X, meta, folds, lp


def _lgb_params(lp: dict, threads: int) -> dict:
    return {"objective": "binary", "learning_rate": 0.05, "num_threads": threads, "verbose": -1, **lp["lgb"]}


# ───────────────────────── null importance
def cmd_nullimp(a) -> None:
    import lightgbm as lgb

    from src.pipeline.features import tp68_lib as L

    e40, t, df, X, meta, folds, lp = load()
    tr = np.flatnonzero(df["year"].to_numpy() <= 2021)
    race = df["race_id"].to_numpy()[tr]
    y = (df["finish_position"].to_numpy()[tr] == 1).astype(int)
    Xt = X.iloc[tr]
    p = _lgb_params(lp, int(os.environ["TP40_THREADS"]))
    cols = list(X.columns)
    _log(f"学習行 {len(tr)}（2020〜2021 年）× {len(cols)} 列")
    real = np.zeros(len(cols))
    for s in range(3):
        e40._wait()
        b = lgb.train({**p, "seed": s}, lgb.Dataset(Xt, y), num_boost_round=a.rounds)
        real += b.feature_importance("gain") / 3
        _log(f"実ラベル seed {s} 完了")
    real_s = pd.Series(real, index=cols)
    rng = np.random.default_rng(0)
    null = np.zeros((a.n_null, len(cols)))
    for r in range(a.n_null):
        e40._wait()
        yp = L.permute_within_race(y, race, rng)
        b = lgb.train({**p, "seed": 100 + r}, lgb.Dataset(Xt, yp), num_boost_round=a.rounds)
        null[r] = b.feature_importance("gain")
        _log(f"並べ替えラベル {r + 1}/{a.n_null}")
    S = L.null_summary(real_s, null, cols)
    OUT.mkdir(parents=True, exist_ok=True)
    S.to_parquet(OUT / "nullimp.parquet", index=False)
    S["grp"] = S["col"].str.split("_").str[0]
    summ = {"n_cols": len(cols), "rows": int(len(tr)), "rounds": a.rounds, "n_null": a.n_null,
            "zero_gain_cols": int((S["real_gain"] <= 0).sum()),
            "above_null_q": {q: int((S["real_gain"] > S[f"null_q{q}"]).sum()) for q in (50, 75, 90, 95, 99)},
            "by_group_above_q75": S.assign(ok=S["real_gain"] > S["null_q75"]).groupby("grp")["ok"].agg(["sum", "count"]).rename(columns={"sum": "keep", "count": "n"}).sort_values("n", ascending=False).head(30).to_dict("index"),
            "real_gain_share_top100": float(real_s.sort_values(ascending=False).head(100).sum() / real_s.sum())}
    _jsave(OUT / "nullimp_summary.json", summ)
    _log(json.dumps({k: v for k, v in summ.items() if k != "by_group_above_q75"}, ensure_ascii=False))


def cmd_variants(a) -> None:
    from src.pipeline.features import tp68_lib as L

    S = pd.read_parquet(OUT / "nullimp.parquet")
    real = S.set_index("col")["real_gain"]
    v: dict = {}
    for q in (75, 95):
        v[f"nk_q{q}"] = [c for c in S["col"] if real[c] > S.set_index("col").loc[c, f"null_q{q}"]]
    for n in (800, 500, 300, 150):
        v[f"rfe_{n}"] = L.rfe_columns(real, n)
    _jsave(OUT / "variants.json", v)
    _log({k: len(c) for k, c in v.items()})


# ───────────────────────── 学習・比較
def _compare(e40, t, meta, oof, name: str) -> dict:
    fin_map = meta.set_index(KEY)["finish_position"]
    prev = pd.read_parquet(BASE_OOF)
    for d in (prev, oof):
        d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    cmp = t._cmp(prev, oof, fin_map, name)
    per_year = {str(y): t._cmp(prev[prev["year"] == y], oof[oof["year"] == y], fin_map, f"{name}_{y}") for y in YEARS}
    return {"compare": cmp, "per_year": per_year}


def cmd_run(a) -> None:
    from src.pipeline.features import tp68_lib as L

    e40, t, df, X, meta, folds, lp = load()
    base_cols = list(X.columns)
    n_add = 0
    if a.name != "full" and not a.add_miss and not a.add_cal:
        v = _jload(OUT / "variants.json")
        X = X[v[a.name]]
    adds = []
    if a.add_miss:
        g = L.prefix_groups(base_cols)
        M = L.missing_features(X[base_cols], g)
        km = _miss_cluster(X[base_cols], g, df)
        adds += [M, km]
    if a.add_cal:
        adds.append(L.calendar_features(df["date"]))
    if adds:
        E = pd.concat(adds, axis=1)
        E.index = X.index
        dup = set(E.columns) & set(X.columns)
        assert not dup, dup
        X = pd.concat([X, E], axis=1)
        n_add = E.shape[1]
        _log(f"足した列 {n_add}: {list(E.columns)[:8]}…")
    od = OUT / f"run_{a.name}"
    od.mkdir(parents=True, exist_ok=True)
    lv, oof = e40._run_subset(a.name, X, meta, folds, lp, YEARS, SEEDS)
    oof.to_parquet(od / "oof.parquet", index=False)
    res = {"name": a.name, "n_cols": int(X.shape[1]), "n_added": n_add, "levels": lv, **_compare(e40, t, meta, oof, a.name)}
    _jsave(od / "result.json", res)
    c = res["compare"]
    _log(f"{a.name}: Δwin={c['Δwin']:+.5f} 上限={c['win_hi']:+.5f} Δtop2={c['Δtop2']:+.5f} Δtop3={c['Δtop3']:+.5f} 採用={c['採用']} 列={X.shape[1]}")
    for y, cy in res["per_year"].items():
        _log(f"  {y}: Δwin={cy['Δwin']:+.5f} Δtop2={cy['Δtop2']:+.5f} Δtop3={cy['Δtop3']:+.5f}")


def _miss_cluster(Xb: pd.DataFrame, groups: dict, df: pd.DataFrame) -> pd.DataFrame:
    """欠損の群別割合を k-means（k=8, ≤2021 年の行で当てはめ）でクラスタ化した ID。ラベルは使わない。"""
    from sklearn.cluster import MiniBatchKMeans

    F = np.column_stack([Xb[cols].isna().mean(axis=1).to_numpy() for cols in groups.values()])
    fit = df["year"].to_numpy() <= 2021
    km = MiniBatchKMeans(n_clusters=8, random_state=0, n_init=3, batch_size=4096).fit(F[fit])
    return pd.DataFrame({"miss_cluster": km.predict(F).astype("float32")}, index=Xb.index)


# ───────────────────────── 疑似ラベル
def _fit3(lp: dict, Xtr, ytr: dict, wtr: dict | None, Xva, threads: int, seed: int) -> dict:
    import lightgbm as lgb

    from src.pipeline.models import win_models as wm

    p = dict(wm.BASE_PARAMS, **lp["lgb"], seed=seed, num_threads=threads)
    cats = [c for c in Xtr.columns if str(Xtr[c].dtype) == "category"]
    out = {}
    for k in ("win", "top2", "top3"):
        ds = lgb.Dataset(Xtr, label=ytr[k], weight=None if wtr is None else wtr[k], categorical_feature=cats or "auto")
        out[k] = lgb.train(dict(p, objective="binary"), ds, num_boost_round=wm.N_ROUNDS).predict(Xva)
    return out


def cmd_pseudo(a) -> None:
    from src.pipeline.features import tp68_lib as L
    from src.pipeline.models import zoo

    e40, t, df, X, meta, folds, lp = load()
    year = df["year"].to_numpy()
    fin = df["finish_position"].to_numpy()
    race = df["race_id"].to_numpy()
    Y = {"win": (fin == 1).astype(int), "top2": (fin <= 2).astype(int), "top3": (fin <= 3).astype(int)}
    i22, i23, i24 = (np.flatnonzero(year <= 2022), np.flatnonzero(year == 2023), np.flatnonzero(year == 2024))
    thr = int(os.environ["TP40_THREADS"])
    fin_map = meta.set_index(KEY)["finish_position"]
    od = OUT / "pseudo"
    od.mkdir(parents=True, exist_ok=True)

    def to_oof(P: dict) -> pd.DataFrame:
        o = meta.iloc[i24][["race_id", "horse_id", "year"]].copy().reset_index(drop=True)
        for k in P:
            o[f"p_{k}"] = P[k]
        return o

    sums = {n: {k: np.zeros(len(i24)) for k in ("win", "top2", "top3")} for n in ("base", "control", "pseudo")}
    for s in SEEDS:
        e40._wait()
        va = np.concatenate([i23, i24])
        raw = _fit3(lp, X.iloc[i22], {k: v[i22] for k, v in Y.items()}, None, X.iloc[va], thr, s)
        p23 = {k: raw[k][: len(i23)] for k in raw}
        fin_b = zoo.finalize(race[i24], {k: raw[k][len(i23):] for k in raw})
        for k in sums["base"]:
            sums["base"][k] += fin_b[k]
        _log(f"seed {s}: 基準（≤2022 のラベルのみ）完了")
        e40._wait()
        i_c = np.concatenate([i22, i23])
        raw_c = _fit3(lp, X.iloc[i_c], {k: v[i_c] for k, v in Y.items()}, None, X.iloc[i24], thr, s)
        fin_c = zoo.finalize(race[i24], raw_c)
        for k in sums["control"]:
            sums["control"][k] += fin_c[k]
        _log(f"seed {s}: 対照（≤2023 のラベル）完了")
        e40._wait()
        # 疑似ラベルの確率は、レース内の合計をそろえた値（勝率 1、連対 2、複勝 3）
        fin23 = zoo.finalize(race[i23], p23)
        Xa, ya, wa = L.soft_label_augment(X.iloc[i22], {k: v[i22] for k, v in Y.items()}, X.iloc[i23], fin23)
        raw_p = _fit3(lp, Xa, ya, wa, X.iloc[i24], thr, s)
        fin_p = zoo.finalize(race[i24], raw_p)
        for k in sums["pseudo"]:
            sums["pseudo"][k] += fin_p[k]
        _log(f"seed {s}: 疑似ラベル（≤2022 のラベル + 2023 の自己予測）完了")
    oofs = {n: to_oof({k: v / len(SEEDS) for k, v in sums[n].items()}) for n in sums}
    for n, o in oofs.items():
        o.to_parquet(od / f"oof_{n}.parquet", index=False)
    res = {"pseudo_vs_base": t._cmp(oofs["base"], oofs["pseudo"], fin_map, "pseudo vs base"),
           "control_vs_base": t._cmp(oofs["base"], oofs["control"], fin_map, "control vs base"),
           "pseudo_vs_control": t._cmp(oofs["control"], oofs["pseudo"], fin_map, "pseudo vs control")}
    for n in ("base", "control", "pseudo"):
        res[f"ll_{n}"] = {k: float(_ll(oofs[n][f"p_{k}"].to_numpy(), (fin[i24] <= {"win": 1, "top2": 2, "top3": 3}[k]).astype(float))) for k in ("win", "top2", "top3")}
    _jsave(od / "result.json", res)
    for k in ("pseudo_vs_base", "control_vs_base", "pseudo_vs_control"):
        c = res[k]
        _log(f"{k}: Δwin={c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2={c['Δtop2']:+.5f} Δtop3={c['Δtop3']:+.5f}")


def _ll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


# ───────────────────────── ID 系
def cmd_idcheck(a) -> None:
    import re

    import pyarrow.parquet as pq

    f = ROOT / "data/local/modeling/training_pipeline_v4/tp31/v4_frame.parquet"
    cols = list(pq.read_schema(f).names)
    ents = {"馬主": r"owner|^o_|ob_", "生産者": r"breeder|bree", "厩舎所属(美浦/栗東)": r"stable|belong|tc_|region", "外厩": r"gaikyu|outer|farm|ext", "騎手": r"jockey|^jk_|^fdr", "調教師": r"trainer|^tr_", "種牡馬": r"sire|^ped"}
    res = {k: [c for c in cols if re.search(v, c)][:25] for k, v in ents.items()}
    res["counts"] = {k: len([c for c in cols if re.search(v, c)]) for k, v in ents.items()}
    _jsave(OUT / "idcheck.json", res)
    _log(json.dumps(res["counts"], ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("nullimp")
    s.add_argument("--n-null", type=int, default=20)
    s.add_argument("--rounds", type=int, default=250)
    sub.add_parser("variants")
    s = sub.add_parser("run")
    s.add_argument("--name", required=True)
    s.add_argument("--add-miss", action="store_true")
    s.add_argument("--add-cal", action="store_true")
    sub.add_parser("pseudo")
    sub.add_parser("idcheck")
    a = ap.parse_args()
    {"nullimp": cmd_nullimp, "variants": cmd_variants, "run": cmd_run, "pseudo": cmd_pseudo, "idcheck": cmd_idcheck}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
