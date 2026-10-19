"""tp-42: めぐ指数（v3a の絶対指数）から作る『レースレベル』特徴量の評価。tp-39 と同じ土台（V4 の特徴量表・基準 OOF・段階評価）を使う。

python -m src.scripts.data.tp42_eval groups     # 特徴量の群を作って tp42/groups/*.parquet に保存（被覆率も）
python -m src.scripts.data.tp42_eval stage1     # 第 1 段: valid 2022+2023、seed 1 本。V4 全体に対する増分
python -m src.scripts.data.tp42_eval stage2     # 第 2 段: 第 1 段を通った群を valid 2024、新 seed 2 本で確認
python -m src.scripts.data.tp42_eval overlap    # 既存 V4 列との相関 / 線形で説明できる割合（VIF 相当）
python -m src.scripts.data.tp42_eval importance # V4 + 全群での勝率モデルの gain 重要度（fold 2023）
python -m src.scripts.data.tp42_eval calib      # レースの強さの帯別の較正（基準 OOF）
python -m src.scripts.data.tp42_eval subst      # 既存のレース内相対化列を外したとき、RL_B が代わりになるか
test 2025・2026 年は使わない。市場オッズ・人気は使わない。環境変数: TP39_THREADS, TP39_LOAD_LIMIT, TP39_MAX_HEAVY（tp39_eval の待機の仕組みを共用）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.pipeline.features import factory_self as fs  # noqa: E402
from src.pipeline.features import tp39_growth as tg  # noqa: E402
from src.pipeline.features import tp42_racelevel as rl  # noqa: E402
from src.scripts.data import tp39_eval as te  # noqa: E402

KEY = ["race_id", "horse_id"]
OUT = te.V / "tp42"
te.OUT = OUT                                   # tp39_eval の段階評価をそのまま使い、出力先だけ替える
GROUP_NAMES = ["RL_A", "RL_A2", "RL_B", "RL_C", "RL_ALL"]
REL_COLS = re.compile(r"(_rz|_rrank|_rgap)$")  # 既存のレース内相対化列（14 列 × 3）


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_all_groups(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    G = pd.concat([te._load_group(g, df) for g in names], axis=1)
    return G.loc[:, ~G.columns.duplicated()]


# ───────────────────────── 群の作成
def cmd_groups(a) -> None:
    runs = fs.load_runs(te.TABLES, list(range(2020, 2026)))
    megu = tg.load_megu(te.MEGU_ROOT, list(range(2020, 2026)))
    d = rl.prepare(runs, megu)
    _log(f"走 {len(d):,} 行 / 指数あり {int(d['a'].notna().sum()):,} 行 / megu: {megu.attrs['megu'].get('dir')} {megu.attrs['megu'].get('version')}")
    gs = rl.build_all(d)
    gdir = OUT / "groups"
    gdir.mkdir(parents=True, exist_ok=True)
    allc = [gs["RL_A"], gs["RL_A2"][[c for c in gs["RL_A2"].columns if c not in KEY]], gs["RL_B"][[c for c in gs["RL_B"].columns if c not in KEY]],
            gs["RL_C"][[c for c in gs["RL_C"].columns if c not in KEY]]]
    gs["RL_ALL"] = pd.concat(allc, axis=1)
    cov = {}
    yr = d["date"].dt.year.to_numpy()
    for n, t in gs.items():
        t.to_parquet(gdir / f"{n}.parquet", index=False)
        cols = [c for c in t.columns if c not in KEY]
        by = {int(y): float(t.loc[yr == y, cols].notna().mean().mean()) for y in (2021, 2022, 2023, 2024, 2025)}
        cov[n] = {"cols": len(cols), "columns": cols, "nonnull_rate_by_year": by}
        _log(f"{n}: {len(cols)} 列 被覆 {by}")
    te._jsave(OUT / "consts.json", {"megu": megu.attrs["megu"]})
    te._jsave(OUT / "coverage.json", cov)


# ───────────────────────── 段階評価
def cmd_stage1(a) -> None:
    groups = [g for g in os.environ.get("TP42_GROUPS", "").split(",") if g] or GROUP_NAMES
    te._stage(os.environ.get("TP42_LABEL", "stage1"), [2022, 2023], [0], groups, "stage1")


def cmd_stage2(a) -> None:
    allg: dict = {}
    for lab in ("stage1", "stage1b"):
        allg.update((te._jload(OUT / lab / "results.json") or {}).get("groups", {}))
    passed = [g for g in GROUP_NAMES if g in allg and allg[g]["vs_V4"]["Δwin"] < -0.00012 and allg[g]["vs_V4"]["win_hi"] < 0.0001]
    _log(f"第 1 段を通過: {passed}")
    te._jsave(OUT / "stage1_passed.json", passed)
    if not passed:
        return
    te._stage("stage2", [2024], [2, 3], passed + (["ALL_PASSED"] if len(passed) > 1 else []), "stage2")


# ───────────────────────── 既存列との重なり
def cmd_overlap(a) -> None:
    from sklearn.linear_model import Ridge

    from src.scripts.data import tp27_eval as t

    df, base, meta, folds, lp = t.setup()
    X0 = te._frame(df)
    num = [c for c in X0.columns if str(X0[c].dtype) not in ("category", "object", "string")]
    G = te._load_group("RL_ALL", df)
    mask = df["date"].dt.year.isin([2022, 2023, 2024]).to_numpy()
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(np.flatnonzero(mask), size=min(60000, int(mask.sum())), replace=False))
    A = X0.iloc[idx][num].astype("float32")
    A = A.fillna(A.median())
    keep = A.std() > 1e-9
    A = A.loc[:, keep]
    Z = ((A - A.mean()) / A.std()).to_numpy("float32")
    names = list(A.columns)
    res = {}
    ntr = int(len(idx) * 0.7)
    for c in G.columns:
        y = G.iloc[idx][c].astype("float64")
        ok = y.notna().to_numpy()
        if ok.mean() < 0.3 or y[ok].std() < 1e-9:
            continue
        yy = (y - y[ok].mean()) / y[ok].std()
        yv = np.where(ok, yy, 0.0)
        corr = (Z[ok].T @ yv[ok].astype("float32")) / ok.sum()
        top = np.argsort(-np.abs(corr))[:3]
        trn, tst = np.arange(len(idx)) < ntr, np.arange(len(idx)) >= ntr
        m = Ridge(alpha=30.0).fit(Z[trn & ok], yv[trn & ok])
        pred = m.predict(Z[tst & ok])
        r2 = 1.0 - float(((yv[tst & ok] - pred) ** 2).sum() / max(((yv[tst & ok] - yv[tst & ok].mean()) ** 2).sum(), 1e-12))
        res[c] = {"max_abs_corr": float(np.abs(corr[top[0]])), "top_corr": [(names[i], round(float(corr[i]), 3)) for i in top], "r2_from_V4": round(r2, 3),
                  "vif_equiv": round(1.0 / max(1.0 - r2, 1e-3), 2), "nonnull": round(float(ok.mean()), 3)}
        _log(f"{c}: R²(V4)={r2:.3f} max|r|={abs(corr[top[0]]):.3f} {res[c]['top_corr'][:2]}")
    te._jsave(OUT / "overlap.json", res)


# ───────────────────────── 重要度
def cmd_importance(a) -> None:
    from src.pipeline.models import zoo
    from src.scripts.data import tp27_eval as t

    df, base, meta, folds, lp = t.setup()
    X = pd.concat([te._frame(df), _load_all_groups(df, ["RL_ALL"])], axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    f = [f for f in folds if int(meta["year"].to_numpy()[f.valid_index][0]) == 2023][0]
    tr, va = f.train_index, f.valid_index
    fin = meta["finish_position"].to_numpy()
    ytr = {"win": (fin[tr] == 1).astype(int), "top2": (fin[tr] <= 2).astype(int), "top3": (fin[tr] <= 3).astype(int)}
    race = meta["race_id"].to_numpy()
    te._wait()
    mo: dict = {}
    zoo.fit_predict("lgb_binary", X.iloc[tr], ytr, race[tr], X.iloc[va], race[va], te.THREADS, seed=0, params=dict(lp), model_out=mo)
    out = {}
    for tgt, bst in mo["boosters"].items():
        gain = pd.Series(bst.feature_importance("gain"), index=bst.feature_name())
        gain = gain / gain.sum()
        new = [c for c in gain.index if c.startswith("rl")]
        rk = gain.rank(ascending=False)
        out[tgt] = {"new_total_gain_share": float(gain[new].sum()), "n_new": len(new),
                    "top_new": [(c, round(float(gain[c]), 5), int(rk[c])) for c in gain[new].sort_values(ascending=False).index[:15]],
                    "top_all": [(c, round(float(v), 5)) for c, v in gain.sort_values(ascending=False).head(10).items()]}
        _log(f"{tgt}: 新列の gain 割合 {out[tgt]['new_total_gain_share']:.4f} 上位 {out[tgt]['top_new'][:4]}")
    te._jsave(OUT / "importance.json", out)


# ───────────────────────── 較正（レースの強さの帯別）
def cmd_calib(a) -> None:
    from src.scripts.data import tp27_eval as t

    df, base, meta, folds, lp = t.setup()
    G = te._load_group("RL_ALL", df)
    keep = ["rlB_mean", "rlB_max", "rlB_gap12", "rlB_sd", "rlA_oppm_ewm", "rlC_E"]
    m = pd.concat([df[KEY + ["date"]], G[keep]], axis=1)
    fin = meta.set_index(KEY)["finish_position"]
    res = {}
    for lab in ("stage1", "stage2"):
        p = FR / lab / "base.parquet"
        if not p.exists():
            continue
        o = pd.read_parquet(p)
        o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
        o = o.merge(m, on=KEY, how="left").merge(fin.rename("fin").reset_index(), on=KEY, how="left")
        o = o[o["fin"] > 0]
        for feat in ("rlB_mean", "rlB_gap12", "rlB_sd"):
            race = o.groupby("race_id")[feat].transform("first")
            q = pd.qcut(race, 5, duplicates="drop")
            rows = []
            for b, g in o.groupby(q, observed=True):
                w, t3 = (g["fin"] == 1).to_numpy(), (g["fin"] <= 3).to_numpy()
                rows.append({"bin": str(b), "n": int(len(g)), "races": int(g["race_id"].nunique()),
                             "p_win_mean": float(g["p_win"].mean()), "win_rate": float(w.mean()),
                             "logloss_win": float(-(w * np.log(np.clip(g["p_win"], 1e-9, 1)) + (~w) * np.log(np.clip(1 - g["p_win"], 1e-9, 1))).mean()),
                             "p_top3_mean": float(g["p_top3"].mean()), "top3_rate": float(t3.mean())})
            res[f"{lab}|{feat}"] = rows
            _log(f"{lab}|{feat}: " + " / ".join(f"{r['bin'][:14]} 勝率差{r['p_win_mean'] - r['win_rate']:+.4f} LL{r['logloss_win']:.4f}" for r in rows))
    te._jsave(OUT / "calib.json", res)


FR = te.FRAME31


# ───────────────────────── 既存の相対化列を外したとき
def cmd_subst(a) -> None:
    from src.scripts.data import tp27_eval as t

    df, base, meta, folds, lp = t.setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    X0 = te._frame(df)
    rel = [c for c in X0.columns if REL_COLS.search(c)]
    X1 = X0.drop(columns=rel)
    d = OUT / "subst"
    d.mkdir(parents=True, exist_ok=True)
    resp = d / "results.json"
    res = te._jload(resp)
    res["removed_cols"] = len(rel)
    lv0, o0 = te._run_subset("V4-rel", X1, meta, folds, lp, [2022, 2023], [0])
    lv1, o1 = te._run_subset("V4-rel+RL_B", pd.concat([X1, _load_all_groups(df, ["RL_B"])], axis=1), meta, folds, lp, [2022, 2023], [0])
    base_o = pd.read_parquet(FR / "stage1" / "base.parquet")
    res["V4-rel"] = {"levels": lv0, "vs_V4": t._cmp(base_o, o0, fin_map, "V4-rel")}
    res["V4-rel+RL_B"] = {"levels": lv1, "vs_V4": t._cmp(base_o, o1, fin_map, "V4-rel+RL_B"), "vs_V4-rel": t._cmp(o0, o1, fin_map, "RL_B|-rel")}
    te._jsave(resp, res)
    _log(json.dumps({k: {a: b for a, b in v["vs_V4"].items() if isinstance(b, float)} for k, v in res.items() if isinstance(v, dict) and "vs_V4" in v}, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["groups", "stage1", "stage2", "overlap", "importance", "calib", "subst"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"groups": cmd_groups, "stage1": cmd_stage1, "stage2": cmd_stage2, "overlap": cmd_overlap, "importance": cmd_importance, "calib": cmd_calib, "subst": cmd_subst}[a.cmd](a)


if __name__ == "__main__":
    main()
