"""tp-42: めぐ指数（v3a の絶対指数）から作る『レースレベル』特徴量の評価。tp-39 と同じ土台（V4 の特徴量表・基準 OOF・段階評価）を使う。

python -m src.scripts.data.tp42_eval groups     # 特徴量の群を作って tp42/groups/*.parquet に保存（被覆率も）
python -m src.scripts.data.tp42_eval stage1     # 第 1 段: valid 2022+2023、seed 1 本。V4 全体に対する増分
python -m src.scripts.data.tp42_eval stage2     # 第 2 段: 第 1 段を通った群を valid 2024、新 seed 2 本で確認
python -m src.scripts.data.tp42_eval overlap    # 既存 V4 列との相関 / 線形で説明できる割合（VIF 相当）
python -m src.scripts.data.tp42_eval importance # V4 + 全群での勝率モデルの gain 重要度（fold 2023）
python -m src.scripts.data.tp42_eval calib      # レースの強さの帯別の較正（基準 OOF）
python -m src.scripts.data.tp42_eval subst      # 既存のレース内相対化列を外したとき、RL_B が代わりになるか
python -m src.scripts.data.tp42_eval member    # 第 3 段: V4 アンサンブルのメンバー1つ（TP42_MEMBER）に RL 群を足して LOYO(valid 2022–24)で OOF（tp-27 の member と同じ土台）
python -m src.scripts.data.tp42_eval ens       # 第 3 段: 7 メンバーのうち差し替えたものを使った LOYO アンサンブルを v4_e2 と比べる（TP42_MEMBERS）
python -m src.scripts.data.tp42_eval corr39    # tp-39（成長曲線）の群との相関・重複
python -m src.scripts.data.tp42_eval add       # V4 に別の既存群（TP42_BASEGROUPS: tp39 の群）を足した上に RL を足す増分
python -m src.scripts.data.tp42_eval stack      # 軽い一次選別: V4 のアンサンブル OOF の log p_win に特徴量を 1 つ足す条件付きロジット（tp-39 と同じ）
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
VARIANT = os.environ.get("TP42_VARIANT", "")                   # 先読み解消版・鮮度版などの出力先を分ける（例 lf / lf_l7 / lf_l14）
OUT = te.V / (f"tp42_{VARIANT}" if VARIANT else "tp42")
MEGU_ROOT = te.ROOT / os.environ["TP42_MEGU_ROOT"] if os.environ.get("TP42_MEGU_ROOT") else te.MEGU_ROOT   # 指数の読み先（既定: 最終候補 v3a）
LAG_DAYS = int(os.environ.get("TP42_LAG_DAYS", "0"))            # 指数の反映遅れ（日）。0 なら遅れなし
te.OUT = OUT                                   # tp39_eval の段階評価をそのまま使い、出力先だけ替える
GROUP_NAMES = ["RL_A", "RL_A2", "RL_B", "RL_C", "RL_ALL"]
REL_COLS = re.compile(r"(^fm_|(_rank|_rk|_gap|_gap12|_own_z|_n_better)$)")  # 既存のレース内相対化列（V4 では *_rank / *_gap / メンバー構成 fm_*。*_rz 等の名前は V4 の表には無い）


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_all_groups(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    G = pd.concat([te._load_group(g, df) for g in names], axis=1)
    return G.loc[:, ~G.columns.duplicated()]


# ───────────────────────── 群の作成
def cmd_groups(a) -> None:
    runs = fs.load_runs(te.TABLES, list(range(2020, 2026)))
    keep_low = os.environ.get("TP42_KEEP_LOW") == "1"                    # 低信頼フラグの行も使う（先読みの有無と被覆率の影響を分けて見るため）
    megu = tg.load_megu(MEGU_ROOT, list(range(2020, 2026)), spec={"low_col": "__none__"} if keep_low else None)
    d = rl.prepare(runs, megu)
    _log(f"走 {len(d):,} 行 / 指数あり {int(d['a'].notna().sum()):,} 行 / megu: {megu.attrs['megu'].get('dir')} {megu.attrs['megu'].get('version')}")
    gs = rl.build_all(d, LAG_DAYS)
    _log(f"variant={VARIANT or '(既定)'} 指数の読み先={MEGU_ROOT.name} 反映遅れ={LAG_DAYS}日")
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
    te._jsave(OUT / "consts.json", {"megu": megu.attrs["megu"], "variant": VARIANT, "lag_days": LAG_DAYS, "keep_low": keep_low})
    te._jsave(OUT / "coverage.json", cov)


# ───────────────────────── 段階評価
def cmd_stage1(a) -> None:
    groups = [g for g in os.environ.get("TP42_GROUPS", "").split(",") if g] or GROUP_NAMES
    te._stage(os.environ.get("TP42_LABEL") or "stage1", [2022, 2023], [0], groups, "stage1")


def cmd_stage2c(a) -> None:
    """第 2 段を、群を指定して別ラベルで並行実行する（環境変数 TP42_GROUPS、結果は stage2c/）。"""
    groups = [g for g in os.environ.get("TP42_GROUPS", "").split(",") if g]
    te._stage(os.environ.get("TP42_LABEL") or "stage2c", [2024], [2, 3], groups, "stage2")


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
    if not len(rel):
        raise RuntimeError("外す列が 0 件（正規表現が V4 の列名に合っていない）")
    base_o = pd.read_parquet(FR / "stage1" / "base.parquet")
    res["V4-rel"] = {"levels": lv0, "vs_V4": t._cmp(base_o, o0, fin_map, "V4-rel")}
    res["V4-rel+RL_B"] = {"levels": lv1, "vs_V4": t._cmp(base_o, o1, fin_map, "V4-rel+RL_B"), "vs_V4-rel": t._cmp(o0, o1, fin_map, "RL_B|-rel")}
    te._jsave(resp, res)
    _log(json.dumps({k: {a: b for a, b in v["vs_V4"].items() if isinstance(b, float)} for k, v in res.items() if isinstance(v, dict) and "vs_V4" in v}, ensure_ascii=False))


def cmd_stack(a) -> None:
    import shutil

    if not (OUT / "nprior.parquet").exists():
        shutil.copy(te.V / "tp39" / "nprior.parquet", OUT / "nprior.parquet")          # 馬ごとの過去走数（履歴 0〜2 走の部分集合用）
    te.GROUP_NAMES = ["RL_A", "RL_A2", "RL_B", "RL_C"]
    te.cmd_stack(a)


# ───────────────────────── 第 3 段: アンサンブルのメンバー
def _member_X(name: str):
    from src.scripts.data import tp25e_eval as e
    from src.scripts.data import tp27_eval as t27

    df, meta, folds, members, TLX = e.load_v3()
    recipe, params, X = members[name]
    X = pd.concat([X, TLX], axis=1)
    aux = pd.read_parquet(t27.AUX_DIR / "aux_B.parquet")
    aux["race_id"], aux["horse_id"] = aux["race_id"].astype(str), aux["horse_id"].astype(str)
    m = df[KEY].merge(aux, on=KEY, how="left")
    A = m[[c for c in m.columns if c not in KEY]]
    A.index = df.index
    E = t27._extra_for_ensemble(df, "core", t27.NN_TOP if recipe == "nn" else None)
    return df, meta, folds, recipe, params, pd.concat([X, A, E], axis=1)


def cmd_member(a) -> None:
    from src.pipeline.models import zoo_runner
    from src.scripts.data import tp27_eval as t27

    name = os.environ["TP42_MEMBER"]
    grp = os.environ.get("TP42_ADD", "RL_ALL").split(",")
    df, meta, folds, recipe, params, X = _member_X(name)
    Xn = pd.concat([X, _load_all_groups(df, grp)], axis=1)
    Xn = Xn.loc[:, ~Xn.columns.duplicated()]
    p = OUT / "member_oof" / f"{name}__v4rl.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        _log(f"{p.name} 既存")
        return
    te._wait()
    t0 = time.time()
    res, oof = zoo_runner.run_oof(f"{name}+{'+'.join(grp)}", recipe, Xn, meta, folds, t27.SEEDS + [2], te.THREADS, params=dict(params), out_dir=None)
    oof.to_parquet(p, index=False)
    lv = {m: float(res[m].mean()) for m in ("logloss_win", "logloss_top2", "logloss_top3")}
    base = pd.read_parquet(t27.OUT / "oof" / f"{name}__v4.parquet")
    fin_map = meta.set_index(KEY)["finish_position"]
    base["race_id"], base["horse_id"] = base["race_id"].astype(str), base["horse_id"].astype(str)
    c = t27._cmp(base, oof, fin_map, f"{name}+RL")
    te._jsave(OUT / "member_oof" / f"{name}__v4rl.json", {"levels": lv, "cols": int(Xn.shape[1]), "vs_v4_member": c, "seconds": time.time() - t0, "groups": grp, "variant": VARIANT})
    _log(f"{name}+RL: win {lv['logloss_win']:.5f} top2 {lv['logloss_top2']:.5f} top3 {lv['logloss_top3']:.5f} cols={Xn.shape[1]} ({time.time() - t0:.0f}s) Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop3 {c['Δtop3']:+.5f}")


def cmd_ens(a) -> None:
    """tp27_eval.cmd_ens と同じ LOYO アンサンブル。TP42_MEMBERS（カンマ区切り）の名前は RL を足したメンバーの OOF に差し替える。"""
    from src.pipeline.models import ensemble as en
    from src.pipeline.models import tp05_data
    from src.scripts.data import tp27_eval as t27

    swap = [x for x in os.environ.get("TP42_MEMBERS", "").split(",") if x]
    names7 = ["lgb_extras_ped", "xgb_extras_ped", "lgb_extras", "lgb_softmax", "linear", "mtl_nn_extras_ped", "mtl_nn_s"]
    D = tp05_data.load_dataset(t27.WORK_V)
    meta = D["meta"]
    fin_map = meta.set_index(KEY)["finish_position"]
    out = {"swapped": swap, "variant": VARIANT, "results": {}}
    for tag, sw in (("baseline", []), ("swapped", swap)):
        base = meta[KEY + ["finish_position", "year"]].copy()
        for n in names7:
            p = OUT / "member_oof" / f"{n}__v4rl.parquet" if n in sw else t27.OUT / "oof" / f"{n}__v4.parquet"
            o = pd.read_parquet(p)[KEY + ["p_win", "p_top2", "p_top3"]].copy()
            o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
            base = base.merge(o.rename(columns={f"p_{t}": f"{n}__{t}" for t in ("win", "top2", "top3")}), on=KEY, how="inner")
        race, years, fin = base["race_id"].to_numpy(), base["year"].to_numpy(), base["finish_position"].to_numpy()
        TOT = {"win": 1.0, "top2": 2.0, "top3": 3.0}
        Y = {"win": (fin == 1).astype(float), "top2": (fin <= 2).astype(float), "top3": (fin <= 3).astype(float)}
        METHOD = {"win": "weighted_logit", "top2": "greedy", "top3": "greedy"}
        ENS = base[KEY + ["year"]].copy()
        llv_all = {}
        for t in TOT:
            P = np.stack([base[f"{n}__{t}"].to_numpy() for n in names7], axis=1)
            llv, pred = en.loyo_eval(METHOD[t], P, years, race, Y[t], TOT[t])
            ENS[f"p_{t}"] = pred
            llv_all[t] = float(llv)
        out["results"][tag] = {"loyo": llv_all, "rows": int(len(ENS))}
        ENS.to_parquet(OUT / f"ens_{tag}.parquet", index=False)
        _log(f"{tag}: LOYO {llv_all}")
    b = pd.read_parquet(OUT / "ens_baseline.parquet")
    s = pd.read_parquet(OUT / "ens_swapped.parquet")
    out["swapped_vs_baseline"] = t27._cmp(b[KEY + ["p_win", "p_top2", "p_top3"]], s[KEY + ["p_win", "p_top2", "p_top3"]], fin_map, "swapped vs baseline")
    te._jsave(OUT / "ens.json", out)
    _log("swapped vs baseline: " + str({k: round(v, 5) for k, v in out["swapped_vs_baseline"].items() if isinstance(v, float)}))


# ───────────────────────── tp-39 との重なり / 上積み
def cmd_corr39(a) -> None:
    G = pd.read_parquet(OUT / "groups" / "RL_ALL.parquet")
    G["race_id"], G["horse_id"] = G["race_id"].astype(str), G["horse_id"].astype(str)
    res = {}
    for g in ("GR_mga", "GR_meg", "SL_mga", "KR_mga", "POP_megu"):
        pth = te.V / "tp39" / "groups" / f"{g}.parquet"
        if not pth.exists():
            continue
        T = pd.read_parquet(pth)
        T["race_id"], T["horse_id"] = T["race_id"].astype(str), T["horse_id"].astype(str)
        m = G.merge(T, on=KEY, how="inner")
        m = m[m["race_id"].str[:4].isin(["2022", "2023", "2024"])]
        if len(m) > 80000:
            m = m.sample(80000, random_state=0)
        a_cols = [c for c in G.columns if c not in KEY]
        b_cols = [c for c in T.columns if c not in KEY]
        C = m[a_cols + b_cols].corr(method="pearson").loc[a_cols, b_cols]
        mx = C.abs().max(axis=1)
        res[g] = {"cols": len(b_cols), "max_abs_corr_by_rl": {c: round(float(v), 3) for c, v in mx.sort_values(ascending=False).head(8).items()},
                  "best_pairs": [(i, j, round(float(C.loc[i, j]), 3)) for i, j in C.abs().stack().sort_values(ascending=False).head(5).index]}
        _log(f"{g}: {res[g]['best_pairs'][:3]}")
    te._jsave(OUT / "corr39.json", res)


def cmd_add(a) -> None:
    """V4 + 既存の別群(TP42_BASEGROUPS=tp39 の群名) を土台に、RL_ALL を足した増分を測る（valid 2022–23, seed 0）。tp-39 の群は tp39/groups から読む。"""
    from src.scripts.data import tp27_eval as t

    base_groups = [g for g in os.environ["TP42_BASEGROUPS"].split(",") if g]
    df, base, meta, folds, lp = t.setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    X0 = te._frame(df)
    def load39(g):
        T = pd.read_parquet(te.V / "tp39" / "groups" / f"{g}.parquet")
        T["race_id"], T["horse_id"] = T["race_id"].astype(str), T["horse_id"].astype(str)
        m = df[KEY].merge(T, on=KEY, how="left")
        m = m[[c for c in m.columns if c not in KEY]]
        m.index = df.index
        return m
    B = pd.concat([load39(g) for g in base_groups], axis=1)
    Xb = pd.concat([X0, B], axis=1)
    Xb = Xb.loc[:, ~Xb.columns.duplicated()]
    d = OUT / "add"
    d.mkdir(parents=True, exist_ok=True)
    resp = d / "results.json"
    res = te._jload(resp)
    tag = "+".join(base_groups)
    lv0, o0 = te._run_subset(f"V4+{tag}", Xb, meta, folds, lp, [2022, 2023], [0])
    Xn = pd.concat([Xb, _load_all_groups(df, ["RL_ALL"])], axis=1)
    Xn = Xn.loc[:, ~Xn.columns.duplicated()]
    lv1, o1 = te._run_subset(f"V4+{tag}+RL_ALL", Xn, meta, folds, lp, [2022, 2023], [0])
    base_o = pd.read_parquet(FR / "stage1" / "base.parquet")
    res[tag] = {"cols_base": int(Xb.shape[1]), "levels_base": lv0, "levels_new": lv1, "tag_vs_V4": t._cmp(base_o, o0, fin_map, tag),
                "RL_over_tag": t._cmp(o0, o1, fin_map, f"RL|{tag}"), "RL_vs_V4": t._cmp(base_o, o1, fin_map, f"{tag}+RL")}
    te._jsave(resp, res)
    _log(tag + ": " + json.dumps({k: {a_: b_ for a_, b_ in v.items() if a_ in ("Δwin", "win_hi", "Δtop3")} for k, v in res[tag].items() if isinstance(v, dict) and "Δwin" in v}, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["groups", "stage1", "stage2", "stage2c", "member", "ens", "corr39", "add", "overlap", "importance", "calib", "subst", "stack"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"groups": cmd_groups, "stage1": cmd_stage1, "stage2": cmd_stage2, "stage2c": cmd_stage2c, "member": cmd_member, "ens": cmd_ens, "corr39": cmd_corr39, "add": cmd_add, "overlap": cmd_overlap, "importance": cmd_importance, "calib": cmd_calib, "subst": cmd_subst, "stack": cmd_stack}[a.cmd](a)


if __name__ == "__main__":
    main()
