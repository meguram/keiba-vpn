"""tp-41: 似た馬は同じレースで着順が近くなるか（ペア分析・組合せ券の確率の補正・特徴量の評価）。

python -m src.scripts.data.tp41_eval pairs      # V4 の OOF（2022-24）と同じレースのペア表を作る（軽い）
python -m src.scripts.data.tp41_eval describe   # 類似度の軸ごとの 観測/期待（同時3着以内）・着順差・調整後の回帰（レース単位クラスタ）
python -m src.scripts.data.tp41_eval calib      # 同じレースの似た馬の状況別に、V4 の勝率・複勝率が過大/過小か
python -m src.scripts.data.tp41_eval joint      # 組合せ券（馬連・3連複・ワイド）の確率に類似度の補正を入れて 2022-23 で当てはめ、2024 で評価
python -m src.scripts.data.tp41_eval bet        # 補正あり/なしで買い目を上位 k 点で選んだときの 2024 年の的中率・回収率（確定の払戻で精算）
python -m src.scripts.data.tp41_eval groups     # 特徴量の群（SIM_*）を作って tp41/groups/*.parquet に保存
python -m src.scripts.data.tp41_eval stage1     # 第 1 段: valid 2022+2023、seed 1 本。V4 全体に対する増分
python -m src.scripts.data.tp41_eval stage2     # 第 2 段: 第 1 段を通った群を valid 2024、新 seed 2 本で確認
test 2025・2026 年は使わない。市場オッズ・人気は使わない。環境変数: TP41_THREADS（既定 3）, TP41_LOAD_LIMIT（既定 14）。
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
V = ROOT / "data/local/modeling/training_pipeline_v4"
WORK = ROOT / "data/local/modeling/training_pipeline"
TABLES = ROOT / "data/page_reference/tables"
OUT = Path(os.environ.get("TP41_OUT") or V / "tp41")
FRAME31 = V / "tp31"
for _k, _v in {"TP_BASE_DIR": V, "TP_WORK": V, "TP25C_OUT": V / "tp25c", "TP25E_OUT": V / "tp25e", "TP27_FAC_FIELD": V / "factory_field", "TP27_OUT": V / "tp27"}.items():
    os.environ.setdefault(_k, str(_v))
os.environ.setdefault("TP27_THREADS", os.environ.get("TP41_THREADS", "3"))

from src.pipeline.features import factory_self as fs  # noqa: E402
from src.pipeline.features import tp38_pedigree as P  # noqa: E402
from src.pipeline.features import tp41_similar as ts  # noqa: E402
from src.pipeline.inference import exotic_probabilities as ex  # noqa: E402

KEY = ["race_id", "horse_id"]
THREADS = int(os.environ.get("TP41_THREADS", 3))
LOAD_LIMIT = float(os.environ.get("TP41_LOAD_LIMIT", 14))
GROUP_NAMES = list(ts.GROUPS)
AXES = ["d_str", "d_sty", "d_gate", "d_apt", "cos_ped", "same_trainer", "same_sire", "same_dsire", "both_front"]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _wait() -> None:
    t0 = time.time()
    while os.getloadavg()[0] >= LOAD_LIMIT and time.time() - t0 < 3600:
        time.sleep(30)


def _jsave(p: Path, d) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def _jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ───────────────────────── 入力
def load_all():
    from src.pipeline.features import factory_ped as fp

    years = list(range(2020, 2026))
    runs = fs.load_runs(TABLES, years)
    megu = ts.load_megu(TABLES, years)
    T = ts.runner_table(runs, megu)
    A, _ = fp.build_ped_tables(T["horse_id"].unique())
    T = ts.add_static(T, A)
    T = ts.standardize_strength(T)
    T = ts.sort_for_races(T)
    horse_index = pd.Index(sorted(set(A["horse_id"].astype(str))))
    anc, uniq = P.prep_ancestors(A, horse_index)
    X, _ = P.horse_vectors(anc, len(horse_index), len(uniq))
    return T, X, horse_index


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ───────────────────────── ペア表
def cmd_pairs(a=None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    T, X, hi = load_all()
    ens = pd.read_parquet(V / "tp27/ens/v4_e2.parquet")
    ens["race_id"], ens["horse_id"] = ens["race_id"].astype(str), ens["horse_id"].astype(str)
    full = T.groupby("race_id").size()
    T = T.merge(ens[KEY + ["p_win", "p_top2", "p_top3"]], on=KEY, how="inner")
    cnt = T.groupby("race_id").size()
    keep = cnt.index[(cnt == full.reindex(cnt.index)) & (cnt >= 5)]
    T = T[T["race_id"].isin(keep)]
    T = ts.sort_for_races(T)
    T["year"] = T["date"].dt.year
    _log(f"V4 OOF のレース {T['race_id'].nunique()}（行 {len(T)}）")
    Pp = ts.race_pairs(T, X, hi)
    i, j = Pp["i"].to_numpy(), Pp["j"].to_numpy()
    fin = T["fin"].to_numpy(float)
    nf = T["nfield"].to_numpy(float)
    y2, y3 = (fin <= 2).astype(float), (fin <= 3).astype(float)
    Pp["o3"], Pp["o2"] = y3[i] * y3[j], y2[i] * y2[j]
    fpn = (fin - 1.0) / np.maximum(nf - 1.0, 1.0)
    Pp["dfp"] = np.abs(fpn[i] - fpn[j])
    p3 = T["p_top3"].to_numpy(float)
    Pp["d_str4"] = np.abs(_logit(p3[i]) - _logit(p3[j]))
    Pp["year"] = T["year"].to_numpy()[i]
    Pp["nf"] = nf[i]
    # 期待（V4 の確率に整合する割引 Harville の同時確率）
    W3, W2 = np.full(len(Pp), np.nan), np.full(len(Pp), np.nan)
    rid = T["race_id"].to_numpy()
    starts = np.r_[0, np.flatnonzero(rid[1:] != rid[:-1]) + 1, len(rid)]
    pos = 0
    t0 = time.time()
    for k, (a_, b_) in enumerate(zip(starts[:-1], starts[1:])):
        m = b_ - a_
        npair = m * (m - 1) // 2
        dist, _ = ex.fit_rank_distribution(T["p_win"].to_numpy(float)[a_:b_], T["p_top2"].to_numpy(float)[a_:b_], T["p_top3"].to_numpy(float)[a_:b_])
        wide, uma = dist.all_wide(), dist.all_umaren()
        ii, jj = np.triu_indices(m, 1)
        W3[pos:pos + npair] = [wide[(int(x), int(y))] for x, y in zip(ii, jj)]
        W2[pos:pos + npair] = [uma[(int(x), int(y))] for x, y in zip(ii, jj)]
        pos += npair
        if k % 2000 == 0:
            _log(f"  期待 {k}/{len(starts) - 1} ({time.time() - t0:.0f}s)")
    Pp["w3"], Pp["w2"] = W3, W2
    Pp = Pp.drop(columns=["i", "j"])
    Pp.to_parquet(OUT / "pairs.parquet", index=False)
    T.to_parquet(OUT / "runners.parquet", index=False)
    _log(f"ペア {len(Pp)} 件、同時3着以内の観測 {Pp['o3'].mean():.4f} / 期待 {np.nanmean(W3):.4f}")


# ───────────────────────── 記述分析
def _boot_oe(df: pd.DataFrame, cat: str, o: str, e: str, B: int = 300, seed: int = 0) -> pd.DataFrame:
    codes, uniq = pd.factorize(df[cat], sort=True)
    ok = codes >= 0
    race = pd.factorize(df["race_id"])[0]
    R, K = race.max() + 1, len(uniq)
    O, E, N = (np.zeros((R, K)) for _ in range(3))
    np.add.at(O, (race[ok], codes[ok]), df[o].to_numpy(float)[ok])
    np.add.at(E, (race[ok], codes[ok]), df[e].to_numpy(float)[ok])
    np.add.at(N, (race[ok], codes[ok]), 1.0)
    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, R, R), minlength=R) for _ in range(B)]).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (W @ O) / (W @ E)
    out = pd.DataFrame({"bin": [str(u) for u in uniq], "n": N.sum(0).astype(int), "obs": O.sum(0), "exp": E.sum(0)})
    out["ratio"] = out["obs"] / out["exp"]
    out["lo"], out["hi"] = np.nanpercentile(r, 2.5, axis=0), np.nanpercentile(r, 97.5, axis=0)
    return out


def _bins(s: pd.Series, q: int = 5) -> pd.Series:
    if s.nunique(dropna=True) <= 2:
        return s.map({0.0: "0 (なし)", 1.0: "1 (あり)"})
    try:
        return pd.qcut(s, q, duplicates="drop").astype("string")
    except ValueError:
        return pd.Series([None] * len(s), index=s.index, dtype="string")


def _cluster_glm(y, X, offset, groups):
    import statsmodels.api as sm

    mod = sm.GLM(y, sm.add_constant(X, has_constant="add"), family=sm.families.Binomial(), offset=offset)
    r = mod.fit(cov_type="cluster", cov_kwds={"groups": groups})
    return r


def _cluster_ols_fe(y, X, race_codes):
    """レース固定効果（レース内の平均を引く）+ クラスタ頑健 SE。"""
    import statsmodels.api as sm

    df = pd.DataFrame(X.copy())
    df["_y"] = y
    df["_r"] = race_codes
    dm = df.groupby("_r").transform(lambda c: c - c.mean())
    r = sm.OLS(dm["_y"].to_numpy(), dm.drop(columns=["_y"]).to_numpy()).fit(cov_type="cluster", cov_kwds={"groups": race_codes})
    return r


def cmd_describe(a=None) -> None:
    Pp = pd.read_parquet(OUT / "pairs.parquet")
    _log(f"ペア {len(Pp)}、レース {Pp['race_id'].nunique()}")
    res: dict = {"n_pairs": int(len(Pp)), "n_races": int(Pp["race_id"].nunique()), "o3_obs": float(Pp["o3"].mean()), "o3_exp": float(Pp["w3"].mean()),
                 "o2_obs": float(Pp["o2"].mean()), "o2_exp": float(Pp["w2"].mean())}
    axes = {"d_str": "力の差（as-of の強さ）", "d_str4": "力の差（V4 予測の複勝 logit）", "d_sty": "位置取り（最初のコーナーの相対位置）の差", "d_gate": "馬番の差/頭数",
            "d_apt": "適性（距離×芝割合）の距離", "cos_ped": "血統のコサイン類似度", "same_trainer": "同じ調教師", "same_sire": "同じ父", "same_dsire": "同じ母父",
            "both_front": "ともに先行型"}
    tabs = {}
    for ax, nm in axes.items():
        d = Pp[Pp[ax].notna()].copy()
        d["bin"] = _bins(d[ax])
        d = d[d["bin"].notna()]
        t3 = _boot_oe(d, "bin", "o3", "w3")
        t2 = _boot_oe(d, "bin", "o2", "w2")
        dfp = d.groupby("bin", observed=True)["dfp"].agg(["mean", "median"]).reset_index()
        tabs[ax] = {"name": nm, "co_top3": t3.to_dict("records"), "co_top2": t2.to_dict("records"), "dfp": dfp.to_dict("records"), "n": int(len(d))}
        _log(f"{ax}: 観測/期待(同時3着以内) " + " | ".join(f"{r.bin}:{r.ratio:.3f}" for r in t3.itertuples()))
    res["bins"] = tabs
    # 力が同じくらい（V4 の差が小さい層）に限って、他の軸が残るか
    near = Pp[Pp["d_str4"] < Pp["d_str4"].quantile(0.33)]
    tabs2 = {}
    for ax in ["d_sty", "d_apt", "cos_ped", "d_gate"]:
        d = near[near[ax].notna()].copy()
        d["bin"] = _bins(d[ax], 4)
        d = d[d["bin"].notna()]
        tabs2[ax] = {"co_top3": _boot_oe(d, "bin", "o3", "w3").to_dict("records"), "dfp": d.groupby("bin", observed=True)["dfp"].mean().reset_index().to_dict("records")}
    res["bins_strength_matched"] = tabs2
    # 調整後（回帰）: 同時3着以内 = logit(期待) をオフセット、類似度を標準化して投入。レース単位クラスタ頑健 SE
    coef = {}
    base_cols = ["d_str4"]
    for tag, extra in {"単独:d_sty": ["d_sty"], "単独:d_apt": ["d_apt"], "単独:cos_ped": ["cos_ped"], "単独:d_gate": ["d_gate"], "単独:same_trainer": ["same_trainer"],
                       "単独:same_sire": ["same_sire"], "単独:both_front": ["both_front"], "単独:d_str": ["d_str"],
                       "全部": ["d_str", "d_sty", "d_apt", "cos_ped", "d_gate", "same_trainer", "same_sire", "both_front"]}.items():
        cols = base_cols + [c for c in extra if c not in base_cols]
        d = Pp.dropna(subset=cols + ["w3", "w2"]).copy()
        Z = (d[cols] - d[cols].mean()) / d[cols].std().replace(0, 1)
        grp = pd.factorize(d["race_id"])[0]
        for tgt, off in (("o3", "w3"), ("o2", "w2")):
            r = _cluster_glm(d[tgt].to_numpy(float), Z.to_numpy(), _logit(d[off].to_numpy(float)), grp)
            nm = ["const"] + cols
            coef[f"{tag}|{tgt}"] = {nm[k]: {"beta": float(r.params[k]), "se": float(r.bse[k]), "z": float(r.tvalues[k])} for k in range(len(nm))}
            coef[f"{tag}|{tgt}"]["_n"] = int(len(d))
        # 着順差（レース内の固定効果）
        r = _cluster_ols_fe(d["dfp"].to_numpy(float), Z, grp)
        nm = cols
        coef[f"{tag}|dfp"] = {nm[k]: {"beta": float(r.params[k]), "se": float(r.bse[k]), "z": float(r.tvalues[k])} for k in range(len(nm))}
        _log(f"{tag}: " + ", ".join(f"{c} z={coef[f'{tag}|o3'][c]['z']:+.1f}" for c in cols))
    res["glm"] = coef
    _jsave(OUT / "describe.json", res)
    _log("保存: " + str(OUT / "describe.json"))


# ───────────────────────── 個別の確率の較正（同じレースの似た馬の状況別）
def cmd_calib(a=None) -> None:
    T0, X, hi = load_all()
    R = pd.read_parquet(OUT / "runners.parquet")
    keep = set(R["race_id"])
    T = T0[T0["race_id"].isin(keep)].copy()
    F = ts.runner_features(T, X, hi)
    M = R[KEY + ["fin", "year", "p_win", "p_top2", "p_top3"]].merge(F, on=KEY, how="inner")
    M["y_win"], M["y_top2"], M["y_top3"] = (M["fin"] == 1).astype(float), (M["fin"] <= 2).astype(float), (M["fin"] <= 3).astype(float)
    res: dict = {"n": int(len(M)), "races": int(M["race_id"].nunique())}

    def table(sub: pd.DataFrame, col: str, bins: list | None = None, q: int = 5) -> list[dict]:
        d = sub[sub[col].notna()].copy()
        d["bin"] = pd.cut(d[col], bins, include_lowest=True).astype("string") if bins is not None else _bins(d[col], q)
        d = d[d["bin"].notna()]
        out = []
        for tgt, p in (("win", "p_win"), ("top3", "p_top3")):
            t = _boot_oe(d, "bin", f"y_{tgt}", p)
            t["target"] = tgt
            out.append(t)
        return pd.concat(out, ignore_index=True).to_dict("records")

    front = M[M["sp_own_front"] == 1.0]
    res["先行馬: 他の先行馬の数"] = table(front, "sp_front_n", [-0.5, 0.5, 1.5, 2.5, 99])
    res["差し・追込馬: 他の先行馬の数"] = table(M[M["sp_own_front"] == 0.0], "sp_front_n", [-0.5, 0.5, 1.5, 2.5, 99])
    for c in ["sp_same_n", "sp_rel_s", "sa_n", "sa_rel_s", "sd_close_n", "sd_rel_s", "sp_prod_n", "sp_prod_rel_s", "ss_dens", "ss_z", "ss_gap_up", "ss_gap_dn"]:
        res[c] = table(M, c)
    res["同じ調教師の他馬がいる: 同厩の中の順位"] = table(M[M["st_trainer_n"] > 0], "st_trainer_rank", [-0.5, 0.5, 1.5, 99])
    res["同厩の他馬の数"] = table(M, "st_trainer_n", [-0.5, 0.5, 1.5, 99])
    # グループの合計勝率の過大/過小: 先行馬の予測勝率の合計 vs 実際の勝利数（先行馬が何頭いるレースか別）
    M["pwf"] = M["p_win"] * M["sp_own_front"].fillna(0)
    M["ywf"] = M["y_win"] * M["sp_own_front"].fillna(0)
    rr = M.groupby("race_id").agg(nfront=("sp_own_front", "sum"), pwf=("pwf", "sum"), ywf=("ywf", "sum"))
    rr["bin"] = pd.cut(rr["nfront"], [-0.5, 0.5, 1.5, 2.5, 3.5, 99]).astype("string")
    rr = rr.reset_index().rename(columns={"index": "race_id"})
    rr["race_id"] = rr["race_id"].astype(str)
    gt = _boot_oe(rr, "bin", "ywf", "pwf")
    res["先行馬グループ: 勝利数/予測勝率の合計（先行馬の数別）"] = gt.to_dict("records")
    _jsave(OUT / "calib.json", res)
    for k, v in res.items():
        if isinstance(v, list):
            _log(k + ": " + " | ".join(f"{r['bin']}:{r['ratio']:.3f}[{r['lo']:.2f},{r['hi']:.2f}]" + (f"({r['target']})" if 'target' in r else '') for r in v[:10]))


# ───────────────────────── 組合せ券の確率の補正
TILT_AXES = ["d_sty", "d_apt", "cos_ped", "d_gate", "same_trainer", "same_sire", "both_front", "d_str", "d_str4"]


def _race_arrays(T: pd.DataFrame, Pp: pd.DataFrame):
    """レースごとに、3連複の全組合せの log P と、軸ごとのペア類似度の 3 組の合計、実際の 3 連複の位置を作る。"""
    rid = T["race_id"].to_numpy()
    starts = np.r_[0, np.flatnonzero(rid[1:] != rid[:-1]) + 1, len(rid)]
    Pz = Pp[TILT_AXES].copy()
    mu, sd = Pz.mean(), Pz.std().replace(0, 1)
    Z = ((Pz - mu) / sd).fillna(0.0).to_numpy()
    # 欠損の印（cos_ped が無いペア）も別軸にして、欠損 = 中立にする
    out, pos = [], 0
    fin = T["fin"].to_numpy(float)
    years = T["year"].to_numpy()
    for a_, b_ in zip(starts[:-1], starts[1:]):
        m = b_ - a_
        npair = m * (m - 1) // 2
        dist, _ = ex.fit_rank_distribution(T["p_win"].to_numpy(float)[a_:b_], T["p_top2"].to_numpy(float)[a_:b_], T["p_top3"].to_numpy(float)[a_:b_])
        ii, jj = np.triu_indices(m, 1)
        S = np.zeros((m, m, len(TILT_AXES)))
        S[ii, jj] = Z[pos:pos + npair]
        S[jj, ii] = Z[pos:pos + npair]
        pos += npair
        trio = dist._trio_sums()
        keys = list(trio.keys())
        lp = np.log(np.maximum(np.array([trio[k] for k in keys]), 1e-300))
        kk = np.array(keys)
        St = S[kk[:, 0], kk[:, 1]] + S[kk[:, 0], kk[:, 2]] + S[kk[:, 1], kk[:, 2]]
        top = tuple(sorted(np.argsort(fin[a_:b_])[:3].tolist()))
        act = keys.index(top) if top in trio else -1
        um = dist.all_umaren()
        uk = list(um.keys())
        ulp = np.log(np.maximum(np.array([um[k] for k in uk]), 1e-300))
        ukk = np.array(uk)
        Su = S[ukk[:, 0], ukk[:, 1]]
        utop = tuple(sorted(np.argsort(fin[a_:b_])[:2].tolist()))
        uact = uk.index(utop) if utop in um else -1
        out.append({"year": int(years[a_]), "rid": str(rid[a_]), "hn": T["horse_number"].to_numpy()[a_:b_].astype(int), "keys": kk, "ukeys": ukk,
                    "lp": lp, "S": St, "act": act, "ulp": ulp, "uS": Su, "uact": uact})
    return out


def _nll(g, races, key):
    from scipy.special import logsumexp

    tot, n = 0.0, 0
    grad = np.zeros_like(g)
    for r in races:
        lp, S, act = r[key[0]], r[key[1]], r[key[2]]
        if act < 0:
            continue
        sc = lp + S @ g
        lse = logsumexp(sc)
        tot -= sc[act] - lse
        p = np.exp(sc - lse)
        grad -= S[act] - p @ S
        n += 1
    return tot / n, grad / n


def _fit_gamma(tr, key, idx, k):
    from scipy.optimize import minimize

    def f(gv):
        g = np.zeros(k)
        g[idx] = gv
        v, gr = _nll(g, tr, key)
        return v + 1e-4 * float(gv @ gv), gr[idx] + 2e-4 * gv

    r0 = minimize(f, np.zeros(len(idx)), jac=True, method="L-BFGS-B")
    g = np.zeros(k)
    g[idx] = r0.x
    return g


def cmd_bet(a=None) -> None:
    """3連複・馬連の買い目を『確率の上位 k 点』で選ぶとき、類似度の補正（2022-23 で当てはめ）を入れた選び方と入れない選び方で、2024 年の的中率・回収率を比べる。"""
    from src.scraper import payoff_repair as pr
    from src.scripts.data import tp36_final2026 as f26

    Pp = pd.read_parquet(OUT / "pairs.parquet")
    T = pd.read_parquet(OUT / "runners.parquet")
    arrs = _race_arrays(T, Pp)
    tr = [r for r in arrs if r["year"] in (2022, 2023)]
    te = [r for r in arrs if r["year"] == 2024]
    k = len(TILT_AXES)
    pj = pd.read_parquet(TABLES / "2024/race_result_flat.parquet", columns=["race_id", "payoff"]).drop_duplicates("race_id")
    fix = pr.load_fix(2024)
    hits = {str(r.race_id): f26.parse_payoff_json(r.payoff, fix.get(str(r.race_id))) for r in pj.itertuples(index=False)}
    res: dict = {}
    for name, key, bt, kk_key in (("3連複", ("lp", "S", "act"), "sanrenpuku", "keys"), ("馬連", ("ulp", "uS", "uact"), "umaren", "ukeys")):
        g = _fit_gamma(tr, key, list(range(k)), k)
        res[name] = {"gamma": {TILT_AXES[i]: float(g[i]) for i in range(k)}}
        for K in (1, 3, 5, 10, 20):
            rows = []
            for r in te:
                h = hits.get(r["rid"])
                if not h or bt in h["_broken"] or r[key[2]] < 0:
                    continue
                pay = h.get(bt, {})
                keys = r[kk_key]
                hn = r["hn"]
                base_sc = r[key[0]]
                til_sc = r[key[0]] + r[key[1]] @ g
                out = {"race_id": r["rid"]}
                for tag, sc in (("base", base_sc), ("tilt", til_sc)):
                    top = np.argsort(-sc)[:K]
                    ret, hit = 0.0, 0.0
                    for ix in top:
                        combo = tuple(sorted(int(hn[x]) for x in keys[ix]))
                        if ix == r[key[2]]:
                            hit = 1.0
                            ret = float(pay.get(combo, 0.0)) / 100.0
                    out[f"{tag}_hit"], out[f"{tag}_ret"], out["stake"] = hit, ret, float(min(K, len(sc)))
                rows.append(out)
            D = pd.DataFrame(rows)
            rng = np.random.default_rng(0)
            n = len(D)
            bs = []
            for _ in range(400):
                ix = rng.integers(0, n, n)
                x = D.iloc[ix]
                bs.append((x["base_ret"].sum() / x["stake"].sum(), x["tilt_ret"].sum() / x["stake"].sum()))
            bs = np.array(bs)
            d = bs[:, 1] - bs[:, 0]
            res[name][f"top{K}"] = {"races": int(n), "hit_base": float(D["base_hit"].mean()), "hit_tilt": float(D["tilt_hit"].mean()),
                                    "roi_base": float(D["base_ret"].sum() / D["stake"].sum()), "roi_tilt": float(D["tilt_ret"].sum() / D["stake"].sum()),
                                    "roi_base_ci": [float(np.percentile(bs[:, 0], 2.5)), float(np.percentile(bs[:, 0], 97.5))],
                                    "roi_tilt_ci": [float(np.percentile(bs[:, 1], 2.5)), float(np.percentile(bs[:, 1], 97.5))],
                                    "roi_diff": float(D["tilt_ret"].sum() / D["stake"].sum() - D["base_ret"].sum() / D["stake"].sum()), "roi_diff_ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
            x = res[name][f"top{K}"]
            _log(f"{name} 上位{K}点: 的中 {x['hit_base']:.4f}→{x['hit_tilt']:.4f}  回収率 {x['roi_base']:.3f}→{x['roi_tilt']:.3f} (差 {x['roi_diff']:+.3f} [{x['roi_diff_ci'][0]:+.3f},{x['roi_diff_ci'][1]:+.3f}])")
    _jsave(OUT / "bet.json", res)


def cmd_joint(a=None) -> None:
    from scipy.optimize import minimize

    Pp = pd.read_parquet(OUT / "pairs.parquet")
    T = pd.read_parquet(OUT / "runners.parquet")
    arrs = _race_arrays(T, Pp)
    _log(f"レース {len(arrs)}")
    res: dict = {}
    for name, key in (("3連複", ("lp", "S", "act")), ("馬連", ("ulp", "uS", "uact"))):
        tr = [r for r in arrs if r["year"] in (2022, 2023)]
        te = [r for r in arrs if r["year"] == 2024]
        k = len(TILT_AXES)
        # 軸ごと（単独）と全部
        sets = {f"単独:{ax}": [i] for i, ax in enumerate(TILT_AXES)}
        sets["全部"] = list(range(k))
        sets["d_str4抜きの全部"] = [i for i, ax in enumerate(TILT_AXES) if ax != "d_str4"]
        sets["d_str4+d_apt+d_gate+d_sty+both_front"] = [TILT_AXES.index(x) for x in ("d_str4", "d_apt", "d_gate", "d_sty", "both_front")]
        res[name] = {}
        for tag, idx in sets.items():
            mask = np.zeros(k)
            mask[idx] = 1.0

            def f(gv):
                g = np.zeros(k)
                g[idx] = gv
                v, gr = _nll(g, tr, key)
                return v + 1e-4 * float(gv @ gv), gr[idx] + 2e-4 * gv

            r0 = minimize(f, np.zeros(len(idx)), jac=True, method="L-BFGS-B")
            g = np.zeros(k)
            g[idx] = r0.x
            nll_tr0, _ = _nll(np.zeros(k), tr, key)
            nll_tr1, _ = _nll(g, tr, key)
            nll_te0, _ = _nll(np.zeros(k), te, key)
            nll_te1, _ = _nll(g, te, key)
            # 年ごとの差（2024 のレース単位ブートストラップ用に、レースごとの差も出す）
            diffs = []
            for r in te:
                if r[key[2]] < 0:
                    continue
                from scipy.special import logsumexp

                s0, s1 = r[key[0]], r[key[0]] + r[key[1]] @ g
                diffs.append((s1[r[key[2]]] - logsumexp(s1)) - (s0[r[key[2]]] - logsumexp(s0)))
            diffs = np.array(diffs)
            rng = np.random.default_rng(0)
            bs = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(400)])
            res[name][tag] = {"gamma": {TILT_AXES[i]: float(g[i]) for i in idx}, "train_nll": [nll_tr0, nll_tr1], "test2024_nll": [nll_te0, nll_te1],
                              "d_test": float(nll_te1 - nll_te0), "ci": [float(-np.percentile(bs, 97.5)), float(-np.percentile(bs, 2.5))], "n_test": int(len(diffs))}
            _log(f"{name} {tag}: 2024 の対数損失 {nll_te0:.5f} → {nll_te1:.5f} (Δ{nll_te1 - nll_te0:+.5f}, CI [{res[name][tag]['ci'][0]:+.5f}, {res[name][tag]['ci'][1]:+.5f}]) γ={res[name][tag]['gamma']}")
    _jsave(OUT / "joint.json", res)


# ───────────────────────── 特徴量
def cmd_groups(a=None) -> None:
    from src.scripts.data import tp27_eval as t

    OUT.mkdir(parents=True, exist_ok=True)
    df = t.setup()[0]
    T, X, hi = load_all()
    t0 = time.time()
    F = ts.runner_features(T, X, hi)
    _log(f"特徴量 {F.shape} ({time.time() - t0:.0f}s)")
    keys = df[KEY].reset_index(drop=True)
    gdir = OUT / "groups"
    gdir.mkdir(parents=True, exist_ok=True)
    for g, cols in ts.GROUP_COLS.items():
        G = keys.merge(F[KEY + cols], on=KEY, how="left")
        nn = G[cols].notna().mean()
        keep = [c for c in cols if nn[c] > 0.02 and G[c].nunique(dropna=True) > 1]
        G[KEY + keep].to_parquet(gdir / f"{g}.parquet", index=False)
        _log(f"{g}: {len(keep)} 列（除外 {len(cols) - len(keep)}）、被覆 {float(G[keep].notna().mean().mean()):.3f}")


def _load_group(g: str, df: pd.DataFrame) -> pd.DataFrame:
    G = pd.read_parquet(OUT / "groups" / f"{g}.parquet")
    G["race_id"], G["horse_id"] = G["race_id"].astype(str), G["horse_id"].astype(str)
    m = df[KEY].merge(G, on=KEY, how="left")
    m = m[[c for c in m.columns if c not in KEY]]
    m.index = df.index
    return m


def _frame(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.read_parquet(FRAME31 / "v4_frame.parquet")
    if len(X) != len(df):
        raise RuntimeError(f"V4 の特徴量表の行数が合わない: {len(X)} vs {len(df)}")
    X.index = df.index
    for c in X.columns:
        if str(X[c].dtype) in ("object", "string"):
            X[c] = X[c].astype("category")
    return X


def _run_subset(name, X, meta, folds, lp, years, seeds):
    from src.pipeline.models import zoo_runner

    fs_ = [f for f in folds if int(meta["year"].to_numpy()[f.valid_index][0]) in years]
    _wait()
    t0 = time.time()
    res, oof = zoo_runner.run_oof(name, "lgb_binary", X, meta, fs_, seeds, THREADS, params=dict(lp))
    lv = {mm: float(res[mm].mean()) for mm in ("logloss_win", "logloss_top2", "logloss_top3")}
    _log(f"{name}: win {lv['logloss_win']:.5f} top2 {lv['logloss_top2']:.5f} top3 {lv['logloss_top3']:.5f} cols={X.shape[1]} ({time.time() - t0:.0f}s)")
    return lv, oof


def _stage(label: str, years: list[int], seeds: list[int], groups: list[str], src31: str | None) -> None:
    import shutil

    from src.scripts.data import tp27_eval as t

    df, base, meta, folds, lp = t.setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    X0 = _frame(df)
    d = OUT / label
    d.mkdir(parents=True, exist_ok=True)
    resp = d / "results.json"
    res = _jload(resp)
    bp = d / "base.parquet"
    if not bp.exists() and src31 and (FRAME31 / src31 / "base.parquet").exists():
        shutil.copy(FRAME31 / src31 / "base.parquet", bp)
        r31 = _jload(FRAME31 / src31 / "results.json")
        res["base"] = r31.get("base", {})
        res["base_from"] = f"tp31/{src31}"
        _jsave(resp, res)
    if bp.exists() and "base" in res:
        oof0 = pd.read_parquet(bp)
    else:
        lv, oof0 = _run_subset("V4", X0, meta, folds, lp, years, seeds)
        oof0.to_parquet(bp, index=False)
        res["base"] = {"levels": lv, "cols": int(X0.shape[1])}
        _jsave(resp, res)
    steps = res.setdefault("groups", {})
    for g in groups:
        if g in steps:
            continue
        G = pd.concat([_load_group(x, df) for x in groups if x != "ALL_PASSED"], axis=1) if g == "ALL_PASSED" else _load_group(g, df)
        Xn = pd.concat([X0, G], axis=1)
        lv, oof = _run_subset(f"V4+{g}", Xn, meta, folds, lp, years, seeds)
        c = t._cmp(oof0, oof, fin_map, g)
        steps[g] = {"cols": int(G.shape[1]), "levels": lv, "vs_V4": c}
        _jsave(resp, res)
        _log(f"  {g}: Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f} 採用={c['採用']}")


def cmd_stage1(a=None) -> None:
    _stage("stage1", [2022, 2023], [0], list(ts.GROUPS), "stage1")


def cmd_stage2(a=None) -> None:
    allg = (_jload(OUT / "stage1" / "results.json") or {}).get("groups", {})
    passed = [g for g in GROUP_NAMES if g in allg and allg[g]["vs_V4"]["Δwin"] < -0.00012 and allg[g]["vs_V4"]["win_hi"] < 0.0001]
    _log(f"第 1 段を通過: {passed}")
    _jsave(OUT / "stage1_passed.json", passed)
    if not passed:
        return
    _stage("stage2", [2024], [2, 3], passed + (["ALL_PASSED"] if len(passed) > 1 else []), "stage2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pairs", "describe", "calib", "joint", "bet", "groups", "stage1", "stage2"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"pairs": cmd_pairs, "describe": cmd_describe, "calib": cmd_calib, "joint": cmd_joint, "bet": cmd_bet, "groups": cmd_groups, "stage1": cmd_stage1, "stage2": cmd_stage2}[a.cmd](a)


if __name__ == "__main__":
    main()
