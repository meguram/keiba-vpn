"""tp-38: 血統の『位置・存在割合・クロス・類似種牡馬』の特徴量の構築と評価。

python -m src.scripts.data.tp38_eval static      # 血統表の長い表・クロスの静的量・血統空間の座標とクラスタ・種牡馬の類似度（近傍）を tp38/static/ に保存
python -m src.scripts.data.tp38_eval build       # 群ごとの特徴量を tp38/groups/*.parquet に保存（F1 cell / cross / emb / clu / sim）
python -m src.scripts.data.tp38_eval ridge       # F2（リッジ + 位置の重みの推定）の群を保存（tp38_ridge.py）
python -m src.scripts.data.tp38_eval position    # 位置別の効果の減衰（祖先の効果が世代・位置ごとにどれだけ薄まるか）の分析
python -m src.scripts.data.tp38_eval simcheck    # 類似種牡馬の検出例（人が見て妥当か確認する用）
python -m src.scripts.data.tp38_eval stage1      # 第 1 段: valid 2022+2023、seed 1 本。V4 全体に対する増分
python -m src.scripts.data.tp38_eval stage2      # 第 2 段: 通過群を valid 2024、新 seed 2 本で確認 + 同時追加
python -m src.scripts.data.tp38_eval report      # /tmp/tp38_report.{json,md}
test 2025・2026 は使わない。市場オッズ・人気は特徴量に使わない。環境変数: TP38_THREADS（既定 6）, TP38_LOAD_LIMIT（既定 16）, TP38_OUT（出力先。既定は training_pipeline_v4/tp38）。
"""

from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", os.environ.get("TP38_POLARS_THREADS", "6"))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.pipeline.features import tp38_pedigree as P  # noqa: E402
from src.scripts.data import tp35_eval as e35  # noqa: E402

ROOT = e35.ROOT
V = e35.V
OUT = Path(os.environ.get("TP38_OUT") or V / "tp38")
STATIC = OUT / "static"
t = e35.t
KEY = ["race_id", "horse_id"]
THREADS = int(os.environ.get("TP38_THREADS", 6))
LOAD_LIMIT = float(os.environ.get("TP38_LOAD_LIMIT", 16))
GROUPS = ["PX_cell_sd", "PX_cell_vn", "PX_cell_gd", "PX_cellw_sd", "PX_cell_all", "PX_cross", "PX_emb", "PX_clu", "PX_sim", "PX_ridge_fix", "PX_ridge_als", "PX_ridge_als2", "PX_cell_sf", "PX_emb32"]
ALL_SETS = {"ALL_BEST": ["PX_cell_sd", "PX_emb", "PX_ridge_als"]}                # 各系統（F1 / 血統空間 / F2）で第 1 段が最良だった群の同時追加
_jsave, _jload = e35._jsave, e35._jload


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _wait() -> None:
    t0 = time.time()
    while os.getloadavg()[0] >= LOAD_LIMIT and time.time() - t0 < 3600:
        time.sleep(30)


# ───────────────────────── 入力
def load_inputs():
    from src.pipeline.features import factory_ped as fp

    df, base, meta, folds, lp = t.setup()
    A, H = fp.build_ped_tables(df["horse_id"].unique())
    A["horse_id"] = A["horse_id"].astype(str)
    horse_index = pd.Index(sorted(set(A["horse_id"])))
    anc, uniq = P.prep_ancestors(A, horse_index)
    return df, base, meta, folds, lp, anc, uniq, horse_index


def _read_names(horse_index: pd.Index, uniq: np.ndarray) -> dict[str, str]:
    """祖先 ID → 名前（表示用。血統 JSON の ancestors.name）。世代 3 までを拾う。"""
    want = set(map(str, uniq))
    names: dict[str, str] = {}
    for h in horse_index:
        for d in ("horse_pedigree_5gen", "horse_pedigree_10gen"):
            f = ROOT / "data/local" / d / h[:4] / f"{h}.json"
            if f.exists():
                try:
                    js = json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    continue
                for a in js.get("ancestors") or []:
                    i = str(a.get("horse_id") or "")
                    if i in want and i not in names and (a.get("generation") or 9) <= 3:
                        names[i] = str(a.get("name") or "")
                break
    return names


def cmd_static(a) -> None:
    import scipy.sparse as sp

    STATIC.mkdir(parents=True, exist_ok=True)
    df, base, meta, folds, lp, anc, uniq, horse_index = load_inputs()
    anc.to_parquet(STATIC / "anc.parquet", index=False)
    pd.Series(horse_index).to_frame("horse_id").to_parquet(STATIC / "horses.parquet", index=False)
    pd.Series(uniq).to_frame("anc_id").to_parquet(STATIC / "ancestors.parquet", index=False)
    _log(f"祖先表 {len(anc):,} 行、馬 {len(horse_index):,}、祖先 {len(uniq):,}")
    H, kinds = P.cross_static(anc, len(horse_index))
    H.to_parquet(STATIC / "cross_H.parquet")
    kinds.to_parquet(STATIC / "cross_kinds.parquet", index=False)
    _log(f"クロス: 近交あり {(H['x_F'] > 0).mean():.3f}、近交係数の平均 {H['x_F'].mean():.5f}、最大 {H['x_F'].max():.4f}")
    X, idf = P.horse_vectors(anc, len(horse_index), len(uniq))
    E, cl = P.embed_and_cluster(X)
    np.save(STATIC / "emb.npy", E)
    np.savez(STATIC / "clusters.npz", **{f"k{k}": v for k, v in cl.items()})
    _log(f"埋め込み {E.shape}、クラスタ {list(cl)}")
    sa, XS, depth = P.stallion_vectors(anc, len(uniq))
    np.save(STATIC / "stallion_codes.npy", sa)
    sp.save_npz(STATIC / "stallion_X.npz", XS)
    np.save(STATIC / "stallion_depth.npy", depth)
    _log(f"種牡馬ベクトル {XS.shape}（血統表を復元できた男系祖先）")
    nbs = {}
    for role, pc in P.ROLE_PCLS.items():
        ids = np.sort(anc.loc[anc["pcls"] == pc, "a"].unique())
        nb = P.neighbors(sa, XS, ids, ids)
        nb.to_parquet(STATIC / f"nb_{role}.parquet", index=False)
        nbs[role] = {"stallions": int(len(ids)), "with_vector": int(np.isin(ids, sa).sum()), "with_neighbors": int(nb["a"].nunique()), "mean_sim": float(nb["sim"].mean()), "mean_k": float(len(nb) / max(nb["a"].nunique(), 1))}
        _log(f"近傍 {role}: {nbs[role]}")
    names = _read_names(horse_index, uniq)
    (STATIC / "names.json").write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    _jsave(STATIC / "static_meta.json", {"anc_rows": int(len(anc)), "horses": int(len(horse_index)), "ancestors": int(len(uniq)), "cross_share": float((H["x_F"] > 0).mean()), "neighbors": nbs, "names": len(names)})


def _load_static():
    anc = pd.read_parquet(STATIC / "anc.parquet")
    horse_index = pd.Index(pd.read_parquet(STATIC / "horses.parquet")["horse_id"])
    uniq = pd.read_parquet(STATIC / "ancestors.parquet")["anc_id"].to_numpy()
    return anc, horse_index, uniq


def add_rel(F: pd.DataFrame, race: pd.Series, cols: list[str]) -> pd.DataFrame:
    """列の『同じレースの平均との差』を足す（レース内の相対）。"""
    g = F[cols].groupby(race.to_numpy())
    rel = F[cols] - g.transform("mean")
    rel.columns = [f"{c}_rel" for c in cols]
    return pd.concat([F, rel], axis=1)


def _save_group(name: str, F: pd.DataFrame, df: pd.DataFrame) -> dict:
    d = OUT / "groups"
    d.mkdir(parents=True, exist_ok=True)
    F = F.copy()
    F.index = df.index
    G = pd.concat([df[KEY].reset_index(drop=True), F.reset_index(drop=True)], axis=1)
    G.to_parquet(d / f"{name}.parquet", index=False)
    cov = float(F.notna().any(axis=1).mean())
    _log(f"{name}: {F.shape[1]} 列、被覆 {cov:.3f}")
    return {"cols": int(F.shape[1]), "coverage": cov, "columns": list(F.columns)}


def cmd_build(a) -> None:
    df, base, meta, folds, lp = t.setup()
    anc, horse_index, uniq = _load_static()
    R = P.Runs(df, horse_index)
    race = df["race_id"]
    meta_out = _jload(OUT / "build_meta.json")
    todo = [g for g in (a.only.split(",") if getattr(a, "only", None) else GROUPS) if not g.startswith("PX_ridge")]
    for g in todo:
        t0 = time.time()
        _wait()
        if g.startswith("PX_cell"):
            cond = g.split("_")[-1]
            y = P.win_y(df) if g.startswith("PX_cellw") else None
            F = P.cell_features(R, anc, cond, f"pxc{'w' if y is not None else ''}_{cond}", y=y)
            F = add_rel(F, race, [c for c in F.columns if c.endswith(("_w", "_r"))])
        elif g == "PX_cross":
            H = pd.read_parquet(STATIC / "cross_H.parquet")
            kinds = pd.read_parquet(STATIC / "cross_kinds.parquet")
            hs = np.where(R.h >= 0, R.h, 0)
            S = H.iloc[hs].reset_index(drop=True)
            S.loc[R.h < 0, :] = np.nan
            F = pd.concat([S.astype("float32"), P.cross_asof(R, anc, kinds, "all", tag="xe"), P.cross_asof(R, anc, kinds, "sd", tag="xs")], axis=1)
        elif g in ("PX_emb", "PX_emb32"):
            E = np.load(STATIC / "emb.npy")
            hs = np.where(R.h >= 0, R.h, 0)
            nd = 32 if g == "PX_emb32" else 16
            F = pd.DataFrame(E[hs][:, :nd], columns=[f"pxe_{i}" for i in range(nd)])
            F.loc[R.h < 0, :] = np.nan
        elif g == "PX_clu":
            cl = np.load(STATIC / "clusters.npz")
            parts = []
            for k in (64, 256):
                for cond in ("all", "sd"):
                    parts.append(P.cluster_asof(R, cl[f"k{k}"], cond, f"pxk{k}_{cond}"))
            F = pd.concat(parts, axis=1)
            F = add_rel(F, race, [c for c in F.columns if c.endswith("_m")])
        elif g == "PX_sim":
            nb = {role: pd.read_parquet(STATIC / f"nb_{role}.parquet") for role in P.ROLE_PCLS}
            parts = [P.sim_features(R, anc, nb, cond, f"pxs_{cond}") for cond in ("all", "sd")]
            F = pd.concat(parts, axis=1)
            F = add_rel(F, race, [c for c in F.columns if c.endswith("_sim")])
        else:
            continue
        meta_out[g] = _save_group(g, F, df) | {"seconds": round(time.time() - t0)}
        _jsave(OUT / "build_meta.json", meta_out)


# ───────────────────────── 位置別の効果の減衰
def cmd_position(a) -> None:
    """祖先 a の『直接の産駒（父・母父）での効果』と『別の位置にいるときの効果』の関係を、2024 年末までのデータで推定する。
    各祖先 a×区分 pcls について、その区分の産駒の平均 y（レース内の着順スコア）を出し、a が父として持つ効果（pcls 0）に対する回帰の傾き = 位置ごとの減衰率。
    理論値は 2^-(gen-1)（産駒が祖先の遺伝子の 1/2、孫が 1/4...）。件数の重み付き最小二乗（傾き）と相関を出す。"""
    df, base, meta, folds, lp = t.setup()
    anc, horse_index, uniq = _load_static()
    R = P.Runs(df, horse_index)
    d = df[(df["date"] < "2025-01-01")]
    y = pd.Series(R.y, index=df.index).loc[d.index]
    hh = pd.Series(R.h, index=df.index).loc[d.index]
    runs = pd.DataFrame({"h": hh.to_numpy(), "y": y.to_numpy()}).dropna()
    runs = runs[runs["h"] >= 0]
    ap = anc[["h", "a", "pcls", "w"]].groupby(["h", "a", "pcls"], sort=False)["w"].sum().reset_index()
    m = runs.merge(ap, on="h", how="inner")
    cell = m.groupby(["a", "pcls"])["y"].agg(["sum", "count"]).reset_index()
    cell["mean"] = cell["sum"] / cell["count"]
    # 参照: 父としての効果（pcls 0）。母父（pcls 4）も参照にして、母父を軸にしたときの減衰も出す
    out = {}
    for ref_pc, ref_name in ((0, "sire"), (4, "dam_sire")):
        ref = cell[(cell["pcls"] == ref_pc) & (cell["count"] >= 100)].set_index("a")
        rows = []
        for pc in range(P.N_PC):
            c = cell[(cell["pcls"] == pc) & (cell["a"].isin(ref.index)) & (cell["count"] >= 30)]
            if len(c) < 20:
                continue
            x = ref.loc[c["a"], "mean"].to_numpy()
            yv = c["mean"].to_numpy()
            wt = c["count"].to_numpy().astype(float)
            xm, ym = np.average(x, weights=wt), np.average(yv, weights=wt)
            slope = float(np.sum(wt * (x - xm) * (yv - ym)) / np.sum(wt * (x - xm) ** 2))
            corr = float(np.corrcoef(x, yv)[0, 1])
            rows.append({"pcls": pc, "gen": int(P.PC_GEN[pc]), "side": int(P.PC_SIDE[pc]), "tm": int(P.PC_TM[pc]), "tf": int(P.PC_TF[pc]), "n_anc": int(len(c)), "slope": slope, "corr": corr, "mean_runs": float(c["count"].mean())})
        out[ref_name] = rows
    _jsave(OUT / "position.json", out)
    for k, rows in out.items():
        _log(f"== 参照 {k}")
        for r in rows:
            _log(f"  pcls {r['pcls']:2d} gen{r['gen']} side{r['side']} tm{r['tm']} tf{r['tf']} 祖先 {r['n_anc']:4d} 傾き {r['slope']:+.3f} 相関 {r['corr']:+.3f}")


def cmd_screen(a) -> None:
    """各群の列が、valid 年（2022〜2024）の成績 y（レース内の着順スコア）とどれだけ相関するか。レース内の順位相関で見る（V4 の既存の血統列と並べる）。"""
    df, base, meta, folds, lp = t.setup()
    X0 = e35._frame(df)
    y = pd.Series(P.res_y(df), index=df.index)
    m = ((df["date"] >= "2022-01-01") & (df["date"] < "2025-01-01") & y.notna()).to_numpy()
    race = df["race_id"].to_numpy()

    def within(x: pd.Series) -> float:
        xx = pd.DataFrame({"r": race[m], "x": x.to_numpy()[m], "y": y.to_numpy()[m]})
        xx["x"] = xx.groupby("r")["x"].rank(pct=True)
        xx["y"] = xx.groupby("r")["y"].rank(pct=True)
        ok = xx["x"].notna()
        return float(np.corrcoef(xx.loc[ok, "x"], xx.loc[ok, "y"])[0, 1])

    out = {"V4": {}}
    for c in [c for c in X0.columns if c.startswith(("pd_sire_sd", "pd_dam_sire_sd", "pd_sire_sf", "pd_dam_sire_sf"))][:8]:
        out["V4"][c] = within(X0[c].astype(float))
    for g in GROUPS:
        if not (OUT / "groups" / f"{g}.parquet").exists():
            continue
        G = _load_group(g, df)
        r = {c: within(G[c].astype(float)) for c in G.columns if G[c].notna().any() and G[c].std() > 0}
        out[g] = dict(sorted(r.items(), key=lambda kv: -abs(kv[1]))[:8])
    _jsave(OUT / "screen.json", out)
    for g, r in out.items():
        _log(f"== {g}: " + ", ".join(f"{c}={v:+.3f}" for c, v in r.items()))


def cmd_phi(a) -> None:
    """位置の重み φ（正規化: 父 = 1）と、位置ごとの『予測と実際の成績の共分散』の寄与（クロスフィットで、学習に使っていない馬・日付の走）をまとめる。"""
    from src.pipeline.features import tp38_ridge as RG

    d = _jload(OUT / "ridge_phi.json")
    ph = d["phis"]
    last = [p for p in ph if p["T"] >= "2024-01-01"]            # 学習データが多い後半の作り直し
    out = {"n_fits": len(ph), "n_last": len(last), "positions": []}
    for variant, key in (("als", "phi"), ("als2", "phi2")):
        phis = np.array([p[key] for p in last])
        out[f"phi_{variant}"] = (phis.mean(axis=0) / phis.mean(axis=0)[0]).tolist()
    for v in ("fix", "als", "als2"):
        covs = np.array([p[f"imp_{v}"]["cov"] for p in last if p.get(f"imp_{v}")])
        sds = np.array([p[f"imp_{v}"]["sd"] for p in last if p.get(f"imp_{v}")])
        out[f"cov_{v}"] = covs.mean(axis=0).tolist()
        out[f"sd_{v}"] = sds.mean(axis=0).tolist()
    for i in range(RG.N_POS):
        g, ps = int(RG.POS_GEN[i]), int(RG.POS_POS[i])
        out["positions"].append({"i": i, "gen": g, "pos": ps, "name": P.pos_name(g, ps), "prior": float(RG.POS_PRIOR[i]), "phi_als": out["phi_als"][i], "phi_als2": out["phi_als2"][i], "cov_fix": out["cov_fix"][i], "cov_als": out["cov_als"][i], "cov_als2": out["cov_als2"][i]})
    _jsave(OUT / "phi_summary.json", out)
    tot = {v: float(np.sum(out[f"cov_{v}"])) for v in ("fix", "als", "als2")}
    _log(f"全位置の共分散の合計: {tot}")
    for r in out["positions"][:14] + out["positions"][-3:]:
        _log(f"  {r['gen']}世代 {r['name']:>6s}: 理論 {r['prior']:.3f} φ(als) {r['phi_als']:.3f} φ(als2) {r['phi_als2']:.3f} 共分散シェア fix {r['cov_fix'] / tot['fix']:.3f} als {r['cov_als'] / tot['als']:.3f} als2 {r['cov_als2'] / tot['als2']:.3f}")
    # 世代別・側別のシェア
    for v in ("fix", "als", "als2"):
        c = np.array(out[f"cov_{v}"])
        by_gen = {g: float(c[RG.POS_GEN == g].sum() / c.sum()) for g in range(1, 6)}
        by_side = {"父側": float(c[RG.POS_SIDE == 0].sum() / c.sum()), "母側": float(c[RG.POS_SIDE == 1].sum() / c.sum())}
        _log(f"  {v}: 世代別シェア {by_gen} 側別 {by_side}")


def cmd_simcheck(a) -> None:
    anc, horse_index, uniq = _load_static()
    names = json.loads((STATIC / "names.json").read_text(encoding="utf-8"))
    ids = {i: str(u) for i, u in enumerate(uniq)}
    cnt = anc[anc["pcls"] == 0].groupby("a").size().sort_values(ascending=False)
    nb = pd.read_parquet(STATIC / "nb_sire.parquet")
    out = []
    for aa in cnt.index[:40]:
        sub = nb[nb["a"] == aa].sort_values("sim", ascending=False).head(5)
        nm = lambda c: names.get(ids[int(c)], ids[int(c)])  # noqa: E731
        out.append({"stallion": nm(aa), "runs": int(cnt[aa]), "neighbors": [{"name": nm(r.j), "sim": round(float(r.sim), 3)} for r in sub.itertuples()]})
    _jsave(OUT / "simcheck.json", out)
    for o in out[:25]:
        _log(f"{o['stallion']}({o['runs']}): " + ", ".join(f"{n['name']}({n['sim']})" for n in o["neighbors"]))


# ───────────────────────── 評価（第 1 段・第 2 段）
def _load_group(g: str, df: pd.DataFrame) -> pd.DataFrame:
    T = pd.read_parquet(OUT / "groups" / f"{g}.parquet")
    T["race_id"], T["horse_id"] = T["race_id"].astype(str), T["horse_id"].astype(str)
    m = df[KEY].merge(T, on=KEY, how="left")
    m = m[[c for c in m.columns if c not in KEY]]
    m.index = df.index
    return m


def _stage(label: str, years: list[int], seeds: list[int], groups: list[str]) -> None:
    df, base, meta, folds, lp = t.setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    X0 = e35._frame(df)
    d = OUT / label
    d.mkdir(parents=True, exist_ok=True)
    resp = d / "results.json"
    res = _jload(resp)
    bp = d / "base.parquet"
    if bp.exists() and "base" in res:
        oof0 = pd.read_parquet(bp)
    else:
        lv, oof0 = e35._run_subset("V4", X0, meta, folds, lp, years, seeds)
        oof0.to_parquet(bp, index=False)
        res["base"] = {"levels": lv, "cols": int(X0.shape[1])}
        _jsave(resp, res)
    steps = res.setdefault("groups", {})
    for g in groups:
        if g in steps:
            continue
        if g.startswith("ALL"):
            names = ALL_SETS.get(g) or [x for x in groups if not x.startswith("ALL")]
            G = pd.concat([_load_group(x, df) for x in names], axis=1)
        else:
            if not (OUT / "groups" / f"{g}.parquet").exists():
                continue
            G = _load_group(g, df)
        lv, oof = e35._run_subset(f"V4+{g}", pd.concat([X0, G], axis=1), meta, folds, lp, years, seeds)
        c = t._cmp(oof0, oof, fin_map, g)
        steps[g] = {"cols": int(G.shape[1]), "levels": lv, "vs_V4": c}
        _jsave(resp, res)
        _log(f"  {g}: Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f} 採用={c['採用']}")


def cmd_stage1(a) -> None:
    _stage("stage1", [2022, 2023], [0], [g for g in GROUPS])


def cmd_stage2(a) -> None:
    r = _jload(OUT / "stage1/results.json")
    passed = [g for g, s in r.get("groups", {}).items() if s["vs_V4"]["Δwin"] < -0.00012 and s["vs_V4"]["win_hi"] < 0.0001]
    _log(f"第 1 段を通過: {passed}")
    _jsave(OUT / "stage1/passed.json", passed)
    if not passed:
        return
    order = [g for g in ("PX_cell_sd", "PX_emb", "PX_ridge_als") if g in passed]
    best = ["ALL_BEST"] if len(order) > 1 else []
    rest = [g for g in passed if g not in order]
    _stage("stage2", [2024], [2, 3], order + best + rest + (["ALL_PASSED"] if len(passed) > 1 else []))


def cmd_report(a) -> None:
    s1, s2 = _jload(OUT / "stage1/results.json").get("groups", {}), _jload(OUT / "stage2/results.json").get("groups", {})
    lines = ["# tp-38 血統の位置・クロス・類似種牡馬", "", "| 群 | 列数 | 第 1 段 Δwin(上限) | 第 2 段 Δwin(上限) | Δtop2 | Δtop3 | 採否 |", "|---|---|---|---|---|---|---|"]
    for g in GROUPS + ["ALL_PASSED"]:
        x, y = s1.get(g), s2.get(g)
        if not x and not y:
            continue
        a1 = f"{x['vs_V4']['Δwin']:+.5f}({x['vs_V4']['win_hi']:+.5f})" if x else "-"
        b1 = f"{y['vs_V4']['Δwin']:+.5f}({y['vs_V4']['win_hi']:+.5f})" if y else "-"
        c = y["vs_V4"] if y else {}
        lines.append(f"| {g} | {(x or y)['cols']} | {a1} | {b1} | {c.get('Δtop2', float('nan')):+.5f} | {c.get('Δtop3', float('nan')):+.5f} | {('採用' if c.get('採用') else '不採用') if y else '-'} |")
    Path("/tmp/tp38_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["static", "build", "ridge", "position", "screen", "phi", "simcheck", "stage1", "stage2", "report"])
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "ridge":
        from src.scripts.data import tp38_ridge_run

        tp38_ridge_run.main()
        return
    {"static": cmd_static, "build": cmd_build, "position": cmd_position, "screen": cmd_screen, "phi": cmd_phi, "simcheck": cmd_simcheck, "stage1": cmd_stage1, "stage2": cmd_stage2, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
