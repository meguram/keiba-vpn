"""予想家ペルソナのサブモデル（学習）・内部スタッキング・適材適所の重み（ゲーティング）。tp-77。

python -m src.scripts.data.ptrain inputs                                  # ペルソナごとの入力列と重なりを書き出す
python -m src.scripts.data.ptrain matprior                                # B/D を母方クラスタの事前で作り直す（tp76 の rerun_after_complete_10gen）
python -m src.scripts.data.ptrain sub --personas going,bias --recipes lgb_binary,lgb_softmax,linear [--years 2022,2023,2024 --seeds 0,1]
python -m src.scripts.data.ptrain stack                                   # 各ペルソナの学習器 OOF を、年をまたぐ検証で結合（内部スタッキング）
python -m src.scripts.data.ptrain gate                                    # ペルソナ間の重み（固定／クラス別／面×馬場別／縮約つき連続）を比べる
市場オッズ・人気は使わない。test 2025・2026 は --years に入れない限り使わない。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import tp27_eval as t
from src.scripts.data import tp40_eval as e40
from src.scripts.data import v6_eval as ev

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("PTRAIN_OUT", ROOT / "data/local/modeling/ptrain"))
CFG = ROOT / "config" / "persona_inputs.json"
TP76 = Path(os.environ.get("PTRAIN_TP76", "/home/jovyan/work/tp76_out/features.parquet"))
MINING = Path(os.environ.get("PTRAIN_MINING", "/home/jovyan/work/tp72_out/mining.parquet"))
KEY = ["race_id", "horse_id"]
YEARS = [2022, 2023, 2024]
SEEDS = [0, 1]
PERSONAS = ["ability", "lap", "going", "bias", "pedigree"]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _jload(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _jsave(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 入力の割り当て
def _pick(cols: list[str], spec: dict) -> list[str]:
    ex, pf = set(spec.get("exact") or []), tuple(spec.get("prefix") or [])
    return [c for c in cols if c in ex or (pf and c.startswith(pf))]


def persona_cols(frame_cols: list[str], name: str, cfg: dict | None = None) -> list[str]:
    cfg = cfg or json.loads(CFG.read_text(encoding="utf-8"))
    core = _pick(frame_cols, cfg["core"])
    own = _pick(frame_cols, cfg["personas"][name])
    return list(dict.fromkeys(core + own))


def _load_extra(src: str, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """df の行順に並べた追加列。src: tp76 / mining / mat / tp76mat"""
    if src == "tp76":
        G = pd.read_parquet(TP76)
    elif src == "tp76mat":
        G = pd.read_parquet(OUT / "tp76_mat.parquet")
    elif src == "pace":
        G = pd.read_parquet(OUT / "pace_rows.parquet")
    elif src == "sire":
        G = pd.read_parquet(OUT / "sire_rows.parquet")
    elif src == "cross":
        G = pd.read_parquet(OUT / "cross_rows.parquet")
    elif src == "mining":
        G = pd.read_parquet(MINING, columns=KEY + [c for c in cols if c != "__all__"])
    elif src == "mat":
        G = pd.read_parquet(OUT / "mat/features.parquet")
        drop = [c for c in G.columns if c in ("mat_dam_sire",)]
        G = G.drop(columns=drop)
        for c in [c for c in G.columns if c.startswith("mat_cl")]:
            G[c] = G[c].astype("category")
    else:
        raise KeyError(src)
    on = ["horse_id"] if src == "mat" else KEY
    want = [c for c in G.columns if c not in KEY and c != "date" and c != "horse_id"] if cols == ["__all__"] else [c for c in cols if c in G.columns]
    G = G[on + want].copy()
    for k in on:
        G[k] = G[k].astype(str)
    m = df[on].astype(str).merge(G.drop_duplicates(on), on=on, how="left")
    assert len(m) == len(df)
    m = m[want]
    m.index = df.index
    return m


def persona_frame(name: str, Xb: pd.DataFrame, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cols = persona_cols(list(Xb.columns), name, cfg)
    parts = [Xb[cols]]
    for src, cs in (cfg["personas"][name].get("extra") or {}).items():
        try:
            parts.append(_load_extra(src, df, cs))
        except FileNotFoundError:
            _log(f"{name}: 追加列 {src} が無い（スキップ）")
    X = pd.concat(parts, axis=1)
    return X.loc[:, ~X.columns.duplicated()]


# ---------------------------------------------------------------- 準備
def cmd_inputs(a) -> None:
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    fc = [r["col"] for r in _jload(OUT / "frame_cols.json")["cols"]]
    rows, sets = {}, {}
    for p in PERSONAS:
        cs = persona_cols(fc, p, cfg)
        sets[p] = set(cs)
        ex = cfg["personas"][p].get("extra") or {}
        rows[p] = {"label": cfg["personas"][p]["label"], "frame_cols": len(cs), "extra": {k: len(v) for k, v in ex.items()}}
    ov = {}
    for i, p in enumerate(PERSONAS):
        for q in PERSONAS[i + 1:]:
            ov[f"{p}&{q}"] = len(sets[p] & sets[q])
    core = len(_pick(fc, cfg["core"]))
    covered = set().union(*sets.values())
    out = {"core_cols": core, "personas": rows, "overlap_frame_cols": ov, "frame_cols_total": len(fc), "frame_cols_unassigned": len(set(fc) - covered)}
    _jsave(OUT / "persona_inputs.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=1))


def cmd_matprior(a) -> None:
    """tp76 の B（潜在能力）・D（条件別適性）を、種牡馬の事前の代わりに母方クラスタの事前で作り直す。"""
    from src.pipeline.features import tp76_asof as T
    from src.scripts.data import tp76_screen as S

    mat = pd.read_parquet(OUT / "mat/features.parquet")[["horse_id", f"mat_cl{a.k}"]]
    mat["horse_id"] = mat["horse_id"].astype(str)
    df = S.load_mining()
    df["horse_id"] = df["horse_id"].astype(str)
    df = df.merge(mat, on="horse_id", how="left")
    df["sire"] = "c" + df[f"mat_cl{a.k}"].astype("Int64").astype(str)
    d = T.prepare(df)
    p = {**T.PARAMS}
    consts = T.fit_constants(d)
    kal = T.kalman_ability(d, "z_fin", consts, p)
    ap = T.conditional_aptitude(d, kal["innov"], "z", p)
    s = T._sorted_by_horse(d)
    mu_ag, _ = T.ewm_asof(s["horse_id"].to_numpy(), s["agari_pct"].to_numpy(float), 5.0, 0.5, 1.0)
    ag_e = (s["agari_pct"] - pd.Series(mu_ag, index=s.index)).reindex(d.index)
    ap_ag = T.conditional_aptitude(d, ag_e, "ag", p)
    f = pd.DataFrame({"race_id": d["race_id"], "horse_id": d["horse_id"]}, index=d.index)
    f["B_m_mat"], f["B_sd_mat"] = kal["m"], kal["sd"]
    f["D_dist_mat"], f["D_venue_mat"], f["D_going_mat"], f["D_cush_mat"], f["D_all_mat"] = ap["apt_z_dist"], ap["apt_z_venue"], ap["apt_z_going"], ap["apt_z_cush"], ap["apt_z_all"]
    f["D_ag_all_mat"] = ap_ag["apt_ag_all"]
    f.reset_index(drop=True).to_parquet(OUT / "tp76_mat.parquet", index=False)
    _log(f"matprior: {f.shape} 母方クラスタ K={a.k}（欠損 {float(f['D_all_mat'].isna().mean()):.3f}）")


def cmd_matrows(a) -> None:
    """母方の構造特徴（馬単位）を、学習表の行（race_id, horse_id）に展開して保存する（v6_eval の外部群 S77_MAT）。"""
    df, base, meta, folds, lp = t.setup()
    G = _load_extra("mat", df, ["__all__"])
    out = pd.concat([df[KEY].astype(str).reset_index(drop=True), G.reset_index(drop=True)], axis=1)
    out.to_parquet(OUT / "mat_rows.parquet", index=False)
    _log(f"mat_rows: {out.shape}（欠損 {float(out['mat_fill_frac'].isna().mean()):.3f}）")


# ---------------------------------------------------------------- サブモデルの学習
def cmd_sub(a) -> None:
    from src.pipeline.models import zoo_runner

    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    years = [int(x) for x in a.years.split(",")] if a.years else YEARS
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else SEEDS
    tag = a.tag or ("" if years == YEARS else "_y" + "".join(str(y)[2:] for y in years))
    (OUT / "oof").mkdir(parents=True, exist_ok=True)
    df, base, meta, folds, lp = t.setup()
    Xb = ev.build_frame(ev.config_of("z:RB@e"), df)
    fs = [f for f in folds if int(meta["year"].to_numpy()[f.valid_index][0]) in years]
    res = _jload(OUT / f"sub_levels{tag}.json")
    for p in [x for x in a.personas.split(",") if x]:
        if p.startswith("ctrl_"):       # 対照: 同じ列数を基準フレームから無作為に選んだ『偽のペルソナ』（多様性だけの効果を切り分ける）
            n_cols = persona_frame(p[5:], Xb, df, cfg).shape[1]
            rng = np.random.default_rng(1000 + PERSONAS.index(p[5:]))
            X = Xb[list(rng.choice(np.array(Xb.columns), size=min(n_cols, Xb.shape[1]), replace=False))]
        else:
            X = persona_frame(p, Xb, df, cfg)
        _log(f"{p}: 入力 {X.shape[1]} 列")
        for r in [x for x in a.recipes.split(",") if x]:
            name = f"{p}__{r}"
            fp = OUT / "oof" / f"{name}{tag}.parquet"
            if fp.exists():
                _log(f"{name}: 既に完了")
                continue
            params = dict(lp) if r.startswith("lgb") else {}
            e40._wait()
            t0 = time.time()
            try:
                rs, oof = zoo_runner.run_oof(name, r, X, meta, fs, seeds, e40.THREADS, params=params)
            except Exception as exc:  # noqa: BLE001
                _log(f"{name}: 失敗 {type(exc).__name__}: {exc}")
                res[name] = {"error": f"{type(exc).__name__}: {exc}"}
                _jsave(OUT / f"sub_levels{tag}.json", res)
                continue
            oof.to_parquet(fp, index=False)
            lv = {m: float(rs[m].mean()) for m in ("logloss_win", "logloss_top2", "logloss_top3")}
            res[name] = {"levels": lv, "cols": int(X.shape[1]), "years": years, "seeds": seeds, "seconds": round(time.time() - t0)}
            _jsave(OUT / f"sub_levels{tag}.json", res)
            _log(f"{name}: win {lv['logloss_win']:.5f} top2 {lv['logloss_top2']:.5f} top3 {lv['logloss_top3']:.5f} cols={X.shape[1]} ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------- スタッキング・ゲーティング
TARGETS = ("win", "top2", "top3")
TOTAL = {"win": 1.0, "top2": 2.0, "top3": 3.0}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _norm(race: np.ndarray, p: np.ndarray, total: float) -> np.ndarray:
    from src.pipeline.models import win_models as wm

    return wm.normalize_to_total(race, p, total)


def _fit_logit(X: np.ndarray, y: np.ndarray, off: np.ndarray, lam: float, pen_mask: np.ndarray | None = None) -> np.ndarray:
    from src.scripts.data.tp76_screen import fit_logit

    return fit_logit(X, y, off, lam)


def _targets(meta: pd.DataFrame, key: pd.DataFrame) -> dict[str, np.ndarray]:
    fin = meta.set_index(KEY)["finish_position"].loc[list(zip(key["race_id"], key["horse_id"]))].to_numpy()
    return {"win": (fin == 1).astype(float), "top2": (fin <= 2).astype(float), "top3": (fin <= 3).astype(float)}


def _ll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _load_oofs(names: list[str], tag: str = "") -> dict[str, pd.DataFrame]:
    out = {}
    for n in names:
        f = OUT / "oof" / f"{n}{tag}.parquet"
        if f.exists():
            d = pd.read_parquet(f)
            d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
            out[n] = d
    return out


def _align(oofs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    names = list(oofs)
    key = oofs[names[0]][KEY + ["year"]].copy()
    for n in names[1:]:
        key = key.merge(oofs[n][KEY], on=KEY, how="inner")
    key = key.sort_values(KEY, kind="mergesort").reset_index(drop=True)
    P = {}
    for n in names:
        m = key[KEY].merge(oofs[n], on=KEY, how="left")
        P[n] = {tg: m[f"p_{tg}"].to_numpy() for tg in TARGETS}
    return key, P


def stack_forward(key: pd.DataFrame, P: dict[str, dict[str, np.ndarray]], y: dict[str, np.ndarray], names: list[str], lam: float = 5.0) -> dict[str, np.ndarray]:
    """学習器の logit の線形結合（切片＋傾き）を、前の年までの OOF で当てはめ、次の年に当てる。最初の年は等重みの logit 平均。"""
    year = key["year"].to_numpy()
    race = key["race_id"].to_numpy()
    out = {}
    for tg in TARGETS:
        Z = np.column_stack([_logit(P[n][tg]) for n in names])
        pred = np.zeros(len(key))
        for yr in sorted(set(year)):
            te = year == yr
            tr = year < yr
            if tr.sum() < 1000:
                pred[te] = Z[te].mean(axis=1)
                continue
            X = np.column_stack([np.ones(tr.sum()), Z[tr] - Z[tr].mean(axis=1, keepdims=True)])
            off = Z[tr].mean(axis=1)
            beta = _fit_logit(X, y[tg][tr], off, lam)
            Xt = np.column_stack([np.ones(te.sum()), Z[te] - Z[te].mean(axis=1, keepdims=True)])
            pred[te] = Z[te].mean(axis=1) + Xt @ beta
        out[tg] = _norm(race, _sigmoid(pred), TOTAL[tg])
    return out


def _save_oof(key: pd.DataFrame, pr: dict[str, np.ndarray], name: str) -> None:
    d = key[KEY + ["year"]].copy()
    for tg in TARGETS:
        d[f"p_{tg}"] = pr[tg]
    d.to_parquet(OUT / "oof" / f"{name}.parquet", index=False)


def cmd_stack(a) -> None:
    from src.scripts.data import tp27_eval as t27

    df, base, meta, folds, lp = t27.setup()
    res = {}
    base_oof = pd.read_parquet(Path(a.base))
    base_oof["race_id"], base_oof["horse_id"] = base_oof["race_id"].astype(str), base_oof["horse_id"].astype(str)
    fin_map = meta.set_index(KEY)["finish_position"]
    for p0 in [x for x in a.personas.split(",") if x]:
        p = a.prefix + p0
        names = [f"{p}__{r}" for r in a.recipes.split(",") if r]
        oofs = _load_oofs(names)
        if not oofs:
            continue
        names = list(oofs)
        key, P = _align(oofs)
        y = _targets(meta, key)
        pr = stack_forward(key, P, y, names)
        _save_oof(key, pr, f"{p}__stack")
        single = {n: {tg: _ll(y[tg], P[n][tg]) for tg in TARGETS} for n in names}
        st = {tg: _ll(y[tg], pr[tg]) for tg in TARGETS}
        d = key[KEY + ["year"]].copy()
        for tg in TARGETS:
            d[f"p_{tg}"] = pr[tg]
        cmp = t._cmp(base_oof, d, fin_map, f"{p}__stack vs v6e")
        res[p] = {"learners": names, "single": single, "stack": st, "vs_v6e": cmp}
        _log(f"{p}: stack win {st['win']:.5f} top2 {st['top2']:.5f} top3 {st['top3']:.5f}  (v6e比 Δwin {cmp['Δwin']:+.5f})  単体 " + ", ".join(f"{n.split('__')[1]}={v['win']:.5f}" for n, v in single.items()))
    _jsave(OUT / f"stack_results{a.tag}.json", res)


# ---------------------------------------------------------------- ゲーティング
def regimes(df: pd.DataFrame, rm: pd.DataFrame, key: pd.DataFrame) -> dict[str, np.ndarray]:
    d = df.set_index(KEY)
    sub = d.reindex(pd.MultiIndex.from_frame(key[KEY]))
    dist = sub["distance"].to_numpy(float)
    band = np.select([dist <= 1400, dist <= 1800, dist <= 2200], ["S", "M", "I"], "L")
    surf = sub["surface"].astype(str).to_numpy()
    sd = np.char.add(surf.astype(str), band.astype(str))
    going = np.where(sub["track_condition"].astype(str).to_numpy() == "良", "g", "h")
    fs = sub["field_size"].to_numpy(float)
    fsb = np.select([fs <= 12, fs <= 15], ["s", "m"], "l")
    mon = pd.to_datetime(sub["date"]).dt.month.to_numpy()
    season = np.select([np.isin(mon, [3, 4, 5]), np.isin(mon, [6, 7, 8]), np.isin(mon, [9, 10, 11])], ["sp", "su", "au"], "wi")
    rmi = rm.set_index(KEY).reindex(pd.MultiIndex.from_frame(key[KEY]))
    cls = rmi["cls"].astype(str).to_numpy()
    return {"surf_dist": sd, "class": cls, "surf_going": np.char.add(surf.astype(str), going.astype(str)), "field": fsb, "season": season}


def fit_gate(Z: np.ndarray, y: np.ndarray, reg: np.ndarray | None, tr: np.ndarray, lam_g: float, lam0: float = 1.0, base: np.ndarray | None = None):
    """基準の logit は等重み平均（base を渡せばそれ。例: v6e の logit に対するペルソナの上乗せ）。大域の傾き（ペルソナ別）＋ レジーム別の傾きのずれ（縮約 lam_g）を当てはめる。"""
    K = Z.shape[1]
    base = Z.mean(axis=1) if base is None else base
    Zc = Z - base[:, None]
    parts = [np.ones((len(Z), 1)), Zc]
    cells = []
    if reg is not None:
        cells = sorted(set(reg[tr]))
        for c in cells:
            m = (reg == c).astype(float)[:, None]
            parts.append(Zc * m)
    X = np.hstack(parts)
    pen = np.concatenate([[0.0], np.full(K, lam0), np.full(K * len(cells), lam_g)])
    n, m = X[tr].shape
    beta = np.zeros(m)
    for _ in range(40):
        eta = base[tr] + X[tr] @ beta
        pr = 1.0 / (1.0 + np.exp(-eta))
        g = X[tr].T @ (pr - y[tr]) + pen * beta
        W = pr * (1 - pr)
        H = (X[tr] * W[:, None]).T @ X[tr] + np.diag(pen) + 1e-9 * np.eye(m)
        step = np.linalg.solve(H, g)
        beta -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    return base + X @ beta, beta, cells


def cmd_gate(a) -> None:
    from src.scripts.data import tp27_eval as t27

    df, base, meta, folds, lp = t27.setup()
    rm = pd.read_parquet(e40.OUT / "run_meta.parquet")
    rm["race_id"], rm["horse_id"] = rm["race_id"].astype(str), rm["horse_id"].astype(str)
    names = [f"{a.prefix}{p}__stack" for p in a.personas.split(",") if p]
    oofs = _load_oofs(names)
    names = list(oofs)
    key, P = _align(oofs)
    y = _targets(meta, key)
    RG = regimes(df.assign(race_id=df["race_id"].astype(str), horse_id=df["horse_id"].astype(str)), rm, key)
    year = key["year"].to_numpy()
    race = key["race_id"].to_numpy()
    base_oof = pd.read_parquet(Path(a.base))
    base_oof["race_id"], base_oof["horse_id"] = base_oof["race_id"].astype(str), base_oof["horse_id"].astype(str)
    fin_map = meta.set_index(KEY)["finish_position"]
    results = {"personas": names, "variants": {}}
    lams = [float(x) for x in a.lams.split(",")]

    bm = key[KEY].merge(base_oof[KEY + [f"p_{tg}" for tg in TARGETS]], on=KEY, how="left")
    BASE_Z = {tg: _logit(bm[f"p_{tg}"].to_numpy()) for tg in TARGETS}

    def run_variant(label: str, reg_name: str | None, lam_g: float, mode: str, onbase: bool = False, sel: list[str] | None = None) -> dict:
        """mode: loyo（他の年で当てはめて保留年に当てる）／forward（前の年までで当てはめ、最初の年は基準のまま）。onbase: v6e の logit を基準にして、ペルソナの上乗せだけを当てはめる"""
        pr = {}
        use = sel or names
        for tg in TARGETS:
            Z = np.column_stack([_logit(P[n][tg]) for n in use])
            reg = RG[reg_name] if reg_name else None
            pred = np.zeros(len(key))
            b0 = BASE_Z[tg] if onbase else Z.mean(axis=1)
            for yr in sorted(set(year)):
                te = year == yr
                tr = (year != yr) if mode == "loyo" else (year < yr)
                if tr.sum() < 1000:
                    pred[te] = b0[te]
                    continue
                eta, beta, cells = fit_gate(Z, y[tg], reg, tr, lam_g, base=b0)
                pred[te] = eta[te]
            pr[tg] = _norm(race, _sigmoid(pred), TOTAL[tg])
        d = key[KEY + ["year"]].copy()
        for tg in TARGETS:
            d[f"p_{tg}"] = pr[tg]
        c = t._cmp(base_oof, d, fin_map, label)
        by_year = {}
        for yr in sorted(set(year)):
            te = year == yr
            by_year[str(yr)] = {tg: _ll(y[tg][te], pr[tg][te]) for tg in TARGETS}
        return {"cmp_vs_v6e": c, "by_year": by_year, "ll": {tg: _ll(y[tg], pr[tg]) for tg in TARGETS}, "oof": d}

    equal = {tg: _norm(race, _sigmoid(np.column_stack([_logit(P[n][tg]) for n in names]).mean(axis=1)), TOTAL[tg]) for tg in TARGETS}
    base_ll = {tg: _ll(y[tg], base_oof.merge(key[KEY], on=KEY)[f"p_{tg}"].to_numpy()) for tg in TARGETS}
    results["equal_weights_ll"] = {tg: _ll(y[tg], equal[tg]) for tg in TARGETS}
    results["v6e_ll"] = base_ll
    for onbase in (False, True):
      for mode in ("loyo", "forward"):
        for reg_name in [None, "class", "surf_going", "surf_dist", "field", "season"]:
            for lam_g in ([0.0] if reg_name is None else lams):
                label = f"{'onbase' if onbase else 'personas'}/{mode}/{reg_name or 'global'}/lam{lam_g:g}"
                r = run_variant(label, reg_name, lam_g, mode, onbase)
                d = r.pop("oof")
                if mode == "loyo" and (reg_name is None or lam_g == lams[-1]):
                    d.to_parquet(OUT / "oof" / f"gate{a.tag}_{'onbase' if onbase else 'personas'}__{reg_name or 'global'}__lam{lam_g:g}.parquet", index=False)
                results["variants"][label] = r
                c = r["cmp_vs_v6e"]
                _log(f"{label}: win {r['ll']['win']:.5f} (v6e比 Δwin {c['Δwin']:+.5f} hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f}")
    if a.ablate:
        ab = {}
        for n in names:
            for kind, sel in (("add", [n]), ("drop", [x for x in names if x != n])):
                for mode in ("loyo", "forward"):
                    r = run_variant(f"{kind}/{n}/{mode}", None, 0.0, mode, True, sel)
                    r.pop("oof")
                    ab[f"{kind}/{n}/{mode}"] = r
                    c = r["cmp_vs_v6e"]
                    _log(f"ablate {kind}/{n}/{mode}: Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f}")
        results["ablate"] = ab
    _jsave(OUT / f"gate_results{a.tag}.json", results)


def cmd_disagree(a) -> None:
    """ペルソナ間の不一致（5 ペルソナの logit のばらつき）と、v6e の較正差（実績−予測）の関係を、予測の水準別に見る。"""
    from src.scripts.data import tp27_eval as t27

    df, base, meta, folds, lp = t27.setup()
    names = [f"{a.prefix}{p}__stack" for p in a.personas.split(",") if p]
    oofs = _load_oofs(names)
    names = list(oofs)
    key, P = _align(oofs)
    y = _targets(meta, key)
    base_oof = pd.read_parquet(Path(a.base))
    base_oof["race_id"], base_oof["horse_id"] = base_oof["race_id"].astype(str), base_oof["horse_id"].astype(str)
    bm = key[KEY].merge(base_oof, on=KEY, how="left")
    out = {"personas": names, "targets": {}}
    for tg in ("win", "top3"):
        Z = np.column_stack([_logit(P[n][tg]) for n in names])
        dis = Z.std(axis=1)
        pb = bm[f"p_{tg}"].to_numpy()
        gap = y[tg] - pb
        lvl = pd.qcut(pb, 3, labels=["低", "中", "高"], duplicates="drop")
        dq = pd.Series(dis).groupby(lvl, observed=True).transform(lambda s: pd.qcut(s, 3, labels=["小", "中", "大"], duplicates="drop"))
        tab = pd.DataFrame({"lvl": lvl, "dis": dq, "gap": gap, "y": y[tg], "p": pb, "yr": key["year"]})
        g = tab.groupby(["lvl", "dis"], observed=True).agg(n=("gap", "size"), gap=("gap", "mean"), y=("y", "mean"), p=("p", "mean")).reset_index()
        g["se"] = tab.groupby(["lvl", "dis"], observed=True)["gap"].std().to_numpy() / np.sqrt(g["n"])
        by_year = {}
        for yr in sorted(set(key["year"])):
            tt = tab[tab["yr"] == yr]
            by_year[str(yr)] = tt.groupby(["lvl", "dis"], observed=True)["gap"].mean().round(5).to_dict().__repr__()
        out["targets"][tg] = {"table": g.round(5).to_dict("records"), "corr_dis_absgap": float(np.corrcoef(dis, np.abs(gap))[0, 1]),
                              "corr_dis_gap": float(np.corrcoef(dis, gap)[0, 1])}
        for r in g.round(4).to_dict("records"):
            _log(f"{tg} 水準={r['lvl']} 不一致={r['dis']}: n={r['n']} 実績-予測={r['gap']:+.4f}±{r['se']:.4f}")
    _jsave(OUT / f"disagree{a.tag}.json", out)


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("inputs").set_defaults(fn=cmd_inputs)
    sp.add_parser("matrows").set_defaults(fn=cmd_matrows)
    m = sp.add_parser("matprior")
    m.add_argument("--k", type=int, default=40)
    m.set_defaults(fn=cmd_matprior)
    s = sp.add_parser("sub")
    s.add_argument("--personas", default=",".join(PERSONAS))
    s.add_argument("--recipes", default="lgb_binary,lgb_softmax,linear")
    s.add_argument("--years", default="")
    s.add_argument("--seeds", default="")
    s.add_argument("--tag", default="")
    s.set_defaults(fn=cmd_sub)
    k = sp.add_parser("stack")
    k.add_argument("--personas", default=",".join(PERSONAS))
    k.add_argument("--recipes", default="lgb_binary,lgb_softmax,linear")
    k.add_argument("--base", default=str(e40.V / "v6/eval/oof_z:RB@e.parquet"))
    k.add_argument("--tag", default="")
    k.add_argument("--prefix", default="", help="ペルソナ名の前に付ける接頭辞（対照は ctrl_）")
    k.set_defaults(fn=cmd_stack)
    g = sp.add_parser("gate")
    g.add_argument("--personas", default=",".join(PERSONAS))
    g.add_argument("--lams", default="30,300,3000")
    g.add_argument("--base", default=str(e40.V / "v6/eval/oof_z:RB@e.parquet"))
    g.add_argument("--tag", default="")
    g.add_argument("--prefix", default="")
    g.add_argument("--ablate", action="store_true")
    g.set_defaults(fn=cmd_gate)
    d = sp.add_parser("disagree")
    d.add_argument("--personas", default=",".join(PERSONAS))
    d.add_argument("--base", default=str(e40.V / "v6/eval/oof_z:RB@e.parquet"))
    d.add_argument("--tag", default="")
    d.add_argument("--prefix", default="")
    d.set_defaults(fn=cmd_disagree)
    a = ap.parse_args()
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
