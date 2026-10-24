"""tp-51 段階 A: V4 の 7 メンバーの OOF（valid 2022〜2024）だけで、アンサンブルの『合成の仕方』を比べる。

python -m src.scripts.data.tp51_stack run [--only NAME,...]     # 全候補を LOYO と前進（forward）で評価し、現行（weighted_logit / greedy）との差を出す
python -m src.scripts.data.tp51_stack leakcheck                 # メタ学習が評価年の情報を使っていないことの検査

評価の枠は既存どおり（valid 2022・2023・2024、勝率・連対・複勝の LogLoss、レース単位 paired bootstrap の CI 上限<0、効果≥0.0002、連対・複勝の悪化≤0.0002）。
LOYO = 1 年を外して残りで合成を学習。前進 = 評価年より前の年だけで学習（2023 は 2022 のみ、2024 は 2022〜23）。厳密なリーク防止は前進の方。
環境変数: KEIBA_ROOT, TP51_OUT, TP51_THREADS（既定 3）, TP51_MEMBERS_DIR（既定 V4 の tp27/oof）。
市場オッズ・人気は使わない。test 2025・2026 年は使わない。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("KEIBA_ROOT", Path(__file__).resolve().parents[3]))
V4 = ROOT / "data/local/modeling/training_pipeline_v4"
MEM_DIR = Path(os.environ.get("TP51_MEMBERS_DIR") or V4 / "tp27/oof")
OUT = Path(os.environ.get("TP51_OUT") or ROOT / "data/local/modeling/tp51") / "stack"
THREADS = int(os.environ.get("TP51_THREADS", 3))
KEY = ["race_id", "horse_id"]
MEMBERS = ["lgb_extras_ped", "xgb_extras_ped", "lgb_extras", "lgb_softmax", "linear", "mtl_nn_extras_ped", "mtl_nn_s"]
GBM = ["lgb_extras_ped", "xgb_extras_ped", "lgb_extras", "lgb_softmax"]
NNS = ["linear", "mtl_nn_extras_ped", "mtl_nn_s"]
TARGETS = ("win", "top2", "top3")
TOT = {"win": 1.0, "top2": 2.0, "top3": 3.0}
BASE_METHOD = {"win": "weighted_logit", "top2": "greedy", "top3": "greedy"}      # V4 の現行（v4_e2）
CTX = ["field_size", "surface", "distance", "race_class", "track_condition", "venue_code"]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ───────────────────────── データ
def load_frame() -> pd.DataFrame:
    """メンバー OOF（p_*）+ レース文脈 + 着順。1 行 = 1 頭。"""
    import pyarrow.parquet as pq

    df = None
    for m in MEMBERS:
        o = pd.read_parquet(MEM_DIR / f"{m}__v4.parquet")
        o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
        o = o.rename(columns={f"p_{t}": f"{m}__{t}" for t in TARGETS})
        o = o[KEY + (["year"] if df is None else []) + [f"{m}__{t}" for t in TARGETS]]
        df = o if df is None else df.merge(o, on=KEY, how="inner")
    for f in sorted((OUT.parent / os.environ.get("TP51_MEMBERS_SUBDIR", "members")).glob("*.parquet")):       # 段階 B で作った追加メンバーの OOF
        o = pd.read_parquet(f)
        o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
        o = o.rename(columns={f"p_{t}": f"{f.stem}__{t}" for t in TARGETS})[KEY + [f"{f.stem}__{t}" for t in TARGETS]]
        df = df.merge(o, on=KEY, how="left")
    for f in sorted((OUT.parent / os.environ.get("TP51_MEMBERS_SUBDIR", "members")).glob("*_sd345.parquet")):       # seed を 3 → 6 本にした版（元の 3 本 + 追加の 3 本の logit 平均）
        b = f.stem[: -len("_sd345")]
        for t_ in TARGETS:
            if f"{b}__{t_}" in df.columns and f"{f.stem}__{t_}" in df.columns:
                df[f"{b}_bag6__{t_}"] = 1 / (1 + np.exp(-(logit(df[f"{b}__{t_}"].to_numpy()) + logit(df[f"{f.stem}__{t_}"].to_numpy())) / 2))
    cols = KEY + ["finish_position"] + [c for c in CTX]
    have = set(pq.read_schema(V4 / "features/train_table.parquet").names)
    cols = [c for c in cols if c in have]
    t = pq.read_table(V4 / "features/train_table.parquet", columns=cols).to_pandas()
    t["race_id"], t["horse_id"] = t["race_id"].astype(str), t["horse_id"].astype(str)
    df = df.merge(t, on=KEY, how="left")
    assert df["finish_position"].notna().all()
    for c in ("surface", "race_class", "track_condition", "venue_code"):
        if c in df.columns:
            df[c + "_code"] = pd.factorize(df[c].astype(str))[0]
    yrs = [int(y) for y in os.environ.get("TP51_YEARS", "").split(",") if y]
    if yrs:                                                  # 追加メンバーが一部の年だけのとき（粗い段階）
        df = df[df["year"].isin(yrs)]
        df = df.dropna(subset=[c for c in df.columns if "__" in c])
    df = df.sort_values(["year", "race_id"], kind="stable").reset_index(drop=True)
    return df


def logit(p):
    from src.pipeline.models import ensemble as en

    return en.logit(p)


def member_arrays(df: pd.DataFrame, members=MEMBERS) -> dict:
    """target -> (n × M) の確率。"""
    return {t: np.stack([df[f"{m}__{t}"].to_numpy() for m in members], axis=1) for t in TARGETS}


def meta_features(df: pd.DataFrame, with_ctx: bool = True) -> np.ndarray:
    """メタ学習の入力: 7 メンバー × 3 対象の logit（21）+ 対象ごとの平均・標準偏差（6）+ レース内で標準化した平均 logit（3）+ 文脈。"""
    P = member_arrays(df)
    cols = []
    for t in TARGETS:
        Z = logit(P[t])
        cols += [Z, Z.mean(1, keepdims=True), Z.std(1, keepdims=True)]
        zm = Z.mean(1)
        g = pd.Series(zm).groupby(df["race_id"].to_numpy())
        cols.append(((zm - g.transform("mean").to_numpy()) / (g.transform("std").to_numpy() + 1e-6))[:, None])
    if with_ctx:
        ctx = []
        for c in ("field_size", "distance"):
            if c in df.columns:
                v = df[c].astype(float).to_numpy()
                ctx.append(((v - v.mean()) / (v.std() + 1e-9))[:, None])
        if "surface_code" in df.columns:
            ctx.append(pd.get_dummies(df["surface_code"]).to_numpy(dtype=float))
        if "race_class_code" in df.columns:
            ctx.append(pd.get_dummies(df["race_class_code"]).to_numpy(dtype=float))
        cols += ctx
    return np.hstack(cols)


# ───────────────────────── 評価の枠
def finalize(df: pd.DataFrame, raw: dict) -> pd.DataFrame:
    from src.pipeline.models import win_models as wm

    race = df["race_id"].to_numpy()
    out = df[KEY + ["year"]].copy()
    for t in TARGETS:
        out[f"p_{t}"] = wm.normalize_to_total(race, np.clip(raw[t], 1e-9, 1 - 1e-9), TOT[t])
    return out


def run_scheme(df: pd.DataFrame, fn, mode: str) -> pd.DataFrame:
    """fn(train_df, valid_df) -> {win, top2, top3}（正規化前）。mode: loyo / forward。"""
    years = sorted(df["year"].unique())
    parts = []
    for y in years:
        if mode == "forward":
            tr = df[df["year"] < y]
            if tr.empty:
                continue
        else:
            tr = df[df["year"] != y]
        va = df[df["year"] == y]
        raw = fn(tr.reset_index(drop=True), va.reset_index(drop=True))
        parts.append(finalize(va.reset_index(drop=True), raw))
    return pd.concat(parts, ignore_index=True)


def compare(base: pd.DataFrame, new: pd.DataFrame, fin_map: pd.Series, label: str) -> dict:
    from src.pipeline.models import feature_bakeoff as fb

    k = base[KEY].merge(new[KEY], on=KEY, how="inner")
    b = base.merge(k, on=KEY)[KEY + ["p_win", "p_top2", "p_top3"]]
    n = new.merge(k, on=KEY)[KEY + ["p_win", "p_top2", "p_top3"]]
    c = fb.compare_oof(b, n, fin_map, label)
    return {kk: (float(v) if isinstance(v, (float, np.floating)) else v) for kk, v in c.items()}


# ───────────────────────── 合成の方式（fn）
def fn_spec(methods: dict, members=MEMBERS):
    """既存の合成（ensemble.fit_method / blend）。"""
    from src.pipeline.models import ensemble as en

    def fn(tr, va):
        Ptr, Pva = member_arrays(tr, members), member_arrays(va, members)
        raw = {}
        for t in TARGETS:
            y = (tr["finish_position"].to_numpy() <= {"win": 1, "top2": 2, "top3": 3}[t]).astype(float)
            st = en.fit_method(methods[t], Ptr[t], en.logit(Ptr[t]), y)
            raw[t] = en.blend(methods[t], Pva[t], en.logit(Pva[t]), st)
        return raw

    return fn


def _ys(tr):
    f = tr["finish_position"].to_numpy()
    return {"win": (f <= 1).astype(float), "top2": (f <= 2).astype(float), "top3": (f <= 3).astype(float)}


def fn_lr(feature="members", C=1.0, ctx=False):
    """ロジスティック回帰のメタ。features: members（その対象の 7 logit）/ cross（21 logit）/ full（全メタ特徴）。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def fn(tr, va):
        Y = _ys(tr)
        Ptr, Pva = member_arrays(tr), member_arrays(va)
        if feature == "members":
            Xtr_all = {t: logit(Ptr[t]) for t in TARGETS}
            Xva_all = {t: logit(Pva[t]) for t in TARGETS}
        elif feature == "ownwin":        # その対象の 7 logit + 勝率の 7 logit（連対・複勝の過学習を抑える）
            Xtr_all = {t: np.hstack([logit(Ptr[t]), logit(Ptr["win"])]) if t != "win" else logit(Ptr[t]) for t in TARGETS}
            Xva_all = {t: np.hstack([logit(Pva[t]), logit(Pva["win"])]) if t != "win" else logit(Pva[t]) for t in TARGETS}
        else:
            Ftr, Fva = meta_features(tr, ctx), meta_features(va, ctx)
            sc = StandardScaler().fit(Ftr)
            Xtr_all = {t: sc.transform(Ftr) for t in TARGETS}
            Xva_all = {t: sc.transform(Fva) for t in TARGETS}
        raw = {}
        for t in TARGETS:
            m = LogisticRegression(C=C, max_iter=1000).fit(Xtr_all[t], Y[t])
            raw[t] = m.predict_proba(Xva_all[t])[:, 1]
        return raw

    return fn


def fn_lgbm(num_leaves=7, rounds=250, lr=0.03, min_data=400, ctx=True, seeds=(0,)):
    import lightgbm as lgb

    def fn(tr, va):
        Y = _ys(tr)
        Ftr, Fva = meta_features(tr, ctx), meta_features(va, ctx)
        raw = {t: np.zeros(len(va)) for t in TARGETS}
        for t in TARGETS:
            for s in seeds:
                p = dict(objective="binary", learning_rate=lr, num_leaves=num_leaves, min_data_in_leaf=min_data, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                         lambda_l2=10.0, verbose=-1, num_threads=THREADS, seed=s)
                b = lgb.train(p, lgb.Dataset(Ftr, Y[t]), num_boost_round=rounds)
                raw[t] += b.predict(Fva) / len(seeds)
        return raw

    return fn


def fn_nn(hidden=32, epochs=25, lr=2e-3, wd=1e-4, ctx=True, seeds=(0, 1, 2)):
    """小さな NN のメタ（勝率はレース内 softmax、連対・複勝は sigmoid。3 つの損失を同時に最小化）。"""
    import torch
    from sklearn.preprocessing import StandardScaler

    torch.set_num_threads(THREADS)

    def fn(tr, va):
        Ftr, Fva = meta_features(tr, ctx), meta_features(va, ctx)
        sc = StandardScaler().fit(Ftr)
        Xtr, Xva = torch.tensor(sc.transform(Ftr), dtype=torch.float32), torch.tensor(sc.transform(Fva), dtype=torch.float32)
        f = tr["finish_position"].to_numpy()
        y2, y3 = torch.tensor((f <= 2).astype(np.float32)), torch.tensor((f <= 3).astype(np.float32))
        yw = torch.tensor((f <= 1).astype(np.float32))
        rid = pd.factorize(tr["race_id"].to_numpy())[0]
        order = np.argsort(rid, kind="stable")
        bounds = np.flatnonzero(np.diff(rid[order])) + 1
        races = np.split(order, bounds)
        vr = pd.factorize(va["race_id"].to_numpy())[0]
        out_win, out2, out3 = np.zeros(len(va)), np.zeros(len(va)), np.zeros(len(va))
        for s in seeds:
            torch.manual_seed(s)
            net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(hidden, 3))
            opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
            rng = np.random.default_rng(s)
            for ep in range(epochs):
                net.train()
                perm = rng.permutation(len(races))
                for i in range(0, len(perm), 256):
                    idx = np.concatenate([races[j] for j in perm[i:i + 256]])
                    o = net(Xtr[idx])
                    # 勝率: レース内 softmax の対数尤度
                    rr = pd.factorize(rid[idx])[0]
                    rr_t = torch.tensor(rr)
                    mx = torch.zeros(rr.max() + 1).scatter_reduce(0, rr_t, o[:, 0], reduce="amax", include_self=False)
                    ex = torch.exp(o[:, 0] - mx[rr_t])
                    den = torch.zeros(rr.max() + 1).index_add(0, rr_t, ex)
                    lw = -(torch.log(ex / den[rr_t] + 1e-9) * yw[idx]).sum() / max(1.0, float(yw[idx].sum()))
                    l2 = torch.nn.functional.binary_cross_entropy_with_logits(o[:, 1], y2[idx])
                    l3 = torch.nn.functional.binary_cross_entropy_with_logits(o[:, 2], y3[idx])
                    loss = lw + l2 + l3
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            net.eval()
            with torch.no_grad():
                o = net(Xva).numpy()
            e = np.exp(o[:, 0] - pd.Series(o[:, 0]).groupby(vr).transform("max").to_numpy())
            out_win += (e / pd.Series(e).groupby(vr).transform("sum").to_numpy()) / len(seeds)
            out2 += 1 / (1 + np.exp(-o[:, 1])) / len(seeds)
            out3 += 1 / (1 + np.exp(-o[:, 2])) / len(seeds)
        return {"win": out_win, "top2": out2, "top3": out3}

    return fn


def fn_group_stack(C=1.0, ctx=True):
    """2 段: GBM 群の平均 logit と NN 群の平均 logit を作り、それを文脈と一緒に LR で合成。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def fn(tr, va):
        Y = _ys(tr)
        raw = {}
        Ptr_g, Pva_g = member_arrays(tr, GBM), member_arrays(va, GBM)
        Ptr_n, Pva_n = member_arrays(tr, NNS), member_arrays(va, NNS)
        for t in TARGETS:
            Atr, Ava = logit(Ptr_g[t]).mean(1), logit(Pva_g[t]).mean(1)
            Btr, Bva = logit(Ptr_n[t]).mean(1), logit(Pva_n[t]).mean(1)
            Xtr = np.column_stack([Atr, Btr, Atr * Btr, np.abs(Atr - Btr)])
            Xva = np.column_stack([Ava, Bva, Ava * Bva, np.abs(Ava - Bva)])
            if ctx:
                for c in ("field_size", "distance"):
                    if c in tr.columns:
                        mu, sd = tr[c].astype(float).mean(), tr[c].astype(float).std() + 1e-9
                        Xtr = np.column_stack([Xtr, (tr[c].astype(float) - mu) / sd, Atr * (tr[c].astype(float) - mu) / sd])
                        Xva = np.column_stack([Xva, (va[c].astype(float) - mu) / sd, Ava * (va[c].astype(float) - mu) / sd])
            sc = StandardScaler().fit(Xtr)
            m = LogisticRegression(C=C, max_iter=1000).fit(sc.transform(Xtr), Y[t])
            raw[t] = m.predict_proba(sc.transform(Xva))[:, 1]
        return raw

    return fn


def fn_segment(K=20000.0):
    """面 × 距離帯ごとの weighted_logit を、全体の重みへ縮約して使う（専門家の混合）。"""
    from src.pipeline.models import ensemble as en

    def seg(d):
        dist = d["distance"].astype(float).to_numpy()
        band = np.where(dist <= 1400, 0, np.where(dist <= 1900, 1, 2))
        return d["surface_code"].to_numpy() * 3 + band

    def fn(tr, va):
        Ptr, Pva = member_arrays(tr), member_arrays(va)
        Y = _ys(tr)
        s_tr, s_va = seg(tr), seg(va)
        raw = {}
        for t in TARGETS:
            Ztr, Zva = en.logit(Ptr[t]), en.logit(Pva[t])
            wg = en.fit_simplex_weights(Ztr, Y[t])
            out = en.sigmoid(Zva @ wg)
            for sgm in np.unique(s_va):
                mt, mv = s_tr == sgm, s_va == sgm
                if mt.sum() < 2000:
                    continue
                ws = en.fit_simplex_weights(Ztr[mt], Y[t][mt])
                lam = mt.sum() / (mt.sum() + K)
                w = lam * ws + (1 - lam) * wg
                out[mv] = en.sigmoid(Zva[mv] @ w)
            raw[t] = out
        return raw

    return fn


def fn_post(kind: str):
    """現行の合成の後段: temperature（logit 空間の傾きと切片）/ isotonic / Harville 整合の混合。"""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    base_fn = fn_spec(BASE_METHOD)

    def fn(tr, va):
        # 学習側の『現行合成の予測』は、学習年だけで作る必要がある（内側 LOYO）。
        yrs = sorted(tr["year"].unique())
        inner = []
        for y in yrs:
            a = tr[tr["year"] != y].reset_index(drop=True)
            b = tr[tr["year"] == y].reset_index(drop=True)
            if a.empty:
                continue
            raw = base_fn(a, b)
            inner.append((b, finalize(b, raw)))
        raw_va = base_fn(tr, va)
        fin_va = finalize(va, raw_va)
        if not inner:       # 学習側が 1 年だけ（前進の 2023）だと内側の LOYO ができない → 補正なし
            return {t: fin_va[f"p_{t}"].to_numpy() for t in TARGETS}
        out = {}
        if kind in ("temp", "iso"):
            for t in TARGETS:
                ptr = np.concatenate([p[f"p_{t}"].to_numpy() for _, p in inner])
                ytr = np.concatenate([(b["finish_position"].to_numpy() <= {"win": 1, "top2": 2, "top3": 3}[t]).astype(float) for b, _ in inner])
                if kind == "temp":
                    m = LogisticRegression(C=1e6, max_iter=500).fit(logit(ptr)[:, None], ytr)
                    out[t] = m.predict_proba(logit(fin_va[f"p_{t}"].to_numpy())[:, None])[:, 1]
                else:
                    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(ptr, ytr)
                    out[t] = iso.predict(fin_va[f"p_{t}"].to_numpy())
        else:     # harville: 連対・複勝を『直接の予測』と『勝率からの Harville』の混合にする
            from src.pipeline.models import zoo

            ptr_all = pd.concat([p for _, p in inner], ignore_index=True)
            btr_all = pd.concat([b for b, _ in inner], ignore_index=True)
            race = btr_all["race_id"].to_numpy()
            fin = btr_all["finish_position"].to_numpy()
            g2 = zoo.fit_gamma(race, ptr_all["p_win"].to_numpy(), fin, 2)
            g3 = zoo.fit_gamma(race, ptr_all["p_win"].to_numpy(), fin, 3)
            h2, h3 = zoo.harville_topk(race, ptr_all["p_win"].to_numpy(), g2, g3)
            out["win"] = fin_va["p_win"].to_numpy()
            rv = va["race_id"].to_numpy()
            v2, v3 = zoo.harville_topk(rv, fin_va["p_win"].to_numpy(), g2, g3)
            for t, hh, vv in (("top2", h2, v2), ("top3", h3, v3)):
                yy = (fin <= {"top2": 2, "top3": 3}[t]).astype(float)
                pd_ = ptr_all[f"p_{t}"].to_numpy()
                m = LogisticRegression(C=1e6, max_iter=500).fit(np.column_stack([logit(pd_), logit(np.clip(hh, 1e-6, 1 - 1e-6))]), yy)
                out[t] = m.predict_proba(np.column_stack([logit(fin_va[f"p_{t}"].to_numpy()), logit(np.clip(vv, 1e-6, 1 - 1e-6))]))[:, 1]
        return out

    return fn


def fn_mix(by_target: dict):
    """対象ごとに別の合成を使う（例: 勝率だけ LR、連対・複勝は現行）。"""

    def fn(tr, va):
        cache = {}
        raw = {}
        for t in TARGETS:
            f = by_target[t]
            if id(f) not in cache:
                cache[id(f)] = f(tr, va)
            raw[t] = cache[id(f)][t]
        return raw

    return fn


def fn_avg(fns, weights=None):
    """複数の合成の出力を logit 空間で平均する（L3）。"""
    from src.pipeline.models import ensemble as en

    def fn(tr, va):
        outs = [f(tr, va) for f in fns]
        w = np.full(len(fns), 1.0 / len(fns)) if weights is None else np.asarray(weights, float) / np.sum(weights)
        return {t: en.sigmoid(sum(wi * en.logit(o[t]) for wi, o in zip(w, outs))) for t in TARGETS}

    return fn


def extra_members() -> list[str]:
    return [f.stem for f in sorted((OUT.parent / os.environ.get("TP51_MEMBERS_SUBDIR", "members")).glob("*.parquet"))]


SCHEMES = {
    "S0_spec": fn_spec(BASE_METHOD),
    "S1_wlogit_all": fn_spec({t: "weighted_logit" for t in TARGETS}),
    "S1b_greedy_all": fn_spec({t: "greedy" for t in TARGETS}),
    "S2_stack_lr": fn_spec({t: "stack_lr" for t in TARGETS}),
    "S3_lr_cross": fn_lr("cross", 0.1, ctx=False),
    "S4_lr_full": fn_lr("full", 0.05, ctx=True),
    "S5_lgbm_meta": fn_lgbm(),
    "S5b_lgbm_meta_noctx": fn_lgbm(ctx=False),
    "S5c_lgbm_meta_deep": fn_lgbm(num_leaves=15, rounds=400, min_data=800),
    "S6_nn_meta": fn_nn(),
    "S6b_nn_meta_noctx": fn_nn(ctx=False),
    "S7_group_stack": fn_group_stack(),
    "S8_segment": fn_segment(),
    "S9_temp": fn_post("temp"),
    "S9b_iso": fn_post("iso"),
    "S10_harville_mix": fn_post("harville"),
}
SCHEMES["S3c_lr_cross_C003"] = fn_lr("cross", 0.03, ctx=False)
SCHEMES["S3d_lr_cross_C1"] = fn_lr("cross", 1.0, ctx=False)
SCHEMES["S3e_lr_ownwin"] = fn_lr("ownwin", 0.1, ctx=False)
SCHEMES["S3w_win_cross_lr"] = fn_mix({"win": SCHEMES["S3_lr_cross"], "top2": SCHEMES["S0_spec"], "top3": SCHEMES["S0_spec"]})
SCHEMES["S4w_win_full_lr"] = fn_mix({"win": SCHEMES["S4_lr_full"], "top2": SCHEMES["S0_spec"], "top3": SCHEMES["S0_spec"]})
SCHEMES["S3x_win_cross_oth_ownwin"] = fn_mix({"win": SCHEMES["S3_lr_cross"], "top2": SCHEMES["S3e_lr_ownwin"], "top3": SCHEMES["S3e_lr_ownwin"]})
SCHEMES["S11_L3_lgbm_nn"] = fn_avg([SCHEMES["S0_spec"], SCHEMES["S5_lgbm_meta"], SCHEMES["S6_nn_meta"]])
SCHEMES["S11b_L3_lr"] = fn_avg([SCHEMES["S0_spec"], SCHEMES["S3_lr_cross"]])
SCHEMES["S11c_L3_all"] = fn_avg([SCHEMES["S0_spec"], SCHEMES["S5_lgbm_meta"], SCHEMES["S6_nn_meta"], SCHEMES["S3_lr_cross"]])
SCHEMES["S11d_L3_group"] = fn_avg([SCHEMES["S0_spec"], SCHEMES["S7_group_stack"]])
# 勝率・連対は L3（現行＋クロス対象 LR）、複勝は現行のまま（L3 は複勝だけ悪化するため）
SCHEMES["S11e_L3lr_win2"] = fn_mix({"win": SCHEMES["S11b_L3_lr"], "top2": SCHEMES["S11b_L3_lr"], "top3": SCHEMES["S0_spec"]})
SCHEMES["S11f_L3all_win2"] = fn_mix({"win": SCHEMES["S11c_L3_all"], "top2": SCHEMES["S11c_L3_all"], "top3": SCHEMES["S0_spec"]})
SCHEMES["S11g_L3_lr_full"] = fn_avg([SCHEMES["S0_spec"], SCHEMES["S3_lr_cross"], SCHEMES["S4_lr_full"]])
SCHEMES["S11h_L3_lr_full_win2"] = fn_mix({"win": SCHEMES["S11g_L3_lr_full"], "top2": SCHEMES["S11g_L3_lr_full"], "top3": SCHEMES["S0_spec"]})
SCHEMES["S11i_L3_w05"] = fn_avg([SCHEMES["S0_spec"], SCHEMES["S3_lr_cross"]], weights=[0.65, 0.35])
SCHEMES["S11j_L3_w05_win2"] = fn_mix({"win": SCHEMES["S11i_L3_w05"], "top2": SCHEMES["S11i_L3_w05"], "top3": SCHEMES["S0_spec"]})


def add_extra_schemes() -> None:
    """追加メンバーごとに『V4 の 7 + そのメンバー』の現行合成（P_*）と、『V4 の 7 + 全追加メンバー』（P_all）を足す。"""
    ex = [n for n in extra_members() if not n.endswith("_sd345")]
    bags = [n[: -len("_sd345")] for n in extra_members() if n.endswith("_sd345")]
    for b in bags:
        if b in MEMBERS:
            SCHEMES[f"R_{b}_bag6"] = fn_spec(BASE_METHOD, [f"{b}_bag6" if m == b else m for m in MEMBERS])
    if len(bags) > 1:
        SCHEMES["R_all_bag6"] = fn_spec(BASE_METHOD, [f"{m}_bag6" if m in bags else m for m in MEMBERS])
    for n in ex:
        SCHEMES[f"P_{n}"] = fn_spec(BASE_METHOD, MEMBERS + [n])
    if len(ex) > 1:
        SCHEMES["P_all"] = fn_spec(BASE_METHOD, MEMBERS + ex)
        SCHEMES["P_all_L3"] = fn_avg([SCHEMES["P_all"], SCHEMES["S5_lgbm_meta"], SCHEMES["S6_nn_meta"]])



def cmd_run(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    add_extra_schemes()
    df = load_frame()
    fin_map = df.set_index(KEY)["finish_position"]
    _log(f"データ {df.shape}、年 {df['year'].value_counts().sort_index().to_dict()}")
    names = [n for n in (a.only.split(",") if a.only else SCHEMES)]
    names = [n for n in names if n in SCHEMES]
    res = json.loads((OUT / "results.json").read_text(encoding="utf-8")) if (OUT / "results.json").exists() else {}
    bases = {}
    for mode in ("loyo", "forward"):
        p = OUT / f"oof_S0_spec_{mode}.parquet"
        if p.exists():
            bases[mode] = pd.read_parquet(p)
        else:
            bases[mode] = run_scheme(df, SCHEMES["S0_spec"], mode)
            bases[mode].to_parquet(p, index=False)
    for n in names:
        for mode in ("loyo", "forward"):
            key = f"{n}|{mode}"
            if key in res and not a.force:
                continue
            t0 = time.time()
            oof = run_scheme(df, SCHEMES[n], mode)
            oof.to_parquet(OUT / f"oof_{n}_{mode}.parquet", index=False)
            c = compare(bases[mode], oof, fin_map, f"{n} {mode}")
            res[key] = c
            _log(f"{key}: Δwin {c.get('Δwin', float('nan')):+.5f} (hi {c.get('win_hi', float('nan')):+.5f}) Δtop2 {c.get('Δtop2', float('nan')):+.5f} Δtop3 {c.get('Δtop3', float('nan')):+.5f} 採用={c.get('採用')} ({time.time() - t0:.0f}s)")
            (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_peryear(a) -> None:
    """主な方式を年ごとに比べる（前進の 2024 年は『2022〜23 の 2 年で合成を学習』した、v5 の実運用に最も近い条件）。"""
    df = load_frame()
    fin_map = df.set_index(KEY)["finish_position"]
    out = {}
    for n in a.schemes.split(","):
        for mode in ("loyo", "forward"):
            pb, pn = OUT / f"oof_S0_spec_{mode}.parquet", OUT / f"oof_{n}_{mode}.parquet"
            if not (pb.exists() and pn.exists()):
                continue
            b, o = pd.read_parquet(pb), pd.read_parquet(pn)
            for y in sorted(set(b["year"]) & set(o["year"])):
                c = compare(b[b["year"] == y], o[o["year"] == y], fin_map, f"{n} {mode} {y}")
                out[f"{n}|{mode}|{y}"] = c
                _log(f"{n}|{mode}|{y}: Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f}")
    (OUT / "peryear.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_leakcheck(a) -> None:
    """前進評価で、評価年の行の値を書き換えても、同じ年の予測が（評価年の情報を学習に使っていなければ）変わらないことを確認する。"""
    df = load_frame()
    y = sorted(df["year"].unique())[-1]
    tr = df[df["year"] < y].reset_index(drop=True)
    va = df[df["year"] == y].reset_index(drop=True)
    rng = np.random.default_rng(0)
    a1 = SCHEMES[a.scheme](tr, va)
    fn = SCHEMES[a.scheme]
    # 評価年の着順だけを置換し、入力は同じにして予測が変わらないことを確認する
    va3 = va.copy()
    va3["finish_position"] = rng.permutation(va3["finish_position"].to_numpy())
    c = fn(tr, va3)
    diff = max(float(np.abs(c[t] - a1[t]).max()) for t in TARGETS)
    print(json.dumps({"scheme": a.scheme, "評価年の着順を置換した時の予測の最大差": diff, "判定": "OK（評価年のラベルを使っていない）" if diff < 1e-9 else "NG"}, ensure_ascii=False))
    if diff >= 1e-9:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--only", default="")
    r.add_argument("--force", action="store_true")
    py = sp.add_parser("peryear")
    py.add_argument("--schemes", default="S3w_win_cross_lr,S4w_win_full_lr,S11e_L3lr_win2,S11f_L3all_win2,S11h_L3_lr_full_win2,S3_lr_cross,S11c_L3_all")
    lc = sp.add_parser("leakcheck")
    lc.add_argument("--scheme", default="S5_lgbm_meta")
    a = ap.parse_args()
    {"run": cmd_run, "leakcheck": cmd_leakcheck, "peryear": cmd_peryear}[a.cmd](a)


if __name__ == "__main__":
    main()
