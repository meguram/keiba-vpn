"""騎手 × 馬体重帯の相性の列が、第 2 段（市場を offset にした xgb）の精度を上げるかの確かめ（2026-11-18、ユーザーの依頼「騎手と馬体重帯の相性成績で精度が向上した事例があるそうです、見てみてください」）。

    python -m src.scripts.data.jockey_weight_affinity feats                 # 騎手 × 体重帯の as-of の列を作る（OUT/feats.parquet）
    python -m src.scripts.data.jockey_weight_affinity member ARM [--valid-years 2019,2020,2021] [--seeds 3]
        # ARM: base（基準）・today（当日の体重帯）・prev（前走の体重帯）・perm（today の列を年の中で並べ替えた負の対照）
    python -m src.scripts.data.jockey_weight_affinity report [--valid-years 2019,2020,2021]   # 勝ち馬の logloss の差（開催日 × 場で束ねた区間）
    python -m src.scripts.data.jockey_weight_affinity direct [--valid-years 2019,2020,2021]   # 基準の OOF に列を 1 係数で足す直接の検定（主な根拠）

- 本番・モデル・model_cycles_market.py は変えない。部品（load_cache・apply_spec・fit_pred・folds_from_meta・load_market）を使い回す。
- spec は第 2 段の今の候補 s2_13_noise_s3（offset raw・雑音 0.54・同じ drop）。腕ごとに同じ道具で流し、列の足し引きだけを変える。
- 列は、その日より前の日のレースだけで数える（as-of）。件数で「騎手全体 → 全体平均」に縮める。
  当日の馬体重は発走の約 70 分前に分かる。前走の版は、前走の馬体重の帯で引く（引く表は同じ）。
- 表は 2016 年からしか無いので、2016・17 年の行は履歴が薄い（縮みが強く、全体平均に寄る）。
- 2025・26 年は使わない（台帳の決まり。学習表は全年を読んでから行を外す）。2022〜24 は兆しがあったときだけ 1 回流す。
- 列の版: 既定は OUT/feats.parquet。JW_FEATS で別の版を指せる（xgb の腕は数え漏れを直す前の feats_v1_xgb_arms.parquet で流した）。
- 入力は CYCLES_OUT（既定 retrain_out/cycles16_fix3）。出力は JW_OUT（既定 retrain_out/jockey_weight）。GCS は読まない。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data import model_cycles as mc
from src.scripts.data import model_cycles_market as mcm

DEFAULT_CYC = Path("/home/jovyan/work/retrain_out/cycles16_fix3")   # 第 2 段の今の候補の学習表・市場の表（CYCLES_OUT が無いときに main で向ける）

OUT = Path(os.environ.get("JW_OUT") or "/home/jovyan/work/retrain_out/jockey_weight")
KEY = ["race_id", "horse_id"]
BANDS = [0, 439, 479, 519, 10_000]                 # 〜439・440〜479・480〜519・520〜kg
BAND_LABELS = ["~439", "440-479", "480-519", "520~"]
K_BAND = 50.0                                     # 騎手 × 帯の率を騎手全体へ縮める強さ（騎乗数）
K_JOCKEY = 100.0                                  # 騎手全体の率を全体平均へ縮める強さ
K_AE = 10.0                                       # 実/見込み の縮め（見込みの複勝数）
YEARS = range(2016, 2025)                         # 結果の表は 2025・26 年を読まない（学習表・市場の表は全年を読み、2025・26 年の行を外して使わない）
FORBIDDEN_YEARS = (2025, 2026)
COLS = ["jwb_top3", "jwb_win", "jwb_ae3", "jwb_n"]           # xgb の腕に足す列
EXTRA_COLS = ["jwb_ae1"]                                     # 直接の検定だけに使う列
SUMS = ["n", "y1", "y3", "e3f", "y3e", "qf", "y1q"]
K_AE1 = 5.0                                       # 単勝の 実/見込み の縮め（見込みの勝ち数）
# 第 2 段の今の候補 s2_13_noise_s3 の spec（cycles16_fix3/market/oof/s2_13_noise_s3.json と同じ）
SPEC = {
    "blank_before": {"year": 2020, "prefix": ["wc_", "tl_"]},
    "drop": ["wc_cc_d1", "wc_cc_d2", "wc_cc_curv", "wc_cc_slope", "wc_cc_slope3", "wc_c1_d1", "wc_c1_d2", "wc_c1_curv", "wc_c1_slope", "wc_c1_slope3",
             "wc_ca_d1", "wc_ca_d2", "wc_ca_curv", "wc_ca_slope", "wc_ca_slope3", "wc_ccH_d1", "wc_cc_n", "wc_c1_n", "wc_ca_n", "wc_ccH_n"],
    "drop_prefix": ["rb_nk_", "track_bias_", "sire_x_track_bias", "dam_sire_x_track_bias"],
    "offset": "raw",
    "offset_noise_sd": 0.54,
}
ARMS = ("base", "today", "prev", "perm")


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def band_of(w: pd.Series) -> pd.Series:
    w = pd.to_numeric(w, errors="coerce").where(lambda s: s.between(300, 650))
    return pd.cut(w, BANDS, labels=BAND_LABELS).astype(object).where(w.notna(), None)


def load_rows() -> pd.DataFrame:
    """2016〜2024 年の結果の行（騎手・当日の馬体重・確定オッズ・着順）と、市場の複勝の見込み e3。"""
    from src.pipeline.models import zoo

    cols = ["race_id", "date", "horse_id", "jockey_id", "weight", "odds", "finish_position", "jockey_weight", "sex_age"]
    d = pd.concat([pd.read_parquet(mc.TABLES / str(y) / "race_result_flat.parquet", columns=cols) for y in YEARS], ignore_index=True)
    d["race_id"], d["horse_id"], d["jockey_id"] = d["race_id"].astype(str), d["horse_id"].astype(str), d["jockey_id"].astype(str)
    d = d.drop_duplicates(KEY)
    assert not d["race_id"].str[:4].astype(int).isin(FORBIDDEN_YEARS).any()
    for c in ("weight", "odds", "finish_position", "jockey_weight"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["date"] = pd.to_datetime(d["date"])
    ok = d["odds"] > 0
    d["q"] = np.nan
    d.loc[ok, "q"] = mcm.q_from_odds(d.loc[ok, "race_id"].to_numpy(), d.loc[ok, "odds"].to_numpy())
    e = d[ok].copy()
    _, q3 = zoo.harville_topk(e["race_id"].to_numpy(), e["q"].to_numpy(), 1.0, 1.0)
    n = e.groupby("race_id")["race_id"].transform("size").to_numpy()
    e["e3"] = mcm.norm_race(e["race_id"].to_numpy(), q3, 1.0) * np.minimum(3, n)
    d = d.merge(e[KEY + ["e3"]], on=KEY, how="left")
    d["band"] = band_of(d["weight"])
    d["y1"] = (d["finish_position"] == 1).astype(float)
    d["y3"] = (d["finish_position"] <= 3).astype(float)
    d["done"] = d["finish_position"].notna()
    return d


def genryo_flag(d: pd.DataFrame) -> np.ndarray:
    """減量騎手の近似（結果の表に見習いの印が無いため）。同じレース・同じ性（牝とそれ以外）・同じ年齢の組で、
    斤量の最頻値より 1〜4kg 軽い騎乗を 1、最頻値の騎乗を 0。最頻値の割合が 6 割未満の組（ハンデ戦など）と 3 頭未満の組は欠け。"""
    sa = d["sex_age"].astype(str)
    grp = d["race_id"].astype(str) + "|" + np.where(sa.str.startswith("牝"), "F", "M") + "|" + sa.str[1:]
    jw = d["jockey_weight"].to_numpy(float)
    mode = pd.Series(jw).groupby(grp.to_numpy()).transform(lambda x: x.mode().iat[0] if x.notna().any() else np.nan).to_numpy()
    share = pd.Series(jw == mode).groupby(grp.to_numpy()).transform("mean").to_numpy()
    size = pd.Series(jw).groupby(grp.to_numpy()).transform("size").to_numpy()
    diff = mode - jw
    out = np.where(np.isclose(diff, 0), 0.0, np.where((diff >= 1) & (diff <= 4), 1.0, np.nan))
    return np.where((share >= 0.6) & (size >= 3), out, np.nan)


def asof_sums(d: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """keys × 日付ごとに、その日までの累計（騎乗数・勝ち・複勝・見込みの勝ち・複勝）。当てるときに同じ日を除く（lookup）。"""
    s = d[d["done"]].assign(n=1.0, e3f=lambda x: x["e3"].fillna(0.0), y3e=lambda x: np.where(x["e3"].notna(), x["y3"], 0.0),
                            qf=lambda x: x["q"].fillna(0.0), y1q=lambda x: np.where(x["q"].notna(), x["y1"], 0.0))
    g = s.groupby(keys + ["date"])[SUMS].sum().sort_index()
    return g.groupby(level=list(range(len(keys)))).cumsum().reset_index()


def lookup(d: pd.DataFrame, keys: list[str], band_col: str, c: pd.DataFrame) -> pd.DataFrame:
    """行ごとに、その行の日付より厳密に前の日の最後の累計を当てる（同じ日は数えない）。"""
    left = d[["race_id", "horse_id", "date", "jockey_id", band_col]].rename(columns={band_col: "band"}).copy()
    left["_i"] = np.arange(len(left))
    right = c.rename(columns={"date": "cdate"}).sort_values("cdate")
    lg = left.dropna(subset=keys).sort_values("date")
    r = pd.merge_asof(lg, right, left_on="date", right_on="cdate", by=keys, direction="backward", allow_exact_matches=False)
    full = pd.DataFrame({"_i": np.arange(len(left))}).merge(r[["_i"] + SUMS], on="_i", how="left")
    return full.fillna(0.0)


def build_features(d: pd.DataFrame, band_col: str) -> pd.DataFrame:
    """騎手 × 帯（band_col の帯で引く）の縮めた勝率・複勝率・実/見込みの比（騎手全体で割った交互作用だけの値）・騎乗数。"""
    cj = asof_sums(d, ["jockey_id"])
    cjb = asof_sums(d.dropna(subset=["band"]), ["jockey_id", "band"])
    j = lookup(d, ["jockey_id"], band_col, cj)
    jb = lookup(d, ["jockey_id", "band"], band_col, cjb)
    done = d[d["done"]]
    early = done[done["date"] < pd.Timestamp("2019-01-01")]              # 全体平均（縮めの行き先の定数）は、最初の検証年より前の行だけで当てる
    done = early if len(early) else done
    g1, g3 = float(done["y1"].mean()), float(done["y3"].mean())
    j1 = (j["y1"] + K_JOCKEY * g1) / (j["n"] + K_JOCKEY)
    j3 = (j["y3"] + K_JOCKEY * g3) / (j["n"] + K_JOCKEY)
    jae = (j["y3e"] + K_AE) / (j["e3f"] + K_AE)
    jae1 = (j["y1q"] + K_AE1) / (j["qf"] + K_AE1)
    out = pd.DataFrame({
        "jwb_top3": (jb["y3"] + K_BAND * j3) / (jb["n"] + K_BAND),
        "jwb_win": (jb["y1"] + K_BAND * j1) / (jb["n"] + K_BAND),
        "jwb_ae3": np.log(((jb["y3e"] + K_AE * jae) / (jb["e3f"] + K_AE)) / jae),
        "jwb_n": jb["n"],
        "jwb_ae1": np.log(((jb["y1q"] + K_AE1 * jae1) / (jb["qf"] + K_AE1)) / jae1),   # 単勝版（直接の検定だけに使う。xgb の腕には入れていない）
    })
    nob = d[band_col].isna().to_numpy() | d["jockey_id"].isin(["", "nan", "None"]).to_numpy()
    out.loc[nob, COLS + EXTRA_COLS] = np.nan
    return out


def cmd_feats(_a) -> None:
    t0 = time.time()
    d = load_rows().sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    d["band_prev"] = d.groupby("horse_id")["band"].shift(1)            # 前走（この表の中の 1 つ前の出走）の馬体重の帯
    today = build_features(d, "band")
    prev = build_features(d, "band_prev")
    o = d[KEY + ["date", "jockey_id", "band", "band_prev"]].copy()
    for c in COLS + EXTRA_COLS:
        o[f"today_{c}"] = today[c].to_numpy()
        o[f"prev_{c}"] = prev[c].to_numpy()
    o["genryo"] = genryo_flag(d)
    OUT.mkdir(parents=True, exist_ok=True)
    o.to_parquet(OUT / "feats.parquet", index=False)
    log(f"feats: {len(o)} 行 {time.time() - t0:.0f}s 欠け today {o['today_jwb_top3'].isna().mean():.3f} prev {o['prev_jwb_top3'].isna().mean():.3f}")


def feats_path() -> Path:
    return Path(os.environ.get("JW_FEATS") or OUT / "feats.parquet")


def attach(X: pd.DataFrame, meta: pd.DataFrame, arm: str) -> pd.DataFrame:
    if arm == "base":
        return X
    f = pd.read_parquet(feats_path())
    f["race_id"], f["horse_id"] = f["race_id"].astype(str), f["horse_id"].astype(str)
    src = "prev" if arm == "prev" else "today"
    j = meta[KEY + ["year"]].merge(f[KEY + [f"{src}_{c}" for c in COLS]], on=KEY, how="left")
    add = j[[f"{src}_{c}" for c in COLS]].to_numpy(float)
    if arm == "perm":                                # 負の対照: 列の組を年の中で行ごと並べ替える（分布は同じ・行との結び付きだけ壊す）
        rng = np.random.default_rng(20261118)
        yr = j["year"].to_numpy()
        for y in np.unique(yr):
            idx = np.flatnonzero(yr == y)
            add[idx] = add[rng.permutation(idx)]
    X = X.copy()
    for i, c in enumerate(COLS):
        X[c] = add[:, i]
    return X


def cmd_member(a) -> None:
    from src.pipeline.models import zoo

    if a.arm not in ARMS:
        raise SystemExit(f"ARM は {ARMS} のどれか")
    vys = tuple(int(y) for y in a.valid_years.split(","))
    if min(vys) < 2019:
        raise SystemExit("縮めの行き先の全体平均を 2019 年より前の行で当てているので、検証は 2019 年から")
    if any(y in FORBIDDEN_YEARS for y in vys):
        raise SystemExit("2025・26 年は使わない（台帳の決まり）")
    out = OUT / "oof" / f"{a.arm}_{'-'.join(map(str, vys))}.parquet"
    if out.exists() and not a.force:
        log(f"{out} 既存")
        return
    X, meta, info = mc.load_cache()
    keep = ~meta["year"].isin(FORBIDDEN_YEARS).to_numpy()            # 学習・検証とも 2025 年の行は触れない
    X, meta = X[keep].reset_index(drop=True), meta[keep].reset_index(drop=True)
    n0 = X.shape[1]
    X = mc.apply_spec(X, meta, {k: v for k, v in SPEC.items() if k not in ("offset", "offset_noise_sd")})
    if n0 - X.shape[1] < mcm.BASE_DROP_N:
        raise SystemExit("土台の 38 列の除外が効いていない")
    X = attach(X, meta, a.arm)
    mcm.check_no_market_features(X.columns)
    params = dict(info["params"])
    race = meta["race_id"].to_numpy()
    year = meta["year"].to_numpy()
    fin = meta["finish_position"].to_numpy().astype(float)
    q = mcm.load_market(meta, "raw")
    margin = mcm.logit(q)
    y = (fin == 1).astype(int)
    noise = float(SPEC["offset_noise_sd"])
    rng = np.random.default_rng(12345)
    pred = np.full(len(meta), np.nan)
    t0 = time.time()
    for tr, va in mc.folds_from_meta(meta, valid_years=vys):
        tr = tr[np.isfinite(fin[tr])]
        acc = np.zeros(len(va))
        for s in range(a.seeds):
            mtr = margin[tr] + rng.normal(0.0, noise, len(tr))
            acc += mcm.fit_pred(X.iloc[tr], y[tr], X.iloc[va], params, s, mtr, margin[va])
        pred[va] = acc / a.seeds
        log(f"{a.arm} fold {int(year[va][0])} 済み（{time.time() - t0:.0f}s）")
    ok = np.isfinite(pred)
    o = meta.loc[ok, KEY + ["year", "finish_position"]].copy().reset_index(drop=True)
    o["p_win"] = mcm.norm_race(race[ok], pred[ok], 1.0)
    o["q"] = q[ok]
    out.parent.mkdir(parents=True, exist_ok=True)
    o.to_parquet(out, index=False)
    out.with_suffix(".json").write_text(json.dumps({"arm": a.arm, "spec": SPEC, "seeds": a.seeds, "cols": int(X.shape[1]), "valid_years": list(vys), "feats": str(feats_path()),
                                                    "seconds": round(time.time() - t0)}, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"{a.arm}: cols={X.shape[1]} {time.time() - t0:.0f}s → {out}")


def winner_ll(o: pd.DataFrame, col: str) -> pd.DataFrame:
    """レースごとの勝ち馬の -log p（勝ち馬が 1 頭のレースだけ）。"""
    w = o[o["finish_position"] == 1]
    w = w[w.groupby("race_id")["race_id"].transform("size") == 1]
    return pd.DataFrame({"race_id": w["race_id"].to_numpy(), "year": w["year"].to_numpy(), "ll": -np.log(np.clip(w[col].to_numpy(float), 1e-9, 1.0))})


def cluster_boot(diff: pd.Series, cluster: pd.Series, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """開催日 × 場（race_id の先頭 10 桁）で束ねたブートストラップの 95% 区間（平均の差）。"""
    g = pd.DataFrame({"d": diff.to_numpy(), "c": cluster.to_numpy()}).groupby("c")["d"].agg(["sum", "size"])
    s, k = g["sum"].to_numpy(), g["size"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n, len(g)))
    m = s[idx].sum(1) / k[idx].sum(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def compare(base: pd.DataFrame, arm: pd.DataFrame) -> dict:
    a = winner_ll(base, "p_win").merge(winner_ll(arm, "p_win"), on=["race_id", "year"], suffixes=("_b", "_a"))
    a["d"] = a["ll_a"] - a["ll_b"]                   # 負なら足した版が良い
    lo, hi = cluster_boot(a["d"], a["race_id"].str[:10])
    by_year = {int(y): round(float(g["d"].mean()), 5) for y, g in a.groupby("year")}
    return {"races": int(len(a)), "ll_base": round(float(a["ll_b"].mean()), 5), "ll_arm": round(float(a["ll_a"].mean()), 5),
            "diff": round(float(a["d"].mean()), 5), "ci95": [round(lo, 5), round(hi, 5)], "by_year": by_year}


def cmd_report(a) -> None:
    tag = "-".join(a.valid_years.split(","))
    oof = {k: pd.read_parquet(OUT / "oof" / f"{k}_{tag}.parquet") for k in ARMS if (OUT / "oof" / f"{k}_{tag}.parquet").exists()}
    if "base" not in oof:
        raise SystemExit("基準（base）の OOF が無い")
    for o in oof.values():
        o["race_id"] = o["race_id"].astype(str)
    res = {"valid_years": a.valid_years, "market_ll": None, "arms": {}}
    res["market_ll"] = round(float(winner_ll(oof["base"], "q")["ll"].mean()), 5)
    for k, o in oof.items():
        if k != "base":
            res["arms"][k] = compare(oof["base"], o)
    (OUT / f"report_{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


def cond_logit(race: np.ndarray, off: np.ndarray, Z: np.ndarray, win: np.ndarray, iters: int = 25) -> tuple[np.ndarray, np.ndarray, float]:
    """勝ち馬の条件付き logit（レース内の softmax）。log p = off + Z·β をレース内で正規化。Newton 法。
    返り値: β・β の標準誤差・当てはめた β での勝ち馬の平均 -log p。行は race で並んでいること。"""
    k = Z.shape[1]
    b = np.zeros(k)
    starts = np.flatnonzero(np.r_[True, race[1:] != race[:-1]])
    nr = len(starts)
    for _ in range(iters):
        eta = off + Z @ b
        eta = eta - np.maximum.reduceat(eta, starts).repeat(np.diff(np.r_[starts, len(race)]))
        e = np.exp(eta)
        ssum = np.add.reduceat(e, starts)
        p = e / ssum.repeat(np.diff(np.r_[starts, len(race)]))
        ez = np.add.reduceat(p[:, None] * Z, starts)
        g = (Z[win == 1].sum(0) - ez.sum(0))
        ezz = np.add.reduceat(p[:, None, None] * Z[:, :, None] * Z[:, None, :], starts)
        H = -(ezz - ez[:, :, None] * ez[:, None, :]).sum(0)
        step = np.linalg.solve(H, g)
        b = b - step
        if np.abs(step).max() < 1e-8:
            break
    else:
        log(f"警告: 条件付き logit が {iters} 回で収束しなかった（最後の一歩 {np.abs(step).max():.2e}）")
    se = np.sqrt(np.diag(np.linalg.inv(-H)))
    ll = -float(np.log(p[win == 1]).sum() / nr)
    return b, se, ll


def apply_ll(race: np.ndarray, off: np.ndarray, Z: np.ndarray, win: np.ndarray, b: np.ndarray) -> float:
    starts = np.flatnonzero(np.r_[True, race[1:] != race[:-1]])
    eta = off + Z @ b
    lse = np.log(np.add.reduceat(np.exp(eta - eta.max()), starts)) + eta.max()
    return -float((eta[win == 1] - lse).mean())


DIRECT_TESTS = {                                   # 名前 → (比べる相手に入れる項, 確かめる項)。genryo0 は減量の旗（欠けは 0）、genryo_miss は欠けの印
    "today_ae3": ([], ["today_jwb_ae3"]), "today_ae1": ([], ["today_jwb_ae1"]), "prev_ae3": ([], ["prev_jwb_ae3"]), "prev_ae1": ([], ["prev_jwb_ae1"]),
    "today_win": ([], ["today_jwb_win"]), "today_top3": ([], ["today_jwb_top3"]),
    "genryo": (["genryo_miss"], ["genryo0"]),
    "genryo_x_band": (["genryo_miss", "genryo0"], [f"genryo_x_{b}" for b in BAND_LABELS[1:]]),
}


def _std(Z: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """rows の行の平均・SD で標準化し、欠けは 0（平均）にする。"""
    mu, sd = np.nanmean(Z[rows], 0), np.nanstd(Z[rows], 0)
    return np.nan_to_num((Z - mu) / np.where(sd > 0, sd, 1.0))


def cmd_direct(a) -> None:
    """基準の OOF（xgb）の log p を offset にして、列を 1 係数（または数係数）で直接足す検定。
    xgb に列を足す比べ方は colsample の抽選の揺れが大きいので、小さな効きはこちらで見る（レビューの指摘）。
    比べる相手（null の項だけを足した版）との尤度比で検定する。年を 1 つ外して当て（標準化も残りの年だけで）、外した年で logloss の差も出す。"""
    from scipy.stats import chi2

    tag = "-".join(a.valid_years.split(","))
    o = pd.read_parquet(OUT / "oof" / f"base_{tag}.parquet")
    o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
    f = pd.read_parquet(feats_path())
    f["race_id"], f["horse_id"] = f["race_id"].astype(str), f["horse_id"].astype(str)
    d = o.merge(f, on=KEY, how="left").sort_values(["race_id", "horse_id"]).reset_index(drop=True)
    w = d[d["finish_position"] == 1].groupby("race_id").size()
    d = d[d["race_id"].isin(w[w == 1].index)].reset_index(drop=True)          # 勝ち馬が 1 頭のレースだけ
    d["genryo_miss"] = d["genryo"].isna().astype(float)
    d["genryo0"] = d["genryo"].fillna(0.0)
    for b in BAND_LABELS[1:]:                                                   # 減量 × 帯（〜439 を基準）。欠けの行は 0
        d[f"genryo_x_{b}"] = np.where(d["band"] == b, d["genryo0"], 0.0)
    race = d["race_id"].to_numpy()
    off = np.log(np.clip(d["p_win"].to_numpy(float), 1e-9, 1.0))
    win = (d["finish_position"] == 1).to_numpy().astype(int)
    year = d["year"].to_numpy()
    nr = int(len(w[w == 1]))
    allrows = np.ones(len(d), bool)

    def fit_ll(Z, rows):
        if Z.shape[1] == 0:
            return np.zeros(0), np.zeros(0), apply_ll(race[rows], off[rows], np.zeros((int(rows.sum()), 1)), win[rows], np.zeros(1))
        return cond_logit(race[rows], off[rows], Z[rows], win[rows])

    ll0 = fit_ll(np.zeros((len(d), 0)), allrows)[2]
    res = {"valid_years": a.valid_years, "races": nr, "ll_base": round(ll0, 5), "feats": str(feats_path()), "tests": {}}
    for name, (null, cols) in DIRECT_TESTS.items():
        Z = _std(d[null + cols].to_numpy(float), allrows)
        _, _, ll_null = fit_ll(Z[:, :len(null)], allrows)
        b, se, ll = fit_ll(Z, allrows)
        lr = 2 * (ll_null - ll) * nr
        loyo = {}
        for y in np.unique(year):
            tr, te = year != y, year == y
            Zy = _std(d[null + cols].to_numpy(float), tr)              # 標準化も外した年を除いて当てる
            bn, _, _ = fit_ll(Zy[:, :len(null)], tr)
            bf, _, _ = fit_ll(Zy, tr)
            ll_te_null = apply_ll(race[te], off[te], Zy[te][:, :max(len(null), 1)] if null else np.zeros((int(te.sum()), 1)), win[te],
                                  bn if null else np.zeros(1))
            loyo[int(y)] = round(apply_ll(race[te], off[te], Zy[te], win[te], bf) - ll_te_null, 6)
        k = len(null)
        res["tests"][name] = {"null": null, "cols": cols, "beta_per_sd": [round(float(x), 4) for x in b[k:]], "se": [round(float(x), 4) for x in se[k:]],
                              "z": [round(float(x), 2) for x in (b / se)[k:]], "ll_gain_in_sample": round(ll_null - ll, 6), "lr": round(float(lr), 2),
                              "df": len(cols), "p": float(chi2.sf(lr, len(cols))), "loyo_ll_diff": loyo,
                              "nonmissing": round(float(d[cols[0]].notna().mean()) if not name.startswith("genryo") else 1 - float(d["genryo_miss"].mean()), 3)}
    ps = sorted((v["p"], k_) for k_, v in res["tests"].items())               # Holm の補正（検定の数で割り引く）
    m = len(ps)
    adj, run = {}, 0.0
    for i_, (pv, k_) in enumerate(ps):
        run = max(run, min(1.0, (m - i_) * pv))
        adj[k_] = run
    for k_, v in res["tests"].items():
        v["p_holm"] = round(adj[k_], 4)
        v["p"] = round(v["p"], 4)
    (OUT / f"direct_{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("feats").set_defaults(fn=cmd_feats)
    m = sp.add_parser("member")
    m.add_argument("arm")
    m.add_argument("--valid-years", default="2019,2020,2021")
    m.add_argument("--seeds", type=int, default=3)
    m.add_argument("--force", action="store_true")
    m.set_defaults(fn=cmd_member)
    dr = sp.add_parser("direct")
    dr.add_argument("--valid-years", default="2019,2020,2021")
    dr.set_defaults(fn=cmd_direct)
    r = sp.add_parser("report")
    r.add_argument("--valid-years", default="2019,2020,2021")
    r.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    if not os.environ.get("CYCLES_OUT"):           # import の時点では環境を書き換えない（同じプロセスのほかの import に効かせない）
        mc.CYC = DEFAULT_CYC
        mcm.MK = DEFAULT_CYC / "market"
    a.fn(a)


if __name__ == "__main__":
    main()
