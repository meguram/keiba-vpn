"""tp-39: 成長曲線（馬体重の変化 × パフォーマンスの馬ごとの傾き = 補正ポイント）の分析と特徴量の評価。

python -m src.scripts.data.tp39_eval metrics    # パフォーマンス指標の比較（被覆・信頼性 ICC・相関）
python -m src.scripts.data.tp39_eval describe   # 馬ごとの傾きの個体差（置換検定）、前半/後半の再現性、母集団の応答（年齢・性別・距離・面・季節・馬格）、V4 残差
python -m src.scripts.data.tp39_eval groups     # 定数を FIT_YEARS（2020–21）で作り、特徴量の群を tp39/groups/*.parquet に保存
python -m src.scripts.data.tp39_eval prior      # 母集団の成長曲線を縮約の事前分布にしたときの予測の質（2022–24）、履歴ゼロの馬への適用
python -m src.scripts.data.tp39_eval stage1     # 第 1 段: valid 2022+2023、seed 1 本。V4 全体に対する増分（群ごとの OOF も保存）
python -m src.scripts.data.tp39_eval stage2     # 第 2 段: 第 1 段を通った群を valid 2024、新 seed 2 本で確認
python -m src.scripts.data.tp39_eval subset     # 履歴 0〜2 走の馬だけでの増分（stage の OOF を使う）
python -m src.scripts.data.tp39_eval flagged    # 補正ポイントの高い/低い馬の V4 に対する成績（馬単位とレース単位）
python -m src.scripts.data.tp39_eval boost      # V4 の出力を初期値にして、新しい特徴量だけで補正する LightGBM（threads=2・数分。V4 全体の再学習はしない）。群ごと/まとめて、履歴 0〜2 走も
python -m src.scripts.data.tp39_eval stack      # 軽い一次選別: V4 の対数確率に特徴量を 1 つ足す条件付きロジット（係数 γ を 2022–23 で当て 2024 で検証）。GBM の再学習なし
test 2025・2026 年は使わない。市場オッズ・人気は使わない。環境変数: TP39_THREADS（既定 3）, TP39_LOAD_LIMIT（既定 14）。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
OUT = V / "tp39"
FRAME31 = V / "tp31"
for _k, _v in {"TP_BASE_DIR": V, "TP_WORK": V, "TP25C_OUT": V / "tp25c", "TP25E_OUT": V / "tp25e", "TP27_FAC_FIELD": V / "factory_field", "TP27_OUT": V / "tp27"}.items():
    os.environ.setdefault(_k, str(_v))
os.environ.setdefault("TP27_THREADS", os.environ.get("TP39_THREADS", "3"))

from src.pipeline.features import factory_self as fs  # noqa: E402
from src.pipeline.features import tp39_growth as tg  # noqa: E402

MEGU_ROOT = ROOT / os.environ.get("TP39_MEGU_ROOT", tg.MEGU_ROOT)           # めぐ指数の読み先（既定: 最終候補 v3a。page_reference 反映後は TP39_MEGU_ROOT=data/page_reference/tables）

KEY = ["race_id", "horse_id"]
THREADS = int(os.environ.get("TP39_THREADS", 3))
LOAD_LIMIT = float(os.environ.get("TP39_LOAD_LIMIT", 12))
MAX_HEAVY = int(os.environ.get("TP39_MAX_HEAVY", 3))            # 全体で同時に動かす重い処理は 3 本まで（自分を入れて）→ 他が 3 本以上いる間は待つ
GROUP_NAMES = list(tg.GROUPS)
VALID_YEARS = (2022, 2023, 2024)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _heavy_others() -> int:
    """自分以外で動いている重い評価・学習（stage / ridge / fit など）の数。"""
    import re
    import subprocess

    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
    n = 0
    for line in out:
        l = line.strip()
        if "tp39_eval" in l or not l.startswith(("/opt/conda/bin/python", "python")):
            continue
        if re.search(r"(_eval|run\.py).*\b(stage\d*|ridge|train|fit|member|ens|mtl)\b", l):
            n += 1
    return n


def _wait() -> None:
    """ホストの負荷が LOAD_LIMIT を下回り、かつ他の重い評価が MAX_HEAVY 本未満になるまで待つ（上限なし）。"""
    while os.getloadavg()[0] >= LOAD_LIMIT or _heavy_others() >= MAX_HEAVY:
        time.sleep(30)


def _jsave(p: Path, d) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, p)


def _jload(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _load_d() -> tuple[pd.DataFrame, dict]:
    """prepare 済みの走表と定数。定数は OUT/consts.json があれば使うが、読んだ megu の版が違えば作り直す（meg / comp の定数が megu に依存するため）。"""
    runs = fs.load_runs(TABLES, list(range(2020, 2026)))
    megu = tg.load_megu(MEGU_ROOT, list(range(2020, 2026)))
    d = tg.prepare(runs, megu)
    ver = {"megu": megu.attrs["megu"]}
    cp = OUT / "consts.json"
    consts = json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else {}
    if not consts or consts.get("megu") != ver["megu"] or "mga_chg" not in consts.get("shrink", {}):
        consts = tg.fit_constants(d)
        consts.update(ver)
        _jsave(cp, consts)
    return d, consts


def _full(d: pd.DataFrame, consts: dict) -> pd.DataFrame:
    return tg.add_size(tg.add_residuals(tg.apply_metrics(d, consts), consts), consts)


# ───────────────────────── 指標の比較
def _icc1(df: pd.DataFrame, col: str, min_n: int = 6) -> dict:
    s = df[["horse_id", col]].dropna()
    n = s.groupby("horse_id")[col].transform("size")
    s = s[n >= min_n]
    g = s.groupby("horse_id")[col]
    ni = g.size().to_numpy(float)
    mi = g.mean().to_numpy(float)
    N, k = ni.sum(), len(ni)
    gm = s[col].mean()
    msb = (ni * (mi - gm) ** 2).sum() / (k - 1)
    msw = ((s[col] - g.transform("mean")) ** 2).sum() / (N - k)
    n0 = (N - (ni ** 2).sum() / N) / (k - 1)
    return {"horses": int(k), "rows": int(N), "icc1": float((msb - msw) / (msb + (n0 - 1) * msw)), "var_total": float(s[col].var())}


def cmd_metrics(a) -> None:
    d, consts = _load_d()
    d = _full(d, consts)
    res: dict = {"megu": consts.get("megu"), "coverage": {}, "icc_P": {}, "icc_R": {}, "corr": {}, "notes": {}}
    for y, g in d.groupby("year"):
        res["coverage"][int(y)] = {m: float(g[f"P_{m}"].notna().mean()) for m in tg.METRICS}
    fy = d[d["year"].isin(tg.FIT_YEARS)]
    for m in tg.METRICS:
        res["icc_P"][m] = _icc1(fy, f"P_{m}")
        res["icc_R"][m] = _icc1(fy, f"R_{m}")
    res["corr"] = fy[[f"P_{m}" for m in tg.METRICS]].corr().round(3).to_dict()
    res["notes"] = {"scale": consts["scale"], "megu_valid_share": float(d["megu_index"].notna().mean()),
                    "time_index": "netkeiba のタイム指数は per-run の値がローカルの result_tbl に無く、race_index の time_index_m / speed_* は出走前の値（as-of 未検証）なので、自前の指数 megu_index・着順の正規スコア・タイム偏差を使う"}
    _jsave(OUT / "metrics.json", res)
    _log("指標の比較を保存")
    for m in tg.METRICS:
        _log(f"  {m}: ICC(P)={res['icc_P'][m]['icc1']:.3f} ICC(R)={res['icc_R'][m]['icc1']:.3f} 被覆(2022)={res['coverage'][2022][m]:.3f}")


# ───────────────────────── 馬ごとの傾きの個体差
def _ols_by_horse(code: np.ndarray, x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    n = np.bincount(code, minlength=code.max() + 1).astype(float)
    sx, sy = np.bincount(code, x), np.bincount(code, y)
    sxx, sxy, syy = np.bincount(code, x * x), np.bincount(code, x * y), np.bincount(code, y * y)
    with np.errstate(invalid="ignore", divide="ignore"):
        cxx, cxy, cyy = sxx - sx ** 2 / n, sxy - sx * sy / n, syy - sy ** 2 / n
        b = cxy / cxx
        rss = np.maximum(cyy - b * cxy, 0.0)
        se = np.sqrt(rss / np.maximum(n - 2, 1) / cxx)
    return pd.DataFrame({"n": n, "cxx": cxx, "b": b, "se": se, "t": b / se})


def _prep_pairs(d: pd.DataFrame, m: str, kind: str, min_pairs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    v = d[f"R_{m}"].notna() & d[f"x_{kind}"].notna()
    s = d.loc[v, ["horse_id", "dnum", f"R_{m}", f"x_{kind}"]].copy()
    s = s[s.groupby("horse_id")["dnum"].transform("size") >= min_pairs]
    s["code"] = pd.factorize(s["horse_id"])[0]
    return s["code"].to_numpy(), s[f"x_{kind}"].to_numpy(float), s[f"R_{m}"].to_numpy(float), s


def _perm_x(code: np.ndarray, x: np.ndarray, rng) -> np.ndarray:
    """馬ごとに x の並びを入れ替える（馬の中での日付順序を壊す = 傾きの帰無分布）。"""
    order = np.argsort(code + rng.random(len(code)) * 0.999, kind="stable")
    out = np.empty_like(x)
    # 元の馬ごとの位置集合へ、同じ馬の x をランダムな順で入れる
    idx_sorted = np.argsort(code, kind="stable")
    out[idx_sorted] = x[order]
    return out


def _split_half(s: pd.DataFrame, code: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    """馬ごとに時系列の前半/後半の傾きを出し、馬の間での相関（再現性）を返す。s は日付順に並んでいること。"""
    pos = s.groupby("horse_id").cumcount().to_numpy()
    n = s.groupby("horse_id")["dnum"].transform("size").to_numpy()
    first = pos < (n // 2)
    second = pos >= (n - n // 2)
    # code が連続でなくなるので再番号
    c1 = pd.factorize(code[first])[0]
    c2 = pd.factorize(code[second])[0]
    k1, k2 = pd.Series(code[first]).drop_duplicates().to_numpy(), pd.Series(code[second]).drop_duplicates().to_numpy()
    r1, r2 = _ols_by_horse(c1, x[first], y[first]), _ols_by_horse(c2, x[second], y[second])
    r1.index, r2.index = k1, k2
    j = r1.join(r2, lsuffix="1", rsuffix="2", how="inner").replace([np.inf, -np.inf], np.nan).dropna()
    j = j[(j["n1"] >= 5) & (j["n2"] >= 5) & (j["cxx1"] > 1e-3) & (j["cxx2"] > 1e-3)]
    if len(j) < 30:
        return {"horses": int(len(j)), "r": None}
    return {"horses": int(len(j)), "r": float(np.corrcoef(j["b1"].clip(-4, 4), j["b2"].clip(-4, 4))[0, 1])}


def cmd_describe(a) -> None:
    d, consts = _load_d()
    d = _full(d, consts)
    res: dict = {"megu": consts.get("megu"), "consts": {"shrink": consts["shrink"], "scale": consts["scale"], "fit_years": consts["fit_years"]}, "hetero": {}, "population": {}, "v4_resid": {}}
    rng = np.random.default_rng(0)
    B = 40
    for m in tg.FEATURE_METRICS:
        for kind in tg.KINDS:
            code, x, y, s = _prep_pairs(d, m, kind, 10)
            if len(s) < 1000:
                continue
            obs = _ols_by_horse(code, x, y)
            obs = obs[(obs["n"] >= 10) & (obs["cxx"] > 1e-3)].replace([np.inf, -np.inf], np.nan).dropna()
            sh = lambda t: {"pos": float((t > 2).mean()), "neg": float((t < -2).mean()), "none": float((t.abs() <= 2).mean())}
            nulls = []
            sp_null = []
            for _ in range(B):
                xp = _perm_x(code, x, rng)
                r = _ols_by_horse(code, xp, y)
                r = r[(r["n"] >= 10) & (r["cxx"] > 1e-3)].replace([np.inf, -np.inf], np.nan).dropna()
                nulls.append(sh(r["t"]))
                sp_null.append(_split_half(s, code, xp, y)["r"])
            nm = {k: float(np.mean([z[k] for z in nulls])) for k in ("pos", "neg", "none")}
            sp = _split_half(s, code, x, y)
            sp_n = [v for v in sp_null if v is not None]
            sd_n = float(np.std(sp_n)) if sp_n else None
            res["hetero"][f"{m}_{kind}"] = {
                "horses": int(len(obs)), "pairs_per_horse_median": float(np.median(obs["n"])), "beta_mean": float(obs["b"].clip(-4, 4).mean()),
                "beta_sd": float(obs["b"].clip(-4, 4).std()), "observed_share": sh(obs["t"]), "null_share": nm,
                "excess": {k: sh(obs["t"])[k] - nm[k] for k in ("pos", "neg")}, "split_half": sp,
                "split_half_null_mean": float(np.mean(sp_n)) if sp_n else None, "split_half_null_sd": sd_n,
                "note": "t>2: 増えると上がる / t<-2: 増えると下がる（減ると上がる）/ それ以外: 無関係。null = 馬ごとに x の順序を入れ替えた帰無分布（40 回の平均）。split_half = 時系列の前半/後半の傾きの馬間相関"}
            _log(f"  {m}_{kind}: horses={len(obs)} 増えると上がる {sh(obs['t'])['pos']:.3f} (null {nm['pos']:.3f}) 下がる {sh(obs['t'])['neg']:.3f} (null {nm['neg']:.3f}) split-half r={sp['r']}")
    # 母集団の応答（R の馬内偏差）の概形
    pop = consts["pop"]
    res["population"]["chg_marg"] = {m: pop["chg"][m]["l3"] for m in tg.FEATURE_METRICS}
    res["population"]["lev_marg"] = {m: pop["lev"][m]["l3"] for m in tg.FEATURE_METRICS}
    res["population"]["beta_global"] = {k: v["global"] for k, v in pop["beta"].items()}
    cells = {}
    for k, v in pop["beta"].items():
        cells[k] = dict(sorted(v["cells"].items())[:200])
    res["population"]["beta_cells"] = cells
    # 年齢・性別・距離・面・季節・馬格ごとの、体重変化の帯別の R の馬内偏差（FIT_YEARS 以降の全期間、記述）
    s = d[d["chg_bin"].notna() & d["R_pt"].notna()].copy()
    cnt = s.groupby("horse_id")["R_pt"].transform("count")
    s["dev"] = (s["R_pt"] - s.groupby("horse_id")["R_pt"].transform("mean")).where(cnt >= 4)
    s = s[s["dev"].notna()]
    s["cat"] = pd.cut(s["wt_chg"], [-99, -9.5, -3.5, 3.5, 9.5, 99], labels=["大幅減(-10kg~)", "減(-4~-9)", "維持(-3~+3)", "増(+4~+9)", "大幅増(+10kg~)"]).astype(str)
    s["dist_b"] = pd.cut(s["dist"], [0, 1300, 1700, 2100, 9999], labels=["~1300", "1400-1700", "1800-2100", "2200~"]).astype(str)
    def tbl(col):
        g = s.groupby([col, "cat"])["dev"].agg(["mean", "size", "std"])
        g["se"] = g["std"] / np.sqrt(g["size"])
        return {f"{i[0]}|{i[1]}": {"mean": float(r["mean"]), "n": int(r["size"]), "lo": float(r["mean"] - 1.96 * r["se"]), "hi": float(r["mean"] + 1.96 * r["se"])} for i, r in g.iterrows() if r["size"] >= 300}
    for col, nm_ in (("age_c", "年齢"), ("sex", "性別"), ("dist_b", "距離"), ("surface", "面"), ("season", "季節(0=冬,1=春,2=夏,3=秋)"), ("sz_bin", "馬格(0=小..4=大)")):
        res["population"][f"dev_by_{col}"] = {"label": nm_, "cells": tbl(col)}
    # V4 の残差（V4 OOF 2022–2024）に対する体重変化の帯
    ens = pd.read_parquet(V / "tp27/ens/v4_e2.parquet")
    ens["race_id"], ens["horse_id"] = ens["race_id"].astype(str), ens["horse_id"].astype(str)
    mm = ens.merge(d[KEY + ["fin", "wt_chg", "age_c", "sex", "dist", "surface", "season", "sz_bin"]], on=KEY, how="inner")
    mm["y_win"], mm["y_top2"], mm["y_top3"] = (mm["fin"] == 1).astype(float), (mm["fin"] <= 2).astype(float), (mm["fin"] <= 3).astype(float)
    mm["cat"] = pd.cut(mm["wt_chg"], [-99, -9.5, -3.5, 3.5, 9.5, 99], labels=["大幅減", "減", "維持", "増", "大幅増"]).astype("string")
    mm["dist_b"] = pd.cut(mm["dist"], [0, 1300, 1700, 2100, 9999], labels=["~1300", "1400-1700", "1800-2100", "2200~"]).astype("string")
    from src.scripts.data.tp34_eval import resid_table

    res["v4_resid"]["all"] = resid_table(mm[mm["cat"].notna()], "cat", ["大幅減", "減", "維持", "増", "大幅増"], years=False).to_dict("records")
    for col, nm_ in (("age_c", "年齢"), ("sex", "性別"), ("dist_b", "距離"), ("surface", "面")):
        t = mm[mm["cat"].isin(["大幅減", "大幅増"]) & mm[col].notna()].copy()
        t["cell"] = t[col].astype(str) + "|" + t["cat"].astype(str)
        res["v4_resid"][col] = resid_table(t, "cell", None, years=False).to_dict("records")
    _jsave(OUT / "describe.json", res)
    _log("記述分析を保存: " + str(OUT / "describe.json"))


# ───────────────────────── 群
def cmd_groups(a) -> None:
    from src.scripts.data import tp27_eval as t

    df, base, meta, folds, lp = t.setup()
    runs = fs.load_runs(TABLES, list(range(2020, 2026)))
    megu = tg.load_megu(MEGU_ROOT, list(range(2020, 2026)))
    t0 = time.time()
    d = tg.prepare(runs, megu)
    consts = tg.fit_constants(d)
    consts.update({"megu": megu.attrs["megu"]})
    _log(f"megu: {megu.attrs['megu']}")
    _jsave(OUT / "consts.json", consts)
    parts = tg.build_groups(d, consts)
    gdir = OUT / "groups"
    gdir.mkdir(parents=True, exist_ok=True)
    keys = df[KEY].reset_index(drop=True)
    npr = d[KEY].copy()
    npr["n_prior"] = d.groupby("horse_id").cumcount().to_numpy()
    npr["year"] = d["year"].to_numpy()
    npr.to_parquet(OUT / "nprior.parquet", index=False)
    for g, G in parts.items():
        T = keys.merge(G, on=KEY, how="left")
        feat = [c for c in T.columns if c not in KEY]
        nn = T[feat].notna().mean()
        keep = [c for c in feat if nn[c] > 0.02 and T[c].nunique(dropna=True) > 1]
        tmp = gdir / f"{g}.parquet.tmp"
        T[KEY + keep].to_parquet(tmp, index=False)
        os.replace(tmp, gdir / f"{g}.parquet")                       # 評価が読んでいる途中で壊れないよう、書いてから置き換える
        _log(f"{g}: {len(keep)} 列（除外 {len(feat) - len(keep)}）、欠損率 {float(T[keep].isna().mean().mean()):.3f} ({time.time() - t0:.0f}s)")


# ───────────────────────── 縮約の事前分布の質・履歴ゼロへの適用
def cmd_prior(a) -> None:
    d, consts = _load_d()
    d = _full(d, consts)
    parts = tg.build_groups(d, consts)
    POP = pd.concat([parts["POP_free"].drop(columns=KEY), parts["POP_megu"].drop(columns=KEY)], axis=1)
    y = d["year"].to_numpy()
    race = d["race_id"].to_numpy()
    n0 = d.groupby("horse_id").cumcount().to_numpy()
    res: dict = {"megu": consts.get("megu"),
                 "note": "FIT_YEARS(2020-21) の定数だけを使い、2022-24 の走で『その走の R − その馬の過去平均 R』を予測する。MSE が小さいほど良い。", "metrics": {}}
    for m in ("pt", "comp"):
        sl = parts[f"SL_{m}"].drop(columns=KEY)
        sl0 = parts[f"SL0_{m}"].drop(columns=KEY)
        out_m: dict = {"slope": {}, "debut": {}}
        for kind in tg.KINDS:
            t = f"gc_{kind}_{m}"
            t0_ = f"gc0_{kind}_{m}"
            dev = d[f"R_{m}"].to_numpy(float) - sl[f"{t}_ybar"].to_numpy(float)
            xc = d[f"x_{kind}"].to_numpy(float) - sl[f"{t}_xbar"].to_numpy(float)
            n = sl[f"{t}_n"].to_numpy(float)
            sh = consts["shrink"][f"{m}_{kind}"]
            raw = np.nan_to_num(sl[f"{t}_beta"].to_numpy(float), nan=0.0)
            pcol = f"gc_pop_{kind}_{m}"
            popt = POP[pcol].to_numpy(float) if pcol in POP else np.zeros(len(d))
            preds = {"ゼロ(母集団の傾き 0)": np.zeros(len(d)), "全体の傾き β0": sh["beta0"] * xc, "母集団曲線(年齢×性別×馬格の傾き)": sl[f"{t}_cpc_pop"].to_numpy(float),
                     "母集団の応答表(帯の平均)": np.nan_to_num(popt, nan=0.0), "馬ごと(縮約なし)": raw * xc,
                     "馬ごと(縮約先=全体の傾き)": sl0[f"{t0_}_cpc"].to_numpy(float), "馬ごと(縮約先=母集団曲線)": sl[f"{t}_cpc"].to_numpy(float)}
            ok = np.isin(y, VALID_YEARS) & np.isfinite(dev) & np.isfinite(xc) & (n >= 1)
            bins = {"履歴の対(n)=1〜2": (n >= 1) & (n <= 2), "3〜5": (n >= 3) & (n <= 5), "6〜10": (n >= 6) & (n <= 10), "11以上": n >= 11, "全体": n >= 1}
            out = {}
            for bn, bm in bins.items():
                mk = ok & bm
                if mk.sum() < 500:
                    continue
                base_se = dev[mk] ** 2
                row = {"rows": int(mk.sum())}
                for nm, p in preds.items():
                    se = (dev[mk] - np.nan_to_num(p[mk], nan=0.0)) ** 2
                    row[nm] = {"mse": float(se.mean()), "gain_vs_zero": float(base_se.mean() - se.mean())}
                a_ = (dev[mk] - np.nan_to_num(preds["馬ごと(縮約先=母集団曲線)"][mk])) ** 2
                b_ = (dev[mk] - np.nan_to_num(preds["馬ごと(縮約先=全体の傾き)"][mk])) ** 2
                pr = pd.DataFrame({"r": race[mk], "dd": a_ - b_}).groupby("r")["dd"].mean().to_numpy()
                rng = np.random.default_rng(0)
                bs = np.array([pr[rng.integers(0, len(pr), len(pr))].mean() for _ in range(1000)])
                row["prior_cell_minus_global_mse"] = {"diff": float(pr.mean()), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975))}
                out[bn] = row
            out_m["slope"][kind] = out
        # 履歴 0〜2 走の馬: R を母集団の馬格表で予測
        for lab, mk0 in (("履歴0走", n0 == 0), ("履歴1走", n0 == 1), ("履歴2走", n0 == 2)):
            r = d[f"R_{m}"].to_numpy(float)
            sz = POP[f"gc_pop_size_{m}"].to_numpy(float)
            mk = np.isin(y, VALID_YEARS) & mk0 & np.isfinite(r) & np.isfinite(sz)
            if mk.sum() < 500:
                continue
            e0, e1 = r[mk] ** 2, (r[mk] - sz[mk]) ** 2
            pr = pd.DataFrame({"r": race[mk], "dd": e1 - e0}).groupby("r")["dd"].mean().to_numpy()
            rng = np.random.default_rng(0)
            bs = np.array([pr[rng.integers(0, len(pr), len(pr))].mean() for _ in range(1000)])
            out_m["debut"][lab] = {"rows": int(mk.sum()), "mse_zero": float(e0.mean()), "mse_pop_size": float(e1.mean()), "diff": float(pr.mean()), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975))}
        res["metrics"][m] = out_m
        _log(f"prior {m}: " + ", ".join(f"{k}: " + "/".join(str(r["rows"]) for r in v.values()) for k, v in out_m["slope"].items()))
    _jsave(OUT / "prior.json", res)
    _log("縮約の事前分布の質を保存")


# ───────────────────────── 段階評価
def _load_group(g: str, df: pd.DataFrame) -> pd.DataFrame:
    T = pd.read_parquet(OUT / "groups" / f"{g}.parquet")
    T["race_id"], T["horse_id"] = T["race_id"].astype(str), T["horse_id"].astype(str)
    m = df[KEY].merge(T, on=KEY, how="left")
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


def _run_subset(name, X, meta, folds, lp, years: list[int], seeds: list[int]):
    from src.pipeline.models import zoo_runner

    fs_ = [f for f in folds if int(meta["year"].to_numpy()[f.valid_index][0]) in years]
    _wait()
    t0 = time.time()
    res, oof = zoo_runner.run_oof(name, "lgb_binary", X, meta, fs_, seeds, THREADS, params=dict(lp))
    lv = {mm: float(res[mm].mean()) for mm in ("logloss_win", "logloss_top2", "logloss_top3")}
    _log(f"{name}: win {lv['logloss_win']:.5f} top2 {lv['logloss_top2']:.5f} top3 {lv['logloss_top3']:.5f} cols={X.shape[1]} ({time.time() - t0:.0f}s)")
    return lv, oof


def _stage(label: str, years: list[int], seeds: list[int], groups: list[str], src31: str | None) -> None:
    from src.scripts.data import tp27_eval as t

    df, base, meta, folds, lp = t.setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    X0 = _frame(df)
    d = OUT / label
    d.mkdir(parents=True, exist_ok=True)
    resp = d / "results.json"
    res = _jload(resp)
    cv = _jload(OUT / "consts.json")
    res.setdefault("megu_versions_by_group", {})
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
        G = G.loc[:, ~G.columns.duplicated()]
        Xn = pd.concat([X0, G], axis=1)
        lv, oof = _run_subset(f"V4+{g}", Xn, meta, folds, lp, years, seeds)
        oof.to_parquet(d / f"oof_{g}.parquet", index=False)
        c = t._cmp(oof0, oof, fin_map, g)
        steps[g] = {"cols": int(G.shape[1]), "levels": lv, "vs_V4": c, "megu_free": g in tg.MEGU_FREE_GROUPS, "megu_version": cv.get("megu")}
        _jsave(resp, res)
        _log(f"  {g}: Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f} 採用={c['採用']}")


def cmd_stage1(a) -> None:
    groups = [g for g in os.environ.get("TP39_GROUPS", "").split(",") if g] or list(tg.MEGU_FREE_GROUPS)      # 既定は megu に依存しない群だけ（megu 系は最終版の通知後に TP39_GROUPS で指定）
    _stage(os.environ.get("TP39_LABEL", "stage1"), [2022, 2023], [0], groups, "stage1")


def cmd_stage2(a) -> None:
    allg: dict = {}
    for lab in ("stage1", "stage1b", "stage1c"):
        allg.update((_jload(OUT / lab / "results.json") or {}).get("groups", {}))
    passed = [g for g in GROUP_NAMES if g in allg and allg[g]["vs_V4"]["Δwin"] < -0.00012 and allg[g]["vs_V4"]["win_hi"] < 0.0001]
    _log(f"第 1 段を通過: {passed}")
    _jsave(OUT / "stage1_passed.json", passed)
    if not passed:
        return
    _stage("stage2", [2024], [2, 3], passed + (["ALL_PASSED"] if len(passed) > 1 else []), "stage2")


# ───────────────────────── 履歴 0〜2 走の馬だけの増分
def _cmp_rows(oof0: pd.DataFrame, oof: pd.DataFrame, fin_map: pd.Series, keep: pd.DataFrame, n_boot: int = 1000) -> dict:
    from src.pipeline.models import win_models as wm

    m = oof0.merge(oof, on=KEY, suffixes=("_b", "_v")).merge(keep, on=KEY, how="inner")
    fin = fin_map.loc[list(zip(m["race_id"], m["horse_id"]))].to_numpy()
    out: dict = {"rows": int(len(m)), "races": int(m["race_id"].nunique())}
    for t_, k in (("win", 1), ("top2", 2), ("top3", 3)):
        yy = (fin <= k).astype(float)
        rid = m["race_id"].to_numpy()
        bb = wm.paired_bootstrap_diff(wm.per_race_loss(rid, m[f"p_{t_}_b"].to_numpy(), yy), wm.per_race_loss(rid, m[f"p_{t_}_v"].to_numpy(), yy), n_boot=n_boot)
        out[f"Δ{t_}"], out[f"{t_}_lo"], out[f"{t_}_hi"] = bb["diff"], bb["lo"], bb["hi"]
    return out


def cmd_subset(a) -> None:
    from src.scripts.data import tp27_eval as t

    df, base, meta, folds, lp = t.setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    npr = pd.read_parquet(OUT / "nprior.parquet")
    npr["race_id"], npr["horse_id"] = npr["race_id"].astype(str), npr["horse_id"].astype(str)
    res: dict = {}
    for lab in ("stage1", "stage1b", "stage2"):
        rj = _jload(OUT / lab / "results.json")
        if not rj.get("groups"):
            continue
        oof0 = pd.read_parquet(OUT / lab / "base.parquet")
        for g in rj["groups"]:
            f = OUT / lab / f"oof_{g}.parquet"
            if not f.exists():
                continue
            oof = pd.read_parquet(f)
            r = {}
            for nm, lo, hi in (("履歴0走", 0, 0), ("履歴0〜2走", 0, 2), ("履歴3〜5走", 3, 5), ("履歴6走以上", 6, 999)):
                keep = npr[(npr["n_prior"] >= lo) & (npr["n_prior"] <= hi)][KEY]
                r[nm] = _cmp_rows(oof0, oof, fin_map, keep)
            res[f"{lab}|{g}"] = r
            _log(f"{lab}|{g}: 履歴0〜2走 Δwin {r['履歴0〜2走']['Δwin']:+.5f} [{r['履歴0〜2走']['win_lo']:+.5f},{r['履歴0〜2走']['win_hi']:+.5f}] rows={r['履歴0〜2走']['rows']}")
    _jsave(OUT / "subset.json", res)


# ───────────────────────── 補正ポイントの高い/低い馬の成績（馬単位とレース単位）
def cmd_flagged(a) -> None:
    d, consts = _load_d()
    d = _full(d, consts)
    parts = tg.build_groups(d, consts)
    ens = pd.read_parquet(V / "tp27/ens/v4_e2.parquet")
    ens["race_id"], ens["horse_id"] = ens["race_id"].astype(str), ens["horse_id"].astype(str)
    res: dict = {"megu": consts.get("megu"), "note": "V4 の OOF(2022-24)に対し、補正ポイント(縮約後, 体重変化)が上位/下位 10% の馬の成績。馬単位: 実際の率 − V4 予測の平均。レース単位: 該当馬が 1 頭以上いるレースで、該当馬のどれかが勝つ割合 − V4 での期待(Σp_win)"}
    for m in ("pt", "comp"):
        S = parts[f"SL_{m}"]
        mm = ens.merge(S, on=KEY, how="inner").merge(d[KEY + ["fin"]], on=KEY, how="inner")
        mm["y_win"], mm["y_top3"] = (mm["fin"] == 1).astype(float), (mm["fin"] <= 3).astype(float)
        res[m] = {"n_rows": int(len(mm))}
        for col in (f"gc_chg_{m}_cp", f"gc_chg_{m}_cp_ex", f"gc_lev_{m}_cp", f"gc_lev_{m}_cp_ex", f"gc_chg_{m}_cpc", f"gc_lev_{m}_cpc", f"gc_abs_{m}_cp", f"gc_abs_{m}_cp_ex", f"gc_labs_{m}_cp"):
            v = mm[col]
            ok = v.notna()
            hi_t, lo_t = v[ok].quantile(0.9), v[ok].quantile(0.1)
            out = {}
            for nm, mask in (("上位10%", ok & (v >= hi_t)), ("下位10%", ok & (v <= lo_t))):
                s_ = mm[mask]
                races = s_.groupby("race_id").agg(win=("y_win", "max"), exp_win=("p_win", "sum"), n=("y_win", "size"))
                rng = np.random.default_rng(0)
                w, e = races["win"].to_numpy(float), races["exp_win"].to_numpy(float)
                bs = np.array([(w[i] - e[i]).mean() for i in (rng.integers(0, len(w), len(w)) for _ in range(1000))])
                out[nm] = {"horses": int(len(s_)), "races": int(len(races)), "mean_per_race": float(races["n"].mean()),
                           "horse_win_obs": float(s_["y_win"].mean()), "horse_win_v4": float(s_["p_win"].mean()),
                           "horse_top3_obs": float(s_["y_top3"].mean()), "horse_top3_v4": float(s_["p_top3"].mean()),
                           "race_win_obs": float(w.mean()), "race_win_v4": float(e.mean()), "race_diff": float((w - e).mean()),
                           "race_diff_lo": float(np.quantile(bs, 0.025)), "race_diff_hi": float(np.quantile(bs, 0.975))}
            res[m][col] = {"thresholds": [float(lo_t), float(hi_t)], **out}
            _log(f"{col}: 上位 レース差 {out['上位10%']['race_diff']:+.4f} [{out['上位10%']['race_diff_lo']:+.4f},{out['上位10%']['race_diff_hi']:+.4f}] / 下位 {out['下位10%']['race_diff']:+.4f} [{out['下位10%']['race_diff_lo']:+.4f},{out['下位10%']['race_diff_hi']:+.4f}]")
    _jsave(OUT / "flagged.json", res)


def cmd_boost(a) -> None:
    """V4 の OOF（2022–24）の logit を初期値（init_score）にして、新しい特徴量だけで補正する小さな LightGBM を学習し、検証年の LogLoss の差を見る。
    V4 全体の再学習（stage1/2）より軽い近似。学習年の末尾 20%（日付順）を早期終了に使う。"""
    import lightgbm as lgb
    from src.pipeline.models import win_models as wm

    ens = pd.read_parquet(V / "tp27/ens/v4_e2.parquet")
    ens["race_id"], ens["horse_id"] = ens["race_id"].astype(str), ens["horse_id"].astype(str)
    runs = fs.load_runs(TABLES, list(range(2020, 2026)))[KEY + ["finish_position", "date"]]
    m = ens.merge(runs, on=KEY, how="inner").merge(pd.read_parquet(OUT / "nprior.parquet")[KEY + ["n_prior"]].astype({"race_id": str, "horse_id": str}), on=KEY, how="left")
    m = m.sort_values(["date", "race_id"], kind="stable").reset_index(drop=True)
    cols_by_group = {}
    mats = {}
    for g in GROUP_NAMES:
        T = pd.read_parquet(OUT / "groups" / f"{g}.parquet")
        T["race_id"], T["horse_id"] = T["race_id"].astype(str), T["horse_id"].astype(str)
        mm = m[KEY].merge(T.drop_duplicates(KEY), on=KEY, how="left")
        mats[g] = mm.drop(columns=KEY).reset_index(drop=True)
        cols_by_group[g] = list(mats[g].columns)
    sets = {g: [g] for g in GROUP_NAMES}
    sets["ALL_FREE"] = list(tg.MEGU_FREE_GROUPS)
    sets["ALL_MEGU"] = list(tg.MEGU_GROUPS)
    sets["ALL"] = list(GROUP_NAMES)
    only = [x for x in os.environ.get("TP39_BOOST_SETS", "").split(",") if x]
    year = m["year"].to_numpy()
    fin = m["finish_position"].to_numpy()
    npr = m["n_prior"].to_numpy(float)
    rid = m["race_id"].to_numpy()
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    splits = {"fit22-23→test24": ((2022, 2023), 2024), "fit22→test23": ((2022,), 2023)}
    params = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 200, "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1,
              "lambda_l2": 10.0, "verbosity": -1, "num_threads": int(os.environ.get("TP39_BOOST_THREADS", 2)), "seed": 0, "deterministic": True, "force_row_wise": True}
    res: dict = {"megu": _jload(OUT / "consts.json").get("megu"), "note": "V4 の OOF(2022-24) の logit を init_score にして新しい特徴量だけで補正（LightGBM、2 threads）。Δ = 検証年の二値 LogLoss の差（負 = 改善）。95% 上限はレース単位のブートストラップ。",
                 "results": {}}
    for name, groups in sets.items():
        if only and name not in only:
            continue
        X = pd.concat([mats[g] for g in groups], axis=1)
        X = X.loc[:, ~X.columns.duplicated()]
        for split, (fit_y, test_y) in splits.items():
            for variant in ("all", "sub"):                                # sub = 前走までの出走数が 2 以下の馬だけで学習・評価（履歴の浅い馬専用の補正）
                tr = np.flatnonzero(np.isin(year, fit_y))
                te = np.flatnonzero(year == test_y)
                if variant == "sub":
                    tr, te = tr[npr[tr] <= 2], te[npr[te] <= 2]
                cut = int(len(tr) * 0.8)
                tri, vai = tr[:cut], tr[cut:]
                out = {"cols": int(X.shape[1])}
                for tname, k, base_col in (("win", 1, "p_win"), ("top3", 3, "p_top3")):
                    y = (fin <= k).astype(float)
                    init = lg(m[base_col].to_numpy(float))
                    dtr = lgb.Dataset(X.iloc[tri], y[tri], init_score=init[tri])
                    dva = lgb.Dataset(X.iloc[vai], y[vai], init_score=init[vai], reference=dtr)
                    bst = lgb.train(params, dtr, num_boost_round=400, valid_sets=[dva], callbacks=[lgb.early_stopping(40, verbose=False)])
                    raw = bst.predict(X.iloc[te], raw_score=True, num_iteration=bst.best_iteration or None) + init[te]
                    pn = 1.0 / (1.0 + np.exp(-raw))
                    pb = m[base_col].to_numpy(float)[te]
                    row = {"best_iter": int(bst.best_iteration or 0)}
                    for sub_name, smask in (("全体", np.ones(len(te), bool)), ("履歴0〜2走", npr[te] <= 2)):
                        if smask.sum() < 500:
                            continue
                        la, lb = wm.per_race_loss(rid[te][smask], pb[smask], y[te][smask]), wm.per_race_loss(rid[te][smask], pn[smask], y[te][smask])
                        bb = wm.paired_bootstrap_diff(la, lb, n_boot=500)
                        row[sub_name] = {"rows": int(smask.sum()), "d": bb["diff"], "lo": bb["lo"], "hi": bb["hi"], "base": wm.binary_logloss(pb[smask], y[te][smask])}
                    out[tname] = row
                key = f"{name}|{split}" + ("|sub" if variant == "sub" else "")
                res["results"][key] = out
                w, t3 = out["win"]["全体"], out["top3"]["全体"]
                _log(f"{key}: cols={out['cols']} rows={w['rows']} Δwin {w['d']:+.5f} (hi {w['hi']:+.5f}) Δtop3 {t3['d']:+.5f} (hi {t3['hi']:+.5f}) it={out['win']['best_iter']}/{out['top3']['best_iter']}")
                _jsave(OUT / "boost.json", res)


def _race_nll(codes: np.ndarray, starts: np.ndarray, a: np.ndarray, win: np.ndarray) -> np.ndarray:
    """レースごとの負の対数尤度（勝ち馬の softmax 確率）。codes はレース順に並んだ行のレース番号、starts は各レースの先頭行。"""
    mx = np.maximum.reduceat(a, starts)
    lse = np.log(np.add.reduceat(np.exp(a - mx[codes]), starts)) + mx
    num = np.add.reduceat(np.where(win, a, 0.0), starts)
    return lse - num


def cmd_stack(a) -> None:
    from scipy.optimize import minimize_scalar

    ens = pd.read_parquet(V / "tp27/ens/v4_e2.parquet")
    ens["race_id"], ens["horse_id"] = ens["race_id"].astype(str), ens["horse_id"].astype(str)
    runs = fs.load_runs(TABLES, list(range(2020, 2026)))[KEY + ["finish_position"]]
    m = ens.merge(runs, on=KEY, how="inner")
    npr = pd.read_parquet(OUT / "nprior.parquet")[KEY + ["n_prior"]]
    npr["race_id"], npr["horse_id"] = npr["race_id"].astype(str), npr["horse_id"].astype(str)
    m = m.merge(npr, on=KEY, how="left")
    feats = {}
    for g in GROUP_NAMES:
        T = pd.read_parquet(OUT / "groups" / f"{g}.parquet")
        T["race_id"], T["horse_id"] = T["race_id"].astype(str), T["horse_id"].astype(str)
        T = T.drop_duplicates(KEY)
        mm = m[KEY].merge(T, on=KEY, how="left")
        for c in [c for c in mm.columns if c not in KEY]:
            if c not in feats:
                feats[c] = (g, mm[c].to_numpy(float))
    m = m.sort_values(["race_id", "horse_id"], kind="stable").reset_index()                  # 行の順は feats と合わせるため元の index を残す
    order = m["index"].to_numpy()
    codes_all = pd.factorize(m["race_id"])[0]
    year = m["year"].to_numpy()
    win = (m["finish_position"].to_numpy() == 1)
    ok_race = pd.Series(win).groupby(codes_all).transform("sum").to_numpy() == 1               # 勝ち馬が 1 頭のレースだけ
    s = np.log(np.clip(m["p_win"].to_numpy(float), 1e-9, 1))
    nprior = m["n_prior"].to_numpy(float)

    def sub(mask):
        idx = np.flatnonzero(mask & ok_race)
        has_win = pd.Series(win[idx]).groupby(codes_all[idx]).transform("sum").to_numpy() == 1        # 部分集合の中に勝ち馬がいるレースだけ（部分集合内の勝ち馬を当てる問題にする）
        multi = pd.Series(1, index=idx).groupby(codes_all[idx]).transform("size").to_numpy() >= 2
        idx = idx[has_win & multi]
        c = pd.factorize(codes_all[idx])[0]
        st = np.r_[0, np.flatnonzero(np.diff(c)) + 1]
        return idx, c, st

    splits = {"fit22-23→test24": ((2022, 2023), (2024,)), "fit22→test23": ((2022,), (2023,))}
    res: dict = {"note": "V4 の log p_win に γ×(標準化した特徴量) を足した softmax（条件付きロジット）。γ は学習年で最尤、検証年のレース単位の負の対数尤度の差（負 = 改善）。履歴 0〜2 走は n_prior≤2 の行だけで当てて評価。",
                 "n_features": len(feats), "results": {}}
    rng = np.random.default_rng(0)
    for split, (fit_y, test_y) in splits.items():
        for sname, smask in (("全体", np.ones(len(m), bool)), ("履歴0〜2走", nprior <= 2)):
            fi, fc, fst = sub(np.isin(year, fit_y) & smask)
            ti, tc, tst = sub(np.isin(year, test_y) & smask)
            if len(fi) < 5000 or len(ti) < 3000:
                continue
            base_t = _race_nll(tc, tst, s[ti], win[ti])
            rows = []
            for name, (g, x_all) in feats.items():
                x = x_all[order]
                xf = x[fi]
                if np.isfinite(xf).mean() < 0.05 or np.nanstd(xf) < 1e-9:
                    continue
                mu, sd = np.nanmean(xf), np.nanstd(xf)
                z = np.clip(np.nan_to_num((x - mu) / sd, nan=0.0), -4, 4)
                f = lambda gam: float(_race_nll(fc, fst, s[fi] + gam * z[fi], win[fi]).mean())
                r = minimize_scalar(f, bounds=(-1.5, 1.5), method="bounded")
                gam = float(r.x)
                dt = _race_nll(tc, tst, s[ti] + gam * z[ti], win[ti]) - base_t
                bs = np.array([dt[rng.integers(0, len(dt), len(dt))].mean() for _ in range(200)])
                rows.append({"feature": name, "group": g, "gamma": gam, "d_nll": float(dt.mean()), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)),
                             "t": float(dt.mean() / (dt.std() / np.sqrt(len(dt)) + 1e-12)), "fit_gain": float(f(0.0) - r.fun)})
            rows.sort(key=lambda r: r["t"])
            res["results"][f"{split}|{sname}"] = {"races_fit": int(len(fst)), "races_test": int(len(tst)), "tested": len(rows), "base_nll": float(base_t.mean()),
                                                  "top": rows[:15], "n_improved_t<-2": int(sum(r["t"] < -2 for r in rows)), "n_worse_t>2": int(sum(r["t"] > 2 for r in rows))}
            top = rows[:3]
            _log(f"{split}|{sname}: tested={len(rows)} t<-2: {sum(r['t'] < -2 for r in rows)} ベスト: " + ", ".join(f"{r['feature']} Δ{r['d_nll']:+.5f} t={r['t']:.1f}" for r in top))
    _jsave(OUT / "stack.json", res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["metrics", "describe", "groups", "prior", "stage1", "stage2", "subset", "flagged", "stack", "boost"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"metrics": cmd_metrics, "describe": cmd_describe, "groups": cmd_groups, "prior": cmd_prior, "stage1": cmd_stage1, "stage2": cmd_stage2,
     "subset": cmd_subset, "flagged": cmd_flagged, "stack": cmd_stack, "boost": cmd_boost}[a.cmd](a)


if __name__ == "__main__":
    main()
