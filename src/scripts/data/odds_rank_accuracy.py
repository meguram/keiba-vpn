"""想定オッズ（§77・sec77）で作った人気順が、実際の人気順とどれだけ合うかを測る（調査だけ。モデル・本番は変えない）。

    ODDS_RANK_DATA_ROOT=<本体のリポジトリ> nice -n 10 python -m src.scripts.data.odds_rank_accuracy   # → data/local/research/odds_rank/result.json

材料（すべて手元。GCS は読まない・費用 0 円）:
- 想定オッズ: data/local/modeling/odds_pred/wp_oof_pred.parquet の share_raw（年次 walk-forward の OOF。各年より前の年だけで学習）。
  本番の版（serving/meta.json）は 2020〜2025 年で学習しているので、2025 年の本番の版の出力は学習の期間の中になる。
  そのため、本番と同じ作り方で 2025 年を見ていない版（2025 年の fold）を主にする。
- 実際の人気: 同じ表の popularity / win_odds（race_odds 由来の確定）。
- 基準: 旧来の想定オッズ（同じ表の win_V8。各年より前だけで学習）・AI 勝率（同じ表の p_win。v6 の OOF。2022〜2024 年だけ）・
  前走の人気（race_result_flat の popularity）・無作為（式で出す）。2026 年の AI 勝率は predictions_v6（本番の当日）→ predictions_backfill（後付け）。
- 本番の版の 2026 年の対数誤差（日ごと）: data/local/meta/odds_sec77_compare.json（odds_sec77_backfill の出力）。
- 発走前（T-60 など）: data/local/research/persona1/pre_odds.parquet（スナップショットは 2026-10-18 から）。
2026 年は人気・オッズの列とレースの条件の列だけを読む（着順・払戻の列は読まない）。
"""

from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("ODDS_RANK_DATA_ROOT") or ROOT)       # worktree から動かすときは本体のリポジトリを指す
OOF = DATA / "data/local/modeling/odds_pred/wp_oof_pred.parquet"
TABLES = DATA / "data/page_reference/tables"
PRE = DATA / "data/local/research/persona1/pre_odds.parquet"
INF = DATA / "data/local/inference"
CMP77 = DATA / "data/local/meta/odds_sec77_compare.json"
DRIFT = DATA / "data/local/meta/odds_drift/latest_t60.json"
OUT = Path(os.environ.get("ODDS_RANK_OUT") or DATA / "data/local/research/odds_rank")
B = int(os.environ.get("ODDS_RANK_BOOT", "1000"))
SEED = 20261118
RACE_COLS = ["race_id", "date", "surface", "grade", "race_class", "field_size", "start_time", "race_name"]
GAP_BINS, GAP_LABELS = [0, 1.3, 2, 3, 1e9], ["1.3 倍未満", "1.3〜2 倍", "2〜3 倍", "3 倍以上"]


def _log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ───────────── 材料
def race_meta(years) -> pd.DataFrame:
    fs = []
    for y in years:
        f = TABLES / str(y) / "race_result_race.parquet"           # レースの条件の列だけ読む（着順・払戻の列は読まない）
        fs.append(pd.read_parquet(f, columns=RACE_COLS).drop_duplicates("race_id"))
    r = pd.concat(fs, ignore_index=True)
    r["race_id"] = r["race_id"].astype(str)
    return r


def classes(r: pd.DataFrame) -> pd.DataFrame:
    g = r["grade"].fillna("").astype(str)
    cls = np.select([g.isin(["G1", "G2", "G3"]), g.isin(["OP", "L"]), g == "3勝", g == "2勝", g == "1勝", g == "未勝利", g == "新馬"],
                    ["重賞", "OP・L", "3勝", "2勝", "1勝", "未勝利", "新馬"], "その他")
    r = r.assign(cls=cls)
    r["shinba"] = np.where(r["cls"] == "新馬", "新馬", "新馬以外")
    r["jusho"] = np.where(r["cls"] == "重賞", "重賞", "重賞以外")
    fsz = pd.to_numeric(r["field_size"], errors="coerce")
    r["fs_band"] = pd.cut(fsz, [0, 10, 14, 18], labels=["〜10 頭", "11〜14 頭", "15〜18 頭"]).astype(str)
    h = pd.to_numeric(r["start_time"].astype(str).str[:2], errors="coerce")
    r["time_band"] = np.select([h < 12, h < 14], ["〜11 時台", "12〜13 時台"], "14 時台〜")
    r["month"] = r["date"].astype(str).str[:7]
    r["year"] = r["race_id"].str[:4].astype(int)
    r["surf"] = r["surface"].fillna("").replace({"障": "障害"})
    return r


def prev_popularity(years) -> pd.DataFrame:
    """馬ごとの 1 つ前の出走の人気（race_result_flat。人気・日付の列だけ読む）。取消・除外（人気が無い行）は飛ばして、その前の出走をとる。"""
    fs = []
    for y in years:
        f = TABLES / str(y) / "race_result_flat.parquet"
        if f.exists():
            fs.append(pd.read_parquet(f, columns=["race_id", "horse_id", "date", "popularity"]))
    d = pd.concat(fs, ignore_index=True)
    d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    d["popularity"] = pd.to_numeric(d["popularity"], errors="coerce").astype("float64")
    d = d[d["popularity"] > 0].drop_duplicates(["race_id", "horse_id"]).sort_values(["horse_id", "date", "race_id"])
    d["prev_pop"] = d.groupby("horse_id")["popularity"].shift(1)
    return d[["race_id", "horse_id", "prev_pop"]]


# ───────────── レースごとの指標
BANDS = (("err_1_3", 1, 3), ("err_4_6", 4, 6), ("err_7p", 7, 99))


def per_race(df: pd.DataFrame, score: str, actual_rank: str = "act_rank", higher_better: bool = True, rng=None) -> pd.DataFrame:
    """score（大きいほど人気と予測）の順位と、実際の順位（1=1 番人気）を比べる。欠けた score は最下位（同順位は無作為）。
    u_err_*: 同じ馬を無作為の順に置いたときの |順位差| の期待値（人気帯の誤差を、端ほど片側にしかずれない分を除いて比べるため）。"""
    rng = rng or np.random.default_rng(SEED)
    d = df[["race_id", actual_rank, score]].copy()
    s = pd.to_numeric(d[score], errors="coerce").astype("float64").to_numpy()
    s = s if higher_better else -s
    s = np.where(np.isnan(s), -1e9 + rng.random(len(s)), s)
    d["_s"] = s
    d["pred_rank"] = d.groupby("race_id")["_s"].rank(ascending=False, method="first")
    d["act"] = d[actual_rank]
    out = []
    for rid, g in d.groupby("race_id", sort=False):
        n = len(g)
        if n < 4:
            continue
        a, p = g["act"].to_numpy(dtype=float), g["pred_rank"].to_numpy(dtype=float)
        rho = stats.spearmanr(a, p).statistic
        tau = stats.kendalltau(a, p).statistic          # tau-b（同順位あり）
        top1 = float(a[np.argmin(p)] == a.min())
        ov3 = float(len(set(np.where(p <= 3)[0]) & set(np.where(a <= 3)[0])))
        err = np.abs(a - p)
        uerr = ((a - 1) * a / 2 + (n - a) * (n - a + 1) / 2) / n
        row = {"race_id": rid, "n": n, "spearman": rho, "kendall": tau, "top1": top1, "top3": ov3, "mae_rank": err.mean()}
        for lab, lo, hi in BANDS:
            mk = (a >= lo) & (a <= hi)
            row[lab] = err[mk].mean() if mk.any() else np.nan
            row["u_" + lab] = uerr[mk].mean() if mk.any() else np.nan
        out.append(row)
    return pd.DataFrame(out)


def uniform(n: np.ndarray) -> dict:
    n = n.astype(float)
    return {"spearman": 0.0, "kendall": 0.0, "top1": float(np.mean(1 / n)), "top3": float(np.mean(9 / n)), "mae_rank": float(np.mean((n * n - 1) / (3 * n)))}


METRICS = ["spearman", "kendall", "top1", "top3", "mae_rank", "err_1_3", "err_4_6", "err_7p", "u_err_1_3", "u_err_4_6", "u_err_7p"]


def boot_mean(x: np.ndarray, rng, b: int = B) -> tuple[float, float, float]:
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    idx = rng.integers(0, len(x), size=(b, len(x)))
    mm = x[idx].mean(axis=1)
    return (float(x.mean()), float(np.quantile(mm, 0.025)), float(np.quantile(mm, 0.975)))


def boot_cluster(x: np.ndarray, grp: np.ndarray, rng, b: int = B) -> tuple[float, float, float]:
    """群（日）で束ねたブートストラップ（レースの平均）。"""
    ok = ~np.isnan(x)
    x, grp = x[ok], grp[ok]
    keys = np.unique(grp)
    sums = np.array([x[grp == k].sum() for k in keys])
    cnts = np.array([(grp == k).sum() for k in keys])
    idx = rng.integers(0, len(keys), size=(b, len(keys)))
    mm = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    return (float(x.mean()), float(np.quantile(mm, 0.025)), float(np.quantile(mm, 0.975)))


def summarize(pr: pd.DataFrame, rng) -> dict:
    o = {"races": int(len(pr)), "horses": int(pr["n"].sum())}
    for k in METRICS:
        if k in pr.columns:
            o[k] = boot_mean(pr[k].to_numpy(dtype=float), rng)
    return o


def diff(pa: pd.DataFrame, pb: pd.DataFrame, rng, metrics=("spearman", "top1", "top3", "mae_rank")) -> dict:
    """レースで束ねた対のブートストラップ（同じレースの差）。"""
    mm = pa.merge(pb, on="race_id", suffixes=("_a", "_b"))
    return {k: boot_mean((mm[f"{k}_a"] - mm[f"{k}_b"]).to_numpy(dtype=float), rng) for k in metrics}


def diff_groups(a: pd.DataFrame, b: pd.DataFrame, k: str, rng) -> tuple[float, float, float]:
    """別々のレースの群の平均の差（それぞれをレースで引き直す）。"""
    x, y = a[k].dropna().to_numpy(dtype=float), b[k].dropna().to_numpy(dtype=float)
    mx = x[rng.integers(0, len(x), size=(B, len(x)))].mean(axis=1)
    my = y[rng.integers(0, len(y), size=(B, len(y)))].mean(axis=1)
    d = mx - my
    return (float(x.mean() - y.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)))


def diff_uniform(pa: pd.DataFrame, rng, metrics=("spearman", "top1", "top3", "mae_rank")) -> dict:
    n = pa["n"].to_numpy(dtype=float)
    u = {"spearman": 0 * n, "top1": 1 / n, "top3": 9 / n, "mae_rank": (n * n - 1) / (3 * n)}
    return {k: boot_mean(pa[k].to_numpy(dtype=float) - u[k], rng) for k in metrics}


def gap_of(d: pd.DataFrame, odds_col: str) -> pd.Series:
    """レースごとの『2 番目に低いオッズ ÷ 最も低いオッズ』。"""
    w = d[["race_id", odds_col]].sort_values(["race_id", odds_col])
    return w.groupby("race_id")[odds_col].apply(lambda x: x.iloc[1] / x.iloc[0] if len(x) > 1 else np.nan)


def by_gap(pr: pd.DataFrame, gap: pd.Series, rng) -> dict:
    sr = pr.set_index("race_id").join(gap.rename("gap"))
    sr["gap_band"] = pd.cut(sr["gap"], GAP_BINS, labels=GAP_LABELS, right=False).astype(str)   # 1.3 倍ちょうどは「1.3〜2 倍」へ
    return {str(v): summarize(g.reset_index(), rng) for v, g in sr.groupby("gap_band") if v in GAP_LABELS}


# ───────────── 部 1: 想定オッズ vs 確定の人気（2022〜2025）
def part_final(rng) -> dict:
    o = pd.read_parquet(OOF, columns=["race_id", "horse_id", "year", "share_raw", "win_B", "win_V8", "win_odds", "popularity", "p_win"])
    o["race_id"], o["horse_id"] = o["race_id"].astype(str), o["horse_id"].astype(str)
    o = o[o["popularity"] > 0]
    o["act_rank"] = o.groupby("race_id")["popularity"].rank(method="average")
    meta = classes(race_meta(range(2022, 2026)))
    pp = prev_popularity(range(2019, 2026))
    o = o.merge(pp, on=["race_id", "horse_id"], how="left")
    _log(f"部 1: {len(o)} 頭・{o['race_id'].nunique()} レース。前走の人気がある割合 {o['prev_pop'].notna().mean():.3f}・V8 がある割合 {o['win_V8'].notna().mean():.3f}")

    P = {"sec77": per_race(o, "share_raw"), "prev_pop": per_race(o, "prev_pop", higher_better=False, rng=np.random.default_rng(SEED + 1)),
         "v8": per_race(o, "win_V8", higher_better=False, rng=np.random.default_rng(SEED + 2))}
    P["p_win"] = per_race(o[o["year"] <= 2024], "p_win")
    for k in list(P):
        P[k] = P[k].merge(meta, on="race_id", how="left")
    P["sec77_2224"] = P["sec77"][P["sec77"]["year"] <= 2024]

    res: dict = {"overall": {}, "by": {}, "vs": {}, "calib": {}, "coverage": {}}
    s = P["sec77"]
    for y in (2022, 2023, 2024, 2025):
        for k in ("sec77", "prev_pop", "v8"):
            res["overall"][f"{k}_{y}"] = summarize(P[k][P[k]["year"] == y], rng)
        res["overall"][f"uniform_{y}"] = uniform(s.loc[s["year"] == y, "n"].to_numpy())
        if y <= 2024:
            res["overall"][f"p_win_{y}"] = summarize(P["p_win"][P["p_win"]["year"] == y], rng)
    res["overall"]["sec77_2224"] = summarize(P["sec77_2224"], rng)
    res["overall"]["p_win_2224"] = summarize(P["p_win"], rng)
    s25 = s[s["year"] == 2025]
    yr = lambda k, y: P[k][P[k]["year"] == y]  # noqa: E731
    res["vs"]["2025_vs_v8"] = diff(s25, yr("v8", 2025), rng)
    res["vs"]["2025_vs_prev_pop"] = diff(s25, yr("prev_pop", 2025), rng)
    res["vs"]["2025_vs_uniform"] = diff_uniform(s25, rng)
    res["vs"]["2224_vs_p_win"] = diff(P["sec77_2224"], P["p_win"], rng)
    res["vs"]["2224_vs_v8"] = diff(P["sec77_2224"], P["v8"][P["v8"]["year"] <= 2024], rng)
    res["vs"]["2224_vs_prev_pop"] = diff(P["sec77_2224"], P["prev_pop"][P["prev_pop"]["year"] <= 2024], rng)
    # 分け方（2025 年）
    pv25, v825 = yr("prev_pop", 2025), yr("v8", 2025)
    for col in ("month", "surf", "cls", "fs_band", "shinba", "jusho"):
        res["by"][col] = {}
        for v, g in s25.groupby(col):
            res["by"][col][str(v)] = {"sec77": summarize(g, rng), "prev_pop": summarize(pv25[pv25[col] == v], rng), "uniform": uniform(g["n"].to_numpy()),
                                      "v8": summarize(v825[v825[col] == v], rng), "vs_v8": diff(g, v825[v825[col] == v], rng, ("spearman", "top1"))}
    # クラスの差（そのクラス − それ以外。別々のレースの群）
    res["cls_vs_rest"] = {str(v): {k: diff_groups(s25[s25["cls"] == v], s25[s25["cls"] != v], k, rng) for k in ("spearman", "top1")}
                          for v in s25["cls"].unique()}
    # 2022〜24 の AI 勝率との比較（クラス・頭数）
    res["by_pwin"] = {}
    for col in ("cls", "shinba", "fs_band"):
        res["by_pwin"][col] = {}
        a, b = P["sec77_2224"], P["p_win"]
        for v, g in a.groupby(col):
            gb = b[b[col] == v]
            res["by_pwin"][col][str(v)] = {"sec77": summarize(g, rng), "p_win": summarize(gb, rng), "vs_p_win": diff(g, gb, rng, ("spearman", "top1"))}
    o = o.merge(meta[["race_id", "cls"]], on="race_id", how="left")
    res["coverage"] = {k: float(v) for k, v in o[o["year"] == 2025].groupby("cls")["prev_pop"].apply(lambda x: x.notna().mean()).items()}
    # 対数オッズの誤差と較正（年ごと。win_B は各年より前の年の OOF で当てはめた変換。2022 年は変換なし）
    res["calib"]["by_year"] = {}
    for y in (2022, 2023, 2024, 2025):
        d = o[(o["year"] == y) & (o["win_odds"] > 0)]
        e = np.log(d["win_B"]) - np.log(d["win_odds"])
        res["calib"]["by_year"][str(y)] = {"mae_log": float(e.abs().mean()), "rmse_log": float(np.sqrt((e ** 2).mean())), "bias_log": float(e.mean()), "n": int(len(e))}
    d = o[(o["year"] == 2025) & (o["win_odds"] > 0)].copy()
    d["lp"], d["la"] = np.log(d["win_B"]), np.log(d["win_odds"])
    e = d["lp"] - d["la"]
    res["calib"]["mae_log"] = float(e.abs().mean())
    res["calib"]["bias_log"] = float(e.mean())
    res["calib"]["slope"] = float(np.polyfit(d["lp"], d["la"], 1)[0])
    d["bin"] = pd.qcut(d["lp"], 10, labels=False, duplicates="drop")
    res["calib"]["bins"] = [{"pred": float(np.exp(g["lp"].mean())), "act": float(np.exp(g["la"].mean())), "n": int(len(g)),
                             "mae": float((g["lp"] - g["la"]).abs().mean())} for _, g in d.groupby("bin")]
    d["band"] = pd.cut(d["popularity"], [0, 3, 6, 99], labels=["1〜3", "4〜6", "7〜"]).astype(str)
    res["calib"]["by_band"] = {b_: {"mae_log": float((g["lp"] - g["la"]).abs().mean()), "bias_log": float((g["lp"] - g["la"]).mean()), "n": int(len(g))}
                               for b_, g in d.groupby("band")}
    # 本番の版を 2026 年に後から当てた対数誤差（日ごとの集計を頭数で重み付けしてまとめる）
    if CMP77.exists():
        c = json.loads(CMP77.read_text(encoding="utf-8"))
        rows = [v["sec77"] for v in c.values() if isinstance(v, dict) and v.get("sec77")]
        n = np.array([r_["n"] for r_ in rows], dtype=float)
        rm = np.array([r_["rmse"] for r_ in rows]) ** 2
        bi = np.array([r_["bias"] for r_ in rows])
        res["calib"]["prod_2026"] = {"days": len(rows), "n": int(n.sum()), "rmse_log": float(np.sqrt((rm * n).sum() / n.sum())), "bias_log": float((bi * n).sum() / n.sum()),
                                     "first": min(k for k in c if c[k].get("sec77")), "last": max(k for k in c if c[k].get("sec77"))}
    # 1 番人気の的中と、1・2 番人気の差（確定のオッズの差＝事後の診断／想定オッズの差＝前もって分かる）
    w = o[o["year"] == 2025]
    res["by"]["gap_band"] = {k: {"sec77": v} for k, v in by_gap(s25, gap_of(w, "win_odds"), rng).items()}
    res["by"]["gap_pred"] = {k: {"sec77": v} for k, v in by_gap(s25, gap_of(w, "win_B"), rng).items()}
    s_rank = w.copy()
    s_rank["pred_rank"] = s_rank.groupby("race_id")["share_raw"].rank(ascending=False, method="first")
    res["confusion_final"] = _transition(s_rank["act_rank"].round().clip(upper=10), s_rank["pred_rank"].clip(upper=10))
    return res


def _transition(a: pd.Series, p: pd.Series) -> dict:
    """a（行）→ p（列）の遷移の割合（行で 1）。"""
    t = pd.crosstab(a.astype(int), p.astype(int), normalize="index")
    return {"rows": [int(x) for x in t.index], "cols": [int(x) for x in t.columns], "p": t.round(4).values.tolist()}


# ───────────── 部 2: 発走前（T-60 など）→ 確定（2026-10-18 以降）
def part_pre(rng) -> dict:
    pre = pd.read_parquet(PRE, columns=["race_id", "phase", "snapshot_at", "horse_number", "win_odds", "popularity"])   # 人気・オッズの列だけ
    pre["race_id"] = pre["race_id"].astype(str)
    pre = pre[(pre["popularity"] > 0) & (pre["win_odds"] > 0)]
    fin = pd.read_parquet(TABLES / "2026/race_result_flat.parquet", columns=["race_id", "horse_number", "horse_id", "date", "popularity", "odds"])
    fin["race_id"] = fin["race_id"].astype(str)
    fin = fin[(fin["popularity"] > 0) & (fin["odds"] > 0)].rename(columns={"popularity": "pop_f", "odds": "odds_f"})
    meta = classes(race_meta([2026]))
    pp = prev_popularity(range(2024, 2027))
    pw, pw_src = _pwin_2026()
    res: dict = {"phases": {}, "phases_common": {}, "phase_minutes": {}, "by": {}, "transition": {}, "drift": {}, "filters": {}, "days": {}, "base": {}, "noise": {}}
    phases = [p for p in ("t60", "t15", "t10", "t05", "t02") if p in set(pre["phase"])]
    # 時点の実際の時刻（発走の何分前に取ったか）
    st = meta.set_index("race_id")
    pre["start"] = pd.to_datetime(pre["race_id"].map(st["date"]).astype(str) + " " + pre["race_id"].map(st["start_time"]).astype(str), errors="coerce")
    snap = pd.to_datetime(pre["snapshot_at"], errors="coerce", utc=True).dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    pre["min_before"] = (pre["start"] - snap).dt.total_seconds() / 60
    for ph in phases:
        mb = pre.loc[pre["phase"] == ph].drop_duplicates("race_id")["min_before"]
        res["phase_minutes"][ph] = {"median": float(mb.median()), "p10": float(mb.quantile(0.1)), "p90": float(mb.quantile(0.9))}
    M = {}
    PR = {}
    for ph in phases:
        e = pre[pre["phase"] == ph].drop_duplicates(["race_id", "horse_number"], keep="last")
        mm = e.merge(fin, on=["race_id", "horse_number"], how="inner")
        mm["act_rank"] = mm.groupby("race_id")["pop_f"].rank(method="average")   # 両方で揃う馬だけで順位を付け直す（取消の馬を外す）
        mm["neg_pop"] = -mm.groupby("race_id")["popularity"].rank(method="average")
        pr = per_race(mm, "neg_pop").merge(meta, on="race_id", how="left")
        res["phases"][ph] = summarize(pr, rng)
        res["phases"][ph]["days"] = int(mm["date"].nunique())
        M[ph], PR[ph] = mm, pr
    sets = [set(PR[p]["race_id"]) for p in ("t60", "t15", "t05", "t02") if p in PR]
    common = set.intersection(*sets) if sets else set()
    for ph in ("t60", "t15", "t05", "t02"):
        if ph in PR:
            res["phases_common"][ph] = summarize(PR[ph][PR[ph]["race_id"].isin(common)], rng)
    if "t60" not in PR:
        return res
    P60, M60 = PR["t60"], M["t60"]
    day = P60["date"].astype(str).to_numpy()
    res["phases"]["t60"]["day_ci"] = {k: boot_cluster(P60[k].to_numpy(dtype=float), day, rng) for k in ("spearman", "top1", "top3")}
    for col in ("fs_band", "cls", "surf", "shinba", "time_band"):
        res["by"][col] = {str(v): summarize(g, rng) for v, g in P60.groupby(col)}
    res["days"] = {str(dd): {"races": int(len(g)), "spearman": float(g["spearman"].mean()), "top1": float(g["top1"].mean())} for dd, g in P60.groupby("date")}
    # 遷移の表（T-60 の人気 → 確定の人気。10 以降は 10 にまとめる）
    M60["r60"] = M60.groupby("race_id")["popularity"].rank(method="first")
    M60["rf"] = M60.groupby("race_id")["pop_f"].rank(method="first")
    res["transition"] = _transition(M60["r60"].clip(upper=10), M60["rf"].clip(upper=10))
    # 対数オッズの変化 log(確定/T-60)：人気帯・オッズ帯ごと（負 = 確定で売れた）
    M60["dl"] = np.log(M60["odds_f"] / M60["win_odds"])
    M60["band"] = pd.cut(M60["r60"], [0, 1, 3, 6, 99], labels=["1 番人気", "2〜3", "4〜6", "7〜"]).astype(str)
    M60["oband"] = pd.cut(M60["win_odds"], [0, 3, 6, 10, 20, 50, 1e9], labels=["<3", "3-6", "6-10", "10-20", "20-50", "50+"]).astype(str)
    for key in ("band", "oband"):
        res["drift"][key] = {}
        for v, g in M60.groupby(key):
            rb = _race_boot(g, "dl", rng)
            res["drift"][key][str(v)] = {"n": int(len(g)), "mean": rb[0], "lo": rb[1], "hi": rb[2], "median": float(g["dl"].median()),
                                         "share_down20": float((g["dl"] < np.log(0.8)).mean())}
    if DRIFT.exists():
        res["drift"]["existing"] = {b_["band"]: {"n": b_["n"], "mean": b_["mean"]} for b_ in json.loads(DRIFT.read_text(encoding="utf-8")).get("bands", [])}
    # 本命の市場の確率（1/オッズを足して 1 にした値）の変化
    for c, src in (("q60", "win_odds"), ("qf", "odds_f")):
        inv = 1 / M60[src]
        M60[c] = inv / inv.groupby(M60["race_id"]).transform("sum")
    fav = M60[M60["r60"] == 1].assign(dq=lambda x: x["qf"] - x["q60"])
    res["drift"]["fav_share"] = {"t60": float(fav["q60"].mean()), "final": float(fav["qf"].mean()), "diff": _race_boot(fav, "dq", rng),
                                 "diff_day": boot_cluster(fav["dq"].to_numpy(dtype=float), fav["date"].astype(str).to_numpy(), rng)}
    # 雑音の手がかり: 最大オッズと Σ1/オッズ（控除）の中央値
    for nm, col in (("t60", "win_odds"), ("final", "odds_f")):
        g = M60.groupby("race_id")[col]
        res["noise"][nm] = {"max_odds_median": float(g.max().median()), "overround_median": float((1 / M60[col]).groupby(M60["race_id"]).sum().median())}
    # T-60 の人気で絞り込んだときに、確定の人気と食い違う馬の割合（区間はレース・日で束ねる）
    rid = M60["race_id"].to_numpy()
    rday = M60.groupby("race_id")["date"].first()
    for k in (1, 2, 3, 5):
        a60, af = (M60["r60"] <= k).to_numpy(), (M60["rf"] <= k).to_numpy()
        per = pd.DataFrame({"race_id": rid, "in_out": a60 & ~af, "in": a60, "flip": a60 != af}).groupby("race_id").agg(io=("in_out", "sum"), i=("in", "sum"), f=("flip", "any"))
        dd = rday.loc[per.index].astype(str).to_numpy()
        res["filters"][f"top{k}"] = {"in60_out_f": _ratio_boot(per["io"].to_numpy(float), per["i"].to_numpy(float), None, rng),
                                     "in60_out_f_day": _ratio_boot(per["io"].to_numpy(float), per["i"].to_numpy(float), dd, rng),
                                     "out60_in_f": float(((~a60) & af).sum() / max((~a60).sum(), 1)),
                                     "flip_share_all": float((a60 != af).mean()),
                                     "race_any_flip": boot_mean(per["f"].to_numpy(float), rng), "race_any_flip_day": boot_cluster(per["f"].to_numpy(float), dd, rng)}
    # 1 番人気の入れ替わりを、T-60 の 1・2 番人気の差で分ける（想定オッズの表の上限の目安）
    g60 = gap_of(M60, "win_odds")
    res["by"]["gap_t60"] = by_gap(P60, g60, rng)
    # 基準: AI 勝率・前走の人気の順は、T-60 と確定のどちらに近いか（2026・同じレース）
    M60 = M60.merge(pp, on=["race_id", "horse_id"], how="left").merge(pw, on=["race_id", "horse_id"], how="left")
    res["base"]["pwin_cover"] = float(M60["p_win"].notna().mean())
    res["base"]["pwin_source"] = {k: float(v) for k, v in M60.groupby("_src")["race_id"].nunique().items()} if "_src" in M60.columns else {}
    for name, sc, hb in (("p_win", "p_win", True), ("prev_pop", "prev_pop", False)):
        sub = M60[M60.groupby("race_id")[sc].transform(lambda x: x.notna().mean()) >= (0.99 if name == "p_win" else 0)]
        a = per_race(sub.assign(act_rank=sub["rf"]), sc, higher_better=hb)
        b = per_race(sub.assign(act_rank=sub["r60"]), sc, higher_better=hb)
        res["base"][name] = {"vs_final": summarize(a, rng), "vs_t60": summarize(b, rng), "final_minus_t60": diff(a, b, rng, ("spearman", "top1", "top3"))}
    res["base"]["pwin_files"] = pw_src
    return res


def _race_boot(g: pd.DataFrame, col: str, rng) -> tuple[float, float, float]:
    """レースで束ねたブートストラップ（馬の平均）。"""
    s = g.groupby("race_id")[col].agg(["sum", "size"])
    v, n = s["sum"].to_numpy(), s["size"].to_numpy()
    idx = rng.integers(0, len(s), size=(B, len(s)))
    mm = v[idx].sum(axis=1) / n[idx].sum(axis=1)
    return (float(v.sum() / n.sum()), float(np.quantile(mm, 0.025)), float(np.quantile(mm, 0.975)))


def _ratio_boot(num: np.ndarray, den: np.ndarray, grp, rng) -> tuple[float, float, float]:
    """Σnum/Σden の区間。grp が None ならレースで、あれば群（日）で束ねる。"""
    if grp is not None:
        keys = np.unique(grp)
        num = np.array([num[grp == k].sum() for k in keys])
        den = np.array([den[grp == k].sum() for k in keys])
    idx = rng.integers(0, len(num), size=(B, len(num)))
    mm = num[idx].sum(axis=1) / np.maximum(den[idx].sum(axis=1), 1)
    return (float(num.sum() / max(den.sum(), 1)), float(np.quantile(mm, 0.025)), float(np.quantile(mm, 0.975)))


def _pwin_2026() -> tuple[pd.DataFrame, dict]:
    fs = {}
    for root in ("predictions_backfill/bundle_v6_2025", "predictions_v6"):      # 後の（本番の当日）を優先
        for f in glob.glob(str(INF / root / "2026*" / "*.parquet")):
            fs[Path(f).stem] = (root, f)
    if not fs:
        return pd.DataFrame(columns=["race_id", "horse_id", "p_win", "_src"]), {}
    parts = []
    for root, f in fs.values():
        x = pd.read_parquet(f, columns=["race_id", "horse_id", "p_win"])
        x["_src"] = root
        parts.append(x)
    d = pd.concat(parts, ignore_index=True)
    d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    src = pd.Series([r for r, _ in fs.values()]).value_counts().to_dict()
    return d.drop_duplicates(["race_id", "horse_id"], keep="last"), {k: int(v) for k, v in src.items()}


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else round(float(x), 5)
    if isinstance(x, np.integer):
        return int(x)
    return x


# ───────────── 部 3: 本番の想定オッズ（2026）→ T-60 の人気・確定の人気（同じレース）
SEC77_DIR = Path(os.environ.get("SEC77_2026_DIR") or "/home/jovyan/work/odds_rank_out/sec77_2026")   # odds_rank_fetch_sec77 の保存先


def _sec77_2026() -> pd.DataFrame:
    import ast

    rows = []
    for f in sorted(SEC77_DIR.glob("2026*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        ents = d.get("entries") or []
        if isinstance(ents, str):
            ents = ast.literal_eval(ents)
        meta = d.get("_cache_meta") or {}
        if isinstance(meta, str):
            meta = ast.literal_eval(meta)
        for e in ents:
            if e.get("source") == "sec77" and e.get("predicted_win_odds"):
                rows.append({"race_id": str(d.get("race_id") or f.stem), "horse_number": int(e["horse_number"]), "sec77_odds": float(e["predicted_win_odds"]),
                             "computed_at": meta.get("computed_at")})
    return pd.DataFrame(rows)


def part_sec77(rng) -> dict:
    s = _sec77_2026()
    if s.empty:
        return {}
    pre = pd.read_parquet(PRE, columns=["race_id", "phase", "snapshot_at", "horse_number", "win_odds", "popularity"])
    pre["race_id"] = pre["race_id"].astype(str)
    e = pre[(pre["phase"] == "t60") & (pre["popularity"] > 0) & (pre["win_odds"] > 0)].drop_duplicates(["race_id", "horse_number"], keep="last")
    fin = pd.read_parquet(TABLES / "2026/race_result_flat.parquet", columns=["race_id", "horse_number", "popularity", "odds"])   # 人気・オッズだけ
    fin["race_id"] = fin["race_id"].astype(str)
    fin = fin[(fin["popularity"] > 0) & (fin["odds"] > 0)].rename(columns={"popularity": "pop_f", "odds": "odds_f"})
    meta = classes(race_meta([2026]))
    m = s.merge(e, on=["race_id", "horse_number"], how="inner").merge(fin, on=["race_id", "horse_number"], how="inner")
    # 3 つがそろう馬だけで順位を付け直す
    m["r60"] = m.groupby("race_id")["popularity"].rank(method="average")
    m["rf"] = m.groupby("race_id")["pop_f"].rank(method="average")
    m["neg_s"] = -m["sec77_odds"]
    res: dict = {"files": len(list(SEC77_DIR.glob("2026*.json"))), "races": int(m["race_id"].nunique()), "horses": int(len(m))}
    # 想定オッズを作った時刻（発走の何分前・T-60 の写しより前か）
    st = meta.set_index("race_id")
    start = pd.to_datetime(m["race_id"].map(st["date"]).astype(str) + " " + m["race_id"].map(st["start_time"]).astype(str), errors="coerce")
    comp = pd.to_datetime(m["computed_at"], errors="coerce", utc=True).dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    snap = pd.to_datetime(m["snapshot_at"], errors="coerce", utc=True).dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    mb = ((start - comp).dt.total_seconds() / 60).groupby(m["race_id"]).first()
    res["computed_min_before"] = {"median": float(mb.median()), "p10": float(mb.quantile(0.1)), "p90": float(mb.quantile(0.9)),
                                  "after_t60_share": float(((comp > snap).groupby(m["race_id"]).first()).mean())}
    a = per_race(m.assign(act_rank=m["r60"]), "neg_s").merge(meta, on="race_id", how="left")
    b = per_race(m.assign(act_rank=m["rf"]), "neg_s").merge(meta, on="race_id", how="left")
    c = per_race(m.assign(act_rank=m["rf"], neg60=-m["r60"]), "neg60")
    res["vs_t60"], res["vs_final"], res["t60_vs_final"] = summarize(a, rng), summarize(b, rng), summarize(c, rng)
    res["final_minus_t60"] = diff(b, a, rng, ("spearman", "top1", "top3"))
    gp = gap_of(m, "sec77_odds")
    res["gap_vs_t60"] = by_gap(a, gp, rng)
    for col in ("fs_band", "cls", "shinba"):
        res.setdefault("by", {})[col] = {str(v): summarize(g, rng) for v, g in a.groupby(col)}
    # 対数オッズの誤差（想定 − T-60・想定 − 確定）
    for nm, col in (("t60", "win_odds"), ("final", "odds_f")):
        er = np.log(m["sec77_odds"]) - np.log(m[col])
        res[f"log_err_{nm}"] = {"mae": float(er.abs().mean()), "rmse": float(np.sqrt((er ** 2).mean())), "bias": float(er.mean())}
    # 絞り込み: 想定オッズの k 番以内を、T-60 の k 番人気以内の代わりに使うと
    m["rs"] = m.groupby("race_id")["neg_s"].rank(ascending=False, method="first")
    m["r60f"] = m.groupby("race_id")["popularity"].rank(method="first")
    res["filters"] = {}
    for k in (1, 2, 3, 5):
        a_s, a_6 = (m["rs"] <= k).to_numpy(), (m["r60f"] <= k).to_numpy()
        per = pd.DataFrame({"race_id": m["race_id"].to_numpy(), "io": a_s & ~a_6, "i": a_s, "f": a_s != a_6}).groupby("race_id").agg(io=("io", "sum"), i=("i", "sum"), f=("f", "any"))
        res["filters"][f"top{k}"] = {"in_s_out_60": _ratio_boot(per["io"].to_numpy(float), per["i"].to_numpy(float), None, rng),
                                     "race_any_flip": boot_mean(per["f"].to_numpy(float), rng)}
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    res = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "boot": B}
    res["final"] = part_final(rng)
    _log("部 1 終わり")
    res["pre"] = part_pre(rng)
    _log("部 2 終わり")
    res["sec77_2026"] = part_sec77(rng)
    _log(f"部 3 終わり（{res['sec77_2026'].get('races', 0)} レース）")
    (OUT / "result.json").write_text(json.dumps(_clean(res), ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"保存: {OUT / 'result.json'}")


if __name__ == "__main__":
    main()
