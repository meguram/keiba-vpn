"""走法軸（先行力・Cr性能・まくり特性・消耗耐性・出脚）の「旧定義（横位置）」と「新定義（累積タイム順位）」を同じ手順で比べる（読むだけ・JSON を書く）。

背景: horse_lap の sections[].position は通過順位でなく横位置だった。新しい run_axes.py は区間の累積タイム順位を使う。
入力（ホストのローカルだけ。GCS は読まない）:
  新定義の run_axes.py（開発 PC で書き換えたもの。/tmp に置く）… importlib で別名モジュールとして読み、compute() を呼ぶ
  /home/jovyan/acq/horse_lap/horse_lap_*.jsonl … 新定義の入力
  data/local/running_profile/run_axes.parquet … 旧定義（2024 年以降の走だけ使う）
  data/page_reference/tables/<年>/race_result_flat.parquet … 人気・頭数・着順（期待値の学習は 2016〜2023）
  data/meta/course_master/jra_courses.json … 周長・最終直線
  （任意）/tmp/pos_*.parquet … 別エージェントの検証値（as-of 値の突き合わせ）
評価（旧新で全く同じ手順）: 各馬の「その走より前の直近 10 走の軸の平均（2 走以上）」を as-of 値にし、レース内で平均を引く。
  1SD あたりの、人気補正後の勝率・複勝率（pt）の傾きを、レース単位ブートストラップ（300 回）で 95% 区間つきで出す。
  期間: 選択 = 2024-10〜2025-06、確認 = 2025-07〜2025-12（1 回だけ見る）、参考 = 2026（選択に使わない）。
  標本: own = 旧新それぞれ自分の as-of が取れる走 / common = 旧新どちらも取れる走（同じ走で比べる。差の区間は同じレースの引き直し）。
再現性: 馬ごとに時系列の最初の 4 走（と 6 走）を奇数番目/偶数番目に分けた平均の相関 → Spearman-Brown で 1 走の信頼性に換算。
使い方（ホストで）:
  cd <repo> && nice -n 19 /opt/conda/bin/python -I /tmp/axis_rank_eval.py NEW_RUN_AXES.py OUT.json [--reuse] [--sel-only]
  --reuse: /tmp/run_axes_new.parquet があれば作り直さない / --sel-only: 選択期間だけ（確認・参考を見ない。デバッグ用）
"""
from __future__ import annotations

import importlib.util
import json
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path.cwd()
sys.path.insert(0, str(REPO))

from src.scripts.data.axis_score_curves import expected_tables, load_results, place_cut  # noqa: E402

NEW_PARQUET = Path("/tmp/run_axes_new.parquet")
OLD_PARQUET = REPO / "data/local/running_profile/run_axes.parquet"
LAPS = sorted(glob("/home/jovyan/acq/horse_lap/horse_lap_*.jsonl"))
AXES = ["front", "cr", "makuri", "endure", "debut"]
SUBSET_AXES = ["front", "cr", "makuri"]
PERIODS = {"選択": ("2024-10-01", "2025-06-30"), "確認": ("2025-07-01", "2025-12-31"), "参考2026": ("2026-01-01", "2026-12-31")}
LAST = 10
MIN_PREV = 2
B = 300
rng = np.random.default_rng(20261119)


# ── 準備 ────────────────────────────────────────────────────────────────

def build_new(py_path: str, reuse: bool) -> pd.DataFrame:
    if reuse and NEW_PARQUET.exists():
        return pd.read_parquet(NEW_PARQUET)
    spec = importlib.util.spec_from_file_location("run_axes_new", py_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_axes_new"] = mod
    spec.loader.exec_module(mod)
    df = mod.compute(LAPS)
    df.to_parquet(NEW_PARQUET, index=False)
    return df


def prep_axes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["race_id"] = df["race_id"].astype(str)
    df["horse_id"] = df["horse_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= pd.Timestamp("2024-01-01")]
    df = df.drop_duplicates(["race_id", "horse_id"], keep="last")
    return df.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)


def asof_mean(df: pd.DataFrame, col: str) -> pd.Series:
    """その走より前の直近 LAST 走（欠測の走は数えない）の平均。取れた走が MIN_PREV 未満なら NaN。df は馬・日付で整列済み。"""
    v = df[col]
    g = df["horse_id"]
    x = v.fillna(0.0)
    c = v.notna().astype(float)
    cs = x.groupby(g).cumsum()
    cc = c.groupby(g).cumsum()
    s = cs.groupby(g).shift(1).fillna(0) - cs.groupby(g).shift(LAST + 1).fillna(0)
    n = cc.groupby(g).shift(1).fillna(0) - cc.groupby(g).shift(LAST + 1).fillna(0)
    return (s / n.where(n > 0)).where(n >= MIN_PREV)


def load_course_master() -> pd.DataFrame:
    cm = pd.DataFrame(json.load(open(REPO / "data/meta/course_master/jra_courses.json", encoding="utf-8")))
    return cm.groupby(["venue", "surface", "distance"]).agg(final_straight_m=("final_straight_m", "mean"), lap_length_m=("lap_length_m", "mean")).reset_index()


def build_eval_frame(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """走ごと 1 行: 結果・人気補正・コース属性・旧新の as-of 値。"""
    for tag, df in (("old", old), ("new", new)):
        for a in AXES:
            df[f"{a}_asof_{tag}"] = asof_mean(df, a)
    keys = ["race_id", "horse_id"]
    cols_old = keys + ["date", "venue", "surface", "distance"] + [a for a in AXES] + [f"{a}_asof_old" for a in AXES]
    o = old[cols_old].rename(columns={a: f"{a}_old" for a in AXES})
    n = new[keys + ["date", "venue", "surface", "distance"] + AXES + [f"{a}_asof_new" for a in AXES]].rename(columns={a: f"{a}_new" for a in AXES})
    n = n.rename(columns={"date": "date_n", "venue": "venue_n", "surface": "surface_n", "distance": "distance_n"})
    d = o.merge(n, on=keys, how="outer")
    for c in ("date", "venue", "surface", "distance"):
        d[c] = d[c].where(d[c].notna(), d[c + "_n"])
    d = d.drop(columns=["date_n", "venue_n", "surface_n", "distance_n"])
    # 同一履歴（common）: 旧新どちらも値がある走だけを履歴に使い直す。旧は 2024 年の走の約 87% で横位置が無く、
    # そのまま比べると旧の as-of が薄い（定義でなくデータの欠けの差）ので、両方が揃う走だけで as-of を作り直して比べる。
    d = d.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    for a in AXES:
        d[f"{a}_c_old"] = d[f"{a}_old"].where(d[f"{a}_new"].notna())
        d[f"{a}_c_new"] = d[f"{a}_new"].where(d[f"{a}_old"].notna())
        d[f"{a}_asof_cold"] = asof_mean(d, f"{a}_c_old")
        d[f"{a}_asof_cnew"] = asof_mean(d, f"{a}_c_new")

    res = load_results(range(2016, 2027))
    yr = res["race_id"].str[:4].astype(int)
    Tw, Tp = expected_tables(res[yr.isin(list(range(2016, 2024)))])
    d = d.merge(res[["race_id", "horse_id", "popularity", "field_size", "finish_position"]], on=keys, how="left")
    d = d[(d["finish_position"] > 0) & (d["field_size"] >= 5) & (d["popularity"] >= 1) & d["surface"].isin(["芝", "ダート"])].copy()
    nf = d["field_size"].to_numpy().astype(int)
    nb = np.clip(nf, 5, 18)
    pp = np.minimum(d["popularity"].to_numpy().astype(int), nb)
    d["resid_w"] = ((d["finish_position"] == 1).to_numpy(float) - Tw[nb, pp]) * 100
    d["resid_p"] = ((d["finish_position"].to_numpy() <= place_cut(nf)).astype(float) - Tp[nb, pp]) * 100
    cm = load_course_master()
    d = d.merge(cm, on=["venue", "surface", "distance"], how="left")
    return d.reset_index(drop=True)


# ── 傾き ────────────────────────────────────────────────────────────────

def race_rel(d: pd.DataFrame, v: pd.Series) -> pd.Series:
    return v - v.groupby(d["race_id"]).transform("mean")


def _sums(x, y, rc, nr):
    return np.bincount(rc, x * y, minlength=nr), np.bincount(rc, x * x, minlength=nr)


def _boot(sxy, sxx, idx):
    return sxy[idx].sum(1) / sxx[idx].sum(1)


def slope_one(x, y, race, label=""):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, race = x[ok], y[ok], race[ok]
    if len(x) < 300:
        return None
    sd = x.std()
    x = x / sd
    _, rc = np.unique(race, return_inverse=True)
    sxy, sxx = _sums(x, y, rc, rc.max() + 1)
    idx = rng.integers(0, len(sxy), (B, len(sxy)))
    b = _boot(sxy, sxx, idx)
    return {"slope": float(sxy.sum() / sxx.sum()), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)), "n": int(len(x)), "races": int(len(sxy))}


def slope_pair(xo, xn, y, race):
    """同じ走・同じレースの引き直しで、旧・新・差（新 − 旧）を出す。"""
    ok = np.isfinite(xo) & np.isfinite(xn) & np.isfinite(y)
    xo, xn, y, race = xo[ok], xn[ok], y[ok], race[ok]
    if len(y) < 300:
        return None
    xo, xn = xo / xo.std(), xn / xn.std()
    _, rc = np.unique(race, return_inverse=True)
    nr = rc.max() + 1
    so = _sums(xo, y, rc, nr)
    sn = _sums(xn, y, rc, nr)
    idx = rng.integers(0, nr, (B, nr))
    bo, bn = _boot(*so, idx), _boot(*sn, idx)
    out = {}
    for k, s, b in (("old", so, bo), ("new", sn, bn), ("diff", None, bn - bo)):
        est = (sn[0].sum() / sn[1].sum() - so[0].sum() / so[1].sum()) if k == "diff" else s[0].sum() / s[1].sum()
        out[k] = {"slope": float(est), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)), "n": int(len(y)), "races": int(nr)}
    return out


def in_period(d: pd.DataFrame, p: tuple[str, str]) -> np.ndarray:
    return d["date"].between(pd.Timestamp(p[0]), pd.Timestamp(p[1])).to_numpy()


def subset_masks(d: pd.DataFrame) -> dict[str, np.ndarray]:
    m = {"全体": np.ones(len(d), bool)}
    ll = d["lap_length_m"]
    q = ll.quantile(1 / 3)
    m[f"周長短(≤{q:.0f}m)"] = (ll <= q).to_numpy()
    dirt = d["surface"] == "ダート"
    qs = d.loc[dirt, "final_straight_m"].quantile(1 / 3)
    m[f"ダート短直線(≤{qs:.0f}m)"] = (dirt & (d["final_straight_m"] <= qs)).to_numpy()
    return m


def evaluate(d: pd.DataFrame, periods: dict) -> dict:
    out: dict = {}
    subs = subset_masks(d)
    for a in AXES:
        out[a] = {"own": {}, "common": {}}
        ao, an = d[f"{a}_asof_old"], d[f"{a}_asof_new"]
        # common: 旧新どちらも値がある走だけで as-of を作り直し、どちらの as-of も取れる走で、レース内の平均を引く
        co, cn = d[f"{a}_asof_cold"], d[f"{a}_asof_cnew"]
        cm = co.notna() & cn.notna()
        xo_c = race_rel(d[cm], co[cm]).reindex(d.index)
        xn_c = race_rel(d[cm], cn[cm]).reindex(d.index)
        xo_o = race_rel(d[ao.notna()], ao[ao.notna()]).reindex(d.index)
        xn_o = race_rel(d[an.notna()], an[an.notna()]).reindex(d.index)
        for sname, smask in subs.items():
            if sname != "全体" and a not in SUBSET_AXES:
                continue
            for pname, p in periods.items():
                pm = smask & in_period(d, p)
                for tname, ycol in (("勝率", "resid_w"), ("複勝率", "resid_p")):
                    y = d[ycol].to_numpy()
                    rid = d["race_id"].to_numpy()
                    key = f"{sname}|{pname}|{tname}"
                    out[a]["common"][key] = slope_pair(xo_c.to_numpy()[pm], xn_c.to_numpy()[pm], y[pm], rid[pm])
                    out[a]["own"][key] = {"old": slope_one(xo_o.to_numpy()[pm], y[pm], rid[pm]), "new": slope_one(xn_o.to_numpy()[pm], y[pm], rid[pm])}
    return out


# ── 再現性・欠測・相関 ──────────────────────────────────────────────────────

def reliability(df: pd.DataFrame, col: str, k: int) -> dict | None:
    t = df[["horse_id", col]].dropna()
    t["i"] = t.groupby("horse_id").cumcount()
    t = t[t["i"] < k]
    w = t.pivot(index="horse_id", columns="i", values=col).dropna()
    if len(w) < 200:
        return None
    a = w[[i for i in range(k) if i % 2 == 0]].mean(axis=1)
    b = w[[i for i in range(k) if i % 2 == 1]].mean(axis=1)
    r = float(np.corrcoef(a, b)[0, 1])
    R = 2 * r / (1 + r)                       # k 走平均の信頼性
    return {"r_half": round(r, 3), "rho1": round(R / (k - (k - 1) * R), 3), "horses": int(len(w))}


def missing_report(old: pd.DataFrame, new: pd.DataFrame) -> dict:
    rep = {"rows": {"old": len(old), "new": len(new)}, "nan_rate_pct": {}}
    for tag, df in (("old", old), ("new", new)):
        rep["nan_rate_pct"][tag] = {a: round(float(df[a].isna().mean() * 100), 1) for a in AXES}
        rep["nan_rate_pct"][tag + "_by_year"] = {str(y): {a: round(float(g[a].isna().mean() * 100), 1) for a in AXES} for y, g in df.groupby(df["date"].dt.year)}
    g = new.assign(mk_nan=new["makuri"].isna()).groupby(["surface", "venue", "distance"]).agg(n=("mk_nan", "size"), nan=("mk_nan", "mean")).reset_index()
    g = g[g["n"] >= 100]
    rep["new_makuri_nan_courses"] = {f"{r.venue}{r.surface}{r.distance}": [int(r.n), round(float(r.nan) * 100, 0)] for r in g.itertuples() if r.nan > 0.5}
    rep["new_makuri_nan_by_surface"] = {s: round(float(x.isna().mean() * 100), 1) for s, x in new.groupby("surface")["makuri"]}
    rep["new_makuri_nan_by_distance_band"] = {str(k): round(float(x.isna().mean() * 100), 1) for k, x in new.groupby(pd.cut(new["distance"], [0, 1200, 1400, 1700, 2000, 2400, 9999]), observed=True)["makuri"]}
    return rep


def run_level_agreement(d: pd.DataFrame) -> dict:
    out = {}
    for a in AXES:
        t = d[[f"{a}_old", f"{a}_new"]].dropna()
        out[a] = {"n": int(len(t)), "pearson": round(float(t.corr().iloc[0, 1]), 3), "max_abs_diff": round(float((t[f"{a}_old"] - t[f"{a}_new"]).abs().max()), 4)}
    return out


def front_debut_corr(d: pd.DataFrame, periods: dict) -> dict:
    out = {}
    for tag in ("old", "new"):
        for pname, p in periods.items():
            m = in_period(d, p) & d[f"front_asof_{tag}"].notna().to_numpy() & d[f"debut_asof_{tag}"].notna().to_numpy()
            t = d[m]
            fr, de = race_rel(t, t[f"front_asof_{tag}"]), race_rel(t, t[f"debut_asof_{tag}"])
            out[f"{tag}|{pname}"] = {"corr_asof_rel": round(float(np.corrcoef(fr, de)[0, 1]), 3), "n": int(len(t))}
    return out


def verify_with_pos(d: pd.DataFrame, new: pd.DataFrame) -> dict:
    """別エージェントの検証値（/tmp/pos_*.parquet）との突き合わせ。"""
    out: dict = {}
    try:
        runs = pd.read_parquet("/tmp/pos_runs.parquet", columns=["race_id", "horse_id", "k", "r1C", "r2C", "r3C", "r4C", "r_first", "field_size"])
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    runs["race_id"] = runs["race_id"].astype(str)
    runs["horse_id"] = runs["horse_id"].astype(str)
    keys = ["race_id", "horse_id"]
    m = new[keys + ["front", "cr", "makuri"]].merge(runs, on=keys, how="inner")
    out["run_level_spearman_vs_passing_order"] = {}
    for name, a, raw, cond in (
        ("cr_new vs 通過順位 3C-4C(k=2,4)", "cr", m["r3C"] - m["r4C"], m["k"].isin([2, 4])),
        ("makuri_new vs 通過順位 2C-4C(k=4)", "makuri", m["r2C"] - m["r4C"], m["k"] == 4),
        ("front_new vs 最初の通過順位(-r_first)", "front", -m["r_first"], m["k"] > 0),
    ):
        t = pd.DataFrame({"a": m[a], "r": raw})[cond].dropna()
        out["run_level_spearman_vs_passing_order"][name] = {"rho": round(float(t.corr(method="spearman").iloc[0, 1]), 3), "n": int(len(t))}
    # 旧（横位置）との相関も並べる
    mo = d[keys + ["cr_old", "makuri_old", "front_old"]].merge(runs, on=keys, how="inner")
    for name, a, raw, cond in (
        ("cr_old vs 通過順位 3C-4C(k=2,4)", "cr_old", mo["r3C"] - mo["r4C"], mo["k"].isin([2, 4])),
        ("makuri_old vs 通過順位 2C-4C(k=4)", "makuri_old", mo["r2C"] - mo["r4C"], mo["k"] == 4),
        ("front_old vs -r_first", "front_old", -mo["r_first"], mo["k"] > 0),
    ):
        t = pd.DataFrame({"a": mo[a], "r": raw})[cond].dropna()
        out["run_level_spearman_vs_passing_order"][name] = {"rho": round(float(t.corr(method="spearman").iloc[0, 1]), 3), "n": int(len(t))}
    # as-of 値の突き合わせ（相手は直近 5 走・縮小つき・コース標準化、こちらは直近 10 走の素平均なので完全一致はしない）
    pairs = [("cr", "/tmp/pos_cr1.parquet", ["x_C1", "x_C3"]), ("makuri", "/tmp/pos_mk1.parquet", ["x_M3", "x_M4"]),
             ("front", "/tmp/pos_front1_asof.parquet", ["F1b 最初の通過順位 全k", "F1l 累積タイム順位 600/1000m(2024〜)"])]
    out["asof_corr_new_vs_other_agent"] = {}
    for a, path, cols in pairs:
        try:
            o = pd.read_parquet(path)
        except Exception as e:  # noqa: BLE001
            out["asof_corr_new_vs_other_agent"][a] = {"error": str(e)}
            continue
        o["race_id"] = o["race_id"].astype(str)
        o["horse_id"] = o["horse_id"].astype(str)
        mm = d[keys + [f"{a}_asof_new", f"{a}_asof_old"]].merge(o[keys + cols], on=keys, how="inner")
        for c in cols:
            row = {}
            for tag in ("new", "old"):
                t = mm[[f"{a}_asof_{tag}", c]].dropna()
                row[tag] = {"corr": round(float(t.corr().iloc[0, 1]), 3), "n": int(len(t))} if len(t) > 500 else None
            out["asof_corr_new_vs_other_agent"][f"{a}: {c}"] = row
    return out


# ── 表示 ────────────────────────────────────────────────────────────────

def fmt(r) -> str:
    if r is None:
        return "-"
    star = "*" if (r["lo"] > 0 or r["hi"] < 0) else " "
    return f"{r['slope']:+.2f}[{r['lo']:+.2f},{r['hi']:+.2f}]{star}"


def print_tables(res: dict) -> None:
    for a, v in res.items():
        for sample in ("common", "own"):
            print(f"\n### {a} / 標本 {sample}（傾き = 1SD あたり人気補正後の pt。* は区間が 0 を跨がない）")
            print("条件|期間|目的 | 旧 | 新 | 差(新-旧) | n(旧/新)")
            for key, r in v[sample].items():
                if r is None:
                    continue
                if sample == "common":
                    print(f"{key} | {fmt(r['old'])} | {fmt(r['new'])} | {fmt(r['diff'])} | {r['new']['n']}")
                else:
                    print(f"{key} | {fmt(r['old'])} | {fmt(r['new'])} | - | {r['old']['n'] if r['old'] else '-'}/{r['new']['n'] if r['new'] else '-'}")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    reuse, sel_only = "--reuse" in sys.argv, "--sel-only" in sys.argv
    new_py, out_path = args[0], Path(args[1])
    periods = {k: v for k, v in PERIODS.items() if not sel_only or k == "選択"}

    new = prep_axes(build_new(new_py, reuse))
    old = prep_axes(pd.read_parquet(OLD_PARQUET))
    print("旧", len(old), "走 / 新", len(new), "走", flush=True)
    miss = missing_report(old, new)
    d = build_eval_frame(old, new)
    print("評価用", len(d), "走（結果あり・5 頭以上・人気あり）。コース属性の結合率", round(float(d["lap_length_m"].notna().mean()), 3), flush=True)

    res = evaluate(d, periods)
    print_tables(res)

    rel = {}
    for tag, df in (("old", old), ("new", new)):
        rel[tag] = {a: {"first4": reliability(df, a, 4), "first6": reliability(df, a, 6)} for a in AXES}
    for tag in ("old", "new"):      # 同一履歴（旧新どちらも値がある走だけ）
        rel[tag + "_common"] = {a: {"first4": reliability(d.rename(columns={f"{a}_c_{tag}": a}), a, 4), "first6": reliability(d.rename(columns={f"{a}_c_{tag}": a}), a, 6)} for a in AXES}
    print("\n### 再現性（1 走換算 rho1）旧 | 新（own）と 旧 | 新（common）")
    for a in AXES:
        print(a, "first4", rel["old"][a]["first4"], "|", rel["new"][a]["first4"], "|| common", rel["old_common"][a]["first4"], "|", rel["new_common"][a]["first4"])
        print(a, "first6", rel["old"][a]["first6"], "|", rel["new"][a]["first6"], "|| common", rel["old_common"][a]["first6"], "|", rel["new_common"][a]["first6"])

    extra = {"missing": miss, "run_level_agreement_old_vs_new": run_level_agreement(d), "front_debut_corr": front_debut_corr(d, periods),
             "verify_pos": verify_with_pos(d, new), "as_of_rate_pct": {tag: {a: round(float(d[f"{a}_asof_{tag}"].notna().mean() * 100), 1) for a in AXES} for tag in ("old", "new")}}
    print("\n### 欠測・整合・相関")
    print(json.dumps(extra, ensure_ascii=False, indent=1)[:12000])
    out_path.write_text(json.dumps({"periods": periods, "last": LAST, "slopes": res, "reliability": rel, **extra}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
