"""tp-34: 休養明け・レース間隔の分析（記述分析と特徴量の評価）。

python -m src.scripts.data.tp34_eval describe   # 間隔の帯・叩き何戦目・属性別の好走率と V4 予測に対する残差（レース単位ブートストラップの 95% 区間）、再現性
python -m src.scripts.data.tp34_eval groups     # 特徴量の群（R1〜R8）を作って tp34/groups/*.parquet に保存
python -m src.scripts.data.tp34_eval stage1     # 第 1 段: valid 2022+2023、seed 1 本。V4 全体に対する増分
python -m src.scripts.data.tp34_eval stage2     # 第 2 段: 第 1 段を通った群を valid 2024、新 seed 2 本で確認
test 2025・2026 年は使わない。市場オッズ・人気は使わない。環境変数: TP34_THREADS（既定 3）, TP34_LOAD_LIMIT（既定 14）。
V4 の特徴量表（tp31/v4_frame.parquet、新しい土台 training_pipeline_v4）と V4 アンサンブルの OOF（tp27/ens/v4_e2.parquet）を読む。
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
OUT = V / "tp34"
FRAME31 = V / "tp31"
for _k, _v in {"TP_BASE_DIR": V, "TP_WORK": V, "TP25C_OUT": V / "tp25c", "TP25E_OUT": V / "tp25e", "TP27_FAC_FIELD": V / "factory_field", "TP27_OUT": V / "tp27"}.items():
    os.environ.setdefault(_k, str(_v))
os.environ.setdefault("TP27_THREADS", os.environ.get("TP34_THREADS", "3"))

from src.pipeline.features import factory_self as fs  # noqa: E402
from src.pipeline.features import tp34_rest as tr  # noqa: E402
from src.scripts.data import tp27_eval as t  # noqa: E402

KEY = ["race_id", "horse_id"]
THREADS = int(os.environ.get("TP34_THREADS", 3))
LOAD_LIMIT = float(os.environ.get("TP34_LOAD_LIMIT", 14))
GROUP_NAMES = list(tr.GROUPS)


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


# ───────────────────────── 記述分析
def _boot_resid(m: pd.DataFrame, cat: str, ycol: str, pcol: str, B: int = 400, seed: int = 0) -> pd.DataFrame:
    """カテゴリごとに 実際の率 − 予測の平均（残差）と、レース単位ブートストラップの 95% 区間。"""
    codes, uniq = pd.factorize(m[cat], sort=False)
    ok = codes >= 0
    race = pd.factorize(m["race_id"])[0]
    R, K = race.max() + 1, len(uniq)
    y, p = m[ycol].to_numpy(float), m[pcol].to_numpy(float)
    S, N, Y, P = (np.zeros((R, K)) for _ in range(4))
    np.add.at(S, (race[ok], codes[ok]), (y - p)[ok])
    np.add.at(N, (race[ok], codes[ok]), 1.0)
    np.add.at(Y, (race[ok], codes[ok]), y[ok])
    np.add.at(P, (race[ok], codes[ok]), p[ok])
    rng = np.random.default_rng(seed)
    W = np.zeros((B, R))
    for b in range(B):
        W[b] = np.bincount(rng.integers(0, R, R), minlength=R)
    tn = W @ N
    with np.errstate(invalid="ignore", divide="ignore"):
        res = (W @ S) / tn
    n = N.sum(0)
    out = pd.DataFrame({"cat": uniq, "n": n.astype(int), "rate": Y.sum(0) / np.maximum(n, 1), "pred": P.sum(0) / np.maximum(n, 1)})
    out["resid"] = out["rate"] - out["pred"]
    out["lo"] = np.nanpercentile(res, 2.5, axis=0)
    out["hi"] = np.nanpercentile(res, 97.5, axis=0)
    return out


def resid_table(m: pd.DataFrame, cat: str, order: list | None = None, years: bool = True) -> pd.DataFrame:
    """勝率・連対率・複勝率それぞれの残差表（カテゴリ行）。years=True なら複勝残差の年別（2022/23/24）も付ける。"""
    parts = {}
    for k, (y, p) in {"win": ("y_win", "p_win"), "top2": ("y_top2", "p_top2"), "top3": ("y_top3", "p_top3")}.items():
        r = _boot_resid(m, cat, y, p)
        parts[k] = r.set_index("cat")
    base = parts["win"][["n"]].copy()
    for k, r in parts.items():
        base[f"{k}_rate"], base[f"{k}_pred"], base[f"{k}_resid"], base[f"{k}_lo"], base[f"{k}_hi"] = r["rate"], r["pred"], r["resid"], r["lo"], r["hi"]
    if years:
        for yr in (2022, 2023, 2024):
            sub = m[m["year"] == yr]
            g = sub.groupby(cat).apply(lambda x: float(x["y_top3"].mean() - x["p_top3"].mean()) if len(x) >= 30 else np.nan)
            base[f"top3_resid_{yr}"] = g.reindex(base.index)
    if order is not None:
        base = base.reindex([o for o in order if o in base.index])
    return base.reset_index().rename(columns={"index": "cat", "cat": "cat"})


def _load_describe_frame() -> pd.DataFrame:
    runs = fs.load_runs(TABLES, list(range(2020, 2026)))
    d = tr.prepare(runs)
    d["jk_chg"] = np.where(fs.lag(d, "jockey_id").isna(), np.nan, (d["jockey_id"].astype(str) != fs.lag(d, "jockey_id").astype(str)).astype(float))
    d["prev_fp"] = fs.lag(d, "fp")
    d["prev_margin"] = fs.lag(d, "margin_len")
    ens = pd.read_parquet(V / "tp27/ens/v4_e2.parquet")
    ens["race_id"], ens["horse_id"] = ens["race_id"].astype(str), ens["horse_id"].astype(str)
    cols = KEY + ["fin", "fp", "gap", "band", "tataki", "is_lay", "is_lay60", "tat_cat", "cls", "age", "surface", "dist", "venue_code", "month", "trainer_id", "jockey_id",
                  "jk_chg", "prev_fp", "prev_margin", "date", "nfield"]
    m = ens.merge(d[cols], on=KEY, how="inner")
    m["y_win"] = (m["fin"] == 1).astype(float)
    m["y_top2"] = (m["fin"] <= 2).astype(float)
    m["y_top3"] = (m["fin"] <= 3).astype(float)
    return m, d


def cmd_describe(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m, d = _load_describe_frame()
    _log(f"V4 OOF {len(m)} 行（{m['race_id'].nunique()} レース）、間隔あり {int(m['gap'].notna().sum())}")
    res: dict = {"n_rows": int(len(m)), "n_races": int(m["race_id"].nunique())}
    mg = m[m["gap"].notna()].copy()

    mg["band_l"] = mg["band"].map(lambda x: tr.BAND_NAMES[int(x)])
    res["band"] = resid_table(mg, "band_l", tr.BAND_NAMES).to_dict("records")
    mg["tat_l"] = mg["tat_cat"].map({"T1": "叩き1戦目(休み明け初戦)", "T2": "叩き2戦目", "T3": "叩き3戦目", "T4p": "叩き4戦目以降", "T0": "不明"})
    res["tataki"] = resid_table(mg[mg["tat_cat"] != "T0"], "tat_l", ["叩き1戦目(休み明け初戦)", "叩き2戦目", "叩き3戦目", "叩き4戦目以降"]).to_dict("records")

    lay = mg[(mg["tat_cat"] == "T1")].copy()
    ref = mg[(mg["gap"] >= 15) & (mg["gap"] <= 56)].copy()
    lay["grp"], ref["grp"] = "休み明け初戦(90日以上)", "標準的な間隔(3〜8週)"
    both = pd.concat([lay, ref], ignore_index=True)
    res["lay_vs_ref"] = resid_table(both, "grp").to_dict("records")

    def cut_table(sub: pd.DataFrame, name: str, col: str, bins, labels) -> list[dict]:
        s = sub.copy()
        s[name] = pd.cut(s[col], bins, labels=labels).astype("string")
        s = s[s[name].notna()]
        s["cell"] = s[name]
        return resid_table(s, "cell", labels).to_dict("records")

    for nm, sub in (("休み明け初戦", lay), ("標準的な間隔", ref)):
        res[f"age|{nm}"] = cut_table(sub, "ag", "age", [0, 3.5, 4.5, 5.5, 99], ["3歳", "4歳", "5歳", "6歳以上"])
        res[f"cls|{nm}"] = cut_table(sub, "cl", "cls", [-1, 0.5, 1.5, 3.5, 99], ["未勝利・新馬", "1勝", "2〜3勝", "OP以上"])
        res[f"dist|{nm}"] = cut_table(sub, "ds", "dist", [0, 1300, 1700, 2100, 9999], ["〜1300m", "1400〜1700m", "1800〜2100m", "2200m〜"])
    for nm, sub in (("休み明け初戦", lay), ("標準的な間隔", ref)):
        s = sub.copy()
        s["cell"] = s["surface"].astype(str)
        res[f"surface|{nm}"] = resid_table(s[s["cell"].isin(["芝", "ダート"])], "cell", ["芝", "ダート"]).to_dict("records")
        s["cell"] = s["month"].map(lambda x: f"{int(x)}月")
        res[f"month|{nm}"] = resid_table(s, "cell", [f"{i}月" for i in range(1, 13)], years=False).to_dict("records")

    # 前走の結果 × 間隔
    s = mg[mg["prev_fp"].notna()].copy()
    s["pr"] = pd.cut(s["prev_fp"], [-0.01, 0.0001, 0.2, 0.5, 1.01], labels=["前走1着", "前走上位(〜2割)", "前走中位", "前走下位"]).astype("string")
    s["gb"] = pd.cut(s["gap"], [0, 14, 56, 91, 9999], labels=["〜2週", "3〜8週", "9〜13週", "14週〜"]).astype("string")
    s["cell"] = s["gb"] + " × " + s["pr"]
    res["prev×gap"] = resid_table(s, "cell", [f"{g} × {p}" for g in ["〜2週", "3〜8週", "9〜13週", "14週〜"] for p in ["前走1着", "前走上位(〜2割)", "前走中位", "前走下位"]], years=False).to_dict("records")

    # 乗り替わり × 間隔
    s = mg[mg["jk_chg"].notna()].copy()
    s["cell"] = s["gb"] if "gb" in s else pd.cut(s["gap"], [0, 14, 56, 91, 9999], labels=["〜2週", "3〜8週", "9〜13週", "14週〜"]).astype("string")
    s["cell"] = s["cell"].astype(str) + " × " + s["jk_chg"].map({0.0: "継続騎乗", 1.0: "乗り替わり"})
    res["jockey×gap"] = resid_table(s, "cell", [f"{g} × {j}" for g in ["〜2週", "3〜8週", "9〜13週", "14週〜"] for j in ["継続騎乗", "乗り替わり"]], years=False).to_dict("records")

    # 休養中の調教の本数
    lp = WORK / "features/training_laps.parquet"
    if lp.exists():
        wt = tr.work_table(pd.read_parquet(lp))
        key = mg[["horse_id", "date"]].copy()
        key["race_date"] = key["date"].dt.strftime("%Y-%m-%d")
        j = key.merge(wt, on=["horse_id", "race_date"], how="left")
        j.index = mg.index
        s = mg[(mg["gap"] >= 60)].copy()
        s["n_work"] = j.loc[s.index, "n_work"].to_numpy()
        s["cell"] = pd.cut(s["n_work"], [-1, 0, 2, 5, 9, 99], labels=["記録なし", "1〜2本", "3〜5本", "6〜9本", "10本以上"]).astype("string")
        s["cell"] = s["cell"].fillna("記録なし")
        res["work|休養60日以上"] = resid_table(s, "cell", ["記録なし", "1〜2本", "3〜5本", "6〜9本", "10本以上"], years=False).to_dict("records")

    # 再現性: 馬ごとの休み明けの得意不得意（split-half）
    d2 = d[d["tataki"].notna()].copy()
    ret = d2[(d2["tataki"] == 1)].copy()
    rest = d2[(d2["tataki"] != 1)].groupby("horse_id")["fp"].mean().rename("fp_other")
    ret = ret.merge(rest, on="horse_id", how="inner")
    ret["x"] = ret["fp"] - ret["fp_other"]
    ret["k"] = ret.groupby("horse_id").cumcount()
    cnt = ret.groupby("horse_id")["x"].transform("size")
    r4 = ret[cnt >= 4]
    if len(r4):
        a_ = r4[r4["k"] % 2 == 0].groupby("horse_id")["x"].mean()
        b_ = r4[r4["k"] % 2 == 1].groupby("horse_id")["x"].mean()
        ab = pd.concat([a_, b_], axis=1, keys=["a", "b"]).dropna()
        r = float(np.corrcoef(ab["a"], ab["b"])[0, 1])
        res["horse_split_half"] = {"horses": int(len(ab)), "r": r, "spearman_brown": float(2 * r / (1 + r)) if r > -1 else None,
                                   "mean_x": float(r4["x"].mean()), "note": "x = 休み明け初戦の fp − 同じ馬のそれ以外の走の平均 fp（正 = 休み明けで悪い）。偶数番目/奇数番目の休み明けの平均 x の相関"}
    # 再現性: 調教師ごとの休み明けの仕上げ（前半 2022-23 / 後半 2024 の残差の相関）
    s = m[(m["tat_cat"] == "T1")].copy()
    s["half"] = np.where(s["year"] <= 2023, "h1", "h2")
    s["rt3"] = s["y_top3"] - s["p_top3"]
    s["rw"] = s["y_win"] - s["p_win"]
    piv = s.groupby(["trainer_id", "half"]).agg(n=("rt3", "size"), rt3=("rt3", "mean"), rw=("rw", "mean")).reset_index()
    p1, p2 = piv[piv["half"] == "h1"].set_index("trainer_id"), piv[piv["half"] == "h2"].set_index("trainer_id")
    jn = p1.join(p2, lsuffix="_1", rsuffix="_2", how="inner")
    jn = jn[(jn["n_1"] >= 15) & (jn["n_2"] >= 8)]
    if len(jn) >= 10:
        res["trainer_split"] = {"trainers": int(len(jn)), "r_top3": float(np.corrcoef(jn["rt3_1"], jn["rt3_2"])[0, 1]), "r_win": float(np.corrcoef(jn["rw_1"], jn["rw_2"])[0, 1]),
                                "note": "休み明け初戦（叩き 1 戦目）の V4 に対する残差を、前半（2022-23）と後半（2024）で調教師ごとに出して相関を見た"}
    _jsave(OUT / "describe.json", res)
    _log("記述分析を保存: " + str(OUT / "describe.json"))


# ───────────────────────── 特徴量の群
def cmd_groups(a) -> None:
    df, base, meta, folds, lp = t.setup()
    runs = fs.load_runs(TABLES, list(range(2020, 2026)))
    laps = pd.read_parquet(WORK / "features/training_laps.parquet") if (WORK / "features/training_laps.parquet").exists() else None
    t0 = time.time()
    parts = tr.build_all(runs, laps)
    gdir = OUT / "groups"
    gdir.mkdir(parents=True, exist_ok=True)
    keys = df[KEY].reset_index(drop=True)
    for g, G in parts.items():
        T = keys.merge(G, on=KEY, how="left")
        feat = [c for c in T.columns if c not in KEY]
        nn = T[feat].notna().mean()
        keep = [c for c in feat if nn[c] > 0.02 and T[c].nunique(dropna=True) > 1]
        T[KEY + keep].to_parquet(gdir / f"{g}.parquet", index=False)
        _log(f"{g}: {len(keep)} 列（除外 {len(feat) - len(keep)}）、欠損率 {float(T[keep].isna().mean().mean()):.3f} ({time.time() - t0:.0f}s)")


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
    df, base, meta, folds, lp = t.setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    X0 = _frame(df)
    d = OUT / label
    d.mkdir(parents=True, exist_ok=True)
    resp = d / "results.json"
    res = _jload(resp)
    bp = d / "base.parquet"
    if not bp.exists() and src31 and (FRAME31 / src31 / "base.parquet").exists():
        shutil.copy(FRAME31 / src31 / "base.parquet", bp)                       # tp-31 と同じ設定の V4 基準の OOF を再利用する
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


def cmd_stage1(a) -> None:
    groups = [g for g in os.environ.get("TP34_GROUPS", "").split(",") if g] or GROUP_NAMES          # 並列に回すときは TP34_GROUPS / TP34_LABEL で分担する
    _stage(os.environ.get("TP34_LABEL", "stage1"), [2022, 2023], [0], groups, "stage1")


def cmd_stage2(a) -> None:
    allg: dict = {}
    for lab in ("stage1", "stage1b", "stage1c", "stage1d"):             # 後のラベルが前を上書きする（stage1c/d は excl_cum の端数のリーク修正後にやり直した群）
        allg.update((_jload(OUT / lab / "results.json") or {}).get("groups", {}))
    passed = [g for g in GROUP_NAMES if g in allg and allg[g]["vs_V4"]["Δwin"] < -0.00012 and allg[g]["vs_V4"]["win_hi"] < 0.0001]
    _log(f"第 1 段を通過: {passed}")
    _jsave(OUT / "stage1_passed.json", passed)
    if not passed:
        return
    _stage("stage2", [2024], [2, 3], passed + (["ALL_PASSED"] if len(passed) > 1 else []), "stage2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["describe", "groups", "stage1", "stage2"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"describe": cmd_describe, "groups": cmd_groups, "stage1": cmd_stage1, "stage2": cmd_stage2}[a.cmd](a)


if __name__ == "__main__":
    main()
