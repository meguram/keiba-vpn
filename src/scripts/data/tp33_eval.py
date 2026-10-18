"""tp-33: 輸送の有無・滞在（追い切りの場所）と好走率の分析と、特徴量の評価。

python -m src.scripts.data.tp33_eval audit     # 追い切りの場所・調教師の拠点が判定できるかの監査 → /tmp/tp33_data_audit.md
python -m src.scripts.data.tp33_eval groups    # 特徴量群（TT_tier / TT_move / TT_stay / TT_rate）を作って tp33/groups/*.parquet に保存
python -m src.scripts.data.tp33_eval descr     # 記述分析（輸送区分別・滞在別の好走率と、V4 の予測に対する残差。信頼区間つき）
python -m src.scripts.data.tp33_eval stage1    # 第 1 段: valid 2022+2023、seed 1 本。V4 全体に対する増分
python -m src.scripts.data.tp33_eval stage2    # 第 2 段: 通過群を valid 2024、新 seed 2 本で確認
test 2025・2026 は使わない。市場オッズ・人気は使わない。環境変数: TP33_THREADS（既定 4）, TP33_LOAD_LIMIT（既定 15）。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
V = ROOT / "data/local/modeling/training_pipeline_v4"
OUT = V / "tp33"
LAPS = ROOT / "data/local/modeling/training_pipeline/features/training_laps.parquet"
for _k, _v in {"TP_BASE_DIR": V, "TP_WORK": V, "TP25C_OUT": V / "tp25c", "TP25E_OUT": V / "tp25e", "TP27_FAC_FIELD": V / "factory_field", "TP27_OUT": V / "tp27"}.items():
    os.environ.setdefault(_k, str(_v))
os.environ.setdefault("TP27_THREADS", os.environ.get("TP33_THREADS", "4"))

from src.pipeline.features import tp33_transport as tt  # noqa: E402
from src.scripts.data import tp27_eval as t  # noqa: E402

KEY = ["race_id", "horse_id"]
THREADS = int(os.environ.get("TP33_THREADS", 4))
LOAD_LIMIT = float(os.environ.get("TP33_LOAD_LIMIT", 15))
GROUPS = ["TT_tier", "TT_move", "TT_stay", "TT_rate"]


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


def _laps() -> pd.DataFrame:
    return pd.read_parquet(LAPS, columns=["horse_id", "race_date", "train_date", "course", "days_before"])


def _table() -> pd.DataFrame:
    cols = ["race_id", "horse_id", "date", "year", "venue_code", "trainer_id", "finish_position", "grade", "race_class", "surface", "distance", "field_size"]
    try:
        d = pd.read_parquet(V / "features/train_table.parquet", columns=cols)
    except Exception:
        cols.remove("year")
        d = pd.read_parquet(V / "features/train_table.parquet", columns=cols)
        d["year"] = pd.to_datetime(d["date"]).dt.year
    d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    d["finish_position"] = pd.to_numeric(d["finish_position"], errors="coerce")
    return d.sort_values(["date", "race_id"], kind="stable").reset_index(drop=True)


# ───────────────────────── 監査
def cmd_audit(a) -> None:
    L = tt.prep_laps(_laps())
    d = _table()
    lines = ["# tp-33 データ監査: 輸送と滞在を判定できるか", ""]
    lc = L["loc"].value_counts()
    lines += ["## 追い切り（horse_training 由来、約 %s 本）の場所" % f"{len(L):,}", "| 場所 | 本数 | 割合 |", "|---|---|---|"]
    for k, v in lc.items():
        lines.append(f"| {k} | {v:,} | {v / len(L):.2%} |")
    lines += ["", "場所の判定: コース名の先頭（栗/美/南/北=トレセン、札/函/福/新/東/中/京/阪/小=開催場の調教場、CW・DP=栗東）。地方競馬場名は NAR、読めないものは UNK。", ""]
    home = tt.trainer_home(L, d)
    x = L[L["loc"].isin(tt.CENTER_LOCS)].merge(d[["horse_id", "date", "trainer_id"]].rename(columns={"date": "race_date"}).astype({"horse_id": str, "race_date": str}), on=["horse_id", "race_date"])
    pur = x.groupby("trainer_id")["loc"].agg(lambda s: float(max((s == "KU").mean(), (s == "MI").mean())))
    cnt = x.groupby("trainer_id").size()
    lines += ["## 調教師の拠点（栗東/美浦）", f"- 判定できた調教師: {len(home)} 人（出走のある調教師 {d['trainer_id'].nunique()} 人中）。栗東 {int((home == 'KU').sum())}・美浦 {int((home == 'MI').sum())}",
              f"- 純度（多数派の割合）: 中央値 {pur.median():.3f}、0.9 未満 {int((pur < 0.9).sum())} 人（うち追い切り 50 本以上 {int(((pur < 0.9) & (cnt >= 50)).sum())} 人）",
              f"- 拠点が判定できない調教師の出走数: {int((~d['trainer_id'].isin(home.index)).sum()):,} 行 / {len(d):,} 行", ""]
    d["home"] = d["trainer_id"].map(home)
    d["year"] = pd.to_datetime(d["date"]).dt.year
    W = L[(L["days_before"] >= 1) & (L["days_before"] <= 28)].copy()
    W["_k"] = W["horse_id"] + "|" + W["race_date"]
    d["_k"] = d["horse_id"] + "|" + d["date"].astype(str).str.slice(0, 10)
    d["cov28"] = d["_k"].isin(set(W["_k"]))
    cov = d.groupby("year")["cov28"].mean()
    lines += ["## レース前 28 日の追い切りの被覆（出走行のうち 1 本以上ある割合）", "| 年 | 被覆 |", "|---|---|"] + [f"| {y} | {v:.1%} |" for y, v in cov.items()] + [""]
    d["venue_code"] = d["venue_code"].astype(str).str.zfill(2)
    km = [tt.KM_HOME.get((h, v), np.nan) for h, v in zip(d["home"], d["venue_code"])]
    d["tier"] = [tt.tier_of_km(k, v) if np.isfinite(k) else -1 for k, v in zip(km, d["venue_code"])]
    tab = d.groupby("tier").size()
    lines += ["## 輸送区分（0 ホーム / 1 近距離 / 2 中〜長距離 / 3 遠方・北海道、-1 不明）の出走数", "| 区分 | 出走数 | 割合 |", "|---|---|---|"] + [f"| {k} | {v:,} | {v / len(d):.1%} |" for k, v in tab.items()] + [""]
    route = d.assign(route=d["home"].fillna("UNK") + ">" + d["venue_code"].map(tt.VENUES).fillna("?")).groupby("route").size().sort_values(ascending=False).head(24)
    lines += ["## 主な輸送ルート（拠点>開催場）", "| ルート | 出走数 |", "|---|---|"] + [f"| {k} | {v:,} |" for k, v in route.items()] + [""]
    # 滞在（レース前 28 日に拠点以外で追い切り）
    Wm = W.merge(d[["_k", "home", "venue_code"]].drop_duplicates("_k"), on="_k")
    Wm["away"] = (Wm["loc"] != Wm["home"]) & Wm["loc"].isin(list(tt.CENTER_LOCS) + list(tt.VENUES))
    stay = Wm.groupby("_k")["away"].any()
    d["stay"] = d["_k"].map(stay)
    sv = d[d["cov28"]].groupby("venue_code")["stay"].mean()
    lines += ["## 滞在（拠点以外での追い切りが 28 日以内にある）の割合（開催場別。追い切りの被覆がある行のみ）", "| 開催場 | 割合 |", "|---|---|"] + [f"| {tt.VENUES.get(k, k)} | {v:.1%} |" for k, v in sv.items()] + [""]
    lines += ["## 判定不可の部分", "- 追い切りが無い/読めない行（被覆の裏）は滞在を判定不可（NaN、被覆フラグ tt_cov28 を持たせる）。",
              "- 地方競馬場・不明なコース名は拠点外の滞在としては数えない（NAR/UNK は away に含めない）。",
              "- 調教師の拠点は追い切りの場所から推定（所属の欄は保存されていない）。純度が低い調教師・出走が少ない調教師は不明のまま。",
              "- 輸送の実際の経路（陸送/空路/船）・所要時間・当日の到着時刻は取得できない。道のりは座標から概算。", ""]
    Path("/tmp/tp33_data_audit.md").write_text("\n".join(lines), encoding="utf-8")
    _log("監査を /tmp/tp33_data_audit.md に保存")


# ───────────────────────── 特徴量群
def cmd_groups(a) -> None:
    d = _table()
    parts = tt.build(d, _laps())
    gdir = OUT / "groups"
    gdir.mkdir(parents=True, exist_ok=True)
    for g, G in parts.items():
        T = pd.concat([d[KEY].reset_index(drop=True), G.reset_index(drop=True)], axis=1)
        T.to_parquet(gdir / f"{g}.parquet", index=False)
        _log(f"{g}: {G.shape[1]} 列、欠損率 {float(G.isna().mean().mean()):.3f}")


def _load_group(g: str, df: pd.DataFrame) -> pd.DataFrame:
    T = pd.read_parquet(OUT / "groups" / f"{g}.parquet")
    T["race_id"], T["horse_id"] = T["race_id"].astype(str), T["horse_id"].astype(str)
    m = df[KEY].merge(T, on=KEY, how="left")
    m = m[[c for c in m.columns if c not in KEY]]
    m.index = df.index
    for c in m.columns:
        if str(m[c].dtype) in ("object", "string"):
            m[c] = m[c].astype("category")
    return m


# ───────────────────────── 記述分析
def _boot(res: np.ndarray, rid: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """レース単位のブートストラップで平均の 95% 区間（クラスター頑健）。"""
    codes, uniq = pd.factorize(rid)
    s = np.bincount(codes, weights=res, minlength=len(uniq))
    c = np.bincount(codes, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    m = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
    return float(res.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cmd_descr(a) -> None:
    d = _table()
    G = pd.concat([pd.read_parquet(OUT / "groups" / f"{g}.parquet").drop(columns=KEY) for g in GROUPS], axis=1)
    d = pd.concat([d, G], axis=1)
    ens = pd.read_parquet(V / "tp27/ens/v4_e2.parquet")
    ens["race_id"], ens["horse_id"] = ens["race_id"].astype(str), ens["horse_id"].astype(str)
    m = d.merge(ens[KEY + ["p_win", "p_top2", "p_top3"]], on=KEY, how="inner").reset_index(drop=True)
    fp = m["finish_position"]
    m["win"], m["top2"], m["top3"] = (fp == 1).astype(float), (fp <= 2).astype(float), (fp <= 3).astype(float)
    rows = []

    def row(label: str, mask: pd.Series, base: pd.Series | None = None) -> dict:
        s = m[mask]
        r = {"区分": label, "出走数": int(len(s))}
        for k in ("win", "top3"):
            r[f"{k}率"] = float(s[k].mean())
            r[f"予測{k}"] = float(s[f"p_{k}"].mean())
            mean, lo, hi = _boot((s[k] - s[f"p_{k}"]).to_numpy(), s["race_id"].to_numpy())
            r[f"{k}残差"], r[f"{k}残差_lo"], r[f"{k}残差_hi"] = mean, lo, hi
        return r

    for t_ in (0, 1, 2, 3):
        rows.append(row(f"輸送区分 {t_}", m["tt_tier"] == t_))
    for h in (0.0, 1.0):
        for t_ in (0, 1, 2, 3):
            rows.append(row(f"{'栗東' if h else '美浦'}所属 × 区分 {t_}", (m["tt_home_ku"] == h) & (m["tt_tier"] == t_)))
    rows.append(row("滞在あり（拠点以外で追い切り）", m["tt_stay_flag"] == 1))
    rows.append(row("滞在なし（追い切りの記録あり）", (m["tt_stay_flag"] == 0) & (m["tt_cov28"] == 1)))
    for lab, mk in (("遠征（区分2以上）× 滞在あり", (m["tt_far"] == 1) & (m["tt_stay_flag"] == 1)), ("遠征（区分2以上）× 滞在なし", (m["tt_far"] == 1) & (m["tt_stay_flag"] == 0)),
                    ("ホーム/近距離 × 滞在あり", (m["tt_far"] == 0) & (m["tt_stay_flag"] == 1)), ("ホーム/近距離 × 滞在なし", (m["tt_far"] == 0) & (m["tt_stay_flag"] == 0))):
        rows.append(row(lab, mk))
    rows.append(row("初めての遠征（区分2以上）", m["tt_first_far"] == 1))
    rows.append(row("遠征の2戦目以降", (m["tt_far"] == 1) & (m["tt_first_far"] == 0)))
    for k in (1, 2, 3):
        rows.append(row(f"遠征の連続 {k}{'+' if k == 3 else ''} 走目", (m["tt_far"] == 1) & ((m["tt_far_streak"] >= k) if k == 3 else (m["tt_far_streak"] == k - 1))))
    rows.append(row("同じ開催場の連続（前走も同じ場）", m["tt_prev_same_venue"] == 1))
    rows.append(row("前走と別の開催場への移動", m["tt_prev_same_venue"] == 0))
    rows.append(row("そのコースが初めて", m["tt_first_at_venue"] == 1))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "descr_table.csv", index=False)
    # 年別の再現性（残差 2022〜24。生の好走率の差は全年）
    ys = []
    for y in (2022, 2023, 2024):
        s = m[m["year"] == y]
        for lab, mk in (("遠征（区分2以上）", s["tt_far"] == 1), ("滞在あり", s["tt_stay_flag"] == 1), ("初めての遠征", s["tt_first_far"] == 1)):
            ss = s[mk]
            mean, lo, hi = _boot((ss["top3"] - ss["p_top3"]).to_numpy(), ss["race_id"].to_numpy(), 500)
            ys.append({"区分": lab, "年": y, "出走数": int(len(ss)), "top3残差": mean, "lo": lo, "hi": hi})
    pd.DataFrame(ys).to_csv(OUT / "descr_by_year.csv", index=False)
    # 生の好走率の年別（2020〜24）: 同じ場×面×格で層別した 遠征(区分2以上) − ホーム/近距離 の差
    allr = d.copy()
    fp = allr["finish_position"]
    allr["top3"] = (fp <= 3).astype(float)
    allr["strat"] = allr["venue_code"].astype(str) + "|" + allr["surface"].astype(str) + "|" + allr["grade"].astype(str)
    br = []
    for y, g in allr[allr["tt_far"].notna()].groupby("year"):
        diffs, ws = [], []
        for _, s in g.groupby("strat"):
            a1, a0 = s[s["tt_far"] == 1]["top3"], s[s["tt_far"] == 0]["top3"]
            if len(a1) >= 20 and len(a0) >= 20:
                diffs.append(a1.mean() - a0.mean())
                ws.append(min(len(a1), len(a0)))
        if diffs:
            br.append({"年": int(y), "層の数": len(diffs), "遠征−ホーム/近距離の複勝率差（層別・加重）": float(np.average(diffs, weights=ws))})
    pd.DataFrame(br).to_csv(OUT / "descr_raw_by_year.csv", index=False)
    # 交絡を制御した回帰（複勝 ~ 遠征 + 滞在 + 初遠征 + 予測logit + 場 + クラス、レース単位のクラスター頑健）
    try:
        import statsmodels.formula.api as smf

        mm = m.dropna(subset=["tt_far", "tt_stay_flag"]).copy()
        mm["logit_p3"] = np.log(mm["p_top3"].clip(1e-4, 1 - 1e-4) / (1 - mm["p_top3"].clip(1e-4, 1 - 1e-4)))
        mm["grade"] = mm["grade"].astype(str)
        fit = smf.logit("top3 ~ tt_far + tt_stay_flag + tt_first_far + logit_p3 + C(venue_code) + C(grade)", data=mm).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(mm["race_id"])[0]})
        co = pd.DataFrame({"coef": fit.params, "se": fit.bse, "p": fit.pvalues}).loc[["tt_far", "tt_stay_flag", "tt_first_far", "logit_p3"]]
        co.to_csv(OUT / "descr_logit.csv")
        _log("回帰: " + co.round(4).to_string())
    except Exception as e:  # noqa: BLE001
        _log(f"回帰失敗: {e}")
    _log(R[["区分", "出走数", "top3率", "予測top3", "top3残差", "top3残差_lo", "top3残差_hi"]].round(4).to_string())


# ───────────────────────── 評価
def _frame(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.read_parquet(V / "tp31/v4_frame.parquet")
    X.index = df.index
    for c in X.columns:
        if str(X[c].dtype) in ("object", "string"):
            X[c] = X[c].astype("category")
    return X


def _run_subset(name, X, meta, folds, lp, years, seeds):
    from src.pipeline.models import zoo_runner

    fs = [f for f in folds if int(meta["year"].to_numpy()[f.valid_index][0]) in years]
    _wait()
    t0 = time.time()
    res, oof = zoo_runner.run_oof(name, "lgb_binary", X, meta, fs, seeds, THREADS, params=dict(lp))
    lv = {m: float(res[m].mean()) for m in ("logloss_win", "logloss_top2", "logloss_top3")}
    _log(f"{name}: win {lv['logloss_win']:.5f} top2 {lv['logloss_top2']:.5f} top3 {lv['logloss_top3']:.5f} cols={X.shape[1]} ({time.time() - t0:.0f}s)")
    return lv, oof


def _stage(label: str, years: list[int], seeds: list[int], groups: list[str]) -> None:
    df, base, meta, folds, lp = t.setup()
    fin_map = meta.set_index(KEY)["finish_position"]
    X0 = _frame(df)
    d = OUT / label
    d.mkdir(parents=True, exist_ok=True)
    resp = d / "results.json"
    res = _jload(resp)
    bp = d / "base.parquet"
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
        names = [x for x in groups if x not in ("ALL_PASSED", "TT_all")] if g == "ALL_PASSED" else (GROUPS if g == "TT_all" else [g])
        G = pd.concat([_load_group(x, df) for x in names], axis=1)
        Xn = pd.concat([X0, G], axis=1)
        lv, oof = _run_subset(f"V4+{g}", Xn, meta, folds, lp, years, seeds)
        c = t._cmp(oof0, oof, fin_map, g)
        steps[g] = {"cols": int(G.shape[1]), "levels": lv, "vs_V4": c}
        _jsave(resp, res)
        _log(f"  {g}: Δwin {c['Δwin']:+.5f} (hi {c['win_hi']:+.5f}) Δtop2 {c['Δtop2']:+.5f} Δtop3 {c['Δtop3']:+.5f} 採用={c['採用']}")


def cmd_stage1(a) -> None:
    _stage("stage1", [2022, 2023], [0], GROUPS + ["TT_all"])


def cmd_stage2(a) -> None:
    r = _jload(OUT / "stage1/results.json")
    passed = [g for g, s in r.get("groups", {}).items() if g != "TT_all" and s["vs_V4"]["Δwin"] < -0.00012 and s["vs_V4"]["win_hi"] < 0.0001]
    _log(f"第 1 段を通過: {passed}")
    cand = passed + (["ALL_PASSED"] if len(passed) > 1 else []) + ["TT_all"]
    _stage("stage2", [2024], [2, 3], cand)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["audit", "groups", "descr", "stage1", "stage2"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"audit": cmd_audit, "groups": cmd_groups, "descr": cmd_descr, "stage1": cmd_stage1, "stage2": cmd_stage2}[a.cmd](a)


if __name__ == "__main__":
    main()
