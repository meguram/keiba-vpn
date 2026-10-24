"""騎手の乗り替わり（前走騎手の格 → 今走騎手の格）に、期待値（回収率・モデル予測との差）が取れる型があるかを調べる。

python -m src.scripts.data.jockey_switch_ev build      # tp-59 の master.parquet から as-of の特徴を作る（<JSW_OUT>/frame.parquet）
python -m src.scripts.data.jockey_switch_ev analyze    # 型ごとの集計・検定（<JSW_OUT>/result.json）
python -m src.scripts.data.jockey_switch_ev report     # 章 HTML（docs/html/modeling/chapters/jockey_switch_ev.html）

事前に固定した設計（結果を見てから増やさない）:
- 有力ジョッキー = その日より前の直近 365 日（当日を含まない）の平地の勝利数の上位 N 人（同数は騎乗数、次に ID で決める）。N = 10 が主、5 と 20 は感度。
  回収率は定義に使わない。jockey_id をキーにする。見習い減量・外国人・短期免許は ID では区別できないため、勝利数の順位に任せる（短期免許は 365 日の勝利数が少なく、ほぼ非有力に入る）。
- 前走騎手の格も『今日時点の』順位で測る（両方を同じ物差しにする。tp-32 の jk_rate_prev と同じ考え方）。
- 型: 継続 / 非有力→有力(NE) / 有力→非有力(EN) / 有力→有力(EE) / 非有力→非有力(NN)。前走が不明（2020 年以前の履歴）は除く。
- 仮説の族 K = 型 4 + NE・EN の層別 17 ずつ = 38 セル × 買い方 2（単勝・複勝）= 76。層別は事前に列挙した範囲だけ。
- 選択期間 2021〜2024（2020 年は 365 日の窓が埋まらないので助走）、test 2025 は最良の数セルの評価 1 回、2026 は記述統計のみ。
- 回収率は 1 出走 100 円・レース単位（同じレースの馬をまとめて 1 単位）。市場オッズは払戻の計算と比較にだけ使う（特徴量にしない）。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.scripts.data import tp59_lib as L

ROOT = Path(__file__).resolve().parents[3]
TP59_OUT = Path(os.environ.get("TP59_OUT", "/home/jovyan/work/tp59_out"))
OUT = Path(os.environ.get("JSW_OUT", "/home/jovyan/work/jsw_out"))
TOPS = (5, 10, 20)
PRIMARY = 10
SEL_YEARS = (2021, 2022, 2023, 2024)
WINDOW_DAYS = 365


# ------------------------------------------------------------------ 特徴の作成（as-of）
def elite_flags(d: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """日 × 騎手の『その日より前の直近 365 日の平地勝利数が上位 N 人か』。戻り値: {N: bool の DataFrame（index=日, columns=jockey_id）}。"""
    flat = d[d["surface"] != "障"]
    daily = flat.groupby(["date", "jockey_id"]).agg(w=("win", "sum"), n=("win", "size")).reset_index()
    days = pd.date_range(d["date"].min(), d["date"].max(), freq="D")
    w = daily.pivot(index="date", columns="jockey_id", values="w").reindex(days).fillna(0.0)
    n = daily.pivot(index="date", columns="jockey_id", values="n").reindex(days).fillna(0.0)
    cw, cn = w.cumsum(), n.cumsum()
    # 当日を含まない直近 365 日 = [D-365, D-1]
    w365 = cw.shift(1).fillna(0.0) - cw.shift(1 + WINDOW_DAYS).fillna(0.0)
    n365 = cn.shift(1).fillna(0.0) - cn.shift(1 + WINDOW_DAYS).fillna(0.0)
    ids = w365.columns.to_numpy()
    key = w365.to_numpy() * 1e7 + n365.to_numpy()          # 勝利数 → 騎乗数 の順
    order = np.argsort(-key, axis=1, kind="stable")        # 同点は列順（= ID 順）
    rank = np.empty_like(order)
    rows = np.arange(key.shape[0])[:, None]
    rank[rows, order] = np.arange(key.shape[1])[None, :]
    out = {}
    for top in TOPS:
        out[top] = pd.DataFrame(rank < top, index=days, columns=ids)
    return out


def trainer_main_jockey(d: pd.DataFrame) -> pd.DataFrame:
    """月初の時点（前日まで）の直近 365 日で、その調教師の馬に最も多く騎乗した騎手（主戦）。index=(trainer_id, 月初)。"""
    flat = d[d["surface"] != "障"][["date", "trainer_id", "jockey_id"]].dropna()
    months = pd.date_range(d["date"].min().replace(day=1), d["date"].max(), freq="MS")
    rows = []
    for m in months:
        h = flat[(flat["date"] < m) & (flat["date"] >= m - pd.Timedelta(days=WINDOW_DAYS))]
        if h.empty:
            continue
        c = h.groupby(["trainer_id", "jockey_id"]).size().reset_index(name="n").sort_values(["trainer_id", "n", "jockey_id"], ascending=[True, False, True])
        top = c.drop_duplicates("trainer_id")[["trainer_id", "jockey_id"]].rename(columns={"jockey_id": "main_jockey"})
        top["month"] = m
        rows.append(top)
    return pd.concat(rows, ignore_index=True)


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(TP59_OUT / "master.parquet")
    d["jockey_id"] = d["jockey_id"].astype("string")
    d["prev_jockey_id"] = d["prev_jockey_id"].astype("string")
    d = d.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)

    ef = elite_flags(d)
    first_valid = d["date"].min() + pd.Timedelta(days=WINDOW_DAYS)
    d["rank_valid"] = d["date"] >= first_valid
    for top, mat in ef.items():
        def look(col: str) -> np.ndarray:
            col_idx = mat.columns.get_indexer(d[col].astype(object))
            row_idx = mat.index.get_indexer(d["date"])
            arr = mat.to_numpy()
            ok = (col_idx >= 0) & (row_idx >= 0)
            res = np.full(len(d), np.nan)
            res[ok] = arr[row_idx[ok], col_idx[ok]].astype(float)
            return res

        d[f"el{top}_now"] = look("jockey_id")
        d[f"el{top}_prev"] = look("prev_jockey_id")
        # 勝利数の履歴が無い（その日までに騎乗が無い）騎手は非有力として扱う
        d.loc[d["jockey_id"].notna() & d[f"el{top}_now"].isna(), f"el{top}_now"] = 0.0
        d.loc[d["prev_jockey_id"].notna() & d[f"el{top}_prev"].isna(), f"el{top}_prev"] = 0.0

    has_prev = d["prev_jockey_id"].notna() & (d["run_no"] > 0) & d["rank_valid"] & (d["surface"] != "障")
    d["known"] = has_prev
    d["same_j"] = has_prev & (d["jockey_id"] == d["prev_jockey_id"])
    for top in TOPS:
        e0, e1 = d[f"el{top}_prev"], d[f"el{top}_now"]
        t = np.where(~has_prev, "NA", np.where(d["same_j"], "SAME", np.where((e0 == 0) & (e1 == 1), "NE", np.where((e0 == 1) & (e1 == 0), "EN", np.where((e0 == 1) & (e1 == 1), "EE", "NN")))))
        d[f"type{top}"] = t

    # テン乗り: その馬に、この騎手が過去（データ内）に乗ったことがあるか
    d = d.sort_values(["horse_id", "date", "race_id"], kind="stable")
    d["pair_n"] = d.groupby(["horse_id", "jockey_id"], sort=False).cumcount()
    d["ten_nori"] = (d["pair_n"] == 0).astype(float)
    # 見習い減量の代理: 同じ性齢の最頻の斤量より 2kg 以上軽い（平地、頭数 8 以上）。印が無いデータなので代理
    mode_w = d.groupby(["race_id", "sex_age"])["jockey_weight"].transform(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan)
    d["app_proxy"] = ((mode_w - d["jockey_weight"] >= 2.0) & (d["field_size"] >= 8) & (d["surface"] != "障")).astype(float)
    d["prev_app"] = d.groupby("horse_id", sort=False)["app_proxy"].shift(1)
    # 主戦への乗り替わり
    mj = trainer_main_jockey(d)
    d["month"] = d["date"].values.astype("datetime64[M]").astype("datetime64[ns]")
    d = d.merge(mj.rename(columns={"month": "month"}), on=["trainer_id", "month"], how="left")
    d["to_main"] = ((d["jockey_id"] == d["main_jockey"]) & d["main_jockey"].notna()).astype(float)
    d = d.sort_values(["date", "race_id", "horse_number"], kind="stable").reset_index(drop=True)

    d = L.add_excess(d)
    keep = ["race_id", "date", "year", "horse_id", "horse_number", "jockey_id", "prev_jockey_id", "trainer_id", "surface", "cls", "dist_band", "popularity",
            "prev_fin", "win", "top2", "top3", "win_odds", "ret_win", "ret_place", "ex_win", "ex_place", "p_win", "p_top2", "p_top3", "q_win",
            "known", "same_j", "ten_nori", "app_proxy", "prev_app", "to_main"] + [f"type{t}" for t in TOPS] + \
           [f"el{t}_{s}" for t in TOPS for s in ("now", "prev")]
    d[keep].to_parquet(OUT / "frame.parquet", index=False)
    k = d[d["known"]]
    print("rows", len(d), "known", len(k))
    print(k.groupby("year")["type10"].value_counts().unstack().fillna(0).astype(int).to_string())


# ------------------------------------------------------------------ 集計
def cells(d: pd.DataFrame) -> dict[str, np.ndarray]:
    """事前に列挙したセル（主定義 N=10）。NE / EN のみ層別する。"""
    t = d["type10"].to_numpy()
    out: dict[str, np.ndarray] = {f"type:{k}": t == k for k in ("NE", "EN", "EE", "NN")}
    pf = d["prev_fin"].to_numpy(float)
    pop = d["popularity"].to_numpy(float)
    strata = {
        "ten_nori": d["ten_nori"].to_numpy() == 1, "modori": d["ten_nori"].to_numpy() == 0,
        "prev_good(1-3)": (pf >= 1) & (pf <= 3), "prev_mid(4-9)": (pf >= 4) & (pf <= 9), "prev_bad(10+)": pf >= 10,
        "pop_1-3": (pop >= 1) & (pop <= 3), "pop_4-6": (pop >= 4) & (pop <= 6), "pop_7+": pop >= 7,
        "turf": d["surface"].to_numpy() == "芝", "dirt": d["surface"].to_numpy() == "ダート",
        "maiden": d["cls"].isin(["c0_newcomer", "c1_maiden"]).to_numpy(), "win1+": ~d["cls"].isin(["c0_newcomer", "c1_maiden"]).to_numpy(),
        "dist_short": (d["dist_band"] == "d_sprint").to_numpy(), "dist_mile": d["dist_band"].isin(["d_mile_s", "d_mile"]).to_numpy(),
        "dist_mid+": d["dist_band"].isin(["d_mid", "d_long"]).to_numpy(),
        "to_stable_main": d["to_main"].to_numpy() == 1,
        "app_to_regular": (d["prev_app"].to_numpy() == 1) & (d["app_proxy"].to_numpy() == 0),
    }
    for ty in ("NE", "EN"):
        for nm, m in strata.items():
            out[f"{ty}|{nm}"] = (t == ty) & m
    return out


def cell_stats(d: pd.DataFrame, m: np.ndarray, seed: int = 0) -> dict:
    s = d[m]
    if len(s) == 0:
        return {"n": 0}
    race = s["race_id"].to_numpy()
    win = L.boot_roi(race, s["ret_win"].to_numpy(), 1000, seed)
    plc = L.boot_roi(race, s["ret_place"].to_numpy(), 1000, seed)
    res = {"n": int(len(s)), "races": win["races"],
           "win_rate": float(s["win"].mean()), "top2_rate": float(s["top2"].mean()), "top3_rate": float(s["top3"].mean()),
           "q_win_mean": float(s["q_win"].mean()),
           "roi_win": win["roi"], "roi_win_lo": win["lo"], "roi_win_hi": win["hi"],
           "roi_place": plc["roi"], "roi_place_lo": plc["lo"], "roi_place_hi": plc["hi"],
           "ex_win": float(s["ex_win"].mean()), "ex_place": float(s["ex_place"].mean())}
    p = s[s["p_win"].notna()]
    if len(p):
        r2, (c, ow, pw, o3, p3) = L.race_sums(p["race_id"].to_numpy(), np.ones(len(p)), p["win"], p["p_win"], p["top3"], p["p_top3"])
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(c), size=(1000, len(c)))
        rw = ow[idx].sum(1) / np.maximum(pw[idx].sum(1), 1e-9)
        r3 = o3[idx].sum(1) / np.maximum(p3[idx].sum(1), 1e-9)
        res.update({"n_pred": int(len(p)), "win_ratio": float(ow.sum() / pw.sum()), "win_ratio_lo": float(np.quantile(rw, 0.025)), "win_ratio_hi": float(np.quantile(rw, 0.975)),
                    "top3_ratio": float(o3.sum() / p3.sum()), "top3_ratio_lo": float(np.quantile(r3, 0.025)), "top3_ratio_hi": float(np.quantile(r3, 0.975)),
                    "win_minus_pred_pt": float(100 * (ow.sum() - pw.sum()) / c.sum())})
    return res


def race_profit(d: pd.DataFrame, m: np.ndarray, col: str, races: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """レースごとの利益（円、100 円/件）と、買ったレースか。買っていないレースの利益は 0。"""
    s = d[m]
    prof = s[col].to_numpy(float) - 100.0
    r, (tot, cnt) = L.race_sums(s["race_id"].to_numpy(), prof, np.ones(len(s)))
    tot_s = pd.Series(tot, index=r).reindex(races).fillna(0.0).to_numpy()
    cnt_s = pd.Series(cnt, index=r).reindex(races).fillna(0.0).to_numpy()
    return tot_s, cnt_s > 0


def t_test_p(x: np.ndarray, bought: np.ndarray) -> tuple[float, float]:
    """買ったレースだけで、レース単位の利益の平均が 0 より大きいか（片側）。"""
    x = x[bought]
    if len(x) < 3 or x.std(ddof=1) < 1e-9:
        return float("nan"), 1.0
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    return float(t), float(1 - norm.cdf(t))


def holm(p: np.ndarray) -> np.ndarray:
    k = len(p)
    order = np.argsort(p)
    adj = np.empty(k)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (k - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def analyze() -> None:
    d = pd.read_parquet(OUT / "frame.parquet")
    d = d[d["known"]].copy()
    res: dict = {"meta": {"primary_top": PRIMARY, "sel_years": list(SEL_YEARS)}}
    C = cells(d)
    names = list(C)
    sel = d["year"].isin(SEL_YEARS).to_numpy()

    # 型の分布（参考）
    res["type_counts"] = {str(y): d[d["year"] == y]["type10"].value_counts().to_dict() for y in sorted(d["year"].unique())}
    res["baseline"] = {"all": {str(y): cell_stats(d[d["year"] == y], np.ones((d["year"] == y).sum(), bool)) for y in sorted(d["year"].unique())},
                       "same_j": {p: cell_stats(d[(d["year"].isin(ys))], (d[d["year"].isin(ys)]["type10"] == "SAME").to_numpy()) for p, ys in (("sel", SEL_YEARS), ("2025", (2025,)), ("2026", (2026,)))}}

    # 3 期間 × セルの集計
    periods = {"sel": sel, "2025": (d["year"] == 2025).to_numpy(), "2026": (d["year"] == 2026).to_numpy()}
    res["cells"] = {nm: {p: cell_stats(d[pm], m[pm]) for p, pm in periods.items()} for nm, m in C.items()}

    # 選択期間の検定（買い方 2 × セル 38 = 76）。絶対の回収率 > 100% と、オッズ帯の期待より上（超過 > 0）
    ds = d[sel]
    races = np.array(sorted(ds["race_id"].unique()))
    tests = []
    prof_mat = []
    for nm in names:
        ms = C[nm][sel]
        for bet, col, exc in (("win", "ret_win", "ex_win"), ("place", "ret_place", "ex_place")):
            x, bought = race_profit(ds, ms, col, races)
            t, p = t_test_p(x, bought)
            xe, _ = race_profit(ds.assign(_e=ds[exc] + 100.0), ms, "_e", races)     # 超過払戻（オッズ帯の期待払戻を引いた）。+100 -100 で同じ関数を使う
            te, pe = t_test_p(xe, bought)
            tests.append({"cell": nm, "bet": bet, "n": int(ms.sum()), "t_abs": t, "p_abs": p, "t_exc": te, "p_exc": pe})
            prof_mat.append(x)
    P = np.array([t["p_abs"] for t in tests])
    PE = np.array([t["p_exc"] for t in tests])
    for t, a, b in zip(tests, holm(P), holm(PE)):
        t["holm_abs"], t["holm_exc"] = float(a), float(b)
    res["tests"] = tests
    M = np.array(prof_mat).T
    rc = L.reality_check(M, 2000, 0)
    res["reality_check_abs"] = {k: (float(v) if not isinstance(v, np.ndarray) else None) for k, v in rc.items()}
    res["reality_check_abs"]["best_cell"] = f"{tests[rc['best']]['cell']}:{tests[rc['best']]['bet']}"
    res["n_hypotheses"] = len(tests)
    res["min_holm_abs"] = float(min(t["holm_abs"] for t in tests))
    res["min_holm_exc"] = float(min(t["holm_exc"] for t in tests))
    res["best5"] = [{"cell": tests[i]["cell"], "bet": tests[i]["bet"], "t": tests[i]["t_abs"]} for i in np.argsort(-np.nan_to_num(np.array([t["t_abs"] for t in tests]), nan=-9))[:5]]

    # 感度: 有力の定義 N = 5, 20（主要 4 型のみ。選択期間・2025・2026）
    sens = {}
    for top in (5, 10, 20):
        tt = d[f"type{top}"].to_numpy()
        sens[str(top)] = {ty: {p: cell_stats(d[pm], (tt == ty)[pm]) for p, pm in periods.items()} for ty in ("NE", "EN", "EE", "NN", "SAME")}
    res["sensitivity"] = sens

    res["note"] = "best5 = 選択期間の t 値の上位 5（76 本の最大）。2025/2026 の列は同じセルをそのまま当てた評価（選択に使っていない）"
    (OUT / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("n_hypotheses", "min_holm_abs", "min_holm_exc", "reality_check_abs", "best5")}, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "analyze", "report"])
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    elif a.cmd == "analyze":
        analyze()
    else:
        from src.scripts.data import jockey_switch_ev_report as R

        R.main(json.loads((OUT / "result.json").read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
