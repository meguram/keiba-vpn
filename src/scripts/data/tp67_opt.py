"""tp-67（最適化）10 人気以下の消印のしきい値と区分の最適化。

python -m src.scripts.data.tp67_opt grid  --base base.parquet [--train-table train_table.parquet] --out DIR   # しきい値・区切り・頭数別・相対順位・勝率/連対率・分類器・条件別
python -m src.scripts.data.tp67_opt bets  --base base.parquet --payoff payoff.parquet --thr 0.08 --out DIR      # 買い目（3連複・ワイド）から消印馬を外した場合の点数・的中率・回収率
python -m src.scripts.data.tp67_opt recheck --results-dir data/local/prediction_results --out DIR            # 運用で貯まった予測成績（印別の消）から、現行の設定で目標 5% を満たすかを再確認（V5 切替後など）

制約: 消印の馬の 3 着内率の Clopper-Pearson 95% 上側限界 ≤5%（各期間）。選択は設計年（2022〜23）のみ、確認 2024・参考 2025・最終 2026 は検証だけ。
評価の対象は prep（tp67_kill）が作る base.parquet（8 頭以上、確定着順の 3 着内）。人気・オッズは対象の区分けにだけ使い、予測の入力にしない。
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data.tp67_kill import PERIODS, cp_upper

CONSTRAINT = 0.05
INTERNAL = 0.03       # 内部の基準（設計年の上側限界 ≤3%）。ユーザーが最初に挙げた値で、目標 5% に余裕を持たせる


def stats(sub: pd.DataFrame, rest_pool: pd.DataFrame, n_races: int) -> dict:
    """sub = 消印の馬。rest_pool = 同じ区分で消印にならなかった馬（印なしの『残り』）。"""
    n = len(sub)
    h = int(sub["hit3"].sum()) if n else 0
    rn = len(rest_pool)
    rh = int(rest_pool["hit3"].sum()) if rn else 0
    return {"n": n, "hit": h, "rate": (h / n) if n else None, "cp_up": cp_upper(h, n), "per_race": n / n_races if n_races else None,
            "race_share": sub["race_id"].nunique() / n_races if n_races else None, "hits_per_race": h / n_races if n_races else None,
            "top3_removed_share": h / (3 * n_races) if n_races else None, "rest_rate": (rh / rn) if rn else None}


def eval_mask(c: pd.DataFrame, mask: pd.Series, nr: dict) -> dict:
    out = {}
    for k, yrs in PERIODS.items():
        yy = c["year"].isin(yrs)
        out[k] = stats(c[yy & mask], c[yy & ~mask], nr[k])
        out[k]["base_rate"] = float(c[yy]["hit3"].mean())
    return out


def lift(r: dict) -> float:
    return r["rate"] / r["base_rate"] if r["rate"] is not None and r["base_rate"] else float("nan")


def design_ok(res: dict, bound: float = INTERNAL) -> bool:
    return res["設計2022-23"]["cp_up"] <= bound


def grid(args) -> None:
    m = pd.read_parquet(args.base)
    c = m[m["cls"] == "p10p"].copy()
    nr = {k: m[m["year"].isin(v)]["race_id"].nunique() for k, v in PERIODS.items()}
    c["rank_in_race"] = c.groupby("race_id")["p_top3"].rank(method="first")          # 10 人気以下の中で複勝確率が低い順
    c["size_grp"] = np.where(c["n"] >= 16, "16+", "<=15")
    ts = [round(x, 2) for x in np.arange(0.03, 0.151, 0.01)]
    res: dict = {"nr": nr, "constraint": CONSTRAINT, "internal": INTERNAL, "families": {}}
    count = 0

    def add(fam, key, mask):
        nonlocal count
        count += 1
        res["families"].setdefault(fam, {})[key] = eval_mask(c, mask, nr)

    for t in ts:
        add("A_p3", f"p3<={t}", c["p_top3"] <= t)
    for p0 in (11, 12, 13, 14):
        for t in ts:
            add("B_pop_cut", f"pop>={p0}&p3<={t}", (c["pop"] >= p0) & (c["p_top3"] <= t))
    for grp in ("16+", "<=15"):
        for t in ts:
            add("C_field_size", f"{grp}&p3<={t}", (c["size_grp"] == grp) & (c["p_top3"] <= t))
    for k in range(1, 8):
        for t in (0.10, 0.15):
            add("D_rank_k", f"bottom{k}&p3<={t}", (c["rank_in_race"] <= k) & (c["p_top3"] <= t))
    for w in (0.005, 0.01, 0.015, 0.02, 0.03):
        add("E_pwin", f"pwin<={w}", c["p_win"] <= w)
    for q in (0.02, 0.03, 0.04, 0.05, 0.07):
        add("E_ptop2", f"ptop2<={q}", c["p_top2"] <= q)
    for w, t in ((0.01, 0.08), (0.015, 0.10), (0.02, 0.12)):
        add("E_combo", f"pwin<={w}|p3<={t}", (c["p_win"] <= w) | (c["p_top3"] <= t))
    res["n_configs"] = count

    # 条件別（train_table がある 2022〜25 年のみ。2026 年の条件は未取得なので最終確認は不可）
    if args.train_table:
        t0 = pd.read_parquet(args.train_table, columns=["race_id", "horse_id", "surface", "distance", "race_class", "grade", "track_condition"])
        t0["race_id"] = t0["race_id"].astype(str)
        t0["horse_id"] = t0["horse_id"].astype(str)
        cc = c.assign(horse_id=c["horse_id"].astype(str)).merge(t0, on=["race_id", "horse_id"], how="inner")
        res["cond"] = {}
        cc["dist_grp"] = pd.cut(cc["distance"], [0, 1400, 1800, 2200, 9999], labels=["短(~1400)", "マイル(~1800)", "中(~2200)", "長(2201~)"])
        for name, col in (("surface", "surface"), ("dist_grp", "dist_grp"), ("track_condition", "track_condition")):
            for gval, g in cc.groupby(col, observed=True):
                for t in (0.05, 0.08, 0.10, 0.12):
                    mk = g["p_top3"] <= t
                    res["cond"][f"{name}={gval}&p3<={t}"] = {k: stats(g[g["year"].isin(v) & mk], g[g["year"].isin(v) & ~mk], g[g["year"].isin(v)]["race_id"].nunique()) for k, v in PERIODS.items() if k != "最終2026"}
    # 分類器（消し専用）。設計は 2022↔2023 の相互、2024←2022〜23、2025←2022〜24。特徴量の無い 2026 年は評価できない。
    if args.train_table:
        res["clf"] = classifier(c, args.train_table, ts)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "tp67_opt_grid.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("configs", count)
    for key, r in res["families"]["A_p3"].items():
        print(key, {k: (round(v["rate"] * 100, 2), round(v["cp_up"] * 100, 2), round(v["per_race"], 2), round(v["race_share"] * 100)) for k, v in r.items()})


def classifier(c: pd.DataFrame, train_table: str, ts) -> dict:
    import lightgbm as lgb

    t = pd.read_parquet(train_table)
    t["race_id"] = t["race_id"].astype(str)
    t["horse_id"] = t["horse_id"].astype(str)
    drop = {"finish_position", "jockey_id", "trainer_id"}
    num = [x for x in t.columns if x not in drop and x not in c.columns and pd.api.types.is_numeric_dtype(t[x])]
    for x in ("surface", "venue", "track_condition", "grade"):
        t[x + "_c"] = t[x].astype("category").cat.codes
    cat = [x + "_c" for x in ("surface", "venue", "track_condition", "grade")]
    m = c.assign(horse_id=c["horse_id"].astype(str)).merge(t[["race_id", "horse_id"] + num + cat], on=["race_id", "horse_id"], how="inner")
    m["lg"] = np.log(m["p_top3"].clip(1e-4, 1 - 1e-4) / (1 - m["p_top3"].clip(1e-4, 1 - 1e-4)))
    feats = [x for x in num + cat if x != "has_speed_index"] + ["lg"]
    for x in feats:
        m[x] = pd.to_numeric(m[x], errors="coerce").astype("float32")
    prm = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
               verbose=-1, num_threads=3, seed=1)
    m["kp"] = np.nan
    for ty, tr in ((2022, [2023]), (2023, [2022]), (2024, [2022, 2023]), (2025, [2022, 2023, 2024])):
        mdl = lgb.train(prm, lgb.Dataset(m.loc[m["year"].isin(tr), feats], m.loc[m["year"].isin(tr), "hit3"]), num_boost_round=200)
        m.loc[m["year"] == ty, "kp"] = mdl.predict(m.loc[m["year"] == ty, feats])
    out: dict = {}
    nr = {k: m[m["year"].isin(v)]["race_id"].nunique() for k, v in PERIODS.items() if k != "最終2026"}
    for k in (0.03, 0.04, 0.05, 0.06, 0.08, 0.10):
        mk = m["kp"] <= k
        out[f"kp<={k}"] = {p: stats(m[m["year"].isin(v) & mk], m[m["year"].isin(v) & ~mk], nr[p]) for p, v in PERIODS.items() if p != "最終2026"}
        for p, v in PERIODS.items():
            if p != "最終2026":
                out[f"kp<={k}"][p]["base_rate"] = float(m[m["year"].isin(v)]["hit3"].mean())
    return out


def bets(args) -> None:
    """買い目から消印馬を外した場合: ◎軸（p_win 最大）の 3 連複流し／ワイド流し／◎○ 2 頭軸の 3 連複流し。相手 = ◎(○)以外の全馬（同じ人気区分に限らない）から、消印の馬を外す。"""
    m = pd.read_parquet(args.base)
    m = m[m["year"] <= 2025].copy()
    pay = pd.read_parquet(args.payoff)
    pay["race_id"] = pay["race_id"].astype(str)
    bad = set(pay.loc[pay["ambiguous"] | ~pay["parsed"], "race_id"])
    pay = pay[~pay["race_id"].isin(bad) & pay["bet_type"].isin(["sanrenpuku", "wide"])]
    trio: dict = {}
    wide: dict = {}
    for r in pay.itertuples(index=False):
        key = frozenset(int(x) for x in str(r.combo).split("-"))
        (trio if r.bet_type == "sanrenpuku" else wide).setdefault(r.race_id, {})[key] = float(r.payout)
    thr = float(args.thr)
    agg: dict = {}
    for rid, g in m.groupby("race_id", sort=False):
        if rid not in trio or rid not in wide:
            continue
        g = g.sort_values("p_win", ascending=False)
        hn = g["horse_number"].astype(int).tolist()
        kill = set(g.loc[(g["cls"] == "p10p") & (g["p_top3"] <= thr), "horse_number"].astype(int))
        axis, second = hn[0], hn[1]
        for name, opp0, ax in (("sanpuku_a1", [h for h in hn if h != axis], (axis,)), ("sanpuku_a2", [h for h in hn if h not in (axis, second)], (axis, second)),
                               ("wide_a1", [h for h in hn if h != axis], (axis,))):
            for variant, opp in (("keep", opp0), ("drop", [h for h in opp0 if h not in kill])):
                if name.startswith("sanpuku_a1"):
                    combos = [frozenset((axis, a, b)) for a, b in combinations(opp, 2)]
                    book = trio[rid]
                elif name == "sanpuku_a2":
                    combos = [frozenset((axis, second, a)) for a in opp]
                    book = trio[rid]
                else:
                    combos = [frozenset((axis, a)) for a in opp]
                    book = wide[rid]
                pts = len(combos)
                ret = sum(book.get(cmb, 0.0) for cmb in combos)
                hit = float(any(cmb in book for cmb in combos))
                d = agg.setdefault((name, variant, int(g["year"].iloc[0])), {"races": 0, "points": 0, "stake": 0.0, "ret": 0.0, "hit": 0.0})
                d["races"] += 1
                d["points"] += pts
                d["stake"] += pts * 100.0
                d["ret"] += ret
                d["hit"] += hit
    out: dict = {"thr": thr, "rows": []}
    for name in ("sanpuku_a1", "sanpuku_a2", "wide_a1"):
        for y in (2022, 2023, 2024, 2025):
            k, d_ = agg[(name, "keep", y)], agg[(name, "drop", y)]
            out["rows"].append({"scheme": name, "year": y, "races": k["races"], "points_keep": k["points"] / k["races"], "points_drop": d_["points"] / d_["races"],
                                "points_cut": 1 - d_["points"] / k["points"], "hit_keep": k["hit"] / k["races"], "hit_drop": d_["hit"] / d_["races"],
                                "roi_keep": k["ret"] / k["stake"], "roi_drop": d_["ret"] / d_["stake"]})
    (Path(args.out) / "tp67_opt_bets.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for r in out["rows"]:
        print(r["scheme"], r["year"], f"点数 {r['points_keep']:.1f}→{r['points_drop']:.1f} (-{r['points_cut'] * 100:.1f}%) 的中 {r['hit_keep'] * 100:.1f}→{r['hit_drop'] * 100:.1f}% 回収 {r['roi_keep'] * 100:.1f}→{r['roi_drop'] * 100:.1f}%")


def recheck(args) -> None:
    """運用で貯まった予測成績（data/local/prediction_results/<日>.json の summary.marks.kesi）から、消印の 3 着内率と上側限界を再計算する。
    V5 の確率に切り替わった後（shadow の蓄積や、予測の作り直し後の日次集計）に、現行の設定（config/kesi_mark.json）で目標 5% を満たすかを見る。
    月別と全体、モデル別（その日の models）を出す。件数が少ない間は上側限界が広い（判定は件数が貯まってから）。"""
    from src.utils.race_probabilities import kesi_config

    cfg = kesi_config()
    rows = []
    for f in sorted(Path(args.results_dir).glob("[0-9]" * 8 + ".json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        k = ((d.get("summary") or {}).get("marks") or {}).get("kesi")
        if not k or not k.get("n"):
            continue
        rows.append({"date": f.stem, "month": f.stem[:6], "models": ",".join(sorted(map(str, (d.get("models") or {}).keys()))) if isinstance(d.get("models"), dict) else str(d.get("models")),
                     "n": int(k["n"]), "hit": int(k["f1"] + k["f2"] + k["f3"])})
    df = pd.DataFrame(rows)
    out: dict = {"config": {k: cfg.get(k) for k in ("version", "min_field", "min_pop", "p3_max")}, "target": float((cfg.get("target_upper") or CONSTRAINT))}
    if len(df):
        def agg(g):
            n, h = int(g["n"].sum()), int(g["hit"].sum())
            return {"n": n, "hit": h, "rate": h / n if n else None, "cp_up": cp_upper(h, n), "ok": cp_upper(h, n) <= out["target"]}
        out["all"] = agg(df)
        out["by_month"] = {m: agg(g) for m, g in df.groupby("month")}
        out["by_models"] = {m: agg(g) for m, g in df.groupby("models")}
    (Path(args.out) / "tp67_opt_recheck.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid")
    g.add_argument("--base", required=True)
    g.add_argument("--train-table", default=None)
    g.add_argument("--out", required=True)
    b = sub.add_parser("bets")
    b.add_argument("--base", required=True)
    b.add_argument("--payoff", required=True)
    b.add_argument("--thr", default="0.08")
    b.add_argument("--out", required=True)
    r = sub.add_parser("recheck")
    r.add_argument("--results-dir", required=True)
    r.add_argument("--out", required=True)
    args = ap.parse_args()
    {"grid": grid, "bets": bets, "recheck": recheck}[args.cmd](args)


if __name__ == "__main__":
    main()
