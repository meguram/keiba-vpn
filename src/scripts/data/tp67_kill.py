"""tp-67（改）消印＝『3着以内に来る確率が低いと十分な確信を持って言える馬』。人気区分別の目標を、期間別に Clopper-Pearson の上側信頼限界で判定する。

目標（ユーザー確定 2026-10-21）: 1〜5 人気 ≤10%、6〜9 人気は付けない、10 人気以下 ≤5%（実績の 3 着内率）。
python -m src.scripts.data.tp67_kill prep   --frame frame_all.parquet --rank-dir DIR --out DIR      # 評価用の表（確定着順の 3 着内、8 頭以上）
python -m src.scripts.data.tp67_kill calib  --base base.parquet --out DIR                          # V4 OOF の低確率域の較正（人気区分別）
python -m src.scripts.data.tp67_kill rule   --base base.parquet --out DIR                          # しきい値の探索（設計年）と期間別の判定
python -m src.scripts.data.tp67_kill top5   --base base.parquet --train-table train_table.parquet --out DIR   # 1〜5 人気で 10% 以下を示せる集合があるか
python -m src.scripts.data.tp67_kill overlap --base base.parquet --out DIR                         # 他の印（◎○▲★△）との重なり

入力は tp-36 の frame_all（V4 の OOF 予測 p_top3、確定の単勝オッズ・人気）と rank_tbl の確定着順。人気・オッズは対象の区分け（選別）にだけ使い、
予測モデルの入力にはしない。3 着内は確定着順 ≤3。8 頭未満のレースは除く。2026 年は一度だけの最終確認。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

CLASSES = {"p1_5": (1, 5), "p6_9": (6, 9), "p10p": (10, 99)}
TARGETS = {"p1_5": 0.10, "p6_9": None, "p10p": 0.05}
PERIODS = {"設計2022-23": [2022, 2023], "確認2024": [2024], "参考2025": [2025], "最終2026": [2026]}
P3_BINS = [0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 1.0]
MIN_FIELD = 8


def cp_upper(h: int, n: int, alpha: float = 0.05) -> float:
    """二項分布の Clopper-Pearson 上側（両側 1-alpha の上限）。n=0 は 1.0。"""
    if n <= 0:
        return 1.0
    return float(beta.ppf(1 - alpha / 2, h + 1, n - h)) if n > h else 1.0


def prep(args) -> None:
    d = pd.read_parquet(args.frame)
    d["race_id"] = d["race_id"].astype(str)
    d["horse_id"] = d["horse_id"].astype(str)
    rk = pd.concat([pd.read_parquet(Path(args.rank_dir) / f"rank{y}.parquet") for y in (2022, 2023, 2024, 2025, 2026)])
    rk = rk.assign(race_id=rk["race_id"].astype(str), horse_id=rk["horse_id"].astype(str))[["race_id", "horse_id", "rank"]].drop_duplicates(["race_id", "horse_id"])
    m = d.merge(rk, on=["race_id", "horse_id"], how="left")
    bad = m.groupby("race_id")["rank"].apply(lambda s: s.isna().any())
    m = m[~m["race_id"].map(bad)].copy()               # 着順が取れない馬のいるレース（2026 年に約 5%）は除く
    m["hit3"] = ((m["rank"] > 0) & (m["rank"] <= 3)).astype(int)
    m["n"] = m.groupby("race_id")["horse_id"].transform("size")
    m["pop"] = m["popularity"].where(m["popularity"].notna(), m.groupby("race_id")["win_odds"].rank(method="first"))
    m["cls"] = pd.cut(m["pop"], [0, 5, 9, 99], labels=list(CLASSES))
    m = m[(m["n"] >= MIN_FIELD) & m["win_odds"].notna() & m["p_top3"].notna()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m.to_parquet(out / "base.parquet")
    print(m.shape, m.groupby("year")["race_id"].nunique().to_dict())


def calib(args) -> None:
    m = pd.read_parquet(args.base)
    m["pb"] = pd.cut(m["p_top3"], P3_BINS, include_lowest=True)
    res: dict = {}
    for c in CLASSES:
        g = m[m["cls"] == c].groupby("pb", observed=True).agg(n=("hit3", "size"), hit=("hit3", "sum"), rate=("hit3", "mean"), pred=("p_top3", "mean")).reset_index()
        g["cp_up"] = [cp_upper(int(h), int(n)) for h, n in zip(g["hit"], g["n"])]
        g["pb"] = g["pb"].astype(str)
        res[c] = g.to_dict("records")
        print(c, "\n", g.round(4).to_string())
    (Path(args.out) / "tp67_calib.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def _period_stats(x: pd.DataFrame, n_races: int) -> dict:
    n = len(x)
    h = int(x["hit3"].sum()) if n else 0
    return {"n": n, "hit": h, "rate": (h / n) if n else None, "cp_up": cp_upper(h, n), "races_with": int(x["race_id"].nunique()),
            "race_share": (x["race_id"].nunique() / n_races) if n_races else None, "per_race": (n / n_races) if n_races else None}


def rule(args) -> None:
    m = pd.read_parquet(args.base)
    nr = {k: m[m["year"].isin(v)]["race_id"].nunique() for k, v in PERIODS.items()}
    res: dict = {"targets": TARGETS, "periods": PERIODS, "n_races": nr, "classes": {}}
    for c, tgt in TARGETS.items():
        sub = m[m["cls"] == c]
        res["classes"][c] = {"base": {k: _period_stats(sub[sub["year"].isin(v)], nr[k]) for k, v in PERIODS.items()}, "thr": {}}
        if tgt is None:
            continue
        for t in (0.02, 0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20, 1.0):
            res["classes"][c]["thr"][str(t)] = {k: _period_stats(sub[sub["year"].isin(v) & (sub["p_top3"] <= t)], nr[k]) for k, v in PERIODS.items()}
    (Path(args.out) / "tp67_rule.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for c, tgt in TARGETS.items():
        if tgt is None:
            continue
        print("==", c, "目標", tgt)
        for t, per in res["classes"][c]["thr"].items():
            print(f"  p3<={t}: " + " | ".join(f"{k} n={s['n']} {((s['rate'] or 0) * 100):.2f}% 上限{s['cp_up'] * 100:.2f}% {((s['race_share'] or 0) * 100):.0f}%R" for k, s in per.items()))


def top5(args) -> None:
    """1〜5 人気で、3 着内率が 10% 以下だと言える集合があるか: V4 の複勝確率だけ／消し要因の信号を足した分類器／単独の信号の両端。"""
    import lightgbm as lgb
    from sklearn.metrics import log_loss, roc_auc_score

    b = pd.read_parquet(args.base)
    b["horse_id"] = b["horse_id"].astype(str)
    t = pd.read_parquet(args.train_table)
    t["race_id"] = t["race_id"].astype(str)
    t["horse_id"] = t["horse_id"].astype(str)
    drop = {"finish_position", "jockey_id", "trainer_id"}
    num = [c for c in t.columns if c not in drop and c not in b.columns and pd.api.types.is_numeric_dtype(t[c])]
    for c in ("surface", "venue", "track_condition", "grade"):
        t[c + "_c"] = t[c].astype("category").cat.codes
    cat = [c + "_c" for c in ("surface", "venue", "track_condition", "grade")]
    m = b.merge(t[["race_id", "horse_id"] + num + cat], on=["race_id", "horse_id"], how="inner")
    pb_cols = ["pb_ben_m3", "pb_ben_1", "pb_exp_rel", "pb_pos_sd5"]
    if args.pb:
        # tp-64（過去走の展開利）の最小 4 列を、消し専用の分類器の入力に足す
        pb = pd.read_parquet(args.pb, columns=["race_id", "horse_id"] + pb_cols)
        pb = pb.assign(race_id=pb["race_id"].astype(str), horse_id=pb["horse_id"].astype(str))
        m = m.merge(pb, on=["race_id", "horse_id"], how="left")
        num = num + pb_cols
    m["lg"] = np.log(m["p_top3"].clip(1e-4, 1 - 1e-4) / (1 - m["p_top3"].clip(1e-4, 1 - 1e-4)))
    feats = [c for c in num + cat if c != "has_speed_index"] + ["lg"]
    for c in feats:
        m[c] = pd.to_numeric(m[c], errors="coerce").astype("float32")
    s = m[(m["cls"] == "p1_5") & (m["year"] <= 2025)].copy()
    prm = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
               lambda_l2=10, verbose=-1, num_threads=3, seed=1)

    def fit(tr):
        return lgb.train(prm, lgb.Dataset(tr[feats], tr["hit3"]), num_boost_round=300)

    s["kp"] = np.nan
    for ty, tr in ((2022, [2023]), (2023, [2022]), (2024, [2022, 2023]), (2025, [2022, 2023, 2024])):
        s.loc[s["year"] == ty, "kp"] = fit(s[s["year"].isin(tr)]).predict(s.loc[s["year"] == ty, feats])
    res: dict = {"metrics": {}, "buckets": [], "single_factor_min": [], "with_pb": bool(args.pb)}
    for y in (2022, 2023, 2024, 2025):
        q = s[s["year"] == y]
        res["metrics"][str(y)] = {"auc_p3": float(roc_auc_score(q["hit3"], q["p_top3"])), "auc_clf": float(roc_auc_score(q["hit3"], q["kp"])),
                                  "ll_p3": float(log_loss(q["hit3"], q["p_top3"].clip(1e-4, 1 - 1e-4))), "ll_clf": float(log_loss(q["hit3"], q["kp"].clip(1e-4, 1 - 1e-4)))}
    s["kb"] = pd.qcut(s["kp"], [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 0.6, 1.0], duplicates="drop")
    g = s.groupby("kb", observed=True).agg(n=("hit3", "size"), hit=("hit3", "sum"), rate=("hit3", "mean"), kp=("kp", "mean"), p3=("p_top3", "mean")).reset_index()
    g["cp_up"] = [cp_upper(int(h), int(n)) for h, n in zip(g["hit"], g["n"])]
    g["kb"] = g["kb"].astype(str)
    res["buckets"] = g.to_dict("records")
    d = s[s["year"] <= 2023]
    for col in feats:
        x = pd.to_numeric(d[col], errors="coerce")
        if x.notna().sum() < 5000 or x.nunique() < 4:
            continue
        for side, q in (("low", 0.03), ("high", 0.97)):
            thr = float(x.quantile(q))
            sel = d[(x <= thr) if side == "low" else (x >= thr)]
            if len(sel) >= 150:
                res["single_factor_min"].append({"feature": col, "side": side, "thr": thr, "n": len(sel), "rate": float(sel["hit3"].mean())})
    res["single_factor_min"].sort(key=lambda r: r["rate"])
    res["single_factor_min"] = res["single_factor_min"][:10]
    low = s[s["p_top3"] <= 0.12]
    res["p3_le_0.12"] = {str(y): {"n": int((low["year"] == y).sum()), "rate": float(low[low["year"] == y]["hit3"].mean())} for y in (2022, 2023, 2024, 2025)}
    (Path(args.out) / ("tp67_top5_pb.json" if args.pb else "tp67_top5.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res["metrics"], ensure_ascii=False, indent=1))
    print(g.round(3).to_string())
    print(res["single_factor_min"][:5])


def overlap(args) -> None:
    """採用ルール（10 人気以下 かつ 複勝確率 ≤ しきい値）の馬が、◎○▲★△のどれかと重なっていないか（assign_mece_marks）。"""
    from src.utils.race_probabilities import assign_mece_marks

    m = pd.read_parquet(args.base)
    thr = float(args.thr)
    m = m[m["year"] <= 2025]
    cnt = {"kill": 0, "honmei": 0, "pair": 0, "show_val": 0, "star": 0, "anchor": 0, "none": 0}
    by_year: dict = {}
    for rid, g in m.groupby("race_id", sort=False):
        g = g.reset_index(drop=True)
        es = []
        for i in range(len(g)):
            e = {"horse_number": int(g.loc[i, "horse_number"]), "win_prob": float(g.loc[i, "p_win"]), "top2_prob": float(g.loc[i, "p_top2"]), "top3_prob": float(g.loc[i, "p_top3"]),
                 "win_odds": float(g.loc[i, "win_odds"])}
            e["ev_win"] = e["win_prob"] * e["win_odds"]
            pm = g.loc[i, "place_mid"]
            if np.isfinite(pm):
                e["ev_place"] = e["top3_prob"] * float(pm)
            es.append(e)
        assign_mece_marks(es)
        for i, e in enumerate(es):
            if g.loc[i, "cls"] == "p10p" and e["top3_prob"] <= thr:
                cnt["kill"] += 1
                cnt[e["mark_type"]] += 1
                y = int(g.loc[i, "year"])
                by_year.setdefault(y, {"kill": 0, "marked": 0})
                by_year[y]["kill"] += 1
                by_year[y]["marked"] += int(e["mark_type"] != "none")
    (Path(args.out) / "tp67_overlap.json").write_text(json.dumps({"thr": thr, "counts": cnt, "by_year": by_year}, ensure_ascii=False, indent=1))
    print(cnt, by_year)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prep")
    a.add_argument("--frame", required=True)
    a.add_argument("--rank-dir", required=True)
    a.add_argument("--out", required=True)
    for name in ("calib", "rule", "top5", "overlap"):
        p = sub.add_parser(name)
        p.add_argument("--base", required=True)
        p.add_argument("--out", required=True)
        if name == "top5":
            p.add_argument("--train-table", required=True)
            p.add_argument("--pb", default=None, help="tp-64 の pb_features.parquet（pb_ben_m3 / pb_ben_1 / pb_exp_rel / pb_pos_sd5 を入力に足す）")
        if name == "overlap":
            p.add_argument("--thr", default="0.07")
    args = ap.parse_args()
    {"prep": prep, "calib": calib, "rule": rule, "top5": top5, "overlap": overlap}[args.cmd](args)


if __name__ == "__main__":
    main()
