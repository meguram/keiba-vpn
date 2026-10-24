"""tp-72 第 C 部: 穴馬が好走するときの共通要素（case-control）。

python -m src.scripts.data.tp72_cc run [--out DIR]

穴馬 = 確定の人気 9 番以下 / 単勝 12 倍以上 / 20 倍以上（3 通りを並べる）。好走 = 3 着以内 / 1 着のみ。
対照 = 同じ年・同じ単勝オッズ帯・同じ頭数の、好走しなかった穴馬（その平均好走率との差で見る。初回実行は頭数を揃えておらず、少頭数が好走に見える対照の取り違えだったため直した）。
記述子は tp72_desc.py の 39 個（事前に固定）。各（仕様 × 記述子 × 水準）で、その水準にいる穴馬の「好走 − 同年・同オッズ帯の平均」のレース単位クラスタ z を片側で検定し、BH。
軽量な多変量ロジスティック回帰（L2）を併記する。市場オッズ・人気は穴馬の定義と層別にだけ使い、記述子には入れない。
発見 2022〜2024、確認 2025 を 1 回（q<0.10 と上位）、2026 は記述統計のみ。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.scripts.data.tp72_desc import descriptors
from src.scripts.data.tp72_eval import CONF, DESC, DISC, OUT, bh, boot_roi, cluster_z, phi, prepare

MIN_N = 200
UNIVERSES = {"pop>=9": lambda d: d["popularity"] >= 9, "odds>=12": lambda d: d["win_odds"] >= 12, "odds>=20": lambda d: d["win_odds"] >= 20}
OUTCOMES = {"top3": "y3", "win": "y1"}


def wide_roi(d: pd.DataFrame, mask: np.ndarray, payoff: pd.DataFrame, partner: str) -> list[float]:
    """該当の穴馬を軸に、相手 1 頭（partner = 'fav' 1 番人気 / 'model' モデルの勝率最大の別の馬）とのワイド 1 点買いの回収率（レース単位ブートストラップ区間つき）。"""
    sub = d[mask].copy()
    if sub.empty:
        return [float("nan")] * 3
    key = d[["race_id", "horse_number", "popularity", "p_win"]].copy()
    if partner == "fav":
        fav = key[key["popularity"] == 1].drop_duplicates("race_id").set_index("race_id")["horse_number"]
    else:
        fav = key.sort_values("p_win", ascending=False).drop_duplicates("race_id").set_index("race_id")["horse_number"]
    sub["partner"] = sub["race_id"].map(fav)
    sub = sub[sub["partner"].notna() & (sub["partner"] != sub["horse_number"])]
    if sub.empty:
        return [float("nan")] * 3
    a = np.minimum(sub["horse_number"].astype(int), sub["partner"].astype(int))
    b = np.maximum(sub["horse_number"].astype(int), sub["partner"].astype(int))
    sub["combo"] = a.astype(str).str.zfill(2) + "-" + b.astype(str).str.zfill(2)
    pw = payoff[payoff["bet_type"] == "wide"].copy()
    pw["combo"] = pw["combo"].astype(str).str.split("-").map(lambda x: f"{int(x[0]):02d}-{int(x[1]):02d}" if len(x) == 2 else "")
    pm = pw.drop_duplicates(["race_id", "combo"]).set_index(["race_id", "combo"])["payout"]
    idx = pd.MultiIndex.from_arrays([sub["race_id"], sub["combo"]])
    ret = pm.reindex(idx).fillna(0.0).to_numpy() / 100.0
    rc = pd.factorize(sub["race_id"])[0]
    nr = rc.max() + 1
    s = np.bincount(rc, weights=ret, minlength=nr)
    c = np.bincount(rc, minlength=nr).astype(float)
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(1000):
        i = rng.integers(0, nr, nr)
        vals.append(s[i].sum() / c[i].sum())
    return [float(s.sum() / c.sum()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def run(out_dir: Path) -> dict:
    d = prepare()
    Dm, L = descriptors(d)
    nrace = int(d["rc"].max()) + 1
    yr = d["year"].to_numpy()
    tests = []
    for un, uf in UNIVERSES.items():
        um = uf(d).to_numpy()
        for on, oc in OUTCOMES.items():
            base = d.loc[um].groupby(["year", "odds_band", "fs_c"])[oc].transform("mean")      # 同じ年・同オッズ帯・同頭数の穴馬の平均好走率（頭数を揃えないと少頭数が好走に見える）
            e = pd.Series(0.0, index=d.index)
            e.loc[um] = d.loc[um, oc] - base
            d["_e"] = e
            for k in Dm.columns:
                for lv, lab in enumerate(L[k]):
                    lm = um & (Dm[k].to_numpy() == lv)
                    row = {"spec": f"{un}|{on}", "desc": k, "level": lab, "lv": lv}
                    for tag, sel in (("disc", np.isin(yr, DISC)), ("y2022", yr == 2022), ("y2023", yr == 2023), ("y2024", yr == 2024), ("y2025", yr == CONF), ("y2026", yr == DESC)):
                        mm = lm & sel
                        n = int(mm.sum())
                        z, tot, _ = cluster_z(d, mm, "_e", nrace) if n else (float("nan"), 0.0, 0)
                        row[tag] = {"n": n, "z": z, "lift_pt": (tot / n * 100) if n else float("nan"),
                                    "hit": float(d[oc].to_numpy()[mm].mean()) if n else float("nan")}
                    ok = row["disc"]["n"] >= MIN_N and not np.isnan(row["disc"]["z"])
                    row["p"] = float(1 - phi(row["disc"]["z"])) if ok else 1.0
                    tests.append(row)
    ps = np.array([t["p"] for t in tests])
    q = bh(ps)
    for t, qq in zip(tests, q):
        t["q"] = float(qq)
    zs = np.array([t["disc"]["z"] for t in tests if t["disc"]["n"] >= MIN_N and not np.isnan(t["disc"]["z"])])
    summary = {"n_tests": len(tests), "n_testable": int(len(zs)), "z_median": float(np.median(zs)), "z_mad_sd": float(1.4826 * np.median(np.abs(zs - np.median(zs)))),
               "n_z_gt_1.645": int((zs > 1.645).sum()), "expected_false_pos": float(0.05 * len(zs)), "n_q_lt_0.10": int((q < 0.10).sum())}
    # 確認と回収率: q<0.10 と、z の上位 20（仕様ごとではなく全体）
    top = sorted([t for t in tests if t["p"] < 1.0], key=lambda t: -t["disc"]["z"])[:20]
    chosen = {id(t): t for t in top + [t for t in tests if t["q"] < 0.10]}.values()
    payoff = pd.read_parquet(Path(os.environ.get("TP72_PAYOFF", "data/local/modeling/betting/payoff.parquet")))
    payoff["race_id"] = payoff["race_id"].astype(str)
    dd = d.assign(_vi=0)
    for t in chosen:
        un, on = t["spec"].split("|")
        um = UNIVERSES[un](d).to_numpy()
        lm = um & (Dm[t["desc"]].to_numpy() == t["lv"])
        t["roi"] = {}
        for tag, sel in (("disc", np.isin(yr, DISC)), ("y2025", yr == CONF), ("y2026", yr == DESC)):
            mm = lm & sel
            t["roi"][tag] = {"win": boot_roi(d, mm, "r1"), "place": boot_roi(d, mm, "r3"),
                             "wide_fav": wide_roi(dd, mm, payoff, "fav"), "wide_model": wide_roi(dd, mm, payoff, "model")}
    # 軽量な多変量ロジスティック（発見期間・穴馬 odds>=12・3 着以内）
    logit = {}
    try:
        from sklearn.linear_model import LogisticRegression

        um = (d["win_odds"] >= 12).to_numpy() & np.isin(yr, DISC)
        cols = []
        X = []
        names = []
        for k in Dm.columns:
            for lv, lab in enumerate(L[k]):
                col = (Dm[k].to_numpy() == lv)[um].astype(np.float32)
                if col.sum() >= MIN_N:
                    X.append(col)
                    names.append(f"{k}={lab}")
        ob = pd.get_dummies(d.loc[um, "odds_band"].astype(int), prefix="ob").to_numpy(np.float32)
        Xm = np.column_stack(X + [ob])
        lr = LogisticRegression(C=0.05, max_iter=300)
        lr.fit(Xm, d.loc[um, "y3"].to_numpy())
        coef = lr.coef_[0][: len(names)]
        order = np.argsort(-coef)
        logit = {"n": int(um.sum()), "top_pos": [(names[i], float(np.exp(coef[i]))) for i in order[:20]], "top_neg": [(names[i], float(np.exp(coef[i]))) for i in order[::-1][:10]]}
    except Exception as exc:  # noqa: BLE001
        logit = {"error": str(exc)}
    out = {"summary": summary, "tests": tests, "chosen_n": len(list(chosen)), "logit": logit}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cc.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    r = run(Path(a.out))
    print(json.dumps(r["summary"], ensure_ascii=False, indent=1))
