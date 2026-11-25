"""めぐサイクルの最後の 1 回だけの確かめ（事前登録: config/megu_cycles/final_prereg.json）。

- final: 目標を満たした 5 候補を、凍結した設定のまま 2025 年で 1 回だけ数える（台帳 eval_only・test の期間）。
  確率は凍結と同じ作り（2016〜2024 年で学習した学習の平均、v5・v7）。判定のオッズは凍結と同じ再現（t60・t15・t05）。
- drift: 本物の発走前オッズ（2026-10-18〜11-20）で、再現の楽観（確定の順位でずれを割り当てる持ち込み）を市場の値だけで測る
  （台帳 eval_only・封印の期間）。候補の確率は使わない。

  KEIBA_SESSION=megu-cycles python -m src.research.megu_cycles.mc_final drift
  KEIBA_SESSION=megu-cycles python -m src.research.megu_cycles.mc_final final
費用 0 円（手元の表だけ）。重いので nice -n 19。
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from src.research.megu_cycles import cycles as C
from src.research.megu_cycles import mc_data as D
from src.research.megu_cycles import mc_eval as E

PREREG = D.ROOT / "config/megu_cycles/final_prereg.json"
OUT_JSON = D.ROOT / "docs/operations/megu_cycles/final_once.json"
DRIFT_JSON = D.ROOT / "docs/operations/megu_cycles/drift_check.json"
FINAL_YEAR = 2025
DRIFT_END = "2026-11-20"


def _frozen() -> list:
    P = json.loads(PREREG.read_text(encoding="utf-8"))
    return [PREREG, D.GOAL, D.ROOT / "src/research/megu_cycles/mc_final.py", D.ROOT / "src/research/megu_cycles/cycles.py",
            D.ROOT / "src/research/megu_cycles/mc_data.py", D.ROOT / "src/research/megu_cycles/mc_eval.py"] + \
        [C.CAND / f"{n}.json" for n in P["candidates"]]


# ---------------------------------------------------------------- 2025 年の表と確率（guard の中だけで呼ぶ）
def frame_2025() -> pd.DataFrame:
    cols = sorted(set(D.KEEP) | set(D.feature_columns()))
    d = pd.read_parquet(D.FRAME, columns=cols, filters=[("year", "==", FINAL_YEAR)])
    assert (pd.to_numeric(d["year"]) == FINAL_YEAR).all()
    d = d.sort_values(["race_id", "hn"]).reset_index(drop=True)
    from src.research.course.v2 import attach_final_odds
    d = attach_final_odds(d, [FINAL_YEAR])
    d = d[d["odds_final"] > 0].reset_index(drop=True)
    inv = 1.0 / d["odds_final"].to_numpy(float)
    d["q_final"] = inv / pd.Series(inv).groupby(d["race_id"].to_numpy()).transform("sum").to_numpy()
    d["graded"] = pd.to_numeric(d["cls"], errors="coerce").isin(D.GRADED_CLS)
    d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    return d


def bag_2025(version: str) -> pd.DataFrame:
    """凍結と同じ学習の平均（BAG_SETS・XGB_ROUNDS・RICH_VERSIONS[version]）を 2016〜2024 年で学び、2025 年を予測する。"""
    import re

    import pyarrow.parquet as pq
    import xgboost as xgb
    yr_all = pd.to_numeric(pq.read_table(D.META_RICH, columns=["year"]).column("year").to_pandas()).to_numpy()
    meta = pd.read_parquet(D.META_RICH, filters=[("year", "<=", FINAL_YEAR)])
    keep = yr_all <= FINAL_YEAR
    assert len(meta) == keep.sum()
    names = [c for c in pq.ParquetFile(D.X_RICH).schema_arrow.names
             if c not in D.RICH_EXCLUDE and not re.search(D.RICH_VERSIONS[version], c)
             and not re.search(r"odds|pop|ninki|tansho|fukusho|payoff|market|ret_", c, re.I)]
    X = pd.read_parquet(D.X_RICH, columns=names)
    assert len(X) == len(yr_all)
    assert (meta["race_id"].astype(str).to_numpy() == pd.read_parquet(D.META_RICH, columns=["race_id"])["race_id"].astype(str).to_numpy()[keep]).all()
    X = X[keep].reset_index(drop=True)
    for c in X.columns:
        if isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].cat.codes.where(X[c].notna(), np.nan)
        elif not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.astype(np.float32)
    yr = pd.to_numeric(meta["year"]).to_numpy(int)
    y = (pd.to_numeric(meta["finish_position"], errors="coerce").to_numpy() == 1).astype(int)
    tr, te = yr <= 2024, yr == FINAL_YEAR
    P = np.zeros(te.sum())
    for bs in D.BAG_SETS:
        m = xgb.train(dict(D.XGB_PARAMS, **bs), xgb.DMatrix(X[tr], label=y[tr]), D.XGB_ROUNDS)
        P += m.predict(xgb.DMatrix(X[te])) / len(D.BAG_SETS)
    out = meta.loc[te, ["race_id", "horse_id"]].astype(str).reset_index(drop=True)
    out["pr_win"] = P
    return out


def table_for(d: pd.DataFrame, bag: pd.DataFrame, phase: str) -> pd.DataFrame:
    e = d.merge(bag.drop_duplicates(["race_id", "horse_id"]), on=["race_id", "horse_id"], how="left")
    e = e.sort_values(["race_id", "hn"]).reset_index(drop=True)
    e["p_rich"] = E.race_norm(e, e["pr_win"].fillna(1e-4).to_numpy(float))
    pre = D.simulate_pre(e, phase=phase)
    for k in range(D.N_PRE):
        e[f"odds_pre{k}"] = pre[k]
    return e


def cmd_final() -> None:
    from src.ops import data_ledger as DL
    P = json.loads(PREREG.read_text(encoding="utf-8"))
    t0 = time.time()
    with DL.guard(tool="src/research/megu_cycles/mc_final.py:final", family="megu_cycles", version="final-once-2025",
                  period=tuple(P["period_eval"]), purpose="eval_only", frozen_paths=_frozen(),
                  reason="50 サイクルの後の最後の 1 回だけの確かめ（事前登録 final_prereg.json）。結果で候補・設定を変えない") as tok:
        d = frame_2025()
        tok.check_dates(d["date"].astype(str))
        bags = {v: bag_2025(v) for v in ("v5", "v7")}
        res, daily = {}, {}
        prev = {r["spec"]["name"]: r["confirm"]["roi"] for r in E.confirm_log()}
        for name in P["candidates"]:
            spec = json.loads((C.CAND / f"{name}.json").read_text(encoding="utf-8"))
            ver = "v7" if "v7" in spec["probability"]["file"] else "v5"
            e = table_for(d, bags[ver], spec.get("phase", "t60"))
            if spec.get("group"):
                e = e[C.group_of(e) == spec["group"]].reset_index(drop=True)
            e = e[E.seg_mask(e, spec["segment"])].reset_index(drop=True)
            ab = tuple(spec["combo_ab_t60"])
            st = C.strat_ev(lambda x, k, ab=ab: C.combo_p(x, k, *ab, col="p_rich"), spec["t"], spec["omax"])
            r = E.evaluate(e, st, "win")
            r["smartrc_2025"] = C.smartrc_check(e, st, "win", year=FINAL_YEAR)
            r["roi_2022_24"] = prev.get(name)
            r["pass"] = bool(r["roi"] is not None and r["roi"] >= 1.0 and prev.get(name) and r["roi"] >= 0.9 * prev[name])
            res[name] = r
            S = np.vstack([st(e, k) for k in range(D.N_PRE)]).mean(0)
            daily[name] = pd.DataFrame({"date": e["date"].astype(str).str[:10], "s": S, "p": S * e["ret_tan"].fillna(0).to_numpy()}).groupby("date").sum()
            print(name, C.short(r), "合格" if r["pass"] else "不合格", flush=True)
        tok.finish(result_path=OUT_JSON)
    names = list(daily)
    dates = sorted(set().union(*[set(g.index) for g in daily.values()]))
    A = np.stack([daily[n].reindex(dates).fillna(0)[["s", "p"]].to_numpy() for n in names])
    roi = A[:, :, 1].sum(1) / A[:, :, 0].sum(1)
    rng = np.random.default_rng(20261125)
    B = A[:, rng.integers(0, len(dates), size=(4000, len(dates))), :].sum(2)
    rb = B[:, :, 1] / B[:, :, 0]
    sd = rb.std(1)
    cmax = float(np.quantile((-(rb - roi[:, None]) / sd[:, None]).max(0), 0.95))
    for i, n in enumerate(names):
        res[n]["lb_simultaneous_95"] = float(roi[i] - cmax * sd[i])
    OUT_JSON.write_text(json.dumps({"results": res, "c_max": cmax, "sec": round(time.time() - t0)}, ensure_ascii=False, indent=1, default=float),
                        encoding="utf-8")
    print("同時の下限", {n: round(res[n]["lb_simultaneous_95"], 3) for n in names})


# ---------------------------------------------------------------- 本物の発走前オッズで、再現の楽観を測る（市場の値だけ）
def cmd_drift() -> None:
    from src.ops import data_ledger as DL
    from src.scripts.data.bet_cycles import PRE_ODDS
    with DL.guard(tool="src/research/megu_cycles/mc_final.py:drift", family="megu_cycles", version="drift-check-v1",
                  period=("2026-10-18", DRIFT_END), purpose="eval_only", frozen_paths=_frozen(), reads=("着順",),
                  reason="再現オッズの楽観（確定の順位でずれを割り当てる持ち込み）を、本物の発走前オッズと市場の値だけで測る（事前登録 final_prereg.json）") as tok:
        pre = pd.read_parquet(PRE_ODDS)
        pre["race_id"] = pre["race_id"].astype(str)
        fl = pd.read_parquet(D.ROOT / "data/page_reference/tables/2026/race_result_flat.parquet",
                             columns=["race_id", "date", "horse_number", "odds", "popularity", "finish_position"])
        fl["race_id"] = fl["race_id"].astype(str)
        dd = pd.to_datetime(fl["date"]).dt.strftime("%Y-%m-%d")
        fl = fl[(dd >= "2026-10-18") & (dd <= DRIFT_END)]       # 申告した期間だけ（2026-11-24 の 1 回目は下限が無く、台帳に range_outside が残った）
        for c in ("horse_number", "odds", "popularity", "finish_position"):
            fl[c] = pd.to_numeric(fl[c], errors="coerce")
        fl = fl[(fl["odds"] > 0)]
        tok.check_dates(fl["date"].astype(str))
        out = {}
        for phase in ("t60", "t15", "t05"):
            p = pre[pre["phase"] == phase][["race_id", "horse_number", "win_odds"]].copy()
            p["horse_number"] = pd.to_numeric(p["horse_number"], errors="coerce")
            p["win_odds"] = pd.to_numeric(p["win_odds"], errors="coerce")
            m = fl.merge(p, on=["race_id", "horse_number"], how="inner")
            m = m[m["win_odds"] > 0]
            n_all = fl.groupby("race_id").size()
            m = m[m.groupby("race_id")["race_id"].transform("size").to_numpy() == n_all.reindex(m["race_id"]).to_numpy()]
            m = m.sort_values(["race_id", "horse_number"]).reset_index(drop=True)
            rid = m["race_id"].to_numpy()

            def qn(x):
                inv = 1.0 / x
                return inv / pd.Series(inv).groupby(rid).transform("sum").to_numpy()
            win = (m["finish_position"].to_numpy() == 1)
            ll = lambda q: float(-np.mean(np.log(np.clip(q[win], 1e-9, 1))))
            q_real, q_fin = qn(m["win_odds"].to_numpy(float)), qn(m["odds"].to_numpy(float))
            # 再現: 他のレースのずれ（d = log(確定/本物の時点)、確定の人気の順に並べたもの）を借りて、確定の人気の順位で当てる
            m["d"] = np.log(m["odds"].to_numpy(float) / m["win_odds"].to_numpy(float))
            races = {r: g.sort_values("popularity")["d"].to_numpy() for r, g in m.groupby("race_id")}
            keys = list(races)
            rng = np.random.default_rng(7)
            res = {"races": len(keys), "ll_real": ll(q_real), "ll_final": ll(q_fin), "ll_sim": {}}
            for s in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
                lls = []
                for rep in range(5):
                    sim = np.empty(len(m))
                    for r, idx in pd.Series(np.arange(len(m))).groupby(rid).indices.items():
                        src = races[keys[int(rng.integers(len(keys)))]]
                        while len(keys) > 1 and src is races[r]:
                            src = races[keys[int(rng.integers(len(keys)))]]
                        pop = m["popularity"].to_numpy()[idx]
                        rk = np.clip(np.nan_to_num(pop, nan=len(src)).astype(int) - 1, 0, len(src) - 1)
                        sim[idx] = m["odds"].to_numpy(float)[idx] * np.exp(-s * src[rk])
                    lls.append(ll(qn(sim)))
                res["ll_sim"][str(s)] = float(np.mean(lls))
            xs = np.array([float(k) for k in res["ll_sim"]])
            ys = np.array(list(res["ll_sim"].values()))
            target = res["ll_real"]
            s_star = None
            for i in range(len(xs) - 1):
                if (ys[i] - target) * (ys[i + 1] - target) <= 0 and ys[i + 1] != ys[i]:
                    s_star = float(xs[i] + (target - ys[i]) * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i]))
                    break
            res["s_star"] = s_star
            out[phase] = res
            print(phase, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != "ll_sim"},
                  {k: round(v, 4) for k, v in res["ll_sim"].items()}, flush=True)
        tok.finish(result_path=DRIFT_JSON)
    # サイクル 43 の曲線を s* で読む（候補ごと、その候補の時点の s*）
    c43 = json.loads((D.ROOT / "docs/operations/megu_cycles/cycle_43.json").read_text(encoding="utf-8"))["sensitivity"]
    est = {}
    for name, curve in c43.items():
        spec = json.loads((C.CAND / f"{name}.json").read_text(encoding="utf-8"))
        s_star = out[spec.get("phase", "t60")]["s_star"]
        xs = np.array([float(k) for k in curve])
        ys = np.array([v["roi"] for v in curve.values()])
        est[name] = {"phase": spec.get("phase", "t60"), "s_star": s_star,
                     "roi_2017_21_at_s_star": (float(np.interp(s_star, xs, ys)) if s_star is not None else None)}
    out["adjusted_estimate"] = est
    DRIFT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("差し引いた見積もり", est)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["final", "drift"])
    a = ap.parse_args()
    {"final": cmd_final, "drift": cmd_drift}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
