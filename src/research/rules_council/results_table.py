"""R0-07 結果の表（説明のための 1 回の集計。新しい議題ではない。decisions.md「R0-07 結果の表の手順の判断」）。

閉じた議題の、閉じた時点の案を、探す側 70%（2016〜24 の開催の 70%）で 1 回だけ数える。合否・選び直しには使わない。
- 行: RC-2 和集合・RC-4 生産者・RC-5 H1・(b) 消し（残した側／消した側）。RC-6 c13 は rules_council_rc6_split.json から写す（数え直さない）。
  RC-3 は数えない（閉じた時点に買い方の案が無い。1 号の決定）。rc3_inner は使わない（記録のために残す）。
- 買い方: 単勝を 1 点 100 円（均一）。(b) は stake（再現 5 本の平均）の重みも並べる。取消・除外は full に入っていない。中止・失格は負けの点として数える。
- 判定: 再現 T-60 の種 1000（full_pre[0]）。5 本（full_pre[0..4]）の回収率の幅も同じ 1 回で出す。
- 出す列（(a)〜(j) と 1 号の列）: 対象・買った・見送った（0 頭）・判定できない・母集団の外のレース数と割合、点数、1 レースの点数の平均と最大、
  的中率（点数あたり・レースあたり）、回収率（95% 開催で束ねたブートストラップ）、5 本の幅、
  基準線（主: 判定の帯 = 再現 T-60 の帯 × クラス群 を全部買う、補助 (i) 母集団の一桁を全部買う、(ii) 確定の帯 × クラス群）、
  分解（ROI = 基準線(ii) × 超過の市場比 × 払戻の項）と「払戻の項を除いた回収率」、市場比（確定の q・判定の q。どちらもレース内で 1 にそろえる）、
  払戻のずれ（確定 ÷ 判定のオッズの平均・中央値）、帯の割合、年ごとの回収率、最大の落ち込み（100 円単位）、最長の連敗（買ったレースの数）、同着の件数。
費用 0 円（手元の表だけ。GCS は読まない）。2025・26 の行は無い（full は 2024 年まで）。確かめの側 30% は数えない。
使い方: KEIBA_SESSION=rules-council python -m src.research.rules_council.results_table（nice -n 19。1 回だけ）
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.rules_council import engine as E
from src.research.rules_council import rc_frame as F

REPO = F.CR.REPO
TABLES = REPO / "data/page_reference/tables"
OUT = REPO / "docs/operations/retrain_trials/rules_council_results_table.json"
UNION = REPO / "config/rules_council/forward_union_rc2.json"
GOING_RANK = {"良": 0, "稍": 1, "稍重": 1, "重": 2, "不": 3, "不良": 3}
N_BOOT = 2000
FROZEN = [REPO / f for f in ("src/research/rules_council/results_table.py", "src/research/rules_council/engine.py", "src/research/rules_council/rc_frame.py",
                             "src/research/rules_council/rc2.py", "src/research/rules_council/debut.py", "src/research/rules_council/dayevents.py",
                             "config/rules_council/forward_union_rc2.json", "src/research/megu_cycles/cycles.py", "src/research/megu_cycles/mc_eval.py",
                             "src/research/megu_cycles/mc_data.py", "config/megu_cycles/candidates/c13_flat.json",
                             "docs/operations/retrain_trials/rules_council_rc6_split.json")]
SEED = 20261128


def _cgrp(cls: np.ndarray) -> np.ndarray:
    return np.select([cls <= 1, cls == 2, cls == 3, cls == 4], [0, 1, 2, 3], 4)


def _norm_q(odds: np.ndarray, rid: np.ndarray) -> np.ndarray:
    inv = np.where(odds > 0, 1.0 / np.where(odds > 0, odds, 1.0), 0.0)
    return inv / pd.Series(inv).groupby(rid).transform("sum").to_numpy()


def base_frame() -> pd.DataFrame:
    d = pd.read_parquet(F.OUTW / "full.parquet", columns=["race_id", "horse_id", "date", "year", "kaisai", "cls", "fin", "rt", "odds_final", "win", "q", "obin"])
    d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    pre = np.load(F.OUTW / "full_pre.npy")
    for k in range(pre.shape[0]):
        d[f"pre{k}"] = pre[k]
    d["srch"] = E.search70(d["kaisai"].to_numpy())
    d["ran"] = True                     # 取消・除外は full に入っていない（rc_frame が odds_final > 0 で絞る）。中止・失格は払戻の無い負けの点として数える
    d["dnf"] = ~(pd.to_numeric(d["fin"], errors="coerce") >= 1)
    d["cgrp"] = _cgrp(pd.to_numeric(d["cls"], errors="coerce").fillna(0).astype(int).to_numpy())
    d["fbin"] = F.odds_bin(d["odds_final"].to_numpy(float))
    rid = d["race_id"].to_numpy()
    d["qf"] = d["q"].to_numpy(float)                                       # rc_frame の q（確定。レース内で 1）
    d["qj"] = _norm_q(d["pre0"].to_numpy(float), rid)
    d["ret"] = np.nan_to_num(d["rt"].to_numpy(float))                      # pay_ok で絞った払戻（rc2_real・engine と同じ）
    d["wn"] = np.nan_to_num(d["win"].to_numpy(float))
    return d


# ---------------------------------------------------------------- 行ごとの母集団と買う馬（閉じた時点の案。動かさない）
def rc2_union(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """母集団: 芝ダの全レース（full は芝ダ・障害なし）。判定できる: 一桁。買う: 一桁 かつ 12 水準のどれか（1 頭は 1 点）。"""
    L = pd.read_parquet(F.OUTW / "full_levels.parquet")
    U = json.loads(UNION.read_text(encoding="utf-8"))
    m = np.zeros(len(L), bool)
    for lv in U["levels"]:
        t, v = lv.split("=", 1)
        m |= (L[t].astype("string") == v).fillna(False).to_numpy(dtype=bool)
    pop = np.ones(len(d), bool)
    return pop, np.ones(len(d), bool), m


def rc3_inner(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """母集団: 同じ日・場・芝ダで、発走の 75 分前までに終わった前のレースが 2 つ以上。買う: 一桁 かつ 枠 ≤ 4 かつ（前の 3 着内の内枠 − 期待）≥ +2。"""
    br = pd.read_parquet(F.OUTW / "daybias_race.parquet")
    br["race_id"] = br["race_id"].astype(str)
    x = d[["race_id"]].merge(br, on="race_id", how="left")
    pop = (x["prior_races"] >= 2).to_numpy()
    bias = (x["prior_inner3"] - x["prior_inner3_exp"]).to_numpy()
    parts = []
    for y in range(2016, F.YEAR_END + 1):
        parts.append(pd.read_parquet(TABLES / str(y) / "race_result_flat.parquet", columns=["race_id", "horse_id", "bracket_number"]))
    b = pd.concat(parts, ignore_index=True).astype({"race_id": str, "horse_id": str}).drop_duplicates(["race_id", "horse_id"])
    inner = d[["race_id", "horse_id"]].merge(b, on=["race_id", "horse_id"], how="left")["bracket_number"]
    inner = pd.to_numeric(inner, errors="coerce")
    known = inner.notna().to_numpy()
    return pop, known, known & (inner.to_numpy() <= 4) & (bias >= 2)


def rc4_breeder(d: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """母集団: 新馬・未勝利の初出走（2020〜24 の年を数える）。判定できる: 生産者の as-of の初出走 ≥ 10。
    買う: 一桁 かつ 縮めた勝率 (w+1)/(n+10) がその年より前の分布（探す側・一桁・分かる走）の上位 20%（debut.cmd_bound と同じ作り）。"""
    cols = pd.read_parquet(F.OUTW / "debut_cols.parquet", columns=["race_id", "horse_id", "breeder_n", "breeder_w"])
    cols["race_id"], cols["horse_id"] = cols["race_id"].astype(str), cols["horse_id"].astype(str)
    x = d[["race_id", "horse_id", "year", "srch", f"pre{k}"]].merge(cols, on=["race_id", "horse_id"], how="left")
    deb_row = x["horse_id"].notna() & x["breeder_n"].notna()
    yr = x["year"].astype(int).to_numpy()
    pop = x["race_id"].isin(set(cols["race_id"])).to_numpy() & (yr >= 2020)
    ok = deb_row.to_numpy() & (x["breeder_n"].fillna(0).to_numpy() >= 10)
    rate = ((x["breeder_w"] + 1) / (x["breeder_n"] + 10)).to_numpy()
    fav = x[f"pre{k}"].to_numpy() < 10
    ref = x["srch"].to_numpy() & ok & fav & (x[f"pre{k}"].to_numpy() <= 50)
    r = rate.copy()
    r[ref] = r[ref] + 1e-12 * np.random.default_rng(0).random(int(ref.sum()))   # cmd_bound と同じ: 部分集合 y の並び（full の順）で引く
    buy = np.zeros(len(x), bool)
    for yy in range(2020, F.YEAR_END + 1):
        prev = r[ref & (yr < yy)]
        if len(prev) < 100:
            continue
        t80 = np.quantile(prev, 0.8)
        buy |= ok & (yr == yy) & (r >= t80)
    return pop, ok, buy


def rc5_h1(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """母集団: 芝ダ。発走の 75 分前までに終わった直前のレースの馬場が、その日の最初のレースより悪い。買う: 一桁 かつ 前の日までの稍重以上で 3 着内 ≥ 1。
    dayevents.cmd_stage0b と同じ作り。"""
    rr = [pd.read_parquet(TABLES / str(y) / "race_result_race.parquet", columns=["race_id", "date", "venue", "surface", "start_time", "track_condition"])
          for y in range(2016, F.YEAR_END + 1)]
    race = pd.concat(rr, ignore_index=True)
    race["race_id"] = race["race_id"].astype(str)
    race["surf"] = race["surface"].astype(str).str[0]
    race = race[race["surf"].isin(["芝", "ダ"])].copy()
    race["g"] = race["track_condition"].astype(str).map(lambda v: GOING_RANK.get(v, GOING_RANK.get(v[:1], np.nan)))
    t = pd.to_datetime(race["start_time"].astype(str), format="%H:%M", errors="coerce")
    race["min"] = t.dt.hour * 60 + t.dt.minute
    race = race.sort_values(["date", "venue", "surf", "min"])
    worse = {}
    for _, g in race.groupby(["date", "venue", "surf"]):
        m, gg, ids = g["min"].to_numpy(), g["g"].to_numpy(), g["race_id"].to_numpy()
        for k in range(len(g)):
            ok = np.flatnonzero(m <= m[k] - 75)
            worse[ids[k]] = bool(len(ok) >= 1 and gg[ok[-1]] > gg[0]) if len(ok) else False
    hist = []
    for y in range(2016, F.YEAR_END + 1):
        h = pd.read_parquet(TABLES / str(y) / "race_result_flat.parquet", columns=["race_id", "date", "horse_id", "finish_position", "track_condition", "surface"])
        hist.append(h[h["surface"].astype(str).str[0].isin(["芝", "ダ"])])
    h = pd.concat(hist, ignore_index=True)
    h["race_id"], h["horse_id"] = h["race_id"].astype(str), h["horse_id"].astype(str)
    fp = pd.to_numeric(h["finish_position"], errors="coerce")
    gr = h["track_condition"].astype(str).map(lambda v: GOING_RANK.get(v, GOING_RANK.get(v[:1], np.nan)))
    h["wet3"] = ((fp >= 1) & (fp <= 3) & (gr >= 1)).astype(float)
    h = h.sort_values(["horse_id", "date"])
    day = h.groupby(["horse_id", "date"])["wet3"].sum().groupby(level=0).cumsum() - h.groupby(["horse_id", "date"])["wet3"].sum()
    h = h.merge(day.rename("wet3_prev").reset_index(), on=["horse_id", "date"], how="left")
    fr = h[["race_id", "horse_id", "wet3_prev"]].drop_duplicates(["race_id", "horse_id"])
    x = d[["race_id", "horse_id"]].merge(fr, on=["race_id", "horse_id"], how="left")
    pop = x["race_id"].map(worse).fillna(False).astype(bool).to_numpy()
    known = x["wet3_prev"].notna().to_numpy()
    return pop, known, known & (x["wet3_prev"].to_numpy() >= 1)


def c13_kill(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(b): c13 の買い目（種 0 の 1 本＝均一、再現 5 本の平均＝stake）と、一桁で RC-2 の消しの 7 水準に当たる行。kill_overlap と同じ作り。"""
    from src.research.megu_cycles import cycles as C
    from src.research.megu_cycles import mc_data as D
    from src.research.megu_cycles import mc_eval as ME
    from src.research.rules_council import rc2 as RC
    sp = json.load(open(C.CAND / "c13_flat.json"))
    C.RICH_FILE = D.OUT / sp["probability"]["file"]
    m = C.load_all()
    m = m[ME.seg_mask(m, sp["segment"])].reset_index(drop=True)
    ab = tuple(sp["combo_ab_t60"])
    st = C.strat_ev(lambda x, k: C.combo_p(x, k, *ab, col="p_rich"), sp["t"], sp["omax"])
    S = np.vstack([st(m, k) for k in range(D.N_PRE)])
    b = pd.DataFrame({"race_id": m["race_id"].astype(str), "horse_id": m["horse_id"].astype(str), "stake": S.mean(0), "s0": S[0]})
    R = RC.RC2(0, mode="k70")
    s = json.loads(open(F.OUTW / "rc2_search.json", encoding="utf-8").read())
    nm = {R.P.level_name(j): j for j in range(R.P.n_levels)}
    sel = np.zeros(R.P.n_levels, bool)
    for n in [r["level"] for r in s["down1"]]:
        sel[nm[n]] = True
    hit = np.where(R.flat >= 0, sel[np.maximum(R.flat, 0)], False).any(1)
    kill = set(zip(R.P.d["race_id"].astype(str)[hit], R.P.d["horse_id"].astype(str)[hit]))
    x = d[["race_id", "horse_id"]].merge(b, on=["race_id", "horse_id"], how="left")
    seg = x["stake"].notna().to_numpy()
    kl = np.array([(a, c) in kill for a, c in zip(x["race_id"], x["horse_id"])])
    Sx = np.vstack([d[["race_id", "horse_id"]].merge(b.assign(sk=S[k]), on=["race_id", "horse_id"], how="left")["sk"].fillna(0).to_numpy()
                    for k in range(D.N_PRE)])
    return seg, x["stake"].fillna(0).to_numpy(), Sx, kl


# ---------------------------------------------------------------- 物差し
def metrics(d: pd.DataFrame, pop_r: np.ndarray, judge: np.ndarray, buy: np.ndarray, w: np.ndarray | None = None) -> dict:
    """pop_r: 行が母集団のレースに入るか。judge: 判定できる一桁の馬か。buy: 買う馬（judge の中）。w: 賭け金の重み（None は均一）。探す側だけ。"""
    s = d["srch"].to_numpy() & d["ran"].to_numpy()
    rid = d["race_id"].to_numpy()
    P = s & pop_r
    B = P & judge & buy
    wt = np.where(B, 1.0 if w is None else w, 0.0)
    races_all = pd.unique(rid[s])
    races_pop = pd.unique(rid[P])
    races_judge = pd.unique(rid[P & judge])
    races_buy = pd.unique(rid[B & (wt > 0)])
    ret, wn, qf, qj = d["ret"].to_numpy(), d["wn"].to_numpy(), d["qf"].to_numpy(), d["qj"].to_numpy()
    oj, of = d["pre0"].to_numpy(float), d["odds_final"].to_numpy(float)
    cg, fb = d["cgrp"].to_numpy(), d["fbin"].to_numpy()
    jb = F.odds_bin(oj)
    ks = d["kaisai"].to_numpy()
    idx = np.flatnonzero(wt > 0)
    S = wt[idx].sum()
    out = {"races_search_side": int(len(races_all)), "races_pop": int(len(races_pop)), "races_out_of_pop": int(len(races_all) - len(races_pop)),
           "races_unjudgeable": int(len(races_pop) - len(races_judge)), "races_skipped_zero": int(len(races_judge) - len(races_buy)),
           "races_bought": int(len(races_buy)), "bets": float(S)}
    if S <= 0:
        return out | {"note": "買い目が無い"}
    per_race = pd.Series(wt[idx]).groupby(rid[idx]).sum()
    won_race = pd.Series(wn[idx] * (wt[idx] > 0)).groupby(rid[idx]).max()
    out |= {"bets_per_race_mean": float(per_race.mean()), "bets_per_race_max": float(per_race.max()),
            "hit_rate_bets": float((wt[idx] * wn[idx]).sum() / S), "hit_rate_races": float(won_race.mean()),
            "roi": float((wt[idx] * ret[idx]).sum() / S), "dead_heat_wins": int(((wn[idx] > 0) & (ret[idx] > 0) & (ret[idx] < of[idx] * 0.999)).sum())}
    # 区間（開催で束ねたブートストラップ）
    uk = np.unique(ks[idx])
    num = pd.Series(wt[idx] * ret[idx]).groupby(ks[idx]).sum().reindex(uk).to_numpy()
    den = pd.Series(wt[idx]).groupby(ks[idx]).sum().reindex(uk).to_numpy()
    rng = np.random.default_rng(SEED)
    bs = []
    for i in range(N_BOOT):
        if i % 500 == 0:
            E.check_stop()
        j = rng.integers(len(uk), size=len(uk))
        bs.append(num[j].sum() / den[j].sum())
    out["roi_ci95"] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
    # 基準線: 主 = 判定の帯 × クラス群 を全部買う（母集団・探す側・判定できる一桁の全馬）。補助 (i) 母集団の一桁を全部買う、(ii) 確定の帯 × クラス群
    A = P & judge

    def cell_roi(bins):
        key = bins * 5 + cg
        n = pd.Series(np.ones(A.sum())).groupby(key[A]).sum()
        r = pd.Series(ret[A]).groupby(key[A]).sum() / n
        return key, r
    kj, rj = cell_roi(jb)
    kf, rf_ = cell_roi(fb)
    out["base_judge_band"] = float((wt[idx] * pd.Series(kj[idx]).map(rj).fillna(0).to_numpy()).sum() / S)
    out["base_i_all_fav"] = float(ret[A].mean())
    base_ii = float((wt[idx] * pd.Series(kf[idx]).map(rf_).fillna(0).to_numpy()).sum() / S)
    out["base_ii_final_band"] = base_ii
    # 分解: ROI = 基準線 × 超過の市場比 × 払戻の項。ρ は E.band_ratio（小さい層は帯だけ、さらに 1 に落とす）
    mr = float((wt[idx] * wn[idx]).sum() / (wt[idx] * qf[idx]).sum())
    out |= {"mr_final_q": mr, "mr_judge_q": float((wt[idx] * wn[idx]).sum() / (wt[idx] * qj[idx]).sum())}
    n_b = np.bincount(fb[A], minlength=len(F.ODDS_EDGES) - 1).astype(float)
    roi_b = np.divide(np.bincount(fb[A], ret[A], len(F.ODDS_EDGES) - 1), n_b, out=np.zeros_like(n_b), where=n_b > 0)
    base_band = float((wt[idx] * roi_b[fb[idx]]).sum() / S)
    out["base_ii_final_band_only"] = base_band
    for name, by_cls, base in (("decomp_band_x_class", True, base_ii), ("decomp_band_only_as_2_1", False, base_band)):
        rho = E.band_ratio(d, A, np.where(A, wn, 0.0), by_cls=by_cls)
        mr_b = float((wt[idx] * qf[idx] * rho[idx]).sum() / (wt[idx] * qf[idx]).sum())
        out[name] = {"base": base, "mr_band": mr_b, "excess_mr": mr / mr_b, "payout_term": (out["roi"] / base) / (mr / mr_b),
                     "roi_without_payout_term": base * mr / mr_b}
    drift = of[idx] / oj[idx]
    o_ = np.argsort(drift)
    cw = np.cumsum(wt[idx][o_])
    out["payout_drift"] = {"mean_w": float(np.average(drift, weights=wt[idx])), "median_w": float(drift[o_][np.searchsorted(cw, cw[-1] / 2)])}
    bands = np.select([oj[idx] < 3, oj[idx] < 5, oj[idx] < 10], ["1〜3 倍", "3〜5 倍", "5〜10 倍"], "10 倍以上")
    out["judge_band_share"] = {b: float(wt[idx][bands == b].sum() / S) for b in ("1〜3 倍", "3〜5 倍", "5〜10 倍", "10 倍以上")}
    out["dnf_bets"] = float(wt[idx][d["dnf"].to_numpy()[idx]].sum())
    yr = d["year"].astype(int).to_numpy()[idx]
    out["roi_by_year"] = {int(y): float((wt[idx] * ret[idx])[yr == y].sum() / wt[idx][yr == y].sum()) for y in np.unique(yr)}
    # 最大の落ち込み（100 円単位）・最長の連敗（買ったレースの数）。日付・レースの順
    g = pd.DataFrame({"date": d["date"].astype(str).to_numpy()[idx], "rid": rid[idx], "pl": wt[idx] * (ret[idx] - 1), "w": wn[idx] * (wt[idx] > 0)})
    g = g.groupby(["date", "rid"]).agg(pl=("pl", "sum"), w=("w", "max")).reset_index().sort_values(["date", "rid"])
    c = g["pl"].cumsum().to_numpy()
    out["max_drawdown_units"] = float(np.max(np.maximum.accumulate(np.r_[0, c])[1:] - c))
    streak = best = 0
    for v in g["w"].to_numpy():
        streak = 0 if v > 0 else streak + 1
        best = max(best, streak)
    out["longest_losing_races"] = int(best)
    return out


def roi_seed(d: pd.DataFrame, pop_r, judge_k, buy_k) -> float | None:
    s = d["srch"].to_numpy() & d["ran"].to_numpy() & pop_r & judge_k & buy_k
    return float(d["ret"].to_numpy()[s].mean()) if s.any() else None


def main() -> None:
    E.check_stop()
    from src.ops import data_ledger as DL
    st = subprocess.run(["git", "status", "--porcelain", "--", "src/research/rules_council/results_table.py"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    if st:
        raise SystemExit(f"道具に commit していない変更がある（1 回だけ流す決まり）:\n{st}")
    if OUT.exists():
        raise SystemExit(f"{OUT} がある。この集計は 1 回だけ（流し直さない）")
    st = subprocess.run(["git", "status", "--porcelain", "--", *map(str, FROZEN)], capture_output=True, text=True, cwd=REPO).stdout.strip()
    if st:
        raise SystemExit(f"結果を決めるファイルに commit していない変更がある:\n{st}")
    with DL.guard(tool="rules_council.results_table", family="rules_council", version="r0_07_table", period=("2016-01-01", "2024-12-31"),
                  purpose="select", reads=("着順", "払戻"), session="rules-council", frozen_paths=FROZEN,
                  reason="R0-07 説明のための結果の表（探す側 70% の 1 回。検定ではない。合否・選び直しに使わない）") as tok:
        d = base_frame()
        fav = {k: d[f"pre{k}"].to_numpy() < 10 for k in range(5)}
        rows = {}
        pop, known, lv = rc2_union(d)
        rows["RC-2 和集合"] = metrics(d, pop, known & fav[0], lv) | {"roi_5seeds": [roi_seed(d, pop, known & fav[k], lv) for k in range(5)]}
        E.check_stop()
        pop, known, b = rc4_breeder(d, 0)
        r4 = metrics(d, pop, known & fav[0], b)
        r4["roi_5seeds"] = []
        for k in range(5):
            p_, k_, b_ = rc4_breeder(d, k)
            r4["roi_5seeds"].append(roi_seed(d, p_, k_ & fav[k], b_))
        rows["RC-4 生産者"] = r4
        E.check_stop()
        pop, known, b = rc5_h1(d)
        rows["RC-5 H1"] = metrics(d, pop, known & fav[0], b) | {"roi_5seeds": [roi_seed(d, pop, known & fav[k], b) for k in range(5)]}
        E.check_stop()
        seg, stake, Sx, kl = c13_kill(d)
        s0 = Sx[0]
        allj = np.ones(len(d), bool)
        for nm_, f_ in (("c13 全部", lambda k: Sx[k] > 0), ("残した側", lambda k: (Sx[k] > 0) & ~(kl & fav[k])), ("消した側", lambda k: (Sx[k] > 0) & kl & fav[k])):
            rows[f"(b) {nm_}（均一・種 0）"] = metrics(d, seg, allj, f_(0)) | {"roi_5seeds": [roi_seed(d, seg, allj, f_(k)) for k in range(5)]}
        rows["(b) c13 全部（stake）"] = metrics(d, seg, allj, stake > 0, w=stake)
        rows["(b) 残した側（stake）"] = metrics(d, seg, allj, (stake > 0) & ~(kl & fav[0]), w=stake)
        rows["(b) 消した側（stake）"] = metrics(d, seg, allj, (stake > 0) & kl & fav[0], w=stake)
        tok.check_dates(d["date"].astype(str))
        sp = json.loads((REPO / "docs/operations/retrain_trials/rules_council_rc6_split.json").read_text(encoding="utf-8"))
        rows["RC-6 c13（2-1 の写し）"] = {"copied_from": "docs/operations/retrain_trials/rules_council_rc6_split.json", "side": sp["side"], "bets": sp["bets"],
                                         "roi": sp["point"]["roi"], "roi_ci90": sp["ci90"]["roi"], "base_ii_final_band_only": sp["point"]["base"],
                                         "excess_mr": sp["point"]["excess_mr"], "payout_term": sp["point"]["payout_term"], "hit_rate_bets": "未集計",
                                         "note": "数え直していない（RC-6 の 1-2・2-1 の判断）。stake の重み・2017〜24"}
        rows["RC-3"] = {"note": "数えない（閉じた時点に買い方の案が無い）。上限の計算（買い方ではない）: 3 着内・1 着の倍率の最大 1.178（rules_council_rc3_bound.json）"}
        rows["RC-5 乗り替わり・取消"] = {"note": "数えない（当日の乗り替わりは過去の出馬表から作れない。取消は 2016〜19 年しか分からない）"}
        rows["RC-4 生産者"]["note"] = ("閾値（上位 20%）は記述 16 の組の中の最大を選んだ値（選んだ後）。上位 10% は as-of で 0.98（記録の写し）。"
                                     "線の元の分布は 2016〜19 年を含む（生産者の列の生き残りの先読み。楽観の向き）。母集団は 2020〜24")
        res = {"note": ("この表の数字は合否ではない。探した期間（探す側 70%）の値で、選んだ後・割引き前の値。100% 超は未来でも続く根拠にならない。"
                        "年・帯・分解は記述で、選び直しと合否に使わない"),
               "side": "探す側 70%（2016〜24。RC-4 は 2020〜24、(b) は 2017〜24）", "judge_odds": "再現 T-60（種 1000。5 本の幅は roi_5seeds）",
               "settle_odds": "確定の単勝の払戻", "rows": rows}
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        tok.finish(result_path=OUT)
    print(json.dumps({k: {"bets": v.get("bets"), "roi": v.get("roi"), "base": v.get("base_judge_band")} for k, v in rows.items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
