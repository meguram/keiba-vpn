"""めぐ分析の第 2 段: 調教師ごとの最終追い切りの出し方（場所・時計・強さ・いつもとの差）を切り口に足したとき、市場より勝つか（統括役の依頼、2026-11-23）。

第 1 段（megu_why）の診断から、結果（3 着内率・回収率）で組を拾う探し方はしない。少数の切り口を先に決め（CUTS）、市場比で測る。
- 切り口 対 外の比（cal）= 勝ち数 ÷ 見込み。見込みは「追い切りの記録があって切り口に当たらない走」の、同じ芝ダ・群・確定の人気の勝率から作る。
- 市場比（edge）= 勝ち数 ÷ 市場の見込み（確定の単勝オッズの逆数をレースの中で和 1 に直した勝率）の和。参考に並べる。
- 選ぶ: 2019〜21 年（追い切りの記録の被覆がそろう年から。2016〜18 年は薄い）。cal の片側の下限（試した数で Bonferroni）が 1 を超え、300 走以上。
- 確かめ: 2022〜24 年。cal > 1 かつ単勝の回収率が外の同じ人気の基準線より高い。
- 2025・2026 年: 確かめを通った切り口だけ、1 回だけ数える（final。台帳に eval_only）。結果を見て選び直さない。

追い切りの値はすべて発走より前（as-of）:
- 最終追い切り = レース週の平日の 4F が最も速い 1 本（src/pipeline/features/final_week_workout.pick_best_weekday。train_date < race_date）。
- 調教師のいつも = その調教師の、そのレースの日より前の日のレースの最終追い切り（同じ日は含めない）。
- 調教師 × パターンの成績 = その日より前の日の着順だけ（縮小: 調教師の平均へ m 走ぶん寄せる）。
追い切りの記録が無い走（欠け）は、どの切り口にも当てはまらないとして数え、欠けの率を年ごとに出す。オッズ・人気は切り口に使わない。

使い方:
  KEIBA_SESSION=megu-why python -m src.research.course.megu_oikiri select   # 2016〜24 年だけ（2025・26 年は開かない。final の後は流せない）
  KEIBA_SESSION=megu-why python -m src.research.course.megu_oikiri final   # 確かめを通った切り口の 2025・26 年（1 回だけ）
出力: <COURSE_OUT>/megu_why/oikiri_*.json・docs/operations/retrain_trials/megu_oikiri.json。費用 0 円（手元の表だけ）。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from src.research.course import run as CR
from src.research.course import v2 as V2
from src.research.course.megu_why import OUTW, POP_CAP, edge_ci

OUT_REPO = CR.REPO / "docs/operations/retrain_trials/megu_oikiri.json"
LAPS = CR.REPO / "data/local/modeling/training_pipeline_v4/tp35/training_laps_fresh.parquet"
STRONG = {"強め", "一杯", "G強", "G一", "直強", "直一"}
M_SHRINK = 20          # 調教師 × パターンの成績を調教師の平均へ寄せる走数
N_HABIT = 20           # 調教師のいつもを数える最低の走数
SEL = (2019, 2021)
CONF = (2022, 2024)
N_MIN = 300
ALPHA = 0.05
CUTS = {
    "slope": ("最終追い切りが坂路", lambda f: f["kind"].eq("slope")),
    "wood": ("最終追い切りがウッド", lambda f: f["kind"].eq("wood")),
    "poly": ("最終追い切りがポリ", lambda f: f["kind"].eq("poly")),
    "accel": ("終いが加速（1F が前の 1F より速い）", lambda f: f["l1"] < f["l2"]),
    "strong": ("強め以上（強め・一杯）", lambda f: f["strong"]),
    "faster": ("調教師のいつもより速い（同じ場所の 4F が平均より 1 標準偏差以上速い）", lambda f: f["t4_z"] <= -1.0),
    "slower": ("調教師のいつもより遅い（1 標準偏差以上）", lambda f: f["t4_z"] >= 1.0),
    "place_change": ("調教師がいつもあまり使わない場所（その場所の割合 20% 未満）", lambda f: f["place_share"] < 0.2),
    "stronger": ("調教師のいつもより強い（強め以上の割合 30% 未満の調教師が強め以上）", lambda f: f["strong"] & (f["strong_share"] < 0.3)),
    "trpat_good": ("調教師 × パターンの過去の 3 着内率が調教師の平均より 5pt 以上高い（縮小後）", lambda f: f["trpat_diff"] >= 0.05),
    "trpat_bad": ("調教師 × パターンの過去の 3 着内率が調教師の平均より 5pt 以上低い（縮小後）", lambda f: f["trpat_diff"] <= -0.05),
}


# ---------------------------------------------------------------- 追い切りの切り口（as-of）
def workouts(year_end: int) -> pd.DataFrame:
    """(horse_id, race_date) ごとの最終追い切り（場所・種別・4F・終い・強さ）。"""
    from src.pipeline.features import final_week_workout as W
    from src.pipeline.features import tp35_lap_groups as LG
    raw = pd.read_parquet(LAPS)
    raw = raw[(raw["race_date"] >= "2015-10-01") & (raw["race_date"] <= f"{year_end}-12-31")].copy()
    raw["horse_id"] = raw["horse_id"].astype(str)
    t = LG.prepare_laps(raw)
    bw = W.pick_best_weekday(t)
    key = t[["horse_id", "race_date", "days_before", "t4", "leg_color", "l2"]].rename(columns={"days_before": "BW_days", "t4": "BW_t4", "l2": "l2_raw"})
    bw = bw.merge(key.drop_duplicates(["horse_id", "race_date", "BW_days", "BW_t4"]), on=["horse_id", "race_date", "BW_days", "BW_t4"], how="left")
    bw["horse_id"] = bw["horse_id"].astype(str)
    return bw.rename(columns={"BW_kind": "kind", "BW_place": "place", "BW_t4": "t4", "BW_l1": "l1", "BW_l2": "l2", "BW_pat": "pat0"})


def oikiri_features(fr: pd.DataFrame, bw: pd.DataFrame) -> pd.DataFrame:
    """全レースの表 fr（race_id・horse_id・date・trainer_id・fin）に、追い切りの切り口の列を足す（調教師のいつもは前の日まで）。"""
    f = fr[["race_id", "horse_id", "date", "trainer_id", "fin"]].copy()
    f["race_date"] = pd.to_datetime(f["date"]).dt.strftime("%Y-%m-%d")
    f["horse_id"] = f["horse_id"].astype(str)
    f = f.merge(bw[["horse_id", "race_date", "kind", "place", "pat0", "t4", "l1", "l2", "leg_color"]].drop_duplicates(["horse_id", "race_date"]),
                on=["horse_id", "race_date"], how="left")
    assert len(f) == len(fr)
    f["has"] = f["kind"].notna()
    f["strong"] = f["leg_color"].isin(STRONG)
    f["pat"] = f["place"].astype(str) + "|" + f["pat0"].astype(str)
    f["top3"] = (pd.to_numeric(f["fin"], errors="coerce") <= 3).astype(float)
    f = f.sort_values(["race_date", "race_id"]).reset_index()
    tr = f["trainer_id"].astype(str)
    h = f[f["has"]]

    def prior(gkeys: list, cols: dict) -> pd.DataFrame:
        """gkeys × 日の和を作り、日の順に累計して、同じ日を除いた前の日までの和（行に戻す）。"""
        g = h.groupby(gkeys + ["race_date"]).agg(**cols).reset_index().sort_values("race_date")
        cs = g.groupby(gkeys)[list(cols)].cumsum() - g[list(cols)]
        return pd.concat([g[gkeys + ["race_date"]], cs.add_prefix("p_")], axis=1)

    hh = h.assign(t4sq=h["t4"] ** 2, one=1.0, st=h["strong"].astype(float), tr=tr[h.index])
    h = hh
    a = prior(["tr"], {"n": ("one", "sum"), "ns": ("st", "sum"), "k3": ("top3", "sum")})
    b = prior(["tr", "place"], {"n": ("one", "sum"), "s": ("t4", "sum"), "ss": ("t4sq", "sum")})
    c = prior(["tr", "pat"], {"n": ("one", "sum"), "k3": ("top3", "sum")})
    f["tr"] = tr
    f = f.merge(a, on=["tr", "race_date"], how="left").merge(b.rename(columns={"p_n": "p_npl"}), on=["tr", "place", "race_date"], how="left") \
         .merge(c.rename(columns={"p_n": "p_npt", "p_k3": "p_k3pt"}), on=["tr", "pat", "race_date"], how="left")
    n = f["p_n"].fillna(0)
    f["strong_share"] = np.where(n >= N_HABIT, f["p_ns"] / n.where(n > 0), np.nan)
    f["place_share"] = np.where(n >= N_HABIT, f["p_npl"].fillna(0) / n.where(n > 0), np.nan)
    npl = f["p_npl"].fillna(0)
    mu = f["p_s"] / npl.where(npl > 0)
    sd = np.sqrt(np.maximum(f["p_ss"] / npl.where(npl > 0) - mu ** 2, 0))
    f["t4_z"] = np.where((npl >= 10) & (sd > 0.2), (f["t4"] - mu) / sd, np.nan)
    p_tr = (f["p_k3"] + M_SHRINK * 0.21) / (n + M_SHRINK)            # 調教師の平均（全体の 3 着内率 約 21% へ寄せる）
    npt = f["p_npt"].fillna(0)
    p_pt = (f["p_k3pt"].fillna(0) + M_SHRINK * p_tr) / (npt + M_SHRINK)
    f["trpat_diff"] = np.where(npt >= 10, p_pt - p_tr, np.nan)
    f = f.sort_values("index").set_index("index")
    f.index.name = None
    return f[["has", "kind", "place", "pat", "t4", "l1", "l2", "strong", "strong_share", "place_share", "t4_z", "trpat_diff"]]


# ---------------------------------------------------------------- 測る
def market(d: pd.DataFrame) -> np.ndarray:
    inv = 1.0 / pd.to_numeric(d["odds_final"], errors="coerce").where(lambda x: x > 0)
    return (inv / inv.groupby(d["race_id"].astype(str).to_numpy()).transform("sum")).fillna(0.0).to_numpy()


def measure(d: pd.DataFrame, f: pd.DataFrame, m: np.ndarray, key_cols=("surf", "group")) -> dict:
    """切り口に当たる走 m の、勝ち数・市場比（区間）・単複の回収率・同じ人気の基準線。
    基準線は「追い切りの記録があって切り口に当たらない走」で作る（記録の有無そのものが結果と相関するため。2019〜21 年は記録の被覆が低く、生き残りの偏りがある）。
    較正した見込み（exp_w）= 同じ基準の走の人気ごとの勝率を当たる走に当てた勝ち数の和。cal = 勝ち数 ÷ exp_w（本命・大穴の偏りを除いた比）。"""
    pay = d["pay_ok"].fillna(False).astype(bool).to_numpy()
    fin = pd.to_numeric(d["fin"], errors="coerce").to_numpy()
    rt = np.where(pay, pd.to_numeric(d["ret_tan"], errors="coerce").fillna(0).to_numpy(), 0.0)
    rf = np.where(pay, pd.to_numeric(d["ret_fuku"], errors="coerce").fillna(0).to_numpy(), 0.0)
    q = market(d)
    pop = np.clip(np.nan_to_num(pd.to_numeric(d["pop_final"], errors="coerce").to_numpy(float), nan=99), 0, POP_CAP).astype(int)
    key = pd.Series(pd.Series(pop).astype(str).to_numpy(), index=d.index)
    for c in key_cols:
        key = key + "|" + d[c].astype(str)
    mm = m & pay
    out = {"n": int(mm.sum()), "races": int(d.loc[mm, "race_id"].nunique())}
    if not mm.any():
        return out
    w = float((fin[mm] == 1).sum())
    out |= {"win": w / mm.sum(), "top3": float((fin[mm] <= 3).mean()), "roi_t": float(rt[mm].mean()), "roi_f": float(rf[mm].mean()),
            "edge": edge_ci(w, float(q[mm].sum())), "q_mean": float(q[mm].mean()), "pop_mean": float(pop[mm].mean())}
    has = f["has"].to_numpy(bool)
    rest = pay & ~m & has
    g = pd.DataFrame({"k": key[rest].to_numpy(), "t": rt[rest], "f": rf[rest], "w": (fin[rest] == 1).astype(float)}).groupby("k").mean()
    kk = key[mm]
    exp_w = float(kk.map(g["w"]).fillna(0).sum())
    out |= {"base_t": float(kk.map(g["t"]).dropna().mean()), "base_f": float(kk.map(g["f"]).dropna().mean()), "cal": edge_ci(w, exp_w)}
    return out


def years_of(d: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(d["year"]).to_numpy()


def cut_masks(f: pd.DataFrame) -> dict:
    return {k: fn(f).fillna(False).to_numpy(bool) & f["has"].to_numpy(bool) for k, (_, fn) in CUTS.items()}


def frame_all(year_end: int) -> pd.DataFrame:
    """全レース（全馬場・重賞を含む）の表と、確定オッズ。year_end 年まで。"""
    from src.research.course import v3 as V3
    d = pd.read_parquet(V3.OUT3 / "frame.parquet", filters=[("year", "<=", year_end)])
    return V2.attach_final_odds(d, range(2016, year_end + 1))


def sides(d: pd.DataFrame) -> dict:
    """平場（めぐ分析の B と同じ: 良馬場・重賞以外）と重賞（G1〜G3、全馬場）の行。"""
    flat = ((d["track"].astype(str) == "良") & d["cls"].map(CR.group_of).notna()).to_numpy()
    graded = d["grade"].astype(str).isin(["G1", "G2", "G3"]).to_numpy()
    return {"flat": flat, "graded": graded}


def cmd_select() -> None:
    OUTW.mkdir(parents=True, exist_ok=True)
    e = CR.env(2024)
    subprocess.run([sys.executable, "-m", "src.research.course.megu_oikiri", "_select_inner"], env=e, cwd=CR.REPO, check=True)


def _select_inner() -> None:
    from src.ops import data_ledger as DL
    if (OUTW / "oikiri_final.json").exists() and os.environ.get("MEGU_OIKIRI_RESELECT") != "1":
        sys.exit("2025・26 年の評価（final）が済んでいるので、選び直さない（新しい確かめの期間を立てるときだけ MEGU_OIKIRI_RESELECT=1）")
    with DL.guard(tool="src/research/course/megu_oikiri.py:_select_inner", family="course", version="megu_oikiri",
                  period=("2016-01-01", "2024-12-31"), purpose="select",
                  reason="追い切りの切り口 11 本を 2019〜21 年で選び、2022〜24 年で確かめる（事前登録の決まり）。2025・26 年は開かない",
                  frozen_paths=[]) as tok:
        res = _select_body()
        tok.finish(result_path=OUTW / "oikiri_select.json")
    return res


def _select_body() -> dict:
    d = frame_all(2024)
    if (years_of(d) > 2024).any():
        sys.exit("2025 年以降の行が入っている")
    d["group"] = d["cls"].map(CR.group_of).fillna("重賞")
    bw = workouts(2024)
    f = oikiri_features(d, bw)
    yr = years_of(d)
    S = sides(d)
    cm = cut_masks(f)
    K = len(CUTS)
    z = __import__("scipy.stats", fromlist=["norm"]).norm.ppf(1 - ALPHA / (2 * K))
    res = {"made_at": datetime.now(CR.JST).isoformat(timespec="seconds"), "cuts": {k: v[0] for k, v in CUTS.items()}, "k": K,
           "select_years": SEL, "confirm_years": CONF, "rule": f"選ぶ: {SEL[0]}〜{SEL[1]} 年の「切り口 対 切り口の外」の比（勝ち数 ÷ 切り口の外の同じ人気の勝率から作る見込み）の片側の下限（Bonferroni、α={ALPHA}/{K}）> 1 かつ {N_MIN} 走以上。"
           f"確かめ: {CONF[0]}〜{CONF[1]} 年の較正した比 > 1 かつ単勝の回収率 > 同じ人気の基準線（どちらも追い切りの記録のある走で作る）",
           "history": ("1 回目（oikiri_select_run1_uncalibrated.json）は、市場比（確定オッズの見込み）と、記録の無い走を含む基準線で選んだ。"
                       "記録の有無と本命・大穴の偏りで上に偏ると分かったので、2 回目で比べ方を直した。"
                       "1 回目と 2 回目で、選ばれた 3 本（wood・accel・faster）も、確かめを通った 2 本（accel・faster）も同じだった。"
                       "ただし直した後に、確かめの期間の比は上がった（accel 1.009→1.060、faster 1.056→1.081）。1 回目で 2022〜24 年の値を見ているので、確かめは 1 回見た後の値。"
                       "この道具の中では 2025・2026 年を開いていないが、同じ担当が先に megu_why で B の 2025・26 年の成績を見ている。"
                       "選びの 1 回目・2 回目は台帳の guard を通さずに流した（後から declare で申告した）。"
                       "3 回目（guard を入れた版、片側・両側の下限を足しただけ）は final の後に流した（担当名が入らず台帳では「不明」）。選ばれた切り口は同じ。"
                       "その後、final の後は選べないようにした"),
           "coverage": {}, "flat": {}, "graded": {}}
    for y in range(2016, 2025):
        my = yr == y
        res["coverage"][str(y)] = {"flat": float(f.loc[my & S["flat"], "has"].mean()), "graded": float(f.loc[my & S["graded"], "has"].mean())}
    from scipy.stats import chi2
    for side in ("flat", "graded"):
        for k, m in cm.items():
            row = {}
            for nm, (a, b) in (("select", SEL), ("confirm", CONF)):
                sel = S[side] & (yr >= a) & (yr <= b)
                row[nm] = measure(d[sel].reset_index(drop=True), f[sel].reset_index(drop=True), m[sel])
            s = row["select"]
            if s.get("n"):
                w, q = s["cal"]["w"], s["cal"]["q"]
                s["cal_lo_bonf"] = float(chi2.ppf(ALPHA / K, 2 * w) / 2 / q) if w > 0 else 0.0          # 片側（決まりはこちら）
                s["cal_lo_bonf2"] = float(chi2.ppf(ALPHA / (2 * K), 2 * w) / 2 / q) if w > 0 else 0.0   # 両側（参考）
            passed = side == "flat" and s.get("n", 0) >= N_MIN and s.get("cal_lo_bonf", 0) > 1.0
            c = row["confirm"]
            row["selected"] = bool(passed)
            row["confirmed"] = bool(passed and c.get("n") and (c["cal"]["ratio"] or 0) > 1.0 and c["roi_t"] > (c.get("base_t") or 9))
            res[side][k] = row
    res["confirmed"] = [k for k, r in res["flat"].items() if r["confirmed"]]
    res["note"] = ("重賞は件数が少ないので選ばない（平場で選び・確かめた切り口を当てるだけ）。"
                   f"区間の下限は片側（z 相当 {__import__('scipy.stats', fromlist=['norm']).norm.ppf(1 - ALPHA / K):.2f}）。両側（z {z:.2f}）の値は cal_lo_bonf2。この道具の中では 2025・2026 年を開いていない")
    (OUTW / "oikiri_select.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    old = json.loads(OUT_REPO.read_text(encoding="utf-8")) if OUT_REPO.exists() else {}
    if old.get("final"):
        res["final"] = old["final"]                      # 1 回だけの評価の結果は残す（選びを流し直しても消さない）
    OUT_REPO.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
    print("select", res["confirmed"], {k: (r["select"].get("n"), round(r["select"].get("cal", {}).get("ratio", 0), 3), round(r["select"].get("cal_lo_bonf", 0), 3),
                                          round(r["confirm"].get("cal", {}).get("ratio", 0), 3)) for k, r in res["flat"].items()}, flush=True)
    return res


def cmd_final() -> None:
    if (OUTW / "oikiri_final.json").exists():
        sys.exit("2025・26 年の評価は 1 回だけ（済んでいる）")
    if not OUT_REPO.exists():
        sys.exit(f"{OUT_REPO} が無い（select を先に流す）")
    sel = json.loads((OUTW / "oikiri_select.json").read_text(encoding="utf-8"))
    if not sel["confirmed"]:
        print("確かめを通った切り口が無いので、2025・2026 年は数えない", flush=True)
        return
    CR.wait_resources()
    e = CR.env(2026) | {"FOLKLORE_OUT": str(CR.OUT / "eval" / "master"), "GRADED_OUT": str(CR.OUT / "eval")}
    subprocess.run([sys.executable, "-m", "src.research.course.megu_oikiri", "_final_inner"], env=e, cwd=CR.REPO, check=True)


def _final_inner() -> None:
    """確かめを通った切り口だけを、2025・2026 年に 1 回だけ当てる。めぐ分析の B（凍結の第 3 版）の買い目と、B ∩ 切り口も並べる（追い切りあり／なし）。"""
    from src.ops import data_ledger as DL
    from src.research.course.course_rule_bets import STATS, chosen_rules
    from src.research.course.megu_why import bet_masks
    from src.research.course.v3 import C3, OUT3, PREREG3
    from src.research.graded import features as GF
    from src.research.graded.run import with_returns
    sel = json.loads((OUTW / "oikiri_select.json").read_text(encoding="utf-8"))
    cuts = sel["confirmed"]
    P = json.loads(PREREG3.read_text(encoding="utf-8"))
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    c = C3(**{k: v for k, v in P["params"].items() if k != "name"})
    kept = json.loads((OUT3 / P["cells_file"]).read_text(encoding="utf-8"))
    rules = chosen_rules(stats, kept)
    with DL.guard(tool="src/research/course/megu_oikiri.py:_final_inner", family="course", version=P["sha256_cells"][:12],
                  period=("2025-01-01", "2026-11-20"), purpose="eval_only",
                  reason=f"追い切りの切り口（2019〜21 で選び 2022〜24 で確かめた {','.join(cuts)}）を 2025・26 年に 1 回だけ当てる。めぐ分析の B ∩ 切り口も並べる。選び直さない",
                  frozen_paths=[]) as tok:
        d, L = CR.frame_and_levels(with_returns(GF.build(out=None)))
        keep = (pd.to_datetime(d["date"]) <= "2026-11-20").to_numpy()       # 台帳の申告の終わり（前向きの期間 11-21〜 を混ぜない）
        d, L = d[keep].reset_index(drop=True), L[keep].reset_index(drop=True)
        L = L.drop(columns=["__race_id", "__horse_id"])
        yrs = pd.to_numeric(d["year"]).to_numpy()
        d = V2.attach_final_odds(d, range(2016, int(yrs.max()) + 1))
        d["group"] = d["cls"].map(CR.group_of).fillna("重賞")
        tok.check_dates(d.loc[yrs >= 2025, "date"])
        f = oikiri_features(d, workouts(2026))
        S = sides(d)
        cm = cut_masks(f)
        res = {"made_at": datetime.now(CR.JST).isoformat(timespec="seconds"), "cuts": cuts, "laps_max": str(pd.read_parquet(LAPS, columns=["race_date"])["race_date"].max()),
               "coverage": {}, "flat": {}, "graded": {}, "b": {}}
        for y in (2025, 2026):
            my = yrs == y
            res["coverage"][str(y)] = {"flat": float(f.loc[my & S["flat"], "has"].mean()), "graded": float(f.loc[my & S["graded"], "has"].mean())}
        for side in ("flat", "graded"):
            for k in cuts:
                res[side][k] = {}
                for nm, yy in (("2025", (2025,)), ("2026", (2026,)), ("2025_2026", (2025, 2026))):
                    m = S[side] & np.isin(yrs, yy)
                    res[side][k][nm] = measure(d[m].reset_index(drop=True), f[m].reset_index(drop=True), cm[k][m])
        # めぐ分析の B（追い切りなし）と B ∩ 切り口（追い切りあり）
        for nm, yy in (("2025", (2025,)), ("2026", (2026,))):
            m = S["flat"] & np.isin(yrs, yy)
            dd, LL, ff = d[m].reset_index(drop=True), L[m].reset_index(drop=True), f[m].reset_index(drop=True)
            bm = bet_masks(dd, LL, kept, c, rules)
            row = {"B": measure(dd, ff, bm["buy_t"])}
            for k in cuts:
                row[f"B∩{k}"] = measure(dd, ff, bm["buy_t"] & cut_masks(ff)[k])
                row[f"B−{k}"] = measure(dd, ff, bm["buy_t"] & ~cut_masks(ff)[k])
            res["b"][nm] = row
        tok.finish(result_path=OUTW / "oikiri_final.json")
    (OUTW / "oikiri_final.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    allr = json.loads(OUT_REPO.read_text(encoding="utf-8"))
    allr["final"] = res
    OUT_REPO.write_text(json.dumps(allr, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
    print("final", json.dumps({k: {p: (v.get("n"), v.get("roi_t"), v.get("base_t"), (v.get("cal") or {}).get("ratio")) for p, v in r.items()} for k, r in res["flat"].items()},
                              ensure_ascii=False, default=float), flush=True)


def main() -> None:
    if sys.argv[1:2] == ["_select_inner"]:
        return _select_inner()
    if sys.argv[1:2] == ["_final_inner"]:
        return _final_inner()
    if sys.argv[1:2] == ["select"]:
        return cmd_select()
    if sys.argv[1:2] == ["final"]:
        return cmd_final()
    sys.exit("使い方: python -m src.research.course.megu_oikiri select | final")


if __name__ == "__main__":
    main()
