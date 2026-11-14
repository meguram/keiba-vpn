"""重賞のルールベース予想家の実行（研究。本番の予測・画面には載せない。手元だけで動き、GCS・GCP は使わない）。

  python -m src.research.graded.run frame      # 1 出走 = 1 行の表と水準（<GRADED_OUT>/frame.parquet・levels.parquet）
  python -m src.research.graded.run tally      # パターンの as-of の件数（tally.parquet）と、2024 年末で止めた版（tally_frozen.parquet）
  python -m src.research.graded.run select     # 縮める強さ・型の採否・束ね・点の足し方を 2016〜2024 の中だけで決め、prereg.json に凍結する
  python -m src.research.graded.run smoke      # evaluate と同じ処理を確認期間（2022〜2024）で流す試運転（2025・2026 年の行は読み込むが、評価にも出力にも使わない）
  python -m src.research.graded.run evaluate   # 2025・2026 年の重賞で 1 回だけ評価する（EVALUATED.json があれば止まる）
  python -m src.research.graded.run export     # 重賞ごとのパターン表・レースごとの JSON・HTML の材料（graded_report.json）

2 回目の評価（回収率で選ぶ版。ユーザーの依頼で、2025・2026 年を一度見た後に行う。計画書 第 17 節）:
  GRADED_OUT=<新しい置き場> で frame・tally を作り直し（中止の馬を含む master）、
  python -m src.research.graded.run select_roi  # 回収率で選ぶ版の一覧とパターンの採否を 2016〜2024 の中だけで決め、prereg_roi.json に凍結する
  python -m src.research.graded.run smoke2      # 2 回目の評価と同じ処理を確認期間（2022〜2024）で流す試運転（確認期間は採否に使ったので参考）
  python -m src.research.graded.run evaluate2   # 2025・2026 年で 1 回だけ評価する（第 1 版の規則のまま直したデータで流した成績版と、回収率で選ぶ全版）

期間（計画書 第 5・6 節）: 発見 = 2018〜2021 年の重賞、確認 = 2022〜2024 年、検証 = 2025・2026 年（1 回だけ）。
CPU は他の学習と分け合う: nice -n 10 で動かし、スレッドは 8 以下（下の環境変数）。
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.research.graded import catalog as CAT  # noqa: E402
from src.research.graded import model as M  # noqa: E402
from src.research.graded import tally as TL  # noqa: E402
from src.research.graded.features import GRADED_OUT, build  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
# 事前登録の置き場（commit する）。作業ツリーに書けないときは GRADED_PREREG で別の場所に書き、写して commit する
PREREG = Path(os.environ.get("GRADED_PREREG") or Path(__file__).with_name("prereg.json"))
PREREG_ROI = Path(os.environ.get("GRADED_PREREG_ROI") or Path(__file__).with_name("prereg_roi.json"))
ROUND2_NOTE = ("2 回目の評価（2025・2026 年の結果を一度見た後）。ユーザーの依頼（統括役経由、2026-11-14）: 各パターンに回収率を持たせ、"
               "回収率まで裏付けのあるパターンだけで印を付ける。基準はいろいろな版を並べる。")
DISC, CONF, EVAL = (2018, 2021), (2022, 2024), (2025, 2026)
Z_DISC, Z_CONF = 1.645, 1.2816          # 発見は片側 95%、確認は片側 90%
LAMS = (0.5, 1.0, 2.0)
PAYOFFS = [REPO / "data/local/modeling/betting/payoff.parquet", REPO / "data/local/modeling/betting_2026/payoff.parquet"]
# 比べる基準（p_win の順で ◎○▲）。v6e16（2016 年からの表）の 2026 年の予測は、2026 年の学習側の表が無く GCS も要るため作れなかった
# （2026-11-13 の調査）。2026 年は本番 v6e（2020 年からの表、bundle_v6_2025）を推論の経路で流した予測（重賞 95 レース）で比べる。
BASELINES = {
    "v6e16": {2025: Path("/home/jovyan/work/retrain_out/base_2016/final_eval_v6e16/test2025_predictions.parquet")},
    "v6e本番": {2025: REPO / "data/local/modeling/training_pipeline/final_eval_v6e/test2025_predictions.parquet",
               2026: GRADED_OUT / "v6e_2026/pred_2026_graded_prod_v6e_serving.parquet"},
}
JST = timezone(timedelta(hours=9))
ROW_COLS = ["row", "race_id", "horse_id", "year", "dt", "field", "y3", "y1", "fin", "hn", "rkey", "course", "grade", "race_name", "venue",
            "surf", "dist", "jockey_name", "ret_tan", "ret_fuku", "ret_tan_c", "ret_fuku_c", "pay_ok"]


def _graded_target(d: pd.DataFrame) -> pd.Series:
    return d["grade"].isin(TL.GRADES) & (d["year"] >= DISC[0])


def cmd_frame() -> None:
    d = build(out=GRADED_OUT / "frame.parquet")
    p = d[pd.to_numeric(d["cls"], errors="coerce") >= 4]
    L = CAT.levels(p)
    L["__race_id"] = p["race_id"].astype(str).to_numpy()
    L["__horse_id"] = p["horse_id"].astype(str).to_numpy()
    L.to_parquet(GRADED_OUT / "levels.parquet")
    print("levels", L.shape)


def _load():
    d = pd.read_parquet(GRADED_OUT / "frame.parquet")
    d["dt"] = pd.to_datetime(d["dt"])
    L = pd.read_parquet(GRADED_OUT / "levels.parquet")
    # 行番号の対応を確かめる（frame.parquet は index=False で保存しているので、読み直した行番号が水準の表の行番号と一致するはず）
    chk = d.loc[L.index, ["race_id", "horse_id"]].astype(str)
    assert (chk["race_id"].to_numpy() == L["__race_id"].to_numpy()).all() and (chk["horse_id"].to_numpy() == L["__horse_id"].to_numpy()).all(), \
        "frame と levels の行がずれている"
    return d, L.drop(columns=["__race_id", "__horse_id"])


def with_returns(d: pd.DataFrame) -> pd.DataFrame:
    """d に馬ごとの単勝・複勝の払戻（回収率の物差し。payout.attach_returns）を付ける。"""
    from src.research.graded import payout as PO
    hits, ok = PO.load_win_place()
    return PO.attach_returns(d, hits, ok)


def cmd_tally() -> None:
    d, L = _load()
    d = with_returns(d)
    tm = _graded_target(d)
    ids = [t.id for t in CAT.TYPES]
    TL.tally_all(d, L, tm, ids).to_parquet(GRADED_OUT / "tally.parquet", index=False)
    tm2 = d["grade"].isin(TL.GRADES) & (d["year"] >= EVAL[0])
    TL.tally_all(d, L, tm2, ids, freeze=True).to_parquet(GRADED_OUT / "tally_frozen.parquet", index=False)


def _rows(d: pd.DataFrame) -> pd.DataFrame:
    r = d.reset_index().rename(columns={"index": "row"})
    r = r[[c for c in ROW_COLS if c in r.columns]].copy()
    r["y3"] = pd.to_numeric(r["y3"], errors="coerce").fillna(0)
    r["y1"] = pd.to_numeric(r["y1"], errors="coerce").fillna(0)
    return r


def _ctx(path: Path, rows: pd.DataFrame) -> pd.DataFrame:
    return M.attach_context(pd.read_parquet(path), rows)


def _yr(T: pd.DataFrame, span) -> pd.DataFrame:
    return T[(T["year"] >= span[0]) & (T["year"] <= span[1])]


def cmd_select() -> None:
    if (GRADED_OUT / "EVALUATED.json").exists():
        sys.exit("2025・2026 年の評価は済んでいる。評価の後に採否・規則を作り直さない")
    if PREREG.exists() and os.environ.get("GRADED_RESELECT") != "1":
        sys.exit(f"{PREREG} は凍結済み。作り直すなら GRADED_RESELECT=1（評価の前に限る）")
    d, _ = _load()
    rows = _rows(d)
    T = _ctx(GRADED_OUT / "tally.parquet", rows)
    main_types = [t.id for t in CAT.TYPES if not t.optional]
    Td, Tc = _yr(T, DISC), _yr(T, CONF)
    # 1) 縮める強さ（発見期間の対数損失）
    ll = M.logloss_m(Td[Td["type"].isin(main_types)])
    best = ll.iloc[0]
    mg, ms, mm = float(best.m_g), float(best.m_s), float(best.m_m)
    print("m", mg, ms, mm, flush=True)
    # 2) 型の採否（発見 → 確認）
    Sd, Sc = M.shrink(Td, mg, ms, mm), M.shrink(Tc, mg, ms, mm)
    zd, zc = M.type_z(Sd), M.type_z(Sc)
    z = zd[["z", "races"]].add_suffix("_disc").join(zc[["z", "races"]].add_suffix("_conf"), how="outer")
    from math import erf, sqrt
    zz = z.loc[[t for t in main_types if t in z.index and pd.notna(z.loc[t, "z_disc"])], "z_disc"]   # BH は採否の対象の型だけで数える
    pval = zz.map(lambda v: 0.5 * (1 - erf(v / sqrt(2))))
    rank = pval.rank(method="first")
    k = int(len(pval))
    bh_ok = pval <= (rank / k) * 0.10
    passed_bh = set(pval[bh_ok].index) if bh_ok.any() else set()
    max_rank = max([rank[t] for t in passed_bh], default=0)
    bh = {t: bool(rank[t] <= max_rank) for t in pval.index}
    adopt = [t for t in main_types if t in z.index and z.loc[t, "z_disc"] >= Z_DISC and z.loc[t, "z_conf"] >= Z_CONF]
    disc_only = [t for t in main_types if t in z.index and z.loc[t, "z_disc"] >= Z_DISC and t not in adopt]
    print("adopt", adopt, flush=True)
    sets = {"preday": [t for t in adopt if not CAT.BY_ID[t].day_of], "dayof": list(adopt)}
    # 3) 束ね（発見期間の点の相関）
    reps = {}
    for name, ts in sets.items():
        order = sorted(ts, key=lambda t: -z.loc[t, "z_disc"])
        reps[name] = M.cluster_types(Sd[Sd["type"].isin(order)], order) if order else {}
    # 4) 点の足し方（確認期間のレース内 AUC）
    variants = []
    for lam in LAMS:
        Sc_l = M.shrink(Tc, mg, ms, mm, lam)
        for wname in ("equal", "z"):
            for name in sets:
                use = sorted(set(reps[name].values()))
                if not use:
                    continue
                w = {t: float(z.loc[t, "z_disc"]) for t in use} if wname == "z" else None
                auc = M.race_auc(M.score(Sc_l, use, w))
                variants.append({"set": name, "lam": lam, "weight": wname, "auc_conf": auc})
    vt = pd.DataFrame(variants)
    pick = vt[vt["set"] == "preday"].sort_values("auc_conf", ascending=False).iloc[0] if len(vt) else None
    pre = {
        "frozen_at": datetime.now(JST).isoformat(timespec="seconds"),
        "note": "2016〜2024 年の中だけで決めた。2025・2026 年の評価の前に凍結し、評価の後に変えない。",
        "periods": {"discovery": DISC, "confirm": CONF, "evaluate": EVAL},
        "thresholds": {"z_disc": Z_DISC, "z_conf": Z_CONF, "min_main_n": M.MIN_MAIN_N, "min_main_ed": M.MIN_MAIN_ED, "min_sub_n": M.MIN_SUB_N,
                       "min_sub_races": M.MIN_SUB_RACES, "sub_fallback_n": M.SUB_FALLBACK_N, "max_share": M.MAX_SHARE, "good_edhit": M.GOOD_EDHIT,
                       "cluster_corr": 0.7},
        "shrink": {"m_g": mg, "m_s": ms, "m_m": mm, "grid_logloss": ll.head(10).to_dict("records")},
        "score": {"lam": float(pick["lam"]) if pick is not None else 1.0, "weight": str(pick["weight"]) if pick is not None else "equal",
                  "variants": variants, "marks": M.MARKS, "tie_break": "主の好成績パターンの数が多い順、さらに馬番の小さい順"},
        "types": {t: {"name": CAT.BY_ID[t].name, "family": CAT.BY_ID[t].family, "day_of": CAT.BY_ID[t].day_of,
                      "z_disc": _f(z.loc[t, "z_disc"]) if t in z.index else None, "z_conf": _f(z.loc[t, "z_conf"]) if t in z.index else None,
                      "races_disc": _f(z.loc[t, "races_disc"]) if t in z.index else None, "bh10_disc": bh.get(t, False),
                      "status": ("採用" if t in adopt else "発見だけで効いた（確認で消えた）" if t in disc_only else "発見で効かなかった"),
                      "rep": {s: reps[s].get(t) for s in sets}} for t in main_types},
        "sets": {s: {"adopted": sets[s], "used": sorted(set(reps[s].values()))} for s in sets},
        "comparison": {"megu": "前日版の使う型に M01・M02（めぐ指数の順位）を足した版。採否の対象にしない、比較だけ",
                       "frozen_counts": "件数を 2024 年末で止めた版（採否・m・λ は同じ）"},
        "evaluation": {"metrics": "◎の勝率・複勝率、◎○▲の3着内の頭数、◎の単勝・複勝の回収率（各100円）、◎→○▲の馬連・ワイドの流し（2点・200円）の回収率",
                       "bootstrap": "レース単位 2000 回、95% 区間",
                       "breakdown": "G1・G2・G3 別、芝・ダート別、主あり（同じレース名 × 同じコースの過去の回が 4 回以上）／副・全体だけ の別（各 5 レース以上）",
                       "simple_baseline": "単純な基準: 前走のクラス × 着順（B02）だけの点で並べた印（採否に関係なく、同じ m・λ で縮める）",
                       "baseline":"v6e の p_win の順。v6e16（2025 年、bundle_v6_2024 の test）と本番 v6e（2025 年は test、2026 年は bundle_v6_2025 を推論の経路で流した予測。重賞 95 レース）"},
    }
    PREREG.write_text(json.dumps(pre, ensure_ascii=False, indent=1, default=_f), encoding="utf-8")
    print(vt.sort_values("auc_conf", ascending=False).head(8).to_string())
    print("prereg written", PREREG)


def cmd_select_roi() -> None:
    """回収率で選ぶ版の一覧・パターンの採否・使う型を 2016〜2024 年の中だけで決め、PREREG_ROI に凍結する。"""
    from src.research.graded import roi as R
    if (GRADED_OUT / "EVALUATED.json").exists():
        sys.exit("この置き場では評価が済んでいる。評価の後に採否・規則を作り直さない")
    if PREREG_ROI.exists() and os.environ.get("GRADED_RESELECT") != "1":
        sys.exit(f"{PREREG_ROI} は凍結済み。作り直すなら GRADED_RESELECT=1（評価の前に限る）")
    pre = json.loads(PREREG.read_text(encoding="utf-8"))
    d, _ = _load()
    rows = _rows(with_returns(d))
    T = _ctx(GRADED_OUT / "tally.parquet", rows)
    T = T[(T["year"] >= DISC[0]) & (T["year"] <= CONF[1])]
    sh, sc = pre["shrink"], pre["score"]
    S = M.shrink(T, sh["m_g"], sh["m_s"], sh["m_m"], sc["lam"])
    vers, U, rho = R.select_versions(S, rows, pre, DISC, CONF)
    U.to_parquet(GRADED_OUT / "roi_units.parquet", index=False)
    out = {
        "frozen_at": datetime.now(JST).isoformat(timespec="seconds"),
        "evaluation_round": 2,
        "note": ROUND2_NOTE + " 版の一覧・パターンの採否は 2016〜2024 年の中だけで決め、評価の前に凍結した。評価の後に版を足したり選び直したりしない。",
        "base_prereg": {"path": "src/research/graded/prereg.json", "frozen_at": pre["frozen_at"], "uses": "縮める強さ m・主の重み λ・型の採否（採用した型）"},
        "periods": {"discovery": DISC, "confirm": CONF, "evaluate": EVAL},
        "grid": R.GRID, "small_grid": R.SMALL_GRID, "crit_m": R.CRIT_M, "families": [R.FAMILY_BIG, R.FAMILY_SMALL], "primary": R.version_name(R.PRIMARY), "n_versions": len(vers),
        "rules": {"alpha": R.ALPHA, "q": R.Q, "n_boot": R.N_BOOT, "bootstrap": "レース単位（レースごとにポアソン(1) の重み）",
                  "cap": {"単勝": 50, "複勝": 10}, "unit": "型 × 水準 × 向き（好 = as-of の点 x が正、凡 = 負）。x を作った層が版の層に入る行だけ",
                  "performance": "3 着内の数が期待（Σ 3/頭数）を上回る（好）／下回る（凡）ことも両方の期間で満たす",
                  "score": "残ったパターンの x だけを等しい重みで足す（ほかは 0）。束ねは残った x の発見期間の |相関| ≥ 0.7",
                  "payout_source": "race_result_race の payoff（2016〜2026 年。2020〜2024 年は payoff.parquet と全行一致。同着の単勝が 2 頭つながった表記など、単勝の馬番・払戻を分けられない 9 レースは外す）",
                  "shrunk_point": "縮めた点 = (払戻の和 + m × 全体の回収率) / (走数 + m)。m = 0 が「点＋0.10」。どれも全体 ±0.10 と比べる",
                  "tie": "同点は成績版の点: 点 → 第 1 版（成績版）の点 → 主の好成績パターンの数 → 馬番。同点は馬番: 点 → 主の好成績パターンの数 → 馬番"},
        "overall_roi": rho,
        "versions": vers,
    }
    PREREG_ROI.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=_f), encoding="utf-8")
    print(pd.DataFrame([{"name": v["name"], "good": v["n_good"], "bad": v["n_bad"], "used": len(v["used"])} for v in vers]).to_string())
    print("rho", rho)
    print("prereg_roi written", PREREG_ROI)


def _f(v):
    try:
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
    except (TypeError, ValueError):
        return str(v)


# ------------------------------------------------------------------ 評価
def load_payoff() -> dict:
    out: dict = {}
    for p in PAYOFFS:
        pay = pd.read_parquet(p, columns=["race_id", "bet_type", "combo", "payout", "parsed"])
        pay = pay[pay["bet_type"].isin(["tansho", "fukusho", "umaren", "wide"]) & pay["parsed"].fillna(False).astype(bool)]
        for rid, bt, combo, v in zip(pay["race_id"].astype(str), pay["bet_type"], pay["combo"].astype(str), pay["payout"]):
            if bt in ("umaren", "wide"):
                a, b = sorted(int(x) for x in combo.split("-"))
                combo = f"{a}-{b}"
            out.setdefault((rid, bt), {})[combo] = int(v)
    return out


def race_metrics(mk: pd.DataFrame, pay: dict) -> pd.DataFrame:
    """mk: race_id, hn, rank（1 = ◎）, y1, y3。レースごとの指標と購入額・払戻。4 券種の払戻がそろわないレースは除く。"""
    rows = []
    for rid, g in mk.groupby("race_id"):
        if not all((rid, bt) in pay for bt in ("tansho", "fukusho", "umaren", "wide")):
            continue   # 4 券種の払戻がそろったレースだけ（rank_bakeoff_eval と同じ定義）
        g = g.sort_values("rank")
        top = g.iloc[0]
        h1 = int(top["hn"])
        h23 = [int(x) for x in g["hn"].iloc[1:3]]
        r = {"race_id": rid, "win": float(top["y1"]), "place": float(top["y3"]), "top3_in3": float(g["y3"].iloc[:3].sum())}
        r["ret_tansho"] = pay.get((rid, "tansho"), {}).get(str(h1), 0)
        r["ret_fukusho"] = pay.get((rid, "fukusho"), {}).get(str(h1), 0)
        for bt in ("umaren", "wide"):
            r[f"ret_{bt}"] = sum(pay.get((rid, bt), {}).get(f"{min(h1, h)}-{max(h1, h)}", 0) for h in h23)
        r["cost_tansho"], r["cost_fukusho"], r["cost_umaren"], r["cost_wide"] = 100, 100, 100 * len(h23), 100 * len(h23)
        rows.append(r)
    return pd.DataFrame(rows)


def summarize(rt: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    def stat(df):
        s = {"win": df["win"].mean(), "place": df["place"].mean(), "top3_in3": df["top3_in3"].mean()}
        for bt in ("tansho", "fukusho", "umaren", "wide"):
            s[f"roi_{bt}"] = df[f"ret_{bt}"].sum() / df[f"cost_{bt}"].sum()
        return s
    point = stat(rt)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(rt))
    boots = [stat(rt.iloc[rng.choice(idx, len(idx))]) for _ in range(n_boot)]
    out = {"races": int(len(rt))}
    for k, v in point.items():
        b = np.array([x[k] for x in boots])
        out[k] = {"est": float(v), "lo": float(np.quantile(b, 0.025)), "hi": float(np.quantile(b, 0.975))}
    return out


def diff_ci(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 2000, seed: int = 1) -> dict:
    """同じレースでの差（a − b）の区間。"""
    m = a.merge(b, on="race_id", suffixes=("_a", "_b"))
    if len(m) == 0:
        return {"races": 0}
    rng = np.random.default_rng(seed)
    out = {}
    for k in ("win", "place", "top3_in3"):
        dlt = (m[f"{k}_a"] - m[f"{k}_b"]).to_numpy()
        bs = [dlt[rng.choice(len(dlt), len(dlt))].mean() for _ in range(n_boot)]
        out[k] = {"est": float(dlt.mean()), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975))}
    for bt in ("tansho", "fukusho", "umaren", "wide"):
        ra, ca, rb, cb = (m[f"ret_{bt}_a"].to_numpy(), m[f"cost_{bt}_a"].to_numpy(), m[f"ret_{bt}_b"].to_numpy(), m[f"cost_{bt}_b"].to_numpy())
        bs = []
        for _ in range(n_boot):
            i = rng.choice(len(m), len(m))
            bs.append(ra[i].sum() / ca[i].sum() - rb[i].sum() / cb[i].sum())
        out[f"roi_{bt}"] = {"est": float(ra.sum() / ca.sum() - rb.sum() / cb.sum()), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975))}
    out["races"] = int(len(m))
    return out


def v6e_marks(rows: pd.DataFrame, paths: dict) -> pd.DataFrame:
    parts = []
    for y, p in paths.items():
        if p.exists():
            v = pd.read_parquet(p, columns=["race_id", "horse_id", "p_win"])
            v["race_id"], v["horse_id"] = v["race_id"].astype(str), v["horse_id"].astype(str)
            parts.append(v[v["race_id"].str[:4] == str(y)])
    if not parts:
        return pd.DataFrame()
    v = pd.concat(parts).merge(rows[["race_id", "horse_id", "hn", "y1", "y3", "year"]], on=["race_id", "horse_id"], how="inner")
    v = v.sort_values(["race_id", "p_win", "hn"], ascending=[True, False, True])
    v["rank"] = v.groupby("race_id").cumcount() + 1
    return v


SIMPLE_TYPES = ["B02"]   # 単純な基準: 「前走のクラス × 着順」だけで並べた印（計画書 第 8 節。ルールを積み上げた効果を見る）


def scored(T: pd.DataFrame, pre: dict, set_name: str | None, extra: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """set_name が None なら extra の型だけを等しい重みで使う（単純な基準）。"""
    sh, sc = pre["shrink"], pre["score"]
    S = M.shrink(T, sh["m_g"], sh["m_s"], sh["m_m"], sc["lam"])
    use = (list(pre["sets"][set_name]["used"]) if set_name else []) + list(extra or [])
    if not use:
        raise SystemExit(f"{set_name}: 採用された型が無い")
    zmin = min([pre["types"][t]["z_disc"] for t in use if pre["types"].get(t, {}).get("z_disc") is not None] or [1.0])
    w = None
    if sc["weight"] == "z" and set_name:   # 比較だけの型（z が無い型）は最小の重み
        w = {t: (pre["types"][t]["z_disc"] if pre["types"].get(t, {}).get("z_disc") is not None else zmin) for t in use}
    return S, M.marks(M.score(S, use, w))


def race_strata(T: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """評価の内訳に使うレースの属性: 芝ダ と、主の集計が使えたか（同じレース名 × 同じコースの過去の回が MIN_MAIN_ED 回以上、
    かつ走数が MIN_MAIN_N 以上。shrink が主の層を使う条件と同じ）。T は点に使った版の件数（凍結版なら凍結した件数）。"""
    a = T.groupby("race_id")[["all_main_ed", "all_main_n"]].max()
    r = rows.drop_duplicates("race_id").set_index("race_id")[["surf"]].join(a)
    ok = (r["all_main_ed"].fillna(0) >= M.MIN_MAIN_ED) & (r["all_main_n"].fillna(0) >= M.MIN_MAIN_N)
    r["main"] = np.where(ok, "主あり", "副・全体だけ")
    return r


def _eval_version(rt: pd.DataFrame, span, gmap: pd.Series, strata: pd.DataFrame | None, rt_base: dict, ref: pd.DataFrame | None,
                  ref_name: str | None, n_boot: int) -> dict:
    """1 つの版の期間ごとの指標。strata が None のときは内訳・基準との比較を省く（回収率で選ぶ版のうち主に見る版以外）。"""
    rt = rt.copy()
    rt["year"] = rt["race_id"].str[:4].astype(int)
    rt["grade"] = rt["race_id"].map(gmap)
    ent = {}
    periods = [(str(y), rt["year"] == y) for y in range(span[0], span[1] + 1)] + [(f"{span[0]}-{span[1]}", rt["year"] >= span[0])]
    for lab, sel in periods:
        r = rt[sel]
        if len(r) == 0:
            continue
        e = {"all": summarize(r, n_boot=n_boot)}
        if strata is not None:
            for gr in TL.GRADES:
                if (r["grade"] == gr).sum() >= 5:
                    e[gr] = summarize(r[r["grade"] == gr], n_boot=n_boot)
            for col in ("surf", "main"):
                lab_s = r["race_id"].map(strata[col])
                for v in sorted(lab_s.dropna().unique()):
                    if (lab_s == v).sum() >= 5:
                        e[f"{col}:{v}"] = summarize(r[lab_s == v], n_boot=n_boot)
            for bk, rb in rt_base.items():
                b = rb[rb["race_id"].isin(r["race_id"])]
                if len(b):
                    e[f"base_{bk}"] = summarize(b, n_boot=n_boot)
                    e[f"self_same_races_{bk}"] = summarize(r[r["race_id"].isin(b["race_id"])], n_boot=n_boot)
                    e[f"diff_vs_{bk}"] = diff_ci(r, b, n_boot=n_boot)
        if ref is not None:
            dd = diff_ci(r, ref[ref["race_id"].isin(r["race_id"])], n_boot=n_boot)
            if dd["races"]:
                e[f"diff_vs_{ref_name}"] = dd
        ent[lab] = e
    return ent


def cmd_evaluate(smoke: bool = False, round2: bool = False) -> None:
    """2025・2026 年の評価（1 回だけ）。smoke=True は同じ処理を確認期間（2022〜2024）で流す試運転で、
    2025・2026 年の行は読み込むが評価にも出力にも使わず、EVALUATED.json も作らない（出力は <GRADED_OUT>/smoke_conf/）。
    round2=True は 2 回目の評価（PREREG_ROI の回収率で選ぶ全版を足す。第 1 版の評価とは別の置き場 GRADED_OUT で流す）。"""
    from src.research.graded import roi as R
    flag = GRADED_OUT / "EVALUATED.json"
    span = CONF if smoke else EVAL
    out_dir = GRADED_OUT / ("smoke_conf" if smoke else ".")
    if flag.exists() and not smoke:
        sys.exit(f"評価は済んでいる（{flag}）。やり直さない")
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = json.loads(PREREG.read_text(encoding="utf-8"))
    pre_roi = json.loads(PREREG_ROI.read_text(encoding="utf-8")) if round2 else None
    d, _ = _load()
    rows = _rows(with_returns(d) if round2 else d)
    T = _yr(_ctx(GRADED_OUT / "tally.parquet", rows), span)
    pay = load_payoff()
    # 版の名前 → (点に使った件数の表, 印, 詳しく出すか)。内訳の「主あり」はその版の件数で決める
    S_pre, mk_pre = scored(T, pre, "preday")
    versions = {
        "前日版": (T, mk_pre, True),
        "当日版": (T, scored(T, pre, "dayof")[1], True),
    }
    if not smoke and not round2:
        Tf = _ctx(GRADED_OUT / "tally_frozen.parquet", rows)
        versions["前日版（件数を2024年末で凍結）"] = (Tf, scored(Tf, pre, "preday")[1], True)
    if not round2:
        versions["前日版＋めぐ指数（比較）"] = (T, scored(T, pre, "preday", ["M01", "M02"])[1], True)
    versions["単純な基準（前走のクラス×着順だけ）"] = (T, scored(T, pre, None, SIMPLE_TYPES)[1], True)
    empty = []
    if round2:
        for v in pre_roi["versions"]:
            mk = R.marks_for(S_pre, v, mk_pre.set_index("row")["score"])
            if mk is None:
                empty.append(v["name"])
                continue
            versions[v["name"]] = (T, mk, bool(v["primary"]))
    ev_rows = rows[(rows["year"] >= span[0]) & (rows["year"] <= span[1])]
    ev_rows = ev_rows[ev_rows["race_id"].isin(set(rows.loc[rows["grade"].isin(TL.GRADES), "race_id"]))]
    base = {} if smoke else {k: v6e_marks(ev_rows, p) for k, p in BASELINES.items()}
    res: dict = {"evaluated_at": datetime.now(JST).isoformat(timespec="seconds"), "prereg_frozen_at": pre["frozen_at"], "versions": {},
                 "baselines": {k: {str(y): str(p) for y, p in v.items()} for k, v in BASELINES.items()}}
    if round2:
        res |= {"evaluation_round": 2, "note": ROUND2_NOTE, "prereg_roi_frozen_at": pre_roi["frozen_at"], "primary": pre_roi["primary"],
                "n_roi_versions": len(pre_roi["versions"]), "empty_versions": empty, "ties": {}}
    gmap = rows.drop_duplicates("race_id").set_index("race_id")["grade"]
    rt_base = {k: race_metrics(v, pay) for k, v in base.items() if len(v)}
    if not smoke:
        # 結果を出す前に、評価済みの印を先に立てる（途中で落ちても、結果を見た後にやり直せる状態を残さない）
        flag.write_text(json.dumps({"evaluated_at": res["evaluated_at"], "evaluation_round": 2 if round2 else 1,
                                    "note": "2025・2026 年を評価済み（または評価の途中で止まった）。結果を見て採否・規則を変えない"},
                                   ensure_ascii=False), encoding="utf-8")
    rt_ref = race_metrics(mk_pre, pay)
    for name, (Tv, mk, full) in versions.items():
        rt = race_metrics(mk, pay)
        strata = race_strata(Tv, rows) if full else None
        ref = rt_ref if (round2 and name != "前日版") else None
        res["versions"][name] = _eval_version(rt, span, gmap, strata, rt_base if full else {}, ref, "前日版", 2000 if full else 1000)
        if round2:
            res["ties"][name] = R.tie_share(mk[mk["race_id"].isin(set(rt["race_id"]))])
        if full or not round2:
            mk.to_parquet(out_dir / f"marks_{name}.parquet", index=False)
        # 版ごとに書き足す（途中で止まっても、それまでの版の結果を残す。終えた版は completed に並ぶ）
        res.setdefault("completed", []).append(name)
        (out_dir / "eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print("evaluated", name, flush=True)
    for bk, v in base.items():
        if len(v):
            v.to_parquet(out_dir / f"marks_{bk}.parquet", index=False)
    (out_dir / "eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    lab = f"{span[0]}-{span[1]}"
    if smoke:
        print(json.dumps({k: v.get(lab, {}).get("all") for k, v in list(res["versions"].items())[:6]}, ensure_ascii=False, indent=1)[:3000])
        return
    flag.write_text(json.dumps({"evaluated_at": res["evaluated_at"], "finished": True, "evaluation_round": 2 if round2 else 1,
                                "note": "2025・2026 年を評価済み。結果を見て採否・規則を変えない"}, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v.get(lab, {}).get("all") for k, v in list(res["versions"].items())[:6]}, ensure_ascii=False, indent=1)[:3000])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["frame", "tally", "select", "evaluate", "smoke", "export", "select_roi", "smoke2", "evaluate2"])
    a = ap.parse_args()
    if a.cmd in ("smoke", "smoke2", "evaluate2"):
        cmd_evaluate(smoke=a.cmd != "evaluate2", round2=a.cmd != "smoke")
        return
    if a.cmd == "select_roi":
        cmd_select_roi()
        return
    if a.cmd == "export":
        from src.research.graded.export import cmd_export
        cmd_export()
        return
    {"frame": cmd_frame, "tally": cmd_tally, "select": cmd_select, "evaluate": cmd_evaluate}[a.cmd]()


if __name__ == "__main__":
    main()
