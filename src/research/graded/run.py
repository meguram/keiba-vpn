"""重賞のルールベース予想家の実行（研究。本番の予測・画面には載せない。手元だけで動き、GCS・GCP は使わない）。

  python -m src.research.graded.run frame      # 1 出走 = 1 行の表と水準（<GRADED_OUT>/frame.parquet・levels.parquet）
  python -m src.research.graded.run tally      # パターンの as-of の件数（tally.parquet）と、2024 年末で止めた版（tally_frozen.parquet）
  python -m src.research.graded.run select     # 縮める強さ・型の採否・束ね・点の足し方を 2016〜2024 の中だけで決め、prereg.json に凍結する
  python -m src.research.graded.run smoke      # evaluate と同じ処理を確認期間（2022〜2024）で流す試運転（2025・2026 年の行は読み込むが、評価にも出力にも使わない）
  python -m src.research.graded.run evaluate   # 2025・2026 年の重賞で 1 回だけ評価する（EVALUATED.json があれば止まる）
  python -m src.research.graded.run cycles / freeze_cycles / track_cycles  # 分析サイクル（ゴール 30%・120%、消し 95%）→ 規則の凍結 → 前向きの記録
  python -m src.research.graded.run select_buy / smoke_buy / evaluate_buy  # 買いの組み合わせ（凍結 → 試運転 → 4 回目の評価）
  python -m src.research.graded.run select_kill / smoke_kill / evaluate_kill  # 消しの版（凍結 → 試運転 → 3 回目の評価）
  python -m src.research.graded.run export_rules  # 各重賞で点に効いているルールの全件の表の材料（graded_rules.json）
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
PREREG_KILL = Path(os.environ.get("GRADED_PREREG_KILL") or Path(__file__).with_name("prereg_kill.json"))
PREREG_BUY = Path(os.environ.get("GRADED_PREREG_BUY") or Path(__file__).with_name("prereg_buy.json"))
ROUND2_NOTE = ("2 回目の評価（2025・2026 年の結果を一度見た後）。ユーザーの依頼（統括役経由、2026-11-14）: 各パターンに回収率を持たせ、"
               "回収率まで裏付けのあるパターンだけで印を付ける。基準はいろいろな版を並べる。")
DISC, CONF, EVAL = (2018, 2021), (2022, 2024), (2025, 2026)
ROUND3_NOTE = ("3 回目の評価（2025・2026 年の結果を二度見た後）。ユーザーの依頼（統括役経由、2026-11-15）: 好成績の点で ◎○ を選び、"
               "極端に成績の悪いルール（[0-0-0-n] など）に当てはまる馬を消して、相手を消去法で絞る。版はまとめて prereg_kill.json に凍結した。")
ROUND4_NOTE = ("4 回目の評価（2025・2026 年の結果を三度見た後）。ユーザーの依頼（統括役経由、2026-11-15）: 「買い」は 3 着内が 7〜8 割を占め、"
               "単勝か複勝の回収率が 100% を超える抽出条件（切り口の AND）に限る。過学習は許す。版はまとめて prereg_buy.json に凍結した。")
DEV_PAY = (2020, 2024)   # 払戻の表（payoff.parquet）は 2020 年から。開発期間の買い方の値はこの範囲だけ（版の選択には使わない）
Z_DISC, Z_CONF = 1.645, 1.2816          # 発見は片側 95%、確認は片側 90%
LAMS = (0.5, 1.0, 2.0)
PAYOFFS = [REPO / "data/local/modeling/betting/payoff.parquet", REPO / "data/local/modeling/betting_2026/payoff.parquet"]
# 比べる基準（p_win の順で ◎○▲）。v6e16（2016 年からの表）の 2026 年の予測は、2026 年の学習側の表が無く GCS も要るため作れなかった
# （2026-11-13 の調査）。2026 年は本番 v6e（2020 年からの表、bundle_v6_2025）を推論の経路で流した予測（重賞 95 レース）で比べる。
# 2026 年の本番 v6e の予測は第 1 回の置き場で作った。2 回目の評価（別の置き場）でも同じファイルを読む
# （2 回目の評価では GRADED_OUT の下を探して見つからず、2026 年の比較が抜けた。評価の後に eval_posthoc_v6e.json で足した）
V6E_2026_DIR = Path(os.environ.get("GRADED_V6E_2026_DIR", "/home/jovyan/work/graded_out/v6e_2026"))
BASELINES = {
    "v6e16": {2025: Path("/home/jovyan/work/retrain_out/base_2016/final_eval_v6e16/test2025_predictions.parquet")},
    "v6e本番": {2025: REPO / "data/local/modeling/training_pipeline/final_eval_v6e/test2025_predictions.parquet",
               2026: V6E_2026_DIR / "pred_2026_graded_prod_v6e_serving.parquet"},
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


# ------------------------------------------------------------------ 消しの版（3 回目の評価）
def _kill_inputs(span, rows: pd.DataFrame):
    """as-of の件数 → (前日版の点の S, 前日版の印に頭数を付けたもの)。"""
    pre = json.loads(PREREG.read_text(encoding="utf-8"))
    T = _yr(_ctx(GRADED_OUT / "tally.parquet", rows), span)
    S, mk = scored(T, pre, "preday")
    mk = mk.merge(rows[["row", "field"]], on="row", how="left")
    return pre, S, mk


def cmd_select_kill() -> None:
    """消しの版の一覧と主に見る版を 2016〜2024 年の中だけで決め（払戻を使わない消しの精度で選ぶ）、PREREG_KILL に凍結する。"""
    from src.research.graded import kill as K
    if (GRADED_OUT / "EVALUATED_KILL.json").exists():
        sys.exit("消しの版は評価が済んでいる。評価の後に版を作り直さない")
    if PREREG_KILL.exists() and os.environ.get("GRADED_RESELECT") != "1":
        sys.exit(f"{PREREG_KILL} は凍結済み。作り直すなら GRADED_RESELECT=1（評価の前に限る）")
    pre_roi = json.loads(PREREG_ROI.read_text(encoding="utf-8"))
    prim_roi = next(v for v in pre_roi["versions"] if v["primary"])
    d, _ = _load()
    rows = _rows(d)
    pre, S, mk = _kill_inputs((DISC[0], CONF[1]), rows)
    S_kill = S[S["type"].isin(pre["sets"]["preday"]["used"])]
    S_roi = S[S["type"].isin(prim_roi["used"])]
    pay = load_payoff()
    dev = mk[(mk["year"] >= DEV_PAY[0]) & (mk["year"] <= DEV_PAY[1])]
    vers, cache = [], {}
    for p in K.all_versions():
        kk = K.kill_key(p)
        if kk not in cache:
            kr = K.killed_rows(S_kill, S_roi, p, prim_roi)
            hits = 1 if p.get("only_roi") else p["hits"]
            killed = kr[kr >= hits]
            cache[kk] = (killed, K.precision_by_period(killed, mk, {"disc": DISC, "conf": CONF}))
        killed, prec = cache[kk]
        rt = K.race_metrics_kill(K.order_marks(dev, set(killed.index), p["honmei"]), pay)
        sb = K.summarize_bets(rt, n_boot=200)
        vers.append({"name": K.version_name(p), "params": p, "precision": prec, "passes": K.passes(prec),
                     "dev_bets": {"years": list(DEV_PAY), "races": sb["races"], **{k: v["est"] if isinstance(v, dict) else v for k, v in sb.items() if k != "races"}},
                     "dev_kill": K.kill_stat(rt)})
        print(vers[-1]["name"], vers[-1]["passes"], {k: (v["killed"], round(v["top3_rate"] or 0, 3), round(v["e3_rate"] or 0, 3)) for k, v in prec.items()}, flush=True)
    prim = K.choose_primary(vers)
    for v in vers:
        v["primary"] = v["name"] == prim
    out = {
        "frozen_at": datetime.now(JST).isoformat(timespec="seconds"),
        "evaluation_round": 3,
        "note": ROUND3_NOTE + " 版の一覧・主に見る版は 2016〜2024 年の中だけで、払戻を使わない消しの精度で決めた。評価の後に版を足したり選び直したりしない。",
        "base_prereg": {"path": "src/research/graded/prereg.json", "frozen_at": pre["frozen_at"], "uses": "前日版の点・印（m・λ・型の重み・使う型）"},
        "roi_prereg": {"path": "src/research/graded/prereg_roi.json", "frozen_at": pre_roi["frozen_at"], "uses": "主に見る回収率版の避けるパターン（軸 回収率の避ける）"},
        "periods": {"discovery": DISC, "confirm": CONF, "evaluate": EVAL, "dev_bets": DEV_PAY},
        "grid": {"crit": K.CRITS, "nmin": K.NMINS, "layer": list(K.LAYERS), "hits": K.HITS, "roi7": K.ROI7, "honmei": K.HONMEI},
        "constants": K.constants(), "n_versions": len(vers), "primary": prim,
        "rules": {
            "kill_types": "前日版の点に使う型（束ねの代表）。軸 回収率の避ける を入れる版は、主に見る回収率版の 7 パターン（型 × 水準 × 凡）も足す",
            "stat": "馬 × 型の as-of の件数で、点 x<0、かつ版の層のどれかで 走数 ≥ n の下限・異なるレースの数 ≥ 下限（主は 3 回。2016 年からの集計では主の過去の回が 2024 年でも 8 回までなので。副・全重賞は max(3, n の下限/2)）、かつ基準を満たす",
            "crit": {"3着内0": "その層の 3 着内の数が 0", "上限が期待の半分以下": f"3 着内率の Jeffreys 上側 {K.UPPER_Q:.0%} ≤ {K.UPPER_RATIO} × 期待（Σ3/頭数 ÷ 走数）"},
            "layer": {"主だけ": "主（同じレース名 × 同じコース。走数 15 未満でも使う）", "主＋副": "主か副（副は全体が 300 走未満なら 3 勝クラス以上）", "全部": "主・副・全重賞"},
            "roi": "回収率の避けるパターンは、主・副の層で x<0 のとき 1 つの当たりとして数える（export.roi_flag と同じ条件）",
            "hits": "消しに当たる型の数（重複なし）が下限以上の馬を消し",
            "marks": "前日版の点の順（点 → 主の好成績パターンの数 → 馬番）。そのまま: ◎○ は点の上位 2 頭、残りは消しでない馬 → 消した馬（補い）の順。"
                     "消しは◎○にしない: 消しでない馬 → 消した馬（補い）の順に並べて ◎○ から付ける",
            "bets": "◎ の単勝・複勝（各 100 円）、◎→○▲ の馬連・ワイド（2 点）、◎→消しでない馬すべて（◎ 以外。0 点のレースは購入額 0）、"
                    "◎→相手 3・4・5 頭（並びの 2 番目から。足りなければ消した馬で補った並びのまま）。購入額 = 点数 × 100 円",
            "primary": f"払戻を使わない: 発見 {DISC}・確認 {CONF} の両方で 消した馬の 3 着内率 ≤ 期待 × {K.PRECISION_RATIO}、各期間で消した馬 {K.MIN_KILLED} 頭以上。"
                       f"そのうち 2 期間を合わせた 1 レースあたりの消しの数が最大の版（◎○ は {K.PRIMARY_HONMEI}）。同数は n の下限が大きい → 層が狭い → 3着内0 → 当たる数の下限が大きい → 回収率の避けるを入れない の順",
            "bootstrap": "レース単位のポアソン(1) の重み（2000 回、主に見る版以外は 1000 回）。1・2 回目の評価（run.summarize・diff_ci の復元抽出）とは方式が違うので、前日版の区間は過去の eval.json と少し違う",
            "dnf": "完走しなかった馬（着順が無い）は 3 着外として数え、期待（Σ3/頭数）にも入る。重賞では 1 年に数頭",
            "dev_bets": f"開発期間の買い方の値は {DEV_PAY} の重賞だけ（payoff.parquet が 2020 年から）。版の選択には使わない参考",
        },
        "versions": vers,
    }
    PREREG_KILL.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=_f), encoding="utf-8")
    print("primary", prim, "passes", sum(v["passes"] for v in vers), "/", len(vers))
    print("prereg_kill written", PREREG_KILL)


def cmd_evaluate_kill(smoke: bool = False) -> None:
    """消しの版の評価（3 回目の評価、2025・2026 年で 1 回だけ）。smoke=True は確認期間（2022〜2024、版の選択に使った期間）で配管を確かめるだけ。"""
    from src.research.graded import kill as K
    flag = GRADED_OUT / "EVALUATED_KILL.json"
    span = CONF if smoke else EVAL
    out_dir = GRADED_OUT / ("smoke_kill" if smoke else ".")
    if flag.exists() and not smoke:
        sys.exit(f"消しの版は評価済み（{flag}）。やり直さない")
    out_dir.mkdir(parents=True, exist_ok=True)
    pk = json.loads(PREREG_KILL.read_text(encoding="utf-8"))
    if pk.get("constants") != json.loads(json.dumps(K.constants())):
        sys.exit("kill.py の定数が凍結（prereg_kill.json の constants）と違う。凍結した規則のまま評価する")
    if [K.version_name(p) for p in K.all_versions()] != [v["name"] for v in pk["versions"]]:
        sys.exit("kill.py の版の一覧が凍結と違う")
    pre_roi = json.loads(PREREG_ROI.read_text(encoding="utf-8"))
    prim_roi = next(v for v in pre_roi["versions"] if v["primary"])
    d, _ = _load()
    rows = _rows(d)
    pre, S, mk = _kill_inputs(span, rows)
    S_kill = S[S["type"].isin(pre["sets"]["preday"]["used"])]
    S_roi = S[S["type"].isin(prim_roi["used"])]
    pay = load_payoff()
    res = {"evaluated_at": datetime.now(JST).isoformat(timespec="seconds"), "evaluation_round": 3, "note": ROUND3_NOTE, "smoke": smoke,
           "prereg_kill_frozen_at": pk["frozen_at"], "primary": pk["primary"], "n_versions": pk["n_versions"], "span": span, "versions": {}}
    if not smoke:
        flag.write_text(json.dumps({"evaluated_at": res["evaluated_at"], "evaluation_round": 3,
                                    "note": "消しの版を 2025・2026 年で評価済み（または途中で止まった）。結果を見て版を変えない"}, ensure_ascii=False), encoding="utf-8")
    base = mk.assign(kill=False)
    rt_base = K.race_metrics_kill(base, pay)
    periods = [(str(y), (y, y)) for y in range(span[0], span[1] + 1)] + [(f"{span[0]}-{span[1]}", span)]

    def per_period(rt, full):
        yr = rt["race_id"].str[:4].astype(int)
        e = {}
        for lab, (y0, y1) in periods:
            r = rt[(yr >= y0) & (yr <= y1)]
            if len(r) == 0:
                continue
            e[lab] = {"all": K.summarize_bets(r, n_boot=2000 if full else 1000), "kill": K.kill_stat(r),
                      "diff_vs_前日版": K.diff_bets(r, rt_base[rt_base["race_id"].isin(r["race_id"])], n_boot=2000 if full else 1000)}
        return e
    res["versions"]["前日版（消しなし）"] = per_period(rt_base, True)
    cache = {}
    for v in pk["versions"]:
        p = v["params"]
        kk = K.kill_key(p)
        if kk not in cache:
            kr = K.killed_rows(S_kill, S_roi, p, prim_roi)
            cache[kk] = kr[kr >= (1 if p.get("only_roi") else p["hits"])]
        mkv = K.order_marks(mk, set(cache[kk].index), p["honmei"])
        rt = K.race_metrics_kill(mkv, pay)
        res["versions"][v["name"]] = per_period(rt, v["primary"])
        if v["primary"]:
            mkv.to_parquet(out_dir / "marks_kill_primary.parquet", index=False)
        res.setdefault("completed", []).append(v["name"])
        (out_dir / "eval_kill.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_f), encoding="utf-8")
    if not smoke:
        flag.write_text(json.dumps({"evaluated_at": res["evaluated_at"], "finished": True, "evaluation_round": 3,
                                    "note": "消しの版を 2025・2026 年で評価済み。結果を見て版を変えない"}, ensure_ascii=False), encoding="utf-8")
    lab = f"{span[0]}-{span[1]}"
    for name in ["前日版（消しなし）", pk["primary"]]:
        e = res["versions"][name][lab]
        print(name, {k: round(e["all"][k]["est"], 3) for k in ("win", "place")}, {k: round(e["all"][k]["est"], 3) for k in e["all"] if k.startswith("roi_") and e["all"][k]["est"] is not None}, e["kill"])


# ------------------------------------------------------------------ 買いの組み合わせ（4 回目の評価）
def _buy_inputs():
    from src.research.graded import buy as BY
    d, L = _load()
    d = with_returns(d)
    rows = _rows(d)
    types = [c for c in L.columns if c not in ("M01", "M02")]
    X_all, names = BY.encode(L, types)
    pos = pd.Series(np.arange(len(L)), index=L.index)
    b = TL.base_frame(d)
    b["year"] = pd.to_datetime(b["dt"]).dt.year
    b["horse_id"] = d.loc[b["row"], "horse_id"].astype(str).to_numpy()
    b = b[pd.to_numeric(d.loc[b["row"], "fin"], errors="coerce").notna().to_numpy()]   # 着順の無い馬（取消・除外・中止）は外す
    b = b.reset_index(drop=True)
    Xb = X_all[pos.reindex(b["row"]).to_numpy()]
    g = d[d["grade"].isin(TL.GRADES)].drop_duplicates("race_id")
    rows = rows[rows["race_id"].isin(set(g["race_id"]))].reset_index(drop=True)
    rows["fin"] = pd.to_numeric(rows["fin"], errors="coerce")
    rows = rows[rows["fin"].notna()].reset_index(drop=True)
    Xr = X_all[pos.reindex(rows["row"]).to_numpy()]
    return d, rows, types, names, b, Xb, Xr, g


def _buy_input_hashes(names: dict) -> dict:
    """凍結の照合に使う入力の sha256（frame・levels と、水準の番号 → 名前の辞書）。"""
    import hashlib
    h = {k: hashlib.sha256((GRADED_OUT / k).read_bytes()).hexdigest() for k in ("frame.parquet", "levels.parquet")}
    h["names"] = hashlib.sha256(json.dumps(names, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return h


def _courses(g: pd.DataFrame, year_end: int) -> dict:
    x = g[pd.to_datetime(g["dt"]).dt.year <= year_end].sort_values("dt").groupby("rkey").tail(1)
    return dict(zip(x["rkey"].astype(str), x["course"].astype(str)))


def _buy_search(b, Xb, g, year_end: int, seeds) -> pd.DataFrame:
    from src.research.graded import buy as BY
    m = (b["year"] <= year_end).to_numpy()
    bb, XX = b[m].reset_index(drop=True), Xb[m]
    cr = _courses(g, year_end)
    parts = []
    for s in [None, *seeds]:
        src = bb if s is None else BY.permute_within_race(bb, s)
        c = BY.search_period(src, XX, cr)
        c["perm"] = -1 if s is None else s
        parts.append(c)
        print("searched", year_end, "perm" if s is not None else "real", s, len(c), flush=True)
    return pd.concat(parts, ignore_index=True)


def _buy_eval_versions(cells: pd.DataFrame, rows: pd.DataFrame, Xr: np.ndarray, ref: np.ndarray, versions: list[dict]) -> dict:
    """版ごとに、本物の組と偶然の組（並べ替えのそれぞれ）で買いの馬を決め、指標を出す。"""
    from src.research.graded import buy as BY
    H = BY.hits_for(cells, rows, Xr)
    out = {}
    okm = {BY.version_name(v): BY.cells_ok(cells, v) for v in versions}
    for v in versions:
        name = BY.version_name(v)
        ok = okm[name]
        e = {}
        for pm in sorted(cells["perm"].unique()):
            sel = ok & (cells["perm"].to_numpy() == pm)
            h = H[sel[H["cell"].to_numpy()]] if len(H) else H
            cnt = np.bincount(h["i"].to_numpy(), minlength=len(rows)) if len(h) else np.zeros(len(rows), int)
            met = BY.buy_metrics(rows, cnt >= v["hits"], ref)
            met["cells"] = int(sel.sum())
            e["real" if pm == -1 else f"perm{pm}"] = met
        out[name] = e
    return out


def cmd_select_buy() -> None:
    """買いの組を 2016〜2024 年の中だけで決めて凍結する。開発の検証: 2016〜2021 年で探して 2022〜2024 年の重賞に当てはめ、
    本物の組と偶然の組（並べ替え）で比べ、主に見る版を選ぶ（選び方は buy.SELECT_MIN_BUY の説明）。そのあと 2016〜2024 年で探し直した組を凍結する。"""
    import hashlib

    from src.research.graded import buy as BY
    if (GRADED_OUT / "EVALUATED_BUY.json").exists():
        sys.exit("買いの版は評価が済んでいる。評価の後に作り直さない")
    if PREREG_BUY.exists() and os.environ.get("GRADED_RESELECT") != "1":
        sys.exit(f"{PREREG_BUY} は凍結済み。作り直すなら GRADED_RESELECT=1（評価の前に限る）")
    pre = json.loads(PREREG.read_text(encoding="utf-8"))
    d, rows, types, names, b, Xb, Xr, g = _buy_inputs()
    seeds = list(range(1, BY.N_PERM + 1))
    versions = BY.all_versions()
    # 開発の検証（2016〜2021 → 2022〜2024）
    cd = _buy_search(b, Xb, g, DISC[1], seeds)
    cr = _courses(g, DISC[1])
    sel = ((rows["year"] >= CONF[0]) & (rows["year"] <= CONF[1])).to_numpy()
    rc = rows[sel].reset_index(drop=True)
    keep = (rc["course"].astype(str).to_numpy() == rc["rkey"].astype(str).map(cr).to_numpy())
    T = _yr(_ctx(GRADED_OUT / "tally.parquet", rows), CONF)
    _, mk = scored(T, pre, "preday")
    ref = rc["row"].map(mk.set_index("row")["score"]).fillna(-np.inf).to_numpy()
    dev = _buy_eval_versions(cd, rc[keep].reset_index(drop=True), Xr[sel][keep], ref[keep], versions)
    dev_info = {"search_years": [2016, DISC[1]], "test_years": list(CONF), "test_races": int(rc[keep]["race_id"].nunique()),
                "test_races_other_course": int(rc[~keep]["race_id"].nunique())}
    prim = BY.choose_primary(versions, dev)
    # 2016〜2024 年で探し直して凍結
    cf = _buy_search(b, Xb, g, CONF[1], seeds)
    cf["types"] = cf["ts"].map(lambda ts: "|".join(types[t] for t in ts))
    cf["levels"] = [" | ".join(names[types[t]][v] for t, v in zip(ts, cs)) for ts, cs in zip(cf["ts"], cf["cs"])]
    path = GRADED_OUT / "buy_cells.parquet"
    w = cf.assign(ts=cf["ts"].map(list), cs=cf["cs"].map(list))
    w.to_parquet(path, index=False)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    counts = {}
    for v in versions:
        ok = BY.cells_ok(cf, v)
        counts[BY.version_name(v)] = {("real" if p == -1 else f"perm{p}"): int((ok & (cf["perm"].to_numpy() == p)).sum()) for p in sorted(cf["perm"].unique())}
    out = {
        "frozen_at": datetime.now(JST).isoformat(timespec="seconds"), "evaluation_round": 4,
        "note": ROUND4_NOTE + " 組と版・主に見る版は 2016〜2024 年の中だけで決めた（主に見る版は 2016〜2021 年で探して 2022〜2024 年に当てはめた開発の検証で選んだ）。",
        "constants": BY.constants(), "types": types, "n_type_combos_per_race": {"主": BY.n_combos(len(types), 3), "副": BY.n_combos(len(types), 2)},
        "rules": BY.RULES, "versions": [{"name": BY.version_name(v), "params": v, "primary": BY.version_name(v) == prim,
                                          "dev": dev[BY.version_name(v)], "cells_2016_2024": counts[BY.version_name(v)]} for v in versions],
        "n_versions": len(versions), "primary": prim, "dev": dev_info,
        "inputs": _buy_input_hashes(names),
        "cells": {"path": str(path), "sha256": sha, "rows": int(len(cf)), "real": int((cf["perm"] == -1).sum()),
                  "note": "組の一覧は大きいのでリポジトリに入れず、この sha256 で凍結を確かめる（評価の前に一致を確かめる）"},
    }
    PREREG_BUY.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=_f), encoding="utf-8")
    print("primary", prim, "cells", len(cf), "sha", sha[:12])


def cmd_evaluate_buy(smoke: bool = False) -> None:
    """買いの版の評価（4 回目の評価、2025・2026 年で 1 回だけ）。smoke=True は確認期間（2022〜2024。組を探した期間の中なので、
    成績は必ず良く出る。配管の確認だけ）。"""
    import hashlib

    from src.research.graded import buy as BY
    flag = GRADED_OUT / "EVALUATED_BUY.json"
    span = CONF if smoke else EVAL
    out_dir = GRADED_OUT / ("smoke_buy" if smoke else ".")
    if flag.exists() and not smoke:
        sys.exit(f"買いの版は評価済み（{flag}）。やり直さない")
    out_dir.mkdir(parents=True, exist_ok=True)
    pb = json.loads(PREREG_BUY.read_text(encoding="utf-8"))
    if pb.get("constants") != json.loads(json.dumps(BY.constants())):
        sys.exit("buy.py の定数が凍結と違う")
    if [BY.version_name(v) for v in BY.all_versions()] != [v["name"] for v in pb["versions"]]:
        sys.exit("buy.py の版の一覧が凍結と違う")
    path = Path(pb["cells"]["path"])
    if path.parent.resolve() != GRADED_OUT.resolve():
        sys.exit(f"GRADED_OUT（{GRADED_OUT}）が凍結の置き場（{path.parent}）と違う")
    if hashlib.sha256(path.read_bytes()).hexdigest() != pb["cells"]["sha256"]:
        sys.exit(f"{path} が凍結（sha256）と違う")
    pre = json.loads(PREREG.read_text(encoding="utf-8"))
    d, rows, types, names, b, Xb, Xr, g = _buy_inputs()
    if types != pb["types"]:
        sys.exit("型の並びが凍結と違う")
    if _buy_input_hashes(names) != pb["inputs"]:
        sys.exit("frame・levels・水準の辞書が凍結と違う")
    cells = pd.read_parquet(path)
    cells["ts"], cells["cs"] = cells["ts"].map(tuple), cells["cs"].map(tuple)
    lv = [" | ".join(names[types[t]][v] for t, v in zip(ts, cs)) for ts, cs in zip(cells["ts"], cells["cs"])]
    if lv != cells["levels"].tolist():
        sys.exit("組の水準の番号と名前が凍結と食い違う")
    cr = _courses(g, CONF[1])
    sel = ((rows["year"] >= span[0]) & (rows["year"] <= span[1])).to_numpy()
    rc = rows[sel].reset_index(drop=True)
    keep = rc["course"].astype(str).to_numpy() == rc["rkey"].astype(str).map(cr).to_numpy()
    res = {"evaluated_at": datetime.now(JST).isoformat(timespec="seconds"), "evaluation_round": 4, "note": ROUND4_NOTE, "smoke": smoke,
           "prereg_buy_frozen_at": pb["frozen_at"], "primary": pb["primary"], "n_versions": pb["n_versions"], "span": span,
           "races": int(rc[keep]["race_id"].nunique()), "races_other_course": int(rc[~keep]["race_id"].nunique()),
           "races_other_course_list": sorted(set(rc.loc[~keep, "rkey"].astype(str)))}
    if not smoke:
        flag.write_text(json.dumps({"evaluated_at": res["evaluated_at"], "evaluation_round": 4,
                                    "note": "買いの版を 2025・2026 年で評価済み（または途中で止まった）。結果を見て版を変えない"}, ensure_ascii=False), encoding="utf-8")
    T = _yr(_ctx(GRADED_OUT / "tally.parquet", rows), span)
    _, mk = scored(T, pre, "preday")
    ref = rc["row"].map(mk.set_index("row")["score"]).fillna(-np.inf).to_numpy()
    rk, Xk, refk = rc[keep].reset_index(drop=True), Xr[sel][keep], ref[keep]
    res["versions"] = _buy_eval_versions(cells, rk, Xk, refk, BY.all_versions())
    # 主に見る版の買いの馬の一覧（HTML 用）とレース単位の区間
    pv = next(v for v in BY.all_versions() if BY.version_name(v) == pb["primary"])
    okc = BY.cells_ok(cells, pv) & (cells["perm"].to_numpy() == -1)
    H = BY.hits_for(cells[okc].reset_index(drop=True), rk, Xk)
    cnt = np.bincount(H["i"].to_numpy(), minlength=len(rk)) if len(H) else np.zeros(len(rk), int)
    buy = cnt >= pv["hits"]
    res["primary_ci"] = BY.race_ci(rk, buy, refk)
    cols = ["race_id", "rkey", "horse_id", "hn", "fin", "ret_tan", "ret_fuku", "pay_ok"]
    rk.assign(n_cells=cnt, buy=buy, ref_score=refk)[cols + ["n_cells", "buy", "ref_score"]].to_parquet(out_dir / "buy_primary_horses.parquet", index=False)
    (out_dir / "eval_buy.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_f), encoding="utf-8")
    if not smoke:
        flag.write_text(json.dumps({"evaluated_at": res["evaluated_at"], "finished": True, "evaluation_round": 4,
                                    "note": "買いの版を 2025・2026 年で評価済み。結果を見て版を変えない"}, ensure_ascii=False), encoding="utf-8")
    e = res["versions"][pb["primary"]]
    print(json.dumps({k: e[k]["buy"] | {"share": e[k]["buy_share"], "hon": e[k]["honmei_buy"]} for k in e}, ensure_ascii=False, default=_f)[:3000])


# ------------------------------------------------------------------ 分析サイクル（ゴール: ◎ 勝率 30%・単回収 120%、サブゴール: 消しの 95%）
CYCLE_NOTE = ("ユーザーの依頼（統括役経由、2026-11-15）: 2025・2026 年の重賞で ◎ の勝率 30% 以上・単勝の回収率 120% 以上になる組み合わせを、"
              "すべての重賞で見つける。サブゴール: 消した馬の 95% 以上が 4 着以下（1 レース 3 頭以上を消す）。最大 20 サイクル。"
              "A（当てはめ版）は 2025・2026 年に直接合わせた値で、評価ではない。B（正直な版）は 2018〜2024 年で選んで 2025・2026 年に当てはめた値。")


def _cycle_table(year_max: int | None = None, levels: bool = False):
    """levels=True なら (表, {(row, 型): (水準, 層)}) を返す（早見表の例の馬の水準を書くため）。"""
    from src.research.graded import cycles as CY
    pre = json.loads(PREREG.read_text(encoding="utf-8"))
    d, _ = _load()
    d = with_returns(d)
    rows = _rows(d)
    rows["fin"] = pd.to_numeric(rows["fin"], errors="coerce")
    T = _ctx(GRADED_OUT / "tally.parquet", rows)
    T = T[(T["year"] >= CY.DEV[0]) & (T["year"] <= (year_max or CY.EVAL[1]))]
    S_pre, mk_pre = scored(T, pre, "preday")
    _, mk_day = scored(T, pre, "dayof")
    X = S_pre[~S_pre["type"].isin(["M01", "M02"])].pivot_table(index="row", columns="type", values="x", aggfunc="first").fillna(0.0)   # めぐ指数は比較だけの型なので使わない
    X.columns = [f"x_{c}" for c in X.columns]
    r = rows[rows["row"].isin(X.index) & rows["fin"].notna() & rows["grade"].isin(TL.GRADES)].copy()
    r = r.merge(mk_pre[["row", "score", "rank"]].rename(columns={"score": "score_pre", "rank": "rank_pre"}), on="row", how="left")
    r = r.merge(mk_day[["row", "score"]].rename(columns={"score": "score_day"}), on="row", how="left")
    r[["score_pre", "score_day"]] = r[["score_pre", "score_day"]].fillna(0.0)
    r["top3_pre"] = (r["rank_pre"] <= 3).astype(float)
    r["top5_pre"] = (r["rank_pre"] <= 5).astype(float)
    r["y3"] = (r["fin"] <= 3).astype(float)
    r["y1"] = (r["fin"] == 1).astype(float)
    r["pay_ok"] = r["pay_ok"].fillna(False).astype(bool)
    # 消しの版（prereg_kill.json）で基準を通った版の、馬ごとの消しの当たりの数（as-of の件数。消しの決め方の特徴に使う）
    from src.research.graded import kill as K
    pk = json.loads(PREREG_KILL.read_text(encoding="utf-8"))
    prim_roi = next(v for v in json.loads(PREREG_ROI.read_text(encoding="utf-8"))["versions"] if v["primary"])
    S_kill, S_roi = S_pre[S_pre["type"].isin(pre["sets"]["preday"]["used"])], S_pre[S_pre["type"].isin(prim_roi["used"])]
    kcols = []
    for j, v in enumerate(x for x in pk["versions"] if x["passes"] and x["params"].get("honmei") == "そのまま"):
        kr = K.killed_rows(S_kill, S_roi, v["params"], prim_roi)
        r[f"khit_{j + 1}"] = r["row"].map(kr).fillna(0.0).to_numpy(float)
        kcols.append(f"khit_{j + 1}")
    names = list(CY.BASE) + kcols + list(X.columns)
    F = np.column_stack([r[list(CY.BASE) + kcols].to_numpy(float), X.reindex(r["row"]).to_numpy(float)])
    tab = CY.Table(r.reset_index(drop=True), F, names)
    if not levels:
        return tab
    L = S_pre[S_pre["row"].isin(r.loc[r["year"] >= CY.EVAL[0], "row"])]   # 例に書くのは 2025・2026 年の馬だけ
    return tab, {(int(a), b): (str(c), str(e)) for a, b, c, e in zip(L["row"], L["type"], L["level"], L["layer"])}


def cmd_cycles() -> None:
    """分析サイクルを最大 20 回まわし、サイクルごとの値を <GRADED_OUT>/cycles.json に書き足す。A の重賞ごと・全体で 1 つの両方で
    ゴールとサブゴールを満たした時点で止める（cycles.goal_met）。"""
    import time

    from src.research.graded import cycles as CY
    t = _cycle_table()
    B = CY.Best(t)
    rng = np.random.default_rng(20261115)
    terms: dict[str, dict] = {}
    path = GRADED_OUT / "cycles.json"
    log = {"note": CYCLE_NOTE, "started_at": datetime.now(JST).isoformat(timespec="seconds"), "goal": {"win": CY.WIN_GOAL, "roi_t": CY.ROI_GOAL},
           "sub_goal": {"precision": CY.KILL_GOAL, "killed_per_race": CY.KILL_MIN}, "periods": {"dev": CY.DEV, "eval": CY.EVAL},
           "n_features": len(t.names), "rows": int(len(t.r)), "races": int(len(t.starts)), "cycles": []}
    for c in range(1, 21):
        t0 = time.time()
        hyp, cand = CY.space(c, t, rng)
        W, lab = CY.matrix(t, cand)
        terms.update({n: d for d, n in cand})
        B.update(W, lab)
        rep = B.report()
        met = CY.goal_met(rep)
        log["cycles"].append({"cycle": c, "at": datetime.now(JST).isoformat(timespec="seconds"), "hypothesis": hyp, "n_new": len(cand),
                              "seconds": round(time.time() - t0, 1), "met": met, "report": rep})
        path.write_text(json.dumps(log, ensure_ascii=False, indent=1, default=_f), encoding="utf-8")
        a, b = rep["A"], rep["B"]
        print(f"cycle {c} {hyp[:30]} tried={rep['tried']} A達成 {a['honmei_by_rkey']['achieved']}/{rep['n_rkeys']} "
              f"A全体 勝率 {a['honmei_by_rkey']['all']['win']:.3f} 単回収 {a['honmei_by_rkey']['all']['roi_t']:.2f} "
              f"A一つ {a['honmei_global']['val']['win']:.3f}/{a['honmei_global']['val']['roi_t']:.2f} "
              f"B重賞ごと {b['honmei_by_rkey']['all']['win']:.3f}/{b['honmei_by_rkey']['all']['roi_t']:.2f} "
              f"B一つ {b['honmei_global']['val']['win']:.3f}/{b['honmei_global']['val']['roi_t']:.2f} "
              f"消しA {a['kill_by_rkey']['all']['precision']:.3f}({a['kill_by_rkey']['all']['killed_per_race']:.1f}) "
              f"消しB {b['kill_by_rkey']['all']['precision']:.3f} 消しB一つ {b['kill_global']['val']['precision']:.3f} {met}", flush=True)
        if met["all"]:
            log["stopped"] = {"cycle": c, "reason": "A の重賞ごと・全体で 1 つの両方で、ゴールとサブゴールを満たした"}
            break
    else:
        log["stopped"] = {"cycle": 20, "reason": "20 サイクルに達した"}
    log["finished_at"] = datetime.now(JST).isoformat(timespec="seconds")
    path.write_text(json.dumps(log, ensure_ascii=False, indent=1, default=_f), encoding="utf-8")
    # C: 最後のサイクルの A（と比べるための B）の規則を、重みごと凍結の材料にする（commit は別の手順）
    rep = log["cycles"][-1]["report"]
    rules = {}
    for m in ("A", "B"):
        rules[m] = {"honmei_by_rkey": {x["rkey"]: CY.rule_of(x["rule"], terms) for x in rep[m]["honmei_by_rkey"]["per_rkey"] if x["rule"]},
                    "kill_by_rkey": {x["rkey"]: CY.rule_of(x["rule"], terms) for x in rep[m]["kill_by_rkey"]["per_rkey"] if x["rule"]},
                    "honmei_global": CY.rule_of(rep[m]["honmei_global"]["sel"], terms), "kill_global": CY.rule_of(rep[m]["kill_global"]["sel"], terms)}
    (GRADED_OUT / "cycles_rules.json").write_text(json.dumps({"from_cycles_finished_at": log["finished_at"], "stopped": log["stopped"],
                                                               "features": t.names, "rules": rules}, ensure_ascii=False, indent=1), encoding="utf-8")


def _final_odds() -> pd.DataFrame:
    """2025・2026 年の (race_id, horse_id) → 馬名・確定の単勝オッズ・人気・頭数（race_result_flat）。早見表で結果を見せるためだけに使い、決め方には使わない。"""
    from src.research.graded import cycles as CY
    from src.research.graded.features import TABLES
    parts = []
    for y in range(CY.EVAL[0], CY.EVAL[1] + 1):
        p = TABLES / str(y) / "race_result_flat.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p, columns=["race_id", "horse_id", "horse_name", "odds", "popularity", "field_size"]))
    if not parts:
        sys.exit(f"{CY.EVAL[0]}〜{CY.EVAL[1]} 年の race_result_flat.parquet が無い")
    m = pd.concat(parts)
    m["race_id"], m["horse_id"] = m["race_id"].astype(str), m["horse_id"].astype(str)
    m["odds_final"] = pd.to_numeric(m["odds"], errors="coerce")
    m["pop_final"] = pd.to_numeric(m["popularity"], errors="coerce")
    m["field_size"] = pd.to_numeric(m["field_size"], errors="coerce")
    m = m[["race_id", "horse_id", "horse_name", "odds_final", "pop_final", "field_size"]].drop_duplicates()   # 全く同じ行だけ落とす（値の違う重複は結合の validate で止める）
    return m


def cmd_playbook() -> None:
    """重賞ごとの早見表の値: 最後のサイクルの規則（cycles_rules.json）を当て直し、2025・2026 年と 2018〜2024 年の成績を
    <GRADED_OUT>/playbook.json に書く。2025・2026 年の値がサイクルの記録と違えば止める（評価はやり直さない）。"""
    from src.research.graded import playbook as PB
    src = json.loads((GRADED_OUT / "cycles_rules.json").read_text(encoding="utf-8"))
    log = json.loads((GRADED_OUT / "cycles.json").read_text(encoding="utf-8"))
    if src["from_cycles_finished_at"] != log["finished_at"]:
        sys.exit("cycles_rules.json と cycles.json のサイクルが違う")
    t, lv = _cycle_table(levels=True)
    if t.names != src["features"]:
        sys.exit("特徴の列が cycles_rules.json と違う")
    m = t.r[["race_id", "horse_id"]].astype(str).merge(_final_odds(), on=["race_id", "horse_id"], how="left", validate="many_to_one")
    for c in ("horse_name", "odds_final", "pop_final", "field_size"):   # 結果を見せるためだけ（行の並び・特徴は変えない）
        t.r[c] = m[c].to_numpy()
    ev = ((t.r["year"] >= PB.CY.EVAL[0]) & (t.r["year"] <= PB.CY.EVAL[1])).to_numpy()
    miss = int(t.r.loc[ev, ["horse_name", "odds_final", "pop_final"]].isna().any(axis=1).sum())
    if miss:   # 取り消し等で数頭欠けるのは表で「—」にする。キーの型ずれなどで大きく欠けたら止める
        print("確定オッズ・人気・馬名の無い馬:", miss, "頭", flush=True)
        if miss > 0.01 * ev.sum():
            sys.exit(f"確定オッズ・人気・馬名の無い馬が多すぎる（{miss} / {int(ev.sum())} 頭）。race_id・horse_id の型を確かめる")
    pk = json.loads(PREREG_KILL.read_text(encoding="utf-8"))
    kn = [v["name"].removeprefix("消し版") for v in pk["versions"] if v["passes"] and v["params"].get("honmei") == "そのまま"]
    res = PB.build_cards(t, src["rules"], log["cycles"][-1]["report"], kn, lv)
    nw = [(c["rkey"], x["date"]) for c in res["cards"] for x in c["races"] if not x["winners"]]
    if nw:   # 勝ち馬が表（型の点のある馬）に無い回
        print("勝ち馬が表に無い回:", nw, flush=True)
    res |= {"made_at": datetime.now(JST).isoformat(timespec="seconds"), "from_cycles_finished_at": log["finished_at"], "tried": log["cycles"][-1]["report"]["tried"],
            "periods": {"dev": list(PB.CY.DEV), "eval": list(PB.CY.EVAL)}, "criteria": {"roi_t_over": PB.ROI_OK, "kill_top3_at_most": PB.KILL_OK}}
    (GRADED_OUT / "playbook.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_f), encoding="utf-8")
    print(json.dumps(res["summary"], ensure_ascii=False))


PREREG_CYCLES = Path(os.environ.get("GRADED_PREREG_CYCLES") or Path(__file__).with_name("prereg_cycles.json"))
CYCLES_LEDGER = Path(os.environ.get("GRADED_CYCLES_LEDGER") or (REPO / "data/meta/graded/cycles_forward_ledger.json"))


def _cycles_hash(body: dict) -> str:
    """凍結の検査に使うハッシュ（規則・記録を始める日・特徴の列を覆う）。"""
    import hashlib
    x = {k: body[k] for k in ("rules", "track_from", "features")}
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def cmd_freeze_cycles() -> None:
    """C: 分析サイクルの最後の規則（A を主、B を比べる相手）を prereg_cycles.json に凍結する。前向きの記録は track_from 以降の重賞だけ。"""
    if PREREG_CYCLES.exists() and os.environ.get("GRADED_RESELECT") != "1":
        sys.exit(f"{PREREG_CYCLES} は凍結済み")
    if CYCLES_LEDGER.exists() and json.loads(CYCLES_LEDGER.read_text(encoding="utf-8")).get("races"):
        sys.exit(f"台帳 {CYCLES_LEDGER} に記録がある。凍結し直さない")
    src = json.loads((GRADED_OUT / "cycles_rules.json").read_text(encoding="utf-8"))
    body = {"frozen_at": datetime.now(JST).isoformat(timespec="seconds"), "track_from": "2026-11-16",
            "note": "分析サイクルの最後の A（2025・2026 年に当てはめた規則）を、前向きに記録するために凍結した。track_from 以降の重賞だけに当てはめ、"
                    "規則は書き換えない（ハッシュで検査）。B は比べる相手。前向きの値が A の当てはめの値（評価ではない）にどれだけ近いかを見る",
            "features": src["features"], "rules": src["rules"], "from": {"cycles_finished_at": src["from_cycles_finished_at"], "stopped": src["stopped"]}}
    body["sha256_rules"] = _cycles_hash(body)
    PREREG_CYCLES.write_text(json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8")
    print("frozen", PREREG_CYCLES)


def cmd_track_cycles() -> None:
    """C: 凍結した規則を track_from 以降の重賞（集計 tally に入ったもの）に当てはめ、台帳に書き足す（同じレースは書き直さない）。
    新しいレースを入れるには、先に frame・tally を作り直す（python -m src.research.graded.run frame / tally）。"""

    from src.research.graded import cycles as CY
    P = json.loads(PREREG_CYCLES.read_text(encoding="utf-8"))
    if _cycles_hash(P) != P["sha256_rules"]:
        sys.exit("prereg_cycles.json の規則が凍結のハッシュと違う。追記を拒否する")
    if CYCLES_LEDGER.exists() and json.loads(CYCLES_LEDGER.read_text(encoding="utf-8")).get("prereg_frozen_at") != P["frozen_at"]:
        sys.exit("台帳の凍結の日時が prereg_cycles.json と違う（別の規則の記録が混ざる）。追記を拒否する")
    t = _cycle_table(year_max=9999)   # 前向きの記録は 2027 年以降も入れる
    if t.names != P["features"]:
        sys.exit("特徴の列が凍結と違う")
    led = json.loads(CYCLES_LEDGER.read_text(encoding="utf-8")) if CYCLES_LEDGER.exists() else {"prereg_frozen_at": P["frozen_at"], "races": {}}
    dt = t.r.groupby(t.race_of)["dt"].first().astype(str).str[:10].to_numpy() if "dt" in t.r.columns else None
    if dt is None:
        sys.exit("表に日付の列が無い")
    new = (dt >= P["track_from"]) & ~t.race["race_id"].isin(list(led["races"])).to_numpy()
    for i in np.flatnonzero(new):
        rid, rk = t.race["race_id"].iloc[i], t.race["rkey"].iloc[i]
        sel = np.zeros(len(t.starts), bool)
        sel[i] = True
        e = {"rkey": rk, "grade": t.race["grade"].iloc[i], "date": dt[i], "recorded_at": datetime.now(JST).isoformat(timespec="seconds")}
        for m in ("A", "B"):
            R = P["rules"][m]
            for kind in ("honmei", "kill"):
                r_rk = R[f"{kind}_by_rkey"].get(rk)
                e[f"{m}_{kind}_by_rkey"] = CY.apply_rule(t, r_rk, sel)["races"][0] if r_rk else None
                e[f"{m}_{kind}_global"] = CY.apply_rule(t, R[f"{kind}_global"], sel)["races"][0]
        led["races"][rid] = e
    led["updated_at"] = datetime.now(JST).isoformat(timespec="seconds")
    CYCLES_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    CYCLES_LEDGER.write_text(json.dumps(led, ensure_ascii=False, indent=1), encoding="utf-8")
    print("tracked", int(new.sum()), "new races, total", len(led["races"]), "->", CYCLES_LEDGER)


# ------------------------------------------------------------------ うまうまー式の穴馬の分析
PREREG_LONGSHOT = Path(os.environ.get("GRADED_PREREG_LONGSHOT") or Path(__file__).with_name("prereg_longshot.json"))
LONGSHOT_NOTE = ("ユーザーの依頼（統括役経由、2026-11-15）: うまうまー式に、特定の重賞の 6 番人気以下の穴馬を分母にして、前走○○組かつ上がり○位以内のような"
                 "組み合わせで集計する。主の層は、同じレース名 × 2016〜2024 年で一番多く行われたコース（代わりの会場の年は外す）。"
                 "人気は分母を選ぶためだけに使い、条件には使わない。2016〜2024 年で決めて凍結し、2025・2026 年で 1 回だけ確かめる"
                 "（この分析では初回。ただし 2025・2026 年の重賞は、ほかの分析で 4 度見ている）。")


def _popularity() -> pd.Series:
    """(race_id, horse_id) → 単勝人気（data/page_reference/tables/<年>/race_result_flat.parquet。分母を選ぶためだけに使う）。"""
    from src.research.graded.features import TABLES, YEARS
    parts = []
    for y in YEARS:
        p = TABLES / str(y) / "race_result_flat.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p, columns=["race_id", "horse_id", "popularity"]))
    m = pd.concat(parts)
    m["race_id"], m["horse_id"] = m["race_id"].astype(str), m["horse_id"].astype(str)
    return m.drop_duplicates(["race_id", "horse_id"]).set_index(["race_id", "horse_id"])["popularity"]


def _modal_courses(g: pd.DataFrame, year_end: int) -> dict:
    """重賞ごとに、year_end までで一番多く行われたコース（同数なら新しい方）。代わりの会場で開いた年（2024 年の宝塚記念の京都など）を外す
    うまうまー式の作法に合わせる（最新の回のコースだと、代わりの会場の年が主の層になってしまう）。"""
    x = g[pd.to_datetime(g["dt"]).dt.year <= year_end]
    a = x.groupby(["rkey", "course"]).agg(n=("race_id", "nunique"), last=("dt", "max")).reset_index()
    a = a.sort_values(["rkey", "n", "last"], ascending=[True, False, False]).groupby("rkey").head(1)
    return dict(zip(a["rkey"].astype(str), a["course"].astype(str)))


def _longshot_inputs():
    from src.research.graded import buy as BY
    d, L = _load()
    d = with_returns(d)
    rows = _rows(d)
    g = d[d["grade"].isin(TL.GRADES)].drop_duplicates("race_id")
    rows = rows[rows["race_id"].isin(set(g["race_id"]))].copy()
    rows["fin"] = pd.to_numeric(rows["fin"], errors="coerce")
    rows = rows[rows["fin"].notna()]
    pop = _popularity()
    rows["pop"] = pd.Series(list(zip(rows["race_id"].astype(str), rows["horse_id"].astype(str)))).map(pop).to_numpy()
    rows = rows.reset_index(drop=True)
    types = [c for c in L.columns if c not in ("M01", "M02")]
    X_all, names = BY.encode(L, types)
    pos = pd.Series(np.arange(len(L)), index=L.index)
    X = X_all[pos.reindex(rows["row"]).to_numpy()]
    rows["top3"] = (rows["fin"] <= 3).astype(float)
    for k in (1, 2, 3):
        rows[f"w{k}"] = (rows["fin"] == k).astype(float)
    rows["pay_ok"] = rows["pay_ok"].fillna(False).astype(bool)
    rows["np"] = rows["pay_ok"].astype(float)
    rows["rt"] = np.where(rows["pay_ok"], rows["ret_tan"].fillna(0), 0.0)
    rows["rf"] = np.where(rows["pay_ok"], rows["ret_fuku"].fillna(0), 0.0)
    cr = _modal_courses(g, CONF[1])
    rows["main"] = rows["course"].astype(str).to_numpy() == rows["rkey"].astype(str).map(cr).to_numpy()
    raw = ["wt", "age", "interval", "p_fin", "p_tdiff", "p_grade", "p_vc", "p_pos4", "p_l3rank"]
    rows = rows.join(d.loc[rows["row"], raw].reset_index(drop=True))
    return rows, X, types, names


def _ls_search(rows: pd.DataFrame, X: np.ndarray, sel: np.ndarray, by_rkey: bool):
    """6 番人気以下の分母で、重賞ごと（by_rkey）または全重賞をまとめて、切り口 1〜2 個の組の件数を数える。"""
    from src.research.graded import longshot as LS
    out = []
    groups = rows[sel].groupby("rkey").indices if by_rkey else {"全重賞": np.arange(sel.sum())}
    idx_all = np.flatnonzero(sel)
    for rk, ii in groups.items():
        ii = idx_all[ii]
        Y = np.column_stack([np.ones(len(ii)), rows["top3"].to_numpy()[ii]] + [rows[c].to_numpy()[ii] for c in ("w1", "w2", "w3", "np", "rt", "rf")])
        C = LS.cells(X[ii], Y, rows["race_id"].to_numpy()[ii])
        C = LS.score(C, len(ii), int(Y[:, 1].sum()))
        C["rkey"] = rk
        C["races"] = int(rows["race_id"].to_numpy()[ii].size and len(set(rows["race_id"].to_numpy()[ii])))
        out.append(C)
    return pd.concat(out, ignore_index=True)


def cmd_select_longshot() -> None:
    """2016〜2024 年の 6 番人気以下の分母で条件を探し、採る条件・偶然の数・BH の印を prereg_longshot.json に凍結する。"""
    from src.research.graded import longshot as LS
    if (GRADED_OUT / "EVALUATED_LONGSHOT.json").exists():
        sys.exit("穴馬の分析は評価が済んでいる。作り直さない")
    if PREREG_LONGSHOT.exists() and os.environ.get("GRADED_RESELECT") != "1":
        sys.exit(f"{PREREG_LONGSHOT} は凍結済み")
    rows, X, types, names = _longshot_inputs()
    dev = ((rows["year"] <= CONF[1]) & (rows["pop"] >= LS.POP_MIN) & rows["main"]).to_numpy()
    res = {}
    for by in (True, False):
        C = _ls_search(rows, X, dev, by)
        m = len(C)
        C["bh"] = LS.bh(C["pbin"].to_numpy(), m)
        perm = []
        for sd in range(1, LS.N_PERM + 1):
            rp = rows.copy()
            rp.loc[dev, ["w1", "w2", "w3", "top3", "np", "rt", "rf"]] = LS.permute(rows.loc[dev, ["w1", "w2", "w3", "top3", "np", "rt", "rf"]],
                                                                                  rows.loc[dev, "race_id"].to_numpy(), sd).to_numpy()
            Cp = _ls_search(rp, X, dev, by)
            Cp["bh"] = LS.bh(Cp["pbin"].to_numpy(), len(Cp))
            perm.append({"pass": int(Cp["pass"].sum()), "pass_bh": int((Cp["pass"] & Cp["bh"]).sum()), "bh": int(Cp["bh"].sum()),
                         "pass_by_rkey": Cp[Cp["pass"]].groupby("rkey").size().to_dict()})
        res[by] = (C, m, perm)
        print("searched", "by_rkey" if by else "pooled", m, "pass", int(C["pass"].sum()), "perm", [p["pass"] for p in perm], flush=True)
    C, m, perm = res[True]
    keep = []
    for rk, Cg in C[C["pass"]].groupby("rkey"):
        ii = np.flatnonzero(dev & (rows["rkey"] == rk).to_numpy())
        keep.append(LS.dedupe(Cg, X[ii], rows["row"].to_numpy()[ii]))
    P = pd.concat(keep) if keep else C.iloc[:0]
    def pack(D):
        return [{"rkey": r.rkey, "types": [types[t] for t in r.ts], "ts": list(map(int, r.ts)), "cs": list(map(int, r.cs)),
                 "levels": [names[types[t]][c] for t, c in zip(r.ts, r.cs)], "n": int(r.n), "k": int(r.k), "rec": [int(r.w1), int(r.w2), int(r.w3), int(r.n - r.w1 - r.w2 - r.w3)],
                 "N": int(r.N), "K": int(r.K), "recall": float(r.recall), "coverage": float(r.coverage), "rate": float(r.rate), "lift": float(r.lift),
                 "roi_t": float(r.rt / r.np) if r.np else None, "roi_f": float(r.rf / r.np) if r.np else None, "edk": int(r.edk), "pbin": float(r.pbin), "bh": bool(r.bh)}
                for r in D.itertuples()]
    Cg, mg, permg = res[False]
    pooled = Cg[(Cg["k"] >= LS.POOLED_K_MIN) & Cg["bh"]].sort_values("pbin")
    devi = np.flatnonzero(dev)
    pooled = LS.dedupe(pooled.head(LS.POOLED_TOP * 3), X[devi], rows["row"].to_numpy()[devi]).sort_values("pbin").head(LS.POOLED_TOP)
    base = {rk: {"N": int(g["N"].iloc[0]), "K": int(g["K"].iloc[0])} for rk, g in C.groupby("rkey")}
    out = {"frozen_at": datetime.now(JST).isoformat(timespec="seconds"), "note": LONGSHOT_NOTE,
           "constants": {"POP_MIN": LS.POP_MIN, "K_MIN": LS.K_MIN, "ED_MIN": LS.ED_MIN, "RECALL_MIN": LS.RECALL_MIN, "LIFT_MIN": LS.LIFT_MIN, "FDR": LS.FDR,
                         "N_PERM": LS.N_PERM, "max_depth": 2, "years": [2016, CONF[1]], "POOLED_K_MIN": LS.POOLED_K_MIN,
                         "POOLED_TOP": LS.POOLED_TOP, "EVAL_END": LS.EVAL_END},
           "inputs": _buy_input_hashes(names),
           "note_by_rkey": "重賞ごとの採る条件の数は、結果を入れ替えた偶然のデータとほぼ同じで、BH で残る条件は 0。評価の by_rkey の値は偶然の条件の当てはめとして読む",
           "types": types, "n_cells_by_rkey": m, "n_cells_pooled": mg, "n_rkeys": len(base), "base": base,
           "pass_real": int(C["pass"].sum()), "pass_real_bh": int((C["pass"] & C["bh"]).sum()), "pass_dedup": int(len(P)), "perm": perm,
           "pooled": pack(pooled.assign(rkey="全重賞")), "pooled_perm": permg,
           "pooled_real": {"cells": int(mg), "pass": int(Cg["pass"].sum()), "bh": int(Cg["bh"].sum()), "bh_k_min": int((Cg["bh"] & (Cg["k"] >= LS.POOLED_K_MIN)).sum())},
           "pooled_base": {"N": int(Cg["N"].iloc[0]), "K": int(Cg["K"].iloc[0])},
           "examples_dev": LS.example_stats(rows, (2016, CONF[1]))}
    import hashlib
    cp = GRADED_OUT / "longshot_cells.json"
    cp.write_text(json.dumps(pack(P), ensure_ascii=False, separators=(",", ":"), default=_f), encoding="utf-8")
    out["cells_file"] = {"path": str(cp), "sha256": hashlib.sha256(cp.read_bytes()).hexdigest(), "n": int(len(P)),
                         "note": "重賞ごとの採る条件の一覧は大きいのでリポジトリに入れず、sha256 で凍結を確かめる"}
    PREREG_LONGSHOT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=_f), encoding="utf-8")
    print("prereg_longshot", PREREG_LONGSHOT, "cells", len(P), "perm", [p["pass"] for p in perm])


def cmd_evaluate_longshot() -> None:
    """穴馬の分析の評価（この分析では初回。2025・2026 年の重賞で 1 回だけ）。"""
    import hashlib

    from src.research.graded import cycles as CY
    from src.research.graded import longshot as LS
    flag = GRADED_OUT / "EVALUATED_LONGSHOT.json"
    if flag.exists():
        sys.exit(f"穴馬の分析は評価済み（{flag}）")
    P = json.loads(PREREG_LONGSHOT.read_text(encoding="utf-8"))
    cp = Path(P["cells_file"]["path"])
    if hashlib.sha256(cp.read_bytes()).hexdigest() != P["cells_file"]["sha256"]:
        sys.exit(f"{cp} が凍結と違う")
    cells = json.loads(cp.read_text(encoding="utf-8"))
    rows, X, types, names = _longshot_inputs()
    if types != P["types"]:
        sys.exit("型の並びが凍結と違う")
    if _buy_input_hashes(names) != P["inputs"]:
        sys.exit("frame・levels・水準の辞書が凍結と違う")
    for c in cells + P["pooled"]:
        if [names[types[t]][v] for t, v in zip(c["ts"], c["cs"])] != c["levels"]:
            sys.exit("条件の水準の番号と名前が凍結と食い違う")
    PC = json.loads(PREREG_CYCLES.read_text(encoding="utf-8"))
    if _cycles_hash(PC) != PC["sha256_rules"]:
        sys.exit("prereg_cycles.json の規則が凍結と違う")
    res = {"evaluated_at": datetime.now(JST).isoformat(timespec="seconds"), "note": LONGSHOT_NOTE, "prereg_frozen_at": P["frozen_at"]}
    flag.write_text(json.dumps({"evaluated_at": res["evaluated_at"], "note": "穴馬の分析を 2025・2026 年で評価済み（または途中で止まった）"}, ensure_ascii=False), encoding="utf-8")
    ev = ((rows["year"] >= EVAL[0]) & (pd.to_datetime(rows["dt"]) <= pd.Timestamp(P["constants"]["EVAL_END"])) & rows["main"]).to_numpy()
    res["eval_end"] = P["constants"]["EVAL_END"]
    den = ev & (rows["pop"] >= LS.POP_MIN).to_numpy()
    # 前日版の点（前日の想定で上位 5 頭を除く、の代わり）
    pre = json.loads(PREREG.read_text(encoding="utf-8"))
    T = _yr(_ctx(GRADED_OUT / "tally.parquet", _rows(_load()[0])), EVAL)   # 前日版の順位は cmd_evaluate と同じ全行の母集団で作る
    _, mk = scored(T, pre, "preday")
    rows["rank_pre"] = rows["row"].map(mk.set_index("row")["rank"]).to_numpy()
    den_pre = ev & (rows["rank_pre"] > 5).to_numpy()
    def summary(sel, mcount, ths=(1, 5, 20)):
        D = rows[sel]
        base = LS.rec(D)
        K = int(D["top3"].sum())
        out = {"races": int(D["race_id"].nunique()), "denominator": base, "K": K}
        for h in ths:
            m = mcount[sel] >= h
            hit = LS.rec(D[m])
            out[f"hit_{h}"] = hit | {"recall": int(D["top3"].to_numpy()[m].sum()) / K if K else None, "coverage": float(m.mean()) if len(m) else None,
                                     "lift": (hit["rate"] / base["rate"]) if base["rate"] and hit["rate"] is not None else None}
        return out
    # 重賞ごとの条件: 馬ごとに当たる条件の数
    cnt = np.zeros(len(rows), int)
    by = rows.groupby("rkey").indices
    for c in cells:
        ii = by.get(c["rkey"])
        if ii is None:
            continue
        ii = ii[ev[ii]]
        cnt[ii[LS.matches(X[ii], c["ts"], c["cs"])]] += 1
    res["by_rkey"] = {"pop6": summary(den, cnt), "preday_out_top5": summary(den_pre, cnt)}
    # 全重賞をまとめた条件（BH で残ったもの）: 条件ごとと、どれかに当たる馬
    pooled = [c for c in P["pooled"] if c["bh"]]
    pc = np.zeros(len(rows), int)
    per = []
    D = rows[den]
    base = LS.rec(D)
    for c in pooled:
        m = LS.matches(X, c["ts"], c["cs"])
        pc += m
        h = LS.rec(rows[den & m])
        per.append({"types": c["types"], "levels": c["levels"], "dev": {k: c[k] for k in ("rec", "recall", "lift", "roi_t", "roi_f")},
                    "eval": h | {"recall": int(rows["top3"].to_numpy()[den & m].sum()) / max(int(D["top3"].sum()), 1),
                                 "lift": (h["rate"] / base["rate"]) if base["rate"] and h["rate"] is not None else None}})
    res["pooled"] = {"cells": per, "pop6": summary(den, pc, (1, 3)), "preday_out_top5": summary(den_pre, pc, (1, 3))}
    res["examples_eval"] = LS.example_stats(rows, EVAL)
    # 20 サイクルの消しの規則（B・全体で 1 つ、前日版の点の下位 3 頭に近いもの）との重なり
    t = _cycle_table()
    if t.names != PC["features"]:
        sys.exit("分析サイクルの特徴の列が凍結と違う")
    kr = CY.apply_rule(t, PC["rules"]["B"]["kill_global"], (t.race["year"] >= EVAL[0]).to_numpy())["races"]
    killed = {(x["race_id"], h) for x in kr for h in x["killed"]}
    rows["killed"] = [(r, int(h)) in killed for r, h in zip(rows["race_id"].astype(str), rows["hn"])]
    kk = rows["killed"].to_numpy()
    res["kill_overlap"] = {"rule": PC["rules"]["B"]["kill_global"]["label"] + f"・下位 {PC['rules']['B']['kill_global']['k']} 頭",
                           "pop6_killed_share": float(kk[den].mean()), "pop6_killed": LS.rec(rows[den & kk]), "pop6_not_killed": LS.rec(rows[den & ~kk]),
                           "pooled_hit_killed_share": float(kk[den & (pc >= 1)].mean()) if (den & (pc >= 1)).any() else None,
                           "pooled_hit_killed": LS.rec(rows[den & (pc >= 1) & kk]), "pooled_hit_not_killed": LS.rec(rows[den & (pc >= 1) & ~kk]),
                           "rkey_hit5_killed": LS.rec(rows[den & (cnt >= 5) & kk]), "rkey_hit5_not_killed": LS.rec(rows[den & (cnt >= 5) & ~kk])}
    (GRADED_OUT / "eval_longshot.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_f), encoding="utf-8")
    flag.write_text(json.dumps({"evaluated_at": res["evaluated_at"], "finished": True, "note": "穴馬の分析を 2025・2026 年で評価済み"}, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("by_rkey",)}, ensure_ascii=False, default=_f)[:2500])
    print(json.dumps(res["pooled"]["pop6"], ensure_ascii=False, default=_f)[:1500])
    print(json.dumps(res["kill_overlap"], ensure_ascii=False, default=_f)[:1500])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["frame", "tally", "select", "evaluate", "smoke", "export", "select_roi", "smoke2", "evaluate2", "export_rules", "select_kill", "smoke_kill", "evaluate_kill", "select_buy", "smoke_buy", "evaluate_buy", "cycles", "freeze_cycles", "track_cycles", "select_longshot", "evaluate_longshot", "playbook"])
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
    if a.cmd == "select_longshot":
        cmd_select_longshot()
        return
    if a.cmd == "evaluate_longshot":
        cmd_evaluate_longshot()
        return
    if a.cmd == "playbook":
        cmd_playbook()
        return
    if a.cmd == "cycles":
        cmd_cycles()
        return
    if a.cmd == "freeze_cycles":
        cmd_freeze_cycles()
        return
    if a.cmd == "track_cycles":
        cmd_track_cycles()
        return
    if a.cmd == "select_buy":
        cmd_select_buy()
        return
    if a.cmd in ("smoke_buy", "evaluate_buy"):
        cmd_evaluate_buy(smoke=a.cmd == "smoke_buy")
        return
    if a.cmd == "select_kill":
        cmd_select_kill()
        return
    if a.cmd in ("smoke_kill", "evaluate_kill"):
        cmd_evaluate_kill(smoke=a.cmd == "smoke_kill")
        return
    if a.cmd == "export_rules":
        from src.research.graded.rules_table import cmd_export_rules
        cmd_export_rules()
        return
    {"frame": cmd_frame, "tally": cmd_tally, "select": cmd_select, "evaluate": cmd_evaluate}[a.cmd]()


if __name__ == "__main__":
    main()
