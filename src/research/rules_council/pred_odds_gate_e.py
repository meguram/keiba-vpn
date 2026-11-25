"""R0-14 の門 E（着順・払戻を結ばない。1 号の決定「R0-14 の R3」、予測は config/rules_council/r0_14_predictions.json）。

2022〜24 の、§77 の想定オッズの年ごとの walk-forward の OOF（wp_oof_pred.parquet の win_B）で判定した買い目と、
T-60 の再現（5 通り）で判定した買い目を、確定オッズと v5bag の勝率だけで比べる。
- q̂ = 1/win_B をレース内で 1 にそろえた値。q_final = 確定オッズから作った値（同じくそろえる）。
- 組んだ勝率 p は凍結の候補の a・b（combo_p）。判定に使う q を q̂ に替えて、凍結と同じ決まり（p × 判定のオッズ ≥ t・判定のオッズ ≤ 50）で選ぶ。
- 模型の上の値: p_f（同じ a・b で q を q_final に替えた組んだ勝率）を真とみなした、選んだ集合の
  「模型の上の市場比」= Σ p_f ÷ Σ q_final と、「模型の上の回収率」= Σ p_f × 確定オッズ ÷ 点数。
- ε = log q_final − log q̂、d̂ = log p − log q̂、d_f = log p − log q_final。
出す: P-a（選んだ集合の E[ε]、t 1.1・1.3・1.5）、P-b（模型の上の市場比: q̂ で選んだ集合 ÷ 再現で選んだ集合）、P-c（族ごと）、
P-e（d̂ の 10 分位ごとの corr(d̂, d_f)・E[ε]・sd(ε)）。打ち切りの線は r0_14_predictions.json の stop_lines のとおり（この道具は値だけを出す）。
着順・払戻の列は、load_cand が表を読んだ後に許可リストで落とす（stake の計算は p_rich と判定のオッズだけを使う）。費用 0 円（手元の表だけ。GCS は読まない）。土日の 07:30〜17:00 は止める。
使い方: python -m src.research.rules_council.pred_odds_gate_e（nice -n 19）
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.rules_council import engine as E

ROOT = Path(__file__).resolve().parents[3]
WP = ROOT / "data/local/modeling/odds_pred/wp_oof_pred.parquet"
OUT = ROOT / "docs/operations/retrain_trials/rules_council_r0_14_gate_e.json"
CANDS = ("c13_flat", "c21_maiden_t60", "c23_flat_t15")
YEARS = (2022, 2023, 2024)
ALLOW = {"race_id", "horse_id", "year", "kaisai", "cls", "odds_final", "q_final", "p_rich", "stake",
         "odds_pre0", "odds_pre1", "odds_pre2", "odds_pre3", "odds_pre4", "win_B"}
FORBID = re.compile(r"^fin$|^p_fin$|^y1$|^y3$|^ret_|^pay|top3|place", re.I)
N_BOOT = 2000
SEED = 20261126


def _norm(inv: np.ndarray, rid: np.ndarray) -> np.ndarray:
    return inv / pd.Series(inv).groupby(rid).transform("sum").to_numpy()


def frame(name: str) -> tuple[pd.DataFrame, dict, tuple]:
    from src.research.rules_council.benter_diag import load_cand
    d, sp, ab = load_cand(name)
    d = d[d["year"].astype(int).isin(YEARS)].copy()
    d["race_id"], d["horse_id"] = d["race_id"].astype(str), d["horse_id"].astype(str)
    d["kaisai"] = d["race_id"].str[:8]
    d = d[[c for c in d.columns if c in ALLOW]]                                    # 着順・払戻の列をここで落とす
    w = pd.read_parquet(WP, columns=["race_id", "horse_id", "year", "win_B"])
    w = w[w["year"].astype(int).isin(YEARS)].astype({"race_id": str, "horse_id": str})[["race_id", "horse_id", "win_B"]]
    d = d.merge(w, on=["race_id", "horse_id"], how="left")
    ok = d.groupby("race_id")["win_B"].transform(lambda s: bool(s.notna().all() and (s > 0).all()))
    d = d[ok.to_numpy()].sort_values(["race_id"]).reset_index(drop=True)
    bad = [c for c in d.columns if FORBID.search(c)]
    assert not bad, f"着順・払戻の系の列が残っている: {bad}"
    return d, sp, ab


def stakes(d: pd.DataFrame, sp: dict, ab: tuple, odds_cols: list[str]) -> np.ndarray:
    """odds_cols の各列を判定のオッズにして、凍結と同じ決まりで選んだ買う割合（列の平均）。"""
    from src.research.megu_cycles import cycles as C
    st = C.strat_ev(lambda x, k: C.combo_p(x, k, *ab, col="p_rich"), sp["t"], sp["omax"])
    out = []
    for c in odds_cols:
        e = d.copy()
        e["odds_pre0"] = e[c].to_numpy(float)
        out.append(st(e, 0))
    return np.vstack(out).mean(0)


def p_combo(d: pd.DataFrame, ab: tuple, q: np.ndarray) -> np.ndarray:
    rid = d["race_id"].to_numpy()
    z = ab[0] * np.log(np.clip(d["p_rich"].to_numpy(float), 1e-6, 1)) + ab[1] * np.log(np.clip(q, 1e-6, 1))
    z = z - pd.Series(z).groupby(rid).transform("max").to_numpy()
    return _norm(np.exp(z), rid)


def boot_mean(x: np.ndarray, w: np.ndarray, ks: np.ndarray) -> list[float]:
    uk, inv = np.unique(ks, return_inverse=True)
    num = np.bincount(inv, w * x, len(uk))
    den = np.bincount(inv, w, len(uk))
    rng = np.random.default_rng(SEED)
    bs = []
    for i in range(N_BOOT):
        if i % 500 == 0:
            E.check_stop()
        j = rng.integers(len(uk), size=len(uk))
        bs.append(num[j].sum() / max(den[j].sum(), 1e-12))
    return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def boot_ratio(num: np.ndarray, den: np.ndarray, ks: np.ndarray) -> list[float]:
    uk, inv = np.unique(ks, return_inverse=True)
    a, b = np.bincount(inv, num, len(uk)), np.bincount(inv, den, len(uk))
    rng = np.random.default_rng(SEED + 1)
    bs = []
    for _ in range(N_BOOT):
        j = rng.integers(len(uk), size=len(uk))
        bs.append(a[j].sum() / max(b[j].sum(), 1e-12))
    return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def summarize(d: pd.DataFrame, w: np.ndarray, pf: np.ndarray, eps: np.ndarray, eps_own: np.ndarray | None = None) -> dict:
    m = w > 0
    S = float(w[m].sum())
    if S <= 0:
        return {"bets": 0.0}
    qf, of = d["q_final"].to_numpy(float), d["odds_final"].to_numpy(float)
    pr = d["p_rich"].to_numpy(float)
    ks = d["kaisai"].to_numpy()[m]
    yr = d["year"].astype(int).to_numpy()[m]
    out = {"bets": S, "races": int(d.loc[m, "race_id"].nunique()),
           "model_market_ratio_pf": float((w * pf)[m].sum() / (w * qf)[m].sum()),
           "model_market_ratio_pf_ci95": boot_ratio((w * pf)[m], (w * qf)[m], ks),
           "market_ratio_frozen_def": float((w * pr)[m].sum() / (w * qf)[m].sum()),
           "market_ratio_frozen_def_ci95": boot_ratio((w * pr)[m], (w * qf)[m], ks),
           "model_roi_pf": float((w * pf * of)[m].sum() / S),
           "mean_eps_vs_qhat": float(np.average(eps[m], weights=w[m])), "mean_eps_vs_qhat_ci95": boot_mean(eps[m], w[m], ks),
           "by_year": {int(y): {"bets": float(w[m][yr == y].sum()), "mean_eps_vs_qhat": float(np.average(eps[m][yr == y], weights=w[m][yr == y]))}
                       for y in np.unique(yr) if w[m][yr == y].sum() > 0}}
    if eps_own is not None:
        out["mean_eps_vs_own_judge_q"] = float(np.average(eps_own[m], weights=w[m]))
    if S < 50:
        out["note"] = "点数が 50 に満たない（記述だけ）"
    return out


def run_one(name: str) -> dict:
    E.check_stop()
    d, sp, ab = frame(name)
    rid = d["race_id"].to_numpy()
    qh = _norm(1.0 / d["win_B"].to_numpy(float), rid)
    qf = d["q_final"].to_numpy(float)
    d["odds_hat"] = d["win_B"].to_numpy(float)
    eps = np.log(np.clip(qf, 1e-9, 1)) - np.log(np.clip(qh, 1e-9, 1))
    pf = p_combo(d, ab, qf)
    ph = p_combo(d, ab, qh)
    r = {"cand": name, "a_b": list(ab), "t": sp["t"], "omax": sp["omax"], "rows": int(len(d)), "races": int(d["race_id"].nunique())}
    w_rep = d["stake"].to_numpy(float)                                              # 凍結の再現 5 通りの平均（load_cand のとおり）
    # 再現の集合を、自分の判定の q（再現の 5 通り）で測った ε（log q_final − log q_rep の 5 通りの平均）
    eps_rep = np.mean([np.log(np.clip(qf, 1e-9, 1)) - np.log(np.clip(_norm(1.0 / d[f"odds_pre{k}"].to_numpy(float), rid), 1e-9, 1))
                       for k in range(5) if f"odds_pre{k}" in d], axis=0)
    r["baseline_all_rows_mean_eps_vs_qhat"] = float(eps.mean())
    r["replica"] = summarize(d, w_rep, pf, eps, eps_own=eps_rep)
    r["by_t"] = {}
    for t in (1.1, 1.3, 1.5):
        spt = dict(sp, t=t)
        w_h = stakes(d, spt, ab, ["odds_hat"])
        r["by_t"][str(t)] = summarize(d, w_h, pf, eps)
    w_h = stakes(d, sp, ab, ["odds_hat"])
    r["qhat"] = summarize(d, w_h, pf, eps, eps_own=eps)
    if r["replica"].get("model_market_ratio_pf") and r["qhat"].get("model_market_ratio_pf"):
        r["P_b_ratio_qhat_over_replica"] = r["qhat"]["model_market_ratio_pf"] / r["replica"]["model_market_ratio_pf"]
    r["replica_phase"] = sp.get("phase", "t60")
    r["overlap_kept_norm"] = float(np.minimum(w_h, w_rep).sum() / max(min(w_h.sum(), w_rep.sum()), 1e-9))
    # P-e: d̂ の 10 分位（その族の全行）
    del ph
    pr = d["p_rich"].to_numpy(float)                                                 # 同じ p（v5bag の組む前の勝率）で差を取る（凍結の定義）
    dh = np.log(np.clip(pr, 1e-9, 1)) - np.log(np.clip(qh, 1e-9, 1))
    df = np.log(np.clip(pr, 1e-9, 1)) - np.log(np.clip(qf, 1e-9, 1))
    r["spearman_dhat_df_all"] = float(pd.Series(dh).corr(pd.Series(df), method="spearman"))
    dec = pd.qcut(dh, 10, labels=False, duplicates="drop")
    r["deciles"] = [{"decile": int(k), "n": int((dec == k).sum()), "corr_dhat_df": float(np.corrcoef(dh[dec == k], df[dec == k])[0, 1]),
                     "mean_eps": float(eps[dec == k].mean()), "sd_eps": float(eps[dec == k].std())} for k in sorted(np.unique(dec))]
    return r


def main() -> None:
    E.check_stop()
    out = {"note": "R0-14 の門 E。着順・払戻は結ばない（許可リストと assert）。2022〜24 の §77 の walk-forward OOF（win_B）と確定オッズと v5bag の勝率だけ。"
                   "模型の上の値は、同じ a・b で q を確定にした組んだ勝率を真とみなした値で、実際の成績ではない",
           "predictions": "config/rules_council/r0_14_predictions.json",
           "reading": ("ε = log q_final − log q̂。replica の mean_eps_vs_qhat は作りの上で負に寄る（再現のオッズは確定から作るので、再現で選ぶと q_final の低い馬を選ぶ）。"
                       "P-a の対照には、replica の mean_eps_vs_own_judge_q と、全行の基準 baseline_all_rows_mean_eps_vs_qhat を使う。"
                       "model_market_ratio_pf は q_final と組んだ p_f を真とみなす循環の値で、再現の側が構造的に有利。凍結の定義の市場比は market_ratio_frozen_def（Σp_rich ÷ Σq_final）。"
                       "c23 の replica は T-15 の再現（a・b も T-15 の値）。2022 年の win_B は較正なし（win_A と同じ）なので by_year で分ける。"
                       "分位の中の相関は値の範囲の制限で形が決まるので、P-e は spearman_dhat_df_all と並べて読む"),
           "cands": {}}
    for name in CANDS:
        out["cands"][name] = run_one(name)
        print(name, flush=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
